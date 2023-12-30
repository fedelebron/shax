{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE LambdaCase #-}

module Transpose where

import Control.Monad.State
import Control.Lens hiding (op)
import Data.List (sort, (\\))
import GHC.Stack
import Data.Maybe

import qualified Data.Map as M
import Control.Monad.State
import Text.PrettyPrint.HughesPJClass (prettyShow)



import qualified Environment as E
import Types
import Binding
import Error
import Shaxpr
import Definition
import Control.Monad.Except (throwError)
import Control.Arrow (second)


type TangentVarName = VarName
type CotangentVarName = VarName
data CotangentState = CotangentState {
  -- Bindings for the transposed program.
  _env :: E.Env Binding,
  -- Bindings for the linear program.
  _originalEnv :: E.Env Binding,
  -- A mapping from variable name in the tangent program, to cotangents in the
  -- transposed program. The semantics are that if (x, xs) is in the map, then
  -- the cotangent of x is the sum of all xs.
  _cotangentSummandsMap :: E.Env [CotangentVarName],
  -- A mapping from a variable name in the tangent program, to the cotangent
  -- variable that holds the sum of all cotangent summands.
  _cotangentMap :: E.Env CotangentVarName,
  -- A map between constants in the linear program and the corresponding constants
  -- in the dual program.
  _constantRenames :: E.Env CotangentVarName
}
makeLenses 'CotangentState

type CotangentComputation a = StateT CotangentState (Either Error) a

addCotangent :: TangentVarName -> CotangentVarName -> CotangentComputation ()
addCotangent v ct = cannotFail $ do
  cotangentSummandsMap %= E.alter (\case
                            Nothing -> Just [ct]
                            Just as -> Just (ct:as)) v
  return ()

addCotangentBinding :: ShaxprF CotangentVarName -> CotangentComputation VarName
addCotangentBinding e = cannotFail $ do
    v <- E.nextName <$> use env
    env %= E.insert v (Binding v e)
    return v

maybeGetCotangentSummands :: HasCallStack => VarName -> State CotangentState (Maybe [VarName])
maybeGetCotangentSummands v = do
    cotangents <- E.lookup v <$> use cotangentSummandsMap
    return $ case cotangents of
        Left _ -> Nothing
        Right ct -> Just ct

maybeRenameConstant :: VarName -> CotangentComputation VarName
maybeRenameConstant v = E.lookupWithDefault v v <$> use constantRenames

getOriginalVarType :: HasCallStack => VarName -> CotangentComputation TensorType
getOriginalVarType v = do
  lEnv <- use originalEnv
  vBind <- lift (E.lookup v lEnv)
  let ty = exprTy (bindExpr vBind)
  case ty of
    Nothing -> throwError (Error ("Failed to get original type of " ++ prettyShow v) callStack)
    Just t -> return t

transposeDef :: HasCallStack => Definition -> Either Error Definition
transposeDef def@(Definition name paramTys binds rets) = do
  -- For simplicity, we require that each parameter is read once and only once.
  -- This doesn't mean it can't be _used_ more than once, just that for each
  -- parameter index i, there's exactly one `x = param{i}` statement.
  let paramReads = M.fromList [(i, (v, t)) | Binding v (ParamShaxprF t i) <- binds]
      nParams = length paramTys
  assertTrue (M.keys paramReads == [0 .. nParams - 1]) $
    Error "Cannot transpose function with more or fewer than 1 uses per parameter." callStack
  
  let defEnv = E.fromDefinition def
  retBinds <- mapM (`E.lookup` defEnv) rets
  retTypes <- case mapM (exprTy . bindExpr) retBinds of
                Just xs -> Right xs
                Nothing -> Left (Error "Failed to get type of return values." callStack)

  -- For simplicity, we hoist all parameter reads and constants to the top of the
  -- cotangent program.
  let cotangentParamReads = [Binding (VarName i) (ParamShaxprF (Just t) i) | (t, i) <- zip retTypes [0..]]
      numRets = length retTypes
      cotangentParamCotangents = [(v, VarName i) | (Binding v _, i) <- zip retBinds [0 .. numRets - 1]]
      cotangentParamCotangentSummands = map (second return) cotangentParamCotangents
      linearProgramConstants = [b | b@(Binding _ (ConstantShaxprF _ _)) <- binds]
      dualProgramConstants = [Binding (VarName v) (bindExpr b) | (v, b) <- zip [numRets .. ] linearProgramConstants]
      linearToDualConstants = [(bindLabel b, bindLabel b') | (b, b') <- zip linearProgramConstants dualProgramConstants]

      initialEnv = foldr (\b@(Binding v _) -> E.insert v b) E.empty (cotangentParamReads ++ dualProgramConstants)
      initialCotangents = foldr (uncurry E.insert) E.empty cotangentParamCotangents
      initialCotangentSummands = foldr (uncurry E.insert) E.empty cotangentParamCotangentSummands
      constantRenames = foldr (uncurry E.insert) E.empty linearToDualConstants
      linearEnv = E.fromDefinition def
  
      initialState = CotangentState initialEnv linearEnv initialCotangentSummands initialCotangents constantRenames
  
  CotangentState transposedEnv _ _ ctMap _ <- execStateT (mapM_ transposeBinding (reverse binds)) initialState
  paramReadCotangents <- mapM ((`E.lookup` ctMap) . fst) (M.elems paramReads)
  paramCotangents <- mapM (`E.lookup` transposedEnv) paramReadCotangents
  
  return Definition {
    defName = "t" ++ name,
    defArgTys = retTypes,
    defBinds = E.toBindings transposedEnv,
    defRet = map bindLabel paramCotangents
  }

transposeBinding :: Binding -> CotangentComputation ()
transposeBinding (Binding v (ConstantShaxprF _ _)) = do
  -- Constants were hoisted to the top of the dual program already, and 
  -- they never propagate cotangents (they have no predecessors in the linear
  -- program), so there's nothing to do.
  return ()
transposeBinding (Binding v (ShaxprF mty op args)) = do
    -- Given v, get dL/dv = \ddot{v}.
    -- We're walking the program backwards, and we've just seen the
    -- creation of variable v. At this point we'll write down the
    -- cotangent of v as the sum of all its cotangent summands.
    ctVars <- cannotFail $ maybeGetCotangentSummands v
    cotangent <- case ctVars of
      -- The primal parameters (dual returns) are their own cotangents.
      -- Since we don't want to say "x = id x" for them, we skip them.
        Just cts
            | cts == [v] -> return v
            | otherwise -> do
              -- We write the sum of cotangents, and record the final
              -- variable (which holds the full sum of cotangents)
              -- in cotangentMap.
              ctv <- flushCotangents mty v cts
              cotangentMap %= E.insert v ctv
              return ctv
      -- This would be the case when a variable was used, but has no path
      -- to any return variable. This shouldn't happen with traced
      -- functions, but it can happen if Definitions are written manually.
        _ -> throwError (Error "Drops aren't implemented." callStack)
    -- Now we pull back dL/dV along f for each argument.
    appendCotangents cotangent mty op args

-- TODO: Use a binary tree of summands, instead of a sequential sum.
flushCotangents :: HasCallStack => Maybe TensorType -> VarName -> [VarName] -> CotangentComputation VarName
flushCotangents _ v [] = throwError $ Error ("Cannot happen! No cotangents found for " ++ show v) callStack
-- Maybe not needed?
flushCotangents _ _ [ct] = return ct -- addCotangentBinding (IdShaxprF mty ct)
flushCotangents mty _ (c:cs) = foldM combine c cs
  where
    combine = (addCotangentBinding .) . AddShaxprF mty    

appendCotangents :: HasCallStack => VarName -> Maybe TensorType -> Op -> [VarName] -> CotangentComputation ()
appendCotangents _ _ (Param _) [] = do
  return ()
appendCotangents dLdZ mty (BinaryPointwise Add) [x, y] = do
  -- If Z = X + Y, then dL/dX = dL/dZ, and dL/dY = dL/dZ.
  addCotangent x dLdZ
  addCotangent y dLdZ
appendCotangents dLdZ mty (BinaryPointwise Sub) [x, y] = do
  -- If Z = X - Y, then dL/dX = dL/dZ, and dL/dY = -dL/dZ.
  addCotangent x dLdZ
  ct <- addCotangentBinding  (NegateShaxprF mty dLdZ)
  addCotangent y ct
appendCotangents dLdZ mty (BinaryPointwise Mul) [x, y] = do
  -- By convention, x is a constant, and y is the actual tangent in the linear
  -- program. Thus if Z = X . Y, we have dL/dX = 0, and dL/dY = X . dL/dZ.
  -- Note x here is a linear program variable, and we want its value in the
  -- dual program. This is the same value, but the constant may have been
  -- renamed in the dual program, so we possibly rename it.
  x' <- maybeRenameConstant x
  ct <- addCotangentBinding (MulShaxprF mty x' dLdZ)
  addCotangent y ct
appendCotangents dLdZ mty (Reshape sh') [x] = do
  t@(TensorType bt sh) <- getOriginalVarType x
  ct <- addCotangentBinding (ReshapeShaxprF (Just t) sh dLdZ)
  addCotangent x ct
appendCotangents dLdZ mty (Transpose perm) [x] = do
  tx <- getOriginalVarType x
  -- If Z = transpose p X, then dL/dX = transpose p^{-1} dL/dZ.
  ct <- addCotangentBinding (TransposeShaxprF (Just tx) (inversePerm perm) dLdZ)
  addCotangent x ct
appendCotangents dLdZ mty (Slice sixs eixs) [x]
    | Just (TensorType bt shout) <- mty = do
  t@(TensorType _ shin) <- getOriginalVarType x
  let lo = sixs
      hi = zipWith (-) shin eixs
  ct <- addCotangentBinding (PadShaxprF (Just t) (zip lo hi) 0.0 dLdZ)
  addCotangent x ct
appendCotangents dLdZ _ (Broadcast ixs shout) [x] = do
  t@(TensorType bt shin) <- getOriginalVarType x
  let rank = length shout
      allOutputDimIxs = [0 .. rank - 1]
      broadcastedDimIxs = allOutputDimIxs \\ ixs
      mty = Just (TensorType bt shin)
  ct <- addCotangentBinding (ReduceSumShaxprF mty broadcastedDimIxs dLdZ)
  addCotangent x ct
appendCotangents dLdZ mty (DotGeneral (DotDimensionNumbers [2] [1] [0] [0])) [x, y] = do
  tx <- getOriginalVarType x
  ty <- getOriginalVarType y
  (bt, b, n, m, p) <- case (tx, ty) of
    (TensorType bt [b, n, m], TensorType _ [_, _, p]) -> return (bt, b, n, m, p)
    _ -> throwError (Error "Invalid dot argument types." callStack)
  -- Note that both x and y can be variables. We'll pull back the cotangent
  -- through both of them, which raises the question of what happens when one
  -- is a constant, since dL/dc = 0 for constants c. Fear not, dear reader,
  -- since we discharge cotangent sums when a variable is created (remember,
  -- we're walking the original program _backwards_!). Thus, when we see a
  -- constant binding in the original program, we'll simply not discharge the
  -- cotangent sums for it.
  x' <- maybeRenameConstant x
  y' <- maybeRenameConstant y
  -- A calculation best done over a glass of scotch yields:
  -- If Z = XY, then dL/dX = dL/dZ Y^T, dL/dY = X^T dL/dZ.
  yt <- addCotangentBinding (TransposeShaxprF (Just (TensorType bt [b, p, m])) [0, 2, 1] y')
  dLdzyt <- addCotangentBinding (DotGeneralShaxprF (Just tx) (DotDimensionNumbers [2] [1] [0] [0]) dLdZ yt)
  addCotangent x dLdzyt
  xt <- addCotangentBinding (TransposeShaxprF (Just (TensorType bt [b, m, n])) [0, 2, 1] x')
  xtdLdz <- addCotangentBinding (DotGeneralShaxprF (Just ty) (DotDimensionNumbers [2] [1] [0] [0]) xt dLdZ)
  addCotangent y xtdLdz
appendCotangents _ _ op _ = error $ "Unimplemented transpose of " ++ show op

inversePerm :: [Int] -> [Int]
inversePerm x = map snd $ sort $ zip x [0 .. length x - 1]