{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE LambdaCase #-}

module Transpose where

import Control.Monad.State
import Control.Lens hiding (op)
import Data.List (sort, (\\))
import GHC.Stack
import Data.Maybe

import qualified Data.Map as M
import Control.Monad (foldM)
import Text.PrettyPrint.HughesPJClass (pPrintPrec, PrettyLevel(..), render, prettyShow)



import qualified Environment as E
import Types
import Binding
import Error
import Shaxpr
import Definition
import Linearize
import Control.Monad.Except (throwError)
import Control.Arrow (first, second)
import Control.Monad (forM_)

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
  -- This is an alternative to implementing `dup` from the You Only Linearize Once paper.
  _cotangentSummandsMap :: E.Env [CotangentVarName],
  -- A mapping from a variable name in the tangent program, to the cotangent
  -- variable that holds the sum of all cotangent summands.
  _cotangentMap :: E.Env CotangentVarName,
  -- A map between constants in the linear program and the corresponding constants
  -- in the dual program.
  _constantRenames :: E.Env CotangentVarName,
  -- A map between environment reads in the linear program, and the corresponding
  -- environment read in the dual program.
  _environmentRenames :: E.Env VarName
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

maybeRename :: VarName -> CotangentComputation VarName
maybeRename v = do
  constRenames <- use constantRenames
  envRenames <- use environmentRenames
  case E.lookup v constRenames of
    Left _ -> case E.lookup v envRenames of
                Left _ -> return v
                Right w -> return w
    Right w -> return w

isEnvironmentRead :: VarName -> CotangentComputation Bool
isEnvironmentRead v = cannotFail (E.member v <$> use environmentRenames)

getOriginalVarType :: HasCallStack => VarName -> CotangentComputation TensorType
getOriginalVarType v = do
  lEnv <- use originalEnv
  vBind <- lift (E.lookup v lEnv)
  let ty = exprTy (bindExpr vBind)
  case ty of
    Nothing -> throwError (Error ("Failed to get original type of " ++ prettyShow v) callStack)
    Just t -> return t

transposeDef :: HasCallStack => LinearizedDefinition -> Either Error LinearizedDefinition
transposeDef linearizedDefinition = do
  let LinearizedDefinition nonLinearDef linearDef envSize = linearizedDefinition
  -- For simplicity, we require that each parameter is read once and only once, in the
  -- linear program. 
  -- Note in the linear program, the first parameters correspond to tangent vectors for the
  -- primal program's parameters, while the rest of the parameters correspond to environment
  -- variables passed in from the primal program.
  let numTangentParams = length (defArgTys linearDef) - envSize
      numTangentReturns = length (defRet linearDef)
      isTangentVectorParamIndex k = k < numTangentParams
  let paramReads = M.fromList [(i, (v, t)) | Binding v (ParamShaxprF t i) <- defBinds linearDef,
                                             isTangentVectorParamIndex i]
  assertTrue (M.keys paramReads == [0 .. numTangentParams - 1]) $
    Error "Cannot transpose linear function with more or fewer than 1 uses per tangent parameter." callStack

  -- The first parameters of the dual program will have the same type as the return values
  -- of the linear program.
  let defEnv = E.fromDefinition linearDef
  linearRetBinds <- mapM (`E.lookup` defEnv) (defRet linearDef)
  linearRetTypes <- case mapM (exprTy . bindExpr) linearRetBinds of
                         Just xs -> Right xs
                         Nothing -> Left (Error "Failed to get type of return values." callStack)
  let numCotangents = length linearRetTypes
  -- For simplicity, we hoist all cotangent parameter reads, environment parameter
  -- reads, and constants, to the top of the dual program.
  --
  -- We have one cotangent parameter read for each return value of the linear program.
  let cotangentParamReads = [Binding (VarName i) (ParamShaxprF (Just t) i) | (t, i) <- zip linearRetTypes [0..]]
  -- We have one environment param read for each variable in the environment. These
  -- will be parameter reads in the linear program, with parameter indices greater
  -- than the number of parameters of the primal program (nParams).
  let environmentReads = M.fromList [(v, (i - numTangentReturns, t)) | Binding v (ParamShaxprF t i) <- defBinds linearDef,
                                                                       i >= numTangentParams]
      nEnvReads = length environmentReads
      environmentParamReads = [Binding (VarName i) (ParamShaxprF t i) | (i, t) <- M.elems environmentReads] 
      environmentParamTypes = map (fromJust . exprTy . bindExpr) environmentParamReads
      environmentRenameMap = E.Env (fmap (VarName . fst) environmentReads)
  -- We will have one dual program constant for each linear program constant. Note these will
  -- usually be zero, unless we're checkpointing, in which case we're effectively putting part
  -- of the primal program in the dual program.
  let linearProgramConstants = [b | b@(Binding _ (ConstantShaxprF _ _)) <- defBinds linearDef]
      dualProgramConstants = [Binding (VarName v) (bindExpr b) | (v, b) <- zip [numCotangents + nEnvReads .. ] linearProgramConstants]
  -- We set the incoming cotangents of all parameters to be themselves. Note that the cotangent
  -- map domains are _linear_ program variables, while its codomain are _dual_ program variables.
  -- Thus we use `retBinds` as the keys, and [0 .. ] (the indices of the first dual program params)
  -- as the codomain.
  let cotangentParamCotangents = [(v, VarName i) | (v, i) <- zip (defRet linearDef) [0 ..],
                                                   i < numCotangents]
      cotangentParamCotangentSummands = map (second return) cotangentParamCotangents
  -- Constants are renamed in the dual program, so we use this map to keep track of the pairing.
  -- Once again the keys come from the linear program, and the values come from the dual program.
  let linearToDualConstants = [(bindLabel b, bindLabel b') | (b, b') <- zip linearProgramConstants dualProgramConstants]
      initialEnv = foldr (\b@(Binding v _) -> E.insert v b) E.empty (cotangentParamReads ++ environmentParamReads ++ dualProgramConstants)
      initialCotangents = foldr (uncurry E.insert) E.empty cotangentParamCotangents
      initialCotangentSummands = foldr (uncurry E.insert) E.empty cotangentParamCotangentSummands
      constRenames = foldr (uncurry E.insert) E.empty linearToDualConstants
      linearEnv = E.fromDefinition linearDef
  
      initialState = CotangentState initialEnv linearEnv initialCotangentSummands initialCotangents constRenames environmentRenameMap
  
  CotangentState transposedEnv _ _ ctMap _ _ <- execStateT (mapM_ transposeBinding (reverse (defBinds linearDef))) initialState
  paramReadCotangents <- mapM ((`E.lookup` ctMap) . fst) (M.elems paramReads)
  paramCotangents <- mapM (`E.lookup` transposedEnv) paramReadCotangents
  
  let dualDef = Definition {
    defName = defName linearDef ++ "t",
    defArgTys = linearRetTypes ++ environmentParamTypes,
    defBinds = E.toBindings transposedEnv,
    defRet = map bindLabel paramCotangents
  }

  return $ LinearizedDefinition {
    nonlinear = nonLinearDef,
    linear = dualDef,
    envSize = envSize
  }

transposeBinding :: Binding -> CotangentComputation ()
transposeBinding (Binding v (ConstantShaxprF _ _)) = do
  -- Constants were hoisted to the top of the dual program already, and 
  -- they never propagate cotangents (they have no predecessors in the linear
  -- program), so there's nothing to do.
  return ()
transposeBinding b@(Binding v (ShaxprF mty op args)) = do
    -- Given v, get dL/dv = \ddot{v}.
    -- We're walking the program backwards, and we've just seen the
    -- creation of variable v. At this point we'll write down the
    -- cotangent of v as the sum of all its cotangent summands.
    ctVars <- cannotFail $ maybeGetCotangentSummands v
    case ctVars of
      -- The primal parameters (dual returns) are their own cotangents.
      -- Since we don't want to say "x = id x" for them, we skip them.
        Just cts
            | cts == [v] -> do
              -- We pull back dL/dV along f for each argument.
              appendCotangents v mty op args
            | otherwise -> do
              -- We write the sum of cotangents, and record the final
              -- variable (which holds the full sum of cotangents)
              -- in cotangentMap.
              ctv <- flushCotangents mty v cts
              cotangentMap %= E.insert v ctv
              -- Now we pull back dL/dV along f for each argument.
              appendCotangents ctv mty op args
      -- This would be the case when a variable was used, but has no cotangent path
      -- to any return variable. This can happen with environment reads or constants,
      -- but also variables which are never used (drops). 
      -- TODO: Check that this is _actually_ a constant, or an environment read.
        Nothing -> return ()

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
  x' <- maybeRename x
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
appendCotangents dLdZ mty bb@(DotGeneral (DotDimensionNumbers [2] [1] [0] [0])) [x, y] = do
  tx <- getOriginalVarType x
  ty <- getOriginalVarType y
  (bt, b, n, m, p) <- case (tx, ty) of
    (TensorType bt [b, n, m], TensorType _ [_, _, p]) -> return (bt, b, n, m, p)
    _ -> throwError (Error "Invalid dot argument types." callStack)
  -- ========
  -- Note that exactly one of x and y are environment parameters, since otherwise we'd have
  -- a nonlinear multiplication (x * y) in the linear program.
  -- ========
  -- A calculation best done over a glass of scotch yields:
  -- If Z = XY, then dL/dX = dL/dZ Y^T, dL/dY = X^T dL/dZ.
  x' <- maybeRename x
  y' <- maybeRename y

  isXEnv <- isEnvironmentRead x
  isYEnv <- isEnvironmentRead y

  if isXEnv then do
    xt <- addCotangentBinding (TransposeShaxprF (Just (TensorType bt [b, m, n])) [0, 2, 1] x')
    xtdLdz <- addCotangentBinding (DotGeneralShaxprF (Just ty) (DotDimensionNumbers [2] [1] [0] [0]) xt dLdZ)
    addCotangent y xtdLdz
  else do
    assertTrue isYEnv (Error ("Found nonlinear dot product " ++ prettyShow bb ++ " in transpose. In linear program, the arguments are " ++ prettyShow [x, y] ++ ", in the cotangent program they are " ++ prettyShow [x', y']) callStack)
    yt <- addCotangentBinding (TransposeShaxprF (Just (TensorType bt [b, p, m])) [0, 2, 1] y')
    dLdzyt <- addCotangentBinding (DotGeneralShaxprF (Just tx) (DotDimensionNumbers [2] [1] [0] [0]) dLdZ yt)
    addCotangent x dLdzyt

appendCotangents _ _ op _ = error $ "Unimplemented transpose of " ++ show op

inversePerm :: [Int] -> [Int]
inversePerm x = map snd $ sort $ zip x [0 .. length x - 1]