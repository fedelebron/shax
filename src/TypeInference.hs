{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TemplateHaskell #-}

module TypeInference (inferTypes, checkDefArgumentTypes, checkTypes) where

import Control.Monad
import Control.Monad.Except (MonadError, throwError)
import Control.Monad.State
import Data.List (sort, (\\), intercalate)
import qualified Data.Map as M
import GHC.Stack
import Text.PrettyPrint.HughesPJClass
import Control.Lens hiding (op)


import Bind
import qualified Environment as E
import Definition
import Error
import Shaxpr
import Types
import Tensor

data BindingState = BindingState {
  _labelToType :: M.Map VarName TensorType,
  _bindEnv :: E.Env Binding
}
makeLenses 'BindingState

type BindingComputation = StateT BindingState (Either Error)

checkTypes :: Definition -> Either Error ()
checkTypes def = do
  let untyped = eraseDefTypes def
  reTyped <- inferTypes untyped
  let originalTypes = bindTypes def
      inferredTypes = bindTypes reTyped
      diff = diffMaps originalTypes inferredTypes
  unless (null diff) $
    throwError (Error ("Definition failed typecheck, differences between given types and correct types:" ++ prettyShow diff) callStack)

eraseDefTypes :: Definition -> Def VarName
eraseDefTypes def = Def {
  defName = defName def,
  defRet = map eraseVarType (defRet def),
  defArgTys = defArgTys def,
  defBinds = map eraseBindType (defBinds def)
}
  where
    eraseVarType (Var v _) = v
    eraseBindType (Bind v e) = Bind (eraseVarType v) (fmap eraseVarType e)

bindTypes :: Definition -> M.Map VarName TensorType
bindTypes = M.fromList . map (makeType . bindVar) . defBinds
  where
    makeType (Var vn t) = (vn, t)

diffMaps :: (Eq k, Eq v) => M.Map k v -> M.Map k v -> [(k, v, v)]
diffMaps m1 m2 = map snd . filter fst $ zipWith diffEntries (M.toList m1) (M.toList m2)
  where
    diffEntries (k, v) (k', v')
      | k /= k' = error "Cannot diff maps with different keys."
      | otherwise = (v /= v', (k, v, v'))


inferTypes :: HasCallStack => Def VarName -> Either Error Definition
inferTypes defn = do
  let oldBinds = defBinds defn
      initialState = BindingState M.empty E.empty
  finalState <- execStateT (mapM_ typeBind oldBinds) initialState
  case mapM (`M.lookup` (view labelToType finalState)) (defRet defn) of
    Just tys -> 
      return $ Def {
        defName = defName defn,
        defArgTys = defArgTys defn,
        defRet = zipWith Var (defRet defn) tys,
        defBinds = E.toBindings (view bindEnv finalState)
      }
    Nothing -> throwError (Error "Failed to type return bindings." callStack)
  where
    paramTypes = M.fromList (zip [0 ..] (defArgTys defn))
    typeBind :: Bind VarName -> BindingComputation ()
    typeBind b@(Bind vn (ShaxprF op args)) =
      prependToErrors ("While typing " ++ prettyShow b ++ ": ") $ do
        argTys <- mapM lookupVariableType args
        correctType <- lift $ inferExprType paramTypes op argTys
        let v' = Var vn correctType
        bindEnv %= E.insert v' (Bind v' (ShaxprF op (zipWith Var args argTys)))
        labelToType %= M.insert vn correctType

lookupVariableType :: HasCallStack => VarName -> BindingComputation TensorType
lookupVariableType vn = do
  ty <- cannotFail (M.lookup vn <$> use labelToType)
  case ty of
    Nothing -> throwError $ Error ("Undefined variable " ++ prettyShow vn) callStack
    Just t -> return t

inferExprType :: HasCallStack => M.Map Int TensorType -> Op -> [TensorType] -> Either Error TensorType
inferExprType paramTypes op argTys = case (op, argTys) of
  (Param k, []) ->
    case M.lookup k paramTypes of
      Nothing -> Left $ Error ("Invalid parameter number: " ++ show k ++ ". No such argument provided.") callStack
      Just ty -> Right ty
  (Constant k, []) -> return (tensorType k)
  (UnaryPointwise _, [x]) -> return x
  (BinaryPointwise op, [x, y]) -> broadcastSemantics op x y
  (Reshape sh, [TensorType bt sh']) -> do
    assertTrue (product sh == product sh') (Error ("Invalid reshape: " ++ show sh' ++ " -> " ++ show sh) callStack)
    return (TensorType bt sh)
  (Transpose ixs, [TensorType bt sh]) -> do
    assertTrue (length ixs == length sh) $ Error ("Invalid transposition indices " ++ show ixs ++ " for argument with shape " ++ show sh) callStack 
    assertTrue (sort ixs == [0 .. length sh - 1]) $ Error ("Invalid transposition indices, not a permutation: " ++ show ixs) callStack
    TensorType bt <$> applyPermutation ixs sh
  (Broadcast ixs shout, [TensorType bt shin]) -> do
    assertTrue (and [shout !! (ixs !! i) == shin !! i | i <- [0 .. length ixs - 1]]) $
      Error ("Invalid broadcast: " ++ show shin ++ " -> " ++ show shout ++ " with ixs = " ++ show ixs) callStack
    return (TensorType bt shout)
  (Slice sixs eixs, [TensorType bt shin]) -> do
    assertTrue (all (uncurry (<)) (zip sixs eixs)) $
      Error ("Invalid start and end slice indices: " ++ show sixs ++ ", " ++ show eixs) callStack
    assertTrue (all (>= 0) sixs) $ Error ("Invalid start indices for slice: " ++ show sixs) callStack
    assertTrue (length sixs == length eixs) $
      Error ("Start and end slice indices have different lengths: " ++ show sixs ++ ", " ++ show eixs) callStack
    assertTrue (length sixs == length shin) $
      Error ("Indices have rank different from operand: " ++ show sixs ++ ", " ++ show shin) callStack
    assertTrue (all (uncurry (<=)) (zip eixs shin)) $
      Error ("Slice end indices too large: " ++ show eixs ++ ", " ++ show shin) callStack
    return (TensorType bt (zipWith (-) eixs sixs))
  (Pad lohi _, [TensorType bt shin]) -> do
    assertTrue (length lohi == length shin) $
      Error
        ("Mus have exactly one padding lo, hi for each dimension, got " ++ show lohi ++ " for shape " ++ show shin)
        callStack
    let shout = zipWith (+) shin (map (uncurry (+)) lohi)
    return (TensorType bt shout)
  (ReduceSum ixs, [TensorType bt shin]) -> do
    let allDimIxs = [0 .. length shin - 1]
    assertTrue (all (`elem` allDimIxs) ixs) $
      Error ("Invalid reduction indices: " ++ show ixs ++ " for reduction of shape " ++ show shin) callStack
    let shout = map (shin !!) (allDimIxs \\ ixs)
    return (TensorType bt shout)
  ( DotGeneral (DotDimensionNumbers lhsC rhsC lhsB rhsB),
    [TensorType b lhsSh, TensorType b' rhsSh]
    ) -> do
      assertTrue (b == b') $ Error ("Cannot dot tensors of different base types: " ++ show (b, b')) callStack
      let lhsRank = length lhsSh
          rhsRank = length rhsSh
      assertTrue (all (\x -> x >= 0 && x < lhsRank) lhsC) $
        Error ("Invalid contracting dimensions " ++ show lhsC ++ " for operand with shape " ++ show lhsSh) callStack
      assertTrue (all (\x -> x >= 0 && x < lhsRank) lhsB) $
        Error ("Invalid batching dimensions " ++ show lhsB ++ " for operand with shape " ++ show lhsSh) callStack
      assertTrue (all (\x -> x >= 0 && x < rhsRank) rhsC) $
        Error ("Invalid contracting dimensions " ++ show rhsC ++ " for operand with shape " ++ show rhsSh) callStack
      assertTrue (all (\x -> x >= 0 && x < rhsRank) rhsB) $
        Error ("Invalid batching dimensions " ++ show rhsB ++ " for operand with shape " ++ show rhsSh) callStack
      let lhsB' = [lhsSh !! i | i <- lhsB]
          rhsB' = [rhsSh !! i | i <- rhsB]
      assertTrue (lhsB' == rhsB') $
        Error ("Differing batch dimensions: " ++ show (lhsB', rhsB')) callStack
      let lhsContracting' = [lhsSh !! i | i <- lhsC]
          rhsContracting' = [rhsSh !! i | i <- rhsC]
      assertTrue (product lhsContracting' == product rhsContracting') $
        Error ("Differing contracting dimension products: " ++ show (lhsContracting', rhsContracting')) callStack
      let lhsNonContracting =
            [ lhsSh !! i | i <- [0 .. lhsRank - 1], i `notElem` lhsB, i `notElem` lhsC
            ]
          rhsNonContracting =
            [ rhsSh !! i | i <- [0 .. rhsRank - 1], i `notElem` rhsB, i `notElem` rhsC
            ]
          sh = lhsB' ++ lhsNonContracting ++ rhsNonContracting
      return (TensorType b sh)
  (Select, [TensorType TBool shb, TensorType tx shx, TensorType ty shy]) -> do
    assertTrue (tx == ty) $ Error ("Cannot select tensors of different base types: " ++ show (tx, ty)) callStack
    assertTrue (shb == shx && shx == shy) $ Error ("Invalid argument shapes for select: " ++ intercalate ", " (map show [shb, shx, shy])) callStack
    return (TensorType tx shx)
  _ -> Left $ Error ("Invalid number of arguments for " ++ prettyShow op ++ ": " ++ show (length argTys)) callStack

applyPermutation :: (HasCallStack, MonadError Error m) => [Int] -> Shape -> m Shape
applyPermutation perm sh =
  let n = length sh
   in if sort perm == [0 .. n - 1]
        then return [sh !! i | i <- perm]
        else
          throwError
            ( Error
                ( "Invalid permutation: "
                    ++ show perm
                    ++ " for shape "
                    ++ show perm
                )
                callStack
            )

broadcastSemantics :: (HasCallStack) => BinaryScalarOp -> TensorType -> TensorType -> Either Error TensorType
broadcastSemantics op (TensorType abt ash) (TensorType bbt bsh) =
  let (ash', bsh') = padToSameLength ash bsh
      result = zipWith max ash' bsh'
      resultBaseType = if op `elem` [Eq]
                       then TBool
                       else abt
   in do
        zipWithM_ checkMatchingDim ash' bsh'
        unless (abt == bbt) $ throwError (Error "Different binary operation argument base types not supported." callStack)
        return $ TensorType resultBaseType result
  where
    checkMatchingDim :: (HasCallStack) => Int -> Int -> Either Error ()
    checkMatchingDim i j
      | (i == j) || (i == 1) || (j == 1) = return ()
      | otherwise = throwError $ Error ("Incompatible dimensions for broadcast: " ++ show (i, j)) callStack

padToSameLength :: [Int] -> [Int] -> ([Int], [Int])
padToSameLength xs ys =
  let nx = length xs
      ny = length ys
   in ( replicate (ny - nx) 1 ++ xs,
        replicate (nx - ny) 1 ++ ys
      )

checkDefArgumentTypes :: (HasCallStack) => [Tensor] -> Definition -> Either Error ()
checkDefArgumentTypes args def =
  let argTys = map tensorType args
      paramTys = defArgTys def
   in assertTrue (argTys == paramTys) $
        Error
          ( "Invalid argument types ("
              ++ show argTys
              ++ ") when calling definition "
              ++ defName def
              ++ ", which has parameter types "
              ++ show paramTys
          )
          callStack