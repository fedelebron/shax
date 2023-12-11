{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE FlexibleContexts #-}

module TypeInference(inferTypes) where

import Control.Monad
import Control.Monad.Except(MonadError, throwError)
import Control.Monad.State
import qualified Data.Map as M
import Text.PrettyPrint.HughesPJClass
import GHC.Stack
import Data.List (sort)

import Error
import Definition
import Shaxpr
import Binding
import Types

type BindingState = M.Map VarName (ShaxprF VarName)
type BindingComputation = StateT BindingState (Either Error)

inferTypes :: Definition -> Either Error Definition
inferTypes defn = do
  let oldBinds = defBinds defn
      initialState = M.empty
  newBinds <- execStateT (mapM_ typeBind oldBinds) initialState
  let newBinds' = [Binding v e | (v, e) <- M.toList newBinds]
  return $ defn { defBinds = newBinds' }
    where
      paramTypes = M.fromList (zip [0..] (defArgTys defn))
      typeBind :: Binding -> BindingComputation ()
      typeBind (Binding v ex@(ShaxprF mty op args)) = do
        s <- get
        s' <- case mty of
          Just _ -> return (M.insert v ex s)
          Nothing -> do
            argTys <- lift $ mapM (lookupVariableType s) args
            ty <- lift $ inferExprType paramTypes op argTys
            return $ M.insert v (ShaxprF (Just ty) op args) s
        put s'

lookupVariableType :: HasCallStack => BindingState -> VarName -> Either Error TensorType
lookupVariableType s vn = do
  case M.lookup vn s of
    Nothing -> Left $ Error ("Undefined variable " ++ prettyShow vn) callStack
    Just e@(ShaxprF mt _ _) -> case mt of
      Nothing -> Left $ Error ("Internal error! " ++ prettyShow e ++ " had no type!") callStack
      Just t -> Right t

inferExprType :: M.Map Int TensorType -> Op -> [TensorType] -> Either Error TensorType
inferExprType paramTypes op argTys = case (op, argTys) of
  (Param k, []) ->
    case M.lookup k paramTypes of
      Nothing -> Left $ Error ("Invalid parameter number: " ++ show k ++ ". No such argument provided.") callStack
      Just ty -> Right ty
  (Constant k, []) -> return (someArrayType k)
  (UnaryPointwise _, [x]) -> return x
  (BinaryPointwise _, [x, y]) -> broadcastSemantics x y
  (Reshape sh, [TensorType bt sh']) -> do
    assertTrue (product sh == product sh') (Error ("Invalid reshape: " ++ show sh' ++ " -> " ++ show sh) callStack)
    return (TensorType bt sh)
  (Transpose ixs, [TensorType bt sh]) -> TensorType bt <$> applyPermutation ixs sh
  (Broadcast ixs shout, [TensorType bt shin]) -> do
    assertTrue (all id [shout !! (ixs !! i) == shin !! i | i <- ixs]) $ Error ("Invalid broadcast: " ++ show shin ++ " -> " ++ show shout ++ " with ixs = " ++ show ixs) callStack
    return (TensorType bt shout)
  (Signum, [x]) -> return x
  (DotGeneral (DotDimensionNumbers lhsC rhsC lhsB rhsB),
              [TensorType b lhsSh, TensorType b' rhsSh]) -> do
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
    let lhsBatch = [lhsSh !! i | i <- lhsB]
        rhsBatch = [rhsSh !! i | i <- rhsB]
    assertTrue (lhsBatch == rhsBatch) $
      Error ("Differing batch dimensions: " ++ show (lhsBatch, rhsBatch)) callStack
    let lhsContracting = [lhsSh !! i | i <- lhsC]
        rhsContracting = [rhsSh !! i | i <- rhsC]
    assertTrue (lhsContracting == rhsContracting) $
      Error ("Differing contracting dimensions: " ++ show (lhsContracting, rhsContracting)) callStack        
    let lhsNonContracting = [lhsSh !! i | i <- [0 .. lhsRank - 1],
                                          i `notElem` lhsB,
                                          i `notElem` lhsC]
        rhsNonContracting = [rhsSh !! i | i <- [0 .. rhsRank - 1],
                                          i `notElem` rhsB,
                                          i `notElem` rhsC]
        sh = lhsBatch ++ lhsNonContracting ++ rhsNonContracting
    return (TensorType b sh)


    
  _ -> Left $ Error ("Invalid number of arguments for " ++ prettyShow op ++ ": " ++ show (length argTys)) callStack

applyPermutation :: (HasCallStack, MonadError Error m) => [Int] -> Shape -> m Shape
applyPermutation perm sh =
    let n = length sh
    in  if sort perm == [0 .. n - 1]
        then return [sh !! i | i <- perm]
        else throwError (Error ("Invalid permutation: "
                                ++ show perm ++ " for shape " ++ show perm)
                               callStack)

broadcastSemantics :: HasCallStack => TensorType -> TensorType -> Either Error TensorType
broadcastSemantics (TensorType abt ash) (TensorType bbt bsh) =
  let (ash', bsh') = padToSameLength ash bsh
      result = zipWith max ash' bsh'
  in do
    zipWithM_ checkMatchingDim ash' bsh'
    unless (abt == bbt) $ throwError (Error "Different binary operation argument base types not supported." callStack)
    return $ TensorType abt result
  where
    checkMatchingDim :: HasCallStack => Int -> Int -> Either Error ()
    checkMatchingDim i j | (i == j) || (i == 1) || (j == 1) = return ()
                         | otherwise = throwError $ Error ("Incompatible dimensions for broadcast: " ++ show (i, j)) callStack
  
padToSameLength :: [Int] -> [Int] -> ([Int], [Int])
padToSameLength xs ys = let nx = length xs
                            ny = length ys
                        in (replicate (ny - nx) 1 ++ xs,
                            replicate (nx - ny) 1 ++ ys)