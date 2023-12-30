{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Optimizations
  ( foldConstants,
    eliminateDeadCode,
    eliminateCommonSubexpressions,
    materializeBroadcasts,
    canonicalizeDotGeneral,
    lowerReductionsToSumsOfSlices,
    OptimizationPass (..),
    applyPasses,
  )
where

import qualified BiMap as BM
import Binding
import BindingMonad
import BroadcastSemantics
import Control.Lens hiding (op)
import Control.Monad.Except (throwError)
import Control.Monad.State
import Data.Either (fromRight)
import Data.Fix
import Data.List ((\\))
import qualified Data.Map as M
import Data.Maybe (fromJust)
import qualified Data.Set as S
import Definition
import qualified Environment as E
import Error
import Eval
import GHC.Generics (Generic)
import GHC.Stack
import Shaxpr
import Test.QuickCheck
import Test.QuickCheck.Arbitrary.Generic
import Text.PrettyPrint.HughesPJClass (prettyShow)
import Types

import Debug.Trace

data OptimizationPass
  = DCE
  | CSE
  | ConstantFolding
  | MaterializeBroadcasts
  | CanonicalizeDotGeneral
  | LowerReductionsToSumsOfSlices
  deriving (Show, Eq, Ord, Generic)
  deriving (Arbitrary) via GenericArbitrary OptimizationPass

applyPass :: OptimizationPass -> Definition -> Definition
applyPass DCE = eliminateDeadCode
applyPass ConstantFolding = foldConstants
applyPass CSE = eliminateCommonSubexpressions
applyPass MaterializeBroadcasts = materializeBroadcasts
applyPass CanonicalizeDotGeneral = canonicalizeDotGeneral
applyPass LowerReductionsToSumsOfSlices = lowerReductionsToSumsOfSlices

applyPasses :: [OptimizationPass] -> Definition -> Definition
applyPasses = foldr (.) id . map applyPass

foldConstants :: Definition -> Definition
foldConstants (Definition name argTys binds rets) =
  let foldedDef = Definition name argTys newBinds rets
   in eliminateDeadCode foldedDef
  where
    newBinds = go E.empty binds
    go _ [] = []
    go env (b@(Binding v e@(ShaxprF mty op args)) : bs)
      | Param k <- op = b : go env bs
      | Right args' <- mapM (`E.lookup` env) args =
          let res = evalShaxpr . Shaxpr . Fix $ ShaxprF mty op (map Fix args')
              c = ConstantShaxprF mty res
           in Binding v c : go (E.insert v c env) bs
      | otherwise = b : go env bs

eliminateDeadCode :: Definition -> Definition
eliminateDeadCode (Definition name argTys binds rets) = Definition name argTys newBinds rets
  where
    newBinds = filter ((`S.member` allUsed) . bindLabel) binds
    allUsed = collectUsed (S.fromList rets) (reverse binds)
    collectUsed forwardUses [] = forwardUses
    collectUsed forwardUses ((Binding v (ShaxprF _ _ args)) : bs)
      | S.member v forwardUses =
          let append = foldr (.) id (map S.insert args)
           in collectUsed (append forwardUses) bs
      | otherwise = collectUsed forwardUses bs

eliminateCommonSubexpressions :: Definition -> Definition
eliminateCommonSubexpressions (Definition name argTys binds rets) =
  Definition name argTys (reverse newBinds) newRets
  where
    (newBinds, renames) = go BM.empty M.empty [] binds
    newRets = map (fromJust . (`M.lookup` renames)) rets
    go _ renames newBinds [] = (newBinds, renames)
    go expressionNames renames newBinds (Binding v e : binds)
      | Just existingName <- BM.lookupKey e' expressionNames =
          go expressionNames (M.insert v existingName renames) newBinds binds
      | otherwise =
          let (v', expressionNames') = BM.insert e' expressionNames
              renames' = M.insert v v' renames
           in go expressionNames' renames' (Binding v' e' : newBinds) binds
      where
        e' = maybeRename <$> e
        maybeRename name
          | Just newName <- M.lookup name renames = newName
          | otherwise = name

materializeBroadcasts :: (HasCallStack) => Definition -> Definition
materializeBroadcasts = walkBindingsOrDie () broadcaster
  where
    broadcaster :: BindingMapper ()
    broadcaster b@(Binding _ (ShaxprF mte op args)) =
      case (op, args) of
        (BinaryPointwise bop, [x, y]) -> do
          ee <- use env
          mtx <- exprTy . bindExpr <$> lift (E.lookup x ee)
          mty <- exprTy . bindExpr <$> lift (E.lookup y ee)
          case (mtx, mty) of
            (Just tx, Just ty) ->
              if tx == ty
                then keepBind b
                else do
                  BroadcastInDimResult ixx ixy common <- lift $ broadcastInDims (tyShape tx) (tyShape ty)
                  x' <- case ixx of
                    Nothing -> return x
                    Just s -> newBind (BroadcastShaxprF mte s common x)
                  y' <- case ixy of
                    Nothing -> return y
                    Just s -> newBind (BroadcastShaxprF mte s common y)
                  newBind (ShaxprF mte op [x', y'])
            _ -> throwError (Error ("Failed to get type of operands in " ++ prettyShow b) callStack)
        _ -> keepBind b

data DimOrder = LHS | RHS deriving (Show)

canonicalizeDotGeneral :: Definition -> Definition
canonicalizeDotGeneral = walkBindingsOrDie () canonicalizer
  where
    canonicalizer :: BindingMapper ()
    canonicalizer b@(Binding _ (DotGeneralShaxprF ty@(Just (TensorType bt sh)) dims x y)) = do
      currentEnv <- use env
      shx <- tyShape . fromJust . bindType <$> lift (E.lookup x currentEnv)
      shy <- tyShape . fromJust . bindType <$> lift (E.lookup y currentEnv)
      let DotDimensionNumbers lhsContractingIxs rhsContractingIxs lhsBatchIxs rhsBatchIxs = dims
          lhsNonContractingIxs = getNonContracting shx lhsContractingIxs lhsBatchIxs
          rhsNonContractingIxs = getNonContracting shy rhsContractingIxs rhsBatchIxs
      x' <- transposeAndReshapeForMatmul LHS lhsContractingIxs lhsNonContractingIxs lhsBatchIxs shx x
      y' <- transposeAndReshapeForMatmul RHS rhsContractingIxs rhsNonContractingIxs rhsBatchIxs shy y
      let batch = map (shx !!) lhsBatchIxs
          lhsNonContracting = map (shx !!) lhsNonContractingIxs
          rhsNonContracting = map (shy !!) rhsNonContractingIxs
      let mmType = Just . TensorType bt $ [product batch, product lhsNonContracting, product rhsNonContracting]
          canonicalDotDims = DotDimensionNumbers [2] [1] [0] [0]
      z <- newBind (DotGeneralShaxprF mmType canonicalDotDims x' y')
      newBind (ReshapeShaxprF ty (batch ++ lhsNonContracting ++ rhsNonContracting) z)
    canonicalizer x = keepBind x
    transposeAndReshapeForMatmul :: DimOrder -> DimIxs -> DimIxs -> DimIxs -> Shape -> VarName -> BindingMonadComputation () VarName
    transposeAndReshapeForMatmul dimOrder contracting nonContracting batch shx x = do
      let permParts = case dimOrder of
            LHS -> [batch, nonContracting, contracting]
            RHS -> [batch, contracting, nonContracting]
          perm = concat permParts
          permutedShape = map (shx !!) perm
          newShape = map (product . map (shx !!)) permParts
      currentEnv <- use env
      xBind <- lift (E.lookup x currentEnv)
      let bt = tyBase (fromJust (bindType xBind))
      x' <- if perm == [0 .. length perm - 1]
            then return x
            else newBind (TransposeShaxprF (Just (TensorType bt permutedShape)) perm x)
      if newShape == permutedShape
      then return x'
      else newBind (ReshapeShaxprF (Just (TensorType bt newShape)) newShape x')
    getNonContracting sh contracting batch =
      let allDims = [0 .. length sh - 1]
       in allDims \\ (batch ++ contracting)

lowerReductionsToSumsOfSlices :: Definition -> Definition
lowerReductionsToSumsOfSlices = walkBindingsOrDie () lower
  where
    lower :: BindingMapper ()
    lower b@(Binding _ (ReduceSumShaxprF mty ixs x))
      | Just (TensorType bt sh) <- mty = do
         TensorType _ sh' <- getBindingType x
         traceM ("About to reduce shape " ++ show sh' ++ " at indices " ++ show ixs)
         v <- go bt sh' ixs x
         newBind (ReshapeShaxprF mty sh v)
    lower x = keepBind x
    go bt _ [] x = return x
    go bt sh (i:ixs) x = do
      let newShape = take i sh ++ (1 : drop (i + 1) sh)
      z <- splitAndSumAtDim bt sh i x
      go bt newShape ixs z
    splitAndSumAtDim bt sh i v = 
      let dim = sh !! i
          rank = length sh
          finalShape = take i sh ++ (1: drop (i + 1) sh)
      in  if dim == 1
            -- If the dimension to split is 1, there's nothing to do.
          then return v
          else if even dim
            -- If the dimension to split is even, we split it in half.
          then do
            let (rightIxs, leftIxs) = indicesForSplitAt i (dim `div` 2) sh
                shLeft = leftIxs
                shRight = zipWith (-) sh rightIxs
                tyLeft = Just (TensorType bt shLeft)
                tyRight = Just (TensorType bt shRight)
            left <- newBind (SliceShaxprF tyLeft (replicate rank 0) leftIxs v)
            right <- newBind (SliceShaxprF tyRight rightIxs sh v)
            left' <- splitAndSumAtDim bt shLeft i left
            right' <- splitAndSumAtDim bt shRight i right
            newBind (AddShaxprF (Just (TensorType bt finalShape)) left' right')
          else do
            -- If the dimension to split is odd, we split it into 1 and n - 1.
            let (rightIxs, leftIxs) = indicesForSplitAt i 1 sh
                shLeft = leftIxs
                shRight = zipWith (-) sh rightIxs
                tyLeft = Just (TensorType bt shLeft)
                tyRight = Just (TensorType bt shRight)
            left <- newBind (SliceShaxprF tyLeft (replicate rank 0) leftIxs v)
            right <- newBind (SliceShaxprF tyRight rightIxs sh v)
            right' <- splitAndSumAtDim bt shRight i right
            -- Note we don't need to recursively split `left`, as its
            -- i'th dimension is 1.\
            newBind (AddShaxprF (Just (TensorType bt finalShape)) left right')


indicesForSplitAt :: Int -> Int -> Shape -> (DimIxs, DimIxs)
indicesForSplitAt i l sh = let complementPos = length sh - i
                               complement = (sh !! i) - l
                               ll = replicate i 0
                               lr = replicate (complementPos - 1) 0
                               rl = take i sh
                               rr = drop (i + 1) sh
                            in  (ll++l:lr, rl++complement:rr)




