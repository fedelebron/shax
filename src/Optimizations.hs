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
import Bind
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
foldConstants (Def name argTys binds rets) =
  let foldedDef = Def name argTys newBinds rets
   in eliminateDeadCode foldedDef
  where
    newBinds = go E.empty binds
    go _ [] = []
    go env (b@(Bind v e@(ShaxprF op args)) : bs)
      | Param k <- op = b : go env bs
      | Right args' <- mapM (`E.lookup` env) args =
          let res = evalShaxpr . Shaxpr . Fix $ ShaxprF op (map Fix args')
              c = ConstantShaxprF res
           in Bind v c : go (E.insert v c env) bs
      | otherwise = b : go env bs

eliminateDeadCode :: Definition -> Definition
eliminateDeadCode (Def name argTys binds rets) = Def name argTys newBinds rets
  where
    newBinds = filter ((`S.member` allUsed) . bindVar) binds
    allUsed = collectUsed (S.fromList rets) (reverse binds)
    collectUsed forwardUses [] = forwardUses
    collectUsed forwardUses ((Bind v (ShaxprF _ args)) : bs)
      | S.member v forwardUses =
          let append = foldr (.) id (map S.insert args)
          in collectUsed (append forwardUses) bs
      | otherwise = collectUsed forwardUses bs

eliminateCommonSubexpressions :: Definition -> Definition
eliminateCommonSubexpressions (Def name argTys binds rets) =
  Def name argTys (reverse newBinds) newRets
  where
    (newBinds, renames) = go BM.empty M.empty [] binds
    newRets = map (fromJust . (`M.lookup` renames)) rets
    go _ renames newBinds [] = (newBinds, renames)
    go expressionNames renames newBinds (Bind v@(Var vn t) e : binds)
      | Just existingName <- BM.lookupKey e' expressionNames =
          go expressionNames (M.insert v (Var existingName t) renames) newBinds binds
      | otherwise =
          let (v', expressionNames') = BM.insert e' expressionNames
              newV = Var v' t
              renames' = M.insert v newV renames
           in go expressionNames' renames' (Bind newV e' : newBinds) binds
      where
        e' = maybeRename <$> e
        maybeRename name = M.findWithDefault name name renames

materializeBroadcasts :: (HasCallStack) => Definition -> Definition
materializeBroadcasts = walkBindingsOrDie () broadcaster
  where
    broadcaster :: BindingMapper ()
    broadcaster b@(Bind (Var _ t) (ShaxprF op@(BinaryPointwise _) [x, y])) |
      Var _ tx <- x,
      Var _ ty <- y,
      tx /= ty = do
        BroadcastInDimResult ixx ixy common <- lift $ broadcastInDims (tyShape tx) (tyShape ty)
        x' <- case ixx of
          Nothing -> return x
          Just s -> newBind t (BroadcastShaxprF s common x)
        y' <- case ixy of
          Nothing -> return y
          Just s -> newBind t (BroadcastShaxprF s common y)
        newBind t (ShaxprF op [x', y'])
    broadcaster b = keepBind b

data DimOrder = LHS | RHS deriving (Show)

canonicalizeDotGeneral :: Definition -> Definition
canonicalizeDotGeneral = walkBindingsOrDie () canonicalizer
  where
    canonicalizer :: BindingMapper ()
    canonicalizer b@(Bind (Var _ ty@(TensorType bt sh)) (DotGeneralShaxprF dims x y)) = do
      let Var _ (TensorType _ shx) = x
          Var _ (TensorType _ shy) = y
      let DotDimensionNumbers lhsContractingIxs rhsContractingIxs lhsBatchIxs rhsBatchIxs = dims
          lhsNonContractingIxs = getNonContracting shx lhsContractingIxs lhsBatchIxs
          rhsNonContractingIxs = getNonContracting shy rhsContractingIxs rhsBatchIxs
      x' <- transposeAndReshapeForMatmul LHS lhsContractingIxs lhsNonContractingIxs lhsBatchIxs shx x
      y' <- transposeAndReshapeForMatmul RHS rhsContractingIxs rhsNonContractingIxs rhsBatchIxs shy y
      let batch = map (shx !!) lhsBatchIxs
          lhsNonContracting = map (shx !!) lhsNonContractingIxs
          rhsNonContracting = map (shy !!) rhsNonContractingIxs
      let mmTy = TensorType bt [product batch, product lhsNonContracting, product rhsNonContracting]
          canonicalDotDims = DotDimensionNumbers [2] [1] [0] [0]
      z <- newBind mmTy (DotGeneralShaxprF canonicalDotDims x' y')
      newBind ty (ReshapeShaxprF (batch ++ lhsNonContracting ++ rhsNonContracting) z)
    canonicalizer x = keepBind x
    transposeAndReshapeForMatmul :: DimOrder -> DimIxs -> DimIxs -> DimIxs -> Shape -> Var -> BindingMonadComputation () Var
    transposeAndReshapeForMatmul dimOrder contracting nonContracting batch shx x = do
      let permParts = case dimOrder of
            LHS -> [batch, nonContracting, contracting]
            RHS -> [batch, contracting, nonContracting]
          perm = concat permParts
          permutedShape = map (shx !!) perm
          newShape = map (product . map (shx !!)) permParts
      let bt = tyBase (varType x)
      x' <- if perm == [0 .. length perm - 1]
            then return x
            else newBind (TensorType bt permutedShape) (TransposeShaxprF perm x)
      if newShape == permutedShape
      then return x'
      else newBind (TensorType bt newShape) (ReshapeShaxprF newShape x')
    getNonContracting sh contracting batch =
      let allDims = [0 .. length sh - 1]
       in allDims \\ (batch ++ contracting)

lowerReductionsToSumsOfSlices :: Definition -> Definition
lowerReductionsToSumsOfSlices = walkBindingsOrDie () lower
  where
    lower :: BindingMapper ()
    lower b@(Bind (Var _ ty@(TensorType bt sh)) (ReduceSumShaxprF ixs x)) = do
         let TensorType _ sh' = varType x
         v <- go bt sh' ixs x
         -- TODO: This need only be a squeeze, not a full reshape.
         newBind ty (ReshapeShaxprF sh v)
    lower x = keepBind x
    go _ _ [] x = return x
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
                tyLeft = TensorType bt shLeft
                tyRight = TensorType bt shRight
            left <- newBind tyLeft (SliceShaxprF (replicate rank 0) leftIxs v)
            right <- newBind tyRight (SliceShaxprF rightIxs sh v)
            left' <- splitAndSumAtDim bt shLeft i left
            right' <- splitAndSumAtDim bt shRight i right
            let finalType = TensorType bt finalShape
            newBind finalType (AddShaxprF left' right')
          else do
            -- If the dimension to split is odd, we split it into 1 and n - 1.
            let (rightIxs, leftIxs) = indicesForSplitAt i 1 sh
                shLeft = leftIxs
                shRight = zipWith (-) sh rightIxs
                tyLeft = TensorType bt shLeft
                tyRight = TensorType bt shRight
            left <- newBind tyLeft (SliceShaxprF (replicate rank 0) leftIxs v)
            right <- newBind tyRight (SliceShaxprF rightIxs sh v)
            right' <- splitAndSumAtDim bt shRight i right
            -- Note we don't need to recursively split `left`, as its
            -- i'th dimension is 1.
            let finalType = TensorType bt finalShape
            newBind finalType (AddShaxprF left right')


indicesForSplitAt :: Int -> Int -> Shape -> (DimIxs, DimIxs)
indicesForSplitAt i l sh = let complementPos = length sh - i
                               complement = (sh !! i) - l
                               ll = replicate i 0
                               lr = replicate (complementPos - 1) 0
                               rl = take i sh
                               rr = drop (i + 1) sh
                            in  (ll++l:lr, rl++complement:rr)