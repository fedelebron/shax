{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE RankNTypes #-}

module HNP(HNP(..), stretchArr) where

import Data.Int (Int64)
import Data.List ((\\))
import qualified Data.Array.Dynamic.MatMul as MM
import qualified Data.Array.Dynamic as D

import Types
import Shaxpr

class HNP a where
  reshape :: Shape -> a -> a
  broadcast :: DimIxs -> Shape -> a -> a
  transpose :: DimIxs -> a -> a
  dotGeneral :: DotDimensionNumbers -> a -> a -> a
  dot :: a -> a -> a
  dot = dotGeneral (DotDimensionNumbers [1] [0] [] [])

wrapArrayOperation :: (forall a. D.Array a -> D.Array a) -> SomeArray -> SomeArray
wrapArrayOperation f (IntArray arr) = IntArray (f arr)
wrapArrayOperation f (FloatArray arr) = FloatArray (f arr)

wrapArrayOperation2 :: (forall a. D.Array a -> D.Array a -> D.Array a) -> SomeArray -> SomeArray -> SomeArray
wrapArrayOperation2 f (IntArray arr) (IntArray arr') = IntArray (f arr arr')
wrapArrayOperation2 f (FloatArray arr) (FloatArray arr') = FloatArray (f arr arr')

broadcastArr :: [Int] -> Shape -> SomeArray -> SomeArray
broadcastArr dims sh = wrapArrayOperation (D.broadcast dims sh)

stretchArr :: Shape -> SomeArray -> SomeArray
stretchArr sh = wrapArrayOperation (D.stretch sh)

reshapeArr :: Shape -> SomeArray -> SomeArray
reshapeArr sh = wrapArrayOperation (D.reshape sh)

transposeArr :: DimIxs -> SomeArray -> SomeArray
transposeArr ixs = wrapArrayOperation (D.transpose ixs)

rerank2Arr :: Int -> (forall a. D.Array a -> D.Array a -> D.Array a) -> SomeArray -> SomeArray -> SomeArray
rerank2Arr n f = wrapArrayOperation2 (D.rerank2 n f)

-- We can't use wrapArrayOperation2 because MM.matMul has a typeclass constraint.
-- Also, MM.matMul can't deal with Integer, so we wrap to and from Int64.
-- Gross, I know.
matMulArr3 :: SomeArray -> SomeArray -> SomeArray
matMulArr3 (IntArray x) (IntArray y) =
  let x' = D.mapA fromIntegral x :: D.Array Int64
      y' = D.mapA fromIntegral y :: D.Array Int64
  in IntArray $ D.mapA fromIntegral (D.rerank2 1 MM.matMul x' y')
matMulArr3 (FloatArray x) (FloatArray y) =
  FloatArray (D.rerank2 1 MM.matMul x y)

instance HNP SomeArray where
  reshape = reshapeArr
  broadcast = broadcastArr
  transpose = transposeArr
  dotGeneral dims x y =
    let DotDimensionNumbers lhsContractingIxs rhsContractingIxs lhsBatchIxs rhsBatchIxs = dims
        lhsNonContractingIxs = getNonContracting x lhsContractingIxs lhsBatchIxs
        rhsNonContractingIxs = getNonContracting y rhsContractingIxs rhsBatchIxs
        x' = transposeAndReshapeForMatmul LHS lhsContractingIxs lhsNonContractingIxs lhsBatchIxs x
        y' = transposeAndReshapeForMatmul RHS rhsContractingIxs rhsNonContractingIxs rhsBatchIxs y
        z = matMulArr3 x' y'
        batch = [shape x !! i | i <- lhsBatchIxs]
        lhsNonContracting = map (shape x !!) lhsNonContractingIxs
        rhsNonContracting = map (shape y !!) rhsNonContractingIxs
    in reshape (batch ++ lhsNonContracting ++ rhsNonContracting) z
    where
      getNonContracting arr contracting batch =
        let allDims = [0 .. length (shape arr) - 1]
        in  allDims \\ (batch ++ contracting)


data DimOrder = LHS | RHS deriving Show
transposeAndReshapeForMatmul :: DimOrder -> DimIxs -> DimIxs -> DimIxs -> SomeArray -> SomeArray
transposeAndReshapeForMatmul dimOrder contracting nonContracting batch x =
  let allDims = [0 .. length (shape x) - 1]
      nonContracting = allDims \\ (batch ++ contracting)
      permParts = case dimOrder of
        LHS -> [batch, nonContracting, contracting]
        RHS -> [batch, contracting, nonContracting]
      perm = concat permParts
      newShape = map product $ map (map (shape x !!)) permParts
  in  reshapeArr newShape (transposeArr perm x)
   

instance HNP Shaxpr where
  reshape = (Shaxpr .) . (. expr) . ReshapeShaxprF Nothing
  broadcast ixs sh x = Shaxpr (BroadcastShaxprF Nothing ixs sh (expr x))
  transpose = (Shaxpr .) . (. expr) . TransposeShaxprF Nothing
  dotGeneral d x y = Shaxpr (DotGeneralShaxprF Nothing d (expr x) (expr y))