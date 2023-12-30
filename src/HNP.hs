{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE InstanceSigs #-}

module HNP (HNP (..), stretchArr) where

import qualified Data.Array.Dynamic as D
import qualified Data.Array.Dynamic.MatMul as MM
import Data.Fix
import Data.Int (Int64)
import Data.List ((\\))
import Shaxpr
import Types

class HNP a where
  reshape :: Shape -> a -> a
  broadcast :: DimIxs -> Shape -> a -> a
  transpose :: DimIxs -> a -> a
  slice :: DimIxs -> DimIxs -> a -> a
  reduceSum :: DimIxs -> a -> a
  dotGeneral :: DotDimensionNumbers -> a -> a -> a
  dot :: a -> a -> a
  dot = dotGeneral (DotDimensionNumbers [1] [0] [] [])

wrapArrayOperation :: (forall a. Num a => D.Array a -> D.Array a) -> SomeArray -> SomeArray
wrapArrayOperation f (IntArray arr) = IntArray (f arr)
wrapArrayOperation f (FloatArray arr) = FloatArray (f arr)

broadcastArr :: [Int] -> Shape -> SomeArray -> SomeArray
broadcastArr dims sh = wrapArrayOperation (D.broadcast dims sh)

sliceArr :: DimIxs -> DimIxs -> SomeArray -> SomeArray
sliceArr sixs eixs = wrapArrayOperation (D.slice (zip sixs eixs))

stretchArr :: Shape -> SomeArray -> SomeArray
stretchArr sh = wrapArrayOperation (D.stretch sh)

reshapeArr :: Shape -> SomeArray -> SomeArray
reshapeArr sh = wrapArrayOperation (D.reshape sh)

transposeArr :: DimIxs -> SomeArray -> SomeArray
transposeArr ixs = wrapArrayOperation (D.transpose ixs)

reduceSumArr :: DimIxs -> SomeArray -> SomeArray
reduceSumArr ixs = wrapArrayOperation f
  where
    f :: forall a. Num a => D.Array a -> D.Array a
    f arr = let sh = D.shapeL arr
                rank = length sh
                allDimIxs = [0 .. rank - 1]
                transposedIxs = (allDimIxs \\ ixs) ++ ixs
            in  D.rerank (rank - length ixs) (D.reduce (+) 0) (D.transpose transposedIxs arr)

-- MM.matMul can't deal with Integer, so we wrap to and from Int64.
-- Gross, I know.
matMulArr3 :: SomeArray -> SomeArray -> SomeArray
matMulArr3 (IntArray x) (IntArray y) =
  let x' = D.mapA fromIntegral x :: D.Array Int64
      y' = D.mapA fromIntegral y :: D.Array Int64
   in IntArray $ D.mapA fromIntegral (D.rerank2 1 MM.matMul x' y')
matMulArr3 (FloatArray x) (FloatArray y) =
  FloatArray (D.rerank2 1 MM.matMul x y)
matMulArr3 _ _ = error "Cannot happen."

instance HNP SomeArray where
  reshape = reshapeArr
  broadcast = broadcastArr
  slice = sliceArr
  transpose = transposeArr
  reduceSum = reduceSumArr
                      

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
         in allDims \\ (batch ++ contracting)

data DimOrder = LHS | RHS deriving (Show)

transposeAndReshapeForMatmul :: DimOrder -> DimIxs -> DimIxs -> DimIxs -> SomeArray -> SomeArray
transposeAndReshapeForMatmul dimOrder contracting nonContracting batch x =
  let permParts = case dimOrder of
        LHS -> [batch, nonContracting, contracting]
        RHS -> [batch, contracting, nonContracting]
      perm = concat permParts
      newShape = map (product . map (shape x !!)) permParts
   in reshapeArr newShape (transposeArr perm x)

instance HNP Shaxpr where
  reshape = ((Shaxpr . Fix) .) . (. expr) . ReshapeShaxprF Nothing
  broadcast ixs sh x = Shaxpr . Fix $ BroadcastShaxprF Nothing ixs sh (expr x)
  slice sixs eixs x = Shaxpr . Fix $ SliceShaxprF Nothing sixs eixs (expr x)
  reduceSum ixs x = Shaxpr . Fix $ ReduceSumShaxprF Nothing ixs (expr x)
  transpose = ((Shaxpr . Fix) .) . (. expr) . TransposeShaxprF Nothing
  dotGeneral d x y = Shaxpr . Fix $ DotGeneralShaxprF Nothing d (expr x) (expr y)