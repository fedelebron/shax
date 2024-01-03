{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE InstanceSigs #-}

module HNP (HNP (..), stretchArr) where

import qualified Data.Array.Dynamic as D
import qualified Data.Array.Dynamic.MatMul as MM
import Data.Fix
import Data.Int (Int64)
import Data.List ((\\), intercalate)
import Shaxpr
import Types
import Tensor

class HNP a where
  reshape :: Shape -> a -> a
  broadcast :: DimIxs -> Shape -> a -> a
  transpose :: DimIxs -> a -> a
  slice :: DimIxs -> DimIxs -> a -> a
  pad :: [(Int, Int)] -> Float -> a -> a
  reduceSum :: DimIxs -> a -> a
  select :: a -> a -> a -> a
  dotGeneral :: DotDimensionNumbers -> a -> a -> a
  dot :: a -> a -> a
  eq :: a -> a -> a
  dot = dotGeneral (DotDimensionNumbers [1] [0] [] [])

wrapArrayOperation :: (forall a. Num a => D.Array a -> D.Array a) -> Tensor -> Tensor
wrapArrayOperation f (IntTensor arr) = IntTensor (f arr)
wrapArrayOperation f (FloatTensor arr) = FloatTensor (f arr)

broadcastArr :: [Int] -> Shape -> Tensor -> Tensor
broadcastArr dims sh = wrapArrayOperation (D.broadcast dims sh)

sliceArr :: DimIxs -> DimIxs -> Tensor -> Tensor
sliceArr sixs eixs = let axisLengths = zipWith (-) eixs sixs
                     in  wrapArrayOperation (D.slice (zip sixs axisLengths))

stretchArr :: Shape -> Tensor -> Tensor
stretchArr sh = wrapArrayOperation (D.stretch sh)

reshapeArr :: Shape -> Tensor -> Tensor
reshapeArr sh = wrapArrayOperation (D.reshape sh)

transposeArr :: DimIxs -> Tensor -> Tensor
transposeArr ixs = wrapArrayOperation (D.transpose ixs)

reduceSumArr :: DimIxs -> Tensor -> Tensor
reduceSumArr ixs = wrapArrayOperation f
  where
    f :: forall a. Num a => D.Array a -> D.Array a
    f arr = let sh = D.shapeL arr
                rank = length sh
                allDimIxs = [0 .. rank - 1]
                transposedIxs = (allDimIxs \\ ixs) ++ ixs
            in  D.rerank (rank - length ixs) (D.reduce (+) 0) (D.transpose transposedIxs arr)

padArr :: [(Int, Int)] -> Float -> Tensor -> Tensor
padArr lohi val (FloatTensor arr) = FloatTensor (D.pad lohi val arr)
padArr lohi val (IntTensor arr) = IntTensor (D.pad lohi (truncate val) arr)

-- MM.matMul can't deal with Integer, so we wrap to and from Int64.
-- Gross, I know.
matMulArr3 :: Tensor -> Tensor -> Tensor
matMulArr3 (IntTensor x) (IntTensor y) =
  let x' = D.mapA fromIntegral x :: D.Array Int64
      y' = D.mapA fromIntegral y :: D.Array Int64
   in IntTensor $ D.mapA fromIntegral (D.rerank2 1 MM.matMul x' y')
matMulArr3 (FloatTensor x) (FloatTensor y) =
  FloatTensor (D.rerank2 1 MM.matMul x y)
matMulArr3 _ _ = error "Cannot happen."

selectArr :: Tensor -> Tensor -> Tensor -> Tensor
selectArr (BoolTensor b) (FloatTensor x) (FloatTensor y) = FloatTensor (D.zipWith3A selector b x y)
selectArr (BoolTensor b) (IntTensor x) (IntTensor y) = IntTensor (D.zipWith3A selector b x y)
selectArr b x y = error $ "Invalid argument types to selectArr: " ++ intercalate ", " (map (show . tensorType) [b, x, y])

selector :: Bool -> a -> a -> a
selector True x _ = x
selector False _ y = y

eqArr :: Tensor -> Tensor -> Tensor
eqArr (FloatTensor xs) (FloatTensor ys) = BoolTensor (D.zipWithA (==) xs ys)
eqArr (IntTensor xs) (IntTensor ys) = BoolTensor (D.zipWithA (==) xs ys)
eqArr (BoolTensor xs) (BoolTensor ys) = BoolTensor (D.zipWithA (==) xs ys)

instance HNP Tensor where
  reshape = reshapeArr
  broadcast = broadcastArr
  slice = sliceArr
  pad = padArr
  transpose = transposeArr
  reduceSum = reduceSumArr
  select = selectArr
  eq = eqArr
                      

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

transposeAndReshapeForMatmul :: DimOrder -> DimIxs -> DimIxs -> DimIxs -> Tensor -> Tensor
transposeAndReshapeForMatmul dimOrder contracting nonContracting batch x =
  let permParts = case dimOrder of
        LHS -> [batch, nonContracting, contracting]
        RHS -> [batch, contracting, nonContracting]
      perm = concat permParts
      newShape = map (product . map (shape x !!)) permParts
   in reshapeArr newShape (transposeArr perm x)

instance HNP Shaxpr where
  reshape = ((Shaxpr . Fix) .) . (. expr) . ReshapeShaxprF
  broadcast ixs sh x = Shaxpr . Fix $ BroadcastShaxprF ixs sh (expr x)
  slice sixs eixs x = Shaxpr . Fix $ SliceShaxprF sixs eixs (expr x)
  pad lohi val x = Shaxpr . Fix $ PadShaxprF lohi val (expr x)
  reduceSum ixs x = Shaxpr . Fix $ ReduceSumShaxprF ixs (expr x)
  transpose = ((Shaxpr . Fix) .) . (. expr) . TransposeShaxprF
  dotGeneral d x y = Shaxpr . Fix $ DotGeneralShaxprF d (expr x) (expr y)
  select b x y = Shaxpr . Fix $ SelectShaxprF (expr b) (expr x) (expr y)
  eq x y = Shaxpr . Fix $ EqShaxprF (expr x) (expr y)