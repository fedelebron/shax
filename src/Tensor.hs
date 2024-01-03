{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE GADTs #-}

module Tensor where

import Data.Array.Dynamic as D
import Text.PrettyPrint.HughesPJClass
import GHC.Stack

import Types

data Tensor where
  FloatTensor :: D.Array Float -> Tensor
  IntTensor :: D.Array Integer -> Tensor
  BoolTensor :: D.Array Bool -> Tensor
  deriving Eq
instance Show Tensor where
  show (FloatTensor arr) = show arr
  show (IntTensor arr) = show arr
  show (BoolTensor arr) = show arr

instance Pretty Tensor where
  pPrint (FloatTensor x) = pPrint x
  pPrint (IntTensor x) = pPrint x
  pPrint (BoolTensor x) = pPrint x

shape :: Tensor -> Shape
shape (FloatTensor arr) = D.shapeL arr
shape (IntTensor arr) = D.shapeL arr
shape (BoolTensor arr) = D.shapeL arr

toFloatList :: Tensor -> [Float]
toFloatList (IntTensor _) = error "Cannot get a float list from an int tensor."
toFloatList (BoolTensor _) = error "Cannot get a float list from a bool tensor."
toFloatList (FloatTensor arr) = D.toList arr

toFloatScalar :: Tensor -> Float
toFloatScalar (IntTensor _) = error "Cannot get a float scalar from an int tensor."
toFloatScalar (BoolTensor _) = error "Cannot get a float scalar from a bool tensor."
toFloatScalar (FloatTensor arr) = D.unScalar arr

tensorType :: Tensor -> TensorType
tensorType (FloatTensor arr) = TensorType TFloat (D.shapeL arr)
tensorType (IntTensor arr) = TensorType TInt (D.shapeL arr)
tensorType (BoolTensor arr) = TensorType TBool (D.shapeL arr)

sameTypeNumBin :: HasCallStack => (forall a. Num a => a -> a -> a) -> String -> Tensor -> Tensor -> Tensor
sameTypeNumBin op _ (FloatTensor x) (FloatTensor y) = FloatTensor (D.zipWithA op x y)
sameTypeNumBin op _ (IntTensor x) (IntTensor y) = IntTensor (D.zipWithA op x y)
sameTypeNumBin _ opName x y = error $ "Invalid binary op " ++ opName ++ " arguments: " ++ show x ++ ", " ++ show y

oneTypeNum :: (forall a. Num a => a -> a) -> String -> Tensor -> Tensor
oneTypeNum op opName (FloatTensor x) = FloatTensor (D.mapA op x)
oneTypeNum op opName (IntTensor x) = IntTensor (D.mapA op x)
oneTypeNum op opName arr = error $ "Cannot apply pointwise op " ++ opName ++ " to argument: " ++ show arr

instance Num Tensor where
  (+) = sameTypeNumBin (+) "+"
  (-) = sameTypeNumBin (-) "-"
  (*) = sameTypeNumBin (*) "*"
  abs = oneTypeNum abs "abs"
  signum = oneTypeNum signum "sign"
  negate = oneTypeNum negate "negate"

  fromInteger = IntTensor . D.fromList [] . return

fromFloatScalar :: Float -> Tensor
fromFloatScalar = FloatTensor . D.fromList [] . return

zeroLike :: Tensor -> Tensor
zeroLike (IntTensor xs) = IntTensor (D.constant (D.shapeL xs) 0)
zeroLike (FloatTensor xs) = FloatTensor (D.constant (D.shapeL xs) 0.0)
zeroLike arr = error $ "Cannot create a zero of the same type as " ++ show (tensorType arr)

instance Fractional Tensor where
  (FloatTensor x) / (FloatTensor y) = FloatTensor (D.zipWithA (/) x y)
  _ / _ = error "Cannot happen."
  fromRational = fromFloatScalar . fromRational

oneTypeFloating :: (forall a. Floating a => a -> a) -> Tensor -> Tensor
oneTypeFloating op (FloatTensor xs) = FloatTensor (D.mapA op xs)
oneTypeFloating _ _ = error "Cannot happen."

instance Floating Tensor where
  cos (FloatTensor x) = FloatTensor (D.mapA cos x)
  cos _ = error "Cannot happen."
  sin (FloatTensor x) = FloatTensor (D.mapA sin x)
  sin _ = error "Cannot happen."
  pi = fromFloatScalar pi
  exp = oneTypeFloating exp
  log = oneTypeFloating log
  asin = oneTypeFloating asin
  acos = oneTypeFloating acos
  atan = oneTypeFloating atan
  sinh = oneTypeFloating sinh
  cosh = oneTypeFloating cosh
  asinh = oneTypeFloating asinh
  acosh = oneTypeFloating acosh
  atanh = oneTypeFloating atanh


instance Ord Tensor where
  min (FloatTensor x) (FloatTensor y) = FloatTensor (D.zipWithA min x y)
  min (IntTensor x) (IntTensor y) = IntTensor (D.zipWithA min x y)
  min _ _ = error "Cannot take min of different types."
  max (FloatTensor x) (FloatTensor y) = FloatTensor (D.zipWithA max x y)
  max (IntTensor x) (IntTensor y) = IntTensor (D.zipWithA max x y)
  max _ _ = error "Cannot take max of different types."
  compare (IntTensor x) (IntTensor y) = compare x y
  compare (IntTensor _) (FloatTensor _) = GT
  compare (FloatTensor _) (IntTensor _) = LT
  compare (FloatTensor x) (FloatTensor y) = compare x y