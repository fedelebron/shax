{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}

module Types(module Types) where

import Data.Array.Dynamic as D
import Text.PrettyPrint.HughesPJClass
import Prelude hiding ((<>))
import GHC.Stack

data TensorBaseType = TFloat | TInt deriving (Show, Eq, Ord)
instance Pretty TensorBaseType where
  pPrint TFloat = text "f32"
  pPrint TInt = text "i32"

type Shape = [Int]
type DimIxs = [Int]

data TensorType = TensorType {
  tyBase :: TensorBaseType,
  tyShape :: Shape
} deriving (Show, Eq, Ord)
instance Pretty TensorType where
  pPrint (TensorType bt sh) = pPrint bt <> pPrint sh

newtype VarName = VarName {
  unVarName :: Int
} deriving (Eq, Show, Ord)
instance Pretty VarName where
  pPrint (VarName x) = text "x" <> int x

data SomeArray where
  FloatArray :: D.Array Float -> SomeArray
  IntArray :: D.Array Integer -> SomeArray
  deriving Eq
instance Show SomeArray where
  show (FloatArray arr) = show arr
  show (IntArray arr) = show arr

instance Pretty SomeArray where
  pPrint (FloatArray x) = pPrint x
  pPrint (IntArray x) = pPrint x

shape :: SomeArray -> Shape
shape (FloatArray arr) = D.shapeL arr
shape (IntArray arr) = D.shapeL arr

toFloatList :: SomeArray -> [Float]
toFloatList (IntArray _) = error "Cannot get a float list from an int array."
toFloatList (FloatArray arr) = D.toList arr

toFloatScalar :: SomeArray -> Float
toFloatScalar (IntArray _) = error "Cannot get a float scalar from an int array."
toFloatScalar (FloatArray arr) = D.unScalar arr

someArrayType :: SomeArray -> TensorType
someArrayType (FloatArray arr) = TensorType TFloat (D.shapeL arr)
someArrayType (IntArray arr) = TensorType TInt (D.shapeL arr)

sameTypeNumBin :: HasCallStack => (forall a. Num a => a -> a -> a) -> String -> SomeArray -> SomeArray -> SomeArray
sameTypeNumBin op opName (FloatArray x) (FloatArray y) = FloatArray (D.zipWithA op x y)
sameTypeNumBin op opName (IntArray x) (IntArray y) = IntArray (D.zipWithA op x y)
sameTypeNumBin _ opName x y = error $ "Invalid binary op " ++ opName ++ " arguments: " ++ show x ++ ", " ++ show y

oneTypeNum :: (forall a. Num a => a -> a) -> SomeArray -> SomeArray
oneTypeNum op (FloatArray x) = FloatArray (D.mapA op x)
oneTypeNum op (IntArray x) = IntArray (D.mapA op x)

instance Num SomeArray where
  (+) = sameTypeNumBin (+) "+"
  (-) = sameTypeNumBin (-) "-"
  (*) = sameTypeNumBin (*) "*"
  abs = oneTypeNum abs
  signum = oneTypeNum signum
  negate = oneTypeNum negate

  fromInteger = IntArray . D.fromList [] . return

fromFloatScalar :: Float -> SomeArray
fromFloatScalar = FloatArray . D.fromList [] . return

zeroLike :: SomeArray -> SomeArray
zeroLike (IntArray xs) = IntArray (D.constant (D.shapeL xs) 0)
zeroLike (FloatArray xs) = FloatArray (D.constant (D.shapeL xs) 0.0)


instance Fractional SomeArray where
  (FloatArray x) / (FloatArray y) = FloatArray (D.zipWithA (/) x y)
  _ / _ = error "Cannot happen."
  fromRational = fromFloatScalar . fromRational

oneTypeFloating :: (forall a. Floating a => a -> a) -> SomeArray -> SomeArray
oneTypeFloating op (FloatArray xs) = FloatArray (D.mapA op xs)
oneTypeFloating _ _ = error "Cannot happen."

instance Floating SomeArray where
  cos (FloatArray x) = FloatArray (D.mapA cos x)
  cos _ = error "Cannot happen."
  sin (FloatArray x) = FloatArray (D.mapA sin x)
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


instance Ord SomeArray where
  min (FloatArray x) (FloatArray y) = FloatArray (D.zipWithA min x y)
  min (IntArray x) (IntArray y) = IntArray (D.zipWithA min x y)
  min _ _ = error "Cannot take min of different types."
  max (FloatArray x) (FloatArray y) = FloatArray (D.zipWithA max x y)
  max (IntArray x) (IntArray y) = IntArray (D.zipWithA max x y)
  max _ _ = error "Cannot take max of different types."
  compare (IntArray x) (IntArray y) = compare x y
  compare (IntArray _) (FloatArray _) = GT
  compare (FloatArray _) (IntArray _) = LT
  compare (FloatArray x) (FloatArray y) = compare x y
