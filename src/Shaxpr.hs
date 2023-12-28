{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE InstanceSigs #-}

module Shaxpr(Shaxpr(..), ShaxprF(..), Op(..), UnaryScalarOp(..), BinaryScalarOp(..),
              fromConstant, close,
              DotDimensionNumbers(..),
              pattern ConstantShaxprF,
              pattern ParamShaxprF,
              pattern AddShaxprF,
              pattern SubShaxprF,
              pattern MulShaxprF,
              pattern DivShaxprF,
              pattern SinShaxprF,
              pattern CosShaxprF,
              pattern ExpShaxprF,
              pattern MaxShaxprF,
              pattern MinShaxprF,
              pattern SignumShaxprF,
              pattern NegateShaxprF,
              pattern ReshapeShaxprF,
              pattern BroadcastShaxprF,
              pattern TransposeShaxprF,
              pattern DotGeneralShaxprF) where

import Types
import Data.Fix
import Text.Show.Deriving
import Data.Eq.Deriving
import Text.PrettyPrint.HughesPJClass
import Prelude hiding ((<>))
import Data.Array.Dynamic as D

data BinaryScalarOp = Add | Sub | Mul | Div | Min | Max deriving (Show, Eq, Ord)
instance Pretty BinaryScalarOp where
  pPrint Add = text "add"
  pPrint Sub = text "sub"
  pPrint Mul = text "mul"
  pPrint Div = text "div"
  pPrint Min = text "min"
  pPrint Max = text "max"


data UnaryScalarOp = Sin | Cos | Exp | Signum | Negate deriving (Show, Eq, Ord)
instance Pretty UnaryScalarOp where
  pPrint Sin = text "sin"
  pPrint Cos = text "cos"
  pPrint Signum = text "sign"
  pPrint Negate = text "negate"
  pPrint Exp = text "exp"

data DotDimensionNumbers = DotDimensionNumbers {
  lhsContracting :: DimIxs,
  rhsContracting :: DimIxs,
  lhsBatch :: DimIxs,
  rhsBatch :: DimIxs
} deriving (Eq, Ord, Show)
instance Pretty DotDimensionNumbers where
  pPrint (DotDimensionNumbers a b c d) = hcat $ punctuate (text " ") (map pPrint [a, b, c, d])

data Op = Param Int
         | Constant SomeArray
         | BinaryPointwise BinaryScalarOp
         | UnaryPointwise UnaryScalarOp
         | Reshape Shape
         | Broadcast DimIxs Shape
         | Transpose DimIxs
         | DotGeneral DotDimensionNumbers
         deriving (Show, Eq, Ord)
instance Pretty Op where
  pPrintPrec _ _ (Param k) = text "arg" <> int k
  pPrintPrec level _ (Constant a) = if level >= PrettyLevel 2
                                     then text (show a)-- pPrint a
                                     else text "constant [elided]"
  pPrintPrec _ _ (BinaryPointwise op) = pPrint op
  pPrintPrec _ _ (UnaryPointwise op) = pPrint op
  pPrintPrec _ _ (Reshape sh) = text $ "reshape " ++ show sh
  pPrintPrec _ _ (Transpose ixs) = text $ "transpose " ++ show ixs
  pPrintPrec _ _ (Broadcast ixs sh) = text $ "broadcast " ++ show ixs ++ " " ++ show sh
  pPrintPrec _ _ (DotGeneral dims) = case dims of
    DotDimensionNumbers [1] [0] [] [] -> text "dot"
    _ -> text $ "dot_general " ++ prettyShow dims

data ShaxprF rep = ShaxprF {
  exprTy :: Maybe TensorType,
  exprOp :: Op,
  exprArgs :: [rep]} deriving (Show, Eq, Ord, Functor)

instance Pretty rep => Pretty (ShaxprF rep) where
  pPrintPrec _ _ (ShaxprF _ (BinaryPointwise op) [x, y]) = pPrint op <> text " " <> pPrint x <> text " " <> pPrint y
  pPrintPrec _ _ (ShaxprF _ (UnaryPointwise op) [x]) = pPrint op <> text " " <> pPrint x
  pPrintPrec k l (ShaxprF _ op xs) =
    let terms = pPrintPrec k l op : map (pPrintPrec k l) xs
    in  mconcat (punctuate (text " ") terms)

$(deriveShow1 ''ShaxprF)
$(deriveEq1 ''ShaxprF)

newtype Shaxpr = Shaxpr {
  expr :: Fix ShaxprF
} deriving (Show, Eq)
instance Pretty (Fix ShaxprF) where
  pPrintPrec k l (Fix x) = pPrintPrec k l x

-- Helper patterns for matching on well-constructed expression trees.
-- Note these are not technically complete, as in, there exist syntactically
-- valid values of type Fix ShaxprF which are not matched by any of these
-- synonyms. For instance, `ShaxprF _ (Constant _) [_]`. This is
-- intentional.
pattern ConstantShaxprF :: Maybe TensorType -> SomeArray -> ShaxprF a
pattern ConstantShaxprF ty x = ShaxprF ty (Constant x) []
pattern ParamShaxprF :: Maybe TensorType -> Int -> ShaxprF a
pattern ParamShaxprF ty k = ShaxprF ty (Param k) []
pattern AddShaxprF :: Maybe TensorType -> a -> a -> ShaxprF a
pattern AddShaxprF ty x y = ShaxprF ty (BinaryPointwise Add) [x, y]
pattern SubShaxprF :: Maybe TensorType -> a -> a -> ShaxprF a
pattern SubShaxprF ty x y = ShaxprF ty (BinaryPointwise Sub) [x, y]
pattern MulShaxprF :: Maybe TensorType -> a -> a -> ShaxprF a
pattern MulShaxprF ty x y = ShaxprF ty (BinaryPointwise Mul) [x, y]
pattern DivShaxprF :: Maybe TensorType -> a -> a -> ShaxprF a
pattern DivShaxprF ty x y = ShaxprF ty (BinaryPointwise Div) [x, y]
pattern MinShaxprF :: Maybe TensorType -> a -> a -> ShaxprF a
pattern MinShaxprF ty x y = ShaxprF ty (BinaryPointwise Min) [x, y]
pattern MaxShaxprF :: Maybe TensorType -> a -> a -> ShaxprF a
pattern MaxShaxprF ty x y = ShaxprF ty (BinaryPointwise Max) [x, y]
pattern SinShaxprF :: Maybe TensorType -> a -> ShaxprF a
pattern SinShaxprF ty x =  ShaxprF ty (UnaryPointwise Sin) [x]
pattern CosShaxprF :: Maybe TensorType -> a -> ShaxprF a
pattern CosShaxprF ty x = ShaxprF ty (UnaryPointwise Cos) [x]
pattern ExpShaxprF :: Maybe TensorType -> a -> ShaxprF a
pattern ExpShaxprF ty x = ShaxprF ty (UnaryPointwise Exp) [x]
pattern ReshapeShaxprF :: Maybe TensorType -> Shape -> a -> ShaxprF a
pattern ReshapeShaxprF ty sh x = ShaxprF ty (Reshape sh) [x]
pattern BroadcastShaxprF :: Maybe TensorType -> DimIxs -> Shape -> a -> ShaxprF a
pattern BroadcastShaxprF ty ixs sh x = ShaxprF ty (Broadcast ixs sh) [x]
pattern TransposeShaxprF :: Maybe TensorType -> DimIxs -> a -> ShaxprF a
pattern TransposeShaxprF ty ixs x = ShaxprF ty (Transpose ixs) [x]
pattern SignumShaxprF :: Maybe TensorType -> a -> ShaxprF a
pattern SignumShaxprF ty x = ShaxprF ty (UnaryPointwise Signum) [x]
pattern NegateShaxprF :: Maybe TensorType -> a -> ShaxprF a
pattern NegateShaxprF ty x = ShaxprF ty (UnaryPointwise Negate) [x]
pattern DotGeneralShaxprF :: Maybe TensorType -> DotDimensionNumbers -> a -> a -> ShaxprF a
pattern DotGeneralShaxprF ty dims x y = ShaxprF ty (DotGeneral dims) [x, y]


arrayTensorType :: SomeArray -> TensorType
arrayTensorType (FloatArray arr) = TensorType TFloat (D.shapeL arr)
arrayTensorType (IntArray arr) = TensorType TInt (D.shapeL arr)

fromConstant :: SomeArray -> Shaxpr
fromConstant arr = Shaxpr . Fix $ ConstantShaxprF (Just $ arrayTensorType arr) arr

instance Num Shaxpr where
  --(+) = (Shaxpr . Fix) . (. unFix . expr) . AddShaxprF Nothing . unFix . expr
  --(+) = ((Shaxpr . Fix) .) . (. unFix . expr) . AddShaxprF Nothing . unFix . expr
  (+) = ((Shaxpr . Fix) .) . (. expr) . AddShaxprF Nothing . expr
  (-) = ((Shaxpr . Fix) .) . (. expr) . SubShaxprF Nothing . expr
  (*) = ((Shaxpr . Fix) .) . (. expr) . MulShaxprF Nothing . expr
  fromInteger = fromConstant . IntArray . D.fromList [] . return
  signum = Shaxpr . Fix . SignumShaxprF Nothing . expr
  negate = Shaxpr . Fix . NegateShaxprF Nothing . expr

instance Fractional Shaxpr where
  (/) = ((Shaxpr . Fix) .) . (. expr) . DivShaxprF Nothing . expr
  fromRational = fromConstant . FloatArray . D.fromList [] . return . fromRational

instance Floating Shaxpr where
  sin = Shaxpr . Fix . SinShaxprF Nothing . expr
  cos = Shaxpr . Fix . CosShaxprF Nothing . expr
  exp = Shaxpr . Fix . ExpShaxprF Nothing . expr

instance Ord Shaxpr where
  min = ((Shaxpr . Fix) .) . (. expr) . MinShaxprF Nothing . expr
  max = ((Shaxpr . Fix) . ) . (. expr) . MaxShaxprF Nothing . expr
  compare = error "Cannot compare abstract expressions for order."

class Closable t where
  close' :: Int -> t -> [Shaxpr]

instance Closable Shaxpr where
  close' = const return

instance Closable [Shaxpr] where
  close' = const id
              
instance Closable b => Closable (Shaxpr -> b) where
  close' k f = close' (k + 1) (f (Shaxpr (Fix (ParamShaxprF Nothing k))))

close :: Closable t => t -> [Shaxpr]
close = close' 0