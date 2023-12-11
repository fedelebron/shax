{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ViewPatterns #-}

module Shaxpr(Shaxpr(..), ShaxprF(..), Op(..), fromConstant, close,
              DotDimensionNumbers(..),
              pattern ConstantShaxprF,
              pattern ParamShaxprF,
              pattern AddShaxprF,
              pattern SubShaxprF,
              pattern MulShaxprF,
              pattern DivShaxprF,
              pattern SinShaxprF,
              pattern CosShaxprF,
              pattern MaxShaxprF,
              pattern MinShaxprF,
              pattern SignumShaxprF,
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


data UnaryScalarOp = Sin | Cos deriving (Show, Eq, Ord)
instance Pretty UnaryScalarOp where
  pPrint Sin = text "sin"
  pPrint Cos = text "cos"

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
         | Signum
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
  pPrintPrec _ _ Signum = text "sign"

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
  pPrintPrec _ _ _ = error "Unimplemented."

$(deriveShow1 ''ShaxprF)
$(deriveEq1 ''ShaxprF)

newtype Shaxpr = Shaxpr {
  expr :: Fix ShaxprF
} deriving (Show, Eq)

-- Helper patterns for matching on well-constructed expression trees.
-- Note these are not technically complete, as in, there exist syntactically
-- valid values of type Fix ShaxprF which are not matched by any of these
-- synonyms. For instance, `Fix ShaxprF _ (Constant _) [_]`. This is
-- intentional.
pattern ConstantShaxprF :: Maybe TensorType -> SomeArray -> Fix ShaxprF
pattern ConstantShaxprF ty x = Fix (ShaxprF ty (Constant x) [])
pattern ParamShaxprF :: Maybe TensorType -> Int -> Fix ShaxprF
pattern ParamShaxprF ty k = Fix (ShaxprF ty (Param k) [])
pattern AddShaxprF :: Maybe TensorType -> Fix ShaxprF -> Fix ShaxprF -> Fix ShaxprF
pattern AddShaxprF ty x y = Fix (ShaxprF ty (BinaryPointwise Add) [x, y])
pattern SubShaxprF :: Maybe TensorType -> Fix ShaxprF -> Fix ShaxprF -> Fix ShaxprF
pattern SubShaxprF ty x y = Fix (ShaxprF ty (BinaryPointwise Sub) [x, y])
pattern MulShaxprF :: Maybe TensorType -> Fix ShaxprF -> Fix ShaxprF -> Fix ShaxprF
pattern MulShaxprF ty x y = Fix (ShaxprF ty (BinaryPointwise Mul) [x, y])
pattern DivShaxprF :: Maybe TensorType -> Fix ShaxprF -> Fix ShaxprF -> Fix ShaxprF
pattern DivShaxprF ty x y = Fix (ShaxprF ty (BinaryPointwise Div) [x, y])
pattern MinShaxprF :: Maybe TensorType -> Fix ShaxprF -> Fix ShaxprF -> Fix ShaxprF
pattern MinShaxprF ty x y = Fix (ShaxprF ty (BinaryPointwise Min) [x, y])
pattern MaxShaxprF :: Maybe TensorType -> Fix ShaxprF -> Fix ShaxprF -> Fix ShaxprF
pattern MaxShaxprF ty x y = Fix (ShaxprF ty (BinaryPointwise Max) [x, y])
pattern SinShaxprF :: Maybe TensorType -> Fix ShaxprF -> Fix ShaxprF
pattern SinShaxprF ty x = Fix (ShaxprF ty (UnaryPointwise Sin) [x])
pattern CosShaxprF :: Maybe TensorType -> Fix ShaxprF -> Fix ShaxprF
pattern CosShaxprF ty x = Fix (ShaxprF ty (UnaryPointwise Cos) [x])
pattern ReshapeShaxprF :: Maybe TensorType -> Shape -> Fix ShaxprF -> Fix ShaxprF
pattern ReshapeShaxprF ty sh x = Fix (ShaxprF ty (Reshape sh) [x])
pattern BroadcastShaxprF :: Maybe TensorType -> DimIxs -> Shape -> Fix ShaxprF -> Fix ShaxprF
pattern BroadcastShaxprF ty ixs sh x = Fix (ShaxprF ty (Broadcast ixs sh) [x])
pattern TransposeShaxprF :: Maybe TensorType -> DimIxs -> Fix ShaxprF -> Fix ShaxprF
pattern TransposeShaxprF ty ixs x = Fix (ShaxprF ty (Transpose ixs) [x])
pattern SignumShaxprF :: Maybe TensorType -> Fix ShaxprF -> Fix ShaxprF
pattern SignumShaxprF ty x = Fix (ShaxprF ty Signum [x])
pattern DotGeneralShaxprF :: Maybe TensorType -> DotDimensionNumbers -> Fix ShaxprF -> Fix ShaxprF -> Fix ShaxprF
pattern DotGeneralShaxprF ty dims x y = Fix (ShaxprF ty (DotGeneral dims) [x, y])


arrayTensorType :: SomeArray -> TensorType
arrayTensorType (FloatArray arr) = TensorType TFloat (D.shapeL arr)
arrayTensorType (IntArray arr) = TensorType TInt (D.shapeL arr)

fromConstant :: SomeArray -> Shaxpr
fromConstant arr = Shaxpr $ ConstantShaxprF (Just $ arrayTensorType arr) arr

instance Num Shaxpr where
  (+) = (Shaxpr .) . (. expr) . AddShaxprF Nothing . expr
  (-) = (Shaxpr .) . (. expr) . SubShaxprF Nothing . expr
  (*) = (Shaxpr .) . (. expr) . MulShaxprF Nothing . expr
  fromInteger = fromConstant . IntArray . D.fromList [] . return
  signum = Shaxpr . SignumShaxprF Nothing . expr

instance Fractional Shaxpr where
  (/) = (Shaxpr .) . (. expr) . DivShaxprF Nothing . expr
  fromRational = fromConstant . FloatArray . D.fromList [] . return . fromRational

instance Floating Shaxpr where
  sin = Shaxpr . SinShaxprF Nothing . expr
  cos = Shaxpr . CosShaxprF Nothing . expr

instance Ord Shaxpr where
  min = (Shaxpr .) . (. expr) . MinShaxprF Nothing . expr
  max = (Shaxpr .) . (. expr) . MaxShaxprF Nothing . expr
  compare = error "Cannot compare abstract expressions for order."

class Closable t where
  close' :: Int -> t -> Shaxpr

instance Closable Shaxpr where
  close' = const id
              
instance Closable b => Closable (Shaxpr -> b) where
  close' k f = close' (k + 1) (f (Shaxpr (ParamShaxprF Nothing k)))

close :: Closable t => t -> Shaxpr
close f = close' 0 f