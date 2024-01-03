{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE ViewPatterns #-}

module Shaxpr
  ( Shaxpr (..),
    ShaxprF (..),
    Op (..),
    UnaryScalarOp (..),
    BinaryScalarOp (..),
    fromConstant,
    close,
    DotDimensionNumbers (..),
    pattern ConstantShaxprF,
    pattern ParamShaxprF,
    pattern AddShaxprF,
    pattern SubShaxprF,
    pattern MulShaxprF,
    pattern DivShaxprF,
    pattern IdShaxprF,
    pattern SinShaxprF,
    pattern CosShaxprF,
    pattern ExpShaxprF,
    pattern MaxShaxprF,
    pattern MinShaxprF,
    pattern EqShaxprF,
    pattern SignumShaxprF,
    pattern NegateShaxprF,
    pattern ReshapeShaxprF,
    pattern BroadcastShaxprF,
    pattern ReduceSumShaxprF,
    pattern SliceShaxprF,
    pattern PadShaxprF,
    pattern TransposeShaxprF,
    pattern DotGeneralShaxprF,
    pattern SelectShaxprF
  )
where

import Data.Array.Dynamic as D
import Prelude hiding ((<>))
import Data.Eq.Deriving
import Data.Fix
import Text.PrettyPrint.HughesPJClass
import Text.Show.Deriving
import Types
import Tensor

data BinaryScalarOp = Add | Sub | Mul | Div | Min | Max | Eq deriving (Show, Eq, Ord)

instance Pretty BinaryScalarOp where
  pPrint Add = text "add"
  pPrint Sub = text "sub"
  pPrint Mul = text "mul"
  pPrint Div = text "div"
  pPrint Min = text "min"
  pPrint Max = text "max"
  pPrint Eq = text "eq"

data UnaryScalarOp = Id | Sin | Cos | Exp | Signum | Negate deriving (Show, Eq, Ord)

instance Pretty UnaryScalarOp where
  pPrint Id = text "id"
  pPrint Sin = text "sin"
  pPrint Cos = text "cos"
  pPrint Signum = text "sign"
  pPrint Negate = text "negate"
  pPrint Exp = text "exp"

data DotDimensionNumbers = DotDimensionNumbers
  { lhsContracting :: DimIxs,
    rhsContracting :: DimIxs,
    lhsBatch :: DimIxs,
    rhsBatch :: DimIxs
  }
  deriving (Eq, Ord, Show)

instance Pretty DotDimensionNumbers where
  pPrint (DotDimensionNumbers a b c d) = hcat $ punctuate (text " ") (map pPrint [a, b, c, d])

data Op
  = Param Int
  | Constant Tensor
  | BinaryPointwise BinaryScalarOp
  | UnaryPointwise UnaryScalarOp
  | Reshape Shape
  | Broadcast DimIxs Shape
  | ReduceSum DimIxs
  -- TODO: Support strides.
  | Slice DimIxs DimIxs 
  -- TODO: Gross: If the argument an integer tensor, truncates the Float.
  | Pad [(Int, Int)] Float 
  | Transpose DimIxs
  | DotGeneral DotDimensionNumbers
  | Select
  deriving (Show, Eq, Ord)

instance Pretty Op where
  pPrintPrec _ _ (Param k) = text "arg" <> int k
  pPrintPrec level _ (Constant a) =
    if level >= PrettyLevel 2
      then text (show a) -- pPrint a
      else text "constant [elided]"
  pPrintPrec _ _ (BinaryPointwise op) = pPrint op
  pPrintPrec _ _ (UnaryPointwise op) = pPrint op
  pPrintPrec _ _ (Reshape sh) = text $ "reshape " ++ show sh
  pPrintPrec _ _ (Transpose ixs) = text $ "transpose " ++ show ixs
  pPrintPrec _ _ (Broadcast ixs sh) = text $ "broadcast " ++ show ixs ++ " " ++ show sh
  pPrintPrec _ _ (Slice sixs eixs) = text $ "slice " ++ show sixs ++ " " ++ show eixs
  pPrintPrec _ _ (Pad lohi val) = text $ "pad " ++ show lohi ++ " " ++ show val
  pPrintPrec _ _ (ReduceSum ixs) = text $ "reduce_sum " ++ show ixs
  pPrintPrec _ _ (DotGeneral dims) = case dims of
    DotDimensionNumbers [1] [0] [] [] -> text "dot"
    _ -> text $ "dot_general " ++ prettyShow dims
  pPrintPrec _ _ Select = text "select"

data ShaxprF rep = ShaxprF
  { exprOp :: Op,
    exprArgs :: [rep]
  }
  deriving (Show, Eq, Ord, Functor)

instance (Pretty rep) => Pretty (ShaxprF rep) where
  pPrintPrec _ _ (ShaxprF (BinaryPointwise op) [x, y]) = pPrint op <> text " " <> pPrint x <> text " " <> pPrint y
  pPrintPrec _ _ (ShaxprF (UnaryPointwise op) [x]) = pPrint op <> text " " <> pPrint x
  pPrintPrec k l (ShaxprF op xs) =
    let terms = pPrintPrec k l op : map (pPrintPrec k l) xs
     in mconcat (punctuate (text " ") terms)

$(deriveShow1 ''ShaxprF)
$(deriveEq1 ''ShaxprF)

newtype Shaxpr = Shaxpr
  { expr :: Fix ShaxprF
  }
  deriving (Show, Eq)

instance Pretty (Fix ShaxprF) where
  pPrintPrec :: PrettyLevel -> Rational -> Fix ShaxprF -> Doc
  pPrintPrec k l (Fix x) = pPrintPrec k l x

-- Helper patterns for matching on well-constructed expression trees.
-- Note these are not technically complete, as in, there exist syntactically
-- valid values of type Fix ShaxprF which are not matched by any of these
-- synonyms. For instance, `ShaxprF (Constant _) [_]`. This is
-- intentional.
pattern ConstantShaxprF :: Tensor -> ShaxprF a
pattern ConstantShaxprF x = ShaxprF (Constant x) []

pattern ParamShaxprF :: Int -> ShaxprF a
pattern ParamShaxprF k = ShaxprF (Param k) []

pattern AddShaxprF :: a -> a -> ShaxprF a
pattern AddShaxprF x y = ShaxprF (BinaryPointwise Add) [x, y]

pattern SubShaxprF :: a -> a -> ShaxprF a
pattern SubShaxprF x y = ShaxprF (BinaryPointwise Sub) [x, y]

pattern MulShaxprF :: a -> a -> ShaxprF a
pattern MulShaxprF x y = ShaxprF (BinaryPointwise Mul) [x, y]

pattern DivShaxprF :: a -> a -> ShaxprF a
pattern DivShaxprF x y = ShaxprF (BinaryPointwise Div) [x, y]

pattern MinShaxprF :: a -> a -> ShaxprF a
pattern MinShaxprF x y = ShaxprF (BinaryPointwise Min) [x, y]

pattern MaxShaxprF :: a -> a -> ShaxprF a
pattern MaxShaxprF x y = ShaxprF (BinaryPointwise Max) [x, y]

pattern EqShaxprF :: a -> a -> ShaxprF a
pattern EqShaxprF x y = ShaxprF (BinaryPointwise Eq) [x, y]

pattern IdShaxprF :: a -> ShaxprF a
pattern IdShaxprF x = ShaxprF (UnaryPointwise Id) [x]

pattern SinShaxprF :: a -> ShaxprF a
pattern SinShaxprF x = ShaxprF (UnaryPointwise Sin) [x]

pattern CosShaxprF :: a -> ShaxprF a
pattern CosShaxprF x = ShaxprF (UnaryPointwise Cos) [x]

pattern ExpShaxprF :: a -> ShaxprF a
pattern ExpShaxprF x = ShaxprF (UnaryPointwise Exp) [x]

pattern ReshapeShaxprF :: Shape -> a -> ShaxprF a
pattern ReshapeShaxprF sh x = ShaxprF (Reshape sh) [x]

pattern BroadcastShaxprF :: DimIxs -> Shape -> a -> ShaxprF a
pattern BroadcastShaxprF ixs sh x = ShaxprF (Broadcast ixs sh) [x]

pattern ReduceSumShaxprF :: DimIxs -> a -> ShaxprF a
pattern ReduceSumShaxprF ixs x = ShaxprF (ReduceSum ixs) [x]

pattern SliceShaxprF :: DimIxs -> DimIxs -> a -> ShaxprF a
pattern SliceShaxprF sixs eixs x = ShaxprF (Slice sixs eixs) [x]

pattern PadShaxprF :: [(Int, Int)] -> Float -> a -> ShaxprF a
pattern PadShaxprF lohi val x = ShaxprF (Pad lohi val) [x]

pattern TransposeShaxprF :: DimIxs -> a -> ShaxprF a
pattern TransposeShaxprF ixs x = ShaxprF (Transpose ixs) [x]

pattern SignumShaxprF :: a -> ShaxprF a
pattern SignumShaxprF x = ShaxprF (UnaryPointwise Signum) [x]

pattern NegateShaxprF :: a -> ShaxprF a
pattern NegateShaxprF x = ShaxprF (UnaryPointwise Negate) [x]

pattern DotGeneralShaxprF :: DotDimensionNumbers -> a -> a -> ShaxprF a
pattern DotGeneralShaxprF dims x y = ShaxprF (DotGeneral dims) [x, y]

pattern SelectShaxprF :: a -> a -> a -> ShaxprF a
pattern SelectShaxprF b x y = ShaxprF Select [b, x, y]

fromConstant :: Tensor -> Shaxpr
fromConstant arr = Shaxpr . Fix $ ConstantShaxprF arr

instance Num Shaxpr where
  (+) = ((Shaxpr . Fix) .) . (. expr) . AddShaxprF . expr
  (-) = ((Shaxpr . Fix) .) . (. expr) . SubShaxprF . expr
  (*) = ((Shaxpr . Fix) .) . (. expr) . MulShaxprF . expr
  fromInteger = fromConstant . IntTensor . D.fromList [] . return
  signum = Shaxpr . Fix . SignumShaxprF . expr
  negate = Shaxpr . Fix . NegateShaxprF . expr

instance Fractional Shaxpr where
  (/) = ((Shaxpr . Fix) .) . (. expr) . DivShaxprF . expr
  fromRational = fromConstant . FloatTensor . D.fromList [] . return . fromRational

instance Floating Shaxpr where
  sin =  Shaxpr . Fix . SinShaxprF . expr
  cos =  Shaxpr . Fix . CosShaxprF . expr
  exp =  Shaxpr . Fix . ExpShaxprF . expr

instance Ord Shaxpr where
  min = (( Shaxpr . Fix) .) . (. expr) . MinShaxprF . expr
  max = (( Shaxpr . Fix) .) . (. expr) . MaxShaxprF . expr
  compare = error "Cannot compare abstract expressions for order."

class Closable t where
  close' :: Int -> t -> [Shaxpr]

instance Closable Shaxpr where
  close' = const return

instance Closable [Shaxpr] where
  close' = const id

instance (Closable b) => Closable ( Shaxpr -> b) where
  close' k f = close' (k + 1) (f ( Shaxpr (Fix (ParamShaxprF k))))

close :: (Closable t) => t -> [Shaxpr]
close = close' 0