{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}

module Types(module Types) where

import Text.PrettyPrint.HughesPJClass
import Prelude hiding ((<>))

data TensorBaseType = TFloat | TInt | TBool deriving (Show, Eq, Ord)
instance Pretty TensorBaseType where
  pPrint TFloat = text "f32"
  pPrint TInt = text "i32"
  pPrint TBool = text "bool"

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

class PrettyVar v where
  prettyVar :: v -> Doc

instance PrettyVar VarName where
  prettyVar = pPrint


data Var = Var {
  varName :: VarName,
  varType :: TensorType
} deriving (Eq, Show, Ord)
instance Pretty Var where
  pPrint (Var name _) = pPrint name

instance PrettyVar Var where
  prettyVar v = hcat [pPrint (varName v),
                      text " :: ",
                      pPrint (varType v)]