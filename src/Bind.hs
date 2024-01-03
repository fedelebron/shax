module Bind(Bind(..), Binding) where

import Text.PrettyPrint.HughesPJClass
import Prelude hiding ((<>))

import Shaxpr
import Types


data Bind v = Bind {
  bindVar :: v,
  bindExpr :: ShaxprF v
} deriving Show

type Binding = Bind Var

instance Pretty v => Pretty (Bind v) where
  pPrintPrec l k (Bind v ex) = pPrint v <> text " = " <> pPrintPrec l k ex