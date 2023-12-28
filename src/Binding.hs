module Binding(Binding(..), bindType) where

import Types
import Shaxpr
import Text.PrettyPrint.HughesPJClass
import Prelude hiding ((<>))

data Binding = Binding {
  bindLabel :: VarName,
  bindExpr :: ShaxprF VarName
} deriving Show
instance Pretty Binding where
  pPrintPrec l k (Binding label ex) =
    let ty = if l >= PrettyLevel 1
             then prettyType (exprTy ex)
             else empty
    in pPrint label <> ty <> text " = " <> pPrintPrec l k ex
    where
      prettyType Nothing = text " :: *"
      prettyType (Just t) = text " :: " <> text (prettyShow t)

bindType :: Binding -> Maybe TensorType
bindType = exprTy . bindExpr