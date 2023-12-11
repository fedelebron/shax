module Definition(Definition(..), toDefinition) where

import Control.Monad.State
import qualified Data.Map as M
import Data.Fix
import Text.PrettyPrint.HughesPJClass hiding (empty)
import Prelude hiding ((<>))

import BiMap
import Types
import Binding
import Shaxpr

data Definition = Definition {
  defName :: String,
  defArgTys :: [TensorType],
  defBinds :: [Binding],
  defRet :: VarName
} deriving Show
instance Pretty Definition where
  pPrintPrec k' l (Definition name argTys binds ret) =
    let args = [text "arg" <> int k <> text " :: " <> pPrint argTy | (k, argTy) <- zip [0.. ] argTys]
        args' = punctuate (text ", ") args
        argList = vcat args'
        header = text "def " <> text name <> text "(" <> argList <> text ") ="
        body = text "let " <> vcat (map (pPrintPrec k' l) binds)
        footer = text "in " <> pPrint ret
    in hang header 2 (body $$ footer)


type LabelState = BiMap (ShaxprF Int)
labeler :: ShaxprF Int -> State LabelState Int
labeler ex = do
  m <- get
  let (k, m') = insert ex m
  put m'
  return k

everywhere :: Monad m => (ShaxprF a -> m a) -> Shaxpr -> m a
everywhere f (Shaxpr (Fix (ShaxprF ty op args))) = (ShaxprF ty op <$> mapM (everywhere f . Shaxpr) args) >>= f

toDefinition :: String -> [TensorType] -> Shaxpr -> Definition
toDefinition name tys f =
  let (returnLabel, labels) = runState (everywhere labeler f) empty
      binds = [Binding (VarName v) (fmap VarName e) | (v, e) <- M.toList (to labels)]
  in  Definition {
                  defName = name,
                  defArgTys = tys,
                  defBinds = binds,
                  defRet = VarName returnLabel
                 }