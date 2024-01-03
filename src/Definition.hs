module Definition(Def(..), Definition, toDef, rename) where

import Control.Monad.State
import qualified Data.Map as M
import Data.Fix
import Text.PrettyPrint.HughesPJClass hiding (empty)
import Prelude hiding ((<>))

import BiMap
import Types
import Bind
import Shaxpr

data Def v = Def {
  defName :: String,
  defArgTys :: [TensorType],
  defBinds :: [Bind v],
  defRet :: [v]
} deriving Show
instance Pretty v => Pretty (Def v) where
  pPrintPrec k' l (Def name argTys binds ret) =
    let args = [text "arg" <> int k <> text " :: " <> pPrint argTy | (k, argTy) <- zip [0.. ] argTys]
        args' = punctuate (text ", ") args
        argList = vcat args'
        header = text "def " <> text name <> text "(" <> argList <> text ") ="
        body = text "let " <> vcat (map (pPrintPrec k' l) binds)
        footer = text "in " <> pPrint ret
    in hang header 2 (body $$ footer)

type Definition = Def Var

type LabelState = BiMap VarName (ShaxprF VarName)
labeler :: ShaxprF VarName -> State LabelState VarName
labeler ex = do
  m <- get
  let (k, m') = insert ex m
  put m'
  return k

everywhere :: Monad m => (ShaxprF a -> m a) -> Shaxpr -> m a
everywhere f (Shaxpr (Fix (ShaxprF op args))) = (ShaxprF op <$> mapM (everywhere f . Shaxpr) args) >>= f

toDef :: String -> [TensorType] -> [Shaxpr] -> Def VarName
toDef name tys fs =
  let (returnLabels, labels) = runState (mapM (everywhere labeler) fs) empty
      binds = [Bind v e | (v, e) <- M.toList (to labels)]
  in  Def {
        defName = name,
        defArgTys = tys,
        defBinds = binds,
        defRet = returnLabels
        }

rename :: String -> Definition -> Definition
rename k def = def {defName = k}