module Definition(Def(..), Definition, LinearizedDefinition(..), toDef, rename, zipDefinitions) where

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
instance (PrettyVar v, Pretty v) => Pretty (Def v) where
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

data LinearizedDefinition = LinearizedDefinition {
  nonlinear :: Definition,
  linear :: Definition,
  -- The last {envSize} outputs of nonlinear
  -- are the last {envSize} inputs to linear.
  envSize :: Int
} deriving Show

instance Pretty LinearizedDefinition where
  pPrintPrec k level (LinearizedDefinition nl l _) = vcat [pPrintPrec k level nl,
                                                           pPrintPrec k level l]

zipDefinitions :: LinearizedDefinition -> Definition
zipDefinitions (LinearizedDefinition nonlinear linear eSize) = 
  let numNonLinearParams = length (defArgTys nonlinear)
      numNonLinearResults = length (defRet nonlinear) - eSize
      numLinearParams = length (defArgTys linear) - eSize
      envReturns = drop numNonLinearResults (defRet nonlinear)
      envParamNumbers = [numLinearParams .. ]
      envParamMap = M.fromList (zip envParamNumbers envReturns)
      maxNonLinearBindName = maximum (map (unVarName . varName . bindVar) (defBinds nonlinear))
      firstLinearBindName = 1 + maxNonLinearBindName
      fixVar (Var (VarName vn) vt) = Var (VarName (firstLinearBindName + vn)) vt
      fixBind (Bind vn e) = Bind (fixVar vn) (fmap fixVar e)
      fixParamRead (Bind v (ParamShaxprF k))
        -- If k >= numLinearParams, this is an environment read.
        -- We map these to identity functions of the corresponding return
        -- values from the primal program.
        | k >= numLinearParams,
          Just v' <- M.lookup k envParamMap = Bind v (IdShaxprF v')
      fixParamRead (Bind v (ParamShaxprF k))
        -- If k < numLinearParams, this is a linear parameter read.
        -- We map these to parameters in the new, zipped definition, all
        -- of which come after the original function's primal params.
        | k < numLinearParams = Bind v (ParamShaxprF (k + numNonLinearParams))
      fixParamRead x = x
      newLinearBinds = map (fixParamRead . fixBind) (defBinds linear)
  in  Def {
    defName = defName nonlinear ++ "_" ++ defName linear,
    defArgTys = defArgTys nonlinear ++ take numLinearParams (defArgTys linear),
    defBinds = defBinds nonlinear ++ newLinearBinds,
    defRet = take numNonLinearResults (defRet nonlinear) ++ map fixVar (defRet linear)
  }
