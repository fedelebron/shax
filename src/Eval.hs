module Eval(evalShaxpr, evalDefinition) where

import qualified Data.Map as M

import Control.Monad.State
import Control.Monad.Except
import Text.PrettyPrint.HughesPJClass
import Data.Fix
import GHC.Stack

import Error
import Types
import Shaxpr
import Binding
import Definition
import BroadcastSemantics
import HNP

wrapBroadcastSemantics :: (SomeArray -> SomeArray -> SomeArray) -> Fix ShaxprF -> Fix ShaxprF -> SomeArray
wrapBroadcastSemantics f x y =
  let x' = evalShaxprF x
      y' = evalShaxprF y
      shx = shape x'
      shy = shape y'
      BroadcastResult lR rR newShape = broadcastShapes shx shy
      x'' = stretchArr newShape (reshape lR x')
      y'' = stretchArr newShape (reshape rR y')
  in f x'' y''


evalShaxpr :: Shaxpr -> SomeArray
evalShaxpr = evalShaxprF . expr

evalShaxprF :: Fix ShaxprF -> SomeArray
evalShaxprF (ConstantShaxprF _ x) = x
evalShaxprF (SignumShaxprF _ x) = signum (evalShaxprF x)
evalShaxprF (AddShaxprF _ x y) = wrapBroadcastSemantics (+) x y
evalShaxprF (MulShaxprF _ x y) = wrapBroadcastSemantics (*) x y
evalShaxprF (DivShaxprF _ x y) = wrapBroadcastSemantics (/) x y
evalShaxprF (CosShaxprF _ x) = cos (evalShaxprF x)
evalShaxprF (SinShaxprF _ x) = sin (evalShaxprF x)
evalShaxprF (MinShaxprF _ x y) = wrapBroadcastSemantics min x y
evalShaxprF (MaxShaxprF _ x y) = wrapBroadcastSemantics max x y
evalShaxprF (BroadcastShaxprF _ sh x) = broadcast sh (evalShaxprF x)
evalShaxprF (TransposeShaxprF _ ixs x) = transpose ixs (evalShaxprF x)
evalShaxprF (ReshapeShaxprF _ sh x) = reshape sh (evalShaxprF x)
evalShaxprF (ParamShaxprF _ _) = error "Cannot evaluate an expression with unbound variables."
evalShaxprF (DotGeneralShaxprF _ dims x y) = dotGeneral dims (evalShaxprF x) (evalShaxprF y)
evalShaxprF e = error $ "Invalid expression being evaluated! " ++ show e

type BindingState = M.Map VarName SomeArray
type BindingComputation = StateT BindingState (Either Error)

evalDefinition :: HasCallStack => Definition -> [SomeArray] -> Either Error SomeArray
evalDefinition defn args = do
  bindValues <- execStateT (mapM_ runBind (defBinds defn)) M.empty
  case M.lookup (defRet defn) bindValues of
    Nothing -> Left (Error "Could not get return value." callStack)
    Just arr -> Right arr
  where
    argMap = M.fromList $ zip [0..] args
    bindTypeMap = M.fromList $ map (\b -> (bindLabel b, exprTy (bindExpr b))) (defBinds defn)
    runBind :: Binding -> BindingComputation ()
    runBind (Binding v (ShaxprF ty op args')) = do
      s <- get
      args'' <- lift $ mapM (argToConstantShaxprF s) args'
      arr <-
        case op of
          Param k -> case M.lookup k argMap of
                        Nothing -> throwError $ Error ("Cannot find argument " ++ prettyShow k) callStack
                        Just arr -> return arr
          _ -> return $ evalShaxprF (Fix (ShaxprF ty op args''))
      let s' = M.insert v arr s
      put s'
    argToConstantShaxprF :: BindingState -> VarName -> Either Error (Fix ShaxprF)
    argToConstantShaxprF s v = case (M.lookup v s, M.lookup v bindTypeMap) of
      (Just arr, Just t) -> Right $ ConstantShaxprF t arr
      _ -> Left $ Error ("Cannot find variable " ++ prettyShow v) callStack