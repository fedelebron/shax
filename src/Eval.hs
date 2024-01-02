module Eval (evalFunc, evalShaxpr, evalDefinition) where

import Binding
import BroadcastSemantics
import Control.Monad.Except
import Control.Monad.State
import Data.Fix
import qualified Data.Map as M
import Definition
import Error
import GHC.Stack
import HNP
import Shaxpr
import Text.PrettyPrint.HughesPJClass
import Types

wrapBroadcastSemantics ::
  (HasCallStack) => (SomeArray -> SomeArray -> SomeArray) -> Fix ShaxprF -> Fix ShaxprF -> SomeArray
wrapBroadcastSemantics f x y =
  let x' = evalShaxprF x
      y' = evalShaxprF y
      BroadcastResult lR rR newShape = broadcastShapes (shape x') (shape y')
      x'' = stretchArr newShape (reshape lR x')
      y'' = stretchArr newShape (reshape rR y')
   in f x'' y''

evalShaxpr :: (HasCallStack) => Shaxpr -> SomeArray
evalShaxpr = evalShaxprF . expr

evalShaxprF :: (HasCallStack) => Fix ShaxprF -> SomeArray
evalShaxprF (Fix (ConstantShaxprF _ x)) = x
evalShaxprF (Fix (SignumShaxprF _ x)) = signum (evalShaxprF x)
evalShaxprF (Fix (NegateShaxprF _ x)) = negate (evalShaxprF x)
evalShaxprF (Fix (AddShaxprF _ x y)) = wrapBroadcastSemantics (+) x y
evalShaxprF (Fix (SubShaxprF _ x y)) = wrapBroadcastSemantics (-) x y
evalShaxprF (Fix (MulShaxprF _ x y)) = wrapBroadcastSemantics (*) x y
evalShaxprF (Fix (DivShaxprF _ x y)) = wrapBroadcastSemantics (/) x y
evalShaxprF (Fix (IdShaxprF _ x)) = evalShaxprF x
evalShaxprF (Fix (CosShaxprF _ x)) = cos (evalShaxprF x)
evalShaxprF (Fix (SinShaxprF _ x)) = sin (evalShaxprF x)
evalShaxprF (Fix (ExpShaxprF _ x)) = exp (evalShaxprF x)
evalShaxprF (Fix (MinShaxprF _ x y)) = wrapBroadcastSemantics min x y
evalShaxprF (Fix (MaxShaxprF _ x y)) = wrapBroadcastSemantics max x y
evalShaxprF (Fix (BroadcastShaxprF _ ixs sh x)) = broadcast ixs sh (evalShaxprF x)
evalShaxprF (Fix (SliceShaxprF _ sixs eixs x)) = slice sixs eixs (evalShaxprF x)
evalShaxprF (Fix (PadShaxprF _ lohi val x)) = pad lohi val (evalShaxprF x)
evalShaxprF (Fix (TransposeShaxprF _ ixs x)) = transpose ixs (evalShaxprF x)
evalShaxprF (Fix (ReshapeShaxprF _ sh x)) = reshape sh (evalShaxprF x)
evalShaxprF (Fix (ReduceSumShaxprF _ ixs x)) = reduceSum ixs (evalShaxprF x)
evalShaxprF (Fix (ParamShaxprF _ _)) = error "Cannot evaluate an expression with unbound variables."
evalShaxprF (Fix (DotGeneralShaxprF _ dims x y)) = dotGeneral dims (evalShaxprF x) (evalShaxprF y)
evalShaxprF e = error $ "Invalid expression being evaluated! " ++ show e

type BindingState = M.Map VarName SomeArray

type BindingComputation = StateT BindingState (Either Error)

evalDefinition :: (HasCallStack) => Definition -> [SomeArray] -> Either Error [SomeArray]
evalDefinition defn args = do
  bindValues <- execStateT (mapM_ runBind (defBinds defn)) M.empty
  let argumentTypes = map someArrayType args
      parameterTypes = defArgTys defn
  assertTrue (argumentTypes == parameterTypes) $
    Error
      ( "Invalid arguments during evaluation! Definition parameters have types "
          ++ prettyShow parameterTypes
          ++ ", given arguments have type "
          ++ show argumentTypes
      )
      callStack
  case mapM (`M.lookup` bindValues) (defRet defn) of
    Nothing -> Left (Error "Could not get return value." callStack)
    Just arrs -> Right arrs
  where
    argMap = M.fromList $ zip [0 ..] args
    bindTypeMap = M.fromList $ map (\b -> (bindLabel b, exprTy (bindExpr b))) (defBinds defn)
    runBind :: Binding -> BindingComputation ()
    runBind bb@(Binding v (ShaxprF ty op args')) = do
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
      (Just arr, Just t) -> Right . Fix $ ConstantShaxprF t arr
      _ -> Left $ Error ("Cannot find variable " ++ prettyShow v) callStack

evalFunc :: (HasCallStack) => Op -> [SomeArray] -> SomeArray
evalFunc op args = evalShaxprF (Fix (ShaxprF Nothing op args'))
  where
    args' = map (Fix . ConstantShaxprF Nothing) args