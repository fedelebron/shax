module Eval (evalFunc, evalShaxpr, evalDefinition, evalLinearizedDefinition) where

import Bind
import BroadcastSemantics
import Control.Monad.Except
import Control.Monad.State
import Data.Fix
import Data.List (splitAt)
import qualified Data.Map as M
import Definition
import Error
import GHC.Stack
import HNP
import Shaxpr
import Text.PrettyPrint.HughesPJClass
import Types
import Tensor

wrapBroadcastSemantics ::
  (HasCallStack) => (Tensor -> Tensor -> Tensor) -> Fix ShaxprF -> Fix ShaxprF -> Tensor
wrapBroadcastSemantics f x y =
  let x' = evalShaxprF x
      y' = evalShaxprF y
      BroadcastResult lR rR newShape = broadcastShapes (shape x') (shape y')
      x'' = stretchArr newShape (reshape lR x')
      y'' = stretchArr newShape (reshape rR y')
   in f x'' y''

evalShaxpr :: HasCallStack => Shaxpr -> Tensor
evalShaxpr = evalShaxprF . expr

evalShaxprF :: HasCallStack => Fix ShaxprF -> Tensor
evalShaxprF (Fix (ConstantShaxprF x)) = x
evalShaxprF (Fix (SignumShaxprF x)) = signum (evalShaxprF x)
evalShaxprF (Fix (NegateShaxprF x)) = negate (evalShaxprF x)
evalShaxprF (Fix (AddShaxprF x y)) = wrapBroadcastSemantics (+) x y
evalShaxprF (Fix (SubShaxprF x y)) = wrapBroadcastSemantics (-) x y
evalShaxprF (Fix (MulShaxprF x y)) = wrapBroadcastSemantics (*) x y
evalShaxprF (Fix (DivShaxprF x y)) = wrapBroadcastSemantics (/) x y
evalShaxprF (Fix (IdShaxprF x)) = evalShaxprF x
evalShaxprF (Fix (CosShaxprF x)) = cos (evalShaxprF x)
evalShaxprF (Fix (SinShaxprF x)) = sin (evalShaxprF x)
evalShaxprF (Fix (ExpShaxprF x)) = exp (evalShaxprF x)
evalShaxprF (Fix (MinShaxprF x y)) = wrapBroadcastSemantics min x y
evalShaxprF (Fix (MaxShaxprF x y)) = wrapBroadcastSemantics max x y
evalShaxprF (Fix (EqShaxprF x y)) = wrapBroadcastSemantics eq x y
evalShaxprF (Fix (BroadcastShaxprF ixs sh x)) = broadcast ixs sh (evalShaxprF x)
evalShaxprF (Fix (SliceShaxprF sixs eixs x)) = slice sixs eixs (evalShaxprF x)
evalShaxprF (Fix (PadShaxprF lohi val x)) = pad lohi val (evalShaxprF x)
evalShaxprF (Fix (TransposeShaxprF ixs x)) = transpose ixs (evalShaxprF x)
evalShaxprF (Fix (ReshapeShaxprF sh x)) = reshape sh (evalShaxprF x)
evalShaxprF (Fix (ReduceSumShaxprF ixs x)) = reduceSum ixs (evalShaxprF x)
evalShaxprF (Fix (ParamShaxprF _)) = error "Cannot evaluate an expression with unbound variables."
evalShaxprF (Fix (DotGeneralShaxprF dims x y)) = dotGeneral dims (evalShaxprF x) (evalShaxprF y)
evalShaxprF (Fix (SelectShaxprF b x y)) = select (evalShaxprF b) (evalShaxprF x) (evalShaxprF y)
evalShaxprF e = error $ "Invalid expression being evaluated! " ++ show e

type BindingState = M.Map Var Tensor

type BindingComputation = StateT BindingState (Either Error)

evalDefinition :: (HasCallStack) => Definition -> [Tensor] -> Either Error [Tensor]
evalDefinition defn args = do
  bindValues <- execStateT (mapM_ runBind (defBinds defn)) M.empty
  let argumentTypes = map tensorType args
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
    runBind :: Binding -> BindingComputation ()
    runBind bb@(Bind v (ShaxprF op args')) = do
      s <- get
      args'' <- lift $ mapM (argToConstantShaxprF s) args'
      arr <-
        case op of
          Param k -> case M.lookup k argMap of
            Nothing -> throwError $ Error ("Cannot find argument " ++ prettyShow k) callStack
            Just arr -> return arr
          _ -> return $ evalShaxprF (Fix (ShaxprF op args''))
      let s' = M.insert v arr s
      put s'
    argToConstantShaxprF :: BindingState -> Var -> Either Error (Fix ShaxprF)
    argToConstantShaxprF s v = case M.lookup v s of
      Just arr -> Right . Fix $ ConstantShaxprF arr
      _ -> Left $ Error ("Cannot find variable " ++ prettyShow v) callStack

evalFunc :: (HasCallStack) => Op -> [Tensor] -> Tensor
evalFunc op args = evalShaxprF (Fix (ShaxprF op args'))
  where
    args' = map (Fix . ConstantShaxprF) args

evalLinearizedDefinition :: LinearizedDefinition -> [Tensor] -> [Tensor] -> Either Error ([Tensor], [Tensor])
evalLinearizedDefinition def x dx = do
  allRet <- evalDefinition (nonlinear def) x
  let (y, env) = splitAt (length allRet - envSize def) allRet
  dy <- evalDefinition (linear def) (dx ++ env)
  return (y, dy)                                                        