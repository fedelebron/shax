{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE InstanceSigs #-}

-- The linearization of a function f at a point p is the best linear
-- approximation to f in a small neighborhood near p.
-- For instance, if we're taking f(x) = sin(x), and we're linearizing it at
-- x_0, that means we want a linear function g such that g(x) is the best
-- linear approximation to f in a neighborhood of x_0. Using Taylor series, we
-- know:
-- f(x_0 + dx) = f(x_0) + dx * cos(x_0) + O(dx^2)
-- Thus the best linear approximation to f at x_0 is g(dx) = f(x_0) + dx * cos(x_0).
-- Given f and x_0, the function linearize in this module computes
-- (f(x_0), \dx -> dx * cos(x_0)).
-- Note how \dx -> dx * cos(x_0) is _always_ linear in dx, and f(x_0) is a constant
-- term in dx.
module Linearize (linearize, LinearizedDefinition(..), evalLinearizedDefinition) where

import GHC.Stack
import Types
import Definition
import Error
import TypeInference (checkDefArgumentTypes)
import qualified Environment as E
import qualified Data.Map as M
import qualified Data.Array.Dynamic as D
import qualified BiMap as B
import Control.Monad.Except (throwError)
import Text.PrettyPrint.HughesPJClass (Pretty, pPrintPrec, pPrint, prettyShow, vcat)
import Debug.Trace

import Control.Monad
import Eval (evalFunc, evalDefinition)
import Binding
import Shaxpr
import Control.Monad.State
import Control.Lens hiding (op)
import BindingMonad
import Data.Maybe (fromJust)

type TangentVarName = VarName

data LinearizationState = LinearizationState {
  -- Maps a variable in the primal program to 
  -- its corresponding variable in the linear program.
  -- In YOLO parlance, this maps u to \dot{u}.
  _primalToTangent :: E.Env TangentVarName,

  -- In YOLO parlance, this maps the returned `du` in the
  -- primal program, to the variable in the tangent program
  -- that reads `du` from the passed environment.
  _nonLinearToLinear :: E.Env TangentVarName,

  -- Bindings for the tangent program.
  _tangentBindings :: E.Env Binding,
  
  -- Types of environment variables passed from
  -- the primal program's return to the linear
  -- program's parameters. Stored in reverse
  -- order.
  _environmentTypes :: [TensorType],

  _nextLinearParameter :: Int
}
makeLenses 'LinearizationState

type LinearizationComputation r = BindingMonadComputation LinearizationState r
type LinearMapper = BindingMapper LinearizationState

newTangentBinding :: ShaxprF TangentVarName -> LinearizationComputation TangentVarName
newTangentBinding expr = do
  tBinds <- use (extra . tangentBindings)
  let nn = E.nextName tBinds :: TangentVarName
  extra . tangentBindings %= E.insert nn (Binding nn expr)
  return nn

getLinearFromEnvironment :: HasCallStack => VarName -> LinearizationComputation TangentVarName
getLinearFromEnvironment v = do
  primalBinding <- (>>= lift) (E.lookup v <$> use env)
  let tt = fromJust . exprTy . bindExpr $ primalBinding
  varMap <- use (extra . nonLinearToLinear)
  case E.lookup v varMap of
    Right w -> return w
    Left _ -> do
      expr <- ParamShaxprF (Just tt) <$> (extra . nextLinearParameter <<+= 1)
      extra . environmentTypes %= (tt:)
      p <- newTangentBinding expr
      extra . nonLinearToLinear %= E.insert v p
      return p

getTangentForPrimal :: VarName -> LinearizationComputation TangentVarName
getTangentForPrimal v = use (extra . primalToTangent) >>= lift . E.lookup v

setTangentForPrimal :: VarName -> TangentVarName -> LinearizationComputation ()
setTangentForPrimal v dotv = cannotFail (extra . primalToTangent %= E.insert v dotv)

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

evalLinearizedDefinition :: LinearizedDefinition -> [SomeArray] -> [SomeArray] -> Either Error ([SomeArray], [SomeArray])
evalLinearizedDefinition def x dx = do
  allRet <- evalDefinition (nonlinear def) x
  let (y, env) = splitAt (length allRet - envSize def) allRet
  -- traceM ("Passing parameters: dx:\n" ++ prettyShow dx ++ "\nEnv:\n" ++ prettyShow env)
  dy <- evalDefinition (linear def) (dx ++ env)
  return (y, dy)

-- linearize p f = (f(p), f'_p)
linearize :: HasCallStack => Definition -> Either Error LinearizedDefinition
linearize def = do
  let numNonlinearParams = length (defArgTys def)
  let initialState = LinearizationState E.empty E.empty E.empty [] numNonlinearParams
  (nonLinearDef, finalState) <- walkBindings initialState (recordTangents linearizeBinding) def

  -- We may need to return some intermediate variables from the nonlinear
  -- definition, to the linear definition. We add all those intermediate
  -- variables to the return list of the nonlinear definition.
  let extraReturns = E.keys (finalState ^. nonLinearToLinear)
  let nonLinearDef' = nonLinearDef {
    defRet = defRet nonLinearDef ++ extraReturns
  }

  returnedTangents <- mapM (`E.lookup` (finalState ^. primalToTangent)) (defRet nonLinearDef)
  let linearDef = Definition {
    defName = "d" ++ defName def,
    defArgTys = defArgTys def ++ reverse (finalState ^. environmentTypes),
    defRet = returnedTangents,
    defBinds = E.toBindings (finalState ^. tangentBindings)
  }
  let eSize = length (finalState ^. environmentTypes)
  return $ LinearizedDefinition nonLinearDef' linearDef eSize

recordTangents :: (Binding -> LinearizationComputation (VarName, TangentVarName)) -> LinearMapper
recordTangents f v = do
  (w, dotw) <- f v
  setTangentForPrimal w dotw
  return w

linearizeBinding :: HasCallStack => Binding -> LinearizationComputation (VarName, TangentVarName)
linearizeBinding b@(Binding _ (ShaxprF Nothing _ _)) = throwError $ Error ("Cannot linearize an untyped binding: " ++ prettyShow b) callStack
linearizeBinding b@(Binding v (ShaxprF mty@(Just t) op args)) = do
  dargs <- mapM getTangentForPrimal args
  case (op, args, dargs) of
    (_, xs, dxs) | isLinearFunc op -> do
      -- If f(x) is linear, then f'(x)(dx) = f(dx).
      y <- newBind (ShaxprF mty op xs)
      dy <- newTangentBinding (ShaxprF mty op dxs)

      return (y, dy)      
    (Param k, [], []) -> do
      -- Parameters pass through unmodified. Note that while in the primal program
      -- we are reading an argument, in the linear program we are reading the
      -- tangent with respect to that argument. This is because the first n indices
      -- of the linear program's arguments correspond to the tangents of all n
      -- indices of the primal program's arguments.
      x <- newBind (ParamShaxprF mty k)
      dx <- newTangentBinding (ParamShaxprF mty k)
      return (x, dx)

    (UnaryPointwise Sin, [x], [dx]) -> do
      -- If f(x) = sin(x), then f'(x)(dx) = cos(x) * dx.
      sinx <- newBind (SinShaxprF mty x)

      cosx <- newBind (CosShaxprF mty x)
      cosx' <- getLinearFromEnvironment cosx
      sinx' <- newTangentBinding (MulShaxprF mty cosx' dx)
  
      return (sinx, sinx')

    (UnaryPointwise Cos, [x], [dx]) -> do
      -- If f(x) = cos(x), then f'(x)(dx) = -sin(x) * dx.
      cosx <- newBind (CosShaxprF mty x)

      sinx <- newBind (SinShaxprF mty x)
      negsinx <- newBind (NegateShaxprF mty sinx)

      negsinx' <- getLinearFromEnvironment negsinx
      cosx' <- newTangentBinding (MulShaxprF mty negsinx' dx)

      return (cosx, cosx')
    (UnaryPointwise Exp, [x], [dx]) -> do
      -- If f(x) = e^x, then f'(x)(dx) = e^x * dx.
      expx <- newBind (ExpShaxprF mty x)

      expx' <- getLinearFromEnvironment expx
      expdx <- newTangentBinding (MulShaxprF mty expx' dx)

      return (expx, expdx)
    (BinaryPointwise Mul, [x, y], [dx, dy]) -> do
      -- If f(x, y) = x * y, then f'(x, y)(dx, dy) = y * dx + x * dy.
      -- By convention, we only put tangents on the right argument,
      -- while the left argument is a primal value, read from the environment.
      xy <- newBind (MulShaxprF mty x y)

      x' <- getLinearFromEnvironment x
      y' <- getLinearFromEnvironment y
      x'dy <- newTangentBinding (MulShaxprF mty x' dy)
      y'dx <- newTangentBinding (MulShaxprF mty y' dx)

      xy' <- newTangentBinding (AddShaxprF mty x'dy y'dx)

      return (xy, xy')
    (BinaryPointwise Div, [x, y], [dx, dy]) -> do
      -- If f(x, y) = x / y, then
      -- f'(x, y)(dx, dy) = dx/y - x dy / y^2.
      --                      = (dx * y - x * dy) / y^2
      --                      = (y * dx - x * dy) / y^2
      -- This is valid only when y != 0.
      xdivy <- newBind (DivShaxprF mty x y)

      x' <- getLinearFromEnvironment x
      y' <- getLinearFromEnvironment y

      ydx <- newTangentBinding (MulShaxprF mty y' dx)
      xdy <- newTangentBinding (MulShaxprF mty x' dy)
      numerator <- newTangentBinding (SubShaxprF mty ydx xdy)
      denominator <- newTangentBinding (MulShaxprF mty y' y')
      xdivy' <- newTangentBinding (DivShaxprF mty numerator denominator)

      return (xdivy, xdivy')
    (Constant k, [], []) -> do
      -- If f(x) = k, then f'(x_0)(dx) = 0.
      constant <- newBind (ConstantShaxprF mty k)
      zero <- newTangentBinding (ConstantShaxprF mty (zeroLike k))

      return (constant, zero)
    (DotGeneral dims, [x, y], [dx, dy]) -> do
      adotb <- newBind (DotGeneralShaxprF mty dims x y)

      x' <- getLinearFromEnvironment x
      y' <- getLinearFromEnvironment y
      a <- newTangentBinding (DotGeneralShaxprF mty dims x' dy)
      b <- newTangentBinding (DotGeneralShaxprF mty dims dx y')
      adotb' <- newTangentBinding (AddShaxprF mty a b)

      return (adotb, adotb')
    _ -> throwError $ Error ("Unimplemented lineariation for " ++ prettyShow b) callStack


isLinearFunc :: Op -> Bool
isLinearFunc (UnaryPointwise op) | op `elem` [Id, Negate] = True
isLinearFunc (BinaryPointwise op) | op `elem` [Add, Sub] = True
isLinearFunc (Param _) = True
isLinearFunc (Reshape _) = True
isLinearFunc (Broadcast _ _) = True
isLinearFunc (Slice _ _) = True
isLinearFunc (Pad _ _) = True
isLinearFunc (ReduceSum _) = True
isLinearFunc (Transpose _) = True
isLinearFunc _ = False    

      
  

{-
linearizeBinding :: HasCallStack => LinearMapper 
linearizeBinding (Binding v e@(ShaxprF mty op args)) = do
  -- We get the primal values for all the arguments, and evaluate the
  -- function, to get its result.
  point <- use (extra . p)
  primals' <- use (extra . primalValues)
  remaps <- use remap
  primalVars <- case mapM (`B.lookupKey` remaps) args of
                  Just xs -> return xs
                  Nothing -> throwError $ Error ("Could not find primal variables corresponding to tangents: " ++ prettyShow args) callStack
  primalVals <- lift $ mapM (`E.lookup` primals') primalVars
  let res = case op of
              Param k -> point !! k
              _ -> evalFunc op primalVals
  setPrimalValue v res
  linearizeFunc mty op primalVals args

linearizeFunc :: HasCallStack => Maybe TensorType -> Op -> [SomeArray] -> [TangentVarName] -> LinearizationComputation VarName
linearizeFunc mty op primals tangents = case (op, primals, tangents) of
  (_, _, dxs) | isLinearFunc op -> do
    newBind (ShaxprF mty op dxs)
  (UnaryPointwise Sin, [x0], [dx]) -> do
    -- If f(x) = sin(x), then f'(x_0)(dx) = cos(x_0) * dx.
    a <- newBind (ConstantShaxprF mty x0)
    b <- newBind (CosShaxprF mty a)
    newBind (MulShaxprF mty b dx)
  (UnaryPointwise Cos, [x0], [dx]) -> do
    -- If f(x) = cos(x), then f'(x_0)(dx) = -cos(x_0) * dx.
    a <- newBind (ConstantShaxprF mty x0)
    b <- newBind (SinShaxprF mty a)
    c <- newBind (NegateShaxprF mty b)
    newBind (MulShaxprF mty c dx)
  (UnaryPointwise Exp, [x0], [dx]) -> do
    -- If f(x) = e^x, then f'(x_0)(dx) = e^{x_0} * dx.
    a <- newBind (ConstantShaxprF mty x0)
    b <- newBind (ExpShaxprF mty a)
    newBind (MulShaxprF mty b dx)
  (BinaryPointwise Mul, [x0, y0], [dx, dy]) -> do
    -- If f(x, y) = x * y, then f'(x_0, y_0)(dx, dy) = y_0 * dx + x_0 * dy.
    -- By convention, we only put tangents on the right argument,
    -- while the left argument is a primal.
    a <- newBind (ConstantShaxprF mty y0)
    b <- newBind (MulShaxprF mty a dx)
    c <- newBind (ConstantShaxprF mty x0)
    d <- newBind (MulShaxprF mty c dy)
    newBind (AddShaxprF mty b d)
  (BinaryPointwise Div, [x0, y0], [dx, dy]) -> do
    -- If f(x, y) = x / y, then
    -- f'(x_0, y_0)(dx, dy) = dx/d_0 - x_0 dy / y_0^2.
    --                      = (dx * y_0 - x_0 * dy) / y_0^2
    --                      = (y_0 * dx - x_0 * dy) / y_0^2
    -- This is valid only when y_0 != 0.
    assertTrue (0.0 `notElem` toFloatList y0) $ Error ("Division by zero in linearization of division.") callStack
    do
      a <- newBind (ConstantShaxprF mty y0)
      b <- newBind (MulShaxprF mty a dx)
      c <- newBind (ConstantShaxprF mty x0)
      d <- newBind (MulShaxprF mty c dy)
      e <- newBind (SubShaxprF mty b d)
      f <- newBind (MulShaxprF mty a a)
      newBind (DivShaxprF mty e f)
  (Constant k, [], []) -> do
    -- If f(x) = k, then f'(x_0)(dx) = 0.
    newBind (ConstantShaxprF mty (zeroLike k))
  (DotGeneral dims, [x0, y0], [dx, dy]) -> do
    x0c <- newBind (ConstantShaxprF (Just (someArrayType x0)) x0)
    y0c <- newBind (ConstantShaxprF (Just (someArrayType y0)) y0)
    a <- newBind (DotGeneralShaxprF mty dims x0c dy)
    b <- newBind (DotGeneralShaxprF mty dims dx y0c)
    newBind (AddShaxprF mty a b)
  (BinaryPointwise Min, [x0, y0], [dx, dy]) -> do
    -- Let f(x, y) = min(x, y).
    -- Then f'(x_0, y_0)(dx, dy) = dx * (df/dx)(x_0, y_0) + dy * (df/dy)(x_0, y_0)
    -- Take coordinate i. Say x_0_i < y_0_i.
    -- Then f(x_0, y_0)_i = min(x_0, y_0)_i = x_0_i.
    -- Thus (df/dx)_i = 1, (df/dy)_i = 0.
    -- Else, say y_0_i < x_0_i. Then (df/dx)_i = 0, (df/dy)_i = 1.
    -- We want it to be the case that if x_0_i = y_0_i, then
    -- (df/dx)_i = (df/dy)_i = 1/2, as if for this coordinate, f_i(x, y) = (x+y)/2.
    -- We accomplish this using a vector m, where each m_i is either 0, 1/2, or 1,
    -- such that
    -- f'(x_0, y_0)(dx, dy) = dx * m + dy * (1-m).
    case (x0, y0) of
      (FloatArray x0s, FloatArray y0s) -> do
        let mask = FloatArray (D.zipWithA balancedMinMask x0s y0s)
            ones = FloatArray (D.constant (D.shapeL x0s) 1.0)
        maskVar <- newBind (ConstantShaxprF mty mask)
        a <- newBind (MulShaxprF mty maskVar dx)
        onesVar <- newBind (ConstantShaxprF mty ones)
        negMaskVar <- newBind (SubShaxprF mty onesVar maskVar)
        b <- newBind (MulShaxprF mty negMaskVar dy)
        newBind (AddShaxprF mty a b)
      _ -> throwError (Error "Cannot take derivative of integer min." callStack)
  (BinaryPointwise Max, [x0, y0], [dx, dy]) -> do
    case (x0, y0) of
      (FloatArray x0s, FloatArray y0s) -> do
        let mask = FloatArray (D.zipWithA balancedMinMask y0s x0s)
            ones = FloatArray (D.constant (D.shapeL x0s) 1.0)
        maskVar <- newBind (ConstantShaxprF mty mask)
        a <- newBind (MulShaxprF mty maskVar dx)
        onesVar <- newBind (ConstantShaxprF mty ones)
        negMaskVar <- newBind (SubShaxprF mty onesVar maskVar)
        b <- newBind (MulShaxprF mty negMaskVar dy)
        newBind (AddShaxprF mty a b)
      _ -> throwError (Error "Cannot take derivative of integer max." callStack)
  _ -> throwError (Error ("Failed to linearize " ++ show op) callStack)

balancedMinMask :: Float -> Float -> Float
balancedMinMask x y =
  case compare x y of
    LT -> 1.0
    EQ -> 0.5
    GT -> 0.0

-}


