{-# LANGUAGE TemplateHaskell #-}

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
module Linearize (linearize) where

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
import Text.PrettyPrint.HughesPJClass (prettyShow)


import Control.Monad
import Eval (evalFunc)
import Binding
import Shaxpr
import Control.Monad.State
import Control.Lens hiding (op)
import BindingMonad

type TangentVarName = VarName

-- primalValues maps primal names to their values during the
-- evaluation of f(p).
-- p is the point at which we're linearizing.
data LinearizationState = LinearizationState {
  _primalValues :: E.Env SomeArray,
  _p :: [SomeArray]
}
makeLenses 'LinearizationState

type LinearizationComputation r = BindingMonadComputation LinearizationState r
type LinearMapper = BindingMapper LinearizationState
setPrimalValue :: VarName -> SomeArray -> LinearizationComputation ()
setPrimalValue name value = do
  extra . primalValues %= E.insert name value
  
-- linearize p f = (f(p), f'_p)
linearize :: HasCallStack => [SomeArray] -> Definition -> Either Error ([SomeArray], Definition)
linearize args def = do
  checkDefArgumentTypes args def
  let initialState = LinearizationState E.empty args
  (tangentDef, finalState) <- walkBindings initialState linearizeBinding def

  -- Note we look up the return values of the _original_ definition, since
  -- `primals` maps _primal_ names to their values, while `tangentDef` is
  -- the _tangent_ program.
  rets <- mapM (`E.lookup` (finalState ^. primalValues)) (defRet def)

  return (rets, tangentDef)  

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

isLinearFunc :: Op -> Bool
isLinearFunc (BinaryPointwise op) | op `elem` [Add, Sub] = True
isLinearFunc (Param _) = True
isLinearFunc (Reshape _) = True
isLinearFunc (Broadcast _ _) = True
isLinearFunc (Transpose _) = True
isLinearFunc _ = False