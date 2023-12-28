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
import Control.Monad.Except (throwError)

import Control.Monad
import Eval (evalFunc)
import Binding
import Shaxpr
import Control.Monad.State
import Control.Lens hiding (op)

type TangentVarName = VarName

-- primals maps primal names to their values during the
-- evaluation of f(p).
-- tangents maps primal names to their tangent variables in f'_p.
-- dBindings are the bindings of f'_p, stored in reverse order.
-- p is the point at which we're linearizing.
data LinearizationState = LinearizationState {
  _primals :: E.Env SomeArray,
  _tangents :: E.Env TangentVarName,
  _dBindings :: [Binding],
  _primalBindings :: E.Env Binding,
  _p :: [SomeArray]
}
makeLenses 'LinearizationState

freshTangent :: State LinearizationState TangentVarName
freshTangent = VarName . length <$> use dBindings

setPrimalValue :: VarName -> SomeArray -> State LinearizationState ()
setPrimalValue name value = do
  primals %= E.insert name value
 
setPrimalTangent :: VarName -> TangentVarName -> State LinearizationState ()
setPrimalTangent v dv = do
  tangents %= E.insert v dv

createTangentBinding :: ShaxprF VarName -> State LinearizationState TangentVarName
createTangentBinding e = do
  dv <- freshTangent
  dBindings %= (Binding dv e <|)
  return dv
  
-- linearize p f = (f(p), f'_p)
-- TODO: Handle operations with implicit broadcasting. We probably want to make
-- broadcasts explicit.
linearize :: HasCallStack => [SomeArray] -> Definition -> Either Error ([SomeArray], Definition)
linearize args def = do
  checkDefArgumentTypes args def
  let initialState = LinearizationState E.empty E.empty [] (E.fromDefinition def) args
  finalState <- execStateT (mapM linearizeBinding (defBinds def)) initialState
  let bindings = finalState ^. dBindings
  newRets <- mapM (`E.lookup` _tangents finalState) (defRet def)
  let linearDef = Definition {
        defName = 'd' : defName def,
        defArgTys = defArgTys def,
        defBinds = reverse bindings,
        defRet = newRets
  }
  primalValues <- mapM (`E.lookup` _primals finalState) (defRet def)

  return (primalValues, linearDef)  

linearizeBinding :: HasCallStack => Binding -> StateT LinearizationState (Either Error) ()
linearizeBinding (Binding label e) = do
  (val, var) <- linearizeExpr e
  cannotFail (setPrimalValue label val)
  cannotFail (setPrimalTangent label var)
  return ()
  
linearizeExpr :: HasCallStack =>  ShaxprF VarName -> StateT LinearizationState (Either Error) (SomeArray, TangentVarName)
linearizeExpr (ShaxprF mty op args) = do
  -- We get the primal values for all the arguments, and evaluate the
  -- function, to get its result.
  point <- use p
  primals' <- use primals
  primalValues <- lift $ mapM (`E.lookup` primals') args
  let res = case op of
              Param k -> point !! k
              _ -> evalFunc op primalValues
  -- We then get the derivatives of all the arguments, and find the derivative
  -- of the function, to get an expression for the derivative of the arguments.
  tangents' <- use tangents
  -- Note `args` refers to variables in the primal program, so we translate them
  -- to variables in the derivative (a.k.a. tangents).
  tangentValues <- lift $ mapM (`E.lookup` tangents') args
  df <- linearizeFunc mty op primalValues tangentValues
  return (res, df)

linearizeFunc :: HasCallStack => Maybe TensorType -> Op -> [SomeArray] -> [TangentVarName] -> StateT LinearizationState (Either Error) TangentVarName
linearizeFunc mty op primals tangents = case (op, primals, tangents) of
  (_, _, dxs) | isLinearFunc op -> cannotFail $ do
    createTangentBinding (ShaxprF mty op dxs)
  (UnaryPointwise Sin, [x0], [dx]) -> cannotFail $ do
    -- If f(x) = sin(x), then f'(x_0)(dx) = cos(x_0) * dx.
    a <- createTangentBinding (ConstantShaxprF mty x0)
    b <- createTangentBinding (CosShaxprF mty a)
    createTangentBinding (MulShaxprF mty b dx)
  (UnaryPointwise Cos, [x0], [dx]) -> cannotFail $ do
    -- If f(x) = cos(x), then f'(x_0)(dx) = -cos(x_0) * dx.
    a <- createTangentBinding (ConstantShaxprF mty x0)
    b <- createTangentBinding (SinShaxprF mty a)
    c <- createTangentBinding (NegateShaxprF mty b)
    createTangentBinding (MulShaxprF mty c dx)
  (UnaryPointwise Exp, [x0], [dx]) -> cannotFail $ do
    -- If f(x) = e^x, then f'(x_0)(dx) = e^{x_0} * dx.
    a <- createTangentBinding (ConstantShaxprF mty x0)
    b <- createTangentBinding (ExpShaxprF mty a)
    createTangentBinding (MulShaxprF mty b dx)
  (BinaryPointwise Mul, [x0, y0], [dx, dy]) -> cannotFail $ do
    -- If f(x, y) = x * y, then f'(x_0, y_0)(dx, dy) = y_0 * dx + x_0 * dy.
    -- By convention, we only put tangents on the right argument,
    -- while the left argument is a primal.
    a <- createTangentBinding (ConstantShaxprF mty y0)
    b <- createTangentBinding (MulShaxprF mty a dx)
    c <- createTangentBinding (ConstantShaxprF mty x0)
    d <- createTangentBinding (MulShaxprF mty c dy)
    createTangentBinding (AddShaxprF mty b d)
  (BinaryPointwise Div, [x0, y0], [dx, dy]) -> do
    -- If f(x, y) = x / y, then
    -- f'(x_0, y_0)(dx, dy) = dx/d_0 - x_0 dy / y_0^2.
    --                      = (dx * y_0 - x_0 * dy) / y_0^2
    --                      = (y_0 * dx - x_0 * dy) / y_0^2
    -- This is valid only when y_0 != 0.
    assertTrue (0.0 `notElem` toFloatList y0) $ Error ("Division by zero in linearization of division.") callStack
    cannotFail $ do
      a <- createTangentBinding (ConstantShaxprF mty y0)
      b <- createTangentBinding (MulShaxprF mty a dx)
      c <- createTangentBinding (ConstantShaxprF mty x0)
      d <- createTangentBinding (MulShaxprF mty c dy)
      e <- createTangentBinding (SubShaxprF mty b d)
      f <- createTangentBinding (MulShaxprF mty a a)
      createTangentBinding (DivShaxprF mty e f)
  (Constant k, [], []) -> cannotFail $ do
    -- If f(x) = k, then f'(x_0)(dx) = 0.
    createTangentBinding (ConstantShaxprF mty (zeroLike k))
  (DotGeneral dims, [x0, y0], [dx, dy]) -> cannotFail $ do
    x0c <- createTangentBinding (ConstantShaxprF (Just (someArrayType x0)) x0)
    y0c <- createTangentBinding (ConstantShaxprF (Just (someArrayType y0)) y0)
    a <- createTangentBinding (DotGeneralShaxprF mty dims x0c dy)
    b <- createTangentBinding (DotGeneralShaxprF mty dims dx y0c)
    createTangentBinding (AddShaxprF mty a b)
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
      (FloatArray x0s, FloatArray y0s) -> cannotFail $ do
        let mask = FloatArray (D.zipWithA balancedMinMask x0s y0s)
            ones = FloatArray (D.constant (D.shapeL x0s) 1.0)
        maskVar <- createTangentBinding (ConstantShaxprF mty mask)
        a <- createTangentBinding (MulShaxprF mty maskVar dx)
        onesVar <- createTangentBinding (ConstantShaxprF mty ones)
        negMaskVar <- createTangentBinding (SubShaxprF mty onesVar maskVar)
        b <- createTangentBinding (MulShaxprF mty negMaskVar dy)
        createTangentBinding (AddShaxprF mty a b)
      _ -> throwError (Error "Cannot take derivative of integer min." callStack)
  (BinaryPointwise Max, [x0, y0], [dx, dy]) -> do
    case (x0, y0) of
      (FloatArray x0s, FloatArray y0s) -> cannotFail $ do
        let mask = FloatArray (D.zipWithA balancedMinMask y0s x0s)
            ones = FloatArray (D.constant (D.shapeL x0s) 1.0)
        maskVar <- createTangentBinding (ConstantShaxprF mty mask)
        a <- createTangentBinding (MulShaxprF mty maskVar dx)
        onesVar <- createTangentBinding (ConstantShaxprF mty ones)
        negMaskVar <- createTangentBinding (SubShaxprF mty onesVar maskVar)
        b <- createTangentBinding (MulShaxprF mty negMaskVar dy)
        createTangentBinding (AddShaxprF mty a b)
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