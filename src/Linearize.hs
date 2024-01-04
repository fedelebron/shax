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
module Linearize (linearize) where

import GHC.Stack
import qualified Data.Map as M
import qualified Data.Array.Dynamic as D
import Control.Monad.Except (throwError)
import Text.PrettyPrint.HughesPJClass (Pretty, pPrintPrec, pPrint, prettyShow, vcat)
import Data.Maybe (fromJust)
import Control.Monad
import Control.Monad.State

import qualified Environment as E
import Types
import Definition
import Error
import TypeInference (checkDefArgumentTypes)
import qualified BiMap as B
import Eval (evalFunc, evalDefinition)
import Bind
import Shaxpr
import Control.Lens hiding (op)
import BindingMonad
import Tensor

type TangentVar = Var

data LinearizationState = LinearizationState {
  -- Maps a variable in the primal program to 
  -- its corresponding variable in the linear program.
  -- In YOLO parlance, this maps u to \dot{u}.
  _primalToTangent :: E.Env TangentVar,

  -- In YOLO parlance, this maps the returned `du` in the
  -- primal program, to the variable in the tangent program
  -- that reads `du` from the passed environment.
  _nonLinearToLinear :: E.Env TangentVar,

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

newTangentBinding :: TensorType -> ShaxprF TangentVar -> LinearizationComputation TangentVar
newTangentBinding ty expr = do
  tBinds <- use (extra . tangentBindings)
  let nn = E.nextName tBinds
      v = Var nn ty :: TangentVar
  extra . tangentBindings %= E.insert v (Bind v expr)
  return v

getLinearFromEnvironment :: HasCallStack => Var -> LinearizationComputation TangentVar
getLinearFromEnvironment v = do
  primalBinding <- (>>= lift) (E.lookup v <$> use env)
  let tt = varType (bindVar primalBinding)
  varMap <- use (extra . nonLinearToLinear)
  case E.lookup v varMap of
    Right w -> return w
    Left _ -> do
      expr <- ParamShaxprF <$> (extra . nextLinearParameter <<+= 1)
      extra . environmentTypes %= (tt:)
      p <- newTangentBinding tt expr
      extra . nonLinearToLinear %= E.insert v p
      return p

getTangentForPrimal :: Var -> LinearizationComputation TangentVar
getTangentForPrimal v = use (extra . primalToTangent) >>= lift . E.lookup v

setTangentForPrimal :: Var -> TangentVar -> LinearizationComputation ()
setTangentForPrimal v dotv = cannotFail (extra . primalToTangent %= E.insert v dotv)


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
  let linearDef = Def {
    defName = "d" ++ defName def,
    defArgTys = defArgTys def ++ reverse (finalState ^. environmentTypes),
    defRet = returnedTangents,
    defBinds = E.toBindings (finalState ^. tangentBindings)
  }
  let eSize = length (finalState ^. environmentTypes)
  return $ LinearizedDefinition nonLinearDef' linearDef eSize

recordTangents :: (Binding -> LinearizationComputation (Var, TangentVar)) -> LinearMapper
recordTangents f v = do
  (w, dotw) <- f v
  setTangentForPrimal w dotw
  return w

linearizeBinding :: HasCallStack => Binding -> LinearizationComputation (Var, TangentVar)
linearizeBinding b@(Bind v@(Var _ t) (ShaxprF op args)) = do
  dargs <- mapM getTangentForPrimal args
  case (op, args, dargs) of
    (_, xs, dxs) | isLinearFunc op -> do
      -- If f(x) is linear, then f'(x)(dx) = f(dx).
      y <- newBind t (ShaxprF op xs)
      dy <- newTangentBinding t (ShaxprF op dxs)

      return (y, dy)      
    (Param k, [], []) -> do
      -- Parameters pass through unmodified. Note that while in the primal program
      -- we are reading an argument, in the linear program we are reading the
      -- tangent with respect to that argument. This is because the first n indices
      -- of the linear program's arguments correspond to the tangents of all n
      -- indices of the primal program's arguments.
      x <- newBind t (ParamShaxprF k)
      dx <- newTangentBinding t (ParamShaxprF k)
      return (x, dx)

    (UnaryPointwise Sin, [x], [dx]) -> do
      -- If f(x) = sin(x), then f'(x)(dx) = cos(x) * dx.
      sinx <- newBind t (SinShaxprF x)

      cosx <- newBind t (CosShaxprF x)
      cosx' <- getLinearFromEnvironment cosx
      sinx' <- newTangentBinding t (MulShaxprF cosx' dx)
  
      return (sinx, sinx')

    (UnaryPointwise Cos, [x], [dx]) -> do
      -- If f(x) = cos(x), then f'(x)(dx) = -sin(x) * dx.
      cosx <- newBind t (CosShaxprF x)

      sinx <- newBind t (SinShaxprF x)
      negsinx <- newBind t (NegateShaxprF sinx)

      negsinx' <- getLinearFromEnvironment negsinx
      cosx' <- newTangentBinding t (MulShaxprF negsinx' dx)

      return (cosx, cosx')
    (UnaryPointwise Exp, [x], [dx]) -> do
      -- If f(x) = e^x, then f'(x)(dx) = e^x * dx.
      expx <- newBind t (ExpShaxprF x)

      expx' <- getLinearFromEnvironment expx
      expdx <- newTangentBinding t (MulShaxprF expx' dx)

      return (expx, expdx)
    (BinaryPointwise Mul, [x, y], [dx, dy]) -> do
      -- If f(x, y) = x * y, then f'(x, y)(dx, dy) = y * dx + x * dy.
      -- By convention, we only put tangents on the right argument,
      -- while the left argument is a primal value, read from the environment.
      xy <- newBind t (MulShaxprF x y)

      x' <- getLinearFromEnvironment x
      y' <- getLinearFromEnvironment y
      x'dy <- newTangentBinding t (MulShaxprF x' dy)
      y'dx <- newTangentBinding t (MulShaxprF y' dx)

      xy' <- newTangentBinding t (AddShaxprF x'dy y'dx)

      return (xy, xy')
    (BinaryPointwise Div, [x, y], [dx, dy]) -> do
      -- If f(x, y) = x / y, then
      -- f'(x, y)(dx, dy) = dx/y - x dy / y^2.
      --                      = (dx * y - x * dy) / y^2
      --                      = (y * dx - x * dy) / y^2
      -- This is valid only when y != 0.
      xdivy <- newBind t (DivShaxprF x y)

      x' <- getLinearFromEnvironment x
      y' <- getLinearFromEnvironment y

      ydx <- newTangentBinding t (MulShaxprF y' dx)
      xdy <- newTangentBinding t (MulShaxprF x' dy)
      numerator <- newTangentBinding t (SubShaxprF ydx xdy)
      denominator <- newTangentBinding t (MulShaxprF y' y')
      xdivy' <- newTangentBinding t (DivShaxprF numerator denominator)

      return (xdivy, xdivy')
    (Constant k, [], []) -> do
      -- If f(x) = k, then f'(x_0)(dx) = 0.
      constant <- newBind t (ConstantShaxprF k)
      zero <- newTangentBinding t (ConstantShaxprF (zeroLike k))

      return (constant, zero)
    (DotGeneral dims, [x, y], [dx, dy]) -> do
      -- If uhhh.... Trust Me Bro(tm), this is the correct
      -- cotangent forwarding rule for DotGeneral.
      adotb <- newBind t (DotGeneralShaxprF dims x y)

      x' <- getLinearFromEnvironment x
      y' <- getLinearFromEnvironment y
      a <- newTangentBinding t (DotGeneralShaxprF dims x' dy)
      b <- newTangentBinding t (DotGeneralShaxprF dims dx y')
      adotb' <- newTangentBinding t (AddShaxprF a b)

      return (adotb, adotb')
    (BinaryPointwise Min, [x, y], [dx, dy]) -> do
      z <- newBind (varType x) (MinShaxprF x y)
      linearizeMinMax z x y dx dy

    (BinaryPointwise Max, [x, y], [dx, dy]) -> do
      z <- newBind (varType x) (MaxShaxprF x y)
      linearizeMinMax z x y dx dy

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

linearizeMinMax :: Var -> Var -> Var -> TangentVar -> TangentVar -> LinearizationComputation (Var, TangentVar)
linearizeMinMax z x y dx dy = do
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
  -- Note the mask `m` is computed in the primal program, while the multiplications and
  -- additions are computed in the linear program.
  -- From the typechecker, we know x and y are float tensors.
  let constType = varType x
      constShape = tyShape constType
      maskType = constType
      boolType = TensorType TBool constShape
      scalarType = TensorType (tyBase constType) []

  let scalarConstType = TensorType (tyBase constType) []
  zero <- newBind scalarType (ConstantShaxprF (fromFloatScalar 0.0))
  one <- newBind scalarType (ConstantShaxprF (fromFloatScalar 1.0))
  two <- newBind scalarType (ConstantShaxprF (fromFloatScalar 2.0))

  zeros <- newBind constType (BroadcastShaxprF [] constShape zero)
  ones <- newBind constType (BroadcastShaxprF [] constShape one)
  twos <- newBind constType (BroadcastShaxprF [] constShape two)

  topEq <- newBind boolType (EqShaxprF z x)
  top <- newBind constType (SelectShaxprF topEq ones zeros)

  bottomEq <- newBind boolType (EqShaxprF z y)
  bottom <- newBind constType (SelectShaxprF bottomEq twos ones)

  multiplier <- newBind maskType (DivShaxprF top bottom)
  negMultiplier <- newBind maskType (SubShaxprF ones multiplier)

  multiplier' <- getLinearFromEnvironment multiplier
  negMultiplier' <- getLinearFromEnvironment negMultiplier

  left <- newTangentBinding constType (MulShaxprF multiplier' dx)
  right <- newTangentBinding constType (MulShaxprF negMultiplier' dy)
  dz <- newTangentBinding constType (AddShaxprF left right)

  return (z, dz)