{-# LANGUAGE TemplateHaskell #-}

module BindingMonad where

import qualified Environment as E
import qualified Data.Map as M
import Control.Lens hiding (op)
import Control.Monad.State


import Binding
import Definition
import Types
import Error
import Shaxpr

type RenameMap = M.Map VarName VarName
data BindingMonadState a = BindingMonadState {
  -- The definition we are processing.
  _originalDefinition :: Definition,
  -- For convenience, we have the original bindings
  -- in an Environment.
  _originalBindings :: E.Env Binding,
  -- The bindings being produced.
  _env :: E.Env Binding,
  -- A mapping from old definition variable names to
  -- new definition variable names.
  _remap :: RenameMap,
  -- Extra metadata that the caller wants to have.
  _extra :: a  
}
makeLenses 'BindingMonadState

type BindingMonadComputation a r = StateT (BindingMonadState a) (Either Error) r
-- A binding map that has extra state of type a.
type BindingMapper a = Binding -> BindingMonadComputation a Binding

makeInitialState :: a -> Definition -> BindingMonadState a
makeInitialState a def =
  let env = E.fromDefinition def
  in BindingMonadState {
    _originalDefinition = def,
    _originalBindings = env,
    _env = E.empty,
    _remap = M.empty,
    _extra = a
  }

walkBindings :: a -> BindingMapper a -> Definition -> Either Error Definition
walkBindings a f def =
  let initialState = makeInitialState a def
      makeBindings = mapM (wrapRenaming f) (defBinds def)
  in do
    (binds, state) <- runStateT makeBindings initialState
    let remaps = state ^. remap
    return def {
      defBinds = binds,
      defRet = map (`maybeRename` remaps) (defRet def)
    }

maybeRename :: VarName -> RenameMap -> VarName
maybeRename x = M.findWithDefault x x

wrapRenaming :: BindingMapper a -> BindingMapper a
wrapRenaming f b@(Binding v (ShaxprF mty op args)) = do
  currentRemap <- use remap
  let args' = map (`maybeRename` currentRemap) args
      b' = Binding v (ShaxprF mty op args')
  b''@(Binding v' (ShaxprF mty op args')) <- f b'
  remap %= M.insert v v'
  env %= E.insert v' b''
  return b''

freshVar :: BindingMonadComputation a VarName
freshVar = E.nextName <$> use env

newBind :: ShaxprF VarName -> BindingMonadComputation a VarName
newBind e = do
  v <- freshVar
  let b = Binding v e
  env %= E.insert v b
  return v