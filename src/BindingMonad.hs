{-# LANGUAGE TemplateHaskell #-}

module BindingMonad where

import qualified Environment as E
import qualified Data.Map as M
import Data.Either
import qualified BiMap as B
import Control.Lens hiding (op)
import Control.Monad.State
import Control.Monad.Except (throwError)
import Text.PrettyPrint.HughesPJClass (prettyShow)


import GHC.Stack


import Binding
import Definition
import Environment
import Types
import Error
import Shaxpr

type RenameMap = B.BiMap VarName VarName
data BindingMonadState a = BindingMonadState {
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
type BindingMapper a = Binding -> BindingMonadComputation a VarName

makeInitialState :: a -> BindingMonadState a
makeInitialState a = BindingMonadState {
    _env = E.empty,
    _remap = B.empty,
    _extra = a
  }

walkBindings :: a -> BindingMapper a -> Definition -> Either Error (Definition, a)
walkBindings a f def =
  let initialState = makeInitialState a
      makeBindings = mapM (wrapRenaming f) (defBinds def)
  in do
    (_, finalState) <- runStateT makeBindings initialState
    let remaps = finalState ^. remap
    let binds = toBindings (finalState ^. env)
    return (def {
      defName = defName def,
      defBinds = binds,
      defRet = map (`maybeRename` remaps) (defRet def)
    }, finalState ^. extra)

walkBindingsOrDie :: HasCallStack => a -> BindingMapper a -> Definition -> Definition
walkBindingsOrDie = (((fst . fromRight err) .) .) . walkBindings
  where
    err = error "Internal error: Failed to walk bindings."

maybeRename :: VarName -> RenameMap -> VarName
maybeRename x = B.lookupValWithDefault x x

wrapRenaming :: HasCallStack => BindingMapper a -> Binding -> BindingMonadComputation a Binding
wrapRenaming f (Binding v (ShaxprF mty op args)) = do
  currentRemap <- use remap
  let args' = map (`maybeRename` currentRemap) args
  v' <- f (Binding v (ShaxprF mty op args'))
  newEnv <- use env
  case E.lookup v' newEnv of
    Left _ -> throwError $ Error ("Binding mapper returned variable " ++ prettyShow v' ++ ", which is not in the modified environment: " ++ prettyShow newEnv) callStack
    Right b'' -> cannotFail $ do
      remap %= B.insertWithKey v v'
      return b''

freshVar :: BindingMonadComputation a VarName
freshVar = E.nextName <$> use env

newBind :: ShaxprF VarName -> BindingMonadComputation a VarName
newBind e = do
  v <- freshVar
  let b = Binding v e
  env %= E.insert v b
  return v

keepBind :: Binding -> BindingMonadComputation a VarName
keepBind = newBind . bindExpr

getBindingType :: HasCallStack => VarName -> BindingMonadComputation a TensorType
getBindingType v = do
  vBind <- (>>= lift) (E.lookup v <$> use env)
  case exprTy (bindExpr vBind) of
    Just t -> return t
    Nothing -> throwError (Error ("Failed to get type of variable " ++ prettyShow v) callStack)