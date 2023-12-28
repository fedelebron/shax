module Environment (Env, empty, insert,
                   Environment.lookup, fromDefinition,
                   toBindings, nextName) where

import Types (VarName(..))
import Binding ( Binding(Binding) )
import qualified Data.Map as M
import Definition
import Error
import GHC.Stack
import Text.PrettyPrint.HughesPJClass (Pretty (..), pPrint, prettyShow)

-- A very thin wrapper aroud a map from variable names to something. This comes
-- in handy quite often, as we map a variable to its binding, or a variable to
-- a rename of that variable, or a variable to its cotangent summands.
newtype Env a = Env {fromMap :: M.Map VarName a} deriving (Show, Eq)

instance Pretty a => Pretty (Env a) where
  pPrint = pPrint . M.toList . fromMap

empty :: Env a
empty = Env M.empty

insert :: VarName -> a -> Env a -> Env a
insert k v = Env . M.insert k v . fromMap

lookup :: (Pretty a, HasCallStack) => VarName -> Env a -> Either Error a
lookup k env =
    maybe
        (Left $ Error ("Key " ++ show k ++ " not found in environment: " ++ prettyShow env) callStack)
        Right
        (M.lookup k (fromMap env))

-- Returns a mapping from each variable in the body to its definition. Note this
-- does not include (and indeed cannot include) procedure parameters.
fromDefinition :: Definition -> Env Binding
fromDefinition = foldr (uncurry insert . bindingToPair) empty . defBinds
  where
    bindingToPair b@(Binding a _) = (a, b)

toBindings :: Env Binding -> [Binding]
toBindings = M.elems . fromMap

-- Gives a variable name that isn't yet a key in this environment.
nextName :: Env a -> VarName
nextName = VarName . (+ 1) . maximum . (-1: ) .map unVarName . M.keys . fromMap