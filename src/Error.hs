module Error (Error (..), assertTrue, cannotFail) where

import Control.Monad.Except (MonadError, throwError)
import GHC.Stack
import Text.PrettyPrint.HughesPJClass (Pretty (..), pPrint, text)
import Data.Functor.Identity
import Control.Monad.State

-- One can get a call stack by adding the type constraint HasCallStack to the
-- function. That brings in an implicit variable named callStack :: CallStack.
data Error = Error
    { description :: String
    , callstack :: CallStack
    }
    deriving (Show)

instance Pretty Error where
    pPrint (Error msg trace) =
        text ("Message: " ++ msg)
            <> text ("\nTrace: " ++ prettyCallStack trace)

assertTrue :: MonadError e m => Bool -> e -> m ()
assertTrue x err = if x then pure () else throwError err

-- Embeds a stateful computation that can't fail, into a stateful computation
-- that can fail. For us, m is usually Either Error.
cannotFail :: MonadError e m => StateT s Identity b -> StateT s m b
cannotFail = mapStateT (pure . runIdentity)