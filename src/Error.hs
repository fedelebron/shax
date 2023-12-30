{-# LANGUAGE FlexibleContexts #-}

module Error (Error (..), assertTrue, cannotFail, prependToErrors) where

import Control.Monad.Except (MonadError, catchError, throwError)
import Control.Monad.State
import Data.Functor.Identity
import GHC.Stack
import Text.PrettyPrint.HughesPJClass (Pretty (..), pPrint, prettyShow, text)

-- One can get a call stack by adding the type constraint HasCallStack to the
-- function. That brings in an implicit variable named callStack :: CallStack.
data Error = Error
    { description :: String,
      callstack :: CallStack
    }

instance Pretty Error where
    pPrint (Error msg trace) =
        text msg <> text (prettyCallStack trace)

instance Show Error where
    show = prettyShow

assertTrue :: (MonadError e m) => Bool -> e -> m ()
assertTrue x err = if x then pure () else throwError err

-- Embeds a stateful computation that can't fail, into a stateful computation
-- that can fail. For us, m is usually Either Error.
cannotFail :: (MonadError e m) => StateT s Identity b -> StateT s m b
cannotFail = mapStateT (pure . runIdentity)

prependToErrors :: (HasCallStack, MonadError Error m) => String -> m a -> m a
prependToErrors s c = c `catchError` handler
    where
        handler (Error s' cs) = throwError (Error (s ++ s') cs)