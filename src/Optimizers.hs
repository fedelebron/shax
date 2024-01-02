module Optimizers where

import Data.Either
import Control.Monad.Trans.Except (except, ExceptT)
import Control.Monad.Trans.Class (lift)
import Text.PrettyPrint.HughesPJClass
import GHC.Stack

import Linearize
import Eval
import Transpose
import Types
import Definition
import Binding
import Shaxpr
import Error
import Eval (evalDefinition)


data GradientDescentOpts = GradientDescentOpts {
  iters :: Int,
  alpha :: Float
}

gradientDescent :: HasCallStack => GradientDescentOpts -> [SomeArray] -> Definition -> ExceptT Error IO ([SomeArray], Float)
gradientDescent opts initialParams def = do
  assertTrue (length (defRet def) == 1) $ Error "Cannot minimize a non-scalar function." callStack
  let varRet = head (defRet def)
  let shRet = tyShape $ head [t | Binding v (ShaxprF (Just t) _ _) <- defBinds def, v == varRet]
  assertTrue (product shRet == 1) $ Error "Cannot minimize a non-scalar function." callStack

  linearized <- except (linearize def)
  transposed <- except (transposeDef linearized)
  go n transposed initialParams
  where
    n = iters opts
    lr = fromFloatScalar (alpha opts)
    go i dft p = do
      (val, delta) <- except (evalLinearizedDefinition dft p [fromFloatScalar 1.0])
      let valScalar = toFloatScalar (head val)
      lift (putStrLn ("Iteration " ++ show (n - i) ++ ": " ++ show valScalar))
      if i == 0
        then except (Right (p, valScalar))
        else do
          let newParams = zipWith (-) p (map (* lr) delta)
          go (i - 1) dft newParams