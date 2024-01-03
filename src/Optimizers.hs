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
import Bind
import Shaxpr
import Error
import Tensor

data GradientDescentOpts = GradientDescentOpts {
  iters :: Int,
  alpha :: Float
}

gradientDescent :: HasCallStack => GradientDescentOpts -> [Tensor] -> Definition -> ExceptT Error IO ([Tensor], Float)
gradientDescent opts initialParams def = do
  assertTrue (length (defRet def) == 1) $ Error "Cannot minimize a non-scalar function." callStack
  let shRet = tyShape . varType . head . defRet $ def
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
      lift (putStrLn ("Iteration " ++ show (n - i + 1) ++ ": " ++ show valScalar))
      if i == 0
        then return (p, valScalar)
        else do
          let newParams = zipWith (-) p (map (* lr) delta)
          go (i - 1) dft newParams