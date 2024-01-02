module Optimizers where

import Data.Either
import Control.Monad.Trans.Except (except, ExceptT)
import Control.Monad.Trans.Class (lift)
import Text.PrettyPrint.HughesPJClass


import Linearize
import Eval
import Transpose
import Types
import Definition
import Error
import Eval (evalDefinition)

{-
data GradientDescentOpts = GradientDescentOpts {
  iters :: Int,
  alpha :: Float
}

gradientDescent :: GradientDescentOpts -> [SomeArray] -> Definition -> ExceptT Error IO ([SomeArray], [SomeArray])
gradientDescent opts initialParams def = do
  (initialVal, linear) <- except (linearize initialParams def)
  dft <- except (transposeDef linear)
  go n def dft [fromFloatScalar 1.0] initialParams
  where
    n = iters opts
    lr = fromFloatScalar (alpha opts)
    go 0 _ _ v p = except (Right (p, v))
    go i f dft v p = do
      lift (putStrLn ("Iteration " ++ show (n - i) ++ ": " ++ show v))
      diffs <- except (evalDefinition dft v)
      let newParams = zipWith (-) p (map (* lr) diffs)
      lift (putStrLn ("Old params: " ++ show p ++ ", diffs: " ++ show diffs))
      v' <- except (evalDefinition f newParams)
      go (i - 1) f dft  v' newParams

-}