{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE TypeApplications #-}

module Main (main) where

import           Control.Monad.Trans.Except
import qualified Data.Array.Dynamic as D
import           Definition
import           Eval
import           HNP
import           Linearize
import           Optimizations
import           Optimizers
import           Shaxpr
import           Text.PrettyPrint.HughesPJClass (Pretty, PrettyLevel(..)
                                               , pPrintPrec, render, prettyShow)
import           Tensor
import           Transpose
import           TypeInference
import           Types
import           Text.Printf (vFmt)
import Optimizers (gradientDescent)

showDef :: Pretty a => Int -> a -> String
showDef k = render . pPrintPrec (PrettyLevel k) 0

rightOrDie :: (Show e) => Either e a -> IO a
rightOrDie (Right x) = return x
rightOrDie (Left e) = error (show e)

medium :: forall a. (HNP a, Ord a, Floating a) => a -> a -> a
medium x0 x8 =
        let x1 = broadcast [1] [2, 6] x0
            x2 = sin x1
            x3 = cos x2
            x4 = x2 + x3
            x5 = exp x4
            x6 = transpose [1, 0] x4
            x7 = x5 `dot` x6
            x9 = x7 * x8
            x10 = x7 + x8
            x11 = min x9 x10
            x12 = max x9 x10
            x13 = x12 - x11
            x14 = reduceSum [0] x13
         in x14

linearizationDemo :: IO ()
linearizationDemo = do
  putStrLn "------------------\nLinearization demo\n------------------"
  let def = toDef "medium" [TensorType TFloat [6], TensorType TFloat [2, 2]] (close (medium @Shaxpr))
  putStrLn "After tracing:"
  putStrLn (showDef 2 def)
  typedDef <- rightOrDie (inferTypes def)
  putStrLn "After type inference:"
  putStrLn (showDef 2 typedDef)
  let typedDef' = lowerReductionsToSumsOfSlices (canonicalizeDotGeneral typedDef)
  putStrLn "After rewrites (canonicalizing `dot`s and lowering `reduce_sum` to pointwise `+` and `slice`):"
  putStrLn (showDef 2 typedDef')
  linearizedDef <- rightOrDie (linearize typedDef')
  putStrLn "Linearized definition(s):"
  putStrLn (showDef 2 linearizedDef)
  let x = [FloatTensor (D.fromList [6] [1 .. 6]), FloatTensor (D.fromList [2, 2] [1 .. 4])]
  (y, dy) <- rightOrDie (evalLinearizedDefinition linearizedDef x x)
  putStrLn "f(x):"
  putStrLn (prettyShow y)
  putStrLn "df(dx):"
  putStrLn (prettyShow dy)
  transposedDef <- rightOrDie (transposeDef linearizedDef)
  putStrLn "Transposed definition(s):"
  putStrLn (showDef 2 transposedDef)
  let ct = [FloatTensor (D.fromList [2] [5, 9])]
  (yy, dct) <- rightOrDie (evalLinearizedDefinition transposedDef x ct)
  putStrLn "f(x) (again):"
  putStrLn (prettyShow yy)
  putStrLn "dft(ct):"
  putStrLn (prettyShow dct)
  putStrLn "Ziped primal and transpose:"
  let vjp = forwardIdentities (zipDefinitions transposedDef)
  putStrLn (showDef 2 vjp)
  rightOrDie (checkTypes vjp)
  resultAndCotangent <- rightOrDie (evalDefinition vjp (x ++ ct))
  putStrLn "VJP:"
  putStrLn (prettyShow resultAndCotangent)


f :: forall a. Floating a => a -> a -> a
f x y = let z = x + y
        in sin z * cos z
descentDemo :: IO ()
descentDemo = do
  putStrLn "---------------------\nGradient descent demo\n---------------------"
  let def = toDef "f" [TensorType TFloat [], TensorType TFloat []] (close (f @Shaxpr))
  let p = [FloatTensor (D.fromList [] [0.5]), FloatTensor (D.fromList [] [-0.75])]
  typedDef <- rightOrDie (inferTypes def)
  putStrLn (showDef 2 typedDef)
  result <- runExceptT $ gradientDescent (GradientDescentOpts {iters = 39, alpha = 0.05}) p typedDef
  case result of
    Right (minParam, minVal) -> do
      putStrLn ("Minimum location: " ++ show minParam)
      putStrLn ("Minimum value: " ++ show minVal)
    Left err -> print err

main :: IO ()
main = descentDemo >> putStrLn "\n" >> linearizationDemo