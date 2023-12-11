{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ExplicitForAll #-}

module Main (main) where


import qualified Data.Array.Dynamic as D
import Text.PrettyPrint.HughesPJClass (pPrintPrec, PrettyLevel(..), render)

import HNP
import Types
import Shaxpr
import Definition
import TypeInference
import Eval

f :: forall a. (HNP a, Ord a, Floating a) => a -> a -> a -> a
f a b c = let d = a + b
              e = transpose [1, 0] (cos d)
              h = sin d
              i = dot e h
          in i * max c 1.0

x :: SomeArray
x = FloatArray $ D.fromList [2, 3] [1.0 .. 6.0]
y :: SomeArray
y = FloatArray $ D.fromList [2, 3] [5.0 .. 10.0]
z :: SomeArray
z = FloatArray $ D.fromList [3, 3] (replicate 9 0)

showDef :: Int -> Definition -> String
showDef k = render . pPrintPrec (PrettyLevel k) 0

main :: IO ()
main = do
    let g = close (f @Shaxpr)
        d = toDefinition "g" [TensorType TFloat [2, 3],
                              TensorType TFloat [2, 3],
                              TensorType TFloat [3, 3]] g
        typedD = inferTypes d
    case typedD of
        Left err -> print err
        Right d' -> do
            putStrLn (showDef 2 d')
            print (evalDefinition d' [x, y, z])