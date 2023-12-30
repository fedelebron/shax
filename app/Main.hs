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
import Linearize
import Optimizations
import Transpose
import Optimizations (lowerReductionsToSumsOfSlices)

f :: forall a. (HNP a, Ord a, Floating a) => a -> a -> a -> a
f a b c = let d = a + b
              e = transpose [1, 0] (cos d)
              h = sin d
              i = dot e h
          in i * max c 1.0
          --in i * max i c



showDef :: Int -> Definition -> String
showDef k = render . pPrintPrec (PrettyLevel k) 0

g :: forall a. (Num a, Floating a) => a -> a -> a
g x y = let z = x + y
        in  cos z * sin z

h :: forall a. HNP a => a -> a -> a
h x y = let z = dotGeneral (DotDimensionNumbers [1, 3] [2] [0] [3]) x y
        in  z

rightOrDie :: Show e => Either e a -> IO a
rightOrDie (Right x) = return x
rightOrDie (Left e) = error (show e)

main' :: [Shaxpr] -> [SomeArray] -> [SomeArray] -> [SomeArray] -> IO ()
main' hs p dx ct = do
    let def = toDefinition "g" (map someArrayType p) hs
    putStrLn "After tracing:"
    putStrLn (showDef 2 def)
    d' <- rightOrDie (inferTypes def)
    putStrLn "After type inference:"
    putStrLn (showDef 2 d')
    putStrLn "Evaluation:"
    print (evalDefinition d' p)

    let d'' = materializeBroadcasts (canonicalizeDotGeneral d')
    putStrLn "After BM + CDG: "
    putStrLn (showDef 2 d'')

    (v, dg) <- rightOrDie (linearize p d'')
    putStrLn "After linearizing:"
    putStrLn (showDef 2 dg)
    putStrLn "Evaluation of primal during linearization:"
    print v
    putStrLn "Evaluation of linearized:"
    print (evalDefinition dg dx)

    let csedg = lowerReductionsToSumsOfSlices
                . foldConstants 
                . eliminateCommonSubexpressions
                $ dg
    putStrLn "After LR + CF + CSE:"
    putStrLn (showDef 2 csedg)

    dgt <- rightOrDie (transposeDef csedg)
    putStrLn "Transposed linearized:"
    putStrLn (showDef 2 dgt)

    putStrLn "Evaluate transposed (at the dual of the linearization point)"
    print (evalDefinition dgt p)

main1 :: IO ()
main1 = let p = [FloatArray (D.fromList [] [1.5]),
                 FloatArray (D.fromList [] [-2.5])]
            dx = p
            ct = [undefined]    
        in main' (close (g @Shaxpr)) p dx ct

main2 :: IO ()
main2 = let x = FloatArray $ D.fromList [2, 3] [1.0 .. 6.0]
            y = FloatArray $ D.fromList [2, 3] [5.0 .. 10.0]
            z = FloatArray $ D.fromList [3, 3] (replicate 9 0)
            p = [x, y, z]
            dx = p
            ct = undefined
        in  main' (close (f @Shaxpr)) p dx ct

main3 :: IO ()
main3 = let p = [FloatArray (D.fromList [2, 4, 8, 2] (enumFromTo 1 (2*4*8*2))),
                 FloatArray (D.fromList [7, 1, 8, 2] (enumFromTo 1 (7*1*8*2)))]
            dx = p
            ct = undefined
        in  main' (close (h @Shaxpr)) p dx ct

ez :: forall a. (Floating a, Num a) => a -> a -> a
ez x y = let z = x  + y
         in  sin z

main4 :: IO ()
main4 = let x = FloatArray (D.fromList [2,6] [1 .. 12])
            p = [x, x]
            dx1 = FloatArray (D.fromList [2,6] [12, 11 .. 1])
            dx = [dx1, dx1]
            ct = [x]
        in  main' (close (ez @Shaxpr)) p dx ct


hard :: forall a. (HNP a, Floating a) => a -> a -> a
hard x0 x8 = let x1 = broadcast [1] [2, 6] x0
                 x2 = sin x1
                 x3 = cos x2
                 x4 = x2 + x3
                 x5 = exp x4
                 x6 = transpose [1, 0] x4
                 x7 = x5 `dot` x6
                 x9 = x7 + x8
                 x10 = reduceSum [1] x9
            in  x10

main5 :: IO ()
main5 = let x1 = FloatArray (D.fromList [6] [1 .. 6])
            x8 = FloatArray (D.fromList [2,2] [1 .. 4])
            ct = FloatArray (D.fromList [2,2] [1 .. 4])
        in  main' (close (hard @Shaxpr)) [x1, x8] [x1, x8] [ct]

main = main5