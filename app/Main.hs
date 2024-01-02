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
import           Transpose
import           TypeInference
import           Types
import           Text.Printf (vFmt)

showDef :: Pretty a => Int -> a -> String
showDef k = render . pPrintPrec (PrettyLevel k) 0

rightOrDie :: (Show e) => Either e a -> IO a
rightOrDie (Right x) = return x
rightOrDie (Left e) = error (show e)

{-
f :: forall a. (HNP a, Ord a, Floating a) => a -> a -> a -> a
f a b c =
        let d = a + b
            e = transpose [1, 0] (cos d)
            h = sin d
            i = dot e h
         in i * max c 1.0

-- in i * max i c



g :: forall a. (Num a, Floating a) => a -> a -> a
g x y =
        let z = x + y
         in cos z * sin z

h :: forall a. (HNP a) => a -> a -> a
h x y =
        let z = dotGeneral (DotDimensionNumbers [1, 3] [2] [0] [3]) x y
         in z


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

        d'' <- rightOrDie (inferTypes (materializeBroadcasts (canonicalizeDotGeneral d')))
        putStrLn "After BM + CDG: "
        putStrLn (showDef 2 d'')

        (v, dg) <- rightOrDie (linearize p d'')
        dg' <- rightOrDie (inferTypes dg)
        putStrLn "After linearizing:"
        putStrLn (showDef 2 dg')
        putStrLn "Evaluation of primal during linearization:"
        print v
        putStrLn "Evaluation of linearized:"
        print (evalDefinition dg' dx)

        let csedg =
                lowerReductionsToSumsOfSlices
                        . foldConstants
                        . eliminateCommonSubexpressions
                        $ dg'
        putStrLn "After LR + CF + CSE:"
        putStrLn (showDef 2 csedg)
        csedg' <- rightOrDie (inferTypes csedg)

        dgt <- eliminateDeadCode <$> rightOrDie (transposeDef csedg')
        putStrLn "Transposed linearized:"
        putStrLn (showDef 2 dgt)
        dgt' <- rightOrDie (inferTypes dgt)

        putStrLn "Evaluate transposed at cotangent:"
        print (evalDefinition dgt ct)

main1 :: IO ()
main1 =
        let p =
                [ FloatArray (D.fromList [] [1.5]),
                  FloatArray (D.fromList [] [-2.5])
                ]
            dx = p
            ct = [undefined]
         in main' (close (g @Shaxpr)) p dx ct

main2 :: IO ()
main2 =
        let x = FloatArray $ D.fromList [2, 3] [1.0 .. 6.0]
            y = FloatArray $ D.fromList [2, 3] [5.0 .. 10.0]
            z = FloatArray $ D.fromList [3, 3] (replicate 9 0)
            p = [x, y, z]
            dx = p
            ct = undefined
         in main' (close (f @Shaxpr)) p dx ct

main3 :: IO ()
main3 =
        let p =
                [ FloatArray (D.fromList [2, 4, 8, 2] (enumFromTo 1 (2 * 4 * 8 * 2))),
                  FloatArray (D.fromList [7, 1, 8, 2] (enumFromTo 1 (7 * 1 * 8 * 2)))
                ]
            dx = p
            ct = undefined
         in main' (close (h @Shaxpr)) p dx ct

ez :: forall a. (Floating a, Num a) => a -> a -> a
ez x y =
        let z = x + y
         in sin z

main4 :: IO ()
main4 =
        let x = FloatArray (D.fromList [2, 6] [1 .. 12])
            p = [x, x]
            dx1 = FloatArray (D.fromList [2, 6] [12, 11 .. 1])
            dx = [dx1, dx1]
            ct = [x]
         in main' (close (ez @Shaxpr)) p dx ct
-}
hard :: forall a. (HNP a, Floating a) => a -> a -> a
hard x0 x8 =
        let x1 = broadcast [1] [2, 6] x0
            x2 = sin x1
            x3 = cos x2
            x4 = x2 + x3
            x5 = exp x4
            x6 = transpose [1, 0] x4
            x7 = x5 `dot` x6
            x9 = x7 + x8
            x10 = reduceSum [1] x9
         in x10
{-
main5 :: IO ()
main5 =
        let x1 = FloatArray (D.fromList [6] [1 .. 6])
            x8 = FloatArray (D.fromList [2, 2] [1 .. 4])
            ct = FloatArray (D.fromList [2] [1 .. 2])
         in main' (close (hard @Shaxpr)) [x1, x8] [x1, x8] [ct]

quadratic :: forall a. (Fractional a) => a -> a -> a
quadratic x y = x * y + 3.0

descentDemo :: IO ()
descentDemo = do
        let p = FloatArray (D.fromList [] [0.0])
            def = toDefinition "quadratic" [TensorType TFloat []] (close (quadratic @Shaxpr))
        typedDef <- rightOrDie (inferTypes def)
        putStrLn (showDef 2 typedDef)

        result <- runExceptT $ gradientDescent (GradientDescentOpts {iters = 20, alpha = 0.05}) [p] typedDef
        case result of
                Right (minParam, minVal) -> do
                        print ("Minimum location: " ++ show minParam)
                        print ("Minimum value: " ++ show minVal)
                Left err -> print err

ezpz :: forall a. Floating a => a -> a
ezpz u = let v = sin u
         in  -v

main6 :: IO ()
main6 = do
        let def = toDefinition "g" [TensorType TFloat []] (close (ezpz @Shaxpr))
        putStrLn "After tracing:"
        putStrLn (showDef 2 def)
        d' <- rightOrDie (inferTypes def)
        putStrLn "After type inference:"
        putStrLn (showDef 2 d')

main :: IO ()
main = main6
-}
f :: forall a. Floating a => a -> a -> a
f x y = let z = x + y
        in sin z * cos z

medium :: forall a. (HNP a, Floating a) => a -> a -> a
medium x0 x8 =
        let x1 = broadcast [1] [2, 6] x0
            x2 = sin x1
            x3 = cos x2
            x4 = x2 + x3
            x5 = exp x4
            x6 = transpose [1, 0] x4
            x7 = x5 `dot` x6
            x9 = x7 * x8
         in x9

main :: IO ()
main = do
  let def = toDefinition "medium" [TensorType TFloat [6], TensorType TFloat [2, 2]] (close (medium @Shaxpr))
  putStrLn "After tracing:"
  putStrLn (showDef 2 def)
  typedDef <- rightOrDie (inferTypes def)
  putStrLn "After type inference:"
  putStrLn (showDef 2 typedDef)
  let typedDef' = canonicalizeDotGeneral typedDef
  linearizedDef <- rightOrDie (linearize typedDef')
  putStrLn "Linearized definition(s):"
  putStrLn (showDef 2 linearizedDef)
  let x = [FloatArray (D.fromList [6] [1 .. 6]), FloatArray (D.fromList [2, 2] [1 .. 4])]
  (y, dy) <- rightOrDie (evalLinearizedDefinition linearizedDef x x)
  putStrLn "f(x):"
  putStrLn (prettyShow y)
  putStrLn "df(dx):"
  putStrLn (prettyShow dy)
  transposedDef <- rightOrDie (transposeDef linearizedDef)
  putStrLn "Transposed definition(s):"
  putStrLn (showDef 2 transposedDef)
  let ct = [FloatArray (D.fromList [2, 2] [2 .. 5])]
  (yy, dct) <- rightOrDie (evalLinearizedDefinition transposedDef x ct)
  putStrLn "f(x) (again):"
  putStrLn (prettyShow yy)
  putStrLn "dft(ct):"
  putStrLn (prettyShow dct)