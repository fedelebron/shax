{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeApplications #-}

module Testing where

import Types
import Shaxpr
import Data.Array.Dynamic as D
import Definition
import Eval
import TypeInference
import Optimizations
import Text.PrettyPrint.HughesPJClass


import Test.QuickCheck hiding (reason, Result)
import Test.QuickCheck.Property (Result, succeeded, failed, reason)


newtype F = F (forall a. (Num a, Floating a) => a -> a -> a)
instance Show F where
  show (F f) = let g = close (f @Shaxpr)
                   d = toDefinition "f" [TensorType TFloat [],
                                         TensorType TFloat []] g
               in  render (pPrintPrec (PrettyLevel 2) 0 d)

instance Arbitrary F where
  arbitrary = oneof $
    [  binNum (+)
    ,  binNum (*)
    ,  binFloat (/)
    -- We don't use ints because then the module may not typecheck... 
    -- , (\i -> F (\_ _ -> fromInteger i)) <$> arbitrary
    , (\r -> F (\_ _ -> fromRational r)) <$> arbitrary
    , pure (F (\x _ -> x))
    , pure (F (\_ x -> x))
    ] ++ oneFloat sin ++ oneFloat cos
    where
      oneFloat :: (forall b. Floating b => b -> b) -> [Gen F]
      oneFloat op = [pure (F (\x _ -> op x)),
                     pure (F (\_ y -> op y))]
      binNum :: (forall b. Num b => b -> b -> b) -> Gen F
      binNum op = (\f' g' -> F $ case f' of F f -> case g' of F g -> \x y -> f x y `op` g x y) <$> arbitrary <*> arbitrary
      binFloat :: (forall b. Floating b => b -> b -> b) -> Gen F
      binFloat op = (\f' g' -> F $ case f' of F f -> case g' of F g -> \x y -> f x y `op` g x y) <$> arbitrary <*> arbitrary

prop_tracingRespectsScalarEvaluation :: F -> Float -> Float -> [OptimizationPass] -> Result
prop_tracingRespectsScalarEvaluation (F f) x y opts =
  let groundTruth = f x y
      g = close (f @Shaxpr)
      d = toDefinition "g" [TensorType TFloat [],
                            TensorType TFloat []] g
      typedD = inferTypes d
  in case typedD of
    Left err -> failed { reason = "Failed typecheck! " ++ prettyShow d } 
    Right d' ->
      let x' = FloatArray (D.fromList [] [x])
          y' = FloatArray (D.fromList [] [y])
          d'' = applyPasses opts d'
          res = evalDefinition d'' [x', y']
      in case res of
        Left err -> failed { reason = "Failed evaluation! " ++ prettyShow d'' }
        Right [answer] ->
          if compareIgnoringNaNs (toFloatList answer) [groundTruth]
          then succeeded
          else failed { reason = "Different results! " ++ show answer ++ " vs " ++ show groundTruth }
           
compareIgnoringNaNs :: [Float] -> [Float] -> Bool
compareIgnoringNaNs xs ys = length xs == length ys
                            && all (uncurry equalOrNaN) (zip xs ys)
  where
    equalOrNaN :: Float -> Float -> Bool
    equalOrNaN x y = (x == y) || (isNaN x && isNaN y)