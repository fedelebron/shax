module Main(main) where

import Testing
import Test.QuickCheck

main :: IO ()
main = do
  quickCheckWith stdArgs { maxSuccess = 25000 } prop_tracingRespectsScalarEvaluation