module AD where

import Error
import Definition
import Linearize
import Transpose
import Optimizations

vjp :: Definition -> Either Error Definition
vjp def = do
  linearized <- linearize def
  transposed <- transposeDef linearized
  let d = zipDefinitions transposed
  return (eliminateDeadCode (eliminateCommonSubexpressions (forwardIdentities d)))

jvp :: Definition -> Either Error Definition
jvp def = do
  linearized <- linearize def
  let d = zipDefinitions linearized
  return (eliminateDeadCode (eliminateCommonSubexpressions (forwardIdentities d)))