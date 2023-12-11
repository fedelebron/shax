module BroadcastSemantics(module BroadcastSemantics) where

import Types

data BroadcastResult = BroadcastResult {
  leftReshape :: [Int],
  rightReshape :: [Int],
  commonShape :: Shape
} deriving Show

broadcastShapes :: Shape -> Shape -> BroadcastResult
broadcastShapes ash bsh =
  let na = length ash
      nb = length bsh
      ash' = replicate (nb - na) 1 ++ ash
      bsh' = replicate (na - nb) 1 ++ bsh
  in BroadcastResult ash' bsh' (zipWith max ash' bsh')