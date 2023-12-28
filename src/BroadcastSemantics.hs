module BroadcastSemantics(module BroadcastSemantics) where

import Types
import Error
import GHC.Stack

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

-- Returns the broadcast_in_dim indices for the left and right operand.
-- In case a broadcast is not needed for an operand, Nothing is returned.

data BroadcastInDimResult = BroadcastInDimResult {
  leftDimIxs :: Maybe [Int],
  rightDimIxs :: Maybe [Int],
  resultShape :: Shape
} deriving Show

broadcastInDims :: HasCallStack => Shape -> Shape -> Either Error BroadcastInDimResult
broadcastInDims left right = do
  (l, r, c) <- go (reverse left) (reverse right) 0 [] [] []
  let l' = if c == left then Nothing else Just (reverse l)
      r' = if c == right then Nothing else Just (reverse r)
  return $ BroadcastInDimResult l' r' c
  where
    go [] [] _ l r c = Right (l, r, c)
    go [] (y:ys) i l r c = go [] ys (i + 1) l (i:r) (y:c)
    go (x:xs) [] i l r c = go xs [] (i + 1) (i:l) r (x:c)
    go (x:xs) (y:ys) i l r c | x == y || x == 1 || y == 1 = go xs ys (i + 1) (i:l) (i:r) (max x y:c)
                             | otherwise = Left $ Error ("Incompatible shapes for broadcast! " ++ show left ++ ", " ++ show right) callStack
