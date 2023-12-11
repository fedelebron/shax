module BiMap(BiMap(..), lookupKey, lookupVal, insert, empty) where

import qualified Data.Map as M

data BiMap a = BiMap {
  to :: M.Map Int a,
  from :: M.Map a Int
} deriving Show

lookupKey :: Ord a => a -> BiMap a -> Maybe Int
lookupKey x m = M.lookup x (from m)

lookupVal :: Int -> BiMap a -> Maybe a
lookupVal x m = M.lookup x (to m)

insert :: Ord a => a -> BiMap a -> (Int, BiMap a)
insert x m = case lookupKey x m of
  Just k -> (k, m)
  Nothing -> let k = M.size (from m)
                 from' = M.insert x k (from m)
                 to' = M.insert k x (to m)
             in  (k, BiMap to' from')

empty :: BiMap a
empty = BiMap M.empty M.empty