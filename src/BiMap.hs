{-# LANGUAGE FlexibleContexts #-}

module BiMap(BiMap(..), lookupKey, lookupVal, insert, empty) where

import qualified Data.Map as M
import Data.Coerce

data BiMap k v = BiMap {
  to :: M.Map k v,
  from :: M.Map v k
} deriving Show

lookupKey :: Ord v => v -> BiMap k v -> Maybe k
lookupKey x m = M.lookup x (from m)

lookupVal :: Ord k => k -> BiMap k v -> Maybe v
lookupVal x m = M.lookup x (to m)

insert :: (Coercible k Int, Ord k, Ord v) => v -> BiMap k v -> (k, BiMap k v)
insert x m = case lookupKey x m of
  Just k -> (k, m)
  Nothing -> let k = coerce (M.size (from m))
                 from' = M.insert x k (from m)
                 to' = M.insert k x (to m)
             in  (k, BiMap to' from')

empty :: BiMap k v
empty = BiMap M.empty M.empty