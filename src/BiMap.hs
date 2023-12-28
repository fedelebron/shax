{-# LANGUAGE FlexibleContexts #-}

module BiMap(BiMap(..), lookupKey, lookupVal, lookupValWithDefault, insert, insertWithKey, empty) where

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

lookupValWithDefault :: Ord k => v -> k -> BiMap k v -> v
lookupValWithDefault v k m = M.findWithDefault v k (to m)

insert :: (Coercible k Int, Ord k, Ord v) => v -> BiMap k v -> (k, BiMap k v)
insert x m = case lookupKey x m of
  Just k -> (k, m)
  Nothing -> let k = coerce (M.size (from m))
                 from' = M.insert x k (from m)
                 to' = M.insert k x (to m)
             in  (k, BiMap to' from')

insertWithKey :: (Ord k, Ord v) => k -> v -> BiMap k v -> BiMap k v
insertWithKey k v m = case (lookupKey v m, lookupVal k m) of
  (Nothing, Nothing) -> let from' = M.insert v k (from m)
                            to' = M.insert k v (to m)
                        in  BiMap to' from'
  _ -> m

empty :: BiMap k v
empty = BiMap M.empty M.empty