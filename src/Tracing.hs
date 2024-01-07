{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeFamilyDependencies #-}
module Tracing where

import qualified Data.Map as M
import Text.PrettyPrint.HughesPJClass
import Control.Monad.Trans.State

import Types
import Shaxpr
import Data.Fix
import Tensor
import Definition

data HsTree a = Leaf a | Branches (M.Map String (HsTree a))
  deriving (Functor, Foldable, Traversable)

instance (Pretty a) => Pretty (HsTree a) where
  pPrint (Leaf x) = pPrint [x]
  pPrint (Branches m) = pPrint (M.elems m)

getLeaf :: HsTree a -> a
getLeaf (Leaf x) = x
getLeaf (Branches m) = error $ "Tried to get a leaf, but have branches: " ++ prettyShow (M.keys m)

getBranch :: HsTree a -> String -> HsTree a
getBranch (Branches m) k | Just v <- M.lookup k m = v
getBranch m k = error $ "Failed to get branch " ++ show k

foo :: Num a => HsTree a -> a -> a -> a
foo param0 x0 x1  = 
  getLeaf (getBranch param0 "left") * x0 + getLeaf (getBranch param0 "right") * x1

a = Branches (M.fromList [("left", Leaf (TensorType TFloat [])),
                          ("right", Leaf (TensorType TFloat []))])
b = Leaf (TensorType TFloat [])
d = Leaf (TensorType TFloat [])

type family ToShaxpr a = r | r -> a
type instance ToShaxpr TensorType = Shaxpr
type instance ToShaxpr (HsTree TensorType) = HsTree Shaxpr

traceHsTreeParam :: Int -> HsTree TensorType -> (HsTree Shaxpr -> b) -> (b, Int)
traceHsTreeParam k tree f = let (tree', k') = runState (traverse mapParam tree) k
                      in ((f tree'), k')
  where
    mapParam _ = do
      k <- get
      put (k + 1)
      return $ Shaxpr (Fix (ParamShaxprF k))

traceTensorParam :: Int -> HsTree TensorType -> (Shaxpr -> b) -> (b, Int)
traceTensorParam k (Leaf _) f = let p = Shaxpr (Fix (ParamShaxprF k))
                                in  (f p, k + 1)
traceTensorParam k _ _ = error "Function expected a Shaxpr, but provided param is not a Leaf."

class TraceableParam a where
  traceParam :: Int -> HsTree TensorType -> (a -> b) -> (b, Int)

instance TraceableParam (HsTree Shaxpr) where
  traceParam = traceHsTreeParam

instance TraceableParam Shaxpr where
  traceParam = traceTensorParam

class Traceable t where
  trace' :: [HsTree TensorType] -> Int -> t -> [Shaxpr]

instance Traceable Shaxpr where
  trace' _ _ = return

instance Traceable [Shaxpr] where
  trace' _  _ = id

instance (TraceableParam t, Traceable b) => Traceable (t -> b) where
  trace' (p:ps) k f = let (f', k') = traceParam k p f
                      in trace' ps k' f'

trace ps f = trace' ps 0 f

traceToDef :: (Traceable t) => String -> [HsTree TensorType] -> t -> Def VarName
traceToDef name ps f = let shaxprs = trace' ps 0 f
                           flatTypes = concatMap (foldr (:) []) ps
                       in  toDef name flatTypes shaxprs