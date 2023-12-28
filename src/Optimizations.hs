{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Optimizations (foldConstants,
                      eliminateDeadCode,
                      eliminateCommonSubexpressions,
                      OptimizationPass(..),
                      applyPasses) where
import qualified Data.Map as M
import qualified Data.Set as S
import qualified BiMap as BM
import qualified Environment as E
import Data.Maybe (fromJust)
import Types
import Binding
import Definition
import Shaxpr
import Data.Fix
import Eval
import GHC.Generics (Generic)

import Test.QuickCheck.Arbitrary.Generic

import Test.QuickCheck

data OptimizationPass = DCE | CSE | ConstantFolding
  deriving (Show, Eq, Ord, Generic)
  deriving (Arbitrary) via GenericArbitrary OptimizationPass

applyPass :: OptimizationPass -> Definition -> Definition
applyPass DCE = eliminateDeadCode
applyPass ConstantFolding = foldConstants
applyPass CSE = eliminateCommonSubexpressions

applyPasses :: [OptimizationPass] -> Definition -> Definition
applyPasses = foldr (.) id . map applyPass

foldConstants :: Definition -> Definition
foldConstants (Definition name argTys binds rets) =
  let foldedDef = Definition name argTys newBinds rets
  in eliminateDeadCode foldedDef
  where
    newBinds = go E.empty binds
    go _ [] = []
    go env (b@(Binding v e@(ShaxprF mty op args)):bs)
      | Param k <- op = b:go env bs
      | Right args' <- mapM (`E.lookup` env) args
        = let res = evalShaxpr . Shaxpr . Fix $ ShaxprF mty op (map Fix args')
              c = ConstantShaxprF mty res
          in  Binding v c : go (E.insert v c env) bs
      | otherwise = b:go env bs
  
eliminateDeadCode :: Definition -> Definition
eliminateDeadCode (Definition name argTys binds rets) = Definition name argTys newBinds rets
  where
    newBinds = filter ((`S.member` allUsed) . bindLabel) binds
    allUsed = collectUsed (S.fromList rets) (reverse binds)
    collectUsed forwardUses [] = forwardUses
    collectUsed forwardUses ((Binding v (ShaxprF _ _ args)):bs)
      | S.member v forwardUses = let append = foldr (.) id (map S.insert args)
                                 in  collectUsed (append forwardUses) bs 
      | otherwise = collectUsed forwardUses bs

eliminateCommonSubexpressions :: Definition -> Definition
eliminateCommonSubexpressions (Definition name argTys binds rets) =
  Definition name argTys (reverse newBinds) newRets
  where
    (newBinds, renames) = go BM.empty M.empty [] binds
    newRets = map (fromJust . (`M.lookup` renames)) rets
    go _ renames newBinds [] = (newBinds, renames)
    go expressionNames renames newBinds (Binding v e:binds)
      | Just existingName <- BM.lookupKey e' expressionNames =
        go expressionNames (M.insert v existingName renames) newBinds binds
      | otherwise =
        let (v', expressionNames') = BM.insert e' expressionNames
            renames' = M.insert v v' renames
        in  go expressionNames' renames' (Binding v' e':newBinds) binds
      where
        e' = maybeRename <$> e
        maybeRename name | Just newName <- M.lookup name renames = newName
                         | otherwise = name


materializeBroadcasts :: Definition -> Definition
materializeBroadcasts = id