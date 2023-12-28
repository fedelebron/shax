{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Optimizations (foldConstants,
                      eliminateDeadCode,
                      eliminateCommonSubexpressions,
                      materializeBroadcasts,
                      OptimizationPass(..),
                      applyPasses) where
import qualified Data.Map as M
import qualified Data.Set as S
import qualified BiMap as BM
import qualified Environment as E
import Data.List ((\\))
import Data.Either (fromRight)
import Data.Maybe (fromJust)
import Text.PrettyPrint.HughesPJClass (prettyShow)

import GHC.Stack
import Control.Monad.State
import Control.Monad.Except (throwError)

import Control.Lens hiding (op)
import BroadcastSemantics

import Types
import Binding
import Definition
import Shaxpr
import Data.Fix
import Eval
import Error
import GHC.Generics (Generic)

import Test.QuickCheck.Arbitrary.Generic

import Test.QuickCheck
import BindingMonad

data OptimizationPass = DCE
                        | CSE
                        | ConstantFolding
                        | MaterializeBroadcasts
                        | CanonicalizeDotGeneral
  deriving (Show, Eq, Ord, Generic)
  deriving (Arbitrary) via GenericArbitrary OptimizationPass

applyPass :: OptimizationPass -> Definition -> Definition
applyPass DCE = eliminateDeadCode
applyPass ConstantFolding = foldConstants
applyPass CSE = eliminateCommonSubexpressions
applyPass MaterializeBroadcasts = materializeBroadcasts
applyPass CanonicalizeDotGeneral = canonicalizeDotGeneral

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


materializeBroadcasts :: HasCallStack => Definition -> Definition
materializeBroadcasts = walkBindingsOrDie () broadcaster
  where
    broadcaster :: BindingMapper ()
    broadcaster b@(Binding _ (ShaxprF mte op args)) =
      case (op, args) of
        (BinaryPointwise bop, [x, y]) -> do
          ee <- use env
          mtx <- exprTy . bindExpr <$> lift (E.lookup x ee)
          mty <- exprTy . bindExpr <$> lift (E.lookup y ee)
          case (mtx, mty) of
            (Just tx, Just ty) ->
              if tx == ty then keepBind b
              else do
                BroadcastInDimResult ixx ixy common <- lift $ broadcastInDims (tyShape tx) (tyShape ty)
                x' <- case ixx of
                        Nothing -> return x
                        Just s -> newBind (BroadcastShaxprF mte s common x)
                y' <- case ixy of
                        Nothing -> return y
                        Just s -> newBind (BroadcastShaxprF mte s common y)
                newBind (ShaxprF mte op [x', y'])
            _ -> throwError (Error ("Failed to get type of operands in " ++ prettyShow b) callStack)
        _ -> keepBind b

data DimOrder = LHS | RHS deriving Show
canonicalizeDotGeneral :: Definition -> Definition
canonicalizeDotGeneral = walkBindingsOrDie () canonicalizer
  where
    canonicalizer :: BindingMapper ()
    canonicalizer b@(Binding _ (DotGeneralShaxprF ty dims x y)) = do
      currentEnv <- use env
      shx <- (tyShape . fromJust . bindType) <$> lift (E.lookup x currentEnv)
      shy <- (tyShape . fromJust . bindType) <$> lift (E.lookup y currentEnv)
      let DotDimensionNumbers lhsContractingIxs rhsContractingIxs lhsBatchIxs rhsBatchIxs = dims
          lhsNonContractingIxs = getNonContracting shx lhsContractingIxs lhsBatchIxs
          rhsNonContractingIxs = getNonContracting shy rhsContractingIxs rhsBatchIxs
      x' <- transposeAndReshapeForMatmul LHS lhsContractingIxs lhsNonContractingIxs lhsBatchIxs shx x
      y' <- transposeAndReshapeForMatmul RHS rhsContractingIxs rhsNonContractingIxs rhsBatchIxs shy y
      undefined
    canonicalizer x = keepBind x
    transposeAndReshapeForMatmul :: DimOrder -> DimIxs -> DimIxs -> DimIxs -> Shape -> VarName -> BindingMonadComputation () VarName
    transposeAndReshapeForMatmul dimOrder contracting nonContracting batch shx x = do
        let permParts = case dimOrder of
                          LHS -> [batch, nonContracting, contracting]
                          RHS -> [batch, contracting, nonContracting]
            perm = concat permParts
            permutedShape = map (shx !!) perm
            newShape = map (product . map (shx !!)) permParts
        currentEnv <- use env
        xBind <- lift (E.lookup x currentEnv)
        let bt = tyBase (fromJust (bindType xBind))
        x' <- newBind (TransposeShaxprF (Just (TensorType bt permutedShape)) perm x)
        newBind (ReshapeShaxprF (Just (TensorType bt newShape)) newShape x')
    getNonContracting sh contracting batch =
      let allDims = [0 .. length sh - 1]
      in  allDims \\ (batch ++ contracting)
