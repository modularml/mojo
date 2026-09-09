//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace HLCF;

/// True for `IfOp`, and for `ElifOp` only when there are no additional elif
/// arms (i.e. it is a plain if/else).
static bool isTwoArmIfLike(ElifOp op) { return op.getElifRegions().empty(); }
static bool isTwoArmIfLike(IfOp op) {
  return true; // no extra regions ever.
}

/// Erase all operations following the given OP in its parent region. The OP
/// itself does not get deleted.
static void eraseOpsAfter(PatternRewriter &rewriter, Operation *op) {
  Block *toErase =
      rewriter.splitBlock(op->getBlock(), op->getNextNode()->getIterator());
  rewriter.eraseBlock(toErase);
}

/// Replace the given op with a region. If the region ends with YieldOp then
/// uses of the results of the original op will be replaced with the
/// corresponding yielded values. Otherwise, the region must be ending with a
/// Return or a similar terminator - in that case we erase all the ops after the
/// original op as dead code.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-block region");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  if (isa<YieldOp>(terminator)) {
    // If the op block ends with yield, we rewire the values in the remaining of
    // the parent block to use the yielded values.
    rewriter.replaceOp(op, terminator->getOperands());
    rewriter.eraseOp(terminator);
  } else {
    // Delete all ops after the op - the block in the op ends with a terminator
    // that renders the remaining of the parent block dead.
    eraseOpsAfter(rewriter, op);
    rewriter.eraseOp(op);
  }
}

/// Fold identical / trivial then/else yields into the condition or a common
/// value. Both yield operand lists must match `op`'s result count.
static LogicalResult hoistIdenticalYieldResults(Operation *op, Value cond,
                                                ValueRange thenYieldOperands,
                                                ValueRange elseYieldOperands,
                                                PatternRewriter &rewriter) {
  if (thenYieldOperands.size() != op->getNumResults() ||
      elseYieldOperands.size() != op->getNumResults())
    return failure();

  bool changed = false;
  bool allChanged = true;
  for (auto [res, opndThen, opndElse] :
       llvm::zip(op->getResults(), thenYieldOperands, elseYieldOperands)) {
    // Replace 'if/elif cond { yield true } else { yield false }' with "cond".
    KGEN::SIMDAttr trueCond, falseCond;
    if (res.getType() == cond.getType() &&
        matchPattern(opndThen, m_Constant(&trueCond)) &&
        matchPattern(opndElse, m_Constant(&falseCond)) &&
        trueCond.getAsBool() == true && falseCond.getAsBool() == false) {
      rewriter.replaceAllUsesWith(res, cond);
      changed = true;
      continue;
    }

    if (opndThen == opndElse) {
      rewriter.replaceAllUsesWith(res, opndThen);
      changed = true;
    } else {
      allChanged = false;
    }
  }
  // Note: this is only called when there is no code in the if other than
  // the two yield instructions, so this is safe.
  if (allChanged) {
    rewriter.eraseOp(op);
    changed = true;
  }

  return changed ? success() : failure();
}

//===----------------------------------------------------------------------===//
// IfOp / ElifOp (2-arm) shared canonicalizations
//===----------------------------------------------------------------------===//

namespace {
/// If both branches have just a YieldOp and yield the same value, replace the
/// result with that value directly.
///
///   Before:
///      %a, %b = hlcf.if %cond {
///        hlcf.yield %c, %d
///      } else {
///        hlcf.yield %c, %d
///      }
///      return %a, %b
///
///   After:
///      return %c, %d
template <typename OpTy>
struct HoistYieldResults : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!isTwoArmIfLike(op))
      return failure();
    if (!isa<YieldOp>(op.getThenTerminator()) ||
        !isa<YieldOp>(op.getElseTerminator()))
      return failure();
    if (&op.getThenBlock().getOperations().front() != op.getThenTerminator())
      return failure();
    if (&op.getElseBlock().getOperations().front() != op.getElseTerminator())
      return failure();

    return hoistIdenticalYieldResults(
        op, op.getCond(), op.getThenTerminator()->getOperands(),
        op.getElseTerminator()->getOperands(), rewriter);
  }
};

/// If the condition is known at compile time, replace the op with the contents
/// of the corresponding branch. If the block we're inserting doesn't end with
/// YieldOp, operations following the original op will be discarded.
template <typename OpTy>
struct RemoveStaticCondition : public OpRewritePattern<OpTy> {
  RemoveStaticCondition(MLIRContext *ctx)
      : OpRewritePattern<OpTy>(ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!isTwoArmIfLike(op))
      return failure();

    KGEN::SIMDAttr condition;
    if (!matchPattern(op.getCond(), m_Constant(&condition)))
      return failure();

    Region &active =
        condition.getAsBool() ? op.getThenRegion() : op.getElseRegion();
    replaceOpWithRegion(rewriter, op, active);

    return success();
  }
};

/// If both branches end with return ops, replace the return ops with yield ops
/// and insert a new return op right after the if/elif. All subsequent ops in
/// the basic block are erased.
///
/// Before:                    After:
/// {                          {
///   ...                        ...
///   if %cond {                 %x = if %cond {
///      A                          A
///      return %a                  yield %a
///   } else {                   } else {
///      B                          B
///      return %b                  yield %b
///   }                          }
///   C                          return %x
/// }                          }
///
/// This also works with 'break' and 'continue'.
template <typename OpTy, typename TerminatorOpT>
struct HoistUnconditionalReturn : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!isTwoArmIfLike(op))
      return failure();

    auto thenTerm = dyn_cast<TerminatorOpT>(op.getThenTerminator());
    auto elseTerm = dyn_cast<TerminatorOpT>(op.getElseTerminator());
    if (!thenTerm || !elseTerm)
      return failure();
    // Both arms become yields of the new op's results; arities must match.
    if (thenTerm->getOperandTypes() != elseTerm->getOperandTypes())
      return failure();

    if constexpr (!std::is_same_v<TerminatorOpT, KGEN::ReturnOp>) {
      // Ensure both terminators branch to the same operation. It doesn't matter
      // if it's the immediate parent.
      if (thenTerm.getLabelAttr() != elseTerm.getLabelAttr())
        return failure();
    }
    DictionaryAttr attrs = thenTerm->getAttrDictionary();

    // Create a new op and put a return right after it. We have to create a new
    // op because the number of results might differ from the original.
    auto newOp =
        OpTy::create(rewriter, op.getLoc(),
                     op.getThenTerminator()->getOperandTypes(), op.getCond());
    TerminatorOpT::create(rewriter, op.getLoc(), TypeRange(),
                          newOp->getResults(), attrs.getValue());

    // Move the 'then' block from the original op to the new one and replace
    // the return terminator with yield.
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().begin());
    rewriter.setInsertionPoint(newOp.getThenTerminator());
    rewriter.replaceOpWithNewOp<YieldOp>(
        newOp.getThenTerminator(), newOp.getThenTerminator()->getOperands());

    // Same for the 'else' block.
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().begin());
    rewriter.setInsertionPoint(newOp.getElseTerminator());
    rewriter.replaceOpWithNewOp<YieldOp>(
        newOp.getElseTerminator(), newOp.getElseTerminator()->getOperands());

    // Erase the original op and all the ops below it.
    eraseOpsAfter(rewriter, op);
    rewriter.eraseOp(op);

    return success();
  }
};

/// If one of the branches is Return, then we can try pulling the code after
/// the op into the other branch and replace the return op with yield. This
/// allows us to hoist return to outer scopes, potentially enabling other
/// optimizations.
///
/// We can only perform this transformation if the op's basic block ends with
/// a return op - in that case it is legal to insert a return after the op,
/// which we want to do in this transformation.
///
/// Before:                    After:
/// {                          {
///   ...                        ...
///   %x = if %cond {            %x = if %cond {
///      %a = A                     %a = A
///      return %a                  yield %a
///   } else {                   } else {
///      %b = B                     %b = B
///      yield %b                   %t = C(%b)
///                                 yield %t
///   }                          }
///   %t = C(%x)                 return %x
///   return %t
/// }                          }
template <typename OpTy>
struct HoistConditionalReturn : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!isTwoArmIfLike(op))
      return failure();

    Block &parentBlock = op->getParentRegion()->front();
    Operation *parentBlockTerm = parentBlock.getTerminator();

    Operation *thenTerm = op.getThenTerminator();
    Operation *elseTerm = op.getElseTerminator();

    // On the other hand, if neither of them is return, then we also have
    // nothing to do.
    // TODO: This should also work for BreakOp and ContinueOp (provided thenTerm
    // and elseTerm are of the same type)
    if (!isa<KGEN::ReturnOp, BreakOp>(thenTerm) &&
        !isa<KGEN::ReturnOp, BreakOp>(elseTerm))
      return rewriter.notifyMatchFailure(
          op, "None of the branches ends with Return/Break");

    // One of the terminators is Return/Break, now make sure that the other one
    // is Yield.
    if (!isa<YieldOp>(thenTerm) && !isa<YieldOp>(elseTerm))
      return rewriter.notifyMatchFailure(
          op, "None of the branches ends with Yield");

    // Figure out which block contains Return and which one contains Yield.
    Operation *yieldTerm = nullptr, *returnTerm = nullptr;
    if (isa<YieldOp>(thenTerm)) {
      yieldTerm = thenTerm;
      returnTerm = elseTerm;
    } else {
      assert(isa<YieldOp>(elseTerm));
      yieldTerm = elseTerm;
      returnTerm = thenTerm;
    }

    // If the parent block doesn't end with return, then we cannot return after
    // the op, which is how we want to hoist return op from its branch. Hence,
    // bail out.  For a bit more generality, we handle nested IfOps/ElifOps too,
    // e.g.:
    //   %4 = hlcf.if %3 -> i1 {
    //     pop.store %arg1, %arg3 : !kgen.pointer<index>
    //     hlcf.yield %1 : i1
    //   } else {
    //     ...
    //     hlcf.if %6 {
    //       kgen.return %0 : i1
    //     } else {
    //       hlcf.yield
    //     }
    //     hlcf.yield %1 : i1
    //   }
    //   kgen.return %4 : i1
    SmallVector<Value> parentBlockTermOperands;
    Operation *actualParentTermOp = nullptr;
    std::function<void(Operation *)> findParentTermOp;
    findParentTermOp = [&](Operation *parentTerm) {
      // Base case, we find a return or break with some operands to return.
      if (isa<KGEN::ReturnOp, BreakOp>(parentTerm)) {
        actualParentTermOp = parentTerm;
        parentBlockTermOperands = parentTerm->getOperands();
        return;
      }

      // Otherwise if we have a yield op from a 2-arm if/elif, then we can keep
      // looking.
      auto yield = dyn_cast<YieldOp>(parentTerm);
      Operation *parentIfLike = yield ? yield->getParentOp() : nullptr;
      if (!yield || !(isa<IfOp>(parentIfLike) ||
                      (isa<ElifOp>(parentIfLike) &&
                       cast<ElifOp>(parentIfLike).getElifRegions().empty())))
        return; // Give up.
      // We're effectively going to hoist the return up the if tree.  We
      // can't do this if it will skip over other operations, so make sure
      // the return immediately follows the if.
      Operation &termAfterIf = *std::next(Block::iterator(parentIfLike));
      Operation *blockTerm = parentIfLike->getBlock()->getTerminator();
      if (!isa<KGEN::ReturnOp, BreakOp, YieldOp>(termAfterIf) &&
          // We can do this for the top level.
          parentTerm != yieldTerm)
        return; // Give up.
      // Search the parent for a return/break.
      findParentTermOp(blockTerm);
      if (!actualParentTermOp)
        return; // If recursion failed, give up.

      // If we succeeded, we may need to remap the returned values, because
      // they may be a consequence of the yield->if result values. For example
      // we remap %4 in the example above to %1.
      for (auto &retVal : parentBlockTermOperands) {
        OpResult retRes = dyn_cast<OpResult>(retVal);
        if (!retRes || retRes.getOwner() != parentIfLike)
          continue;
        if (retRes.getResultNumber() >= yield->getNumOperands()) {
          actualParentTermOp = nullptr;
          return;
        }
        retVal = yield.getOperand(retRes.getResultNumber());
      }
    };
    findParentTermOp(yieldTerm);

    // If we failed, give up.
    if (!actualParentTermOp)
      return rewriter.notifyMatchFailure(
          op, "Parent block doesn't end with Return/Break");

    if (isa<KGEN::ReturnOp>(actualParentTermOp)) {
      if (!isa<KGEN::ReturnOp>(returnTerm))
        return rewriter.notifyMatchFailure(
            op, "Parent block is Return, but exiting terminator is Break");
    } else {
      assert((isa<BreakOp>(actualParentTermOp)));
      if (!isa<BreakOp>(returnTerm))
        return rewriter.notifyMatchFailure(
            op, "Parent block is Break, but exiting terminator is Return");
      if (cast<BreakOp>(actualParentTermOp).getLabelAttr() !=
          cast<BreakOp>(returnTerm).getLabelAttr())
        return rewriter.notifyMatchFailure(
            op, "Break in the parent block's target is different from exiting "
                "terminator break's target");
    }

    // Both arms become yields of the new op's results. The exiting return/break
    // must already produce the same values as the parent terminator.
    if (returnTerm->getNumOperands() != actualParentTermOp->getNumOperands() ||
        !llvm::equal(returnTerm->getOperandTypes(),
                     actualParentTermOp->getOperandTypes()))
      return rewriter.notifyMatchFailure(
          op, "Exiting terminator and parent return/break have different "
              "operand types");
    if (yieldTerm->getNumOperands() != op.getNumResults())
      return rewriter.notifyMatchFailure(
          op, "Yield operand count doesn't match if/elif results");
    if (parentBlockTermOperands.size() != returnTerm->getNumOperands())
      return rewriter.notifyMatchFailure(
          op, "Remapped parent terminator operands don't match exiting "
              "terminator");

    // Now we know that we can transform this. Create a new op (we can't use
    // the original because we might need a different number of result values).
    auto newOp =
        OpTy::create(rewriter, op.getLoc(),
                     actualParentTermOp->getOperandTypes(), op.getCond());

    // Move the original 'then' and 'else' basic blocks into the new op.
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().begin());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().begin());

    // Move the ops from the parent block following the original op to a
    // separate block and then move that block into the 'yield' block in the new
    // if.
    Block *remainderBlock =
        rewriter.splitBlock(op->getBlock(), op->getNextNode()->getIterator());
    rewriter.inlineBlockBefore(remainderBlock, yieldTerm->getBlock(),
                               yieldTerm->getBlock()->end());

    // The remainder block used to use return values of the original op. We
    // now need to rewire that to values from the yield op.
    for (auto [idx, val] : llvm::enumerate(op->getResults()))
      rewriter.replaceAllUsesWith(val, yieldTerm->getOperand(idx));

    // And after that we can erase the yield op.
    rewriter.eraseOp(yieldTerm);

    // At this point our new op has its then and else block constructed, but
    // ending with returns. We need to replace them with yields and insert a
    // return after the new op.
    rewriter.setInsertionPointAfter(newOp);
    if (auto br = dyn_cast<BreakOp>(returnTerm)) {
      BreakOp::create(rewriter, op.getLoc(), newOp->getResults(),
                      br.getLabelAttr());
    } else {
      KGEN::ReturnOp::create(rewriter, op.getLoc(), newOp->getResults());
    }

    // The parent block terminator got sucked into the if, and is either a
    // return/break if it is one level, or a yield if it is deeper.  Replace it
    // with a yield of the operands we found.
    rewriter.setInsertionPoint(parentBlockTerm);
    rewriter.replaceOpWithNewOp<YieldOp>(parentBlockTerm,
                                         parentBlockTermOperands);
    rewriter.setInsertionPoint(returnTerm);
    rewriter.replaceOpWithNewOp<YieldOp>(returnTerm, returnTerm->getOperands());

    // Finally, erase the original op.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Remove unused results of the if/elif and any yields.
template <typename OpTy>
struct RemoveUnusedResults : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &b) const override {
    if (!isTwoArmIfLike(op))
      return failure();

    auto thenYield = dyn_cast<YieldOp>(op.getThenTerminator());
    auto elseYield = dyn_cast<YieldOp>(op.getElseTerminator());
    if ((thenYield && thenYield->getNumOperands() != op.getNumResults()) ||
        (elseYield && elseYield->getNumOperands() != op.getNumResults()))
      return failure();

    llvm::BitVector unused(op.getNumResults());
    SmallVector<Value> toReplace;
    for (auto [i, result] : llvm::enumerate(op.getResults())) {
      if (result.use_empty())
        unused.set(i);
      else
        toReplace.push_back(result);
    }

    if (unused.none())
      return b.notifyMatchFailure(op.getLoc(), "all results have uses");

    if (thenYield)
      b.modifyOpInPlace(thenYield, [&] { thenYield->eraseOperands(unused); });
    if (elseYield)
      b.modifyOpInPlace(elseYield, [&] { elseYield->eraseOperands(unused); });

    auto newOp = OpTy::create(b, op.getLoc(), TypeRange(ValueRange(toReplace)),
                              op.getCond());
    b.replaceAllUsesWith(toReplace, newOp.getResults());
    b.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                         newOp.getThenRegion().begin());
    b.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                         newOp.getElseRegion().begin());
    b.eraseOp(op);
    return success();
  }
};

template <typename OpTy>
static void addIfLikeCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *ctx) {
  results.add<RemoveStaticCondition<OpTy>,
              HoistUnconditionalReturn<OpTy, KGEN::ReturnOp>,
              HoistUnconditionalReturn<OpTy, HLCF::BreakOp>,
              HoistUnconditionalReturn<OpTy, HLCF::ContinueOp>,
              HoistConditionalReturn<OpTy>, HoistYieldResults<OpTy>,
              RemoveUnusedResults<OpTy>>(ctx);
}
} // namespace

void IfOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *ctx) {
  addIfLikeCanonicalizationPatterns<IfOp>(results, ctx);
}

void ElifOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *ctx) {
  addIfLikeCanonicalizationPatterns<ElifOp>(results, ctx);
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

namespace {
/// If the only operation in LoopOp is BreakOp, delete the loop.  Depending on
/// whether the target of the BreakOp is this or outer loop, we might have to
/// keep or delete it.
struct RemoveDeadLoop : OpRewritePattern<LoopOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp op, PatternRewriter &b) const override {
    Block &body = op.getBody().front();
    if (auto br = dyn_cast<BreakOp>(body.getOperations().front())) {
      StringAttr label = br.getLabelAttr();
      if (!label || label == op.getLabelAttr()) {
        SmallVector<Value> operandsToReplace = br.getOperands();
        for (auto [idx, value] : llvm::enumerate(operandsToReplace)) {
          if (auto arg = dyn_cast<BlockArgument>(value)) {
            if (arg.getOwner() != &op.getRegion().front())
              continue;
            // If the break's operand is a block argument of the loop op
            // that is about to be erased, use the loop operand instead.
            operandsToReplace[idx] = op.getOperand(arg.getArgNumber());
          }
        }
        b.replaceOp(op, operandsToReplace);
      } else {
        b.inlineBlockBefore(&body, op);
        eraseOpsAfter(b, op);
        b.eraseOp(op);
      }
      return success();
    }

    return failure();
  }
};

/// Remove unused results from a loop. This requires traversing the body to find
/// matching `break` operations, but the cost is paid only when there is a
/// match.
struct RemoveUnusedLoopResults : OpRewritePattern<LoopOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp loop,
                                PatternRewriter &b) const override {
    llvm::BitVector unused(loop.getNumResults());
    SmallVector<Value> toReplace;
    for (auto [i, result] : llvm::enumerate(loop.getResults())) {
      if (result.use_empty())
        unused.set(i);
      else
        toReplace.push_back(result);
    }

    if (unused.none())
      return b.notifyMatchFailure(loop.getLoc(), "all results have uses");

    // Find all matching break operations.
    StringAttr label = loop.getLabelAttr();
    loop.getBody().walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      // Walk over loops with the same label.
      if (auto inner = dyn_cast<LoopOp>(op);
          inner && inner.getLabelAttr() == label)
        return WalkResult::skip();

      // If this is a matching break, remove the unused operands.
      if (auto breakOp = dyn_cast<BreakOp>(op);
          breakOp && getParentLoop(breakOp, breakOp.getLabelAttr()) == loop)
        b.modifyOpInPlace(breakOp, [&] { breakOp->eraseOperands(unused); });

      return WalkResult::advance();
    });

    auto newLoop =
        LoopOp::create(b, loop.getLoc(), TypeRange(ValueRange(toReplace)),
                       loop.getOperands(), label, loop.getUnrollLevelAttr());
    b.replaceAllUsesWith(toReplace, newLoop.getResults());
    b.inlineRegionBefore(loop.getBody(), newLoop.getBody(),
                         newLoop.getBody().begin());
    b.eraseOp(loop);
    return success();
  }
};

/// Remove loop arguments that are unused.
struct RemoveUnusedLoopArgs : OpRewritePattern<LoopOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp loop,
                                PatternRewriter &b) const override {
    llvm::BitVector unused(loop.getNumOperands());
    for (BlockArgument arg : loop.getBody().getArguments())
      if (arg.use_empty())
        unused.set(arg.getArgNumber());
    if (unused.none())
      return b.notifyMatchFailure(loop.getLoc(), "no unused arguments");

    b.modifyOpInPlace(loop, [&] {
      loop->eraseOperands(unused);
      loop.getBody().front().eraseArguments(unused);
    });

    // Find all matching continue operations.
    StringAttr label = loop.getLabelAttr();
    loop.getBody().walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      // Walk over loops with the same label.
      if (auto inner = dyn_cast<LoopOp>(op);
          inner && inner.getLabelAttr() == label)
        return WalkResult::skip();

      // If this is a matching break, remove the unused operands.
      if (auto cont = dyn_cast<ContinueOp>(op);
          cont && getParentLoop(cont, cont.getLabelAttr()) == loop)
        b.modifyOpInPlace(cont, [&] { cont->eraseOperands(unused); });

      return WalkResult::advance();
    });

    return success();
  }
};
} // namespace

void LoopOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *ctx) {
  results.insert<RemoveDeadLoop, RemoveUnusedLoopResults, RemoveUnusedLoopArgs>(
      ctx);
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

static bool isPureOrReadOnly(Operation &op) {
  auto itf = dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!itf)
    return false;
  SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  itf.getEffects(effects);
  if (effects.empty())
    return true;
  return llvm::all_of(effects, [](auto &e) {
    return isa<mlir::MemoryEffects::Read>(e.getEffect());
  });
}

namespace {
/// Scan the body of a loop with no results up to a small number of consecutive
/// ops, checking if they are all pure or readonly. If this is the case, we know
/// the loop is overall a no-op.
struct RemoveNoopLoop : OpRewritePattern<ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op, PatternRewriter &b) const override {
    if (op.getNumResults())
      return b.notifyMatchFailure(op.getLoc(), "loop has results");

    constexpr unsigned numToScan = 5;
    for (auto [idx, op] :
         llvm::enumerate(op.getBody().front().without_terminator())) {
      if (idx > numToScan)
        return b.notifyMatchFailure(op.getLoc(), "body is too large");
      if (op.getNumRegions())
        return b.notifyMatchFailure(op.getLoc(), "body op with regions");
      if (!isPureOrReadOnly(op))
        return b.notifyMatchFailure(op.getLoc(), "not a pure or readonly op");
    }
    b.eraseOp(op);
    return success();
  }
};
} // namespace

void ForOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *ctx) {
  results.insert<RemoveNoopLoop>(ctx);
}
