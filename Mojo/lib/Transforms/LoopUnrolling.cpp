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

#include "Mojo/ToolCommon/KGENPasses.h"

#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace M;
using namespace HLCF;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOOPUNROLLING
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
/// Unroll For-Loops with unrollFactor attributes:
/// - fully unroll a for-loop.
/// - unroll a for-loop with an unroll factor of a constant value .
/// This pass has to run after elaboration so that all parameter
/// expressions have elaborated with known values.
struct LoopUnrolling : impl::LoopUnrollingBase<LoopUnrolling> {
  using LoopUnrollingBase::LoopUnrollingBase;

  void runOnOperation() override;

private:
  SmallVector<ForOp> parentLoops;

  /// Loops to unroll in program order.
  SmallVector<ForOp> loopsToUnrollInOrder;

  /// Walk loops in program order.
  void walkLoopsPreorder(Operation *cur);

  /// Unroll a for-loop by a factor of unrollFactorN.
  static LogicalResult unrollForLoopN(ForOp loop, int64_t unrollFactorN);
};
} // namespace

void LoopUnrolling::walkLoopsPreorder(Operation *cur) {
  cur->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto loop = dyn_cast<ForOp>(op); loop && loop != cur) {
      // Recurse in nested loops.
      loopsToUnrollInOrder.push_back(loop);

      parentLoops.push_back(loop);
      walkLoopsPreorder(loop);
      parentLoops.pop_back();
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
}

/// Helper function to clone/move the operations to unroll a for loop.
static SmallVector<Value>
mergeOrInlineUnrollBlock(Block *dest, Block *original, int64_t urollFactorN,
                         bool copyLastIter, bool isUnrollByFactor,
                         bool hasParentFor, IRRewriter &rewriter,
                         Operation *inlineBeforeOp, ValueRange initValues) {
  SmallVector<Value> retValues = initValues;
  ForYieldOp newForYield;
  Region *scopeBody = original->getParent();

  for (int64_t i = 0; i < urollFactorN; ++i) {
    IRMapping map;
    Block *block = rewriter.createBlock(scopeBody);

    if (!copyLastIter && i + 1 == urollFactorN) {
      // Last iteration, move ops instead of cloning if required.

      for (BlockArgument arg : original->getArguments())
        rewriter.replaceAllUsesWith(
            arg, block->addArgument(arg.getType(), arg.getLoc()));

      Operation *prevOp = nullptr;
      for (Operation &op :
           llvm::make_early_inc_range(original->getOperations())) {
        if (auto yield = dyn_cast<ForYieldOp>(op)) {
          // Don't move last ForYieldOp.
          if (isUnrollByFactor) {
            // When unroll by a factor, make the new ForYieldOp's return segment
            // contain all the iteration args, including both of the original
            // retVals and other iterArgs. This helps to get the right iteration
            // args to the tail part of unrollFactorN is not divisible by trip
            // count.
            SmallVector<Value> newOperands = yield.getReturnValues();
            llvm::append_range(newOperands, yield.getOtherIterValues());
            rewriter.setInsertionPointAfter(prevOp);

            newForYield = ForYieldOp::create(rewriter, op.getLoc(),
                                             yield.getInductionVar(),
                                             newOperands, ValueRange{});
          } else {
            // Don't move last ForYieldOp.
            newForYield = yield;
          }
          continue;
        }

        // Move ops to the last block of inline.
        if (!prevOp)
          op.moveBefore(block, block->begin());
        else
          op.moveAfter(prevOp);

        prevOp = &op;
      }
    } else {
      for (BlockArgument arg : original->getArguments())
        map.map(arg, block->addArgument(arg.getType(), arg.getLoc()));

      for (Operation &op : original->getOperations()) {
        Operation *newOp = rewriter.clone(op, map);
        if (auto yield = dyn_cast<ForYieldOp>(newOp)) {
          if (isUnrollByFactor) {
            SmallVector<Value> newOperands = yield.getReturnValues();
            llvm::append_range(newOperands, yield.getOtherIterValues());
            newForYield = ForYieldOp::create(rewriter, op.getLoc(),
                                             yield.getInductionVar(),
                                             newOperands, SmallVector<Value>{});

            rewriter.eraseOp(newOp);
          } else {
            newForYield = yield;
          }
        }
      }
    }

    if (isUnrollByFactor) {
      // Add unrolled block before the loop.
      rewriter.mergeBlocks(block, dest, retValues);
    } else {
      // Add unrolled block before the loop.
      rewriter.inlineBlockBefore(block, inlineBeforeOp, retValues);
    }

    // Update next iteration's inputs
    retValues = newForYield.getOperands();

    // Erase ForYieldOp except for the last iteration.
    if (i + 1 != urollFactorN || !hasParentFor)
      rewriter.eraseOp(newForYield);
  }

  return retValues;
}

LogicalResult LoopUnrolling::unrollForLoopN(ForOp loop, int64_t unrollFactorN) {
  std::optional<int64_t> count = loop.getTripCount();
  if (!count)
    return failure();

  IRRewriter rewriter{OpBuilder(loop)};

  int64_t lowerBound = loop.getLowerBoundAsInt().value();
  int64_t step = loop.getStepAsInt().value();
  int64_t newStep = step * unrollFactorN;

  Block &originalBody = loop.getBody().front();

  if (count <= unrollFactorN) {
    // Fully unroll.
    SmallVector<Value> retValues = mergeOrInlineUnrollBlock(
        /*dest=*/nullptr, &originalBody, /*unrollFactorN*/ count.value(),
        /*copyLastIter=*/false, /*isUnrollByFactor*/ false,
        /*hasParentFor*/ false, rewriter, loop.getOperation(),
        loop.getIterArgs());

    // Replace the loop return value, which are first group of operands of
    // ForYieldOp.
    loop.replaceAllUsesWith(llvm::drop_begin(llvm::drop_end(
        retValues, retValues.size() - loop.getNumResults() - 1)));

    // Erase the original loop.
    rewriter.eraseOp(loop);

  } else {
    // Unroll by a factor.
    //
    // Example1: unroll the following ForOp (trip count 6) by 2
    // %idx5 = index.constant 7
    // %idx1 = index.constant 1
    // %1 = hlcf.for [7 to 1 step 1 sgt sub] (%arg0, %arg1, %arg2) -> index {
    //   %0 = index.sub %arg0, 1
    //   kgen.call @foo(%arg1, %arg2) : (index) -> ()
    //   hlcf.for.yield [induction_var (%0)] [retvals (%0)] [iterargs (%0)]
    // } {unrollLevel = #hlcf<unroll_level 2> }
    //
    // To a loop with trip count of 3 and a body of 2 unrolled original body.
    // Also change the result to include all iteration args except the induction
    // variable.
    //
    // %0:2 = hlcf.for [7 to 1 step 2 sgt sub] (%arg0, %arg1, %arg2)->index {
    //   %1 = index.sub %arg0, 1
    //   kgen.call @foo(%arg1, %arg2) : (index) -> ()
    //   %2 = index.sub %1, 1
    //   kgen.call @foo(%1, %1) : (index) -> ()
    //   hlcf.for.yield [induction_var (%2)] [retvals (%2, %2)] [iterargs ()]
    // }
    //
    // Example2: unroll the following ForOp (trip count 5) by 2 where 5 is not
    // divisible  by 2.
    // %idx5 = index.constant 5
    // %idx1 = index.constant 1
    // %1 = hlcf.for [5 to 1 step 1 sgt sub] (%arg0, %arg1, %arg2) -> index {
    //   %0 = index.sub %arg0, 1
    //   kgen.call @foo(%arg1, %arg2) : (index) -> ()
    //   hlcf.for.yield [induction_var (%0)] [retvals (%0)] [iterargs (%0)]
    // } {unrollLevel = #hlcf<unroll_level 2> }
    //
    // To
    // 1. A loop with trip count of 2 and a body of 2 unrolled original body.
    // Also change the result to include all iteration args except the induction
    // variable.
    // 2. Fully unroll the tail iterations.
    //
    // %0:2 = hlcf.for [5 to 2 step 2 sgt sub] (%arg0, %arg1, %arg2)->index {
    //   %1 = index.sub %arg0, 1
    //   kgen.call @foo(%arg1, %arg2) : (index) -> ()
    //   %2 = index.sub %1, 1
    //   kgen.call @foo(%1, %1) : (index) -> ()
    //   hlcf.for.yield [induction_var (%2)] [retvals (%2, %2)] [iterargs ()]
    // }
    //
    // kgen.call @foo(%0:0, %0:1) : (index) -> ()

    int64_t newTailLowerBound;
    switch (loop.getIndVarComputeType()) {
    case HLCF::ForLoopIndVarCompute::ADD:
      newTailLowerBound =
          lowerBound + (count.value() / unrollFactorN) * step * unrollFactorN;
      break;
    case HLCF::ForLoopIndVarCompute::SUB:
      newTailLowerBound =
          lowerBound - (count.value() / unrollFactorN) * step * unrollFactorN;
      break;
    }
    int64_t newUnrollNUpperBound = newTailLowerBound;

    rewriter.setInsertionPoint(loop);
    // Create the new ForOp with reordered operands.
    // create new step
    auto newStepOp = mlir::index::ConstantOp::create(
        rewriter, loop.getStep().getLoc(), newStep);

    auto newUnrollNUpperBoundOp = mlir::index::ConstantOp::create(
        rewriter, loop.getUpperBound().getLoc(), newUnrollNUpperBound);

    // New forOp returns all iterArgs except the induction variable (retVals and
    // otherArgs). This helps to pass all the iteration variables if there is a
    // tail part of the unrolling after this ForOp.
    auto forOp = HLCF::ForOp::create(
        rewriter, loop->getLoc(),
        SmallVector<Type>{llvm::drop_begin(loop.getIterArgs().getTypes())},
        loop.getLowerBound(), newUnrollNUpperBoundOp, newStepOp,
        loop.getIterArgs(), UnrollLevel::none(), loop.getCmpPredicateType(),
        loop.getIndVarComputeType());

    // Create the block for the new ForOp.
    Block *forBody = rewriter.createBlock(&forOp.getBody());

    SmallVector<Value> initValues;
    for (BlockArgument arg : originalBody.getArguments())
      initValues.push_back(forBody->addArgument(arg.getType(), arg.getLoc()));

    bool hasTailFor = (count.value() % unrollFactorN != 0);

    SmallVector<Value> retValues = mergeOrInlineUnrollBlock(
        /*dest*/ forBody, &originalBody, unrollFactorN, hasTailFor,
        /*isUnrollByFactor*/ true, /*hasParentFor*/ true, rewriter,
        /*inlineBeforeOp*/ nullptr, initValues);

    if (hasTailFor) {
      // Add tail iterations.
      rewriter.setInsertionPoint(loop);
      auto newLowerBoundV = mlir::index::ConstantOp::create(
          rewriter, loop->getLoc(), newTailLowerBound);
      SmallVector<Value> newOperands = {newLowerBoundV, loop.getUpperBound(),
                                        loop.getStep(), newLowerBoundV};
      llvm::append_range(newOperands, forOp.getResults());
      loop->setOperands(newOperands);
      return unrollForLoopN(loop, count.value() % unrollFactorN);
    } else {
      // Replace the loop return value, only take the original retVals instead
      // of the combination of retVals and iterArgs from the results of the new
      // ForOp.
      loop.replaceAllUsesWith(
          llvm::drop_end(forOp.getResults(),
                         forOp.getResults().size() - loop.getResults().size()));

      // Erase the original loop.
      rewriter.eraseOp(loop);
    }
  }

  return success();
}

void LoopUnrolling::runOnOperation() {
  parentLoops.clear();
  loopsToUnrollInOrder.clear();

  walkLoopsPreorder(getOperation());
  // Unroll loops from innermost to outermost.
  for (auto loop : llvm::reverse(loopsToUnrollInOrder)) {
    // Only process loops with known tripcounts.
    std::optional<int64_t> tripCount = loop.getTripCount();
    if (!tripCount)
      continue;
    // Try to unroll loops with an explicit unroll factor first.
    // TODO: We should be able to unroll loops with non-constant bounds by an
    // unroll factor.
    if (std::optional<int64_t> factor = loop.getUnrollFactorN()) {
      (void)unrollForLoopN(loop, *factor);
      continue;
    }
    // Otherwise, try to fully unroll the loop if it has fewer trips than the
    // threshold.
    if (loop.isFullUnroll() || (optimizationLevel >= 1 && *tripCount <= 1) ||
        optimizationLevel >= 4)
      (void)unrollForLoopN(loop, tripCount.value());
  }
}
