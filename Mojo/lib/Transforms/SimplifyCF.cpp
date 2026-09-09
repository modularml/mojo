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
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace KGEN;
using namespace HLCF;

namespace M::KGEN {
#define GEN_PASS_DEF_SIMPLIFYCF
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct SimplifyCF : impl::SimplifyCFBase<SimplifyCF> {
  void runOnOperation() override;

private:
  /// Remove the loop if it is trivial. Return true if it was removed.
  bool tryRemovingLoop(LoopOp loop);

  void findTargetLoop(Operation *op, StringAttr label);
  void walkPreorder(Region &region);

  /// Function loops in the pre-order.
  std::vector<LoopOp> loopsInOrder;

  /// Outer loops stack for the walk.
  SmallVector<LoopOp> parentLoops;

  /// Number of break or continue ops that lead to the current loop (breaks and
  /// continues in inner loops don't count).
  DenseMap<LoopOp, int> jumpsCount;

  /// The try operations in pre-order.
  std::vector<LIT::TryOp> triesInOrder;

  /// Try operations that can be elided.
  DenseSet<LIT::TryOp> elidableTries;

  /// The last seen try operation in pre-order traversal.
  LIT::TryOp lastTry;
};
} // namespace

//===----------------------------------------------------------------------===//
// CFG Analysis
//===----------------------------------------------------------------------===//

void SimplifyCF::findTargetLoop(Operation *op, StringAttr label) {
  assert((isa<ContinueOp, BreakOp>(op)));
  for (LoopOp loop : llvm::reverse(parentLoops)) {
    if (isMatchingLoop(loop, label)) {
      ++jumpsCount[loop];
      return;
    }
  }
  llvm_unreachable("no parent loop?");
}

void SimplifyCF::walkPreorder(Region &region) {
  region.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto c = dyn_cast<ContinueOp>(op))
      findTargetLoop(op, c.getLabelAttr());
    if (auto br = dyn_cast<BreakOp>(op))
      findTargetLoop(op, br.getLabelAttr());

    // Keep walking until we see another loop, then we recurse.
    if (auto loop = dyn_cast<LoopOp>(op)) {
      loopsInOrder.push_back(loop);
      parentLoops.push_back(loop);
      walkPreorder(loop.getBody());
      parentLoops.pop_back();
      return WalkResult::skip();
    }

    // Save a try operation if we see one.
    if (auto tryOp = dyn_cast<LIT::TryOp>(op)) {
      triesInOrder.push_back(tryOp);
      elidableTries.insert(tryOp);
      {
        // Set the contextual try for the 'try' region and then visit it.
        llvm::SaveAndRestore restore(lastTry);
        lastTry = tryOp;
        walkPreorder(tryOp.getTryRegion());
      }
      // Visit the rest of the regions.
      walkPreorder(tryOp.getExceptRegion());
      walkPreorder(tryOp.getElseRegion());
      return WalkResult::skip();
    }

    // A raise means we cannot elide the most recent try.
    if (auto raise = dyn_cast<LIT::TryRaiseOp>(op)) {
      assert(lastTry && "no contextual try?");
      elidableTries.erase(lastTry);
    }

    return WalkResult::advance();
  });
}

//===----------------------------------------------------------------------===//
// Erasing Ops
//===----------------------------------------------------------------------===//

/// If the loop body ends with BreakOp and there are no other break or continue
/// ops in the body, the loop can be removed.  Such loops are often generated as
/// inlining by-product, where breaks are used to represent returns from the
/// inlined function.
///
/// Before:                    After:
/// {                          {
///   ...                        ...
///   %x = hlcf.loop {
///      %a = A                  %x = A
///      hlcf.break %a
///   }
///   C                          C
/// }                          }
bool SimplifyCF::tryRemovingLoop(LoopOp loop) {
  Block &body = loop.getBody().front();
  int count = jumpsCount[loop];
  IRRewriter b{OpBuilder(loop)};

  // Fold loops with a single break to that loop.
  StringAttr label;
  auto breakOp = dyn_cast<BreakOp>(body.getTerminator());
  if (count == 1 && breakOp &&
      (!(label = breakOp.getLabelAttr()) || label == loop.getLabelAttr())) {
    // Move out the loop body and then erase the loop.
    b.inlineBlockBefore(&body, loop, loop.getOperands());
    loop.replaceAllUsesWith(breakOp.getOperands());

    b.eraseOp(breakOp);
    b.eraseOp(loop);
    return true;
  }

  // The loop is never branched to, meaning the terminator branches to some
  // other parent operation.
  if (count == 0) {
    // Move out the loop body. The loop results will have no uses after the
    // subsequent operations are erased.
    b.inlineBlockBefore(&body, loop, loop.getOperands());
    Block *toErase = b.splitBlock(loop->getBlock(), loop->getIterator());
    b.eraseBlock(toErase);
    return true;
  }

  return false;
}

/// Given a try in the following form:
///
/// ```mlir
/// lit.try {
///   A
///   lit.try.yield
/// } except (%args) {
///   B
/// } else {
///   C
///   lit.try.yield
/// }
/// ```
///
/// Where `A` represents code that does not raise, transform this into
///
/// ```mlir
/// A
/// C
/// ```
static void removeTrivialTry(LIT::TryOp op) {
  IRRewriter b{OpBuilder(op)};

  // We know the 'try' body has no raises, meaning it always reaches its
  // terminator or early exits to a parent region.
  Block *tryBlock = &op.getTryRegion().front();
  Operation *tryTerm = tryBlock->getTerminator();
  b.inlineBlockBefore(tryBlock, op);

  // If the terminator is not a yield and not a raise, then this is a branch
  // to a parent operation. Erase all subsequent ops.
  if (!isa<LIT::TryYieldOp>(tryTerm)) {
    Block *toErase = b.splitBlock(op->getBlock(), op->getIterator());
    b.eraseBlock(toErase);
    return;
  }

  // It branches to the 'else' region with the operands of the yield.
  Block *elseBlock = &op.getElseRegion().front();
  Operation *elseTerm = elseBlock->getTerminator();
  b.inlineBlockBefore(elseBlock, op, tryTerm->getOperands());
  b.eraseOp(tryTerm);

  // If the terminator is not a yield, then this is a branch to a parent
  // operation. Else all subsequent ops.
  if (!isa<LIT::TryYieldOp>(elseTerm)) {
    Block *toErase = b.splitBlock(op->getBlock(), op->getIterator());
    b.eraseBlock(toErase);
    return;
  }

  // Replace uses of the operation with the yield.
  b.replaceOp(op, elseTerm->getOperands());
  b.eraseOp(elseTerm);
}

//===----------------------------------------------------------------------===//
// Pass Driver
//===----------------------------------------------------------------------===//

void SimplifyCF::runOnOperation() {
  loopsInOrder.clear();
  parentLoops.clear();
  jumpsCount.clear();
  triesInOrder.clear();
  elidableTries.clear();
  lastTry = nullptr;

  // Walk over the functions and count how many jumps (breaks or continues) each
  // loop has. Note that breaks and continues targeting inner loops do not count
  // as jumps in this context - we only care about control flow transfers that
  // move us out of this loop.
  {
    VerboseCompilerTimeTraceScope traceScope("cfgAnalysis");
    walkPreorder(getOperation().getBodyRegion());
  }

  // Try to remove trivial loops. Process in reverse to make sure later ops are
  // visited first.
  VerboseCompilerTimeTraceScope traceScope("eraseOps");
  numErasedLoops = 0;
  numErasedTry = 0;
  for (LoopOp loop : llvm::reverse(loopsInOrder))
    numErasedLoops += tryRemovingLoop(loop);
  // Remove elidable tries.
  for (LIT::TryOp tryOp : llvm::reverse(triesInOrder)) {
    if (elidableTries.contains(tryOp)) {
      removeTrivialTry(tryOp);
      ++numErasedTry;
    }
  }
}
