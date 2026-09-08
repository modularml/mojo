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

#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/Dominance.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_HOISTTRIVIALINVARIANTS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct HoistTrivialInvariants
    : impl::HoistTrivialInvariantsBase<HoistTrivialInvariants> {
  void runOnOperation() override;
};
} // namespace

/// Hoist invariant operations to the earliest legal point we can within the
/// function. Either to the start if they use only input arguments or to the
/// producer of whichever operand is dominated by all other operands.
static void moveInvariants(FunctionLike func, mlir::DominanceInfo &domInfo,
                           Operation *opWithRegion, Region &region,
                           unsigned &numHoisted) {
  // Move the invariants.
  for (Operation &op : llvm::make_early_inc_range(region.front())) {
    // This pass only will hoist pure operations without regions.
    if (!isPure(&op) || op.getNumRegions())
      continue;

    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;

    // A pure operation is invariant if all of its operands are invariant.
    // In basic invariant code motion we just check if is created within this
    // loop / if, if so we don't move it. Otherwise we assume it is safe to move
    // and leave LLVM to decide whether or not it is more performant for it to
    // be hoisted back in.
    bool safe = true;
    for (Value operand : op.getOperands()) {
      if (operand.getParentRegion()->getParentOp() == opWithRegion) {
        safe = false;
        break;
      }
    }

    if (!safe)
      continue;

    // The operand owner that is dominated by all others, but capped at the
    // surrounding function.
    PointerUnion<Operation *, Region *> leastDominatingOperand =
        &func.getBodyRegion();

    // Traverse again to avoid touching dom info on region variant ops.
    for (Value operand : op.getOperands()) {
      if (Operation *parent = operand.getDefiningOp()) {
        if (auto *op = leastDominatingOperand.dyn_cast<Operation *>()) {
          if (domInfo.dominates(op, parent))
            leastDominatingOperand = parent;
        } else if (cast<Region *>(leastDominatingOperand)
                       ->isAncestor(parent->getParentRegion())) {
          leastDominatingOperand = parent;
        }
      } else {
        Region *region = operand.getParentRegion();
        if (auto *op = leastDominatingOperand.dyn_cast<Operation *>()) {
          if (op->getParentRegion()->isProperAncestor(region))
            leastDominatingOperand = region;
        } else if (cast<Region *>(leastDominatingOperand)
                       ->isProperAncestor(region)) {
          leastDominatingOperand = region;
        }
      }
    }

    // Hoist to the earliest legal point. If it's a region, most to the
    // beginning of the region. Otherwise, move to just after the operation.
    if (auto *region = leastDominatingOperand.dyn_cast<Region *>())
      op.moveBefore(&region->front(), region->front().begin());
    else
      op.moveAfter(cast<Operation *>(leastDominatingOperand));
    ++numHoisted;
  }
}

/// Hoist only within functions. Hoisting an operation out of a function-like
/// region may invalid or have complex cost models because the semantics of
/// implicit captures are not understood.
static void moveInvariantsIn(FunctionLike func, mlir::DominanceInfo &domInfo,
                             unsigned &numHoisted) {
  func.getBodyRegion().walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto func = dyn_cast<FunctionLike>(op)) {
      moveInvariantsIn(func, domInfo, numHoisted);
      return WalkResult::skip();
    }

    // We maintain a small list of operations which we are allowed to hoist
    // invariants from.
    if (auto loop = dyn_cast<HLCF::LoopOp>(op)) {
      moveInvariants(func, domInfo, loop, loop.getBody(), numHoisted);
    }

    // We hoist from both branches of the if regardless of the condition with
    // the guarantee that these ops have no side effects and LLVM is free to
    // move them back if that is more optimal.
    else if (auto ifOp = dyn_cast<HLCF::IfOp>(op)) {
      moveInvariants(func, domInfo, ifOp, ifOp.getThenRegion(), numHoisted);
      moveInvariants(func, domInfo, ifOp, ifOp.getElseRegion(), numHoisted);
    }

    else if (auto tryOp = dyn_cast<LIT::TryOp>(op)) {
      moveInvariants(func, domInfo, tryOp, tryOp.getTryRegion(), numHoisted);
      moveInvariants(func, domInfo, tryOp, tryOp.getExceptRegion(), numHoisted);
      moveInvariants(func, domInfo, tryOp, tryOp.getElseRegion(), numHoisted);
    }

    return WalkResult::advance();
  });
}

void HoistTrivialInvariants::runOnOperation() {
  unsigned numHoisted = 0;
  auto func = dyn_cast<FunctionLike>(getOperation());
  if (!func) {
    mlir::emitError(getOperation()->getLoc(),
                    "'hoist-trivial-invariants' must be nested on a "
                    "function-like operation");
    return signalPassFailure();
  }
  moveInvariantsIn(func, getAnalysis<mlir::DominanceInfo>(), numHoisted);
  this->numHoisted = numHoisted;
}
