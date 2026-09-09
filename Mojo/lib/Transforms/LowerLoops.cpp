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
#include "Mojo/KGENDialect/KGENDType.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace HLCF;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERLOOPS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
/// Unroll For-Loops with unrollFactor attributes:
/// - fully unroll a for-loop.
/// - unroll a for-loop with an unroll factor of a constant value .
/// This pass has to run after elaboration so that all parameter
/// expressions have elaborated with known values.
struct LowerLoops : impl::LowerLoopsBase<LowerLoops> {
  using LowerLoopsBase::LowerLoopsBase;

  void runOnOperation() override;

private:
  /// Lower hlcf.for operation to hlcf.loop
  static LogicalResult lowerForLoop(ForOp forLoop);
};
} // namespace

LogicalResult LowerLoops::lowerForLoop(ForOp forLoop) {
  Block &body = forLoop.getBody().front();

  IRRewriter rewriter{OpBuilder(forLoop)};

  // Create HLCF::LoopOp.
  auto loop = LoopOp::create(rewriter, forLoop->getLoc(),
                             forLoop->getResultTypes(), forLoop.getIterArgs());

  // Create the block for the new LoopOp.
  Block *loopBlock = rewriter.createBlock(&loop.getBody());

  // Create and rewire BlockArguments from ForOp to LoopOp.
  for (BlockArgument arg : body.getArguments()) {
    rewriter.replaceAllUsesWith(
        arg, loopBlock->addArgument(arg.getType(), arg.getLoc()));
  }

  // Insert induction variable and bound check and break;
  rewriter.setInsertionPointToStart(&body);

  // Create check condition.
  Value inductionVar = body.getArgument(0);
  HLCF::ForLoopBoundCmpPredicate cmpPredicate = forLoop.getCmpPredicateType();

  // Cast a POP scalar value to builtin index for use in index.cmp.
  auto toIndex = [&](Value v) -> Value {
    if (v.getType().isIndex())
      return v;
    mlir::Location loc = forLoop->getLoc();
    auto popIndexTy =
        SIMDType::get(v.getContext(), 1, KGENDType(KGENDType::index));
    Value asPopIndex = POP::CastOp::create(rewriter, loc, popIndexTy, v);
    return POP::CastToBuiltinOp::create(rewriter, loc, rewriter.getIndexType(),
                                        asPopIndex);
  };
  Value inductionVarIdx = toIndex(inductionVar);
  Value upperBoundIdx = toIndex(forLoop.getUpperBound());

  mlir::index::CmpOp cmpOp;
  switch (cmpPredicate) {
  case HLCF::ForLoopBoundCmpPredicate::SGT:
    cmpOp = mlir::index::CmpOp::create(rewriter, forLoop->getLoc(),
                                       mlir::index::IndexCmpPredicate::SGT,
                                       inductionVarIdx, upperBoundIdx);
    break;

  case HLCF::ForLoopBoundCmpPredicate::SLT:
    cmpOp = mlir::index::CmpOp::create(rewriter, forLoop->getLoc(),
                                       mlir::index::IndexCmpPredicate::SLT,
                                       inductionVarIdx, upperBoundIdx);
    break;
  case HLCF::ForLoopBoundCmpPredicate::SGE:
    cmpOp = mlir::index::CmpOp::create(rewriter, forLoop->getLoc(),
                                       mlir::index::IndexCmpPredicate::SGE,
                                       inductionVarIdx, upperBoundIdx);
    break;

  case HLCF::ForLoopBoundCmpPredicate::SLE:
    cmpOp = mlir::index::CmpOp::create(rewriter, forLoop->getLoc(),
                                       mlir::index::IndexCmpPredicate::SLE,
                                       inductionVarIdx, upperBoundIdx);
    break;
  }

  // Create IfOp with ThenBlock yields and ElseBlock breaks. `index.cmp`
  // produces an `i1`, so normalize to the `scalar<bool>` condition.
  //
  // TODO: this won't be needed after migrating scalar<int>, but we need to
  // create simdcmp above.
  auto boolTy = KGEN::SIMDType::getScalarBoolType(rewriter.getContext());
  Value cmpCond = KGEN::POP::CastFromBuiltinOp::create(
                      rewriter, forLoop->getLoc(), boolTy, cmpOp)
                      .getResult();
  auto ifOp = IfOp::create(rewriter, forLoop->getLoc(), ValueRange{}, cmpCond);

  rewriter.createBlock(&ifOp.getThenRegion());
  YieldOp::create(rewriter, forLoop->getLoc());
  rewriter.createBlock(&ifOp.getElseRegion());
  BreakOp::create(
      rewriter, forLoop->getLoc(),
      body.getArguments().drop_front().take_front(forLoop.getNumResults()),
      loop.getLabelAttr());

  // ForOp's terminator has to be a ForYieldOp.
  auto y = cast<ForYieldOp>(body.getTerminator());
  // Turn ForYieldOp to ContinueOp.
  rewriter.setInsertionPointAfter(y.getOperation());

  // Create `hlcf.continue` with the reordered operands.
  auto cont = HLCF::ContinueOp::create(rewriter, y.getLoc(),
                                       y->getResultTypes(), y.getOperands());

  // Replace `hlcf.for.yield with `hlcf.continue`.
  rewriter.replaceOp(y, cont);

  // Replace ForOp's results with LoopOp's
  forLoop->replaceAllUsesWith(loop.getResults());

  // Move ForOp's body to LoopOp
  rewriter.inlineBlockBefore(&body, loopBlock, loopBlock->begin(),
                             loopBlock->getArguments());

  // Erase the original forLoop.
  rewriter.eraseOp(forLoop);

  return success();
}

void LowerLoops::runOnOperation() {
  getOperation()->walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
    if (auto forLoop = dyn_cast<ForOp>(op)) {
      if (failed(lowerForLoop(forLoop))) {
        mlir::emitError(forLoop->getLoc(),
                        "Failed to lower HLCF::ForOp to HLCF::LoopOp.");
        signalPassFailure();
      }
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
}
