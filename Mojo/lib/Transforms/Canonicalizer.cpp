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
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Return true if the region's body is empty (only contains a terminator).
static bool isEmpty(Region &region) {
  assert(llvm::hasSingleElement(region));
  return llvm::hasSingleElement(region.front());
}

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Canonicalize ifs with no bodies an N results to N selects. This also removes
/// trivially dead ifs.
struct EmptyIfToSelect : public OpRewritePattern<HLCF::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(HLCF::IfOp op,
                                PatternRewriter &b) const override {
    auto thenYield = dyn_cast<HLCF::YieldOp>(op.getThenTerminator());
    auto elseYield = dyn_cast<HLCF::YieldOp>(op.getElseTerminator());
    if (!isEmpty(op.getThenRegion()) || !isEmpty(op.getElseRegion()) ||
        !thenYield || !elseYield)
      return b.notifyMatchFailure(op.getLoc(),
                                  "bodies aren't empty with 'yield'");

    // Replace each result with a 'select' of the yield operands.
    SmallVector<Value> replacements;
    for (auto [i, result] : llvm::enumerate(op.getResults())) {
      replacements.push_back(POP::SelectOp::create(b, op.getLoc(), op.getCond(),
                                                   thenYield.getOperand(i),
                                                   elseYield.getOperand(i)));
    }

    b.replaceOp(op, replacements);
    return success();
  }
};

/// Canonicalize ifs with a single operation in either then or else blocks into
/// a select of the yields. The canonicalization hoists out the operation(s)
/// therefore they're performed unconditionally.
struct IfToSelect : public OpRewritePattern<HLCF::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(HLCF::IfOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return failure();

    auto thenYield = dyn_cast<HLCF::YieldOp>(op.getThenTerminator());
    auto elseYield = dyn_cast<HLCF::YieldOp>(op.getElseTerminator());
    if (!thenYield || !elseYield)
      return failure();

    if (op.getThenBlock().getOperations().size() > 2 ||
        op.getElseBlock().getOperations().size() > 2)
      return failure();

    auto getNonYieldOp = [&](Block &block) -> Operation * {
      Operation *op = &block.getOperations().front();
      if (isa<HLCF::YieldOp>(op))
        return nullptr;
      return op;
    };

    Operation *thenOp = getNonYieldOp(op.getThenBlock());
    Operation *elseOp = getNonYieldOp(op.getElseBlock());

    auto canOpBeHoisted = [](Operation *op, HLCF::YieldOp yield) -> bool {
      // TODO: We can relax the constaraint on pure by expecting operations can
      // be executed speculatively.
      if (!op || !mlir::isPure(op) || op->getNumResults() != 1 ||
          yield.getOperand(0) != op->getResult(0))
        return false;

      // TODO: Revisit this limitation when VariantGetOp can be interpreted as
      // there are few tests failing in interpreter if canonicalization is
      // applied
      if (isa<KGEN::VariantGetOp>(op))
        return false;
      return true;
    };

    // We can only canonicalize the if-statement if operations have no
    // side-effects.
    if ((thenOp && !canOpBeHoisted(thenOp, thenYield)) ||
        (elseOp && !canOpBeHoisted(elseOp, elseYield)))
      return failure();

    auto moveOperation = [&](Operation *opToMove) {
      if (opToMove)
        rewriter.moveOpBefore(opToMove, op);
    };

    moveOperation(thenOp);
    moveOperation(elseOp);

    rewriter.replaceOpWithNewOp<KGEN::POP::SelectOp>(
        op, op.getCond(),
        thenOp ? thenOp->getResult(0) : thenYield.getOperand(0),
        elseOp ? elseOp->getResult(0) : elseYield.getOperand(0));
    return success();
  }
};

struct IfYieldSelect : public OpRewritePattern<HLCF::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(HLCF::IfOp op,
                                PatternRewriter &b) const override {
    auto thenYield = dyn_cast<HLCF::YieldOp>(op.getThenTerminator());
    auto elseYield = dyn_cast<HLCF::YieldOp>(op.getElseTerminator());
    // Constructing dominance info is cheap because we have single-block
    // regions.
    mlir::DominanceInfo domInfo;
    auto dominatesIf = [&](Value value) {
      // Block arguments always dominate the IfOp, because it itself can never
      // have block arguments. Otherwise, check dominance of the defining
      // operation.
      return isa<BlockArgument>(value) ||
             domInfo.properlyDominates(value.getDefiningOp(), op);
    };

    // If only one branch ends in a yield, then we can replace dominating
    // results entirely with that yield.
    if (!thenYield != !elseYield) {
      bool anyChanged = false;
      if (!thenYield)
        thenYield = elseYield;
      for (auto [result, operand] :
           llvm::zip(op.getResults(), thenYield.getOperands())) {
        if (dominatesIf(operand)) {
          b.replaceAllUsesWith(result, operand);
          anyChanged = true;
        }
      }
      return success(anyChanged);
    }

    // The end of the IfOp is unreachable.
    if (!thenYield)
      return failure();

    // Both branches end in a yield. We can hoist each into a select.
    bool anyChanged = false;
    for (auto [result, trueVal, falseVal] :
         llvm::zip(op.getResults(), thenYield.getOperands(),
                   elseYield.getOperands())) {
      if (dominatesIf(trueVal) && dominatesIf(falseVal)) {
        Value select = POP::SelectOp::create(b, op.getLoc(), op.getCond(),
                                             trueVal, falseVal);
        b.replaceAllUsesWith(result, select);
        anyChanged = true;
      }
    }
    return success(anyChanged);
  }
};

/// Canonicalize `!kgen.scalar<index>` and `!kgen.scalar<uindex>` computations
/// to `index` operations.
class IndexifyComparison : public OpRewritePattern<POP::CastToBuiltinOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(POP::CastToBuiltinOp op,
                                PatternRewriter &b) const override {
    if (op.getInput().getType().getResolvedDType() != KGENDType::kBool)
      return b.notifyMatchFailure(op.getLoc(), "not bool dtype");

    auto cmp = op.getInput().getDefiningOp<POP::CmpOp>();
    if (!cmp || !cmp->hasOneUse())
      return b.notifyMatchFailure(op.getLoc(),
                                  "input isn't single-use comparison");

    auto cast = cmp.getLhs().getDefiningOp<POP::CastFromBuiltinOp>();
    if (!cast || !cast->hasOneUse())
      return b.notifyMatchFailure(op.getLoc(), "LHS isn't single-use cast");
    std::optional<KGENDType> dtype =
        cast.getResult().getType().getResolvedDType();
    if (!dtype || !(dtype->isIndex() || dtype->isUIndex()))
      // if (!dtype || !dtype->isIndex())
      return b.notifyMatchFailure(op.getLoc(),
                                  "LHS isn't index or uindex dtype");

    KGEN::SIMDAttr rhs;
    if (!mlir::matchPattern(cmp.getRhs(), mlir::m_Constant(&rhs)))
      return b.notifyMatchFailure(op.getLoc(), "RHS isn't constant");

    auto cst = mlir::index::ConstantOp::create(
        b, op.getLoc(), rhs.getValues().front().getIndexVal());
    b.replaceOpWithNewOp<mlir::index::CmpOp>(
        op, getIndexCmpPredicate(cmp.getPred(), /*isSigned=*/dtype->isIndex()),
        cast.getInput(), cst);
    b.eraseOp(cmp);
    b.eraseOp(cast);
    return success();
  }

private:
  /// Get the equivalent index comparison predicate. POP treats the `index`
  /// dtype as signed.
  static mlir::index::IndexCmpPredicate
  getIndexCmpPredicate(KGEN::CmpPredicate pred, bool isSigned) {
    switch (pred) {
    case KGEN::CmpPredicate::EQ:
      return mlir::index::IndexCmpPredicate::EQ;
    case KGEN::CmpPredicate::NE:
      return mlir::index::IndexCmpPredicate::NE;
    case KGEN::CmpPredicate::LT:
      return isSigned ? mlir::index::IndexCmpPredicate::SLT
                      : mlir::index::IndexCmpPredicate::ULT;
    case KGEN::CmpPredicate::GT:
      return isSigned ? mlir::index::IndexCmpPredicate::SGT
                      : mlir::index::IndexCmpPredicate::UGT;
    case KGEN::CmpPredicate::LE:
      return isSigned ? mlir::index::IndexCmpPredicate::SLE
                      : mlir::index::IndexCmpPredicate::ULE;
    case KGEN::CmpPredicate::GE:
      return isSigned ? mlir::index::IndexCmpPredicate::SGE
                      : mlir::index::IndexCmpPredicate::UGE;
    }
    llvm_unreachable("invalid cmp predicate");
  }
};

/// Replace:
///
/// ```mlir
/// %0 = index.cmp <pred>(%a, %b)
/// %1 = cast_from_builtin %0 : i1 to scalar<bool>
/// %2 = xor %1, %simd_bool_0
/// %3 = cast_to_builtin %2 : scalar<bool> to i1
/// ```
///
/// With:
///
/// ```mlir
/// %3 = index.cmp <not pred>(%a, %b)
/// ```
struct InvertComparison : OpRewritePattern<POP::CastToBuiltinOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(POP::CastToBuiltinOp op,
                                PatternRewriter &b) const override {
    if (op.getInput().getType().getResolvedDType() != KGENDType::kBool)
      return b.notifyMatchFailure(op.getLoc(), "not bool dtype");

    auto notOp = op.getInput().getDefiningOp<POP::SIMDXOrOp>();
    if (!notOp)
      return b.notifyMatchFailure(op.getLoc(), "parent isn't xor");

    KGEN::SIMDAttr zeroAttr;
    if (!mlir::matchPattern(notOp.getRhs(), mlir::m_Constant(&zeroAttr)) ||
        zeroAttr.getValues().front().getBoolVal() != true)
      return b.notifyMatchFailure(notOp.getLoc(), "not xor with true");

    auto inCast = notOp.getLhs().getDefiningOp<POP::CastFromBuiltinOp>();
    if (!inCast)
      return b.notifyMatchFailure(notOp.getLoc(), "lhs parent isn't cast");

    auto cmpOp = inCast.getInput().getDefiningOp<mlir::index::CmpOp>();
    if (!cmpOp)
      return b.notifyMatchFailure(inCast.getLoc(), "parent isn't cmp");

    b.replaceOpWithNewOp<mlir::index::CmpOp>(
        op, getInvertedPred(cmpOp.getPred()), cmpOp.getLhs(), cmpOp.getRhs());
    return success();
  }

private:
  static mlir::index::IndexCmpPredicate
  getInvertedPred(mlir::index::IndexCmpPredicate pred) {
    switch (pred) {
    case mlir::index::IndexCmpPredicate::EQ:
      return mlir::index::IndexCmpPredicate::NE;
    case mlir::index::IndexCmpPredicate::NE:
      return mlir::index::IndexCmpPredicate::EQ;

    case mlir::index::IndexCmpPredicate::SLT:
      return mlir::index::IndexCmpPredicate::SGE;
    case mlir::index::IndexCmpPredicate::SLE:
      return mlir::index::IndexCmpPredicate::SGT;
    case mlir::index::IndexCmpPredicate::SGT:
      return mlir::index::IndexCmpPredicate::SLE;
    case mlir::index::IndexCmpPredicate::SGE:
      return mlir::index::IndexCmpPredicate::SLT;

    case mlir::index::IndexCmpPredicate::ULT:
      return mlir::index::IndexCmpPredicate::UGE;
    case mlir::index::IndexCmpPredicate::ULE:
      return mlir::index::IndexCmpPredicate::UGT;
    case mlir::index::IndexCmpPredicate::UGT:
      return mlir::index::IndexCmpPredicate::ULE;
    case mlir::index::IndexCmpPredicate::UGE:
      return mlir::index::IndexCmpPredicate::ULT;
    }
    llvm_unreachable("invalid cmp predicate");
  }
};

/// Canonicalize
/// `(i < x ? x - i : 0) > 0` to `i < x`. or
/// `(x > i ? x - i : 0) > 0` to `x > i`. or
/// `(x > 0 ? x : 0) > 0` to `x > 0`. or
/// `(0 < x ? x : 0) > 0` to `0 < x`.
/// This is a common pattern
/// in for loop constructs.
/// TODO: Generalize this pattern?
struct SimplifyCompareSelect : OpRewritePattern<mlir::index::CmpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::index::CmpOp op,
                                PatternRewriter &b) const override {
    if (op.getPred() != mlir::index::IndexCmpPredicate::SGT)
      return b.notifyMatchFailure(op.getLoc(), "predicate is not `sgt`");

    IntegerAttr cmpRhs;
    if (!mlir::matchPattern(op.getRhs(), mlir::m_Constant(&cmpRhs)) ||
        !cmpRhs.getValue().isZero())
      return b.notifyMatchFailure(op.getLoc(), "RHS is not zero");

    auto select = op.getLhs().getDefiningOp<POP::SelectOp>();
    if (!select)
      return b.notifyMatchFailure(op.getLoc(), "LHS is not a select");

    auto indexCmp = select.getCondition().getDefiningOp<mlir::index::CmpOp>();
    if (!indexCmp)
      return b.notifyMatchFailure(op.getLoc(),
                                  "select condition is not a comparison.");

    auto falseValue = select.getFalseValue();
    auto trueVal = select.getTrueValue();
    auto trueValSub = trueVal.getDefiningOp<mlir::index::SubOp>();

    IntegerAttr falseV;
    if (!mlir::matchPattern(falseValue, mlir::m_Constant(&falseV)) ||
        !falseV.getValue().isZero())
      return b.notifyMatchFailure(op.getLoc(),
                                  "Select's false value is not zero");

    if (indexCmp.getPred() == mlir::index::IndexCmpPredicate::SLT) {
      if (!(indexCmp.getLhs() == falseValue && indexCmp.getRhs() == trueVal)) {
        if (!trueValSub || trueValSub.getLhs() != indexCmp.getRhs() ||
            trueValSub.getRhs() != indexCmp.getLhs())
          return b.notifyMatchFailure(op.getLoc(),
                                      "select true value is not `x - i`");
      }
    } else if (indexCmp.getPred() == mlir::index::IndexCmpPredicate::SGT) {
      if (!(indexCmp.getLhs() == trueVal && indexCmp.getRhs() == falseValue)) {
        if (!trueValSub || trueValSub.getLhs() != indexCmp.getLhs() ||
            trueValSub.getRhs() != indexCmp.getRhs())
          return b.notifyMatchFailure(op.getLoc(),
                                      "select true value is not `x - i`");
      }
    } else {
      return b.notifyMatchFailure(
          op.getLoc(), "select condition is not `slt` or `sgt` comparison");
    }

    IntegerAttr falseVal;
    if (!mlir::matchPattern(select.getFalseValue(),
                            mlir::m_Constant(&falseVal)) ||
        falseVal != cmpRhs)
      return b.notifyMatchFailure(op.getLoc(),
                                  "select false value is not zero");

    // Just replace the whole thing with `i < x` or `x > i`.
    b.replaceOp(op, indexCmp);
    return success();
  }
};

/// Given an if, the condition argument is known to be true within the 'then'
/// region and false in the 'else' region. Propagate this by replacing the
/// condition with a constant in both regions.
struct ConditionPropagation : OpRewritePattern<HLCF::IfOp> {
  ConditionPropagation(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/9) {}

  LogicalResult matchAndRewrite(HLCF::IfOp op,
                                PatternRewriter &b) const override {
    // The pattern matches if the condition has uses in either region. Lazily
    // create the true and false constants.
    Value trueCst, falseCst;
    for (OpOperand &use : op.getCond().getUses()) {
      if (op.getThenRegion().isAncestor(use.getOwner()->getParentRegion())) {
        if (!trueCst)
          trueCst = KGEN::ParamConstantOp::create(
              b, op.getLoc(),
              KGEN::SIMDAttr::getScalarBool(b.getContext(), true));
        use.set(trueCst);
      } else if (op.getElseRegion().isAncestor(
                     use.getOwner()->getParentRegion())) {
        if (!falseCst)
          falseCst = KGEN::ParamConstantOp::create(
              b, op.getLoc(),
              KGEN::SIMDAttr::getScalarBool(b.getContext(), false));
        use.set(falseCst);
      }
    }
    return success(trueCst || falseCst);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_CANONICALIZER
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct Canonicalizer : public impl::CanonicalizerBase<Canonicalizer> {
  /// Initialize the canonicalizer by building the starting set of patterns.
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    addNonCustomCanonicalizationPatterns(context, owningPatterns);
    patterns = mlir::FrozenRewritePatternSet(std::move(owningPatterns));
    return success();
  }

  /// Add all canonicalization patterns besides the ones from the `custom`
  /// dialect into the pattern set.
  void addNonCustomCanonicalizationPatterns(MLIRContext *context,
                                            RewritePatternSet &patterns);

  void runOnOperation() override;

  /// The patterns that the canonicalizer runs.
  mlir::FrozenRewritePatternSet patterns = {};
};

void Canonicalizer::runOnOperation() {
  // Run the canonicalization patterns
  mlir::GreedyRewriteConfig config;
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  (void)applyPatternsGreedily(getOperation(), patterns, config);
}

void Canonicalizer::addNonCustomCanonicalizationPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  // Add the "static" canonicalization patterns.
  for (auto *dialect : context->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (mlir::RegisteredOperationName op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, context);

  // clang-format off
  patterns.insert<
    EmptyIfToSelect,
    IfToSelect,
    IfYieldSelect,
    IndexifyComparison,
    InvertComparison,
    SimplifyCompareSelect,
    ConditionPropagation
   >(context);
  // clang-format on
}

} // namespace
