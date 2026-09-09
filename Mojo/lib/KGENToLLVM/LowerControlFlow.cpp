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
//
// Lower HLCF control flow to LLVM.
//
//===----------------------------------------------------------------------===//

#include "LLVMLoweringUtils.h"
#include "Mojo/HLCFDialect/Analysis/ControlFlowTree.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace M;
using namespace HLCF;
namespace LLVM = mlir::LLVM;

namespace {
/// This object contains information needed to lower operations. Lowering will
/// happen top-down, so we need to traverse the tree once to build references
/// between terminators and their branch targets and then start lowering the
/// operations.
struct ControlFlowConverter {
  explicit ControlFlowConverter(MLIRContext *ctx, const ControlFlowTree &tree,
                                mlir::LLVMTypeConverter &typeConverter)
      : b(ctx), tree(tree), typeConverter(typeConverter) {}

  /// Lower the control-flow node.
  LogicalResult lowerNode(ControlFlowNode node, unsigned &termId);

  /// Lower the terminator.
  LogicalResult lowerTerminator(ControlFlowTerminator term, unsigned &termId);
  LogicalResult lowerReturn(KGEN::ReturnOp op, ValueRange operands);

  /// The rewriter to use.
  IRRewriter b;

  /// The control flow tree analysis.
  const ControlFlowTree &tree;

  /// The type converter to use to convert result and argument types.
  mlir::LLVMTypeConverter &typeConverter;

  /// A map of operations to their lowered entry and exit blocks. The ID is the
  /// depth-first visit order of the operation.
  SmallVector<std::pair<SmallVector<Block *, 2>, Block *>> blocks;
};
} // namespace

static Block *getTargetBlock(ArrayRef<Block *> entries, Block *after,
                             std::optional<unsigned> index) {
  if (!index)
    return after;
  return entries[*index];
}

LogicalResult ControlFlowConverter::lowerNode(ControlFlowNode node,
                                              unsigned &termId) {
  Block *before = node->getBlock();
  Block *after = b.splitBlock(before, Block::iterator(node));
  SmallVector<Block *, 2> entries;
  for (Region &region : node->getRegions())
    entries.push_back(&region.front());
  blocks.emplace_back(entries, after);
  SmallVector<Operation *> nestedNodes;

  // Process each region in the operation.
  for (Region &region : node->getRegions()) {
    // Rewrite the block argument types and inline the body.
    for (Block &block : region) {
      b.setInsertionPointToStart(&block);
      for (BlockArgument arg : block.getArguments()) {
        Type argType = typeConverter.convertType(arg.getType());
        if (!argType)
          return mlir::emitError(arg.getLoc(),
                                 "failed to convert argument type");
        // Materialize the source conversion.
        auto source = mlir::UnrealizedConversionCastOp::create(
            b, arg.getLoc(), arg.getType(), arg);
        arg.replaceAllUsesExcept(source.getResult(0), source);
        arg.setType(argType);
      }
      // Lower the terminator.
      if (isa<ControlFlowTerminator>(block.getTerminator()))
        if (failed(lowerTerminator(
                cast<ControlFlowTerminator>(block.getTerminator()), termId)))
          return failure();
      // Defer nested nodes.
      for (Operation &op : block.without_terminator())
        if (isa<ControlFlowNode>(op))
          nestedNodes.push_back(&op);
    }

    // Inline the region.
    b.inlineRegionBefore(region, after);
  }

  // Replace the results of the operation with the arguments of the exit block.
  b.setInsertionPointToStart(after);
  for (OpResult result : node->getOpResults()) {
    Type argType = typeConverter.convertType(result.getType());
    if (!argType)
      return mlir::emitError(result.getLoc(), "failed to convert result #")
             << result.getResultNumber() << " type: " << result.getType();
    BlockArgument arg = after->addArgument(argType, result.getLoc());
    auto source = mlir::UnrealizedConversionCastOp::create(
        b, arg.getLoc(), result.getType(), arg);
    result.replaceAllUsesWith(source.getResult(0));
  }

  b.setInsertionPointToEnd(before);
  // Replace the operation.
  if (auto cond = dyn_cast<IfOp>(node.getOperation())) {
    Type condType = typeConverter.convertType(cond.getCond().getType());
    if (!condType)
      return mlir::emitError(cond.getLoc(), "failed to convert condition type");
    // We can not use cast_to_builtin here since mlir type converter insert
    // `UnrealizedConversionCastOp` by default, and we need to do the same to
    // cancel out type conversion.
    auto condCast = mlir::UnrealizedConversionCastOp::create(
        b, node->getLoc(), condType, cond.getCond());
    LLVM::CondBrOp::create(b, node->getLoc(), condCast.getResult(0),
                           entries.front(), ValueRange(), entries.back(),
                           ValueRange());
    b.eraseOp(node);
  } else if (auto sw = dyn_cast<SwitchOp>(node.getOperation())) {
    auto arg = mlir::UnrealizedConversionCastOp::create(
        b, node->getLoc(), typeConverter.getIndexType(), sw.getArg());
    SmallVector<APInt> caseValues;
    for (const auto &it : llvm::enumerate(sw.getCaseValues()))
      caseValues.emplace_back(32, it.value(), /*isSigned=*/true);

    LLVM::SwitchOp::create(b, node->getLoc(), arg.getResult(0), entries.front(),
                           ValueRange(), caseValues,
                           ArrayRef(entries).drop_front(),
                           SmallVector<ValueRange>(entries.size() - 1));
    b.eraseOp(node);
  } else {
    SmallVector<ControlFlowTarget, 1> targets;
    node.getEntryTargets(
        SmallVector<Attribute>(node->getNumOperands(), Attribute()), targets);
    if (targets.size() != 1)
      return node.emitOpError("cannot lower node without 1 entry target");

    // Materialize conversions for the entry inputs.
    SmallVector<Value> inputs;
    inputs.reserve(targets.front().inputs.size());
    for (Value input : targets.front().inputs) {
      Type type = typeConverter.convertType(input.getType());
      if (!type)
        return mlir::emitError(input.getLoc(), "failed to convert input type");
      auto dest = mlir::UnrealizedConversionCastOp::create(b, node->getLoc(),
                                                           type, input);
      inputs.push_back(dest.getResult(0));
    }
    LLVM::BrOp::create(b, node->getLoc(), inputs,
                       getTargetBlock(entries, after, targets.front().index));
    b.eraseOp(node);
  }

  // Process nested nodes.
  for (Operation *node : nestedNodes)
    if (failed(lowerNode(cast<ControlFlowNode>(node), termId)))
      return failure();
  return success();
}

LogicalResult ControlFlowConverter::lowerReturn(KGEN::ReturnOp op,
                                                ValueRange operands) {
  // If the results don't need to be packed, create the LLVM return.
  if (op->getNumOperands() <= 1) {
    b.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), operands);
    return success();
  }

  // Pack the function results in a struct.
  Type type = typeConverter.packFunctionResults(op->getOperandTypes());
  if (!type)
    return emitError(op->getLoc(), "failed to convert return types");
  Value result = LLVM::UndefOp::create(b, op->getLoc(), type);
  for (auto [index, operand] : llvm::enumerate(operands)) {
    result =
        LLVM::InsertValueOp::create(b, op->getLoc(), result, operand, index);
  }

  // Create the LLVM return.
  b.replaceOpWithNewOp<LLVM::ReturnOp>(op, result);
  return success();
}

LogicalResult ControlFlowConverter::lowerTerminator(ControlFlowTerminator term,
                                                    unsigned &termId) {
  // kgen.unreachable -> llvm.unreachable.
  if (auto unreachableOp = dyn_cast<KGEN::UnreachableOp>(term.getOperation())) {
    b.replaceOpWithNewOp<LLVM::UnreachableOp>(unreachableOp);
    return success();
  }

  // Convert the operand types.
  b.setInsertionPoint(term);
  SmallVector<Value> results;
  results.reserve(term->getNumOperands());
  for (OpOperand &operand : term->getOpOperands()) {
    Type type = typeConverter.convertType(operand.get().getType());
    if (!type)
      return mlir::emitError(operand.get().getLoc(),
                             "failed to convert operand type");
    auto dest = mlir::UnrealizedConversionCastOp::create(b, term->getLoc(),
                                                         type, operand.get());
    results.push_back(dest.getResult(0));
  }

  // Rewrite the terminator.
  if (auto returnOp = dyn_cast<KGEN::ReturnOp>(term.getOperation()))
    return lowerReturn(returnOp, results);

  assert(termId < tree.targets.size() && "malformed tree");
  auto &[nodeId, target] = tree.targets[termId];
  assert(nodeId < blocks.size() && "malformed tree");
  if (target.size() != 1)
    return term.emitOpError("cannot lower terminator without 1 target");
  b.replaceOpWithNewOp<LLVM::BrOp>(term, results,
                                   getTargetBlock(blocks[nodeId].first,
                                                  blocks[nodeId].second,
                                                  target.front().index));
  ++termId;
  return success();
}

/// Lower a single control-flow tree.
static LogicalResult
lowerControlFlowTree(Operation *root, const ControlFlowTree &tree,
                     mlir::LLVMTypeConverter &typeConverter) {
  assert(!isa<ControlFlowNode>(root->getParentOp()));
  ControlFlowConverter converter(root->getContext(), tree, typeConverter);

  // Build the control-flow tree.
  converter.blocks.reserve(tree.ops.size());

  unsigned termId = 0;
  return converter.lowerNode(cast<ControlFlowNode>(root), termId);
}

static LogicalResult
lowerControlFlowToLLVM(Operation *op, ControlFlowTreeAnalysis &analysis,
                       mlir::LLVMTypeConverter &typeConverter) {
  // Collect all the roots first since the lowering will break the walk order.
  SmallVector<Operation *> roots;
  op->walk([&](Operation *op) {
    if (isa<ControlFlowNode>(op) && !isa<ControlFlowNode>(op->getParentOp()))
      roots.push_back(op);
  });

  for (Operation *root : roots) {
    const ControlFlowTree &tree =
        analysis.getOrCreate(cast<ControlFlowNode>(root));
    if (failed(lowerControlFlowTree(root, tree, typeConverter)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERCONTROLFLOW
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerControlFlowPass
    : public KGEN::impl::LowerControlFlowBase<LowerControlFlowPass> {
  using LowerControlFlowBase::LowerControlFlowBase;

  void runOnOperation() override;
};
} // namespace

void LowerControlFlowPass::runOnOperation() {
  // Set LLVM lowering options.
  TargetInfoAttr targetInfo = lookupTargetInfo(getOperation());
  if (!targetInfo) {
    mlir::emitError(getOperation()->getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }
  KGEN::POPToLLVMTypeConverter typeConverter(targetInfo);

  // Run HLCF lowerings.
  if (failed(lowerControlFlowToLLVM(
          getOperation(), getAnalysis<HLCF::ControlFlowTreeAnalysis>(),
          typeConverter)))
    return signalPassFailure();

  // Erase unreachable blocks that might arise during HLCF lowering.
  IRRewriter rewriter(&getContext());
  (void)mlir::eraseUnreachableBlocks(rewriter, getOperation()->getRegions());
}
