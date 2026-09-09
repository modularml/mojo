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

#include "LLVMLoweringUtils.h"
#include "Mojo/CODialect/COOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/TransformUtils/AsyncUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;
using namespace CO;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// LowerSuspensionPoints
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERSUSPENSIONPOINTS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerSuspensionPointsPass
    : public KGEN::impl::LowerSuspensionPointsBase<LowerSuspensionPointsPass> {
  using LowerSuspensionPointsBase::LowerSuspensionPointsBase;

  void runOnOperation() override;
  LogicalResult initialize(MLIRContext *ctx) override {
    coroAttrName = StringAttr::get(ctx, "coroutineType");
    return success();
  }
  StringAttr coroAttrName;
};
} // namespace

struct BuildContext {
  BuildContext(LLVMBuilder &builder, Type continuationType)
      : builder(builder), continuationType(continuationType) {}

  Value getContinuationField(Value operand, AsyncContinuationField fieldIndex) {
    Type type;
    switch (fieldIndex) {
    case State:
      type = builder.getI32Type();
      break;
    case CallbackFn:
    case ClosureState:
      type = LLVMPointerType::get(builder.getContext());
      break;
    default:
      assert(false && "LowerSuspension points need not handle continuation "
                      "fields frame, promise, or resume");
    }
    GEPOp slot =
        GEPOp::create(builder,
                      /*resultType=*/LLVMPointerType::get(builder.getContext()),
                      /*elementType=*/continuationType,
                      /*basePtr=*/operand, ArrayRef<GEPArg>({0, fieldIndex}));
    return LoadOp::create(builder, type, slot);
  }

  LLVMBuilder &builder;
  SmallVector<Block *> blockList;
  SmallVector<int32_t> resumeValues;
  Type continuationType;
};

static void addSuspensionPoint(SuspendOp suspend, Block *currentBlock,
                               int32_t suspensionPointID,
                               BuildContext &buildContext) {
  buildContext.builder.setInsertionPoint(suspend);
  // Move operations from suspend region. They represent code to execute after
  // update state but before return.
  Block *susBlock = &suspend.getRegion().front();
  Block *targetBlock = suspend->getBlock();
  auto location = suspend->getIterator();
  while (susBlock) {
    Operation *currentOp = &susBlock->front();
    while (currentOp) {
      Operation *next = currentOp->getNextNode();
      assert(!isa<SuspendOp>(currentOp) &&
             "cannot have a suspend nested inside a suspend");
      currentOp->moveBefore(targetBlock, location);
      if (isa<SuspendEndOp>(currentOp)) {
        auto savePoint = buildContext.builder.saveInsertionPoint();
        buildContext.builder.setInsertionPoint(currentOp);
        ReturnOp::create(buildContext.builder, ValueRange({}));
        currentOp->erase();
        buildContext.builder.restoreInsertionPoint(savePoint);
        break;
      }
      currentOp = next;
    }
    susBlock->replaceAllUsesWith(targetBlock);
    for (auto [src, target] :
         llvm::zip(susBlock->getArguments(), targetBlock->getArguments()))
      src.replaceAllUsesWith(target);
    susBlock = susBlock->getNextNode();
    targetBlock = suspend->getBlock()->splitBlock(suspend);
    if (susBlock) {
      for (auto arg : susBlock->getArguments())
        targetBlock->addArgument(arg.getType(), arg.getLoc());
      location = targetBlock->begin();
    }
  }
  buildContext.blockList.push_back(targetBlock);
  buildContext.resumeValues.push_back(suspensionPointID);
}

static LogicalResult lowerSuspensionPoints(LLVMFuncOp funcOp,
                                           StringAttr coroAttrName) {
  int32_t suspensionPointID = 0;
  if (!funcOp->hasAttr(coroAttrName))
    return success();

  TypeAttr coroType = cast<TypeAttr>(funcOp->getAttr(coroAttrName));
  TargetInfoAttr target = lookupTargetInfo(funcOp);
  if (!target) {
    mlir::emitError(funcOp.getLoc(),
                    "could not find an enclosing target specification");
    return failure();
  }

  ImplicitLocOpBuilder opBuilder(funcOp.getLoc(), funcOp.getContext());
  LLVMBuilder b(opBuilder, target);

  // Find all suspension points. Create a new block for each suspension point.
  BuildContext buildContext(b, coroType.getValue());
  SmallVector<Block *> exitPaths;
  Block *block = &funcOp.getBody().front();
  while (block) {
    Block *nextBlock = block->getNextNode();
    bool continueInResume = false;
    b.setInsertionPointToStart(block);
    Operation *current = &block->getOperations().front();
    while (current) {
      if (isa<ReturnOp>(current))
        exitPaths.push_back(continueInResume ? buildContext.blockList.back()
                                             : block);
      if (auto suspend = dyn_cast<SuspendOp>(current)) {
        current = suspend->getNextNode();
        Block *b = continueInResume ? buildContext.blockList.back() : block;
        addSuspensionPoint(suspend, b, ++suspensionPointID, buildContext);
        suspend->erase();
        continueInResume = true;
        continue;
      }
      current = current->getNextNode();
    }
    block = nextBlock;
  }

  DenseSet<Block *> visited;
  bool hasSuspensionPoints = !buildContext.blockList.empty();
  if (hasSuspensionPoints) {
    // Create the initial switch to direct to the correct resume point.
    Block &initialBlock = funcOp.getBody().front();
    Block *controlBlock =
        b.createBlock(&funcOp.getRegion(), funcOp->getRegion(0).begin());
    for (Value arg : initialBlock.getArguments())
      controlBlock->addArgument(arg.getType(), arg.getLoc());
    initialBlock.getArgument(0).replaceAllUsesWith(
        controlBlock->getArgument(0));
    initialBlock.eraseArgument(0);

    b.setInsertionPoint(controlBlock, controlBlock->begin());
    Value state = buildContext.getContinuationField(
        funcOp.getBody().getArgument(0), AsyncContinuationField::State);
    buildContext.blockList.push_back(&initialBlock);
    buildContext.resumeValues.push_back(0);
    SmallVector<ValueRange> operands(buildContext.blockList.size());
    SwitchOp::create(
        b, state,
        /*defaultDestination=*/&initialBlock,
        /*defaultOperands=*/ValueRange(),
        /*caseValues=*/
        DenseIntElementsAttr::get(
            VectorType::get({(int32_t)buildContext.resumeValues.size()},
                            b.getI32Type()),
            buildContext.resumeValues),
        /*caseDestinations=*/buildContext.blockList,
        /*caseOperands=*/operands);
  }

  // Invoke callback in final block before return.
  Type ptrType = LLVMPointerType::get(buildContext.builder.getContext());
  Value continuation = funcOp.getArgument(0);
  for (Block *current : exitPaths) {
    Operation *terminator = current->getTerminator();
    b.setInsertionPoint(terminator);
    Value callbackFnPtr = buildContext.getContinuationField(
        continuation, AsyncContinuationField::CallbackFn);
    Value parent = buildContext.getContinuationField(
        continuation, AsyncContinuationField::ClosureState);
    SmallVector<Type> params;
    params.push_back(ptrType);
    CallOp callOp = CallOp::create(
        b, LLVMFunctionType::get(LLVMVoidType::get(b.getContext()), params),
        ValueRange{callbackFnPtr, parent});
    callOp.setTailCallKind(TailCallKind::MustTail);
  }
  return success();
}

void LowerSuspensionPointsPass::runOnOperation() {
  if (getOperation().isExternal())
    return;
  if (failed(lowerSuspensionPoints(getOperation(), coroAttrName)))
    return signalPassFailure();
  // HACK: Hoist all constant-sized allocations to the entry block. This is
  // required so that LLVM doesn't generate dynamic allocas. This is a hack
  // because it's indiscriminant.
  Block *body = &getOperation().getBody().front();
  getOperation().walk<mlir::WalkOrder::PreOrder>(
      [&](mlir::LLVM::AllocaOp alloca) {
        if (auto size =
                alloca.getArraySize().getDefiningOp<mlir::LLVM::ConstantOp>()) {
          alloca->moveBefore(body, body->begin());
          size->moveBefore(alloca);
          alloca->setLoc(FusedLoc::get(
              alloca.getContext(), {alloca.getLoc(), getOperation().getLoc()}));
          return WalkResult::skip();
        }
        return WalkResult::advance();
      });
}
