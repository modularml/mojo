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

#include "Mojo/CODialect/CODialect.h"
#include "Mojo/CODialect/COOps.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/TransformUtils/AsyncUtils.h"
#include "Support/Threading/Shared.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace POP;
using namespace CO;

//===----------------------------------------------------------------------===//
// Lower Async Functions
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERASYNCFUNCTIONS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerAsyncFunctionsPass
    : impl::LowerAsyncFunctionsBase<LowerAsyncFunctionsPass> {
public:
  using LowerAsyncFunctionsBase::LowerAsyncFunctionsBase;
  void runOnOperation() override;
};
} // namespace

/// Frame Data stores any metadata necessary to transform the async function
/// into a suspendable procedure. This includes indexing information into the
/// frame type so that we can generate loads and stores and state information so
/// we can reuse loaded frame variables when legal.
struct FrameData {
  /// Error value and result value are excluded from the frame
  FrameData(FuncOp originalFunction, mlir::DominanceInfo &domInfo,
            Value errorValue, Value resultValue,
            function_ref<void(FuncOp, DenseMap<Operation *, int> &)> transform,
            bool isHot);
  FrameData(const FrameData &) = delete;
  FrameData(const FrameData &&other)
      : frameTypes(std::move(other.frameTypes)),
        valueToIndexInFrame(std::move(other.valueToIndexInFrame)),
        operationToIndexInFrame(std::move(other.operationToIndexInFrame)),
        opToState(std::move(other.opToState)),
        virtualBlocksFirstState(std::move(other.virtualBlocksFirstState)),
        argsInFrame(std::move(other.argsInFrame)),
        firstSuspends(std::move(other.firstSuspends)) {}

  FrameData() {}
  /// pairs index of argument from original function with its index in the
  /// frame.
  struct ArgInFrame {
    ArgInFrame(int index, int frameIndex)
        : argIndex(index), frameIndex(frameIndex) {}
    ArgInFrame() {}
    int argIndex = -1;
    int frameIndex = -1;
  };

  /// Given a value, determine the state of its defining op or block argument.
  int getDefinitionStateForValue(Value operand, bool isHot) const;

  /// Update the ops in this virtual block.
  void updateVirtualBlock(Operation *virtualBlock, int newState);

  SmallVector<Type> frameTypes;
  DenseMap<Value, unsigned> valueToIndexInFrame;
  DenseMap<Operation *, unsigned> operationToIndexInFrame;
  DenseMap<Operation *, int> opToState;
  SmallVector<Operation *> virtualBlocksFirstState;
  SmallVector<ArgInFrame> argsInFrame;
  /// A suspend op 'S' is in this data structure if it is possible to reach 'S'
  /// from the entry point without passing through another suspend op. This is
  /// used to clear the state 0 blocks in hot resumes.
  DenseSet<SuspendOp> firstSuspends;
};

static bool needsStateClone(Operation *operation) {
  return operation->hasTrait<OpTrait::ConstantLike>() ||
         isa<KGEN::StructGEPOp, POP::OffsetOp>(operation);
}

struct CloneFrameArgs {
  CloneFrameArgs(ImplicitLocOpBuilder &b, DenseMap<Operation *, int> &opToState,
                 mlir::DominanceInfo &dominanceInfo)
      : builder(b), opToState(opToState), dominanceInfo(dominanceInfo) {}
  void cloneFrameArgsOf(Operation *user) {
    int useState = opToState[user];
    for (auto [index, operand] : llvm::enumerate(user->getOperands())) {
      Operation *definingOp = operand.getDefiningOp();
      if (!definingOp)
        continue;
      if (!needsStateClone(definingOp))
        continue;
      int defState = opToState[definingOp];
      if (defState == useState)
        continue;

      auto existing = constantToStateSpecific.find(definingOp);
      if (existing == constantToStateSpecific.end())
        existing = constantToStateSpecific.try_emplace(definingOp).first;
      auto existingClone = existing->second.find(useState);
      Operation *clonedDefOp;
      if (existingClone == existing->second.end()) {
        builder.setInsertionPoint(user);
        clonedDefOp = builder.clone(*definingOp);
        opToState[clonedDefOp] = useState;
        existing->second.insert({useState, clonedDefOp});
        if (clonedDefOp->getNumOperands() > 0)
          cloneFrameArgsOf(clonedDefOp);
      } else {
        clonedDefOp = existingClone->second;
        if (!dominanceInfo.dominates(clonedDefOp, user)) {
          // We have two uses in the same state where one does not dominate the
          // other. This implies that the first instance of the usage is in a
          // nested region.
          Operation *parent = clonedDefOp->getParentOp();
          while (!dominanceInfo.dominates(parent, user))
            parent = parent->getParentOp();
          clonedDefOp->moveBefore(parent);
        }
      }
      user->setOperand(index, clonedDefOp->getResult(0));
    }
  }
  ImplicitLocOpBuilder &builder;
  DenseMap<Operation *, int> &opToState;
  DenseMap<Operation *, DenseMap<int, Operation *>> constantToStateSpecific;
  mlir::DominanceInfo &dominanceInfo;
};

static void cloneFrameArgs(FuncOp funcOp, ImplicitLocOpBuilder &b,
                           mlir::DominanceInfo &domInfo,
                           FuncOp originalFunction,
                           DenseMap<Operation *, int> &opToState) {
  auto insertPoint = b.saveInsertionPoint();
  CloneFrameArgs cloner(b, opToState, domInfo);
  funcOp.walk([&](Operation *user) {
    if (user->getNumOperands() > 0)
      cloner.cloneFrameArgsOf(user);
  });
  b.restoreInsertionPoint(insertPoint);
}

struct COTypes {
  Type typeForField(AsyncContinuationField field) {
    switch (field) {
    case State:
      return IntegerType::get(cxt, 32);
    case CallbackFn:
      return callbackSignature;
    case Promise:
      return promiseType;
    case ResumeFunction:
    case ClosureState:
    case ErrorSlot:
    case ResultSlot:
      return opaquePointerType;
    case Frame:
      return StructType::get(cxt, frameData.frameTypes);
    }
    llvm_unreachable("invalid AsyncContinuationField value");
  }
  COTypes(MLIRContext *cxt, FrameData &&frameData, StructType promiseType);
  COTypes(const COTypes &&other)
      : continuationType(other.continuationType),
        resumeSignatureType(other.resumeSignatureType),
        opaquePointerType(other.opaquePointerType),
        callbackSignature(other.callbackSignature),
        headerType(other.headerType), cxt(other.cxt),
        frameData(std::move(other.frameData)), promiseType(other.promiseType) {}
  COTypes(const COTypes &) = delete;
  COTypes &operator=(const COTypes &) = delete;

public:
  Type getContinuationType() const { return continuationType; }
  FrameData *getFrameData() { return &frameData; }
  StructType getHeaderType() const { return headerType; }
  Type getResumeSignatureType() const { return resumeSignatureType; }
  StructType getPromiseType() const { return promiseType; }

private:
  Type continuationType;
  Type resumeSignatureType;
  Type opaquePointerType;
  Type callbackSignature;
  StructType headerType;
  MLIRContext *cxt;
  FrameData frameData;
  StructType promiseType;
};

using VirtualBlock = Operation *;

/// Frame Variables is a Cache of extracted frame variables. We may for example
/// reference a frame variable multiple times within a virtual block. We should
/// only extract that variable once for that state.
class FrameVariables {
public:
  FrameVariables(ImplicitLocOpBuilder &builder, const FrameData *frameData,
                 Value errorValue, Value resultValue)
      : builder(builder), frameData(frameData), errorValue(errorValue),
        resultValue(resultValue) {}

  /// Given the original operand, return the value extracted from the frame. Use
  /// a previously extracted value if available.
  Value getFrameValueForOperand(Value continuation, Value operand,
                                Operation *opWithUse, int useState);

  /// Overwrite the cached value for the variable in this state. A state can
  /// contain nested control flow, resulting in frame variables extracted in
  /// nested blocks. Sometime we could optimize this so that frame variables
  /// used multiple times in the same state are extracted in the first shared
  /// parent block, thus removing the need to overwrite state.
  void overwriteValue(int state, Value value);

private:
  ImplicitLocOpBuilder &builder;
  const FrameData *frameData;
  DenseMap<Value, DenseMap<int, Value>> frameVariables;
  Value errorValue;
  Value resultValue;
};

/// Temperature refers to the flavor of coroutine.
/// Hot: coroutine is started upon creation
/// Cold: coroutine requires invocation of resume function to start.
/// Both: It is possible to start hot or cold.
enum class Temp { Hot, Cold, Both };
struct Coroutine {
  Coroutine(FuncOp resumeFunction, FuncOp hotRamp, FuncOp coldRamp,
            Type continuationType)
      : resumeFunction(resumeFunction), hotRamp(hotRamp), coldRamp(coldRamp),
        coroutineType(continuationType) {}
  /// This is the hot resume. If shared with the cold, it will have the frame
  /// bitcasted to the larger cold frame in the first state.
  FuncOp resumeFunction;
  /// Hot ramp is a function that creates a coroutine then executes the first
  /// state.
  FuncOp hotRamp;
  /// Cold ramp is a function that creates a coroutine without starting it.
  FuncOp coldRamp;
  /// The coroutine type is the header + frame.
  Type coroutineType;
};

/// The LowerAsyncBuildContext is responsible for transforming an async function
/// into a ramp function and resume function.
struct LowerAsyncBuildContext {
  LowerAsyncBuildContext(Shared<SymbolTable &> &sharedTable,
                         ImplicitLocOpBuilder &builder,
                         mlir::DominanceInfo &domInfo,
                         TargetInfoAttr targetInfoAttr)
      : sharedTable(sharedTable), builder(builder),
        targetInfoAttr(targetInfoAttr), dominanceInfo(domInfo) {}

  void
  preprocessAsyncFunction(FuncOp funcOp, mlir::DominanceInfo &domInfo,
                          DenseMap<SymbolConstantAttr, Temp> &temperatures);

  /// Given an async function and its frame types, create a ramp function and a
  /// resume function.
  Coroutine createCoroutine(FuncOp originalFunction, Temp temperature);

private:
  /// Given a function, calculate the full coroutine type. If include args is
  /// true, the arguments are assumed to be in state -1. Otherwise they are
  /// considered to be in state 0. In the former case they are always added to
  /// the frame.
  COTypes calculateFrame(FuncOp original, bool includeArgs);

  FuncOp createColdRamp(StringRef prefix, FuncType originalSignature,
                        COTypes &coTypes, FuncOp resumeFunction);
  /// Given an async function (a:A...) -> B, create a resume function
  /// (continuation:C) -> ()
  FuncOp createColdResume(StringRef prefix, FuncOp funcOp, COTypes &coTypes);
  /// Given an async function and cold and hot coroutine types, create a resume
  /// function that can be shared by hot and cold ramps.
  FuncOp createSharedResume(StringRef prefix, FuncOp funcOp,
                            COTypes &coldCoTypes, Type hotCoroType);
  void populateHotResumeFrom(FuncOp original, FuncOp target,
                             COTypes &coTypesOfOriginal);

  /// A hot ramp contains the first state of the given function. If takeOriginal
  /// is false, the original function will be left alone and the hot ramp will
  /// be created by cloning up to the first suspension point.
  FuncOp createHotRamp(StringRef prefix, FuncOp original, COTypes &coTypes,
                       FuncOp resumeFunction, bool takeOriginal);

  /// Create the continuation and initialize the state and resume function.
  Value initializeContinuation(FuncOp rampFunction, FuncOp resumeFunction,
                               COTypes &coTypes);

  /// Given a function and a frame, insert a continuation and load/store values
  /// from frame instead of using local values. If loadFromFrame is true, values
  /// that are in frame are pulled from frame. Load from frame will only be
  /// false in the case of hot ramp generation. This corresponds to the case
  /// where a block is reachable from a suspension point or not a suspension
  /// point. In a resume function we want to pull unconditionally from the frame
  /// but in the hot ramp case that block is only reachable from the non suspend
  /// path.
  void insertFrameLoadsStores(FuncOp resumeFunction, COTypes &coTypes,
                              Value errorValue, Value memoryResultValue,
                              Temp temp, bool loadFromFrame);

  /// Given an async function, populate a function with the paths to
  /// the first suspension point. If there is a path with no suspension point,
  /// the callback is invoked. The hotRamp TAKES the body of the fromFuncOp.
  void takeSlicedFirstStateFrom(FuncOp hotRamp, FuncOp fromFuncOp);

  /// Given an async function, populate a function with the paths to
  /// the first suspension point. If there is a path with no suspension point,
  /// the callback is invoked.
  FrameData cloneFrameAndFirstStateTo(FuncOp hotRamp, FuncOp fromFuncOp,
                                      FrameData *originalFrameData);

  /// Replace all `return x` with `store x, y` where y is the address of the
  /// result slot in the frame.
  ReturnOp lowerReturn(ReturnOp returnOp, Value continuation);
  /// Replace handle argument with local coroutine
  void lowerSuspensionPoint(CO::SuspendOp suspend, Value continuation,
                            COTypes &coTypes);
  /// Store block arguments in the frame if they are used across a suspension
  /// point.
  void storeBlockArgumentsInFrame(Block &block, Operation *key,
                                  const FrameData *frameData,
                                  Value continuation,
                                  FrameVariables &frameVariables);
  /// Store the given op in the frame if it is used across suspension points.
  void storeOpInFrameIfNeeded(const FrameData *frameData, Operation *op,
                              Value continuation,
                              SmallVector<Operation *> &opsToDelete);

  Shared<SymbolTable &> &sharedTable;
  DenseMap<SymbolConstantAttr, SymbolConstantAttr> fromOriginalToHotRamp;
  ImplicitLocOpBuilder &builder;
  TargetInfoAttr targetInfoAttr;
  mlir::DominanceInfo &dominanceInfo;
};

//===----------------------------------------------------------------------===//
// LowerAsyncBuildContext
//===----------------------------------------------------------------------===//

enum class VisitedState { SUS, NOSUS, SUS_AND_NOSUS };

static Operation *insertCoroutineEnd(ImplicitLocOpBuilder &builder,
                                     Value callback, Value closure) {
  auto signatureType = cast<FuncTypeGeneratorType>(callback.getType());
  auto callIndirect = CallIndirectOp::create(
      builder, signatureType.getBody().getResults(), callback, closure);
  callIndirect.setTailKind(TailKind::MustTail);
  return callIndirect;
}

void LowerAsyncBuildContext::takeSlicedFirstStateFrom(FuncOp hotRamp,
                                                      FuncOp fromFuncOp) {
  /// Augment the hot ramp function signature. We will clone ops from the funcOp
  /// into the hot ramp function.
  FrameData emptyFrame;
  COTypes opaqueCoTypes(
      builder.getContext(), std::move(emptyFrame),
      StructType::get(builder.getContext(), fromFuncOp.getResultTypes()));
  SmallVector<Type> inputs;
  SmallVector<ArgConvention> conventions;
  Type closureType = opaqueCoTypes.typeForField(ClosureState);
  Type callbackType = opaqueCoTypes.typeForField(CallbackFn);
  inputs.push_back(callbackType);
  conventions.push_back(ArgConvention::ImmReg);
  inputs.push_back(closureType);
  conventions.push_back(ArgConvention::ImmReg);
  llvm::append_range(inputs, fromFuncOp.getArgumentTypes());
  llvm::append_range(
      conventions,
      fromFuncOp.getFuncTypeGenerator().getBody().getArgConventions());
  FuncType signature = FuncType::get(
      FunctionType::get(builder.getContext(), inputs,
                        fromFuncOp.getResultTypes()),
      conventions, fromFuncOp.getFuncTypeGenerator().getBody().getFnEffects(),
      fromFuncOp.getFuncTypeGenerator().getBody().getMetadata(),
      fromFuncOp.getFuncTypeGenerator().getBody().getArgListAttrs());
  hotRamp.setFuncTypeGenerator(
      GeneratorType::get(/*inputParamTypes=*/{}, signature));
  hotRamp.getBodyRegion().takeBody(fromFuncOp.getBodyRegion());
  Value callback = hotRamp.getBodyRegion().front().insertArgument(
      (unsigned)0, callbackType, hotRamp->getLoc());
  Value closureState = hotRamp.getBodyRegion().front().insertArgument(
      1, closureType, hotRamp->getLoc());

  /// Clone from funcOp into hot ramp until first suspension point.
  SmallVector<std::pair<Operation *, bool>> paths;
  auto addTargets = [&](ArrayRef<HLCF::ControlFlowTarget> targets,
                        Operation *cfn, bool hitSus) {
    for (HLCF::ControlFlowTarget target : targets) {
      if (target.index.has_value()) {
        unsigned index = target.index.value();
        Block &sourceBlock = cfn->getRegion(index).front();
        paths.push_back({&*sourceBlock.begin(), hitSus});
      } else {
        paths.push_back({cfn->getNextNode(), hitSus});
      }
    }
  };
  DenseMap<Operation *, VisitedState> visited;
  DenseSet<Operation *> reachable;
  paths.push_back({&hotRamp.getBodyRegion().front().front(), false});
  while (!paths.empty()) {
    std::pair<Operation *, bool> c = paths.pop_back_val();
    Operation *current = c.first;
    bool hitSus = c.second;
    bool wasVisited = true;
    auto ptr = visited.find(current);
    if (ptr != visited.end()) {
      wasVisited = false;
      VisitedState state = ptr->getSecond();
      if ((state == VisitedState::SUS && !hitSus) ||
          (state == VisitedState::NOSUS && hitSus))
        visited[current] = VisitedState::SUS_AND_NOSUS;
      else
        continue;
    } else {
      visited.insert(
          {current, hitSus ? VisitedState::SUS : VisitedState::NOSUS});
    }
    for (Operation *operation = current; operation != nullptr;
         operation = operation->getNextNode()) {
      if (!hitSus)
        reachable.insert(operation);
      if (auto suspendOp = dyn_cast<SuspendOp>(operation)) {
        Operation *end = suspendOp;
        if (wasVisited) {
          builder.setInsertionPointAfter(suspendOp);
          // keep suspension points around for now.
          for (auto &op : suspendOp.getBody().front().getOperations())
            reachable.insert(&op);
          end = KGEN::ReturnOp::create(builder);
          reachable.insert(end);
          if (end->getNextNode())
            paths.push_back({end->getNextNode(), true});
        }
        break;
      }
      if (auto cfn = dyn_cast<HLCF::ControlFlowNode>(operation)) {
        SmallVector<HLCF::ControlFlowTarget> targets;
        SmallVector<Attribute> controlFlowOperands(cfn->getNumOperands(),
                                                   Attribute());
        cfn.getEntryTargets(controlFlowOperands, targets);
        addTargets(targets, cfn, hitSus);
        break;
      }
      if (isa<ReturnOp>(operation)) {
        if (!hitSus) {
          builder.setInsertionPoint(operation);
          reachable.insert(insertCoroutineEnd(builder, callback, closureState));
        }
        break;
      }
      if (isa<UnreachableOp, SuspendEndOp>(operation))
        break;
      if (auto terminator = dyn_cast<HLCF::ControlFlowTerminator>(operation)) {
        SmallVector<HLCF::ControlFlowTarget> targets;
        SmallVector<Attribute> controlFlowTerminatorOperands(
            terminator->getNumOperands(), Attribute());
        terminator.getBranchTargets(controlFlowTerminatorOperands, targets);
        addTargets(targets, HLCF::getParentNode(terminator), hitSus);
        break;
      }
    }
  }
  SmallVector<Operation *> deleteMe;
  SmallVector<Region *> removalRegions;
  SmallVector<Region *> regions;
  regions.push_back(&hotRamp.getBodyRegion());
  while (!regions.empty()) {
    Region *region = regions.front();
    regions.erase(regions.begin());
    removalRegions.push_back(region);
    for (Operation &op : region->front().getOperations()) {
      for (Region &r : op.getRegions())
        regions.push_back(&r);
    }
  }
  for (auto i = removalRegions.rbegin(); i != removalRegions.rend(); ++i) {
    Region *region = *i;
    for (auto opIter = region->front().getOperations().rbegin();
         opIter != region->front().rend();) {
      Operation &op = *opIter;
      ++opIter;
      if (!reachable.contains(&op))
        op.erase();
    }
  }
}

/// Given a function, return the block arguments of the entry block that
/// correspond to by ref error and by ref result, if they exist.
static std::pair<Value, Value> getErrorAndMemoryValues(FuncOp original) {
  Value errorValue;
  Value memoryResultValue;
  if (original.isThrows() ||
      original.getFuncTypeGenerator().getBody().hasMemoryOnlyResult()) {
    int errorIndex = -1, resultIndex = -1;
    for (auto [i, convention] : llvm::enumerate(
             original.getFuncTypeGenerator().getBody().getArgConventions())) {
      if (convention == ArgConvention::ByRefError)
        errorIndex = i;
      else if (convention == ArgConvention::ByRefResult)
        resultIndex = i;
    }
    if (errorIndex > -1)
      errorValue = original.getArgument(errorIndex);
    if (resultIndex > -1)
      memoryResultValue = original.getArgument(resultIndex);
  }
  return {errorValue, memoryResultValue};
}

/// Copy Context pairs a source operation with a target block.
/// Clones are always written to the end of the target block.
struct CopyContext {
  Operation *source;
  Block *target;
};

FrameData LowerAsyncBuildContext::cloneFrameAndFirstStateTo(
    FuncOp hotRamp, FuncOp fromFuncOp, FrameData *originalFrameData) {
  FrameData emptyFrame;
  COTypes opaqueCoTypes(builder.getContext(), std::move(emptyFrame),
                        /*promiseType=*/{});
  FrameData frameData;
  /// Augment the hot ramp function signature. We will clone ops from the funcOp
  /// into the hot ramp function.
  SmallVector<Type> inputs;
  SmallVector<ArgConvention> conventions;
  Type closureType = opaqueCoTypes.typeForField(M::KGEN::ClosureState);
  Type callbackType = opaqueCoTypes.typeForField(M::KGEN::CallbackFn);
  inputs.push_back(callbackType);
  conventions.push_back(ArgConvention::ImmReg);
  inputs.push_back(closureType);
  conventions.push_back(ArgConvention::ImmReg);
  llvm::append_range(inputs, fromFuncOp.getArgumentTypes());
  llvm::append_range(
      conventions,
      fromFuncOp.getFuncTypeGenerator().getBody().getArgConventions());

  IRMapping mapping;
  Block &block = hotRamp.getBodyRegion().emplaceBlock();
  for (BlockArgument oldArg : fromFuncOp.getArguments()) {
    Value newArg = block.addArgument(oldArg.getType(), oldArg.getLoc());
    mapping.map(oldArg, newArg);
    frameData.valueToIndexInFrame[newArg] =
        originalFrameData->valueToIndexInFrame[oldArg];
  }

  FuncType signature = FuncType::get(
      FunctionType::get(builder.getContext(), inputs,
                        PointerType::get(opaqueCoTypes.getHeaderType())),
      conventions, fromFuncOp.getFuncTypeGenerator().getBody().getFnEffects(),
      fromFuncOp.getFuncTypeGenerator().getBody().getMetadata(),
      fromFuncOp.getFuncTypeGenerator().getBody().getArgListAttrs());
  hotRamp.setFuncTypeGenerator(
      GeneratorType::get(/*inputParamTypes=*/{}, signature));

  Value callback = hotRamp.getBodyRegion().front().insertArgument(
      (unsigned)0, callbackType, hotRamp->getLoc());
  Value closureState = hotRamp.getBodyRegion().front().insertArgument(
      1, closureType, hotRamp->getLoc());

  SmallVector<CopyContext> paths;
  auto addTargets = [&](ArrayRef<HLCF::ControlFlowTarget> targets,
                        Operation *cfn, Operation *clone,
                        Operation *cloneParent, CopyContext &copyContext) {
    for (HLCF::ControlFlowTarget target : targets) {
      if (target.index.has_value()) {
        unsigned index = target.index.value();
        Block &sourceBlock = cfn->getRegion(index).front();
        Region *cloneRegion = &cloneParent->getRegion(index);
        if (cloneRegion->empty()) {
          Block &block = cloneRegion->emplaceBlock();
          for (Value arg : sourceBlock.getArguments()) {
            Value newArg = block.addArgument(arg.getType(), arg.getLoc());
            mapping.map(arg, newArg);
            frameData.valueToIndexInFrame[newArg] =
                originalFrameData->valueToIndexInFrame[arg];
          }
        }
        paths.push_back({&*sourceBlock.begin(), &cloneRegion->front()});
      } else {
        paths.push_back({cfn->getNextNode(), cloneParent->getBlock()});
      }
    }
  };
  DenseSet<Operation *> visited;
  paths.push_back({&fromFuncOp.getBodyRegion().front().front(),
                   &hotRamp.getBodyRegion().front()});
  while (!paths.empty()) {
    CopyContext copyContext = paths.pop_back_val();
    if (visited.contains(copyContext.source))
      continue;
    visited.insert(copyContext.source);
    builder.setInsertionPointToEnd(copyContext.target);
    for (Operation *operation = copyContext.source; operation != nullptr;
         operation = operation->getNextNode()) {
      /// Clone.
      Operation *clone = builder.cloneWithoutRegions(*operation, mapping);
      mapping.map(operation, clone);
      frameData.opToState[clone] = originalFrameData->opToState[operation];
      for (auto [result, image] :
           llvm::zip(operation->getResults(), clone->getResults())) {
        auto indexMaybe = originalFrameData->valueToIndexInFrame.find(result);
        if (indexMaybe != originalFrameData->valueToIndexInFrame.end())
          frameData.valueToIndexInFrame[image] = indexMaybe->second;
      }
      auto maybeIndex =
          originalFrameData->operationToIndexInFrame.find(operation);
      if (maybeIndex != originalFrameData->operationToIndexInFrame.end())
        frameData.operationToIndexInFrame[clone] = maybeIndex->second;

      /// Traverse.
      if (auto suspendOp = dyn_cast<SuspendOp>(operation)) {
        KGEN::ReturnOp::create(builder);
        Block &block = clone->getRegion(0).emplaceBlock();
        for (Value arg : suspendOp.getBody().front().getArguments())
          mapping.map(arg, block.addArgument(arg.getType(), arg.getLoc()));
        paths.push_back({&suspendOp.getBody().front().front(), &block});
        break;
      }
      if (auto cfn = dyn_cast<HLCF::ControlFlowNode>(operation)) {
        SmallVector<HLCF::ControlFlowTarget> targets;
        SmallVector<Attribute> controlFlowOperands(cfn->getNumOperands(),
                                                   Attribute());
        cfn.getEntryTargets(controlFlowOperands, targets);
        addTargets(targets, cfn, clone, clone, copyContext);
        break;
      }
      if (isa<ReturnOp>(operation)) {
        builder.setInsertionPoint(clone);
        insertCoroutineEnd(builder, callback, closureState);
        break;
      }
      if (isa<UnreachableOp, SuspendEndOp>(operation))
        break;
      if (auto terminator = dyn_cast<HLCF::ControlFlowTerminator>(operation)) {
        SmallVector<HLCF::ControlFlowTarget> targets;
        SmallVector<Attribute> controlFlowTerminatorOperands(
            terminator->getNumOperands(), Attribute());
        terminator.getBranchTargets(controlFlowTerminatorOperands, targets);
        addTargets(
            targets, HLCF::getParentNode(terminator), clone,
            HLCF::getParentNode(cast<HLCF::ControlFlowTerminator>(clone)),
            copyContext);
        break;
      }
    }
  }
  for (Type frameType : originalFrameData->frameTypes)
    frameData.frameTypes.push_back(frameType);
  return frameData;
}

FuncOp LowerAsyncBuildContext::createColdResume(StringRef prefix, FuncOp funcOp,
                                                COTypes &coTypes) {
  builder.setInsertionPoint(funcOp);
  StringAttr resumeName = builder.getStringAttr(prefix + "_resume");
  auto resumeSignature =
      FuncType::get(builder.getContext(),
                    PointerType::get(coTypes.getContinuationType()), {});
  FuncOp resumeFunction = FuncOp::create(
      builder, funcOp->getParentOp()->getLoc(), resumeName, resumeSignature);
  resumeFunction.setCoroutineTypeAttr(
      TypeAttr::get(coTypes.getContinuationType()));
  resumeName = sharedTable.modify(
      [resumeFunction, it = funcOp->getIterator()](SymbolTable &symtab) {
        return symtab.insert(resumeFunction, it);
      });
  auto [errorValue, memoryResultValue] = getErrorAndMemoryValues(funcOp);
  resumeFunction.getBodyRegion().takeBody(funcOp.getBodyRegion());
  insertFrameLoadsStores(resumeFunction, coTypes, errorValue, memoryResultValue,
                         Temp::Cold,
                         /*loadFromFrame=*/true);

  // Insert state updates.
  int susId = 0;
  Value continuation = resumeFunction.getBodyRegion().getArgument(0);
  resumeFunction.walk([&](SuspendOp suspendOp) {
    builder.setInsertionPoint(suspendOp);
    Value newState =
        ParamConstantOp::create(builder, builder.getI32IntegerAttr(++susId));
    Value stateSlot = StructGEPOp::create(builder, continuation, State);
    StoreOp::create(builder, newState, stateSlot);
  });
  return resumeFunction;
}

static void walkVirtualBlocks(ArrayRef<Operation *> virtualBlocks,
                              function_ref<void(Operation *)> callback) {
  for (auto virtualBlock : virtualBlocks) {
    Operation *current = virtualBlock;
    // Frame inserts may have resulted in the virtual block boundary being off
    // by the number of frame operands.
    while (current->getPrevNode()) {
      // A control flow node is a virtual block boundary
      if (isa<HLCF::ControlFlowNode>(current->getPrevNode()))
        break;
      current = current->getPrevNode();
    }

    while (current) {
      Operation *op = current;
      current = current->getNextNode();
      callback(op);
      if (isa<SuspendOp, SuspendEndOp, HLCF::ControlFlowTerminator,
              HLCF::ControlFlowNode>(op))
        break;
    }
  }
}

void LowerAsyncBuildContext::populateHotResumeFrom(FuncOp original,
                                                   FuncOp resumeFunction,
                                                   COTypes &coTypes) {
  auto [errorValue, memoryResultValue] = getErrorAndMemoryValues(original);
  resumeFunction.getBodyRegion().takeBody(original.getBodyRegion());
  insertFrameLoadsStores(resumeFunction, coTypes, errorValue, memoryResultValue,
                         Temp::Hot, /*loadFromFrame=*/true);
  // (1) Collect ops to remove.
  SmallVector<Operation *> opsToRemove;
  walkVirtualBlocks(
      coTypes.getFrameData()->virtualBlocksFirstState, [&](Operation *op) {
        if (isa<HLCF::ControlFlowTerminator>(op)) {
          builder.setInsertionPoint(op);
          for (auto [index, type] : llvm::enumerate(op->getOperandTypes()))
            op->setOperand(index, ParamConstantOp::create(
                                      builder, UninitMemAttr::get(type)));
        } else if (!isa<SuspendOp, SuspendEndOp>(op)) {
          opsToRemove.push_back(op);
        }
      });

  // (2) Replace arguments in parent control flow with constants.
  DenseSet<Operation *> parents;
  for (SuspendOp suspendOp : coTypes.getFrameData()->firstSuspends) {
    Operation *current = suspendOp->getParentOp();
    while (current) {
      Operation *parent = current;
      current = current->getParentOp();
      if (parent == resumeFunction)
        break;
      if (parents.contains(parent))
        break;
      parents.insert(parent);
      assert(coTypes.getFrameData()->opToState.contains(parent) &&
             "The function should not be augmented with control flow ops");
      if (coTypes.getFrameData()->opToState[parent] > 0)
        continue;

      builder.setInsertionPoint(parent);
      for (auto [index, operand] : llvm::enumerate(parent->getOperands()))
        parent->setOperand(index,
                           ParamConstantOp::create(
                               builder, UninitMemAttr::get(operand.getType())));
    }
  }

  // (3) Remove the first state.
  for (Operation *op : llvm::reverse(opsToRemove)) {
    if (isa<HLCF::ControlFlowNode>(op) && parents.contains(op))
      continue;
    op->erase();
  }

  llvm::BitVector args(resumeFunction.getBodyRegion().front().getNumArguments(),
                       true);
  args.reset(0);
  resumeFunction.getBodyRegion().front().eraseArguments(args);
  resumeFunction.setFuncTypeGenerator(GeneratorType::get(
      /*inputParamTypes=*/{},
      FuncType::get(builder.getContext(),
                    resumeFunction.getBodyRegion().front().getArgumentTypes(),
                    resumeFunction.getResultTypes())));
}

FuncOp LowerAsyncBuildContext::createSharedResume(StringRef prefix,
                                                  FuncOp originalAsyncFunc,
                                                  COTypes &coldcoTypes,
                                                  Type hotCoroType) {
  // (1) Create the cold resume.
  FuncOp resumeFunction =
      createColdResume(prefix, originalAsyncFunc, coldcoTypes);

  // (2) Replace coldContType with hotContType.
  mlir::AttrTypeReplacer walker;
  Type coldContType = resumeFunction.getCoroutineType().value();
  walker.addReplacement([coldContType, hotCoroType](Type type) {
    if (type == coldContType)
      return hotCoroType;
    return type;
  });
  walker.recursivelyReplaceElementsIn(resumeFunction, true, true, true);

  // (3) Bitcast the continuation to the cold type in the first state.
  builder.setInsertionPointToStart(&resumeFunction.getBodyRegion().front());
  Value hotStartContinuation = resumeFunction.getArgument(0);
  size_t hotContSize =
      cast<StructType>(
          cast<PointerType>(hotStartContinuation.getType()).getElementType())
          .getElementTypes()
          ->size();
  auto pointerBitcast = PointerBitcastOp::create(
      builder, PointerType::get(coldContType), hotStartContinuation);
  Value coldStartContinuation = pointerBitcast.getResult();
  for (VirtualBlock virtualBlock :
       coldcoTypes.getFrameData()->virtualBlocksFirstState) {
    Operation *current = virtualBlock;
    while (current->getPrevNode()) {
      Operation *prev = current->getPrevNode();
      if (isa<SuspendOp, HLCF::ControlFlowNode>(prev))
        break;
      current = prev;
    }

    while (current) {
      auto gep = dyn_cast<StructGEPOp>(current);
      if (gep && gep.getContainer() == hotStartContinuation) {
        auto indexAttr = cast<IntegerAttr>(gep.getIndex());
        if (static_cast<size_t>(indexAttr.getInt()) >= hotContSize)
          current->setOperand(0, coldStartContinuation);
      }
      Operation *next = current->getNextNode();
      if (next && isa<HLCF::ControlFlowNode, HLCF::ControlFlowTerminator,
                      CO::SuspendOp>(next))
        break;
      current = next;
    }
  }
  return resumeFunction;
}
FuncOp LowerAsyncBuildContext::createHotRamp(StringRef prefix, FuncOp original,
                                             COTypes &coTypes,
                                             FuncOp resumeFunction,
                                             bool takeOriginal) {
  StringAttr hotRampName = builder.getStringAttr(prefix + "_hot_ramp");
  builder.setInsertionPoint(resumeFunction);
  FuncOp hotRamp = FuncOp::create(builder, original->getLoc(), hotRampName,
                                  FuncType::get(builder.getContext(), {}, {}));
  hotRampName = sharedTable.modify(
      [hotRamp, it = resumeFunction->getIterator()](SymbolTable &symtab) {
        return symtab.insert(hotRamp, it);
      });

  // Given (args:A) -> B, create (callback:(P) -> (), closure: P, args:A) -> B
  Value errorValue;
  Value memoryResultValue;
  if (takeOriginal) {
    takeSlicedFirstStateFrom(hotRamp, original);
    std::tie(errorValue, memoryResultValue) = getErrorAndMemoryValues(hotRamp);
    insertFrameLoadsStores(hotRamp, coTypes, errorValue, memoryResultValue,
                           Temp::Hot,
                           /*loadFromFrame=*/false);
  } else {
    FrameData rampFrameData(
        cloneFrameAndFirstStateTo(hotRamp, original, coTypes.getFrameData()));
    COTypes rampCoTypes(builder.getContext(), std::move(rampFrameData),
                        coTypes.getPromiseType());
    std::tie(errorValue, memoryResultValue) = getErrorAndMemoryValues(hotRamp);
    insertFrameLoadsStores(hotRamp, rampCoTypes, errorValue, memoryResultValue,
                           Temp::Hot,
                           /*loadFromFrame=*/false);
  }
  // Check parent region for termination (everything after first suspend was
  // deleted).
  if (!hotRamp.getBodyRegion().front().mightHaveTerminator()) {
    builder.setInsertionPointToEnd(&hotRamp.getBodyRegion().front());
    UnreachableOp::create(builder);
  }

  constexpr unsigned indexOfCoroutine = 0;
  constexpr unsigned indexOfCallback = 1;
  constexpr unsigned indexOfClosure = 2;
  Value callback = hotRamp.getBodyRegion().getArgument(indexOfCallback);
  Value closureState = hotRamp.getBodyRegion().getArgument(indexOfClosure);

  // Introduce continuation and replace argument
  BlockArgument continuationArg =
      hotRamp.getBodyRegion().getArgument(indexOfCoroutine);

  builder.setInsertionPointToStart(&hotRamp.getBodyRegion().front());
  Value continuation = initializeContinuation(hotRamp, resumeFunction, coTypes);
  continuationArg.replaceAllUsesWith(continuation);
  hotRamp.getBodyRegion().front().eraseArgument(indexOfCoroutine);
  hotRamp.setFuncTypeGenerator(GeneratorType::get(
      /*inputParamTypes=*/{},
      FuncType::get(builder.getContext(),
                    hotRamp.getBodyRegion().getArgumentTypes(),
                    PointerType::get(coTypes.getHeaderType()))));

  // Store continuation and closure.
  Value closureSlot = StructGEPOp::create(builder, continuation, ClosureState);
  StoreOp::create(builder, closureState, closureSlot);
  Value callbackSlot = StructGEPOp::create(builder, continuation, CallbackFn);
  StoreOp::create(builder, callback, callbackSlot);

  // Store arguments in frame if used across suspension points.
  for (auto [index, frameSlot] : coTypes.getFrameData()->argsInFrame) {
    Value slot = StructGEPOp::create(builder, continuation, Frame + frameSlot);
    Value image = hotRamp.getArgument(index + 2);
    StoreOp::create(builder, image, slot);
  }

  // Store results/error
  auto setByRefArgument = [&](Value argument, unsigned index) {
    Value slot = StructGEPOp::create(builder, continuation, index);
    Value typedSlot = PointerBitcastOp::create(
        builder, KGEN::PointerType::get(argument.getType()), slot);
    StoreOp::create(builder, argument, typedSlot);
  };
  if (errorValue)
    setByRefArgument(errorValue, AsyncContinuationField::ErrorSlot);
  if (memoryResultValue)
    setByRefArgument(memoryResultValue, AsyncContinuationField::ResultSlot);

  // Return the continuation.
  Value bitcast = PointerBitcastOp::create(
      builder, PointerType::get(coTypes.getHeaderType()), continuation);
  int susId = 0;
  hotRamp.walk([&](Operation *op) {
    if (auto returnOp = dyn_cast<ReturnOp>(op)) {
      returnOp->insertOperands(0, bitcast);
    } else if (auto suspend = dyn_cast<SuspendOp>(op)) {
      // Insert state change
      builder.setInsertionPoint(suspend);
      Value newState =
          ParamConstantOp::create(builder, builder.getI32IntegerAttr(++susId));
      Value stateSlot = StructGEPOp::create(builder, continuation, State);
      StoreOp::create(builder, newState, stateSlot);

      Operation *current = &suspend.getBody().front().getOperations().front();
      while (current) {
        Operation *op = current;
        current = current->getNextNode();
        if (isa<SuspendEndOp>(op))
          break;
        else
          op->moveBefore(suspend);
      }
      suspend->erase();
    }
  });
  return hotRamp;
}

FuncOp LowerAsyncBuildContext::createColdRamp(StringRef prefix,
                                              FuncType originalSignature,
                                              COTypes &coTypes,
                                              FuncOp resumeFunction) {
  StringAttr rampName = builder.getStringAttr(prefix + "_ramp");
  unsigned end = originalSignature.getNumArguments();
  if (originalSignature.isThrows())
    --end;
  if (originalSignature.hasMemoryOnlyResult())
    --end;
  SmallVector<Type> args;
  for (unsigned i = 0; i < end; ++i)
    args.push_back(originalSignature.getArgument(i));
  FunctionType rampFunctionType =
      builder.getFunctionType(args, PointerType::get(coTypes.getHeaderType()));
  auto rampSignature = FuncType::get(rampFunctionType);
  builder.setInsertionPoint(resumeFunction);
  FuncOp rampFunction = FuncOp::create(builder, rampName, rampSignature);
  rampName = sharedTable.modify(
      [rampFunction, it = resumeFunction->getIterator()](SymbolTable &symtab) {
        return symtab.insert(rampFunction, it);
      });
  // Replace coroutine argument with local coroutine
  builder.setInsertionPointToStart(
      &rampFunction.getBodyRegion().emplaceBlock());
  for (Type argument :
       rampFunction.getFuncTypeGenerator().getBody().getArguments())
    rampFunction.getBodyRegion().addArgument(argument, rampFunction.getLoc());
  Value continuation =
      initializeContinuation(rampFunction, resumeFunction, coTypes);
  // Store arguments in frame.
  for (auto [index, argSlot] : coTypes.getFrameData()->argsInFrame) {
    Value arg = rampFunction.getArgument(index);
    Value slot = StructGEPOp::create(builder, continuation, Frame + argSlot);
    StoreOp::create(builder, arg, slot);
  }
  Value headerTypedContinuation = PointerBitcastOp::create(
      builder, PointerType::get(coTypes.getHeaderType()), continuation);
  ReturnOp::create(builder, headerTypedContinuation);
  return rampFunction;
}

COTypes LowerAsyncBuildContext::calculateFrame(FuncOp original,
                                               bool includeArgs) {
  auto [errorValue, memoryResultValue] = getErrorAndMemoryValues(original);
  // The transform function clones ops whose values should not be stored in
  // the frame. This includes constants and pointer offsets.
  auto transform = [&](FuncOp originalFunction,
                       DenseMap<Operation *, int> &opToState) {
    cloneFrameArgs(original, builder, dominanceInfo, originalFunction,
                   opToState);
  };
  FrameData frameData(original, dominanceInfo, errorValue, memoryResultValue,
                      transform, /*isHot=*/!includeArgs);
  COTypes coTypes(
      builder.getContext(), std::move(frameData),
      StructType::get(original.getContext(), original.getResultTypes()));
  return coTypes;
}

Coroutine LowerAsyncBuildContext::createCoroutine(FuncOp originalAsyncFunc,
                                                  Temp temperature) {
  StringRef prefix = originalAsyncFunc.getSymName();
  if (temperature == Temp::Cold) {
    FuncType originalSignature =
        originalAsyncFunc.getFuncTypeGenerator().getBody();
    COTypes coTypes(calculateFrame(originalAsyncFunc, /*includeArgs=*/true));
    FuncOp resumeFunction =
        createColdResume(prefix, originalAsyncFunc, coTypes);
    FuncOp coldRamp =
        createColdRamp(prefix, originalSignature, coTypes, resumeFunction);
    Coroutine coro(resumeFunction, {}, coldRamp, coTypes.getContinuationType());
    originalAsyncFunc->erase();
    return coro;
  } else if (temperature == Temp::Hot) {
    COTypes hotCoTypes(
        calculateFrame(originalAsyncFunc, /*includeArgs=*/false));
    builder.setInsertionPoint(originalAsyncFunc);
    StringAttr resumeName = builder.getStringAttr(prefix + "_resume");
    auto resumeSignature =
        FuncType::get(builder.getContext(),
                      PointerType::get(hotCoTypes.getContinuationType()), {});
    FuncOp resumeFunction =
        FuncOp::create(builder, originalAsyncFunc->getParentOp()->getLoc(),
                       resumeName, resumeSignature);
    resumeFunction.setCoroutineTypeAttr(
        TypeAttr::get(hotCoTypes.getContinuationType()));
    resumeName = sharedTable.modify(
        [resumeFunction, it = originalAsyncFunc->getIterator()](
            SymbolTable &symtab) { return symtab.insert(resumeFunction, it); });
    FuncOp hotRamp =
        createHotRamp(prefix, originalAsyncFunc, hotCoTypes, resumeFunction,
                      /*takeOriginal=*/false);
    populateHotResumeFrom(originalAsyncFunc, resumeFunction, hotCoTypes);
    Coroutine coro(resumeFunction, hotRamp, {},
                   hotCoTypes.getContinuationType());
    originalAsyncFunc->erase();
    return coro;
  } else if (temperature == Temp::Both) {
    FuncType originalSignature =
        originalAsyncFunc.getFuncTypeGenerator().getBody();
    FuncOp clone = originalAsyncFunc.clone();
    COTypes hotCoTypes(calculateFrame(clone, /*includeArgs=*/false));
    COTypes coTypes(calculateFrame(originalAsyncFunc, /*includeArgs=*/true));
    FuncOp sharedResume = createSharedResume(prefix, originalAsyncFunc, coTypes,
                                             hotCoTypes.getContinuationType());
    FuncOp hotRamp = createHotRamp(prefix, clone, hotCoTypes, sharedResume,
                                   /*takeOriginal=*/true);
    FuncOp coldRamp =
        createColdRamp(prefix, originalSignature, coTypes, sharedResume);
    Coroutine coro(sharedResume, hotRamp, coldRamp,
                   hotCoTypes.getContinuationType());
    clone->erase();
    originalAsyncFunc->erase();
    return coro;
  }
  llvm_unreachable("temperature must be hot, cold, or both");
}

void LowerAsyncBuildContext::insertFrameLoadsStores(
    FuncOp resumeFunction, COTypes &coTypes, Value errorValue,
    Value memoryResultValue, Temp temp, bool loadFromFrame) {
  const FrameData *frameData = coTypes.getFrameData();
  builder.setInsertionPointToStart(&resumeFunction.getBodyRegion().front());
  resumeFunction.getBodyRegion().insertArgument(
      (unsigned)0, PointerType::get(coTypes.getContinuationType()),
      resumeFunction->getLoc());
  Value continuation = resumeFunction.getArgument(0);
  // For each new operand, extract operand from frame if it was defined in
  // previous state. For each op, store in frame if it is used downstream
  // across a suspension point. For each block, store arguments if they are
  // accessed across suspnsion points.
  FrameVariables frameVariables(builder, frameData, errorValue,
                                memoryResultValue);
  SmallVector<std::pair<Region *, Block::iterator>> regionsToProcess;
  regionsToProcess.push_back({&resumeFunction.getBodyRegion(),
                              resumeFunction.getBodyRegion().front().begin()});
  SmallVector<Operation *> opsToDelete;
  while (!regionsToProcess.empty()) {
    auto [parentRegion, begin] = regionsToProcess.back();
    regionsToProcess.pop_back();
    // Process the ops of a region.
    Operation *current = &*begin;
    while (current) {
      Operation *op = current;
      current = op->getNextNode();

      storeOpInFrameIfNeeded(frameData, op, continuation, opsToDelete);

      // Extract arguments from operands if needed.
      auto useStateMaybe = frameData->opToState.find(op);
      if (useStateMaybe == frameData->opToState.end())
        continue;
      int useState = useStateMaybe->second;
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        auto entry = frameData->valueToIndexInFrame.find(operand);
        if (entry != frameData->valueToIndexInFrame.end() ||
            operand == errorValue || operand == memoryResultValue) {
          // Only extract the value out of the frame if the def was in another
          // state. Block arguments have been cached in frameVariables because
          // region block arguments are processed before body ops.
          int defState = temp == Temp::Hot ? 0 : -1;
          bool isStackAlloc = false;
          Operation *definingOp = operand.getDefiningOp();
          if (definingOp) {
            defState = frameData->opToState.at(definingOp);
            isStackAlloc = isa<StackAllocationOp>(definingOp);
          }

          // Stack allocated variables are an exception. They are pulled from
          // the frame regardless of state status because the stack allocation
          // is replaced with a frame allocation.
          if (!isStackAlloc) {
            if (defState == useState)
              continue;
            if (!loadFromFrame)
              continue;
          }
          Value image = frameVariables.getFrameValueForOperand(
              continuation, operand, op, useState);
          op->setOperand(index, image);
        }
      }

      // Store arguments of block if needed.
      for (Region &region : op->getRegions()) {
        // Start processing at the first op. Blocks cannot be empty because
        // they must be terminated.
        Operation *firstOp = &*region.front().begin();
        regionsToProcess.push_back({&region, firstOp->getIterator()});
        storeBlockArgumentsInFrame(region.front(), firstOp, frameData,
                                   continuation, frameVariables);
      }
    }
  }

  // Arguments will be removed after ramp generation in the case of hot
  // coroutines.
  if (temp == Temp::Cold) {
    llvm::BitVector args(
        resumeFunction.getBodyRegion().front().getNumArguments(), true);
    args.reset(0);
    resumeFunction.getBodyRegion().front().eraseArguments(args);
  }

  resumeFunction.walk([&](Operation *op) {
    if (auto returnOp = dyn_cast<ReturnOp>(op)) {
      lowerReturn(returnOp, continuation);
    } else if (auto suspend = dyn_cast<SuspendOp>(op)) {
      lowerSuspensionPoint(suspend, continuation, coTypes);
    } else if (auto hotInvoke = dyn_cast<HotInvokeOp>(op)) {
      // Hot invoke lowers to a call to the hot ramp function.
      // The hot ramp function's first argument is the callback (resume
      // function). The hot ramp function's second argument is the closure state
      // (this continuation). The hot invoke operation will be replaced with the
      // kgen.call op to the ramp function once we have generated all the ramp
      // functions.
      builder.setInsertionPoint(hotInvoke);
      SmallVector<Value> operands;
      Value resumeFunction = LoadOp::create(
          builder, StructGEPOp::create(builder, continuation, ResumeFunction));
      Value operand0 = PointerBitcastOp::create(
          builder, coTypes.getResumeSignatureType(), resumeFunction);
      Value operand1 = PointerBitcastOp::create(
          builder, coTypes.typeForField(ClosureState), continuation);
      hotInvoke->insertOperands(0, operand0);
      hotInvoke->insertOperands(1, operand1);
    }
  });

  for (auto op : opsToDelete)
    op->erase();
}

Value LowerAsyncBuildContext::initializeContinuation(FuncOp rampFunction,
                                                     FuncOp resumeFunction,
                                                     COTypes &coTypes) {
  Type continuationType = coTypes.getContinuationType();
  // Allocate memory for continuation.
  std::optional<int64_t> size =
      DataLayoutInterface::getTypeStoreSize(targetInfoAttr, continuationType);
  std::optional<int64_t> align =
      DataLayoutInterface::getTypeABIAlign(targetInfoAttr, continuationType);
  Value sizeOf = mlir::index::ConstantOp::create(builder, size.value());
  Value alignOf = mlir::index::ConstantOp::create(builder, align.value());

  Value continuation = AlignedAllocOp::create(
      builder, PointerType::get(continuationType), ValueRange{alignOf, sizeOf});

  // Initialize state to 0.
  Value zero = ParamConstantOp::create(builder, builder.getI32IntegerAttr(0));
  Value stateSlot = StructGEPOp::create(builder, continuation, State);
  StoreOp::create(builder, zero, stateSlot);

  // Store resume function.
  Value resumeFunctionSlot =
      StructGEPOp::create(builder, continuation, ResumeFunction);
  Value functionPointer =
      CreateClosureOp::create(builder, SymbolConstantAttr::get(resumeFunction));
  functionPointer = PointerBitcastOp::create(
      builder, coTypes.typeForField(ResumeFunction), functionPointer);
  StoreOp::create(builder, functionPointer, resumeFunctionSlot);
  return continuation;
}

ReturnOp LowerAsyncBuildContext::lowerReturn(ReturnOp returnOp,
                                             Value continuation) {
  builder.setInsertionPoint(returnOp);
  if (returnOp->getNumOperands()) {
    // Replace ReturnOps with set result.
    Value promiseSlot = StructGEPOp::create(builder, continuation, Promise);
    for (auto [idx, value] : llvm::enumerate(returnOp.getOperands())) {
      StoreOp::create(builder, value,
                      StructGEPOp::create(builder, promiseSlot, idx));
    }
    auto result = ReturnOp::create(builder);
    returnOp->erase();
    return result;
  }
  return returnOp;
}

void LowerAsyncBuildContext::lowerSuspensionPoint(CO::SuspendOp suspend,
                                                  Value continuation,
                                                  COTypes &coTypes) {
  // Replace uses of the suspend argument with the continuation.
  Region &body = suspend.getBody();
  if (body.getArguments().empty())
    return;
  if (!body.getArgument(0).use_empty()) {
    builder.setInsertionPointToStart(&suspend.getBody().front());
    Value header = PointerBitcastOp::create(
        builder, PointerType::get(coTypes.getHeaderType()), continuation);
    body.getArgument(0).replaceAllUsesWith(header);
  }
  body.eraseArgument(0);
}

void LowerAsyncBuildContext::storeBlockArgumentsInFrame(
    Block &block, Operation *key, const FrameData *frameData,
    Value continuation, FrameVariables &frameVariables) {
  if (block.getNumArguments() == 0)
    return;
  builder.setInsertionPointToStart(&block);
  int frameValueState = frameData->opToState.at(key);
  for (BlockArgument argument : block.getArguments()) {
    auto entry = frameData->valueToIndexInFrame.find(
        key->getParentRegion()->getArgument(argument.getArgNumber()));
    if (entry == frameData->valueToIndexInFrame.end())
      continue;
    Value dataSlot =
        StructGEPOp::create(builder, continuation, Frame + entry->getSecond());
    StoreOp::create(builder, argument, dataSlot);
    frameVariables.overwriteValue(frameValueState, argument);
  }
}

void LowerAsyncBuildContext::storeOpInFrameIfNeeded(
    const FrameData *frameData, Operation *op, Value continuation,
    SmallVector<Operation *> &opsToDelete) {
  if (isa<StackAllocLifetimeEndOp, StackAllocLifetimeStartOp>(op)) {
    int index = 0;
    for (Value value : op->getOperands()) {
      auto entry =
          frameData->operationToIndexInFrame.find(value.getDefiningOp());
      if (entry != frameData->operationToIndexInFrame.end())
        op->eraseOperand(index);
      else
        ++index;
    }
    if (op->getNumOperands() == 0)
      opsToDelete.push_back(op);
    return;
  }
  auto entry = frameData->operationToIndexInFrame.find(op);
  if (entry != frameData->operationToIndexInFrame.end()) {
    if (isa<StackAllocationOp>(op)) {
      opsToDelete.push_back(op);
      return;
    }
    builder.setInsertionPointAfter(op);
    [[maybe_unused]] Type frameEntryType =
        frameData->frameTypes[entry->getSecond()];
    assert(frameEntryType == op->getResultTypes().front() &&
           "The frame type slot does not match the value");
    assert(op->getNumResults() == 1 && "TODO: support multiple results");
    Value dataSlot =
        StructGEPOp::create(builder, continuation, Frame + entry->getSecond());
    StoreOp::create(builder, op->getResult(0), dataSlot);
  }
}

//===----------------------------------------------------------------------===//
// CoTypes
//===----------------------------------------------------------------------===//

COTypes::COTypes(MLIRContext *cxt, FrameData &&frameData,
                 StructType promiseType)
    : cxt(cxt), frameData(std::move(frameData)), promiseType(promiseType) {
  opaquePointerType = PointerType::get(KGEN::NoneType::get(cxt));
  SmallVector<Type> inputs;
  SmallVector<Type> results;
  inputs.push_back(opaquePointerType);
  FunctionType resumeFunctionType = FunctionType::get(cxt, inputs, results);
  resumeSignatureType =
      FuncTypeGeneratorType::get(/*inputParamTypes=*/{}, resumeFunctionType);
  FunctionType callbackFunctionType =
      FunctionType::get(cxt, opaquePointerType, results);
  callbackSignature =
      FuncTypeGeneratorType::get(/*inputParamTypes=*/{}, callbackFunctionType);

  // Build Continuation Type.
  size_t size = Promise;
  SmallVector<Type> types(size);
  types[State] = typeForField(State);
  types[ResumeFunction] = typeForField(ResumeFunction);
  types[CallbackFn] = typeForField(CallbackFn);
  types[ClosureState] = typeForField(ClosureState);
  types[ErrorSlot] = typeForField(ErrorSlot);
  types[ResultSlot] = typeForField(ResultSlot);

  // Header type omits the variable sized frame and promise.
  headerType = StructType::get(cxt, types);

  // Only create continuationType if promiseType is valid. Some callers
  // pass a null promiseType when they only need the headerType.
  if (promiseType) {
    types.push_back(typeForField(Promise));
    for (auto [index, frameVariableType] :
         llvm::enumerate(frameData.frameTypes))
      types.push_back(frameVariableType);
    continuationType = StructType::get(cxt, types);
  }
}

//===----------------------------------------------------------------------===//
// FrameVariables
//===----------------------------------------------------------------------===//

Value FrameVariables::getFrameValueForOperand(Value continuation, Value operand,
                                              Operation *opWithUse,
                                              int useState) {
  auto entry = frameData->valueToIndexInFrame.find(operand);
  if (entry == frameData->valueToIndexInFrame.end() && operand != errorValue &&
      operand != resultValue)
    return {};
  DenseMap<int, Value> &frameVariablesForValue =
      frameVariables.try_emplace(operand).first->getSecond();

  // Reuse existing extracted value if possible.
  Value image;
  auto existingImage = frameVariablesForValue.find(useState);
  bool wasExtractedInThisState = existingImage != frameVariablesForValue.end();
  // TODO: can this be parent region also?
  bool wasExtractedInThisRegion =
      wasExtractedInThisState && existingImage->getSecond().getParentRegion() ==
                                     opWithUse->getParentRegion();
  if (wasExtractedInThisRegion) {
    image = existingImage->getSecond();
  } else {
    builder.setInsertionPoint(opWithUse);
    if (operand == errorValue) {
      Value dataSlot = StructGEPOp::create(builder, continuation, ErrorSlot);
      Value ptr = LoadOp::create(builder, dataSlot);
      image = PointerBitcastOp::create(builder, errorValue.getType(), ptr);
    } else if (operand == resultValue) {
      Value dataSlot = StructGEPOp::create(builder, continuation, ResultSlot);
      Value ptr = LoadOp::create(builder, dataSlot);
      image = PointerBitcastOp::create(builder, resultValue.getType(), ptr);
    } else {
      unsigned frameIndex = entry->getSecond();
      Value dataSlot =
          StructGEPOp::create(builder, continuation, Frame + frameIndex);
      if (operand.getDefiningOp() &&
          isa<StackAllocationOp>(operand.getDefiningOp())) {
        auto stackAlloc = dyn_cast<StackAllocationOp>(operand.getDefiningOp());
        if (cast<IntegerAttr>(stackAlloc.getCount()).getInt() == 1) {
          image = dataSlot;
        } else {
          image =
              PointerBitcastOp::create(builder, stackAlloc.getType(), dataSlot);
        }
      } else {
        image = LoadOp::create(builder, dataSlot);
      }
    }
    if (wasExtractedInThisState)
      frameVariablesForValue.erase(existingImage);
    frameVariablesForValue.insert({useState, image});
  }
  return image;
}

void FrameVariables::overwriteValue(int state, Value value) {
  DenseMap<int, Value> &frameVariablesForValue =
      frameVariables.try_emplace(value).first->getSecond();
  auto existing = frameVariablesForValue.find(state);
  if (existing != frameVariablesForValue.end())
    frameVariablesForValue.erase(existing);
  frameVariablesForValue.insert({state, value});
}

//===----------------------------------------------------------------------===//
// FrameData
//===----------------------------------------------------------------------===//

int FrameData::getDefinitionStateForValue(Value operand, bool isHot) const {
  Operation *definingOp = operand.getDefiningOp();
  // Initialize state to the entry state.
  int defState = isHot ? 0 : -1;
  if (!definingOp) {
    BlockArgument blockArgument = cast<BlockArgument>(operand);
    // We always store the function arguments in the frame because they
    // originate in the ramp function.
    Operation *parentOp = blockArgument.getOwner()->getParentOp();
    if (isa<FuncOp>(parentOp))
      return defState;

    Operation *firstOp = &*blockArgument.getOwner()->begin();
    defState = opToState.at(firstOp);
    // In a loop we want to compare state before entering loop to the use
    // inside the body.
    if (isa<HLCF::LoopOp>(parentOp))
      defState = opToState.at(parentOp);
    return defState;
  }
  return opToState.at(definingOp);
}

void FrameData::updateVirtualBlock(VirtualBlock virtualBlock, int newState) {
  Operation *op = virtualBlock;
  int state = newState;
  while (op) {
    if (isa<SuspendEndOp>(op))
      ++state;
    int &oldState = opToState[op];
    if (oldState < state)
      oldState = state;
    op = op->getNextNode();
    int &nextOldState = opToState[op];
    // respect control flow boundary
    if (op && isa<HLCF::ControlFlowNode>(op)) {
      if (nextOldState < state)
        nextOldState = state;
      break;
    }
  }
}

using PathContainer = SmallVector<Operation *>;
struct PathInfo {
  PathInfo(VirtualBlock v) : virtualBlock(v), state(0) {}
  PathInfo(VirtualBlock v, int state, PathContainer const &parent, bool hitSus)
      : virtualBlock(v), path(parent), state(state), hitSus(hitSus) {}
  int existsAt(VirtualBlock virtualBlock) const;
  VirtualBlock virtualBlock;
  PathContainer path;
  int state;
  bool hitSus = false;
};

int PathInfo::existsAt(VirtualBlock virtualBlock) const {
  for (auto [i, node] : llvm::enumerate(path)) {
    if (node == virtualBlock)
      return i;
  }
  return -1;
}

FrameData::FrameData(
    FuncOp originalFunction, mlir::DominanceInfo &domInfo, Value errorValue,
    Value resultValue,
    function_ref<void(FuncOp, DenseMap<Operation *, int> &)> transform,
    bool isHot) {
  // Calculate Control Flow Graph.
  // We need to know the predecessors of each region so that
  // we don't process a region until all its predecessors have
  // been processed.
  DenseMap<VirtualBlock, SmallVector<VirtualBlock>> predecessors;
  {
    SmallVector<Region *> regions;
    DenseSet<Region *> visited;
    regions.push_back(&originalFunction.getBodyRegion());

    auto pushSuccessors =
        [&](SmallVector<HLCF::ControlFlowTarget> const &targets,
            Operation *controlFlowNode, Operation *controlFlowParent,
            Operation *predecessor) {
          // For the first op of the region of each target, add the control flow
          // node as a predecessor
          for (HLCF::ControlFlowTarget target : targets) {
            VirtualBlock successor;
            if (target.index.has_value()) {
              Region *succRegion =
                  &controlFlowParent->getRegion(target.index.value());
              successor = &*succRegion->front().begin();
              regions.push_back(succRegion);
            } else {
              successor = controlFlowParent->getNextNode();
            }
            predecessors[successor].push_back(controlFlowNode);
          }
        };

    // There are three types of ops that form virtual block boundaries within a
    // region: control flow nodes, control flow terminators, and coroutine
    // awaits.
    while (!regions.empty()) {
      Region *region = regions.back();
      regions.pop_back();
      if (visited.contains(region))
        continue;
      visited.insert(region);
      Operation *lastControlFlowNode = nullptr;
      CO::SuspendOp lastAwait = nullptr;
      for (Operation &op : region->front().getOperations()) {
        if (isa<ReturnOp, UnreachableOp>(op))
          continue;

        // add the control flow terminator as a predecessor to the first op of a
        // target block.
        if (auto controlFlowTerminator =
                dyn_cast<HLCF::ControlFlowTerminator>(op)) {
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowTerminatorOperands(
              controlFlowTerminator->getNumOperands(), Attribute());
          controlFlowTerminator.getBranchTargets(controlFlowTerminatorOperands,
                                                 targets);
          Operation *predecessor =
              lastControlFlowNode
                  ? lastControlFlowNode->getNextNode()
                  : &*controlFlowTerminator->getParentRegion()->front().begin();
          if (lastAwait) {
            if (domInfo.dominates(predecessor, lastAwait))
              predecessor = lastAwait->getNextNode();
          }
          pushSuccessors(targets, controlFlowTerminator,
                         getParentNode(controlFlowTerminator), predecessor);
        }
        if (auto controlFlowNode = dyn_cast<HLCF::ControlFlowNode>(op)) {
          lastControlFlowNode = controlFlowNode;
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowNodeOperands(
              controlFlowNode->getNumOperands(), Attribute());
          controlFlowNode.getEntryTargets(controlFlowNodeOperands, targets);
          pushSuccessors(targets, controlFlowNode, controlFlowNode,
                         &*controlFlowNode->getParentRegion()->front().begin());
        }
        if (auto suspend = dyn_cast<CO::SuspendOp>(op)) {
          Operation *next = suspend->getNextNode();
          if (!next)
            continue;
          lastAwait = suspend;
          predecessors[&*suspend.getBody().front().begin()].push_back(suspend);
          // Terminator is used because that will trigger updated state.
          predecessors[next].push_back(
              suspend.getBody().front().getTerminator());
          regions.push_back(&suspend.getBody());
        }
      }
    }
  }
  // Calculate the state of each op.
  {
    SmallVector<PathInfo> paths;
    VirtualBlock initial = &*originalFunction.getBodyRegion().front().begin();
    paths.push_back({initial});
    auto pushSuccessors =
        [&](SmallVector<HLCF::ControlFlowTarget> const &targets,
            Operation *controlFlowVirtualBlock, Operation *controlFlowParent,
            PathContainer &path, int state, bool hitSus) {
          for (HLCF::ControlFlowTarget target : targets) {
            if (target.index.has_value()) {
              auto o = &*controlFlowParent->getRegion(target.index.value())
                             .front()
                             .begin();
              paths.push_back({o, state, path, hitSus});
            } else {
              paths.push_back(
                  {controlFlowParent->getNextNode(), state, path, hitSus});
            }
          }
        };

    int j = 0;
    while (!paths.empty()) {
      if (j > 20000)
        llvm_unreachable("infinite loop");

      ++j;
      VirtualBlock virtualBlock = paths.back().virtualBlock;
      int indexOfMe = paths.back().existsAt(virtualBlock);
      PathContainer path = std::move(paths.back().path);
      int state = paths.back().state;
      bool hitSus = paths.back().hitSus;
      paths.pop_back();

      auto recordedStatePtr = opToState.find(virtualBlock);
      if (recordedStatePtr != opToState.end() &&
          state <= recordedStatePtr->getSecond())
        continue;

      for (auto pred : predecessors[virtualBlock]) {
        if (domInfo.dominates(virtualBlock, pred))
          continue;
        auto predPtr = opToState.find(pred);
        if (predPtr != opToState.end()) {
          if (predPtr->getSecond() > state)
            state = predPtr->getSecond();
        }
      }

      // We have reached a cycle. Terminate.
      if (indexOfMe > -1)
        continue;
      path.push_back(virtualBlock);

      // Iterate through each op in this virtual block to register its state.
      // The boundaries of a node are defined by awaits, control
      // flow nodes, and control flow terminators.
      Operation *current = virtualBlock;
      while (current) {
        Operation *op = current;
        current = op->getNextNode();
        if (auto susEnd = dyn_cast<SuspendEndOp>(op)) {
          Operation *next = susEnd->getParentOp()->getNextNode();
          ++state;
          if (next)
            paths.push_back({next, state, path, hitSus});
          opToState[op] = state;
          break;
        }
        opToState[op] = state;

        if (auto suspend = dyn_cast<CO::SuspendOp>(op)) {
          if (!hitSus)
            firstSuspends.insert(suspend);
          paths.push_back(
              {&*suspend.getBody().front().begin(), state, path, true});
          break;
        }
        if (isa<ReturnOp, UnreachableOp>(op))
          continue;
        if (auto controlFlowNode = dyn_cast<HLCF::ControlFlowNode>(op)) {
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowNodeOperands(
              controlFlowNode->getNumOperands(), Attribute());
          controlFlowNode.getEntryTargets(controlFlowNodeOperands, targets);
          pushSuccessors(targets, controlFlowNode, controlFlowNode, path, state,
                         hitSus);
          break;
        }
        if (auto controlFlowTerminator =
                dyn_cast<HLCF::ControlFlowTerminator>(op)) {
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowTerminatorOperands(
              controlFlowTerminator->getNumOperands(), Attribute());
          controlFlowTerminator.getBranchTargets(controlFlowTerminatorOperands,
                                                 targets);
          pushSuccessors(targets, controlFlowTerminator,
                         getParentNode(controlFlowTerminator), path, state,
                         hitSus);
          break;
        }
      }
    }
  }

  auto propagateChildSuspoints = [&]() -> bool {
    bool wasChange = false;
    for (auto [virtualBlock, preds] : predecessors) {
      auto stateMaybe = opToState.find(virtualBlock);
      if (stateMaybe == opToState.end()) {
        opToState[virtualBlock] = -1;
        continue;
      }

      int initialState = stateMaybe->second;
      int postState = initialState;
      int stateAtContinue = 0;
      int smallestPredState = initialState;
      for (Operation *pred : preds) {
        int predState = opToState[pred];
        if (domInfo.dominates(virtualBlock, pred)) {
          if (predState > stateAtContinue && predState > postState)
            stateAtContinue = predState;
          continue;
        }
        if (predState > postState)
          postState = predState;
        if (predState < smallestPredState)
          smallestPredState = predState;
      }
      wasChange = wasChange || (initialState != postState);
      if (initialState != postState)
        updateVirtualBlock(virtualBlock, postState);

      // Insert a new state at the parent because a child with a suspension
      // point branches to it.
      if (stateAtContinue > 0 && smallestPredState == initialState) {
        int newState = initialState + 1;
        for (auto &[_, state] : opToState) {
          if (state >= newState)
            ++state;
        }
        wasChange = true;
        updateVirtualBlock(virtualBlock, newState);
      }
    }
    return wasChange;
  };

  // FIXED POINT:
  // Update until for every path:
  // (1) if A -> B and A dominates B then state(A) >= state(B)
  // (2) if B -> A and A dominates B and state(A) < state(B), then if there is
  // a predecessor C of A unreachable from B then state(C) < state(A)
  // Condition (2) is achieved by
  // inserting a parent state. Note that intuitively this corresponds to a case
  // of a child cycle with a suspension point branching to a parent cycle.
  bool stateChanged = true;
  while (stateChanged)
    stateChanged = propagateChildSuspoints();

  transform(originalFunction, opToState);

  // Calculate the frame. Whenever there is a use whose def lives in another
  // state, it must be added to the frame. We need to store the location of the
  // value in the frame so that when we generate the resume function the frame
  // values can be extracted.
  auto addToFrame = [&](Type frameVariableType, Operation *definingOp) {
    unsigned index = frameTypes.size();
    frameTypes.push_back(frameVariableType);
    operationToIndexInFrame.insert({definingOp, index});
  };
  auto stackAllocationFrameType =
      [](StackAllocationOp stackAllocation) -> Type {
    int64_t count = cast<IntegerAttr>(stackAllocation.getCount()).getInt();
    if (count == 1) {
      return stackAllocation.getType().getElementType();
    } else {
      return POP::ArrayType::get(stackAllocation.getCount(),
                                 stackAllocation.getType().getElementType());
    }
  };
  SmallVector<Value> coldCoroTypes;
  originalFunction.walk([&](Operation *operation) {
    int useState = opToState[operation];
    if (StackAllocationOp stackAllocationOp =
            dyn_cast<StackAllocationOp>(operation)) {
      if (!stackAllocationOp.getMarkedLifetimes()) {
        Operation *terminator =
            stackAllocationOp->getParentRegion()->front().getTerminator();
        int endState = isa<SuspendEndOp>(terminator) ? opToState[terminator] - 1
                                                     : opToState[terminator];
        if (endState > useState)
          addToFrame(stackAllocationFrameType(stackAllocationOp),
                     stackAllocationOp);
      }
    }

    for (Value operand : operation->getOperands()) {
      // Results and Errors are stored in the header of the continuation, not in
      // the frame.
      if (operand == errorValue || operand == resultValue)
        continue;
      if (valueToIndexInFrame.contains(operand))
        continue;

      // Add to frame if the value was defined in a previous state.
      int defState = getDefinitionStateForValue(operand, isHot);
      if (defState != useState) {
        bool isArgument = false;
        if (auto blockArg = dyn_cast<BlockArgument>(operand))
          isArgument =
              blockArg.getParentBlock()->getParentOp() == originalFunction;

        unsigned index = isArgument ? coldCoroTypes.size() : frameTypes.size();
        if (Operation *definingOp = operand.getDefiningOp()) {
          if (auto stackAllocation = dyn_cast<StackAllocationOp>(definingOp)) {
            // Stack allocations without marked lifetimes are checked for frame
            // membership at definition site.
            if (stackAllocation.getMarkedLifetimes()) {
              addToFrame(stackAllocationFrameType(stackAllocation), definingOp);
            } else {
              valueToIndexInFrame.insert(
                  {operand, operationToIndexInFrame[definingOp]});
              continue;
            }
          } else {
            addToFrame(operand.getType(), definingOp);
          }
        } else {
          if (isArgument)
            coldCoroTypes.push_back(operand);
          else
            frameTypes.push_back(operand.getType());
        }
        valueToIndexInFrame.insert({operand, index});
      }
    }
  });

  // Insert used arguments at end of frame.
  unsigned offset = frameTypes.size();
  for (auto [index, functionArg] : llvm::enumerate(coldCoroTypes)) {
    frameTypes.push_back(functionArg.getType());
    unsigned newIndex = offset + index;
    valueToIndexInFrame[functionArg] = newIndex;
  }
  for (auto [index, functionArg] :
       llvm::enumerate(originalFunction.getRegion().front().getArguments())) {
    auto frameSlotMaybe = valueToIndexInFrame.find(functionArg);
    if (frameSlotMaybe == valueToIndexInFrame.end())
      continue;
    argsInFrame.push_back(ArgInFrame(index, frameSlotMaybe->second));
  }

  // Remember the virtual ops in the first state for hot start resume
  // generation.
  for (auto &virtualBlockAndList : predecessors) {
    VirtualBlock virtualBlock = virtualBlockAndList.first;
    if (opToState[virtualBlock] == 0) {
      // Stack allocations used across frames will be deleted.
      if (isa<StackAllocationOp>(virtualBlock) &&
          valueToIndexInFrame.contains(virtualBlock->getResults().front()))
        virtualBlocksFirstState.push_back(virtualBlock->getNextNode());
      else
        virtualBlocksFirstState.push_back(virtualBlock);
    }
  }
  // If a dominates b then a should appear first.
  llvm::sort(
      virtualBlocksFirstState.begin(), virtualBlocksFirstState.end(),
      [&](Operation *a, Operation *b) { return domInfo.dominates(a, b); });
}

//===----------------------------------------------------------------------===//
// LowerAsyncFunctionsPass
//===----------------------------------------------------------------------===//

static Operation *findNearestCommonAncestor(mlir::DominanceInfo &domInfo,
                                            Operation *lhs, Operation *rhs) {
  auto findOpInCommonRegion = [](Operation *lhs, Operation *rhs) {
    Region *currentRegion = rhs->getParentRegion();
    while (!lhs->getParentRegion()->isAncestor(currentRegion))
      lhs = lhs->getParentOp();
    return lhs;
  };
  Operation *lhsCommon = findOpInCommonRegion(lhs, rhs);
  Operation *rhsCommon = findOpInCommonRegion(rhs, lhs);
  return domInfo.dominates(lhsCommon, rhsCommon) ? lhsCommon : rhsCommon;
}

void LowerAsyncBuildContext::preprocessAsyncFunction(
    FuncOp funcOp, mlir::DominanceInfo &domInfo,
    DenseMap<SymbolConstantAttr, Temp> &temperatures) {

  // Preprocess the function to
  // (1) move stack allocation ops as close to their first use as possible.
  // (2) insert suspension points around hot invokes
  // (3) update the temperatures of calls to async functions
  SmallVector<StackAllocationOp> allocs;
  funcOp.walk([&](Operation *op) {
    if (auto alloc = dyn_cast<StackAllocationOp>(op))
      allocs.push_back(alloc);
    else if (auto hotInvoke = dyn_cast<HotInvokeOp>(op)) {
      builder.setInsertionPoint(op);
      auto suspendOp = SuspendOp::create(builder);
      Block &block = suspendOp->getRegion(0).emplaceBlock();
      builder.setInsertionPointToStart(&block);
      // Partially lower the hot invoke by setting the result type to a
      // Coroutine. This allows us to access the coroutine after the suspension
      // point so we can replace the results properly.
      auto partiallyLoweredHotInvoke = HotInvokeOp::create(
          builder, CO::CoroutineType::get(builder.getContext()),
          hotInvoke.getCallee(), hotInvoke.getCalleeOperands());
      SuspendEndOp::create(builder);
      builder.setInsertionPointAfter(suspendOp);
      auto results =
          GetResultsOp::create(builder, hotInvoke.getResultTypes(),
                               partiallyLoweredHotInvoke->getResult(0));
      for (auto [result, image] :
           llvm::zip(hotInvoke.getResults(), results.getResults()))
        result.replaceAllUsesWith(image);
      hotInvoke->erase();
      SymbolConstantAttr callee =
          cast<SymbolConstantAttr>(partiallyLoweredHotInvoke.getCallee());
      auto maybe = temperatures.find(callee);
      if (maybe == temperatures.end())
        temperatures[callee] = Temp::Hot;
      else if (maybe->getSecond() == Temp::Cold)
        temperatures[callee] = Temp::Both;
    } else if (auto coldInvoke = dyn_cast<InvokeOp>(op)) {
      SymbolConstantAttr callee =
          cast<SymbolConstantAttr>(coldInvoke.getCallee());
      auto maybe = temperatures.find(callee);
      if (maybe == temperatures.end())
        temperatures[callee] = Temp::Cold;
      else if (maybe->getSecond() == Temp::Hot)
        temperatures[callee] = Temp::Both;
    }
  });
  for (StackAllocationOp alloc : allocs) {
    if (alloc->use_empty())
      continue;
    Operation *ancestor = *alloc->user_begin();
    for (Operation *user : llvm::drop_begin(alloc->getUsers())) {
      if (domInfo.dominates(ancestor, user))
        continue;
      if (domInfo.dominates(user, ancestor)) {
        ancestor = user;
        continue;
      }
      // `ancestor` and `user` live in sibling regions. We need to find a
      // common ancestor.
      ancestor = findNearestCommonAncestor(domInfo, ancestor, user);
    }
    alloc->moveBefore(ancestor);
  }
}

void LowerAsyncFunctionsPass::runOnOperation() {
  ModuleOp module = getOperation();
  TargetInfoAttr targetInfo = lookupTargetInfo(module);
  if (!targetInfo) {
    mlir::emitError(module.getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }

  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  Shared<SymbolTable &> sharedTable(symtab);

  // Convert async functions.
  // Save a clone of the original async function for the purpose of generating a
  // hot ramp/resume. Key the clone off the symbol of the original so we can
  // look it up.
  ImplicitLocOpBuilder b(module->getLoc(), module);
  auto &domInfo = getAnalysis<mlir::DominanceInfo>();
  LowerAsyncBuildContext buildContext(sharedTable, b, domInfo, targetInfo);

  DenseMap<SymbolConstantAttr, Temp> temperatures;
  SmallVector<FuncOp> asyncFunctions;
  FrameData empty;
  COTypes opaqueCoroutineTypes(module.getContext(), std::move(empty),
                               /*promiseType=*/{});
  mlir::AttrTypeReplacer replacer;
  Type headerType = PointerType::get(opaqueCoroutineTypes.getHeaderType());
  replacer.addReplacement([&](CoroutineType type) { return headerType; });

  for (auto funcOp : module.getOps<FuncOp>()) {

    if (!funcOp.isAsync()) {
      replacer.recursivelyReplaceElementsIn(funcOp, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/true,
                                            /*replaceTypes=*/true);
      funcOp.walk([&](InvokeOp coldInvoke) {
        SymbolConstantAttr callee =
            cast<SymbolConstantAttr>(coldInvoke.getCallee());
        auto maybe = temperatures.find(callee);
        if (maybe == temperatures.end())
          temperatures[callee] = Temp::Cold;
        else if (maybe->getSecond() == Temp::Hot)
          temperatures[callee] = Temp::Both;
      });

      continue;
    }
    // calculate the frame of the original, unmodified resume.
    buildContext.preprocessAsyncFunction(funcOp, domInfo, temperatures);
    replacer.recursivelyReplaceElementsIn(funcOp, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/true,
                                          /*replaceTypes=*/true);
    asyncFunctions.push_back(funcOp);
  }
  DenseMap<SymbolConstantAttr, std::pair<SymbolConstantAttr, Type>>
      asyncFuncToColdRampFunctions;
  DenseMap<SymbolConstantAttr, std::pair<SymbolConstantAttr, Type>>
      asyncFuncToHotRampFunctions;
  for (FuncOp funcOp : asyncFunctions) {
    // Store a clone of the function so we can generate the hot ramp/resume.
    SymbolConstantAttr key = SymbolConstantAttr::get(funcOp);
    auto temperatureMaybe = temperatures.find(key);

    // TODO: DCE should have eliminated this function. It's useful for unit
    // tests to not erase recursively.
    if (temperatureMaybe == temperatures.end()) {
      funcOp->erase();
      continue;
    }
    Temp temperature = temperatureMaybe->second;
    Coroutine coroutine = buildContext.createCoroutine(funcOp, temperature);
    switch (temperature) {
    case Temp::Hot: {
      SymbolConstantAttr value = SymbolConstantAttr::get(coroutine.hotRamp);
      asyncFuncToHotRampFunctions[key] = {value, coroutine.coroutineType};

      break;
    }
    case Temp::Cold: {
      SymbolConstantAttr coldvalue =
          SymbolConstantAttr::get(coroutine.coldRamp);
      asyncFuncToColdRampFunctions[key] = {coldvalue, coroutine.coroutineType};
      break;
    }
    case Temp::Both: {
      SymbolConstantAttr value = SymbolConstantAttr::get(coroutine.hotRamp);
      asyncFuncToHotRampFunctions[key] = {value, coroutine.coroutineType};

      SymbolConstantAttr coldvalue =
          SymbolConstantAttr::get(coroutine.coldRamp);
      asyncFuncToColdRampFunctions[key] = {coldvalue, coroutine.coroutineType};
      break;
    }
    }
  }

  // Apply all other CO lowerings.
  IRRewriter rewriter(b);
  module.walk([&](Operation *op) {
    if (auto invokeOp = dyn_cast<InvokeOp>(op)) {
      auto symbol = cast<SymbolConstantAttr>(invokeOp.getCallee());
      auto newSymbolPtr = asyncFuncToColdRampFunctions.find(symbol);
      if (newSymbolPtr != asyncFuncToColdRampFunctions.end()) {
        auto [newSymbol, continuationType] = newSymbolPtr->getSecond();
        rewriter.setInsertionPoint(op);
        auto callOp = CallOp::create(rewriter, invokeOp->getLoc(), newSymbol,
                                     invokeOp.getOperands());
        rewriter.replaceOp(invokeOp, callOp);
      } else {
        llvm_unreachable(
            "every callee of an invoke operation should have been lowered");
      }
    } else if (auto hotInvokeOp = dyn_cast<HotInvokeOp>(op)) {
      auto symbol = cast<SymbolConstantAttr>(hotInvokeOp.getCallee());
      auto newSymbolPtr = asyncFuncToHotRampFunctions.find(symbol);
      if (newSymbolPtr != asyncFuncToHotRampFunctions.end()) {
        auto [newSymbol, continuationType] = newSymbolPtr->getSecond();
        rewriter.setInsertionPoint(op);
        auto callOp = CallOp::create(rewriter, hotInvokeOp->getLoc(), newSymbol,
                                     hotInvokeOp.getOperands());
        rewriter.replaceOp(hotInvokeOp, callOp);
      }
    } else if (auto setErrorResultOp = dyn_cast<SetByRefErrorAndResultOp>(op)) {
      rewriter.setInsertionPoint(op);
      Value continuation = setErrorResultOp.getOperand(0);
      auto setByRefArgument = [&](Value argument, unsigned index) {
        Value slot =
            StructGEPOp::create(rewriter, op->getLoc(), continuation, index);
        Value typedSlot = PointerBitcastOp::create(
            rewriter, op->getLoc(), KGEN::PointerType::get(argument.getType()),
            slot);
        StoreOp::create(rewriter, op->getLoc(), argument, typedSlot);
      };
      if (Value error = setErrorResultOp.getError())
        setByRefArgument(error, AsyncContinuationField::ErrorSlot);
      if (!isa<KGEN::NoneType>(
              setErrorResultOp.getResult().getType().getElementType())) {
        Value result = setErrorResultOp.getResult();
        setByRefArgument(result, AsyncContinuationField::ResultSlot);
      }
      op->erase();
    } else if (auto resumeOp = dyn_cast<ResumeOp>(op)) {
      rewriter.setInsertionPoint(op);
      Value continuation = resumeOp.getOperand();
      Value slot = StructGEPOp::create(rewriter, op->getLoc(), continuation,
                                       ResumeFunction);
      Value typed = PointerBitcastOp::create(
          rewriter, op->getLoc(), PointerType::get(resumeOp.getType()), slot);
      Value load = LoadOp::create(rewriter, op->getLoc(), typed);
      resumeOp.replaceAllUsesWith(load);
      resumeOp->erase();
    } else if (auto callbackOp = dyn_cast<GetCallbackPtrOp>(op)) {
      rewriter.setInsertionPoint(op);
      Value continuation = callbackOp.getOperand();
      Value slot =
          StructGEPOp::create(rewriter, op->getLoc(), continuation, CallbackFn);
      Value slotCast = PointerBitcastOp::create(rewriter, op->getLoc(),
                                                callbackOp.getType(), slot);
      callbackOp.replaceAllUsesWith(slotCast);
      callbackOp->erase();
    } else if (auto destroyOp = dyn_cast<DestroyOp>(op)) {
      rewriter.setInsertionPoint(op);
      Value continuation = destroyOp.getOperand();
      AlignedFreeOp::create(rewriter, destroyOp->getLoc(), continuation);
      destroyOp->erase();
    } else if (auto getResults = dyn_cast<GetResultsOp>(op)) {
      rewriter.setInsertionPoint(op);
      Value continuation = getResults.getOperand();
      StructType headerType = opaqueCoroutineTypes.getHeaderType();
      SmallVector<Type> headerPlusPromiseTypes(*headerType.getElementTypes());
      headerPlusPromiseTypes.push_back(StructType::get(
          op->getContext(), llvm::to_vector(getResults.getResultTypes())));
      Value promiseContinuation = PointerBitcastOp::create(
          rewriter, op->getLoc(),
          PointerType::get(StructType::get(headerPlusPromiseTypes)),
          continuation);
      Value promiseSlot = StructGEPOp::create(rewriter, op->getLoc(),
                                              promiseContinuation, Promise);
      for (auto [idx, result] : llvm::enumerate(getResults.getResults())) {
        rewriter.replaceAllUsesWith(
            result, LoadOp::create(rewriter, op->getLoc(),
                                   StructGEPOp::create(rewriter, op->getLoc(),
                                                       promiseSlot, idx)));
      }
      getResults->erase();
    }
  });
}
