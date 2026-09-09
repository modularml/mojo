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

#include "Mojo/CODialect/COOps.h"
#include "Mojo/HLCFDialect/Analysis/CFG.h"
#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Transforms/FoldUtils.h"

#define DEBUG_TYPE "kgen-sccp"

using namespace M;
using namespace HLCF;
using namespace KGEN;

using namespace mlir::dataflow;
using mlir::ChangeResult;

//===----------------------------------------------------------------------===//
// SCCPAnalysis
//===----------------------------------------------------------------------===//

namespace {
class SCCPAnalysis {
public:
  explicit SCCPAnalysis() {}

  /// ConstantValue lattice element type.
  using ConstantState = Lattice<ConstantValue>;

  /// Struct for analysis state that includes:
  /// - A map from Value to ConstantValue lattice element.
  /// - A bump pointer allocator for the lattice elements.
  struct AnalysisStateType {

    /// Value to lattice map;
    DenseMap<Value, ConstantState *> valueLattices;

    // BumpPtr allocator for ConstantStates
    llvm::SpecificBumpPtrAllocator<ConstantState> allocator;

    AnalysisStateType() = default;
    AnalysisStateType(const AnalysisStateType &other);
    AnalysisStateType &operator=(const AnalysisStateType &other) {
      valueLattices = other.valueLattices;
      return *this;
    }
  };

  /// Process a Region operation.
  LogicalResult processRegion(Region &region, AnalysisStateType &state,
                              bool &hasEarlyExits,
                              SmallVector<bool> &shouldContinue,
                              int64_t loopLevel,
                              bool setBlockArgToEntryState = true);

  /// Helper function to rewrite the IR with SCCP analysis results.
  LogicalResult rewrite(MLIRContext *context,
                        MutableArrayRef<Region> initialRegions);

  LogicalResult run(Operation *op);

private:
  /// ConstantValue lattice states for ControlFlow type of nodes.
  struct ControlFlowOperationState {
    /// Work list for analyzing loops, each item in the list is possible
    /// constant values mapped to the inputs to each loop iteration.
    std::queue<SmallVector<Attribute>> entryStates;

    /// Lattice for op results that will be updated by ControlFlowTerminators.
    AnalysisStateType exitStates;

    /// Breaks exiting this loop that the analysis visited or proved dead.
    /// `exitStates` describes the loop's results only once it holds them all.
    llvm::SmallPtrSet<Operation *, 4> resolvedBreaks;
  };

  /// Cache for `getBreaksExiting`.
  DenseMap<Operation *, llvm::SmallPtrSet<Operation *, 4>> breaksExitingLoop;

  /// The breaks that exit `loop`, i.e. those whose parent node is `loop`.
  const llvm::SmallPtrSet<Operation *, 4> &getBreaksExiting(Operation *loop);

  /// Mark as Unknown the results of every loop other than `loop` that is the
  /// parent node of a break or continue in its body.
  void markOuterLoopResultsUnknown(Operation *loop);

  /// Account for a break the analysis reached.
  void resolveBreak(BreakOp breakOp);

  /// Account for nested breaks the caller knows cannot execute.
  void markBreaksUnreachable(Operation *op);
  void markBreaksUnreachable(Region &region);

  /// Process a ControlFlowNode operation.
  /// `state` is the entry state of the lattice analysis values.
  /// `shouldContinue` keeps track if operation traversing
  /// in the parent region should keep going or stop in case early exits
  /// happen, such as break, continue, return.
  LogicalResult processControlFlowNode(ControlFlowNode node,
                                       AnalysisStateType &state,
                                       SmallVector<bool> &shouldContinue,
                                       int64_t loopLevel);

  /// Process a ControlFlowTerminator operation.
  void processControlFlowTerminator(ControlFlowTerminator term,
                                    AnalysisStateType &state);

  /// Visit a general operation to apply the transform function.
  static void visitOperation(Operation *op, AnalysisStateType &state);

  /// Get the lattice element for value from state.
  static ConstantState *getLatticeElement(Value value,
                                          AnalysisStateType &state);

  /// Get the lattice elements for op's operands.
  static void getValuesLattice(SmallVectorImpl<Attribute> &attributes,
                               ValueRange value, AnalysisStateType &state);

  /// Set state to Unknown.
  static void setToEntryState(ConstantState *state);

  /// Set states to Unknown.
  static void setAllToEntryStates(ArrayRef<ConstantState *> states);

  /// Merge lattice from state to current. Join the lattice if exist in both.
  static ChangeResult mergeStates(AnalysisStateType &current,
                                  AnalysisStateType &state);

  /// Update parentOp's exit states.
  static void updateParentOpOutputState(ValueRange termValues,
                                        Operation *parentOp,
                                        AnalysisStateType &termState,
                                        AnalysisStateType &parentOutputState);

  /// Helper function to rewrite the IR with SCCP analysis results.
  LogicalResult replaceWithConstant(OpBuilder &builder,
                                    mlir::OperationFolder &folder, Value value);

  static int64_t getLoopConvergeThreshold(Operation *op, int64_t loopLevel);

  /// Map from a ControlFlowNode/Terminator to its states (entry/exit) and the
  /// allocator of the state.
  DenseMap<Operation *, std::unique_ptr<ControlFlowOperationState>>
      controlFlowOperationStates;

  /// State for the top operation that the analysis starts from.
  AnalysisStateType topState;

  /// Get or create ControlFlowOperationState (if not exist yet).
  ControlFlowOperationState *getOrCreateCFState(Operation *op);
};

} // namespace

/// Print lattice content for debugging.
[[maybe_unused]] static void printState(SCCPAnalysis::AnalysisStateType &state,
                                        llvm::raw_ostream &os) {
  for (auto &[value, lattice] : state.valueLattices) {
    os << "============================\n";
    os << "value: " << value << "\n";
    lattice->print(os);
    os << "\n";
  }
}

SCCPAnalysis::AnalysisStateType::AnalysisStateType(
    const AnalysisStateType &other)
    : valueLattices(other.valueLattices) {}

template <typename OT, typename... Args>
static OT *bumpPtrAllocate(llvm::SpecificBumpPtrAllocator<OT> &allocator,
                           Args... args) {
  return new (allocator.Allocate()) OT(args...);
}

SCCPAnalysis::ControlFlowOperationState *
SCCPAnalysis::getOrCreateCFState(Operation *op) {
  auto iter = controlFlowOperationStates.find(op);
  if (iter == controlFlowOperationStates.end()) {
    iter = controlFlowOperationStates
               .insert({op, std::make_unique<ControlFlowOperationState>()})
               .first;
  }
  return iter->second.get();
}

ChangeResult SCCPAnalysis::mergeStates(AnalysisStateType &current,
                                       AnalysisStateType &state) {
  ChangeResult changed = ChangeResult::NoChange;
  for (auto &[value, lattice] : state.valueLattices) {
    ConstantState *currLattice = getLatticeElement(value, current);
    changed |= currLattice->join(*lattice);
  }
  return changed;
}

SCCPAnalysis::ConstantState *
SCCPAnalysis::getLatticeElement(Value value, AnalysisStateType &state) {
  auto iter = state.valueLattices.find(value);
  if (iter == state.valueLattices.end())
    iter = state.valueLattices
               .insert({value, bumpPtrAllocate(state.allocator, value)})
               .first;

  return iter->second;
}

void SCCPAnalysis::getValuesLattice(SmallVectorImpl<Attribute> &attributes,
                                    ValueRange values,
                                    AnalysisStateType &state) {
  for (Value value : values) {
    ConstantState *lattice = getLatticeElement(value, state);
    assert(!lattice->getValue().isUninitialized() &&
           "All operands should have initialized lattice value.");
    attributes.push_back(lattice->getValue().getConstantValue());
  }
}

void SCCPAnalysis::setToEntryState(ConstantState *state) {
  (void)state->join(ConstantValue::getUnknownConstant());
}

void SCCPAnalysis::setAllToEntryStates(ArrayRef<ConstantState *> states) {
  for (ConstantState *state : states)
    setToEntryState(state);
}

void SCCPAnalysis::visitOperation(Operation *op, AnalysisStateType &state) {
  SmallVector<Attribute> constantOperands;
  getValuesLattice(constantOperands, op->getOperands(), state);

  SmallVector<ConstantState *> results;
  for (Value result : op->getResults())
    results.push_back(getLatticeElement(result, state));

  // Save the original operands and attributes just in case the operation
  // folds in-place. The constant passed in may not correspond to the real
  // runtime value, so in-place updates are not allowed.
  SmallVector<Value> originalOperands(op->getOperands());
  DictionaryAttr originalAttrs = op->getAttrDictionary();

  // Simulate the result of folding this operation to a constant. If folding
  // fails or was an in-place fold, mark the results as overdefined.
  SmallVector<OpFoldResult> foldResults;
  foldResults.reserve(op->getNumResults());
  if (failed(op->fold(constantOperands, foldResults))) {
    setAllToEntryStates(results);
    return;
  }

  // If the folding was in-place, mark the results as overdefined and reset
  // the operation. We don't allow in-place folds as the desire here is for
  // simulated execution, and not general folding.
  if (foldResults.empty()) {
    op->setOperands(originalOperands);
    op->setAttrs(originalAttrs);
    setAllToEntryStates(results);
    return;
  }

  // Merge the fold results into the lattice for this operation.
  assert(foldResults.size() == op->getNumResults() && "invalid result size");
  for (const auto [lattice, foldResult] : llvm::zip(results, foldResults)) {
    // Merge in the result of the fold, either a constant or a value.
    if (Attribute attr = llvm::dyn_cast_if_present<Attribute>(foldResult)) {
      LLVM_DEBUG(llvm::dbgs() << "Folded to constant: " << attr << "\n");
      (void)lattice->join(ConstantValue(attr, op->getDialect()));
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Folded to value: " << cast<Value>(foldResult) << "\n");
      (void)lattice->join(*getLatticeElement(cast<Value>(foldResult), state));
    }
  }
}

int64_t SCCPAnalysis::getLoopConvergeThreshold(Operation *op,
                                               int64_t loopLevel) {
  // Try to avoid explosion with deep nested loops.
  // TODO: use decorator or more sophisticated heuristics to set per loop
  // threshold.
  int64_t result = 5;
  if (loopLevel > 2)
    result = 2;

  if (auto forOp = dyn_cast<ForOp>(op); forOp && forOp.getTripCount()) {
    // Use trip count as threshold for a for-loop.
    result = std::min<int64_t>(forOp.getTripCount().value(), result);
  }
  return result;
}

LogicalResult SCCPAnalysis::processControlFlowNode(
    ControlFlowNode node, AnalysisStateType &state,
    SmallVector<bool> &shouldContinue, int64_t loopLevel) {
  VerboseCompilerTimeTraceScope traceScope(
      "SCCPAnalysis::processControlFlowNode",
      [name = node.getOperation()->getName()] {
        return name.getStringRef().str();
      });

  // TODO: Add support for other ControlFlowNode, e.g. kgen.try, etc.
  // TODO: issue #23376, this function should work more generally for
  // ControlFlowInterfaces.
  if (isa<IfOp, SwitchOp>(node.getOperation())) {

    // TODO: extend this logic to SwitchOp.
    SmallVector<Attribute> constantOperands;
    getValuesLattice(constantOperands, node.getOperation()->getOperands(),
                     state);
    SmallVector<ControlFlowTarget> targets;
    node.getEntryTargets(constantOperands, targets);

    size_t numShouldNotJoin = 0;

    // Regions that no target selects cannot execute.
    llvm::SmallPtrSet<Region *, 2> executedRegions;
    for (ControlFlowTarget target : targets) {
      if (target.index)
        executedRegions.insert(&node->getRegion(target.index.value()));
    }
    for (Region &region : node->getRegions()) {
      if (!executedRegions.contains(&region))
        markBreaksUnreachable(region);
    }

    // Each target starts from the enclosing verdict; whether this op exits
    // early is decided below, once every target is known.
    const bool enclosingShouldContinue = shouldContinue[loopLevel];
    for (ControlFlowTarget target : targets) {
      if (target.index) {
        // Analyze region with entry state.
        bool hasEarlyExits = false;
        shouldContinue[loopLevel] = enclosingShouldContinue;
        if (failed(processRegion(node->getRegion(target.index.value()), state,
                                 hasEarlyExits, shouldContinue, loopLevel)))
          return failure();

        if (hasEarlyExits)
          ++numShouldNotJoin;
      }
    }
    shouldContinue[loopLevel] = enclosingShouldContinue;

    if (numShouldNotJoin == targets.size()) {
      // A break or continue happened for all regions, we should not keep
      // running the rest of the operations in the parent region.
      shouldContinue[loopLevel] = false;
    }
    (void)mergeStates(state, getOrCreateCFState(node)->exitStates);
    controlFlowOperationStates.erase(node.getOperation());
    return success();
  }

  bool skip = false;
  if (isa<LoopOp, ForOp>(node.getOperation())) {

    // Prepare for initial loop inputs.
    SmallVector<Attribute> constantOperands;
    if (auto forOp = dyn_cast<ForOp>(node.getOperation()))
      getValuesLattice(constantOperands, forOp.getIterArgs(), state);
    else
      getValuesLattice(constantOperands, node->getOperands(), state);

    // Prepare the workList for analyzing the loop.
    ControlFlowOperationState *cfStates = getOrCreateCFState(node);
    std::queue<SmallVector<Attribute>> &workList = cfStates->entryStates;
    workList.push(constantOperands);

    int64_t loopIter = 0;
    int64_t threshold =
        getLoopConvergeThreshold(node.getOperation(), loopLevel);

    while (!workList.empty() && loopIter < threshold) {
      SmallVector<Attribute> inputValues = workList.front();
      workList.pop();

      AnalysisStateType nestedState = state;
      // Prepare for input arguments for this iteration.
      for (auto [inputValue, blockArg] :
           llvm::zip(inputValues, node->getRegions().front().getArguments())) {
        if (loopIter > 0 && !inputValue) {
          // After 1st iteration, if any of the loop inputs is already unknown,
          // stop analyzing.
          skip = true;
          break;
        }
        ConstantState *lattice = getLatticeElement(blockArg, nestedState);
        (void)lattice->join(ConstantValue(inputValue, node->getDialect()));
      }
      if (skip)
        break;

      // Process loop body.
      bool hasEarlyExits = false;
      shouldContinue.emplace_back(true);
      if (failed(processRegion(node->getRegions().front(), nestedState,
                               hasEarlyExits, shouldContinue, loopLevel + 1,
                               /*setBlockArgToEntryState=*/false))) {
        shouldContinue.pop_back();
        return failure();
      }

      shouldContinue.pop_back();
      ++loopIter;
    }

    // If the exit state is empty, we might hit unreachable in the loop without
    // visiting a break terminator. In that case, simply mark the loop results
    // as Unknown in the else branch.
    if (!cfStates->exitStates.valueLattices.empty() && workList.empty() &&
        !skip &&
        cfStates->resolvedBreaks.size() ==
            getBreaksExiting(node.getOperation()).size()) {
      // Merge analysis states if analyze loop converges.
      (void)mergeStates(state, cfStates->exitStates);
    } else {
      // Mark loop results as Unknown.
      for (Value result : node.getOperation()->getResults())
        setToEntryState(getLatticeElement(result, state));
      // Unanalyzed iterations may hide edges into or out of other loops.
      markOuterLoopResultsUnknown(node.getOperation());
    }
    // Clean up states (which is being updated by this op's terminators) when
    // analyzing is done.
    controlFlowOperationStates.erase(node);
    return success();
  }

  // Otherwise, process subregions if any and mark all results as Unknown.
  if (node->getNumRegions() > 0) {
    for (Region &region : node->getRegions()) {
      AnalysisStateType nestedState = state;
      bool hasEarlyExits = false;
      if (failed(processRegion(region, nestedState, hasEarlyExits,
                               shouldContinue, loopLevel)))
        return failure();
      (void)mergeStates(state, nestedState);
    }
  }

  for (Value result : node.getOperation()->getResults())
    setToEntryState(getLatticeElement(result, state));

  return success();
}

const llvm::SmallPtrSet<Operation *, 4> &
SCCPAnalysis::getBreaksExiting(Operation *loop) {
  auto [it, inserted] = breaksExitingLoop.try_emplace(loop);
  if (inserted) {
    loop->getRegion(0).walk([&](BreakOp breakOp) {
      if (getParentNode(breakOp) == loop) {
        it->second.insert(breakOp);
      }
    });
  }
  return it->second;
}

void SCCPAnalysis::markOuterLoopResultsUnknown(Operation *loop) {
  loop->getRegion(0).walk([&](ControlFlowTerminator term) {
    if (!isa<BreakOp, ContinueOp>(term.getOperation()))
      return;
    Operation *target = getParentNode(term);
    if (target == loop)
      return;
    ControlFlowOperationState *targetState = getOrCreateCFState(target);
    for (Value res : target->getResults())
      setToEntryState(getLatticeElement(res, targetState->exitStates));
  });
}

void SCCPAnalysis::resolveBreak(BreakOp breakOp) {
  getOrCreateCFState(getParentNode(breakOp))->resolvedBreaks.insert(breakOp);
}

void SCCPAnalysis::markBreaksUnreachable(Operation *op) {
  op->walk([&](BreakOp breakOp) { resolveBreak(breakOp); });
}

void SCCPAnalysis::markBreaksUnreachable(Region &region) {
  region.walk([&](BreakOp breakOp) { resolveBreak(breakOp); });
}

void SCCPAnalysis::updateParentOpOutputState(
    ValueRange termValues, Operation *parentOp, AnalysisStateType &termState,
    AnalysisStateType &parentOutputState) {
  for (auto [operand, opResult] :
       llvm::zip(termValues, parentOp->getResults())) {
    ConstantState *lattice0 = getLatticeElement(operand, termState);
    ConstantState *lattice1 = getLatticeElement(opResult, parentOutputState);
    (void)lattice1->join(*lattice0);
  }
}

void SCCPAnalysis::processControlFlowTerminator(ControlFlowTerminator term,
                                                AnalysisStateType &termState) {
  VerboseCompilerTimeTraceScope traceScope(
      "SCCPAnalysis::processControlTerminator",
      [name = term.getOperation()->getName()] {
        return name.getStringRef().str();
      });

  // TODO: Add support for other ControlFlowTerminators, e.g. kgen.return, etc.
  if (auto breakOp = dyn_cast<BreakOp>(term.getOperation())) {
    Operation *parentLoop = getParentNode(term);
    resolveBreak(breakOp);

    // Update parent loop's exit state.
    ControlFlowOperationState *cfOpStates = getOrCreateCFState(parentLoop);

    updateParentOpOutputState(term->getOperands(), parentLoop, termState,
                              cfOpStates->exitStates);
    return;
  }

  if (auto continueOp = dyn_cast<ContinueOp>(term.getOperation())) {
    Operation *parentLoop = getParentNode(term);
    // Prepare new inputs for parent loop.
    SmallVector<Attribute> constantOperands;
    getValuesLattice(constantOperands, term.getOperation()->getOperands(),
                     termState);

    ControlFlowOperationState *cfOpStates = getOrCreateCFState(parentLoop);
    // Only push the new inputs if it is different from current one.
    cfOpStates->entryStates.push(constantOperands);
    return;
  }

  if (auto yieldOp = dyn_cast<YieldOp>(term.getOperation())) {
    Operation *parentOp = getParentNode(term);
    ControlFlowOperationState *cfOpStates = getOrCreateCFState(parentOp);
    // update parent op's exit state
    updateParentOpOutputState(term->getOperands(), parentOp, termState,
                              cfOpStates->exitStates);
    return;
  }

  if (auto forYieldOp = dyn_cast<ForYieldOp>(term.getOperation())) {
    Operation *parentOp = getParentNode(term);
    SmallVector<Attribute> constantOperands;
    getValuesLattice(constantOperands, forYieldOp.getOperands(), termState);
    SmallVector<ControlFlowTarget> targets;
    forYieldOp.getBranchTargets(constantOperands, targets);
    ControlFlowOperationState *cfOpStates = getOrCreateCFState(parentOp);

    for (ControlFlowTarget &target : targets) {
      if (target.index) {
        // Branch back to for-loop body.
        cfOpStates->entryStates.push(constantOperands);
      } else
        updateParentOpOutputState(forYieldOp.getReturnValues(), parentOp,
                                  termState, cfOpStates->exitStates);
    }
    return;
  }

  // Otherwise, mark all results as Unknown.
  for (Value result : term.getOperation()->getResults())
    setToEntryState(getLatticeElement(result, termState));
}

LogicalResult SCCPAnalysis::processRegion(Region &region,
                                          AnalysisStateType &state,
                                          bool &hasEarlyExits,
                                          SmallVector<bool> &shouldContinue,
                                          int64_t loopLevel,
                                          bool setBlockArgToEntryState) {
  if (!llvm::hasSingleElement(region)) {
    return region.getParentOp()->emitError(
        "'sccp' can only be run on operations with all single block "
        "regions");
  }

  Block &block = region.front();
  if (setBlockArgToEntryState) {
    for (BlockArgument &arg : block.getArguments())
      setToEntryState(getLatticeElement(arg, state));
  }

  for (Operation &op : block) {
    if (!shouldContinue[loopLevel])
      break;

    if (auto node = dyn_cast<ControlFlowNode>(op)) {
      if (failed(
              processControlFlowNode(node, state, shouldContinue, loopLevel)))
        return failure();

      if (!shouldContinue[loopLevel]) {
        // Control left the region here, so the rest of it cannot execute.
        hasEarlyExits = true;
        for (Operation &rest :
             llvm::make_range(std::next(op.getIterator()), block.end()))
          markBreaksUnreachable(&rest);
        break;
      }

      continue;
    }

    if (auto term = dyn_cast<ControlFlowTerminator>(op)) {
      processControlFlowTerminator(term, state);
      // Tell parent region that there is early exit (so that the parent region
      // can decide whether to continue traverse the rest of the operation or
      // not).
      hasEarlyExits = isa<ContinueOp, BreakOp, KGEN::UnreachableOp>(op);
      break;
    }

    if (isa<KGEN::StageClosureOp, CO::ExecuteOp>(op)) {
      //  TODO: Skip inter-procedural analysis for now. Mark anyone of its
      //  result as Unknown.
      for (Value result : op.getResults())
        setToEntryState(getLatticeElement(result, state));
    } else if (op.getNumRegions() > 0) {
      for (Region &region : op.getRegions()) {
        AnalysisStateType nestedState = state;
        bool nestedEarlyExits = false;
        if (failed(processRegion(region, nestedState, nestedEarlyExits,
                                 shouldContinue, loopLevel)))
          return failure();
        (void)mergeStates(state, nestedState);
      }
      if (!shouldContinue[loopLevel]) {
        hasEarlyExits = true;
        break;
      }
      continue;
    } else {
      // Try to fold the operation.
      visitOperation(&op, state);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SCCP Rewrites
//===----------------------------------------------------------------------===//

/// Replace the given value with a constant if the corresponding lattice
/// represents a constant. Returns success if the value was replaced, failure
/// otherwise.
LogicalResult SCCPAnalysis::replaceWithConstant(OpBuilder &builder,
                                                mlir::OperationFolder &folder,
                                                Value value) {
  AnalysisStateType &state = topState;
  ConstantState *lattice = getLatticeElement(value, state);
  if (!lattice || lattice->getValue().isUninitialized())
    return failure();
  const ConstantValue &latticeValue = lattice->getValue();
  if (!latticeValue.getConstantValue())
    return failure();

  // Attempt to materialize a constant for the given value.
  Dialect *dialect = latticeValue.getConstantDialect();
  Value constant = folder.getOrCreateConstant(
      builder.getInsertionBlock(), dialect, latticeValue.getConstantValue(),
      value.getType());
  if (!constant)
    return failure();

  value.replaceAllUsesWith(constant);
  return success();
}

/// Rewrite the given regions using the computing analysis. This replaces the
/// uses of all values that have been computed to be constant, and erases as
/// many newly dead operations.
LogicalResult SCCPAnalysis::rewrite(MLIRContext *context,
                                    MutableArrayRef<Region> initialRegions) {
  VerboseCompilerTimeTraceScope traceScope("SCCPAnalysis::rewrite");

  SmallVector<Block *> worklist;
  auto addToWorklist = [&](MutableArrayRef<Region> regions) {
    for (Region &region : regions)
      for (Block &block : llvm::reverse(region))
        worklist.push_back(&block);
  };

  // An operation folder used to create and unique constants.
  mlir::OperationFolder folder(context);
  OpBuilder builder(context);

  addToWorklist(initialRegions);
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();

    for (Operation &op : llvm::make_early_inc_range(*block)) {
      builder.setInsertionPoint(&op);

      // Replace any result with constants.
      bool replacedAll = op.getNumResults() != 0;
      for (Value result : op.getResults())
        replacedAll &= succeeded(replaceWithConstant(builder, folder, result));

      // If all of the results of the operation were replaced, try to erase
      // the operation completely.
      if (replacedAll && wouldOpBeTriviallyDead(&op)) {
        assert(op.use_empty() && "expected all uses to be replaced");
        op.erase();
        continue;
      }

      // Add any the regions of this operation to the worklist.
      addToWorklist(op.getRegions());
    }

    // Replace any block arguments with constants.
    builder.setInsertionPointToStart(block);
    for (BlockArgument arg : block->getArguments()) {
      // Ignore replaceWithConstant result here. It's okay if the value is not a
      // constant, just don't rewrite it.
      (void)replaceWithConstant(builder, folder, arg);
    }
  }
  return success();
}

LogicalResult SCCPAnalysis::run(Operation *op) {
  for (Region &region : op->getRegions()) {
    bool hasEarlyExits = false;
    SmallVector<bool> shouldContinue{true};

    if (failed(
            processRegion(region, topState, hasEarlyExits, shouldContinue, 0)))
      return failure();
  }

  LLVM_DEBUG(printState(topState, llvm::dbgs()));
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_SCCP
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
/// Sparse Conditional Constant Propagation (SCCP).
/// This pass conditionally propagates constant values following
/// the dataflow graph of the program while eliminating dead branches.
/// This pass doesn't have inter-procedural support (yet).
struct SCCP : impl::SCCPBase<SCCP> {
  explicit SCCP() : SCCPBase() {}

  void runOnOperation() override;
};
} // namespace

void SCCP::runOnOperation() {
  VerboseCompilerTimeTraceScope traceScope("SCCP::runOnOperation");

  SCCPAnalysis analysis;

  if (failed(analysis.run(getOperation())))
    return signalPassFailure();

  // Rewrite the IR with constant result.
  if (failed(analysis.rewrite(&getContext(), getOperation()->getRegions())))
    return signalPassFailure();
}
