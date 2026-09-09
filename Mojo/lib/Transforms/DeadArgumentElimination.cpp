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
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/TransformUtils/CallGraphUtils.h"
#include "Support/STLExtras.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"

#define DEBUG_TYPE "kgen-dead-argument-elimination"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// DeadArgumentEliminationPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_DEADARGUMENTELIMINATION
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct DeadArgumentEliminationPass
    : impl::DeadArgumentEliminationBase<DeadArgumentEliminationPass> {
  void runOnOperation() override;
};
} // namespace

namespace {
struct CallGraphNode
    : public CallGraphNodeBase<CallGraphNode, FuncOp, KGENCallOpInterface> {
  using CallGraphNodeBase::CallGraphNodeBase;

  CallGraphNode *getCalleeNode(KGENCallOpInterface callOp);

  std::vector<CallGraphNode::EdgeT> getCallSites(CallGraphNode *callee);

  bool updated = false;
  SmallVector<unsigned> liveArguments;
};

struct CallGraph : public CallGraphBase<CallGraph, CallGraphNode> {
  explicit CallGraph(const SymbolTable &symtab) : symtab(symtab) {}

  bool shouldAddToGraph(KGENCallOpInterface call, CallGraphNode *node) {
    return !node->func.isExternal();
  }

  const SymbolTable symtab;
};

struct DeadArgumentElimination {
  /// Struct that represents (part of) either a return value or a function
  /// argument. Used so that arguments and return values can be used
  /// interchangeably.
  struct RetOrArg {
    FuncOp func;
    unsigned idx;
    bool isArg;

    RetOrArg(FuncOp func, unsigned idx, bool isArg)
        : func(func), idx(idx), isArg(isArg) {}

    /// Make RetOrArg comparable, so we can put it into a map.
    bool operator<(const RetOrArg &O) const {
      return std::tie(func, idx, isArg) < std::tie(O.func, O.idx, O.isArg);
    }

    /// Make RetOrArg comparable, so we can easily iterate the multimap.
    bool operator==(const RetOrArg &other) const {
      return func == other.func && idx == other.idx && isArg == other.isArg;
    }

    std::string getDescription() {
      return (Twine(isArg ? "Argument #" : "Return value #") + Twine(idx) +
              " of function " + func.getSymName())
          .str();
    }

    Value getValue() const {
      assert(isArg &&
             "RetOrArg::getValue() can only be called for Args not Rets.");
      return FuncOp(func).getArgument(idx);
    }
  };

  /// During our initial pass over the program, we determine that things are
  /// either alive or maybe alive. We don't mark anything explicitly dead (even
  /// if we know they are), since anything not alive with no registered uses
  /// (in Uses) will never be marked alive and will thus become dead in the end.
  enum Liveness { Live, MaybeLive };

  using UseMap = std::multimap<RetOrArg, RetOrArg>;

  /// This maps a return value or argument to any MaybeLive return values or
  /// arguments it uses. This allows the MaybeLive values to be marked live
  /// when any of its users is marked live.
  /// For example (indices are left out for clarity):
  ///  - Uses[ret F] = ret G
  ///    This means that F calls G, and F returns the value returned by G.
  ///  - Uses[arg F] = ret G
  ///    This means that some function calls G and passes its result as an
  ///    argument to F.
  ///  - Uses[ret F] = arg F
  ///    This means that F returns one of its own arguments.
  ///  - Uses[arg F] = arg G
  ///    This means that G calls F and passes one of its own (G's) arguments
  ///    directly to F.
  UseMap uses;

  using LiveSet = llvm::SmallSet<RetOrArg, 4>;
  using LiveFuncSet = llvm::SmallSet<FuncOp, 4>;

  /// This set contains all values that have been determined to be live.
  LiveSet liveValues;

  /// This set contains all values that are cannot be changed in any way.
  LiveFuncSet liveFunctions;

  using UseVector = SmallVector<RetOrArg, 5>;

  DeadArgumentElimination(CallGraph &callGraph, MLIRContext *context,
                          ModuleOp module, const SymbolTable &symbolTable)
      : callGraph(callGraph), context(context), module(module),
        symbolTable(symbolTable) {}

  void run();

private:
  /// Check if use is Live, if not mark it by putting it into maybeLiveUses.
  Liveness markIfNotLive(const RetOrArg &use, UseVector &maybeLiveUses);

  /// Check if the operand/use can be decided Live, if so propagate the Liveness
  /// to its known defs and uses.
  Liveness surveyUse(OpOperand &use, CallGraphNode *node,
                     UseVector &maybeLiveUses, bool isArg);

  /// Check if the uses of the return value or the argument of a function and
  /// record its downstream uses.
  Liveness surveyUses(const RetOrArg &retOrArg, UseVector &maybeLiveUses);

  /// Survey the uses of the arguments and return values of the function.
  void surveyFunction(FuncOp func);

  /// Check if retOrArg is surely to be known as Live.
  bool isLive(const RetOrArg &retOrArg);

  /// Mark the liveness for retOrArg if needed and propagate the info to its
  /// upstream and downstream defs/uses.
  void markValue(const RetOrArg &retOrArg, Liveness l,
                 const UseVector &maybeLiveUses);

  /// Mark retOrArg to be known as Live.
  void markLive(const RetOrArg &retOrArg);

  /// Mark func to be known as Live.
  void markLive(FuncOp func);

  /// Propagate liveness to downstream uses and upstream defs of retOrArg.
  void propagateLiveness(const RetOrArg &retOrArg);

  /// Helper function to create RetOrArg for a function's return value/result.
  static RetOrArg createRet(FuncOp func, unsigned idx);

  /// Helper function to create RetOrArg for a function's argument.
  static RetOrArg createArg(FuncOp func, unsigned idx);

  void print();

  CallGraph &callGraph;
  MLIRContext *context;
  ModuleOp module;
  const SymbolTable &symbolTable;

  /// Remove dead arguments, results and related operations from func in node.
  void removeDeadStuffFromFunction(CallGraphNode *node);

  /// Rewrite callees CallOps and its uses in the caller with new function
  /// signature if there are dead stuff being removed.
  void rewriteCalleesFromFunction(CallGraphNode *node);

  /// Mark any functions that are referenced by any operations other than a
  /// CallOp as live since they maybe called indirectly later.
  void markLiveMaybeIndirectCalled();
};
} // namespace

CallGraphNode *CallGraphNode::getCalleeNode(KGENCallOpInterface callOp) {
  auto iter = std::find_if(
      callsites.begin(), callsites.end(),
      [&](const CallGraphNode::EdgeT &edge) { return edge.call == callOp; });
  return iter == callsites.end() ? nullptr : iter->node;
}

std::vector<CallGraphNode::EdgeT>
CallGraphNode::getCallSites(CallGraphNode *callee) {
  std::vector<CallGraphNode::EdgeT> result;
  std::copy_if(
      callsites.begin(), callsites.end(), std::back_inserter(result),
      [&](const CallGraphNode::EdgeT &edge) { return edge.node == callee; });
  return result;
}

DeadArgumentElimination::RetOrArg
DeadArgumentElimination::createRet(FuncOp func, unsigned idx) {
  return {func, idx, false};
}

DeadArgumentElimination::RetOrArg
DeadArgumentElimination::createArg(FuncOp func, unsigned idx) {
  return {func, idx, true};
}

DeadArgumentElimination::Liveness
DeadArgumentElimination::markIfNotLive(const RetOrArg &use,
                                       UseVector &maybeLiveUses) {
  // We're live if our use or its Function is already marked as live.
  if (isLive(use))
    return Live;

  // We're maybe live otherwise, but remember that we must become live if
  // Use becomes live.
  maybeLiveUses.push_back(use);
  return MaybeLive;
}

DeadArgumentElimination::Liveness
DeadArgumentElimination::surveyUse(OpOperand &inputUse, CallGraphNode *node,
                                   UseVector &maybeLiveUses, bool isArg) {

  if (!isArg && node->func.isExported())
    return Live;

  std::vector<std::reference_wrapper<OpOperand>> worklist;
  DenseSet<Operation *> visited;

  worklist.emplace_back(inputUse);

  while (!worklist.empty()) {
    OpOperand &use = worklist.back();
    worklist.pop_back();
    if (visited.count(use.getOwner()) && !worklist.empty()) {
      // Also check is worklist is not empty before continuing. This is needed
      // because of the special handling in code below that can return `Live`.
      continue;
    }
    visited.insert(use.getOwner());

    if (auto ret = dyn_cast<ReturnOp>(use.getOwner())) {
      DeadArgumentElimination::Liveness result = MaybeLive;
      for (unsigned i = 0, e = ret->getNumOperands(); i < e; ++i) {
        RetOrArg ret = createRet(node->func, i);
        DeadArgumentElimination::Liveness subResult =
            markIfNotLive(ret, maybeLiveUses);
        if (result != Live)
          result = subResult;
      }
      return result;
    }

    if (auto call = dyn_cast<KGENCallOpInterface>(use.getOwner())) {
      // Only support caller is CallOp for now, so that things like
      // AsyncCallOp, kgen.create_closure which are KGENCallOpInterface but
      // should be marked as always Live.
      // TODO: refine this logic we want to support rewriting other types of
      // KGENCallOpInterface.
      if (!isa<CallOp>(call))
        return Live;

      // External callees have no edge in the call graph, so the liveness of
      // their arguments is unknowable here.
      CallGraphNode *calleeNode = node->getCalleeNode(call);
      if (!calleeNode)
        return Live;

      // Value passed to a normal call. It's only live when the corresponding
      // argument to the called function turns out live.
      RetOrArg arg = createArg(calleeNode->func, use.getOperandNumber());
      Liveness result = markIfNotLive(arg, maybeLiveUses);
      if (result == Live || worklist.empty())
        return result;
    }

    if ((!mlir::isPure(use.getOwner()) &&
         !isa<KGENCallOpInterface>(use.getOwner())) ||
        use.getOwner()->mightHaveTrait<OpTrait::IsTerminator>())
      return Live;

    // We can also more aggressively survey use.getOwner()'s results' uses
    // further.
    for (OpResult result : use.getOwner()->getResults()) {
      for (OpOperand &resUse : result.getUses()) {
        // TODO: add cache to avoid duplicated checks.
        worklist.emplace_back(resUse);
      }
    }
  }

  // Should not be reachable here because all uses should eventually reach a
  // return in a function if not before.
  llvm::llvm_unreachable_internal("DeadArgumentElimination surveyUse failed.");
}

DeadArgumentElimination::Liveness
DeadArgumentElimination::surveyUses(const RetOrArg &retOrArg,
                                    UseVector &maybeLiveUses) {
  // Assume it's dead
  Liveness result = MaybeLive;

  CallGraphNode *node = &callGraph.nodes.find(retOrArg.func)->second;
  // Check each use.
  for (OpOperand &use : retOrArg.getValue().getUses()) {
    result = surveyUse(use, node, maybeLiveUses, retOrArg.isArg);
    if (result == Live)
      break;
  }
  return result;
}

/// Performs the initial survey of the specified function, checking out whether
/// it uses any of its incoming arguments or whether any callers use the return
/// value. This fills in the LiveValues set and Uses map.
void DeadArgumentElimination::surveyFunction(FuncOp func) {
  if (liveFunctions.contains(func))
    return;

  if (func.isExported() || func.isExternal()) {
    markLive(func);
    return;
  }

  unsigned retCount = func.getNumResults();

  // Assume all return values are dead
  using RetVals = SmallVector<Liveness, 5>;
  RetVals retValLiveness(retCount, Live);

  // These vectors map each return value to the uses that make it MaybeLive, so
  // we can add those to the Uses map if the return value really turns out to be
  // MaybeLive. Initialized to a list of RetCount empty lists.
  using RetUses = SmallVector<UseVector, 5>;
  RetUses maybeLiveRetUses(retCount);

  LLVM_DEBUG(llvm::dbgs()
             << "DeadArgumentEliminationPass - Inspecting callers for fn: "
             << func.getSymName() << "\n");

  CallGraphNode *node = &callGraph.nodes.find(func)->second;

  for (CallGraphNode *caller : node->callers) {
    for (auto [call, node] : caller->getCallSites(node)) {
      if (!isa<CallOp>(call)) {
        // Only support caller is CallOp for now, so that things like
        // AsyncCallOp, kgen.create_closure which are KGENCallOpInterface but
        // should be marked as always Live.
        // TODO: refine this logic we want to support rewriting other types of
        // KGENCallOpInterface.
        markLive(func);
        return;
      }
    }
  }

  // Don't move any results so that we don't change ABI for callers who are
  // not in the same compilation unit but may still get to call this function
  // due to per-functions caching.
  for (unsigned idx = 0,
                e = func.getFuncTypeGenerator().getBody().getNumResults();
       idx < e; ++idx)
    markLive(createRet(func, idx));

  LLVM_DEBUG(
      llvm::dbgs() << "DeadArgumentEliminationPass - Inspecting args for fn: "
                   << func.getSymName() << "\n");

  // Check arguments.
  UseVector maybeLiveArgUses;
  for (BlockArgument input : func.getArguments()) {
    RetOrArg ra = createArg(func, input.getArgNumber());
    Liveness result = surveyUses(ra, maybeLiveArgUses);
    markValue(ra, result, maybeLiveArgUses);
    maybeLiveArgUses.clear();
  }
}

bool DeadArgumentElimination::isLive(const RetOrArg &retOrArg) {
  return liveValues.count(retOrArg) || liveFunctions.count(retOrArg.func);
}

void DeadArgumentElimination::markValue(const RetOrArg &retOrArg, Liveness l,
                                        const UseVector &maybeLiveUses) {
  switch (l) {
  case Live:
    markLive(retOrArg);
    break;
  case MaybeLive:
    assert(!isLive(retOrArg) && "Use is already live!");
    for (const RetOrArg &maybeLiveUse : maybeLiveUses) {
      if (isLive(maybeLiveUse)) {
        // A use is live, so this value is live.
        markLive(retOrArg);
        break;
      }
      // Note any uses of this value, so this value can be
      // marked live whenever one of the uses becomes live.
      uses.emplace(maybeLiveUse, retOrArg);
    }
    break;
  }
}

void DeadArgumentElimination::markLive(const RetOrArg &retOrArg) {
  if (isLive(retOrArg))
    return; // Already marked Live.

  liveValues.insert(retOrArg);

  LLVM_DEBUG(
      llvm::dbgs() << "DeadArgumentEliminationPass - Marking "
                   << (const_cast<RetOrArg *>(&retOrArg))->getDescription()
                   << " live\n");
  propagateLiveness(retOrArg);
}

void DeadArgumentElimination::markLive(FuncOp func) {
  LLVM_DEBUG(
      llvm::dbgs() << "DeadArgumentEliminationPass - Intrinsically live fn: "
                   << func.getSymName() << "\n");
  // Mark the function as live.
  liveFunctions.insert(func);

  // Mark all arguments as live.
  for (BlockArgument arg : func.getArguments())
    propagateLiveness(createArg(func, arg.getArgNumber()));

  // Mark all return values as live.
  for (unsigned i = 0,
                e = func.getFuncTypeGenerator().getBody().getNumResults();
       i < e; ++i)
    propagateLiveness(createRet(func, i));
}

void DeadArgumentElimination::propagateLiveness(const RetOrArg &retOrArg) {
  // We don't use upper_bound (or equal_range) here, because our recursive call
  // to ourselves is likely to cause the upper_bound (which is the first value
  // not belonging to RA) to become erased and the iterator invalidated.
  auto begin = uses.lower_bound(retOrArg);
  auto e = uses.end();
  UseMap::iterator i;
  for (i = begin; i != e && i->first == retOrArg; ++i)
    markLive(i->second);

  // Erase RA from the Uses map (from the lower bound to wherever we ended up
  // after the loop).
  uses.erase(begin, i);
}

static void getOpsToErase(BlockArgument arg,
                          llvm::SmallSet<Operation *, 4> &opsToErase) {
  std::vector<Operation *> workList;
  for (Operation *user : arg.getUsers())
    workList.emplace_back(user);
  while (!workList.empty()) {
    Operation *curr = workList.back();
    workList.pop_back();

    // If an arg or result is already marked as can be eliminated,
    // all their user should be pure operations.
    assert((mlir::isPure(curr) || isa<KGENCallOpInterface>(curr)) &&
           "Only operations have no side effect can be erased.");
    if (!mlir::isPure(curr))
      continue;

    opsToErase.insert(curr);
    for (OpResult res : curr->getResults()) {
      for (Operation *user : res.getUsers())
        workList.emplace_back(user);
    }
  }
}

void DeadArgumentElimination::removeDeadStuffFromFunction(CallGraphNode *node) {
  FuncOp func = node->func;
  if (liveFunctions.count(func))
    return;

  SmallVector<Type> inputTypes;
  SmallVector<BlockArgument> liveArguments;
  llvm::SmallSet<Operation *, 4> opsToErase;
  SmallVector<ArgConvention> argConventions;

  FuncType currSig = func.getFuncTypeGenerator().getBody();
  for (BlockArgument arg : func.getArguments()) {
    if (liveValues.count(createArg(func, arg.getArgNumber())) == 0) {
      getOpsToErase(arg, opsToErase);
      continue;
    }
    inputTypes.emplace_back(arg.getType());
    argConventions.emplace_back(
        currSig.getArgConventions()[arg.getArgNumber()]);
    liveArguments.emplace_back(arg);
  }

  bool dropArguments = func.getNumArguments() != inputTypes.size();
  // Nothing to remove, return
  if (!dropArguments)
    return;

  OpBuilder b(func);
  FunctionType newFuncType = FunctionType::get(
      func.getContext(), inputTypes,
      func.getFuncTypeGenerator().getBody().getValues().getResults());

  FuncType newSig =
      FuncType::get(newFuncType, argConventions, currSig.getFnEffects(),
                    currSig.getMetadata(), currSig.getArgListAttrs());

  Block *block = func.getBody(0);
  unsigned numArgs = block->getNumArguments();

  // Create and rewire live arguments.
  for (BlockArgument arg : liveArguments) {
    node->liveArguments.emplace_back(arg.getArgNumber());
    arg.replaceAllUsesWith(block->addArgument(arg.getType(), arg.getLoc()));
  }

  // Remove dead ops.
  for (Operation *op : opsToErase) {
    op->dropAllUses();
    op->dropAllReferences();
    op->erase();
  }

  // Drop all uses for old block arguments and erase all of them.
  for (unsigned i = 0; i < numArgs; ++i)
    block->getArgument(i).dropAllUses();
  block->eraseArguments(0, numArgs);

  // Set the FuncTypeGenerator.
  func.setFuncTypeGenerator(GeneratorType::get(/*inputParamTypes=*/{}, newSig));

  // Update node state to prepare for callee rewrites.
  node->updated = true;
}

void DeadArgumentElimination::rewriteCalleesFromFunction(CallGraphNode *node) {
  for (auto [call, callee] : node->callsites) {
    auto iter = callGraph.nodes.find(callee->func);
    CallGraphNode *calleeNode = &iter->second;
    // No rewrite needed.
    if (!calleeNode->updated)
      continue;

    OpBuilder b(call);
    SmallVector<Value> newOperands;

    for (unsigned argIdx : calleeNode->liveArguments)
      newOperands.push_back(call->getOperand(argIdx));

    if (auto kgenCall = dyn_cast<CallOp>(call.getOperation())) {
      auto newOp = CallOp::create(b, call.getLoc(),
                                  SymbolConstantAttr::get(calleeNode->func),
                                  newOperands);

      kgenCall->replaceAllUsesWith(newOp.getResults());

      kgenCall->dropAllUses();
      kgenCall->dropAllReferences();
      kgenCall.erase();
    } else {
      // Current logic of surveyFunction should guarantee that this branch is
      // never reached.
      llvm_unreachable("DeadArgumentElimination for ops other than "
                       "KGEN::CallOp support TBD.");
    }
  }
}

[[maybe_unused]] void DeadArgumentElimination::print() {
  llvm::dbgs() << "Live functions: \n";
  for (FuncOp func : liveFunctions)
    llvm::dbgs() << "  " << func.getSymName() << "\n";

  llvm::dbgs() << "Live values: \n";
  for (RetOrArg value : liveValues)
    llvm::dbgs() << "  " << value.getDescription() << "\n";
}

void DeadArgumentElimination::markLiveMaybeIndirectCalled() {
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](SymbolConstantAttr attr) {
    Operation *op =
        symbolTable.lookup(cast<FlatSymbolRefAttr>(attr.getSymbol()).getAttr());

    FuncOp func = dyn_cast_if_present<FuncOp>(op);
    if (!func)
      return;
    markLive(func);

    LLVM_DEBUG(
        llvm::dbgs()
        << "Function " << func.getName()
        << " maybe live due references other than a CallOp. Mark live.\n");
  });

  module.walk([&](Operation *op) {
    // Only support caller is CallOp for now, so that things like
    // AsyncCallOp, kgen.create_closure which are KGENCallOpInterface but
    // should be marked as always Live.
    if (isa<KGEN::CallOp>(op))
      return WalkResult::advance();

    walker.walk(op->getAttrDictionary());
    for (Type type : op->getResultTypes())
      walker.walk(type);
    for (Region &region : op->getRegions())
      for (Type type : region.getArgumentTypes())
        walker.walk(type);
    return WalkResult::advance();
  });
}

void DeadArgumentElimination::run() {
  markLiveMaybeIndirectCalled();

  // SurveyFunctions to determine liveliness for arguments and result values.
  // This loop needs to run before rewriting happens.
  std::vector<CallGraphNode *> nodes;
  nodes.reserve(callGraph.nodes.size());
  for (auto &[func, node] : callGraph.nodes) {
    surveyFunction(func);
    nodes.push_back(&node);
  }

  LLVM_DEBUG(print());

  // Remove arguments and return values and dead operations within a function
  // in parallel.
  mlir::parallelForEach(context, nodes, [&](CallGraphNode *node) {
    removeDeadStuffFromFunction(node);
  });

  // Rewrite callsites if callee's signature changed (in parallel).
  mlir::parallelForEach(context, nodes, [&](CallGraphNode *node) {
    rewriteCalleesFromFunction(node);
  });

  callGraph.nodes.clear();
}

void DeadArgumentEliminationPass::runOnOperation() {
  VerboseCompilerTimeTraceScope traceScope(
      "DeadArgumentEliminationPass::runOnOperation");

  const SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  CallGraph cg(symtab);
  cg.build(getOperation(), symtab);
  DeadArgumentElimination dae(cg, getOperation().getContext(), getOperation(),
                              symtab);
  dae.run();
}
