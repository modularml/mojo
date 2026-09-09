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

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/ForkJoin.h"
#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/AttrTypeMangler.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/TransformUtils/CallGraphUtils.h"
#include "Mojo/TransformUtils/InliningUtils.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Context.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/STLExtras.h"
#include "Support/Threading/ThreadLocalCache.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "kgen-inlining"

using namespace M;
using namespace KGEN;

/// Insert a new parameter declaration into all nested declaration scopes.
static void propagateNewDecls(ArrayRef<ParamDeclAttr> newDecls,
                              ParameterUseDefGraph &topLevelGraph,
                              ParameterUseDefGraph &graph, Operation *declOp,
                              Region *declScope) {
  // Populate the new declarations into the call scope graph.
  for (ParamDeclAttr decl : newDecls) {
    graph.decls.try_emplace(
        decl.getName(), ParamDeclaration{decl.getType(), declOp, declScope});
  }
  // Recurse on nested scopes.
  for (Region *nestedDecl : graph.nestedDecls) {
    propagateNewDecls(newDecls, topLevelGraph,
                      topLevelGraph.nestedScopes.find(nestedDecl)->second,
                      declOp, declScope);
  }
}

//===----------------------------------------------------------------------===//
// inlineGeneratorCall
//===----------------------------------------------------------------------===//

/// Get the nearest declaration from the operation and the region of the
/// declaration that contains the operation.
static Region *getNearestDeclRegion(Operation *op) {
  Region *region = op->getParentRegion();
  auto decl = dyn_cast<DeclInterface>(region->getParentOp());
  while (!decl) {
    region = region->getParentRegion();
    decl = dyn_cast<DeclInterface>(region->getParentOp());
  }
  return region;
}

/// Generator inputs and results cross parameter domains. Make sure to rebind
/// them if necessary.
static SmallVector<Value> rebindValues(OpBuilder &b, Location loc,
                                       ValueRange inputs, TypeRange outputs) {
  SmallVector<Value> newValues;
  newValues.reserve(inputs.size());
  for (auto [input, output] : llvm::zip(inputs, outputs))
    if (input.getType() != output)
      newValues.push_back(b.createOrFold<RebindOp>(loc, output, input));
    else
      newValues.push_back(input);
  return newValues;
}

/// The operands of returns cross parameter domains. Make sure to rebind them if
/// necessary.
static SmallVector<Value>
rebindReturnOperands(OpBuilder &b, Operation *newReturn, Operation *call) {
  return rebindValues(b, newReturn->getLoc(), newReturn->getOperands(),
                      call->getResultTypes());
}

using MangleDefTy = function_ref<void(const ParamDefinition &, Region *,
                                      ParameterUseDefGraph *, IRMapping &)>;

/// Recursively mangle parameter definitions within the inlined scope
/// corresponding to the callee's use def graph. The mangling callback is
/// more or less arbitrary.
static void recursivelyMangleDefs(IRMapping &map, Region *calleeRegion,
                                  const ParameterUseDefGraph &calleeGraph,
                                  ParameterUseDefGraph &topLevelGraph,
                                  MangleDefTy mangleDef) {
  const ParameterUseDefGraph *calleeNestedGraph =
      &calleeGraph.nestedScopes.find(calleeRegion)->second;
  Region *clonedNestedRegion =
      &map.lookup(calleeRegion->getParentOp())
           ->getRegion(calleeRegion->getRegionNumber());
  for (auto &[_, def] : calleeNestedGraph->defs) {
    mangleDef(def, clonedNestedRegion,
              &topLevelGraph.nestedScopes.find(clonedNestedRegion)->second,
              map);
  }
  for (Region *calleeNestedRegion : calleeNestedGraph->nestedDecls) {
    recursivelyMangleDefs(map, calleeNestedRegion, calleeGraph, topLevelGraph,
                          mangleDef);
  }
}

static void inlineGeneratorCall(GeneratorOp caller, CallOp call,
                                GeneratorOp callee, InlineLevel level,
                                ParameterUseDefGraph &topLevelGraph,
                                const ParameterUseDefGraph &calleeParams,
                                const llvm::SetVector<StringAttr> &calleeDecls,
                                AttrTypeMangler::Cache &manglerCache,
                                NameUniquer &nameUniquer, bool updateDebugInfo,
                                bool debugCallsite) {
  VerboseCompilerTimeTraceScope traceScope(
      "inlineGeneratorCall", [&] { return callee.getSymName().str(); });

  StringAttr label = StringAttr::get(call.getContext(), "inlined_cf_scope");

  // Get the parameters in-scope at the callsite.
  Region *scopeRegion = getNearestDeclRegion(call);
  ParameterUseDefGraph *callScope =
      scopeRegion == topLevelGraph.scope
          ? &topLevelGraph
          : &topLevelGraph.nestedScopes.find(scopeRegion)->second;

  IRRewriter b{OpBuilder(call)};
  AttrTypeMangler mangler(manglerCache);
  bool needsMangling =
      mangler.populate(b, nameUniquer, calleeDecls, topLevelGraph);

  // Make sure to rebind the call operands based on the mangled types of the
  // callee's argument types.
  SmallVector<Type> argTypes = llvm::to_vector(callee.getArgumentTypes());
  if (needsMangling)
    for (Type &type : argTypes)
      type = mangler.mangleRefsIn(type);

  b.setInsertionPointAfter(call);
  if (debugCallsite && callee.getLocScope())
    DebugInfo::LineTableLocOp::create(b, call->getLoc());

  SmallVector<Value> argVals =
      rebindValues(b, call.getLoc(), call->getOperands(), argTypes);

  // Use a LoopOp to be able to break to a label - any returns inlined from
  // callee must only exit the inlined block.
  auto scope = HLCF::LoopOp::create(b, call.getLoc(), call->getResultTypes(),
                                    ValueRange(), label);
  b.createBlock(&scope.getBody());

  // Map the callee inputs.
  IRMapping map;
  for (auto [value, arg] : llvm::zip(argVals, callee.getArguments()))
    map.map(arg, value);
  for (Operation &op : *callee.getBody())
    b.clone(op, map);

  // Clone the nested parameter use-def graphs into the current set of
  // nested graphs.
  callee.walk([&](DeclInterface containedScope) {
    if (containedScope == callee)
      return;
    Operation *clonedScope = map.lookup(&*containedScope);
    for (auto [region, clonedRegion] :
         llvm::zip(containedScope->getRegions(), clonedScope->getRegions())) {
      const ParameterUseDefGraph &nestedGraph =
          calleeParams.nestedScopes.find(&region)->second;
      [[maybe_unused]] bool inserted =
          topLevelGraph.nestedScopes
              .try_emplace(&clonedRegion, nestedGraph.copy(map))
              .second;
      assert(inserted);
    }
  });
  // Re-acquire `callScope` since the reference could have been invalidated
  // by the insertions into `calleeParams.nestedScopes`.
  callScope = scopeRegion == topLevelGraph.scope
                  ? &topLevelGraph
                  : &topLevelGraph.nestedScopes.find(scopeRegion)->second;
  // Decl scopes that were nested under the callee are now nested under the
  // current call scope.
  for (Region *nestedDecl : calleeParams.nestedDecls) {
    callScope->nestedDecls.push_back(
        &map.lookup(nestedDecl->getParentOp())
             ->getRegion(nestedDecl->getRegionNumber()));
  }

  // Do name mangling.
  if (needsMangling) {
    for (Operation *user : calleeParams.paramOps) {
      // Skip the parent decl. It's handled after.
      if (user == callee)
        continue;
      Operation *cloned = map.lookup(user);
      mangler.mangleElementsIn(cloned);
    }
  }

  /// Since the callee might contain nested parameter scopes (e.g.
  /// `kgen.param.if`), we recursively walk them and mangle parameter
  /// definitions.
  auto mangleDef = [&mangler, &needsMangling, &topLevelGraph](
                       const ParamDefinition &def, Region *scopeRegion,
                       ParameterUseDefGraph *defScope, IRMapping &map) {
    Operation *cloned = map.lookup(def.defOp);
    mangler.mangleElementsIn(cloned);
    // Rename declarations.
    auto itf = cast<ParamOpInterface>(def.defOp);
    SmallVector<ParamDeclAttr> newDecls;
    itf.walkDeclarations([&](ParamDeclAttr decl) {
      newDecls.push_back(mangler.mangleDecl(decl, needsMangling));
    });
    cast<ParamOpInterface>(cloned).renameDeclarations(newDecls);

    // At this point, the only nested ops that declares parameters in their
    // scope are ParamDeclareRegionOp and ParamForOp, whose declarations need
    // special treatment.
    if (needsMangling) {
      if (auto regionDecl = dyn_cast<ParamDeclareRegionOp>(cloned)) {
        SmallVector<ParamDeclAttr> newInputDecls;
        for (ParamDeclAttr decl : regionDecl.getInputParams())
          newInputDecls.emplace_back(mangler.mangleDecl(decl, needsMangling));
        regionDecl.setInputParams(newInputDecls);
        newDecls.append(newInputDecls);
      } else if (auto paramFor = dyn_cast<ParamForOp>(cloned)) {
        ParamDeclAttr newDecl =
            mangler.mangleDecl(paramFor.getParamDecl(), needsMangling);
        paramFor.setParamDeclAttr(newDecl);
        newDecls.push_back(newDecl);
      }
    }

    // Populate the new declarations into the call scope graph.
    propagateNewDecls(newDecls, topLevelGraph, *defScope, cloned, scopeRegion);
  };
  for (auto &[_, def] : calleeParams.defs) {
    // Skip the parent decl. It's handled after.
    if (def.defOp == callee)
      continue;
    mangleDef(def, scopeRegion, callScope, map);
  }
  for (Region *calleeNestedRegion : calleeParams.nestedDecls) {
    recursivelyMangleDefs(map, calleeNestedRegion, calleeParams, topLevelGraph,
                          mangleDef);
  }

  if (needsMangling) {
    for (Region *nestedScope : calleeParams.nestedDecls) {
      Operation *clonedOp = map.lookup(nestedScope->getParentOp());
      Region &clonedRegion =
          clonedOp->getRegion(nestedScope->getRegionNumber());
      mangler.recursivelyMangle(&clonedRegion, topLevelGraph);
    }
  }

  // Mangle the DeclInterface declarations.
  // TODO: mangle result parameter names as well.
  b.setInsertionPoint(call);
  for (auto [origDecl, value] :
       llvm::zip(callee.getInputParams(), call.getParamValues())) {
    ParamDeclAttr decl = mangler.mangleDecl(origDecl, needsMangling);
    auto declOp = ParamDeclareOp::create(
        b, call.getLoc(), decl,
        ParamOperatorAttr::getRebind(value, decl.getType()));
    // Register the new declaration.
    propagateNewDecls(decl, topLevelGraph, *callScope, declOp, scopeRegion);
  }

  // When building in debuginfo, mangle parameters in all op locations.
  if (updateDebugInfo) {
    OpRegionBlockWalker walker(
        [&](Operation *op) {
          op->setLoc(cast<LocationAttr>(mangler.mangleRefsIn(op->getLoc())));
          return WalkResult::advance();
        },
        nullptr,
        [&](Block *block) {
          for (BlockArgument arg : block->getArguments())
            arg.setLoc(cast<LocationAttr>(mangler.mangleRefsIn(arg.getLoc())));
          return WalkResult::advance();
        });
    walker.walk(&scope.getBody());
  }

  // Handle all terminators.
  unsigned numReturns = 0;
  callee.getBodyRegion().walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<SourceLocOp>(op)) {
      processSourceLocOp(cast<SourceLocOp>(map.lookup(op)), call.getLoc(), b);
      return WalkResult::advance();
    }

    // If we inlined a "tail" call into a caller,  we need to clear the "tail"
    // marker.  This marker indicates that the callee doesn't access the
    // parent-caller's frame, but it might access the grand-caller's frame.
    TypeSwitch<Operation &>(*op).Case<CallOp, CallParamOp, CallIndirectOp>(
        [&](auto op) {
          auto cloned = cast<decltype(op)>(map.lookup(op));
          if (cloned.getTailKind() == TailKind::Tail)
            cloned.setTailKind(TailKind::None);
          return WalkResult::advance();
        });

    // Walk over nested functions. Control-flow does not cross them.
    if (isa<FuncInterface>(op))
      return WalkResult::skip();
    if (!isa<ReturnOp>(op))
      return WalkResult::advance();

    Operation *cloned = map.lookup(op);
    b.setInsertionPoint(cloned);
    ++numReturns;
    HLCF::BreakOp::create(b, cloned->getLoc(),
                          rebindReturnOperands(b, cloned, call), label);
    cloned->erase();
    return WalkResult::advance();
  });
  b.replaceOp(call, scope.getResults());

  // Always update the callsite locations. We need them to be intact for the
  // elaborator to give useful error messages.
  bool singleExit =
      numReturns == 1 && isa<ReturnOp>(callee.getBody()->getTerminator());
  maybeUpdateDebugInfo(scope, StringAttr(), singleExit);
}

//===----------------------------------------------------------------------===//
// InlineGraph
//===----------------------------------------------------------------------===//

namespace {
/// An inlining graph is a call graph between functions of concrete calls to
/// functions that must be inlined. The root nodes of the graph are
/// `always_inline` functions with no calls to other such functions, and the
/// leaf nodes are non-inlined functions.
///
/// This data structure is used to inline functions starting from the leaves of
/// callgraphs. This is more efficient because inlining from the roots of the
/// callgraph leads to duplicate work (splats callgraph into a tree). It also
/// enables inlined functions to be optimized and pruned as they are processed.
///
/// This structure is implemented as a CRTP class so that the core algorithm can
/// be shared between both inliners.
template <typename DerivedT, typename NodeT>
struct InliningGraphBase : public CallGraphBase<DerivedT, NodeT> {
  explicit InliningGraphBase(AsyncRT::CPUDevice &cpuDevice)
      : cpuDevice(cpuDevice), state(cpuDevice) {}

  using CallGraphBase<DerivedT, NodeT>::getDerived;

  /// Process the graph by performing all requested inlining from the root
  /// nodes.
  void process();

  // Complete processing of a node by incrementing the number of processed calls
  // of all its callers. Note that the same function can appear in the caller
  // list N, indicating that it calls this function N times. This loop will
  // increment the `numProcessedCalls` counters N times as appropriate.
  void complete(NodeT *node);

  /// The cpuDevice to use.
  AsyncRT::CPUDevice &cpuDevice;

  /// The inlining task state.
  AsyncRT::ForkJoin state;
  /// The number of nodes that complete processing. If this is not equal to the
  /// number of nodes, then there are cycles in the graph.
  std::atomic<size_t> numProcessed = 0;
};
} // namespace

template <typename DerivedT, typename NodeT>
void InliningGraphBase<DerivedT, NodeT>::complete(NodeT *node) {
  // Run the function pipeline after inlining has been performed for a function.
  // Make sure the verifier is off. Note that `Pass::runPipeline` is not thread
  // safe due to analysis manager nesting.
  getDerived().onComplete(node);

  // Since the function is complete, compute its callee graph, if it has
  // any callers.
  numProcessed.fetch_add(1);
  if (!getDerived().prepareForInlining(node))
    return;

  // Indicate it as complete to its callers by incrementing the ready counter on
  // the caller nodes. Schedule any ready callers.
  for (NodeT *caller : node->callers) {
    if (caller->numProcessedCalls.fetch_add(1) + 1 != caller->callsites.size())
      continue;
    // This caller is ready. Increment the number of active work items.
    state.fork([caller, this] {
      // Compute the parameter use-def graph of the function as a caller.
      // Inline all callees.
      getDerived().performInlining(caller);
      complete(caller);
    });
  }
}

template <typename DerivedT, typename NodeT>
void InliningGraphBase<DerivedT, NodeT>::process() {
  VerboseCompilerTimeTraceScope traceScope("InliningGraphBase::process");

  // Populate the worklist with root nodes.
  for (auto &[func, node] : this->nodes) {
    // Root nodes are already complete.
    if (!node.callsites.empty())
      continue;
    NodeT *caller = &node;
    // Increment the number of in-flight tasks.
    state.fork([caller, this] { complete(caller); });
  }
  // Wait on all active work items.
  state.join();
}

//===----------------------------------------------------------------------===//
// ParametricInliningGraph
//===----------------------------------------------------------------------===//

namespace {
struct ParametricInliningGraphNode
    : public CallGraphNodeBase<ParametricInliningGraphNode, GeneratorOp,
                               CallOp> {
  explicit ParametricInliningGraphNode(GeneratorOp func)
      : CallGraphNodeBase(func), level(func.getInlineLevel()),
        calleeParamGraph(func.getBodyRegion()) {}
  ParametricInliningGraphNode(ParametricInliningGraphNode &&other)
      : CallGraphNodeBase(other.func), level(other.level),
        calleeParamGraph(other.func.getBodyRegion()) {}

  /// Compute the caller parameter graph and declarations.
  void calculateParams(ParameterCollector::Analysis &paramCache);

  /// The inlining level of the function.
  InlineLevel level;
  /// In parametric inlining, each function has its parameter use-def graph
  /// computed twice: once as a caller, computed when the node is being
  /// processed, and once as a callee, when the fully processed node is called
  /// from somewhere else. Stash the callee graph on the node itself.
  ParameterUseDefGraph calleeParamGraph;
  /// A set of all declarations, regardless of type, in the callee.
  llvm::SetVector<StringAttr> allDecls;
  /// The number of processed calls. When the value of this counter equals the
  /// size of `callsites`, then all calls for this function have been processed.
  std::atomic<size_t> numProcessedCalls = 0;
};

struct ParametricInliningGraph
    : public InliningGraphBase<ParametricInliningGraph,
                               ParametricInliningGraphNode> {
  explicit ParametricInliningGraph(InlineLevel level,
                                   AsyncRT::CPUDevice &cpuDevice,
                                   ParameterCollector::Analysis &paramCache,
                                   unsigned optimizationLevel,
                                   bool updateDebugInfo)
      : InliningGraphBase(cpuDevice), level(level),
        paramCaches(paramCache,
                    cpuDevice.getWorkQueue()->getParallelismLevel()),
        manglerCaches(baseManglerCache,
                      cpuDevice.getWorkQueue()->getParallelismLevel()),
        optimizationLevel(optimizationLevel), updateDebugInfo(updateDebugInfo) {
  }

  void onComplete(ParametricInliningGraphNode *node) {}

  /// CallGraphBase interface for whether to add the node to the graph.
  bool shouldAddToGraph(CallOp call, ParametricInliningGraphNode *node) {
    return shouldInline(node);
  }

  /// Return number of operations within a call
  uint64_t getNumOperations(ParametricInliningGraphNode *node) const;

  /// Return true if it's profitable to inline the call
  bool isProfitableToInline(ParametricInliningGraphNode *node,
                            uint64_t threshold) const;

  /// Only inline functions that satisfy the inlining level.
  bool shouldInline(ParametricInliningGraphNode *node) const {
    assert(node->level == node->func.getInlineLevel());

    // Even if we've not been told to, inline some 'always_inline' functions if
    // they're not too big.
    bool shouldInlineAutomatically = node->level == InlineLevel::Always;
    // Inline nodes that meet the pass's minimum inline level
    bool shouldAlwaysInline =
        (node->level >= level && node->level != InlineLevel::Never);
    bool isWithinLimits = this->getNumOperations(node) <
                          getInlineThreshold(node->level, shouldAlwaysInline);
    return (shouldInlineAutomatically || shouldAlwaysInline) && isWithinLimits;
  }

  /// When a function is finished processing and will be inlined, compute is
  /// callee parameter graph.
  bool prepareForInlining(ParametricInliningGraphNode *node);
  /// Inline all functions by invoking the parametric inliner.
  void performInlining(ParametricInliningGraphNode *caller);

  /// The inlining level.
  InlineLevel level;
  /// Base mangler cache instance. It is always empty.
  AttrTypeMangler::Cache baseManglerCache;
  /// Thread local parameter collector caches.
  ThreadLocalCache<ParameterCollector::Analysis> paramCaches;
  /// Thread local mangler caches.
  ThreadLocalCache<AttrTypeMangler::Cache> manglerCaches;

  /// Get inlining threshold based optimization level and inline level.
  uint64_t getInlineThreshold(InlineLevel level, bool upperLimit) const;

  /// Compiler optimization level.
  unsigned optimizationLevel;

  /// Whether DebugInfo should be updated or not.
  bool updateDebugInfo;
};
} // namespace

void ParametricInliningGraphNode::calculateParams(
    ParameterCollector::Analysis &paramCache) {
  calleeParamGraph.calculate(paramCache);
  func.walk([&](Operation *op) {
    if (auto decl = dyn_cast<DeclInterface>(op)) {
      for (ParamDeclAttr decl : decl.getInputParams())
        allDecls.insert(decl.getName());
    }
    if (auto paramOp = dyn_cast<ParamOpInterface>(op)) {
      paramOp.walkDeclarations(
          [&](ParamDeclAttr decl) { allDecls.insert(decl.getName()); });
    }
  });
}

bool ParametricInliningGraph::prepareForInlining(
    ParametricInliningGraphNode *node) {
  // Skip inlining of functions with no callers.
  if (node->callers.empty())
    return false;
  node->calculateParams(paramCaches.getThreadLocalCache());
  return true;
}

uint64_t ParametricInliningGraph::getInlineThreshold(InlineLevel level,
                                                     bool upperLimit) const {
  // Functions without an explicit 'always_inline' decorator should not be
  // inlined as we are unable to choose a meaningful heuristic.
  if (level == InlineLevel::Never || level == InlineLevel::Automatic)
    return 0;

  // Add an upper bound to constrain compile times.
  // TODO: refine this threshold
  if (upperLimit || level >= InlineLevel::AlwaysNoDebug)
    return 5000;

  if (auto clOpt = KGENPassCLOptions::parametricInlineThreshold())
    return *clOpt;
  switch (optimizationLevel) {
  case 0:
    // TODO: refine this threshold
    return 0;
  case 1:
    // TODO: refine this threshold
    return 2;
  case 2:
    // TODO: refine this threshold
    return 5;
  case 3:
    // That threshold value shows the best compile time on matmul benchmark for
    // a given isProfitableToInline heuristic and optimization level.
    // It may need to be refined in the future if isProfitableToInline changes.
    return 35;
  default:
    return 0;
  }
}

uint64_t ParametricInliningGraph::getNumOperations(
    ParametricInliningGraphNode *node) const {
  // Estimated trip count of the `param.for` loop.
  const uint64_t avgLoopIters =
      *KGENPassCLOptions::parametricInlineEstimatedLoopTripCount();

  std::function<uint64_t(Operation *)> walker =
      [&](Operation *operation) -> uint64_t {
    if (!operation)
      return 0;
    uint64_t count = 0;
    operation->walk([&](Operation *op) {
      if (auto loop = dyn_cast<ParamForOp>(op)) {
        // At this point trip count is not known, therefore use some heuristic
        // to estimate number of iterations.
        uint64_t loopSize = 0;
        for (Operation &oper : loop.getBody().front())
          loopSize += walker(&oper);
        count += avgLoopIters * loopSize;
        // No need to traverse the loop body as it has been already visited.
        return WalkResult::skip();
      }

      if (!isa_and_nonnull<DebugInfo::DebugInfoDialect>(op->getDialect()))
        ++count;

      return WalkResult::advance();
    });
    return count;
  };

  return walker(node->func);
}

bool ParametricInliningGraph::isProfitableToInline(
    ParametricInliningGraphNode *node, uint64_t threshold) const {
  const uint64_t numOps = getNumOperations(node);
  const uint64_t numCalls = node->callsites.size();
  return numCalls == 1 || numCalls * numOps < threshold;
}

void ParametricInliningGraph::performInlining(
    ParametricInliningGraphNode *caller) {
  ParameterUseDefGraph callerParams(caller->func.getBodyRegion());
  callerParams.calculate(paramCaches.getThreadLocalCache());
  NameUniquer uniquer(callerParams, callerParams);
  for (auto [call, callee] : caller->callsites) {
    // Verify we want to inline this - it may have grown in size since we first
    // checked.
    if (shouldInline(callee))
      inlineGeneratorCall(caller->func, call, callee->func, callee->level,
                          callerParams, callee->calleeParamGraph,
                          callee->allDecls, manglerCaches.getThreadLocalCache(),
                          uniquer, updateDebugInfo, !optimizationLevel);
  }
}

//===----------------------------------------------------------------------===//
// InlineParametricPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_INLINEPARAMETRIC
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct InlineParametricPass : impl::InlineParametricBase<InlineParametricPass> {
  explicit InlineParametricPass(const InlineParametricOptions &options = {})
      : InlineParametricBase(options) {}
  void runOnOperation() override;
};
} // namespace

void InlineParametricPass::runOnOperation() {
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  auto &paramCache = getAnalysis<ParameterCollector::Analysis>();

  AsyncRT::CPUDevice &cpuDevice =
      *loadContext(&getContext())->get<AsyncRT::CPUDevice>();
  ParametricInliningGraph graph(
      nodebugOnly ? InlineLevel::AlwaysNoDebug : InlineLevel::Always, cpuDevice,
      paramCache, optimizationLevel, updateDebugInfo);
  graph.build(getOperation(), symtab);
  graph.process();

  // Do one quick pass to inline any call to a function that is ready to be
  // inlined, in case cycles prevent us from inlining trivial functions.
  //
  // Note: we choose not to iterate because most nodebug inline functions should
  // be trivial and not have calls to recursive functions.
  auto inlineReadyFn = [&graph, updateDebugInfo = updateDebugInfo.getValue(),
                        optimizationLevel = optimizationLevel.getValue()](
                           ParametricInliningGraphNode &caller) {
    // Skip nodes that are completely processed.
    if (caller.numProcessedCalls == caller.callsites.size())
      return;
    ParameterUseDefGraph callerParams(caller.func.getBodyRegion());
    callerParams.calculate(graph.paramCaches.getThreadLocalCache());
    NameUniquer uniquer(callerParams, callerParams);
    for (auto [call, callee] : caller.callsites) {
      // Skip nodes that are not complete.
      if (callee->numProcessedCalls != callee->callsites.size())
        continue;
      if (!graph.isProfitableToInline(
              callee, graph.getInlineThreshold(InlineLevel::Always,
                                               /*upperLimit=*/false)))
        continue;
      inlineGeneratorCall(caller.func, call, callee->func, callee->level,
                          callerParams, callee->calleeParamGraph,
                          callee->allDecls,
                          graph.manglerCaches.getThreadLocalCache(), uniquer,
                          updateDebugInfo, !optimizationLevel);
    }
  };

  // Note: use the same threadpool as before, because that's what the thread
  // local caches are initialized for.
  AsyncRT::ForkJoin state(cpuDevice);
  for (ParametricInliningGraphNode &caller :
       llvm::make_second_range(graph.nodes))
    state.fork([&] { inlineReadyFn(caller); });
  state.join();
}
