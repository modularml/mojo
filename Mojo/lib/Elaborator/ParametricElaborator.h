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

#ifndef KGEN_ELABORATOR_PARAMETRICELABORATOR_H
#define KGEN_ELABORATOR_PARAMETRICELABORATOR_H

#include "Cache/CachedTransform.h"
#include "InterpreterCache.h"
#include "Mojo/Interpreter/BytecodeInterpreter.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "ParametricIREvaluator.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/Threading/Shared.h"
#include "Support/Threading/ThreadLocalCache.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/ThreadLocalCache.h"
#include "mlir/Transforms/Passes.h"

namespace M::KGEN {

/// This struct represents the expansion of a callgraph during elaboration.
struct ParametricExpansionGraph {
  ParametricExpansionGraph(AsyncRT::CPUDevice &cpuDevice)
      : worklistCh(AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(cpuDevice)),
        quiesceChain(
            AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(cpuDevice)) {}

  virtual ~ParametricExpansionGraph();

  /// Increment the task count.
  void didAddTask();
  /// Decrement the task count.
  void didCompleteTask();

  /// Map from generator instantiation to expansion tree node.
  Shared<
      llvm::MapVector<std::pair<ParameterExprArrayAttr, GeneratorOpInterface>,
                      std::unique_ptr<PParamNode>>>
      nodes;

  /// The current number of tasks scheduled anywhere in the elaborator on the
  /// worklist.
  std::atomic<size_t> numWorkItems = 1;
  /// This chain is signalled when all active work items have completed. This is
  /// used to starve the workqueue before running evaluators, because evaluation
  /// cannot be reliably performed while the compiler is doing work on other
  /// threads.
  AsyncRT::AsyncValueRef<AsyncRT::Chain> worklistCh;

  /// Concrete functions added directly to the expansion graph.
  std::vector<std::unique_ptr<PImplNode>> elaboratedNodes;

  /// Number of outstanding resources created from this cpuDevice.
  std::atomic<size_t> numOutstandingResources = 1;
  /// If quiesce() has been called, the chain it returned. Otherwise null.
  AsyncValueRef<Chain> quiesceChain;
};

//===----------------------------------------------------------------------===//
// ParametricElaborator
//===----------------------------------------------------------------------===//

/// This is the Elaborator that interprets parametric functions

/// This class provides the elaborator, which constructs the expansion tree as
/// it walks the IR and specializes operations. This outputs IR that has been
/// fully specialized/concretized, with the appropriate functions
/// multi-versioned.

class ParametricElaborator : public InterpreterCache {
public:
  /// Initialize the elaborator and its symbol table.
  ParametricElaborator(SymbolTable &symtab,
                       ParameterCollector::Analysis &paramCache,
                       TargetInfoAttr target, const CompilationOptions &options,
                       ElaboratorCompileAsmFn compileAsmFn,
                       ElaboratorCompileOffloadFn compileOffloadFn,
                       const ElaborateGeneratorsOptions &config);

  //===--------------------------------------------------------------------===//
  // ParametricIREvaluator Interface
  //===--------------------------------------------------------------------===//

  /// Concretize all non-parametric symbol references within the provided
  /// parameter expression.
  ErrorTreeOr<Attribute>
  concretizeSymbolsWithin(Attribute value, PImplNode *parent, Location loc);

  /// Lookup an existing concrete function.
  InstantiatedOpInterface lookupInstantiatedOp(StringAttr name) {
    InstantiatedOpInterface localResult = (*tlInsts)[name];
    if (!localResult) {
      localResult =
          concreteNodes.read([name](auto &map) { return map.at(name)->inst; });
    }
    return localResult;
  }

  GeneratorOpInterface lookupGeneratorOp(StringAttr name) {
    return oldSymTab.lookup<GeneratorOpInterface>(name);
  }

  PImplNode *lookupImplNode(SymbolRefAttr symbol) {
    StringAttr name = cast<FlatSymbolRefAttr>(symbol).getAttr();
    return concreteNodes.read([name](auto &map) { return map.at(name); });
  }

  /// Look up the callee symbol. If it's a FuncOp, return it. Otherwise,
  /// elaborate the generator or interface and return the first concrete
  /// implementation. Return none if the specialization is not ready yet.
  ErrorTreeOr<FuncOp> getConcreteFunction(PImplNode *parent, Location loc,
                                          SymbolConstantAttr symbol);

  /// Look up the callee symbol (for a struct generator). If elaboration is
  /// already in-progress or done, return the concrete symbol reference.
  /// Otherwise, dispatch an elaboration of the struct generator and immediately
  /// return the concrete symbol reference (the symbol may not exist yet, but
  /// its elaboration will have been scheduled).
  ErrorTreeOr<TypeInstanceRefAttr>
  getConcreteStructTypeReference(PImplNode *parent, Location loc,
                                 TypeGeneratorRefAttr genref);

  /// Add an owned function operation that should be appended to the module at
  /// the end of elaboration. This is where generated functions during
  /// elaboration should go.
  void addDeferredFunction(OwningOpRef<FuncOp> func);

  /// Get the target associated with this instance of the elaborator.
  TargetInfoAttr getTarget() const { return target; }
  /// Get the elaborator config.
  const ElaborateGeneratorsOptions &getOptions() const { return config; }

  /// Lookup and return a reference to a cached interpreter result in the
  /// current thread. Populates the value from the global cache if necessary.
  TypedAttr lookupCachedInterpretation(Operation *op,
                                       ParameterExprArrayAttr operands,
                                       SymbolConstantAttr symbol) {
    TypedAttr &tlValue = (*tlInterp)[{op, operands, symbol}];
    if (!tlValue) {
      tlValue = interpCache.read([op, operands, symbol](auto &map) {
        return map.lookup({op, operands, symbol});
      });
    }
    return tlValue;
  }
  /// Write a cached interpreter result to the global cache if it does not
  /// already exist. Always returns the value that is in the cache.
  std::pair<TypedAttr, bool>
  writeGlobalCachedInterpretation(Operation *op,
                                  ParameterExprArrayAttr operands,
                                  SymbolConstantAttr symbol, TypedAttr value) {
    TypedAttr result = value;
    (*tlInterp)[{op, operands, symbol}] = value;

    bool inserted;

    interpCache.modify([op, operands, value, symbol, &result,
                        &inserted](InterpreterMapTy &map) {
      auto [it, success] = map.insert({{op, operands, symbol}, value});
      inserted = success;

      if (!inserted)
        result = it->second;
    });
    if (result != value) {
      static std::mutex mutex;
      std::lock_guard<std::mutex> guard(mutex);
      llvm::errs() << "\nresult != value\n";
      llvm::errs() << "result: " << result << "\n";
      llvm::errs() << "value: " << value << "\n";
      llvm::errs() << "operands: " << operands << "\n";
      llvm::errs() << "\n";
      op->dump();
    }

    assert(result == value && "non-deterministic interpreter results");
    return std::make_pair(result, inserted);
  }

  //===--------------------------------------------------------------------===//
  // Entry Point
  //===--------------------------------------------------------------------===//

  /// Given a list of primary generators (i.e. generators with no input
  /// parameters), run the elaborator. This will generate an expansion tree
  /// rooted on the module with base nodes for each primary generator. Once
  /// specialization completes we will be able to collect all the concrete
  /// implementations for each primary generator and handle any renaming or
  /// fixup that needs to happen to produce the output IR.
  LogicalResult run(ModuleOp theModule,
                    ArrayRef<std::pair<GeneratorOp, ParameterExprArrayAttr>>
                        primaryGenerators);

private:
  //===--------------------------------------------------------------------===//
  // Utility Functions
  //===--------------------------------------------------------------------===//
  bool addConcreteFunc(FuncOp func, StringAttr name,
                       DenseMap<StringAttr, PImplNode *> &map) {
    auto inst = cast<InstantiatedOpInterface>(*func);
    auto node = std::make_unique<PImplNode>(inst, /*parent=*/nullptr,
                                            func.getBodyRegion());
    bool result = map.try_emplace(name, node.get()).second;
    g.elaboratedNodes.push_back(std::move(node));
    return result;
  }

  /// Once a concrete instance has finished specializing, finish processing the
  /// instance and call the verifier.
  void finalizeInstance(PImplNode *node);

  //===--------------------------------------------------------------------===//
  // Operation Processing
  //===--------------------------------------------------------------------===//

  /// Parameter constants are the only operation that can bridge the parameter
  /// domain into the cpuDevice domain. Use it to root concretization of nested
  /// symbol references.
  template <typename OpT>
  ElaborationState processParamConstantOp(PImplNode *parent, OpT op);

  /// Process a `kgen.param.apply` operation by concretizing its operands and
  /// callee and then invoking the interpreter.
  ElaborationState
  processParamApplyOp(PImplNode *inode,
                      ParamApplyOp op); //,
                                        // SymbolConstantAttr calleeSymbol);//,
                                        // FuncOp func, GeneratorOp gen);

  /// Process a call op by binding any necessary input parameters from the
  /// symbol or the call and passing them on to processGeneratorUser.
  ElaborationState processCallOp(PImplNode *parent,
                                 GeneratorUserOpInterface call);

  /// Process a generator user. In general, this is anything that can call into
  /// a generator and might therefore need to be multi-versioned.
  ElaborationState processGeneratorUser(GeneratorUserOpInterface user,
                                        SymbolConstantAttr calleeSymbol,
                                        PImplNode *parent);

  /// Process a param.if op by evaluating the condition and elaborating and
  /// inlining only the branch that was taken. If one of the branches had an
  /// early return, this will split the block after the return and avoid
  /// elaborating the rest of the function.
  ElaborationState processParamIfOp(PImplNode *parent, ParamIfOp op);

  /// Process a param.for op by evaluating the sequence of induction variables
  /// and then instantiating the body for each value of the sequence.
  ElaborationState processParamForOp(PImplNode *parent, ParamForOp op);

  ElaborationState processCodeGenReachableOp(PImplNode *inode,
                                             CodeGenReachableOp op);

  //===--------------------------------------------------------------------===//
  // Worklist
  //===--------------------------------------------------------------------===//

  /// Signal the worklist to tell it a job has completed or has been taken off
  /// the workqueue.
  void signalWorklist() {
    if (g.numWorkItems.fetch_sub(1) == 1)
      g.worklistCh.copy().emplace();
  }
  /// This function is run by tasks that process nodes.
  void processImplNodeTask(PImplNode *node);
  /// Schedule an implementation node on the AsyncRT work queue.
  void scheduleImplNode(PImplNode *inode);
  /// Process the scopes within an implementation node. This function returns
  /// `success` if all scopes completed processing and the node is completely
  /// elaborated. If the function returns `failure`, that means elaboration of
  /// the node hit a suspension point and must halt before completion.
  LogicalResult processImplNode(PImplNode *inode);
  /// Complete processing of an implementation node. If all dependencies have
  /// been completed, this will process them, performing any required
  /// multi-versioning, and finalize and verify the function, unless an error
  /// has occurred.
  void completeImplNodeProcessing(PImplNode *inode);
  /// Process a worklist of ops. Returns error if processing the scope resulted
  /// in an error, returns `skipFrame` if the processing of the current scope
  /// scope should be pre-empted with a new scope, returns `skipNode` if
  /// processing the current implementation node should suspended.
  ElaborationState processScope(PImplNode *node, PImplNode::WorkItem &item);
  /// Process a single operation. Returns error if processing the scope resulted
  /// in an error, returns `skipFrame` if the processing of the current scope
  /// should be pre-empted with a new scope, returns `skipNode` if processing
  /// the current implementation node should be suspended.
  ElaborationState processOp(PImplNode *node, Operation *op);

  /// Process a CompileOffloadOp by replace its parametric attributes
  // to concrete values.
  ElaborationState processCompileOffload(PImplNode *node, CompileOffloadOp op);
  /// bundling CompileOffloadOps into one sliced Module.
  ErrorTreeOrSuccess bundleOffloadModules();

  ErrorTreeOrSuccess bundleCompileOffloadOp(CompileOffloadOp op,
                                            StringAttr name);

  /// Process a deferred op by replacing it with an operation and attributes it
  /// stores.
  ElaborationState processDeferredOp(PImplNode *node, DeferredOp op);

  //===--------------------------------------------------------------------===//
  // Specialization
  //===--------------------------------------------------------------------===//

  PParamNode *getOrCreateNode(ParameterExprArrayAttr values,
                              GeneratorOpInterface gen, size_t depth);

  /// Request specialization of the generator at `genNode`. If the node is ready
  /// complete, then the function returns `advance` and the concrete functions
  /// can be retrieved from the node. Otherwise, the function returns
  /// `skipNode`, indicating that elaboration of the current function should be
  /// suspended.
  ElaborationState specializeGenerator(PImplNode *inode, PParamNode *genNode,
                                       Location from, bool addWaiter);

  /// Instantiate a generator reference by retrieving the concrete
  /// implementations of a reference. If this function returns `advance` but
  /// `inputParamKey` is not populated, then the callee is a direct function
  /// reference.
  std::pair<ElaborationState, PImplNode *> instantiateGeneratorReference(
      PImplNode *parent, Operation *user, SymbolConstantAttr calleeSymbol,
      ParameterExprArrayAttr &inputParamKey, GeneratorOpInterface &gen,
      function_ref<bool(PParamNode *)> shouldWait = [](PParamNode *) {
        return true;
      });
  /// Given a user of a completed parameter node, collect concrete
  /// implementations whose bindings are consistent with the current node.
  FailureOr<PImplNode *> collectConcreteImplementations(Location loc,
                                                        PImplNode *parent,
                                                        PParamNode *calleeNode);

  /// Attempt to diagnose concrete recursion and break recursion where possible.
  /// Return true if recursion was broken at least once. The "generation" is
  /// used to know whether a visited flag is valid. It must be at least 1,
  /// because the initial generation is always 0.
  bool diagnoseAndBreakRecursion(unsigned generation,
                                 ArrayRef<PParamNode *> roots);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// The target we are compiling code for.
  TargetInfoAttr target;
  /// The compilation options.
  const CompilationOptions &options;

  /// The elaborator config.
  ElaborateGeneratorsOptions config;

  /// This is the symbol table of the new module.
  /// Map from name to op + concrete function to implementation node.
  Shared<DenseMap<StringAttr, PImplNode *>> concreteNodes;

  mlir::ThreadLocalCache<DenseMap<StringAttr, InstantiatedOpInterface>> tlInsts;

  /// This is used to cache interpreter invocations across the elaborator.
  using InterpreterMapTy = DenseMap<
      std::tuple<Operation *, ParameterExprArrayAttr, SymbolConstantAttr>,
      TypedAttr>;
  Shared<InterpreterMapTy> interpCache;
  mlir::ThreadLocalCache<InterpreterMapTy> tlInterp;

  using AttrTypeMapTy =
      DenseMap<std::tuple<Region *, ParameterExprArrayAttr>,
               DenseMap<std::pair<size_t, const void *>, const void *>>;

  Shared<AttrTypeMapTy> paramInterpCache;
  mlir::ThreadLocalCache<AttrTypeMapTy> tlParamInterpCache;

  /// This symbol table allows efficient lookups across the module.
  SymbolTable &oldSymTab;

  /// The environment defines the module is being elaborated with.
  EnvAttr env;

  /// Hash table of known ParameterUseDefGraphs. This ensures we only compute a
  /// graph once for each generator. This is extra state generated by
  /// specializeGenerator that is *required for correctness* - this will cause
  /// issues with caching (though it would be easy to simply recompute) unless
  /// we create a ParametricNode or something we can use to store these in a
  /// proper data structure.
  Shared<DenseMap<GeneratorOpInterface,
                  std::unique_ptr<FunctionParameterUseDefGraph>>>
      knownGraphs;

  /// The AsyncRT cpuDevice instance to use.
  AsyncRT::CPUDevice &cpuDevice;

  /// The callgraph being expanded.
  ParametricExpansionGraph g;

  /// This is the cached parameter collector analysis.
  ThreadLocalCache<ParameterCollector::Analysis> paramCache;

  /// Callbacks to use for JIT functionalities.
  ElaboratorCompileAsmFn compileAsmFn;

  /// Callbacks to use for bundling offload functions.
  ElaboratorCompileOffloadFn compileOffloadFn;

  /// Deferred generated symbols to append to the module.
  SmallVector<FuncOp> deferredSymbols;

  /// Bundled offload functions for different targets.
  Shared<llvm::MapVector<TargetInfoAttr, OffloadInfo>> targetOffloadInfos;

  Shared<llvm::SetVector<CompileOffloadOp>> compileOffloadOps;

  friend class ParametricIREvaluator;

public:
  std::atomic<uint64_t> paramReplacerCacheTlsHits = 0;
  std::atomic<uint64_t> paramReplacerCacheGlobalHits = 0;
  std::atomic<uint64_t> paramReplacerCacheMisses = 0;

  std::atomic<uint64_t> applyCacheMisses = 0;
  std::atomic<uint64_t> applyCacheTlsHit = 0;
  std::atomic<uint64_t> applyCacheGlobalHit = 0;
  std::atomic<uint64_t> applyCacheDuplicates = 0;

  Shared<llvm::DenseMap<uint64_t, uint64_t>> cacheSizes;

  Shared<llvm::DenseMap<const void *, std::tuple<Attribute, Type, uint64_t>>>
      replacerStats;
};

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_PARAMETRICELABORATOR_H
