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

#ifndef KGEN_ELABORATOR_IREVALUATORCONTEXT_H
#define KGEN_ELABORATOR_IREVALUATORCONTEXT_H

#include "Mojo/Interpreter/BytecodeInterpreter.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/Threading/Shared.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// ImplNodeBase
//===----------------------------------------------------------------------===//

/// Create a new generator implementation node.
struct ImplNodeBase {
  ImplNodeBase(InstantiatedOpInterface inst, ParameterUseDefGraph &&graph)
      : inst(inst), paramGraph(std::move(graph)) {}

  ImplNodeBase(Region &scope) : paramGraph(scope) {}
  virtual ~ImplNodeBase() = default;

  /// Initialize the fields of the node if created with the single-argument
  /// constructor above.
  void initialize(InstantiatedOpInterface inst, ParameterUseDefGraph &&graph);

  virtual void setToError(ErrorTree &&err) = 0;

  /// This op represents a concrete instantiation of a generator.
  InstantiatedOpInterface inst;

  /// Keep track of the nested parameter scopes within this symbol.
  ParameterUseDefGraph paramGraph;

  /// An error contained by this node. This allows us to delay error handling in
  /// cases where an error is recoverable.
  std::optional<ErrorTree> error;
  /// An atomic indicating whether an error is present. This can be used to
  /// check for an error when the ImplNode is shared.
  std::atomic<bool> hasError = false;

  /// The elaborator will asynchronously dispatch elaboration of generator
  /// instantiations with no result parameters in separate tasks, deferring
  /// handling of the calls until they are complete. This atomic tracks the
  /// number of in-flight so-called "dependencies". Upon hitting zero, the
  /// elaborator will complete processing of this node by handling the calls in
  /// `dependencies`.
  std::atomic<size_t> numDependencies = 1;

  /// This flag is set when the implementation node is done processing. A
  /// separate flag is needed because an error state can cause the node to
  /// complete early. This flag prevents double-completion.
  std::atomic<bool> done = false;

  /// A chain representing SCC completion.
  AsyncValueRef<Chain> sccCh;
};

//===----------------------------------------------------------------------===//
// ParamNodeState
//===----------------------------------------------------------------------===//

/// The state of a parameter node, including the number of tasks waiting on it.
/// These are stored together in an atomic so that the waiter count and the
/// state can be transitioned atomically together, preventing erroneous waiters
/// from being counted due to a race.

class ParamNodeState {
  static constexpr uint64_t IN_PROGRESS_BIT = 62;
  static constexpr uint64_t DONE_BIT = 63;

public:
  /// All parameter nodes are created with `FRESH` status. When a worker has
  /// scheduled the first child of the node, it is moved to `IN_PROGRESS`. When
  /// all children complete processing, the state is moved to `DONE`. Use the
  /// upper 2 bits to represent the status.
  enum State { FRESH = 0, IN_PROGRESS = 1, DONE = 3 };

  /// Attempt to mark the status as `IN_PROGRESS`. Return the previous status.
  State markInProgress() {
    return static_cast<State>(
        value.fetch_or(static_cast<uint64_t>(1) << IN_PROGRESS_BIT) >>
        IN_PROGRESS_BIT);
  }

  /// Add a waiter. Return true if the task is not `DONE` and the waiter was
  /// successfully added.
  bool addWaiter() { return (value.fetch_add(1) >> DONE_BIT) == 0; }

  /// Mark the status as done and return the number of waiters at that time.
  size_t markDone() {
    return value.fetch_or(static_cast<uint64_t>(1) << DONE_BIT) << 2 >> 2;
  }

  /// Reset the state of the node to `FRESH`.
  void refresh() {
    value.fetch_xor(static_cast<uint64_t>(1) << IN_PROGRESS_BIT);
  }

  /// Get the current state.
  State getValue() {
    return static_cast<State>(value.load() >> IN_PROGRESS_BIT);
  }

private:
  /// The upper 2 bits of the value represent the status and the lower 62 bits
  /// the number of waiting tasks.
  std::atomic<uint64_t> value = 0;
};

//===----------------------------------------------------------------------===//
// ParamNodeBase
//===----------------------------------------------------------------------===//
struct ParamNodeBase {
  /// Create an expansion tree node to represent a generator instantiation.
  ParamNodeBase(AsyncRT::CPUDevice &cpuDevice, GeneratorOpInterface gen,
                ParameterExprArrayAttr vals, size_t depth)
      : gen(gen), inputParams(vals), depth(depth),
        paramCh(AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(cpuDevice)) {}

  /// Create a special root node. Root nodes can be identified with a null
  /// symbol.
  ParamNodeBase() = default;

  /// The generator represented by this node.
  GeneratorOpInterface gen;
  /// The input parameters with which the generator is being instantiated.
  ParameterExprArrayAttr inputParams;
  /// The current depth of the node. The depth varies based on the traversal
  /// order of the callgraph.
  size_t depth;

  /// The current state of the node. This flag is used to break recursion.
  ParamNodeState state;

  /// Return the mangled name of this ParamNode. Calculates it on first
  /// invocation.
  StringAttr getMangledName();

  /// An async value is ready if the underlying async value is in Active or
  /// Error state.
  bool getIsError() const { return done == DoneState::ERROR; }

  /// Make explicit copy of this AsyncValueRef, increasing the AsyncValue's
  /// refcount by one.
  AsyncValueRef<Chain> copy() const;

protected:
  /// The name of the parameterized node.
  std::atomic<const void *> mangledName = nullptr;

  /// The chain to signal when this parameter node is done processing.
  AsyncRT::AsyncValueRef<AsyncRT::Chain> paramCh;

  /// Atomic to prevent race on emplace.
  std::atomic<uint8_t> done = false;
  enum DoneState : uint8_t { NOT_DONE, DONE, ERROR };
};

//===----------------------------------------------------------------------===//
// IREvaluatorContext
//===----------------------------------------------------------------------===//

class IREvaluatorContext : public ParameterEvaluationContext {
public:
  IREvaluatorContext(EnvAttr env, MLIRContext *mlirCtx,
                     InterpreterState *state);

  // Transparent thunks
  //
  // A "transparent" thunk forwards to a wrapped function while delegating its
  // public identity (linkage name, LLVM metadata) to that function. There are
  // currently two flavors:
  //
  //   1. Conversion thunks (synthesized by `generateConversionThunk` in
  //      `ExprConversions.cpp`) that bridge calling conventions.
  //   2. Closure `__call__` methods (synthesized by `ClosureEmitter`) that
  //      forward to the closure's underlying impl.
  //
  // Both carry a `kTransparentThunkCalleeExprAttr` attribute holding a
  // parametric expression that resolves - once the thunk's paramDecls are
  // substituted with the callsite's paramValues - to a fully-bound
  // `SymbolConstantAttr` for the wrapped function. Substitution may involve
  // parameter operators (e.g. `GetWitnessAttr` for closures), so resolution
  // must run through an evaluation context.

  /// Look through a transparent thunk to its wrapped function, rebinding the
  /// thunk's `kTransparentThunkCalleeExprAttr` against `symbol`'s parameter
  /// values. Uses `this` as the evaluation context so parameter
  /// operators inside the callee expression are evaluated through the
  /// concrete subclass's hooks. Returns:
  ///   - `nullopt`: `generator` is not a transparent thunk; nothing to do.
  ///   - error: rebinding produced an error; the caller should propagate it.
  ///   - null `SymbolConstantAttr`: rebinding is not yet ready (a
  ///     sub-dependency is still being elaborated); the caller should
  ///     skip/retry.
  ///   - non-null `SymbolConstantAttr`: the fully resolved wrapped callee.
  std::optional<ErrorTreeOr<SymbolConstantAttr>>
  resolveTransparentThunkCallee(GeneratorOp generator,
                                SymbolConstantAttr symbol, Location loc);

protected:
  /// Evaluate an apply-like operator.
  FailureOr<TypedAttr> evaluateGetEnv(ParamOperatorAttr op);

  /// Evaluate POC::DataToStr "data_to_str" operator.
  FailureOr<TypedAttr> evaluateDataToStr(ParamOperatorAttr op, bool reset);

  FailureOr<StringAttr> evaluateStringPart(TypedAttr part, bool reset);

  FailureOr<TypedAttr>
  evaluateGetLinkageNameAttr(GetLinkageNameAttr getLinkageNameAttr);
  FailureOr<TypedAttr>
  evaluateGetSourceNameAttr(GetSourceNameAttr getSourceNameAttr);

  FailureOr<TypedAttr> evaluateGetTypeNameAttr(GetTypeNameAttr getTypeNameAttr);
  FailureOr<TypedAttr> evaluateIsStructTypeAttr(IsStructTypeAttr attr);
  FailureOr<TypedAttr> evaluateFnTypeIsCABIAttr(FnTypeIsCABIAttr attr);
  FailureOr<TypedAttr> evaluateGetBaseTypeNameAttr(GetBaseTypeNameAttr attr);
  FailureOr<TypedAttr> evaluateCompileOffloadClosureAttr(
      CompileOffloadClosureAttr compileOffloadClosureAttr);
  FailureOr<TypedAttr> evaluateCompileAssemblyAttr(CompileAssemblyAttr attr);

  /// Evaluate the mangled name of a function, including evaluating any explicit
  /// linkage name. This may return an empty StringAttr, signalling that the
  /// function is not yet ready.
  ErrorTreeOr<StringAttr> evaluateMangledName(TypedAttr symCst, bool sanitize,
                                              Location errorLoc,
                                              StringRef errorContext);

  std::string stringifyTypeInstanceRef(TypeInstanceRefAttr instanceRef,
                                       bool qualifiedBuiltins);
  void printParamValue(raw_ostream &os, ParamDeclAttr decl, TypedAttr value,
                       bool qualifiedBuiltins);

  /// The function to use to emit an error.
  std::function<void(ErrorTree)> emitError;

  /// The contextual location of an error.
  std::optional<Location> errorLoc;

  EnvAttr env;

  InterpreterState *state = nullptr;

protected:
  /// Helper to get the base type name from a type reference.
  /// Returns the name from the generator's valueDomainType, or nullptr if not
  /// a struct type.
  StringAttr getBaseTypeName(TypedAttr typeRef);

  /// Expose `getGenerator` for the function-reflection attrs that run
  /// through `evaluateWithContext` and need symbol-table access.
  FuncInterface resolveFunctionDecl(SymbolRefAttr symbol) override {
    return getGenerator(symbol);
  }

private:
  MLIRContext *mlirCtx = nullptr;

  virtual ParamNodeBase *lookupParamNodeBase(SymbolRefAttr symbol) = 0;

  virtual GeneratorOp getGenerator(SymbolRefAttr symbol) = 0;

  virtual void addDeferredFunction(OwningOpRef<FuncOp> func) = 0;

  virtual ErrorOr<CrossDeviceFunction>
  compileAsm(MLIRContext *ctx, GeneratorOp, SymbolConstantAttr, StringAttr,
             TargetInfoAttr, EmitAs, EmissionOptions emissionOptions) = 0;

  virtual ImplNodeBase *getParentNode() = 0;
};

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

SymbolConstantAttr extractSymbolConstantAttr(TypedAttr attr);

/// Decomposed view of a `compile_offload` op's `func` attribute. Populated by
/// `extractOffloadFunc` when the func resolves to a fully-bound
/// `SymbolConstantAttr` referencing a valid generator in the supplied symbol
/// table.
struct OffloadFunc {
  SymbolConstantAttr symbol;
  GeneratorOp generator;
};

/// Decompose a `compile_offload` op's `func` attribute. Returns an error if
/// the func does not resolve to a `SymbolConstantAttr`, is not fully bound,
/// or does not reference a valid generator in `symtab`.
ErrorTreeOr<OffloadFunc> extractOffloadFunc(CompileOffloadOp op,
                                            mlir::SymbolTable &symtab);

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_IREVALUATORCONTEXT_H
