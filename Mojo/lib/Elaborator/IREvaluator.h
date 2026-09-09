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

#ifndef KGEN_ELABORATOR_IREVALUATOR_H
#define KGEN_ELABORATOR_IREVALUATOR_H

#include "Cache/CachedTransform.h"
#include "IREvaluatorContext.h"
#include "Mojo/Interpreter/BytecodeInterpreter.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/Threading/Shared.h"
#include "mlir/Support/IndentedOstream.h"

namespace M::KGEN {
class Elaborator;
class FuncOp;
struct ImplNode;
struct ParamNode;
struct ExpansionGraph;

//===----------------------------------------------------------------------===//
// IREvaluator
//===----------------------------------------------------------------------===//

/// This IR evaluator can work during elaboration to concretize parameter
/// expressions and compute symbolic parameter expressions, such as `apply` on
/// a symbol constant or `get_sizeof` and `get_alignof` a decl type.
///
/// Owns a ParameterEvaluator for parameter substitution and provides
/// elaboration-specific handling for ParamOperatorAttr, GetLinkageNameAttr,
/// etc. via the BytecodeInterpreter.
class IREvaluator : public IREvaluatorContext, public BytecodeInterpreter {
public:
  /// Construct the IR evaluator with a symbol table for evaluating symbolic
  /// expressions.
  IREvaluator(Elaborator &elaborator, ImplNode *parent);
  IREvaluator(const IREvaluator &other);

protected:
  /// Handle elaboration-specific attributes (ParamOperator, GetLinkageName,
  /// etc.)
  FailureOr<TypedAttr>
  evaluateContextSpecific(ContextuallyEvaluatedAttrInterface attr) override;

  /// Resolve a type value to its struct declaration.
  /// Returns null decl if acceptAsync is true and the instance is not yet ready
  /// (signaling retry).
  FailureOr<ResolvedStructHandle> resolveStructOp(TypedAttr typeValue,
                                                  bool acceptAsync) override;

  /// Create a nested evaluator with the provided parameter bindings.
  void withEvaluator(
      ArrayRef<ParamDeclAttr> paramDecls, ArrayRef<TypedAttr> paramValues,
      llvm::function_ref<void(ParameterEvaluator &)> callback) override;

  /// Returns true since elaboration is a materialization context.
  bool isMaterializationContext() const override { return true; }

  /// Emit an error using the elaborator's error infrastructure.
  void emitMaterializationError(const Twine &message) override;

  TargetInfoAttr getTargetInfo() const override;

public:
  /// Given a generic parameter expression, substitute known values for
  /// parameters into it and fold it down to a simple constant. This returns an
  /// error if a simple constant cannot be produced (e.g. because there is some
  /// dependence on target information that isn't available).
  ErrorTreeOr<Attribute> concretizeParameterExpr(ImplNode *parent, Location loc,
                                                 Attribute expr);
  ErrorTreeOr<Type> concretizeParameterExpr(ImplNode *parent, Location loc,
                                            Type expr);

  /// Lookup the body of the referenced function. Ensure the function is
  /// inflated as well.
  ErrorOr<Region *> lookupFunctionBody(SymbolRefAttr symbol) override;

  /// Evaluate the function with the provided constant inputs.
  ErrorTreeOr<TypedAttr> evaluateFunction(FuncOp func,
                                          ArrayRef<TypedAttr> inputs);

  /// Evaluate the result slot function with the provided constant inputs.
  ErrorTreeOr<TypedAttr>
  evaluateFunctionWithResultSlot(FuncOp func, ArrayRef<TypedAttr> inputs);

  /// Set the location to associate errors with.
  void setErrorLoc(Location loc) { errorLoc = loc; }

  /// Get compilation error limit from the elaborator.
  int getErrorLimit();

  /// Get error with prelude setting from the elaborator.
  bool getElabErrorIncludePrelude();

  /// Bind a parameter declaration to a value.
  void setDeclBinding(ParamDeclAttr decl, TypedAttr value) {
    evaluator.setDeclBinding(decl, value);
  }

private:
  /// Evaluate an apply-like operator.
  FailureOr<TypedAttr> evaluateApplyLike(ParamOperatorAttr op,
                                         bool withResultSlot);
  FailureOr<TypedAttr> evaluateStringAddress(ParamOperatorAttr op);
  FailureOr<TypedAttr> evaluateGetBaseTypeNameAttr(GetBaseTypeNameAttr attr);

  /// Helper to get the base generator symbol name from a type reference.
  /// Returns the generator's symbol name, or nullptr if not a struct type.
  StringAttr getGeneratorSymbolName(TypedAttr typeRef);

  ParamNodeBase *lookupParamNodeBase(SymbolRefAttr symbol) override;

  GeneratorOp getGenerator(SymbolRefAttr symbol) override;

  ErrorOr<CrossDeviceFunction>
  compileAsm(MLIRContext *ctx, GeneratorOp, SymbolConstantAttr, StringAttr,
             TargetInfoAttr, EmitAs, EmissionOptions emissionOptions) override;

  void addDeferredFunction(OwningOpRef<FuncOp> func) override;

  ImplNodeBase *getParentNode() override;

  /// The parameter evaluator for substituting parameter bindings.
  ParameterEvaluator evaluator;

  /// A reference to the elaborator instance. The elaborator is invoked to
  /// concretize symbol constants prior to interpreting them.
  Elaborator *elaborator;

  /// The contextual node being elaborated.
  ImplNode *parent = nullptr;
};

//===----------------------------------------------------------------------===//
// ImplNode
//===----------------------------------------------------------------------===//

/// This struct represents a concrete instantiation of a generator -- generators
/// may have multiple concrete instantiations -- and contains the current state
/// of elaboration for that concrete instance.
struct ImplNode : public ImplNodeBase {
  /// Create a new generator implementation node.
  ImplNode(InstantiatedOpInterface inst, ParamNode *parent,
           ParameterUseDefGraph &&graph)
      : ImplNodeBase(inst, std::move(graph)), parent(parent) {}

  ImplNode(ParamNode *parent);
  virtual ~ImplNode() = default;

  /// Take the provided error and set this node to an `error` state. Erase all
  /// state dominated by this node.
  void setToError(ErrorTree &&err) override;

  /// Get the current active evaluator instance.
  IREvaluator &getEvaluator() { return stack.back().evaluator; }

  /// The parent expansion tree node.
  ParamNode *parent;
  struct WorkItem {
    /// The operations to process.
    std::vector<Operation *> ops;
    /// The completion callback. This function is invoked when the processing of
    /// a scope completes. The callback should perform any necessary cleanup and
    /// additional work scheduling if necessary. The callback is passed the
    /// current node that owns the work item, and it is allowed to set errors,
    /// access operations, modify bindings and worklists, etc. It is imperative
    /// that the callback closure does not capture any operation handles but
    /// that it accessing them through the node. This is because nodes can be
    /// cloned and the operations get remapped.
    std::function<LogicalResult(ImplNode *)> onComplete;

    /// The evaluator to use. We need one per work item because each represents
    /// a distinct parameter scope.
    IREvaluator evaluator;
  };

  /// The current stack of worklists and scopes.
  std::vector<WorkItem> stack;

  /// This is the list of deferred generator instantiations via calls that need
  /// to be handled when the implementation node is complete and all its
  /// dependencies are ready.
  std::vector<std::pair<Location, ParamNode *>> dependencies;

  /// Intra-SCC edges removed by diagnoseAndBreakRecursion to break the cycle.
  /// Checked in the second pass of completeImplNodeProcessing to propagate
  /// errors from callee nodes that errored during the first pass.
  std::vector<std::pair<Location, ParamNode *>> sccRemovedDeps;

  /// The downstream nodes blocking elaboration of this node. All of them must
  /// complete before elaboration of this node can continue; dropping one
  /// deadlocks elaboration since the wait is invisible to recursion diagnosis.
  SmallVector<std::pair<Location, ParamNode *>, 1> blockers;

  /// Outstanding blocker resumes plus a guard held while the node is being
  /// processed; the node is re-run when this drops to zero.
  std::atomic<size_t> pendingResumes = 0;

  /// Suppresses blocker registration during location concretization, which
  /// cannot yield and restart.
  bool suppressBlockers = false;
};

//===----------------------------------------------------------------------===//
// ParamNode
//===----------------------------------------------------------------------===//

/// This struct is a node in the expansion tree that describes the elaboration.
/// In general, we try to limit effects to a single subtree. The only exception
/// is that creating new generators/funcs generally are children of the root -
/// this is because they're semi-independent of the current node and will
/// elaborate to something concrete we can simply refer to. We try to track
/// dependencies in order to make that graph explicit.
struct ParamNode : public ParamNodeBase {
  /// Create an expansion tree node to represent a generator instantiation.
  ParamNode(AsyncRT::CPUDevice &cpuDevice, GeneratorOpInterface gen,
            ParameterExprArrayAttr vals, size_t depth,
            ExpansionGraph *expansionGraph)
      : ParamNodeBase(cpuDevice, gen, vals, depth), impl(this),
        expansionGraph(expansionGraph) {
    assert(expansionGraph && "Expansion graph cannot be null");
  }

  /// Create a special root node. Root nodes can be identified with a null
  /// symbol.
  ParamNode() : impl(nullptr, this, ParameterUseDefGraph(nullptr)) {}

  /// Return the first concrete node in the subtree rooted on `this`. This is
  /// often called from a node that is either concrete, or only has one
  /// concretization. For generality in cases where the full list of concrete
  /// nodes is required, use getAllConcrete below. Returns an error if there are
  /// no concretizations of this node.
  ErrorTreeOr<ImplNode *> getFirstConcreteNode();

  /// Get the first concrete FuncOp. This finds the first concrete node in the
  /// subtree, and returns its op cast to a FuncOp. This is always safe because
  /// if the node has been concretized, then the op is a FuncOp.
  ErrorTreeOr<FuncOp> getFirstConcreteFunc();

  /// Return an error if expansion of this parameter node failed. If any
  /// implementation succeeded, return success instead.
  ErrorTreeOrSuccess collectErrorsOrSuccess();

  /// The instantiation of the parametric function.
  ImplNode impl;

  /// Add a waiter to the cpuDevice and report task completion to
  /// ParamNodeRuntime.
  void andThenAsync(AsyncValue::Waiter &&waiter);

  /// Construct the async value. This will notify waiters.
  void emplace();

  /// Construct the async value to error state. This is used when we want to
  /// abandon tasks that are not complete.
  void setToError();

private:
  /// The cpuDevice manages the set of tasks kicked off in a given process. The
  /// ParamNode alerts the cpuDevice upon creation and completion of tasks so
  /// that the cpuDevice can sync tasks.
  ExpansionGraph *expansionGraph;
};

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_IREVALUATOR_H
