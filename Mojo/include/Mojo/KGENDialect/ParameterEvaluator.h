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

#ifndef KGEN_KGENDIALECT_PARAMETEREVALUATOR_H
#define KGEN_KGENDIALECT_PARAMETEREVALUATOR_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/ParameterReplacer.h"
#include "Support/ForwardDecls.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FunctionExtras.h"

namespace mlir {
class LockedSymbolTableCollection;
} // namespace mlir

namespace M {
class Error;
} // namespace M

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// Helper methods for inspecting possibly-parameterized attributes and types.
//===----------------------------------------------------------------------===//

// NOTE: None of these are particularly efficient, because they walk the whole
// IR tree without caching.

/// Given a parameter expression, walk it and return any references to named
/// parameters. `requiredSignatureDepth` is set to the minimum signature depth
/// needed for the expression to be valid. This fails if an invalid parameter
/// expression exists.
void collectParameterReferences(Attribute attr,
                                SmallVectorImpl<ParamDeclRefAttr> &results,
                                bool &hasCtxEvalExpr,
                                size_t &requiredSignatureDepth);

/// Given a potentially-parameterized MLIR type, walk it and return any
/// references to named parameters. `requiredSignatureDepth` is set to the
/// minimum signature depth needed for the type to be valid. This fails if an
/// invalid parameter expression exists.
void collectParameterReferences(Type type,
                                SmallVectorImpl<ParamDeclRefAttr> &results,
                                bool &hasCtxEvalExpr,
                                size_t &requiredSignatureDepth);

/// Return true if the specified type contains parameter references, e.g.
/// `!kgen.scalar<dt>` returns true, but `!kgen.scalar<f32>` returns false.
bool isParameterizedType(Type type);

class ParameterEvaluationContext;
class ParameterEvaluator;

//===----------------------------------------------------------------------===//
// ParameterEvaluationContext
//===----------------------------------------------------------------------===//

/// A resolved struct handle containing the generator and optionally the
/// concrete instance.
///
/// There are three possible states on success:
/// 1. `decl` is null: Async concretization was triggered. The caller should
///    return null to signal retry.
/// 2. `decl` is valid, `instance` is null: Generator is available but no
///    concrete instance. Use generator + rebinding for evaluation.
/// 3. `decl` is valid, `instance` is valid: Concrete instance is available.
struct ResolvedStructHandle {
  /// The struct declaration operation. Null if async concretization was
  /// triggered and the caller should retry later.
  StructDeclInterface decl;
  /// The parameter values for the struct. Valid when `decl` is valid.
  ArrayRef<TypedAttr> paramValues;
  /// A context-specific opaque handle for the context to track additional
  /// state. Valid when `decl` is valid.
  void *handle = nullptr;
  /// The concrete struct instance operation, if available.
  StructInstanceOp instance = nullptr;
};

/// This class is used by ParameterEvaluator to evaluate
/// ContextuallyEvaluatedAttrInterface instances, which are attributes whose
/// evaluation may be context-dependent. Sub-classes can store state to help
/// with evaluation.
///
/// The use of this separate context/policy provider class allows
/// ParameterEvaluators, which are stateful and may need to be instantiated
/// multiple times, to be decoupled from the logic of evaluating attributes.
///
/// The base class implements common dispatch logic for evaluating attributes
/// like GetWitnessAttr, StructFieldTypesAttr, etc. Derived classes provide
/// context-specific behavior through virtual hooks:
///   - resolveStructOp(): resolve a type-value to a StructDeclInterface
///   - withEvaluator(): provide a context-appropriate ParameterEvaluator
///   - evaluateContextSpecific(): handle context-specific attributes
class ParameterEvaluationContext {
public:
  virtual ~ParameterEvaluationContext();

  /// Evaluate the provided attribute. First tries context-specific evaluation
  /// via evaluateContextSpecific(), then falls back to the attribute's
  /// evaluateWithContext() interface method.
  ///
  /// If the attribute is not evaluatable, returns failure(). This does not
  /// indicate an unexpected situation, but rather no further evaluation was
  /// possible.
  FailureOr<TypedAttr>
  evaluateExpression(ContextuallyEvaluatedAttrInterface attr);

  /// Resolve a type value to its struct declaration interface.
  ///
  /// The `acceptAsync` argument controls behavior when a concrete instance
  /// is not yet ready:
  /// - If acceptAsync=true AND context supports async: triggers async
  ///   elaboration and returns a handle with null `decl` to signal retry.
  /// - If acceptAsync=false OR context doesn't support async: returns the
  ///   generator (and instance is provided only if already available).
  ///
  /// Returns:
  /// - failure(): An error occurred during resolution.
  /// - success() with null `decl`: Async triggered, caller should retry.
  /// - success() with valid `decl`: Generator available. Check `instance`
  ///   to see if concrete info can be used directly.
  virtual FailureOr<ResolvedStructHandle> resolveStructOp(TypedAttr typeValue,
                                                          bool acceptAsync) = 0;

  /// Resolve the conformance op for a struct and trait in this context.
  /// For now, this conformance op is always parametric (non-concrete).
  ///
  /// The default implementation always looks up the conformance in the
  /// resolved struct's generator.This is correct for every context except the
  /// parser, where conformances are instead resolved via `ASTDecl` lookup
  /// (might need a lazy resolution).
  virtual Operation *resolveConformanceForStruct(ResolvedStructHandle resolved,
                                                 TraitSymbolAttr traitSymbol);

  /// Create an evaluator configured with the provided parameters.
  /// This callback style is necessary because we have derived evaluators that
  /// carry their own state. Eventually we should move all state to the context
  /// so that evaluator creation can be simplified.
  virtual void
  withEvaluator(ArrayRef<ParamDeclAttr> paramDecls,
                ArrayRef<TypedAttr> paramValues,
                llvm::function_ref<void(ParameterEvaluator &)> callback) = 0;

  /// Handle context-specific attributes that aren't covered by the base class.
  /// Returns failure() if the attribute cannot be evaluated, which is the
  /// normal case for unsupported attributes.
  virtual FailureOr<TypedAttr>
  evaluateContextSpecific(ContextuallyEvaluatedAttrInterface attr);

  /// Returns true if this is a materialization context where evaluated
  /// expressions will persist as constants. In such contexts, ill-formed
  /// expressions should emit errors. In non-materialization contexts
  /// (e.g., speculative partial-evaluation during lowering), ill-formed
  /// expressions simply persist unevaluated.
  virtual bool isMaterializationContext() const;

  /// Emit an error for an ill-formed expression during materialization.
  /// This is a no-op in non-materialization contexts, but callers can
  /// always call this method regardless of context.
  virtual void emitMaterializationError(const Twine &message);

  /// Returns the target info for the context, which may not be available in all
  /// contexts. If target info is not available, returns null.
  virtual TargetInfoAttr getTargetInfo() const { return {}; }

  /// Resolve a function symbol to its defining op as a `FuncInterface`
  /// handle. Reflection-side evaluators access signature info through the
  /// returned op via `FuncInterface`, `DeclInterface`, and
  /// `ExportInterface`. The concrete op is the elaborator-stage
  /// `kgen.generator` or the parser-stage `lit.fn`; both implement all
  /// three interfaces, which is what lets reflection work in both phases.
  ///
  /// Returns null if the symbol does not resolve to a function in this
  /// context.
  virtual FuncInterface resolveFunctionDecl(SymbolRefAttr symbol) = 0;
};

/// An evaluation context that exposes a LockedSymbolTableCollection.
class SymTabEvaluationContext : public ParameterEvaluationContext {
public:
  Operation *module;
  mlir::LockedSymbolTableCollection &symtab;

  SymTabEvaluationContext(Operation *module,
                          mlir::LockedSymbolTableCollection &symtab)
      : module(module), symtab(symtab) {}

protected:
  /// Resolve struct info for KGEN dialect structs.
  FailureOr<ResolvedStructHandle> resolveStructOp(TypedAttr typeValue,
                                                  bool acceptAsync) override;

  /// Resolve a function symbol via the symbol table.
  FuncInterface resolveFunctionDecl(SymbolRefAttr symbol) override;

  /// Provide a ParameterEvaluator configured for the struct parameters.
  void withEvaluator(
      ArrayRef<ParamDeclAttr> paramDecls, ArrayRef<TypedAttr> paramValues,
      llvm::function_ref<void(ParameterEvaluator &)> callback) override;

  FailureOr<TypedAttr>
  evaluateContextSpecific(ContextuallyEvaluatedAttrInterface attr) override;

private:
  FailureOr<TypedAttr> inlineApply(ParamOperatorAttr apply);
};

//===----------------------------------------------------------------------===//
// ParameterEvaluator
//===----------------------------------------------------------------------===//

/// This class keeps a set of defined parameter bindings and is used to evaluate
/// and simplify parameter expressions based on those values.
///
/// This class keeps track of two kinds of parameter bindings at the same time:
///
/// 1. Decl-based (name-based) bindings, which are used to substitute
///    ParamDeclRefAttrs.
/// 2. Index-based bindings, which are used to substitute ParamIndexRefAttrs.
///
/// - Rules for ParamDeclRefAttr:
/// If the name referenced by the ParamDeclRefAttr is not registered with the
/// evaluator, the attribute is left unchanged.
///
/// - Rules for ParamIndexRefAttr:
/// The attr/type initially passed to the evaluator is considered to be at
/// depth `inputDepth` (0 for most cases, but can be overridden by the user).
/// Only index references that point back to index bindings at this depth are
/// candidates for substitution (see IRAIDAI and PSTIAIRAID for details).
///
/// In addition, the evaluator will assert if the index is not less than the
/// size of all the index bindings. If partial substitution is being performed
/// (i.e. not all the index bindings are registered, only a given prefix of the
/// index bindings are registered), the user can manually set the total number
/// of index bindings so that the assertion is only triggered for real errors.
class ParameterEvaluator final : public ParameterReplacer<ParameterEvaluator> {
public:
  /// Instantiate a new parameter evaluator with the given parameter values.
  ParameterEvaluator(ArrayRef<ParamDeclAttr> paramDecls,
                     ArrayRef<TypedAttr> declBindings);
  /// Instantiate a new parameter evaluator with the given input parameters.
  ParameterEvaluator(ArrayRef<TypedAttr> declBindings);

  /// Instantiate a new parameter evaluator with the given parameter values.
  ParameterEvaluator(
      DenseMap<StringAttr, TypedAttr> declBindings =
          DenseMap<StringAttr, TypedAttr>(),
      ArrayRef<TypedAttr> indexBindings = SmallVector<TypedAttr>(),
      size_t inputDepth = 0)
      : declBindings(std::move(declBindings)),
        indexBindings(std::move(indexBindings)), inputDepth(inputDepth) {}

  /// Set the evaluation context to use.
  void setEvaluationContext(ParameterEvaluationContext *context) {
    evaluationContext = context;
  }
  ParameterEvaluationContext *getEvaluationContext() const {
    return evaluationContext;
  }

  /// Set a value for the specified parameter declaration to the specified
  /// simplified value.
  void setDeclBinding(StringAttr name, TypedAttr value,
                      bool overwrite = false) {
    assert(overwrite ||
           !declBindings.count(name) && "parameter already declared!");
    declBindings[name] = value;
  }
  void setDeclBinding(ParamDeclAttr decl, TypedAttr value,
                      bool overwrite = false) {
    setDeclBinding(decl.getName(), value, overwrite);
  }

  void setRewritten(
      const DenseMap<std::pair<size_t, const void *>, const void *> &value) {
    rewritten = value;
  }

  const DenseMap<std::pair<size_t, const void *>, const void *> &
  getRewritten() {
    return rewritten;
  }

  bool overwriteDeclBinding(ParamDeclAttr decl, TypedAttr value) {
    auto iter = declBindings.find(decl.getName());
    bool exist = iter != declBindings.end();
    declBindings[decl.getName()] = value;
    return exist;
  }

  /// Iterate over the current parameter values.
  const DenseMap<StringAttr, TypedAttr> &getDeclBindings() const {
    return declBindings;
  }

  /// Overwrite the current set of parameter values.
  void setDeclBindings(const DenseMap<StringAttr, TypedAttr> &values) {
    declBindings = values;
  }

  /// Get the specified type with any nested parameter expressions rewritten.
  Type getReboundType(Type type) { return replace(type); }

  /// Get the specified attribute with any nested parameter expressions
  /// rewritten.
  Attribute getReboundAttribute(Attribute attr) { return replace(attr); }

  /// Get the specified attribute with any nested parameter expressions
  /// rewritten.
  TypedAttr getReboundAttribute(TypedAttr attr) { return replace(attr); }

  TypedAttr getFailableReboundAttribute(TypedAttr attr) {
    return failableReplace(attr);
  }

  /// Dump the parameter evaluator state.
  void dump() const;

  /// Append an index-based parameter binding.
  void appendIndexBinding(TypedAttr value) { indexBindings.push_back(value); }
  void overwriteIndexBinding(size_t idx, TypedAttr value) {
    assert(idx < indexBindings.size() && "invalid index");
    if (indexBindings[idx] != value) {
      indexBindings[idx] = value;
      clearCache();
    }
  }

  /// Return the number of input parameter values that have been added.
  size_t getNumIndexBindings() const { return indexBindings.size(); }
  /// Get all the input parameters.
  ArrayRef<TypedAttr> getIndexBindings() const { return indexBindings; }

  /// Set the relative input depth.
  void setInputDepth(size_t depth) { inputDepth = depth; }
  size_t getInputDepth() const { return inputDepth; }

  void clearCache() { rewritten.clear(); }

private:
  // CRTP methods.
  Type doReplace(Type type, size_t rootDepth);
  Attribute doReplace(Attribute attr, size_t rootDepth);
  friend class ParameterReplacer<ParameterEvaluator>;

  /// Handle the `cond` operator. This needs to return a tri-state: whether the
  /// condition can be narrowed to an integer constant and whether we need to
  /// suspend, which is that the bool represents.
  std::pair<IntegerAttr, bool> narrowCondOp(Attribute attr, size_t rootDepth);

  /// These are the name-based parameter bindings.
  DenseMap<StringAttr, TypedAttr> declBindings;

  /// These are the top-level index-based parameter bindings. This list is
  /// allowed to contain null entries.  When encountered, the parameter replacer
  /// will leave ParamIndexRefAttr referring to them unchanged (actually will
  /// remap the type if needed).
  SmallVector<TypedAttr> indexBindings;

  /// The optional context to use for evaluating contextually evaluated
  /// attributes.
  ParameterEvaluationContext *evaluationContext = nullptr;

public:
  /// The relative depth from the generator where the index-based parameter
  /// bindings are from. This is zero for most applications, but should be set
  /// accordingly when substituting attributes or types inside a generator, see
  /// PSTIAIRAID.
  size_t inputDepth = 0;
};

//===----------------------------------------------------------------------===//
// Helper methods involving parameter evaluation.
//===----------------------------------------------------------------------===//

/// A partially specialized input parameter specification.
struct PartiallySpecializedInputParams {
  ParameterEvaluator evaluator;
  SmallVector<Type, 16> unboundParamTypes;
  llvm::BitVector boundParams;

  /// Bitmask over the source generator's body constraints: a set bit drops
  /// the matching constraint from the specialized metadata. Empty (the
  /// default) retains every body constraint; otherwise the bitvector must
  /// match the generator's body-constraint list in size.
  llvm::BitVector dischargedBodyConstraints;

  /// Given an input parameter specification `paramTypes` and the full set of
  /// bindings `paramBindings`, create a partially specialized input parameter
  /// specification.
  ///
  /// The bindings may be partially specified, with holes represented by
  /// UnboundAttrs.
  static std::optional<PartiallySpecializedInputParams>
  from(ArrayRef<Type> paramTypes, ArrayRef<TypedAttr> paramBindings,
       ParameterEvaluationContext *evaluationContext,
       function_ref<InFlightDiagnostic()> emitErrorFn,
       const llvm::BitVector &dischargedBodyConstraints = llvm::BitVector());
};

//===----------------------------------------------------------------------===//
// ParameterEvaluatorFrame
//===----------------------------------------------------------------------===//

/// A per-stack-frame wrapper around ParameterEvaluator used by the parametric
/// elaborator's interpreter. Holds the evaluator plus caching metadata for the
/// ParameterReplacer's rewritten cache, allowing cache reuse when re-entering
/// the same parameter domain with the same parameters.
struct ParameterEvaluatorFrame {
  ParameterEvaluator evaluator;

  /// Operation as part of cache key.
  Operation *cachedOpKey = nullptr;
  /// Region as part of the cache key.
  Region *cachedRegionKey = nullptr;
  /// Parameters as part of the cache key.
  ParameterExprArrayAttr cachedAttrKey;
  /// A flag to note if the cache has been memorized.
  bool foundCached = false;

  ParameterEvaluatorFrame() = default;
  explicit ParameterEvaluatorFrame(const ParameterEvaluator &eval)
      : evaluator(eval) {}
};

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_PARAMETEREVALUATOR_H
