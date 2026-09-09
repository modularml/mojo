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
// Shared state for parameter inference implementations.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_INFERENCESTATE_H
#define KGEN_MOJOPARSER_INFERENCESTATE_H

#include "DeferredTypingContext.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/MojoParser/Constraints.h"
#include "Mojo/MojoParser/MojoDiags.h"
#include "ParserEvaluationContext.h"

#include <cstddef>

namespace M::KGEN {
class GeneratorType;
} // namespace M::KGEN

namespace M::KGEN::LIT {

class ASTDecl;
class CallParamInf;
class ParamInf;
class ParamMatcher;
class SharedState;
class StructDeclOp;

//===----------------------------------------------------------------------===//
// VerifiedParamBindings
//===----------------------------------------------------------------------===//

/// A set of parameter bindings that have been verified against a particular
/// Pog list signature by parameter inference. The verified-against-signature
/// invariant is what differentiates this from a raw `ParameterExprArrayAttr`,
/// and is what makes the specialization helpers below well-defined.
///
/// A default-constructed (or null) value represents failed inference; users
/// should check the truthiness of the value before invoking any specialization
/// method.
class VerifiedParamBindings {
public:
  VerifiedParamBindings() = default;

  /// Truthiness reflects whether inference succeeded (i.e. the underlying
  /// `ParameterExprArrayAttr` is non-null). Note that a successful inference
  /// can still produce zero bindings (for non-parametric callables); use
  /// `empty()` / `size()` to distinguish "no bindings to apply".
  explicit operator bool() const { return bool(bindings); }
  bool empty() const { return getValues().empty(); }
  size_t size() const { return getValues().size(); }

  /// Raw access for legacy interop with APIs that take a
  /// `ParameterExprArrayAttr` or `ArrayRef<TypedAttr>` (e.g.
  /// `BindParamsAttr::get`, `getCallee`, etc.).
  ParameterExprArrayAttr asAttr() const { return bindings; }
  ArrayRef<TypedAttr> getValues() const {
    return bindings ? bindings.getValue() : ArrayRef<TypedAttr>{};
  }
  /// Mask for body constraints discharged by these bindings.
  const llvm::BitVector &getDischargedBodyConstraints() const {
    return dischargedBodyConstraints;
  }

  /// Specialize a struct declaration with the verified bindings.
  TypedAttr specializeStructType(StructDeclOp structOp) const;

  /// Specialize a generator value (handles `StructMetaType` specially as the
  /// previous `InferenceState::getSpecializedGenerator` did).
  TypedAttr specializeGenerator(TypedAttr generator) const;

  /// Specialize a generator type via the type-level
  /// `GeneratorType::getSpecializedGenerator` API.
  Type specializeGeneratorType(GeneratorType genType) const;

private:
  VerifiedParamBindings(ParameterExprArrayAttr bindings,
                        llvm::BitVector dischargedBodyConstraints,
                        ParameterEvaluationContext *evaluationContext)
      : bindings(bindings),
        dischargedBodyConstraints(std::move(dischargedBodyConstraints)),
        evaluationContext(evaluationContext) {}

  ParameterExprArrayAttr bindings;
  llvm::BitVector dischargedBodyConstraints;
  ParameterEvaluationContext *evaluationContext = nullptr;

  friend class CallParamInf;
  friend class ParamInf;
};

/// Holds an optional in-flight diagnostic that can be discarded (e.g. for
/// try-style inference). InferenceState uses this as its diagnostic sink.
class OptionalDiag {
public:
  OptionalDiag(SharedState &shared, SMLoc defaultLoc, bool discardError);

  // Allow moving
  OptionalDiag(OptionalDiag &&other) = default;
  OptionalDiag &operator=(OptionalDiag &&other) = default;

  // Don't allow copying
  OptionalDiag(const OptionalDiag &) = delete;
  OptionalDiag &operator=(const OptionalDiag &) = delete;

  ~OptionalDiag() {
    if (discardError && diag.has_value())
      diag->abandon();
  }

  void release() { discardError = false; }

  /// Whether the holder of this diagnostic intends to silently discard any
  /// emitted error (e.g. overload filtering paths set this to suppress
  /// noise). Code that *opts to* emit additional diagnostics outside the
  /// `getDiag` channel should consult this so the noise is not bypassed.
  bool isDiscarding() const { return discardError; }

  /// Callback that returns a MojoInflightDiag for the given location (or
  /// default). Invoking the callback records that an error was emitted.
  llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc>)> getDiag();

  bool hasErrorEmitted() const { return diag.has_value(); }

  MojoInflightDiag &attachNote(SMLoc loc) & {
    diag->attachNote(loc);
    return *diag;
  }

  MojoInflightDiag &attachNote(ASTDecl &decl) & {
    diag->attachNote(decl);
    return *diag;
  }

  /// Take the emitted diagnostic if any.
  std::optional<MojoInflightDiag> takeMojoDiag() { return std::move(diag); }

private:
  bool discardError;
  std::optional<MojoInflightDiag> diag;
  std::function<MojoInflightDiag &(std::optional<SMLoc>)> getDiagClosure;
};

class InferenceState {
public:
  InferenceState(ASTDecl &declScope, ArrayRef<Type> declaredParamTypes,
                 PogListAttr declaredParamPogs, SMLoc defaultLoc,
                 bool discardError,
                 DeferredTypingContext *deferredTypingContext = nullptr,
                 ArrayRef<ConstraintAttr> additionalAssumptions = {});
  virtual ~InferenceState() = default;

  ASTDecl &getDeclScope() const { return declScope; }
  SharedState &getShared() const { return shared; }

  /// Return a diagnostic emitter for the given location (or default).
  MojoInflightDiag &getMojoDiag(std::optional<SMLoc> loc) {
    return diag.getDiag()(loc);
  }

  void setInferredValue(size_t paramIdx, TypedAttr paramVal,
                        bool isDefaulted = false);
  virtual bool isExplicitlyUnbound(size_t paramIdx) const = 0;

  /// Verify the body constraints declared on the signature against the
  /// current parameter bindings.
  LogicalResult
  checkBodyConstraints(ArrayRef<ConstraintAttr> extraConstraints = {});

  /// Unprovable constraints from body constraints. These are considered "soft"
  /// errors that won't fail inference by default. Users of InferenceState can
  /// choose to surface these errors at the appropriate time.
  /// Filled in by `checkBodyConstraints`.
  SmallVector<ConstraintAttr> bodyUnprovableConstraints;
  /// Mask of body constraints discharged by inference.
  /// Filled in by `checkBodyConstraints`.
  llvm::BitVector dischargedBodyConstraints;

  /// When non-null, body-constraint inconclusiveness at single-candidate
  /// binding sites is silently accepted and the unprovable body constraints
  /// are appended here for the caller to discharge later. This is a routing
  /// policy, similar to `discardError`, set once for the lifetime of an
  /// inference operation.
  DeferredTypingContext *deferredTypingContext;

  /// Facts that hold at the site driving this inference but are not reachable
  /// from `declScope`'s lexical assumptions. This is set once for the lifetime
  /// of an inference operation. The storage is owned by the caller.
  ArrayRef<ConstraintAttr> additionalAssumptions;

  // Debug util.
  void dump() const;

  /// The number of implicit conversions required.  Normal implicit
  /// conversions count as 2 each, nonmaterializable value conversions count
  /// as 1.
  size_t numImplicitConversions = 0;

protected:
  friend class ParamMatcher;

  ASTDecl &declScope;
  SharedState &shared;
  ParameterEvaluator evaluator;
  ParamIndexRefAttrFinder paramFinder;
  ArrayRef<Type> declaredParamTypes;
  PogListAttr declaredParamPogs;

public:
  OptionalDiag diag;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_INFERENCESTATE_H
