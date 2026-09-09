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

#ifndef KGEN_MOJOPARSER_PARAMETERINFERENCE_H
#define KGEN_MOJOPARSER_PARAMETERINFERENCE_H

#include "InferenceState.h"
#include "OverloadSet.h"

namespace M::KGEN::LIT {

//===----------------------------------------------------------------------===//
// ParamInf
//===----------------------------------------------------------------------===//

/// This class provides the implementation details that help to infer
/// information about the specified parameter.
class ParamInf : public InferenceState {
public:
  /// These are the bindings originally provided to the callable.
  const ParamBindings &paramBindings;

  /// If we're inferring the parameters for a declaration like a def or struct,
  /// maintain a pointer to it so we can emit better diagnostics.  This will be
  /// null when binding a parametric value, like a parametric alias.
  ASTDecl *const declIfKnown;

  ParamInf(const ParamBindings &paramBinding, ArrayRef<Type> declaredParamTypes,
           PogListAttr declaredParamPogs, bool allowImplicitConversions,
           ASTDecl *declIfDirect, bool discardError,
           DeferredTypingContext *deferredTypingContext = nullptr,
           ArrayRef<ConstraintAttr> additionalAssumptions = {});

  // Infer the parameter binding for a struct given a (potentially incomplete)
  // parameter binding. On success, returns the verified bindings (which can be
  // used to specialize a struct type or generator); on failure, returns a null
  // `VerifiedParamBindings`.
  //
  // Body-constraint inconclusiveness is routed according to the
  // `deferredTypingContext` passed at construction time: when non-null,
  // unprovable body constraints are appended to the context and the inference
  // succeeds; otherwise they are raised as a hard error. Per-parameter
  // constraint inconclusiveness is always a hard error.
  VerifiedParamBindings inferForStruct();

  /// After inferring parameter values, this allows access to the results.
  TypedAttr getInferredValue(size_t idx) const {
    return evaluator.getIndexBindings()[idx];
  }

  ParameterExprArrayAttr getInferredValues() const {
    return ParameterExprArrayAttr::get(paramBindings.shared.getContext(),
                                       evaluator.getIndexBindings());
  }

  const ParameterEvaluator &getEvaluator() const { return evaluator; }

  /// Pre-seed an inferred parameter value by index.
  void setInitialInferredValue(size_t paramIdx, TypedAttr paramVal) {
    setInferredValue(paramIdx, paramVal);
  }

private:
  /// Infer all of the parameters we can from 'givenBindings'.
  ///
  /// The 'partial' field specifies this is
  /// performing a partial binding - e.g. because this is not a full type
  /// binding, or because more params can be inferred from arguments to the
  /// call.
  ///
  /// On failure, this will emit a diagnostic through the 'getDiag' callback.
  LogicalResult inferFromParamList();

  // Infer any missing parameter from defaulted value (this is supposed to be
  // invoked after both parameter list and argument list has been scanned).
  LogicalResult inferFromDefaults();

  // Finalize the inference by making any remaining uninferred parameter to
  // UnboundAttr.
  LogicalResult finalizeWithUnbound();

  /// Infer still-unbound parameters from equality (`where a == b`) body
  /// constraints.
  LogicalResult inferFromBodyConstraints();

  /// Convenience getters for fields from paramBindings.
  ASTDecl &getDeclScope() const { return paramBindings.declScope; }
  SharedState &getShared() const { return paramBindings.shared; }
  const CallOperands &getGivenBindings() const {
    return paramBindings.getParameters();
  }

  FailureOr<SmartVariant<CValue, ASTType>>
  inferCValue(ASTExprAnd<AnyValue> operand, size_t argIdx, PogListAttr argPogs,
              CallSyntax syntax, ASTType expectedType,
              OperandsNeedingOriginsList *forwardedNeedingOrigins = nullptr);

  /// Core type matching logic for parameter inference, this matches expected
  /// type against the RValue type of the operand (which means this does not
  /// handle calling convention).
  LogicalResult inferFromRVType(
      ASTExprAnd<AnyValue> operand, size_t argIdx, ASTType expectedType,
      PogListAttr argPogs, CallSyntax syntax,
      OperandsNeedingOriginsList *forwardedNeedingOrigins = nullptr);

  /// Infer and emit a single value for a parameter binding. This returns
  /// failure if it emits a diagnostic, otherwise is returns a parameter value
  /// if resolved, or null if deferred.
  FailureOr<TypedAttr> inferAndEmitOneParam(ASTExprAnd<AnyValue> binding,
                                            ASTType expectedType,
                                            size_t paramIdx);

  bool isExplicitlyUnbound(size_t paramIdx) const override {
    return explicitlyUnboundParams[paramIdx];
  }

  // This says that a parameter with an unresolved dependent type was seen
  // during initial parameter binding application, so resolution of it was
  // deferred.
  //
  // This is to handle cases like:
  //
  // def foo[rank : Int, coord : IndexList[rank, Int]](i : SomeThing[rank]):
  //    pass
  //
  // def foo_user():
  //    var i = SomeThing[2]
  //    foo[coord = Tuple(1, 2)](i)
  //
  // When binding `coord` with `coord: Tuple[Int, Int]`, we do not know that
  // `rank=2` nor `Tuple[Int, Int]` is convertible to `IndexList[2, Int]`, we
  // have to postpone the binding till `rank` is resolved.
  //
  // Do we need to support this? I don't think this is too crazy to require user
  // to type `foo[rank = 2, coord = (1, 2)]` here?
  llvm::BitVector deferredGivenParams;

  /// This describes the parameters that are explicitly unbound (only applicable
  /// for struct binding. For call binding, every parameter might have a
  /// concrete real value).
  llvm::BitVector explicitlyUnboundParams;

  /// True if implicit conversions in argument lists are permitted.
  const bool allowImplicitConversions;
  bool isInferForStruct = false;

  friend class ParamMatcher;
  friend class ParamBindings;
  friend class CallParamInf;
};

//===----------------------------------------------------------------------===//
// CallParamInf
//===----------------------------------------------------------------------===//

/// Parameter inference specialized for matching a call's operands against a
/// callee signature.
class CallParamInf : public ParamInf {
public:
  CallParamInf(const ParamBindings &paramBinding,
               ArrayRef<Type> declaredParamTypes, PogListAttr declaredParamPogs,
               bool allowImplicitConversions, ASTDecl *declIfDirect,
               bool discardError, FnTypeGeneratorType calleeSignature,
               const CallOperands &callOperands,
               const CallOperands::PogAssignment &pogAssignment,
               OperandsNeedingOriginsList &operandsNeedingOrigins,
               ArrayRef<ConstraintAttr> additionalAssumptions = {});

  /// Given an incomplete parameter binding set and the arguments for a call to
  /// the specified signature, try to infer the verified parameter bindings. On
  /// failure, returns a null `VerifiedParamBindings` and records the reason in
  /// the diagnostic/unprovable-constraint side channels.
  VerifiedParamBindings inferForCall();

  /// For each mismatch in "preferred" argument convention, penalize the
  /// overload. This is to resolve ambiguities that can arise from synthesized
  /// thunks for converting calling conventions.
  size_t numMismatchedConventions = 0;

  /// Whether the candidate has a (non-empty) variadic argument.
  bool passesVarArgArgument = false;

private:
  FnTypeGeneratorType calleeSignature;
  const CallOperands &callOperands;
  const CallOperands::PogAssignment &pogAssignment;
  OperandsNeedingOriginsList &operandsNeedingOrigins;

  LogicalResult inferOptionalLiteralSize();

  /// Given an incomplete parameter binding set, try to infer parameters on
  /// Self of a method from the first argument.
  LogicalResult inferCTADParams();

  LogicalResult inferSelfFromInitResult();

  LogicalResult inferResultSlot(RefType expectedRef, size_t argIdx,
                                const ExprDest &dest);

  /// Infer parameters from an operand being passed into this function. This is
  /// only called on the top level function operands being matched up, not
  /// anything in recursive functiontype positions.
  LogicalResult inferOneOperand(ASTExprAnd<AnyValue> operand, size_t operandIdx,
                                size_t argIdx, ASTType expectedType,
                                ArgConvention expectedConvention);
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARAMETERINFERENCE_H
