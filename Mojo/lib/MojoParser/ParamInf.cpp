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

#include "ParamInf.h"
#include "ClosureEmitter.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParamBindings.h"
#include "ParamMatcher.h"
#include "Traits.h"

#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/IRValues.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/lib/MojoParser/Traits.h"

#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// File-local utils
//===----------------------------------------------------------------------===//

static Type inferInitializerType(ASTDecl &declScope, InitializerUValue &init,
                                 ASTExprAnd<AnyValue> operand,
                                 ASTType defaultType) {
  IREmitter emitter(declScope, EC_CallArgValue);
  if (!defaultType)
    return {};
  ASTType inferredType =
      defaultType.getWithUnknownParametersReplaced(declScope.getShared());

  CallOperands operands =
      init.getOperandsForInferredType(inferredType, EC_CallArgValue, emitter);

  // We expect the initializer to return the constructed type.
  // Infer the parameters of this overload candidate against the computed
  // result type of the initializer.
  FailureOr<CalleeResult> initFn =
      OverloadSet::canConstructType(inferredType, operands, declScope);
  if (failed(initFn) || !initFn->isYes())
    return {};
  return FnOrFnLiteralTypeGeneratorType::get(initFn->getYes().getType())
      .getUserResultType();
}

static void assertLegalForwardedOperandOrigins(
    [[maybe_unused]] const OperandsNeedingOriginsList &needingOrigins) {
  // Since we must have picked a single argument implicit conversion, make sure
  // nothing surprising happened...
  assert(needingOrigins.size() == 1 &&
         "implicit conversion must spill exactly one operand");
  // must be an __init__(ref ..., out self).
  assert(needingOrigins.front().argIdx == 0 &&
         "the spilled operand must be the ctor's first argument");
  assert(needingOrigins.front().operandIdx !=
             OperandNeedingOrigin::kExprDestOperandIdx &&
         "the spilled operand must be a real operand, not the ExprDest");
}

/// Try to infer the type of an initializer list/dict/set/slice literal by
/// first binding it to `preferred` and, on failure, to the literal's default
/// type (e.g. `List[Int]` for a list literal).  Returns a null `Type` if
/// neither binding succeeds.
static Type tryInferInitializerType(ASTDecl &declScope, InitializerUValue &init,
                                    ASTExprAnd<AnyValue> operand,
                                    ASTType preferred) {
  if (Type result = inferInitializerType(declScope, init, operand, preferred))
    return result;
  return inferInitializerType(declScope, init, operand,
                              init.getDefaultType(declScope.getShared()));
}

//===----------------------------------------------------------------------===//
// ParameterInference
//===----------------------------------------------------------------------===//

ParamInf::ParamInf(const ParamBindings &paramBinding,
                   ArrayRef<Type> declaredParamTypes,
                   PogListAttr declaredParamPogs, bool allowImplicitConversions,
                   ASTDecl *declIfDirect, bool discardError,
                   DeferredTypingContext *deferredTypingContext,
                   ArrayRef<ConstraintAttr> additionalAssumptions)
    : InferenceState(paramBinding.declScope, declaredParamTypes,
                     declaredParamPogs, paramBinding.getExprLoc(), discardError,
                     deferredTypingContext, additionalAssumptions),
      paramBindings(paramBinding), declIfKnown(declIfDirect),
      deferredGivenParams(declaredParamTypes.size(), false),
      explicitlyUnboundParams(declaredParamTypes.size(), false),
      allowImplicitConversions(allowImplicitConversions) {}

// TODO: Reconsolidate this.
namespace M::KGEN::LIT {
void printUValueTypeInfo(const AnyValue &value, MojoInflightDiag &diag);
void emitWrongTypeDiag(MojoInflightDiag &diag, ASTExprAnd<AnyValue> operand,
                       ASTType expectedType, size_t argIdx,
                       PogListAttr argListAttr, CallSyntax syntax,
                       SharedState &shared, ASTDecl &declScope);
} // namespace M::KGEN::LIT

/// Attempt to resolve the specified operand to a CValue using the provided
/// type, checking whether any UValue's are compatible with the type and
/// inferring any parameters from it.  This emits a diagnostic and returns null
/// on failure.  On success, this makes an attempt to return a CValue, but won't
/// do so if that would require generating dynamic logic (e.g. creating an
/// instance of a value due to an initializer list).  In that case it returns
/// the inferred type of the result.
FailureOr<SmartVariant<CValue, ASTType>>
ParamInf::inferCValue(ASTExprAnd<AnyValue> operand, size_t argIdx,
                      PogListAttr argPogs, CallSyntax syntax,
                      ASTType expectedType,
                      OperandsNeedingOriginsList *forwardedNeedingOrigins) {
  // If this is already a CValue then we're done.
  if (auto cv = operand.ir.getIfCValue())
    return SmartVariant<CValue, ASTType>(cv);

  auto emitWrongTypeDiag = [&](ASTType expectedType) -> MojoInflightDiag & {
    auto &diag = getMojoDiag(operand.expr->getLoc());
    ::emitWrongTypeDiag(diag, operand, evaluator.getReboundType(expectedType),
                        argIdx, argPogs, syntax, getShared(), getDeclScope());
    return diag;
  };

  // Check to see if the expected type has an initializer with the
  // specified operands.  Remove any parameters from the expected type
  // since those are what we're inferring from the arguments.  The result
  // 'actualType' will have those newly inferred parameters.
  if (auto initValue = operand.ir.getIfInitializer()) {
    // Try binding the literal to `expectedType` (e.g. `List[$0]` becomes
    // `List[?]` so the unbound parameter is inferred), then fall back to the
    // literal's default type.
    ASTType initType = tryInferInitializerType(getDeclScope(), *initValue,
                                               operand, expectedType);

    // If there were declaration errors, assume success to not raise
    // spurious errors due to not resolving to those erroneous
    // declarations.
    if (!initType) { // TODO: Could improve this error to talk about inits.
      emitWrongTypeDiag(expectedType);
      return failure();
    }

    // If we're in a parameter binding expression, we can just emit the value as
    // a PValue and return it.  This is more powerful than the logic below,
    // because it allows implicit conversions, e.g. when we default a list
    // literal like [1, 2] to List[Int], it supports implicit conversion to
    // Span[Int, _].  The logic below does not support this.
    if (syntax == CallSyntax::kParamBindings) {
      IREmitter emitter(getDeclScope(), ExprContext::EC_ParameterList);
      auto value = emitter.emitPValue(operand, EC_ParameterList, initType);
      if (!value)
        return failure();
      return SmartVariant<CValue, ASTType>(value);
    }

    ParamMatcher matcher(operand.expr, *this, allowImplicitConversions);

    // If we found one, we resolve our value to the inferred type.
    if (succeeded(matcher.matchTypes(initType, expectedType)))
      return SmartVariant<CValue, ASTType>(initType);

    // Check whether an implicit conversion
    if (allowImplicitConversions) {
      CallOperands callOperands(CallSyntax::kImplicitConvert, operand.expr,
                                EC_OverloadResolution, {operand});
      FailureOr<CalleeResult> pValue = OverloadSet::canConstructType(
          expectedType.getWithUnknownParametersReplaced(getShared()),
          callOperands, getDeclScope(), forwardedNeedingOrigins);

      // If we found one, we succeed if the returned type is compatible with the
      // expected type.  Infer the parameters of this overload candidate against
      // the computed result type of the initializer.
      if (succeeded(pValue) && pValue->isYes()) {
        auto sig =
            FnOrFnLiteralTypeGeneratorType::get(pValue->getYes().getType());
        if (succeeded(
                matcher.matchTypes(sig.getUserResultType(), expectedType))) {
          ++numImplicitConversions;
          return SmartVariant<CValue, ASTType>(
              ASTType(sig.getUserResultType()));
        }
      }
    }
    // TODO: Could improve this to talk about initializers.
    auto &diag = emitWrongTypeDiag(expectedType);
    matcher.failureReason->addExplanation(diag);
    return failure();
  }

  if (auto inferredAttr = operand.ir.getIfInferredBaseAttrRef()) {
    // Resolve `.member` against the expected parameter type, e.g.
    // `takes_dtype(.float32)` with `dtype: DType` becomes `DType.float32`.
    IREmitter emitter(getDeclScope(), ExprContext::EC_CallArgValue);
    ExprDest dest(expectedType, ExprContext::EC_CallArgValue);
    if (auto cValue = inferredAttr.emitAsCValue(emitter, dest))
      return SmartVariant<CValue, ASTType>(cValue);
    // emitAsCValue diagnoses into the shared engine; also record on the
    // ParamInf diag so overload fitness failure reporting stays consistent.
    getMojoDiag(operand.expr->getLoc())
        << "cannot resolve inferred attribute reference";
    return failure();
  }

  auto orValue = operand.ir.getIfOverloadSet();
  assert(orValue && "Unknown UValue!");

  // If we have a reference to an overloaded method like foo(a.method),
  // then we can't resolve it.
  // TODO(partial application => closures): Given we just resolved argVal,
  // we could form the "a.method" expression with a closure.
  if (orValue->baseValue) { // Cannot merge base value.
    emitWrongTypeDiag(expectedType);
    return failure(); // TODO: Improve this.
  }

  // If the overload set has a single entry, just get it.
  if (auto pv = orValue->getIfPValue())
    return SmartVariant<CValue, ASTType>(CValue(pv));

  // If the expected type is concrete, then we can filter the overload set down
  // to a single entry and emit errors if not.
  if (!paramFinder.hasReferences(expectedType)) {
    auto emitError = [&](SMLoc loc) -> MojoInflightDiag & {
      return getMojoDiag(loc);
    };

    auto argValResult =
        orValue->filterOverloadSetForValueType(expectedType, emitError);
    if (!argValResult.isYes())
      return failure();
    return SmartVariant<CValue, ASTType>(CValue(argValResult.getYes().callee));
  }

  // FIXME: This emits an error unconditionally (not to getDiags) on failure.
  if (PValue result = orValue->filterOverloadSetForParamBindings())
    return SmartVariant<CValue, ASTType>(CValue(result));

  // Otherwise, we don't have a contextual error.
  emitWrongTypeDiag(expectedType);
  return failure(); // TODO: Improve this.
}

/// Core type matching logic for parameter inference, handling the expected
/// type without convention-specific processing. This function is called after
/// the expected type has been adjusted for calling conventions.
///
/// NOTE: This function performs parameter inference and error reporting,
/// while 'OverloadFitness::scoreOperandFitness' computes fitness metrics
/// after inference is complete. They serve different phases of overload
/// resolution and should remain separate.
LogicalResult
ParamInf::inferFromRVType(ASTExprAnd<AnyValue> operand, size_t argIdx,
                          ASTType expectedType, PogListAttr argPogs,
                          CallSyntax syntax,
                          OperandsNeedingOriginsList *forwardedNeedingOrigins) {
  // Make sure the diagnostic machinery knows about our getDeclScope() so
  // parameter names get emitted correctly.
  DeclResolver::DiagnosticDeclContextChanger x(declIfKnown);

  auto emitWrongTypeDiag = [&](ASTType expectedType) -> MojoInflightDiag & {
    auto &diag = getMojoDiag(operand.expr->getLoc());
    ::emitWrongTypeDiag(diag, operand, evaluator.getReboundType(expectedType),
                        argIdx, argPogs, syntax, getShared(), getDeclScope());
    return diag;
  };

  expectedType = evaluator.getReboundType(expectedType);

  // Okay, we got a normal value argument convention and stripped off any
  // ArgConvention-related !lit.ref from the expected type.  See if we can
  // resolve the argument to a CValue.
  FailureOr<SmartVariant<CValue, ASTType>> argValOr = inferCValue(
      operand, argIdx, argPogs, syntax, expectedType, forwardedNeedingOrigins);
  if (failed(argValOr))
    return failure();
  CValue argVal = dyn_cast<CValue>(*argValOr);
  if (!argVal) // Already checked the type is ok.
    return success();

  // If the argument types exactly match, then they are good.
  ASTType argType = argVal.getRValueType();
  if (argType.isEqualCanon(expectedType))
    return success();

  // We have a non-parametric expected type, and a wildcard type, we can
  // match any operand.
  if (sugarIsa<NameLookupArgWildcardType>(argType) &&
      !paramFinder.hasReferences(expectedType))
    return success();

  // TODO: Optionally compute fitness metrics (# implicit conversions,
  // convention mismatches) during inference, so they don't need to be
  // recomputed by scoreOperandFitness() afterward.
  ParamMatcher matcher(operand.expr, *this, allowImplicitConversions);

  ParamMatcher::FailableScope simpleEqualityFailableScope(matcher);
  if (succeeded(matcher.matchTypes(argType, expectedType)))
    return success(); // Types were equal after matching.

  // Save the failure code and the bindings that were inferred so we can
  // restore them if the other attempts fail.
  auto savedFailureInfo = simpleEqualityFailableScope.saveState();
  simpleEqualityFailableScope.revert();

  // Handle values of nonmaterializable types.  These freely convert to their
  // nonmaterializable target type: even when implicit conversions are disabled.
  // We can accept this argument if that converted type is compatible with
  // our expected type.
  if (auto nonmaterializableTarget =
          argType.getNonmaterializableTarget(getShared())) {
    ParamMatcher::FailableScope failableScope(matcher);
    // Infer the parameters of this overload candidate against the computed
    // result type of the initializer.

    if (succeeded(matcher.matchTypes(nonmaterializableTarget, expectedType))) {
      // Implicit conversion for nonmaterializable types to their target
      // type is allowed even if !allowImplicitConversions and count as half
      // as much of a mismatch as a normal implicit conversion.  This enables
      // exact matches to be more specific, and literals to be more compatible
      // than an actual conversion.
      ++numImplicitConversions;
      return success();
    }

    // Roll back any error and inferred bindings.
    failableScope.revert();
  }

  // If implicit conversions are enabled and the target type is known, then
  // we can check to see if any of the constructors for the result type can
  // work.  If disabled, then we have a failure.
  if (!allowImplicitConversions) {
    // Restore the information from the original failure so we have a simple
    // diagnostic.
    ParamMatcher::FailableScope::restore(savedFailureInfo, matcher);
    auto &diag = emitWrongTypeDiag(expectedType);
    matcher.failureReason->addExplanation(diag);
    return failure();
  }

  // If we had one, this bumps our # implicit conversions.
  numImplicitConversions += 2;

  // If the expected type has been fully resolved, check it for implicit
  // conversions using the normal type machinery.  This will handle things like
  // function pointer conversions that the code below doesn't.
  if (!paramFinder.hasReferences(expectedType)) {
    auto conversion = IREmitter::classifyImplicitConversionWithDetails(
        {argVal, operand.expr}, expectedType, getDeclScope(),
        /*additionalAssumptions=*/{}, /*deferralCtx=*/nullptr);
    if (conversion.isYes())
      return success();

    // Restore the original failure so the diagnostic stays simple.
    ParamMatcher::FailableScope::restore(savedFailureInfo, matcher);
    auto &diag = emitWrongTypeDiag(expectedType);
    if (conversion.isNo())
      std::move(conversion).getNo().addExplanation(diag);
    else
      attachConstraintNotes(diag, conversion.getUnknown(), "unproven");
    matcher.failureReason->addExplanation(diag);
    return failure();
  }

  /// When checking if an implicit conversion is possible, apply the bindings
  /// inferred so far (plus a distinct new attribute relating back to the
  /// original decls for ones that are missing) to the signature with
  /// getSpecializedSignature so we benefit from the already-fixed substitutions
  /// being applied to the input types.  This can make them more concrete and
  /// help with inferring dependent types based on already-bound parameters.  If
  /// we inferred a value for the parameter from previous arguments, substitute
  /// it into the expected types of subsequent arguments.  This allows us to
  /// handle dependent argument types like:
  ///     def foo[dt: DType](p: UnsafePointer[Scalar[dt]], v:
  ///     Scalar[p.type.type]):
  /// where the type of 'v' depends on 'dt' being inferred.

  // Determine if we can construct the requested type given the existing value
  // we have.  If so, get the type inferred signature of the init method that
  // would make it work.
  if (sugarIsa<StructType>(expectedType)) {
    // The expected type may be parameterized, and that type may both have
    // parameters that we are trying to infer as well as parameters that are
    // already known.  For example, if expectedType is known to be
    // 'SIMD[uint8, 1]', then we can infer which constructor to use when the
    // input is an IntLiteral.
    //
    // On the other hand, if expectedType is something like 'SIMD[?, 1]' and the
    // argument is an Int8, then we need the implicit conversion to infer the
    // base element.  Our solution to this is to rip and replace parameters that
    // contain unbound parameters, replacing them with UnboundAttr so inference
    // can find them.
    auto getImplicitConvertTarget =
        [&](CValue argVal, OperandsNeedingOriginsList *needingOrigins) -> Type {
      CallOperands ctorOperands(CallSyntax::kImplicitConvert, operand.expr,
                                EC_TypeParamValue, {{argVal, operand.expr}});
      auto nonParamType =
          expectedType.getWithUnknownParametersReplaced(getShared());
      FailureOr<CalleeResult> pValue = OverloadSet::canConstructType(
          nonParamType, ctorOperands, getDeclScope(), needingOrigins);
      if (failed(pValue) || !pValue->isYes())
        return {};

      // If we found one, we succeed if the returned type is compatible with the
      // expected type.  Infer the parameters of this overload candidate against
      // the computed result type of the initializer.
      auto initSig =
          FnOrFnLiteralTypeGeneratorType::get(pValue->getYes().getType());
      return initSig.getUserResultType();
    };

    Type targetType = getImplicitConvertTarget(argVal, forwardedNeedingOrigins);
    if (syntax == CallSyntax::kParamBindings && targetType &&
        forwardedNeedingOrigins && !forwardedNeedingOrigins->empty()) {
      // The implicit ctor picked during `inferFromRVType` must only take one
      // argument, so the spill can only be for its first argument.
      assertLegalForwardedOperandOrigins(*forwardedNeedingOrigins);
      OperandNeedingOrigin forwarded = forwardedNeedingOrigins->front();
      // Do the inference with a comptime.origin right away, we don't have to
      // wait till emission time for origin refinement PValues.
      CValue newVal;
      if (forwarded.getArgConvention() != ArgConvention::OwnedMem)
        newVal = PMBValue::getFromPValue(argVal.getIfPValue());
      else
        newVal = PMRValue::getFromPValue(argVal.getIfPValue());
      targetType = getImplicitConvertTarget(newVal, forwardedNeedingOrigins);
      // Must succeed because the target type is already checked in the previous
      // call to `getImplicitConvertTarget`.
      assert(targetType);
    }

    if (targetType) {
      // If we found one, we succeed if the returned type is compatible with
      // the expected type.  Infer the parameters of this overload candidate
      // against the computed result type of the initializer.
      ParamMatcher::FailableScope failableScope(matcher);
      if (succeeded(matcher.matchTypes(targetType, expectedType)))
        return success();
      failableScope.revert();
    }
  }

  // Otherwise, none of that worked. We aren't sure what to do here - it could
  // be any of these things, so we need to emit an error.  If out failure is
  // due to an uninferred parameter, and if that parameter had a default, then
  // we can bind it.
  if (savedFailureInfo.failureReason.getIfDependentOnUnresolved()) {
    // If we're in the parameter binding list *for a call* then we can
    // re-evaluate this binding after the arguments of the call are resolved.
    //
    // For struct binding, we enforce strict left-to-right order.
    //
    // FIXME(MOCO-3300): for call binding, we need to resolve deferred parameter
    // binding before default as well (IT IS NOT THE CASE AT THE MOMENT).
    if (syntax == CallSyntax::kParamBindings && !isInferForStruct) {
      deferredGivenParams.set(argIdx);
      return success();
    }

    // At this point, if we still have an unresolvable dependent type, give it
    // one last shot and try to pull default parameter value
    //
    // def store[
    //     dtype: DType
    //     width: Int = 1,
    // ](
    //     self: UnsafePointer[Scalar[dtype], ...],
    //     val: SIMD[dtype, width],
    // )
    //
    // # here Int(8) need to be implicitly converted to SIMD[dtype, 1],
    // store(ptr, Int(8))
    //
    // Otherwise, check to see if this is due to an uninferred param with a
    // default value.  If so, bind the default and try again.
    size_t paramIdx =
        savedFailureInfo.failureReason.getIfDependentOnUnresolved().value();
    if (auto value = declaredParamPogs.getDefault(paramIdx)) {
      assert(!evaluator.getIndexBindings()[paramIdx] &&
             "shouldn't have inferred this if we failed because of it");
      value = evaluator.getReboundAttribute(value);
      setInferredValue(paramIdx, value);
      if (failed(
              inferFromRVType(operand, argIdx, expectedType, argPogs, syntax)))
        return failure();

      TypedAttr newValue =
          evaluator.getReboundAttribute(evaluator.getIndexBindings()[paramIdx]);
      if (newValue != value)
        setInferredValue(paramIdx, value);
      return success();
    }
  }

  // Restore the information from the original failure so we have a simple
  // diagnostic.
  ParamMatcher::FailableScope::restore(savedFailureInfo, matcher);

  if (matcher.failureReason->isUnboundButInferrable()) {
    // Be more specific about this case.
    auto &diag = getMojoDiag(operand.expr->getLoc());
    diag << "failed to infer from type " << argType;
    matcher.failureReason->addExplanation(diag);
    return failure();
  }

  auto &diag = emitWrongTypeDiag(expectedType);
  matcher.failureReason->addExplanation(diag);
  return failure();
}

/// Infer and emit a single value for a parameter binding. This returns
/// failure if it emits a diagnostic, otherwise is returns a parameter value
/// if resolved, or null if deferred.
FailureOr<TypedAttr>
ParamInf::inferAndEmitOneParam(ASTExprAnd<AnyValue> binding,
                               ASTType expectedType, size_t paramIdx) {
  // Forward any deferral context so that binding a value into a trait-bounded
  // slot whose conformance is only unprovable is deferred by the conversion
  // machinery rather than hard-erroring here.
  IREmitter emitter(getDeclScope(), EC_ParameterList, deferredTypingContext);

  // We don't typecheck the '_' magic parameter, we propagate it.
  //
  // NOTE: we have to return a `_` here to mark the parameter has been
  // explicitly unbound instead of `nullptr` (maybe unless we know this is not a
  // partial binding?). Consider the following cases
  //
  // struct T[a : Int = 1] : pass
  // comptime T1 = T[_]
  // comptime T2 = T[]
  //
  // if we return nullptr here, ParamInf can not distinguish between T1 and T2,
  // and in both cases, `a` will be bound with the default value.
  //
  // NOTE: in a non-partial binding context, `_` can be also used as a place
  // holder, in this case we don't infer it to `_`
  if (isa_and_nonnull<UnboundAttr>(binding.ir.getIfPValue().get())) {
    // `_` means different things when used in struct binding or call bindings,
    // for struct binding, it is a concrete unknown value; for call binding, it
    // means something to be inferred.
    // We can not simply return and bind a `_` value here either, because it
    // could be dependent by other parameters/default values, which need to be
    // handled properly.
    if (isInferForStruct)
      explicitlyUnboundParams.set(paramIdx);

    return TypedAttr(); // Deferred
  }

  // If we have a UValue or something else, convert it to a PValue.
  if (!binding.ir.getIfPValue()) {
    FailureOr<SmartVariant<CValue, ASTType>> cvOr =
        inferCValue(binding, paramIdx, declaredParamPogs,
                    CallSyntax::kParamBindings, expectedType);
    if (failed(cvOr))
      return failure();

    CValue argVal = dyn_cast<CValue>(*cvOr);
    // If we had an initializer list this will succeed but not actually create
    // the instance because the logic is shared with the dynamic argument
    // checking logic that can't create an instance.  We don't have that problem
    // so just do it.  We need to use the returned type as the expected type
    // because the expected type might be something like Span, and the inferred
    // type might be List (e.g. as the default type for a list literal).
    if (!argVal) {
      argVal =
          emitter.emitPValue(binding, EC_ParameterList, cast<ASTType>(*cvOr));
      assert(argVal && "This should always succeed; it was checked");
    }

    // Finally, check that this CValue is a PValue.
    binding.ir = argVal.getIfPValue();
    if (!binding.ir) {
      getMojoDiag(binding.expr->getLoc())
          << "cannot use a dynamic value in a parameter list"
          << binding.expr->getRange();
      return failure();
    }
  }

  // If the expected type has unresolved bindings, try to infer them from the
  // argument.  This is a non-trivial operation because we support inferring
  // from the value directly, but also inferring as a result of implicit
  // conversions.
  if (paramFinder.hasReferences(expectedType)) {
    OperandsNeedingOriginsList paramOrigins;
    if (failed(inferFromRVType(binding, paramIdx, expectedType,
                               declaredParamPogs, CallSyntax::kParamBindings,
                               &paramOrigins)))
      return failure();

    // inferFromRVType above might have refined the expected type.
    expectedType = evaluator.getReboundType(expectedType);
    if (!paramOrigins.empty()) {
      assertLegalForwardedOperandOrigins(paramOrigins);
      // If we take the implicit conversion path and need to turn the value into
      // a reference. The expected type must have be fully concretized in order
      // for conversion to be successful.
      assert(!paramFinder.hasReferences(expectedType));

      if (paramOrigins.front().getArgConvention() != ArgConvention::OwnedMem)
        binding.ir = PMBValue::getFromPValue(binding.ir.getIfPValue());
      else
        binding.ir = PMRValue::getFromPValue(binding.ir.getIfPValue());

      // Eagerly issue the conversion right away with a comptime.origin ref.
      return emitter
          .emitPValue({binding.ir, binding.expr}, EC_ParameterList,
                      expectedType)
          .get();
    }
  }

  if (paramFinder.hasReferences(expectedType)) {
    deferredGivenParams.set(paramIdx);
    return TypedAttr(); // Deferred.
  }

  TypedAttr bindingVal = binding.ir.getIfPValue();
  // Reject invalid *'s, varargs will have been already handled.
  if (sugarIsa<UnpackedAttr>(bindingVal)) {
    getMojoDiag(binding.expr->getLoc())
        << "invalid unpack in non-variadic parameter binding"
        << binding.expr->getRange();
    return failure();
  }

  auto tryEmitBindingConversion = [&](TypedAttr candidate) -> TypedAttr {
    // Use the more refined type value to check convertibility.
    candidate = ParamMatcher::preAlignParam(candidate);

    // Check the type matches what is expected, and perform an implicit
    // conversion if needed.
    if (expectedType.isEqualCanon(candidate.getType()))
      // Align sugar if necessary.
      return ParamOperatorAttr::getRebind(candidate, expectedType);

    if (!IREmitter::canImplicitlyConvertToType(
            {candidate, binding.expr}, expectedType, emitter.getDeclScope(),
            /*additionalAssumptions=*/{}, emitter.deferredTypingContext))
      return {};

    return emitter
        .emitPValue({candidate, binding.expr}, EC_ParameterList, expectedType)
        .get();
  };

  if (TypedAttr convertedBinding = tryEmitBindingConversion(bindingVal))
    return convertedBinding;

  // A type value may become valid for this parameter inside a refined scope,
  // e.g. `T: AnyType` used as a `GuardedTrait` parameter under
  // `comptime if conforms_to(T, GuardedTrait)`. Keep the shadow local to this
  // binding instead of changing the declaration of `T` for the whole scope.
  // If the refinement still doesn't satisfy `expectedType`, prefer the
  // refined value in the diagnostic so the user sees the type the compiler
  // actually considered (e.g. `AnyType_GuardedTrait downcast(T)` rather than
  // just `AnyType T`).
  TypedAttr diagBindingVal = bindingVal;
  if (LIT::isTypeExpr(bindingVal)) {
    TypedAttr refinedBindingVal =
        maybeRefineTypeValueWithAssumptions(bindingVal, emitter.getDeclScope());
    if (refinedBindingVal != bindingVal) {
      if (TypedAttr convertedBinding =
              tryEmitBindingConversion(refinedBindingVal))
        return convertedBinding;
      diagBindingVal = refinedBindingVal;
    }
  }

  // Otherwise, the parameter is simply the wrong type, emit an error about this
  // problem.
  DeclResolver::DiagnosticDeclContextChanger x(&(getDeclScope()));
  MojoInflightDiag &diag = getMojoDiag(binding.expr->getLoc());
  // Why only structs? Seems arbitrary, push higher?
  if (declIfKnown && declIfKnown->getUserNameIfOperation().has_value())
    diag << "'" << *declIfKnown->getUserNameIfOperation() << "' ";
  diag << "parameter "
       << ParamDeclRefAttr::get(declaredParamPogs.getName(paramIdx),
                                declaredParamTypes[paramIdx])
       << " has " << expectedType << " type, but value has type "
       << diagBindingVal.getType() << binding.expr->getRange();

  // Add some extra note for binding a thin function to a closure trait, which
  // would otherwise be confusing to many.
  if (ClosureEmitter::isClosureType(shared, expectedType) &&
      sugarIsa<FnLiteralTypeGeneratorType>(diagBindingVal.getType())) {
    llvm::SMRange smRange =
        shared.diags.convertToSMRange(binding.expr->getRange());
    StringRef src(smRange.Start.getPointer(),
                  smRange.End.getPointer() - smRange.Start.getPointer());
    diag.attachNote(binding.expr->getLoc())
        << "a thin function cannot bind to a closure trait; use 'type_of("
        << src << ")' to pass its type instead";
  }

  return failure();
}

/// Infer all of the parameters we can from 'givenBindings'.
///
/// The 'partial' field specifies this is
/// performing a partial binding - e.g. because this is not a full type
/// binding, or because more params can be inferred from arguments to the
/// call.
///
/// On failure, this will emit a diagnostic through the 'getDiag' callback.
LogicalResult ParamInf::inferFromParamList() {
  // Use the temporary operands list if we had to remove an ellipsis, otherwise
  // use the original operands list.
  const CallOperands &givenBindings = this->getGivenBindings();

  auto getStagedDiag = [&](SMLoc loc) -> MojoInflightDiag & {
    // TODO: getMojoDiag should always require an SMLoc.
    return getMojoDiag(loc);
  };

  // Do basic validation of the argument list using shared logic.
  CallOperands::PogAssignment pogAssignment;
  if (failed(givenBindings.assignToPogs(declaredParamPogs,
                                        /*isParameterList=*/true, pogAssignment,
                                        getStagedDiag)))
    return failure();

  // We may have pre-checked and out-of-order inferred parameters.  Avoid
  // stomping on them.
  auto applyBinding = [&](size_t idx,
                          FailureOr<TypedAttr> paramVal) -> LogicalResult {
    if (failed(paramVal))
      return failure(); // Already diagnosed.

    // Ignore this if the parameter value is deferred.
    if (!*paramVal)
      return success();

    auto existing = evaluator.getIndexBindings()[idx];
    if (!existing) {
      setInferredValue(idx, *paramVal);
      return success();
    }

    assert(isEqualCanon(existing, *paramVal) &&
           "inferred to different values but didn't notice");
    return success();
  };

  for (auto [idx, pog] : llvm::enumerate(declaredParamPogs.getPogs())) {
    // Note that 'signature' changes the type as we go, so don't use
    // llvm::enumerate on the argument type list!
    Type expectedType = evaluator.getReboundType(declaredParamTypes[idx]);

    switch (pogAssignment.operandIdxs[idx]) {
    default: {
      size_t operandIdx = pogAssignment.operandIdxs[idx];
      FailureOr<TypedAttr> paramVal = inferAndEmitOneParam(
          givenBindings.values[operandIdx], expectedType, idx);
      // Exit if an error was already emitted.
      if (failed(applyBinding(idx, paramVal)))
        return failure();
      continue;
    }
    case CallOperands::PogAssignment::kPA_Unspecified:
    case CallOperands::PogAssignment::kPA_Default:
      // Default values and unspecified values are handled separately in param
      // inference.  TODO: Pull default values into here.
      break;
    case CallOperands::PogAssignment::kPA_Variadic:
      if (!pog.isPosVarArg()) {
        getMojoDiag({}) << "Keyword variadics are not supported for parameters";
        return failure();
      }

      // If there are no parameter values, then leave the parameter uninferred
      // for now.  It could be inferred from an call-argument or be left
      // unbound.
      if (pogAssignment.posVariadicIdxs.empty())
        continue;

      // Unpacked variadics (`Tuple[*elts]` where elts is a variadic list) can
      // be passed directly as a whole variadic parameter.
      auto [varArgsEltType, expectedValueList] =
          ASTType(expectedType).getParameterListInfo();

      // TODO: What is UnpackedAttr about? Why isn't this part of the operand?
      if (pogAssignment.posVariadicIdxs.size() == 1) {
        auto &operand = givenBindings[pogAssignment.posVariadicIdxs[0]];
        if (auto unpacked = dyn_cast_or_null<UnpackedAttr>(
                operand.ir.getIfPValue().get())) {
          // FIXME: Make sure to only unpack *x in pos varargs and **x in kw
          // varargs.
          FailureOr<TypedAttr> paramVal = inferAndEmitOneParam(
              {unpacked.getValue(), operand.expr}, expectedType, idx);
          // Exit if an error was already emitted.
          if (failed(applyBinding(idx, paramVal)))
            return failure();
          continue;
        }
      }

      Type mergedTypeBound;
      if (paramFinder.hasReferences(varArgsEltType) &&
          sugarIsa<AnyTraitType>(ASTType(varArgsEltType).extractMetaType())) {
        if (pogAssignment.posVariadicIdxs.empty()) {
          // TODO: The correct thing is to infer to `NeverTrait` such that an
          // empty type list can be upcasted to any TypeList with a less refined
          // bound. Use `AnyType` here, better than nothing.
          mergedTypeBound = shared.lookupBuiltinTraitType(
              "AnyType", givenBindings.getExprLoc());
        }

        // To infer `TypeList[Trait : type_of(AnyType), *values: Trait]`
        // with `TypeList[Int, Float, Bool]`, merge the types list to a common
        // trait bound first to infer the trait instead of always using the
        // first element types.
        for (auto idx : pogAssignment.posVariadicIdxs) {
          TypedAttr typeValue = givenBindings[idx].ir.getIfPValue().get();
          // This is an invalid element, bail out and an error will be when
          // emitting the binding below.
          if (!typeValue || !LIT::isFirstLevelTypeExpr(typeValue)) {
            mergedTypeBound = nullptr;
            break;
          }

          if (!mergedTypeBound) {
            mergedTypeBound = typeValue.getType();
            continue;
          }

          mergedTypeBound = mergeTwoMetaTypeBounds(shared, mergedTypeBound,
                                                   typeValue.getType());
        }
      }

      // Otherwise, we infer the variadic to be the elements of the variadic
      // list being passed in.
      SmallVector<TypedAttr> elements;
      bool isDeferred = false;
      for (auto operandIdx : pogAssignment.posVariadicIdxs) {
        OperandValue operand = givenBindings[operandIdx];

        // Passing `_` to a variadic is not allowed. Users should pass `*_` to
        // unbind a variadic parameter.
        if (isa_and_nonnull<UnboundAttr>(operand.ir.getIfPValue().get())) {
          auto &diag = getMojoDiag(operand.expr->getLoc());
          diag << "unbound syntax (i.e. `_`) cannot be passed as a variadic "
                  "parameter";
          return failure();
        }

        // Merge if needed.
        if (mergedTypeBound) {
          operand.ir =
              UpcastAttr::get(mergedTypeBound, operand.ir.getIfPValue());
        }

        // FIXME: pack and install variadics parameter correctly.
        FailureOr<TypedAttr> paramVal =
            inferAndEmitOneParam(operand, varArgsEltType, idx);
        if (failed(paramVal)) // Exit if an error was already emitted.
          return failure();

        if (!*paramVal) {
          isDeferred = true;
          continue;
        }

        varArgsEltType = evaluator.getReboundType(varArgsEltType);
        // Realign sugar.
        if (paramVal->getType() != varArgsEltType)
          paramVal = ParamOperatorAttr::getRebind(*paramVal, varArgsEltType);
        elements.push_back(*paramVal);
      }

      if (!isDeferred) {
        // Infer the values list to the elements.
        auto vaType = evaluator.getReboundType(expectedValueList.getType());
        auto paramVA =
            ParamListAttr::get(elements, cast<ParamListType>(vaType));
        ParamMatcher matcher(getGivenBindings().callExpr, *this,
                             /*implConversions*/ false);
        if (failed(matcher.matchParams(paramVA, expectedValueList)))
          return failure();
        // The ParameterList now has a concrete type.
        auto listValue =
            LIT::getEmptyStructValue(evaluator.getReboundType(expectedType));
        setInferredValue(idx, listValue);
      }
      continue;
    }
  }

  return success();
}

VerifiedParamBindings ParamInf::inferForStruct() {
  CrashReporter handler(paramBindings.getExprLoc(), "ParamInf::inferForStruct",
                        getShared());

  auto attachNoteOnError = llvm::scope_exit([&]() {
    if (diag.hasErrorEmitted() && declIfKnown) {
      if (llvm::isa_and_nonnull<FnOp>(declIfKnown->getIfOperation())) {
        diag.attachNote(*declIfKnown) << "function declared here";
      } else {
        diag.attachNote(*declIfKnown)
            << "'" << *declIfKnown->getUserNameIfOperation()
            << "' declared here";
      }
    }

    // Skip the unprovable-constraint diagnostic when the caller has asked
    // the inference state to discard errors (i.e. it's filtering candidates,
    // not committing to this binding).
    if (!diag.isDiscarding() && !bodyUnprovableConstraints.empty()) {
      // If the caller installed a deferred typing context, record
      // the body-unprovable constraints there and suppress the error.
      // Otherwise, fall back to the existing hard-error path.
      if (deferredTypingContext) {
        SMLoc deferralLoc = paramBindings.getExprLoc();
        for (ConstraintAttr c : bodyUnprovableConstraints)
          deferredTypingContext->deferredConstraints.push_back(
              {c, deferralLoc});
      } else {
        emitUnprovableConstraintsFromFitness(
            bodyUnprovableConstraints, paramBindings.shared,
            paramBindings.getExprLoc(), declIfKnown);
      }
    }
  });

  isInferForStruct = true;

  if (failed(inferFromParamList()))
    return {};

  if (failed(inferFromBodyConstraints()))
    return {};

  if (paramBindings.bindingKind != ParamBindings::kWithEllipsis &&
      failed(inferFromDefaults())) {
    return {};
  }

  if (failed(finalizeWithUnbound()))
    return {};

  if (failed(checkBodyConstraints()))
    return {};

  ParameterExprArrayAttr rawBindings = ParameterExprArrayAttr::get(
      getShared().getContext(), evaluator.getIndexBindings());
  return VerifiedParamBindings(rawBindings,
                               std::move(dischargedBodyConstraints),
                               evaluator.getEvaluationContext());
}

// Infer any missing parameter from defaulted value (this is supposed to be
// invoked after both parameter list and argument list has been scanned).
LogicalResult ParamInf::inferFromDefaults() {

  auto setDefault = [&](TypedAttr value, size_t idx) -> LogicalResult {
    // The default value is explicitly unbound.
    if (explicitlyUnboundParams[idx])
      return success();

    value = evaluator.getReboundAttribute(value);

    // Don't try to infer from default values that have unresolved references
    // to other parameters.
    //
    // TODO: If the references points to a `_` parameter, we might still want to
    // install it without erasing the index reference.
    if (paramFinder.hasReferences(value))
      return success();

    ASTType argType = evaluator.getReboundType(declaredParamTypes[idx]);
    FailureOr<TypedAttr> paramVal = inferAndEmitOneParam(
        {value, getGivenBindings().getExpr()}, argType, idx);
    if (failed(paramVal))
      return failure();
    if (*paramVal && !evaluator.getIndexBindings()[idx] &&
        !deferredGivenParams[idx])
      setInferredValue(idx, *paramVal, /*isDefaulted=*/true);
    return success();
  };

  // Lastly, See if we can fulfill any missing parameters with default values
  // for their type (variadic attr always have a default empty value if not
  // inferable).
  for (size_t idx = 0, e = declaredParamTypes.size(); idx != e; ++idx) {
    if (evaluator.getIndexBindings()[idx])
      continue;

    // If available, we use a default parameter value.
    if (TypedAttr defaultParam = declaredParamPogs.getDefault(idx)) {
      // Default parameter values may reference other parameter values, so we
      // need to evaluate these.
      // If the default value is dependent, and we can not fully resolve all its
      // dependencies, do not try to set the value of it.
      if (failed(setDefault(defaultParam, idx)))
        return failure();
      continue;
    }

    // FIXME: this need a more systematical fix.
    // Determine if we can use a default parameter for CTAD
    if (paramBindings.ctadPogs.size() > idx) {
      if (TypedAttr defaultCTAD =
              paramBindings.ctadPogs[idx].getDefaultValue()) {
        defaultCTAD = evaluator.getReboundAttribute(defaultCTAD);
        if (!paramFinder.hasReferences(defaultCTAD)) {
          if (failed(setDefault(defaultCTAD, idx)))
            return failure();
          continue;
        }
      }
    }

    // TODO: move the special handling of Origin outside the default parameter
    // inference.
    if (isInferForStruct)
      continue;

    // Otherwise, check to see if this is an singleton parameter like Origin. So
    // long as its type is fully resolved, we can go ahead and instantiate it.
    if (auto paramType =
            sugarDynCast<LIT::StructType>(declaredParamTypes[idx])) {
      if (paramType.getSymbol().getLeafReference().strref() == "Origin" &&
          paramType.getParamValues().size() == 2 &&
          isa<OriginType>(paramType.getParamValues()[1].getType())) {
        IREmitter emitter(getDeclScope(), EC_TypeParamValue);

        paramType = cast<LIT::StructType>(evaluator.getReboundType(paramType));
        // Make sure the origin type is fully resolved before instantiating it.
        if (paramFinder.hasReferences(paramType))
          continue;

        auto origin = // Get the Origin value.
            evaluator.getReboundAttribute(paramType.getParamValues()[1]);

        // If the !lit.origin is unbound, then we have a partial binding - don't
        // bind a concrete Origin around an unbound Origin, just let other
        // things leave it unbound also.  We don't want things like
        // Span[mut=False] to bind the Origin.
        if (isa<UnboundAttr>(origin))
          continue;

        TypedAttr paramVal =
            emitter.getStdlibOriginOf(origin, getDeclScope().getLoc());
        setInferredValue(idx, paramVal);
        continue;
      }
    }
  }

  // Do another pass to fill in empty variadic, we need to do it after user
  // provided default value is installed, the variadic might be dependent by
  // those value in cases like:
  //
  // struct HasParamList[*values: Int]:
  //     def __init__(out self):
  //         pass
  //
  // struct HasDefaultParam[strides: HasParamList[...] = HasParamList[4]()]:
  //     pass
  for (size_t idx = 0, e = declaredParamTypes.size(); idx != e; ++idx) {
    if (evaluator.getIndexBindings()[idx])
      continue;

    // If not specified/inferrable, variadic always have a default empty value.
    bool isInferableVA = [&]() -> bool {
      auto pog = declaredParamPogs.getPogs()[idx];
      // Since we reached this point, the parameter binding can not have `...`,
      // and according to the rules. It must be producing the most concrete
      // type. So, if this is a positional variadic, we always default it to
      // empty.
      if (pog.isPosVarArg())
        return true;

      // Parameters from an enclosing struct are smashed onto the beginning of
      // method parameter lists, and their types are switched to Inferred. As
      // part of that, we lose track of whether it was pos_var_arg.
      // E.g.,
      //
      // struct S[*values: Int]:
      //     @staticmethod
      //     def foo():
      //         pass
      //
      // # we should be able to infer *value to empty.
      // S.foo()
      //
      // FIXME: maybe we really should preserve the variadic kind when
      // prepending contextual parameters such that we don't need the check
      // here? But on the other hand, what does it mean to have a
      // inferred-pos-var-arg parameter?
      if (pog.getPassingKind() == PassingKind::Inferred &&
          ASTType(declaredParamTypes[idx]).getParameterListInfo().valueList)
        return !isInferForStruct;

      return false;
    }();

    if (isInferableVA) {
      // Infer the param_list to an empty list, and the ParameterList itself to
      // its singleton value.
      auto [varArgsEltType, expectedValueList] =
          ASTType(evaluator.getReboundType(declaredParamTypes[idx]))
              .getParameterListInfo();

      // If there are no values, default to an empty list.
      if (isa<ParamIndexRefAttr>(expectedValueList)) {
        auto paramVA = ParamListAttr::get(
            {}, cast<ParamListType>(expectedValueList.getType()));
        ParamMatcher matcher(getGivenBindings().callExpr, *this,
                             /*implConversions*/ false);
        if (failed(matcher.matchParams(paramVA, expectedValueList))) {
          auto &diag = getMojoDiag({});
          diag << "could not infer default variadic parameter "
               << declaredParamPogs.getPogs()[idx].getName();
          return failure();
        }
      }

      // The list itself doesn't have a value, so default it to {} now that it
      // has a concrete type.
      auto listValue = LIT::getEmptyStructValue(
          evaluator.getReboundType(declaredParamTypes[idx]));
      setInitialInferredValue(idx, listValue);
    }
  }

  return success();
}

namespace {
/// Collects the indices of same-scope parameter references
struct ParamIndexRefCollector
    : public IndexParameterReplacer<ParamIndexRefCollector> {
  Attribute tryReplace(Attribute attr, size_t depth) {
    if (auto ref = dyn_cast<ParamIndexRefAttr>(attr);
        ref && ref.getDepth() == depth)
      indices.insert(ref.getIndex());
    return nullptr;
  }
  Type tryReplace(Type, size_t) { return {}; }

  static void collect(TypedAttr value, llvm::SmallDenseSet<size_t, 4> &out) {
    ParamIndexRefCollector collector;
    collector.replace(value);
    out.insert(collector.indices.begin(), collector.indices.end());
  }

  llvm::SmallDenseSet<size_t, 4> indices;
};
} // namespace

/// The two sides of an equality proposition.
using EqualityPair = std::pair<TypedAttr, TypedAttr>;

/// Collect the individual equality (`==`) propositions.
static void collectEqualityPropositions(TypedAttr prop,
                                        SmallVectorImpl<EqualityPair> &out) {
  if (std::optional<EqualityPair> identity = getIdentityProposition(prop)) {
    out.push_back(*identity);
  } else if (auto op = sugarDynCast<ParamOperatorAttr>(prop)) {
    if (op.getOpcode() == POC::And)
      for (TypedAttr operand : op.getOperands())
        collectEqualityPropositions(operand, out);
  }
}

LogicalResult ParamInf::inferFromBodyConstraints() {
  ArrayRef<ConstraintAttr> bodyConstraints =
      declaredParamPogs.getBodyConstraints();
  if (bodyConstraints.empty())
    return success();

  // Gather every equality proposition, flattening conjunctions so a
  // `where a == b and b == c` clause contributes both equalities rather than a
  // single (non-invertible) `and` proposition.
  SmallVector<EqualityPair> equalities;
  for (ConstraintAttr constraint : bodyConstraints)
    collectEqualityPropositions(constraint.getProposition(), equalities);
  if (equalities.empty())
    return success();

  const ExprNode *expr = getGivenBindings().getExpr();

  // Map each parameter to the equalities that reference it.
  DenseMap<size_t, SmallVector<size_t>> paramToEqualities;
  for (size_t idx = 0; idx < equalities.size(); ++idx) {
    llvm::SmallDenseSet<size_t, 4> refs;
    ParamIndexRefCollector::collect(equalities[idx].first, refs);
    ParamIndexRefCollector::collect(equalities[idx].second, refs);
    for (size_t param : refs)
      paramToEqualities[param].push_back(idx);
  }

  // Solve equalities off a worklist seeded with all of them. Inferring a
  // parameter from one equality can make another usable, so binding a
  // parameter re-queues only the equalities that reference it.
  //
  // The total number of queue entries is bounded by the seed plus the number
  // of (parameter -> equality) dependency edges built above.
  SmallVector<size_t> worklist;
  for (size_t idx = 0; idx < equalities.size(); ++idx)
    worklist.push_back(idx);

  while (!worklist.empty()) {
    EqualityPair eq = equalities[worklist.pop_back_val()];

    // Fold any get witness expressions in the constraint. We can only drive
    // inference when exactly one side is fully determined and the other still
    // has unbound parameters.
    TypedAttr lhs = evaluator.getReboundAttribute(eq.first);
    TypedAttr rhs = evaluator.getReboundAttribute(eq.second);
    bool lhsOpen = paramFinder.hasReferences(lhs);
    bool rhsOpen = paramFinder.hasReferences(rhs);
    if (lhsOpen == rhsOpen)
      continue;

    TypedAttr determined = lhsOpen ? rhs : lhs;
    TypedAttr unbound = lhsOpen ? lhs : rhs;

    // A successful match binds precisely the open side's parameters (the
    // determined side is concrete).
    llvm::SmallDenseSet<size_t, 4> unboundParams;
    ParamIndexRefCollector::collect(unbound, unboundParams);

    ParamMatcher matcher(expr, *this, /*allowImplicitConversions=*/false);
    ParamMatcher::FailableScope failableScope(matcher);
    if (succeeded(matcher.matchParams(determined, unbound))) {
      // Re-queue the equalities referencing a parameter this match bound.
      for (size_t param : unboundParams) {
        if (!evaluator.getIndexBindings()[param])
          continue;
        for (size_t dep : paramToEqualities[param])
          worklist.push_back(dep);
      }
      continue;
    }

    // The constraint isn't usable for inference (yet, or at all). Roll back
    // any tentative bindings the failed match made; genuine violations are
    // reported later by `checkBodyConstraints`.
    if (matcher.failureReason)
      failableScope.revert();
  }

  return success();
}

// TODO: We probably don't have to do this? This is just to make sure we reached
// the same end state as the old parameter inference. Understand why.
LogicalResult ParamInf::finalizeWithUnbound() {
  bool defaultToUnbound = paramBindings.bindingKind != ParamBindings::kStandard;

  auto emitInferenceFailure = [&](size_t paramIdx) {
    MojoInflightDiag &diag = getMojoDiag(paramBindings.getExprLoc());
    if (declIfKnown &&
        isa_and_nonnull<StructDeclOp>(declIfKnown->getIfOperation()))
      diag << "'" << *declIfKnown->getUserNameIfOperation() << "' ";

    {
      // The parameter name is scoped to 'declScope'.
      DeclResolver::DiagnosticDeclContextChanger x(&paramBindings.declScope);
      diag << "failed to infer parameter "
           << ParamDeclRefAttr::get(declaredParamPogs.getName(paramIdx),
                                    declaredParamTypes[paramIdx]);
    }

    // If this is a method on a struct and we couldn't infer something from
    // its self parameters, complain about the struct.
    if (declIfKnown && isa_and_nonnull<FnOp>(declIfKnown->getIfOperation())) {
      if (auto structOp = dyn_cast<StructDeclOp>(
              cast<FnOp>(declIfKnown->getIfOperation())->getParentOp())) {
        auto structSig = structOp.getSignature();
        if (paramIdx < structSig.getNumParams()) {
          diag << " of parent struct '" << structOp.getDeclName().getValue()
               << "'";

          if (auto *parentDecl = declIfKnown->getParentDecl())
            diag.attachNote(*parentDecl) << " struct declared here";
          else
            diag.attachNote(structOp.getLoc()) << " struct declared here";
          return;
        }
      }
    }

    if (isInferForStruct)
      diag << ", specify the parameter or use '_' or '...' to unbind the "
              "parameter explicitly";
  };

  // This is the end of parameter inference, replace any fail-to-infer parameter
  // to unboundAttr.
  for (auto [idx, pog] : llvm::enumerate(declaredParamPogs.getPogs())) {
    TypedAttr inferred = evaluator.getIndexBindings()[idx];
    if (inferred) {
      assert(!sugarIsa<UnboundAttr>(inferred));
      continue;
    }

    bool installUnbound = [&]() -> bool {
      // Call must produce a concrete type
      if (!isInferForStruct)
        return false;

      // There is a explicit unbound value provided and unbound is allowed.
      if (isExplicitlyUnbound(idx))
        return true;

      // `...` provides a default `_` value for any missing parameter. Besides,
      // we always allow inferred-only/implicit auto-parameterized parameter to
      // be defaulted to `_`. This is to allow:
      //
      // struct S[a: Int, //, b: Param[a]]:
      //   pass
      //
      // comptime _ = S[_] # NOTE that we don't require `a = _` here.
      //
      return pog.getPassingKind() == PassingKind::Inferred ||
             pog.getPassingKind() == PassingKind::Implicit || defaultToUnbound;
    }();

    if (installUnbound) {
      Type targetType = evaluator.getReboundType(declaredParamTypes[idx]);
      evaluator.overwriteIndexBinding(idx, UnboundAttr::get(targetType));
      continue;
    }

    if (pog.getPassingKind() != PassingKind::KwOnly) {
      emitInferenceFailure(idx);
      return failure();
    }

    // Error on a missing keyword parameter.
    MojoInflightDiag &diag = getMojoDiag({});
    diag << "missing required keyword-only parameter: " << pog.getName();
    if (isInferForStruct)
      diag << ", specify the parameter or use '_' or '...' to unbind the "
              "parameter explicitly";
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CallParamInf Implementation
//===----------------------------------------------------------------------===//

CallParamInf::CallParamInf(const ParamBindings &paramBinding,
                           ArrayRef<Type> declaredParamTypes,
                           PogListAttr declaredParamPogs,
                           bool allowImplicitConversions, ASTDecl *declIfDirect,
                           bool discardError,
                           FnTypeGeneratorType calleeSignature,
                           const CallOperands &callOperands,
                           const CallOperands::PogAssignment &pogAssignment,
                           OperandsNeedingOriginsList &operandsNeedingOrigins,
                           ArrayRef<ConstraintAttr> additionalAssumptions)
    : ParamInf(paramBinding, declaredParamTypes, declaredParamPogs,
               allowImplicitConversions, declIfDirect, discardError,
               /*deferredTypingContext=*/nullptr, additionalAssumptions),
      calleeSignature(calleeSignature), callOperands(callOperands),
      pogAssignment(pogAssignment),
      operandsNeedingOrigins(operandsNeedingOrigins) {}

/// Check the expected type against the provided operand. This identifies any
/// problems with the operand type, which it handled by emitting a diagnostic
/// and returning failure.
///
/// This can be called on a function signature with incomplete bindings, which
/// means that 'origExpectedType' may have unbound parameters.  As such, this
/// will infer parameters from the operand and return the inferred type.
///
/// operandIdx indicates the index of the operand in the CallOperands list, the
/// argIdx indicates which declared argument this corresponds to.  Note that
/// these may differ when using keyword arguments, and variadics have multiple
/// values that fulfill the same declared argument.
LogicalResult CallParamInf::inferOneOperand(ASTExprAnd<AnyValue> operand,
                                            size_t operandIdx, size_t argIdx,
                                            ASTType expectedType,
                                            ArgConvention expectedConvention) {

  auto argPogs = calleeSignature.getArgListAttrs();

  // Make sure the diagnostic machinery knows about our getDeclScope() so
  // parameter names get emitted correctly.
  DeclResolver::DiagnosticDeclContextChanger x(declIfKnown);

  auto emitWrongTypeDiag = [&](ASTType expectedType) -> MojoInflightDiag & {
    auto &diag = getMojoDiag(operand.expr->getLoc());
    ::emitWrongTypeDiag(diag, operand, expectedType, argIdx, argPogs,
                        callOperands.syntax, getShared(), getDeclScope());
    return diag;
  };

  expectedType = evaluator.getReboundType(expectedType);
  ASTType expectedRVType =
      RefType::stripRefConvention(expectedType, expectedConvention);

  // TODO: Calculate OverloadFitness's fitness (# implicit conversions etc).
  ParamMatcher matcher(operand.expr, *this, allowImplicitConversions);

  // This gets set if we need to spill the argument to memory to get an origin.
  bool needsArgInMemory = false;

  // We'll bind the next provided value.
  switch (expectedConvention) {
  case ArgConvention::OwnedReg:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError:
    llvm_unreachable("not used by the mojo parser");
  case ArgConvention::Mut: {
    // The actual value must be an lvalue if callee takes things by-ref.
    auto argVal = operand.ir.getIfLValue();
    if (!argVal) {
      auto &diag = getMojoDiag(operand.expr->getLoc());
      if ((callOperands.syntax == CallSyntax::kMethodCall ||
           callOperands.syntax == CallSyntax::kMethodCallSynthetic) &&
          argIdx == 0) {
        diag << "invalid use of mutating method on rvalue of type ";
        if (ASTType type = operand.ir.getRValueTypeIfResolvable())
          diag << type;
        else
          printUValueTypeInfo(operand.ir, diag);
      } else {
        diag << "value passed to mutable argument " << argPogs.getName(argIdx)
             << " must be mutable";
      }
      diag << operand.expr->getRange();
      return failure();
    }

    // If this is a wildcard type, we can match any operand.
    if (sugarIsa<NameLookupArgWildcardType>(argVal.getRValueType()))
      return success();

    // Ok we have an LValue.  The reference element types must match.
    if (failed(matcher.matchTypes(argVal.getRValueType(), expectedRVType))) {
      // ByRef argument types must exactly match, no conversions are allowed.
      auto &diag = getMojoDiag(operand.expr->getLoc());
      diag << "l-value of type " << operand.ir.getIfLValue().getRValueType()
           << " cannot be converted to reference of type " << expectedRVType
           << operand.expr->getRange();
      matcher.failureReason->addExplanation(diag);
      return failure();
    }
    break;
  }
  case ArgConvention::Ref:
  case ArgConvention::MutRef: {
    auto expectedRef = sugarCast<RefType>(expectedType);

    // If we are binding the reference to a value in memory directly, check for
    // reference compatibility directly.
    if (operand.ir.isMValue()) {
      RefType valueRefType = operand.ir.getMValueType();
      // If the IRValue type is MBValue or MRValue then we need infer an
      // immutable ref, to match behavior where we don't allow passing an
      // MBValue or MRValue as 'mut'.
      if (!operand.ir.getIfMLValue() && !operand.ir.getIfMBPValue() &&
          !valueRefType.isMutableKnown(false))
        valueRefType = valueRefType.getWithMutability(false);

      // Refine the element type first.
      if (failed(matcher.matchTypes(valueRefType.getElementType(),
                                    expectedRef.getElementType()))) {
        emitWrongTypeDiag(expectedType);
        return failure();
      }
      expectedType = evaluator.getReboundType(expectedType);

      // Now that element type has been matched, see if the origin is already
      // specified, allow implicit conversions, allowing you to pass a concrete
      // origin to something expecting a union or AnyOrigin.  This check happens
      // here (instead of in matchTypes) because function arguments can be
      // rebound when origins disagree, but this isn't correct/possible in
      // arbitrary nested positions.
      if (!paramFinder.hasReferences(expectedType)) {
        if (!IREmitter::canZeroCostConvert(valueRefType, expectedType,
                                           getShared(), getDeclScope())
                 .isTrue()) {
          emitWrongTypeDiag(expectedType);
          return failure();
        }
      } else {
        // Otherwise, match the references as a whole - this matches the origins
        // up to infer from the value.
        if (failed(matcher.matchTypes(valueRefType, expectedType))) {
          emitWrongTypeDiag(expectedType);
          return failure();
        }
      }
      break;
    }

    // Otherwise, we are binding something like a PValue or SRValue to a
    // reference argument, which doesn't have a origin.  This is a problem
    // because origins can be propagated through the type system of the
    // function call to other arguments and they all need to line up.  We
    // handle this in two phases: during overload resolution we bind this to
    // an immortal origin, and then after the candidate is selected, we
    // re-emit these arguments to memory and re-infer all the parameters.
    //
    // One detail is how we do this: we bind these arguments to immutable
    // temporaries, because we specifically do NOT want 'ref' arguments with
    // parametric mutability to treat these things as mutable.
    if (sugarCast<RefType>(expectedType).isMutableKnown(true)) {
      auto &diag = getMojoDiag(operand.expr->getLoc());
      diag << "mutable reference argument " << argPogs.getName(argIdx)
           << "cannot bind to temporary value";
      return diag;
    }

    // Otherwise, we'll need to drop this value into a temporary. Notice this so
    // we can handle it after we infer the element type.
    needsArgInMemory = true;

    // Until then, infer it as AnyOrigin.  We bind the origin directly and then
    // handle it like any other argument because we can support
    // implicit conversions.
    auto anyOrigin =
        AnyOriginAttr::get(expectedRef.getContext(), /*isMut=*/false);
    ParamMatcher::FailableScope failableScope1(matcher);
    if (failed(
            matcher.matchSingleEltStruct(anyOrigin, expectedRef.getOrigin()))) {
      // Ignore failures because we only want to set a value if none is
      // already known so things aren't ambiguous.
      // TODO: it would be cleaner to check to see if this is already inferred
      // and only default it if not.
      failableScope1.revert();
    }

    // The address space of the temp will be the default.
    auto addrSpace =
        IntegerAttr::get(IndexType::get(expectedRef.getContext()), 0);

    ParamMatcher::FailableScope failableScope2(matcher);
    if (failed(matcher.matchSingleEltStruct(addrSpace,
                                            expectedRef.getAddressSpace()))) {
      failableScope2.revert();
    }

    // Handle the element type compatibility check below to allow implicit
    // conversions etc.
    [[fallthrough]];
  }
  case ArgConvention::OwnedMem:
  case ArgConvention::DeinitMem:
  case ArgConvention::ImmMem:
  case ArgConvention::ImmReg:
    break;
  }

  // Call the core matching logic after handling the convention.
  OperandsNeedingOriginsList forwardedNeedingOrigins;
  if (failed(inferFromRVType(operand, argIdx, expectedRVType, argPogs,
                             callOperands.syntax, &forwardedNeedingOrigins)))
    return failure();

  // TODO: We can potentially generalize beyond `operand.ir.getIfCValue()`?
  if (!needsArgInMemory && !forwardedNeedingOrigins.empty()) {
    // In order to construct the current argument, we need to to invoke a
    // implicit conversion that turns the argument into a memory value.  Keep
    // the conversion's signature and argument index, since the spill has to
    // satisfy the constructor rather than this call.
    OperandNeedingOrigin forwarded = forwardedNeedingOrigins.front();
    assertLegalForwardedOperandOrigins(forwardedNeedingOrigins);

    // Adjust the operand idx.
    forwarded.operandIdx = operandIdx;
    operandsNeedingOrigins.push_back(forwarded);
  }

  // We may have refined expectedRVType.
  expectedRVType = evaluator.getReboundType(expectedRVType);

  // If the argument needed to be spilled to memory to get an origin,
  // record it so call emission can reinfer and reemit this candidate if
  // selected from the overload set, but with the argument in a temporary
  // vardecl.
  if (needsArgInMemory)
    operandsNeedingOrigins.push_back(
        {operandIdx, argIdx, expectedRVType, calleeSignature});

  // If a register-passable type is being passed in-memory, remember this.
  if (expectedConvention != ArgConvention::ImmReg &&
      expectedRVType.isRegisterPassable(operand.expr->getLoc(), getShared()))
    ++numMismatchedConventions;

  // Allow overloading on "owned" vs "by-ref" arguments.
  // If the argument convention is owned but the operand is not an RValue then
  // we'll need to copy the value (or this is entirely invalid).  If the
  // argument convention is borrowed/ref but the value is an RValue then we have
  // an RValue decay.  Model these so that APIs can overload on owned vs
  // borrowed effectively.
  if (!operand.ir.getIfCValue() ||
      operand.ir.getIfCValue().getRValueType().isEqualCanon(expectedRVType)) {
    if (operand.ir.getIfBValue() || operand.ir.getIfLValue()) {
      // Heavily penalize implicit copies.
      if (expectedConvention == ArgConvention::OwnedMem ||
          expectedConvention == ArgConvention::DeinitMem)
        numMismatchedConventions += 2;
    } else {
      assert((operand.ir.getIfUValue() || operand.ir.getIfRValue()) &&
             "UValue and RValue expressions are always owned");
      // Slightly penalize RValue->ref conversions.
      if (expectedConvention != ArgConvention::OwnedMem &&
          expectedConvention != ArgConvention::DeinitMem)
        ++numMismatchedConventions;
    }
  }

  return success();
}

/// Try to infer parameters of Self from an initializer if specialized.
///
/// Consider:
///    struct S[a: Int]:
///      def __init__(out self): ...
///      def __init__(out self: S[1], x: Int): ...
///
/// When constructed with no arguments, the first constructor must be used and
/// it is impossible to infer the value of 'a', so you must use `S[1]()`.  This
/// is the usual case.
///
/// However the second initializer is more specialized due to its custom Self -
/// it only applies when 'a' is 1, so we can infer that would be the value to
/// use if it is selected because one arg is passed to the initializer `S(42)`.
///
/// This function helps to infer the 'a' parameter when more specialized.  This
/// custom logic is required because often (eg in this case) the "actual" type
/// will have UnboundAttr parameters, instead of fully bound ones like a normal
/// argument.
LogicalResult CallParamInf::inferSelfFromInitResult() {
  DeclResolver::DiagnosticDeclContextChanger x(declIfKnown);

  ASTType returnedType =
      evaluator.getReboundType(calleeSignature.getUserResultType());

  auto reportConflict = [&](size_t paramIdx, TypedAttr actual,
                            TypedAttr expected) -> LogicalResult {
    getMojoDiag(getGivenBindings().callExpr->getLoc())
        << "return type " << returnedType << " parameter "
        << ParamIndexRefAttr::get(/*depth*/ 0, paramIdx, actual.getType())
        << " value " << actual << " doesn't match expected value " << expected;
    return failure();
  };

  // Match up the parameter bindings if the 'actual' param is an UnboundAttr and
  // the expected has something more specific than a reference to the contextual
  // parameter.
  for (auto [idx, retParam] :
       llvm::enumerate(returnedType.getParamBindings())) {
    // If this is simply a reference to the enclosing parameter (as in a normal
    // Self) init, then we can't infer anything from it.  In the example above,
    // this ignores the "a" parameter in "def __init__() -> S[a]:" which is what
    // "out self" desugars to.
    auto selfParam = evaluator.getIndexBindings()[idx];
    if (retParam == selfParam)
      continue;

    // Otherwise, if the self parameter got inferred, propagate the result
    // from it to the returned parameter.  This handles things like:
    //   struct X[A: AnyType]:
    //     def __init__[T: Movable](arg: Int, out self: X[T]):
    // which gets used as X[String](42) inferring T and A.
    ParamMatcher matcher(getGivenBindings().callExpr, *this,
                         allowImplicitConversions);
    if (selfParam) {
      // TODO: Macro'ize this when error handling logic is fixed.
      if (failed(matcher.matchParams(selfParam, retParam))) {
        return reportConflict(idx, retParam, selfParam);
      }
    } else if (!paramFinder.hasReferences(retParam)) {
      // Otherwise if the the returned parameter has no unbound parameter
      // references then we infer the self parameter from it. This infers X=42:
      //   struct X[A: Int]:
      //     def __init__(out self: X[42]):
      auto selfType =
          evaluator.getReboundType(calleeSignature.getInputParamTypes()[idx]);
      auto selfParam = ParamIndexRefAttr::get(/*depth*/ 0, idx, selfType);
      if (failed(matcher.matchParams(retParam, selfParam))) {
        return reportConflict(idx, selfParam, retParam);
      }
    }
  }

  return success();
}

/// This method is called for ByRefResult arguments of the callee.  It checks to
/// see if the callee has a parametric address space or origin. If so, it looks
/// at the ExprDest the call is being emitted into and infers the desired
/// values, or marks it as needing to be spilled if not.
LogicalResult CallParamInf::inferResultSlot(RefType expectedRef, size_t argIdx,
                                            const ExprDest &dest) {

  // Penalize generic code slightly.
  if (ASTType(expectedRef.getElementType())
          .isRegisterPassable(getGivenBindings().callExpr->getLoc(),
                              getShared()))
    ++numMismatchedConventions;

  bool needsAddrSpace =
      paramFinder.hasReferences(expectedRef.getAddressSpace());
  bool needsOrigin = paramFinder.hasReferences(expectedRef.getOrigin());
  if (!needsAddrSpace && !needsOrigin)
    return success(); // Nothing to do.

  RefType actualRef;
  // If we have a concrete MLValue, we can use it to infer the desired values.
  if (MLValue mlDest = dest.getDirectMLValueIfPresent()) {
    actualRef = mlDest.getRefType();
  } else {
    // If the ExprDest lacks a concrete MLValue, we can't infer anything. We
    // need the caller to spill the result into a buffer and reinfer us. Until
    // then, bind it as AnyOrigin to avoid failing to infer the parameters.

    operandsNeedingOrigins.push_back({OperandNeedingOrigin::kExprDestOperandIdx,
                                      argIdx, expectedRef.getElementType(),
                                      calleeSignature});

    if (needsOrigin)
      actualRef = expectedRef.getWithOrigin(
          AnyOriginAttr::get(expectedRef.getContext(), /*isMut=*/true));
    if (needsAddrSpace)
      actualRef = actualRef.getWithAddressSpace(
          IntegerAttr::get(IndexType::get(expectedRef.getContext()), 0));
  }

  ParamMatcher matcher(getGivenBindings().callExpr, *this,
                       /*allowImplicitConversions=*/false);

  if (failed(matcher.matchSingleEltStruct(actualRef.getAddressSpace(),
                                          expectedRef.getAddressSpace())) ||
      failed(matcher.matchSingleEltStruct(actualRef.getOrigin(),
                                          expectedRef.getOrigin())))
    return failure();
  return success();
}

/// Given an incomplete parameter binding set, try to infer parameters on Self
/// of a method from the first argument.
LogicalResult CallParamInf::inferCTADParams() {
  // Consider "conditional conformance" cases like:
  //     struct X[A: AnyType]:
  //       def foo[B: Movable](self: X[B]): ...
  //
  // When resolving a function call like `someX.foo()`, we install the
  // bindings for 'A' from the typeof(someX) when resolving the
  // AttributeRefExpr and then infer 'B' from someX again.
  //
  // However, when we have something like `X.foo(someX)` we cannot install the
  // bindings for 'A' at AttributeRef resolution time, and 'someX' is only
  // bound by parameter inference to 'B'.  Notice this and infer the parameter
  // directly from A.  This is also important for operator resolution, which
  // works effectively the same way.
  //
  // TODO: Provide a first class representation for conditional conformance
  // that doesn't have us shadowing parameters like this!

  // We can only do this if we have an argument.
  assert(!callOperands.empty() && !callOperands[0].keyword &&
         "init should have positional self argument");

  auto selfConvention = calleeSignature.getArgConventions()[0];
  ASTType declaredSelfType = RefType::stripRefConvention(
      calleeSignature.getArgument(0), selfConvention);

  // Get the ASTDecl for the declared self type.  This will give us the struct
  // that we are referring to without bound parameters.
  ASTDecl *decl = declaredSelfType.getDecl(getShared());
  if (!decl)
    return success();

  // Get the Self type, with parameters bound to the structs CTAD parameters.
  ASTType selfType = decl->getTypeDeclSelf();
  if (!selfType)
    return success();

  // We need to convert named parameters like "T", which are ParamDeclRefAttr
  // into ParamIndexRefAttr(0) style of representation.
  if (auto structDecl =
          dyn_cast_or_null<StructDeclOp>(decl->getIfOperation())) {
    IndexRefRemapper remapper(structDecl.getParams(), /*resultParams*/ {});
    selfType = remapper.replace(selfType.mlirType);
  }

  // If passing self by reference, wrap the Self type with the RefType
  // paraphernalia like origins.
  if (hasAddress(selfConvention))
    selfType = sugarCast<RefType>(calleeSignature.getArgument(0))
                   .getWithElement(selfType);

  // Infer the first operand against this type - it was presumably already
  // inferred against the methods declared type of 'self' as well.
  return inferOneOperand(callOperands[0], /*operandIdx*/ 0, /*argIdx*/ 0,
                         selfType, selfConvention);
}

LogicalResult CallParamInf::inferOptionalLiteralSize() {
  if (callOperands.values.empty())
    return success();

  // TODO: we can potentially generalize it to other literals too, for
  // now just support __list_literal__, as we haven't finalized the
  // decision on how to expose the feature to the user yet.
  if (auto literalMarker = callOperands.values.back().keyword;
      !literalMarker || literalMarker.strref() != "__list_literal__") {
    return success();
  }

  for (auto [idx, pog] : llvm::enumerate(declaredParamPogs.getPogs())) {
    if (pog.getName().strref() == "__literal_size__") {
      IREmitter emitter(getDeclScope(), ExprContext::EC_ParameterList);
      SyntheticNode dummyNode(callOperands.getExprLoc());
      auto size = IntegerAttr::get(IndexType::get(getShared().getContext()),
                                   callOperands.values.size() - 1);
      PValue literalSize =
          emitter
              .emitInt({ASTExprAnd<AnyValue>(size, &dummyNode)},
                       ExprContext::EC_ParameterList)
              .getIfPValue();
      if (!literalSize) {
        getMojoDiag(callOperands.getExprLoc())
            << "cannot infer the size of the list literal with a previously "
               "diagnosed error";
        return failure();
      }

      setInferredValue(idx, literalSize);
      return success();
    }
  }
  return success();
}

VerifiedParamBindings CallParamInf::inferForCall() {
  isInferForStruct = false;

  CrashReporter handler(paramBindings.getExprLoc(),
                        "CallParamInf::inferForCall", getShared());

  // First try to infer parameters from the already provided bindings.
  if (failed(inferFromParamList()))
    return {};

  if (failed(inferOptionalLiteralSize()))
    return {};

  // Match up the operands provided by the call to the input arguments.  Keep in
  // mind that the callee signature might not match at all, so we have to be
  // careful here!
  PogListAttr argPogs = calleeSignature.getArgListAttrs();
  for (auto [expectedArgIdx, expectedConvention] :
       llvm::enumerate(calleeSignature.getArgConventions())) {

    // Note that 'calleeSignature' changes the type as we go, so don't use
    // llvm::enumerate on the argument type list!
    Type expectedType =
        evaluator.getReboundType(calleeSignature.getArgument(expectedArgIdx));

    switch (pogAssignment.operandIdxs[expectedArgIdx]) {
    case CallOperands::PogAssignment::kPA_Unspecified:
      // There is no provided operand for a by-ref result and error slot.
      // If this is the result slot with parametric components, attempt to
      // infer result origin/address space from it.
      if (expectedConvention == ArgConvention::ByRefResult) {
        auto expectedRef = sugarCast<RefType>(expectedType);
        if (failed(inferResultSlot(expectedRef, expectedArgIdx,
                                   callOperands.dest)))
          return {};
      } else {
        assert(expectedConvention == ArgConvention::ByRefError &&
               "unknown unspecified operand");
      }
      continue;

      // The normal case matches up individual values with operands.
    default: {
      size_t operandIdx = pogAssignment.operandIdxs[expectedArgIdx];
      const OperandValue &operand = callOperands[operandIdx];
      if (operand.unpackStyle == ArgUnpackStyle::kStar ||
          operand.unpackStyle == ArgUnpackStyle::kStarStar) {
        auto &diag = getMojoDiag(operand.expr->getLoc());
        diag << "unpacked positional arguments are only supported for callees "
                "that expect a variadic pack argument at this position; to "
                "forward a runtime pack to a fixed-arity callee, route the "
                "call through a dispatcher whose argument is itself a "
                "variadic pack (e.g. `def shim[Ts: TypeList[Trait=AnyType, "
                "...], //, callee: def(*args: *Ts) thin](...): "
                "callee(*pack)`)";
        return {};
      }
      if (failed(inferOneOperand(operand, operandIdx, expectedArgIdx,
                                 expectedType, expectedConvention)))
        return {};
      continue;
    }
    case CallOperands::PogAssignment::kPA_Default: {
      // Default values are matched.
      auto defaultVal = argPogs.getDefault(expectedArgIdx);
      assert(defaultVal && "default value is missing");
      defaultVal = evaluator.getReboundAttribute(defaultVal);
      if (failed(inferOneOperand({defaultVal, getGivenBindings().getExpr()},
                                 /*FIXME*/ ~0ULL, expectedArgIdx, expectedType,
                                 expectedConvention)))
        return {};
      continue;
    }

    case CallOperands::PogAssignment::kPA_Variadic:
      // Handle variadics below.
      break;
    }

    // Handle overload ranking for variadics.
    if (!pogAssignment.posVariadicIdxs.empty() ||
        !pogAssignment.kwVariadicIdxs.empty()) {
      // Remember that there is a variadic argument for overload ranking.
      passesVarArgArgument = true;
    } else {
      // We consider an empty varargs list to be an implicit conversion,
      // so an exact signature match takes precedence.
      ++numImplicitConversions;
    }

    // Keyword argument variadics.
    if (calleeSignature.isKwVarArg(expectedArgIdx)) {
      // Support forwarding an entire list with "**kwargs".
      if (pogAssignment.kwVariadicIdxs.size() == 1 &&
          callOperands[pogAssignment.kwVariadicIdxs[0]].unpackStyle ==
              ArgUnpackStyle::kStarStar) {
        size_t operandIdx = pogAssignment.kwVariadicIdxs[0];
        if (failed(inferOneOperand(callOperands[operandIdx], operandIdx,
                                   expectedArgIdx, expectedType,
                                   expectedConvention)))
          return {};
        continue;
      }

      Type valTy = ASTType(expectedType).getKwargsDictRefValueType();
      auto refValType = RefType::getAnyOrigin(valTy, /*isMut=*/true);
      for (auto operandIdx : pogAssignment.kwVariadicIdxs) {
        // KWVarArg values are passed to StringDict::_insert, which takes
        // the argument as an owned value (they are transferred into the dict).
        if (failed(inferOneOperand(callOperands[operandIdx], operandIdx,
                                   expectedArgIdx, refValType,
                                   ArgConvention::OwnedMem)))
          return {};
      }
      continue;
    }

    // Otherwise we have positional variadics: homogeneous or pack.

    // Determine if we can use an value for this argument directly, or
    // if we need an implicit conversion, or memory materialization to get
    // an origin.
    auto canUseMValue = [&](AnyValue value, ASTType expectedType,
                            ArgConvention convention) -> bool {
      // The operand must an MValue and must have the same element type as
      // the variadic list element type (otherwise a conversion is needed).
      if (!value.isMValue())
        return false; // Can't use it if not an MValue obviously.

      // The origin has to be in the default address space.
      if (!value.getMValueType().isDefaultAddrSpace())
        return false;

      // The argument must have a compatible element type (and we might
      // infer the type of the variadic from it.  If not, there must be an
      // implicit conversion going on.  We can test for type equality here
      // because inferOneOperand will have inferred the type from the arg.
      // TODO: Move this logic into inferOneOperand.
      expectedType = evaluator.getReboundType(expectedType);
      if (!expectedType.isEqualCanon(value.getMValueType().getElementType()))
        return false; // Implicit conversion will generate a new temp.

      // If this is a owned operand, we can use it if we have an RValue.
      if (convention == ArgConvention::OwnedMem)
        return !!value.getIfRValue();

      // TODO: What about "mut" arguments getting passed MBValues?
      return true;
    };

    // Given a call argument that will be bound to the specified operand of a
    // callee, get the memory origin of the value (if it can be used) or mark it
    // as needing to be spilled if not.
    auto getArgOrigin = [&](AnyValue value, ASTType expectedType, size_t argIdx,
                            size_t operandIdx, ArgConvention convention,
                            OriginType expectedOriginType) -> TypedAttr {
      if (canUseMValue(value, expectedType, convention)) {
        // The argument could be mutable, but the arg convention may expect
        // immutable.
        auto opOrigin = value.getMValueType().getOrigin();
        return OriginMutCastAttr::get(opOrigin, expectedOriginType);
      }
      // The value isn't in memory (or isn't usable in memory) yet.  We will
      // tell call emission that it needs to dump it in memory and try again
      // to use this callee.  Until then, we use AnyOrigin as a placeholder.
      operandsNeedingOrigins.push_back({operandIdx, argIdx,
                                        evaluator.getReboundType(expectedType),
                                        calleeSignature});
      return AnyOriginAttr::get(expectedOriginType);
    };

    // If we have a varargs argument, then it will eat the rest of the
    // arguments, but we have to check each of them.
    if (calleeSignature.isPosVarArg(expectedArgIdx)) {
      // Support forwarding an entire list with "*list".
      if (pogAssignment.posVariadicIdxs.size() == 1 &&
          callOperands[pogAssignment.posVariadicIdxs[0]].unpackStyle ==
              ArgUnpackStyle::kStar) {
        size_t operandIdx = pogAssignment.posVariadicIdxs[0];
        if (failed(inferOneOperand(callOperands[operandIdx], operandIdx,
                                   expectedArgIdx, expectedType,
                                   expectedConvention)))
          return {};
        continue;
      }

      // Otherwise, we're binding a sequence of values into the list.
      ASTType expectedRVType =
          RefType::stripRefConvention(expectedType, expectedConvention);
      // The expected origin type will always have its mutability known because
      // the arg convention of the VariadicList is always constant.
      auto variadicListInfo = expectedRVType.getVariadicListInfo();
      auto expectedOriginType =
          cast<OriginType>(variadicListInfo.origin.getType());
      auto argConvention =
          calleeSignature.getVariadicConvention(expectedArgIdx);

      // TODO: This is subtly wrong in a way that doesn't matter. We're passing
      // the ultimate origin in as the origin for the RefType, but we need to
      // infer the union all of the arg origins: not just the first arg's
      // origin.  inferOneOperand currently doesn't do anything with this except
      // for 'ref' convention, that we don't support in variadics.  When we do
      // or when we get rid of implicit origins, this will need to be adjusted
      // to pass in something that matches anything so the code below can
      // infer the correct origin union.
      auto varArgsEltType =
          RefType::get(variadicListInfo.elementType, variadicListInfo.origin);

      SmallVector<TypedAttr> argOrigins;
      for (auto operandIdx : pogAssignment.posVariadicIdxs) {
        auto &operand = callOperands[operandIdx];

        if (operand.unpackStyle == ArgUnpackStyle::kStar ||
            operand.unpackStyle == ArgUnpackStyle::kStarStar) {
          getMojoDiag(operand.expr->getLoc())
              << "cannot unpack a value into a variadic argument";
          return {};
        }

        if (failed(inferOneOperand(operand, operandIdx, expectedArgIdx,
                                   varArgsEltType, argConvention)))
          return {};

        // Keep track of all the arg origins so we can infer from them later.
        argOrigins.push_back(getArgOrigin(
            operand.ir, variadicListInfo.elementType, expectedArgIdx,
            operandIdx, argConvention, expectedOriginType));
      }

      // Infer the origin of the variadic list from the unified origins of the
      // arguments.
      auto commonOrigin = OriginUnionAttr::get(argOrigins, expectedOriginType);
      ParamMatcher matcher(getGivenBindings().callExpr, *this,
                           /*noImplicitConversions=*/false);
      if (failed(matcher.matchParams(commonOrigin, variadicListInfo.origin)))
        return {};

      continue;
    }

    // Otherwise we have a pack argument, then we're binding a variadic
    // parameter with multiple type values.  We need to consume all remaining
    // arguments and use their RValue types as bindings.
    assert(calleeSignature.isPack(expectedArgIdx) && "Unknown variadic");
    ASTType variadicPackType =
        RefType::stripRefConvention(expectedType, expectedConvention);
    variadicPackType = evaluator.getReboundType(variadicPackType);
    ASTType::VariadicPackInfo expectedInfo =
        variadicPackType.getVariadicPackInfo();

    // Support forwarding an entire pack with "*pack".
    if (pogAssignment.posVariadicIdxs.size() == 1 &&
        callOperands[pogAssignment.posVariadicIdxs[0]].unpackStyle ==
            ArgUnpackStyle::kStar) {
      size_t operandIdx = pogAssignment.posVariadicIdxs[0];
      auto &operand = callOperands[operandIdx];

      ASTType actualPackType = // FIXME: This is wrong for UValues.
          operand.ir.getRValueTypeIfResolvable();
      assert(actualPackType &&
             "unpacked positional operand must have a resolvable type");

      // Check that the actual type is the same struct type as the expected
      // VariadicPack. If not, the user tried to unpack a non-pack type
      // (e.g., a List) which is not allowed.
      ASTDecl *actualDecl = actualPackType.getDecl(getShared());
      ASTDecl *expectedDecl = variadicPackType.getDecl(getShared());
      if (!actualDecl || actualDecl != expectedDecl) {
        auto &diag = getMojoDiag(operand.expr->getLoc());
        diag << "cannot unpack value of type " << actualPackType
             << " into a variadic pack argument; expected a VariadicPack";
        return {};
      }

      ASTType::VariadicPackInfo actualInfo =
          actualPackType.getVariadicPackInfo();
      if (actualInfo.isOwned != expectedInfo.isOwned) {
        auto &diag = getMojoDiag(operand.expr->getLoc());
        diag << "cannot unpack a variadic pack into a call that requires a "
                "different ownership. Expected "
             << expectedInfo.isOwned << ", got " << actualInfo.isOwned;
        return {};
      }

      // Skip matching the origin, since the expected origin is an implicit
      // origin that will be filled in during call emission. Just make sure
      // that the element types match.
      RefPackType actualRefPackType =
          actualPackType.getVariadicPackInfo(getShared());
      RefPackType expectedRefPackType =
          variadicPackType.getVariadicPackInfo(getShared());

      auto actualMutable = actualRefPackType.getOriginType().getIsMutable();
      auto expectedMutable = expectedRefPackType.getOriginType().getIsMutable();
      auto bothMutable =
          ParamOperatorAttr::get(POC::And, actualMutable, expectedMutable);
      if (bothMutable != expectedMutable) {
        auto &diag = getMojoDiag(operand.expr->getLoc());
        diag << "cannot unpack a variadic pack into a call that requires a "
                "stricter mutability. Expected "
             << expectedMutable << ", got " << actualMutable;
        return {};
      }

      ParamMatcher matcher(operand.expr, *this, allowImplicitConversions);
      // Helper to format a diagnostic. `actualSide` and `expectedSide` get
      // the actual/expected types embedded so the user can see exactly what
      // each side looked like — both the element trait and the concrete
      // element-type list, since either may be the source of the conflict.
      auto emitPackMismatchDiag = [&]() -> MojoInflightDiag & {
        auto &diag = getMojoDiag(operand.expr->getLoc());
        diag << "cannot unpack a pack of type "
             << actualRefPackType.getParamListElementType() << " ("
             << actualRefPackType.getVariadic()
             << ") into a call that expects a pack of type "
             << expectedRefPackType.getParamListElementType() << " ("
             << expectedRefPackType.getVariadic() << ")";
        return diag;
      };
      if (failed(matcher.matchParams(actualRefPackType.getVariadic(),
                                     expectedRefPackType.getVariadic())) ||
          failed(matcher.matchParams(actualRefPackType.getOrigin(),
                                     expectedRefPackType.getOrigin()))) {
        auto &diag = emitPackMismatchDiag();
        matcher.failureReason->addExplanation(diag);
        return {};
      }

      // Now that we bound the elements of the TypeList, we can infer the
      // value of the TypeList struct.
      auto typeListType =
          evaluator.getReboundType(expectedInfo.typeListStruct.getType());
      auto typeListValue = LIT::getEmptyStructValue(typeListType);
      (void)matcher.matchParams(typeListValue, expectedInfo.typeListStruct);
      continue;
    }

    // Otherwise, we're binding a sequence of values into the pack.
    RefPackType packType = variadicPackType.getVariadicPackInfo(getShared());

    // Figure out that the element type of the list is, e.g. AnyType or
    // Stringable.
    Type elementType = packType.getParamListElementType();
    auto expectedOriginType = packType.getOriginType();

    // It is possible the pack element types are not being inferred - for
    // example, they could have been explicitly specified.  If this is the
    // case, then we need to perform an implicit conversion to the element
    // type that was explicitly specified.
    ParamListAttr eltsTypesIfResolved =
        dyn_cast<ParamListAttr>(packType.getVariadic());

    // Verify that the number of elements in the pack matches the arguments.
    if (eltsTypesIfResolved && eltsTypesIfResolved.getValues().size() !=
                                   pogAssignment.posVariadicIdxs.size()) {
      size_t numExpected = eltsTypesIfResolved.getValues().size();
      size_t numActual = pogAssignment.posVariadicIdxs.size();
      auto exprLoc = getGivenBindings().getExprLoc();
      if (!pogAssignment.posVariadicIdxs.empty())
        exprLoc = callOperands[pogAssignment.posVariadicIdxs[0]].expr->getLoc();
      MojoInflightDiag &diag = getMojoDiag(exprLoc);
      diag << "expected " << numExpected << " element" << plural(numExpected)
           << " in variadic pack, got " << numActual << " argument value"
           << plural(numActual);
      return {};
    }

    SmallVector<TypedAttr> types;
    SmallVector<TypedAttr> argOrigins;
    IREmitter emitter(getDeclScope(), EC_TypeParamValue);
    const ExprNode *packArgExpr = nullptr;
    for (auto operandIdx : pogAssignment.posVariadicIdxs) {
      const auto &operand = callOperands[operandIdx];

      if (operand.unpackStyle == ArgUnpackStyle::kStar ||
          operand.unpackStyle == ArgUnpackStyle::kStarStar) {
        getMojoDiag(operand.expr->getLoc())
            << "concatenating unpacked positional arguments is not supported";
        return {};
      }

      // Remember the first argument expression for the pack.
      if (packArgExpr == nullptr)
        packArgExpr = operand.expr;

      // If the element types for the pack were specified, convert the value
      // to that type.
      TypedAttr eltTypeValue;
      if (eltsTypesIfResolved) {
        eltTypeValue = eltsTypesIfResolved.getValues()[types.size()];
      } else {
        // Otherwise, infer the variadic element type from the value's type.
        ASTType toPush = operand.ir.getRValueTypeIfResolvable();

        // Initializer UValues (list/dict/set/slice literals) don't have a
        // resolvable RValue type until they're bound to a target type.
        // Apply the same fallback `inferCValue` uses so a literal passed to
        // a trait-bound pack binds to its default type instead of bailing
        // out with a bogus "unresolved type" diagnostic.
        if (!toPush) {
          if (auto initValue = operand.ir.getIfInitializer())
            toPush = tryInferInitializerType(getDeclScope(), *initValue,
                                             operand, ASTType(elementType));
        }

        if (!toPush) {
          getMojoDiag(operand.expr->getLoc())
              << "could not infer type of parameter pack "
              << argPogs.getName(expectedArgIdx)
              << " given value with unresolved type";
          return {};
        }

        // Infer nonmaterializable types as their materialization target.
        if (ASTType nmTarget = toPush.getNonmaterializableTarget(getShared()))
          toPush = nmTarget;

        Type metatype = toPush.extractMetaType();
        eltTypeValue = TypeParamAttr::get(toPush, metatype);
        // Make sure the value is compatible with the expected trait, this
        // produces better error messages.  It would be great to sink this
        // into matchType at some point!
        auto conversion = IREmitter::classifyImplicitConversionWithDetails(
            {eltTypeValue, operand.expr}, elementType, emitter.getDeclScope(),
            /*additionalAssumptions=*/{},
            /*deferralCtx=*/nullptr);
        if (!conversion.isYes()) {
          // Packs cannot be constrained by concrete types so elementType is
          // always a trait and reporting non-conformance instead of a type
          // mismatch is safe. This path is only reachable for packs (isPack
          // is asserted above), and ParamMatcher relies on the same invariant
          // via an unconditional cast<TraitType>.
          auto &diag = getMojoDiag(operand.expr->getLoc());
          diag << "an element of " << argPogs.getName(expectedArgIdx)
               << " with type " << toPush << " does not conform to trait "
               << elementType
               << "; either prove the conformance with 'conforms_to'"
                  ", or add conformance";
          if (conversion.isNo())
            std::move(conversion).getNo().addExplanation(diag);
          else
            attachConstraintNotes(diag, conversion.getUnknown(), "unproven");
          return {};
        }

        // Perform a conversion (e.g. from a concrete to trait type) as
        // needed.
        // FIXME(MOCO-3601): We have been very unprincipled about converting
        // using TypeParamAttr/UpcastAttr: They both are used as a way to
        // `rebind` type values. We have to use upcast here because we
        // have a upcast inserted for variadic element type for Tuple.
        if (!ASTType(eltTypeValue.getType()).isEqualCanon(elementType)) {
          if (isa<NonStructTypeType>(eltTypeValue.getType())) {
            eltTypeValue = emitter.emitPValue({eltTypeValue, operand.expr},
                                              EC_TypeParamValue, elementType);
          } else {
            eltTypeValue = UpcastAttr::get(elementType, eltTypeValue);
          }
        } else if (eltTypeValue.getType() != elementType) {
          // If they're only canonically equal, not exactly equal, insert a
          // conversion.
          eltTypeValue = emitter.emitPValue({eltTypeValue, operand.expr},
                                            EC_TypeParamValue, elementType);
        }
      }

      RefType refType =
          packType.getElementRefTypeFor(ASTType(eltTypeValue).mlirType);
      ArgConvention packEltConvention =
          calleeSignature.getVariadicConvention(expectedArgIdx);
      if (failed(inferOneOperand(operand, operandIdx, expectedArgIdx, refType,
                                 packEltConvention))) {
        return {};
      }

      // Keep track of all the arg origins so we can infer from them later.
      argOrigins.push_back(getArgOrigin(operand.ir, refType.getElementType(),
                                        expectedArgIdx, operandIdx,
                                        packEltConvention, expectedOriginType));
      types.push_back(eltTypeValue);
    }

    ParamMatcher matcher(packArgExpr, *this, allowImplicitConversions);

    // Infer the origin of the pack from the unified origins of the
    // arguments.
    auto commonOrigin = OriginUnionAttr::get(argOrigins, expectedOriginType);
    if (failed(matcher.matchParams(commonOrigin, packType.getOrigin())))
      return {};

    // Infer the value of type list from the types we have.
    auto variadicType =
        sugarCast<ParamListType>(packType.getVariadic().getType());

    // If there are no arguments for the pack, use the location of the call.
    if (!packArgExpr)
      packArgExpr = getGivenBindings().getExpr();
    auto actualVA = ParamListAttr::get(types, variadicType);
    if (succeeded(matcher.matchParams(actualVA, packType.getVariadic()))) {
      // Now that we bound the elements of the TypeList, we can infer the
      // value of the TypeList struct.
      auto typeListType =
          evaluator.getReboundType(expectedInfo.typeListStruct.getType());
      auto typeListValue = LIT::getEmptyStructValue(typeListType);
      (void)matcher.matchParams(typeListValue, expectedInfo.typeListStruct);
      continue;
    }

    // FIXME: This is a bad error, we could improve it.
    size_t numOperands = pogAssignment.posVariadicIdxs.size();
    MojoInflightDiag &diag = getMojoDiag({packArgExpr->getLoc()});
    diag << "assigning " << numOperands << " operand" << plural(numOperands)
         << " to an unresolvable variadic pack argument";
    return {};
  }

  // If this is a result in a returnsSelf function like an __init__, infer
  // self parameters (which could be specialized and shadowed).
  //   struct Example[T: AnyType]:
  //      def __init__[U: Movable](owned value: U) -> Example[U]:
  //         pass
  // All of the arguments have been resolved here so all parameters must be
  // inferred (or not able to).
  if (declIfKnown && declIfKnown->getIfOperation()) {
    if (cast<FnOp>(declIfKnown->getIfOperation())
            .getSpecialFunctionInfo()
            .hasSelfResult())
      if (failed(inferSelfFromInitResult()))
        return {};
  }

  // Check to see if this is a CTAD parameter - a parameter on the struct
  // that encloses the method.  Consider "conditional conformance" cases like:
  //     struct X[A: AnyType]:
  //       def foo[B: Movable](self: X[B]): ...
  // When resolving a function call like `someX.foo()`, we install the
  // bindings for 'A' from the typeof(someX) when resolving the
  // AttributeRefExpr and then infer 'B' from someX again.
  //
  // However, when we have something like `X.foo(someX)` we cannot install the
  // bindings for 'A' at AttributeRef resolution time, and 'someX' is only
  // bound by parameter inference to 'B'.  Notice this and infer the parameter
  // directly from A.  This is also important for operator resolution, which
  // works effectively the same way.
  //
  // TODO: Provide a first class representation for conditional conformance
  // that doesn't have us shadowing parameters like this!
  // A trait member's enclosing declaration is a trait rather than a struct, so
  // the struct-shaped CTAD inference never applies to one.
  if (declIfKnown && !declIfKnown->getIfWitness()) {
    auto fnOp = cast<FnOp>(declIfKnown->getIfOperation());
    if (!fnOp.getIsStatic() && isa<StructDeclOp>(fnOp->getParentOp())) {
      if (failed(inferCTADParams()))
        return {};
    }
  }

  // Infer any parameters that are only reachable through equality `where`
  // clauses.
  if (failed(inferFromBodyConstraints()))
    return {};

  // Lastly, See if we can fulfill any missing parameters with default values
  // for their type (variadic attr always have a default empty value if not
  // inferable).
  if (failed(inferFromDefaults()))
    return {};

  if (deferredGivenParams.any()) {
    // Simply try it again now that more parameter has been inferred.
    if (failed(inferFromParamList()))
      return {};
  }

  // See if we still have any unbound attr, if so, report error. (This must be a
  // full binding context).
  if (failed(finalizeWithUnbound()))
    return {};

  if (failed(checkBodyConstraints(paramBindings.additionalConstraints)))
    return {};

  ParameterExprArrayAttr rawBindings = ParameterExprArrayAttr::get(
      getShared().getContext(), evaluator.getIndexBindings());
  return VerifiedParamBindings(rawBindings,
                               std::move(dischargedBodyConstraints),
                               evaluator.getEvaluationContext());
}
