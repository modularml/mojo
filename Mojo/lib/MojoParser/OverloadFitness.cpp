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
// This file implements the components for overload fitness evaluation.
//
//===----------------------------------------------------------------------===//

#include "OverloadFitness.h"
#include "ClosureEmitter.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParamInf.h"

#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/LITDialect/SpecialFunctions.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Diagnostic emission implementation
//===----------------------------------------------------------------------===//

/// Attach extra type conversion error detail or hints to the user when
/// reporting an error passing `operand` to an argument of type `argType`.
static void addTypeConversionDetail(MojoInflightDiag &diag,
                                    ASTExprAnd<AnyValue> operand,
                                    ASTType argType) {
  auto loc = operand.expr->getLoc();
  ASTType operandType = operand.ir.getRValueTypeIfResolvable();
  if (!operandType) {
    diag.attachNote(loc) << "try resolving the overloaded function first";
    return;
  }
  // Try to detect mismatched memory result type.
  auto lhsSig = sugarDynCast<FnTypeGeneratorType>(operandType);
  auto rhsSig = sugarDynCast<FnTypeGeneratorType>(argType);
  if (lhsSig && rhsSig) {
    auto getByRefResult = [](FnTypeGeneratorType sig) -> std::pair<bool, Type> {
      return {sig.getBody().hasMemoryOnlyResult(),
              ASTType(sig.getUserResultType())};
    };
    auto [lhsByRef, lhsRetType] = getByRefResult(lhsSig);
    auto [rhsByRef, rhsRetType] = getByRefResult(rhsSig);
    if (lhsByRef == rhsByRef || lhsRetType != rhsRetType)
      return;
    // Different result semantics but same result type.
    diag.attachNote(loc) << "memory-only type bound to generic result type: "
                         << (lhsByRef ? "payload" : "argument") << " returns "
                         << ASTType(lhsRetType) << " by reference";
    return;
  }
}

/// Emit a tailored diagnostic when failing to convert a value to type !lit.ref.
/// This happens when the user is forming a Reference incorrectly which happens
/// when confusion and details run the highest.
static void diagnoseFailedRefTypeConversion(MojoInflightDiag &diag,
                                            ASTExprAnd<AnyValue> operand,
                                            RefType argType,
                                            SharedState &shared,
                                            ASTDecl &declScope) {
  diag << "ref " << ASTType(argType.getElementType());

  auto loc = operand.expr->getLoc();
  if (operand.ir.getIfRValue()) {
    diag.attachNote(loc) << "cannot bind an RValue to a reference";
    return;
  }
  if (!operand.ir.isMValue()) {
    diag.attachNote(loc) << "operand does not have a memory representation";
    return;
  }

  auto operandRefTy = operand.ir.getMValueType();
  if (!ASTType(argType.getElementType())
           .isEqualCanon(operandRefTy.getElementType())) {
    diag.attachNote(loc) << "operand element type "
                         << ASTType(operandRefTy.getElementType())
                         << " doesn't match expected element type "
                         << ASTType(argType.getElementType());
  } else if (argType.getAddressSpace() != operandRefTy.getAddressSpace()) {
    diag.attachNote(loc) << "operand address space "
                         << operandRefTy.getAddressSpace()
                         << " doesn't match expected address space "
                         << argType.getAddressSpace();
  } else if (!IREmitter::canZeroCostConvert(operandRefTy.getOriginType(),
                                            argType.getOriginType(), shared,
                                            declScope)
                  .isTrue()) {
    diag.attachNote(loc) << "operand mutability " << operandRefTy.isMutable()
                         << " doesn't match expected mutability "
                         << argType.isMutable();
  } else if (!IREmitter::canZeroCostConvert(operandRefTy, argType, shared,
                                            declScope)
                  .isTrue()) {
    auto operandO = operandRefTy.getOrigin();
    auto argO = argType.getOrigin();
    // Strip off mutcasts etc - if the origins still differ we can complain
    // about the simpler thing.
    auto operandOS = OriginType::stripMutCastAndRebind(operandO);
    auto argOS = OriginType::stripMutCastAndRebind(argO);
    if (operandOS != argOS) {
      argO = argOS;
      operandO = operandOS;
    }

    diag.attachNote(loc) << "operand origin '"
                         << ASTType::getOriginAsString(operandO, &shared)
                         << "' doesn't match expected origin '"
                         << ASTType::getOriginAsString(argO, &shared) << "'";
  }
}

namespace M::KGEN::LIT {
void printUValueTypeInfo(const AnyValue &value, MojoInflightDiag &diag) {
  if (auto initList = value.getIfInitializer()) {
    switch (initList->syntax) {
    case InitializerUValue::kSliceLiteral:
      diag << "slice literal";
      break;
    case InitializerUValue::kListLiteral:
      diag << "list literal";
      break;
    case InitializerUValue::kDictLiteral:
      diag << "dictionary literal";
      break;
    case InitializerUValue::kSetInitLiteral:
      diag << "initializer list or set literal";
      break;
    }
  } else if (value.getIfInferredBaseAttrRef()) {
    diag << "inferred attribute reference";
  } else
    diag << "unknown overload";
}

void emitWrongTypeDiag(MojoInflightDiag &diag, ASTExprAnd<AnyValue> operand,
                       ASTType expectedType, size_t argIdx,
                       PogListAttr argListAttr, CallSyntax syntax,
                       SharedState &shared, ASTDecl &declScope) {
  // Special case implicit conversions with a custom message.
  if (syntax == CallSyntax::kImplicitConvert) {
    if (ASTType type = operand.ir.getRValueTypeIfResolvable())
      diag << type;
    else
      printUValueTypeInfo(operand.ir, diag);
    diag << " value to " << expectedType;
    return;
  }

  // When the function has no source-level pog metadata (an empty PogList,
  // e.g. a `FuncType` constructed outside the Mojo parser), fall back to a
  // generic diagnostic without the argument name.
  if (StringAttr name = argListAttr.getName(argIdx); !name.empty()) {
    diag << "value passed to " << name << " cannot be converted from "
         << operand.expr->getRange();
  } else {
    diag << "value cannot be converted from " << operand.expr->getRange();
  }
  ASTType rValueType = operand.ir.getRValueTypeIfResolvable();
  bool isConvertingTypeValue = expectedType.extractMetaType() == rValueType;
  if (rValueType) {
    if (isConvertingTypeValue)
      diag << "type value " << expectedType;
    else
      diag << rValueType;
  } else {
    printUValueTypeInfo(operand.ir, diag);
  }
  diag << " to ";

  if (auto refType = sugarDynCast<RefType>(expectedType)) {
    diagnoseFailedRefTypeConversion(diag, operand, refType, shared, declScope);
    return;
  }

  diag << (isConvertingTypeValue ? "an instance of " : "") << expectedType;
  if (isConvertingTypeValue)
    diag << "; did you mean to instantiate " << expectedType << "?";
  addTypeConversionDetail(diag, operand, expectedType);
}
} // namespace M::KGEN::LIT

//===----------------------------------------------------------------------===//
// OverloadFitness
//===----------------------------------------------------------------------===//

bool OverloadFitness::isBetter(const OverloadFitness &other) const {
  // Neither operand should be invalid or have constraint issues (they should be
  // filtered out before comparison).
  assert(getValidity() >= Validity::kFunctionConstraintInconclusive &&
         other.getValidity() >= Validity::kFunctionConstraintInconclusive);

  // We first compare the number of implicit conversions.
  size_t numConversions = getNumImplicitConversions();
  size_t otherNumConversions = other.getNumImplicitConversions();
  if (numConversions != otherNumConversions)
    return numConversions < otherNumConversions;

  // We consider exact matches of concrete types to be more specific than
  // those needing nonmaterializable conversions, both of these more
  // specific than varargs matches (for example, when overloading a
  // `foo(Int)` and `foo(Int*)` we should pick the former if both work).
  if (payload.passesVarArgArgument != other.payload.passesVarArgArgument)
    return payload.passesVarArgArgument < other.payload.passesVarArgArgument;

  // Otherwise these candidates are almost identical, so we try to decide based
  // on the number of input conventions mismatches (e.g. register-passable
  // passed in memory).
  if (payload.numMismatchedConventions !=
      other.payload.numMismatchedConventions)
    return payload.numMismatchedConventions <
           other.payload.numMismatchedConventions;

  // If still ambiguous, we compare the number of bindings. This allows us to
  // treat a function with more parameters as worse than a parameter-less
  // function, so things like:
  //    def foo(a: Int):   and  def foo[T: AnyType](a: T):
  // resolve to the more specific function.
  return paramBindings.size() < other.paramBindings.size();
}

OverloadFitness OverloadFitness::evaluate(ASTDecl *candidate,
                                          const OverloadSet &callable,
                                          PValue selfPValue) {
  DeclResolver &resolver = *callable.getShared().declResolver;
  FnTypeGeneratorType signature = candidate->getDeclFullSignature();

  if (auto witness = candidate->getIfWitness(); witness && selfPValue) {
    // TODO(MOCO-1259): Support static methods with associated aliases
    signature = substituteTraitAliasesIntoSignature(
        resolver, witness->traitSymbol, signature, selfPValue);
  }

  ParamInf inference(callable.paramBindings, signature.getInputParamTypes(),
                     signature.getParamListAttrs(),
                     /*allowImplicitConversions=*/true, candidate,
                     /*discardError=*/true,
                     /*deferredTypingContext=*/nullptr,
                     callable.additionalAssumptions);
  // Don't yield constraint failure for a single overload failure: we want a
  // better error message diagnosed for the entire set.
  VerifiedParamBindings bindings = inference.inferForStruct();

  if (!bindings) {
    // Failed inference must have emitted a diagnostic. Return it as an invalid
    // fitness.
    assert(inference.diag.hasErrorEmitted());
    return std::move(*inference.diag.takeMojoDiag());
  }

  OverloadFitness fitness(
      std::move(bindings),
      /*noArgsNeedingOrigins*/ OperandsNeedingOriginsList());
  fitness.unprovableConstraints =
      std::move(inference.bodyUnprovableConstraints);
  return fitness;
}

/// Extract the closure name from a self operand. This handles two cases:
/// (1) Closure parameters: the type is a ParamType wrapping a ParamDeclRefAttr,
///     and the name is the parameter name (e.g. "C").
/// (2) Closure instances: the value is defined by a VarDeclOp (a nested
///     function materialized as a closure), and the name is the variable name
///     (e.g. "kernel").
static StringAttr closureNameFromSelfOperand(SharedState &shared,
                                             CValue selfCValue) {
  auto paramType = sugarDynCast<ParamType>(selfCValue.getRValueType().mlirType);
  if (paramType) {
    auto paramRef = dyn_cast<ParamDeclRefAttr>(paramType.getParam());
    if (paramRef) {
      if (ClosureEmitter::isClosureType(shared, paramRef.getType()))
        return paramRef.getName();
    }
  }
  Value mlirValue = selfCValue.getMlirValue();
  if (mlirValue) {
    if (auto varDecl = mlirValue.getDefiningOp<VarDeclOp>()) {
      if (ClosureEmitter::isClosureType(shared, varDecl.getType()))
        return varDecl.getNameAttr();
    }
  }
  return {};
}

static ArrayRef<ClosureParamCapture>
closureParamCapturesIfClosure(ASTDecl *funcIfDirect,
                              const CallOperands &operands,
                              const OverloadSet &callable) {
  if (!funcIfDirect)
    return {};

  StringRef declNames = [=]() -> StringRef {
    if (auto witness = funcIfDirect->getIfWitness())
      return witness->getWitnessEntry().witnessName;
    return cast<FnOp>(funcIfDirect->getIfOperation()).getDeclName();
  }();
  if (!declNames.starts_with("__call__"))
    return {};

  if (operands.empty())
    return {};
  auto selfCValue = operands[0].ir.getIfCValue();
  if (!selfCValue)
    return {};
  StringAttr closureName =
      closureNameFromSelfOperand(funcIfDirect->getShared(), selfCValue);
  if (!closureName)
    return {};

  // Look up the captures registered for the closure. The captures live on the
  // operation that owns the closure definition:
  //  - closure instances and closure-typed function parameters register on the
  //    enclosing function (the closure is called within that function's body);
  //  - closure-typed struct parameters/fields register on the struct (the
  //    closure is callable from any of the struct's methods).
  Value mlirValue = selfCValue.getMlirValue();
  if (!mlirValue)
    return {};

  SharedState &shared = funcIfDirect->getShared();
  return shared.lookupClosureCaptureFromOp(
      mlirValue.getParentBlock()->getParentOp(), closureName);
}

/// Determine whether the specified signature can be invoked with the
/// parameter bindings specified in `callable` and the arguments specified in
/// `callOperands`.
///
/// The 'funcIfDirect' member is set if this is a direct call, or null if
/// indirect.  It can be used to tune diagnostics.
OverloadFitness OverloadFitness::evaluate(FnTypeGeneratorType signature,
                                          ASTDecl *funcIfDirect,
                                          const OverloadSet &callable,
                                          const CallOperands &operands,
                                          bool allowImplicitConversions) {
  SMLoc callLoc = callable.getExpr()->getLoc();
  SharedState &shared = callable.getShared();

  // An implicit conversion can only be performed by a constructor marked
  // @implicit, so reject other candidates before running parameter inference.
  // Types with large __init__ overload sets (e.g. SIMD) make the inference
  // cost significant when converting many values, as in collection literals.
  if (funcIfDirect && callable.syntax == CallSyntax::kImplicitConvert &&
      funcIfDirect->getDeclImplicitConversionKind() ==
          ImplicitConversionKind::None) {
    // Name the target with the concrete Self type rather than the signature's
    // result type, which is still unsubstituted this early.
    assert(callable.selfResultType &&
           "implicit conversion without a self result type?");
    MojoInflightDiag diag = shared.emitError(callLoc);
    diag << "cannot implicitly convert";
    if (ASTType fromType = operands[0].ir.getRValueTypeIfResolvable())
      diag << " " << fromType;
    diag << " to " << callable.selfResultType << ": add an explicit cast";
    return std::move(diag);
  }

  if (!operands.empty() && funcIfDirect) {
    if (auto selfCValue = operands[0].ir.getIfCValue()) {
      if (auto selfPValue = PValue(selfCValue.getRValueType().mlirType)) {
        // TODO(MOCO-1259): Support static methods with associated aliases
        if (WitnessDecl *witness = funcIfDirect->getIfWitness();
            witness &&
            isa<FnTypeGeneratorType>(witness->getWitnessEntry().witnessType)) {
          signature = substituteTraitAliasesIntoSignature(
              *shared.declResolver, witness->traitSymbol, signature,
              selfPValue);
        }
      }
    }
  }

  // If a variadic keyword arg is expected, we collect the unknown kw operands.
  PogListAttr argListAttr = signature.getArgListAttrs();
  CallOperands::PogAssignment pogAssignment;
  std::optional<MojoInflightDiag> stagedDiag;
  auto getStagedDiag = [&](SMLoc loc) -> MojoInflightDiag & {
    stagedDiag = shared.emitError(loc);
    return *stagedDiag;
  };

  if (failed(operands.assignToPogs(argListAttr, /*isParameterList=*/false,
                                   pogAssignment, getStagedDiag)))
    return std::move(*stagedDiag);

  // Check that the signature can be rebound with this set of bindings.
  OperandsNeedingOriginsList operandsNeedingOrigins;
  CallParamInf inference(
      callable.paramBindings, signature.getInputParamTypes(),
      signature.getParamListAttrs(), allowImplicitConversions, funcIfDirect,
      /*discardError=*/false, signature, operands, pogAssignment,
      operandsNeedingOrigins, callable.additionalAssumptions);
  // Check if we're calling a closure's __call__ method and need to set
  // captured closure parameters. Only applies to method call syntax on a
  // __call__ method — not direct calls that happen to pass a closure as an
  // argument.
  ArrayRef<ClosureParamCapture> implicitParams =
      closureParamCapturesIfClosure(funcIfDirect, operands, callable);
  if (!implicitParams.empty()) {
    // Capture slots are at the front of the method's own parameter list
    // (Inferred PassingKind), but getFullSignature() prepends contextual
    // params (e.g. the trait's _Self). Skip past them.
    //
    // TODO: we won't be needing this after migrating to parametric trait.
    size_t contextualParams = [&]() -> size_t {
      if (funcIfDirect->getIfWitness())
        return 1; // _Self param for trait witness
      auto fnOp = cast<FnOp>(funcIfDirect->getIfOperation());
      return signature.getInputParamTypes().size() -
             fnOp.getFuncTypeGenerator().getInputParamTypes().size();
    }();
    const CallOperands &paramBindings = callable.paramBindings.getParameters();
    size_t paramIdx = contextualParams;
    for (const auto &[paramName, paramType] : implicitParams) {
      // These capture slots are inferred parameters, but a user is allowed to
      // bind them explicitly by keyword. Only seed the captured value when the
      // user hasn't already provided it
      StringAttr slotName = signature.getParamName(paramIdx);
      if (!slotName || !paramBindings.findKwArg(slotName)) {
        TypedAttr paramValue = ParamDeclRefAttr::get(paramName, paramType);
        inference.setInitialInferredValue(paramIdx, paramValue);
      }
      ++paramIdx;
    }
  }

  VerifiedParamBindings verifiedBindings = inference.inferForCall();
  if (!verifiedBindings) {
    // Failed inference must have emitted a diagnostic.
    assert(inference.diag.hasErrorEmitted());
    return std::move(*inference.diag.takeMojoDiag());
  }

  // If anything was bound, apply it to the signature so the expected argument
  // types are updated. The bindings produced by inference have already been
  // rebound by `setInferredValue` against `signature.getInputParamTypes()`, so
  // there is no need for an additional rebinding loop here.
  signature =
      cast<GeneratorType>(verifiedBindings.specializeGeneratorType(signature));

  // This is the result we will return if we succeed.
  OverloadFitness result(std::move(verifiedBindings),
                         std::move(operandsNeedingOrigins));
  result.unprovableConstraints = std::move(inference.bodyUnprovableConstraints);

  // Get the overload ranking metrics.
  result.payload.numImplicitConversions = inference.numImplicitConversions;
  result.payload.numMismatchedConventions = inference.numMismatchedConventions;
  result.payload.passesVarArgArgument = inference.passesVarArgArgument;

  // Check that the result didn't bind to a type that would require changing to
  // a different result convention.
  for (Type outputType : signature.getResults())
    if (!ASTType(outputType).isRegisterPassable(callLoc, shared)) {
      return shared.emitError(callLoc)
             << "result cannot bind __TypeOfAllTypes type to memory-only type "
             << ASTType(outputType);
    }

  // Fail if this is a constructor call that returns the wrong result type. This
  // can happen with weird things like this:
  //     struct A[X: Int]: def __init__(out self: A[4]): pass
  //     var a = A[1]()  # Infers to A[4]; error!
  //     var b = A[4]()  # Ok!
  if (callable.syntax == CallSyntax::kTypeCall) {
    // Check to see if any of the parameter bound to the result type disagree
    // with the 'Self' parameters, which are bound into the verified bindings.
    auto resultType = ASTType(signature.getUserResultType());
    auto numBindings = resultType.getParamBindings().size();
    for (auto [paramIdx, actual, expected] : llvm::enumerate(
             resultType.getParamBindings(),
             result.getParamBindings().getValues().take_front(numBindings))) {
      if (!isEqualCanon(actual, expected)) {
        DeclResolver::DiagnosticDeclContextChanger x(funcIfDirect);
        assert(resultType.getDecl(shared) &&
               "result type should have an associated decl");
        assert(resultType.getDecl(shared)->getIfOperation() &&
               "result type decl should have an operation");
        auto declOp =
            cast<StructDeclOp>(resultType.getDecl(shared)->getIfOperation());
        auto paramType = declOp.getSignature().getInputParamTypes()[paramIdx];
        MojoInflightDiag resultTypeDiag = shared.emitError(callLoc);
        resultTypeDiag << "return type " << resultType << " parameter "
                       << ParamDeclRefAttr::get(
                              declOp.getParams()[paramIdx].getName(), paramType)
                       << " has value " << actual
                       << " that doesn't match expected " << expected;
        return std::move(resultTypeDiag);
      }
    }
  }

  // Otherwise we succeeded!
  return result;
}
