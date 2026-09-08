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
/// The IREmitter class is the main driver for expression emission, providing
/// helper functions used by the individual node emission hooks.
//
//===----------------------------------------------------------------------===//

#include "IREmitter.h"
#include "ExprNodes.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParamInf.h"
#include "ParserEvaluationContext.h"
#include "Traits.h"

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/POPDialect/POPOps.h"

#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// Store-Time Type Refinement
//===----------------------------------------------------------------------===//

/// Applies store-time type refinement to a finalized VarDeclOp.
///
/// Why this exists:
/// - Declaration-time inference cannot always refine safely, because the final
///   reference shape/origin is known only after store emission.
/// - Some pattern bindings produce nested refs (`!lit.ref<!lit.ref<T>>`), so
///   refinement must load then rebind to avoid refining a placeholder form.
/// - We must update the matching ASTDecl entry (using operation identity) so
///   shadowed names resolve to the correct refined value in later lookups.
/// - Preserves value kind semantics (MLValue for `var`, MBValue otherwise).
///
/// If refinement does not change the element type, this is a no-op.
static void maybeApplyTypeRefinement(VarDeclOp varOp, ASTDecl &declScope,
                                     OpBuilder &builder) {
  // 1. Extract element type from the VarDecl.
  RefType refType = cast<RefType>(varOp.getType());
  Type elementType = refType.getElementType();

  // For bind/ref: element is !lit.ref<T, origin>, extract inner element.
  RefType innerRef;
  if ((innerRef = dyn_cast<RefType>(elementType)))
    elementType = innerRef.getElementType();

  // 2. Check if refinement applies.
  Type refinedType = maybeRefineTypeWithAssumptions(elementType, declScope);
  if (refinedType == elementType)
    return; // No refinement needed.

  if (varOp.getKind() == VarDeclKind::Synthesized)
    return;

  // 3. Look up the ASTDecl registered for this VarDeclOp. With variable
  // shadowing, multiple decls can share a name in the same scope, so
  // disambiguate by matching on the defining Operation*.
  ASTDecl *varASTDecl = nullptr;
  for (ASTDecl *decl : declScope.lookupInCurrentScope(varOp.getNameAttr())) {
    if (decl->getIfOperation() == varOp.getOperation()) {
      varASTDecl = decl;
      break;
    }
  }
  assert(varASTDecl &&
         "non-synthetic VarDeclOp should have a matching ASTDecl");

  // Use the VarDecl's own location which has proper debug scope.
  Location mlirLoc = varOp.getLoc();

  if (innerRef) {
    // Double-ref case (bind/ref with ref-typed tuple elements):
    // !lit.ref<!lit.ref<T, inner>, outer>
    //
    // Load the VarDecl to collapse to a single ref, then rebind. This mirrors
    // what trait_downcast does naturally as a function call — the load peels
    // the outer ref, and the rebind refines the element type.
    //
    // load: !lit.ref<!lit.ref<T, inner>, outer> -> !lit.ref<T, inner>
    // rebind: !lit.ref<T, inner> -> !lit.ref<T(Trait), inner>
    RefType refinedInner = innerRef.getWithElement(refinedType);
    Value loaded = RefLoadOp::create(builder, mlirLoc, varOp.getResult());
    Value reboundValue =
        RebindOp::create(builder, mlirLoc, refinedInner, loaded);
    varASTDecl->setIRValue(MBValue(reboundValue));
  } else {
    // Single-ref case (var patterns, function args):
    // !lit.ref<T> -> !lit.ref<T(Trait)>
    //
    // Classify the rebound reference using canonical mutability-based
    // classification. Keep Bound declarations as MBValue for consistency with
    // DeclRef lookup behavior.
    RefType refinedRefType = refType.getWithElement(refinedType);
    Value reboundValue =
        RebindOp::create(builder, mlirLoc, refinedRefType, varOp.getResult());
    CValue refinedCValue = CValue::getMValueForRef(reboundValue);
    if (varOp.getKind() == VarDeclKind::Bound)
      refinedCValue = MBValue(reboundValue);
    varASTDecl->setIRValue(refinedCValue);
  }
}

//===----------------------------------------------------------------------===//
// IREmitter
//===----------------------------------------------------------------------===//

/// Create an IREmitter for a dynamic context with a builder.
IREmitter::IREmitter(ASTDecl &declScope, OpBuilder builder)
    : SharedStateUser(declScope.getShared()), builder(builder),
      paramContext(EC_InvalidContext), declScope(declScope) {}

/// Create an IREmitter for a parameter context.
IREmitter::IREmitter(ASTDecl &declScope, ExprContext paramContext,
                     DeferredTypingContext *deferredTypingContext)
    : SharedStateUser(declScope.getShared()), builder({}),
      paramContext(paramContext), declScope(declScope),
      deferredTypingContext(deferredTypingContext) {}

/// Emit an error about use of a dynamic value (the expression) in a context
/// that only allows parameter expressions.  This always returns a null
/// PValue.
PValue IREmitter::emitErrorForDynamicValueInParameter(const ExprNode *expr,
                                                      const char *message) {
  assert(paramContext != EC_InvalidContext &&
         "parameter context not set correctly");
  if (!message)
    message = "cannot use a dynamic value";
  emitError(expr->getLoc(), message)
      << getContextMessage(paramContext) << expr->getRange();
  return {};
}

PValue
IREmitter::emitErrorForDynamicValueInParameter(llvm::SMLoc loc,
                                               const char *customMessage) {
  return emitErrorForDynamicValueInParameter(shared.translateLocation(loc),
                                             customMessage);
}

/// Emit an error about use of a dynamic value (the expression) in a context
/// that only allows parameter expressions.  This always returns a null
/// PValue.
PValue IREmitter::emitErrorForDynamicValueInParameter(Location loc,
                                                      const char *message) {
  assert(paramContext != EC_InvalidContext &&
         "parameter context not set correctly");
  if (!message)
    message = "cannot use a dynamic value";
  emitError(loc, message) << getContextMessage(paramContext);
  return {};
}

//===----------------------------------------------------------------------===//
// Emission helpers for various value classifications.

CValue IREmitter::emitRValue(ASTExprAnd<AnyValue> value, ExprDest &dest) {
  if (!value) // Already diagnosed error.
    return {};

  // If the value is still unresolved, materialize it.
  CValue cValue = value.ir.getIfCValue();
  if (!cValue) {
    cValue = emitCValue(value, dest);
    if (!cValue)
      return {};
  }

  // If this is already an RValue/PValue then we are done.
  if (auto rvRep = cValue.getIfRValue())
    return emitCResult(rvRep, value.expr, dest);

  // If the value dest expects a different result type than the lvalue or bvalue
  // that we have, then we'll need to do a conversion, and that conversion will
  // return an rvalue. Use it first which may avoid a copy of a value.
  if (auto knownDestType = dest.getExpectedTypeIfSpecified()) {
    if (!cValue.getRValueType().isEqualCanon(knownDestType)) {
      return emitConstructorCall(
          knownDestType, CallOperands(CallSyntax::kImplicitConvert, value.expr,
                                      std::move(dest), value));
    }
  }

  // Otherwise, this is an LValue or BValue, emit a copy.
  return emitCopyOfValue({cValue, value.expr}, dest);
}

RValue IREmitter::emitRValue(ASTExprAnd<AnyValue> value, ExprContext context,
                             ASTType resultType) {
  ExprDest dest(resultType, context);
  CValue result = emitRValue(value, dest);
  while (true) {
    if (!result) {
      dest.resetForError(*this);
      return {};
    }
    // Typically emitRValue will return an RValue, but it might return a BValue.
    if (auto rv = result.getIfRValue())
      return rv;

    // It may return a BValue though (e.g. when accessing subfields with
    // computed lvalue bases), in which case we'll emit a copy of it.
    ExprDest copyDest(context);
    result = emitCopyOfValue({result, value.expr}, copyDest);
  }
}

CValue IREmitter::emitCValue(ASTExprAnd<AnyValue> value, ExprContext context,
                             ASTType resultType) {
  ExprDest dest(resultType, context);
  if (auto c = emitCValue(value, dest))
    return c;
  dest.resetForError(*this);
  return {};
}

CValue IREmitter::emitCValue(ASTExprAnd<AnyValue> value, ExprDest &dest) {
  if (!value) // Already diagnosed error.
    return {};
  // If this is already an CValue, then we are done.
  if (auto cRep = value.ir.getIfCValue()) {
    if (!dest.isSpecified())
      return cRep;
    auto result = emitResult(value.ir, value.expr, dest);
    assert(!result || result.getIfCValue());
    return result.getIfCValue();
  }

  // If the value being materialized is an unresolved overload set, try to
  // materialize it.
  if (OverloadSetUValue overloads = value.ir.getIfOverloadSet()) {
    assert(overloads && "unknown overloaded value");
    return overloads->emitAsCValue(*this, dest);
  }

  if (auto initValue = value.ir.getIfInitializer())
    return initValue->emitAsCValue(*this, dest);

  if (auto inferredAttr = value.ir.getIfInferredBaseAttrRef())
    return inferredAttr.emitAsCValue(*this, dest);

  llvm_unreachable("unknown UValue in emitCValue");
}

/// Emit an expression providing an immutable borrowed reference to a value.
BValue IREmitter::emitBValue(ASTExprAnd<AnyValue> value, ExprDest &dest) {
  if (!value)
    return {};

  // Handle dynamic LValues by loading from them.
  if (auto dlv = value.ir.getIfDLValue()) {
    value.ir = dlv->emitLoad(dest, *this);
    if (!value.ir)
      return {};
  }

  // If the value being materialized is an unresolved overload set, try to
  // materialize it.
  if (value.ir.getIfUValue()) {
    value.ir = emitCValue(value, dest);
    if (!value.ir)
      return {};
  }

  // If there is a value destination, resolve it into an RValue or BValue.
  if (dest.isSpecified()) {
    value.ir = emitResult(value.ir, value.expr, dest);
    // Emitting the result to the dest could promote back to RValue, so re-emit
    // it with a now-empty (assigned from context) destination.
    return emitBValue(value, dest);
  }

  // Handle M*Value's by decaying to MBValue.
  if (value.ir.isMValue()) {
    // Maintain parametric mutability and comptime if we have it.
    if (auto mbp = value.ir.getIfMBPValue())
      return mbp;
    if (auto pmb = value.ir.getIfPMBValue())
      return pmb;
    // Otherwise decay MLValue/MRValue to MBValue.
    return MBValue(value.ir.getMValueReference());
  }

  // Decay SRValue's into SBValue or MBValue.
  if (auto srVal = value.ir.getIfSRValue()) { // Decay => SBValue/MRValue
    if (ASTType(srVal.getType()).isTrivial(value.expr->getLoc(), shared))
      return SBValue(srVal);
    // If this is a nontrivial value, we need to create an MRValue (and decay
    // that) so we can track its lifetime correctly.
    auto mrVal = emitMRValue(value, dest.getContext());
    if (!mrVal)
      return {};
    return MBValue(mrVal);
  }

  // Finally, we know we have a BValue.
  auto resultBV = value.ir.getIfBValue();
  assert(resultBV && "unknown value kind");
  return resultBV;
}

BValue IREmitter::emitBValue(ASTExprAnd<AnyValue> value, ExprContext context,
                             ASTType resultType) {
  ExprDest dest(resultType, context);
  if (auto result = emitBValue(value, dest))
    return result;
  dest.resetForError(*this);
  return {};
}

LValue IREmitter::emitLValue(ASTExprAnd<AnyValue> value, ExprDest &dest) {
  if (!value) {
    dest.resetForError(*this);
    return {};
  }

  if (LValue lValue = value.ir.getIfLValue()) {
    if (!dest.isSpecified())
      return lValue;
    auto result = emitResult(value.ir, value.expr, dest);
    assert(!result || result.getIfLValue());
    return result.getIfLValue();
  }

  emitError(value.expr->getLoc())
      << "expression must be mutable" << getContextMessage(dest.context)
      << value.expr->getRange();
  dest.resetForError(*this);
  return {};
}

/// This verifies that the specified PValue can be materialized to a runtime
/// value, emits an error if it cannot.
static LogicalResult emitErrorIfUnmaterializableValue(IREmitter &emitter,
                                                      ASTExprAnd<PValue> value,
                                                      ExprContext context) {
  TypedAttr attr = value.ir.get();
  // We cannot emit types as values yet.
  if (LIT::isTypeExpr(attr) && !isa<ModuleAttr>(attr)) {
    const ExprNode *expr = value.expr;
    MojoInflightDiag diag = emitter.emitError(
        expr->getLoc(), "dynamic type values not permitted yet");
    if (context == EC_VarInit)
      diag << "; try creating a 'comptime' instead of a 'var'";
    else if (context == EC_CallArgValue)
      diag << "; try passing types as a parameters instead of arguments";
    diag << expr->getRange();
    return failure();
  }

  // We cannot emit a value that contains an origin in its type (e.g. a
  // StringSpan or UnsafePointer) because the origin will be incorrect -
  // referring to immortal compile-time memory.
  if (ASTType(attr.getType()).containsUnmaterializableOrigins(emitter.shared)) {
    const ExprNode *expr = value.expr;
    auto diag = emitter.emitError(
        expr->getLoc(), "cannot materialize compile-time value of type ");
    diag << ASTType(attr.getType()) << " to a runtime value"
         << expr->getRange();
    diag.attachNote(expr->getLoc())
        << "the type contains an origin referring to a compile-time value";
    return failure();
  }

  return success();
}

SRValue IREmitter::emitPValueToSRValue(ASTExprAnd<PValue> value,
                                       ExprContext context) {
  TypedAttr attr = value.ir.get();
  const ExprNode *expr = value.expr;

  // If this is a parameter, we need to materialize it, either as an
  // index.constant or as a parameter expression.
  if (!builder) {
    emitErrorForDynamicValueInParameter(expr);
    return {};
  }

  // Diagnose issues about types that cannot be comptime -> runtime
  // materialized.
  if (failed(emitErrorIfUnmaterializableValue(*this, value, context)))
    return {};

  Location location = expr->getLocation(*this);

  // If the value being materialized is itself parameterized, then we cannot
  // materialize it as an SSA value - there will be no way to bind parameters to
  // it.
  // TODO: We should have a general predicate from this provided by the KGEN
  // parameter utilities.
  if (auto signature = dyn_cast<FnTypeGeneratorType>(attr.getType())) {
    // Reject any unbound non-singleton parameters. This can't be materialized
    // to a runtime value, because it needs to get instantiated.
    for (auto [paramType, pog] :
         llvm::zip(signature.getInputParamTypes(),
                   signature.getParamListAttrs().getPogs())) {
      // Singleton values like origins are fine. They will be removed by
      // lowerlit before code generation.
      if (ASTType(paramType).isSingleton(shared))
        continue;

      auto diag =
          emitError(expr->getLoc(),
                    "cannot use parametric function as a runtime closure")
          << expr->getRange();
      diag.attachNote(expr->getLoc())
          << "parameter " << ParamDeclRefAttr::get(pog.getName(), paramType)
          << " of type " << ASTType(paramType) << " is not bound";
      return {};
    }

    // Materialize signatures as closures.
    if (signature.isCapturing()) {
      emitError(
          expr->getLoc(),
          "TODO: capturing closures cannot be materialized as runtime values");
      return {};
    }
    return SRValue(CreateClosureOp::create(*builder, location, signature, attr,
                                           ValueRange()));
  }

  ASTType valueType = value.ir.getRValueType();

  // If the type is trivial, materialize using param.constant.
  if (valueType.isTrivial(value.expr->getLoc(), shared))
    return SRValue(ParamConstantOp::create(*builder, location, value.ir));

  // If the type is implicitly copyable, it should be cheap to be implicitly
  // materialized as well.
  //
  // NOTE: we need to leave a backdoor to allow implicit materialization for
  // default argument. This is because we parse default argument value
  // into PValue at the moment, meaning that to emit the value for default
  // argument, we will have to materialize it first. Using `EC_context` to tell
  // whether we are generating default arg value is not a typical usage of
  // `EC_context`, but it is much cleaner/simpler than passing a flag all the
  // way down from `emitPreemittedArgumentAsDynamicValue`.
  bool isDefaultArg = (context == EC_CallArgDefaultValue);
  if (isDefaultArg ||
      valueType.isImplicitlyCopyable(value.expr->getLoc(), shared, declScope))
    return SRValue(ParamMaterializeOp::create(*builder, location, value.ir));

  if (isa<ModuleType>(valueType)) {
    emitError(expr->getLoc(), "cannot use package name ")
        << valueType << " as a runtime value" << expr->getRange();
    return {};
  }

  auto diag =
      emitError(expr->getLoc(), "cannot materialize comptime value of type ")
      << valueType << " to runtime because it is not 'ImplicitlyCopyable'"
      << expr->getRange();

  // Attach the fix it by wrapping materialize[]() around the expression.
  diag.attachNote(expr->getLoc())
      << "use 'materialize' to explicitly materialize the value"
      << FixIt::insertBeforeToken(expr->getRangeStart(), "materialize[")
      << FixIt::insertAfterToken(expr->getRangeEnd(), "]()", shared.diags);
  return {};
}

SRValue IREmitter::emitSRValue(ASTExprAnd<AnyValue> anyValue,
                               ExprContext context, ASTType resultType) {
  const ExprNode *expr = anyValue.expr;

  // Emit using resultType if present, and eliminate LValue/OverloadSetUValue's.
  RValue value = emitRValue(anyValue, context, resultType);
  if (!value)
    return {};

  if (!value.getRValueType().isRegisterPassable(expr->getLoc(), shared)) {
    emitError(expr->getLoc()) << "cannot load non-register passable type into "
                                 "SSA register (compiler bug, please report!)";
    return {};
  }

  // If we have a value in memory, use a LoadConsumeOp to load it.
  if (auto mrValue = value.getIfMRValue()) {
    if (!builder) {
      emitErrorForDynamicValueInParameter(expr);
      return {};
    }
    Value result =
        LoadConsumeOp::create(*builder, expr->getLocation(*this), mrValue);
    return SRValue(result);
  }

  // If this is already an SRValue, return it.
  if (auto rvalue = value.getIfSRValue())
    return rvalue;

  auto pValue = value.getIfPValue();
  assert(pValue && "must be PValue if register-passable and not SRValue");
  return emitPValueToSRValue({pValue, expr}, context);
}

MRValue IREmitter::emitMRValue(ASTExprAnd<AnyValue> value, ExprContext context,
                               ASTType resultType) {
  auto rVal = emitRValue(value, context, resultType);
  if (!rVal)
    return {};

  if (auto mr = rVal.getIfMRValue())
    return mr;

  // Promote SRValue/PValue to MRValue.
  if (rVal.isSValue() || rVal.getIfPValue()) {
    Location argLoc = value.expr->getLocation(*this);
    VarDeclOp varOp = emitVarDecl("anonymous*", rVal.getRValueType(), argLoc,
                                  VarDeclKind::Synthesized);
    if (!varOp)
      return {};
    ExprDest dest(MLValue(varOp), context);
    if (!emitRValue({rVal, value.expr}, dest))
      dest.resetForError(*this);
    return MRValue(varOp);
  }

  llvm_unreachable("unknown RValue");
}

/// This helper emits the specified value as an MBValue which has
/// memory-only representation, materializing PValues as needed. This
/// returns null if emission fails.
MBValue IREmitter::emitMBValue(ASTExprAnd<AnyValue> value, ExprContext context,
                               ASTType resultType) {
  BValue bValue = emitBValue(value, context, resultType);
  if (!bValue)
    return {};

  if (auto mb = bValue.getIfMBValue())
    return mb;

  // Drop parametric mutability.
  if (auto mbp = bValue.getIfMBPValue())
    return MBValue(mbp);

  // Mojo can't turn an SBValue into an MBValue - the former only occurs in
  // special places, and cannot be lifetime tracked back to the original RValue
  // it was derived from.  If this assert fires, something is wrong up-stack of
  // this code.
  assert((!bValue.getIfSBValue() || bValue.getRValueType().isRegisterPassable(
                                        value.expr->getLoc(), shared)) &&
         "Cannot convert an SBValue to an MBValue");

  // PValue's and SValues need to be emitted into an owned memory temporary,
  // which we can then decay to an MBValue.
  assert(bValue.getIfPValue() || bValue.isSValue());
  auto mrVal = emitMRValue({bValue, value.expr}, context);
  if (!mrVal)
    return {};
  return MBValue(mrVal);
}

PValue IREmitter::emitPValue(ASTExprAnd<AnyValue> value, ExprContext context,
                             ASTType resultType) {
  if (!value)
    return {};

  // Clear the builder to indicate that an PValue must be emitted.
  llvm::SaveAndRestore savedBuilder(builder, {});
  llvm::SaveAndRestore savedContext(paramContext, context);

  // If there is a result type, coerce before checking for PValue.
  if (resultType) {
    value.ir = emitRValue(value, context, resultType);
    if (!value)
      return {};
  }

  // Resolve any unresolved values using the result type.
  value.ir = emitCValue(value, context, resultType);

  // If this is a DLValue, see if it can be emitted as a PValue. PValues are
  // immutable, so try to load the DLValue in a parameter context.
  if (auto dl = value.ir.getIfDLValue()) {
    ExprDest dest(context);
    value.ir = dl->emitLoad(dest, *this);
    if (!value.ir) {
      dest.resetForError(*this);
      return {};
    }
  }

  // If this is a parameter, return it.
  if (auto result = value.ir.getIfPValue())
    return result;

  // Otherwise diagnose this as "not a parameter" unless the value failed to
  // emit entirely.
  if (value.ir)
    emitErrorForDynamicValueInParameter(value.expr);
  return {};
}

/// This helper emits the specified expression as a 'ref' expression value,
/// and returns the value of RefType for the result.
/// This emits an error and returns null if emission fails.
Value IREmitter::emitRefValue(ASTExprAnd<AnyValue> value, ExprContext context) {
  // A DLValue can be a ref when its "load" operation returns a ref.  This can
  // happen when a computed getter returns a ref - e.g. for Dict.
  if (auto dlValue = value.ir.getIfDLValue()) {
    ExprDest dest(context);
    value.ir = dlValue->emitLoad(dest, *this);
    if (!value.ir) {
      dest.resetForError(*this);
      return {};
    }
  }

  // If this got resolved to an MValue then we're done.
  if (value.ir.isMValue())
    return value.ir.getMValueReference();

  // Otherwise we can't support other non-MValue's like borrowed registers or
  // RValue's.
  auto diag = emitError(value.expr->getLoc(), "value");
  if (auto cv = value.ir.getIfCValue())
    diag << " of type " << cv.getRValueType();
  diag << " doesn't have a memory origin" << getContextMessage(context)
       << value.expr->getRange();
  return {};
}

//===----------------------------------------------------------------------===//
// Emission helpers for various value classifications.

/// If the type of the specified value differs from the destination type, emit
/// a rebind operation to convert it.
Value IREmitter::emitRebindOpIfNeeded(Value value, Type destType, SMLoc loc) {
  if (!value || value.getType() == destType)
    return value;

  // Sanity check that rebind isn't *introducing* reference mutability.
  if (auto srcRefType = dyn_cast<RefType>(value.getType()))
    if (auto dstRefType = dyn_cast<RefType>(destType)) {
      assert(!(srcRefType.isMutableKnown(false) &&
               dstRefType.isMutableKnown(true)) &&
             "Rebind is introducing mutability");
      assert(isEqualCanon(srcRefType.getAddressSpace(),
                          dstRefType.getAddressSpace()) &&
             "rebind cannot change address space");

      // If we are casting away mutability, use RefImmutOp.
      if (!srcRefType.isMutableKnown(false) &&
          dstRefType.isMutableKnown(false)) {
        value = RefImmutOp::create(*builder, translateLocation(loc), value);
        return emitRebindOpIfNeeded(value, destType, loc);
      }

      // Origin-only widening (subtree, union, etc.): use RefUpcastOp.  Don't
      // use it for bitcasts that remove sugar from the element type though.
      if (!isEqualCanon(srcRefType.getOrigin(), dstRefType.getOrigin())) {
        value = RefUpcastOp::create(
            *builder, translateLocation(loc),
            srcRefType.getWithOrigin(dstRefType.getOrigin()), value);
        return emitRebindOpIfNeeded(value, destType, loc);
      }
    }

  return RebindOp::create(*builder, translateLocation(loc), destType, value);
}

/// If needed, convert the specified value to the target destination type,
/// with a noop cast.  This is used to adjust inconsequential details of the
/// type or for simple things like upcasts.  This does not invoke constructors
/// or do other non-trivial conversions.
///
/// This produces an error and returns null on an invalid conversion.
CValue IREmitter::rebindValue(ASTExprAnd<CValue> value, Type destType) {
  // Materialize a parameter rebind.
  if (auto pvalue = value.ir.getIfPValue())
    return ParamOperatorAttr::getRebind(pvalue.get(), destType);
  if (auto dlValue = value.ir.getIfDLValue()) {
    dlValue->elementType = destType;
    return dlValue;
  }

  // Cannot perform value rebind if only parameters are allowed.
  if (!builder)
    return emitErrorForDynamicValueInParameter(value.expr);

  // Materialize a rebind operation.
  auto loc = value.expr->getLoc();
  if (auto refValue = value.ir.getIfMLValue())
    return MLValue(emitRebindOpIfNeeded(refValue, destType, loc));
  if (auto refValue = value.ir.getIfMRValue())
    return MRValue(emitRebindOpIfNeeded(refValue, destType, loc));
  if (auto refValue = value.ir.getIfMBValue())
    return MBValue(emitRebindOpIfNeeded(refValue, destType, loc));
  if (auto refValue = value.ir.getIfMBPValue())
    return MBPValue(emitRebindOpIfNeeded(refValue, destType, loc));
  if (auto sbValue = value.ir.getIfSBValue())
    return SBValue(emitRebindOpIfNeeded(sbValue, destType, loc));

  auto srValue = value.ir.getIfSRValue();
  assert(srValue && "Unknown value kind");
  return SRValue(emitRebindOpIfNeeded(srValue, destType, loc));
}

/// Emit the specified value into the current destination if present.  This
/// accepts (and silently propagates) null values.
///
/// Note that the `value` provided here may require an implicit conversion
/// into the destination slot, so the input may be memory-only and result be
/// register-passable (and visa-versa).
AnyValue IREmitter::emitResult(AnyValue value, const ExprNode *expr,
                               ExprDest &dest) {
  if (!value) {
    dest.resetForError(*this);
    return {};
  }
  ExprContext context = dest.getContext();

  // If no destination is specified or it is just a contextual type hint or this
  // is a parameter to be destructed, then we can propagate the value directly.
  if (!dest.isSpecified() || isa<LValueInitializerType>(dest.representation)) {
    dest.representation = NullRepresentation();
    return value;
  }

  // If the value is still unresolved, materialize it into the destination.
  auto cValue = value.getIfCValue();
  if (!cValue)
    return emitCValue({value, expr}, dest);
  value = {}; // Only use cValue below.

  // OK, if there is a destination specified, handle them by converging the set
  // of value types we have.
  auto rvType = cValue.getRValueType();

  // If there is a known type for the destination but the value disagrees, emit
  // an implicit conversion directly into the destination.  This keeps values in
  // registers and avoids a "convert + clone" pair for memory->memory
  // conversions.
  if (ASTType requiredType =
          dest.resolveImpliedType(expr->getLoc(), rvType, *this)) {
    if (!requiredType.isEqualCanon(rvType)) {
      if (requiredType.hasUnboundParameters()) {
        dest.representation =
            cast<LValueContextualType>(dest.representation).expr;
        cValue = emitConstructorCall(
            requiredType, CallOperands(CallSyntax::kImplicitConvert, expr,
                                       std::move(dest), {{cValue, expr}}));
        // The constructor call must have resolved the dest.
        assert(!dest.isSpecified());
      } else {
        cValue =
            emitImplicitConversionToType({cValue, expr}, requiredType, dest);
      }
      // If this resolved the value dest, then we're done.   This handles the
      // null result case as well.
      if (!dest.isSpecified())
        return cValue;
      assert(cValue);
    }

    // At this point the canonical types line up, but the sugar may not. Align
    // the sugar so clients don't have to deal with it.
    if (requiredType.mlirType != rvType.mlirType) {
      auto rebindType = requiredType;
      if (cValue.isMValue())
        rebindType = cValue.getMValueType().getWithElement(requiredType);
      cValue = rebindValue({cValue, expr}, rebindType);
      if (!cValue)
        return {};
    }
    rvType = cValue.getRValueType();
  }

  // If the destination is just a required type, then we now know it must agree
  // and therefore don't need to do anything more.
  if (isa<ASTType>(dest.representation)) {
    dest.representation = NullRepresentation(); // Resolved the ExprDest;
    return cValue;
  }

  // If this destination was an LValue whose buffer was already taken to be
  // filled in by a client, then this is just completing the transaction.
  if (isa<LValueBufferTaken>(dest.representation)) {
    dest.representation = NullRepresentation(); // Resolved the ExprDest;

    // The client directly filled in an LValue we provided which is great, but
    // that LValue we provided took ownership of the value, so we need to return
    // the result as a borrow, not an owned reference.
    assert(cValue.isMValue() && "Must be an MValue providing result");
    return MBValue(cValue.getMValueReference());
  }

  // We know we have an RValue/BValue and the destination is some kind of
  // LValue.  Emit the dest to figure out where to store it.
  LValue destLV = dest.getLValueForResult(expr->getLoc(), rvType,
                                          /*allowIncompatibleTypes=*/true,
                                          /*requireMLValue=*/false, *this);
  if (!destLV) {
    dest.resetForError(*this);
    return {};
  }

  // This will have completely resolved all the ExprDest possibilities.
  assert(!dest.isSpecified() || isa<LValueBufferTaken>(dest.representation));
  dest.representation = NullRepresentation(); // Resolved the ExprDest;

  // Finally, store the value into the lvalue.
  return emitStoreToLValue({cValue, expr}, destLV, context);
}

CValue IREmitter::emitCResult(CValue value, const ExprNode *expr,
                              ExprDest &dest) {
  // Emitting a CValue always produces a CValue.
  auto result = emitResult(value, expr, dest);
  assert((!result || result.getIfCValue()) &&
         "emitting a CValue as a result should always produce a CValue");
  return result.getIfCValue();
}

/// Destructuring the specific PValue against the provided target expr
/// (which specifies the pattern).
LogicalResult IREmitter::emitDestructuringPValue(PValue value,
                                                 const ExprNode *targetExpr) {
  // Clear the builder to indicate that an PValue must be emitted.
  llvm::SaveAndRestore savedBuilder(builder, {});
  return targetExpr->emitDestructuringPValue(value, *this);
}

/// Emit the specified expression into the specified destination.
AnyValue IREmitter::emitExpr(const ExprNode *expr, ExprDest &dest) {
  assert(expr && "cannot emit a null node");
  if (auto result = expr->emitIR(dest, *this))
    return result;
  dest.resetForError(*this);
  return {};
}

AnyValue IREmitter::emitExpr(const ExprNode *expr, ExprContext context,
                             ASTType resultType) {
  ExprDest dest(resultType, context);
  return emitExpr(expr, dest);
}

RValue IREmitter::emitExprRValue(const ExprNode *expr, ExprContext context,
                                 ASTType resultType) {
  return emitRValue({emitExpr(expr, context, resultType), expr}, context,
                    resultType);
}

CValue IREmitter::emitExprCValue(const ExprNode *expr, ExprContext context,
                                 ASTType resultType) {
  return emitCValue({emitExpr(expr, context, resultType), expr}, context);
}

SRValue IREmitter::emitExprSRValue(const ExprNode *expr, ExprContext context,
                                   ASTType resultType) {
  return emitSRValue({emitExpr(expr, context, resultType), expr}, context,
                     resultType);
}

PValue IREmitter::emitExprPValue(const ExprNode *expr, ExprContext context,
                                 ASTType resultType) {
  // Clear the builder to indicate that an PValue must be emitted.
  llvm::SaveAndRestore savedBuilder(builder, {});
  llvm::SaveAndRestore savedContext(paramContext, context);

  // Emit the expression using the contextual type if present.
  AnyValue rep = emitExpr(expr, context, resultType);
  return emitPValue({rep, expr}, context);
}

LValue IREmitter::emitExprLValue(const ExprNode *expr, ExprDest &dest) {
  AnyValue anyValue = emitExpr(expr, dest);
  return emitLValue({anyValue, expr}, dest);
}

/// Emit a copy of the specified value, producing a new owned instance of the
/// value in the specified destination.  This returns an RValue if
/// there is no consuming dest, otherwise a BValue.
CValue IREmitter::emitCopyOfValue(ASTExprAnd<CValue> value, ExprDest &dest) {
  ASTType valueType = value.ir.getRValueType();
  SMLoc exprLoc = value.expr->getLoc();
  if (!value.ir)
    return {};

  // Resolve away DLValue's.
  if (auto dlValue = value.ir.getIfDLValue())
    return dlValue->emitLoad(dest, *this);

  bool isRegisterPassable = valueType.isRegisterPassable(exprLoc, shared);

  // If the value is PValue and register passable, then we can materialize a
  // unique value directly into a register.
  if (auto pValue = value.ir.getIfPValue()) {
    if (isRegisterPassable) {
      value.ir = emitPValueToSRValue({pValue, value.expr}, dest.context);
      return emitCResult(value.ir, value.expr, dest);
    }
  }

  // If the value's type is trivial then we don't need to do anything except
  // convert to an RValue and emit to the destination.
  if (valueType.isTrivial(exprLoc, shared)) {
    // It is ok to upgrade SBValue to SRValue for trivial types.
    if (auto sbVal = value.ir.getIfSBValue())
      value.ir = SRValue(sbVal);

    // All trivial types are register passable right now, so we can load memory
    // values and produce an SRValue.
    if (value.ir.isMValue()) {
      if (!builder) {
        emitErrorForDynamicValueInParameter(value.expr);
        return {};
      }
      Value address = value.ir.getMValueReference();
      Value result =
          RefLoadOp::create(*builder, value.expr->getLocation(*this), address);
      value.ir = SRValue(result);
    }

    return emitCResult(value.ir, value.expr, dest);
  }

  // Otherwise, we'll need to memcpy or invoke the copy ctor method which will
  // take the destination by reference, so we're dealing with a memory case.
  bool isNonDefaultAddressSpace = dest.isNonDefaultAddressSpace();

  // If the value's type is trivially copyable, emit a memcpy. This allows
  // us to handle values residing in the non-default address space.
  // Only use memcpy for non-register passable types, unless we're forced to by
  // address space constraints.
  if (value.ir.isMValue() &&
      valueType.isProvablyImplicitlyTriviallyCopyable(exprLoc, shared,
                                                      declScope) &&
      (!isRegisterPassable || isNonDefaultAddressSpace)) {
    if (!builder) {
      emitErrorForDynamicValueInParameter(value.expr);
      return {};
    }
    Value address = value.ir.getMValueReference();
    MLValue destBuffer = dest.getMLValueForResult(exprLoc, valueType, *this);
    if (!destBuffer)
      return {};
    MemcpyOp::create(*builder, translateLocation(exprLoc), address, destBuffer);
    value.ir = MRValue(destBuffer);
    return emitCResult(value.ir, value.expr, dest);
  }

  // Memory-only copy ctor will take the destination as address space zero, so
  // we need to reject ExprDest's expecting it in GPU memory.
  if (isNonDefaultAddressSpace) {
    emitError(exprLoc, "value of type ")
        << valueType << " cannot be copied into a non-default address space"
        << value.expr->getRange();
    return {};
  }

  // Materialize any PValue directly, so we can handle non-copyable and
  // non-movable types.
  if (auto pValue = value.ir.getIfPValue()) {
    // PValues don't have origins and are immortal with respect to the compiler.
    // Emit a memcpy into the LValue. Creating an SSA value of the memory-only
    // type for the sake of memcpy is safe because the bulk store will ensure
    // the variable does not get promoted off the stack, and after struct
    // lowering, the type is erased down to its MLIR constituents anyways.

    // FIXME: This isn't correct - it is emitting memory-only values into an
    // SSA value and then using lit.ref.store on the memory only value!
    SRValue regValue = emitPValueToSRValue({pValue, value.expr}, dest.context);
    if (!regValue)
      return {};
    MLValue destBuffer = dest.getMLValueForResult(exprLoc, valueType, *this);
    if (!destBuffer)
      return {};
    regValue = emitRebindOpIfNeeded(
        regValue, ASTType(destBuffer.getType()).getReferenceElementType(),
        exprLoc);
    RefStoreOp::create(*builder, translateLocation(exprLoc), regValue,
                       destBuffer);
    CValue result = MRValue(destBuffer);
    return emitCResult(result, value.expr, dest);
  }

  // Verify that the type is copyable in this way so we can generate tailored
  // error messages, rather than just allowing IREmitter to do it.
  if (!valueType.isImplicitlyCopyable(exprLoc, shared, declScope)) {
    // If the value is an RValue, it might be that the type isn't copyable or
    // movable at all. If so, give a specific error about this.
    if (value.ir.getIfRValue() &&
        !valueType.isMovable(exprLoc, shared, declScope) &&
        !valueType.isExplicitlyCopyable(exprLoc, shared, declScope)) {
      emitError(exprLoc, "value of type ")
          << valueType
          << " cannot be copied or moved; consider conforming it to 'Movable'"
          << value.expr->getRange();
      return {};
    }

    auto diag = emitError(exprLoc, "value of type ")
                << valueType << " cannot be implicitly"
                << " copied, it does not conform to 'ImplicitlyCopyable'"
                << value.expr->getRange();

    // Decide if we can take ownership of the specified value.
    auto canTransferFrom = [&]() -> bool {
      // Can only transfer from an MValue.  If it is already an RValue, then
      // transferring won't help!
      if (!value.ir.isMValue() || value.ir.getIfRValue())
        return false;

      Value val = OriginTrackable::findUnderlyingValueFromField(
          value.ir.getMValueReference());
      if (!val)
        return false;
      // Can't transfer from (e.g.) read arguments.
      return cast<RefType>(val.getType()).isMutableKnown(true);
    };

    // Suggest transfer if the type is movable, or if it is a transferable
    // MValue.
    if ((valueType.isMovable(exprLoc, shared, declScope) ||
         canTransferFrom())) {
      diag.attachNote(exprLoc)
          << "consider transferring the value with '^'"
          << FixIt::insertAfterToken(value.expr->getRangeEnd(), "^",
                                     shared.diags);
    }

    // Suggest .copy() if the type is explicitly copyable and we're trying to
    // implicitly copy it.
    if (valueType.isExplicitlyCopyable(exprLoc, shared, declScope)) {
      diag.attachNote(exprLoc)
          << "you can copy it explicitly with '.copy()'"
          << FixIt::insertAfterToken(value.expr->getRangeEnd(), ".copy()",
                                     shared.diags);
    }
    return {};
  }

  // Invoke `T(*, copy: Self)`.
  CallOperands operands(CallSyntax::kImplicitCopyCtor, value.expr,
                        std::move(dest));
  operands.add(StringAttr::get(shared.getContext(), "copy"), value,
               ArgUnpackStyle::kKeyword);
  return emitConstructorCall(valueType, std::move(operands));
}

CValue IREmitter::emitStoreToLValue(ASTExprAnd<CValue> value, LValue destLV,
                                    ExprContext context) {
  // Convert nonmaterializables.
  if (auto nmTarget =
          value.ir.getRValueType().getNonmaterializableTarget(shared)) {
    if (nmTarget.isEqualCanon(destLV.getRValueType())) {
      // If the destination is an MLValue with a matching type, then just
      // materialize directly into it and return instead of allocating a
      // temporary if the conversion constructor requires one.
      ExprDest nmConversionDest(destLV, context);
      return emitConstructorCall(
          nmTarget, CallOperands(CallSyntax::kTypeCall, value.expr,
                                 std::move(nmConversionDest), {value}));
    }
  }

  assert(value.ir.getRValueType().isEqualCanon(destLV.getRValueType()) &&
         "Types should match");

  // If the destination is a computed LValue, then perform a write.
  if (auto dlValue = destLV.getIfDLValue())
    return dlValue->emitStore(value, *this);

  // If the destination is a RLValue, then we are resolving a 'ref' or 'bind'
  // value into a VarDeclOp.
  if (auto rlValue = destLV.getIfRLValue()) {
    // The destination must be a VarDeclOp by construction.
    VarDeclOp refOp = cast<VarDeclOp>(rlValue.getDefiningOp());
    assert(refOp &&
           (refOp.getKind() == VarDeclKind::Ref ||
            refOp.getKind() == VarDeclKind::Bind) &&
           "not a ref or bind to initialize!");

    // Handle 'bind' by determining if this is a 'var' or immutable 'ref'.
    if (refOp.getKind() == VarDeclKind::Bind) {
      // If the value isn't a reference, we materialize it into a var binding.
      if (!value.ir.isMValue()) {
        // Switch the vardecl so that uses of it are treated as MBValue instead
        // of MLValues.
        refOp.setKind(VarDeclKind::Bound);
        refOp.getResult().setType(
            refOp.getType().getWithElement(value.ir.getRValueType()));

        // Apply type refinement now that we have the final type.
        maybeApplyTypeRefinement(refOp, declScope, *builder);

        // Now we store the value into the var decl.
        ExprDest bindDest(MLValue(refOp), context);
        emitBValue({value.ir, value.expr}, bindDest);
        return MBValue(refOp);
      }

      // Otherwise, handle this as an immutable ref.
      refOp.setKind(VarDeclKind::Ref);
      Value refValue = value.ir.getMValueReference();

      if (!cast<RefType>(refValue.getType()).isMutableKnown(false)) {
        refValue = RefImmutOp::create(*builder, value.expr->getLocation(*this),
                                      refValue);
      }
      value.ir = MBValue(refValue);
    }

    // If this is a 'ref', then we want non-MValues to be an error.
    Value mValue = emitRefValue(value, EC_RefBinding);
    if (!mValue)
      return {};

    // Now that we have the origin of the input, we can replace the placeholder
    // with the actual type so that uses of it will have the correct origin.
    refOp.getResult().setType(refOp.getType().getWithElement(mValue.getType()));

    RefStoreOp::create(*builder, translateLocation(value.expr->getLoc()),
                       mValue, refOp);

    // Apply type refinement after the store so that any load emitted by
    // the refinement (to collapse double-refs) reads initialized memory.
    maybeApplyTypeRefinement(refOp, declScope, *builder);

    return CValue::getMValueForRef(mValue); // Return the input reference.
  }

  // Otherwise, we know we have an MLValue destination.
  MLValue destRef = destLV.getIfMLValue();
  assert(destRef && "No other known LValue");
  ASTType valueType = value.ir.getRValueType();
  SMLoc exprLoc = value.expr->getLoc();

  // For tuple-element var patterns, type refinement is applied after the store
  // completes (matching the store-time strategy used for bind/ref patterns).
  auto applyRefinementIfVarDecl = [&] {
    if (builder)
      if (auto varOp = destRef.getDefiningOp<VarDeclOp>())
        maybeApplyTypeRefinement(varOp, declScope, *builder);
  };

  bool isRegisterPassable = valueType.isRegisterPassable(exprLoc, shared);
  bool isDefaultAS = cast<RefType>(destRef.getType()).isDefaultAddrSpace();

  // Verify that the result MLValue is in the right address space for a
  // copy/move constructor call, if it would come down to that.
  if (!isDefaultAS && !valueType.isProvablyImplicitlyTriviallyCopyable(
                          exprLoc, shared, declScope)) {
    emitError(exprLoc, "value of type ")
        << valueType
        << " cannot be copied or moved into a non-default address space"
        << value.expr->getRange();
    return {};
  }

  // If the input is an LValue/BValue (incl PValue) that we don't own, or if it
  // has no move ctor, or it's invalid to use its move ctor, then copy it
  // into the destination.
  if (!value.ir.getIfRValue() || value.ir.getIfPValue() ||
      (!isDefaultAS && !isRegisterPassable)) {
    ExprDest dest(destLV, context);
    auto result = emitCopyOfValue(value, dest);
    if (!result)
      dest.resetForError(*this);
    else
      applyRefinementIfVarDecl();
    return result;
  }

  // Otherwise this is a movable RValue that we own.
  // If it is a register passable, assign with a store.
  if (isRegisterPassable) {
    // Materialize a PValue or load a MRValue if present.
    SRValue val = emitSRValue(value, context, valueType);
    if (!val)
      return {};
    if (!builder) {
      emitErrorForDynamicValueInParameter(value.expr);
      return {};
    }
    // Store the value to memory after adjusting sugar.  StoreOp takes
    // ownership of the input SRValue.
    val = emitRebindOpIfNeeded(
        val, ASTType(destRef.getType()).getReferenceElementType(), exprLoc);
    RefStoreOp::create(*builder, translateLocation(exprLoc), val, destRef);
    applyRefinementIfVarDecl();
    // Must return a borrow of the result, use SBValue if we can to avoid a load
    // but otherwise we need a MBValue for non-trivial types.
    if (valueType.isTrivial(exprLoc, shared))
      return SBValue(val);
    return MBValue(destRef);
  }

  // Otherwise, assign with a move constructor.  We own the RValue, so prefer
  // to use move ctor if present.
  if (valueType.isMovable(exprLoc, shared, declScope)) {
    // Invoke `T(*, deinit move: Self)`.
    ExprDest moveDest(destRef, context);
    CallOperands operands(CallSyntax::kImplicitMoveCtor, value.expr,
                          std::move(moveDest));
    operands.add(StringAttr::get(shared.getContext(), "move"), value,
                 ArgUnpackStyle::kKeyword);
    if (!emitConstructorCall(valueType, std::move(operands)))
      return {};
    applyRefinementIfVarDecl();
    return MBValue(destRef);
  }

  // Otherwise, we have to move this thing but don't have a move constructor!
  emitError(value.expr->getLoc())
      << "cannot transfer value into destination, because " << valueType
      << " doesn't conform to 'Movable'";
  return {};
}

/// Emit IR for the specified expression without adding it to the current
/// execution context.  This even allows evaluating dynamic expressions in a
/// parameter context.  When the result is computed, evaluate the specified
/// callback on the result and then discard the result.
///
/// On failure, an error is emitted and the callback is not invoked.
///
/// This is used for evaluating expressions like `origin_of(x)` and
/// `type_of(x)` and `ref [x] T`.
void IREmitter::emitExpressionWithoutEvaluatingIt(
    const ExprNode *expr, ExprContext exprContext,
    std::function<void(CValue, IREmitter &emitter)> callback) {
  SMLoc loc = expr->getLoc();
  // The emitter indicates what context to do name lookup against, but cannot
  // be used to emit the IR into.  Find something in the declScope with an
  // Operation (e.g. a function), which will allow us to put in a Block to emit
  // into.  This is a bit of a hack, but is required because some things scan
  // up the region hierarchy.
  ASTDecl *curDecl = &declScope;
  Operation *opToInsertInto = nullptr;
  // Scan for an operation with a region.
  while (!(opToInsertInto = curDecl->getIfOperation()) ||
         opToInsertInto->getNumRegions() == 0) {
    curDecl = curDecl->getParentDecl();
    if (!curDecl) {
      emitError(loc, "INTERNAL ERROR: could not find context to emit IR "
                     "into.  Please file a bug.");
      return;
    }
  }

  auto location = expr->getLocation(*this);

  // Okay we found an operation with a region.  Abuse it :-) by adding a new
  // block, which keeps any code we're emitting contained.
  Region &r = opToInsertInto->getRegion(0);
  Block &tmpBlock = r.emplaceBlock();
  IREmitter tmpEmitter(declScope, OpBuilder::atBlockBegin(&tmpBlock));

  // Go further and add a 'try' op to it, ensuring that throwing functions are
  // allowed in this expression.
  VarDeclOp errDecl =
      tmpEmitter.emitVarDecl("__try_error__", UnresolvedType::get(getContext()),
                             location, VarDeclKind::Synthesized);
  auto tryOp = TryOp::create(*tmpEmitter.builder, location, errDecl);

  // Parse the expression into the try block.
  tmpEmitter.builder->createBlock(&tryOp.getTryRegion());

  // Emit the expression and invoke the callback on success.
  CValue subExprValue = tmpEmitter.emitExprCValue(expr, exprContext);
  if (subExprValue)
    callback(subExprValue, tmpEmitter);

  // Finally, remove our temp block
  tmpBlock.erase();
}

//===----------------------------------------------------------------------===//
// Emission helpers for specific value types.

ASTType IREmitter::emitExprType(const ExprNode *expr, bool allowUnbound) {
  // We have two ambiguous expressions that can either be types or dynamic
  // values: an empty tuple () and None.  In a type context, we want to treat
  // these as types, and not dynamic values.  Sniff these out to see if we have
  // them.
  const ExprNode *innerExpr = expr->getWithoutParens();
  if (innerExpr->kind == ExprNode::kNoneLiteral)
    return shared.getNoneType();
  if (innerExpr->isEmptyTuple())
    return getBuiltinTupleInstantiation(expr->getLoc(), {});

  // A non-empty tuple literal is never ambiguous: unlike () and None, it has
  // no valid reading as a type.  Diagnose this directly instead of falling
  // through to the generic constructor-call machinery below.
  if (isa<TupleNode>(innerExpr)) {
    emitError(
        expr->getLoc(),
        "expected a type, found a tuple value; use 'Tuple[...]' to write a "
        "tuple type")
        << expr->getRange();
    return {};
  }

  auto value = emitExprPValue(expr, EC_Type);
  return emitType({value, expr}, allowUnbound);
}

/// This emits the specified PValue as a type, binding defaulted parameters
/// etc if needed.
ASTType IREmitter::emitType(ASTExprAnd<PValue> value, bool allowUnbound) {
  if (!value.ir)
    return {};

  ASTType type = value.ir.getIfTypeValue();
  if (!type) {
    emitError(value.expr->getLoc(), "expected a type, not a value")
        << value.expr->getRange();
    return {};
  }

  // If the caller accepts a fully unbound type and the type is unbound, return
  // it now without verifying the bindings.
  if (allowUnbound)
    return type;

  // Check for a function type.
  if (auto sig = dyn_cast<FnTypeGeneratorType>(type)) {
    // For a fully bound type, require that the origin set is concrete.
    if (isa<UnboundAttr>(sig.getCaptureOrigins())) {
      emitError(value.expr->getLoc(),
                "function type missing required origin set parameter")
          << value.expr->getRange();
      return {};
    }

    // Function types with non-singleton parameters can only be used at
    // comptime.
    for (auto [paramType, pog] : llvm::zip(sig.getInputParamTypes(),
                                           sig.getParamListAttrs().getPogs())) {
      // Singleton values like origins are fine. They will be removed by
      // lowerlit before code generation.
      if (ASTType(paramType).isSingleton(shared))
        continue;

      auto diag =
          emitError(value.expr->getLoc(),
                    "cannot use parametric function as a runtime closure")
          << value.expr->getRange();
      diag.attachNote(value.expr->getLoc())
          << "parameter " << ParamDeclRefAttr::get(pog.getName(), paramType)
          << " of type " << ASTType(paramType) << " is not bound";
      return {};
    }
  }
  if (llvm::any_of(type.getParamBindings(),
                   [](TypedAttr attr) { return isa<UnboundAttr>(attr); })) {
    emitError(value.expr->getLoc())
        << type << " is not concrete, use '[]' to bind missing parameters";
    return {};
  }

  // Reject generator types (e.g. comptime aliases with unbound parameters)
  // where a concrete type is required. Generators appear as ParamType whose
  // metatype (after stripping sugar) is a GeneratorType.
  if (auto gen = sugarDynCast<GeneratorType>(type.extractMetaType())) {
    if (!gen.isFullyBound()) {
      emitError(value.expr->getLoc())
          << type
          << " is not a concrete type, use '[]' to bind missing parameters";
      return {};
    }
  }

  return type;
}

RValue IREmitter::emitScalarBool(ASTExprAnd<CValue> value,
                                 ExprContext context) {
  if (!value.ir)
    return {};

  ASTType valueRValueType = value.ir.getRValueType();

  // If this is already an 'scalar<bool>', then we're done.
  if (isScalarOf<KGENDType::kBool>(valueRValueType.mlirType))
    return emitRValue(value, context);

  // TODO: Python manual includes this off-hand comment:
  // Also, an object that doesn’t define a __bool__() method and whose __len__()
  // method returns zero is considered to be false in a Boolean context.

  // For stdlib Bool: handle it before the typeHasMember / extractStructField
  // paths.  Both of those call lookupAndResolveDecl → resolveBody(Bool), which
  // can trigger a circular dependency when evaluating a struct-conformance
  // ‘where’ clause that involves Scalar (e.g. SIMD’s DevicePassable where
  // clause): resolveBody(Bool) processes Bool.__del__is_trivial (= True),
  // which resolves ALL Bool.__init__ overloads, including
  // Bool.__init__(value: Scalar[DType.bool]). That resolution needs
  // Scalar[DType.bool] → Scalar, which is already in declsCurrentlyProcessing.
  //
  // By handling Bool up front, we extract _mlir_value from the PValue
  // attribute directly — zero resolution required.
  ASTType boolType =
      shared.lookupBuiltinType("Bool", declScope, value.expr->getLoc());
  if (value.ir.getRValueType().isEqualCanon(boolType)) {
    if (PValue pvalue = value.ir.getIfPValue()) {
      if (auto structAttr = dyn_cast<LITStructAttr>(pvalue.get())) {
        auto mlirValueName = StringAttr::get(getContext(), "_mlir_value");
        for (auto &[name, val] : structAttr.getValues())
          if (name == mlirValueName)
            return emitRValue({PValue(val), value.expr}, context);
      }
      if (auto extractVal = ASTType::extractStructField(
              pvalue.get(), "_mlir_value", value.expr->getLoc(), shared))
        return emitRValue({PValue(extractVal), value.expr}, context);
    }
  }

  // Check for the presence of a __mlir_bool__ method.  If it exists, we can
  // avoid a redundant call to __bool__ for Bool types.
  if (!shared.typeHasMember(valueRValueType, "__mlir_bool__",
                            value.expr->getLoc())) {
    // Use the __bool__ method to convert the user defined type to
    // something that is a Bool or other type that implements __mlir_bool__.
    value.ir =
        emitNamedMethodCall("__bool__", CallOperands{CallSyntax::kMethodCall,
                                                     value.expr,
                                                     context,
                                                     {{value.ir, value.expr}}});
  }

  // Re-check after potential __bool__ conversion: if it’s now a Bool value,
  // extract _mlir_value directly.
  if (value.ir.getRValueType().isEqualCanon(boolType)) {
    if (PValue pvalue = value.ir.getIfPValue()) {
      if (auto structAttr = dyn_cast<LITStructAttr>(pvalue.get())) {
        auto mlirValueName = StringAttr::get(getContext(), "_mlir_value");
        for (auto &[name, val] : structAttr.getValues())
          if (name == mlirValueName)
            return emitRValue({PValue(val), value.expr}, context);
      }
      if (auto extractVal = ASTType::extractStructField(
              pvalue.get(), "_mlir_value", value.expr->getLoc(), shared))
        return emitRValue({PValue(extractVal), value.expr}, context);
    }
  }

  // For other types that implement __mlir_bool__, call the method.
  CValue litBoolCall = emitNamedMethodCall(
      "__mlir_bool__", CallOperands{CallSyntax::kMethodCall,
                                    value.expr,
                                    context,
                                    {{value.ir, value.expr}}});

  // If we got back a sugared PValue call to the method, then drop the sugar.
  // This reduces the size of the printed IR, making it easier to read, and the
  // user never wants to see a call to this function in a diagnostic anyway.
  if (auto pvalue = litBoolCall.getIfPValue())
    if (auto sugar = llvm::dyn_cast_or_null<SugarAttr>(pvalue.get()))
      if (sugar.getKind() == SugarKind::AlwaysInlineBuiltin)
        litBoolCall = sugar.getExpanded();

  return emitRValue({litBoolCall, value.expr}, context);
}

RValue IREmitter::emitExprScalarBool(const ExprNode *condExpr,
                                     ExprContext context) {
  return emitScalarBool({emitExprCValue(condExpr, context), condExpr}, context);
}

CValue IREmitter::emitAndMatchPredicates(
    ASTExprAnd<CValue> lhs, llvm::function_ref<ASTExprAnd<CValue>()> emitRhs,
    bool evaluateRhsEvenIfLhsFalse) {
  auto asBoolAttr = sugarDynCastIfPresent<SIMDAttr>(lhs.ir.getIfPValue());
  if (asBoolAttr && asBoolAttr.getAsBool())
    return emitRhs().ir;
  if (asBoolAttr && !asBoolAttr.getAsBool()) {
    // Always emit the rhs when requested (e.g. case guards), even if the
    // conjunction can never succeed, so diagnostics still fire.
    if (evaluateRhsEvenIfLhsFalse && !emitRhs().ir)
      return {};
    return lhs.ir;
  }

  if (!builder)
    return emitErrorForDynamicValueInParameter(lhs.expr);

  Location loc = lhs.expr->getLocation(*this);
  SRValue lhsSR = emitSRValue(lhs, EC_BoolCondition);
  if (!lhsSR)
    return {};

  auto boolType = SIMDType::getScalarBoolType(getContext());
  HLCF::ElifOp elifOp = HLCF::ElifOp::create(
      *builder, loc, TypeRange{boolType}, lhsSR,
      [&]() -> LogicalResult {
        ASTExprAnd<CValue> rhs = emitRhs();
        if (!rhs.ir)
          return failure();
        SRValue rhsSR = emitSRValue(rhs, EC_BoolCondition);
        if (!rhsSR)
          return failure();
        HLCF::YieldOp::create(*builder, loc, rhsSR);
        return success();
      },
      [&]() -> LogicalResult {
        Value falseVal = ParamConstantOp::create(
            *builder, loc, SIMDAttr::getScalarBool(getContext(), false));
        HLCF::YieldOp::create(*builder, loc, falseVal);
        return success();
      });
  if (!elifOp)
    return {};
  return SRValue(elifOp.getResult(0));
}

CValue IREmitter::emitOrMatchPredicates(
    ASTExprAnd<CValue> lhs, llvm::function_ref<ASTExprAnd<CValue>()> emitRhs) {
  auto asBoolAttr = sugarDynCastIfPresent<SIMDAttr>(lhs.ir.getIfPValue());
  if (asBoolAttr && asBoolAttr.getAsBool())
    return lhs.ir;
  if (asBoolAttr && !asBoolAttr.getAsBool())
    return emitRhs().ir;

  if (!builder)
    return emitErrorForDynamicValueInParameter(lhs.expr);

  Location loc = lhs.expr->getLocation(*this);
  SRValue lhsSR = emitSRValue(lhs, EC_BoolCondition);
  if (!lhsSR)
    return {};

  auto boolType = SIMDType::getScalarBoolType(getContext());
  HLCF::ElifOp elifOp = HLCF::ElifOp::create(
      *builder, loc, TypeRange{boolType}, lhsSR,
      [&]() -> LogicalResult {
        Value trueVal = ParamConstantOp::create(
            *builder, loc, SIMDAttr::getScalarBool(getContext(), true));
        HLCF::YieldOp::create(*builder, loc, trueVal);
        return success();
      },
      [&]() -> LogicalResult {
        ASTExprAnd<CValue> rhs = emitRhs();
        if (!rhs.ir)
          return failure();
        SRValue rhsSR = emitSRValue(rhs, EC_BoolCondition);
        if (!rhsSR)
          return failure();
        HLCF::YieldOp::create(*builder, loc, rhsSR);
        return success();
      });
  if (!elifOp)
    return {};
  return SRValue(elifOp.getResult(0));
}

CValue IREmitter::emitIndex(ASTExprAnd<AnyValue> value, ExprContext context) {
  // If the value is already of index type, just use it.
  if (CValue cvalue = value.ir.getIfCValue())
    if (isa<IndexType>(cvalue.getRValueType().mlirType))
      return cvalue;

  auto result = emitNamedMethodCall(
      "__mlir_index__",
      CallOperands{CallSyntax::kMethodCall, value.expr, context, {value}});

  // If we got back a sugared PValue call to the method, then drop the sugar.
  // This reduces the size of the printed IR, making it easier to read, and the
  // user never wants to see a call to this function in a diagnostic anyway.
  if (auto pvalue = result.getIfPValue())
    if (auto sugar = llvm::dyn_cast_or_null<SugarAttr>(pvalue.get()))
      if (sugar.getKind() == SugarKind::AlwaysInlineBuiltin)
        result = sugar.getExpanded();

  return result;
}

CValue IREmitter::emitIndex(const ExprNode *expr, ExprContext context) {
  return emitIndex({emitExprCValue(expr, context), expr}, context);
}

CValue IREmitter::emitBool(ASTExprAnd<PValue> value, ExprDest &dest) {
  ASTType boolType =
      shared.lookupBuiltinType("Bool", declScope, value.expr->getLoc());

  // Fast path for scalar bool SIMD attributes (e.g. from True/False literals
  // or `@always_inline("builtin")` comparisons): build the Bool struct
  // attribute directly rather than going through emitConstructorCall →
  // OverloadSet::lookup("__init__", Bool) → lookupAndResolveDecl →
  // resolveBody(Bool).  resolveBody(Bool) triggers Bool.__del__is_trivial
  // (= True), which resolves ALL Bool.__init__ overloads including
  // Bool.__init__(value: Scalar[DType.bool]). That in turn needs
  // Scalar[DType.bool] → Scalar, causing a circular dependency when Scalar is
  // already in declsCurrentlyProcessing (e.g. while evaluating SIMD's
  // DevicePassable 'where' clause).
  if (auto simdAttr = dyn_cast<SIMDAttr>(value.ir.get())) {
    if (isScalarOf<KGENDType::kBool>(simdAttr.getType())) {
      if (auto boolStructType =
              sugarDynCast<LIT::StructType>(boolType.mlirType)) {
        auto mlirValueFieldName = StringAttr::get(getContext(), "_mlir_value");
        std::tuple<StringAttr, TypedAttr> field{mlirValueFieldName,
                                                (TypedAttr)simdAttr};
        if (TypedAttr boolStructAttr =
                LITStructAttr::get({field}, boolStructType))
          return emitRValue({AnyValue(PValue(boolStructAttr)), value.expr},
                            dest);
      }
    }
  }

  CallOperands operands(CallSyntax::kImplicitConvert, value.expr,
                        std::move(dest), {value});
  return emitConstructorCall(boolType, std::move(operands));
}

CValue IREmitter::emitBool(ASTExprAnd<PValue> value, ExprContext context) {
  ExprDest dest(context);
  return emitBool(value, dest);
}

CValue IREmitter::emitInt(ASTExprAnd<AnyValue> indexValue, ExprDest &dest) {
  ASTType intType = shared.lookupBuiltinType("Int", getDeclScope(),
                                             indexValue.expr->getLoc());

  // Int is now SIMD[DType.int, 1], so its mlir_value init expects !pop.simd<1,
  // index>, not a bare index.  If the caller passed us a plain IntegerAttr with
  // IndexType (the historical form), wrap it in a POP::SIMDAttr.
  AnyValue simdValue = indexValue.ir;
  if (PValue pval = simdValue.getIfPValue()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(pval.get())) {
      if (intAttr.getType().isIndex()) {
        MLIRContext *ctx = getContext();
        simdValue = PValue{SIMDAttr::get(
            DTypeValue(intAttr.getInt(), KGENDType::index),
            SIMDType::get(
                /*size=*/1, DTypeConstantAttr::get(ctx, KGENDType::index)))};
      }
    }
  }
  ASTExprAnd<AnyValue> simdIndexValue{simdValue, indexValue.expr};

  // Build Int from __mlir_type.index explicitly: Int.__init__(*, mlir_value=…)
  CallOperands intCtorOperands(CallSyntax::kTypeCall, indexValue.expr,
                               std::move(dest));
  intCtorOperands.add(StringAttr::get(getContext(), "mlir_value"), indexValue,
                      ArgUnpackStyle::kKeyword);
  return emitConstructorCall(intType, std::move(intCtorOperands));
}

CValue IREmitter::emitInt(ASTExprAnd<AnyValue> indexValue,
                          ExprContext context) {
  ExprDest dest(context);
  return emitInt(indexValue, dest);
}

/// This returns an instance of Tuple[...] with the specified element types
/// installed.
ASTType IREmitter::getBuiltinTupleInstantiation(llvm::SMLoc loc,
                                                ArrayRef<Type> elements) {
  auto tupleType = shared.lookupBuiltinType("Tuple", declScope, loc);
  if (tupleType.isTypeCheckErrorType())
    return {};
  ASTDecl *typeDecl = ASTType(tupleType).getDecl(shared);

  SyntheticNode tmpExpr(loc);
  ParamBindings bindings(getDeclScope(), &tmpExpr);
  for (ASTType elt : elements)
    bindings.add(&tmpExpr, PValue(elt));

  // Check the bindings.
  auto metaType = cast<StructMetaType>(tupleType.extractMetaType());
  TypeSignatureType sig = metaType.getSignature();
  ParamInf inference(bindings, sig.getParamTypes(), sig.getParamListAttrs(),
                     /*allowImplicitConversions=*/true, typeDecl,
                     /*discardError=*/false,
                     /*deferredTypingContext=*/deferredTypingContext);
  VerifiedParamBindings verifiedBindings = inference.inferForStruct();
  if (!verifiedBindings)
    return {};

  // Ok, we succeeded at reparameterizing the type.
  return verifiedBindings.specializeGenerator(PValue(tupleType));
}

//===----------------------------------------------------------------------===//
// Error handling helpers.

MLValue IREmitter::findNearestErrorSlot() {
  assert(builder && "cannot raise in a context without a builder");
  Operation *opForRaise = findOpProcessingRaise(builder->getInsertionBlock());
  // Return null to indicate that the current context cannot raise.
  if (!opForRaise)
    return {};

  // In a raising function, the error slot is always the second last argument.
  if (auto func = dyn_cast<FnOp>(opForRaise))
    return func.getArgument(func.getNumArguments() - 2);

  // Otherwise, the error slot is carried by the surrounding try op.
  return cast<LIT::TryOp>(opForRaise).getErr();
}

/// When a try block gets its error type inferred, this function makes sure the
/// inferred type doesn't capture an origin from within a try body.  Such a
/// thing would be an out of scope reference, e.g.:
///
///    try:
///      var x = 42
///      raise Pointer(to=x)
///    except e: # x is not in scope here.
void IREmitter::checkInferredErrorType(ASTType rvalueType, SMLoc loc) {
  SmallVector<TypedAttr> origins =
      shared.cachedOriginFinder.findOriginsIn({rvalueType});
  // Typically we have something like Error or TypeCheckError which will have no
  // origins, so we can avoid doing work.
  if (origins.empty())
    return;

  // Ok, find the try block that we're inferring for.  We must be in a 'try'
  // because that is the only thing that can infer an error type.
  assert(builder && "cannot raise in a context without a builder");
  auto tryOp =
      dyn_cast<LIT::TryOp>(findOpProcessingRaise(builder->getInsertionBlock()));
  if (!tryOp)
    return; // Functions don't infer their error type.

  // Unfortunately, we don't have a good way to do a lookup given an origin
  // attribute, so we scan the body of the try block for any vardecls. Is this
  // the only thing that can declare an origin?
  SmallPtrSet<Attribute, 8> originSet;
  for (auto o : origins)
    originSet.insert(OriginType::stripMutCastAndRebind(o));
  tryOp.getTryRegion().walk([&](VarDeclOp varDecl) {
    if (originSet.contains(varDecl.getType().getOrigin())) {
      auto diag = emitError(loc);
      diag << "inferred error type " << rvalueType << " captures origin ";
      if (varDecl.isSynthetic()) {
        diag << "of temporary";
      } else {
        diag << "'"
             << ASTType::getOriginAsString(varDecl.getType().getOrigin(),
                                           &shared)
             << "'";
      }
      diag << " from within try body; it is not in scope in except body";
      diag.attachNote(varDecl.getLoc()) << "origin declared here";
    }
  });
}

//===----------------------------------------------------------------------===//
// Return emission helpers.

void IREmitter::emitNormalReturn(ImplicitLocOpBuilder &builder, Value value,
                                 bool emitEndFunc) {
  auto func = getBlockParentOfType<FnOp>(builder.getInsertionBlock());
  assert(func && "Emitting a return in a non-function?");

  auto signature = func.getFuncTypeGenerator();
  if (value) {
    // If we have a value, then make sure any sugar is adjusted.

    // Rebind away any sugar if it exists.
    if (value.getType() != func.getMLIRResultType())
      value = RebindOp::create(builder, func.getMLIRResultType(), value);
  } else {
    // If we're missing a value, then we either have a memory result that has
    // already been emitted to its slot, or a function that returns None. Either
    // way, generate a None or i1 to return with lit.return.

    // If the function returns a None type value by-reference, fill it in.  This
    // happens in throwing functions.
    if (signature.hasMemoryOnlyResult() &&
        ASTType(func.getUserResultType()).isNoneType()) {
      assert(signature.getArgConventions().back() ==
                 ArgConvention::ByRefResult &&
             "by-ref result should be the last argument");

      // This value will also get returned unless the function throws.
      value =
          ParamConstantOp::create(builder, NoneAttr::get(func.getContext()));
      RefStoreOp::create(builder, value, func.getArguments().back());
    }

    // Otherwise, the resulting actual function result must be a none-type or a
    // bool for a throwing result.
    if (signature.isThrows())
      value = ParamConstantOp::create(
          builder, SIMDAttr::getScalarBool(builder.getContext(), false));
    else if (!value)
      value =
          ParamConstantOp::create(builder, NoneAttr::get(func.getContext()));
  }

  // Handle any `deinit` argument by marking it destroyed.
  for (auto [conv, arg] : llvm::zip(signature.getArgConventions(),
                                    func.getBody()->getArguments())) {
    if (conv == ArgConvention::DeinitMem)
      LIT::OwnershipMarkDestroyedOp::create(builder, arg);
  }

  // Finally we emit a normal return with lit.return.
  assert(value && "Didn't specify a return value for the function");
  LIT::ReturnOp::create(builder, value);

  // If requested, emit the end func.
  if (emitEndFunc)
    EndFnOp::create(builder);
}

/// Emit a normal return (not a 'raise' return) out of the function, along
/// with any special logic that goes with it.  If the value is missing this is
/// treated as a 'return;' synthesizing a None result.
void IREmitter::emitNormalReturn(Location loc, Value value, bool emitEndFunc) {

  // If this function returns in a register, load the result value from the
  // result slot temp. We compile things like:
  //    def example(out x: Int):
  // to have a local vardecl that can be mutated, and is loaded implicitly
  // when a "return" with no expression is used.
  if (!value) {
    auto func = getBlockParentOfType<FnOp>(builder->getInsertionBlock());
    if (func.getNamedResultAttr() &&
        !func.getFuncTypeGenerator().hasMemoryOnlyResult()) {
      auto *funcDecl = declScope.getNearestDeclOfType<FnOp>();
      assert(funcDecl && "must be in a function");
      ArrayRef<ASTDecl *> declList =
          funcDecl->lookupInCurrentScope(func.getNamedResultAttr());
      assert(declList.size() == 1 && "result temp should always be findable");
      auto irVal = declList[0]->getIfIRValue().getIfMLValue();
      assert(irVal && "result temp should always be in memory");
      SyntheticNode exprTmp(funcDecl->getLoc());
      // Move the source by interpreting the MLValue as an MRvalue.
      value = emitSRValue({MRValue(irVal), &exprTmp}, EC_ReturnValue);
      if (!value)
        return;
    }
  }

  ImplicitLocOpBuilder b(loc, *builder);
  emitNormalReturn(b, value, emitEndFunc);
}

//===--------------------------------------------------------------------===//
// Var emission helpers.
//===--------------------------------------------------------------------===//

VarDeclOp IREmitter::emitVarDecl(const Twine &name, Type type, Location loc,
                                 VarDeclKind kind) {
  if (!builder) {
    emitErrorForDynamicValueInParameter(loc);
    return {};
  }
  StringAttr originAttr = declScope.mangleParamName(name);
  return VarDeclOp::create(*builder, loc, type, name.str(), originAttr, kind);
}

VarDeclOp IREmitter::emitVarDecl(StringAttr name, Type type, Location loc,
                                 VarDeclKind kind) {
  return emitVarDecl(name.strref(), type, loc, kind);
}

//===--------------------------------------------------------------------===//
// Origin helpers.
//===--------------------------------------------------------------------===//

/// Given a value of !lit.origin type, return an instance of
/// Origin[mut, lit.origin]().
PValue IREmitter::getStdlibOriginOf(TypedAttr litOrigin, SMLoc loc) {
  assert(sugarIsa<OriginType>(litOrigin.getType()) && "Need a !lit.origin");
  SyntheticNode expr(loc);

  // Convert to Origin type, start by looking it up.
  ASTType originType = shared.lookupBuiltinType("Origin", declScope, loc);
  if (sugarIsa<TypeCheckErrorType>(originType))
    return {}; // Sanity check the returned declaration.
  auto originStructType = sugarCast<LIT::StructType>(originType);

  // Get the mutability as a Bool.
  auto resultMutability =
      sugarCast<OriginType>(litOrigin.getType()).getIsMutable();
  std::tuple<StringAttr, TypedAttr> field = {
      StringAttr::get(getContext(), "_mlir_value"), resultMutability};
  auto boolStructType =
      cast<LIT::StructType>(originStructType.getSignature().getParamTypes()[0]);
  PValue mutBool = LITStructAttr::get({field}, boolStructType);

  ASTDecl *decl = originType.getDecl(shared);
  auto litStruct = dyn_cast_if_present<StructDeclOp>(decl->getIfOperation());
  if (!litStruct || litStruct.getParams().size() != 2 ||
      !sugarIsa<OriginType>(litStruct.getParams()[1].getType())) {
    emitError(loc, "malformed Origin type");
    return {};
  }

  // Bind the Origin parameters.
  originType = litStruct.bindReference({mutBool, litOrigin});

  // If we are referencing an origin parameter, we need to find the
  // Origin it corresponds to and use that. Consider:
  //   def test[X: ImmOrigin](ref [X] a: Int):
  // turns into:
  //   def test[X._mlir_origin, //, X: Origin](ref [X._mlir_origin] a: Int)
  // as such, we'll see references to the mlir_origin, but we want to find
  // X. Scrub around to try to figure this out.
  TypedAttr paramVal;
  if (auto paramRef = sugarDynCast<ParamDeclRefAttr>(
          OriginType::stripMutCastAndRebind(litOrigin))) {
    auto [curDecl, paramDecls, paramIdx] =
        getDeclScope().lookupParamReference(paramRef);
    // We found the MLIR origin, but it is an autoparam at the start of
    // the list.  Find the Origin which can be explicitly declared
    // anywhere later.
    if (curDecl) {
      for (size_t i = paramIdx + 1, e = paramDecls.size(); i < e; ++i) {
        if (isEqualCanon(paramDecls[i].getType(), originType))
          return ParamDeclRefAttr::get(paramDecls[i]);
      }
    }
  }

  // There is no actual reason to construct the stateless Origin type here, just
  // directly make the same attribute the ctor will fold to.
  return LIT::getEmptyStructValue(originType);
}

//===--------------------------------------------------------------------===//
// Parametric closure trait helpers.
//===--------------------------------------------------------------------===//

ASTDecl *IREmitter::createParametricClosureTrait(SharedState &shared) {
  OpBuilder b(shared.getTopLevelDecl().getIfOperation());
  b.setInsertionPointToStart(
      &cast<ModuleOp>(shared.getTopLevelDecl().getIfOperation())
           .getBodyRegion()
           .front());
  MLIRContext *ctx = b.getContext();

  // A illegal name to avoid collisions.
  StringRef name = UNI_CLOSURE_TRAIT_NAME;
  auto closureTrait =
      TraitDeclOp::create(b, mlir::UnknownLoc::get(ctx), // synthetic trait
                          StringAttr::get(ctx, name));
  ASTDecl &traitDecl = shared.declResolver->addFullyResolvedDecl(
      &*closureTrait, name, SMLoc(), &shared.getTopLevelDecl());

  // Populate the trait with parent and self methods.
  SmallVector<TraitSymbolAttr> parents;
  DenseSet<TraitSymbolAttr> immediateParents;
  for (auto traitName : {"AnyType", "Movable", "Deinitable"}) {
    ASTDecl *traitDecl = shared.lookupBuiltinTrait(traitName, SMLoc());
    if (traitDecl) { // builtin might be disabled.
      if (failed(shared.declResolver->resolveSignature(*traitDecl, SMLoc())))
        return nullptr; // should this be an assertion?
      parents.push_back(
          cast<TraitDeclOp>(traitDecl->getIfOperation()).bindReference({}));
    }
  }

  SmallVector<ParamDeclAttr> decls = {
      // The parameter decl list, passed in as a param_list of param decls.
      ParamDeclAttr::get("P#0", ParamListType::get(TypeType::get(ctx))),
      // The argument type list.
      ParamDeclAttr::get("A#1", ParamListType::get(TypeType::get(ctx))),
      // A single Result type.
      ParamDeclAttr::get("R#2", TypeType::get(ctx)),
      // The metadata.
      ParamDeclAttr::get("M#3", NonStructTypeType::get(ctx)),
      // The pog_list for parameters
      ParamDeclAttr::get("POG_P#4", NonStructTypeType::get(ctx)),
      // The pog_list for arguments.
      ParamDeclAttr::get("POG_A#5", NonStructTypeType::get(ctx)),
  };
  SmallVector<PassingKind> passingKinds(decls.size(), PassingKind::PosOnly);

  [[maybe_unused]] LogicalResult result =
      shared.declResolver->addSelfTypeToTrait(closureTrait, traitDecl, parents,
                                              immediateParents, decls,
                                              passingKinds);

  // This need to built a full signature with Self prepended properly.

  auto prependParamList = [](TypedAttr toPrepend,
                             TypedAttr paramList) -> TypedAttr {
    return ParamListConcatAttr::get(
        {ParamListAttr::get({toPrepend},
                            cast<ParamListType>(paramList.getType())),
         paramList});
  };

  // An extra _Self parameter.
  ASTType declSelf = traitDecl.getTypeDeclSelf();
  auto selfParam = QuoteAttr::get(
      TypeParamAttr::get(declSelf.extractMetaType(), TypeType::get(ctx)));

  // An extra `mut self` argument: we don't have two trait for FnMut/FnImm, use
  // mut self for better generality.
  auto selfOriginType = OriginType::get(shared.getContext(), true);
  auto mutSelfRef = LIT::RefType::get(
      ParamType::get(ParamIndexRefAttr::get(0, declSelf.extractMetaType())),
      ImplicitOriginRefAttr::get(0, 0, selfOriginType));
  auto mutSelf =
      QuoteAttr::get(TypeParamAttr::get(mutSelfRef, TypeType::get(ctx)));

  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
      closureTrait.getLoc(), &closureTrait.getFields().front());

  auto params = prependParamList(selfParam, ParamDeclRefAttr::get(decls[0]));
  auto args = prependParamList(mutSelf, ParamDeclRefAttr::get(decls[1]));
  auto retTy = ParamDeclRefAttr::get(decls[2]);
  auto metadata = ParamDeclRefAttr::get(decls[3]);
  auto paramListPog = ParamDeclRefAttr::get(decls[4]);
  auto argListPog = ParamDeclRefAttr::get(decls[5]);

  auto aliasOp = AliasDeclOp::create(
      builder, ParamDeclAttr::get(builder.getStringAttr("__call__"),
                                  KGEN::FuncGeneratorTypeBuilderType::get(
                                      ctx, params, args, retTy, metadata,
                                      paramListPog, argListPog)));

  (void)shared.declResolver->addFullyResolvedDecl(aliasOp, "__call__", SMLoc(),
                                                  &traitDecl);
  return &traitDecl;
}

namespace {
/// Shifts every reference to the closure's own parameters and implicit origins
/// up by one, making room for the `_Self` parameter and `mut self` argument the
/// universal parametric closure trait prepends. This is the index-reference
/// analogue of `IndexRefRemapper`'s `offset` (which only covers parameter
/// refs), extended to implicit-origin refs.
struct SelfPrependShifter : IndexParameterReplacer<SelfPrependShifter> {
  Attribute tryReplace(Attribute attr, size_t depth) {
    if (auto ref = dyn_cast<ParamIndexRefAttr>(attr);
        ref && ref.getDepth() == depth)
      return ParamIndexRefAttr::get(depth, ref.getIndex() + 1,
                                    this->replaceImpl(ref.getType(), depth));
    if (auto ref = dyn_cast<ImplicitOriginRefAttr>(attr);
        ref && ref.getDepth() == depth)
      return ImplicitOriginRefAttr::get(depth, ref.getIndex() + 1,
                                        ref.getType());
    return nullptr;
  }
  Type tryReplace(Type, size_t) { return {}; }
};
} // namespace

TraitSymbolAttr
IREmitter::bindParamsToClosureTraitFromSig(FnTypeGeneratorType sig) {
  MLIRContext *ctx = shared.getContext();
  ASTDecl *closureTraitDecl = shared.getUniversalParametricClosureTrait();

  // The universal closure trait prepends a `_Self` parameter and a `mut self`
  // argument/origin, so shift the closure's own parameter and implicit-origin
  // references up by one to make room. Each component is then "quoted": the
  // quote freezes the (now out-of-scope) references so they can be pulled into
  // the trait's builder without creating an invalid binding -- the builder
  // unquotes them when it folds into the generator type. A parameter's declared
  // type, an argument type and the result type are all encoded this way.
  SelfPrependShifter shifter;
  auto quote = [&](Type type) -> TypedAttr {
    // Canonicalize the type value (as the previous `emitPValue` path did): the
    // closure trait it feeds into must stay canonical.
    return QuoteAttr::get(TypeParamAttr::get(
        getCanonicalType(shifter.replace(type)), TypeType::get(ctx)));
  };

  // NOTE: this has to be in sync with `createParametricClosureTrait`.

  // 1st, the parameter decl list.
  SmallVector<TypedAttr> paramDecls =
      llvm::map_to_vector(sig.getInputParamTypes(), quote);

  auto paramDeclList =
      ParamListAttr::get(paramDecls, ParamListType::get(TypeType::get(ctx)));

  // 2nd, the argument type list.
  SmallVector<TypedAttr> argTypes =
      llvm::map_to_vector(sig.getBody().getArguments(), quote);
  auto argTypeList =
      ParamListAttr::get(argTypes, ParamListType::get(TypeType::get(ctx)));

  // 3rd, the result type.
  auto resultType = quote(sig.getBody().getResultType());

  // 4th, the metadata. Adjust for the extra `mut self` argument (prepend a Mut
  // convention and one implicit origin decl); we can only do it here since we
  // cannot build a parameter expression on the metadata when constructing the
  // alias decl op in the closure trait.
  auto originData = FnMetaOriginDataAttr::get(
      ctx, sig.getFnMetaOriginData().getNumImplicitOriginDecls() + 1,
      sig.getFnMetaOriginData().getCaptureOrigins(),
      sig.getFnMetaOriginData().getIsNestedOriginsReadOnly(),
      sig.getFnMetaOriginData().getDefinesInteriorOrigins());
  SmallVector<ArgConvention> argConventions;
  argConventions.push_back(ArgConvention::Mut);
  llvm::append_range(argConventions, sig.getFnMetadata().getArgConventions());
  auto metadata = FnMetadataAttr::get(
      ctx, argConventions, sig.getFnMetadata().getFnEffects(), originData);

  // Account for `_Self`/`self` inserted when constructing the closure trait.
  // We have to do it here since we can not parametrize a pog_list. (So we can
  // not build a concat(_Self, pog_list)) at the construction site. It would be
  // nicer if we can fully parametrize a pog list construction, but for now,
  // closure trait is the only user, and the pattern is fixed.
  SmallVector<PogMetadataAttr> pogParams;
  pogParams.push_back(PogMetadataAttr::get(StringAttr::get(ctx, "_Self"),
                                           PassingKind::Inferred));
  llvm::append_range(pogParams, sig.getParamListAttrs().getPogs());

  SmallVector<PogMetadataAttr> pogArgs;
  pogArgs.push_back(
      PogMetadataAttr::get(StringAttr::get(ctx, ""), PassingKind::PosOnly));
  llvm::append_range(pogArgs, sig.getArgListAttrs().getPogs());

  auto traitDeclOp = cast<TraitDeclOp>(closureTraitDecl->getIfOperation());
  return traitDeclOp.bindReference({paramDeclList, argTypeList, resultType,
                                    metadata, PogListAttr::get(ctx, pogParams),
                                    PogListAttr::get(ctx, pogArgs)});
}
