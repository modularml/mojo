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
// This file implements the ExprContext enum and the ExprDest class.
//
//===----------------------------------------------------------------------===//

#include "Mojo/MojoParser/ExprDest.h"
#include "IREmitter.h"
#include "Mojo/MojoParser/ExprNode.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// ExprContext
//===----------------------------------------------------------------------===//

const char *LIT::getContextMessage(ExprContext context) {
  switch (context) {
  case EC_InvalidContext:
    assert(0 && "cannot emit an invalid context");
    return "";
  case EC_VarInit:
    return " in 'var' initializer";
  case EC_Assignment:
    return " in assignment";
  case EC_Type:
    return " in type specification";
  case EC_AttributeRefBase:
    return " in attribute base value";
  case EC_AliasValue:
    return " in comptime initializer";
  case EC_CallArgValue:
    return " in call argument";
  case EC_CallArgDefaultValue:
    return " in default call argument";
  case EC_CallRefArgValue:
    return " in 'ref' argument";
  case EC_CallCalleeValue:
    return " in callee";
  case EC_TypeParamValue:
    return " in type parameter";
  case EC_CallParamValue:
    return " in call parameter";
  case EC_OperatorOperandValue:
    return " in operator argument";
  case EC_InplaceBinOpDest:
    return " for in-place operator destination";
  case EC_TypePattern:
    return " in type pattern";
  case EC_FieldInitValue:
    return " in field initializer";
  case EC_DefaultArgument:
    return " in default argument";
  case EC_VarArgArgument:
    return " in vararg argument compiler implementation internals";
  case EC_PackArgument:
    return " in variadic pack argument compiler implementation internals";
  case EC_KWArgsArgument:
    return " in keyword arguments dict compiler implementation internals";
  case EC_DefaultParam:
    return " in default parameter";
  case EC_BoolCondition:
    return " in boolean condition";
  case EC_CondExpr:
    return " in 'if' expression value";
  case EC_ComptimeIfCondition:
    return " in 'comptime if' condition";
  case EC_ComptimeForSeq:
    return " in 'comptime for' sequence initializer";
  case EC_ForIterator:
    return " in 'for' iterator expression";
  case EC_WithContextMgr:
    return " in 'with' context manager";
  case EC_WithExitResult:
    return " in 'with' call to '__exit__' on context manager";
  case EC_MatchSubject:
    return " in 'match' subject";
  case EC_ComptimeAssert:
    return " in 'comptime assert' expression";
  case EC_RaiseValue:
    return " in raised value";
  case EC_ReturnValue:
    return " in return value";
  case EC_Requires:
    return " in 'requires' clause";
  case EC_MLIRMagic:
    return " in MLIR magic";
  case EC_TopLevelStmt:
    return " in expression statement";
  case EC_CollectionLiteral: // [x, y], {x:y, q:r}
    return " in collection literal";
  case EC_CollectionCompElt: // [x for x in y]
    return " in comprehension expression";
  case EC_TupleElement: // (x, y)
    return " in tuple element";
  case EC_SubscriptBase: // x[y]
    return " in subscript base";
  case EC_Subscript: // y[x]
    return " in subscript";
  case EC_SliceIndex: // y[:x:]
    return " in slice index";
  case EC_ParameterList: // something[paramValue]
    return " in parameter list";
  case EC_Destructor:
    return " in '__del__' resolution";
  case EC_Capture:
    return " in capture";
  case EC_Decorator:
    return " in decorator";
  case EC_Trait:
    return " in trait conformance checking";
  case EC_Closure:
    return " in internal closure formation";
  case EC_Origin:
    return " in origin specifier";
  case EC_TypeOf:
    return " in type_of";
  case EC_ConformsTo:
    return " in conforms_to";
  case EC_FunctionsInModule:
    return " in __functions_in_module";
  case EC_PyBindGen:
    return " in Python binding generation";
  case EC_MergeWith:
    return " in implicit '__merge_with__' call";
  case EC_RefBinding:
    return " in 'ref' binding";
  case EC_SynthesizedMethod:
    return " in synthesized method";
  case EC_ConversionThunk:
    return " in synthesized conversion thunk call";
  case EC_OverloadResolution:
    return " in overload resolution (INTERNAL ERROR)";
  }
  llvm_unreachable("invalid expr context");
}

//===----------------------------------------------------------------------===//
// ExprDest
//===----------------------------------------------------------------------===//

ExprDest::ExprDest(VarDeclOp dest, ExprContext context)
    : representation(dest.getOperation()), context(context) {}

void ExprDest::dump() const { llvm::errs() << *this; }

[[maybe_unused]] raw_ostream &LIT::operator<<(raw_ostream &os,
                                              const ExprDest &value) {
  os << "ExprDest context=" << (int)value.context << " destination = ";

  auto &representation = value.representation;
  if (isa<NullRepresentation>(representation)) {
    os << "NullRepresentation";
  } else if (auto lv = dyn_cast<LValue>(representation)) {
    os << "LValue: ";
    lv.dump();
  } else if (isa<LValueBufferTaken>(representation)) {
    os << "LValueBufferTaken";
  } else if (auto expr = dyn_cast<const ExprNode *>(representation)) {
    os << "ExprNode: ";
    expr->print(os);
  } else if (auto *op = dyn_cast<Operation *>(representation)) {
    os << "Operation*: " << *op;
  } else if (auto type = dyn_cast<ASTType>(representation)) {
    os << "ASTType: " << type;
  } else if (isa<LValueInitializerType>(representation)) {
    os << "LValueInitializerType: "
       << cast<LValueInitializerType>(representation).type;
  } else if (isa<LValueContextualType>(representation)) {
    os << "LValueContextualType: "
       << cast<LValueContextualType>(representation).type << "\n ExprNode: ";
    cast<LValueContextualType>(representation).expr->print(os);
  } else {
    os << "UNKNOWN VALUE DEST!";
  }

  os << " patternDeclKind=";
  switch (value.getPatternDeclKind()) {
  case PatternDeclKind::kNone:
    os << "kNone";
    break;
  case PatternDeclKind::kVar:
    os << "kVar";
    break;
  case PatternDeclKind::kRef:
    os << "kRef";
    break;
  case PatternDeclKind::kBind:
    os << "kBind";
    break;
  }

  os << '\n';
  return os;
}

/// If this indicates an explicit expected RValue type, return that type.
ASTType ExprDest::getExpectedTypeIfSpecified() const {
  // These have no implied type.
  if (isa<NullRepresentation, LValueBufferTaken, Operation *, const ExprNode *>(
          representation))
    return {};

  // If we just have a contextual type, return it.
  if (ASTType type = dyn_cast<ASTType>(representation))
    return type;
  if (isa<LValueInitializerType>(representation))
    return cast<LValueInitializerType>(representation).type;
  if (isa<LValueContextualType>(representation))
    return cast<LValueContextualType>(representation).type;
  return cast<LValue>(representation).getRValueType();
}

/// When an error is emitted instead of generating IR, this method resets the
/// ExprDest so it doesn't complain when emission is done.
void ExprDest::resetForError(IREmitter &emitter) {
  // We generally just abandon this ExprDest, but if this was set up to
  // initialize something that could infer types, we need to infer them to
  // TypeCheckErrorType to avoid downstream errors using whatever we failed to
  // initialize.

  if (auto *opDest = dyn_cast<Operation *>(representation)) {
    if (auto varOp = dyn_cast<VarDeclOp>(opDest)) {
      assert(isa<UnresolvedType>(varOp.getType().getElementType()) &&
             "Cannot resolve an already-resolved vardecl");
      varOp.getResult().setType(varOp.getType().getWithElement(
          emitter.shared.getTypeCheckErrorType()));
    }
  } else if (auto target = getLValueExprNode()) {
    // If emitting the RHS failed, use a "type check error" expression as the
    // RHS so we can make sure to emit any vars declared, to silence
    // downstream errors.
    //     var x = <bad>
    //     use(x)  # Don't warn here.
    ExprDest dest(LValueInitializerType{emitter.shared.getTypeCheckErrorType()},
                  getContext());
    (void)emitter.emitExprLValue(target, dest);
  }

  representation = NullRepresentation();
}

/// Inspect the ExprDest to see if it implies a specific type for the value
/// being computed, emitting ExprNode targets if present to get their implied
/// type if present.  This returns null if there is no implied type.
///
/// This may be used in concrete value context with a known type (in which
/// case 'existingValueType' will hold the known value type) or in ambiguous
/// cases where this is being used to resolve a type (in which case it will be
/// null).
///
/// Note that this will mutate the ExprDest if it is an ExprNode, turning it
/// into an LValue to store to.
ASTType ExprDest::resolveImpliedType(SMLoc loc, Type existingValueType,
                                     IREmitter &emitter) {
  // These have no implied type.
  if (isa<NullRepresentation, LValueBufferTaken, Operation *>(representation))
    return {};

  // If we just have a contextual type, return it.
  if (ASTType type = dyn_cast<ASTType>(representation))
    return type;

  assert(!isa<LValueInitializerType>(representation) &&
         "LValueInitializerType should be resolved before this");

  // If we have an un-emitted expression, emit it using our existingValueType to
  // get an LValue.
  if (auto expr = getLValueExprNode()) {
    // If we have a contextual type available, pass that down to the emitter so
    // implicitly declared variables and discard patterns can know their type.
    if (!existingValueType) {
      if (isa<LValueContextualType>(representation))
        return cast<LValueContextualType>(representation).type;
      return {};
    }

    if (ASTType nmTarget = ASTType(existingValueType)
                               .getNonmaterializableTarget(emitter.shared))
      existingValueType = nmTarget;

    // The concrete existing value type must match the partially bound type.
    if (isa<LValueContextualType>(representation)) {
      auto ctxType = cast<LValueContextualType>(representation);

      if (ctxType.type.hasUnboundParameters()) {
        if (!ctxType.type.isEqualAllowingUnbound(existingValueType,
                                                 emitter.shared))
          // The context type is partially bound, and the existing type does not
          // simply match it. We need an extra implicit conversion.
          return ctxType.type;
      } else {
        // If the context type is concrete, this must be the lvalue type we
        // are going to be emitting.
        existingValueType = ctxType.type;
      }
    }

    // Propagate var/ref context (if any) into the generated declarations.
    ExprDest dest(LValueInitializerType{existingValueType}, context);
    dest.patternDeclKind = patternDeclKind;

    // Emit the target as an LValue to understand what we're assigning into.
    LValue exprLValue = emitter.emitExprLValue(expr, dest);
    if (!exprLValue) { // Error already emitted.
      representation = NullRepresentation();
      return {};
    }
    representation = exprLValue;
  }

  // We must have an LValue at this point.
  auto lvalue = cast<LValue>(representation);

  // If this is a "bind" operation (e.g. in a for stmt) infer the type of the
  // var decl from the assignment and yield the MLValue.
  if (RLValue rlValue = lvalue.getIfRLValue()) {
    // Unbound 'bind' values will have two layers of !lit.ref on the RValue
    // type.
    VarDeclOp refOp = cast<VarDeclOp>(rlValue.getDefiningOp());
    if (refOp.getKind() == VarDeclKind::Bind)
      return cast<RefType>(refOp.getType().getElementType()).getElementType();
  }

  // If we have an lvalue already specified, return it.
  return lvalue.getRValueType();
}

/// If this ExprDest specifies an MLValue that will be returned by
/// getMLValueForResult with the specified type, return it. Otherwise return
/// null.
///
/// NOTE: This needs to be kept in sync with getLValueForResult.
MLValue ExprDest::getDefinedMLValueIfExists(ASTType resultType,
                                            IREmitter &emitter) {
  // Handle inference of a 'var' declaration's type.
  if (auto *opDest = dyn_cast<Operation *>(representation)) {
    // If the result type has a nonmaterializable type, then we infer the var
    // to its materialized type.
    ASTType nmTarget = resultType.getNonmaterializableTarget(emitter.shared);
    ASTType materializedType = nmTarget ? nmTarget : resultType;

    auto varOp = cast<VarDeclOp>(opDest);
    assert(isa<UnresolvedType>(varOp.getType().getElementType()) &&
           "Cannot resolve an already-resolved vardecl");
    varOp.getResult().setType(varOp.getType().getWithElement(materializedType));

    // Now that we inferred the 'var' type, we can treat this like a normal
    // MLValue.
    representation = LValue(MLValue(varOp.getResult()));
  }

  // If we have an uncollapsed expression, emit it to learn more about it.
  if (const ExprNode *target = getLValueExprNode()) {
    // The concrete existing value type must match the partially bound type.
    if (isa<LValueContextualType>(representation)) {
      auto cxtType = cast<LValueContextualType>(representation);
      if (cxtType.type.hasUnboundParameters()) {
        // can not directly use the dest as a lvalue, need a conversion.
        if (!cxtType.type.isEqualAllowingUnbound(resultType, emitter.shared))
          return {};
        // Else, partially bound contextual type, but the existing type is a
        // simple match, just take the type.
      } else {
        // Context contextual type, this must be the lvalue type we are going
        // to be emitting.
        resultType = cxtType.type;
      }
    }

    ExprDest dest(LValueInitializerType{resultType}, getContext());
    dest.patternDeclKind = patternDeclKind;
    if (LValue lValue = emitter.emitExprLValue(target, dest)) {
      representation = lValue;
    } else {
      dest.resetForError(emitter);
      representation = NullRepresentation(); // Error already emitted!
    }
  }

  // Check for the simple case.
  if (LValue lValue = dyn_cast<LValue>(representation)) {
    if (MLValue refValue = lValue.getIfMLValue()) {
      if (emitter.canZeroCostConvert(lValue.getRValueType(), resultType)
              .isTrue())
        return refValue;
    }

    // If this is a "bind" operation (e.g. in a for stmt) infer the type of the
    // var decl from the assignment and yield the MLValue.
    if (RLValue rlValue = lValue.getIfRLValue()) {
      VarDeclOp refOp = cast<VarDeclOp>(rlValue.getDefiningOp());
      if (refOp.getKind() == VarDeclKind::Bind) {
        refOp.setKind(VarDeclKind::Bound);
        refOp.getResult().setType(refOp.getType().getWithElement(resultType));

        // Type refinement is applied in emitStoreToLValue after the store
        // completes, not here during type inference.

        representation = LValue(MLValue(refOp));
        return MLValue(refOp);
      }
    }
  }

  // Otherwise, this would create a new buffer.
  return {};
}

MLValue ExprDest::getDirectMLValueIfPresent() const {
  // If this is an already obvious MLValue, return it.
  if (LValue lValue = dyn_cast<LValue>(representation))
    if (MLValue refValue = lValue.getIfMLValue())
      return refValue;

  // An unresolved VarDeclOp.  Supporting this allows inferring the address
  // space and origin even though it has no known element type.
  if (auto *opDest = dyn_cast<Operation *>(representation))
    return MLValue(cast<VarDeclOp>(opDest));
  return {};
}

/// Project a ExprDest into an lvalue with the specified underlying (RValue)
/// type.
///
/// When `allowIncompatibleTypes` is true, the method is allowed to return an
/// LValue of a different type when the underlying storage requires this. This
/// is a guarantee from the caller that it is prepared to handle a type
/// conversion on its side, eliminating a temporary buffer in register-passable
/// cases like `var x : Float32 = 1`.
///
/// When `allowIncompatibleTypes` is false, this always returns an LValue of
/// the requested type, which may return a temporary buffer.  In this case it
/// will not consume the ExprDest, so any user should reemit the ultimate
/// value through it with emitResult.
///
/// NOTE: This needs to be kept in sync with getDefinedMLValueIfExists.
LValue ExprDest::getLValueForResult(SMLoc loc, ASTType resultType,
                                    bool allowIncompatibleTypes,
                                    bool requireMLValue, IREmitter &emitter) {
  // Handle inference of a 'var' declaration's type.
  if (auto *opDest = dyn_cast<Operation *>(representation)) {
    // If the result type has a nonmaterializable type, then we infer the var
    // to its materialized type.
    ASTType nmTarget = resultType.getNonmaterializableTarget(emitter.shared);
    ASTType materializedType = nmTarget ? nmTarget : resultType;

    auto varOp = cast<VarDeclOp>(opDest);
    assert(isa<UnresolvedType>(varOp.getType().getElementType()) &&
           "Cannot resolve an already-resolved vardecl");
    varOp.getResult().setType(varOp.getType().getWithElement(materializedType));

    // Now that we inferred the 'var' type, we can treat this like a normal
    // MLValue.
    representation = LValue(MLValue(varOp.getResult()));
  }

  // We have several cases where we can produce an LValue but it may have the
  // wrong type.  The client may be cool with this (when allowIncompatibleTypes
  // is true), but if not we generate a new temporary buffer.

  // If we have an expression node destination, then we need to bind this
  // value to a pattern (aka "target" in Python internals nomenclature).
  if (isa<const ExprNode *>(representation) ||
      isa<LValueContextualType>(representation)) {
    // resolveImpliedType will resolve ExprNode destinations into LValues.
    (void)resolveImpliedType(loc, resultType, emitter);

    if (LValue lValue = dyn_cast<LValue>(representation)) {
      if (RLValue rlValue = lValue.getIfRLValue()) {
        VarDeclOp refOp = cast<VarDeclOp>(rlValue.getDefiningOp());
        // If this is a "bind" operation (e.g. in a for stmt) then the callee
        // will fill in the produced MLValue, but subsequent accesses will need
        // to treat the value as bound (and therefore immutable).
        if (refOp.getKind() == VarDeclKind::Bind) {
          // Switch the vardecl so that uses of it are treated as MBValue
          // instead of MLValues.
          refOp.setKind(VarDeclKind::Bound);
          refOp.getResult().setType(refOp.getType().getWithElement(resultType));

          // Type refinement is applied in emitStoreToLValue after the store
          // completes, not here during type inference.
          representation = MLValue(refOp);
        }
      }
    }
  }

  // If we have an lvalue already specified, return it.
  if (LValue lValue = dyn_cast<LValue>(representation)) {
    // If asking for a buffer of the type we happen to have, or if the client
    // doesn't care if it matches, then we can directly return it.
    if (allowIncompatibleTypes ||
        emitter.canZeroCostConvert(lValue.getRValueType(), resultType)
            .isTrue()) {
      // If the client accepts any sort of LValue, then we succeed.
      if (!requireMLValue || lValue.getIfMLValue()) {
        representation = LValueBufferTaken(); // Buffer taken!
        return lValue;
      }
    }

    // Otherwise, create a temporary buffer.
  }

  // Finally, if no destination specifies otherwise, we synthesize a new
  // LValue on demand.
  if (!emitter.builder) {
    representation = NullRepresentation();
    bool isRegisterPassable =
        resultType.isRegisterPassable(loc, emitter.shared);
    emitter.emitError(loc, "cannot synthesize lvalue of ")
        << (isRegisterPassable ? "register-passable "
                               : "non-register-passable ")
        << "type " << resultType << getContextMessage(emitter.paramContext);
    return {};
  }

  // If we're generating a memory location, use a required type if present or
  // the value type if not.
  ASTType slotType = resultType;
  if (auto requiredType = dyn_cast_or_null<ASTType>(representation)) {
    if (allowIncompatibleTypes || requiredType.isEqualCanon(slotType))
      slotType = requiredType;
  }

  // We model this as an mutable let value with a separately stored
  // initializer.  We return an LValue for it because this method is used
  // for the initialization.
  return MLValue(emitter.emitVarDecl("anonymous*", slotType,
                                     emitter.translateLocation(loc),
                                     VarDeclKind::Synthesized));
}

/// Return an MLValue for this destination of the specified type that we can
/// initialize. This uses and consumes the destination if it matches the type
/// of the value dest.
MLValue ExprDest::getMLValueForResult(SMLoc loc, ASTType resultType,
                                      IREmitter &emitter) {
  LValue lv =
      getLValueForResult(loc, resultType, /*allowIncompatibleTypes=*/false,
                         /*requireMLValue=*/true, emitter);
  if (!lv)
    return {};

  assert(lv.getIfMLValue());
  return lv.getIfMLValue();
}

/// Return true if this is an MLValue that could be in a non-default address
/// space.
bool ExprDest::isNonDefaultAddressSpace() const {
  if (MLValue mlValue = getDirectMLValueIfPresent())
    return !mlValue.getRefType().isDefaultAddrSpace();
  return false;
}
