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
// This file defines the ExprContext enum and the ExprDest class.  ExprContext
// is used to indicate the context in which an expression is being emitted.
// ExprDest indicates the location an expression is being emitted into
// (including the ExprContext).
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_EXPRESTDEST_H
#define KGEN_MOJOPARSER_EXPRESTDEST_H

#include "Mojo/MojoParser/IRValues.h"

namespace M::KGEN::LIT {
using llvm::SMLoc;
class IREmitter;
class VarDeclOp;

//===----------------------------------------------------------------------===//
// ExprContext
//===----------------------------------------------------------------------===//

/// This enum is used to pass down a bit of context information to make
/// diagnostics more specific.  Each comment gives an example where the
/// expression is named "x".
enum ExprContext {
  EC_InvalidContext,       // Not a valid context, will abort.
  EC_VarInit,              // var thing = x
  EC_Assignment,           // y = x
  EC_Type,                 // var v : x         (and many other places)
  EC_AttributeRefBase,     // x.field
  EC_AliasValue,           // alias something = x
  EC_CallArgValue,         // foo(x)
  EC_CallArgDefaultValue,  // def foo(x = 10) where x uses default value
  EC_CallRefArgValue,      // foo(x) where x is passed by 'ref'.
  EC_CallCalleeValue,      // x()
  EC_TypeParamValue,       // Vector[x]
  EC_CallParamValue,       // f[x]()
  EC_OperatorOperandValue, // x + y
  EC_InplaceBinOpDest,     // x += 42
  EC_TypePattern,          // (x) : Type
  EC_FieldInitValue,       // SomeType{value: x}
  EC_DefaultArgument,      // def f(arg = x):
  EC_VarArgArgument,       // def f(x: *Int):    -> creation of VariadicList.
  EC_PackArgument,         // def f[..](*x: *Ts) -> creation of VariadicPack
  EC_KWArgsArgument,       // def f(x: **Int):   -> creation of KWArgs dict
  EC_DefaultParam,         // def f[p: Int = x]():
  EC_BoolCondition,        // if x  /  while x  /  x and y  /  a if x else b
  EC_CondExpr,             // x if a else y
  EC_ComptimeIfCondition,  // comptime if x
  EC_ComptimeForSeq,       // comptime for y in x
  EC_ForIterator,          // for x internal details
  EC_WithContextMgr,       // with x:
  EC_WithExitResult,       // with (result of __exit__ call)
  EC_MatchSubject,         // match x:
  EC_ComptimeAssert,       // comptime assert expression
  EC_RaiseValue,           // raise x
  EC_ReturnValue,          // return x;
  EC_Requires,             // requires x
  EC_MLIRMagic,            // __mlir_type[x] / __mlir_attr[x]
  EC_TopLevelStmt,         // x
  EC_CollectionLiteral,    // [x, y], {x:y, q:r}, {x, y, z}
  EC_CollectionCompElt,    // [x for x in y]
  EC_TupleElement,         // (x, y)
  EC_SubscriptBase,        // x[y]
  EC_Subscript,            // y[x]
  EC_SliceIndex,           // y[:x:]
  EC_ParameterList,        // something[x]
  EC_Destructor,           // Looking up T's destructor for `var x : T`
  EC_Capture,              // def f(): var x = 4; def nested(): use(x)
  EC_Decorator,            // @x
  EC_Trait,                // trait conformance checking for `T`
  EC_Closure,              // closure formation
  EC_Origin,               // origin specifier
  EC_TypeOf,               // type_of(x)
  EC_ConformsTo,           // conforms_to
  EC_FunctionsInModule,    // __functions_in_module()
  EC_PyBindGen,            // within Python binding generation
  EC_MergeWith,            // implicit __merge_with__ call
  EC_RefBinding,           // ref r = x
  EC_SynthesizedMethod,    // synthesized method call
  EC_ConversionThunk,      // conversion thunk call
  EC_OverloadResolution,   // internal overload resolution error
};
const char *getContextMessage(ExprContext context);

//===----------------------------------------------------------------------===//
// ExprDest
//===----------------------------------------------------------------------===//

/// This is used in ExprDest when emitting an LValue expression whose type may
/// be inferred from the RHS value in an assignment.  This allows implicitly
/// declared variables and discard patterns to infer their type in `_ = foo()`.
struct LValueInitializerType {
  ASTType type;
};

/// This is a marker type to indicate when a ExprDest has had its internal
/// LValue taken by getLValueForResult, but which hasn't had an emitResult to
/// write it back yet.
struct LValueBufferTaken {};

/// This struct is used to represent an LValue with a contextual type,
/// The contextual type could be only partially bound type, and a concrete type
/// will be inferred from the RHS expression. This supports:
///
/// struct PartiallyBoundParam[T: AnyType]:
///   @implicit
///   def __init__(out self: PartiallyBoundParam[Self.T],  x:Self.T):
///     pass

/// def infer_partially_bound_var_type():
///   var v: PartiallyBoundParam = 1
struct LValueContextualType {
  ASTType type;
  ExprNode *expr;
};

/// This class represents the destination context that an expression is being
/// emitted in, when it may produce an RValue.  Example destinations include:
///   - an LValue:
///       This handles cases like `a.b = 42` or `var x: Int = 42`, as well as
///       a return slot with memory-only results in `return x()`.  In this
///       case, the emitted expression must conform to type of the LValue.
///   - an untyped let/var decl, e.g. `var x = 42`
///       In this case, the ExprNode conforms to the initializer expression.
///   - an ExprNode:
///       This handles assignments to "targets" (Python nomenclature), e.g.:
///          1) a discard pattern, e.g. `_ = 42`
///          2) an implicitly declared var decl, e.g. `x = 42` in a def.
///          3) tuples and lists thereof, e.g. `(a, _) = foo()`
///       In this case, the ExprNode type often conforms to the expression. This
///       is used in "x = ..." and "for x in ..." etc. For target emission,
///       the ExprDest tracks contextual information about whether the dest is
///       wrapped by a `var` or `ref` context.
///   - an RValue type:
///       This indicates that the result may be treated in any way (e.g. dumping
///       into a temporary memory location as an MRValue or returned in an SSA
///       register as an SRValue) but needs to have the specified RValue type.
///   - LValueInitializerType:
///       This is used when emitting LValues when there is an inferred type for
///       the LValue, e.g. in `_ = foo()`.
///
/// Any expression may also have no proscribed result (as in the case of
/// `someExpr()` with an ignored result), in which case emission will create
/// storage when needed on demand.
class ExprDest {
public:
  /*implicit*/
  ExprDest(ExprContext context)
      : representation(NullRepresentation()), context(context) {}

  // The destination is some unemitted expression.
  ExprDest(const ExprNode *expr, ExprContext context)
      : representation(expr), context(context) {
    assert(expr && "Cannot init with null ExprNode*");
  }

  ExprDest(LValue dest, ExprContext context)
      : representation(dest), context(context) {}
  ExprDest(VarDeclOp dest, ExprContext context);
  ExprDest(ASTType requiredType, ExprContext context)
      : representation(requiredType), context(context) {
    if (!requiredType || isa<TypeCheckErrorType>(requiredType))
      representation = NullRepresentation();
  }
  ExprDest(LValueInitializerType type, ExprContext context)
      : representation(type), context(context) {}
  ExprDest(LValueContextualType dest, ExprContext context)
      : representation(dest), context(context) {
    assert(dest.expr && dest.type &&
           "Cannot init with null ExprNode* or ASTType");
  }
  ExprDest(ExprDest &&rhs)
      : representation(std::move(rhs.representation)), context(rhs.context),
        patternDeclKind(rhs.patternDeclKind) {
    rhs.representation = NullRepresentation();
    rhs.patternDeclKind = PatternDeclKind::kNone;
  }
  ExprDest &operator=(ExprDest &&rhs) {
    representation = std::move(rhs.representation);
    patternDeclKind = rhs.patternDeclKind;
    context = rhs.context;
    rhs.representation = NullRepresentation();
    rhs.patternDeclKind = PatternDeclKind::kNone;
    return *this;
  }

  ExprDest(const ExprDest &) = delete;
  ExprDest &operator=(const ExprDest &) = delete;

  ~ExprDest() {
    assert(!isSpecified() && "ExprDest destroyed without being emitted into");
  }

  /// This returns the context the expression is getting emitted into (for
  /// diagnostic QoI purposes).
  ExprContext getContext() const { return context; }
  PatternDeclKind getPatternDeclKind() const { return patternDeclKind; }
  void setPatternDeclKind(PatternDeclKind kind) { patternDeclKind = kind; }

  /// Return true if there is a specification for this destination.  If not,
  /// an expression will be emitted to generate a PValue, SRValue, LValue, etc.
  bool isSpecified() const { return !isa<NullRepresentation>(representation); }
  bool isOperation() const { return isa<Operation *>(representation); }

  /// Return true if this ExprDest has an already assigned memory location, or
  /// false if none is specified.  False means that emitting to this destination
  /// will produce an anonymous temporary whenever the ExprDest is ultimately
  /// assigned to.
  bool hasExistingMemoryDest() const {
    // These will not produce a temporary when assigned to.
    return isa<LValue, Operation *>(representation);
  }

  /// Return the LValueInitializerType this contains if it is one.
  ASTType getIfInitializerType() const {
    // LPValueInitializer does not support null test, test isa first and then do
    // a cast.
    if (isa<LValueInitializerType>(representation))
      return cast<LValueInitializerType>(representation).type;
    return {};
  }

  /// Return the ExprNode for the speculatively generated lvalue.
  const ExprNode *getLValueExprNode() const {
    if (isa<const ExprNode *>(representation))
      return cast<const ExprNode *>(representation);
    if (isa<LValueContextualType>(representation))
      return cast<LValueContextualType>(representation).expr;
    return nullptr;
  }

  /// If this indicates an explicit expected RValue type, return that type.
  ASTType getExpectedTypeIfSpecified() const;

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
  ASTType resolveImpliedType(SMLoc loc, Type existingValueType,
                             IREmitter &emitter);

  /// Project a ExprDest into an lvalue with the specified underlying (RValue)
  /// type.
  ///
  /// When `allowIncompatibleTypes` is true, the method is allowed to return an
  /// LValue of a different type when the underlying storage requires this. This
  /// is a guarantee from the caller that it is prepared to handle a type
  /// conversion on its side, eliminating a temporary buffer in
  /// register-passable cases like `var x : F32 = 1`.
  ///
  /// When `allowIncompatibleTypes` is false, this always returns an LValue of
  /// the requested type, which may return a temporary buffer.  In this case it
  /// will not consume the ExprDest, so any user should reemit the ultimate
  /// value through it with emitResult.
  LValue getLValueForResult(SMLoc loc, ASTType resultType,
                            bool allowIncompatibleTypes, bool requireMLValue,
                            IREmitter &emitter);

  /// Return an MLValue for this destination of the specified type that we can
  /// initialize.  This uses and consumes the destination if it matches the type
  /// of the value dest.
  MLValue getMLValueForResult(SMLoc loc, ASTType resultType,
                              IREmitter &emitter);

  /// If this ExprDest specifies an MLValue that will be returned by
  /// getMLValueForResult with the specified type, return it. Otherwise return
  /// null.
  MLValue getDefinedMLValueIfExists(ASTType resultType, IREmitter &emitter);

  /// If this destination is already a concrete `MLValue` (for example
  /// assignment into an existing `ref` binding), or a VarDecl that is getting
  /// inferred return it.
  ///
  /// BEWARE: in a situation like "var x = foo()" this will return the MLValue
  /// for the VarDecl, which will have an UnresolvedType element type.
  MLValue getDirectMLValueIfPresent() const;

  /// Return true if this is an MLValue that could be in a non-default address
  /// space.
  bool isNonDefaultAddressSpace() const;

  /// When an error is emitted instead of generating IR, this method resets the
  /// ExprDest so it doesn't complain when emission is done.
  void resetForError(IREmitter &emitter);

  void dump() const;

private:
  //  This should only be accessed by IREmitter::emitResult.
  friend class IREmitter;
  SmartVariant<NullRepresentation, LValue, LValueBufferTaken, const ExprNode *,
               Operation *, ASTType, LValueInitializerType,
               LValueContextualType>
      representation;
  ExprContext context;

  /// This enum value keeps track of whether a "target" is being emitted in a
  /// var or ref wrapper.  This affects the behavior of a synthesized VarDecl:
  PatternDeclKind patternDeclKind = PatternDeclKind::kNone;

  friend raw_ostream &operator<<(raw_ostream &os, const ExprDest &value);
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_EXPRESTDEST_H
