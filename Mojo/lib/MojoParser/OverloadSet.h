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
// This file declares support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_CALLEMISSION_H
#define KGEN_MOJOPARSER_CALLEMISSION_H

#include "DeferredTypingContext.h"
#include "ParamBindings.h"

#include "Mojo/Support/TriResult.h"

namespace M::KGEN::LIT {

/// The callee an overload filter selected, or why none was selected.
using CalleeResult = TriResult<PValue>;

/// The callee a value-type overload filter selected, together with the decl it
/// came from. Most callers want one or the other, so they are named rather
/// than positional.
struct SelectedCallee {
  PValue callee;
  ASTDecl *decl;
};

/// What a value-type overload filter selected, or why it selected nothing.
using CalleeAndDeclResult = TriResult<SelectedCallee>;

/// Returns true if `candidate` is a compiler-synthesized function whose own
/// where clause can never be satisfied (e.g. the move/copy-init synthesized
/// for a `Movable`/`Copyable` conformance). Such a function is never actually
/// callable, so it should not be treated as a real candidate when diagnosing a
/// failed call or a failed trait conformance.
bool isNeverCallableSynthesizedCandidate(ASTDecl *candidate);

//===----------------------------------------------------------------------===//
// OverloadSet
//===----------------------------------------------------------------------===//

/// This class represents an unresolved overload set with partially bound
/// callees, e.g. "foo" or "a.foo" where "foo" is an overloaded declaration or
/// an incompletely bound function (e.g. one with result parameters).  This is
/// resolved when emitted to an RValue or when binding more things into it as
/// part of the expression tree.
///
/// Note that it is possible to have an overload set with methods from multiple
/// different self types that are related to each other.  For example when Mojo
/// has classes, it will be common to have super-class methods that expect
/// 'self' to be converted to a different type in order to invoke it.  For
/// nonmaterializable types like IntLiteral, we can have methods on both Int and
/// IntLiteral, etc.  Filtering the overload set will pick the appropriate
/// method.
class OverloadSet {
public:
  /// In a method reference like `x.foo`, this is the base object being invoked,
  /// e.g. `x`.
  ASTExprAnd<AnyValue> baseValue;

  /// This is the basename of the declaration set, used in diagnostics.
  StringRef baseName;

  /// The function overload set that may be called directly.
  SmallVector<ASTDecl *, 1> fnDecls;

  /// Any bound parameters as well as the callExpr overall this represents.
  ParamBindings paramBindings;

  /// This is syntax how this overload set was formed.
  CallSyntax syntax;

  /// If this is a constructor call like T[1](x), this contains the type being
  /// constructed, which may include parameter values like "1".
  ASTType selfResultType;

  /// Facts that hold at this call site but are not reachable from
  /// `paramBindings.declScope`'s lexical assumptions. These are used as
  /// assumptions when discharging candidates' constraints.
  SmallVector<ConstraintAttr> additionalAssumptions;

  /// When doing resolution, we should only raise new errors if previous errors
  /// haven't already been raised about functions in the overload set.  The most
  /// common issue is when one of the included declarations is erroneous.
  /// Emitting further errors about overload resolution failure can then be
  /// spurious, since we can't properly consider the erroneous declarations
  /// which otherwise might match.  This flag guards against raising those extra
  /// errors.
  bool erroneous;

  /// Form an overload set with the specified function overloads and the given
  /// parameter bindings. The parameter bindings are taken ownership of.
  OverloadSet(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
              ParamBindings &&paramBindings, CallSyntax syntax,
              bool erroneous = false);
  OverloadSet(OverloadSet &&other) = default;

  /// Form an OverloadSet with a lookup of a named method on the specified type,
  /// but without the candidate set filtered with operands.   If successful,
  /// this provides a non-null OverloadSet.
  ///
  /// On failure, this returns a null OverloadSet and invokes errorHandler if
  /// the problem hasn't already been diagnosed and it is non-null. This does
  /// not emit an error on failure.
  static OverloadSet lookup(ASTDecl &declScope, ASTType type,
                            StringRef methodName, const ExprNode *callExpr,
                            CallSyntax syntax,
                            function_ref<void()> errorHandler = {});

  /// Lookup of a named method on the specified type, filtered to match a
  /// concrete operand set. If successful, this provides a non-null PValue for a
  /// single callee. If non-null, it invokes lookupFailureErrorHandler if the
  /// lookup of the named method fails.  If that succeeds, it will complain
  /// about overload resolution when 'shouldPrintOverloadErrors' is true.
  ///
  /// NOTE: This can mutate the operand list, e.g. when calling a static method
  /// that doesn't need a self value, and by emitting PValues when not in an
  /// parameter context. The actual emission needs to use the updated argument
  /// list.
  static PValue lookupAndResolve(ASTType type, StringRef methodName,
                                 CallOperands &operands,
                                 function_ref<void()> lookupFailureErrorHandler,
                                 bool shouldPrintOverloadErrors,
                                 IREmitter &emitter);

  /// Same as the above but a convenience when never emitting an error.
  static PValue lookupAndResolve(ASTType type, StringRef methodName,
                                 CallOperands &operands, IREmitter &emitter) {
    return lookupAndResolve(type, methodName, operands, {}, false, emitter);
  }

  bool isNull() const { return fnDecls.empty(); }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  /// An overload set is erroneous primarily when constructed with erroneous
  /// decls.  If an overload set is erroneous, you can't necessarily trust
  /// lookup results when processing to find further errors.
  bool isErroneous() const { return erroneous; }

  ASTDecl &getDeclScope() const { return paramBindings.declScope; }
  SharedState &getShared() const { return paramBindings.shared; }
  const ExprNode *getExpr() const { return paramBindings.getExpr(); }
  SMLoc getExprLoc() const;

  /// Perform substitutions of the specified bindings into the symbol, returning
  /// the resultant LITSymbolConstant attr or producing an error message and
  /// returning null. This allows producing a reference to a parameterized
  /// function without the parameters specified.  They can be bound later.
  TypedAttr getBoundConstantAttr() const;

  /// Evaluate the fnDecls candidates and see if there is an unambiguous
  /// candidate that works with the specified parameter bindings and provided
  /// arguments.  If so, return the single entry that works.
  ///
  /// If not, generate a diagnostic (when `emitDiagnosticOnFailure` is true) and
  /// return null.
  ///
  /// When `forwardedNeedingOrigins` is non-null, instead of materializing the
  /// operands that need origins, simply forward the information to the caller.
  ///
  /// A candidate whose body constraints can be neither proven nor disproven in
  /// this scope is inconclusive: it is not selected, but it is not rejected
  /// either. A `no` is a definitive "no viable candidate"; an `unknown` means
  /// the best surviving candidates were left undecided by this scope, and must
  /// not be read as a definitive rejection.
  ///
  /// NOTE: This can mutate the operand list, e.g. when calling a static method
  /// that doesn't need a self value, and by pre-emitting PValues when not in an
  /// parameter context. The actual emission needs to use the updated argument
  /// list.
  CalleeResult filterOverloadSet(
      CallOperands &operands, bool emitDiagnosticOnFailure, IREmitter &emitter,
      bool disableMaterialization = false,
      OperandsNeedingOriginsList *forwardedNeedingOrigins = nullptr) const;

  /// Evaluate the fnDecls candidates and see if there is an unambiguous
  /// candidate that works with the specified parameter bindings on the overload
  /// set. If so, return the single entry that works.  If not, generate a
  /// diagnostic and return null.
  ///
  /// If `deferredTypingContext` is non-null, body-constraint inconclusiveness
  /// at the single-candidate site is silently accepted and recorded onto the
  /// context instead of raising a hard error.
  PValue filterOverloadSetForParamBindings(
      DeferredTypingContext *deferredTypingContext = nullptr) const;

  /// Try to resolve the overload set to a single function candidate, using the
  /// expected type if provided or using current bindings if an emitter is
  /// provided.  This emits errors if 'emitter' is non-null, but does not if it
  /// is null.
  ///
  /// If `deferredTypingContext` is non-null, body-constraint inconclusiveness
  /// is handled via the context instead of being raised as a hard error.
  PValue
  getDirectSymbol(ASTType expectedType,
                  DeferredTypingContext *deferredTypingContext = nullptr) const;

  /// Try to emit the overload set as a PValue.
  PValue getIfPValue() const;

  /// Emit this as a CValue if it can be resolved, otherwise emit an ambiguity
  /// error and return null.
  CValue emitAsCValue(IREmitter &emitter, ExprDest &dest);

  /// Emit a function call to the specified callee with the specified operand
  /// values.  This emits an error and returns null on failure.
  ///
  /// `callNode` is the call like expression (e.g. a CallNode, binary operator,
  /// etc) that results in the call, or potentially a random value that is being
  /// fed into an implicit conversion.  This should only be used for location
  /// information.
  CValue emitCall(CallOperands &&callOperands, IREmitter &emitter);

  /// Filter down and complete this overload set based on knowledge that we need
  /// to produce a function pointer with the specified type.  This returns a
  /// pair of PValue for the callee (or null if not resolvable) and the selected
  /// method declaration (or null if no unique match).
  ///
  /// A candidate whose body constraints can be neither proven nor disproven in
  /// this scope is inconclusive: it cannot be selected, but neither can it be
  /// ruled out, so no other candidate may be selected over it either. When
  /// `emitError` is set this is diagnosed like any other inconclusive overload
  /// set; otherwise the `unknown` answer reports the same situation to callers
  /// that probe silently and must account for every candidate that survives
  /// (e.g. witness selection).
  CalleeAndDeclResult filterOverloadSetForValueType(
      ASTType functionType,
      function_ref<MojoInflightDiag &(llvm::SMLoc)> emitError) const;

  /// If the specified type can be constructed with the specified operands
  /// return the initializer that would be invoked. If not, return a `no`
  /// answer. If there were erroneous declarations when processing return
  /// failure so we don't indicate downstream errors.
  ///
  /// If there were erroneous declarations, an error has been raised about a
  /// constructor that likely would have applied, which should be considered in
  /// any error reporting. This does not generate any IR.
  ///
  /// This allows implicit conversions of operands so long as "operands" doesn't
  /// indicate that this is itself an implicit conversion.
  ///
  /// A `no` means no initializer was selected, a definitive "cannot
  /// construct". An `unknown` means a candidate was left inconclusive because
  /// its body constraints could be neither proven nor disproven in
  /// `declScope`; it must not be read as a definitive answer.
  static FailureOr<CalleeResult> canConstructType(
      ASTType requiredType, CallOperands &operands, ASTDecl &declScope,
      OperandsNeedingOriginsList *forwardedNeedingOrigins = nullptr);

  LLVM_DUMP_METHOD void dump() const;

private:
  OverloadSet(ASTDecl &declScope, const ExprNode *expr, CallSyntax syntax,
              bool erroneous)
      : paramBindings(declScope, expr), syntax(syntax), erroneous(erroneous) {}
};

/// This provides a wrapper around OverloadSet which is reference counted,
/// allowing OverloadSetUValue to maintain it while still being copyable.
struct OverloadSetUValue::OverloadSetWrapper
    : public NonAtomicallyReferenceCounted<OverloadSetWrapper> {

  OverloadSetWrapper(OverloadSet &&overloadSet)
      : overloadSet(std::move(overloadSet)) {}
  OverloadSet overloadSet;
};

//===----------------------------------------------------------------------===//
// OverloadSetUValue implementation details
//===----------------------------------------------------------------------===//

template <typename... Args>
inline OverloadSetUValue OverloadSetUValue::create(Args &&...args) {
  return OverloadSetUValue(takeRCRef(
      new OverloadSetWrapper(OverloadSet(std::forward<Args>(args)...))));
}

inline const OverloadSet &OverloadSetUValue::operator*() const {
  return storage.getPointer()->overloadSet;
}

inline OverloadSet &OverloadSetUValue::operator*() {
  return storage.getPointer()->overloadSet;
}

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_CALLEMISSION_H
