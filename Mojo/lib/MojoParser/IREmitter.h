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
// This file defines IREmitter, the main "IR builder" class for the Mojo parser.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_EXPREMITTER_H
#define KGEN_MOJOPARSER_EXPREMITTER_H

#include "DeferredTypingContext.h"
#include "Mojo/MojoParser/ExprDest.h"

#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/MojoParser/Constraints.h"
#include "Mojo/MojoParser/IRValues.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/Support/TriBool.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/SMLoc.h"

#include <variant>

namespace M::KGEN::LIT {
template <typename ValueType>
struct ASTExprAnd;
class IREmitter;
class CallOperands;
class TraitType;

//===----------------------------------------------------------------------===//
// ConversionFailure
//===----------------------------------------------------------------------===//

/// Why an implicit conversion was rejected. `classifyImplicitConversion` tries
/// a sequence of unrelated conversion strategies, and each one fails for its
/// own kind of reason, so each gets its own alternative here.
///
/// This is only used when an answer is definitively `no`. For inconclusive
/// answers, the `ImplicitConversionResult` returned by
/// `classifyImplicitConversionWithDetails` carries the undecided constraints.
class ConversionFailure {
public:
  /// Rejected without recording a reason.
  struct None {};

  /// A constraint was refuted.
  struct RefutedConstraints {
    SmallVector<ConstraintAttr, 2> constraints;
  };

  using Reason = std::variant<None, RefutedConstraints>;

  ConversionFailure() = default;
  ConversionFailure(ConversionFailure &&) = default;
  ConversionFailure &operator=(ConversionFailure &&) = default;

  bool empty() const { return std::holds_alternative<None>(reason); }
  void clear() { reason = None{}; }

  /// Record `newReason`, unless a reason is already recorded. `None` is a
  /// no-op.
  void recordIfEmpty(Reason newReason) {
    if (empty() && !std::holds_alternative<None>(newReason))
      reason = std::move(newReason);
  }

  /// Attach a note per captured reason, consuming it. No-op when empty.
  void addExplanation(MojoInflightDiag &diag) &&;

  /// The recorded reason if it is a `T`, else null. Callers case on the
  /// alternative they can act on; a null result means a different reason was
  /// recorded, or none was, which is not the same as that reason being empty.
  template <typename T>
  const T *getReasonIf() const {
    return std::get_if<T>(&reason);
  }

private:
  Reason reason;
};

//===----------------------------------------------------------------------===//
// IREmitter
//===----------------------------------------------------------------------===//

/// This class is the main helper for expr and stmt emission, providing helper
/// functions for working with IRValues.  This maintains an optional builder
/// when in a dynamic context) as well as a scope to perform lookups against.
class IREmitter : public SharedStateUser {
public:
  /// Create an IREmitter for a dynamic context with a builder.
  IREmitter(ASTDecl &declScope, OpBuilder builder);
  /// Create an IREmitter for a parameter context.
  IREmitter(ASTDecl &declScope, ExprContext paramContext,
            DeferredTypingContext *deferredTypingContext = nullptr);

  /// Get an emitter set up for parameter expressions only with the specified
  /// context.
  IREmitter getParamEmitter(ExprContext context) {
    return IREmitter(declScope, context, deferredTypingContext);
  }

  //===--------------------------------------------------------------------===//
  // Emitter State.

  /// This is the current builder to emit into if we are allowed to generate a
  /// value.  This will be None when in a context that only allows parameters.
  /// It is mutable to support expressions that require internal control flow.
  std::optional<OpBuilder> builder;

  /// When builder is null, this specifies the original reason we're emitting
  /// into a parameter context, e.g. we're in an alias body or type
  /// specification.
  ExprContext paramContext;

  /// This is scope to resolve declaration references against.
  ASTDecl &declScope;

  /// When non-null, body-constraint inconclusiveness during emission is
  /// silently accepted at single-candidate emission sites, and the unprovable
  /// body constraints are inserted into this context for the caller to
  /// discharge later. Per-parameter-constraint inconclusiveness is *not*
  /// affected by this context: those remain hard errors.
  DeferredTypingContext *deferredTypingContext = nullptr;

  /// Return information about the scope we're looking into.
  ASTDecl &getDeclScope() const { return declScope; }

  /// Emit an error about use of a dynamic value (the expression) in a context
  /// that only allows parameter expressions.  This always returns a null
  /// PValue.
  PValue emitErrorForDynamicValueInParameter(const ExprNode *expr,
                                             const char *customMessage = {});
  PValue emitErrorForDynamicValueInParameter(SMLoc loc,
                                             const char *customMessage = {});
  PValue emitErrorForDynamicValueInParameter(Location loc,
                                             const char *customMessage = {});

  PValue bindNonStructTypeToTrait(ASTExprAnd<CValue> value, TraitType trait);

  //===--------------------------------------------------------------------===//
  // Value conversion helpers, handle post-elaborator type equality.
  //

  /// If needed, convert the specified value to the target destination type,
  /// with a noop cast.  This is used to adjust inconsequential details of the
  /// type or for simple things like upcasts.  This does not invoke constructors
  /// or do other non-trivial conversions.
  ///
  /// This produces an error and returns null on an invalid conversion.
  CValue rebindValue(ASTExprAnd<CValue> value, Type destType);

  /// If the type of the specified value differs from the destination type, emit
  /// a rebind operation to convert it.
  Value emitRebindOpIfNeeded(Value value, Type destType, SMLoc loc);

  /// Returns whether a value of the specified type can be coerced to the other
  /// type with a zero-cost conversion like a rebind, meaning values of the two
  /// types have exactly the same representation post-elaboration:
  ///   - `yes`     the conversion is free (including shedding generator body
  ///               constraints that `declScope` proves).
  ///   - `no`      the representations differ, or a generator body constraint
  ///               is refuted in `declScope`.
  ///   - `unknown` a generator body constraint is unprovable in `declScope`.
  ///
  /// `declScope` supplies the `where` assumptions used to discharge generator
  /// body constraints; `additionalAssumptions` are extra facts to consider
  /// alongside it.
  static TriBool
  canZeroCostConvert(ASTType fromType, ASTType toType, SharedState &shared,
                     ASTDecl &declScope,
                     ArrayRef<ConstraintAttr> additionalAssumptions = {});

  /// Same as the above, using this emitter's shared state and scope.
  TriBool
  canZeroCostConvert(ASTType fromType, ASTType toType,
                     ArrayRef<ConstraintAttr> additionalAssumptions = {}) {
    return canZeroCostConvert(fromType, toType, shared, declScope,
                              additionalAssumptions);
  }

  // Returns the upcastability verdict for converting a type value to a trait:
  //   - `yes`     the type value can be upcast to the trait;
  //   - `no`      it definitively cannot;
  //   - `unknown` conformance is unprovable but not contradicted, so the
  //               upcast is potentially satisfiable based on the context.
  // Returns failure for non-applicable cases (i.e., `fromType` is not a
  // typetype and/or `toType` is not a trait type).
  //
  // When `scopeDependent` is non-null, it is set to whether the returned
  // verdict was derived from `declScope`'s `where` assumptions rather than from
  // the `(fromType, toType)` pair alone. An assumption-derived `yes`/`no` is
  // scope-dependent: the same type pair can yield a different verdict in a
  // scope with a different assumption set, so callers that memoize keyed on the
  // type pair must not cache a scope-dependent verdict.

  static FailureOr<TriBool> canMetaTypeUpCastTo(SharedState &shared, SMLoc loc,
                                                ASTType fromType,
                                                ASTType toType,
                                                ASTDecl *declScope,
                                                bool *scopeDependent = nullptr);

  /// Same, additionally collecting the problematic constraints so a diagnosing
  /// caller can name them. An upcast to a trait is decided by a conformance
  /// query, so a rejection reports exactly what that query would have.
  static FailureOr<ConstraintResult>
  canMetaTypeUpCastToWithDetails(SharedState &shared, SMLoc loc,
                                 ASTType fromType, ASTType toType,
                                 ASTDecl *declScope, bool *scopeDependent);

  /// Given a value of a type that can be zero cost converted to another type,
  /// emit a rebind or other operation to get it in the right type.
  static PValue emitZeroCostConvert(PValue value, ASTType toType,
                                    SharedState &shared);
  CValue emitZeroCostConvert(ASTExprAnd<CValue> value, ASTType toType);

  /// Given two values that need to match, try to coerce one to the other if
  /// they disagree on type.  This emits an error (when loc is non-null) and
  /// returns failure if the request is ambiguous or impossible.
  ///
  /// The 'configEmitter' function is called to set the insertion point of the
  /// emitter for the true/false branches of the conditional.
  ///
  /// The 'contextualType' (if specified) is the type the expression is being
  /// emitted into.
  ParseResult
  coerceTypesToEachOther(SMLoc loc, CValue &lhs, const ExprNode *lhsExpr,
                         CValue &rhs, const ExprNode *rhsExpr,
                         std::function<void(bool isLHS)> configEmitter,
                         ASTType contextualType = {});

  /// If there is a common type shared between the two reference types, return
  /// it. Otherwise return null.
  static RefType getCommonRefType(RefType ref1, RefType ref2);

  //===--------------------------------------------------------------------===//
  // Emission helpers for various value classifications.

  /// This emits the value to the specified value dest, transferring ownership
  /// to the destination and returning a reference if dest consumes it, or the
  /// RValue directly if not.

  /// This emits the value to the specified destination as a concrete RValue.
  /// This transfers ownership to the destination, and it will return a
  /// reference if the destination consumes the RValue or the RValue itself if
  /// not. This method will also emit a copy if required to obtain and RValue.
  CValue emitRValue(ASTExprAnd<AnyValue> value, ExprDest &dest);
  RValue emitRValue(ASTExprAnd<AnyValue> value, ExprContext context,
                    ASTType resultType = {});
  CValue emitCValue(ASTExprAnd<AnyValue> value, ExprContext context,
                    ASTType resultType = {});
  CValue emitCValue(ASTExprAnd<AnyValue> value, ExprDest &dest);
  BValue emitBValue(ASTExprAnd<AnyValue> value, ExprDest &dest);
  BValue emitBValue(ASTExprAnd<AnyValue> value, ExprContext context,
                    ASTType resultType = {});
  LValue emitLValue(ASTExprAnd<AnyValue> value, ExprDest &dest);

  /// Emit a register passable PValue to an SRValue.
  SRValue emitPValueToSRValue(ASTExprAnd<PValue> value, ExprContext context);

  /// This helper emits the specified value as a SRValue which has an SSA
  /// value representation, materializing PValues and loading LValues as
  /// needed.  This returns null if emission fails, and should never be used
  /// with values that are memory-only.
  SRValue emitSRValue(ASTExprAnd<AnyValue> value, ExprContext context,
                      ASTType resultType = {});

  /// This helper emits the specified value as an MRValue which has
  /// memory-only representation, materializing PValues as needed. This
  /// returns null if emission fails.
  MRValue emitMRValue(ASTExprAnd<AnyValue> value, ExprContext context,
                      ASTType resultType = {});

  /// This helper emits the specified value as an MBValue which has
  /// memory-only representation, materializing PValues as needed. This
  /// returns null if emission fails.
  MBValue emitMBValue(ASTExprAnd<AnyValue> value, ExprContext context,
                      ASTType resultType = {});

  /// This helper emits the specified expression as a parameter value,
  /// diagnosing the problem if the expression is only valid as a runtime value.
  /// This returns null if emission fails.
  PValue emitPValue(ASTExprAnd<AnyValue> value, ExprContext context,
                    ASTType resultType = {});

  /// This helper emits the specified expression as a 'ref' expression value,
  /// and returns the value of RefType for the result.
  /// This emits an error and returns null if emission fails.
  Value emitRefValue(ASTExprAnd<AnyValue> value, ExprContext context);

  //===--------------------------------------------------------------------===//
  // Function Calls

  /// Emit an indirect call to a resolved value, checking for compatibility and
  /// then generating the call logic.  This emits an error and returns null on
  /// failure.
  CValue emitIndirectCall(CValue callee, CallOperands &&operands);

  /// Emit an indirect call to a resolved value in a try block, invoking a
  /// callback to generate logic in the 'catch' block that is wrapped around the
  /// call. This ensures that the ExprDest is updated and live after the try
  /// block, which only works if the "catch" logic doesn't fall through.
  ///
  /// This emits an error and returns null on failure.
  CValue emitIndirectCallInTryBlock(
      CValue callee, CallOperands &&operands,
      std::function<void(VarDeclOp errDecl)> emitCatchLogic);

  /// This helper emits a named method call with the provided `operands`,
  /// where the first positional operand is the receiver of the call. This emits
  /// an error if the call is invalid and returns null. The `operands` must
  /// contain at least one positional operand.
  ///
  /// `callNode` is the call like expression (e.g. a CallNode, binary operator,
  /// etc) that results in the call, or potentially a random value that is being
  /// fed into an implicit conversion.  This should only be used for location
  /// information.
  CValue emitNamedMethodCall(StringRef methodName, CallOperands &&operands);

  /// Emit a call to __new__ or __init__, returning an instance of the specified
  /// type.  If `allowImplicitConversion` is true, the provided args are allowed
  /// to implicitly convert to the expectations of the constructor signatures.
  CValue emitConstructorCall(ASTType type, CallOperands &&operands);

  /// Convert a CValue string expression into a DataToStr-wrapped parameter
  /// attribute. Handles t-string -> String conversion, StringSpan conversion,
  /// PValue emission, and DataToStr wrapping. Returns a null TypedAttr on
  /// failure.
  TypedAttr emitStringExprAsDataToStr(CValue val, ExprNode *expr, SMLoc loc,
                                      ExprContext context);

  //===--------------------------------------------------------------------===//
  // Type conversion helpers.

  /// Emit a metatype conversion to a trait type by materializing the meta type
  /// of the specified CValue into a witness table for the trait.  For example,
  /// if 'value' has struct type, and the trait is Movable, then this forms a
  /// TypeParamAttr PValue.
  PValue emitMetaTypeToTraitConversion(ASTExprAnd<CValue> value,
                                       TraitType trait);

  /// Emit an upcast of a type value (`valueExpr`) to a trait-typed value
  /// (`toType`).
  /// Returns a null `PValue` on a diagnosed error and `failure()` when the
  /// conversion is not applicable.
  FailureOr<PValue> emitTypeValueUpCastToTrait(ASTExprAnd<CValue> valueExpr,
                                               ASTType toType);

  /// This returns an instance of Tuple[...] with the specified element types
  /// installed.
  ASTType getBuiltinTupleInstantiation(SMLoc loc, ArrayRef<Type> elements);

  //===--------------------------------------------------------------------===//
  // Parametric closure trait helpers.

  /// Create the universal parametric closure trait, a synthetic trait that
  /// any fn signature can be adapted to
  static ASTDecl *createParametricClosureTrait(SharedState &shared);

  /// Bind the universal parametric closure trait's parameters to match the
  /// given function signature, returning the resulting concrete `TraitType`.
  TraitSymbolAttr bindParamsToClosureTraitFromSig(FnTypeGeneratorType sig);
  //===--------------------------------------------------------------------===//
  // Emission helpers for various value classifications.

  /// Emit the specified value into the current destination if present.  This
  /// accepts (and silently propagates) null values.
  ///
  /// Note that the `value` provided here may require an implicit conversion
  /// into the destination slot, so the input may be memory-only and result be
  /// register-passable (and visa-versa).
  AnyValue emitResult(AnyValue value, const ExprNode *expr, ExprDest &dest);
  CValue emitCResult(CValue value, const ExprNode *expr, ExprDest &dest);

  /// Destructing the specific PValue against the provided target expr
  /// (which specifies the pattern).
  LogicalResult emitDestructuringPValue(PValue value,
                                        const ExprNode *targetExpr);

  /// Return true if 'value' may be implicitly converted to 'requiredType'
  /// by invoking (one level of) conversion operations.  This does not generate
  /// any IR.
  /// When `deferralCtx` is non-null and the only obstacle to the conversion is
  /// an unprovable (`unknown`) trait conformance, the conversion is
  /// reported as convertible (and not cached since it's context-dependent).
  static TriBool classifyImplicitConversion(
      ASTExprAnd<CValue> value, ASTType requiredType, ASTDecl &declScope,
      ArrayRef<ConstraintAttr> additionalAssumptions = {},
      DeferredTypingContext *deferralCtx = nullptr);

  /// The answer from `classifyImplicitConversionWithDetails`:
  /// - A `no`/`unknown` carries why the conversion was rejected. Only the
  ///   strategies that know a reason record one, so it can come back empty even
  ///   for a rejection.
  /// - An `unknown` carries the undecided constraints.
  using ImplicitConversionResult =
      TriResult<void, ConversionFailure, SmallVector<ConstraintAttr, 2>>;

  /// Same as `classifyImplicitConversion`, additionally collecting non-yes
  /// reasons.
  static ImplicitConversionResult classifyImplicitConversionWithDetails(
      ASTExprAnd<CValue> value, ASTType requiredType, ASTDecl &declScope,
      ArrayRef<ConstraintAttr> additionalAssumptions,
      DeferredTypingContext *deferralCtx);

  /// Boolean form of `classifyImplicitConversion`, collapsing an undecided
  /// answer the way callers that cannot represent one have always seen it.
  static bool canImplicitlyConvertToType(
      ASTExprAnd<CValue> value, ASTType requiredType, ASTDecl &declScope,
      ArrayRef<ConstraintAttr> additionalAssumptions = {},
      DeferredTypingContext *deferralCtx = nullptr) {
    TriBool verdict = classifyImplicitConversion(
        value, requiredType, declScope, additionalAssumptions, deferralCtx);
    if (verdict.isUnknown())
      return deferralCtx != nullptr;
    return verdict.isTrue();
  }

  /// This emits an implicit conversion to the specified type if the types
  /// differ, including emitting any implicit constructor calls as well as
  /// implicit promotions like origin conversions.
  CValue emitImplicitConversionToType(
      ASTExprAnd<CValue> value, ASTType requiredType, ExprDest &dest,
      ArrayRef<ConstraintAttr> additionalAssumptions = {});

  /// Emit the specified expression into the specified destination.
  AnyValue emitExpr(const ExprNode *expr, ExprDest &dest);

  /// Emit the specified node with the indicated expression context and an
  /// optional contextual type.
  AnyValue emitExpr(const ExprNode *expr, ExprContext context,
                    ASTType resultType = {});

  /// This emits the specified value to a RValue with the specified context.
  RValue emitExprRValue(const ExprNode *expr, ExprContext context,
                        ASTType resultType = {});

  /// This emits the specified value rep as a CValue.
  CValue emitExprCValue(const ExprNode *expr, ExprContext context,
                        ASTType resultType = {});

  /// This helper emits the specified value rep as an SRValue, materializing
  /// it as an operation if it is a parameter.  This returns null if emission
  /// fails.
  SRValue emitExprSRValue(const ExprNode *expr, ExprContext context,
                          ASTType resultType = {});

  /// This helper emits the specified expression as a meta value, and optionally
  /// converts the result to a specified expected type.  This emits an error if
  /// the expression cannot be emitted, if it cannot be converted to the
  /// expected type, or if it isn't a valid runtime value.  This returns null if
  /// emission fails.
  PValue emitExprPValue(const ExprNode *expr, ExprContext context,
                        ASTType resultType = {});

  /// Emit the specified expression as an LValue which can be loaded and stored.
  /// The ExprDest may specify an inferred type for the LValue.
  ///
  /// This diagnoses the expression with the specified message if it isn't a
  /// valid LValue.
  LValue emitExprLValue(const ExprNode *expr, ExprDest &dest);
  LValue emitExprLValue(const ExprNode *expr, ExprContext context) {
    ExprDest dest(context);
    return emitExprLValue(expr, dest);
  }

  /// Emit a copy of the specified value, producing a new owned instance of the
  /// value in the specified destination.  This returns an RValue if
  /// there is no consuming dest, otherwise a BValue.
  CValue emitCopyOfValue(ASTExprAnd<CValue> value, ExprDest &dest);

  /// Given a value with a known type, emit a store to the specified LValue.
  /// This returns an borrowed reference to the value after it is done.
  CValue emitStoreToLValue(ASTExprAnd<CValue> value, LValue destLV,
                           ExprContext context);

  /// Emit IR for the specified expression without adding it to the current
  /// execution context.  This even allows evaluating dynamic expressions in a
  /// parameter context.  When the result is computed, evaluate the specified
  /// callback on the result and then discard the result.
  ///
  /// On failure, an error is emitted and the callback is not invoked.
  ///
  /// This is used for evaluating expressions like `origin_of(x)` and
  /// `type_of(x)` and `ref [x] T`.
  void emitExpressionWithoutEvaluatingIt(
      const ExprNode *expr, ExprContext exprContext,
      std::function<void(CValue, IREmitter &emitter)> callback);

  //===--------------------------------------------------------------------===//
  // Emission helpers for specific value types.

  /// This helper emits the specified expression tree as a type, e.g. turning
  /// "Int" into the type for it.  This emits an error and returns null on
  /// failure. If `allowUnbound` is set, then a type with no bound parameters is
  /// allowed.
  ASTType emitExprType(const ExprNode *expr, bool allowUnbound = false);

  /// This emits the specified PValue as a type, binding defaulted parameters
  /// etc if needed.
  ASTType emitType(ASTExprAnd<PValue> value, bool allowUnbound = false);

  /// Emit the specified expression as a condition, converting it to an MLIR
  /// scalar<bool> value that we can test directly.  This reports and error and
  /// returns null on error.
  RValue emitScalarBool(ASTExprAnd<CValue> value, ExprContext context);

  /// Emit the specified expression as a condition, converting it to an MLIR
  /// scalar<bool> value that we can test directly.  This reports and error and
  /// returns null on error.
  RValue emitExprScalarBool(const ExprNode *condExpr, ExprContext context);

  /// Short-circuiting conjunction of match predicates (`lhs && emitRhs()`).
  ///
  /// When `lhs` is a known-true constant the rhs is emitted directly; when it
  /// is known-false the rhs is skipped unless `evaluateRhsEvenIfLhsFalse` is
  /// set (used by `case` guards so a never-matching pattern still typechecks
  /// its guard). Otherwise this lowers to a nested `hlcf.elif` that yields the
  /// rhs only if `lhs` is true.
  CValue
  emitAndMatchPredicates(ASTExprAnd<CValue> lhs,
                         llvm::function_ref<ASTExprAnd<CValue>()> emitRhs,
                         bool evaluateRhsEvenIfLhsFalse = false);

  /// Short-circuiting disjunction of match predicates (`lhs || emitRhs()`).
  ///
  /// When `lhs` is a known-true constant the rhs is skipped; when it is
  /// known-false the rhs is emitted directly. Otherwise this lowers to a
  /// nested `hlcf.elif` that yields true if `lhs` matches, else the rhs.
  CValue
  emitOrMatchPredicates(ASTExprAnd<CValue> lhs,
                        llvm::function_ref<ASTExprAnd<CValue>()> emitRhs);

  /// Given a value, emit it into an MLIR value by invoking its `__mlir_index__`
  /// method.
  CValue emitIndex(ASTExprAnd<AnyValue> value, ExprContext context);

  /// Emit the specified expression, converting it into an MLIR value by convert
  /// it to an index value and then invoking `__mlir_index__` method.
  CValue emitIndex(const ExprNode *expr, ExprContext context);

  /// Emit a `Bool`-typed value from an `i1` value.
  CValue emitBool(ASTExprAnd<PValue> value, ExprDest &dest);
  CValue emitBool(ASTExprAnd<PValue> value, ExprContext context);

  /// Emit a `Int`-typed value from an `index` value.
  CValue emitInt(ASTExprAnd<AnyValue> indexValue, ExprDest &dest);
  CValue emitInt(ASTExprAnd<AnyValue> indexValue, ExprContext context);

  /// Given an expression that can be used in `origin_of` or a ref expression,
  /// analyze it to determine which origin it represents.  If it doesn't work,
  /// emit an error and return null.
  TypedAttr extractOriginOf(const ExprNode *expr, CValue value);

  /// Given a value of !lit.origin type, return an instance of
  /// Origin[mut, lit.origin]().
  PValue getStdlibOriginOf(TypedAttr litOrigin, SMLoc loc);

  //===--------------------------------------------------------------------===//
  // Statement emission helpers.

  /// Find the nearest error slot to use if the emitter is currently within a
  /// context that can raise. Otherwise, return null.
  MLValue findNearestErrorSlot();

  /// Verify inferred error types for escaping origins.
  void checkInferredErrorType(ASTType rvalueType, SMLoc loc);

  /// Emit a normal return (not a 'raise' return) out of the function, along
  /// with any special logic that goes with it.  `funcDecl` indicates the
  /// function we are returning out of.
  static void emitNormalReturn(ImplicitLocOpBuilder &builder,
                               Value value = Value(), bool emitEndFunc = true);

  /// Emit a normal return (not a 'raise' return) out of the function, along
  /// with any special logic that goes with it.  If the value is missing this is
  /// treated as a 'return;' synthesizing a None result.
  void emitNormalReturn(Location loc, Value value = Value(),
                        bool emitEndFunc = true);

  /// Helper to emit a VarDeclOp with a uniquely generated origin name.
  VarDeclOp emitVarDecl(const Twine &name, Type type, Location loc,
                        VarDeclKind kind);
  VarDeclOp emitVarDecl(StringAttr name, Type type, Location loc,
                        VarDeclKind kind);

  /// Internal implementation of call emission, use emitCall/emitIndirectCall
  /// or higher level wrappers instead.
  /// In case customOpName is provided, emit a custom MLIR operation instead
  /// with the given name for the given custom op definition struct type.
  CValue emitCallUnchecked(RValue callee, CallOperands &&operands);
};

//===----------------------------------------------------------------------===//
// Type Refinement Utilities
//===----------------------------------------------------------------------===//

/// Refine a parametric type based on where-clause constraints in the scope.
/// Returns the refined type if refinement applies, otherwise the original type.
Type maybeRefineTypeWithAssumptions(Type varType, ASTDecl &declScope);

/// Refine a type-valued parameter expression based on where-clause constraints
/// in the scope. Returns the refined type value if refinement applies,
/// otherwise the original type value.
TypedAttr maybeRefineTypeValueWithAssumptions(TypedAttr typeValue,
                                              ASTDecl &declScope);

/// Emit a `kgen.rebind` for `value` when type refinement applies in the
/// current scope, otherwise return `value` unchanged. Use this overload when
/// only a raw SSA `Value` is available (e.g., from a GEP or a block
/// argument).
Value maybeEmitRefinementRebind(Value value, ASTDecl &declScope,
                                OpBuilder &builder, Location loc);

/// CValue overload. Delegates to `IREmitter::rebindValue` so the returned
/// CValue preserves its original kind (MLValue, MBValue, SBValue, etc.).
CValue maybeEmitRefinementRebind(ASTExprAnd<CValue> value, IREmitter &emitter);

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_EXPREMITTER_H
