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
// This file implements the core of call emission, without compatibility/type
// checking and error emission.
//
//===----------------------------------------------------------------------===//

#include "ExprNodes.h"
#include "IREmitter.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParamInf.h"
#include "ParserEvaluationContext.h"

#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/ParameterReplacer.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPOps.h"

#include "Support/Compiler/OperationUtils.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// This helper function emits a call to the VariadicList/VariadicPack
/// constructor and returns the result value.
static CValue emitVariadicCtor(ASTType variadicType, CValue variadicList,
                               const ExprNode *expr, IREmitter &emitter) {
  return emitter.emitConstructorCall(
      variadicType, CallOperands(CallSyntax::kTypeCall, expr, EC_CallArgValue,
                                 {{variadicList, expr}}));
}

//===----------------------------------------------------------------------===//
// CallEmitter (implementation detail)
//===----------------------------------------------------------------------===//

namespace {
class CallEmitter {
public:
  CallEmitter(RValue callee, const ExprNode *callExpr, IREmitter &emitter,
              ExprDest &dest);

  ~CallEmitter() {
    // If we tear this down without emitting to the destination, then an error
    // must have happened.
    dest.resetForError(emitter);
  }

  /// Emit IR for a single argument, according to its convention.
  AnyValue emitOneArgVal(ASTExprAnd<AnyValue> operand, unsigned argIdx,
                         ArgConvention convention, Type expectedType,
                         size_t sequenceIndex = 0);

  /// Emit IR for forwarding a whole variadic pack as a single argument.
  AnyValue emitWholePackForward(ASTExprAnd<AnyValue> operand,
                                Type expectedType);

  /// Emit all arguments and return their values in a vector.
  /// This function iterates by expected arguments since we're building the
  /// argument list of the call. Default arguments are applied (if available and
  /// an operand isn't provided for the arg), and variadics (including packs)
  /// are collected from the operand list and emitted as the appropriate
  /// variadic/pack type to the callee.
  FailureOr<std::pair<SmallVector<ASTExprAnd<AnyValue>>, llvm::BitVector>>
  emitArgValues(const CallOperands &operands);

  /// This function emits the specified pre-emitted argument into a single MLIR
  /// Value suitable for passing to the callee with the specified convention.
  Value emitPreemittedArgumentAsDynamicValue(ASTExprAnd<AnyValue> argValAndExpr,
                                             bool isDefaultArgVal,
                                             ArgConvention convention,
                                             Type declaredArgType,
                                             ArrayRef<Value> callArgsSoFar);

  /// Emit a function call in a parameter context.
  TypedAttr
  emitCallInParamContext(ArrayRef<ASTExprAnd<AnyValue>> argumentValues);

  /// Emit any after-call actions collected during call emission.
  void emitAfterCallActions() { afterCallActions.emit(); }

  /// Emit warnings about incorrect code in a direct call.
  void emitDirectCallWarnings(LIT::CallOp call,
                              const CallOperands &callOperands);

  /// The underlying expression emitter instance.
  IREmitter &emitter;

  /// Given a call to a function with a memory only result and the desired value
  /// destination, decide if it is safe to directly emit into the slot.  Doing
  /// so requires a form of alias analysis to determine whether any input
  /// arguments could alias the result slot.  We cannot emit into the result
  /// slot when passing the value as an argument like 'x = foo(x)' or 'x = x +
  /// 1'.
  ///
  /// At this point, we've already applied implicit conversions and converted
  /// things to RValues or BValues as required by the argument convention, but
  /// things may still be in parameter space.
  bool isSafeToUseValueDestForDirectResult(RefType destRefType,
                                           ArrayRef<Value> argumentValues);

  /// The (type-checked and resolved) callee we are emitting the call to.
  RValue callee;
  /// The call's expression node.
  const ExprNode *callExpr;
  /// The mlir location of the call expression above, stored for convenience.
  Location loc;
  /// The destination context we're emitting into.
  ExprDest &dest;
  /// The signature type of the callee, stored for convenience.
  FnTypeGeneratorType calleeSig;

private:
  /// This struct accumulates information about IR to emit after the call, e.g.
  /// writebacks for computed mut lvalues, and origin markers.
  struct AfterCallActions {
    CallEmitter &callEmitter;

    // The first entry of this is a ExprDest for a DLValue that we can invoke
    // for the setter.
    SmallVector<std::pair<ExprDest, MLValue>> lvalueWritebacks;

    AfterCallActions(CallEmitter &callEmitter) : callEmitter(callEmitter) {}

    /// Emit all after-call actions.
    void emit();

    ~AfterCallActions() {
      // If an error happens before we emit the write backs, make sure to nuke
      // them so they don't crash the compiler.
      while (!lvalueWritebacks.empty())
        lvalueWritebacks.pop_back_val().first.resetForError(
            callEmitter.emitter);
    }
  } afterCallActions;

  /// Emit the given (remaining) operands as a variadic or pack sequence,
  /// appending to the given argument value vector.
  LogicalResult emitRemainingPosOperands(
      size_t argIdx, MutableArrayRef<OperandValue> remainingOperands,
      ArgConvention convention, Type expectedType,
      SmallVectorImpl<ASTExprAnd<AnyValue>> &argumentValues);
};
} // end anonymous namespace.

CallEmitter::CallEmitter(RValue calleeVal, const ExprNode *callExpr,
                         IREmitter &emitter, ExprDest &dest)
    : emitter(emitter), callee(calleeVal), callExpr(callExpr),
      loc(emitter.translateLocation(callExpr->getLoc())), dest(dest),
      calleeSig(sugarDynCast<FnTypeGeneratorType>(calleeVal.getRValueType())),
      afterCallActions(*this) {
  // Turn function literals into symbol constants.
  //
  // TODO: we don't really need the conversion here, we should instead handle
  // calls to function literals directly.
  if (!calleeSig) {
    // This must be a function literal type that is fully bound.
    auto fnLiteralGenType =
        sugarCast<FnLiteralTypeGeneratorType>(callee.getRValueType());
    assert(fnLiteralGenType.getInputParamTypes().empty());
    callee = fnLiteralGenType.getSymbolConstantAttr();
    calleeSig = cast<FnTypeGeneratorType>(callee.getType());
  }
  assert(calleeSig);
  // Make sure the sugar on the callee agrees with calleeSig.  This strips off
  // top level sugar on the function type itself.
  if (callee.getRValueType().mlirType != calleeSig) {
    callee = emitter.emitRValue({callee, callExpr},
                                ExprContext::EC_CallCalleeValue, calleeSig);
    assert(callee && callee.getRValueType().mlirType == calleeSig &&
           "rebinding sugar should work!");
  }
}

void CallEmitter::AfterCallActions::emit() {
  IREmitter &emitter = callEmitter.emitter;

  // Emit the elements and clear the writebacks so the ExprDest's get
  // destroyed when they are emitted into.
  while (!lvalueWritebacks.empty()) {
    // Get 'dest' (the computed LValue as a ExprDest) and 'lValue' (the memory
    // temporary we're working with) so we can do a writeback.
    auto [dest, lValue] = lvalueWritebacks.pop_back_val();

    // The lValue is the MLValue of the temporary holding the 'get'd value.  We
    // pass it as an MRValue to the "set" method, allowing the value to be
    // consumed directly by an 'owned' argument without a copy.
    if (!emitter.emitResult(MRValue(lValue), callEmitter.callExpr, dest))
      dest.resetForError(emitter);
  }
}

AnyValue CallEmitter::emitWholePackForward(ASTExprAnd<AnyValue> operand,
                                           Type expectedType) {
  auto expectedRefType = sugarCast<RefType>(expectedType);

  Value refValue = emitter.emitRefValue(operand, EC_CallRefArgValue);
  if (!refValue)
    return {};

  // Rebind to the expected VariadicPack type except for the origin. Use the
  // provided !lit.ref.pack's origin after a mutability cast.
  // TODO: Actually support mutability casting for !lit.ref.pack.
  RefType actualRefType = sugarCast<RefType>(refValue.getType());
  ASTType actualPackType = actualRefType.getElementType();
  ASTType expectedPackType = expectedRefType.getElementType();
  RefPackType actualRefPackType =
      actualPackType.getVariadicPackInfo(emitter.shared);
  RefPackType expectedRefPackType =
      expectedPackType.getVariadicPackInfo(emitter.shared);

  TypedAttr actualOrigin = actualRefPackType.getOrigin();
  TypedAttr compatibleOrigin = OriginMutCastAttr::get(
      actualOrigin, expectedRefPackType.getOrigin().getType());

  SmallVector<TypedAttr> paramBindings(expectedPackType.getParamBindings());
  // Swap out the origin for the casted origin.
  SyntheticNode synthBoolNode(operand.expr->getLoc());
  PValue expectedMutability =
      cast<OriginType>(expectedRefPackType.getOrigin().getType())
          .getIsMutable();
  CValue mutabilityValue = emitter.emitBool(
      {expectedMutability, &synthBoolNode}, ExprContext::EC_PackArgument);
  paramBindings[0] = mutabilityValue.getIfPValue().get();
  paramBindings[1] = compatibleOrigin;
  paramBindings[2] =
      emitter.getStdlibOriginOf(compatibleOrigin, operand.expr->getLoc());

  Type compatibleVariadicPackType =
      cast<StructDeclOp>(
          expectedPackType.getDecl(emitter.shared)->getIfOperation())
          .bindReference(paramBindings);
  auto compatibleRefType =
      actualRefType.getWithElement(compatibleVariadicPackType);

  refValue = emitter.emitRebindOpIfNeeded(refValue, compatibleRefType,
                                          operand.expr->getLoc());

  // If the value coming in is owned, then so is our result.
  if (operand.ir.getIfRValue() && actualRefType.isMutableKnown(true))
    return MRValue(refValue);

  return CValue::getMValueForRef(refValue);
}

AnyValue CallEmitter::emitOneArgVal(ASTExprAnd<AnyValue> operand,
                                    unsigned argIdx, ArgConvention convention,
                                    Type expectedType, size_t sequenceIndex) {
  // Convert to the element type of a variadic list or pack.
  if (calleeSig.isPosVarArg(argIdx)) {
    ASTType listType = RefType::stripRefConvention(expectedType, convention);
    expectedType = listType.getVariadicListInfo().getElementRefType();
    convention = calleeSig.getVariadicConvention(argIdx);
  } else if (calleeSig.isPack(argIdx)) {
    ASTType packType = RefType::stripRefConvention(expectedType, convention);
    RefPackType refPackType = packType.getVariadicPackInfo(emitter.shared);

    // Operands being applied to a concrete pack type argument must be
    // converted to the pack element type at that index.  The calleeSig has the
    // pack type resolved to a concrete list of types it is expecting.
    expectedType =
        ASTType(refPackType.getVariadicIfResolved().getValues()[sequenceIndex]);
    // Get the !lit.ref with the origin and other paraphernalia.
    expectedType = refPackType.getElementRefTypeFor(expectedType);
    convention = calleeSig.getVariadicConvention(argIdx);
  }

  switch (convention) {
  case ArgConvention::OwnedReg:
    llvm_unreachable("not used by the mojo parser");
  case ArgConvention::Mut:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError:
    // By-ref arguments, must be lvalues.
    assert(operand.ir.getIfLValue() && "Call should already be type checked");
    return operand.ir;
  case ArgConvention::OwnedMem:
  case ArgConvention::DeinitMem:
    // Owned conventions pass rvalues.
    expectedType = sugarCast<RefType>(expectedType).getElementType();
    return emitter.emitRValue(operand, EC_CallArgValue, expectedType);

  case ArgConvention::Ref:
  case ArgConvention::MutRef: {
    // If we're in a parameter context, just leave it alone - param call
    // emission will handle it.
    if (!emitter.builder) {
      // TODO: We should use PMBValue consistently here.
      if (auto pv = operand.ir.getIfPValue())
        return pv;
      if (auto pmb = operand.ir.getIfPMBValue())
        return pmb;
    }

    // Emit the operand as a 'ref'.
    Value refValue = emitter.emitRefValue(operand, EC_CallRefArgValue);
    if (!refValue)
      return {};

    auto refValueType = sugarCast<RefType>(refValue.getType());
    auto expectedRefType = sugarCast<RefType>(expectedType);

    // Origins must be convertible, this is checked by OverloadFitness.
    // The destination may be less mutable because of canZeroCostConvert.
    // This also lazy materializes cast to immutable that MBValue avoided.
    if (!refValueType.isMutableKnown(false) &&
        expectedRefType.isMutableKnown(false)) {
      refValue = RefImmutOp::create(
          *emitter.builder, operand.expr->getLocation(emitter), refValue);
      refValueType = sugarCast<RefType>(refValue.getType());
    }

    // Materialize the callee's selected ref type. Overload checking already
    // verified any origin conversion; the element type may also differ by an
    // identity-preserving refinement wrapper, e.g. `downcast(T, Trait) -> T`.
    refValue = emitter.emitRebindOpIfNeeded(refValue, expectedType,
                                            operand.expr->getLoc());
    refValueType = sugarCast<RefType>(refValue.getType());

    assert((refValueType == expectedType ||
            getCanonicalType(refValueType) == getCanonicalType(expectedType)) &&
           "Should have exact match now");
    return CValue::getMValueForRef(refValue);
  }

  case ArgConvention::ImmMem:
    // by-ref arguments are converted to the expected r-value type.
    expectedType = sugarCast<RefType>(expectedType).getElementType();
    [[fallthrough]];
  case ArgConvention::ImmReg:
    return emitter.emitBValue(operand, EC_CallArgValue, expectedType);
  }
  llvm_unreachable("unknown argument convention");
}

/// Emit the given (remaining) operands as a variadic or pack sequence,
/// appending to the given argument value vector.
LogicalResult CallEmitter::emitRemainingPosOperands(
    size_t argIdx, MutableArrayRef<OperandValue> remainingOperands,
    ArgConvention convention, Type expectedType,
    SmallVectorImpl<ASTExprAnd<AnyValue>> &argumentValues) {
  assert((calleeSig.isPosVarArg(argIdx) || calleeSig.isPack(argIdx)) &&
         "Must be a variadic we're emitting for");
  assert(!remainingOperands.empty() && "not called for empty lists");

  if (remainingOperands.front().unpackStyle == ArgUnpackStyle::kStar) {
    assert(remainingOperands.size() == 1 &&
           "a positional operand after a '*' unpack should have been rejected "
           "before emission");
    auto &operand = remainingOperands.front();

    // Splatting a variadic list is straight-forward, just pass the splat
    // directly.
    if (calleeSig.isPosVarArg(argIdx)) {
      auto rvType = sugarCast<RefType>(expectedType).getElementType();
      if (convention != ArgConvention::OwnedMem)
        operand.ir = emitter.emitMBValue(operand, EC_CallArgValue, rvType);
      else
        operand.ir = emitter.emitMRValue(operand, EC_CallArgValue, rvType);
      if (!operand.ir)
        return failure();
      argumentValues.push_back(operand);
      return success();
    }

    assert(calleeSig.isPack(argIdx) && "only variadic packs support unpacking");
    ASTType actualPackType = operand.ir.getRValueTypeIfResolvable();
    assert(actualPackType &&
           actualPackType.getVariadicPackInfo(emitter.shared) &&
           "param inference validated unpacked pack type");

    auto emittedArg = emitWholePackForward(operand, expectedType);
    if (!emittedArg)
      return failure();
    argumentValues.push_back({emittedArg, operand.expr});
    return success();
  }

  // Emit all of the remaining values to make sure they're converted to the
  // right type.
  for (auto [idx, operand] : llvm::enumerate(remainingOperands)) {
    auto emittedArg =
        emitOneArgVal(operand, argIdx, convention, expectedType, idx);
    if (!emittedArg)
      return failure();
    operand.ir = emittedArg;
  }

  // When emitting the arguments, use the convention of the elements, not the
  // convention of the VariadicList/VariadicPack struct.
  ASTType listOrPackType =
      RefType::stripRefConvention(expectedType, convention);
  convention = calleeSig.getVariadicConvention(argIdx);
  assert(hasAddress(convention) && "variadics always passed in memory");

  // Handle emission in a compile-time context.  Parameter calls need to
  // generate parameter attributes.
  if (!emitter.builder) {
    SmallVector<TypedAttr> args;
    for (OperandValue operand : remainingOperands) {
      if (auto pv = operand.ir.getIfPValue()) { // RValue
        if (convention == ArgConvention::OwnedMem)
          args.push_back(PMRValue::getFromPValue(pv.get()));
        else
          args.push_back(PMBValue::getFromPValue(pv.get()));
      } else if (auto pmb = operand.ir.getIfPMBValue()) { // ref argument.
        args.push_back(pmb.get());
      } else if (auto pmr = operand.ir.getIfPMRValue()) { // ref argument.
        args.push_back(pmr.get());
      } else {
        emitter.emitErrorForDynamicValueInParameter(callExpr,
                                                    "cannot use dynamic value");
        return failure();
      }
    }

    CValue argValue;
    if (calleeSig.isPosVarArg(argIdx)) {
      auto variadicInfo = listOrPackType.getVariadicListInfo();
      // We have an array of references.
      auto array = POP::ArrayAttr::getWithElementType(
          args, variadicInfo.getElementRefType());
      // We pass a reference to the array.
      argValue = StoreToMemAttr::get(
          array, RefType::get(array.getType(), ComptimeOriginAttr::get(
                                                   array.getContext(), false)));
    } else {
      RefPackType packType = listOrPackType.getVariadicPackInfo(emitter.shared);
      argValue = PValue(RefPackAttr::get(args, packType));
    }
    argValue = emitVariadicCtor(listOrPackType, argValue, callExpr, emitter);
    if (!argValue)
      return failure();
    argumentValues.push_back({argValue, remainingOperands[0].expr});
    return success();
  }

  // If not all remaining operands are compile-time values, use an operation to
  // create a variadic or pack sequence.
  SmallVector<Value> args;
  for (auto &operand : remainingOperands) {
    assert(convention != ArgConvention::ByRefResult &&
           "cannot have variadics of this convention, so can pass in empty "
           "callArgsSoFar");
    Value argVal = emitPreemittedArgumentAsDynamicValue(
        operand, /*isDefaultArgVal=*/false, convention, expectedType,
        ArrayRef<Value>());
    if (!argVal)
      return failure();
    args.push_back(argVal);
  }

  // Create a uniform representation of origins.
  TypedAttr eltOrigin;
  if (calleeSig.isPosVarArg(argIdx))
    eltOrigin = listOrPackType.getVariadicListInfo().origin;
  else
    eltOrigin = listOrPackType.getVariadicPackInfo(emitter.shared).getOrigin();

  // Rebind all the elements to the common origin ParamInf determined for us.
  for (auto &arg : args) {
    auto argType = cast<RefType>(arg.getType());
    if (argType.getOrigin() == eltOrigin)
      continue; // Already the right origin.
    // Cast to common origin with a rebind.
    arg = emitter.emitRebindOpIfNeeded(arg, argType.getWithOrigin(eltOrigin),
                                       callExpr->getLoc());
  }

  CValue argVal;
  if (calleeSig.isPosVarArg(argIdx)) { // Positional homogenous varargs
    auto refType = listOrPackType.getVariadicListInfo().getElementRefType();
    // Create a stack temporary containing an array of the values.
    auto arrayType = POP::ArrayType::get(args.size(), refType);
    auto varDecl = emitter.emitVarDecl("__passed_varargs__", arrayType, loc,
                                       VarDeclKind::Synthesized);
    // Put all the elements into an array with one store.
    auto array =
        POP::ArrayCreateOp::create(*emitter.builder, loc, arrayType, args);
    RefStoreOp::create(*emitter.builder, loc, array, varDecl);
    argVal = SRValue(varDecl);
  } else { // Bundle them up into a VariadicPack instance.
    RefPackType packType = listOrPackType.getVariadicPackInfo(emitter.shared);
    argVal =
        SRValue(RefPackCreateOp::create(*emitter.builder, loc, packType, args));
  }
  argVal = emitVariadicCtor(listOrPackType, argVal, callExpr, emitter);
  if (!argVal)
    return failure();
  argumentValues.push_back({argVal, remainingOperands[0].expr});
  return success();
}

FailureOr<std::pair<SmallVector<ASTExprAnd<AnyValue>>, llvm::BitVector>>
CallEmitter::emitArgValues(const CallOperands &operands) {
  PogListAttr argListAttr = calleeSig.getArgListAttrs();

  // Map the operands onto the callee's arguments using the same logic type
  // checking used, so that emission and inference agree on which operand
  // fulfills which argument.
  CallOperands::PogAssignment pogAssignment;
  std::optional<MojoInflightDiag> stagedDiag;
  auto getStagedDiag = [&](SMLoc loc) -> MojoInflightDiag & {
    stagedDiag = emitter.emitError(loc);
    return *stagedDiag;
  };
  if (failed(operands.assignToPogs(argListAttr, /*isParameterList=*/false,
                                   pogAssignment, getStagedDiag)))
    return failure();

  SmallVector<ASTExprAnd<AnyValue>> argumentValues;
  llvm::BitVector isDefaultMask(calleeSig.getNumArguments(), false);
  argumentValues.reserve(calleeSig.getNumArguments());

  for (auto [argIdx, expectedType, convention, pogAttr] :
       llvm::enumerate(calleeSig.getArguments(), calleeSig.getArgConventions(),
                       argListAttr.getPogs())) {
    switch (pogAssignment.operandIdxs[argIdx]) {
    case CallOperands::PogAssignment::kPA_Unspecified:
      // This is the return or error slot for the call, so we need a temporary
      // to emit into, but don't know the type until the arguments (and their
      // origins) are all emitted. Just skip over it for now.
      assert(isResultSlot(convention) && "unknown unspecified operand");
      assert(calleeSig.hasMemoryOnlyResult() ||
             (calleeSig.isThrows() &&
              pogAttr.getPassingKind() == PassingKind::Implicit));
      argumentValues.push_back({AnyValue(), callExpr});
      continue;

    // The normal case fulfills the argument with a single operand, whether it
    // was passed positionally or by keyword.
    default: {
      size_t operandIdx = pogAssignment.operandIdxs[argIdx];
      const OperandValue &operand = operands[operandIdx];
      AnyValue argVal =
          emitOneArgVal(operand, argIdx, convention, expectedType);
      if (!argVal)
        return failure();
      argumentValues.push_back({argVal, operand.expr});
      continue;
    }

    case CallOperands::PogAssignment::kPA_Default: {
      // Apply the default argument. We need to emit it with our expected type
      // because there may be an implicit conversion.
      PValue defaultVal = argListAttr.getDefault(argIdx);
      assert(defaultVal && "default value is missing");
      AnyValue argVal = emitOneArgVal({defaultVal, callExpr}, argIdx,
                                      convention, expectedType);
      if (!argVal)
        return failure();
      isDefaultMask.set(argIdx);
      argumentValues.push_back({argVal, callExpr});
      continue;
    }

    case CallOperands::PogAssignment::kPA_Variadic:
      // Handle variadics below.
      break;
    }

    // Keyword variadics either forward a whole '**' unpack, or collect the
    // keyword operands that no named argument claimed into a dictionary.
    if (calleeSig.isKwVarArg(argIdx)) {
      ArrayRef<size_t> kwOperandIdxs = pogAssignment.kwVariadicIdxs;
      if (kwOperandIdxs.size() == 1 &&
          operands[kwOperandIdxs[0]].unpackStyle == ArgUnpackStyle::kStarStar) {
        const OperandValue &splat = operands[kwOperandIdxs[0]];
        AnyValue argVal =
            emitOneArgVal(splat, argIdx, convention, expectedType);
        if (!argVal)
          return failure();
        argumentValues.push_back({argVal, splat.expr});
        continue;
      }

      // Otherwise, initialize a dictionary; the keyword operands are inserted
      // into it below.
      auto dict = emitter.emitConstructorCall(
          sugarCast<RefType>(expectedType).getElementType(),
          CallOperands(CallSyntax::kTypeCall, callExpr, EC_KWArgsArgument));
      auto kwargsDict = emitter.emitMRValue({dict, callExpr}, EC_CallArgValue);

      // Fill the **kwargs dict with the keyword operands that no named argument
      // claimed.  There is no dict when a '**' unpack was forwarded whole.
      for (size_t operandIdx : pogAssignment.kwVariadicIdxs) {
        const OperandValue &operand = operands[operandIdx];
        assert(operand.keyword &&
               "should have been rejected during `assignToPogs`");
        SMLoc loc = operand.expr->getLoc();
        SyntheticNode tmpNode(loc);
        ExprDest kwargsDest(EC_KWArgsArgument);
        CValue literalKey = StringLiteralNode::emitCtorCall(
            operand.keyword.strref(), &tmpNode, kwargsDest, emitter);
        if (!literalKey)
          return {};

        // Then we set the element with the given key and the operand as value.
        CallOperands insertOperands(CallSyntax::kMethodCall, callExpr,
                                    EC_KWArgsArgument,
                                    {{MLValue(kwargsDict), callExpr},
                                     {literalKey, operand.expr},
                                     operand});
        emitter.emitNamedMethodCall("_insert", std::move(insertOperands));
      }
      argumentValues.push_back({kwargsDict, callExpr});
      continue;
    }

    // Positional variadics and packs are handled all at once (or fail).
    if (!pogAssignment.posVariadicIdxs.empty()) {
      SmallVector<OperandValue> variadicOperands;
      variadicOperands.reserve(pogAssignment.posVariadicIdxs.size());
      for (size_t operandIdx : pogAssignment.posVariadicIdxs)
        variadicOperands.push_back(operands[operandIdx]);

      if (failed(emitRemainingPosOperands(argIdx, variadicOperands, convention,
                                          expectedType, argumentValues)))
        return failure();
      continue;
    }

    // With no operands assigned to it, fulfill this with an empty variadic
    // list or an empty pack.
    if (calleeSig.isPosVarArg(argIdx)) {
      ASTType listType = RefType::stripRefConvention(expectedType, convention);
      auto refType = listType.getVariadicListInfo().getElementRefType();
      // VarArgs arguments are fulfilled with an null pointer to a zero element
      // array. It will never be dereferenced.
      auto arrayType = POP::ArrayType::get(0, refType);
      auto immutOriginType = OriginType::get(emitter.getContext(), false);
      // The origin of the reference is an empty union of the immut origin type.
      auto nullRefType =
          RefType::get(arrayType, OriginUnionAttr::get({}, immutOriginType));
      auto argAttr = PointerAttr::get(0, nullRefType);
      auto result =
          emitVariadicCtor(listType, PValue(argAttr), callExpr, emitter);
      if (!result)
        return failure();
      argumentValues.push_back({result, callExpr});
      continue;
    }

    // Pack arguments are fulfilled with an empty #lit.ref.pack.
    assert(calleeSig.isPack(argIdx) && "unknown variadic kind");
    ASTType packType = RefType::stripRefConvention(expectedType, convention);
    assert(sugarCast<ParamListAttr>(packType.getVariadicPackInfo().typeList)
               .getValues()
               .empty() &&
           "pack type already checked against operand count");
    RefPackType refPackType = packType.getVariadicPackInfo(emitter.shared);
    auto argAttr = RefPackAttr::get(ArrayRef<TypedAttr>(), refPackType);
    auto result =
        emitVariadicCtor(packType, PValue(argAttr), callExpr, emitter);
    if (!result)
      return failure();
    argumentValues.push_back({result, callExpr});
  }

  return std::make_pair(std::move(argumentValues), std::move(isDefaultMask));
}

/// Given a call to a function with a memory only result and the desired value
/// destination, decide if it is safe to directly emit into the slot.  Doing so
/// requires a form of alias analysis to determine whether any input arguments
/// could alias the result slot.  We cannot emit into the result slot when
/// passing the value as an argument like 'x = foo(x)' or 'x = x + 1'.
///
/// At this point, we've already applied implicit conversions and converted
/// things to RValues or BValues as required by the argument convention, formed
/// variadic list/packs, and emitted to the final SSA values that will get
/// passed.
bool CallEmitter::isSafeToUseValueDestForDirectResult(
    RefType destRefType, ArrayRef<Value> argValues) {

  // We don't handle a "ref result" ExprDest initializing a pattern yet.
  if (calleeSig.isRefResult() && !dest.isOperation())
    return false;

  ASTType destRValueType = destRefType.getElementType();

  // Check to see if the destination provides a buffer.  If not, we can't use
  // the dest as a direct memory result.
  MLValue destBuffer = dest.getDefinedMLValueIfExists(destRValueType, emitter);
  if (!destBuffer)
    return false;

  // Ok, we got a destination buffer, make sure it agrees on address space. If
  // not, we force a temporary.
  if (!isEqualCanon(destBuffer.getRefType().getAddressSpace(),
                    destRefType.getAddressSpace()))
    return false;

  // If this is a throwing function, then we cannot write to a field of a
  // origin tracked value.  Consider:
  //     x.f = foo()
  // The contract for the result slot is that 'x.f' has to be uninitialized
  // before the function call, so we would have to destroy any live value in the
  // slot before calling the function:
  //     x.f.__del__()
  //     foo(x.f, __error__)  # Passed as byref_result
  // However, when the function throws, if 'x' is unused, we need to be able to
  // delete the full 'x' value.  However, we cannot call the destructor on 'x'
  // because the whole value isn't initialized.  Address this by not assigning
  // into submembers in throwing functions.
  if (calleeSig.isThrows()) {
    // FIXME: This is pretty grotty, basically doing dataflow analysis here in
    // the parser.  It would be better to just always emit a temporary (when the
    // type movable and the function throws) and then use a pass (CheckLifetimes
    // or similar) to eliminate the movector+temporary when possible.

    // See if the destination buffer is something that ownership can track.
    Value underlyingDest =
        OriginTrackable::findUnderlyingValueFromField(destBuffer);
    // If we don't know what it is, it may be a 'ref' result, handle
    // conservatively.
    if (!underlyingDest)
      return false;

    // Dig deeper into the nature of the thing we're assigning into to try to
    // enable a few important cases.
    OriginTrackable trackable(underlyingDest);

    // We can't allow assigning into a field because we cannot partially destroy
    // the value, but we can overwrite the whole thing.
    if (underlyingDest != destBuffer) {
      // byref_result arguments of initializers can be piecewise destroyed on a
      // thrown error.
      if (!trackable.isFullObjectLiveOnEntry ||
          trackable.endInitState !=
              OriginTrackable::ExitInitState::InitOnNormal)
        return false;
    }

    // We also cannot assign to values that must be live-out from the function
    // on an error return.  This includes (for example) by-ref arguments.
    if (trackable.endInitState == OriginTrackable::ExitInitState::EndsInit)
      return false;

    // If the type is movable, then ~always use a temporary result.  The reason
    // is that we need to support things like:
    // def func():
    //     val1 = 0
    //     try:
    //         val1 = fn_that_raises()  # Here.
    //     except:
    //         pass
    //     use(val1) # Should work!
    //
    // If we don't do this end up with an uninitialized value on the error
    // return path.  Movable types are supposed to be efficiently movable.
    if (destRValueType.isMovable(callExpr->getLoc(), emitter.shared,
                                 emitter.declScope)) {
      // If we're the top level, then nothing can use the result, so we can
      // microoptimize this case.
      Operation *opForRaise = nullptr;
      if (emitter.builder)
        opForRaise =
            findOpProcessingRaise(emitter.builder->getInsertionBlock());

      bool isOk = false;
      if (opForRaise) {
        // If we're exiting the entire function, we are ok.
        if (isa<FnOp>(opForRaise))
          isOk = true;

        // If that didn't work, see if we're initializing a vardecl within the
        // current exception region.  If so, we know that the exception couldn't
        // use the old value because it won't be live outside.
        else if (VarDeclOp varDeclOp =
                     underlyingDest.getDefiningOp<VarDeclOp>()) {
          if (underlyingDest.use_empty()) {
            // If there are no assignments into this so far, then it must be
            // uninit.
            isOk = true;
          } else if (opForRaise->isAncestor(varDeclOp)) {
            isOk = true;
          }
        }
      }

      if (!isOk)
        return false;
    }
  }

  // Collect all of the types of all the arguments so we can collect the
  // origins they may reference.
  SmallVector<Type> argTypes;
  for (auto [value, convention] :
       llvm::zip(argValues, calleeSig.getArgConventions())) {
    if (isResultSlot(convention))
      continue;

    argTypes.push_back(value.getType());
  }

  // We're doing field sensitive comparisons below, so we record a destination
  // "x.y" as a full "x.y" path, to make sure it doesn't conflict with reads of
  // "x.z".  We do want "x.y.q" and "x" to conflict with "x.y" though.  Keep
  // track of the fully field sensitive origin, and the containing origins.
  SmallPtrSet<Attribute, 2> destOrigins, destContainerOrigins;

  // processOriginUnionElts takes apart origin unions for us.
  auto refOrigin = sugarCast<RefType>(destBuffer.getType()).getOrigin();
  processOriginUnionElts(getCanonicalAttr(refOrigin), [&](TypedAttr origin) {
    // AnyOrigin is assumed to be ok since it is used for
    // UnsafePointer etc.  We don't want to track it.
    if (sugarIsa<AnyOriginAttr>(origin))
      return;
    // We want to track the fully field sensitive origin, and
    // the containing origins.
    destOrigins.insert(origin);
    while (auto fieldAttr = dyn_cast<OriginFieldAttr>(origin)) {
      origin = fieldAttr.getBase();
      destContainerOrigins.insert(origin);
    }
  });

  // Check to see if any of the the origins they may be accessing are the
  // origin in question.  If any of them is a possible reference to the
  // destination slot, then we must fail.
  CachedOriginFinder &finder = emitter.shared.cachedOriginFinder;
  for (TypedAttr origin :
       finder.findOriginsIn(argTypes, calleeSig.getCaptureOrigins())) {
    // Look through any immcasts.
    origin = OriginType::stripMutCastAndRebind(origin);

    // If this is accessing any container origins directly, then we have a store
    // to "x.y" and another use of "x" which can't be allowed.
    if (destContainerOrigins.contains(origin))
      return false;

    // Otherwise, check to see if this is descended from the specific
    // destination, e.g. the dest is "x.y" and this is "x.y.z".
    while (1) {
      if (destOrigins.contains(origin))
        return false;
      if (auto fieldAttr = dyn_cast<OriginFieldAttr>(origin))
        origin = fieldAttr.getBase();
      else
        break;
    }
  }

  // If no problems are found, it is safe!
  return true;
}

/// This function emits the specified pre-emitted argument into a single MLIR
/// Value suitable for passing to the callee with the specified convention.
Value CallEmitter::emitPreemittedArgumentAsDynamicValue(
    ASTExprAnd<AnyValue> argValAndExpr, bool isDefaultArgVal,
    ArgConvention convention, Type declaredArgType,
    ArrayRef<Value> callArgsSoFar) {
  assert(emitter.builder && "Should only be called in dynamic context");

  ExprContext ctx = isDefaultArgVal ? EC_CallArgDefaultValue : EC_CallArgValue;

  // This checks any returned MValue argument convention for validity.
  auto checkMValueAddrSpace = [&](AnyValue someMValue) -> Value {
    // Propagate errors.
    if (!someMValue)
      return {};
    assert(someMValue.isMValue() && "Not an MValue");

    // All argument conventions take things in the default address space, so any
    // use of references in other address spaces need to do a copyinit.  Right
    // now Copyable requires the source to be borrowed (it doesn't allow a
    // `ref ` existing so there is no way to define a
    // non-TrivialRegisterPassable type in another address space. Diagnose
    // this error with a specific message, and copy trivially-copyable types
    // into address space 0 implicitly.
    auto refType = someMValue.getMValueType();

    if (!refType.isDefaultAddrSpace()) {
      auto *expr = argValAndExpr.expr;
      // Non-trivially copyable types cannot be copied from a non-default
      // address space, because copyinit doesn't allow 'ref'.
      if (!ASTType(refType.getElementType())
               .isProvablyImplicitlyTriviallyCopyable(
                   expr->getLoc(), emitter.shared, emitter.declScope)) {
        emitter.emitError(
            expr->getLoc(),
            "non-implicitly trivially copyable value cannot be copied from a "
            "non-default address space")
            << expr->getRange();
        return {};
      }

      // If this is a trivially copyable value, then we can do a copy by doing a
      // load.
      auto srVal = emitter.emitSRValue({someMValue, expr}, ctx);
      someMValue = emitter.emitMRValue({srVal, expr}, ctx);
      if (!someMValue)
        return {};
    }
    return someMValue.getMValueReference();
  };

  switch (convention) {
  case ArgConvention::OwnedReg:
    llvm_unreachable("not used by the mojo parser");
  case ArgConvention::OwnedMem:
  case ArgConvention::DeinitMem:
    // Promote PValue's if needed.
    return checkMValueAddrSpace(emitter.emitMRValue(argValAndExpr, ctx));
  case ArgConvention::ImmReg:
    // If this is already an SValue, then use it.
    if (argValAndExpr.ir.isSValue())
      return argValAndExpr.ir.getSValueRegister();

    // Otherwise, materialize or load the value.
    return emitter.emitSRValue(argValAndExpr, ctx);

  case ArgConvention::ImmMem: {
    // Promote PValue's if needed.
    Value result =
        checkMValueAddrSpace(emitter.emitMBValue(argValAndExpr, ctx));

    // Drop mutability for a MBValue.
    if (result && !sugarCast<RefType>(result.getType()).isMutableKnown(false))
      result = RefImmutOp::create(
          *emitter.builder, argValAndExpr.expr->getLocation(emitter), result);
    return result;
  }
  case ArgConvention::Ref:
  case ArgConvention::MutRef:
    assert(argValAndExpr.ir.isMValue() &&
           "Ref args are already emitted to boxes during overload resolution");
    // These can be in any address space.
    return argValAndExpr.ir.getMValueReference();

  case ArgConvention::ByRefError: {
    // If the error type is Never, just emit a temporary and use it
    // unconditionally.
    auto errorType = ASTType(cast<RefType>(declaredArgType).getElementType());
    if (sugarIsa<NeverType>(errorType)) {
      auto loc = argValAndExpr.expr->getLocation(emitter);
      auto tmp = emitter.emitVarDecl("__never_error__", errorType, loc,
                                     VarDeclKind::Synthesized);
      return MLValue(tmp);
    }

    // If the callee throws and is not async, we pass the contextual error
    // slot.
    MLValue errSlot = emitter.findNearestErrorSlot();
    if (!errSlot) {
      auto diag = emitter.emitError(callExpr->getLoc())
                  << "cannot call function that may raise in a context that "
                     "cannot raise"
                  << callExpr->getRange();
      diag.attachNote(callExpr->getLoc())
          << "try surrounding the call in a 'try' block";
      if (auto func = getBlockParentOfType<FnOp>(
              emitter.builder->getInsertionBlock())) {
        diag.attachNote(func.getLoc())
            << "or mark surrounding function as 'raises'";
      }
      return {};
    }
    // Make sure the raised error type matches the contextual error type. We
    // don't support throwing an Int in a context that catches a Float for
    // example.
    // TODO: we can add support for implicit conversions in the future.  We can
    // make many types implicitly convert to Error by stringizing.  Until then,
    // people can do this manually by catching and rethrowing.
    if (isa<UnresolvedType>(errSlot.getRValueType())) {
      // If the contextual caught type is unresolved, then we're the first call
      // in a try block.  Resolve the error type to whatever type we raise.
      auto errorVar = cast<VarDeclOp>(errSlot.getDefiningOp());
      errorVar.changeElementType(errorType);
      emitter.checkInferredErrorType(errorType, callExpr->getLoc());
    } else if (!errorType.isEqualCanon(errSlot.getRValueType())) {
      emitter.emitError(callExpr->getLoc())
          << "cannot call function that may raise " << errorType
          << " in context that supports an error type of "
          << errSlot.getRValueType() << callExpr->getRange();
      return {};
    }
    return errSlot;
  }

  case ArgConvention::ByRefResult:
  case ArgConvention::Mut: {
    // byref_result can have a placeholder when there is no specified
    // destination, but can also have a destination specified directly.
    auto resultRefType = sugarCast<RefType>(declaredArgType);
    if (!argValAndExpr.ir) {
      assert(convention == ArgConvention::ByRefResult &&
             "value must be present for 'mut' convention");
      auto resultRValueType = resultRefType.getElementType();

      // Often the result of the call will be directly assigned into a
      // user-defined var or other location with existing storage.  In these
      // cases, we really want to assign directly into the existing slot.
      //
      // However, we cannot do that if the destination slot is also being
      // passed into the call as an input value, as in: `x = foo(x)` or `x = x
      // + 1`. In these cases we really do need a temporary+copy in the var
      // slot. At this point we've got enough information about the arguments
      // to make that assessment in a correct way.
      Value resultSlotVal;
      if (isSafeToUseValueDestForDirectResult(resultRefType, callArgsSoFar)) {
        // Use the preferred location of the destination slot.
        resultSlotVal = dest.getMLValueForResult(callExpr->getLoc(),
                                                 resultRValueType, emitter);
      } else {
        auto loc = argValAndExpr.expr->getLocation(emitter);
        resultSlotVal =
            emitter.emitVarDecl("__call_result_tmp__", resultRValueType, loc,
                                VarDeclKind::Synthesized);
      }
      argValAndExpr.ir = MLValue(resultSlotVal);
    }

    // We know that the operand is an LValue, but it might be
    // dynamic/computed.
    LValue lv = argValAndExpr.ir.getIfLValue();
    assert(lv && "type checking ensures we will have an lvalue");

    // If this is already an MLValue, we want to pass the reference directly,
    // but we require the address spaces to line up.
    if (MLValue ref = lv.getIfMLValue()) {
      if (isEqualCanon(ref.getRefType().getAddressSpace(),
                       resultRefType.getAddressSpace()))
        return ref;
    }

    // Otherwise, we have a computed lvalue or a stored lvalue in the wrong
    // address space.  In general we need to do a load from the LValue into
    // a temporary slot and then a writeback when we're done.  Start by loading
    // the LValue, which for a DLValue will call the getter.
    ExprDest tmpValueDest(lv.getRValueType(), EC_CallArgValue);
    CValue loadVal;
    if (DLValue dlv = lv.getIfDLValue()) {
      // If the getter return as mutable reference, we don't want to force an
      // RValue yet.
      loadVal = dlv->emitLoad(tmpValueDest, emitter);
    } else {
      loadVal = emitter.emitRValue({lv, argValAndExpr.expr}, tmpValueDest);
    }
    if (!loadVal)
      return {};

    // One interesting case to watch for are types like Dict.  These types have
    // both a getter and a setter, but the getter will return a mutable
    // reference when self is mutable.  The semantics of Dict are that we must
    // call the setter to create an entry if it doesn't exist, as in:
    //
    //    dict[elt] = value
    //
    // But we prefer to use the mutable reference directly when possible, as in:
    //
    //    dict[elt] += 4
    //
    // This is because it avoids the need to load the element, add 4, then
    // store the result back. We know this is safe, because the getter has to
    // be called in any case.
    if (auto mlVal = loadVal.getIfMLValue()) {
      // Don't do this for assignments.
      if (convention == ArgConvention::Mut &&
          sugarCast<RefType>(mlVal.getType()).isDefaultAddrSpace())
        return mlVal;
    }

    // Okay, we really do need to do a write back.  Force loadVal to an MRValue
    // so it lives in memory (dropping SRValues into a temp) and loading any
    // immutable references returned by a getitem.
    MRValue loadMR =
        emitter.emitMRValue({loadVal, argValAndExpr.expr}, EC_CallArgValue);
    if (!loadMR)
      return {};

    // After the call, write the updated value in loadMR back to the lvalue.
    afterCallActions.lvalueWritebacks.push_back(
        {ExprDest(lv, ctx), MLValue(loadMR)});
    // Use the address of the temporary for the call.
    return checkMValueAddrSpace(loadMR);
  }
  }

  llvm_unreachable("unexpected argument convention");
}

/// This function drops `byref_result` result slots from an argument list,
/// leaving only the formal arguments. This logic is valid for parameter calls
/// only.
static ArrayRef<ASTExprAnd<AnyValue>>
dropResultSlots(ArrayRef<ASTExprAnd<AnyValue>> argumentValues,
                FnTypeGeneratorType sig) {
  // TODO: What about throwing functions?
  if (sig.hasMemoryOnlyResult() &&
      sig.getNumArguments() == argumentValues.size())
    return argumentValues.drop_back();
  return argumentValues;
}

TypedAttr CallEmitter::emitCallInParamContext(
    ArrayRef<ASTExprAnd<AnyValue>> argumentValues) {
  assert(!emitter.builder && "not in parameter context");

  // TODO: We can support throwing parameter calls by inserting a 'force to
  // normal value' check which aborts (at compile time) if interpretation
  // throws an error.
  if (calleeSig.isThrows()) {
    return emitter.emitErrorForDynamicValueInParameter(
        callExpr, "cannot call raising function");
  }
  if (calleeSig.isAsync()) {
    return emitter.emitErrorForDynamicValueInParameter(
        callExpr, "cannot call async function");
  }

  // Emitting a call in a parameter context. Generate an apply operator.
  SmallVector<TypedAttr> operands({callee.getIfPValue().get()});

  // If the callee has implicit origins, we need to bind them to immortal
  // references and rebind the callee.
  FnTypeGeneratorType boundSigType = calleeSig;
  if (calleeSig.getNumImplicitOriginDecls()) {
    SmallVector<TypedAttr> implicitOrigins;
    for (auto [convention, argType] :
         llvm::zip(calleeSig.getArgConventions(), calleeSig.getArguments())) {
      if (hasImplicitOrigin(convention)) {
        auto originType = cast<RefType>(argType).getOriginType();
        implicitOrigins.push_back(ComptimeOriginAttr::get(originType));
      }
    }
    FunctionType newFnType = calleeSig.substituteImplicitOriginsIntoValues(
        implicitOrigins, [&]() -> InFlightDiagnostic {
          llvm_unreachable("substitution should always succeed");
        });
    boundSigType = calleeSig.getWithBody(
        calleeSig.getBody().getWithValuesReplaced(newFnType));

    // Rebind it to the new signature.
    // FIXME: Extend apply to handle implicit origins directly, this makes it
    // super hard to read the generated IR because of redundant signatures.
    operands[0] = ParamOperatorAttr::getRebind(operands[0], boundSigType);
  }

  auto argTypes = boundSigType.getArguments();
  auto argConventions = boundSigType.getArgConventions();
  assert(!calleeSig.isThrows() && "Throwing functions not handled");
  if (boundSigType.hasMemoryOnlyResult()) {
    argTypes = argTypes.drop_back();
    argConventions = argConventions.drop_back();
  }

  for (auto [argValAndExpr, calleeArgType, convention] :
       llvm::zip(argumentValues, argTypes, argConventions)) {
    TypedAttr arg;
    if (auto pmb = argValAndExpr.ir.getIfPMBValue()) {
      assert(hasAddress(convention) &&
             "Can only pass memory values to memory arguments");
      // If it needs to be in memory and already is, use it.
      arg = pmb;
    } else {
      arg = argValAndExpr.ir.getIfPValue();
      if (!arg)
        return emitter.emitErrorForDynamicValueInParameter(argValAndExpr.expr);

      if (hasAddress(convention))
        arg = StoreToMemAttr::get(arg, calleeArgType);
    }

    // Emit a rebind if the refined type does not match the callee arg type.
    arg = ParamOperatorAttr::getRebind(arg, calleeArgType);
    operands.push_back(arg);
  }

  Type resultType = boundSigType.getUserResultType();

  // Check to see if this is a call to a @always_inline("builtin") function,
  // like Int.__add__ etc.  If so, we need to inlined the body instead of making
  // an apply operator attr. We can only tell the inline level by finding the
  // lit.fn of the callee.  We require knowing the inline level because we have
  // to recursively resolve the body of the function, which we don't want to do
  // unilaterally.
  if (auto result =
          emitter.shared.foldInlineBuiltinFunction(operands, loc,
                                                   /*isError*/ false)) {
    assert(result.getType() == resultType && "result type mismatch");

    // Maintain sugar so diagnostics don't show the inlined function call.  If
    // the operands of the apply themselves have sugar then we don't have to
    // retain their canonical form because the canonical rep will discard it
    // all.
    auto sugaredCall = ParamOperatorAttr::get(POC::Apply, operands, resultType);
    return SugarAttr::getAlwaysInlineBuiltin(sugaredCall, result);
  }

  TypedAttr result; // Memory-only types uses ApplyResultSlot.
  if (!boundSigType.hasMemoryOnlyResult())
    result = ParamOperatorAttr::get(POC::Apply, operands, resultType);
  else
    result = ParamOperatorAttr::get(POC::ApplyResultSlot, operands, resultType);

  // If the result was a returned reference, load it before returning it.
  if (boundSigType.isRefResult()) {
    result = ParamOperatorAttr::get(
        POC::LoadFromMem, result,
        sugarCast<RefType>(result.getType()).getElementType());
  }
  return result;
}

//===----------------------------------------------------------------------===//
// IREmitter::emitCallUnchecked
//===----------------------------------------------------------------------===//

/// Emit warnings about incorrect code in a direct call.  This is invoked after
/// the full IR for the call is emitted, so we know that it was a valid call.
void CallEmitter::emitDirectCallWarnings(LIT::CallOp call,
                                         const CallOperands &callOperands) {
  // Check for a known callee.
  auto symbol = sugarDynCast<SymbolConstantAttr>(call.getCallee());
  if (!symbol)
    return;

  // Figure out what is getting called.
  ASTDecl *calleeDecl =
      emitter.getDeclResolver().getDeclForFuncSymbol(symbol.getSymbol());
  if (!calleeDecl)
    return;
  auto calleeFunc = cast<FnOp>(calleeDecl->getIfOperation());

  // The __del__ special function takes its operand as an owning reference,
  // and destroys it.  It is a bit silly, but you can call it directly on an
  // RValue and it will destroy the RValue explicitly.  However, some folks
  // will call it on a local variable (or other !RValue reference) which will
  // actually cause a COPY of the source value, and then explicitly destroy
  // this copy of the value.  Emit a warning in this case.
  if (calleeFunc.getSpecialFunctionKind() == SpecialFunctionKind::kDeinit &&
      callOperands.size() == 1 && // defensive.
      callOperands[0].ir.getIfRValue().isNull()) {
    emitter.emitWarning(loc)
        << "explicit call to '__deinit__' destroys a copy of "
           "the value; consider removing this call"
        << callOperands[0].expr->getRange();
    return;
  }
}

// As we emit the arguments, we check to see if there are any exclusivity
// violations provided by the argument.
namespace {
struct ExclusivityChecker : public SharedStateUser {
  ExclusivityChecker(RValue callee, const ExprNode *callExpr, CallSyntax syntax,
                     ArrayRef<ASTExprAnd<AnyValue>> argumentValues,
                     IREmitter &emitter)
      : SharedStateUser(emitter.shared), callee(callee), callExpr(callExpr),
        syntax(syntax), argumentValues(argumentValues),
        builder(emitter.builder), declScope(emitter.declScope) {

    // Handle __unsafe_nested_origins_read_only.
    originsAccessesAreReadOnly = getCalleeType().getIsNestedOriginsReadOnly();

    // Check capture origins first so we know if argument values may overlap.
    checkCaptureOrigins();
  }

  /// As each argument is emitted, check against previous arguments for
  /// exclusivity violations.
  void checkArgument(Value val, unsigned argIdx, FnTypeGeneratorType signature);

  /// This method takes a look at the origins accessed by the call.  If any of
  /// them might be to the current caller's stack frame, it returns true. This
  /// allows us to set the LLVM tailcall marker.
  bool mayAccessCallerStack() const;

  /// Return the type of the callee.
  FnTypeGeneratorType getCalleeType() const {
    return sugarCast<FnTypeGeneratorType>(callee.getRValueType());
  }

private:
  RValue callee;
  const ExprNode *callExpr;
  CallSyntax syntax;
  /// These are the arguments that are being emitted.
  ArrayRef<ASTExprAnd<AnyValue>> argumentValues;
  std::optional<OpBuilder> builder;
  ASTDecl &declScope;

  /// True if the __unsafe_nested_origins_read_only decorator is
  /// on the callee. Nested type-embedded origins are treated as read-only.
  bool originsAccessesAreReadOnly = false;
  /// True if the call accesses AnyOriginAttr.
  bool hasAnyOriginAccess = false;

  /// For each origin that is referenced, we keep track of what argIdx it came
  /// from, and whether it was potentially mutated.
  struct OriginInfo {
    /// The argument that accessed this origin, or the capture set if null.
    std::optional<unsigned> argIdx;
    bool isImmut;
    /// True if this is the leaf of a nested access (e.g. "a.x.y.z"), false if
    /// this is the parent origin of a leaf access (e.g. "a.x" from that
    /// reference).
    bool isLeaf;
  };
  SmallDenseMap<TypedAttr, OriginInfo, 8> originAccesses;

  /// Look at the capture origin set on the callee and register uses of them.
  /// The capture origins are considered accessed as a single unit, so they
  /// never conflict with themselves, but they may conflict with argument
  /// accesses.
  void checkCaptureOrigins();

  void checkOriginAccess(Value val, std::optional<unsigned> argIdx,
                         TypedAttr origin);

  void diagViolation(Value val, unsigned argIdx, TypedAttr origin,
                     const OriginInfo &previousAccess);
};
} // end anonymous namespace

/// This method takes a look at the origins accessed by the call.  If any of
/// them might be to the current caller's stack frame, it returns true. This
/// allows us to set the LLVM tailcall marker.
bool ExclusivityChecker::mayAccessCallerStack() const {
  // Conservatively handle AnyOriginAttr.
  if (hasAnyOriginAccess)
    return true;

  auto isParamDeclOutsideFunction = [&](ParamDeclRefAttr paramDecl) {
    // Walk up the decl hierarchy to find the one that contains the parameter.
    for (auto *curDecl = &declScope; curDecl;
         curDecl = curDecl->getParentDecl()) {
      Operation *declOp = curDecl->getIfOperation();
      if (!declOp)
        continue;

      PogListAttr paramListAttr;
      ArrayRef<ParamDeclAttr> paramDecls;
      [[maybe_unused]] size_t numImplicitOrigins = 0;
      // TODO: we need a decl interface to do this!
      if (auto fnDecl = dyn_cast<LIT::FnOp>(declOp)) {
        paramListAttr = fnDecl.getFuncTypeGenerator().getParamListAttrs();
        paramDecls = fnDecl.getParams();
        numImplicitOrigins =
            fnDecl.getFuncTypeGenerator().getNumImplicitOriginDecls();
      } else if (auto structDecl = dyn_cast<LIT::StructDeclOp>(declOp)) {
        paramListAttr = structDecl.getSignature().getParamListAttrs();
        paramDecls = structDecl.getParams();
        numImplicitOrigins = 0;
      } else
        continue;

      assert(paramListAttr.size() + numImplicitOrigins == paramDecls.size() &&
             "Unexpected number of parameters");

      for (auto [idx, param] : llvm::enumerate(paramDecls)) {
        if (param.getName() == paramDecl.getName())
          return true;
      }
    }
    // Couldn't find it in a parent decl.
    return false;
  };

  // At this point all the origins have been collected, just see if any are
  // local references. When an access to "x.y" happens we'll see both "x" and
  // "x.y" here.
  for (const auto &[origin, info] : originAccesses) {
    // Static origins and subfields are ignorable.  Fields will have their bases
    // included.
    if (isa<StaticOriginAttr, OriginFieldAttr, ComptimeOriginAttr,
            InteriorOriginAttr, OriginSubtreeAttr>(origin) ||
        // FIXME: Why are we getting UnboundAttr's here?
        isa<UnboundAttr>(origin))
      continue;

    // Scan up our decl hierarchy to see if this parameter is defined on a
    // function or struct.  If so, it can't be local to this function.
    // This seems unfortunate, there should be a better way to do this.
    auto paramDecl = dyn_cast<ParamDeclRefAttr>(origin);
    if (!paramDecl)
      origin.dump();
    assert(paramDecl && "Unknown origin in mayAccessCallerStack");
    if (!isParamDeclOutsideFunction(paramDecl))
      return true;
  }
  // If we can't find any local accesses, then we're good to go.
  return false;
}

void ExclusivityChecker::checkCaptureOrigins() {
  TypedAttr captureOrigins = getCalleeType().getCaptureOrigins();
  for (TypedAttr origin : shared.cachedOriginFinder.findOriginsIn(
           /*types=*/{}, captureOrigins)) {
    // If we are to treat all these as read-only, then strip the mutability.
    if (originsAccessesAreReadOnly)
      origin = OriginMutCastAttr::get(origin, false);
    checkOriginAccess(Value(), /*argIdx=*/{}, origin);
  }
}

/// Given an argument value being passed with a specified convention, check to
/// see if the following origin (which may be part of the argument convention,
/// or buried in the type) is a legal access given the other things we've
/// already seen.
void ExclusivityChecker::checkOriginAccess(Value val,
                                           std::optional<unsigned> argIdx,
                                           TypedAttr rawOrigin) {

  // Accesses to an origin union is an access to each of the members.
  if (auto unionAttr = sugarDynCast<OriginUnionAttr>(rawOrigin)) {
    for (auto elt : unionAttr.getOperands())
      checkOriginAccess(val, argIdx, elt);
    return;
  }

  // FIXME(MOCO-3241): Closures are capturing things too aggressively causing
  // false exclusivity errors.
  if (!argIdx) {
    hasAnyOriginAccess = true;
    return;
  }

  // Determine whether the access was immutable.
  bool isImmut = OriginType::isMutableKnown(rawOrigin, false);

  // Look through immcasts to determine the accessed origin.
  TypedAttr origin = OriginMutCastAttr::strip(rawOrigin);

  // Accesses to the global origin never conflict.
  if (sugarIsa<AnyOriginAttr>(origin)) {
    hasAnyOriginAccess = true;
    return;
  }

  assert(!sugarIsa<ParamIndexRefAttr>(origin) &&
         "unexpected ParamIndexRefAttr reached checkOriginAccess");

  // Process each level of the origin hierarchy, checking to see if we have a
  // conflict at any level.
  bool isLeaf = true; // First level is considered a leaf access.
  while (1) {
    assert(!isa<OriginUnionAttr>(origin) && !isa<OriginMutCastAttr>(origin) &&
           "union/mutcast are canonicalized to the outside");

    // Determine whether we've seen this origin before.
    auto [it, isNew] =
        originAccesses.insert({origin, {argIdx, isImmut, isLeaf}});
    if (!isNew) {
      assert(val && "capture origins cannot self-conflict");
      // If we've seen this origin before, check to see if either access is
      // potentially mutating.  Read/read aliasing is fine, but
      // write/write and read/write are not.
      if ((!isImmut || !it->second.isImmut) &&
          // Only diagnose leaf field conflicts. It's fine to mutate "a.x" and
          // "a.y" independently.
          (isLeaf || it->second.isLeaf)) {
        // If not, we have a problem!
        diagViolation(val, *argIdx, rawOrigin, it->second);
        return;
      }

      // Otherwise we have a non-conflicting access.  This can be because we
      // have a read of a subfield of another read, or because we have a
      // write/write or read/write of different subfields (e.g. 'a.x' vs 'a.y').

      // Ok, this is a read/read.  If this origin was previously seen
      // as a non-leaf, upgrade it to a leaf access, so any subsequent subfield
      // modifications are known to conflict.
      it->second.isLeaf |= isLeaf;

      // Upgrade the inner access to a write if our access is a write.
      it->second.isImmut &= isImmut;
    }

    // Only the first level is considered a leaf access.
    isLeaf = false;

    // See if this is field access, process the base next.
    if (auto fieldAttr = dyn_cast<OriginFieldAttr>(origin)) {
      origin = fieldAttr.getBase();
      continue;
    }

    // If this is derived from an interior origin, process the base as an
    // immutable access.
    if (auto interior = dyn_cast<InteriorOriginAttr>(origin)) {
      origin = interior.getBase();
      // This is going to read the base.
      isImmut = true;
      continue;
    }

    // If this is derived from an subtree origin, process the base as an
    // immutable access.
    if (auto subtree = dyn_cast<OriginSubtreeAttr>(origin)) {
      origin = subtree.getBase();
      // Don't set isImmut; this could mutate the base if it's a mutable origin.
      isLeaf = true;
      continue;
    }

    // We're done.
    break;
  }
}

/// As each argument is emitted, check against previous arguments for
/// exclusivity violations. This handles normal scalar arguments as well as
/// unpacking the elements passed to variadic lists and packs and processing
/// each independently.
void ExclusivityChecker::checkArgument(Value argVal, unsigned argIdx,
                                       FnTypeGeneratorType signature) {
  ArgConvention convention = signature.getArgConvention(argIdx);

  // If this is a result argument, then we only look at the origin of the
  // destination that we're storing into, not any nested references that may
  // be in the result. This returned value is derived from the other arguments
  // passed to the function, it doesn't conflict with them.
  if (convention == ArgConvention::ByRefResult ||
      convention == ArgConvention::ByRefError) {
    argVal = RebindOp::strip(argVal);
    checkOriginAccess(argVal, argIdx,
                      sugarCast<RefType>(argVal.getType()).getOrigin());
    return;
  }

  // We get passed the MLIR representation for the dynamic argument, which
  // includes variadic and pack constructions.  Make sure to handle each
  // variadic argument separately.
  auto checkArg = [&](Value argVal) {
    // We sometimes get rebinds for downcasts of origins or for sugar alignment.
    // Ignore those so we can see the actual incoming value's origin.
    argVal = RefImmutOp::stripRebinds(argVal);

    // Return true if this is the origin for the argument itself.
    auto isArgOrigin = [&](TypedAttr origin) -> bool {
      return hasAddress(convention) &&
             isEqualCanon(cast<RefType>(argVal.getType()).getOrigin(), origin);
    };

    // Find all the of the origins that are buried in the specified type.
    for (TypedAttr origin :
         shared.cachedOriginFinder.findOriginsIn(argVal.getType())) {
      // We may have looked through a rebind and got to a mutable origin. If
      // we are only reading, force the origin back to immutable so we don't
      // get confused.
      bool forceToImmutOrigin = false;
      if (isArgOrigin(origin) && hasAddress(convention) &&
          convention != ArgConvention::Mut &&
          cast<RefType>(signature.getArgument(argIdx)).isMutableKnown(false))
        forceToImmutOrigin = true;

      // Callees marked `@__unsafe_nested_origins_read_only` promise
      // not to mutate origins of 'ref' arguments or nested arguments, but still
      // affect 'mut' and result slots.
      if (originsAccessesAreReadOnly) {
        if (!isArgOrigin(origin) || convention == ArgConvention::Ref)
          forceToImmutOrigin = true;
      }

      if (forceToImmutOrigin)
        origin = OriginMutCastAttr::get(origin, false);
      checkOriginAccess(argVal, argIdx, origin);
    }
  };

  // Normal arguments.
  if (!signature.isPack(argIdx) && !signature.isPosVarArg(argIdx)) {
    checkArg(argVal);
    return;
  }

  // Handle VariadicList and VariadicPack.  They are constructed objects dumped
  // into a VarDecl because they need to be passed to the callee with ownership.
  convention = signature.getVariadicConvention(argIdx);
  TypedAttr extraOrigin; // Unused here.
  for (auto elt :
       OriginTrackable::decodeIndividualVariadicArguments(argVal, extraOrigin))
    checkArg(elt);
}

/// Emit an error about an access to a conflicting origin after a previous
/// access was seen.
void ExclusivityChecker::diagViolation(Value val, unsigned argIdx,
                                       TypedAttr origin,
                                       const OriginInfo &previousAccess) {
  bool isImmut = OriginType::isMutableKnown(origin, false);
  origin = OriginMutCastAttr::strip(origin);

  FnTypeGeneratorType calleeType = getCalleeType();
  MojoInflightDiag diag = emitError(callExpr->getLoc());
  ArgConvention convention = calleeType.getArgConvention(argIdx);

  diag << "aliasing values passed ";
  diag << (previousAccess.isImmut ? "immutably" : "mutably");
  if (std::optional<unsigned> prevIdx = previousAccess.argIdx) {
    diag << " to " << calleeType.getArgName(*prevIdx) << " argument";
    diag << argumentValues[*prevIdx].expr->getRange();
  } else {
    // TODO: Dig into the closure to get better error messages.
    diag << " as an implicit closure capture";
  }

  diag << " and ";
  if (convention != ArgConvention::ByRefResult) {
    diag << "passed " << (isImmut ? "immutably" : "mutably") << " to "
         << calleeType.getArgName(argIdx) << " argument";
    diag << argumentValues[argIdx].expr->getRange();
  } else {
    diag << "constructed as a result";
  }

  diag << " in ";
  switch (syntax) {
  default:
    // If the callee is a direct call, dig out the source name.
    if (PValue pv = callee.getIfPValue()) {
      if (auto symbol = dyn_cast<SymbolConstantAttr>(pv.get())) {
        // Figure out what is getting called and include it.
        if (ASTDecl *calleeDecl =
                getDeclResolver().getDeclForFuncSymbol(symbol.getSymbol())) {
          auto calleeFunc = cast<FnOp>(calleeDecl->getIfOperation());

          // Print "'Type' initializer" instead of just '__init__'.
          if (calleeFunc.getSpecialFunctionInfo().isInitializer()) {
            // Use the bound result type, which is the result of the init,
            // not the generic unbound one from the containing decl.
            diag << ASTType(calleeType.getUserResultType()) << " initializer ";
          } else if (auto sourceName = calleeFunc.getSourceNameAttr()) {
            diag << sourceName << ' ';
          }
        }
      }
    }
    diag << "call";
    break;
  case CallSyntax::kImplicitConvert:
    diag << "implicit conversion to "
         << ASTType(calleeType.getUserResultType());
    break;
  case CallSyntax::kImplicitCopyCtor:
    diag << "implicit copy constructor of "
         << ASTType(calleeType.getUserResultType());
    break;
  case CallSyntax::kImplicitMoveCtor:
    diag << "implicit move constructor of "
         << ASTType(calleeType.getUserResultType());
    break;
  }

  // Attach a note to explain what is going on in more detail.
  diag.attachNote(callExpr->getLoc());

  // If the origin in question is because of the top-level ref binding, then
  // we have a common problem where something is passed both mutable and
  // borrowed.
  if (calleeType.isAnyVarArg(argIdx))
    convention = calleeType.getVariadicConvention(argIdx);

  // Special case the result slot.
  if (convention == ArgConvention::ByRefResult) {
    diag << "introduce a temporary to avoid mutating the call result while "
            "accessing it through an argument";
    return;
  }

  if (hasAddress(convention) &&
      OriginMutCastAttr::strip(sugarCast<RefType>(val.getType()).getOrigin()) ==
          origin) {
    diag << "'" << ASTType::getOriginAsString(origin, &shared)
         << "' value is passed through aliasing '" << getUserSyntax(convention)
         << "' argument " << calleeType.getArgName(argIdx)
         << argumentValues[argIdx].expr->getRange();
    return;
  }

  ASTType argType = RefType::stripRefConvention(val.getType(), convention);

  // Otherwise, it is a more complicated buried origin in a type like a
  // Reference or Span.
  diag << "'" << ASTType::getOriginAsString(origin, &shared)
       << "' memory accessed through reference embedded in value of type "
       << argType;

  // Diagnostics can be very confusing when generated by synthesized functions
  // like trait stubs.  Generate a note when this happens to make it more
  // obvious what is going on.
  if (builder && builder->getBlock()) {
    Block &block = *builder->getBlock();
    FnOp parentFunc = dyn_cast<FnOp>(block.getParentOp());
    if (!parentFunc)
      parentFunc = block.getParentOp()->getParentOfType<FnOp>();
    if (parentFunc && parentFunc.isSynthetic()) {
      // sourceName
      diag.attachNote(parentFunc.getLoc()) << "in synthesized method";
      if (auto sourceName = parentFunc.getSourceNameAttr())
        diag << ' ' << sourceName;
    }
  }
}

/// When emitting a call where all of the arguments are PValues, and the callee
/// is @always_inline("builtin"), we can safely emit the call in a parameter
/// context.  We know it doesn't have side effects because of the checks that
/// @always_inline("builtin") performs.
static bool shouldEmitParameterCall(RValue callee,
                                    ArrayRef<ASTExprAnd<AnyValue>> argValues,
                                    SharedState &shared) {
  auto calleeSig = sugarCast<FnTypeGeneratorType>(callee.getRValueType());

  // If this returns something like an UnsafePointer with an unusual origin,
  // materialization will fail, so we can't emit it as a parameter call.
  if (ASTType(calleeSig.getUserResultType())
          .containsUnmaterializableOrigins(shared))
    return false;

  argValues = dropResultSlots(argValues, calleeSig);

  // We cannot inline this if any of the arguments are dynamic.
  auto isPValue = [](ASTExprAnd<AnyValue> arg) { return arg.ir.getIfPValue(); };
  if (!callee.getIfPValue() || !llvm::all_of(argValues, isPValue))
    return false;

  // If this is an @always_inline("builtin") function, we must emit its body
  // inline.
  if (auto calleeSymbolCst = sugarDynCast<SymbolConstantAttr>(
          ParamOperatorAttr::stripRebind(callee.getIfPValue()))) {
    if (ASTDecl *calleeDecl = shared.getDeclResolver().getDeclForFuncSymbol(
            calleeSymbolCst.getSymbol())) {
      if (cast<FnOp>(calleeDecl->getIfOperation()).getInlineLevel() ==
          InlineLevel::AlwaysBuiltin)
        return true;
    }
  }
  return false;
}

/// Rewrite closure-capture witnesses inside `type` back to the external value
/// they are known to equal.
static Type externClosureCapturesInType(Type type, ASTDecl &declScope) {
  // Start the capture search from the nearest enclosing operation.
  Operation *startOp = nullptr;
  for (ASTDecl *scope = &declScope; scope && !startOp;
       scope = scope->getParentDecl())
    startOp = scope->getIfOperation();
  if (!startOp)
    return type;

  SharedState &shared = declScope.getShared();

  // `startOp` is fixed for this call, so memoize the capture lookup per closure
  // parameter name.
  DenseMap<StringAttr, ArrayRef<ClosureParamCapture>> captureCache;
  auto capturesFor = [&](StringAttr name) -> ArrayRef<ClosureParamCapture> {
    auto [it, inserted] = captureCache.try_emplace(name);
    if (inserted)
      it->second = shared.lookupClosureCaptureFromOp(startOp, name);
    return it->second;
  };

  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](GetWitnessAttr witness) -> TypedAttr {
    auto closureParam = sugarDynCast<ParamDeclRefAttr>(witness.getTypeValue());
    if (!closureParam)
      return witness;
    for (const ClosureParamCapture &capture :
         capturesFor(closureParam.getName()))
      if (capture.first == witness.getWitnessName())
        return ParamDeclRefAttr::get(witness.getWitnessName(),
                                     witness.getType());
    return witness;
  });
  return replacer.replace(type);
}

CValue IREmitter::emitCallUnchecked(RValue callee,
                                    CallOperands &&callOperands) {
  const ExprNode *callExpr = callOperands.callExpr;
  CallEmitter callEmitter(callee, callExpr, *this, callOperands.dest);
  auto calleeSig = callEmitter.calleeSig;
  // Function literals might have been converted.
  callee = callEmitter.callee;

  // Check to see if the callee is a throwing function whose thrown type
  // mismatches any fixed contextual error type, but which is implicitly
  // convertible.  In this case, we want to compile something like:
  //   def test() raises Float32: throws_int()
  // into:
  //   def test() raises Float32:
  //     try:
  //       throws_int()
  //     except err:
  //       raise err
  // This is safe because the thrown type is implicitly convertible to Float32,
  // and the exception will be implicitly converted when rethrown.
  if (calleeSig.isThrows() && builder) {
    MLValue errSlot = findNearestErrorSlot();
    ASTType thrownType = calleeSig.getUserThrownType();

    auto isErrorTypeConvertible = [&]() -> bool {
      // FIXME: ImplicitOrigins aren't bound here, but implicit origins may
      // be in the error type of callee_sig, which will break implicit
      // conversion checking below.  Implicit origins need to go away!
      // Example:
      //  def callee(a: String) raises Pointer[String, origin_of(a)]: ...
      //  def caller(a: String) raises Pointer[String, origin_of(a)]:
      //      callee(a)  # calee.error type is a pointer referring to a's
      //      origin.
      mlir::AttrTypeWalker walker;
      // This is overly conservative because it isn't tracking depths.
      walker.addWalk(
          [](ImplicitOriginRefAttr sig) { return WalkResult::interrupt(); });
      if (walker.walk(thrownType).wasInterrupted())
        return false;

      return canImplicitlyConvertToType(
          {UnboundAttr::get(thrownType), callExpr}, errSlot.getRValueType(),
          getDeclScope());
    };

    if (errSlot &&
        // We don't need a temp if the types line up, which is common.
        !isEqualCanon(thrownType, errSlot.getRValueType()) &&
        // Inferred contextual types will be resolved correctly
        !sugarIsa<UnresolvedType>(errSlot.getRValueType()) &&
        // Throw NeverType isn't actually thrown.
        !sugarIsa<NeverType>(thrownType) &&
        // Thrown type must be implicitly convertible to the error slot type.
        isErrorTypeConvertible()) {

      return emitIndirectCallInTryBlock(
          callee, std::move(callOperands), [&](VarDeclOp errDecl) {
            // Move the error out of the temporary and into the overall error
            // slot, performing the implicit conversion.
            ExprDest moveDest(errSlot, EC_RaiseValue);
            (void)emitResult(MRValue(errDecl), callExpr, moveDest);
            auto loc = translateLocation(callExpr->getLoc());
            RaiseOp::create(*builder, loc);
            TryYieldOp::create(*builder, loc);
            // Check for invalid origin references.
            checkInferredErrorType(errDecl.getType().getElementType(),
                                   callExpr->getLoc());
          });
    }
  }

  // Capture the types of any "fully sugared" arguments that are passed in so
  // we can propagate them to the result type.  We capture the sugared forms of
  // things so we compile generic calls (e.g. SIMD.add(someInt8, 42)) into a
  // result type of "Int8" even though the signature indicates "SIMD[Int8, 1]".
  //
  // Note that these types usually won't line up with the signature type
  // (for example, N values passed into a variadic could have N entries here,
  // not one) and the types may not match the signature.
  SmallVector<ASTType> sugaredActualArgumentTypes;
  for (const OperandValue &argVal : callOperands.values) {
    if (auto actualType = argVal.ir.getRValueTypeIfResolvable()) {
      if (SugarAttr::hasTopLevelSugar(actualType))
        sugaredActualArgumentTypes.push_back(actualType);
    }
  }
  // Given the final result type for the call, if it isn't fully sugared, see if
  // there is anything else we can learn from.  This is intended to propagate
  // "fully generic" operations with input sugar.
  auto findSugaredResultTypeFor = [&](Type userResultType) -> Type {
    // If the result is already sugared, leave it alone.
    if (SugarAttr::hasTopLevelSugar(userResultType))
      return {};
    // Otherwise, take the first thing that is structurally the same but that is
    // sugared.
    for (auto sugaredType : sugaredActualArgumentTypes) {
      if (sugaredType && isEqualCanon(sugaredType, userResultType))
        return sugaredType;
    }
    return {};
  };

  // We first emit all the arguments.
  auto argumentValuesOr = callEmitter.emitArgValues(callOperands);
  if (failed(argumentValuesOr))
    return {};

  ArrayRef<ASTExprAnd<AnyValue>> argumentValues = (*argumentValuesOr).first;
  const llvm::BitVector &isDefaultMask = (*argumentValuesOr).second;

  if (!builder || shouldEmitParameterCall(callee, argumentValues, shared)) {
    TypedAttr paramCallResult;
    {
      llvm::SaveAndRestore savedBuilder(builder, {});
      assert(callOperands.dest.getContext() != EC_InvalidContext &&
             "parametric emitCallUnchecked must include an ExprContext");
      llvm::SaveAndRestore savedContext(paramContext,
                                        callOperands.dest.getContext());
      argumentValues = dropResultSlots(argumentValues, calleeSig);
      paramCallResult = callEmitter.emitCallInParamContext(argumentValues);
    }

    // Propagate fully-sugared input types to the result if any line up.
    if (paramCallResult) {
      if (auto sugaredResult =
              findSugaredResultTypeFor(paramCallResult.getType()))
        paramCallResult =
            ParamOperatorAttr::getRebind(paramCallResult, sugaredResult);
    }

    // Re-express any forwarded closure-capture witness as captured type.
    if (paramCallResult) {
      Type externed =
          externClosureCapturesInType(paramCallResult.getType(), declScope);
      if (externed != paramCallResult.getType())
        paramCallResult =
            ParamOperatorAttr::getRebind(paramCallResult, externed);
    }

    // The dest might force further calls.  We delay calling it until after
    // restoring the builder so that it is NOT forced to be in the parameter
    // context.  In particular, dest may cause a call to set the paramCallResult
    // into a DLValue.
    return emitCResult(paramCallResult, callExpr, callOperands.dest);
  }

  // Ok, we're going to emit this expression as a dynamic call.  If all of the
  // arguments are PValues and any of them are not materializable, then emit a
  // custom error message with a nice fixit that pulls a "comptime" around the
  // call.
  auto isPValue = [](ASTExprAnd<AnyValue> arg) { return arg.ir.getIfPValue(); };
  if (llvm::all_of(argumentValues, isPValue) &&
      !ASTType(calleeSig.getUserResultType()).isNoneType()) {
    for (auto [argIdx, argValAndExpr] : llvm::enumerate(argumentValues)) {
      auto argType = argValAndExpr.ir.getIfPValue().getRValueType();
      const ExprNode *expr = argValAndExpr.expr;
      if (argType.isImplicitlyCopyable(expr->getLoc(), shared, declScope))
        continue;
      // We allow implicitly materializing default values to runtime.
      if (isDefaultMask.test(argIdx))
        continue;

      auto diag = emitError(expr->getLoc(),
                            "cannot materialize comptime value of type ")
                  << argType
                  << " to runtime because it is not 'ImplicitlyCopyable'"
                  << expr->getRange();
      diag.attachNote(callExpr->getLoc())
          << "use 'comptime' to evaluate the entire call at 'comptime' and "
             "materialize its result"
          << FixIt::insertBeforeToken(callExpr->getRangeStart(), "comptime(")
          << FixIt::insertAfterToken(callExpr->getRangeEnd(), ")",
                                     shared.diags);
      return {};
    }
  }

  // Otherwise, materialize PValue and DLValue's as SSA values for emission.
  Location loc = translateLocation(callExpr->getLoc());

  // As we emit the arguments, we check to see if there are any exclusivity
  // violations provided by the argument.
  ExclusivityChecker exclusivityChecker(callee, callExpr, callOperands.syntax,
                                        argumentValues, *this);

  SmallVector<Value> callArgs;
  SmallVector<TypedAttr> implicitOrigins;
  ArrayRef<ArgConvention> conventions = calleeSig.getArgConventions();

  for (auto [argIdx, argValAndExpr, conventionX, declaredArgTypeX] :
       llvm::enumerate(argumentValues, conventions, calleeSig.getArguments())) {
    ArgConvention convention = conventionX;
    Type declaredArgType = declaredArgTypeX;

    // See if we have an implicit origin bound for this argument.
    bool needsImplicitOrigin =
        hasImplicitOrigin(convention) &&
        isa<ImplicitOriginRefAttr>(cast<RefType>(declaredArgType).getOrigin());

    if (isResultSlot(convention)) {
      // Async function signatures have results slots even though they are not
      // actually provided.
      // TODO: Why are these in the signature, why do they take implicit
      // origins for these things?
      if (calleeSig.isAsync()) {
        implicitOrigins.push_back(OriginUnionAttr::get(getContext()));
        continue;
      }

      // 'ref' results and typed errors can have origins derived from implicit
      // origins of earlier arguments.  Make sure to remap the implicit origins
      // into place.  This works because both result slots are at the end of the
      // list.
      // Add implicit origins for the error+result or just result slot.
      size_t numImpOrigins =
          calleeSig.getNumImplicitOriginDecls() - implicitOrigins.size();
      implicitOrigins.append(
          numImpOrigins, AnyOriginAttr::get(getContext(), /*isMutable=*/true));
      FunctionType remappedCalleeType =
          calleeSig.substituteImplicitOriginsIntoValues(
              implicitOrigins, [&]() -> InFlightDiagnostic {
                llvm_unreachable("substitution should always succeed");
              });
      implicitOrigins.resize(implicitOrigins.size() - numImpOrigins);
      declaredArgType = remappedCalleeType.getInput(argIdx);
    }

    bool isDefaultArgVal = isDefaultMask.test(argIdx);
    Value arg = callEmitter.emitPreemittedArgumentAsDynamicValue(
        argValAndExpr, isDefaultArgVal, convention, declaredArgType, callArgs);
    if (!arg)
      return {};

    // If we need an implicit origin, add it to the list.
    if (needsImplicitOrigin)
      implicitOrigins.push_back(cast<RefType>(arg.getType()).getOrigin());

    // The argument looks good on its own, check to see if it is an exclusivity
    // violation with a previous argument.
    exclusivityChecker.checkArgument(arg, argIdx, calleeSig);

    // All looks good!
    callArgs.push_back(arg);
  }

  // Now that we have the origins for the arguments, we can calculate what the
  // substituted signature should be.
  FunctionType expectedCalleeType =
      calleeSig.substituteImplicitOriginsIntoValues(
          implicitOrigins, [&]() -> InFlightDiagnostic {
            llvm_unreachable("substitution should always succeed");
          });

  // Now that all of the arguments have been emitted, coerce them to the
  // expected type if needed.  We do this after the first pass above, because
  // there can be forward references from the result slot to the later
  // arguments' origins.
  for (auto &&[arg, expectedType] :
       llvm::zip(callArgs, expectedCalleeType.getInputs())) {
    // Make sure the parameters of an argument line up by emitting a rebind
    // operation.
    if (arg.getType() == expectedType)
      continue;

    // Use rebindValue to do this so we get assertions and checks.
    arg = rebindValue({SRValue(arg), callExpr}, expectedType).getIfSRValue();
    assert(arg && "rebindValue always succeeds");
  }

  assert(expectedCalleeType.getResults().size() == 1 &&
         "All mojo functions return one value");
  Type resultType = expectedCalleeType.getResults()[0];
  CValue callResult;
  if (auto target = callee.getIfPValue()) {
    if (calleeSig.isAsync()) {
      // If the callee is an async function, emit an async call. Then wrap the
      // `!co.routine<T>` result in a `Coroutine[T]` object.
      auto call = AsyncCallOp::create(*builder, loc, target.get(),
                                      implicitOrigins, callArgs);
      // Emit the implicit conversion to Coroutine[T]. We emit into the call's
      // destination to avoid an extra copy/move of the Coroutine object.
      callResult = materializeAsyncCallAsCoroutine(
          *this, call, callExpr, calleeSig, callOperands.dest);
      if (!callResult)
        return {};
    } else {
      auto call = CallOp::create(*builder, loc, resultType, target.get(),
                                 implicitOrigins, callArgs);
      callResult = SRValue(call.getResult(0));

      // Set the LLVM "tail" marker if the call may not access the caller's
      // stack.  This flag doesn't force a tail call, it says the call is
      // eligible because it doesn't access the current stack frame.
      if (!exclusivityChecker.mayAccessCallerStack())
        call.setTailKind(TailKind::Tail);

      // If there are any callee-specific warnings to emit, do so after
      // successfully emitting the call.
      callEmitter.emitDirectCallWarnings(call, callOperands);
    }
  } else {
    // TODO(MOCO-788): We need a `lit.async.call_indirect` to model indirect
    // async calls.
    if (calleeSig.isAsync()) {
      emitError(callExpr->getLoc())
          << "TODO: indirect calls to async functions not yet supported";
      return {};
    }
    // If the callee isn't a PValue, it must be a dynamic callee.
    Value calleeVal = emitSRValue({callee, callExpr}, EC_CallCalleeValue);
    assert(calleeVal && "don't have a callee of expected type");
    auto call = CallIndirectOp::create(*builder, loc, resultType, calleeVal,
                                       implicitOrigins, callArgs);
    callResult = SRValue(call.getResult(0));

    if (!exclusivityChecker.mayAccessCallerStack())
      call.setTailKind(TailKind::Tail);
  }

  // If there were any writebacks to handle, emit them before handling raised
  // errors.
  callEmitter.emitAfterCallActions();

  // If there is a memory result slot, the value we filled in is our MRValue
  // result and we've already handled the ExprDest by emitting into it.
  if (calleeSig.hasMemoryOnlyResult() && !calleeSig.isAsync()) {
    callResult = MRValue(callArgs.back());
  }

  // If returning a reference, we need to convert to an MValue from
  // the SRValue we've got.
  if (calleeSig.isRefResult()) {
    auto resultVal = emitSRValue({callResult, callExpr}, EC_CallCalleeValue);
    if (!resultVal)
      return {};

    // Use the appropriate classification for the value based on its mutability.
    callResult = CValue::getMValueForRef(resultVal);
  }

  // In the case of something like "someInt8+42" we want to turn the
  // result into "Int8" even though the signature would make it a "SIMD[Int8,1]"
  if (auto sugaredResult =
          findSugaredResultTypeFor(callResult.getRValueType())) {
    if (callResult.isMValue())
      sugaredResult = callResult.getMValueType().getWithElement(sugaredResult);
    callResult = rebindValue({callResult, callExpr}, sugaredResult);
    assert(callResult && "rebindValue always succeeds");
  }

  // Re-express any forwarded closure-capture witness as captured type.
  if (callResult) {
    Type externed =
        externClosureCapturesInType(callResult.getRValueType(), declScope);
    if (externed != callResult.getRValueType()) {
      if (callResult.isMValue())
        externed = callResult.getMValueType().getWithElement(externed);
      callResult = rebindValue({callResult, callExpr}, externed);
      assert(callResult && "rebindValue always succeeds");
    }
  }

  // Apply type refinement to the call result if it has a parametric type
  // that could be refined via where-clause constraints in the calling scope.
  // This enables calling trait methods or passing aliased types to functions
  // when the underlying type parameter is refined.
  if (callResult)
    callResult = maybeEmitRefinementRebind({callResult, callExpr}, *this);

  // Otherwise, register-passable results are the call result which may need to
  // be emitted into a ExprDest.
  return emitCResult(callResult, callExpr, callOperands.dest);
}
