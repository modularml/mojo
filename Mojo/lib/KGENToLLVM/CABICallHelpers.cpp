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

#include "CABICallHelpers.h"
#include "CABILowering.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// CABICallHelper implementation
//===----------------------------------------------------------------------===//

std::unique_ptr<CABIInfo> CABICallHelper::createABIHandler() const {
  const llvm::Triple &triple = tc->getTarget().getTriple();
  // POPToLLVMTypeConverter inherits from LLVMDataLayout
  const LLVMDataLayout &dataLayout = *tc;
  return M::KGEN::createCABIInfo(triple, ctx, dataLayout);
}

std::pair<LLVMFunctionType, bool>
CABICallHelper::buildFunctionType(const SignatureClassification &sigClass,
                                  ValueRange originalArgs, Type origRetTy,
                                  size_t numFixedArgs, bool isVariadic) const {
  ArrayRef<CoercionInfo> argClass = sigClass.args;
  const CoercionInfo &retClass = sigClass.ret;
  SmallVector<Type> paramTypes;
  Type resultType;
  bool usesSRet = retClass.useSRet;

  // Add sret parameter if needed
  if (usesSRet) {
    paramTypes.push_back(LLVMPointerType::get(ctx));
  }

  // Build parameter types (only fixed params for variadic functions)
  size_t numParams = isVariadic ? numFixedArgs : argClass.size();
  for (size_t idx = 0; idx < numParams; ++idx) {
    const auto &coercion = argClass[idx];
    if (coercion.isIdentity()) {
      paramTypes.push_back(originalArgs[idx].getType());
    } else if (coercion.useIndirect) {
      paramTypes.push_back(LLVMPointerType::get(ctx));
    } else if (coercion.coercedType) {
      paramTypes.push_back(coercion.coercedType);
      if (coercion.isTwoRegister()) {
        assert(coercion.coercedSecondType &&
               "two-register classification must have coercedSecondType");
        paramTypes.push_back(coercion.coercedSecondType);
      }
    }
  }

  // Build result type
  if (usesSRet) {
    resultType = LLVMVoidType::get(ctx);
  } else if (retClass.isTwoRegister()) {
    // Two-register return: use LLVM struct {firstType, secondType}
    assert(retClass.coercedType && retClass.coercedSecondType &&
           "two-register classification must have both coerced types");
    resultType = LLVMStructType::getLiteral(
        ctx, {retClass.coercedType, retClass.coercedSecondType});
  } else if (retClass.coercedType) {
    resultType = retClass.coercedType;
  } else if (origRetTy) {
    resultType = origRetTy;
  } else {
    resultType = LLVMVoidType::get(ctx);
  }

  auto funcType = LLVMFunctionType::get(resultType, paramTypes, isVariadic);
  return {funcType, usesSRet};
}

Value CABICallHelper::createEntryBlockAlloca(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) const {
  auto ptrType = LLVMPointerType::get(ctx);

  // Find the enclosing LLVM function.
  auto func = parentOp->getParentOfType<LLVMFuncOp>();
  assert(func && "external_call must be inside an LLVM function");

  // Save current insertion point and emit alloca at the start of the
  // entry block.
  OpBuilder::InsertionGuard guard(rewriter);
  Block &entryBlock = func.getBody().front();
  rewriter.setInsertionPointToStart(&entryBlock);

  Value count =
      ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(1));
  return AllocaOp::create(rewriter, loc, ptrType, elemType, count,
                          /*alignment=*/0);
}

Value CABICallHelper::bitcastViaMemory(
    Value sourceValue, Type destType, Type allocType, Location loc,
    ConversionPatternRewriter &rewriter) const {
  Value alloca = createEntryBlockAlloca(rewriter, loc, allocType);
  StoreOp::create(rewriter, loc, sourceValue, alloca);
  return LoadOp::create(rewriter, loc, destType, alloca);
}

Value CABICallHelper::createOffsetGEP(
    Value basePtr, int64_t byteOffset, Location loc,
    ConversionPatternRewriter &rewriter) const {
  auto ptrType = LLVMPointerType::get(ctx);
  return GEPOp::create(rewriter, loc, ptrType, rewriter.getI8Type(), basePtr,
                       GEPArg(byteOffset), GEPNoWrapFlags::inbounds);
}

SmallVector<Value> CABICallHelper::prepareTwoRegisterArgument(
    Value originalArg, Type firstType, Type secondType, Location loc,
    ConversionPatternRewriter &rewriter) const {
  // Allocate struct containing both register types
  Type allocType = LLVMStructType::getLiteral(ctx, {firstType, secondType});
  Value alloca = createEntryBlockAlloca(rewriter, loc, allocType);

  // Store original struct
  StoreOp::create(rewriter, loc, originalArg, alloca);

  // Load first register at offset 0
  Value first = LoadOp::create(rewriter, loc, firstType, alloca);

  // Load second register at the byte offset of the second struct field.
  // This equals getTypeStoreSize(firstType) because all implemented ABIs
  // guarantee firstType is exactly 8 bytes (i64 or f64 for System V AMD64;
  // i64 for ARM64 AAPCS non-HFA), so the LLVM struct {firstType, secondType}
  // never has alignment padding between the two slots.
  int64_t offset = tc->getTypeStoreSize(firstType);
  Value gep = createOffsetGEP(alloca, offset, loc, rewriter);
  Value second = LoadOp::create(rewriter, loc, secondType, gep);

  return {first, second};
}

Value CABICallHelper::handleTwoRegisterReturn(
    Value callResult, Type firstType, Type secondType, Type originalReturnType,
    Location loc, ConversionPatternRewriter &rewriter) const {
  // Allocate struct containing both register types
  Type allocType = LLVMStructType::getLiteral(ctx, {firstType, secondType});
  Value alloca = createEntryBlockAlloca(rewriter, loc, allocType);

  // Extract and store first register
  Value first = ExtractValueOp::create(rewriter, loc, callResult, 0);
  StoreOp::create(rewriter, loc, first, alloca);

  // Extract and store second register at the byte offset of the second field.
  // See prepareTwoRegisterArgument for why this offset is always correct.
  int64_t offset = tc->getTypeStoreSize(firstType);
  Value gep = createOffsetGEP(alloca, offset, loc, rewriter);
  Value second = ExtractValueOp::create(rewriter, loc, callResult, 1);
  StoreOp::create(rewriter, loc, second, gep);

  // Load as original struct type
  return LoadOp::create(rewriter, loc, originalReturnType, alloca);
}

SmallVector<Value>
CABICallHelper::prepareArg(const CoercionInfo &coercion, Value originalArg,
                           Location loc,
                           ConversionPatternRewriter &rewriter) const {
  SmallVector<Value> result;

  if (coercion.isIdentity()) {
    // No coercion - pass directly
    result.push_back(squashPointlessCasts(originalArg));
    return result;
  }

  if (coercion.useIndirect) {
    // Allocate stack space, store struct, pass pointer
    Value alloca = createEntryBlockAlloca(rewriter, loc, originalArg.getType());
    StoreOp::create(rewriter, loc, originalArg, alloca);
    result.push_back(alloca);
    return result;
  }

  // Bitcast struct to integer/float type via store-to-stack + load pattern.
  //
  // Following Clang's approach: allocate using the coerced type(s) to
  // ensure the allocation is large enough for the loads. This prevents UB
  // when coercion rounds up (e.g., 3-byte struct -> i32 = 4 bytes).
  assert(coercion.coercedType && "non-identity, non-indirect coercion must "
                                 "have a coercedType — classifier bug");

  if (coercion.isTwoRegister()) {
    assert(coercion.coercedSecondType &&
           "two-register classification must have coercedSecondType");
    // Two registers: prepare both register values
    auto values =
        prepareTwoRegisterArgument(originalArg, coercion.coercedType,
                                   coercion.coercedSecondType, loc, rewriter);
    result.append(values.begin(), values.end());
  } else {
    // Single register: use bitcast helper
    Value coercedVal = bitcastViaMemory(originalArg, coercion.coercedType,
                                        coercion.coercedType, loc, rewriter);
    result.push_back(coercedVal);
  }

  return result;
}

std::pair<SmallVector<Value>, Value> CABICallHelper::buildCallArgs(
    ArrayRef<CoercionInfo> argClass, const CoercionInfo &retClass,
    ValueRange originalArgs, Type origRetTy, Location loc,
    ConversionPatternRewriter &rewriter) const {
  SmallVector<Value> callArgs;
  Value sretPointer = nullptr;

  // Prepare sret pointer if needed
  if (retClass.useSRet) {
    sretPointer = createEntryBlockAlloca(rewriter, loc, origRetTy);
    callArgs.push_back(sretPointer);
  }

  // Prepare each argument
  for (auto [idx, coercion] : llvm::enumerate(argClass)) {
    SmallVector<Value> coercedArgs =
        prepareArg(coercion, originalArgs[idx], loc, rewriter);
    callArgs.append(coercedArgs.begin(), coercedArgs.end());
  }

  return {callArgs, sretPointer};
}

Value CABICallHelper::extractReturn(const CoercionInfo &retClass,
                                    Value callResult, Value sretPointer,
                                    Type origRetTy, Location loc,
                                    ConversionPatternRewriter &rewriter) const {
  if (retClass.useSRet) {
    // Load from sret pointer
    return LoadOp::create(rewriter, loc, origRetTy, sretPointer);
  }

  if (retClass.isIdentity()) {
    // No coercion - use directly
    return callResult;
  }

  if (retClass.isTwoRegister()) {
    assert(retClass.coercedType && retClass.coercedSecondType &&
           "two-register classification must have both coerced types");
    // Two-register return: extract both values and reconstruct struct
    return handleTwoRegisterReturn(callResult, retClass.coercedType,
                                   retClass.coercedSecondType, origRetTy, loc,
                                   rewriter);
  }

  // Single-register: bitcast from coerced type back to struct
  assert(retClass.coercedType &&
         "non-identity, non-sret, non-two-register return must have a "
         "coercedType — classifier bug");
  return bitcastViaMemory(callResult, origRetTy, retClass.coercedType, loc,
                          rewriter);
}

CABICallPrep CABICallHelper::prepareCall(TypeRange argTypes,
                                         ValueRange argValues, Type origRetTy,
                                         Location loc,
                                         ConversionPatternRewriter &rewriter,
                                         size_t numFixedArgs,
                                         bool isVariadic) const {
  auto handler = createABIHandler();

  // Classification runs on LLVM types so register/layout rules see the same
  // shape the C compiler does.
  SmallVector<Type> llvmArgTypes;
  llvmArgTypes.reserve(argTypes.size());
  for (Type t : argTypes)
    llvmArgTypes.push_back(tc->convertType(t));
  Type llvmRetTy = origRetTy ? tc->convertType(origRetTy) : Type();

  SignatureClassification sigClass =
      handler->computeSignatureInfo(llvmArgTypes, llvmRetTy, loc, numFixedArgs);

  auto [sig, usesSRet] = buildFunctionType(sigClass, argValues, origRetTy,
                                           numFixedArgs, isVariadic);
  auto [callArgs, sretPtr] = buildCallArgs(sigClass.args, sigClass.ret,
                                           argValues, origRetTy, loc, rewriter);
  return {sig,          std::move(sigClass.args),
          sigClass.ret, std::move(callArgs),
          sretPtr,      usesSRet};
}

/// Return the number of operands `call` passes to its callee, including the
/// variadic arguments of a call to a variadic callee.
///
/// Note this is *not* `getArgOperands().size()`: llvm/llvm-project#214724,
/// changes `CallOpInterface::getArgOperands` to report only the operands
/// corresponding to the callee's declared parameters (excluding the variadic
/// tail).
static size_t getNumCallArgs(CallOp call) {
  // The callee is operand 0 for indirect calls (no callee symbol attribute).
  size_t numConsumed = call.getCallee().has_value() ? 0 : 1;
  return call.getCalleeOperands().size() - numConsumed;
}

void CABICallHelper::applySRetAttrIfNeeded(CallOp call, Type origRetTy,
                                           bool usesSRet, OpBuilder &builder) {
  if (!usesSRet)
    return;
  size_t numArgAttrs = getNumCallArgs(call);
  SmallVector<Attribute> argAttrs(numArgAttrs, builder.getDictionaryAttr({}));
  argAttrs[0] = builder.getDictionaryAttr({builder.getNamedAttr(
      LLVMDialect::getStructRetAttrName(), TypeAttr::get(origRetTy))});
  call.setArgAttrsAttr(builder.getArrayAttr(argAttrs));
}

void CABICallHelper::applyByvalAttrsToCall(CallOp call,
                                           ArrayRef<CoercionInfo> argClass,
                                           TypeRange origArgTypes,
                                           bool usesSRet, OpBuilder &builder,
                                           size_t startArgIdx) const {
  // Fast path: nothing to do when no argument in range needs indirect passing.
  bool hasIndirectByval = false;
  for (size_t i = startArgIdx; i < argClass.size(); ++i)
    if (argClass[i].useIndirect && argClass[i].useByval) {
      hasIndirectByval = true;
      break;
    }
  if (!hasIndirectByval)
    return;

  // Retrieve or create the per-argument attribute array.
  size_t numArgAttrs = getNumCallArgs(call);
  SmallVector<Attribute> attrs;
  if (auto existing = call.getArgAttrsAttr())
    attrs.assign(existing.begin(), existing.end());
  else
    attrs.assign(numArgAttrs, builder.getDictionaryAttr({}));

  assert(origArgTypes.size() >= argClass.size() &&
         "origArgTypes must cover all argClass entries");

  // Compute the call-arg index where startArgIdx begins: advance past sret
  // and past any fixed args that precede startArgIdx (respecting two-register
  // expansion).
  unsigned paramIdx = usesSRet ? 1 : 0;
  for (size_t i = 0; i < startArgIdx; ++i)
    paramIdx += argClass[i].isTwoRegister() ? 2 : 1;

  for (size_t idx = startArgIdx; idx < argClass.size(); ++idx) {
    const auto &coercion = argClass[idx];
    if (coercion.useIndirect && coercion.useByval) {
      // paramIdx must be in bounds: prepareArg emits exactly one call arg per
      // classification entry, so the sizes must agree.
      assert(paramIdx < numArgAttrs &&
             "paramIdx out of range — classifier/coercion mismatch");
      auto byvalAttr = builder.getNamedAttr(LLVMDialect::getByValAttrName(),
                                            TypeAttr::get(origArgTypes[idx]));
      auto existing = cast<DictionaryAttr>(attrs[paramIdx]);
      SmallVector<NamedAttribute> merged(existing.begin(), existing.end());
      merged.push_back(byvalAttr);
      attrs[paramIdx] = builder.getDictionaryAttr(merged);
    }
    paramIdx += coercion.isTwoRegister() ? 2 : 1;
  }
  call.setArgAttrsAttr(builder.getArrayAttr(attrs));
}
