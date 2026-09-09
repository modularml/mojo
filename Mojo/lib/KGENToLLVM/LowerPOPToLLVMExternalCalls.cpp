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

#include "LowerPOPToLLVMExternalCalls.h"
#include "CABICallHelpers.h"
#include "CABILowering.h"
#include "LLVMLoweringUtils.h"
#include "Mojo/POPDialect/POPOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace M;
using namespace M::KGEN;
using namespace KGEN;
using namespace POP;
namespace LLVM = mlir::LLVM;

namespace {

//===----------------------------------------------------------------------===//
// ConvertPOPExternalCall
//===----------------------------------------------------------------------===//

/// Lower an external call. Add the callee to the symbol table.
struct ConvertPOPExternalCall : public ConvertSymbolOpToLLVM<ExternalCallOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

private:
  //===--------------------------------------------------------------------===//
  // Helpers for matchAndRewrite
  //===--------------------------------------------------------------------===//

  /// Compute the call argument index where variadic args start.
  /// Accounts for sret pointer and two-register expansion of fixed args.
  unsigned
  computeVariadicCallArgStart(ArrayRef<CoercionInfo> argClassifications,
                              size_t numFixedArgs, bool usesSRet) const {
    unsigned start = usesSRet ? 1 : 0;
    for (size_t i = 0; i < numFixedArgs; ++i) {
      if (argClassifications[i].isTwoRegister())
        start += 2;
      else
        start += 1;
    }
    return start;
  }

  /// Overlay byval attributes for indirect args onto the given per-parameter
  /// attribute lists. On x86-64 System V, MEMORY class structs are passed by
  /// pointer with the byval attribute so the callee knows to copy from stack.
  void overlayByvalAttrs(MutableArrayRef<mlir::NamedAttrList> paramAttrs,
                         ArrayRef<CoercionInfo> argClassifications,
                         ExternalCallOp op, size_t numFixedArgs, bool usesSRet,
                         bool isVariadic) const {
    size_t numParams = isVariadic ? numFixedArgs : argClassifications.size();
    unsigned paramIdx = usesSRet ? 1 : 0;
    for (size_t idx = 0; idx < numParams; ++idx) {
      const auto &coercion = argClassifications[idx];
      if (coercion.useIndirect && coercion.useByval) {
        Type llvmArgType =
            getTypeConverter()->convertType(op.getOperandTypes()[idx]);
        paramAttrs[paramIdx].set(LLVM::LLVMDialect::getByValAttrName(),
                                 mlir::TypeAttr::get(llvmArgType));
      }
      if (coercion.isTwoRegister())
        paramIdx += 2;
      else
        paramIdx += 1;
    }
  }

  /// On ARM64, bitcast variadic float args < 64 bits to integer to prevent
  /// LLVM's float→double promotion. This ensures raw bits are placed in GPRs
  /// for va_arg struct reads. On x86-64, floats go in XMM registers and
  /// must NOT be bitcast.
  void applyARM64VariadicFloatBitcast(
      SmallVectorImpl<Value> &callArgs,
      ArrayRef<CoercionInfo> argClassifications, size_t numFixedArgs,
      bool usesSRet, Location loc, ConversionPatternRewriter &rewriter) const {
    unsigned varStart =
        computeVariadicCallArgStart(argClassifications, numFixedArgs, usesSRet);
    for (unsigned i = varStart; i < callArgs.size(); ++i) {
      if (auto floatTy = dyn_cast<FloatType>(callArgs[i].getType())) {
        unsigned bitWidth = floatTy.getWidth();
        if (bitWidth < 64) {
          Type intType = IntegerType::get(getContext(), bitWidth);
          callArgs[i] =
              LLVM::BitcastOp::create(rewriter, loc, intType, callArgs[i]);
        }
      }
    }
  }

  /// Seed per-parameter attribute lists with the op's POP-level argAttrs.
  /// ABI coercion can change the parameter list: sret prepends a hidden
  /// pointer, two-register args expand to two parameters, etc. The original
  /// argAttrs from the POP op use POP-level indexing, so we seed the
  /// LLVM-indexed list with user-provided attrs at the correct positions.
  /// Only positions within [0, paramAttrs.size()) are filled, so variadic
  /// callers can pass a list sized for fixed params only.
  void seedUserArgAttrs(MutableArrayRef<mlir::NamedAttrList> paramAttrs,
                        mlir::ArrayAttr argAttrs,
                        ArrayRef<CoercionInfo> argClassifications,
                        bool usesSRet) const {
    if (!argAttrs)
      return;
    unsigned paramIdx = usesSRet ? 1 : 0;
    for (size_t popIdx = 0; popIdx < argClassifications.size(); ++popIdx) {
      if (paramIdx >= paramAttrs.size())
        break;
      if (popIdx < argAttrs.size()) {
        if (auto dict = dyn_cast<mlir::DictionaryAttr>(argAttrs[popIdx])) {
          for (auto namedAttr : dict)
            paramAttrs[paramIdx].set(namedAttr.getName(), namedAttr.getValue());
        }
      }
      paramIdx += argClassifications[popIdx].isTwoRegister() ? 2 : 1;
    }
  }

  /// Build the final LLVM-level per-parameter argAttrs that we expect on the
  /// declaration: user-provided attrs remapped to LLVM indices, plus the sret
  /// and byval overlays applied by the lowering. Returns null if no attrs
  /// would be set. For variadic functions, the returned array is sized to
  /// match the LLVM function's fixed parameter count only — variadic args
  /// are not part of the function type and their attributes go on the call.
  mlir::ArrayAttr buildLLVMArgAttrs(mlir::ArrayAttr argAttrs,
                                    ArrayRef<CoercionInfo> argClassifications,
                                    ExternalCallOp op, size_t numFixedArgs,
                                    bool usesSRet, bool isVariadic,
                                    ConversionPatternRewriter &rewriter) const {
    size_t numArgsForSig =
        isVariadic ? numFixedArgs : argClassifications.size();
    unsigned numLLVMParams = usesSRet ? 1 : 0;
    for (size_t i = 0; i < numArgsForSig; ++i)
      numLLVMParams += argClassifications[i].isTwoRegister() ? 2 : 1;

    SmallVector<mlir::NamedAttrList> paramAttrs(numLLVMParams);
    seedUserArgAttrs(paramAttrs, argAttrs, argClassifications, usesSRet);

    if (usesSRet) {
      Type llvmRetType =
          getTypeConverter()->convertType(op.getResult().getType());
      paramAttrs[0].set(LLVM::LLVMDialect::getStructRetAttrName(),
                        mlir::TypeAttr::get(llvmRetType));
    }

    overlayByvalAttrs(paramAttrs, argClassifications, op, numFixedArgs,
                      usesSRet, isVariadic);

    bool anySet = llvm::any_of(paramAttrs, [](const mlir::NamedAttrList &nal) {
      return !nal.empty();
    });
    if (!anySet)
      return nullptr;

    SmallVector<Attribute> dicts;
    dicts.reserve(numLLVMParams);
    for (auto &nal : paramAttrs)
      dicts.push_back(nal.getDictionary(rewriter.getContext()));
    return rewriter.getArrayAttr(dicts);
  }

public:
  /// Lower a POP external_call to an LLVM call, applying C ABI struct
  /// coercion when needed.
  ///
  /// **Why C ABI coercion is necessary:**
  ///
  /// When Mojo calls a C function that takes or returns a struct by value,
  /// the struct cannot simply be passed as-is in LLVM IR. The platform's
  /// C calling convention dictates *how* the struct's bytes are delivered
  /// to the callee — in integer registers, floating-point registers, on
  /// the stack, or via a hidden pointer — depending on the struct's size,
  /// field types, and target architecture.
  ///
  /// For example, a C function `struct Pair add(struct Pair p)` where
  /// `Pair` is `{int a, int b}` (8 bytes) expects its argument in a
  /// single 64-bit integer register on both x86-64 and ARM64. Without
  /// coercion, this lowers to an LLVM IR parameter of type `{i32, i32}`,
  /// which LLVM's backend
  /// decomposes into two separate i32 values in two registers (%edi and
  /// %esi on x86-64). The callee, compiled by Clang with ABI coercion,
  /// expects both fields packed into %rdi — so it reads garbage for the
  /// second field. C ABI coercion is the frontend's responsibility;
  /// LLVM's backend does not re-derive it for aggregate types.
  ///
  /// The rules differ by platform:
  /// - **x86-64 System V**: classifies each 8-byte "eightbyte" of the
  ///   struct as INTEGER or SSE based on field types; structs >16 bytes
  ///   are passed by pointer.
  /// - **ARM64 AAPCS**: non-HFA structs always use integer registers;
  ///   HFA (Homogeneous Float Aggregate) structs use SIMD registers;
  ///   structs >16 bytes are passed by pointer.
  /// - Other targets (e.g., Win64, 32-bit x86) use a DefaultCABIInfo
  ///   pass-through and will need dedicated implementations when supported.
  ///
  /// **How this function works (outline):**
  ///
  /// 1. **Classify** each argument and the return value using a
  ///    platform-specific ABI handler (SystemVABIInfo or AAPCSABIInfo).
  ///    Each type gets a CoercionInfo describing how it must be passed:
  ///    identity (no change), coerce to integer/float, split across two
  ///    registers, or pass indirectly via pointer.
  ///
  /// 2. **Build the LLVM function signature** from the classifications.
  ///    Coerced args become their target types (e.g., kgen.struct → i64);
  ///    two-register args expand to two parameters; indirect args become
  ///    pointers; sret returns prepend a hidden pointer parameter.
  ///
  /// 3. **Prepare call arguments** by storing each struct to the stack
  ///    and reloading it as the coerced type (a store/load "bitcast").
  ///    Identity args pass through unchanged.
  ///
  /// 4. **Reverse-coerce the return value**: store the coerced result
  ///    back to the stack and reload it as the original struct type.
  ///
  /// The classification pipeline is unified: even when all types are
  /// identity (no coercion needed), the same code path runs — identity
  /// classifications simply produce pass-through behavior identical to
  /// standard LLVM type conversion.
  LogicalResult
  matchAndRewrite(ExternalCallOp op, ExternalCallOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    const llvm::Triple &triple = getTypeConverter()->getTarget().getTriple();
    CABICallHelper cabi(getTypeConverter(), getContext(), op.getOperation());

    // Determine number of fixed arguments (for variadic functions).
    size_t numFixedArgs = adaptor.getOperands().size();
    std::optional<mlir::TypedAttr> declaredFixedArgs = op.getNumFixedArgs();
    bool isVariadic = declaredFixedArgs.has_value();
    if (isVariadic) {
      // Elaboration has folded every parameter expression by now, and the op
      // verifier has already rejected a negative count.
      int64_t declared = cast<IntegerAttr>(*declaredFixedArgs).getInt();
      assert(declared >= 0 && "'numFixedArgs' must be a non-negative constant");
      // The operand bound is not an invariant: LowerArgConventions expands
      // argument packs into individual operands after the verifier has run, so
      // the verifier cannot bound the count. A count above the post-expansion
      // operand count is reachable from Mojo source and would index out of
      // bounds during argument classification, so it stays a diagnostic.
      if (static_cast<size_t>(declared) > adaptor.getOperands().size())
        return mlir::emitError(loc,
                               "'numFixedArgs' must not exceed the number of "
                               "call operands: expected at most ")
               << adaptor.getOperands().size() << ", found " << declared;
      numFixedArgs = declared;
    }

    // Classify arguments/return and build the coerced LLVM signature + args.
    // op.getOperandTypes() holds POP-level types for classification; the
    // adaptor operands hold the already-converted LLVM values for coercion.
    Type llvmReturnType;
    if (!op.getResults().empty())
      llvmReturnType =
          getTypeConverter()->convertType(op.getResult().getType());
    auto prep = cabi.prepareCall(op.getOperandTypes(), adaptor.getOperands(),
                                 llvmReturnType, loc, rewriter, numFixedArgs,
                                 isVariadic);
    auto &argClassifications = prep.argClass;
    const CoercionInfo &returnClassification = prep.retClass;
    const bool usesSRet = prep.usesSRet;
    const LLVM::LLVMFunctionType &signature = prep.signature;

    // Step 4: Compute the final LLVM-level attributes we expect on the
    // declaration. These include target-aware passthrough, arg attrs remapped
    // to LLVM parameter indices with sret and byval overlays applied, and
    // (when there is an LLVM-level return value) the op's resAttrs. Computing
    // these once lets us both compare against an existing declaration and
    // initialize a new one from the same source of truth.
    mlir::ArrayAttr passthrough = attachTargetPassthroughAttrs(
        rewriter, getTypeConverter()->getTarget(), op.getFuncAttrsAttr());
    mlir::ArrayAttr argAttrs = op.getArgAttrsAttr();
    mlir::DictionaryAttr resAttrs = op.getResAttrsAttr();
    auto memory = dyn_cast_or_null<LLVM::MemoryEffectsAttr>(op.getMemoryAttr());

    mlir::ArrayAttr expectedArgAttrs =
        buildLLVMArgAttrs(argAttrs, argClassifications, op, numFixedArgs,
                          usesSRet, isVariadic, rewriter);
    // resAttrs are dropped when sret is active (no LLVM-level return value).
    mlir::ArrayAttr expectedResAttrs;
    if (resAttrs && !usesSRet)
      expectedResAttrs = rewriter.getArrayAttr(resAttrs);

    // Step 5: Lookup existing function (unified path - no early return!)
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(op.getCallee().getValue());

    if (func && func.getFunctionType() != signature) {
      return mlir::emitError(loc,
                             "existing function with conflicting signature")
                 .attachNote(func.getLoc())
             << "see function declaration here";
    }
    if (func &&
        std::make_tuple(func.getPassthroughAttr(), func.getArgAttrsAttr(),
                        func.getResAttrsAttr(), func.getMemoryEffectsAttr()) !=
            std::make_tuple(passthrough, expectedArgAttrs, expectedResAttrs,
                            memory)) {
      return mlir::emitError(loc,
                             "existing function with conflicting attributes")
                 .attachNote(func.getLoc())
             << "see function declaration here";
    }

    // Step 6: Create function if needed (only branch on creation)
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.clearInsertionPoint();
      func = LLVM::LLVMFuncOp::create(rewriter,
                                      mlir::UnknownLoc::get(getContext()),
                                      op.getCallee(), signature);
      func.setPassthroughAttr(passthrough);
      if (expectedArgAttrs)
        func.setArgAttrsAttr(expectedArgAttrs);
      if (expectedResAttrs)
        func.setResAttrsAttr(expectedResAttrs);
      if (memory)
        func.setMemoryEffectsAttr(memory);
      symtab.insert(func);
    }

    // Darwin ARM64: bitcast variadic floats < 64 bits to integer to prevent
    // LLVM's float→double promotion. Darwin's flat va_list reads all variadic
    // args from the GP save area, so float values must be in GPRs (as
    // integers). Linux AAPCS64 has a separate VR save area; floats stay as
    // floats and land in SIMD registers where va_arg for HFA structs reads
    // them.
    if (isVariadic && triple.isAArch64() && triple.isOSDarwin()) {
      applyARM64VariadicFloatBitcast(prep.callArgs, argClassifications,
                                     numFixedArgs, usesSRet, loc, rewriter);
    }

    // Step 7: Create call
    LLVM::CallOp call = createLLVMCall(rewriter, loc, func, prep.callArgs);

    // Mirror byval onto the call args. A direct call lowers from the call-site
    // attributes, so the declaration attribute alone does not make the caller
    // emit the by-memory copy. This covers both fixed and variadic args.
    SmallVector<Type> llvmArgTypes;
    for (Type t : op.getOperandTypes())
      llvmArgTypes.push_back(getTypeConverter()->convertType(t));
    cabi.applyByvalAttrsToCall(call, argClassifications, llvmArgTypes, usesSRet,
                               rewriter, /*startArgIdx=*/0);

    // Step 8: Handle return value
    if (op.getResults().empty()) {
      // Void return
      rewriter.eraseOp(op);
    } else if (!returnClassification.isIdentity()) {
      // ABI coercion: reverse coercion on the return value.
      Value callResult = call.getNumResults() > 0 ? call.getResult() : Value();
      Value result =
          cabi.extractReturn(returnClassification, callResult, prep.sretPtr,
                             llvmReturnType, loc, rewriter);
      rewriter.replaceOp(op, result);
    } else {
      // Identity return: use standard replacement
      replaceCallWithLLVMCall(rewriter, op, call);
    }

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void M::KGEN::populateLowerPOPExternalCallPatterns(
    mlir::RewritePatternSet &patterns, POPToLLVMTypeConverter &typeConverter,
    mlir::SymbolTable &symtab) {
  patterns.insert<ConvertPOPExternalCall>(typeConverter, symtab);
}
