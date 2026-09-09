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
// Shared C ABI call-lowering helpers used by ConvertPOPExternalCall
// (LowerPOPToLLVMExternalCalls.cpp) and CallIndirectOpConversion /
// ConvertKGENCall (LowerKGENToLLVM.cpp) to apply C ABI struct coercion.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_KGENTOLLVM_CABICALLHELPERS_H
#define KGEN_LIB_KGENTOLLVM_CABICALLHELPERS_H

#include "CABILowering.h"
#include "LLVMLoweringUtils.h"
#include "Mojo/KGENDialect/KGENEnums.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace M::KGEN {

/// Apply the tail-call kind from a kgen.call / kgen.call_indirect to the
/// already-created llvm.call.  Extracted here so both ConvertKGENCall and
/// CallIndirectOpConversion share a single copy.
inline void applyTailKind(mlir::LLVM::CallOp call, TailKind kind) {
  switch (kind) {
  case TailKind::MustTail:
    call.setTailCallKind(mlir::LLVM::tailcallkind::TailCallKind::MustTail);
    break;
  case TailKind::NoTail:
    call.setTailCallKind(mlir::LLVM::tailcallkind::TailCallKind::NoTail);
    break;
  case TailKind::Tail:
    call.setTailCallKind(mlir::LLVM::tailcallkind::TailCallKind::Tail);
    break;
  case TailKind::None:
    break;
  }
}

/// Result of CABICallHelper::prepareCall().  Bundles everything computed
/// during the classify/build phase that callers need to emit the actual call.
struct CABICallPrep {
  mlir::LLVM::LLVMFunctionType signature;
  /// Per-argument ABI classifications (needed by callers that set byval attrs).
  mlir::SmallVector<CoercionInfo> argClass;
  CoercionInfo retClass;
  mlir::SmallVector<mlir::Value> callArgs;
  mlir::Value sretPtr;
  bool usesSRet;
};

//===----------------------------------------------------------------------===//
// CABICallHelper
//===----------------------------------------------------------------------===//

/// Helper struct that holds the C ABI coercion methods shared across call
/// lowering patterns. Constructed with a type converter, context, and parent
/// operation (used to locate the enclosing LLVM function for alloca placement).
struct CABICallHelper {
  const POPToLLVMTypeConverter *tc;
  mlir::MLIRContext *ctx;
  mlir::Operation *parentOp;

  CABICallHelper(const POPToLLVMTypeConverter *tc, mlir::MLIRContext *ctx,
                 mlir::Operation *parentOp)
      : tc(tc), ctx(ctx), parentOp(parentOp) {}

  /// Create a platform-specific C ABI handler for argument/return
  /// classification. Returns a DefaultCABIInfo (pass-through) when C ABI
  /// is disabled or the platform is unsupported.
  std::unique_ptr<CABIInfo> createABIHandler() const;

  /// Build LLVM function type with C ABI type coercion applied.
  /// Returns the function type and whether sret is used.
  /// For variadic functions, only fixed parameters are included in the
  /// function type.
  std::pair<mlir::LLVM::LLVMFunctionType, bool>
  buildFunctionType(const SignatureClassification &sigClass,
                    mlir::ValueRange originalArgs, mlir::Type origRetTy,
                    size_t numFixedArgs, bool isVariadic) const;

  /// Create an alloca in the entry block of the enclosing function.
  /// This ensures stack allocations don't grow unboundedly when the
  /// external_call is inside a loop.
  mlir::Value createEntryBlockAlloca(mlir::ConversionPatternRewriter &rewriter,
                                     mlir::Location loc,
                                     mlir::Type elemType) const;

  /// Bitcast a value by storing to stack and loading as a different type.
  /// This is the standard LLVM pattern for struct<->scalar bitcasts.
  ///
  /// The allocation is sized to `allocType` which should be the larger of
  /// sourceValue's type and destType to prevent undefined behavior when
  /// coercion rounds up (e.g., 3-byte struct -> i32 = 4 bytes).
  mlir::Value bitcastViaMemory(mlir::Value src, mlir::Type destType,
                               mlir::Type allocType, mlir::Location loc,
                               mlir::ConversionPatternRewriter &rewriter) const;

  /// Create a GEP to access memory at a byte offset from a pointer.
  /// Used for accessing the second register in two-register struct coercion.
  mlir::Value createOffsetGEP(mlir::Value basePtr, int64_t byteOffset,
                              mlir::Location loc,
                              mlir::ConversionPatternRewriter &rewriter) const;

  /// Prepare a two-register argument for C ABI calling convention.
  /// Allocates a struct of both types, stores the original value, then loads
  /// both registers at the correct offsets.
  llvm::SmallVector<mlir::Value>
  prepareTwoRegisterArgument(mlir::Value orig, mlir::Type firstTy,
                             mlir::Type secondTy, mlir::Location loc,
                             mlir::ConversionPatternRewriter &rewriter) const;

  /// Handle a two-register return value from C ABI call.
  /// Extracts both values from the call result, stores them at the correct
  /// offsets, then loads as the original struct type.
  mlir::Value
  handleTwoRegisterReturn(mlir::Value callResult, mlir::Type firstTy,
                          mlir::Type secondTy, mlir::Type origRetTy,
                          mlir::Location loc,
                          mlir::ConversionPatternRewriter &rewriter) const;

  /// Coerce a single argument and return the LLVM value(s) to pass.
  llvm::SmallVector<mlir::Value>
  prepareArg(const CoercionInfo &coercion, mlir::Value orig, mlir::Location loc,
             mlir::ConversionPatternRewriter &rewriter) const;

  /// Build the actual call arguments with C ABI coercion applied.
  /// Handles sret pointer preparation if needed.
  /// Returns {callArgs, sretPointer}.
  std::pair<llvm::SmallVector<mlir::Value>, mlir::Value>
  buildCallArgs(llvm::ArrayRef<CoercionInfo> argClass,
                const CoercionInfo &retClass, mlir::ValueRange originalArgs,
                mlir::Type origRetTy, mlir::Location loc,
                mlir::ConversionPatternRewriter &rewriter) const;

  /// Handle the return value from C ABI call.
  /// Applies reverse coercion (bitcast from integer back to struct).
  /// For sret, loads from the sret pointer.
  mlir::Value extractReturn(const CoercionInfo &retClass,
                            mlir::Value callResult, mlir::Value sretPtr,
                            mlir::Type origRetTy, mlir::Location loc,
                            mlir::ConversionPatternRewriter &rewriter) const;

  /// Classify arguments and return type, build the coerced LLVM function
  /// signature, and prepare coerced call arguments in one step.
  ///
  /// `argTypes` drives ABI classification; `argValues` supplies the SSA values
  /// for coercion (they may differ when the op holds POP types but the adaptor
  /// holds already-converted LLVM values, as in ConvertPOPExternalCall).
  /// Pass `numFixedArgs` and `isVariadic=true` for variadic callees.
  /// The returned CABICallPrep contains everything needed to emit the call.
  CABICallPrep prepareCall(mlir::TypeRange argTypes, mlir::ValueRange argValues,
                           mlir::Type origRetTy, mlir::Location loc,
                           mlir::ConversionPatternRewriter &rewriter,
                           size_t numFixedArgs = SIZE_MAX,
                           bool isVariadic = false) const;

  /// Set the llvm.sret attribute on arg_attrs[0] of `call` when usesSRet is
  /// true.  The attribute array is sized from the operands actually passed to
  /// the callee, so it is correct for direct calls (callee is a symbol, not in
  /// operands), indirect calls (callee is operands[0], consumed by the op),
  /// and variadic calls (whose variadic tail arg_attrs must still be
  /// addressable).
  static void applySRetAttrIfNeeded(mlir::LLVM::CallOp call,
                                    mlir::Type origRetTy, bool usesSRet,
                                    mlir::OpBuilder &builder);

  /// Apply byval attributes to indirect (MEMORY-class) arguments on x86-64.
  /// On ARM64 and other platforms this is a no-op.
  ///
  /// Must be called *after* applySRetAttrIfNeeded so that sret is already
  /// recorded in arg_attrs before we merge byval entries on top.
  ///
  /// `origArgTypes[i]` must be the LLVM pointee type for argument i
  /// (i.e. the type before coercion, as stored in the alloca created by
  /// prepareArg).  This is the type that LLVM will use for the memcpy it
  /// emits to satisfy the x86-64 SysV by-memory passing convention.
  ///
  /// `startArgIdx` selects which args to process: pass 0 (default) to
  /// handle all fixed args, or `numFixedArgs` to handle only variadic args
  /// (the pattern used by ConvertPOPExternalCall, where fixed args already
  /// have byval on the function declaration via addByvalAttrsToFunc).
  void applyByvalAttrsToCall(mlir::LLVM::CallOp call,
                             llvm::ArrayRef<CoercionInfo> argClass,
                             mlir::TypeRange origArgTypes, bool usesSRet,
                             mlir::OpBuilder &builder,
                             size_t startArgIdx = 0) const;
};

} // namespace M::KGEN

#endif // KGEN_LIB_KGENTOLLVM_CABICALLHELPERS_H
