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

#ifndef KGEN_LIB_KGENTOLLVM_CABILOWERING_H
#define KGEN_LIB_KGENTOLLVM_CABILOWERING_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace llvm {
class Triple;
} // namespace llvm

namespace mlir {
class MLIRContext;
class Type;
class Location;
namespace LLVM {
class LLVMStructType;
} // namespace LLVM
} // namespace mlir

namespace M::KGEN {

// Forward declarations
class LLVMDataLayout;
class StructType;

//===----------------------------------------------------------------------===//
// ABI Classification Types
//===----------------------------------------------------------------------===//

/// Classification of how a struct should be passed according to C ABI rules.
enum class ABIArgClass {
  // Direct passing (single register)
  Integer, // Pass as integer type (i8, i16, i32, i64)
  SSE,     // Pass as float/vector type (f32, f64, <2 x float>, etc.)

  // Direct passing (two registers)
  IntegerPair, // Pass as two integer values
  SSEPair,     // Pass as two SSE values
  Mixed,       // Pass as one integer + one SSE value

  // Indirect passing
  Memory, // Pass by pointer (struct exceeds register limits)

  // Special cases
  NoClass // Uninitialized or empty
};

/// Information about how to coerce a struct type for C ABI compliance.
///
/// This structure captures all the information needed to transform a
/// struct argument or return value into the appropriate ABI-compliant form.
struct CoercionInfo {
  /// The classification of this argument
  ABIArgClass argClass = ABIArgClass::NoClass;

  /// Type coercion for direct passing
  /// - Single-register: coercedType is the only type (e.g., i32, f64)
  /// - Two-register: coercedType + coercedSecondType (e.g., i64 + i64)
  // First/only register type
  mlir::Type coercedType = nullptr;
  // Second register type (two-register only)
  mlir::Type coercedSecondType = nullptr;

  /// For indirect passing. If true, pass pointer to struct
  bool useIndirect = false;
  // In conjunction with useIndirect, set 'byval' on that pointer
  bool useByval = false;

  /// For return values >16 bytes (sret convention)
  bool useSRet = false; // If true, use hidden pointer parameter

  /// Default constructor creates NoClass (identity/pass-through)
  CoercionInfo() = default;

  /// Factory for indirect argument passing.
  /// Large structs (>16 bytes) are passed by pointer on the stack.
  /// The pointer itself is allocated and passed by the caller.
  static CoercionInfo indirectArgument(bool useByval) {
    CoercionInfo info;
    info.argClass = ABIArgClass::Memory;
    info.useIndirect = true; // Argument passed via pointer
    info.useByval = useByval;
    info.useSRet = false;
    return info;
  }

  /// Factory for structured return (sret).
  /// Large return values (>16 bytes) use the sret calling convention:
  /// caller allocates space and passes a pointer in a register (X8/RDI),
  /// callee writes the result there.
  static CoercionInfo sretReturn() {
    CoercionInfo info;
    info.argClass = ABIArgClass::Memory;
    info.useIndirect = false; // Pointer passed directly in register
    info.useSRet = true;
    return info;
  }

  /// Check if this is a pass-through (no coercion needed)
  bool isIdentity() const { return argClass == ABIArgClass::NoClass; }

  /// Check if this requires two registers
  bool isTwoRegister() const {
    return argClass == ABIArgClass::IntegerPair ||
           argClass == ABIArgClass::SSEPair || argClass == ABIArgClass::Mixed;
  }
};

/// ABI classification for a whole call signature: one CoercionInfo per
/// argument plus the return value.
struct SignatureClassification {
  llvm::SmallVector<CoercionInfo> args;
  CoercionInfo ret;
};

//===----------------------------------------------------------------------===//
// Platform ABI Interface
//===----------------------------------------------------------------------===//

/// Abstract interface for platform-specific C ABI implementations.
///
/// Concrete implementations exist for:
/// - SystemVABIInfo (x86-64 System V AMD64 ABI)
/// - AAPCSABIInfo (ARM64 AAPCS)
/// - DefaultCABIInfo (pass-through for other architectures)
/// - Future: Win64ABIInfo (Windows x64 calling convention)
class CABIInfo {
public:
  virtual ~CABIInfo() = default;

  /// Classify a whole signature at once. This is the entry point for callers.
  /// Types must be LLVM dialect types (convert POP types first). Args at index
  /// >= numFixedArgs are variadic (SIZE_MAX for non-variadic).
  ///
  /// The base implementation classifies each argument and the return value
  /// independently. Targets whose register assignment depends on argument
  /// order (e.g. System V rollback-to-stack) override this to thread a
  /// register budget across the signature.
  virtual SignatureClassification
  computeSignatureInfo(mlir::TypeRange argTypes, mlir::Type retType,
                       mlir::Location loc,
                       size_t numFixedArgs = SIZE_MAX) const;

  const LLVMDataLayout &getDataLayout() const { return dataLayout; }

protected:
  CABIInfo(mlir::MLIRContext *ctx, const LLVMDataLayout &dataLayout)
      : ctx(ctx), dataLayout(dataLayout) {}

  /// Classify a single argument type. Internal building block for
  /// computeSignatureInfo; not safe to call on its own because it cannot apply
  /// ordering-dependent rules such as rollback-to-stack. Type must be an LLVM
  /// dialect type. isVariadicArg marks a variadic (...) argument.
  virtual CoercionInfo classifyArgumentType(mlir::Type type, mlir::Location loc,
                                            bool isVariadicArg) const = 0;

  /// Classify a return type. Internal building block for computeSignatureInfo.
  /// Type must be an LLVM dialect type.
  virtual CoercionInfo classifyReturnType(mlir::Type type,
                                          mlir::Location loc) const = 0;

  mlir::MLIRContext *ctx;
  const LLVMDataLayout &dataLayout;
};

//===----------------------------------------------------------------------===//
// Default (Pass-through) ABI Implementation
//===----------------------------------------------------------------------===//

/// Default ABI implementation that does no coercion (pass-through).
///
/// This is used for architectures where we don't have platform-specific
/// ABI knowledge, or where the default LLVM lowering is sufficient.
/// All types are passed and returned as-is with no ABI adjustments.
class DefaultCABIInfo : public CABIInfo {
public:
  DefaultCABIInfo(mlir::MLIRContext *ctx, const LLVMDataLayout &dataLayout)
      : CABIInfo(ctx, dataLayout) {}

protected:
  CoercionInfo classifyArgumentType(mlir::Type type, mlir::Location loc,
                                    bool isVariadicArg) const override {
    // Pass through all types unchanged
    return CoercionInfo{}; // Identity: argClass=NoClass
  }

  CoercionInfo classifyReturnType(mlir::Type type,
                                  mlir::Location loc) const override {
    // Return all types unchanged
    return CoercionInfo{}; // Identity: argClass=NoClass
  }
};

//===----------------------------------------------------------------------===//
// Factory Function
//===----------------------------------------------------------------------===//

/// Create a platform-specific CABIInfo implementation.
///
/// \param triple Target architecture triple
/// \param ctx MLIR context
/// \param dataLayout Data layout for size/alignment queries
/// \return Platform-specific ABI handler, or nullptr if unsupported
std::unique_ptr<CABIInfo> createCABIInfo(const llvm::Triple &triple,
                                         mlir::MLIRContext *ctx,
                                         const LLVMDataLayout &dataLayout);

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

namespace CabiUtils {

/// Get the size of an LLVM struct type in bytes.
/// Uses the LLVM type's actual layout including padding.
int64_t getStructSize(mlir::LLVM::LLVMStructType type,
                      const LLVMDataLayout &dataLayout);

/// Check if an LLVM struct contains only float/double types.
/// Used for ARM64 HFA (Homogeneous Float Aggregate) detection.
bool isAllFloatStruct(mlir::LLVM::LLVMStructType type);

/// Get the integer type for a given byte size: 1→i8, 2→i16, ≤4→i32, ≤8→i64.
mlir::IntegerType getIntegerTypeForSize(int64_t size, mlir::MLIRContext *ctx);

/// Classify a 1-8 byte struct as an integer type for register passing.
CoercionInfo classifySmallIntegerStruct(int64_t size, mlir::MLIRContext *ctx);

/// Check if C ABI functionality is enabled.
/// C ABI coercion is enabled by default.
/// Use --skip-c-abi-coercion flag to disable (for debugging).
bool isCABIEnabled();

} // namespace CabiUtils

} // namespace M::KGEN

#endif // KGEN_LIB_KGENTOLLVM_CABILOWERING_H
