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

#include "CABIAAPCS.h"
#include "LLVMLoweringUtils.h"
#include "mlir/IR/Diagnostics.h"

using namespace M;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

AAPCSABIInfo::AAPCSABIInfo(mlir::MLIRContext *ctx,
                           const LLVMDataLayout &dataLayout, bool isDarwinOS)
    : CABIInfo(ctx, dataLayout), isDarwin(isDarwinOS) {}

//===----------------------------------------------------------------------===//
// Argument Classification
//===----------------------------------------------------------------------===//

CoercionInfo AAPCSABIInfo::classifyArgumentType(mlir::Type type,
                                                mlir::Location loc,
                                                bool isVariadicArg) const {
  // Arrays should be passed indirectly, as per the C specification
  if (isa<mlir::LLVM::LLVMArrayType>(type))
    return CoercionInfo::indirectArgument(/*useByval=*/false);

  // Only LLVM struct types need C ABI classification
  auto structType = dyn_cast<mlir::LLVM::LLVMStructType>(type);
  if (!structType) {
    // Non-struct (scalar, pointer, etc.): pass through directly, no coercion
    return CoercionInfo{}; // Identity: argClass=NoClass
  }

  // HFA (Homogeneous Float Aggregate) classification:
  //
  // Fixed args: always identity — HFA passes in SIMD registers (V0–V3).
  //
  // Variadic args depend on the OS ABI:
  //   Darwin (macOS/iOS): va_list is flat (GP area only). HFA variadic args
  //     must be coerced to integer so they land in GP registers where
  //     va_arg reads them.
  //   Linux AAPCS64: va_list has separate GP and VR save areas (AAPCS64 spec
  //     IHI0055 §B.4). HFA variadic args remain identity so they land in VR
  //     registers where va_arg for HFA structs reads them.
  //
  // Reference implementation: clang/lib/CodeGen/Targets/AArch64.cpp,
  //   EmitDarwinVAArg() vs EmitAAPCSVAArg().
  if (CabiUtils::isAllFloatStruct(structType) &&
      (!isVariadicArg || !isDarwin)) {
    return CoercionInfo{}; // Identity: defer to standard lowering (SIMD)
  }

  // Get struct size in bytes
  int64_t size = CabiUtils::getStructSize(structType, dataLayout);

  // RULE 1: Structs >16 bytes are passed by pointer
  if (size > 16) {
    return CoercionInfo::indirectArgument(/*useByval=*/false);
  }

  // ARM64 AAPCS: Non-HFA structs are always passed in GPRs (integer
  // registers), regardless of whether fields are int, float, or mixed.
  // HFA structs (all-float, same type, ≤4 fields) use SIMD registers,
  // but those are handled above by the isAllFloatStruct early return.

  // RULE 2: 1-8 byte non-HFA structs → coerce to iN
  if (size <= 8) {
    return CabiUtils::classifySmallIntegerStruct(size, ctx);
  }

  // RULE 3: 9-16 byte non-HFA structs → two-register passing
  return classifyTwoRegisterStruct(size, loc);
}

//===----------------------------------------------------------------------===//
// Return Value Classification
//===----------------------------------------------------------------------===//

CoercionInfo AAPCSABIInfo::classifyReturnType(mlir::Type type,
                                              mlir::Location loc) const {
  // Only LLVM struct types need C ABI classification
  auto structType = dyn_cast<mlir::LLVM::LLVMStructType>(type);
  if (!structType) {
    // Non-struct (scalar, pointer, etc.): return directly, no coercion
    return CoercionInfo{}; // Identity: argClass=NoClass
  }

  // All-float structs require HFA classification (not yet implemented).
  if (CabiUtils::isAllFloatStruct(structType)) {
    return CoercionInfo{}; // Identity: defer to standard lowering
  }

  // Get struct size in bytes
  int64_t size = CabiUtils::getStructSize(structType, dataLayout);

  // RULE: Structs >16 bytes use indirect return (X8 pointer)
  if (size > 16) {
    return CoercionInfo::sretReturn();
  }

  // ARM64 AAPCS: Non-HFA structs always use GPRs for return values.

  // RULE: 1-8 byte non-HFA structs → return in X0 as iN
  if (size <= 8) {
    return CabiUtils::classifySmallIntegerStruct(size, ctx);
  }

  // RULE: 9-16 byte non-HFA structs → two-register return
  return classifyTwoRegisterStruct(size, loc);
}

//===----------------------------------------------------------------------===//
// Phase 2: Two-Register Classification (9-16 byte structs)
//===----------------------------------------------------------------------===//

CoercionInfo AAPCSABIInfo::classifyTwoRegisterStruct(int64_t size,
                                                     mlir::Location loc) const {

  // ARM64 AAPCS: Split struct into two 8-byte chunks
  // First chunk: bytes 0-7 (always 8 bytes)
  // Second chunk: bytes 8-15 (may be less than 8 bytes)

  CoercionInfo info;
  info.argClass = ABIArgClass::IntegerPair; // Both passed as integers on ARM64

  // First register: always i64 (8 bytes)
  info.coercedType = CabiUtils::getIntegerTypeForSize(8, ctx);

  // Second register: size depends on remaining bytes
  int64_t secondSize = size - 8; // Bytes 8 through (size-1)
  info.coercedSecondType = CabiUtils::getIntegerTypeForSize(secondSize, ctx);

  return info;
}

//===----------------------------------------------------------------------===//
// Phase 3: HFA Classification (TODO - future work)
//===----------------------------------------------------------------------===//

// TODO: Implement HFA (Homogeneous Float Aggregate) detection and
// classification HFA rules:
// - All fields must be the same float type (f16, f32, or f64)
// - Maximum 4 fields
// - Passed in V0-V3 (SIMD registers)
// - Example: struct { float x, y, z; } is HFA with 3 fields
