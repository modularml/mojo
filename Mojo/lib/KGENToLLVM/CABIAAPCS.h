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

#ifndef KGEN_LIB_KGENTOLLVM_CABIAAPCS_H
#define KGEN_LIB_KGENTOLLVM_CABIAAPCS_H

#include "CABILowering.h"

namespace M::KGEN {

/// Implementation of ARM64 AAPCS calling convention.
///
/// This class handles classification and coercion of struct types according
/// to the AAPCS rules, which are the standard on ARM64 (Linux, macOS, iOS).
///
/// Key rules:
/// - Structs ≤8 bytes (all-integer): coerce to i8/i16/i32/i64
/// - Structs 9-16 bytes: split into two registers (X0-X1)
/// - Structs >16 bytes: pass by pointer (indirect)
/// - HFA (Homogeneous Float Aggregate): up to 4 float/vector fields in V0-V3
/// - **Darwin vs Linux difference for variadic HFA:**
///   Darwin (macOS/iOS): va_list is flat (GP/stack only); HFA variadic args
///   must be coerced to integer so va_arg reads from GP save area.
///   Linux AAPCS64: va_list has separate GP and VR save areas; HFA variadic
///   args remain identity so va_arg reads correctly from VR save area.
// TODO: A proposed LLVM-native ABI lowering library (LLVMABI) would replace
// this class. See:
// https://discourse.llvm.org/t/rfc-an-abi-lowering-library-for-llvm/84495
class AAPCSABIInfo : public CABIInfo {
public:
  AAPCSABIInfo(mlir::MLIRContext *ctx, const LLVMDataLayout &dataLayout,
               bool isDarwinOS);

protected:
  CoercionInfo classifyArgumentType(mlir::Type type, mlir::Location loc,
                                    bool isVariadicArg) const override;

  CoercionInfo classifyReturnType(mlir::Type type,
                                  mlir::Location loc) const override;

private:
  //===--------------------------------------------------------------------===//
  // Phase 2: Two-register classification (9-16 byte structs)
  //===--------------------------------------------------------------------===//

  /// Classify a 9-16 byte struct into two registers (Phase 2 implementation).
  ///
  /// ARM64 AAPCS is simpler than x86-64: just split the struct into two
  /// 8-byte chunks, each passed as an integer register (X0-X7).
  ///
  /// \param size Total size in bytes (must be 9-16)
  /// \param loc Source location for diagnostics
  /// \return Coercion with coercedType and coercedSecondType
  CoercionInfo classifyTwoRegisterStruct(int64_t size,
                                         mlir::Location loc) const;

  bool isDarwin; // true = Darwin (macOS/iOS), false = Linux/other

  //===--------------------------------------------------------------------===//
  // Phase 3: HFA classification (TODO - future work)
  //===--------------------------------------------------------------------===//

  // HFA (Homogeneous Float Aggregate) detection
  // - All fields are float32, float64, or float16
  // - Up to 4 fields total
  // - Passed in V0-V3 (SIMD registers)
  //
  // Phase 3 methods (not yet implemented):
  // - bool isHFA(StructType type, mlir::Type &baseType, int &count) const;
  // - CoercionInfo classifyHFA(...);
};

} // namespace M::KGEN

#endif // KGEN_LIB_KGENTOLLVM_CABIAAPCS_H
