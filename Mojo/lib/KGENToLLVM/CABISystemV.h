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

#ifndef KGEN_LIB_KGENTOLLVM_CABISYSTEMV_H
#define KGEN_LIB_KGENTOLLVM_CABISYSTEMV_H

#include "CABILowering.h"

namespace M::KGEN {

/// Implementation of x86-64 System V AMD64 ABI calling convention.
///
/// This class handles classification and coercion of struct types according
/// to the System V ABI rules, which are the standard on Linux, BSD, and macOS.
///
/// Key rules:
/// - Structs ≤8 bytes (all-integer): coerce to i8/i16/i32/i64
/// - Structs 9-16 bytes: split into two eightbytes, classify each
/// - Structs >16 bytes: pass by pointer (MEMORY class)
/// - Float/SSE handling: classify per eightbyte
/// - Variadic arguments: on x86-64, variadic args always use stack
class SystemVABIInfo : public CABIInfo {
public:
  SystemVABIInfo(mlir::MLIRContext *ctx, const LLVMDataLayout &dataLayout);

  /// Classify a whole signature, applying the rollback-to-stack rule: an
  /// aggregate that does not fit in the remaining argument registers is
  /// passed entirely in memory rather than split.
  SignatureClassification
  computeSignatureInfo(mlir::TypeRange argTypes, mlir::Type retType,
                       mlir::Location loc,
                       size_t numFixedArgs = SIZE_MAX) const override;

protected:
  CoercionInfo classifyArgumentType(mlir::Type type, mlir::Location loc,
                                    bool isVariadicArg) const override;

  CoercionInfo classifyReturnType(mlir::Type type,
                                  mlir::Location loc) const override;

private:
  /// Number of integer and SSE argument registers a classified argument
  /// consumes. Identity (scalar/pointer) args are derived from their type.
  std::pair<unsigned, unsigned> argRegisterUsage(const CoercionInfo &info,
                                                 mlir::Type llvmType) const;

  /// Common classification logic for both arguments and return values.
  /// \param useSRet If true, >16 byte structs use sret; otherwise indirect.
  CoercionInfo classifyStructType(mlir::Type type, mlir::Location loc,
                                  bool useSRet) const;

  /// Classify 1-8 byte all-float structs for SSE register passing.
  ///
  /// \param size Size in bytes (1-8)
  /// \return Coercion to f32 (≤4 bytes) or f64 (5-8 bytes)
  CoercionInfo classifySmallSSEStruct(int64_t size) const;

  //===--------------------------------------------------------------------===//
  // Phase 2: Complete eightbyte classification (9-16 byte structs)
  //===--------------------------------------------------------------------===//

  /// Classification for each 8-byte chunk of a struct.
  enum class EightbyteClass {
    NoClass, // Uninitialized
    Integer, // Integer types (passed in GP registers)
    SSE,     // Float/double types (passed in SSE registers)
    Memory   // Must use memory (stack)
  };

  /// Classify a 9-16 byte struct into two eightbytes.
  ///
  /// This implements the System V ABI's two-register passing for medium
  /// structs. Each eightbyte is classified separately, then combined to
  /// determine the overall passing method.
  ///
  /// \param structType The LLVM struct type to classify
  /// \param size Total size in bytes (must be 9-16)
  /// \param loc Source location for diagnostics
  /// \return Coercion information (IntegerPair, SSEPair, Mixed, or Memory)
  CoercionInfo classifyTwoEightbyteStruct(mlir::LLVM::LLVMStructType structType,
                                          int64_t size, mlir::Location loc,
                                          bool useSRet) const;

  /// Classify a single 8-byte region of a struct.
  ///
  /// This examines all fields that overlap with the byte range
  /// [offset, offset+maxSize) and determines the eightbyte class.
  ///
  /// \param structType The LLVM struct type to examine
  /// \param offset Starting byte offset of this eightbyte
  /// \param maxSize Maximum size to consider (≤8)
  /// \return {classification, region size in bytes}
  std::pair<EightbyteClass, int64_t>
  classifyEightbyte(mlir::LLVM::LLVMStructType structType, int64_t offset,
                    int64_t maxSize) const;

  /// Get the LLVM type to use for an eightbyte.
  ///
  /// \param eightbyteClass The classification (Integer or SSE)
  /// \param size Number of bytes in this eightbyte (1-8)
  /// \return Appropriate integer (i8/i16/i32/i64) or float (f32/f64) type
  mlir::Type getEightbyteType(EightbyteClass eightbyteClass,
                              int64_t size) const;
};

} // namespace M::KGEN

#endif // KGEN_LIB_KGENTOLLVM_CABISYSTEMV_H
