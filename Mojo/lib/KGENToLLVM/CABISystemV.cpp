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

#include "CABISystemV.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

SystemVABIInfo::SystemVABIInfo(mlir::MLIRContext *ctx,
                               const LLVMDataLayout &dataLayout)
    : CABIInfo(ctx, dataLayout) {}

//===----------------------------------------------------------------------===//
// Argument Classification
//===----------------------------------------------------------------------===//

CoercionInfo SystemVABIInfo::classifyArgumentType(mlir::Type type,
                                                  mlir::Location loc,
                                                  bool isVariadicArg) const {
  // On x86-64 System V, variadic struct args follow the same classification
  // rules as non-variadic args. LLVM handles placing them on the stack.
  // (isVariadicArg is used by ARM64 AAPCS but not needed here.)
  (void)isVariadicArg;

  // Arrays should be passed indirectly, as per the C specification
  if (isa<mlir::LLVM::LLVMArrayType>(type))
    return CoercionInfo::indirectArgument(/*useByval=*/false);

  return classifyStructType(type, loc, /*useSRet=*/false);
}

//===----------------------------------------------------------------------===//
// Return Value Classification
//===----------------------------------------------------------------------===//

CoercionInfo SystemVABIInfo::classifyReturnType(mlir::Type type,
                                                mlir::Location loc) const {
  return classifyStructType(type, loc, /*useSRet=*/true);
}

//===----------------------------------------------------------------------===//
// Common Struct Classification
//===----------------------------------------------------------------------===//

CoercionInfo SystemVABIInfo::classifyStructType(mlir::Type type,
                                                mlir::Location loc,
                                                bool useSRet) const {
  // Only LLVM struct types need C ABI classification
  auto structType = dyn_cast<mlir::LLVM::LLVMStructType>(type);
  if (!structType) {
    return CoercionInfo{}; // Non-struct types pass through
  }

  int64_t size = CabiUtils::getStructSize(structType, dataLayout);

  // Structs >16 bytes use MEMORY class (pointer for args, sret for returns)
  if (size > 16) {
    return useSRet ? CoercionInfo::sretReturn()
                   : CoercionInfo::indirectArgument(/*useByval=*/true);
  }

  // 1-8 byte structs: all-float → SSE registers, otherwise → integer registers
  // Per System V ABI: mixed int/float eightbytes use integer classification.
  if (size <= 8) {
    if (CabiUtils::isAllFloatStruct(structType)) {
      return classifySmallSSEStruct(size);
    }
    return CabiUtils::classifySmallIntegerStruct(size, ctx);
  }

  // 9-16 byte structs: two-eightbyte classification
  return classifyTwoEightbyteStruct(structType, size, loc, useSRet);
}

//===----------------------------------------------------------------------===//
// Small SSE Struct Classification (≤8 byte all-float structs)
//===----------------------------------------------------------------------===//

CoercionInfo SystemVABIInfo::classifySmallSSEStruct(int64_t size) const {
  assert(size <= 8 &&
         "SSE struct classification only handles types up to 8 bytes");
  CoercionInfo info;
  info.argClass = ABIArgClass::SSE;

  // For ≤4 bytes (e.g., single float): use f32
  // For ≤8 bytes (e.g., double, or {float, float}): use f64
  // Note: {float, float} is technically <2 x float> in Clang, but f64
  // works through the store/load bitcast pattern in prepareCoercedArgument.
  if (size <= 4) {
    info.coercedType = mlir::Float32Type::get(ctx);
  } else {
    info.coercedType = mlir::Float64Type::get(ctx);
  }

  return info;
}

//===----------------------------------------------------------------------===//
// Phase 2: Two-Eightbyte Classification (9-16 byte structs)
//===----------------------------------------------------------------------===//

CoercionInfo SystemVABIInfo::classifyTwoEightbyteStruct(
    mlir::LLVM::LLVMStructType structType, int64_t size, mlir::Location loc,
    bool useSRet) const {
  // Classify each eightbyte independently
  auto [class1, size1] = classifyEightbyte(structType, 0, 8);
  int64_t secondMaxSize = size - 8;
  auto [class2, size2] = classifyEightbyte(structType, 8, secondMaxSize);

  // If either eightbyte is Memory, the whole struct uses memory
  if (class1 == EightbyteClass::Memory || class2 == EightbyteClass::Memory) {
    return useSRet ? CoercionInfo::sretReturn()
                   : CoercionInfo::indirectArgument(/*useByval=*/true);
  }

  CoercionInfo info;

  // Determine the ABIArgClass based on the two eightbyte classes
  if (class1 == EightbyteClass::Integer && class2 == EightbyteClass::Integer) {
    info.argClass = ABIArgClass::IntegerPair;
  } else if (class1 == EightbyteClass::SSE && class2 == EightbyteClass::SSE) {
    info.argClass = ABIArgClass::SSEPair;
  } else {
    info.argClass = ABIArgClass::Mixed;
  }

  info.coercedType = getEightbyteType(class1, size1);
  info.coercedSecondType = getEightbyteType(class2, size2);

  return info;
}

std::pair<SystemVABIInfo::EightbyteClass, int64_t>
SystemVABIInfo::classifyEightbyte(mlir::LLVM::LLVMStructType structType,
                                  int64_t offset, int64_t maxSize) const {
  auto fields = structType.getBody();
  if (fields.empty()) {
    return {EightbyteClass::NoClass, maxSize};
  }

  EightbyteClass result = EightbyteClass::NoClass;
  int64_t fieldOffset = 0;

  for (mlir::Type fieldType : fields) {
    int64_t fieldSize = dataLayout.getTypeStoreSize(fieldType);
    int64_t fieldAlign = dataLayout.getTypeABIAlign(fieldType);

    // Align fieldOffset to satisfy the field's ABI alignment requirement.
    // TODO(MOCO-3369): Add LLVMDataLayout::getStructFieldOffsets() to replace
    // this manual accumulation with a single pre-computed offset array,
    // mirroring llvm::DataLayout::getStructLayout()->getElementOffset().
    fieldOffset = llvm::alignTo(fieldOffset, fieldAlign);

    int64_t fieldEnd = fieldOffset + fieldSize;

    // Check if this field overlaps with [offset, offset+maxSize)
    // TODO(Code Review #7): Per the System V ABI spec, a field that straddles
    // an eightbyte boundary (starts in one 8-byte region but extends into the
    // next) should classify the whole struct as MEMORY. This is missing and can
    // cause silent miscompilation for packed structs.
    //
    // Bug: A field at offset O with size S straddles if:
    //   fieldStartEightbyte = fieldOffset / 8
    //   fieldEndEightbyte = (fieldEnd - 1) / 8
    //   if (fieldStartEightbyte != fieldEndEightbyte) → MEMORY class
    //
    // Example: struct __attribute__((packed)) { uint32_t a; uint64_t b; }
    //   - Field 'a': bytes 0-3 (eightbyte 0)
    //   - Field 'b': bytes 4-11 (straddles eightbytes 0 and 1) → should be
    //   MEMORY
    //
    // Why no tests fail: All existing C ABI integration tests use naturally-
    // aligned structs. The compiler adds padding between fields, preventing
    // straddles. To trigger this bug requires __attribute__((packed)) structs,
    // but Mojo currently has no way to declare packed structs that match C's
    // layout, so we cannot write end-to-end tests for this case.
    //
    // Assert: fail loudly on straddling fields rather than silently
    // misclassifying them. See TODO above for the full explanation.
    assert((fieldOffset / 8) == ((fieldEnd - 1) / 8) &&
           "packed struct field straddles eightbyte boundary; needs MEMORY "
           "classification");

    if (fieldEnd > offset && fieldOffset < offset + maxSize) {
      // Determine field class
      EightbyteClass fieldClass = EightbyteClass::Integer; // default

      if (isa<mlir::FloatType>(fieldType)) {
        fieldClass = EightbyteClass::SSE;
      } else if (auto vecType = dyn_cast<mlir::VectorType>(fieldType)) {
        // Vector types with float elements (converted from SIMDType)
        if (isa<mlir::FloatType>(vecType.getElementType())) {
          fieldClass = EightbyteClass::SSE;
        }
      } else if (auto nestedStruct =
                     dyn_cast<mlir::LLVM::LLVMStructType>(fieldType)) {
        // Recursively classify the nested struct fields that overlap with
        // this eightbyte region [offset, offset+maxSize).
        // Translate the eightbyte region into nested-struct coordinates:
        //   overlap = [max(offset,fieldOffset), min(offset+maxSize, fieldEnd))
        //   nestedOffset = overlapStart - fieldOffset
        int64_t overlapStart = std::max(offset, fieldOffset);
        int64_t overlapEnd = std::min(offset + maxSize, fieldEnd);
        int64_t nestedOffset = overlapStart - fieldOffset;
        int64_t nestedMaxSize = overlapEnd - overlapStart;
        fieldClass =
            classifyEightbyte(nestedStruct, nestedOffset, nestedMaxSize).first;
      }

      // Merge: INTEGER wins over SSE (System V ABI rule)
      if (result == EightbyteClass::NoClass) {
        result = fieldClass;
      } else if (result == EightbyteClass::SSE &&
                 fieldClass == EightbyteClass::Integer) {
        result = EightbyteClass::Integer;
      }
      // SSE + SSE = SSE, Integer + anything = Integer (already set)
    }

    fieldOffset += fieldSize;
  }

  if (result == EightbyteClass::NoClass) {
    result = EightbyteClass::Integer;
  }

  return {result, maxSize};
}

mlir::Type SystemVABIInfo::getEightbyteType(EightbyteClass eightbyteClass,
                                            int64_t size) const {
  if (eightbyteClass == EightbyteClass::SSE) {
    // SSE class: use float types
    if (size <= 4) {
      return mlir::Float32Type::get(ctx);
    }
    assert(size <= 8 && "SSE eightbyte must not exceed 8 bytes");
    return mlir::Float64Type::get(ctx);
  }

  // Integer class (or NoClass): use integer types
  return CabiUtils::getIntegerTypeForSize(size, ctx);
}

//===----------------------------------------------------------------------===//
// Whole-signature classification (rollback-to-stack)
//===----------------------------------------------------------------------===//

std::pair<unsigned, unsigned>
SystemVABIInfo::argRegisterUsage(const CoercionInfo &info,
                                 mlir::Type llvmType) const {
  switch (info.argClass) {
  case ABIArgClass::Integer:
    return {1, 0};
  case ABIArgClass::SSE:
    return {0, 1};
  case ABIArgClass::IntegerPair:
    return {2, 0};
  case ABIArgClass::SSEPair:
    return {0, 2};
  case ABIArgClass::Mixed:
    return {1, 1};
  case ABIArgClass::Memory:
    // Passed in memory; consumes no argument registers.
    return {0, 0};
  case ABIArgClass::NoClass:
    break; // Identity arg: derive usage from the LLVM type below.
  }

  if (isa<mlir::LLVM::LLVMPointerType>(llvmType))
    return {1, 0};
  if (auto intTy = dyn_cast<mlir::IntegerType>(llvmType)) {
    unsigned regs = (intTy.getWidth() + 63) / 64;
    return {std::max(regs, 1u), 0};
  }
  if (auto floatTy = dyn_cast<mlir::FloatType>(llvmType)) {
    // f32/f64 use one SSE register; wider x87/quad types are not register args.
    if (floatTy.getWidth() <= 64)
      return {0, 1};
    return {0, 0};
  }
  if (auto vecTy = dyn_cast<mlir::VectorType>(llvmType)) {
    unsigned bytes = dataLayout.getTypeStoreSize(vecTy);
    unsigned regs = (bytes + 15) / 16;
    return {0, std::max(regs, 1u)};
  }

  // pointer / integer / float / vector are handled above; an identity arg of
  // any other type means the register accounting is incomplete. Guessing a
  // count here would silently mis-budget and break rollback decisions.
  assert(false && "unhandled identity argument type in register accounting");
  return {0, 0};
}

SignatureClassification
SystemVABIInfo::computeSignatureInfo(mlir::TypeRange argTypes,
                                     mlir::Type retType, mlir::Location loc,
                                     size_t numFixedArgs) const {
  SignatureClassification result;
  if (retType)
    result.ret = classifyReturnType(retType, loc);

  // SysV provides 6 GP and 8 SSE argument registers. An sret return reserves
  // one GP register for the hidden result pointer.
  unsigned freeInt = 6;
  unsigned freeSSE = 8;
  if (result.ret.useSRet)
    freeInt -= 1;

  result.args.reserve(argTypes.size());
  for (auto [idx, type] : llvm::enumerate(argTypes)) {
    bool isVariadicArg = idx >= numFixedArgs;
    CoercionInfo info = classifyArgumentType(type, loc, isVariadicArg);

    // Only fixed args take part in register accounting; variadic args keep
    // their classification.
    if (!isVariadicArg) {
      auto [needInt, needSSE] = argRegisterUsage(info, type);
      if (needInt <= freeInt && needSSE <= freeSSE) {
        freeInt -= needInt;
        freeSSE -= needSSE;
      } else if (!info.isIdentity() && !info.useIndirect) {
        // An in-register aggregate that does not fit rolls back to memory
        // rather than being split across registers and stack.
        info = CoercionInfo::indirectArgument(/*useByval=*/true);
      }
      // A scalar that does not fit stays direct; the backend places it on the
      // stack.
    }
    result.args.push_back(info);
  }
  return result;
}
