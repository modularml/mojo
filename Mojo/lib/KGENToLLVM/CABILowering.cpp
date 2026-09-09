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

#include "CABILowering.h"
#include "CABIAAPCS.h"
#include "CABISystemV.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace M;
using namespace M::KGEN;

namespace {
/// Command-line options for C ABI lowering.
struct CABIOptions {
  llvm::cl::opt<bool> skipCoercion{
      "skip-c-abi-coercion",
      llvm::cl::desc("Disable C ABI struct coercion (for debugging)"),
      llvm::cl::init(false), llvm::cl::Hidden};
};
} // namespace

static llvm::ManagedStatic<CABIOptions> CABIOpts;

//===----------------------------------------------------------------------===//
// CABIInfo base implementation
//===----------------------------------------------------------------------===//

SignatureClassification
CABIInfo::computeSignatureInfo(mlir::TypeRange argTypes, mlir::Type retType,
                               mlir::Location loc, size_t numFixedArgs) const {
  SignatureClassification result;
  if (retType)
    result.ret = classifyReturnType(retType, loc);
  result.args.reserve(argTypes.size());
  for (auto [idx, type] : llvm::enumerate(argTypes)) {
    bool isVariadicArg = idx >= numFixedArgs;
    result.args.push_back(classifyArgumentType(type, loc, isVariadicArg));
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Factory Function
//===----------------------------------------------------------------------===//

std::unique_ptr<CABIInfo>
M::KGEN::createCABIInfo(const llvm::Triple &triple, mlir::MLIRContext *ctx,
                        const LLVMDataLayout &dataLayout) {

  // Check if C ABI functionality is enabled
  if (!CabiUtils::isCABIEnabled()) {
    // C ABI disabled - use default pass-through ABI
    return std::make_unique<DefaultCABIInfo>(ctx, dataLayout);
  }

  // Detect architecture and select appropriate ABI
  llvm::Triple::ArchType arch = triple.getArch();

  switch (arch) {
  case llvm::Triple::x86_64:
    // x86-64: Use System V AMD64 ABI (Linux, macOS, BSD)
    return std::make_unique<SystemVABIInfo>(ctx, dataLayout);

  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
  case llvm::Triple::aarch64_32:
    // ARM64: Use AAPCS (Procedure Call Standard for ARM 64-bit)
    // Used on: Linux ARM64, macOS Apple Silicon, iOS
    // isDarwin controls variadic HFA coercion (Darwin=GP-only va_list).
    return std::make_unique<AAPCSABIInfo>(ctx, dataLayout, triple.isOSDarwin());

  case llvm::Triple::x86:
    // 32-bit x86: Use default pass-through ABI
    // TODO: Could implement cdecl/stdcall/fastcall variants if needed
    return std::make_unique<DefaultCABIInfo>(ctx, dataLayout);

  default:
    // Unsupported architecture: use default pass-through ABI
    return std::make_unique<DefaultCABIInfo>(ctx, dataLayout);
  }
}

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

int64_t CabiUtils::getStructSize(mlir::LLVM::LLVMStructType type,
                                 const LLVMDataLayout &dataLayout) {
  // Use LLVMDataLayout which works with LLVM types.
  // This gives us the actual LLVM layout including padding from alignment.
  return dataLayout.getTypeStoreSize(type);
}

/// Recursively determine the canonical leaf float bit width for a struct.
///
/// Traverses all fields (recursing into nested LLVMStructType fields) and
/// returns the bit width if every leaf field is the same float type.
/// Returns 0 if any non-float field is found, the struct is empty, or
/// float types are mixed (e.g., f32 and f64 together).
static unsigned getLeafFloatBitWidth(mlir::LLVM::LLVMStructType type) {
  std::optional<unsigned> canonicalBitWidth;
  for (mlir::Type fieldType : type.getBody()) {
    unsigned fieldBitWidth = 0;
    if (auto floatType = dyn_cast<mlir::FloatType>(fieldType)) {
      fieldBitWidth = floatType.getWidth();
    } else if (auto vecType = dyn_cast<mlir::VectorType>(fieldType)) {
      if (auto floatElem = dyn_cast<mlir::FloatType>(vecType.getElementType()))
        fieldBitWidth = floatElem.getWidth();
      else
        return 0; // Non-float vector element
    } else if (auto nestedStruct =
                   dyn_cast<mlir::LLVM::LLVMStructType>(fieldType)) {
      fieldBitWidth = getLeafFloatBitWidth(nestedStruct);
      if (fieldBitWidth == 0)
        return 0; // Nested struct contains non-float or mixed floats
    } else {
      return 0; // Non-float field (integer, pointer, etc.)
    }
    if (!canonicalBitWidth)
      canonicalBitWidth = fieldBitWidth;
    else if (*canonicalBitWidth != fieldBitWidth)
      return 0; // Heterogeneous: mixed float types (e.g., f32 + f64)
  }
  return canonicalBitWidth.value_or(0);
}

bool CabiUtils::isAllFloatStruct(mlir::LLVM::LLVMStructType type) {
  auto fields = type.getBody();
  if (fields.empty())
    return false;

  // ARM64 AAPCS HFA (Homogeneous Floating-point Aggregate) limit: at most 4
  // top-level fields. Note: this checks top-level fields, not leaf float count.
  if (fields.size() > 4)
    return false;

  return getLeafFloatBitWidth(type) != 0;
}

mlir::IntegerType CabiUtils::getIntegerTypeForSize(int64_t size,
                                                   mlir::MLIRContext *ctx) {
  unsigned bitWidth;
  if (size == 1) {
    bitWidth = 8;
  } else if (size == 2) {
    bitWidth = 16;
  } else if (size <= 4) {
    bitWidth = 32;
  } else if (size <= 8) {
    bitWidth = 64;
  } else {
    llvm_unreachable("Invalid size for integer type");
  }
  return mlir::IntegerType::get(ctx, bitWidth);
}

CoercionInfo CabiUtils::classifySmallIntegerStruct(int64_t size,
                                                   mlir::MLIRContext *ctx) {
  CoercionInfo info;
  info.argClass = ABIArgClass::Integer;
  info.coercedType = getIntegerTypeForSize(size, ctx);
  return info;
}

bool CabiUtils::isCABIEnabled() {
  // C ABI coercion is enabled by default.
  // Use --skip-c-abi-coercion to disable (for debugging).
  return !CABIOpts->skipCoercion;
}
