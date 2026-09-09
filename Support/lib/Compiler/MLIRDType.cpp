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

#include "Support/Compiler/MLIRDType.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/ML/DType.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

using namespace M;

bool M::areEquivalentFloatTypes(DType dtype, mlir::FloatType fpType) {
  assert(dtype.isFloat() && "expected a float dtype");
  switch (dtype.getValue()) {
#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, MLIR_TYPE, ...)           \
  case DType::SHORT_NAME:                                                      \
    return isa<MLIR_TYPE>(fpType);
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
  default:
    return false;
  }
}

FloatType M::getEquivalentFloatType(MLIRContext *ctx, DType dtype) {
  switch (dtype.getValue()) {
#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, MLIR_TYPE, ...)           \
  case DType::SHORT_NAME:                                                      \
    return MLIR_TYPE::get(ctx);
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
  default:
    return {}; // null denotes failure
  }
}

bool M::hasEquivalentFloatType(DType dtype) {
  switch (dtype.getValue()) {
#define DECLARE_FLOAT(SHORT_NAME, ...) case DType::SHORT_NAME:
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
    return true;
  default:
    return false;
  }
}

IntegerType M::getEquivalentIntegerType(MLIRContext *ctx, DType dtype) {
  if (dtype.isBool())
    return IntegerType::get(ctx, 1, IntegerType::Signless);
  if (dtype.isInt())
    return IntegerType::get(ctx, dtype.getWidthInBits(),
                            dtype.isSInt() ? IntegerType::Signed
                                           : IntegerType::Unsigned);
  return {}; // null denotes failure
}

bool M::hasEquivalentIntegerType(DType dtype) {
  return dtype.isInt() || dtype.isBool();
}

DType M::getEquivalentDType(FloatType fpType) {
#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, MLIR_TYPE, ...)           \
  if (llvm::isa<MLIR_TYPE>(fpType))                                            \
    return DType::SHORT_NAME;
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT

  return {}; // invalid denotes failure
}

DType M::getEquivalentDType(IntegerType intType) {
  if (intType.isSignless()) {
    if (intType.getWidth() == 1)
      return DType::kBool;
    else
      return {}; // invalid denotes failure
  }
  FailureOr<DType> optDType =
      DType::getInt(intType.getIntOrFloatBitWidth(), intType.isSignedInteger());
  if (failed(optDType))
    return {}; // invalid denotes failure
  return *optDType;
}

std::pair<DType, std::optional<int64_t>> M::getEquivalentDType(Type type) {
  std::optional<int64_t> vecSize;

  // Consume any vector size present
  if (auto vecTy = dyn_cast<VectorType>(type)) {
    // DTypes can only represent static shapes
    if (!vecTy.hasStaticShape())
      return {{}, vecSize};
    vecSize = vecTy.getNumElements();
    type = vecTy.getElementType();
  }

  DType dtype;
  if (auto intty = dyn_cast<IntegerType>(type))
    dtype = M::getEquivalentDType(intty);
  else if (auto fpty = dyn_cast<FloatType>(type))
    dtype = M::getEquivalentDType(fpty);

  return {dtype, vecSize};
}
