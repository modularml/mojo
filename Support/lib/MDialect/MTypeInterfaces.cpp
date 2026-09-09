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

#include "Support/MDialect/MTypeInterfaces.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include <climits>
#include <cstdint>
#include <optional>

using namespace M;

//===----------------------------------------------------------------------===//
// DataLayoutInterface
//===----------------------------------------------------------------------===//

namespace {
struct IntegerLayout
    : public DataLayoutInterface::ExternalModel<IntegerLayout, IntegerType> {
  /// The size of an integer type is its width rounded up to the nearest byte.
  std::optional<int64_t> getTypeSize(Type type, TargetInfoAttr target) const {
    return llvm::divideCeil(cast<IntegerType>(type).getWidth(), CHAR_BIT);
  }

  /// The alignment of an integer type is its width in bytes rounded up to the
  /// nearest power of 2, but capped at the pointer width.
  std::optional<int64_t> getTypeAlign(Type type, TargetInfoAttr target) const {
    return target.getDataLayout().getIntegerABIAlign(
        cast<IntegerType>(type).getWidth());
  }
};

struct FloatLayout
    : public DataLayoutInterface::ExternalModel<FloatLayout, Type> {
  /// The size of an integer type is its width in bytes.
  std::optional<int64_t> getTypeSize(Type type, TargetInfoAttr target) const {
    return cast<FloatType>(type).getWidth() / CHAR_BIT;
  }

  /// The alignment of a float type is its width in bytes rounded up to the
  /// nearest power of 2, but capped at the pointer width.
  std::optional<int64_t> getTypeAlign(Type type, TargetInfoAttr target) const {
    return target.getDataLayout().getFloatABIAlign(
        cast<FloatType>(type).getWidth());
  }
};

struct FunctionLayout
    : public DataLayoutInterface::ExternalModel<FunctionLayout, FunctionType> {
  /// The size of a function type is the pointer width.
  std::optional<int64_t> getTypeSize(Type type, TargetInfoAttr target) const {
    return llvm::divideCeil(target.getDataLayout().getPointerBitWidth(),
                            CHAR_BIT);
  }

  /// The align of a function type is the pointer width.
  std::optional<int64_t> getTypeAlign(Type type, TargetInfoAttr target) const {
    return target.getDataLayout().getPointerABIAlign();
  }
};

struct IndexLayout
    : public DataLayoutInterface::ExternalModel<IndexLayout, IndexType> {
  /// The size of an index type is the one found in the TargetInfoAttr.
  std::optional<int64_t> getTypeSize(Type type, TargetInfoAttr target) const {
    return llvm::divideCeil(target.resolveIndexBitWidth(), CHAR_BIT);
  }

  /// The align of an index type is the pointer width.
  std::optional<int64_t> getTypeAlign(Type type, TargetInfoAttr target) const {
    return target.getDataLayout().getPointerABIAlign();
  }
};
} // namespace

void MDialect::injectTypeInterfaces() {
#define DECLARE_FLOAT(_, __, ___, MLIR_TYPE, ...)                              \
  MLIR_TYPE::attachInterface<FloatLayout>(*getContext());
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
  IntegerType::attachInterface<IntegerLayout>(*getContext());
  FunctionType::attachInterface<FunctionLayout>(*getContext());
  IndexType::attachInterface<IndexLayout>(*getContext());
}

std::optional<int64_t>
DataLayoutInterface::getTypeStoreSize(TargetInfoAttr target, Type type) {
  if (auto iface = llvm::dyn_cast<DataLayoutInterface>(type))
    return iface.getTypeSize(target);
  return {};
}

std::optional<int64_t>
DataLayoutInterface::getTypeAllocSize(TargetInfoAttr target, Type type) {
  std::optional<int64_t> typeSize = getTypeStoreSize(target, type);
  std::optional<int64_t> typeABIAlign = getTypeABIAlign(target, type);
  if (!typeSize || !typeABIAlign)
    return {};
  return llvm::alignTo(*typeSize, *typeABIAlign);
}

std::optional<int64_t>
DataLayoutInterface::getTypeABIAlign(TargetInfoAttr target, Type type) {
  if (auto iface = llvm::dyn_cast<DataLayoutInterface>(type))
    return iface.getTypeAlign(target);
  return {};
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MTypeInterfaces.cpp.inc"
