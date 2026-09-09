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

#ifndef SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOTYPES_H
#define SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOTYPES_H

#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/Casting.h"
#include <cstdint>

//===----------------------------------------------------------------------===//
// DebugInfoType
//===----------------------------------------------------------------------===//

namespace M::DebugInfo {
class DIFileAttr;
class DIScopeAttr;

/// This class represents the base class of DebugInfo types.
class DIType : public Type {
public:
  using Type::Type;

  /// Return the size of the type bits, or zero if the size cannot be
  /// determined.
  uint64_t getSizeInBits() const;

  /// Return the alignment of the type bits, or zero if the alignment cannot
  /// be determined.
  uint32_t getAlignInBits() const;

  /// Support LLVM type casting.
  static bool classof(Type type) {
    return llvm::isa<DebugInfoDialect>(type.getDialect());
  }
};
} // namespace M::DebugInfo

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h.inc"

//===----------------------------------------------------------------------===//
// DIBasicType
//===----------------------------------------------------------------------===//

namespace M::DebugInfo {
/// Simple basic type wrapper for types encoded as a boolean.
class DIBasicBoolType : public DIBasicType {
public:
  static DIBasicType get(MLIRContext *ctx, const Twine &name,
                         uint64_t sizeInBits, uint32_t alignInBits);
};

/// Simple basic type wrapper for types encoded as an unsigned integer.
class DIBasicUIntType : public DIBasicType {
public:
  static DIBasicType get(MLIRContext *ctx, const Twine &name,
                         uint64_t sizeInBits, uint32_t alignInBits);
};

/// Simple basic type wrapper for types encoded as a signed integer.
class DIBasicSIntType : public DIBasicType {
public:
  static DIBasicType get(MLIRContext *ctx, const Twine &name,
                         uint64_t sizeInBits, uint32_t alignInBits);
};

/// Simple basic type wrapper for types encoded as a float.
class DIBasicFloatType : public DIBasicType {
public:
  static DIBasicType get(MLIRContext *ctx, const Twine &name,
                         uint64_t sizeInBits, uint32_t alignInBits);
};
} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOTYPES_H
