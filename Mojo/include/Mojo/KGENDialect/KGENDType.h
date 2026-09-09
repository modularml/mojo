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
// This file declares the KGEN dtype constants and helpers
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENDTYPE_H
#define KGEN_KGENDIALECT_KGENDTYPE_H

#include "Support/ForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/DType.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace M::KGEN {

/// KGEN dtype is always compatible with GraphCompiler's DType but has some
/// additional types.
class KGENDType : public DType {
public:
  using DType::DType;
  using DType::getWidthInBits;

  enum ExtraCases : uint8_t {
    // Represents an address (e.g. a pointer). The size of the address is not
    // specified.
    address = kFirstExtendedOption,
    // Represents a signless integer that has the same size as a pointer.
    index,
    // TODO: add description
    uindex,
  };

  KGENDType(DType dtype) : DType(dtype) {}
  KGENDType(ExtraCases type) : DType(type) {}

  /// Return the equivalent KGENDType for the given builtin MLIR type.
  static std::pair<KGENDType, std::optional<int64_t>>
  getEquivalentDType(Type type);
  /// Return the equivalent builtin MLIR type for the given KGENDType.
  Type getEquivalentBuiltinType(MLIRContext *ctx);

  constexpr bool isAddress() const { return getValue() == ExtraCases::address; }
  constexpr bool isIndex() const { return getValue() == ExtraCases::index; }
  constexpr bool isUIndex() const { return getValue() == ExtraCases::uindex; }

  /// Returns true if the underlying dtype is arithmetic.
  constexpr bool isArithmetic() const {
    return isIndex() || isUIndex() || DType::isArithmetic();
  }

  /// Returns true if the underlying dtype is an integer and is signed. The
  /// index dtype is signed.
  constexpr bool isSInt() const { return isIndex() || DType::isSInt(); }

  /// Returns true if the underlying dtype is an integer and is unsigned. The
  /// uindex dtype is unsigned.
  constexpr bool isUInt() const { return isUIndex() || DType::isUInt(); }

  /// Returns true if the type is any valid integer representation.
  constexpr bool isIntLike() const {
    return isIndex() || isUIndex() || isInt() || isBool() || isAddress();
  }

  ssize_t getWidthInBits(TargetInfoAttr target) const {
    if (isAddress() || isIndex() || isUIndex())
      return target ? target.resolveIndexBitWidth() : 64;
    return DType::getWidthInBits();
  }

  /// Return the element type for it's string representation.
  static FailureOr<KGENDType> getFromString(StringRef str);

  /// Return a string form of this eltType suitable for printing and error
  /// messages. If the `libForm` flag is true, then the result will will follow
  /// the library spelling, e.g. `uint16` instead of `ui16`.
  std::string getAsString(bool libForm = false) const;
};

inline raw_ostream &operator<<(raw_ostream &os, KGENDType value) {
  return os << value.getAsString();
}
} // namespace M::KGEN

namespace llvm {
template <>
struct DenseMapInfo<M::KGEN::KGENDType> {
  static M::KGEN::KGENDType getEmptyKey() { return M::KGEN::KGENDType(); }

  static M::KGEN::KGENDType getTombstoneKey() {
    return M::KGEN::KGENDType(M::KGEN::KGENDType::ExtraCases::index + 1);
  }

  static unsigned getHashValue(const M::KGEN::KGENDType dtype) {
    return DenseMapInfo<uint8_t>::getHashValue(dtype.getValue());
  }

  static bool isEqual(const M::KGEN::KGENDType lhs,
                      const M::KGEN::KGENDType rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // KGEN_KGENDIALECT_KGENDTYPE_H
