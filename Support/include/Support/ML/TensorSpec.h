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
// This file declares the TensorSpec and TensorSpec classes, which hold a
// TensorShape and TensorDType together in one value.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_TENSORSPEC_H
#define SUPPORT_ML_TENSORSPEC_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LLVMYAMLForwardDecls.h"
#include "Support/ML/DType.h"
#include "Support/ML/TensorShape.h"
#include <cstddef>
#include <initializer_list>
#include <string>

namespace M {

/// TensorSpec is a memory efficient representation of a shape and
/// element type, implemented using TensorShape.
class TensorSpec : public TensorShape {
public:
  TensorSpec() : TensorShape() { setEltType(DType::invalid); }
  template <typename ShapeType>
  TensorSpec(const ShapeType &shape, DType eltType) : TensorShape(shape) {
    setEltType(eltType);
  }
  template <typename ShapeType>
  TensorSpec(const ShapeType &shape, TensorRankStyle style, DType eltType)
      : TensorShape(shape, style) {
    setEltType(eltType);
  }
  template <typename T>
  TensorSpec(const std::initializer_list<T> &shape, DType eltType)
      : TensorShape(shape) {
    static_assert(std::is_integral_v<T>, "shape dimensions must be integral");
    setEltType(eltType);
  }
  template <typename T>
  TensorSpec(const std::initializer_list<T> &shape, TensorRankStyle style,
             DType eltType)
      : TensorShape(shape, style) {
    static_assert(std::is_integral_v<T>, "shape dimensions must be integral");
    setEltType(eltType);
  }

  // This class stores the ElementType in the auxiliary storage field of the
  // underlying TensorShape.
  DType getEltType() const { return DType(getAuxiliaryStorage()); }
  void setEltType(DType type) { setAuxiliaryStorage(type.getValue()); }

  size_t getSizeInBytes() const {
    auto sizeOr = getEltType().getSizeInBytesChecked(getNumElements());
    assert(succeeded(sizeOr) && "overflow");
    return *sizeOr;
  }

  /// This turns the printed form of a TensorSpec back into a TensorSpec or
  /// failure if it is an unrecognized format.
  static ErrorOr<TensorSpec> parseFromString(StringRef str);

  void print(raw_ostream &os) const;
  std::string getAsString() const;

  bool operator==(const TensorSpec &rhs) const {
    return storage.equalsIncludingAux(rhs.storage);
  }
  bool operator!=(const TensorSpec &rhs) const { return !(*this == rhs); }
};

inline raw_ostream &operator<<(raw_ostream &os, const TensorSpec &value) {
  value.print(os);
  return os;
}

// TensorSpec should always be two words, the same as TensorSpec.
static_assert(sizeof(void *) != 8 || sizeof(TensorSpec) == 16);

} // namespace M

LLVM_FWD_YAML_DECLARE_SCALAR_TRAITS(M::TensorSpec)

#endif // SUPPORT_ML_TENSORSPEC_H
