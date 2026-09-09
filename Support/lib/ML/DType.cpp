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

#include "Support/ML/DType.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <sys/types.h>
#include <utility>
using namespace M;

/// Returns true if the type is a float8 type.
bool DType::isFloat8() const { return isFloat() && getWidthInBits() == 8; }

/// Return the width of this element in bits.  This returns -1 for unknown
/// width values.
ssize_t DType::getWidthInBits() const {
  // Handle complex separately from per-element types below.  We know that
  // complex element types are always at least a byte in size.
  if (isComplex()) {
    ssize_t strippedWidth = stripComplex().getWidthInBits();
    if (strippedWidth == -1)
      return -1;
    return strippedWidth * 2;
  }

  // This switch handles special cases inline, or determines the logarithmic
  // size of each element and breaks for the overflow check.
  switch (getValue()) {
  default:
    if (isInt())
      return getIntegerWidthInBits();
    if (auto *semantics = getFloatSemantics())
      return APFloat::getSizeInBits(*semantics);
    return -1;

    // Handle other types.
  case DType::kBool:
    return 8;
  }
}

/// Return the in-memory size for an array of the specified type with the
/// specified number of elements, or -1 for non-numeric types or too large
/// values.  This supports densely packed sub-byte types like i1, i2, i4.
ssize_t DType::getSizeInBytes(size_t numElements) const {
  auto sizeOr = getSizeInBytesChecked(numElements);
  if (failed(sizeOr))
    return -1;
  return *sizeOr;
}

/// Return the in-memory size for an array of the specified type with the
/// specified number of elements, or Error for non-numeric types or too large
/// values.  This supports densely packed sub-byte types like i1, i2, i4.
FailureOr<size_t> DType::getSizeInBytesChecked(size_t numElements) const {
  // Handle complex separately from per-element types below.
  if (isComplex()) {
    if (numElements > std::numeric_limits<ssize_t>::max() >> 1)
      return failure();
    return stripComplex().getSizeInBytesChecked(numElements * 2);
  }

  // This switch handles special cases inline, or determines the logarithmic
  // size of each element and breaks for the overflow check.
  size_t widthShift;
  if (isInt()) {
    // For integers, we just return the bit-width turned into bytes.  We treat
    // i1/i2/i4 types as being a single byte.
    widthShift = getIntegerWidthInLogBits();
  } else if (isFloat()) {
    ssize_t bitCount = getWidthInBits();
    if (!llvm::isPowerOf2_32(bitCount)) {
      if (numElements > std::numeric_limits<size_t>::max() / bitCount)
        return failure();
      return llvm::divideCeil(numElements * bitCount, 8);
    }
    widthShift = llvm::Log2_32(bitCount);
  } else if (isBool()) {
    widthShift = 3; // kBool is stored in a single byte each.
  } else {
    // Unhandled dtype
    return failure();
  }

  // i1,i2,i4,fp4 values are packed densely in memory.
  // We're going to do a truncating division (with a shift right) by the
  // element size, so make sure we round up to the next byte.
  if (widthShift <= 3) {
    if (numElements > std::numeric_limits<size_t>::max() >> widthShift)
      return failure();
    return llvm::divideCeil(numElements << widthShift, 8);
  }

  auto byteShift = widthShift - 3;
  // Otherwise we're growing. Convert to byte shift amount.
  if (numElements > std::numeric_limits<ssize_t>::max() >> byteShift)
    return failure();
  return numElements << byteShift;
}

/// Return a complex type if it is valid, otherwise fail.
FailureOr<DType> DType::getComplexChecked(DType eltType) {
  if (eltType.getWidthInBits() < 8 || eltType.isComplex())
    return failure();
  return getComplex(eltType);
}

/// This turns the printed form of a dtype back into a DType or
/// returns None if it is an unrecognized name.
FailureOr<DType> DType::getFromString(StringRef str) {
  if (str.empty())
    return failure();
  switch (str[0]) {
  case 'f':
#define DECLARE_FLOAT(SHORT_NAME, ...)                                         \
  if (str == #SHORT_NAME)                                                      \
    return DType(DType::SHORT_NAME);
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
    return failure();
  case 'u':
  case 's':
    if (str.size() >= 3 && str[1] == 'i') {
      unsigned width = 0;
      if (str.drop_front(2).getAsInteger(10, width))
        return failure();
      return getInt(width, /*isSigned=*/str[0] == 's');
    }
    return failure();

  case 'b':
    if (str == "bool")
      return DType(kBool);
    // Handle the bf16 special case, since it's a floating point type which
    // does not start with the letter 'f'.
    if (str == "bf16")
      return DType(bf16);
    return failure();
  case 'c':
    if (str.starts_with("complex<") && str.back() == '>') {
      auto elt = getFromString(str.drop_front(8).drop_back());
      if (failed(elt))
        return failure();
      return getComplexChecked(*elt);
    }
    return failure();
  case 'i':
    if (str == "invalid")
      return DType(invalid);
    return failure();
  default:
    // TODO: Could handle the eltType<unknown42> syntax if we wanted to.
    return failure();
  }
}

/// Return a string form of this eltType suitable for printing and error
/// messages.
std::string DType::getAsString() const {
  if (isComplex())
    return "complex<" + stripComplex().getAsString() + ">";
  if (isUInt())
    return "ui" + llvm::utostr(getIntegerWidthInBits());
  if (isSInt())
    return "si" + llvm::utostr(getIntegerWidthInBits());

  switch (getValue()) {
#define DECLARE_FLOAT(SHORT_NAME, ...)                                         \
  case DType::SHORT_NAME:                                                      \
    return #SHORT_NAME;
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
  case kBool:
    return "bool";
  case invalid:
    return "invalid";
  default:
    return "eltType<unknown" + llvm::utostr(getValue()) + ">";
  }
}

void DType::print(raw_ostream &os) const { os << getAsString(); }
void DType::dump() const { print(llvm::errs()); }

ErrorOr<std::pair<int32_t, int32_t>> DType::getMaxAndMinValue() const {
  return dispatch<ErrorOr<std::pair<int32_t, int32_t>>>()
      .when<DType::si32>([&]() {
        return std::pair(std::numeric_limits<int32_t>::max(),
                         std::numeric_limits<int32_t>::min());
      })
      .when<DType::si16>([&]() {
        return std::pair(std::numeric_limits<int16_t>::max(),
                         std::numeric_limits<int16_t>::min());
      })
      .when<DType::ui16>([&]() {
        return std::pair(std::numeric_limits<uint16_t>::max(),
                         std::numeric_limits<uint16_t>::min());
      })
      .when<DType::si8>([&]() {
        return std::pair(std::numeric_limits<int8_t>::max(),
                         std::numeric_limits<int8_t>::min());
      })
      .when<DType::ui8>([&]() {
        return std::pair(std::numeric_limits<uint8_t>::max(),
                         std::numeric_limits<uint8_t>::min());
      })
      .otherwise([&]() {
        return Error("Unsupported quantization dtype " + getAsString());
      });
}

/// This method returns the LLVM floating point semantics for the given DType,
/// or nullptr if the DType is not a floating point type LLVM knows about
/// (e.g. TF32).
const llvm::fltSemantics *DType::getFloatSemantics() const {
  switch (getValue()) {
  default:
    return nullptr;

#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, MLIR_TYPE, CXX_TYPE,      \
                      APFLOAT_TYPE, ...)                                       \
  case DType::SHORT_NAME:                                                      \
    return &APFLOAT_TYPE();
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
  }
}

size_t DType::getAlignment() const {
  constexpr auto getAlign = [](const auto *ptr) {
    return alignof(decltype(*ptr));
  };
  return dispatch<size_t>((void *)nullptr)
      .when<DType::f16>(getAlign)
      .when<DType::bf16>(getAlign)
      .when<DType::bf16>(getAlign)
      .when<DType::f4e2m1fn>(getAlign)
      .when<DType::f6e2m3fn>(getAlign)
      .when<DType::f6e3m2fn>(getAlign)
      .when<DType::f8e8m0fnu>(getAlign)
      .when<DType::f8e4m3fn>(getAlign)
      .when<DType::f8e4m3fnuz>(getAlign)
      .when<DType::f8e5m2>(getAlign)
      .when<DType::f8e5m2fnuz>(getAlign)
      .whenCXXType(getAlign);
}
