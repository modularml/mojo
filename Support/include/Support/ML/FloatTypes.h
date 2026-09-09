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
// This file declares `Float16` and `BFloat16` which are 16-bit floating point
// representations. `BFloat16` is often used on NVidia GPUs and other
// accelerators: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format.
// `Float16` is the IEEE "half" precision floating point representation:
// https://en.wikipedia.org/wiki/Half-precision_floating-point_format.
//
// `BFloat16` and `Float16` are available on some targets as `__bf16` and
// `_Float16` in Clang as a non-standard extension. They are being added to
// C++23 as `std::bfloat16_t` and `std::float16_t`, respectively, in the new
// `<stdfloat>` library. Until support is standardized, we maintain this for
// conversion to/from other types.
//
// The `bfloat16` format is quite simple: it's a truncated IEEE single-width
// floating point number. The sign and exponent (8 bits) are the same, but
// instead of a 23-bit mantissa, `bfloat16` has 7 bits in its mantissa. The
// result is a much less precise number that can represent the same "range" of
// numbers as a normal 32-bit `float`.
//
// The IEE `fp16` format has a 5-bit exponent and 10-bit mantissa.
//
// Visually:
//  +-----------+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  + IEEE fp16
//              |S|E|E|E|E|E|M|M|M|M|M|M|M|M|M|M| | | | | | | | | | | | | | | |
//  +-----------+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  + bfloat16
//              |S|E|E|E|E|E|E|E|E|M|M|M|M|M|M|M| | | | | | | | | | | | | | | |
//  +-----------+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  + IEEE fp32
//              |S|E|E|E|E|E|E|E|E|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|
//  +-----------+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//
// Note that this `BFloat16` implementation _truncates_ when converting from
// float32. This means that compared to torch, we will round some numbers
// towards zero when converting from higher precisions.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_FLOAT16_H
#define SUPPORT_ML_FLOAT16_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/bit.h"
#include <cstddef>
#include <cstdint>

namespace M {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

namespace Detail {

static constexpr bool is_little_endian =
    llvm::endianness::native == llvm::endianness::little;

template <llvm::APFloat::Semantics Semantics, int BitWidth>
struct float_conversion_generic_t {
  float_conversion_generic_t(float v) : bits(toBits(v)) {}
  explicit float_conversion_generic_t(uint8_t rawBits) : bits(rawBits) {}

  operator float() {
    // We use APFloat to do the heavy lifting here. This is probably not the
    // most efficient way, but it should be battle tested.
    llvm::APInt apInt(8, bits);
    llvm::APFloat apFloat(llvm::APFloat::EnumToSemantics(Semantics), apInt);
    bool ignore;
    apFloat.convert(llvm::APFloat::IEEEsingle(),
                    llvm::APFloat::rmNearestTiesToEven, &ignore);
    return apFloat.convertToFloat();
  }

  operator uint8_t() const { return bits; }

private:
  static inline uint8_t toBits(float v) {
    // We use APFloat to do the heavy lifting here. This is probably not the
    // most efficient way, but it should be battle tested.
    llvm::APFloat apFloat(v);
    bool ignore;
    apFloat.convert(llvm::APFloat::EnumToSemantics(Semantics),
                    llvm::APFloat::rmNearestTiesToEven, &ignore);

    // APInt will store a uint64_t array, which in this case should be
    // singleton. We will index into this, taking endianness into account.
    const llvm::APInt apInt = apFloat.bitcastToAPInt();
    constexpr size_t index = is_little_endian ? 0 : BitWidth - 1;
    return reinterpret_cast<const uint8_t *>(apInt.getRawData())[index];
  }

  uint8_t bits;
};

template <llvm::APFloat::Semantics Semantics>
struct float8_generic_t : float_conversion_generic_t<Semantics, 8> {
  using Base = float_conversion_generic_t<Semantics, 8>;
  using Base::Base;
};
} // namespace Detail

//===----------------------------------------------------------------------===//
// Float4
//===----------------------------------------------------------------------===//

namespace Float4 {
struct float4_e2m1fn_t
    : Detail::float_conversion_generic_t<llvm::APFloat::S_Float4E2M1FN, 4> {
  explicit float4_e2m1fn_t(uint8_t rawBits)
      : float_conversion_generic_t<llvm::APFloat::S_Float4E2M1FN, 4>(rawBits) {}
};
}; // namespace Float4

//===----------------------------------------------------------------------===//
// Float6
//===----------------------------------------------------------------------===//

namespace Float6 {
struct float6_e2m3fn_t
    : Detail::float_conversion_generic_t<llvm::APFloat::S_Float6E2M3FN, 6> {
  explicit float6_e2m3fn_t(uint8_t rawBits)
      : float_conversion_generic_t<llvm::APFloat::S_Float6E2M3FN, 6>(rawBits) {}
};
struct float6_e3m2fn_t
    : Detail::float_conversion_generic_t<llvm::APFloat::S_Float6E3M2FN, 6> {
  explicit float6_e3m2fn_t(uint8_t rawBits)
      : float_conversion_generic_t<llvm::APFloat::S_Float6E3M2FN, 6>(rawBits) {}
};
}; // namespace Float6

//===----------------------------------------------------------------------===//
// Float8
//===----------------------------------------------------------------------===//

namespace Float8 {
struct float8_e8m0fnu_t
    : Detail::float8_generic_t<llvm::APFloat::S_Float8E8M0FNU> {
  using Base = Detail::float8_generic_t<llvm::APFloat::S_Float8E8M0FNU>;
  using Base::Base;
};
struct float8_e3m4_t : Detail::float8_generic_t<llvm::APFloat::S_Float8E3M4> {
  using Base = Detail::float8_generic_t<llvm::APFloat::S_Float8E3M4>;
  using Base::Base;
};
struct float8_e4m3_t : Detail::float8_generic_t<llvm::APFloat::S_Float8E4M3> {
  using Base = Detail::float8_generic_t<llvm::APFloat::S_Float8E4M3>;
  using Base::Base;
};
struct float8_e4m3fn_t
    : Detail::float8_generic_t<llvm::APFloat::S_Float8E4M3FN> {
  using Base = Detail::float8_generic_t<llvm::APFloat::S_Float8E4M3FN>;
  using Base::Base;
};
struct float8_e4m3fnuz_t
    : Detail::float8_generic_t<llvm::APFloat::S_Float8E4M3FNUZ> {
  using Base = Detail::float8_generic_t<llvm::APFloat::S_Float8E4M3FNUZ>;
  using Base::Base;
};
struct float8_e5m2_t : Detail::float8_generic_t<llvm::APFloat::S_Float8E5M2> {
  using Base = Detail::float8_generic_t<llvm::APFloat::S_Float8E5M2>;
  using Base::Base;
};
struct float8_e5m2fnuz_t
    : Detail::float8_generic_t<llvm::APFloat::S_Float8E5M2FNUZ> {
  using Base = Detail::float8_generic_t<llvm::APFloat::S_Float8E5M2FNUZ>;
  using Base::Base;
};
} // namespace Float8

//===----------------------------------------------------------------------===//
// BFloat
//===----------------------------------------------------------------------===//

namespace BFloat {

struct bfloat16_t {
  // https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
  // Minimum negative value found by enabling sign bit on maximum value
  static constexpr uint16_t MIN_BITS = 0xFF7F;
  static constexpr uint16_t MAX_BITS = 0x7F7F;

  bfloat16_t(float v) : bits(floatToBf16Bits(v)) {}
  explicit bfloat16_t(uint16_t rawBits) : bits(rawBits) {}

  static bfloat16_t min() { return bfloat16_t(MIN_BITS); }
  static bfloat16_t max() { return bfloat16_t(MAX_BITS); }

  operator float() {
    constexpr size_t index = Detail::is_little_endian ? 1 : 0;
    float result = 0.;
    auto shorts = reinterpret_cast<uint16_t *>(&result);
    shorts[index] = bits;
    return result;
  }

private:
  static inline uint16_t floatToBf16Bits(float v) {
    constexpr size_t index = Detail::is_little_endian ? 1 : 0;
    return reinterpret_cast<uint16_t *>(&v)[index];
  }

  uint16_t bits;
};

} // namespace BFloat

//===----------------------------------------------------------------------===//
// Float16
//===----------------------------------------------------------------------===//

namespace Float16 {
struct float16_t {
  static constexpr uint16_t MIN_BITS = 0xFBFF;
  static constexpr uint16_t MAX_BITS = 0x7BFF;

  float16_t(float v) : bits(floatToF16Bits(v)) {}
  explicit float16_t(uint16_t rawBits) : bits(rawBits) {}

  static float16_t min() { return float16_t(MIN_BITS); }
  static float16_t max() { return float16_t(MAX_BITS); }

  operator float() {
    // If the system is big-endian, reverse the byte order of the bits.
    if constexpr (!Detail::is_little_endian)
      bits = (bits >> 8) | (bits << 8);

    // We use APFloat to do the heavy lifting here. This is probably not the
    // most efficient way, but it should be battle tested.
    llvm::APInt apInt(16, bits);
    llvm::APFloat apFloat(llvm::APFloat::IEEEhalf(), apInt);
    bool _;
    apFloat.convert(llvm::APFloat::IEEEsingle(),
                    llvm::APFloat::rmNearestTiesToEven, &_);
    return apFloat.convertToFloat();
  }

private:
  static inline uint16_t floatToF16Bits(float v) {
    // We use APFloat to do the heavy lifting here. This is probably not the
    // most efficient way, but it should be battle tested.
    llvm::APFloat apFloat(v);
    bool _;
    apFloat.convert(llvm::APFloat::IEEEhalf(),
                    llvm::APFloat::rmNearestTiesToEven, &_);

    // APInt will store a uint64_t array, which in this case should be
    // singleton. We will index into this, taking endianness into account.
    const llvm::APInt apInt = apFloat.bitcastToAPInt();
    constexpr size_t index = Detail::is_little_endian ? 0 : 3;
    return reinterpret_cast<const uint16_t *>(apInt.getRawData())[index];
  }

  uint16_t bits;
};

} // namespace Float16

} // namespace M

#endif // SUPPORT_ML_FLOAT16_H
