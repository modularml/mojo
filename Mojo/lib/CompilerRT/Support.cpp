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

#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"

#include <cmath>

#if defined(__x86_64__)
// On x86 bfloat16 is passed in SSE registers. Since both float and __bf16
// are passed in the same register we can use the wider type and careful casting
// to conform to x86_64 psABI. This only works with the assumption that we're
// dealing with little-endian values passed in wider registers.
// Ideally this would directly use __bf16, but that type isn't supported by all
// compilers.
using BF16ABIType = float;
#else
// Default to uint16_t if we have nothing else.
using BF16ABIType = uint16_t;
#endif

// Constructs the 16 bit representation for a bfloat value from a float value.
// This implementation is adapted from Eigen.
static uint16_t float2bfloat(float floatValue) {
  const uint32_t kF32BfMantiBitDiff = 16;

  // Union used to make the int/float aliasing explicit so we can access the raw
  // bits.
  union Float32Bits {
    uint32_t u;
    float f;
  };

  if (std::isnan(floatValue))
    return std::signbit(floatValue) ? 0xFFC0 : 0x7FC0;

  Float32Bits floatBits;
  floatBits.f = floatValue;
  uint16_t bfloatBits;

  // Least significant bit of resulting bfloat.
  uint32_t lsb = (floatBits.u >> kF32BfMantiBitDiff) & 1;
  uint32_t roundingBias = 0x7fff + lsb;
  floatBits.u += roundingBias;
  bfloatBits = static_cast<uint16_t>(floatBits.u >> kF32BfMantiBitDiff);
  return bfloatBits;
}

// Provide a float->bfloat conversion routine in case the runtime doesn't have
// one.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT BF16ABIType
// NOLINTNEXTLINE(bugprone-reserved-identifier)
__truncsfbf2(float f) {
  uint16_t bf = float2bfloat(f);
  // The output can be a float type, bitcast it from uint16_t.
  BF16ABIType ret = 0;
  std::memcpy(&ret, &bf, sizeof(bf));
  return ret;
}

// Provide a double->bfloat conversion routine in case the runtime doesn't have
// one.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT BF16ABIType
// NOLINTNEXTLINE(bugprone-reserved-identifier)
__truncdfbf2(double d) {
  // This does a double rounding step, but it's precise enough for our use
  // cases.
  return __truncsfbf2(static_cast<float>(d));
}
