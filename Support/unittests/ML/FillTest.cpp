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

#include "Support/ML/Fill.h"
#include "llvm/ADT/APFloat.h"
#include "gtest/gtest.h"

using namespace M;

template <typename ScalarFunc>
static void checkBitsAsAPFloatIsVal(const llvm::fltSemantics &semantic,
                                    const DType &dtype, double target,
                                    ScalarFunc func) {
  uint64_t bits = 0;
  ErrorOrSuccess result = func(&bits, dtype);
  ASSERT_FALSE(result.isError());
  APFloat flt = llvm::APFloat(
      semantic, llvm::APInt(llvm::APFloat::getSizeInBits(semantic), bits));
  EXPECT_DOUBLE_EQ(target, flt.convertToDouble());
}

TEST(Fill, getScalarZeros) {
  checkBitsAsAPFloatIsVal(llvm::APFloat::BFloat(), DType::bf16, 1,
                          M::getScalarOne);
  checkBitsAsAPFloatIsVal(llvm::APFloat::IEEEhalf(), DType::f16, 1,
                          M::getScalarOne);
  checkBitsAsAPFloatIsVal(llvm::APFloat::IEEEsingle(), DType::f32, 1,
                          M::getScalarOne);
  checkBitsAsAPFloatIsVal(llvm::APFloat::IEEEdouble(), DType::f64, 1,
                          M::getScalarOne);

  checkBitsAsAPFloatIsVal(llvm::APFloat::BFloat(), DType::bf16, -1,
                          M::getScalarNegativeOne);
  checkBitsAsAPFloatIsVal(llvm::APFloat::IEEEhalf(), DType::f16, -1,
                          M::getScalarNegativeOne);
  checkBitsAsAPFloatIsVal(llvm::APFloat::IEEEsingle(), DType::f32, -1,
                          M::getScalarNegativeOne);
  checkBitsAsAPFloatIsVal(llvm::APFloat::IEEEdouble(), DType::f64, -1,
                          M::getScalarNegativeOne);
}

template <typename T, typename ScalarFunc>
static void checkIntegersIsOne(const DType &dtype, T expected,
                               ScalarFunc func) {
  T actual;
  ErrorOrSuccess result = func(&actual, dtype);
  ASSERT_FALSE(result.isError());
  EXPECT_EQ(expected, actual);
}

TEST(Fill, getScalarInts) {
  checkIntegersIsOne<bool>(DType::kBool, true, M::getScalarOne);

  checkIntegersIsOne<uint8_t>(DType::ui8, 1, M::getScalarOne);
  checkIntegersIsOne<uint16_t>(DType::ui16, 1, M::getScalarOne);
  checkIntegersIsOne<uint32_t>(DType::ui32, 1, M::getScalarOne);
  checkIntegersIsOne<uint64_t>(DType::ui64, 1, M::getScalarOne);

  checkIntegersIsOne<int8_t>(DType::si8, 1, M::getScalarOne);
  checkIntegersIsOne<int16_t>(DType::si16, 1, M::getScalarOne);
  checkIntegersIsOne<int32_t>(DType::si32, 1, M::getScalarOne);
  checkIntegersIsOne<int64_t>(DType::si64, 1, M::getScalarOne);

  checkIntegersIsOne<int8_t>(DType::si8, -1, M::getScalarNegativeOne);
  checkIntegersIsOne<int16_t>(DType::si16, -1, M::getScalarNegativeOne);
  checkIntegersIsOne<int32_t>(DType::si32, -1, M::getScalarNegativeOne);
  checkIntegersIsOne<int64_t>(DType::si64, -1, M::getScalarNegativeOne);
}

template <typename T>
static inline void fillRandomFloats(SmallVector<T> &data, size_t numElements,
                                    const DType &dtype) {
  ErrorOrSuccess result = M::fillRandom(data.data(), numElements, dtype);
  ASSERT_FALSE(result.isError());
}

static inline double getAsDouble(uint64_t bits,
                                 const llvm::fltSemantics &semantic) {
  return llvm::APFloat(semantic,
                       llvm::APInt(APFloat::getSizeInBits(semantic), bits))
      .convertToDouble();
}

TEST(Fill, getRandomFloats) {
  // Note: fillRandom is deterministically random, so reruns on the same
  // machine will result in the same results.
  const size_t numElements = 100;
  SmallVector<uint16_t> f16s(numElements);
  SmallVector<uint16_t> bf16s(numElements);
  SmallVector<uint32_t> f32s(numElements);
  SmallVector<uint64_t> f64s(numElements);

  fillRandomFloats(f16s, numElements, DType::f16);
  fillRandomFloats(bf16s, numElements, DType::bf16);
  fillRandomFloats(f32s, numElements, DType::f32);
  fillRandomFloats(f64s, numElements, DType::f64);

  // we just check everything is close to the doubles
  for (auto [f16, bf16, f32, f64] : llvm::zip(f16s, bf16s, f32s, f64s)) {
    double f64Val = getAsDouble(f64, llvm::APFloat::IEEEdouble());
    double f32Val = getAsDouble(f32, llvm::APFloat::IEEEsingle());
    double f16Val = getAsDouble(f16, llvm::APFloat::IEEEhalf());
    double bf16Val = getAsDouble(bf16, llvm::APFloat::BFloat());

    // everything is between [-1, 1.0)
    EXPECT_LT(f64Val, 1.0);
    EXPECT_GE(f64Val, -1.0);
    EXPECT_LT(f32Val, 1.0);
    EXPECT_GE(f32Val, -1.0);

    // Check all nums are roughly same (deterministically random property).
    // f16 and bf16 use `fillWithRandomSpecialFloats()` which draws the same
    // numbers as the generator for f32.
    EXPECT_NEAR(f32Val, f16Val, 0.01);
    EXPECT_NEAR(f32Val, bf16Val, 0.025);
  }
}
