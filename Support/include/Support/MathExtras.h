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

#ifndef SUPPORT_MATHEXTRAS_H
#define SUPPORT_MATHEXTRAS_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"

#include "llvm/ADT/FloatingPointMode.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <type_traits>

namespace M {

/// Checks if the two input values are numerically the same.
/// When the type is integral, then equality is checked. When the type is
/// floating point, then this checks if the two input values are numerically the
/// close using the abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol) formula.
/// The default absolute and relative tolerances are picked from the numpy
/// default values. If IsNanSensitive is false, then two NaNs are considered
/// equal.
template <typename T, bool IsNanSensitive = true>
static bool isClose(T a, T b,
                    [[maybe_unused]] double absoluteTolerance = 1.0E-5,
                    [[maybe_unused]] double relativeTolerance = 1.0E-8) {
  static_assert(std::is_arithmetic_v<T>, "isClose requires an arithmetic type");
  if constexpr (std::is_integral_v<T>) {
    return a == b;
  } else {
    if (LLVM_UNLIKELY(!IsNanSensitive && std::isnan(a) && std::isnan(b)))
      return true;
    if (LLVM_UNLIKELY(std::isnan(a) || std::isnan(b)))
      return false;
    return std::fabs(a - b) <=
           std::max(static_cast<T>(relativeTolerance) *
                        std::max(std::fabs(a), std::fabs(b)),
                    static_cast<T>(absoluteTolerance));
  }
}

/// Computes the mean of the input array.
template <typename Range>
inline auto mean(const Range &values)
    -> std::remove_reference_t<decltype(*llvm::adl_begin(values))> {
  using value_type =
      std::remove_reference_t<decltype(*llvm::adl_begin(values))>;
  value_type init(0);
  auto begin = llvm::adl_begin(values);
  auto end = llvm::adl_end(values);
  size_t size = std::distance(begin, end);
  if (!size)
    return init;
  return std::accumulate(begin, end, init) / size;
}

/// Computes the trimmed mean of the sorted input array. The trimmed mean is a
/// method to remove outliers before computing the mean. The percentage of
/// outliers is determined by the `percentage` argument passed in. This function
/// assumes the input values are already sorted.
template <typename Range>
inline auto trimmedMean(const Range &values, double percent = 0.05)
    -> std::remove_reference_t<decltype(*llvm::adl_begin(values))> {
  assert(llvm::is_sorted(values) && "values are assumed to be sorted");
  assert(percent >= 0.0 && percent < 1.0 && "percent must be in [0, 1)");
  size_t size = std::size(values);
  if (size < 3)
    return mean(values);
  double k = size * percent / 2;
  return mean(llvm::make_range(
      std::next(llvm::adl_begin(values), static_cast<size_t>(std::lround(k))),
      std::prev(llvm::adl_end(values), static_cast<size_t>(std::round(k)))));
}

/// Computes the median of the input array assuming it is sorted.
template <typename Range>
inline auto median(const Range &values)
    -> std::remove_reference_t<decltype(*llvm::adl_begin(values))> {
  assert(llvm::is_sorted(values) && "values are assumed to be sorted");

  auto begin = llvm::adl_begin(values);
  auto end = llvm::adl_end(values);

  // Get the size of the container.
  auto size = std::distance(begin, end);

  // If the array is less than or equal to 2 elements, then the median is the
  // mean.
  if (size < 3)
    return mean(values);

  auto mid = size / 2;
  auto iter = begin;
  std::advance(iter, mid);
  auto midValue = *iter;
  // If the size is odd, the center is the median.
  if (size % 2 == 1)
    return midValue;
  // Otherwise, the average of the two elements in the center are the median.
  auto midValue2 = *std::prev(iter);
  return (midValue + midValue2) / 2;
}

/// Computes the percentile of the input array assuming it is sorted.
template <typename Range>
inline auto percentile(const Range &values, double percent)
    -> std::remove_reference_t<decltype(*llvm::adl_begin(values))> {
  assert(llvm::is_sorted(values) && "values are assumed to be sorted");
  assert(percent >= 0.0 && percent < 1.0 && "percentile must be in [0, 1)");

  using value_type =
      std::remove_reference_t<decltype(*llvm::adl_begin(values))>;

  auto begin = llvm::adl_begin(values);
  auto end = llvm::adl_end(values);
  auto size = std::distance(begin, end);
  if (size == 0)
    return value_type(0);
  return *std::next(llvm::adl_begin(values),
                    static_cast<size_t>(values.size() * percent));
}

/// Fill the buffer with random values from a distribution. Random engine seed
/// is always zero, so this function fills the buffer with "deterministic
/// random" values.
template <typename EltType, typename Distribution>
void fillWithRandomDistribution(MutableArrayRef<EltType> buffer,
                                Distribution distribution) {
  std::default_random_engine randEngine(/*seed=*/0);
  std::generate(buffer.begin(), buffer.end(),
                [&]() { return distribution(randEngine); });
}

/// Fill the provided buffer with random floating point values. This function
/// accepts a lower and upper bound on the random values.
template <typename EltType>
void fillWithRandomFloats(MutableArrayRef<EltType> buffer, EltType lb,
                          EltType ub) {
  // A micro-optimization to avoid calling the random number generator if the
  // bounds are equal.
  if (lb == ub) {
    if (lb == 0)
      memset(buffer.data(), 0, buffer.size() * sizeof(EltType));
    else
      std::fill(buffer.begin(), buffer.end(), lb);
    return;
  }
  fillWithRandomDistribution(buffer,
                             std::uniform_real_distribution<EltType>(lb, ub));
}

/// Fill the provided buffer with random floating point values. This function
/// accepts a lower and upper bound on the random values and supports arbitrary
/// floating point types, even those without native c++ types (e.g. FP24, BF16),
/// just use the proper semantic e.g. BFloat(), IEEESingle()...
template <typename storageType>
void fillWithRandomSpecialFloats(MutableArrayRef<storageType> buffer, float lb,
                                 float ub, const llvm::fltSemantics &semantic) {
  static_assert(std::is_integral_v<storageType> &&
                    std::is_unsigned_v<storageType>,
                "Please use uintXX_t as a storage type.");
  assert(llvm::APFloat::getSizeInBits(semantic) ==
             sizeof(storageType) * CHAR_BIT &&
         "Semantic and storage type bit width do not match up.");
  assert(llvm::APFloat::getSizeInBits(semantic) <= 32 &&
         "Backing number generator supports precision up to a "
         "IEEESingle (aka float32). Change it to double if need more bits.");

  std::uniform_real_distribution<float> distribution(lb, ub);
  fillWithRandomDistribution(
      buffer, [&](std::default_random_engine &engine) -> storageType {
        float result = distribution(engine);
        APFloat converter(result);

        bool losesInfo; // guaranteed to lose info for things less than float
        converter.convert(semantic, llvm::RoundingMode::NearestTiesToEven,
                          &losesInfo);
        storageType finalBits = converter.bitcastToAPInt().getZExtValue();
        return finalBits;
      });
}

/// Fill the provided buffer with random integer values. This function accepts a
/// lower and upper bound on the random values. The default DistributionT is
/// EltType, but since the MSVC does not define
/// std::uniform_int_distribution on character types, so use short or unsigned
/// short if the input is character width.
template <
    typename EltType,
    typename DistributionT = std::conditional_t<
        sizeof(EltType) == 1,
        std::conditional_t<std::is_unsigned_v<EltType>, unsigned short, short>,
        EltType>>
void fillWithRandomInts(MutableArrayRef<EltType> buffer, EltType lb,
                        EltType ub) {
  // A micro-optimization to avoid calling the random number generator if the
  // bounds are equal.
  if (lb == ub) {
    if (lb == 0)
      memset(buffer.data(), 0, buffer.size() * sizeof(EltType));
    else
      std::fill(buffer.begin(), buffer.end(), lb);
    return;
  }
  fillWithRandomDistribution(
      buffer, std::uniform_int_distribution<DistributionT>(lb, ub));
}

} // namespace M

#endif // SUPPORT_MATHEXTRAS_H
