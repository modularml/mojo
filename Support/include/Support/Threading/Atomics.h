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
// This file provides helper functions for working with std::atomic.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_THREADING_ATOMICS_H
#define SUPPORT_THREADING_ATOMICS_H

#include "Support/Threading/SpinWaiter.h"

#include <atomic>
#include <cstddef>
#include <type_traits>

namespace M {

/// Define `M::ardware_destructive_interference_size` in a portable way.
/// This is the alignment necessary to avoid false sharing between two atomic
/// operations.
#if defined(__cpp_lib_hardware_interference_size) && !defined(_MSC_VER)
using std::hardware_destructive_interference_size;
#else
static constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

/// This is a wrapper class for std::atomic which ensures that the atomic is at
/// the start of its own cache line as thus has no false sharing with other
/// atomics.  Note that this means that it is generally not space efficient.  If
/// you want things packed with it, put them after it (not before it) in a
/// struct.
template <typename ElementType>
class alignas(hardware_destructive_interference_size) AlignedAtomic
    : public std::atomic<ElementType> {
public:
  using std::atomic<ElementType>::atomic;
};

/// This method atomically adds 'accumValue' into 'accum'. This exists because
/// std::atomic doesn't provide a native add operation for floating point
/// values. atomicAdd only works on arithmetic values.
template <typename T>
static void atomicAdd(std::atomic<T> &accumValue, const T &value) {
  static_assert(std::is_arithmetic_v<T>,
                "the input type T must be an arithmetic type.");
  if constexpr (std::is_integral_v<T>) {
    // We use the native add operation for integral values.
    accumValue += value;
  } else {
    // No add operation exists in the C++17 standard so we perform the addition
    // via a compare_exchange_weak loop.
    T prevAccumValue = accumValue;

    SpinWaiter<> waiter;
    while (!accumValue.compare_exchange_weak(prevAccumValue,
                                             prevAccumValue + value)) {
      // Wait a bit and retry.
      waiter.wait();
    }
  }
}

/// This method atomically multiplies 'prodValue' into 'prod'. This exists
/// because std::atomic doesn't provide a native multiply operation. atomicMul
/// only works on arithmetic values.
template <typename T>
static void atomicMul(std::atomic<T> &prodValue, const T &prod) {
  static_assert(std::is_arithmetic_v<T>,
                "the input type T must be an arithmetic type.");
  // No multiplication operation exists in the C++17 standard so we perform the
  // addition via a compare_exchange_weak loop.
  T prevProdValue = prodValue;

  SpinWaiter<> waiter;
  while (
      !prodValue.compare_exchange_weak(prevProdValue, prevProdValue * prod)) {
    // Wait a bit and retry.
    waiter.wait();
  }
}

/// This method atomically updates 'maxValue' to 'value' if it is less than it
/// is already. This exists because std::atomic doesn't provide a native max
/// operation.
template <typename T>
static void atomicMax(std::atomic<T> &maxValue, const T &value) {
  T previousMax = maxValue;

  // Note that compare_exchange_weak updates `previousMax` on failure.
  SpinWaiter<> waiter;
  while (previousMax < value &&
         !maxValue.compare_exchange_weak(previousMax, value)) {
    // Wait a bit and retry.
    waiter.wait();
  }
}

/// This method atomically updates 'minValue' to 'value' if it is greater than
/// it is already. This exists because std::atomic doesn't provide a native min
/// operation.
template <typename T>
static void atomicMin(std::atomic<T> &minValue, const T &value) {
  T previousMin = minValue;

  // Note that compare_exchange_weak updates `previousMin` on failure.
  SpinWaiter<> waiter;
  while (previousMin > value &&
         !minValue.compare_exchange_weak(previousMin, value)) {
    // Wait a bit and retry.
    waiter.wait();
  }
}

} // namespace M

#endif // SUPPORT_THREADING_ATOMICS_H
