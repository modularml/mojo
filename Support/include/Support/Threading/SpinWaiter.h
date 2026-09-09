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

#ifndef SUPPORT_THREADING_SPINWAITER_H
#define SUPPORT_THREADING_SPINWAITER_H

#include "Support/PlatformUtils.h"

#include <chrono>
#include <cstddef>
#include <optional>

#ifdef _MSC_VER
#include <immintrin.h> // _mm_pause
#endif

namespace M {

namespace Detail {
// This is the non-templated base class of SpinWaiter.
class SpinWaiterBase {
protected:
  enum {
    // This is the number of times we will spin without doing any system
    // operations.
    rawSpins = 4,
    // This is the number of times we spin with pause, no-op, or equivalent
    // instructions.
    nopSpins = 64,
    // If that doesn't work we yield the thread back to the OS.
    yieldSpins = 128,
    // If that doesn't work we sleep the thread.
  };

  bool yieldToOS();
  size_t iterations = 0;
};
} // namespace Detail

/// This class is used in busy-wait loops to provide exponential backoff and
/// to defer to the OS under long waits.  This helps improve situations with
/// high contention, by allowing the thread we're waiting for to have proper
/// access to the memory hierarchy and CPU cores needed to make forward
/// progress.
///
/// This is "free" to initialize in cases where it isn't used, just setting a
/// non-atomic integer to zero.
template <bool shouldYieldToOS = true>
class SpinWaiter : public Detail::SpinWaiterBase {
public:
  SpinWaiter() = default;

  /// This method is called by spinning algorithms that realize they need to try
  /// again.  This returns false when a non-appreciable amount of time has
  /// elapsed (which happens in the first few iterations), and true if this
  /// waited for a longer time.
  bool wait() {
    // Directly spin a few times.
    if (++iterations < rawSpins)
      return false;

    // If a direct spin didn't resolve the issue, do a more serious SMT-aware
    // pause if we know of one.
    if (iterations < nopSpins ||
        // The client can disable the more expensive yielding mechanisms below
        // by setting "shouldYieldToOS" to true.
        !shouldYieldToOS) {
#if MODULAR_WINDOWS && MODULAR_X86_64
      _mm_pause();
#elif MODULAR_X86_64
      __builtin_ia32_pause();
#elif MODULAR_ARM
      // The isb instruction is the closest to the original x86 pause
      // instruction. Unlike the x86 pause instruction which delays execution by
      // O(100) cycles, the isb will typically delay execution by about 50
      // cycles.
      __asm__ volatile("isb" ::: "memory");
#else
      // Hail mary to slow this thread down so other threads can make progress
      // without us fully occupying the load/store unit.
      __asm volatile("nop; nop; nop; nop" : : : "memory");
#error "Unexpected architecture for SpinWaiter"
#endif
      return true;
    }

    // Ok, we're going to yield to the OS, call our out-of-line implementation
    // of this.
    return yieldToOS();
  }

  /// Return true if this waiter is going to do heavy weight OS operations to
  /// slow the current thread's progress.
  bool isDoneWithNopSpins() const { return iterations >= nopSpins; }
};

/// This is like SpinWaiter<> but allows configurable busy waiting based on wall
/// time.
class BusyWaitSpinWaiter {
public:
  BusyWaitSpinWaiter(std::chrono::nanoseconds busyWaitTime)
      : busyWaitTime(busyWaitTime) {}

  /// Wait for another step using progressively more heavy-weight mechanisms.
  /// This returns true if we should block on a semaphore.
  bool wait() {
    // If we are cheap-waiting, just return quickly.
    if (!waiter.isDoneWithNopSpins()) {
      waiter.wait();
      return false;
    }

    // Otherwise we're going to intentionally burn time.  If this is the first
    // iteration of this, figure out what wall time we are.
    if (busyWaitTime == std::chrono::nanoseconds::zero())
      return true;

    if (!busyWaitEndTime.has_value())
      busyWaitEndTime =
          std::chrono::high_resolution_clock::now() + busyWaitTime;

    // When we reach the busy wait end time, return true so the caller can block
    // on a semaphore.
    return std::chrono::high_resolution_clock::now() >= *busyWaitEndTime;
  }

private:
  /// This is a spin waiter that never yields to the OS with sched_yield etc.
  /// We would rather block on the semaphore.
  SpinWaiter<false> waiter;

  /// This is how long to spin on the waiter.
  /// TODO: This should eventually go away or turn into a constant.
  const std::chrono::nanoseconds busyWaitTime;

  /// This is the time we should stop busy waiting.
  std::optional<std::chrono::high_resolution_clock::time_point> busyWaitEndTime;
};

} // namespace M

#endif // SUPPORT_THREADING_SPINWAITER_H
