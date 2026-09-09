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

#ifndef ASYNCRT_SUPPORT_SEMAPHORE_H
#define ASYNCRT_SUPPORT_SEMAPHORE_H

#include <memory>
#if defined(_WIN64) || defined(_WIN32)
#include <BaseTsd.h>
using ssize_t = SSIZE_T;
#else // defined(_WIN64) || defined(_WIN32)
#include <sys/types.h>
#endif // defined(_WIN64) || defined(_WIN32)

namespace M::AsyncRT {
/// This is an interface to a basic semaphore with post and timed wait
/// functionality. This is essentially a lowest-common-denominator interface
/// that is meant to be able to be backed by a GCD semaphore, or a POSIX
/// semaphore, or in the worst case a counter protected by a mutex.
class Semaphore {
public:
  /// Initialize a semaphore, the initial value is the starting count for the
  /// value, it may not be negative because that would indicate that there are
  /// blocked threads already.
  Semaphore(ssize_t initialValue = 0);
  Semaphore(Semaphore &&);
  ~Semaphore();

  /// Increments the semaphore.  If it was previously negative, then this will
  /// wake up one of the waiters.
  void post();

  /// Decrement the semaphore: if the resulting value is less than zero, wait
  /// for someone to signal the semaphore.
  bool wait();

  /// Decrement the semaphore: if the resulting value is less than zero, wait
  /// for someone to signal the semaphore.  This wait will eventually time out
  /// in approximately `timeoutNS` nanoseconds.
  ///
  /// Using this is extremely unwise because you've built a polling algorithm
  /// that will burn power when something is supposed to be quiesced.
  ///
  /// Timeouts are generally not the answer!
  bool wait(int64_t timeoutNS);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace M::AsyncRT

#endif // ASYNCRT_SUPPORT_SEMAPHORE_H
