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

#include "AsyncRT/Support/Semaphore.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#else
#include <cassert>
#include <cerrno>
#include <semaphore.h>
#endif

using namespace M::AsyncRT;

/// This class provides the implementation for the Semaphore object. Because we
/// have so many different implementation details, we encapsulate the
/// platform-specific details into this pImpl class.
class Semaphore::Impl {
public:
  /// Manage semaphore lifetime. In cases where this wraps other APIs, this
  /// should be used to (for example) call sem_destroy.
  Impl(ssize_t initialValue);
  ~Impl();

  /// Increment the semaphore.
  void post();

  bool wait();
  bool wait(int64_t timeoutNS);

private:
#if defined(__APPLE__)
  dispatch_semaphore_t sema;
#else
  sem_t sema;
#endif
};

Semaphore::Semaphore(Semaphore &&other) = default;

//===----------------------------------------------------------------------===//
// Semaphore::Impl function implementations
//===----------------------------------------------------------------------===//

#if defined(__APPLE__)
//===----------------------------------------------------------------------===//
// Semaphore::Impl for Apple platforms
//===----------------------------------------------------------------------===//

Semaphore::Impl::Impl(ssize_t initialValue)
    : sema(dispatch_semaphore_create(initialValue)) {
  assert(initialValue >= 0 && "semaphore initial value cannot be negative");
}
Semaphore::Impl::~Impl() { dispatch_release(sema); }
void Semaphore::Impl::post() { dispatch_semaphore_signal(sema); }

bool Semaphore::Impl::wait() {
  return 0 != dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
}

bool Semaphore::Impl::wait(int64_t timeoutNS) {
  dispatch_time_t timeout =
      dispatch_time(DISPATCH_TIME_NOW, /*nsecToAdd*/ timeoutNS);
  return 0 != dispatch_semaphore_wait(sema, timeout);
}

#else
//===----------------------------------------------------------------------===//
// Semaphore::Impl for POSIX platforms with sem_timedwait
//===----------------------------------------------------------------------===//

Semaphore::Impl::Impl(ssize_t initialValue) {
  assert(initialValue >= 0 && "semaphore initial value cannot be negative");
  if (-1 == sem_init(&sema, 0, initialValue))
    llvm::report_fatal_error("Unable to initialize an unnamed semaphore.");
}

Semaphore::Impl::~Impl() {
  [[maybe_unused]] int rc = sem_destroy(&sema);
  assert(rc == 0 && "Unable to destroy the unnamed semaphore.");
}

void Semaphore::Impl::post() { sem_post(&sema); }

bool Semaphore::Impl::wait() {
  int rc;
  // If we have no timeout, then we just have check for having been interrupted
  // by a signal handler.
  while ((rc = sem_wait(&sema)) == -1 && errno == EINTR)
    continue;

  // If sem_wait returned 0 then we're good, we acquired the semaphore.
  // Otherwise, we hit an error and were unable to acquire the semaphore.
  return rc != 0;
}

bool Semaphore::Impl::wait(int64_t timeoutNS) {
  // Get the current time - the timeout on sem_timedwait is an absolute timeout
  // since the epoch.
  struct timespec ts;
  if (-1 == clock_gettime(CLOCK_REALTIME, &ts))
    llvm::report_fatal_error("Unable to call clock_gettime");

  ts.tv_nsec += timeoutNS;
  // The semaphore may be interrupted by a signal handler, so check for this
  // case and continue if that is what happens.
  int rc;
  while ((rc = sem_timedwait(&sema, &ts)) == -1 && errno == EINTR)
    continue;

  // Semaphore successfully decremented, return no error.
  if (rc == 0)
    return false;

  // Timeout occurred.
  if (rc == -1 && errno == ETIMEDOUT)
    return true;

  llvm::report_fatal_error(
      "sem_timedwait failed for a reason other than EINTR or ETIMEDOUT.");
}
#endif

//===----------------------------------------------------------------------===//
// Semaphore function implementations (just forward to Semaphore::Impl)
//===----------------------------------------------------------------------===//

Semaphore::Semaphore(ssize_t initialValue)
    : impl(std::make_unique<Semaphore::Impl>(initialValue)) {}

// Empty destructor needed here so we can forward declare Semaphore::Impl into
// the header.
Semaphore::~Semaphore() = default;

void Semaphore::post() { impl->post(); }

bool Semaphore::wait() { return impl->wait(); }

bool Semaphore::wait(int64_t timeoutNS) { return impl->wait(timeoutNS); }
