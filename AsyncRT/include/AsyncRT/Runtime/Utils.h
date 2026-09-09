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

#ifndef ASYNCRT_RUNTIME_UTILS_H
#define ASYNCRT_RUNTIME_UTILS_H

#include "AsyncRT/Runtime/AsyncValue.h"
#include "AsyncRT/Runtime/RuntimeCLOptions.h"

#include <cstddef>

namespace M::AsyncRT {

/// Run a lambda or other callable with a new Runtime instance configured
/// according to the command line argument specification.  Encircle this with
/// a AsyncValue leak checker to catch simple bugs in the test suite.
template <typename BodyFn>
auto runWithLeakCheckedRuntime(const char *testName, BodyFn bodyFn) {
  // If we are leak checking, remember how many AsyncValue's we started with.
  ssize_t numStartingLiveAsyncValues = 0;
  if constexpr (AsyncValue::isAllocationTrackingEnabled())
    numStartingLiveAsyncValues = AsyncValue::getNumAllocatedInstances();

  // Check leak status on exit from scope.
  struct LeakChecker {
    const char *testName;
    ssize_t numStartingLiveAsyncValues;

    ~LeakChecker() { // Make sure we're not leaking AsyncValues.
      if constexpr (AsyncValue::isAllocationTrackingEnabled()) {
        ssize_t numLiveAsyncValues = AsyncValue::getNumAllocatedInstances();
        if (numLiveAsyncValues != numStartingLiveAsyncValues) {
          fprintf(stderr,
                  "Evaluation of testcase '%s' leaked %d async values (before: "
                  "%d, after: %d)!\n",
                  testName,
                  int(numLiveAsyncValues - numStartingLiveAsyncValues),
                  int(numStartingLiveAsyncValues), int(numLiveAsyncValues));
          abort();
        }
      }
    }
  } checker{testName, numStartingLiveAsyncValues};

  // Execute the body with a new runtime.
  return bodyFn();
}

} // namespace M::AsyncRT

#endif // ASYNCRT_RUNTIME_UTILS_H
