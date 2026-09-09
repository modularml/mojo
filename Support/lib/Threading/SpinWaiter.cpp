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

#include "Support/Threading/SpinWaiter.h"

#include <chrono>
#include <thread>

using namespace M;

bool Detail::SpinWaiterBase::yieldToOS() {
  // If that didn't work, we yield the thread back to the OS.  This is much
  // heavier weight but can cause the OS to reschedule the problematic thread.
  if (iterations < yieldSpins) {
    std::this_thread::yield();
    return true;
  }

  // Otherwise, we're in pretty serious trouble, actually go to sleep for
  // longer times.
  std::this_thread::sleep_for(std::chrono::microseconds(iterations / 128));
  iterations += 128;
  return true;
}
