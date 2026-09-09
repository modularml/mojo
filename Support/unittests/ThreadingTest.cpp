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

#include "Support/Threading/ThreadAffinity.h"

#include "Support/ErrorOr.h"
#include "Support/Threading/SignalAltStack.h"
#include "gtest/gtest.h"
#include <thread>

using namespace M;

TEST(Threading, haveThreadAffinity) {
#if defined(__linux__)
  // On linux, we expect haveThreadAffinity() to return True.
  // This may not be the case on every linux system in the world (presumably,
  // it's possible to build the kernel without affinity support), but all linux
  // systems of interest at this time should support this. If this test fails,
  // it could be a recurrence of the bug documented in GEX-3070.
  EXPECT_TRUE(haveThreadAffinity()) << "linux system without thread affinity?";
#endif

#if defined(__APPLE__)
  // On macos, we don't have/use thread affinity.
  EXPECT_FALSE(haveThreadAffinity());
#endif
}

TEST(Threading, signalAltStack) {
  if (!signalAltStackEnabled())
    GTEST_SKIP() << "No altstack in this build";

  // A freshly spawned thread has no alternate stack of its own. The test main
  // thread does, from LLVM's crash handler registration.
  bool preinstalled = false;
  std::thread([&]() {
    preinstalled = hasSignalAltStack();
    if (preinstalled)
      return;

    {
      ScopedSignalAltStack altStack;
      EXPECT_TRUE(hasSignalAltStack());

      // A nested guard must not disturb the outer one when it goes away.
      {
        ScopedSignalAltStack nested;
        EXPECT_TRUE(hasSignalAltStack());
      }
      EXPECT_TRUE(hasSignalAltStack());
    }
    EXPECT_FALSE(hasSignalAltStack());
  }).join();

  if (preinstalled)
    GTEST_SKIP() << "threads on this configuration already have one";
}
