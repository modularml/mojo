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
// Worker threads need their own alternate signal stack for a stack overflow in
// a work item to be reportable. See MOCO-4563.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "Support/Threading/SignalAltStack.h"
#include "llvm/Support/Compiler.h"
#include "gtest/gtest.h"
#include <chrono>
#include <thread>

using namespace M;
using namespace M::AsyncRT;

namespace {

CPUDeviceRef getTestCPUDevice() {
  CPUDeviceOptions options;
  options.numThreads = 2;
  options.mainWillDonate = false;
  return getOrCreateCPUDevice(CPUDeviceSource::Test, options);
}

TEST(WorkerSignalAltStack, WorkersHaveOne) {
  CPUDeviceRef cpuDevice = getTestCPUDevice();

  // The calling thread has an alternate stack of its own from LLVM's handler
  // registration, so the task has to report which thread it ran on for the
  // result to mean anything.
  struct Observation {
    std::thread::id threadId;
    bool hasAltStack;
  };
  AsyncValueRef<Observation> result =
      AsyncValueRef<Observation>::allocate(*cpuDevice);
  cpuDevice->getWorkQueue()->addTask(WorkItem([result = result.copy()]() {
    result.copy().emplace(
        Observation{std::this_thread::get_id(), hasSignalAltStack()});
  }));
  await(result);

  ASSERT_NE(result.get().threadId, std::this_thread::get_id());
  EXPECT_TRUE(result.get().hasAltStack);
}

// Volatile so the compiler cannot prove the recursion never returns, which
// keeps -Winfinite-recursion quiet.
volatile int keepRecursing = 1;

/// Exhausts the calling thread's stack. The volatile padding and its use after
/// the call keep this off the tail-call path.
LLVM_ATTRIBUTE_NOINLINE void overflowStack(int depth) {
  volatile char pad[4096];
  pad[0] = static_cast<char>(depth);
  if (keepRecursing)
    overflowStack(depth + 1);
  pad[sizeof(pad) - 1] = pad[0];
}

// What matters is the output, not the exit status: without an alternate stack
// the handler faults again and the process dies writing nothing. A symbolized
// frame is the property crash deduplication needs. The "Stack dump:" banner
// comes from PrettyStackTrace, which this binary's main does not install.
TEST(WorkerSignalAltStackDeathTest, StackOverflowOnAWorkerIsReported) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH_IF_SUPPORTED(
      {
        CPUDeviceRef cpuDevice = getTestCPUDevice();
        cpuDevice->getWorkQueue()->addTask(
            WorkItem([]() { overflowStack(0); }));

        // The worker takes the process down; this only bounds the wait so a
        // regression fails the test instead of hanging it.
        std::this_thread::sleep_for(std::chrono::seconds(10));
      },
      "overflowStack");
}

} // namespace
