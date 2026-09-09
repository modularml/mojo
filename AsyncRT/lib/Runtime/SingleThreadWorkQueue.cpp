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

#include "AsyncRT/Runtime/WorkQueue.h"

#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Support/ConcurrentQueue.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"

#include <thread>

#define DEBUG_TYPE "asyncrt"

using namespace M;
using namespace AsyncRT;

namespace {

/// This class implements a work queue that uses only the caller's thread to
/// execute work. It spawns no additional threads. However, the work queue
/// itself is thread safe, and addTask and await may be called from any
/// threads.
class SingleThreadWorkQueue : public WorkQueue {
public:
  SingleThreadWorkQueue(CompactCPUDevicePtr cpuDevicePtr) {
    CompactCPUDevicePtr::setCurrentCPUDevice(cpuDevicePtr);
  }

  void shutdown() override;

  ~SingleThreadWorkQueue() override {
    // Note we can't assert state == kShutdown since queue may be created
    // and destroyed without ever being included in a cpuDevice.
    assert(!workItems.dequeue());

    // Clear thread-local Runtime pointer.
    CompactCPUDevicePtr::setCurrentCPUDevice({});
  }

  void addTask(WorkItem &&workItem, int taskId = -1) override {
    assert(workItem);
    workItems.enqueue(std::move(workItem));
  }

  void addLocalTask(WorkItem &&workItem) override {
    addTask(std::move(workItem));
  }

  void await(llvm::ArrayRef<AnyAsyncValueRef> values) override;

  size_t getParallelismLevel() const override { return 1; }

private:
  /// Execute blocks of work until stopPredicate is true.
  template <typename StopPredicateFn>
  void runUntil(StopPredicateFn stopPredicate);

  // Execute a single profiled work item.
  void doWork(WorkItem &&workItem) {
    // Do the work.
    {
      TimeTraceScope scope(AllWorkItemsProfilerEntry::create("asyncrt.doWork"));
      workItem.task();
    }
  }

  /// Pending work items.
  ConcurrentQueue<WorkItem> workItems;
};
} // namespace

void SingleThreadWorkQueue::shutdown() {
  // Complete any work that's still in-flight.
  while (auto workItem = workItems.dequeue()) {
    doWork(std::move(workItem));
  }
}

void SingleThreadWorkQueue::await(llvm::ArrayRef<AnyAsyncValueRef> values) {
  // We are done when values_remaining drops to zero.
  std::atomic<size_t> numRemaining = values.size();

  // As each value becomes available, we can decrement our counts.
  for (auto &value : values)
    value.andThenSync([&numRemaining]() { numRemaining.fetch_sub(1); });

  if (numRemaining.load() == 0)
    return;

  // Run work items until numRemaining drops to zero.
  runUntil([&]() -> bool { return numRemaining.load() == 0; });

  assert(numRemaining.load() == 0 &&
         "Some AsyncValues are not ready yet no further "
         "tasks are available to run. Are all input AsyncValues ready?");
}

/// Time to sleep while waiting for work in the work queue.
static const std::chrono::microseconds sleepTime(100);

template <typename StopPredicateFn>
void SingleThreadWorkQueue::runUntil(StopPredicateFn stopPredicate) {
  std::chrono::microseconds totalSlept(0);
  while (true) {
    totalSlept += sleepTime;
    assert(
        totalSlept < std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::seconds(5)) &&
        "SingleThreadWorkQueue has slept for more than 5 seconds while "
        "waiting for callbacks. Some AsyncValues are not ready yet no further "
        "tasks are available to run. Are all input AsyncValues ready?");

    while (auto workItem = workItems.dequeue()) {
      totalSlept = std::chrono::microseconds(0);
      doWork(std::move(workItem));
      if (stopPredicate())
        return;
    }
    // If no work was done, still check if we are done.
    if (stopPredicate())
      return;

    // wait for any callbacks to fire
    std::this_thread::sleep_for(sleepTime);
  }
}

std::unique_ptr<WorkQueue>
M::AsyncRT::createSingleThreadWorkQueue(CompactCPUDevicePtr cpuDevicePtr) {
  return std::make_unique<SingleThreadWorkQueue>(cpuDevicePtr);
}
