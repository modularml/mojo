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
// Implementation of DelegateThreadPoolWorkQueue, a WorkQueue that delegates to
// a set of partitioned WorkQueues (typically one per NUMA node), emulating a
// single unified queue.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/CompactCPUDevicePtr.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "AsyncRT/Support/Semaphore.h"

#include <atomic>
#include <cassert>
#include <vector>

using namespace M;
using namespace M::AsyncRT;

namespace {

/// Mapping for each global worker ID in the DelegateThreadPoolWorkQueue to a
/// partition ID and local worker ID within that partition.
struct WorkerIdMapping {
  size_t partitionIdx;
  size_t localWorkerId;
};

/// A WorkQueue that delegates to a set of partitioned WorkQueues (typically one
/// per NUMA node), emulating a single unified queue.
class DelegateThreadPoolWorkQueue : public WorkQueue {
public:
  DelegateThreadPoolWorkQueue(CompactCPUDevicePtr cpuDevicePtr,
                              std::vector<WorkQueue *> delegates);

  ~DelegateThreadPoolWorkQueue() override;

  void addTask(WorkItem &&work, int taskId = kDefaultTaskId) override;

  void addLocalTask(WorkItem &&workItem) override;

  void await(ArrayRef<AnyAsyncValueRef> values) override;

  // No-op, as delegate work queues are not owned.
  void shutdown() override {}

  /// Returns the total parallelism across all delegates.
  size_t getParallelismLevel() const override { return totalParallelism; }

  /// Returns the CPU IDs across all delegates.
  ArrayRef<size_t> getCpuIds() const override { return allCpuIds; }

  /// Always returns kAnyNumaNode as not associated with a specific NUMA node.
  int getNumaNode() const override { return kAnyNumaNode; }

  /// Uses the global worker ID from TLS for routing, checks if the current
  /// thread is associated with the task ID.
  bool shouldRunInlineForTask(int taskId) const override {
    if (taskId < 0)
      return true;
    size_t globalId = static_cast<size_t>(taskId);
    return globalId < workerMappings.size() &&
           getCurrentGlobalWorkerID() == globalId;
  }

private:
  std::vector<WorkQueue *> delegateWorkQueues;
  SmallVector<WorkerIdMapping> workerMappings;
  SmallVector<size_t> allCpuIds;
  size_t totalParallelism = 0;
  std::atomic<size_t> roundRobinIdx{0};
};

DelegateThreadPoolWorkQueue::DelegateThreadPoolWorkQueue(
    CompactCPUDevicePtr cpuDevicePtr, std::vector<WorkQueue *> delegates)
    : delegateWorkQueues(std::move(delegates)) {
  // Initialise the CPU ids, total parallelism and worker mappings for the
  // delegates.
  for (size_t i = 0; i < delegateWorkQueues.size(); ++i) {
    WorkQueue &q = *delegateWorkQueues[i];
    size_t n = q.getParallelismLevel();
    for (size_t localId = 0; localId < n; ++localId)
      workerMappings.push_back({i, localId});
    totalParallelism += n;
    for (size_t cpuId : q.getCpuIds())
      allCpuIds.push_back(cpuId);
  }

  // Set the calling thread's TLS to the top-level global device.
  CompactCPUDevicePtr::setCurrentCPUDevice(cpuDevicePtr);
}

DelegateThreadPoolWorkQueue::~DelegateThreadPoolWorkQueue() {
  // Clear the calling thread's TLS pointer.
  CompactCPUDevicePtr::setCurrentCPUDevice({});
}

void DelegateThreadPoolWorkQueue::addTask(WorkItem &&work, int taskId) {
  // Default global scheduling is to round-robin across partitions. Tasks stay
  // NUMA-local and don't cross partition boundaries. Note this means an idle
  // partition won't steal from a busy one.
  if (taskId == kDefaultTaskId) {
    size_t idx = roundRobinIdx.fetch_add(1, std::memory_order_relaxed) %
                 delegateWorkQueues.size();
    delegateWorkQueues[idx]->addTask(std::move(work), kDefaultTaskId);
    return;
  }

  // For specific task IDs uses the global worker ID from TLS for routing, looks
  // up the worker thread mapping and redirects to the appropriate delegate.
  size_t globalId = static_cast<size_t>(taskId);
  assert(globalId < workerMappings.size() && "taskId out of range");
  const WorkerIdMapping &m = workerMappings[globalId];
  delegateWorkQueues[m.partitionIdx]->addTask(
      std::move(work), static_cast<int>(m.localWorkerId));
}

void DelegateThreadPoolWorkQueue::addLocalTask(WorkItem &&workItem) {
  size_t globalId = getCurrentGlobalWorkerID();
  if (globalId == SIZE_MAX) {
    // For main or foreign threads treat as a global task with round-robin
    // routing.
    addTask(std::move(workItem), kDefaultTaskId);
    return;
  }
  // Uses the global worker ID from TLS for routing, looks up the worker thread
  // mapping and redirects to the appropriate delegate.
  assert(globalId < workerMappings.size() && "global worker ID out of range");
  delegateWorkQueues[workerMappings[globalId].partitionIdx]->addLocalTask(
      std::move(workItem));
}

void DelegateThreadPoolWorkQueue::await(ArrayRef<AnyAsyncValueRef> values) {
  size_t globalId = getCurrentGlobalWorkerID();
  if (globalId != SIZE_MAX && globalId < workerMappings.size()) {
    delegateWorkQueues[workerMappings[globalId].partitionIdx]->await(values);
    return;
  }
  // For main or foreign thread register semaphore waiters and sleep until all
  // values are ready.
  if (llvm::all_of(values, [](const auto &av) { return av.isReady(); }))
    return;
  std::atomic<ssize_t> numRemaining = static_cast<ssize_t>(values.size());
  Semaphore sema;
  for (auto &value : values) {
    value.andThenSync([&numRemaining, &sema]() {
      if (numRemaining.fetch_sub(1, std::memory_order_seq_cst) != 1)
        return;
      sema.post();
    });
  }
  sema.wait();
}

} // namespace

std::unique_ptr<WorkQueue> M::AsyncRT::createDelegateThreadPoolWorkQueue(
    CompactCPUDevicePtr cpuDevicePtr,
    std::vector<WorkQueue *> delegateWorkQueues) {
  assert(!delegateWorkQueues.empty() &&
         "DelegateThreadPoolWorkQueue requires at least one delegate");
  return std::make_unique<DelegateThreadPoolWorkQueue>(
      cpuDevicePtr, std::move(delegateWorkQueues));
}
