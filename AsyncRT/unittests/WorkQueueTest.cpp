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
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "Support/Threading/HWInfo.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <atomic>
#include <climits>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

using namespace M;
using namespace M::AsyncRT;

namespace {

// Enqueue a worker-task that records the result of `shouldRunInlineForTask` for
// a given `checkTaskId`.
static AsyncValueRef<bool> enqueueInlineCheck(CPUDevice &cpuDevice,
                                              WorkQueue &workQueue,
                                              int dispatchTaskId,
                                              int checkTaskId) {
  AsyncValueRef<bool> result = AsyncValueRef<bool>::allocate(cpuDevice);
  WorkItem probe([&workQueue, checkTaskId, ready = result.copy()]() mutable {
    ready.copy().emplace(workQueue.shouldRunInlineForTask(checkTaskId));
  });
  workQueue.addTask(std::move(probe), dispatchTaskId);
  return result;
}

/// Test task-based scheduling with taskId affinity.
/// With conservative worker 0 avoidance:
/// - 4 workers (0, 1, 2, 3), but we use workers 1, 2, 3 for affinity tasks
/// - taskId = 1 + (hint % 3) for non-negative hints
TEST(WorkQueueTest, TaskIdRouting) {
  CPUDeviceOptions options;
  options.numThreads = 4;
  options.mainWillDonate = false;
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);
  WorkQueue *workQueue = cpuDevice->getWorkQueue();

  // Use 3 different taskIds (1, 2, 3) since worker 0 is avoided
  constexpr int numTaskIds = 3;
  constexpr int numTasksPerTaskId = 10;

  // Track which thread executed each task and per-taskId thread mapping.
  std::unordered_map<std::thread::id, int> threadTaskCounts;
  std::unordered_map<int, std::vector<std::thread::id>> perTaskIdThreads;
  std::mutex mapMutex;

  // Create work items with different task IDs (1, 2, 3).
  std::vector<AsyncValueRef<int>> results;
  results.reserve(numTaskIds * numTasksPerTaskId);
  for (int taskId = 1; taskId <= numTaskIds; ++taskId) {
    for (int task = 0; task < numTasksPerTaskId; ++task) {
      results.emplace_back(AsyncValueRef<int>::allocate(*cpuDevice));
      AsyncValueRef<int> &result = results.back();

      WorkItem workItem([taskId, task, result = result.copy(),
                         &threadTaskCounts, &perTaskIdThreads, &mapMutex]() {
        // Record which thread is executing this task and per-taskId mapping.
        std::thread::id threadId = std::this_thread::get_id();
        {
          std::lock_guard<std::mutex> lock(mapMutex);
          threadTaskCounts[threadId] += 1;
          perTaskIdThreads[taskId].push_back(threadId);
        }
        result.copy().emplace(taskId * 100 + task);
      });

      workQueue->addTask(std::move(workItem), taskId);
    }
  }

  // Wait for all tasks to complete.
  for (AsyncValueRef<int> &result : results)
    await(result);

  // Verify all tasks completed
  for (size_t i = 0; i < results.size(); ++i) {
    int taskId = 1 + static_cast<int>(i / numTasksPerTaskId);
    int task = i % numTasksPerTaskId;
    EXPECT_EQ(results[i].get(), taskId * 100 + task);
  }

  // Stronger routing checks:
  // 1) All tasks for a given taskId run on the same thread.
  for (int tid = 1; tid <= numTaskIds; ++tid) {
    ASSERT_EQ(perTaskIdThreads.count(tid), 1u)
        << "Missing records for taskId " << tid;
    const std::vector<std::thread::id> &v = perTaskIdThreads[tid];
    ASSERT_FALSE(v.empty());
    const std::thread::id first = v.front();
    for (const std::thread::id &threadId : v)
      EXPECT_EQ(threadId, first)
          << "TaskId " << tid << " ran on multiple threads";
  }

  // 2) TaskIds 1, 2, 3 run on distinct threads.
  std::set<std::thread::id> workerThreads;
  for (int tid = 1; tid <= numTaskIds; ++tid)
    workerThreads.insert(perTaskIdThreads[tid].front());
  EXPECT_EQ(workerThreads.size(), static_cast<size_t>(numTaskIds))
      << "TaskIds 1-3 should map to three distinct workers";
}

/// Test that negative taskIds are handled correctly (global queue).
TEST(WorkQueueTest, NegativeTaskId) {
  CPUDeviceOptions options;
  options.numThreads = 4;
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);
  WorkQueue *workQueue = cpuDevice->getWorkQueue();

  auto result = AsyncValueRef<int>::allocate(*cpuDevice);

  WorkItem workItem([result = result.copy()]() { result.copy().emplace(42); });

  // Negative taskId should use global queue
  workQueue->addTask(std::move(workItem), -5);

  await(result);
  EXPECT_EQ(result.get(), 42);
}

/// Test taskId with mainWillDonate mode.
/// Since we conservatively skip worker 0, all tasks should complete
/// without needing await on main thread.
TEST(WorkQueueTest, TaskIdWithMainWillDonate) {
  CPUDeviceOptions options;
  options.numThreads = 4;
  options.mainWillDonate = true;
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);
  WorkQueue *workQueue = cpuDevice->getWorkQueue();

  constexpr int numTasks = 20;
  std::vector<AsyncValueRef<int>> results;
  results.reserve(numTasks);

  // Create tasks with various taskIds (all avoid worker 0)
  for (int i = 0; i < numTasks; ++i) {
    results.emplace_back(AsyncValueRef<int>::allocate(*cpuDevice));
    AsyncValueRef<int> &result = results.back();

    WorkItem workItem([i, result = result.copy()]() {
      // Small delay to simulate work
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      result.copy().emplace(i);
    });

    // Use taskIds 1, 2, 3 (avoiding worker 0)
    int taskId = 1 + (i % 3);
    workQueue->addTask(std::move(workItem), taskId);
  }

  // Tasks should complete without needing await on main thread
  std::promise<void> done;
  std::future<void> fut = done.get_future();
  std::thread checker([&results, d = std::move(done)]() mutable {
    for (AsyncValueRef<int> &result : results)
      await(result);
    d.set_value();
  });

  // Bound the wait to avoid hangs on regression.
  std::future_status status = fut.wait_for(std::chrono::seconds(3));
  EXPECT_EQ(status, std::future_status::ready)
      << "Tasks should complete without main-thread await";

  if (status == std::future_status::ready) {
    checker.join();
  } else {
    // Avoid blocking the test thread; failure already recorded.
    checker.detach();
  }

  // Verify all tasks completed correctly.
  if (status == std::future_status::ready) {
    for (int i = 0; i < numTasks; ++i)
      EXPECT_EQ(results[i].get(), i);
  }
}

/// Test that kDefaultTaskId uses global queue scheduling.
TEST(WorkQueueTest, DefaultTaskId) {
  CPUDeviceOptions options;
  options.numThreads = 4;
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);
  WorkQueue *workQueue = cpuDevice->getWorkQueue();

  auto result = AsyncValueRef<int>::allocate(*cpuDevice);

  WorkItem workItem([result = result.copy()]() { result.copy().emplace(99); });

  // Should use default (global queue) scheduling.
  workQueue->addTask(std::move(workItem), kDefaultTaskId);

  await(result);
  EXPECT_EQ(result.get(), 99);
}

TEST(WorkQueueTest, ShouldRunInlineMatchesAssignedWorker) {
  CPUDeviceOptions options;
  options.numThreads = 4;
  options.mainWillDonate = false;
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);
  WorkQueue &workQueue = *cpuDevice->getWorkQueue();

  // Each worker executes a task pinned to its taskId and reports whether the
  // inline heuristic agrees; tasks dispatched and checked with the same taskId
  // should all come back `true`.
  std::vector<AsyncValueRef<bool>> inlineResults;
  inlineResults.reserve(3);
  // Test with taskIds 1, 2, 3.
  for (int taskId = 1; taskId <= 3; ++taskId) {
    inlineResults.emplace_back(
        enqueueInlineCheck(*cpuDevice, workQueue, taskId, taskId));
  }

  for (size_t idx = 0; idx < inlineResults.size(); ++idx) {
    await(inlineResults[idx]);
    EXPECT_TRUE(inlineResults[idx].get());
  }

  // When a worker is pinned to taskId 1 but queries taskId 2, it should decline
  // to inline because it is running on the wrong worker thread.
  AsyncValueRef<bool> mismatch =
      enqueueInlineCheck(*cpuDevice, workQueue, /*dispatchTaskId=*/1,
                         /*checkTaskId=*/2);
  await(mismatch);
  EXPECT_FALSE(mismatch.get());
}

TEST(WorkQueueTest, ShouldRunInlineHonorsWorkerAffinity) {
  CPUDeviceOptions options;
  options.numThreads = 3;
  options.mainWillDonate = true;
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);
  WorkQueue &workQueue = *cpuDevice->getWorkQueue();

  // Enqueue tasks to workers 1 and 2.
  // The worker that actually executes the task should still report inline =
  // true.
  std::vector<AsyncValueRef<bool>> results;
  for (int taskId = 1; taskId <= 2; ++taskId) {
    results.emplace_back(
        enqueueInlineCheck(*cpuDevice, workQueue, taskId, taskId));
  }

  for (AsyncValueRef<bool> &ready : results) {
    await(ready);
    EXPECT_TRUE(ready.get());
  }
}

TEST(WorkQueueTest, ShouldRunInlineFromForeignThread) {
  CPUDeviceOptions options;
  options.numThreads = 2;
  options.mainWillDonate = false; // Ensure main thread is "foreign"
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);
  WorkQueue &workQueue = *cpuDevice->getWorkQueue();

  // From a foreign thread (main thread not donating), we should inline only
  // for default/negative taskId; positive taskIds belong to workers and should
  // return false here since we're not on any worker.
  EXPECT_TRUE(workQueue.shouldRunInlineForTask(kDefaultTaskId));
  EXPECT_TRUE(workQueue.shouldRunInlineForTask(-5));
  EXPECT_FALSE(workQueue.shouldRunInlineForTask(0));
  EXPECT_FALSE(workQueue.shouldRunInlineForTask(1));
}

TEST(WorkQueueTest, ShouldRunInlineSingleThreadedAlwaysTrue) {
  CPUDeviceOptions options;
  options.withSingleThreaded();
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);
  WorkQueue &workQueue = *cpuDevice->getWorkQueue();

  // The single worker should inline regardless of the taskId value.
  AsyncValueRef<bool> inlineCheck =
      enqueueInlineCheck(*cpuDevice, workQueue, /*dispatchTaskId=*/0,
                         /*checkTaskId=*/3);
  await(inlineCheck);
  EXPECT_TRUE(inlineCheck.get());

  EXPECT_TRUE(workQueue.shouldRunInlineForTask(kDefaultTaskId));
}

//===----------------------------------------------------------------------===//
// Partitioned (NUMA-aware) WorkQueue tests
//===----------------------------------------------------------------------===//

namespace {

/// Returns a NUMA node id with a non-empty CPU list, or an empty optional if
/// the host has no usable NUMA topology (e.g. running in a container that
/// doesn't expose /sys/devices/system/node/).
std::optional<int> firstUsableNumaNode() {
  const ErrorOr<NUMATopology> &topologyOr = NUMATopology::get();
  if (topologyOr.isError())
    return std::nullopt;
  for (int node : topologyOr->getNumaNodes()) {
    if (!topologyOr->getCpuIdsForNumaNode(node).empty())
      return node;
  }
  return std::nullopt;
}

/// Constructs a Runtime and returns its CompactCPUDevicePtr for use with the
/// partitioned WorkQueue factory. Holds the CPUDeviceRef alive for the caller.
struct PartitionedHarness {
  CPUDeviceRef cpuDevice;
  CompactCPUDevicePtr cpuDevicePtr;

  std::vector<std::unique_ptr<WorkQueue>> ownedPartitions;
  std::unique_ptr<WorkQueue> workQueue;

  ~PartitionedHarness() {
    if (workQueue)
      workQueue->shutdown();
    for (auto &partition : ownedPartitions)
      if (partition)
        partition->shutdown();
  }
};

PartitionedHarness makeHarness(int numaNode) {
  CPUDeviceOptions options;
  options.withSingleThreaded(); // Minimal "host" cpuDevice just to supply ptr.
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);
  CompactCPUDevicePtr ptr = cpuDevice->getCompactPtr();
  auto partWQ = createPartitionedThreadPoolWorkQueue(
      ptr, numaNode, std::chrono::microseconds(200), "NumaTest",
      /*globalWorkerIdOffset=*/0);
  std::vector<WorkQueue *> rawPtrs = {partWQ.get()};
  std::vector<std::unique_ptr<WorkQueue>> owned;
  owned.push_back(std::move(partWQ));
  return {std::move(cpuDevice), ptr, std::move(owned),
          createDelegateThreadPoolWorkQueue(ptr, std::move(rawPtrs))};
}

} // namespace

TEST(WorkQueueTest, PartitionedReportsPartition) {
  std::optional<int> nodeOr = firstUsableNumaNode();
  if (!nodeOr)
    GTEST_SKIP() << "host has no NUMA topology available";
  int node = *nodeOr;

  PartitionedHarness h = makeHarness(node);
  // The delegate wrapping the partition reports kAnyNumaNode; the partition's
  // CPU set is still surfaced through getCpuIds().
  EXPECT_EQ(h.workQueue->getNumaNode(), kAnyNumaNode);
  std::vector<size_t> expected =
      NUMATopology::get()->getCpuIdsForNumaNode(node);
  ArrayRef<size_t> reported = h.workQueue->getCpuIds();
  EXPECT_EQ(reported.size(), expected.size());
  EXPECT_TRUE(std::equal(reported.begin(), reported.end(), expected.begin()));
}

TEST(WorkQueueTest, PartitionedParallelismMatchesCpuIds) {
  std::optional<int> nodeOr = firstUsableNumaNode();
  if (!nodeOr)
    GTEST_SKIP() << "host has no NUMA topology available";

  PartitionedHarness h = makeHarness(*nodeOr);
  EXPECT_EQ(h.workQueue->getParallelismLevel(),
            h.workQueue->getCpuIds().size());
}

#if defined(__linux__)
TEST(WorkQueueTest, PartitionedRunsOnlyOnSelectedCpus) {
  std::optional<int> nodeOr = firstUsableNumaNode();
  if (!nodeOr)
    GTEST_SKIP() << "host has no NUMA topology available";

  PartitionedHarness h = makeHarness(*nodeOr);
  ArrayRef<size_t> cpuIds = h.workQueue->getCpuIds();
  ASSERT_FALSE(cpuIds.empty());
  std::set<size_t> allowedCpus(cpuIds.begin(), cpuIds.end());

  // Dispatch one task per worker with a positive taskId, forcing each task
  // onto its pinned worker. Each task records sched_getcpu() so we can check
  // that every observation lands in the NUMA node's CPU set.
  const size_t numWorkers = cpuIds.size();
  std::vector<AsyncValueRef<int>> observed;
  observed.reserve(numWorkers);
  for (size_t workerPartitionID = 0; workerPartitionID < numWorkers;
       ++workerPartitionID) {
    observed.emplace_back(AsyncValueRef<int>::allocate(*h.cpuDevice));
    AsyncValueRef<int> &slot = observed.back();
    WorkItem probe([slot = slot.copy()]() mutable {
      // sched_getcpu may return -1 on failure; store as int for inspection.
      slot.copy().emplace(::sched_getcpu());
    });
    h.workQueue->addTask(std::move(probe), static_cast<int>(workerPartitionID));
  }

  for (AsyncValueRef<int> &slot : observed)
    await(slot);

  for (size_t i = 0; i < observed.size(); ++i) {
    int cpu = observed[i].get();
    ASSERT_GE(cpu, 0) << "sched_getcpu failed for worker " << i;
    EXPECT_NE(allowedCpus.find(static_cast<size_t>(cpu)), allowedCpus.end())
        << "worker " << i << " ran on CPU " << cpu
        << " which is not in the NUMA partition";
  }
}
#endif // defined(__linux__)

TEST(WorkQueueTest, UnpartitionedReportsNoPartition) {
  CPUDeviceOptions options;
  options.numThreads = 2;
  options.mainWillDonate = false;
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);
  WorkQueue *workQueue = cpuDevice->getWorkQueue();

  EXPECT_EQ(workQueue->getCpuIds().size(), 2u);
  EXPECT_EQ(workQueue->getNumaNode(), M::kAnyNumaNode);
}

TEST(WorkQueueTest, PartitionedRejectsUnknownNumaNode) {
  const ErrorOr<NUMATopology> &topologyOr = NUMATopology::get();
  if (topologyOr.isError())
    GTEST_SKIP() << "host has no NUMA topology available";

  CPUDeviceOptions options;
  options.withSingleThreaded();
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);
  CompactCPUDevicePtr ptr = cpuDevice->getCompactPtr();

  // Pick a NUMA node id that definitely doesn't exist on this host.
  EXPECT_DEATH_IF_SUPPORTED(
      {
        auto wq = createPartitionedThreadPoolWorkQueue(
            ptr, /*numaNode=*/INT_MAX, std::chrono::microseconds(200),
            "NumaTest", /*globalWorkerIdOffset=*/0);
        (void)wq;
      },
      "no CPUs for requested NUMA node");
}

//===----------------------------------------------------------------------===//
// DelegateThreadPoolWorkQueue tests
//===----------------------------------------------------------------------===//

/// Verify that a DelegateThreadPoolWorkQueue created from NUMA partitions
/// correctly sets global worker IDs so that addLocalTask from a worker thread
/// routes to the right partition, shouldRunInlineForTask returns true when
/// called from the matching worker, and foreign-thread behavior is correct:
/// kDefaultTaskId/negative IDs are always inlineable, positive global worker
/// IDs are not, and addLocalTask from a foreign thread completes work.
TEST(WorkQueueTest, DelegateNUMAWorkerRouting) {
  const ErrorOr<NUMATopology> &topologyOr = NUMATopology::get();
  if (topologyOr.isError())
    GTEST_SKIP() << "host has no NUMA topology available";
  const std::vector<int> &nodes = topologyOr->getNumaNodes();
  if (nodes.size() < 1)
    GTEST_SKIP() << "no usable NUMA nodes on this host";

  CPUDeviceOptions options;
  options.withSingleThreaded();
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);
  CompactCPUDevicePtr ptr = cpuDevice->getCompactPtr();

  auto busyWait = std::chrono::microseconds(200);
  std::vector<std::unique_ptr<WorkQueue>> ownedPartitions;
  std::vector<WorkQueue *> rawPtrs;
  size_t globalWorkerIdOffset = 0;
  for (int node : nodes) {
    if (topologyOr->getCpuIdsForNumaNode(node).empty())
      continue;
    ownedPartitions.push_back(createPartitionedThreadPoolWorkQueue(
        ptr, node, busyWait, "DelegateNUMA", globalWorkerIdOffset));
    globalWorkerIdOffset += ownedPartitions.back()->getParallelismLevel();
    rawPtrs.push_back(ownedPartitions.back().get());
  }
  if (ownedPartitions.empty())
    GTEST_SKIP() << "no usable NUMA nodes with CPUs";

  auto delegateWQ = createDelegateThreadPoolWorkQueue(ptr, std::move(rawPtrs));
  const size_t totalWorkers = delegateWQ->getParallelismLevel();

  // Foreign-thread checks: kDefaultTaskId/negative always inlineable, positive
  // global worker IDs are not owned by this (foreign) thread.
  EXPECT_TRUE(delegateWQ->shouldRunInlineForTask(kDefaultTaskId));
  EXPECT_TRUE(delegateWQ->shouldRunInlineForTask(-5));
  EXPECT_FALSE(delegateWQ->shouldRunInlineForTask(0));
  EXPECT_FALSE(
      delegateWQ->shouldRunInlineForTask(static_cast<int>(totalWorkers - 1)));

  // addLocalTask from a foreign thread falls back to round-robin addTask.
  auto localResult = AsyncValueRef<int>::allocate(*cpuDevice);
  delegateWQ->addLocalTask(
      [slot = localResult.copy()]() mutable { slot.copy().emplace(42); });
  await(localResult);
  EXPECT_EQ(localResult.get(), 42);

  // Dispatch one task per global worker ID and verify that:
  //   1) shouldRunInlineForTask(globalId) returns true from that worker, and
  //   2) getCurrentGlobalWorkerID() returns the expected global ID.
  std::vector<AsyncValueRef<bool>> inlineResults(totalWorkers);
  std::vector<AsyncValueRef<size_t>> globalIdResults(totalWorkers);
  for (size_t globalId = 0; globalId < totalWorkers; ++globalId) {
    inlineResults[globalId] = AsyncValueRef<bool>::allocate(*cpuDevice);
    globalIdResults[globalId] = AsyncValueRef<size_t>::allocate(*cpuDevice);
    delegateWQ->addTask(
        [&delegateWQ, globalId, inlineSlot = inlineResults[globalId].copy(),
         idSlot = globalIdResults[globalId].copy()]() mutable {
          inlineSlot.copy().emplace(
              delegateWQ->shouldRunInlineForTask(static_cast<int>(globalId)));
          idSlot.copy().emplace(getCurrentGlobalWorkerID());
        },
        static_cast<int>(globalId));
  }

  for (size_t i = 0; i < totalWorkers; ++i) {
    await(inlineResults[i]);
    await(globalIdResults[i]);
    EXPECT_TRUE(inlineResults[i].get())
        << "shouldRunInlineForTask returned false for global worker " << i;
    EXPECT_EQ(globalIdResults[i].get(), i)
        << "globalWorkerIDInTLS mismatch for global worker " << i;
  }

  delegateWQ->shutdown();
  for (auto &p : ownedPartitions)
    p->shutdown();
}

#if defined(__linux__)
/// Verifies that each worker in a DelegateThreadPoolWorkQueue runs on exactly
/// the CPU at its local-index position within its partition's CPU list,
/// regardless of whether those CPU IDs are contiguous.
///
/// For example, with NUMA 0: CPUs {0, 2, 4, 6} and global offset 0:
///   global=0, local=0 → must run on CPU 0  (getCpuIds()[0])
///   global=1, local=1 → must run on CPU 2  (getCpuIds()[1])
///   global=2, local=2 → must run on CPU 4  (getCpuIds()[2])
///   global=3, local=3 → must run on CPU 6  (getCpuIds()[3])
///
/// NUMA 1: CPUs {1, 3, 5, 7} and global offset 4:
///   global=4, local=0 → must run on CPU 1  (getCpuIds()[0])
///   global=5, local=1 → must run on CPU 3  (getCpuIds()[1])
///   ...
TEST(WorkQueueTest, DelegateWorkerCpuAffinityMatchesLocalIndex) {
  const ErrorOr<NUMATopology> &topologyOr = NUMATopology::get();
  if (topologyOr.isError())
    GTEST_SKIP() << "host has no NUMA topology available";

  CPUDeviceOptions options;
  options.withSingleThreaded();
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);
  CompactCPUDevicePtr ptr = cpuDevice->getCompactPtr();

  auto busyWait = std::chrono::microseconds(200);

  // Build partitions and record the expected CPU for each global worker ID.
  // expectedCpu[globalId] = partition.getCpuIds()[localId]
  std::vector<std::unique_ptr<WorkQueue>> ownedPartitions;
  std::vector<WorkQueue *> rawPtrs;
  std::vector<size_t> expectedCpu;
  size_t globalWorkerIdOffset = 0;
  for (int node : topologyOr->getNumaNodes()) {
    std::vector<size_t> cpuIds = topologyOr->getCpuIdsForNumaNode(node);
    if (cpuIds.empty())
      continue;
    ownedPartitions.push_back(createPartitionedThreadPoolWorkQueue(
        ptr, node, busyWait, "DelegateCpuAffinity", globalWorkerIdOffset));
    // getCpuIds() is ordered by local worker index, matching
    // cpuIDs[workerPartitionID].
    for (size_t localId = 0; localId < cpuIds.size(); ++localId)
      expectedCpu.push_back(cpuIds[localId]);
    globalWorkerIdOffset += ownedPartitions.back()->getParallelismLevel();
    rawPtrs.push_back(ownedPartitions.back().get());
  }
  if (ownedPartitions.empty())
    GTEST_SKIP() << "no usable NUMA nodes with CPUs";

  auto delegateWQ = createDelegateThreadPoolWorkQueue(ptr, std::move(rawPtrs));
  const size_t totalWorkers = delegateWQ->getParallelismLevel();
  ASSERT_EQ(expectedCpu.size(), totalWorkers);

  // Dispatch one task per global worker and record which CPU sched_getcpu()
  // reports. The task is pinned to its global worker via addTask(globalId).
  std::vector<AsyncValueRef<int>> cpuResults(totalWorkers);
  for (size_t globalId = 0; globalId < totalWorkers; ++globalId) {
    cpuResults[globalId] = AsyncValueRef<int>::allocate(*cpuDevice);
    delegateWQ->addTask(
        [slot = cpuResults[globalId].copy()]() mutable {
          slot.copy().emplace(::sched_getcpu());
        },
        static_cast<int>(globalId));
  }

  for (size_t globalId = 0; globalId < totalWorkers; ++globalId) {
    await(cpuResults[globalId]);
    int observedCpu = cpuResults[globalId].get();
    ASSERT_GE(observedCpu, 0)
        << "sched_getcpu() failed for global worker " << globalId;
    EXPECT_EQ(static_cast<size_t>(observedCpu), expectedCpu[globalId])
        << "global worker " << globalId << " ran on CPU " << observedCpu
        << " but expected CPU " << expectedCpu[globalId];
  }

  delegateWQ->shutdown();
  for (auto &p : ownedPartitions)
    p->shutdown();
}
#endif // defined(__linux__)

//===----------------------------------------------------------------------===//
// CPUDevice NUMA-partitioned tests
//===----------------------------------------------------------------------===//

/// Verify that a CPUDevice created with numaPartitioned=true has type
/// kGlobalPartitioned, owns one kNUMAPartition sub-device per NUMA node, and
/// that getNumaDevice() returns the right device or nullptr.
TEST(WorkQueueTest, NUMAPartitionedCPUDeviceStructure) {
  const ErrorOr<NUMATopology> &topologyOr = NUMATopology::get();
  if (topologyOr.isError())
    GTEST_SKIP() << "host has no NUMA topology available";

  CPUDeviceOptions options;
  options.numaPartitioned = true;
  options.mainWillDonate = false;
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);

  EXPECT_EQ(cpuDevice->getType(), CPUDeviceType::kGlobalPartitioned);
  EXPECT_EQ(cpuDevice->getNumaNode(), M::kAnyNumaNode);

  const std::vector<int> &numaNodes = topologyOr->getNumaNodes();
  EXPECT_EQ(cpuDevice->numNumaDevices(), numaNodes.size());

  for (int node : numaNodes) {
    ErrorOr<CPUDevice *> numaDeviceOr = cpuDevice->getNumaDevice(node);
    ASSERT_FALSE(numaDeviceOr.isError())
        << "missing NUMA device for node " << node;
    CPUDevice *numaDevice = *numaDeviceOr;
    EXPECT_EQ(numaDevice->getType(), CPUDeviceType::kNUMAPartition);
    EXPECT_EQ(numaDevice->getNumaNode(), node);
    EXPECT_EQ(numaDevice->numNumaDevices(), 0u);
  }

  EXPECT_TRUE(cpuDevice->getNumaDevice(INT_MAX).isError());
}

/// Verify that work dispatched through a kGlobalPartitioned CPUDevice completes
/// correctly.
TEST(WorkQueueTest, NUMAPartitionedCPUDeviceDispatchesWork) {
  const ErrorOr<NUMATopology> &topologyOr = NUMATopology::get();
  if (topologyOr.isError())
    GTEST_SKIP() << "host has no NUMA topology available";

  CPUDeviceOptions options;
  options.numaPartitioned = true;
  options.mainWillDonate = false;
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);

  constexpr int numTasks = 16;
  std::vector<AsyncValueRef<int>> results;
  results.reserve(numTasks);
  for (int i = 0; i < numTasks; ++i) {
    results.emplace_back(AsyncValueRef<int>::allocate(*cpuDevice));
    cpuDevice->getWorkQueue()->addTask(
        [i, slot = results.back().copy()]() mutable {
          slot.copy().emplace(i);
        });
  }
  for (int i = 0; i < numTasks; ++i) {
    await(results[i]);
    EXPECT_EQ(results[i].get(), i);
  }
}

/// Verify that a non-partitioned CPUDevice has type kGlobal with no NUMA
/// sub-devices.
TEST(WorkQueueTest, UnpartitionedCPUDeviceIsKGlobal) {
  CPUDeviceOptions options;
  options.mainWillDonate = false;
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(CPUDeviceSource::Test, options);
  EXPECT_EQ(cpuDevice->getType(), CPUDeviceType::kGlobal);
  EXPECT_EQ(cpuDevice->getNumaNode(), M::kAnyNumaNode);
  EXPECT_EQ(cpuDevice->numNumaDevices(), 0u);
  EXPECT_TRUE(cpuDevice->getNumaDevice(0).isError());
}

} // namespace
