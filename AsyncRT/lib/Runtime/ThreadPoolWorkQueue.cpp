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
// This is a multi-threaded work queue implementation.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/AsyncValue.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "AsyncRT/Support/Chain.h"
#include "AsyncRT/Support/ConcurrentMPMCQueue.h"
#include "AsyncRT/Support/LockFreeRingBuffer.h"
#include "AsyncRT/Support/Semaphore.h"
#include "AsyncRT/Support/ThreadAffinity.h"
#include "Support/AlignedAlloc.h"
#include "Support/Configuration.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/Profiling/TimeProfiler.h"
#if MODULAR_ASYNCRT_MAX_PROFILING_LEVEL != 0
#include "Support/Profiling/Ranges.h"
#include "Support/internal/Tracy/Tracy.h"
#else
// TODO: This is duplicating some things, maybe find a better way
#define TRACY_ZONE_SCOPED_NC(name, color)
#define TRACY_ZONE_SCOPED_NCT(name, color, text)
#endif
#include "Support/Threading/Atomics.h"
#include "Support/Threading/HWInfo.h"
#include "Support/Threading/SignalAltStack.h"
#include "Support/Threading/SpinWaiter.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/thread.h"

#include <atomic>
#include <cmath>
#include <functional>
#include <thread>

#define DEBUG_TYPE "asyncrt"

using namespace M;
using namespace M::AsyncRT;

//
// Terminology:
//  - Worker thread: a thread we create which is running a dedicated runItems
//    loop.
//  - Main thread: if in mainWillDonate mode, this is the thread which created
//    the work queue. That thread may call await to donate itself to processing
//    work items alongside the worker threads while waiting for values. That
//    thread must also be the one to call shutdown.
//  - Foreign thread: any thread other than a worker or main thread. Foreign
//    threads may call addTasks and await. If not in mainWillDonate mode,
//    a foreign thread may also call shutdown. A foreign thread will never
//    donate itself to processing work items.
//

//===----------------------------------------------------------------------===//
// Compile-time config
//===----------------------------------------------------------------------===//

/// Number of task list slots per thread.
constexpr size_t kTaskListSlotsPerThread = 1024;

/// Max number of worker threads.
constexpr size_t kMaxWorkers = 1024;

/// Default worker-thread stack size. Inline compile nesting overflows the small
/// stack spawned threads inherit (~512 KB on macOS); 8 MiB matches the typical
/// Linux main-thread stack. See GEX-3876.
constexpr unsigned kDefaultWorkerStackSizeBytes = 8u * 1024u * 1024u;

inline std::optional<unsigned> getWorkerStackSizeBytes() {
  static const unsigned bytes = [] {
    if (auto config = Config::open(); !config.isError()) {
      if (auto v = config->maybeGetValue("runtime.worker_stack_size_mb")) {
        int mb;
        if (!v->getAsInteger(10, mb) && mb > 0)
          return static_cast<unsigned>(mb) * 1024u * 1024u;
      }
    }
    return kDefaultWorkerStackSizeBytes;
  }();
  return bytes;
}

//===----------------------------------------------------------------------===//
// WorkerThread
//===----------------------------------------------------------------------===//

namespace {
#if ASYNCRT_WORKER_STATS
#define ASYNCRT_PRINT_WORKER_STATS(X) X;
#else
#define ASYNCRT_PRINT_WORKER_STATS(X)
#endif
/// Tracks the overall shutdown progress for the work queue.
enum WorkQueueState : uint8_t { kReady = 0, kShuttingDown = 1, kShutdown = 2 };
enum WorkType : uint8_t { kLocal = 0, kAffinity = 1, kGlobal = 2 };

/// Provides the state needed to synchronize the workers in the thread pool.
/// We use a uint64_t bit-vec (SuspendedThreadsBitvec) to represent the
/// suspended bit of each thread. This comes with a limitation that the system
/// could have just 64 threads. However most modern server cpu's have more than
/// 64 cores in a node. We implement scaling support through a simple
/// multi-cast scheme where each bit of the bit-vec represents more
/// than 1 thread, ie a `workerGroup` instead of a `worker`.
/// For example for a 128 cpu machine, bit 0 represents {worker0, worker1}.
/// If bit0 is set, it means either worker0 | worker1 is suspended.
/// This results in ambiguity when we query the bit-vec for addTask()/await()
/// to wakeup threads because the exact sleeping localWorkerID is unknown to
/// post the appropriate semaphore. We handle this in the following way, 1) when
/// workerId is unknown, we will wakeup all the threads represented by the
/// bit-vec bit. For example, 0 -> {worker0->post(), worker1->post()} This can
/// be expensive, but hopefully, we do not have to sleep/wake up threads often
/// during model execution. 2) when localWorkerID is known, we always post
/// the semaphore since bit-vec information may have interference from other
/// threads. This can lead to spurious posts() but nonetheless ensures, the
/// threads wake up to execute the task.

/// Bit index i is true if any thread in the workedGroupID i is suspended.
using SuspendedThreadsBitvec = uint64_t;
constexpr size_t bitVectorWidth = sizeof(SuspendedThreadsBitvec) * 8;
constexpr SuspendedThreadsBitvec
getSuspendedThreadIdMask(size_t workerGroupID) {
  return UINT64_C(1) << workerGroupID;
}

struct SharedThreadState {
  static_assert(std::atomic<SuspendedThreadsBitvec>::is_always_lock_free,
                "suspendedThreads should always be lock free");

  SharedThreadState(CompactCPUDevicePtr cpuDevicePtr, bool mainWillDonate,
                    size_t numWorkers, int numaNode = kAnyNumaNode)
      : cpuDevicePtr(cpuDevicePtr), mainWillDonate(mainWillDonate),
        numaNode(numaNode) {
    // Keeping numWorkers in a workerGroup a power of 2 to simplify arithmetic.
    multicastFactor =
        numWorkers > bitVectorWidth
            ? static_cast<size_t>(std::ceil(
                  std::log2(numWorkers / static_cast<float>(bitVectorWidth))))
            : 0;
  }

  /// The cpuDevice on behalf of which this thread is processing work items.
  CompactCPUDevicePtr cpuDevicePtr;

  /// If true, the 'main' thread which constructed the work queue is going to
  /// call await to donate itself as another worker alongside the
  /// numWorkers - 1 other worker threads. That thread must eventually call
  /// shutdown.
  ///
  /// Otherwise there is no 'main' thread, just 'worker' and 'foreign' threads.
  bool mainWillDonate;

  /// NUMA node this queue is partitioned to, or kAnyNumaNode if unpartitioned.
  /// Used by runOnThread() to decide whether to initialise globalWorkerIDInTLS.
  int numaNode;

  /// Track when the overall work queue is entering or exited the shutdown
  /// quiescence period.
  std::atomic<WorkQueueState> state = kReady;

  /// This flag indicates when a worker thread should quit working and get
  /// ready to be joined.
  std::atomic<bool> doneFlag = false;
  /// computed so that number of workers per groups is 2^multicastFactor.
  size_t multicastFactor;
  /// This keeps a bitset of suspended threads, indexed by localWorkerID.
  /// This will thrash around a lot when the workqueue is close to empty and
  /// threads are starting and stopping themselves, but should stay zero and
  /// read-only when there is a lot of work to do.
  ///
  /// This is aligned because the state above is immutable or (in the case of
  /// doneFlag) almost never changing. We don't want doneFlag to be on the same
  /// cache line as suspendedThreads.
  AlignedAtomic<SuspendedThreadsBitvec> suspendedThreads = 0;

  /// When a worker is about to go to sleep, it calls this method so andThenSync
  /// can know to wake it up when more work materializes. We set the bit of the
  /// corresponding workerGroupID
  void markSuspended(size_t localWorkerID) {
    // number of workers per groups is 2^multicastFactor.
    auto workerGroupID = localWorkerID >> multicastFactor;
    suspendedThreads.fetch_or(getSuspendedThreadIdMask(workerGroupID),
                              std::memory_order_seq_cst);
  }

  /// If the specified localWorkerID is suspended, take its bit out of the
  /// suspendedThreads bitset and return true.  Otherwise return false.
  /// NOTE: takeSuspended may unset even if some other threads in the
  /// same workerGroup are suspended. This is fine since, we will always call
  /// the localWorkerID->sema.post().
  bool takeSuspendedThread(size_t localWorkerID) {
    // number of workers per groups is 2^multicastFactor.
    auto workerGroupID = localWorkerID >> multicastFactor;
    SuspendedThreadsBitvec workerBit = getSuspendedThreadIdMask(workerGroupID);
    auto oldValue =
        suspendedThreads.fetch_and(~workerBit, std::memory_order_seq_cst);
    return oldValue & workerBit;
  }

  /// If there are any workerGroup's with suspended threads, return the id for
  /// one of them. Otherwise return -1. Since we do not know the
  /// localWorkerID which is suspended, we assume all worker's are suspended
  /// and hence will post all the semaphores.
  int takeAnySuspendedThread() {
    SuspendedThreadsBitvec loadedSuspendedThreads =
        suspendedThreads.load(std::memory_order_seq_cst);
    if (loadedSuspendedThreads == 0)
      return -1;

    // Iteratively compare/xchg to extract the low bit out of suspendedThreads.
    SpinWaiter<> spinner;
    do {
      // Clear the lowest bit set in suspendedThreads with `x & (x-1)` idiom.
      SuspendedThreadsBitvec newSuspendedThreads =
          loadedSuspendedThreads & (loadedSuspendedThreads - 1);

      // Try to atomically swap in the new value.
      if (suspendedThreads.compare_exchange_weak(loadedSuspendedThreads,
                                                 newSuspendedThreads)) {
        // When we succeed, that means we were successful in clearing the
        // lowermost bit.  Map that bit back into a localWorkerID and return
        // it.
        return llvm::countr_zero(loadedSuspendedThreads ^ newSuspendedThreads);
      }

      spinner.wait();
    } while (loadedSuspendedThreads != 0);

    // We saw a candidate but it fell away.
    return -1;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// WorkQueueThread
//===----------------------------------------------------------------------===//

namespace {

/// The index of the current thread within the WorkQueueThread workers
/// vector. Will be left zero for 'main' and 'foreign' threads.
static thread_local size_t localWorkerIDInTLS = 0;

/// The global worker ID assigned to this thread by a DelegateWorkQueue, or
/// SIZE_MAX if this thread is not a worker of a DelegateWorkQueue.
static thread_local size_t globalWorkerIDInTLS = SIZE_MAX;

/// Wrapper around an std::thread created for each worker thread, or
/// a placeholder for the 'main' thread.
struct WorkQueueThread {
  /// Overall state shared by all threads.
  SharedThreadState &sharedState;

  /// 'Local' work items which can be run on this thread as they become
  /// available. No threading synchronization is required here since work items
  /// are added to and removed only by the unique thread (currently) tied to
  /// this object. However, we do need to protect against runItems being called
  /// recursively.
  ///
  /// Work items on this list always take precedence over those in taskList and
  /// overflowTaskList.
  size_t nextLocalTaskListIndex = 0;
  SmallVector<WorkItem, 6> localTaskList;
  /// Thread Local Queue of tasks processed according to taskId ordering.
  /// This is like localTaskList but can have multiple producers.
  LockFreeRingBuffer<WorkItem> affinityTaskList;
  /// The lock-free queue of pending tasks available for any worker to
  /// process.
  ///
  /// Work items on this list always take precedence over those in
  /// overflowTaskList.
  MoodyCamel::ConcurrentQueue<WorkItem> &taskList;

  /// The mutex-protected queue of pending 'overflow' work items available for
  /// any worker to process. Since synchronization is expensive, should only be
  /// checked before the worker thread would otherwise sleep.
  std::mutex &overflowMutex; // Protects overflowTaskList
  SmallVectorImpl<WorkItem> &overflowTaskList;

  /// Spill queue for the affinityTaskList and its mutex. If the
  /// affinityTaskList is full, we spill over to this queue and later execute
  /// from the localTaskList maintaining the affinity. This is assumed to be
  /// a rare event and hence okay with slow handling like overflowTaskList.
  std::mutex localSpillQueueMutex; // Protects localSpillQueue
  SmallVector<WorkItem> localSpillQueue;
  /// Unique index for this thread.
  size_t localWorkerID;

  /// The CPU we'd prefer this worker to have affinity for, or ~0 if no
  /// affinity is intended for this worker.
  size_t cpuID;

  /// Amount of time to spend spinning while waiting for work before going to
  /// sleep on a semaphore. Tuning this number is especially important for
  /// cases that interact with other threadpools. Ideally we should autotune
  /// this during the `warmup` phase or come up with heuristics based on
  /// the fallback ops distribution.
  std::chrono::microseconds busyWaitTime;

  /// This is a per-worker semaphore that this blocks on when they run
  /// out of things to do.
  Semaphore sema;

  /// The system's identifier for the thread associated with this
  /// WorkQueueThread, either a 'worker' or the 'main' thread if in
  /// mainWillDonate mode.
  uint64_t threadID = 0;

  /// The underlying worker thread, or none if this WorkQueueThread represents
  /// the 'main' thread in mainWillDonate mode. llvm::thread (not std::thread)
  /// to set a larger stack size cross-platform.
  std::optional<llvm::thread> thread;
  // The thread identifier prefix used to name the threads
  std::string_view poolName;
  /// Global worker ID for this thread within a DelegateWorkQueue, or SIZE_MAX
  /// if not owned by a DelegateWorkQueue.
  const size_t globalWorkerId;
#if ASYNCRT_WORKER_STATS
  uint64_t affinityAccessCount = 0;
  uint64_t globalAccessCount = 0;
  std::chrono::duration<double, std::micro> affinityListAccessTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> localListAccessTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> taskListAccessTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> affinityWorkTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> localWorkTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> taskListWorkTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> spinAffinityListAccessTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> spinAffinityWorkTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> spinTaskListAccessTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> spinTaskListWorkTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> sleepTime =
      std::chrono::microseconds(0);
#endif
  /// Create a WorkQueueThread representing the worker with localWorkerID.
  /// If necessary, the underlying worker thread will be created and it will
  /// enter its runItems loop.
  WorkQueueThread(SharedThreadState &sharedState,
                  MoodyCamel::ConcurrentQueue<WorkItem> &taskList,
                  std::mutex &overflowMutex,
                  SmallVectorImpl<WorkItem> &overflowTaskList,
                  size_t localWorkerID, size_t cpuID,
                  std::chrono::microseconds busyWaitTime,
                  std::string_view poolName, size_t globalWorkerId = SIZE_MAX)
      : sharedState(sharedState), affinityTaskList(kTaskListSlotsPerThread),
        taskList(taskList), overflowMutex(overflowMutex),
        overflowTaskList(overflowTaskList), localWorkerID(localWorkerID),
        cpuID(cpuID), busyWaitTime(busyWaitTime), poolName(poolName),
        globalWorkerId(globalWorkerId) {
    if (sharedState.mainWillDonate && localWorkerID == 0) {
      // We can leave localWorkerIDInTLS as zero.
      // Remember the caller is to be our 'main' thread, and will call
      // await to process work items.
      threadID = llvm::get_threadid();
      assert(threadID && "get_threadid returned zero for the main thread");
    } else {
      // Lambda because llvm::thread calls the callable directly, not via
      // std::invoke (so it can't bind a pointer-to-member + this).
      thread.emplace(getWorkerStackSizeBytes(), [this]() { runOnThread(); });
    }
  }

  ~WorkQueueThread() {
    if (localWorkerID == 0) {
      ASYNCRT_PRINT_WORKER_STATS(
          llvm::dbgs() << "WorkerID,schedulerTasks(us),affinityQueueAccess(us),"
                          "affinityQueueWork(us),affinityAccessCount,"
                          "globalAccess(us),globalWork("
                          "us),globalAccessCount,sleep+wakeup(us)\n");
    }
    ASYNCRT_PRINT_WORKER_STATS(
        llvm::dbgs()
        << "Thread" << localWorkerID << "," << (localWorkTime).count() << ","
        << (affinityListAccessTime - affinityWorkTime).count() +
               (spinAffinityListAccessTime - spinAffinityWorkTime).count()
        << "," << (affinityWorkTime).count() + (spinAffinityWorkTime).count()
        << "," << affinityAccessCount << ","
        << (taskListAccessTime - taskListWorkTime).count() +
               (spinTaskListAccessTime - spinTaskListWorkTime).count()
        << "," << (taskListWorkTime).count() + (spinTaskListWorkTime).count()
        << "," << globalAccessCount << "," << sleepTime.count() << "\n");
    assert(localTaskList.empty() &&
           "destroying workqueuethread with pending local work items");
    std::lock_guard<std::mutex> guard(localSpillQueueMutex);
    assert(localSpillQueue.empty() &&
           "destroying Workqueuethread with pending fallback work items");
  }

  /// Schedule this work item on the localTaskList to be executed on the next
  /// runItems loop.
  void addLocalTask(WorkItem &&workItem) {
    localTaskList.emplace_back(std::move(workItem));
  }

  /// Schedules work on to the thread local queue of this worker. If the
  /// lockFreeRingBuffer is full, enqueue into the spill queue.
  void addAffinityTask(WorkItem &&workItem) {
    if (!affinityTaskList.enqueue(workItem)) {
      std::lock_guard<std::mutex> guard(localSpillQueueMutex);
      localSpillQueue.emplace_back(std::move(workItem));
    }
  }

  /// Joins the thread. Asserts that `sharedState.done` is true because
  /// otherwise the thread will never join.
  void join() {
    assert(sharedState.doneFlag.load() &&
           "must not destroy a WorkQueueThread object that is not pending "
           "completion.");
    if (thread.has_value())
      thread->join();
  }

  // Execute a single work item, which may have come from either addTask
  // or addLocalTask (via an AsyncValue waiter).
  template <bool IsWaiter>
  void doWork(WorkItem &&workItem, WorkType type) {
    TRACY_ZONE_SCOPED_NCT("WorkQueueThread::doWork", TRACY_COLOR_BLUE,
                          "Unique task ID: " +
                              std::to_string(workItem.uniqueTaskId));

#if ASYNCRT_WORKER_STATS
    auto start = std::chrono::high_resolution_clock::now();
#endif
    // Register this worker with the profiler so its GPU-driver activity is
    // attributed to a named thread track in the trace. The first call after
    // enable() does the registration; subsequent calls are one TLS read.
#if MODULAR_ASYNCRT_MAX_PROFILING_LEVEL != 0
    M::Profiling::registerCurrentThreadIfEnabled();
#endif

    // Do the work.
    {
      TimeTraceScope scope(AllWorkItemsProfilerEntry::create(
          IsWaiter ? "asyncrt.waiter" : "asyncrt.doWork"));
      workItem.task();
    }
#if ASYNCRT_WORKER_STATS
    auto end = std::chrono::high_resolution_clock::now();

    if (type == kLocal)
      localWorkTime += end - start;
    if (!IsWaiter) {
      if (type == kAffinity)
        affinityWorkTime += end - start;
      else if (type == kGlobal)
        taskListWorkTime += end - start;
    } else {
      if (type == kAffinity)
        spinAffinityWorkTime += end - start;
      else if (type == kGlobal)
        spinTaskListWorkTime += end - start;
    }
#endif
  }

  /// This implements the main worker loop, used by runOnThread, await and
  /// shutdown. The loop runs until earlyStopPredicate or lateStopPredicate
  /// return true. The "early" predicate is called for every work item that
  /// is executed, and the "late" one is called when waking up from a
  /// suspended state.
  ///
  /// The loop will busy wait or sleep waiting for new work items only if
  /// waitForTasks is true, otherwise the loop will exit once the work queue
  /// and local task list is empty.
  ///
  /// The given labels are used only for profiling entries when spinning or
  /// sleeping.
  template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
  void runItemsOnOwningThread(EarlyStopPredicateFn earlyStopPredicate,
                              LateStopPredicateFn lateStopPredicate,
                              bool waitForTasks, StringLiteral spinningLabel,
                              StringLiteral sleepingLabel);

  /// As above, but without setting thread affinity for calls from the 'main'
  /// thread.
  template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
  void runItemsImpl(EarlyStopPredicateFn earlyStopPredicate,
                    LateStopPredicateFn lateStopPredicate, bool waitForTasks,
                    StringLiteral spinningLabel, StringLiteral sleepingLabel);

private:
  /// The main function invoked by std::thread.
  void runOnThread();
};
} // namespace

void WorkQueueThread::runOnThread() {
  assert((!sharedState.mainWillDonate || localWorkerID != 0) &&
         "the WorkQueueThread for the main thread should not be run");

  // Work items recurse arbitrarily deep (inline expansion, type conversion
  // search), so a worker can exhaust its stack. Without this the crash handler
  // has nowhere to run and the overflow goes unreported.
  ScopedSignalAltStack signalAltStack;

  // Set the current localWorkerID in thread local storage so we can find it
  // later when re-entering.
  localWorkerIDInTLS = localWorkerID;
  // Set global worker ID in TLS for NUMA-partitioned queues owned by a
  // DelegateThreadPoolWorkQueue. Non-NUMA queues leave globalWorkerIDInTLS
  // at its sentinel value (SIZE_MAX).
  if (sharedState.numaNode != kAnyNumaNode)
    globalWorkerIDInTLS = globalWorkerId;

  // Set the current cpuDevice in thread local storage.
  CompactCPUDevicePtr::setCurrentCPUDevice(sharedState.cpuDevicePtr);

  // Capture the worker's thread id so we can distinguish worker threads
  // from different work queues.
  threadID = llvm::get_threadid();
  assert(threadID && "get_threadid returned zero for a worker thread");

  // On systems that support it, give the thread a symbolic name that will show
  // up in profilers and debuggers.
  llvm::set_thread_name(poolName + llvm::Twine(localWorkerID));

  // On systems that support it, give the thread affinity for one CPU.
  AsyncRT::setThreadAffinity(cpuID);

  // Run work items until the system is asked to shut down.
  runItemsOnOwningThread(
      /*earlyStopPredicate=*/[]() { return false; }, // Always loop.
      /*lateStopPredicate=*/
      [this]() {
        // On wakeup from suspend, check to see if we're supposed to
        // shutdown and stop executing work.
        return sharedState.doneFlag.load(std::memory_order_acquire);
      },
      /*waitForTasks=*/true,
      /*spinningLabel=*/"asyncrt.runOnThread.spinning",
      /*sleepingLabel=*/"asyncrt.runOnThread.sleeping");
}

template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
void WorkQueueThread::runItemsOnOwningThread(
    EarlyStopPredicateFn earlyStopPredicate,
    LateStopPredicateFn lateStopPredicate, bool waitForTasks,
    StringLiteral spinningLabel, StringLiteral sleepingLabel) {
  if (sharedState.mainWillDonate && localWorkerID == 0) {
    // Temporarily set the main thread's affinity while it is processing work.
    AsyncRT::runWithThreadAffinity(cpuID, [&]() {
      runItemsImpl<EarlyStopPredicateFn, LateStopPredicateFn>(
          earlyStopPredicate, lateStopPredicate, waitForTasks, spinningLabel,
          sleepingLabel);
    });
  } else {
    runItemsImpl<EarlyStopPredicateFn, LateStopPredicateFn>(
        earlyStopPredicate, lateStopPredicate, waitForTasks, spinningLabel,
        sleepingLabel);
  }
}
template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
void WorkQueueThread::runItemsImpl(EarlyStopPredicateFn earlyStopPredicate,
                                   LateStopPredicateFn lateStopPredicate,
                                   bool waitForTasks,
                                   StringLiteral spinningLabel,
                                   StringLiteral sleepingLabel) {
  TRACY_ZONE_SCOPED_NC("WorkQueueThread::runItemsImpl", TRACY_COLOR_BLUE);

  while (true) {
  KeepRunning:
    // Stop immediately if there is nothing to do.
    if (earlyStopPredicate())
      return;

    // Prefer to run local work items as soon as they are available.
    // CAUTION: A work function may append to this list, and may even
    //          invoke runItems recursively.
    auto start = std::chrono::high_resolution_clock::now();
    while (nextLocalTaskListIndex < localTaskList.size()) {
      WorkItem workItem = std::move(localTaskList[nextLocalTaskListIndex++]);

      // May append to localTaskList.
      // May re-enter this loop.
      {
        TRACY_ZONE_SCOPED_NC("Invoke WorkQueueThread::doWork (immediate,local)",
                             TRACY_COLOR_BLUE);
        doWork</*IsWaiter=*/true>(std::move(workItem), kLocal);
      }
    }
    localTaskList.clear();
    nextLocalTaskListIndex = 0;
#if ASYNCRT_WORKER_STATS
    auto end = std::chrono::high_resolution_clock::now();
    localListAccessTime +=
        std::chrono::duration<double, std::micro>(end - start);
    start = std::chrono::high_resolution_clock::now();
#endif
    // Check for tasks in local taskId affinities queue.
    if (auto workItem = affinityTaskList.dequeue()) {
      {
        TRACY_ZONE_SCOPED_NC(
            "Invoke WorkQueueThread::doWork (immediate,affinity)",
            TRACY_COLOR_BLUE);
        doWork</*IsWaiter=*/false>(std::move(workItem), kAffinity);
      }
#if ASYNCRT_WORKER_STATS
      auto end = std::chrono::high_resolution_clock::now();
      affinityListAccessTime += (end - start);
      ++affinityAccessCount;
#endif
      goto KeepRunning;
    }
#if ASYNCRT_WORKER_STATS
    end = std::chrono::high_resolution_clock::now();
    affinityListAccessTime += (end - start);
    start = std::chrono::high_resolution_clock::now();
#endif
    // In the normal case we happily pick up and do work.

    if (WorkItem workItem; taskList.try_dequeue(workItem)) {
      {
        TRACY_ZONE_SCOPED_NC(
            "Invoke WorkQueueThread::doWork (immediate,global)",
            TRACY_COLOR_BLUE);
        doWork</*IsWaiter=*/false>(std::move(workItem), kGlobal);
      }
#if ASYNCRT_WORKER_STATS
      auto end = std::chrono::high_resolution_clock::now();
      taskListAccessTime += (end - start);
      ++globalAccessCount;
#endif
      goto KeepRunning;
    }
#if ASYNCRT_WORKER_STATS
    end = std::chrono::high_resolution_clock::now();
    taskListAccessTime += (end - start);
#endif

    if (!waitForTasks)
      return;

    {
      auto spinning =
          InternalProfilerEntry::create(spinningLabel, (uint64_t)localWorkerID);

      // If we've run out of work to do, we need to quiesce and ultimately block
      // in the kernel on the semaphore.  However, we don't want to immediately
      // give up hope, because we may be "right about to" get new work incoming.
      // We also want to make sure to use exponential backoff to avoid pummeling
      // the memory hierarchy of the threads that are doing useful work.  As
      // such, we use a BusyWaitSpinWaiter.
      BusyWaitSpinWaiter spinWaiter(busyWaitTime);

      start = std::chrono::high_resolution_clock::now();
      // Spin until we find some work to do.
      while (!spinWaiter.wait()) {
        // If we ever succeed in finding work to do, go back to running like
        // normal.

        if (auto workItem = affinityTaskList.dequeue()) {
          {
            TRACY_ZONE_SCOPED_NC(
                "Invoke WorkQueueThread::doWork (spin,affinity)",
                TRACY_COLOR_BLUE);
            doWork</*IsWaiter=*/true>(std::move(workItem), kAffinity);
          }
#if ASYNCRT_WORKER_STATS
          auto end = std::chrono::high_resolution_clock::now();
          spinAffinityListAccessTime += (end - start);
          ++affinityAccessCount;
#endif
          goto KeepRunning;
        }
#if ASYNCRT_WORKER_STATS
        end = std::chrono::high_resolution_clock::now();
        spinAffinityListAccessTime += (end - start);
        start = std::chrono::high_resolution_clock::now();
#endif

        if (WorkItem workItem; taskList.try_dequeue(workItem)) {
          {
            TRACY_ZONE_SCOPED_NC("Invoke WorkQueueThread::doWork (spin,global)",
                                 TRACY_COLOR_BLUE);
            doWork</*IsWaiter=*/true>(std::move(workItem), kGlobal);
          }
#if ASYNCRT_WORKER_STATS
          auto end = std::chrono::high_resolution_clock::now();
          ++globalAccessCount;
          spinTaskListAccessTime += (end - start);
#endif
          goto KeepRunning;
        }
#if ASYNCRT_WORKER_STATS
        end = std::chrono::high_resolution_clock::now();
        spinTaskListAccessTime += (end - start);
#endif
        // If we're spinning and the early or the late stop condition happens,
        // then we're done.  Checking the late stop condition here make sure
        // our threads shut down promptly when a cpuDevice is torn down.
        if (earlyStopPredicate() || lateStopPredicate()) {
          std::move(spinning).record();
          return;
        }
      }
      std::move(spinning).record();
    }

    // The lock-free task queue appears to be empty. Since we're about to go
    // to sleep anyway, we can justify the expense of pumping any items out
    // of the overflow/localSpill task queues into the lock-free queues.
    // Note we don't worry about preserving order for the overflow tasks since
    // there's no guarantee of fairness anyway.
    {
      std::lock_guard<std::mutex> guard(localSpillQueueMutex);
      if (!localSpillQueue.empty()) {
        while (!localSpillQueue.empty()) {
          WorkItem workItem = localSpillQueue.pop_back_val();
          localTaskList.emplace_back(std::move(workItem));
        }
        goto KeepRunning;
      }
    }

    {
      std::lock_guard<std::mutex> guard(overflowMutex);
      if (!overflowTaskList.empty()) {
        while (!overflowTaskList.empty()) {
          WorkItem workItem = overflowTaskList.pop_back_val();
          if (!taskList.enqueue(std::move(workItem))) {
            // Oops, went too far.
            overflowTaskList.emplace_back(std::move(workItem));
            break;
          }
        }
        goto KeepRunning;
      }
    }

    // We've waited long enough for new work to show up, so check one last time
    // and yield the thread to the OS so we don't burn power and starve other
    // tasks on the system.

    sharedState.markSuspended(localWorkerID);
    // Lets reason about ordering of markSuspended here and takeSuspended in
    // addTask.
    // T0(scheduler)                            T1(worker)
    // if(takeSuspended())                      markSuspended()
    // sema.post()                              sema.wait()
    //
    // Ordering 1: markSuspended() andThen takeSuspended().
    // if sema.post() andThen sema.wait() T1 does not go to sleep.
    // else T1 sleeps and wakes immediately.
    //
    // Ordering 2: takeSuspended() andThen markSuspended()
    // T0 is not going to post semaphore, but the task is already
    // enqueued. Run it now, unMark and go back to KeepRunning.

    start = std::chrono::high_resolution_clock::now();
    if (auto labelledTask = affinityTaskList.dequeue()) {
      {
        TRACY_ZONE_SCOPED_NC(
            "Invoke WorkQueueThread::doWork (pre-suspend,affinity)",
            TRACY_COLOR_BLUE);
        doWork</*IsWaiter=*/false>(std::move(labelledTask), kAffinity);
      }
#if ASYNCRT_WORKER_STATS
      auto end = std::chrono::high_resolution_clock::now();
      ++affinityAccessCount;
      affinityListAccessTime += (end - start);
#endif
      goto KeepRunning;
    }
#if ASYNCRT_WORKER_STATS
    end = std::chrono::high_resolution_clock::now();
    affinityListAccessTime += (end - start);
    start = std::chrono::high_resolution_clock::now();
#endif
    // The same ordering explanation as above holds for the taskList too.
    // Let's say there are 2 threads in the pool with both threads busy waiting
    // on their way to sleep. The addTask() sees them as busy and does not post
    // any semaphores. However they both go to sleep not to be woken up by
    // anyone. We prefer checking for a dequeue here rather than always posting
    // a semaphore after enqueue in the addTask(). Also scenario is highly
    // unlikely for numThreads > 1.

    if (WorkItem labelledTask; taskList.try_dequeue(labelledTask)) {
      {
        TRACY_ZONE_SCOPED_NC(
            "Invoke WorkQueueThread::doWork (pre-suspend,global)",
            TRACY_COLOR_BLUE);
        doWork</*IsWaiter=*/false>(std::move(labelledTask), kGlobal);
      }
#if ASYNCRT_WORKER_STATS
      auto end = std::chrono::high_resolution_clock::now();
      ++globalAccessCount;
      taskListAccessTime += (end - start);
#endif
      goto KeepRunning;
    }
#if ASYNCRT_WORKER_STATS
    end = std::chrono::high_resolution_clock::now();
    taskListAccessTime += (end - start);
#endif

    if (earlyStopPredicate()) {
      return;
    }

    {
      start = std::chrono::high_resolution_clock::now();
      // Ok, finally block.
      TimeTraceScope scope(InternalProfilerEntry::create(
          sleepingLabel, (uint64_t)localWorkerID));
      sema.wait();
#if ASYNCRT_WORKER_STATS
      auto end = std::chrono::high_resolution_clock::now();
      sleepTime += (end - start);
#endif
    }

    // On wakeup, do NOT exit immediately even when lateStopPredicate() is
    // true. Returning here skips the affinity queue drain at the top of the
    // outer loop, leaving items in LockFreeRingBuffer and triggering its
    // non-empty destructor assertion when the WorkQueueThread is torn down.
    //
    // Instead, fall through to the next iteration so the outer loop drains
    // all pending local, affinity, and global work. The spinning phase exits
    // cleanly once every queue is empty and lateStopPredicate() becomes true.
  }
}

//===----------------------------------------------------------------------===//
// ThreadPoolWorkQueue
//===----------------------------------------------------------------------===//

namespace {
/// This class provides a thread-pool that implements the WorkQueue
/// interface. It starts a dynamic number of threads and distributes work to
/// it by means of a concurrent-safe queue.
class ThreadPoolWorkQueue : public WorkQueue {
public:
  /// Initialize the thread pool and start up the worker threads, with one
  /// thread per entry in cpuIDs. By the time the constructor finishes, all
  /// the worker threads have started and shall only be cancelled by the
  /// destructor.
  ThreadPoolWorkQueue(CompactCPUDevicePtr cpuDevicePtr, ArrayRef<size_t> cpuIDs,
                      size_t taskListCapacity, bool mainWillDonate,
                      std::chrono::microseconds threadBusyWaitTime,
                      std::string_view poolName, int numaNode = kAnyNumaNode,
                      size_t globalWorkerIdOffset = SIZE_MAX);

  ~ThreadPoolWorkQueue() override;

  void shutdown() override;

  void addTask(WorkItem &&workItem, int taskId = -1) override;

  void addLocalTask(WorkItem &&workItem) override;

  void await(ArrayRef<AnyAsyncValueRef> values) override;

  size_t getParallelismLevel() const final {
    // `numWorkers` is set to the number of worker threads that are created
    // by the work queue, plus one for the 'main' thread if in mainWillDonate
    // mode.
    // TODO(#1903): This is a poor heuristic for subdividing work.
    return numWorkers;
  }

  ArrayRef<size_t> getCpuIds() const final override { return partitionCpuIds; }

  int getNumaNode() const final override { return numaNode; }

  /// Returns true when we're already on the correct worker for the given
  /// taskId, allowing inline execution instead of enqueuing.
  bool shouldRunInlineForTask(int taskId) const final override {
    if (taskId < 0) // kDefaultTaskId or invalid
      return true;

    WorkQueueThread *current = getOwningWorkQueueThread();
    if (!current)
      return false;

    return current->localWorkerID == static_cast<size_t>(taskId);
  }

private:
  /// If the caller is a worker thread or the 'main' thread for this work queue
  /// then return the WorkQueueThread which represents it. Otherwise, if the
  /// caller is a 'foreign' thread (including workers from other work queues)
  /// then return null.
  WorkQueueThread *getOwningWorkQueueThread() const {
    size_t localWorkerID = localWorkerIDInTLS;

    if (localWorkerID >= numWorkers)
      // Presumably a 'worker' thread from some other work queue.
      return nullptr;

    WorkQueueThread *worker = workers + localWorkerID;

    if (worker->threadID != llvm::get_threadid())
      // A 'foreign' thread.
      return nullptr;

    // Either the 'main' or a 'worker' thread associated with this work queue.
    return worker;
  }

  /// Returns the WorkQueueThread for localWorkerID.
  WorkQueueThread *getWorkQueueThread(size_t localWorkerID) const {
    assert(localWorkerID < numWorkers && "invalid worker id");
    return workers + localWorkerID;
  }

  /// This is the set of WorkQueueThread objects in the WorkQueue. If in
  /// mainWillDonate mode then the first entry will represent the 'main'
  /// thread.
  const size_t numWorkers;
  WorkQueueThread *workers = nullptr;

  /// NUMA node this queue is partitioned to, or kAnyNumaNode if this queue
  /// is not partitioned.
  const int numaNode;

  /// CPU IDs assigned to this queue.
  const SmallVector<size_t> partitionCpuIds;

  // Base synchronization state is held in this class, each thread holds a
  // reference to this structure.
  SharedThreadState sharedState;

  /// The lock-free queue of pending tasks available for any worker.
  /// It may become full.
  MoodyCamel::ConcurrentQueue<WorkItem> taskList;
  /// The mutex-protected queue of pending tasks available for any worker.
  /// Only used when the taskList is full.
  std::mutex overflowMutex; // protects overflowTaskList
  SmallVector<WorkItem> overflowTaskList;
  /// Log2(number of threads per bit of SuspendedThreadsBitvec)
  size_t multicastFactor = 0;
  std::string poolName;
#if ASYNCRT_WORKER_STATS
  AlignedAtomic<double> affinityEnqueueTime = 0.0f;
  AlignedAtomic<double> taskListEnqueueTime = 0.0f;
  AlignedAtomic<uint64_t> taskListEnqueueCount = 0;
  AlignedAtomic<uint64_t> affinityEnqueueCount = 0;
#endif
};
} // namespace

ThreadPoolWorkQueue::ThreadPoolWorkQueue(
    CompactCPUDevicePtr cpuDevicePtr, ArrayRef<size_t> cpuIDs,
    size_t taskListCapacity, bool mainWillDonate,
    std::chrono::microseconds threadBusyWaitTime, std::string_view poolName,
    int numaNode, size_t globalWorkerIdOffset)
    : numWorkers(cpuIDs.size()), numaNode(numaNode), partitionCpuIds(cpuIDs),
      sharedState(cpuDevicePtr, mainWillDonate, numWorkers, numaNode),
      taskList(taskListCapacity), poolName(poolName) {
  assert(numWorkers <= kMaxWorkers && "too many workers for bitvec width");

  // Keeping numWorkers in a workerGroup a power of 2 to simplify arithmetic.
  multicastFactor = numWorkers > bitVectorWidth
                        ? static_cast<size_t>(std::ceil(std::log2(
                              numWorkers / static_cast<float>(bitVectorWidth))))
                        : 0;
  // Initialize each thread with its required state.
  // Note that we're constructing the array manually since WorkQueueThreads have
  // non-moveable atomics.
  workers = static_cast<WorkQueueThread *>(M::alignedAlloc(
      alignof(WorkQueueThread), sizeof(WorkQueueThread) * numWorkers));
  assert(workers && "Allocation of workers failed");
  assert((numaNode == kAnyNumaNode) == (globalWorkerIdOffset == SIZE_MAX) &&
         "globalWorkerIdOffset must be provided iff queue is a NUMA partition");
  for (size_t localWorkerID = 0; localWorkerID < numWorkers; ++localWorkerID) {
    size_t globalWorkerId = numaNode != kAnyNumaNode
                                ? globalWorkerIdOffset + localWorkerID
                                : SIZE_MAX;
    new (workers + localWorkerID)
        WorkQueueThread(sharedState, taskList, overflowMutex, overflowTaskList,
                        localWorkerID, cpuIDs[localWorkerID],
                        threadBusyWaitTime, this->poolName, globalWorkerId);
  }

  // Set the main thread's TLS pointer for non-partition work queues. Mojo code
  // can run synchronously on the main thread, so TLS must reflect the current
  // device regardless of mainWillDonate. NUMA partition work queues skip this:
  // their parent global device sets the TLS pointer.
  if (numaNode == kAnyNumaNode)
    CompactCPUDevicePtr::setCurrentCPUDevice(cpuDevicePtr);
}

ThreadPoolWorkQueue::~ThreadPoolWorkQueue() {
// Note we can't assert state == kShutdown since queue may be created
// and destroyed without ever being included in a cpuDevice.
#if ASYNCRT_WORKER_STATS
  llvm::dbgs()
      << "affinityEnqueueTime,affinityEnqueueCount,taskListEnqueueTime,"
         "taskListEnqueueCount\n";
  llvm::dbgs() << affinityEnqueueTime << "," << affinityEnqueueCount << ","
               << taskListEnqueueTime << "," << taskListEnqueueCount << "\n";
#endif
  WorkItem workItem;
  assert(!taskList.try_dequeue(workItem) &&
         "destroying ThreadPoolWorkQueue with pending work items");

  // Clear thread-local Runtime pointer.
  CompactCPUDevicePtr::setCurrentCPUDevice({});

  // Destroy all the threads datastructures. The array came from
  // alignedAlloc, so it must be released with alignedFree.
  for (size_t i = 0; i < numWorkers; ++i)
    workers[i].~WorkQueueThread();
  M::alignedFree(workers);
}

void ThreadPoolWorkQueue::shutdown() {
  TimeTraceScope scope(InternalProfilerEntry::create("asyncrt.shutdown"));

  WorkQueueThread *callingWorker = getOwningWorkQueueThread();

  if (sharedState.mainWillDonate) {
    assert(callingWorker && callingWorker->localWorkerID == 0 &&
           "must shutdown from the 'main' thread in mainWillDonate mode");
  } else {
    assert(
        !callingWorker &&
        "must shutdown from a 'foreign' thread if not in mainWillDonate mode");
  }

  if (callingWorker) {
    // Donate this thread to help drain the work queue if there's anything left.
    callingWorker->runItemsOnOwningThread(
        /*earlyStopPredicate=*/[]() { return false; }, // Always loop
        /*lateStopPredicate=*/[]() { return false; },  // Always loop
        /*waitForTasks=*/false,
        /*spinningLabel=*/"asyncrt.shutdown.spinning",
        /*sleepingLabel=*/"asyncrt.shutdown.sleeping");
  }
  // else: the existing workers will keep processing work items until they
  // test the lateStopPredicate. This is as good a synchronization we can
  // guarantee if not in mainWillDonate mode.

  // Tell all the threads to exit.
  sharedState.doneFlag.store(true, std::memory_order_release);

  // Post on the semaphore for every thread to wake up if it is waiting.
  for (size_t i = 0; i < numWorkers; ++i)
    workers[i].sema.post();

  // Mark no threads as suspended, even though they may not have woken up,
  // cleared their own bit and exited yet.  This ensures that any in-flight
  // andThenSync calls won't try to wake these threads as we start joining and
  // tearing them down.
  sharedState.suspendedThreads.store(0);

  // Join all the threads when they shut down cleanly.
  for (size_t i = 0; i < numWorkers; ++i)
    workers[i].join();
}

void ThreadPoolWorkQueue::addTask(WorkItem &&workItem, int taskId) {
  TRACY_ZONE_SCOPED_NCT("ThreadPoolWorkQueue::addTask", TRACY_COLOR_BLUE,
                        "Unique task ID: " +
                            std::to_string(workItem.uniqueTaskId));

  assert(workItem);
#if ASYNCRT_WORKER_STATS
  auto start = std::chrono::high_resolution_clock::now();
#endif

  if (taskId >= 0) {
    auto workThread = getWorkQueueThread(taskId);
    // Either add to thread local lock-free queues or to its spill queue.
    // Any task with taskId >=0 always finds a place in either of these
    // two queues.
    workThread->addAffinityTask(std::move(workItem));
    // Wake up the thread just in case.
    // NOTE: This may be a spurious post() because the thread may already be
    // awake. It does not cause any harm because the worst that can happen
    // is that the thread goes to sleep the next iteration of runItemsImpl
    // rather than now.
    if (multicastFactor == 0) {
      if (sharedState.takeSuspendedThread(taskId))
        workThread->sema.post();
    } else {
      // TODO: post() should be low overhead if thread is already awake.
      // Nevertheless profile and check.
      workThread->sema.post();
    }
#if ASYNCRT_WORKER_STATS
    auto end = std::chrono::high_resolution_clock::now();
    atomicAdd(affinityEnqueueCount, (uint64_t)1);
    atomicAdd(affinityEnqueueTime,
              std::chrono::duration<double, std::micro>(end - start).count());
#endif
    return;
  }
  // Try to add this work to the lock-free queue.
  if (taskList.enqueue(std::move(workItem))) {
    // If there are any suspended workers, kick one of them now to make sure
    // there's at least one worker still awake to pick up work.
    int workerIDToPoke = sharedState.takeAnySuspendedThread();
    if (workerIDToPoke != -1) {
      if (multicastFactor == 0)
        getWorkQueueThread(static_cast<size_t>(workerIDToPoke))->sema.post();
      else {
        size_t start = workerIDToPoke << multicastFactor;
        size_t range = 1 << multicastFactor;
        for (size_t i = start; i < start + range; ++i) {
          if (i < numWorkers)
            getWorkQueueThread(i)->sema.post();
        }
      }
    }
#if ASYNCRT_WORKER_STATS
    auto end = std::chrono::high_resolution_clock::now();
    atomicAdd(taskListEnqueueCount, (uint64_t)1);
    atomicAdd(taskListEnqueueTime,
              std::chrono::duration<double, std::micro>(end - start).count());
#endif
    return;
  }

  // The lock-free queue is full. We now have four choices:
  //  - Run the task now on the callers stack. However, that risks overflow,
  //    and obviously would require us to give up on the 'tasks are never run
  //    immediately' API contract.
  //  - Push the task onto a local task list. That give up worker balancing,
  //    and won't work if the caller is a non-awaiting foreign thread (since
  //    the local task list is deliberately synchronization free).
  //  - Make the lock-free queue dynamically resizable. However, it's not clear
  //    how to do that without giving up its nice lock-free push and pop
  //    independence.
  //  - Push the task onto an overflow list, which we can mutex protect just
  //    like your grandfather would have written. Workers can check the
  //    overflow list when they would otherwise about to go to sleep. In this
  //    way the mutex overhead is only paid for in the uncommon case. However,
  //    we obviously risk starving these tasks.
  // We go for the last option.
  std::lock_guard<std::mutex> guard(overflowMutex);
  overflowTaskList.emplace_back(std::move(workItem));
}

void ThreadPoolWorkQueue::addLocalTask(WorkItem &&workItem) {
  TRACY_ZONE_SCOPED_NCT("ThreadPoolWorkQueue::addLocalTask", TRACY_COLOR_BLUE,
                        "Unique task ID: " +
                            std::to_string(workItem.uniqueTaskId));

  assert(workItem && "invalid work item");

  WorkQueueThread *callerWorker = getOwningWorkQueueThread();

  if (callerWorker == nullptr) {
    // Called from a foreign thread, so there's no local task list we can
    // enqueue to on this thread. Add as a task instead.
    addTask(std::move(workItem));
    return;
  }

  // Called from either a worker thread or the 'main' thread. Safe to enqueue
  // directly.
  callerWorker->addLocalTask(std::move(workItem));
}

void ThreadPoolWorkQueue::await(ArrayRef<AnyAsyncValueRef> values) {
  TRACY_ZONE_SCOPED_NC("ThreadPoolWorkQueue::await", TRACY_COLOR_BLUE);

  // If all the values are ready, then we don't have to do anything.
  if (llvm::all_of(values, [](auto &av) { return av.isReady(); }))
    return;

  // Figure out which WorkerThread this is being invoked from. This could be
  // one of our workers, the 'main' thread, or a 'foreign' thread.
  WorkQueueThread *awaitingWorker = getOwningWorkQueueThread();

  // We are done when numRemaining drops to zero.
  std::atomic<ssize_t> numRemaining = values.size();

  if (awaitingWorker) {
    // The caller is a worker or main thread, so is willing to donate itself
    // to processing work items while awaiting.

    // As each value becomes available, we can decrement our counts.  When done,
    // we signal the semaphore for this worker to make sure to wake it up if it
    // fell asleep.
    for (auto &value : values) {
      value.andThenSync([&numRemaining, awaitingWorker]() {
        // Decrement the count of async values that we're waiting on.
        // TODO: This can probably use more relaxed memory consistency!
        if (numRemaining.fetch_sub(1, std::memory_order_seq_cst) != 1)
          return;

        // When it drops to zero, we're good to go and whatever thread is
        // waiting for this will exit out of its 'runItems' loop.  That said,
        // the thread may be suspended on a semaphore.  Check for this, and if
        // so, signal its semaphore so it wakes up and notes that it is done.
        // If the worker doing the await() has suspended, make sure to wake it
        // up so it notices that it is done.
        // NOTE: This may be a spurious post() because the thread may already be
        // awake. It does not cause any harm because the worst that can happen
        // is that the thread goes to sleep the next iteration of runItemsImpl
        // rather than now.
        awaitingWorker->sema.post();
      });
    }

    // Run work items until all values are available.
    awaitingWorker->runItemsOnOwningThread(
        /*earlyStopPredicate=*/
        [&numRemaining]() {
          // Exit early as soon as numRemaining drops to zero.
          // TODO: Relaxed memory consistency!
          return numRemaining.load(std::memory_order_seq_cst) == 0;
        },
        /*lateStopPredicate=*/
        []() {
          // No additional shutdown check after waking, the early
          // check will suffice.
          return false;
        },
        /*waitForTasks=*/true,
        /*spinningLabel=*/"asyncrt.await.spinning",
        /*sleepingLabel=*/"asyncrt.await.sleeping");

  } else {
    // The caller is a 'foreign' thread. Sleep until all values are available,
    // letting the other workers do work on the caller's behalf.
    //
    // Ideally we'd sleep only until all our values are ready or the other
    // foreign thread is done with its runItems loop, whichever is sooner.
    Semaphore sema;
    // As each value becomes available, we can decrement our counts.  When done,
    // we signal the semaphore to wake up the awaiting foreign thread.
    for (auto &value : values) {
      value.andThenSync([&numRemaining, &sema]() {
        // Decrement the count of async values that we're waiting on.
        // TODO: This can probably use more relaxed memory consistency!
        if (numRemaining.fetch_sub(1, std::memory_order_seq_cst) != 1)
          return;

        sema.post();
      });
    }
    sema.wait();
  }

  assert(numRemaining.load() == 0 &&
         "exited await loop without all values being ready");
}

//===----------------------------------------------------------------------===//
// createThreadPoolWorkQueue entrypoint
//===----------------------------------------------------------------------===//

std::unique_ptr<WorkQueue> M::AsyncRT::createThreadPoolWorkQueue(
    CompactCPUDevicePtr cpuDevicePtr, size_t numThreads, size_t maxThreads,
    bool mainWillDonate, bool withAffinity,
    std::chrono::microseconds threadBusyWaitTime, std::string_view poolName) {
  // Using numThreads as a hint, figure out a CPU for each worker thread and
  // the main thread. The CPU ids may end up as kNoAffinity, but the vector
  // size will still guide the construction of worker threads.
  //
  // TODO: This function should return the error back to caller.
  if (maxThreads == 0 || maxThreads > kMaxWorkers)
    maxThreads = kMaxWorkers;
  auto cpuIDOr = getThreadAffinityCpuIds(withAffinity, numThreads, maxThreads);
  if (cpuIDOr.isError())
    llvm::report_fatal_error(cpuIDOr.getError());
  std::vector<size_t> cpuIDs = std::move(*cpuIDOr);
  assert(!cpuIDs.empty() && "no cpu ids");

  // cpuIDs.size() is guaranteed to be at least 1 here.
  const size_t taskListCapacity = cpuIDs.size() * kTaskListSlotsPerThread;
  LLVM_DEBUG(llvm::dbgs()
             << "createThreadPoolWorkQueue: Task list has capacity of at least "
             << taskListCapacity << " slots.\n");

  return std::make_unique<ThreadPoolWorkQueue>(cpuDevicePtr, cpuIDs,
                                               taskListCapacity, mainWillDonate,
                                               threadBusyWaitTime, poolName);
}

std::unique_ptr<WorkQueue> M::AsyncRT::createPartitionedThreadPoolWorkQueue(
    CompactCPUDevicePtr cpuDevicePtr, int numaNode,
    std::chrono::microseconds threadBusyWaitTime, std::string_view poolName,
    size_t globalWorkerIdOffset) {
  assert(numaNode != kAnyNumaNode &&
         "partitioned WorkQueue requires a specific NUMA node");

  const ErrorOr<NUMATopology> &topologyOr = NUMATopology::get();
  if (topologyOr.isError())
    llvm::report_fatal_error(topologyOr.getError());

  std::vector<size_t> cpuIDs = topologyOr->getCpuIdsForNumaNode(numaNode);
  if (cpuIDs.empty())
    llvm::report_fatal_error(
        "createPartitionedThreadPoolWorkQueue: no CPUs for requested NUMA "
        "node");
  assert(cpuIDs.size() <= kMaxWorkers &&
         "too many workers for partitioned WorkQueue");

  const size_t taskListCapacity = cpuIDs.size() * kTaskListSlotsPerThread;
  LLVM_DEBUG(llvm::dbgs() << "createPartitionedThreadPoolWorkQueue: NUMA node "
                          << numaNode << ", " << cpuIDs.size()
                          << " workers, task list capacity " << taskListCapacity
                          << ".\n");

  return std::make_unique<ThreadPoolWorkQueue>(
      cpuDevicePtr, cpuIDs, taskListCapacity, /*mainWillDonate=*/false,
      threadBusyWaitTime, poolName, numaNode, globalWorkerIdOffset);
}

size_t M::AsyncRT::getCurrentGlobalWorkerID() { return globalWorkerIDInTLS; }
