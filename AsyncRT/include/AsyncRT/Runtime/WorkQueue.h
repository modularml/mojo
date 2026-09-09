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
// This file declares the M::AsyncRT::WorkQueue interface, which allows clients
// of AsyncRT to implement work queues that map onto their systems in a nice
// way.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_RUNTIME_WORKQUEUE_H
#define ASYNCRT_RUNTIME_WORKQUEUE_H

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/CompactCPUDevicePtr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/Profiling/TimeProfiler.h"
#include "Support/Threading/Atomics.h"
#include "Support/Threading/HWInfo.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"

#include <chrono>
#include <memory>
#include <vector>

/// This is the default taskId for all tasks not originating from
/// async_parallelize. We set it to -1 to indicate that the task
/// should enqueue to Global queue.
constexpr int kDefaultTaskId = -1;

namespace M::AsyncRT {

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

namespace Detail {
// Extract the result type of a function passed to addTask(Runtime, fn).
template <typename T>
struct UnwrapErrorOr {
  using type = T;
};
template <typename T>
struct UnwrapErrorOr<ErrorOr<T>> {
  using type = T;
};

template <typename F>
using ResultType = typename UnwrapErrorOr<std::invoke_result_t<F>>::type;
} // namespace Detail

//===----------------------------------------------------------------------===//
// Common types
//===----------------------------------------------------------------------===//

/// Functions to execute for a 'task'.
using TaskFunction = llvm::unique_function<void()>;

/// Profiling entries for capturing the waiting time of tasks and other
/// internal AsyncRT measurements.
using InternalProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kAsyncRT, 2), Trace::kAsyncRT>;

/// Profiling entries for capturing every execution of a task or local task.
using AllWorkItemsProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kAsyncRT, 3), Trace::kAsyncRT>;

using namespace std::chrono_literals;

/// A work item to be added, or held, by a work queue. Contains the 'task'
/// function. Depending on build type may contain extra bookkeeping data.
struct WorkItem {
  TaskFunction task;

#if defined(TRACY_ENABLE) && TRACY_VERBOSITY >= 2
  /// Assign a unique task id for the work item. This is used to identify the
  /// work item between enqueue and execution when Tracy NCT text zones are
  /// enabled.
  uint64_t uniqueTaskId = getUniqueTaskIdForWorkItem();
#else
  uint64_t uniqueTaskId = 0;
#endif // defined(TRACY_ENABLE) && TRACY_VERBOSITY >= 2

  WorkItem() = default;

  WorkItem(const WorkItem &) = delete;
  WorkItem &operator=(const WorkItem &) = delete;

  WorkItem(WorkItem &&) = default;
  WorkItem &operator=(WorkItem &&) = default;

  /// NOTE: Intentionally not marking explicit to promote from nullptr.
  WorkItem(std::nullptr_t null) {}

  /// NOTE: Intentionally not marking explicit to promote from function.
  template <typename FnTy, typename ResultTy = Detail::ResultType<FnTy>,
            std::enable_if_t<(std::is_void<ResultTy>()), int> = 0>
  WorkItem(FnTy f) : task(std::forward<FnTy>(f)) {}

  explicit operator bool() { return (bool)task; }
};

//===----------------------------------------------------------------------===//
// WorkQueue
//===----------------------------------------------------------------------===//

/// This is an interface to various implementations of work queues:
/// different execution methods which are often current. These
/// implementations may be very domain or host system specific, but the
/// interface to them is kept intentionally simple to just `addTask` (which
/// adds a block of work to be done as a C++ lambda), and `await` which runs
/// work items until some specific values are ready to go.
///
/// This is aligned to hardware_destructive_interference_size because
/// implementations of this often need that alignment, and without this the
/// destructor unique_ptr destructor is invoked incorrectly.
class alignas(hardware_destructive_interference_size) WorkQueue {
public:
  virtual ~WorkQueue() = default;

  /// Enqueue a work item for later execution, possibly on another thread.
  /// Thread-safe. The work item will NEVER be run immediately. There is no
  /// intrinsic guarantee of fairness, and the caller is responsible for
  /// using AsyncValues or other mechanisms to prevent task starvation.
  /// taskId, when >=0 indicates the thread local ring buffer to which
  /// this task needs to be enqueued. taskId = kDefaultTaskId indicates that
  /// the task will be pushed to the common taskList shared by all workers.
  /// In the current implementation, taskId gets a non-negative value only
  /// from `async_parallelize` from mojo. Every where else it should be
  /// kDefaultTaskId.
  virtual void addTask(WorkItem &&work, int taskId = kDefaultTaskId) = 0;

  /// Enqueue a work item for later execution, but on the current thread where
  /// possible. The work item will NEVER be run immediately.
  ///
  /// This method is appropriate for short running work items where the
  /// cost of thread context switching would likely dominate the cost of
  /// simply executing the block of work. For example, the AsyncValue machinery
  /// uses this method to ensure waiters are executed promptly, but off of
  /// the callers stack.
  virtual void addLocalTask(WorkItem &&workItem) = 0;

  /// Returns when the given values are ready, either as emplaced values or
  /// as errors. Depending on the WorkQueue implementation and the caller's
  /// thread, the await may sleep, may 'donate' itself to running work items,
  /// or both.
  ///
  /// It is valid for await to be called recursively, ie a task may itself
  /// call await, effectively 'blocking' it. However, that just means the
  /// task will start processing other tasks while 'waiting' for its values
  /// to become ready. Try to avoid this in favor of using synchronization
  /// via AsyncValues only.
  ///
  /// It is valid for the caller to be running on any thread, including a
  /// worker thread managed by this WorkQueue, a worker thread managed by
  /// some other WorkQueue, the 'main' thread which created the WorkQueue,
  /// or some 'foreign' thread.
  ///
  /// CAUTION: Though await will only return when all values are ready, that
  /// does NOT imply all the waiters for those values have been run (and any
  /// work triggered by those waiter have been run, and so on to quiescence).
  /// Furthermore, since await itself relies on waiters, two awaits on the
  /// same value from different threads can return in any order. Thus, care
  /// must be taken when using await to decide when a computation is 'done'
  /// and the resources it depends on can be destroyed. Generally, only a
  /// shutdown() can guarantee that all in-flight computation has completed.
  virtual void await(ArrayRef<AnyAsyncValueRef> values) = 0;

  /// Return the pool size maintained by this work queue. Kernels can use
  /// this as a hint indicating the maximum useful number of work items
  /// they should break themselves into.
  virtual size_t getParallelismLevel() const = 0;

  /// Returns the CPU IDs this work queue is partitioned to, or an empty
  /// range if the queue is not partitioned.
  virtual ArrayRef<size_t> getCpuIds() const { return {}; }

  /// Returns the NUMA node this work queue is partitioned to, or
  /// `kAnyNumaNode` if the queue is not partitioned.
  virtual int getNumaNode() const { return kAnyNumaNode; }

  /// Returns true when the caller is already executing on the worker thread
  /// implied by `taskId`, allowing immediate execution instead of enqueuing.
  /// Implementations should keep this lightweight.
  virtual bool shouldRunInlineForTask(int taskId) const { return true; }

  /// Shutdown the thread pool and quiesce in preparation for destruction.
  /// Must be called before the WorkQueue is destroyed. Must be called from
  /// outside of any task. Depending on WorkQueue implementation, may need
  /// to be called from the same thread which created the WorkQueue.
  virtual void shutdown() = 0;

protected:
  WorkQueue() = default;
  virtual void vtableAnchor();
  WorkQueue(const WorkQueue &) = delete;
  void operator=(const WorkQueue &) = delete;
};

/// Creates a thread pool that only uses the host donor thread, involving no
/// synchronization.
std::unique_ptr<WorkQueue>
createSingleThreadWorkQueue(CompactCPUDevicePtr cpuDevicePtr);

/// Creates a thread pool able to distribute the execution of work items
/// across numThreads.
///
/// If numThreads is zero it will default to a sensible number based on the
/// current physical system. The maxThreads parameter is used to bound
/// numThreads in this case. If maxThreads is zero, it is ignored.
///
/// If mainWillDonate is false then numThreads worker threads will
/// be created. Arbitrary threads may then addTasks and call await, but will not
/// themselves contribute to processing work items. This is most appropriate for
/// multi-threaded servers which wish to share the same work queue across
/// multiple request threads.
///
/// If mainWillDonate is true (currently the default) then only numThreads - 1
/// worker threads will be created, on the assumption the calling thread will
/// eventually call await and 'donate' itself to processing work items alongside
/// the worker threads. This is most appropriate for systems driven my a single,
/// distinguished main thread, such as a REPL or execution tool.
///
/// The work queue must be shutdown before being destroyed. Until shutdown has
/// returned any number of work items may be executing, so no resources they
/// depend on should be destroyed. If mainWillDonate is true, the calling
/// thread must be the one to call shutdown, at which point it may (again)
/// contribute to processing outstanding work items. Otherwise shutdown
/// may be called from any foreign thread.
std::unique_ptr<WorkQueue> createThreadPoolWorkQueue(
    CompactCPUDevicePtr cpuDevicePtr, size_t numThreads, size_t maxThreads,
    bool mainWillDonate, bool withAffinity,
    std::chrono::microseconds threadBusyWaitTime, std::string_view poolName);

/// Creates a partitioned thread pool WorkQueue whose worker threads are
/// restricted to the CPU cores of a single NUMA node. Partitioned queues must
/// always be owned by a DelegateThreadPoolWorkQueue. Each worker thread records
/// its local worker ID as well as its global worker ID (global worker ID
/// offset + local worker ID), this is used so it can be identified when
/// delegating.
std::unique_ptr<WorkQueue> createPartitionedThreadPoolWorkQueue(
    CompactCPUDevicePtr cpuDevicePtr, int numaNode,
    std::chrono::microseconds threadBusyWaitTime, std::string_view poolName,
    size_t globalWorkerIdOffset);

/// Returns the global worker ID recorded by the current thread at startup, or
/// SIZE_MAX if the current thread is not a worker of a
/// DelegateThreadPoolWorkQueue.
size_t getCurrentGlobalWorkerID();

/// Creates a WorkQueue that delegates to a set of partitioned WorkQueues,
/// emulating a single unified queue. Each delegate must be created with a
/// mutually exclusive, contiguous range of global worker IDs.
std::unique_ptr<WorkQueue>
createDelegateThreadPoolWorkQueue(CompactCPUDevicePtr cpuDevicePtr,
                                  std::vector<WorkQueue *> delegates);

} // namespace M::AsyncRT

#endif // ASYNCRT_RUNTIME_WORKQUEUE_H
