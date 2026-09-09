# `M::AsyncRT::WorkQueue`

This document introduces the `M::AsyncRT::WorkQueue`, key design points and how
to use it.

## Overview

The `M::AsyncRT::WorkQueue` is an abstract interface for executing work items
concurrently. It is the core abstraction for managing CPU parallelism in
AsyncRT, providing a thread pool that distributes tasks across available CPU
cores.

### Creating a WorkQueue

A WorkQueue is created through factory functions rather than direct
construction:

- **`createSingleThreadWorkQueue`**: Creates a WorkQueue that only uses the
  calling (donor) thread with no synchronization overhead. Useful for
  single-threaded platforms.

- **`createThreadPoolWorkQueue`**: Creates a multi-threaded WorkQueue with the
  following parameters:
  - `numThreads`: Number of worker threads. If 0, defaults based on system
    configuration.
  - `maxThreads`: Upper bound for `numThreads` when auto-detecting.
  - `mainWillDonate`: If true, the creating thread will participate in work
    processing during `await()` calls.
  - `withAffinity`: If true, workers are pinned to specific CPU cores.
  - `threadBusyWaitTime`: Duration to spin before sleeping when idle (default
    1ms).
  - `poolName`: Prefix for thread names (visible in debuggers/profilers).

### Thread Types

The WorkQueue distinguishes between three types of threads:

1. **Worker threads**: Threads created by the WorkQueue that run a dedicated
   work-processing loop. Each worker has a unique `workerID` (0 to N-1).

2. **Main thread**: If `mainWillDonate` is true, the thread that created the
   WorkQueue is designated as the "main" thread (workerID 0). It participates
   in work processing during `await()` and must be the one to call
   `shutdown()`.

3. **Foreign threads**: Any other thread that interacts with the WorkQueue.
   Foreign threads may call `addTask()` and `await()` but do not donate
   themselves to processing work items.

### Worker Allocation and CPU Affinity

When `withAffinity` is enabled, worker threads are pinned to specific CPU cores:

1. **Default thread count**: If `numThreads` is 0:
   - On systems with P-cores and E-cores: uses the number of performance cores.
   - With affinity enabled: uses the number of physical cores.
   - Without affinity: uses the number of logical cores (including
     hyperthreads).

2. **CPU selection**: The `CPUSystemInfo::getPreferredCpuIDs()` function
   determines which CPUs to use, typically preferring:
   - Performance cores over efficiency cores.
   - Physical cores over hyperthreads (when affinity is set).
   - Cores within a single NUMA node when possible.

3. **Cgroup limits**: In containerized environments, thread count is
   automatically capped based on CPU limits (millicores / 1000).

4. **Affinity setting**: Each worker thread calls `setThreadAffinity(cpuID)`
   at startup to bind thread execution to the specified CPU core and set the
   memory policy to prefer memory allocation on the NUMA node that CPU core
   resides in if possible.

### Task Queues

The WorkQueue uses a hierarchy of task queues to balance efficiency with work
distribution:

1. **Local task list** (`localTaskList`): A per-worker list with no
   synchronization. Used for `addLocalTask()` calls from the owning thread.
   Work items here take highest priority. Ideal for short-running continuations
   (e.g., AsyncValue waiters) where context-switch overhead would dominate.

2. **Affinity task list** (`affinityTaskList`): A per-worker lock-free ring
   buffer. Used when `addTask()` is called with a non-negative `taskId`
   (typically from `async_parallelize` in Mojo). Tasks are processed by the
   specific worker indicated by `taskId`, enabling cache-friendly execution
   patterns.

3. **Global task list** (`taskList`): A lock-free MPMC queue shared by all
   workers. Used for `addTask()` with `taskId = kDefaultTaskId` (-1). Any
   worker can dequeue and process these tasks.

4. **Overflow task list** (`overflowTaskList`): A mutex-protected fallback
   queue used when the global task list is full. Workers check this before
   going to sleep.

Work items are processed in priority order: local → affinity → global →
overflow.

### Ownership and Lifecycle

The `WorkQueue` is typically owned by an `M::AsyncRT::CPUDevice` instance, which
creates and manages it based on `CPUDeviceOptions`. The lifecycle is:

1. **Creation**: Via `createThreadPoolWorkQueue()` or
   `createSingleThreadWorkQueue()`. Worker threads start immediately.

2. **Usage**: Clients add work via `addTask()` / `addLocalTask()` and wait for
   results via `await()`.

3. **Shutdown**: Must call `shutdown()` before destruction. This:
   - Drains remaining work items (main thread helps if `mainWillDonate`).
   - Sets the done flag to signal workers to exit.
   - Posts all worker semaphores to wake sleeping threads.
   - Joins all worker threads.

4. **Destruction**: After `shutdown()` returns, the WorkQueue can be destroyed.

### Idle Behavior and Sleep/Wake

When a worker has no tasks to process:

1. **Busy-wait phase**: Spins with exponential backoff for `busyWaitTime`
   (default 1ms), checking for new tasks.

2. **Overflow check**: Before sleeping, pumps any overflow/spill queues into
   the main queues.

3. **Sleep**: Marks itself as suspended in a shared bitvector and waits on its
   per-worker semaphore.

4. **Wake**: When `addTask()` sees suspended workers, it posts the appropriate
   semaphore(s) to wake them.

For systems with more than 64 worker threads, a multicast scheme groups workers
together in the suspension bitvector, waking all workers in a group when any
might be suspended.

### Key Design Principles

- **Non-blocking assumption**: Work items should not block. See
  [WorkQueueNonblocking.md](WorkQueueNonblocking.md) for rationale and
  strategies.

- **No immediate execution**: `addTask()` never runs work inline; tasks are
  always deferred. This prevents stack overflow and ensures predictable
  behavior.

- **Thread donation**: `await()` from a worker/main thread donates that thread
  to process work items while waiting, avoiding deadlock and maximizing CPU
  utilization.

## Pinning Device Tasks

When executing kernels that interact with GPUs or other accelerators, it's
beneficial to run those tasks on consistent CPU threads that are pinned to CPU
cores in the same NUMA node as the device. This reduces PCIe/interconnect
latency for GPU command submission and improves cache locality for any
CPU-side work associated with the device.

### Why Pin Device Tasks?

1. **NUMA locality**: GPUs are connected to specific PCIe buses, which belong
   to specific NUMA nodes. Running device tasks on CPUs in the same NUMA node
   minimizes memory access latency and maximizes PCIe bandwidth.

2. **Consistent thread affinity**: GPU driver contexts (CUDA contexts, HIP
   contexts) often have thread-local state. Running device operations from a
   consistent thread avoids context switching overhead in the driver.

3. **Predictable scheduling**: Pinning device tasks to specific workers
   ensures that GPU-bound work doesn't compete with CPU-bound work for the
   same threads.

### How Device Task Pinning Works

When `mgp_generic_execute` runs a kernel that references an accelerator
`DeviceContext`, it determines a `taskId` that routes the task to a specific
worker thread. The `taskId` selection follows this priority order:

#### 1. Explicit Configuration (Highest Priority)

Users can explicitly specify which CPU cores should handle each device via the
`runtime.device_task_cpu_ids` configuration option:

**Environment variable:**

```bash
export MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS="0,32,1,33,2,34,3,35"
```

**modular.cfg:**

```ini
[runtime]
device_task_cpu_ids = 0,32,1,33,2,34,3,35
```

The list is indexed by device ID. For example, with the above configuration:

- Device 0 → CPU 0 (worker whose `cpuID` is 0)
- Device 1 → CPU 32
- Device 2 → CPU 1
- etc.

This is useful when automatic NUMA detection doesn't work correctly or when
you want fine-grained control over the mapping.

#### 2. Automatic NUMA Topology Detection

If no explicit configuration is provided and the work queue has at least as
many workers as physical CPU cores, the runtime attempts to infer the optimal
CPU for each GPU:

1. Query `NUMATopology::get()` to discover the system's NUMA layout.
2. For each GPU, get its PCI bus address via `device->getPciBusId()`.
3. Map the PCI bus to a NUMA node via `NUMATopology::getNumaNodeForPciBus()`.
4. Get the list of CPU IDs in that NUMA node via `getCpuIdsForNumaNode()`.
5. Select the first available CPU in that NUMA node for this device.

This mapping is computed once at startup and cached in a static
`gpuToCpuCoreMapping`.

#### 3. Fallback: Round-Robin Assignment (Lowest Priority)

If NUMA topology detection fails or the thread pool is constrained (e.g., by
cgroups), the runtime falls back to a simple round-robin assignment:

```cpp
taskId = 1 + (deviceHint % (numWorkers - 1))
```

This deliberately skips worker 0 to avoid potential stalls when
`mainWillDonate` is true (worker 0 is the main thread, which may not always
be processing work items).

### Inline vs. Out-of-Line Execution

Once a `taskId` is determined, the runtime decides whether to execute the
kernel inline or dispatch it to the affinity queue:

- **Sync kernel with no device affinity** (`taskId == kDefaultTaskId`): Run
  inline on the current thread.

- **Sync kernel with device affinity**: Check `shouldRunInlineForTask(taskId)`:
  - If already on the correct worker → run inline.
  - Otherwise → dispatch to that worker's affinity queue and wait.

- **Async kernel**: Always dispatch to the affinity queue (or global queue if
  no affinity) for out-of-line execution.

The `shouldRunInlineForTask()` check is lightweight—it compares the current
thread's `workerID` (stored in thread-local storage) against the target
`taskId`.

### Tuning Device Task Pinning

The `utils/benchmarking/tools/autotune_gpu_numa` tool can help determine the
optimal CPU-to-GPU mapping for your system. It benchmarks GPU kernel launch
latency from each NUMA node and outputs a recommended configuration.

This tool does have a dependency on the Python `click` module, so if you will
need to ensure this is installed first.

```bash
bazelw run //utils/benchmarking/tools/autotune_gpu_numa:autotune_gpu_numa
```

This script will infer the most optimal mapping of GPUs to CPU NUMA nodes, and
therefore CPU cores based on the results of a GPU kernel launch benchmark. The
results are presented and instructions are provided as to how to use this either
in `modular.cfg` or the environment variable.

This is particularly useful for multi-GPU systems where the default NUMA
detection may not yield optimal results due to complex PCIe topologies.
