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
// This file declares top level "god object" that organizes AsyncRT thread pool,
// memory allocator, etc.
//
// This header file is intended to be a low-dependency header that other things
// compose on top of.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_CPUDEVICE_H
#define ASYNCRT_CPUDEVICE_H

#include "AsyncRT/Runtime/Allocator.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CompactCPUDevicePtr.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "AsyncRT/Support/Chain.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "Support/STLExtras.h"
#include "Support/StringExtras.h"
#include "Support/Threading/HWInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Process.h"

#include <atomic>
#include <map>
#include <memory>
#include <string>

namespace M {
class Error;
} // namespace M

namespace M::AsyncRT {
class Allocator;
class WorkQueue;

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

struct AllocatorOptions {
  bool leakCheckedAllocator = false;
  bool tcmallocAllocator = true;
  bool profilingAllocator = false;
  bool useAfterFreeAllocator = false;
};

/// Collects all the options which influence a cpuDevice.
struct CPUDeviceOptions {
  enum class AllocatorType {
    /// Allocator that just calls malloc/free.
    kMalloc,
    /// Allocator that calls into tcmalloc.
    kTCMalloc,
    /// Allocator that does leak checking.
    kLeakChecker,
    /// Allocator that does profiling (and leak checking).
    kProfiler,
    /// Allocator that read/write protects every freed block
    /// to detect use-after-free errors without ASAN. Nat available
    /// on all targets.
    kUseAfterFree,
  };

  enum class WorkQueueType {
    /// Workqueue that only ever uses one thread.
    kSingleThread,
    /// Default thread pool that uses std::thread and semaphores.
    kThreadPool,
  };

  enum class ProfilerDebuginfo {
    /// No debug info generated.
    kNoProfiler,
    /// Generating debug info for Linux `perf`.
    kPerfProfiler,
    /// Generate debug info by loading kernels as a shared library. Should work
    /// will all profilers.
    kSOProfiler
  };

  size_t numThreads = 0;
  size_t maxThreads = 0;

  /// Filepath to write profile to, which enables profiling only if set.
  std::string profileFilename;

  /// Runtime configurable filter for profiling types (`Trace::Type`).
  /// Currently this only takes "type" into account and ignores "level".
  /// So any non-zero value enables the level, in other words `11111` and
  /// `22222` and `12121` all have the same effect. Set this in Runtime's ctor
  /// via CPUDeviceOptions.runtimeProfilingTypeMask.
  ///
  /// For example:
  ///
  /// AsyncRT::CPUDeviceOptions rtOpt;
  /// rtOpt.runtimeProfilingTypeMask = 1 << Trace::typeBitshift(Trace::kOther);
  /// auto rt = AsyncRT::getOrCreateCPUDevice(AsyncRT::CPUDeviceSource::Test,
  /// rtOpt);
  ///
  /// Creates a Runtime that will only record `kOther` type events.
  uint64_t runtimeProfilingTypeMask = Trace::kFullyEnabled;

  bool mainWillDonate = true;
  // TODO arekay - revert to time units
  //  std::chrono::microseconds threadBusyWaitTime = 200us;
  size_t threadBusyWaitTime = []() -> size_t {
    if (auto env = llvm::sys::Process::GetEnv("MODULAR_THREAD_BUSY_WAIT_US")) {
      size_t value;
      if (!llvm::StringRef(*env).getAsInteger(10, value))
        return value;
    }
    return 200;
  }();
  // Affinity is disabled by default due to performance issues with multiple
  // processes. Can be enabled by MODULAR_ENABLE_AFFINITY environment variable,
  // which in turn can be overridden by --cpu-affinity CLI flag.
  bool withAffinity = []() {
    auto env = llvm::sys::Process::GetEnv("MODULAR_ENABLE_AFFINITY");
    return env.has_value() && M::isTrueLike(*env);
  }();
  std::string poolName = "🔥 Thread";
  bool leakCheckedAllocator = false;
  bool tcmallocAllocator = true;
  bool profilingAllocator = false;
  bool useAfterFreeAllocator = false;
  WorkQueueType workQueueType{CPUDeviceOptions::WorkQueueType::kThreadPool};
  /// When true and workQueueType is kThreadPool, creates a partitioned
  /// WorkQueue per NUMA node and wraps them in a DelegateThreadPoolWorkQueue.
  bool numaPartitioned = false;

  AllocatorType allocatorType{
#ifdef MODULAR_DEBUG
      CPUDeviceOptions::AllocatorType::kLeakChecker
#else
      CPUDeviceOptions::AllocatorType::kMalloc
#endif
  };

  ProfilerDebuginfo profilerDebuginfo = ProfilerDebuginfo::kNoProfiler;

  CPUDeviceOptions() = default;

  StringRef getProfileFilename() const {
    if constexpr (!kIsProfilingEnabled) {
      if (!profileFilename.empty())
        llvm::errs()
            << "WARNING: The --time-profile option was given but this build"
               " does not support profiling. Rebuild with "
               "MODULAR_ASYNCRT_MAX_PROFILING_LEVEL greater than 0 to enable "
               "it.\n";
      return "";
    }
#ifdef MODULAR_DEBUG
    if (!profileFilename.empty())
      llvm::errs()
          << "WARNING: Using the --time-profile option in debug mode is"
             " not recommended due to increased overhead. Please use"
             " a release build.\n";
#endif
    return profileFilename;
  }

  // Temporary shim, remove once we separate the Allocator from the Runtime
  // Extract the Allocator-specific options from the CPUDeviceOptions into a
  // new struct.
  AllocatorOptions getAllocatorOptions() const {
    return {leakCheckedAllocator, tcmallocAllocator, profilingAllocator,
            useAfterFreeAllocator};
  }

  /// Print information about the cpuDevice configuration to standard out.
  void printRuntimeConfig() const {
    printf("cpuDevice using ");
    switch (allocatorType) {
    case CPUDeviceOptions::AllocatorType::kMalloc:
      printf("malloc");
      break;
    case CPUDeviceOptions::AllocatorType::kTCMalloc:
      printf("tcmalloc");
      break;
    case CPUDeviceOptions::AllocatorType::kLeakChecker:
      printf("leak check");
      break;
    case CPUDeviceOptions::AllocatorType::kProfiler:
      printf("profiling");
      break;
    case CPUDeviceOptions::AllocatorType::kUseAfterFree:
      printf("use-after-free");
      break;
    }
    printf(" allocator, and ");
    switch (workQueueType) {
    case CPUDeviceOptions::WorkQueueType::kSingleThread:
      printf("single thread work queue");
      break;
    case CPUDeviceOptions::WorkQueueType::kThreadPool:
      printf("thread pool work queue");
      break;
    }

    switch (numThreads) {
    case 0:
      printf(" with autosensed threads.\n");
      break;
    default:
      printf(" with %d thread%s.\n", (int)numThreads, &"s"[numThreads == 1]);
      break;
    }
  }

  /// Return the number of threads specified at the command-line.
  size_t getNumThreads() const { return numThreads; }

  /// Equality for all fields that affect cpuDevice behavior.
  bool operator==(const CPUDeviceOptions &other) const;
  bool operator!=(const CPUDeviceOptions &other) const {
    return !(*this == other);
  }

  /// Create a copy of the CPUDeviceOptions.
  CPUDeviceOptions copy() const;

  CPUDeviceOptions &forDebug() {
    workQueueType = WorkQueueType::kSingleThread;
    leakCheckedAllocator = true;
    return *this;
  }

  CPUDeviceOptions &withMainWillNotDonate(bool mainWillNotDonate = true) {
    this->mainWillDonate = !mainWillNotDonate;
    return *this;
  }

  CPUDeviceOptions &withCPUAffinity(bool cpuAffinity = true) {
    this->withAffinity = cpuAffinity;
    return *this;
  }

  CPUDeviceOptions &
  withLeakCheckedAllocator(bool newLeakCheckedAllocator = true) {
    leakCheckedAllocator = newLeakCheckedAllocator;
    return *this;
  }

  CPUDeviceOptions &withTCMallocAllocator(bool newTcmallocAllocator = true) {
    tcmallocAllocator = newTcmallocAllocator;
    return *this;
  }

  CPUDeviceOptions &withProfilingAllocator(bool newProfilingAllocator = true) {
    profilingAllocator = newProfilingAllocator;
    return *this;
  }

  CPUDeviceOptions &withSingleThreaded(bool singleThreaded = true) {
    workQueueType = singleThreaded ? WorkQueueType::kSingleThread
                                   : WorkQueueType::kThreadPool;
    return *this;
  }

  CPUDeviceOptions &withNumThreads(size_t newNumThreads) {
    numThreads = newNumThreads;
    return *this;
  }

  CPUDeviceOptions &withMaxThreads(size_t newMaxThreads) {
    maxThreads = newMaxThreads;
    return *this;
  }

  CPUDeviceOptions &withProfileFilename(StringRef newProfileFilename) {
    profileFilename = newProfileFilename;
    return *this;
  }
};

/// Indicates the type of CPUDevice with regards to partitioning.
enum class CPUDeviceType {
  /// Top-level device with no NUMA partitioning.
  kGlobal,
  /// Top-level device with per-NUMA-node partitioning.
  kGlobalPartitioned,
  /// Per-NUMA-node partition owned by a kGlobalPartitioned CPUDevice.
  kNUMAPartition,
};

/// Indicates how a CPUDevice was created, for diagnostics and tracing.
enum class CPUDeviceSource {
  /// Created by M::Context.
  MaxContext,
  /// Created by Mojo stdlib / CompilerRT.
  MojoStdlib,
  /// Created for CPU device context.
  CPUDeviceContext,
  /// Created for unit tests, benchmarks, or test harnesses.
  Test,
};

/// This represents one instance of the AsyncRT cpuDevice, which can have
/// multiple threads, a private heap for data, a way of reporting errors, and
/// other context objects. This is also the natural unit for task cancellation.
///
/// Runtime is reference-counted so that CPUDeviceRef can be RCRef<CPUDevice>
/// and support shared ownership. It inherits ReferenceCounted and must only be
/// destroyed via dropRef().
class CPUDevice final : public M::ReferenceCounted<CPUDevice> {
public:
  /// Construct cpuDevice with the already reserved cpuDevicePtr, and already
  /// created allocator and workQueue. The work queue must have been constructed
  /// with the same cpuDevicePtr.
  ///
  /// \p source indicates how a CPUDevice was created (for diagnostics).
  /// If profileFilename is non-empty then time profiling will be activated
  /// and the profile JSON and text will be written to files with that prefix.
  CPUDevice(CompactCPUDevicePtr cpuDevicePtr,
            std::unique_ptr<Allocator> allocator,
            std::unique_ptr<WorkQueue> workQueue, CPUDeviceSource source,
            CPUDeviceType type, int numaNode = kAnyNumaNode,
            StringRef profileFilename = {},
            uint64_t runtimeProfilingTypeMask = Trace::kFullyEnabled,
            CPUDeviceOptions::ProfilerDebuginfo profilerDebuginfo =
                CPUDeviceOptions::ProfilerDebuginfo::kNoProfiler,
            std::map<int, CPUDevice *> &&numaDevices = {});
  ~CPUDevice();

  /// How this device was created (for diagnostics).
  CPUDeviceSource getSource() const { return source; }

  /// Returns the type of device with regards to partitioning.
  CPUDeviceType getType() const { return type; }

  /// Return a CompactCPUDevicePtr that identifies this Runtime instance.
  CompactCPUDevicePtr getCompactPtr() const {
    return CompactCPUDevicePtr(runtimeIndex);
  }

  /// Return a reference to a pre-allocated Chain value that is already ready.
  /// This can be used by logic that needs to flag that a side effect has
  /// already happened, without doing an extraneous memory allocation.
  const AsyncValueRef<Chain> &getReadyChain() const { return readyChain; }

  /// Returns the cpuDevice managing the work queue to which the callers thread
  /// is associated (ie the callers thread is either a worker thread for that
  /// cpuDevice or is a 'main' thread which has donated itself to running work
  /// items on behalf of the cpuDevice). If no cpuDevice has been associated
  /// with this thread but a global cpuDevice exists, automatically associates
  /// this thread with it. Returns null only if no global cpuDevice exists at
  /// all.
  static CPUDevice *getCurrentCPUDeviceOrNull();

  //===--------------------------------------------------------------------===//
  // Profiling
  //===--------------------------------------------------------------------===//

  /// Return a reference to the profiler instance, if its been initialized.
  std::optional<TimeTraceProfiler> &getProfiler() { return profiler; }

  /// Which profiler should we generate debug information for.
  CPUDeviceOptions::ProfilerDebuginfo getProfilerDebuginfo() const {
    return profilerDebuginfo;
  }

  //===--------------------------------------------------------------------===//
  // Memory Management
  //===--------------------------------------------------------------------===//

  /// Get direct access to the low level allocator.
  Allocator *getAllocator() { return allocator.get(); }

  /// Returns the current cpuDevice allocator. This assumes that a global
  /// allocator is present and would assert otherwise.
  static Allocator *getCurrentAllocator() {
    auto rt = CPUDevice::getCurrentCPUDeviceOrNull();
    assert(
        rt &&
        "a global cpuDevice must be set before getting the current allocator");
    return rt->getAllocator();
  }

  //===--------------------------------------------------------------------===//
  // Concurrency
  //===--------------------------------------------------------------------===//

  /// Get direct access to the low level WorkQueue.  You should typically
  /// interface with the higher level algorithms in Algorithms.h.
  WorkQueue *getWorkQueue() { return workQueue.get(); }

  /// Returns the NUMA node this device is targeting if the type is
  /// kNUMAPartition, otherwise returns kAnyNumaNode.
  int getNumaNode() const { return numaNode; }

  /// Returns the number of NUMA node partitioned sub-devices owned by this
  /// device if the type is kGlobalPartitioned, otherwise returns 0.
  size_t numNumaDevices() const { return numaDevices.size(); }

  /// Returns a non-owning pointer to the NUMA node partitioned device for
  /// \p numaNode owned by this device if the type is kGlobalPartitioned or
  /// an error if no such device exists or the type is not kGlobalPartitioned.
  ErrorOr<CPUDevice *> getNumaDevice(int numaNode) const;

private:
  CPUDevice(const CPUDevice &) = delete;
  void operator=(const CPUDevice &) = delete;

  /// The 'signature' for the type id registration system the cpuDevice depends
  /// on. This is expected to be unique for the running process. This can be
  /// used to catch, at runtime, accidental multiple definitions for Modular
  /// cpuDevice statics across dynamic libraries / executables.
  intptr_t signature;

  /// These are the allocator and workQueue's that were configured by the client
  /// for this Runtime.
  std::unique_ptr<Allocator> allocator;
  std::unique_ptr<WorkQueue> workQueue;

  /// An active profiler used for the cpuDevice, or nullopt if profiling is
  /// disabled. This is only set when profileFilename is non-empty.
  std::optional<TimeTraceProfiler> profiler;

  /// Should the cpuDevice output debug info for `perf`.
  CPUDeviceOptions::ProfilerDebuginfo profilerDebuginfo;

  /// This is the index # for the cpuDevice object created.  This is held by the
  /// CompactCPUDevicePtr.
  uint8_t runtimeIndex;

  /// How this device was created (for diagnostics).
  CPUDeviceSource source;

  /// Topological role of this device.
  CPUDeviceType type;

  /// NUMA node this device is targeting if the type is kNUMAPartition,
  /// otherwise kAnyNumaNode.
  int numaNode;

  /// This is a preallocated Chain value that is marked as ready, for use by
  /// getReadyChain.
  AsyncValueRef<Chain> readyChain;

  /// NUMA node partitioned sub-devices owned by this device if the type is
  /// kGlobalPartitioned, keyed by NUMA node ID.
  std::map<int, CPUDevice *> numaDevices;

  friend void checkUniqueCPUDevice(const CPUDevice &cpuDevice);
};

//===----------------------------------------------------------------------===//
// Runtime construction
//===----------------------------------------------------------------------===//

/// Creates a suitable allocator given the options.
std::unique_ptr<Allocator>
getAllocator(const AllocatorOptions &options = AllocatorOptions());

using CPUDeviceRef = M::RCRef<CPUDevice>;

//===----------------------------------------------------------------------===//
// Debugging helpers
//===----------------------------------------------------------------------===//

/// In debug builds, assert the given cpuDevice's 'signature' agrees with what
/// the host's idea of signature for its dynamic library / executable.
/// This can be used to catch, at runtime, accidental multiple definitions for
/// Modular cpuDevice statics across dynamic libraries / executables.
inline void checkUniqueCPUDevice(const CPUDevice &cpuDevice) {
  assert(cpuDevice.signature ==
             (TypeID::getSignature() ^ CompactCPUDevicePtr::getSignature()) &&
         "It appears your process has statically linked the Modular Runtime "
         "multiple times across dynamic library / executable boundaries. "
         "Please don't do that.");
}

} // namespace M::AsyncRT

#endif // ASYNCRT_CPUDEVICE_H
