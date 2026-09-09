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
// This file exposes a basic set of command line options for setting up and
// configuring an M::AsyncRT::CPUDevice for tools to use.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_RUNTIME_RUNTIMECLOPTIONS_H
#define ASYNCRT_RUNTIME_RUNTIMECLOPTIONS_H

#include "AsyncRT/Runtime/CPUDevice.h"
#include "Support/ADT/GenericUniquePtrSet.h"
#include "Support/CommandLine.h"
#include "Support/Profiling/TimeProfiler.h"
#include "Support/RCRef.h"
#include "llvm/Support/Threading.h"
#include <chrono>
#include <thread>
#include <type_traits>

namespace M::AsyncRT {

class CPUDevice;

/// Contains a number of command-line options that are shared among binaries
/// that use the AsyncRT Runtime and want configurability of Allocator,
/// WorkQueue, stopping behavior, etc.
///
class RuntimeCLOptions {
public:
  CPUDeviceOptions &options;
  RuntimeCLOptions(CPUDeviceOptions &o) : options(o) {}

private:
  llvm::cl::OptionCategory CPUDeviceOptionsCategory{
      "CPUDevice command line options"};
  //===--------------------------------------------------------------------===//
  // Core Runtime configuration.
  //===--------------------------------------------------------------------===//
  // Enable HostAllocator types to be specified on the command line.
  M::cl::MOpt<CPUDeviceOptions::WorkQueueType, true> workQueueType{
      "workqueue", llvm::cl::desc("Specify workqueue type:"),
      llvm::cl::values(
          clEnumValN(CPUDeviceOptions::WorkQueueType::kSingleThread,
                     "single-thread",
                     "Work queue that only ever uses one thread"),
          clEnumValN(CPUDeviceOptions::WorkQueueType::kThreadPool,
                     "thread-pool",
                     "Default threaded work queue based on std::thread")),
      llvm::cl::location(options.workQueueType),
      llvm::cl::cat(CPUDeviceOptionsCategory)};

  // Enable HostAllocator types to be specified on the command line.
  M::cl::MOpt<CPUDeviceOptions::AllocatorType, true> allocatorType{
      "allocator", llvm::cl::desc("Specify allocator type:"),
      llvm::cl::values(

          clEnumValN(CPUDeviceOptions::AllocatorType::kMalloc, "malloc",
                     "System malloc/free"),
          clEnumValN(CPUDeviceOptions::AllocatorType::kTCMalloc, "tcmalloc",
                     "TCMalloc new/delete. Not available on all targets"),
          clEnumValN(CPUDeviceOptions::AllocatorType::kLeakChecker,
                     "leak-checker", "Allocator with leak checking"),
          clEnumValN(CPUDeviceOptions::AllocatorType::kProfiler, "profiler",
                     "Allocator with profiling and leak checking"),
          clEnumValN(CPUDeviceOptions::AllocatorType::kUseAfterFree,
                     "use-after-free",
                     "Allocator to detect use-after-free errors. Not available "
                     "on all targets.")),
      llvm::cl::location(options.allocatorType),
      llvm::cl::cat(CPUDeviceOptionsCategory)};

  // Specify the number of threads in the thread pool. Only consulted when
  // the selected workqueue type is `kThreadPool`. The default number of
  // threads is the result of M::getNumThreads().
  M::cl::MOpt<size_t, true> numThreads{
      "num-threads",
      llvm::cl::desc("Specify the number of threads to run the work queue "
                     "items. If zero "
                     "(default), will be chosen by heuristics."),
      llvm::cl::location(options.numThreads),
      llvm::cl::cat(CPUDeviceOptionsCategory)};
  M::cl::MOpt<size_t, true> maxThreads{
      "max-threads",
      llvm::cl::desc("Bound num-threads in the case of auto-configuration."),
      llvm::cl::location(options.maxThreads),
      llvm::cl::cat(CPUDeviceOptionsCategory)};

  // Specify the amount of time a worker thread should spin for before
  // sleeping. The optimal value here depends on the system latency for thread
  // sleep and wake-up, as well as other external factors like how many other
  // threadpools are sharing the system.
  M::cl::MOpt<size_t, true> threadBusyWaitTime{
      "thread-busy-wait-time-us",
      llvm::cl::desc(
          "Specify the number of microseconds for threads to spin before "
          "locking. Zero indicates that threads should never spin."),
      llvm::cl::location(options.threadBusyWaitTime),
      llvm::cl::cat(CPUDeviceOptionsCategory)};

  // Specify whether the workqueue should be created using thread affinity.
  // Can also be controlled via MODULAR_ENABLE_AFFINITY environment variable.
  M::cl::MOpt<bool, true> cpuAffinity{
      "cpu-affinity",
      llvm::cl::desc(
          "Assign CPU affinity to threads within the work queue. "
          "This flag takes precedence over the MODULAR_ENABLE_AFFINITY "
          "environment variable."),
      llvm::cl::location(options.withAffinity),
      llvm::cl::cat(CPUDeviceOptionsCategory)};

  // Filename to hold the time profiling output (as JSON text).
  M::cl::MOpt<std::string, true> profileFilename{
      "time-profile",
      llvm::cl::desc(
          kIsProfilingEnabled
              ? "Specify the filename base for profiling output. The tracing "
                "data will be written to a file called "
                "\"<base>.time-trace\". "
                "This will be a JSON text in the standard profiling format. "
                "The events will be written to a text file called "
                "\"<base>.time-events.csv\". An empty filename base disables "
                "profiling (the default)."
              : "Specify the filename base for profiling output. WARNING: "
                "This "
                "option is ignored in this build. Rebuild with "
                "MODULAR_ASYNCRT_MAX_PROFILING_LEVEL greater than 0 to enable "
                "it."),
      llvm::cl::location(options.profileFilename),
      llvm::cl::cat(CPUDeviceOptionsCategory)};

  // Should we generate debuginfo for a profiler?
  M::cl::MOpt<CPUDeviceOptions::ProfilerDebuginfo, true> profilerDebuginfo{
      "profiler-debuginfo",
      llvm::cl::desc(
          "Output debug symbols in a way that a profiler can understand. After "
          "running under perf, use `perf inject --jit -i perf.data -o "
          "perf.jit.data` to add debug info for kernels to the profile."),
      llvm::cl::values(
          clEnumValN(CPUDeviceOptions::ProfilerDebuginfo::kNoProfiler, "none",
                     "Do not generate debuginfo"),
          clEnumValN(CPUDeviceOptions::ProfilerDebuginfo::kPerfProfiler, "perf",
                     "Generate debuginfo for perf."),
          clEnumValN(CPUDeviceOptions::ProfilerDebuginfo::kSOProfiler, "so",
                     "Generate debuginfo by loading compiled kernels into a "
                     "shared library. Should work with all profilers.")),
      llvm::cl::location(options.profilerDebuginfo),
      llvm::cl::cat(CPUDeviceOptionsCategory)};
};

} // namespace M::AsyncRT

#endif // ASYNCRT_RUNTIME_RUNTIMECLOPTIONS_H
