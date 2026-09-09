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

#include "AsyncRT/Runtime/RuntimeCLOptions.h"
#include "AsyncRT/Runtime/CPUDevice.h"

using namespace M::AsyncRT;

bool CPUDeviceOptions::operator==(const CPUDeviceOptions &other) const {
  return numThreads == other.numThreads && maxThreads == other.maxThreads &&
         profileFilename == other.profileFilename &&
         runtimeProfilingTypeMask == other.runtimeProfilingTypeMask &&
         mainWillDonate == other.mainWillDonate &&
         threadBusyWaitTime == other.threadBusyWaitTime &&
         withAffinity == other.withAffinity &&
         leakCheckedAllocator == other.leakCheckedAllocator &&
         tcmallocAllocator == other.tcmallocAllocator &&
         profilingAllocator == other.profilingAllocator &&
         useAfterFreeAllocator == other.useAfterFreeAllocator &&
         workQueueType == other.workQueueType &&
         numaPartitioned == other.numaPartitioned &&
         allocatorType == other.allocatorType &&
         profilerDebuginfo == other.profilerDebuginfo;
}

CPUDeviceOptions CPUDeviceOptions::copy() const {
  CPUDeviceOptions cpuDeviceOptions;
  switch (allocatorType) {
  case CPUDeviceOptions::AllocatorType::kMalloc:
    cpuDeviceOptions.tcmallocAllocator = false;
    break;
  case CPUDeviceOptions::AllocatorType::kTCMalloc:
    cpuDeviceOptions.tcmallocAllocator = true;
    break;
  case CPUDeviceOptions::AllocatorType::kLeakChecker:
    cpuDeviceOptions.leakCheckedAllocator = true;
    break;
  case CPUDeviceOptions::AllocatorType::kProfiler:
    cpuDeviceOptions.profilingAllocator = true;
    break;
  case CPUDeviceOptions::AllocatorType::kUseAfterFree:
#if HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR
    cpuDeviceOptions.useAfterFreeAllocator = true;
#else
    llvm::errs() << "The use-after-free allocator is not available for this "
                    "target. Using the leak-checker cpuDevice instead.";
    cpuDeviceOptions.leakCheckedAllocator = true;
#endif
    break;
  }
  cpuDeviceOptions.workQueueType = workQueueType;
  switch (workQueueType) {
  case CPUDeviceOptions::WorkQueueType::kSingleThread:
    break;
  case CPUDeviceOptions::WorkQueueType::kThreadPool:
    cpuDeviceOptions.numThreads = numThreads;
    cpuDeviceOptions.maxThreads = maxThreads;
    cpuDeviceOptions.withAffinity = withAffinity;
    cpuDeviceOptions.threadBusyWaitTime = threadBusyWaitTime;
    break;
  }
  cpuDeviceOptions.profileFilename = getProfileFilename();
  cpuDeviceOptions.profilerDebuginfo = profilerDebuginfo;
  cpuDeviceOptions.poolName = poolName;
  cpuDeviceOptions.mainWillDonate = mainWillDonate;
  return cpuDeviceOptions;
}
