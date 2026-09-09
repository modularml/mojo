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

#include "AsyncRT/Runtime/HostSystem.h"
#include "AsyncRT/Runtime/Globals/RuntimeGlobal.h"
#include "Support/Threading/HWInfo.h"

#include "llvm/Support/ErrorHandling.h"

#include <map>
#include <mutex>
#include <string>

namespace M::AsyncRT {

CPUDeviceRef getOrCreateCPUDevice(CPUDeviceSource source,
                                  const CPUDeviceOptions &options,
                                  bool allowUsingExistingOptions) {
  std::lock_guard<std::mutex> lock(getGlobalCPUDeviceMutex());
  CPUDevice *existingCPUDevice = getGlobalCPUDevicePointer();
  if (existingCPUDevice) {
    if (getStoredGlobalCPUDeviceCreationOptions() != options &&
        !allowUsingExistingOptions)
      llvm::report_fatal_error("AsyncRT::getOrCreateCPUDevice called "
                               "requesting different options to "
                               "those used to create the existing CPUDevice.");
    return CPUDeviceRef::copy(existingCPUDevice);
  }

  assert(CPUDevice::getCurrentCPUDeviceOrNull() == nullptr &&
         "creating a CPUDevice from a thread already associated with an outer "
         "CPUDevice");
  CompactCPUDevicePtr cpuDevicePtr = CompactCPUDevicePtr::reserve();
  std::unique_ptr<Allocator> allocator =
      getAllocator(options.getAllocatorOptions());
  std::unique_ptr<WorkQueue> workQueue;
  std::map<int, CPUDevice *> numaDevices;
  CPUDeviceType topLevelType = CPUDeviceType::kGlobal;
  switch (options.workQueueType) {
  case CPUDeviceOptions::WorkQueueType::kSingleThread:
    workQueue = createSingleThreadWorkQueue(cpuDevicePtr);
    break;
  case CPUDeviceOptions::WorkQueueType::kThreadPool:
    if (options.numaPartitioned) {
      // For each NUMA node, create a separate CPUDevice with its own
      // CompactCPUDevicePtr and partitioned ThreadPoolWorkQueue. The top-level
      // CPUDevice owns these NUMA node partitioned devices; its
      // DelegateThreadPoolWorkQueue has pointers to the delegate
      // ThreadPoolWorkQueues.
      const ErrorOr<NUMATopology> &topologyOr = NUMATopology::get();
      if (topologyOr.isError())
        llvm::report_fatal_error(topologyOr.getError());
      const std::vector<int> &numaNodes = topologyOr->getNumaNodes();
      std::vector<WorkQueue *> delegateWorkQueues;
      delegateWorkQueues.reserve(numaNodes.size());
      size_t globalWorkerIdOffset = 0;
      auto busyWait = std::chrono::microseconds(options.threadBusyWaitTime);
      for (size_t i = 0; i < numaNodes.size(); ++i) {
        std::string partitionName =
            options.poolName + " (NUMA " + std::to_string(i) + ")";
        CompactCPUDevicePtr numaDevicePtr = CompactCPUDevicePtr::reserve();
        std::unique_ptr<Allocator> partitionAllocator =
            getAllocator(options.getAllocatorOptions());
        std::unique_ptr<WorkQueue> partitionWorkQueue =
            createPartitionedThreadPoolWorkQueue(numaDevicePtr, numaNodes[i],
                                                 busyWait, partitionName,
                                                 globalWorkerIdOffset);
        globalWorkerIdOffset += partitionWorkQueue->getParallelismLevel();
        delegateWorkQueues.push_back(partitionWorkQueue.get());
        numaDevices[numaNodes[i]] =
            new CPUDevice(numaDevicePtr, std::move(partitionAllocator),
                          std::move(partitionWorkQueue), source,
                          CPUDeviceType::kNUMAPartition, numaNodes[i]);
      }
      workQueue = createDelegateThreadPoolWorkQueue(
          cpuDevicePtr, std::move(delegateWorkQueues));
      topLevelType = CPUDeviceType::kGlobalPartitioned;
    } else {
      workQueue = createThreadPoolWorkQueue(
          cpuDevicePtr, options.numThreads, options.maxThreads,
          options.mainWillDonate, options.withAffinity,
          std::chrono::microseconds(options.threadBusyWaitTime),
          options.poolName);
    }
    break;
  }
  CPUDeviceRef newCPUDevice = CPUDeviceRef::take(
      new CPUDevice(cpuDevicePtr, std::move(allocator), std::move(workQueue),
                    source, topLevelType, kAnyNumaNode, options.profileFilename,
                    options.runtimeProfilingTypeMask, options.profilerDebuginfo,
                    std::move(numaDevices)));

  getStoredGlobalCPUDeviceCreationOptions() = options;

  // Initialise the NUMA topology, to be used later when creating allocators and
  // work-queues.
  (void)NUMATopology::get();

  setGlobalCPUDevicePointer(newCPUDevice.getPointer());
  return newCPUDevice.copy();
}

} // namespace M::AsyncRT
