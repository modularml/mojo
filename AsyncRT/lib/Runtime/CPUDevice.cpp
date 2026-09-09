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
// This file implements the core AsyncRT Runtime.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/Allocator.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CompactCPUDevicePtr.h"
#include "AsyncRT/Runtime/Globals/RuntimeGlobal.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "AsyncRT/Support/Chain.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <chrono>

using namespace M;
using namespace M::AsyncRT;

void WorkQueue::vtableAnchor() {}
void Allocator::vtableAnchor() {}

/// Create "Chain" AsyncValue, making sure that "Chain" type is registered
/// before the construction. "Chain" is core to AsyncRT implementation, so it
/// needs to be registered unconditionally from AsyncRT.
static AsyncValueRef<Chain> createReadyChain(CPUDevice &cpuDevice) {
  return AsyncValueRef<Chain>::createReady(cpuDevice);
}

//===----------------------------------------------------------------------===//
// CompactCPUDevicePtr
//===----------------------------------------------------------------------===//

CompactCPUDevicePtr::CompactCPUDevicePtr(CPUDevice *cpuDevice)
    : CompactCPUDevicePtr(cpuDevice ? cpuDevice->getCompactPtr()
                                    : CompactCPUDevicePtr()) {}

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

CPUDevice *CPUDevice::getCurrentCPUDeviceOrNull() {
  // First check the thread-local Runtime pointer, set when Runtime is created
  // for WorkQueue worker threads and the thread which created the Runtime, and
  // if that's set use this.
  if (CompactCPUDevicePtr tlsRuntimePtr =
          CompactCPUDevicePtr::getCurrentCPUDevice())
    return tlsRuntimePtr.get();

  // Next check if the Runtime exists by checking the global Runtime pointer,
  // and if it does set the thread-local Runtime pointer and use that.
  CPUDevice *globalRuntimePtr = getGlobalCPUDevicePointer();
  if (globalRuntimePtr)
    CompactCPUDevicePtr::setCurrentCPUDevice(
        CompactCPUDevicePtr(globalRuntimePtr));
  return globalRuntimePtr;
}

CPUDevice::CPUDevice(CompactCPUDevicePtr cpuDevicePtr,
                     std::unique_ptr<Allocator> allocator,
                     std::unique_ptr<WorkQueue> workQueue,
                     CPUDeviceSource source, CPUDeviceType type, int numaNode,
                     StringRef profileFilename,
                     uint64_t runtimeProfilingTypeMask,
                     CPUDeviceOptions::ProfilerDebuginfo profilerDebuginfo,
                     std::map<int, CPUDevice *> &&numaDevices)
    : signature(TypeID::getSignature() ^ CompactCPUDevicePtr::getSignature()),
      allocator(std::move(allocator)), workQueue(std::move(workQueue)),
      profilerDebuginfo(profilerDebuginfo), runtimeIndex(cpuDevicePtr.index),
      source(source), type(type), numaNode(numaNode),
      readyChain(createReadyChain(*this)), numaDevices(std::move(numaDevices)) {
  switch (this->type) {
  case CPUDeviceType::kGlobal:
    assert(this->numaNode == kAnyNumaNode &&
           "kGlobal CPUDevice must not have NUMA affinity");
    assert(this->numaDevices.empty() &&
           "kGlobal CPUDevice must not own NUMA sub-devices");
    break;
  case CPUDeviceType::kGlobalPartitioned:
    assert(this->numaNode == kAnyNumaNode &&
           "kGlobalPartitioned CPUDevice must not have NUMA affinity");
    assert(!this->numaDevices.empty() &&
           "kGlobalPartitioned CPUDevice must own at least one NUMA device");
    break;
  case CPUDeviceType::kNUMAPartition:
    assert(this->numaNode != kAnyNumaNode &&
           "kNUMAPartition CPUDevice must have a valid NUMA node");
    assert(this->numaDevices.empty() &&
           "kNUMAPartition CPUDevice must not own sub-partitions");
    break;
  }

  // Establish association of cpuDevice to cpuDevice index.
  Detail::CPUDeviceTable::getSingleton().setCPUDevice(cpuDevicePtr.index, this);

  // NOTE: Users can't pass in profileFilename AND activate the time
  // profiler in the caller.
  if (!profileFilename.empty())
    profiler.emplace(/*timeTraceGranularity=*/0, "Main", profileFilename,
                     runtimeProfilingTypeMask);
}

CPUDevice::~CPUDevice() {
  // Explicitly shutdown the workQueue while the CPUDevice is still alive.
  // Shutting down the workqueue will execute unfinished tasks, and those tasks
  // can add new tasks to the cpuDevice, so we need to make sure to tie all this
  // off before invalidating the workQueue pointer.
  workQueue->shutdown();

  // Delete owned NUMA sub-devices, each partition device's destructor shuts
  // down its own work queue, draining all remaining work before this destructor
  // continues.
  for (auto &[node, device] : numaDevices)
    CPUDeviceRef::take(device);

  // Remove association of cpuDevice to cpuDevice index.
  Detail::CPUDeviceTable::getSingleton().clearCPUDevice(runtimeIndex);

  // Clear global pointer if it pointed to this cpuDevice (same pattern as
  // Context).
  clearGlobalCPUDevicePointerIfEquals(this);

  // We're done with profiling.
  if (profiler) {
    if (auto e = profiler->write("-"))
      llvm::report_fatal_error("unable to write time trace profile");
  }
}

ErrorOr<CPUDevice *> CPUDevice::getNumaDevice(int numaNode) const {
  if (type != CPUDeviceType::kGlobalPartitioned)
    return Error("Request NUMA node sub-device from non partitioned CPUDevice");

  auto it = numaDevices.find(numaNode);
  if (it != numaDevices.end())
    return it->second;
  else
    return Error("Requested NUMA node device not found");
}

std::unique_ptr<Allocator>
AsyncRT::getAllocator(const AllocatorOptions &options) {
  // Create base allocator: UseAfterFree, TCMalloc, or Malloc
  // These are mutually exclusive and must be enabled at compile time.
  std::unique_ptr<Allocator> allocator;
  if (options.useAfterFreeAllocator) {
#if HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR
    allocator = createUseAfterFreeAllocator();
#else
    llvm_unreachable("cannot use the user-after-free allocator");
#endif
  } else if (options.tcmallocAllocator) {
    allocator = createTCMallocAllocator();
  } else {
    allocator = createMallocAllocator();
  }
  // Optionally wrap in one or more debug allocators.
  if (options.leakCheckedAllocator)
    allocator = createLeakCheckAllocator(std::move(allocator));
  if (options.profilingAllocator)
    allocator = createProfilingAllocator(std::move(allocator));
  return allocator;
}
