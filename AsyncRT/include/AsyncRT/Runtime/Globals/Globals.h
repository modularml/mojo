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

#ifndef ASYNCRT_RUNTIME_GLOBALS_H
#define ASYNCRT_RUNTIME_GLOBALS_H

#include "Support/SymbolExport.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

namespace M::AsyncRT {
class CPUDevice;
class CompactCPUDevicePtr;

namespace Detail {
class CPUDeviceTable;
} // namespace Detail

class Globals {

public:
  /// This is a TLS CompactCPUDevicePtr pointing to the cpuDevice on behalf of
  /// which the thread is processing work items. That thread may be a 'worker'
  /// thread of the cpuDevice's work queue, or a 'main' thread which is also
  /// donating itself to processing work items for the cpuDevice.
  ///
  /// NOTE: MSVC does not allow a thread_local to have DLL linkage, so we must
  /// hide this under a function.
  static MODULAR_CXX_EXPORT CompactCPUDevicePtr &getCurrentCPUDeviceInTLS();

  static MODULAR_CXX_EXPORT Detail::CPUDeviceTable &getCPUDeviceTableSingleton(
      const std::function<Detail::CPUDeviceTable *()> &ctor);

private:
  friend class AsyncValue;
  /// This is a global counter of the number of AsyncValue instances currently
  /// live in the process.  This is intended to be used for debugging only, and
  /// is only kept in sync if `isAllocationTrackingEnabled()` returns true.
  static MODULAR_CXX_EXPORT std::atomic<ssize_t> totalAllocatedAsyncValues;
};

// TCMalloc has internal global state that needs to live here in AsyncRTGlobals.
// Since we are using a hacked version of TCMalloc that doesn't replace malloc,
// we want to limit the scope of these functions to the TCMallocAllocator class.
struct TCMallocGlobals {
  static MODULAR_CXX_EXPORT void *tc_new(size_t alignment, size_t size);
  /// Allocate from a specific TCMalloc NUMA partition (0 or 1).
  static MODULAR_CXX_EXPORT void *tc_new(size_t alignment, size_t size,
                                         size_t numaPartition);
  static MODULAR_CXX_EXPORT void tc_delete(void *ptr);
};

MODULAR_CXX_EXPORT std::string getRuntimeGlobalsBinaryID();

/// Get a unique task id for identifying work items.
/// Thread-safe and globally unique across all AsyncRT users.
MODULAR_CXX_EXPORT uint64_t getUniqueTaskIdForWorkItem();

} // namespace M::AsyncRT

#endif // ASYNCRT_RUNTIME_GLOBALS_H
