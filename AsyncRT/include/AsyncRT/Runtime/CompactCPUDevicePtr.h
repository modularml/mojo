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
// A `Runtime*` encoded in 8 bits.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_RUNTIME_COMPACT_CPU_DEVICE_PTR_H
#define ASYNCRT_RUNTIME_COMPACT_CPU_DEVICE_PTR_H

#include "AsyncRT/Runtime/Globals/Globals.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <mutex>

namespace M::AsyncRT {

class CPUDevice;

namespace Detail {

//===----------------------------------------------------------------------===//
// CPUDeviceTable
//===----------------------------------------------------------------------===//

/// Global singleton which maintains the cpuDevice index to CPUDevice map.
class CPUDeviceTable {
public:
  /// Returns CPUDevice with given index, which must have already been added or
  /// registered.
  CPUDevice *getCPUDevice(uint8_t index) const;

  /// Reserves an index for a CPUDevice, returning the index. The actual
  /// CPUDevice must be set by setCPUDevice() below once known.
  uint8_t reserveIndex();

  /// Sets the CPUDevice for the already reserved index.
  void setCPUDevice(uint8_t index, CPUDevice *cpuDevice);

  /// Unregisters the CPUDevice with the given index.
  void clearCPUDevice(uint8_t index);

  /// Returns the number of active CPUDevices.
  size_t numActiveCPUDevices() const;

  /// Index representing 'no CPUDevice'.
  static constexpr uint8_t kInvalidIndex = 255;

  static CPUDeviceTable &getSingleton() {
    return Globals::getCPUDeviceTableSingleton(
        []() { return new CPUDeviceTable(); });
  }

private:
  CPUDeviceTable();

  /// Protects mutation to both of the following fields.
  mutable std::mutex mu;
  llvm::SmallVector<uint8_t, 256> freeIndices;
  CPUDevice *allCPUDevices[kInvalidIndex];
};

} // namespace Detail

//===----------------------------------------------------------------------===//
// CompactCPUDevicePtr
//===----------------------------------------------------------------------===//

/// The `CompactCPUDevicePtr` type provides a pointer compressed version of
/// `Runtime*` that fits in 8 bits.  This allows every AsyncValue to carry a
/// backpointer to the Runtime which allocated it, and allows deallocating the
/// memory for the AsyncValue through the Runtime's allocator.
class CompactCPUDevicePtr {
public:
  constexpr CompactCPUDevicePtr() = default;
  CompactCPUDevicePtr(const CompactCPUDevicePtr &) = default;
  CompactCPUDevicePtr &operator=(const CompactCPUDevicePtr &) = default;

  static CompactCPUDevicePtr reserve() {
    return CompactCPUDevicePtr(
        Detail::CPUDeviceTable::getSingleton().reserveIndex());
  }

  // Implicitly convert Runtime* to CompactCPUDevicePtr.
  /*implicit*/ CompactCPUDevicePtr(CPUDevice *cpuDevice);
  /*implicit*/ CompactCPUDevicePtr(CPUDevice &cpuDevice)
      : CompactCPUDevicePtr(&cpuDevice) {}

  CPUDevice *operator->() const { return get(); }
  CPUDevice &operator*() const { return *get(); }
  CPUDevice *get() const {
    return Detail::CPUDeviceTable::getSingleton().getCPUDevice(index);
  }

  CPUDevice *getOrNull() const {
    return index == Detail::CPUDeviceTable::kInvalidIndex
               ? nullptr
               : Detail::CPUDeviceTable::getSingleton().getCPUDevice(index);
  }

  /// Explicitly testing for truth value determines whether this pointer is
  /// "null".
  explicit operator bool() const {
    return index != Detail::CPUDeviceTable::kInvalidIndex;
  }

  bool operator==(CompactCPUDevicePtr that) const {
    return index == that.index;
  }

  /// We implicitly convert to Runtime& since we are used interchangeably with
  /// it.
  /*implicit*/ operator CPUDevice &() const { return *get(); }

  /// Returns a 'signature' for the CompactCPUDevicePtr subsystem which is
  /// expected to be unique for the running process. This can be used to catch,
  /// at runtime, accidental multiple definitions for Modular cpuDevice
  /// statics across dynamic libraries / executables.
  ///
  /// (This is just the address of the underlying cpuDevice table, but
  /// please don't depend on that.)
  static intptr_t getSignature() {
    return reinterpret_cast<intptr_t>(&Detail::CPUDeviceTable::getSingleton());
  }

  /// Returns the CompactCPUDevicePtr to the Runtime which is managing the
  /// caller's thread. Returns the invalid CompactCPUDevicePtr if no such
  /// cpuDevice has been associated.
  static CompactCPUDevicePtr getCurrentCPUDevice() {
    return Globals::getCurrentCPUDeviceInTLS();
  }

  /// Sets the thread-local CPUDevice pointer, providing it hasn't already been
  /// set to another CPUDevice.
  static void setCurrentCPUDevice(CompactCPUDevicePtr ptr) {
    CompactCPUDevicePtr current = getCurrentCPUDevice();

    /// Invariant: There should only ever be one CPUDevice alive at once so the
    // thread-local CPUDevice pointer should never be overwritten by a different
    // CPUDevice.
    bool willSet = !current || !ptr || current == ptr;
    if (willSet) {
      Globals::getCurrentCPUDeviceInTLS() = ptr;
    } else {
      assert(
          false &&
          "The thread-local CPUDevice pointer should never be overwritten by a "
          "different CPUDevice.");
    }
  }

private:
  friend class CPUDevice;

  explicit CompactCPUDevicePtr(uint8_t index) : index{index} {
    assert(index < Detail::CPUDeviceTable::kInvalidIndex &&
           "Too many Runtime instances created");
  }
  uint8_t index = Detail::CPUDeviceTable::kInvalidIndex;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_RUNTIME_COMPACT_CPU_DEVICE_PTR_H
