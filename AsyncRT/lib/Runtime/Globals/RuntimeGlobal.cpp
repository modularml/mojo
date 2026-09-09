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

#include "AsyncRT/Runtime/Globals/RuntimeGlobal.h"
#include "AsyncRT/Runtime/CPUDevice.h"

#include <mutex>

namespace M::AsyncRT {

namespace {

static std::mutex &getGlobalCPUDeviceMutexImpl() {
  static std::mutex m;
  return m;
}

static CPUDevice *&getGlobalCPUDevicePtrImpl() {
  static CPUDevice *ptr = nullptr;
  return ptr;
}

static CPUDeviceOptions &storedGlobalCPUDeviceCreationOptionsImpl() {
  static CPUDeviceOptions opts;
  return opts;
}

} // namespace

std::mutex &getGlobalCPUDeviceMutex() { return getGlobalCPUDeviceMutexImpl(); }

CPUDevice *getGlobalCPUDevicePointer() { return getGlobalCPUDevicePtrImpl(); }

void setGlobalCPUDevicePointer(CPUDevice *ptr) {
  getGlobalCPUDevicePtrImpl() = ptr;
}

void clearGlobalCPUDevicePointerIfEquals(CPUDevice *ptr) {
  std::lock_guard<std::mutex> lock(getGlobalCPUDeviceMutexImpl());
  if (getGlobalCPUDevicePtrImpl() == ptr) {
    getGlobalCPUDevicePtrImpl() = nullptr;
  }
}

CPUDeviceOptions &getStoredGlobalCPUDeviceCreationOptions() {
  return storedGlobalCPUDeviceCreationOptionsImpl();
}

} // namespace M::AsyncRT
