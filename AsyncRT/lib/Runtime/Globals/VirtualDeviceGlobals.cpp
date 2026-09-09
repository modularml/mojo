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

#include "AsyncRT/Runtime/Globals/VirtualDeviceGlobals.h"

#include <atomic>
#include <mutex>

namespace M::AsyncRT {

// Global count of virtual devices for device creation.
// When set to a positive value, Device::create() will return VirtualDevice
// instead of real hardware devices for GPU APIs, and Device::numberOfDevices()
// will return this count. Using atomic for thread-safe access.
static std::atomic<int> g_virtualDeviceCount{0};

MODULAR_CXX_EXPORT void setVirtualDeviceCount(int count) {
  g_virtualDeviceCount = count;
}

MODULAR_CXX_EXPORT int getVirtualDeviceCount() { return g_virtualDeviceCount; }

MODULAR_CXX_EXPORT bool isVirtualDeviceMode() {
  return g_virtualDeviceCount > 0;
}

// Global API for virtual device compilation.
// This specifies which API to use (e.g., "cuda", "hip", "metal") for
// virtual devices created in compile-only mode.
// Using function-local static for thread-safe lazy initialization.
static std::string &getVirtualDeviceAPIImpl() {
  static std::string api;
  return api;
}
static std::mutex &getVirtualDeviceAPIMutex() {
  static std::mutex mutex;
  return mutex;
}

MODULAR_CXX_EXPORT void setVirtualDeviceAPI(StringRef api) {
  std::lock_guard<std::mutex> lock(getVirtualDeviceAPIMutex());
  getVirtualDeviceAPIImpl() = api.str();
}

MODULAR_CXX_EXPORT std::string getVirtualDeviceAPI() {
  std::lock_guard<std::mutex> lock(getVirtualDeviceAPIMutex());
  return getVirtualDeviceAPIImpl();
}

// Global target architecture for virtual device compilation.
// This specifies the GPU architecture (e.g., "sm_80", "sm_90") for
// virtual devices created in compile-only mode.
// Using function-local static for thread-safe lazy initialization.
static std::string &getVirtualDeviceTargetArchImpl() {
  static std::string arch;
  return arch;
}
static std::mutex &getVirtualDeviceTargetArchMutex() {
  static std::mutex mutex;
  return mutex;
}

MODULAR_CXX_EXPORT void setVirtualDeviceTargetArch(StringRef arch) {
  std::lock_guard<std::mutex> lock(getVirtualDeviceTargetArchMutex());
  getVirtualDeviceTargetArchImpl() = arch.str();
}

MODULAR_CXX_EXPORT std::string getVirtualDeviceTargetArch() {
  std::lock_guard<std::mutex> lock(getVirtualDeviceTargetArchMutex());
  return getVirtualDeviceTargetArchImpl();
}

} // namespace M::AsyncRT
