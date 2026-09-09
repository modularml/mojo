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
// Global Runtime pointer and mutex. Implemented in the RuntimeGlobals shared
// library so there is a single definition per process (ODR). M::AsyncRT
// uses these (e.g. AsyncRT::getOrCreateCPUDevice) to manage the single global
// cpuDevice.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_RUNTIME_GLOBALS_RUNTIMEGLOBAL_H
#define ASYNCRT_RUNTIME_GLOBALS_RUNTIMEGLOBAL_H

#include "Support/SymbolExport.h"

#include <mutex>

namespace M::AsyncRT {

class CPUDevice;
struct CPUDeviceOptions;

/// Returns the mutex that protects the global cpuDevice pointer.
MODULAR_CXX_EXPORT std::mutex &getGlobalCPUDeviceMutex();

/// Returns the current global cpuDevice pointer, or nullptr if none set.
/// Caller must hold getGlobalCPUDeviceMutex().
MODULAR_CXX_EXPORT CPUDevice *getGlobalCPUDevicePointer();

/// Sets the global cpuDevice pointer. Caller must hold
/// getGlobalCPUDeviceMutex().
MODULAR_CXX_EXPORT void setGlobalCPUDevicePointer(CPUDevice *ptr);

/// If the global CPUDevice pointer equals \p ptr, clears it. Called from
/// CPUDevice::~CPUDevice() when the CPUDevice is destroyed.
MODULAR_CXX_EXPORT void clearGlobalCPUDevicePointerIfEquals(CPUDevice *ptr);

/// Options used when the global cpuDevice was first created (Init path). Caller
/// must hold getGlobalCPUDeviceMutex() when reading or writing.
MODULAR_CXX_EXPORT CPUDeviceOptions &getStoredGlobalCPUDeviceCreationOptions();

} // namespace M::AsyncRT

#endif // ASYNCRT_RUNTIME_GLOBALS_RUNTIMEGLOBAL_H
