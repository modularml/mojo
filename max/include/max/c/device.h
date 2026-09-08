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

#ifndef MAX_C_DEVICE_H
#define MAX_C_DEVICE_H

#include "max/c/symbol_export.h"
#include "max/c/types.h"

/// Creates a device of the specified type and ID.
///
/// For HOST devices, the ID is typically 0.
/// For ACCELERATOR devices, the ID specifies which GPU to use (0 for first
/// ACCELERATOR, etc.).
///
/// @param type The type of device to create (M_HOST or M_ACCELERATOR).
/// @param id The device ID. Use 0 for the default device of that type.
/// @param status The status object for reporting errors.
///
/// @returns A pointer to the device. You are responsible for the memory
/// associated with the pointer returned. The memory can be deallocated by
/// calling `M_freeDevice()`. Returns `NULL` if the device could not be created,
/// with an error message in the status.
MODULAR_API_EXPORT M_Device *M_newDevice(M_DeviceType type, int id,
                                         M_Status *status);

/// Gets the type of a device.
///
/// @param device The device.
///
/// @returns The device type (M_HOST or M_ACCELERATOR).
MODULAR_API_EXPORT M_DeviceType M_getDeviceType(const M_Device *device);

/// Gets the ID of a device.
///
/// @param device The device.
///
/// @returns The device ID.
MODULAR_API_EXPORT int M_getDeviceId(const M_Device *device);

/// Checks if the device is a host device.
///
/// @param device The device.
///
/// @returns `1` if the device is the host, `0` otherwise.
MODULAR_API_EXPORT int M_isHostDevice(const M_Device *device);

/// Synchronizes the device, ensuring all operations complete.
///
/// This blocks until all pending operations on the device have completed.
///
/// @param device The device to synchronize.
/// @param status The status object for reporting errors.
MODULAR_API_EXPORT void M_synchronizeDevice(M_Device *device, M_Status *status);

/// Gets the label of a device (e.g., "cpu" or "gpu").
///
/// @param device The device.
///
/// @returns A string containing the device label. The returned string is owned
/// by the device and should not be freed. Returns `NULL` if the device is
/// invalid.
MODULAR_API_EXPORT const char *M_getDeviceLabel(const M_Device *device);

/// Deallocates the memory for a device. No-op if `device` is `NULL`.
///
/// @param device The device to deallocate.
MODULAR_API_EXPORT void M_freeDevice(M_Device *device);

/// Returns the number of available accelerator (GPU) devices.
///
/// @returns The number of accelerator devices available on the system.
MODULAR_API_EXPORT int M_getAcceleratorCount(void);

#endif // MAX_C_DEVICE_H
