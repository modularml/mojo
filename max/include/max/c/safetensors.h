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

#ifndef MAX_C_SAFETENSORS_H
#define MAX_C_SAFETENSORS_H

#include "max/c/symbol_export.h"
#include "max/c/types.h"

/// Loads and parses one or more Safetensors files onto a device.
///
/// The tensors from every file are flattened into a single namespace keyed by
/// tensor name. It is an error for the same tensor name to appear in more than
/// one file.
///
/// The data is loaded onto `device`. To read tensor data on the host (for
/// example, to build a weights registry with
/// `M_newWeightsRegistryFromSafetensors()`), pass a host device created with
/// `M_newDevice(M_HOST, ...)`. The runtime copies the weights to an
/// accelerator when the model is initialized.
///
/// @param paths An array of null-terminated file paths to Safetensors files.
/// @param numPaths The number of entries in `paths`.
/// @param device The device to load the tensor data onto.
/// @param status The status object for reporting errors.
///
/// @returns A pointer to the loaded Safetensors. You are responsible for the
/// memory associated with the pointer returned, which can be deallocated by
/// calling `M_freeSafetensors()`. Returns `NULL` if loading fails, with an
/// error message in `status`.
MODULAR_API_EXPORT M_Safetensors *M_loadSafetensors(const char **paths,
                                                    size_t numPaths,
                                                    M_Device *device,
                                                    M_Status *status);

/// Returns the number of tensors across all loaded Safetensors files.
///
/// @param safetensors The loaded Safetensors.
MODULAR_API_EXPORT size_t
M_getSafetensorCount(const M_Safetensors *safetensors);

/// Returns the name of the tensor at the given index.
///
/// Indices are stable for the lifetime of the `M_Safetensors` object and range
/// from `0` to `M_getSafetensorCount() - 1`. The returned string is owned by
/// the `M_Safetensors` object and must not be freed by the caller.
///
/// @param safetensors The loaded Safetensors.
/// @param index The tensor index.
///
/// @returns The null-terminated tensor name, or `NULL` if `index` is out of
/// range.
MODULAR_API_EXPORT const char *
M_getSafetensorName(const M_Safetensors *safetensors, size_t index);

/// Returns a pointer to the raw data for the named tensor.
///
/// The pointer is **borrowed**: it is valid for the lifetime of the
/// `M_Safetensors` object and must not be freed by the caller. When the
/// Safetensors were loaded onto a host device, this is a host-accessible
/// pointer.
///
/// @param safetensors The loaded Safetensors.
/// @param name The null-terminated tensor name.
/// @param status The status object for reporting errors.
///
/// @returns A pointer to the tensor data, or `NULL` if `name` is not present,
/// with an error message in `status`.
MODULAR_API_EXPORT const void *
M_getSafetensorData(const M_Safetensors *safetensors, const char *name,
                    M_Status *status);

/// Returns the size in bytes of the named tensor's data.
///
/// @param safetensors The loaded Safetensors.
/// @param name The null-terminated tensor name.
/// @param status The status object for reporting errors.
///
/// @returns The size in bytes, or `0` if `name` is not present, with an error
/// message in `status`.
MODULAR_API_EXPORT size_t M_getSafetensorNumBytes(
    const M_Safetensors *safetensors, const char *name, M_Status *status);

/// Returns the element data type of the named tensor.
///
/// @param safetensors The loaded Safetensors.
/// @param name The null-terminated tensor name.
/// @param status The status object for reporting errors.
///
/// @returns The tensor's `M_Dtype`, or `M_UNKNOWN` if `name` is not present,
/// with an error message in `status`.
MODULAR_API_EXPORT M_Dtype M_getSafetensorDtype(
    const M_Safetensors *safetensors, const char *name, M_Status *status);

/// Returns the number of dimensions (rank) of the named tensor.
///
/// @param safetensors The loaded Safetensors.
/// @param name The null-terminated tensor name.
/// @param status The status object for reporting errors.
///
/// @returns The rank, or `0` if `name` is not present, with an error message
/// in `status`. Note that a scalar tensor also has rank `0`; check `status` to
/// disambiguate.
MODULAR_API_EXPORT size_t M_getSafetensorRank(const M_Safetensors *safetensors,
                                              const char *name,
                                              M_Status *status);

/// Writes the shape of the named tensor into a caller-provided buffer.
///
/// The caller must provide a buffer with space for at least
/// `M_getSafetensorRank()` entries.
///
/// @param safetensors The loaded safetensors.
/// @param name The null-terminated tensor name.
/// @param shapeOut A buffer that receives the dimension sizes.
/// @param status The status object for reporting errors.
MODULAR_API_EXPORT void M_getSafetensorShape(const M_Safetensors *safetensors,
                                             const char *name,
                                             int64_t *shapeOut,
                                             M_Status *status);

/// Builds a weights registry mapping every loaded tensor name to its data.
///
/// This is a convenience over `M_newWeightsRegistry()` for the common case of
/// initializing a model directly from Safetensors files. The tensor names in
/// the files must match the external weight names (`constant_external`) in the
/// compiled graph; this function does not perform any name translation.
///
/// The returned registry **borrows** the tensor data from `safetensors`, which
/// must outlive both the registry and any model initialized from it.
///
/// @param safetensors The loaded Safetensors.
/// @param status The status object for reporting errors.
///
/// @returns A pointer to the weights registry, to be freed with
/// `M_freeWeightsRegistry()`. Returns `NULL` on failure, with an error message
/// in `status`.
MODULAR_API_EXPORT M_WeightsRegistry *
M_newWeightsRegistryFromSafetensors(const M_Safetensors *safetensors,
                                    M_Status *status);

/// Deallocates the memory for the loaded Safetensors. No-op if `safetensors`
/// is `NULL`.
///
/// Any pointers returned by `M_getSafetensorData()` and any weights registry
/// created by `M_newWeightsRegistryFromSafetensors()` are invalidated.
///
/// @param safetensors The Safetensors to deallocate.
MODULAR_API_EXPORT void M_freeSafetensors(M_Safetensors *safetensors);

#endif // MAX_C_SAFETENSORS_H
