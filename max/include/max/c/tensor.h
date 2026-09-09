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

#ifndef MAX_C_TENSOR_H
#define MAX_C_TENSOR_H

#include "max/c/device.h"
#include "max/c/symbol_export.h"
#include "max/c/types.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
#include <cstdint>
#endif

/// Creates a tensor specification.
///
/// You need this in order to set the input tensors with `M_borrowTensorInto()`.
///
/// When storing tensor data in memory, we always use a diminishing stride size.
/// That is, earlier dimensions in the shape have larger strides than later
/// dimensions. For example, a C array declared as `int arr[1][2][3]` would have
/// a shape specified as `{1, 2, 3}`.
///
/// @param shape The shape of the tensor.
/// @param rankSize The rank size of the tensor.
/// @param dtype The datatype for the tensor.
/// @param tensorName The name for the tensor.  This string gets copied as part
/// of the operation of `M_newTensorSpec`, so your original string need not
/// remain valid after the completion of this call.
/// @param device The device on which the tensor resides.
///
/// @returns A pointer to the tensor spec.  You are responsible for the memory
/// associated with the pointer returned.  The memory can be deallocated by
/// calling `M_freeTensorSpec()`.
MODULAR_API_EXPORT M_TensorSpec *
M_newTensorSpec(const int64_t *shape, int64_t rankSize, M_Dtype dtype,
                const char *tensorName, const M_Device *device);

/// Returns if the given spec has a dynamic rank.
///
/// @param spec The tensor spec.
///
/// @return `1` if the rank is dynamic. `0` otherwise.
MODULAR_API_EXPORT int M_isDynamicRanked(const M_TensorSpec *spec);

/// Gets the element at a particular axis.
///
/// @param spec The tensor spec.
/// @param axis The requested axis
/// @return The dimension at requested axis if the spec and axis
/// are valid and has static rank.  Otherwise, `0`. A dimension equaling
/// `kDynamicDimensionValue` indicates dynamic dimension e.g. batch-size
/// of a model expecting a batched tensor.
MODULAR_API_EXPORT int64_t M_getDimAt(const M_TensorSpec *spec, size_t axis);

/// Gets the rank from the tensor spec.
///
/// @param spec The tensor spec.
///
/// @return The number of dimensions in the tensor spec if the spec is static
/// and valid, `kDynamicRankValue` if dynamic. Otherwise, `0`.
MODULAR_API_EXPORT int64_t M_getRank(const M_TensorSpec *spec);

/// Gets the datatype from the tensor spec.
///
/// @param spec The tensor spec.
///
/// @return The element type from the tensor spec if the tensor spec is valid.
/// Otherwise, `M_UNKNOWN`.
MODULAR_API_EXPORT M_Dtype M_getDtype(const M_TensorSpec *spec);

/// Gets the name of the tensor from the tensor spec.
///
/// @param spec The tensor spec.
///
/// @return A null-terminated string containing the name of the tensor if the
/// `spec` is valid.  Otherwise, `NULL`.
/// The memory associated with the returned string is owned by `spec`.
MODULAR_API_EXPORT const char *M_getName(M_TensorSpec *spec);

/// Creates a map of tensor names to async tensors.
///
/// @param context The runtime context.
///
/// @return A pointer to the tensor map.  You are responsible for the memory
/// associated with the pointer returned. The memory can be deallocated by
/// calling `M_freeAsyncTensorMap()`.
MODULAR_API_EXPORT M_AsyncTensorMap *
M_newAsyncTensorMap(const M_RuntimeContext *context);

/// Adds a tensor to the tensor map.
///
/// You are responsible for the lifetime of the input tensor data.  Its data
/// gets "borrowed" into the Tensor Map.
///
/// When `tensorSpec` targets an accelerator, the data is borrowed in place
/// (zero-copy) only if `input` already resides on that device; a host pointer
/// is instead staged onto the device via a copy. In the staged case, in-place
/// mutations the model makes to its device input are not reflected back into
/// your host pointer.
///
/// @param tensors The tensor map, from `M_newAsyncTensorMap()`.
/// @param input The input tensor data.
/// @param tensorSpec The tensor spec, from `M_newTensorSpec()`.  This gets
/// copied as part of the operation of `M_borrowTensorInto`, so your original
/// tensorSpec need not exist through the lifetime of the tensor map.
/// @param status The status object for reporting errors.
MODULAR_API_EXPORT void M_borrowTensorInto(M_AsyncTensorMap *tensors,
                                           void *input,
                                           const M_TensorSpec *tensorSpec,
                                           M_Status *status);

/// Gets a tensor from the tensor map by name.
///
/// @param tensorMap The tensor map.
/// @param name The name of the tensor.
/// @param status The status object for reporting errors.
///
/// @return A pointer to the tensor.  You are responsible for the memory
/// associated with the pointer returned. The memory can be deallocated by
/// calling `M_freeTensor()`.  The held tensor inside the return value is simply
/// borrowed from the corresponding input `M_AsyncTensorMap`.  If the tensor
/// map or name are invalid, a `NULL` pointer is returned and the `status`
/// parameter contains an error message.
MODULAR_API_EXPORT M_AsyncTensor *
M_getTensorByNameFrom(M_AsyncTensorMap *tensorMap, const char *name,
                      M_Status *status);

/// Gets the number of elements for the tensor.
///
/// @param tensor The tensor which must not be `NULL`.
///
/// @return The number of elements for the given tensor.
MODULAR_API_EXPORT size_t M_getTensorNumElements(const M_AsyncTensor *tensor);

/// Gets the corresponding `M_Dtype` for the tensor.
///
/// @param tensor The tensor which must not be `NULL`.
///
/// @return The corresponding `M_Dtype` for the tensor.
MODULAR_API_EXPORT M_Dtype M_getTensorType(const M_AsyncTensor *tensor);

/// Gets a pointer to underlying data of the tensor.
///
/// @param tensor The tensor which must not be `NULL`.
///
/// @return A pointer to the underlying data of the tensor.  This pointer is
/// valid for the lifetime of the underlying tensor.
MODULAR_API_EXPORT const void *M_getTensorData(const M_AsyncTensor *tensor);

/// Gets a Tensor Spec for the tensor.
///
/// @param tensor The tensor.
///
/// @return The tensor spec for the tensor if the tensor is valid.  Otherwise,
/// `NULL`.
MODULAR_API_EXPORT M_TensorSpec *M_getTensorSpec(const M_AsyncTensor *tensor);

/// Gets the device type from a tensor specification.
///
/// @param spec The tensor spec.
///
/// @return The device type (CPU or GPU).
MODULAR_API_EXPORT M_DeviceType
M_getDeviceTypeFromSpec(const M_TensorSpec *spec);

/// Gets the device ID from a tensor specification.
///
/// @param spec The tensor spec.
///
/// @return The device ID. Returns `0` if the spec is invalid.
MODULAR_API_EXPORT int M_getDeviceIdFromSpec(const M_TensorSpec *spec);

/// Gets the device on which a tensor resides.
///
/// @param tensor The tensor.
///
/// @return The device on which the tensor resides, or `NULL` if the tensor is
/// invalid. The caller owns the returned device and must free it with
/// `M_freeDevice()`.
MODULAR_API_EXPORT M_Device *M_getTensorDevice(const M_AsyncTensor *tensor);

/// Copies a tensor to a different device.
///
/// Creates a copy of the tensor on the specified device.
///
/// @param tensor The tensor to copy.
/// @param device The target device.
/// @param status The status object for reporting errors.
///
/// @returns A pointer to the tensor on the target device. The caller owns the
/// returned memory and must deallocate it by calling `M_freeTensor()`. Returns
/// `NULL` if the operation fails, with an error message in the status.
MODULAR_API_EXPORT M_AsyncTensor *
M_copyTensorToDevice(M_AsyncTensor *tensor, M_Device *device, M_Status *status);

/// Deallocates the memory for the tensor.  No-op if `tensor` is NULL.
///
/// @param tensor The tensor to deallocate.
MODULAR_API_EXPORT void M_freeTensor(M_AsyncTensor *tensor);

/// Deallocates the memory for the array of tensor names.  No-op if `names` is
/// `NULL`.
///
/// @param names The tensor names to deallocate.
MODULAR_API_EXPORT void M_freeTensorNameArray(M_TensorNameArray *names);

/// Deallocates the memory for the tensor spec.  No-op if `spec` is `NULL`.
///
/// @param spec The tensor spec to deallocate.
MODULAR_API_EXPORT void M_freeTensorSpec(M_TensorSpec *spec);

/// Deallocates the memory for the tensor map.  No-op if `tensorMap` is `NULL`.
///
/// @param tensorMap The tensor map to deallocate.
MODULAR_API_EXPORT void M_freeAsyncTensorMap(M_AsyncTensorMap *tensorMap);

#endif // MAX_C_TENSOR_H
