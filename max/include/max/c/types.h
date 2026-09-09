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

#ifndef MAX_C_TYPES_H
#define MAX_C_TYPES_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
extern "C" {
#else
#include <stddef.h>
#include <stdint.h>
#endif

/// Contains the success or failure of an API call.
///
/// In general, any API that may fail accepts a `M_Status` argument
/// that is filled in with a meaningful error message on failure.
///
/// You can create this with `M_newStatus()`.
/// When you're done, call `M_freeStatus()`.
typedef struct M_Status M_Status;

/// Specifies the MAX Engine configuration.
///
/// Configuration properties include the number of threads, artifact path, etc.
///
/// You can create this with `M_newRuntimeConfig()`.
/// When you're done, call `M_freeRuntimeConfig()`.
typedef struct M_RuntimeConfig M_RuntimeConfig;

/// Contains information that needs to be shared between APIs.
///
/// You can create this with `M_newRuntimeContext()`.
/// When you're done, call `M_freeRuntimeContext()`.
typedef struct M_RuntimeContext M_RuntimeContext;

/// Specifies the configuration required for model compilation.
///
/// You can create this with `M_newCompileConfig()`.
/// When you're done, call `M_freeCompileConfig()`.
typedef struct M_CompileConfig M_CompileConfig;

/// Contains an async value to a compiled model.
///
/// `M_AsyncCompiledModel` can be passed to other APIs that accept compiled
/// models as a function parameter.  This async value will eventually resolve to
/// a compiled model or an error in the case of compilation failure.
///
/// You can create this with `M_compileModel()`.
/// When you're done, call `M_freeCompiledModel()`.
typedef struct M_AsyncCompiledModel M_AsyncCompiledModel;

/// Contains a future used for inference.
///
/// The future will resolve to a model that's ready for inference.
///
/// You can create this with `M_initModel()`.
/// When you're done, call `M_freeModel()`.
typedef struct M_AsyncModel M_AsyncModel;

/// Represents all data types supported by the framework.
// This table matches the enum values in `DType` in `DType.h`. We cannot use
// DType.h directly here because DType.h is C++ code.
#ifdef __cplusplus
typedef enum M_Dtype : int {
#else
typedef enum M_Dtype {
#endif
  M_UNKNOWN = 0,

  //------ Encoding for ordinary primitives --------------------------------//

  // Bit 7 encode primitive category: 0 = Float/Other, 1 = SInt/UInt
  mIsInteger = 1 << 7,

  // Bit 6 encode for Float/Other category encodes "isFloat".
  mIsFloat = 1 << 6,

  // Bit 5 for integer and floating point types indicate if type is complex.
  // This keeps the element types densely packed, allowing table lookups.
  // Note that we support many integer and floating point element types in
  // complex number, but they must be at least a byte in size:
  //   `complex si1` is not supported (but `complex kBool` is).
  mIsComplex = 1 << 5,

  //===--- Signed and Unsigned Integer Types ----------------------------===//
  // This supports any power-of-two integer type up to a larger width than
  // MLIR supports.  The width is encoded in logarithmic form, which enables
  // small lookup tables indexed by the enum value.

  /// Bit 0 encodes "isSigned".
  mIsSigned = 1,

  kIntWidthShift = 1,
  // i1's densely packed in memory.
  M_INT1 = (0 << kIntWidthShift) | mIsInteger | mIsSigned,
  M_UINT1 = (0 << kIntWidthShift) | mIsInteger,
  M_INT2 = (1 << kIntWidthShift) | mIsInteger | mIsSigned,
  M_UINT2 = (1 << kIntWidthShift) | mIsInteger,
  M_INT4 = (2 << kIntWidthShift) | mIsInteger | mIsSigned,
  M_UINT4 = (2 << kIntWidthShift) | mIsInteger,
  M_INT8 = (3 << kIntWidthShift) | mIsInteger | mIsSigned,
  M_UINT8 = (3 << kIntWidthShift) | mIsInteger,
  M_INT16 = (4 << kIntWidthShift) | mIsInteger | mIsSigned,
  M_UINT16 = (4 << kIntWidthShift) | mIsInteger,
  M_INT32 = (5 << kIntWidthShift) | mIsInteger | mIsSigned,
  M_UINT32 = (5 << kIntWidthShift) | mIsInteger,
  M_INT64 = (6 << kIntWidthShift) | mIsInteger | mIsSigned,
  M_UINT64 = (6 << kIntWidthShift) | mIsInteger,
  M_INT128 = (7 << kIntWidthShift) | mIsInteger | mIsSigned,
  M_UINT128 = (7 << kIntWidthShift) | mIsInteger,

  //===--- Floating point types -----------------------------------------===//

  /// Bits 0 through 3 indicate the kind of FP value.
  M_FLOAT4_E2M1FN = 0 | mIsFloat,
  M_FLOAT6_E2M3FN = 1 | mIsFloat,
  M_FLOAT6_E3M2FN = 2 | mIsFloat,
  /// Some slots are left blank here to enable us to support more lower
  /// precision types in the future.
  M_FLOAT8_E8M0FNU = 9 | mIsFloat,
  M_FLOAT8_E3M4 = 10 | mIsFloat,
  M_FLOAT8_E4M3FN = 11 | mIsFloat,
  M_FLOAT8_E4M3FNUZ = 12 | mIsFloat,
  M_FLOAT8_E5M2 = 13 | mIsFloat,
  M_FLOAT8_E5M2FNUZ = 14 | mIsFloat,
  M_FLOAT16 = 15 | mIsFloat,
  M_BFLOAT16 = 16 | mIsFloat,
  M_FLOAT32 = 17 | mIsFloat,
  M_FLOAT64 = 18 | mIsFloat,

  //===--- Encodings for other types ------------------------------------===//

  // kBool != ui1.  Like it, this only contains 1-bit of data, but it occupies
  // 1-byte of storage.  The rest of the byte is guaranteed to be zeros.
  M_BOOL = 1,

} M_Dtype;

/// Contains an async value to a tensor for inference.
///
/// You can get this from `M_getTensorByNameFrom()`. When you're done, call
/// `M_freeTensor()`.
typedef struct M_AsyncTensor M_AsyncTensor;

/// Contains an array of tensor names of model inputs or outputs.
///
/// You can get this from `M_getInputNames()` and `M_getOutputNames()`.
/// When you're done, call `M_freeTensorNameArray()`.
typedef struct M_TensorNameArray M_TensorNameArray;

/// Contains the representation of a shape and an element type.
///
/// You can create this with `M_newTensorSpec()`. When you're done, call
/// `M_freeTensorSpec()`.
typedef struct M_TensorSpec M_TensorSpec;

/// Contains a collection of tensors.
///
/// The collection of tensors is used to represent inputs and outputs when
/// executing a model.
///
/// You can create this with `M_newAsyncTensorMap()`. When you're done, call
/// `M_freeAsyncTensorMap()`.
typedef struct M_AsyncTensorMap M_AsyncTensorMap;

/// Contains an `AllocatorType`. You can choose between kCaching and kSystem
/// kCaching trades off higher memory usage for better performance.
/// kSystem uses the default system allocator.
typedef enum M_AllocatorType {
  kSystem = 0,
  kCaching = 1,
} M_AllocatorType;

/// Represents the type of a value.
typedef enum M_ValueType {
  M_STRING_VALUE = 0,
  M_DOUBLE_VALUE = 1,
  M_LONG_VALUE = 2,
  M_BOOL_VALUE = 3,
  M_TENSOR_VALUE = 4,
  M_LIST_VALUE = 5,
  M_TUPLE_VALUE = 6,
  M_DICT_VALUE = 7,
  M_NONE_VALUE = 8,
  M_UNKNOWN_VALUE = 9,
  M_MOJO_VALUE = 10,
  M_PYTHON_MOJO_VALUE = 11,
} M_ValueType;

/// Maps unique weight names to their backing data.
///
/// You can create this with `M_newWeightsRegistry()`.
/// When you're done, call `M_freeWeightsRegistry()`.
typedef struct M_WeightsRegistry M_WeightsRegistry;

/// Holds the tensors parsed from one or more safetensors files.
///
/// The tensor data is borrowed from the loaded files and stays valid for the
/// lifetime of this object. You can enumerate tensors, read their data and
/// metadata, or build a `M_WeightsRegistry` from all of them with
/// `M_newWeightsRegistryFromSafetensors()`.
///
/// You can create this with `M_loadSafetensors()`.
/// When you're done, call `M_freeSafetensors()`.
typedef struct M_Safetensors M_Safetensors;

/// Represents the type of device. `M_HOST` is the CPU; `M_ACCELERATOR`
/// is any non-CPU compute device (GPU or NPU). The closed set of valid
/// device labels is enforced by `EngineContext::create` -- see
/// `Support/DeviceSpecs.h` for the label constants and
/// `max/internal/lib/API/c/EngineContext.cpp` for the validating guard.
#ifdef __cplusplus
typedef enum M_DeviceType : int {
#else
typedef enum M_DeviceType {
#endif
  M_HOST = 0,
  M_ACCELERATOR = 1,
} M_DeviceType;

/// Contains a device handle.
///
/// A device represents a computational unit (CPU or GPU) that can execute
/// operations and hold tensors.
///
/// You can create this with `M_newDevice()`. When you're done, call
/// `M_freeDevice()`.
typedef struct M_Device M_Device;

/// Represents the result output style for debug printing.
typedef enum M_ResultOutputStyle {
  M_COMPACT = 0,
  M_FULL = 1,
  M_BINARY = 2,
  M_BINARY_MAX_CHECKPOINT = 3,
  M_NONE = 4,
} M_ResultOutputStyle;

#ifdef __cplusplus
}
#endif

#endif // MAX_C_TYPES_H
