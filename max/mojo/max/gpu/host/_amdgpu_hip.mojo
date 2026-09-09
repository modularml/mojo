# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from std.ffi import external_call

from max.gpu.host import DeviceContext, DeviceFunction, DeviceStream
from max.gpu.host.device_context import (
    _CString,
    _checked,
    _DeviceContextPtr,
    _DeviceFunctionPtr,
    _DeviceStreamPtr,
)


struct _ihipDevice_t:
    pass


struct _ihipStream_t:
    pass


struct _ihipModule_t:
    pass


comptime hipDevice_t = OptionalPointer[_ihipDevice_t, UntrackedOrigin[mut=True]]
comptime hipStream_t = OptionalPointer[_ihipStream_t, UntrackedOrigin[mut=True]]
comptime hipModule_t = OptionalPointer[_ihipModule_t, UntrackedOrigin[mut=True]]


# Accessor function to get access to the underlying hipDevice_t from an abstract DeviceContext.
# Use `var hip_dev: hipDevice_t = HIP(ctx)` where ctx is a `DeviceContext` to get access to the
# underlying hipDevice_t.
@always_inline
def HIP(ctx: DeviceContext) raises -> hipDevice_t:
    var result = hipDevice_t()
    # const char *AsyncRT_DeviceContext_hip_device(hipDevice_t *result, const DeviceContext *ctx)
    _checked(
        external_call["AsyncRT_DeviceContext_hip_device", _CString[]](
            Pointer(to=result),
            ctx._handle,
        )
    )
    return result


# Accessor function to get access to the underlying hipStream_t from an abstract DeviceStream.
# Use `var hip_stream: hipStream_t = HIP(ctx.stream())` where ctx is a `DeviceContext` to get access to the underlying hipStream_t.
@always_inline
def HIP(stream: DeviceStream) raises -> hipStream_t:
    var result = hipStream_t()
    # const char *AsyncRT_DeviceStream_hip_stream(hipStream_t *result, const DeviceStream *stream)
    _checked(
        external_call[
            "AsyncRT_DeviceStream_hip_stream",
            _CString[],
        ](
            Pointer(to=result),
            stream._handle,
        )
    )
    return result


# Accessor function to get access to the underlying hipModule_t from a DeviceFunction.
@always_inline
def HIP_MODULE(func: DeviceFunction) raises -> hipModule_t:
    var result = hipModule_t()
    # const char *AsyncRT_DeviceFunction_hip_module(hipModule_t *result, const DeviceFunction *func)
    _checked(
        external_call[
            "AsyncRT_DeviceFunction_hip_module",
            _CString[],
        ](
            Pointer(to=result),
            func._handle,
        )
    )
    return result
