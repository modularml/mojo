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

from . import DeviceContext
from std.ffi import external_call

from max.gpu.host.device_context import (
    _CString,
    _checked,
    _DeviceContextPtr,
)


struct _MTLDevice:
    pass


comptime MTLDevice = Pointer[_MTLDevice, MutAnyOrigin]


# Accessor function to get access to the underlying MTLDevice from an abstract DeviceContext.
# Use `var metal_device: MTLDevice = metal_device(ctx)` where ctx is a `DeviceContext` to get access to the
# underlying MTLDevice.
@always_inline
def metal_device(ctx: DeviceContext) raises -> MTLDevice:
    var result = Optional[MTLDevice]()
    # const char *AsyncRT_DeviceContext_metal_device(MTL::Device **result, const DeviceContext *ctx)
    _checked(
        external_call[
            "AsyncRT_DeviceContext_metal_device",
            _CString[],
        ](
            Pointer(to=result),
            ctx._handle,
        )
    )
    return result.unsafe_value()
