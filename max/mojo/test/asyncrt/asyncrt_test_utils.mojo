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

from std.sys.info import _accelerator_arch
from std.sys.defines import get_defined_string, is_defined

from max.gpu.host import DeviceContext
from max.gpu.host.info import GPUInfo


def api() -> String:
    comptime if is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]():
        comptime api = get_defined_string["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]()

        comptime if api == "gpu":
            return String(GPUInfo.from_name[_accelerator_arch()]().api)
        return String(api)
    return "default"


def create_test_device_context(*, device_id: Int = 0) raises -> DeviceContext:
    # Create an instance of the DeviceContext
    var test_ctx: DeviceContext

    comptime if is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]():
        print("Using DeviceContext: V2 - " + api())
        test_ctx = DeviceContext(device_id=device_id, api=api())
    elif is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V1"]():
        raise Error("DeviceContextV1 is unsupported")
    else:
        print("Using DeviceContext: default")
        test_ctx = DeviceContext(device_id=device_id)

    return test_ctx
