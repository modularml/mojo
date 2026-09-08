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

from std.sys.arg import argv

import max.gpu.host
import max.gpu.host._nvidia_cuda as cuda


def compute_capability_to_arch_name(major: Int, minor: Int) -> StaticString:
    if major == 1:
        return "tesla"
    if major == 2:
        return "fermi"
    if major == 3:
        return "kepler"
    if major == 5:
        return "maxwell"
    if major == 6:
        return "pascal"
    if major == 7 and minor == 0:
        return "volta"
    if major == 7 and minor == 5:
        return "turing"
    if major == 8 and (minor == 0 or minor == 6 or minor == 7):
        return "ampere"
    if major == 8 and minor == 9:
        return "ada lovelace"
    if major == 9:
        return "hopper"
    if major == 10:
        return "blackwell (datacenter)"
    if major == 12:
        return "blackwell (consumer)"
    return "Unknown"


def main() raises:
    var args = argv()
    var api: String = "cuda"
    var device_id: Int = 0

    if len(args) == 2:
        api = args[1]
        device_id = 0
    elif len(args) == 3:
        api = args[1]
        device_id = Int(args[2])

    var ctx = host.DeviceContext(device_id, api=api)

    var compute_capability = ctx.compute_capability()
    var arch_name = ctx.arch_name()
    var major, minor = divmod(compute_capability, 10)

    if ctx.api() == "cuda":
        print("GPUInfo(")
        print('name="' + ctx.name() + '", ')
        print('api="' + String(ctx.api()) + '", ')
        print(
            'arch_name="'
            + compute_capability_to_arch_name(major, minor)
            + '", '
        )
        print(
            "compute=" + String(Float32(major) + (Float32(minor) / 10)) + ", "
        )
        print('version="' + arch_name + '", ')
        print(
            "sm_count="
            + String(
                ctx.get_attribute(host.DeviceAttribute.MULTIPROCESSOR_COUNT)
            )
            + ", "
        )
        print(
            "warp_size="
            + String(ctx.get_attribute(host.DeviceAttribute.WARP_SIZE))
            + ", "
        )
        print("threads_per_sm=-1, ")
        print(
            "shared_memory_per_multiprocessor="
            + String(
                ctx.get_attribute(
                    host.DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
                )
            )
            + ", "
        )
        print(
            "max_registers_per_block="
            + String(
                ctx.get_attribute(host.DeviceAttribute.MAX_REGISTERS_PER_BLOCK)
            )
            + ", "
        )
        print("max_thread_block_size=1024, ")
        print(")")
    elif ctx.api() == "hip":
        print("GPUInfo(")
        print('name="' + ctx.name() + '", ')
        print('api="' + String(ctx.api()) + '", ')
        print('arch_name="' + arch_name + '", ')
        print(
            "compute=" + String(Float32(major) + (Float32(minor) / 10)) + ", "
        )
        print(
            "sm_count="
            + String(
                ctx.get_attribute(host.DeviceAttribute.MULTIPROCESSOR_COUNT)
            )
            + ", "
        )
        print(
            "warp_size="
            + String(ctx.get_attribute(host.DeviceAttribute.WARP_SIZE))
            + ", "
        )
        print("threads_per_sm=-1, ")
        print(
            "max_registers_per_block="
            + String(
                ctx.get_attribute(host.DeviceAttribute.MAX_REGISTERS_PER_BLOCK)
            )
            + ", "
        )
        print("max_thread_block_size=1024, ")
        print(")")
