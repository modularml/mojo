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
# DOC: max/gpu/fundamentals.mdx

from std.sys import (
    exit,
    has_accelerator,
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
)

from max.gpu.host import DeviceAttribute, DeviceContext


def main() raises:
    comptime if not has_accelerator():
        print("No GPU detected")
        exit(0)
    else:
        var num_devices = DeviceContext.number_of_devices()
        print("Number of GPUs:", num_devices)

        var ctx = DeviceContext()  # Get context for the default device, 0
        print("Device ID:", ctx.id())
        print("Device API:", ctx.api())
        print("Device name:", ctx.name())
        print("Device API version:", ctx.get_api_version())
        var mem_info = ctx.get_memory_info()
        print("Total memory:", mem_info[1])
        print("Free memory:", mem_info[0])

        print(
            "DeviceAttribute.MAX_THREADS_PER_BLOCK:",
            ctx.get_attribute(DeviceAttribute.MAX_THREADS_PER_BLOCK),
        )
        print(
            "DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK:",
            ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK),
        )

        comptime if not has_apple_gpu_accelerator():
            # Not currently defined for Apple GPUs

            print(
                "DeviceAttribute.MULTIPROCESSOR_COUNT:",
                ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
            )
            print(
                "DeviceAttribute.MAX_THREADS_PER_MULTIPROCESSOR:",
                ctx.get_attribute(
                    DeviceAttribute.MAX_THREADS_PER_MULTIPROCESSOR
                ),
            )
            print(
                "DeviceAttribute.WARP_SIZE:",
                ctx.get_attribute(DeviceAttribute.WARP_SIZE),
            )
            print(
                "DeviceAttribute.MAX_REGISTERS_PER_MULTIPROCESSOR:",
                ctx.get_attribute(
                    DeviceAttribute.MAX_REGISTERS_PER_MULTIPROCESSOR
                ),
            )
            print(
                "DeviceAttribute.MAX_REGISTERS_PER_BLOCK:",
                ctx.get_attribute(DeviceAttribute.MAX_REGISTERS_PER_BLOCK),
            )

        comptime if not (
            has_amd_gpu_accelerator() or has_apple_gpu_accelerator()
        ):
            # Not currently defined for AMD and Apple GPUs

            print(
                "DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR:",
                ctx.get_attribute(
                    DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR
                ),
            )
            print(
                "DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:",
                ctx.get_attribute(
                    DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
                ),
            )
