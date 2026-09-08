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
"""This module provides GPU-specific global constants and configuration values.

The module defines hardware-specific constants like warp size and thread block limits
that are used throughout the GPU programming interface. It handles both NVIDIA and AMD
GPU architectures, automatically detecting and configuring the appropriate values based
on the available hardware.

The constants are resolved at compile time based on the target GPU architecture and
are used to optimize code generation and ensure hardware compatibility.
"""

from std.sys.info import (
    CompilationTarget,
    _accelerator_arch,
    _is_amd_rdna,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    is_amd_gpu,
    is_apple_gpu,
    is_nvidia_gpu,
)

from .host.info import GPUInfo

# ===-----------------------------------------------------------------------===#
# WARP_SIZE
# ===-----------------------------------------------------------------------===#


comptime WARP_SIZE: Int = _resolve_warp_size()
"""The number of threads that execute in lockstep within a warp on the GPU.

This constant represents the hardware warp size, which is the number of threads that execute
instructions synchronously as a unit. The value is architecture-dependent:
- 32 threads per warp on NVIDIA GPUs
- 32 threads per warp on AMD RDNA GPUs
- 64 threads per warp on AMD CDNA GPUs
- 0 if no GPU is detected

The warp size is a fundamental parameter that affects:
- Thread scheduling and execution
- Memory access coalescing
- Synchronization primitives
- Overall performance optimization
"""


def _resolve_warp_size() -> Int:
    comptime if is_nvidia_gpu():
        return 32
    elif _is_amd_rdna():
        return 32
    elif is_amd_gpu():
        return 64
    elif is_apple_gpu():
        return 32
    elif _accelerator_arch() == "":
        return 0
    else:
        return GPUInfo.from_name[_accelerator_arch()]().warp_size


# ===-----------------------------------------------------------------------===#
# WARPGROUP_SIZE
# ===-----------------------------------------------------------------------===#


comptime WARPGROUP_SIZE: Int = _resolve_warpgroup_size()
"""The number of threads in a warpgroup on Nvidia GPUs.

On Nvidia GPUs after hopper, a warpgroup consists of 4 subsequent arps
i.e. 128 threads. The first warp id must be multiple of 4.

Warpgroup is used for wgmma instructions on Hopper and tcgen05.ld on Blackwell.
"""


def _resolve_warpgroup_size() -> Int:
    # We can't constrain it here because the constant is used on host for
    # compilation test w/o nvidia GPUs.
    # comptime assert is_nvidia_gpu(), "Warpgroup only applies to Nvidia GPUs."

    return 128


# ===-----------------------------------------------------------------------===#
# MAX_THREADS_PER_BLOCK_METADATA
# ===-----------------------------------------------------------------------===#

comptime MAX_THREADS_PER_BLOCK_METADATA = _resolve_max_threads_per_block_metadata()
"""This is metadata tag that is used in conjunction with __llvm_metadata to
give a hint to the compiler about the max threads per block that's used."""


def _resolve_max_threads_per_block_metadata() -> __mlir_type.`!kgen.string`:
    comptime if is_nvidia_gpu() or has_nvidia_gpu_accelerator():
        return "nvvm.maxntid".value
    elif is_amd_gpu() or has_amd_gpu_accelerator():
        return "rocdl.flat_work_group_size".value
    elif is_apple_gpu():
        # That attribute is used to represent Metal's [[max_total_threads_per_threadgroup(x)]]
        return "pop.air.max_work_group_size".value
    else:
        CompilationTarget.unsupported_target_error[
            operation="MAX_THREADS_PER_BLOCK_METADATA",
        ]()
