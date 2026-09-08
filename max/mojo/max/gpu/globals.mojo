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

import std._gpu.globals

comptime MAX_THREADS_PER_BLOCK_METADATA = std._gpu.globals.MAX_THREADS_PER_BLOCK_METADATA
"""This is metadata tag that is used in conjunction with __llvm_metadata to
give a hint to the compiler about the max threads per block that's used."""

comptime WARP_SIZE = std._gpu.globals.WARP_SIZE
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

comptime WARPGROUP_SIZE = std._gpu.globals.WARPGROUP_SIZE
"""The number of threads in a warpgroup on Nvidia GPUs.

On Nvidia GPUs after hopper, a warpgroup consists of 4 subsequent arps
i.e. 128 threads. The first warp id must be multiple of 4.

Warpgroup is used for wgmma instructions on Hopper and tcgen05.ld on Blackwell.
"""
