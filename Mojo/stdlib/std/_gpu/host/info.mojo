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
"""Contains information about GPU architectures and their capabilities.

This module provides detailed specifications for various GPU models including
NVIDIA and AMD GPUs. It includes information about compute capabilities,
memory specifications, thread organization, and performance characteristics.
"""

# Contributor note: if you're adding support for a new GPU architecture, see
# `Mojo/docs/contributing/stdlib/adding-gpu-targets.md` for a step-by-step guide covering
# the MLIR target configuration, the `data_layout` string format, and the
# locations in this file that need to be updated.

from std.math import ceildiv, floor
from std.sys.info import (
    CompilationTarget,
    _accelerator_arch,
    _TargetType,
)


@always_inline
def get_gpu_target[
    # TODO: Ideally this is an Optional[StaticString] but blocked by MOCO-1039
    target_arch: StaticString = _accelerator_arch(),
]() -> _TargetType:
    """Gets the GPU target information for the specified architecture.

    Parameters:
        target_arch: GPU architecture name (defaults to current accelerator architecture).

    Returns:
        Target type information for the specified GPU architecture.
    """
    comptime assert (
        target_arch != ""
    ), "target_arch must be a valid GPU architecture."
    return GPUInfo.from_name[target_arch]().target()


comptime _KB = 1024
comptime _K = 1024

# NVIDIA Architecture Families
comptime NvidiaMaxwellFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=96 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Maxwell architecture family (sm_50-sm_53)."""

comptime NvidiaPascalFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=64 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Pascal architecture family (sm_60-sm_62)."""

comptime NvidiaTuringFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=64 * _KB,
    max_registers_per_block=32 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Turing architecture family (sm_75)."""

# Ampere architecture has three distinct variants based on compute capability:
# - sm_80: High-end datacenter (A100)
# - sm_86: Workstation/cloud (A10, RTX A-series)
# - sm_87: Embedded/edge (Jetson Orin)

comptime NvidiaAmpereDatacenterFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=164 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Ampere datacenter architecture family (sm_80)."""

comptime NvidiaAmpereWorkstationFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=48 * 32,
    shared_memory_per_multiprocessor=100 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Ampere workstation architecture family (sm_86)."""

comptime NvidiaAmpereEmbeddedFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=48 * 32,
    shared_memory_per_multiprocessor=164 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Ampere embedded architecture family (sm_87)."""

comptime NvidiaAdaFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=48 * 32,
    shared_memory_per_multiprocessor=100 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Ada Lovelace architecture family (sm_89)."""

comptime NvidiaHopperFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=228 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Hopper architecture family (sm_90)."""

comptime NvidiaBlackwellFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=228 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Blackwell datacenter architecture family (sm_100)."""

comptime NvidiaBlackwellConsumerFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=48 * 32,
    shared_memory_per_multiprocessor=100 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Blackwell consumer architecture family (sm_120)."""

# AMD Architecture Families
comptime AMDCDNA2Family = AcceleratorArchitectureFamily(
    warp_size=64,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=64 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""AMD CDNA2 architecture family (gfx90a)."""

comptime AMDCDNA3Family = AcceleratorArchitectureFamily(
    warp_size=64,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=64 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""AMD CDNA3 architecture family (gfx94x)."""

comptime AMDCDNA4Family = AcceleratorArchitectureFamily(
    warp_size=64,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=160 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""AMD CDNA4 architecture family (gfx95x)."""

comptime AMDRDNAFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=32 * 32,
    shared_memory_per_multiprocessor=32 * _KB,
    max_registers_per_block=32 * _K,
    max_thread_block_size=_K,
)
"""AMD RDNA architecture family."""

# Apple Architecture Families
comptime AppleMetalFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=32 * 32,
    shared_memory_per_multiprocessor=32 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""Apple Metal GPU architecture family."""

# ===-----------------------------------------------------------------------===#
# AcceleratorArchitectureFamily
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct AcceleratorArchitectureFamily(TrivialRegisterPassable):
    """Defines common defaults for a GPU architecture family.

    This struct captures the shared characteristics across GPUs in the same
    architecture family, reducing redundancy when defining new GPU models.
    """

    var warp_size: Int
    """Number of threads in a warp/wavefront."""

    var threads_per_multiprocessor: Int
    """Maximum number of threads per streaming multiprocessor."""

    var shared_memory_per_multiprocessor: Int
    """Size of shared memory available per multiprocessor in bytes."""

    var max_registers_per_block: Int
    """Maximum number of registers that can be allocated to a thread block."""

    var max_thread_block_size: Int
    """Maximum number of threads allowed in a thread block."""


# ===-----------------------------------------------------------------------===#
# NoGPU
# ===-----------------------------------------------------------------------===#


def _get_empty_target() -> _TargetType:
    """Creates an empty target configuration for when no GPU is available.

    Returns:
        An empty MLIR target configuration.
    """
    return __mlir_attr[
        `#kgen.target<triple = "", `,
        `arch = "", `,
        `features = "", `,
        `data_layout="",`,
        `index_bit_width = 0,`,
        `simd_bit_width = 0`,
        `> : !kgen.target`,
    ]


comptime NoGPU = GPUInfo(
    name="NoGPU",
    api="none",
    arch_name="no_gpu",
    compute=0,
    version="",
    sm_count=0,
    warp_size=0,
    threads_per_multiprocessor=0,
    shared_memory_per_multiprocessor=0,
    max_registers_per_block=0,
    max_thread_block_size=0,
)
"""Placeholder for when no GPU is available."""


# ===-----------------------------------------------------------------------===#
# Apple Silicon
# ===-----------------------------------------------------------------------===#
def _get_metal_m1_target() -> _TargetType:
    """Creates an MLIR target configuration for M1 Metal GPU.

    Returns:
        MLIR target configuration for M1 Metal.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `arch = "apple-m1", `,
        `features = "+metal3_2,+air2_7_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_metal_m2_target() -> _TargetType:
    """Creates an MLIR target configuration for M2 Metal GPU.

    Returns:
        MLIR target configuration for M2 Metal.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `arch = "apple-m2", `,
        `features = "+metal3_2,+air2_7_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_metal_m3_target() -> _TargetType:
    """Creates an MLIR target configuration for M3 Metal GPU.

    Returns:
        MLIR target configuration for M3 Metal.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `arch = "apple-m3", `,
        `features = "+metal3_2,+air2_7_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_metal_m4_target() -> _TargetType:
    """Creates an MLIR target configuration for M4 Metal GPU.

    Returns:
        MLIR target configuration for M4 Metal.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `arch = "apple-m4", `,
        `features = "+metal3_2,+air2_7_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_metal_m5_target() -> _TargetType:
    """Creates an MLIR target configuration for M5 Metal GPU.

    Returns:
        MLIR target configuration for M5 Metal.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `arch = "apple-m5", `,
        `features = "+metal3_2,+air2_7_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_metal_m1_metal4_target() -> _TargetType:
    """Creates an MLIR target configuration for M1 Metal GPU with Metal 4.0.

    Returns:
        MLIR target configuration for M1 Metal 4.0.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `arch = "apple-m1", `,
        `features = "+metal4_0,+air2_8_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_metal_m2_metal4_target() -> _TargetType:
    """Creates an MLIR target configuration for M2 Metal GPU with Metal 4.0.

    Returns:
        MLIR target configuration for M2 Metal 4.0.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `arch = "apple-m2", `,
        `features = "+metal4_0,+air2_8_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_metal_m3_metal4_target() -> _TargetType:
    """Creates an MLIR target configuration for M3 Metal GPU with Metal 4.0.

    Returns:
        MLIR target configuration for M3 Metal 4.0.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `arch = "apple-m3", `,
        `features = "+metal4_0,+air2_8_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_metal_m4_metal4_target() -> _TargetType:
    """Creates an MLIR target configuration for M4 Metal GPU with Metal 4.0.

    Returns:
        MLIR target configuration for M4 Metal 4.0.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `arch = "apple-m4", `,
        `features = "+metal4_0,+air2_8_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_metal_m5_metal4_target() -> _TargetType:
    """Creates an MLIR target configuration for M5 Metal GPU with Metal 4.0.

    Returns:
        MLIR target configuration for M5 Metal 4.0.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `arch = "apple-m5", `,
        `features = "+metal4_0,+air2_8_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime MetalM1 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M1",
    api="metal",
    arch_name="apple-m1",
    compute=3.0,  # Metal version 3.0
    version="metal_3",
    sm_count=8,  # M1 has 8 GPU cores
)
"""Apple M1 GPU configuration."""

comptime MetalM2 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M2",
    api="metal",
    arch_name="apple-m2",
    compute=3.0,  # Metal version 3.0
    version="metal_3",
    sm_count=10,  # M2 has 10 GPU cores
)
"""Apple M2 GPU configuration."""

comptime MetalM3 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M3",
    api="metal",
    arch_name="apple-m3",
    compute=3.0,  # Metal version 3.0 for M3
    version="metal_3",
    sm_count=10,  # M3 has 10 GPU cores
)
"""Apple M3 GPU configuration."""

comptime MetalM4 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M4",
    api="metal",
    arch_name="apple-m4",
    compute=3.0,  # Metal version 3.0 for M4
    version="metal_3",
    sm_count=10,  # M4 has 10 GPU cores
)
"""Apple M4 GPU configuration."""

comptime MetalM5 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M5",
    api="metal",
    arch_name="apple-m5",
    compute=3.0,  # Metal version 3.0 for M5
    version="metal_3",
    sm_count=10,  # M5 has 10 GPU cores
)
"""Apple M5 GPU configuration."""

comptime MetalM1Metal4 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M1 Metal4",
    api="metal",
    arch_name="apple-m1-metal4",
    compute=4.0,  # Metal 4.0, requires macOS 26
    version="metal_4",
    sm_count=8,  # M1 has 8 GPU cores
)
"""Apple M1 GPU configuration for Metal 4."""

comptime MetalM2Metal4 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M2 Metal4",
    api="metal",
    arch_name="apple-m2-metal4",
    compute=4.0,  # Metal 4.0, requires macOS 26
    version="metal_4",
    sm_count=10,  # M2 has 10 GPU cores
)
"""Apple M2 GPU configuration for Metal 4."""

comptime MetalM3Metal4 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M3 Metal4",
    api="metal",
    arch_name="apple-m3-metal4",
    compute=4.0,  # Metal 4.0, requires macOS 26
    version="metal_4",
    sm_count=10,  # M3 has 10 GPU cores
)
"""Apple M3 GPU configuration for Metal 4."""

comptime MetalM4Metal4 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M4 Metal4",
    api="metal",
    arch_name="apple-m4-metal4",
    compute=4.0,  # Metal 4.0, requires macOS 26
    version="metal_4",
    sm_count=10,  # M4 has 10 GPU cores
)
"""Apple M4 GPU configuration for Metal 4."""

comptime MetalM5Metal4 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M5 Metal4",
    api="metal",
    arch_name="apple-m5-metal4",
    compute=4.0,  # Metal 4.0, requires macOS 26
    version="metal_4",
    sm_count=10,  # M5 has 10 GPU cores
)
"""Apple M5 GPU configuration for Metal 4."""

# ===-----------------------------------------------------------------------===#
# A100
# ===-----------------------------------------------------------------------===#

# Note: features = "+ptx81" means that the kernel should be compiled using
# PTX version 8.1. This must be less than or equal to the installed CUDA
# driver's maximum supported PTX version. Currently we hardcode this to
# PTX version 8.1 which means that you need to have a CUDA driver included with
# CUDA 12.5 toolkit. The mapping from CUDA Driver to PTX version can be found by
# looking at the PTX ISA in the versioned docs
# https://developer.nvidia.com/cuda-toolkit-archive.


def _get_a100_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA A100 GPU.

    Returns:
        MLIR target configuration for A100.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_80", `,
        `features = "+ptx81,+sm_80", `,
        `tune_cpu = "sm_80", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime A100 = GPUInfo.from_family(
    family=NvidiaAmpereDatacenterFamily,
    name="A100",
    api="cuda",
    arch_name="ampere",
    compute=8.0,
    version="sm_80",
    sm_count=108,
)
"""NVIDIA A100 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# A10
# ===-----------------------------------------------------------------------===#


def _get_a10_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA A10 GPU.

    Returns:
        MLIR target configuration for A10.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx81,+sm_86", `,
        `tune_cpu = "sm_86", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime A10 = GPUInfo.from_family(
    family=NvidiaAmpereWorkstationFamily,
    name="A10",
    api="cuda",
    arch_name="ampere",
    compute=8.6,
    version="sm_86",
    sm_count=72,
)
"""NVIDIA A10 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# Jetson Orin Nano
# ===-----------------------------------------------------------------------===#


def _get_orin_nano_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA Jetson Orin Nano GPU.

    Returns:
        MLIR target configuration for Orin Nano.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_87", `,
        `features = "+ptx81,+sm_87", `,
        `tune_cpu = "sm_87", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime OrinNano = GPUInfo.from_family(
    family=NvidiaAmpereEmbeddedFamily,
    name="Orin Nano",
    api="cuda",
    arch_name="ampere",
    compute=8.7,
    version="sm_87",
    sm_count=8,
)
"""NVIDIA Orin Nano GPU configuration."""

# ===-----------------------------------------------------------------------===#
# Jetson Thor
# ===-----------------------------------------------------------------------===#


def _get_jetson_thor_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA Jetson Thor.

    Returns:
        MLIR target configuration for Jetson Thor.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_110", `,
        `features = "+ptx90,+sm_110", `,
        `tune_cpu = "sm_110", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


comptime JetsonThor = GPUInfo.from_family(
    family=NvidiaBlackwellFamily,
    name="Jetson Thor",
    api="cuda",
    arch_name="blackwell",
    compute=11.0,
    version="sm_110",
    sm_count=20,
)
"""NVIDIA Jetson Thor GPU configuration."""

# ===-----------------------------------------------------------------------===#
# DGX Spark
# ===-----------------------------------------------------------------------===#


def _get_dgx_spark_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA DGX Spark.

    Returns:
        MLIR target configuration for DGX Spark.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_121a", `,
        `features = "+ptx88,+sm_121a", `,
        `tune_cpu = "sm_121a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime DGXSpark = GPUInfo.from_family(
    family=NvidiaBlackwellConsumerFamily,
    name="DGX Spark",
    api="cuda",
    arch_name="blackwell",
    compute=12.1,
    version="sm_121",
    sm_count=48,
)
"""NVIDIA DGX Spark GPU configuration."""

# ===-----------------------------------------------------------------------===#
# L4
# ===-----------------------------------------------------------------------===#


def _get_l4_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA L4 GPU.

    Returns:
        MLIR target configuration for L4.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81,+sm_89", `,
        `tune_cpu = "sm_89", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime L4 = GPUInfo.from_family(
    family=NvidiaAdaFamily,
    name="L4",
    api="cuda",
    arch_name="ada",
    compute=8.9,
    version="sm_89",
    sm_count=58,
)
"""NVIDIA L4 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# RTX 4090 M
# ===-----------------------------------------------------------------------===#


def _get_rtx4090m_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA RTX 4090 Mobile GPU.

    Returns:
        MLIR target configuration for RTX 4090M.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81,+sm_89", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime RTX4090m = GPUInfo.from_family(
    family=NvidiaAdaFamily,
    name="RTX4090m",
    api="cuda",
    arch_name="ada lovelace",
    compute=8.9,
    version="sm_89",
    sm_count=76,
)
"""NVIDIA RTX 4090 Mobile GPU configuration."""

# ===-----------------------------------------------------------------------===#
# RTX 4090
# ===-----------------------------------------------------------------------===#


def _get_rtx4090_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA RTX 4090.

    Returns:
        MLIR target configuration for RTX 4090.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81,+sm_89", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime RTX4090 = GPUInfo.from_family(
    family=NvidiaAdaFamily,
    name="RTX4090",
    api="cuda",
    arch_name="ada lovelace",
    compute=8.9,
    version="sm_89",
    sm_count=128,
)
"""NVIDIA RTX 4090 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# H100
# ===-----------------------------------------------------------------------===#


def _get_h100_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA H100 GPU.

    Returns:
        MLIR target configuration for H100.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_90a", `,
        `features = "+ptx85,+sm_90a", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


# https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
comptime H100 = GPUInfo.from_family(
    family=NvidiaHopperFamily,
    name="H100",
    api="cuda",
    arch_name="hopper",
    compute=9.0,
    version="sm_90a",
    sm_count=132,
)
"""NVIDIA H100 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# B100
# ===-----------------------------------------------------------------------===#


def _get_b100_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA B100 GPU.

    Returns:
        MLIR target configuration for B100.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_100a", `,
        `features = "+ptx88,+sm_100a", `,
        `tune_cpu = "sm_100a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


# https://resources.nvidia.com/en-us-blackwell-architecture
# TODO: Update once we have B100 access.
comptime B100 = GPUInfo.from_family(
    family=NvidiaBlackwellFamily,
    name="B100",
    api="cuda",
    arch_name="blackwell",
    compute=10.0,
    version="sm_100a",
    sm_count=132,
)
"""NVIDIA B100 GPU configuration."""

comptime B200 = GPUInfo.from_family(
    family=NvidiaBlackwellFamily,
    name="B200",
    api="cuda",
    arch_name="blackwell",
    compute=10.0,
    version="sm_100a",
    sm_count=148,
)
"""NVIDIA B200 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# B300
# ===-----------------------------------------------------------------------===#


def _get_b300_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA B300 GPU.

    Returns:
        MLIR target configuration for B300.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_103a", `,
        `features = "+ptx88,+sm_103a", `,
        `tune_cpu = "sm_103a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime B300 = GPUInfo.from_family(
    family=NvidiaBlackwellFamily,
    name="B300",
    api="cuda",
    arch_name="blackwell",
    compute=10.3,
    version="sm_103a",
    sm_count=160,
)
"""NVIDIA B300 GPU configuration."""


def _is_sm10x_gpu(info: GPUInfo) -> Bool:
    """Returns True for any Blackwell datacenter GPU (B100, B200, B300).

    Use this to check if the GPU supports SM100-class features. For
    architecture-specific tuning, compare against individual GPUInfo
    constants (e.g., `ctx.default_device_info == B300`).

    Args:
        info: GPU info to check.

    Returns:
        True if the GPU is a Blackwell datacenter GPU.
    """
    return (
        info == materialize[B100]()
        or info == materialize[B200]()
        or info == materialize[B300]()
    )


def _is_sm12x_gpu(info: GPUInfo) -> Bool:
    """Returns True for any Blackwell consumer GPU (sm_120 / sm_121).

    Covers the RTX 50-series / RTX PRO (sm_120) and GB10 / DGX Spark (sm_121),
    which have no SM100 warp-specialized path and route block-scaled / NVFP4
    work to the cuBLASLt vendor kernels. Mirrors `_is_sm10x_gpu`.

    Args:
        info: GPU info to check.

    Returns:
        True if the GPU is a Blackwell consumer (sm_12x) GPU.
    """
    return info.compute >= 12.0 and info.compute < 13.0


# ===-----------------------------------------------------------------------===#
# RTX5090
# ===-----------------------------------------------------------------------===#


def _get_rtx5090_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA RTX5090 GPU.

    Returns:
        MLIR target configuration for RTX5090.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_120a", `,
        `features = "+ptx87,+sm_120a", `,
        `tune_cpu = "sm_120a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


# https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
comptime RTX5090 = GPUInfo.from_family(
    family=NvidiaBlackwellConsumerFamily,
    name="RTX5090",
    api="cuda",
    arch_name="blackwell",
    compute=12.0,
    version="sm_120a",
    sm_count=170,
)
"""NVIDIA RTX 5090 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# RTX3090
# ===-----------------------------------------------------------------------===#


def _get_rtx3090_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA GeForce RTX 3090.

    Returns:
        MLIR target configuration for NVIDIA GeForce RTX 3090.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx63,+sm_86", `,
        `tune_cpu = "sm_86", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


# https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/
comptime RTX3090 = GPUInfo.from_family(
    family=NvidiaAmpereWorkstationFamily,
    name="NVIDIA GeForce RTX 3090",
    api="cuda",
    arch_name="ampere",
    compute=8.6,
    version="sm_86",
    sm_count=82,
)
"""NVIDIA GeForce RTX 3090 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# GTX1080Ti
# ===-----------------------------------------------------------------------===#


def _get_gtx1080ti_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA GTX 1080 Ti GPU.

    Returns:
        MLIR target configuration for GTX 1080 Ti.
    """
    # Note: GTX 1080 Ti doesn't specify tune_cpu, data_layout, or index_bit_width
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_61", `,
        `features = "+ptx50,+sm_61", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime GTX1080Ti = GPUInfo.from_family(
    family=NvidiaPascalFamily,
    name="NVIDIA GeForce GTX 1080 Ti",
    api="cuda",
    arch_name="pascal",
    compute=6.1,
    version="sm_61",
    sm_count=28,
)
"""NVIDIA GeForce GTX 1080 Ti GPU configuration."""


# ===-----------------------------------------------------------------------===#
# GTX1060
# ===-----------------------------------------------------------------------===#


def _get_gtx1060_target() -> _TargetType:
    """
    Creates an MLIR target configuration for NVIDIA GTX 1060 GPU.

    Returns:
        MLIR target configuration for GTX 1060.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_61", `,
        `features = "+ptx50,+sm_61", `,
        `tune_cpu = "sm_61", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime GTX1060 = GPUInfo.from_family(
    family=NvidiaPascalFamily,
    name="NVIDIA GeForce GTX 1060",
    api="cuda",
    arch_name="pascal",
    compute=6.1,
    version="sm_61",
    sm_count=10,
)
"""NVIDIA GeForce GTX 1060 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# GTX970
# ===-----------------------------------------------------------------------===#


def _get_gtx970_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA GTX 970 GPU.

    Returns:
        MLIR target configuration for GTX 970.
    """
    # Note: GTX 970 doesn't specify tune_cpu, data_layout, or index_bit_width
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_52", `,
        `features = "+ptx50,+sm_52", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime GTX970 = GPUInfo.from_family(
    family=NvidiaMaxwellFamily,
    name="NVIDIA GeForce GTX 970",
    api="cuda",
    arch_name="maxwell",
    compute=5.2,
    version="sm_52",
    sm_count=13,
)
"""NVIDIA GeForce GTX 970 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# Tesla P100
# ===-----------------------------------------------------------------------===#


def _get_teslap100_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA Tesla P100 GPU.

    Returns:
        MLIR target configuration for Tesla P100.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_60", `,
        `features = "+ptx50,+sm_60", `,
        `tune_cpu = "sm_60", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime TeslaP100 = GPUInfo.from_family(
    family=NvidiaPascalFamily,
    name="NVIDIA Tesla P100",
    api="cuda",
    arch_name="pascal",
    compute=6.0,
    version="sm_60",
    sm_count=56,
)
"""NVIDIA Tesla P100 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# RTX2060
# ===-----------------------------------------------------------------------===#


def _get_rtx2060_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA RTX 2060 GPU.

    Returns:
        MLIR target configuration for RTX 2060.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_75", `,
        `features = "+ptx63,+sm_75", `,
        `tune_cpu = "sm_75", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime RTX2060 = GPUInfo.from_family(
    family=NvidiaTuringFamily,
    name="RTX2060",
    api="cuda",
    arch_name="turing",
    compute=7.5,
    version="sm_75",
    sm_count=30,
)
"""NVIDIA RTX 2060 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# MI250X
# ===-----------------------------------------------------------------------===#


def _get_mi250x_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD MI250X GPU.

    Returns:
        MLIR target configuration for MI250X.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx90a", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime MI250X = GPUInfo.from_family(
    family=AMDCDNA2Family,
    name="MI250X",
    api="hip",
    arch_name="gfx90a",
    compute=9.0,
    version="CDNA2",
    sm_count=220,
)
"""AMD MI250X GPU configuration."""


# ===-----------------------------------------------------------------------===#
# MI300X
# ===-----------------------------------------------------------------------===#


def _get_mi300x_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD MI300X GPU.

    Returns:
        MLIR target configuration for MI300X.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx942", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime MI300X = GPUInfo.from_family(
    family=AMDCDNA3Family,
    name="MI300X",
    api="hip",
    arch_name="gfx942",
    compute=9.4,
    version="CDNA3",
    sm_count=304,
)
"""AMD MI300X GPU configuration."""


# ===-----------------------------------------------------------------------===#
# MI300A
# ===-----------------------------------------------------------------------===#


def _get_mi300a_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD MI300A APU.

    Returns:
        MLIR target configuration for MI300A.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx942", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime MI300A = GPUInfo.from_family(
    family=AMDCDNA3Family,
    name="MI300A",
    api="hip",
    arch_name="gfx942",
    compute=9.4,
    version="CDNA3",
    sm_count=228,
)
"""AMD MI300A APU configuration.

The MI300A is an Accelerated Processing Unit (APU) that integrates Zen 4 CPU
cores with CDNA 3 GPU compute units and unified HBM3 memory. It shares the
`gfx942` ISA with the MI300X but has fewer compute units (228 vs 304) and
unified host/device memory. Found in systems such as the CINES Adastra
supercomputer.
"""


# ===-----------------------------------------------------------------------===#
# MI355X
# ===-----------------------------------------------------------------------===#


def _get_mi355x_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD MI355X GPU.

    Returns:
        MLIR target configuration for MI355X.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx950", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime MI355X = GPUInfo.from_family(
    family=AMDCDNA4Family,
    name="MI355X",
    api="hip",
    arch_name="gfx950",
    compute=9.5,
    version="CDNA4",
    sm_count=256,
)
"""AMD MI355X GPU configuration."""


# ===-----------------------------------------------------------------------===#
# Radeon 7xxx, 9xxx, 780m
# ===-----------------------------------------------------------------------===#


def _get_9070_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 9070 GPU.

    Returns:
        MLIR target configuration for 9070.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1201", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_9060_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 9060 GPU.

    Returns:
        MLIR target configuration for 9060.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1200", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_7900_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 7900 GPU.

    Returns:
        MLIR target configuration for 7900.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1100", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_7800_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 7800/7700 GPU.

    Returns:
        MLIR target configuration for 7800/7700.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1101", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_7600_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 7600 GPU.

    Returns:
        MLIR target configuration for 7600.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1102", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_6900_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 6900 GPU.

    Returns:
        MLIR target configuration for 6900.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1030", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_780m_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 780m GPU.

    Returns:
        MLIR target configuration for 780m.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1103", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_880m_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 880M GPU.

    Returns:
        MLIR target configuration for 880M.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1150", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_8060s_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 8060S GPU.

    Returns:
        MLIR target configuration for 8060S.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1151", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_860m_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 860M GPU.

    Returns:
        MLIR target configuration for 860M.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1152", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


def _get_steamdeck_target() -> _TargetType:
    """Creates an MLIR target configuration for the Steam Deck's Van Gogh APU.

    Returns:
        MLIR target configuration for the Steam Deck's Van Gogh APU.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1033", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime Radeon9070 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 9070",
    api="hip",
    arch_name="gfx1201",
    compute=12.0,
    version="RDNA4",
    sm_count=64,
)
"""AMD Radeon 9070 GPU configuration."""

comptime Radeon9060 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 9060",
    api="hip",
    arch_name="gfx1200",
    compute=12.0,
    version="RDNA4",
    sm_count=32,
)
"""AMD Radeon 9060 GPU configuration."""

comptime Radeon7900 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 7900",
    api="hip",
    arch_name="gfx1100",
    compute=11.0,
    version="RDNA3",
    sm_count=96,
)
"""AMD Radeon 7900 GPU configuration."""

comptime Radeon7800 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 7800/7700",
    api="hip",
    arch_name="gfx1101",
    compute=11.0,
    version="RDNA3",
    sm_count=60,
)
"""AMD Radeon 7800/7700 GPU configuration."""

comptime Radeon7600 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 7600",
    api="hip",
    arch_name="gfx1102",
    compute=11.0,
    version="RDNA3",
    sm_count=32,
)
"""AMD Radeon 7600 GPU configuration."""

comptime Radeon6900 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 6900",
    api="hip",
    arch_name="gfx1030",
    compute=10.3,
    version="RDNA2",
    sm_count=60,
)
"""AMD Radeon 6900 GPU configuration."""


comptime Radeon780m = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 780M",
    api="hip",
    arch_name="gfx1103",
    compute=11.0,
    version="RDNA3",
    sm_count=12,
)
"""AMD Radeon 780M GPU configuration."""

comptime Radeon880m = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 880M",
    api="hip",
    arch_name="gfx1150",
    compute=11.5,
    version="RDNA3.5",
    sm_count=12,
)
"""AMD Radeon 880M GPU configuration."""

comptime Radeon8060s = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 8060S",
    api="hip",
    arch_name="gfx1151",
    compute=11.5,
    version="RDNA3.5",
    sm_count=40,
)
"""AMD Radeon 8060S GPU configuration."""

comptime Radeon860m = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 860M",
    api="hip",
    arch_name="gfx1152",
    compute=11.5,
    version="RDNA3.5",
    sm_count=8,
)
"""AMD Radeon 860M GPU configuration."""

comptime SteamDeck = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Steam Deck",
    api="hip",
    arch_name="gfx1033",
    compute=10.3,
    version="RDNA2",
    sm_count=8,
)
"""Steam Deck (Van Gogh) APU configuration."""

# ===-----------------------------------------------------------------------===#
# GPUInfo
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct GPUInfo(Copyable, Equatable, Movable, RegisterPassable, Writable):
    """Comprehensive information about a GPU architecture.

    This struct contains detailed specifications about GPU capabilities,
    including compute units, memory, thread organization, and performance
    characteristics.
    """

    var name: StaticString
    """The model name of the GPU."""

    var api: StaticString
    """The graphics/compute API the GPU is programmed through.

    Each vendor has its own API, so this also identifies the vendor: `"cuda"`
    for NVIDIA, `"hip"` for AMD, `"metal"` for Apple, and `"none"` for `NoGPU`.
    A stdlib plugin contributes its own name for hardware the stdlib has no
    built-in knowledge of. Comparisons must use these exact spellings.
    """

    var arch_name: StaticString
    """The architecture name of the GPU (e.g., sm_80, gfx942)."""

    var compute: Float32
    """Compute capability version number for NVIDIA GPUs."""

    var version: StaticString
    """Version string of the GPU architecture."""

    var sm_count: Int
    """Number of streaming multiprocessors (SMs) on the GPU."""

    var warp_size: Int
    """Number of threads in a warp/wavefront."""

    var threads_per_multiprocessor: Int
    """Maximum number of threads per streaming multiprocessor."""

    var shared_memory_per_multiprocessor: Int
    """Size of shared memory available per multiprocessor in bytes."""

    var max_registers_per_block: Int
    """Maximum number of registers that can be allocated to a thread block."""

    var max_thread_block_size: Int
    """Maximum number of threads allowed in a thread block."""

    def target(self) -> _TargetType:
        """Gets the MLIR target configuration for this GPU.

        Returns:
            MLIR target configuration for the GPU.
        """
        if self.name == "NVIDIA Tesla P100":
            return _get_teslap100_target()
        if self.name == "NVIDIA GeForce GTX 1060":
            return _get_gtx1060_target()
        if self.name == "NVIDIA GeForce GTX 1080 Ti":
            return _get_gtx1080ti_target()
        if self.name == "NVIDIA GeForce GTX 970":
            return _get_gtx970_target()
        if self.name == "RTX2060":
            return _get_rtx2060_target()
        if self.name == "NVIDIA GeForce RTX 3090":
            return _get_rtx3090_target()
        if self.name == "A100":
            return _get_a100_target()
        if self.name == "A10":
            return _get_a10_target()
        if self.name == "L4":
            return _get_l4_target()
        if self.name == "RTX4090m":
            return _get_rtx4090m_target()
        if self.name == "RTX4090":
            return _get_rtx4090_target()
        if self.name == "H100":
            return _get_h100_target()
        if self.name == "B300":
            return _get_b300_target()
        if self.name == "B100" or self.name == "B200":
            return _get_b100_target()
        if self.name == "DGX Spark":
            return _get_dgx_spark_target()
        if self.name == "RTX5090":
            return _get_rtx5090_target()
        if self.name == "Jetson Thor":
            return _get_jetson_thor_target()
        if self.name == "MI250X":
            return _get_mi250x_target()
        if self.name == "MI300X":
            return _get_mi300x_target()
        if self.name == "MI300A":
            return _get_mi300a_target()
        if self.name == "MI355X":
            return _get_mi355x_target()
        if self.name == "Radeon 780M":
            return _get_780m_target()
        if self.name == "Radeon 880M":
            return _get_880m_target()
        if self.name == "Radeon 8060S":
            return _get_8060s_target()
        if self.name == "Radeon 860M":
            return _get_860m_target()
        if self.name == "Radeon 6900":
            return _get_6900_target()
        if self.name == "Radeon 7900":
            return _get_7900_target()
        if self.name == "Radeon 7800/7700":
            return _get_7800_target()
        if self.name == "Radeon 7600":
            return _get_7600_target()
        if self.name == "Radeon 9070":
            return _get_9070_target()
        if self.name == "Radeon 9060":
            return _get_9060_target()
        if self.name == "Steam Deck":
            return _get_steamdeck_target()
        if self.name == "M1":
            return _get_metal_m1_target()
        if self.name == "M1 Metal4":
            return _get_metal_m1_metal4_target()
        if self.name == "M2":
            return _get_metal_m2_target()
        if self.name == "M2 Metal4":
            return _get_metal_m2_metal4_target()
        if self.name == "M3":
            return _get_metal_m3_target()
        if self.name == "M3 Metal4":
            return _get_metal_m3_metal4_target()
        if self.name == "M4":
            return _get_metal_m4_target()
        if self.name == "M4 Metal4":
            return _get_metal_m4_metal4_target()
        if self.name == "M5":
            return _get_metal_m5_target()
        if self.name == "M5 Metal4":
            return _get_metal_m5_metal4_target()

        if self.name == "":
            return _get_empty_target()
        return _get_a100_target()

    @staticmethod
    def from_target[target: _TargetType]() -> Self:
        """Creates a `GPUInfo` instance from an MLIR target.

        Parameters:
            target: MLIR target configuration.

        Returns:
            GPU info corresponding to the target.
        """
        return _get_info_from_target[CompilationTarget[target]._arch()]()

    @staticmethod
    def from_name[name: StaticString]() -> Self:
        """Creates a `GPUInfo` instance from a GPU architecture name.

        Parameters:
            name: GPU architecture name (e.g., "sm_80", "gfx942").

        Returns:
            GPU info corresponding to the architecture name.
        """
        return _get_info_from_target[name]()

    @staticmethod
    def from_family(
        family: AcceleratorArchitectureFamily,
        name: StaticString,
        api: StaticString,
        arch_name: StaticString,
        compute: Float32,
        version: StaticString,
        sm_count: Int,
    ) -> Self:
        """Creates a `GPUInfo` instance using architecture family defaults.

        This constructor simplifies GPU definition by inheriting common
        characteristics from an architecture family while allowing specific
        values to be overridden.

        Args:
            family: Architecture family providing default values.
            name: The model name of the GPU.
            api: The graphics/compute API supported by the GPU.
            arch_name: The architecture name of the GPU.
            compute: Compute capability version number.
            version: Version string of the GPU architecture.
            sm_count: Number of streaming multiprocessors.

        Returns:
            A fully configured GPUInfo instance.
        """
        return Self(
            name=name,
            api=api,
            arch_name=arch_name,
            compute=compute,
            version=version,
            sm_count=sm_count,
            warp_size=family.warp_size,
            threads_per_multiprocessor=family.threads_per_multiprocessor,
            shared_memory_per_multiprocessor=family.shared_memory_per_multiprocessor,
            max_registers_per_block=family.max_registers_per_block,
            max_thread_block_size=family.max_thread_block_size,
        )

    def __eq__(self, other: Self) -> Bool:
        """Checks if two `GPUInfo` instances represent the same GPU model.

        Args:
            other: Another `GPUInfo` instance to compare against.

        Returns:
            True if both instances represent the same GPU model.
        """
        return self.name == other.name

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes GPU information to a writer.

        Outputs all GPU specifications and capabilities to the provided writer
        in a human-readable format.

        Args:
            writer: A Writer instance to output the GPU information.
        """
        writer.write("name: ", self.name, "\n")
        writer.write("api: ", self.api, "\n")
        writer.write("arch_name: ", self.arch_name, "\n")
        writer.write("compute: ", self.compute, "\n")
        writer.write("version: ", self.version, "\n")
        writer.write("sm_count: ", self.sm_count, "\n")
        writer.write("warp_size: ", self.warp_size, "\n")
        writer.write(
            "threads_per_multiprocessor: ",
            self.threads_per_multiprocessor,
            "\n",
        )
        writer.write(
            "shared_memory_per_multiprocessor: ",
            self.shared_memory_per_multiprocessor,
            "\n",
        )
        writer.write(
            "max_registers_per_block: ", self.max_registers_per_block, "\n"
        )
        writer.write(
            "max_thread_block_size: ", self.max_thread_block_size, "\n"
        )


# ===-----------------------------------------------------------------------===#
# _build_unsupported_arch_error
# ===-----------------------------------------------------------------------===#


def _build_unsupported_arch_error[target_arch: StaticString]() -> String:
    """Builds a helpful error message for unsupported GPU architectures.

    Provides a comprehensive list of all supported GPU architectures across
    all vendors with documentation links.

    Parameters:
        target_arch: The unsupported target architecture string.

    Returns:
        A detailed error message with supported architectures and doc links.
    """
    comptime nvidia_archs = (
        "sm_52 (Maxwell), sm_60/sm_61 (Pascal), sm_75 (Turing), sm_80 (Ampere"
        " A100), sm_86 (Ampere A10), sm_87 (Orin), sm_89 (Ada L4/RTX4090),"
        " sm_90/sm_90a (Hopper H100), sm_100/sm_100a (Blackwell B100/B200),"
        " sm_110 (Jetson Thor), sm_120/sm_120a (Blackwell RTX5090), sm_121 (DGX"
        " Spark)"
    )
    comptime amd_archs = (
        "gfx90a (MI250X), gfx942 (MI300X/MI300A), gfx950 (MI355X), gfx1030"
        " (Radeon 6900), gfx1033 (Van Gogh), gfx1100 (Radeon 7900), gfx1101"
        " (Radeon 7800), gfx1102 (Radeon 7600), gfx1103 (Radeon 780M),"
        " gfx1150/gfx1151/gfx1152 (Radeon 8xx), gfx1200 (Radeon 9060), gfx1201"
        " (Radeon 9070)"
    )
    comptime apple_archs = (
        "metal:1 (M1), metal:2 (M2), metal:3 (M3), metal:4 (M4)"
    )

    var prefix: String

    comptime if target_arch == "":
        prefix = "Unknown GPU architecture detected."
    else:
        prefix = String(
            "GPU architecture '", target_arch, "' is not supported."
        )

    return String(
        prefix,
        "\n\nSupported GPU architectures:\n\n",
        "  NVIDIA: ",
        nvidia_archs,
        "\n  See: https://developer.nvidia.com/cuda-gpus\n\n",
        "  AMD: ",
        amd_archs,
        (
            "\n  See:"
            " https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html"
            "\n\n"
        ),
        "  Apple: ",
        apple_archs,
        (
            "\n  See:"
            " https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf"
        ),
    )


# ===-----------------------------------------------------------------------===#
# _get_info_from_target
# ===-----------------------------------------------------------------------===#

# All supported target architectures in canonical form.
# This is the canonical list used for validation in _get_info_from_target.
# Normalization: "nvidia:80" -> "sm_80", "mi300x" -> "gfx942",
#                "amdgpu:gfx942" -> "gfx942", "metal:4" -> "apple-m4".
#
# SYNC: This list must stay in sync with the TargetTraits accelerator tables
#       in Mojo/lib/Target/. Run the following test to verify:
#       bazel test //Mojo/test/mojo-tool:build/verify_supported_accelerators_sync.mojo.test
comptime _all_targets = (
    StaticString("sm_52"),
    StaticString("sm_60"),
    StaticString("sm_61"),
    StaticString("sm_75"),
    StaticString("sm_80"),
    StaticString("sm_86"),
    StaticString("sm_87"),
    StaticString("sm_89"),
    StaticString("sm_90"),
    StaticString("sm_90a"),
    StaticString("sm_100"),
    StaticString("sm_100a"),
    StaticString("sm_103"),
    StaticString("sm_103a"),
    StaticString("sm_110"),
    StaticString("sm_110a"),
    StaticString("sm_120"),
    StaticString("sm_120a"),
    StaticString("sm_121"),
    StaticString("sm_121a"),
    StaticString("gfx90a"),
    StaticString("gfx942"),
    StaticString("mi300a"),
    StaticString("gfx950"),
    StaticString("gfx1030"),
    StaticString("gfx1033"),
    StaticString("gfx1100"),
    StaticString("gfx1101"),
    StaticString("gfx1102"),
    StaticString("gfx1103"),
    StaticString("gfx1150"),
    StaticString("gfx1151"),
    StaticString("gfx1152"),
    StaticString("gfx1200"),
    StaticString("gfx1201"),
    StaticString("apple-m1"),
    StaticString("apple-m1-metal4"),
    StaticString("apple-m2"),
    StaticString("apple-m2-metal4"),
    StaticString("apple-m3"),
    StaticString("apple-m3-metal4"),
    StaticString("apple-m4"),
    StaticString("apple-m4-metal4"),
    StaticString("apple-m5"),
    StaticString("apple-m5-metal4"),
    StaticString("cuda"),
)


@always_inline
def _get_info_from_target[target_arch0: StaticString]() -> GPUInfo:
    """Gets `GPUInfo` for a specific target architecture.

    Maps target architecture strings to corresponding `GPUInfo` instances.

    Parameters:
        target_arch0: Target architecture string (e.g., "sm_80", "gfx942").

    Returns:
        `GPUInfo` instance for the specified target architecture.
    """
    # Normalize the target architecture to canonical form.
    # NVIDIA: "nvidia:sm_90a" -> "sm_90a", "nvidia:sm90" -> "sm_90", "nvidia:80" -> "sm_80", "sm80" -> "sm_80"
    # AMD: "mi300x" -> "gfx942", "mi355x" -> "gfx950", "amdgpu:gfx942" -> "gfx942", "amd:gfx942" -> "gfx942"
    # Apple: "metal:4" -> "apple-m4"
    #
    # Every rule below must leave each `_all_targets` entry unchanged: these are
    # substring replacements, so a pattern that is a prefix of an already
    # canonical name corrupts it.
    comptime target_arch = (
        target_arch0
        # NVIDIA normalization
        .replace("nvidia:sm_", "sm_")
        .replace("nvidia:sm", "sm_")
        .replace("nvidia:", "sm_")
        .replace("sm", "sm_")
        .replace("sm__", "sm_")
        # AMD normalization. Both "amdgpu:" (LLVM/ROCm target prefix) and "amd:"
        # (vendor name) are accepted, mirroring the "nvidia:" prefix above.
        .replace("mi250x", "gfx90a")
        .replace("mi300x", "gfx942")
        .replace("mi355x", "gfx950")
        .replace("amdgpu:", "")
        .replace("amd:", "")
        # Apple normalization, general "metal:" → "apple-m" replacement.
        .replace("metal:", "apple-m")
    )

    comptime assert (
        StaticString(target_arch) in _all_targets
    ), _build_unsupported_arch_error[target_arch0]()

    comptime if target_arch == "sm_52":
        return materialize[GTX970]()
    elif target_arch == "sm_60":
        return materialize[TeslaP100]()
    elif target_arch == "sm_61":
        # FIXME GTX1060 and GTX1080Ti architecture wise are different (sm_count is different). We need to differentiate between them here at compile time.
        # return materialize[GTX1060]()
        return materialize[GTX1080Ti]()
    elif target_arch == "sm_75":
        return materialize[RTX2060]()
    elif target_arch == "sm_80":
        return materialize[A100]()
    elif target_arch == "sm_86":
        return materialize[A10]()
    elif target_arch == "sm_87":
        return materialize[OrinNano]()
    elif target_arch == "sm_89":
        return materialize[L4]()
    elif target_arch == "sm_90" or target_arch == "sm_90a":
        return materialize[H100]()
    elif target_arch == "sm_100" or target_arch == "sm_100a":
        # FIXME (KERN-1814): Unlike H100 and H200, blackwell devices (B100 vs B200)
        # architecture wise are different. We need to differentiate between them here.
        return materialize[B200]()
    elif target_arch == "sm_103" or target_arch == "sm_103a":
        return materialize[B300]()
    elif target_arch == "sm_110" or target_arch == "sm_110a":
        return materialize[JetsonThor]()
    elif target_arch == "sm_120" or target_arch == "sm_120a":
        return materialize[RTX5090]()
    elif target_arch == "sm_121" or target_arch == "sm_121a":
        return materialize[DGXSpark]()
    # AMD (gfx IDs; "mi250x"/"mi300x"/"mi355x" aliases are normalized above)
    elif target_arch == "gfx90a":
        return materialize[MI250X]()
    elif target_arch == "gfx942":
        return materialize[MI300X]()
    # MI300A shares the gfx942 ISA with MI300X but has fewer CUs and
    # unified host/device memory. Reached via explicit "mi300a" opt-in
    # (e.g. `GPUInfo.from_name["amdgpu:mi300a"]()`) since gfx942-only
    # detection cannot distinguish the two parts.
    elif target_arch == "mi300a":
        return materialize[MI300A]()
    elif target_arch == "gfx950":
        return materialize[MI355X]()
    elif target_arch == "gfx1030":
        return materialize[Radeon6900]()
    elif target_arch == "gfx1033":
        return materialize[SteamDeck]()
    elif target_arch == "gfx1100":
        return materialize[Radeon7900]()
    elif target_arch == "gfx1101":
        return materialize[Radeon7800]()
    elif target_arch == "gfx1102":
        return materialize[Radeon7600]()
    elif target_arch == "gfx1103":
        return materialize[Radeon780m]()
    elif target_arch == "gfx1150":
        return materialize[Radeon880m]()
    elif target_arch == "gfx1151":
        return materialize[Radeon8060s]()
    elif target_arch == "gfx1152":
        return materialize[Radeon860m]()
    elif target_arch == "gfx1200":
        return materialize[Radeon9060]()
    elif target_arch == "gfx1201":
        return materialize[Radeon9070]()
    elif target_arch == "apple-m1":
        return materialize[MetalM1]()
    elif target_arch == "apple-m1-metal4":
        return materialize[MetalM1Metal4]()
    elif target_arch == "apple-m2":
        return materialize[MetalM2]()
    elif target_arch == "apple-m2-metal4":
        return materialize[MetalM2Metal4]()
    elif target_arch == "apple-m3":
        return materialize[MetalM3]()
    elif target_arch == "apple-m3-metal4":
        return materialize[MetalM3Metal4]()
    elif target_arch == "apple-m4":
        return materialize[MetalM4]()
    elif target_arch == "apple-m4-metal4":
        return materialize[MetalM4Metal4]()
    elif target_arch == "apple-m5":
        return materialize[MetalM5]()
    elif target_arch == "apple-m5-metal4":
        return materialize[MetalM5Metal4]()
    # "cuda" means generic CUDA — use runtime GPU detection.
    elif target_arch == "cuda":
        return _get_info_from_target[_accelerator_arch()]()
    elif _accelerator_arch() == "":
        return materialize[NoGPU]()
    else:
        return _get_info_from_target[_accelerator_arch()]()


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


def is_gpu[target: StringSlice]() -> Bool:
    """Checks if the target is a GPU (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is a GPU, False otherwise.
    """
    return is_gpu(target)


def is_gpu(target: StringSlice) -> Bool:
    """Checks if the target is a GPU (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is a GPU, False otherwise.
    """
    return target == "gpu"


def is_cpu[target: StringSlice]() -> Bool:
    """Checks if the target is a CPU (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is a CPU, False otherwise.
    """
    return is_cpu(target)


def is_cpu(target: StringSlice) -> Bool:
    """Checks if the target is a CPU (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is a CPU, False otherwise.
    """
    return target == "cpu"


def is_accelerator[target: StringSlice]() -> Bool:
    """Checks if the target is an accelerator (compile-time version).

    True for any non-CPU compute target.

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is an accelerator, False otherwise.
    """
    return is_accelerator(target)


def is_accelerator(target: StringSlice) -> Bool:
    """Checks if the target is an accelerator (runtime version).

    True for any non-CPU compute target.

    Args:
        target: Target string to check.

    Returns:
        True if the target is an accelerator, False otherwise.
    """
    return is_gpu(target)


def is_valid_target[target: StringSlice]() -> Bool:
    """Checks if the target is valid (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is valid (CPU or GPU), False otherwise.
    """
    return is_valid_target(target)


def is_valid_target(target: StringSlice) -> Bool:
    """Checks if the target is valid (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is valid (CPU or GPU), False otherwise.
    """
    return is_cpu(target) or is_accelerator(target)
