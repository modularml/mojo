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

from std.sys import has_amd_gpu_accelerator

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from layout import ComptimeInt, RowMajorLayout
from layout.tensor_engine import DefaultEngine
from linalg.matmul.gpu import _amdgpu_matmul_config_from_block_shape
from linalg.matmul.gpu.amd import AMDMatmul, AMDPingPongMatmul, KernelConfig
from std.testing import assert_true

from std.utils.index import Index


struct RegisterCounts(Movable):
    """Register usage statistics from AMD GPU assembly."""

    var vgprs: Int
    var sgprs: Int
    var agprs: Int
    var vgpr_spills: Int
    var sgpr_spills: Int

    def __init__(out self):
        self.vgprs = 0
        self.sgprs = 0
        self.agprs = 0
        self.vgpr_spills = 0
        self.sgpr_spills = 0

    def print_summary(self):
        """Print register usage summary."""
        print("\nRegister usage:")
        print("  VGPRs: ", self.vgprs)
        print("  SGPRs: ", self.sgprs)
        print("  AGPRs: ", self.agprs)
        print("  VGPR spills: ", self.vgpr_spills)
        print("  SGPR spills: ", self.sgpr_spills)


def parse_directive_value(asm: String, directive: String) -> Int:
    """Parse numeric value from an assembly directive.

    Args:
        asm: The assembly string to search.
        directive: The directive name (e.g., ".vgpr_count").

    Returns:
        The parsed integer value, or 0 if not found.
    """
    if directive not in asm:
        return 0

    var directive_idx = asm.find(directive)
    var line_end = asm.find("\n", directive_idx)
    if line_end <= directive_idx:
        return 0

    var line = asm[byte=directive_idx:line_end]
    var colon_idx = line.find(":")
    if colon_idx < 0:
        return 0

    try:
        var num_part = line[byte = colon_idx + 1 :].strip()
        return Int(num_part)
    except:
        return 0


def parse_register_counts(asm: String) -> RegisterCounts:
    """Extract all register counts from AMD GPU assembly.

    Args:
        asm: The assembly string containing register directives.

    Returns:
        RegisterCounts struct with parsed values.
    """
    var counts = RegisterCounts()
    counts.vgprs = parse_directive_value(asm, ".vgpr_count")
    counts.sgprs = parse_directive_value(asm, ".sgpr_count")
    counts.agprs = parse_directive_value(asm, ".agpr_count")
    counts.vgpr_spills = parse_directive_value(asm, ".vgpr_spill_count")
    counts.sgpr_spills = parse_directive_value(asm, ".sgpr_spill_count")
    return counts^


def validate_register_counts(counts: RegisterCounts) raises:
    """Verify that spill counts are zero.

    Args:
        counts: The register counts to validate.

    Raises:
        If spill counts are non-zero.
    """
    assert_true(counts.vgpr_spills == 0, "VGPR spill count should be 0")
    assert_true(counts.sgpr_spills == 0, "SGPR spill count should be 0")


def compile_kernel_to_asm[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    M: Int,
    N: Int,
    K: Int,
    block_m: Int,
    block_n: Int,
]() raises -> String:
    """Compile AMD AMDMatmul kernel and return assembly.

    Returns:
        The compiled assembly string.
    """
    # Use the dispatch helper to get a valid configuration
    comptime config = _amdgpu_matmul_config_from_block_shape[
        c_type,
        a_type,
        b_type,
        transpose_b=True,
        K=K,
    ](Index(block_m, block_n))

    comptime a_tt_layout = RowMajorLayout[ComptimeInt[M], ComptimeInt[K]]
    comptime b_tt_layout = RowMajorLayout[ComptimeInt[N], ComptimeInt[K]]
    comptime c_tt_layout = RowMajorLayout[ComptimeInt[M], ComptimeInt[N]]

    comptime kernel = AMDMatmul[
        a_type,
        b_type,
        c_type,
        True,
        config,
    ].run[
        c_tt_layout,
        a_tt_layout,
        b_tt_layout,
        DefaultEngine[element_width=1],
        DefaultEngine[element_width=1],
        DefaultEngine[element_width=1],
    ]

    # Compile for AMD GPU
    var compiled = _compile_code[
        kernel,
        target=get_gpu_target["gfx950"](),
    ]()
    return compiled.asm


def compile_pingpong_kernel_to_asm[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    M: Int,
    N: Int,
    K: Int,
    block_k: Int,
    mma_k: Int,
]() raises -> String:
    """Compile AMD ping-pong matmul kernel and return assembly.

    Parameters:
        c_type: Output tensor data type.
        a_type: Input tensor A data type.
        b_type: Input tensor B data type.
        M: Matrix dimension M (rows of A and C).
        N: Matrix dimension N (columns of B and C).
        K: Matrix dimension K (columns of A, rows of B).
        block_k: The K dimension of the block shape.
        mma_k: The K dimension of the MMA shape.

    Returns:
        The compiled assembly string.
    """
    comptime a_tt_layout = RowMajorLayout[ComptimeInt[M], ComptimeInt[K]]
    comptime b_tt_layout = RowMajorLayout[ComptimeInt[N], ComptimeInt[K]]
    comptime c_tt_layout = RowMajorLayout[ComptimeInt[M], ComptimeInt[N]]

    comptime pingpong_config = KernelConfig(
        block_shape=Index(256, 256, block_k),
        warp_shape=Index(128, 64, block_k),
        mma_shape=Index(16, 16, mma_k),
    )

    comptime kernel = AMDPingPongMatmul[
        a_type,
        b_type,
        c_type,
        pingpong_config,
        enable_swizzle=True,
    ].run[
        a_tt_layout,
        b_tt_layout,
        c_tt_layout,
        DefaultEngine[element_width=1],
        DefaultEngine[element_width=1],
        DefaultEngine[element_width=1],
    ]

    # Compile for AMD GPU
    var compiled = _compile_code[
        kernel,
        target=get_gpu_target["gfx950"](),
    ]()
    return compiled.asm


def print_test_header[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_m: Int,
    block_n: Int,
]():
    """Print test configuration header."""
    print("=== AMD Matmul Assembly Check ===")
    print("Data types: A=", a_type, ", B=", b_type, ", C=", c_type, sep="")
    print("Block shape: BM=", block_m, ", BN=", block_n, sep="")


def test_matmul_config[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    block_m: Int,
    block_n: Int,
]() raises:
    """Test AMD matmul kernel compilation and verify register usage.

    Compiles the kernel, extracts register counts, and validates them.
    """
    print_test_header[a_type, b_type, c_type, block_m, block_n]()

    var asm = compile_kernel_to_asm[
        c_type,
        a_type,
        b_type,
        M=8192,
        N=8192,
        K=8192,
        block_m=block_m,
        block_n=block_n,
    ]()

    var counts = parse_register_counts(asm)
    counts.print_summary()
    validate_register_counts(counts)

    print("=== Assembly check passed ===\n")


def test_amd_matmul_bf16_max_config() raises:
    """Test AMD AMDMatmul assembly for BF16 with max config (256x256x64)."""
    print("== test_amd_matmul_bf16_max_config (256x256x64)")

    comptime if not has_amd_gpu_accelerator():
        print("Skipping test - AMD GPU not available")
        return

    test_matmul_config[
        .bfloat16,  # c_type
        .bfloat16,  # a_type
        .bfloat16,  # b_type
        block_m=256,
        block_n=256,
    ]()


def test_amd_matmul_fp8_max_config() raises:
    """Test AMD AMDMatmul assembly for FP8 with max config (256x256x128)."""
    print("== test_amd_matmul_fp8_max_config (256x256x128)")

    comptime if not has_amd_gpu_accelerator():
        print("Skipping test - AMD GPU not available")
        return

    test_matmul_config[
        .bfloat16,  # c_type
        .float8_e4m3fn,  # a_type
        .float8_e4m3fn,  # b_type
        block_m=256,
        block_n=256,
    ]()


def test_amd_pingpong_fp8_max_config() raises:
    """Test AMD ping-pong matmul kernel assembly for FP8 (256x256x128)."""
    print("== test_amd_pingpong_fp8_max_config (256x256x128)")

    comptime if not has_amd_gpu_accelerator():
        print("Skipping test - AMD GPU not available")
        return

    print("=== AMD Ping-Pong Matmul Assembly Check ===")
    print("Data types: A=float8_e4m3fn, B=float8_e4m3fn, C=bfloat16")
    print("Block shape: BM=256, BN=256, BK=128")
    print("MMA shape: 16x16x128")

    var asm = compile_pingpong_kernel_to_asm[
        .bfloat16,  # c_type
        .float8_e4m3fn,  # a_type
        .float8_e4m3fn,  # b_type
        M=8192,
        N=8192,
        K=256,
        block_k=128,
        mma_k=128,
    ]()

    var counts = parse_register_counts(asm)
    counts.print_summary()
    validate_register_counts(counts)

    print("=== Assembly check passed ===\n")


def test_amd_pingpong_bf16_max_config() raises:
    """Test AMD ping-pong matmul kernel assembly for BF16 (256x256x64)."""
    print("== test_amd_pingpong_bf16_max_config (256x256x64)")

    comptime if not has_amd_gpu_accelerator():
        print("Skipping test - AMD GPU not available")
        return

    print("=== AMD Ping-Pong Matmul Assembly Check ===")
    print("Data types: A=bfloat16, B=bfloat16, C=bfloat16")
    print("Block shape: BM=256, BN=256, BK=64")
    print("MMA shape: 16x16x32")

    var asm = compile_pingpong_kernel_to_asm[
        .bfloat16,  # c_type
        .bfloat16,  # a_type
        .bfloat16,  # b_type
        M=8192,
        N=8192,
        K=256,
        block_k=64,
        mma_k=32,
    ]()

    var counts = parse_register_counts(asm)
    counts.print_summary()
    validate_register_counts(counts)

    print("=== Assembly check passed ===\n")


def main() raises:
    test_amd_matmul_bf16_max_config()
    test_amd_matmul_fp8_max_config()
    test_amd_pingpong_fp8_max_config()
    test_amd_pingpong_bf16_max_config()
