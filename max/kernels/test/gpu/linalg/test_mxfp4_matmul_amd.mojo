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
"""Tests for the native MXFP4 block-scaled matmul kernel on AMD CDNA4.

Validates BlockScaledMatmulAMD against a per-element GPU reference
that uses the llvm.amdgcn.cvt.scalef32.pk.f32.fp4 intrinsic for FP4→FP32
dequantization and scalar accumulation.

Usage:
  mojo test_mxfp4_matmul_amd.mojo
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.memory import bitcast
from std.random import random_ui64
from std.sys.intrinsics import llvm_intrinsic

from internal_utils import assert_almost_equal
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import MXFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd.block_scaled_matmul_amd import (
    BlockScaledMatmulAMD,
    _launch_block_scaled_split_k,
)


# ===----------------------------------------------------------------------=== #
# Reference kernel: scalar FP4 dequant matmul on GPU
# ===----------------------------------------------------------------------=== #


def block_scaled_matmul_ref(
    a_ptr: ImmPointer[UInt8, ImmutAnyOrigin],
    b_ptr: ImmPointer[UInt8, ImmutAnyOrigin],
    a_scales_ptr: ImmPointer[Float8_e8m0fnu, ImmutAnyOrigin],
    b_scales_ptr: ImmPointer[Float8_e8m0fnu, ImmutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    M_dev: Int32,
    N_dev: Int32,
    K_dev: Int32,
):
    """Per-element GPU reference for MXFP4 block-scaled matmul.

    Each thread computes one (m, n) output element by dequantizing
    packed FP4 data via the CDNA4 cvt.scalef32.pk.f32.fp4 intrinsic,
    multiplying with E8M0 scales, and accumulating in FP32.
    """
    var M = Int(M_dev)
    var N = Int(N_dev)
    var K = Int(K_dev)

    @always_inline
    def cast_fp4x2_to_fp32x2[
        byte_select: Int
    ](packed: Int32, scale: Float32) -> SIMD[.float32, 2]:
        return llvm_intrinsic[
            "llvm.amdgcn.cvt.scalef32.pk.f32.fp4",
            SIMD[.float32, 2],
        ](packed, scale, Int32(byte_select))

    var m = global_idx.x
    var n = global_idx.y

    if m >= M or n >= N:
        return

    var k_groups = K // MXFP4_SF_VECTOR_SIZE

    var am_scales_ptr = a_scales_ptr + m * k_groups
    var bn_scales_ptr = b_scales_ptr + n * k_groups

    var am_ptr = a_ptr + m * (K // 2)
    var bn_ptr = b_ptr + n * (K // 2)

    var accum = SIMD[.float32, 2](0)

    for ko in range(k_groups):
        var a_scale = am_scales_ptr[ko].cast[.float32]()
        var b_scale = bn_scales_ptr[ko].cast[.float32]()

        for ki in range(0, MXFP4_SF_VECTOR_SIZE // 2, 4):
            var a_data = bitcast[.int32, 1](am_ptr.load[width=4](ki))
            var b_data = bitcast[.int32, 1](bn_ptr.load[width=4](ki))

            comptime for byte_select in range(4):
                accum += cast_fp4x2_to_fp32x2[byte_select](
                    a_data, a_scale
                ) * cast_fp4x2_to_fp32x2[byte_select](b_data, b_scale)

        am_ptr += MXFP4_SF_VECTOR_SIZE // 2
        bn_ptr += MXFP4_SF_VECTOR_SIZE // 2

    c_ptr[m * N + n] = accum.reduce_add()


# ===----------------------------------------------------------------------=== #
# Test harness
# ===----------------------------------------------------------------------=== #


def test_mxfp4_matmul[
    M_static: Int,
    N_static: Int,
    K_static: Int,
    BM: Int = 128,
    BN: Int = 128,
    BK_ELEMS: Int = 128,
    WM: Int = 64,
    WN: Int = 64,
    MMA_M: Int = 16,
    MMA_N: Int = 16,
    MMA_K: Int = 128,
](ctx: DeviceContext) raises:
    """Test BlockScaledMatmulAMD against a GPU reference kernel.

    Launches BlockScaledMatmulAMD directly with the provided BM/BN/BK_ELEMS/WM/WN
    and MMA shape. Defaults match the current production tile config and
    the 16x16x128 MFMA shape.

    Parameters:
        M_static: Number of rows in A / C.
        N_static: Number of rows in B (transposed) / cols in C.
        K_static: Logical K dimension (FP4 elements, must be multiple of 128).
        BM: Block tile rows.
        BN: Block tile cols.
        BK_ELEMS: Block tile K in logical FP4 elements.
        WM: Warp tile rows.
        WN: Warp tile cols.
        MMA_M: MFMA tile rows. Default 16.
        MMA_N: MFMA tile cols. Default 16.
        MMA_K: MFMA K-depth in logical FP4 elements. Default 128.
    """
    comptime assert (
        K_static % 128 == 0
    ), "K must be a multiple of 128 (MFMA K dimension)"
    comptime assert (
        K_static % MXFP4_SF_VECTOR_SIZE == 0
    ), "K must be a multiple of MXFP4_SF_VECTOR_SIZE (32)"
    comptime assert BK_ELEMS % 128 == 0, "BK_ELEMS must be a multiple of 128"
    comptime assert BM % WM == 0, "BM must be divisible by WM"
    comptime assert BN % WN == 0, "BN must be divisible by WN"

    print(
        M_static,
        "x",
        N_static,
        "x",
        K_static,
        " [BM=",
        BM,
        " BN=",
        BN,
        " BK=",
        BK_ELEMS,
        " WM=",
        WM,
        " WN=",
        WN,
        " MMA=",
        MMA_M,
        "x",
        MMA_N,
        "x",
        MMA_K,
        "]",
    )

    comptime input_dtype = DType.uint8
    comptime scales_dtype = DType.float8_e8m0fnu
    comptime output_dtype = DType.float32

    comptime K_PACKED = K_static // 2
    comptime K_SCALES = K_static // MXFP4_SF_VECTOR_SIZE

    comptime a_size = M_static * K_PACKED
    comptime b_size = N_static * K_PACKED
    comptime c_size = M_static * N_static
    comptime a_scales_size = M_static * K_SCALES
    comptime b_scales_size = N_static * K_SCALES

    comptime a_shape = row_major[M_static, K_PACKED]()
    comptime b_shape = row_major[N_static, K_PACKED]()
    comptime c_shape = row_major[M_static, N_static]()
    comptime a_scales_shape = row_major[M_static, K_SCALES]()
    comptime b_scales_shape = row_major[N_static, K_SCALES]()

    var a_host = ctx.enqueue_create_host_buffer[input_dtype](a_size)
    var b_host = ctx.enqueue_create_host_buffer[input_dtype](b_size)
    var a_scales_host = ctx.enqueue_create_host_buffer[scales_dtype](
        a_scales_size
    )
    var b_scales_host = ctx.enqueue_create_host_buffer[scales_dtype](
        b_scales_size
    )
    var c_host = ctx.enqueue_create_host_buffer[output_dtype](c_size)
    var c_host_ref = ctx.enqueue_create_host_buffer[output_dtype](c_size)

    for i in range(a_size):
        a_host[i] = UInt8(random_ui64(0, 255))
    for i in range(b_size):
        b_host[i] = UInt8(random_ui64(0, 255))
    for i in range(a_scales_size):
        a_scales_host[i] = bitcast[scales_dtype](UInt8(random_ui64(125, 129)))
    for i in range(b_scales_size):
        b_scales_host[i] = bitcast[scales_dtype](UInt8(random_ui64(125, 129)))

    var a_dev = ctx.enqueue_create_buffer[input_dtype](a_size)
    var b_dev = ctx.enqueue_create_buffer[input_dtype](b_size)
    var a_scales_dev = ctx.enqueue_create_buffer[scales_dtype](a_scales_size)
    var b_scales_dev = ctx.enqueue_create_buffer[scales_dtype](b_scales_size)
    var c_dev = ctx.enqueue_create_buffer[output_dtype](c_size)
    var c_ref_dev = ctx.enqueue_create_buffer[output_dtype](c_size)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(a_scales_dev, a_scales_host)
    ctx.enqueue_copy(b_scales_dev, b_scales_host)

    var a_tt = TileTensor(a_dev, a_shape).as_immut()
    var b_tt = TileTensor(b_dev, b_shape).as_immut()
    var c_tt = TileTensor(c_dev, c_shape)
    var a_scales_tt = TileTensor(a_scales_dev, a_scales_shape).as_immut()
    var b_scales_tt = TileTensor(b_scales_dev, b_scales_shape).as_immut()

    # --- Direct launch with explicit tile params ---
    comptime Kernel = BlockScaledMatmulAMD[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        WM=WM,
        WN=WN,
        MMA_M=MMA_M,
        MMA_N=MMA_N,
        MMA_K=MMA_K,
    ]
    comptime kernel = Kernel.run[
        .float32,
        type_of(c_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(a_scales_tt).LayoutType,
        type_of(b_scales_tt).LayoutType,
        type_of(c_tt).Engine,
        type_of(a_tt).Engine,
        type_of(b_tt).Engine,
        type_of(a_scales_tt).Engine,
        type_of(b_scales_tt).Engine,
    ]
    ctx.enqueue_function[kernel](
        c_tt,
        a_tt,
        b_tt,
        a_scales_tt,
        b_scales_tt,
        grid_dim=(ceildiv(N_static, BN), ceildiv(M_static, BM)),
        block_dim=Kernel.num_threads,
    )

    # --- Reference ---
    comptime BLOCK_DIM = 32
    ctx.enqueue_function[block_scaled_matmul_ref](
        a_dev,
        b_dev,
        a_scales_dev,
        b_scales_dev,
        c_ref_dev,
        Int32(M_static),
        Int32(N_static),
        Int32(K_static),
        grid_dim=(ceildiv(M_static, BLOCK_DIM), ceildiv(N_static, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_host_ref, c_ref_dev)
    ctx.synchronize()

    assert_almost_equal(
        c_host.unsafe_ptr(),
        c_host_ref.unsafe_ptr(),
        c_size,
        atol=0.05,
        rtol=0.05,
    )

    print("  PASSED")


def test_mxfp4_matmul_split_k[
    M_static: Int,
    N_static: Int,
    K_static: Int,
    num_splits: Int,
    BM: Int = 64,
    BN: Int = 128,
    BK_ELEMS: Int = 256,
    WM: Int = 64,
    WN: Int = 32,
](ctx: DeviceContext) raises:
    """Test the inter-block split-K launcher against the GPU reference.

    Launches `_launch_block_scaled_split_k` (workspace + reduce path) for the
    given `num_splits` and verifies bit-exactness against the same scalar
    dequant reference used by `test_mxfp4_matmul`. Covers both the K-band
    accumulation and the reduce-kernel sum/cast.
    """
    print(
        M_static,
        "x",
        N_static,
        "x",
        K_static,
        " [split-K num_splits=",
        num_splits,
        " BM=",
        BM,
        " BN=",
        BN,
        " BK=",
        BK_ELEMS,
        " WN=",
        WN,
        "]",
    )

    comptime input_dtype = DType.uint8
    comptime scales_dtype = DType.float8_e8m0fnu
    comptime output_dtype = DType.float32

    comptime K_PACKED = K_static // 2
    comptime K_SCALES = K_static // MXFP4_SF_VECTOR_SIZE

    comptime a_size = M_static * K_PACKED
    comptime b_size = N_static * K_PACKED
    comptime c_size = M_static * N_static
    comptime a_scales_size = M_static * K_SCALES
    comptime b_scales_size = N_static * K_SCALES

    comptime a_shape = row_major[M_static, K_PACKED]()
    comptime b_shape = row_major[N_static, K_PACKED]()
    comptime c_shape = row_major[M_static, N_static]()
    comptime a_scales_shape = row_major[M_static, K_SCALES]()
    comptime b_scales_shape = row_major[N_static, K_SCALES]()

    var a_host = ctx.enqueue_create_host_buffer[input_dtype](a_size)
    var b_host = ctx.enqueue_create_host_buffer[input_dtype](b_size)
    var a_scales_host = ctx.enqueue_create_host_buffer[scales_dtype](
        a_scales_size
    )
    var b_scales_host = ctx.enqueue_create_host_buffer[scales_dtype](
        b_scales_size
    )
    var c_host = ctx.enqueue_create_host_buffer[output_dtype](c_size)
    var c_host_ref = ctx.enqueue_create_host_buffer[output_dtype](c_size)

    for i in range(a_size):
        a_host[i] = UInt8(random_ui64(0, 255))
    for i in range(b_size):
        b_host[i] = UInt8(random_ui64(0, 255))
    for i in range(a_scales_size):
        a_scales_host[i] = bitcast[scales_dtype](UInt8(random_ui64(125, 129)))
    for i in range(b_scales_size):
        b_scales_host[i] = bitcast[scales_dtype](UInt8(random_ui64(125, 129)))

    var a_dev = ctx.enqueue_create_buffer[input_dtype](a_size)
    var b_dev = ctx.enqueue_create_buffer[input_dtype](b_size)
    var a_scales_dev = ctx.enqueue_create_buffer[scales_dtype](a_scales_size)
    var b_scales_dev = ctx.enqueue_create_buffer[scales_dtype](b_scales_size)
    var c_dev = ctx.enqueue_create_buffer[output_dtype](c_size)
    var c_ref_dev = ctx.enqueue_create_buffer[output_dtype](c_size)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(a_scales_dev, a_scales_host)
    ctx.enqueue_copy(b_scales_dev, b_scales_host)

    var a_tt = TileTensor(a_dev, a_shape).as_immut()
    var b_tt = TileTensor(b_dev, b_shape).as_immut()
    var c_tt = TileTensor(c_dev, c_shape)
    var a_scales_tt = TileTensor(a_scales_dev, a_scales_shape).as_immut()
    var b_scales_tt = TileTensor(b_scales_dev, b_scales_shape).as_immut()

    # --- Split-K launch (workspace + reduce path) ---
    _launch_block_scaled_split_k[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        WM=WM,
        WN=WN,
        num_splits=num_splits,
    ](c_tt, a_tt, b_tt, a_scales_tt, b_scales_tt, M_static, ctx)

    # --- Reference ---
    comptime BLOCK_DIM = 32
    ctx.enqueue_function[block_scaled_matmul_ref](
        a_dev,
        b_dev,
        a_scales_dev,
        b_scales_dev,
        c_ref_dev,
        Int32(M_static),
        Int32(N_static),
        Int32(K_static),
        grid_dim=(ceildiv(M_static, BLOCK_DIM), ceildiv(N_static, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_host_ref, c_ref_dev)
    ctx.synchronize()

    assert_almost_equal(
        c_host.unsafe_ptr(),
        c_host_ref.unsafe_ptr(),
        c_size,
        atol=0.05,
        rtol=0.05,
    )

    print("  PASSED")


def main() raises:
    with DeviceContext() as ctx:
        print("===> MXFP4 block-scaled matmul (native CDNA4 MFMA)")

        # === Bucket A: baseline aligned correctness ===
        print("\n--- A: baseline aligned shapes ---")

        test_mxfp4_matmul[128, 128, 128](ctx)
        test_mxfp4_matmul[128, 128, 256](ctx)
        test_mxfp4_matmul[128, 128, 512](ctx)
        test_mxfp4_matmul[256, 128, 256](ctx)
        test_mxfp4_matmul[128, 256, 256](ctx)
        test_mxfp4_matmul[256, 256, 256](ctx)
        test_mxfp4_matmul[256, 256, 512](ctx)
        test_mxfp4_matmul[128, 128, 1024](ctx)

        # === Bucket B: Kimi K2.5 unaligned-M OOB stress matrix ===
        print("\n--- B: Kimi K2.5 unaligned-M OOB stress ---")

        # M=1 (decode, single row) across all projections.
        test_mxfp4_matmul[1, 7168, 2048](ctx)
        test_mxfp4_matmul[1, 2048, 7168](ctx)
        test_mxfp4_matmul[1, 4096, 7168](ctx)
        test_mxfp4_matmul[1, 7168, 18432](ctx)
        test_mxfp4_matmul[1, 18432, 7168](ctx)
        test_mxfp4_matmul[1, 36864, 7168](ctx)

        # M=17 (short prefill)
        test_mxfp4_matmul[17, 7168, 2048](ctx)
        test_mxfp4_matmul[17, 2048, 7168](ctx)
        test_mxfp4_matmul[17, 4096, 7168](ctx)
        test_mxfp4_matmul[17, 18432, 7168](ctx)

        # M=53 (mid-range unaligned)
        test_mxfp4_matmul[53, 7168, 2048](ctx)
        test_mxfp4_matmul[53, 7168, 18432](ctx)

        # M=73 (mid-range unaligned)
        test_mxfp4_matmul[73, 4096, 7168](ctx)
        test_mxfp4_matmul[73, 7168, 18432](ctx)
        test_mxfp4_matmul[73, 36864, 7168](ctx)

        # M=111 (near BM=128 boundary — last block is 111 rows, 17-row short)
        test_mxfp4_matmul[111, 7168, 2048](ctx)
        test_mxfp4_matmul[111, 2048, 7168](ctx)
        test_mxfp4_matmul[111, 18432, 7168](ctx)

        # M=129 (crosses 1 full block + 1-row partial)
        test_mxfp4_matmul[129, 7168, 2048](ctx)
        test_mxfp4_matmul[129, 4096, 7168](ctx)

        # M=257 (crosses 2 full blocks + 1-row partial)
        test_mxfp4_matmul[257, 7168, 2048](ctx)
        test_mxfp4_matmul[257, 18432, 7168](ctx)

        print("\n--- B': exaggerated OOB stress ---")

        # M = BM - 1 — last block is one row short of full.
        # Maximum partial-block footprint (127 real rows, 1 OOB).
        test_mxfp4_matmul[127, 7168, 2048](ctx)
        test_mxfp4_matmul[127, 36864, 7168](ctx)

        # M = 2*BM - 1 — one full block + 127-row partial.
        test_mxfp4_matmul[255, 7168, 8192](ctx)

        # M=1 + huge N + deepest K — maximum DRAM/scale volume with 1
        # real row and 127 OOB rows per block.
        test_mxfp4_matmul[1, 36864, 18432](ctx)
        print("\n--- T: tile-shape parameter sweep ---")

        # Baseline (same as default Kernel) — sanity check.
        test_mxfp4_matmul[128, 128, 512, BM=128, BN=128, BK_ELEMS=128](ctx)

        # Deeper BK: num_k_tiles=2, enables Level 1 intra-BK pipelining.
        test_mxfp4_matmul[128, 128, 512, BM=128, BN=128, BK_ELEMS=256](ctx)
        test_mxfp4_matmul[256, 256, 1024, BM=128, BN=128, BK_ELEMS=256](ctx)

        # Wider M block: 8 warps/block, same warp tile.
        test_mxfp4_matmul[256, 128, 512, BM=256, BN=128, BK_ELEMS=128](ctx)
        test_mxfp4_matmul[512, 128, 1024, BM=256, BN=128, BK_ELEMS=128](ctx)

        # Wider N block.
        test_mxfp4_matmul[128, 256, 512, BM=128, BN=256, BK_ELEMS=128](ctx)

        # Biggest block we can run at 1024 threads/workgroup: 256×256
        # with WM=WN=64 = 16 warps = 1024 threads (at the limit).
        test_mxfp4_matmul[256, 256, 512, BM=256, BN=256, BK_ELEMS=128](ctx)

        # Combined: bigger block + deeper BK (the most-likely-fastest
        # config for Kimi medium-M shapes).
        test_mxfp4_matmul[256, 128, 1024, BM=256, BN=128, BK_ELEMS=256](ctx)

        # Partial-block with non-default tile: makes sure OOB handling
        # scales with BM.
        test_mxfp4_matmul[73, 4096, 7168, BM=256, BN=128, BK_ELEMS=128](ctx)

        print("\n--- T2: small-M tuning configs (BM=64, BN=32, WN=32) ---")

        # K=2048 → K_BYTES=1024. Verify at each BK_ELEMS that K divides.
        # 128 → 1024/64 = 16 iters (÷) ✓
        test_mxfp4_matmul[
            32,
            7168,
            2048,
            BM=64,
            BN=32,
            BK_ELEMS=128,
            WN=32,
        ](ctx)
        # 256 → 1024/128 = 8 iters ✓
        test_mxfp4_matmul[
            32,
            7168,
            2048,
            BM=64,
            BN=32,
            BK_ELEMS=256,
            WN=32,
        ](ctx)
        # 512 → 1024/256 = 4 iters ✓
        test_mxfp4_matmul[
            32,
            7168,
            2048,
            BM=64,
            BN=32,
            BK_ELEMS=512,
            WN=32,
        ](ctx)

        test_mxfp4_matmul[
            32,
            7168,
            2048,
            BM=64,
            BN=32,
            BK_ELEMS=1024,
            WN=32,
        ](ctx)

        print("\n--- T3: 32x32x64 MFMA shape ---")

        # Aligned square shapes — sanity that the 32x32 path produces
        # correct results across one and multiple block tiles.
        test_mxfp4_matmul[128, 128, 128, MMA_M=32, MMA_N=32, MMA_K=64](ctx)
        test_mxfp4_matmul[256, 256, 256, MMA_M=32, MMA_N=32, MMA_K=64](ctx)

        # Partial-M (73 rows) — exercises buffer_store OOB clamp on the
        # new mfma32=True path.
        test_mxfp4_matmul[73, 128, 256, MMA_M=32, MMA_N=32, MMA_K=64](ctx)

        # Single-row decode regime — most aggressive M-tail.
        test_mxfp4_matmul[1, 128, 256, MMA_M=32, MMA_N=32, MMA_K=64](ctx)

        # Production-scale K — exercises the SCALE_WORDS_PER_ROW loader
        # across many BK iterations at a Kimi K2.5 shape. Mirrors the
        # 73x4096x7168 case in bucket B but at the 32x32x64 MFMA shape.
        test_mxfp4_matmul[73, 4096, 7168, MMA_M=32, MMA_N=32, MMA_K=64](ctx)

        print("\n--- SK: inter-block split-K (workspace + reduce) ---")

        # Production Kimi K2.5 shapes with the split factors the dispatch
        # heuristic picks: up-proj N=4096,K=7168 → 14-way; down-proj
        # N=7168,K=2048 → 8-way. BK_ELEMS=256 → BK_BYTES=128, so K_BYTES
        # // num_splits must be a multiple of 128.
        #   7168→K_BYTES=3584; 3584/14=256 (mult of 128) ✓
        #   2048→K_BYTES=1024; 1024/8 =128 (mult of 128) ✓
        test_mxfp4_matmul_split_k[1, 4096, 7168, num_splits=14](ctx)
        test_mxfp4_matmul_split_k[16, 4096, 7168, num_splits=14](ctx)
        test_mxfp4_matmul_split_k[64, 4096, 7168, num_splits=14](ctx)
        test_mxfp4_matmul_split_k[1, 7168, 2048, num_splits=8](ctx)
        test_mxfp4_matmul_split_k[16, 7168, 2048, num_splits=8](ctx)
        test_mxfp4_matmul_split_k[64, 7168, 2048, num_splits=8](ctx)

        # Small split factors / single-tile-per-split edge.
        test_mxfp4_matmul_split_k[1, 7168, 2048, num_splits=2](ctx)
        test_mxfp4_matmul_split_k[32, 4096, 7168, num_splits=4](ctx)

        # Unaligned-M OOB stress under split-K (M not a multiple of BM=64):
        # exercises the per-split [M, N] buffer-descriptor row clamp.
        test_mxfp4_matmul_split_k[17, 4096, 7168, num_splits=14](ctx)
        test_mxfp4_matmul_split_k[63, 7168, 2048, num_splits=8](ctx)

        print("\n--- SK16: narrow-M split-K tile (BM=16, WM=16, WN=64) ---")

        # The M<=16 decode bucket uses a 2-warp BM=16 tile (BN=128, WN=64 →
        # num_warps_n=2, num_m_mmas=1). Verify it against the scalar
        # reference for the production Kimi shapes at the split factors the
        # dispatch heuristic picks (up→14-way, down→4-way), including an
        # M < BM partial (M=7) to exercise the per-split row clamp.
        test_mxfp4_matmul_split_k[
            1, 4096, 7168, num_splits=14, BM=16, BN=128, WM=16, WN=64
        ](ctx)
        test_mxfp4_matmul_split_k[
            16, 4096, 7168, num_splits=14, BM=16, BN=128, WM=16, WN=64
        ](ctx)
        test_mxfp4_matmul_split_k[
            7, 4096, 7168, num_splits=14, BM=16, BN=128, WM=16, WN=64
        ](ctx)
        test_mxfp4_matmul_split_k[
            1, 7168, 2048, num_splits=4, BM=16, BN=128, WM=16, WN=64
        ](ctx)
        test_mxfp4_matmul_split_k[
            16, 7168, 2048, num_splits=4, BM=16, BN=128, WM=16, WN=64
        ](ctx)

        print("\n==== All MXFP4 block-scaled matmul tests passed ====")
