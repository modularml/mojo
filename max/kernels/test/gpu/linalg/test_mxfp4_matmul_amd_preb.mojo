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
"""Direct kernel-level tests for `BlockScaledMatmulAMD_PreB`.

Bypasses the grouped dispatcher (which still passes row-major scales until
the Phase-2 wiring lands) and exercises the kernel against a per-element
GPU reference (`block_scaled_matmul_amd`). Scales are preshuffled
host-side via `_preshuffle_scales_host`, mirroring `PreshuffledScaleLoader`'s
address math.

Usage:
  br test_mxfp4_matmul_amd_preb.mojo.test
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, global_idx
from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.memory import CacheOperation
from max.gpu.host.info import MI355X
from std.math import ceildiv
from std.memory import bitcast
from std.random import random_ui64, seed
from std.sys.intrinsics import llvm_intrinsic
from std.utils import StaticTuple

from internal_utils import assert_almost_equal
from layout import (
    Coord,
    Idx,
    TensorLayout,
    TensorEngine,
    TileTensor,
    row_major,
)
from linalg.fp4_utils import MXFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import BlockScaledMatmulAMD_PreB, Shuffler


# ===----------------------------------------------------------------------=== #
# Per-element GPU reference (dequant + scalar accumulate). Uses
# `llvm.amdgcn.cvt.scalef32.pk.f32.fp4` rather than MFMA, so a bug common
# to all MFMA paths shows up here. Copied from test_mxfp4_matmul_amd.mojo.
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
    """Per-element GPU reference for MXFP4 block-scaled matmul."""
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
# Grid wrapper kernel: drives BlockScaledMatmulAMD_PreB.run with a 2D grid where
# block_idx.x = n_tile and block_idx.y = m_tile (mirrors the dispatcher's
# direct-mode launch).
# ===----------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(
            BlockScaledMatmulAMD_PreB[
                BM=BM, BN=BN, BK_ELEMS=BK_ELEMS, WN=WN, b_prefetch=b_prefetch
            ].num_threads
        )
    )
)
def _preb_grid_kernel[
    BM: Int,
    BN: Int,
    BK_ELEMS: Int,
    WN: Int,
    b_prefetch: Bool,
    dram_to_lds: Bool,
    b_cache_policy: CacheOperation,
    cluster_drain_sched: Bool,
    mfma_cluster: Int,
    out_dtype: DType,
    LayoutC: TensorLayout,
    LayoutA: TensorLayout,
    LayoutBPre: TensorLayout,
    LayoutSFA: TensorLayout,
    LayoutSFB: TensorLayout,
    CEngine: TensorEngine,
    AEngine: TensorEngine,
    BPreEngine: TensorEngine,
    SFAEngine: TensorEngine,
    SFBEngine: TensorEngine,
    N: Int,
    K_BYTES: Int,
](
    c: TileTensor[mut=True, out_dtype, LayoutC, MutAnyOrigin, Engine=CEngine],
    a: TileTensor[.uint8, LayoutA, ImmutAnyOrigin, Engine=AEngine],
    b_pre: TileTensor[.uint8, LayoutBPre, ImmutAnyOrigin, Engine=BPreEngine],
    sfa: TileTensor[
        .float8_e8m0fnu, LayoutSFA, ImmutAnyOrigin, Engine=SFAEngine
    ],
    sfb: TileTensor[
        .float8_e8m0fnu, LayoutSFB, ImmutAnyOrigin, Engine=SFBEngine
    ],
):
    BlockScaledMatmulAMD_PreB[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        WN=WN,
        b_prefetch=b_prefetch,
        b_cache_policy=b_cache_policy,
        dram_to_lds=dram_to_lds,
        cluster_drain_sched=cluster_drain_sched,
        mfma_cluster=mfma_cluster,
    ].run[
        out_dtype,
        LayoutC,
        LayoutA,
        LayoutBPre,
        LayoutSFA,
        LayoutSFB,
        CEngine,
        AEngine,
        BPreEngine,
        SFAEngine,
        SFBEngine,
        N,
        K_BYTES,
    ](
        c, a, b_pre, sfa, sfb, Int(block_idx.x), Int(block_idx.y)
    )


# ===----------------------------------------------------------------------=== #
# Test harness
# ===----------------------------------------------------------------------=== #


def _test_case[
    M_static: Int,
    N_static: Int,
    K_static: Int,
    BM: Int = 64,
    BN: Int = 128,
    BK_ELEMS: Int = 512,
    WN: Int = 64,
    b_prefetch: Bool = False,
    dram_to_lds: Bool = False,
    b_cache_policy: CacheOperation = CacheOperation.ALWAYS,
    cluster_drain_sched: Bool = False,
    mfma_cluster: Int = 4,
    DUMP_ASM: Bool = False,
](name: String, ctx: DeviceContext) raises:
    """One direct-launch correctness case for the preb kernel."""
    comptime assert K_static % 128 == 0, "K must be a multiple of 128"
    comptime assert N_static % BN == 0, "N_static must be a multiple of BN"
    comptime assert N_static % 32 == 0, "N must be a multiple of 32 (mn_pack=2)"

    comptime packed_K = K_static // 2
    comptime scale_K = K_static // MXFP4_SF_VECTOR_SIZE

    # M can be unaligned wrt BM and the scale-cell stride (32). The grid
    # uses ceildiv (OOB rows in the trailing block are clamped by the A V#
    # and the C V#), and the preshuffled A-scale buffer is zero-padded to
    # the next multiple of 32 — same shape as per-expert padding in prod.
    comptime padded_M = ceildiv(M_static, 32) * 32

    print(
        "  ",
        name,
        " M=",
        M_static,
        " N=",
        N_static,
        " K=",
        K_static,
        " BM=",
        BM,
        " BN=",
        BN,
        " BK_ELEMS=",
        BK_ELEMS,
        " WN=",
        WN,
        " b_prefetch=",
        b_prefetch,
    )

    # ---- Host buffers + random init ----
    # Scale buffers held as uint8 throughout (E8M0 is byte-equivalent) so
    # we can call `Shuffler[1].preshuffle_scale_4d` directly; we bitcast
    # back to float8_e8m0fnu at the reference / preb kernel call sites.
    var a_h = ctx.enqueue_create_host_buffer[.uint8](M_static * packed_K)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](N_static * packed_K)
    var sfa_h = ctx.enqueue_create_host_buffer[.uint8](M_static * scale_K)
    var sfb_h = ctx.enqueue_create_host_buffer[.uint8](N_static * scale_K)
    var sfa_pre_h = ctx.enqueue_create_host_buffer[.uint8](padded_M * scale_K)
    var sfb_pre_h = ctx.enqueue_create_host_buffer[.uint8](N_static * scale_K)
    ctx.synchronize()

    for i in range(M_static * packed_K):
        a_h[i] = UInt8(random_ui64(0, 255))
    for i in range(N_static * packed_K):
        b_h[i] = UInt8(random_ui64(0, 255))
    # Clamp E8M0 to [125..129] = magnitudes [0.25..4] to keep f32 in range.
    for i in range(M_static * scale_K):
        sfa_h[i] = UInt8(random_ui64(125, 129))
    for i in range(N_static * scale_K):
        sfb_h[i] = UInt8(random_ui64(125, 129))

    var sfa_h_tt = TileTensor(
        sfa_h, row_major(Coord(Idx[1], Idx[M_static], Idx[scale_K]))
    )
    var sfb_h_tt = TileTensor(
        sfb_h, row_major(Coord(Idx[1], Idx[N_static], Idx[scale_K]))
    )
    _ = Shuffler[1].preshuffle_scale_4d[MN=M_static, K_SCALES=scale_K](
        sfa_h_tt, sfa_pre_h
    )
    _ = Shuffler[1].preshuffle_scale_4d[MN=N_static, K_SCALES=scale_K](
        sfb_h_tt, sfb_pre_h
    )

    # ---- Device buffers + upload ----
    var a_d = ctx.enqueue_create_buffer[.uint8](M_static * packed_K)
    var b_d = ctx.enqueue_create_buffer[.uint8](N_static * packed_K)
    var b_pre_d = ctx.enqueue_create_buffer[.uint8](N_static * packed_K)
    var sfa_d = ctx.enqueue_create_buffer[.uint8](M_static * scale_K)
    var sfb_d = ctx.enqueue_create_buffer[.uint8](N_static * scale_K)
    var sfa_pre_d = ctx.enqueue_create_buffer[.uint8](padded_M * scale_K)
    var sfb_pre_d = ctx.enqueue_create_buffer[.uint8](N_static * scale_K)
    var c_d = ctx.enqueue_create_buffer[.float32](M_static * N_static)
    var c_ref_d = ctx.enqueue_create_buffer[.float32](M_static * N_static)
    c_d.enqueue_fill(Float32(0.0))
    c_ref_d.enqueue_fill(Float32(0.0))

    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(sfa_d, sfa_h)
    ctx.enqueue_copy(sfb_d, sfb_h)
    ctx.enqueue_copy(sfa_pre_d, sfa_pre_h)
    ctx.enqueue_copy(sfb_pre_d, sfb_pre_h)

    # ---- GPU preshuffle B → b_pre_d ----
    var b_raw_tt = TileTensor(
        b_d, row_major[1, N_static, packed_K]()
    ).as_immut()
    var b_pre_dst_tt = TileTensor(
        b_pre_d,
        Shuffler[1].b_5d_grouped_layout[N=N_static, K_BYTES=packed_K],
    )
    Shuffler[1].preshuffle_b_5d[N=N_static, K_BYTES=packed_K](
        b_raw_tt, b_pre_dst_tt, ctx
    )

    # ---- Reference: per-element dequant + scalar accumulate.
    # Independent code path from the MFMA kernel under test — catches
    # bugs that would be hidden by a same-MFMA-family reference.
    comptime BLOCK_DIM = 32
    ctx.enqueue_function[block_scaled_matmul_ref](
        a_d,
        b_d,
        sfa_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        sfb_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        c_ref_d,
        Int32(M_static),
        Int32(N_static),
        Int32(K_static),
        grid_dim=(ceildiv(M_static, BLOCK_DIM), ceildiv(N_static, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    # ---- Preb kernel under test ----
    var a_tt = TileTensor(
        a_d, row_major(Coord(M_static, Idx[packed_K]))
    ).as_immut()
    var b_pre_tt = TileTensor(
        b_pre_d, row_major[1, N_static * packed_K]()
    ).as_immut()
    # Scales: pass the preshuffled buffers but wrap them with a row-major
    # layout — the kernel uses the underlying bytes through
    # `PreshuffledScaleLoader`, the TileTensor layout is unused.
    var sfa_tt = TileTensor(
        sfa_pre_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        row_major[padded_M, scale_K](),
    ).as_immut()
    var sfb_tt = TileTensor(
        sfb_pre_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        row_major[N_static, scale_K](),
    ).as_immut()
    var c_tt = TileTensor(c_d, row_major[M_static, N_static]())

    comptime kernel = _preb_grid_kernel[
        BM,
        BN,
        BK_ELEMS,
        WN,
        b_prefetch,
        dram_to_lds,
        b_cache_policy,
        cluster_drain_sched,
        mfma_cluster,
        .float32,
        type_of(c_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_pre_tt).LayoutType,
        type_of(sfa_tt).LayoutType,
        type_of(sfb_tt).LayoutType,
        type_of(c_tt).Engine,
        type_of(a_tt).Engine,
        type_of(b_pre_tt).Engine,
        type_of(sfa_tt).Engine,
        type_of(sfb_tt).Engine,
        N_static,
        packed_K,
    ]
    ctx.enqueue_function[kernel, dump_asm=DUMP_ASM](
        c_tt,
        a_tt,
        b_pre_tt,
        sfa_tt,
        sfb_tt,
        grid_dim=(N_static // BN, ceildiv(M_static, BM)),
        block_dim=BlockScaledMatmulAMD_PreB[
            BM=BM, BN=BN, BK_ELEMS=BK_ELEMS, WN=WN, b_prefetch=b_prefetch
        ].num_threads,
    )
    ctx.synchronize()

    # ---- Compare ----
    var c_h = ctx.enqueue_create_host_buffer[.float32](M_static * N_static)
    var c_ref_h = ctx.enqueue_create_host_buffer[.float32](M_static * N_static)
    ctx.enqueue_copy(c_h, c_d)
    ctx.enqueue_copy(c_ref_h, c_ref_d)
    ctx.synchronize()

    assert_almost_equal(
        c_h.unsafe_ptr(),
        c_ref_h.unsafe_ptr(),
        M_static * N_static,
        atol=0.05,
        rtol=0.05,
    )
    print("    PASS")

    _ = a_d^
    _ = b_d^
    _ = b_pre_d^
    _ = sfa_d^
    _ = sfb_d^
    _ = sfa_pre_d^
    _ = sfb_pre_d^
    _ = c_d^
    _ = c_ref_d^


# ===----------------------------------------------------------------------=== #
# Test matrix
# ===----------------------------------------------------------------------=== #


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "test_mxfp4_matmul_amd_preb requires MI355X"

    print("===> BlockScaledMatmulAMD_PreB — direct kernel correctness")

    # flydsl stage1 champion config: tile_m=32/n=128/k=256, 4 waves (WN=32),
    # b_nt=2 (STREAMING), 2-stage pipeline, K=7168 (28 K-steps). Set
    # DUMP_ASM=True here to dump the kernel ISA for comparison vs
    # stage1_champion.s.
    _test_case[
        8,
        128,
        7168,
        BM=32,
        BN=128,
        WN=32,
        BK_ELEMS=256,
        b_prefetch=True,
        b_cache_policy=CacheOperation.STREAMING,
    ]("champion config (STREAMING, WN=32)", ctx)

    _test_case[256, 256, 256, BM=64, BN=64, WN=64, BK_ELEMS=256](
        "single warp, no-prefetch", ctx
    )
    _test_case[
        256, 256, 256, BM=64, BN=64, WN=64, BK_ELEMS=256, b_prefetch=True
    ]("single warp, b_prefetch=True", ctx)

    _test_case[234, 1024, 512, BM=64, BN=128, WN=64, BK_ELEMS=256](
        "OOB test on M, no-prefetch", ctx
    )
    _test_case[
        234, 1024, 512, BM=64, BN=128, WN=64, BK_ELEMS=256, b_prefetch=True
    ]("OOB test on M, b_prefetch=True", ctx)

    _test_case[256, 1024, 512, BM=16, BN=64, WN=16, BK_ELEMS=256](
        "size 16 warp tile, no-prefetch", ctx
    )
    _test_case[
        256, 1024, 512, BM=16, BN=64, WN=16, BK_ELEMS=256, b_prefetch=True
    ]("size 16 warp tile, b_prefetch=True", ctx)

    _test_case[24, 1024, 2048, BM=64, BN=256, WN=64, BK_ELEMS=256](
        "Testing BM > M", ctx
    )
    _test_case[
        24, 1024, 2048, BM=64, BN=256, WN=64, BK_ELEMS=256, b_prefetch=True
    ]("Testing BM > M, b_prefetch=True", ctx)

    _test_case[1001, 1024, 2048, BM=64, BN=256, WN=64, BK_ELEMS=256](
        "Testing large M", ctx
    )
    _test_case[
        1001, 1024, 2048, BM=64, BN=256, WN=64, BK_ELEMS=256, b_prefetch=True
    ]("Testing large M, b_prefetch=True", ctx)

    # Default production tile (BK_ELEMS=512, num_k_mmas=4).
    _test_case[256, 1024, 2048, BM=64, BN=128, WN=64, BK_ELEMS=512](
        "default prod tile", ctx
    )
    _test_case[
        256, 1024, 2048, BM=64, BN=128, WN=64, BK_ELEMS=512, b_prefetch=True
    ]("default prod tile, b_prefetch=True", ctx)

    # WN=16 with partial M (decode shape).
    _test_case[3, 1024, 2048, BM=16, BN=64, WN=16, BK_ELEMS=256](
        "decode-shape, WN=16", ctx
    )
    _test_case[
        3, 1024, 2048, BM=16, BN=64, WN=16, BK_ELEMS=256, b_prefetch=True
    ]("decode-shape, WN=16, b_prefetch=True", ctx)

    # WN=16 only — N-side shrui without M-side (BM aligned, WN=16).
    _test_case[64, 64, 512, BM=64, BN=64, WN=16, BK_ELEMS=256](
        "WN=16 only", ctx
    )
    _test_case[64, 64, 512, BM=64, BN=64, WN=16, BK_ELEMS=256, b_prefetch=True](
        "WN=16 only, b_prefetch=True", ctx
    )

    # M=1 decode — smallest possible M with both tile-size variants.
    _test_case[1, 128, 512, BM=16, BN=64, WN=16, BK_ELEMS=256](
        "M=1 decode, BM=16", ctx
    )
    _test_case[1, 1024, 2048, BM=64, BN=128, WN=64, BK_ELEMS=256](
        "M=1 decode, BM=64", ctx
    )

    # dram_to_lds=True — direct DRAM->LDS DMA A staging (flydsl
    # dram_to_lds). Same shapes, both prefetch modes.
    _test_case[
        256, 256, 256, BM=64, BN=64, WN=64, BK_ELEMS=256, dram_to_lds=True
    ]("single warp, async", ctx)
    _test_case[
        256,
        256,
        256,
        BM=64,
        BN=64,
        WN=64,
        BK_ELEMS=256,
        b_prefetch=True,
        dram_to_lds=True,
    ]("single warp, async + b_prefetch", ctx)
    _test_case[
        256, 1024, 2048, BM=64, BN=128, WN=64, BK_ELEMS=512, dram_to_lds=True
    ]("default prod tile, async", ctx)
    _test_case[
        256,
        1024,
        2048,
        BM=64,
        BN=128,
        WN=64,
        BK_ELEMS=512,
        b_prefetch=True,
        dram_to_lds=True,
    ]("default prod tile, async + b_prefetch", ctx)
    _test_case[
        234, 1024, 512, BM=64, BN=128, WN=64, BK_ELEMS=256, dram_to_lds=True
    ]("OOB test on M, async", ctx)
    _test_case[
        3, 1024, 2048, BM=16, BN=64, WN=16, BK_ELEMS=256, dram_to_lds=True
    ]("decode-shape, WN=16, async", ctx)

    # cluster_drain_sched=True (b_prefetch only) — per-cluster setprio + partial
    # vmcnt staircase. Covers default prod tile (num_k_mmas=4), the smaller
    # decode tile (num_k_mmas=2), WN=16, and a non-default mfma_cluster.
    _test_case[
        256,
        1024,
        2048,
        BM=64,
        BN=128,
        WN=64,
        BK_ELEMS=512,
        b_prefetch=True,
        cluster_drain_sched=True,
    ]("default prod tile, cluster_drain_sched", ctx)
    _test_case[
        256,
        1024,
        2048,
        BM=64,
        BN=128,
        WN=64,
        BK_ELEMS=512,
        b_prefetch=True,
        cluster_drain_sched=True,
        mfma_cluster=2,
    ]("default prod tile, cluster_drain_sched, cluster=2", ctx)
    _test_case[
        3,
        1024,
        2048,
        BM=16,
        BN=64,
        WN=16,
        BK_ELEMS=256,
        b_prefetch=True,
        cluster_drain_sched=True,
    ]("decode-shape, WN=16, cluster_drain_sched", ctx)
    _test_case[
        256,
        256,
        256,
        BM=64,
        BN=64,
        WN=64,
        BK_ELEMS=256,
        b_prefetch=True,
        cluster_drain_sched=True,
    ]("single warp, cluster_drain_sched (1 tile)", ctx)
    _test_case[
        234,
        1024,
        512,
        BM=64,
        BN=128,
        WN=64,
        BK_ELEMS=256,
        b_prefetch=True,
        cluster_drain_sched=True,
    ]("OOB test on M, cluster_drain_sched", ctx)

    print("==== all preb direct kernel tests passed ====")
