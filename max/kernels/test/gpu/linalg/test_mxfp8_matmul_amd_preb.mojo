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
"""Direct kernel-level tests for `BlockScaledMatmulAMD_PreB` at `matrix_format=CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3` (MXFP8).

MXFP8 sibling of `test_mxfp4_matmul_amd_preb.mojo`: same 16x16x128 f8f6f4 MFMA and
the same preshuffled layouts, but one byte per element (so `K_BYTES == K`) and the
E4M3 operand format. `BK_ELEMS` counts ELEMENTS, so 256 here holds the same byte
footprint as 512 on the FP4 path. The reference is a per-element dequant + scalar
accumulate through `.float8_e4m3fn` — no MFMA and no code shared with the
kernel, so a fragment-layout bug cannot cancel out.

Also a determinism gate (`_probe_grouped_determinism`) at M3 shapes: a race can
be finite and stable after launch 0, so only relaunch + byte-diff catches it.

Usage:
  br test_mxfp8_matmul_amd_preb.mojo.test
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, global_idx
from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.info import MI355X
from max.gpu.memory import CacheOperation
from std.math import align_up, ceildiv, isnan
from std.memory import bitcast
from std.random import random_ui64, seed
from std.sys import size_of
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
from linalg.arch.amd.block_scaled_mma import CDNA4F8F6F4MatrixFormat
from linalg.fp4_utils import MXFP8_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import (
    BlockScaledMatmulAMD_PreB,
    Shuffler,
    block_scaled_grouped_matmul_amd_preb,
)

comptime FP8_LANE_BYTES = 32

# Must be >= 3: the target failure is run 0 disagreeing with a
# self-consistent run 1..N-1.
comptime DET_RUNS = 8


# ===----------------------------------------------------------------------=== #
# Per-element GPU reference (dequant + scalar accumulate), no MFMA.
# ===----------------------------------------------------------------------=== #


def block_scaled_matmul_fp8_ref(
    a_ptr: ImmPointer[UInt8, ImmutAnyOrigin],
    b_ptr: ImmPointer[UInt8, ImmutAnyOrigin],
    a_scales_ptr: ImmPointer[Float8_e8m0fnu, ImmutAnyOrigin],
    b_scales_ptr: ImmPointer[Float8_e8m0fnu, ImmutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    M_arg: Int32,
    N_arg: Int32,
    K_arg: Int32,
):
    """Per-element GPU reference for MXFP8 block-scaled matmul."""
    # `Int` is not `DevicePassable`, so the extents arrive as `Int32`.
    var M = Int(M_arg)
    var N = Int(N_arg)
    var K = Int(K_arg)
    var m = global_idx.x
    var n = global_idx.y

    if m >= M or n >= N:
        return

    var k_groups = K // MXFP8_SF_VECTOR_SIZE

    var am_scales_ptr = a_scales_ptr + m * k_groups
    var bn_scales_ptr = b_scales_ptr + n * k_groups

    # One byte per element at MXFP8, so the row stride is K (not K // 2).
    var am_ptr = (a_ptr + m * K).bitcast[Float8_e4m3fn]()
    var bn_ptr = (b_ptr + n * K).bitcast[Float8_e4m3fn]()

    var accum = Float32(0)

    for ko in range(k_groups):
        var a_scale = am_scales_ptr[ko].cast[.float32]()
        var b_scale = bn_scales_ptr[ko].cast[.float32]()

        # E8M0 scales are exact powers of two, so hoisting them is bit-exact.
        var part = Float32(0)
        for ki in range(MXFP8_SF_VECTOR_SIZE):
            part += am_ptr[ki].cast[.float32]() * bn_ptr[ki].cast[.float32]()
        accum += part * a_scale * b_scale

        am_ptr += MXFP8_SF_VECTOR_SIZE
        bn_ptr += MXFP8_SF_VECTOR_SIZE

    c_ptr[m * N + n] = accum


# ===----------------------------------------------------------------------=== #
# Grid wrapper: block_idx.x = n_tile, block_idx.y = m_tile.
# ===----------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(
            BlockScaledMatmulAMD_PreB[
                BM=BM,
                BN=BN,
                BK_ELEMS=BK_ELEMS,
                WN=WN,
                b_prefetch=b_prefetch,
                scale_group=scale_group,
                b_addr_split=b_addr_split,
                matrix_format=CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3,
            ].num_threads
        )
    )
)
def _preb_fp8_grid_kernel[
    BM: Int,
    BN: Int,
    BK_ELEMS: Int,
    WN: Int,
    b_prefetch: Bool,
    scale_group: Int,
    b_addr_split: Bool,
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
        scale_group=scale_group,
        b_addr_split=b_addr_split,
        matrix_format=CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3,
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


def _e4m3_byte(f: Float32) -> UInt8:
    """Byte encoding of `f` as E4M3."""
    return bitcast[.uint8, 1](Float8_e4m3fn(f.cast[.float8_e4m3fn]()))[0]


def _rand_e4m3_byte() -> UInt8:
    """Random finite E4M3 byte in [-1, 1].

    A uniform byte fill is unusable: 0x7F/0xFF are NaN and 0x7E is 448.
    """
    return _e4m3_byte(Float32(Int(random_ui64(0, 2000))) / 1000.0 - 1.0)


def _fnv1a64[dtype: DType](buf: HostBuffer[dtype], n: Int) -> UInt64:
    """FNV-1a over the buffer's raw bytes."""
    var ptr = buf.unsafe_ptr().bitcast[UInt8]()
    var h = UInt64(0xCBF29CE484222325)
    for i in range(n * size_of[dtype]()):
        h = (h ^ UInt64(Int(ptr[i]))) * UInt64(0x100000001B3)
    return h


def _count_nans[dtype: DType](buf: HostBuffer[dtype], n: Int) -> Int:
    var count = 0
    for i in range(n):
        if isnan(buf[i]):
            count += 1
    return count


def _test_case[
    M_static: Int,
    N_static: Int,
    K_static: Int,
    BM: Int = 64,
    BN: Int = 128,
    BK_ELEMS: Int = 256,
    WN: Int = 64,
    b_prefetch: Bool = False,
    scale_group: Int = 1,
    b_addr_split: Bool = False,
](name: String, ctx: DeviceContext) raises:
    """One direct-launch MXFP8 correctness case for the preb kernel."""
    # BK_ELEMS % 256 == 0 keeps num_k_mmas even (preshuffled-scale k_pack=2).
    comptime assert BK_ELEMS % 256 == 0, "BK_ELEMS must be a multiple of 256"
    comptime assert K_static % BK_ELEMS == 0, "K must be a multiple of BK_ELEMS"
    comptime assert N_static % BN == 0, "N_static must be a multiple of BN"
    comptime assert N_static % 32 == 0, "N must be a multiple of 32 (mn_pack=2)"

    comptime K_BYTES = K_static
    comptime scale_K = K_static // MXFP8_SF_VECTOR_SIZE
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
    var a_h = ctx.enqueue_create_host_buffer[.uint8](M_static * K_BYTES)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](N_static * K_BYTES)
    var sfa_h = ctx.enqueue_create_host_buffer[.uint8](M_static * scale_K)
    var sfb_h = ctx.enqueue_create_host_buffer[.uint8](N_static * scale_K)
    var sfa_pre_h = ctx.enqueue_create_host_buffer[.uint8](padded_M * scale_K)
    var sfb_pre_h = ctx.enqueue_create_host_buffer[.uint8](N_static * scale_K)
    ctx.synchronize()

    for i in range(M_static * K_BYTES):
        a_h[i] = _rand_e4m3_byte()
    for i in range(N_static * K_BYTES):
        b_h[i] = _rand_e4m3_byte()
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
    # The scale layout is format-independent, so the FP4 default is reused.
    _ = Shuffler[1].preshuffle_scale_4d[MN=M_static, K_SCALES=scale_K](
        sfa_h_tt, sfa_pre_h
    )
    _ = Shuffler[1].preshuffle_scale_4d[MN=N_static, K_SCALES=scale_K](
        sfb_h_tt, sfb_pre_h
    )

    # ---- Device buffers + upload ----
    var a_d = ctx.enqueue_create_buffer[.uint8](M_static * K_BYTES)
    var b_d = ctx.enqueue_create_buffer[.uint8](N_static * K_BYTES)
    var b_pre_d = ctx.enqueue_create_buffer[.uint8](N_static * K_BYTES)
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

    # ---- GPU preshuffle B -> b_pre_d (matrix_format=CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3 strides) ----
    var b_raw_tt = TileTensor(b_d, row_major[1, N_static, K_BYTES]()).as_immut()
    var b_pre_dst_tt = TileTensor(
        b_pre_d,
        Shuffler[1].b_5d_grouped_layout[N=N_static, K_BYTES=K_BYTES],
    )
    Shuffler[1].preshuffle_b_5d[N=N_static, K_BYTES=K_BYTES](
        b_raw_tt, b_pre_dst_tt, ctx
    )

    # ---- Reference ----
    comptime BLOCK_DIM = 32
    ctx.enqueue_function[block_scaled_matmul_fp8_ref](
        a_d,
        b_d,
        sfa_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu](),
        sfb_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu](),
        c_ref_d,
        Int32(M_static),
        Int32(N_static),
        Int32(K_static),
        grid_dim=(ceildiv(M_static, BLOCK_DIM), ceildiv(N_static, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    # ---- Preb kernel under test ----
    var a_tt = TileTensor(
        a_d, row_major(Coord(M_static, Idx[K_BYTES]))
    ).as_immut()
    var b_pre_tt = TileTensor(
        b_pre_d, row_major[1, N_static * K_BYTES]()
    ).as_immut()
    # Preshuffled scale buffers wrapped row-major: the kernel addresses the
    # bytes through `PreshuffledScaleLoader`, so this layout is bookkeeping.
    var sfa_tt = TileTensor(
        sfa_pre_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu](),
        row_major[padded_M, scale_K](),
    ).as_immut()
    var sfb_tt = TileTensor(
        sfb_pre_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu](),
        row_major[N_static, scale_K](),
    ).as_immut()
    var c_tt = TileTensor(c_d, row_major[M_static, N_static]())

    comptime kernel = _preb_fp8_grid_kernel[
        BM,
        BN,
        BK_ELEMS,
        WN,
        b_prefetch,
        scale_group,
        b_addr_split,
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
        K_BYTES,
    ]
    ctx.enqueue_function[kernel](
        c_tt,
        a_tt,
        b_pre_tt,
        sfa_tt,
        sfb_tt,
        grid_dim=(N_static // BN, ceildiv(M_static, BM)),
        block_dim=BlockScaledMatmulAMD_PreB[
            BM=BM,
            BN=BN,
            BK_ELEMS=BK_ELEMS,
            WN=WN,
            b_prefetch=b_prefetch,
            matrix_format=CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3,
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
# Grouped (MoE) path through the PUBLIC dispatcher: band selection, expert
# dispatch, and the per-expert A-scale slot stride.
# ===----------------------------------------------------------------------=== #


def _test_grouped_case[
    num_experts: Int,
    N: Int,
    K: Int,
](name: String, num_tokens_by_expert: List[Int], ctx: DeviceContext,) raises:
    comptime K_BYTES = K  # MXFP8: one byte per element
    comptime scale_K = K // MXFP8_SF_VECTOR_SIZE

    var num_active = len(num_tokens_by_expert)
    var total_tokens = 0
    var max_tokens = 0
    for i in range(num_active):
        total_tokens += num_tokens_by_expert[i]
        max_tokens = max(max_tokens, num_tokens_by_expert[i])
    var max_padded_M = align_up(max_tokens, 32)

    print(
        "  ",
        name,
        " experts=",
        num_active,
        " total_tokens=",
        total_tokens,
        " N=",
        N,
        " K=",
        K,
    )

    var a_h = ctx.enqueue_create_host_buffer[.uint8](total_tokens * K_BYTES)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](num_experts * N * K_BYTES)
    var sfa_h = ctx.enqueue_create_host_buffer[.uint8](total_tokens * scale_K)
    var sfb_h = ctx.enqueue_create_host_buffer[.uint8](
        num_experts * N * scale_K
    )
    var sfb_pre_h = ctx.enqueue_create_host_buffer[.uint8](
        num_experts * N * scale_K
    )
    var a_off_h = ctx.enqueue_create_host_buffer[.uint32](num_active + 1)
    var eid_h = ctx.enqueue_create_host_buffer[.int32](num_active)
    ctx.synchronize()

    for i in range(total_tokens * K_BYTES):
        a_h[i] = _rand_e4m3_byte()
    for i in range(num_experts * N * K_BYTES):
        b_h[i] = _rand_e4m3_byte()
    for i in range(total_tokens * scale_K):
        sfa_h[i] = UInt8(random_ui64(125, 129))
    for i in range(num_experts * N * scale_K):
        sfb_h[i] = UInt8(random_ui64(125, 129))

    a_off_h[0] = UInt32(0)
    for i in range(num_active):
        a_off_h[i + 1] = a_off_h[i] + UInt32(num_tokens_by_expert[i])
        eid_h[i] = Int32(i)

    var a_d = ctx.enqueue_create_buffer[.uint8](total_tokens * K_BYTES)
    var b_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * K_BYTES)
    var b_pre_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * K_BYTES)
    var sfa_d = ctx.enqueue_create_buffer[.uint8](total_tokens * scale_K)
    var sfb_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * scale_K)
    var sfa_pre_d = ctx.enqueue_create_buffer[.uint8](
        num_experts * max_padded_M * scale_K
    )
    var sfb_pre_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * scale_K)
    var a_off_d = ctx.enqueue_create_buffer[.uint32](num_active + 1)
    var eid_d = ctx.enqueue_create_buffer[.int32](num_active)
    var c_d = ctx.enqueue_create_buffer[.float32](total_tokens * N)
    var c_ref_d = ctx.enqueue_create_buffer[.float32](total_tokens * N)
    c_d.enqueue_fill(Float32(0.0))
    c_ref_d.enqueue_fill(Float32(0.0))

    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(sfa_d, sfa_h)
    ctx.enqueue_copy(sfb_d, sfb_h)
    ctx.enqueue_copy(a_off_d, a_off_h)
    ctx.enqueue_copy(eid_d, eid_h)

    # B weights + B scales: preshuffled once, as at session.load in production.
    var b_raw_tt = TileTensor(
        b_d, row_major[num_experts, N, K_BYTES]()
    ).as_immut()
    var b_pre_dst_tt = TileTensor(
        b_pre_d,
        Shuffler[num_experts].b_5d_grouped_layout[N=N, K_BYTES=K_BYTES],
    )
    Shuffler[num_experts].preshuffle_b_5d[N=N, K_BYTES=K_BYTES](
        b_raw_tt, b_pre_dst_tt, ctx
    )
    var sfb_raw_tt = TileTensor(
        sfb_h, row_major(Coord(Idx[num_experts], Idx[N], Idx[scale_K]))
    )
    _ = Shuffler[num_experts].preshuffle_scale_4d[MN=N, K_SCALES=scale_K](
        sfb_raw_tt, sfb_pre_h
    )
    ctx.enqueue_copy(sfb_pre_d, sfb_pre_h)

    # A scales: per-expert fixed-stride slots, same launcher as the FP4 path.
    var sfa_raw_tt = TileTensor(
        sfa_d, row_major(Coord(total_tokens, Idx[scale_K]))
    ).as_immut()
    var sfa_pre_tt = TileTensor(
        sfa_pre_d, row_major(Coord(num_experts * max_padded_M, Idx[scale_K]))
    )
    var a_off_pre_tt = TileTensor(
        a_off_d, row_major(Coord(num_active + 1))
    ).as_immut()
    Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=scale_K](
        sfa_raw_tt,
        sfa_pre_tt,
        a_off_pre_tt,
        num_active,
        max_tokens,
        ctx.default_device_info.sm_count * 2,
        ctx,
    )

    # Reference: one ungrouped per-element matmul per active expert slot.
    comptime BLOCK_DIM = 32
    for slot in range(num_active):
        var tok_start = Int(a_off_h[slot])
        var m_slot = Int(a_off_h[slot + 1]) - tok_start
        if m_slot == 0:
            continue
        var eid = Int(eid_h[slot])
        ctx.enqueue_function[block_scaled_matmul_fp8_ref](
            (a_d.unsafe_ptr() + tok_start * K_BYTES).as_imm(),
            (b_d.unsafe_ptr() + eid * N * K_BYTES).as_imm(),
            (sfa_d.unsafe_ptr() + tok_start * scale_K)
            .bitcast[Float8_e8m0fnu]()
            .as_imm(),
            (sfb_d.unsafe_ptr() + eid * N * scale_K)
            .bitcast[Float8_e8m0fnu]()
            .as_imm(),
            c_ref_d.unsafe_ptr() + tok_start * N,
            Int32(m_slot),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(m_slot, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    # Public dispatcher at matrix_format=CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3.
    var c_tt = TileTensor(c_d, row_major(Coord(total_tokens, Idx[N])))
    var a_tt = TileTensor(
        a_d, row_major(Coord(total_tokens, Idx[K_BYTES]))
    ).as_immut()
    var b_pre_flat = TileTensor(
        b_pre_d, row_major[num_experts, N * K_BYTES]()
    ).as_immut()
    var sfa_tt = TileTensor(
        sfa_pre_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu](),
        row_major(Coord(num_experts * max_padded_M, Idx[scale_K])),
    ).as_immut()
    var sfb_tt = TileTensor(
        sfb_pre_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu](),
        row_major[num_experts * N, scale_K](),
    ).as_immut()
    var a_off_tt = TileTensor(
        a_off_d, row_major(Coord(num_active + 1))
    ).as_immut()
    var eid_tt = TileTensor(eid_d, row_major(Coord(num_active))).as_immut()

    block_scaled_grouped_matmul_amd_preb[lane_bytes=FP8_LANE_BYTES](
        c_tt,
        a_tt,
        b_pre_flat,
        sfa_tt,
        sfb_tt,
        a_off_tt,
        eid_tt,
        max_tokens,
        num_active,
        ctx,
        total_tokens,
    )
    ctx.synchronize()

    var c_h = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    var c_ref_h = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    ctx.enqueue_copy(c_h, c_d)
    ctx.enqueue_copy(c_ref_h, c_ref_d)
    ctx.synchronize()

    assert_almost_equal(
        c_h.unsafe_ptr(),
        c_ref_h.unsafe_ptr(),
        total_tokens * N,
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
    _ = a_off_d^
    _ = eid_d^
    _ = c_d^
    _ = c_ref_d^


def _probe_grouped_determinism[
    num_experts: Int,
    N: Int,
    K: Int,
](
    name: String,
    num_tokens_by_expert: List[Int],
    estimated_total_m: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Relaunch `DET_RUNS` times; require byte-identity, no NaNs, and a ref match.

    Stale kernel-filled reads are self-consistent; pair with `--runs_per_test`.
    """
    comptime K_BYTES = K
    comptime scale_K = K // MXFP8_SF_VECTOR_SIZE

    var num_active = len(num_tokens_by_expert)
    var total_tokens = 0
    var max_tokens = 0
    for i in range(num_active):
        total_tokens += num_tokens_by_expert[i]
        max_tokens = max(max_tokens, num_tokens_by_expert[i])
    var max_padded_M = align_up(max_tokens, 32)
    var n_elem = total_tokens * N

    print(
        "  ",
        name,
        " etm=",
        estimated_total_m,
        " tokens=",
        total_tokens,
        " experts=",
        num_active,
        " N=",
        N,
        " K=",
        K,
    )

    var a_h = ctx.enqueue_create_host_buffer[.uint8](total_tokens * K_BYTES)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](num_experts * N * K_BYTES)
    var sfa_h = ctx.enqueue_create_host_buffer[.uint8](total_tokens * scale_K)
    var sfb_h = ctx.enqueue_create_host_buffer[.uint8](
        num_experts * N * scale_K
    )
    var sfb_pre_h = ctx.enqueue_create_host_buffer[.uint8](
        num_experts * N * scale_K
    )
    var a_off_h = ctx.enqueue_create_host_buffer[.uint32](num_active + 1)
    var eid_h = ctx.enqueue_create_host_buffer[.int32](num_active)
    ctx.synchronize()

    for i in range(total_tokens * K_BYTES):
        a_h[i] = _rand_e4m3_byte()
    for i in range(num_experts * N * K_BYTES):
        b_h[i] = _rand_e4m3_byte()
    for i in range(total_tokens * scale_K):
        sfa_h[i] = UInt8(random_ui64(125, 129))
    for i in range(num_experts * N * scale_K):
        sfb_h[i] = UInt8(random_ui64(125, 129))

    a_off_h[0] = UInt32(0)
    for i in range(num_active):
        a_off_h[i + 1] = a_off_h[i] + UInt32(num_tokens_by_expert[i])
        eid_h[i] = Int32(i)

    var a_d = ctx.enqueue_create_buffer[.uint8](total_tokens * K_BYTES)
    var b_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * K_BYTES)
    var b_pre_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * K_BYTES)
    var sfa_d = ctx.enqueue_create_buffer[.uint8](total_tokens * scale_K)
    var sfb_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * scale_K)
    var sfa_pre_d = ctx.enqueue_create_buffer[.uint8](
        num_experts * max_padded_M * scale_K
    )
    var sfb_pre_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * scale_K)
    var a_off_d = ctx.enqueue_create_buffer[.uint32](num_active + 1)
    var eid_d = ctx.enqueue_create_buffer[.int32](num_active)
    var c_d = ctx.enqueue_create_buffer[.float32](n_elem)
    var c_ref_d = ctx.enqueue_create_buffer[.float32](n_elem)
    c_ref_d.enqueue_fill(Float32(0.0))

    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(sfa_d, sfa_h)
    ctx.enqueue_copy(sfb_d, sfb_h)
    ctx.enqueue_copy(a_off_d, a_off_h)
    ctx.enqueue_copy(eid_d, eid_h)

    var b_raw_tt = TileTensor(
        b_d, row_major[num_experts, N, K_BYTES]()
    ).as_immut()
    var b_pre_dst_tt = TileTensor(
        b_pre_d,
        Shuffler[num_experts].b_5d_grouped_layout[N=N, K_BYTES=K_BYTES],
    )
    Shuffler[num_experts].preshuffle_b_5d[N=N, K_BYTES=K_BYTES](
        b_raw_tt, b_pre_dst_tt, ctx
    )
    var sfb_raw_tt = TileTensor(
        sfb_h, row_major(Coord(Idx[num_experts], Idx[N], Idx[scale_K]))
    )
    _ = Shuffler[num_experts].preshuffle_scale_4d[MN=N, K_SCALES=scale_K](
        sfb_raw_tt, sfb_pre_h
    )
    ctx.enqueue_copy(sfb_pre_d, sfb_pre_h)

    var sfa_raw_tt = TileTensor(
        sfa_d, row_major(Coord(total_tokens, Idx[scale_K]))
    ).as_immut()
    var sfa_pre_tt = TileTensor(
        sfa_pre_d, row_major(Coord(num_experts * max_padded_M, Idx[scale_K]))
    )
    var a_off_pre_tt = TileTensor(
        a_off_d, row_major(Coord(num_active + 1))
    ).as_immut()
    Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=scale_K](
        sfa_raw_tt,
        sfa_pre_tt,
        a_off_pre_tt,
        num_active,
        max_tokens,
        ctx.default_device_info.sm_count * 2,
        ctx,
    )

    # Reference: one ungrouped per-element matmul per active expert slot.
    comptime BLOCK_DIM = 32
    for slot in range(num_active):
        var tok_start = Int(a_off_h[slot])
        var m_slot = Int(a_off_h[slot + 1]) - tok_start
        if m_slot == 0:
            continue
        var eid = Int(eid_h[slot])
        ctx.enqueue_function[block_scaled_matmul_fp8_ref](
            (a_d.unsafe_ptr() + tok_start * K_BYTES).as_imm(),
            (b_d.unsafe_ptr() + eid * N * K_BYTES).as_imm(),
            (sfa_d.unsafe_ptr() + tok_start * scale_K)
            .bitcast[Float8_e8m0fnu]()
            .as_imm(),
            (sfb_d.unsafe_ptr() + eid * N * scale_K)
            .bitcast[Float8_e8m0fnu]()
            .as_imm(),
            c_ref_d.unsafe_ptr() + tok_start * N,
            Int32(m_slot),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(m_slot, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    var c_tt = TileTensor(c_d, row_major(Coord(total_tokens, Idx[N])))
    var a_tt = TileTensor(
        a_d, row_major(Coord(total_tokens, Idx[K_BYTES]))
    ).as_immut()
    var b_pre_flat = TileTensor(
        b_pre_d, row_major[num_experts, N * K_BYTES]()
    ).as_immut()
    var sfa_tt = TileTensor(
        sfa_pre_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu](),
        row_major(Coord(num_experts * max_padded_M, Idx[scale_K])),
    ).as_immut()
    var sfb_tt = TileTensor(
        sfb_pre_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu](),
        row_major[num_experts * N, scale_K](),
    ).as_immut()
    var a_off_tt = TileTensor(
        a_off_d, row_major(Coord(num_active + 1))
    ).as_immut()
    var eid_tt = TileTensor(eid_d, row_major(Coord(num_active))).as_immut()

    var c_h = ctx.enqueue_create_host_buffer[.float32](n_elem)
    var c_ref_h = ctx.enqueue_create_host_buffer[.float32](n_elem)
    ctx.synchronize()
    ctx.enqueue_copy(c_ref_h, c_ref_d)
    ctx.synchronize()

    var hashes = List[UInt64]()
    var max_err = Float32(0)
    var nans = 0

    for _run in range(DET_RUNS):
        # Zero every launch so a skipped store cannot inherit the previous
        # output.
        c_d.enqueue_fill(Float32(0.0))
        block_scaled_grouped_matmul_amd_preb[lane_bytes=FP8_LANE_BYTES](
            c_tt,
            a_tt,
            b_pre_flat,
            sfa_tt,
            sfb_tt,
            a_off_tt,
            eid_tt,
            max_tokens,
            num_active,
            ctx,
            estimated_total_m,
        )
        ctx.synchronize()
        ctx.enqueue_copy(c_h, c_d)
        ctx.synchronize()
        hashes.append(_fnv1a64(c_h, n_elem))
        nans += _count_nans(c_h, n_elem)
        for i in range(n_elem):
            var d = abs(c_h[i] - c_ref_h[i])
            if d > max_err:
                max_err = d

    var all_equal = True
    for r in range(len(hashes)):
        if hashes[r] != hashes[0]:
            all_equal = False
    if not all_equal:
        for r in range(len(hashes)):
            print("      run", r, "hash", hashes[r])

    # Loose absolute bound: catch a wrong-but-stable result, not fp8 rounding.
    var ref_ok = max_err <= 0.5
    print(
        "     byte-identical:",
        all_equal,
        "| nans:",
        nans,
        "| max_err_vs_ref:",
        max_err,
    )
    var ok = all_equal and nans == 0 and ref_ok
    print("     PASS" if ok else "     FAIL")

    _ = a_d^
    _ = b_d^
    _ = b_pre_d^
    _ = sfa_d^
    _ = sfb_d^
    _ = sfa_pre_d^
    _ = sfb_pre_d^
    _ = a_off_d^
    _ = eid_d^
    _ = c_d^
    _ = c_ref_d^

    return ok


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "test_mxfp8_matmul_amd_preb requires MI355X"

    print(
        "===> BlockScaledMatmulAMD_PreB at"
        " matrix_format=CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3 (MXFP8) —"
        " correctness"
    )

    _test_case[M_static=64, N_static=128, K_static=512]("base", ctx)
    _test_case[M_static=128, N_static=256, K_static=512]("multi-tile", ctx)
    _test_case[M_static=64, N_static=128, K_static=1024]("deep-k", ctx)
    _test_case[M_static=70, N_static=128, K_static=512]("ragged-m", ctx)

    _test_case[M_static=64, N_static=128, K_static=512, b_prefetch=True](
        "prefetch", ctx
    )

    # Reach the two band-table scheduling knobs directly, so their coverage
    # does not depend on a dispatcher band continuing to select them. K=1024
    # gives 4 outer-K tiles, the minimum `scale_group=4` can fold; a wrong
    # lane transpose or LDS-strip offset is silently wrong scales, not a crash,
    # and the scales here are randomised so the reference can see it.
    _test_case[
        M_static=64,
        N_static=128,
        K_static=1024,
        BK_ELEMS=256,
        b_prefetch=True,
        scale_group=4,
    ]("scale-group-4", ctx)
    _test_case[M_static=64, N_static=128, K_static=1024, b_addr_split=True](
        "b-addr-split", ctx
    )
    _test_case[
        M_static=64,
        N_static=128,
        K_static=1024,
        BK_ELEMS=256,
        b_prefetch=True,
        scale_group=4,
        b_addr_split=True,
    ]("scale-group-4-b-addr-split", ctx)

    # Decode-shaped: BM=16 / WN=16 exercises the odd-num_mmas scale-cell
    # straddle (the `_a_scale_shift` / `_b_scale_shift` rotation).
    _test_case[M_static=16, N_static=64, K_static=512, BM=16, BN=64, WN=16](
        "decode-bm16", ctx
    )
    _test_case[
        M_static=16,
        N_static=64,
        K_static=512,
        BM=16,
        BN=64,
        WN=16,
        b_prefetch=True,
    ]("decode-bm16-prefetch", ctx)

    print(
        "===> grouped dispatcher at"
        " matrix_format=CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3"
    )
    _test_grouped_case[num_experts=4, N=128, K=512](
        "grouped-small", [16, 32, 7, 48], ctx
    )
    # Real M3 down shape; its (N, packed_K) key also collides with MXFP4 gate_up.
    _test_grouped_case[num_experts=2, N=6144, K=3072](
        "grouped-m3-down", [24, 8], ctx
    )
    # Top band (etm > 2048) against a reference, at a real token count. `1290`
    # leaves a 10-row tail in the last BM=128 m-tile and pads the A scales to
    # 1312, which is the ragged shape production routing produces and the one
    # the wide scale fetch and the soffset-split B addressing clamp differently
    # from the per-tile loads they replace.
    _test_grouped_case[num_experts=2, N=6144, K=6144](
        "grouped-m3-gate_up etm=2580 ragged", [1290, 1290], ctx
    )

    print(
        "===> grouped dispatcher run-to-run determinism (MiniMax-M3 MXFP8"
        " shapes)"
    )
    # M3 projections, just above the etm=256 edge and one higher band
    # (etm = M/2 at topk=4, ep=8).
    var det_ok = True
    det_ok &= _probe_grouped_determinism[1, 6144, 6144](
        "m3-gate_up etm=264", [264], 264, ctx
    )
    det_ok &= _probe_grouped_determinism[2, 6144, 6144](
        "m3-gate_up etm=512", [256, 256], 512, ctx
    )
    det_ok &= _probe_grouped_determinism[1, 6144, 3072](
        "m3-down etm=264", [264], 264, ctx
    )
    det_ok &= _probe_grouped_determinism[2, 6144, 3072](
        "m3-down etm=1024", [512, 512], 1024, ctx
    )

    # `estimated_total_m` only steers the dispatcher's band pick (it is not
    # a kernel bound), so it can be set independently of the real token
    # count above to reach the `etm > 2048` band -- the one real serving
    # traffic hits -- without paying for a huge per-element reference matmul.
    # Both shapes' `LB==32` dispatch has a single band above 2048 (no further
    # split), so one value per shape covers the branch; the second pair below
    # re-enters the same tile with a ragged token count, which is a data case
    # the aligned lists cannot reach.
    det_ok &= _probe_grouped_determinism[2, 6144, 6144](
        "m3-gate_up etm=2049", [256, 256], 2049, ctx
    )
    det_ok &= _probe_grouped_determinism[2, 6144, 3072](
        "m3-down etm=2049", [512, 512], 2049, ctx
    )
    # Both top bands run BM=128; 266 and 246 leave 10- and 118-row tails.
    det_ok &= _probe_grouped_determinism[2, 6144, 6144](
        "m3-gate_up etm=2049 ragged", [266, 246], 2049, ctx
    )
    det_ok &= _probe_grouped_determinism[2, 6144, 3072](
        "m3-down etm=2049 ragged", [266, 246], 2049, ctx
    )
    if not det_ok:
        raise Error(
            "grouped MXFP8 dispatcher is run-to-run nondeterministic or"
            " disagrees with the reference; see the per-run hashes above"
        )
