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
"""Exhaustive kernel-level tests for the preshuffled-B grouped MXFP6 kernels.

The MXFP6 twin of `test_block_scaled_grouped_matmul_amd_kernels.mojo`, running that
file's ENTIRE case matrix -- every shape, tile config, routing pattern and
launch knob, in the same order -- once per FP6 encoding. A production prefill
trace dispatches both kernels, so both need FP6:

  - `PreShuffledBGroupedGEMM.launch[persistent=True]`   -- persistent 1D grid
                                                          + XCD swizzle
  - `PreShuffledBGroupedGEMM.launch[persistent=False]`  -- direct 3D grid

Both encodings run the full matrix rather than E3M2 taking a subset. E2M3 and
E3M2 share a byte layout and a `bits_per_element`, so in principle only the
hardware value table differs and layout bugs cannot distinguish them -- running
both anyway costs compile time and buys certainty that the shared assumption
actually holds on silicon.

The reference is not another MFMA kernel (as in the MXFP4 file) but a
per-element software decode that also emits `sum |a*b|`. That costs a slower
reference and buys a tolerance tied to the accumulated magnitude instead of to
a result that may have cancelled to near zero, where a relative bound means
nothing.

Currently MI355X-only.

Usage:
  br test_mxfp6_grouped_matmul_amd_kernels.mojo.test
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from max.gpu.memory import CacheOperation
from std.math import align_up, ceildiv
from std.memory import bitcast
from std.random import random_ui64, seed
from std.testing import assert_true
from std.utils import StaticTuple

from layout import Coord, Idx, TileTensor, row_major
from linalg.arch.amd.block_scaled_mma import CDNA4F8F6F4MatrixFormat
from linalg.fp6_utils import (
    FP6Format,
    MXFP6_SF_VECTOR_SIZE,
    decode_fp6_to_f32,
    unpack_fp6_x32,
)
from linalg.matmul.gpu.amd import (
    PreShuffledBGroupedGEMM,
    Shuffler,
    block_scaled_grouped_matmul_amd_preb,
)

# A lane feeds the MFMA 24 bytes of FP6, split into a 16-byte and an 8-byte
# plane by the preshuffle.
comptime FP6_LANE_BYTES = 24


def _mfma_format[fmt: FP6Format]() -> CDNA4F8F6F4MatrixFormat:
    comptime if fmt == FP6Format.E2M3:
        return CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3
    return CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2


# ===----------------------------------------------------------------------=== #
# Input helpers
# ===----------------------------------------------------------------------=== #


def _fill_random_bytes(buf: HostBuffer[.uint8], n: Int):
    """Every 6-bit code is a finite number in both FP6 encodings, so random
    bytes need no NaN/Inf filtering."""
    for i in range(n):
        buf[i] = UInt8(random_ui64(0, 255))


def _fill_random_e8m0(buf: HostBuffer[.float8_e8m0fnu], n: Int):
    """Scales clamped to E8M0 byte range [125..129] = magnitudes [0.25..4],
    keeping f32 accumulators in range while still exercising scale-dequant."""
    for i in range(n):
        buf[i] = bitcast[.float8_e8m0fnu](UInt8(random_ui64(125, 129)))


def _build_routing(
    a_offsets_host: HostBuffer[.uint32],
    expert_ids_host: HostBuffer[.int32],
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
):
    a_offsets_host[0] = UInt32(0)
    for i in range(len(num_tokens_by_expert)):
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host[i] = Int32(expert_ids_list[i])


# ===----------------------------------------------------------------------=== #
# Reference — one thread per (m, n), decoding in software.
#
# Independent of the path under test: the kernel never software-decodes, it
# hands raw bits to the MFMA and the hardware decodes them. `test_fp6_utils`
# pins this decoder against hand-written tables
# pins the hardware against the same convention.
#
# A near-copy of the reference in `test_mxfp6_matmul_amd.mojo`; kept separate
# because the two files have no shared-source target.
# ===----------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256))
)
def _mxfp6_matmul_ref[
    fmt: FP6Format
](
    a_ptr: ImmPointer[UInt8, ImmutAnyOrigin],
    b_ptr: ImmPointer[UInt8, ImmutAnyOrigin],
    a_sf_ptr: ImmPointer[Float8_e8m0fnu, ImmutAnyOrigin],
    b_sf_ptr: ImmPointer[Float8_e8m0fnu, ImmutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    mag_ptr: MutPointer[Float32, MutAnyOrigin],
    M_dev: Int32,
    N_dev: Int32,
    K_dev: Int32,
    c_row_stride: Int32,
):
    """Computes one expert's tile; `c_row_stride` lets the caller write into a
    row range of the grouped output without copying."""
    var M = Int(M_dev)
    var N = Int(N_dev)
    var K = Int(K_dev)

    var m = Int(global_idx.x)
    var n = Int(global_idx.y)
    if m >= M or n >= N:
        return

    var k_groups = K // 32
    var k_bytes = (K * 6) // 8

    var accum = Float32(0)
    var magnitude = Float32(0)

    for ko in range(k_groups):
        var a_scale = a_sf_ptr[unsafe_offset=m * k_groups + ko].cast[.float32]()
        var b_scale = b_sf_ptr[unsafe_offset=n * k_groups + ko].cast[.float32]()

        # 32 elements is one MX block = 24 packed bytes, always 8-byte aligned
        # because k_bytes is a multiple of 24 whenever K is a multiple of 32.
        var a_base = m * k_bytes + ko * 24
        var b_base = n * k_bytes + ko * 24

        var fa = SIMD[.uint8, 32](0)
        var fb = SIMD[.uint8, 32](0)
        comptime for chunk in range(3):
            fa = fa.insert[offset=chunk * 8](
                a_ptr.load[width=8](a_base + chunk * 8)
            )
            fb = fb.insert[offset=chunk * 8](
                b_ptr.load[width=8](b_base + chunk * 8)
            )

        var av = decode_fp6_to_f32[fmt](unpack_fp6_x32(fa)) * a_scale
        var bv = decode_fp6_to_f32[fmt](unpack_fp6_x32(fb)) * b_scale
        var prod = av * bv
        accum += prod.reduce_add()
        magnitude += abs(prod).reduce_add()

    c_ptr[unsafe_offset=m * Int(c_row_stride) + n] = accum
    mag_ptr[unsafe_offset=m * Int(c_row_stride) + n] = magnitude


def _per_expert_reference[
    fmt: FP6Format, num_experts: Int, N: Int, K: Int
](
    ctx: DeviceContext,
    num_active: Int,
    a_offsets_host: HostBuffer[.uint32],
    expert_ids_host: HostBuffer[.int32],
    a_dev: DeviceBuffer[.uint8],
    b_dev: DeviceBuffer[.uint8],
    a_scales_dev: DeviceBuffer[.float8_e8m0fnu],
    b_scales_dev: DeviceBuffer[.float8_e8m0fnu],
    mut c_ref_dev: DeviceBuffer[.float32],
    mut mag_dev: DeviceBuffer[.float32],
) raises:
    comptime K_BYTES = (K * 6) // 8
    comptime scale_K = K // MXFP6_SF_VECTOR_SIZE
    comptime REF_BLOCK = 16

    for i in range(num_active):
        var token_start = Int(a_offsets_host[i])
        var num_tokens = Int(a_offsets_host[i + 1]) - token_start
        var expert_id = Int(expert_ids_host[i])

        if num_tokens <= 0 or expert_id < 0:
            continue

        ctx.enqueue_function[_mxfp6_matmul_ref[fmt]](
            a_dev.unsafe_ptr() + token_start * K_BYTES,
            b_dev.unsafe_ptr() + expert_id * N * K_BYTES,
            a_scales_dev.unsafe_ptr() + token_start * scale_K,
            b_scales_dev.unsafe_ptr() + expert_id * N * scale_K,
            c_ref_dev.unsafe_ptr() + token_start * N,
            mag_dev.unsafe_ptr() + token_start * N,
            Int32(num_tokens),
            Int32(N),
            Int32(K),
            Int32(N),
            grid_dim=(ceildiv(num_tokens, REF_BLOCK), ceildiv(N, REF_BLOCK)),
            block_dim=(REF_BLOCK, REF_BLOCK),
        )


# ===----------------------------------------------------------------------=== #
# Shared test body — only `persistent` and `cu_count` differ between
# `test_persistent` and `test_direct`.
# ===----------------------------------------------------------------------=== #


def _run_preb[
    fmt: FP6Format,
    num_experts: Int,
    N: Int,
    K: Int,
    BK_ELEMS: Int,
    persistent: Bool,
    cu_count: Int,  # struct param — `total_wg = cu_count * wg_per_cu`
    BM: Int = 64,
    BN: Int = 128,
    WN: Int = 64,
    b_cache_policy: CacheOperation = CacheOperation.ALWAYS,
    cluster_drain_sched: Bool = False,
    mfma_cluster: Int = 4,
    pipeline_depth: Int = 2,
    waves_per_eu: Int = 0,
    wg_per_cu: Int = 2,  # struct param — sizes the persistent grid
    static_grid_z: Bool = False,  # comptime grid.z = n_local_experts (direct)
    use_launcher: Bool = False,  # go through the tuned band dispatcher
](
    name: String,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
    grid_m_cap: Int = -1,  # direct grid.y cap (-1 => full stride)
    ascale_stride_toks: Int = -1,  # A-scale slot stride source (-1 => max)
    poison_ascale_pad: Bool = False,  # fill slot pad with E8M0 NaN
) raises -> Bool:
    comptime assert K % 128 == 0, "K must be a multiple of 128"
    comptime K_BYTES = (K * 6) // 8
    comptime scale_K = K // MXFP6_SF_VECTOR_SIZE

    var total_tokens = 0
    var max_tokens = 0
    var num_active = len(num_tokens_by_expert)
    for ne in num_tokens_by_expert:
        total_tokens += ne
        max_tokens = max(max_tokens, ne)

    comptime label = "[persistent]" if persistent else "[direct]    "
    comptime fmt_name = "E2M3" if fmt == FP6Format.E2M3 else "E3M2"
    print(
        "  ",
        label,
        fmt_name,
        name,
        " E=",
        num_experts,
        " N=",
        N,
        " K=",
        K,
        " active=",
        num_active,
        " total_tokens=",
        total_tokens,
        " BM=",
        BM,
        " BN=",
        BN,
        " BK_ELEMS=",
        BK_ELEMS,
        " WN=",
        WN,
    )

    # Host buffers + random init.
    var a_h = ctx.enqueue_create_host_buffer[.uint8](total_tokens * K_BYTES)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](num_experts * N * K_BYTES)
    var a_sc_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        total_tokens * scale_K
    )
    var b_sc_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    var a_off_h = ctx.enqueue_create_host_buffer[.uint32](num_active + 1)
    var eid_h = ctx.enqueue_create_host_buffer[.int32](num_active)
    ctx.synchronize()

    _fill_random_bytes(a_h, total_tokens * K_BYTES)
    _fill_random_bytes(b_h, num_experts * N * K_BYTES)
    _fill_random_e8m0(a_sc_h, total_tokens * scale_K)
    _fill_random_e8m0(b_sc_h, num_experts * N * scale_K)
    _build_routing(a_off_h, eid_h, num_tokens_by_expert, expert_ids_list)

    # Per-expert preshuffled-A-scale slot stride: align_up(stride, 32).
    var ascale_toks = (
        ascale_stride_toks if ascale_stride_toks > 0 else max_tokens
    )
    var max_padded_M = align_up(ascale_toks, 32)

    # Device buffers + upload.
    var a_d = ctx.enqueue_create_buffer[.uint8](total_tokens * K_BYTES)
    var b_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * K_BYTES)
    var b_pre_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * K_BYTES)
    var a_sc_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        total_tokens * scale_K
    )
    var b_sc_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    var a_sc_pre_d = ctx.enqueue_create_buffer[.uint8](
        num_experts * max_padded_M * scale_K
    )
    var b_sc_pre_d = ctx.enqueue_create_buffer[.uint8](
        num_experts * N * scale_K
    )
    var a_off_d = ctx.enqueue_create_buffer[.uint32](num_active + 1)
    var eid_d = ctx.enqueue_create_buffer[.int32](num_active)
    var c_d = ctx.enqueue_create_buffer[.float32](total_tokens * N)
    var c_ref_d = ctx.enqueue_create_buffer[.float32](total_tokens * N)
    var mag_d = ctx.enqueue_create_buffer[.float32](total_tokens * N)

    # Inactive slots (M=0 or expert_id=-1) leave their output range unwritten
    # by both the kernel and the reference, so both sides need a known value.
    c_d.enqueue_fill(Float32(0.0))
    c_ref_d.enqueue_fill(Float32(0.0))
    mag_d.enqueue_fill(Float32(0.0))

    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(a_sc_d, a_sc_h)
    ctx.enqueue_copy(b_sc_d, b_sc_h)
    ctx.enqueue_copy(a_off_d, a_off_h)
    ctx.enqueue_copy(eid_d, eid_h)

    # GPU preshuffle b_d -> b_pre_d. The 24-byte fragment goes through the
    # plane-split kernel; the tuned 16-byte atom path cannot express it.
    var b_raw_tt = TileTensor(
        b_d, row_major[num_experts, N, K_BYTES]()
    ).as_immut()
    var b_pre_dst_tt = TileTensor(b_pre_d, row_major[num_experts, N, K_BYTES]())
    Shuffler[num_experts].preshuffle_b_planes[
        N=N, K_BYTES=K_BYTES, lane_bytes=FP6_LANE_BYTES
    ](b_raw_tt, b_pre_dst_tt, ctx)

    # GPU preshuffle of A-scales into per-expert fixed-stride slots. The scale
    # path is format-independent: one E8M0 byte per 32 elements per lane, for
    # every f8f6f4 format.
    var a_sc_raw_u8_tt = TileTensor(
        a_sc_d.unsafe_ptr().bitcast[Scalar[.uint8]](),
        row_major(Coord(total_tokens, Idx[scale_K])),
    ).as_immut()
    # A fresh test buffer reads as zeros, which is a *valid* E8M0 exponent --
    # so the pad slots the preshuffle deliberately skips are benign here in a
    # way they are not in production, where the allocator hands back pooled
    # memory holding the previous tensor's bytes and 0xFF is E8M0's NaN. Poison
    # them so the matmul's per-expert V# bound (the sole thing keeping those
    # slots out of the result) is actually under test.
    if poison_ascale_pad:
        a_sc_pre_d.enqueue_fill(UInt8(0xFF))

    var a_sc_pre_tt = TileTensor(
        a_sc_pre_d,
        row_major(Coord(num_experts * max_padded_M, Idx[scale_K])),
    )
    var a_off_tt_for_pre = TileTensor(
        a_off_d, row_major(Coord(num_active + 1))
    ).as_immut()
    Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=scale_K](
        a_sc_raw_u8_tt,
        a_sc_pre_tt,
        a_off_tt_for_pre,
        num_active,
        ascale_toks,
        ctx.default_device_info.sm_count * 2,
        ctx,
    )

    # CPU preshuffle of B-scales — static weights, done once at session.load in
    # production; the existing helper takes comptime MN, which for B is N.
    var b_sc_pre_h = ctx.enqueue_create_host_buffer[.uint8](
        num_experts * N * scale_K
    )
    ctx.synchronize()
    var b_sc_raw_u8_tt = TileTensor(
        b_sc_h.unsafe_ptr().bitcast[UInt8](),
        row_major(Coord(Idx[num_experts], Idx[N], Idx[scale_K])),
    )
    Shuffler[num_experts].preshuffle_scale_4d[MN=N, K_SCALES=scale_K](
        b_sc_raw_u8_tt, b_sc_pre_h
    )
    ctx.enqueue_copy(b_sc_pre_d, b_sc_pre_h)

    # Reference: per-expert software decode against raw B and raw scales.
    _per_expert_reference[fmt, num_experts, N, K](
        ctx,
        num_active,
        a_off_h,
        eid_h,
        a_d,
        b_d,
        a_sc_d,
        b_sc_d,
        c_ref_d,
        mag_d,
    )

    # Run the preb kernel under test. Scales are the preshuffled buffers;
    # bitcast uint8 ptr -> float8_e8m0fnu to match the dispatcher signature
    # (the kernel bitcasts back to uint8 for V# construction).
    var a_tt = TileTensor(
        a_d, row_major(Coord(total_tokens, Idx[K_BYTES]))
    ).as_immut()
    var b_pre_tt = TileTensor(
        b_pre_d, row_major[num_experts, N * K_BYTES]()
    ).as_immut()
    var a_sc_tt = TileTensor(
        a_sc_pre_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        row_major(Coord(num_experts * max_padded_M, Idx[scale_K])),
    ).as_immut()
    var b_sc_tt = TileTensor(
        b_sc_pre_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        row_major[num_experts, N, scale_K](),
    ).as_immut()
    var a_off_tt = TileTensor(a_off_d, row_major(Coord(num_active + 1)))
    var eid_tt = TileTensor(eid_d, row_major(Coord(num_active)))
    var c_tt = TileTensor(c_d, row_major(Coord(total_tokens, Idx[N])))

    # The launcher picks the tile config from (lane_bytes, N, K, etm) and
    # infers the format from the a / a_scales shapes, so it exercises the
    # dispatch surface the graph op actually calls -- not just the GEMM.
    comptime if use_launcher:
        block_scaled_grouped_matmul_amd_preb[
            fp6_format=0 if fmt == FP6Format.E2M3 else 1
        ](
            c_tt,
            a_tt,
            b_pre_tt,
            a_sc_tt,
            b_sc_tt,
            a_off_tt,
            eid_tt,
            ascale_toks,
            num_active,
            ctx,
            total_tokens,
        )
    else:
        PreShuffledBGroupedGEMM[
            cu_count=cu_count,
            wg_per_cu=wg_per_cu,
            matrix_format=_mfma_format[fmt](),
        ].launch[
            BM=BM,
            BN=BN,
            BK_ELEMS=BK_ELEMS,
            WN=WN,
            persistent=persistent,
            b_cache_policy=b_cache_policy,
            cluster_drain_sched=cluster_drain_sched,
            mfma_cluster=mfma_cluster,
            pipeline_depth=pipeline_depth,
            waves_per_eu=waves_per_eu,
            static_grid_z=static_grid_z,
        ](
            c_tt,
            a_tt,
            b_pre_tt,
            a_sc_tt,
            b_sc_tt,
            a_off_tt,
            eid_tt,
            ascale_toks,  # max_num_tokens_per_expert = A-scale slot stride
            num_active,
            ctx,
            grid_m_cap,
        )
    ctx.synchronize()

    var c_h = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    var c_ref_h = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    var mag_h = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    ctx.enqueue_copy(c_h, c_d)
    ctx.enqueue_copy(c_ref_h, c_ref_d)
    ctx.enqueue_copy(mag_h, mag_d)
    ctx.synchronize()

    # float32 accumulation error grows with the number of summed terms, so the
    # bound is K-scaled and applied to sum |a*b| -- not to a result that may
    # have cancelled to near zero, where any relative bound is meaningless.
    comptime ULP_F32 = 5.9604644775390625e-8
    var rel_tol = Float64(16 * K) * ULP_F32

    var mismatches = 0
    var saw_nonzero = False
    for i in range(total_tokens * N):
        var want = Float64(c_ref_h[i])
        var got = Float64(c_h[i])
        if want != Float64(0.0):
            saw_nonzero = True
        if abs(got - want) > rel_tol * Float64(mag_h[i]):
            if mismatches < 3:
                print(
                    "      [",
                    i // N,
                    ",",
                    i % N,
                    "] got=",
                    got,
                    " want=",
                    want,
                    " mag=",
                    Float64(mag_h[i]),
                )
            mismatches += 1

    _ = a_d^
    _ = b_d^
    _ = b_pre_d^
    _ = a_sc_d^
    _ = b_sc_d^
    _ = a_sc_pre_d^
    _ = b_sc_pre_d^
    _ = a_off_d^
    _ = eid_d^
    _ = c_d^
    _ = c_ref_d^
    _ = mag_d^

    # An all-zero reference would make the comparison vacuous -- it would pass
    # against a kernel that wrote nothing.
    if not saw_nonzero:
        print("    FAIL  reference is all zero; comparison proves nothing")
        return False
    if mismatches > 0:
        print("    FAIL ", mismatches, " wrong of ", total_tokens * N)
        return False
    print("    PASS")
    return True


def test_persistent[
    fmt: FP6Format,
    num_experts: Int,
    N: Int,
    K: Int,
    BK_ELEMS: Int = 512,
    cu_count: Int = 256,  # default = MI355X
    BM: Int = 64,
    BN: Int = 128,
    WN: Int = 64,
    b_cache_policy: CacheOperation = CacheOperation.ALWAYS,
    cluster_drain_sched: Bool = False,
    mfma_cluster: Int = 4,
    pipeline_depth: Int = 2,
    waves_per_eu: Int = 0,
    wg_per_cu: Int = 2,
](
    name: String,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
) raises -> Bool:
    """`PreShuffledBGroupedGEMM.launch[persistent=True]` — 1D `total_wg` grid
    walking a global tile counter; XCD swizzle on `block_idx.x`.

    `cu_count` overrides the launch grid size; lowering it shrinks
    `total_wg = cu_count * wg_per_cu`, which lets the multi-wave case trigger
    the per-WG `while target_tile < expert_end` path on a small workload.
    """
    return _run_preb[
        fmt,
        num_experts,
        N,
        K,
        BK_ELEMS=BK_ELEMS,
        persistent=True,
        cu_count=cu_count,
        BM=BM,
        BN=BN,
        WN=WN,
        b_cache_policy=b_cache_policy,
        cluster_drain_sched=cluster_drain_sched,
        mfma_cluster=mfma_cluster,
        pipeline_depth=pipeline_depth,
        waves_per_eu=waves_per_eu,
        wg_per_cu=wg_per_cu,
    ](name, num_tokens_by_expert, expert_ids_list, ctx)


def test_direct[
    fmt: FP6Format,
    num_experts: Int,
    N: Int,
    K: Int,
    BK_ELEMS: Int = 512,
    cu_count: Int = 256,
    BM: Int = 64,
    BN: Int = 128,
    WN: Int = 64,
    b_cache_policy: CacheOperation = CacheOperation.ALWAYS,
    cluster_drain_sched: Bool = False,
    mfma_cluster: Int = 4,
    pipeline_depth: Int = 2,
    waves_per_eu: Int = 0,
    wg_per_cu: Int = 2,
    static_grid_z: Bool = False,
](
    name: String,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
    grid_m_cap: Int = -1,
    ascale_stride_toks: Int = -1,
    poison_ascale_pad: Bool = False,
) raises -> Bool:
    """`PreShuffledBGroupedGEMM.launch[persistent=False]` — 3D workload-sized
    grid: one WG per (n_tile, m_tile, expert). `wg_per_cu` is accepted for
    signature parity with `test_persistent`; the direct grid ignores it."""
    return _run_preb[
        fmt,
        num_experts,
        N,
        K,
        BK_ELEMS=BK_ELEMS,
        persistent=False,
        cu_count=cu_count,
        BM=BM,
        BN=BN,
        WN=WN,
        b_cache_policy=b_cache_policy,
        cluster_drain_sched=cluster_drain_sched,
        mfma_cluster=mfma_cluster,
        pipeline_depth=pipeline_depth,
        waves_per_eu=waves_per_eu,
        wg_per_cu=wg_per_cu,
        static_grid_z=static_grid_z,
    ](
        name,
        num_tokens_by_expert,
        expert_ids_list,
        ctx,
        grid_m_cap,
        ascale_stride_toks,
        poison_ascale_pad,
    )


def test_launcher[
    fmt: FP6Format,
    num_experts: Int,
    N: Int,
    K: Int,
](
    name: String,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
) raises -> Bool:
    """`block_scaled_grouped_matmul_amd_preb` — the tuned band dispatcher.

    Covers what driving `PreShuffledBGroupedGEMM.launch` directly cannot: that
    the launcher infers `lane_bytes=24` from the a / a_scales shapes rather
    than mistaking FP6 for FP8, that `fp6_format` reaches the MFMA, and that
    whichever tile config the (N, K, etm) band selects is actually valid for a
    24-byte lane fragment.
    """
    return _run_preb[
        fmt,
        num_experts,
        N,
        K,
        BK_ELEMS=512,  # ignored; the launcher picks its own
        persistent=True,
        cu_count=256,
        use_launcher=True,
    ](name, num_tokens_by_expert, expert_ids_list, ctx)


def main() raises:
    seed(0)
    var ctx = DeviceContext()

    print("===> preshuffled-B grouped MXFP6 — exhaustive kernel-level tests")

    var ok = True

    # ----------------------------------------------------------------- #
    # Shape conventions (matching test_mxfp4_moe_matmul_amd_routed.mojo):
    #   L2 decode/prefill shape:  N=512,   K=2048
    #   L2 gate+up aspect:        N=1024,  K=512
    #   L2 down aspect:           N=512,   K=1024
    #   L3 kimi gate+up (real):   N=14336, K=4096
    #   L3 kimi down (real):      N=4096,  K=7168
    # ----------------------------------------------------------------- #

    # ----------------------------------------------------------------- #
    # Tuned band dispatcher — block_scaled_grouped_matmul_amd_preb
    # ----------------------------------------------------------------- #
    print("---- preb launcher (tuned band dispatch) ----")

    # K must be a multiple of 512 for the FP6 bands. Token counts straddle the
    # etm band edges the launcher switches on (<=256, <=512, <=2100, above).
    ok &= test_launcher[FP6Format.E2M3, 1, 512, 2048](
        "single-decode", [16], [0], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 1, 512, 2048](
        "single-mid", [300], [0], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 4, 512, 2048](
        "multi-mixed", [32, 64, 128, 256], [0, 1, 2, 3], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 4, 512, 2048](
        "inactive-M0", [64, 0, 128, 32], [0, 1, 2, 3], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 1024, 512](
        "gate-up-aspect", [96, 96], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 512, 1024](
        "down-aspect", [96, 96], [0, 1], ctx
    )
    # Crosses into the prefill band (etm > 2100).
    ok &= test_launcher[FP6Format.E2M3, 2, 512, 2048](
        "prefill", [1200, 1200], [0, 1], ctx
    )

    ok &= test_launcher[FP6Format.E3M2, 1, 512, 2048](
        "single-decode", [16], [0], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 1, 512, 2048](
        "single-mid", [300], [0], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 4, 512, 2048](
        "multi-mixed", [32, 64, 128, 256], [0, 1, 2, 3], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 4, 512, 2048](
        "inactive-M0", [64, 0, 128, 32], [0, 1, 2, 3], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 1024, 512](
        "gate-up-aspect", [96, 96], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 512, 1024](
        "down-aspect", [96, 96], [0, 1], ctx
    )
    # Crosses into the prefill band (etm > 2100).
    ok &= test_launcher[FP6Format.E3M2, 2, 512, 2048](
        "prefill", [1200, 1200], [0, 1], ctx
    )

    # Production kimi up-proj shape (N=4096, K=7168): the launcher's own
    # etm bands for this (N, K) at lane_bytes=24 -- straddle each edge
    # (<=20, <=127, <=255, above) so both the boundary and the tile picked
    # just past it get a real dispatch-table run, not just the raw kernel
    # coverage below (which bypasses the launcher).
    ok &= test_launcher[FP6Format.E2M3, 2, 4096, 7168](
        "up etm==20", [10, 10], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 4096, 7168](
        "up etm==21", [11, 10], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 4096, 7168](
        "up etm==127", [64, 63], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 4096, 7168](
        "up etm==128", [64, 64], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 4096, 7168](
        "up etm==255", [128, 127], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 4096, 7168](
        "up etm==256", [128, 128], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 4096, 7168](
        "up etm large", [2500, 2500], [0, 1], ctx
    )

    ok &= test_launcher[FP6Format.E3M2, 2, 4096, 7168](
        "up etm==20", [10, 10], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 4096, 7168](
        "up etm==21", [11, 10], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 4096, 7168](
        "up etm==127", [64, 63], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 4096, 7168](
        "up etm==128", [64, 64], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 4096, 7168](
        "up etm==255", [128, 127], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 4096, 7168](
        "up etm==256", [128, 128], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 4096, 7168](
        "up etm large", [2500, 2500], [0, 1], ctx
    )

    # Real production grouped-expert shapes (EP=8 on MI355X: hidden=6144,
    # expert_intermediate=3072, fused gate+up output=2*3072=6144). Deep-tuning
    # pass band edges: <=3, <=7, <=31, <=255, above (see
    # mxfp6-grouped-deep-tuning-bands.md) -- straddle each of the three NEW
    # edges (<=3/4, <=7/8, <=31/32) in addition to the unchanged <=255/256
    # edge already covered below.
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 6144](
        "m3 gate_up etm==3", [2, 1], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 6144](
        "m3 gate_up etm==4", [2, 2], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 6144](
        "m3 gate_up etm==7", [4, 3], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 6144](
        "m3 gate_up etm==8", [4, 4], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 6144](
        "m3 gate_up etm==31", [16, 15], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 6144](
        "m3 gate_up etm==32", [16, 16], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 6144](
        "m3 gate_up etm==33", [17, 16], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 6144](
        "m3 gate_up etm==127", [64, 63], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 6144](
        "m3 gate_up etm==128", [64, 64], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 6144](
        "m3 gate_up etm==255", [128, 127], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 6144](
        "m3 gate_up etm==256", [128, 128], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 6144](
        "m3 gate_up etm large", [2500, 2500], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 3072](
        "m3 down etm==3", [2, 1], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 3072](
        "m3 down etm==4", [2, 2], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 3072](
        "m3 down etm==7", [4, 3], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 3072](
        "m3 down etm==8", [4, 4], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 3072](
        "m3 down etm==31", [16, 15], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 3072](
        "m3 down etm==32", [16, 16], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 3072](
        "m3 down etm==33", [17, 16], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 3072](
        "m3 down etm==127", [64, 63], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 3072](
        "m3 down etm==128", [64, 64], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 3072](
        "m3 down etm==255", [128, 127], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 3072](
        "m3 down etm==256", [128, 128], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 3072](
        "m3 down etm large", [2500, 2500], [0, 1], ctx
    )

    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 6144](
        "m3 gate_up etm==3", [2, 1], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 6144](
        "m3 gate_up etm==4", [2, 2], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 6144](
        "m3 gate_up etm==7", [4, 3], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 6144](
        "m3 gate_up etm==8", [4, 4], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 6144](
        "m3 gate_up etm==31", [16, 15], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 6144](
        "m3 gate_up etm==32", [16, 16], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 6144](
        "m3 gate_up etm==33", [17, 16], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 6144](
        "m3 gate_up etm==127", [64, 63], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 6144](
        "m3 gate_up etm==128", [64, 64], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 6144](
        "m3 gate_up etm==255", [128, 127], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 6144](
        "m3 gate_up etm==256", [128, 128], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 6144](
        "m3 gate_up etm large", [2500, 2500], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 3072](
        "m3 down etm==3", [2, 1], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 3072](
        "m3 down etm==4", [2, 2], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 3072](
        "m3 down etm==7", [4, 3], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 3072](
        "m3 down etm==8", [4, 4], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 3072](
        "m3 down etm==31", [16, 15], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 3072](
        "m3 down etm==32", [16, 16], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 3072](
        "m3 down etm==33", [17, 16], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 3072](
        "m3 down etm==127", [64, 63], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 3072](
        "m3 down etm==128", [64, 64], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 3072](
        "m3 down etm==255", [128, 127], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 3072](
        "m3 down etm==256", [128, 128], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 3072](
        "m3 down etm large", [2500, 2500], [0, 1], ctx
    )

    # Generic (N, K) fallback band -- covers any MXFP6 shape that doesn't
    # match one of the shape-specific bands above (narrow N or short K, e.g.
    # per-rank expert widths outside the two measured production models).
    # These used to fall through to the untuned BM=64 fallback and hit the
    # same register-spill cliff the shape-specific bands exist to avoid; see
    # Reuses the M3 bands' etm edges (<=3, <=7, <=31, <=255, above) at a
    # narrow-N shape (N=2048) and a short-K shape (K=512) neither of which
    # matches N==4096/N==6144 above.
    ok &= test_launcher[FP6Format.E2M3, 2, 2048, 3072](
        "generic narrow-N etm==3", [2, 1], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 2048, 3072](
        "generic narrow-N etm==4", [2, 2], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 2048, 3072](
        "generic narrow-N etm==7", [4, 3], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 2048, 3072](
        "generic narrow-N etm==8", [4, 4], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 2048, 3072](
        "generic narrow-N etm==31", [16, 15], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 2048, 3072](
        "generic narrow-N etm==32", [16, 16], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 2048, 3072](
        "generic narrow-N etm==255", [128, 127], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 2048, 3072](
        "generic narrow-N etm==256", [128, 128], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 2048, 3072](
        "generic narrow-N etm large", [2500, 2500], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 2048, 3072](
        "generic narrow-N etm==32", [16, 16], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 2048, 3072](
        "generic narrow-N etm large", [2500, 2500], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 512](
        "generic short-K etm==3", [2, 1], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 512](
        "generic short-K etm==32", [16, 16], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E2M3, 2, 6144, 512](
        "generic short-K etm large", [2500, 2500], [0, 1], ctx
    )
    ok &= test_launcher[FP6Format.E3M2, 2, 6144, 512](
        "generic short-K etm==32", [16, 16], [0, 1], ctx
    )

    # ----------------------------------------------------------------- #
    # Preb persistent kernel — launch[persistent=True]
    # ----------------------------------------------------------------- #
    print("---- preb persistent kernel ----")

    # Structural edge cases (L2 decode shape: N=512, K=2048).
    ok &= test_persistent[
        FP6Format.E2M3, 1, 512, 2048, cluster_drain_sched=True
    ]("single-tiny", [16], [0], ctx)
    ok &= test_persistent[
        FP6Format.E3M2, 1, 512, 2048, cluster_drain_sched=True
    ]("single-tiny", [16], [0], ctx)
    ok &= test_persistent[
        FP6Format.E2M3, 1, 512, 2048, cluster_drain_sched=True
    ]("single-mid", [128], [0], ctx)
    ok &= test_persistent[
        FP6Format.E3M2, 1, 512, 2048, cluster_drain_sched=True
    ]("single-mid", [128], [0], ctx)
    ok &= test_persistent[
        FP6Format.E2M3, 4, 512, 2048, cluster_drain_sched=True
    ]("multi-mixed", [32, 64, 128, 256], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2, 4, 512, 2048, cluster_drain_sched=True
    ]("multi-mixed", [32, 64, 128, 256], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E2M3, 4, 512, 2048, cluster_drain_sched=True
    ]("inactive-M0", [64, 0, 128, 32], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2, 4, 512, 2048, cluster_drain_sched=True
    ]("inactive-M0", [64, 0, 128, 32], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E2M3, 4, 512, 2048, cluster_drain_sched=True
    ]("inactive-eid-1", [64, 64, 128, 32], [0, -1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2, 4, 512, 2048, cluster_drain_sched=True
    ]("inactive-eid-1", [64, 64, 128, 32], [0, -1, 2, 3], ctx)
    # More tiles than total_wg (= 512 on MI355X) — exercise multi-wave per WG.
    # 9 experts × m_count=4 × gx_n(N=1024)=8 = 288 tiles? need >512.
    # 9 × m_count=8 × gx_n=8 = 576 → M=512 per expert.
    ok &= test_persistent[
        FP6Format.E2M3, 9, 1024, 512, cluster_drain_sched=True
    ](
        "multi-wave",
        [512, 512, 512, 512, 512, 512, 512, 512, 512],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        ctx,
    )
    ok &= test_persistent[
        FP6Format.E3M2, 9, 1024, 512, cluster_drain_sched=True
    ](
        "multi-wave",
        [512, 512, 512, 512, 512, 512, 512, 512, 512],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        ctx,
    )
    # Kimi-decode-like: 49 active experts, very few tokens each, mixed
    # inactive slots (slot 40: M=0; slot 47: expert_id=-1) — matches the
    # L2.2 / L2.3 pattern from test_mxfp4_moe_matmul_amd_routed.
    ok &= test_persistent[
        FP6Format.E2M3, 49, 512, 2048, cluster_drain_sched=True
    ](
        "kimi-decode-49experts",
        [
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            0,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            -1,
            48,
        ],
        ctx,
    )
    ok &= test_persistent[
        FP6Format.E3M2, 49, 512, 2048, cluster_drain_sched=True
    ](
        "kimi-decode-49experts",
        [
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            0,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            -1,
            48,
        ],
        ctx,
    )
    # Kimi-prefill scaled (L2.4): 49 active experts, ~40 tokens each.
    ok &= test_persistent[
        FP6Format.E2M3, 49, 512, 2048, cluster_drain_sched=True
    ](
        "kimi-prefill-49experts",
        [
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
        ],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
        ],
        ctx,
    )
    ok &= test_persistent[
        FP6Format.E3M2, 49, 512, 2048, cluster_drain_sched=True
    ](
        "kimi-prefill-49experts",
        [
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
        ],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
        ],
        ctx,
    )
    # Tile-size variants on the persistent path. Cover the kernel's
    # supported BM/WN matrix (BM ∈ {16} ∪ multiples of 32, same for WN).
    print("---- preb persistent kernel — tile-size variants ----")

    # BM=16 path: 1-row-per-sub-MMA M, shrui scale rotation on odd-parity CTAs.
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        512,
        2048,
        BM=16,
        BN=128,
        WN=64,
        cluster_drain_sched=True,
    ]("bm16-default-wn", [16, 32, 8, 48], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        512,
        2048,
        BM=16,
        BN=128,
        WN=64,
        cluster_drain_sched=True,
    ]("bm16-default-wn", [16, 32, 8, 48], [0, 1, 2, 3], ctx)
    # WN=16 path: N-side shrui rotation; BM unchanged.
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        512,
        2048,
        BM=64,
        BN=64,
        WN=16,
        cluster_drain_sched=True,
    ]("wn16-default-bm", [64, 32, 16, 48], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        512,
        2048,
        BM=64,
        BN=64,
        WN=16,
        cluster_drain_sched=True,
    ]("wn16-default-bm", [64, 32, 16, 48], [0, 1, 2, 3], ctx)
    # Both BM=16 and WN=16 — exercises both shrui paths simultaneously.
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        256,
        2048,
        BM=16,
        BN=64,
        WN=16,
        cluster_drain_sched=True,
    ]("bm16-wn16", [16, 32, 8, 24], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        256,
        2048,
        BM=16,
        BN=64,
        WN=16,
        cluster_drain_sched=True,
    ]("bm16-wn16", [16, 32, 8, 24], [0, 1, 2, 3], ctx)
    # Smaller BN=64 with default BM/WN — exercises the N-tile dispatcher
    # at a non-default BN.
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        256,
        2048,
        BM=64,
        BN=64,
        WN=64,
        cluster_drain_sched=True,
    ]("bn64", [64, 128, 32, 96], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        256,
        2048,
        BM=64,
        BN=64,
        WN=64,
        cluster_drain_sched=True,
    ]("bn64", [64, 128, 32, 96], [0, 1, 2, 3], ctx)
    # ----------------------------------------------------------------- #
    # Preb direct kernel — launch[persistent=False]
    # ----------------------------------------------------------------- #
    print("---- preb direct kernel ----")
    ok &= test_direct[FP6Format.E2M3, 1, 512, 2048, cluster_drain_sched=True](
        "single-tiny", [16], [0], ctx
    )
    ok &= test_direct[FP6Format.E3M2, 1, 512, 2048, cluster_drain_sched=True](
        "single-tiny", [16], [0], ctx
    )
    ok &= test_direct[FP6Format.E2M3, 1, 512, 2048, cluster_drain_sched=True](
        "single-mid", [128], [0], ctx
    )
    ok &= test_direct[FP6Format.E3M2, 1, 512, 2048, cluster_drain_sched=True](
        "single-mid", [128], [0], ctx
    )
    ok &= test_direct[FP6Format.E2M3, 4, 512, 2048, cluster_drain_sched=True](
        "multi-mixed", [32, 64, 128, 256], [0, 1, 2, 3], ctx
    )
    ok &= test_direct[FP6Format.E3M2, 4, 512, 2048, cluster_drain_sched=True](
        "multi-mixed", [32, 64, 128, 256], [0, 1, 2, 3], ctx
    )
    ok &= test_direct[FP6Format.E2M3, 4, 512, 2048, cluster_drain_sched=True](
        "inactive-M0", [64, 0, 128, 32], [0, 1, 2, 3], ctx
    )
    ok &= test_direct[FP6Format.E3M2, 4, 512, 2048, cluster_drain_sched=True](
        "inactive-M0", [64, 0, 128, 32], [0, 1, 2, 3], ctx
    )
    ok &= test_direct[FP6Format.E2M3, 4, 512, 2048, cluster_drain_sched=True](
        "inactive-eid-1", [64, 64, 128, 32], [0, -1, 2, 3], ctx
    )
    ok &= test_direct[FP6Format.E3M2, 4, 512, 2048, cluster_drain_sched=True](
        "inactive-eid-1", [64, 64, 128, 32], [0, -1, 2, 3], ctx
    )
    # Large single-expert prefill (where direct typically wins over persistent).
    ok &= test_direct[FP6Format.E2M3, 1, 1024, 512, cluster_drain_sched=True](
        "single-large", [8192], [0], ctx
    )
    ok &= test_direct[FP6Format.E3M2, 1, 1024, 512, cluster_drain_sched=True](
        "single-large", [8192], [0], ctx
    )
    # Tile-size variants on the direct path.
    print("---- preb direct kernel — tile-size variants ----")
    ok &= test_direct[
        FP6Format.E2M3,
        4,
        256,
        2048,
        BM=16,
        BN=64,
        WN=16,
        cluster_drain_sched=True,
    ]("bm16-wn16", [3, 7, 1, 5], [0, 1, 2, 3], ctx)
    ok &= test_direct[
        FP6Format.E3M2,
        4,
        256,
        2048,
        BM=16,
        BN=64,
        WN=16,
        cluster_drain_sched=True,
    ]("bm16-wn16", [3, 7, 1, 5], [0, 1, 2, 3], ctx)
    ok &= test_direct[
        FP6Format.E2M3,
        4,
        512,
        2048,
        BM=64,
        BN=64,
        WN=16,
        cluster_drain_sched=True,
    ]("wn16-default-bm", [128, 64, 32, 48], [0, 1, 2, 3], ctx)
    ok &= test_direct[
        FP6Format.E3M2,
        4,
        512,
        2048,
        BM=64,
        BN=64,
        WN=16,
        cluster_drain_sched=True,
    ]("wn16-default-bm", [128, 64, 32, 48], [0, 1, 2, 3], ctx)
    # Kimi-prefill scaled.
    ok &= test_direct[FP6Format.E2M3, 49, 512, 2048, cluster_drain_sched=True](
        "kimi-prefill-49experts",
        [
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
        ],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
        ],
        ctx,
    )
    ok &= test_direct[FP6Format.E3M2, 49, 512, 2048, cluster_drain_sched=True](
        "kimi-prefill-49experts",
        [
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
        ],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
        ],
        ctx,
    )
    # ----------------------------------------------------------------- #
    # Production dispatch-band coverage — real kimi N/K with the EXACT
    # (BM, BN, BK_ELEMS, WN, persistent, b_cache_policy) each band in
    # block_scaled_grouped_matmul_amd.mojo launches. One representative token
    # distribution per band (a few active experts + an inactive slot
    # where useful). STREAMING vs ALWAYS is result-identical (cache hint);
    # these cases assert the exact instantiation compiles, runs, and is
    # correct. M values are illustrative of each band's regime, not the
    # dispatcher's estimated_total_m selector (this path bypasses it).
    # ----------------------------------------------------------------- #
    print("---- production band coverage: KIMI up-proj (N=4096, K=7168) ----")
    # band M==1 (decode)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        4096,
        7168,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("up M==1", [1, 1, 1, 1], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        4096,
        7168,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("up M==1", [1, 1, 1, 1], [0, 1, 2, 3], ctx)
    # band 2<=M<=4
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        4096,
        7168,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        cluster_drain_sched=True,
    ]("up 2..4", [4, 2, 3, 4], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        4096,
        7168,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        cluster_drain_sched=True,
    ]("up 2..4", [4, 2, 3, 4], [0, 1, 2, 3], ctx)
    # band 17<=M<=400
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        4096,
        7168,
        BM=32,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        cluster_drain_sched=True,
    ]("up 17..400", [200, 64, 128, 32], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        4096,
        7168,
        BM=32,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        cluster_drain_sched=True,
    ]("up 17..400", [200, 64, 128, 32], [0, 1, 2, 3], ctx)
    # default persistent fallback (M in the 5..16 / 401..4095 gaps)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        4096,
        7168,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("up default-fallback", [512, 128, 256, 0], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        4096,
        7168,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("up default-fallback", [512, 128, 256, 0], [0, 1, 2, 3], ctx)
    # use_direct path (persistent=False). Token count kept modest for memory;
    # the dispatcher is bypassed so M doesn't select the band — the config does.
    ok &= test_direct[
        FP6Format.E2M3,
        1,
        4096,
        7168,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("up direct", [1024], [0], ctx)
    ok &= test_direct[
        FP6Format.E3M2,
        1,
        4096,
        7168,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("up direct", [1024], [0], ctx)
    print("---- production band coverage: KIMI down-proj (N=7168, K=2048) ----")
    # band M==1 (decode)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        7168,
        2048,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        cluster_drain_sched=True,
    ]("down M==1", [1, 1, 1, 1], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        7168,
        2048,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        cluster_drain_sched=True,
    ]("down M==1", [1, 1, 1, 1], [0, 1, 2, 3], ctx)
    # band 2<=M<=7 (B cached)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        7168,
        2048,
        BM=16,
        BN=256,
        BK_ELEMS=256,
        WN=64,
        cluster_drain_sched=True,
    ]("down 2..7 cached", [4, 2, 6, 3], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        7168,
        2048,
        BM=16,
        BN=256,
        BK_ELEMS=256,
        WN=64,
        cluster_drain_sched=True,
    ]("down 2..7 cached", [4, 2, 6, 3], [0, 1, 2, 3], ctx)
    # band 8<=M<=16 (B STREAMING)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        7168,
        2048,
        BM=16,
        BN=256,
        BK_ELEMS=256,
        WN=64,
        b_cache_policy=CacheOperation.STREAMING,
        cluster_drain_sched=True,
    ]("down 8..16 STREAMING", [12, 8, 16, 10], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        7168,
        2048,
        BM=16,
        BN=256,
        BK_ELEMS=256,
        WN=64,
        b_cache_policy=CacheOperation.STREAMING,
        cluster_drain_sched=True,
    ]("down 8..16 STREAMING", [12, 8, 16, 10], [0, 1, 2, 3], ctx)
    # band 17<=M<=37 / 385<=M<=400 (same config, B cached)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        7168,
        2048,
        BM=32,
        BN=256,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("down 17..37 / 385..400 cached", [32, 24, 37, 0], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        7168,
        2048,
        BM=32,
        BN=256,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("down 17..37 / 385..400 cached", [32, 24, 37, 0], [0, 1, 2, 3], ctx)
    # band 38<=M<=384 (B STREAMING)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        7168,
        2048,
        BM=32,
        BN=256,
        BK_ELEMS=512,
        WN=64,
        b_cache_policy=CacheOperation.STREAMING,
        cluster_drain_sched=True,
    ]("down 38..384 STREAMING", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        7168,
        2048,
        BM=32,
        BN=256,
        BK_ELEMS=512,
        WN=64,
        b_cache_policy=CacheOperation.STREAMING,
        cluster_drain_sched=True,
    ]("down 38..384 STREAMING", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    # band 401<=M<=1200 (BK_ELEMS=256, B cached)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        7168,
        2048,
        BM=64,
        BN=256,
        BK_ELEMS=256,
        WN=64,
        cluster_drain_sched=True,
    ]("down 401..1200 cached", [256, 256, 128, 64], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        7168,
        2048,
        BM=64,
        BN=256,
        BK_ELEMS=256,
        WN=64,
        cluster_drain_sched=True,
    ]("down 401..1200 cached", [256, 256, 128, 64], [0, 1, 2, 3], ctx)
    # default persistent fallback (M in the 1201..4095 gap)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("down default-fallback", [384, 128, 128, 0], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("down default-fallback", [384, 128, 128, 0], [0, 1, 2, 3], ctx)
    # use_direct path (persistent=False); token count kept modest for memory.
    ok &= test_direct[
        FP6Format.E2M3,
        1,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("down direct", [512], [0], ctx)
    ok &= test_direct[
        FP6Format.E3M2,
        1,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("down direct", [512], [0], ctx)
    print("---- preb waves_per_eu EU-bounding cap ----")
    ok &= test_persistent[
        FP6Format.E2M3, 4, 512, 2048, waves_per_eu=1, cluster_drain_sched=True
    ]("wpe=1 persistent", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2, 4, 512, 2048, waves_per_eu=1, cluster_drain_sched=True
    ]("wpe=1 persistent", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E2M3, 4, 512, 2048, waves_per_eu=2, cluster_drain_sched=True
    ]("wpe=2 persistent", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2, 4, 512, 2048, waves_per_eu=2, cluster_drain_sched=True
    ]("wpe=2 persistent", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    ok &= test_direct[
        FP6Format.E2M3,
        1,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        waves_per_eu=2,
        cluster_drain_sched=True,
    ]("wpe=2 direct", [512], [0], ctx)
    ok &= test_direct[
        FP6Format.E3M2,
        1,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        waves_per_eu=2,
        cluster_drain_sched=True,
    ]("wpe=2 direct", [512], [0], ctx)
    # Each band config the retuned dispatcher selects.
    print("---- retuned dispatcher band configs ----")
    comptime SX = CacheOperation.STREAMING
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        4096,
        7168,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("up BM16/BN128 STREAM", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        4096,
        7168,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("up BM16/BN128 STREAM", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        4096,
        7168,
        BM=32,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("up BM32/BN128 STREAM", [256, 256, 256, 256], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        4096,
        7168,
        BM=32,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("up BM32/BN128 STREAM", [256, 256, 256, 256], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        7168,
        2048,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("down BN64 tiny", [2, 1, 0, 1], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        7168,
        2048,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("down BN64 tiny", [2, 1, 0, 1], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        7168,
        2048,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("down BM16/BN128 STREAM", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        7168,
        2048,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("down BM16/BN128 STREAM", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        7168,
        2048,
        BM=32,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("down BM32/BN128 STREAM", [256, 256, 256, 256], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        7168,
        2048,
        BM=32,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("down BM32/BN128 STREAM", [256, 256, 256, 256], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("down BM64/BN128 STREAM", [512, 512, 256, 256], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("down BM64/BN128 STREAM", [512, 512, 256, 256], [0, 1, 2, 3], ctx)
    ok &= test_direct[
        FP6Format.E2M3,
        1,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=256,
        WN=64,
        cluster_drain_sched=True,
    ]("down direct BK256", [512], [0], ctx)
    ok &= test_direct[
        FP6Format.E3M2,
        1,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=256,
        WN=64,
        cluster_drain_sched=True,
    ]("down direct BK256", [512], [0], ctx)
    # Remaining bands the retuned dispatcher can select, plus the real
    # EP=4 batch-1 decode point and a skewed-routing stress case.
    # up etm<=20 ALWAYS band (16,64,512,16).
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        4096,
        7168,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("up BM16/BN64 tiny ALWAYS", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        4096,
        7168,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("up BM16/BN64 tiny ALWAYS", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    # up etm<=4095 STREAM band (64,128,512,64). Token count kept modest to
    # stay under the harness HBM ceiling; config is what's under test.
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        4096,
        7168,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("up BM64/BN128 STREAM", [192, 192, 128, 128], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        4096,
        7168,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("up BM64/BN128 STREAM", [192, 192, 128, 128], [0, 1, 2, 3], ctx)
    # up else direct band (64,128,512,64).
    ok &= test_direct[
        FP6Format.E2M3,
        1,
        4096,
        7168,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("up direct", [512], [0], ctx)
    ok &= test_direct[
        FP6Format.E3M2,
        1,
        4096,
        7168,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("up direct", [512], [0], ctx)
    # etm==1 wg_per_cu=1 variant for both shapes (16,64,512,16). wg_per_cu is
    # a grid-sizing comptime; results must match the wg_per_cu=2 default.
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        4096,
        7168,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        cluster_drain_sched=True,
    ]("up etm1 wg_per_cu=1", [1, 0, 0, 0], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        4096,
        7168,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        cluster_drain_sched=True,
    ]("up etm1 wg_per_cu=1", [1, 0, 0, 0], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        7168,
        2048,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        cluster_drain_sched=True,
    ]("down etm1 wg_per_cu=1", [1, 0, 0, 0], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        7168,
        2048,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        cluster_drain_sched=True,
    ]("down etm1 wg_per_cu=1", [1, 0, 0, 0], [0, 1, 2, 3], ctx)
    # True EP=4 batch-1 decode point: 2 experts, 1 row each (M=2), tile
    # (16,64,512,16) for both up and down.
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        4096,
        7168,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("up EP4 decode M=2", [1, 1, 0, 0], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        4096,
        7168,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("up EP4 decode M=2", [1, 1, 0, 0], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        7168,
        2048,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("down EP4 decode M=2", [1, 1, 0, 0], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        7168,
        2048,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("down EP4 decode M=2", [1, 1, 0, 0], [0, 1, 2, 3], ctx)
    # Skewed routing (one hot expert, etm~203 -> (16,128,512,32) STREAM band).
    # Stresses persistent work-stealing under imbalance.
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        4096,
        7168,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("up skewed hot expert", [200, 1, 1, 1], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        4096,
        7168,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("up skewed hot expert", [200, 1, 1, 1], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        7168,
        2048,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("down skewed hot expert", [200, 1, 1, 1], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        7168,
        2048,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("down skewed hot expert", [200, 1, 1, 1], [0, 1, 2, 3], ctx)
    # MiniMax-M3 dispatcher band configs (N=6144; up K=6144, down K=3072).
    # Exactly the tiles block_scaled_grouped_matmul_amd_preb selects for M3 (defaults
    # for the scheduler knobs, matching run_kernel). Certifies each M3 band
    # instantiation against the per-expert reference.
    print("---- MiniMax-M3 dispatcher band configs ----")
    # up: etm<=128 BN64 STREAM
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
    ]("M3 up etm<=128 BN64 STREAM", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
    ]("M3 up etm<=128 BN64 STREAM", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    # up: etm 7..14 dip micro-band BM16/BN128 STREAM
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        6144,
        6144,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
    ]("M3 up etm7..14 BM16/BN128 STREAM", [4, 2, 3, 4], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        6144,
        6144,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
    ]("M3 up etm7..14 BM16/BN128 STREAM", [4, 2, 3, 4], [0, 1, 2, 3], ctx)
    # up: etm<=512 BM32/BN128 STREAM
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        6144,
        6144,
        BM=32,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
    ]("M3 up etm<=512 BM32/BN128 STREAM", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        6144,
        6144,
        BM=32,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
    ]("M3 up etm<=512 BM32/BN128 STREAM", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    # up: etm<=2047 BM64/BN128
    ok &= test_persistent[
        FP6Format.E2M3, 4, 6144, 6144, BM=64, BN=128, BK_ELEMS=512, WN=32
    ](
        "M3 up etm<=1023 BM64/BN128/WN32",
        [256, 256, 256, 256],
        [0, 1, 2, 3],
        ctx,
    )
    ok &= test_persistent[
        FP6Format.E3M2, 4, 6144, 6144, BM=64, BN=128, BK_ELEMS=512, WN=32
    ](
        "M3 up etm<=1023 BM64/BN128/WN32",
        [256, 256, 256, 256],
        [0, 1, 2, 3],
        ctx,
    )
    ok &= test_persistent[
        FP6Format.E2M3, 4, 6144, 6144, BM=64, BN=128, BK_ELEMS=512, WN=64
    ](
        "M3 up etm<=2047 BM64/BN128/WN64",
        [256, 256, 256, 256],
        [0, 1, 2, 3],
        ctx,
    )
    ok &= test_persistent[
        FP6Format.E3M2, 4, 6144, 6144, BM=64, BN=128, BK_ELEMS=512, WN=64
    ](
        "M3 up etm<=2047 BM64/BN128/WN64",
        [256, 256, 256, 256],
        [0, 1, 2, 3],
        ctx,
    )
    # up: etm<=4095 BM128/BN128
    ok &= test_persistent[
        FP6Format.E2M3, 4, 6144, 6144, BM=128, BN=128, BK_ELEMS=512, WN=64
    ]("M3 up etm<=4095 BM128/BN128", [256, 128, 256, 128], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2, 4, 6144, 6144, BM=128, BN=128, BK_ELEMS=512, WN=64
    ]("M3 up etm<=4095 BM128/BN128", [256, 128, 256, 128], [0, 1, 2, 3], ctx)
    # up: else direct BM64/BN128
    ok &= test_direct[
        FP6Format.E2M3, 1, 6144, 6144, BM=64, BN=128, BK_ELEMS=512, WN=64
    ]("M3 up direct", [512], [0], ctx)
    ok &= test_direct[
        FP6Format.E3M2, 1, 6144, 6144, BM=64, BN=128, BK_ELEMS=512, WN=64
    ]("M3 up direct", [512], [0], ctx)
    # down: etm<=96 BN64 STREAM
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
    ]("M3 down etm<=96 BN64 STREAM", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
    ]("M3 down etm<=96 BN64 STREAM", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    # down: etm 7..14 dip micro-band BM16/BN128 STREAM
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        6144,
        3072,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
    ]("M3 down etm7..14 BM16/BN128 STREAM", [4, 2, 3, 4], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        6144,
        3072,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
    ]("M3 down etm7..14 BM16/BN128 STREAM", [4, 2, 3, 4], [0, 1, 2, 3], ctx)
    # down: etm<=384 BM32/BN256
    ok &= test_persistent[
        FP6Format.E2M3, 4, 6144, 3072, BM=32, BN=256, BK_ELEMS=512, WN=64
    ]("M3 down etm<=384 BM32/BN256", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2, 4, 6144, 3072, BM=32, BN=256, BK_ELEMS=512, WN=64
    ]("M3 down etm<=384 BM32/BN256", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    # down: etm<=1536 BM64/BN128
    ok &= test_persistent[
        FP6Format.E2M3, 4, 6144, 3072, BM=64, BN=128, BK_ELEMS=512, WN=32
    ]("M3 down etm<=1536 BM64/BN128", [256, 256, 256, 256], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2, 4, 6144, 3072, BM=64, BN=128, BK_ELEMS=512, WN=32
    ]("M3 down etm<=1536 BM64/BN128", [256, 256, 256, 256], [0, 1, 2, 3], ctx)
    # down: else BM128/BN128 persistent
    ok &= test_persistent[
        FP6Format.E2M3, 4, 6144, 3072, BM=128, BN=128, BK_ELEMS=512, WN=64
    ]("M3 down else BM128/BN128", [256, 128, 256, 128], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2, 4, 6144, 3072, BM=128, BN=128, BK_ELEMS=512, WN=64
    ]("M3 down else BM128/BN128", [256, 128, 256, 128], [0, 1, 2, 3], ctx)
    # down decode band: direct capped grid.y + static grid.z vs ref, across
    # balanced / skew / BM32 / static-z empties / A-scale decoupling.
    ok &= test_direct[
        FP6Format.E2M3,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct+cap bal",
        [8, 8, 8, 8],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    ok &= test_direct[
        FP6Format.E3M2,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct+cap bal",
        [8, 8, 8, 8],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    ok &= test_direct[
        FP6Format.E2M3,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct+cap skew",
        [32, 4, 4, 4],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    ok &= test_direct[
        FP6Format.E3M2,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct+cap skew",
        [32, 4, 4, 4],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    ok &= test_direct[
        FP6Format.E2M3,
        4,
        6144,
        3072,
        BM=32,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct+cap BM32",
        [8, 8, 8, 8],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    ok &= test_direct[
        FP6Format.E3M2,
        4,
        6144,
        3072,
        BM=32,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct+cap BM32",
        [8, 8, 8, 8],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    # static grid.z over-launch: 8 slots, 2 active; 6 M==0 slots early-return.
    ok &= test_direct[
        FP6Format.E2M3,
        8,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct+cap static-z empties",
        [32, 0, 8, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7],
        ctx,
        grid_m_cap=32,
    )
    ok &= test_direct[
        FP6Format.E3M2,
        8,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct+cap static-z empties",
        [32, 0, 8, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7],
        ctx,
        grid_m_cap=32,
    )
    # Decoupling: A-scale stride (512) >> grid.y cap (32).
    ok &= test_direct[
        FP6Format.E2M3,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct decouple stride=512",
        [32, 4, 4, 4],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
        ascale_stride_toks=512,
    )
    ok &= test_direct[
        FP6Format.E3M2,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct decouple stride=512",
        [32, 4, 4, 4],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
        ascale_stride_toks=512,
    )
    # Same geometry, but with E8M0 NaN in every A-scale pad slot. In production
    # the slot buffer is pooled memory whose pad rows hold stale bytes, and
    # `max-debug.nan-check` reports NaN there on every MXFP6 serve. Correctness
    # rests entirely on the per-expert V# bound clamping reads past
    # align_up(num_tokens, 32); if it ever fails, poisoned slots reach the
    # result. 4 real tokens against a 512-token stride is the widest pad the
    # decode band produces.
    ok &= test_direct[
        FP6Format.E2M3,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct stride=512 POISONED pad",
        [32, 4, 4, 4],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
        ascale_stride_toks=512,
        poison_ascale_pad=True,
    )
    ok &= test_direct[
        FP6Format.E2M3,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct stride=512 POISONED pad, 1 token",
        [1, 1, 1, 1],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
        ascale_stride_toks=512,
        poison_ascale_pad=True,
    )
    # The up-proj A-scale slot buffer, poisoned. This geometry (K=6144 ->
    # scale_K=192) is the one MXFP6 alone drives through the standalone
    # preshuffle: an IR diff against MXFP8 shows FP6 invoking
    # `mo.mxfp4.preshuffle.scale.4d_per_expert` 912x versus MXFP8's 456x --
    # twice per MoE layer per device instead of once -- because MXFP8's fused
    # `ep.fused_silu` writes up-proj scales straight into slot layout and FP6
    # has no such producer. 192 is also the scale width of the tensor
    # `max-debug.nan-check` reports NaN in on every MXFP6 serve, and the
    # existing poisoned case only covers the down-proj width (scale_K=96).
    ok &= test_direct[
        FP6Format.E2M3,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct stride=512 POISONED pad",
        [32, 4, 4, 4],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
        ascale_stride_toks=512,
        poison_ascale_pad=True,
    )
    ok &= test_direct[
        FP6Format.E2M3,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct stride=512 POISONED pad, 1 token",
        [1, 1, 1, 1],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
        ascale_stride_toks=512,
        poison_ascale_pad=True,
    )
    # gate_up decode band (deep K=6144): same coverage as the down cases.
    ok &= test_direct[
        FP6Format.E2M3,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct+cap bal",
        [8, 8, 8, 8],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    ok &= test_direct[
        FP6Format.E3M2,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct+cap bal",
        [8, 8, 8, 8],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    ok &= test_direct[
        FP6Format.E2M3,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct+cap skew",
        [32, 4, 4, 4],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    ok &= test_direct[
        FP6Format.E3M2,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct+cap skew",
        [32, 4, 4, 4],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    ok &= test_direct[
        FP6Format.E2M3,
        4,
        6144,
        6144,
        BM=32,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct+cap BM32",
        [8, 8, 8, 8],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    ok &= test_direct[
        FP6Format.E3M2,
        4,
        6144,
        6144,
        BM=32,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct+cap BM32",
        [8, 8, 8, 8],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    # static grid.z over-launch: 8 slots, 2 active; 6 M==0 slots early-return.
    ok &= test_direct[
        FP6Format.E2M3,
        8,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct+cap static-z empties",
        [32, 0, 8, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7],
        ctx,
        grid_m_cap=32,
    )
    ok &= test_direct[
        FP6Format.E3M2,
        8,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct+cap static-z empties",
        [32, 0, 8, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7],
        ctx,
        grid_m_cap=32,
    )
    # Decoupling under deep K: A-scale stride (512) >> grid.y cap (32).
    ok &= test_direct[
        FP6Format.E2M3,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct decouple stride=512",
        [32, 4, 4, 4],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
        ascale_stride_toks=512,
    )
    ok &= test_direct[
        FP6Format.E3M2,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct decouple stride=512",
        [32, 4, 4, 4],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
        ascale_stride_toks=512,
    )
    # up/gate (unfused, N=3072, K=6144): etm<=4096 BM64/BN128 persistent
    ok &= test_persistent[
        FP6Format.E2M3, 4, 3072, 6144, BM=64, BN=128, BK_ELEMS=512, WN=64
    ](
        "M3 up/gate etm<=4096 BM64/BN128",
        [256, 128, 256, 128],
        [0, 1, 2, 3],
        ctx,
    )
    ok &= test_persistent[
        FP6Format.E3M2, 4, 3072, 6144, BM=64, BN=128, BK_ELEMS=512, WN=64
    ](
        "M3 up/gate etm<=4096 BM64/BN128",
        [256, 128, 256, 128],
        [0, 1, 2, 3],
        ctx,
    )
    # up/gate (unfused): else BM128/BN128 persistent
    ok &= test_persistent[
        FP6Format.E2M3, 4, 3072, 6144, BM=128, BN=128, BK_ELEMS=512, WN=64
    ]("M3 up/gate else BM128/BN128", [256, 128, 256, 128], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2, 4, 3072, 6144, BM=128, BN=128, BK_ELEMS=512, WN=64
    ]("M3 up/gate else BM128/BN128", [256, 128, 256, 128], [0, 1, 2, 3], ctx)
    # M3 etm<=2 low-batch-decode band
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        b_cache_policy=SX,
    ]("M3 up etm<=2 decode BN64/wg1 (1 expert)", [1], [0], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        b_cache_policy=SX,
    ]("M3 up etm<=2 decode BN64/wg1 (1 expert)", [1], [0], ctx)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        b_cache_policy=SX,
    ]("M3 up etm<=2 decode BN64/wg1 (2 experts)", [1, 1], [0, 1], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        b_cache_policy=SX,
    ]("M3 up etm<=2 decode BN64/wg1 (2 experts)", [1, 1], [0, 1], ctx)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        b_cache_policy=SX,
    ]("M3 down etm<=2 decode BN64/wg1 (1 expert)", [1], [0], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        b_cache_policy=SX,
    ]("M3 down etm<=2 decode BN64/wg1 (1 expert)", [1], [0], ctx)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        b_cache_policy=SX,
    ]("M3 down etm<=2 decode BN64/wg1 (2 experts)", [1, 1], [0, 1], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        b_cache_policy=SX,
    ]("M3 down etm<=2 decode BN64/wg1 (2 experts)", [1, 1], [0, 1], ctx)
    # ----------------------------------------------------------------- #
    # Pipeline-depth parameterization (deeper-prefetch lever).
    # `pipeline_depth > 2` sizes the B-fragment ring to `pipeline_depth`
    # slots and swaps the b_prefetch steady loop's end-of-iter draining
    # barrier() for the non-draining `s_waitcnt[lgkmcnt=0]` + bare s_barrier.
    # This is the CORRECTNESS gate for the depth plumbing + barrier seam: at
    # depth 3/4 the deeper ring is allocated and the non-draining barrier
    # engages, and the result must still match the per-expert reference.
    #
    # FALSE-NEGATIVE GUARD: this is numerics-only. A build that (regressively)
    # left the draining barrier in place would ALSO pass here — the perf
    # effect (B loads actually staying outstanding) is NOT observable from
    # numerics and MUST be rocprof-verified on GPU when depth is raised in a
    # dispatch band. Passing this test alone does not prove the lever works.
    #
    # M3 decode-band tiles (N=6144; up K=6144, down K=3072), a few active
    # experts + an inactive slot, on both the persistent and direct grids.
    print("---- pipeline_depth > 2 (deeper prefetch) correctness ----")
    # M3 up decode tile, depth 3 and 4 (persistent).
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        pipeline_depth=3,
    ]("M3 up decode depth=3", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        pipeline_depth=3,
    ]("M3 up decode depth=3", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        pipeline_depth=4,
    ]("M3 up decode depth=4", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        pipeline_depth=4,
    ]("M3 up decode depth=4", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    # M3 up decode tile, BN128/WN32 variant, depth 3 (persistent).
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        6144,
        6144,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        pipeline_depth=3,
    ]("M3 up decode BN128 depth=3", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        6144,
        6144,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        pipeline_depth=3,
    ]("M3 up decode BN128 depth=3", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    # M3 down decode tile, depth 3 (persistent).
    ok &= test_persistent[
        FP6Format.E2M3,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        pipeline_depth=3,
    ]("M3 down decode depth=3", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    ok &= test_persistent[
        FP6Format.E3M2,
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        pipeline_depth=3,
    ]("M3 down decode depth=3", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    # Direct grid at depth 3 and 4 — a larger single expert so the steady
    # loop (where the non-draining barrier lives) runs many iterations.
    ok &= test_direct[
        FP6Format.E2M3,
        1,
        6144,
        6144,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        pipeline_depth=3,
    ]("M3 up direct depth=3", [512], [0], ctx)
    ok &= test_direct[
        FP6Format.E3M2,
        1,
        6144,
        6144,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        pipeline_depth=3,
    ]("M3 up direct depth=3", [512], [0], ctx)
    ok &= test_direct[
        FP6Format.E2M3,
        1,
        6144,
        6144,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        pipeline_depth=4,
    ]("M3 up direct depth=4", [512], [0], ctx)
    ok &= test_direct[
        FP6Format.E3M2,
        1,
        6144,
        6144,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        pipeline_depth=4,
    ]("M3 up direct depth=4", [512], [0], ctx)
    assert_true(ok, "one or more MXFP6 preb grouped cases failed")
    print("==== all preb grouped MXFP6 kernel tests passed ====")
