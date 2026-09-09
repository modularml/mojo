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
"""Direct kernel-level tests for `BlockScaledMatmulAMD_PreB` in its MXFP6 modes.

The MXFP6 twin of `test_block_scaled_matmul_amd_preb.mojo`, running that file's ENTIRE
case matrix -- every shape, tile config and launch knob, in the same order --
once per FP6 encoding, through the same 2D grid wrapper with `matrix_format`
set to FP6. Routing coverage lives in
`test_mxfp6_grouped_matmul_amd_kernels.mojo`.

Two FP6 properties this exercises that FP4 never did:

  - The 24-byte lane fragment rides in a 32-byte register carrier, so the
    payload width and the register width are different numbers.
  - `a_lds_swizzle` falls back to identity, because a 24-byte fragment gives a
    non-power-of-two granule count that XOR-16 cannot permute.

`BK_ELEMS` carries over from the MXFP4 cases unchanged: the divisibility
constraint is `K % BK_ELEMS == 0` in both formats, since K_BYTES and BK_BYTES
scale by the same `bits_per_element / 8`.

Usage:
  br test_mxfp6_matmul_amd_preb.mojo.test
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, global_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import CacheOperation
from std.math import ceildiv
from std.random import random_ui64, seed
from std.testing import assert_true
from std.utils import StaticTuple

from layout import (
    Coord,
    Idx,
    TensorLayout,
    TensorEngine,
    TileTensor,
    row_major,
)
from linalg.arch.amd.block_scaled_mma import CDNA4F8F6F4MatrixFormat
from linalg.fp6_utils import (
    FP6Format,
    MXFP6_SF_VECTOR_SIZE,
    decode_fp6_to_f32,
    unpack_fp6_x32,
)
from linalg.matmul.gpu.amd import BlockScaledMatmulAMD_PreB, Shuffler

# A lane feeds the MFMA 24 bytes of FP6, split by the preshuffle into a
# 16-byte and an 8-byte plane.
comptime FP6_LANE_BYTES = 24


def _mfma_format[fmt: FP6Format]() -> CDNA4F8F6F4MatrixFormat:
    comptime if fmt == FP6Format.E2M3:
        return CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3
    return CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2


# ===----------------------------------------------------------------------=== #
# Per-element GPU reference: software decode + scalar accumulate, independent
# of the MFMA path under test (the kernel hands raw bits to the hardware and
# never decodes). `test_fp6_utils` pins this decoder against hand-written
# tables.
#
# Also emits `sum |a*b|` so the tolerance can be tied to the accumulated
# magnitude rather than to a result that may have cancelled to near zero.
#
# Copied from `test_mxfp6_matmul_amd.mojo`, as the MXFP4 preb test copies its
# reference from `test_block_scaled_matmul_amd.mojo` -- these targets take one source
# file each, so there is no shared-helper target to hold it.
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
):
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

    c_ptr[unsafe_offset=m * N + n] = accum
    mag_ptr[unsafe_offset=m * N + n] = magnitude


# ===----------------------------------------------------------------------=== #
# Grid wrapper: drives `BlockScaledMatmulAMD_PreB.run` with block_idx.x = n_tile and
# block_idx.y = m_tile, mirroring the dispatcher's direct-mode launch.
# ===----------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(
            BlockScaledMatmulAMD_PreB[
                BM=BM,
                BN=BN,
                BK_ELEMS=BK_ELEMS,
                WN=WN,
                matrix_format=matrix_format,
                b_prefetch=b_prefetch,
            ].num_threads
        )
    )
)
def _preb_grid_kernel[
    BM: Int,
    BN: Int,
    BK_ELEMS: Int,
    WN: Int,
    matrix_format: CDNA4F8F6F4MatrixFormat,
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
        matrix_format=matrix_format,
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
    fmt: FP6Format,
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
](name: String, ctx: DeviceContext) raises -> Bool:
    """One direct-launch correctness case for the preb kernel in FP6 mode."""
    comptime assert K_static % 128 == 0, "K must be a multiple of 128"
    comptime assert N_static % BN == 0, "N_static must be a multiple of BN"
    comptime assert N_static % 32 == 0, "N must be a multiple of 32 (mn_pack=2)"

    comptime K_BYTES = (K_static * 6) // 8
    comptime scale_K = K_static // MXFP6_SF_VECTOR_SIZE

    # M can be unaligned wrt BM and the scale-cell stride (32). The grid uses
    # ceildiv (OOB rows in the trailing block are clamped by the A V# and the
    # C V#), and the preshuffled A-scale buffer is zero-padded to the next
    # multiple of 32 -- the same shape as per-expert padding in production.
    comptime padded_M = ceildiv(M_static, 32) * 32
    comptime fmt_name = "E2M3" if fmt == FP6Format.E2M3 else "E3M2"

    print(
        "  ",
        fmt_name,
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

    # Scale buffers stay uint8 throughout (E8M0 is byte-equivalent) so
    # `Shuffler.preshuffle_scale_4d` can take them directly; bitcast back to
    # float8_e8m0fnu at the reference / kernel call sites.
    var a_h = ctx.enqueue_create_host_buffer[.uint8](M_static * K_BYTES)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](N_static * K_BYTES)
    var sfa_h = ctx.enqueue_create_host_buffer[.uint8](M_static * scale_K)
    var sfb_h = ctx.enqueue_create_host_buffer[.uint8](N_static * scale_K)
    var sfa_pre_h = ctx.enqueue_create_host_buffer[.uint8](padded_M * scale_K)
    var sfb_pre_h = ctx.enqueue_create_host_buffer[.uint8](N_static * scale_K)
    ctx.synchronize()

    # Every 6-bit code is a finite number in both encodings, so random bytes
    # need no NaN/Inf filtering.
    for i in range(M_static * K_BYTES):
        a_h[i] = UInt8(random_ui64(0, 255))
    for i in range(N_static * K_BYTES):
        b_h[i] = UInt8(random_ui64(0, 255))
    # Clamp E8M0 to [125..129] = magnitudes [0.25..4] to keep f32 in range.
    for i in range(M_static * scale_K):
        sfa_h[i] = UInt8(random_ui64(125, 129))
    for i in range(N_static * scale_K):
        sfb_h[i] = UInt8(random_ui64(125, 129))

    # The scale path is format-independent: one E8M0 byte per 32 elements per
    # lane, identical for every f8f6f4 format.
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

    var a_d = ctx.enqueue_create_buffer[.uint8](M_static * K_BYTES)
    var b_d = ctx.enqueue_create_buffer[.uint8](N_static * K_BYTES)
    var b_pre_d = ctx.enqueue_create_buffer[.uint8](N_static * K_BYTES)
    var sfa_d = ctx.enqueue_create_buffer[.uint8](M_static * scale_K)
    var sfb_d = ctx.enqueue_create_buffer[.uint8](N_static * scale_K)
    var sfa_pre_d = ctx.enqueue_create_buffer[.uint8](padded_M * scale_K)
    var sfb_pre_d = ctx.enqueue_create_buffer[.uint8](N_static * scale_K)
    var c_d = ctx.enqueue_create_buffer[.float32](M_static * N_static)
    var c_ref_d = ctx.enqueue_create_buffer[.float32](M_static * N_static)
    var mag_d = ctx.enqueue_create_buffer[.float32](M_static * N_static)
    c_d.enqueue_fill(Float32(0.0))
    c_ref_d.enqueue_fill(Float32(0.0))
    mag_d.enqueue_fill(Float32(0.0))

    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(sfa_d, sfa_h)
    ctx.enqueue_copy(sfb_d, sfb_h)
    ctx.enqueue_copy(sfa_pre_d, sfa_pre_h)
    ctx.enqueue_copy(sfb_pre_d, sfb_pre_h)

    # GPU preshuffle B -> b_pre_d through the plane-split kernel; the tuned
    # 16-byte atom path cannot express a 24-byte fragment.
    var b_raw_tt = TileTensor(b_d, row_major[1, N_static, K_BYTES]()).as_immut()
    var b_pre_dst_tt = TileTensor(b_pre_d, row_major[1, N_static, K_BYTES]())
    Shuffler[1].preshuffle_b_planes[
        N=N_static, K_BYTES=K_BYTES, lane_bytes=FP6_LANE_BYTES
    ](b_raw_tt, b_pre_dst_tt, ctx)

    comptime BLOCK_DIM = 16
    ctx.enqueue_function[_mxfp6_matmul_ref[fmt]](
        a_d.unsafe_ptr(),
        b_d.unsafe_ptr(),
        sfa_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        sfb_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        c_ref_d.unsafe_ptr(),
        mag_d.unsafe_ptr(),
        Int32(M_static),
        Int32(N_static),
        Int32(K_static),
        grid_dim=(ceildiv(M_static, BLOCK_DIM), ceildiv(N_static, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    var a_tt = TileTensor(
        a_d, row_major(Coord(M_static, Idx[K_BYTES]))
    ).as_immut()
    var b_pre_tt = TileTensor(
        b_pre_d, row_major[1, N_static * K_BYTES]()
    ).as_immut()
    # Scales: the preshuffled buffers wrapped in a row-major layout -- the
    # kernel reads the underlying bytes through `PreshuffledScaleLoader`, so
    # the TileTensor layout is unused.
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
        _mfma_format[fmt](),
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
            matrix_format=_mfma_format[fmt](),
            b_prefetch=b_prefetch,
        ].num_threads,
    )
    ctx.synchronize()

    var c_h = ctx.enqueue_create_host_buffer[.float32](M_static * N_static)
    var c_ref_h = ctx.enqueue_create_host_buffer[.float32](M_static * N_static)
    var mag_h = ctx.enqueue_create_host_buffer[.float32](M_static * N_static)
    ctx.enqueue_copy(c_h, c_d)
    ctx.enqueue_copy(c_ref_h, c_ref_d)
    ctx.enqueue_copy(mag_h, mag_d)
    ctx.synchronize()

    # float32 accumulation error grows with the number of summed terms, so the
    # bound is K-scaled and applied to sum |a*b| -- not to a result that may
    # have cancelled to near zero, where any relative bound is meaningless.
    comptime ULP_F32 = 5.9604644775390625e-8
    var rel_tol = Float64(16 * K_static) * ULP_F32

    var mismatches = 0
    var saw_nonzero = False
    for i in range(M_static * N_static):
        var want = Float64(c_ref_h[i])
        var got = Float64(c_h[i])
        if want != Float64(0.0):
            saw_nonzero = True
        if abs(got - want) > rel_tol * Float64(mag_h[i]):
            if mismatches < 3:
                print(
                    "      [",
                    i // N_static,
                    ",",
                    i % N_static,
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
    _ = sfa_d^
    _ = sfb_d^
    _ = sfa_pre_d^
    _ = sfb_pre_d^
    _ = c_d^
    _ = c_ref_d^
    _ = mag_d^

    # An all-zero reference would make the comparison vacuous: it would pass
    # against a kernel that wrote nothing.
    if not saw_nonzero:
        print("    FAIL  reference is all zero; comparison proves nothing")
        return False
    if mismatches > 0:
        print("    FAIL ", mismatches, " wrong of ", M_static * N_static)
        var row_bad = List[Int]()
        var col_bad = List[Int]()
        for _ in range(BM):
            row_bad.append(0)
        for _ in range(BN):
            col_bad.append(0)
        for r in range(M_static):
            for c in range(N_static):
                var i = r * N_static + c
                if abs(
                    Float64(c_h[i]) - Float64(c_ref_h[i])
                ) > rel_tol * Float64(mag_h[i]):
                    row_bad[r % BM] += 1
                    col_bad[c % BN] += 1
        print("      wrong-count by row % BM:")
        for r in range(BM):
            if row_bad[r] > 0:
                print("        r%BM=", r, "->", row_bad[r])
        print("      wrong-count by col % BN:")
        for c in range(BN):
            if col_bad[c] > 0:
                print("        c%BN=", c, "->", col_bad[c])
        return False
    print("    PASS")
    return True


def main() raises:
    seed(0)
    var ctx = DeviceContext()

    print(
        "===> BlockScaledMatmulAMD_PreB in FP6 mode — direct kernel correctness"
    )

    var ok = True

    # flydsl stage1 champion config: tile_m=32/n=128/k=256, 4 waves (WN=32),
    # b_nt=2 (STREAMING), 2-stage pipeline, K=7168 (28 K-steps). Set
    # DUMP_ASM=True here to dump the kernel ISA for comparison vs
    # stage1_champion.s.
    ok &= _test_case[
        FP6Format.E2M3,
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
    ok &= _test_case[
        FP6Format.E3M2,
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
    ok &= _test_case[
        FP6Format.E2M3, 256, 256, 256, BM=64, BN=64, WN=64, BK_ELEMS=256
    ]("single warp, no-prefetch", ctx)
    ok &= _test_case[
        FP6Format.E3M2, 256, 256, 256, BM=64, BN=64, WN=64, BK_ELEMS=256
    ]("single warp, no-prefetch", ctx)
    ok &= _test_case[
        FP6Format.E2M3,
        256,
        256,
        256,
        BM=64,
        BN=64,
        WN=64,
        BK_ELEMS=256,
        b_prefetch=True,
    ]("single warp, b_prefetch=True", ctx)
    ok &= _test_case[
        FP6Format.E3M2,
        256,
        256,
        256,
        BM=64,
        BN=64,
        WN=64,
        BK_ELEMS=256,
        b_prefetch=True,
    ]("single warp, b_prefetch=True", ctx)
    ok &= _test_case[
        FP6Format.E2M3, 234, 1024, 512, BM=64, BN=128, WN=64, BK_ELEMS=256
    ]("OOB test on M, no-prefetch", ctx)
    ok &= _test_case[
        FP6Format.E3M2, 234, 1024, 512, BM=64, BN=128, WN=64, BK_ELEMS=256
    ]("OOB test on M, no-prefetch", ctx)
    ok &= _test_case[
        FP6Format.E2M3,
        234,
        1024,
        512,
        BM=64,
        BN=128,
        WN=64,
        BK_ELEMS=256,
        b_prefetch=True,
    ]("OOB test on M, b_prefetch=True", ctx)
    ok &= _test_case[
        FP6Format.E3M2,
        234,
        1024,
        512,
        BM=64,
        BN=128,
        WN=64,
        BK_ELEMS=256,
        b_prefetch=True,
    ]("OOB test on M, b_prefetch=True", ctx)
    ok &= _test_case[
        FP6Format.E2M3, 256, 1024, 512, BM=16, BN=64, WN=16, BK_ELEMS=256
    ]("size 16 warp tile, no-prefetch", ctx)
    ok &= _test_case[
        FP6Format.E3M2, 256, 1024, 512, BM=16, BN=64, WN=16, BK_ELEMS=256
    ]("size 16 warp tile, no-prefetch", ctx)
    ok &= _test_case[
        FP6Format.E2M3,
        256,
        1024,
        512,
        BM=16,
        BN=64,
        WN=16,
        BK_ELEMS=256,
        b_prefetch=True,
    ]("size 16 warp tile, b_prefetch=True", ctx)
    ok &= _test_case[
        FP6Format.E3M2,
        256,
        1024,
        512,
        BM=16,
        BN=64,
        WN=16,
        BK_ELEMS=256,
        b_prefetch=True,
    ]("size 16 warp tile, b_prefetch=True", ctx)
    ok &= _test_case[
        FP6Format.E2M3, 24, 1024, 2048, BM=64, BN=256, WN=64, BK_ELEMS=256
    ]("Testing BM > M", ctx)
    ok &= _test_case[
        FP6Format.E3M2, 24, 1024, 2048, BM=64, BN=256, WN=64, BK_ELEMS=256
    ]("Testing BM > M", ctx)
    ok &= _test_case[
        FP6Format.E2M3,
        24,
        1024,
        2048,
        BM=64,
        BN=256,
        WN=64,
        BK_ELEMS=256,
        b_prefetch=True,
    ]("Testing BM > M, b_prefetch=True", ctx)
    ok &= _test_case[
        FP6Format.E3M2,
        24,
        1024,
        2048,
        BM=64,
        BN=256,
        WN=64,
        BK_ELEMS=256,
        b_prefetch=True,
    ]("Testing BM > M, b_prefetch=True", ctx)
    ok &= _test_case[
        FP6Format.E2M3, 1001, 1024, 2048, BM=64, BN=256, WN=64, BK_ELEMS=256
    ]("Testing large M", ctx)
    ok &= _test_case[
        FP6Format.E3M2, 1001, 1024, 2048, BM=64, BN=256, WN=64, BK_ELEMS=256
    ]("Testing large M", ctx)
    ok &= _test_case[
        FP6Format.E2M3,
        1001,
        1024,
        2048,
        BM=64,
        BN=256,
        WN=64,
        BK_ELEMS=256,
        b_prefetch=True,
    ]("Testing large M, b_prefetch=True", ctx)
    ok &= _test_case[
        FP6Format.E3M2,
        1001,
        1024,
        2048,
        BM=64,
        BN=256,
        WN=64,
        BK_ELEMS=256,
        b_prefetch=True,
    ]("Testing large M, b_prefetch=True", ctx)
    # Default production tile (BK_ELEMS=512, num_k_mmas=4).
    ok &= _test_case[
        FP6Format.E2M3, 256, 1024, 2048, BM=64, BN=128, WN=64, BK_ELEMS=512
    ]("default prod tile", ctx)
    ok &= _test_case[
        FP6Format.E3M2, 256, 1024, 2048, BM=64, BN=128, WN=64, BK_ELEMS=512
    ]("default prod tile", ctx)
    ok &= _test_case[
        FP6Format.E2M3,
        256,
        1024,
        2048,
        BM=64,
        BN=128,
        WN=64,
        BK_ELEMS=512,
        b_prefetch=True,
    ]("default prod tile, b_prefetch=True", ctx)
    ok &= _test_case[
        FP6Format.E3M2,
        256,
        1024,
        2048,
        BM=64,
        BN=128,
        WN=64,
        BK_ELEMS=512,
        b_prefetch=True,
    ]("default prod tile, b_prefetch=True", ctx)
    # WN=16 with partial M (decode shape).
    ok &= _test_case[
        FP6Format.E2M3, 3, 1024, 2048, BM=16, BN=64, WN=16, BK_ELEMS=256
    ]("decode-shape, WN=16", ctx)
    ok &= _test_case[
        FP6Format.E3M2, 3, 1024, 2048, BM=16, BN=64, WN=16, BK_ELEMS=256
    ]("decode-shape, WN=16", ctx)
    ok &= _test_case[
        FP6Format.E2M3,
        3,
        1024,
        2048,
        BM=16,
        BN=64,
        WN=16,
        BK_ELEMS=256,
        b_prefetch=True,
    ]("decode-shape, WN=16, b_prefetch=True", ctx)
    ok &= _test_case[
        FP6Format.E3M2,
        3,
        1024,
        2048,
        BM=16,
        BN=64,
        WN=16,
        BK_ELEMS=256,
        b_prefetch=True,
    ]("decode-shape, WN=16, b_prefetch=True", ctx)
    # WN=16 only — N-side shrui without M-side (BM aligned, WN=16).
    ok &= _test_case[
        FP6Format.E2M3, 64, 64, 512, BM=64, BN=64, WN=16, BK_ELEMS=256
    ]("WN=16 only", ctx)
    ok &= _test_case[
        FP6Format.E3M2, 64, 64, 512, BM=64, BN=64, WN=16, BK_ELEMS=256
    ]("WN=16 only", ctx)
    ok &= _test_case[
        FP6Format.E2M3,
        64,
        64,
        512,
        BM=64,
        BN=64,
        WN=16,
        BK_ELEMS=256,
        b_prefetch=True,
    ]("WN=16 only, b_prefetch=True", ctx)
    ok &= _test_case[
        FP6Format.E3M2,
        64,
        64,
        512,
        BM=64,
        BN=64,
        WN=16,
        BK_ELEMS=256,
        b_prefetch=True,
    ]("WN=16 only, b_prefetch=True", ctx)
    # M=1 decode — smallest possible M with both tile-size variants.
    ok &= _test_case[
        FP6Format.E2M3, 1, 128, 512, BM=16, BN=64, WN=16, BK_ELEMS=256
    ]("M=1 decode, BM=16", ctx)
    ok &= _test_case[
        FP6Format.E3M2, 1, 128, 512, BM=16, BN=64, WN=16, BK_ELEMS=256
    ]("M=1 decode, BM=16", ctx)
    ok &= _test_case[
        FP6Format.E2M3, 1, 1024, 2048, BM=64, BN=128, WN=64, BK_ELEMS=256
    ]("M=1 decode, BM=64", ctx)
    ok &= _test_case[
        FP6Format.E3M2, 1, 1024, 2048, BM=64, BN=128, WN=64, BK_ELEMS=256
    ]("M=1 decode, BM=64", ctx)
    # dram_to_lds=True — direct DRAM->LDS DMA A staging (flydsl
    # dram_to_lds). Same shapes, both prefetch modes.
    ok &= _test_case[
        FP6Format.E2M3,
        256,
        256,
        256,
        BM=64,
        BN=64,
        WN=64,
        BK_ELEMS=256,
        dram_to_lds=True,
    ]("single warp, async", ctx)
    ok &= _test_case[
        FP6Format.E3M2,
        256,
        256,
        256,
        BM=64,
        BN=64,
        WN=64,
        BK_ELEMS=256,
        dram_to_lds=True,
    ]("single warp, async", ctx)
    ok &= _test_case[
        FP6Format.E2M3,
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
    ok &= _test_case[
        FP6Format.E3M2,
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
    ok &= _test_case[
        FP6Format.E2M3,
        256,
        1024,
        2048,
        BM=64,
        BN=128,
        WN=64,
        BK_ELEMS=512,
        dram_to_lds=True,
    ]("default prod tile, async", ctx)
    ok &= _test_case[
        FP6Format.E3M2,
        256,
        1024,
        2048,
        BM=64,
        BN=128,
        WN=64,
        BK_ELEMS=512,
        dram_to_lds=True,
    ]("default prod tile, async", ctx)
    ok &= _test_case[
        FP6Format.E2M3,
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
    ok &= _test_case[
        FP6Format.E3M2,
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
    ok &= _test_case[
        FP6Format.E2M3,
        234,
        1024,
        512,
        BM=64,
        BN=128,
        WN=64,
        BK_ELEMS=256,
        dram_to_lds=True,
    ]("OOB test on M, async", ctx)
    ok &= _test_case[
        FP6Format.E3M2,
        234,
        1024,
        512,
        BM=64,
        BN=128,
        WN=64,
        BK_ELEMS=256,
        dram_to_lds=True,
    ]("OOB test on M, async", ctx)
    ok &= _test_case[
        FP6Format.E2M3,
        3,
        1024,
        2048,
        BM=16,
        BN=64,
        WN=16,
        BK_ELEMS=256,
        dram_to_lds=True,
    ]("decode-shape, WN=16, async", ctx)
    ok &= _test_case[
        FP6Format.E3M2,
        3,
        1024,
        2048,
        BM=16,
        BN=64,
        WN=16,
        BK_ELEMS=256,
        dram_to_lds=True,
    ]("decode-shape, WN=16, async", ctx)
    # cluster_drain_sched=True (b_prefetch only) — per-cluster setprio + partial
    # vmcnt staircase. Covers default prod tile (num_k_mmas=4), the smaller
    # decode tile (num_k_mmas=2), WN=16, and a non-default mfma_cluster.
    ok &= _test_case[
        FP6Format.E2M3,
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
    ok &= _test_case[
        FP6Format.E3M2,
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
    ok &= _test_case[
        FP6Format.E2M3,
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
    ok &= _test_case[
        FP6Format.E3M2,
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
    ok &= _test_case[
        FP6Format.E2M3,
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
    ok &= _test_case[
        FP6Format.E3M2,
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
    ok &= _test_case[
        FP6Format.E2M3,
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
    ok &= _test_case[
        FP6Format.E3M2,
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
    ok &= _test_case[
        FP6Format.E2M3,
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
    ok &= _test_case[
        FP6Format.E3M2,
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

    assert_true(ok, "one or more MXFP6 dense preb cases failed")
    print("ALL PASS")
