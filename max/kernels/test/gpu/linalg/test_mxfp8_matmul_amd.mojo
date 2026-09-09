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

"""Correctness tests for the DENSE row-major `BlockScaledMatmulAMD` at MXFP8.

MXFP8 sibling of `test_mxfp4_matmul_amd.mojo` (`lane_bytes=32`), and the
row-major counterpart to `test_mxfp8_matmul_amd_preb.mojo`. One byte per
element, so `K_BYTES == K` and a lane's operand is TWO 16-byte halves
`K_HALF_STRIDE` apart in K rather than 32 contiguous bytes -- the fragment
loaders tile K per half so the geometry stays identical to MXFP4.

The reference is a per-element dequant + scalar accumulate through
`.float8_e4m3fn`: no MFMA and no code shared with the kernel, so a
fragment-layout error cannot cancel out. `data-only` and `scales-only` phases
split a data-layout bug from a scale-index bug, which uniform fills cannot.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.memory import bitcast
from std.random import random_ui64, seed
from std.testing import assert_almost_equal, assert_equal

from layout import TileTensor
from layout.tile_layout import row_major
from linalg.fp4_utils import MXFP8_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd.block_scaled_matmul_amd import (
    BlockScaledMatmulAMD,
    _launch_block_scaled_split_k,
    _sk_perf_clamped_max_m,
    block_scaled_matmul_amd,
)

from linalg.arch.amd.block_scaled_mma import CDNA4F8F6F4MatrixFormat

comptime FP8_LANE_BYTES = 32


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
    """Per-element MXFP8 reference: dequant E4M3 x E8M0, accumulate in f32."""
    var M = Int(M_arg)
    var N = Int(N_arg)
    var K = Int(K_arg)

    var m = global_idx.x
    var n = global_idx.y
    if m >= M or n >= N:
        return

    var k_groups = K // MXFP8_SF_VECTOR_SIZE
    var am_scales = a_scales_ptr + m * k_groups
    var bn_scales = b_scales_ptr + n * k_groups
    # One byte per element at MXFP8, so the row stride is K (not K // 2).
    var am = (a_ptr + m * K).bitcast[Float8_e4m3fn]()
    var bn = (b_ptr + n * K).bitcast[Float8_e4m3fn]()

    var accum = Float32(0)
    for ko in range(k_groups):
        var a_scale = am_scales[ko].cast[.float32]()
        var b_scale = bn_scales[ko].cast[.float32]()
        # E8M0 scales are exact powers of two, so hoisting them is bit-exact.
        var part = Float32(0)
        for ki in range(MXFP8_SF_VECTOR_SIZE):
            part += am[ki].cast[.float32]() * bn[ki].cast[.float32]()
        accum += part * a_scale * b_scale
        am += MXFP8_SF_VECTOR_SIZE
        bn += MXFP8_SF_VECTOR_SIZE

    c_ptr[m * N + n] = accum


def _e4m3_byte(f: Float32) -> UInt8:
    """Byte encoding of `f` as E4M3."""
    return bitcast[.uint8, 1](Float8_e4m3fn(f.cast[.float8_e4m3fn]()))[0]


def _rand_e4m3_byte() -> UInt8:
    """Random finite E4M3 byte in [-1, 1].

    A uniform byte fill is unusable: 0x7F/0xFF are NaN and 0x7E is 448.
    """
    return _e4m3_byte(Float32(Int(random_ui64(0, 2000))) / 1000.0 - 1.0)


def _rand_positive_e4m3_byte() -> UInt8:
    """Random finite E4M3 byte in (0, 1].

    Sign-free products add, so the reference stays ~K*eps. Still varies
    along K, unlike const-fill.
    """
    return _e4m3_byte(Float32(Int(random_ui64(1, 1000))) / 1000.0)


def _operand_byte[ones_data: Bool, positive_data: Bool]() -> UInt8:
    """One A/B byte in the fill mode the case asked for.

    Untaken branches draw no RNG, so appending a const-fill case cannot
    re-roll data of cases above it.
    """
    if ones_data:
        return _e4m3_byte(Float32(1.0))
    if positive_data:
        return _rand_positive_e4m3_byte()
    return _rand_e4m3_byte()


def _test_case[
    M_static: Int,
    N_static: Int,
    K_static: Int,
    BM: Int = 128,
    BN: Int = 128,
    BK_ELEMS: Int = 128,
    WM: Int = 64,
    WN: Int = 64,
    num_stages: Int = 1,
    unit_scales: Bool = False,
    ones_data: Bool = False,
    positive_data: Bool = False,
    out_dtype: DType = DType.float32,
](name: String, ctx: DeviceContext) raises:
    comptime assert (
        K_static % MXFP8_SF_VECTOR_SIZE == 0
    ), "K must be a multiple of 32"
    comptime assert not (
        ones_data and positive_data
    ), "ones_data and positive_data are alternative fills; pick one"
    # Sign-free operands leave the reference at ~K*eps, so the oracle tightens.
    comptime rtol = 5e-3 if positive_data else 1e-2

    # MXFP8: one byte per element.
    comptime K_BYTES = K_static
    comptime K_SCALES = K_static // MXFP8_SF_VECTOR_SIZE

    print("  ", name, " ", M_static, "x", N_static, "x", K_static)

    var a_h = ctx.enqueue_create_host_buffer[.uint8](M_static * K_BYTES)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](N_static * K_BYTES)
    var sfa_h = ctx.enqueue_create_host_buffer[.uint8](M_static * K_SCALES)
    var sfb_h = ctx.enqueue_create_host_buffer[.uint8](N_static * K_SCALES)
    ctx.synchronize()

    for i in range(M_static * K_BYTES):
        a_h[i] = _operand_byte[ones_data, positive_data]()
    for i in range(N_static * K_BYTES):
        b_h[i] = _operand_byte[ones_data, positive_data]()
    # E8M0: 127 == 2^0. Vary the exponent so a mis-indexed scale block reads a
    # genuinely different power of two.
    for i in range(M_static * K_SCALES):
        sfa_h[i] = 127 if unit_scales else UInt8(Int(random_ui64(124, 130)))
    for i in range(N_static * K_SCALES):
        sfb_h[i] = 127 if unit_scales else UInt8(Int(random_ui64(124, 130)))

    var a_d = ctx.enqueue_create_buffer[.uint8](M_static * K_BYTES)
    var b_d = ctx.enqueue_create_buffer[.uint8](N_static * K_BYTES)
    var sfa_d = ctx.enqueue_create_buffer[.uint8](M_static * K_SCALES)
    var sfb_d = ctx.enqueue_create_buffer[.uint8](N_static * K_SCALES)
    var c_d = ctx.enqueue_create_buffer[out_dtype](M_static * N_static)
    var c_ref_d = ctx.enqueue_create_buffer[.float32](M_static * N_static)
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(sfa_d, sfa_h)
    ctx.enqueue_copy(sfb_d, sfb_h)

    var c_tt = TileTensor(c_d, row_major[M_static, N_static]())
    var a_tt = TileTensor(a_d, row_major[M_static, K_BYTES]()).as_immut()
    var b_tt = TileTensor(b_d, row_major[N_static, K_BYTES]()).as_immut()
    var sfa_tt = TileTensor(
        sfa_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu](),
        row_major[M_static, K_SCALES](),
    ).as_immut()
    var sfb_tt = TileTensor(
        sfb_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu](),
        row_major[N_static, K_SCALES](),
    ).as_immut()

    comptime Kernel = BlockScaledMatmulAMD[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        WM=WM,
        WN=WN,
        num_stages=num_stages,
        matrix_format=CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3,
    ]
    comptime kernel = Kernel.run[
        out_dtype,
        type_of(c_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(sfa_tt).LayoutType,
        type_of(sfb_tt).LayoutType,
        type_of(c_tt).Engine,
        type_of(a_tt).Engine,
        type_of(b_tt).Engine,
        type_of(sfa_tt).Engine,
        type_of(sfb_tt).Engine,
    ]
    ctx.enqueue_function[kernel](
        c_tt,
        a_tt,
        b_tt,
        sfa_tt,
        sfb_tt,
        grid_dim=(ceildiv(N_static, BN), ceildiv(M_static, BM)),
        block_dim=Kernel.num_threads,
    )

    comptime BLOCK_DIM = 32
    ctx.enqueue_function[block_scaled_matmul_fp8_ref](
        a_d.unsafe_ptr().as_imm(),
        b_d.unsafe_ptr().as_imm(),
        sfa_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu]().as_imm(),
        sfb_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu]().as_imm(),
        c_ref_d.unsafe_ptr(),
        Int32(M_static),
        Int32(N_static),
        Int32(K_static),
        grid_dim=(ceildiv(M_static, BLOCK_DIM), ceildiv(N_static, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    var c_h = ctx.enqueue_create_host_buffer[out_dtype](M_static * N_static)
    var c_ref_h = ctx.enqueue_create_host_buffer[.float32](M_static * N_static)
    ctx.enqueue_copy(c_h, c_d)
    ctx.enqueue_copy(c_ref_h, c_ref_d)
    ctx.synchronize()

    for i in range(M_static * N_static):
        assert_almost_equal(
            c_h[i].cast[DType.float32](),
            c_ref_h[i],
            atol=1e-2,
            rtol=rtol,
            msg=String("mismatch at ", i),
        )
    print("     OK")


def test_mxfp8_matmul_split_k[
    M_static: Int,
    N_static: Int,
    K_static: Int,
    num_splits: Int,
    BM: Int = 64,
    BN: Int = 128,
    BK_ELEMS: Int = 256,
    WM: Int = 64,
    WN: Int = 32,
    unit_scales: Bool = False,
](ctx: DeviceContext) raises:
    """MXFP8 sibling of `test_mxfp4_matmul_split_k`.

    Launches `_launch_block_scaled_split_k[lane_bytes=32]` (workspace + reduce
    path) for the given `num_splits` and verifies against the same
    per-element reference used by `_test_case`.

    `unit_scales` pins every E8M0 scale to 2^0. Operands stay random so a
    K-band error still fails; the tighter spread keeps the reference in atol.
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
        " unit_scales=",
        unit_scales,
        "]",
    )

    comptime assert (
        K_static % MXFP8_SF_VECTOR_SIZE == 0
    ), "K must be a multiple of 32"
    # MXFP8: one byte per element.
    comptime K_BYTES = K_static
    comptime K_SCALES = K_static // MXFP8_SF_VECTOR_SIZE

    var a_h = ctx.enqueue_create_host_buffer[.uint8](M_static * K_BYTES)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](N_static * K_BYTES)
    var sfa_h = ctx.enqueue_create_host_buffer[.uint8](M_static * K_SCALES)
    var sfb_h = ctx.enqueue_create_host_buffer[.uint8](N_static * K_SCALES)
    ctx.synchronize()

    for i in range(M_static * K_BYTES):
        a_h[i] = _rand_e4m3_byte()
    for i in range(N_static * K_BYTES):
        b_h[i] = _rand_e4m3_byte()
    for i in range(M_static * K_SCALES):
        sfa_h[i] = 127 if unit_scales else UInt8(Int(random_ui64(124, 130)))
    for i in range(N_static * K_SCALES):
        sfb_h[i] = 127 if unit_scales else UInt8(Int(random_ui64(124, 130)))

    var a_d = ctx.enqueue_create_buffer[.uint8](M_static * K_BYTES)
    var b_d = ctx.enqueue_create_buffer[.uint8](N_static * K_BYTES)
    var sfa_d = ctx.enqueue_create_buffer[.uint8](M_static * K_SCALES)
    var sfb_d = ctx.enqueue_create_buffer[.uint8](N_static * K_SCALES)
    var c_d = ctx.enqueue_create_buffer[.float32](M_static * N_static)
    var c_ref_d = ctx.enqueue_create_buffer[.float32](M_static * N_static)
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(sfa_d, sfa_h)
    ctx.enqueue_copy(sfb_d, sfb_h)

    var c_tt = TileTensor(c_d, row_major[M_static, N_static]())
    var a_tt = TileTensor(a_d, row_major[M_static, K_BYTES]()).as_immut()
    var b_tt = TileTensor(b_d, row_major[N_static, K_BYTES]()).as_immut()
    var sfa_tt = TileTensor(
        sfa_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu](),
        row_major[M_static, K_SCALES](),
    ).as_immut()
    var sfb_tt = TileTensor(
        sfb_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu](),
        row_major[N_static, K_SCALES](),
    ).as_immut()

    _launch_block_scaled_split_k[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        WM=WM,
        WN=WN,
        num_splits=num_splits,
        matrix_format=CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3,
    ](c_tt, a_tt, b_tt, sfa_tt, sfb_tt, M_static, ctx)

    comptime BLOCK_DIM = 32
    ctx.enqueue_function[block_scaled_matmul_fp8_ref](
        a_d.unsafe_ptr().as_imm(),
        b_d.unsafe_ptr().as_imm(),
        sfa_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu]().as_imm(),
        sfb_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu]().as_imm(),
        c_ref_d.unsafe_ptr(),
        Int32(M_static),
        Int32(N_static),
        Int32(K_static),
        grid_dim=(ceildiv(M_static, BLOCK_DIM), ceildiv(N_static, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    var c_h = ctx.enqueue_create_host_buffer[.float32](M_static * N_static)
    var c_ref_h = ctx.enqueue_create_host_buffer[.float32](M_static * N_static)
    ctx.enqueue_copy(c_h, c_d)
    ctx.enqueue_copy(c_ref_h, c_ref_d)
    ctx.synchronize()

    for i in range(M_static * N_static):
        assert_almost_equal(
            c_h[i],
            c_ref_h[i],
            atol=1e-2,
            rtol=1e-2,
            msg=String("mismatch at ", i),
        )
    print("     OK")


def _test_dispatch[
    M_static: Int,
    N_static: Int,
    K_static: Int,
    unit_scales: Bool = False,
    ones_data: Bool = False,
    positive_data: Bool = False,
    out_dtype: DType = DType.float32,
](name: String, ctx: DeviceContext) raises:
    """Drives the public dispatcher rather than a hand-picked tile shape.

    `block_scaled_matmul_amd` picks BM/BN/BK from M and K, so this is the
    only way to cover its BK-bucket gates. A gate that admits a BK the kernel
    cannot tile fails the comptime assert in `BlockScaledMatmulAMD.run` --
    i.e. this case fails to BUILD rather than producing a wrong answer.

    `out_dtype` is a kernel parameter, not just an epilogue: f32 and bf16
    output are different code objects.
    """
    comptime assert (
        K_static % MXFP8_SF_VECTOR_SIZE == 0
    ), "K must be a multiple of 32"
    comptime assert not (
        ones_data and positive_data
    ), "ones_data and positive_data are alternative fills; pick one"
    # See `_test_case`: sign-free operands earn a tighter oracle.
    comptime rtol = 5e-3 if positive_data else 1e-2
    comptime K_BYTES = K_static
    comptime K_SCALES = K_static // MXFP8_SF_VECTOR_SIZE

    print("  ", name, " ", M_static, "x", N_static, "x", K_static)

    var a_h = ctx.enqueue_create_host_buffer[.uint8](M_static * K_BYTES)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](N_static * K_BYTES)
    var sfa_h = ctx.enqueue_create_host_buffer[.uint8](M_static * K_SCALES)
    var sfb_h = ctx.enqueue_create_host_buffer[.uint8](N_static * K_SCALES)
    ctx.synchronize()

    for i in range(M_static * K_BYTES):
        a_h[i] = _operand_byte[ones_data, positive_data]()
    for i in range(N_static * K_BYTES):
        b_h[i] = _operand_byte[ones_data, positive_data]()
    for i in range(M_static * K_SCALES):
        sfa_h[i] = 127 if unit_scales else UInt8(Int(random_ui64(124, 130)))
    for i in range(N_static * K_SCALES):
        sfb_h[i] = 127 if unit_scales else UInt8(Int(random_ui64(124, 130)))

    var a_d = ctx.enqueue_create_buffer[.uint8](M_static * K_BYTES)
    var b_d = ctx.enqueue_create_buffer[.uint8](N_static * K_BYTES)
    var sfa_d = ctx.enqueue_create_buffer[.uint8](M_static * K_SCALES)
    var sfb_d = ctx.enqueue_create_buffer[.uint8](N_static * K_SCALES)
    var c_d = ctx.enqueue_create_buffer[out_dtype](M_static * N_static)
    var c_ref_d = ctx.enqueue_create_buffer[.float32](M_static * N_static)
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(sfa_d, sfa_h)
    ctx.enqueue_copy(sfb_d, sfb_h)

    var c_tt = TileTensor(c_d, row_major[M_static, N_static]())
    var a_tt = TileTensor(a_d, row_major[M_static, K_BYTES]()).as_immut()
    var b_tt = TileTensor(b_d, row_major[N_static, K_BYTES]()).as_immut()
    var sfa_tt = TileTensor(
        sfa_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu](),
        row_major[M_static, K_SCALES](),
    ).as_immut()
    var sfb_tt = TileTensor(
        sfb_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu](),
        row_major[N_static, K_SCALES](),
    ).as_immut()

    block_scaled_matmul_amd[lane_bytes=FP8_LANE_BYTES](
        c_tt, a_tt, b_tt, sfa_tt, sfb_tt, ctx
    )

    comptime BLOCK_DIM = 32
    ctx.enqueue_function[block_scaled_matmul_fp8_ref](
        a_d.unsafe_ptr().as_imm(),
        b_d.unsafe_ptr().as_imm(),
        sfa_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu]().as_imm(),
        sfb_d.unsafe_ptr().unsafe_bitcast[Float8_e8m0fnu]().as_imm(),
        c_ref_d.unsafe_ptr(),
        Int32(M_static),
        Int32(N_static),
        Int32(K_static),
        grid_dim=(ceildiv(M_static, BLOCK_DIM), ceildiv(N_static, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    var c_h = ctx.enqueue_create_host_buffer[out_dtype](M_static * N_static)
    var c_ref_h = ctx.enqueue_create_host_buffer[.float32](M_static * N_static)
    ctx.enqueue_copy(c_h, c_d)
    ctx.enqueue_copy(c_ref_h, c_ref_d)
    ctx.synchronize()

    for i in range(M_static * N_static):
        assert_almost_equal(
            c_h[i].cast[DType.float32](),
            c_ref_h[i],
            atol=1e-2,
            rtol=rtol,
            msg=String("mismatch at ", i),
        )
    print("     OK")


def _test_route_policy() raises:
    """Pins the split-K M ceiling at the measured o_proj crossover.

    Both routes agree, so a wrong ceiling is a silent perf regression.
    """
    print("\n--- split-K route policy (performance pin) ---")

    # o_proj key: workspace 1365, clamped to the measured crossover 320.
    assert_equal(
        _sk_perf_clamped_max_m[FP8_LANE_BYTES, 6144, 2048, 1365](), 320
    )

    # Clamp is keyed to that (format, N, K); other shapes keep the byte budget.
    assert_equal(
        _sk_perf_clamped_max_m[FP8_LANE_BYTES, 6144, 4096, 1365](), 1365
    )
    assert_equal(
        _sk_perf_clamped_max_m[FP8_LANE_BYTES, 2560, 2048, 1092](), 1092
    )
    assert_equal(_sk_perf_clamped_max_m[16, 6144, 2048, 1365](), 1365)

    # A tighter workspace budget still wins.
    assert_equal(_sk_perf_clamped_max_m[FP8_LANE_BYTES, 6144, 2048, 128](), 128)
    print("     OK")


def main() raises:
    seed(0)
    _test_route_policy()
    with DeviceContext() as ctx:
        print("===> dense row-major MXFP8 matmul (lane_bytes=32)")

        # const-fill: C must equal K exactly, proving count/tiling/grid.
        _test_case[128, 128, 128, unit_scales=True, ones_data=True](
            "const-fill", ctx
        )
        # data layout alone, then scale indexing alone, then both. Each isolates
        # one half of a relative data-vs-scale permutation.
        _test_case[128, 128, 256, unit_scales=True]("data-only", ctx)
        _test_case[128, 128, 256, ones_data=True]("scales-only", ctx)
        _test_case[128, 128, 256]("random", ctx)
        _test_case[128, 128, 512]("deep-k", ctx)
        # K values that are NOT multiples of 512, so the dispatcher's BK gates
        # have to reject the aggressive buckets. At MXFP8 those gates compare
        # bytes against element-derived thresholds; getting the conversion
        # wrong routes these to a BK the kernel cannot tile.
        # Through the public dispatcher, at small M so the BK-bucket gates are
        # live, and at K values that are NOT multiples of 512 so those gates
        # must reject the aggressive buckets. The gates compare A's BYTE extent
        # against thresholds named in ELEMENTS; if the conversion is wrong at
        # MXFP8 they admit a BK the kernel cannot tile and the build fails.
        _test_dispatch[16, 128, 768]("dispatch-m16-k768", ctx)
        _test_dispatch[64, 128, 768]("dispatch-m64-k768", ctx)
        _test_dispatch[16, 128, 384]("dispatch-m16-k384", ctx)
        _test_case[256, 256, 256]("multi-tile", ctx)
        # Attention-shaped: o_proj-like K, non-square N.
        _test_case[128, 2304, 6144]("attn-qkv-shaped", ctx)

        print("\n--- SK: inter-block split-K (workspace + reduce) ---")

        # MiniMax M3 dimensions: hidden_size=6144, 128 experts with top_k=4,
        # expert intermediate=3072, dense MLP intermediate=12288, and GQA
        # with 64 query heads / 4 KV heads at head_dim=128 (fused QKV has
        # N=9216).
        #
        # `num_splits` below is `_pick_num_splits`'s actual choice for each
        # (N, K) at BK_ELEMS=256 on this GPU. It depends on N, not just K.
        # MoE gate_up and dense QKV both have K=6144, but they don't get
        # the same split factor.
        #
        # MoE gate_up/down get the most coverage here because they dominate
        # M3: 57 of the 60 layers route through them per-expert. The dense
        # shapes are covered too since every request hits them.
        test_mxfp8_matmul_split_k[1, 6144, 6144, num_splits=8](
            ctx
        )  # MoE gate_up
        test_mxfp8_matmul_split_k[16, 6144, 6144, num_splits=8](ctx)
        test_mxfp8_matmul_split_k[64, 6144, 6144, num_splits=8](ctx)
        test_mxfp8_matmul_split_k[1, 6144, 3072, num_splits=6](ctx)  # MoE down
        test_mxfp8_matmul_split_k[16, 6144, 3072, num_splits=6](ctx)
        test_mxfp8_matmul_split_k[64, 6144, 3072, num_splits=6](ctx)
        test_mxfp8_matmul_split_k[1, 9216, 6144, num_splits=6](ctx)  # dense QKV
        test_mxfp8_matmul_split_k[16, 9216, 6144, num_splits=6](ctx)
        test_mxfp8_matmul_split_k[1, 6144, 8192, num_splits=8](
            ctx
        )  # dense O-proj
        test_mxfp8_matmul_split_k[1, 24576, 6144, num_splits=2](
            ctx
        )  # dense MLP gate_up
        test_mxfp8_matmul_split_k[1, 6144, 12288, num_splits=8](
            ctx
        )  # dense MLP down

        # Unaligned-M OOB stress under split-K (M not a multiple of BM=64).
        test_mxfp8_matmul_split_k[17, 6144, 6144, num_splits=8](ctx)
        test_mxfp8_matmul_split_k[63, 6144, 3072, num_splits=6](ctx)

        print("\n--- SK16: narrow-M split-K tile (BM=16, WM=16, WN=64) ---")

        test_mxfp8_matmul_split_k[
            1, 6144, 6144, num_splits=8, BM=16, BN=128, WM=16, WN=64
        ](ctx)
        test_mxfp8_matmul_split_k[
            16, 6144, 6144, num_splits=8, BM=16, BN=128, WM=16, WN=64
        ](ctx)
        test_mxfp8_matmul_split_k[
            7, 6144, 6144, num_splits=8, BM=16, BN=128, WM=16, WN=64
        ](ctx)
        test_mxfp8_matmul_split_k[
            1, 6144, 3072, num_splits=6, BM=16, BN=128, WM=16, WN=64
        ](ctx)
        test_mxfp8_matmul_split_k[
            16, 6144, 3072, num_splits=6, BM=16, BN=128, WM=16, WN=64
        ](ctx)

        print("\n--- dispatch: public entry point now routes to split-K ---")

        # Same MoE shapes as above, but through the public dispatcher.
        # Confirms it actually routes them into split-K.
        _test_dispatch[16, 6144, 6144]("dispatch-sk-moe-gateup", ctx)
        _test_dispatch[16, 6144, 3072]("dispatch-sk-moe-down", ctx)

        print("\n--- BK_ELEMS=128 split tiles (LDS-swizzle-eligible) ---")

        # BK_ELEMS=128 is swizzle-eligible (`num_k_tiles==1`); 256 is not.
        # o_proj key, random data and scales. Appended last (draws RNG).
        test_mxfp8_matmul_split_k[64, 6144, 2048, num_splits=4, BK_ELEMS=128](
            ctx
        )
        test_mxfp8_matmul_split_k[17, 6144, 2048, num_splits=4, BK_ELEMS=128](
            ctx
        )
        test_mxfp8_matmul_split_k[
            16,
            6144,
            2048,
            num_splits=4,
            BM=16,
            BN=128,
            BK_ELEMS=128,
            WM=16,
            WN=64,
        ](ctx)

        # BM=64 band clamps to 2 splits; SK4 above is a different object.
        # Random K, unit scales (reference atol), ragged M for OOB rows.
        test_mxfp8_matmul_split_k[
            64, 6144, 2048, num_splits=2, BK_ELEMS=128, unit_scales=True
        ](ctx)
        test_mxfp8_matmul_split_k[
            17, 6144, 2048, num_splits=2, BK_ELEMS=128, unit_scales=True
        ](ctx)
        # Random-scale twin of the ragged SK2 case; unit_scales is blind to
        # a scale-band offset. M=17 keeps the reference inside atol.
        test_mxfp8_matmul_split_k[17, 6144, 2048, num_splits=2, BK_ELEMS=128](
            ctx
        )

        # o_proj key, one token either side of `_sk_route_max_m` (320): M=319
        # split-K, M=321 dense. Const-fill so C==K; appended last (no RNG).
        _test_dispatch[319, 6144, 2048, unit_scales=True, ones_data=True](
            "dispatch-oproj-m319", ctx
        )
        _test_dispatch[321, 6144, 2048, unit_scales=True, ones_data=True](
            "dispatch-oproj-m321", ctx
        )

        print("\n--- num_stages=2: LDS ping-pong pipeline ---")

        # Const-fill cannot see a stage-base/swizzle fault; signed random at
        # this (N, K) exceeds atol. Complementary pair. Appended last (RNG).
        _test_case[
            256, 6144, 2048, num_stages=2, unit_scales=True, ones_data=True
        ]("dbuf-const-fill", ctx)
        # Random data and scales, M-OOB rows. N=2048 keeps the reference in atol.
        _test_case[64, 2048, 2048, num_stages=2]("dbuf-random-oob", ctx)

        print("\n--- bf16 out: varying operands on the objects that ship ---")

        # o_proj key, bf16 out, positive data: const-fill cannot see a granule
        # swap; signed random exceeds atol. M=321 dense, M=319 split-K. Last (RNG).
        _test_dispatch[
            321, 6144, 2048, positive_data=True, out_dtype=DType.bfloat16
        ]("dispatch-oproj-m321-bf16", ctx)
        _test_dispatch[
            319, 6144, 2048, positive_data=True, out_dtype=DType.bfloat16
        ]("dispatch-oproj-m319-bf16", ctx)

        # Shape-gated decode projection tiles, at exact-fill and unaligned M.
        _test_dispatch[4, 6144, 2048]("dispatch-sk-m3-o-m4", ctx)
        _test_dispatch[16, 6144, 2048]("dispatch-sk-m3-o-m16", ctx)
        _test_dispatch[20, 6144, 2048]("dispatch-sk-m3-o-m20", ctx)
        _test_dispatch[32, 6144, 2048]("dispatch-sk-m3-o-m32", ctx)
        _test_dispatch[48, 6144, 2048]("dispatch-sk-m3-o-m48", ctx)
        _test_dispatch[64, 6144, 2048]("dispatch-sk-m3-o-m64", ctx)
        _test_dispatch[128, 6144, 2048]("dispatch-sk-m3-o-m128", ctx)

        # Reference-checked here because the epilogue test's oracle is a
        # second launch of the same tile.
        _test_dispatch[4, 2560, 6144]("dispatch-sk-m3-qkv-n2560-m4", ctx)
        _test_dispatch[16, 2560, 6144]("dispatch-sk-m3-qkv-n2560-m16", ctx)
        _test_dispatch[4, 2304, 6144]("dispatch-sk-m3-qkv-n2304-m4", ctx)
