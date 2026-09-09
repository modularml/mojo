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

"""Split-K epilogue and workspace-cap tests for `BlockScaledMatmulAMD`.

Both formats run the same test bodies, parameterized on `lane_bytes` (16 =
MXFP4, 32 = MXFP8) -- the dispatcher is dtype-normalized, so the two differ
only in operand packing, not in routing.

Neither test needs a dequant reference. Each runs the launch twice over the
same inputs, once plain and once with an epilogue, so the plain arm's stored
output IS the value the epilogue must have seen; matmul correctness itself
belongs to `test_mxfp{4,8}_matmul_amd.mojo`.
"""

from std.atomic import Atomic
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.random import random_ui64
from std.testing import assert_equal, assert_true
from std.utils.index import IndexList

from layout import TileTensor
from layout.tile_layout import row_major
from linalg.fp4_utils import MXFP4_SF_VECTOR_SIZE, MXFP8_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd.block_scaled_matmul_amd import (
    _launch_block_scaled_split_k,
    block_scaled_matmul_amd,
)

comptime MXFP4_LANE_BYTES = 16
comptime MXFP8_LANE_BYTES = 32

# `c` is pre-filled with this: a reduce carrying an epilogue routes every
# element through the lambda and never stores, so the sentinel must survive.
comptime SENTINEL = Float32(-98765.0)


def _rand_operand_byte[lane_bytes: Int]() -> UInt8:
    """Random operand byte for the format `lane_bytes` selects.

    E2M1 nibbles have no NaN, so MXFP4 takes any byte. E4M3 does (0x7F, 0xFF)
    and 0x7E is 448, so MXFP8 encodes a finite draw from [-1, 1] instead.
    """
    comptime if lane_bytes == MXFP4_LANE_BYTES:
        return UInt8(Int(random_ui64(0, 255)))
    else:
        var f = Float32(Int(random_ui64(0, 2000))) / 1000.0 - 1.0
        return bitcast[.uint8, 1](Float8_e4m3fn(f.cast[.float8_e4m3fn]()))[0]


def _assert_epilogue_arm(
    epi: ImmPointer[Float32, _],
    reduced: ImmPointer[Float32, _],
    fire: ImmPointer[Int32, _],
    c: ImmPointer[Float32, _],
    saw_wide: Int,
    n_elems: Int,
    arm: String,
) raises:
    """Checks the three epilogue invariants against the plain arm's output.

    `reduced` came from the same launch without an epilogue, so it carries the
    same accumulation order and the value check is exact.
    """
    assert_equal(
        saw_wide,
        0,
        msg=String(arm, ": epilogue ran at width > 1, so split-K did not"),
    )
    var live = 0
    for i in range(n_elems):
        if reduced[i] != 0.0:
            live += 1
        assert_equal(
            epi[i],
            reduced[i] * 2.0 + 1.0,
            msg=String(arm, ": epilogue value mismatch at ", i),
        )
        assert_equal(
            Int(fire[i]),
            1,
            msg=String(arm, ": epilogue fired more than once at ", i),
        )
        assert_equal(
            c[i],
            SENTINEL,
            msg=String(arm, ": stored to c despite an epilogue at ", i),
        )

    # Liveness: `2*0 + 1` matches a dead kernel on both sides, so the value
    # check above passes vacuously if the GEMM produced zeros.
    assert_true(
        live * 2 > n_elems,
        msg=String(arm, ": only ", live, " of ", n_elems, " outputs nonzero"),
    )


def test_split_k_epilogue[
    lane_bytes: Int,
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
    """Split-K carrying an `elementwise_lambda_fn`.

    The epilogue must see each cell exactly once, holding the SUMMED value.
    The affine `2*val + 1` separates the failures: per-split accumulation
    gives `2*sum + num_splits` and last-writer-wins gives `2*partial + 1`,
    neither of which can imitate `2*sum + 1`. `fire_d` then counts visits per
    cell, and `c` must keep its sentinel.
    """
    comptime is_fp4 = lane_bytes == MXFP4_LANE_BYTES
    comptime SF_VECTOR_SIZE = (
        MXFP4_SF_VECTOR_SIZE if is_fp4 else MXFP8_SF_VECTOR_SIZE
    )
    comptime assert (
        K_static % SF_VECTOR_SIZE == 0
    ), "K must be a multiple of the 32-element scale block"
    comptime K_BYTES = K_static // (2 if is_fp4 else 1)
    comptime K_SCALES = K_static // SF_VECTOR_SIZE
    comptime N_ELEMS = M_static * N_static

    print(
        "   MXFP",
        4 if is_fp4 else 8,
        " ",
        M_static,
        "x",
        N_static,
        "x",
        K_static,
        " [split-K + epilogue num_splits=",
        num_splits,
        "]",
    )

    var a_h = ctx.enqueue_create_host_buffer[.uint8](M_static * K_BYTES)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](N_static * K_BYTES)
    var sfa_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        M_static * K_SCALES
    )
    var sfb_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        N_static * K_SCALES
    )
    var epi_h = ctx.enqueue_create_host_buffer[.float32](N_ELEMS)
    var fire_h = ctx.enqueue_create_host_buffer[.int32](N_ELEMS)
    var wide_h = ctx.enqueue_create_host_buffer[.int32](1)
    var c_init_h = ctx.enqueue_create_host_buffer[.float32](N_ELEMS)
    ctx.synchronize()

    for i in range(M_static * K_BYTES):
        a_h[i] = _rand_operand_byte[lane_bytes]()
    for i in range(N_static * K_BYTES):
        b_h[i] = _rand_operand_byte[lane_bytes]()
    for i in range(M_static * K_SCALES):
        sfa_h[i] = bitcast[.float8_e8m0fnu](UInt8(Int(random_ui64(124, 130))))
    for i in range(N_static * K_SCALES):
        sfb_h[i] = bitcast[.float8_e8m0fnu](UInt8(Int(random_ui64(124, 130))))
    for i in range(N_ELEMS):
        epi_h[i] = Float32(0)
        fire_h[i] = Int32(0)
        c_init_h[i] = SENTINEL
    wide_h[0] = Int32(0)

    var a_d = ctx.enqueue_create_buffer[.uint8](M_static * K_BYTES)
    var b_d = ctx.enqueue_create_buffer[.uint8](N_static * K_BYTES)
    var sfa_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](M_static * K_SCALES)
    var sfb_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](N_static * K_SCALES)
    var c_d = ctx.enqueue_create_buffer[.float32](N_ELEMS)
    var reduced_d = ctx.enqueue_create_buffer[.float32](N_ELEMS)
    var epi_d = ctx.enqueue_create_buffer[.float32](N_ELEMS)
    var fire_d = ctx.enqueue_create_buffer[.int32](N_ELEMS)
    var wide_d = ctx.enqueue_create_buffer[.int32](1)
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(sfa_d, sfa_h)
    ctx.enqueue_copy(sfb_d, sfb_h)

    var c_tt = TileTensor(c_d, row_major[M_static, N_static]())
    var reduced_tt = TileTensor(reduced_d, row_major[M_static, N_static]())
    var a_tt = TileTensor(a_d, row_major[M_static, K_BYTES]()).as_immut()
    var b_tt = TileTensor(b_d, row_major[N_static, K_BYTES]()).as_immut()
    var sfa_tt = TileTensor(sfa_d, row_major[M_static, K_SCALES]()).as_immut()
    var sfb_tt = TileTensor(sfb_d, row_major[N_static, K_SCALES]()).as_immut()

    var epi_ptr = epi_d.unsafe_ptr()
    var fire_ptr = fire_d.unsafe_ptr()
    var wide_ptr = wide_d.unsafe_ptr()

    @__parameter
    @__copy_capture(epi_ptr, fire_ptr, wide_ptr)
    @always_inline
    def record[
        _dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[_dtype, width]):
        # Routing oracle: the reduce hands one scalar per cell, the in-tile
        # `RegTileEpilogue` hands `chunk_width` contiguous columns, so width
        # > 1 can only be a non-split launch.
        comptime if width > 1:
            wide_ptr[0] = Int32(1)
        var base = idx[0] * N_static + idx[1]
        comptime for w in range(width):
            epi_ptr[base + w] = val[w].cast[.float32]() * 2.0 + 1.0
            _ = Atomic[Int32].fetch_add(fire_ptr + base + w, Int32(1))

    var c_h = ctx.enqueue_create_host_buffer[.float32](N_ELEMS)
    var reduced_h = ctx.enqueue_create_host_buffer[.float32](N_ELEMS)
    var epi_out_h = ctx.enqueue_create_host_buffer[.float32](N_ELEMS)
    var fire_out_h = ctx.enqueue_create_host_buffer[.int32](N_ELEMS)
    var wide_out_h = ctx.enqueue_create_host_buffer[.int32](1)

    # Direct launch. The plain arm fixes `num_splits`, so it reduces in the
    # same order as the epilogue arm and the value check below is exact.
    ctx.enqueue_copy(c_d, c_init_h)
    ctx.enqueue_copy(epi_d, epi_h)
    ctx.enqueue_copy(fire_d, fire_h)
    ctx.enqueue_copy(wide_d, wide_h)

    _launch_block_scaled_split_k[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        WM=WM,
        WN=WN,
        num_splits=num_splits,
        lane_bytes=lane_bytes,
    ](reduced_tt, a_tt, b_tt, sfa_tt, sfb_tt, M_static, ctx)

    _launch_block_scaled_split_k[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        WM=WM,
        WN=WN,
        num_splits=num_splits,
        lane_bytes=lane_bytes,
        elementwise_lambda_fn=record,
    ](c_tt, a_tt, b_tt, sfa_tt, sfb_tt, M_static, ctx)

    ctx.enqueue_copy(c_h, c_d)
    ctx.enqueue_copy(reduced_h, reduced_d)
    ctx.enqueue_copy(epi_out_h, epi_d)
    ctx.enqueue_copy(fire_out_h, fire_d)
    ctx.enqueue_copy(wide_out_h, wide_d)
    ctx.synchronize()

    _assert_epilogue_arm(
        epi_out_h.unsafe_ptr(),
        reduced_h.unsafe_ptr(),
        fire_out_h.unsafe_ptr(),
        c_h.unsafe_ptr(),
        Int(wide_out_h[0]),
        N_ELEMS,
        "direct launch",
    )

    # Same pair through the PUBLIC entry point, which proves the dispatcher
    # can instantiate split-K + epilogue at all. Routing does not depend on
    # the lambda, so the plain call picks the same split factor and the value
    # check stays exact.
    ctx.enqueue_copy(c_d, c_init_h)
    ctx.enqueue_copy(epi_d, epi_h)
    ctx.enqueue_copy(fire_d, fire_h)
    ctx.enqueue_copy(wide_d, wide_h)

    block_scaled_matmul_amd[lane_bytes=lane_bytes](
        reduced_tt, a_tt, b_tt, sfa_tt, sfb_tt, ctx
    )
    block_scaled_matmul_amd[
        lane_bytes=lane_bytes, elementwise_lambda_fn=record
    ](c_tt, a_tt, b_tt, sfa_tt, sfb_tt, ctx)

    ctx.enqueue_copy(c_h, c_d)
    ctx.enqueue_copy(reduced_h, reduced_d)
    ctx.enqueue_copy(epi_out_h, epi_d)
    ctx.enqueue_copy(fire_out_h, fire_d)
    ctx.enqueue_copy(wide_out_h, wide_d)
    ctx.synchronize()

    _assert_epilogue_arm(
        epi_out_h.unsafe_ptr(),
        reduced_h.unsafe_ptr(),
        fire_out_h.unsafe_ptr(),
        c_h.unsafe_ptr(),
        Int(wide_out_h[0]),
        N_ELEMS,
        "dispatch",
    )
    print("      OK")


def test_dispatch_workspace_cap[
    lane_bytes: Int,
    M_static: Int,
    N_static: Int,
    K_static: Int,
    expect_split_k: Bool,
](name: String, ctx: DeviceContext) raises:
    """Pins where the dispatcher stops choosing split-K.

    Routing only, so there is no reference matmul — the epilogue's call width
    is the oracle (1 = the reduce ran, > 1 = the non-split fallback), over an
    element count that keeps `width == 1` from also meaning "never fired". The
    `_sk_max_m` cap has no margin at the top: the Llama-405B O-proj shapes land
    on `_sk_max_m == 512` exactly, so shrinking `SK_MAX_WORKSPACE_BYTES`,
    changing `SK_CTA_WAVES`, or a different `sm_count` would reroute them
    silently.
    """
    comptime is_fp4 = lane_bytes == MXFP4_LANE_BYTES
    comptime SF_VECTOR_SIZE = (
        MXFP4_SF_VECTOR_SIZE if is_fp4 else MXFP8_SF_VECTOR_SIZE
    )
    comptime K_BYTES = K_static // (2 if is_fp4 else 1)
    comptime K_SCALES = K_static // SF_VECTOR_SIZE

    print(
        "   MXFP",
        4 if is_fp4 else 8,
        " ",
        name,
        " ",
        M_static,
        "x",
        N_static,
        "x",
        K_static,
    )

    var a_h = ctx.enqueue_create_host_buffer[.uint8](M_static * K_BYTES)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](N_static * K_BYTES)
    var sf_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        max(M_static, N_static) * K_SCALES
    )
    var wide_h = ctx.enqueue_create_host_buffer[.int32](1)
    ctx.synchronize()

    for i in range(M_static * K_BYTES):
        a_h[i] = _rand_operand_byte[lane_bytes]()
    for i in range(N_static * K_BYTES):
        b_h[i] = _rand_operand_byte[lane_bytes]()
    for i in range(max(M_static, N_static) * K_SCALES):
        sf_h[i] = bitcast[.float8_e8m0fnu](UInt8(127))
    wide_h[0] = Int32(0)

    var a_d = ctx.enqueue_create_buffer[.uint8](M_static * K_BYTES)
    var b_d = ctx.enqueue_create_buffer[.uint8](N_static * K_BYTES)
    var sfa_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](M_static * K_SCALES)
    var sfb_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](N_static * K_SCALES)
    var c_d = ctx.enqueue_create_buffer[.float32](M_static * N_static)
    var seen_d = ctx.enqueue_create_buffer[.int32](1)
    var wide_d = ctx.enqueue_create_buffer[.int32](1)
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(sfa_d, sf_h)
    ctx.enqueue_copy(sfb_d, sf_h)
    ctx.enqueue_copy(wide_d, wide_h)
    ctx.enqueue_copy(seen_d, wide_h)

    var c_tt = TileTensor(c_d, row_major[M_static, N_static]())
    var a_tt = TileTensor(a_d, row_major[M_static, K_BYTES]()).as_immut()
    var b_tt = TileTensor(b_d, row_major[N_static, K_BYTES]()).as_immut()
    var sfa_tt = TileTensor(sfa_d, row_major[M_static, K_SCALES]()).as_immut()
    var sfb_tt = TileTensor(sfb_d, row_major[N_static, K_SCALES]()).as_immut()

    var seen_ptr = seen_d.unsafe_ptr()
    var wide_ptr = wide_d.unsafe_ptr()

    @__parameter
    @__copy_capture(seen_ptr, wide_ptr)
    @always_inline
    def probe[
        _dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[_dtype, width]):
        comptime if width > 1:
            wide_ptr[0] = Int32(1)
        # Count elements, not calls: the split arm's reduce hands one scalar per
        # cell and the non-split arm hands `width` at a time, so the total is
        # M*N either way. Without this, `wide == 0` on the split arm is the
        # absence of evidence -- an epilogue that never fired reads the same.
        _ = Atomic[Int32].fetch_add(seen_ptr, Int32(width))

    block_scaled_matmul_amd[lane_bytes=lane_bytes, elementwise_lambda_fn=probe](
        c_tt, a_tt, b_tt, sfa_tt, sfb_tt, ctx
    )

    var wide_out_h = ctx.enqueue_create_host_buffer[.int32](1)
    var seen_out_h = ctx.enqueue_create_host_buffer[.int32](1)
    ctx.enqueue_copy(wide_out_h, wide_d)
    ctx.enqueue_copy(seen_out_h, seen_d)
    ctx.synchronize()

    # Liveness first: both M values here are multiples of their branch's BM, so
    # no OOB row fires and the count is exactly M*N.
    assert_equal(
        Int(seen_out_h[0]),
        M_static * N_static,
        msg=String(
            "epilogue covered ",
            Int(seen_out_h[0]),
            " of ",
            M_static * N_static,
            " elements at M=",
            M_static,
        ),
    )
    assert_equal(
        Int(wide_out_h[0]),
        0 if expect_split_k else 1,
        msg=String(
            "routing changed: expected ",
            "split-K" if expect_split_k else "the non-split fallback",
            " at M=",
            M_static,
        ),
    )
    print("      OK")


def main() raises:
    with DeviceContext() as ctx:
        print("===> MX split-K epilogue (MXFP4 + MXFP8)")

        print("\n--- SK + epilogue: the lambda rides the reduce ---")

        # M3's fused QKV+IndexQK GEMM at TP4: N = 2048 (Q) + 128 (K) + 128 (V)
        # + 128 (IndexQ) + 128 (IndexK) = 2560, K = hidden 6144. Its K/V and
        # IndexK scatter IS the `elementwise_lambda_fn`. Both formats reach the
        # same 12-way split there, so one `num_splits` serves both arms.
        comptime for lane_bytes in [MXFP4_LANE_BYTES, MXFP8_LANE_BYTES]:
            # These params match what dispatch picks at MXFP4; at MXFP8 it now
            # picks BN=64, so there the two arms exercise different tiles.
            test_split_k_epilogue[
                lane_bytes, 16, 2560, 6144, num_splits=12, BM=16, WM=16, WN=64
            ](ctx)
            test_split_k_epilogue[lane_bytes, 32, 2560, 6144, num_splits=12](
                ctx
            )
            # M off a BM multiple, so the matmul clamps 47 OOB rows out of the
            # BM=64 tile while the reduce reads only rows 0..16. Not a reduce
            # tail: N=2560 is a multiple of the reduce's 256-thread block, so
            # M*N is too and no thread ever falls off the end.
            test_split_k_epilogue[lane_bytes, 17, 2560, 6144, num_splits=12](
                ctx
            )
            # A second (N, split) pair, so the controls cannot be passing by
            # fitting one geometry.
            test_split_k_epilogue[lane_bytes, 16, 6144, 3072, num_splits=6](ctx)

        print(
            "\n--- SK workspace cap: where the dispatcher stops splitting ---"
        )

        # N=2560, K=6144 splits 12 ways, so the 128 MiB budget admits
        # M <= 128*1024*1024 / (12 * 2560 * 4) = 1092. Straddle it: a decode M
        # splits, a prefill-shaped M must not (M=2048 would want 240 MiB).
        comptime for lane_bytes in [MXFP4_LANE_BYTES, MXFP8_LANE_BYTES]:
            test_dispatch_workspace_cap[
                lane_bytes, 1024, 2560, 6144, expect_split_k=True
            ]("cap-under-m1024", ctx)
            test_dispatch_workspace_cap[
                lane_bytes, 2048, 2560, 6144, expect_split_k=False
            ]("cap-over-m2048", ctx)

        # `N % SK_BN != 0` must route past split-K whatever the workspace says:
        # the matmul writes its f32 workspace through `RegTileWriter`, which
        # spills the last column block into the next row instead of clipping it.
        # N=2564 keeps the wide-N gate off (ceildiv(2564,32)=81 < sm_count) and
        # `_pick_num_splits` still finds a legal factor there, so without the
        # alignment term this arm would split.
        comptime for lane_bytes in [MXFP4_LANE_BYTES, MXFP8_LANE_BYTES]:
            test_dispatch_workspace_cap[
                lane_bytes, 1024, 2564, 6144, expect_split_k=False
            ]("n-unaligned-2564", ctx)

        print("\n==== All MX split-K epilogue tests passed ====")
