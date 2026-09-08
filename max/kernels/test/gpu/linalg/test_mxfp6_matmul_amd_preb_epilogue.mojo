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
"""`mxfp6_block_scaled_matmul_amd` fused-epilogue equivalence, preshuffled-B.

The preshuffled-B struct gained its `elementwise_lambda_fn` with this test.
The fused QKV ops reach their epilogue through the split-K and default
launches instead -- nothing sets `preshuffled_b` for them yet -- so this test
is currently the only exercise of the preshuffled-B epilogue.

Two gates per case, because neither alone is sufficient:

1. The lambda must receive exactly what the direct store would have written.
   Both paths take the same `c_reg` accumulator and the same cast, so the
   comparison is bit-exact, not tolerance-based. A wrong (m, n) mapping lands
   values in the wrong place and shows up here.
2. With a lambda set, DRAM must be left untouched. The C buffer is filled with
   a sentinel beforehand and must still hold it afterwards. Gate 1 alone would
   pass if the kernel wrote BOTH the lambda destination and C.

Run:
  ./bazelw test //max/kernels/test/gpu/linalg:test_mxfp6_matmul_amd_preb_epilogue.mojo.test
"""

from std.math import ceildiv
from std.random import random_ui64, seed
from std.sys import align_of
from std.testing import assert_equal, assert_true
from std.utils import IndexList

from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.info import MI355X

from layout import Coord, Idx, TileTensor, row_major
from linalg.fp6_utils import MXFP6_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import Shuffler
from linalg.matmul.gpu.amd.block_scaled_matmul_amd import (
    mxfp6_block_scaled_matmul_amd,
)

comptime FP6_LANE_BYTES = 24


def _fill_random(mut buf: HostBuffer[.uint8], n: Int, lo: Int, hi: Int):
    for i in range(n):
        buf[i] = UInt8(Int(random_ui64(UInt64(lo), UInt64(hi))))


def _run_case[
    M_static: Int,
    N_static: Int,
    K_static: Int,
    preshuffled_b: Bool = True,
](name: String, ctx: DeviceContext) raises:
    comptime assert K_static % 128 == 0, "K must be a multiple of 128"
    comptime assert N_static % 64 == 0, "N must be a multiple of BN=64"
    comptime K_BYTES = (K_static * 6) // 8
    comptime scale_K = K_static // MXFP6_SF_VECTOR_SIZE
    comptime SENTINEL = Float32(-12345.0)

    print("  ", name, " M=", M_static, " N=", N_static, " K=", K_static)

    var a_h = ctx.enqueue_create_host_buffer[.uint8](M_static * K_BYTES)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](N_static * K_BYTES)
    var sfa_h = ctx.enqueue_create_host_buffer[.uint8](M_static * scale_K)
    var sfb_h = ctx.enqueue_create_host_buffer[.uint8](N_static * scale_K)
    var sfb_pre_h = ctx.enqueue_create_host_buffer[.uint8](N_static * scale_K)
    ctx.synchronize()

    _fill_random(a_h, M_static * K_BYTES, 0, 255)
    _fill_random(b_h, N_static * K_BYTES, 0, 255)
    _fill_random(sfa_h, M_static * scale_K, 125, 129)
    _fill_random(sfb_h, N_static * scale_K, 125, 129)

    var sfb_h_tt = TileTensor[mut=True](
        sfb_h, row_major(Coord(Idx[1], Idx[N_static], Idx[scale_K]))
    )
    _ = Shuffler[1].preshuffle_scale_4d[MN=N_static, K_SCALES=scale_K](
        sfb_h_tt, sfb_pre_h
    )

    var a_d = ctx.enqueue_create_buffer[.uint8](M_static * K_BYTES)
    var b_d = ctx.enqueue_create_buffer[.uint8](N_static * K_BYTES)
    var b_pre_d = ctx.enqueue_create_buffer[.uint8](N_static * K_BYTES)
    var sfa_d = ctx.enqueue_create_buffer[.uint8](M_static * scale_K)
    var sfb_pre_d = ctx.enqueue_create_buffer[.uint8](N_static * scale_K)
    var sfb_raw_d = ctx.enqueue_create_buffer[.uint8](N_static * scale_K)
    var c_direct_d = ctx.enqueue_create_buffer[.float32](M_static * N_static)
    var c_sentinel_d = ctx.enqueue_create_buffer[.float32](M_static * N_static)
    var out_d = ctx.enqueue_create_buffer[.float32](M_static * N_static)

    c_direct_d.enqueue_fill(Float32(0.0))
    c_sentinel_d.enqueue_fill(SENTINEL)
    out_d.enqueue_fill(Float32(0.0))
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(sfa_d, sfa_h)
    ctx.enqueue_copy(sfb_pre_d, sfb_pre_h)
    ctx.enqueue_copy(sfb_raw_d, sfb_h)

    var b_raw_tt = TileTensor[mut=False](b_d, row_major[1, N_static, K_BYTES]())
    var b_pre_dst_tt = TileTensor[mut=True](
        b_pre_d, row_major[1, N_static, K_BYTES]()
    )
    Shuffler[1].preshuffle_b_planes[
        N=N_static, K_BYTES=K_BYTES, lane_bytes=FP6_LANE_BYTES
    ](b_raw_tt, b_pre_dst_tt, ctx)

    var a_tt = TileTensor[mut=False](
        a_d, row_major(Coord(M_static, Idx[K_BYTES]))
    )
    var b_pre_tt = TileTensor[mut=False](
        b_pre_d, row_major(Coord(Idx[N_static], Idx[K_BYTES]))
    )
    var sfa_tt = TileTensor[mut=False](
        sfa_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        row_major(Coord(M_static, Idx[scale_K])),
    )
    var sfb_pre_tt = TileTensor[mut=False](
        sfb_pre_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        row_major(Coord(Idx[N_static], Idx[scale_K])),
    )

    var sfb_raw_tt = TileTensor[mut=False](
        sfb_raw_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        row_major(Coord(Idx[N_static], Idx[scale_K])),
    )
    var b_row_tt = TileTensor[mut=False](
        b_d, row_major(Coord(Idx[N_static], Idx[K_BYTES]))
    )
    var c_direct_tt = TileTensor[mut=True](
        c_direct_d, row_major(Coord(M_static, Idx[N_static]))
    )
    comptime if preshuffled_b:
        mxfp6_block_scaled_matmul_amd[preshuffled_b=True](
            c_direct_tt, a_tt, b_pre_tt, sfa_tt, sfb_pre_tt, ctx
        )
    else:
        mxfp6_block_scaled_matmul_amd[preshuffled_b=False](
            c_direct_tt, a_tt, b_row_tt, sfa_tt, sfb_raw_tt, ctx
        )

    var out_tt = TileTensor[mut=True](
        out_d, row_major(Coord(M_static, Idx[N_static]))
    )
    var c_sentinel_tt = TileTensor[mut=True](
        c_sentinel_d, row_major(Coord(M_static, Idx[N_static]))
    )

    @always_inline
    @__copy_capture(out_tt)
    def epilogue_fn[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        out_tt.store[width=width, alignment=alignment](
            Coord(idx), rebind[SIMD[DType.float32, width]](val)
        )

    comptime if preshuffled_b:
        mxfp6_block_scaled_matmul_amd[
            preshuffled_b=True, elementwise_lambda_fn=epilogue_fn
        ](c_sentinel_tt, a_tt, b_pre_tt, sfa_tt, sfb_pre_tt, ctx)
    else:
        mxfp6_block_scaled_matmul_amd[
            preshuffled_b=False, elementwise_lambda_fn=epilogue_fn
        ](c_sentinel_tt, a_tt, b_row_tt, sfa_tt, sfb_raw_tt, ctx)

    var direct_h = ctx.enqueue_create_host_buffer[.float32](M_static * N_static)
    var out_host = ctx.enqueue_create_host_buffer[.float32](M_static * N_static)
    var sentinel_h = ctx.enqueue_create_host_buffer[.float32](
        M_static * N_static
    )
    ctx.enqueue_copy(direct_h, c_direct_d)
    ctx.enqueue_copy(out_host, out_d)
    ctx.enqueue_copy(sentinel_h, c_sentinel_d)
    ctx.synchronize()

    var mismatches = 0
    var saw_nonzero = False
    for i in range(M_static * N_static):
        if direct_h[i] != Float32(0.0):
            saw_nonzero = True
        if direct_h[i] != out_host[i]:
            if mismatches < 3:
                print(
                    "      [",
                    i // N_static,
                    ",",
                    i % N_static,
                    "] direct=",
                    direct_h[i],
                    " lambda=",
                    out_host[i],
                )
            mismatches += 1
    assert_equal(mismatches, 0)
    assert_true(saw_nonzero, "reference output was all zero -- test is vacuous")

    var clobbered = 0
    for i in range(M_static * N_static):
        if sentinel_h[i] != SENTINEL:
            clobbered += 1
    assert_equal(clobbered, 0)

    print("    OK: ", M_static * N_static, " elements match; C untouched")

    _ = a_d^
    _ = b_d^
    _ = b_pre_d^
    _ = sfa_d^
    _ = sfb_pre_d^
    _ = sfb_raw_d^
    _ = c_direct_d^
    _ = c_sentinel_d^
    _ = out_d^


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "test_mxfp6_matmul_amd_preb_epilogue requires MI355X"

    print("===> mxfp6 preb fused epilogue: equivalence + no DRAM write")

    print("  -- decode tile (M <= 64), non-preb epilogue --")
    _run_case[M_static=16, N_static=6144, K_static=3072]("decode-M16", ctx)
    _run_case[M_static=64, N_static=6144, K_static=3072]("decode-M64", ctx)

    print("  -- preshuffled-B tile (M > 64), the epilogue added here --")
    _run_case[M_static=128, N_static=6144, K_static=3072]("preb-M128", ctx)
    _run_case[M_static=512, N_static=6144, K_static=3072]("preb-M512", ctx)
    _run_case[M_static=2048, N_static=6144, K_static=3072]("preb-M2048", ctx)
    _run_case[M_static=128, N_static=2048, K_static=3072]("preb-narrowN", ctx)
    _run_case[M_static=130, N_static=6144, K_static=3072]("preb-Mtail", ctx)

    # The launch the fused QKV ops actually take. Nothing sets `preshuffled_b`
    # for them, so every M > DECODE_M_MAX call lands in `_launch_block_scaled`
    # at BM=96/BN=64/WM=32 -- a three-warps-along-M epilogue whose `m`
    # derivation differs from both cases above, and which no case here reached.
    print(
        "  -- non-preshuffled row-major tile (BM=96), the production launch --"
    )
    _run_case[M_static=128, N_static=6144, K_static=3072, preshuffled_b=False](
        "rowmajor-M128", ctx
    )
    _run_case[M_static=130, N_static=6144, K_static=3072, preshuffled_b=False](
        "rowmajor-Mtail", ctx
    )
