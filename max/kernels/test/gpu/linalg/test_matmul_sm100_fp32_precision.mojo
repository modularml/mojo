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
"""Precision contract of the SM100 fp32 matmul dispatch (KERN-3151).

For fp32 `transpose_b` shapes with static N <= 256 and static K >= 2048 (the
MoE router/gate GEMM class, e.g. N=128, K=6144), `matmul_dispatch_sm100`
routes m <= 64 to the split-K GEMV (CUDA-core IEEE-fp32 FMA) and m > 64 to
the UMMA tile GEMM, whose operands tcgen05 truncates to TF32's 10-bit
mantissa (there is no fp32 UMMA kind; accumulation stays fp32). The comptime
`use_tf32` matmul parameter (default True) opts out of the truncation:
with `use_tf32=False`, every M stays on the IEEE-fp32 GEMV.

This test drives the real dispatch in both modes against an fp64 host
reference computed from the same fp32 inputs:

1. Default (use_tf32=True), m <= 64: the split-K GEMV is fp32-exact
   (mean rel err ~1e-6).
2. Default, m > 64: TF32-level error (mean rel err ~2e-3), asserted as a
   two-sided band on purpose — if the default path ever changes precision
   in either direction, this fails and the contract gets re-decided
   consciously. The same 64 input rows therefore produce different outputs
   at m=64 vs m=128 (the dispatch cliff).
3. use_tf32=False, m in {64, 65, 128, 512}: fp32-exact at every M. The
   m=64 and m=128 results match the default m=64 launch bit for bit on the
   shared rows — the parameter does not perturb the m <= 64 path, and for
   m > 64 it selects the same GEMV instantiation with the same per-block
   reduction order, so M no longer changes results.
"""

from std.random import random_float64, seed

from max.gpu.host import DeviceContext
from std.testing import assert_true
from layout import TileTensor, Coord, Idx, DefaultEngine, row_major
from linalg.matmul.gpu import _matmul_gpu

comptime N = 128
comptime K = 6144
comptime MAX_M = 512
comptime REF_ROWS = 64

comptime f32 = DType.float32
comptime f64 = DType.float64

# Measured on B200: the split-K GEMV's mean rel err vs fp64 is ~5.5e-7 at
# m=64; the TF32 tile GEMM sits at ~1.7e-3 for every m > 64. The bounds
# below keep >4x margin on both sides of each band. Max-rel is deliberately
# not asserted: near-zero reference values from the random +-1 inputs make
# it a cancellation artifact, not a precision signal.
comptime FP32_MEAN_REL_BOUND = 1e-5
comptime FP32_MAX_ABS_BOUND = 5e-3
comptime TF32_MEAN_REL_LOW = 5e-4
comptime TF32_MEAN_REL_HIGH = 5e-2


@fieldwise_init
struct Stats(Copyable, Movable):
    var mean_rel: Float64
    var max_abs: Float64


def _stats_vs_ref(
    got: TileTensor[f32, ...],
    reference: TileTensor[f64, ...],
    rows: Int,
) raises -> Stats:
    var sum_rel = Float64(0.0)
    var max_abs = Float64(0.0)
    for mm in range(rows):
        for nn in range(N):
            var g = Float64(got.raw_load(mm * N + nn))
            var r = reference.raw_load(mm * N + nn)
            var ad = abs(g - r)
            var denom = abs(r)
            if denom < 1e-30:
                denom = 1e-30
            sum_rel += ad / denom
            if ad > max_abs:
                max_abs = ad
    return Stats(sum_rel / Float64(rows * N), max_abs)


def _run_matmul[
    use_tf32: Bool = True
](
    a_host: TileTensor[
        f32,
        address_space=.GENERIC,
        ...,
        Engine=DefaultEngine[element_width=1],
    ],
    b_dev: TileTensor[f32, ...],
    c_host: TileTensor[
        mut=True,
        f32,
        address_space=.GENERIC,
        ...,
        Engine=DefaultEngine[element_width=1],
    ],
    m: Int,
    ctx: DeviceContext,
) raises:
    """Drives the real dispatch entry (`matmul[target="gpu"]` forwards to
    `_matmul_gpu`) on the first `m` rows of `a_host`. N and K are
    compile-time static so the split-K GEMV guard in matmul_dispatch_sm100
    applies; the runtime `m` picks the kernel and the comptime `use_tf32`
    picks the dispatch mode."""
    var a_dev_buf = ctx.enqueue_create_buffer[f32](m * K)
    ctx.enqueue_copy(a_dev_buf, a_host._storage)
    var a_dev = TileTensor(a_dev_buf, row_major(Coord(Int(m), Idx[K])))

    var c_dev_buf = ctx.enqueue_create_buffer[f32](m * N)
    var c_dev = TileTensor(c_dev_buf, row_major(Coord(Int(m), Idx[N])))

    _matmul_gpu[use_tensor_core=True, transpose_b=True, use_tf32=use_tf32](
        c_dev, a_dev, b_dev, ctx
    )

    ctx.enqueue_copy(c_host._storage, c_dev_buf)
    ctx.synchronize()
    _ = a_dev_buf^
    _ = c_dev_buf^


def main() raises:
    with DeviceContext() as ctx:
        seed(0xC0FFEE)
        var a_host_buf = ctx.enqueue_create_host_buffer[f32](MAX_M * K)
        var a_host = TileTensor(
            a_host_buf, row_major(Coord(Idx[MAX_M], Idx[K]))
        )
        var b_host_buf = ctx.enqueue_create_host_buffer[f32](N * K)
        var b_host = TileTensor(b_host_buf, row_major(Coord(Idx[N], Idx[K])))
        for i in range(a_host.num_elements()):
            a_host.raw_store(i, Float32(random_float64(-1.0, 1.0)))
        for i in range(b_host.num_elements()):
            b_host.raw_store(i, Float32(random_float64(-1.0, 1.0)))

        var b_dev_buf = ctx.enqueue_create_buffer[f32](N * K)
        var b_dev = TileTensor(b_dev_buf, row_major(Coord(Idx[N], Idx[K])))
        ctx.enqueue_copy(b_dev_buf, b_host._storage)
        ctx.synchronize()

        # fp64 reference for the first REF_ROWS rows, from the same fp32
        # inputs. transpose_b: C[mm, nn] = sum_k A[mm, k] * B[nn, k].
        var ref_buf = ctx.enqueue_create_host_buffer[f64](REF_ROWS * N)
        var ref_host = TileTensor(
            ref_buf, row_major(Coord(Idx[REF_ROWS], Idx[N]))
        )
        for mm in range(REF_ROWS):
            for nn in range(N):
                var acc = Float64(0.0)
                for kk in range(K):
                    acc += Float64(a_host.raw_load(mm * K + kk)) * Float64(
                        b_host.raw_load(nn * K + kk)
                    )
                ref_host.raw_store(mm * N + nn, acc)

        # Kept for the cross-m comparisons below.
        var c64_buf = ctx.enqueue_create_host_buffer[f32](REF_ROWS * N)
        var c64 = TileTensor(c64_buf, row_major(Coord(Idx[REF_ROWS], Idx[N])))
        var c128_tf32_buf = ctx.enqueue_create_host_buffer[f32](REF_ROWS * N)
        var c128_tf32 = TileTensor(
            c128_tf32_buf, row_major(Coord(Idx[REF_ROWS], Idx[N]))
        )

        print("[phase 1] use_tf32=True, m <= 64: split-K GEMV, IEEE fp32")
        var ms_small = [1, 6, 12, 64]
        for idx in range(len(ms_small)):
            var m = ms_small[idx]
            var c_buf = ctx.enqueue_create_host_buffer[f32](m * N)
            var c = TileTensor(c_buf, row_major(Coord(Int(m), Idx[N])))
            _run_matmul(a_host, b_dev, c, m, ctx)
            var st = _stats_vs_ref(c, ref_host, min(m, REF_ROWS))
            print(t"  m={m}: mean_rel={st.mean_rel} max_abs={st.max_abs}")
            assert_true(
                st.mean_rel < FP32_MEAN_REL_BOUND,
                String("m=", m, " mean_rel above the fp32-exact bound"),
            )
            assert_true(
                st.max_abs < FP32_MAX_ABS_BOUND,
                String("m=", m, " max_abs above the fp32-exact bound"),
            )
            if m == 64:
                for i in range(REF_ROWS * N):
                    c64.raw_store(i, c.raw_load(i))

        print("[phase 2] use_tf32=True, m > 64: TF32 UMMA tile GEMM")
        var ms_large = [65, 128, 512]
        for idx in range(len(ms_large)):
            var m = ms_large[idx]
            var c_buf = ctx.enqueue_create_host_buffer[f32](m * N)
            var c = TileTensor(c_buf, row_major(Coord(Int(m), Idx[N])))
            _run_matmul(a_host, b_dev, c, m, ctx)
            var st = _stats_vs_ref(c, ref_host, REF_ROWS)
            print(t"  m={m}: mean_rel={st.mean_rel} max_abs={st.max_abs}")
            assert_true(
                st.mean_rel > TF32_MEAN_REL_LOW,
                String(
                    "m=",
                    m,
                    " mean_rel below the TF32 band: the default m>64 path",
                    " got more precise — re-decide this contract and update",
                    " the test",
                ),
            )
            assert_true(
                st.mean_rel < TF32_MEAN_REL_HIGH,
                String("m=", m, " mean_rel above the TF32 band"),
            )
            if m == 128:
                for i in range(REF_ROWS * N):
                    c128_tf32.raw_store(i, c.raw_load(i))

        # The dispatch cliff: the same 64 input rows produce different
        # outputs at m=64 (split-K GEMV) vs m=128 (TF32 tile GEMM).
        var sum_rel_diff = Float64(0.0)
        for i in range(REF_ROWS * N):
            var g64 = Float64(c64.raw_load(i))
            var g128 = Float64(c128_tf32.raw_load(i))
            var denom = abs(g64)
            if denom < 1e-30:
                denom = 1e-30
            sum_rel_diff += abs(g64 - g128) / denom
        var mean_rel_diff = sum_rel_diff / Float64(REF_ROWS * N)
        print(t"  cliff m=64 vs m=128[0:64]: mean_rel_diff={mean_rel_diff}")
        assert_true(
            mean_rel_diff > 1e-4,
            String(
                "expected the m=64 vs m=128 outputs to diverge under the",
                " default dispatch (split-K GEMV vs TF32 tile GEMM)",
            ),
        )

        print("[phase 3] use_tf32=False: every M on the IEEE-fp32 GEMV")
        var ms_precise = [64, 65, 128, 512]
        for idx in range(len(ms_precise)):
            var m = ms_precise[idx]
            var c_buf = ctx.enqueue_create_host_buffer[f32](m * N)
            var c = TileTensor(c_buf, row_major(Coord(Int(m), Idx[N])))
            _run_matmul[use_tf32=False](a_host, b_dev, c, m, ctx)
            var st = _stats_vs_ref(c, ref_host, min(m, REF_ROWS))
            print(t"  m={m}: mean_rel={st.mean_rel} max_abs={st.max_abs}")
            assert_true(
                st.mean_rel < FP32_MEAN_REL_BOUND,
                String(
                    "m=",
                    m,
                    " mean_rel above the fp32-exact bound with",
                    " use_tf32=False",
                ),
            )
            assert_true(
                st.max_abs < FP32_MAX_ABS_BOUND,
                String(
                    "m=",
                    m,
                    " max_abs above the fp32-exact bound with",
                    " use_tf32=False",
                ),
            )
            # Dispatch-logic checks: use_tf32=False must leave the m<=64
            # bucket untouched (m=64 case) and must route m>64 to the same
            # GEMV instantiation (m=128 case) — both bit-identical to the
            # default m=64 launch on the shared rows.
            if m == 64 or m == 128:
                var mismatches = 0
                for i in range(REF_ROWS * N):
                    if c.raw_load(i) != c64.raw_load(i):
                        mismatches += 1
                print(t"  bit-exact vs default m=64: {mismatches} diffs")
                assert_true(
                    mismatches == 0,
                    String(
                        "expected use_tf32=False m=",
                        m,
                        " to be bit-identical to the default m=64 launch on",
                        " the shared rows",
                    ),
                )

        print("=== TEST PASSED ===")
