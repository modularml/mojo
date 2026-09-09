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
"""Unit tests for the Apple M5 weight-only FP8 (W8A16) matmul.

All shapes require `compute_capability() == 5` (Apple M5, Metal 4). The FP8
sibling of `test_apple_fp4_matmul.mojo`, exercising the two dispatch routes of
`enqueue_apple_fp8_matmul` against an INDEPENDENT fp32 host reference (the
dequant -> bf16-matmul chain, computed in fp32):

- `M == 1` decode: the register-resident W8A16 GEMV (`enqueue_apple_fp8_gemv`).
  This is the star path -- the batch-1 weight-bandwidth win.
- `M > 1` prefill / interim: the materialize (FP8 -> bf16) + dense bf16 MMA path.

The host reference reads the SAME `float8_e4m3fn` weight bytes the device reads
and widens them to f32 (`E4M3 -> f32`, exact on the value stored), so there is no
weight-quantization mismatch between device and reference; the only slack is the
fp32 vs bf16-MMA reduction order and the bf16 activation rounding, bounded by the
same tolerance the FP4 suite uses. The per-tensor scalar `weight_scale` is a
graph-level post-matmul fold (`quant_ops._matmul_float8`), NOT part of the
kernel, so these tests check the RAW `x @ W_fp8^T` the kernel produces (a
`weight_scale`-fold parity test belongs to the Python integration test).

The FP8 weight is the B operand (`out = x @ W^T`, `transpose_b=True`): W is
`[N, K]`, stored `float8_e4m3fn [N, K]` (one byte per element, no packing).
"""

from std.random import random_float64, random_si64, seed
from max.gpu.host import DeviceContext, HostBuffer

from layout import TileTensor
from layout.tile_layout import row_major

from linalg.matmul.gpu.apple import enqueue_apple_matmul
from linalg.matmul.gpu.apple.fp8_gemv import (
    enqueue_apple_fp8_gemv,
    enqueue_apple_fp8_matmul,
    enqueue_fp8_materialize,
)
from linalg.matmul.gpu.apple.matmul2d_fp8 import enqueue_matmul2d_fp8


# ===----------------------------------------------------------------------=== #
# Shared host helpers
# ===----------------------------------------------------------------------=== #


def _fill_random_fp8_weight(
    weight: HostBuffer[.float8_e4m3fn],
    N: Int,
    K: Int,
):
    """Fills `weight` `[N, K]` with random `float8_e4m3fn` values in ~[-4, 4].

    The range spans several E4M3 exponents (exercises the widening cast beyond
    the tiny-magnitude regime) while staying well inside E4M3's ~+-448 range so
    the matmul output magnitude is sane for the bf16 MMA path.
    """
    for i in range(N * K):
        weight[i] = (random_float64() * 8.0 - 4.0).cast[.float8_e4m3fn]()


def _check_vs_host_ref[
    c_type: DType
](
    out_host: HostBuffer[c_type],
    act_host: HostBuffer[.bfloat16],
    weight_host: HostBuffer[.float8_e4m3fn],
    M: Int,
    N: Int,
    K: Int,
    name: String,
) raises:
    """Asserts `out_host == act @ dequant(weight)^T` (independent fp32 ref).

    The fp32 host reference widens the FP8 weight straight to f32 (`E4M3 -> f32`,
    exact on the stored value) and accumulates in fp32. Keep `N*K` small (the host
    loop is O(M*N*K)). Tolerance matches the bf16-MMA bound the FP4 suite uses --
    loose enough for BOTH the fp32-accumulate GEMV (`M == 1`) and the bf16 MMA
    (`M > 1`) routes.
    """
    var pass_ = True
    for i in range(M):
        for j in range(N):
            var acc = Float32(0)
            for k in range(K):
                var av = Float32(act_host[i * K + k])
                var wv = weight_host[j * K + k].cast[.float32]()
                acc += av * wv
            var got = Float32(out_host[i * N + j])
            if abs(got - acc) > Float32(1e-2) + Float32(1.6e-2) * abs(acc):
                if pass_:
                    print("FAIL[", name, "]:", i, j, "got", got, "exp", acc)
                pass_ = False
    if not pass_:
        raise Error("FAILED (", name, "; see FAIL lines above)")


def _run_fp8_matmul[
    c_type: DType
](ctx: DeviceContext, M: Int, N: Int, K: Int, name: String) raises:
    """Run `enqueue_apple_fp8_matmul` and check vs the fp32 host reference.

    `out = activation[M, K] @ weight[N, K]^T` (transpose_b=True). Routes through
    the production launcher, so `M == 1` exercises the GEMV and `M > 1` the
    materialize -> dense path. Runs for the requested output `c_type`.
    """
    print("== fp8-matmul", name, M, "x", N, "x", K, "c_type", c_type)

    # Host inputs.
    var act_host = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    var weight_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](N * K)
    for i in range(M * K):
        act_host[i] = random_si64(Int64(-2), Int64(2)).cast[.bfloat16]()
    _fill_random_fp8_weight(weight_host, N, K)

    # Device buffers.
    var act_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var weight_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](N * K)
    var out_dev = ctx.enqueue_create_buffer[c_type](M * N)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(weight_dev, weight_host)

    var act_tt = TileTensor(act_dev.unsafe_ptr(), row_major(M, K)).as_immut()
    var weight_tt = TileTensor(
        weight_dev.unsafe_ptr(), row_major(N, K)
    ).as_immut()
    var out_tt = TileTensor(out_dev.unsafe_ptr(), row_major(M, N))

    enqueue_apple_fp8_matmul[c_type=c_type](out_tt, act_tt, weight_tt, ctx)

    var out_host = ctx.enqueue_create_host_buffer[c_type](M * N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    # DRIV-199: keep device buffers alive past synchronize.
    _ = act_dev^
    _ = weight_dev^
    _ = out_dev^

    _check_vs_host_ref[c_type](out_host, act_host, weight_host, M, N, K, name)
    print("PASS")


def _run_fp8_gemv_direct[
    c_type: DType
](ctx: DeviceContext, N: Int, K: Int, name: String) raises:
    """Call `enqueue_apple_fp8_gemv` directly (M == 1) and check vs host ref.

    Covers the GEMV entry point explicitly (not only via the launcher's `m == 1`
    route), so the decode kernel stays validated even if the launcher dispatch
    changes.
    """
    print("== fp8-gemv", name, "1 x", N, "x", K, "c_type", c_type)

    var act_host = ctx.enqueue_create_host_buffer[.bfloat16](K)
    var weight_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](N * K)
    for i in range(K):
        act_host[i] = random_si64(Int64(-2), Int64(2)).cast[.bfloat16]()
    _fill_random_fp8_weight(weight_host, N, K)

    var act_dev = ctx.enqueue_create_buffer[.bfloat16](K)
    var weight_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](N * K)
    var out_dev = ctx.enqueue_create_buffer[c_type](N)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(weight_dev, weight_host)

    var act_tt = TileTensor(act_dev, row_major(1, K)).as_immut()
    var weight_tt = TileTensor(weight_dev, row_major(N, K)).as_immut()
    var out_tt = TileTensor(out_dev, row_major(1, N))

    enqueue_apple_fp8_gemv[c_type=c_type](out_tt, act_tt, weight_tt, N, K, ctx)

    var out_host = ctx.enqueue_create_host_buffer[c_type](N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    _ = act_dev^
    _ = weight_dev^
    _ = out_dev^

    _check_vs_host_ref[c_type](out_host, act_host, weight_host, 1, N, K, name)
    print("PASS")


def _run_tiled_fp8[
    c_type: DType,
    check_host: Bool = True,
](ctx: DeviceContext, M: Int, N: Int, K: Int, name: String) raises:
    """Run the tiled FP8 matmul (`enqueue_matmul2d_fp8`) and check correctness.

    `out = activation[M, K] @ weight[N, K]^T` (transpose_b=True), bf16 A x fp8 B
    -> fp32 on the native Apple MMA. Any M/N/K (bounded MMA + guarded store).

    - `check_host=True` (reduced shapes): assert the tiled output vs an
      INDEPENDENT fp32 host reference (the strongest check; the host loop is
      O(M*N*K), so keep N*K small).
    - `check_host=False` (real production shapes, host loop too slow): assert
      parity vs the materialize -> dense bf16 oracle
      (`enqueue_fp8_materialize` + `enqueue_apple_matmul`). Both paths round the
      identical E4M3 weight to the same bf16 (bf16 is exact for E4M3) and use a
      bf16 MMA, so they differ only in fp32 reduction ORDER -- a tight tolerance.
    """
    print(
        "== tiled-fp8",
        name,
        M,
        "x",
        N,
        "x",
        K,
        "c_type",
        c_type,
    )

    var act_host = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    var weight_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](N * K)
    for i in range(M * K):
        act_host[i] = random_si64(Int64(-2), Int64(2)).cast[.bfloat16]()
    _fill_random_fp8_weight(weight_host, N, K)

    var act_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var weight_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](N * K)
    var out_dev = ctx.enqueue_create_buffer[c_type](M * N)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(weight_dev, weight_host)

    var act_tt = TileTensor(act_dev.unsafe_ptr(), row_major(M, K)).as_immut()
    var weight_tt = TileTensor(
        weight_dev.unsafe_ptr(), row_major(N, K)
    ).as_immut()
    var out_tt = TileTensor(out_dev.unsafe_ptr(), row_major(M, N))

    enqueue_matmul2d_fp8[c_type=c_type](out_tt, act_tt, weight_tt, ctx)

    var out_host = ctx.enqueue_create_host_buffer[c_type](M * N)
    ctx.enqueue_copy(out_host, out_dev)

    comptime if check_host:
        ctx.synchronize()
        _ = act_dev^
        _ = weight_dev^
        _ = out_dev^
        _check_vs_host_ref[c_type](
            out_host, act_host, weight_host, M, N, K, name
        )
        print("PASS")
    else:
        # Parity vs the materialize -> dense bf16 oracle (GPU-only; host loop is
        # too slow at production N*K). Reduction-order diff only.
        var wdense_dev = ctx.enqueue_create_buffer[.bfloat16](N * K)
        var oracle_dev = ctx.enqueue_create_buffer[c_type](M * N)
        var wdense_tt = TileTensor(wdense_dev, row_major(N, K))
        enqueue_fp8_materialize[.bfloat16](wdense_tt, weight_tt, ctx)
        var oracle_tt = TileTensor(oracle_dev.unsafe_ptr(), row_major(M, N))
        enqueue_apple_matmul[
            in_type=.bfloat16, c_type=c_type, transpose_b=True
        ](oracle_tt, act_tt, wdense_tt.as_immut(), ctx)
        var oracle_host = ctx.enqueue_create_host_buffer[c_type](M * N)
        ctx.enqueue_copy(oracle_host, oracle_dev)
        ctx.synchronize()
        _ = act_dev^
        _ = weight_dev^
        _ = out_dev^
        _ = wdense_dev^
        _ = oracle_dev^
        var pass_ = True
        for i in range(M * N):
            var f = Float32(out_host[i])
            var o = Float32(oracle_host[i])
            if abs(f - o) > Float32(1e-2) + Float32(1.6e-2) * abs(o):
                if pass_:
                    print(
                        "PARITY FAIL[",
                        name,
                        "]: idx",
                        i,
                        "tiled",
                        f,
                        "oracle",
                        o,
                    )
                pass_ = False
        if not pass_:
            raise Error("FAILED parity (", name, "; see lines above)")
        print("PARITY PASS vs materialize->dense")


# ===----------------------------------------------------------------------=== #
# Test entry points
# ===----------------------------------------------------------------------=== #


def test_decode_gemv(ctx: DeviceContext) raises:
    """M == 1 decode (the GEMV): production-shaped 1 x hidden, bf16 + fp32 out.
    """
    seed(0)
    # Direct GEMV entry point, K a multiple of TILE_K=16.
    _run_fp8_gemv_direct[.float32](ctx, 64, 256, "gemv-tiny")
    _run_fp8_gemv_direct[.bfloat16](ctx, 64, 256, "gemv-tiny-bf16")
    # Ragged N (per-warp N guard), K % 16 == 0.
    _run_fp8_gemv_direct[.float32](ctx, 200, 128, "gemv-ragged-n")
    # Wider N / deeper K (1 x hidden decode magnitudes).
    _run_fp8_gemv_direct[.float32](ctx, 512, 256, "gemv-flux-ish")
    _run_fp8_gemv_direct[.bfloat16](ctx, 1024, 512, "gemv-wide-bf16")
    # Via the production launcher (m == 1 route) -- same shapes, both dtypes.
    _run_fp8_matmul[.float32](ctx, 1, 512, 256, "launch-gemv-f32")
    _run_fp8_matmul[.bfloat16](ctx, 1, 512, 256, "launch-gemv-bf16")


def test_multi_row_launcher(ctx: DeviceContext) raises:
    """M > 1 via the production launcher: routes to the tiled FP8 matmul.

    `enqueue_apple_fp8_matmul` sends every `M > 1` shape (wide-N AND narrow-N) to
    `enqueue_matmul2d_fp8`; the materialize -> dense path is retained only for a
    fused epilogue or pre-M5. Checks vs the independent fp32 host reference.
    """
    seed(1)
    # Wide-N (n > k).
    _run_fp8_matmul[.float32](ctx, 16, 128, 64, "multi-row-f32")
    _run_fp8_matmul[.bfloat16](ctx, 16, 128, 64, "multi-row-bf16")
    # Ragged M and N (tile edge logic).
    _run_fp8_matmul[.float32](ctx, 100, 200, 64, "multi-row-ragged")
    # Narrow-N (n <= k) at co-batched M=32 -- the dispatch fix routes this to
    # tiled (was materialize -> dense). Reduced N*K for the host-ref check.
    _run_fp8_matmul[.float32](ctx, 32, 64, 256, "multi-row-narrow-n-m32")
    _run_fp8_matmul[.bfloat16](ctx, 32, 96, 288, "multi-row-narrow-n-bf16")


def test_k_tail_edge(ctx: DeviceContext) raises:
    """K not a multiple of TILE_K=16: exercises the GEMV width-1 scalar tail."""
    seed(2)
    # M == 1 GEMV tail (one full 16-chunk + partial, and sub-16 K).
    _run_fp8_gemv_direct[.float32](ctx, 64, 80, "gemv-k80")
    _run_fp8_gemv_direct[.float32](ctx, 64, 48, "gemv-k48")
    _run_fp8_gemv_direct[.float32](ctx, 64, 24, "gemv-k24")
    _run_fp8_gemv_direct[.bfloat16](ctx, 200, 40, "gemv-k40-bf16")
    # M > 1 with ragged K (materialize handles any K per-thread bounds).
    _run_fp8_matmul[.float32](ctx, 16, 64, 48, "multi-row-k48")


def test_tiled_matmul2d(ctx: DeviceContext) raises:
    """Tiled `matmul2d` FP8 path (Slice 4): cooperative-SMEM + per-lane reg.

    Interior-only (N tile-aligned, K a multiple of the strip depth). Reduced
    shapes assert vs the independent fp32 host reference; the real Nemotron
    large-N/small-K decode Linears assert parity vs materialize -> dense (the
    O(M*N*K) host loop is too slow at production N*K).
    """
    seed(3)
    # M=1 decode (the star path), reduced N*K, independent fp32 host ref.
    _run_tiled_fp8[.float32](ctx, 1, 64, 64, "decode-tiny")
    _run_tiled_fp8[.float32](ctx, 1, 256, 128, "decode-multistrip")
    _run_tiled_fp8[.bfloat16](ctx, 1, 256, 128, "decode-bf16-out")
    _run_tiled_fp8[.float32](ctx, 1, 544, 64, "decode-ragged-n")
    _run_tiled_fp8[.float32](ctx, 1, 128, 72, "decode-k-tail")
    # Small M>1 + partial-M/N tiles (store guard + bounded MMA).
    _run_tiled_fp8[.float32](ctx, 16, 128, 192, "small-m")
    _run_tiled_fp8[.bfloat16](ctx, 40, 96, 128, "partial-m-ragged-n-bf16")
    _run_tiled_fp8[.float32](ctx, 100, 256, 64, "partial-m-multi-tile")
    # Narrow-N (n < k) at co-batched M=32 -- the regime the dispatch fix newly
    # routes here. Reduced N*K asserts vs the independent fp32 host reference.
    _run_tiled_fp8[.float32](ctx, 32, 96, 256, "narrow-n-m32")
    _run_tiled_fp8[.bfloat16](ctx, 32, 128, 320, "narrow-n-m32-bf16")
    # Real Nemotron decode Linears (dispatch-routed + bench targets): parity vs
    # materialize -> dense (the O(M*N*K) host loop is too slow at production N*K).
    # Wide-N at M=1 (the star decode path) and narrow-N at co-batched M=32 (the
    # shapes the dispatch fix moves off the ~5-9x-slower materialize route).
    _run_tiled_fp8[.float32, check_host=False](
        ctx, 1, 17504, 3136, "in_proj-real"
    )
    _run_tiled_fp8[.float32, check_host=False](
        ctx, 1, 12544, 3136, "mlp_up-real"
    )
    _run_tiled_fp8[.float32, check_host=False](
        ctx, 32, 3136, 7680, "out_proj-m32-real"
    )
    _run_tiled_fp8[.float32, check_host=False](
        ctx, 32, 3136, 12544, "mlp_down-m32-real"
    )


def main() raises:
    var ctx = DeviceContext()
    if ctx.compute_capability() != 5:
        print("SKIP: Apple M5 (compute_capability == 5) required")
        return
    test_decode_gemv(ctx)
    test_multi_row_launcher(ctx)
    test_k_tail_edge(ctx)
    test_tiled_matmul2d(ctx)
    print("ALL TESTS PASSED")
