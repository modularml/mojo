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
"""Weight-read-bandwidth microbench for the Apple M5 W8A16 FP8 decode matmuls.

Times the FP8 W8A16 decode paths against the bf16 decode baseline
`enqueue_apple_matmul[bf16, transpose_b=True]` (2 bytes/weight -- what the model
runs today at bf16 decode) at the four FP8-quantized Nemotron-3-Nano-4B-FP8
decode Linears (Mamba in/out-proj, MLP up/down), across BOTH the batch-1 decode
(`M == 1`) and the co-batched decode (`M == 32`) regimes.

The `M == 32` regime is the one the concurrent-decode (c32) serving pathology
lives in. The FP8 checkpoint REGRESSES ~-31% step-median vs bf16 at c32 despite
halved weight bytes -- much worse than the narrow-N materialize route alone
predicts -- so this bench pins WHICH FP8 route burns the loss and by how much:

- `bf16 tiled` (`enqueue_apple_matmul`): the baseline the model runs today.
- `tiled-fp8` (`enqueue_matmul2d_fp8`): the wide-N route; bf16 A x fp8 B -> fp32
  on the native Apple MMA, direct DRAM B feed. Reads N*K weight bytes once.
- `materialize` (`_enqueue_apple_fp8_materialize_dense`): the M>1 narrow-N route
  as it dispatches today -- dequant the fp8 weight to a transient bf16 `[N, K]`
  buffer, then dense bf16 MMA. Moves ~5*N*K bytes (read fp8 + write bf16 + read
  bf16 in the GEMM) = 2.5x the bf16 traffic, every step.
- `gemv` (`enqueue_apple_fp8_gemv`): the register-resident M=1 GEMV (M=1 only;
  a rank-1 update wastes the simdgroup MMA), shown for the M==1 rows only.

Weight bytes dominate all paths at these shapes (the `[M, K]` activation and
`[M, N]` output are negligible even at M=32), and the weight is read ONCE
regardless of M-row validity (a bandwidth-bound matmul reads B once; KB
`apple-m5-c32-decode-gemm-isolation`). So the reported weight-read GB/s (fp8
paths: `N*K` useful bytes; bf16: `2*N*K`) is directly comparable across M -- a
path that moves redundant traffic (materialize) shows a LOW GB/s against the
useful-weight denominator, exposing the waste. Wall-clock (us) is the honest
comparator; ratios vs the bf16 baseline make the routing decision explicit.

Warmup + hot timing loops with a per-iter `synchronize` in warmup and a single
batched `synchronize` for the hot loop mirror `bench_apple_fp4_matmul.mojo` (the
methodology behind the FP4 GEMV's measured 1.53x), so the fixed per-launch sync
overhead is paid by every path and the reported ratio is a conservative lower
bound on the pure kernel delta.
"""

from std.sys.info import _accelerator_arch
from max.gpu.host import DeviceContext
from std.time import perf_counter

from layout import TileTensor
from layout.tile_layout import row_major

from linalg.matmul.gpu.apple import enqueue_apple_matmul
from linalg.matmul.gpu.apple.fp8_gemv import (
    _enqueue_apple_fp8_materialize_dense,
    enqueue_apple_fp8_gemv,
)
from linalg.matmul.gpu.apple.matmul2d_fp8 import enqueue_matmul2d_fp8


def _bench_fp8_shape(
    n: Int,
    k: Int,
    m: Int,
    name: String,
    ctx: DeviceContext,
    warmup: Int = 25,
    hot: Int = 50,
) raises:
    """Time the FP8 decode paths vs the bf16 decode matmul at `[M, N, K]`.

    Reports each path's per-call time (us), weight-read GB/s (fp8 paths count
    `N*K` useful weight bytes, bf16 counts `2*N*K`), and the wall-clock ratio vs
    the bf16 baseline. Timing is data-independent (no path branches on values),
    so the fill is arbitrary. `gemv` runs only at `M == 1` (its contract).
    """
    # Host inputs: values exactly representable in both E4M3 and bf16 (harmless
    # for timing, keeps the two weight buffers numerically consistent).
    var act_host = ctx.enqueue_create_host_buffer[.bfloat16](m * k)
    var w_fp8_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](n * k)
    var w_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](n * k)
    for i in range(m * k):
        act_host[i] = BFloat16(Float32((i % 5) - 2))
    for i in range(n * k):
        var v = Float32((i % 7) - 3) * Float32(0.5)
        w_fp8_host[i] = v.cast[.float8_e4m3fn]()
        w_bf16_host[i] = v.cast[.bfloat16]()

    var act_dev = ctx.enqueue_create_buffer[.bfloat16](m * k)
    var w_fp8_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](n * k)
    var w_bf16_dev = ctx.enqueue_create_buffer[.bfloat16](n * k)
    var out_dev = ctx.enqueue_create_buffer[.float32](m * n)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(w_fp8_dev, w_fp8_host)
    ctx.enqueue_copy(w_bf16_dev, w_bf16_host)

    var act_tt = TileTensor(act_dev.unsafe_ptr(), row_major(m, k)).as_immut()
    var w_fp8_tt = TileTensor(
        w_fp8_dev.unsafe_ptr(), row_major(n, k)
    ).as_immut()
    var w_bf16_tt = TileTensor(
        w_bf16_dev.unsafe_ptr(), row_major(n, k)
    ).as_immut()
    var out_tt = TileTensor(out_dev.unsafe_ptr(), row_major(m, n))

    # --- bf16 decode matmul (2 bytes/weight): the baseline the model runs. ---
    for _ in range(warmup):
        enqueue_apple_matmul[
            in_type=DType.bfloat16,
            c_type=DType.float32,
            transpose_b=True,
        ](out_tt, act_tt, w_bf16_tt, ctx)
        ctx.synchronize()
    var t_bf16 = perf_counter()
    for _ in range(hot):
        enqueue_apple_matmul[
            in_type=DType.bfloat16,
            c_type=DType.float32,
            transpose_b=True,
        ](out_tt, act_tt, w_bf16_tt, ctx)
    ctx.synchronize()
    var bf16_sec = (perf_counter() - t_bf16) / Float64(hot)

    # --- tiled FP8 W8A16 matmul (1 byte/weight, direct DRAM fp8 B feed): the
    # wide-N route. Reads N*K weight bytes once (bandwidth-bound). ---
    for _ in range(warmup):
        enqueue_matmul2d_fp8[c_type=DType.float32](
            out_tt, act_tt, w_fp8_tt, ctx
        )
        ctx.synchronize()
    var t_tiled = perf_counter()
    for _ in range(hot):
        enqueue_matmul2d_fp8[c_type=DType.float32](
            out_tt, act_tt, w_fp8_tt, ctx
        )
    ctx.synchronize()
    var tiled_sec = (perf_counter() - t_tiled) / Float64(hot)

    # --- materialize+dense (fp8 -> transient bf16 [N,K] -> dense bf16 MMA): the
    # M>1 narrow-N route as it dispatches today. Moves ~5*N*K bytes. ---
    for _ in range(warmup):
        _enqueue_apple_fp8_materialize_dense[.float32, None](
            out_tt, act_tt, w_fp8_tt, m, n, k, ctx
        )
        ctx.synchronize()
    var t_mat = perf_counter()
    for _ in range(hot):
        _enqueue_apple_fp8_materialize_dense[.float32, None](
            out_tt, act_tt, w_fp8_tt, m, n, k, ctx
        )
    ctx.synchronize()
    var mat_sec = (perf_counter() - t_mat) / Float64(hot)

    var wbytes = Float64(n) * Float64(k)  # fp8: 1 byte/weight
    var bf16_gbs = (2.0 * wbytes) / (bf16_sec * 1e9)  # bf16: 2 bytes/weight
    var tiled_gbs = wbytes / (tiled_sec * 1e9)
    var mat_gbs = wbytes / (
        mat_sec * 1e9
    )  # useful-weight denom (exposes waste)

    print(
        "  ",
        name,
        " N=",
        n,
        " K=",
        k,
        " M=",
        m,
        ":  bf16 ",
        bf16_sec * 1e6,
        "us (",
        bf16_gbs,
        "GB/s) | tiled-fp8 ",
        tiled_sec * 1e6,
        "us (",
        tiled_gbs,
        "GB/s, ",
        bf16_sec / tiled_sec,
        "x bf16) | materialize ",
        mat_sec * 1e6,
        "us (",
        mat_gbs,
        "GB/s, ",
        bf16_sec / mat_sec,
        "x bf16)",
    )

    # --- fp8 W8A16 GEMV (M=1 only; a rank-1 update wastes the simdgroup MMA). --
    if m == 1:
        for _ in range(warmup):
            enqueue_apple_fp8_gemv[c_type=DType.float32](
                out_tt, act_tt, w_fp8_tt, n, k, ctx
            )
            ctx.synchronize()
        var t_gemv = perf_counter()
        for _ in range(hot):
            enqueue_apple_fp8_gemv[c_type=DType.float32](
                out_tt, act_tt, w_fp8_tt, n, k, ctx
            )
        ctx.synchronize()
        var gemv_sec = (perf_counter() - t_gemv) / Float64(hot)
        var gemv_gbs = wbytes / (gemv_sec * 1e9)
        print(
            "        (M=1 gemv ",
            gemv_sec * 1e6,
            "us (",
            gemv_gbs,
            "GB/s, ",
            bf16_sec / gemv_sec,
            "x bf16))",
        )

    _ = act_host^
    _ = w_fp8_host^
    _ = w_bf16_host^
    _ = act_dev^
    _ = w_fp8_dev^
    _ = w_bf16_dev^
    _ = out_dev^


def main() raises:
    comptime if "metal" not in _accelerator_arch():
        print("SKIP: Apple GPU required")
        return
    var ctx = DeviceContext()
    if ctx.compute_capability() != 5:
        print("SKIP: Apple M5 required (compute_capability == 5)")
        return

    print("== bench_apple_fp8_gemv (decode M in {1, 32}; warmup=25, hot=50)")
    print(
        "   Nemotron-3-Nano-4B-FP8 quantized decode Linears (weight W[N, K]):"
    )
    # (N, K, name). in_proj/mlp_up are wide-N (n>k); out_proj/mlp_down narrow-N.
    for m in [1, 32]:
        print("  -- M =", m, "--")
        _bench_fp8_shape(17504, 3136, m, "mamba_in_proj ", ctx)
        _bench_fp8_shape(3136, 7680, m, "mamba_out_proj", ctx)
        _bench_fp8_shape(12544, 3136, m, "mlp_up        ", ctx)
        _bench_fp8_shape(3136, 12544, m, "mlp_down      ", ctx)
