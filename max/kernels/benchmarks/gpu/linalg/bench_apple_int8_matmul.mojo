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
"""Throughput benchmark for the Apple M5 int8 W8A8 matmul (`int8_matmul.mojo`).

Reports, at the FLUX.2-Klein transformer Linear shapes (M=4096 render tokens):

- the W8A8 GEMM+dequant kernel Tops/s (int8 A x int8 B -> int32 -> per-row/col
  dequant -> bf16), to confirm the in-tree kernel reproduces the ~1.5-1.7x-bf16
  prototype (productization sometimes regresses vs a hot-loop bench);
- the online per-row activation-quant kernel ms, so the TRUE W8A8 wall cost
  (quant + GEMM) is visible relative to the GEMM alone.

Standalone `perf_counter` harness (the Apple matmul convention), per-call timed
(synchronize each iter). Weights + activation are pre-quantized here (the GEMM
bench times only the GEMM+dequant); the quant kernel is timed separately.

Run: mojo max/kernels/benchmarks/gpu/linalg/bench_apple_int8_matmul.mojo
"""

from max.gpu.host import DeviceContext
from std.time import perf_counter

from layout import TileTensor
from layout.tile_layout import row_major

from linalg.matmul.gpu.apple.int8_matmul import (
    enqueue_apple_int8_matmul,
    enqueue_apple_int8_quantize_activation,
)


def _bench_gemm(
    m: Int, n: Int, k: Int, label: String, ctx: DeviceContext
) raises:
    var a = ctx.enqueue_create_buffer[.int8](m * k)
    var b = ctx.enqueue_create_buffer[.int8](n * k)
    var asc = ctx.enqueue_create_buffer[.float32](m)
    var bsc = ctx.enqueue_create_buffer[.float32](n)
    var bias = ctx.enqueue_create_buffer[.bfloat16](1)
    var c = ctx.enqueue_create_buffer[.bfloat16](m * n)
    var a_tt = TileTensor(a.unsafe_ptr(), row_major(m, k)).as_immut()
    var b_tt = TileTensor(b.unsafe_ptr(), row_major(n, k)).as_immut()
    var as_tt = TileTensor(asc.unsafe_ptr(), row_major(m)).as_immut()
    var bs_tt = TileTensor(bsc.unsafe_ptr(), row_major(n)).as_immut()
    var bias_tt = TileTensor(bias.unsafe_ptr(), row_major(1)).as_immut()
    var c_tt = TileTensor(c.unsafe_ptr(), row_major(m, n))

    for _ in range(30):
        enqueue_apple_int8_matmul[c_type=DType.bfloat16](
            c_tt, a_tt, b_tt, as_tt, bs_tt, bias_tt, ctx
        )
        ctx.synchronize()
    var best = Float64(1.0e30)
    for _ in range(3):
        var s = perf_counter()
        for _ in range(20):
            enqueue_apple_int8_matmul[c_type=DType.bfloat16](
                c_tt, a_tt, b_tt, as_tt, bs_tt, bias_tt, ctx
            )
            ctx.synchronize()
        var avg = (perf_counter() - s) / 20.0
        if avg < best:
            best = avg
    var tops = 2.0 * Float64(m) * Float64(n) * Float64(k) / (best * 1e12)
    print(
        "  GEMM ",
        label,
        m,
        "x",
        n,
        "x",
        k,
        ":",
        best * 1000.0,
        "ms",
        tops,
        "Tops/s",
    )
    _ = a^
    _ = b^
    _ = asc^
    _ = bsc^
    _ = bias^
    _ = c^


def _bench_quant(m: Int, k: Int, label: String, ctx: DeviceContext) raises:
    var a = ctx.enqueue_create_buffer[.bfloat16](m * k)
    var q = ctx.enqueue_create_buffer[.int8](m * k)
    var s = ctx.enqueue_create_buffer[.float32](m)
    var a_tt = TileTensor(a.unsafe_ptr(), row_major(m, k)).as_immut()
    var q_tt = TileTensor(q.unsafe_ptr(), row_major(m, k))
    var s_tt = TileTensor(s.unsafe_ptr(), row_major(m))

    for _ in range(30):
        enqueue_apple_int8_quantize_activation[.bfloat16](q_tt, a_tt, s_tt, ctx)
        ctx.synchronize()
    var best = Float64(1.0e30)
    for _ in range(3):
        var st = perf_counter()
        for _ in range(20):
            enqueue_apple_int8_quantize_activation[.bfloat16](
                q_tt, a_tt, s_tt, ctx
            )
            ctx.synchronize()
        var avg = (perf_counter() - st) / 20.0
        if avg < best:
            best = avg
    print("  QUANT", label, "M=", m, "K=", k, ":", best * 1000.0, "ms")
    _ = a^
    _ = q^
    _ = s^


def main() raises:
    var ctx = DeviceContext()
    if ctx.compute_capability() != 5:
        print("SKIP: Apple M5 (compute_capability == 5) required")
        return
    print("=== Apple M5 int8 W8A8, FLUX.2-Klein Linear shapes, M=4096 ===")
    print(
        "--- GEMM+dequant (int8->int32->bf16), vs bf16 ~46-57 TF/s roofline ---"
    )
    _bench_gemm(4096, 27648, 3072, "s.qkv 27648  ", ctx)
    _bench_gemm(4096, 18432, 3072, "ff.in 18432  ", ctx)
    _bench_gemm(4096, 3072, 9216, "ff.out 9216  ", ctx)
    _bench_gemm(4096, 3072, 12288, "s.out 12288  ", ctx)
    _bench_gemm(4096, 3072, 3072, "attn 3072    ", ctx)
    print("--- online per-row activation quant (bf16->int8), M=4096 ---")
    # One quant per distinct activation K (attn/ff/single inputs).
    _bench_quant(4096, 3072, "K=3072 (attn/qkv in)", ctx)
    _bench_quant(4096, 9216, "K=9216 (ff.out in)  ", ctx)
    _bench_quant(4096, 12288, "K=12288 (s.out in)  ", ctx)
    print("DONE")
