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
"""TFLOPS benchmark for the Apple M5 weight-only NVFP4 (W4A16) matmul.

Times four paths of the W4A16 matmul `out = activation[M,K] @ weight[N,K]^T` and
reports each + the speedup of the materialize path over the fused path:

- `adaptive`: the production dispatch `enqueue_apple_fp4_matmul` -- routes
  `M >= 256` to materialize->dense (with a per-call transient-buffer alloc),
  `M < 256` to the fused cooperative-SMEM kernel.
- `fused`: the fused cooperative-SMEM kernel invoked directly (BM=128 large M,
  BM=64 small M) -- the dequant on the MMA critical path.
- `mat-alloc`: materialize->dense WITH the real per-call `enqueue_create_buffer`
  + free (`_enqueue_apple_fp4_materialize_dense`, what the launcher actually
  runs at large M) -- isolates whether the per-call alloc erodes the win.
- `mat-prealloc`: materialize->dense reusing a buffer allocated ONCE outside the
  hot loop -- the alloc-free ceiling; `mat-alloc` - `mat-prealloc` is the
  per-call alloc/free overhead.

Warmup + hot timing loops, mirroring `bench_apple_gpu_matmul.mojo`.
"""

from std.sys.info import _accelerator_arch
from max.gpu.host import DeviceContext
from std.os import getenv
from std.time import perf_counter

from layout import TileTensor
from layout.tile_layout import row_major

from linalg.fp4_utils import NVFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.apple import enqueue_apple_matmul
from linalg.matmul.gpu.apple.fp4_dequant import enqueue_fp4_materialize
from linalg.matmul.gpu.apple.fp4_matmul import (
    _enqueue_apple_fp4_materialize_dense,
    _launch_apple_fp4_matmul,
    enqueue_apple_fp4_matmul,
)


def _fill_packed(
    packed: MutPointer[UInt8, _],
    scales: MutPointer[Float8_e4m3fn, _],
    npacked: Int,
    nscale: Int,
    seed: UInt64,
):
    """Fill packed FP4 nibbles + FP8 block scales deterministically (xorshift64).
    """
    var state = seed
    for i in range(npacked):
        state ^= state << UInt64(13)
        state ^= state >> UInt64(7)
        state ^= state << UInt64(17)
        packed[i] = UInt8(state & UInt64(0xFF))
    for i in range(nscale):
        state ^= state << UInt64(13)
        state ^= state >> UInt64(7)
        state ^= state << UInt64(17)
        var v = Int(state % UInt64(4)) + 1
        scales[i] = (Float32(v) * Float32(0.5)).cast[.float8_e4m3fn]()


def _bench_fp4_shape(
    m: Int, n: Int, k: Int, ctx: DeviceContext, warmup: Int = 20, hot: Int = 20
) raises:
    """Time fused W4A16 vs materialize+bf16 on one shape; report both + ratio.
    """
    var packed_k = k // 2
    var scale_k = (k + NVFP4_SF_VECTOR_SIZE - 1) // NVFP4_SF_VECTOR_SIZE

    var act_host = ctx.enqueue_create_host_buffer[.bfloat16](m * k)
    var packed_host = ctx.enqueue_create_host_buffer[.uint8](n * packed_k)
    var scale_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](n * scale_k)
    for i in range(m * k):
        act_host[i] = BFloat16(Float32((i % 5) - 2))
    _fill_packed(
        packed_host.unsafe_ptr(),
        scale_host.unsafe_ptr(),
        n * packed_k,
        n * scale_k,
        UInt64(0xF94ED7042B),
    )

    var act_dev = ctx.enqueue_create_buffer[.bfloat16](m * k)
    var packed_dev = ctx.enqueue_create_buffer[.uint8](n * packed_k)
    var scale_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](n * scale_k)
    var wdense_dev = ctx.enqueue_create_buffer[.bfloat16](n * k)
    var out_dev = ctx.enqueue_create_buffer[.float32](m * n)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(packed_dev, packed_host)
    ctx.enqueue_copy(scale_dev, scale_host)

    var act_tt = TileTensor(act_dev.unsafe_ptr(), row_major(m, k)).as_immut()
    var packed_tt = TileTensor(
        packed_dev.unsafe_ptr(), row_major(n, packed_k)
    ).as_immut()
    var scale_tt = TileTensor(
        scale_dev.unsafe_ptr(), row_major(n, scale_k)
    ).as_immut()
    var wdense_tt = TileTensor(wdense_dev.unsafe_ptr(), row_major(n, k))
    var out_tt = TileTensor(out_dev.unsafe_ptr(), row_major(m, n))

    var flops = 2.0 * Float64(m) * Float64(n) * Float64(k)

    # --- adaptive: the production dispatch (materialize w/ alloc for M>=256). ---
    for _ in range(warmup):
        enqueue_apple_fp4_matmul[c_type=DType.float32](
            out_tt, act_tt, packed_tt, scale_tt, ctx
        )
        ctx.synchronize()
    var t0 = perf_counter()
    for _ in range(hot):
        enqueue_apple_fp4_matmul[c_type=DType.float32](
            out_tt, act_tt, packed_tt, scale_tt, ctx
        )
        ctx.synchronize()
    var adaptive_sec = (perf_counter() - t0) / Float64(hot)

    # --- fused: the cooperative-SMEM kernel directly (BM/BK by M, like the
    # dispatch: BM=128/BK=64 for M>=256, BM=64/BK=32 below). ---
    @__parameter
    def _run_fused() raises:
        if m >= 256:
            _launch_apple_fp4_matmul[
                DType.float32, None, BM=128, BK=64, coalesce_scales=True
            ](out_tt, act_tt, packed_tt, scale_tt, m, n, ctx)
        else:
            _launch_apple_fp4_matmul[
                DType.float32, None, BM=64, BK=32, coalesce_scales=False
            ](out_tt, act_tt, packed_tt, scale_tt, m, n, ctx)

    for _ in range(warmup):
        _run_fused()
        ctx.synchronize()
    var t1 = perf_counter()
    for _ in range(hot):
        _run_fused()
        ctx.synchronize()
    var fused_sec = (perf_counter() - t1) / Float64(hot)

    # --- mat-alloc: materialize->dense WITH the real per-call alloc + free. ---
    for _ in range(warmup):
        _enqueue_apple_fp4_materialize_dense[.float32, None](
            out_tt, act_tt, packed_tt, scale_tt, m, n, k, ctx
        )
        ctx.synchronize()
    var t2 = perf_counter()
    for _ in range(hot):
        _enqueue_apple_fp4_materialize_dense[.float32, None](
            out_tt, act_tt, packed_tt, scale_tt, m, n, k, ctx
        )
        ctx.synchronize()
    var matalloc_sec = (perf_counter() - t2) / Float64(hot)

    # --- mat-prealloc: materialize->dense reusing a buffer (alloc-free ceiling). ---
    for _ in range(warmup):
        enqueue_fp4_materialize[.bfloat16](wdense_tt, packed_tt, scale_tt, ctx)
        enqueue_apple_matmul[
            in_type=DType.bfloat16, c_type=DType.float32, transpose_b=True
        ](out_tt, act_tt, wdense_tt.as_immut(), ctx)
        ctx.synchronize()
    var t3 = perf_counter()
    for _ in range(hot):
        enqueue_fp4_materialize[.bfloat16](wdense_tt, packed_tt, scale_tt, ctx)
        enqueue_apple_matmul[
            in_type=DType.bfloat16, c_type=DType.float32, transpose_b=True
        ](out_tt, act_tt, wdense_tt.as_immut(), ctx)
        ctx.synchronize()
    var matprealloc_sec = (perf_counter() - t3) / Float64(hot)

    print(
        "  ",
        m,
        "x",
        n,
        "x",
        k,
        ":  adaptive",
        adaptive_sec * 1000.0,
        "ms (",
        flops / (adaptive_sec * 1e12),
        "TF/s) | fused",
        fused_sec * 1000.0,
        "ms | mat-alloc",
        matalloc_sec * 1000.0,
        "ms | mat-prealloc",
        matprealloc_sec * 1000.0,
        "ms | fused/mat-alloc",
        fused_sec / matalloc_sec,
        "| alloc-overhead-ms",
        (matalloc_sec - matprealloc_sec) * 1000.0,
    )

    _ = act_host^
    _ = packed_host^
    _ = scale_host^
    _ = act_dev^
    _ = packed_dev^
    _ = scale_dev^
    _ = wdense_dev^
    _ = out_dev^


def main() raises:
    comptime if "metal" not in _accelerator_arch():
        print("SKIP: Apple GPU required")
        return
    var ctx = DeviceContext()
    if ctx.compute_capability() != 5:
        print("SKIP: Apple M5 required (compute_capability == 5)")
        return

    print("== bench_apple_fp4_matmul (warmup=20, hot=20)")
    # FLUX.2-ish Linear shapes (square-ish big GEMM, image-gen prefill M).
    _bench_fp4_shape(2048, 3072, 3072, ctx)
    _bench_fp4_shape(2048, 12288, 3072, ctx)  # MLP up-proj-ish
    _bench_fp4_shape(2048, 3072, 12288, ctx)  # MLP down-proj-ish
    _bench_fp4_shape(512, 3072, 3072, ctx)
    # Decode-ish small M (W4A16's intended low-batch regime).
    _bench_fp4_shape(64, 3072, 3072, ctx)
    _bench_fp4_shape(64, 12288, 3072, ctx)
