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
"""TFLOPS benchmark for the Apple M5 simdgroup-tiled matmul kernel.

Calls apple_matmul_kernel directly with explicit warmup + hot timing loops.
"""

from std.collections import Optional
from std.sys.info import _accelerator_arch
from max.gpu.host import DeviceContext
from std.os import getenv
from std.time import perf_counter
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul.gpu.apple.matmul_kernel import enqueue_apple_matmul


def _fill_small_int[
    dtype: DType
](buf: MutPointer[Scalar[dtype], _], count: Int, seed: UInt64):
    """Fill `buf` with deterministic uniform values in `{-2, -1, 0, 1, 2}`.

    Inlined xorshift64 keeps the sequence reproducible across runs. With
    inputs bounded to `{-2..2}`, CPU reference can use a tight absolute
    tolerance.
    """
    var state = seed
    for i in range(count):
        state ^= state << UInt64(13)
        state ^= state >> UInt64(7)
        state ^= state << UInt64(17)
        var v = Int(state % UInt64(5)) - 2
        buf[i] = Scalar[dtype](v)


def _verify[
    in_type: DType, transpose_b: Bool
](
    a_host: MutPointer[Scalar[in_type], MutAnyOrigin],
    b_host: MutPointer[Scalar[in_type], MutAnyOrigin],
    d_host: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    """Sample-verify `D = A @ B` (or `A @ B.T`) on CPU vs `d_host`.

    Strided sample at up to 64x64 output positions; fp32 reference.
    Prints `verify: OK` or `verify: FAIL ...`.
    """
    comptime kTol: Float32 = 1.0e-3
    comptime kMaxSampleM: Int = 64
    comptime kMaxSampleN: Int = 64
    var sample_m = min(kMaxSampleM, m)
    var sample_n = min(kMaxSampleN, n)
    var step_m = max(1, m // sample_m)
    var step_n = max(1, n // sample_n)

    var mismatch_count = 0
    var max_abs_err: Float32 = 0.0
    var first_i = -1
    var first_j = -1
    var first_expected: Float32 = 0.0
    var first_got: Float32 = 0.0

    for ii in range(sample_m):
        var i = ii * step_m
        if i >= m:
            break
        for jj in range(sample_n):
            var j = jj * step_n
            if j >= n:
                break
            var expected: Float32 = 0.0
            for kk in range(k):
                var a_val = Float32(a_host[i * k + kk])
                var b_val: Float32 = 0.0
                comptime if transpose_b:
                    # B stored as (N, K), row-major: B[j, kk].
                    b_val = Float32(b_host[j * k + kk])
                else:
                    # B stored as (K, N), row-major: B[kk, j].
                    b_val = Float32(b_host[kk * n + j])
                expected += a_val * b_val
            var got = d_host[i * n + j]
            var diff = expected - got
            var err = -diff if diff < Float32(0.0) else diff
            if err > max_abs_err:
                max_abs_err = err
            if err > kTol:
                if mismatch_count == 0:
                    first_i = i
                    first_j = j
                    first_expected = expected
                    first_got = got
                mismatch_count += 1

    if mismatch_count == 0:
        print("    verify: OK   max_abs_err =", max_abs_err)
    else:
        print(
            "    verify: FAIL",
            mismatch_count,
            "mismatches /",
            sample_m * sample_n,
            "samples; max_abs_err =",
            max_abs_err,
            "; first [",
            first_i,
            ",",
            first_j,
            "] expected =",
            first_expected,
            "got =",
            first_got,
        )


def _bench_shape[
    in_type: DType, transpose_b: Bool
](
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
    warmup: Int = 30,
    hot: Int = 20,
    verify: Bool = True,
    force_split_k: Optional[Bool] = None,
) raises -> Float64:
    var b_size = n * k if transpose_b else k * n
    var a_host = ctx.enqueue_create_host_buffer[in_type](m * k)
    var b_host = ctx.enqueue_create_host_buffer[in_type](b_size)
    _fill_small_int[in_type](a_host.unsafe_ptr(), m * k, UInt64(0xA17ED1042B))
    _fill_small_int[in_type](b_host.unsafe_ptr(), b_size, UInt64(0xB17ED7042B))

    var a_dev = ctx.enqueue_create_buffer[in_type](m * k)
    var b_dev = ctx.enqueue_create_buffer[in_type](b_size)
    var d_dev = ctx.enqueue_create_buffer[.float32](m * n)
    var d_host = ctx.enqueue_create_host_buffer[.float32](m * n)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var b_rows = n if transpose_b else k
    var b_cols = k if transpose_b else n
    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(m, k)).as_immut()
    var b_tt = TileTensor(
        b_dev.unsafe_ptr(), row_major(b_rows, b_cols)
    ).as_immut()
    var d_tt = TileTensor(d_dev.unsafe_ptr(), row_major(m, n))

    if verify:
        enqueue_apple_matmul[in_type=in_type, transpose_b=transpose_b](
            d_tt, a_tt, b_tt, ctx, force_split_k
        )
        ctx.enqueue_copy(d_host, d_dev)
        ctx.synchronize()
        _verify[in_type, transpose_b](
            a_host.unsafe_ptr().as_unsafe_any_origin(),
            b_host.unsafe_ptr().as_unsafe_any_origin(),
            d_host.unsafe_ptr().as_unsafe_any_origin(),
            m,
            n,
            k,
        )

    # Warmup runs (untimed).
    for _ in range(warmup):
        enqueue_apple_matmul[in_type=in_type, transpose_b=transpose_b](
            d_tt, a_tt, b_tt, ctx, force_split_k
        )
        ctx.synchronize()

    # Hot runs (timed).
    var start = perf_counter()
    for _ in range(hot):
        enqueue_apple_matmul[in_type=in_type, transpose_b=transpose_b](
            d_tt, a_tt, b_tt, ctx, force_split_k
        )
        ctx.synchronize()
    var elapsed = perf_counter() - start

    var avg_sec = elapsed / Float64(hot)
    var flops = 2.0 * Float64(m) * Float64(n) * Float64(k)
    var tflops = flops / (avg_sec * 1e12)

    var tb_str = String("T") if transpose_b else String("N")
    var route = String("auto")
    if force_split_k:
        route = String("split") if force_split_k.value() else String("single")
    print(
        " ",
        in_type,
        " ",
        m,
        "x",
        n,
        "x",
        k,
        " N" + tb_str,
        "[" + route + "]:",
        " avg",
        avg_sec * 1000.0,
        "ms,",
        tflops,
        "TFLOPS",
    )

    # Keep buffers alive until timing is done.
    _ = a_host^
    _ = b_host^
    _ = a_dev^
    _ = b_dev^
    _ = d_dev^
    _ = d_host^
    return avg_sec


def _bench_split_compare[
    in_type: DType, transpose_b: Bool
](m: Int, n: Int, k: Int, ctx: DeviceContext) raises:
    """Forced single-pass vs forced split-K on one shape; report the speedup.

    Folds in the former standalone `bench_apple_split_k.mojo`. These small-M*N
    / deep-K shapes auto-route to split-K, so the single-pass baseline must be
    forced (`force_split_k=False`) -- otherwise both sides run split-K.
    """
    var single = _bench_shape[in_type, transpose_b](
        m, n, k, ctx, verify=False, force_split_k=False
    )
    var split = _bench_shape[in_type, transpose_b](
        m, n, k, ctx, verify=False, force_split_k=True
    )
    print("    -> speedup (single / split):", single / split, "x")


def main() raises:
    comptime if "metal" not in _accelerator_arch():
        print("SKIP: Apple GPU required")
        return
    var ctx = DeviceContext()
    if ctx.compute_capability() != 5:
        print("SKIP: Apple M5 required (compute_capability == 5)")
        return

    # Verification on by default; `BENCH_VERIFY=0` to skip.
    var verify = getenv("BENCH_VERIFY", "1") != "0"
    print(
        "== bench_apple_gpu_matmul (warmup=30, hot=20, verify=",
        verify,
        ")",
    )

    # Peak-perf anchor: 8192^3 fp16 NT (research-port reference).
    _ = _bench_shape[.float16, True](8192, 8192, 8192, ctx, verify=verify)
    # NN/NT asymmetry check.
    _ = _bench_shape[.float16, False](8192, 8192, 8192, ctx, verify=verify)
    # bf16 same-shape comparison.
    _ = _bench_shape[.bfloat16, True](8192, 8192, 8192, ctx, verify=verify)
    # fp32 sanity (expected slower).
    _ = _bench_shape[.float32, True](8192, 8192, 8192, ctx, verify=verify)
    # Llama-3-ish MLP up-proj (prefill).
    _ = _bench_shape[.float16, True](2048, 14336, 4096, ctx, verify=verify)
    # MLP down-proj.
    _ = _bench_shape[.float16, True](2048, 4096, 14336, ctx, verify=verify)
    # Ragged shape.
    _ = _bench_shape[.float16, True](100, 1003, 97, ctx, verify=verify)
    # Small square.
    _ = _bench_shape[.float16, True](512, 512, 512, ctx, verify=verify)

    # Single-pass vs split-K on under-occupied (small-M*N / deep-K) shapes,
    # forced via the `force_split_k` flag (folded in from bench_apple_split_k).
    print("== single-pass vs split-K (forced via force_split_k):")
    _bench_split_compare[.float16, False](64, 64, 8192, ctx)
    _bench_split_compare[.float16, False](64, 64, 16384, ctx)
    _bench_split_compare[.float16, False](128, 128, 8192, ctx)
    _bench_split_compare[.float16, False](256, 256, 8192, ctx)
    _bench_split_compare[.float16, False](64, 256, 8192, ctx)
