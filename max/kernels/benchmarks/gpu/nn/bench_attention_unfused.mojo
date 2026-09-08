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
"""Benchmark comparing flash_attention vs mha_gpu_naive vs manual
matmul+softmax attention.

The "manual" path computes softmax(Q @ K^T * scale) @ V using the standard
matmul kernel with a compute epilogue that fuses the scale multiply in-register
(preserving TMA stores on H100) plus the standalone GPU softmax kernel.
"""

from std.collections import Optional
from std.math import ceildiv, rsqrt
from std.sys import (
    align_of,
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
)

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu import *
from max.gpu.host import DeviceContext
from internal_utils import CacheBustingBuffer, arg_parse
from internal_utils._utils import InitializationType
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul import matmul
from linalg.utils import elementwise_compute_lambda_type
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import NullMask
from nn.softmax import softmax_inline


from std.utils.index import Index, IndexList


# ---------------------------------------------------------------------------
# Flash attention benchmark
# ---------------------------------------------------------------------------
def bench_flash[
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    cache_busting: Bool = True,
](
    mut m: Bench,
    seq_len: Int,
    num_keys: Int,
    batch_size: Int,
    bench: Bool,
    ctx: DeviceContext,
) raises:
    var scale = rsqrt(Float32(depth))
    comptime kv_num_heads = num_heads // group

    var q_size = batch_size * seq_len * num_heads * depth
    var k_size = batch_size * num_keys * kv_num_heads * depth
    var v_size = k_size
    var o_size = q_size

    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](
        q_size, simd_size, ctx, cache_busting
    )
    var cb_k = CacheBustingBuffer[qkv_type](
        k_size, simd_size, ctx, cache_busting
    )
    var cb_v = CacheBustingBuffer[qkv_type](
        v_size, simd_size, ctx, cache_busting
    )
    var cb_o = CacheBustingBuffer[qkv_type](
        o_size, simd_size, ctx, cache_busting
    )

    comptime random_distribution = InitializationType.uniform_distribution
    cb_q.init_on_device(random_distribution, ctx)
    cb_k.init_on_device(random_distribution, ctx)
    cb_v.init_on_device(random_distribution, ctx)

    def _run_flash(
        q_ptr: ImmPointer[Scalar[qkv_type], _],
        k_ptr: ImmPointer[Scalar[qkv_type], _],
        v_ptr: ImmPointer[Scalar[qkv_type], _],
        o_ptr: MutPointer[Scalar[qkv_type], _],
        ctx: DeviceContext,
    ) raises {imm}:
        var q = TileTensor(
            q_ptr,
            row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
        )
        var k = TileTensor(
            k_ptr,
            row_major(
                (
                    batch_size,
                    num_keys,
                    Idx[kv_num_heads],
                    Idx[depth],
                )
            ),
        )
        var v = TileTensor(
            v_ptr,
            row_major(
                (
                    batch_size,
                    num_keys,
                    Idx[kv_num_heads],
                    Idx[depth],
                )
            ),
        )
        var output = TileTensor(
            o_ptr,
            row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
        )
        flash_attention(output, q, k, v, NullMask(), scale, ctx)

    def _kernel_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {mut cb_q, mut cb_k, mut cb_v, mut cb_o, imm}:
        _run_flash(
            cb_q.offset_ptr(iteration),
            cb_k.offset_ptr(iteration),
            cb_v.offset_ptr(iteration),
            cb_o.offset_ptr(iteration),
            ctx,
        )

    if bench:

        @always_inline
        def bench_func(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, _kernel_launch, ctx)

        def compute_flops() {imm} -> Int:
            return 4 * batch_size * num_heads * seq_len * num_keys * depth

        m.bench_function(
            bench_func,
            BenchId(
                "flash_attention",
                # fmt: off
                input_id=String(
                    "qkv_type=", qkv_type,
                    "/num_heads=", num_heads,
                    "/depth=", depth,
                    "/seq_len=", seq_len,
                    "/num_keys=", num_keys,
                    "/batch_size=", batch_size,
                ),
                # fmt: on
            ),
            [ThroughputMeasure(BenchMetric.flops, compute_flops())],
        )
    else:
        _run_flash(
            cb_q.unsafe_ptr(),
            cb_k.unsafe_ptr(),
            cb_v.unsafe_ptr(),
            cb_o.unsafe_ptr(),
            ctx,
        )

    ctx.synchronize()

    _ = cb_q
    _ = cb_k
    _ = cb_v
    _ = cb_o


# ---------------------------------------------------------------------------
# Naive GPU MHA benchmark
# ---------------------------------------------------------------------------
def bench_naive[
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    cache_busting: Bool = True,
](
    mut m: Bench,
    seq_len: Int,
    num_keys: Int,
    batch_size: Int,
    bench: Bool,
    ctx: DeviceContext,
) raises:
    var scale = rsqrt(Float32(depth))
    comptime kv_num_heads = num_heads // group

    var q_size = batch_size * seq_len * num_heads * depth
    var k_size = batch_size * num_keys * kv_num_heads * depth
    var v_size = k_size
    var o_size = q_size

    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](
        q_size, simd_size, ctx, cache_busting
    )
    var cb_k = CacheBustingBuffer[qkv_type](
        k_size, simd_size, ctx, cache_busting
    )
    var cb_v = CacheBustingBuffer[qkv_type](
        v_size, simd_size, ctx, cache_busting
    )
    var cb_o = CacheBustingBuffer[qkv_type](
        o_size, simd_size, ctx, cache_busting
    )

    comptime random_distribution = InitializationType.uniform_distribution
    cb_q.init_on_device(random_distribution, ctx)
    cb_k.init_on_device(random_distribution, ctx)
    cb_v.init_on_device(random_distribution, ctx)

    def _run_naive(
        q_ptr: ImmPointer[Scalar[qkv_type], _],
        k_ptr: ImmPointer[Scalar[qkv_type], _],
        v_ptr: ImmPointer[Scalar[qkv_type], _],
        o_ptr: MutPointer[Scalar[qkv_type], _],
        ctx: DeviceContext,
    ) raises {imm}:
        var q = TileTensor(
            q_ptr,
            row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
        )
        var k = TileTensor(
            k_ptr,
            row_major(
                (
                    batch_size,
                    num_keys,
                    Idx[kv_num_heads],
                    Idx[depth],
                )
            ),
        )
        var v = TileTensor(
            v_ptr,
            row_major(
                (
                    batch_size,
                    num_keys,
                    Idx[kv_num_heads],
                    Idx[depth],
                )
            ),
        )
        var output = TileTensor(
            o_ptr,
            row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
        )
        mha_gpu_naive(
            q,
            k,
            v,
            NullMask(),
            output,
            scale,
            batch_size,
            seq_len,
            num_keys,
            num_heads,
            depth,
            group,
            ctx,
        )

    def _kernel_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {mut cb_q, mut cb_k, mut cb_v, mut cb_o, imm}:
        _run_naive(
            cb_q.offset_ptr(iteration),
            cb_k.offset_ptr(iteration),
            cb_v.offset_ptr(iteration),
            cb_o.offset_ptr(iteration),
            ctx,
        )

    if bench:

        @always_inline
        def bench_func(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, _kernel_launch, ctx)

        def compute_flops() {imm} -> Int:
            return 4 * batch_size * num_heads * seq_len * num_keys * depth

        m.bench_function(
            bench_func,
            BenchId(
                "mha_gpu_naive",
                # fmt: off
                input_id=String(
                    "qkv_type=", qkv_type,
                    "/num_heads=", num_heads,
                    "/depth=", depth,
                    "/seq_len=", seq_len,
                    "/num_keys=", num_keys,
                    "/batch_size=", batch_size,
                ),
                # fmt: on
            ),
            [ThroughputMeasure(BenchMetric.flops, compute_flops())],
        )
    else:
        _run_naive(
            cb_q.unsafe_ptr(),
            cb_k.unsafe_ptr(),
            cb_v.unsafe_ptr(),
            cb_o.unsafe_ptr(),
            ctx,
        )

    ctx.synchronize()

    _ = cb_q
    _ = cb_k
    _ = cb_v
    _ = cb_o


# ---------------------------------------------------------------------------
# Manual matmul + softmax benchmark
# ---------------------------------------------------------------------------
def bench_manual[
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    cache_busting: Bool = True,
](
    mut m: Bench,
    seq_len: Int,
    num_keys: Int,
    batch_size: Int,
    bench: Bool,
    ctx: DeviceContext,
) raises:
    var scale = rsqrt(Float32(depth))

    # Data in [batch*heads, seq, depth] layout for direct matmul use.
    var total_heads = batch_size * num_heads

    var q_size = total_heads * seq_len * depth
    var k_size = total_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size

    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](
        q_size, simd_size, ctx, cache_busting
    )
    var cb_k = CacheBustingBuffer[qkv_type](
        k_size, simd_size, ctx, cache_busting
    )
    var cb_v = CacheBustingBuffer[qkv_type](
        v_size, simd_size, ctx, cache_busting
    )
    var cb_o = CacheBustingBuffer[qkv_type](
        o_size, simd_size, ctx, cache_busting
    )

    comptime random_distribution = InitializationType.uniform_distribution
    cb_q.init_on_device(random_distribution, ctx)
    cb_k.init_on_device(random_distribution, ctx)
    cb_v.init_on_device(random_distribution, ctx)

    # Score buffer (intermediate, not cache-busted).
    var score_device = ctx.enqueue_create_buffer[qkv_type](
        total_heads * seq_len * num_keys
    )

    # In-register compute lambda: multiply by scale, kernel does the store.
    @__parameter
    @always_inline
    @__copy_capture(scale)
    def scale_compute_lambda[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        return val * scale.cast[_dtype]()

    comptime scale_fn = Optional[elementwise_compute_lambda_type](
        scale_compute_lambda
    )

    def _run_manual(
        q_base: ImmPointer[Scalar[qkv_type], _],
        k_base: ImmPointer[Scalar[qkv_type], _],
        v_base: ImmPointer[Scalar[qkv_type], _],
        o_base: MutPointer[Scalar[qkv_type], _],
        s_base: MutPointer[Scalar[qkv_type], _],
        ctx: DeviceContext,
    ) raises {imm}:
        # Step 1: Q @ K^T * scale  (per-head 2D matmul with compute
        # epilogue so we keep TMA stores on H100).
        for h in range(total_heads):
            var q_2d = TileTensor(
                q_base + h * seq_len * depth,
                row_major((seq_len, Idx[depth])),
            )
            var k_2d = TileTensor(
                k_base + h * num_keys * depth,
                row_major((num_keys, Idx[depth])),
            )
            var s_2d = TileTensor(
                s_base + h * seq_len * num_keys,
                row_major((seq_len, num_keys)),
            )
            matmul[
                transpose_b=True,
                elementwise_compute_lambda_fn=scale_fn,
                target="gpu",
            ](s_2d, q_2d, k_2d, ctx)

        # Step 2: softmax over the last axis (num_keys).
        var score_3d = TileTensor(
            s_base,
            row_major((total_heads, seq_len, num_keys)),
        )

        @__parameter
        @__copy_capture(score_3d)
        def input_fn[
            _simd_width: Int
        ](coords: Coord) -> SIMD[qkv_type, _simd_width]:
            return score_3d.load[width=_simd_width, alignment=1](coords)

        softmax_inline[qkv_type, 1, 3, input_fn, target="gpu"](
            Coord(total_heads, seq_len, num_keys),
            score_3d,
            2,
            ctx,
        )

        # Step 3: Score @ V  (per-head 2D matmul, no epilogue).
        for h in range(total_heads):
            var s_2d = TileTensor(
                s_base + h * seq_len * num_keys,
                row_major((seq_len, num_keys)),
            )
            var v_2d = TileTensor(
                v_base + h * num_keys * depth,
                row_major((num_keys, Idx[depth])),
            )
            var o_2d = TileTensor(
                o_base + h * seq_len * depth,
                row_major((seq_len, Idx[depth])),
            )
            matmul[target="gpu"](o_2d, s_2d, v_2d, ctx)

    def _kernel_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {mut cb_q, mut cb_k, mut cb_v, mut cb_o, mut score_device, imm,}:
        _run_manual(
            cb_q.offset_ptr(iteration),
            cb_k.offset_ptr(iteration),
            cb_v.offset_ptr(iteration),
            cb_o.offset_ptr(iteration),
            score_device.unsafe_ptr(),
            ctx,
        )

    if bench:

        @always_inline
        def bench_func(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, _kernel_launch, ctx)

        def compute_flops() {imm} -> Int:
            return 4 * batch_size * num_heads * seq_len * num_keys * depth

        m.bench_function(
            bench_func,
            BenchId(
                "manual_matmul",
                # fmt: off
                input_id=String(
                    "qkv_type=", qkv_type,
                    "/num_heads=", num_heads,
                    "/depth=", depth,
                    "/seq_len=", seq_len,
                    "/num_keys=", num_keys,
                    "/batch_size=", batch_size,
                ),
                # fmt: on
            ),
            [ThroughputMeasure(BenchMetric.flops, compute_flops())],
        )
    else:
        _run_manual(
            cb_q.unsafe_ptr(),
            cb_k.unsafe_ptr(),
            cb_v.unsafe_ptr(),
            cb_o.unsafe_ptr(),
            score_device.unsafe_ptr(),
            ctx,
        )

    ctx.synchronize()

    _ = cb_q
    _ = cb_k
    _ = cb_v
    _ = cb_o
    _ = score_device


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() raises:
    comptime qkv_type = get_defined_dtype["qkv_type", .bfloat16]()
    comptime depth = get_defined_int["depth", 512]()
    comptime num_heads = get_defined_int["num_heads", 8]()
    comptime group = get_defined_int["group", 1]()
    comptime cache_busting = get_defined_bool["cache_busting", True]()
    comptime do_bench_flash = get_defined_bool["bench_flash", True]()
    comptime do_bench_naive = get_defined_bool["bench_naive", False]()
    comptime do_bench_manual = get_defined_bool["bench_manual", True]()

    var seq_len = Int(arg_parse("seq_len", 16384))
    var num_keys = Int(arg_parse("num_keys", 16384))
    var batch_size = Int(arg_parse("batch_size", 1))
    var bench = arg_parse("benchmark", True)

    var m = Bench()
    with DeviceContext() as ctx:
        comptime if do_bench_flash:
            bench_flash[qkv_type, depth, num_heads, group, cache_busting](
                m, seq_len, num_keys, batch_size, bench, ctx
            )
        comptime if do_bench_naive:
            bench_naive[qkv_type, depth, num_heads, group, cache_busting](
                m, seq_len, num_keys, batch_size, bench, ctx
            )
        comptime if do_bench_manual:
            bench_manual[qkv_type, depth, num_heads, group, cache_busting](
                m, seq_len, num_keys, batch_size, bench, ctx
            )
    m.dump_report()
