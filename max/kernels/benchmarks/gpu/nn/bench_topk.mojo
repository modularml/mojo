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

from std.math import ceildiv, cos, iota, log, pi, sin
from std.random import random_float64

from max.algorithm.reduction import max as reduce_max
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import arg_parse

from layout import Coord, Idx, TileTensor, coord_to_index_list, row_major

from nn.topk import _top_k_cpu, _topk_gpu, _topk_topp_sampling_fi, topk_gpu
from nn.sampling import topk_topp_masked_probs, topk_topp_sampling_from_prob
from nn.topk_bitonic import (
    PERSISTENT_TOPK_MAX_N,
    persistent_topk_block,
)
from std.testing import assert_almost_equal, assert_equal

from std.utils import IndexList
from std.sys import (
    get_defined_int,
    get_defined_bool,
    get_defined_dtype,
)
from std.sys.info import has_apple_gpu_accelerator, size_of


def bench_topk_batched[
    dtype: DType, out_idx_type: DType, rank: Int
](
    ctx: DeviceContext,
    mut m: Bench,
    test_case: TestCase,
    fill_fn_name: String,
    top_p: Float32 = 1.0,
) raises:
    # Fetch arguments

    var batch_size = test_case.batch_size
    var N = test_case.N
    var K = test_case.K
    var block_size = test_case.block_size
    var num_blocks_per_input = test_case.num_blocks_per_input
    comptime largest = test_case.largest
    comptime sampling = test_case.sampling
    # Instantiate data in host memory
    var out_idx_len = 1 if sampling else K

    var in_size = batch_size * N
    var topk_vals_size = batch_size * K
    var topk_idxs_size = batch_size * out_idx_len

    var in_buffer_ptr = List(length=in_size, fill=Scalar[dtype](0))
    var topk_vals_ptr = List(length=topk_vals_size, fill=Scalar[dtype](0))
    var topk_idxs_ptr = List(
        length=topk_idxs_size, fill=Scalar[out_idx_type](0)
    )

    var in_buffer = TileTensor(
        in_buffer_ptr,
        row_major(batch_size, N),
    )
    var topk_vals = TileTensor(
        topk_vals_ptr,
        row_major(batch_size, K),
    )
    var topk_idxs = TileTensor(
        topk_idxs_ptr, row_major(batch_size, out_idx_len)
    )

    # Fill the buffer
    fill_buffer[rank, dtype](in_buffer, fill_fn_name)

    # Move data to device
    var device_in_buffer = ctx.enqueue_create_buffer[dtype](in_size)
    var device_out_vals_buffer = ctx.enqueue_create_buffer[dtype](
        topk_vals_size
    )
    var device_out_idxs_buffer = ctx.enqueue_create_buffer[out_idx_type](
        topk_idxs_size
    )

    var device_in = TileTensor(device_in_buffer, row_major(batch_size, N))
    var device_out_vals = TileTensor(
        device_out_vals_buffer,
        row_major(batch_size, K),
    )
    var device_out_idxs = TileTensor(
        device_out_idxs_buffer, row_major(batch_size, out_idx_len)
    )

    if not num_blocks_per_input:
        num_blocks_per_input = min(ceildiv(N, block_size), 8)

    var local_topk_size = batch_size * num_blocks_per_input * K
    var device_local_topk_vals_buffer = ctx.enqueue_create_buffer[dtype](
        local_topk_size
    )
    var device_local_topk_idxs_buffer = ctx.enqueue_create_buffer[out_idx_type](
        local_topk_size
    )

    var device_local_topk_vals = TileTensor(
        device_local_topk_vals_buffer,
        row_major(batch_size, num_blocks_per_input * K),
    )
    var device_local_topk_idxs = TileTensor(
        device_local_topk_idxs_buffer,
        row_major(batch_size, num_blocks_per_input * K),
    )

    ctx.enqueue_copy(device_in_buffer, in_buffer_ptr)

    var K_dev_buffer = ctx.enqueue_create_buffer[.int64](batch_size)
    var k = TileTensor(K_dev_buffer, row_major(batch_size))
    var K_host_ptr = List(length=batch_size, fill=Int64(K))
    var K_host_buffer = TileTensor(K_host_ptr, row_major(batch_size))

    var max_k = Int(
        reduce_max(
            Span(
                unsafe_ptr=K_host_buffer.ptr,
                length=K_host_buffer.num_elements(),
            )
        )
    )

    ctx.enqueue_copy(K_dev_buffer, K_host_ptr)

    # Top-p buffer.
    var top_p_dev_buffer = ctx.enqueue_create_buffer[.float32](batch_size)
    var top_p_host_ptr = List(length=batch_size, fill=top_p)
    ctx.enqueue_copy(top_p_dev_buffer, top_p_host_ptr)
    var top_p_tt = TileTensor(top_p_dev_buffer, row_major(batch_size))

    ctx.synchronize()

    @always_inline
    def bench_func(
        mut b: Bencher,
    ) {var K_dev_buffer, var top_p_dev_buffer, imm,}:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            _topk_gpu[sampling=sampling, largest=largest](
                ctx,
                max_k,
                device_in,
                device_local_topk_vals,
                device_local_topk_idxs,
                device_out_vals,
                device_out_idxs,
                k=TileTensor(k.ptr, row_major(Int64(batch_size)))
                .as_unsafe_any_origin()
                .as_immut(),
                block_size=block_size,
                num_blocks_per_input=num_blocks_per_input,
                top_p=top_p_tt.as_unsafe_any_origin().as_immut(),
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    var kernel_name = String(
        "bench-topk", "/N=", N, "/K=", K, "/batch_size=", batch_size
    )

    var num_bytes = device_in.num_elements() * size_of[dtype]()
    m.bench_function(
        bench_func,
        BenchId(kernel_name),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )

    # Copy results back to host
    ctx.enqueue_copy(topk_vals_ptr, device_out_vals_buffer)
    ctx.enqueue_copy(topk_idxs_ptr, device_out_idxs_buffer)
    ctx.synchronize()

    # ASSERT equality with CPU topk kernel reference
    comptime if not sampling:
        var topk_vals_cpu_ptr = List(
            length=topk_vals_size, fill=Scalar[dtype](0)
        )
        var topk_idxs_cpu_ptr = List(length=topk_vals_size, fill=Int64(0))
        var topk_vals_cpu = TileTensor(
            topk_vals_cpu_ptr, row_major(batch_size, K)
        )
        var topk_idxs_cpu = TileTensor(
            topk_idxs_cpu_ptr, row_major(batch_size, K)
        )

        _top_k_cpu[dtype=dtype, out_idx_type=DType.int64, largest=largest](
            in_buffer,
            max_k,
            rank - 1,
            topk_vals_cpu,
            topk_idxs_cpu,
            1,
            True,
            k=K_host_buffer.as_unsafe_any_origin().as_immut(),
        )

        for i in range(topk_vals.num_elements()):
            assert_almost_equal(
                topk_vals_ptr[i],
                topk_vals_cpu_ptr[i],
            )

            comptime if dtype == .float32:
                assert_equal(
                    topk_idxs_ptr[i],
                    topk_idxs_cpu_ptr[i].cast[out_idx_type](),
                )
        _ = topk_idxs_cpu_ptr^
        _ = topk_vals_cpu_ptr^

    # Consume device buffers
    _ = device_in_buffer^
    _ = device_out_vals_buffer^
    _ = device_out_idxs_buffer^
    _ = device_local_topk_vals_buffer^
    _ = device_local_topk_idxs_buffer^
    _ = K_dev_buffer^
    _ = top_p_dev_buffer^
    _ = top_p_host_ptr^
    _ = K_host_ptr^
    _ = topk_idxs_ptr^
    _ = topk_vals_ptr^
    _ = in_buffer_ptr^


def bench_topk_multi_rank[
    dtype: DType,
    rank: Int,
    out_idx_type: DType = .int,
](
    ctx: DeviceContext,
    mut m: Bench,
    input_shape: IndexList[rank],
    test_case: TestCase,
    fill_fn_name: String,
) raises:
    # Fetch arguments
    # var input_shape = test_case.input_shape
    var K = test_case.K
    var block_size = test_case.block_size
    var num_blocks_per_input: Int = min(
        ceildiv(input_shape.flattened_length(), block_size), 8
    ) if not test_case.num_blocks_per_input else test_case.num_blocks_per_input

    comptime largest = test_case.largest
    comptime sampling = test_case.sampling
    # Instantiate data in host memory
    var out_idx_len = 1 if sampling else K
    var out_vals_shape = input_shape
    out_vals_shape[rank - 1] = K
    var out_idxs_shape = input_shape
    out_idxs_shape[rank - 1] = out_idx_len

    var in_size = input_shape.flattened_length()
    var out_vals_size = out_vals_shape.flattened_length()
    var out_idxs_size = out_idxs_shape.flattened_length()

    var in_buffer_ptr = List(length=in_size, fill=Scalar[dtype](0))
    var topk_vals_ptr = List(length=out_vals_size, fill=Scalar[dtype](0))
    var topk_idxs_ptr = List(length=out_idxs_size, fill=Scalar[out_idx_type](0))

    var in_buffer = TileTensor(in_buffer_ptr, row_major(Coord(input_shape)))
    var topk_vals = TileTensor(topk_vals_ptr, row_major(Coord(out_vals_shape)))
    var topk_idxs = TileTensor(topk_idxs_ptr, row_major(Coord(out_idxs_shape)))

    # Fill the buffer
    fill_buffer[rank, dtype](in_buffer, fill_fn_name)

    # Move data to device
    var device_in_buffer = ctx.enqueue_create_buffer[dtype](in_size)
    var device_out_vals_buffer = ctx.enqueue_create_buffer[dtype](out_vals_size)
    var device_out_idxs_buffer = ctx.enqueue_create_buffer[out_idx_type](
        out_idxs_size
    )

    var device_in = TileTensor(device_in_buffer, row_major(Coord(input_shape)))
    var device_out_vals = TileTensor(
        device_out_vals_buffer, row_major(Coord(out_vals_shape))
    )
    var device_out_idxs = TileTensor(
        device_out_idxs_buffer, row_major(Coord(out_idxs_shape))
    )

    ctx.enqueue_copy(device_in_buffer, in_buffer_ptr)
    var batch_size: Int

    comptime if rank == 1:
        batch_size = 1
    elif rank == 2:
        batch_size = input_shape[0]
    else:  # rank > 2
        var last_dim = input_shape[rank - 1]
        batch_size = input_shape.flattened_length() // last_dim

    var K_host_ptr = List(length=batch_size, fill=Int64(K))
    var K_host_buffer = TileTensor(K_host_ptr, row_major(batch_size))

    var K_dev_buffer = ctx.enqueue_create_buffer[.int64](batch_size)
    var k = TileTensor(K_dev_buffer, row_major(batch_size))
    ctx.enqueue_copy(K_dev_buffer, K_host_ptr)
    ctx.synchronize()
    var max_k = Int(
        reduce_max(
            Span(
                unsafe_ptr=K_host_buffer.ptr,
                length=K_host_buffer.num_elements(),
            )
        )
    )

    @always_inline
    def bench_func(mut b: Bencher) {var k, imm}:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            topk_gpu[sampling=sampling, largest=largest](
                ctx,
                max_k,
                device_in,
                device_out_vals,
                device_out_idxs,
                k=TileTensor(k.ptr, row_major(Int64(batch_size)))
                .as_unsafe_any_origin()
                .as_immut(),
                block_size=block_size,
                num_blocks_per_input=num_blocks_per_input,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    var kernel_name = "topk-multirank"
    var num_bytes = device_in.num_elements() * size_of[dtype]()
    m.bench_function(
        bench_func,
        BenchId(kernel_name),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )

    # Copy results back to host
    ctx.enqueue_copy(topk_vals_ptr, device_out_vals_buffer)
    ctx.enqueue_copy(topk_idxs_ptr, device_out_idxs_buffer)
    ctx.synchronize()

    # ASSERT equality with CPU topk kernel reference
    comptime if not sampling:
        var topk_vals_cpu_ptr = List(
            length=out_vals_size, fill=Scalar[dtype](0)
        )
        var topk_idxs_cpu_ptr = List(length=out_vals_size, fill=Int64(0))
        var topk_vals_cpu = TileTensor(
            topk_vals_cpu_ptr, row_major(Coord(out_vals_shape))
        )
        var topk_idxs_cpu = TileTensor(
            topk_idxs_cpu_ptr, row_major(Coord(out_idxs_shape))
        )

        _top_k_cpu[dtype=dtype, out_idx_type=DType.int64, largest=largest](
            in_buffer,
            max_k,
            rank - 1,
            topk_vals_cpu,
            topk_idxs_cpu,
            1,
            True,
            k=K_host_buffer.as_unsafe_any_origin().as_immut(),
        )

        for i in range(topk_vals.num_elements()):
            assert_almost_equal(
                topk_vals_ptr[i],
                topk_vals_cpu_ptr[i],
            )

            comptime if dtype == .float32:
                assert_equal(
                    topk_idxs_ptr[i],
                    topk_idxs_cpu_ptr[i].cast[out_idx_type](),
                )
        _ = topk_idxs_cpu_ptr^
        _ = topk_vals_cpu_ptr^

    # Consume device buffers
    _ = device_in_buffer^
    _ = device_out_vals_buffer^
    _ = device_out_idxs_buffer^
    _ = K_dev_buffer^
    _ = K_host_ptr^
    _ = topk_idxs_ptr^
    _ = topk_vals_ptr^
    _ = in_buffer_ptr^


def bench_topk_fi[
    dtype: DType,
    out_idx_type: DType,
](
    ctx: DeviceContext,
    mut m: Bench,
    test_case: TestCase,
    fill_fn_name: String,
    top_p: Float32 = 1.0,
    temperature: Float32 = 1.0,
) raises:
    var batch_size = test_case.batch_size
    var N = test_case.N
    var K = test_case.K

    var in_size = batch_size * N

    var in_buffer_ptr = List(length=in_size, fill=Scalar[dtype](0))
    var in_buffer = TileTensor(
        in_buffer_ptr,
        row_major(batch_size, N),
    )
    fill_buffer[2, dtype](in_buffer, fill_fn_name)

    # Device buffers.
    var device_in_buffer = ctx.enqueue_create_buffer[dtype](in_size)
    var device_out_idxs_buffer = ctx.enqueue_create_buffer[out_idx_type](
        batch_size
    )
    var device_temp_buffer = ctx.enqueue_create_buffer[.float32](batch_size)

    var device_in = TileTensor(device_in_buffer, row_major(batch_size, N))
    var device_out_idxs = TileTensor(
        device_out_idxs_buffer,
        row_major(batch_size, Idx[1]),
    )
    var temp_tt = TileTensor(device_temp_buffer, row_major(batch_size))

    ctx.enqueue_copy(device_in_buffer, in_buffer_ptr)

    # Fill temperature on device.
    var temp_host_ptr = List(length=batch_size, fill=temperature)
    ctx.enqueue_copy(device_temp_buffer, temp_host_ptr)

    # Create per-row seed buffer on device.
    var seed_device_buffer = ctx.enqueue_create_buffer[.uint64](batch_size)
    var seed_host_ptr = List(length=batch_size, fill=UInt64(0))
    for i in range(batch_size):
        seed_host_ptr[i] = UInt64(42 + i)
    ctx.enqueue_copy(seed_device_buffer, seed_host_ptr)
    ctx.synchronize()
    var seed_tt = TileTensor(seed_device_buffer, row_major(batch_size))

    @always_inline
    def bench_func(mut b: Bencher) {imm}:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            _topk_topp_sampling_fi[dtype, out_idx_type](
                ctx,
                K,
                top_p,
                device_in,
                device_out_idxs,
                temperature=temp_tt.as_unsafe_any_origin().as_immut(),
                rng_seed=seed_tt.as_unsafe_any_origin().as_immut(),
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    var kernel_name = String(
        "bench-topk-fi",
        "/N=",
        N,
        "/K=",
        K,
        "/batch_size=",
        batch_size,
        "/top_p=",
        top_p,
    )

    var num_bytes = device_in.num_elements() * size_of[dtype]()
    m.bench_function(
        bench_func,
        BenchId(kernel_name),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )

    _ = device_in_buffer^
    _ = device_out_idxs_buffer^
    _ = device_temp_buffer^
    _ = seed_device_buffer^
    _ = seed_host_ptr^
    _ = temp_host_ptr^
    _ = in_buffer_ptr^


def bench_topk_topp_dist[
    dtype: DType,
    out_idx_type: DType,
    emit_dist: Bool,
](
    ctx: DeviceContext,
    mut m: Bench,
    test_case: TestCase,
    fill_fn_name: String,
    top_p: Float32 = 0.95,
    temperature: Float32 = 1.0,
    logit_sigma: Float64 = 2.0,
) raises:
    """Benchmarks the fused softmax + top-k/top-p sampler that spec decode uses.

    Shapes the run like `sampler.fused_token_sampling_with_dist`: raw logits in,
    per-row temperature / top-k / top-p, and (under `emit_dist`) the masked
    renormalized distribution written back alongside the sampled token. Running
    the same shape with `emit_dist` off isolates what the distribution costs,
    which is the cutoff search rather than the [batch, vocab] store.

    How many iterations the searches take depends on the shape of the softmax,
    so a fill that collapses it (iota puts all the mass on one token) measures
    nothing. `fill_normal` with `logit_sigma` is the realistic one.
    """
    var batch_size = test_case.batch_size
    var d = test_case.N
    var k = test_case.K

    var in_size = batch_size * d
    var in_buffer_ptr = List(length=in_size, fill=Scalar[dtype](0))
    var in_buffer = TileTensor(in_buffer_ptr, row_major(batch_size, d))
    fill_buffer[2, dtype](in_buffer, fill_fn_name, logit_sigma)

    var logits_dev = ctx.enqueue_create_buffer[dtype](in_size)
    ctx.enqueue_copy(logits_dev, in_buffer_ptr)

    var tokens_dev = ctx.enqueue_create_buffer[out_idx_type](batch_size)
    # Kept allocated but unused when `emit_dist` is off, so the two variants
    # differ only in the kernel's work.
    var dist_dev = ctx.enqueue_create_buffer[.float32](in_size)

    var temp_host = ctx.enqueue_create_host_buffer[.float32](batch_size)
    var top_p_host = ctx.enqueue_create_host_buffer[.float32](batch_size)
    var top_k_host = ctx.enqueue_create_host_buffer[out_idx_type](batch_size)
    var seed_host = ctx.enqueue_create_host_buffer[.uint64](batch_size)
    for row in range(batch_size):
        temp_host[row] = temperature
        top_p_host[row] = top_p
        top_k_host[row] = Scalar[out_idx_type](k)
        seed_host[row] = UInt64(42 + row)

    var temp_dev = ctx.enqueue_create_buffer[.float32](batch_size)
    var top_p_dev = ctx.enqueue_create_buffer[.float32](batch_size)
    var top_k_dev = ctx.enqueue_create_buffer[out_idx_type](batch_size)
    var seed_dev = ctx.enqueue_create_buffer[.uint64](batch_size)
    ctx.enqueue_copy(temp_dev, temp_host)
    ctx.enqueue_copy(top_p_dev, top_p_host)
    ctx.enqueue_copy(top_k_dev, top_k_host)
    ctx.enqueue_copy(seed_dev, seed_host)
    ctx.synchronize()

    @always_inline
    def bench_func(mut b: Bencher) {mut tokens_dev, mut dist_dev, imm}:
        @always_inline
        def kernel_launch(
            ctx: DeviceContext,
        ) raises {mut tokens_dev, mut dist_dev, imm}:
            topk_topp_sampling_from_prob[
                from_logits=True, emit_dist=emit_dist, dist_dtype=DType.float32
            ](
                ctx,
                TileTensor(logits_dev, row_major(batch_size, d)),
                TileTensor(tokens_dev, row_major(batch_size)),
                d,
                rng_seed=TileTensor(seed_dev, row_major(batch_size))
                .as_unsafe_any_origin()
                .as_immut(),
                top_k_arr=TileTensor(top_k_dev, row_major(batch_size))
                .as_unsafe_any_origin()
                .as_immut(),
                top_p_arr=TileTensor(top_p_dev, row_major(batch_size))
                .as_unsafe_any_origin()
                .as_immut(),
                temperature=TileTensor(temp_dev, row_major(batch_size))
                .as_unsafe_any_origin()
                .as_immut(),
                out_dist=TileTensor(
                    dist_dev,
                    row_major(batch_size, d) if emit_dist else row_major(1, 1),
                ).as_unsafe_any_origin(),
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    var kernel_name = String(
        "bench-topk-topp-dist",
        "/dtype=",
        dtype,
        "/N=",
        d,
        "/K=",
        k,
        "/batch_size=",
        batch_size,
        "/top_p=",
        top_p,
        "/emit_dist=",
        emit_dist,
    )

    var num_bytes = in_size * size_of[dtype]()
    m.bench_function(
        bench_func,
        BenchId(kernel_name),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )

    _ = logits_dev^
    _ = tokens_dev^
    _ = dist_dev^
    _ = temp_dev^
    _ = top_p_dev^
    _ = top_k_dev^
    _ = seed_dev^
    _ = temp_host^
    _ = top_p_host^
    _ = top_k_host^
    _ = seed_host^
    _ = in_buffer_ptr^


def bench_topk_topp_masked[
    dtype: DType
](
    ctx: DeviceContext,
    mut m: Bench,
    test_case: TestCase,
    fill_fn_name: String,
    top_p: Float32 = 0.95,
    temperature: Float32 = 1.0,
    logit_sigma: Float64 = 2.0,
) raises:
    """Benchmarks the target-side masked-probability kernel of spec decode.

    Shares `_topk_topp_cutoff_search` with the sampler benchmarked above, so it
    is here to confirm the search's cost moves for both callers, not just the
    one the sampler exercises.
    """
    var batch_size = test_case.batch_size
    var d = test_case.N
    var k = test_case.K

    var in_size = batch_size * d
    var in_buffer_ptr = List(length=in_size, fill=Scalar[dtype](0))
    var in_buffer = TileTensor(in_buffer_ptr, row_major(batch_size, d))
    fill_buffer[2, dtype](in_buffer, fill_fn_name, logit_sigma)

    var logits_dev = ctx.enqueue_create_buffer[dtype](in_size)
    ctx.enqueue_copy(logits_dev, in_buffer_ptr)
    var probs_dev = ctx.enqueue_create_buffer[.float32](in_size)

    var temp_host = ctx.enqueue_create_host_buffer[.float32](batch_size)
    var top_p_host = ctx.enqueue_create_host_buffer[.float32](batch_size)
    var top_k_host = ctx.enqueue_create_host_buffer[.int64](batch_size)
    for row in range(batch_size):
        temp_host[row] = temperature
        top_p_host[row] = top_p
        top_k_host[row] = Int64(k)
    var temp_dev = ctx.enqueue_create_buffer[.float32](batch_size)
    var top_p_dev = ctx.enqueue_create_buffer[.float32](batch_size)
    var top_k_dev = ctx.enqueue_create_buffer[.int64](batch_size)
    ctx.enqueue_copy(temp_dev, temp_host)
    ctx.enqueue_copy(top_p_dev, top_p_host)
    ctx.enqueue_copy(top_k_dev, top_k_host)
    ctx.synchronize()

    @always_inline
    def bench_func(mut b: Bencher) {mut probs_dev, imm}:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {mut probs_dev, imm}:
            topk_topp_masked_probs[dtype](
                ctx,
                TileTensor(logits_dev, row_major(batch_size, d)),
                TileTensor(probs_dev, row_major(batch_size, d)),
                d,
                top_k_arr=TileTensor(top_k_dev, row_major(batch_size))
                .as_unsafe_any_origin()
                .as_immut(),
                top_p_arr=TileTensor(top_p_dev, row_major(batch_size))
                .as_unsafe_any_origin()
                .as_immut(),
                temperature=TileTensor(temp_dev, row_major(batch_size))
                .as_unsafe_any_origin()
                .as_immut(),
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    m.bench_function(
        bench_func,
        BenchId(
            String(
                "bench-topk-topp-masked",
                "/dtype=",
                dtype,
                "/N=",
                d,
                "/K=",
                k,
                "/batch_size=",
                batch_size,
                "/top_p=",
                top_p,
            )
        ),
        [ThroughputMeasure(BenchMetric.bytes, in_size * size_of[dtype]())],
    )

    _ = logits_dev^
    _ = probs_dev^
    _ = temp_dev^
    _ = top_p_dev^
    _ = top_k_dev^
    _ = temp_host^
    _ = top_p_host^
    _ = top_k_host^
    _ = in_buffer_ptr^


def fill_random[
    rank: Int, dtype: DType
](mut buffer: TileTensor[mut=True, dtype, ...]):
    comptime min_val = -1e9
    comptime max_val = 1e9
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.raw_store(i, random_value.cast[dtype]())


def fill_normal[
    rank: Int, dtype: DType
](mut buffer: TileTensor[mut=True, dtype, ...], sigma: Float64):
    """Fills with N(0, sigma) logits, the shape a trained LM actually emits.

    Box-Muller off the uniform generator. The softmax of these decides how many
    tokens the nucleus holds, and therefore how many passes the pivot and cutoff
    searches need -- the quantity this benchmark exists to measure.
    """
    var total_elements = buffer.num_elements()
    var i = 0
    while i < total_elements:
        var u1 = max(random_float64(0.0, 1.0), 1e-12)
        var u2 = random_float64(0.0, 1.0)
        var r = sigma * (-2.0 * log(u1)) ** 0.5
        buffer.raw_store(i, (r * cos(2.0 * pi * u2)).cast[dtype]())
        if i + 1 < total_elements:
            buffer.raw_store(i + 1, (r * sin(2.0 * pi * u2)).cast[dtype]())
        i += 2


def fill_constant[
    rank: Int, dtype: DType
](mut buffer: TileTensor[mut=True, dtype, ...]):
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        if i % 3 == 1:
            buffer.raw_store(i, 1.0)
        else:
            buffer.raw_store(i, 0.0)


def fill_iota[
    rank: Int, dtype: DType
](mut buf: TileTensor[mut=True, dtype, ...]):
    iota(
        buf.ptr,
        coord_to_index_list(buf.layout.shape_coord()).flattened_length(),
    )


def fill_buffer[
    rank: Int, dtype: DType
](
    mut buffer: TileTensor[mut=True, dtype, ...],
    mode: String,
    sigma: Float64 = 2.0,
) raises:
    if mode == "fill_constant":
        fill_constant[rank, dtype](buffer)
    elif mode == "fill_random":
        fill_random[rank, dtype](buffer)
    elif mode == "fill_iota":
        fill_iota[rank, dtype](buffer)
    elif mode == "fill_normal":
        fill_normal[rank, dtype](buffer, sigma)
    else:
        raise Error("fill mode not found")


@fieldwise_init
struct TestCase[_sampling: Bool, _largest: Bool = True](ImplicitlyCopyable):
    comptime sampling = Self._sampling
    comptime largest = Self._largest
    var N: Int
    var K: Int
    var block_size: Int
    var batch_size: Int
    var num_blocks_per_input: Int


def main() raises:
    # If no N was provided (kbench env args or --N= on the CLI), run the
    # built-in dispatch grid instead of the parameterized benchmark.
    var N = arg_parse("N", -1)
    var gumbel_from_probs = arg_parse("gumbel_from_probs", 0)
    if N < 0:
        if gumbel_from_probs > 0:
            bench_gumbel_from_probs()
            return
        bench_dispatch_all()
        return

    var K = arg_parse("K", 50)
    var block_size = arg_parse("block_size", 256)
    var batch_size = arg_parse("batch_size", 8)
    var num_blocks_per_input = arg_parse("num_blocks_per_input", 0)
    var fill_fn_name = arg_parse("fill_fn_name", "fill_iota")
    var top_p = Float32(arg_parse("top_p", 0.95))
    var logit_sigma = arg_parse("logit_sigma", 2.0)

    comptime dtype = get_defined_dtype["dtype", .float32]()
    comptime rank = get_defined_int["rank", 2]()
    comptime out_idx_type = get_defined_dtype["out_idx_type", .int]()
    comptime sampling = get_defined_bool["sampling", False]()
    comptime largest = get_defined_bool["largest", True]()
    comptime use_fi = get_defined_bool["USE_FI_TOPK_KERNEL", False]()
    # Runtime rather than compile-time: `emit_dist` reaches the kernel as a
    # comptime parameter either way, and a runtime switch here means comparing
    # the two variants does not mean two builds.
    var use_dist = arg_parse("topp_dist", False)
    var emit_dist = arg_parse("emit_dist", True)
    var in_dtype_name = arg_parse("in_dtype", String("float32"))
    var masked_probs = arg_parse("masked_probs", False)

    var m = Bench()
    m.config.show_progress = False
    with DeviceContext() as ctx:
        var test_case = TestCase[_sampling=sampling, _largest=largest](
            N=N,
            K=K,
            block_size=block_size,
            batch_size=batch_size,
            num_blocks_per_input=num_blocks_per_input,
        )

        comptime if has_apple_gpu_accelerator():
            if masked_probs or use_dist:
                raise Error(
                    "the masked_probs and topp_dist benchmarks require"
                    " a non-Apple GPU"
                )
        else:
            if masked_probs:

                @__parameter
                def run_masked[in_dtype: DType]() raises:
                    bench_topk_topp_masked[in_dtype](
                        ctx,
                        m,
                        test_case,
                        fill_fn_name,
                        top_p=top_p,
                        logit_sigma=logit_sigma,
                    )

                if in_dtype_name == "bfloat16":
                    run_masked[.bfloat16]()
                else:
                    run_masked[.float32]()
                m.dump_report()
                return

            if use_dist:

                @__parameter
                def run_dist[in_dtype: DType, emit: Bool]() raises:
                    bench_topk_topp_dist[in_dtype, DType.int64, emit](
                        ctx,
                        m,
                        test_case,
                        fill_fn_name,
                        top_p=top_p,
                        logit_sigma=logit_sigma,
                    )

                # The pipeline feeds this kernel f32 logits today; bf16
                # halves the bytes every pass of the search re-reads, so
                # both are benchmarked.
                @__parameter
                def run_dist_emit[emit: Bool]() raises:
                    if in_dtype_name == "bfloat16":
                        run_dist[.bfloat16, emit]()
                    else:
                        run_dist[.float32, emit]()

                if emit_dist:
                    run_dist_emit[True]()
                else:
                    run_dist_emit[False]()
                m.dump_report()
                return

        comptime if use_fi:
            bench_topk_fi[dtype, out_idx_type](ctx, m, test_case, fill_fn_name)
        else:
            bench_topk_batched[dtype, out_idx_type, rank](
                ctx, m, test_case, fill_fn_name
            )

    m.dump_report()


from std.benchmark import BenchConfig
from nn.topk import fused_token_sampling_gpu, gumbel_sampling_fused_gpu


def bench_dispatch[
    dtype: DType, max_k: Int
](mut b: Bench, ctx: DeviceContext, batch_size: Int, N: Int) raises:
    var buf0 = ctx.enqueue_create_buffer[dtype](batch_size * N)
    var buf1 = ctx.enqueue_create_buffer[dtype](batch_size * N)
    var buf2 = ctx.enqueue_create_buffer[dtype](batch_size * N)
    var buf3 = ctx.enqueue_create_buffer[dtype](batch_size * N)
    buf0.enqueue_fill(Scalar[dtype](0.01))
    buf1.enqueue_fill(Scalar[dtype](0.02))
    buf2.enqueue_fill(Scalar[dtype](0.03))
    buf3.enqueue_fill(Scalar[dtype](0.04))

    comptime out_k = 1 if max_k == -1 else max_k
    var out_buf = ctx.enqueue_create_buffer[.int32](batch_size * out_k)
    var seed_buf = ctx.enqueue_create_buffer[.uint64](batch_size)
    seed_buf.enqueue_fill(UInt64(42))
    ctx.synchronize()

    var out_tt = TileTensor(out_buf, row_major(batch_size, out_k))
    var seed_tt = TileTensor(seed_buf, row_major(batch_size))
    var seed_imm = seed_tt.as_unsafe_any_origin().as_immut()

    # Regime labels mirror the dispatch threshold in fused_token_sampling_gpu
    # (two-stage kernel below max_k = 10, FI rejection sampling at or above).
    comptime regime = "gumbel" if max_k == -1 else (
        "topk_lt10" if max_k < 10 else "topk_ge10"
    )
    var label = (
        String(regime)
        + "_b"
        + String(batch_size)
        + "_v"
        + String(N)
        + "_k"
        + String(max_k)
    )
    var iter0 = 0

    @always_inline
    def do_bench(mut bb: Bencher) raises {mut iter0, imm}:
        @always_inline
        def launch(
            dctx: DeviceContext,
        ) raises {
            imm buf0,
            imm buf1,
            imm buf2,
            imm buf3,
            imm out_tt,
            imm seed_imm,
            imm batch_size,
            imm N,
            mut iter0,
        }:
            var r = iter0 % 4
            var in_imm = (
                TileTensor(
                    buf0 if r
                    == 0 else (buf1 if r == 1 else (buf2 if r == 2 else buf3)),
                    row_major(batch_size, N),
                )
                .as_unsafe_any_origin()
                .as_immut()
            )
            fused_token_sampling_gpu(
                dctx,
                max_k,
                Float32(1.0),
                in_imm,
                out_tt,
                seed=seed_imm,
            )
            iter0 += 1

        bencher_iter_custom(bb, launch, ctx)

    b.bench_function(do_bench, BenchId(label))

    _ = buf0^
    _ = buf1^
    _ = buf2^
    _ = buf3^
    _ = out_buf^
    _ = seed_buf^


def bench_dispatch_all() raises:
    comptime dtype = DType.float32
    var batch_sizes = [1, 8, 32, 128]
    var vocab_sizes = [32000, 128000]

    with DeviceContext() as ctx:
        var b = Bench()
        b.config.max_iters = 1000
        b.config.show_progress = False
        for bs in batch_sizes:
            for v in vocab_sizes:
                bench_dispatch[dtype, -1](b, ctx, bs, v)
                bench_dispatch[dtype, 5](b, ctx, bs, v)
                bench_dispatch[dtype, 20](b, ctx, bs, v)
                bench_dispatch[dtype, 50](b, ctx, bs, v)

        # Bitonic sort top-k (MLA indexer shape: k = N = 2048).
        bench_bitonic_topk(b, ctx)
        # Streaming path (N > 2048, k = 2048): GLM 5.x long-context / prefill.
        bench_bitonic_topk(b, ctx, N=16384, K=2048, batch_size=48)
        bench_bitonic_topk(b, ctx, N=163840, K=2048, batch_size=8)
        bench_bitonic_topk(b, ctx, N=2560, K=2048, batch_size=2048)

        print()
        b.dump_report()


def bench_gumbel_from_probs() raises:
    # The fused sampler's `from_probs` path refuses to build on Apple
    # (`_block_reduce_topk` caps its shared storage at WARP_SIZE there while
    # the kernel launches full-sized blocks), so the benchmark cannot be
    # instantiated for Metal.
    comptime if has_apple_gpu_accelerator():
        raise Error("the gumbel_from_probs benchmark requires a non-Apple GPU")
    else:
        comptime dtype = DType.float32
        comptime vocab = 200064

        with DeviceContext() as ctx:
            var b = Bench()
            b.config.max_iters = 200
            b.config.show_progress = False
            for rows in [32, 96]:
                var probs_buf = ctx.enqueue_create_buffer[dtype](rows * vocab)
                var out_buf = ctx.enqueue_create_buffer[.int64](rows)
                var seed_buf = ctx.enqueue_create_buffer[.uint64](rows)
                probs_buf.enqueue_fill(Scalar[dtype](1.0 / vocab))
                seed_buf.enqueue_fill(UInt64(42))
                ctx.synchronize()

                var probs = (
                    TileTensor(probs_buf, row_major(rows, vocab))
                    .as_unsafe_any_origin()
                    .as_immut()
                )
                var out = TileTensor(out_buf, row_major(rows))
                var seeds = (
                    TileTensor(seed_buf, row_major(rows))
                    .as_unsafe_any_origin()
                    .as_immut()
                )

                @always_inline
                def bench_fn(mut bb: Bencher) raises {imm}:
                    @always_inline
                    def launch(dctx: DeviceContext) raises {imm}:
                        gumbel_sampling_fused_gpu[from_probs=True](
                            dctx, probs, out, seed=seeds
                        )

                    bencher_iter_custom(bb, launch, ctx)

                b.bench_function(
                    bench_fn,
                    BenchId(
                        String(
                            "gumbel_from_probs/rows=", rows, "/vocab=", vocab
                        )
                    ),
                )
                _ = probs_buf^
                _ = out_buf^
                _ = seed_buf^

            print()
            b.dump_report()


def bench_bitonic_topk(
    mut b: Bench,
    ctx: DeviceContext,
    N: Int = PERSISTENT_TOPK_MAX_N,
    K: Int = PERSISTENT_TOPK_MAX_N,
    batch_size: Int = 1,
) raises:
    """Benchmark persistent_topk_block at MLA indexer shapes.

    Defaults to the single-block shape (N=K=2048); larger N exercises the
    streaming path.  Uses the Bench harness so results appear in the same table
    as the existing topk_gpu entries for easy side-by-side comparison.
    """
    comptime dtype = DType.float32

    var scores_buf = ctx.enqueue_create_buffer[dtype](batch_size * N)
    var idxs_buf = ctx.enqueue_create_buffer[.int32](batch_size * K)
    # Fill scores with a non-trivial pattern so the sort is exercised.
    var scores_tt = TileTensor(scores_buf, row_major(batch_size, N))
    scores_buf.enqueue_fill(Scalar[dtype](0.5))
    ctx.synchronize()

    @always_inline
    def bench_fn(mut bb: Bencher) {mut idxs_buf, imm}:
        @always_inline
        def launch(dctx: DeviceContext) raises {mut idxs_buf, imm}:
            persistent_topk_block(
                dctx,
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    scores_tt.ptr
                ),
                rebind[MutPointer[Int32, MutAnyOrigin]](idxs_buf.unsafe_ptr()),
                N,
                K,
                batch_size,
            )

        bencher_iter_custom(bb, launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId(
            String(
                "topk_gpu_bitonic",
                "/N=",
                N,
                "/K=",
                K,
                "/batch_size=",
                batch_size,
            )
        ),
    )

    _ = scores_buf
    _ = idxs_buf
