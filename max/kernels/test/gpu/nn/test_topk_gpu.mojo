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

from std.math import ceildiv, iota, nan
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

from layout import (
    Coord,
    Idx,
    DefaultEngine,
    TileTensor,
    coord_to_index_list,
    row_major,
)

from nn.topk import (
    _top_k_cpu,
    _topk_dead_val,
    _topk_gpu,
    gumbel_sampling_fused_gpu,
    gumbel_sampling_gpu,
    topk_gpu,
)
from std.testing import assert_almost_equal, assert_equal, assert_true

from std.utils import IndexList

comptime DEBUG_BENCH = False
comptime PRINT_OUTPUT = False


def time_kernel(
    mut m: Bench,
    ctx: DeviceContext,
    kernel_name: String,
    func: Some[def(DeviceContext) raises -> None],
) raises:
    @always_inline
    def bench_func(mut m: Bencher) {imm}:
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            func(ctx)

        bencher_iter_custom(m, kernel_launch, ctx)

    m.bench_function(
        bench_func,
        BenchId(
            kernel_name
        ),  # ThroughputMeasure(BenchMetric.elements, 2 * size)
    )


def test_case_batched[
    dtype: DType,
    out_idx_type: DType = .int,
    rank: Int = 2,
](
    ctx: DeviceContext,
    test_case: TestCase,
    fill_fn: Some[
        def[
            dtype: DType
        ](
            TileTensor[
                mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
            ]
        ) -> None
    ],
) raises:
    # Fetch arguments
    var m = Bench()
    var batch_size = test_case.batch_size
    var N = test_case.N
    var K = test_case.K
    var block_size = test_case.block_size
    var num_blocks_per_input = test_case.num_blocks_per_input
    comptime largest = test_case.largest
    comptime sampling = test_case.sampling
    # Instantiate data in host memory
    var out_idx_len = 1 if sampling else K

    # Allocate host memory
    var in_shape = IndexList[2](batch_size, N)
    var out_vals_shape = IndexList[2](batch_size, K)
    var out_idxs_shape = IndexList[2](batch_size, out_idx_len)

    var in_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        in_shape.flattened_length()
    )
    var topk_vals_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        out_vals_shape.flattened_length()
    )
    var topk_idxs_host_ptr = ctx.enqueue_create_host_buffer[out_idx_type](
        out_idxs_shape.flattened_length()
    )

    # Create LayoutTensor for fill_fn (required by function signature)
    var in_tensor = TileTensor(in_host_ptr, row_major(Coord(in_shape)))

    # Fill the buffer with consecutive values
    fill_fn(in_tensor)

    # Create device buffers
    var device_in = ctx.enqueue_create_buffer[dtype](
        in_shape.flattened_length()
    )
    var device_out_vals = ctx.enqueue_create_buffer[dtype](
        out_vals_shape.flattened_length()
    )
    var device_out_idxs = ctx.enqueue_create_buffer[out_idx_type](
        out_idxs_shape.flattened_length()
    )

    var num_blocks_per_input_: Int = ceildiv(
        N, block_size
    ) if not num_blocks_per_input else num_blocks_per_input.value()
    var local_topk_shape = IndexList[2](batch_size, num_blocks_per_input_ * K)
    var device_local_topk_vals = ctx.enqueue_create_buffer[dtype](
        local_topk_shape.flattened_length()
    )
    var device_local_topk_idxs = ctx.enqueue_create_buffer[out_idx_type](
        local_topk_shape.flattened_length()
    )

    ctx.enqueue_copy(device_in, in_host_ptr)

    # Create K buffers
    var K_shape = IndexList[1](batch_size)
    var K_device_buffer = ctx.enqueue_create_buffer[.int64](
        K_shape.flattened_length()
    )
    var K_host_ptr = ctx.enqueue_create_host_buffer[.int64](
        K_shape.flattened_length()
    )
    for i in range(batch_size):
        K_host_ptr[i] = Int64(K)

    var max_k = Int(
        reduce_max(
            Span(
                unsafe_ptr=K_host_ptr.unsafe_ptr(),
                length=K_shape.flattened_length(),
            )
        )
    )

    ctx.enqueue_copy(K_device_buffer, K_host_ptr)
    ctx.synchronize()

    # Create tile tensors for kernel calls

    var in_runtime_layout = row_major(batch_size, N)
    var out_vals_runtime_layout = row_major(batch_size, K)
    var out_idxs_runtime_layout = row_major(batch_size, out_idx_len)
    var local_topk_runtime_layout = row_major(
        (batch_size, num_blocks_per_input_ * K)
    )
    var k_runtime_layout = row_major(batch_size)

    var device_in_tt = TileTensor(device_in, in_runtime_layout)
    var device_out_vals_tt = TileTensor(
        device_out_vals, out_vals_runtime_layout
    )
    var device_out_idxs_tt = TileTensor(
        device_out_idxs, out_idxs_runtime_layout
    )
    var device_local_topk_vals_tt = TileTensor(
        device_local_topk_vals, local_topk_runtime_layout
    )
    var device_local_topk_idxs_tt = TileTensor(
        device_local_topk_idxs, local_topk_runtime_layout
    )
    var k_tt = TileTensor(K_device_buffer, k_runtime_layout)

    comptime if DEBUG_BENCH:

        @always_inline
        def run_func(ctx: DeviceContext) raises {var}:
            _topk_gpu[sampling=sampling, largest=largest](
                ctx,
                max_k,
                device_in_tt,
                device_local_topk_vals_tt,
                device_local_topk_idxs_tt,
                device_out_vals_tt,
                device_out_idxs_tt,
                k=k_tt.as_unsafe_any_origin().as_immut(),
                block_size=block_size,
                num_blocks_per_input=num_blocks_per_input,
            )
            ctx.enqueue_copy(topk_vals_host_ptr, device_out_vals)
            ctx.enqueue_copy(topk_idxs_host_ptr, device_out_idxs)
            ctx.synchronize()

        comptime msg = "tk-smpl-gpu" if sampling else "tk-gpu"
        time_kernel(m, ctx, msg, run_func)

    _topk_gpu[sampling=sampling, largest=largest](
        ctx,
        max_k,  # max_k
        device_in_tt,
        device_local_topk_vals_tt,
        device_local_topk_idxs_tt,
        device_out_vals_tt,
        device_out_idxs_tt,
        k=k_tt.as_unsafe_any_origin().as_immut(),
        block_size=block_size,
        num_blocks_per_input=num_blocks_per_input,
    )

    # Copy results back to host
    ctx.enqueue_copy(topk_vals_host_ptr, device_out_vals)
    ctx.enqueue_copy(topk_idxs_host_ptr, device_out_idxs)
    ctx.synchronize()

    comptime if PRINT_OUTPUT:
        var _msg1: String = "Top-K values"
        var _msg2 = "Sample token index" if sampling else String(
            "Top K indices"
        )
        print(_msg1, "and", _msg2, "output available in host pointers")

    # Regression check for sampled token must be in [0, N).
    # Catches the p==-1 sentinel leaking through as an invalid token id.
    comptime if sampling:
        for b in range(batch_size):
            var tok = Int(topk_idxs_host_ptr[b])
            assert_true(
                tok >= 0 and tok < N,
                "token out of range [0, N): got "
                + String(tok)
                + " for N="
                + String(N),
            )
    # ASSERT equality with CPU topk kernel reference
    comptime if not sampling:
        var topk_vals_cpu_ptr = ctx.enqueue_create_host_buffer[dtype](
            out_vals_shape.flattened_length()
        )
        var topk_idxs_cpu_ptr = ctx.enqueue_create_host_buffer[.int64](
            out_vals_shape.flattened_length()
        )

        # Create tile tensors for CPU reference
        var in_host_tt = TileTensor(in_host_ptr, in_runtime_layout)
        var topk_vals_cpu_tt = TileTensor(
            topk_vals_cpu_ptr, out_vals_runtime_layout
        )
        var topk_idxs_cpu_tt = TileTensor(
            topk_idxs_cpu_ptr, out_vals_runtime_layout
        )
        var k_host_tt = TileTensor(K_host_ptr, k_runtime_layout)

        comptime if DEBUG_BENCH:

            @always_inline
            def run_func_cpu(ctx: DeviceContext) raises {var}:
                _top_k_cpu[
                    dtype=dtype,
                    out_idx_type=DType.int64,
                    largest=largest,
                ](
                    in_host_tt,
                    max_k,
                    rank - 1,
                    topk_vals_cpu_tt,
                    topk_idxs_cpu_tt,
                    1,
                    True,
                    k=k_host_tt.as_unsafe_any_origin().as_immut(),
                )

            time_kernel(m, ctx, "topk-cpu", run_func_cpu)

        _top_k_cpu[dtype=dtype, out_idx_type=DType.int64, largest=largest](
            in_host_tt,
            max_k,
            rank - 1,
            topk_vals_cpu_tt,
            topk_idxs_cpu_tt,
            1,
            True,
            k=k_host_tt.as_unsafe_any_origin().as_immut(),
        )

        for i in range(out_vals_shape.flattened_length()):
            assert_almost_equal(
                topk_vals_host_ptr[i],
                topk_vals_cpu_ptr[i],
            )

            comptime if dtype == .float32:
                assert_equal(
                    topk_idxs_host_ptr[i],
                    topk_idxs_cpu_ptr[i].cast[out_idx_type](),
                )

    # Free device buffers
    _ = device_in^
    _ = device_out_vals^
    _ = device_out_idxs^
    _ = device_local_topk_vals^
    _ = device_local_topk_idxs^
    _ = K_device_buffer^

    comptime if DEBUG_BENCH:
        m.dump_report()


def test_case_multi_rank[
    dtype: DType,
    rank: Int,
    out_idx_type: DType = .int,
](
    ctx: DeviceContext,
    test_case: TestCaseMultiRank[rank=rank, ...],
    fill_fn: Some[
        def[
            dtype: DType
        ](
            TileTensor[
                mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
            ]
        ) -> None
    ],
) raises:
    # Fetch arguments
    var input_shape = test_case.input_shape
    var K = test_case.K
    var block_size = test_case.block_size
    var num_blocks_per_input = test_case.num_blocks_per_input
    comptime largest = test_case.largest
    comptime sampling = test_case.sampling
    # Instantiate data in host memory
    var out_idx_len = 1 if sampling else K
    var out_vals_shape = input_shape
    out_vals_shape[rank - 1] = K
    var out_idxs_shape = input_shape
    out_idxs_shape[rank - 1] = out_idx_len

    # Allocate host memory
    var in_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        input_shape.flattened_length()
    )
    var topk_vals_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        out_vals_shape.flattened_length()
    )
    var topk_idxs_host_ptr = ctx.enqueue_create_host_buffer[out_idx_type](
        out_idxs_shape.flattened_length()
    )

    # Create LayoutTensor for fill_fn (required by function signature)
    var in_tensor = TileTensor(in_host_ptr, row_major(Coord(input_shape)))

    # Fill the buffer with consecutive values
    fill_fn(in_tensor)

    # Create device buffers
    var device_in = ctx.enqueue_create_buffer[dtype](
        input_shape.flattened_length()
    )
    var device_out_vals = ctx.enqueue_create_buffer[dtype](
        out_vals_shape.flattened_length()
    )
    var device_out_idxs = ctx.enqueue_create_buffer[out_idx_type](
        out_idxs_shape.flattened_length()
    )

    ctx.enqueue_copy(device_in, in_host_ptr)
    var batch_size: Int

    comptime if rank == 1:
        batch_size = 1
    elif rank == 2:
        batch_size = input_shape[0]
    else:  # rank > 2
        var last_dim = input_shape[rank - 1]
        batch_size = input_shape.flattened_length() // last_dim

    # Create K buffers
    var K_shape = IndexList[1](batch_size)
    var K_host_ptr = ctx.enqueue_create_host_buffer[.int64](
        K_shape.flattened_length()
    )
    for i in range(batch_size):
        K_host_ptr[i] = Int64(K)

    var K_device_buffer = ctx.enqueue_create_buffer[.int64](
        K_shape.flattened_length()
    )
    ctx.enqueue_copy(K_device_buffer, K_host_ptr)
    ctx.synchronize()
    var max_k = Int(
        reduce_max(
            Span(
                unsafe_ptr=K_host_ptr.unsafe_ptr(),
                length=K_shape.flattened_length(),
            )
        )
    )

    # Create tile tensors for kernel calls
    var in_runtime_layout = row_major(Coord(input_shape))
    var out_vals_runtime_layout = row_major(Coord(out_vals_shape))
    var out_idxs_runtime_layout = row_major(Coord(out_idxs_shape))
    var k_runtime_layout = row_major(batch_size)

    var device_in_tt = TileTensor(device_in, in_runtime_layout)
    var device_out_vals_tt = TileTensor(
        device_out_vals, out_vals_runtime_layout
    )
    var device_out_idxs_tt = TileTensor(
        device_out_idxs, out_idxs_runtime_layout
    )
    var k_tt = TileTensor(K_device_buffer, k_runtime_layout)

    topk_gpu[sampling=sampling, largest=largest](
        ctx,
        max_k,
        device_in_tt,
        device_out_vals_tt,
        device_out_idxs_tt,
        k=k_tt.as_unsafe_any_origin().as_immut(),
        block_size=block_size,
        num_blocks_per_input=num_blocks_per_input,
    )

    # Copy results back to host
    ctx.enqueue_copy(topk_vals_host_ptr, device_out_vals)
    ctx.enqueue_copy(topk_idxs_host_ptr, device_out_idxs)
    ctx.synchronize()

    # ASSERT equality with CPU topk kernel reference
    comptime if not sampling:
        var topk_vals_cpu_ptr = ctx.enqueue_create_host_buffer[dtype](
            out_vals_shape.flattened_length()
        )
        var topk_idxs_cpu_ptr = ctx.enqueue_create_host_buffer[.int64](
            out_idxs_shape.flattened_length()
        )

        # Create tile tensors for CPU reference
        var in_host_tt = TileTensor(in_host_ptr, in_runtime_layout)
        var topk_vals_cpu_tt = TileTensor(
            topk_vals_cpu_ptr, out_vals_runtime_layout
        )
        var topk_idxs_cpu_tt = TileTensor(
            topk_idxs_cpu_ptr, out_vals_runtime_layout
        )
        var k_host_tt = TileTensor(K_host_ptr, k_runtime_layout)

        _top_k_cpu[dtype=dtype, out_idx_type=DType.int64, largest=largest](
            in_host_tt,
            max_k,
            rank - 1,
            topk_vals_cpu_tt,
            topk_idxs_cpu_tt,
            1,
            True,
            k=k_host_tt.as_unsafe_any_origin().as_immut(),
        )

        for i in range(out_vals_shape.flattened_length()):
            assert_almost_equal(
                topk_vals_host_ptr[i],
                topk_vals_cpu_ptr[i],
            )

            comptime if dtype == .float32:
                assert_equal(
                    topk_idxs_host_ptr[i],
                    topk_idxs_cpu_ptr[i].cast[out_idx_type](),
                )

    # Free device buffers
    _ = device_in^
    _ = device_out_vals^
    _ = device_out_idxs^
    _ = K_device_buffer^


def fill_random[
    dtype: DType
](
    buffer: TileTensor[
        mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
    ]
):
    comptime min_val = -1e9
    comptime max_val = 1e9
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.raw_store(i, random_value.cast[dtype]())


def fill_constant[
    dtype: DType
](
    buffer: TileTensor[
        mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
    ]
):
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        if i % 3 == 1:
            buffer.raw_store(i, 1.0)
        else:
            buffer.raw_store(i, 0.0)


def fill_nan[
    dtype: DType
](
    buffer: TileTensor[
        mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
    ]
):
    """Fill all elements with NaN — regression guard for Bug 1 (all-NaN row
    emits invalid token) and Bug 3 (p==-1 sentinel causes OOB read/write)."""
    var nan_val = nan[dtype]()
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        buffer.raw_store(i, nan_val)


def fill_iota[
    dtype: DType
](buf: TileTensor[mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]]):
    iota(
        buf._storage,
        coord_to_index_list(buf.layout.shape_coord()).flattened_length(),
    )


struct TestCase[_sampling: Bool, _largest: Bool = True](ImplicitlyCopyable):
    comptime sampling = Self._sampling
    comptime largest = Self._largest
    var name: String
    var N: Int
    var K: Int
    var block_size: Int
    var batch_size: Int
    var num_blocks_per_input: Optional[Int]

    def __init__(
        out self,
        N: Int,
        K: Int,
        block_size: Int,
        batch_size: Int,
        num_blocks_per_input: Optional[Int] = None,
        name: String = "",
    ):
        self.name = name
        self.N = N
        self.K = K
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_blocks_per_input = num_blocks_per_input


struct TestCaseMultiRank[_sampling: Bool, rank: Int, _largest: Bool = True](
    ImplicitlyCopyable
):
    comptime sampling = Self._sampling
    comptime largest = Self._largest
    var input_shape: IndexList[Self.rank]
    var K: Int
    var block_size: Optional[Int]
    var num_blocks_per_input: Optional[Int]

    def __init__(
        out self,
        input_shape: IndexList[Self.rank],
        K: Int,
        block_size: Optional[Int] = None,
        num_blocks_per_input: Optional[Int] = None,
    ):
        self.input_shape = input_shape
        self.K = K
        self.block_size = block_size
        self.num_blocks_per_input = num_blocks_per_input


def print_test_case(test_case: TestCase):
    var num_blocks_per_in_msg = "auto"
    if test_case.num_blocks_per_input:
        num_blocks_per_in_msg = String(test_case.num_blocks_per_input.value())
    print(
        "==== Running Top-K "
        + (test_case.name + " " if test_case.name else "")
        + "sampling=",
        test_case.sampling,
        ", N=",
        test_case.N,
        ", K=",
        test_case.K,
        ", block_size=",
        test_case.block_size,
        ", batch_size=",
        test_case.batch_size,
        ", num_blocks_per_input=",
        num_blocks_per_in_msg,
    )


def print_test_case(test_case: TestCaseMultiRank):
    var num_blocks_per_in_msg = "auto"
    if test_case.num_blocks_per_input:
        num_blocks_per_in_msg = String(test_case.num_blocks_per_input.value())
    var block_size_msg = "auto"
    if test_case.block_size:
        block_size_msg = String(test_case.block_size.value())
    print(
        "==== Running Top-K sampling=",
        test_case.sampling,
        ", input_shape=",
        test_case.input_shape,
        ", K=",
        test_case.K,
        ", block_size=",
        block_size_msg,
        ", num_blocks_per_input=",
        num_blocks_per_in_msg,
    )


def test_min_topk[dtype: DType](ctx: DeviceContext) raises:
    comptime llama3_vocab_size = 128256

    comptime test_case0 = TestCase[_sampling=False, _largest=False](
        N=1024,
        K=1,
        block_size=256,
        batch_size=1,
    )
    print_test_case(test_case0)
    test_case_batched[
        dtype,
        out_idx_type=DType.uint64,
    ](ctx, test_case0, fill_iota)

    comptime test_case1 = TestCase[_sampling=False, _largest=False](
        N=32000,
        K=5,
        block_size=512,
        batch_size=16,
        num_blocks_per_input=8,
    )
    print_test_case(test_case1)
    test_case_batched[dtype,](ctx, test_case1, fill_iota)

    comptime test_case2 = TestCase[_sampling=False, _largest=False](
        N=llama3_vocab_size,
        K=10,
        block_size=1024,
        batch_size=64,
        num_blocks_per_input=6,
    )
    print_test_case(test_case2)
    # Changed from fill_random to fill_iota for deterministic test data.
    # With random data, duplicate/similar values can cause GPU and CPU to
    # produce different (but equally valid) index orderings.
    test_case_batched[dtype,](ctx, test_case2, fill_iota)


def test_multi_rank[dtype: DType, sampling: Bool](ctx: DeviceContext) raises:
    comptime llama3_vocab_size = 128256

    comptime test_case_multi_rank1 = TestCaseMultiRank[
        _sampling=sampling, rank=1, _largest=True
    ](
        input_shape=IndexList[1](4096),
        K=10,
        block_size=256,
    )
    print_test_case(test_case_multi_rank1)
    test_case_multi_rank[dtype](ctx, test_case_multi_rank1, fill_iota)

    comptime test_case_multi_rank2 = TestCaseMultiRank[
        _sampling=sampling, rank=2, _largest=True
    ](
        input_shape=IndexList[2](10, 1024),
        K=5,
        block_size=512,
    )
    print_test_case(test_case_multi_rank2)
    test_case_multi_rank[dtype](ctx, test_case_multi_rank2, fill_iota)

    comptime test_case_multi_rank3 = TestCaseMultiRank[
        _sampling=sampling, rank=3, _largest=True
    ](
        input_shape=IndexList[3](2, 128, llama3_vocab_size),
        K=5,
        num_blocks_per_input=2,
    )
    print_test_case(test_case_multi_rank3)
    test_case_multi_rank[dtype](ctx, test_case_multi_rank3, fill_iota)


def test_gumbel_zero_temperature[dtype: DType](ctx: DeviceContext) raises:
    """Regression: temperature == 0 must not cause NaN or an out-of-range token.

    Without the clamp, 0.0 / 0.0 = NaN (all-zero logits at temp=0), causing
    the argmax to return an undefined token.  With `max(temp, 1e-6)` the
    division is safe and the token stays in [0, N).
    """
    print("==== Running gumbel temp=0 regression: N=256 batch_size=2")
    comptime N = 256
    comptime batch_size = 2

    var device_in = ctx.enqueue_create_buffer[dtype](batch_size * N)
    var device_temp = ctx.enqueue_create_buffer[.float32](batch_size)
    var device_out = ctx.enqueue_create_buffer[.int32](batch_size)

    # All-zero logits at temperature 0: the case that divides by zero without
    # the clamp.
    with device_in.map_to_host() as in_host:
        for i in range(batch_size * N):
            in_host[i] = Scalar[dtype](0)
    with device_temp.map_to_host() as temp_host:
        for i in range(batch_size):
            temp_host[i] = Float32(0)

    var in_tt = TileTensor(device_in, row_major(batch_size, N))
    var temp_tt = TileTensor(device_temp, row_major(batch_size))
    var out_tt = TileTensor(device_out, row_major(batch_size, 1))

    gumbel_sampling_gpu(
        ctx,
        in_tt.as_unsafe_any_origin().as_immut(),
        out_tt,
        temperature=temp_tt.as_unsafe_any_origin().as_immut(),
    )

    with device_out.map_to_host() as out_host:
        for b in range(batch_size):
            var tok = Int(out_host[b])
            assert_true(
                tok >= 0 and tok < N,
                "gumbel temp=0: token out of range [0, N), got "
                + String(tok)
                + " (N="
                + String(N)
                + ")",
            )


def test_gumbel_fused_matches_two_kernel[
    dtype: DType
](ctx: DeviceContext, N: Int, batch_size: Int) raises:
    """Bit-identity: fused Gumbel-argmax must match the two-kernel path.

    Both paths use the same Philox RNG seeded identically per vocab element, so
    the noised logits are bit-identical and the argmax winner must match exactly
    for every batch row.
    """
    print(
        "==== Running gumbel fused == two-kernel: N=",
        N,
        " batch_size=",
        batch_size,
        sep="",
    )

    var device_in = ctx.enqueue_create_buffer[dtype](batch_size * N)
    var device_temp = ctx.enqueue_create_buffer[.float32](batch_size)
    var device_seed = ctx.enqueue_create_buffer[.uint64](batch_size)
    var out_ref = ctx.enqueue_create_buffer[.int32](batch_size)
    var out_fused = ctx.enqueue_create_buffer[.int32](batch_size)

    with device_in.map_to_host() as in_host:
        for i in range(batch_size * N):
            in_host[i] = Scalar[dtype](random_float64(-5.0, 5.0))
    with device_temp.map_to_host() as temp_host:
        for b in range(batch_size):
            temp_host[b] = Float32(0.5 + 0.1 * Float32(b))
    with device_seed.map_to_host() as seed_host:
        for b in range(batch_size):
            seed_host[b] = UInt64(1234 + b)

    var in_tt = TileTensor(device_in, row_major(batch_size, N))
    var temp_tt = TileTensor(device_temp, row_major(batch_size))
    var seed_tt = TileTensor(device_seed, row_major(batch_size))
    var ref_tt = TileTensor(out_ref, row_major(batch_size, 1))
    var fused_tt = TileTensor(out_fused, row_major(batch_size, 1))

    gumbel_sampling_gpu(
        ctx,
        in_tt.as_unsafe_any_origin().as_immut(),
        ref_tt,
        temperature=temp_tt.as_unsafe_any_origin().as_immut(),
        seed=seed_tt.as_unsafe_any_origin().as_immut(),
    )
    gumbel_sampling_fused_gpu(
        ctx,
        in_tt.as_unsafe_any_origin().as_immut(),
        fused_tt,
        temperature=temp_tt.as_unsafe_any_origin().as_immut(),
        seed=seed_tt.as_unsafe_any_origin().as_immut(),
    )
    ctx.synchronize()

    with out_ref.map_to_host() as ref_host, out_fused.map_to_host() as fused_host:
        for b in range(batch_size):
            assert_equal(
                Int(fused_host[b]),
                Int(ref_host[b]),
                "fused != two-kernel at row "
                + String(b)
                + ": fused="
                + String(Int(fused_host[b]))
                + " ref="
                + String(Int(ref_host[b])),
            )

    _ = device_in^
    _ = device_temp^
    _ = device_seed^
    _ = out_ref^
    _ = out_fused^


def test_topk_zero_k_row[dtype: DType](ctx: DeviceContext) raises:
    """Regression: a per-row ``k == 0`` must emit the skip-token sentinel, not garbage.

    A per-row ``k == 0`` makes ``_topk_stage2`` take the sampling skip branch
    (loop guard ``k >= k_batch`` fires immediately at ``k == 0``): it writes the
    output token index ``0`` and fills the values buffer positions ``k..max_k``
    with `_topk_dead_val` (``-inf`` for ``largest=True``).  The subsequent
    ``range(k_batch)`` sampling loop runs zero iterations, so the ``0`` sentinel
    survives.  Token ``0`` (not ``-1``) is used because this index is consumed
    downstream as an array index (gather_nd / embedding lookup), where a
    negative index would read out of bounds.  A batch row with ``k == 0`` must
    therefore produce token index ``0``, while neighboring ``k > 0`` rows still
    sample a valid token in ``[0, N)``.
    """
    print("==== Running Top-K [k=0 zero row] sampling=True, N=256 batch_size=4")
    comptime N = 256
    comptime batch_size = 4
    comptime block_size = 256
    comptime largest = True
    comptime sampling = True
    comptime zero_row = 1  # the single row whose per-row k is 0
    comptime nonzero_k = 5
    comptime out_idx_len = 1  # sampling emits one token per row
    comptime max_k = nonzero_k
    comptime dead_val = _topk_dead_val[dtype, largest]()

    var num_blocks_per_input_ = ceildiv(N, block_size)

    # Device buffers; host staging is handled by `map_to_host()` below.
    var device_in = ctx.enqueue_create_buffer[dtype](batch_size * N)
    var K_device_buffer = ctx.enqueue_create_buffer[.int64](batch_size)
    var device_out_vals = ctx.enqueue_create_buffer[dtype](batch_size * max_k)
    var device_out_idxs = ctx.enqueue_create_buffer[.int32](
        batch_size * out_idx_len
    )
    var device_local_topk_vals = ctx.enqueue_create_buffer[dtype](
        batch_size * num_blocks_per_input_ * max_k
    )
    var device_local_topk_idxs = ctx.enqueue_create_buffer[.int32](
        batch_size * num_blocks_per_input_ * max_k
    )

    # Fill inputs on host; `map_to_host` pushes them to device at block exit.
    with device_in.map_to_host() as in_host:
        var in_tensor = TileTensor(in_host, row_major(batch_size, N))
        fill_iota[dtype](in_tensor)
    # Per-row K: one row is 0 (the skip case under test), the rest nonzero_k.
    with K_device_buffer.map_to_host() as k_host:
        for i in range(batch_size):
            k_host[i] = Int64(0) if i == zero_row else Int64(nonzero_k)

    var device_in_tt = TileTensor(device_in, row_major(batch_size, N))
    var device_out_vals_tt = TileTensor(
        device_out_vals, row_major(batch_size, max_k)
    )
    var device_out_idxs_tt = TileTensor(
        device_out_idxs, row_major(batch_size, out_idx_len)
    )
    var device_local_topk_vals_tt = TileTensor(
        device_local_topk_vals,
        row_major(batch_size, num_blocks_per_input_ * max_k),
    )
    var device_local_topk_idxs_tt = TileTensor(
        device_local_topk_idxs,
        row_major(batch_size, num_blocks_per_input_ * max_k),
    )
    var k_tt = TileTensor(K_device_buffer, row_major(batch_size))

    _topk_gpu[sampling=sampling, largest=largest](
        ctx,
        max_k,
        device_in_tt,
        device_local_topk_vals_tt,
        device_local_topk_idxs_tt,
        device_out_vals_tt,
        device_out_idxs_tt,
        k=k_tt.as_unsafe_any_origin().as_immut(),
        block_size=block_size,
    )

    # Read outputs; `map_to_host` copies device->host and syncs on block entry.
    with device_out_idxs.map_to_host() as idxs_host:
        # Primary assertion: the k=0 row emits the 0 skip-token sentinel (0,
        # not -1, so the token is safe to use as a downstream array index).
        var zero_tok = Int(idxs_host[zero_row])
        assert_equal(
            zero_tok,
            0,
            "k=0 row must emit skip-token sentinel 0, got " + String(zero_tok),
        )

        # Neighboring k>0 rows must still sample a valid token in [0, N).
        for b in range(batch_size):
            if b == zero_row:
                continue
            var tok = Int(idxs_host[b])
            assert_true(
                tok >= 0 and tok < N,
                "nonzero-k row token out of range [0, N): got "
                + String(tok)
                + " for N="
                + String(N),
            )

    # Secondary check: the k=0 row's values buffer holds the dead value (-inf)
    # in positions k..max_k (here k==0, so the whole row).
    with device_out_vals.map_to_host() as vals_host:
        for j in range(max_k):
            assert_equal(
                vals_host[zero_row * max_k + j],
                dead_val,
                "k=0 row vals["
                + String(j)
                + "] must be the dead value sentinel",
            )


def main() raises:
    comptime llama3_vocab_size = 128256
    with DeviceContext() as ctx:
        comptime dtype = DType.float32
        comptime bf16_type = DType.bfloat16

        # var test_cases: [TestCase] = []
        # var N_values = [1024, 32000, 128256]
        # var K_values = [1, 5, 10]
        # var block_size_values = [256, 512, 1024]
        # var batch_size_values = [1, 16, 64, 256]
        # var _samplingvalues = [False, True]

        comptime test_case0 = TestCase[_sampling=False](
            N=1024,
            K=256,
            block_size=256,
            batch_size=1,
        )
        print_test_case(test_case0)
        test_case_batched[
            dtype,
            out_idx_type=DType.uint64,
        ](ctx, test_case0, fill_iota)

        comptime test_case1 = TestCase[_sampling=False](
            N=1024,
            K=1,
            block_size=256,
            batch_size=1,
        )
        print_test_case(test_case1)
        test_case_batched[
            dtype,
            out_idx_type=DType.uint64,
        ](ctx, test_case1, fill_iota)

        comptime test_case2 = TestCase[_sampling=False](
            N=32000,
            K=5,
            block_size=512,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case2)
        test_case_batched[dtype](ctx, test_case2, fill_iota)

        comptime test_case3 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=10,
            block_size=1024,
            batch_size=64,
            num_blocks_per_input=6,
        )
        print_test_case(test_case3)
        # Changed from fill_random to fill_iota for deterministic test data
        test_case_batched[dtype](ctx, test_case3, fill_iota)

        comptime test_case4 = TestCase[_sampling=True](
            N=1024,
            K=1,
            block_size=256,
            batch_size=1,
        )
        print_test_case(test_case4)
        test_case_batched[dtype,](ctx, test_case4, fill_iota)

        comptime test_case5 = TestCase[_sampling=True](
            N=32000,
            K=5,
            block_size=512,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case5)
        test_case_batched[dtype](ctx, test_case5, fill_iota)

        comptime test_case6 = TestCase[_sampling=True](
            N=llama3_vocab_size,
            K=10,
            block_size=1024,
            batch_size=64,
            num_blocks_per_input=6,
        )
        print_test_case(test_case6)
        test_case_batched[
            dtype,
            out_idx_type=DType.int32,
        ](ctx, test_case6, fill_random)

        comptime test_case7 = TestCase[_sampling=False](
            N=1024,
            K=5,
            block_size=256,
            batch_size=16,
        )
        print_test_case(test_case7)
        test_case_batched[dtype](ctx, test_case7, fill_iota)

        comptime test_case8 = TestCase[_sampling=False](
            N=32000,
            K=25,
            block_size=1024,
            batch_size=64,
            num_blocks_per_input=2,
        )
        print_test_case(test_case8)
        test_case_batched[dtype](ctx, test_case8, fill_iota)

        comptime test_case9 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=1,
            block_size=1024,
            batch_size=256,
            num_blocks_per_input=8,
        )
        print_test_case(test_case9)
        test_case_batched[dtype](ctx, test_case9, fill_iota)

        comptime test_case10 = TestCase[_sampling=True](
            N=1024,
            K=10,
            block_size=256,
            batch_size=64,
        )
        print_test_case(test_case10)
        test_case_batched[dtype](ctx, test_case10, fill_iota)

        comptime test_case11 = TestCase[_sampling=True](
            N=32000,
            K=1,
            block_size=512,
            batch_size=256,
            num_blocks_per_input=8,
        )
        print_test_case(test_case11)
        test_case_batched[bf16_type](ctx, test_case11, fill_random)

        comptime test_case12 = TestCase[_sampling=True](
            N=llama3_vocab_size,
            K=5,
            block_size=1024,
            batch_size=1,
        )
        print_test_case(test_case12)
        test_case_batched[bf16_type](ctx, test_case12, fill_random)

        comptime test_case13 = TestCase[_sampling=False](
            N=1024,
            K=10,
            block_size=1024,
            batch_size=256,
        )
        print_test_case(test_case13)
        test_case_batched[
            bf16_type,
            out_idx_type=DType.uint64,
        ](ctx, test_case13, fill_iota)

        comptime test_case14 = TestCase[_sampling=False](
            N=32000,
            K=1,
            block_size=512,
            batch_size=1,
        )
        print_test_case(test_case14)
        test_case_batched[bf16_type](ctx, test_case14, fill_random)

        comptime test_case15 = TestCase[_sampling=True](
            N=llama3_vocab_size,
            K=5,
            block_size=1024,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case15)
        test_case_batched[bf16_type](ctx, test_case15, fill_iota)

        comptime test_case16 = TestCase[_sampling=True](
            N=1024,
            K=5,
            block_size=256,
            batch_size=64,
        )
        print_test_case(test_case16)
        test_case_batched[
            bf16_type,
            out_idx_type=DType.int64,
        ](ctx, test_case16, fill_iota)

        comptime test_case17 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=1,
            block_size=512,
            batch_size=16,
            num_blocks_per_input=16,
        )
        print_test_case(test_case17)
        test_case_batched[bf16_type](ctx, test_case17, fill_random)

        comptime test_case18 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=5,
            block_size=1024,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case18)
        test_case_batched[bf16_type](ctx, test_case18, fill_random)

        comptime test_case19 = TestCase[_sampling=False](
            N=1024,
            K=5,
            block_size=256,
            batch_size=64,
        )
        print_test_case(test_case19)
        test_case_batched[bf16_type](ctx, test_case19, fill_random)

        # Test with identical values
        comptime test_case20 = TestCase[_sampling=False](
            N=50,
            K=25,
            block_size=256,
            batch_size=2,
        )
        print_test_case(test_case20)
        test_case_batched[dtype](ctx, test_case20, fill_constant)

        comptime test_case_21 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=75,
            block_size=512,
            batch_size=2,
            num_blocks_per_input=8,
        )
        print_test_case(test_case_21)
        test_case_batched[.float32](ctx, test_case_21, fill_random)

        comptime test_case_22 = TestCase[_sampling=False](
            N=50,
            K=25,
            block_size=1024,
            batch_size=1,
        )
        print_test_case(test_case_22)
        test_case_batched[.float32](ctx, test_case_22, fill_random)

        # Test with zero batch size
        comptime test_case_23 = TestCase[_sampling=False](
            N=1024,
            K=1,
            block_size=256,
            batch_size=0,
        )
        print_test_case(test_case_23)
        test_case_batched[dtype](ctx, test_case_23, fill_iota)

        # Run minimum top-k tests
        test_min_topk[dtype](ctx)

        # Test all NaN input
        comptime test_nan_256 = TestCase[_sampling=True](
            name="[All NaN]",
            N=256,
            K=1,
            block_size=256,
            batch_size=1,
        )
        print_test_case(test_nan_256)
        test_case_batched[dtype](ctx, test_nan_256, fill_nan)

        # Test OOB (N=257) + All NaN
        comptime test_nan_257 = TestCase[_sampling=True](
            name="[All NaN + OOB]",
            N=257,
            K=1,
            block_size=256,
            batch_size=1,
        )
        print_test_case(test_nan_257)
        test_case_batched[dtype](ctx, test_nan_257, fill_nan)

        # All NaN + Batch=4
        comptime test_nan_batch = TestCase[_sampling=True](
            name="[All NaN + Batch=4]",
            N=256,
            K=1,
            block_size=256,
            batch_size=4,
        )
        print_test_case(test_nan_batch)
        test_case_batched[dtype](ctx, test_nan_batch, fill_nan)

        # Regression: large top_k (2048) forces the public topk_gpu to reduce
        # num_blocks_per_input so the stage-2 reduction fits the device
        # per-block shared-memory limit. Verify the result still matches the CPU
        # reference (values and indices) with the clamp active. Pre-fix these
        # launched with CUDA_ERROR_INVALID_VALUE.
        comptime test_case_large_k = TestCaseMultiRank[
            _sampling=False, rank=2, _largest=True
        ](input_shape=IndexList[2](2, 4096), K=2048)
        print_test_case(test_case_large_k)
        test_case_multi_rank[dtype](ctx, test_case_large_k, fill_iota)

        # Same, but with an explicit num_blocks_per_input=8 that the clamp must
        # override (8 * 2048 candidates would overflow stage-2 shared memory).
        comptime test_case_large_k_nb = TestCaseMultiRank[
            _sampling=False, rank=2, _largest=True
        ](input_shape=IndexList[2](2, 4096), K=2048, num_blocks_per_input=8)
        print_test_case(test_case_large_k_nb)
        test_case_multi_rank[dtype](ctx, test_case_large_k_nb, fill_iota)

        # Run multi-rank tests
        test_multi_rank[dtype, False](ctx)
        test_multi_rank[dtype, True](ctx)
        test_multi_rank[bf16_type, False](ctx)
        test_multi_rank[bf16_type, True](ctx)

        # Regression: temperature == 0 in gumbel path must not yield NaN or -1.
        test_gumbel_zero_temperature[dtype](ctx)

        test_gumbel_fused_matches_two_kernel[dtype](ctx, N=256, batch_size=2)
        test_gumbel_fused_matches_two_kernel[dtype](ctx, N=32000, batch_size=8)
        test_gumbel_fused_matches_two_kernel[dtype](
            ctx, N=llama3_vocab_size, batch_size=4
        )
        test_gumbel_fused_matches_two_kernel[bf16_type](
            ctx, N=32000, batch_size=8
        )

        # Regression: a per-row k=0 must emit the skip-token sentinel (0), not garbage.
        test_topk_zero_k_row[dtype](ctx)
