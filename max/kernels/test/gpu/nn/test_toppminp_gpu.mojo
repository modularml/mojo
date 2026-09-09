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

from std.math import iota
from std.random import random_float64

from max.algorithm.functional import parallelize_over_rows
from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, Bencher, BenchId
from max.gpu.host import DeviceContext
from layout import (
    Idx,
    Coord,
    DefaultEngine,
    TileTensor,
    row_major,
)
from nn.softmax import softmax_inline
from nn.toppminp_gpu import min_p_sampling_gpu, top_p_sampling_gpu
from std.testing import assert_almost_equal, assert_equal

from std.utils import IndexList

comptime DEBUG_BENCH = False
comptime PRINT_OUTPUT = False


struct TestCase[_dtype: DType, _out_idx_type: DType, _is_top_p: Bool](
    ImplicitlyCopyable
):
    comptime is_top_p = Self._is_top_p
    comptime dtype = Self._dtype
    comptime out_idx_type = Self._out_idx_type
    var batch_size: Int
    var vocab_size: Int
    var temperature: Scalar[Self.dtype]
    var p_threshold: Scalar[Self.dtype]

    def __init__(
        out self,
        batch_size: Int,
        vocab_size: Int,
        temperature: Scalar[Self.dtype] = Scalar[Self.dtype](1.0),
        p_threshold: Scalar[Self.dtype] = Scalar[Self.dtype](0.9),
    ):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.p_threshold = p_threshold


def time_kernel[](
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

    m.bench_function(bench_func, BenchId(kernel_name))


def fill_random[
    dtype: DType
](
    mut buffer: TileTensor[
        mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
    ]
):
    comptime min_val = -1e6
    comptime max_val = 1e6
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.raw_store(i, random_value.cast[dtype]())


def fill_iota[
    dtype: DType
](
    mut buf: TileTensor[
        mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
    ]
):
    iota(buf._storage, buf.layout.product())


def merge[
    dtype: DType,
](
    mut buf: TileTensor[
        mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
    ],
    start: Int,
    mid: Int,
    end: Int,
):
    """Merge two sorted subarrays into one sorted array."""
    var left_size = mid - start
    var right_size = end - mid

    # Create temporary arrays
    var left_ptr = List[Scalar[dtype]](unsafe_uninit_length=left_size)
    var right_ptr = List[Scalar[dtype]](unsafe_uninit_length=right_size)

    # Copy data to temporary arrays
    for i in range(left_size):
        left_ptr[i] = buf.raw_load(start + i)
    for i in range(right_size):
        right_ptr[i] = buf.raw_load(mid + i)

    # Merge back into original array
    var i = 0  # Index for left subarray
    var j = 0  # Index for right subarray
    var k = start  # Index for merged array

    while i < left_size and j < right_size:
        if left_ptr[i] >= right_ptr[j]:  # Use >= for descending order
            buf.raw_store(k, left_ptr[i])
            i += 1
        else:
            buf.raw_store(k, right_ptr[j])
            j += 1
        k += 1

    # Copy remaining elements if any
    while i < left_size:
        buf.raw_store(k, left_ptr[i])
        i += 1
        k += 1

    while j < right_size:
        buf.raw_store(k, right_ptr[j])
        j += 1
        k += 1


def merge_sort_recursive[
    dtype: DType
](
    mut buf: TileTensor[
        mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
    ],
    start: Int,
    end: Int,
):
    """Recursive merge sort implementation."""
    if end - start > 1:
        var mid = start + (end - start) // 2
        merge_sort_recursive(buf, start, mid)
        merge_sort_recursive(buf, mid, end)
        merge(buf, start, mid, end)


def sort_buf_descending[
    dtype: DType
](
    mut buf: TileTensor[
        mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
    ],
    vocab_size: Int,
):
    """Sort each batch separately in descending order using parallel merge sort.
    """
    comptime assert buf.flat_rank == 2, "rank must be 2"
    var batch_size = buf.num_elements() // vocab_size

    for batch_id in range(batch_size):
        var start = batch_id * vocab_size
        var end = start + vocab_size
        merge_sort_recursive(buf, start, end)


def test_is_sorted_descending[
    dtype: DType
](
    mut buf: TileTensor[
        mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
    ],
    vocab_size: Int,
) -> Bool:
    comptime assert buf.flat_rank == 2, "rank must be 2"
    var batch_size = buf.num_elements() // vocab_size
    var _sorted_flag = List(length=batch_size, fill=True)
    var sorted_flag = Span(_sorted_flag)

    def process_rows(start_batch: Int, end_batch: Int) {var}:
        # Process a chunk of batches
        for batch_id in range(start_batch, end_batch):
            var offset = batch_id * vocab_size
            for i in range(vocab_size - 1):
                if buf.raw_load(offset + i) < buf.raw_load(offset + i + 1):
                    print(
                        "[",
                        batch_id,
                        "][",
                        i,
                        "]: ",
                        buf.raw_load(offset + i),
                        " < ",
                        buf.raw_load(offset + i + 1),
                    )
                    sorted_flag[batch_id] = False
                    break

    comptime parallelism_grain_size = 1
    # Create shape with batch_size as the second dimension
    var shape = IndexList[1](
        batch_size,
    )
    parallelize_over_rows(process_rows, shape, 0, parallelism_grain_size)

    # Check if all batches are sorted by AND-ing all flags
    var all_sorted = True
    for i in range(batch_size):
        all_sorted = all_sorted and sorted_flag[i]

    return all_sorted


def print_test_case(test_case: TestCase):
    print(
        "==== Running",
        "Top-P" if test_case.is_top_p else "Min-P",
        ", dtype=",
        test_case.dtype,
        ", out_idx_type=",
        test_case.out_idx_type,
        "sampling with batch_size=",
        test_case.batch_size,
        ", vocab_size=",
        test_case.vocab_size,
        ", temperature=",
        test_case.temperature,
        ", p_threshold=",
        test_case.p_threshold,
    )


def test_case_sampling(
    ctx: DeviceContext,
    test_case: TestCase,
    fill_fn: Some[
        def[
            dtype: DType
        ](
            mut TileTensor[
                mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
            ]
        ) -> None
    ],
) raises:
    print_test_case(test_case)
    comptime rank = 2
    comptime dtype = test_case.dtype
    comptime out_idx_type = test_case.out_idx_type
    comptime is_top_p = test_case.is_top_p
    var batch_size = test_case.batch_size
    var vocab_size = test_case.vocab_size
    var temperature = test_case.temperature
    var p_threshold = test_case.p_threshold

    var _m: Bench

    comptime if DEBUG_BENCH:
        _m = Bench()

    # Create input tensors
    var in_logits_ptr = ctx.enqueue_create_host_buffer[dtype](
        batch_size * vocab_size
    )
    var in_logits = TileTensor(
        in_logits_ptr, row_major(Coord(batch_size, vocab_size))
    )
    var token_ids_ptr = ctx.enqueue_create_host_buffer[out_idx_type](
        batch_size * 1
    )
    var token_ids = TileTensor(
        token_ids_ptr, row_major(Coord(batch_size, Int(1)))
    )
    var p_thresholds_ptr = ctx.enqueue_create_host_buffer[dtype](batch_size)
    var p_thresholds = TileTensor(
        p_thresholds_ptr, row_major(Coord(batch_size))
    )

    # Fill tensors
    fill_fn(in_logits)
    for i in range(batch_size):
        p_thresholds.raw_store(i, p_threshold)

    # Create device buffers
    var device_in_buf = ctx.enqueue_create_buffer[dtype](
        batch_size * vocab_size
    )
    var device_token_ids_buf = ctx.enqueue_create_buffer[out_idx_type](
        batch_size * 1
    )
    var device_p_thresholds_buf = ctx.enqueue_create_buffer[dtype](batch_size)

    # Copy to device
    ctx.enqueue_copy(device_in_buf, in_logits._storage)
    ctx.enqueue_copy(device_p_thresholds_buf, p_thresholds._storage)

    # Copy to CPU and perform softmax & sort for correctness testing
    var in_logits_cpu_test_ptr = ctx.enqueue_create_host_buffer[dtype](
        batch_size * vocab_size
    )
    var probs_cpu_test_ptr = ctx.enqueue_create_host_buffer[dtype](
        batch_size * vocab_size
    )
    var in_logits_cpu_test = TileTensor(
        in_logits_cpu_test_ptr,
        row_major(batch_size, vocab_size),
    )
    var probs_cpu_test = TileTensor(
        probs_cpu_test_ptr,
        row_major(batch_size, vocab_size),
    )
    for i in range(in_logits.num_elements()):
        in_logits_cpu_test.raw_store(i, in_logits.raw_load(i) / temperature)

    softmax_inline[simd_width=1, rank=rank](
        in_logits_cpu_test,
        probs_cpu_test,
        axis=1,
    )
    sort_buf_descending(probs_cpu_test, vocab_size)

    var device_in_tensor = TileTensor(
        device_in_buf,
        row_major(batch_size, vocab_size),
    )
    var device_token_ids_tensor = TileTensor(
        device_token_ids_buf,
        row_major(batch_size, Idx[1]),
    )
    var device_p_thresholds_tensor = TileTensor(
        device_p_thresholds_buf,
        row_major(
            batch_size,
        ),
    )

    comptime if DEBUG_BENCH:

        @always_inline
        def run_func(ctx: DeviceContext) raises {var}:
            if is_top_p:
                top_p_sampling_gpu(
                    ctx,
                    device_p_thresholds_tensor,
                    device_in_tensor,
                    device_token_ids_tensor,
                    temperature=temperature,
                )
            else:
                min_p_sampling_gpu(
                    ctx,
                    device_p_thresholds_tensor,
                    device_in_tensor,
                    device_token_ids_tensor,
                    temperature=temperature,
                )
            ctx.synchronize()

        time_kernel(
            _m,
            ctx,
            "top-p-sampling" if is_top_p else "min-p-sampling",
            run_func,
        )

    # Run sampling
    comptime if is_top_p:
        top_p_sampling_gpu[_test_sort=True](
            ctx,
            device_p_thresholds_tensor,
            device_in_tensor,
            device_token_ids_tensor,
            temperature=temperature,
        )
    else:
        min_p_sampling_gpu[_test_sort=True](
            ctx,
            device_p_thresholds_tensor,
            device_in_tensor,
            device_token_ids_tensor,
            temperature=temperature,
        )
    # Copy results back
    ctx.enqueue_copy(token_ids._storage, device_token_ids_buf)
    ctx.enqueue_copy(in_logits._storage, device_in_buf)  # for testing
    ctx.synchronize()

    # Check if the probs are sorted in descending order, this validates the
    # softmax, and the sort. The random sampling is much simpler compared
    # to the softmax & sort kernels so this is a good check.
    assert_equal(test_is_sorted_descending(in_logits, vocab_size), True)
    # (More rigorous) Check if the sorted probs are the same as the cpu test
    for i in range(in_logits.num_elements()):
        try:
            assert_almost_equal(
                in_logits.raw_load(i), probs_cpu_test.raw_load(i), atol=5e-3
            )
        except e:
            print(
                "i: ",
                i,
                "in_logits: ",
                in_logits.raw_load(i),
                "probs_cpu_test: ",
                probs_cpu_test.raw_load(i),
            )
            raise e^

    comptime if PRINT_OUTPUT:
        print("Sampled token indices:", token_ids)

    comptime if DEBUG_BENCH:
        _m.dump_report()


def test_toppminp_gpu[
    dtype: DType,
    out_idx_type: DType,
](
    ctx: DeviceContext,
    fill_fn: Some[
        def[
            dtype: DType
        ](
            mut TileTensor[
                mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
            ]
        ) -> None
    ],
) raises:
    comptime test_case1 = TestCase[dtype, out_idx_type, _is_top_p=True](
        batch_size=1, vocab_size=1024, temperature=1.0, p_threshold=0.9
    )
    comptime test_case2 = TestCase[dtype, out_idx_type, _is_top_p=True](
        batch_size=16, vocab_size=32000, temperature=10.0, p_threshold=0.95
    )
    comptime test_case3 = TestCase[dtype, out_idx_type, _is_top_p=False](
        batch_size=64,
        vocab_size=128256,
        temperature=0.7,
        p_threshold=0.1,
    )

    test_case_sampling(ctx, test_case1, fill_fn)
    test_case_sampling(ctx, test_case2, fill_fn)
    test_case_sampling(ctx, test_case3, fill_fn)


def test_all_out_idx_types[
    dtype: DType,
](
    ctx: DeviceContext,
    fill_fn: Some[
        def[
            dtype: DType
        ](
            mut TileTensor[
                mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
            ]
        ) -> None
    ],
) raises:
    test_toppminp_gpu[dtype, DType.int32](ctx, fill_fn)
    test_toppminp_gpu[dtype, DType.int64](ctx, fill_fn)
    test_toppminp_gpu[dtype, DType.uint64](ctx, fill_fn)


def test_all_types(
    ctx: DeviceContext,
    fill_fn: Some[
        def[
            dtype: DType
        ](
            mut TileTensor[
                mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
            ]
        ) -> None
    ],
) raises:
    print("\n=== Testing Float32 ===")
    test_all_out_idx_types[.float32](ctx, fill_fn)
    print("\n=== Testing BFloat16 ===")
    test_all_out_idx_types[.bfloat16](ctx, fill_fn)


def main() raises:
    with DeviceContext() as ctx:
        print("\n====== Testing Fill Iota ======\n")
        test_all_types(ctx, fill_iota)
        print("\n====== Testing Fill Random ======\n")
        test_all_types(ctx, fill_random)
