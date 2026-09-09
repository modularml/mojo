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

from std.random import random_float64

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.argmaxmin import argmax, argmin
from nn.argmaxmin_gpu import argmax_gpu, argmin_gpu
from std.testing import assert_equal
from std.utils.index import IndexList


def test_argmaxmin_gpu[
    dtype: DType,
    output_type: DType,
    fill_fn: def[rank: Int, dtype: DType](
        TileTensor[mut=True, dtype, ...]
    ) capturing[_] -> None,
    largest: Bool = True,
    rank: Int = 2,
](
    ctx: DeviceContext, N: Int, batch_size: Int = 12, num_batches: Int = 6
) raises:
    # Instantiate data in host memory
    var in_shape: IndexList[rank]
    var out_shape: IndexList[rank]

    comptime if rank == 1:
        out_shape = IndexList[rank](1)
        in_shape = IndexList[rank](N)
    elif rank == 2:
        out_shape = IndexList[rank](batch_size, 1)
        in_shape = IndexList[rank](batch_size, N)
    elif rank == 3:
        out_shape = IndexList[rank](num_batches, batch_size, 1)
        in_shape = IndexList[rank](num_batches, batch_size, N)
    else:
        raise Error("Test case doesn't support rank above 3 (just add it)")

    # Compute sizes
    var in_size = 1
    var out_size = 1
    for i in range(rank):
        in_size *= in_shape[i]
        out_size *= out_shape[i]

    # Allocate host memory
    var in_host_ptr = ctx.enqueue_create_host_buffer[dtype](in_size)
    var in_host = TileTensor(
        in_host_ptr,
        row_major(Coord(in_shape)),
    )
    var out_idxs_host_ptr = ctx.enqueue_create_host_buffer[output_type](
        out_size
    )
    var _out_idxs_host = TileTensor(
        out_idxs_host_ptr,
        row_major(Coord(out_shape)),
    )

    # Fill the buffer with consecutive values
    fill_fn[rank](in_host)

    # Allocate device buffers
    var device_in = ctx.enqueue_create_buffer[dtype](in_size)
    var device_out_idxs = ctx.enqueue_create_buffer[output_type](out_size)

    ctx.enqueue_copy(device_in, in_host_ptr)

    # Create device TileTensors
    var device_in_tensor = TileTensor(
        device_in,
        row_major(Coord(in_shape)),
    )
    var device_out_tensor = TileTensor(
        device_out_idxs,
        row_major(Coord(out_shape)),
    )

    comptime if largest:
        argmax_gpu(
            ctx,
            device_in_tensor,
            device_out_tensor,
        )
    else:
        argmin_gpu(
            ctx,
            device_in_tensor,
            device_out_tensor,
        )

    ctx.enqueue_copy(out_idxs_host_ptr, device_out_idxs)
    ctx.synchronize()

    # Test for correctness against CPU reference
    var out_idxs_cpu_ptr = ctx.enqueue_create_host_buffer[.int64](out_size)
    var out_idxs_cpu = TileTensor(
        out_idxs_cpu_ptr,
        row_major(Coord(out_shape)),
    )

    comptime if largest:
        argmax(
            in_host,
            rank - 1,
            out_idxs_cpu,
        )
    else:
        argmin(
            in_host,
            rank - 1,
            out_idxs_cpu,
        )

    for i in range(out_size):
        assert_equal(
            out_idxs_host_ptr[i],
            out_idxs_cpu_ptr[i].cast[output_type](),
        )

    # Cleanup device buffers
    _ = device_in^
    _ = device_out_idxs^


def _test_argmaxmin_gpu_helper_2[
    idx_type: DType,
    fill_fn: def[rank: Int, dtype: DType](
        TileTensor[mut=True, dtype, ...]
    ) capturing[_] -> None,
    largest: Bool,
](ctx: DeviceContext) raises:
    test_argmaxmin_gpu[
        DType.float32, idx_type, fill_fn, largest=largest, rank=1
    ](ctx, N=102_400)
    test_argmaxmin_gpu[
        DType.float32, idx_type, fill_fn, largest=largest, rank=2
    ](ctx, N=16_384, batch_size=32)
    test_argmaxmin_gpu[
        DType.float32, idx_type, fill_fn, largest=largest, rank=3
    ](ctx, N=1024, batch_size=12, num_batches=10)


def test_argmaxmin_gpu_helper[
    idx_type: DType,
    fill_fn: def[rank: Int, dtype: DType](
        TileTensor[mut=True, dtype, ...]
    ) capturing[_] -> None,
](ctx: DeviceContext) raises:
    # argmax
    _test_argmaxmin_gpu_helper_2[idx_type, fill_fn, largest=True](ctx)

    # argmin
    _test_argmaxmin_gpu_helper_2[idx_type, fill_fn, largest=False](ctx)


def main() raises:
    @__parameter
    def fill_random[
        rank: Int, dtype: DType
    ](buffer: TileTensor[mut=True, dtype, ...]):
        comptime min_val = -1e9
        comptime max_val = 1e9
        var total_elements = buffer.num_elements()
        for i in range(total_elements):
            var random_value = random_float64(min_val, max_val)
            buffer.raw_store(i, random_value.cast[dtype]())

    with DeviceContext() as ctx:  # argmax tests
        # index
        test_argmaxmin_gpu_helper[.int, fill_random](ctx)

        # int64
        test_argmaxmin_gpu_helper[.int64, fill_random](ctx)

        # int32
        test_argmaxmin_gpu_helper[.int32, fill_random](ctx)

        # uint64
        test_argmaxmin_gpu_helper[.uint64, fill_random](ctx)
