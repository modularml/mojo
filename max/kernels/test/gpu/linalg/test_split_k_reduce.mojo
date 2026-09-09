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

from std.math import isclose
from std.random import rand

from max.gpu.host import DeviceContext
from layout import (
    TileTensor,
    Coord,
    Idx,
    row_major,
)
from linalg.matmul.gpu import split_k_reduce
from std.testing import assert_almost_equal

from std.utils import IndexList


def test_split_k_reduce_rank3[
    c_type: DType,
    work_space_type: DType,
](M: Int, N: Int, num_partitions: Int, ctx: DeviceContext) raises:
    print(
        "test_split_k_reduce_rank3",
        work_space_type,
        "->",
        c_type,
        num_partitions,
        M,
        N,
    )

    var c_host = ctx.enqueue_create_host_buffer[c_type](M * N)
    var c_host_ref = ctx.enqueue_create_host_buffer[c_type](M * N)

    # Random buffer for host computation.
    var epilogue_data_host = ctx.enqueue_create_host_buffer[c_type](M * N)
    rand[c_type](epilogue_data_host.unsafe_ptr(), M * N)

    var work_space_size = num_partitions * M * N
    var work_space_host = ctx.enqueue_create_host_buffer[work_space_type](
        work_space_size
    )
    rand[work_space_type](work_space_host.unsafe_ptr(), work_space_size)

    # Naive host reduction. The accumulation is in FP32 since CPU may not have
    # native BF16 instructions.
    for i in range(M * N):
        var sum = work_space_host[i].cast[.float32]()
        for j in range(1, num_partitions):
            sum += work_space_host[i + j * M * N].cast[.float32]()
        sum += epilogue_data_host[i].cast[.float32]()
        c_host_ref[i] = sum.cast[c_type]()

    var work_space_device = ctx.enqueue_create_buffer[work_space_type](
        num_partitions * M * N
    )
    var c_device = ctx.enqueue_create_buffer[c_type](M * N)
    var epilogue_data_device = ctx.enqueue_create_buffer[c_type](M * N)

    ctx.enqueue_copy(work_space_device, work_space_host)
    ctx.enqueue_copy(epilogue_data_device, epilogue_data_host)

    # Create TileTensors
    var c = TileTensor(
        c_device,
        row_major(Coord(Int(M), Int(N))),
    )
    var work_space = TileTensor(
        work_space_device,
        row_major(Coord(Int(num_partitions), Int(M), Int(N))),
    )
    var epilogue_buffer = TileTensor(
        epilogue_data_device,
        row_major(Coord(Int(M), Int(N))),
    )

    @__parameter
    @always_inline
    @__copy_capture(c, epilogue_buffer)
    def epilogue_fn[
        _dtype: DType, _width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[_dtype, _width]) capturing -> None:
        var another_val = rebind[SIMD[_dtype, _width]](
            epilogue_buffer.load[width=_width](Coord(idx[0], idx[1]))
        )
        c.store(
            Coord(idx[0], idx[1]),
            rebind[SIMD[c_type, _width]](val + another_val),
        )

    split_k_reduce[elementwise_lambda_fn=epilogue_fn](c, work_space, ctx)

    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    comptime rtol = 1e-4 if c_type == DType.float32 else 1e-2
    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i], rtol=rtol):
            print(
                i,
                c_host[i],
                c_host_ref[i],
                abs((c_host[i] - c_host_ref[i]) / c_host_ref[i]),
            )
        assert_almost_equal(c_host[i], c_host_ref[i], rtol=rtol)


def main() raises:
    with DeviceContext() as ctx:
        # Rank-3 work space.
        test_split_k_reduce_rank3[.bfloat16, .bfloat16](64, 64, 2, ctx)
        test_split_k_reduce_rank3[.bfloat16, .float32](32, 32, 5, ctx)
        test_split_k_reduce_rank3[.float32, .float32](32, 64, 3, ctx)
