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

from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from asyncrt_test_utils import create_test_device_context
from max.gpu import *
from max.gpu.host import DeviceContext, get_gpu_target
from std.testing import TestSuite, assert_equal

from std.utils import IndexList
from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import Index


def run_elementwise[dtype: DType](ctx: DeviceContext) raises:
    print("-")
    print("run_elementwise[", dtype, "]:")

    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()

    comptime rank = 2
    comptime dim_x = 2
    comptime dim_y = 8
    comptime length = dim_x * dim_y

    var in0 = ctx.enqueue_create_buffer[dtype](length)
    var out = ctx.enqueue_create_buffer[dtype](length)

    # Initialize the input and outputs with known values.
    with in0.map_to_host() as in_host, out.map_to_host() as out_host:
        for i in range(length):
            in_host[i] = Scalar[dtype](i)
            out_host[i] = Scalar[dtype](length + i)

    var in_buffer = Span(unsafe_ptr=in0.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out.unsafe_ptr(), length=length)

    @always_inline
    @__copy_capture(in_buffer, out_buffer)
    @__parameter
    def func[simd_width: Int, alignment: Int = 1](idx0: Coord):
        var idx = rebind[IndexList[2]](coord_to_index_list(idx0))
        out_buffer.unsafe_ptr().unsafe_store(
            idx[0] * dim_y + idx[1],
            in_buffer.unsafe_ptr().unsafe_load[width=simd_width](
                idx[0] * dim_y + idx[1]
            )
            + 42,
        )

    elementwise[func, pack_size, target="gpu"](
        (2, 8),
        ctx,
    )

    with out.map_to_host() as out_host:
        for i in range(length):
            print("at index", i, "the value is", out_host[i])
            assert_equal(
                out_host[i],
                Scalar[dtype](i + 42),
                String(
                    "at index ",
                    i,
                    " the value is ",
                    out_host[i],
                ),
            )


def test_elementwise_float32() raises:
    var ctx = create_test_device_context()
    run_elementwise[.float32](ctx)


def test_elementwise_bfloat16() raises:
    var ctx = create_test_device_context()
    run_elementwise[.bfloat16](ctx)


def test_elementwise_float16() raises:
    var ctx = create_test_device_context()
    run_elementwise[.float16](ctx)


def test_elementwise_int8() raises:
    var ctx = create_test_device_context()
    run_elementwise[.int8](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
