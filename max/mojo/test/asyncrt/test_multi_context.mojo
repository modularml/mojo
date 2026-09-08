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

from asyncrt_test_utils import create_test_device_context
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal


def vec_func(
    in0: Pointer[Float32, MutAnyOrigin],
    in1: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    len_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[unsafe_offset=tid] = in0[unsafe_offset=tid] + in1[unsafe_offset=tid]


def test_multi_function() raises:
    var ctx1 = create_test_device_context()
    var ctx2 = create_test_device_context()
    _run_test_multi_function(ctx1, ctx2)


def _run_test_multi_function(ctx1: DeviceContext, ctx2: DeviceContext) raises:
    comptime length = 1024

    var in0_dev1 = ctx1.enqueue_create_buffer[.float32](length)
    var in0_dev2 = ctx2.enqueue_create_buffer[.float32](length)
    var in1_dev1 = ctx1.enqueue_create_buffer[.float32](length)
    var in1_dev2 = ctx2.enqueue_create_buffer[.float32](length)
    var out_dev1 = ctx1.enqueue_create_buffer[.float32](length)
    var out_dev2 = ctx2.enqueue_create_buffer[.float32](length)

    # Initialize the input and outputs with known values.
    with in0_dev1.map_to_host() as in0_host1, in0_dev2.map_to_host() as in0_host2:
        for i in range(length):
            in0_host1[i] = Float32(i)
            in0_host2[i] = Float32(i)

    # Setup right side constants.
    in1_dev1.enqueue_fill(2.0)
    in1_dev2.enqueue_fill(2.0)

    # Write known bad values to out_dev.
    out_dev1.enqueue_fill(101.0)
    out_dev2.enqueue_fill(102.0)

    var block_dim = 32

    ctx1.enqueue_function[vec_func](
        in0_dev1,
        in1_dev1,
        out_dev1,
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    ctx2.enqueue_function[vec_func](
        in0_dev2,
        in1_dev2,
        out_dev2,
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out_dev1.map_to_host() as out_host1, out_dev2.map_to_host() as out_host2:
        for i in range(length):
            if i < 10:
                print("at index", i, "the value is", out_host1[i])
                print("at index", i, "the value is", out_host2[i])
            assert_equal(
                out_host1[i],
                Float32(i + 2),
                String(
                    "at index ",
                    i,
                    " the value is ",
                    out_host1[i],
                ),
            )
            assert_equal(
                out_host2[i],
                Float32(i + 2),
                String(
                    "at index ",
                    i,
                    " the value is ",
                    out_host2[i],
                ),
            )


def main() raises:
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    # TestSuite.discover_tests[__functions_in_module()]().run()
    var suite = TestSuite()

    suite.test[test_multi_function]()

    suite^.run()
