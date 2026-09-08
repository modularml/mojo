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


def vec_func[
    op: def(Float32, Float32) capturing[_] -> Float32
](
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
    output[unsafe_offset=tid] = op(
        in0[unsafe_offset=tid], in1[unsafe_offset=tid]
    )


def test_capture_2_5() raises:
    var ctx = create_test_device_context()
    run_captured_func(ctx, 2.5)


def test_capture_neg_1_5() raises:
    var ctx = create_test_device_context()
    run_captured_func(ctx, -1.5)


@no_inline
def run_captured_func(ctx: DeviceContext, captured: Float32) raises:
    print("-")
    print("run_captured_func(", captured, "):")

    comptime length = 1024

    var in0 = ctx.enqueue_create_buffer[.float32](length)
    var in1 = ctx.enqueue_create_buffer[.float32](length)
    in1.enqueue_fill(2)
    var out = ctx.enqueue_create_buffer[.float32](length)

    # Initialize the input and outputs with known values.
    with in0.map_to_host() as in0_host, out.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            out_host[i] = Float32(length + i)

    @__parameter
    def add_with_captured(left: Float32, right: Float32) -> Float32:
        return left + right + captured

    var block_dim = 32

    comptime kernel = vec_func[add_with_captured]
    var kernel_func = ctx.compile_function[kernel]()
    ctx.enqueue_function(
        kernel_func,
        in0,
        in1,
        out,
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out.map_to_host() as out_host:
        for i in range(length):
            if i < 10:
                print("at index", i, "the value is", out_host[i])
            assert_equal(
                out_host[i],
                Float32(Float32(i + 2) + captured),
                String("at index ", i, " the value is ", out_host[i]),
            )


def main() raises:
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    # TestSuite.discover_tests[__functions_in_module()]().run()
    var suite = TestSuite()

    suite.test[test_capture_2_5]()
    suite.test[test_capture_neg_1_5]()

    suite^.run()
