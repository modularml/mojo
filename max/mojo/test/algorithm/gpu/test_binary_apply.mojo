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

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal, TestSuite


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


# Force the capture to be captured instead of inlined away.
@no_inline
def run_binary_add(ctx: DeviceContext, capture: Float32) raises:
    print("== run_binary_add")

    comptime length = 1024

    var in0 = ctx.enqueue_create_buffer[.float32](length)
    var in1 = ctx.enqueue_create_buffer[.float32](length)
    var out = ctx.enqueue_create_buffer[.float32](length)

    with in0.map_to_host() as in0_host, in1.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = 2

    @__parameter
    def add(lhs: Float32, rhs: Float32) -> Float32:
        return capture + lhs + rhs

    var block_dim = 32
    comptime kernel = vec_func[add]
    ctx.enqueue_function[kernel](
        in0,
        in1,
        out,
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    print(
        "number of captures:",
        ctx.compile_function[vec_func[add]]()._func_impl.num_captures,
    )
    assert_equal(
        ctx.compile_function[vec_func[add]]()._func_impl.num_captures,
        1,
    )
    ctx.synchronize()

    with out.map_to_host() as out_host:
        var expected: List[Float32] = [
            4.5,
            5.5,
            6.5,
            7.5,
            8.5,
            9.5,
            10.5,
            11.5,
            12.5,
            13.5,
        ]
        for i in range(10):
            print("at index", i, "the value is", out_host[i])
            assert_equal(out_host[i], expected[i])


def test_binary_apply() raises:
    with DeviceContext() as ctx:
        run_binary_add(ctx, 2.5)


def main() raises:
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    # TestSuite.discover_tests[__functions_in_module()]().run()
    var suite = TestSuite()

    suite.test[test_binary_apply]()

    suite^.run()
