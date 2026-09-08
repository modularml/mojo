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
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal
from std.sys import has_apple_gpu_accelerator

comptime T = DType.float32 if has_apple_gpu_accelerator() else DType.float64
comptime S = Scalar[T]


struct TwoS(DevicePassable, TrivialRegisterPassable):
    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "TwoS"

    var s0: S
    var s1: S

    def __init__(out self, s: S):
        self.s0 = 1
        self.s1 = s


struct OneS(DevicePassable):
    comptime device_type: AnyType = TwoS

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(TwoS(self.s), target)

    @staticmethod
    def get_type_name() -> String:
        return "OneS"

    var s: S

    def __init__(out self, s: S):
        self.s = s


def vec_func(
    in0: Pointer[S, MutAnyOrigin],
    in1: Pointer[S, MutAnyOrigin],
    output: Pointer[S, MutAnyOrigin],
    s: TwoS,
    len_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[unsafe_offset=tid] = (
        in0[unsafe_offset=tid] + in1[unsafe_offset=tid] + s.s1 + s.s0
    )


def test_function_checked() raises:
    var ctx = create_test_device_context()
    _run_test_function_checked(ctx)


def _run_test_function_checked(ctx: DeviceContext) raises:
    comptime length = 1024
    comptime block_dim = 32

    var scalar: S = 2
    var ones = OneS(scalar)

    # Initialize the input and outputs with known values.
    var in0 = ctx.enqueue_create_buffer[T](length)
    var out = ctx.enqueue_create_buffer[T](length)
    with in0.map_to_host() as in0_host, out.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = Scalar[T](i)
            out_host[i] = Scalar[T](length + i)
    var in1 = ctx.enqueue_create_buffer[T](length)
    in1.enqueue_fill(scalar)

    var compiled_vec_func = ctx.compile_function[vec_func]()
    ctx.enqueue_function(
        compiled_vec_func,
        in0,
        in1,
        out,
        ones,
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
                Scalar[T](i + 5),
                String(
                    "at index ",
                    i,
                    " the value is ",
                    out_host[i],
                ),
            )


def test_function_experimental() raises:
    var ctx = create_test_device_context()
    _run_test_function_experimental(ctx)


def _run_test_function_experimental(ctx: DeviceContext) raises:
    comptime length = 1024
    comptime block_dim = 32

    var scalar: S = 2
    var ones = OneS(scalar)

    # Initialize the input and outputs with known values.
    var in0 = ctx.enqueue_create_buffer[T](length)
    var out = ctx.enqueue_create_buffer[T](length)
    with in0.map_to_host() as in0_host, out.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = Scalar[T](i)
            out_host[i] = Scalar[T](length + i)
    var in1 = ctx.enqueue_create_buffer[T](length)
    in1.enqueue_fill(scalar)

    var compiled_vec_func = ctx.compile_function[vec_func]()
    ctx.enqueue_function(
        compiled_vec_func,
        in0,
        in1,
        out,
        ones,
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
                Scalar[T](i + 5),
                String(
                    "at index ",
                    i,
                    " the value is ",
                    out_host[i],
                ),
            )


def main() raises:
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    # TestSuite.discover_tests[__functions_in_module()]().run()
    var suite = TestSuite()

    suite.test[test_function_checked]()
    suite.test[test_function_experimental]()

    suite^.run()
