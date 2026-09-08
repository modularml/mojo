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

from std.sys import is_gpu

from asyncrt_test_utils import create_test_device_context
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal
from std.sys import has_apple_gpu_accelerator

comptime T = DType.float32 if has_apple_gpu_accelerator() else DType.float64
comptime S = Scalar[T]


trait MaybeZeroSized(TrivialRegisterPassable):
    def value(self) -> S:
        ...


@fieldwise_init
struct ZeroSized(
    DevicePassable, MaybeZeroSized, TrivialRegisterPassable, Writable
):
    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "ZeroSized"

    @always_inline
    def value(self) -> S:
        return 2

    def write_to(self, mut writer: Some[Writer]):
        comptime assert not is_gpu(), "ZeroSized is not supported on GPUs"
        writer.write("ZeroSized(")
        writer.write(self.value())
        writer.write(")")


@fieldwise_init
struct NotZeroSized(
    DevicePassable, MaybeZeroSized, TrivialRegisterPassable, Writable
):
    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "ZeroSized"

    var val: S

    def __init__(out self):
        self.val = 2

    @always_inline
    def value(self) -> S:
        return self.val

    def write_to(self, mut writer: Some[Writer]):
        comptime assert not is_gpu(), "ZeroSized is not supported on GPUs"
        writer.write("NotZeroSized(")
        writer.write(self.value())
        writer.write(")")


def _vec_func_zero(
    zs: ZeroSized,
    in0: Pointer[S, MutAnyOrigin],
    in1: Pointer[S, MutAnyOrigin],
    output: Pointer[S, MutAnyOrigin],
    len_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[unsafe_offset=tid] = (
        in0[unsafe_offset=tid] + in1[unsafe_offset=tid] + zs.value()
    )


def _vec_func_not_zero(
    zs: NotZeroSized,
    in0: Pointer[S, MutAnyOrigin],
    in1: Pointer[S, MutAnyOrigin],
    output: Pointer[S, MutAnyOrigin],
    len_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[unsafe_offset=tid] = (
        in0[unsafe_offset=tid] + in1[unsafe_offset=tid] + zs.value()
    )


def _vec_func[
    zero_sized_t: MaybeZeroSized
](
    zs: zero_sized_t,
    in0: Pointer[S, MutAnyOrigin],
    in1: Pointer[S, MutAnyOrigin],
    output: Pointer[S, MutAnyOrigin],
    len_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[unsafe_offset=tid] = (
        in0[unsafe_offset=tid] + in1[unsafe_offset=tid] + zs.value()
    )


def test_function_compilation() raises:
    var ctx = create_test_device_context()
    _run_test_function_compilation(ctx)


def _run_test_function_compilation(ctx: DeviceContext) raises:
    # Compile all combinations with and without declaring the trait in
    # the signature.

    print("Compiling _vec_func[NotZeroSized]")
    var compiled_vec_func_0 = ctx.compile_function[_vec_func[NotZeroSized]]()

    print("Compiling _vec_func[ZeroSizet]")
    var compiled_vec_func_1 = ctx.compile_function[_vec_func[ZeroSized]]()

    print("Compiling _vec_func_not_zero")
    var compiled_vec_func_2 = ctx.compile_function[_vec_func_not_zero]()

    print("Compiling _vec_func_zero")
    var compiled_vec_func_3 = ctx.compile_function[_vec_func_zero]()

    _ = compiled_vec_func_0
    _ = compiled_vec_func_1
    _ = compiled_vec_func_2
    _ = compiled_vec_func_3


def test_function_checked() raises:
    var ctx = create_test_device_context()
    _run_test_function_checked(ctx)


def _run_test_function_checked(ctx: DeviceContext) raises:
    comptime length = 1024
    comptime block_dim = 32

    comptime zero_sized_t = ZeroSized
    comptime vec_func = _vec_func[zero_sized_t]
    # alias vec_func = _vec_func_not_zero
    # alias vec_func = _vec_func_zero

    var zs = zero_sized_t()
    print(zs)

    var scalar: S = 2

    # Initialize the input and outputs with known values.
    var in0 = ctx.enqueue_create_buffer[T](length)
    var out = ctx.enqueue_create_buffer[T](length)
    with in0.map_to_host() as in0_host, out.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = Scalar[T](i)
            out_host[i] = Scalar[T](length + i)
    var in1 = ctx.enqueue_create_buffer[T](length)
    in1.enqueue_fill(scalar)

    print("compiling vec_func")
    var compiled_vec_func = ctx.compile_function[vec_func]()
    print("calling vec_func")
    ctx.enqueue_function(
        compiled_vec_func,
        zs,
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
                Scalar[T](i + 4),
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

    suite.test[test_function_compilation]()
    suite.test[test_function_checked]()

    suite^.run()
