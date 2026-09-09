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

import extensibility
from max.gpu.host import DeviceContext
from max.gpu.host.device_context import DeviceExternalFunction
from max.gpu import global_idx
from std.math import ceildiv
from std.os import abort, getenv
from extensibility import (
    foreach,
    DynamicTensor,
    VariadicTensors,
    InputTensor,
    OutputTensor,
    InputVariadicTensors,
)
from extensibility import OutputVariadicTensors
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from std.utils.index import IndexList


@extensibility.register("my_add")
struct MyAdd:
    @staticmethod
    def execute(
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        y: InputTensor[dtype=output.dtype, rank=output.rank, ...],
    ):
        output[0] = x[0] + y[0]


@extensibility.register_shape_function("my_add")
def my_add_shape(
    x: InputTensor,
    y: InputTensor,
) raises -> IndexList[x.rank]:
    raise "NotImplemented"


@extensibility.register("op_with_device_context")
struct OpWidthDeviceContext:
    @staticmethod
    def execute(
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ):
        output[0] = x[0]


@extensibility.register_shape_function("op_with_device_context")
def op_with_device_context_shape(
    x: InputTensor,
) raises -> IndexList[x.rank]:
    raise "NotImplemented"


@extensibility.register("op_with_multiple_outputs")
struct OpWithMultipleOutputs:
    @staticmethod
    def execute(
        out0: OutputTensor,
        out1: OutputTensor[dtype=out0.dtype, rank=out0.rank, ...],
        x: InputTensor[dtype=out0.dtype, rank=out0.rank, ...],
    ):
        out0[0] = 2 * x[0]
        out1[0] = 4 * x[0]


@extensibility.register_shape_function("op_with_multiple_outputs")
def op_with_multiple_outputs_shape(
    x: InputTensor,
) raises -> IndexList[x.rank]:
    raise "NotImplemented"


@extensibility.register("op_without_outputs")
struct OpWithoutOutputs:
    @staticmethod
    def execute(
        x: InputTensor,
    ):
        print(x[0])


struct MyIntMemory(Movable):
    var val: Int

    def __init__(out self, val: Int):
        self.val = val

    def __deinit__(deinit self):
        print("MyInt ")


@extensibility.register("make_my_int_memory")
struct MakeMyIntMemory:
    @staticmethod
    def execute(x: InputTensor[dtype=.int32, rank=1, ...]) -> MyIntMemory:
        return MyIntMemory(Int(x[0]))


@fieldwise_init
struct MyIntReg(TrivialRegisterPassable):
    var val: Int


@extensibility.register("make_my_int_reg")
struct MakeMyIntReg:
    @staticmethod
    def execute(x: InputTensor[dtype=.int32, rank=1, ...]) -> MyIntReg:
        return MyIntReg(Int(x[0]))


@extensibility.register("variadic_input_to_output")
struct VariadicInputToOutput:
    @staticmethod
    def execute[
        dtype: DType,
        size: Int,
    ](
        output: OutputVariadicTensors[dtype=dtype, rank=1, size=size, ...],
        bias: InputTensor[dtype=dtype, rank=1, ...],
        input: InputVariadicTensors[dtype=dtype, rank=1, size=size, ...],
    ):
        comptime for i in range(size):
            for j in range(input[i].size()):
                output[i][j] = input[i][j]
            output[i][0] += bias[0]


@extensibility.register("variadic_add")
struct VariadicAdd:
    @staticmethod
    def execute[
        dtype: DType,
        size: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=1, ...],
        bias: InputTensor[dtype=dtype, rank=1, ...],
        input: InputVariadicTensors[dtype=dtype, rank=1, size=size, ...],
    ):
        for i in range(output.size()):
            output[i] = bias[i]

            comptime for j in range(size):
                output[i] += input[j][i]


@extensibility.register("binary_kernel_with_raises")
struct BinaryKernelWithRaises:
    @staticmethod
    def execute(
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        y: InputTensor[dtype=output.dtype, rank=output.rank, ...],
    ) raises:
        output[0] = x[0] + y[0]


@extensibility.register_shape_function("binary_kernel_with_raises")
def binary_kernel_with_raises_shape(
    x: InputTensor,
    y: InputTensor,
) raises -> IndexList[x.rank]:
    raise "NotImplemented"


@extensibility.register("mutable_input_tensor")
struct MutableInputTensorKernel:
    @staticmethod
    def execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.unsafe_store(0, 0)


@extensibility.register("op_with_int_parameter")
struct OpWithIntParameter[IntParameter: Int]:
    @staticmethod
    def execute(
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
    ):
        output[0] = x[0]
        print(Self.IntParameter)


@extensibility.register("op_with_dtype_parameter")
struct OpWithDTypeParameter[DTypeParameter: DType]:
    @staticmethod
    def execute(
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
    ):
        output[0] = x[0]
        print(Self.DTypeParameter)


@extensibility.register("op_with_string_parameter")
struct OpWithStringParameter[StringParameter: String]:
    @staticmethod
    def execute(
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
    ):
        output[0] = x[0]
        print(Self.StringParameter)


@extensibility.register("op_with_string_slice_parameter")
struct OpWithStringSliceParameter[StringParameter: StringSlice]:
    @staticmethod
    def execute(
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
    ):
        output[0] = x[0]
        print(Self.StringParameter)


@extensibility.register("op_with_static_string_parameter")
struct OpWithStaticStringParameter[StringParameter: StaticString]:
    @staticmethod
    def execute(
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
    ):
        output[0] = x[0]
        print(Self.StringParameter)


@extensibility.register("op_with_external_cubin")
struct ExternalCubinVecAdd:
    """Custom op that uses an external cubin for vector addition."""

    @staticmethod
    def execute[
        target: StaticString
    ](
        output: OutputTensor[rank=1, ...],
        lhs: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        rhs: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert target == "gpu"
        var gpu_ctx = ctx

        var external_func: DeviceExternalFunction
        with open(getenv("CUBIN_PATH"), "r") as file:
            var cubin_data = file.read_bytes()

            external_func = DeviceExternalFunction(
                gpu_ctx,
                function_name="vec_add",  # matches extern "C" name
                # DeviceExternalFunction takes a StringSlice, which is probably wrong.
                # The cubin is [very, very likely] invalid UTF8.
                asm=String(StringSlice(unsafe_from_utf8=cubin_data)),
            )

        var length = output.dim_size(0)
        var block_dim = 32
        var grid_dim = (length + block_dim - 1) // block_dim

        # Execute the external cubin kernel
        gpu_ctx.enqueue_function(
            external_func,
            lhs.unsafe_ptr(),
            rhs.unsafe_ptr(),
            output.unsafe_ptr(),
            length,
            grid_dim=(grid_dim,),
            block_dim=(block_dim,),
        )


@extensibility.register("intentional_gpu_crash")
struct IntentionalGpuCrash:
    """A custom op that launches a GPU kernel which executes a trap instruction.

    This causes a real GPU hardware fault (e.g. CUDA_ERROR_ILLEGAL_INSTRUCTION)
    to test that runtime error reporting includes source notes. Only supported
    on NVIDIA GPUs; ROCm handles GPU traps by calling host-side abort() which
    kills the process.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=1, ...],
        x: InputTensor[dtype=output.dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert target == "gpu"
        var gpu_ctx = ctx

        def crash_kernel():
            abort()

        gpu_ctx.enqueue_function[crash_kernel](grid_dim=(1,), block_dim=(1,))
        gpu_ctx.synchronize()


@extensibility.register("fill_from_host_scalar")
struct FillFromHostScalar:
    """Fills the GPU output with the value of a host-resident scalar input.

    The value is read on the host at enqueue time -- the dispatch-metadata
    pattern (attention cache lengths and friends). Under device-graph
    synthesis the read happens while recording, so the value bakes into the
    recorded kernel and the input's contents must key the graph cache:
    a changed value must re-record, never replay stale dispatch.
    """

    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        h: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var value = h.unsafe_ptr()[unsafe_offset=0]
        var ptr = output.unsafe_ptr()
        var length = Int32(output.size())

        def fill_kernel() {var}:
            var tid = global_idx.x
            if tid >= Int(length):
                return
            ptr[unsafe_offset=tid] = value

        comptime block_dim = 256
        ctx.enqueue_function(
            fill_kernel,
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )
