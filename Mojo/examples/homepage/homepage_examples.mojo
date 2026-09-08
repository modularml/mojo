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

# Code examples from the mojolang.org landing page.

from std.algorithm import vectorize
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_gpu
from std.math import ceildiv
from std.python import Python, PythonObject
from std.reflection import reflect

from std.sys import has_accelerator, simd_width_of

from layout import TileTensor
from layout.tensor_engine import DefaultEngine
from layout.tile_layout import Layout, row_major
from extensibility import InputTensor, OutputTensor


comptime float_dtype = DType.float32

comptime size = 8

comptime layout = row_major[size]()

# GPU programming example
# The homepage example shows only the vector_add kernel.
# we use a.layout.size() instead of the comptime size because
# there's no explanation of where "size" comes from.


def vector_add(
    a: TileTensor[
        float_dtype,
        type_of(layout),
        Engine=DefaultEngine[element_width=1],
        ...,
    ],
    b: TileTensor[
        float_dtype,
        type_of(layout),
        Engine=DefaultEngine[element_width=1],
        ...,
    ],
    result: TileTensor[
        mut=True,
        float_dtype,
        type_of(layout),
        Engine=DefaultEngine[element_width=1],
        ...,
    ],
):
    var i = global_idx.x
    if i < a.layout.size():
        result[i] = a[i] + b[i]


def run_gpu_programming_example() raises:
    comptime assert (
        has_accelerator()
    ), "This example requires a supported accelerator"

    var ctx = DeviceContext()
    var a_buffer = ctx.enqueue_create_buffer[float_dtype](size)
    var b_buffer = ctx.enqueue_create_buffer[float_dtype](size)
    var result_buffer = ctx.enqueue_create_buffer[float_dtype](size)

    # Map input buffers to host to fill with values from CPU
    with a_buffer.map_to_host() as host_buffer:
        var a_tensor = TileTensor(host_buffer, layout)
        for i in range(size):
            a_tensor[i] = Float32(i)
        print("a vector:", a_tensor)

    with b_buffer.map_to_host() as host_buffer:
        var b_tensor = TileTensor(host_buffer, layout)
        for i in range(size):
            b_tensor[i] = Float32(i)
        print("b vector:", b_tensor)

    # Wrap device buffers in `TileTensor`
    var a_tensor = TileTensor(a_buffer, layout)
    var b_tensor = TileTensor(b_buffer, layout)
    var result_tensor = TileTensor(result_buffer, layout)
    print("wrapped")

    # The grid is divided up into blocks, making sure there's an extra
    # full block for any remainder. This hasn't been tuned for any specific
    # GPU.
    comptime BLOCK_SIZE = 16
    comptime num_blocks = ceildiv(size, BLOCK_SIZE)

    # wrapper closure for the actual kernel
    def wrapper() {var}:
        vector_add(a_tensor, b_tensor, result_tensor)

    # Launch the compiled function on the GPU. The last two named parameters
    # are the dimensions of the grid in blocks, and the block dimensions.
    ctx.enqueue_function(wrapper, grid_dim=(num_blocks), block_dim=(BLOCK_SIZE))

    # Move the output tensor back onto the CPU so that we can read the results.
    with result_buffer.map_to_host() as host_buffer:
        var host_tensor = TileTensor(host_buffer, layout)
        print("Resulting vector:", host_tensor)


# Python interop example


def mojo_square_array(array_obj: PythonObject) raises:
    comptime simd_width = simd_width_of[DType.int64]()
    var ptr = array_obj.ctypes.data.unsafe_get_as_pointer[.int64]()

    def pow[width: Int](i: Int) {mut ptr}:
        var elem = ptr.unsafe_load[width=width](i)
        ptr.unsafe_store[width=width](i, elem * elem)

    vectorize[simd_width](len(array_obj), pow)


def run_python_interop_example() raises:
    var np = Python.import_module("numpy")
    # values: List[Int64] = [1, 3, 5, 7, 11]
    var pylist = Python.list(1, 3, 5, 7, 11)
    var nparray = np.array(pylist, np.int64)
    mojo_square_array(nparray)
    print(nparray)


# Metaprogramming example


trait FauxEquatable(Deinitable):
    # Generic implementation using reflection: compare all fields
    def __eq__(self, other: Self) -> Bool:
        comptime r = reflect[Self]

        comptime for i in range(r.field_names().length):
            comptime assert conforms_to(r.field_types()[i], Equatable)
            if r.field_ref[i](self) != r.field_ref[i](other):
                return False
        return True


@fieldwise_init
struct EqTest(Copyable, FauxEquatable):
    var i: Int
    var s: String


def run_metaprogramming_example():
    var v1 = EqTest(1, "Lucy")
    var v2 = EqTest(1, "Lucy")
    var v3 = EqTest(2, "Linus")
    print(v1 == v2)
    print(v1 == v3)


def main() raises:
    run_gpu_programming_example()
    run_python_interop_example()
    run_metaprogramming_example()
