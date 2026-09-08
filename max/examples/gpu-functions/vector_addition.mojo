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

from std.math import ceildiv
from std.sys import has_accelerator

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime float_dtype = DType.float32
comptime VECTOR_WIDTH = 10
comptime BLOCK_SIZE = 5
comptime layout = row_major[VECTOR_WIDTH]()


def main() raises:
    comptime assert has_accelerator(), "This example requires a supported GPU"

    # Get context for the attached GPU
    var ctx = DeviceContext()

    # Allocate data on the GPU address space
    var lhs_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
    var rhs_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
    var out_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)

    # Fill in values across the entire width
    lhs_buffer.enqueue_fill(1.25)
    rhs_buffer.enqueue_fill(2.5)

    # Wrap the device buffers in tensors
    var lhs_tensor = TileTensor(lhs_buffer, layout)
    var rhs_tensor = TileTensor(rhs_buffer, layout)
    var out_tensor = TileTensor(out_buffer, layout)

    # Calculate the number of blocks needed to cover the vector
    var grid_dim = ceildiv(VECTOR_WIDTH, BLOCK_SIZE)

    # Launch the vector_addition function as a GPU kernel
    ctx.enqueue_function[vector_addition](
        lhs_tensor,
        rhs_tensor,
        out_tensor,
        Int32(VECTOR_WIDTH),
        grid_dim=grid_dim,
        block_dim=BLOCK_SIZE,
    )

    # Map to host so that values can be printed from the CPU
    with out_buffer.map_to_host() as host_buffer:
        var host_tensor = TileTensor(host_buffer, layout)
        print("Resulting vector:", host_tensor)


def vector_addition(
    lhs_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
    rhs_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
    out_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
    size_dev: Int32,
):
    """The calculation to perform across the vector on the GPU."""
    var size = Int(size_dev)
    var global_tid = global_idx.x
    if global_tid < size:
        out_tensor[global_tid] = lhs_tensor[global_tid] + rhs_tensor[global_tid]
