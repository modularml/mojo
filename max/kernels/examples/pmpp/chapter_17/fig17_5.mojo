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

from std.collections import List
from spmv_utils import COOMatrix, generate_sparse_matrix, spmv_cpu, verify
from max.gpu import block_idx, thread_idx, block_dim
from max.gpu.host import DeviceContext
from std.atomic import Atomic


def spmv_coo_kernel(
    cooMatrix: COOMatrix,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < cooMatrix.numNonzeros:
        var row = Int(cooMatrix.rowIdx[i])
        var col = Int(cooMatrix.colIdx[i])
        var val = cooMatrix.value[i]

        # Atomic add to y[row]
        _ = Atomic.fetch_add(y + row, x[col] * val)


def main() raises:
    var rows = 1024
    var cols = 1024
    var sparsity: Float32 = 0.9

    var h_row_idx = List[UInt32]()
    var h_col_idx = List[UInt32]()
    var h_values = List[Float32]()

    generate_sparse_matrix(rows, cols, sparsity, h_row_idx, h_col_idx, h_values)

    var numNonzeros = len(h_values)
    print("COO Matrix: ", rows, "x", cols, ", ", numNonzeros, " non-zeros.")

    # Reference
    var h_x = List[Float32](capacity=cols)
    for _ in range(cols):
        h_x.append(1.0)

    var h_y_ref = List[Float32](capacity=rows)
    for _ in range(rows):
        h_y_ref.append(0.0)

    spmv_cpu(rows, cols, h_row_idx, h_col_idx, h_values, h_x, h_y_ref)

    # Prepare host pointers
    var h_rowIdx_ptr = alloc[UInt32](numNonzeros)
    var h_colIdx_ptr = alloc[UInt32](numNonzeros)
    var h_value_ptr = alloc[Float32](numNonzeros)
    var h_x_ptr = alloc[Float32](cols)
    var h_y_ptr = alloc[Float32](rows)  # Output on host

    for i in range(numNonzeros):
        h_rowIdx_ptr[i] = h_row_idx[i]
        h_colIdx_ptr[i] = h_col_idx[i]
        h_value_ptr[i] = h_values[i]

    for i in range(cols):
        h_x_ptr[i] = h_x[i]

    for i in range(rows):
        h_y_ptr[i] = 0.0  # Initialize host output

    var ctx = DeviceContext()

    # Device allocation
    var d_rowIdx_buf = ctx.enqueue_create_buffer[.uint32](numNonzeros)
    var d_colIdx_buf = ctx.enqueue_create_buffer[.uint32](numNonzeros)
    var d_value_buf = ctx.enqueue_create_buffer[.float32](numNonzeros)
    var d_x_buf = ctx.enqueue_create_buffer[.float32](cols)
    var d_y_buf = ctx.enqueue_create_buffer[.float32](rows)

    # Copy to device
    ctx.enqueue_copy(d_rowIdx_buf, h_rowIdx_ptr)
    ctx.enqueue_copy(d_colIdx_buf, h_colIdx_ptr)
    ctx.enqueue_copy(d_value_buf, h_value_ptr)
    ctx.enqueue_copy(d_x_buf, h_x_ptr)
    # Initialize y on device to 0
    var h_y_zeros = alloc[Float32](rows)
    for i in range(rows):
        h_y_zeros[i] = 0.0
    ctx.enqueue_copy(d_y_buf, h_y_zeros)
    h_y_zeros.free()

    var d_cooMatrix = COOMatrix(
        numNonzeros,
        rows,
        cols,
        d_rowIdx_buf.unsafe_ptr().as_unsafe_any_origin(),
        d_colIdx_buf.unsafe_ptr().as_unsafe_any_origin(),
        d_value_buf.unsafe_ptr().as_unsafe_any_origin(),
    )

    var blockSize = 256
    var numBlocks = (numNonzeros + blockSize - 1) // blockSize

    ctx.enqueue_function[spmv_coo_kernel](
        d_cooMatrix,
        d_x_buf,
        d_y_buf,
        grid_dim=numBlocks,
        block_dim=blockSize,
    )

    # Copy result back
    ctx.enqueue_copy(h_y_ptr, d_y_buf)
    ctx.synchronize()

    if verify(h_y_ref, h_y_ptr, rows):
        print("COO Kernel Test Passed!")
    else:
        print("COO Kernel Test Failed!")

    h_rowIdx_ptr.free()
    h_colIdx_ptr.free()
    h_value_ptr.free()
    h_x_ptr.free()
    h_y_ptr.free()
