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
from spmv_utils import ELLMatrix, generate_sparse_matrix, spmv_cpu, verify
from max.gpu import block_idx, thread_idx, block_dim
from max.gpu.host import DeviceContext


def spmv_ell_kernel(
    ellMatrix: ELLMatrix,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
):
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row < ellMatrix.numRows:
        var sum: Float32 = 0.0
        for t in range(ellMatrix.nnzPerRow):
            var i = t * ellMatrix.numRows + row
            var col = ellMatrix.colIdx[i]
            var val = ellMatrix.value[i]
            sum += x[Int(col)] * val
        y[row] = sum


def main() raises:
    var rows = 1024
    var cols = 1024
    var sparsity: Float32 = 0.9

    var h_row_idx = List[UInt32]()
    var h_col_idx = List[UInt32]()
    var h_values = List[Float32]()
    generate_sparse_matrix(rows, cols, sparsity, h_row_idx, h_col_idx, h_values)

    # Reference
    var h_x = List[Float32](capacity=cols)
    for _ in range(cols):
        h_x.append(1.0)

    var h_y_ref = List[Float32](capacity=rows)
    for _ in range(rows):
        h_y_ref.append(0.0)

    spmv_cpu(rows, cols, h_row_idx, h_col_idx, h_values, h_x, h_y_ref)

    # Convert to ELL
    var nnz_per_row = List[Int](capacity=rows)
    for _ in range(rows):
        nnz_per_row.append(0)

    for i in range(len(h_row_idx)):
        var r_idx = h_row_idx[i]
        nnz_per_row[Int(r_idx)] += 1

    var max_nnz = 0
    for i in range(len(nnz_per_row)):
        var c = nnz_per_row[i]
        if c > max_nnz:
            max_nnz = c

    print("ELL Matrix: ", rows, "x", cols, ", Max NNZ/Row: ", max_nnz)

    var h_ell_colIdx = List[UInt32](capacity=rows * max_nnz)
    var h_ell_value = List[Float32](capacity=rows * max_nnz)
    for _ in range(rows * max_nnz):
        h_ell_colIdx.append(0)
        h_ell_value.append(0.0)

    var current_nnz = List[Int](capacity=rows)
    for _ in range(rows):
        current_nnz.append(0)

    var numNonzeros = len(h_values)
    for i in range(numNonzeros):
        var r = Int(h_row_idx[i])
        var c = h_col_idx[i]
        var v = h_values[i]
        var t = current_nnz[r]

        # Index: t * rows + r (column-major storage)
        var ell_idx = t * rows + r
        h_ell_colIdx[ell_idx] = c
        h_ell_value[ell_idx] = v
        current_nnz[r] += 1

    # Prepare host pointers
    var h_colIdx_ptr = alloc[UInt32](rows * max_nnz)
    var h_value_ptr = alloc[Float32](rows * max_nnz)
    var h_x_ptr = alloc[Float32](cols)
    var h_y_ptr = alloc[Float32](rows)

    for i in range(rows * max_nnz):
        h_colIdx_ptr[i] = h_ell_colIdx[i]
        h_value_ptr[i] = h_ell_value[i]
    for i in range(cols):
        h_x_ptr[i] = h_x[i]
    for i in range(rows):
        h_y_ptr[i] = 0.0

    var ctx = DeviceContext()

    # Device allocation
    var d_colIdx_buf = ctx.enqueue_create_buffer[.uint32](rows * max_nnz)
    var d_value_buf = ctx.enqueue_create_buffer[.float32](rows * max_nnz)
    var d_x_buf = ctx.enqueue_create_buffer[.float32](cols)
    var d_y_buf = ctx.enqueue_create_buffer[.float32](rows)

    # Copy to device
    ctx.enqueue_copy(d_colIdx_buf, h_colIdx_ptr)
    ctx.enqueue_copy(d_value_buf, h_value_ptr)
    ctx.enqueue_copy(d_x_buf, h_x_ptr)

    # Initialize y on device
    var h_y_zeros = alloc[Float32](rows)
    for i in range(rows):
        h_y_zeros[i] = 0.0
    ctx.enqueue_copy(d_y_buf, h_y_zeros)
    h_y_zeros.free()

    var d_ellMatrix = ELLMatrix(
        rows,
        cols,
        max_nnz,
        d_colIdx_buf.unsafe_ptr().as_unsafe_any_origin(),
        d_value_buf.unsafe_ptr().as_unsafe_any_origin(),
    )

    var blockSize = 256
    var numBlocks = (rows + blockSize - 1) // blockSize

    ctx.enqueue_function[spmv_ell_kernel](
        d_ellMatrix,
        d_x_buf,
        d_y_buf,
        grid_dim=numBlocks,
        block_dim=blockSize,
    )

    ctx.enqueue_copy(h_y_ptr, d_y_buf)
    ctx.synchronize()

    if verify(h_y_ref, h_y_ptr, rows):
        print("ELL Kernel Test Passed!")
    else:
        print("ELL Kernel Test Failed!")

    h_colIdx_ptr.free()
    h_value_ptr.free()
    h_x_ptr.free()
    h_y_ptr.free()
