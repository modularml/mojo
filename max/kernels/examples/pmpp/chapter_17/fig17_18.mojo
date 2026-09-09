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
from spmv_utils import CSCMatrix, generate_sparse_matrix, spmv_cpu, verify
from max.gpu import block_idx, thread_idx, block_dim
from max.gpu.host import DeviceContext
from std.atomic import Atomic


@fieldwise_init
struct Element(Comparable, ImplicitlyCopyable):
    var r: UInt32
    var c: UInt32
    var v: Float32

    def __lt__(self, other: Self) -> Bool:
        if self.c != other.c:
            return self.c < other.c
        return self.r < other.r

    def __eq__(self, other: Self) -> Bool:
        return self.c == other.c and self.r == other.r


def spmv_csc_kernel(
    cscMatrix: CSCMatrix,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
):
    var col = block_idx.x * block_dim.x + thread_idx.x
    if col < cscMatrix.numCols:
        var inValue = x[col]
        var start = cscMatrix.colPtrs[col]
        var end = cscMatrix.colPtrs[col + 1]
        for i in range(Int(start), Int(end)):
            var row = cscMatrix.rowIdxs[i]
            var val = cscMatrix.values[i]
            _ = Atomic.fetch_add(y + Int(row), inValue * val)


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

    # Convert to CSC
    # Sort elements by column
    var numNonzeros = len(h_values)
    var elements = List[Element](capacity=numNonzeros)
    for i in range(numNonzeros):
        elements.append(Element(h_row_idx[i], h_col_idx[i], h_values[i]))

    sort(elements)

    var h_csc_colPtrs = List[UInt32](capacity=cols + 1)
    for _ in range(cols + 1):
        h_csc_colPtrs.append(0)

    var h_csc_rowIdxs = List[UInt32](capacity=numNonzeros)
    var h_csc_values = List[Float32](capacity=numNonzeros)

    for i in range(numNonzeros):
        var e = elements[i]
        h_csc_rowIdxs.append(e.r)
        h_csc_values.append(e.v)
        h_csc_colPtrs[Int(e.c) + 1] += 1

    for i in range(cols):
        h_csc_colPtrs[i + 1] += h_csc_colPtrs[i]

    print("CSC Matrix: ", rows, "x", cols, ", ", numNonzeros, " non-zeros.")

    # Prepare host pointers
    var h_colPtrs_ptr = alloc[UInt32](cols + 1)
    var h_rowIdxs_ptr = alloc[UInt32](numNonzeros)
    var h_values_ptr = alloc[Float32](numNonzeros)
    var h_x_ptr = alloc[Float32](cols)
    var h_y_ptr = alloc[Float32](rows)

    for i in range(cols + 1):
        h_colPtrs_ptr[i] = h_csc_colPtrs[i]
    for i in range(numNonzeros):
        h_rowIdxs_ptr[i] = h_csc_rowIdxs[i]
        h_values_ptr[i] = h_csc_values[i]
    for i in range(cols):
        h_x_ptr[i] = h_x[i]
    for i in range(rows):
        h_y_ptr[i] = 0.0

    var ctx = DeviceContext()

    # Device allocation
    var d_colPtrs_buf = ctx.enqueue_create_buffer[.uint32](cols + 1)
    var d_rowIdxs_buf = ctx.enqueue_create_buffer[.uint32](numNonzeros)
    var d_values_buf = ctx.enqueue_create_buffer[.float32](numNonzeros)
    var d_x_buf = ctx.enqueue_create_buffer[.float32](cols)
    var d_y_buf = ctx.enqueue_create_buffer[.float32](rows)

    # Copy to device
    ctx.enqueue_copy(d_colPtrs_buf, h_colPtrs_ptr)
    ctx.enqueue_copy(d_rowIdxs_buf, h_rowIdxs_ptr)
    ctx.enqueue_copy(d_values_buf, h_values_ptr)
    ctx.enqueue_copy(d_x_buf, h_x_ptr)

    # Initialize y on device to 0
    var h_y_zeros = alloc[Float32](rows)
    for i in range(rows):
        h_y_zeros[i] = 0.0
    ctx.enqueue_copy(d_y_buf, h_y_zeros)
    h_y_zeros.free()

    var d_cscMatrix = CSCMatrix(
        rows,
        cols,
        numNonzeros,
        d_colPtrs_buf.unsafe_ptr().as_unsafe_any_origin(),
        d_rowIdxs_buf.unsafe_ptr().as_unsafe_any_origin(),
        d_values_buf.unsafe_ptr().as_unsafe_any_origin(),
    )

    var blockSize = 256
    var numBlocks = (cols + blockSize - 1) // blockSize

    ctx.enqueue_function[spmv_csc_kernel](
        d_cscMatrix,
        d_x_buf,
        d_y_buf,
        grid_dim=numBlocks,
        block_dim=blockSize,
    )

    # Copy result back
    ctx.enqueue_copy(h_y_ptr, d_y_buf)
    ctx.synchronize()

    if verify(h_y_ref, h_y_ptr, rows):
        print("CSC Kernel Test Passed!")
    else:
        print("CSC Kernel Test Failed!")

    h_colPtrs_ptr.free()
    h_rowIdxs_ptr.free()
    h_values_ptr.free()
    h_x_ptr.free()
    h_y_ptr.free()
