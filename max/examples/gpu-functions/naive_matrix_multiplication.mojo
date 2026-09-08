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

comptime I = 5
comptime J = 4
comptime K = 6

comptime m_layout = row_major[I, J]()
comptime n_layout = row_major[J, K]()
comptime p_layout = row_major[I, K]()


def main() raises:
    comptime assert (
        has_accelerator()
    ), "This example requires a supported accelerator"

    var ctx = DeviceContext()
    var m_buffer = ctx.enqueue_create_buffer[float_dtype](
        comptime (m_layout.size())
    )
    var n_buffer = ctx.enqueue_create_buffer[float_dtype](
        comptime (n_layout.size())
    )
    var p_buffer = ctx.enqueue_create_buffer[float_dtype](
        comptime (p_layout.size())
    )

    # Map input buffers to host to fill with values from CPU
    with m_buffer.map_to_host() as host_buffer:
        var m_tensor = TileTensor(host_buffer, m_layout)
        for m_row in range(I):
            for m_col in range(J):
                m_tensor[m_row, m_col] = Float32(m_row - m_col)
        print("M matrix:", m_tensor)

    with n_buffer.map_to_host() as host_buffer:
        var n_tensor = TileTensor(host_buffer, n_layout)
        for n_row in range(J):
            for n_col in range(K):
                n_tensor[n_row, n_col] = Float32(n_row + n_col)
        print("N matrix:", n_tensor)

    # Wrap device buffers in `TileTensor`
    var m_tensor = TileTensor(m_buffer, m_layout)
    var n_tensor = TileTensor(n_buffer, n_layout)
    var p_tensor = TileTensor(p_buffer, p_layout)

    # The grid is divided up into blocks, making sure there's an extra
    # full block for any remainder. This hasn't been tuned for any specific
    # GPU.
    comptime BLOCK_SIZE = 16
    comptime num_col_blocks = ceildiv(I, BLOCK_SIZE)
    comptime num_row_blocks = ceildiv(J, BLOCK_SIZE)

    # Launch the compiled function on the GPU. The target device is specified
    # first, followed by all function arguments. The last two named parameters
    # are the dimensions of the grid in blocks, and the block dimensions.
    ctx.enqueue_function[naive_matrix_multiplication](
        m_tensor,
        n_tensor,
        p_tensor,
        grid_dim=(num_col_blocks, num_row_blocks),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )

    # Move the output tensor back onto the CPU so that we can read the results.
    with p_buffer.map_to_host() as host_buffer:
        var host_tensor = TileTensor(host_buffer, p_layout)
        print("Resulting matrix:", host_tensor)


def naive_matrix_multiplication(
    m: TileTensor[float_dtype, type_of(m_layout), MutAnyOrigin],
    n: TileTensor[float_dtype, type_of(n_layout), MutAnyOrigin],
    p: TileTensor[float_dtype, type_of(p_layout), MutAnyOrigin],
):
    """Naive matrix multiplication of M_ij x N_jk = P_ik."""
    var row = global_idx.y
    var col = global_idx.x

    var m_dim = Int(p.dim[0]())
    var n_dim = Int(p.dim[1]())
    var k_dim = Int(m.dim[1]())

    if row < m_dim and col < n_dim:
        for j_index in range(k_dim):
            p[row, col] = p[row, col] + m[row, j_index] * n[j_index, col]
