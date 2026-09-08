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

# DOC: max/develop/custom-ops-matmul.mdx

from std.math import ceildiv
from std.math.uutils import udivmod
from std.sys.info import has_accelerator, has_amd_gpu_accelerator, simd_width_of

import extensibility

from max.gpu.host import DeviceContext
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_dim,
    block_idx,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_all,
)
from layout import (
    TensorLayout,
    TileTensor,
    col_major,
    row_major,
    stack_allocation,
)
from layout.layout_tensor import Layout, LayoutTensor, copy_dram_to_sram_async
from layout.tensor_core import TensorCore
from layout.tile_io import GenericToSharedAsyncTileCopier

from extensibility import InputTensor, ManagedTensorSlice, OutputTensor

from std.utils import StaticTuple
from std.utils.index import Index

# The number of threads per block to use for the optimized kernels
# Used only in llvm_metadata for MAX_THREADS_PER_BLOCK_METADATA
# Not the most performant for all kernels, used sparingly on nvidia accelerators

comptime OPTIMIZED_NUM_THREADS = 256 if has_amd_gpu_accelerator() else 1024

# The block size to use for the optimized kernels
comptime OPTIMIZED_BLOCK_SIZE = 16 if has_amd_gpu_accelerator() else 32


# ===-----------------------------------------------------------------------=== #
# Outer-product accumulator for `TileTensor`
# ===-----------------------------------------------------------------------=== #


@always_inline
def outer_product_acc(
    res: TileTensor[mut=True, ...],
    lhs: TileTensor,
    rhs: TileTensor,
) where (
    type_of(res).flat_rank == 2
    and type_of(lhs).flat_rank == 1
    and type_of(rhs).flat_rank == 1
    and type_of(res).shape_known
    and type_of(lhs).shape_known
    and type_of(rhs).shape_known
):
    """Updates result tensor with the outer product of two vectors.

    Computes `res += outer(lhs, rhs)` where `lhs` and `rhs` are vectors and
    `res` is a matrix. This is a `TileTensor` mirror of
    `layout.math.outer_product_acc`, used so the block-tiled kernels do not
    need to bridge their register tiles back to `LayoutTensor`.

    Constraints:

        All tensors must have statically known shapes.
        `res` must be rank 2.
        `lhs` and `rhs` must be rank 1.
        `res.shape[0]` `==` `lhs.shape[0]` and `res.shape[1]` `==` `rhs.shape[0]`.

    Args:
        res: The result matrix to accumulate into, shape (M, N).
        lhs: The left-hand side vector, shape (M,).
        rhs: The right-hand side vector, shape (N,).
    """
    comptime dtype = res.dtype

    comptime M = type_of(res).LayoutType.static_shape[0]
    comptime N = type_of(res).LayoutType.static_shape[1]

    comptime assert (
        type_of(lhs).LayoutType.static_shape[0] == M
    ), "lhs shape mismatch"
    comptime assert (
        type_of(rhs).LayoutType.static_shape[0] == N
    ), "rhs shape mismatch"

    comptime for i in range(M):
        comptime for j in range(N):
            res[i, j] += rebind[res.ElementType](lhs[i].cast[dtype]()) * rebind[
                res.ElementType
            ](rhs[j].cast[dtype]())


# ===-----------------------------------------------------------------------=== #

# Naive matrix multiplication (CPU)
# ===-----------------------------------------------------------------------=== #


def naive_matrix_multiplication_cpu(
    output: ManagedTensorSlice,
    a: ManagedTensorSlice[dtype=output.dtype, rank=output.rank, ...],
    b: ManagedTensorSlice[dtype=output.dtype, rank=output.rank, ...],
):
    """A naive matrix multiplication used as a fallback on CPU hardware."""
    var M = a.shape()[0]
    var N = b.shape()[1]
    var K = b.shape()[0]

    for row in range(M):
        for col in range(N):
            for k in range(K):
                output[row, col] = output[row, col] + a[row, k] * b[k, col]


# ===-----------------------------------------------------------------------=== #

# Naive matrix multiplication (GPU)
# ===-----------------------------------------------------------------------=== #


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def naive_matrix_multiplication[
    dtype: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    BM: Int,
    BN: Int,
](
    a: TileTensor[dtype, a_layout, MutAnyOrigin],
    b: TileTensor[dtype, b_layout, MutAnyOrigin],
    c: TileTensor[mut=True, dtype, c_layout, MutAnyOrigin],
):
    """
    GEMM kernel that performs matrix multiplication C = A * B.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout: The layout of the input tensor A.
        b_layout: The layout of the input tensor B.
        c_layout: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.

    Args:
        a: The input tensor A.
        b: The input tensor B.
        c: The output tensor C.

    This kernel uses a simple for loop structure to compute the matrix
    multiplication. Each thread computes a single element of the output matrix
    C by accumulating the dot product of the corresponding row of A and column
    of B.

    The kernel assumes that the input matrices A and B are compatible for
    matrix multiplication, i.e., the number of columns in A equals the number
    of rows in B.
    """
    comptime assert a.flat_rank == 2, "a must be rank 2"
    comptime assert b.flat_rank == 2, "b must be rank 2"
    comptime assert c.flat_rank == 2, "c must be rank 2"

    var M = Int(a.dim[0]())
    var N = Int(b.dim[1]())
    var K = Int(b.dim[0]())

    # Calculate the column and row indices for each thread.
    var row = block_dim.x * block_idx.x + thread_idx.x
    var col = block_dim.y * block_idx.y + thread_idx.y

    # Initialize a register to accumulate the result for this thread.
    var dst_reg: c.ElementType = 0

    # Iterate over the K dimension to compute the dot product.
    if row < M and col < N:
        for k_index in range(K):
            # Multiply the elements and accumulate the result.
            dst_reg = dst_reg + a[row, k_index] * b[k_index, col]

    # Write the final accumulated result to the output matrix.
    c[row, col] = dst_reg


# ===-----------------------------------------------------------------------=== #

# Matrix multiplication with global memory coalescing
# ===-----------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(OPTIMIZED_NUM_THREADS)
    )
)
def coalescing_matrix_multiplication[
    dtype: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    BM: Int,
    BN: Int,
](
    a: TileTensor[dtype, a_layout, MutAnyOrigin],
    b: TileTensor[dtype, b_layout, MutAnyOrigin],
    c: TileTensor[mut=True, dtype, c_layout, MutAnyOrigin],
):
    """
    GEMM kernel that performs matrix multiplication C = A * B with
    memory coalescing optimizations.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout: The layout of the input tensor A.
        b_layout: The layout of the input tensor B.
        c_layout: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.

    Args:
        a: The input tensor A.
        b: The input tensor B.
        c: The output tensor C.

    This kernel optimizes memory access patterns by ensuring that
    threads within a warp access contiguous memory locations.

    Each thread computes a single element of the output matrix C by
    accumulating the partial results in a register. The final result
    is then stored back to the output matrix.
    """
    comptime assert a.flat_rank == 2, "a must be rank 2"
    comptime assert b.flat_rank == 2, "b must be rank 2"
    comptime assert c.flat_rank == 2, "c must be rank 2"

    var M = Int(a.dim[0]())
    var N = Int(b.dim[1]())
    var K = Int(b.dim[0]())

    # Calculate the column and row indices for each thread.
    # Have adjacent threads work on the same row to allow for memory coalescing
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x

    # Initialize a register to accumulate the result for this thread.
    var dst_reg: c.ElementType = 0

    # Iterate over the K dimension to compute the dot product.
    if row < M and col < N:
        for k_index in range(K):
            # Multiply the elements and accumulate the result.
            dst_reg = dst_reg + a[row, k_index] * b[k_index, col]

    # Write the final accumulated result to the output matrix.
    c[row, col] = dst_reg


# ===-----------------------------------------------------------------------=== #

# Matrix multiplication with shared memory tiling
# ===-----------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(OPTIMIZED_NUM_THREADS)
    )
)
def tiled_matrix_multiplication[
    dtype: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
    NUM_THREADS: Int,
](
    a: TileTensor[dtype, a_layout, MutAnyOrigin],
    b: TileTensor[dtype, b_layout, MutAnyOrigin],
    c: TileTensor[mut=True, dtype, c_layout, MutAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B using
    shared memory to improve performance.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout: The layout of the input tensor A.
        b_layout: The layout of the input tensor B.
        c_layout: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        NUM_THREADS: The total number of threads per block.

    Args:
        a: The input tensor A.
        b: The input tensor B.
        c: The output tensor C.

    This kernel uses a tiling strategy to compute the matrix multiplication.
    Each thread block computes a BM x BN tile of the output matrix C. The
    input matrices A and B are loaded into shared memory in tiles of size
    BM x BK and BK x BN, respectively.

    The kernel assumes that the input matrices A and B are compatible for
    matrix multiplication, i.e., the number of columns in A equals the
    number of rows in B.
    """
    comptime assert a.flat_rank == 2, "a must be rank 2"
    comptime assert b.flat_rank == 2, "b must be rank 2"
    comptime assert c.flat_rank == 2, "c must be rank 2"

    # Calculate the column and row indices for each thread
    var row, col = udivmod(thread_idx.x, BN)

    # Get the tile of the output matrix C that this thread block is responsible for
    var dst = c.tile[BM, BN](block_idx.y, block_idx.x)

    # Allocate shared memory for tiles of input matrices A and B
    var a_smem = stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[BM, BK]()
    )
    var b_smem = stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[BK, BN]()
    )

    # Initialize the register to accumulate the result
    var dst_reg: c.ElementType = 0

    # Define the layout for loading tiles of A and B into shared memory
    comptime load_a_layout = row_major[NUM_THREADS // BK, BK]()
    comptime load_b_layout = row_major[BK, NUM_THREADS // BK]()

    # Iterate over tiles of input matrices A and B
    for block in range(Int(b.dim[0]()) // BK):
        # Get the tiles of A and B for the current iteration
        var a_tile = a.tile[BM, BK](block_idx.y, block)
        var b_tile = b.tile[BK, BN](block, block_idx.x)

        # Asynchronously copy tiles of A and B from global memory to shared
        # memory.
        GenericToSharedAsyncTileCopier[load_a_layout]().copy(a_smem, a_tile)
        GenericToSharedAsyncTileCopier[load_b_layout]().copy(b_smem, b_tile)

        # Commit the issued async copies as a group and wait for them to
        # complete before any thread reads from shared memory.
        async_copy_commit_group()
        async_copy_wait_all()
        barrier()

        # Perform matrix multiplication on the tiles in shared memory
        comptime for k in range(BK):
            dst_reg += a_smem[row, k] * b_smem[k, col]

        # Synchronize threads before loading the next tiles
        barrier()

    # Write the result to the output matrix
    dst[row, col] += dst_reg


# ===-----------------------------------------------------------------------=== #

# Matrix multiplication with shared memory tiling and register tiling
# ===-----------------------------------------------------------------------=== #


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def tiled_register_matrix_multiplication[
    dtype: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    NUM_THREADS: Int,
](
    a: TileTensor[dtype, a_layout, MutAnyOrigin],
    b: TileTensor[dtype, b_layout, MutAnyOrigin],
    c: TileTensor[mut=True, dtype, c_layout, MutAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B using
    shared memory and register tiling.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout: The layout of the input tensor A.
        b_layout: The layout of the input tensor B.
        c_layout: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        TM: The tile size in the M dimension.
        NUM_THREADS: The number of threads per block.

    Args:
        a: The input tensor A.
        b: The input tensor B.
        c: The output tensor C.

    This kernel uses a tiled approach to compute the matrix multiplication. It
    loads tiles of matrices A and B into shared memory, and then each thread
    computes a partial result using the tiles in shared memory. The partial
    results are accumulated in registers and finally stored back to the output
    matrix C.

    The kernel assumes that the input matrices A and B are compatible for
    matrix multiplication, i.e., the number of columns in A equals the number
    of rows in B.
    """
    comptime assert a.flat_rank == 2, "a must be rank 2"
    comptime assert b.flat_rank == 2, "b must be rank 2"
    comptime assert c.flat_rank == 2, "c must be rank 2"

    # Calculate the column and row indices for each thread.
    var row, col = udivmod(thread_idx.x, BN)

    # Get the tile of the output matrix C that this thread is
    # responsible for computing.
    var dst = c.tile[BM, BN](block_idx.y, block_idx.x).tile[TM, 1](row, col)

    # Allocate shared memory for tiles of A and B.
    var a_smem = stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[BM, BK]()
    )
    var b_smem = stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[BK, BN]()
    )

    # Allocate a register tile to store the partial results.
    var dst_reg = stack_allocation[dtype=dtype, address_space=.LOCAL](
        row_major[TM]()
    )
    dst_reg.copy_from(dst)

    # Define the layout for loading tiles of A and B into shared
    # memory.
    comptime load_a_layout = row_major[NUM_THREADS // BK, BK]()
    comptime load_b_layout = row_major[BK, NUM_THREADS // BK]()

    # Iterate over the tiles of A and B in the K dimension.
    for block in range(Int(b.dim[0]()) // BK):
        # Get the tiles of A and B for the current block.
        var a_tile = a.tile[BM, BK](block_idx.y, block)
        var b_tile = b.tile[BK, BN](block, block_idx.x)

        # Asynchronously load the tiles of A and B into shared memory.
        GenericToSharedAsyncTileCopier[load_a_layout]().copy(a_smem, a_tile)
        GenericToSharedAsyncTileCopier[load_b_layout]().copy(b_smem, b_tile)

        # Commit and wait for the async copies before reading shared memory.
        async_copy_commit_group()
        async_copy_wait_all()
        barrier()

        # Iterate over the elements in the K dimension within the tiles.
        comptime for k in range(BK):
            # Get the corresponding tiles from shared memory.
            var a_tile = a_smem.tile[TM, 1](row, k)
            var b_tile = b_smem.tile[1, BN](k, 0)
            var b_val = b_tile[0, col]

            # Multiply the elements and accumulate the partial results.
            comptime for t in range(TM):
                dst_reg[t] += a_tile[t, 0] * b_val

        # Synchronize all threads before loading the next tiles.
        barrier()

    # Write the final accumulated results to the output matrix.
    dst.copy_from(dst_reg)


# ===-----------------------------------------------------------------------=== #

# Matrix multiplication with block tiling
# ===-----------------------------------------------------------------------=== #


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def block_tiled_matrix_multiplication[
    dtype: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    NUM_THREADS: Int,
](
    a: TileTensor[dtype, a_layout, MutAnyOrigin],
    b: TileTensor[dtype, b_layout, MutAnyOrigin],
    c: TileTensor[mut=True, dtype, c_layout, MutAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout: The layout of the input tensor A.
        b_layout: The layout of the input tensor B.
        c_layout: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        TM: The tile size in the M dimension.
        TN: The tile size in the N dimension.
        NUM_THREADS: The total number of threads per block.

    Args:
        a: The input tensor A.
        b: The input tensor B.
        c: The output tensor C.

    This kernel uses a 2D block tiling strategy to compute the matrix
    multiplication. Each thread block computes a BM x BN tile of the output
    matrix C. Within each thread block, threads are further divided into
    TM x TN tiles to enable thread-level parallelism.

    The kernel loads tiles of A and B into shared memory to reduce global
    memory accesses. It then performs the matrix multiplication using
    register-level tiling and accumulates the results in registers.

    The kernel assumes that the input matrices A and B are compatible for
    matrix multiplication, i.e., the number of columns in A equals the number
    of rows in B.
    """
    comptime assert a.flat_rank == 2, "a must be rank 2"
    comptime assert b.flat_rank == 2, "b must be rank 2"
    comptime assert c.flat_rank == 2, "c must be rank 2"

    var partition_row, partition_col = udivmod(thread_idx.x, BN // TN)

    var dst = c.tile[BM, BN](block_idx.y, block_idx.x).tile[TM, TN](
        partition_row, partition_col
    )

    var a_smem = stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[BM, BK]()
    )
    var b_smem = stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[BK, BN]()
    )

    var dst_reg = stack_allocation[dtype=dtype, address_space=.LOCAL](
        row_major[TM, TN]()
    )
    dst_reg.copy_from(dst)

    var a_reg = stack_allocation[dtype=dtype, address_space=.LOCAL](
        row_major[TM]()
    )
    var b_reg = stack_allocation[dtype=dtype, address_space=.LOCAL](
        row_major[TN]()
    )

    comptime load_a_layout = row_major[NUM_THREADS // BK, BK]()
    comptime load_b_layout = row_major[BK, NUM_THREADS // BK]()

    var ntiles = Int(b.dim[0]()) // BK

    for block in range(ntiles):
        var a_tile = a.tile[BM, BK](block_idx.y, block)
        var b_tile = b.tile[BK, BN](block, block_idx.x)
        GenericToSharedAsyncTileCopier[load_a_layout]().copy(a_smem, a_tile)
        GenericToSharedAsyncTileCopier[load_b_layout]().copy(b_smem, b_tile)

        async_copy_commit_group()
        async_copy_wait_all()
        barrier()

        comptime for k in range(BK):
            var a_tile = a_smem.tile[TM, 1](partition_row, k)
            var b_tile = b_smem.tile[1, TN](k, partition_col)
            a_reg.copy_from(a_tile)
            b_reg.copy_from(b_tile)
            outer_product_acc(dst_reg, a_reg, b_reg)
        barrier()

    dst.copy_from(dst_reg)


# ===-----------------------------------------------------------------------=== #

# Matrix multiplication with vectorized memory access
# ===-----------------------------------------------------------------------=== #


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def block_tiled_vectorized_matrix_multiplication[
    dtype: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    NUM_THREADS: Int,
](
    a: TileTensor[dtype, a_layout, MutAnyOrigin],
    b: TileTensor[dtype, b_layout, MutAnyOrigin],
    c: TileTensor[mut=True, dtype, c_layout, MutAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B with
    vectorized memory access.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout: The layout of the input tensor A.
        b_layout: The layout of the input tensor B.
        c_layout: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        TM: The tile size in the M dimension.
        TN: The tile size in the N dimension.
        NUM_THREADS: The total number of threads per block.

    Args:
        a: The input tensor A.
        b: The input tensor B.
        c: The output tensor C.

    This kernel uses a 2D block tiling strategy to compute the matrix
    multiplication. Each thread block computes a BM x BN tile of the output
    matrix C. Within each thread block, threads are further divided into TM x
    TN tiles to enable thread-level parallelism.

    The kernel loads tiles of A and B into shared memory using vectorized
    memory access to improve memory bandwidth utilization. It then performs the
    matrix multiplication using register-level tiling and accumulates the
    results in registers.

    The kernel assumes that the input matrices A and B are compatible for
    matrix multiplication, i.e., the number of columns in A equals the number
    of rows in B.
    """
    comptime assert a.flat_rank == 2, "a must be rank 2"
    comptime assert b.flat_rank == 2, "b must be rank 2"
    comptime assert c.flat_rank == 2, "c must be rank 2"

    comptime simd_width = simd_width_of[dtype]()
    var partition_row, partition_col = udivmod(thread_idx.x, BN // TN)

    # Get the tile of the output matrix C that this thread is responsible
    # for computing.
    var dst = c.tile[BM, BN](block_idx.y, block_idx.x).tile[TM, TN](
        partition_row, partition_col
    )

    # Allocate shared memory for tiles of A and B.
    # Use column-major layout for A to get the transpose.
    var a_smem = stack_allocation[dtype=dtype, address_space=.SHARED](
        col_major[BM, BK]()
    )
    var b_smem = stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[BK, BN]()
    )

    # Allocate register tiles to store the partial results and operands.
    var dst_reg = stack_allocation[dtype=dtype, address_space=.LOCAL](
        row_major[TM, TN]()
    )
    dst_reg.copy_from(dst)

    var a_reg = stack_allocation[dtype=dtype, address_space=.LOCAL](
        row_major[TM]()
    )
    var b_reg = stack_allocation[dtype=dtype, address_space=.LOCAL](
        row_major[TN]()
    )

    comptime load_b_layout = row_major[BK, NUM_THREADS // BK]()

    # Each thread loads one `simd_width` chunk along K from a single row of A.
    var inner_row_a, inner_col_a = udivmod(thread_idx.x, BK // simd_width)

    var ntiles = Int(b.dim[0]()) // BK

    # Iterate over the tiles of A and B in the K dimension.
    for block in range(ntiles):
        var a_tile = a.tile[BM, BK](block_idx.y, block)
        var b_tile = b.tile[BK, BN](block, block_idx.x)

        # Vectorized K-load, scalar scatter into column-major `a_smem` —
        # transposes A on the way in.
        var a_load = a_tile.vectorize[1, simd_width]()[inner_row_a, inner_col_a]
        comptime for v in range(simd_width):
            a_smem[inner_row_a, inner_col_a * simd_width + v] = rebind[
                a_smem.ElementType
            ](a_load[v])

        # Asynchronously load the tile of B into shared memory using
        # vectorized memory access.
        GenericToSharedAsyncTileCopier[load_b_layout]().copy(
            b_smem.vectorize[1, simd_width](),
            b_tile.vectorize[1, simd_width](),
        )

        async_copy_commit_group()
        async_copy_wait_all()
        barrier()

        # Iterate over the elements in the K dimension within the tiles.
        comptime for k in range(BK):
            # Load the corresponding tiles from shared memory into registers.
            var a_tile = a_smem.tile[TM, 1](partition_row, k)
            var b_tile = b_smem.tile[1, TN](k, partition_col)
            a_reg.copy_from(a_tile)
            b_reg.copy_from(b_tile)

            # Perform outer product and accumulate the partial results.
            outer_product_acc(dst_reg, a_reg, b_reg)

        barrier()

    # Write the final accumulated results to the output matrix.
    dst.copy_from(dst_reg)


# ===-----------------------------------------------------------------------=== #

# Matrix multiplication using Tensor Cores
# ===-----------------------------------------------------------------------=== #


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def tensor_core_matrix_multiplication[
    dtype: DType,
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
](
    A: LayoutTensor[dtype, layout_a, MutAnyOrigin],
    B: LayoutTensor[dtype, layout_b, MutAnyOrigin],
    C: LayoutTensor[dtype, layout_c, MutAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B using
    tensor cores.

    Parameters:
        dtype: The data type of the input and output tensors.
        layout_a: The layout of the input tensor A.
        layout_b: The layout of the input tensor B.
        layout_c: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        WM: The warp tile size in the M dimension.
        WN: The warp tile size in the N dimension.
        MMA_M: Tensor core instruction shape in M dimension.
        MMA_N: Tensor core instruction shape in N dimension.
        MMA_K: Tensor core instruction shape in K dimension.

    Args:
        A: The input tensor A.
        B: The input tensor B.
        C: The output tensor C.

    This kernel uses a tiled approach with tensor cores to compute the matrix
    multiplication. It loads tiles of matrices A and B into shared memory, and
    then each warp computes a partial result using tensor cores. The partial
    results are accumulated in registers and finally stored back to the output
    matrix C.

    The kernel assumes that the input matrices A and B are compatible for
    matrix multiplication, i.e., the number of columns in A equals the number
    of rows in B.
    """
    comptime M = C.shape[0]()  # Number of rows in matrix C
    comptime N = C.shape[1]()  # Number of columns in matrix C
    comptime K = A.shape[1]()  # Number of columns in matrix A

    # Calculate warp tile coordinates within the block
    var warp_y, warp_x = udivmod(warp_id(), BN // WN)

    # Get the warp tile of the output matrix C
    var C_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x).tile[WM, WN](
        warp_y, warp_x
    )

    # Ensure warp tile dimensions are multiples of instruction shape
    comptime assert (
        WM % MMA_M == 0 and WN % MMA_N == 0 and K % MMA_K == 0
    ), "Warp tile should be an integer multiple of instruction shape"

    # Create tensor core operation object
    var mma_op = TensorCore[A.dtype, C.dtype, Index(MMA_M, MMA_N, MMA_K)]()

    # Allocate shared memory for tiles of A and B
    var A_sram_tile = LayoutTensor[
        A.dtype,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()
    var B_sram_tile = LayoutTensor[
        B.dtype,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()

    # Allocate register tile for accumulating partial results
    var c_reg = (
        LayoutTensor[
            C.dtype,
            Layout.row_major(WM // MMA_M, (WN * 4) // MMA_N),
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    # Iterate over tiles of A and B in the K dimension
    for k_i in range(K // BK):
        barrier()  # Synchronize before loading new tiles

        # Get the tiles of A and B for the current iteration
        var A_dram_tile = A.tile[BM, BK](block_idx.y, k_i)
        var B_dram_tile = B.tile[BK, BN](k_i, block_idx.x)

        # Load tiles of A and B into shared memory asynchronously
        copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](
            A_sram_tile.vectorize[1, 4](), A_dram_tile.vectorize[1, 4]()
        )
        copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](
            B_sram_tile.vectorize[1, 4](), B_dram_tile.vectorize[1, 4]()
        )

        async_copy_wait_all()  # Wait for async copies to complete
        barrier()  # Synchronize after loading tiles

        # Get the warp tiles of A and B from shared memory
        var A_warp_tile = A_sram_tile.tile[WM, BK](warp_y, 0)
        var B_warp_tile = B_sram_tile.tile[BK, WN](0, warp_x)

        # Iterate over the elements in the K dimension within the tiles
        comptime for mma_k in range(BK // MMA_K):
            comptime for mma_m in range(WM // MMA_M):
                comptime for mma_n in range(WN // MMA_N):
                    # Get the register tile for the current MMA operation
                    var c_reg_m_n = c_reg.tile[1, 4](mma_m, mma_n)

                    # Get the MMA tiles of A and B
                    var A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](
                        mma_m, mma_k
                    )
                    var B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](
                        mma_k, mma_n
                    )

                    # Load fragments of A and B into registers
                    var a_reg = mma_op.load_a(A_mma_tile)
                    var b_reg = mma_op.load_b(B_mma_tile)

                    # Perform MMA operation and accumulate the result
                    var d_reg_m_n = mma_op.mma_op(
                        a_reg,
                        b_reg,
                        c_reg_m_n,
                    )

                    # Store the accumulated result back to the register tile
                    c_reg_m_n.copy_from(d_reg_m_n)

    # Write the final accumulated results to the output matrix
    comptime for mma_m in range(WM // MMA_M):
        comptime for mma_n in range(WN // MMA_N):
            var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            var c_reg_m_n = c_reg.tile[1, 4](mma_m, mma_n)
            mma_op.store_d(C_mma_tile, c_reg_m_n)


# ===-----------------------------------------------------------------------=== #

# The matrix multiplication graph operation
# ===-----------------------------------------------------------------------=== #


@extensibility.register("matrix_multiplication")
struct MatrixMultiplication[algorithm: StaticString]:
    """
    The central custom operation that dispatches to multiple different
    matrix multiplication implementations, depending on target hardware and
    selected algorithm.
    """

    @staticmethod
    def execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
    ](
        output: OutputTensor[rank=2, ...],
        a: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        b: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        # the context is needed for some GPU calls
        ctx: DeviceContext,
    ) raises:
        # At graph compilation time, we will know what device we are compiling
        # this operation for, so we can specialize it for the target hardware.
        comptime if target == "gpu":
            var a_tt = a.to_tile_tensor().as_unsafe_any_origin()
            var b_tt = b.to_tile_tensor().as_unsafe_any_origin()
            var out_tt = output.to_tile_tensor().as_unsafe_any_origin()

            var M = Int(a_tt.dim[0]())
            var N = Int(b_tt.dim[1]())

            var gpu_ctx = ctx

            # Zero out the memory in the outbound tensor.
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output.dtype](
                    gpu_ctx,
                    out_tt.ptr,
                    M * N,
                    owning=False,
                ),
                0,
            )

            # We support several compile-time variants for the matrix
            # multiplication calculation:
            # - "naive": A naive matrix multiplication using TileTensors.
            # - "coalescing": Matrix multiplication with memory coalescing
            #   optimizations.
            # - "tiled": Matrix multiplication using a tiling strategy.
            # - "tiled_register": Matrix multiplication using shared memory
            #   and register tiling .
            # - "block_tiled": Matrix multiplication using a 2D block tiling
            #   strategy.
            # - "block_tiled_vectorized": Matrix multiplication using a
            #   further-optimized 2D block tiling strategy.
            # - "tensor_core": Matrix multiplication using Tensor Cores.
            # In each case, the specific matrix multiplication function is
            # compiled and enqueued to run on the GPU.
            comptime if Self.algorithm == "naive":
                comptime BM = 16
                comptime BN = 16
                comptime matmul_kernel = naive_matrix_multiplication[
                    output.dtype,
                    type_of(a_tt).LayoutType,
                    type_of(b_tt).LayoutType,
                    type_of(out_tt).LayoutType,
                    BM,
                    BN,
                ]
                gpu_ctx.enqueue_function[matmul_kernel](
                    a_tt,
                    b_tt,
                    out_tt,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(BN, BM),
                )
            elif Self.algorithm == "coalescing":
                comptime BM = OPTIMIZED_BLOCK_SIZE
                comptime BN = OPTIMIZED_BLOCK_SIZE
                comptime coalescing_matmul_kernel = coalescing_matrix_multiplication[
                    output.dtype,
                    type_of(a_tt).LayoutType,
                    type_of(b_tt).LayoutType,
                    type_of(out_tt).LayoutType,
                    BM,
                    BN,
                ]
                gpu_ctx.enqueue_function[coalescing_matmul_kernel](
                    a_tt,
                    b_tt,
                    out_tt,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(BN, BM),
                )
            elif Self.algorithm == "tiled":
                comptime BM = OPTIMIZED_BLOCK_SIZE
                comptime BN = OPTIMIZED_BLOCK_SIZE
                comptime BK = OPTIMIZED_BLOCK_SIZE
                comptime NUM_THREADS = BM * BN
                comptime tiled_matmul_kernel = tiled_matrix_multiplication[
                    output.dtype,
                    type_of(a_tt).LayoutType,
                    type_of(b_tt).LayoutType,
                    type_of(out_tt).LayoutType,
                    BM,
                    BN,
                    BK,
                    NUM_THREADS,
                ]
                gpu_ctx.enqueue_function[tiled_matmul_kernel](
                    a_tt,
                    b_tt,
                    out_tt,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(BM * BN),
                )
            elif Self.algorithm == "tiled_register":
                comptime BM = 64
                comptime BN = 64
                comptime BK = 8
                comptime TM = 16
                comptime NUM_THREADS = (BM * BN) // TM
                comptime tiled_register_matmul_kernel = tiled_register_matrix_multiplication[
                    output.dtype,
                    type_of(a_tt).LayoutType,
                    type_of(b_tt).LayoutType,
                    type_of(out_tt).LayoutType,
                    BM,
                    BN,
                    BK,
                    TM,
                    NUM_THREADS,
                ]
                gpu_ctx.enqueue_function[tiled_register_matmul_kernel](
                    a_tt,
                    b_tt,
                    out_tt,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(NUM_THREADS),
                )
            elif Self.algorithm == "block_tiled":
                comptime BM = 128
                comptime BN = 128
                comptime BK = 8
                comptime TM = 8
                comptime TN = 8
                comptime NUM_THREADS = (BM * BN) // (TM * TN)
                comptime block_tiled_matmul_kernel = block_tiled_matrix_multiplication[
                    output.dtype,
                    type_of(a_tt).LayoutType,
                    type_of(b_tt).LayoutType,
                    type_of(out_tt).LayoutType,
                    BM,
                    BN,
                    BK,
                    TM,
                    TN,
                    NUM_THREADS,
                ]
                gpu_ctx.enqueue_function[block_tiled_matmul_kernel](
                    a_tt,
                    b_tt,
                    out_tt,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(NUM_THREADS),
                )
            elif Self.algorithm == "block_tiled_vectorized":
                comptime BM = 128
                comptime BN = 128
                comptime BK = 8
                comptime TM = 8
                comptime TN = 8
                comptime NUM_THREADS = (BM * BN) // (TM * TN)
                comptime block_tiled_vectorized_matmul_kernel = block_tiled_vectorized_matrix_multiplication[
                    output.dtype,
                    type_of(a_tt).LayoutType,
                    type_of(b_tt).LayoutType,
                    type_of(out_tt).LayoutType,
                    BM,
                    BN,
                    BK,
                    TM,
                    TN,
                    NUM_THREADS,
                ]
                gpu_ctx.enqueue_function[block_tiled_vectorized_matmul_kernel](
                    a_tt,
                    b_tt,
                    out_tt,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(NUM_THREADS),
                )
            elif Self.algorithm == "tensor_core":
                comptime if has_accelerator():
                    var a_layout = a_tt.to_layout_tensor()
                    var b_layout = b_tt.to_layout_tensor()
                    var out_layout = out_tt.to_layout_tensor()

                    comptime BM = 64
                    comptime BN = 64
                    comptime BK = OPTIMIZED_BLOCK_SIZE
                    comptime WM = 32
                    comptime WN = WARP_SIZE
                    # different MMA shapes for AMD and NVIDIA, see:
                    # https://max.modular.com/api/mojo/layout/tensor_core/TensorCore/
                    comptime MMA_M = 16
                    comptime MMA_N = 16 if has_amd_gpu_accelerator() else 8
                    comptime MMA_K = 4
                    comptime NUM_WARPS = (BM // WM) * (BN // WN)
                    comptime tensor_core_matmul_kernel = tensor_core_matrix_multiplication[
                        output.dtype,
                        a_layout.layout,
                        b_layout.layout,
                        out_layout.layout,
                        BM,
                        BN,
                        BK,
                        WM,
                        WN,
                        MMA_M,
                        MMA_N,
                        MMA_K,
                    ]
                    gpu_ctx.enqueue_function[tensor_core_matmul_kernel](
                        a_layout,
                        b_layout,
                        out_layout,
                        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                        block_dim=(NUM_WARPS * WARP_SIZE),
                    )
                else:
                    raise Error("Tensor Cores are not available on this device")
            else:
                raise Error("No known matmul algorithm:", Self.algorithm)

        else:
            naive_matrix_multiplication_cpu(output, a, b)
