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

from std.math import ceildiv, isclose
from std.sys import argv

from max.gpu.host import DeviceContext
from max.gpu import block_idx, global_idx, warp_id
from max.gpu.memory import async_copy_wait_all
from max.gpu.sync import barrier
from std.memory import alloc
from std.testing import assert_almost_equal
from std.utils.numerics import get_accum_type

from layout import (
    Idx,
    Layout,
    TensorLayout,
    TileTensor,
    row_major,
    stack_allocation,
)
from layout.layout_tensor import (
    copy_sram_to_local,
    copy_dram_to_sram_async,
    copy_local_to_dram,
)


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def gemm_kernel[
    c_dtype: DType,
    CLayoutType: TensorLayout,
    a_dtype: DType,
    ALayoutType: TensorLayout,
    b_dtype: DType,
    BLayoutType: TensorLayout,
    NUM_THREADS: Int,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    TM: Int,
    TN: Int,
](
    mat_c: TileTensor[c_dtype, CLayoutType, MutUntrackedOrigin],
    mat_a: TileTensor[a_dtype, ALayoutType, ImmUntrackedOrigin],
    mat_b: TileTensor[b_dtype, BLayoutType, ImmUntrackedOrigin],
) where (
    BM > -1
    and BN > -1
    and BK > -1
    and WM > -1
    and WN > -1
    and TM > -1
    and TN > -1
    and mat_a.rank == 2
    and mat_b.rank == 2
    and mat_c.rank == 2
    and mat_a.all_dims_known
    and mat_b.all_dims_known
    and mat_c.all_dims_known
):
    var K = mat_a.dim[1]()

    var a_tile_sram = stack_allocation[
        mat_a.dtype,
        address_space=.SHARED,
    ](row_major[BM, BK]())

    var b_tile_sram = stack_allocation[
        mat_b.dtype,
        address_space=.SHARED,
    ](row_major[BK, BN]())

    var n_warp_n = BN // WN
    var warp_m = warp_id() // n_warp_n
    var warp_n = warp_id() % n_warp_n

    # Allocate register tiles.
    var a_reg = stack_allocation[
        mat_a.dtype,
        address_space=.LOCAL,
    ](
        row_major[TM]()
    )  # TM elements for M-dimension vector
    var b_reg = stack_allocation[
        mat_b.dtype,
        address_space=.LOCAL,
    ](
        row_major[TN]()
    )  # TN elements for N-dimension vector
    var c_reg = stack_allocation[
        mat_c.dtype,
        address_space=.LOCAL,
    ](
        row_major[TM, TN]()
    ).fill(0)

    comptime warp_layout = row_major[8, 4]()

    for k_i in range(ceildiv(K, Scalar[mat_a.linear_idx_type](BK))):
        var a_tile_dram = mat_a.tile[BM, BK]((block_idx.y, Int(k_i)))
        copy_dram_to_sram_async[
            thread_layout=Layout.row_major(NUM_THREADS // BK, BK)
        ](a_tile_sram.to_layout_tensor(), a_tile_dram.to_layout_tensor())

        var b_tile_dram = mat_b.tile[BK, BN]((Int(k_i), block_idx.x))
        copy_dram_to_sram_async[
            thread_layout=Layout.row_major(NUM_THREADS // BN, BN)
        ](b_tile_sram.to_layout_tensor(), b_tile_dram.to_layout_tensor())

        async_copy_wait_all()
        barrier()

        comptime for k_j in range(BK):  # Renamed to avoid shadowing outer k_i
            var a_smem_warp_row = a_tile_sram.tile[WM, BK](
                (warp_m, Idx[0])
            ).slice[:, k_j : k_j + 1]()

            var b_smem_warp_row = b_tile_sram.tile[BK, WN](
                (Idx[0], warp_n)
            ).slice[k_j : k_j + 1, :]()
            copy_sram_to_local[src_warp_layout=warp_layout.to_layout(), axis=0](
                a_reg.to_layout_tensor(), a_smem_warp_row.to_layout_tensor()
            )
            copy_sram_to_local[src_warp_layout=warp_layout.to_layout(), axis=1](
                b_reg.to_layout_tensor(), b_smem_warp_row.to_layout_tensor()
            )
            outer_product_acc(c_reg, a_reg, b_reg)

        # Otherwise a data race, faster threads will modify shared memory.
        barrier()

    var c_warp_tile = mat_c.tile[BM, BN]((block_idx.y, block_idx.x)).tile[
        WM, WN
    ]((warp_m, warp_n))

    copy_local_to_dram[dst_thread_layout=warp_layout.to_layout()](
        c_warp_tile.to_layout_tensor(), c_reg.to_layout_tensor()
    )


def test_gemm_kernel_dynamic(ctx: DeviceContext) raises:
    comptime NUM_THREADS = 256
    comptime BM = 64
    comptime BN = 64
    comptime BK = 16
    comptime WM = 32
    comptime WN = 16
    comptime TM = 4
    comptime TN = 4

    comptime M = 1024
    comptime N = 1024
    comptime K = 128

    var a_host = ctx.enqueue_create_host_buffer[.float32](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float32](K * N)
    var c_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    var c_host_ref = ctx.enqueue_create_host_buffer[.float32](M * N)

    for i in range(M * K):
        a_host[i] = Float32(i)

    for i in range(K * N):
        b_host[i] = Float32(i)

    var a_device = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    var mat_a = TileTensor(a_device, row_major[M, K]())
    var mat_b = TileTensor(b_device, row_major[K, N]())
    var mat_c = TileTensor(c_device, row_major[M, N]())

    comptime kernel = gemm_kernel[
        .float32,
        mat_c.LayoutType,
        .float32,
        mat_a.LayoutType,
        .float32,
        mat_b.LayoutType,
        NUM_THREADS,
        BM,
        BN,
        BK,
        WM,
        WN,
        TM,
        TN,
    ]

    ctx.enqueue_function[kernel](
        mat_c,
        mat_a.as_immut(),
        mat_b.as_immut(),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(NUM_THREADS),
    )

    ctx.enqueue_copy(c_host, c_device)

    var c_tensor_ref = TileTensor(c_device_ref, row_major[M, N]())

    # Naive gemm.
    comptime BLOCK_DIM = 16
    comptime gemm_naive = matmul_kernel_naive[
        .float32,
        mat_c.LayoutType,
        .float32,
        mat_a.LayoutType,
        .float32,
        mat_b.LayoutType,
        BLOCK_DIM,
    ]

    ctx.enqueue_function[gemm_naive](
        c_tensor_ref,
        mat_a.as_immut(),
        mat_b.as_immut(),
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref, c_device_ref)
    ctx.synchronize()
    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i], atol=1e-2):
            print(i, c_host[i], c_host_ref[i])
        # Relaxed tolerance for tiled accumulation - different accumulation
        # order leads to different FP rounding errors, especially on B200.
        # With M×N×K = 1024×1024×128 FP32 operations on sequential integer
        # inputs, relative errors up to ~0.014% are expected and acceptable.
        assert_almost_equal(c_host[i], c_host_ref[i], rtol=3e-4)

    if is_benchmark():
        comptime nrun = 200
        comptime nwarmup = 2

        @always_inline
        def run_func(ctx: DeviceContext) raises {imm}:
            ctx.enqueue_function[kernel](
                mat_c,
                mat_a.as_immut(),
                mat_b.as_immut(),
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=(NUM_THREADS),
            )

        # Warmup
        for _i in range(nwarmup):
            ctx.enqueue_function[kernel](
                mat_c,
                mat_a.as_immut(),
                mat_b.as_immut(),
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=(NUM_THREADS),
            )

        var nstime = Float64(ctx.execution_time(run_func, nrun)) / Float64(nrun)
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        print(nrun, "runs avg(s)", sectime, "TFlops/s", TFlop / sectime)

    _ = c_device
    _ = c_device_ref
    _ = a_device
    _ = b_device


def test_gemm_kernel_minimal(ctx: DeviceContext) raises:
    """Minimal debug test with small dimensions to isolate bugs.

    Uses single block (64x64) and single K-tile (16) for easier debugging.
    """
    comptime NUM_THREADS = 256
    comptime BM = 64
    comptime BN = 64
    comptime BK = 16
    comptime WM = 32
    comptime WN = 16
    comptime TM = 4
    comptime TN = 4

    # Small dimensions - single block, single K iteration
    comptime M = 64
    comptime N = 64
    comptime K = 16

    var a_host = ctx.enqueue_create_host_buffer[.float32](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float32](K * N)
    var c_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    var c_host_ref = ctx.enqueue_create_host_buffer[.float32](M * N)

    # Initialize with sequential integers like the main test
    for i in range(M * K):
        a_host[i] = Float32(i)

    for i in range(K * N):
        b_host[i] = Float32(i)

    var a_device = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    var mat_a = TileTensor(a_device, row_major[M, K]())
    var mat_b = TileTensor(b_device, row_major[K, N]())
    var mat_c = TileTensor(c_device, row_major[M, N]())

    comptime kernel = gemm_kernel[
        .float32,
        mat_c.LayoutType,
        .float32,
        mat_a.LayoutType,
        .float32,
        mat_b.LayoutType,
        NUM_THREADS,
        BM,
        BN,
        BK,
        WM,
        WN,
        TM,
        TN,
    ]

    ctx.enqueue_function[kernel](
        mat_c,
        mat_a.as_immut(),
        mat_b.as_immut(),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(NUM_THREADS),
    )

    ctx.enqueue_copy(c_host, c_device)

    var c_tensor_ref = TileTensor(c_device_ref, row_major[M, N]())

    # Naive gemm for reference
    comptime BLOCK_DIM = 16
    comptime gemm_naive = matmul_kernel_naive[
        .float32,
        mat_c.LayoutType,
        .float32,
        mat_a.LayoutType,
        .float32,
        mat_b.LayoutType,
        BLOCK_DIM,
    ]

    ctx.enqueue_function[gemm_naive](
        c_tensor_ref,
        mat_a.as_immut(),
        mat_b.as_immut(),
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref, c_device_ref)
    ctx.synchronize()

    # Print first few elements for inspection
    print("=== Minimal Test Results (M=64, N=64, K=16) ===")
    print("First 10 elements:")
    for i in range(min(10, M * N)):
        var diff = c_host[i] - c_host_ref[i]
        var rel_err: Float32 = (
            abs(diff / c_host_ref[i]) if c_host_ref[i] != 0 else 0.0
        )
        print(
            "  [",
            i,
            "] optimized:",
            c_host[i],
            " reference:",
            c_host_ref[i],
            " diff:",
            diff,
            " rel_err:",
            rel_err,
        )

    # Check element at row boundary (element 64 = position [1,0])
    print("\nRow boundary check:")
    var i = 64
    var diff = c_host[i] - c_host_ref[i]
    var rel_err: Float32 = (
        abs(diff / c_host_ref[i]) if c_host_ref[i] != 0 else 0.0
    )
    print(
        "  [",
        i,
        "] (row 1, col 0) optimized:",
        c_host[i],
        " reference:",
        c_host_ref[i],
        " diff:",
        diff,
        " rel_err:",
        rel_err,
    )

    # Validate all elements
    print("\nValidating all elements...")
    var max_rel_err: Float32 = 0.0
    var max_err_idx = 0
    for i in range(M * N):
        var diff = abs(c_host[i] - c_host_ref[i])
        var rel_err: Float32 = (
            diff / abs(c_host_ref[i]) if c_host_ref[i] != 0 else 0.0
        )
        if rel_err > max_rel_err:
            max_rel_err = rel_err
            max_err_idx = i

        if not isclose(c_host[i], c_host_ref[i], rtol=3e-4):
            print(
                "MISMATCH at",
                i,
                ":",
                c_host[i],
                "vs",
                c_host_ref[i],
                "(rel_err:",
                rel_err,
                ")",
            )

    print("Max relative error:", max_rel_err, "at index", max_err_idx)
    print("Test", "PASSED" if max_rel_err < 3e-4 else "FAILED")

    _ = c_device
    _ = c_device_ref
    _ = a_device
    _ = b_device


def main() raises:
    with DeviceContext() as ctx:
        # Run minimal test first for debugging
        var run_minimal = False
        for arg in argv():
            if arg == "--minimal" or arg == "--debug":
                run_minimal = True
                break

        if run_minimal:
            print("Running minimal debug test...")
            test_gemm_kernel_minimal(ctx)
        else:
            # Run full test
            test_gemm_kernel_dynamic(ctx)


def matmul_kernel_naive[
    c_dtype: DType,
    CLayoutType: TensorLayout,
    a_dtype: DType,
    ALayoutType: TensorLayout,
    b_dtype: DType,
    BLayoutType: TensorLayout,
    BLOCK_DIM: Int,
    transpose_b: Bool = False,
    s_type: DType = get_accum_type[c_dtype](),
](
    c: TileTensor[c_dtype, CLayoutType, MutUntrackedOrigin],
    a: TileTensor[a_dtype, ALayoutType, ImmUntrackedOrigin],
    b: TileTensor[b_dtype, BLayoutType, ImmUntrackedOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
) where (c.flat_rank == 2 and a.flat_rank == 2 and b.flat_rank == 2):
    # `Int` is not device-passable; widen the fixed-width args.
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)
    var x = global_idx.x
    var y = global_idx.y

    if x >= m or y >= n:
        return

    var accum = Scalar[s_type]()

    comptime if transpose_b:
        for i in range(k):
            accum += rebind[Scalar[s_type]](a[x, i].cast[s_type]()) * rebind[
                Scalar[s_type]
            ](b[y, i].cast[s_type]())

    else:
        for i in range(k):
            accum += rebind[Scalar[s_type]](a[x, i].cast[s_type]()) * rebind[
                Scalar[s_type]
            ](b[i, y].cast[s_type]())

    comptime assert c.flat_rank >= 2
    c[x, y] = accum.cast[c.dtype]()


@always_inline
def outer_product_acc(
    res: TileTensor[mut=True, ...],
    lhs: TileTensor,
    rhs: TileTensor,
) where lhs.flat_rank == 1 and rhs.flat_rank == 1:
    """Updates result tensor with the outer product of two vectors.

    Computes `res += outer(lhs, rhs)` where `lhs` and `rhs` are vectors and
    `res` is a matrix.

    Args:
        res: The result matrix to accumulate into, shape (M, N).
        lhs: The left-hand side vector, shape (M,).
        rhs: The right-hand side vector, shape (N,).

    Constraints:

        All tensors must have statically known shapes.
        `res` must be rank 2.
        `lhs` and `rhs` must be rank 1.
        `res.shape[0]` `==` `lhs.shape[0]` and `res.shape[1]` `==` `rhs.shape[0]`.
    """

    comptime assert lhs.element_size == res.element_size
    comptime assert lhs.element_size == rhs.element_size
    comptime assert res.flat_rank == 2
    comptime assert lhs.flat_rank >= 1
    comptime assert rhs.flat_rank >= 1

    comptime dtype = res.dtype
    comptime M = res.static_shape[0]
    comptime N = res.static_shape[1]

    comptime for i in range(M):
        comptime for j in range(N):
            res[i, j] += rebind[res.ElementType](
                (lhs[Idx[i]]).cast[dtype]()
            ) * rebind[res.ElementType](rhs[Idx[j]].cast[dtype]())
