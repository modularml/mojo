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
"""AMD RDNA matmul kernel with WMMA for RDNA 3+ and naive fallback for older.

RDNA 3+ (gfx11xx/gfx12xx): 16x16x16 WMMA with Wave32. The K-loop runs in one
of two modes: a double-buffered LDS loop (default), or a single-buffer
register-staged pipeline that keeps the next K-tile's global loads in flight
during the current tile's WMMA -- used for the large transpose_b path where its
bigger BLOCK_K fits only with one buffer. Block swizzle for L2 locality.

Dispatched large-shape config (register pipeline, transpose_b, k % 64 == 0):
  128x128 block, 8 warps (4x2), warp_tile 2x4, BLOCK_K=64

RDNA 1/2 (gfx10xx): Falls back to a per-thread naive matmul.
"""

from std.math import ceildiv
from std.sys import size_of
from std.sys.info import _is_amd_rdna2_or_earlier

from max.gpu import (
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.compute.mma import mma as _mma_intrinsic
from layout import TensorLayout, TileTensor
from std.memory import unsafe_stack_allocation
from std.utils import Index, IndexList
from std.utils.numerics import get_accum_type

from ....utils import elementwise_epilogue_type
from ....utils_gpu import block_swizzle

# Defaults for the naive kernel path (RDNA 1/2)
comptime BLOCK_M = 64
comptime BLOCK_N = 64
comptime NUM_WARPS = 4
comptime NUM_THREADS = NUM_WARPS * WARP_SIZE  # 128

# WMMA hardware constants
comptime MMA_M = 16
comptime MMA_N = 16
comptime MMA_K = 16
comptime AB_FRAG_SIZE = 16
comptime CD_FRAG_SIZE = 8

# Shared-memory row padding (elements): pads the per-row stride to
# BLOCK_K + SMEM_PAD to spread per-lane WMMA fragment rows across LDS banks.
# Kept fixed across BLOCK_K configs (stride 24/40/72 for BK 16/32/64) and
# validated empirically (correctness + throughput).
comptime SMEM_PAD = 8

# Addressable LDS per workgroup on RDNA 3/3.5/4 (gfx11xx/gfx12xx). Used to
# decide when double-buffering fits.
comptime RDNA_LDS_BYTES = 64 * 1024


@__name(t"gemm_kernel_rdna_{c_type}_{a_type}_{b_type}_{transpose_b}")
def gemm_kernel_rdna[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
    BLOCK_K: Int = 32,
    BLOCK_M: Int = 128,
    BLOCK_N: Int = 128,
    WARPS_M: Int = 8,
    WARPS_N: Int = 2,
    WARP_TILE_M: Int = 1,
    WARP_TILE_N: Int = 4,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin],
    m: Int32,
    n: Int32,
    k: Int32,
):
    """GEMM kernel for AMD RDNA GPUs.

    On RDNA 3+ (gfx11xx/gfx12xx), uses 16x16x16 WMMA instructions (see
    :func:`_wmma_matmul_kernel` for the K-loop strategies). On older RDNA
    (gfx10xx), falls back to a per-thread naive matmul.

    Parameters:
        c_type: Element type of the output tile `c`.
        a_type: Element type of the input tile `a`.
        b_type: Element type of the input tile `b`.
        c_layout: Memory layout of the output tile `c`.
        a_layout: Memory layout of the input tile `a`.
        b_layout: Memory layout of the input tile `b`.
        transpose_b: Whether `b` is stored as `(N, K)` instead of `(K, N)`
            (defaults to `True`).
        elementwise_lambda_fn: Optional per-element epilogue that replaces
            the direct store to `c` (defaults to `None`).
        s_type: Accumulation type used in the inner K-loop (defaults to
            the natural accumulation type of `c_type`).
        BLOCK_K: K-dimension tile size in elements processed per
            K-iteration (defaults to 32).
        BLOCK_M: M-dimension block size in rows of the output tile
            (defaults to 128).
        BLOCK_N: N-dimension block size in columns of the output tile
            (defaults to 128).
        WARPS_M: Number of warps along the M dimension of the warp grid
            (defaults to 8).
        WARPS_N: Number of warps along the N dimension of the warp grid
            (defaults to 2).
        WARP_TILE_M: Number of 16x16 MMA tiles each warp computes along M
            (defaults to 1).
        WARP_TILE_N: Number of 16x16 MMA tiles each warp computes along N
            (defaults to 4).

    Args:
        c: Output tile of shape `(m, n)` holding the product `a @ b`.
        a: Input tile of shape `(m, k)`.
        b: Input tile of shape `(k, n)` when `transpose_b` is `False`,
            or `(n, k)` when `True`.
        m: Number of rows in the output and in `a`.
        n: Number of columns in the output and in `b`.
        k: Contraction dimension shared by `a` and `b`.
    """
    var _m = Int(m)
    var _n = Int(n)
    var _k = Int(k)
    comptime assert c.flat_rank == 2, "c must have flat_rank == 2"
    comptime assert a.flat_rank == 2, "a must have flat_rank == 2"
    comptime assert b.flat_rank == 2, "b must have flat_rank == 2"

    comptime if _is_amd_rdna2_or_earlier() or a_type not in (
        DType.float16,
        DType.bfloat16,
    ):
        _naive_matmul_kernel[
            c_type,
            a_type,
            b_type,
            c_layout,
            a_layout,
            b_layout,
            transpose_b,
            elementwise_lambda_fn,
            s_type,
        ](c, a, b, _m, _n, _k)
    else:
        _wmma_matmul_kernel[
            c_type,
            a_type,
            b_type,
            c_layout,
            a_layout,
            b_layout,
            transpose_b,
            elementwise_lambda_fn,
            s_type,
            BLOCK_K,
            BLOCK_M,
            BLOCK_N,
            WARPS_M,
            WARPS_N,
            WARP_TILE_M,
            WARP_TILE_N,
        ](c, a, b, _m, _n, _k)


def _naive_matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    transpose_b: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    s_type: DType,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    """Per-thread naive matmul for RDNA 1/2 (no WMMA support).

    Each thread in the 128-thread workgroup iterates over output elements
    assigned to it within the block's 64x64 tile. With 128 threads covering
    4096 elements, each thread handles 32 output elements.
    """
    comptime assert c.flat_rank == 2, "c must have flat_rank == 2"
    comptime assert a.flat_rank == 2, "a must have flat_rank == 2"
    comptime assert b.flat_rank == 2, "b must have flat_rank == 2"

    var block_m_offset = block_idx.y * BLOCK_M
    var block_n_offset = block_idx.x * BLOCK_N
    var tid = thread_idx.x

    # 128 threads handle 64*64 = 4096 elements → 32 elements per thread
    comptime for elem in range(32):
        var linear = tid * 32 + elem
        var local_row, local_col = divmod(linear, BLOCK_N)
        var global_row = block_m_offset + local_row
        var global_col = block_n_offset + local_col

        if global_row < m and global_col < n:
            var accum = Scalar[s_type](0)

            comptime if transpose_b:
                for i in range(k):
                    accum += rebind[Scalar[s_type]](
                        a[global_row, i].cast[s_type]()
                    ) * rebind[Scalar[s_type]](b[global_col, i].cast[s_type]())
            else:
                for i in range(k):
                    accum += rebind[Scalar[s_type]](
                        a[global_row, i].cast[s_type]()
                    ) * rebind[Scalar[s_type]](b[i, global_col].cast[s_type]())

            comptime if elementwise_lambda_fn:
                comptime elementwise_lambda = elementwise_lambda_fn.value()
                elementwise_lambda[c_type, 1](
                    Index(global_row, global_col),
                    accum.cast[c_type](),
                )
            else:
                c[global_row, global_col] = accum.cast[c_type]()


@always_inline
def _load_tile_to_smem[
    dtype: DType,
    tile_layout: TensorLayout,
    transpose_b: Bool,
    is_b_tile: Bool,
    BLOCK_ROWS: Int,
    BLOCK_K: Int,
    SMEM_STRIDE: Int,
    NUM_THREADS: Int,
](
    smem: UnsafePointer[mut=True, Scalar[dtype], _, address_space=.SHARED],
    tile: TileTensor[dtype, tile_layout, ImmutAnyOrigin],
    block_row_offset: Int,
    k_offset: Int,
    max_rows: Int,
    max_k: Int,
    tid: Int,
):
    """Cooperatively load a tile from global to shared memory.

    Uses vectorized 128-bit loads when the access pattern is coalesced (A tiles
    and B tiles with transpose_b=True). Falls back to scalar loads for strided
    access (B tiles with transpose_b=False, which requires transposing).
    SMEM_STRIDE includes padding to avoid LDS bank conflicts during compute.
    """
    comptime assert tile.flat_rank == 2, "tile must have flat_rank == 2"

    # Vectorize when access is coalesced along K (stride-1 dimension).
    # A (M,K) row-major and B (N,K) with transpose_b: coalesced along K.
    # B (K,N) without transpose: strided (needs transpose), use scalar path.
    comptime can_vectorize = not (is_b_tile and not transpose_b)

    comptime if can_vectorize:
        # 128-bit vector loads: 8 elements for fp16/bf16 (8 × 2 = 16 bytes).
        # K-dimension bounds are NOT checked here; callers must ensure
        # k % BLOCK_K == 0 so that every vector load stays in bounds.
        comptime VECTOR_WIDTH = min(BLOCK_K, 8)
        comptime assert (
            BLOCK_K % VECTOR_WIDTH == 0
        ), "BLOCK_K must be divisible by VECTOR_WIDTH"
        comptime total_vectors = BLOCK_ROWS * BLOCK_K // VECTOR_WIDTH
        comptime vecs_per_thread = ceildiv(total_vectors, NUM_THREADS)

        # Coalesced vectorized loads: adjacent threads load adjacent vectors
        comptime for i in range(vecs_per_thread):
            var vec_idx = i * NUM_THREADS + tid
            if vec_idx < total_vectors:
                var elem_idx = vec_idx * VECTOR_WIDTH
                var row = elem_idx // BLOCK_K
                var col = elem_idx % BLOCK_K
                var global_row = block_row_offset + row

                if global_row < max_rows:
                    var vec = tile.load_linear[width=VECTOR_WIDTH](
                        IndexList[2](global_row, k_offset + col)
                    )
                    smem.store(row * SMEM_STRIDE + col, vec)
                else:
                    smem.store(
                        row * SMEM_STRIDE + col,
                        SIMD[dtype, VECTOR_WIDTH](0),
                    )
    else:
        # Scalar path for B tile with transpose_b=False (strided global access)
        comptime assert BLOCK_ROWS * BLOCK_K % NUM_THREADS == 0, (
            "Scalar tile load requires BLOCK_ROWS * BLOCK_K divisible by"
            " NUM_THREADS"
        )
        comptime elems_per_thread = BLOCK_ROWS * BLOCK_K // NUM_THREADS

        # Coalesced: adjacent threads access adjacent elements
        comptime for i in range(elems_per_thread):
            var elem_idx = i * NUM_THREADS + tid
            var row, col = divmod(elem_idx, BLOCK_K)
            var global_row = block_row_offset + row
            var global_col = k_offset + col

            var val = Scalar[dtype](0)

            # B is (K, N): load transposed into shared memory as (N, K)
            if global_col < max_k and global_row < max_rows:
                val = rebind[Scalar[dtype]](tile[global_col, global_row])

            smem[row * SMEM_STRIDE + col] = val


@always_inline
def _load_tile_regs[
    dtype: DType,
    tile_layout: TensorLayout,
    BLOCK_ROWS: Int,
    BLOCK_K: Int,
    NUM_THREADS: Int,
    VECS_PER_THREAD: Int,
    VECTOR_WIDTH: Int,
](
    tile: TileTensor[dtype, tile_layout, ImmutAnyOrigin],
    block_row_offset: Int,
    k_offset: Int,
    max_rows: Int,
    tid: Int,
) -> Array[SIMD[dtype, VECTOR_WIDTH], VECS_PER_THREAD]:
    """Issue a tile's coalesced global loads into a register buffer.

    Load phase of the register-staged pipeline (drained by _store_tile_regs);
    lets the caller overlap the next K-tile's global loads with WMMA on the
    current tile. Coalesced (transpose_b=True) path only.
    """
    var regs = Array[SIMD[dtype, VECTOR_WIDTH], VECS_PER_THREAD](
        fill=SIMD[dtype, VECTOR_WIDTH](0)
    )
    comptime total_vectors = BLOCK_ROWS * BLOCK_K // VECTOR_WIDTH
    comptime for i in range(VECS_PER_THREAD):
        var vec_idx = i * NUM_THREADS + tid
        if vec_idx < total_vectors:
            var elem_idx = vec_idx * VECTOR_WIDTH
            var global_row = block_row_offset + elem_idx // BLOCK_K
            if global_row < max_rows:
                regs[i] = tile.load_linear[width=VECTOR_WIDTH](
                    IndexList[2](global_row, k_offset + elem_idx % BLOCK_K)
                )
    return regs^


@always_inline
def _store_tile_regs[
    dtype: DType,
    BLOCK_ROWS: Int,
    BLOCK_K: Int,
    SMEM_STRIDE: Int,
    NUM_THREADS: Int,
    VECS_PER_THREAD: Int,
    VECTOR_WIDTH: Int,
](
    smem: UnsafePointer[mut=True, Scalar[dtype], _, address_space=.SHARED],
    regs: Array[SIMD[dtype, VECTOR_WIDTH], VECS_PER_THREAD],
    tid: Int,
):
    """Drain a register-loaded tile into LDS."""
    comptime total_vectors = BLOCK_ROWS * BLOCK_K // VECTOR_WIDTH
    comptime for i in range(VECS_PER_THREAD):
        var vec_idx = i * NUM_THREADS + tid
        if vec_idx < total_vectors:
            var elem_idx = vec_idx * VECTOR_WIDTH
            var row = elem_idx // BLOCK_K
            smem.store(row * SMEM_STRIDE + elem_idx % BLOCK_K, regs[i])


@always_inline
def _compute_ktile[
    a_type: DType,
    b_type: DType,
    s_type: DType,
    SMEM_STRIDE: Int,
    K_ITERS: Int,
    WARP_TILE_M: Int,
    WARP_TILE_N: Int,
](
    a_smem: UnsafePointer[mut=True, Scalar[a_type], _, address_space=.SHARED],
    b_smem: UnsafePointer[mut=True, Scalar[b_type], _, address_space=.SHARED],
    mut c_accum: Array[SIMD[s_type, CD_FRAG_SIZE], WARP_TILE_M * WARP_TILE_N],
    warp_m: Int,
    warp_n: Int,
    effective_lane: Int,
):
    """Accumulate one LDS tile's WMMA products into ``c_accum``.

    Shared by the register-pipelined and double-buffered loops; ``a_smem`` /
    ``b_smem`` are the current tile's A and B buffers.
    """
    comptime for k_inner in range(K_ITERS):
        var a_frag = Array[SIMD[a_type, AB_FRAG_SIZE], WARP_TILE_M](
            fill=SIMD[a_type, AB_FRAG_SIZE](0)
        )
        comptime for wm in range(WARP_TILE_M):
            var a_row = (
                warp_m * WARP_TILE_M * MMA_M + wm * MMA_M + effective_lane
            )
            comptime k_base = k_inner * MMA_K
            a_frag[wm] = a_smem.load[width=AB_FRAG_SIZE](
                a_row * SMEM_STRIDE + k_base
            )
        var b_frag = Array[SIMD[b_type, AB_FRAG_SIZE], WARP_TILE_N](
            fill=SIMD[b_type, AB_FRAG_SIZE](0)
        )
        comptime for wn in range(WARP_TILE_N):
            var b_row = (
                warp_n * WARP_TILE_N * MMA_N + wn * MMA_N + effective_lane
            )
            comptime k_base = k_inner * MMA_K
            b_frag[wn] = b_smem.load[width=AB_FRAG_SIZE](
                b_row * SMEM_STRIDE + k_base
            )
        comptime for wm in range(WARP_TILE_M):
            comptime for wn in range(WARP_TILE_N):
                var c_idx = wm * WARP_TILE_N + wn
                _mma_intrinsic(
                    c_accum[c_idx], a_frag[wm], b_frag[wn], c_accum[c_idx]
                )


def _wmma_matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    transpose_b: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    s_type: DType,
    BLOCK_K: Int = 32,
    BLOCK_M: Int = 128,
    BLOCK_N: Int = 128,
    WARPS_M: Int = 8,
    WARPS_N: Int = 2,
    WARP_TILE_M: Int = 1,
    WARP_TILE_N: Int = 4,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    """WMMA-based GEMM kernel for RDNA 3+ GPUs.

    16x16x16 WMMA. Each workgroup computes a BLOCK_M x BLOCK_N output tile with
    a WARPS_M x WARPS_N warp grid, each warp handling WARP_TILE_M x WARP_TILE_N
    16x16 tiles.

    The K-loop strategy is chosen at compile time from the LDS budget: when two
    A and two B tiles fit, it double-buffers with compute-before-prefetch
    overlap (both transpose_b values). When they don't (e.g. 128x128 at
    BLOCK_K=64), it falls back to a single tile fed by a register-staged
    prefetch (_load_tile_regs / _store_tile_regs) that keeps the next K-tile's
    global loads in flight during the current tile's WMMA -- this needs
    coalesced loads, so it only applies when transpose_b is True.
    """
    comptime assert c.flat_rank == 2, "c must have flat_rank == 2"
    comptime assert a.flat_rank == 2, "a must have flat_rank == 2"
    comptime assert b.flat_rank == 2, "b must have flat_rank == 2"
    comptime assert BLOCK_K % MMA_K == 0, "BLOCK_K must be a multiple of MMA_K"
    comptime assert (
        WARPS_M * WARP_TILE_M * MMA_M == BLOCK_M
    ), "WARPS_M * WARP_TILE_M * MMA_M must equal BLOCK_M"
    comptime assert (
        WARPS_N * WARP_TILE_N * MMA_N == BLOCK_N
    ), "WARPS_N * WARP_TILE_N * MMA_N must equal BLOCK_N"

    # Derive tile configuration from parameters
    comptime K_ITERS = BLOCK_K // MMA_K
    comptime SMEM_STRIDE = BLOCK_K + SMEM_PAD
    comptime NUM_WARPS = WARPS_M * WARPS_N
    comptime NUM_THREADS = NUM_WARPS * WARP_SIZE
    comptime NUM_C_TILES = WARP_TILE_M * WARP_TILE_N

    # Bytes needed to double-buffer (two A tiles + two B tiles). When this
    # exceeds the LDS budget we can't double-buffer, so use the single-tile
    # register-staged pipeline instead -- which requires coalesced loads.
    comptime DOUBLE_BUFFER_BYTES = 2 * (
        BLOCK_M * SMEM_STRIDE * size_of[a_type]()
        + BLOCK_N * SMEM_STRIDE * size_of[b_type]()
    )

    # Block coordinates with swizzle for L2 locality
    var grid_dim = IndexList[2](ceildiv(n, BLOCK_N), ceildiv(m, BLOCK_M))
    var swizzled = block_swizzle(
        IndexList[2](block_idx.x, block_idx.y), grid_dim
    )
    var block_n = swizzled[0]
    var block_m = swizzled[1]

    var block_m_offset = block_m * BLOCK_M
    var block_n_offset = block_n * BLOCK_N

    # Thread identification
    var tid = thread_idx.x
    var wid = warp_id()
    var lid = lane_id()

    # Warp position in the WARPS_M x WARPS_N grid
    var warp_m, warp_n = divmod(wid, WARPS_N)

    # Effective lane for RDNA WMMA (lanes 0-15 and 16-31 hold same data)
    var effective_lane = lid % 16

    # Initialize C accumulators (WARP_TILE_M * WARP_TILE_N tiles per warp)
    var c_accum = Array[SIMD[s_type, CD_FRAG_SIZE], NUM_C_TILES](
        fill=SIMD[s_type, CD_FRAG_SIZE](0)
    )

    # Dispatch guarantees k % BLOCK_K == 0, so integer division is exact.
    var num_k_tiles = k // BLOCK_K

    comptime if transpose_b and DOUBLE_BUFFER_BYTES > RDNA_LDS_BYTES:
        # Single LDS tile fed by a register-staged prefetch: the next K-tile's
        # global loads stay in registers during the current tile's WMMA. One
        # buffer lets a larger BLOCK_K fit than double-buffering would.
        var a_smem = unsafe_stack_allocation[
            BLOCK_M * SMEM_STRIDE, a_type, address_space=.SHARED
        ]()
        var b_smem = unsafe_stack_allocation[
            BLOCK_N * SMEM_STRIDE, b_type, address_space=.SHARED
        ]()

        # Per-thread 128-bit vector counts for the register-staged loads.
        comptime VW = min(BLOCK_K, 8)
        comptime A_VECS = ceildiv(BLOCK_M * BLOCK_K // VW, NUM_THREADS)
        comptime B_VECS = ceildiv(BLOCK_N * BLOCK_K // VW, NUM_THREADS)

        # Prologue: stage tile 0 into LDS via registers.
        var a_regs = _load_tile_regs[
            a_type, a_layout, BLOCK_M, BLOCK_K, NUM_THREADS, A_VECS, VW
        ](a, block_m_offset, 0, m, tid)
        var b_regs = _load_tile_regs[
            b_type, b_layout, BLOCK_N, BLOCK_K, NUM_THREADS, B_VECS, VW
        ](b, block_n_offset, 0, n, tid)
        _store_tile_regs[
            a_type, BLOCK_M, BLOCK_K, SMEM_STRIDE, NUM_THREADS, A_VECS, VW
        ](a_smem, a_regs, tid)
        _store_tile_regs[
            b_type, BLOCK_N, BLOCK_K, SMEM_STRIDE, NUM_THREADS, B_VECS, VW
        ](b_smem, b_regs, tid)
        barrier()

        for k_tile in range(num_k_tiles):
            # Prefetch next tile's globals (kept in flight during compute).
            if k_tile + 1 < num_k_tiles:
                var nk = (k_tile + 1) * BLOCK_K
                a_regs = _load_tile_regs[
                    a_type, a_layout, BLOCK_M, BLOCK_K, NUM_THREADS, A_VECS, VW
                ](a, block_m_offset, nk, m, tid)
                b_regs = _load_tile_regs[
                    b_type, b_layout, BLOCK_N, BLOCK_K, NUM_THREADS, B_VECS, VW
                ](b, block_n_offset, nk, n, tid)

            _compute_ktile[
                a_type,
                b_type,
                s_type,
                SMEM_STRIDE,
                K_ITERS,
                WARP_TILE_M,
                WARP_TILE_N,
            ](a_smem, b_smem, c_accum, warp_m, warp_n, effective_lane)
            barrier()  # all warps done reading LDS

            if k_tile + 1 < num_k_tiles:
                _store_tile_regs[
                    a_type,
                    BLOCK_M,
                    BLOCK_K,
                    SMEM_STRIDE,
                    NUM_THREADS,
                    A_VECS,
                    VW,
                ](a_smem, a_regs, tid)
                _store_tile_regs[
                    b_type,
                    BLOCK_N,
                    BLOCK_K,
                    SMEM_STRIDE,
                    NUM_THREADS,
                    B_VECS,
                    VW,
                ](b_smem, b_regs, tid)
                barrier()  # LDS ready for next compute
    else:
        # Double-buffered LDS with compute-before-prefetch overlap; handles
        # both transpose_b values.
        comptime assert DOUBLE_BUFFER_BYTES <= RDNA_LDS_BYTES, (
            "double-buffer tiles exceed LDS; this config needs transpose_b=True"
            " to use the register-staged pipeline"
        )
        var a_smem_0 = unsafe_stack_allocation[
            BLOCK_M * SMEM_STRIDE, a_type, address_space=.SHARED
        ]()
        var a_smem_1 = unsafe_stack_allocation[
            BLOCK_M * SMEM_STRIDE, a_type, address_space=.SHARED
        ]()
        var b_smem_0 = unsafe_stack_allocation[
            BLOCK_N * SMEM_STRIDE, b_type, address_space=.SHARED
        ]()
        var b_smem_1 = unsafe_stack_allocation[
            BLOCK_N * SMEM_STRIDE, b_type, address_space=.SHARED
        ]()

        # Load first K-tile into buffer 0.
        _load_tile_to_smem[
            a_type,
            a_layout,
            transpose_b,
            is_b_tile=False,
            BLOCK_ROWS=BLOCK_M,
            BLOCK_K=BLOCK_K,
            SMEM_STRIDE=SMEM_STRIDE,
            NUM_THREADS=NUM_THREADS,
        ](a_smem_0, a, block_m_offset, 0, m, k, tid)
        _load_tile_to_smem[
            b_type,
            b_layout,
            transpose_b,
            is_b_tile=True,
            BLOCK_ROWS=BLOCK_N,
            BLOCK_K=BLOCK_K,
            SMEM_STRIDE=SMEM_STRIDE,
            NUM_THREADS=NUM_THREADS,
        ](b_smem_0, b, block_n_offset, 0, n, k, tid)
        barrier()

        for k_tile in range(num_k_tiles):
            var a_cur = a_smem_0 if k_tile % 2 == 0 else a_smem_1
            var b_cur = b_smem_0 if k_tile % 2 == 0 else b_smem_1
            var a_next = a_smem_1 if k_tile % 2 == 0 else a_smem_0
            var b_next = b_smem_1 if k_tile % 2 == 0 else b_smem_0

            _compute_ktile[
                a_type,
                b_type,
                s_type,
                SMEM_STRIDE,
                K_ITERS,
                WARP_TILE_M,
                WARP_TILE_N,
            ](a_cur, b_cur, c_accum, warp_m, warp_n, effective_lane)

            # Prefetch next K-tile into the other buffer (after compute).
            if k_tile + 1 < num_k_tiles:
                var next_k_offset = (k_tile + 1) * BLOCK_K
                _load_tile_to_smem[
                    a_type,
                    a_layout,
                    transpose_b,
                    is_b_tile=False,
                    BLOCK_ROWS=BLOCK_M,
                    BLOCK_K=BLOCK_K,
                    SMEM_STRIDE=SMEM_STRIDE,
                    NUM_THREADS=NUM_THREADS,
                ](a_next, a, block_m_offset, next_k_offset, m, k, tid)
                _load_tile_to_smem[
                    b_type,
                    b_layout,
                    transpose_b,
                    is_b_tile=True,
                    BLOCK_ROWS=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                    SMEM_STRIDE=SMEM_STRIDE,
                    NUM_THREADS=NUM_THREADS,
                ](b_next, b, block_n_offset, next_k_offset, n, k, tid)

            barrier()

    # --- Store C results to global memory ---
    # WMMA output mapping: lane l, element v -> C[row=v*2+l//16, col=l%16]
    var lane_row_offset, lane_col = divmod(
        lid, 16
    )  # lane_row_offset: 0 for lanes 0-15, 1 for lanes 16-31

    comptime for wm in range(WARP_TILE_M):
        comptime for wn in range(WARP_TILE_N):
            var c_idx = wm * WARP_TILE_N + wn

            comptime for v in range(CD_FRAG_SIZE):
                var global_row = (
                    block_m_offset
                    + warp_m * WARP_TILE_M * MMA_M
                    + wm * MMA_M
                    + v * 2
                    + lane_row_offset
                )
                var global_col = (
                    block_n_offset
                    + warp_n * WARP_TILE_N * MMA_N
                    + wn * MMA_N
                    + lane_col
                )

                if global_row < m and global_col < n:
                    comptime if elementwise_lambda_fn:
                        comptime elementwise_lambda = (
                            elementwise_lambda_fn.value()
                        )
                        elementwise_lambda[c_type, 1](
                            Index(global_row, global_col),
                            c_accum[c_idx][v].cast[c_type](),
                        )
                    else:
                        c[global_row, global_col] = c_accum[c_idx][v].cast[
                            c_type
                        ]()
