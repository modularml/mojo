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
"""RDNA Conv2D implicit GEMM kernel with WMMA.

Fuses im2col into the RDNA WMMA matmul kernel's A-tile shared memory loader,
eliminating the large intermediate im2col buffer. The B-tile (filter) is loaded
normally from a pre-transposed [N, K] layout.

For common VAE decoder shapes (C_in=128/256/512), C_in is always a multiple of
BLOCK_K, so consecutive K positions within a tile share the same (r,s) filter
position. This enables vectorized 8-wide loads from the NHWC input: the same
load width as the standard matmul kernel's A-tile loader.
"""

from max.gpu import (
    WARP_SIZE,
    block_idx,
    thread_idx,
    lane_id,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.compute.mma import mma as _mma_intrinsic
from layout import TensorEngine, TensorLayout, TileTensor
from std.math import ceildiv
from std.memory import unsafe_stack_allocation
from std.utils import Index, IndexList
from std.utils.numerics import get_accum_type

from linalg.utils import elementwise_epilogue_type
from linalg.utils_gpu import block_swizzle

# WMMA hardware constants
comptime MMA_M = 16
comptime MMA_N = 16
comptime MMA_K = 16
comptime AB_FRAG_SIZE = 16
comptime CD_FRAG_SIZE = 8

# Shared memory row padding (same as matmul kernel)
comptime SMEM_PAD = 8


# =========================================================================
# Im2col A-tile loader
# =========================================================================


@always_inline
def _load_im2col_a_tile[
    dtype: DType,
    BLOCK_M: Int,
    BLOCK_K: Int,
    SMEM_STRIDE: Int,
    NUM_THREADS: Int,
](
    smem: UnsafePointer[mut=True, Scalar[dtype], _, address_space=.SHARED],
    input_ptr: UnsafePointer[mut=False, Scalar[dtype], _],
    block_m_offset: Int,
    k_offset: Int,
    M: Int,
    # Conv geometry
    HW_out: Int,
    W_out: Int,
    H_in: Int,
    W_in: Int,
    C_in: Int,
    S: Int,
    SC: Int,
    pad_h: Int,
    pad_w: Int,
    tid: Int,
):
    """Load an A-tile from NHWC input with on-the-fly im2col into shared memory.

    Uses vectorized 8-wide loads. Requires C_in % BLOCK_K == 0 so that all
    elements within a vector share the same (r, s) filter position, mapping to
    contiguous channels in the NHWC input.
    """
    comptime VECTOR_WIDTH = min(BLOCK_K, 8)
    comptime total_vectors = BLOCK_M * BLOCK_K // VECTOR_WIDTH
    comptime vecs_per_thread = ceildiv(total_vectors, NUM_THREADS)

    var HWC_in = H_in * W_in * C_in
    var WC_in = W_in * C_in

    comptime for i in range(vecs_per_thread):
        var vec_idx = i * NUM_THREADS + tid
        if vec_idx < total_vectors:
            var elem_idx = vec_idx * VECTOR_WIDTH
            var row = elem_idx // BLOCK_K
            var col = elem_idx % BLOCK_K
            var global_m = block_m_offset + row
            var global_k = k_offset + col

            var vec = SIMD[dtype, VECTOR_WIDTH](0)

            if global_m < M:
                # Decompose m -> (batch, h_out, w_out)
                var batch = global_m // HW_out
                var spatial = global_m - batch * HW_out
                var h_out = spatial // W_out
                var w_out = spatial - h_out * W_out

                # Decompose k -> (r, s, c)
                # Since C_in % BLOCK_K == 0 and col < BLOCK_K,
                # all VECTOR_WIDTH elements share the same (r, s).
                var r = global_k // SC
                var sc_val = global_k - r * SC
                var s_val = sc_val // C_in
                var c = sc_val - s_val * C_in

                var h_src = h_out - pad_h + r
                var w_src = w_out - pad_w + s_val

                if h_src >= 0 and h_src < H_in and w_src >= 0 and w_src < W_in:
                    var base_idx = (
                        batch * HWC_in + h_src * WC_in + w_src * C_in + c
                    )
                    vec = input_ptr.load[width=VECTOR_WIDTH](base_idx)

            smem.store(row * SMEM_STRIDE + col, vec)


# =========================================================================
# B-tile loader (reused from matmul pattern)
# =========================================================================


@always_inline
def _load_b_tile_to_smem[
    dtype: DType,
    tile_layout: TensorLayout,
    BLOCK_N: Int,
    BLOCK_K: Int,
    SMEM_STRIDE: Int,
    NUM_THREADS: Int,
    tile_engine: TensorEngine,
](
    smem: UnsafePointer[mut=True, Scalar[dtype], _, address_space=.SHARED],
    tile: TileTensor[mut=True, dtype, tile_layout, _, Engine=tile_engine],
    block_n_offset: Int,
    k_offset: Int,
    max_n: Int,
    tid: Int,
):
    """Load B-tile from [N, K] filter into shared memory with vectorized loads.

    K-dimension bounds checking is omitted because the dispatch only routes here
    when K % BLOCK_K == 0, so k_offset + col is always in-bounds.
    """
    comptime VECTOR_WIDTH = min(BLOCK_K, 8)
    comptime total_vectors = BLOCK_N * BLOCK_K // VECTOR_WIDTH
    comptime vecs_per_thread = ceildiv(total_vectors, NUM_THREADS)

    comptime for i in range(vecs_per_thread):
        var vec_idx = i * NUM_THREADS + tid
        if vec_idx < total_vectors:
            var elem_idx = vec_idx * VECTOR_WIDTH
            var row = elem_idx // BLOCK_K
            var col = elem_idx % BLOCK_K
            var global_row = block_n_offset + row

            if global_row < max_n:
                var vec = tile.load_linear[width=VECTOR_WIDTH](
                    IndexList[2](global_row, k_offset + col)
                )
                smem.store(row * SMEM_STRIDE + col, vec)
            else:
                smem.store(
                    row * SMEM_STRIDE + col,
                    SIMD[dtype, VECTOR_WIDTH](0),
                )


# =========================================================================
# Conv2D implicit GEMM kernel
# =========================================================================


@__name(t"rdna_conv2d_{out_type}_{in_type}_{filter_type}")
def conv2d_kernel_rdna[
    out_type: DType,
    in_type: DType,
    filter_type: DType,
    out_layout: TensorLayout,
    filter_nk_layout: TensorLayout,
    out_engine: TensorEngine,
    filter_nk_engine: TensorEngine,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[out_type](),
    BLOCK_K: Int = 32,
    BLOCK_M: Int = 128,
    BLOCK_N: Int = 128,
    WARPS_M: Int = 8,
    WARPS_N: Int = 2,
    WARP_TILE_M: Int = 1,
    WARP_TILE_N: Int = 4,
](
    output: TileTensor[out_type, out_layout, MutAnyOrigin, Engine=out_engine],
    input_ptr: UnsafePointer[Scalar[in_type], ImmutAnyOrigin],
    filter_nk: TileTensor[
        filter_type, filter_nk_layout, MutAnyOrigin, Engine=filter_nk_engine
    ],
    # GEMM dimensions
    M: Int32,
    N: Int32,
    K: Int32,
    # Conv geometry
    HW_out: Int32,
    W_out: Int32,
    H_in: Int32,
    W_in: Int32,
    C_in: Int32,
    R: Int32,
    S: Int32,
    pad_h: Int32,
    pad_w: Int32,
):
    """Conv2D implicit GEMM kernel for RDNA 3+ GPUs.

    Identical to the RDNA WMMA matmul kernel except the A-tile is loaded
    directly from the NHWC input with on-the-fly im2col coordinate
    computation, eliminating the intermediate im2col buffer.

    The B-tile (filter) must be pre-transposed to [N, K] = [C_out, R*S*C_in]
    layout for vectorized loading with transpose_b=True semantics.

    Output [M, N] in row-major maps directly to NHWC output.

    Constraints (enforced by the dispatch layer):
    - stride = (1, 1) and dilation = (1, 1)
    - K % BLOCK_K == 0 and C_in % BLOCK_K == 0

    Parameters:
        out_type: `DType` of the output tensor.
        in_type: `DType` of the input activation tensor.
        filter_type: `DType` of the filter tensor.
        out_layout: `TensorLayout` of the output `TileTensor`.
        filter_nk_layout: `TensorLayout` of the pre-transposed [N, K] filter
            `TileTensor`.
        out_engine: `TensorEngine` of the output `TileTensor`.
        filter_nk_engine: `TensorEngine` of the pre-transposed [N, K] filter
            `TileTensor`.
        elementwise_lambda_fn: Optional fused epilogue lambda applied per
            output element (defaults to `None`).
        s_type: `DType` used for the accumulator (defaults to the accumulator
            type for `out_type`).
        BLOCK_K: Tile size along the K reduction dimension (defaults to 32).
        BLOCK_M: Tile size along the M output-row dimension (defaults to 128).
        BLOCK_N: Tile size along the N output-column dimension (defaults to
            128).
        WARPS_M: Number of warps tiling the M dimension (defaults to 8).
        WARPS_N: Number of warps tiling the N dimension (defaults to 2).
        WARP_TILE_M: MMA tiles per warp along M (defaults to 1).
        WARP_TILE_N: MMA tiles per warp along N (defaults to 4).

    Args:
        output: Output `TileTensor` of shape [M, N] in row-major, mapping to
            NHWC output.
        input_ptr: Pointer to the NHWC input activation tensor of `in_type`.
        filter_nk: Filter `TileTensor` pre-transposed to [N, K] = [C_out,
            R*S*C_in] layout.
        M: Number of output rows; equals batch * H_out * W_out.
        N: Number of output columns; equals C_out.
        K: Reduction dimension length; equals R * S * C_in.
        HW_out: Product of output spatial dimensions H_out * W_out.
        W_out: Output spatial width.
        H_in: Input spatial height.
        W_in: Input spatial width.
        C_in: Number of input channels.
        R: Filter height.
        S: Filter width.
        pad_h: Zero padding applied to the input height.
        pad_w: Zero padding applied to the input width.
    """
    var _M = Int(M)
    var _N = Int(N)
    var _K = Int(K)
    var _HW_out = Int(HW_out)
    var _W_out = Int(W_out)
    var _H_in = Int(H_in)
    var _W_in = Int(W_in)
    var _C_in = Int(C_in)
    var _R = Int(R)
    var _S = Int(S)
    var _pad_h = Int(pad_h)
    var _pad_w = Int(pad_w)
    comptime assert output.flat_rank == 2, "output must have flat_rank == 2"
    comptime assert filter_nk.flat_rank == 2, "filter must have flat_rank == 2"
    comptime assert BLOCK_K % MMA_K == 0
    comptime assert WARPS_M * WARP_TILE_M * MMA_M == BLOCK_M
    comptime assert WARPS_N * WARP_TILE_N * MMA_N == BLOCK_N

    comptime K_ITERS = BLOCK_K // MMA_K
    comptime SMEM_STRIDE = BLOCK_K + SMEM_PAD
    comptime NUM_WARPS = WARPS_M * WARPS_N
    comptime NUM_THREADS = NUM_WARPS * WARP_SIZE
    comptime NUM_C_TILES = WARP_TILE_M * WARP_TILE_N

    # Block coordinates with swizzle for L2 locality
    var grid_dim = IndexList[2](ceildiv(_N, BLOCK_N), ceildiv(_M, BLOCK_M))
    var swizzled = block_swizzle(
        IndexList[2](block_idx.x, block_idx.y), grid_dim
    )
    var block_n = swizzled[0]
    var block_m = swizzled[1]

    var block_m_offset = block_m * BLOCK_M
    var block_n_offset = block_n * BLOCK_N

    var tid = thread_idx.x
    var wid = warp_id()
    var lid = lane_id()

    var warp_m, warp_n = divmod(wid, WARPS_N)
    var effective_lane = lid % 16

    # Pre-compute conv geometry values used in im2col loader
    var SC = _S * _C_in

    # Double-buffered shared memory
    var a_smem_0 = unsafe_stack_allocation[
        BLOCK_M * SMEM_STRIDE,
        in_type,
        address_space=.SHARED,
    ]()
    var a_smem_1 = unsafe_stack_allocation[
        BLOCK_M * SMEM_STRIDE,
        in_type,
        address_space=.SHARED,
    ]()
    var b_smem_0 = unsafe_stack_allocation[
        BLOCK_N * SMEM_STRIDE,
        filter_type,
        address_space=.SHARED,
    ]()
    var b_smem_1 = unsafe_stack_allocation[
        BLOCK_N * SMEM_STRIDE,
        filter_type,
        address_space=.SHARED,
    ]()

    # Initialize C accumulators
    var c_accum = Array[SIMD[s_type, CD_FRAG_SIZE], NUM_C_TILES](
        fill=SIMD[s_type, CD_FRAG_SIZE](0)
    )

    var num_k_tiles = _K // BLOCK_K

    # --- Load first _K-tile into buffer 0 ---
    _load_im2col_a_tile[in_type, BLOCK_M, BLOCK_K, SMEM_STRIDE, NUM_THREADS](
        a_smem_0,
        input_ptr,
        block_m_offset,
        0,
        _M,
        _HW_out,
        _W_out,
        _H_in,
        _W_in,
        _C_in,
        _S,
        SC,
        _pad_h,
        _pad_w,
        tid,
    )
    _load_b_tile_to_smem[
        filter_type,
        filter_nk_layout,
        BLOCK_N,
        BLOCK_K,
        SMEM_STRIDE,
        NUM_THREADS,
        tile_engine=filter_nk_engine,
    ](b_smem_0, filter_nk, block_n_offset, 0, _N, tid)
    barrier()

    # --- Main _K-dimension loop with double buffering ---
    for k_tile in range(num_k_tiles):
        var a_cur = a_smem_0 if k_tile % 2 == 0 else a_smem_1
        var b_cur = b_smem_0 if k_tile % 2 == 0 else b_smem_1
        var a_next = a_smem_1 if k_tile % 2 == 0 else a_smem_0
        var b_next = b_smem_1 if k_tile % 2 == 0 else b_smem_0

        # --- COMPUTE: WMMA on current buffer ---
        comptime for k_inner in range(K_ITERS):
            var a_frag = Array[SIMD[in_type, AB_FRAG_SIZE], WARP_TILE_M](
                fill=SIMD[in_type, AB_FRAG_SIZE](0)
            )
            comptime for wm in range(WARP_TILE_M):
                var a_row = (
                    warp_m * WARP_TILE_M * MMA_M + wm * MMA_M + effective_lane
                )
                comptime k_base = k_inner * MMA_K
                a_frag[wm] = a_cur.load[width=AB_FRAG_SIZE](
                    a_row * SMEM_STRIDE + k_base
                )

            var b_frag = Array[SIMD[filter_type, AB_FRAG_SIZE], WARP_TILE_N](
                fill=SIMD[filter_type, AB_FRAG_SIZE](0)
            )
            comptime for wn in range(WARP_TILE_N):
                var b_row = (
                    warp_n * WARP_TILE_N * MMA_N + wn * MMA_N + effective_lane
                )
                comptime k_base = k_inner * MMA_K
                b_frag[wn] = b_cur.load[width=AB_FRAG_SIZE](
                    b_row * SMEM_STRIDE + k_base
                )

            comptime for wm in range(WARP_TILE_M):
                comptime for wn in range(WARP_TILE_N):
                    var c_idx = wm * WARP_TILE_N + wn
                    _mma_intrinsic(
                        c_accum[c_idx],
                        a_frag[wm],
                        b_frag[wn],
                        c_accum[c_idx],
                    )

        # --- PREFETCH: load next _K-tile ---
        if k_tile + 1 < num_k_tiles:
            var next_k_offset = (k_tile + 1) * BLOCK_K
            _load_im2col_a_tile[
                in_type, BLOCK_M, BLOCK_K, SMEM_STRIDE, NUM_THREADS
            ](
                a_next,
                input_ptr,
                block_m_offset,
                next_k_offset,
                _M,
                _HW_out,
                _W_out,
                _H_in,
                _W_in,
                _C_in,
                _S,
                SC,
                _pad_h,
                _pad_w,
                tid,
            )
            _load_b_tile_to_smem[
                filter_type,
                filter_nk_layout,
                BLOCK_N,
                BLOCK_K,
                SMEM_STRIDE,
                NUM_THREADS,
                tile_engine=filter_nk_engine,
            ](b_next, filter_nk, block_n_offset, next_k_offset, _N, tid)

        barrier()

    # --- Store C results to global memory ---
    var lane_row_offset, lane_col = divmod(lid, 16)

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

                if global_row < _M and global_col < _N:
                    comptime if elementwise_lambda_fn:
                        comptime elementwise_lambda = (
                            elementwise_lambda_fn.value()
                        )
                        elementwise_lambda[out_type, 1](
                            Index(global_row, global_col),
                            c_accum[c_idx][v].cast[out_type](),
                        )
                    else:
                        output[global_row, global_col] = c_accum[c_idx][v].cast[
                            out_type
                        ]()
