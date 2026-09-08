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
"""
Implements blockwise scaled FP8 matrix multiplication kernels targeting the
NVIDIA SM100 (Blackwell) architecture using 1D A-scales and 2D B-scales.
"""
from std.collections import Optional
from std.math import ceildiv, gcd
from std.math.uutils import umod, ufloordiv
from std.sys import align_of, size_of

from max.gpu import WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import block_rank_in_cluster, elect_one_sync
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import block_idx, lane_id, warp_id as get_warp_id
from max.gpu.memory import external_memory
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.compute.arch.tcgen05 import *
from layout import Coord, TensorLayout, TileTensor, coord, row_major
from layout.tensor_core_async import (
    tile_layout_k_major_typed,
)
from structured_kernels.kernel_common import _to_batched_3d
from structured_kernels.tile_types import (
    SMemTileArray2D,
    SMemTileArray2DRowMajor,
    swizzle_mode_to_bytes,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tensor_tile
from std.logger import Logger
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from internal_utils.fp8_utils import cast_saturating

from ....arch.sm100 import MmaOpSM100_SS
from ....utils import elementwise_epilogue_type

comptime logger = Logger()


@__name(
    t"matmul_sm100_blockwise_scaled_fp8_1d2d_{a_type}_{b_type}_{c_type}",
)
def matmul_sm100_blockwise_scaled_fp8_1d2d_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_layout: TensorLayout,
    c_layout: TensorLayout,
    a_scales_layout: TensorLayout,
    b_scales_layout: TensorLayout,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    a_scales_tile_rank: Int,
    a_scales_tile_shape: IndexList[a_scales_tile_rank],
    a_scales_desc_shape: IndexList[a_scales_tile_rank],
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: Int = 128,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    # B-scale N-direction block size; defaults to 128 to match the
    # original DeepSeek-style (n_g=k_g=128) callers. Set to a smaller
    # value (e.g. 64) when N-direction scale granularity is finer than
    # the kernel's BK.
    b_scaling_block_n: Int = 128,
](
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    c: TileTensor[mut=True, c_type, c_layout, MutAnyOrigin],
    a_scales_tma_op: TMATensorTile[
        a_scales_type,
        a_scales_tile_rank,
        a_scales_tile_shape,
        a_scales_desc_shape,
    ],
    b_scales: TileTensor[b_scales_type, b_scales_layout, ImmutAnyOrigin],
    num_iters: Int32,
):
    var _num_iters = Int(num_iters)
    comptime assert transpose_b, "Only support transposed B"
    comptime assert num_threads == 128

    comptime accum_type = get_accum_type[a_type]()

    comptime assert (
        b_scales_type == a_scales_type == accum_type == .float32
    ), "Only support float32 for a_scales and b_scales"

    comptime N = c_layout.static_shape[1]
    comptime K = a_layout.static_shape[2]
    var M = c.dim[0]()

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = BM // MMA_M
    comptime num_n_mmas = BN // MMA_N
    comptime num_k_mmas = BK // MMA_K

    comptime assert N % BN == 0, "N must be divisible by BN"
    comptime assert (
        BN <= BK or gcd(BN, BK) == BN - BK
    ), "BN <= BK or gcd(BN, BK) == BN - BK"

    # make sure A and B scales are compatible
    comptime b_scales_n = b_scales_layout.static_shape[0]
    comptime b_scales_k = b_scales_layout.static_shape[1]
    comptime a_scales_k = a_scales_layout.static_shape[1]

    # B-scale N-direction block size is supplied by the caller via
    # `b_scaling_block_n` (defaults to 128). It cannot be derived from
    # `N // b_scales_n` because N is not required to be a multiple of
    # the scale block size (the last N-block can be partial).
    comptime B_SCALING_BLOCK_N = b_scaling_block_n
    comptime B_SCALING_BLOCK_K = K // b_scales_k
    comptime A_SCALING_BLOCK = K // a_scales_k

    var a_smem = external_memory[
        Scalar[a_type],
        address_space=.SHARED,
        alignment=128,
        name="tmem_test_dynamic_shared_memory",
    ]().as_unsafe_any_origin()

    comptime a_size = tile_layout_k_major_typed[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ].static_product
    comptime b_size = tile_layout_k_major_typed[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ].static_product
    comptime a_scales_size = BM

    comptime assert (
        (a_size * size_of[a_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (b_size * size_of[b_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (a_scales_size * size_of[a_scales_type]()) % 16
    ) == 0, "preserve alignment"

    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()
    var a_scales_smem = (b_smem + b_size).bitcast[Scalar[a_scales_type]]()

    # 3D view of the same shared memory tile are used for TMA loads
    # 2D view of the same shared memory tile are used for MMA operations

    # TileTensor views for both TMA producer and MMA consumer. The size guards
    # keep the TileTensor array's explicit dimensions in lockstep with the
    # TMA/MMA swizzled tile layouts used for pointer partitioning.
    comptime ASMemTiles = SMemTileArray2D[
        a_type,
        BM,
        BK,
        1,
        swizzle_mode_to_bytes[a_swizzle],
    ]
    comptime BSMemTiles = SMemTileArray2D[
        b_type,
        BN,
        BK,
        1,
        swizzle_mode_to_bytes[b_swizzle],
    ]
    comptime assert ASMemTiles.tile_size == a_size, "A SMEM tile size mismatch"
    comptime assert BSMemTiles.tile_size == b_size, "B SMEM tile size mismatch"
    var a_smem_tt = ASMemTiles(a_smem)
    var b_smem_tt = BSMemTiles(b_smem)
    var a_smem_tile = a_smem_tt[0]
    var b_smem_tile = b_smem_tt[0]

    var a_scales_smem_tiles = SMemTileArray2DRowMajor[
        a_scales_type, 1, BM, 1, 128
    ](a_scales_smem)
    var a_scales_smem_tile_2D_view = a_scales_smem_tiles[0]
    var a_scales_smem_tile_3D_view = _to_batched_3d(a_scales_smem_tile_2D_view)

    var ptr_tmem_addr = (a_scales_smem + a_scales_size).bitcast[UInt32]()

    comptime a_expected_bytes = a_size * size_of[a_type]()
    comptime b_expected_bytes = b_size * size_of[b_type]()
    comptime a_scales_expected_bytes = a_scales_size * size_of[a_scales_type]()
    comptime expected_bytes = a_expected_bytes + b_expected_bytes + a_scales_expected_bytes

    var tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    var mma_mbar = tma_mbar + 1

    var warp_id = get_warp_id()
    var elect_one_warp = warp_id == 0
    var elect_one_thread = elect_one_warp and elect_one_sync()

    if elect_one_thread:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    comptime max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # wait for tensor memory to be allocated
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=1,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    # final results accumulator regs for C
    comptime c_frag_size = MMA_M * MMA_N // num_threads
    var c_frag = Array[Scalar[accum_type], c_frag_size](
        fill=Scalar[accum_type](0)
    )

    # temporary accumulators for TMEM loads
    comptime total_repeat = BN // 8
    comptime repeat = 1  # a higher repeat will probably get us better performance, but it will increase register pressure
    comptime temp_cfrags_size = 4 * repeat

    comptime assert (
        total_repeat % repeat == 0
    ), "total_repeat must be divisible by repeat"
    var c_frag_temp: Array[Scalar[accum_type], temp_cfrags_size]

    for k_iter in range(_num_iters):
        if elect_one_thread:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

            a_tma_op.async_copy_3d(
                a_smem_tile,
                tma_mbar[0],
                (
                    k_iter * BK,
                    block_idx.y * BM,
                    block_idx.z,
                ),
            )

            a_scales_tma_op.async_copy_3d(
                a_scales_smem_tile_3D_view,
                tma_mbar[0],
                (
                    block_idx.y * BM,
                    k_iter,
                    block_idx.z,
                ),
            )

            b_tma_op.async_copy_3d(
                b_smem_tile,
                tma_mbar[0],
                (
                    k_iter * BK,
                    block_idx.x * BN,
                    block_idx.z,
                ) if transpose_b else (
                    block_idx.x * BN,
                    k_iter * BK,
                    block_idx.z,
                ),
            )

        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1  # flips between 0 and 1 representing the pipeline stage

        # Preload a_scales from SMEM into registers while all threads
        # are synchronized (right after tma_mbar.wait). This eliminates
        # the SMEM read in the scaling loop below, preventing a race
        # where the next iteration's TMA loads could overwrite a_scales
        # in SMEM while slower warps are still reading it.
        var m_offset = warp_id * 16 + ufloordiv(lane_id(), 4)
        var a_scale_0 = a_scales_smem_tile_2D_view[0, m_offset]
        var a_scale_1 = a_scales_smem_tile_2D_view[0, m_offset + 8]

        if elect_one_thread:
            mma_op.mma(
                a_smem_tile,
                b_smem_tile,
                tmem_addr,
                init_c=(True),  # Initialize C on first iteration
            )

            mma_op.commit(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

        # Ensure all warps have finished reading a_scales from SMEM
        # (preloaded above) before the scaling loop, where warp divergence
        # could let the fastest warp loop back and start new TMA loads
        # that overwrite a_scales_smem. This barrier is effectively free
        # because all warps just converged at mma_mbar.wait().
        barrier()

        comptime for ld_iter in range(total_repeat // repeat):
            c_frag_temp = tcgen05_ld[
                datapaths=16,
                bits=256,
                repeat=repeat,
                dtype=accum_type,
                pack=False,
                width=temp_cfrags_size,
            ](tmem_addr + UInt32(ld_iter * 8 * repeat))
            tcgen05_load_wait()  # wait for the load to finish

            var b_scale: Scalar[b_scales_type]

            # N-direction scale block size is independent of BK_kernel;
            # derive it from the b_scales layout (set by the caller).
            comptime if BN != B_SCALING_BLOCK_N:
                var global_n = block_idx.x * BN

                var begin_n = min(
                    BN, B_SCALING_BLOCK_N - umod(global_n, B_SCALING_BLOCK_N)
                )
                comptime end_n = BN  # if N % BN !=0 then it should be  min(BN, N - block_idx.x * BN)

                var idx0 = ufloordiv(global_n, B_SCALING_BLOCK_N)
                var next_n = begin_n if begin_n < end_n else BN

                if ld_iter < (next_n // 8):
                    b_scale = rebind[Scalar[b_scales_type]](
                        b_scales[idx0, k_iter]
                    )
                else:
                    b_scale = rebind[Scalar[b_scales_type]](
                        b_scales[idx0 + 1, k_iter]
                    )

            else:
                b_scale = rebind[Scalar[b_scales_type]](
                    b_scales[block_idx.x, k_iter]
                )

            # Use preloaded a_scales from registers (not SMEM)
            comptime for j in range(temp_cfrags_size // 2):
                var a_scale = a_scale_0 if j % 2 == 0 else a_scale_1

                var scale = rebind[Scalar[accum_type]](a_scale) * rebind[
                    Scalar[accum_type]
                ](b_scale)
                var scale_pair = SIMD[accum_type, 2](scale)

                comptime idx = ld_iter * temp_cfrags_size + 2 * j
                var c_pair = SIMD[accum_type, 2](c_frag[idx], c_frag[idx + 1])
                var t_pair = SIMD[accum_type, 2](
                    c_frag_temp[2 * j], c_frag_temp[2 * j + 1]
                )
                var result = c_pair + t_pair * scale_pair
                c_frag[idx] = result[0]
                c_frag[idx + 1] = result[1]

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    comptime num_warps = num_threads // WARP_SIZE
    warp_id = get_warp_id()

    var ctile, ctile_coords, _ = c.tile_with_offset[BM, BN](
        Coord(block_idx.y, block_idx.x)
    )
    comptime c_coord_type = type_of(ctile_coords)

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            var c_gmem_warp_tile, _c_gmem_warp_tile_coords, _ = (
                ctile.tile_with_offset[MMA_M // num_warps, MMA_N](
                    Coord(4 * m_mma + warp_id, n_mma)
                )
            )
            var c_gmem_warp_tile_coords = ctile_coords + rebind[c_coord_type](
                _c_gmem_warp_tile_coords
            )

            var c_gmem_frag, _c_gmem_frag_coords, _ = (
                c_gmem_warp_tile.vectorize[1, 2]().distribute_with_offset[
                    row_major[8, 4]()
                ](lane_id())
            )
            var new_c_gmem_frag_coords = rebind[c_coord_type](
                _c_gmem_frag_coords
            )
            new_c_gmem_frag_coords[1] *= 2
            var c_gmem_frag_coords = (
                c_gmem_warp_tile_coords + new_c_gmem_frag_coords
            )

            comptime num_vecs_m = type_of(c_gmem_frag).static_shape[0]
            comptime num_vecs_n = type_of(c_gmem_frag).static_shape[1]
            comptime c_row_stride = type_of(c).static_stride[0]
            comptime assert (
                type_of(c).static_stride[1] == 1
            ), "the last dim's stride must be 1"

            comptime for n_vec in range(num_vecs_n):
                comptime for m_vec in range(num_vecs_m):
                    comptime i_vec = n_vec * num_vecs_m + m_vec
                    var dst_idx = Int(c_gmem_frag.layout(coord[m_vec, n_vec]))
                    var dst_m_offset, dst_n_offset = divmod(
                        dst_idx, c_row_stride
                    )
                    var m = UInt32(c_gmem_frag_coords[0] + dst_m_offset)
                    var n = UInt32(c_gmem_frag_coords[1] + dst_n_offset)

                    if m < UInt32(M) and n < UInt32(N):
                        # Saturating: a plain cast of an out-of-range f32 to
                        # FP8 yields NaN on NVIDIA (the `cvt` is not
                        # `satfinite`). No-op for non-FP8 `c_type`.
                        var c_mn = cast_saturating[c_type](
                            SIMD[accum_type, 2](
                                c_frag[2 * i_vec],
                                c_frag[2 * i_vec + 1],
                            )
                        )

                        comptime if elementwise_lambda_fn:
                            comptime alignment = align_of[SIMD[c_type, 2]]()
                            comptime epilogue = elementwise_lambda_fn.value()
                            epilogue[alignment=alignment](
                                (Int(m), Int(n)), c_mn
                            )
                        else:
                            c_gmem_frag[m_vec, n_vec] = c_mn


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(a_scales_tma_op, `nvvm.grid_constant`)
@__name(
    t"matmul_sm100_blockwise_scaled_fp8_1d2d_wrapper_{a_type}_{b_type}_{c_type}",
)
def matmul_sm100_blockwise_scaled_fp8_1d2d_wrapper[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_layout: TensorLayout,
    c_layout: TensorLayout,
    a_scales_layout: TensorLayout,
    b_scales_layout: TensorLayout,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    a_scales_tile_rank: Int,
    a_scales_tile_shape: IndexList[a_scales_tile_rank],
    a_scales_desc_shape: IndexList[a_scales_tile_rank],
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: Int = 128,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    b_scaling_block_n: Int = 128,
](
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    c: TileTensor[mut=True, c_type, c_layout, MutAnyOrigin],
    a_scales_tma_op: TMATensorTile[
        a_scales_type,
        a_scales_tile_rank,
        a_scales_tile_shape,
        a_scales_desc_shape,
    ],
    b_scales: TileTensor[b_scales_type, b_scales_layout, ImmutAnyOrigin],
    num_iters: Int32,
):
    var _num_iters2 = Int(
        num_iters
    )  # NOTE: This wrapper is necessary because batched blockwise scaling has a wrapper kernel
    # for allocating matrices across the z index that kernel calls the function
    # `matmul_sm100_blockwise_scaled_fp8_1d2d_kernel` as well. That function requires the decroators
    # to not be present on the function so we moved it to this wrapper.

    matmul_sm100_blockwise_scaled_fp8_1d2d_kernel[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        a_layout,
        c_layout,
        a_scales_layout,
        b_scales_layout,
        a_tile_rank,
        a_tile_shape,
        a_desc_shape,
        b_tile_rank,
        b_tile_shape,
        b_desc_shape,
        a_scales_tile_rank,
        a_scales_tile_shape,
        a_scales_desc_shape,
        block_tile_shape,
        mma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=num_threads,
        elementwise_lambda_fn=elementwise_lambda_fn,
        b_scaling_block_n=b_scaling_block_n,
    ](
        a_tma_op,
        b_tma_op,
        c,
        a_scales_tma_op,
        b_scales,
        num_iters,
    )


def matmul_sm100_blockwise_scaled_fp8[
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor,
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    ctx: DeviceContext,
) raises:
    """
    Enqueues a blockwise scaled FP8 GEMM on SM100 with 1D A-scales and 2D
    B-scales onto the supplied device context.

    Validates the operand dtypes, ranks, and scale granularities, builds the
    TMA tile descriptors for A, B, and A-scales, and launches the
    `matmul_sm100_blockwise_scaled_fp8_1d2d_wrapper` kernel over a
    two-dimensional grid covering the M and N dimensions.

    Parameters:
        transpose_b: Whether B is loaded transposed; must be `True` (only
            the transposed-B path is supported).
        umma_shape: 3D MMA instruction shape `(MMA_M, MMA_N, MMA_K)` used by
            the TCgen05 tensor core operations.
        block_tile_shape: 3D block tile shape `(BM, BN, BK)` giving the
            per-CTA tile dimensions along the M, N, and K axes. `BK` must
            be 64 or 128.
        a_swizzle: TMA swizzle mode applied to A shared memory loads
            (defaults to `TensorMapSwizzle.SWIZZLE_128B`).
        b_swizzle: TMA swizzle mode applied to B shared memory loads
            (defaults to `TensorMapSwizzle.SWIZZLE_128B`).
        elementwise_lambda_fn: Optional elementwise epilogue lambda applied
            to each output element of `c` in place of a direct store
            (defaults to `None`).

    Args:
        c: Rank-2 output `TileTensor` accumulating the scaled GEMM result.
        a: Rank-2 `TileTensor` of A operands with `float8_e4m3fn` elements.
        b: Rank-2 `TileTensor` of B operands with `float8_e4m3fn` elements.
        a_scales: Rank-2 `TileTensor` of A scale factors with `float32`
            elements and 1D scaling granularity along K.
        b_scales: Rank-2 `TileTensor` of B scale factors with `float32`
            elements and 2D scaling granularity along N and K.
        ctx: Device context used to enqueue the kernel launch.
    """
    comptime assert transpose_b, "Only support transposed B"

    comptime a_type = type_of(a).dtype
    comptime b_type = type_of(b).dtype
    comptime c_type = type_of(c).dtype
    comptime a_scales_type = type_of(a_scales).dtype
    comptime b_scales_type = type_of(b_scales).dtype

    comptime assert (
        a_type == b_type and a_type == .float8_e4m3fn
    ), "Only support float8_e4m3fn"

    comptime assert (
        a_scales_type == b_scales_type and a_scales_type == .float32
    ), "Only support float32 for scales"

    comptime assert (
        type_of(a).LayoutType.rank == 2
        and type_of(b).LayoutType.rank == 2
        and type_of(c).LayoutType.rank == 2
        and type_of(a_scales).LayoutType.rank == 2
        and type_of(b_scales).LayoutType.rank == 2
    ), (
        "basic blockwise FP8 expects rank-2 A, B, C, A-scales, and B-scales"
        " TileTensors"
    )

    var a_3D = _to_batched_3d(a)
    var b_3D = _to_batched_3d(b)
    var a_scales_3D = _to_batched_3d(a_scales)

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime assert BK in (
        64,
        128,
    ), "blockwise scaled fp8 only supports BK in (64, 128)"

    var M = c.dim[0]()
    var N = c.dim[1]()
    var K = a_3D.dim[2]()

    var a_scales_dim0 = a_scales_3D.dim[1]()
    var a_scales_dim1 = a_scales_3D.dim[2]()
    var b_scales_dim1 = b_scales.dim[1]()

    # The K-direction scale granularity is fixed at BK
    # (k_scale_granularity == BK). The N-direction granularity may be
    # finer and is independent of BK_kernel.
    if (
        Int(a_scales_dim0) != Int(b_scales_dim1)
        or Int(K) % Int(a_scales_dim0) != 0
        or (Int(K) // Int(a_scales_dim0)) != BK
    ):
        raise Error(
            "a_scales_3D.dim[1]() must equal b_scales.dim[1](), K must be"
            " divisible by a_scales.dim(1), and (K // a_scales.dim(1)) must"
            " equal BK."
        )

    var padding_size = 16 // size_of[a_scales_type]()
    if Int(a_scales_dim1) % padding_size != 0:
        raise Error(
            "a_scales_3D.dim[2]() must be divisible by 16 bytes. This is"
            " required by NVIDIA SM90+ TMA instructions!"
        )

    logger.info(
        "Executing Basic 1D2D Blockwise Scaled FP8 GEMM (BLOCK_SCALE_SIZE ="
        " 128)"
    )
    logger.info("Problem Shape: MNK=[", M, ", ", N, ", ", K, "]", sep="")
    logger.info(
        "A Scales Shape: [",
        a_scales_3D.dim[1](),
        ", ",
        a_scales_3D.dim[2](),
        "]",
        sep="",
    )
    logger.info(
        "B Scales Shape: [",
        b_scales.dim[0](),
        ", ",
        b_scales.dim[1](),
        "]",
        sep="",
    )

    var a_tma_op = create_tensor_tile[
        Index(1, BM, BK),
        swizzle_mode=a_swizzle,
    ](ctx, a_3D)

    comptime b_tile_shape = Index(1, BN, BK) if transpose_b else Index(
        1, BK, BN
    )

    var b_tma_op = create_tensor_tile[
        b_tile_shape,
        swizzle_mode=b_swizzle,
    ](ctx, b_3D)

    var a_scales_tma_op = create_tensor_tile[
        Index(1, 1, BM),
        __desc_shape=Index(1, 1, BM),
    ](ctx, a_scales_3D)
    # NOTE: desc shape must be specified otherwise a constraint fails

    comptime smem_use = (
        BM * size_of[a_type]() + BN * size_of[b_type]()
    ) * BK + 24 + size_of[a_scales_type]() * BM

    comptime block_dim = 128

    var c_kernel = c.as_unsafe_any_origin()
    var b_scales_kernel = b_scales.as_immut().as_unsafe_any_origin()

    comptime kernel = matmul_sm100_blockwise_scaled_fp8_1d2d_wrapper[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        type_of(a_3D).LayoutType,
        type_of(c_kernel).LayoutType,
        type_of(a_scales_3D).LayoutType,
        type_of(b_scales_kernel).LayoutType,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        type_of(a_scales_tma_op).rank,
        type_of(a_scales_tma_op).tile_shape,
        type_of(a_scales_tma_op).desc_shape,
        block_tile_shape,
        umma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_kernel,
        a_scales_tma_op,
        b_scales_kernel,
        Int32(ceildiv(Int(K), BK)),
        grid_dim=(ceildiv(Int(N), BN), ceildiv(Int(M), BM)),
        block_dim=(block_dim),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )
