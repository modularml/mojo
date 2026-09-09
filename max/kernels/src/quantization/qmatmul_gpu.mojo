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
"""Provides GPU kernels for block-wise quantized int4 matrix multiplication.

Computes `C = A @ B` on the GPU, where `B` holds 4-bit quantized integer
weights packed eight values per 32-bit word and `A` and `C` are `bfloat16`.
The weights are dequantized in shared memory and fed through a multistage,
software-pipelined tensor-core GEMM.

## Limitations

- Requires an NVIDIA GPU target.
- `B` must be repacked before the matmul: use `gpu_qint4_repack_Q4_0` for GGUF
  Q4_0 weights or `gpu_qint4_repack_GPTQ` for GPTQ weights (with optional
  activation-order permutation).
"""

from std.math import ceildiv
from std.math.uutils import umod, ufloordiv, udivmod, uceildiv

from std.sys import align_of, is_nvidia_gpu, simd_width_of, size_of

from std.bit import log2_floor
from std.collections import OptionalReg
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.info import is_gpu
from max.gpu.memory import (
    async_copy,
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from layout import (
    IntTuple,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    lt_to_tt,
    lt_to_tt_idx,
    row_major,
)
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensorIter,
    copy_dram_to_sram,
    copy_local_to_dram,
    copy_local_to_shared,
    copy_sram_to_dram,
)
from layout.swizzle import Swizzle, make_ldmatrix_swizzle, make_swizzle
from layout.tile_io import GenericToSharedAsyncTileCopier
from layout.tensor_core import TensorCore, get_fragment_size, get_mma_shape
from linalg.matmul.gpu._multistage_gemm_gpu import warp_split_k_reduction
from linalg.utils import GemmShape, elementwise_epilogue_type
from linalg.utils_gpu import MatmulConfig, block_swizzle
from std.memory.unsafe import bitcast


from std.utils.index import Index, StaticTuple
from std.utils.numerics import get_accum_type


@always_inline
@doc_hidden
def args_to_tuple[swap: Bool](arg_0: Int, arg_1: Int) -> Tuple[Int, Int]:
    """Returns the two integer arguments as a tuple, swapping their order when `swap` is set.

    Parameters:
        swap: Whether to swap the order of the two arguments.

    Args:
        arg_0: The first integer argument.
        arg_1: The second integer argument.

    Returns:
        A tuple of the two arguments in the requested order.
    """
    comptime if swap:
        return (arg_1, arg_0)
    else:
        return (arg_0, arg_1)


@always_inline
@doc_hidden
def multistage_mma_q[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    transpose_b: Bool,
    group_size: Int,
    pack_factor: Int,
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    a_smem_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    b_smem_layout: Layout,
    scales_type: DType,
    scales_layout: Layout,
    scales_smem_layout: Layout,
    # Hack:
    /,
    *,
    swizzle_a: Bool = True,
    static_num_iters: Int = UNKNOWN_VALUE,
    prefetch_init: Bool = True,
    continue_prefetch_b: Bool = False,
    transpose_b_next: Bool = False,
    b_next_gmem_layout: Layout = Layout(),
    b_next_smem_layout: Layout = Layout(),
    next_op_b_iter_alignment: Int = align_of[b_type](),
](
    c: LayoutTensor[mut=True, c_type, c_layout, address_space=.LOCAL, ...],
    a_iter_arg: LayoutTensorIter[_, a_layout, ...],
    b_iter_arg: LayoutTensorIter[b_type, b_layout, ...],
    a_smem_iter_arg: LayoutTensorIter[
        mut=True,
        a_type,
        a_smem_layout,
        address_space=.SHARED,
        ...,
    ],
    mut b_smem_iter: LayoutTensorIter[
        mut=True,
        b_type,
        b_smem_layout,
        address_space=.SHARED,
        ...,
    ],
    scales_smem_iter_arg: LayoutTensorIter[
        scales_type,
        scales_smem_layout,
        address_space=.SHARED,
        ...,
    ],
    scales_iter_arg: LayoutTensorIter[scales_type, scales_layout, ...],
    num_iters: Int,
    /,
    *,
    num_b_rows: Optional[Int] = None,
):
    """Performs the multi-stage tensor-core MMA loop for a quantized matrix multiplication tile.

    Iterates over the K dimension, prefetching A, B, and per-group scale
    tiles into shared memory across `num_pipeline_stages` stages while issuing
    tensor-core MMA instructions, with optional swizzling of the A-tile copy.

    Parameters:
        BM: The block tile size along the M dimension.
        BN: The block tile size along the N dimension.
        BK: The block tile size along the K dimension.
        WM: The warp tile size along the M dimension.
        WN: The warp tile size along the N dimension.
        num_threads: The number of threads per thread block.
        num_pipeline_stages: The number of software-pipelined prefetch stages.
        transpose_b: Whether the B matrix is stored transposed.
        group_size: The number of K elements sharing a single scale.
        pack_factor: The number of 4-bit elements packed into one `uint32`.
        c_type: The dtype of the output accumulator tile.
        c_layout: The layout of the output accumulator tile.
        a_type: The dtype of the A matrix elements.
        a_layout: The layout of the A matrix in global memory.
        a_smem_layout: The layout of the A tile in shared memory.
        b_type: The dtype of the unpacked B matrix elements.
        b_layout: The layout of the B matrix in global memory.
        b_smem_layout: The layout of the B tile in shared memory.
        scales_type: The dtype of the per-group scales.
        scales_layout: The layout of the scales in global memory.
        scales_smem_layout: The layout of the scales tile in shared memory.
        swizzle_a: Whether to apply an ldmatrix swizzle to the A-tile copy.
        static_num_iters: The compile-time-known iteration count, if known.
        prefetch_init: Whether to issue the initial prefetch of leading stages.
        continue_prefetch_b: Whether to continue prefetching B tiles across iterations.
        transpose_b_next: Whether the next op's B matrix is stored transposed.
        b_next_gmem_layout: The global-memory layout of the next op's B matrix.
        b_next_smem_layout: The shared-memory layout of the next op's B matrix.
        next_op_b_iter_alignment: The required alignment for the next op's B iterator.

    Args:
        c: The local accumulator tile receiving the MMA results.
        a_iter_arg: The global-memory iterator over A tiles.
        b_iter_arg: The global-memory iterator over B tiles.
        a_smem_iter_arg: The shared-memory iterator over A prefetch buffers.
        b_smem_iter: The shared-memory iterator over B prefetch buffers.
        scales_smem_iter_arg: The shared-memory iterator over scale prefetch buffers.
        scales_iter_arg: The global-memory iterator over scale tiles.
        num_iters: The number of K-tile iterations to execute.
        num_b_rows: The runtime row count of the B matrix, if known.
    """
    comptime simd_size = simd_width_of[a_type]()
    comptime simd_b_size = simd_width_of[b_type]()
    comptime num_scales_stages = ceildiv(
        (num_pipeline_stages - 1) * BK, group_size
    ) + 1
    comptime repack_tile = Index(64, 16)

    var tid = UInt32(umod(thread_idx.x, num_threads))
    var warp_id, lane_id = divmod(tid, UInt32(WARP_SIZE))

    comptime num_warps_m = BM // WM
    comptime num_warps_n = BN // WN
    var warp_y, warp_x = divmod(warp_id, UInt32(num_warps_n))

    var a_iter = a_iter_arg
    var b_iter = b_iter_arg
    var scales_iter = scales_iter_arg
    var a_smem_iter = a_smem_iter_arg
    var scales_smem_iter = scales_smem_iter_arg
    # work around mut argument can't have default value.
    comptime async_copy_a_layout = Layout.row_major(
        num_threads * simd_size // BK, BK // simd_size
    )

    comptime async_copy_b_layout = Layout.row_major(
        ceildiv(num_threads * simd_b_size, b_smem_layout.stride[0].value()),
        num_threads
        // ceildiv(num_threads * simd_b_size, b_smem_layout.stride[0].value()),
    )
    comptime swizzle_b = transpose_b or b_type.is_half_float()

    comptime async_copy_scales_layout = Layout.row_major(1, WARP_SIZE)
    comptime async_copy_scales_veclen = BN // WARP_SIZE

    comptime smem_reg_scales_layout = Layout.row_major(8, 4)

    # Trait-based (`tile_layout.Layout`) thread layouts for the
    # `tile_io` async copier. These mirror `async_copy_a_layout` and
    # `async_copy_b_layout` above, but use `row_major[*Int]()` so the
    # resulting type unifies with `GenericToSharedAsyncTileCopier`'s
    # `thread_layout: Layout[shape_types, stride_types]` parameter
    # (the legacy `Layout.row_major(...)` builder produces a different
    # type that fails to unify).
    # TODO: Once the remaining sync `copy_*` callsites in this kernel
    # also migrate to `tile_io`, the legacy `async_copy_a_layout` /
    # `async_copy_b_layout` declarations above can be removed and these
    # become the single source of truth.
    comptime async_copy_a_layout_tt = row_major[
        num_threads * simd_size // BK, BK // simd_size
    ]()
    comptime async_copy_b_layout_tt = row_major[
        ceildiv(num_threads * simd_b_size, b_smem_layout.stride[0].value()),
        num_threads
        // ceildiv(num_threads * simd_b_size, b_smem_layout.stride[0].value()),
    ]()

    # Swizzle the async A-tile copy if requested. `make_ldmatrix_swizzle`
    # mirrors what `LayoutTensor.copy_dram_to_sram_async[swizzle=True]`
    # constructs internally (see layout_tensor.mojo:6751); reproducing it
    # here keeps the destination addressing identical across the
    # migration. The destination's leading-dim stride is `BK`.
    comptime async_swizzle_a = Optional[Swizzle](
        make_ldmatrix_swizzle[a_type, BK, log2_floor(simd_size)]()
    ) if swizzle_a else Optional[Swizzle]()

    # `a_iter_arg` / `b_iter_arg` have an inferred dtype, so call sites must
    # `.bitcast[a_type | b_type, target_address_space=.GENERIC]()`
    # the iterator's dereferenced tile before passing it in. The helpers
    # require concrete `a_type` / `b_type` arguments so the body can reference
    # `simd_size` / `simd_b_size` and the captured swizzle without
    # additional type unification.
    #
    # Both `dst` (SMEM) and `src` (DRAM) are converted with the same
    # `linear_idx_type` -- the parent DRAM tile's
    # `linear_idx_type`. Mixing `int32` SMEM offsets with `int64` DRAM
    # offsets in a single cp.async-driven distribute() emits cross-width
    # conversions (cvt.u64.u32 + mul.wide.u32 in the index computation)
    # that measurably degrade performance on B200. Picking the parent
    # tensor's index type for both sides gives uniform-width arithmetic
    # and matches the pre-migration codegen pattern (bfe.u64/shl.b64
    # when the parent has a runtime dim, bfe.u32/shl.b32 when the parent
    # is fully static and small).
    comptime a_idx_t = a_iter_arg.linear_idx_type
    comptime b_idx_t = b_iter_arg.linear_idx_type

    @always_inline
    @__parameter
    def _async_copy_a_tile(
        dst: LayoutTensor[mut=True, a_type, address_space=.SHARED, ...],
        src: LayoutTensor[a_type, address_space=.GENERIC, ...],
    ):
        GenericToSharedAsyncTileCopier[
            async_copy_a_layout_tt,
            swizzle=async_swizzle_a,
        ]().copy(
            lt_to_tt_idx[linear_idx_type=a_idx_t](dst).vectorize[
                1, simd_size
            ](),
            lt_to_tt_idx[linear_idx_type=a_idx_t](src).vectorize[
                1, simd_size
            ](),
        )

    @always_inline
    @__parameter
    def _async_copy_b_tile(
        dst: LayoutTensor[mut=True, b_type, address_space=.SHARED, ...],
        src: LayoutTensor[b_type, address_space=.GENERIC, ...],
    ):
        GenericToSharedAsyncTileCopier[async_copy_b_layout_tt]().copy(
            lt_to_tt_idx[linear_idx_type=b_idx_t](dst).vectorize[
                1, simd_b_size
            ](),
            lt_to_tt_idx[linear_idx_type=b_idx_t](src).vectorize[
                1, simd_b_size
            ](),
        )

    # Prefetch (num_pipeline_stages - 1) stages.
    comptime if prefetch_init:
        comptime for stage in range(num_pipeline_stages - 1):
            comptime if a_iter.address_space == .GENERIC:
                var a_smem_tile = a_smem_iter.next_unsafe(
                    a_smem_iter.linear_uint_type(stage)
                )[]

                _async_copy_a_tile(
                    a_smem_tile,
                    a_iter[].bitcast[a_type, target_address_space=.GENERIC](),
                )

                a_iter._incr()

            comptime if b_iter.address_space == .GENERIC:
                var b_smem_tile = b_smem_iter.next_unsafe(
                    b_smem_iter.linear_uint_type(stage)
                )[]

                _async_copy_b_tile(
                    b_smem_tile,
                    b_iter[].bitcast[b_type, target_address_space=.GENERIC](),
                )

                b_iter._incr()

            # Every group_size rows share a scale
            # Only load scales when necessary
            comptime if scales_iter.address_space == .GENERIC:
                if stage % (group_size // BK) == 0:
                    comptime scales_stage = stage // (group_size // BK)
                    var scales_smem_tile = scales_smem_iter.next_unsafe(
                        scales_smem_iter.linear_uint_type(scales_stage)
                    )[]

                    # We only need one warp for copying scales...
                    if tid < UInt32(WARP_SIZE):
                        var src_fragments = (
                            scales_iter[]
                            .bitcast[
                                scales_type,
                                target_address_space=.GENERIC,
                            ]()
                            .vectorize[1, async_copy_scales_veclen]()
                            .distribute[async_copy_scales_layout](Int(tid))
                        )
                        var dst_fragments = scales_smem_tile.vectorize[
                            1, async_copy_scales_veclen
                        ]().distribute[async_copy_scales_layout](Int(tid))

                        comptime element_size_bytes = size_of[
                            scales_type
                        ]() * async_copy_scales_veclen
                        async_copy[element_size_bytes](
                            src_fragments.ptr.address_space_cast[.GLOBAL](),
                            dst_fragments.ptr.address_space_cast[
                                .SHARED
                            ]().unsafe_mut_cast[True](),
                        )

                    scales_iter._incr()

            async_copy_commit_group()

        # Guard stage 0.
        async_copy_wait_group(Int32(num_pipeline_stages - 2))
        barrier()

    comptime mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_k_mmas = BK // MMA_K
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    comptime accum_type = get_accum_type[a_type]()
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime a_frag_size = frag_size[0]
    comptime b_frag_size = frag_size[1]
    comptime c_frag_size = frag_size[2]

    comptime a_reg_layout = Layout.row_major(2 * num_m_mmas, a_frag_size)
    # Register tiles.
    var a_reg_tiles = (
        LayoutTensor[
            mut=True,
            a_type,
            a_reg_layout,
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .split[2]()
    )
    comptime b_reg_layout = Layout.row_major(2 * num_n_mmas, b_frag_size)
    var b_reg_tiles = (
        LayoutTensor[
            mut=True,
            a_type,
            b_reg_layout,
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .vectorize[1, b_frag_size]()
        .split[2]()
    )

    var scales_reg_tiles = (
        LayoutTensor[
            mut=True,
            scales_type,
            Layout.row_major(num_n_mmas, 1),
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .vectorize[1, 1]()
    )

    var a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)

    comptime b_wtile_dim0 = WN // repack_tile[0] if transpose_b else (
        (BK * repack_tile[0]) // pack_factor
    )
    comptime b_wtile_dim1 = (
        (BK * repack_tile[0]) // pack_factor
    ) if transpose_b else WN // repack_tile[0]
    var b_wtile_coord0 = Int(warp_x) if transpose_b else 0
    var b_wtile_coord1 = 0 if transpose_b else Int(warp_x)
    var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
        b_wtile_coord0, b_wtile_coord1
    )
    var scales_warp_tile = scales_smem_iter[].tile[ceildiv(BK, group_size), WN](
        0, Int(warp_x)
    )

    var mma_op = TensorCore[accum_type, a_type, mma_shape, transpose_b]()

    comptime swizzle_a_pattern = make_ldmatrix_swizzle[
        a_type, a_warp_tile.stride[0]()
    ]() if swizzle_a else Optional[Swizzle]()

    mma_op.load_a[swizzle_a_pattern](
        a_warp_tile, a_reg_tiles[0].vectorize[1, a_frag_size]()
    )

    # load scales into regs
    # for thread 0-3, scales for col 0, 8, 16, ..., 56 are stored locally
    # thread 4-7 stores scales for col 1, 9, 17, ..., 57
    scales_reg_tiles.vectorize[simd_size, 1]().copy_from(
        scales_warp_tile.vectorize[1, simd_size]().distribute[
            smem_reg_scales_layout, axis=0
        ](Int(lane_id))
    )

    mma_op.load_b(b_warp_tile, b_reg_tiles[0], scales_reg_tiles, 0)

    for k_tile_id in range(num_iters):
        var a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)
        var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
            b_wtile_coord0,
            b_wtile_coord1,
        )

        # Perform prefetch registers and mma until current shared memory tile's
        # data has all been loaded to registers.
        comptime for k_mma in range(num_k_mmas):
            var current = k_mma % 2
            var next = (k_mma + 1) % 2

            if k_mma == num_k_mmas - 1:
                a_smem_iter._incr()
                b_smem_iter._incr()

                a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)
                b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
                    b_wtile_coord0, b_wtile_coord1
                )

                # prefetch scales into regs every (group_size) rows
                if (k_tile_id + 1) % (group_size // BK) == 0:
                    scales_smem_iter._incr()
                    scales_warp_tile = scales_smem_iter[].tile[
                        ceildiv(BK, group_size), WN
                    ](0, Int(warp_x))
                    scales_reg_tiles.vectorize[simd_size, 1]().copy_from(
                        scales_warp_tile.vectorize[1, simd_size]().distribute[
                            smem_reg_scales_layout, axis=0
                        ](Int(lane_id))
                    )

            mma_op.load_a[swizzle_a_pattern](
                a_warp_tile,
                a_reg_tiles[next].vectorize[1, a_frag_size](),
                (k_mma + 1) % num_k_mmas,
            )
            mma_op.load_b(
                b_warp_tile,
                b_reg_tiles[next],
                scales_reg_tiles,
                (k_mma + 1) % num_k_mmas,
            )

            mma_op.mma(
                a_reg_tiles[current].vectorize[1, a_frag_size](),
                b_reg_tiles[current],
                c.vectorize[1, c_frag_size](),
            )

            if k_mma + 2 == num_k_mmas:
                var prefetch_tile_id = k_tile_id + num_pipeline_stages - 1

                # Prefetch one k tile (if valid) from global memory to current
                # shared memory buffer.
                if prefetch_tile_id < num_iters:
                    comptime if a_iter.address_space == .GENERIC:
                        var a_smem_prefetch_tile = a_smem_iter.next_unsafe(
                            a_smem_iter.linear_uint_type(
                                num_pipeline_stages - 1
                            )
                        )[]

                        _async_copy_a_tile(
                            a_smem_prefetch_tile,
                            a_iter[].bitcast[
                                a_type,
                                target_address_space=.GENERIC,
                            ](),
                        )

                        a_iter._incr()

                    comptime if b_iter.address_space == .GENERIC:
                        var b_smem_prefetch_tile = b_smem_iter.next_unsafe(
                            b_smem_iter.linear_uint_type(
                                num_pipeline_stages - 1
                            )
                        )[]

                        _async_copy_b_tile(
                            b_smem_prefetch_tile,
                            b_iter[].bitcast[
                                b_type,
                                target_address_space=.GENERIC,
                            ](),
                        )

                        b_iter._incr()

                    comptime if scales_iter.address_space == .GENERIC:
                        # Every group_size rows share a scale
                        # Only load scales when necessary
                        if (k_tile_id + num_pipeline_stages - 1) % (
                            group_size // BK
                        ) == 0:
                            var scales_smem_tile = scales_smem_iter.next_unsafe(
                                scales_smem_iter.linear_uint_type(
                                    num_scales_stages - 1
                                )
                            )[]

                            # We only need one warp for copying scales...
                            if tid < UInt32(WARP_SIZE):
                                var src_fragments = (
                                    scales_iter[]
                                    .bitcast[
                                        scales_type,
                                        target_address_space=.GENERIC,
                                    ]()
                                    .vectorize[1, async_copy_scales_veclen]()
                                    .distribute[async_copy_scales_layout](
                                        Int(tid)
                                    )
                                )
                                var dst_fragments = scales_smem_tile.vectorize[
                                    1, async_copy_scales_veclen
                                ]().distribute[async_copy_scales_layout](
                                    Int(tid)
                                )

                                comptime element_size_bytes = size_of[
                                    scales_type
                                ]() * async_copy_scales_veclen
                                async_copy[element_size_bytes](
                                    src_fragments.ptr.address_space_cast[
                                        .GLOBAL
                                    ](),
                                    dst_fragments.ptr.address_space_cast[
                                        .SHARED
                                    ]().unsafe_mut_cast[True](),
                                )

                            scales_iter._incr()

                async_copy_commit_group()

                # Guard the next k tile's shared memory buffer.
                async_copy_wait_group(Int32(num_pipeline_stages - 2))
                barrier()


@__name(
    t"multistage_qgemm_{a_type}_{b_packed_type}_{c_type}_g{group_size}",
)
@doc_hidden
def multistage_qgemm_kernel[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_packed_type: DType,
    b_layout: Layout,
    group_size: Int,
    pack_factor: Int,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_packed_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[mut=True, c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[mut=False, a_type, a_layout, ImmutAnyOrigin],
    b_packed: LayoutTensor[mut=False, b_packed_type, b_layout, ImmutAnyOrigin],
):
    """Implements the GPU kernel for a multi-stage quantized GEMM with per-group scales.

    Unpacks the quantized B weights and scales from the packed buffer, sets up
    the circular shared-memory prefetch buffers, drives the `multistage_mma_q`
    MMA loop, performs the warp split-K reduction, and stores the accumulator
    to global memory with an optional elementwise epilogue.

    Parameters:
        c_type: The dtype of the output matrix.
        c_layout: The layout of the output matrix.
        a_type: The dtype of the A matrix elements.
        a_layout: The layout of the A matrix.
        b_packed_type: The dtype of the packed B weight buffer.
        b_layout: The layout of the packed B weight buffer.
        group_size: The number of K elements sharing a single scale.
        pack_factor: The number of 4-bit elements packed into one `uint32`.
        transpose_b: Whether the B matrix is stored transposed.
        config: The matmul configuration describing tile and warp shapes.
        elementwise_lambda_fn: An optional elementwise epilogue applied per output element.

    Args:
        c: The output accumulator matrix in global memory.
        a: The left-hand (activation) matrix in global memory.
        b_packed: The packed quantized weight buffer in global memory.
    """
    comptime assert (
        is_nvidia_gpu()
    ), "Quantized gemm only supports NVIDIA hardwares for now."
    comptime simd_size = simd_width_of[c_type]()

    comptime repack_tile = Index(64, 16)
    comptime group_bytes = group_size // 2 + 2

    var M = c.dim[0]()
    comptime N = Int(b_layout.shape[0])
    comptime K = Int(b_layout.shape[1]) // group_bytes * group_size

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime WM = config.warp_tile_shape[0]
    comptime WN = config.warp_tile_shape[1]
    comptime num_pipeline_stages = config.num_pipeline_stages

    comptime num_warps_m = config.num_warps_m()
    comptime num_warps_n = config.num_warps_n()
    comptime num_threads = config.num_threads()

    # Unpack quantized weights
    comptime scales_type = DType.bfloat16
    comptime b_type = DType.uint32
    comptime b_weight_layout = Layout.row_major(N // 64, K * 64 // pack_factor)
    var b = LayoutTensor[b_type, b_weight_layout](
        b_packed.ptr.bitcast[Scalar[b_type]](),
    )

    comptime b_scales_layout = Layout.row_major(K // group_size, N)
    var b_scales_ptr = b_packed.ptr + N * K // 2
    var scales = LayoutTensor[scales_type, b_scales_layout](
        b_scales_ptr.bitcast[Scalar[scales_type]](),
    )

    comptime num_warp_k_partitions = config.num_warp_k_partitions
    comptime num_threads_per_warp_k_part = num_threads // num_warp_k_partitions

    var tid = thread_idx.x
    var ln_id = lane_id()
    var warp_k_part_id = (
        ufloordiv(tid, num_threads_per_warp_k_part) if num_warp_k_partitions
        > 1 else 0
    )
    var warp_id = ufloordiv(umod(tid, num_threads_per_warp_k_part), WARP_SIZE)

    # Only apply block swizzling for half precision types.
    comptime swizzle_block = a_type.is_half_float() and b_type.is_half_float()

    var block_idx = block_swizzle(
        (block_idx.x, block_idx.y),
        (grid_dim.x, grid_dim.y),
    ) if swizzle_block else Index(block_idx.x, block_idx.y)

    # Coordinates of the current warp.
    var warp_y, warp_x = udivmod(warp_id, num_warps_n)

    # Prepare circular shared memory buffer for A and B.
    # Each pipeline stage has its own buffer.
    comptime alignment = align_of[SIMD[a_type, simd_size]]()
    var a_smem = external_memory[
        Scalar[a_type],
        address_space=.SHARED,
        alignment=alignment,
    ]()
    comptime a_smem_size = num_pipeline_stages * BM * BK

    comptime IteratorTypeA = LayoutTensorIter[
        a_type,
        Layout.row_major(BM, BK),
        _,
        address_space=.SHARED,
        alignment=alignment,
        circular=True,
    ]
    var a_smem_iter = IteratorTypeA(
        a_smem + warp_k_part_id * a_smem_size,
        IteratorTypeA.linear_uint_type(a_smem_size),
    )

    # There is one pre-allocated shared buffer. Explicitly offset B after at A's end.
    var b_smem = (a_smem + num_warp_k_partitions * a_smem_size).bitcast[
        Scalar[b_type]
    ]()
    comptime b_smem_size = num_pipeline_stages * BK * BN // pack_factor
    comptime BD_0 = BN // repack_tile[0]
    comptime BD_1 = (BK * repack_tile[0]) // pack_factor
    comptime b_smem_layout = Layout.row_major(BD_0, BD_1)

    comptime IteratorTypeB = LayoutTensorIter[
        b_type,
        b_smem_layout,
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var b_smem_iter = IteratorTypeB(
        b_smem + warp_k_part_id * b_smem_size,
        IteratorTypeB.linear_uint_type(b_smem_size),
    )

    # multiple stages may share the same scales
    comptime num_scales_stages = ceildiv(
        (num_pipeline_stages - 1) * BK, group_size
    ) + 1
    var scales_smem = (b_smem + num_warp_k_partitions * b_smem_size).bitcast[
        Scalar[scales_type]
    ]()
    comptime scales_smem_size = num_scales_stages * BN * ceildiv(BK, group_size)
    comptime scales_smem_layout = Layout.row_major(ceildiv(BK, group_size), BN)

    comptime IteratorTypeScales = LayoutTensorIter[
        scales_type,
        scales_smem_layout,
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var scales_smem_iter = IteratorTypeScales(
        scales_smem + warp_k_part_id * scales_smem_size,
        IteratorTypeScales.linear_uint_type(scales_smem_size),
    )

    # global memory iterator
    var bk_start: Int = (K // BK // num_warp_k_partitions) * warp_k_part_id
    var a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](block_idx[1], bk_start)
    var b_tile_coords = args_to_tuple[transpose_b](bk_start, block_idx[0])
    comptime b_tile_axis = 1 if transpose_b else 0
    var b_gmem_iter = b.tiled_iterator[BD_0, BD_1, axis=b_tile_axis](
        b_tile_coords[0], b_tile_coords[1]
    )
    comptime groups_per_iter = ceildiv(BK, group_size)
    var bk_scales_start: Int = (
        K // (groups_per_iter * group_size) // num_warp_k_partitions
    ) * warp_k_part_id
    var scales_gmem_iter = scales.tiled_iterator[
        ceildiv(BK, group_size), BN, axis=0
    ](bk_scales_start, block_idx[0])

    comptime mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    comptime accum_type = get_accum_type[a_type]()
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime c_frag_size = frag_size[2]

    comptime c_reg_layout = Layout.row_major(
        num_m_mmas * num_n_mmas, c_frag_size
    )
    var c_reg_tile = (
        LayoutTensor[
            mut=True,
            accum_type,
            c_reg_layout,
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    multistage_mma_q[
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads_per_warp_k_part,
        num_pipeline_stages,
        transpose_b,
        group_size,
        pack_factor,
    ](
        c_reg_tile,
        a_gmem_iter,
        b_gmem_iter,
        a_smem_iter,
        b_smem_iter,
        scales_smem_iter,
        scales_gmem_iter,
        ceildiv(K // num_warp_k_partitions, BK),
    )

    # reduce within the threadblock
    comptime if num_warp_k_partitions > 1:
        warp_split_k_reduction[
            BM,
            BN,
            num_threads_per_warp_k_part,
            num_warp_k_partitions,
        ](
            warp_k_part_id,
            c_reg_tile,
        )
        if warp_k_part_id > 0:
            return

    # Map global memory tile down to thread.
    var c_gmem_tile = c.tile[BM, BN](block_idx[1], block_idx[0])
    var c_gmem_warp_tile = c_gmem_tile.tile[WM, WN](warp_y, warp_x)

    @always_inline
    @__parameter
    def apply_epilogue():
        # This block is identical to the one used for f32 case
        # but putting this in a lambda function leads to test failures
        # TODO: Refactor to remove code duplication
        comptime assert (
            elementwise_lambda_fn is not None
        ), "elementwise_lambda_fn is not valid"
        comptime thread_layout = Layout.row_major(
            8, 4
        ) if is_nvidia_gpu() else Layout.row_major(4, 16)
        comptime dst_simd_width_x = 1 if is_nvidia_gpu() else 4
        comptime dst_simd_width_y = 2 if is_nvidia_gpu() else 1
        comptime src_simd_width_x = 1 if is_nvidia_gpu() else 1
        comptime src_simd_width_y = 2 if is_nvidia_gpu() else 4
        comptime epilogue = elementwise_lambda_fn.value()
        var c_gmem_frag = c_gmem_warp_tile.vectorize[
            dst_simd_width_x, dst_simd_width_y
        ]().distribute[thread_layout](ln_id)
        var c_reg_frag = c_reg_tile.vectorize[
            src_simd_width_x, src_simd_width_y
        ]().transpose()
        var thread_offset = c_gmem_frag.distance(c.ptr)

        comptime for i in range(type_of(c_gmem_frag).layout.size()):
            comptime src_idx = c_reg_frag.layout(i)
            comptime dst_static_idx = type_of(c_gmem_frag).layout(i)
            var dst_idx: Int

            comptime if c_gmem_frag.layout.all_dims_known():
                dst_idx = dst_static_idx
            else:
                dst_idx = Int(c_gmem_frag.runtime_layout(i))
            comptime alignment = align_of[SIMD[c_type, src_simd_width_y]]()
            var m = (Int(thread_offset) + dst_idx) // N
            var n = (Int(thread_offset) + dst_idx) % N
            if m < M and n < N:
                var vec = (c_reg_frag.ptr + src_idx).load[
                    width=src_simd_width_y,
                    alignment=align_of[SIMD[c_type, src_simd_width_y]](),
                ]()

                comptime if dst_simd_width_x == 1:
                    epilogue[alignment=alignment]((m, n), vec)
                else:
                    comptime for j in range(dst_simd_width_x):
                        if m + j < M:
                            epilogue[alignment=alignment](
                                (m + j, n), vec[j].cast[c_type]()
                            )

    # Store FP32 mma results to half precision buffer in global memory.
    # Each thread's fragment has 2x2 fp32 values. Casting to half float and
    # directly storing to global memory results in 2 4B writes. Following cutlass,
    # we stage the fragments in shared memory so that each thread can store 16B.

    comptime if c_type.is_half_float() and is_nvidia_gpu():
        comptime swizzle = make_swizzle[
            num_rows=MMA_M // 2, row_size=WN, access_size=MMA_N
        ]()

        var accum_smem_warp_tile = LayoutTensor[
            c_type,
            Layout.row_major(WM, WN),
            address_space=.SHARED,
        ](a_smem.bitcast[Scalar[c_type]]() + warp_id * WM * WN)

        copy_local_to_shared[
            thread_layout=Layout.row_major(8, 4),
            swizzle=swizzle,
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            c_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.
        comptime if elementwise_lambda_fn:
            comptime epilogue = elementwise_lambda_fn.value()
            comptime warp_layout = Layout.row_major(
                WARP_SIZE * simd_size // WN, WN // simd_size
            )
            var c_gmem_frag = c_gmem_warp_tile.vectorize[
                1, simd_size
            ]().distribute[warp_layout](thread_idx.x)
            var c_smem_frag = accum_smem_warp_tile.vectorize[
                1, simd_size
            ]().distribute[warp_layout](thread_idx.x)
            var thread_offset = c_gmem_frag.distance(c.ptr)
            comptime num_stores_per_thread = type_of(c_gmem_frag).layout.size()

            var c_smem_frag_offset = c_smem_frag.distance(
                accum_smem_warp_tile.ptr
            )

            comptime for i in range(num_stores_per_thread):
                comptime src_idx = type_of(c_smem_frag).layout(i)
                comptime src_idx_base = src_idx % swizzle.size()
                comptime src_idx_diff = src_idx - src_idx_base
                var swizzled_idx = swizzle(
                    c_smem_frag_offset
                    + type_of(c_smem_frag_offset)(src_idx_base)
                ) + type_of(c_smem_frag_offset)(src_idx_diff)

                comptime dst_static_idx = type_of(c_gmem_frag).layout(i)
                var dst_idx: Int

                comptime if c_gmem_frag.layout.all_dims_known():
                    dst_idx = dst_static_idx
                else:
                    dst_idx = Int(c_gmem_frag.runtime_layout(i))

                var m = (Int(thread_offset) + dst_idx) // N
                var n = (Int(thread_offset) + dst_idx) % N
                comptime alignment = align_of[SIMD[c_type, simd_size]]()
                if m < M and n < N:
                    epilogue[alignment=alignment](
                        (m, n),
                        accum_smem_warp_tile.ptr.load[
                            width=simd_size, alignment=alignment
                        ](swizzled_idx).cast[c_type](),
                    )
        else:
            copy_sram_to_dram[
                thread_layout=Layout.row_major(
                    WARP_SIZE * simd_size // WN, WN // simd_size
                ),
                swizzle=swizzle,
            ](
                c_gmem_warp_tile.vectorize[1, simd_size](),
                accum_smem_warp_tile.vectorize[1, simd_size](),
            )

    elif c_type.is_half_float() and not is_nvidia_gpu():
        comptime if elementwise_lambda_fn:
            apply_epilogue()

        else:
            var c_reg_tile_out = LayoutTensor[
                mut=True,
                c_type,
                c_reg_tile.layout,
                MutAnyOrigin,
                address_space=.LOCAL,
            ].stack_allocation()

            comptime for i in range(c_reg_tile.shape[0]()):
                comptime for j in range(c_reg_tile.shape[1]()):
                    c_reg_tile_out[i, j] = c_reg_tile[i, j].cast[c_type]()
            copy_local_to_dram[dst_thread_layout=Layout.row_major(4, 16)](
                c_gmem_warp_tile.vectorize[4, 1](),
                c_reg_tile_out.vectorize[1, 4](),
            )
    # Store FP32 results to FP32 buffer in global memory.
    else:
        comptime if elementwise_lambda_fn:
            apply_epilogue()
        else:
            comptime if is_nvidia_gpu():
                copy_local_to_dram[dst_thread_layout=Layout.row_major(8, 4)](
                    c_gmem_warp_tile.vectorize[1, 2](),
                    c_reg_tile.vectorize[1, 2]().transpose(),
                )
            else:
                copy_local_to_dram[dst_thread_layout=Layout.row_major(4, 16)](
                    c_gmem_warp_tile.vectorize[4, 1](),
                    c_reg_tile.vectorize[1, 4](),
                )


# For a 4-bit weight matrix of shape (64 * TN, 16 * TK), we first
# repack elements within each 64x16 tile, and in memory every 8
# 4-bit elements are packed in one uint32.
#              K
#       0       16      32
#      0+-------+-------+-----+
#       | 64x16 | 64x16 | ... |
#       | tile  | tile  |     |
# N   64+-------+-------+-----+   Matrix with 4-bit elements
#       | 64x16 | 64x16 | ... |
#       | tile  | tile  |     |
#    128+-------+-------+-----+
#
#
# Stored as uint32 tensor (64 * TN, 2 * TK):
#              K
#     0       2      4
#     0+------+------+-----+
#      | 64x2 | 64x2 | ... |
#      | tile | tile |     |
# N  64+------+------+-----+    uint32 Matrix
#      | 64x2 | 64x2 | ... |
#      | tile | tile |     |
#   128+------+------+-----+
#
# Elements within each tile are stored continuously
#                  K
#      0     1     2     3     4
#     0+-----+-----+-----+-----+
#      |  0  |  1  | 128 | 129 |
#     1+-----+-----+-----+-----+
#      |  2  |  3  | 130 | 131 |
#     2+-----+-----+-----+-----+
#      |  4  |  5  | 132 | 133 |
# N   3+-----+-----+-----+-----+
#      | ... | ... | ... | ... |
#    64+-----+-----+-----+-----+
#      |TK*  |TK*  | ... | ... |
#      |128  |128+1| ... | ... |
#    65+-----+-----+-----+-----+
#      |TK*  |TK*  | ... | ... |
#      |128+2|128+3| ... | ... |
#    65+-----+-----+-----+-----+
# This data layout can be expressed by UInt32 LayoutTensor
# with shape = IntTuple(IntTuple(64, TN),IntTuple(2, TK))
# and stride = IntTuple(IntTuple(2, TK * 128),IntTuple(1, 128))
@always_inline
@doc_hidden
def pack_Q_tile(input: SIMD[.uint8, 16]) -> SIMD[.uint32, 4]:
    """Packs sixteen bytes (thirty-two 4-bit weights, two per byte) into four `uint32` lanes for the repacked weight layout.

    Args:
        input: A SIMD vector of sixteen `uint8` values; both nibbles of each byte hold a 4-bit weight.

    Returns:
        A SIMD vector of four `uint32` values with the nibbles interleaved into the repacked layout.
    """
    # Q-tile is the smallest indivisible unit when performing gemm
    # operations with quantized matrices.

    var res: SIMD[.uint32, 4] = 0

    comptime for i in range(4):
        res[i] |= input[i * 4 + 0].cast[.uint32]() & 0x0F
        res[i] |= (input[i * 4 + 0].cast[.uint32]() & 0xF0) << 12
        res[i] |= (input[i * 4 + 1].cast[.uint32]() & 0x0F) << 4
        res[i] |= (input[i * 4 + 1].cast[.uint32]() & 0xF0) << 16

        res[i] |= (input[i * 4 + 2].cast[.uint32]() & 0x0F) << 8
        res[i] |= (input[i * 4 + 2].cast[.uint32]() & 0xF0) << 20
        res[i] |= (input[i * 4 + 3].cast[.uint32]() & 0x0F) << 12
        res[i] |= (input[i * 4 + 3].cast[.uint32]() & 0xF0) << 24

    return res


@always_inline
@doc_hidden
def unpack_4bit_int(val: SIMD[.uint32, _], idx: Int) -> UInt8:
    """Extracts a single 4-bit value from the packed `uint32` lane at the given nibble index.

    Args:
        val: The packed `uint32` SIMD value.
        idx: The nibble index of the 4-bit element to extract.

    Returns:
        The extracted 4-bit value as a `UInt8`.
    """
    var u32_val = rebind[UInt32](val)
    return (u32_val >> UInt32(idx * 4)).cast[.uint8]() & 0x0F


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
@__name(t"repack_Q4_0_for_sm8x_{scales_type}")
@doc_hidden
def repack_Q4_0_for_sm8x[
    q_layout: Layout,
    repack_layout: Layout,
    scales_type: DType,
](
    q_weight: LayoutTensor[mut=False, .uint8, q_layout, ImmutAnyOrigin],
    q_packed_weight: LayoutTensor[
        mut=True, .uint8, repack_layout, MutAnyOrigin
    ],
):
    """Repacks Q4_0 quantized weights into the tiled layout expected by the SM8x quantized GEMM kernel.

    Parameters:
        q_layout: The layout of the input Q4_0 weight buffer.
        repack_layout: The layout of the output repacked weight buffer.
        scales_type: The dtype to cast the per-group scales to.

    Args:
        q_weight: The input Q4_0 quantized weight tensor in global memory.
        q_packed_weight: The output repacked weight tensor in global memory.
    """
    comptime group_size = 32
    comptime group_bytes = size_of[DType.float16]() + (group_size // 2)
    comptime pack_factor = 8
    comptime repack_tile = Index(64, 16)
    comptime WARP_SIZE = 32
    comptime BN = 128
    comptime BK = 1024

    var tid = thread_idx.x
    var warp_id = ufloordiv(tid, WARP_SIZE)
    comptime num_warps_x = BN // repack_tile[0]
    var warp_x = umod(warp_id, num_warps_x)
    var warp_y = ufloordiv(warp_id, num_warps_x)
    var lane_id: Int = umod(tid, WARP_SIZE)
    var block_idx = Index(block_idx.x, block_idx.y)

    comptime N = Int(q_layout.shape[0])
    comptime K = Int(q_layout.shape[1]) // group_bytes * group_size

    comptime K_groups = K // group_size
    comptime BK_groups = BK // group_size

    comptime uint_K = K // pack_factor
    comptime uint_BK = BK // pack_factor

    @always_inline
    @__parameter
    def convert_bytes_to_bf16[
        scales_type: DType
    ](input_bytes: SIMD[.uint8, _]) -> Scalar[scales_type]:
        var f32_values = bitcast[.float16, 1](input_bytes).cast[.float32]()
        return bitcast[scales_type, 2](f32_values)[1]

    comptime repacked_b_layout = Layout(
        IntTuple(
            IntTuple(64, N // 64),
            IntTuple(2, uint_K // 2),
        ),
        IntTuple(
            IntTuple(2, 128 * (uint_K // 2)),
            IntTuple(1, 128),
        ),
    )
    var repack_weights = LayoutTensor[.uint32, repacked_b_layout](
        q_packed_weight.ptr.bitcast[UInt32](),
    )

    comptime b_scales_layout = Layout.row_major(K_groups, N)
    var b_scales_ptr = q_packed_weight.ptr + N * K // 2
    var repack_scales = LayoutTensor[scales_type, b_scales_layout](
        b_scales_ptr.bitcast[Scalar[scales_type]](),
    )

    # We keep 128x2 Q4_0 GGUF blocks in smem
    var smem = external_memory[
        UInt8,
        address_space=.SHARED,
        alignment=align_of[UInt8](),
    ]()
    var qb_smem = LayoutTensor[
        .uint8,
        Layout.row_major(BN, 2 * group_bytes),
        address_space=.SHARED,
    ](smem.bitcast[UInt8]())

    var q_gmem_tile = q_weight.tile[BN, BK_groups * group_bytes](
        block_idx[0], block_idx[1]
    )
    var q_gmem_iter = q_gmem_tile.tiled_iterator[BN, 2 * group_bytes, axis=1](
        0, 0
    )

    var repacked_gmem_tile = repack_weights.tile[BN, uint_BK](
        block_idx[0], block_idx[1]
    )
    var repacked_gemm_iter = repacked_gmem_tile.tiled_iterator[
        BN, 2 * group_size // pack_factor, axis=1
    ](0, 0)

    var scales_gmem_tile = repack_scales.tile[BK_groups, BN](
        block_idx[1], block_idx[0]
    )
    var scales_gmem_iter = scales_gmem_tile.tiled_iterator[2, BN, axis=0](0, 0)

    # We load 128x2 Q4_0 GGUF blocks to smem.
    # Each warp repacks 64x1 Q4_0 GGUF blocks, which are
    # 64x32 4-bit weights. We repack weights into 64x16
    # tiles for our quantized matmul kernel, so there are
    # two tile for each warp.
    # frag_0 stores frags of the first 64x16 tile,
    # frag_1 stores frags of the second,
    for i in range(ceildiv(BK_groups, 2)):
        barrier()
        copy_dram_to_sram[thread_layout=Layout.row_major(128, 1)](
            qb_smem.vectorize[1, 4](),
            q_gmem_iter[]
            .bitcast[.uint8, target_address_space=.GENERIC]()
            .vectorize[1, 4](),
        )
        q_gmem_iter._incr()
        barrier()
        var q_warp_tile = qb_smem.tile[repack_tile[0], group_bytes](
            warp_x, warp_y
        )

        if (BK_groups * block_idx[1] + i * 2 + warp_y) < K_groups:
            var frag_0: SIMD[.uint8, 16] = 0
            var frag_1: SIMD[.uint8, 16] = 0
            var raw_Q_tile = q_warp_tile.tile[repack_tile[0], group_bytes]()
            comptime thd_layout = Layout.row_major(8, 4)
            # The first 2 Bytes is the scale for this Q4_0 block
            # GGUF pack elements 0-15 in the lower 4-bit of the 16 Bytes,
            # and elements 16-31 in the higher 4-bit of the 16 Bytes.
            #
            # This gets elements 0, 1, 8, 9, 16, 17, 24, 25 for
            # thread 0.
            var thread_tile = (
                raw_Q_tile.slice[:, 2:]()
                .vectorize[1, 2]()
                .distribute[thd_layout](lane_id)
            )

            comptime for i_e in range(16):
                var val = thread_tile.load[2](i_e // 2, i_e % 2)
                frag_0[i_e] = (val[0] & 0x0F) | ((val[1] & 0x0F) << 4)
                frag_1[i_e] = ((val[0] & 0xF0) >> 4) | (val[1] & 0xF0)

            var repack_warp_tile = repacked_gemm_iter[].tile[
                64, group_size // pack_factor
            ](warp_x, warp_y)
            repack_warp_tile.vectorize[2, 2]().store(
                lane_id, 0, pack_Q_tile(frag_0)
            )
            repack_warp_tile.vectorize[2, 2]().store(
                lane_id, 1, pack_Q_tile(frag_1)
            )
            repacked_gemm_iter._incr()

            comptime scales_thread_layout = Layout(
                IntTuple(4, 8), IntTuple(16, 1)
            )
            var rt_scales_thread_layout = RuntimeLayout[
                scales_thread_layout,
                element_type=q_warp_tile.layout_int_type,
                linear_idx_type=q_warp_tile.linear_idx_type,
            ]()

            # cast scales to bf16 before storing back
            var scales_warp_tile = scales_gmem_iter[].tile[1, 64](
                warp_y, warp_x
            )

            scales_warp_tile[0, 2 * lane_id] = convert_bytes_to_bf16[
                scales_type
            ](
                q_warp_tile.vectorize[1, 2]()[
                    Int(rt_scales_thread_layout(lane_id)), 0
                ]
            )

            scales_warp_tile[0, 2 * lane_id + 1] = convert_bytes_to_bf16[
                scales_type
            ](
                q_warp_tile.vectorize[1, 2]()[
                    Int(rt_scales_thread_layout(lane_id)) + 8, 0
                ]
            )

            scales_gmem_iter._incr()


# Tensors of GPTQ format are stored in a non-transposed way.
# Assume the original transposed matrix is of shape [N, K], the qweight shard
# will be a uint32 matrix of shape [K // 8, N], and scales will be of shape
# [K_groups, N]. The input is a uint8 tensor of shape
# [K_groups * group_bytes, N].
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
@__name(t"repack_GPTQ_for_sm8x_{scales_type}_g{group_size}_{has_perm}")
@doc_hidden
def repack_GPTQ_for_sm8x[
    in_layout: Layout,
    out_layout: Layout,
    scales_type: DType,
    group_size: Int,
    has_perm: Bool,
    *,
    perm_layout: Layout = Layout(),
](
    in_tensor: LayoutTensor[mut=False, .uint8, in_layout, ImmutAnyOrigin],
    out_tensor: LayoutTensor[mut=True, .uint8, out_layout, MutAnyOrigin],
    perm_idx: LayoutTensor[mut=False, .int32, perm_layout, ImmutAnyOrigin],
):
    """Repacks GPTQ quantized weights into the tiled layout expected by the SM8x quantized GEMM kernel.

    Parameters:
        in_layout: The layout of the input GPTQ weight buffer.
        out_layout: The layout of the output repacked weight buffer.
        scales_type: The dtype to cast the per-group scales to.
        group_size: The number of K elements sharing a single scale.
        has_perm: Whether a permutation index is applied to the K dimension.
        perm_layout: The layout of the permutation index tensor.

    Args:
        in_tensor: The input GPTQ quantized weight tensor in global memory.
        out_tensor: The output repacked weight tensor in global memory.
        perm_idx: The permutation index tensor, or a null tensor when `has_perm` is False.
    """
    comptime raw_scales_type = DType.float16
    comptime weights_bytes_per_group = group_size // 2
    comptime group_bytes = size_of[DType.float16]() + weights_bytes_per_group
    comptime pack_factor = 8
    comptime repack_tile = Index(64, 16)
    comptime BN = 128
    comptime BK = 1024

    var tid = thread_idx.x
    var warp_id = ufloordiv(tid, WARP_SIZE)
    comptime num_warps_x = BN // repack_tile[0]
    var warp_x = umod(warp_id, num_warps_x)
    var warp_y = ufloordiv(warp_id, num_warps_x)
    var lane_id: Int = umod(tid, WARP_SIZE)
    var block_idx = Index(block_idx.x, block_idx.y)

    comptime N = Int(in_layout.shape[1])
    comptime K = Int(in_layout.shape[0]) // group_bytes * group_size

    comptime K_groups = K // group_size
    comptime BK_groups = BK // group_size

    comptime uint_K = K // pack_factor
    comptime uint_BK = BK // pack_factor

    @always_inline
    @__parameter
    def convert_bytes_to_bf16[
        scales_type: DType
    ](input_bytes: SIMD[raw_scales_type, _]) -> Scalar[scales_type]:
        var f32_values = bitcast[.float16, 1](input_bytes).cast[.float32]()
        return bitcast[scales_type, 2](f32_values)[1]

    # Define 4-bit weights and scales for the raw input
    comptime raw_weights_layout = Layout.row_major(uint_K, N)
    var raw_weights = LayoutTensor[.uint32, raw_weights_layout](
        in_tensor.ptr.bitcast[UInt32](),
    ).transpose()
    comptime raw_scales_layout = Layout.row_major(K_groups, N)
    var raw_scales_ptr = in_tensor.ptr + N * K // 2
    var raw_scales = LayoutTensor[raw_scales_type, raw_scales_layout](
        raw_scales_ptr.bitcast[Scalar[raw_scales_type]](),
    ).transpose()

    # Define 4-bit weights and scales for the repacked buffer
    comptime repacked_weights_layout = Layout(
        IntTuple(
            IntTuple(64, N // 64),
            IntTuple(2, uint_K // 2),
        ),
        IntTuple(
            IntTuple(2, 128 * (uint_K // 2)),
            IntTuple(1, 128),
        ),
    )
    var repack_weights = LayoutTensor[.uint32, repacked_weights_layout](
        out_tensor.ptr.bitcast[UInt32](),
    )
    comptime repacked_scales_layout = Layout.row_major(K_groups, N)
    var repacked_scales_ptr = out_tensor.ptr + N * K // 2
    var repack_scales = LayoutTensor[scales_type, repacked_scales_layout](
        repacked_scales_ptr.bitcast[Scalar[scales_type]](),
    )

    # We keep 128x2 GPTQ blocks in smem
    var smem = external_memory[
        UInt8,
        address_space=.SHARED,
        alignment=align_of[UInt8](),
    ]()
    var weights_smem = LayoutTensor[
        .uint8,
        Layout.row_major(BN, 2 * weights_bytes_per_group),
        address_space=.SHARED,
    ](smem.bitcast[UInt8]())
    var weights_smem_uint4 = LayoutTensor[
        .uint32,
        Layout.row_major(BN, 2 * group_size // pack_factor),
        address_space=.SHARED,
    ](smem.bitcast[UInt32]())

    var raw_weights_gmem_tile = raw_weights.tile[BN, uint_BK](
        block_idx[0], block_idx[1]
    )
    var raw_weights_gmem_iter = raw_weights_gmem_tile.tiled_iterator[
        BN, 2 * weights_bytes_per_group // size_of[DType.uint32](), axis=1
    ](0, 0)
    var raw_scales_gmem_tile = raw_scales.tile[BN, BK_groups](
        block_idx[0], block_idx[1]
    )
    var raw_scales_gmem_iter = raw_scales_gmem_tile.tiled_iterator[
        BN, 2, axis=1
    ](0, 0)

    var repacked_weights_gmem_tile = repack_weights.tile[BN, uint_BK](
        block_idx[0], block_idx[1]
    )
    var repacked_weights_gmem_iter = repacked_weights_gmem_tile.tiled_iterator[
        BN, 2 * group_size // pack_factor, axis=1
    ](0, 0)

    var repacked_scales_gmem_tile = repack_scales.tile[BK_groups, BN](
        block_idx[1], block_idx[0]
    )
    var repacked_scales_gmem_iter = repacked_scales_gmem_tile.tiled_iterator[
        2, BN, axis=0
    ](0, 0)

    # We load 128x2 GPTQ blocks to smem.
    # Each warp repacks 64x1 GPTQ blocks, which are
    # 64xgroup_size 4-bit weights. We repack weights into 64x16
    # tiles for our quantized matmul kernel, so there are
    # (group_size // 16) tiles for each warp.
    # repack_reg_tile[0] stores frags of the one 64x16 tile,
    for i in range(ceildiv(BK_groups, 2)):
        comptime if has_perm:
            pass
        else:
            barrier()
            copy_dram_to_sram[thread_layout=Layout.row_major(128, 1)](
                weights_smem_uint4.vectorize[1, 1](),
                raw_weights_gmem_iter[].vectorize[1, 1](),
            )
            raw_weights_gmem_iter._incr()
            barrier()

        if (BK_groups * block_idx[1] + i * 2 + warp_y) < K_groups:
            var repacked_warp_tile = repacked_weights_gmem_iter[].tile[
                repack_tile[0], group_size // pack_factor
            ](warp_x, warp_y)

            comptime for i_Q_tile in range(group_size // repack_tile[1]):
                var tmp: SIMD[.uint8, 16] = 0
                comptime thd_layout = Layout.row_major(8, 4)

                comptime if has_perm:
                    var p_block_idx = perm_idx.tile[BK](block_idx[1])
                    var p_group_idx = p_block_idx.tile[group_size](
                        2 * i + warp_y
                    )
                    var p_Qtile_idx = p_group_idx.tile[repack_tile[1]](i_Q_tile)
                    var thd_idx = p_Qtile_idx.vectorize[2]().distribute[
                        thd_layout, axis=1
                    ](lane_id)
                    var n_idx = lane_id // 4

                    var weights_K = raw_weights.tile[BN, uint_K](
                        block_idx[0], 0
                    )
                    var weights_K_wrap = weights_K.tile[repack_tile[0], uint_K](
                        warp_x, 0
                    )

                    comptime for i_e in range(16):
                        if i_e % 2 == 0 and i_e > 0:
                            n_idx += 8
                        var k_idx: Int = Int(thd_idx[i_e % 2][0])
                        var packed_int = weights_K_wrap[
                            n_idx, k_idx // pack_factor
                        ]
                        tmp[i_e] |= unpack_4bit_int(packed_int, k_idx % 8)

                        k_idx = Int(thd_idx[i_e % 2][1])
                        packed_int = weights_K_wrap[n_idx, k_idx // pack_factor]
                        tmp[i_e] |= unpack_4bit_int(packed_int, k_idx % 8) << 4

                else:
                    var raw_weights_warp_tile = weights_smem.tile[
                        repack_tile[0], weights_bytes_per_group
                    ](warp_x, warp_y)
                    var raw_Q_tile = raw_weights_warp_tile.tile[
                        repack_tile[0], repack_tile[1] // 2
                    ](0, i_Q_tile)
                    # This gets elements 0, 1, 8, 9 in each mma_tile for
                    # thread 0.
                    var thread_tile = raw_Q_tile.distribute[thd_layout](lane_id)

                    comptime for i_e in range(16):
                        tmp[i_e] = thread_tile.load[1](i_e // 2, i_e % 2)

                var repacked_Q_tile = repacked_warp_tile.tile[
                    repack_tile[0], repack_tile[1] // pack_factor
                ](0, i_Q_tile)
                repacked_Q_tile.vectorize[2, 2]().store[4](
                    lane_id, 0, pack_Q_tile(tmp)
                )

            repacked_weights_gmem_iter._incr()

            # cast scales to bf16 before storing back
            var scales_warp_tile = repacked_scales_gmem_iter[].tile[1, 64](
                warp_y, warp_x
            )
            var raw_scales_warp_tile = raw_scales_gmem_iter[].tile[64, 1](
                warp_x, warp_y
            )

            comptime scales_thread_layout = Layout(
                IntTuple(4, 8), IntTuple(16, 1)
            )
            var rt_scales_thread_layout = RuntimeLayout[
                scales_thread_layout,
                element_type=scales_warp_tile.layout_int_type,
                linear_idx_type=scales_warp_tile.linear_idx_type,
            ]()

            scales_warp_tile[0, 2 * lane_id] = convert_bytes_to_bf16[
                scales_type
            ](raw_scales_warp_tile[Int(rt_scales_thread_layout(lane_id)), 0])

            scales_warp_tile[0, 2 * lane_id + 1] = convert_bytes_to_bf16[
                scales_type
            ](
                raw_scales_warp_tile[
                    Int(rt_scales_thread_layout(lane_id)) + 8, 0
                ]
            )

            repacked_scales_gmem_iter._incr()
            raw_scales_gmem_iter._incr()


@always_inline
@doc_hidden
def q_smem_usage[config: MatmulConfig, group_size: Int]() -> Int:
    """Computes the shared memory footprint in bytes for the quantized GEMM kernel under the given configuration.

    Parameters:
        config: The matmul configuration describing tile and warp shapes.
        group_size: The number of K elements sharing a single scale.

    Returns:
        The maximum shared memory in bytes needed across A, B, scales, accumulator, and split-K reduction buffers.
    """
    comptime num_warp_k_partitions = config.num_warp_k_partitions
    comptime block_mnk = config.block_tile_shape
    comptime num_pipeline_stages = config.num_pipeline_stages
    comptime pack_factor = 8

    # fmt: off
    var a_usage = block_mnk[0] * block_mnk[2] * num_pipeline_stages * size_of[config.a_type]()
    var b_usage = block_mnk[1] * block_mnk[2] * num_pipeline_stages * size_of[DType.uint32]() // pack_factor
    var c_usage = block_mnk[0] * block_mnk[1] * size_of[DType.float32]()
    var num_scales_stages = uceildiv((num_pipeline_stages - 1) * block_mnk[2], group_size) + 1
    var scales_usage = block_mnk[1] * ceildiv(
        block_mnk[2], group_size
    ) * num_scales_stages * size_of[config.a_type]()
    var slice_k_reduction = Int(block_mnk[0] * block_mnk[1] * ufloordiv(num_warp_k_partitions, 2) * size_of[DType.float32]())
    # fmt: on

    var smem_usage = num_warp_k_partitions * (a_usage + b_usage + scales_usage)
    return max(c_usage, smem_usage, slice_k_reduction)


@doc_hidden
def multistage_gemm_q[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    *,
    group_size: Int,
    pack_factor: Int,
    config: MatmulConfig[a_type, b_type, c_type, True],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: LayoutTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: LayoutTensor[mut=False, b_type, address_space=.GENERIC, ...],
    runtime_config: MatmulConfig[a_type, b_type, c_type, True],
    ctx: DeviceContext,
) raises:
    """Enqueues the multi-stage quantized GEMM kernel, reducing pipeline stages or warp partitions when the shared memory budget is exceeded.

    Parameters:
        c_type: The dtype of the output matrix.
        a_type: The dtype of the A matrix elements.
        b_type: The dtype of the packed B weight buffer.
        group_size: The number of K elements sharing a single scale.
        pack_factor: The number of 4-bit elements packed into one `uint32`.
        config: The matmul configuration describing tile and warp shapes.
        elementwise_lambda_fn: An optional elementwise epilogue applied per output element.

    Args:
        c: The output matrix in global memory.
        a: The left-hand (activation) matrix in global memory.
        b: The packed quantized weight buffer in global memory.
        runtime_config: The runtime matmul configuration used for grid and block dimensions.
        ctx: The device context used to enqueue the kernel.

    Raises:
        An error if the input tensors are not rank-2.
    """
    comptime assert c.rank == 2
    comptime assert a.rank == 2
    comptime assert b.rank == 2
    var M = c.dim[0]()
    var N = c.dim[1]()

    comptime smem_usage = q_smem_usage[config, group_size]()
    comptime max_smem = ctx.default_device_info.shared_memory_per_multiprocessor

    comptime if smem_usage > max_smem:
        # Strategy:
        # 1. First attempt: Reduce pipeline stages until minimum of 3
        # 2. If still insufficient: Halve the number of warp partitions
        # and retry pipeline stages reduction
        comptime for partition_reduction in range(
            log2_floor(config.num_warp_k_partitions) + 1
        ):
            comptime for num_stages in range(config.num_pipeline_stages, 2, -1):
                comptime adjusted_config = MatmulConfig[
                    a_type, b_type, c_type, True
                ](
                    block_tile_shape=config.block_tile_shape,
                    warp_tile_shape=config.warp_tile_shape,
                    num_pipeline_stages=num_stages,
                    num_k_partitions=config.num_k_partitions,
                    num_warp_k_partitions=(
                        config.num_warp_k_partitions
                        // (2**partition_reduction)
                    ),  # Reduce warp partitions by powers of 2
                )

                comptime adjusted_smem = q_smem_usage[
                    adjusted_config, group_size
                ]()

                comptime if adjusted_smem < max_smem:
                    comptime gemm_kernel_type = multistage_qgemm_kernel[
                        c_type,  # c_type
                        c.layout,
                        a_type,  # a_type
                        a.layout,
                        b_type,  # b_type
                        b.layout,
                        group_size,
                        pack_factor,
                        True,
                        adjusted_config,
                        elementwise_lambda_fn,
                    ]

                    ctx.enqueue_function[gemm_kernel_type](
                        c,
                        a,
                        b,
                        grid_dim=adjusted_config.grid_dim(M, N),
                        block_dim=adjusted_config.block_dim(),
                        shared_mem_bytes=adjusted_smem,
                        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                            UInt32(adjusted_smem),
                        ),
                    )

                    return

    comptime gemm_kernel_type = multistage_qgemm_kernel[
        c_type,  # c_type
        c.layout,
        a_type,  # a_type
        a.layout,
        b_type,  # b_type
        b.layout,
        group_size,
        pack_factor,
        True,
        config,
        elementwise_lambda_fn,
    ]

    ctx.enqueue_function[gemm_kernel_type](
        c,
        a,
        b,
        grid_dim=runtime_config.grid_dim(M, N),
        block_dim=runtime_config.block_dim(),
        shared_mem_bytes=smem_usage,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_usage),
        ),
    )


@always_inline
def matmul_gpu_qint4[
    c_type: DType,
    a_type: DType,
    //,
    group_size: Int,
    target: StaticString,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_tt: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a_tt: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Launches a GPU int4 quantized matrix multiplication for the given tile tensors.

    Parameters:
        c_type: The dtype of the output matrix.
        a_type: The dtype of the A matrix elements.
        group_size: The number of K elements sharing a single scale.
        target: The target platform string, which must identify a GPU.
        elementwise_lambda_fn: An optional elementwise epilogue applied per output element.

    Args:
        c_tt: The output tile tensor in global memory.
        a_tt: The left-hand (activation) tile tensor in global memory.
        b_tt: The packed quantized weight tile tensor in global memory.
        ctx: The device context used to enqueue the kernel.

    Raises:
        An error if the input tensors are not rank-2 or the target is not a GPU.

    Constraints:
        Requires an NVIDIA GPU target. `a_type` and `c_type` must both be
        `bfloat16`.
    """
    var c = c_tt.to_layout_tensor()
    var a = a_tt.to_layout_tensor()
    var b = b_tt.to_layout_tensor()
    comptime assert c.rank == 2
    comptime assert a.rank == 2
    comptime assert b.rank == 2
    comptime assert is_gpu[target](), "unsupported target"
    var cuda_ctx = ctx.value()

    matmul_gpu_qint4_impl[group_size, target, elementwise_lambda_fn](
        c, a, b, cuda_ctx
    )


@always_inline
def matmul_gpu_qint4_impl[
    c_type: DType,
    a_type: DType,
    //,
    group_size: Int,
    target: StaticString,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: LayoutTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext],
) raises:
    """Dispatches a GPU int4 quantized matrix multiplication to the tuned kernel configuration for the runtime M dimension.

    Selects a `MatmulConfig` specialized for the static K and N dimensions and
    the runtime M value, then delegates to `multistage_gemm_q`.

    Parameters:
        c_type: The dtype of the output matrix.
        a_type: The dtype of the A matrix elements.
        group_size: The number of K elements sharing a single scale.
        target: The target platform string, which must identify a GPU.
        elementwise_lambda_fn: An optional elementwise epilogue applied per output element.

    Args:
        c: The output matrix in global memory.
        a: The left-hand (activation) matrix in global memory.
        b: The packed quantized weight buffer in global memory.
        ctx: The device context used to enqueue the kernel.

    Raises:
        An error if the input tensors are not rank-2.
    """
    comptime assert c.rank == 2
    comptime assert a.rank == 2
    comptime assert b.rank == 2
    # comptime assert is_gpu[target](), "unsupported target"
    var cuda_ctx = ctx.value()

    comptime pack_factor = 8

    comptime a_shape = a.layout.shape
    comptime b_shape = b.layout.shape
    comptime c_shape = c.layout.shape
    var shape = GemmShape.get[transpose_b=True](c, a, b)
    var m = shape.M

    comptime static_K = Int(a_shape[1])
    comptime static_N = Int(c_shape[1])

    comptime if static_K == 4096 and static_N == 4096:
        if m <= 16:
            comptime M16_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(16, 64, 128),
                warp_tile_shape=Index(16, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M16_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M16_config,
                cuda_ctx,
            )
            return
        if 16 < m <= 32:
            comptime M32_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(32, 64, 128),
                warp_tile_shape=Index(16, 64, 32),
                num_pipeline_stages=3,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M32_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M32_config,
                cuda_ctx,
            )
            return
        if 32 < m <= 64:
            comptime M64_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(64, 64, 32),
                warp_tile_shape=Index(64, 64, 32),
                num_pipeline_stages=5,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M64_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M64_config,
                cuda_ctx,
            )
            return
        if 64 < m <= 128:
            comptime M128_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(64, 128, 32),
                warp_tile_shape=Index(64, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M128_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M128_config,
                cuda_ctx,
            )
            return
        if 128 < m <= 512:
            comptime M512_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(64, 64, 32),
                warp_tile_shape=Index(64, 64, 32),
                num_pipeline_stages=3,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M512_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M512_config,
                cuda_ctx,
            )
            return

    comptime if static_K == 4096 and static_N == 6144:
        if m <= 16:
            comptime M16_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(16, 64, 128),
                warp_tile_shape=Index(16, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M16_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M16_config,
                cuda_ctx,
            )
            return
        if 16 < m <= 32:
            comptime M32_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(32, 64, 128),
                warp_tile_shape=Index(16, 64, 32),
                num_pipeline_stages=3,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M32_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M32_config,
                cuda_ctx,
            )
            return
        if 32 < m <= 64:
            comptime M64_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(64, 64, 32),
                warp_tile_shape=Index(64, 64, 32),
                num_pipeline_stages=5,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M64_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M64_config,
                cuda_ctx,
            )
            return
        if 64 < m <= 128:
            comptime M128_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(64, 128, 32),
                warp_tile_shape=Index(64, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M128_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M128_config,
                cuda_ctx,
            )
            return
        if 128 < m <= 256:
            comptime M256_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(128, 128, 32),
                warp_tile_shape=Index(64, 64, 32),
                num_pipeline_stages=3,
                num_k_partitions=1,
                num_warp_k_partitions=2,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M256_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M256_config,
                cuda_ctx,
            )
            return

    comptime if static_K == 4096 and static_N == 14336:
        if m <= 16:
            comptime M16_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(16, 64, 32),
                warp_tile_shape=Index(16, 64, 32),
                num_pipeline_stages=5,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M16_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M16_config,
                cuda_ctx,
            )
            return
        if 16 < m <= 32:
            comptime M32_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(16, 64, 32),
                warp_tile_shape=Index(16, 64, 32),
                num_pipeline_stages=5,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M32_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M32_config,
                cuda_ctx,
            )
            return
        if 32 < m <= 64:
            comptime M64_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(32, 64, 32),
                warp_tile_shape=Index(32, 64, 32),
                num_pipeline_stages=3,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M64_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M64_config,
                cuda_ctx,
            )
            return
        if 64 < m <= 256:
            comptime M128_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(64, 64, 32),
                warp_tile_shape=Index(64, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M128_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M128_config,
                cuda_ctx,
            )
            return

    comptime if static_K == 14336 and static_N == 4096:
        if m <= 16:
            comptime M16_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(16, 64, 32),
                warp_tile_shape=Index(16, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=8,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M16_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M16_config,
                cuda_ctx,
            )
            return
        if 16 < m <= 32:
            comptime M32_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(16, 64, 32),
                warp_tile_shape=Index(16, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M32_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M32_config,
                cuda_ctx,
            )
            return
        if 32 < m <= 64:
            comptime M64_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(32, 64, 32),
                warp_tile_shape=Index(32, 64, 32),
                num_pipeline_stages=4,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M64_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M64_config,
                cuda_ctx,
            )
            return
        if 64 < m <= 128:
            comptime M128_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(32, 64, 32),
                warp_tile_shape=Index(32, 64, 32),
                num_pipeline_stages=3,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M128_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M128_config,
                cuda_ctx,
            )
            return
        if 128 < m <= 512:
            comptime M512_config = MatmulConfig[
                a_type, DType.uint8, c_type, True
            ](
                block_tile_shape=Index(64, 64, 32),
                warp_tile_shape=Index(64, 64, 32),
                num_pipeline_stages=3,
                num_k_partitions=1,
                num_warp_k_partitions=4,
            )
            multistage_gemm_q[
                group_size=group_size,
                pack_factor=pack_factor,
                config=M512_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                b,
                M512_config,
                cuda_ctx,
            )
            return

    comptime default_config = MatmulConfig[a_type, .uint8, c_type, True](
        block_tile_shape=Index(128, 128, 32),
        warp_tile_shape=Index(64, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    multistage_gemm_q[
        group_size=group_size,
        pack_factor=pack_factor,
        config=default_config,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](
        c,
        a,
        b,
        default_config,
        cuda_ctx,
    )


@always_inline
def gpu_qint4_repack_Q4_0[
    target: StaticString,
](
    b_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    b_packed_tt: TileTensor[mut=True, .uint8, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Launches the GPU kernel that repacks Q4_0 weights into the packed GEMM layout.

    Parameters:
        target: The target platform string, which must identify a GPU.

    Args:
        b_tt: The input Q4_0 quantized weight tile tensor in global memory.
        b_packed_tt: The output repacked weight tile tensor in global memory.
        ctx: The device context used to enqueue the kernel.

    Raises:
        An error if the input tensors are not rank-2 or the target is not a GPU.
    """
    # Host signature accepts TileTensor; bridge inward to LayoutTensor for the
    # downstream `enqueue_function` whose kernel params are still LayoutTensor.
    # Mirrors `matmul_gpu_qint4` above. Per
    # Kernels/claude_kb/entries/exceptions/host-function-signatures.md the
    # bridge moves inward, it does not disappear.
    var b = b_tt.to_layout_tensor()
    var b_packed = b_packed_tt.to_layout_tensor()
    comptime assert b.rank == 2
    comptime assert b_packed.rank == 2
    comptime assert is_gpu[target](), "unsupported target"
    var cuda_ctx = ctx.value()

    comptime pack_factor = 8
    comptime group_size = 32
    comptime group_bytes = 2 + (group_size // 2)
    comptime BN = 128
    comptime BK = 1024

    comptime N = Int(b.layout.shape[0])
    comptime K = Int(b.layout.shape[1]) // group_bytes * group_size

    var smem_usage: Int = BN * 2 * group_bytes

    comptime repack = repack_Q4_0_for_sm8x[
        b.layout, b_packed.layout, DType.bfloat16
    ]

    cuda_ctx.enqueue_function[repack](
        b,
        b_packed,
        grid_dim=(ceildiv(N, BN), ceildiv(K, BK), 1),
        block_dim=(128, 1, 1),
        shared_mem_bytes=smem_usage,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_usage)
        ),
    )


@always_inline
def gpu_qint4_repack_GPTQ[
    group_size: Int,
    target: StaticString,
](
    b_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    b_packed_tt: TileTensor[mut=True, .uint8, address_space=.GENERIC, ...],
    perm_idx: OptionalReg[
        LayoutTensor[
            mut=False,
            .int32,
            Layout.row_major(UNKNOWN_VALUE),
            ImmutAnyOrigin,
        ]
    ] = None,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Launches the GPU kernel that repacks GPTQ weights into the packed GEMM layout.

    Parameters:
        group_size: The number of K elements sharing a single scale.
        target: The target platform string, which must identify a GPU.

    Args:
        b_tt: The input GPTQ quantized weight tile tensor in global memory.
        b_packed_tt: The output repacked weight tile tensor in global memory.
        perm_idx: An optional permutation index tensor for the K dimension.
        ctx: The device context used to enqueue the kernel.

    Raises:
        An error if the input tensors are not rank-2, the target is not a GPU, or the input and output dimensions are mismatched.
    """
    # `b`/`b_packed` host params accept TileTensor and bridge inward to
    # LayoutTensor for `enqueue_function` (same pattern as `matmul_gpu_qint4`).
    # `perm_idx` stays LayoutTensor: the caller builds a bespoke immutable
    # LayoutTensor view for it, so it is left out of this rotation.
    var b = b_tt.to_layout_tensor()
    var b_packed = b_packed_tt.to_layout_tensor()
    comptime assert b.rank == 2
    comptime assert b_packed.rank == 2
    comptime assert is_gpu[target](), "unsupported target"
    var cuda_ctx = ctx.value()

    comptime pack_factor = 8
    comptime group_bytes = 2 + (group_size // 2)
    comptime BN = 128
    comptime BK = 1024

    comptime N = Int(b.layout.shape[1])
    comptime K = Int(b.layout.shape[0]) // group_bytes * group_size

    comptime assert N == Int(
        b_packed.layout.shape[0]
    ), "qmatmul: Mismatched input/output dimension."
    comptime assert K == (
        Int(b_packed.layout.shape[1]) // group_bytes * group_size
    ), "qmatmul: Mismatched input/output dimension."

    var smem_usage: Int = BN * 2 * group_bytes

    if perm_idx:
        comptime repack = repack_GPTQ_for_sm8x[
            b.layout,
            b_packed.layout,
            DType.bfloat16,
            group_size,
            True,
            perm_layout=perm_idx.T.layout,
        ]

        cuda_ctx.enqueue_function[repack](
            b,
            b_packed,
            perm_idx.value(),
            grid_dim=(ceildiv(N, BN), ceildiv(K, BK), 1),
            block_dim=(128, 1, 1),
        )

    else:
        comptime repack = repack_GPTQ_for_sm8x[
            b.layout,
            b_packed.layout,
            DType.bfloat16,
            group_size,
            False,
        ]

        # Create null tensor using MutUntrackedOrigin (null pointer with no real origin)
        var null_tensor = LayoutTensor[.int32, Layout(), MutUntrackedOrigin](
            None
        )

        cuda_ctx.enqueue_function[repack](
            b,
            b_packed,
            null_tensor,
            grid_dim=(ceildiv(N, BN), ceildiv(K, BK), 1),
            block_dim=(128, 1, 1),
            shared_mem_bytes=smem_usage,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_usage)
            ),
        )
