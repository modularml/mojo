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
from std.math.uutils import umod, ufloordiv, udivmod, uceildiv
from std.sys import (
    align_of,
    has_amd_gpu_accelerator,
    has_amd_rdna_gpu_accelerator,
    is_nvidia_gpu,
    simd_width_of,
    size_of,
)

import max.gpu.primitives.warp as warp
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from max.gpu.compute.mma import mma
from layout.layout import *
from layout import (
    Coord,
    Idx,
    LayoutTensor,
    lt_to_tt,
    RuntimeLayout,
    RuntimeTuple,
    TensorLayout,
    TileTensor,
)
from layout.layout_tensor import (
    LayoutTensorIter,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_local,
    copy_local_to_shared,
    copy_sram_to_dram,
)
from layout.swizzle import Swizzle, make_ldmatrix_swizzle, make_swizzle
from layout.tensor_core import TensorCore, get_fragment_size, get_mma_shape

from std.utils import StaticTuple
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type

from ...utils import elementwise_epilogue_type
from ...utils_gpu import MatmulConfig, block_swizzle
from .amd import AMDMatmul

from ...structuring import SMemTile


@always_inline
def distance[
    dtype: DType, //
](
    arg0: UnsafePointer[mut=False, Scalar[dtype], _],
    arg1: UnsafePointer[mut=False, Scalar[dtype], _],
) -> Int:
    return (Int(arg0) - Int(arg1)) // size_of[dtype]()


comptime WarpSplitKReductionSMem[
    c_type: DType, BM: Int, BN: Int, num_warp_k_partitions: Int
] = SMemTile[
    c_type,
    Layout.row_major(1, BM * BN * (num_warp_k_partitions // 2)),
]


@always_inline
def warp_split_k_reduction[
    c_type: DType,
    c_layout: Layout,
    //,
    BM: Int,
    BN: Int,
    num_threads_per_warp_k_part: Int,
    num_warp_k_partitions: Int,
](
    warp_k_part_id: Int,
    c_reg_tile: LayoutTensor[
        mut=True, c_type, c_layout, address_space=.LOCAL, ...
    ],
    smem: UnsafePointer[mut=True, Scalar[c_type], _, address_space=.SHARED],
):
    comptime red_layout = Layout.row_major(1, num_threads_per_warp_k_part)

    comptime num_mmas = c_layout.shape[0].value()
    comptime c_frag_size = c_layout.shape[1].value()

    var i_red = num_warp_k_partitions // 2
    var tid = thread_idx.x

    while i_red > 0:
        barrier()
        var red_tb_smem = LayoutTensor[
            c_type,
            Layout.row_major(1, BM * BN),
            MutAnyOrigin,
            address_space=.SHARED,
        ](
            (
                smem.bitcast[Scalar[c_type]]()
                + ((warp_k_part_id % i_red) * BM * BN)
            ).as_unsafe_any_origin()
        ).vectorize[
            1, c_frag_size
        ]()
        if i_red <= warp_k_part_id < 2 * i_red:
            copy_local_to_shared[thread_layout=red_layout](
                red_tb_smem,
                c_reg_tile.vectorize[1, c_frag_size](),
            )
        barrier()
        if warp_k_part_id < i_red:
            var red_tb_thread_tile = red_tb_smem.distribute[red_layout](tid)
            var c_reg_tile_vectorized = c_reg_tile.vectorize[
                1, c_frag_size
            ]().transpose()

            comptime for i in range(num_mmas):
                c_reg_tile_vectorized[0, i] += rebind[
                    type_of(c_reg_tile_vectorized[0, i])
                ](red_tb_thread_tile[0, i])
        i_red //= 2


@always_inline
def warp_split_k_reduction[
    c_type: DType,
    c_layout: Layout,
    //,
    BM: Int,
    BN: Int,
    num_threads_per_warp_k_part: Int,
    num_warp_k_partitions: Int,
](
    warp_k_part_id: Int,
    c_reg_tile: LayoutTensor[
        mut=True, c_type, c_layout, address_space=.LOCAL, ...
    ],
):
    comptime c_frag_size = c_layout.shape[1].value()

    var smem = external_memory[
        Scalar[c_type],
        address_space=.SHARED,
        alignment=align_of[SIMD[c_type, c_frag_size]](),
    ]()

    warp_split_k_reduction[
        BM, BN, num_threads_per_warp_k_part, num_warp_k_partitions
    ](warp_k_part_id, c_reg_tile, smem)


@always_inline
def multistage_mma[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    a_smem_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    b_smem_layout: Layout,
    //,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    transpose_b: Bool,
    # Hack:
    /,
    *,
    swizzle_a: Bool = True,
    static_num_iters: Int = -1,
    prefetch_init: Bool = True,
    continue_prefetch_b: Bool = False,
    transpose_b_next: Bool = False,
    b_next_gmem_layout: Layout = Layout(),
    b_next_smem_layout: Layout = Layout(),
    next_op_b_iter_masked: Bool = False,
    next_op_b_iter_alignment: Int = align_of[b_type](),
    next_op_b_layout_int_type: DType = .int64,
    next_op_b_linear_idx_type: DType = .int64,
    k_group_size: Int = 1,
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
    num_iters: Int,
    /,
    *,
    num_b_rows: Optional[Int] = None,
    next_op_b_iter: LayoutTensorIter[
        b_type,
        b_next_gmem_layout,
        ImmutAnyOrigin,
        alignment=next_op_b_iter_alignment,
        layout_int_type=next_op_b_layout_int_type,
        linear_idx_type=next_op_b_linear_idx_type,
        masked=next_op_b_iter_masked,
    ] = LayoutTensorIter[
        b_type,
        b_next_gmem_layout,
        ImmutAnyOrigin,
        alignment=next_op_b_iter_alignment,
        layout_int_type=next_op_b_layout_int_type,
        linear_idx_type=next_op_b_linear_idx_type,
        masked=next_op_b_iter_masked,
    ](),
):
    comptime simd_size = simd_width_of[a_type]()

    # In the slice-K method, we pass `num_threads_per_warp_k_part` as `num_threads`
    # in the parameters. This ensures that `tid` represents the relative thread position
    # within each warp_k_part_id groups.
    var tid = UInt32(umod(thread_idx.x, num_threads))
    var warp_id = warp.broadcast(tid // UInt32(WARP_SIZE))

    comptime num_warps_m = BM // WM
    comptime num_warps_n = BN // WN
    var warp_y, warp_x = divmod(warp_id, UInt32(num_warps_n))

    var a_iter = a_iter_arg
    var b_iter = b_iter_arg
    var a_smem_iter = a_smem_iter_arg
    # work around mut argument can't have default value.
    var next_b_iter = next_op_b_iter

    # If there are more threads than vectors, thread layout should be based on
    # the latter so that a vector is only mapped to one thread.
    comptime a_num_vecs = BM * BK // simd_size
    comptime async_copy_a_layout = Layout.row_major(
        min(num_threads, a_num_vecs) * simd_size // BK, BK // simd_size
    )

    comptime b_num_ves = BN * BK // simd_size
    comptime async_copy_b_layout = Layout.row_major(
        min(num_threads, b_num_ves)
        * simd_size
        // b_smem_layout.shape[1].value(),
        b_smem_layout.shape[1].value() // simd_size,
    )

    # TODO (KERN-1337): Enable swizzle for matrix B for FP8 data type and transpose_b==False
    comptime swizzle_b = (
        transpose_b or b_type.is_half_float()
    ) and is_nvidia_gpu()

    @always_inline
    @__parameter
    def _mask_tensor_row(
        tensor: LayoutTensor, num_rows: Int, out result: type_of(tensor)
    ):
        return {
            tensor.ptr,
            RuntimeLayout[
                element_type=tensor.layout_int_type,
                linear_idx_type=tensor.linear_idx_type,
            ](
                RuntimeTuple[
                    tensor.layout.shape, element_type=tensor.layout_int_type
                ](num_rows, tensor.dim[1]()),
                tensor.runtime_layout.stride,
            ),
        }

    @always_inline
    @__parameter
    def _copy_tensor_to_sram[
        thread_layout: Layout, swizzle: Bool
    ](dst: LayoutTensor[mut=True, ...], src: LayoutTensor):
        comptime if is_nvidia_gpu():
            copy_dram_to_sram_async[
                thread_layout=thread_layout,
                swizzle=swizzle,
                num_threads=num_threads,
            ](
                dst.vectorize[1, simd_size](),
                src.vectorize[1, simd_size](),
            )
        else:
            copy_dram_to_sram[thread_layout=thread_layout](
                dst.vectorize[1, simd_size](),
                src.vectorize[1, simd_size](),
            )

    # Prefetch (num_pipeline_stages - 1) stages.
    comptime if prefetch_init:
        comptime for stage in range(num_pipeline_stages - 1):
            comptime if a_iter.address_space == .GENERIC:
                var a_smem_tile = a_smem_iter.next_unsafe(
                    a_smem_iter.linear_uint_type(stage)
                )[]
                _copy_tensor_to_sram[async_copy_a_layout, swizzle_a](
                    a_smem_tile, a_iter[]
                )

                a_iter._incr()

            comptime if b_iter.address_space == .GENERIC:
                var b_smem_tile = b_smem_iter.next_unsafe(
                    b_smem_iter.linear_uint_type(stage)
                )[]

                if num_b_rows:
                    var num_rows_bound = (
                        num_b_rows.value() if transpose_b else max(
                            0, num_b_rows.value() - stage * BK
                        )
                    )
                    var b_tensor = _mask_tensor_row(b_iter[], num_rows_bound)
                    _copy_tensor_to_sram[async_copy_b_layout, swizzle_b](
                        b_smem_tile, b_tensor
                    )
                else:
                    _copy_tensor_to_sram[async_copy_b_layout, swizzle_b](
                        b_smem_tile, b_iter[]
                    )

                b_iter._incr()

            async_copy_commit_group()

        # Guard stage 0.
        async_copy_wait_group(Int32(num_pipeline_stages - 2))
        barrier()

    comptime mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_k_mmas = BK // MMA_K
    comptime num_k_mma_iters: Int = num_k_mmas // k_group_size
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N
    comptime assert (
        num_k_mmas % (2 * k_group_size) == 0
    ), "num_k_mmas must be an integer multiple of 2*k_group_size"

    comptime accum_type = get_accum_type[a_type]()
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime a_frag_size = frag_size[0]
    comptime b_frag_size = frag_size[1]
    comptime c_frag_size = frag_size[2]

    comptime num_reg_tiles = 2 * k_group_size
    # Register tiles.
    comptime a_reg_layout = Layout.row_major(
        2 * k_group_size * num_m_mmas, a_frag_size
    )
    var a_reg_tiles = (
        LayoutTensor[
            a_type,
            a_reg_layout,
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .split[2 * k_group_size]()
    )

    comptime b_reg_layout = Layout.row_major(
        2 * k_group_size * num_n_mmas, b_frag_size
    )
    var b_reg_tiles = (
        LayoutTensor[
            b_type,
            b_reg_layout,
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .vectorize[1, b_frag_size]()
        .split[2 * k_group_size]()
    )

    var a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)

    comptime b_wtile_dim0 = WN if transpose_b else BK
    comptime b_wtile_dim1 = BK if transpose_b else WN
    var b_wtile_coord0 = Int(warp_x) if transpose_b else 0
    var b_wtile_coord1 = 0 if transpose_b else Int(warp_x)
    var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
        b_wtile_coord0, b_wtile_coord1
    )

    var mma_op = TensorCore[accum_type, a_type, mma_shape, transpose_b]()

    comptime swizzle_a_pattern = make_ldmatrix_swizzle[
        a_type, a_warp_tile.stride[0]()
    ]() if swizzle_a else Optional[Swizzle]()

    comptime for i in range(k_group_size):
        comptime if a_iter.address_space == .LOCAL:
            # Assume input is the 16x8 output of 16x8x16 or 16x8x8 mma.
            # Need to cast address space because it's not known at parse time to be LOCAL.
            copy_local_to_local(a_reg_tiles[i], a_iter[])
            a_iter._incr()
        else:
            mma_op.load_a[swizzle_a_pattern](
                a_warp_tile, a_reg_tiles[i].vectorize[1, a_frag_size](), i
            )

        mma_op.load_b(b_warp_tile, b_reg_tiles[i], i, Int(warp_x))

    comptime if static_num_iters >= 0:
        comptime assert (
            a_iter.address_space == .SHARED or a_iter.address_space == .LOCAL
        ), (
            "Using input in registers or shared memory requires static"
            " iteration bound.\n"
        )

        comptime for k_tile_id in range(static_num_iters):
            var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
                b_wtile_coord0,
                b_wtile_coord1,
            )

            # Perform prefetch registers and mma until current shared memory tile's
            # data has all been loaded to registers.
            comptime for k_mma0 in range(num_k_mma_iters):
                comptime for k_mma1 in range(k_group_size):
                    comptime k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                    comptime k_mma_next = k_mma + UInt32(k_group_size)
                    comptime next = Int(k_mma_next % UInt32(num_reg_tiles))

                    comptime if k_mma_next == UInt32(num_k_mmas):
                        comptime prefetch_tile_id = k_tile_id + num_pipeline_stages - 1

                        # Prefetch one k tile (if valid) from global memory to current
                        # shared memory buffer.
                        comptime if b_iter.address_space == .GENERIC:
                            comptime if prefetch_tile_id < static_num_iters:
                                var b_smem_prefetch_tile = (
                                    b_smem_iter.next_unsafe(
                                        b_smem_iter.linear_uint_type(
                                            num_pipeline_stages - 1
                                        )
                                    )[]
                                )

                                if num_b_rows:
                                    var num_rows_bound = num_b_rows.value() if transpose_b else max(
                                        0,
                                        num_b_rows.value()
                                        - prefetch_tile_id * BK,
                                    )
                                    var b_tensor = _mask_tensor_row(
                                        b_iter[], num_rows_bound
                                    )
                                    _copy_tensor_to_sram[
                                        async_copy_b_layout, swizzle_b
                                    ](b_smem_prefetch_tile, b_tensor)
                                else:
                                    _copy_tensor_to_sram[
                                        async_copy_b_layout, swizzle_b
                                    ](b_smem_prefetch_tile, b_iter[])

                                b_iter._incr()

                            async_copy_commit_group()

                            # Guard the next k tile's shared memory buffer.
                            async_copy_wait_group(
                                Int32(num_pipeline_stages - 2)
                            )
                            barrier()

                        comptime if a_iter.address_space == .SHARED:
                            a_smem_iter._incr()
                        b_smem_iter._incr()

                        a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)
                        b_warp_tile = b_smem_iter[].tile[
                            b_wtile_dim0, b_wtile_dim1
                        ](b_wtile_coord0, b_wtile_coord1)

                    comptime kidx = Int(k_mma_next % UInt32(num_k_mmas))

                    comptime if a_iter.address_space == .SHARED:
                        mma_op.load_a[swizzle_a_pattern](
                            a_warp_tile,
                            a_reg_tiles[next].vectorize[1, a_frag_size](),
                            kidx,
                        )
                    else:
                        # Assume input is the 16x8 output of 16x8x16 or 16x8x8 mma.
                        copy_local_to_local(a_reg_tiles[next], a_iter[])
                        a_iter._incr()

                    mma_op.load_b(
                        b_warp_tile,
                        b_reg_tiles[next],
                        kidx,
                        Int(warp_x),
                    )

                comptime for k_mma1 in range(k_group_size):
                    comptime k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                    comptime current = k_mma % UInt32(num_reg_tiles)
                    mma_op.mma(
                        a_reg_tiles[Int(current)].vectorize[1, a_frag_size](),
                        b_reg_tiles[Int(current)],
                        c.vectorize[1, c_frag_size](),
                    )

        return

    for k_tile_id in range(num_iters):
        var a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)
        var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
            b_wtile_coord0,
            b_wtile_coord1,
        )

        # Perform prefetch registers and mma until current shared memory tile's
        # data has all been loaded to registers.
        comptime for k_mma0 in range(num_k_mma_iters):
            comptime for k_mma1 in range(k_group_size):
                comptime k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                comptime k_mma_next = k_mma + UInt32(k_group_size)
                comptime next = Int(k_mma_next % UInt32(num_reg_tiles))

                comptime if k_mma_next == UInt32(num_k_mmas):
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
                            _copy_tensor_to_sram[
                                async_copy_a_layout, swizzle_a
                            ](a_smem_prefetch_tile, a_iter[])

                            a_iter._incr()

                        comptime if b_iter.address_space == .GENERIC:
                            var b_smem_prefetch_tile = b_smem_iter.next_unsafe(
                                b_smem_iter.linear_uint_type(
                                    num_pipeline_stages - 1
                                )
                            )[]

                            if num_b_rows:
                                var num_rows_bound = (
                                    num_b_rows.value() if transpose_b else max(
                                        0,
                                        num_b_rows.value()
                                        - prefetch_tile_id * BK,
                                    )
                                )
                                var b_tensor = _mask_tensor_row(
                                    b_iter[], num_rows_bound
                                )
                                _copy_tensor_to_sram[
                                    async_copy_b_layout, swizzle_b
                                ](b_smem_prefetch_tile, b_tensor)
                            else:
                                _copy_tensor_to_sram[
                                    async_copy_b_layout, swizzle_b
                                ](b_smem_prefetch_tile, b_iter[])

                            b_iter._incr()
                    else:
                        comptime if continue_prefetch_b:
                            var b_smem_prefetch_tile = b_smem_iter.next_unsafe(
                                b_smem_iter.linear_uint_type(
                                    num_pipeline_stages - 1
                                )
                            )[].reshape[b_next_smem_layout]()

                            comptime row_size = b_next_smem_layout.stride[
                                0
                            ].value()

                            comptime b_prefetch_thread_layout = Layout.row_major(
                                num_threads * simd_size // row_size,
                                row_size // simd_size,
                            )
                            comptime swizzle_prefetch_b = (
                                transpose_b_next or b_type.is_half_float()
                            ) and is_nvidia_gpu()

                            if num_b_rows:
                                # TODO: can we guard at compile time num_b_rows is set here?
                                var num_rows_bound = num_b_rows.value() if transpose_b_next else max(
                                    0,
                                    num_b_rows.value()
                                    - (prefetch_tile_id - num_iters) * BK,
                                )

                                var b_tensor = _mask_tensor_row(
                                    next_b_iter[], num_rows_bound
                                )
                                _copy_tensor_to_sram[
                                    b_prefetch_thread_layout, swizzle_prefetch_b
                                ](b_smem_prefetch_tile, b_tensor)

                            else:
                                _copy_tensor_to_sram[
                                    b_prefetch_thread_layout, swizzle_prefetch_b
                                ](b_smem_prefetch_tile, next_b_iter[])

                            next_b_iter._incr()

                    async_copy_commit_group()

                    # Guard the next k tile's shared memory buffer.
                    async_copy_wait_group(Int32(num_pipeline_stages - 2))
                    barrier()

                    a_smem_iter._incr()
                    b_smem_iter._incr()

                    a_warp_tile = a_smem_iter[].tile[WM, BK](Int(warp_y), 0)
                    b_warp_tile = b_smem_iter[].tile[
                        b_wtile_dim0, b_wtile_dim1
                    ](b_wtile_coord0, b_wtile_coord1)

                comptime kidx = Int(k_mma_next % UInt32(num_k_mmas))
                mma_op.load_a[swizzle_a_pattern](
                    a_warp_tile,
                    a_reg_tiles[next].vectorize[1, a_frag_size](),
                    kidx,
                )
                mma_op.load_b(
                    b_warp_tile,
                    b_reg_tiles[next],
                    kidx,
                    Int(warp_x),
                )

            comptime for k_mma1 in range(k_group_size):
                comptime k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                comptime current = k_mma % UInt32(num_reg_tiles)
                mma_op.mma(
                    a_reg_tiles[Int(current)].vectorize[1, a_frag_size](),
                    b_reg_tiles[Int(current)],
                    c.vectorize[1, c_frag_size](),
                )


@__name(
    t"multistage_gemm_kernel_{c_type}_{a_type}_{b_type}_{transpose_b}",
)
def multistage_gemm_kernel[
    c_type: DType,
    CLT: TensorLayout,
    a_type: DType,
    ALT: TensorLayout,
    b_type: DType,
    BLT: TensorLayout,
    transpose_b: Bool,
    c_linear_idx_type: DType,
    a_linear_idx_type: DType,
    b_linear_idx_type: DType,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b, ...],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_tt: TileTensor[
        c_type, CLT, MutAnyOrigin, linear_idx_type=c_linear_idx_type
    ],
    a_tt: TileTensor[
        a_type, ALT, ImmutAnyOrigin, linear_idx_type=a_linear_idx_type
    ],
    b_tt: TileTensor[
        b_type, BLT, ImmutAnyOrigin, linear_idx_type=b_linear_idx_type
    ],
):
    var c = c_tt.to_layout_tensor()
    var a = a_tt.to_layout_tensor()
    var b = b_tt.to_layout_tensor()
    # float16 shares bf16's MMA path here; its accumulation precision can
    # differ from bf16, so it is opt-in via the float16 quantization encoding.
    comptime assert (
        a_type in (DType.float32, DType.bfloat16, DType.float16)
        and a_type == b_type
    ) or (
        a_type in (DType.float8_e4m3fn, DType.float8_e5m2)
        and a_type == b_type
        and c_type == .float32
    ), "Pipeline gemm only supports tf32, F16, BF16, E4M3, and E5M2 mma"
    comptime simd_size = simd_width_of[c_type]()

    var M: Int = c.dim[0]()
    var N: Int = b.dim[0 if transpose_b else 1]()
    var K: Int = b.dim[1 if transpose_b else 0]()

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime WM = config.warp_tile_shape[0]
    comptime WN = config.warp_tile_shape[1]
    comptime num_pipeline_stages = config.num_pipeline_stages

    comptime num_warps_m = config.num_warps_m()
    comptime num_warps_n = config.num_warps_n()
    comptime num_threads = config.num_threads()

    comptime num_warp_k_partitions = config.num_warp_k_partitions
    comptime num_threads_per_warp_k_part = num_threads // num_warp_k_partitions

    var tid = thread_idx.x
    var ln_id = lane_id()
    var warp_k_part_id = (
        ufloordiv(tid, num_threads_per_warp_k_part) if num_warp_k_partitions
        > 1 else 0
    )
    var warp_id = warp.broadcast(
        ufloordiv(umod(tid, num_threads_per_warp_k_part), WARP_SIZE)
    )

    # Only apply block swizzling for half precision types.
    comptime swizzle_block = a_type.is_half_float() and b_type.is_half_float() and is_nvidia_gpu()

    # NOTE: the condition ( not (N // BN & 1)) is for a temporary solution
    # for solving mismatches in some shapes
    var block_idx_swizzle = block_swizzle(
        Index[dtype=DType.uint32](block_idx.x, block_idx.y),
        Index[dtype=DType.uint32](grid_dim.x, grid_dim.y),
    ) if swizzle_block else Index[dtype=DType.uint32](block_idx.x, block_idx.y)

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
    comptime a_smem_size: Int = num_pipeline_stages * BM * BK
    comptime IteratorTypeA = LayoutTensorIter[
        a_type,
        Layout.row_major(BM, BK),
        _,
        address_space=a_smem.address_space,
        alignment=alignment,
        circular=True,
    ]
    var a_smem_iter = IteratorTypeA(
        a_smem + IteratorTypeA.linear_uint_type(warp_k_part_id * a_smem_size),
        IteratorTypeA.linear_uint_type(a_smem_size),
    )

    # There is one pre-allocated shared buffer. Explicitly offset B after at A's end.
    var b_smem = (a_smem + num_warp_k_partitions * a_smem_size).bitcast[
        Scalar[b_type]
    ]()
    comptime b_smem_size: Int = num_pipeline_stages * BK * BN
    comptime BD_0 = BN if transpose_b else BK
    comptime BD_1 = BK if transpose_b else BN
    comptime b_smem_layout = Layout.row_major(BD_0, BD_1)
    comptime IteratorTypeB = LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        circular=True,
    ]
    var b_smem_iter = IteratorTypeB(
        (
            b_smem
            + IteratorTypeB.linear_uint_type(warp_k_part_id * b_smem_size)
        ).as_unsafe_any_origin(),
        IteratorTypeB.linear_uint_type(b_smem_size),
    )

    # create input layout tensors A and Bv
    # global memory iterator
    var bk_start: Int = (
        ufloordiv(ufloordiv(K, BK), num_warp_k_partitions) * warp_k_part_id
    )
    var a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](
        block_idx_swizzle[1], bk_start
    )
    var b_tile_coords = (block_idx_swizzle[0], bk_start) if transpose_b else (
        bk_start,
        block_idx_swizzle[0],
    )
    comptime b_tile_axis = 1 if transpose_b else 0
    var b_gmem_iter = b.tiled_iterator[BD_0, BD_1, axis=b_tile_axis](
        b_tile_coords[0], b_tile_coords[1]
    )

    # Compute MMA config
    comptime mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_k_mmas = BK // MMA_K
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
            accum_type,
            c_reg_layout,
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()  # ALIGN-TODO: pass alignment here?
        .fill(0)
    )

    multistage_mma[
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads_per_warp_k_part,
        num_pipeline_stages,
        transpose_b,
        k_group_size=config.k_group_size,
        swizzle_a=is_nvidia_gpu(),
    ](
        c_reg_tile,
        a_gmem_iter,
        b_gmem_iter,
        a_smem_iter,
        b_smem_iter,
        uceildiv(ufloordiv(K, num_warp_k_partitions), BK),
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
    var c_gmem_tile = c.tile[BM, BN](block_idx_swizzle[1], block_idx_swizzle[0])
    var c_gmem_warp_tile = c_gmem_tile.tile[WM, WN](warp_y, warp_x)

    # The NVIDIA C store below writes 2 elements at a time along N, which needs
    # every row of C to be aligned to that vector. A row starts at a multiple of
    # N * size_of[c_type](), so an odd fp32 N (e.g. GPT-2's 50257 vocab
    # projection) leaves every other row 4B-misaligned and the store faults.
    # AMD vectorizes along M instead, one column at a time, so it is exempt.
    var c_rows_aligned = not is_nvidia_gpu() or (
        N * size_of[c_type]() % align_of[SIMD[c_type, 2]]() == 0
    )
    var warp_row = Int(block_idx_swizzle[1]) * BM + Int(warp_y) * WM
    var warp_col = Int(block_idx_swizzle[0]) * BN + Int(warp_x) * WN
    comptime store_vec_rows = 1 if is_nvidia_gpu() else 4
    var c_tile_in_range = (
        warp_row + WM <= M and warp_col + WN <= N and M % store_vec_rows == 0
    )

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
            comptime dst_static_idx = Int(type_of(c_gmem_frag).layout(i))
            var dst_idx: Int

            comptime if c_gmem_frag.layout.all_dims_known():
                dst_idx = dst_static_idx
            else:
                dst_idx = Int(c_gmem_frag.runtime_layout(i))
            comptime alignment = align_of[SIMD[c_type, src_simd_width_y]]()
            var m, n = divmod(Int(thread_offset) + dst_idx, N)
            if m < M and n < N:
                var vec = (c_reg_frag.ptr + src_idx).load[
                    width=src_simd_width_y,
                    alignment=align_of[SIMD[c_type, src_simd_width_y]](),
                ]()

                comptime if dst_simd_width_x == 1:
                    epilogue[alignment=alignment]((m, n), vec)
                else:
                    # One element per row, so the vector's alignment does not
                    # carry over to the individual stores.
                    comptime for j in range(dst_simd_width_x):
                        if m + j < M:
                            epilogue[alignment=align_of[Scalar[c_type]]()](
                                (m + j, n), vec[j].cast[c_type]()
                            )

    @always_inline
    @__parameter
    def store_c_scalar():
        """Writes C one element at a time, bounded by the real (row, col).

        Used for the C tiles the vectorized stores cannot handle: an odd fp32 N
        (misaligned rows), a warp whose columns run past N, and on AMD a 4-row
        group that runs past M. Those fragments carry a linear offset that
        lands on the next row, so they must be rejected on their true
        coordinates rather than on the ones a linear offset implies.
        """
        # Mirrors the vectorized stores this replaces: NVIDIA distributes
        # `row_major(8, 4)` over 2-element vectors along N, AMD distributes
        # `row_major(4, 16)` over 4-element vectors along M.
        comptime threads_m = 8 if is_nvidia_gpu() else 4
        comptime threads_n = 4 if is_nvidia_gpu() else 16
        comptime vec_width = 2 if is_nvidia_gpu() else 4
        comptime rows_per_vec = 1 if is_nvidia_gpu() else vec_width
        comptime cols_per_vec = vec_width if is_nvidia_gpu() else 1
        comptime row_step = threads_m * rows_per_vec
        comptime col_step = threads_n * cols_per_vec
        comptime frag_rows = WM // row_step
        comptime frag_cols = WN // col_step

        var c_reg_frag = c_reg_tile.vectorize[1, vec_width]().transpose()
        var thread_row = warp_row + (Int(ln_id) // threads_n) * rows_per_vec
        var thread_col = warp_col + (Int(ln_id) % threads_n) * cols_per_vec

        comptime for frag_col in range(frag_cols):
            comptime for frag_row in range(frag_rows):
                # `layout()` walks the leading mode fastest.
                comptime src_idx = c_reg_frag.layout(
                    frag_col * frag_rows + frag_row
                )
                var row = thread_row + frag_row * row_step
                var col = thread_col + frag_col * col_step
                var vec = (c_reg_frag.ptr + src_idx).load[width=vec_width]()

                comptime for j in range(vec_width):
                    var row_j = row + (0 if is_nvidia_gpu() else j)
                    var col_j = col + (j if is_nvidia_gpu() else 0)

                    if row_j < M and col_j < N:
                        comptime if elementwise_lambda_fn:
                            comptime epilogue = elementwise_lambda_fn.value()
                            epilogue[alignment=align_of[Scalar[c_type]]()](
                                (row_j, col_j), vec[j].cast[c_type]()
                            )
                        else:
                            c[row_j, col_j] = vec[j].cast[c_type]()

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
            MutAnyOrigin,
            address_space=.SHARED,
        ](
            (
                a_smem.bitcast[Scalar[c_type]]() + warp_id * WM * WN
            ).as_unsafe_any_origin()
        )

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

                var m, n = divmod(Int(thread_offset) + dst_idx, N)
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
        if c_tile_in_range:
            comptime if elementwise_lambda_fn:
                apply_epilogue()

            else:
                var c_reg_tile_out = LayoutTensor[
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
        else:
            store_c_scalar()
    # Store FP32 results to FP32 buffer in global memory.
    else:
        if c_rows_aligned and c_tile_in_range:
            comptime if elementwise_lambda_fn:
                apply_epilogue()
            else:
                comptime if is_nvidia_gpu():
                    copy_local_to_dram[
                        dst_thread_layout=Layout.row_major(8, 4)
                    ](
                        c_gmem_warp_tile.vectorize[1, 2](),
                        c_reg_tile.vectorize[1, 2]().transpose(),
                    )
                else:
                    copy_local_to_dram[
                        dst_thread_layout=Layout.row_major(4, 16)
                    ](
                        c_gmem_warp_tile.vectorize[4, 1](),
                        c_reg_tile.vectorize[1, 4](),
                    )
        else:
            store_c_scalar()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads())
    )
)
@__name(
    t"multistage_gemm_split_k_kernel_{c_type}_{a_type}_{b_type}_{transpose_b}",
)
def multistage_gemm_split_k_kernel[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    work_space_type: DType,
    workspace_layout: Layout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
    work_space: LayoutTensor[work_space_type, workspace_layout, MutAnyOrigin],
    num_partitions: Int32,
):
    var _num_partitions = Int(num_partitions)
    var M = c.dim[0]()
    comptime N = b.shape[0]() if transpose_b else b.shape[1]()
    comptime K = b.shape[1]() if transpose_b else b.shape[0]()
    comptime BK = config.block_tile_shape[2]

    comptime work_space_tensor_type = LayoutTensor[
        work_space_type, c_layout, MutAnyOrigin
    ]

    var work_space_part = work_space_tensor_type(
        work_space.ptr + block_idx.z * M * N,
        RuntimeLayout[
            c_layout,
            element_type=work_space_tensor_type.layout_int_type,
            linear_idx_type=work_space_tensor_type.linear_idx_type,
        ].row_major(
            IndexList[2, element_type=work_space_tensor_type.layout_int_type](
                M, N
            )
        ),
    )
    comptime k_partition_config = MatmulConfig[
        a_type,
        b_type,
        work_space_type,
        transpose_b,
    ](
        block_tile_shape=config.block_tile_shape,
        warp_tile_shape=config.warp_tile_shape,
        num_pipeline_stages=config.num_pipeline_stages,
    )

    var ws_tt = lt_to_tt(work_space_part)

    comptime if (
        has_amd_gpu_accelerator()
        and not has_amd_rdna_gpu_accelerator()
        and transpose_b
    ):
        # `.split` makes the K axis dynamic, but AMDMatmul derives its
        # K-loop bound from the static shape. Carve comptime K/P tiles so
        # each partition keeps a static K (parent strides preserved).
        comptime assert K % config.num_k_partitions == 0, (
            "AMD split-K carves static K/P tiles, so K must be divisible by"
            " num_k_partitions (no ragged remainder)."
        )
        comptime K_part = K // config.num_k_partitions
        # AMDMatmul's K-loop iterates K_part // BK tiles by integer division,
        # so a ragged tail would be silently dropped. The dispatch gate keeps
        # BK == 64 and K a multiple of num_k_partitions * 64.
        comptime assert (
            K_part % BK == 0
        ), "AMD split-K requires each K partition to be a multiple of BK."
        var z = Int(block_idx.z)
        var a_amd = lt_to_tt(a).tile(Coord(M, Idx[K_part]), Coord(0, z))
        var b_amd = lt_to_tt(b).tile(Coord(Idx[N], Idx[K_part]), Coord(0, z))
        AMDMatmul[
            a_type,
            b_type,
            work_space_type,
            transpose_b,
            k_partition_config,
        ].run[
            type_of(ws_tt).LayoutType,
            type_of(a_amd).LayoutType,
            type_of(b_amd).LayoutType,
            type_of(ws_tt).Engine,
            type_of(a_amd).Engine,
            type_of(b_amd).Engine,
        ](
            ws_tt, a_amd, b_amd
        )

    else:
        # If K is not divisible by num_partitions, the first
        # num_partitions-1 parts are rounded up to a multiple of BK.
        var a_part = a.split[axis=1, split_alignment=BK](
            _num_partitions, block_idx.z
        )
        var b_part = b.split[axis=1 if transpose_b else 0, split_alignment=BK](
            _num_partitions, block_idx.z
        )
        var a_tt = lt_to_tt(a_part)
        var b_tt = lt_to_tt(b_part)
        multistage_gemm_kernel[config=k_partition_config,](ws_tt, a_tt, b_tt)
