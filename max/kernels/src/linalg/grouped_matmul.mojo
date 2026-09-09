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

"""Provides grouped matrix multiplication kernels for CPU, AMD, and NVIDIA GPU targets."""

from std.collections import Optional
from std.math import ceildiv
from std.sys import align_of, simd_width_of, size_of
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_amd_rdna_gpu_accelerator,
    has_apple_gpu_accelerator,
)

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.host.info import H100, _is_sm10x_gpu, is_gpu
from max.gpu import block_idx, global_idx, warp_id, lane_id, thread_idx
from max.gpu.memory import external_memory
from max.gpu.primitives.grid_controls import PDLLevel
from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id
from std.collections.string.string_span import get_static_string

from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.compute.arch.tcgen05 import *
from layout import (
    Coord,
    Idx,
    IntTuple,
    Layout,
    LayoutTensor,
    LTToTTLayout,
    row_major,
    RuntimeLayout,
    TensorLayout,
    TensorEngine,
    TileTensor,
    UNKNOWN_VALUE,
)
from layout.tensor_core_async import tile_layout_k_major
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
)

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from .arch.sm100 import MmaOpSM100_SS
from .matmul.gpu.sm90.dispatch import _find_largest_bn_for_sm90_matmul
from .matmul.gpu.sm90.grouped_matmul import grouped_matmul_sm90
from .matmul.vendor.blas import matmul as vendor_matmul
from .utils import elementwise_epilogue_type, lora_qkv_plane_row_offset
from .utils_gpu import MatmulConfig, _bk_base
from .grouped_matmul_sm100 import grouped_matmul_sm100_persistent

from .matmul.gpu import (
    _amdgpu_matmul_build_block_shape_list,
    _amdgpu_matmul_config_from_block_shape,
)
from .matmul.gpu.amd import AMDMatmul
from .matmul.gpu.apple.matmul2d_fp8 import enqueue_grouped_matmul2d_fp8
from std.algorithm import vectorize


# ===----------------------------------------------------------------------=== #
# Naive grouped matmul
# ===----------------------------------------------------------------------=== #


# grouped matmul computes:
# for i in range(num_active_experts)
#     C[a_offsets[i]:a_offsets[i+1], :] = A[a_offsets[i]:a_offsets[i+1], :] @ B[expert_ids[i], :, :].T


@__name(t"naive_grouped_matmul_kernel_{c_type}_{a_type}_{b_type}")
def naive_grouped_matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    CLayout: TensorLayout,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    AOffsetsLayout: TensorLayout,
    ExpertIdsLayout: TensorLayout,
    *,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    a_plane_splits: IndexList[2] = Index(0, 0),
](
    c: TileTensor[mut=True, c_type, CLayout, MutAnyOrigin],
    a: TileTensor[mut=False, a_type, ALayout, MutAnyOrigin],
    b: TileTensor[mut=False, b_type, BLayout, MutAnyOrigin],
    a_offsets: TileTensor[mut=False, .uint32, AOffsetsLayout, MutAnyOrigin],
    expert_ids: TileTensor[mut=False, .int32, ExpertIdsLayout, MutAnyOrigin],
):
    """Computes one element per thread of the grouped matmul product ``C[a_offsets[z]:a_offsets[z+1], :] = A[...] @ B[expert_ids[z], :, :].T`` for each active expert ``z``, with an optional elementwise epilogue.

    Skips the matmul for ``expert == -1`` (inactive LoRA blocks) but still
    invokes the elementwise lambda when provided.
    """
    comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
    comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"
    comptime assert c.flat_rank == 2, "c must be rank 2"
    comptime assert a.flat_rank == 2, "a must be rank 2"
    comptime assert b.flat_rank == 3, "b must be rank 3"

    var M: Int = Int(a_offsets[block_idx.z + 1] - a_offsets[block_idx.z])
    var N = Int(b.dim[1]())
    var K = Int(b.dim[2]())

    var a_start_row = a_offsets[block_idx.z]

    var expert = expert_ids[block_idx.z]
    var b_by_expert = b.ptr + Int64(expert) * Int64(N) * Int64(K)

    # indices in current matmul
    var n = global_idx.x
    var m = global_idx.y

    if n >= N or m >= M:
        return

    # Per-output-column-region activation row offset. Default 0; the LoRA-B QKV
    # expand uses it to pick the matching Q/K/V plane of the `[3M, R]` planar
    # shrink output. `a` is then `[3 * M_total, K]`, so the per-plane row stride
    # is `a.dim[0] // 3`; the offset is `plane(n) * stride`.
    var a_row_off = 0
    comptime if a_plane_splits[0] > 0:
        a_row_off = lora_qkv_plane_row_offset[a_plane_splits](
            Int(n), Int(a.dim[0]()) // 3
        )
    var a_by_expert = a.ptr + Int64(Int(a_start_row) + a_row_off) * Int64(K)

    comptime accum_type = get_accum_type[a_type]()

    var accum = Scalar[accum_type](0.0)

    # avoid doing matmul if expert is -1. We use this value to indicate that
    # the block is not active for LoRA use cases.
    # NOTE: we still call elementwise lambda even if expert is -1
    if expert != -1:
        for k in range(K):
            accum += (
                a_by_expert[m * K + k].cast[accum_type]()
                * b_by_expert[n * K + k].cast[accum_type]()
            )

    comptime if elementwise_lambda_fn:
        comptime elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](
            Index(a_start_row + UInt32(m), n), accum.cast[c_type]()
        )
    else:
        var c_by_expert = c.ptr + Int64(a_start_row) * Int64(N)
        c_by_expert[m * N + n] = accum.cast[c_type]()


# ===----------------------------------------------------------------------=== #
# H100 grouped matmul
# ===----------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads)),
)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__name(
    t"grouped_matmul_kernel_sm100_{a_type}_{b_type}_{c_type}_t{num_threads}",
)
def grouped_matmul_kernel_sm100[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    static_K: Int,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    CLayout: TensorLayout,
    AOffsetsLayout: TensorLayout,
    ExpertIdsLayout: TensorLayout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    transpose_b: Bool = True,
    num_threads: Int = 128,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    a_offsets: TileTensor[mut=False, .uint32, AOffsetsLayout, MutAnyOrigin],
    expert_ids: TileTensor[mut=False, .int32, ExpertIdsLayout, MutAnyOrigin],
    c: TileTensor[mut=True, c_type, CLayout, MutAnyOrigin],
    num_iters: Int32,
):
    var _num_iters = Int(num_iters)
    comptime assert transpose_b, "Only support transposed B in layout"
    comptime assert num_threads == 128 or num_threads == 256
    comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
    comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"

    var M = a_offsets[block_idx.z + 1] - a_offsets[block_idx.z]
    comptime N = c.static_shape[1]
    comptime K = static_K

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]  # BM
    comptime MMA_N = mma_shape[1]  # BN
    comptime MMA_K = mma_shape[2]  # 16
    comptime num_m_mmas = BM // MMA_M
    comptime num_n_mmas = BN // MMA_N
    comptime num_k_mmas = BK // MMA_K

    var a_start_row = a_offsets[block_idx.z]
    var expert = expert_ids[block_idx.z]
    var b_start_row = expert * Int32(N)

    var m_start = block_idx.y * BM
    var n_start = block_idx.x * BN
    var a_m_start = Int(a_start_row) + m_start
    var b_n_start = Int(b_start_row) + n_start
    if m_start >= Int(M) or n_start >= N:
        return

    # we don't do the whole mma_shape_A vibes, rather, we directly declare it
    # tile_layout_k_major is cutlass equiv of tile_to_mma_shape
    # and sA_layout gets computed directly, by hand
    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()
    comptime sub_a_smem_layout = tile_layout_k_major[
        a_type, BM, 64, swizzle_mode=a_swizzle
    ]()
    comptime sub_b_smem_layout = tile_layout_k_major[
        b_type, BN, 64, swizzle_mode=b_swizzle
    ]()

    var a_smem = rebind[
        UnsafePointer[
            Scalar[a_type], UntrackedOrigin[mut=True], address_space=.SHARED
        ]
    ](
        external_memory[
            Scalar[a_type],
            address_space=.SHARED,
            alignment=128,
            name="tmem_test_dynamic_shared_memory",
        ]()
    )

    # a_smem_layout is a description of how tile is arranged in memory, and LayoutTensor is a pointer to memory + a layout, taking in a_smem as its pointer
    comptime sub_a_smem_tile_t = LayoutTensor[
        a_type,
        sub_a_smem_layout,
        MutUntrackedOrigin,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime sub_b_smem_tile_t = LayoutTensor[
        b_type,
        sub_b_smem_layout,
        MutUntrackedOrigin,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime a_size = a_smem_layout.size()
    comptime b_size = b_smem_layout.size()

    comptime assert (
        (a_size * size_of[a_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (b_size * size_of[b_type]()) % 16
    ) == 0, "preserve alignment"
    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()

    var a_smem_tile = TileTensor(a_smem, LTToTTLayout[a_smem_layout]())
    var b_smem_tile = TileTensor(b_smem, LTToTTLayout[b_smem_layout]())

    # Shared memory pointer to hold tensor memory address, after last smem pointer and expected smem size
    var ptr_tmem_addr = (b_smem + b_size).bitcast[UInt32]()

    comptime accum_type = get_accum_type[a_type]()

    comptime c_frag_size = MMA_M * MMA_N // num_threads  # MMA_M * MMA_N is the size of the accumulator, num_threads is the number of threads in the warp, c_frag_size is the num of elements in the accumulator per thread
    var c_frag: Array[
        Scalar[accum_type], c_frag_size
    ]  # array of accumulator elements

    comptime a_expected_bytes = a_size * size_of[a_type]()
    comptime b_expected_bytes = b_size * size_of[b_type]()
    comptime expected_bytes = a_expected_bytes + b_expected_bytes

    var tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    var mma_mbar = tma_mbar + 1

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = warp_id() == 0
    var elect_one_thread = thread_idx.x == 0
    comptime max_tmem_cols = 512

    # allocate all 2^18 bytes of smem for tcgen05, all 512 cols allocated
    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    # Create MmaOpSM100_SS instance to handle MMA operations
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

    for i in range(
        _num_iters
    ):  # K // BK, which is K // 64 or K // 128 depending on BK
        # so only one thread per CTA does the copy
        if elect_one_thread:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

            comptime for j in range(
                BK // 64
            ):  # so we do the copy in 64 chunks or 64 elements at a time (BK // 64). but hmm, we said that the K atom can only be 32 bytes (16 elements)
                comptime k = 64 * j
                comptime a_offset = a_smem_layout(IntTuple(0, k))
                comptime b_offset = b_smem_layout(IntTuple(0, k))
                comptime assert ((a_offset * size_of[a_type]()) % 128) == 0
                comptime assert ((b_offset * size_of[b_type]()) % 128) == 0
                var sub_a_smem_tile = sub_a_smem_tile_t(a_smem + a_offset)
                # the answer to the above comment. # The descriptor layout i.e. data per copy can be smaller than the shared memory
                # tile shape due to WGMMA requirement. E.g. k-major no swizzle WGMMA BM x 16B to be
                # one continuous chunk in shared memory. We need to break down tile shape in K by 16B.
                # so the async_copy takes care of that. TMA engine will copy the data from global tensor into smem tile A
                var k_start: Int = i * BK + k
                a_tma_op.async_copy(
                    sub_a_smem_tile,
                    tma_mbar[0],
                    (k_start, a_m_start),
                )
                var sub_b_smem_tile = sub_b_smem_tile_t(b_smem + b_offset)
                b_tma_op.async_copy(
                    sub_b_smem_tile,
                    tma_mbar[0],
                    (k_start, b_n_start),
                )
        # wait for the copy to finish
        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        # now we do the mma, again only one thread issues the instruction
        if elect_one_thread:
            # Use MmaOpSM100_SS to perform the MMA operation
            mma_op.mma(
                a_smem_tile,
                b_smem_tile,
                tmem_addr,
                init_c=(i == 0),  # Initialize C on first iteration
            )

            mma_op.commit(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    # eventually all of c has been accumulated, so we load it from tmem_addr into c_frag registers using tcgen05_ld
    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat=BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)

    tcgen05_load_wait()  # wait for the load to finish

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    comptime num_warps = num_threads // WARP_SIZE
    var warp_id = warp_id()

    comptime c_gmem_layout = Layout(IntTuple(UNKNOWN_VALUE, N), IntTuple(N, 1))
    comptime c_gmem_type = LayoutTensor[
        c_type,
        c_gmem_layout,
        MutAnyOrigin,
        layout_int_type=.int32,
        address_space=.GENERIC,
    ]

    # FIXME: A list literal initializer should be enough here, but somehow Mojo fails to infer that.
    var c_gmem_runtime_layout = RuntimeLayout[c_gmem_layout](
        Index(M, N), Index(N, 1)
    )

    var c_by_expert = c_gmem_type(
        c.ptr + a_start_row * UInt32(N), c_gmem_runtime_layout
    )

    var ctile, ctile_coords, _ = c_by_expert.tile_with_offset[BM, BN](
        block_idx.y, block_idx.x
    )
    comptime c_coord_type = type_of(ctile_coords)

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            var c_gmem_warp_tile, _c_gmem_warp_tile_coords, _ = (
                ctile.tile_with_offset[MMA_M // num_warps, MMA_N](
                    4 * m_mma + warp_id, n_mma
                )
            )
            var c_gmem_warp_tile_coords = ctile_coords + rebind[c_coord_type](
                _c_gmem_warp_tile_coords
            )

            var c_gmem_frag, _c_gmem_frag_coords, _ = (
                c_gmem_warp_tile.vectorize[1, 2]().distribute_with_offset[
                    Layout.row_major(8, 4)
                ](lane_id())
            )
            var new_c_gmem_frag_coords = rebind[c_coord_type](
                _c_gmem_frag_coords
            )
            new_c_gmem_frag_coords[1] *= 2
            var c_gmem_frag_coords = (
                c_gmem_warp_tile_coords + new_c_gmem_frag_coords
            )

            comptime num_vecs_m = c_gmem_frag.layout.shape[0].value()
            comptime num_vecs_n = c_gmem_frag.layout.shape[1].value()

            comptime for n_vec in range(num_vecs_n):
                comptime for m_vec in range(num_vecs_m):
                    comptime i_vec = n_vec * num_vecs_m + m_vec
                    comptime dst_idx = type_of(c_gmem_frag).layout(
                        IntTuple(m_vec, n_vec)
                    )
                    comptime dst_m_offset, dst_n_offset = divmod(dst_idx, N)
                    var m = UInt32(c_gmem_frag_coords[0] + dst_m_offset)
                    var n = UInt32(c_gmem_frag_coords[1] + dst_n_offset)

                    if m < M and n < UInt32(N):
                        var c_mn = SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()

                        comptime if elementwise_lambda_fn:
                            comptime alignment = align_of[SIMD[c_type, 2]]()
                            comptime epilogue = elementwise_lambda_fn.value()
                            epilogue[alignment=alignment](
                                (Int(a_start_row + m), Int(n)), c_mn
                            )
                        else:
                            c_gmem_frag[m_vec, n_vec] = rebind[
                                c_gmem_frag.element_type
                            ](c_mn)


def grouped_matmul_sm100[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    *,
    transpose_b: Bool = True,
    mma_shape: IndexList[3] = Index(64, 128, 16),
    block_tile_shape: IndexList[3] = Index(64, 128, 64),
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    max_num_tokens_per_expert: Int,
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Launches the SM100 grouped matmul kernel with TMA descriptors for ragged MoE matrix multiplication on Blackwell GPUs.
    """
    comptime num_experts = b.static_shape[0]
    comptime N = b.static_shape[1]
    comptime K = b.static_shape[2]

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime assert K % BK == 0
    comptime assert BK == 64

    # hard coded 64 for BK

    comptime a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime c_swizzle = TensorMapSwizzle.SWIZZLE_NONE
    # equivalent of cutlass tma atom a, it is a handle that is passed to async_copy, to accurately tell the TMA engine how to copy from global tensor a into smem tile A
    var a_tma_op = create_tensor_tile[Index(BM, BK), swizzle_mode=a_swizzle](
        ctx, a
    )
    var b_2d = TileTensor(b.ptr, row_major[num_experts * N, K]())
    var b_tma_op = create_tensor_tile[
        Index(BN, BK) if transpose_b else Index(BK, BN),
        swizzle_mode=b_swizzle,
    ](ctx, b_2d)
    comptime block_dim = 128
    comptime smem_use = (
        BM * size_of[a_type]() + BN * size_of[b_type]()
    ) * BK + 24

    comptime kernel = grouped_matmul_kernel_sm100[
        a_type,
        b_type,
        c_type,
        K,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        type_of(c).LayoutType,
        type_of(a_offsets).LayoutType,
        type_of(expert_ids).LayoutType,
        block_tile_shape,
        mma_shape,
        a_swizzle,
        b_swizzle,
        c_swizzle,
        transpose_b=transpose_b,
        num_threads=block_dim,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        a_offsets,
        expert_ids,
        c,
        Int32(ceildiv(K, BK)),
        grid_dim=(
            ceildiv(N, BN),
            ceildiv(max_num_tokens_per_expert, BM),
            num_active_experts,
        ),
        block_dim=(block_dim),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )


@__name(t"grouped_matmul_amd_{a_type}_{b_type}_{c_type}")
def grouped_matmul_amd_kernel_launcher[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    LayoutC: TensorLayout,
    LayoutA: TensorLayout,
    LayoutB: TensorLayout,
    AOffsetsLayout: TensorLayout,
    ExpertIdsLayout: TensorLayout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_tensor: TileTensor[mut=True, c_type, LayoutC, MutAnyOrigin],
    a_tensor: TileTensor[a_type, LayoutA, MutAnyOrigin],
    b_tensor: TileTensor[b_type, LayoutB, MutAnyOrigin],
    a_offsets: TileTensor[mut=False, .uint32, AOffsetsLayout, MutAnyOrigin],
    expert_ids: TileTensor[mut=False, .int32, ExpertIdsLayout, MutAnyOrigin],
    num_active_experts: Int32,
):
    """Computes the AMD GPU grouped matmul by dispatching per-expert tiles through ``AMDMatmul``, with separate zero-fill handling for inactive (``expert_id == -1``) blocks.

    For active experts, delegates the per-tile matmul (and optional
    elementwise epilogue) to ``AMDMatmul``. For inactive experts, zeroes the
    output row range and invokes the epilogue with zero values so that
    LoRA-style inactive blocks still produce a defined output.
    """
    comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
    comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"
    comptime assert transpose_b, "Only support transposed B in grouped matmul."

    var M = a_offsets[block_idx.z + 1] - a_offsets[block_idx.z]
    comptime N = c_tensor.static_shape[1]
    comptime K = b_tensor.static_shape[1]

    var expert_id = expert_ids[block_idx.z]
    var a_start_row = a_offsets[block_idx.z]

    var a_ptr = a_tensor.ptr + a_start_row * UInt32(K)
    var b_ptr = b_tensor.ptr + expert_id * Int32(N) * Int32(K)
    var c_ptr = c_tensor.ptr + a_start_row * UInt32(N)

    @always_inline
    @__parameter
    def elementwise_epilogue_fn_wrapper[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        comptime if elementwise_lambda_fn:
            comptime elementwise_epilogue = elementwise_lambda_fn.value()
            var batch_idx = IndexList[2](
                Int(a_start_row + UInt32(idx[0])), idx[1]
            )
            elementwise_epilogue(batch_idx, val)

    # Only perform matmul if expert_id is not -1
    # AMD matmul kernel performs the epilogue function
    if expert_id != -1:
        var c_tile = TileTensor(c_ptr, row_major(Coord(Int(M), Idx[N])))
        var a_tile = TileTensor(a_ptr, row_major(Coord(Int(M), Idx[K])))
        var b_tile = TileTensor(b_ptr, row_major[N, K]())
        AMDMatmul[
            a_type,
            b_type,
            c_type,
            transpose_b,
            config,
            Optional[elementwise_epilogue_type](
                elementwise_epilogue_fn_wrapper
            ) if elementwise_lambda_fn else None,
        ].run[
            type_of(c_tile).LayoutType,
            type_of(a_tile).LayoutType,
            type_of(b_tile).LayoutType,
            type_of(c_tile).Engine,
            type_of(a_tile).Engine,
            type_of(b_tile).Engine,
        ](
            c_tile, a_tile, b_tile
        )

    # Perform the epilogue function separately if expert_id is -1
    else:
        var c_tile = TileTensor(c_ptr, row_major(Coord(Int(M), Idx[N])))
        _ = c_tile.fill(0.0)

        comptime if elementwise_lambda_fn:
            comptime epilogue = elementwise_lambda_fn.value()

            comptime BM = config.block_tile_shape[0]
            comptime BN = config.block_tile_shape[1]
            comptime vec_width = simd_width_of[c_type]()
            comptime alignment = align_of[SIMD[c_type, vec_width]]()

            var block_m = block_idx.y
            var block_n = block_idx.x

            # Early exit if this block is completely outside the matrix bounds
            if UInt32(block_m * BM) >= M:
                return

            comptime threads_per_block = 256
            comptime elements_per_thread = ceildiv(BM * BN, threads_per_block)

            var tid: Int = thread_idx.x
            var thread_start = tid * elements_per_thread
            var thread_end = min(thread_start + elements_per_thread, BM * BN)

            var elements_to_process = thread_end - thread_start

            @always_inline
            def process_elements[width: Int](idx: Int) {mut}:
                var elem_idx = thread_start + idx
                var tile_row, tile_col = divmod(elem_idx, BN)
                var local_row: UInt32 = UInt32(block_m * BM + tile_row)
                var local_col: UInt32 = UInt32(block_n * BN + tile_col)

                if local_row < M:
                    var remaining_in_row = UInt32(N) - local_col
                    var remaining_in_tile_row = BN - tile_col
                    var actual_width = min(
                        width,
                        min(Int(remaining_in_row), remaining_in_tile_row),
                    )

                    if actual_width == width and local_col + UInt32(
                        width
                    ) <= UInt32(N):
                        var zero_vec = SIMD[c_type, width](0.0)
                        epilogue[
                            dtype=c_type,
                            width=width,
                            alignment=align_of[SIMD[c_type, width]](),
                        ]((Int(local_row), Int(local_col)), zero_vec)
                    else:
                        for i in range(actual_width):
                            if local_col + UInt32(i) < UInt32(N):
                                var zero_scalar = SIMD[c_type, 1](0.0)
                                epilogue[dtype=c_type, width=1, alignment=1](
                                    (
                                        Int(local_row),
                                        Int(local_col + UInt32(i)),
                                    ),
                                    zero_scalar,
                                )

            vectorize[vec_width](elements_to_process, process_elements)


@always_inline
def dispatch_amd_matmul_by_block_shape[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool,
    N: Int,
    K: Int,
    default_block_tile_shape: IndexList[3],
    use_heuristic: Bool = False,
    *,
    LauncherFnType: ImplicitlyCopyable
    & def[
        config: MatmulConfig[a_type, b_type, c_type, transpose_b]
    ]() raises -> None,
](launcher_fn: LauncherFnType, M: Int, ctx: DeviceContext) raises:
    """Dispatches to the best kernel configuration based on runtime M dimension.
    """

    comptime if use_heuristic:
        comptime block_shape_list = _amdgpu_matmul_build_block_shape_list[N]()

        # Auto-tune block shape selection: Find the configuration that minimizes
        # SM idle time by scoring how evenly work distributes across all SMs.
        # Lower score = better load balance (fewer idle SMs in the last wave).
        var best_idx = -1
        var best_score = Int.MAX
        var sm_count = ctx.default_device_info.sm_count

        comptime for i in range(len(block_shape_list)):
            comptime block_shape = block_shape_list[i]
            comptime block_m = block_shape[0]
            comptime block_n = block_shape[1]
            comptime n_blocks = ceildiv(N, block_n)

            var m_blocks = ceildiv(M, block_m)
            var total_blocks = m_blocks * n_blocks
            var batch, extra = divmod(total_blocks - 1, sm_count)
            var score = batch * sm_count + (sm_count - extra - 1)

            if score < best_score:
                best_idx = i
                best_score = score

        # Dispatch to the best configuration if found
        comptime for i in range(len(block_shape_list)):
            if best_idx == i:
                comptime config = _amdgpu_matmul_config_from_block_shape[
                    c_type,
                    a_type,
                    b_type,
                    transpose_b,
                    K,
                    pdl_level=PDLLevel(),
                ](block_shape_list[i])
                launcher_fn[config]()
                return

    # Fallback to default config
    @always_inline
    @__parameter
    def default_config_launcher[
        block_m: Int,
        block_n: Int,
        block_k: Int,
    ]() raises:
        comptime default_config = MatmulConfig[
            a_type, b_type, c_type, transpose_b
        ](
            block_tile_shape=Index(block_m, block_n, block_k),
            warp_tile_shape=Index(
                block_m // 2,
                block_n // 2,
                block_k,
            ),
            num_pipeline_stages=1,
            num_k_partitions=1,
        )
        launcher_fn[default_config]()

    # auto-tuned sizes
    if M == 128 and N == 256 and K == 256:
        default_config_launcher[32, 32, 128]()
    elif M == 256 and N == 512 and K == 1024:
        default_config_launcher[32, 32, 128]()
    elif M == 384 and N == 768 and K == 1024:
        default_config_launcher[32, 64, 128]()
    elif M == 1977 and N == 192 and K == 1024:
        default_config_launcher[64, 96, 128]()
    elif M == 1977 and N == 1280 and K == 1024:
        default_config_launcher[96, 96, 64]()
    else:
        default_config_launcher[64, 64, 64]()


def grouped_matmul_amd[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    *,
    transpose_b: Bool = True,
    block_tile_shape: IndexList[3] = Index(128, 128, 64),
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[a_type, address_space=.GENERIC, ...],
    a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    max_num_tokens_per_expert: Int,
    b: TileTensor[b_type, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Launches the AMD grouped matmul kernel, selecting the best block-tile configuration for the runtime M dimension via ``dispatch_amd_matmul_by_block_shape``.
    """
    comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
    comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"
    comptime assert b.flat_rank == 3, "b must be rank 3"

    if num_active_experts == 0 or max_num_tokens_per_expert == 0:
        return

    comptime num_experts = b.static_shape[0]
    comptime N = b.static_shape[1]
    comptime K = b.static_shape[2]

    var total_M = 0
    for i in range(num_active_experts):
        total_M += Int(a_offsets[i + 1] - a_offsets[i])

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime assert K % BK == 0

    # Reshape b from (num_experts, N, K) to (num_experts*N, K) for 2D view
    var b_2d = TileTensor(b.ptr, row_major[num_experts * N, K]())

    comptime block_dim = 256

    @always_inline
    def launch_kernel[
        config: MatmulConfig[a_type, b_type, c_type, transpose_b]
    ]() raises {
        c,
        a,
        b_2d,
        a_offsets,
        expert_ids,
        num_active_experts,
        max_num_tokens_per_expert,
        ctx,
    }:
        comptime kernel = grouped_matmul_amd_kernel_launcher[
            c_type,
            a_type,
            b_type,
            type_of(c).LayoutType,
            type_of(a).LayoutType,
            type_of(b_2d).LayoutType,
            type_of(a_offsets).LayoutType,
            type_of(expert_ids).LayoutType,
            transpose_b,
            config,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ]
        ctx.enqueue_function[kernel](
            c,
            a,
            b_2d,
            a_offsets,
            expert_ids,
            Int32(num_active_experts),
            grid_dim=(
                ceildiv(N, config.block_tile_shape[1]),
                ceildiv(max_num_tokens_per_expert, config.block_tile_shape[0]),
                num_active_experts,
            ),
            block_dim=(block_dim),
        )

    # Dispatch to the best configuration based on runtime dimensions
    dispatch_amd_matmul_by_block_shape[
        c_type,
        a_type,
        b_type,
        transpose_b,
        N,
        K,
        block_tile_shape,
    ](launch_kernel, total_M, ctx)


# ===----------------------------------------------------------------------=== #
# Entry Point and Dispatch (TileTensor overloads)
# ===----------------------------------------------------------------------=== #


@always_inline
def grouped_matmul[
    *,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    a_plane_splits: IndexList[2] = Index(0, 0),
](
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[address_space=.GENERIC, ...],
    b: TileTensor[address_space=.GENERIC, ...],
    a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    expert_usage_stats: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    ctx: DeviceContext,
    host_stats: Optional[Tuple[Int, Int]] = None,
) raises:
    """TileTensor implementation of `grouped_matmul`.

    `expert_usage_stats` is a rank-1 uint32 *device* tensor laid out as
    `[max_tokens_per_expert, num_active_experts]` (from `moe_create_indices`).
    The SM100 persistent kernel reads `num_active_experts` from it on-device.
    The SM90/AMD/naive paths need host scalars for their launch grid, so they
    resolve them via `resolve_usage_stats` below.

    `host_stats` is an optional `(max_tokens_per_expert, num_active_experts)`
    pair already known on the host. When set, the SM90/AMD/naive paths use it
    directly instead of copying `expert_usage_stats` back from the device; the
    host-scalar overload passes it so those callers skip the copy.
    """
    comptime assert c.rank == 2 and c.flat_rank == 2
    comptime assert a.rank == 2 and a.flat_rank == 2
    comptime assert b.rank == 3 and b.flat_rank == 3
    comptime assert a_offsets.rank == 1 and a_offsets.flat_rank == 1
    comptime assert expert_ids.rank == 1 and expert_ids.flat_rank == 1

    comptime c_type = c.dtype
    comptime a_type = a.dtype
    comptime b_type = b.dtype

    comptime is_expert_shape_static = (
        b.static_shape[0] != UNKNOWN_VALUE
        and b.static_shape[1] != UNKNOWN_VALUE
        and b.static_shape[2] != UNKNOWN_VALUE
        and a.static_shape[1] != UNKNOWN_VALUE
        and c.static_shape[1] != UNKNOWN_VALUE
    )
    # The SM90 and SM100 TMA/UMMA warp-specialized kernels only support a
    # 16-bit (or smaller) C output: their TMA store path is sized for the
    # 16-bit output tile and `blackwell_tma_umma_warp_specialized_kernel`
    # hard-asserts `c_type != float32`. A float32 C output (e.g. an
    # un-quantized float32 model running LoRA, where the LoRA grouped matmul
    # inherits the float32 activation dtype) must fall through to the naive
    # path, which accumulates in float32 and supports a float32 store + epilogue.
    comptime c_is_fp32 = c_type == DType.float32
    # The activation plane select (per-output-column-region) is implemented in
    # the SM100 persistent load stage and the naive kernel only. When it is set
    # (`a_plane_splits[0] > 0`), disable the SM90 and AMD paths so dispatch falls
    # through to naive on those targets (the same generic behavior as before this
    # feature existed).
    comptime a_plane_select_on = a_plane_splits[0] > 0
    comptime is_sm90_kernel_applicable = (
        ctx.default_device_info == H100
        and is_expert_shape_static
        and not c_is_fp32
        and not a_plane_select_on
    )
    comptime is_sm100_kernel_applicable = (
        _is_sm10x_gpu(ctx.default_device_info)
        and is_expert_shape_static
        and not c_is_fp32
    )

    # `grouped_matmul_amd` is only valid when `K` is aligned to `BK` and
    # at least `2 * BK`. If there's only a single K tile,
    # the 2-stage software pipeline will reprocess it causing incorrect outputs.
    comptime amd_bk = _bk_base[a_type, amd_kernel=True]()
    comptime static_K = b.static_shape[2]
    comptime is_amd_kernel_applicable = (
        has_amd_gpu_accelerator()
        and not has_amd_rdna_gpu_accelerator()
        and is_expert_shape_static
        and static_K >= 2 * amd_bk
        and static_K % amd_bk == 0
        and not a_plane_select_on
    )

    # Apple weight-only FP8 (W8A16) MoE: bf16 activation x float8_e4m3fn weight.
    # Routes to the tiled simdgroup-MMA grouped kernel (the FP8 analog of the
    # dense Apple FP8 Linear) instead of the scalar-cast `naive_grouped_matmul`.
    # A fused epilogue is not applied by the tiled interior store, so shapes with
    # one fall through to naive (the MoE decode path passes none). Pre-M5 Apple
    # also falls through (the M5 native-fp8 MMA is unvalidated pre-M5). CUDA/AMD
    # builds compile this branch out (`has_apple_gpu_accelerator()` is comptime).
    comptime is_apple_fp8_moe_applicable = (
        has_apple_gpu_accelerator()
        and a_type == .bfloat16
        and b_type == .float8_e4m3fn
        and is_expert_shape_static
        and not elementwise_lambda_fn
    )

    @always_inline
    def description_fn() {var c, var a, var b, imm} -> String:
        # fmt: off
        return String(
            "(gpu",
            ";A=", Int(c.dim[0]()), "x", Int(a.dim[1]()), "x", a_type,
            ";C=", Int(c.dim[0]()), "x", Int(c.dim[1]()), "x", c_type,
            ";num_experts=", Int(b.dim[0]()),
            ")"
        )
        # fmt: on

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        get_static_string[
            "grouped_matmul_",
            String(a_type) + "x" + String(b_type) + "_to_" + String(c_type),
            "_has_epilogue" if elementwise_lambda_fn else "",
        ](),
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=get_safe_task_id(ctx),
    ):
        # Resolve the host scalars the SM90/AMD/naive launch grids need. Prefer
        # caller-supplied `host_stats`; otherwise copy them from
        # `expert_usage_stats` (device->host + sync). The SM100 persistent path
        # reads the device tensor directly and never calls this.
        @always_inline
        @__parameter
        def resolve_usage_stats() raises -> Tuple[Int, Int]:
            if host_stats:
                return host_stats.value()
            var host_buf = ctx.enqueue_create_host_buffer[.uint32](2)
            var dev_buf = DeviceBuffer[.uint32](
                ctx,
                expert_usage_stats.ptr.as_unsafe_any_origin(),
                2,
                owning=False,
            )
            ctx.enqueue_copy(dst_buf=host_buf, src_buf=dev_buf)
            ctx.synchronize()
            return (Int(host_buf[0]), Int(host_buf[1]))

        comptime if is_sm90_kernel_applicable:
            comptime static_N = c.static_shape[1]
            comptime BN = _find_largest_bn_for_sm90_matmul[a_type, static_N]()
            comptime mma_k = 32 // size_of[a_type]()
            comptime wgmma_shape = IndexList[3](64, BN, mma_k)

            var stats = resolve_usage_stats()
            var max_num_tokens_per_expert = stats[0]
            var num_active_experts = stats[1]

            grouped_matmul_sm90[
                wgmma_shape=wgmma_shape,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                c,
                a,
                a_offsets,
                max_num_tokens_per_expert,
                b,
                expert_ids,
                num_active_experts,
                ctx,
            )
        elif is_sm100_kernel_applicable:
            comptime N = b.static_shape[1]
            comptime K = b.static_shape[2]
            # Pad contiguous bytes to the UMMA minimum K (32 bytes)
            # so BK is always large enough for the UMMA instruction.
            comptime MMA_K = 32 // size_of[a_type]()
            comptime contiguous_bytes = max(K, MMA_K) * size_of[a_type]()

            def get_swizzle_mode(contiguous_bytes: Int) -> TensorMapSwizzle:
                if contiguous_bytes >= TensorMapSwizzle.SWIZZLE_128B.bytes():
                    return TensorMapSwizzle.SWIZZLE_128B
                elif contiguous_bytes >= TensorMapSwizzle.SWIZZLE_64B.bytes():
                    return TensorMapSwizzle.SWIZZLE_64B
                elif contiguous_bytes >= TensorMapSwizzle.SWIZZLE_32B.bytes():
                    return TensorMapSwizzle.SWIZZLE_32B
                else:
                    return TensorMapSwizzle.SWIZZLE_NONE

            comptime a_swizzle = get_swizzle_mode(contiguous_bytes)
            comptime b_swizzle = a_swizzle
            comptime BK = (a_swizzle.bytes() // size_of[a_type]())
            # For cta_group = 2, N must be divisible by 256 to ensure correct tiling and memory alignment for the kernel.
            comptime cta_group = 2 if N % 256 == 0 else 1
            comptime block_tile_shape = Index(128, 32 // cta_group, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * cta_group,
                block_tile_shape[1] * cta_group,
                MMA_K,
            )
            comptime cluster_shape = Index(cta_group, 1, 1)
            comptime transpose_b = True
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )

            grouped_matmul_sm100_persistent[
                c_type=c_type,
                a_type=a_type,
                b_type=b_type,
                transpose_b=transpose_b,
                config=config,
                cta_group=cta_group,
                a_swizzle=a_swizzle,
                b_swizzle=b_swizzle,
                elementwise_lambda_fn=elementwise_lambda_fn,
                a_plane_splits=a_plane_splits,
            ](
                c,
                a,
                a_offsets,
                b,
                expert_ids,
                expert_usage_stats,
                ctx,
            )
        elif is_amd_kernel_applicable:
            var stats = resolve_usage_stats()
            var max_num_tokens_per_expert = stats[0]
            var num_active_experts = stats[1]
            grouped_matmul_amd[elementwise_lambda_fn=elementwise_lambda_fn](
                c,
                a,
                a_offsets,
                max_num_tokens_per_expert,
                b,
                expert_ids,
                num_active_experts,
                ctx,
            )
        elif is_apple_fp8_moe_applicable:
            var stats = resolve_usage_stats()
            var max_num_tokens_per_expert = stats[0]
            var num_active_experts = stats[1]
            # M5 has the native fp8-operand simdgroup MMA; pre-M5 Apple falls
            # back to the dtype-generic naive kernel (still correct there).
            if ctx.compute_capability() == 5:
                # The tiled launcher is hard-typed W8A16 (bf16 A, fp8 B); rebind
                # the dispatch's abstract-dtype operands to the concrete types
                # the guard already proves (mirrors the dense Apple matmul
                # dispatch in `matmul/gpu/__init__.mojo`).
                comptime ABf16 = TileTensor[
                    .bfloat16,
                    type_of(a).LayoutType,
                    type_of(a).origin,
                    address_space=type_of(a).address_space,
                    linear_idx_type=type_of(a).linear_idx_type,
                    Engine=type_of(a).Engine,
                ]
                comptime BFp8 = TileTensor[
                    .float8_e4m3fn,
                    type_of(b).LayoutType,
                    type_of(b).origin,
                    address_space=type_of(b).address_space,
                    linear_idx_type=type_of(b).linear_idx_type,
                    Engine=type_of(b).Engine,
                ]
                enqueue_grouped_matmul2d_fp8[c_type=c_type](
                    c,
                    rebind[ABf16](a),
                    rebind[BFp8](b),
                    a_offsets,
                    expert_ids,
                    max_num_tokens_per_expert,
                    num_active_experts,
                    ctx,
                )
            else:
                naive_grouped_matmul[
                    elementwise_lambda_fn=elementwise_lambda_fn
                ](
                    c,
                    a,
                    b,
                    a_offsets,
                    expert_ids,
                    max_num_tokens_per_expert,
                    num_active_experts,
                    ctx,
                )
        else:
            var stats = resolve_usage_stats()
            var max_num_tokens_per_expert = stats[0]
            var num_active_experts = stats[1]
            naive_grouped_matmul[
                elementwise_lambda_fn=elementwise_lambda_fn,
                a_plane_splits=a_plane_splits,
            ](
                c,
                a,
                b,
                a_offsets,
                expert_ids,
                max_num_tokens_per_expert,
                num_active_experts,
                ctx,
            )


@always_inline
def grouped_matmul[
    *,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    a_plane_splits: IndexList[2] = Index(0, 0),
](
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[address_space=.GENERIC, ...],
    b: TileTensor[address_space=.GENERIC, ...],
    a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Host-scalar overload for callers (LoRA SGMV, tests, benchmarks) that hold
    the usage stats as scalars rather than a `moe_create_indices` device tensor.

    The SM100 persistent kernel reads the stats on-device, so the scalars are
    staged into a 2-element device buffer. They are also forwarded via
    `host_stats` so the SM90/AMD/naive paths read them directly rather than
    copying the staged buffer back to the host -- i.e. no host->device->host
    round-trip. The staged buffer is consumed only on SM100; this overload is
    off the MoE decode path, so its small staging cost does not matter.
    """
    var usage_stats_buf = ctx.enqueue_create_buffer[.uint32](2)
    with usage_stats_buf.map_to_host() as host:
        host[0] = UInt32(max_num_tokens_per_expert)
        host[1] = UInt32(num_active_experts)
    var expert_usage_stats = TileTensor(usage_stats_buf, row_major(Coord(2)))
    grouped_matmul[
        elementwise_lambda_fn=elementwise_lambda_fn,
        a_plane_splits=a_plane_splits,
    ](
        c,
        a,
        b,
        a_offsets,
        expert_ids,
        expert_usage_stats,
        ctx,
        host_stats=(max_num_tokens_per_expert, num_active_experts),
    )
    _ = usage_stats_buf^


@always_inline
def naive_grouped_matmul[
    *,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    a_plane_splits: IndexList[2] = Index(0, 0),
](
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, address_space=.GENERIC, ...],
    a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """TileTensor primary implementation of `naive_grouped_matmul`."""
    comptime assert c.rank == 2 and c.flat_rank == 2
    comptime assert a.rank == 2 and a.flat_rank == 2
    comptime assert b.rank == 3 and b.flat_rank == 3
    comptime assert a_offsets.rank == 1 and a_offsets.flat_rank == 1
    comptime assert expert_ids.rank == 1 and expert_ids.flat_rank == 1
    comptime assert transpose_b, "Only support transposed B in grouped matmul."

    comptime kernel = naive_grouped_matmul_kernel[
        c.dtype,
        a.dtype,
        b.dtype,
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a_offsets).LayoutType,
        type_of(expert_ids).LayoutType,
        elementwise_lambda_fn=elementwise_lambda_fn,
        a_plane_splits=a_plane_splits,
    ]
    ctx.enqueue_function[kernel](
        c,
        a.as_immut(),
        b.as_immut(),
        a_offsets,
        expert_ids,
        grid_dim=(
            ceildiv(Int(c.dim[1]()), 32),
            ceildiv(max_num_tokens_per_expert, 16),
            num_active_experts,
        ),
        block_dim=(32, 16, 1),
    )


# ===----------------------------------------------------------------------=== #
# Rowwise / per-token dynamic-scaled FP8 grouped matmul (SM100 B200)
# ===----------------------------------------------------------------------=== #
#
# Target: NVIDIA SM100 (B200). Correctness-first naive ``block_idx.z`` grouped
# kernel - NO persistent / TileScheduler / TMA path (deliberately deferred).
#
# This serves rowwise (per-output-channel) weight scales + per-token (colwise)
# dynamic activation scales - the compressed-tensors FP8 layout used by
# ``RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic`` - which the 128x128
# blockwise grouped FP8 kernel (``grouped_matmul_dynamic_scaled_fp8``) cannot
# express (it hard-asserts ``(1,128,128)`` scale granularity).
#
# The simplification vs. the blockwise kernel: there is NO per-K scale
# streaming. We accumulate the FP8 x FP8 products in fp32 over the FULL K, then
# apply a SINGLE epilogue scale:
#
#     out[t, n] = (sum_k a[t, k] * b[expert, n, k]) * a_scale[t] * b_scale[expert, n]
#
# Correctness invariants (mirrors the dense rowwise path in
# ``fp8_quantization.matmul_dynamic_scaled_fp8`` and the blockwise grouped
# reference ``naive_blockwise_scaled_fp8_grouped_matmul``):
#   1. ``a_scale`` is indexed by the GLOBAL ragged row ``a_start_row + m``,
#      NOT the per-expert local row ``m``.
#   2. ``b_scale`` is indexed by the REAL expert id ``expert_ids[z]``, NOT the
#      group index ``z`` (these differ for sparse routing).
#   3. Accumulate in fp32; scale ONCE after the full-K reduction. A partial-K
#      rescale would be silently wrong.
#   4. Empty groups (M == 0) produce no work; ``expert == -1`` (LoRA-style
#      inactive block) skips the matmul but the row range is still empty so no
#      output is written for it here.


@__name(t"grouped_matmul_rowwise_scaled_fp8_kernel_{c_type}_{a_type}_{b_type}")
def grouped_matmul_rowwise_scaled_fp8_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    accum_type: DType,
    CLayout: TensorLayout,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    AScalesLayout: TensorLayout,
    BScalesLayout: TensorLayout,
    AOffsetsLayout: TensorLayout,
    ExpertIdsLayout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    a_scales_engine: TensorEngine,
    b_scales_engine: TensorEngine,
    a_offsets_engine: TensorEngine,
    expert_ids_engine: TensorEngine,
    *,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, CLayout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[mut=False, a_type, ALayout, MutAnyOrigin, Engine=a_engine],
    b: TileTensor[mut=False, b_type, BLayout, MutAnyOrigin, Engine=b_engine],
    a_scales: TileTensor[
        mut=False,
        a_scales_type,
        AScalesLayout,
        MutAnyOrigin,
        Engine=a_scales_engine,
    ],
    b_scales: TileTensor[
        mut=False,
        b_scales_type,
        BScalesLayout,
        MutAnyOrigin,
        Engine=b_scales_engine,
    ],
    a_offsets: TileTensor[
        mut=False,
        .uint32,
        AOffsetsLayout,
        MutAnyOrigin,
        Engine=a_offsets_engine,
    ],
    expert_ids: TileTensor[
        mut=False,
        .int32,
        ExpertIdsLayout,
        MutAnyOrigin,
        Engine=expert_ids_engine,
    ],
):
    """Computes the naive grouped FP8 matmul with rowwise weight scales and per-token activation scales, accumulating in fp32 and applying a single post-reduction scale.

    For each token ``t`` in group ``g``'s row range and output channel ``n``,
    computes ``out[t, n] = (sum_k a[t, k] * b[expert, n, k]) * a_scale[t] * b_scale[expert, n]``.
    The ``a_scale`` is indexed by the global ragged row and ``b_scale`` by the
    real expert id, per the correctness invariants documented in the file
    header. Skips the matmul for ``expert == -1`` (inactive LoRA blocks).
    """
    comptime assert transpose_b, "Only support transposed B (B is [E, N, K])."
    comptime assert (
        accum_type == .float32
    ), "Only float32 accumulation is supported."
    comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
    comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"
    comptime assert c.flat_rank == 2, "c must be rank 2"
    comptime assert a.flat_rank == 2, "a must be rank 2"
    comptime assert b.flat_rank == 3, "b must be rank 3 ([E, N, K])"
    comptime assert a_scales.flat_rank == 2, "a_scales must be rank 2 ([T, 1])"
    comptime assert (
        b_scales.flat_rank == 3
    ), "b_scales must be rank 3 ([E, N, 1])"

    var M: Int = Int(a_offsets[block_idx.z + 1] - a_offsets[block_idx.z])
    var N = Int(b.dim[1]())
    var K = Int(b.dim[2]())

    var a_start_row = rebind[UInt32](a_offsets[block_idx.z])
    var a_by_expert = a.ptr + Int64(a_start_row) * Int64(K)

    var expert = rebind[Int32](expert_ids[block_idx.z])
    var b_by_expert = b.ptr + Int64(expert) * Int64(N) * Int64(K)

    # indices in current matmul
    var n = global_idx.x
    var m = global_idx.y

    if n >= N or m >= M:
        return

    # Global ragged row index for this token (correctness invariant #1).
    var m_global = a_start_row + UInt32(m)

    var accum = Scalar[accum_type](0.0)

    # ``expert == -1`` marks an inactive (LoRA) block; skip the matmul.
    if expert != -1:
        for k in range(K):
            accum += (
                a_by_expert[m * K + k].cast[accum_type]()
                * b_by_expert[n * K + k].cast[accum_type]()
            )

        # Apply the rowwise + per-token scale ONCE after the full-K reduction
        # (correctness invariant #3). a_scale is per global token (invariant
        # #1); b_scale is per (real expert, output channel) (invariant #2).
        comptime assert a_scales.flat_rank == 2
        comptime assert b_scales.flat_rank == 3
        var a_scale = rebind[Scalar[a_scales_type]](
            a_scales[Int(m_global), 0]
        ).cast[accum_type]()
        var b_scale = rebind[Scalar[b_scales_type]](
            b_scales[Int(expert), n, 0]
        ).cast[accum_type]()
        accum = accum * a_scale * b_scale

    comptime if elementwise_lambda_fn:
        comptime elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](
            Index(Int(m_global), n), accum.cast[c_type]()
        )
    else:
        var c_by_expert = c.ptr + Int64(a_start_row) * Int64(N)
        c_by_expert[m * N + n] = accum.cast[c_type]()


@always_inline
def grouped_matmul_rowwise_dynamic_scaled_fp8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_offsets_type: DType,
    expert_ids_type: DType,
    //,
    transpose_b: Bool = True,
    target: StaticString = "cpu",
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, Engine=_, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, Engine=_, ...],
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, Engine=_, ...],
    a_scales: TileTensor[
        mut=False, a_scales_type, address_space=.GENERIC, Engine=_, ...
    ],
    b_scales: TileTensor[
        mut=False, b_scales_type, address_space=.GENERIC, Engine=_, ...
    ],
    a_offsets: TileTensor[
        mut=False, a_offsets_type, address_space=.GENERIC, Engine=_, ...
    ],
    expert_ids: TileTensor[
        mut=False, expert_ids_type, address_space=.GENERIC, Engine=_, ...
    ],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Grouped (ragged MoE) FP8 matmul with rowwise weight + per-token act scales.

    Target: NVIDIA SM100 (B200). Correctness-first naive grouped kernel; no
    persistent / TMA path. Computes, for each token ``t`` in group ``g``'s row
    range and each output channel ``n``::

        out[t, n] = (sum_k a[t, k] * b[expert_ids[g], n, k])
                    * a_scale[t] * b_scale[expert_ids[g], n]

    accumulated in fp32 with a single post-reduction scale.

    Parameters:
        c_type: Output dtype (typically ``bfloat16``).
        a_type: Activation dtype (``float8_e4m3fn``).
        b_type: Weight dtype (``float8_e4m3fn``).
        a_scales_type: Per-token activation scale dtype (``float32``).
        b_scales_type: Per-channel weight scale dtype (``float32``).
        a_offsets_type: Ragged-offset dtype (``uint32``).
        expert_ids_type: Expert-id dtype (``int32``).
        transpose_b: Must be ``True``; ``b`` is ``[E, N, K]``.
        target: Compilation target string.
        elementwise_lambda_fn: Optional output epilogue applied with the
            ``(global_row, n)`` index.

    Args:
        c: Output ``[total_tokens, N]``.
        a: Activations ``[total_tokens, K]``.
        b: Weights ``[num_experts, N, K]`` (already transposed; K innermost).
        a_scales: Per-token activation scales ``[total_tokens, 1]``.
        b_scales: Per-channel weight scales ``[num_experts, N, 1]``.
        a_offsets: Ragged row offsets ``[num_active_experts + 1]``.
        expert_ids: Real expert ids ``[num_active_experts]``.
        max_num_tokens_per_expert: Max tokens routed to any active expert.
        num_active_experts: Number of active experts (groups).
        ctx: Device context.
    """
    comptime assert c.rank == 2 and c.flat_rank == 2
    comptime assert a.rank == 2 and a.flat_rank == 2
    comptime assert b.rank == 3 and b.flat_rank == 3
    comptime assert a_scales.rank == 2 and a_scales.flat_rank == 2
    comptime assert b_scales.rank == 3 and b_scales.flat_rank == 3
    comptime assert a_offsets.rank == 1 and a_offsets.flat_rank == 1
    comptime assert expert_ids.rank == 1 and expert_ids.flat_rank == 1

    comptime assert transpose_b, "Only support transpose_b = True."
    comptime assert (
        a_type == b_type == .float8_e4m3fn
    ), "input A and B dtype should be float8_e4m3fn"
    comptime assert (
        a_scales_type == .float32 and b_scales_type == .float32
    ), "A and B scales must be float32 for rowwise/per-token granularity"
    comptime assert a_offsets_type == .uint32, (
        "Only uint32 is supported for a_offsets in grouped rowwise scaled fp8"
        " matmul"
    )
    comptime assert expert_ids_type == .int32, (
        "Only int32 is supported for expert_ids in grouped rowwise scaled fp8"
        " matmul"
    )
    comptime assert is_gpu[target](), (
        "grouped rowwise dynamic scaled fp8 matmul only supports GPUs with"
        " native FP8 support"
    )

    if num_active_experts == 0 or max_num_tokens_per_expert == 0:
        return

    comptime accum_type = get_accum_type[a_type]()

    comptime kernel = grouped_matmul_rowwise_scaled_fp8_kernel[
        c_type,
        a_type,
        b_type,
        a_scales_type,
        b_scales_type,
        accum_type,
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a_scales).LayoutType,
        type_of(b_scales).LayoutType,
        type_of(a_offsets).LayoutType,
        type_of(expert_ids).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
        type_of(a_scales).Engine,
        type_of(b_scales).Engine,
        type_of(a_offsets).Engine,
        type_of(expert_ids).Engine,
        transpose_b=transpose_b,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        get_static_string[
            "grouped_matmul_rowwise_dynamic_scaled_fp8_",
            String(a_type) + "x" + String(b_type) + "_to_" + String(c_type),
        ](),
        task_id=get_safe_task_id(ctx),
    ):
        ctx.enqueue_function[kernel](
            c,
            a.as_immut(),
            b.as_immut(),
            a_scales.as_immut(),
            b_scales.as_immut(),
            a_offsets,
            expert_ids,
            grid_dim=(
                ceildiv(Int(c.dim[1]()), 32),
                ceildiv(max_num_tokens_per_expert, 16),
                num_active_experts,
            ),
            block_dim=(32, 16, 1),
        )


@always_inline
def grouped_matmul_vendor[
    *,
    transpose_b: Bool = True,
    use_tf32: Bool = False,
](
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, address_space=.GENERIC, ...],
    a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """TileTensor primary implementation of `grouped_matmul_vendor`."""
    comptime assert c.rank == 2 and c.flat_rank == 2
    comptime assert a.rank == 2 and a.flat_rank == 2
    comptime assert b.rank == 3 and b.flat_rank == 3
    comptime assert a_offsets.rank == 1 and a_offsets.flat_rank == 1
    comptime assert expert_ids.rank == 1 and expert_ids.flat_rank == 1
    comptime assert transpose_b, "Only support transposed B in grouped matmul."
    comptime assert (
        a.dtype == b.dtype
    ), "A and B must have the same dtype for vendor BLAS"

    comptime c_type = c.dtype
    comptime a_type = a.dtype
    comptime b_type = b.dtype

    def _ri(v: Int) -> Int64:
        return Int64(v)

    # Extract dimensions from TileTensors directly.
    var c_N = Int(c.dim[1]())
    var a_K = Int(a.dim[1]())
    var b_N = Int(b.dim[1]())
    var b_K = Int(b.dim[2]())

    @always_inline
    def vendor_description_fn() {var c, var a, var b, imm} -> String:
        # fmt: off
        return String(
            "(gpu",
            ";A=", Int(c.dim[0]()), "x", Int(a.dim[1]()), "x", a_type,
            ";C=", Int(c.dim[0]()), "x", Int(c.dim[1]()), "x", c_type,
            ";num_experts=", Int(b.dim[0]()),
            ";num_active_experts=", num_active_experts,
            ";max_num_tokens_per_expert=", max_num_tokens_per_expert,
            ";transpose_b=", transpose_b,
            ")"
        )
        # fmt: on

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        get_static_string[
            "grouped_matmul_vendor_",
            String(a_type) + "x" + String(b_type) + "_to_" + String(c_type),
        ](),
        Trace[TraceLevel.OP]._get_detail_str(vendor_description_fn),
        task_id=get_safe_task_id(ctx),
    ):
        for i in range(num_active_experts):
            var expert_id = expert_ids.raw_load(i)

            var token_start = a_offsets.raw_load(i)
            var token_end = a_offsets.raw_load(i + 1)
            var num_tokens = Int(token_end - token_start)

            # Skip if no tokens for this expert
            if num_tokens <= 0:
                continue

            # Handle experts with expert_id = -1 by writing zeros
            if expert_id < 0:
                var c_ptr = c.ptr + token_start * UInt32(c_N)
                var buff = DeviceBuffer(
                    ctx, c_ptr, num_tokens * c_N, owning=False
                )
                ctx.enqueue_memset(buff, 0)
                continue

            # Create TileTensor views into the tensors for this expert
            var a_slice = TileTensor(
                a.ptr + token_start * UInt32(a_K),
                row_major(Coord(_ri(num_tokens), _ri(a_K))),
            )
            var b_slice = TileTensor(
                b.ptr + expert_id * Int32(b_N) * Int32(b_K),
                row_major(Coord(_ri(b_N), _ri(b_K))),
            )
            var c_slice = TileTensor(
                c.ptr + token_start * UInt32(c_N),
                row_major(Coord(_ri(num_tokens), _ri(c_N))),
            )

            vendor_matmul[use_tf32](
                ctx,
                c_slice,
                a_slice,
                b_slice,
                c_row_major=True,
                transpose_b=transpose_b,
            )
