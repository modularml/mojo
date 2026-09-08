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

from std.math import sqrt
from std.math.uutils import udivmod
from std.memory import bitcast
from std.sys import size_of

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu import WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import block_rank_in_cluster
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import block_idx, lane_id, thread_idx, warp_id as get_warp_id
from max.gpu.memory import external_memory
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.compute.arch.tcgen05 import *
from layout import IntTuple, Layout, LayoutTensor
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
)
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type, max_finite
from std.utils.static_tuple import StaticTuple


def cpu_matmul_naive[
    *, transpose_a: Bool, transpose_b: Bool
](C: LayoutTensor[mut=True, ...], A: LayoutTensor, B: LayoutTensor):
    comptime M = C.layout[0].size()
    comptime N = C.layout[1].size()
    # layout_a is M x K
    comptime layout_a = A.layout.transpose() if transpose_a else A.layout
    # layout_b is K x N
    comptime layout_b = B.layout.transpose() if transpose_b else B.layout
    comptime K = layout_a[1].size()
    comptime assert M == layout_a[0].size(), String(
        "C.M = ", M, "; A.M = ", layout_a[0].size()
    )
    comptime assert N == layout_b[1].size(), String(
        "C.N = ", M, "; B.N = ", layout_b[1].size()
    )
    comptime assert K == layout_b[0].size(), String(
        "A.K = ", K, "; B.K = ", layout_b[0].size()
    )
    for n in range(N):
        for m in range(M):
            var acc: Float32 = 0.0
            for k in range(K):
                var a_idx: Int

                comptime if transpose_a:
                    a_idx = k * M + m
                else:
                    a_idx = m * K + k
                var b_idx: Int

                comptime if transpose_b:
                    b_idx = n * K + k
                else:
                    b_idx = k * N + n
                acc += (
                    A.ptr.load(a_idx).cast[.float32]()
                    * B.ptr.load(b_idx).cast[.float32]()
                )
            var c_idx = m * N + n
            C.ptr.store(c_idx, acc.cast[C.dtype]())


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
def tma_umma_kernel_ss[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    c_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_a: Bool = False,
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    num_threads: Int = 128,
](
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters_dev: Int32,
):
    var num_iters = Int(num_iters_dev)
    comptime assert num_threads == 128 or num_threads == 256
    comptime assert (
        a_type == b_type and a_type == .float8_e4m3fn
    ), "a_type and b_type must be the same and float8_e4m3fn type"

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = BM // MMA_M
    comptime num_n_mmas = BN // MMA_N
    comptime num_k_mmas = BK // MMA_K

    comptime a_k_major = not transpose_a
    comptime b_k_major = transpose_b
    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]() if a_k_major else tile_layout_mn_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if b_k_major else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    var a_smem = rebind[
        MutPointer[Scalar[a_type], address_space=.SHARED, MutUntrackedOrigin]
    ](
        external_memory[
            Scalar[a_type],
            address_space=.SHARED,
            alignment=128,
            name="tmem_test_dynamic_shared_memory",
        ]()
    )
    comptime a_smem_tile_t = LayoutTensor[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime b_smem_tile_t = LayoutTensor[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
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

    var a_smem_tile = a_smem_tile_t(a_smem.as_unsafe_any_origin())
    var b_smem_tile = b_smem_tile_t(b_smem.as_unsafe_any_origin())

    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (b_smem + b_size).bitcast[UInt32]()

    comptime accum_type = get_accum_type[a_type]()

    comptime c_frag_size = MMA_M * MMA_N // num_threads
    var c_frag: Array[Scalar[accum_type], c_frag_size]

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

    var elect_one_warp = get_warp_id() == 0
    var elect_one_thread = thread_idx.x == 0
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    comptime max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    comptime if num_threads > 128:
        if thread_idx.x >= 128:
            tmem_addr += 1 << 20  # offset for lane 16

    comptime a_canonical_layout = tile_to_descriptor[
        a_type, a_smem_layout, is_k_major=a_k_major
    ]()
    comptime b_canonical_layout = tile_to_descriptor[
        b_type, b_smem_layout, is_k_major=b_k_major
    ]()
    comptime a_stride01 = a_canonical_layout[0].stride[1].value()
    comptime a_stride11 = a_canonical_layout[1].stride[1].value()
    comptime aSBO = (
        a_stride01 if a_k_major
        or a_swizzle == TensorMapSwizzle.SWIZZLE_NONE else a_stride11
    ) * size_of[a_type]()
    comptime aLBO = (
        a_stride11 if a_k_major
        or a_swizzle == TensorMapSwizzle.SWIZZLE_NONE else a_stride01
    ) * size_of[a_type]()
    comptime b_stride01 = b_canonical_layout[0].stride[1].value()
    comptime b_stride11 = b_canonical_layout[1].stride[1].value()
    comptime bSBO = (
        b_stride01 if b_k_major
        or b_swizzle == TensorMapSwizzle.SWIZZLE_NONE else b_stride11
    ) * size_of[b_type]()
    comptime bLBO = (
        b_stride11 if b_k_major
        or b_swizzle == TensorMapSwizzle.SWIZZLE_NONE else b_stride01
    ) * size_of[b_type]()

    var adesc = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](a_smem_tile.ptr)
    var bdesc = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](b_smem_tile.ptr)

    var idesc = UMMAInsDescriptor[UMMAKind.KIND_F8F6F4].create[
        accum_type,
        a_type,
        b_type,
        Index[dtype=.uint32](mma_shape[0], mma_shape[1]),
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    ]()

    for i in range(num_iters):
        if elect_one_thread:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

            var m = block_idx.y * BM
            var n = block_idx.x * BN
            var k = i * BK
            a_tma_op.async_copy(
                a_smem_tile,
                tma_mbar[0],
                (m, k) if transpose_a else (k, m),
            )
            b_tma_op.async_copy(
                b_smem_tile,
                tma_mbar[0],
                (k, n) if transpose_b else (n, k),
            )

        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        if elect_one_thread:
            if i == 0:
                mma[c_scale=0](adesc, bdesc, tmem_addr, idesc)

                comptime for j in range(1, num_k_mmas):
                    comptime idx = IntTuple(0, MMA_K * j)
                    comptime a_offset = a_smem_layout(idx) * size_of[a_type]()
                    comptime b_offset = b_smem_layout(idx) * size_of[b_type]()
                    mma[c_scale=1](
                        adesc + a_offset, bdesc + b_offset, tmem_addr, idesc
                    )
            else:
                comptime for j in range(num_k_mmas):
                    comptime idx = IntTuple(0, MMA_K * j)
                    comptime a_offset = a_smem_layout(idx) * size_of[a_type]()
                    comptime b_offset = b_smem_layout(idx) * size_of[b_type]()
                    mma[c_scale=1](
                        adesc + a_offset, bdesc + b_offset, tmem_addr, idesc
                    )

            mma_arrive(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat=BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)

    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    comptime num_warps = num_threads // WARP_SIZE
    var warp_id = get_warp_id()

    comptime if num_threads > 128:
        var warp_id_q, warp_id_r = udivmod(warp_id, 4)
        warp_id = 2 * warp_id_r + warp_id_q

    var ctile = c.tile[BM, BN](block_idx.y, block_idx.x)

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            var c_gmem_warp_tile = ctile.tile[MMA_M // num_warps, MMA_N](
                4 * m_mma + warp_id, n_mma
            )

            var c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](lane_id())

            comptime num_vecs_m = c_gmem_frag.layout.shape[0].value()
            comptime num_vecs_n = c_gmem_frag.layout.shape[1].value()

            comptime for n_vec in range(num_vecs_n):
                comptime for m_vec in range(num_vecs_m):
                    comptime i_vec = n_vec * num_vecs_m + m_vec

                    c_gmem_frag[m_vec, n_vec] = rebind[
                        c_gmem_frag.element_type
                    ](
                        SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()
                    )


@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
def tma_umma_kernel_ts_fp8[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    c_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    num_threads: Int = 128,
](
    a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters_dev: Int32,
):
    var num_iters = Int(num_iters_dev)
    comptime assert num_threads == 128 or num_threads == 256
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = BM // MMA_M
    comptime num_n_mmas = BN // MMA_N

    comptime assert (
        num_m_mmas == 1 and num_n_mmas == 1
    ), "num_m_mmas and num_n_mmas must be 1"
    comptime assert (
        a_type == b_type and a_type == .float8_e4m3fn
    ), "a_type and b_type must be the same and float8_e4m3fn type"
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    var b_smem = rebind[
        MutPointer[Scalar[b_type], address_space=.SHARED, MutUntrackedOrigin]
    ](
        external_memory[
            Scalar[b_type],
            address_space=.SHARED,
            alignment=128,
            name="tmem_test_dynamic_shared_memory",
        ]()
    )
    comptime b_smem_tile_t = LayoutTensor[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]

    var b_smem_tile = b_smem_tile_t(b_smem.as_unsafe_any_origin())
    comptime b_size = b_smem_tile_t.layout.size()

    comptime accum_type = get_accum_type[a_type]()

    comptime assert (
        (b_size * size_of[b_type]()) % 16
    ) == 0, "preserve alignment"
    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (b_smem + b_size).bitcast[UInt32]()

    comptime c_frag_size = MMA_M * MMA_N // num_threads
    var c_frag: Array[Scalar[accum_type], c_frag_size]

    comptime b_expected_bytes = b_size * size_of[b_type]()
    comptime expected_bytes = b_expected_bytes

    var tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    var mma_mbar = tma_mbar + 1

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = get_warp_id() == 0
    var elect_one_thread = thread_idx.x == 0
    comptime max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    comptime if num_threads > 128:
        if thread_idx.x >= 128:
            tmem_addr += 1 << 20  # offset for lane 16
    var c_tmem: UInt32 = tmem_addr
    var a_tmem: UInt32 = tmem_addr + UInt32(MMA_N)

    comptime b_canonical_layout = tile_to_descriptor[
        b_type, b_smem_layout, is_k_major=transpose_b
    ]()
    comptime b_stride01 = b_canonical_layout[0].stride[1].value()
    comptime b_stride11 = b_canonical_layout[1].stride[1].value()
    comptime bSBO = (b_stride01 if transpose_b else b_stride11) * size_of[
        b_type
    ]()
    comptime bLBO = (b_stride11 if transpose_b else b_stride01) * size_of[
        b_type
    ]()

    var bdesc = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](b_smem_tile.ptr)

    var idesc = UMMAInsDescriptor[UMMAKind.KIND_F8F6F4].create[
        accum_type,
        a_type,
        b_type,
        Index[dtype=.uint32](mma_shape[0], mma_shape[1]),
        transpose_b=transpose_b,
    ]()

    comptime num_warps = num_threads // WARP_SIZE
    var warp_id = get_warp_id()

    comptime if num_threads > 128:
        var warp_id_q, warp_id_r = udivmod(warp_id, 4)
        warp_id = 2 * warp_id_r + warp_id_q

    comptime a_frag_size = BM * BK * size_of[a_type]() // 4 // num_threads
    var a_frag = Array[UInt32, a_frag_size](uninitialized=True)

    # FP8 elements are 1 byte each; load 8 elements per vector so each
    # SIMD vec is 8 bytes (== 2 uint32) and the split-into-uint32 pattern
    # used below stays identical to the BF16 ts kernel.
    comptime simd_size_a = 8

    for i in range(num_iters):
        # Load A from global memory to registers.
        var a_gmem_tile = a.tile[BM, BK](block_idx.y, i)
        var a_gmem_warp_tile = a_gmem_tile.tile[BM // num_warps, BK](warp_id, 0)
        var a_gmem_frag = a_gmem_warp_tile.vectorize[
            1, simd_size_a
        ]().distribute[Layout.row_major(8, 4)](lane_id())
        comptime num_vecs_m = a_gmem_frag.layout.shape[0].value()
        comptime num_vecs_k = a_gmem_frag.layout.shape[1].value()

        comptime for k in range(num_vecs_k):
            comptime for j in range(num_vecs_m):
                var vec = a_gmem_frag[j, k]
                comptime idx = k * num_vecs_m + j
                a_frag[2 * idx] = bitcast[.uint32, 1](vec.split()[0])
                a_frag[2 * idx + 1] = bitcast[.uint32, 1](vec.split()[1])

        tcgen05_st[
            datapaths=16,
            bits=256,
            repeat=BK * size_of[a_type]() // 4 // 8,
            pack=False,
        ](a_tmem, a_frag)

        # store_wait synchronizes within a warp. One warp could go ahead
        # while other warps are still storing to tmem.
        tcgen05_store_wait()
        barrier()

        # Load B by TMA
        if elect_one_thread:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

            b_tma_op.async_copy(
                b_smem_tile,
                tma_mbar[0],
                (i * BK, block_idx.x * BN) if transpose_b else (
                    block_idx.x * BN,
                    i * BK,
                ),
            )

        # Sync TMA and tcgen05_st because the latter can sync across warps.
        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        if elect_one_thread:
            # MMA_K elements of A occupy MMA_K * size_of[a_type]() bytes per
            # row, i.e. that many bytes / 4 columns of 32-bit tmem.
            comptime atmem_kstride = mma_shape[2] * size_of[a_type]() // 4
            if i == 0:
                mma[c_scale=0](a_tmem, bdesc, c_tmem, idesc)

                comptime for j in range(1, BK // mma_shape[2]):
                    comptime b_idx = IntTuple(MMA_N * 0, MMA_K * j)
                    comptime b_offset = b_smem_layout(b_idx) * size_of[b_type]()
                    mma[c_scale=1](
                        a_tmem + UInt32(j * atmem_kstride),
                        bdesc + b_offset,
                        c_tmem,
                        idesc,
                    )
            else:
                comptime for j in range(BK // mma_shape[2]):
                    comptime b_idx = IntTuple(MMA_N * 0, MMA_K * j)
                    comptime b_offset = b_smem_layout(b_idx) * size_of[b_type]()
                    mma[c_scale=1](
                        a_tmem + UInt32(j * atmem_kstride),
                        bdesc + b_offset,
                        c_tmem,
                        idesc,
                    )

            mma_arrive(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    # Each thread owns a row in c tile. This is inefficient but to
    # test the instruction shape.
    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat=BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](c_tmem)

    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    var ctile = c.tile[BM, BN](block_idx.y, block_idx.x)

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            var c_gmem_warp_tile = ctile.tile[MMA_M // num_warps, MMA_N](
                4 * m_mma + warp_id, n_mma
            )

            var c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](lane_id())

            comptime num_vecs_m = c_gmem_frag.layout.shape[0].value()
            comptime num_vecs_n = c_gmem_frag.layout.shape[1].value()

            comptime for n_vec in range(num_vecs_n):
                comptime for m_vec in range(num_vecs_m):
                    comptime i_vec = n_vec * num_vecs_m + m_vec

                    c_gmem_frag[m_vec, n_vec] = rebind[
                        c_gmem_frag.element_type
                    ](
                        SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()
                    )


def test_tma_umma[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    prob_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_a: Bool = False,
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    a_smem: Bool = True,
    cta_group: Int = 1,
](ctx: DeviceContext) raises:
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    print(
        "mma_"
        + ("s" if a_smem else "t")
        + "s_"
        + String(a_type)
        + "_"
        + String(b_type)
        + "_"
        + String(c_type)
        + " problem shape "
        + String(prob_shape)
        + " block tile "
        + String(block_tile_shape)
        + " transa="
        + String(transpose_a)
        + " transb="
        + String(transpose_b)
        + "; inst shape "
        + String(mma_shape)
        + " A "
        + (String(a_swizzle) if a_smem else "tmem")
        + " B "
        + String(b_swizzle)
    )

    comptime M = prob_shape[0]
    comptime N = prob_shape[1]
    comptime K = prob_shape[2]

    var a = ManagedLayoutTensor[
        a_type,
        Layout.row_major(K, M) if transpose_a else Layout.row_major(M, K),
    ](ctx)

    var a_extreme: Float32 = sqrt(sqrt(max_finite[a_type]().cast[.float32]()))
    random(
        a.tensor[update=False](),
        min=(-a_extreme).cast[a_type](),
        max=a_extreme.cast[a_type](),
    )

    comptime b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var b = ManagedLayoutTensor[b_type, b_layout](ctx)
    var b_col_major = ManagedLayoutTensor[b_type, Layout.row_major(N, K)](ctx)

    var b_extreme: Float32 = sqrt(sqrt(max_finite[b_type]().cast[.float32]()))
    random(
        b.tensor[update=False](),
        min=(-b_extreme).cast[b_type](),
        max=b_extreme.cast[b_type](),
    )

    var c = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    var c_ref = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    var a_tma_op = create_tensor_tile[
        Index(BK, BM) if transpose_a else Index(BM, BK),
        swizzle_mode=a_swizzle,
    ](ctx, a.device_tensor())
    var b_tma_op = create_tensor_tile[
        Index(BN, BK) if transpose_b else Index(BK, BN),
        swizzle_mode=b_swizzle,
    ](ctx, b.device_tensor())

    comptime block_dim = 2 * MMA_M

    comptime if a_smem:
        comptime smem_use = (BM + BN) * size_of[a_type]() * BK + 24
        comptime kernel = tma_umma_kernel_ss[
            a_type,
            b_type,
            c_type,
            type_of(a_tma_op).rank,
            type_of(a_tma_op).tile_shape,
            type_of(a_tma_op).desc_shape,
            type_of(b_tma_op).rank,
            type_of(b_tma_op).tile_shape,
            type_of(b_tma_op).desc_shape,
            Layout.row_major(M, N),
            block_tile_shape,
            mma_shape,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            cluster_shape=cluster_shape,
            a_swizzle=a_swizzle,
            b_swizzle=b_swizzle,
            num_threads=block_dim,
        ]
        ctx.enqueue_function[kernel](
            a_tma_op,
            b_tma_op,
            c.device_tensor(),
            Int32(K // BK),
            grid_dim=(N // BN, M // BM),
            block_dim=(block_dim),
            shared_mem_bytes=smem_use,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_use)
            ),
        )

    else:
        comptime smem_use = BN * size_of[b_type]() * BK + 24
        comptime kernel = tma_umma_kernel_ts_fp8[
            a_type,
            b_type,
            c_type,
            Layout.row_major(M, K),
            type_of(b_tma_op).rank,
            type_of(b_tma_op).tile_shape,
            type_of(b_tma_op).desc_shape,
            Layout.row_major(M, N),
            block_tile_shape,
            mma_shape,
            transpose_b=transpose_b,
            b_swizzle=b_swizzle,
            num_threads=block_dim,
        ]

        ctx.enqueue_function[kernel](
            a.device_tensor(),
            b_tma_op,
            c.device_tensor(),
            Int32(K // BK),
            grid_dim=(N // BN, M // BM),
            block_dim=(block_dim),
            shared_mem_bytes=smem_use,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_use)
            ),
        )

    comptime if not transpose_b:
        # NOTE: Matrix B should always be in col-major layout for cublasLt to work
        var b_host_col_major = b_col_major.tensor()
        var b_tensor = b.tensor()
        for i in range(N):
            for j in range(K):
                b_host_col_major[i, j] = b_tensor[j, i]

        vendor_blas.matmul(
            ctx,
            c_ref.device_tensor[update=False](),
            a.device_tensor[update=False](),
            b_col_major.device_tensor[update=True](),
            c_row_major=True,
            transpose_b=True,
        )

    elif M >= 64 and N >= 64 and K >= 64:
        vendor_blas.matmul(
            ctx,
            c_ref.device_tensor[update=False](),
            a.device_tensor[update=False](),
            b.device_tensor[update=False](),
            c_row_major=True,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )
    else:
        cpu_matmul_naive[transpose_a=transpose_a, transpose_b=transpose_b](
            c_ref.tensor[update=False](),
            a.tensor[update=False](),
            b.tensor[update=False](),
        )
        _ = c_ref.device_tensor()  # update host

    ctx.synchronize()

    var c_host = c.tensor()
    var c_host_ref = c_ref.tensor()

    for m in range(M):
        for n in range(N):
            # Increased tolerance for FP8/bfloat16 accumulation errors
            # FP8/bf16 matrix multiplication can have larger numerical errors
            # due to reduced precision in intermediate accumulations
            assert_almost_equal(
                c_host[m, n],
                c_host_ref[m, n],
                atol=0.01,
                rtol=0.01,
                msg=String(m) + ", " + String(n),
            )
            # print(m, n, c_host[m, n], c_host_ref[m, n])

    _ = a^
    _ = b^
    _ = b_col_major^
    _ = c^
    _ = c_ref^


def main() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.float8_e4m3fn
        comptime for swizzle in [
            TensorMapSwizzle.SWIZZLE_32B,
            TensorMapSwizzle.SWIZZLE_64B,
            TensorMapSwizzle.SWIZZLE_128B,
        ]:
            comptime for BK_scale in range(0, 2):
                comptime BK = (swizzle.bytes() // size_of[dtype]()) * (
                    1 + BK_scale
                )

                comptime for mma_size_scale in range(0, 2):
                    comptime MMA_M = 64 * (1 + mma_size_scale)
                    comptime MMA_K = 32

                    comptime for size_scale in range(1, 3):
                        comptime for transpose_b in range(0, 2):
                            test_tma_umma[
                                dtype,
                                dtype,
                                .bfloat16,
                                Index(
                                    MMA_M * size_scale,
                                    128 * size_scale,
                                    BK * size_scale,
                                ),
                                Index(MMA_M, 128, BK),
                                Index(MMA_M, 128, MMA_K),
                                a_swizzle=swizzle,
                                b_swizzle=swizzle,
                                transpose_b=Bool(transpose_b),
                            ](ctx)

                            test_tma_umma[
                                dtype,
                                dtype,
                                .bfloat16,
                                Index(
                                    MMA_M * size_scale,
                                    128 * size_scale,
                                    BK * size_scale,
                                ),
                                Index(MMA_M, 128, BK),
                                Index(MMA_M, 128, MMA_K),
                                b_swizzle=swizzle,
                                transpose_b=Bool(transpose_b),
                                a_smem=False,
                            ](ctx)

                            test_tma_umma[
                                dtype,
                                dtype,
                                .bfloat16,
                                Index(64, 128, 128),
                                Index(64, 128, 128),
                                Index(64, 128, 32),
                                b_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                                transpose_b=Bool(transpose_b),
                                a_smem=False,
                            ](ctx)

                            test_tma_umma[
                                dtype,
                                dtype,
                                .bfloat16,
                                Index(64, 128, 512),
                                Index(64, 128, 128),
                                Index(64, 128, 32),
                                b_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                                transpose_b=Bool(transpose_b),
                                a_smem=False,
                            ](ctx)

                            test_tma_umma[
                                dtype,
                                dtype,
                                .bfloat16,
                                Index(64, 128, 64),
                                Index(64, 128, 64),
                                Index(64, 128, 32),
                                b_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                                transpose_b=Bool(transpose_b),
                                a_smem=False,
                            ](ctx)
