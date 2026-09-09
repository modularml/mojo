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

from std.collections import Optional
from std.math import ceildiv
from std.math.uutils import ufloordiv, udivmod, uceildiv
from std.os import abort
from std.sys import size_of
from std.sys.info import align_of, simd_width_of

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
from max.gpu.memory import external_memory
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from layout.layout import size
from layout.layout_tensor import (
    LayoutTensorIter,
    copy_local_to_dram,
    copy_local_to_shared,
    copy_sram_to_dram,
)
from layout.swizzle import make_swizzle
from layout.tensor_core import get_fragment_size, get_mma_shape
from linalg.matmul.gpu._multistage_gemm_gpu import multistage_mma
from linalg.utils import elementwise_epilogue_type
from linalg.utils_gpu import block_swizzle
from std.testing import assert_almost_equal

from std.utils import StaticTuple
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type


struct BackToBackMatmulConfig[
    dst_type: DType,
    src_type: DType,
    transpose_b: Bool = False,
    transpose_c: Bool = False,
](TrivialRegisterPassable):
    # A is MxK
    # B is KxL
    # C is LxN
    # D is MxN
    # We block over M and L, yielding BM and BL.
    # BM x BN x BK
    var block_tile_shape: IndexList[3, element_type=.uint64]

    # WM x WN x WK
    var warp_tile_shape: IndexList[3, element_type=.uint64]

    var num_pipeline_stages: Int

    def num_warps_m(self) -> Int:
        return self.block_tile_shape[0] // self.warp_tile_shape[0]

    def num_warps_n(self) -> Int:
        return self.block_tile_shape[1] // self.warp_tile_shape[1]

    def num_threads(self) -> Int:
        return self.num_warps_m() * self.num_warps_n() * WARP_SIZE

    def shared_mem_usage(self, K: Int) -> Int:
        return (
            self.block_tile_shape[0] * K
            + (
                self.num_pipeline_stages
                * self.block_tile_shape[1]
                * self.block_tile_shape[2]
            )
        ) * size_of[Self.src_type]()

    def grid_dim(self, M: Int) -> IndexList[3]:
        return Index(1, uceildiv(M, self.block_tile_shape[0]), 1)

    def block_dim(self) -> IndexList[3]:
        return Index(self.num_threads(), 1, 1)

    def __init__(
        out self,
        block_tile_shape: IndexList[3, element_type=.uint64],
        warp_tile_shape: IndexList[3, element_type=.uint64],
        num_pipeline_stages: Int = 2,
    ):
        self.block_tile_shape = block_tile_shape
        self.warp_tile_shape = warp_tile_shape
        self.num_pipeline_stages = num_pipeline_stages


# Computes D = A * B * C
# Dimensions:
# A: M x K
# B: K x L
# C: L x N
#
# The algorithm is optimized for small K.
# E.g. in flash attention `softmax(Q*K')*V`
# typical values for M and K are 1k-8k and 64-128, respectively.
#
# In particular, we keep the entire `A[block_x,:]` segment in
# shared memory, to avoid loading it more than once.
#
# Additionally, in attention we have M=L, K=N.
#
# Materializing a MxL matrix when `K` is small greatly increases
# bandwidth requirements, hence the need for fusion here.
#
#
# The rough algorithm as described in the flash attention 2 paper
# (algorithm 1, forward pass):
# https://arxiv.org/pdf/2307.08691
# Block sizes: Bm
#
#
# We parallelize blocks across rows of A/D and columns of C/D
# One invocation evaluates `(A[block_x, :] * B) * C[:, block_y]`
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads())
    )
)
def b2b_gemm[
    d_type: DType,
    in_type: DType,
    d_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    transpose_b: Bool,
    transpose_c: Bool,
    config: BackToBackMatmulConfig[d_type, in_type, transpose_b, transpose_c],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    D: LayoutTensor[d_type, d_layout, MutAnyOrigin],
    A: LayoutTensor[in_type, a_layout, MutAnyOrigin],
    B: LayoutTensor[in_type, b_layout, MutAnyOrigin],
    C: LayoutTensor[in_type, c_layout, MutAnyOrigin],
):
    comptime assert (
        A.dtype in (DType.float32, DType.bfloat16)
        and A.dtype == B.dtype == C.dtype
    ), "B2B gemm only supports tf32 or BF16 mma"
    comptime assert (
        Int(a_layout.shape[1]) != UNKNOWN_VALUE
    ), "The number of columns of `A` must be known."

    comptime simd_size = simd_width_of[d_type]()

    # A is M x K
    # B is K x L
    # C is L x N
    # B is M x N
    var M: Int = D.dim[0]()
    var L: Int = B.dim[0 if transpose_b else 1]()
    # var K: Int = B.dim[1 if transpose_b else 0]()
    # TODO: allow dynamic `K`, so long as it still
    # fits in shared memory, we shouldn't require static.
    comptime K = Int(A.layout.shape[1])
    comptime N = Int(D.layout.shape[1])

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime WM = config.warp_tile_shape[0]
    comptime WN = config.warp_tile_shape[1]
    comptime num_pipeline_stages: Int = config.num_pipeline_stages
    comptime assert WN == BN
    # We have, roughly
    #
    #
    #     D[0:BM,0:BN] = 0
    #     for bl in range(ceildiv(L,BN)):
    #         AB[0:BM,0:BN] = 0
    #         for bk in range(ceildiv(K,BK)):
    #             AB += A[0:BM,0:BK] * B[0:BK,0:BN]
    #         for bk in range(ceildiv(BN,BK)):
    #             D += AB[0:BM,(0:BK)+bk*BK] * C[0:BK,0:BN]

    # To avoid recalculating `A*B`:
    comptime assert N == BN
    # TODO: lift this restriction
    comptime assert K % BK == 0, "K must be an integer multiple of BK"
    comptime assert BN % BK == 0, "BN must be an integer multiple of BK"
    comptime assert (
        K == BK
    ), "FIXME: currently, K == BK must be true, but that is a bug."

    var num_l_iter = uceildiv(L, BN)
    comptime num_warps_m = config.num_warps_m()
    comptime num_warps_n = config.num_warps_n()
    comptime num_threads = config.num_threads()

    var tid = thread_idx.x
    # var ln_id = lane_id()
    var warp_id = ufloordiv(tid, WARP_SIZE)

    # Only apply block swizzling for half precision types.
    comptime swizzle_block = in_type.is_half_float()

    # NOTE: the condition ( not (N // BN & 1)) is for a temporary solution
    # for solving mismatches in some shapes
    var block_idx = block_swizzle(
        (block_idx.x, block_idx.y),
        (grid_dim.x, grid_dim.y),
    ) if swizzle_block else Index(block_idx.x, block_idx.y)

    # Coordinates of the current warp.
    var warp_y, warp_x = udivmod(warp_id, num_warps_n)

    # Prepare shared memory buffers for A, B, and C.
    # We load our entire local `A` block into shared
    # memory and reuse it on each iteration.
    var a_smem = external_memory[
        Scalar[in_type],
        address_space=.SHARED,
        alignment=align_of[SIMD[in_type, simd_size]](),
    ]()
    comptime a_smem_size = BM * K  # single block
    var a_smem_iter = LayoutTensorIter[
        in_type,
        Layout.row_major(BM, BK),
        address_space=a_smem.address_space,
    ](
        a_smem,
        a_smem_size,
    )

    # Each pipeline stage has its own buffer.
    # We reuse `b_smem` for both `B` and `C`
    var b_smem = (a_smem + a_smem_size).bitcast[Scalar[in_type]]()
    comptime b_smem_size = num_pipeline_stages * BK * BN
    comptime BD_0 = BN if transpose_b else BK
    comptime BD_1 = BK if transpose_b else BN
    comptime b_smem_layout = Layout.row_major(BD_0, BD_1)
    var b_smem_iter = LayoutTensorIter[
        in_type,
        b_smem_layout,
        address_space=.SHARED,
        circular=True,
    ](b_smem, b_smem_size)
    # C may not have the same layout
    # (the common case is in fact `b_transpose and not c_transpose`)
    comptime CD_0 = BN if transpose_c else BK
    comptime CD_1 = BK if transpose_c else BN
    comptime c_smem_layout = Layout.row_major(CD_0, CD_1)

    # create input layout tensors A and Bv
    # global memory iterator for local block
    var a_gmem_iter = A.tiled_iterator[BM, BK, axis=1](block_idx[1], 0)
    # We iterate over the entire `b`
    #
    # var b_tile_coords = (block_idx[0], 0) if transpose_b else (0, block_idx[0])
    comptime b_tile_axis = 1 if transpose_b else 0
    comptime c_tile_axis = 1 if transpose_c else 0

    # Compute MMA config
    comptime mma_shape = get_mma_shape[in_type, get_accum_type[in_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_k_mmas = BK // MMA_K
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    comptime accum_type = get_accum_type[in_type]()
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime a_frag_size = frag_size[0]
    comptime b_frag_size = frag_size[1]
    # alias c_frag_size = b_frag_size
    comptime d_frag_size = frag_size[2]
    # (WM*WN // WARP_SIZE)
    comptime layout = Layout.row_major(num_m_mmas * num_n_mmas, d_frag_size)
    var d_reg_tile = (
        LayoutTensor[
            accum_type,
            layout,
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    var ab_reg_tile = LayoutTensor[
        accum_type,
        layout,
        MutAnyOrigin,
        address_space=.LOCAL,
    ].stack_allocation()
    for l in range(num_l_iter):
        _ = ab_reg_tile.fill(0)
        var b_tile_coords = (l, 0) if transpose_b else (0, l)
        var c_tile_coords = (0, l * (BN // BK)) if transpose_c else (
            l * (BN // BK),
            0,
        )
        # We fetch c_gmem_iter when done
        var b_gmem_iter = B.tiled_iterator[BD_0, BD_1, axis=b_tile_axis](
            b_tile_coords[0], b_tile_coords[1]
        )
        var c_gmem_iter = C.tiled_iterator[CD_0, CD_1, axis=c_tile_axis](
            c_tile_coords[0], c_tile_coords[1]
        )
        var num_rows_b = min(BN, L - BN * l)
        # FIXME: this is a lot of code duplication, for only
        # a few different branches within `multistage_mma`!
        # Maybe fetch `A` outside, and always use `prefetch_a=False`?
        if l == 0:
            # We need to fetch A on the first iteration
            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                transpose_b,
                b_next_smem_layout=c_smem_layout,
                next_op_b_iter_masked=type_of(c_gmem_iter).masked,
                continue_prefetch_b=True,
                prefetch_init=True,
                transpose_b_next=transpose_c,
                k_group_size=2,
            ](
                ab_reg_tile,
                a_gmem_iter,
                b_gmem_iter,
                a_smem_iter,
                b_smem_iter,
                uceildiv(K, BK),
                num_b_rows=num_rows_b,
                next_op_b_iter=c_gmem_iter.bitcast[in_type](),
            )
        else:
            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                transpose_b,
                b_next_smem_layout=c_smem_layout,
                next_op_b_iter_masked=type_of(c_gmem_iter).masked,
                continue_prefetch_b=True,
                prefetch_init=True,
                transpose_b_next=transpose_c,
                k_group_size=2,
            ](
                ab_reg_tile,
                a_smem_iter,  # don't prefetch a
                b_gmem_iter,
                a_smem_iter,
                b_smem_iter,
                uceildiv(K, BK),
                num_b_rows=num_rows_b,
                next_op_b_iter=c_gmem_iter.bitcast[in_type](),
            )
        # var ab0 = ab_reg_tile.ptr[0]
        # # ab_reg_tile.ptr[0] = ab0.cast[accum_type]()
        # _printf["ab_thread_idx %ld, val: %g\n"](tid, ab0)
        # print("thread_idx: ", tid, "val: ", ab0)
        # _printf["ab_thread_idx %ld\n"](tid)
        # Now we have `ab_reg_tile` as local memory, which
        # `multistage_mma` conveniently accepts.
        # NOTE that `multistage_mma` gets `a_type` from `a_smem_iter`.
        # Thus, if `ab_reg_tile.dtype != in_type` (e.g., if accumulate
        # `Float16` to `Float32), the downcasting should happen in
        # `multistage_mma`.
        # FIXME: need an elementwise def to apply to A*B!
        #
        # Also, we have
        # var a_reg_tiles = tb[a_type]().row_major[
        #     2 * num_m_mmas, a_frag_size
        # ]().local().alloc().split[2]()
        #
        # This gets copied to
        # num_m_mmas, a_frag_size
        # note
        # a_frag_size = d_frag_size * MMA_K // MMA_N
        var ab_iter = ab_reg_tile.tiled_iterator[
            MMA_K // MMA_N * num_m_mmas, d_frag_size
        ](0, 0)
        var c_smem_iter = b_smem_iter.reshape[c_smem_layout]()
        # Compensate for prefetch
        c_gmem_iter += num_pipeline_stages - 1
        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            transpose_c,
            next_op_b_iter_masked=False,
            b_next_smem_layout=b_smem_layout,
            prefetch_init=False,
            static_num_iters=BN // BK,
        ](
            d_reg_tile,
            ab_iter,
            c_gmem_iter,
            a_smem_iter,  # ignored
            c_smem_iter,
            uceildiv(N, BK),
            num_b_rows=num_rows_b,
        )

    # Map global memory tile down to thread.
    # we should have block_idx[0] == 0
    var d_gmem_tile = D.tile[BM, BN](block_idx[1], 0)
    var d_gmem_warp_tile = d_gmem_tile.tile[WM, WN](warp_y, warp_x)

    var ln_id = lane_id()
    # d_reg_tile = ab_reg_tile

    # Store FP32 mma results to half precision buffer in global memory.
    # Each thread's fragment has 2x2 fp32 values. Casting to half float and
    # directly storing to global memory results in 2 4B writes. Following cutlass,
    # we stage the fragments in shared memory so that each thread can store 16B.
    comptime if d_type.is_half_float():
        comptime swizzle = make_swizzle[
            num_rows=MMA_M // 2, row_size=WN, access_size=MMA_N
        ]()

        var accum_smem_warp_tile = LayoutTensor[
            accum_type,
            Layout.row_major(WM, WN),
            address_space=.SHARED,
        ](a_smem.bitcast[Scalar[accum_type]]() + warp_id * WM * WN)

        copy_local_to_shared[
            thread_layout=Layout.row_major(8, 4),
            swizzle=swizzle,
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            d_reg_tile.vectorize[1, 2]().transpose(),
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
            var d_gmem_frag = d_gmem_warp_tile.vectorize[
                1, simd_size
            ]().distribute[warp_layout](thread_idx.x)
            var d_smem_frag = accum_smem_warp_tile.vectorize[
                1, simd_size
            ]().distribute[warp_layout](thread_idx.x)
            var thread_offset = d_gmem_frag.distance(D.ptr)
            comptime num_stores_per_thread = type_of(d_gmem_frag).layout.size()
            comptime src_align = align_of[
                SIMD[accum_type, simd_width_of[accum_type]()]
            ]()
            comptime dst_align = align_of[SIMD[d_type, simd_size]]()

            var d_smem_frag_offset = d_smem_frag.distance(
                accum_smem_warp_tile.ptr
            )

            comptime for i in range(num_stores_per_thread):
                comptime src_idx = type_of(d_smem_frag).layout(i)
                comptime src_idx_base = src_idx % swizzle.size()
                comptime src_idx_diff = src_idx - src_idx_base
                var swizzled_idx = swizzle(
                    d_smem_frag_offset
                    + Scalar[d_smem_frag.linear_idx_type](src_idx_base)
                ) + Scalar[d_smem_frag.linear_idx_type](src_idx_diff)

                comptime dst_static_idx = type_of(d_gmem_frag).layout(i)

                var dst_idx: Int
                comptime if d_layout.all_dims_known():
                    dst_idx = dst_static_idx
                else:
                    dst_idx = Int(d_gmem_frag.runtime_layout(i))

                var m = Int(
                    (
                        thread_offset
                        + Scalar[d_gmem_frag.linear_idx_type](dst_idx)
                    )
                    // type_of(thread_offset)(N)
                )
                var n = Int(
                    (
                        thread_offset
                        + Scalar[d_gmem_frag.linear_idx_type](dst_idx)
                    )
                    % type_of(thread_offset)(N)
                )
                if m < M and n < N:
                    epilogue(
                        (m, n),
                        accum_smem_warp_tile.ptr.load[
                            width=simd_size, alignment=src_align
                        ](swizzled_idx).cast[d_type](),
                    )
        else:
            copy_sram_to_dram[
                thread_layout=Layout.row_major(
                    WARP_SIZE * simd_size // WN, WN // simd_size
                ),
                swizzle=swizzle,
            ](
                d_gmem_warp_tile.vectorize[1, simd_size](),
                accum_smem_warp_tile.vectorize[1, simd_size](),
            )

    # Store FP32 results to FP32 buffer in global memory.
    else:
        comptime if elementwise_lambda_fn:
            comptime epilogue = elementwise_lambda_fn.value()
            var d_gmem_frag = d_gmem_warp_tile.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](ln_id)
            var d_reg_frag = d_reg_tile.vectorize[1, 2]().transpose()
            var thread_offset = d_gmem_frag.distance(D.ptr)

            comptime for i in range(type_of(d_gmem_frag).layout.size()):
                comptime src_idx = d_reg_frag.layout(i)

                var dst_idx: Int
                comptime if d_layout.all_dims_known():
                    comptime dst_static_idx = type_of(d_gmem_frag).layout(i)
                    dst_idx = dst_static_idx
                else:
                    dst_idx = Int(d_gmem_frag.runtime_layout(i))

                var m = Int(
                    (
                        thread_offset
                        + Scalar[d_gmem_frag.linear_idx_type](dst_idx)
                    )
                    // type_of(thread_offset)(N)
                )
                var n = Int(
                    (
                        thread_offset
                        + Scalar[d_gmem_frag.linear_idx_type](dst_idx)
                    )
                    % type_of(thread_offset)(N)
                )
                if m < M and n < N:
                    var vec = (d_reg_frag.ptr + src_idx).load[
                        width=2, alignment=align_of[SIMD[d_type, 2]]()
                    ]()
                    epilogue((m, n), vec)

        else:
            copy_local_to_dram[dst_thread_layout=Layout.row_major(8, 4)](
                d_gmem_warp_tile.vectorize[1, 2](),
                d_reg_tile.vectorize[1, 2]().transpose(),
            )


def multistage_b2b_gemm[
    dst_type: DType,
    src_type: DType,
    transpose_b: Bool,
    transpose_c: Bool,
    //,
    config: BackToBackMatmulConfig[
        dst_type, src_type, transpose_b, transpose_c
    ],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    D: LayoutTensor,
    A: LayoutTensor,
    B: LayoutTensor,
    C: LayoutTensor,
    ctx: DeviceContext,
):
    try:
        comptime assert dst_type == D.dtype
        comptime assert src_type == A.dtype
        comptime assert src_type == B.dtype
        comptime assert src_type == C.dtype
        comptime b2b_fn = b2b_gemm[
            dst_type,
            src_type,
            D.layout,
            A.layout,
            B.layout,
            C.layout,
            transpose_b,
            transpose_c,
            config,
            elementwise_lambda_fn,
        ]
        var smem_use: Int = config.shared_mem_usage(
            size(Layout(A.layout.shape[1]))
        )
        print("smem_use =", smem_use)
        ctx.enqueue_function[b2b_fn](
            D,
            A,
            B,
            C,
            grid_dim=config.grid_dim(Int(D.runtime_layout.shape[0])),
            block_dim=config.block_dim(),
            shared_mem_bytes=smem_use,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_use)
            ),
        )
    except e:
        abort(String(e))


def matmul_naive(
    C: LayoutTensor[mut=True, ...],
    A: LayoutTensor,
    B: LayoutTensor,
):
    comptime assert len(C.layout) == 2
    comptime assert len(A.layout) == 2
    comptime assert len(B.layout) == 2
    comptime M: Int = size(Layout(C.layout.shape[0]))
    comptime N: Int = size(Layout(C.layout.shape[1]))
    comptime K: Int = size(Layout(A.layout.shape[1]))
    comptime assert M == size(Layout(A.layout.shape[0]))
    comptime assert N == size(Layout(B.layout.shape[1]))
    comptime assert K == size(Layout(B.layout.shape[0]))
    for m in range(M):
        for n in range(N):
            C[m, n] = Scalar[C.dtype]()
    for m in range(M):
        for k in range(K):
            for n in range(N):
                # C[m, n] += A[m, k].cast[C.dtype]() * rebind[Scalar[C.dtype]](B[k, n].cast[C.dtype]())
                C[m, n] += rebind[Scalar[C.dtype]](
                    A[m, k].cast[C.dtype]()
                ) * rebind[Scalar[C.dtype]](B[k, n].cast[C.dtype]())
                # C[m, n] += rebind[Scalar[C.dtype]](A[m, k].cast[C.dtype]()) * B[k, n].cast[C.dtype]()


def test_b2b_matmul(ctx: DeviceContext) raises:
    # alias M = 32
    comptime M = 640
    comptime N = 64
    comptime K = 64
    # alias L = 64
    # alias L = 128
    comptime L = 384

    comptime layout_a = Layout.row_major(M, K)
    comptime layout_b = Layout.row_major(K, L)
    comptime layout_c = Layout.row_major(L, N)
    comptime layout_d = Layout.row_major(M, N)

    comptime dst_type = DType.float32
    comptime src_type = DType.bfloat16

    var mat_a = ManagedLayoutTensor[src_type, layout_a](ctx)
    var mat_b = ManagedLayoutTensor[src_type, layout_b](ctx)
    var mat_c = ManagedLayoutTensor[src_type, layout_c](ctx)
    var mat_d = ManagedLayoutTensor[dst_type, layout_d](ctx)
    var stack_d = Array[Scalar[dst_type], layout_d.size()](fill={})
    comptime layout_ab = Layout.row_major(M, L)
    var stack_ab = Array[Scalar[dst_type], layout_ab.size()](fill={})
    var stack_ab_downcast = Array[Scalar[src_type], layout_ab.size()](fill={})
    var host_d_ref = LayoutTensor[dst_type, layout_d](stack_d)
    var host_ab = LayoutTensor[dst_type, layout_ab](stack_ab)
    var host_ab_downcast = LayoutTensor[src_type, layout_ab](stack_ab_downcast)

    var mat_a_tensor = mat_a.tensor()
    var mat_b_tensor = mat_b.tensor()
    var mat_c_tensor = mat_c.tensor()
    for m in range(M):
        for k in range(K):
            # mat_a.tensor[m, k] = ((m + 1) / K).cast[src_type]()
            # mat_a.tensor[m, k] = (1 / K).cast[src_type]()
            mat_a_tensor[m, k] = (Float64(k + m * K) / Float64(M * K)).cast[
                src_type
            ]()
    for k in range(K):
        for l in range(L):
            # mat_b.tensor[k, l] = 1
            mat_b_tensor[k, l] = BFloat16(l + k * L)
    for l in range(L):
        for n in range(N):
            mat_c_tensor[l, n] = (Float64((n * L + l)) * 0.125).cast[src_type]()
            # mat_c.tensor[l, n] = n + l * N
    matmul_naive(host_ab, mat_a_tensor, mat_b_tensor)
    for m in range(M):
        for l in range(L):
            host_ab_downcast[m, l] = host_ab[m, l].cast[src_type]()
    matmul_naive(host_d_ref, host_ab_downcast, mat_c_tensor)
    # print("Host Matrix:\n", host_d_ref)

    comptime config = BackToBackMatmulConfig[dst_type, src_type](
        IndexList[3, element_type=.uint64](32, 64, 64),
        IndexList[3, element_type=.uint64](16, 64, 16),
        num_pipeline_stages=2,
    )
    multistage_b2b_gemm[config](
        mat_d.device_tensor(),
        mat_a.device_tensor(),
        mat_b.device_tensor(),
        mat_c.device_tensor(),
        ctx,
    )

    ctx.synchronize()
    # print("Device Matrix:\n", mat_d.tensor)
    var mat_d_tensor = mat_d.tensor()
    for m in range(M):
        for n in range(N):
            assert_almost_equal(
                mat_d_tensor[m, n],
                host_d_ref[m, n],
            )


def main() raises:
    with DeviceContext() as ctx:
        test_b2b_matmul(ctx)
