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

from std.builtin._closure import __ownership_keepalive
from std.math import ceildiv
from std.sys import argv, size_of

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu import WARP_SIZE
from max.gpu.sync import barrier
from max.gpu import block_idx, lane_id, thread_idx, warp_id
from max.gpu.primitives.cluster import block_rank_in_cluster
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import external_memory
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.compute.arch.tcgen05 import *

# Additional imports for testing
from internal_utils import assert_almost_equal
from layout import IntTuple, Layout, LayoutTensor
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

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
def kernel_3[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_tma_rank: Int,
    b_tma_rank: Int,
    a_tile_shape: IndexList[a_tma_rank],
    b_tile_shape: IndexList[b_tma_rank],
    c_layout: Layout,
    a_desc_shape: IndexList[a_tma_rank],
    b_desc_shape: IndexList[b_tma_rank],
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: Int = 128,
](
    a_tma_op: TMATensorTile[a_type, a_tma_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tma_rank, b_tile_shape, b_desc_shape],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters_dev: Int32,
):
    var num_iters = Int(num_iters_dev)
    comptime assert num_threads == 128 or num_threads == 256
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]  # BM
    comptime MMA_N = mma_shape[1]  # BN
    comptime MMA_K = mma_shape[2]  # 16
    comptime num_m_mmas = BM // MMA_M
    comptime num_n_mmas = BN // MMA_N
    comptime num_k_mmas = BK // MMA_K

    # we don't do the whole mma_shape_A vibes, rather, we directly declare it
    # tile_layout_k_major is cutlass equiv of tile_to_mma_shape
    # and sA_layout gets computed directly, by hand
    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()
    comptime sub_a_smem_layout = tile_layout_k_major[
        a_type, BM, 64, swizzle_mode=a_swizzle
    ]()
    comptime sub_b_smem_layout = tile_layout_k_major[
        b_type, BN, 64, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, 64, swizzle_mode=b_swizzle
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

    # a_smem_layout is a description of how tile is arranged in memory, and LayoutTensor is a pointer to memory + a layout, taking in a_smem as its pointer
    comptime a_smem_tile_t = LayoutTensor[
        a_type,
        a_smem_layout,
        _,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime b_smem_tile_t = LayoutTensor[
        b_type,
        b_smem_layout,
        _,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime sub_a_smem_tile_t = LayoutTensor[
        a_type,
        sub_a_smem_layout,
        _,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime sub_b_smem_tile_t = LayoutTensor[
        b_type,
        sub_b_smem_layout,
        _,
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

    var a_smem_tile = a_smem_tile_t(a_smem)
    var b_smem_tile = b_smem_tile_t(b_smem)

    comptime accum_type = get_accum_type[a_type]()

    comptime c_frag_size = MMA_M * MMA_N // num_threads  # MMA_M * MMA_N is the size of the accumulator, num_threads is the number of threads in the warp, c_frag_size is the num of elements in the accumulator per thread
    var c_frag: Array[
        Scalar[accum_type], c_frag_size
    ]  # array of accumulator elements

    comptime a_expected_bytes = a_size * size_of[a_type]()
    comptime b_expected_bytes = b_size * size_of[b_type]()
    comptime expected_bytes = a_expected_bytes + b_expected_bytes

    var tma_mbar = (b_smem + b_size).bitcast[SharedMemBarrier]()
    var mma_mbar = tma_mbar + 1

    var ptr_tmem_addr = (mma_mbar + 1).bitcast[UInt32]()

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = warp_id() == 0
    var elect_one_thread = thread_idx.x == 0
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    comptime max_tmem_cols = 512

    # allocate all 2^18 bytes of smem for tcgen05, all 512 cols allocated
    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    # give me a tensor for matrices A and B, sliced down to the portion that this CTA is responsible for, and referring to original global tensor
    # so it's the 128 x 64 portion the CTA will copy from global tensor into smem tile A and B
    comptime a_canonical_layout = tile_to_descriptor[a_type, a_smem_layout]()
    comptime b_canonical_layout = tile_to_descriptor[
        b_type, b_smem_layout, is_k_major=transpose_b
    ]()
    comptime aSBO = a_canonical_layout[0].stride[1].value() * size_of[a_type]()
    comptime aLBO = a_canonical_layout[1].stride[1].value() * size_of[a_type]()
    comptime b_stride01 = b_canonical_layout[0].stride[1].value()
    comptime b_stride11 = b_canonical_layout[1].stride[1].value()
    comptime bSBO = (b_stride01 if transpose_b else b_stride11) * size_of[
        b_type
    ]()
    comptime bLBO = (b_stride11 if transpose_b else b_stride01) * size_of[
        b_type
    ]()

    var adesc = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](a_smem_tile.ptr)
    var bdesc = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](b_smem_tile.ptr)

    var idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        accum_type,
        a_type,
        b_type,
        Index[dtype=.uint32](mma_shape[0], mma_shape[1]),
        transpose_b=transpose_b,
    ]()

    for i in range(
        num_iters
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
                a_tma_op.async_copy(
                    sub_a_smem_tile,
                    tma_mbar[0],
                    (i * BK + k, block_idx.y * BM),
                )
                var sub_b_smem_tile = sub_b_smem_tile_t(b_smem + b_offset)
                b_tma_op.async_copy(
                    sub_b_smem_tile,
                    tma_mbar[0],
                    (
                        i * BK + k,
                        block_idx.x * BN,
                    ) if transpose_b else (
                        block_idx.x * BN,
                        i * BK + k,
                    ),
                )
        # wait for the copy to finish
        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        # now we do the mma, again only one thread issues the instruction
        if elect_one_thread:
            comptime for j in range(
                num_k_mmas
            ):  # BK by MMA_K chunks for the mma acc required this time, since you can only do MMA_K at a time (16 elements)
                comptime idx = IntTuple(0, MMA_K * j)
                comptime a_offset = a_smem_layout(idx) * size_of[a_type]()
                comptime b_offset = b_smem_layout(idx) * size_of[b_type]()

                # use c_scale=0 for the first mma only on the first iteration to initialize
                var c_scale_value: UInt32 = UInt32(
                    0 if (i == 0 and j == 0) else 1
                )
                mma(
                    adesc + a_offset,
                    bdesc + b_offset,
                    tmem_addr,
                    idesc,
                    c_scale=c_scale_value,
                )

            mma_arrive(mma_mbar)

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

    var ctile = c.tile[BM, BN](block_idx.y, block_idx.x)

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            var c_gmem_warp_tile = ctile.tile[MMA_M // num_warps, MMA_N](
                4 * m_mma + warp_id(), n_mma
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


def blackwell_kernel_3[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](
    c: LayoutTensor[mut=True, c_type, c_layout, _],
    a: LayoutTensor[mut=True, a_type, a_layout, _],
    b: LayoutTensor[mut=True, b_type, b_layout, _],
    ctx: DeviceContext,
) raises:
    var M = c.dim[0]()
    var N = c.dim[1]()
    var K = a.dim[1]()

    comptime assert transpose_b, "Only support transposed B"

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    # hard coded 64 for BK

    # equivalent of cutlass tma atom a, it is a handle that is passed to async_copy, to accurately tell the TMA engine how to copy from global tensor a into smem tile A
    var a_tma_op = create_tensor_tile[Index(BM, 64), swizzle_mode=a_swizzle](
        ctx, a
    )
    var b_tma_op = create_tensor_tile[
        Index(BN, 64) if transpose_b else Index(64, BN),
        swizzle_mode=b_swizzle,
    ](ctx, b)

    comptime smem_use = (
        BM * size_of[a_type]() + BN * size_of[b_type]()
    ) * BK + 24

    comptime block_dim = 128

    comptime kernel = kernel_3[
        a_type,
        b_type,
        c_type,
        type_of(a_tma_op).rank,
        type_of(b_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(b_tma_op).tile_shape,
        c.layout,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).desc_shape,
        block_tile_shape,
        umma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c,
        Int32(K // BK),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(block_dim),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )


comptime WARP_GROUP_SIZE = 128
comptime NumWarpPerWarpGroup = 4


def get_dict_of_shapes(
    index: Int, dict: Dict[Int, Tuple[Int, Int, Int]]
) -> Tuple[Int, Int, Int]:
    try:
        return dict[index]
    except error:
        print("error")
        return (128, 128, 128)


def make_dict_of_shapes() -> Dict[Int, Tuple[Int, Int, Int]]:
    var dic = Dict[Int, Tuple[Int, Int, Int]]()
    dic[0] = (4096, 4096, 4096)
    return dic^


def benchmark_blackwell_matmul(ctx: DeviceContext) raises:
    comptime a_type = DType.bfloat16
    comptime b_type = DType.bfloat16
    comptime c_type = DType.bfloat16
    comptime umma_shape = Index(64, 256, 16)
    comptime transpose_b = True
    comptime BK = 64

    comptime dict_of_shapes = make_dict_of_shapes()

    print("Benchmarking kernel_3")
    print("============================================")
    print("Shapes: [M, N, K]")
    print("Data types: a=", a_type, ", b=", b_type, ", c=", c_type)
    print("UMMA shape:", umma_shape[0], "x", umma_shape[1], "x", umma_shape[2])
    print("BK:", BK)
    print("transpose_b:", transpose_b)
    print()

    comptime for i in range(len(dict_of_shapes)):
        comptime shape = get_dict_of_shapes(i, dict_of_shapes)
        try:
            print(
                "Benchmarking shape: [",
                shape[0],
                ",",
                shape[1],
                ",",
                shape[2],
                "]",
            )
            test_blackwell_kernel_3[
                a_type,
                b_type,
                c_type,
                umma_shape,
                transpose_b,
                BK,
                benchmark=True,
                M=shape[0],
                N=shape[1],
                K=shape[2],
            ](ctx)
        except e:
            print("Error: Failed to run benchmark for this shape")


def test_blackwell_kernel_3[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    umma_shape: IndexList[3],
    transpose_b: Bool = True,
    BK: Int = 64,
    benchmark: Bool = False,
    M: Int = 4096,
    N: Int = 4096,
    K: Int = 4096,
](ctx: DeviceContext) raises:
    print(M, "x", N, "x", K)

    comptime a_layout = Layout.row_major(M, K)
    comptime b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    comptime c_layout = Layout.row_major(M, N)

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](M * K)
    var a_host = LayoutTensor[a_type, a_layout](a_host_ptr)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](N * K)
    var b_host = LayoutTensor[b_type, b_layout](b_host_ptr)
    var c_host = ManagedLayoutTensor[c_type, c_layout](ctx)
    var c_host_ref = ManagedLayoutTensor[c_type, c_layout](ctx)

    var a_device = ctx.enqueue_create_buffer[a_type](M * K)
    var a_device_lt = LayoutTensor[a_type, a_layout](a_device.unsafe_ptr())
    var b_device = ctx.enqueue_create_buffer[b_type](N * K)
    var b_device_lt = LayoutTensor[b_type, b_layout](b_device.unsafe_ptr())
    var c_device = ctx.enqueue_create_buffer[c_type](M * N)
    var c_device_lt = LayoutTensor[c_type, c_layout](c_device.unsafe_ptr())
    var c_device_ref = ctx.enqueue_create_buffer[c_type](M * N)
    var c_device_ref_lt = LayoutTensor[c_type, c_layout](
        c_device_ref.unsafe_ptr()
    )

    # Initialize matmul operands
    for m_idx in range(M):
        for k_idx in range(K):
            a_host[m_idx, k_idx] = Float32(k_idx).cast[a_type]()
    for n_idx in range(N):
        for k_idx in range(K):
            b_host[n_idx, k_idx] = Float32(1 if n_idx == k_idx else 0).cast[
                b_type
            ]()
    for i in range(M * N):
        c_host.tensor[update=False]().ptr[i] = Scalar[c_type](0)
        c_host_ref.tensor[update=False]().ptr[i] = Scalar[c_type](0)

    # Move operands to the Device

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_copy(c_device, c_host.tensor[update=False]().ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref.tensor[update=False]().ptr)

    comptime block_tile_shape = Index(umma_shape[0], umma_shape[1], BK)

    blackwell_kernel_3[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
    ](
        c_device_lt,
        a_device_lt,
        b_device_lt,
        ctx,
    )

    ctx.synchronize()
    if benchmark:
        comptime num_runs = 100
        comptime num_warmup = 10

        @always_inline
        def run_kernel(ctx: DeviceContext) raises {imm}:
            blackwell_kernel_3[
                transpose_b=transpose_b,
                umma_shape=umma_shape,  # 64, 128, 16
                block_tile_shape=block_tile_shape,  # 64, 128, 64 (BM, BN, entirety of BK)
            ](
                c_device_lt,
                a_device_lt,
                b_device_lt,
                ctx,
            )

        # Warmup
        for _ in range(num_warmup):
            run_kernel(ctx)
        ctx.synchronize()
        print("finished warmup")

        var nstime = (
            Float64(ctx.execution_time(run_kernel, num_runs)) / num_runs
        )
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12

        print("  Average time: ", sectime * 1000, " ms")
        print("  Performance: ", TFlop / sectime, " TFLOPS")
        print()
    else:
        vendor_blas.matmul(
            ctx,
            c_device_ref_lt,
            a_device_lt,
            b_device_lt,
            c_row_major=True,
            transpose_b=transpose_b,
        )

        ctx.synchronize()

        ctx.enqueue_copy(c_host.tensor[update=False]().ptr, c_device)
        ctx.enqueue_copy(c_host_ref.tensor[update=False]().ptr, c_device_ref)
        ctx.synchronize()
        comptime rtol = 1e-2
        assert_almost_equal(
            c_host.tensor[update=False]().ptr,
            c_host_ref.tensor[update=False]().ptr,
            M * N,
            atol=0.0001,
            rtol=rtol,
        )

    # `a_device_lt` / `b_device_lt` are raw pointer views, so past the copies
    # above nothing else refers to the buffers backing them.
    __ownership_keepalive(a_device, b_device)


def main() raises:
    with DeviceContext() as ctx:
        if is_benchmark():
            # Run the benchmark
            print("\n\n========== Running Benchmarks ==========\n")
            benchmark_blackwell_matmul(ctx)

        test_blackwell_kernel_3[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            umma_shape=Index(64, 256, 16),
            transpose_b=True,
            BK=64,
            M=4096,
            N=4096,
            K=4096,
        ](ctx)
