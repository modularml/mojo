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
from std.math.uutils import udivmod
from std.random import rand, randint, random_float64
from std.sys import align_of, argv, size_of

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.intrinsics import lop
from max.gpu.memory import external_memory

from internal_utils import assert_almost_equal
from std.random import rand
from layout import (
    Coord,
    CoordLike,
    Idx,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    lt_to_tt,
    row_major,
)
from layout.layout import *
from layout.layout_tensor import copy_dram_to_sram
from linalg.matmul.gpu import multistage_gemm
from linalg.utils_gpu import MatmulKernels
from std.memory import unsafe_memset_zero
from std.memory.unsafe import bitcast
from quantization import Q4sym
from quantization.qmatmul_gpu import multistage_gemm_q, pack_Q_tile

from std.utils import StaticTuple
from std.utils.index import Index, IndexList


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


@always_inline
def args_to_tuple[swap: Bool](arg_0: Int, arg_1: Int) -> Tuple[Int, Int]:
    comptime if swap:
        return Tuple(arg_1, arg_0)
    else:
        return Tuple(arg_0, arg_1)


def repack_Q4_0_for_sm8x[
    q_layout: Layout,
    repack_layout: Layout,
    scales_type: DType,
](
    q_weight: LayoutTensor[.uint8, q_layout, ImmutAnyOrigin],
    q_packed_weight: LayoutTensor[.uint8, repack_layout, MutAnyOrigin],
):
    comptime group_size = 32
    comptime group_bytes = size_of[DType.float16]() + (group_size // 2)
    comptime pack_factor = 8
    comptime repack_tile = Index(64, 16)
    comptime WARP_SIZE = 32
    comptime BN = 128
    comptime BK = 1024

    var tid = thread_idx.x
    var warp_id, lane_id = udivmod(tid, WARP_SIZE)
    comptime num_warps_x = BN // repack_tile[0]
    var warp_y, warp_x = udivmod(warp_id, num_warps_x)
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
    var repack_weights = LayoutTensor[.uint32, repacked_b_layout, _](
        q_packed_weight.ptr.bitcast[UInt32](),
    )

    comptime b_scales_layout = Layout.row_major(K_groups, N)
    var b_scales_ptr = q_packed_weight.ptr + N * K // 2
    var repack_scales = LayoutTensor[scales_type, b_scales_layout, _](
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
        _,
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
            var thread_tile = (
                raw_Q_tile.slice[:, 2:]()
                .vectorize[1, 2]()
                .distribute[thd_layout](lane_id)
            )

            comptime for i_ele in range(16):
                var val = thread_tile.load[2](i_ele // 2, i_ele % 2)
                frag_0[i_ele] = (val[0] & 0x0F) | ((val[1] & 0x0F) << 4)
                frag_1[i_ele] = ((val[0] & 0xF0) >> 4) | (val[1] & 0xF0)

            var repack_warp_tile = repacked_gemm_iter[].tile[
                64, group_size // pack_factor
            ](warp_x, warp_y)
            # The repack_warp_tile is of shape [64, (2, 2)]. In this case,
            # elements [0, 0], [0, 1], [1, 0] and [1, 1] are stored continuously
            # in the memory. We need to use a element shape of [2, 2] to
            # correctly vectorize this tensor.
            repack_warp_tile.vectorize[2, 2]().store(
                lane_id, 0, pack_Q_tile(frag_0)
            )
            repack_warp_tile.vectorize[2, 2]().store(
                lane_id, 1, pack_Q_tile(frag_1)
            )
            repacked_gemm_iter._incr()

            comptime scales_thread_layout = Layout(
                IntTuple(4, 8),
                IntTuple(16, 1),
            )
            var rt_scales_thread_layout = RuntimeLayout[scales_thread_layout]()

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


# this kernel dequantizes a repacked INT4 matrix into bf16 format
# Assuming a 64x16 (nxk) packing scheme
# Tile [i, j] stores part of the original matrix [i*64:(i+1)*64, j*16:(j+1)*16]
# Within each tile, weights are repacked similarly to the Marlin kernel.
# The memory address for tile [i, j] is (i * (N//64) + j) * tile_size,
# where tile_size is 64 * 16 * 4 / pack_factor = 512 Bytes.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
def create_ref_b[
    type_q: DType,
    type_b: DType,
    b_q_layout: Layout,
    b_layout: Layout,
    group_size: Int,
    pack_factor: Int,
](
    b_packed: LayoutTensor[type_q, b_q_layout, ImmutAnyOrigin],
    b_out: LayoutTensor[type_b, b_layout, MutAnyOrigin],
):
    comptime WARP_SIZE = 32
    comptime BLOCK_N = 128
    comptime BLOCK_K = 32
    comptime repack_tile = Index(64, 16)
    comptime TILE_N = 64
    comptime TILE_K = 16
    comptime num_k_warps = BLOCK_K // repack_tile[1]

    var tid = thread_idx.x
    var warp_id, lane_id = udivmod(tid, WARP_SIZE)
    var block_idx = Index(block_idx.x, block_idx.y)
    var warp_x, warp_y = udivmod(warp_id, num_k_warps)

    comptime group_bytes = group_size // 2 + 2
    comptime N = Int(b_q_layout.shape[0])
    comptime K = Int(b_q_layout.shape[1]) // group_bytes * group_size

    # Unpack quantized weights
    comptime scales_type = DType.bfloat16
    comptime b_type = DType.uint32
    comptime b_weight_layout = Layout.row_major(N // 64, K * 64 // pack_factor)
    var b_q = LayoutTensor[b_type, b_weight_layout, _](
        b_packed.ptr.bitcast[Scalar[b_type]](),
    )

    comptime b_scales_layout = Layout.row_major(K // group_size, N)
    var b_scales_ptr = b_packed.ptr + N * K // 2
    var scales = LayoutTensor[scales_type, b_scales_layout, _](
        b_scales_ptr.bitcast[Scalar[scales_type]](),
    )

    var b_q_gmem_tile = b_q.tile[
        BLOCK_N // repack_tile[0], (BLOCK_K * repack_tile[0]) // pack_factor
    ](block_idx[0], block_idx[1])
    var warp_q_tile = b_q_gmem_tile.tile[
        1, (repack_tile[0] * repack_tile[1]) // pack_factor
    ](warp_x, warp_y)

    var scales_tile = scales.tile[ceildiv(BLOCK_K, group_size), BLOCK_N](
        (block_idx[1] * BLOCK_K) // group_size, block_idx[0]
    )
    var warp_scales_tile = scales_tile.tile[
        ceildiv(BLOCK_K, group_size), repack_tile[0]
    ](0, warp_x)
    comptime smem_reg_scales_layout = Layout.row_major(8, 4)
    var scales_reg_tiles = (
        LayoutTensor[
            scales_type,
            Layout.row_major(repack_tile[0] // 8, 1),
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .vectorize[1, 1]()
    )
    # load scales
    scales_reg_tiles.vectorize[8, 1]().copy_from(
        warp_scales_tile.vectorize[1, 8]().distribute[
            smem_reg_scales_layout, axis=0
        ](lane_id)
    )

    var b_out_tile = b_out.tile[BLOCK_N, BLOCK_K](block_idx[0], block_idx[1])
    var warp_out_tile = b_out_tile.tile[repack_tile[0], repack_tile[1]](
        warp_x, warp_y
    )
    var mma_tile_iter_1 = warp_out_tile.tiled_iterator[8, 8, axis=0](0, 0)
    var mma_tile_iter_2 = warp_out_tile.tiled_iterator[8, 8, axis=0](0, 1)

    var vec = bitcast[.int32, 4](warp_q_tile.vectorize[1, 4]()[0, lane_id])

    @always_inline
    def int4tobf16(i4: Int32, scale: BFloat16) -> SIMD[.bfloat16, 2]:
        comptime MASK: Int32 = 0x000F000F
        comptime I4s_TO_BF16s_MAGIC_NUM: Int32 = 0x43004300
        comptime lut: Int32 = (0xF0 & 0xCC) | 0xAA
        var BF16_BIAS = SIMD[.bfloat16, 2](-136, -136)
        var BF16_SCALE = SIMD[.bfloat16, 2](scale, scale)
        var BF16_ZERO = SIMD[.bfloat16, 2](0, 0)
        var BF16_ONE = SIMD[.bfloat16, 2](1, 1)

        var t = lop[lut](i4, MASK, I4s_TO_BF16s_MAGIC_NUM)

        var v = (
            bitcast[.bfloat16, 2](t)
            .fma(BF16_ONE, BF16_BIAS)
            .fma(BF16_SCALE, BF16_ZERO)
        )
        return v

    comptime write_back_layout = Layout.row_major(1, 32)
    comptime write_back_type = type_of(
        mma_tile_iter_1[].vectorize[1, 2]()[0, 0]
    )

    var lane_row, lane_col = udivmod(lane_id, 4)

    comptime for i in range(0, TILE_N // 8, 2):
        var q_int = vec[i // 2]

        var v1 = int4tobf16(
            q_int, bitcast[.bfloat16, 1](scales_reg_tiles[i, 0])
        )
        mma_tile_iter_1[].vectorize[1, 2]()[lane_row, lane_col] = rebind[
            write_back_type
        ](v1)
        q_int >>= 4
        var v2 = int4tobf16(
            q_int, bitcast[.bfloat16, 1](scales_reg_tiles[i, 0])
        )
        mma_tile_iter_2[].vectorize[1, 2]()[lane_row, lane_col] = rebind[
            write_back_type
        ](v2)
        q_int >>= 4
        mma_tile_iter_1._incr()
        mma_tile_iter_2._incr()

        v1 = int4tobf16(
            q_int, bitcast[.bfloat16, 1](scales_reg_tiles[i + 1, 0])
        )
        mma_tile_iter_1[].vectorize[1, 2]()[lane_row, lane_col] = rebind[
            write_back_type
        ](v1)
        q_int >>= 4
        v2 = int4tobf16(
            q_int, bitcast[.bfloat16, 1](scales_reg_tiles[i + 1, 0])
        )
        mma_tile_iter_2[].vectorize[1, 2]()[lane_row, lane_col] = rebind[
            write_back_type
        ](v2)
        mma_tile_iter_1._incr()
        mma_tile_iter_2._incr()


def random_float16(min: Float64 = 0, max: Float64 = 1) -> Float16:
    # Avoid pulling in a __truncdfhf2 dependency for a float64->float16
    # conversion by casting through float32 first.
    return random_float64(min=min, max=max).cast[.float32]().cast[.float16]()


struct _block_Q4_0:
    comptime group_size = 32

    var base_scale: Float16
    var q_bits: Array[UInt8, Self.group_size // 2]


def test_repack_Q4_0_for_sm8x[
    NType: CoordLike, KType: CoordLike, //
](ctx: DeviceContext, n: NType, k: KType) raises:
    print("test repack_Q4_0_for_sm8x")

    def fill_random[dtype: DType](mut array: Array[Scalar[dtype], ...]):
        rand(array.unsafe_ptr(), len(array), min=0, max=255)

    def build_b_buffer(N: Int, K: Int, b_ptr: MutPointer[UInt8, _]):
        var k_groups = ceildiv(K, 32)
        var block_ptr = b_ptr.bitcast[_block_Q4_0]()

        for _ in range(N):
            for _ in range(k_groups):
                block_ptr[].base_scale = random_float16()
                fill_random(block_ptr[].q_bits)
                block_ptr += 1

    comptime group_size = 32
    comptime pack_factor = 8
    var N = Int(n.value())
    var K = Int(k.value())
    comptime BN = 128
    comptime BK = 1024
    comptime group_bytes = 2 + (group_size // 2)

    comptime _gguf_b_dim0 = NType.static_value
    comptime _gguf_b_dim1 = (KType.static_value // group_size) * group_bytes
    comptime _dequan_dim0 = KType.static_value
    comptime _dequan_dim1 = NType.static_value

    var dynamic_gguf_b_shape = IndexList[2](N, (K // group_size) * group_bytes)
    var dynamic_dequan_shape = IndexList[2](K, N)

    var gguf_b_size = N * ((K // group_size) * group_bytes)
    var repacked_b_size = N * ((K // group_size) * group_bytes)
    var dequan_size = K * N

    var gguf_b_host_ptr = ctx.enqueue_create_host_buffer[.uint8](gguf_b_size)
    var repacked_b_host_ptr = ctx.enqueue_create_host_buffer[.uint8](
        repacked_b_size
    )
    var gguf_dequan_ref_host_ptr = ctx.enqueue_create_host_buffer[
        DType.bfloat16
    ](dequan_size)
    var repacked_dequan_host_ptr = ctx.enqueue_create_host_buffer[
        DType.bfloat16
    ](dequan_size)

    comptime gguf_b_host_layout = Layout.row_major(_gguf_b_dim0, _gguf_b_dim1)
    var gguf_b_host_lt = LayoutTensor[.uint8, gguf_b_host_layout, _](
        gguf_b_host_ptr,
    )
    comptime dequan_host_layout = Layout.row_major(_dequan_dim0, _dequan_dim1)
    comptime dequan_lt_type = LayoutTensor[.bfloat16, dequan_host_layout, _]
    var gguf_dequan_ref_host_lt = dequan_lt_type(
        gguf_dequan_ref_host_ptr,
        RuntimeLayout[
            dequan_host_layout,
            element_type=dequan_lt_type.layout_int_type,
            linear_idx_type=dequan_lt_type.linear_idx_type,
        ].row_major(
            dynamic_dequan_shape.cast[dequan_lt_type.layout_int_type]()
        ),
    )

    build_b_buffer(N, K, gguf_b_host_ptr.unsafe_ptr())
    Q4sym[group_size, DType.bfloat16].dequantize_and_write_to_tensor(
        lt_to_tt(gguf_b_host_lt).as_immut(),
        lt_to_tt(gguf_dequan_ref_host_lt),
        rebind[IndexList[gguf_dequan_ref_host_lt.rank]](
            gguf_dequan_ref_host_lt.runtime_layout.shape.value.canonicalize()
        ),
    )

    var gguf_b_device = ctx.enqueue_create_buffer[.uint8](gguf_b_size)
    var repacked_b_device = ctx.enqueue_create_buffer[.uint8](repacked_b_size)
    var repacked_dequan_device = ctx.enqueue_create_buffer[.bfloat16](
        dequan_size
    )

    ctx.enqueue_copy(gguf_b_device, gguf_b_host_ptr)
    ctx.enqueue_copy(repacked_b_device, repacked_b_host_ptr)

    comptime gguf_b_layout = Layout.row_major(_gguf_b_dim0, _gguf_b_dim1)
    comptime repacked_b_layout = Layout.row_major(_gguf_b_dim0, _gguf_b_dim1)
    comptime repack_dequan_layout = Layout.row_major(_dequan_dim0, _dequan_dim1)
    comptime repacked_b_old_layout = Layout.row_major(
        NType.static_value // 64,
        KType.static_value * 64 // pack_factor,
    )
    comptime gguf_b_tensor_type = LayoutTensor[.uint8, gguf_b_layout, _]
    comptime repacked_dequan_tensor_type = LayoutTensor[
        .bfloat16,
        repack_dequan_layout,
        _,
    ]

    var gguf_b_tensor = gguf_b_tensor_type(
        gguf_b_device.unsafe_ptr(),
        RuntimeLayout[
            gguf_b_layout,
            element_type=gguf_b_tensor_type.layout_int_type,
            linear_idx_type=gguf_b_tensor_type.linear_idx_type,
        ].row_major(
            dynamic_gguf_b_shape.cast[gguf_b_tensor_type.layout_int_type]()
        ),
    )
    var repacked_b_tensor = LayoutTensor[.uint8, repacked_b_layout, _](
        repacked_b_device.unsafe_ptr(),
    )
    var repacked_dequan_tensor = repacked_dequan_tensor_type(
        repacked_dequan_device.unsafe_ptr(),
        RuntimeLayout[
            repack_dequan_layout,
            element_type=repacked_dequan_tensor_type.layout_int_type,
            linear_idx_type=repacked_dequan_tensor_type.linear_idx_type,
        ].row_major(
            dynamic_dequan_shape.cast[
                repacked_dequan_tensor_type.layout_int_type
            ]()
        ),
    )

    var smem_usage: Int = BN * 2 * group_bytes

    comptime repack = repack_Q4_0_for_sm8x[
        gguf_b_tensor.layout,
        repacked_b_tensor.layout,
        DType.bfloat16,
    ]

    ctx.enqueue_function[repack](
        gguf_b_tensor,
        repacked_b_tensor,
        grid_dim=(ceildiv(N, BN), ceildiv(K, BK), 1),
        block_dim=(128, 1, 1),
        shared_mem_bytes=smem_usage,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_usage)
        ),
    )

    comptime dequan = create_ref_b[
        DType.uint8,
        DType.bfloat16,
        repacked_b_tensor.layout,
        repacked_dequan_tensor.layout,
        group_size,
        pack_factor,
    ]

    ctx.enqueue_function[dequan](
        repacked_b_tensor,
        repacked_dequan_tensor,
        grid_dim=(ceildiv(N, 128), ceildiv(K, 32), 1),
        block_dim=(128, 1, 1),
        shared_mem_bytes=smem_usage,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_usage)
        ),
    )

    ctx.enqueue_copy(repacked_b_host_ptr, repacked_b_device)
    ctx.enqueue_copy(repacked_dequan_host_ptr, repacked_dequan_device)

    ctx.synchronize()

    comptime rtol = 2e-2
    assert_almost_equal(
        gguf_dequan_ref_host_ptr.unsafe_ptr(),
        repacked_dequan_host_ptr.unsafe_ptr(),
        dequan_size,
        atol=0.0001,
        rtol=rtol,
    )

    _ = repacked_dequan_tensor
    _ = gguf_b_tensor
    _ = repacked_b_tensor
    _ = gguf_b_device^
    _ = repacked_b_device^
    _ = repacked_dequan_device^


def test_quantized[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    dtype: DType,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    # quantization configs
    comptime group_size = 128
    comptime has_zero_point = False
    comptime pack_factor = 8
    comptime group_bytes = group_size // 2 + 2

    comptime repack_tile = Index(64, 16)

    print("test multistage matmul")
    comptime static_M = MType.static_value
    comptime static_N = NType.static_value
    comptime static_K = KType.static_value
    comptime a_type = DType.bfloat16

    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    comptime _b_dim0 = NType.static_value
    comptime _b_dim1 = (KType.static_value // group_size) * group_bytes

    var dynamic_b_shape = IndexList[2](N, (K // group_size) * group_bytes)
    var dynamic_b_ref_shape = IndexList[2](N, K)

    var a_size = M * K
    var b_size = N * ((K // group_size) * group_bytes)
    var b_ref_size = N * K
    var c_size = M * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dtype](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[a_type](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[a_type](c_size)

    rand(a_host_ptr.unsafe_ptr(), a_size)

    var b_scales_ptr = (b_host_ptr.unsafe_ptr() + N * K // 2).bitcast[
        Scalar[a_type]
    ]()
    var b_scales_size = (K // group_size) * N
    # elements of b matrix is between [-1, 1]
    rand(b_scales_ptr, b_scales_size, min=0, max=0.125)
    randint(
        b_host_ptr.unsafe_ptr().bitcast[UInt32](),
        N * (K // pack_factor),
        Int(UInt32.MIN),
        Int(UInt32.MAX),
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device = ctx.enqueue_create_buffer[dtype](b_size)
    var b_device_ref = ctx.enqueue_create_buffer[a_type](b_ref_size)
    var c_device = ctx.enqueue_create_buffer[a_type](c_size)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    comptime b_layout = Layout.row_major(_b_dim0, _b_dim1)
    comptime b_ref_layout = Layout.row_major(
        NType.static_value, KType.static_value
    )
    comptime b_tensor_type = LayoutTensor[dtype, b_layout, _]
    comptime b_ref_tensor_type = LayoutTensor[a_type, b_ref_layout, _]

    var b_tensor = b_tensor_type(
        b_device.unsafe_ptr(),
        RuntimeLayout[
            b_layout,
            element_type=b_tensor_type.layout_int_type,
            linear_idx_type=b_tensor_type.linear_idx_type,
        ].row_major(dynamic_b_shape.cast[b_tensor_type.layout_int_type]()),
    )
    var b_ref_tensor = b_ref_tensor_type(
        b_device_ref.unsafe_ptr(),
        RuntimeLayout[
            b_ref_layout,
            element_type=b_ref_tensor_type.layout_int_type,
            linear_idx_type=b_ref_tensor_type.linear_idx_type,
        ].row_major(
            dynamic_b_ref_shape.cast[b_ref_tensor_type.layout_int_type]()
        ),
    )

    var c_device_ref = ctx.enqueue_create_buffer[a_type](c_size)

    comptime kernels = MatmulKernels[a_type, dtype, a_type, True]()
    comptime config = kernels.ampere_128x128_4
    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]

    # Create TileTensors for the matmul operands
    var a_tt_shape = row_major(Coord(m, Idx[KType.static_value]))
    var b_tt_shape = row_major(
        Coord(
            Idx[NType.static_value],
            Idx[(KType.static_value // group_size) * group_bytes],
        )
    )
    var c_tt_shape = row_major(Coord(m, Idx[NType.static_value]))

    var c_dev_tt = TileTensor(c_device, c_tt_shape)
    var a_dev_tt = TileTensor(a_device, a_tt_shape)
    var b_dev_tt = TileTensor(b_device, b_tt_shape)

    var c_dev_lt = c_dev_tt.to_layout_tensor()
    var a_dev_lt = a_dev_tt.to_layout_tensor()
    var b_dev_lt = b_dev_tt.to_layout_tensor()

    if is_benchmark():
        comptime nrun = 200
        comptime nwarmup = 2

        @always_inline
        def run_func(ctx: DeviceContext) raises {imm}:
            multistage_gemm_q[
                group_size=group_size, pack_factor=pack_factor, config=config
            ](
                c_dev_lt,
                a_dev_lt,
                b_dev_lt,
                config,
                ctx,
            )

        # Warmup
        for _ in range(nwarmup):
            multistage_gemm_q[
                group_size=group_size, pack_factor=pack_factor, config=config
            ](
                c_dev_lt,
                a_dev_lt,
                b_dev_lt,
                config,
                ctx,
            )

        var nstime = Float64(ctx.execution_time(run_func, nrun)) / Float64(nrun)
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12
        print(
            "Transpose B ",
            "True",
            nrun,
            " runs avg(s)",
            sectime,
            "TFlops/s",
            TFlop / sectime,
        )

    multistage_gemm_q[
        group_size=group_size, pack_factor=pack_factor, config=config
    ](
        c_dev_lt,
        a_dev_lt,
        b_dev_lt,
        config,
        ctx,
    )

    comptime dequan = create_ref_b[
        dtype,
        a_type,
        b_tensor.layout,
        b_ref_tensor.layout,
        group_size,
        pack_factor,
    ]

    ctx.enqueue_function[dequan](
        b_tensor,
        b_ref_tensor,
        grid_dim=(ceildiv(N, 128), ceildiv(K, 32), 1),
        block_dim=(128, 1, 1),
        # dump_llvm=Path("./pipeline-gemm.ir"),
        # dump_asm=Path("./pipeline-gemm-2.ptx"),
    )

    ctx.enqueue_copy(c_host_ptr, c_device)

    comptime kernels_ref = MatmulKernels[a_type, a_type, a_type, True]()
    comptime config_ref = kernels_ref.ampere_128x128_4
    var c_ref_tt_shape = row_major(Coord(m, Idx[NType.static_value]))
    var b_ref_tt_shape = row_major(
        Coord(Idx[NType.static_value], Idx[KType.static_value])
    )
    var c_ref_tt = TileTensor(c_device_ref, c_ref_tt_shape)
    var b_ref_tt = TileTensor(b_device_ref, b_ref_tt_shape)
    multistage_gemm[transpose_b=True, config=config_ref](
        c_ref_tt,
        a_dev_tt,
        b_ref_tt,
        ctx,
    )

    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)

    ctx.synchronize()

    comptime rtol = 1e-2
    assert_almost_equal(
        c_host_ptr.unsafe_ptr(),
        c_host_ref_ptr.unsafe_ptr(),
        c_size,
        atol=0.0001,
        rtol=rtol,
    )

    _ = a_device^
    _ = b_device^
    _ = b_device_ref^
    _ = c_device^
    _ = c_device_ref^

    _ = b_tensor


def main() raises:
    with DeviceContext() as ctx:
        test_repack_Q4_0_for_sm8x(
            ctx,
            Idx[4096],
            Idx[4096],
        )
        test_quantized[.uint8](ctx, Idx[482], Idx[6144], Idx[4096])
        test_quantized[.uint8](ctx, Idx[482], Idx[4096], Idx[4096])
        test_quantized[.uint8](ctx, Idx[482], Idx[28672], Idx[4096])
        test_quantized[.uint8](ctx, Idx[482], Idx[4096], Idx[14336])
        test_quantized[.uint8](ctx, Idx[482], Idx[128256], Idx[4096])
        test_quantized[.uint8](ctx, Int(482), Idx[6144], Idx[4096])
        test_quantized[.uint8](ctx, Int(482), Idx[4096], Idx[4096])
        test_quantized[.uint8](ctx, Int(482), Idx[28672], Idx[4096])
        test_quantized[.uint8](ctx, Int(482), Idx[4096], Idx[14336])
        test_quantized[.uint8](ctx, Int(482), Idx[128256], Idx[4096])
