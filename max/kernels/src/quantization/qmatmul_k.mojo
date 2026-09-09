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
"""Provides CPU kernels for K-quant block-wise quantized matrix multiplication."""

from std.collections import Optional
from std.math import ceildiv
from std.sys import CompilationTarget, align_of, simd_width_of, size_of
from std.sys.intrinsics import llvm_intrinsic

from std.algorithm import tile

from max.algorithm import sync_parallelize
from max.gpu.host import DeviceContext
from layout import LayoutTensor, TileTensor
from linalg.accumulate import _Accumulator
from linalg.arch.cpu.neon_intrinsics import _neon_dotprod_lane
from linalg.arch.cpu.vnni_intrinsics import (
    dot_i8_to_i32_saturated_x86,
    dot_i16_to_i32_x86,
)
from linalg.matmul import elementwise_epilogue_type
from linalg.utils import partition_work
from std.memory import (
    alloc,
    bitcast,
    dealloc,
    unsafe_memcpy,
    unsafe_stack_allocation,
    Allocation,
)
from std.memory.alloc import Layout as AllocLayout

from max.runtime.asyncrt import parallelism_level

from std.utils.index import Index

from ._utils import roundeven_to_int32


struct _block_QK_K:
    comptime quantized_k = 256

    @staticmethod
    def calc_group_count[group_size: Int]() -> Int:
        return _block_QK_K.quantized_k // group_size


struct _block_Q4_K:
    comptime group_size = 32
    comptime group_count = _block_QK_K.calc_group_count[Self.group_size]()

    var base_scale: Float16
    var base_min: Float16
    var q_scales_and_mins: Array[UInt8, (2 * _block_Q4_K.group_count * 6) // 8]
    var q_bits: Array[UInt8, _block_QK_K.quantized_k // 2]


struct _block_Q6_K:
    comptime group_size = 16
    comptime group_count = _block_QK_K.calc_group_count[Self.group_size]()

    var q_bits_lo: Array[UInt8, _block_QK_K.quantized_k // 2]
    var q_bits_hi: Array[UInt8, _block_QK_K.quantized_k // 4]
    var q_scales: Array[Int8, _block_Q6_K.group_size]
    var base_scale: Float16


struct _packed_bit_array[bit_width: Int, block_m: Int, block_n: Int]:
    """Packed storage for an array of bit data.

    Logically, the array has a size of block_m by block_n. Physically, block_m
    is a multiple of the tuple width (corresponding to VNNI or equivalent
    instructions) and the tuple width is projected into the N dimension.
    This allows the SIMD width extracted from this array to be consistent with
    the native SIMD width for int32/float32 types without needing to play games
    with this array's dimensions.
    """

    comptime _size = Self.block_m * Self.block_n
    comptime _simd_width = simd_width_of[DType.uint8]()
    comptime _tuple_width = 4
    comptime _packed_stride = Self.block_n * Self._tuple_width
    comptime _tile_n = Self._packed_stride // Self._simd_width

    var bits: Array[UInt8, Self._size * Self.bit_width // 8]

    """
    For the 4-bit encoding, the following encoding is used (one lane of the
    SIMD register is depicted) where two rows of M are bundled together:
        [ b3 b2 b1 b0 a3 a2 a1 a0 ]
    """

    @always_inline
    def _pack_int4(mut self, var src_ptr: UnsafePointer[UInt8, ...]):
        comptime assert Self.bit_width == 4
        comptime assert (Self.block_m % (2 * Self._tuple_width)) == 0

        var bits_ptr: UnsafePointer[
            UInt8, origin_of(self.bits)
        ] = self.bits.unsafe_ptr()

        for _m in range(0, Self.block_m, 2 * Self._tuple_width):
            comptime for col in range(Self._tile_n):
                var packed_bits = SIMD[.uint8, Self._simd_width](0)

                comptime for i in range(2):
                    var bytes = (src_ptr + i * Self._packed_stride).load[
                        width=Self._simd_width
                    ]()
                    packed_bits |= bytes << UInt8(i * 4)

                src_ptr += Self._simd_width

                bits_ptr.store(packed_bits)
                bits_ptr += Self._simd_width

            src_ptr += Self._packed_stride

    @always_inline
    def _unpack_int4(self, var dst_ptr: UnsafePointer[mut=True, UInt8, _]):
        comptime assert Self.bit_width == 4
        comptime assert (Self.block_m % (2 * Self._tuple_width)) == 0

        var bits_ptr: UnsafePointer[
            UInt8, origin_of(self.bits)
        ] = self.bits.unsafe_ptr()

        for _ in range(0, Self.block_m, 2 * Self._tuple_width):
            comptime for col in range(Self._tile_n):
                var packed_bits = bits_ptr.load[width=Self._simd_width]()
                bits_ptr += Self._simd_width

                comptime for i in range(2):
                    var bytes = (packed_bits >> UInt8(i * 4)) & 15
                    (dst_ptr + i * Self._packed_stride).store(bytes)

                dst_ptr += Self._simd_width

            dst_ptr += Self._packed_stride

    # For the 6-bit encoding, the following encoding is used (one lane of the
    # SIMD register is depicted) where four rows of M are bundled together:
    #     [ d1 d0 a5 a4 a3 a2 a1 a0 ]
    #     [ d3 d2 b5 b4 b3 b2 b1 b0 ]
    #     [ d5 d6 c5 c4 c3 c2 c1 c0 ]

    @always_inline
    def _pack_int6(mut self, var src_ptr: UnsafePointer[UInt8, ...]):
        comptime assert Self.bit_width == 6
        comptime assert (Self.block_m % (4 * Self._tuple_width)) == 0

        var bits_ptr: UnsafePointer[
            UInt8, origin_of(self.bits)
        ] = self.bits.unsafe_ptr()

        for _m in range(0, Self.block_m, 4 * Self._tuple_width):
            var src_col_ptr = src_ptr

            comptime for col in range(Self._tile_n):
                var hi_bytes = (src_col_ptr + 3 * Self._packed_stride).load[
                    width=Self._simd_width
                ]()

                comptime for i in range(3):
                    var bytes = (src_col_ptr + i * Self._packed_stride).load[
                        width=Self._simd_width
                    ]()
                    var packed_bits = bytes | (
                        ((hi_bytes >> UInt8(i * 2)) & 3) << 6
                    )

                    bits_ptr.store(packed_bits)
                    bits_ptr += Self._simd_width

                src_col_ptr += Self._simd_width

            src_ptr += Self._packed_stride * 4

    @always_inline
    def _unpack_int6[
        zero_point: UInt8
    ](self, var dst_ptr: UnsafePointer[mut=True, UInt8, _]):
        comptime assert Self.bit_width == 6
        comptime assert (Self.block_m % (4 * Self._tuple_width)) == 0

        var bits_ptr: UnsafePointer[
            UInt8, origin_of(self.bits)
        ] = self.bits.unsafe_ptr()

        for _m in range(0, Self.block_m, 4 * Self._tuple_width):
            var dst_col_ptr = dst_ptr

            comptime for col in range(Self._tile_n):
                var hi_bytes = SIMD[.uint8, length=Self._simd_width](0)

                comptime for i in range(3):
                    var packed_bits = bits_ptr.load[width=Self._simd_width]()
                    bits_ptr += Self._simd_width

                    (dst_col_ptr + i * Self._packed_stride).store(
                        (packed_bits & 63) - zero_point,
                    )

                    hi_bytes |= (packed_bits >> 6) << UInt8(i * 2)

                (dst_col_ptr + 3 * Self._packed_stride).store(
                    hi_bytes - zero_point,
                )

                dst_col_ptr += Self._simd_width

            dst_ptr += Self._packed_stride * 4

    @always_inline
    def pack(mut self, var src_ptr: UnsafePointer[UInt8, ...]):
        """Packs the supplied external buffer to local storage."""
        comptime assert (Self._packed_stride % Self._simd_width) == 0

        comptime if Self.bit_width == 4:
            return self._pack_int4(src_ptr)
        elif Self.bit_width == 6:
            return self._pack_int6(src_ptr)
        else:
            comptime assert False, "unsupported bit width"

    @always_inline
    def unpack[
        *, zero_point: UInt8 = 0
    ](self, var dst_ptr: UnsafePointer[mut=True, UInt8, _]):
        """Unpacks the local storage to the supplied external buffer."""
        comptime assert (Self._packed_stride % Self._simd_width) == 0

        comptime if Self.bit_width == 4:
            comptime assert zero_point == 0, "zero point not implemented"
            return self._unpack_int4(dst_ptr)
        elif Self.bit_width == 6:
            return self._unpack_int6[zero_point](dst_ptr)
        else:
            comptime assert False, "unsupported bit width"


struct _block_Q4_K_packed[block_n: Int = 1]:
    var base_scales: Array[Float16, Self.block_n]
    var base_mins: Array[Float16, Self.block_n]
    var q_scales_and_mins: _packed_bit_array[
        6, 2 * _block_Q4_K.group_count, Self.block_n
    ]
    var q_bits: _packed_bit_array[4, _block_QK_K.quantized_k, Self.block_n]


struct _block_Q6_K_packed[block_n: Int = 1]:
    var base_scales: Array[Float16, Self.block_n]
    var q_scales: Array[Int8, _block_Q6_K.group_count * Self.block_n]
    var q_bits: _packed_bit_array[6, _block_QK_K.quantized_k, Self.block_n]


struct _block_Q8_K_packed[group_size: Int, tile_m: Int = 1]:
    comptime group_count = _block_QK_K.calc_group_count[Self.group_size]()

    var q_bits: Array[Int8, _block_QK_K.quantized_k * Self.tile_m]
    var scales: Array[Float32, Self.tile_m]
    var group_sums: Array[Int16, Self.group_count * Self.tile_m]


def _quantize_a_Q8_K[
    group_size: Int, dtype: DType, *, interleave_group_sums: Bool = False
](a: LayoutTensor[mut=False, dtype, ...]) -> Allocation[
    _block_Q8_K_packed[group_size]
]:
    comptime assert a.rank == 2
    comptime quantized_k = _block_QK_K.quantized_k
    comptime group_count = quantized_k // group_size

    var M = a.dim[0]()
    var K = a.dim[1]()

    var packed_base_alloc = alloc(
        AllocLayout[_block_Q8_K_packed[group_size]](
            count=M * (K // quantized_k)
        )
    )
    var packed_ptr: UnsafePointer[
        _block_Q8_K_packed[group_size], origin_of(packed_base_alloc._alloc)
    ] = packed_base_alloc.unsafe_ptr()

    for ko in range(0, K, quantized_k):
        var am_ptr = a.ptr + ko

        @always_inline
        def process_rows[tile_m: Int](m: Int) {mut am_ptr, mut packed_ptr, imm}:
            comptime assert (
                size_of[_block_Q8_K_packed[group_size]]() * tile_m
                == size_of[_block_Q8_K_packed[group_size, tile_m]]()
            ), "tiled block size should be multiple of the single block size"

            var block_ptr = packed_ptr.bitcast[
                _block_Q8_K_packed[group_size, tile_m]
            ]()
            var q_bits_ptr: UnsafePointer[
                Int8, origin_of(block_ptr[].q_bits)
            ] = block_ptr[].q_bits.unsafe_ptr()

            for row in range(tile_m):
                var max_value_simd = SIMD[dtype, group_size](Scalar[dtype].MIN)

                for g in range(group_count):
                    var fp_data = am_ptr.load[width=group_size](g * group_size)
                    max_value_simd = max(abs(fp_data), max_value_simd)

                var max_value = max_value_simd.reduce_max()
                var scale = (max_value / 127.0).cast[.float32]()
                var multiplier = 127.0 / max_value if max_value != 0.0 else 0.0

                for g in range(group_count):
                    var fp_data = am_ptr.load[width=group_size](g * group_size)
                    var q_data_i32 = roundeven_to_int32(fp_data * multiplier)
                    var q_data_i8 = q_data_i32.cast[.int8]()
                    var group_sum = q_data_i32.reduce_add()

                    q_bits_ptr.store(
                        g * tile_m * group_size + row * group_size,
                        q_data_i8,
                    )

                    comptime if interleave_group_sums:
                        block_ptr[].group_sums[
                            ((g >> 1) * tile_m + row) * 2 + (g & 1)
                        ] = group_sum.cast[.int16]()
                    else:
                        block_ptr[].group_sums[
                            g * tile_m + row
                        ] = group_sum.cast[.int16]()

                block_ptr[].scales[row] = scale

                am_ptr += K

            packed_ptr += tile_m

        tile[[4, 2, 1]](0, M, process_rows)
        # TODO(MOCO-2074): Suppress false positive unused var warning.
        _ = am_ptr

    return packed_base_alloc^


def _expand_q_bits_lo[
    *, width: Int
](
    var src_ptr: UnsafePointer[UInt8, ...],
    var dst_ptr: UnsafePointer[mut=True, UInt8, ...],
):
    for _k in range(0, _block_QK_K.quantized_k // 2, width):
        var src_q_bits = src_ptr.load[width=width]()
        src_ptr += width

        comptime for i in range(2):
            dst_ptr.store((src_q_bits >> UInt8(i * 4)) & 15)
            dst_ptr += width


def _expand_and_merge_q_bits_hi[
    *, width: Int, bit_count: Int
](
    var src_ptr: UnsafePointer[UInt8, ...],
    var dst_ptr: UnsafePointer[mut=True, UInt8, ...],
):
    comptime values_per_byte = 8 // bit_count
    comptime bit_mask = (1 << bit_count) - 1

    for _k in range(0, _block_QK_K.quantized_k // values_per_byte, width):
        var src_q_bits = src_ptr.load[width=width]()
        src_ptr += width

        for _ in range(values_per_byte):
            var dst_q_bits_lo = dst_ptr.load[width=width]()
            var dst_q_bits_hi = (src_q_bits & UInt8(bit_mask)) << 4
            src_q_bits >>= UInt8(bit_count)

            dst_ptr.store(dst_q_bits_hi | dst_q_bits_lo)
            dst_ptr += width


def _copy_column_q_bits_to_block[
    block_n: Int
](
    var src_ptr: UnsafePointer[UInt8, _],
    var dst_ptr: UnsafePointer[mut=True, UInt8, _],
):
    """Interleaves the linear source buffer to the blocked destination
    buffer.
    """
    for _k in range(0, _block_QK_K.quantized_k, 4):
        dst_ptr.store(src_ptr.load[width=4]())
        src_ptr += 4
        dst_ptr += block_n * 4


def _pack_block_Q4_K[
    block_n: Int,
](
    var src_ptr: UnsafePointer[mut=False, _block_Q4_K, _],
    stride: Int,
    dst_ptr: UnsafePointer[mut=True, _block_Q4_K_packed[block_n], _],
):
    comptime group_size = _block_Q4_K.group_size
    comptime group_count = _block_Q4_K.group_count

    comptime assert (
        size_of[_block_Q4_K]() * block_n
        == size_of[_block_Q4_K_packed[block_n]]()
    ), "packed block size should be multiple of the unpacked block size"

    var q_scales_buf = Array[UInt8, group_count * block_n](uninitialized=True)
    var q_mins_buf = Array[UInt8, group_count * block_n](uninitialized=True)
    var q_mins_ptr: UnsafePointer[
        UInt8, origin_of(q_mins_buf)
    ] = q_mins_buf.unsafe_ptr()
    var q_bits_block_buf = Array[UInt8, _block_QK_K.quantized_k * block_n](
        uninitialized=True
    )
    var q_bits_block_ptr: UnsafePointer[
        UInt8, origin_of(q_bits_block_buf)
    ] = q_bits_block_buf.unsafe_ptr()

    for n in range(block_n):
        dst_ptr[].base_scales[n] = src_ptr[].base_scale
        dst_ptr[].base_mins[n] = src_ptr[].base_min

        # Decode the packed 6-bit scales and minimums to a local working buffer.
        for g in range(group_count):
            var q_scale: UInt8
            var q_min: UInt8
            if g < 4:
                q_scale = src_ptr[].q_scales_and_mins[g] & 63
                q_min = src_ptr[].q_scales_and_mins[g + 4] & 63
            else:
                var q_scale_lo = src_ptr[].q_scales_and_mins[g + 4] & 15
                var q_min_lo = src_ptr[].q_scales_and_mins[g + 4] >> 4
                var q_scale_hi = src_ptr[].q_scales_and_mins[g - 4] >> 6
                var q_min_hi = src_ptr[].q_scales_and_mins[g - 0] >> 6
                q_scale = (q_scale_hi << 4) | q_scale_lo
                q_min = (q_min_hi << 4) | q_min_lo
            q_scales_buf[g * block_n + n] = q_scale
            q_mins_buf[g * block_n + n] = q_min

        var q_bits_column_buf = Array[UInt8, _block_QK_K.quantized_k](
            uninitialized=True
        )

        _expand_q_bits_lo[width=32](
            src_ptr[].q_bits.unsafe_ptr(), q_bits_column_buf.unsafe_ptr()
        )
        _copy_column_q_bits_to_block[block_n](
            q_bits_column_buf.unsafe_ptr(),
            q_bits_block_ptr + n * 4,
        )

        src_ptr += stride

    # Allocate a staging buffer to pack the scales and minimums as a single
    # blob and to do processor specific reordering of the values for the
    # compute kernel.
    var q_scales_and_mins_buf = Array[UInt8, 2 * group_count * block_n](
        uninitialized=True
    )
    var q_scales_and_mins_ptr: UnsafePointer[
        UInt8, origin_of(q_scales_and_mins_buf)
    ] = q_scales_and_mins_buf.unsafe_ptr()
    var q_scales_reorder_buf = q_scales_and_mins_ptr
    var q_mins_reorder_buf = q_scales_and_mins_ptr + group_count * block_n

    # Scales are not currently transformed.
    unsafe_memcpy(
        dest=q_scales_reorder_buf,
        src=q_scales_buf.unsafe_ptr(),
        count=group_count * block_n,
    )

    # Minimums are row interleaved with a stride to enable use of int16->int32
    # multiply/add instructions.
    #
    # For x86: The compute kernel uses `pmaddwd` + `paddd' (optimized to
    # `vpdpwssd` on processors that support VNNI). The two rows are interleaved
    # to form pairs of int16 values:
    #       [n0_g0 n0_g1 : n1_g0 n1_g1 : n2_g0 n2_g1 : n3_g0 n3_g1]
    #
    # For NEON: The compute kernel uses `smull(2)` and `smlal(2)` instructions
    # to do an `int16*int16` widening multiply/add to an int32 accumulator. The
    # two rows are split across the lower and upper halves of the register:
    #       [n0_g0 n1_g0 n2_g0 n3_g0 : n0_g1 n1_g1 n2_g1 n3_g1]
    for g in range(0, group_count, 2):
        var q_mins_row_0_ptr = q_mins_ptr + g * block_n
        var q_mins_row_1_ptr = q_mins_row_0_ptr + block_n
        for n in range(block_n):
            var q_mins_row_0_val = q_mins_row_0_ptr[n]
            var q_mins_row_1_val = q_mins_row_1_ptr[n]

            comptime if CompilationTarget.has_sse4():
                var reorder_idx = g * block_n + n * 2
                q_mins_reorder_buf[reorder_idx + 0] = q_mins_row_0_val
                q_mins_reorder_buf[reorder_idx + 1] = q_mins_row_1_val
            elif CompilationTarget.has_neon():
                comptime split_width = simd_width_of[DType.int32]()
                var n_idx_hi, n_idx_lo = divmod(n, split_width)
                var reorder_idx = (
                    g * block_n + n_idx_hi * split_width * 2 + n_idx_lo
                )
                q_mins_reorder_buf[reorder_idx + 0] = q_mins_row_0_val
                q_mins_reorder_buf[reorder_idx + split_width] = q_mins_row_1_val
            else:
                comptime assert False, "unsupported architecture"

    dst_ptr[].q_scales_and_mins.pack(q_scales_and_mins_buf.unsafe_ptr())
    dst_ptr[].q_bits.pack(q_bits_block_buf.unsafe_ptr())


def _pack_block_Q6_K[
    block_n: Int,
](
    var src_ptr: UnsafePointer[mut=False, _block_Q6_K, _],
    stride: Int,
    dst_ptr: UnsafePointer[mut=True, _block_Q6_K_packed[block_n], _],
):
    comptime group_count = _block_Q6_K.group_count

    comptime assert (
        size_of[_block_Q6_K]() * block_n
        == size_of[_block_Q6_K_packed[block_n]]()
    ), "packed block size should be multiple of the unpacked block size"

    var q_bits_block_buf = unsafe_stack_allocation[
        _block_QK_K.quantized_k * block_n, DType.uint8
    ]()

    for n in range(block_n):
        dst_ptr[].base_scales[n] = src_ptr[].base_scale

        for g in range(group_count):
            dst_ptr[].q_scales[g * block_n + n] = src_ptr[].q_scales[g]

        var q_bits_column_buf = unsafe_stack_allocation[
            _block_QK_K.quantized_k, DType.uint8
        ]()

        _expand_q_bits_lo[width=64](
            src_ptr[].q_bits_lo.unsafe_ptr(), q_bits_column_buf
        )
        _expand_and_merge_q_bits_hi[width=32, bit_count=2](
            src_ptr[].q_bits_hi.unsafe_ptr(), q_bits_column_buf
        )
        _copy_column_q_bits_to_block[block_n](
            q_bits_column_buf, q_bits_block_buf + n * 4
        )

        src_ptr += stride

    dst_ptr[].q_bits.pack(q_bits_block_buf)


def matmul_Q4_K_pack_b(
    b_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    b_packed_tt: TileTensor[mut=True, .uint8, address_space=.GENERIC, ...],
):
    """Packs Q4_K quantized weights into the blocked layout consumed by the compute kernel.

    Args:
        b_tt: Source tensor holding the unpacked Q4_K quantized weights.
        b_packed_tt: Destination tensor for the packed weights.
    """
    var b = b_tt.to_layout_tensor()
    var b_packed = b_packed_tt.to_layout_tensor()
    comptime assert b.rank == 2
    comptime assert b_packed.rank == 2
    var N = b.dim[0]()
    var K = b.dim[1]()
    var k_blocks = K // size_of[_block_Q4_K]()

    comptime simd_width = simd_width_of[DType.float32]()
    comptime block_n = simd_width * 2

    var src_ptr = b.ptr.bitcast[_block_Q4_K]()
    var dst_ptr = b_packed.ptr.bitcast[_block_Q4_K_packed[block_n]]()

    for _kb in range(k_blocks):
        var src_n_ptr = src_ptr

        for _n in range(0, N, block_n):
            _pack_block_Q4_K[block_n](src_n_ptr, k_blocks, dst_ptr)

            src_n_ptr += k_blocks * block_n
            dst_ptr += 1

        src_ptr += 1


def matmul_Q6_K_pack_b(
    b_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    b_packed_tt: TileTensor[mut=True, .uint8, address_space=.GENERIC, ...],
):
    """Packs Q6_K quantized weights into the blocked layout consumed by the compute kernel.

    Args:
        b_tt: Source tensor holding the unpacked Q6_K quantized weights.
        b_packed_tt: Destination tensor for the packed weights.
    """
    var b = b_tt.to_layout_tensor()
    var b_packed = b_packed_tt.to_layout_tensor()
    comptime assert b.rank == 2
    comptime assert b_packed.rank == 2
    var N = b.dim[0]()
    var K = b.dim[1]()
    var k_blocks = K // size_of[_block_Q6_K]()

    comptime simd_width = simd_width_of[DType.float32]()
    comptime block_n = simd_width * 2

    var src_ptr = b.ptr.bitcast[_block_Q6_K]()
    var dst_ptr = b_packed.ptr.bitcast[_block_Q6_K_packed[block_n]]()

    for _kb in range(k_blocks):
        var src_n_ptr = src_ptr

        for _n in range(0, N, block_n):
            _pack_block_Q6_K[block_n](src_n_ptr, k_blocks, dst_ptr)

            src_n_ptr += k_blocks * block_n
            dst_ptr += 1

        src_ptr += 1


@always_inline
def _matmul_group_stream_x86[
    tile_m: Int,
    tile_n: Int,
    simd_width: Int,
    b_width: SIMDLength,
    b_count: Int,
    //,
    group_size: Int,
    tile_k: Int,
](
    a_q_bits_ptr: UnsafePointer[Int8, _],
    mut c_int32_group: _Accumulator[.int32, tile_m, tile_n, simd_width],
    stream_b_vals_fn: Some[
        def(mut b_vals: Array[SIMD[.uint8, b_width], b_count]) -> None
    ],
):
    # The array type is spelled with free `b_width`/`b_count` parameters
    # because dependent expressions like `SIMDLength(simd_width) * 4` fold
    # differently in the caller's closure signature and fail to unify.
    comptime assert b_width == SIMDLength(simd_width) * 4
    comptime assert b_count == tile_n * tile_k

    var b_vals = Array[SIMD[.uint8, b_width], b_count](fill=0)

    comptime for k in range(0, group_size, tile_k * 4):
        stream_b_vals_fn(b_vals)

        comptime for tk in range(tile_k):
            comptime for col in range(tile_n):
                comptime for row in range(tile_m):
                    var a_val = SIMD[.int32, simd_width](
                        bitcast[.int32, 1](
                            (a_q_bits_ptr + row * group_size + k + tk * 4).load[
                                width=4
                            ]()
                        )
                    )
                    c_int32_group[row, col] = dot_i8_to_i32_saturated_x86(
                        c_int32_group[row, col],
                        bitcast[.int32, simd_width](b_vals[col * tile_k + tk]),
                        a_val,
                    )


@always_inline
def _matmul_group_stream_neon_dotprod[
    tile_m: Int,
    tile_n: Int,
    simd_width: Int,
    b_width: SIMDLength,
    b_count: Int,
    //,
    group_size: Int,
    tile_k: Int,
](
    a_q_bits_ptr: UnsafePointer[Int8, ...],
    mut c_int32_group: _Accumulator[.int32, tile_m, tile_n, simd_width],
    stream_b_vals_fn: Some[
        def(mut b_vals: Array[SIMD[.uint8, b_width], b_count]) -> None
    ],
):
    comptime assert b_width == SIMDLength(simd_width) * 4
    comptime assert b_count == tile_n * tile_k

    var b_vals = Array[SIMD[.uint8, b_width], b_count](fill=0)

    comptime for k in range(0, group_size, 16):
        var a_tile = Array[_, tile_m](
            fill_with=lambda (row: Int) {imm a_q_bits_ptr} -> SIMD[.int8, 16]: (
                a_q_bits_ptr + row * group_size + k
            ).load[width=16]()
        )

        comptime for lane in range(0, 4, tile_k):
            stream_b_vals_fn(b_vals)

            comptime for tk in range(tile_k):
                comptime for col in range(tile_n):
                    comptime for row in range(tile_m):
                        c_int32_group[row, col] = _neon_dotprod_lane[lane + tk](
                            c_int32_group[row, col],
                            rebind[SIMD[.int8, SIMDLength(simd_width) * 4]](
                                b_vals[col * tile_k + tk].cast[.int8]()
                            ),
                            a_tile[row],
                        )


@always_inline
def _matmul_group_stream[
    tile_m: Int,
    tile_n: Int,
    simd_width: Int,
    b_width: SIMDLength,
    b_count: Int,
    //,
    group_size: Int,
    tile_k: Int,
](
    a_q_bits_ptr: UnsafePointer[Int8, _],
    mut c_int32_group: _Accumulator[.int32, tile_m, tile_n, simd_width],
    stream_b_vals_fn: Some[
        def(mut b_vals: Array[SIMD[.uint8, b_width], b_count]) -> None
    ],
):
    comptime assert Bool(tile_k.is_power_of_two()) and tile_k <= 4
    comptime assert b_width == SIMDLength(simd_width) * 4
    comptime assert b_count == tile_n * tile_k

    comptime if CompilationTarget.has_sse4():
        return _matmul_group_stream_x86[group_size, tile_k=tile_k](
            a_q_bits_ptr, c_int32_group, stream_b_vals_fn
        )
    elif CompilationTarget.has_neon():
        return _matmul_group_stream_neon_dotprod[group_size, tile_k=tile_k](
            a_q_bits_ptr, c_int32_group, stream_b_vals_fn
        )
    else:
        comptime assert False, "unsupported architecture"


@always_inline
def _matmul_group_unpacked[
    tile_m: Int,
    tile_n: Int,
    simd_width: Int,
    //,
    group_size: Int,
](
    a_q_bits_ptr: UnsafePointer[mut=False, Int8, _],
    mut b_q_bits_ptr: UnsafePointer[mut=True, UInt8, _],
    mut c_int32_group: _Accumulator[.int32, tile_m, tile_n, simd_width],
):
    """Streaming matrix multiplication where the B matrix has been unpacked to
    local storage.
    """

    def stream_b_vals(
        mut b_vals: Array[SIMD[.uint8, SIMDLength(simd_width) * 4], tile_n * 1]
    ) {mut b_q_bits_ptr, imm}:
        comptime for col in range(tile_n):
            b_vals[col] = b_q_bits_ptr.load[width=SIMDLength(simd_width) * 4]()
            b_q_bits_ptr += simd_width * 4

    _matmul_group_stream[group_size, tile_k=1](
        a_q_bits_ptr, c_int32_group, stream_b_vals
    )


@always_inline
def _apply_base_scales[
    tile_m: Int, tile_n: Int, simd_width: Int
](
    b_base_scales_ptr: UnsafePointer[Float16, _],
    c_int32_block: _Accumulator[.int32, tile_m, tile_n, simd_width],
    mut c_float: _Accumulator[.float32, tile_m, tile_n, simd_width],
):
    # Convert to floating point and apply the block scale of matrix B.
    comptime for col in range(tile_n):
        var b_scale = (
            (b_base_scales_ptr + col * simd_width)
            .load[width=simd_width]()
            .cast[.float32]()
        )

        comptime for row in range(tile_m):
            c_float[row, col] = (
                c_int32_block[row, col].cast[.float32]() * b_scale
            )


@always_inline
def _apply_zero_point_correction[
    group_count: Int, tile_m: Int, tile_n: Int, simd_width: Int
](
    a_group_sums_ptr: UnsafePointer[Int16, _],
    b_q_mins_ptr: UnsafePointer[UInt8, _],
    b_base_mins_ptr: UnsafePointer[Float16, _],
    mut c_float: _Accumulator[.float32, tile_m, tile_n, simd_width],
):
    """Applies the zero point correction to the running float accumulator."""
    comptime block_n = tile_n * simd_width

    var corrections = _Accumulator[.int32, tile_m, tile_n, simd_width]()
    corrections.init()

    for g in range(0, group_count, 2):
        comptime if CompilationTarget.has_sse4():
            # Use `pmaddwd` + `paddd' (optimized to `vpdpwssd` on processors
            # that support VNNI) to multiply/add a pair of minimum values with
            # a pair of group sums from matrix A.
            comptime for col in range(tile_n):
                # The minimum values vector is encoded as pairs of int16 values
                # from group_0 and group_1:
                #       [n0_g0 n0_g1 : n1_g0 n1_g1 : n2_g0 n2_g1 : n3_g0 n3_g1]
                var q_mins = b_q_mins_ptr.load[
                    width=SIMDLength(simd_width) * 2
                ](g * block_n + col * simd_width * 2).cast[.int16]()

                comptime for row in range(tile_m):
                    var a_group_sums = a_group_sums_ptr.load[width=2](
                        g * tile_m + row * 2
                    )
                    corrections[row, col] = dot_i16_to_i32_x86(
                        corrections[row, col],
                        q_mins,
                        bitcast[.int16, SIMDLength(simd_width) * 2](
                            SIMD[.int32, simd_width](
                                bitcast[.int32, 1](a_group_sums)
                            )
                        ),
                    )

        elif CompilationTarget.has_neon():
            # Use `smull(2)` and `smlal(2)` instructions to do an `int16*int16`
            # widening multiply/add to an int32 accumulator.
            var group_sums = (a_group_sums_ptr + g * tile_m).load[
                width=tile_m * 2
            ]()

            comptime for col in range(tile_n):
                # The minimum values vector is encoded as pairs of int16 values
                # from group_0 and group_1:
                #       [n0_g0 n1_g0 n2_g0 n3_g0 : n0_g1 n1_g1 n2_g1 n3_g1]
                var q_mins = b_q_mins_ptr.load[width=simd_width * 2](
                    g * block_n + col * simd_width * 2
                ).cast[.int16]()

                # Logically slice the minimum values vector. This selects
                # between `smull` (lower half) or `smull2` (upper half).
                var q_mins_lo_hi = q_mins.split()

                comptime for row in range(tile_m):
                    # Note: The ARM64 backend fuses `smull` with an int32 add to
                    # form `smlal` instructions. Also, the element broadcast is
                    # fused to with the instruction to generate the form
                    # `smlal r, a, b[lane]`. The intrinsic `vmlal_lane_s16` uses
                    # the same IR pattern to emit this instruction.
                    corrections[row, col] += llvm_intrinsic[
                        "llvm.aarch64.neon.smull.v4i32",
                        SIMD[.int32, simd_width],
                    ](
                        q_mins_lo_hi[0],
                        SIMD[length=simd_width](group_sums[row * 2 + 0]),
                    )
                    corrections[row, col] += llvm_intrinsic[
                        "llvm.aarch64.neon.smull.v4i32",
                        SIMD[.int32, simd_width],
                    ](
                        q_mins_lo_hi[1],
                        SIMD[length=simd_width](group_sums[row * 2 + 1]),
                    )

        else:
            comptime assert False, "unsupported architecture"

    # Scale the correction value by the shared base minimum and update the
    # float accumulator.
    comptime for col in range(tile_n):
        var base_mins = (
            (b_base_mins_ptr + col * simd_width)
            .load[width=simd_width]()
            .cast[.float32]()
        )

        comptime for row in range(tile_m):
            c_float[row, col] -= (
                corrections[row, col].cast[.float32]() * base_mins
            )


@always_inline
def _apply_a_scales[
    tile_m: Int, tile_n: Int, simd_width: Int
](
    a_scales_ptr: UnsafePointer[Float32, _],
    mut c_float: _Accumulator[.float32, tile_m, tile_n, simd_width],
):
    comptime if CompilationTarget.has_neon():
        # NEON supports a multiply instruction that can broadcast from a
        # vector element, so help the compiler produce that by doing a
        # vector load.
        var a_scale = a_scales_ptr.load[width=tile_m]()

        comptime for row in range(tile_m):
            comptime for col in range(tile_n):
                c_float[row, col] *= a_scale[row]

    else:
        comptime for row in range(tile_m):
            var a_scale = a_scales_ptr[row]

            comptime for col in range(tile_n):
                c_float[row, col] *= a_scale


@always_inline
def _accumulate_and_store[
    tile_m: Int,
    tile_n: Int,
    simd_width: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_ptr: UnsafePointer[mut=True, Float32, _],
    N: Int,
    accumulate: Bool,
    mut c_float: _Accumulator[.float32, tile_m, tile_n, simd_width],
    m: Int,
    n: Int,
    is_last_k_iter: Bool,
):
    if accumulate:
        var c_existing = _Accumulator[.float32, tile_m, tile_n, simd_width]()

        c_existing.load(c_ptr, N)

        comptime for col in range(tile_n):
            comptime for row in range(tile_m):
                c_float[row, col] += c_existing[row, col]

    c_float.store(c_ptr, N)

    comptime if elementwise_lambda_fn:
        comptime func = elementwise_lambda_fn.value()

        if is_last_k_iter:
            comptime for mm in range(tile_m):
                comptime for nn in range(tile_n):
                    var val = c_float[mm, nn]
                    func[.float32, simd_width](
                        Index(
                            m + mm,
                            n + nn * simd_width,
                        ),
                        val,
                    )


@always_inline
def _matmul_group_packed_Q4_K[
    tile_m: Int,
    tile_n: Int,
    simd_width: Int,
    //,
](
    a_q_bits_ptr: UnsafePointer[Int8, _],
    mut b_q_bits_ptr: UnsafePointer[UInt8, _],
    mut c_int32_group: _Accumulator[.int32, tile_m, tile_n, simd_width],
):
    comptime group_size = _block_Q4_K.group_size
    comptime tile_k = 2

    def stream_b_vals(
        mut b_vals: Array[
            SIMD[.uint8, SIMDLength(simd_width) * 4], tile_n * tile_k
        ]
    ) {mut b_q_bits_ptr, imm}:
        comptime for col in range(tile_n):
            var packed_bits = b_q_bits_ptr.load[
                width=SIMDLength(simd_width) * 4
            ]()
            b_q_bits_ptr += simd_width * 4

            comptime for i in range(2):
                var bytes = (packed_bits >> UInt8(i * 4)) & 15
                b_vals[col * tile_k + i] = bytes

    _matmul_group_stream[group_size, tile_k=tile_k](
        a_q_bits_ptr, c_int32_group, stream_b_vals
    )


@always_inline
def _matmul_Q4_K_tile[
    tile_m: Int,
    tile_n: Int,
    simd_width: Int,
    //,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    a_ptr: UnsafePointer[_block_Q8_K_packed[_block_Q4_K.group_size], _],
    b_ptr: UnsafePointer[_block_Q4_K_packed[], _],
    b_q_scales_and_mins_buf: UnsafePointer[UInt8, _],
    c_ptr: UnsafePointer[mut=True, Float32, _],
    N: Int,
    accumulate: Bool,
    m: Int,
    n: Int,
    is_last_k_iter: Bool,
    matmul_group_fn: Some[
        # The pointer origin is concrete because a `_` wildcard becomes a
        # distinct anonymous parameter on each side and fails to unify.
        def(
            a_ptr: UnsafePointer[Int8, ImmutAnyOrigin],
            mut c_int32: _Accumulator[.int32, tile_m, tile_n, simd_width],
        ) -> None
    ],
):
    comptime group_size = _block_Q4_K.group_size
    comptime group_count = _block_Q4_K.group_count

    comptime block_n = tile_n * simd_width

    var a_tile_ptr = a_ptr.bitcast[_block_Q8_K_packed[group_size, tile_m]]()
    var b_tile_ptr = b_ptr.bitcast[_block_Q4_K_packed[block_n]]()

    var b_q_scales_ptr = b_q_scales_and_mins_buf
    var b_q_mins_ptr = b_q_scales_and_mins_buf + group_count * block_n

    var c_int32_block = _Accumulator[.int32, tile_m, tile_n, simd_width]()

    c_int32_block.init()

    var a_q_bits_ptr: UnsafePointer[
        Int8, origin_of(a_tile_ptr[].q_bits)
    ] = a_tile_ptr[].q_bits.unsafe_ptr()

    for g in range(group_count):
        var c_int32_group = _Accumulator[.int32, tile_m, tile_n, simd_width]()

        c_int32_group.init()

        # Matrix multiply a single group of the block.
        matmul_group_fn(
            a_q_bits_ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
            c_int32_group,
        )
        a_q_bits_ptr += tile_m * group_size

        # Scale the accumulator for this group and add to the block level
        # accumulators.
        comptime for col in range(tile_n):
            var b_q_scale_val = b_q_scales_ptr.load[width=simd_width](
                col * simd_width + g * block_n
            ).cast[.int32]()

            comptime for row in range(tile_m):
                c_int32_block[row, col] += (
                    c_int32_group[row, col] * b_q_scale_val
                )

    var c_float = _Accumulator[.float32, tile_m, tile_n, simd_width]()

    _apply_base_scales(
        b_tile_ptr[].base_scales.unsafe_ptr(), c_int32_block, c_float
    )

    _apply_zero_point_correction[group_count](
        a_tile_ptr[].group_sums.unsafe_ptr(),
        b_q_mins_ptr,
        b_tile_ptr[].base_mins.unsafe_ptr(),
        c_float,
    )

    _apply_a_scales(a_tile_ptr[].scales.unsafe_ptr(), c_float)

    _accumulate_and_store[elementwise_lambda_fn=elementwise_lambda_fn](
        c_ptr, N, accumulate, c_float, m, n, is_last_k_iter
    )


def _matmul_Q4_K_columns[
    tile_n: Int,
    simd_width: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    var a_ptr: UnsafePointer[
        mut=True, _block_Q8_K_packed[_block_Q4_K.group_size], _
    ],
    b_ptr: UnsafePointer[_block_Q4_K_packed[], _],
    var c_ptr: UnsafePointer[mut=True, Float32, _],
    M: Int,
    N: Int,
    accumulate: Bool,
    n: Int,
    is_last_k_iter: Bool,
):
    comptime group_size = _block_Q4_K.group_size
    comptime group_count = _block_Q4_K.group_count

    comptime alignment = align_of[SIMD[.float32, simd_width]]()
    comptime block_n = tile_n * simd_width

    var b_tile_ptr = b_ptr.bitcast[_block_Q4_K_packed[block_n]]()

    # Unpack the scales and minimums to uint8 values.
    var b_q_scales_and_mins_buf = unsafe_stack_allocation[
        2 * group_count * block_n, DType.uint8, alignment=alignment
    ]()
    b_tile_ptr[].q_scales_and_mins.unpack(b_q_scales_and_mins_buf)

    # Fast path for M=1 that avoids materializing the unpacked weights.
    if M == 1:
        # The pointee is read-only, and the tracked origin must be erased so
        # the closure's captured pointer does not alias the `b_ptr` argument
        # in the `_matmul_Q4_K_tile` call below.
        var b_q_bits_ptr = (
            b_tile_ptr[]
            .q_bits.bits.unsafe_ptr()
            .as_imm()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )

        def matmul_group_packed(
            a_q_bits_ptr: UnsafePointer[Int8, ImmutAnyOrigin],
            mut c_int32_group: _Accumulator[.int32, 1, tile_n, simd_width],
        ) {mut b_q_bits_ptr, imm}:
            _matmul_group_packed_Q4_K(a_q_bits_ptr, b_q_bits_ptr, c_int32_group)

        _matmul_Q4_K_tile[elementwise_lambda_fn=elementwise_lambda_fn,](
            a_ptr,
            b_ptr,
            b_q_scales_and_mins_buf,
            c_ptr,
            N,
            accumulate,
            0,
            n,
            is_last_k_iter,
            matmul_group_packed,
        )
        _ = b_q_bits_ptr

        return

    # Unpack the quantized bits to uint8 values.
    var b_q_bits = unsafe_stack_allocation[
        _block_QK_K.quantized_k * block_n, DType.uint8, alignment=alignment
    ]()
    b_tile_ptr[].q_bits.unpack(b_q_bits)

    @always_inline
    def process_rows[
        tile_m: Int
    ](m: Int) {
        var b_q_scales_and_mins_buf,
        var b_q_bits,
        mut a_ptr,
        mut c_ptr,
        imm,
    }:
        var b_q_bits_ptr = b_q_bits.unsafe_origin_cast[MutAnyOrigin]()

        def matmul_group_unpacked(
            a_ptr: UnsafePointer[Int8, ImmutAnyOrigin],
            mut c_int32_group: _Accumulator[.int32, tile_m, tile_n, simd_width],
        ) {mut b_q_bits_ptr, imm}:
            _matmul_group_unpacked[group_size](
                a_ptr, b_q_bits_ptr, c_int32_group
            )

        _matmul_Q4_K_tile[elementwise_lambda_fn=elementwise_lambda_fn,](
            a_ptr,
            b_ptr,
            b_q_scales_and_mins_buf,
            c_ptr,
            N,
            accumulate,
            m,
            n,
            is_last_k_iter,
            matmul_group_unpacked,
        )
        _ = b_q_bits_ptr

        a_ptr += tile_m
        c_ptr += tile_m * N

    tile[[4, 2, 1]](0, M, process_rows)


@always_inline
def _matmul_group_packed_Q6_K[
    tile_m: Int,
    tile_n: Int,
    simd_width: Int,
    //,
    *,
    zero_point: UInt8,
](
    a_q_bits_ptr: UnsafePointer[Int8, _],
    mut b_q_bits_ptr: UnsafePointer[UInt8, _],
    mut c_int32_group: _Accumulator[.int32, tile_m, tile_n, simd_width],
):
    comptime group_size = _block_Q6_K.group_size
    comptime tile_k = 4

    def stream_b_vals(
        mut b_vals: Array[
            SIMD[.uint8, SIMDLength(simd_width) * 4], tile_n * tile_k
        ]
    ) {mut b_q_bits_ptr, imm}:
        comptime for col in range(tile_n):
            var hi_bytes = SIMD[.uint8, length=SIMDLength(simd_width) * 4](0)

            comptime for i in range(3):
                var packed_bits = b_q_bits_ptr.load[
                    width=SIMDLength(simd_width) * 4
                ]()
                b_q_bits_ptr += simd_width * 4

                var bytes = packed_bits & 63
                b_vals[col * tile_k + i] = bytes - zero_point

                hi_bytes |= (packed_bits >> 6) << UInt8(i * 2)

            b_vals[col * tile_k + 3] = hi_bytes - zero_point

    _matmul_group_stream[group_size, tile_k=tile_k](
        a_q_bits_ptr, c_int32_group, stream_b_vals
    )


@always_inline
def _matmul_Q6_K_tile[
    tile_m: Int,
    tile_n: Int,
    simd_width: Int,
    //,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    a_ptr: UnsafePointer[
        mut=False, _block_Q8_K_packed[_block_Q6_K.group_size], _
    ],
    b_ptr: UnsafePointer[mut=False, _block_Q6_K_packed[], _],
    c_ptr: UnsafePointer[mut=True, Float32, _],
    N: Int,
    accumulate: Bool,
    m: Int,
    n: Int,
    is_last_k_iter: Bool,
    matmul_group_fn: Some[
        # See `_matmul_Q4_K_tile` for why the origin is concrete.
        def(
            a_ptr: UnsafePointer[Int8, ImmutAnyOrigin],
            mut c_int32_group: _Accumulator[.int32, tile_m, tile_n, simd_width],
        ) -> None
    ],
):
    comptime group_size = _block_Q6_K.group_size
    comptime group_count = _block_Q6_K.group_count

    comptime block_n = tile_n * simd_width

    var a_tile_ptr = a_ptr.bitcast[_block_Q8_K_packed[group_size, tile_m]]()
    var b_tile_ptr = b_ptr.bitcast[_block_Q6_K_packed[block_n]]()

    var c_int32_block = _Accumulator[.int32, tile_m, tile_n, simd_width]()

    c_int32_block.init()

    var a_q_bits_ptr: UnsafePointer[
        Int8, origin_of(a_tile_ptr[].q_bits)
    ] = a_tile_ptr[].q_bits.unsafe_ptr()

    for g in range(group_count):
        var c_int32_group = _Accumulator[.int32, tile_m, tile_n, simd_width]()

        c_int32_group.init()

        comptime if CompilationTarget.has_sse4():
            # Initialize the accumulators with the zero point correction
            # values. This is necessary for x86 as there are no VNNI
            # instructions for s8s8.
            comptime for row in range(tile_m):
                var group_sum = (
                    a_tile_ptr[].group_sums[g * tile_m + row].cast[.int32]()
                )
                var correction_val = SIMD[.int32, simd_width](-32 * group_sum)

                comptime for col in range(tile_n):
                    c_int32_group[row, col] = correction_val
        # Matrix multiply a single group of the block.
        matmul_group_fn(
            a_q_bits_ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
            c_int32_group,
        )

        a_q_bits_ptr += tile_m * group_size

        var b_q_scales_ptr: UnsafePointer[
            Int8, origin_of(b_tile_ptr[].q_scales)
        ] = b_tile_ptr[].q_scales.unsafe_ptr()

        # Scale the accumulator for this group and add to the block level
        # accumulators.
        comptime for col in range(tile_n):
            var b_q_scale_val = b_q_scales_ptr.load[width=simd_width](
                col * simd_width + g * block_n
            ).cast[.int32]()

            comptime for row in range(tile_m):
                c_int32_block[row, col] += (
                    c_int32_group[row, col] * b_q_scale_val
                )

    var c_float = _Accumulator[.float32, tile_m, tile_n, simd_width]()

    _apply_base_scales(
        b_tile_ptr[].base_scales.unsafe_ptr(), c_int32_block, c_float
    )

    _apply_a_scales(a_tile_ptr[].scales.unsafe_ptr(), c_float)

    _accumulate_and_store[elementwise_lambda_fn=elementwise_lambda_fn](
        c_ptr, N, accumulate, c_float, m, n, is_last_k_iter
    )


def _matmul_Q6_K_columns[
    tile_n: Int,
    simd_width: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    var a_ptr: UnsafePointer[
        mut=True, _block_Q8_K_packed[_block_Q6_K.group_size], _
    ],
    b_ptr: UnsafePointer[_block_Q6_K_packed[], _],
    var c_ptr: UnsafePointer[mut=True, Float32, _],
    M: Int,
    N: Int,
    accumulate: Bool,
    n: Int,
    is_last_k_iter: Bool,
):
    comptime group_size = _block_Q6_K.group_size
    comptime group_count = _block_Q6_K.group_count

    comptime alignment = align_of[SIMD[.float32, simd_width]]()
    comptime block_n = tile_n * simd_width

    var b_tile_ptr = b_ptr.bitcast[_block_Q6_K_packed[block_n]]()

    # NEON has support for s8s8 dot products, so shift the quantized bits down
    # to avoid performing any zero point corrections.
    comptime b_zero_point = 32 if CompilationTarget.has_neon() else 0

    # Fast path for M=1 that avoids materializing the unpacked weights.
    if M == 1:
        # See the Q4_K fast path for why the origin is erased.
        var b_q_bits_ptr = (
            b_tile_ptr[]
            .q_bits.bits.unsafe_ptr()
            .as_imm()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )

        def matmul_group_packed(
            a_q_bits_ptr: UnsafePointer[Int8, ImmutAnyOrigin],
            mut c_int32_group: _Accumulator[.int32, 1, tile_n, simd_width],
        ) {mut b_q_bits_ptr, imm}:
            _matmul_group_packed_Q6_K[zero_point=UInt8(b_zero_point)](
                a_q_bits_ptr, b_q_bits_ptr, c_int32_group
            )

        _matmul_Q6_K_tile[elementwise_lambda_fn=elementwise_lambda_fn,](
            a_ptr,
            b_ptr,
            c_ptr,
            N,
            accumulate,
            0,
            n,
            is_last_k_iter,
            matmul_group_packed,
        )
        _ = b_q_bits_ptr

        return

    # Unpack the quantized bits to uint8 values.
    var b_q_bits = unsafe_stack_allocation[
        _block_QK_K.quantized_k * block_n, DType.uint8, alignment=alignment
    ]()
    b_tile_ptr[].q_bits.unpack[zero_point=UInt8(b_zero_point)](b_q_bits)

    @always_inline
    def process_rows[
        tile_m: Int
    ](m: Int) {var b_q_bits, mut a_ptr, mut c_ptr, imm}:
        var b_q_bits_ptr = b_q_bits.as_unsafe_any_origin()

        def matmul_group_unpacked(
            a_ptr: UnsafePointer[Int8, ImmutAnyOrigin],
            mut c_int32_group: _Accumulator[.int32, tile_m, tile_n, simd_width],
        ) {mut b_q_bits_ptr}:
            _matmul_group_unpacked[group_size](
                a_ptr, b_q_bits_ptr, c_int32_group
            )

        _matmul_Q6_K_tile[elementwise_lambda_fn=elementwise_lambda_fn,](
            a_ptr,
            b_ptr,
            c_ptr,
            N,
            accumulate,
            m,
            n,
            is_last_k_iter,
            matmul_group_unpacked,
        )
        _ = b_q_bits_ptr

        a_ptr += tile_m
        c_ptr += tile_m * N

    tile[[4, 2, 1]](0, M, process_rows)


@always_inline
def _matmul_Qb_K[
    group_size: Int,
    b_type: AnyType,
    //,
    columns_fn: def[
        tile_n: Int,
        simd_width: Int,
        elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    ](
        var a_ptr: UnsafePointer[mut=True, _block_Q8_K_packed[group_size], _],
        b_ptr: UnsafePointer[b_type, _],
        var c_ptr: UnsafePointer[mut=True, Float32, _],
        M: Int,
        N: Int,
        accumulate: Bool,
        n: Int,
        is_last_k_iter: Bool,
    ) capturing -> None,
    *,
    interleave_group_sums: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    a: LayoutTensor[mut=False, .float32, address_space=.GENERIC, ...],
    b: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    c: LayoutTensor[mut=True, .float32, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
):
    comptime assert a.rank == 2
    comptime assert b.rank == 2
    comptime assert c.rank == 2

    comptime simd_width = simd_width_of[DType.float32]()

    var M = a.dim[0]()
    var N = b.dim[0]()
    var K = a.dim[1]()
    var k_blocks = K // _block_QK_K.quantized_k

    var a_packed_base_alloc = _quantize_a_Q8_K[
        group_size, interleave_group_sums=interleave_group_sums
    ](a)
    var a_packed_base_ptr: UnsafePointer[
        _block_Q8_K_packed[group_size], origin_of(a_packed_base_alloc._alloc)
    ] = a_packed_base_alloc.unsafe_ptr()

    comptime grain_size = simd_width * 2

    var work_count = ceildiv(N, grain_size)
    var num_workers = min(work_count, parallelism_level(ctx))

    def task_func(
        task_id: Int,
    ) {
        var a_packed_base_ptr,
        var k_blocks,
        var M,
        var N,
        var K,
        var work_count,
        var num_workers,
        imm,
    }:
        var block_range = partition_work(task_id, num_workers, work_count, 1)
        var task_n_start = block_range[0] * grain_size
        var task_n_count = block_range[1] * grain_size

        var a_packed_ptr = a_packed_base_ptr
        var b_packed_ptr = b.ptr.bitcast[b_type]()

        for k_block in range(k_blocks):
            var bn_packed_ptr = b_packed_ptr + task_n_start
            var cn_ptr = c.ptr + task_n_start
            var accumulate = k_block > 0

            # only run epilogue for the last iter of K loop
            var is_last_k_iter = k_block == k_blocks - 1

            @always_inline
            def process_cols[
                tile_n: Int
            ](n_idx: Int) {
                mut a_packed_ptr, mut bn_packed_ptr, mut cn_ptr, imm
            }:
                columns_fn[
                    tile_n,
                    simd_width,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](
                    a_packed_ptr,
                    bn_packed_ptr,
                    cn_ptr,
                    M,
                    N,
                    accumulate,
                    task_n_start + n_idx * simd_width,
                    is_last_k_iter,
                )

                bn_packed_ptr += tile_n * simd_width
                cn_ptr += tile_n * simd_width

            tile[[2, 1]](0, ceildiv(task_n_count, simd_width), process_cols)
            # TODO(MOCO-2074): Suppress false positive unused var warning.
            _ = bn_packed_ptr
            _ = cn_ptr
            _ = accumulate
            _ = is_last_k_iter

            a_packed_ptr += M
            b_packed_ptr += N

    sync_parallelize(task_func, num_workers, ctx)

    dealloc(a_packed_base_alloc^)


def matmul_Q4_K[
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None
](
    a_tt: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    b_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    c_tt: TileTensor[mut=True, .float32, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
):
    """Computes a matrix multiplication with Q4_K block-quantized weights.

    Dispatches to an x86 or ARM NEON implementation at compile time; other
    targets fail to compile.

    Parameters:
        elementwise_lambda_fn: Optional epilogue applied to each output element.

    Args:
        a_tt: Left-hand operand tensor in float32.
        b_tt: Right-hand operand tensor holding Q4_K quantized uint8 weights.
        c_tt: Output tensor in float32.
        ctx: Optional device context for parallel execution.
    """
    var a = a_tt.to_layout_tensor()
    var b = b_tt.to_layout_tensor()
    var c = c_tt.to_layout_tensor()
    comptime assert a.rank == 2
    comptime assert b.rank == 2
    comptime assert c.rank == 2

    _matmul_Qb_K[
        group_size=_block_Q4_K.group_size,
        b_type=_block_Q4_K_packed[],
        columns_fn=_matmul_Q4_K_columns,
        interleave_group_sums=True,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](a, b, c, ctx)


def matmul_Q6_K[
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None
](
    a_tt: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    b_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    c_tt: TileTensor[mut=True, .float32, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
):
    """Computes a matrix multiplication with Q6_K block-quantized weights.

    Dispatches to an x86 or ARM NEON implementation at compile time; other
    targets fail to compile.

    Parameters:
        elementwise_lambda_fn: Optional epilogue applied to each output element.

    Args:
        a_tt: Left-hand operand tensor in float32.
        b_tt: Right-hand operand tensor holding Q6_K quantized uint8 weights.
        c_tt: Output tensor in float32.
        ctx: Optional device context for parallel execution.
    """
    var a = a_tt.to_layout_tensor()
    var b = b_tt.to_layout_tensor()
    var c = c_tt.to_layout_tensor()
    comptime assert a.rank == 2
    comptime assert b.rank == 2
    comptime assert c.rank == 2

    _matmul_Qb_K[
        group_size=_block_Q6_K.group_size,
        b_type=_block_Q6_K_packed[],
        columns_fn=_matmul_Q6_K_columns,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](a, b, c, ctx)
