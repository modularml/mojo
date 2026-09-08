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
"""Provides CPU kernels for block-wise quantized int4 matrix multiplication."""

from std.collections import Optional
from std.math import ceildiv
from std.sys import CompilationTarget, align_of, simd_width_of, size_of

from std.algorithm import tile

from max.algorithm import sync_parallelize
from max.gpu.host import DeviceContext
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    DefaultEngine,
)
from linalg.accumulate import _Accumulator
from linalg.arch.cpu.neon_intrinsics import _neon_dotprod_lane, _neon_matmul
from linalg.arch.cpu.vnni_intrinsics import (
    dot_i8_to_i32_saturated_x86,
    pmaddubs,
    pmaddw,
)
from linalg.matmul import elementwise_epilogue_type
from linalg.utils import partition_work
from std.memory import (
    alloc,
    bitcast,
    dealloc,
    unsafe_stack_allocation,
)
from std.memory.alloc import Layout as AllocLayout

from max.runtime.asyncrt import parallelism_level

from std.utils.index import Index

from ._utils import roundeven_to_int32

comptime K_BATCH_SIZE = 512
"""Defines the batch size of K used to pack A and unpack B weights."""


def matmul_qint4_pack_b[
    group_size: Int
](
    b: TileTensor[
        mut=False, .uint8, address_space=.GENERIC, Engine=DefaultEngine[], ...
    ],
    b_rot: TileTensor[
        mut=True, .uint8, address_space=.GENERIC, Engine=DefaultEngine[], ...
    ],
) raises:
    """Repacks block-wise quantized int4 weights into the tiled layout
    expected by the `matmul_qint4` kernels.

    Parameters:
        group_size: Number of elements per quantization group.

    Args:
        b: Source tensor holding packed uint8 weights with float16 scales.
        b_rot: Destination tensor for the repacked weights.

    Raises:
        If N is not a multiple of 32.
    """
    comptime assert b.rank == 2
    comptime assert b_rot.rank == 2
    comptime n_tiles = 2
    comptime n_groups = n_tiles * simd_width_of[DType.float32]()
    comptime bytes_per_group_int4 = size_of[DType.float16]() + (group_size // 2)

    var N = Int(b.dim[0]())
    var K = Int(b.dim[1]()) // bytes_per_group_int4 * group_size

    if N % 32 != 0:
        raise ("N must be a multiple of 32")

    var k_groups = ceildiv(K, group_size)

    var src_ptr = b._storage
    var dst_ptr = b_rot._storage

    for _ in range(0, N, n_groups):
        for nn in range(n_groups):
            var dst_k_ptr = dst_ptr
            for _ in range(0, K, group_size):
                var scale = src_ptr.unsafe_bitcast[Float16]().unsafe_load()
                dst_k_ptr.unsafe_bitcast[Float16]().unsafe_store(nn, scale)
                src_ptr = src_ptr.unsafe_offset(size_of[DType.float16]())
                dst_k_ptr = dst_k_ptr.unsafe_offset(
                    size_of[DType.float16]() * n_groups
                )

                var b_data_i4 = src_ptr.unsafe_load[width=group_size // 2]()
                src_ptr = src_ptr.unsafe_offset(group_size // 2)

                var b_data_i8_lo = b_data_i4 & 15
                var b_data_i8_hi = b_data_i4 >> 4
                var b_data_i8 = b_data_i8_lo.join(b_data_i8_hi)

                comptime for i in range(0, group_size, 8):
                    var b_tuple_lo = b_data_i8.slice[4, offset=i]()
                    var b_tuple_hi = b_data_i8.slice[4, offset=i + 4]()
                    var b_tuple = (b_tuple_lo << 0) + (b_tuple_hi << 4)
                    dst_k_ptr.unsafe_offset(4 * nn).unsafe_store(b_tuple)
                    dst_k_ptr = dst_k_ptr.unsafe_offset(4 * n_groups)

        dst_ptr = dst_ptr.unsafe_offset(
            n_groups * k_groups * bytes_per_group_int4
        )


def _quantize_a_block[
    group_size: Int, aq_type: DType, dtype: DType
](a_ptr: ImmPointer[Scalar[dtype], _]) -> Tuple[
    SIMD[aq_type, group_size], Float32
]:
    comptime a_zero_point = 128 if aq_type.is_unsigned() else 0

    var fp_data = a_ptr.unsafe_load[width=group_size]()
    var max_value = abs(fp_data).reduce_max()
    var scale = (max_value / 127.0).cast[.float32]()
    var multiplier = 127.0 / max_value if max_value != 0.0 else 0.0

    var quant_data_s8 = roundeven_to_int32(fp_data * multiplier).cast[.int8]()
    var quant_data = quant_data_s8.cast[aq_type]() + Scalar[aq_type](
        a_zero_point
    )

    return (quant_data, scale)


def _quantize_a_buffer[
    group_size: Int,
    dtype: DType,
    aq_type: DType,
    *,
    aq_interleave: Int = group_size,
](
    a: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    a_quant: LayoutTensor[mut=True, aq_type, ...],
    a_scale: LayoutTensor[mut=True, .float32, ...],
):
    """Converts a floating point buffer to a symmetrically quantized
    representation. The data is in a packed layout that can be efficiently
    indexed by the matrix multiply kernels.
    """
    comptime assert (
        group_size % aq_interleave
    ) == 0, "interleave must be a factor of group size"
    comptime assert a.rank == 2
    comptime assert a_quant.rank == 2
    comptime assert a_scale.rank == 2

    var M = a.dim[0]()
    var K = a.dim[1]()

    var a_quant_ptr = a_quant.ptr
    var a_scale_ptr = a_scale.ptr

    # Pack the quantized integers and scales in batches of K.
    for ko in range(0, K, K_BATCH_SIZE):
        var ko_count = min(K_BATCH_SIZE, K - ko)

        var am_ptr = a.ptr.unsafe_offset(ko)

        @always_inline
        def process_rows[
            tile_m: Int
        ](m: Int) {mut am_ptr, mut a_quant_ptr, mut a_scale_ptr, imm}:
            for row in range(tile_m):
                var ak_quant_ptr = a_quant_ptr.unsafe_offset(
                    row * aq_interleave
                )
                var ak_scale_ptr = a_scale_ptr.unsafe_offset(row)

                for ki in range(0, ko_count, group_size):
                    var quant_data: SIMD[aq_type, group_size]
                    var scale: Float32
                    (quant_data, scale) = _quantize_a_block[
                        group_size, aq_type
                    ](am_ptr.unsafe_offset(ki))

                    # Interleave this local block to the output buffer.
                    #
                    # This supports the i8mm use case where the instruction
                    # expects a 2x8 matrix of data loaded from two rows. This
                    # loop slices and outputs data at the `tile_m` stride.
                    #
                    # For the non-i8mm use case, no interleaving occurs and
                    # this is a simple store.
                    #
                    # For either case, when M=1, the data layout is effectively
                    # a flat array of data. The M=1 kernels assume this and
                    # ignore the K batching and interleave concepts.
                    comptime for i in range(0, group_size, aq_interleave):
                        ak_quant_ptr.unsafe_store(
                            quant_data.slice[aq_interleave, offset=i](),
                        )
                        ak_quant_ptr = ak_quant_ptr.unsafe_offset(
                            tile_m * aq_interleave
                        )

                    ak_scale_ptr.unsafe_store(scale)
                    ak_scale_ptr = ak_scale_ptr.unsafe_offset(tile_m)

                am_ptr = am_ptr.unsafe_offset(K)

            a_quant_ptr = a_quant_ptr.unsafe_offset(tile_m * ko_count)
            a_scale_ptr = a_scale_ptr.unsafe_offset(
                tile_m * (ko_count // group_size)
            )

        tile[[4, 2, 1]](0, M, process_rows)


def _unpack_weights[
    group_size: Int,
    tile_n: Int,
    simd_width: Int,
    needs_correction: Bool,
    is_i8mm: Bool,
](
    _b_s8_ptr: MutPointer[Int8, _],
    _b_packed_ptr: ImmPointer[UInt8, _],
    _b_scale_ptr: MutPointer[Float32, _],
    _b_correction_ptr: MutPointer[Int32, _],
    batch_k: Int,
):
    var b_s8_ptr = _b_s8_ptr
    var b_packed_ptr = _b_packed_ptr
    var b_scale_ptr = _b_scale_ptr
    var b_correction_ptr = _b_correction_ptr

    for _ in range(0, batch_k, group_size):
        comptime for col in range(tile_n):
            var b_scale = (
                b_packed_ptr.unsafe_bitcast[Float16]()
                .unsafe_load[width=simd_width](col * simd_width)
                .cast[.float32]()
            )
            b_scale_ptr.unsafe_store(col * simd_width, b_scale)

        b_scale_ptr = b_scale_ptr.unsafe_offset(tile_n * simd_width)
        b_packed_ptr = b_packed_ptr.unsafe_offset(
            size_of[DType.float16]() * tile_n * simd_width
        )

        var b_column_sums = Array[SIMD[.int32, simd_width], tile_n](fill=0)

        for _ in range(0, group_size, 8):
            comptime for col in range(tile_n):
                var b_data_packed = b_packed_ptr.unsafe_load[
                    width=simd_width * 4
                ](col * simd_width * 4).cast[.uint8]()
                var b_data_i4_lo = (b_data_packed & 15).cast[.int8]() - 8
                var b_data_i4_hi = (b_data_packed >> 4).cast[.int8]() - 8

                comptime if needs_correction:
                    comptime a_zero_point = SIMD[.uint8, simd_width * 4](128)
                    var a_zp = bitcast[.int32, simd_width](a_zero_point)
                    var b_lo = bitcast[.int32, simd_width](b_data_i4_lo)
                    var b_hi = bitcast[.int32, simd_width](b_data_i4_hi)

                    comptime if CompilationTarget.has_vnni():
                        b_column_sums[col] = dot_i8_to_i32_saturated_x86(
                            b_column_sums[col], a_zp, b_lo
                        )
                        b_column_sums[col] = dot_i8_to_i32_saturated_x86(
                            b_column_sums[col], a_zp, b_hi
                        )
                    else:
                        # Get the partial 16-bit dot product low and high.
                        # The full 32-bit dot product is finished in the
                        # apply_a_scale_avx2 function.
                        var pdot_lo = bitcast[.int16, 2 * simd_width](
                            pmaddubs(a_zp, b_lo)
                        )
                        var pdot_hi = bitcast[.int16, 2 * simd_width](
                            pmaddubs(a_zp, b_hi)
                        )
                        var ci16 = bitcast[.int16, 2 * simd_width](
                            b_column_sums[col]
                        )
                        # Add the low and high 16-bit partial dot products.
                        ci16 -= pdot_lo + pdot_hi

                        b_column_sums[col] = bitcast[.int32, simd_width](ci16)

                comptime if is_i8mm:
                    var intl = bitcast[.int32, simd_width](
                        b_data_i4_lo
                    ).interleave(bitcast[.int32, simd_width](b_data_i4_hi))
                    b_data_i4_lo = bitcast[.int8, simd_width * 4](
                        intl.slice[simd_width, offset=0]()
                    )
                    b_data_i4_hi = bitcast[.int8, simd_width * 4](
                        intl.slice[simd_width, offset=simd_width]()
                    )

                    b_s8_ptr.unsafe_store(col * simd_width * 8, b_data_i4_lo)
                    b_s8_ptr.unsafe_store(
                        col * simd_width * 8 + (tile_n // 2) * simd_width * 4,
                        b_data_i4_hi,
                    )

                else:
                    b_s8_ptr.unsafe_store(col * simd_width * 4, b_data_i4_lo)
                    b_s8_ptr.unsafe_store(
                        col * simd_width * 4 + tile_n * simd_width * 4,
                        b_data_i4_hi,
                    )

            b_s8_ptr = b_s8_ptr.unsafe_offset(2 * tile_n * simd_width * 4)
            b_packed_ptr = b_packed_ptr.unsafe_offset(tile_n * simd_width * 4)

        comptime if needs_correction:
            comptime for col in range(tile_n):
                b_correction_ptr.unsafe_store(
                    simd_width * col,
                    -b_column_sums[
                        col
                    ] if CompilationTarget.has_vnni() else b_column_sums[col],
                )

            b_correction_ptr = b_correction_ptr.unsafe_offset(
                tile_n * simd_width
            )


@always_inline
def _scale_and_accumulate[
    group_size: Int,
    b_scale_type: DType,
    tile_m: Int,
    tile_n: Int,
    simd_width: Int,
](
    a_scale_ptr: ImmPointer[Float32, _],
    b_scale_ptr: ImmPointer[Scalar[b_scale_type], _],
    mut c_int32: _Accumulator[.int32, tile_m, tile_n, simd_width],
    mut c_float: _Accumulator[.float32, tile_m, tile_n, simd_width],
):
    # Load the per-column scale values for the B matrix.
    comptime ScaleType = SIMD[.float32, simd_width]
    var b_scale = Array[_, tile_n](
        fill_with=lambda (col: Int) -> ScaleType: b_scale_ptr.unsafe_load[
            width=simd_width
        ](col * simd_width).cast[.float32]()
    )

    @__parameter
    @always_inline
    def apply_a_scale[row: Int](a_scale: Float32):
        comptime for col in range(tile_n):
            var dot = c_int32[row, col]

            # Without VNNI on x86 the 2-wide 8-bit to 16-bit dot
            # product was calculated in process_group_packed.
            # Now complete the 4-wide 8-bit to 32-bit dot product.
            comptime if (
                CompilationTarget.has_avx2()
                and not CompilationTarget.has_vnni()
            ):
                dot = pmaddw(
                    dot,
                    bitcast[.int32, simd_width](
                        SIMD[.int16, 2 * simd_width](1)
                    ),
                )

            c_float[row, col] += dot.cast[.float32]() * a_scale * b_scale[col]

    # Convert and rescale the integer accumulators and accumulate to the output
    # float accumulators.
    comptime if CompilationTarget.has_neon():
        # NEON supports a multiply instruction that can broadcast from a
        # vector element, so help the compiler produce that by doing a vector
        # load.
        var a_scale = a_scale_ptr.unsafe_load[width=tile_m]()

        comptime for row in range(tile_m):
            apply_a_scale[row](a_scale[row])

    else:
        comptime for row in range(tile_m):
            apply_a_scale[row](a_scale_ptr.unsafe_load(row))


trait _MatmulQInt4Kernel:
    @staticmethod
    def aq_type() -> DType:
        """Returns the type to use for representing quantized A data."""
        ...

    @staticmethod
    def aq_tuple_type() -> DType:
        """Returns the type to use for representing tuples of quantized A data.
        """
        ...

    @staticmethod
    def quantize_a_buffer[
        group_size: Int, dtype: DType, aq_type: DType
    ](
        a: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
        a_quant: LayoutTensor[mut=True, aq_type, ...],
        a_scale: LayoutTensor[mut=True, .float32, ...],
    ):
        ...

    @staticmethod
    def process_group_packed[
        group_size: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: ImmPointer[Int8, _],
        a_scale_ptr: ImmPointer[Float32, _],
        b_ptr: ImmPointer[Int8, _],
        mut c_float: _Accumulator[.float32, 1, tile_n, simd_width],
    ):
        ...

    @staticmethod
    def process_group_unpacked[
        group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: ImmPointer[Int8, _],
        a_scale_ptr: ImmPointer[Float32, _],
        b_base_ptr: ImmPointer[Int8, _],
        b_ptr: ImmPointer[Float32, _],
        b_correction_ptr: ImmPointer[Int32, _],
        mut c_float: _Accumulator[.float32, tile_m, tile_n, simd_width],
    ):
        ...


struct _MatmulQInt4Kernel_x86_vnni(_MatmulQInt4Kernel):
    @always_inline
    @staticmethod
    def aq_type() -> DType:
        return .uint8

    @always_inline
    @staticmethod
    def aq_tuple_type() -> DType:
        return .int32

    @always_inline
    @staticmethod
    def quantize_a_buffer[
        group_size: Int, dtype: DType, aq_type: DType
    ](
        a: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
        a_quant: LayoutTensor[mut=True, aq_type, ...],
        a_scale: LayoutTensor[mut=True, .float32, ...],
    ):
        comptime assert a.rank == 2
        comptime assert a_quant.rank == 2
        comptime assert a_scale.rank == 2
        return _quantize_a_buffer[group_size](a, a_quant, a_scale)

    @staticmethod
    def process_group_packed[
        group_size: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: ImmPointer[Int8, _],
        a_scale_ptr: ImmPointer[Float32, _],
        b_ptr: ImmPointer[Int8, _],
        mut c_float: _Accumulator[.float32, 1, tile_n, simd_width],
    ):
        var c_int32 = _Accumulator[.int32, 1, tile_n, simd_width]()

        c_int32.init()

        # Skip over the float16 scales.
        var b_offset = size_of[DType.float16]() * tile_n * simd_width

        var b_column_sums = Array[SIMD[.int32, simd_width], tile_n](fill=0)

        comptime for k in range(0, group_size, 8):
            var a_val_lo = bitcast[.int32, 1](a_ptr.unsafe_load[width=4](k))
            var a_val_hi = bitcast[.int32, 1](a_ptr.unsafe_load[width=4](k + 4))

            comptime for col in range(tile_n):
                var b_data_packed = b_ptr.unsafe_load[width=simd_width * 4](
                    b_offset
                ).cast[.uint8]()
                b_offset += simd_width * 4

                var b_data_i4_lo = (b_data_packed & 15).cast[.int8]() - 8
                var b_data_i4_hi = (b_data_packed >> 4).cast[.int8]() - 8

                comptime a_zero_point = SIMD[.uint8, simd_width * 4](128)

                b_column_sums[col] = dot_i8_to_i32_saturated_x86(
                    b_column_sums[col],
                    bitcast[.int32, simd_width](a_zero_point),
                    bitcast[.int32, simd_width](b_data_i4_lo),
                )
                b_column_sums[col] = dot_i8_to_i32_saturated_x86(
                    b_column_sums[col],
                    bitcast[.int32, simd_width](a_zero_point),
                    bitcast[.int32, simd_width](b_data_i4_hi),
                )

                c_int32[0, col] = dot_i8_to_i32_saturated_x86(
                    c_int32[0, col],
                    SIMD[.int32, simd_width](a_val_lo),
                    bitcast[.int32, simd_width](b_data_i4_lo),
                )
                c_int32[0, col] = dot_i8_to_i32_saturated_x86(
                    c_int32[0, col],
                    SIMD[.int32, simd_width](a_val_hi),
                    bitcast[.int32, simd_width](b_data_i4_hi),
                )

        comptime for col in range(tile_n):
            c_int32[0, col] -= b_column_sums[col]

        var b_scale_ptr = b_ptr.unsafe_bitcast[Float16]()

        _scale_and_accumulate[group_size](
            a_scale_ptr, b_scale_ptr, c_int32, c_float
        )

    @always_inline
    @staticmethod
    def process_group_unpacked[
        group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: ImmPointer[Int8, _],
        a_scale_ptr: ImmPointer[Float32, _],
        b_ptr: ImmPointer[Int8, _],
        b_scale_ptr: ImmPointer[Float32, _],
        b_correction_ptr: ImmPointer[Int32, _],
        mut c_float: _Accumulator[.float32, tile_m, tile_n, simd_width],
    ):
        var c_int32 = _Accumulator[.int32, tile_m, tile_n, simd_width]()

        # Initialize the integer accumulators with the zero point corrections.
        comptime for col in range(tile_n):
            var correction_val = b_correction_ptr.unsafe_load[width=simd_width](
                col * simd_width
            )

            comptime for row in range(tile_m):
                c_int32[row, col] = correction_val

        var b_offset = 0

        comptime for k in range(0, group_size, 4):
            comptime for col in range(tile_n):
                var b_val = bitcast[.int32, simd_width](
                    b_ptr.unsafe_load[width=simd_width * 4](b_offset)
                )
                b_offset += simd_width * 4

                comptime for row in range(tile_m):
                    var a_val = SIMD[.int32, simd_width](
                        bitcast[.int32, 1](
                            a_ptr.unsafe_load[width=4](row * group_size + k)
                        )
                    )
                    c_int32[row, col] = dot_i8_to_i32_saturated_x86(
                        c_int32[row, col], a_val, b_val
                    )

        _scale_and_accumulate[group_size](
            a_scale_ptr, b_scale_ptr, c_int32, c_float
        )


struct _MatmulQInt4Kernel_x86_avx(_MatmulQInt4Kernel):
    @always_inline
    @staticmethod
    def aq_type() -> DType:
        return .uint8

    @always_inline
    @staticmethod
    def aq_tuple_type() -> DType:
        return .int32

    @always_inline
    @staticmethod
    def quantize_a_buffer[
        group_size: Int, dtype: DType, aq_type: DType
    ](
        a: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
        a_quant: LayoutTensor[mut=True, aq_type, ...],
        a_scale: LayoutTensor[mut=True, .float32, ...],
    ):
        comptime assert a.rank == 2
        comptime assert a_quant.rank == 2
        comptime assert a_scale.rank == 2
        return _quantize_a_buffer[group_size](a, a_quant, a_scale)

    @staticmethod
    def process_group_packed[
        group_size: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: ImmPointer[Int8, _],
        a_scale_ptr: ImmPointer[Float32, _],
        b_ptr: ImmPointer[Int8, _],
        mut c_float: _Accumulator[.float32, 1, tile_n, simd_width],
    ):
        var c_int32 = _Accumulator[.int32, 1, tile_n, simd_width]()

        c_int32.init()

        # Skip over the float16 scales.
        var b_offset = size_of[DType.float16]() * tile_n * simd_width

        var b_column_sums = Array[SIMD[.int32, simd_width], tile_n](fill=0)

        comptime for k in range(0, group_size, 8):
            var a_lo = SIMD[.int32, simd_width](
                bitcast[.int32, 1](a_ptr.unsafe_load[width=4](k + 0))
            )
            var a_hi = SIMD[.int32, simd_width](
                bitcast[.int32, 1](a_ptr.unsafe_load[width=4](k + 4))
            )

            comptime for col in range(tile_n):
                var b_data_packed = b_ptr.unsafe_load[width=simd_width * 4](
                    b_offset
                ).cast[.uint8]()
                b_offset += simd_width * 4

                var b_data_i4_lo = (b_data_packed & 15).cast[.int8]() - 8
                var b_data_i4_hi = (b_data_packed >> 4).cast[.int8]() - 8

                comptime a_zero_point = SIMD[.uint8, simd_width * 4](128)

                var a_zp = bitcast[.int32, simd_width](a_zero_point)
                var b_lo = bitcast[.int32, simd_width](b_data_i4_lo)
                var b_hi = bitcast[.int32, simd_width](b_data_i4_hi)

                # Get the partial 16-bit dot product low and high.
                # The full 32-bit dot product is finished in the
                # apply_a_scale function.
                var pdot_lo = bitcast[.int16, 2 * simd_width](
                    pmaddubs(a_zp, b_lo)
                )
                var pdot_hi = bitcast[.int16, 2 * simd_width](
                    pmaddubs(a_zp, b_hi)
                )
                var b_column_sum_i16 = bitcast[.int16, 2 * simd_width](
                    b_column_sums[col]
                )
                # Add the low and high 16-bit partial dot products.
                b_column_sum_i16 -= pdot_lo + pdot_hi

                b_column_sums[col] = bitcast[.int32, simd_width](
                    b_column_sum_i16
                )

                var si16_lo = bitcast[.int16, 2 * simd_width](
                    pmaddubs(a_lo, b_lo)
                )
                var si16_hi = bitcast[.int16, 2 * simd_width](
                    pmaddubs(a_hi, b_hi)
                )
                var ci16 = bitcast[.int16, 2 * simd_width](c_int32[0, col])
                ci16 += si16_lo + si16_hi
                c_int32[0, col] = bitcast[.int32, simd_width](ci16)

        comptime for col in range(tile_n):
            var b_column_sum_i16 = bitcast[.int16, 2 * simd_width](
                b_column_sums[col]
            )
            var ci16 = bitcast[.int16, 2 * simd_width](c_int32[0, col])
            ci16 += b_column_sum_i16
            c_int32[0, col] = bitcast[.int32, simd_width](ci16)

        var b_scale_ptr = b_ptr.unsafe_bitcast[Float16]()

        _scale_and_accumulate[group_size](
            a_scale_ptr, b_scale_ptr, c_int32, c_float
        )

    @always_inline
    @staticmethod
    def process_group_unpacked[
        group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: ImmPointer[Int8, _],
        a_scale_ptr: ImmPointer[Float32, _],
        b_ptr: ImmPointer[Int8, _],
        b_scale_ptr: ImmPointer[Float32, _],
        b_correction_ptr: ImmPointer[Int32, _],
        mut c_float: _Accumulator[.float32, tile_m, tile_n, simd_width],
    ):
        var c_int32 = _Accumulator[.int32, tile_m, tile_n, simd_width]()

        # Initialize the integer accumulators with the zero point corrections.
        comptime for col in range(tile_n):
            var correction_val = b_correction_ptr.unsafe_load[width=simd_width](
                col * simd_width
            )

            comptime for row in range(tile_m):
                c_int32[row, col] = correction_val

        var b_offset = 0

        comptime for k in range(0, group_size, 4):
            comptime for col in range(tile_n):
                var b_val = bitcast[.int32, simd_width](
                    b_ptr.unsafe_load[width=simd_width * 4](b_offset)
                )
                b_offset += simd_width * 4

                comptime for row in range(tile_m):
                    var a_val = SIMD[.int32, simd_width](
                        bitcast[.int32, 1](
                            a_ptr.unsafe_load[width=4](row * group_size + k)
                        )
                    )
                    var si16 = bitcast[.int16, 2 * simd_width](
                        pmaddubs(a_val, b_val)
                    )
                    var ci16 = bitcast[.int16, 2 * simd_width](
                        c_int32[row, col]
                    )
                    ci16 += si16
                    c_int32[row, col] = bitcast[.int32, simd_width](ci16)

        _scale_and_accumulate[group_size](
            a_scale_ptr, b_scale_ptr, c_int32, c_float
        )


struct _MatmulQInt4Kernel_neon_dotprod(_MatmulQInt4Kernel):
    @always_inline
    @staticmethod
    def aq_type() -> DType:
        return .int8

    @always_inline
    @staticmethod
    def aq_tuple_type() -> DType:
        return .int32

    @always_inline
    @staticmethod
    def quantize_a_buffer[
        group_size: Int, dtype: DType, aq_type: DType
    ](
        a: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
        a_quant: LayoutTensor[mut=True, aq_type, ...],
        a_scale: LayoutTensor[mut=True, .float32, ...],
    ):
        comptime assert a.rank == 2
        comptime assert a_quant.rank == 2
        comptime assert a_scale.rank == 2
        return _quantize_a_buffer[group_size](a, a_quant, a_scale)

    @staticmethod
    def process_group_packed[
        group_size: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: ImmPointer[Int8, _],
        a_scale_ptr: ImmPointer[Float32, _],
        b_ptr: ImmPointer[Int8, _],
        mut c_float: _Accumulator[.float32, 1, tile_n, simd_width],
    ):
        var c_int32 = _Accumulator[.int32, 1, tile_n, simd_width]()

        c_int32.init()

        # Skip over the float16 scales.
        var b_offset = size_of[DType.float16]() * tile_n * simd_width

        comptime for k in range(0, group_size, 16):
            var a_val = a_ptr.unsafe_load[width=16](k)

            comptime for lane in range(0, 4, 2):
                comptime for col in range(tile_n):
                    var b_data_packed = b_ptr.unsafe_load[
                        width=SIMDLength(simd_width) * 4
                    ](b_offset).cast[.uint8]()
                    b_offset += simd_width * 4

                    var b_data_i4_lo = (b_data_packed & 15).cast[.int8]() - 8
                    var b_data_i4_hi = (b_data_packed >> 4).cast[.int8]() - 8

                    c_int32[0, col] = _neon_dotprod_lane[lane](
                        c_int32[0, col], b_data_i4_lo, a_val
                    )
                    c_int32[0, col] = _neon_dotprod_lane[lane + 1](
                        c_int32[0, col], b_data_i4_hi, a_val
                    )

        var b_scale_ptr = b_ptr.unsafe_bitcast[Float16]()

        _scale_and_accumulate[group_size](
            a_scale_ptr, b_scale_ptr, c_int32, c_float
        )

    @always_inline
    @staticmethod
    def process_group_unpacked[
        group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: ImmPointer[Int8, _],
        a_scale_ptr: ImmPointer[Float32, _],
        b_ptr: ImmPointer[Int8, _],
        b_scale_ptr: ImmPointer[Float32, _],
        b_correction_ptr: ImmPointer[Int32, _],
        mut c_float: _Accumulator[.float32, tile_m, tile_n, simd_width],
    ):
        var c_int32 = _Accumulator[.int32, tile_m, tile_n, simd_width]()

        c_int32.init()

        var b_offset = 0

        comptime AType = SIMD[.int8, 16]
        comptime for k in range(0, group_size, 16):
            var a_tile = Array[_, tile_m](
                fill_with=lambda (row: Int) -> AType: a_ptr.unsafe_load[
                    width=16
                ](row * group_size + k)
            )

            comptime for lane in range(4):
                comptime for col in range(tile_n):
                    var b_val = b_ptr.unsafe_load[
                        width=SIMDLength(simd_width) * 4
                    ](b_offset)
                    b_offset += simd_width * 4

                    comptime for row in range(tile_m):
                        c_int32[row, col] = _neon_dotprod_lane[lane](
                            c_int32[row, col], b_val, a_tile[row]
                        )

        _scale_and_accumulate[group_size](
            a_scale_ptr, b_scale_ptr, c_int32, c_float
        )


struct _MatmulQInt4Kernel_neon_i8mm(_MatmulQInt4Kernel):
    @always_inline
    @staticmethod
    def aq_type() -> DType:
        return .int8

    @always_inline
    @staticmethod
    def aq_tuple_type() -> DType:
        return .int64

    @staticmethod
    def quantize_a_buffer[
        group_size: Int, dtype: DType, aq_type: DType
    ](
        a: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
        a_quant: LayoutTensor[mut=True, aq_type, ...],
        a_scale: LayoutTensor[mut=True, .float32, ...],
    ):
        comptime assert a.rank == 2
        comptime assert a_quant.rank == 2
        comptime assert a_scale.rank == 2

        # Interleave the quantized data to produce the block format required
        # for the NEON `smmla` instruction.
        return _quantize_a_buffer[group_size, aq_interleave=8](
            a, a_quant, a_scale
        )

    @staticmethod
    def process_group_packed[
        group_size: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: ImmPointer[Int8, _],
        a_scale_ptr: ImmPointer[Float32, _],
        b_ptr: ImmPointer[Int8, _],
        mut c_float: _Accumulator[.float32, 1, tile_n, simd_width],
    ):
        # The data layout for quantized A data is identical for the NEON dot
        # product kernel when M=1, so delegate to that implementation.
        _MatmulQInt4Kernel_neon_dotprod.process_group_packed[group_size](
            a_ptr, a_scale_ptr, b_ptr, c_float
        )

    @always_inline
    @staticmethod
    def process_group_unpacked[
        group_size: Int, tile_m: Int, tile_n: Int, simd_width: Int
    ](
        a_ptr: ImmPointer[Int8, _],
        a_scale_ptr: ImmPointer[Float32, _],
        b_ptr: ImmPointer[Int8, _],
        b_scale_ptr: ImmPointer[Float32, _],
        b_correction_ptr: ImmPointer[Int32, _],
        mut c_float: _Accumulator[.float32, tile_m, tile_n, simd_width],
    ):
        comptime block_m = max(tile_m // 2, 1)
        var c_int32_block = _Accumulator[
            .int32, block_m, tile_n * 2, simd_width
        ]()

        c_int32_block.init()

        var a_offset = 0
        var b_offset = 0

        comptime for k in range(0, group_size, 8):
            var a_tile = Array[
                SIMD[.int8, SIMDLength(simd_width) * 4], block_m
            ](fill=0)

            comptime if tile_m > 1:
                comptime for row in range(block_m):
                    a_tile[row] = a_ptr.unsafe_load[
                        width=SIMDLength(simd_width) * 4
                    ](a_offset)
                    a_offset += simd_width * 4
            else:
                var a_val = a_ptr.unsafe_load[width=simd_width * 2](a_offset)
                a_tile[0] = rebind[SIMD[.int8, SIMDLength(simd_width) * 4]](
                    a_val.join(SIMD[.int8, simd_width * 2](0))
                )
                a_offset += simd_width * 2

            comptime for col in range(tile_n * 2):
                var b_val = b_ptr.unsafe_load[width=SIMDLength(simd_width) * 4](
                    b_offset
                )
                b_offset += simd_width * 4

                comptime for row in range(block_m):
                    c_int32_block[row, col] = _neon_matmul(
                        c_int32_block[row, col], a_tile[row], b_val
                    )

        var c_int32 = _Accumulator[.int32, tile_m, tile_n, simd_width]()

        # Swizzle 2x2 blocks to 1x4 vectors:
        # [a0 a1 b0 b1] [a2 a3 b2 b3] -> [a0 a1 a2 a3] [b0 b1 b2 b3]
        #
        # Note that these linear accumulators have a lifetime that overlaps the
        # blocked accumulators from above. Only an extra register is needed to
        # do the swizzling.
        comptime for row in range(0, tile_m, 2):
            comptime for col in range(tile_n):
                var c_val_0 = c_int32_block[row // 2, col * 2]
                var c_val_1 = c_int32_block[row // 2, col * 2 + 1]

                c_int32[row, col] = c_val_0.shuffle[0, 1, 4, 5](c_val_1)

                comptime if tile_m > 1:
                    c_int32[row + 1, col] = c_val_0.shuffle[2, 3, 6, 7](c_val_1)

        _scale_and_accumulate[group_size](
            a_scale_ptr, b_scale_ptr, c_int32, c_float
        )


def _matmul_qint4_m_1[
    kernel: _MatmulQInt4Kernel,
    group_size: Int,
    aq_type: DType,
    b_layout: Layout = Layout.row_major[2](),
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    a_quant: LayoutTensor[mut=False, aq_type, address_space=.GENERIC, ...],
    a_scale: LayoutTensor[mut=False, .float32, address_space=.GENERIC, ...],
    b: LayoutTensor[
        mut=False,
        .uint8,
        b_layout,
        address_space=.GENERIC,
        ...,
    ],
    c: LayoutTensor[mut=True, .float32, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
):
    comptime assert a_quant.rank == 2
    comptime assert a_scale.rank == 2
    comptime assert b.rank == 2
    comptime assert c.rank == 2

    comptime simd_width = simd_width_of[DType.float32]()
    comptime bytes_per_group_int4 = size_of[DType.float16]() + (group_size // 2)

    var N = b.dim[0]()
    var K = a_quant.dim[1]()
    var k_groups = K // group_size

    comptime grain_size = simd_width * 2

    var work_count = ceildiv(N, grain_size)
    var num_workers = min(work_count, parallelism_level(ctx))

    def task_func(
        task_id: Int,
    ) {var N, var K, var k_groups, var work_count, var num_workers, imm}:
        var block_range = partition_work(task_id, num_workers, work_count, 1)
        var task_n_start = block_range[0] * grain_size
        var task_n_count = block_range[1] * grain_size

        var b_ptr = b.ptr.unsafe_bitcast[Int8]()

        @always_inline
        def process_cols[tile_n: Int](n_idx: Int) {imm}:
            var n = task_n_start + n_idx * simd_width

            var c_float = _Accumulator[.float32, 1, tile_n, simd_width]()

            c_float.init()

            var ak_ptr = a_quant.ptr.unsafe_bitcast[Int8]()
            var ak_scale_ptr = a_scale.ptr
            var bk_ptr = b_ptr.unsafe_offset(
                n * k_groups * bytes_per_group_int4
            )

            for _ in range(0, K, group_size):
                kernel.process_group_packed[group_size](
                    ak_ptr, ak_scale_ptr, bk_ptr, c_float
                )

                ak_ptr = ak_ptr.unsafe_offset(group_size)
                ak_scale_ptr = ak_scale_ptr.unsafe_offset(1)
                bk_ptr = bk_ptr.unsafe_offset(
                    tile_n * simd_width * bytes_per_group_int4
                )

            c_float.store(c.ptr.unsafe_offset(c._offset(Index(0, n))), N)

            comptime if elementwise_lambda_fn:
                comptime func = elementwise_lambda_fn.value()

                comptime for nn in range(tile_n):
                    var val = c_float[0, nn]
                    func[.float32, simd_width](
                        Index(0, n + nn * simd_width), val
                    )

        tile[[2, 1]](0, ceildiv(task_n_count, simd_width), process_cols)

    sync_parallelize(task_func, num_workers, ctx)


def _matmul_qint4_m_any[
    kernel: _MatmulQInt4Kernel,
    group_size: Int,
    aq_type: DType,
    b_layout: Layout = Layout.row_major[2](),
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    a_quant: LayoutTensor[mut=False, aq_type, address_space=.GENERIC, ...],
    a_scale: LayoutTensor[mut=False, .float32, address_space=.GENERIC, ...],
    b: LayoutTensor[
        mut=False,
        .uint8,
        b_layout,
        address_space=.GENERIC,
        ...,
    ],
    c: LayoutTensor[mut=True, .float32, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
):
    comptime simd_width = simd_width_of[DType.float32]()
    comptime alignment = align_of[SIMD[.float32, simd_width]]()
    comptime bytes_per_group_int4 = size_of[DType.float16]() + (group_size // 2)

    var M = a_quant.dim[0]()
    var N = b.dim[0]()
    var K = a_quant.dim[1]()
    var k_groups = K // group_size

    comptime grain_size = simd_width * 2

    var work_count = ceildiv(N, grain_size)
    var num_workers = min(work_count, parallelism_level(ctx))

    def task_func(
        task_id: Int,
    ) {var M, var N, var K, var k_groups, var work_count, var num_workers, imm}:
        var block_range = partition_work(task_id, num_workers, work_count, 1)
        var task_n_start = block_range[0] * grain_size
        var task_n_count = block_range[1] * grain_size

        var b_ptr = b.ptr

        for ko in range(0, K, K_BATCH_SIZE):
            var ko_count = min(K_BATCH_SIZE, K - ko)
            var ko_group = ko // group_size

            # TODO(MOCO-4664): Capture the loop-scoped values by copy to work
            # around the wrong debug-info scope emitted for implicit
            # nested-scope captures, which breaks --debug-level=full builds.
            @always_inline
            def process_cols[
                tile_n: Int
            ](n_idx: Int) {var ko, var ko_count, var ko_group, imm}:
                var n = task_n_start + n_idx * simd_width

                comptime k_batch_groups = K_BATCH_SIZE // group_size

                var b_s8_buf = unsafe_stack_allocation[
                    K_BATCH_SIZE * tile_n * simd_width,
                    DType.int8,
                    alignment=alignment,
                ]()
                var b_scale_buf = unsafe_stack_allocation[
                    k_batch_groups * tile_n * simd_width,
                    DType.float32,
                    alignment=alignment,
                ]()

                # If the A matrix is quantized using an unsigned data type,
                # then a zero point correction is required to the block int32
                # accumulator.
                comptime needs_correction = aq_type.is_unsigned()

                var b_correction_buf = unsafe_stack_allocation[
                    k_batch_groups * tile_n * simd_width,
                    DType.int32,
                    alignment=alignment,
                ]() if needs_correction else MutPointer[
                    Int32, MutUntrackedOrigin
                ].unsafe_dangling()

                _unpack_weights[
                    group_size,
                    tile_n,
                    simd_width,
                    needs_correction=needs_correction,
                    is_i8mm=kernel.aq_tuple_type() == .int64,
                ](
                    b_s8_buf,
                    b_ptr.unsafe_offset(
                        (n * k_groups + ko_group * tile_n * simd_width)
                        * bytes_per_group_int4
                    ),
                    b_scale_buf,
                    b_correction_buf,
                    ko_count,
                )

                var ak_ptr = a_quant.ptr.unsafe_offset(ko * M)
                var ak_scale_ptr = a_scale.ptr.unsafe_offset(ko_group * M)

                @always_inline
                def process_rows[
                    tile_m: Int
                ](m: Int) {mut ak_scale_ptr, mut ak_ptr, imm}:
                    var c_ptr = c.ptr.unsafe_offset(c._offset(Index(m, n)))
                    var c_float = _Accumulator[
                        .float32, tile_m, tile_n, simd_width
                    ]()

                    if ko == 0:
                        c_float.init()
                    else:
                        c_float.load(c_ptr, N)

                    var bk_s8_ptr = b_s8_buf
                    var bk_scale_ptr = b_scale_buf
                    var bk_correction_ptr = b_correction_buf

                    for _ in range(0, ko_count, group_size):
                        kernel.process_group_unpacked[group_size](
                            ak_ptr.unsafe_bitcast[Int8](),
                            ak_scale_ptr,
                            bk_s8_ptr,
                            bk_scale_ptr,
                            bk_correction_ptr,
                            c_float,
                        )

                        ak_ptr = ak_ptr.unsafe_offset(tile_m * group_size)
                        ak_scale_ptr = ak_scale_ptr.unsafe_offset(tile_m)
                        bk_s8_ptr = bk_s8_ptr.unsafe_offset(
                            group_size * tile_n * simd_width
                        )
                        bk_scale_ptr = bk_scale_ptr.unsafe_offset(
                            tile_n * simd_width
                        )
                        bk_correction_ptr = bk_correction_ptr.unsafe_offset(
                            tile_n * simd_width
                        )

                    c_float.store(c_ptr, N)

                    # we only need to call the epilogue for the last iteration,
                    # otherwise we give intermediate results.
                    var last_k_iter = ceildiv(K, K_BATCH_SIZE) - 1
                    if ko == last_k_iter * K_BATCH_SIZE:
                        comptime if elementwise_lambda_fn:
                            comptime func = elementwise_lambda_fn.value()

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

                tile[[4, 2, 1]](0, M, process_rows)

            tile[[2, 1]](0, ceildiv(task_n_count, simd_width), process_cols)

    sync_parallelize(task_func, num_workers, ctx)


def _matmul_qint4[
    kernel: _MatmulQInt4Kernel,
    group_size: Int,
    b_layout: Layout = Layout.row_major[2](),
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    a: LayoutTensor[mut=False, .float32, address_space=.GENERIC, ...],
    b: LayoutTensor[
        mut=False,
        .uint8,
        b_layout,
        address_space=.GENERIC,
        ...,
    ],
    c: LayoutTensor[mut=True, .float32, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
):
    comptime simd_width = simd_width_of[DType.float32]()
    comptime alignment = align_of[SIMD[.float32, simd_width]]()

    var M = a.dim[0]()
    var K = a.dim[1]()
    var k_groups = K // group_size

    comptime aq_type = kernel.aq_type()

    var a_quant_base = alloc(
        AllocLayout[Scalar[aq_type]](count=M * K, alignment=alignment)
    )
    var a_scale_base = alloc(AllocLayout[Float32](count=M * k_groups))

    var a_quant_ptr: MutPointer[
        Scalar[aq_type], origin_of(a_quant_base._alloc)
    ] = a_quant_base.unsafe_ptr()
    var a_quant = LayoutTensor[aq_type, Layout.row_major[2]()](
        a_quant_ptr,
        RuntimeLayout[Layout.row_major[2]()].row_major(Index(M, K)),
    )
    var a_scale_ptr: MutPointer[
        Float32, origin_of(a_scale_base._alloc)
    ] = a_scale_base.unsafe_ptr()
    var a_scale = LayoutTensor[.float32, Layout.row_major[2]()](
        a_scale_ptr,
        RuntimeLayout[Layout.row_major[2]()].row_major(Index(M, k_groups)),
    )

    kernel.quantize_a_buffer[group_size](a, a_quant, a_scale)

    if M == 1:
        _matmul_qint4_m_1[
            kernel, group_size, elementwise_lambda_fn=elementwise_lambda_fn
        ](a_quant, a_scale, b, c, ctx)
    else:
        _matmul_qint4_m_any[
            kernel, group_size, elementwise_lambda_fn=elementwise_lambda_fn
        ](a_quant, a_scale, b, c, ctx)

    dealloc(a_quant_base^)
    dealloc(a_scale_base^)


def matmul_qint4[
    group_size: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    a_tt: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    b_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    c_tt: TileTensor[mut=True, .float32, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
):
    """Computes a matrix multiply of a float32 A matrix against block-wise
    quantized int4 B weights, producing a float32 result.

    Dispatches to an architecture-specific kernel (VNNI, AVX2, NEON i8mm,
    or NEON dotprod) at compile time.

    Parameters:
        group_size: Number of elements per quantization group.
        elementwise_lambda_fn: Optional epilogue applied to each output element.

    Args:
        a_tt: Input A tensor in float32.
        b_tt: Input B tensor holding packed uint8 int4 weights.
        c_tt: Output C tensor in float32.
        ctx: Optional device context for parallel execution.
    """
    var a = a_tt.to_layout_tensor()
    var b = b_tt.to_layout_tensor()
    var c = c_tt.to_layout_tensor()

    @__parameter
    def kernel_dispatch[kernel: _MatmulQInt4Kernel]():
        return _matmul_qint4[
            kernel,
            group_size=group_size,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](a, b, c, ctx)

    comptime if CompilationTarget.has_vnni():
        kernel_dispatch[_MatmulQInt4Kernel_x86_vnni]()
    elif CompilationTarget.has_avx2():
        kernel_dispatch[_MatmulQInt4Kernel_x86_avx]()
    elif (
        CompilationTarget.has_neon_int8_matmul()
        and not CompilationTarget.is_apple_silicon()
    ):
        kernel_dispatch[_MatmulQInt4Kernel_neon_i8mm]()
    elif CompilationTarget.has_neon_int8_dotprod():
        kernel_dispatch[_MatmulQInt4Kernel_neon_dotprod]()
    else:
        comptime assert False, "unsupported architecture"
