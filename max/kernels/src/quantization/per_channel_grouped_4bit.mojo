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
"""Provides per-channel grouped 4-bit quantization and K-quant dequantization kernels."""

from std.math import ceil, ceildiv
from std.sys.info import size_of

from layout import Layout, LayoutTensor, TileTensor
from std.memory import UnsafePointer, bitcast, unsafe_memcpy
from std.utils import IndexList, StaticTuple, product


@always_inline
def _to_StaticTuple[
    dtype: DType, size: SIMDLength
](data: SIMD[dtype, size]) -> StaticTuple[Scalar[dtype], size]:
    """Convert SIMD to StaticTuple."""

    var res = StaticTuple[Scalar[dtype], size]()

    comptime for i in range(size):
        res[i] = data[i]
    return res


@always_inline
def _to_SIMD[
    dtype: DType, size: Int
](data: StaticTuple[Scalar[dtype], size]) -> SIMD[dtype, size]:
    """Convert StaticTuple to SIMD."""
    var res = SIMD[dtype, size]()

    comptime for i in range(size):
        res[i] = data[i]
    return res


@always_inline
def calculate_symmetric_vector[
    input_dtype: DType, simd_width: SIMDLength, output_bits: Int
](data: SIMD[input_dtype, simd_width]) -> Tuple[
    SIMD[.uint8, simd_width],
    Scalar[input_dtype],
]:
    """
    Symmetrically quantizes the given SIMD vector `data` with input type
    `input_dtype` and `simd_width` elements, assuming we want
    the results to fit in an unsigned integer of size `output_bits`.

    Parameters:
        input_dtype: The dtype of the input tensor.
        simd_width: The width of the SIMD input.
        output_bits: The bits we want to fit the unsigned integral result in.

    Args:
        data: The input SIMD we want to quantize.

    Returns:
      A vector of the quantized values.
      The associated scale factor.
    """
    comptime assert (
        output_bits >= 1 and output_bits <= 8
    ), "expected a scalar type"
    comptime assert (
        input_dtype.is_floating_point()
    ), "expect accumulating over floating point only."
    var max_value = data.reduce_max()
    var min_value = data.reduce_min()
    var result_range = max(max_value, -min_value)

    # quantize as if we were signed, and convert back to unsigned later
    var positive_steps = (1 << (output_bits - 1)) - 1
    var negative_steps = 1 << (output_bits - 1)

    # Handle the case we only have one type of value in `data`, e.g.
    # data = [12., 12., 12., 12.], then the scale is 1/12.0 and the quantized
    # data is ~[1, 1, 1, 1] + `negative_steps`
    var f32_scale = (max_value / 1.0) if (
        result_range == 0
    ) else result_range / Scalar[input_dtype](positive_steps)

    # TODO: consider clipping values
    var data_rounded = round(data / f32_scale).cast[.int8]()

    # each bit pattern in `data_quantized`
    var data_quantized = (data_rounded + Int8(negative_steps)).cast[
        DType.uint8
    ]()

    return data_quantized, f32_scale


struct Q4sym[
    group_size: Int,
    float_dtype: DType = .float32,
](Defaultable):
    """
    Q4sym: compresses values of type `float_dtype` to 4bit unsigned integers
    which have been dynamically symmetrically quantized with the given scale
    factor.

    `group_size` determines the number of elements which share quantization
    parameters.

    We store things in a strided fashion:
    Example:

        Assume `group_size = 8` and we want to process uint4 numbers:
        A, B, C, D, E, F, G, H which have associated bits aaaa, bbbb, cccc, ....

           eeeeaaaa|ffffbbbb|ggggcccc|hhhhdddd

    To uncompress to floating point, take the decoded uint4 value, subtract
    the implicit zero-point of 2^4=8, and multiply by the scale factor.

    Parameters:
        group_size: The number of encoded numbers stored in this struct.
        float_dtype: The floating point dtype this struct works with.
    """

    var scale: StaticTuple[UInt8, 2]
    """The FP16 scale of the group, stored as individual bytes."""

    var bits: StaticTuple[UInt8, SIMDLength(Self.group_size) // 2]
    """The bits of the encoded uint4 numbers."""

    @staticmethod
    @always_inline
    def _check_constraints():
        # TODO
        comptime assert (
            Self.group_size.is_power_of_two()
        ), "`group_size` must be a power of 2."
        comptime assert (
            Self.group_size == 8
            or Self.group_size == 16
            or Self.group_size == 32
        ), "Only support some `group_sizes`"
        comptime assert (
            Self.float_dtype.is_floating_point()
        ), "Must be floating point type"

    @always_inline
    def __init__(out self):
        """Construct a default initialized Q4sym."""
        self.scale = StaticTuple[UInt8, 2]()
        self.bits = StaticTuple[UInt8, SIMDLength(Self.group_size) // 2]()
        self._check_constraints()

    @always_inline
    def __init__(out self, data: SIMD[Self.float_dtype, Self.group_size]):
        """
        Construct an encoded Q4sym from data.

        Args:
            data: The floating point data to encode and store.
        """
        var quantization_tuple = calculate_symmetric_vector[
            Self.float_dtype, Self.group_size, 4
        ](data)
        var qdata = quantization_tuple[0]
        var f_scale = quantization_tuple[1]

        # TODO: add warning if we overflow/underflow
        var f16_scale = f_scale.cast[.float16]()
        self.scale = _to_StaticTuple(bitcast[.uint8, 2](f16_scale))
        self.bits = _to_StaticTuple(self._encode_bits(qdata))
        self._check_constraints()

    @staticmethod
    @always_inline
    def _encode_bits(
        qdata: SIMD[.uint8, Self.group_size]
    ) -> SIMD[.uint8, SIMDLength(Self.group_size) // 2]:
        var lo_hi = qdata.split()
        return lo_hi[0] | (lo_hi[1] << 4)

    @always_inline
    def _decode_bits(mut self) -> SIMD[.uint8, Self.group_size]:
        # Extract the lower 4 bits of all bits in the `l_bits` format
        var bits_simd = _to_SIMD[.uint8, SIMDLength(Self.group_size) // 2](
            self.bits
        )
        var bits_upper = (bits_simd & 0xF0) >> 4
        var bits_lower = bits_simd & 0x0F
        return rebind[SIMD[.uint8, Self.group_size]](
            bits_lower.join(bits_upper)
        )

    @always_inline
    def decode_scale(mut self) -> Float16:
        """
        Obtain the scale factor.

        Returns:
          The decoded scale factor.
        """
        # We avoid bit-casting directly, as the code-generation might hit a
        # path which attempts to bitcast 2xui8 --> f16 which is not typically supported
        var scale_bytes: SIMD[.uint8, 2] = _to_SIMD[.uint8, 2](self.scale)

        # NOTE: this may break on different endian systems...
        var upcast_bytes: SIMD[.uint16, 2] = scale_bytes.cast[.uint16]()
        upcast_bytes[1] = upcast_bytes[1] << 8
        var final_result: UInt16 = upcast_bytes.reduce_add()
        var scale_decoded = bitcast[.float16, 1](final_result)
        return scale_decoded

    @always_inline
    def decode_unsigned(mut self) -> SIMD[.uint8, Self.group_size]:
        """
        Decode the stored uint4 numbers to uint8.

        Returns:
          The decoded stored numbers as uint8 numbers. These have an implicit
          zero-point of 8.
        """
        # Obtain the unsigned quantized values, these have a zp of 8
        return self._decode_bits()

    @always_inline
    def decode_signed(mut self) -> SIMD[.int8, Self.group_size]:
        """
        Decode the stored uint4 numbers to requantized int4 numbers.

        This is done by simply subtracting an implicit zp of 8 from the
        unsigned decoding.

        Returns:
          The decoded stored numbers as int8 numbers. These have a zero-point of
          0.
        """
        var decoded_result = self.decode_unsigned()
        return decoded_result.cast[.int8]() - 8

    @always_inline
    def decode_fully(mut self) -> SIMD[Self.float_dtype, Self.group_size]:
        """
        Decode the stored numbers into floating point representation.

        Returns:
          The decoded numbers.
        """
        # Obtain the fully dequantized values
        var signed_result = self.decode_signed()
        var scale_decoded = self.decode_scale().cast[Self.float_dtype]()
        var answer = (
            scale_decoded.cast[Self.float_dtype]()
            * signed_result.cast[Self.float_dtype]()
        )
        return answer

    # TODO: support other axis of quantization, right now assume inner-most dim.
    # TODO: support axis which not divisible by group_size
    @staticmethod
    def quantize_and_write_to_tensor[
        input_rank: Int
    ](
        input_tt: TileTensor[
            mut=False, Self.float_dtype, address_space=.GENERIC, ...
        ],
        output_tt: TileTensor[mut=True, .uint8, address_space=.GENERIC, ...],
        input_shape: IndexList[input_rank],
    ):
        """
        Encodes the floating point numbers in `input_tt` along the
        inner-most dimension and writes the result to output_tt.

        Parameters:
            input_rank: The rank of the input tensor.

        Args:
            input_tt: The input tensor we are encoding.
            output_tt: The output tensor containing the encoded input.
                The shape of the output should be the same as the input
                except along the inner dimension where if the original inner
                dimension was `d`, the corresponding output dimension should be:
                    ceil(`d` / group_size) * size_of(self).
            input_shape: The shape of the input tensor.
        """
        var input_tensor = input_tt.to_layout_tensor()
        var output_tensor = output_tt.to_layout_tensor()
        comptime assert (
            input_rank == input_tensor.rank
        ), "input_rank must match tensor rank"
        comptime assert (
            input_tensor.rank == output_tensor.rank
        ), "input tensor and output tensor must have the same rank"
        # TODO: check contiguous inputs and outputs

        # Read and quantize `input_tensor`` to blocked format, dump the raw
        # struct/block into `output_tensor`
        assert (
            input_shape[input_tensor.rank - 1] % Self.group_size == 0
        ), "Only support fully divisible dimensions right now."

        var blob_output_ptr = output_tensor.ptr
        var base_block_ptr = blob_output_ptr.bitcast[
            Q4sym[Self.group_size, Self.float_dtype]
        ]()

        # as we support only inner-most dim, treat like rank-2 tensor
        var outer_stride = product(input_tensor.runtime_layout.stride.value)
        var input_inner_stride = input_tensor.runtime_layout.shape.value[
            input_tensor.rank - 1
        ]
        var output_inner_stride = ceildiv(
            Int(input_tensor.runtime_layout.shape[input_tensor.rank - 1]),
            Self.group_size,
        )

        # TODO: vectorize parallelize, blah blah blah
        for i in range(outer_stride):
            for j in range(output_inner_stride):
                var flat_index_input = (
                    input_inner_stride * i + j * Self.group_size
                )
                var loaded_group = input_tensor.ptr.load[width=Self.group_size](
                    flat_index_input
                )

                var flat_index_output = output_inner_stride * i + j
                var output_ptr = base_block_ptr + flat_index_output

                # TODO: use the memory more directly instead of unsafe_memcpy
                var encoded_data = Q4sym[Self.group_size, Self.float_dtype](
                    loaded_group
                )
                var src_ptr = UnsafePointer(to=encoded_data).address_space_cast[
                    output_ptr.address_space
                ]()
                unsafe_memcpy(
                    dest=output_ptr,
                    src=src_ptr,
                    count=1,
                )
                _ = encoded_data^

    @staticmethod
    def dequantize_and_write_to_tensor[
        output_rank: Int
    ](
        input_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
        output_tt: TileTensor[
            mut=True, Self.float_dtype, address_space=.GENERIC, ...
        ],
        output_shape: IndexList[output_rank],
    ):
        """
        Dequantizes the block-quantized values in `input_tt` along the
        inner-most dimension and writes the decoded result to `output_tt`.

        Parameters:
            output_rank: The rank of the output tensor.

        Args:
            input_tt: The input tensor we are decoding.
            output_tt: The output tensor containing the decoded input.
            output_shape: The shape of the output tensor.
        """
        var input_tensor = input_tt.to_layout_tensor()
        var output_tensor = output_tt.to_layout_tensor()
        comptime assert (
            output_rank == output_tensor.rank
        ), "output_rank must match tensor rank"
        comptime assert (
            input_tensor.rank == output_tensor.rank
        ), "input tensor and output tensor must have the same rank"
        # Read and dequantize `input_tensor` which are the bytes of the raw
        # blocked format. Write the corresponding results to `output_tensor`
        assert (
            output_tensor.runtime_layout.shape.value[output_tensor.rank - 1]
            % Self.group_size
            == 0
        ), "Only support fully divisible dimensions right now."

        # TODO: check contiguous inputs and outputs

        var uint8_input_ptr = input_tensor.ptr
        var base_block_ptr = uint8_input_ptr.bitcast[
            Q4sym[Self.group_size, Self.float_dtype]
        ]()

        # as we support only inner-most dim, treat like rank-2 tensor
        var output_inner_dim = output_tensor.runtime_layout.shape.value[
            output_tensor.rank - 1
        ]
        var outer_dim = product(output_tensor.runtime_layout.stride.value)

        # Note: this is calculated assuming a pointer of Q4Sym's
        var input_inner_dim = ceildiv(output_inner_dim, Self.group_size)

        # TODO: vectorize parallelize, blah blah blah
        for i in range(outer_dim):
            for j in range(input_inner_dim):
                var flat_index_input = input_inner_dim * i + j
                var encoded = Q4sym[Self.group_size, Self.float_dtype]()
                unsafe_memcpy(
                    dest=UnsafePointer(to=encoded),
                    src=base_block_ptr + flat_index_input,
                    count=1,
                )

                var flat_index_output = (
                    output_inner_dim * i + j * Self.group_size
                )
                output_tensor.ptr.store(
                    flat_index_output,
                    encoded.decode_fully(),
                )


######
# Q4_K
######


struct block_QK_K:
    """Defines the element count per block shared by K-quant formats."""

    comptime quantized_k = 256
    """The number of elements quantized per block."""


struct block_Q4_K:
    """Represents a single Q4_K quantization block with grouped 4-bit values."""

    comptime group_size = 32
    """The number of elements per quantization group."""
    comptime group_count = block_QK_K.quantized_k // Self.group_size
    """The number of groups within a block."""

    var base_scale: Float16
    """The super-block scale factor."""
    var base_min: Float16
    """The super-block minimum value."""
    var q_scales_and_mins: Array[UInt8, (2 * block_Q4_K.group_count * 6) // 8]
    """The 6-bit quantized per-group scales and mins, packed into bytes."""
    # 256 total elements / 8 groups => 32 elements per group.
    var q_bits: Array[UInt8, block_QK_K.quantized_k // 2]
    """The 4-bit quantized values, two per byte."""


def scale_min_k4(
    src_ptr: UnsafePointer[block_Q4_K, address_space=.GENERIC, ...],
    g: Int,
) -> Tuple[Float32, Float32]:
    """Extracts the 6-bit quantized scale and min for group `g` from a Q4_K block.

    Args:
        src_ptr: Pointer to the source Q4_K block.
        g: The group index to extract the scale and min for.

    Returns:
        The quantized scale and min as float32 values.
    """
    if g < 4:
        var q_scale = src_ptr[].q_scales_and_mins[g] & 63
        var q_min = src_ptr[].q_scales_and_mins[g + 4] & 63

        return q_scale.cast[.float32](), q_min.cast[.float32]()
    else:
        var q_scale_lo = src_ptr[].q_scales_and_mins[g + 4] & 15
        var q_min_lo = src_ptr[].q_scales_and_mins[g + 4] >> 4
        var q_scale_hi = src_ptr[].q_scales_and_mins[g - 4] >> 6
        var q_min_hi = src_ptr[].q_scales_and_mins[g - 0] >> 6
        var q_scale = (q_scale_hi << 4) | q_scale_lo
        var q_min = (q_min_hi << 4) | q_min_lo

        return q_scale.cast[.float32](), q_min.cast[.float32]()


def q4_k_dequantize_impl(
    input_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    output_tt: TileTensor[mut=True, .float32, address_space=.GENERIC, ...],
):
    """Dequantizes a Q4_K encoded tensor into a float32 output tensor.

    Args:
        input_tt: The input tensor containing Q4_K encoded data.
        output_tt: The output tensor to write dequantized float32 values to.
    """
    var input_tensor = input_tt.to_layout_tensor()
    var output_tensor = output_tt.to_layout_tensor()
    comptime group_nelems = block_Q4_K.group_size
    # 2 elements per byte.
    comptime group_nbytes = group_nelems // 2
    comptime block_nelems = block_QK_K.quantized_k
    comptime block_nbytes = size_of[block_Q4_K]()

    var num_blocks = input_tensor.size() // block_nbytes
    var input_q4_k_ptr = input_tensor.ptr.bitcast[block_Q4_K]()
    var output_ptr = output_tensor.ptr.bitcast[Float32]()
    for block_idx in range(num_blocks):
        var src_ptr = input_q4_k_ptr + block_idx
        var dst_ptr = output_ptr + (block_idx * block_nelems)

        # d
        var b_scale = src_ptr[].base_scale.cast[.float32]()
        # min
        var b_min = src_ptr[].base_min.cast[.float32]()

        # Process 2 groups at a time to load 6-bit scales/mins.
        comptime for group_idx in range(0, block_Q4_K.group_count, 2):
            var q_scale: Float32
            var q_min: Float32

            # group_idx: low bits
            q_scale, q_min = scale_min_k4(src_ptr, group_idx)
            var d1 = b_scale * q_scale
            var m1 = b_min * q_min

            # group_idx + 1: high bits
            q_scale, q_min = scale_min_k4(src_ptr, group_idx + 1)
            var d2 = b_scale * q_scale
            var m2 = b_min * q_min

            var q_bits_idx = group_idx * group_nbytes
            var dst_idx = group_idx * group_nelems

            # Dequantize 1st group bits.
            comptime for elem_idx in range(group_nelems):
                dst_ptr[dst_idx + elem_idx] = (
                    d1
                    * (src_ptr[].q_bits[q_bits_idx + elem_idx] & 0xF).cast[
                        DType.float32
                    ]()
                    - m1
                )

            # Dequantize 2nd group bits.
            comptime for elem_idx in range(group_nelems):
                dst_ptr[dst_idx + group_nelems + elem_idx] = (
                    d2
                    * (src_ptr[].q_bits[q_bits_idx + elem_idx] >> 4).cast[
                        DType.float32
                    ]()
                    - m2
                )


######
# Q6_K
######


struct block_Q6_K:
    """Represents a single Q6_K quantization block with grouped 6-bit values."""

    comptime group_size = 16
    """The number of elements per quantization group."""
    comptime group_count = block_QK_K.quantized_k // Self.group_size
    """The number of groups within a block."""

    # Low 4 bits.
    var q_bits_lo: Array[UInt8, block_QK_K.quantized_k // 2]
    """The low 4 bits of the 6-bit quantized values, two per byte."""
    # High 2 bits.
    var q_bits_hi: Array[UInt8, block_QK_K.quantized_k // 4]
    """The high 2 bits of the 6-bit quantized values, four per byte."""
    # int8 scales.
    var q_scales: Array[Int8, block_Q6_K.group_size]
    """The int8 per-group scale factors."""
    # Superblock scale.
    var base_scale: Float16
    """The super-block scale factor."""


def q6_k_dequantize_impl[
    output_rank: Int
](
    input_tt: TileTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    output_tt: TileTensor[mut=True, .float32, address_space=.GENERIC, ...],
    output_shape: IndexList[output_rank],
):
    """Dequantizes a Q6_K encoded tensor into a float32 output tensor.

    Parameters:
        output_rank: The rank of the output tensor.

    Args:
        input_tt: The input tensor containing Q6_K encoded data.
        output_tt: The output tensor to write dequantized float32 values to.
        output_shape: The shape of the output tensor.
    """
    var input_tensor = input_tt.to_layout_tensor()
    var output_tensor = output_tt.to_layout_tensor()
    comptime group_nelems = block_Q6_K.group_size
    comptime block_nelems = block_QK_K.quantized_k
    comptime block_nbytes = size_of[block_Q6_K]()

    var num_blocks = (output_shape[0] * output_shape[1]) // block_nelems
    var input_q6_k_ptr = input_tensor.ptr.bitcast[block_Q6_K]()
    var dst_ptr = output_tensor.ptr.bitcast[Float32]()
    var dst_idx = 0
    for block_idx in range(num_blocks):
        var src_ptr = input_q6_k_ptr + block_idx

        var d = src_ptr[].base_scale.cast[.float32]()

        var ql: UnsafePointer[
            UInt8, origin_of(src_ptr[].q_bits_lo)
        ] = src_ptr[].q_bits_lo.unsafe_ptr()
        var qh: UnsafePointer[
            UInt8, origin_of(src_ptr[].q_bits_hi)
        ] = src_ptr[].q_bits_hi.unsafe_ptr()
        var sc: UnsafePointer[
            Int8, origin_of(src_ptr[].q_scales)
        ] = src_ptr[].q_scales.unsafe_ptr()

        # Process 8 groups at a time.
        comptime for _ in range(0, block_Q6_K.group_count, 8):
            comptime for l in range(32):
                var sc_idx = l // 16
                var q1 = ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)).cast[
                    DType.int8
                ]() - 32
                var q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)).cast[
                    DType.int8
                ]() - 32
                var q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)).cast[
                    DType.int8
                ]() - 32
                var q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)).cast[
                    DType.int8
                ]() - 32

                dst_ptr[l + 0] = (
                    d * sc[sc_idx + 0].cast[.float32]() * q1.cast[.float32]()
                )
                dst_ptr[l + 32] = (
                    d * sc[sc_idx + 2].cast[.float32]() * q2.cast[.float32]()
                )
                dst_ptr[l + 64] = (
                    d * sc[sc_idx + 4].cast[.float32]() * q3.cast[.float32]()
                )
                dst_ptr[l + 96] = (
                    d * sc[sc_idx + 6].cast[.float32]() * q4.cast[.float32]()
                )

            dst_ptr += 128
            dst_idx += 128
            ql += 64
            qh += 32
            sc += 8
