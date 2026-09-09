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

from std.math import isclose
from std.random import rand
from std.sys.info import simd_width_of

from std.algorithm.functional import vectorize
from layout import Coord, TileTensor, row_major
from nn.conv.conv_transpose import (
    ConvTransposedPacked,
    conv_transpose_naive,
    conv_transpose_shape,
    pack_filter,
    pack_filter_shape,
)
from nn.conv.conv_utils import (
    ConvInfoStatic,
    ConvShape,
    append_shape,
    extend_shape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
)

from std.testing import assert_equal, TestSuite

from std.utils.index import Index, IndexList

comptime simd_size: Int = simd_width_of[DType.float32]()
comptime dtype = DType.float32


@always_inline
def extend_shape_5d[
    rank: Int
](in_shape: IndexList[rank], first: Int, last: Int) -> IndexList[5]:
    var out_shape = IndexList[5](1)
    out_shape[0] = first
    out_shape[4] = last

    comptime if rank == 1:
        out_shape[3] = in_shape[0]
    elif rank == 2:
        out_shape[2] = in_shape[0]
        out_shape[3] = in_shape[1]
    elif rank == 3:
        out_shape[1] = in_shape[0]
        out_shape[2] = in_shape[1]
        out_shape[3] = in_shape[2]

    return out_shape


@always_inline
def extend_shape_3d[rank: Int](in_shape: IndexList[rank]) -> IndexList[3]:
    var out_shape = IndexList[3](1)

    comptime for i in range(rank):
        out_shape[2 - i] = in_shape[rank - i - 1]

    return out_shape


@always_inline
def append_shape_5d[
    rank: Int
](in_shape: IndexList[rank], last2nd: Int, last: Int) -> IndexList[5]:
    var out_shape = IndexList[5](1)
    out_shape[3] = last2nd
    out_shape[4] = last

    comptime if rank == 1:
        out_shape[2] = in_shape[0]
    elif rank == 2:
        out_shape[1] = in_shape[0]
        out_shape[2] = in_shape[1]
    elif rank == 3:
        out_shape[0] = in_shape[0]
        out_shape[1] = in_shape[1]
        out_shape[2] = in_shape[2]

    return out_shape


def test_conv_transposed[
    dtype: DType, rank: Int
](
    N: Int,
    input_dims: IndexList[rank],
    C: Int,
    filter_dims: IndexList[rank],
    F: Int,
    stride: IndexList[rank],
    dilation: IndexList[rank],
    pad: IndexList[2 * rank],
    num_groups: Int,
) raises:
    print("test_conv_transposed")

    var output_dims = IndexList[rank](1)

    comptime for i in range(rank):
        output_dims[i] = (
            (input_dims[i] - 1) * stride[i]
            - pad[2 * i]
            - pad[2 * i + 1]
            + dilation[i] * (filter_dims[i] - 1)
            + 1
        )

    var pad_d = IndexList[2](0)
    var pad_h = IndexList[2](0)
    var pad_w = IndexList[2](0)

    comptime if rank == 1:
        pad_w = Index(pad[0], pad[1])
    elif rank == 2:
        pad_h = Index(pad[0], pad[1])
        pad_w = Index(pad[2], pad[3])
    elif rank == 3:
        pad_d = Index(pad[0], pad[1])
        pad_h = Index(pad[2], pad[3])
        pad_w = Index(pad[4], pad[5])

    var conv_shape = ConvShape[rank](
        n=N,
        input_dims=Coord(input_dims),
        output_dims=Coord(output_dims),
        filter_dims=Coord(filter_dims),
        c=C,
        f=F,
        stride=Coord(stride),
        dilation=Coord(dilation),
        pad_d=Coord(pad_d),
        pad_h=Coord(pad_h),
        pad_w=Coord(pad_w),
        num_groups=num_groups,
    )

    var C_per_group = C // num_groups

    var input_size = N * conv_shape.input_image_flat_size() * C
    var input_ptr = List(length=input_size, fill=Scalar[dtype](0))
    rand(input_ptr)

    var filter_size = conv_shape.filter_window_flat_size() * C_per_group * F
    var filter_ptr = List(length=filter_size, fill=Scalar[dtype](0))
    rand(filter_ptr)

    var output_size = N * conv_shape.output_image_flat_size() * F
    var output_ptr = List(length=output_size, fill=Scalar[dtype](0))
    var output_ref_ptr = List(length=output_size, fill=Scalar[dtype](0))

    # Find the tile size used in packing.
    comptime micro_kernel_height = get_direct_conv_micro_kernel_height()
    comptime micro_kernel_width = get_direct_conv_micro_kernel_width()

    # Rounded C and F size for pre-packed filter.
    # alias micro_kernel_f_size = get_direct_conv_micro_kernel_width() * simd_size
    # var rounded_F = ceildiv(F, micro_kernel_f_size) * micro_kernel_f_size

    # Input buffer.
    var input_shape = extend_shape(input_dims, N, C)
    var input = TileTensor(input_ptr, row_major(Coord(input_shape)))
    var input_ref = TileTensor(
        input_ptr,
        row_major(Coord(extend_shape_5d(input_dims, N, C))),
    )

    # Filter buffer.
    var filter_shape = append_shape(filter_dims, F, C_per_group)
    var filter = TileTensor(filter_ptr, row_major(Coord(filter_shape)))
    var filter_ref = TileTensor(
        filter_ptr,
        row_major(Coord(append_shape_5d(filter_dims, F, C_per_group))),
    )

    var packed_filter_shape = pack_filter_shape(filter, num_groups)
    var packed_filter_ptr = List(
        length=packed_filter_shape.flattened_length(), fill=Scalar[dtype](0)
    )
    var packed_filter = TileTensor(
        packed_filter_ptr,
        row_major(Coord(rebind[IndexList[rank + 3]](packed_filter_shape))),
    )

    var output_shape = extend_shape(output_dims, N, F)
    var output = TileTensor(output_ptr, row_major(Coord(output_shape)))
    var output_ref = TileTensor(
        output_ref_ptr,
        row_major(Coord(extend_shape_5d(output_dims, N, F))),
    )

    # Bias for epilogue. Kept as raw alloc because it's captured by
    # `@__copy_capture` closures, which require ImplicitlyCopyable.
    var bias_ptr = alloc[Scalar[dtype]](F)
    rand(bias_ptr, F)

    pack_filter(filter, packed_filter, num_groups)

    # Reference.
    conv_transpose_naive[dtype](
        output_ref,
        input_ref,
        filter_ref,
        extend_shape_3d[rank](stride),
        extend_shape_3d[rank](dilation),
        pad_d,
        pad_h,
        pad_w,
    )

    # Add bias and activatiion separately.
    var output_image_size = output_dims.flattened_length()
    for n in range(N):
        for i in range(output_image_size):
            var output_ref_ptr = output_ref._storage + F * (
                i + output_image_size * n
            )

            @always_inline
            def body0[
                width: Int
            ](offset: Int) {var output_ref_ptr, var bias_ptr}:
                output_ref_ptr.store(
                    offset,
                    10.0
                    * (
                        output_ref_ptr.load[width=width](offset)
                        + bias_ptr.load[width=width](offset)
                    ),
                )

            vectorize[simd_size](F, body0)

    # Test.
    comptime conv_attr = ConvInfoStatic[rank]()

    # Test epilogue
    @always_inline
    @__copy_capture(output, bias_ptr)
    @__parameter
    def epilogue[_rank: Int](coords: IndexList[_rank], f_size: Int):
        @always_inline
        def body1[width: Int](idx: Int) {var}:
            var curr_coords = rebind[IndexList[rank + 2]](coords)
            curr_coords[rank + 1] += idx

            var output_idx = output.layout(Coord(curr_coords))

            var vec = output.raw_load[width=width](output_idx)

            output.raw_store(
                output_idx,
                10.0
                * (vec + bias_ptr.load[width=width](curr_coords[rank + 1])),
            )

        vectorize[simd_size](f_size, body1)

    ConvTransposedPacked[
        input.origin,
        packed_filter.origin,
        output.origin,
        dtype,
        dtype,
        dtype,
        conv_attr,
        epilogue,
    ].run(
        output,
        input,
        packed_filter,
        rebind[ConvShape[rank]](conv_shape),
    )

    # Check results, return on the first failed comparison.
    for i in range(output_size):
        if not all(
            isclose(
                output_ref.raw_load[width=1](i),
                output.raw_load[width=1](i),
                atol=1e-4,  # absolute error tolerance
                rtol=1e-4,  # relative error tolerance
            )
        ):
            print("Input shape: ", input_shape)
            print("filter shape: ", filter_shape)
            print("num groups", num_groups)
            print("flat output index:", i)
            print("Golden value: ", output_ref.raw_load[width=1](i))
            print("Actual value: ", output.raw_load[width=1](i))
            bias_ptr.free()
            return

    # CHECK: Succeed
    print("Succeed")
    bias_ptr.free()
    _ = output_ref_ptr^
    _ = output_ptr^
    _ = packed_filter_ptr^
    _ = filter_ptr^
    _ = input_ptr^


def test_conv_transpose_shape_basic() raises:
    """Test conv_transpose_shape function with basic cases."""
    # Test 4D: Basic 2D conv transpose (N=1, H=3, W=3, C=1) x (R=3, S=3, F=2, C=1)
    # With stride=1, dilation=1, no padding
    # Expected output: (1, 5, 5, 2)
    var input_ptr = List(length=9, fill=Float32(0))
    var kernel_ptr = List(length=18, fill=Float32(0))
    var strides_ptr = List(length=2, fill=Int32(1))
    var dilations_ptr = List(length=2, fill=Int32(1))
    var pads_ptr = List(length=4, fill=Int32(0))
    var output_pads_ptr = List(length=2, fill=Int32(0))

    var input = TileTensor(input_ptr, row_major(Coord(Index(1, 3, 3, 1))))
    var kernel = TileTensor(kernel_ptr, row_major(Coord(Index(3, 3, 2, 1))))
    var strides = TileTensor(strides_ptr, row_major(Coord(Index(2))))
    var dilations = TileTensor(dilations_ptr, row_major(Coord(Index(2))))
    var pads = TileTensor(pads_ptr, row_major(Coord(Index(4))))
    var output_pads = TileTensor(output_pads_ptr, row_major(Coord(Index(2))))

    var shape = conv_transpose_shape[.float32, .int32, .int32, .int32, .int32](
        input, kernel, strides, dilations, pads, output_pads
    )

    assert_equal(shape[0], 1)
    assert_equal(shape[1], 5)
    assert_equal(shape[2], 5)
    assert_equal(shape[3], 2)
    _ = output_pads_ptr^
    _ = pads_ptr^
    _ = dilations_ptr^
    _ = strides_ptr^
    _ = kernel_ptr^
    _ = input_ptr^


def test_2d_stride_3_2_pad_1_1_2_2() raises:
    test_conv_transposed[.float32, 2](
        1,  # N
        Index(3, 3),
        1,  # C
        Index(3, 3),
        2,  # F
        Index(3, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1, 2, 2),  # pad h, w
        1,  # num_groups
    )


def test_2d_basic_no_pad() raises:
    test_conv_transposed[.float32, 2](
        1,  # N
        Index(3, 3),
        1,  # C
        Index(3, 3),
        2,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0, 0, 0),  # pad h, w
        1,  # num_groups
    )


def test_2d_dilation_2_2() raises:
    test_conv_transposed[.float32, 2](
        1,  # N
        Index(3, 3),
        1,  # C
        Index(3, 3),
        1,  # F
        Index(1, 1),  # stride
        Index(2, 2),  # dilation
        Index(0, 0, 0, 0),  # pad_h, pad_w
        1,  # num_groups
    )


def test_2d_stride_3_2_kernel_2_2() raises:
    test_conv_transposed[.float32, 2](
        1,  # N
        Index(3, 3),
        1,  # C
        Index(2, 2),
        2,  # F
        Index(3, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0, 0, 0),  # pad_h, pad_w
        1,  # num_groups
    )


def test_3d_stride_1_3_2() raises:
    test_conv_transposed[.float32, 3](
        1,  # N
        Index(2, 3, 3),
        1,  # C
        Index(2, 2, 2),
        2,  # F
        Index(1, 3, 2),  # stride
        Index(1, 1, 1),  # dilation
        IndexList[6](0),  # pad
        1,  # num_groups
    )


def test_3d_stride_2_1_2_dilation_1_1_2() raises:
    test_conv_transposed[.float32, 3](
        1,  # N
        Index(3, 4, 7),
        1,  # C
        Index(3, 2, 2),
        2,  # F
        Index(2, 1, 2),  # stride
        Index(1, 1, 2),  # dilation
        IndexList[6](0),  # pad
        1,  # num_groups
    )


def test_3d_with_padding() raises:
    test_conv_transposed[.float32, 3](
        1,  # N
        Index(4, 3, 3),
        1,  # C
        Index(1, 4, 2),
        2,  # F
        Index(1, 3, 2),  # stride
        Index(1, 1, 1),  # dilation
        IndexList[6](1, 0, 2, 1, 0, 1),  # pad
        1,  # num_groups
    )


def test_3d_complex_padding_dilation() raises:
    test_conv_transposed[.float32, 3](
        1,  # N
        Index(4, 5, 7),
        1,  # C
        Index(3, 2, 1),
        2,  # F
        Index(1, 3, 2),  # stride
        Index(2, 3, 1),  # dilation
        IndexList[6](2, 2, 1, 1, 1, 1),  # pad
        1,  # num_groups
    )


def test_3d_multi_channel() raises:
    test_conv_transposed[.float32, 3](
        1,  # N
        Index(5, 5, 5),
        4,  # C
        Index(3, 3, 3),
        8,  # F
        Index(1, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        IndexList[6](0, 0, 0, 0, 0, 0),  # pad
        1,  # num_groups
    )


def main() raises:
    var suite = TestSuite()

    # Test conv_transpose_shape function
    suite.test[test_conv_transpose_shape_basic]()

    # Test full conv transposed operations
    suite.test[test_2d_stride_3_2_pad_1_1_2_2]()
    suite.test[test_2d_basic_no_pad]()
    suite.test[test_2d_dilation_2_2]()
    suite.test[test_2d_stride_3_2_kernel_2_2]()
    suite.test[test_3d_stride_1_3_2]()
    suite.test[test_3d_stride_2_1_2_dilation_1_1_2]()
    suite.test[test_3d_with_padding]()
    suite.test[test_3d_complex_padding_dilation]()
    suite.test[test_3d_multi_channel]()

    suite^.run()

    # Large shapes commented out to save CI cost.

    # # StarGan shape
    # test_conv_transposed[.float32, 2](
    #     16,  # N
    #     Index(32, 32),
    #     256,  # C
    #     Index(4, 4),
    #     128,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1, 1, 1),  # pad_h, w
    #     1,  # num_groups
    # )

    # test_conv_transposed[.float32, 2](
    #     16,  # N
    #     Index(64, 64),
    #     128,  # C
    #     Index(4, 4),
    #     64,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1, 1, 1),  # pad_h, pad_w
    #     1,  # num_groups
    # )

    # # 3d Unet shapes
    # test_conv_transposed[.float32, 3](
    #     1,  # N
    #     Index(4, 4, 4),
    #     320,  # C
    #     Index(2, 2, 2),
    #     320,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     IndexList[6](0),  # pad
    #     1,  # num_groups
    # )

    # test_conv_transposed[.float32, 3](
    #     1,  # N
    #     Index(8, 8, 8),
    #     320,  # C
    #     Index(2, 2, 2),
    #     256,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     IndexList[6](0),  # pad
    #     1,  # num_groups
    # )

    # test_conv_transposed[.float32, 3](
    #     1,  # N
    #     Index(16, 16, 16),
    #     256,  # C
    #     Index(2, 2, 2),
    #     128,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     IndexList[6](0),  # pad
    #     1,  # num_groups
    # )

    # test_conv_transposed[.float32, 3](
    #     1,  # N
    #     Index(32, 32, 32),
    #     128,  # C
    #     Index(2, 2, 2),
    #     64,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     IndexList[6](0),  # pad
    #     1,  # num_groups
    # )

    # test_conv_transposed[.float32, 3](
    #     1,  # N
    #     Index(64, 64, 64),
    #     64,  # C
    #     Index(2, 2, 2),
    #     32,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     IndexList[6](0),  # pad
    #     1,  # num_groups
    # )
