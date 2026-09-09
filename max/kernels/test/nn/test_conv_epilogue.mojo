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
from layout import Coord, Layout, LayoutTensor, RuntimeLayout
from layout import lt_to_tt
from nn.conv.conv import (
    ConvDirectNHWC,
    ConvInfoStatic,
    pack_conv_filter_shape,
    pack_filter,
)
from nn.conv.conv_utils import (
    ConvShape,
    append_shape,
    extend_shape,
)

from std.utils.index import Index, IndexList

comptime simd_size: Int = simd_width_of[DType.float32]()
comptime dtype = DType.float32


# CHECK-LABEL: test_conv_epilogue
def test[
    rank: Int, dtype: DType, filter_packed: Bool
](
    N: Int,
    input_dims: IndexList[rank],
    C: Int,
    filter_dims: IndexList[rank],
    F: Int,
    stride: IndexList[rank],
    dilation: IndexList[rank],
    pad: IndexList[2 * rank],  # pad in d, h, w
    num_groups: Int,
) raises:
    print("== test_conv_epilogue")

    var output_dims = IndexList[rank](1)

    comptime for i in range(rank):
        output_dims[i] = (
            input_dims[i]
            + pad[2 * i]
            + pad[2 * i + 1]
            - dilation[i] * (filter_dims[i] - 1)
            - 1
        ) // stride[i] + 1

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

    # Kept as raw alloc because it's captured by closure bodies, which
    # require ImplicitlyCopyable.
    var bias_ptr = alloc[Scalar[dtype]](F)
    rand(bias_ptr, F)

    # Find the tile size used in packing.
    # alias micro_kernel_height = get_direct_conv_micro_kernel_height()
    # alias micro_kernel_width = get_direct_conv_micro_kernel_width()

    # Rounded C and F size for pre-packed filter.
    # var micro_kernel_f_size = micro_kernel_width * simd_size
    # var rounded_F = ceildiv(F, micro_kernel_f_size) * micro_kernel_f_size

    # Input buffer.
    comptime layout_p2 = Layout.row_major[rank + 2]()
    comptime layout_p3 = Layout.row_major[rank + 3]()
    var input_shape = extend_shape(input_dims, N, C)
    var input = LayoutTensor[dtype, layout_p2](
        input_ptr, RuntimeLayout[layout_p2].row_major(input_shape)
    )

    # Filter buffer.
    var filter_shape = append_shape(filter_dims, C_per_group, F)
    var filter = LayoutTensor[dtype, layout_p2](
        filter_ptr, RuntimeLayout[layout_p2].row_major(filter_shape)
    )

    var packed_filter_shape = pack_conv_filter_shape(
        lt_to_tt(filter), num_groups
    )
    var packed_filter_ptr = List(
        length=packed_filter_shape.flattened_length(), fill=Scalar[dtype](0)
    )
    var packed_filter = LayoutTensor[dtype, layout_p3](
        packed_filter_ptr,
        RuntimeLayout[layout_p3].row_major(packed_filter_shape),
    )

    var output_shape = extend_shape(output_dims, N, F)
    var output = LayoutTensor[dtype, layout_p2](
        output_ptr, RuntimeLayout[layout_p2].row_major(output_shape)
    )
    var output_ref = LayoutTensor[dtype, layout_p2](
        output_ref_ptr, RuntimeLayout[layout_p2].row_major(output_shape)
    )

    comptime if filter_packed:
        pack_filter(lt_to_tt(filter), lt_to_tt(packed_filter), num_groups)

    comptime conv_attr = ConvInfoStatic[rank]()

    @always_inline
    @__parameter
    def null_epilogue[rank: Int](coords: IndexList[rank], f_size: Int):
        pass

    comptime if filter_packed:
        ConvDirectNHWC[
            layout_p2,
            layout_p3,
            layout_p2,
            dtype,
            dtype,
            dtype,
            True,
            conv_attr,
        ].run(
            output_ref,
            input,
            packed_filter,
            conv_shape,
        )
    else:
        ConvDirectNHWC[
            layout_p2,
            layout_p2,
            layout_p2,
            dtype,
            dtype,
            dtype,
            False,
            conv_attr,
        ].run(
            output_ref,
            input,
            filter,
            conv_shape,
        )

    # Add bias and activatiion separately.
    var output_image_size = output_dims.flattened_length()
    for n in range(N):
        for i in range(output_image_size):
            var output_ref_ptr = output_ref.ptr + F * (
                i + output_image_size * n
            )

            @always_inline
            def body0[width: Int](offset: Int) {var}:
                output_ref_ptr.store(
                    offset,
                    10.0
                    * (
                        output_ref_ptr.load[width=width](offset)
                        + bias_ptr.load[width=width](offset)
                    ),
                )

            vectorize[simd_size](F, body0)

    # Test epilogue
    @always_inline
    @__parameter
    def epilogue[_rank: Int](coords: IndexList[_rank], f_size: Int):
        @always_inline
        def body1[width: Int](idx: Int) {imm}:
            var curr_coords = rebind[IndexList[rank + 2]](coords)
            curr_coords[rank + 1] += idx

            var vec = output.load[width=width](curr_coords)

            output.store(
                curr_coords,
                10.0
                * (vec + bias_ptr.load[width=width](curr_coords[rank + 1])),
            )

        vectorize[simd_size](f_size, body1)

    comptime if filter_packed:
        ConvDirectNHWC[
            layout_p2,
            layout_p3,
            layout_p2,
            dtype,
            dtype,
            dtype,
            True,
            conv_attr,
            epilogue,
        ].run(
            output,
            input,
            packed_filter,
            conv_shape,
        )
    else:
        ConvDirectNHWC[
            layout_p2,
            layout_p2,
            layout_p2,
            dtype,
            dtype,
            dtype,
            False,
            conv_attr,
            epilogue,
        ].run(
            output,
            input,
            filter,
            conv_shape,
        )

    # Check results, return on the first failed comparison.
    for i in range(output_size):
        if not isclose(
            output_ref.ptr[i],
            output.ptr[i],
            atol=1e-4,  # absolute error tolerance
            rtol=1e-4,  # relative error tolerance
        ):
            print("Input shape: ", input_shape)
            print("filter shape: ", filter_shape)
            print("filter packed", filter_packed)
            print("num groups", num_groups)
            print("flat output index:", i)
            print("Golden value: ", output_ref.ptr[i])
            print("Actual value: ", output.ptr[i])
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


def main() raises:
    # No packing or padding.
    test[2, .float32, False](
        1,  # N
        Index(6, 5),  # H, W
        1,  # C
        Index(3, 4),  # R, S
        4,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0, 0, 0),  # pad_h, pad_w
        1,  # num_groups
    )

    test[3, .float32, False](
        1,  # N
        Index(4, 8, 13),
        16,  # C
        Index(1, 2, 5),
        64,  # F
        Index(1, 1, 2),  # stride
        Index(1, 1, 1),  # dilation
        IndexList[6](0),  # pad_d, pad_h, pad_w
        1,  # num_groups
    )

    test[1, .float32, False](
        1,  # N
        Index(14),
        7,  # C
        Index(3),
        256,  # F
        Index(3),  # stride
        Index(1),  # dilation
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    # Pre-packed test w/o padding.

    test[2, .float32, True](
        1,  # N
        Index(12, 12),
        12,  # C
        Index(3, 3),
        64,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0, 0, 0),  # pad_h, pad_w
        1,  # num_groups
    )

    test[3, .float32, True](
        5,  # N
        Index(9, 12, 11),
        8,  # C
        Index(3, 3, 4),
        64,  # F
        Index(2, 2, 2),  # stride
        Index(1, 1, 1),  # dilation
        IndexList[6](0),  # pad_h, pad_w
        1,  # num_groups
    )

    test[1, .float32, True](
        1,  # N
        Index(17),
        11,  # C
        Index(3),
        192,  # F
        Index(3),  # stride
        Index(1),  # dilation
        Index(0, 0),  # pad_h, pad_w
        1,  # num_groups
    )

    # No packing, w/ padding, and F not multiple of simd_size.

    test[2, .float32, False](
        1,  # N
        Index(5, 5),
        3,  # C
        Index(3, 3),
        1,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1, 1, 1),  # pad_h, pad_w
        1,  # num_groups
    )

    test[3, .float32, False](
        1,  # N
        Index(9, 10, 5),
        2,  # C
        Index(2, 4, 3),
        6,  # F
        Index(3, 2, 3),  # stride
        Index(1, 1, 1),  # dilation
        IndexList[6](1, 0, 2, 1, 1, 1),
        1,  # num_groups
    )

    # Pre-packed, F not multiple of simd_size
    test[2, .float32, True](
        1,  # N
        Index(7, 7),
        2,  # C
        Index(3, 3),
        42,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0, 0, 0),  # pad_h, pad_w
        1,  # num_groups
    )

    test[1, .float32, True](
        1,  # N
        Index(11),
        2,  # C
        Index(5),
        7,  # F
        Index(1),  # stride
        Index(1),  # dilation
        Index(2, 2),  # pad_h, pad_w
        1,  # num_groups
    )

    test[3, .float32, True](
        1,  # N
        Index(7, 7, 9),
        2,  # C
        Index(4, 3, 3),
        42,  # F
        Index(2, 2, 2),  # stride
        Index(1, 1, 1),  # dilation
        IndexList[6](2, 1, 1, 1, 1, 1),
        1,  # num_groups
    )

    test[2, .float32, True](
        1,  # N
        Index(14, 14),
        3,  # C
        Index(3, 3),
        16,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1, 1, 1),
        1,  # num_groups
    )

    # grouped conv tests
    test[2, .float32, True](
        1,  # N
        Index(3, 3),
        18,  # C
        Index(3, 3),
        18,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0, 0, 0),
        3,  # num_groups
    )

    test[2, .float32, True](
        3,  # N
        Index(11, 17),
        36,  # C
        Index(3, 3),
        93,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1, 2, 2),
        3,  # num_groups
    )

    test[2, .float32, True](
        1,  # N
        Index(11, 17),
        36,  # C
        Index(2, 6),
        198,  # F
        Index(2, 3),  # stride
        Index(1, 1),  # dilation
        Index(1, 0, 3, 2),  # pad_h
        2,  # num_groups
    )

    # depthwise conv
    test[2, .float32, True](
        1,  # N
        Index(11, 7),
        33,  # C
        Index(3, 5),
        66,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1, 2, 2),
        33,  # num_groups
    )

    # 1D edge case
    test[1, .float32, True](
        2,  # N
        Index(49),  # W
        1024,  # C
        Index(128),  # S
        1024,  # F
        Index(1),  # stride
        Index(1),  # dilation
        Index(64, 64),  # pad_w
        64,  # num_groups
    )
