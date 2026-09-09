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

# TODO:
# - Think about GPU memory access patterns/hierarchu
# - Use shared memory for transformation matrices (B, G, A) to avoid redundant loads
# - Use shared memory for input tiles to reduce global memory bandwidth
# - Add proper grid dimension calculation instead of hardcoded values
# - Implement proper tiling/slicing for rank 4 LayoutTensor to avoid get_tile workaround
# - Add support for padding/strides in the Winograd convolution
# - Add bounds checking for input dimensions
# - Add test cases for odd sizes, likely broken

from std.math import ceildiv

from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from layout import Idx, IntTuple, Layout, LayoutTensor, TileTensor, row_major
from layout.int_tuple import product
from layout._fillers import random
from nn.conv.conv import conv_gpu
from std.testing import assert_almost_equal, assert_true

from std.utils.index import IndexList
from std.utils.numerics import get_accum_type


@always_inline
def _get_b[
    dtype: DType, element_layout: Layout
](
    out B: LayoutTensor[
        dtype,
        Layout.row_major(4, 4),
        MutAnyOrigin,
        element_layout=element_layout,
    ]
):
    B = type_of(B).stack_allocation()
    # fmt:off
    B[0,0] = 1.0; B[0,1] =  0.0; B[0,2] = -1.0; B[0,3] =  0.0
    B[1,0] = 0.0; B[1,1] =  1.0; B[1,2] =  1.0; B[1,3] =  0.0
    B[2,0] = 0.0; B[2,1] = -1.0; B[2,2] =  1.0; B[2,3] =  0.0
    B[3,0] = 0.0; B[3,1] =  1.0; B[3,2] =  0.0; B[3,3] = -1.0
    # fmt:on


@always_inline
def _get_g[
    dtype: DType, element_layout: Layout
](
    out G: LayoutTensor[
        dtype,
        Layout.row_major(4, 3),
        MutAnyOrigin,
        element_layout=element_layout,
    ]
):
    G = type_of(G).stack_allocation()
    # fmt:off
    G[0,0] = 1.0; G[0,1] =  0.0; G[0,2] = 0.0
    G[1,0] = 0.5; G[1,1] =  0.5; G[1,2] = 0.5
    G[2,0] = 0.5; G[2,1] = -0.5; G[2,2] = 0.5
    G[3,0] = 0.0; G[3,1] =  0.0; G[3,2] = 1.0
    # fmt:on


@always_inline
def _get_a[
    dtype: DType, element_layout: Layout
](
    out A: LayoutTensor[
        dtype,
        Layout.row_major(2, 4),
        MutAnyOrigin,
        element_layout=element_layout,
    ]
):
    A = type_of(A).stack_allocation()
    # fmt:off
    A[0,0] = 1.0; A[0,1] = 1.0; A[0,2] =  1.0; A[0,3] =  0.0
    A[1,0] = 0.0; A[1,1] = 1.0; A[1,2] = -1.0; A[1,3] = -1.0
    # fmt:on


@always_inline
def matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    element_layout: Layout,
    //,
    transpose_b: Bool,
    s_type: DType = get_accum_type[c_type](),
](
    C: LayoutTensor[
        mut=True, c_type, c_layout, element_layout=element_layout, ...
    ],
    A: LayoutTensor[a_type, a_layout, element_layout=element_layout, ...],
    B: LayoutTensor[b_type, b_layout, element_layout=element_layout, ...],
):
    comptime M = Int(c_layout.shape[0])
    comptime N = Int(c_layout.shape[1])
    comptime K = Int(a_layout.shape[1])

    comptime if transpose_b:
        for i in range(M):
            for j in range(N):
                var sum: SIMD[s_type, C.element_size] = 0
                for k in range(K):
                    sum += A[i, k].cast[s_type]() * B[j, k].cast[s_type]()
                C[i, j] = sum.cast[c_type]()
    else:
        for i in range(M):
            for j in range(N):
                var sum: SIMD[s_type, C.element_size] = 0
                for k in range(K):
                    sum += A[i, k].cast[s_type]() * B[k, j].cast[s_type]()
                C[i, j] = sum.cast[c_type]()


# TODO: Workaround because I have not found a way to slice/tile a rank 4 LayoutTensor
# to a rank 2 LayoutTensor
@always_inline
def get_tile[
    dtype: DType, //, tile_size: Int
](
    input_tensor: LayoutTensor[dtype, ...],
    n: Int,
    h: Int,
    w: Int,
    c: Int,
) -> LayoutTensor[
    dtype,
    Layout.row_major(tile_size, tile_size),
    MutAnyOrigin,
    element_layout=input_tensor.element_layout,
]:
    # TODO: Issue because returning a stack variable? Workaround
    # with @always_inline
    var result = LayoutTensor[
        dtype,
        Layout.row_major(tile_size, tile_size),
        MutAnyOrigin,
        element_layout=input_tensor.element_layout,
    ].stack_allocation()

    for i in range(tile_size):
        for j in range(tile_size):
            result[i, j] = input_tensor[n, h + i, w + j, c]

    return result


# Each thread processes a 4x4 input tile to produce a 2x2 output tile.
# The thread accumulates contributions from all input channels for each output channel.
def winograd_conv2d_gpu_nhwc[
    element_layout: Layout,
    //,
    input_layout: Layout,
    filter_layout: Layout,
    output_layout: Layout,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    block_size: Int,
](
    input: LayoutTensor[
        input_type, input_layout, ImmutAnyOrigin, element_layout=element_layout
    ],
    filter: LayoutTensor[
        filter_type,
        filter_layout,
        ImmutAnyOrigin,
        element_layout=element_layout,
    ],
    output: LayoutTensor[
        mut=True,
        output_type,
        output_layout,
        MutAnyOrigin,
        element_layout=element_layout,
    ],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[
        4
    ],  # Format: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
):
    """Implements Winograd F(2x2, 3x3) convolution algorithm for GPU.
    Winograd convolution is an optimization that reduces amount of muls by
    using more adds. This is done by transforming the input and filter into a different form.
    The filters can be pre-transformed once and reused for different inputs (not implemented here).

    Each GPU thread processes a 4x4 input tile to produce a 2x2 output tile.

    Currently only supports:
    - 3x3 filters
    - Stride 1
    - Single input channel
    - Even filter input sizes
    - No padding
    - No dilation
    - NHWC input layout
    - RSCF filter layout
    """
    comptime assert input.rank == filter.rank == output.rank == 4

    # Dimensions
    var C_in = input.dim[3]()  # input channels
    var C_out = output.dim[3]()  # output channels
    var H_out = output.dim[1]()
    var W_out = output.dim[2]()

    # Get transformation matrices
    var b = _get_b[input_type, element_layout]()
    var g = _get_g[input_type, element_layout]()
    var a = _get_a[input_type, element_layout]()

    # Thread indices
    var n = block_idx.z
    var h_out = (block_idx.x * block_dim.x + thread_idx.x) * 2
    var w_out = (block_idx.y * block_dim.y + thread_idx.y) * 2

    # Check bounds
    if h_out + 1 >= H_out or w_out + 1 >= W_out:
        return

    # Allocate scratch space
    var scratch = LayoutTensor[
        input_type,
        Layout.row_major(4, 3),
        MutAnyOrigin,
        element_layout=element_layout,
    ].stack_allocation()
    var scratch_2 = LayoutTensor[
        input_type,
        Layout.row_major(4, 4),
        MutAnyOrigin,
        element_layout=element_layout,
    ].stack_allocation()
    var scratch_3 = LayoutTensor[
        input_type,
        Layout.row_major(2, 4),
        MutAnyOrigin,
        element_layout=element_layout,
    ].stack_allocation()
    var m = LayoutTensor[
        output_type,
        Layout.row_major(4, 4),
        MutAnyOrigin,
        element_layout=element_layout,
    ].stack_allocation()
    var g_transformed = LayoutTensor[
        input_type,
        Layout.row_major(4, 4),
        MutAnyOrigin,
        element_layout=element_layout,
    ].stack_allocation()

    # Pre-transform filter (G^T * filter * G)
    # offsets=(0, 0) specifies indices for the non-sliced dimensions (rank-2)
    var filter_slice = filter.slice[:, :, slice_indices=(0, 1)](offsets=(0, 0))
    matmul[False](scratch, g, filter_slice)
    matmul[True](g_transformed, scratch, g)

    # Process each output channel
    for c_out in range(C_out):
        var output_tile = LayoutTensor[
            output_type,
            Layout.row_major(2, 2),
            MutAnyOrigin,
            element_layout=element_layout,
        ].stack_allocation()

        # Process each input channel
        for c_in in range(C_in):
            # 1. Get input tile

            # TODO: Can we do something like this instead?
            # var input_tile = input_tensor.tile[1,1,4,4](c_out, c_in)
            var input_tile = get_tile[4](
                input.as_unsafe_any_origin(), n, h_out, w_out, c_in
            )

            # 2. Transform input (B^T * d * B)
            matmul[transpose_b=False](scratch_2, b, input_tile)
            matmul[transpose_b=True](input_tile, scratch_2, b)

            # 3. Element-wise multiply with transformed filter and accumulate
            # TODO: Can we do this instead? just need to figure out the casting of dtypes
            # m = input_tile * g_transformed
            for ii in range(4):
                for jj in range(4):
                    m[ii, jj] = (
                        input_tile[ii, jj][0].cast[output_type]()
                        * g_transformed[ii, jj][0].cast[output_type]()
                    )

            # 4. Transform output (A^T * m * A)
            matmul[transpose_b=False](scratch_3, a, m)
            matmul[transpose_b=True](output_tile, scratch_3, a)

            # 5. Store result
            for di in range(2):
                for dj in range(2):
                    output[n, h_out + di, w_out + dj, c_out] = output_tile[
                        di, dj
                    ][0]


def winograd_conv2d_gpu_launcher[
    element_layout: Layout,
    //,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: LayoutTensor[input_type, element_layout=element_layout, ...],
    filter: LayoutTensor[filter_type, element_layout=element_layout, ...],
    output: LayoutTensor[
        mut=True, output_type, element_layout=element_layout, ...
    ],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[
        4
    ],  # Format: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    comptime block_size = 16

    # TODO: Is assert_true the right way to do this?
    assert_true(
        input.dim[1]() % 2 == 0 and input.dim[2]() % 2 == 0,
        "H and W must be even number",
    )
    assert_true(
        input.dim[1]() >= 4 and input.dim[2]() >= 4,
        "Input must be at least 4x4",
    )
    assert_true(
        filter.dim[0]() == 3 and filter.dim[1]() == 3,
        "Filter must be 3x3",
    )
    assert_true(stride[0] == 1 and stride[1] == 1, "Stride not implemented")
    assert_true(
        dilation[0] == 1 and dilation[1] == 1, "Dilation not implemented"
    )
    assert_true(
        padding[0] == 0
        and padding[1] == 0
        and padding[2] == 0
        and padding[3] == 0,
        "Padding not implemented",
    )
    assert_true(num_groups == 1, "Num groups not implemented")
    assert_true(
        input.dim[3]() == filter.dim[2](),
        "Input and filter channels must match",
    )
    assert_true(input.dim[3]() == 1, "Multiple input channels not implemented")

    var grid_dim_x = ceildiv(output.dim[2](), 2 * block_size)
    var grid_dim_y = ceildiv(output.dim[1](), 2 * block_size)
    var grid_dim_z = input.dim[0]()

    comptime kernel = winograd_conv2d_gpu_nhwc[
        element_layout=element_layout,
        input.layout,
        filter.layout,
        output.layout,
        input_type,
        filter_type,
        output_type,
        block_size,
    ]

    ctx.enqueue_function[kernel](
        input.as_imm().as_unsafe_any_origin(),
        filter.as_imm().as_unsafe_any_origin(),
        output.as_unsafe_any_origin(),
        stride,
        dilation,
        padding,
        grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
        block_dim=(block_size, block_size),
    )


@always_inline
def get_output_dim[
    input_dim: IntTuple,
    filter_dim: IntTuple,
    stride: IndexList[2],
    dilation: IndexList[2],
    pad: IndexList[
        4
    ],  # Format: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
]() -> IndexList[4]:
    comptime N = Int(input_dim[0])
    comptime H = Int(input_dim[1])
    comptime W = Int(input_dim[2])
    comptime C = Int(input_dim[3])

    comptime R = Int(filter_dim[0])
    comptime S = Int(filter_dim[1])
    comptime F = Int(filter_dim[3])

    # Extract padding values: pad format is [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
    comptime pad_h = IndexList[2](pad[0], pad[1])
    comptime pad_w = IndexList[2](pad[2], pad[3])

    comptime HO = (
        H + pad_h[0] + pad_h[1] - dilation[0] * (R - 1) - 1
    ) // stride[0] + 1
    comptime WO = (
        W + pad_w[0] + pad_w[1] - dilation[1] * (S - 1) - 1
    ) // stride[1] + 1
    comptime output_dim = IndexList[4](N, HO, WO, F)
    return output_dim


def test_winograd_conv_gpu[
    dtype: DType,
    input_dim: IntTuple,
    filter_dim: IntTuple,
    stride: IndexList[2],
    dilation: IndexList[2],
    pad: IndexList[
        4
    ],  # Format: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
    num_groups: Int = 1,
](ctx: DeviceContext) raises:
    print("== test_conv_winograd_gpu")

    comptime output_dim = get_output_dim[
        input_dim, filter_dim, stride, dilation, pad
    ]()

    # Define TileTensor layouts
    comptime input_tt_layout = row_major(
        (
            Idx[Int(input_dim[0])],
            Idx[Int(input_dim[1])],
            Idx[Int(input_dim[2])],
            Idx[Int(input_dim[3])],
        )
    )
    comptime filter_tt_layout = row_major(
        (
            Idx[Int(filter_dim[0])],
            Idx[Int(filter_dim[1])],
            Idx[Int(filter_dim[2])],
            Idx[Int(filter_dim[3])],
        )
    )
    comptime output_tt_layout = row_major(
        (
            Idx[Int(output_dim[0])],
            Idx[Int(output_dim[1])],
            Idx[Int(output_dim[2])],
            Idx[Int(output_dim[3])],
        )
    )

    # Create device buffers
    var input_device = ctx.enqueue_create_buffer[dtype](product(input_dim))
    var filter_device = ctx.enqueue_create_buffer[dtype](product(filter_dim))
    var output_device = ctx.enqueue_create_buffer[dtype](
        output_dim.flattened_length()
    )
    var output_ref_device = ctx.enqueue_create_buffer[dtype](
        output_dim.flattened_length()
    )

    # Initialize input and filter with random values on host
    with input_device.map_to_host() as input_host:
        var input_host_tt = TileTensor(input_host, input_tt_layout)
        random(input_host_tt)

    with filter_device.map_to_host() as filter_host:
        var filter_host_tt = TileTensor(filter_host, filter_tt_layout)
        random(filter_host_tt)

    # Create device TileTensors
    var input_tt = TileTensor(input_device, input_tt_layout)
    var filter_tt = TileTensor(filter_device, filter_tt_layout)
    var output_tt = TileTensor(output_device, output_tt_layout)
    var output_ref_tt = TileTensor(output_ref_device, output_tt_layout)

    # Run reference convolution
    conv_gpu[dtype, dtype, dtype](
        input_tt,
        filter_tt,
        output_ref_tt,
        stride,
        dilation,
        pad,
        num_groups,
        ctx,
    )

    # Run winograd convolution
    winograd_conv2d_gpu_launcher[dtype, dtype, dtype](
        input_tt.to_layout_tensor(),
        filter_tt.to_layout_tensor(),
        output_tt.to_layout_tensor(),
        stride,
        dilation,
        pad,
        num_groups,
        ctx,
    )

    ctx.synchronize()

    # Verify results
    comptime atol = 1e-06 if dtype == DType.float32 else 1e-1
    comptime rtol = 1e-06 if dtype == DType.float32 else 1e-4

    with output_device.map_to_host() as output_host:
        with output_ref_device.map_to_host() as output_ref_host:
            for x in range(output_dim.flattened_length()):
                assert_almost_equal(
                    output_ref_host[x],
                    output_host[x],
                    atol=atol,
                    rtol=rtol,
                )


def main() raises:
    comptime dtype = DType.float32

    with DeviceContext() as ctx:
        test_winograd_conv_gpu[
            dtype=dtype,
            input_dim=IntTuple(1, 8, 8, 1),
            filter_dim=IntTuple(3, 3, 1, 1),
            stride=IndexList[2](1, 1),
            dilation=IndexList[2](1, 1),
            pad=IndexList[4](
                0, 0, 0, 0
            ),  # [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        ](ctx)

        test_winograd_conv_gpu[
            dtype=dtype,
            input_dim=IntTuple(32, 256, 256, 1),
            filter_dim=IntTuple(3, 3, 1, 1),
            stride=IndexList[2](1, 1),
            dilation=IndexList[2](1, 1),
            pad=IndexList[4](
                0, 0, 0, 0
            ),  # [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        ](ctx)

        test_winograd_conv_gpu[
            dtype=dtype,
            input_dim=IntTuple(1, 4, 16, 1),
            filter_dim=IntTuple(3, 3, 1, 1),
            stride=IndexList[2](1, 1),
            dilation=IndexList[2](1, 1),
            pad=IndexList[4](
                0, 0, 0, 0
            ),  # [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        ](ctx)

        test_winograd_conv_gpu[
            dtype=dtype,
            input_dim=IntTuple(1, 16, 4, 1),
            filter_dim=IntTuple(3, 3, 1, 1),
            stride=IndexList[2](1, 1),
            dilation=IndexList[2](1, 1),
            pad=IndexList[4](
                0, 0, 0, 0
            ),  # [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        ](ctx)

        test_winograd_conv_gpu[
            dtype=DType.bfloat16,
            input_dim=IntTuple(1, 32, 32, 1),
            filter_dim=IntTuple(3, 3, 1, 1),
            stride=IndexList[2](1, 1),
            dilation=IndexList[2](1, 1),
            pad=IndexList[4](
                0, 0, 0, 0
            ),  # [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        ](ctx)
