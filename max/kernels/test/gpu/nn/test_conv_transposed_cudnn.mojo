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
from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from layout._fillers import random
from nn.conv.conv_transpose import conv_transpose_naive, conv_transposed_cudnn
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList

comptime dtype = DType.float32


def test_conv_transposed_cudnn[
    input_len: Int,
    kernel_len: Int,
    in_channels: Int = 1,
    out_channels: Int = 1,
    stride_val: Int = 1,
    dilation_val: Int = 1,
    pad_val: Int = 0,
    dtype: DType = .float32,
](ctx: DeviceContext,) raises:
    """
    Fixed 1-D transposed-convolution test with correct QRSFC kernel layout.
    """

    print(
        "input_len=",
        input_len,
        ", kernel_size=",
        kernel_len,
        ", in_channels=",
        in_channels,
        ", out_channels=",
        out_channels,
        ", stride=",
        stride_val,
        ", pad=",
        pad_val,
        ", dilation=",
        dilation_val,
        ")",
    )

    # Shapes and sizes
    comptime output_len = (
        stride_val * (input_len - 1)
        + dilation_val * (kernel_len - 1)
        - 2 * pad_val
        + 1
    )
    comptime input_size = 1 * 1 * input_len * in_channels
    comptime filter_size = 1 * kernel_len * out_channels * in_channels
    comptime output_size = 1 * 1 * output_len * out_channels

    # Layouts for NHWC tensors
    comptime input_layout = row_major[1, 1, input_len, in_channels]()
    comptime filter_layout = row_major[
        1, kernel_len, out_channels, in_channels
    ]()
    comptime output_layout = row_major[1, 1, output_len, out_channels]()

    # Layouts for NCHW tensors (for cuDNN)
    comptime input_nchw_size = 1 * in_channels * 1 * input_len
    comptime output_nchw_size = 1 * out_channels * 1 * output_len
    comptime filter_nchw_size = in_channels * out_channels * 1 * kernel_len
    comptime input_nchw_layout = row_major[1, in_channels, 1, input_len]()
    comptime output_nchw_layout = row_major[1, out_channels, 1, output_len]()
    comptime filter_nchw_layout = row_major[
        in_channels, out_channels, 1, kernel_len
    ]()

    # Allocate host memory for NHWC tensors
    var input_host_ptr = ctx.enqueue_create_host_buffer[dtype](input_size)
    var filter_host_ptr = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var output_ref_host_ptr = ctx.enqueue_create_host_buffer[dtype](output_size)

    # Create host TileTensors for NHWC
    var input_host = TileTensor(input_host_ptr, input_layout)
    var filter_host = TileTensor(filter_host_ptr, filter_layout)
    var output_ref_host = TileTensor(output_ref_host_ptr, output_layout)

    random(input_host)
    random(filter_host)

    # Parameters (1-D ⇒ only W dimension varies).
    var stride = Index(1, 1, stride_val)
    var dilation = Index(1, 1, dilation_val)
    var pad_d = Index(0, 0)  # depth – none in 1-D
    var pad_h = Index(0, 0)  # height – none in 1-D
    var pad_w = Index(pad_val, pad_val)  # width padding (symmetric)

    # Execute naive reference implementation.
    conv_transpose_naive[dtype](
        TileTensor(
            output_ref_host_ptr,
            row_major(Coord(IndexList[5](1, 1, 1, output_len, out_channels))),
        ),
        TileTensor(
            input_host_ptr,
            row_major(Coord(IndexList[5](1, 1, 1, input_len, in_channels))),
        ),
        TileTensor(
            filter_host_ptr,
            row_major(
                Coord(IndexList[5](1, 1, kernel_len, out_channels, in_channels))
            ),
        ),
        stride,
        dilation,
        pad_d,  # D
        pad_h,  # H
        pad_w,  # W
    )

    # -------------------------------------------------------------
    # 2. Run the same transposed-convolution via cuDNN backward-data
    # -------------------------------------------------------------

    # Allocate host memory for NCHW tensors (for cuDNN)
    var input_nchw_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        input_nchw_size
    )
    var filter_nchw_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        filter_nchw_size
    )
    var output_nchw_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        output_nchw_size
    )

    # Create host TileTensors for NCHW
    var input_nchw_host = TileTensor(input_nchw_host_ptr, input_nchw_layout)
    var filter_nchw_host = TileTensor(filter_nchw_host_ptr, filter_nchw_layout)
    var output_nchw_host = TileTensor(output_nchw_host_ptr, output_nchw_layout)

    # Convert input/output data from NHWC to NCHW layout for cuDNN
    for w in range(input_len):
        for c in range(in_channels):
            input_nchw_host[0, c, 0, w] = input_host[0, 0, w, c]

    # Convert filter data from QRSFC to CFHW layout for cuDNN
    for r in range(1):
        for s in range(kernel_len):
            for f in range(out_channels):
                for c in range(in_channels):
                    filter_nchw_host[c, f, r, s] = filter_host[r, s, f, c]

    # Create device buffers
    var d_input = ctx.enqueue_create_buffer[dtype](input_nchw_size)
    var d_filter = ctx.enqueue_create_buffer[dtype](filter_nchw_size)
    var d_output = ctx.enqueue_create_buffer[dtype](output_nchw_size)

    ctx.enqueue_copy(d_input, input_nchw_host_ptr)
    ctx.enqueue_copy(d_filter, filter_nchw_host_ptr)

    var stride_hw = Index(1, stride_val)
    var dilation_hw = Index(1, dilation_val)
    var padding_hw = Index(0, pad_val)

    # Invoke cuDNN helper.
    conv_transposed_cudnn[dtype, dtype, dtype](
        TileTensor(
            d_input,
            row_major(Coord(IndexList[4](1, in_channels, 1, input_len))),
        ),  # dy (input grad)
        TileTensor(
            d_filter,
            row_major(
                Coord(IndexList[4](in_channels, out_channels, 1, kernel_len))
            ),
        ),  # w (filter)
        TileTensor(
            d_output,
            row_major(Coord(IndexList[4](1, out_channels, 1, output_len))),
        ),  # dx (output)
        stride_hw,
        dilation_hw,
        padding_hw,
        ctx,
    )

    # Copy result back to host
    ctx.enqueue_copy(output_nchw_host_ptr, d_output)
    ctx.synchronize()

    # -------------------------------------------------------------
    # 3. Compare naive vs cuDNN results
    # -------------------------------------------------------------

    # verifying results
    for w in range(output_len):
        for f in range(out_channels):
            assert_almost_equal(
                output_ref_host[0, 0, w, f],
                output_nchw_host[0, f, 0, w],
                rtol=0.0001,
            )
    print("Succeed")

    # Cleanup device buffers
    _ = d_input^
    _ = d_filter^
    _ = d_output^


def main() raises:
    with DeviceContext() as ctx:
        # Check if we're running on an NVIDIA GPU
        if ctx.default_device_info.api != "cuda":
            print("Skipping cuDNN tests - not running on NVIDIA GPU")
            return

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            in_channels=2,
            out_channels=2,
            stride_val=1,
            dilation_val=1,
            pad_val=0,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=1,
            dilation_val=1,
            pad_val=0,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=1,
            dilation_val=1,
            pad_val=1,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=2,
            dilation_val=1,
            pad_val=0,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=1,
            dilation_val=2,
            pad_val=0,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=2,
            dilation_val=1,
            pad_val=1,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=1,
            dilation_val=2,
            pad_val=1,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=2,
            dilation_val=2,
            pad_val=0,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=2,
            dilation_val=2,
            pad_val=1,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=550,
            kernel_len=7,
            in_channels=512,
            out_channels=1024,
            stride_val=1,
            dilation_val=1,
            pad_val=3,
        ](ctx=ctx)

    # Test with multiple device contexts consecutively
    print("\n== Testing with multiple device contexts ==")

    # First context - default device (GPU 0)
    print("Creating first device context (default device)...")
    with DeviceContext() as ctx1:
        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            in_channels=2,
            out_channels=2,
            stride_val=1,
            dilation_val=1,
            pad_val=0,
        ](ctx=ctx1)

    if DeviceContext.number_of_devices() >= 2:
        # Second context - device 1
        print("Creating second device context (device 1)...")
        with DeviceContext(device_id=1) as ctx2:
            test_conv_transposed_cudnn[
                input_len=9,
                kernel_len=4,
                in_channels=2,
                out_channels=2,
                stride_val=1,
                dilation_val=1,
                pad_val=0,
            ](ctx=ctx2)

        print("Multiple device context test completed successfully!")
