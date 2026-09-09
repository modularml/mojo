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
from layout import Idx, TileTensor, row_major
from layout._fillers import random
from nn.conv.conv import conv_cudnn, conv_gpu
from std.testing import assert_almost_equal

from std.utils.index import IndexList


# input: NHWC
# filer: RSCF
def test_conv_cudnn[
    input_dim: IndexList[4],
    filter_dim: IndexList[4],
    output_dim: IndexList[4],
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    stride_dim: IndexList[2],
    dilation_dim: IndexList[2],
    pad_dim: IndexList[
        4
    ],  # Format: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
    num_groups: Int = 1,
](ctx: DeviceContext) raises:
    print(
        "== test_cudnn_conv_gpu: dtype_in=",
        input_type,
        " dtype_filter=",
        filter_type,
        " dtype_out=",
        output_type,
        " num_groups=",
        num_groups,
    )

    # Extract dimensions
    comptime N = input_dim[0]
    comptime H = input_dim[1]
    comptime W = input_dim[2]
    comptime C_in = input_dim[3]

    comptime R = filter_dim[0]
    comptime S = filter_dim[1]
    comptime C = filter_dim[2]
    comptime F = filter_dim[3]

    comptime Nout = output_dim[0]
    comptime Hout = output_dim[1]
    comptime Wout = output_dim[2]
    comptime Cout = output_dim[3]

    comptime input_dim_flattened = N * H * W * C_in
    comptime filter_dim_flattened = R * S * C * F
    comptime output_dim_flattened = Nout * Hout * Wout * Cout

    # Define TileTensor layouts
    comptime input_tt_layout = row_major((Idx[N], Idx[H], Idx[W], Idx[C_in]))
    comptime filter_tt_layout = row_major((Idx[R], Idx[S], Idx[C], Idx[F]))
    comptime filter_nchw_tt_layout = row_major((Idx[F], Idx[C], Idx[R], Idx[S]))
    comptime output_tt_layout = row_major(
        (Idx[Nout], Idx[Hout], Idx[Wout], Idx[Cout])
    )

    # Allocate host memory
    var input_host_ptr = ctx.enqueue_create_host_buffer[input_type](
        input_dim_flattened
    )
    var filter_host_ptr = ctx.enqueue_create_host_buffer[filter_type](
        filter_dim_flattened
    )
    var filter_nchw_host_ptr = ctx.enqueue_create_host_buffer[filter_type](
        filter_dim_flattened
    )
    var output_ref_host_ptr = ctx.enqueue_create_host_buffer[output_type](
        output_dim_flattened
    )
    var output_host_ptr = ctx.enqueue_create_host_buffer[output_type](
        output_dim_flattened
    )

    # Create host TileTensors
    var input_host = TileTensor(
        Span(
            unsafe_ptr=input_host_ptr.unsafe_ptr(), length=input_dim_flattened
        ),
        input_tt_layout,
    )
    var filter_host = TileTensor(
        Span(
            unsafe_ptr=filter_host_ptr.unsafe_ptr(), length=filter_dim_flattened
        ),
        filter_tt_layout,
    )
    var filter_nchw_host = TileTensor(
        Span(
            unsafe_ptr=filter_nchw_host_ptr.unsafe_ptr(),
            length=filter_dim_flattened,
        ),
        filter_nchw_tt_layout,
    )
    var output_ref_host = TileTensor(
        Span(
            unsafe_ptr=output_ref_host_ptr.unsafe_ptr(),
            length=output_dim_flattened,
        ),
        output_tt_layout,
    )
    var output_host = TileTensor(
        Span(
            unsafe_ptr=output_host_ptr.unsafe_ptr(), length=output_dim_flattened
        ),
        output_tt_layout,
    )

    random(input_host)
    random(filter_host)

    # Transpose filter to NCHW
    for r in range(R):
        for s in range(S):
            for c in range(C):
                for f in range(F):
                    filter_nchw_host[f, c, r, s] = filter_host[r, s, c, f]

    _ = output_host.fill(0)
    _ = output_ref_host.fill(0)

    # Allocate device buffers
    var input_dev = ctx.enqueue_create_buffer[input_type](input_dim_flattened)
    var filter_dev = ctx.enqueue_create_buffer[filter_type](
        filter_dim_flattened
    )
    var filter_nchw_dev = ctx.enqueue_create_buffer[filter_type](
        filter_dim_flattened
    )
    var output_dev = ctx.enqueue_create_buffer[output_type](
        output_dim_flattened
    )
    var output_ref_dev = ctx.enqueue_create_buffer[output_type](
        output_dim_flattened
    )

    # Create device TileTensors
    var input_dev_tt = TileTensor(input_dev, input_tt_layout)
    var filter_dev_tt = TileTensor(filter_dev, filter_tt_layout)
    var filter_nchw_dev_tt = TileTensor(filter_nchw_dev, filter_nchw_tt_layout)
    var output_dev_tt = TileTensor(output_dev, output_tt_layout)
    var output_ref_dev_tt = TileTensor(output_ref_dev, output_tt_layout)

    ctx.enqueue_copy(input_dev, input_host_ptr)
    ctx.enqueue_copy(filter_dev, filter_host_ptr)
    ctx.enqueue_copy(filter_nchw_dev, filter_nchw_host_ptr)

    conv_gpu[
        input_type,
        filter_type,
        output_type,
    ](
        input_dev_tt,
        filter_dev_tt,
        output_ref_dev_tt,
        stride_dim,
        dilation_dim,
        pad_dim,
        num_groups,
        ctx,
    )

    # conv_cudnn is a lower-level API that directly calls cuDNN, which only
    # supports symmetric padding. Since cuDNN doesn't support asymmetric
    # padding, we extract the symmetric values. For symmetric padding:
    # [pad_h_before, pad_h_after, pad_w_before, pad_w_after] -> [pad_h, pad_w].
    # Note: This test only uses symmetric padding, so
    # pad_h_before == pad_h_after and pad_w_before == pad_w_after.
    var pad_for_cudnn = IndexList[2](pad_dim[0], pad_dim[2])
    conv_cudnn[input_type, filter_type, output_type](
        input_dev_tt,
        filter_nchw_dev_tt,
        output_dev_tt,
        stride_dim,
        dilation_dim,
        pad_for_cudnn,
        num_groups,
        ctx,
    )

    ctx.enqueue_copy(output_ref_host_ptr, output_ref_dev)
    ctx.enqueue_copy(output_host_ptr, output_dev)
    ctx.synchronize()

    # Verify results
    for n in range(Nout):
        for h in range(Hout):
            for w in range(Wout):
                for f in range(Cout):
                    assert_almost_equal(
                        output_host[n, h, w, f],
                        output_ref_host[n, h, w, f],
                        rtol=0.01,
                    )
    print("Succeed")

    # Cleanup device buffers
    _ = input_dev^
    _ = filter_dev^
    _ = filter_nchw_dev^
    _ = output_dev^
    _ = output_ref_dev^


def main() raises:
    with DeviceContext() as ctx:
        # Test configurations for data types.
        comptime dtype_configs = (DType.float32, DType.float16, DType.bfloat16)

        test_conv_cudnn[
            IndexList[4](1, 1, 550, 1024),  # input  (NHWC)
            IndexList[4](
                1, 7, 1024, 1024
            ),  # filter (RSCF) (height, width, in_channels, out_channels)
            IndexList[4](1, 1, 550, 1024),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[4](
                0, 0, 3, 3
            ),  # pad: [pad_h_before, pad_h_after, pad_w_before, pad_w_after] (symmetric: pad_h=0, pad_w=3)
        ](ctx)

        # Test different data types.
        comptime for i in range(len(dtype_configs)):
            comptime dtype = rebind[DType](dtype_configs[i])

            test_conv_cudnn[
                IndexList[4](1, 8, 8, 16),  # input  (NHWC)
                IndexList[4](3, 3, 16, 32),  # filter (RSCF)
                IndexList[4](1, 6, 6, 32),  # output (NHWC)
                dtype,
                dtype,
                dtype,
                IndexList[2](1, 1),  # stride
                IndexList[2](1, 1),  # dilation
                IndexList[4](
                    0, 0, 0, 0
                ),  # pad: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
            ](ctx)

        # Test grouped convolutions
        # NOTE: Grouped convolutions are not supported by conv_gpu's naive implementation,
        # so we cannot validate against it. These tests would need a different reference
        # implementation or manual validation.

        # TODO(KERN-1846): Add grouped convolution tests once we have a proper
        # reference implementation.
        # The following configurations would be tested:
        # - Standard conv with num_groups=1
        # - 2 groups
        # - 4 groups
        # - Depthwise convolution (num_groups = in_channels)
        # - Grouped convolution with float16

    # Test with multiple device contexts consecutively
    print("\n== Testing with multiple device contexts ==")

    # First context - default device (GPU 0)
    print("Creating first device context (default device)...")
    with DeviceContext() as ctx1:
        test_conv_cudnn[
            IndexList[4](1, 8, 8, 16),  # input  (NHWC)
            IndexList[4](3, 3, 16, 32),  # filter (RSCF)
            IndexList[4](1, 6, 6, 32),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[4](
                0, 0, 0, 0
            ),  # pad: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        ](ctx1)

    if DeviceContext.number_of_devices() >= 2:
        # Second context - device 1
        print("Creating second device context (device 1)...")
        with DeviceContext(device_id=1) as ctx2:
            test_conv_cudnn[
                IndexList[4](1, 8, 8, 16),  # input  (NHWC)
                IndexList[4](3, 3, 16, 32),  # filter (RSCF)
                IndexList[4](1, 6, 6, 32),  # output (NHWC)
                DType.float32,
                DType.float32,
                DType.float32,
                IndexList[2](1, 1),  # stride
                IndexList[2](1, 1),  # dilation
                IndexList[4](
                    0, 0, 0, 0
                ),  # pad: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
            ](ctx2)

        print("Multiple device context test completed successfully!")
