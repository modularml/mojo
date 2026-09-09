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
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from nn.conv.conv import conv_miopen

from std.itertools import product
from std.utils import IndexList


def conv2d_ref_cpu[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    conv_rank: Int,
](
    input_ptr: ImmPointer[Scalar[input_type], _],
    filter_ptr: ImmPointer[Scalar[filter_type], _],
    output_ptr: MutPointer[Scalar[output_type], _],
    input_dim: IndexList[conv_rank + 2],
    filter_dim: IndexList[conv_rank + 2],
    output_dim: IndexList[conv_rank + 2],
    stride: IndexList[conv_rank],
    dilation: IndexList[conv_rank],
    pad: IndexList[conv_rank],
    num_groups: Int,
):
    """CPU reference conv2d: input NHWC, filter RSCF, output NHWC."""
    comptime assert conv_rank == 2, "expected 2D parameters"

    var N = input_dim[0]
    var H = input_dim[1]
    var W = input_dim[2]
    var C_in = input_dim[3]

    var R = filter_dim[0]
    var S = filter_dim[1]
    var C_per_group = filter_dim[2]
    var F_out = filter_dim[3]

    var H_out = output_dim[1]
    var W_out = output_dim[2]

    var F_per_group = F_out // num_groups
    for n in range(N):
        for ho, wo in product(range(H_out), range(W_out)):
            for f in range(F_out):
                var g = f // F_per_group
                var ci_base = g * C_per_group
                var acc = Float32(0)
                for r, s in product(range(R), range(S)):
                    var hi = ho * stride[0] - pad[0] + r * dilation[0]
                    var wi = wo * stride[1] - pad[1] + s * dilation[1]
                    if 0 <= hi < H and 0 <= wi < W:
                        for c in range(C_per_group):
                            var in_idx = (
                                n * H * W * C_in
                                + hi * W * C_in
                                + wi * C_in
                                + ci_base
                                + c
                            )
                            var f_idx = (
                                r * S * C_per_group * F_out
                                + s * C_per_group * F_out
                                + c * F_out
                                + f
                            )
                            acc += Float32(input_ptr[in_idx]) * Float32(
                                filter_ptr[f_idx]
                            )
                # NHWC output
                var out_idx = (
                    n * H_out * W_out * F_out
                    + ho * W_out * F_out
                    + wo * F_out
                    + f
                )
                output_ptr[out_idx] = Scalar[output_type](acc)


def conv3d_ref_cpu[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    conv_rank: Int,
](
    input_ptr: ImmPointer[Scalar[input_type], _],
    filter_ptr: ImmPointer[Scalar[filter_type], _],
    output_ptr: MutPointer[Scalar[output_type], _],
    input_dim: IndexList[conv_rank + 2],
    filter_dim: IndexList[conv_rank + 2],
    output_dim: IndexList[conv_rank + 2],
    stride: IndexList[conv_rank],
    dilation: IndexList[conv_rank],
    pad: IndexList[conv_rank],
    num_groups: Int,
):
    """CPU reference conv3d: input NDHWC, filter QRSCF, output NDHWC."""
    comptime assert conv_rank == 3, "expected 2D inputs"

    var N = input_dim[0]
    var D = input_dim[1]
    var H = input_dim[2]
    var W = input_dim[3]
    var C_in = input_dim[4]

    var Q = filter_dim[0]
    var R = filter_dim[1]
    var S = filter_dim[2]
    var C_per_group = filter_dim[3]
    var F_out = filter_dim[4]

    var D_out = output_dim[1]
    var H_out = output_dim[2]
    var W_out = output_dim[3]

    var F_per_group = F_out // num_groups
    for n in range(N):
        for do, ho, wo in product(range(D_out), range(H_out), range(W_out)):
            for f in range(F_out):
                var g = f // F_per_group
                var ci_base = g * C_per_group
                var acc = Float32(0)
                for q, r, s in product(range(Q), range(R), range(S)):
                    var di = do * stride[0] - pad[0] + q * dilation[0]
                    var hi = ho * stride[1] - pad[1] + r * dilation[1]
                    var wi = wo * stride[2] - pad[2] + s * dilation[2]
                    if 0 <= di < D and 0 <= hi < H and 0 <= wi < W:
                        for c in range(C_per_group):
                            var in_idx = (
                                n * D * H * W * C_in
                                + di * H * W * C_in
                                + hi * W * C_in
                                + wi * C_in
                                + ci_base
                                + c
                            )
                            var f_idx = (
                                q * R * S * C_per_group * F_out
                                + r * S * C_per_group * F_out
                                + s * C_per_group * F_out
                                + c * F_out
                                + f
                            )
                            acc += Float32(input_ptr[in_idx]) * Float32(
                                filter_ptr[f_idx]
                            )
                # NDHWC output
                var out_idx = (
                    n * D_out * H_out * W_out * F_out
                    + do * H_out * W_out * F_out
                    + ho * W_out * F_out
                    + wo * F_out
                    + f
                )
                output_ptr[out_idx] = Scalar[output_type](acc)


# input: NHWC/NDHWC
# filter: RSCF/QRSCF
def test_conv_miopen[
    conv_rank: Int,
    //,
    input_dim: IndexList[conv_rank + 2],
    filter_dim: IndexList[conv_rank + 2],
    output_dim: IndexList[conv_rank + 2],
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    stride_dim: IndexList[conv_rank],
    dilation_dim: IndexList[conv_rank],
    pad_dim: IndexList[conv_rank],
    num_groups: Int = 1,
](ctx: DeviceContext) raises:
    print(
        "== test_miopen_conv: dtype_in=",
        input_type,
        " dtype_filter=",
        filter_type,
        " dtype_out=",
        output_type,
        " input=",
        input_dim,
        " filter=",
        filter_dim,
        " num_groups=",
        num_groups,
    )

    comptime input_dim_flattened = input_dim.flattened_length()
    comptime filter_dim_flattened = filter_dim.flattened_length()
    comptime output_dim_flattened = output_dim.flattened_length()

    # Allocate host memory
    var input_host_ptr = ctx.enqueue_create_host_buffer[input_type](
        input_dim_flattened
    )
    var filter_host_ptr = ctx.enqueue_create_host_buffer[filter_type](
        filter_dim_flattened
    )
    var output_ref_host_ptr = ctx.enqueue_create_host_buffer[output_type](
        output_dim_flattened
    )
    var output_host_ptr = ctx.enqueue_create_host_buffer[output_type](
        output_dim_flattened
    )

    # Create host TileTensors
    comptime input_tt_layout = row_major(Coord(input_dim))
    comptime filter_tt_layout = row_major(Coord(filter_dim))
    comptime output_tt_layout = row_major(Coord(output_dim))

    var input_host = TileTensor(input_host_ptr, input_tt_layout)
    var filter_host = TileTensor(filter_host_ptr, filter_tt_layout)
    var output_ref_host = TileTensor(output_ref_host_ptr, output_tt_layout)
    var output_host = TileTensor(output_host_ptr, output_tt_layout)

    random(input_host)
    random(filter_host)

    _ = output_host.fill(0)
    _ = output_ref_host.fill(0)

    # CPU reference
    comptime if conv_rank == 2:
        conv2d_ref_cpu(
            input_host_ptr.unsafe_ptr(),
            filter_host_ptr.unsafe_ptr(),
            output_ref_host_ptr.unsafe_ptr(),
            input_dim,
            filter_dim,
            output_dim,
            stride_dim,
            dilation_dim,
            pad_dim,
            num_groups,
        )
    else:
        comptime assert conv_rank == 3, "Only 2D/3D convolution supported"
        conv3d_ref_cpu(
            input_host_ptr.unsafe_ptr(),
            filter_host_ptr.unsafe_ptr(),
            output_ref_host_ptr.unsafe_ptr(),
            input_dim,
            filter_dim,
            output_dim,
            stride_dim,
            dilation_dim,
            pad_dim,
            num_groups,
        )

    # Allocate device buffers
    var input_dev = ctx.enqueue_create_buffer[input_type](input_dim_flattened)
    var filter_dev = ctx.enqueue_create_buffer[filter_type](
        filter_dim_flattened
    )
    var output_dev = ctx.enqueue_create_buffer[output_type](
        output_dim_flattened
    )

    # Create device TileTensors
    var input_dev_tensor = TileTensor(input_dev, input_tt_layout)
    var filter_dev_tensor = TileTensor(filter_dev, filter_tt_layout)
    var output_dev_tensor = TileTensor(output_dev, output_tt_layout)

    ctx.enqueue_copy(input_dev, input_host_ptr)
    ctx.enqueue_copy(filter_dev, filter_host_ptr)

    # Test: MIOpen conv with RSCF filter layout
    conv_miopen(
        input_dev_tensor,
        filter_dev_tensor,
        output_dev_tensor,
        stride_dim,
        dilation_dim,
        pad_dim,
        num_groups,
        ctx,
    )

    ctx.enqueue_copy(output_host_ptr, output_dev)
    ctx.synchronize()

    # Verify MIOpen output against CPU reference
    var max_diff = Float32(0)
    for i in range(output_dim_flattened):
        var diff = abs(
            Float32(output_host_ptr[i]) - Float32(output_ref_host_ptr[i])
        )
        if diff > max_diff:
            max_diff = diff
    # Use absolute tolerance appropriate for reduced precision types
    comptime if input_type == .float32:
        if max_diff > 1e-3:
            print("  FAIL: max_diff=", max_diff)
            raise Error("MIOpen output does not match CPU reference")
    else:
        # BF16/F16: accumulation over R*S*C elements with ~1e-2 relative
        # error per product. Observed max_diff ~0.25 for small 3x3x16
        # cases (BF16). 0.3 gives headroom without being overly loose.
        if max_diff > 0.3:
            print("  FAIL: max_diff=", max_diff)
            raise Error("MIOpen output does not match CPU reference")
    print("  PASS: max_diff=", max_diff)

    # Cleanup device buffers
    _ = input_dev^
    _ = filter_dev^
    _ = output_dev^


def main() raises:
    with DeviceContext() as ctx:
        # Test different data types
        comptime dtype_configs = (DType.float32, DType.float16, DType.bfloat16)

        comptime for i in range(len(dtype_configs)):
            comptime dtype = rebind[DType](dtype_configs[i])

            test_conv_miopen[
                IndexList[4](1, 8, 8, 16),  # input  (NHWC)
                IndexList[4](3, 3, 16, 32),  # filter (RSCF)
                IndexList[4](1, 6, 6, 32),  # output (NHWC)
                dtype,
                dtype,
                dtype,
                IndexList[2](1, 1),  # stride
                IndexList[2](1, 1),  # dilation
                IndexList[2](0, 0),  # no pad
            ](ctx)

        # Test with padding (Flux2 VAE decoder-style shape)
        test_conv_miopen[
            IndexList[4](1, 16, 16, 32),  # input  (NHWC)
            IndexList[4](3, 3, 32, 64),  # filter (RSCF)
            IndexList[4](1, 16, 16, 64),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](1, 1),  # padding
        ](ctx)

        # Test with stride=2
        test_conv_miopen[
            IndexList[4](1, 16, 16, 16),  # input  (NHWC)
            IndexList[4](3, 3, 16, 32),  # filter (RSCF)
            IndexList[4](1, 7, 7, 32),  # output (NHWC)
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            IndexList[2](2, 2),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # no pad
        ](ctx)

        # Test 1x1 convolution (channel projection)
        test_conv_miopen[
            IndexList[4](1, 8, 8, 32),  # input  (NHWC)
            IndexList[4](1, 1, 32, 64),  # filter (RSCF)
            IndexList[4](1, 8, 8, 64),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # no pad
        ](ctx)

        # Test grouped convolution (2 groups)
        test_conv_miopen[
            IndexList[4](1, 8, 8, 32),  # input  (NHWC)
            IndexList[4](3, 3, 16, 64),  # filter (RSCF) C_per_group=16
            IndexList[4](1, 6, 6, 64),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # no pad
            num_groups=2,
        ](ctx)

        # Test depthwise convolution (groups == C_in)
        test_conv_miopen[
            IndexList[4](1, 8, 8, 16),  # input  (NHWC)
            IndexList[4](3, 3, 1, 16),  # filter (RSCF) C_per_group=1
            IndexList[4](1, 6, 6, 16),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # no pad
            num_groups=16,
        ](ctx)

        # Test dilation > 1
        test_conv_miopen[
            IndexList[4](1, 16, 16, 16),  # input  (NHWC)
            IndexList[4](3, 3, 16, 32),  # filter (RSCF)
            IndexList[4](1, 12, 12, 32),  # output (NHWC) H_out=(16-2*2)/1+1=12
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](2, 2),  # dilation
            IndexList[2](0, 0),  # no pad
        ](ctx)

        # Test batch size > 1
        test_conv_miopen[
            IndexList[4](4, 8, 8, 16),  # input  (NHWC) N=4
            IndexList[4](3, 3, 16, 32),  # filter (RSCF)
            IndexList[4](4, 6, 6, 32),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # no pad
        ](ctx)

        # Test non-square spatial dims
        test_conv_miopen[
            IndexList[4](1, 12, 20, 16),  # input  (NHWC) H!=W
            IndexList[4](3, 3, 16, 32),  # filter (RSCF)
            IndexList[4](1, 10, 18, 32),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # no pad
        ](ctx)

        # Test channel reduction (C_in > F_out)
        test_conv_miopen[
            IndexList[4](1, 8, 8, 64),  # input  (NHWC) C_in=64
            IndexList[4](3, 3, 64, 16),  # filter (RSCF) F_out=16
            IndexList[4](1, 6, 6, 16),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # no pad
        ](ctx)

        # Test 5x5 filter
        test_conv_miopen[
            IndexList[4](1, 16, 16, 16),  # input  (NHWC)
            IndexList[4](5, 5, 16, 32),  # filter (RSCF)
            IndexList[4](1, 12, 12, 32),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # no pad
        ](ctx)

        # Test stride + padding combined
        test_conv_miopen[
            IndexList[4](1, 16, 16, 16),  # input  (NHWC)
            IndexList[4](3, 3, 16, 32),  # filter (RSCF)
            IndexList[4](1, 8, 8, 32),  # output (NHWC) H_out=(16+2-3)/2+1=8
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](2, 2),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](1, 1),  # pad
        ](ctx)

        # Test larger dims (different MIOpen algorithm paths)
        test_conv_miopen[
            IndexList[4](2, 32, 32, 128),  # input  (NHWC)
            IndexList[4](3, 3, 128, 256),  # filter (RSCF)
            IndexList[4](2, 32, 32, 256),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](1, 1),  # pad
        ](ctx)

        comptime for i in range(len(dtype_configs)):
            comptime dtype = rebind[DType](dtype_configs[i])

            test_conv_miopen[
                IndexList[5](1, 4, 10, 8, 16),  # input  (NHWC)
                IndexList[5](3, 3, 3, 16, 32),  # filter (RSCF)
                IndexList[5](1, 2, 8, 6, 32),  # output (NHWC)
                dtype,
                dtype,
                dtype,
                IndexList[3](1, 1, 1),  # stride
                IndexList[3](1, 1, 1),  # dilation
                IndexList[3](0, 0, 0),  # no pad
            ](ctx)

        # Test grouped convolution (2 groups)
        test_conv_miopen[
            IndexList[5](1, 8, 8, 8, 32),  # input  (NHWC)
            IndexList[5](3, 3, 3, 16, 64),  # filter (RSCF) C_per_group=16
            IndexList[5](1, 6, 6, 6, 64),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[3](1, 1, 1),  # stride
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](0, 0, 0),  # no pad
            num_groups=2,
        ](ctx)

        # Test depthwise convolution (groups == C_in)
        test_conv_miopen[
            IndexList[5](1, 8, 8, 8, 16),  # input  (NHWC)
            IndexList[5](3, 3, 3, 1, 16),  # filter (RSCF) C_per_group=1
            IndexList[5](1, 6, 6, 6, 16),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[3](1, 1, 1),  # stride
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](0, 0, 0),  # no pad
            num_groups=16,
        ](ctx)

    print("\nAll MIOpen conv tests passed!")
