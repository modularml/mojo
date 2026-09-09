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
"""FLUX VAE decoder conv2d integration test.

Validates SM100 structured conv2d dispatch through the nn conv_gpu entry
point using actual FLUX VAE decoder layer dimensions. Each test exercises
the full dispatch path: conv_gpu -> SM100 dispatch -> filter transpose ->
conv2d_fprop, and validates against cuDNN reference.

FLUX VAE decoder uses block_out_channels=[128, 256, 512, 512] with:
- 3x3 convolutions with padding=1 (ResNet blocks, upsample)
- 1x1 convolutions without padding (shortcuts, conv_in)
- All stride=1, dilation=1, groups=1
- BF16 data type
"""

from std.random import rand
from std.testing import assert_false

from layout import Idx, TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.conv.conv import conv_gpu, conv_cudnn

from std.utils.index import IndexList


def test_flux_conv_layer[
    # Input NHWC
    N: Int,
    H: Int,
    W: Int,
    C_in: Int,
    # Filter RSCF
    R: Int,
    S: Int,
    C_out: Int,
    # Padding (symmetric)
    pad: Int,
    # Name for output
    name: StringLiteral,
](ctx: DeviceContext) raises:
    """Test a single FLUX VAE decoder conv layer through conv_gpu dispatch.

    Validates SM100 dispatch output against cuDNN reference.
    """
    comptime Hout = H + 2 * pad - R + 1
    comptime Wout = W + 2 * pad - S + 1

    comptime dtype = DType.bfloat16

    comptime in_size = N * H * W * C_in
    comptime filter_size = R * S * C_in * C_out
    comptime out_size = N * Hout * Wout * C_out

    print(
        "  ",
        name,
        ": ",
        H,
        "x",
        W,
        "x",
        C_in,
        " -> ",
        C_out,
        "ch (",
        R,
        "x",
        S,
        " pad=",
        pad,
        ")",
        sep="",
    )

    # Host memory
    var input_host_ptr = ctx.enqueue_create_host_buffer[dtype](in_size)
    var filter_host_ptr = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var filter_nchw_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        filter_size
    )
    var out_sm100_host_ptr = ctx.enqueue_create_host_buffer[dtype](out_size)
    var out_cudnn_host_ptr = ctx.enqueue_create_host_buffer[dtype](out_size)

    rand(input_host_ptr.as_span())
    rand(filter_host_ptr.as_span())

    # Transpose filter RSCF -> FCRS for cuDNN reference
    for r in range(R):
        for s in range(S):
            for c in range(C_in):
                for f in range(C_out):
                    var rscf_idx = (
                        r * S * C_in * C_out + s * C_in * C_out + c * C_out + f
                    )
                    var fcrs_idx = f * C_in * R * S + c * R * S + r * S + s
                    filter_nchw_host_ptr[fcrs_idx] = filter_host_ptr[rscf_idx]

    # Device memory
    var input_dev = ctx.enqueue_create_buffer[dtype](in_size)
    var filter_rscf_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var filter_fcrs_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var out_sm100_dev = ctx.enqueue_create_buffer[dtype](out_size)
    var out_cudnn_dev = ctx.enqueue_create_buffer[dtype](out_size)

    ctx.enqueue_copy(input_dev, input_host_ptr)
    ctx.enqueue_copy(filter_rscf_dev, filter_host_ptr)
    ctx.enqueue_copy(filter_fcrs_dev, filter_nchw_host_ptr)

    # SM100 path: conv_gpu with RSCF filter (dispatches to SM100 on B200)
    comptime input_tt_layout = row_major((Idx[N], Idx[H], Idx[W], Idx[C_in]))
    comptime filter_rscf_tt_layout = row_major(
        (Idx[R], Idx[S], Idx[C_in], Idx[C_out])
    )
    comptime output_tt_layout = row_major(
        (Idx[N], Idx[Hout], Idx[Wout], Idx[C_out])
    )
    var input_tt = TileTensor(input_dev, input_tt_layout)
    var filter_rscf_tt = TileTensor(filter_rscf_dev, filter_rscf_tt_layout)
    var out_sm100_tt = TileTensor(out_sm100_dev, output_tt_layout)

    conv_gpu[
        dtype,
        dtype,
        dtype,
    ](
        input_tt,
        filter_rscf_tt,
        out_sm100_tt,
        IndexList[2](1, 1),  # stride
        IndexList[2](1, 1),  # dilation
        IndexList[4](pad, pad, pad, pad),  # symmetric padding
        1,  # groups
        ctx,
    )

    # cuDNN reference
    comptime filter_fcrs_tt_layout = row_major(
        (Idx[C_out], Idx[C_in], Idx[R], Idx[S])
    )
    var input_tt_ref = TileTensor(input_dev, input_tt_layout)
    var filter_fcrs_tt = TileTensor(filter_fcrs_dev, filter_fcrs_tt_layout)
    var out_cudnn_tt = TileTensor(out_cudnn_dev, output_tt_layout)

    conv_cudnn[dtype, dtype, dtype](
        input_tt_ref,
        filter_fcrs_tt,
        out_cudnn_tt,
        IndexList[2](1, 1),  # stride
        IndexList[2](1, 1),  # dilation
        IndexList[2](pad, pad),  # padding
        1,  # groups
        ctx,
    )

    ctx.synchronize()

    # Copy results to host
    ctx.enqueue_copy(out_sm100_host_ptr, out_sm100_dev)
    ctx.enqueue_copy(out_cudnn_host_ptr, out_cudnn_dev)
    ctx.synchronize()

    # Validate
    var max_diff: Float32 = 0.0
    var errors = 0
    for i in range(out_size):
        var sm100_val = out_sm100_host_ptr[i].cast[.float32]()
        var cudnn_val = out_cudnn_host_ptr[i].cast[.float32]()
        var diff = abs(sm100_val - cudnn_val)
        if diff > max_diff:
            max_diff = diff
        # BF16 relative tolerance
        var scale = max(abs(cudnn_val), Float32(1e-6))
        if diff / scale > 0.02:  # 2% relative tolerance
            errors += 1

    if errors > 0:
        print("    FAILED: ", errors, " errors, max_diff=", max_diff)
    else:
        print("    PASSED (max_diff=", max_diff, ")")

    assert_false(errors > 0, "SM100 conv2d output mismatch vs cuDNN")

    # Cleanup
    _ = input_dev^
    _ = filter_rscf_dev^
    _ = filter_fcrs_dev^
    _ = out_sm100_dev^
    _ = out_cudnn_dev^


def main() raises:
    print("=" * 60)
    print("FLUX VAE Decoder Conv2D Integration Test")
    print("SM100 dispatch through conv_gpu vs cuDNN reference")
    print("=" * 60)
    print()

    with DeviceContext() as ctx:
        # ============================================================
        # FLUX VAE Decoder Layers (block_out_channels=[128,256,512,512])
        # Spatial sizes: 16x16 -> 32x32 -> 64x64 -> 128x128
        # ============================================================

        print("--- Mid Block (512ch, 16x16) ---")
        # mid_block.conv1: 512 -> 512, 3x3, pad=1
        test_flux_conv_layer[1, 16, 16, 512, 3, 3, 512, 1, "mid_conv1"](ctx)
        # mid_block.conv2: 512 -> 512, 3x3, pad=1
        test_flux_conv_layer[1, 16, 16, 512, 3, 3, 512, 1, "mid_conv2"](ctx)

        print("\n--- Up Block 0 (512->512, 16x16) ---")
        # ResnetBlock: 512 -> 512, 3x3, pad=1
        test_flux_conv_layer[1, 16, 16, 512, 3, 3, 512, 1, "resnet_conv"](ctx)

        print("\n--- Up Block 1 (512->256, 32x32) ---")
        # ResnetBlock: 512 -> 256, 3x3, pad=1
        test_flux_conv_layer[1, 32, 32, 512, 3, 3, 256, 1, "resnet_conv"](ctx)
        # Shortcut: 512 -> 256, 1x1, pad=0
        test_flux_conv_layer[1, 32, 32, 512, 1, 1, 256, 0, "shortcut"](ctx)
        # Upsample: 256 -> 256, 3x3, pad=1
        test_flux_conv_layer[1, 32, 32, 256, 3, 3, 256, 1, "upsample"](ctx)

        print("\n--- Up Block 2 (256->128, 64x64) ---")
        # ResnetBlock: 256 -> 128, 3x3, pad=1
        test_flux_conv_layer[1, 64, 64, 256, 3, 3, 128, 1, "resnet_conv"](ctx)
        # Shortcut: 256 -> 128, 1x1, pad=0
        test_flux_conv_layer[1, 64, 64, 256, 1, 1, 128, 0, "shortcut"](ctx)
        # Upsample: 128 -> 128, 3x3, pad=1
        test_flux_conv_layer[1, 64, 64, 128, 3, 3, 128, 1, "upsample"](ctx)

        print("\n--- Up Block 3 (128->128, 128x128) ---")
        # ResnetBlock: 128 -> 128, 3x3, pad=1
        test_flux_conv_layer[1, 128, 128, 128, 3, 3, 128, 1, "resnet_conv"](ctx)

        print()
        print("=" * 60)
        print("ALL FLUX VAE CONV2D TESTS PASSED!")
        print("=" * 60)
