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
"""Tests for grouped convolution (num_groups > 1) on GPU.

Verifies that the naive GPU conv2d kernel correctly restricts channel
iteration to group-local channels when num_groups > 1.
"""

from std.random import rand
from std.testing import assert_false

from layout import Idx, TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.conv.conv import Naive2dConvolution, conv_gpu
from std.utils.index import Index, IndexList


def test_grouped_conv2d[
    N: Int,
    H: Int,
    W: Int,
    C_in: Int,
    R: Int,
    S: Int,
    C_out: Int,
    num_groups: Int,
    pad: Int,
    name: StringLiteral,
](ctx: DeviceContext) raises:
    """Test grouped conv2d on GPU against CPU reference."""
    comptime C_per_group = C_in // num_groups
    comptime H_out = H + 2 * pad - R + 1
    comptime W_out = W + 2 * pad - S + 1

    comptime dtype = DType.float32

    comptime in_size = N * H * W * C_in
    comptime filter_size = R * S * C_per_group * C_out
    comptime out_size = N * H_out * W_out * C_out

    print(
        "  ",
        name,
        ": N=",
        N,
        " ",
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
        " groups=",
        num_groups,
        " pad=",
        pad,
        ")",
        sep="",
    )

    # Host memory
    var input_host = ctx.enqueue_create_host_buffer[dtype](in_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var out_gpu_host = ctx.enqueue_create_host_buffer[dtype](out_size)
    var out_ref_host = ctx.enqueue_create_host_buffer[dtype](out_size)

    rand(input_host.as_span())
    rand(filter_host.as_span())

    # CPU reference (Naive2dConvolution takes 5D NDHWC shapes with D=1)
    Naive2dConvolution[dtype, dtype, dtype].run(
        out_ref_host.unsafe_ptr(),
        input_host.unsafe_ptr(),
        filter_host.unsafe_ptr(),
        Index(N, 1, H_out, W_out, C_out),
        Index(N, 1, H, W, C_in),
        Index(1, R, S, C_per_group, C_out),
        IndexList[2](0, 0),  # pad_d
        IndexList[2](pad, pad),  # pad_h
        IndexList[2](pad, pad),  # pad_w
        IndexList[3](1, 1, 1),  # stride
        IndexList[3](1, 1, 1),  # dilation
        num_groups,
    )

    # Device memory
    var input_dev = ctx.enqueue_create_buffer[dtype](in_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var out_dev = ctx.enqueue_create_buffer[dtype](out_size)

    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)

    comptime input_tt_layout = row_major((Idx[N], Idx[H], Idx[W], Idx[C_in]))
    comptime filter_tt_layout = row_major(
        (Idx[R], Idx[S], Idx[C_per_group], Idx[C_out])
    )
    comptime output_tt_layout = row_major(
        (Idx[N], Idx[H_out], Idx[W_out], Idx[C_out])
    )
    var input_tt = TileTensor(input_dev, input_tt_layout)
    var filter_tt = TileTensor(filter_dev, filter_tt_layout)
    var out_tt = TileTensor(out_dev, output_tt_layout)

    # Run GPU conv with groups
    conv_gpu[
        dtype,
        dtype,
        dtype,
    ](
        input_tt,
        filter_tt,
        out_tt,
        IndexList[2](1, 1),  # stride
        IndexList[2](1, 1),  # dilation
        IndexList[4](pad, pad, pad, pad),
        num_groups,
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(out_gpu_host, out_dev)
    ctx.synchronize()

    # Compare
    var max_diff: Float32 = 0.0
    var errors = 0
    for i in range(out_size):
        var gpu_val = out_gpu_host[i].cast[.float32]()
        var ref_val = out_ref_host[i].cast[.float32]()
        var diff = abs(gpu_val - ref_val)
        if diff > max_diff:
            max_diff = diff
        var scale = max(abs(ref_val), Float32(1e-6))
        if diff / scale > 0.01:
            errors += 1

    if errors > 0:
        print("    FAILED: ", errors, " errors, max_diff=", max_diff)
    else:
        print("    PASSED (max_diff=", max_diff, ")")

    assert_false(errors > 0, "Grouped conv2d GPU output mismatch")

    # Cleanup
    _ = input_dev^
    _ = filter_dev^
    _ = out_dev^


def main() raises:
    print("=" * 60)
    print("Grouped Convolution GPU Test")
    print("=" * 60)

    with DeviceContext() as ctx:
        print("\n--- 2D Grouped Conv (groups > 1) ---")

        # Basic: 2 groups, small
        test_grouped_conv2d[1, 8, 8, 4, 3, 3, 4, 2, 1, "basic_2groups"](ctx)

        # 4 groups
        test_grouped_conv2d[1, 8, 8, 8, 3, 3, 8, 4, 1, "4groups"](ctx)

        # Depthwise (groups == C_in == C_out)
        test_grouped_conv2d[1, 8, 8, 16, 3, 3, 16, 16, 1, "depthwise"](ctx)

        # HuBERT-like: 768 channels, 16 groups, kernel 128
        # (This is the shape from the bug report)
        test_grouped_conv2d[1, 1, 400, 768, 1, 128, 768, 16, 0, "hubert_like"](
            ctx
        )

        # Larger batch
        test_grouped_conv2d[2, 16, 16, 32, 3, 3, 64, 8, 1, "batch2_8groups"](
            ctx
        )

        # groups=1 sanity check (should still work)
        test_grouped_conv2d[1, 8, 8, 4, 3, 3, 8, 1, 1, "groups1_sanity"](ctx)

        print()
        print("=" * 60)
        print("ALL GROUPED CONV TESTS PASSED!")
        print("=" * 60)
