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

from std.math import exp

from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout._fillers import random
from state_space.causal_conv1d import (
    causal_conv1d_update_cpu,
    causal_conv1d_update_cpu_no_bias,
)
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


@always_inline
def silu_ref[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Reference SiLU implementation: x * sigmoid(x) = x / (1 + exp(-x))."""
    var x_f32 = x.cast[.float32]()
    var neg_x = -x_f32
    var exp_neg_x = exp(neg_x)
    var one = Float32(1.0)
    var sigmoid_x = one / (one + exp_neg_x)
    return (x_f32 * sigmoid_x).cast[dtype]()


def run_causal_conv1d_update[
    dtype: DType,
    has_bias: Bool,
    activation: StaticString,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    state_len: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test causal conv1d update kernel against reference implementation."""
    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    # Input x: (B, C, L)
    var input_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var input_h = LayoutTensor[dtype, layout_3d, _](
        input_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    # Conv state: (B, C, S)
    var conv_state_heap = List(
        length=batch * dim * state_len, fill=Scalar[dtype](0)
    )
    var conv_state_h = LayoutTensor[dtype, layout_3d, _](
        conv_state_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    )

    # Weight: (C, W)
    var weight_heap = List(length=dim * width, fill=Scalar[dtype](0))
    var weight_h = LayoutTensor[dtype, layout_2d, _](
        weight_heap, RuntimeLayout[layout_2d].row_major(Index(dim, width))
    )

    # Bias: (C,)
    var bias_heap = List(length=dim, fill=Scalar[dtype](0))
    var bias_h = LayoutTensor[dtype, layout_1d, _](
        bias_heap, RuntimeLayout[layout_1d].row_major(Index(dim))
    )

    # Output: (B, C, L)
    var result_fused_heap = List(
        length=batch * dim * seqlen, fill=Scalar[dtype](0)
    )
    var result_fused_h = LayoutTensor[dtype, layout_3d, _](
        result_fused_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    var result_unfused_heap = List(
        length=batch * dim * seqlen, fill=Scalar[dtype](0)
    )
    var result_unfused_h = LayoutTensor[dtype, layout_3d, _](
        result_unfused_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    # Copy of conv_state for reference implementation
    var conv_state_ref_heap = List(
        length=batch * dim * state_len, fill=Scalar[dtype](0)
    )
    var conv_state_ref_h = LayoutTensor[dtype, layout_3d, _](
        conv_state_ref_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    )

    # Initialize input data
    random(input_h)
    random(conv_state_h)
    random(weight_h)
    random(bias_h)

    # Copy conv_state for reference
    for i in range(batch * dim * state_len):
        conv_state_ref_h.ptr[i] = conv_state_h.ptr[i]

    # Create TileTensor versions for kernel call
    var input_tt = TileTensor(input_heap, row_major(batch, dim, seqlen))
    var conv_state_tt = TileTensor(
        conv_state_heap,
        row_major(batch, dim, state_len),
    )
    var weight_tt = TileTensor(weight_heap, row_major(dim, width))
    var bias_tt = TileTensor(
        bias_heap,
        row_major(
            dim,
        ),
    )
    var result_fused_tt = TileTensor(
        result_fused_heap,
        row_major(batch, dim, seqlen),
    )

    var input_buf = input_h
    var weight_buf = weight_h
    var bias_buf = bias_h
    var result_unfused_buf = result_unfused_h
    var conv_state_ref_buf = conv_state_ref_h

    # Strides for channel-first layout (B, C, L)
    var x_batch_stride: UInt32 = UInt32(dim * seqlen)
    var x_c_stride: UInt32 = UInt32(seqlen)
    var x_l_stride: UInt32 = 1

    var conv_state_batch_stride: UInt32 = UInt32(dim * state_len)
    var conv_state_c_stride: UInt32 = UInt32(state_len)
    var conv_state_l_stride: UInt32 = 1

    var weight_c_stride: UInt32 = UInt32(width)
    var weight_width_stride: UInt32 = 1

    var out_batch_stride: UInt32 = UInt32(dim * seqlen)
    var out_c_stride: UInt32 = UInt32(seqlen)
    var out_l_stride: UInt32 = 1

    var silu_activation = activation == "silu"

    # Test fused kernel
    if has_bias:
        causal_conv1d_update_cpu[
            dtype,
            dtype,
            dtype,
            dtype,
            dtype,
        ](
            batch,
            dim,
            seqlen,
            width,
            state_len,
            input_tt,
            conv_state_tt,
            weight_tt,
            result_fused_tt,
            bias_tt,
            x_batch_stride,
            x_c_stride,
            x_l_stride,
            conv_state_batch_stride,
            conv_state_c_stride,
            conv_state_l_stride,
            weight_c_stride,
            weight_width_stride,
            out_batch_stride,
            out_c_stride,
            out_l_stride,
            silu_activation,
        )
    else:
        causal_conv1d_update_cpu_no_bias[
            dtype,
            dtype,
            dtype,
            dtype,
        ](
            batch,
            dim,
            seqlen,
            width,
            state_len,
            input_tt,
            conv_state_tt,
            weight_tt,
            result_fused_tt,
            x_batch_stride,
            x_c_stride,
            x_l_stride,
            conv_state_batch_stride,
            conv_state_c_stride,
            conv_state_l_stride,
            weight_c_stride,
            weight_width_stride,
            out_batch_stride,
            out_c_stride,
            out_l_stride,
            silu_activation,
        )

    # Reference implementation
    var width_minus_1: Int = width - 1

    for b in range(batch):
        for c in range(dim):
            var cur_bias: Scalar[dtype] = Scalar[dtype](0.0)
            if has_bias:
                cur_bias = bias_buf.ptr.load(c)

            # Process each position in the input sequence
            for l in range(seqlen):
                var conv_sum: Scalar[dtype] = cur_bias

                for w in range(width):
                    # Position in the virtual concatenated sequence [conv_state, x]
                    var src_pos = state_len + l - (width_minus_1 - w)
                    var input_val: Scalar[dtype] = Scalar[dtype](0.0)

                    if src_pos >= state_len:
                        # Read from x
                        var x_l_pos = src_pos - state_len
                        var x_offset = (
                            UInt32(b) * x_batch_stride
                            + UInt32(c) * x_c_stride
                            + UInt32(x_l_pos) * x_l_stride
                        )
                        input_val = input_buf.ptr.load(x_offset)
                    elif src_pos >= 0:
                        # Read from conv_state
                        var conv_state_offset = (
                            UInt32(b) * conv_state_batch_stride
                            + UInt32(c) * conv_state_c_stride
                            + UInt32(src_pos) * conv_state_l_stride
                        )
                        input_val = conv_state_ref_buf.ptr.load(
                            conv_state_offset
                        )
                    # else: src_pos < 0, treat as 0 (zero padding)

                    var weight_offset = (
                        UInt32(c) * weight_c_stride
                        + UInt32(w) * weight_width_stride
                    )
                    var weight_val = weight_buf.ptr.load(weight_offset)
                    conv_sum = conv_sum + input_val * weight_val

                # Write output
                var out_offset = (
                    UInt32(b) * out_batch_stride
                    + UInt32(c) * out_c_stride
                    + UInt32(l) * out_l_stride
                )
                var out_val = conv_sum
                if silu_activation:
                    out_val = silu_ref[dtype](out_val)
                result_unfused_buf.ptr.store(out_offset, out_val)

            # Update conv_state: shift old values and add new x values
            if seqlen >= state_len:
                # x is longer than state, just copy last state_len values from x
                for s in range(state_len):
                    var x_l_pos = seqlen - state_len + s
                    var x_offset = (
                        UInt32(b) * x_batch_stride
                        + UInt32(c) * x_c_stride
                        + UInt32(x_l_pos) * x_l_stride
                    )
                    var x_val = input_buf.ptr.load(x_offset)
                    var conv_state_offset = (
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32(s) * conv_state_l_stride
                    )
                    conv_state_ref_buf.ptr.store(conv_state_offset, x_val)
            else:
                # Shift conv_state left by seqlen positions, then append x
                for s in range(state_len - seqlen):
                    var src_offset = (
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32((s + seqlen)) * conv_state_l_stride
                    )
                    var dst_offset = (
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32(s) * conv_state_l_stride
                    )
                    var val = conv_state_ref_buf.ptr.load(src_offset)
                    conv_state_ref_buf.ptr.store(dst_offset, val)

                # Copy x values to the end
                for l in range(seqlen):
                    var x_offset = (
                        UInt32(b) * x_batch_stride
                        + UInt32(c) * x_c_stride
                        + UInt32(l) * x_l_stride
                    )
                    var x_val = input_buf.ptr.load(x_offset)
                    var conv_state_offset = (
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32((state_len - seqlen + l)) * conv_state_l_stride
                    )
                    conv_state_ref_buf.ptr.store(conv_state_offset, x_val)

    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            result_fused_h.ptr[i],
            result_unfused_h.ptr[i],
            rtol=rtol,
        )

    # Compare conv_state updates
    var conv_state_size = batch * dim * state_len
    for i in range(conv_state_size):
        assert_almost_equal(
            conv_state_h.ptr[i],
            conv_state_ref_h.ptr[i],
            rtol=rtol,
        )


def test_basic_causal_conv1d_update() raises:
    """Test basic causal conv1d update with bias."""
    run_causal_conv1d_update[.float32, True, "none"](2, 4, 1, 3, 4)


def test_causal_conv1d_update_with_silu() raises:
    """Test causal conv1d update with SiLU activation."""
    run_causal_conv1d_update[.float32, True, "silu"](2, 4, 1, 3, 4)


def test_causal_conv1d_update_without_bias() raises:
    """Test causal conv1d update without bias."""
    run_causal_conv1d_update[.float32, False, "none"](2, 4, 1, 3, 4)


def test_causal_conv1d_update_seqlen_greater_than_one() raises:
    """Test causal conv1d update with seqlen > 1."""
    run_causal_conv1d_update[.float32, True, "none"](2, 4, 4, 3, 4)


def test_causal_conv1d_update_various_widths() raises:
    """Test causal conv1d update with various kernel widths."""
    run_causal_conv1d_update[.float32, True, "none"](2, 4, 1, 2, 3)
    run_causal_conv1d_update[.float32, True, "none"](2, 4, 1, 3, 4)
    run_causal_conv1d_update[.float32, True, "none"](2, 4, 1, 4, 5)


def test_causal_conv1d_update_larger_state() raises:
    """Test causal conv1d update with larger state length."""
    run_causal_conv1d_update[.float32, True, "none"](2, 8, 1, 3, 8)


def test_causal_conv1d_update_combinations() raises:
    """Test causal conv1d update with various bias and activation combinations.
    """
    run_causal_conv1d_update[.float32, False, "none"](2, 4, 1, 3, 4)
    run_causal_conv1d_update[.float32, True, "none"](2, 4, 1, 3, 4)
    run_causal_conv1d_update[.float32, False, "silu"](2, 4, 1, 3, 4)
    run_causal_conv1d_update[.float32, True, "silu"](2, 4, 1, 3, 4)
