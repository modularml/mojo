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
"""Core Causal Conv1D Kernel Implementations.

This module provides CPU and GPU kernel implementations for causal 1D convolution,
supporting both channel-first and channel-last memory layouts.

Causal Convolution:
    In causal convolution, the output at position `i` depends only on inputs at
    positions `[i - width + 1, i]`. This ensures no information leakage from
    future positions, making it suitable for autoregressive sequence modeling.

    Mathematical form for width=4:
        out[i] = sum(x[i-3:i+1] * w[0:4]) + bias  (with boundary handling)

Kernel Categories:

    1. Forward Kernels (CPU & GPU):
        - `causal_conv1d_channel_first_fwd_cpu[_no_bias]`
        - `causal_conv1d_channel_last_fwd_cpu[_with_seq_idx][_no_bias]`
        - `causal_conv1d_channel_first_fwd_gpu[_with_seq_idx][_no_bias]`
        - `causal_conv1d_channel_last_fwd_gpu[_with_seq_idx][_no_bias]`

        SIMD-vectorized implementations with compile-time width specialization.
        Supported widths: 1, 2, 3, 4.

    2. Update Kernels (for autoregressive decode):
        - `causal_conv1d_update_cpu[_no_bias]`
        - `causal_conv1d_update_gpu[_no_bias]`

        Incremental update operations that maintain conv state for efficient
        autoregressive token generation.

Memory Layouts:
    - Channel-first (B, C, L): Standard layout, contiguous channels per position.
    - Channel-last (B, L, C): Contiguous positions per channel, used by some frameworks.

GPU Optimization Parameters:
    - kNThreads=128: Threads per block for sequence processing
    - kNElts=4: Elements processed per thread for better ILP
    - SIMD width 4: Vectorized weight loading for width=4 kernels

Activation Support:
    - None: Direct convolution output
    - SiLU: Sigmoid Linear Unit activation (x * sigmoid(x))
"""

from max.algorithm import sync_parallelize
from max.gpu.host import DeviceContext
from max.gpu import (
    block_dim,
    block_idx,
    thread_idx,
)
from layout import TensorLayout, TensorEngine, TileTensor
from nn.activations import silu


# ===----------------------------------------------------------------------=== #
# CPU Implementations
# ===----------------------------------------------------------------------=== #


def causal_conv1d_channel_first_fwd_cpu[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[mut=False, x_dtype, ...],  # Shape (B, C, L)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # Shape (B, C, L)
    bias: TileTensor[mut=False, bias_dtype, ...],  # Shape (C,), stride = 1
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    bias_stride: UInt32,
    silu_activation: Bool,
    ctx: Optional[DeviceContext] = None,
):
    """CPU implementation of causal conv1d for channel-first layout with bias.

    Optimizations:
    1. Parallelization across batch*channel dimensions using sync_parallelize.
    2. Pre-loaded weights in registers to reduce memory access.

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.
        bias_dtype: Element type of the bias tensor `bias`.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length.
        width: Kernel width.
        x: Input tensor of shape (B, C, L).
        weight: Weight tensor of shape (C, W).
        output: Output tensor of shape (B, C, L).
        bias: Bias tensor of shape (C,).
        x_batch_stride: Stride for the batch dimension of the input tensor.
        x_c_stride: Stride for the channel dimension of the input tensor.
        x_l_stride: Stride for the sequence length dimension of the input tensor.
        weight_c_stride: Stride for the channel dimension of the weight tensor.
        weight_width_stride: Stride for the width dimension of the weight tensor.
        out_batch_stride: Stride for the batch dimension of the output tensor.
        out_c_stride: Stride for the channel dimension of the output tensor.
        out_l_stride: Stride for the sequence length dimension of the output tensor.
        bias_stride: Stride for the bias tensor.
        silu_activation: Whether to apply SiLU activation.
        ctx: The context to execute the work on.
    """
    var width_minus_1: Int = width - 1
    var total_bc = batch * dim

    # Parallelize across batch*channel combinations
    def process_bc(bc_idx: Int) {imm}:
        var b, c = divmod(bc_idx, dim)

        # Bounds checking
        if b >= batch or c >= dim:
            return

        # Validate bias tensor has valid dimensions (use debug_assert since we can't raise in @__parameter fn)
        assert (
            Int(bias.dim[0]()) > 0
        ), "Bias tensor must have at least one element"
        assert c < Int(
            bias.dim[0]()
        ), "Channel index out of bounds for bias tensor"

        var weight_c_base_offset = UInt32(c * Int(weight_c_stride))
        var bias_offset = UInt32(c * Int(bias_stride))
        var cur_bias: Scalar[output_dtype] = Scalar[output_dtype](
            bias.raw_load(bias_offset)
        )

        # Pre-load weights for this channel to reduce memory access
        var w0: Scalar[weight_dtype] = 0
        var w1: Scalar[weight_dtype] = 0
        var w2: Scalar[weight_dtype] = 0
        var w3: Scalar[weight_dtype] = 0
        if width >= 1:
            w0 = Scalar[weight_dtype](weight.raw_load(weight_c_base_offset))
        if width >= 2:
            w1 = Scalar[weight_dtype](
                weight.raw_load(
                    weight_c_base_offset + UInt32(Int(weight_width_stride))
                )
            )
        if width >= 3:
            w2 = Scalar[weight_dtype](
                weight.raw_load(
                    weight_c_base_offset + UInt32(2 * Int(weight_width_stride))
                )
            )
        if width >= 4:
            w3 = Scalar[weight_dtype](
                weight.raw_load(
                    weight_c_base_offset + UInt32(3 * Int(weight_width_stride))
                )
            )

        var x_base = UInt32(b * Int(x_batch_stride) + c * Int(x_c_stride))
        var out_base = UInt32(b * Int(out_batch_stride) + c * Int(out_c_stride))

        # Process all sequence positions
        for l in range(seqlen):
            var conv_sum: Scalar[output_dtype] = cur_bias

            for w in range(width):
                var input_l: Int = l - (width_minus_1 - w)
                if input_l >= 0:
                    var x_offset: UInt32 = x_base + UInt32(
                        UInt32(input_l) * x_l_stride
                    )
                    var input_val: Scalar[x_dtype] = x.raw_load(x_offset)
                    # Select weight based on position
                    var weight_val: Scalar[weight_dtype] = w0 if w == 0 else (
                        w1 if w == 1 else (w2 if w == 2 else w3)
                    )
                    conv_sum = conv_sum + Scalar[output_dtype](
                        input_val * Scalar[x_dtype](weight_val)
                    )

            var out_offset: UInt32 = out_base + UInt32(UInt32(l) * out_l_stride)
            var out_val: Scalar[output_dtype] = conv_sum
            if silu_activation:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            output.raw_store(out_offset, out_val)

    sync_parallelize(process_bc, total_bc, ctx)


def causal_conv1d_channel_first_fwd_cpu_no_bias[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[mut=False, x_dtype, ...],  # Shape (B, C, L)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # Shape (B, C, L)
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    silu_activation: Bool,
    ctx: Optional[DeviceContext] = None,
):
    """CPU implementation of causal conv1d for channel-first layout without bias.

    Optimizations:
    1. Parallelization across batch*channel dimensions using sync_parallelize.
    2. Pre-loaded weights in registers to reduce memory access.

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length.
        width: Kernel width.
        x: Input tensor of shape (B, C, L).
        weight: Weight tensor of shape (C, W).
        output: Output tensor of shape (B, C, L).
        x_batch_stride: Stride for the batch dimension of the input tensor.
        x_c_stride: Stride for the channel dimension of the input tensor.
        x_l_stride: Stride for the sequence length dimension of the input tensor.
        weight_c_stride: Stride for the channel dimension of the weight tensor.
        weight_width_stride: Stride for the width dimension of the weight tensor.
        out_batch_stride: Stride for the batch dimension of the output tensor.
        out_c_stride: Stride for the channel dimension of the output tensor.
        out_l_stride: Stride for the sequence length dimension of the output tensor.
        silu_activation: Whether to apply SiLU activation.
        ctx: The context to execute the work on.
    """
    var width_minus_1: Int = width - 1
    var total_bc = batch * dim

    # Parallelize across batch*channel combinations
    def process_bc(bc_idx: Int) {imm}:
        var b, c = divmod(bc_idx, dim)

        var weight_c_base_offset = UInt32(UInt32(c) * weight_c_stride)

        # Pre-load weights for this channel to reduce memory access
        var w0: Scalar[weight_dtype] = 0
        var w1: Scalar[weight_dtype] = 0
        var w2: Scalar[weight_dtype] = 0
        var w3: Scalar[weight_dtype] = 0
        if width >= 1:
            w0 = Scalar[weight_dtype](weight.raw_load(weight_c_base_offset))
        if width >= 2:
            w1 = Scalar[weight_dtype](
                weight.raw_load(
                    weight_c_base_offset + UInt32(weight_width_stride)
                )
            )
        if width >= 3:
            w2 = Scalar[weight_dtype](
                weight.raw_load(
                    weight_c_base_offset + UInt32(2 * weight_width_stride)
                )
            )
        if width >= 4:
            w3 = Scalar[weight_dtype](
                weight.raw_load(
                    weight_c_base_offset + UInt32(3 * weight_width_stride)
                )
            )

        var x_base = UInt32(UInt32(b) * x_batch_stride + UInt32(c) * x_c_stride)
        var out_base = UInt32(
            UInt32(b) * out_batch_stride + UInt32(c) * out_c_stride
        )

        # Process all sequence positions
        for l in range(seqlen):
            var conv_sum: Scalar[output_dtype] = 0.0

            for w in range(width):
                var input_l: Int = l - (width_minus_1 - w)
                if input_l >= 0:
                    var x_offset: UInt32 = x_base + UInt32(
                        UInt32(input_l) * x_l_stride
                    )
                    var input_val: Scalar[x_dtype] = x.raw_load(x_offset)
                    # Select weight based on position
                    var weight_val: Scalar[weight_dtype] = w0 if w == 0 else (
                        w1 if w == 1 else (w2 if w == 2 else w3)
                    )
                    conv_sum = conv_sum + Scalar[output_dtype](
                        input_val * Scalar[x_dtype](weight_val)
                    )

            var out_offset: UInt32 = out_base + UInt32(UInt32(l) * out_l_stride)
            var out_val: Scalar[output_dtype] = conv_sum
            if silu_activation:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            output.raw_store(out_offset, out_val)

    sync_parallelize(process_bc, total_bc, ctx)


def causal_conv1d_channel_last_fwd_cpu[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[mut=False, x_dtype, ...],  # Shape (B, L, C)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # Shape (B, L, C)
    bias: TileTensor[mut=False, bias_dtype, ...],  # Shape (C,), stride = 1
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    silu_activation: Bool,
):
    """
    Optimized CPU implementation of causal conv1d for channel-last layout with bias.

    Structured for potential SIMD optimizations. Currently similar to naive but
    organized for future vectorization improvements.
    """
    var width_minus_1: Int = width - 1

    for b in range(batch):
        for l in range(seqlen):
            for c in range(dim):
                var conv_sum: Scalar[output_dtype] = Scalar[output_dtype](
                    bias.raw_load(c)
                )

                for w in range(width):
                    var input_l: Int = l - (width_minus_1 - w)
                    if input_l >= 0:
                        var x_offset: UInt32 = (
                            UInt32(b) * x_batch_stride
                            + UInt32(input_l) * x_l_stride
                            + UInt32(c) * x_c_stride
                        )
                        var input_val: Scalar[x.dtype] = x.raw_load(x_offset)
                        var weight_offset: UInt32 = (
                            UInt32(c) * weight_c_stride
                            + UInt32(w) * weight_width_stride
                        )
                        var weight_val: Scalar[weight.dtype] = weight.raw_load(
                            weight_offset
                        )
                        conv_sum = conv_sum + Scalar[output_dtype](
                            input_val * Scalar[x.dtype](weight_val)
                        )

                var out_offset: UInt32 = (
                    UInt32(b) * out_batch_stride
                    + UInt32(l) * out_l_stride
                    + UInt32(c) * out_c_stride
                )
                var out_val: Scalar[output_dtype] = conv_sum
                if silu_activation:
                    comptime if output_dtype.is_floating_point():
                        out_val = silu(out_val)
                    else:
                        out_val = silu(out_val.cast[.float32]()).cast[
                            output_dtype
                        ]()
                output.raw_store(out_offset, out_val)


def causal_conv1d_channel_last_fwd_cpu_no_bias[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[mut=False, x_dtype, ...],  # Shape (B, L, C)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # Shape (B, L, C)
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    silu_activation: Bool,
):
    """
    Optimized CPU implementation of causal conv1d for channel-last layout without bias.

    Structured for potential SIMD optimizations. Currently similar to naive but
    organized for future vectorization improvements.
    """
    var width_minus_1: Int = width - 1

    for b in range(batch):
        for l in range(seqlen):
            for c in range(dim):
                var conv_sum: Scalar[output_dtype] = 0.0

                for w in range(width):
                    var input_l: Int = l - (width_minus_1 - w)
                    if input_l >= 0:
                        var x_offset: UInt32 = (
                            UInt32(b) * x_batch_stride
                            + UInt32(input_l) * x_l_stride
                            + UInt32(c) * x_c_stride
                        )
                        var input_val: Scalar[x.dtype] = x.raw_load(x_offset)
                        var weight_offset: UInt32 = (
                            UInt32(c) * weight_c_stride
                            + UInt32(w) * weight_width_stride
                        )
                        var weight_val: Scalar[weight.dtype] = weight.raw_load(
                            weight_offset
                        )
                        conv_sum = conv_sum + Scalar[output_dtype](
                            input_val * Scalar[x.dtype](weight_val)
                        )

                var out_offset: UInt32 = (
                    UInt32(b) * out_batch_stride
                    + UInt32(l) * out_l_stride
                    + UInt32(c) * out_c_stride
                )
                var out_val: Scalar[output_dtype] = conv_sum
                if silu_activation:
                    comptime if output_dtype.is_floating_point():
                        out_val = silu(out_val)
                    else:
                        out_val = silu(out_val.cast[.float32]()).cast[
                            output_dtype
                        ]()
                output.raw_store(out_offset, out_val)


def causal_conv1d_channel_last_fwd_cpu_with_seq_idx[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
    seq_idx_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[mut=False, x_dtype, ...],  # Shape (B, L, C)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # Shape (B, L, C)
    bias: TileTensor[mut=False, bias_dtype, ...],  # Shape (C,)
    seq_idx: TileTensor[mut=False, seq_idx_dtype, ...],  # Shape (B, L)
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    seq_idx_batch_stride: UInt32,
    seq_idx_l_stride: UInt32,
    silu_activation: Bool,
):
    """Optimized implementation of causal conv1d for channel last data layout with seq_idx.
    """
    var width_minus_1: Int = width - 1

    for b in range(batch):
        for l in range(seqlen):
            var seq_idx_offset: UInt32 = (
                UInt32(b) * seq_idx_batch_stride + UInt32(l) * seq_idx_l_stride
            )
            var cur_seq_idx_val = seq_idx.raw_load(seq_idx_offset)
            var cur_seq_idx: Int32 = Int32(cur_seq_idx_val)

            for c in range(dim):
                var conv_sum: Scalar[output_dtype] = Scalar[output_dtype](
                    bias.raw_load(c)
                )

                for w in range(width):
                    var input_l: Int = l - (width_minus_1 - w)
                    var valid_seq: Bool = True
                    if input_l >= 0:
                        var input_seq_idx_offset: UInt32 = (
                            UInt32(b) * seq_idx_batch_stride
                            + UInt32(input_l) * seq_idx_l_stride
                        )
                        var input_seq_idx_val = seq_idx.raw_load(
                            input_seq_idx_offset
                        )
                        var input_seq_idx: Int32 = Int32(input_seq_idx_val)
                        if input_seq_idx != cur_seq_idx:
                            valid_seq = False

                    if valid_seq and input_l >= 0:
                        var x_offset: UInt32 = (
                            UInt32(b) * x_batch_stride
                            + UInt32(input_l) * x_l_stride
                            + UInt32(c) * x_c_stride
                        )
                        var input_val: Scalar[x_dtype] = x.raw_load(x_offset)
                        var weight_offset: UInt32 = (
                            UInt32(c) * weight_c_stride
                            + UInt32(w) * weight_width_stride
                        )
                        var weight_val: Scalar[weight_dtype] = weight.raw_load(
                            weight_offset
                        )
                        conv_sum = conv_sum + Scalar[output_dtype](
                            input_val * Scalar[x_dtype](weight_val)
                        )

                var out_offset: UInt32 = (
                    UInt32(b) * out_batch_stride
                    + UInt32(l) * out_l_stride
                    + UInt32(c) * out_c_stride
                )
                var out_val: Scalar[output_dtype] = conv_sum
                if silu_activation:
                    comptime if output_dtype.is_floating_point():
                        out_val = silu(out_val)
                    else:
                        out_val = silu(out_val.cast[.float32]()).cast[
                            output_dtype
                        ]()
                output.raw_store(out_offset, out_val)


def causal_conv1d_channel_last_fwd_cpu_no_bias_with_seq_idx[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    seq_idx_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[mut=False, x_dtype, ...],  # Shape (B, L, C)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # Shape (B, L, C)
    seq_idx: TileTensor[mut=False, seq_idx_dtype, ...],  # Shape (B, L)
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    seq_idx_batch_stride: UInt32,
    seq_idx_l_stride: UInt32,
    silu_activation: Bool,
):
    """Optimized implementation of causal conv1d for channel last data layout without bias but with seq_idx.
    """
    var width_minus_1: Int = width - 1

    for b in range(batch):
        for l in range(seqlen):
            var seq_idx_offset: UInt32 = (
                UInt32(b) * seq_idx_batch_stride + UInt32(l) * seq_idx_l_stride
            )
            var cur_seq_idx_val = seq_idx.raw_load(seq_idx_offset)
            var cur_seq_idx: Int32 = Int32(cur_seq_idx_val)

            for c in range(dim):
                var conv_sum: Scalar[output_dtype] = 0.0

                for w in range(width):
                    var input_l: Int = l - (width_minus_1 - w)
                    var valid_seq: Bool = True
                    if input_l >= 0:
                        var input_seq_idx_offset: UInt32 = (
                            UInt32(b) * seq_idx_batch_stride
                            + UInt32(input_l) * seq_idx_l_stride
                        )
                        var input_seq_idx_val = seq_idx.raw_load(
                            input_seq_idx_offset
                        )
                        var input_seq_idx: Int32 = Int32(input_seq_idx_val)
                        if input_seq_idx != cur_seq_idx:
                            valid_seq = False

                    if valid_seq and input_l >= 0:
                        var x_offset: UInt32 = (
                            UInt32(b) * x_batch_stride
                            + UInt32(input_l) * x_l_stride
                            + UInt32(c) * x_c_stride
                        )
                        var input_val: Scalar[x_dtype] = x.raw_load(x_offset)
                        var weight_offset: UInt32 = (
                            UInt32(c) * weight_c_stride
                            + UInt32(w) * weight_width_stride
                        )
                        var weight_val: Scalar[weight_dtype] = weight.raw_load(
                            weight_offset
                        )
                        conv_sum = conv_sum + Scalar[output_dtype](
                            input_val * Scalar[x_dtype](weight_val)
                        )

                var out_offset: UInt32 = (
                    UInt32(b) * out_batch_stride
                    + UInt32(l) * out_l_stride
                    + UInt32(c) * out_c_stride
                )
                var out_val: Scalar[output_dtype] = conv_sum
                if silu_activation:
                    comptime if output_dtype.is_floating_point():
                        out_val = silu(out_val)
                    else:
                        out_val = silu(out_val.cast[.float32]()).cast[
                            output_dtype
                        ]()
                output.raw_store(out_offset, out_val)


# ===----------------------------------------------------------------------=== #
# GPU Implementations
# ===----------------------------------------------------------------------=== #


def causal_conv1d_channel_first_fwd_gpu[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    bias_dtype: DType,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    bias_LT: TensorLayout,
    x_engine: TensorEngine,
    weight_engine: TensorEngine,
    output_engine: TensorEngine,
    bias_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[
        x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine
    ],  # Shape (B, C, L)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],  # Shape (B, C, L)
    bias: TileTensor[
        bias_dtype, bias_LT, MutUntrackedOrigin, Engine=bias_engine
    ],  # Shape (C,), stride = 1
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    bias_stride: UInt32,
    silu_activation: Int8,
):
    """Optimized GPU implementation of causal conv1d for channel-first layout with bias.

    Key optimizations:
    1. SIMD vectorization for input/output operations (kNElts elements per thread).
    2. Efficient memory access patterns with coalesced loads.
    3. Vectorized weight loading and computation for width=2 and width=4.
    4. Optimized activation function with SIMD operations.
    5. Better thread utilization and memory bandwidth usage.

    Grid: (ceildiv(seqlen, kNThreads * kNElts), dim, batch)
    Block: kNThreads

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.
        kNThreads: Number of threads per block used to process the sequence
            dimension.
        kWidth: Compile-time convolution kernel width; must match the runtime
            `width` argument.
        kNElts: Number of sequence elements each thread processes, used for
            SIMD vectorization and ILP.
        bias_dtype: Element type of the bias tensor `bias`.
        x_LT: TensorLayout of the input tensor `x`.
        weight_LT: TensorLayout of the weight tensor `weight`.
        output_LT: TensorLayout of the output tensor `output`.
        bias_LT: TensorLayout of the bias tensor `bias`.
        x_engine: Engine of the input tensor `x`.
        weight_engine: Engine of the weight tensor `weight`.
        output_engine: Engine of the output tensor `output`.
        bias_engine: Engine of the bias tensor `bias`.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length.
        width: Kernel width (must match kWidth compile-time parameter).
        x: Input tensor of shape (B, C, L).
        weight: Weight tensor of shape (C, W).
        output: Output tensor of shape (B, C, L).
        bias: Bias tensor of shape (C,).
        x_batch_stride: Stride for the batch dimension of the input tensor.
        x_c_stride: Stride for the channel dimension of the input tensor.
        x_l_stride: Stride for the sequence length dimension of the input tensor.
        weight_c_stride: Stride for the channel dimension of the weight tensor.
        weight_width_stride: Stride for the width dimension of the weight tensor.
        out_batch_stride: Stride for the batch dimension of the output tensor.
        out_c_stride: Stride for the channel dimension of the output tensor.
        out_l_stride: Stride for the sequence length dimension of the output tensor.
        bias_stride: Stride for the channel dimension of the bias tensor.
        silu_activation: Whether to apply SiLU activation (Int8: 0 or 1).
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = Int(x.dim[0]())
    var nChannels: Int = Int(x.dim[1]())
    var nSeqLen: Int = Int(x.dim[2]())

    if batch_id >= nBatches or channel_id >= nChannels or kWidth != _width:
        return

    # Safety check for null pointers
    if (
        Int(x.ptr) == 0
        or Int(output.ptr) == 0
        or Int(weight.ptr) == 0
        or Int(bias.ptr) == 0
    ):
        return

    # Safety check for bias dimension - if bias is empty or channel_id is out of bounds, use zero bias
    var bias_dim = Int(bias.dim[0]())
    var cur_bias: Scalar[x_dtype] = 0
    if bias_dim > 0 and channel_id < bias_dim:
        var bias_offset = UInt32(channel_id) * bias_stride
        cur_bias = Scalar[x_dtype](bias.raw_load(bias_offset))

    var out_vals: SIMD[output_dtype, kNElts] = 0
    var prev_input_chunk: SIMD[x_dtype, kNElts]
    var input_chunk: SIMD[x_dtype, kNElts]

    # For _width 3, we need to use scalars instead of SIMD (SIMD requires power-of-2 widths)
    # Declare variables for both cases - only one will be used based on kWidth
    var W_2: SIMD[x_dtype, 2] = 0
    var W_4: SIMD[x_dtype, 4] = 0
    var w0: Scalar[x_dtype] = 0
    var w1: Scalar[x_dtype] = 0
    var w2: Scalar[x_dtype] = 0
    var w_single: Scalar[x_dtype] = 0  # For _width 1

    var weight_c_base: UInt32 = UInt32(channel_id) * weight_c_stride
    if kWidth == 1:
        w_single = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(0) * weight_width_stride)
        )
    elif kWidth == 2:
        var w0_val = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(0) * weight_width_stride)
        )
        var w1_val = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(1) * weight_width_stride)
        )
        W_2 = SIMD[x_dtype, 2](w0_val, w1_val)
    elif kWidth == 4:
        var w0_val = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(0) * weight_width_stride)
        )
        var w1_val = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(1) * weight_width_stride)
        )
        var w2_val = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(2) * weight_width_stride)
        )
        var w3_val = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(3) * weight_width_stride)
        )
        W_4 = SIMD[x_dtype, 4](w0_val, w1_val, w2_val, w3_val)
    else:
        w0 = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(0) * weight_width_stride)
        )
        w1 = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(1) * weight_width_stride)
        )
        w2 = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(2) * weight_width_stride)
        )

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return
    var prev_chunk_col: Int = (seq_start - 1) // kNElts
    prev_input_chunk = 0
    if prev_chunk_col >= 0 and prev_chunk_col * kNElts < nSeqLen:
        comptime for i in range(kNElts):
            var prev_seq_idx: Int = prev_chunk_col * kNElts + i
            if prev_seq_idx >= 0 and prev_seq_idx < nSeqLen:
                var prev_offset: UInt32 = (
                    UInt32(batch_id) * x_batch_stride
                    + UInt32(channel_id) * x_c_stride
                    + UInt32(prev_seq_idx) * x_l_stride
                )
                prev_input_chunk[i] = Scalar[x_dtype](x.raw_load(prev_offset))

    var current_chunk_col: Int = seq_start // kNElts
    input_chunk = 0
    if current_chunk_col >= 0 and current_chunk_col * kNElts < nSeqLen:
        comptime for i in range(kNElts):
            var curr_seq_idx: Int = current_chunk_col * kNElts + i
            if curr_seq_idx < nSeqLen:
                var curr_offset: UInt32 = (
                    UInt32(batch_id) * x_batch_stride
                    + UInt32(channel_id) * x_c_stride
                    + UInt32(curr_seq_idx) * x_l_stride
                )
                input_chunk[i] = Scalar[x_dtype](x.raw_load(curr_offset))

    _ = prev_input_chunk.join(input_chunk)
    var silu_active = Bool(Int(silu_activation) != 0)

    comptime for i in range(kNElts):
        var seq_idx: Int = seq_start + i
        if seq_idx >= seq_end:
            break

        # Build input window by loading directly from memory
        # This avoids SIMD slice issues while maintaining correctness
        var conv_result: Scalar[x_dtype] = 0

        comptime if kWidth == 1:
            if seq_idx >= 0 and seq_idx < nSeqLen:
                var load_offset: UInt32 = (
                    UInt32(batch_id) * x_batch_stride
                    + UInt32(channel_id) * x_c_stride
                    + UInt32(seq_idx) * x_l_stride
                )
                var x_val = Scalar[x_dtype](x.raw_load(load_offset))
                conv_result = w_single * x_val
        elif kWidth == 2:
            var input_window: SIMD[x_dtype, 2] = 0

            comptime for w in range(2):
                var input_l: Int = seq_idx - (1 - w)
                if input_l >= 0 and input_l < nSeqLen:
                    var load_offset: UInt32 = (
                        UInt32(batch_id) * x_batch_stride
                        + UInt32(channel_id) * x_c_stride
                        + UInt32(input_l) * x_l_stride
                    )
                    input_window[w] = Scalar[x_dtype](x.raw_load(load_offset))
            var tmp: SIMD[x_dtype, 2] = W_2 * input_window
            conv_result = tmp.reduce_add[1]()
        elif kWidth == 4:
            var input_window: SIMD[x_dtype, 4] = 0

            comptime for w in range(4):
                var input_l: Int = seq_idx - (3 - w)
                if input_l >= 0 and input_l < nSeqLen:
                    var load_offset: UInt32 = (
                        UInt32(batch_id) * x_batch_stride
                        + UInt32(channel_id) * x_c_stride
                        + UInt32(input_l) * x_l_stride
                    )
                    input_window[w] = Scalar[x_dtype](x.raw_load(load_offset))
            var tmp: SIMD[x_dtype, 4] = W_4 * input_window
            conv_result = tmp.reduce_add[1]()
        else:
            # kWidth == 3 case
            var x0: Scalar[x_dtype] = 0
            var x1: Scalar[x_dtype] = 0
            var x2: Scalar[x_dtype] = 0
            var input_l0: Int = seq_idx - 2
            var input_l1: Int = seq_idx - 1
            var input_l2: Int = seq_idx
            if input_l0 >= 0 and input_l0 < nSeqLen:
                var load_offset: UInt32 = (
                    UInt32(batch_id) * x_batch_stride
                    + UInt32(channel_id) * x_c_stride
                    + UInt32(input_l0) * x_l_stride
                )
                x0 = Scalar[x_dtype](x.raw_load(load_offset))
            if input_l1 >= 0 and input_l1 < nSeqLen:
                var load_offset: UInt32 = (
                    UInt32(batch_id) * x_batch_stride
                    + UInt32(channel_id) * x_c_stride
                    + UInt32(input_l1) * x_l_stride
                )
                x1 = Scalar[x_dtype](x.raw_load(load_offset))
            if input_l2 >= 0 and input_l2 < nSeqLen:
                var load_offset: UInt32 = (
                    UInt32(batch_id) * x_batch_stride
                    + UInt32(channel_id) * x_c_stride
                    + UInt32(input_l2) * x_l_stride
                )
                x2 = Scalar[x_dtype](x.raw_load(load_offset))
            conv_result = w0 * x0 + w1 * x1 + w2 * x2

        var out_val: Scalar[output_dtype] = Scalar[output_dtype](
            cur_bias
        ) + Scalar[output_dtype](conv_result)
        if silu_active:
            comptime if output_dtype.is_floating_point():
                out_val = silu(out_val)
            else:
                out_val = silu(out_val.cast[.float32]()).cast[output_dtype]()
        out_vals[i] = out_val

    comptime for i in range(kNElts):
        var seq_idx: Int = seq_start + i
        if seq_idx >= seq_end:
            break
        var out_offset: UInt32 = (
            UInt32(batch_id) * out_batch_stride
            + UInt32(channel_id) * out_c_stride
            + UInt32(seq_idx) * out_l_stride
        )
        output.raw_store(out_offset, Scalar[output_dtype](out_vals[i]))


# Optimized GPU version without bias
def causal_conv1d_channel_first_fwd_gpu_no_bias[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin],  # Shape (B, C, L)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin
    ],  # Shape (B, C, L)
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    silu_activation: Int8,
):
    """
    Optimized causal conv1d implementation for channel first data layout using SIMD operations (no bias).

    Key optimizations:
    1. SIMD vectorization for input/output operations
    2. Efficient memory access patterns with coalesced loads
    3. Vectorized weight loading and computation
    4. Optimized activation function with SIMD operations
    5. Better thread utilization and memory bandwidth usage

    Grid: (ceildiv(seqlen, kNThreads * kNElts), dim, batch)
    Block: kNThreads

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.
        kNThreads: Number of threads per block used to process the sequence
            dimension.
        kWidth: Compile-time convolution kernel width; must match the runtime
            `width` argument.
        kNElts: Number of sequence elements each thread processes, used for
            SIMD vectorization and ILP.
        x_LT: TensorLayout of the input tensor `x`.
        weight_LT: TensorLayout of the weight tensor `weight`.
        output_LT: TensorLayout of the output tensor `output`.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length.
        width: Kernel width (must match kWidth compile-time parameter).
        x: Input tensor of shape (B, C, L).
        weight: Weight tensor of shape (C, W).
        output: Output tensor of shape (B, C, L).
        x_batch_stride: Stride for the batch dimension of the input tensor.
        x_c_stride: Stride for the channel dimension of the input tensor.
        x_l_stride: Stride for the sequence length dimension of the input
            tensor.
        weight_c_stride: Stride for the channel dimension of the weight
            tensor.
        weight_width_stride: Stride for the width dimension of the weight
            tensor.
        out_batch_stride: Stride for the batch dimension of the output
            tensor.
        out_c_stride: Stride for the channel dimension of the output tensor.
        out_l_stride: Stride for the sequence length dimension of the output
            tensor.
        silu_activation: Whether to apply SiLU activation (Int8: 0 or 1).
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = Int(x.dim[0]())
    var nChannels: Int = Int(x.dim[1]())
    var nSeqLen: Int = Int(x.dim[2]())

    if batch_id >= nBatches or channel_id >= nChannels:
        return

    var out_vals: SIMD[x_dtype, kNElts] = 0
    var prev_input_chunk: SIMD[x_dtype, kNElts]
    var input_chunk: SIMD[x_dtype, kNElts]

    var W_2: SIMD[x_dtype, 2] = 0
    var W_4: SIMD[x_dtype, 4] = 0
    var w0: Scalar[x_dtype] = 0
    var w1: Scalar[x_dtype] = 0
    var w2: Scalar[x_dtype] = 0
    var w_single: Scalar[x_dtype] = 0

    var weight_c_base: UInt32 = UInt32(channel_id) * weight_c_stride
    if kWidth == 1:
        w_single = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(0) * weight_width_stride)
        )
    elif kWidth == 2:
        var w0_val = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(0) * weight_width_stride)
        )
        var w1_val = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(1) * weight_width_stride)
        )
        W_2 = SIMD[x_dtype, 2](w0_val, w1_val)
    elif kWidth == 4:
        var w0_val = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(0) * weight_width_stride)
        )
        var w1_val = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(1) * weight_width_stride)
        )
        var w2_val = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(2) * weight_width_stride)
        )
        var w3_val = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(3) * weight_width_stride)
        )
        W_4 = SIMD[x_dtype, 4](w0_val, w1_val, w2_val, w3_val)
    else:
        w0 = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(0) * weight_width_stride)
        )
        w1 = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(1) * weight_width_stride)
        )
        w2 = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(2) * weight_width_stride)
        )

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return
    var prev_chunk_col: Int = (seq_start - 1) // kNElts
    prev_input_chunk = 0
    if prev_chunk_col >= 0 and prev_chunk_col * kNElts < nSeqLen:
        comptime for i in range(kNElts):
            var prev_seq_idx: Int = prev_chunk_col * kNElts + i
            if prev_seq_idx >= 0 and prev_seq_idx < nSeqLen:
                var prev_offset: UInt32 = (
                    UInt32(batch_id) * x_batch_stride
                    + UInt32(channel_id) * x_c_stride
                    + UInt32(prev_seq_idx) * x_l_stride
                )
                prev_input_chunk[i] = Scalar[x_dtype](x.raw_load(prev_offset))

    # Load current chunk
    var current_chunk_col: Int = seq_start // kNElts
    input_chunk = 0
    if current_chunk_col >= 0 and current_chunk_col * kNElts < nSeqLen:
        comptime for i in range(kNElts):
            var curr_seq_idx: Int = current_chunk_col * kNElts + i
            if curr_seq_idx < nSeqLen:
                var curr_offset: UInt32 = (
                    UInt32(batch_id) * x_batch_stride
                    + UInt32(channel_id) * x_c_stride
                    + UInt32(curr_seq_idx) * x_l_stride
                )
                input_chunk[i] = Scalar[x_dtype](x.raw_load(curr_offset))

    var silu_active = Bool(Int(silu_activation) != 0)

    comptime for i in range(kNElts):
        var seq_idx: Int = seq_start + i
        if seq_idx >= seq_end:
            break

        # Build input window by loading directly from memory
        var conv_result: Scalar[x_dtype] = 0

        comptime if kWidth == 1:
            if seq_idx >= 0 and seq_idx < nSeqLen:
                var load_offset: UInt32 = (
                    UInt32(batch_id) * x_batch_stride
                    + UInt32(channel_id) * x_c_stride
                    + UInt32(seq_idx) * x_l_stride
                )
                var x_val = Scalar[x_dtype](x.raw_load(load_offset))
                conv_result = w_single * x_val
        elif kWidth == 2:
            var input_window: SIMD[x_dtype, 2] = 0

            comptime for w in range(2):
                var input_l: Int = seq_idx - (1 - w)
                if input_l >= 0 and input_l < nSeqLen:
                    var load_offset: UInt32 = (
                        UInt32(batch_id) * x_batch_stride
                        + UInt32(channel_id) * x_c_stride
                        + UInt32(input_l) * x_l_stride
                    )
                    input_window[w] = Scalar[x_dtype](x.raw_load(load_offset))
            var tmp: SIMD[x_dtype, 2] = W_2 * input_window
            conv_result = tmp.reduce_add[1]()
        elif kWidth == 4:
            var input_window: SIMD[x_dtype, 4] = 0

            comptime for w in range(4):
                var input_l: Int = seq_idx - (3 - w)
                if input_l >= 0 and input_l < nSeqLen:
                    var load_offset: UInt32 = (
                        UInt32(batch_id) * x_batch_stride
                        + UInt32(channel_id) * x_c_stride
                        + UInt32(input_l) * x_l_stride
                    )
                    input_window[w] = Scalar[x_dtype](x.raw_load(load_offset))
            var tmp: SIMD[x_dtype, 4] = W_4 * input_window
            conv_result = tmp.reduce_add[1]()
        else:
            # kWidth == 3 case
            var x0: Scalar[x_dtype] = 0
            var x1: Scalar[x_dtype] = 0
            var x2: Scalar[x_dtype] = 0
            var input_l0: Int = seq_idx - 2
            var input_l1: Int = seq_idx - 1
            var input_l2: Int = seq_idx
            if input_l0 >= 0 and input_l0 < nSeqLen:
                var load_offset: UInt32 = (
                    UInt32(batch_id) * x_batch_stride
                    + UInt32(channel_id) * x_c_stride
                    + UInt32(input_l0) * x_l_stride
                )
                x0 = Scalar[x_dtype](x.raw_load(load_offset))
            if input_l1 >= 0 and input_l1 < nSeqLen:
                var load_offset: UInt32 = (
                    UInt32(batch_id) * x_batch_stride
                    + UInt32(channel_id) * x_c_stride
                    + UInt32(input_l1) * x_l_stride
                )
                x1 = Scalar[x_dtype](x.raw_load(load_offset))
            if input_l2 >= 0 and input_l2 < nSeqLen:
                var load_offset: UInt32 = (
                    UInt32(batch_id) * x_batch_stride
                    + UInt32(channel_id) * x_c_stride
                    + UInt32(input_l2) * x_l_stride
                )
                x2 = Scalar[x_dtype](x.raw_load(load_offset))
            conv_result = w0 * x0 + w1 * x1 + w2 * x2

        var out_val: Scalar[x_dtype] = conv_result
        if silu_active:
            comptime if x_dtype.is_floating_point():
                out_val = silu(out_val)
            else:
                out_val = silu(out_val.cast[.float32]()).cast[x_dtype]()
        out_vals[i] = out_val

    comptime for i in range(kNElts):
        var seq_idx: Int = seq_start + i
        if seq_idx >= seq_end:
            break
        var out_offset: UInt32 = (
            UInt32(batch_id) * out_batch_stride
            + UInt32(channel_id) * out_c_stride
            + UInt32(seq_idx) * out_l_stride
        )
        output.raw_store(out_offset, Scalar[output_dtype](out_vals[i]))


def causal_conv1d_channel_last_fwd_gpu[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    bias_dtype: DType,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    bias_LT: TensorLayout,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin],  # Shape (B, L, C)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin
    ],  # Shape (B, L, C)
    bias: TileTensor[
        bias_dtype, bias_LT, MutUntrackedOrigin
    ],  # Shape (C,), stride = 1
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    silu_activation: Int8,
):
    """
    Optimized causal conv1d implementation for channel last data layout using SIMD operations.

    Key optimizations:
    1. SIMD vectorization for input/output operations across channels
    2. Efficient memory access patterns with coalesced loads using vectorized tensor views
    3. Vectorized weight loading and computation
    4. Chunked processing of multiple sequence positions per thread
    5. Optimized activation function with SIMD operations
    6. Better thread utilization and memory bandwidth usage

    For channel-last layout (B, L, C), we reshape to (B*L, C) to enable vectorized
    operations along channels, and process multiple sequence positions per thread.
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_chunk_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = _batch
    var nSeqLen: Int = _seqlen
    var nChannels: Int = _dim

    if batch_id >= nBatches or kWidth != _width:
        return

    # Work with 3D tensor directly - manually load SIMD vectors using pointer arithmetic
    # For channel-last (B, L, C), we load SIMD vectors along the channel dimension

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return

    var channel_start: Int = channel_chunk_id * kNElts

    if channel_start >= nChannels:
        return

    # Safety check for null pointers
    if (
        Int(x._storage) == 0
        or Int(output._storage) == 0
        or Int(weight._storage) == 0
        or Int(bias._storage) == 0
    ):
        return

    # Safety check for bias tensor dimensions
    var bias_dim = Int(bias.dim[0]())
    if bias_dim == 0:
        return

    for c_offset in range(kNElts):
        var c_idx: Int = channel_start + c_offset
        if c_idx >= nChannels:
            break

        # Safety check for bias dimension
        if c_idx >= bias_dim:
            break

        var cur_bias: Scalar[output_dtype] = Scalar[output_dtype](
            bias.raw_load(c_idx)
        )
        var W = (weight._storage + c_idx * Int(weight_c_stride)).load[
            width=kWidth
        ]()
        var prev_chunk_col: Int = (seq_start - 1) // kNElts
        var prev_input_chunk: SIMD[x_dtype, kNElts] = 0
        if prev_chunk_col >= 0 and prev_chunk_col * kNElts < nSeqLen:
            var prev_row_2d: Int = batch_id * nSeqLen + prev_chunk_col * kNElts
            if prev_row_2d >= 0 and prev_row_2d < nBatches * nSeqLen:
                var prev_batch, prev_seq = divmod(prev_row_2d, nSeqLen)
                if (
                    prev_batch >= 0
                    and prev_batch < nBatches
                    and prev_seq >= 0
                    and prev_seq < nSeqLen
                ):
                    # Load SIMD vector manually using pointer arithmetic
                    prev_input_chunk = 0

                    comptime for i in range(kNElts):
                        var c_idx_load: Int = channel_start + i
                        if c_idx_load < nChannels:
                            var prev_offset: UInt32 = (
                                UInt32(prev_batch) * x_batch_stride
                                + UInt32(prev_seq) * x_l_stride
                                + UInt32(c_idx_load) * x_c_stride
                            )
                            prev_input_chunk[i] = Scalar[x_dtype](
                                x.raw_load(prev_offset)
                            )

        var current_chunk_col: Int = seq_start // kNElts
        var input_chunk: SIMD[x_dtype, kNElts] = 0
        if current_chunk_col >= 0 and current_chunk_col * kNElts < nSeqLen:
            var current_row_2d: Int = (
                batch_id * nSeqLen + current_chunk_col * kNElts
            )
            if current_row_2d >= 0 and current_row_2d < nBatches * nSeqLen:
                var current_batch, current_seq = divmod(current_row_2d, nSeqLen)
                if (
                    current_batch >= 0
                    and current_batch < nBatches
                    and current_seq >= 0
                    and current_seq < nSeqLen
                ):
                    # Load SIMD vector manually using pointer arithmetic
                    input_chunk = 0

                    comptime for i in range(kNElts):
                        var c_idx_load: Int = channel_start + i
                        if c_idx_load < nChannels:
                            var current_offset: UInt32 = (
                                UInt32(current_batch) * x_batch_stride
                                + UInt32(current_seq) * x_l_stride
                                + UInt32(c_idx_load) * x_c_stride
                            )
                            input_chunk[i] = Scalar[x_dtype](
                                x.raw_load(current_offset)
                            )

        _ = prev_input_chunk.join(input_chunk)
        var out_vals_channel: SIMD[output_dtype, kNElts] = 0
        var silu_active = Bool(Int(silu_activation) != 0)

        comptime for i in range(kNElts):
            var seq_idx: Int = seq_start + i
            if seq_idx >= seq_end:
                break

            var conv_sum: Scalar[output_dtype] = cur_bias

            # Build input window by loading directly from memory
            # For channel-last (B, L, C): offset = _batch * x_batch_stride + seq * x_l_stride + channel * x_c_stride
            var input_window: SIMD[x_dtype, kWidth] = 0

            comptime for w in range(kWidth):
                var input_l: Int = seq_idx - (kWidth - 1 - w)
                if input_l >= 0 and input_l < nSeqLen:
                    var load_offset: UInt32 = (
                        UInt32(batch_id) * x_batch_stride
                        + UInt32(input_l) * x_l_stride
                        + UInt32(c_idx) * x_c_stride
                    )
                    input_window[w] = Scalar[x_dtype](x.raw_load(load_offset))

            var tmp = rebind[SIMD[output_dtype, kWidth]](
                input_window * rebind[type_of(input_window)](W)
            )
            conv_sum = conv_sum + tmp.reduce_add[1]()

            var out_val: Scalar[output_dtype] = conv_sum
            if silu_active:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            out_vals_channel[i] = out_val

        comptime for i in range(kNElts):
            var seq_idx: Int = seq_start + i
            if seq_idx >= seq_end:
                break
            var out_offset: UInt32 = (
                UInt32(batch_id) * out_batch_stride
                + UInt32(seq_idx) * out_l_stride
                + UInt32(c_idx) * out_c_stride
            )
            output.raw_store(out_offset, out_vals_channel[i])


# Optimized GPU version without bias for channel last
def causal_conv1d_channel_last_fwd_gpu_no_bias[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin],  # Shape (B, L, C)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin
    ],  # Shape (B, L, C)
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    silu_activation: Int8,
):
    """
    Optimized causal conv1d implementation for channel last data layout using SIMD operations (no bias).

    Key optimizations:
    1. SIMD vectorization for input/output operations across channels
    2. Efficient memory access patterns with coalesced loads
    3. Vectorized weight loading and computation
    4. Optimized activation function with SIMD operations
    5. Better thread utilization and memory bandwidth usage
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_chunk_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = _batch
    var nSeqLen: Int = _seqlen
    var nChannels: Int = _dim

    if batch_id >= nBatches or kWidth != _width:
        return

    # Work with 3D tensor directly - manually load SIMD vectors using pointer arithmetic
    # For channel-last (B, L, C), we load SIMD vectors along the channel dimension

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return

    var channel_start: Int = channel_chunk_id * kNElts

    if channel_start >= nChannels:
        return

    for c_offset in range(kNElts):
        var c_idx: Int = channel_start + c_offset
        if c_idx >= nChannels:
            break

        var W = (weight._storage + c_idx * Int(weight_c_stride)).load[
            width=kWidth
        ]()
        var prev_chunk_col: Int = (seq_start - 1) // kNElts
        var prev_input_chunk: SIMD[x_dtype, kNElts] = 0
        if prev_chunk_col >= 0 and prev_chunk_col * kNElts < nSeqLen:
            var prev_row_2d: Int = batch_id * nSeqLen + prev_chunk_col * kNElts
            if prev_row_2d >= 0 and prev_row_2d < nBatches * nSeqLen:
                var prev_batch, prev_seq = divmod(prev_row_2d, nSeqLen)
                if (
                    prev_batch >= 0
                    and prev_batch < nBatches
                    and prev_seq >= 0
                    and prev_seq < nSeqLen
                ):
                    # Load SIMD vector manually using pointer arithmetic
                    prev_input_chunk = 0

                    comptime for i in range(kNElts):
                        var c_idx_load: Int = channel_start + i
                        if c_idx_load < nChannels:
                            var prev_offset: UInt32 = (
                                UInt32(prev_batch) * x_batch_stride
                                + UInt32(prev_seq) * x_l_stride
                                + UInt32(c_idx_load) * x_c_stride
                            )
                            prev_input_chunk[i] = Scalar[x_dtype](
                                x.raw_load(prev_offset)
                            )

        var current_chunk_col: Int = seq_start // kNElts
        var input_chunk: SIMD[x_dtype, kNElts] = 0
        if current_chunk_col >= 0 and current_chunk_col * kNElts < nSeqLen:
            var current_row_2d: Int = (
                batch_id * nSeqLen + current_chunk_col * kNElts
            )
            if current_row_2d >= 0 and current_row_2d < nBatches * nSeqLen:
                var current_batch, current_seq = divmod(current_row_2d, nSeqLen)
                if (
                    current_batch >= 0
                    and current_batch < nBatches
                    and current_seq >= 0
                    and current_seq < nSeqLen
                ):
                    # Load SIMD vector manually using pointer arithmetic
                    input_chunk = 0

                    comptime for i in range(kNElts):
                        var c_idx_load: Int = channel_start + i
                        if c_idx_load < nChannels:
                            var current_offset: UInt32 = (
                                UInt32(current_batch) * x_batch_stride
                                + UInt32(current_seq) * x_l_stride
                                + UInt32(c_idx_load) * x_c_stride
                            )
                            input_chunk[i] = Scalar[x_dtype](
                                x.raw_load(current_offset)
                            )

        _ = prev_input_chunk.join(input_chunk)
        var out_vals_channel: SIMD[output_dtype, kNElts] = 0
        var silu_active = Bool(Int(silu_activation) != 0)

        comptime for i in range(kNElts):
            var seq_idx: Int = seq_start + i
            if seq_idx >= seq_end:
                break

            var conv_sum: Scalar[output_dtype] = 0.0

            # Build input window by loading directly from memory
            # For channel-last (B, L, C): offset = _batch * x_batch_stride + seq * x_l_stride + channel * x_c_stride
            var input_window: SIMD[x_dtype, kWidth] = 0

            comptime for w in range(kWidth):
                var input_l: Int = seq_idx - (kWidth - 1 - w)
                if input_l >= 0 and input_l < nSeqLen:
                    var load_offset: UInt32 = (
                        UInt32(batch_id) * x_batch_stride
                        + UInt32(input_l) * x_l_stride
                        + UInt32(c_idx) * x_c_stride
                    )
                    input_window[w] = Scalar[x_dtype](x.raw_load(load_offset))

            var tmp = rebind[SIMD[output_dtype, kWidth]](
                input_window * rebind[type_of(input_window)](W)
            )
            conv_sum = conv_sum + tmp.reduce_add[1]()

            var out_val: Scalar[output_dtype] = conv_sum
            if silu_active:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            out_vals_channel[i] = out_val

        comptime for i in range(kNElts):
            var seq_idx: Int = seq_start + i
            if seq_idx >= seq_end:
                break
            var out_offset: UInt32 = (
                UInt32(batch_id) * out_batch_stride
                + UInt32(seq_idx) * out_l_stride
                + UInt32(c_idx) * out_c_stride
            )
            output.raw_store(out_offset, out_vals_channel[i])


# ============================================================================
# Optimized GPU Implementations with seq_idx as TileTensor
# ============================================================================


# Optimized GPU implementation for channel-last with bias and seq_idx as TileTensor
def causal_conv1d_channel_last_fwd_gpu_with_seq_idx[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
    seq_idx_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    bias_LT: TensorLayout,
    seq_idx_LT: TensorLayout,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin],  # Shape (B, L, C)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin
    ],  # Shape (B, L, C)
    bias: TileTensor[
        bias_dtype, bias_LT, MutUntrackedOrigin
    ],  # Shape (C,), stride = 1
    seq_idx: TileTensor[
        seq_idx_dtype, seq_idx_LT, MutUntrackedOrigin
    ],  # Shape (B, L)
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    seq_idx_batch_stride: UInt32,
    seq_idx_l_stride: UInt32,
    silu_activation: Int8,
):
    """
    Optimized causal conv1d implementation for channel last data layout using SIMD operations with seq_idx support.

    Key optimizations:
    1. SIMD vectorization for input/output operations across channels
    2. Efficient memory access patterns with coalesced loads using vectorized tensor views
    3. Vectorized weight loading and computation
    4. Chunked processing of multiple sequence positions per thread
    5. Optimized activation function with SIMD operations
    6. Better thread utilization and memory bandwidth usage
    7. seq_idx support for conditional processing

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.
        bias_dtype: Element type of the bias tensor `bias`.
        seq_idx_dtype: Element type of the `seq_idx` tensor.
        kNThreads: Number of threads per block used to process the sequence
            dimension.
        kWidth: Compile-time convolution kernel width; must match the runtime
            `width` argument.
        kNElts: Number of sequence elements each thread processes, used for
            SIMD vectorization and ILP.
        x_LT: TensorLayout of the input tensor `x`.
        weight_LT: TensorLayout of the weight tensor `weight`.
        output_LT: TensorLayout of the output tensor `output`.
        bias_LT: TensorLayout of the bias tensor `bias`.
        seq_idx_LT: TensorLayout of the `seq_idx` tensor.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length.
        width: Kernel width (must match `kWidth` compile-time parameter).
        x: Input tensor of shape (B, L, C).
        weight: Weight tensor of shape (C, W).
        output: Output tensor of shape (B, L, C).
        bias: Bias tensor of shape (C,).
        seq_idx: Per-position sequence id tensor of shape (B, L); a
            convolution tap at position `input_l` only contributes when its
            sequence id matches the id at the output position.
        x_batch_stride: Stride for the batch dimension of the input tensor.
        x_c_stride: Stride for the channel dimension of the input tensor.
        x_l_stride: Stride for the sequence length dimension of the input
            tensor.
        weight_c_stride: Stride for the channel dimension of the weight
            tensor.
        weight_width_stride: Stride for the width dimension of the weight
            tensor.
        out_batch_stride: Stride for the batch dimension of the output tensor.
        out_c_stride: Stride for the channel dimension of the output tensor.
        out_l_stride: Stride for the sequence length dimension of the output
            tensor.
        seq_idx_batch_stride: Stride for the batch dimension of the `seq_idx`
            tensor.
        seq_idx_l_stride: Stride for the sequence length dimension of the
            `seq_idx` tensor.
        silu_activation: Whether to apply SiLU activation (Int8: 0 or 1).
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_chunk_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = _batch
    var nSeqLen: Int = _seqlen
    var nChannels: Int = _dim

    if batch_id >= nBatches or kWidth != _width:
        return

    # Work with 3D tensor directly - manually load SIMD vectors using pointer arithmetic
    # For channel-last (B, L, C), we load SIMD vectors along the channel dimension
    # Base pointer for batch_id: x.ptr + batch_id * x_batch_stride

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return

    var channel_start: Int = channel_chunk_id * kNElts

    if channel_start >= nChannels:
        return

    # Safety check for null pointers
    if (
        Int(x._storage) == 0
        or Int(output._storage) == 0
        or Int(weight._storage) == 0
        or Int(bias._storage) == 0
    ):
        return

    # Safety check for bias tensor dimensions
    var bias_dim = Int(bias.dim[0]())
    if bias_dim == 0:
        return

    # Helper function to load SIMD vector from 3D tensor at (_batch, seq, channel_start)
    # For channel-last (B, L, C), offset = _batch * x_batch_stride + seq * x_l_stride + channel_start * x_c_stride
    for c_offset in range(kNElts):
        var c_idx: Int = channel_start + c_offset
        if c_idx >= nChannels:
            break

        # Safety check for bias dimension
        if c_idx >= bias_dim:
            break

        # Access bias with proper offset (assuming stride=1 for bias tensor)
        var bias_offset = UInt32(c_idx)
        var cur_bias: Scalar[output_dtype] = Scalar[output_dtype](
            bias.raw_load(bias_offset)
        )

        # Load weights directly from memory to avoid vectorize issues
        # For kWidth == 3, use scalar operations to avoid SIMD issues
        var w0: Scalar[weight_dtype] = 0
        var w1: Scalar[weight_dtype] = 0
        var w2: Scalar[weight_dtype] = 0
        var w3: Scalar[weight_dtype] = 0

        comptime if kWidth >= 1:
            w0 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 0 * weight_width_stride
                )
            )

        comptime if kWidth >= 2:
            w1 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 1 * weight_width_stride
                )
            )

        comptime if kWidth >= 3:
            w2 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 2 * weight_width_stride
                )
            )

        comptime if kWidth >= 4:
            w3 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 3 * weight_width_stride
                )
            )

        var out_vals_channel: SIMD[output_dtype, kNElts] = 0
        var silu_active = Bool(Int(silu_activation) != 0)

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break

            # Get current seq_idx value
            var cur_seq_idx_offset: UInt32 = (
                UInt32(batch_id) * seq_idx_batch_stride
                + UInt32(seq_pos) * seq_idx_l_stride
            )
            var cur_seq_idx_val = seq_idx.raw_load(cur_seq_idx_offset)
            var cur_seq_idx: Int32 = Int32(cur_seq_idx_val)

            var conv_sum: Scalar[output_dtype] = cur_bias

            # Use scalar operations for all kWidth values to avoid SIMD issues with non-power-of-2 sizes
            # For channel-last (B, L, C): offset = _batch * x_batch_stride + seq * x_l_stride + channel * x_c_stride
            comptime if kWidth == 1:
                var input_l: Int = seq_pos
                if input_l >= 0 and input_l < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    var input_seq_idx: Int32 = Int32(input_seq_idx_val)
                    if input_seq_idx == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        var x_val = Scalar[x_dtype](x.raw_load(load_offset))
                        conv_sum += Scalar[output_dtype](
                            Scalar[output_dtype](x_val)
                            * Scalar[output_dtype](w0)
                        )
            elif kWidth == 2:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 1
                var input_l1: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l0) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l0) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x0 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l1) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l1) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x1 = Scalar[x_dtype](x.raw_load(load_offset))
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                )
            elif kWidth == 3:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 2
                var input_l1: Int = seq_pos - 1
                var input_l2: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l0) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l0) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x0 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l1) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l1) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x1 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l2) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l2) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x2 = Scalar[x_dtype](x.raw_load(load_offset))
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                )
            else:  # kWidth == 4
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var x3: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 3
                var input_l1: Int = seq_pos - 2
                var input_l2: Int = seq_pos - 1
                var input_l3: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l0) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l0) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x0 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l1) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l1) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x1 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l2) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l2) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x2 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l3 >= 0 and input_l3 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l3) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l3) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x3 = Scalar[x_dtype](x.raw_load(load_offset))
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                    + Scalar[output_dtype](w3) * Scalar[output_dtype](x3)
                )
            var out_val: Scalar[output_dtype] = conv_sum
            if silu_active:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            out_vals_channel[i] = out_val

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break
            var out_offset: UInt32 = (
                UInt32(batch_id) * out_batch_stride
                + UInt32(seq_pos) * out_l_stride
                + UInt32(c_idx) * out_c_stride
            )
            output.raw_store(out_offset, out_vals_channel[i])


# Optimized GPU implementation for channel-last without bias but with seq_idx as TileTensor
def causal_conv1d_channel_last_fwd_gpu_no_bias_with_seq_idx[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    seq_idx_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    seq_idx_LT: TensorLayout,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin],  # Shape (B, L, C)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin
    ],  # Shape (B, L, C)
    seq_idx: TileTensor[
        seq_idx_dtype, seq_idx_LT, MutUntrackedOrigin
    ],  # Shape (B, L)
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    seq_idx_batch_stride: UInt32,
    seq_idx_l_stride: UInt32,
    silu_activation: Int8,
):
    """
    Optimized causal conv1d implementation for channel last data layout using SIMD operations (no bias) with seq_idx support.

    Key optimizations:
    1. SIMD vectorization for input/output operations across channels
    2. Efficient memory access patterns with coalesced loads
    3. Vectorized weight loading and computation
    4. Optimized activation function with SIMD operations
    5. Better thread utilization and memory bandwidth usage
    6. seq_idx support for conditional processing
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_chunk_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = _batch
    var nSeqLen: Int = _seqlen
    var nChannels: Int = _dim

    if batch_id >= nBatches or kWidth != _width:
        return

    # Work with 3D tensor directly - manually load SIMD vectors using pointer arithmetic
    # For channel-last (B, L, C), we load SIMD vectors along the channel dimension

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return

    var channel_start: Int = channel_chunk_id * kNElts

    if channel_start >= nChannels:
        return

    for c_offset in range(kNElts):
        var c_idx: Int = channel_start + c_offset
        if c_idx >= nChannels:
            break

        # Load weights directly from memory to avoid vectorize issues
        # For kWidth == 3, use scalar operations to avoid SIMD issues
        var w0: Scalar[weight_dtype] = 0
        var w1: Scalar[weight_dtype] = 0
        var w2: Scalar[weight_dtype] = 0
        var w3: Scalar[weight_dtype] = 0

        comptime if kWidth >= 1:
            w0 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 0 * weight_width_stride
                )
            )

        comptime if kWidth >= 2:
            w1 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 1 * weight_width_stride
                )
            )

        comptime if kWidth >= 3:
            w2 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 2 * weight_width_stride
                )
            )

        comptime if kWidth >= 4:
            w3 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 3 * weight_width_stride
                )
            )

        var out_vals_channel: SIMD[output_dtype, kNElts] = 0
        var silu_active = Bool(Int(silu_activation) != 0)

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break

            # Get current seq_idx value
            var cur_seq_idx_offset: UInt32 = (
                UInt32(batch_id) * seq_idx_batch_stride
                + UInt32(seq_pos) * seq_idx_l_stride
            )
            var cur_seq_idx_val = seq_idx.raw_load(cur_seq_idx_offset)
            var cur_seq_idx: Int32 = Int32(cur_seq_idx_val)

            var conv_sum: Scalar[output_dtype] = 0.0

            # Use scalar operations for all kWidth values to avoid SIMD issues with non-power-of-2 sizes
            # For channel-last (B, L, C): offset = _batch * x_batch_stride + seq * x_l_stride + channel * x_c_stride
            comptime if kWidth == 1:
                var input_l: Int = seq_pos
                if input_l >= 0 and input_l < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    var input_seq_idx: Int32 = Int32(input_seq_idx_val)
                    if input_seq_idx == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        var x_val = Scalar[x_dtype](x.raw_load(load_offset))
                        conv_sum += Scalar[output_dtype](
                            Scalar[output_dtype](x_val)
                            * Scalar[output_dtype](w0)
                        )
            elif kWidth == 2:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 1
                var input_l1: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l0) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l0) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x0 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l1) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l1) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x1 = Scalar[x_dtype](x.raw_load(load_offset))
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                )
            elif kWidth == 3:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 2
                var input_l1: Int = seq_pos - 1
                var input_l2: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l0) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l0) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x0 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l1) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l1) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x1 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l2) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l2) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x2 = Scalar[x_dtype](x.raw_load(load_offset))
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                )
            else:  # kWidth == 4
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var x3: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 3
                var input_l1: Int = seq_pos - 2
                var input_l2: Int = seq_pos - 1
                var input_l3: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l0) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l0) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x0 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l1) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l1) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x1 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l2) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l2) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x2 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l3 >= 0 and input_l3 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l3) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(input_l3) * x_l_stride
                            + UInt32(c_idx) * x_c_stride
                        )
                        x3 = Scalar[x_dtype](x.raw_load(load_offset))
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                    + Scalar[output_dtype](w3) * Scalar[output_dtype](x3)
                )
            var out_val: Scalar[output_dtype] = conv_sum
            if silu_active:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            out_vals_channel[i] = out_val

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break
            var out_offset: UInt32 = (
                UInt32(batch_id) * out_batch_stride
                + UInt32(seq_pos) * out_l_stride
                + UInt32(c_idx) * out_c_stride
            )
            output.raw_store(out_offset, out_vals_channel[i])


# ============================================================================
# Channel-First GPU Implementations with seq_idx as TileTensor
# ============================================================================


# Optimized GPU implementation for channel-first with bias and seq_idx as TileTensor
def causal_conv1d_channel_first_fwd_gpu_with_seq_idx[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
    seq_idx_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    bias_LT: TensorLayout,
    seq_idx_LT: TensorLayout,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin],  # Shape (B, C, L)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin
    ],  # Shape (B, C, L)
    bias: TileTensor[
        bias_dtype, bias_LT, MutUntrackedOrigin
    ],  # Shape (C,), stride = 1
    seq_idx: TileTensor[
        seq_idx_dtype, seq_idx_LT, MutUntrackedOrigin
    ],  # Shape (B, L)
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    seq_idx_batch_stride: UInt32,
    seq_idx_l_stride: UInt32,
    silu_activation: Int8,
):
    """
    Optimized causal conv1d implementation for channel-first data layout using SIMD operations with seq_idx support.

    For channel-first (B, C, L): x_c_stride = L, x_l_stride = 1
    Offset = batch * x_batch_stride + channel * x_c_stride + seq * x_l_stride
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_chunk_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = _batch
    var nSeqLen: Int = _seqlen
    var nChannels: Int = _dim

    if batch_id >= nBatches or kWidth != _width:
        return

    # Work with 3D tensor directly - manually load SIMD vectors using pointer arithmetic
    # For channel-first (B, C, L), we load SIMD vectors along the sequence dimension

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return

    var channel_start: Int = channel_chunk_id * kNElts

    if channel_start >= nChannels:
        return

    # Safety check for null pointers
    if (
        Int(x._storage) == 0
        or Int(output._storage) == 0
        or Int(weight._storage) == 0
        or Int(bias._storage) == 0
    ):
        return

    # Safety check for bias tensor dimensions
    var bias_dim = Int(bias.dim[0]())
    if bias_dim == 0:
        return

    # For channel-first (B, C, L), we process each channel separately
    # Since sequence dimension is contiguous (stride=1), we can load directly from memory
    for c_offset in range(kNElts):
        var c_idx: Int = channel_start + c_offset
        if c_idx >= nChannels:
            break

        # Safety check for bias dimension
        if c_idx >= bias_dim:
            break

        var cur_bias: Scalar[output_dtype] = Scalar[output_dtype](
            bias.raw_load(c_idx)
        )
        # Load weights directly from memory to avoid vectorize issues
        # For kWidth == 3, use scalar operations to avoid SIMD issues
        var w0: Scalar[weight_dtype] = 0
        var w1: Scalar[weight_dtype] = 0
        var w2: Scalar[weight_dtype] = 0
        var w3: Scalar[weight_dtype] = 0

        comptime if kWidth >= 1:
            w0 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 0 * weight_width_stride
                )
            )

        comptime if kWidth >= 2:
            w1 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 1 * weight_width_stride
                )
            )

        comptime if kWidth >= 3:
            w2 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 2 * weight_width_stride
                )
            )

        comptime if kWidth >= 4:
            w3 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 3 * weight_width_stride
                )
            )
        var out_vals_channel: SIMD[output_dtype, kNElts] = 0
        var silu_active = Bool(Int(silu_activation) != 0)

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break

            # Get current seq_idx value
            var cur_seq_idx_offset: UInt32 = (
                UInt32(batch_id) * seq_idx_batch_stride
                + UInt32(seq_pos) * seq_idx_l_stride
            )
            var cur_seq_idx_val = seq_idx.raw_load(cur_seq_idx_offset)
            var cur_seq_idx: Int32 = Int32(cur_seq_idx_val)

            var conv_sum: Scalar[output_dtype] = cur_bias

            # Use scalar operations for all kWidth values to avoid SIMD issues with non-power-of-2 sizes
            # For channel-first (B, C, L): offset = _batch * x_batch_stride + channel * x_c_stride + seq * x_l_stride
            comptime if kWidth == 1:
                var input_l: Int = seq_pos
                if input_l >= 0 and input_l < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    var input_seq_idx: Int32 = Int32(input_seq_idx_val)
                    if input_seq_idx == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l) * x_l_stride
                        )
                        var x_val = Scalar[x_dtype](x.raw_load(load_offset))
                        conv_sum += Scalar[output_dtype](
                            Scalar[output_dtype](x_val)
                            * Scalar[output_dtype](w0)
                        )
            elif kWidth == 2:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 1
                var input_l1: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l0) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l0) * x_l_stride
                        )
                        x0 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l1) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l1) * x_l_stride
                        )
                        x1 = Scalar[x_dtype](x.raw_load(load_offset))
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                )
            elif kWidth == 3:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 2
                var input_l1: Int = seq_pos - 1
                var input_l2: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l0) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l0) * x_l_stride
                        )
                        x0 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l1) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l1) * x_l_stride
                        )
                        x1 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l2) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l2) * x_l_stride
                        )
                        x2 = Scalar[x_dtype](x.raw_load(load_offset))
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                )
            else:  # kWidth == 4
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var x3: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 3
                var input_l1: Int = seq_pos - 2
                var input_l2: Int = seq_pos - 1
                var input_l3: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l0) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l0) * x_l_stride
                        )
                        x0 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l1) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l1) * x_l_stride
                        )
                        x1 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l2) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l2) * x_l_stride
                        )
                        x2 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l3 >= 0 and input_l3 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l3) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l3) * x_l_stride
                        )
                        x3 = Scalar[x_dtype](x.raw_load(load_offset))
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                    + Scalar[output_dtype](w3) * Scalar[output_dtype](x3)
                )
            var out_val: Scalar[output_dtype] = conv_sum
            if silu_active:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            out_vals_channel[i] = out_val

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break
            var out_offset: UInt32 = (
                UInt32(batch_id) * out_batch_stride
                + UInt32(c_idx) * out_c_stride
                + UInt32(seq_pos) * out_l_stride
            )
            output.raw_store(out_offset, out_vals_channel[i])


# Optimized GPU implementation for channel-first without bias but with seq_idx as TileTensor
def causal_conv1d_channel_first_fwd_gpu_no_bias_with_seq_idx[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    seq_idx_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    seq_idx_LT: TensorLayout,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin],  # Shape (B, C, L)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin
    ],  # Shape (B, C, L)
    seq_idx: TileTensor[
        seq_idx_dtype, seq_idx_LT, MutUntrackedOrigin
    ],  # Shape (B, L)
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    seq_idx_batch_stride: UInt32,
    seq_idx_l_stride: UInt32,
    silu_activation: Int8,
):
    """
    Optimized causal conv1d implementation for channel-first data layout using SIMD operations (no bias) with seq_idx support.

    For channel-first (B, C, L): x_c_stride = L, x_l_stride = 1
    Offset = batch * x_batch_stride + channel * x_c_stride + seq * x_l_stride
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_chunk_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = _batch
    var nSeqLen: Int = _seqlen
    var nChannels: Int = _dim

    if batch_id >= nBatches or kWidth != _width:
        return

    # Work with 3D tensor directly - manually load SIMD vectors using pointer arithmetic
    # For channel-first (B, C, L), we load SIMD vectors along the sequence dimension

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return

    var channel_start: Int = channel_chunk_id * kNElts

    if channel_start >= nChannels:
        return

    for c_offset in range(kNElts):
        var c_idx: Int = channel_start + c_offset
        if c_idx >= nChannels:
            break

        # Load weights directly from memory to avoid vectorize issues
        # For kWidth == 3, use scalar operations to avoid SIMD issues
        var w0: Scalar[weight_dtype] = 0
        var w1: Scalar[weight_dtype] = 0
        var w2: Scalar[weight_dtype] = 0
        var w3: Scalar[weight_dtype] = 0

        comptime if kWidth >= 1:
            w0 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 0 * weight_width_stride
                )
            )

        comptime if kWidth >= 2:
            w1 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 1 * weight_width_stride
                )
            )

        comptime if kWidth >= 3:
            w2 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 2 * weight_width_stride
                )
            )

        comptime if kWidth >= 4:
            w3 = Scalar[weight_dtype](
                weight.raw_load(
                    UInt32(c_idx) * weight_c_stride + 3 * weight_width_stride
                )
            )
        var out_vals_channel: SIMD[output_dtype, kNElts] = 0
        var silu_active = Bool(Int(silu_activation) != 0)

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break

            # Get current seq_idx value
            var cur_seq_idx_offset: UInt32 = (
                UInt32(batch_id) * seq_idx_batch_stride
                + UInt32(seq_pos) * seq_idx_l_stride
            )
            var cur_seq_idx_val = seq_idx.raw_load(cur_seq_idx_offset)
            var cur_seq_idx: Int32 = Int32(cur_seq_idx_val)

            var conv_sum: Scalar[output_dtype] = 0.0

            # Use scalar operations for all kWidth values to avoid SIMD issues with non-power-of-2 sizes
            # For channel-first (B, C, L): offset = _batch * x_batch_stride + channel * x_c_stride + seq * x_l_stride
            comptime if kWidth == 1:
                var input_l: Int = seq_pos
                if input_l >= 0 and input_l < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    var input_seq_idx: Int32 = Int32(input_seq_idx_val)
                    if input_seq_idx == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l) * x_l_stride
                        )
                        var x_val = Scalar[x_dtype](x.raw_load(load_offset))
                        conv_sum += Scalar[output_dtype](
                            Scalar[output_dtype](x_val)
                            * Scalar[output_dtype](w0)
                        )
            elif kWidth == 2:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 1
                var input_l1: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l0) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l0) * x_l_stride
                        )
                        x0 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l1) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l1) * x_l_stride
                        )
                        x1 = Scalar[x_dtype](x.raw_load(load_offset))
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                )
            elif kWidth == 3:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 2
                var input_l1: Int = seq_pos - 1
                var input_l2: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l0) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l0) * x_l_stride
                        )
                        x0 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l1) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l1) * x_l_stride
                        )
                        x1 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l2) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l2) * x_l_stride
                        )
                        x2 = Scalar[x_dtype](x.raw_load(load_offset))
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                )
            else:  # kWidth == 4
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var x3: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 3
                var input_l1: Int = seq_pos - 2
                var input_l2: Int = seq_pos - 1
                var input_l3: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l0) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l0) * x_l_stride
                        )
                        x0 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l1) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l1) * x_l_stride
                        )
                        x1 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l2) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l2) * x_l_stride
                        )
                        x2 = Scalar[x_dtype](x.raw_load(load_offset))
                if input_l3 >= 0 and input_l3 < nSeqLen:
                    var input_seq_idx_offset: UInt32 = (
                        UInt32(batch_id) * seq_idx_batch_stride
                        + UInt32(input_l3) * seq_idx_l_stride
                    )
                    var input_seq_idx_val = seq_idx.raw_load(
                        input_seq_idx_offset
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        var load_offset: UInt32 = (
                            UInt32(batch_id) * x_batch_stride
                            + UInt32(c_idx) * x_c_stride
                            + UInt32(input_l3) * x_l_stride
                        )
                        x3 = Scalar[x_dtype](x.raw_load(load_offset))
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                    + Scalar[output_dtype](w3) * Scalar[output_dtype](x3)
                )
            var out_val: Scalar[output_dtype] = conv_sum
            if silu_active:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            out_vals_channel[i] = out_val

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break
            var out_offset: UInt32 = (
                UInt32(batch_id) * out_batch_stride
                + UInt32(c_idx) * out_c_stride
                + UInt32(seq_pos) * out_l_stride
            )
            output.raw_store(out_offset, out_vals_channel[i])


# ============================================================================
# Causal Conv1D Update Kernels
# ============================================================================
# These kernels implement incremental (step-by-step) convolution for inference,
# maintaining a conv_state buffer that gets updated with each step.


def causal_conv1d_update_cpu[
    x_dtype: DType,
    conv_state_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,  # seqlen of x (typically 1 for autoregressive inference)
    width: Int,
    state_len: Int,  # state_len of conv_state (>= width - 1)
    x: TileTensor[
        mut=False, x_dtype, ...
    ],  # Shape (B, C, L) or (B, C) when L=1
    conv_state: TileTensor[mut=True, conv_state_dtype, ...],  # Shape (B, C, S)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # Shape (B, C, L)
    bias: TileTensor[mut=False, bias_dtype, ...],  # Shape (C,)
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    conv_state_batch_stride: UInt32,
    conv_state_c_stride: UInt32,
    conv_state_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    silu_activation: Bool,
):
    """
    CPU implementation of causal conv1d update for incremental inference.

    This kernel:
    1. Concatenates conv_state with x to form a sliding window
    2. Computes convolution output for the new positions
    3. Updates conv_state with the new values from x

    Simple mode (no circular buffer):
    - conv_state holds the last (state_len) values
    - New x values are appended, old values are shifted out

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        conv_state_dtype: Element type of the convolution state tensor
            `conv_state`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.
        bias_dtype: Element type of the bias tensor `bias`.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length of input x (typically 1).
        width: Kernel width.
        state_len: Length of conv_state (>= width - 1).
        x: Input tensor.
        conv_state: Convolution state buffer (modified in-place).
        weight: Convolution weights.
        output: Output tensor.
        bias: Bias tensor.
        x_batch_stride: Stride for batch dimension in x.
        x_c_stride: Stride for channel dimension in x.
        x_l_stride: Stride for sequence length dimension in x.
        conv_state_batch_stride: Stride for batch dimension in conv_state.
        conv_state_c_stride: Stride for channel dimension in conv_state.
        conv_state_l_stride: Stride for state length dimension in conv_state.
        weight_c_stride: Stride for channel dimension in weight.
        weight_width_stride: Stride for kernel width dimension in weight.
        out_batch_stride: Stride for batch dimension in output.
        out_c_stride: Stride for channel dimension in output.
        out_l_stride: Stride for sequence length dimension in output.
        silu_activation: Whether to apply SiLU activation.
    """
    var width_minus_1: Int = width - 1

    for b in range(batch):
        for c in range(dim):
            var weight_c_base = Int(UInt32(c) * weight_c_stride)
            var cur_bias: Scalar[output_dtype] = Scalar[output_dtype](
                bias.raw_load(c)
            )
            # Process each position in the input sequence
            for l in range(seqlen):
                # Compute convolution sum using conv_state and x
                var conv_sum: Scalar[output_dtype] = cur_bias

                for w in range(width):
                    # Position in the virtual concatenated sequence [conv_state, x]
                    var src_pos = state_len + l - (width_minus_1 - w)
                    var input_val: Scalar[x_dtype] = 0.0

                    if src_pos >= state_len:
                        # Read from x
                        var x_l_pos = src_pos - state_len
                        var x_offset = Int(
                            UInt32(b) * x_batch_stride
                            + UInt32(c) * x_c_stride
                            + UInt32(x_l_pos) * x_l_stride
                        )
                        input_val = x.raw_load(x_offset)
                    elif src_pos >= 0:
                        # Read from conv_state
                        var conv_state_offset = Int(
                            UInt32(b) * conv_state_batch_stride
                            + UInt32(c) * conv_state_c_stride
                            + UInt32(src_pos) * conv_state_l_stride
                        )
                        input_val = Scalar[x_dtype](
                            conv_state.raw_load(conv_state_offset)
                        )
                    # else: src_pos < 0, treat as 0 (zero padding)

                    var weight_offset = weight_c_base + Int(
                        UInt32(w) * weight_width_stride
                    )
                    var weight_val: Scalar[weight_dtype] = weight.raw_load(
                        weight_offset
                    )
                    conv_sum = conv_sum + Scalar[output_dtype](
                        input_val * Scalar[x_dtype](weight_val)
                    )

                # Write output
                var out_offset = Int(
                    UInt32(b) * out_batch_stride
                    + UInt32(c) * out_c_stride
                    + UInt32(l) * out_l_stride
                )
                var out_val: Scalar[output_dtype] = conv_sum
                if silu_activation:
                    comptime if output_dtype.is_floating_point():
                        out_val = silu(out_val)
                    else:
                        out_val = silu(out_val.cast[.float32]()).cast[
                            output_dtype
                        ]()
                output.raw_store(out_offset, out_val)

            # Update conv_state: shift old values and add new x values
            if seqlen >= state_len:
                # x is longer than state, just copy last state_len values from x
                for s in range(state_len):
                    var x_l_pos = seqlen - state_len + s
                    var x_offset = Int(
                        UInt32(b) * x_batch_stride
                        + UInt32(c) * x_c_stride
                        + UInt32(x_l_pos) * x_l_stride
                    )
                    var x_val = x.raw_load(x_offset)
                    var conv_state_offset = Int(
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32(s) * conv_state_l_stride
                    )
                    conv_state.raw_store(
                        conv_state_offset, Scalar[conv_state_dtype](x_val)
                    )
            else:
                # Shift conv_state left by seqlen positions, then append x
                for s in range(state_len - seqlen):
                    var src_offset = Int(
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32((s + seqlen)) * conv_state_l_stride
                    )
                    var dst_offset = Int(
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32(s) * conv_state_l_stride
                    )
                    var val = conv_state.raw_load(src_offset)
                    conv_state.raw_store(dst_offset, val)

                # Copy x values to the end
                for l in range(seqlen):
                    var x_offset = Int(
                        UInt32(b) * x_batch_stride
                        + UInt32(c) * x_c_stride
                        + UInt32(l) * x_l_stride
                    )
                    var x_val = x.raw_load(x_offset)
                    var conv_state_offset = Int(
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32((state_len - seqlen + l)) * conv_state_l_stride
                    )
                    conv_state.raw_store(
                        conv_state_offset, Scalar[conv_state_dtype](x_val)
                    )


def causal_conv1d_update_cpu_no_bias[
    x_dtype: DType,
    conv_state_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    state_len: Int,
    x: TileTensor[mut=False, x_dtype, ...],
    conv_state: TileTensor[mut=True, conv_state_dtype, ...],
    weight: TileTensor[mut=False, weight_dtype, ...],
    output: TileTensor[mut=True, output_dtype, ...],
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    conv_state_batch_stride: UInt32,
    conv_state_c_stride: UInt32,
    conv_state_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    silu_activation: Bool,
):
    """CPU implementation of causal conv1d update without bias.

    Performs incremental convolution for autoregressive decode by treating
    `conv_state` followed by `x` as a virtual sliding window, computing the
    output for the new positions, then shifting the newest `state_len`
    values back into `conv_state` in place.

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        conv_state_dtype: Element type of the convolution state tensor
            `conv_state`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.

    Args:
        batch: Number of sequences processed in parallel.
        dim: Number of channels per sequence position.
        seqlen: Number of new input positions in `x` (1 for autoregressive
            decode).
        width: Convolution kernel width in positions.
        state_len: Length of the rolling buffer stored in `conv_state`;
            must be at least `width - 1`.
        x: Input tensor of shape (B, C, L) holding the new positions to
            convolve.
        conv_state: Rolling convolution state of shape (B, C, S) that
            holds the last `state_len` values; updated in place.
        weight: Convolution weights of shape (C, W).
        output: Output tensor of shape (B, C, L) receiving the convolved
            values for the new positions.
        x_batch_stride: Stride in elements between batches of `x`.
        x_c_stride: Stride in elements between channels of `x`.
        x_l_stride: Stride in elements between sequence positions of `x`.
        conv_state_batch_stride: Stride in elements between batches of
            `conv_state`.
        conv_state_c_stride: Stride in elements between channels of
            `conv_state`.
        conv_state_l_stride: Stride in elements between state positions of
            `conv_state`.
        weight_c_stride: Stride in elements between channels of `weight`.
        weight_width_stride: Stride in elements between taps of `weight`
            along the kernel-width dimension.
        out_batch_stride: Stride in elements between batches of `output`.
        out_c_stride: Stride in elements between channels of `output`.
        out_l_stride: Stride in elements between sequence positions of
            `output`.
        silu_activation: Whether to apply the SiLU activation to the
            output values before storing.
    """
    var width_minus_1: Int = width - 1

    for b in range(batch):
        for c in range(dim):
            var weight_c_base = Int(UInt32(c) * weight_c_stride)

            for l in range(seqlen):
                var conv_sum: Scalar[output_dtype] = 0.0

                for w in range(width):
                    var src_pos = state_len + l - (width_minus_1 - w)
                    var input_val: Scalar[x_dtype] = 0.0

                    if src_pos >= state_len:
                        var x_l_pos = src_pos - state_len
                        var x_offset = Int(
                            UInt32(b) * x_batch_stride
                            + UInt32(c) * x_c_stride
                            + UInt32(x_l_pos) * x_l_stride
                        )
                        input_val = x.raw_load(x_offset)
                    elif src_pos >= 0:
                        var conv_state_offset = Int(
                            UInt32(b) * conv_state_batch_stride
                            + UInt32(c) * conv_state_c_stride
                            + UInt32(src_pos) * conv_state_l_stride
                        )
                        input_val = Scalar[x_dtype](
                            conv_state.raw_load(conv_state_offset)
                        )
                    var weight_offset = weight_c_base + Int(
                        UInt32(w) * weight_width_stride
                    )
                    var weight_val: Scalar[weight_dtype] = weight.raw_load(
                        weight_offset
                    )
                    conv_sum = conv_sum + Scalar[output_dtype](
                        input_val * Scalar[x_dtype](weight_val)
                    )

                var out_offset = Int(
                    UInt32(b) * out_batch_stride
                    + UInt32(c) * out_c_stride
                    + UInt32(l) * out_l_stride
                )
                var out_val: Scalar[output_dtype] = conv_sum
                if silu_activation:
                    comptime if output_dtype.is_floating_point():
                        out_val = silu(out_val)
                    else:
                        out_val = silu(out_val.cast[.float32]()).cast[
                            output_dtype
                        ]()
                output.raw_store(out_offset, out_val)

            # Update conv_state
            if seqlen >= state_len:
                for s in range(state_len):
                    var x_l_pos = seqlen - state_len + s
                    var x_offset = Int(
                        UInt32(b) * x_batch_stride
                        + UInt32(c) * x_c_stride
                        + UInt32(x_l_pos) * x_l_stride
                    )
                    var x_val = x.raw_load(x_offset)
                    var conv_state_offset = Int(
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32(s) * conv_state_l_stride
                    )
                    conv_state.raw_store(
                        conv_state_offset, Scalar[conv_state_dtype](x_val)
                    )
            else:
                for s in range(state_len - seqlen):
                    var src_offset = Int(
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32((s + seqlen)) * conv_state_l_stride
                    )
                    var dst_offset = Int(
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32(s) * conv_state_l_stride
                    )
                    var val = conv_state.raw_load(src_offset)
                    conv_state.raw_store(dst_offset, val)

                for l in range(seqlen):
                    var x_offset = Int(
                        UInt32(b) * x_batch_stride
                        + UInt32(c) * x_c_stride
                        + UInt32(l) * x_l_stride
                    )
                    var x_val = x.raw_load(x_offset)
                    var conv_state_offset = Int(
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32((state_len - seqlen + l)) * conv_state_l_stride
                    )
                    conv_state.raw_store(
                        conv_state_offset, Scalar[conv_state_dtype](x_val)
                    )


def causal_conv1d_update_gpu[
    x_dtype: DType,
    conv_state_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
    kNThreads: Int,
    x_LT: TensorLayout,
    conv_state_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    bias_LT: TensorLayout,
    x_engine: TensorEngine,
    conv_state_engine: TensorEngine,
    weight_engine: TensorEngine,
    output_engine: TensorEngine,
    bias_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    state_len: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine],
    conv_state: TileTensor[
        conv_state_dtype,
        conv_state_LT,
        MutUntrackedOrigin,
        Engine=conv_state_engine,
    ],
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],
    bias: TileTensor[
        bias_dtype, bias_LT, MutUntrackedOrigin, Engine=bias_engine
    ],
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    conv_state_batch_stride: UInt32,
    conv_state_c_stride: UInt32,
    conv_state_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    silu_activation: Int8,
):
    """GPU kernel for causal conv1d update operation (for autoregressive decode).

    This kernel performs incremental updates to maintain convolution state for efficient
    autoregressive token generation. It processes a new input sequence and updates both
    the output and the internal convolution state.

    Grid: (batch, ceildiv(dim, kNThreads))
    Block: kNThreads

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        conv_state_dtype: Element type of the convolution state tensor
            `conv_state`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.
        bias_dtype: Element type of the bias tensor `bias`.
        kNThreads: Number of threads per block used to process the channel
            dimension.
        x_LT: TensorLayout of the input tensor `x`.
        conv_state_LT: TensorLayout of the convolution state tensor
            `conv_state`.
        weight_LT: TensorLayout of the weight tensor `weight`.
        output_LT: TensorLayout of the output tensor `output`.
        bias_LT: TensorLayout of the bias tensor `bias`.
        x_engine: Engine of the input tensor `x`.
        conv_state_engine: Engine of the convolution state tensor
            `conv_state`.
        weight_engine: Engine of the weight tensor `weight`.
        output_engine: Engine of the output tensor `output`.
        bias_engine: Engine of the bias tensor `bias`.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length of the new input.
        width: Kernel width.
        state_len: Length of the convolution state buffer.
        x: Input tensor of shape (B, C, L).
        conv_state: Convolution state tensor of shape (B, C, state_len).
        weight: Weight tensor of shape (C, W).
        output: Output tensor of shape (B, C, L).
        bias: Bias tensor of shape (C,).
        x_batch_stride: Stride for the batch dimension of the input tensor.
        x_c_stride: Stride for the channel dimension of the input tensor.
        x_l_stride: Stride for the sequence length dimension of the input tensor.
        conv_state_batch_stride: Stride for the batch dimension of the conv state tensor.
        conv_state_c_stride: Stride for the channel dimension of the conv state tensor.
        conv_state_l_stride: Stride for the sequence length dimension of the conv state tensor.
        weight_c_stride: Stride for the channel dimension of the weight tensor.
        weight_width_stride: Stride for the width dimension of the weight tensor.
        out_batch_stride: Stride for the batch dimension of the output tensor.
        out_c_stride: Stride for the channel dimension of the output tensor.
        out_l_stride: Stride for the sequence length dimension of the output tensor.
        silu_activation: Whether to apply SiLU activation (Int8: 0 or 1).
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)
    var _state_len = Int(state_len)
    var b = block_idx.x
    var c_base = block_idx.y * kNThreads
    var c = c_base + thread_idx.x

    if b >= _batch or c >= _dim:
        return

    var width_minus_1: Int = _width - 1
    var weight_c_base = Int(UInt32(c) * weight_c_stride)
    var cur_bias: Scalar[output_dtype] = Scalar[output_dtype](bias.raw_load(c))
    var silu_active = Bool(silu_activation != 0)

    for l in range(_seqlen):
        var conv_sum: Scalar[output_dtype] = cur_bias

        for w in range(_width):
            var src_pos = _state_len + l - (width_minus_1 - w)
            var input_val: Scalar[x_dtype] = 0.0

            if src_pos >= _state_len:
                var x_l_pos = src_pos - _state_len
                var x_offset = Int(
                    UInt32(b) * x_batch_stride
                    + UInt32(c) * x_c_stride
                    + UInt32(x_l_pos) * x_l_stride
                )
                input_val = x.raw_load(x_offset)
            elif src_pos >= 0:
                var conv_state_offset = Int(
                    UInt32(b) * conv_state_batch_stride
                    + UInt32(c) * conv_state_c_stride
                    + UInt32(src_pos) * conv_state_l_stride
                )
                input_val = Scalar[x_dtype](
                    conv_state.raw_load(conv_state_offset)
                )
            var weight_offset = weight_c_base + Int(
                UInt32(w) * weight_width_stride
            )
            var weight_val: Scalar[weight_dtype] = weight.raw_load(
                weight_offset
            )
            conv_sum = conv_sum + Scalar[output_dtype](
                input_val * Scalar[x_dtype](weight_val)
            )
        var out_offset = Int(
            UInt32(b) * out_batch_stride
            + UInt32(c) * out_c_stride
            + UInt32(l) * out_l_stride
        )
        var out_val: Scalar[output_dtype] = conv_sum
        if silu_active:
            comptime if output_dtype.is_floating_point():
                out_val = silu(out_val)
            else:
                out_val = silu(out_val.cast[.float32]()).cast[output_dtype]()
        output.raw_store(out_offset, out_val)

    # Update conv_state
    if _seqlen >= _state_len:
        for s in range(_state_len):
            var x_l_pos = _seqlen - _state_len + s
            var x_offset = Int(
                UInt32(b) * x_batch_stride
                + UInt32(c) * x_c_stride
                + UInt32(x_l_pos) * x_l_stride
            )
            var x_val = x.raw_load(x_offset)
            var conv_state_offset = Int(
                UInt32(b) * conv_state_batch_stride
                + UInt32(c) * conv_state_c_stride
                + UInt32(s) * conv_state_l_stride
            )
            conv_state.raw_store(
                conv_state_offset, Scalar[conv_state_dtype](x_val)
            )
    else:
        for s in range(_state_len - _seqlen):
            var src_offset = Int(
                UInt32(b) * conv_state_batch_stride
                + UInt32(c) * conv_state_c_stride
                + UInt32((s + _seqlen)) * conv_state_l_stride
            )
            var dst_offset = Int(
                UInt32(b) * conv_state_batch_stride
                + UInt32(c) * conv_state_c_stride
                + UInt32(s) * conv_state_l_stride
            )
            var val = conv_state.raw_load(src_offset)
            conv_state.raw_store(dst_offset, val)

        for l in range(_seqlen):
            var x_offset = Int(
                UInt32(b) * x_batch_stride
                + UInt32(c) * x_c_stride
                + UInt32(l) * x_l_stride
            )
            var x_val = x.raw_load(x_offset)
            var conv_state_offset = Int(
                UInt32(b) * conv_state_batch_stride
                + UInt32(c) * conv_state_c_stride
                + UInt32((_state_len - _seqlen + l)) * conv_state_l_stride
            )
            conv_state.raw_store(
                conv_state_offset, Scalar[conv_state_dtype](x_val)
            )


def causal_conv1d_update_gpu_no_bias[
    x_dtype: DType,
    conv_state_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    kNThreads: Int,
    x_LT: TensorLayout,
    conv_state_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    x_engine: TensorEngine,
    conv_state_engine: TensorEngine,
    weight_engine: TensorEngine,
    output_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    state_len: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine],
    conv_state: TileTensor[
        conv_state_dtype,
        conv_state_LT,
        MutUntrackedOrigin,
        Engine=conv_state_engine,
    ],
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    conv_state_batch_stride: UInt32,
    conv_state_c_stride: UInt32,
    conv_state_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    silu_activation: Int8,
):
    """GPU kernel for causal conv1d update operation without bias (for autoregressive decode).

    This kernel performs incremental updates to maintain convolution state for efficient
    autoregressive token generation. It processes a new input sequence and updates both
    the output and the internal convolution state.

    Grid: (batch, ceildiv(dim, kNThreads))
    Block: kNThreads

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        conv_state_dtype: Element type of the convolution state tensor
            `conv_state`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.
        kNThreads: Number of threads per block used to process the channel
            dimension.
        x_LT: TensorLayout of the input tensor `x`.
        conv_state_LT: TensorLayout of the convolution state tensor
            `conv_state`.
        weight_LT: TensorLayout of the weight tensor `weight`.
        output_LT: TensorLayout of the output tensor `output`.
        x_engine: Engine of the input tensor `x`.
        conv_state_engine: Engine of the convolution state tensor
            `conv_state`.
        weight_engine: Engine of the weight tensor `weight`.
        output_engine: Engine of the output tensor `output`.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length of the new input.
        width: Kernel width.
        state_len: Length of the convolution state buffer.
        x: Input tensor of shape (B, C, L).
        conv_state: Convolution state tensor of shape (B, C, state_len).
        weight: Weight tensor of shape (C, W).
        output: Output tensor of shape (B, C, L).
        x_batch_stride: Stride for the batch dimension of the input tensor.
        x_c_stride: Stride for the channel dimension of the input tensor.
        x_l_stride: Stride for the sequence length dimension of the input tensor.
        conv_state_batch_stride: Stride for the batch dimension of the conv state tensor.
        conv_state_c_stride: Stride for the channel dimension of the conv state tensor.
        conv_state_l_stride: Stride for the sequence length dimension of the conv state tensor.
        weight_c_stride: Stride for the channel dimension of the weight tensor.
        weight_width_stride: Stride for the width dimension of the weight tensor.
        out_batch_stride: Stride for the batch dimension of the output tensor.
        out_c_stride: Stride for the channel dimension of the output tensor.
        out_l_stride: Stride for the sequence length dimension of the output tensor.
        silu_activation: Whether to apply SiLU activation (Int8: 0 or 1).
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)
    var _state_len = Int(state_len)
    var b = block_idx.x
    var c_base = block_idx.y * kNThreads
    var c = c_base + thread_idx.x

    if b >= _batch or c >= _dim:
        return

    var width_minus_1: Int = _width - 1
    var weight_c_base = Int(UInt32(c) * weight_c_stride)
    var silu_active = Bool(silu_activation != 0)

    for l in range(_seqlen):
        var conv_sum: Scalar[output_dtype] = 0.0

        for w in range(_width):
            var src_pos = _state_len + l - (width_minus_1 - w)
            var input_val: Scalar[x_dtype] = 0.0

            if src_pos >= _state_len:
                var x_l_pos = src_pos - _state_len
                var x_offset = Int(
                    UInt32(b) * x_batch_stride
                    + UInt32(c) * x_c_stride
                    + UInt32(x_l_pos) * x_l_stride
                )
                input_val = x.raw_load(x_offset)
            elif src_pos >= 0:
                var conv_state_offset = Int(
                    UInt32(b) * conv_state_batch_stride
                    + UInt32(c) * conv_state_c_stride
                    + UInt32(src_pos) * conv_state_l_stride
                )
                input_val = Scalar[x_dtype](
                    conv_state.raw_load(conv_state_offset)
                )
            var weight_offset = weight_c_base + Int(
                UInt32(w) * weight_width_stride
            )
            var weight_val: Scalar[weight_dtype] = weight.raw_load(
                weight_offset
            )
            conv_sum = conv_sum + Scalar[output_dtype](
                input_val * Scalar[x_dtype](weight_val)
            )
        var out_offset = Int(
            UInt32(b) * out_batch_stride
            + UInt32(c) * out_c_stride
            + UInt32(l) * out_l_stride
        )
        var out_val: Scalar[output_dtype] = conv_sum
        if silu_active:
            comptime if output_dtype.is_floating_point():
                out_val = silu(out_val)
            else:
                out_val = silu(out_val.cast[.float32]()).cast[output_dtype]()
        output.raw_store(out_offset, out_val)

    if _seqlen >= _state_len:
        for s in range(_state_len):
            var x_l_pos = _seqlen - _state_len + s
            var x_offset = Int(
                UInt32(b) * x_batch_stride
                + UInt32(c) * x_c_stride
                + UInt32(x_l_pos) * x_l_stride
            )
            var x_val = x.raw_load(x_offset)
            var conv_state_offset = Int(
                UInt32(b) * conv_state_batch_stride
                + UInt32(c) * conv_state_c_stride
                + UInt32(s) * conv_state_l_stride
            )
            conv_state.raw_store(
                conv_state_offset, Scalar[conv_state_dtype](x_val)
            )
    else:
        for s in range(_state_len - _seqlen):
            var src_offset = Int(
                UInt32(b) * conv_state_batch_stride
                + UInt32(c) * conv_state_c_stride
                + UInt32((s + _seqlen)) * conv_state_l_stride
            )
            var dst_offset = Int(
                UInt32(b) * conv_state_batch_stride
                + UInt32(c) * conv_state_c_stride
                + UInt32(s) * conv_state_l_stride
            )
            var val = conv_state.raw_load(src_offset)
            conv_state.raw_store(dst_offset, val)

        for l in range(_seqlen):
            var x_offset = Int(
                UInt32(b) * x_batch_stride
                + UInt32(c) * x_c_stride
                + UInt32(l) * x_l_stride
            )
            var x_val = x.raw_load(x_offset)
            var conv_state_offset = Int(
                UInt32(b) * conv_state_batch_stride
                + UInt32(c) * conv_state_c_stride
                + UInt32((_state_len - _seqlen + l)) * conv_state_l_stride
            )
            conv_state.raw_store(
                conv_state_offset, Scalar[conv_state_dtype](x_val)
            )
