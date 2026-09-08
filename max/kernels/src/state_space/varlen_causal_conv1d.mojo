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
"""Causal Conv1D with variable length sequence support (vLLM interface).

This module implements causal 1D convolution operations that support variable
length sequences using cumulative sequence lengths (cu_seqlens), compatible
with the vLLM inference interface.

Key Functions:
    - causal_conv1d_varlen_fwd: Forward pass for varlen sequences
    - causal_conv1d_varlen_update: Update function for decode
    - causal_conv1d_varlen_states: Extract states from varlen sequences

vLLM Interface:
    - x: (dim, cu_seq_len) for varlen - sequences concatenated left to right
    - query_start_loc: (batch + 1) int32 - cumulative sequence lengths
    - cache_indices: (batch) int32 - indices into conv_states
    - has_initial_state: (batch) bool - whether to use initial state
    - conv_states: (..., dim, width - 1) - states updated in-place
    - activation: None or "silu" or "swish"
    - pad_slot_id: int - for identifying padded entries
"""

from std.bit import next_power_of_two
from std.algorithm import vectorize
from std.math import ceildiv


from max.gpu import block_idx, thread_idx


from layout import TensorLayout, TileTensor
from layout.coord import Coord
from layout.tensor_engine import TensorEngine

from nn.activations import silu


# ============================================================================
# Constants
# ============================================================================

comptime PAD_SLOT_ID: Int32 = -1

# Lane count of a tap vector holding a WIDTH-tap filter. WIDTH dispatches in
# {1, 2, 3, 4} and 3 is not a valid SIMD width, so round up and zero the
# padding lane; a padded dot product still sums to the WIDTH-tap result.
comptime _TAP_LANES[WIDTH: Int] = Int(next_power_of_two(WIDTH))


# ============================================================================
# Shared helpers
# ============================================================================


@always_inline
def _apply_silu[
    output_dtype: DType
](out_val: Scalar[output_dtype], silu_activation: Bool) -> Scalar[output_dtype]:
    """Optionally apply the SiLU activation, preserving the accumulator dtype.

    Floating-point outputs run SiLU in place; integral outputs promote to f32
    for the activation then cast back. Factored out of the fwd / update / states
    paths (CPU and GPU) which each open-coded this block identically.

    Parameters:
        output_dtype: The output element type.

    Args:
        out_val: The pre-activation convolution output.
        silu_activation: Whether to apply SiLU.

    Returns:
        `out_val`, with SiLU applied when `silu_activation` is True.
    """
    if silu_activation:
        comptime if output_dtype.is_floating_point():
            return silu(out_val)
        else:
            return silu(out_val.cast[.float32]()).cast[output_dtype]()
    return out_val


@always_inline
def _channel_weights[
    weight_dtype: DType,
    WIDTH: Int,
    weight_LT: TensorLayout,
    weight_engine: TensorEngine,
](
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],
    d: Int,
) -> SIMD[weight_dtype, _TAP_LANES[WIDTH]]:
    """Load channel `d`'s `WIDTH` conv weights into a register vector.

    Factored out of the fwd / update GPU kernels, which each preloaded the
    per-channel weights into a fixed 8-wide SIMD. The vector is now the
    padded `WIDTH` (no magic 8), so it holds the taps used plus at most one
    power-of-two rounding lane.

    Parameters:
        weight_dtype: The weight element type.
        WIDTH: The convolution width (number of taps).
        weight_LT: Layout type of the weight tensor.
        weight_engine: Engine of the weight tensor.

    Args:
        weight: The `(dim, width)` weight tensor.
        d: The channel index.

    Returns:
        Channel `d`'s weights, tap `w_idx` in lane `w_idx`, with the padding
        lanes zeroed so a padded dot product still sums to the `WIDTH`-tap
        result.
    """
    var weights = SIMD[weight_dtype, _TAP_LANES[WIDTH]](0)
    comptime for w_idx in range(WIDTH):
        weights[w_idx] = weight.load[width=1, alignment=1](Coord(d, w_idx))
    return weights


# ============================================================================
# Forward-path DRAM I/O owner
# ============================================================================


@fieldwise_init
struct VarlenConvIO[
    x_origin: Origin,
    weight_origin: Origin,
    bias_origin: Origin,
    out_origin: MutOrigin,
    x_dtype: DType,
    weight_dtype: DType,
    bias_dtype: DType,
    out_dtype: DType,
    x_layout: TensorLayout,
    weight_layout: TensorLayout,
    bias_layout: TensorLayout,
    out_layout: TensorLayout,
    x_engine: TensorEngine,
    weight_engine: TensorEngine,
    bias_engine: TensorEngine,
    out_engine: TensorEngine,
    x_addr: AddressSpace,
    weight_addr: AddressSpace,
    bias_addr: AddressSpace,
    out_addr: AddressSpace,
    x_idx: DType,
    weight_idx: DType,
    bias_idx: DType,
    out_idx: DType,
    //,
](ImplicitlyCopyable, Movable):
    """Owner of the forward-path DRAM reads/writes for varlen causal conv1d.

    Holds the `(dim, seqlen)` input `x`, `(dim, width)` `weight`, `(dim,)`
    `bias`, and `(dim, seqlen)` `output` TileTensor views and exposes one
    method per access verb.

    `weight`/`bias` (never axis-permuted) still index through
    `t.load(Coord(...))` -- the view's own `RuntimeLayout` matches the
    `(d, w)` / `(d,)` logical order exactly. `load_x`/`store_out` instead take
    the caller's runtime `dim`/`seqlen` strides and compute the offset
    explicitly (`d*dim_stride + s*seqlen_stride`), matching the conv-state
    ring-buffer's `raw_load`/`raw_store` pattern below. This is required, not
    just symmetric style: `x`/`output`'s *physical* axis order flips under the
    `channels_last` builtin parameter (`kernels.mojo`'s `CausalConv1DVarlenFwd`)
    while the `(d, s)` *logical* argument order here does not, so a
    `Coord(d, s)` load through the view's native layout would silently swap
    axes and read/write the wrong element whenever the caller is
    channels-last. Passing the (already axis-corrected) strides down and
    addressing raw offsets sidesteps the mismatch. See
    `Kernels/claude_kb/entries/patterns/mojo-layout-is-cute-layout-algebra.md`
    ("shape and stride are orthogonal") -- `d`/`s` are the algorithm's shape
    plane; the stride plane is what may vary, so loads/stores must go through
    strides, not through a Coord tied to the view's declared axis order.

    Every view parameter (dtype, layout, origin, storage, address space, index
    type) is inferred from the constructor arguments, so the owner adapts to
    whatever tensor storage the caller's views carry -- the CPU reference and
    GPU kernel pass differently-parameterized views. The read views (`x`,
    `weight`, `bias`) carry origins with a free `mut` (the CPU reference passes
    them immutable, the GPU kernel mutable; reads need no mutability proof); the
    store target `output` pins `MutOrigin` so `store` type-checks. Modeled on
    the owner-per-transition pattern (see `Fp4WeightLoader` in
    `matmul2d_fp4.mojo`), constructed by direct field init so no origin rebase
    is needed. Stateless: the circular conv-state position is not owned here, so
    the same owner serves any path.

    Parameters:
        x_origin: Inferred origin of the input view.
        weight_origin: Inferred origin of the weight view.
        bias_origin: Inferred origin of the bias view.
        out_origin: Inferred mutable origin of the output view.
        x_dtype: Input element type.
        weight_dtype: Weight element type.
        bias_dtype: Bias element type.
        out_dtype: Output element type.
        x_layout: Layout type of the `(dim, seqlen)` input view.
        weight_layout: Layout type of the `(dim, width)` weight view.
        bias_layout: Layout type of the `(dim,)` bias view.
        out_layout: Layout type of the `(dim, seqlen)` output view.
        x_engine: Inferred engine of the input view.
        weight_engine: Inferred engine of the weight view.
        bias_engine: Inferred engine of the bias view.
        out_engine: Inferred engine of the output view.
        x_addr: Inferred address space of the input view.
        weight_addr: Inferred address space of the weight view.
        bias_addr: Inferred address space of the bias view.
        out_addr: Inferred address space of the output view.
        x_idx: Inferred linear-index type of the input view.
        weight_idx: Inferred linear-index type of the weight view.
        bias_idx: Inferred linear-index type of the bias view.
        out_idx: Inferred linear-index type of the output view.
    """

    var x: TileTensor[
        Self.x_dtype,
        Self.x_layout,
        Self.x_origin,
        Engine=Self.x_engine,
        address_space=Self.x_addr,
        linear_idx_type=Self.x_idx,
    ]
    var weight: TileTensor[
        Self.weight_dtype,
        Self.weight_layout,
        Self.weight_origin,
        Engine=Self.weight_engine,
        address_space=Self.weight_addr,
        linear_idx_type=Self.weight_idx,
    ]
    var bias: TileTensor[
        Self.bias_dtype,
        Self.bias_layout,
        Self.bias_origin,
        Engine=Self.bias_engine,
        address_space=Self.bias_addr,
        linear_idx_type=Self.bias_idx,
    ]
    var output: TileTensor[
        Self.out_dtype,
        Self.out_layout,
        Self.out_origin,
        Engine=Self.out_engine,
        address_space=Self.out_addr,
        linear_idx_type=Self.out_idx,
    ]
    # Runtime dim/seqlen strides for `x`/`output`, already axis-corrected by
    # the caller for `channels_last` (see class docstring). `load_x`/
    # `store_out` address through these instead of `Coord(d, s)` because the
    # view's own physical axis order flips under `channels_last` while these
    # arguments' logical (d, s) order does not.
    var x_dim_stride: UInt32
    var x_seqlen_stride: UInt32
    var out_dim_stride: UInt32
    var out_seqlen_stride: UInt32

    @always_inline
    def load_x(self, d: Int, s: Int) -> Scalar[Self.x_dtype]:
        """Load `x[d, s]`: windowed history read of the input.

        Args:
            d: The channel index into the `(dim, seqlen)` input view.
            s: The sequence position index into the `(dim, seqlen)` input view.
        """
        var offset = (
            UInt32(d) * self.x_dim_stride + UInt32(s) * self.x_seqlen_stride
        )
        return self.x.raw_load(offset)

    @always_inline
    def load_weight(self, d: Int, w: Int) -> Scalar[Self.weight_dtype]:
        """Load `weight[d, w]`: per-channel conv tap.

        Args:
            d: The channel index into the `(dim, width)` weight view.
            w: The convolution tap index in `[0, width)`.
        """
        # `[0]` extracts lane 0: generic `Storage` makes `load()` non-scalar.
        return self.weight.load(Coord(d, w))[0]

    @always_inline
    def load_bias(self, d: Int) -> Scalar[Self.bias_dtype]:
        """Load `bias[d]`: per-channel bias.

        Args:
            d: The channel index into the `(dim,)` bias view.
        """
        # `[0]` extracts lane 0 (generic `Storage`, as in `load_weight`).
        return self.bias.load(Coord(d))[0]

    @always_inline
    def store_out(self, d: Int, s: Int, val: Scalar[Self.out_dtype]):
        """Store `output[d, s] = val`: convolution output.

        Args:
            d: The channel index into the `(dim, seqlen)` output view.
            s: The sequence position index into the `(dim, seqlen)` output
                view.
            val: The convolution result to store at `output[d, s]`.
        """
        var offset = (
            UInt32(d) * self.out_dim_stride + UInt32(s) * self.out_seqlen_stride
        )
        self.output.raw_store(offset, val)


# ============================================================================
# CPU Reference Implementations
# ============================================================================


def causal_conv1d_varlen_states_cpu[
    x_dtype: DType,
    cu_seqlens_dtype: DType,
    states_dtype: DType,
](
    total_tokens: Int,
    dim: Int,
    batch: Int,
    state_len: Int,
    x: TileTensor[mut=False, x_dtype, ...],  # Shape (total_tokens, dim)
    cu_seqlens: TileTensor[
        mut=False, cu_seqlens_dtype, ...
    ],  # Shape (batch + 1,)
    states: TileTensor[
        mut=True, states_dtype, ...
    ],  # Shape (batch, dim, state_len)
    x_seqlen_stride: UInt32,
    x_dim_stride: UInt32,
    states_batch_stride: UInt32,
    states_dim_stride: UInt32,
    states_seqlen_stride: UInt32,
):
    """Extract the last state_len elements from each variable length sequence.

    For each sequence in the batch, copies the last state_len tokens (or fewer
    if the sequence is shorter) to the states tensor. If a sequence is shorter
    than state_len, the earlier positions in states are zero-padded.

    This is the CPU reference implementation for causal_conv1d_varlen_states.

    Parameters:
        x_dtype: Data type of the input tensor.
        cu_seqlens_dtype: Data type of the cumulative sequence lengths.
        states_dtype: Data type of the output states tensor.

    Args:
        total_tokens: Total number of tokens across all sequences.
        dim: Number of channels/dimensions.
        batch: Number of sequences.
        state_len: Number of elements to extract per sequence (typically width - 1).
        x: Input tensor of shape (total_tokens, dim).
        cu_seqlens: Cumulative sequence lengths of shape (batch + 1,).
        states: Output states tensor of shape (batch, dim, state_len).
        x_seqlen_stride: Stride for sequence dimension in x.
        x_dim_stride: Stride for dimension in x.
        states_batch_stride: Stride for batch dimension in states.
        states_dim_stride: Stride for dimension in states.
        states_seqlen_stride: Stride for sequence dimension in states.
    """
    # Initialize states to zero
    for b in range(batch):
        for d in range(dim):
            for s in range(state_len):
                var states_offset = (
                    UInt32(b) * states_batch_stride
                    + UInt32(d) * states_dim_stride
                    + UInt32(s) * states_seqlen_stride
                )
                states.raw_store(states_offset, Scalar[states_dtype](0))

    # Extract states for each sequence
    for b in range(batch):
        var end_idx = Int(cu_seqlens.raw_load(b + 1))
        var start_idx_seq = Int(cu_seqlens.raw_load(b))
        var start_idx = max(start_idx_seq, end_idx - state_len)
        var num_elements = end_idx - start_idx

        # Copy elements from x to states
        # states[b, :, -(end_idx - start_idx):] = x[start_idx:end_idx].T
        for i in range(num_elements):
            var x_seq_idx = start_idx + i
            var states_seq_idx = state_len - num_elements + i

            for d in range(dim):
                var x_offset = (
                    UInt32(x_seq_idx) * x_seqlen_stride
                    + UInt32(d) * x_dim_stride
                )
                var states_offset = (
                    UInt32(b) * states_batch_stride
                    + UInt32(d) * states_dim_stride
                    + UInt32(states_seq_idx) * states_seqlen_stride
                )
                var val = x.raw_load(x_offset)
                states.raw_store(states_offset, Scalar[states_dtype](val))


def causal_conv1d_varlen_fwd_cpu[
    x_dtype: DType,
    weight_dtype: DType,
    bias_dtype: DType,
    output_dtype: DType,
    cu_seqlens_dtype: DType,
    cache_indices_dtype: DType,
    has_initial_state_dtype: DType,
    conv_states_dtype: DType,
](
    dim: Int,
    total_seqlen: Int,
    width: Int,
    batch: Int,
    x: TileTensor[
        mut=False, x_dtype, ...
    ],  # Shape (dim, total_seqlen) for varlen
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (dim, width)
    bias: TileTensor[mut=False, bias_dtype, ...],  # Shape (dim,)
    query_start_loc: TileTensor[
        mut=False, cu_seqlens_dtype, ...
    ],  # Shape (batch + 1,)
    cache_indices: TileTensor[
        mut=False, cache_indices_dtype, ...
    ],  # Shape (batch,)
    has_initial_state: TileTensor[
        mut=False, has_initial_state_dtype, ...
    ],  # Shape (batch,)
    conv_states: TileTensor[
        mut=True, conv_states_dtype, ...
    ],  # Shape (..., dim, width - 1)
    output: TileTensor[
        mut=True, output_dtype, ...
    ],  # Shape (dim, total_seqlen)
    x_dim_stride: UInt32,
    x_seqlen_stride: UInt32,
    weight_dim_stride: UInt32,
    weight_width_stride: UInt32,
    out_dim_stride: UInt32,
    out_seqlen_stride: UInt32,
    conv_states_batch_stride: UInt32,
    conv_states_dim_stride: UInt32,
    conv_states_width_stride: UInt32,
    silu_activation: Bool,
    pad_slot_id: Int32,
    has_cache_indices: Bool,
    has_initial_state_flag: Bool,
    has_conv_states: Bool,
    has_bias: Bool,
):
    """Forward pass for causal conv1d with variable length sequences.

    Performs causal 1D convolution on variable length sequences that are
    concatenated together. Uses cumulative sequence lengths to identify
    sequence boundaries.

    This is the CPU reference implementation for causal_conv1d_varlen_fwd.

    Parameters:
        x_dtype: Data type of the input tensor.
        weight_dtype: Data type of the weight tensor.
        bias_dtype: Data type of the bias tensor.
        output_dtype: Data type of the output tensor.
        cu_seqlens_dtype: Data type of the cumulative sequence lengths.
        cache_indices_dtype: Data type of the cache indices.
        has_initial_state_dtype: Data type of the initial-state flags.
        conv_states_dtype: Data type of the convolution states.

    Args:
        dim: Number of channels in the convolution.
        total_seqlen: Total length of the concatenated sequences.
        width: Convolution width (number of taps).
        batch: Number of sequences in the batch.
        x: Input tensor of shape `(dim, total_seqlen)`.
        weight: Weight tensor of shape `(dim, width)`.
        bias: Bias tensor of shape `(dim,)`.
        query_start_loc: Cumulative sequence lengths of shape `(batch + 1,)`.
        cache_indices: Per-sequence indices into `conv_states` of shape
            `(batch,)`.
        has_initial_state: Per-sequence flags of shape `(batch,)` indicating
            whether to use the initial state.
        conv_states: Convolution states of shape `(..., dim, width - 1)`,
            updated in place.
        output: Output tensor of shape `(dim, total_seqlen)`.
        x_dim_stride: Stride for the channel dimension in `x`.
        x_seqlen_stride: Stride for the sequence dimension in `x`.
        weight_dim_stride: Stride for the channel dimension in `weight`.
        weight_width_stride: Stride for the tap dimension in `weight`.
        out_dim_stride: Stride for the channel dimension in `output`.
        out_seqlen_stride: Stride for the sequence dimension in `output`.
        conv_states_batch_stride: Stride for the batch dimension in
            `conv_states`.
        conv_states_dim_stride: Stride for the channel dimension in
            `conv_states`.
        conv_states_width_stride: Stride for the tap dimension in
            `conv_states`.
        silu_activation: Whether to apply the SiLU activation to the output.
        pad_slot_id: Slot ID identifying padded entries to skip.
        has_cache_indices: Whether to consult `cache_indices` for slot lookup.
        has_initial_state_flag: Whether to consult `has_initial_state` per
            sequence.
        has_conv_states: Whether `conv_states` is provided and should be
            updated.
        has_bias: Whether to add `bias` to the convolution sum.
    """
    var width_minus_1 = width - 1

    # Forward-path DRAM I/O owner. `load_x`/`store_out` address through the
    # caller's runtime dim/seqlen strides (channels_last-corrected by the
    # caller) rather than `Coord(d, s)`, matching the conv-state ring-buffer's
    # raw_load/raw_store pattern below -- see `VarlenConvIO`'s docstring.
    var io = VarlenConvIO(
        x,
        weight,
        bias,
        output,
        x_dim_stride,
        x_seqlen_stride,
        out_dim_stride,
        out_seqlen_stride,
    )

    # Process each sequence in the batch
    for b in range(batch):
        # Check if this is a padded entry
        if has_cache_indices:
            var cache_idx_val = Int32(cache_indices.raw_load(b))
            if cache_idx_val == pad_slot_id:
                continue

        var seq_start = Int(query_start_loc.raw_load(b))
        var seq_end = Int(query_start_loc.raw_load(b + 1))
        var seqlen = seq_end - seq_start

        # Determine if we should use initial state
        var use_initial_state = False
        if has_initial_state_flag:
            use_initial_state = Bool(has_initial_state.raw_load(b))

        # Get cache index for this batch
        var cache_idx: Int = b
        if has_cache_indices:
            cache_idx = Int(cache_indices.raw_load(b))

        # Process each channel
        for d in range(dim):
            # Load bias
            var bias_val: Scalar[output_dtype] = 0
            if has_bias:
                bias_val = Scalar[output_dtype](io.load_bias(d))

            # Load weights for this channel
            var weights = List[Scalar[weight_dtype]]()
            for w_idx in range(width):
                weights.append(io.load_weight(d, w_idx))

            # Process each position in the sequence
            for l in range(seqlen):
                var conv_sum = bias_val

                # Convolution sum
                for w_idx in range(width):
                    var input_l = l - (width_minus_1 - w_idx)
                    var input_val: Scalar[x_dtype] = 0

                    if input_l >= 0:
                        # Within current sequence
                        input_val = io.load_x(d, seq_start + input_l)
                    elif use_initial_state and has_conv_states:
                        # Use initial state from conv_states
                        var state_idx = (
                            width_minus_1 + input_l
                        )  # Maps negative to state index
                        if state_idx >= 0:
                            var state_offset = (
                                UInt32(cache_idx) * conv_states_batch_stride
                                + UInt32(d) * conv_states_dim_stride
                                + UInt32(state_idx) * conv_states_width_stride
                            )
                            input_val = Scalar[x_dtype](
                                conv_states.raw_load(state_offset)
                            )

                    conv_sum += Scalar[output_dtype](
                        input_val * Scalar[x_dtype](weights[w_idx])
                    )

                # Apply activation
                var out_val = _apply_silu[output_dtype](
                    conv_sum, silu_activation
                )

                # Store output
                io.store_out(d, seq_start + l, out_val)

            # Update conv_states with final state if provided
            if has_conv_states:
                # Copy last (width-1) elements to conv_states
                for s in range(width_minus_1):
                    var src_l = seqlen - width_minus_1 + s
                    var val: Scalar[conv_states_dtype] = 0

                    if src_l >= 0:
                        var x_offset = (
                            UInt32(d) * x_dim_stride
                            + UInt32((seq_start + src_l)) * x_seqlen_stride
                        )
                        val = Scalar[conv_states_dtype](x.raw_load(x_offset))
                    elif use_initial_state:
                        # Carry over from initial state. `src_l` is negative
                        # here, and the same mapping the convolution above uses
                        # for negative positions applies: state index
                        # `width_minus_1 + src_l`, which is in range because
                        # `src_l >= -width_minus_1`.
                        var state_idx = width_minus_1 + src_l
                        if state_idx >= 0:
                            var state_offset = (
                                UInt32(cache_idx) * conv_states_batch_stride
                                + UInt32(d) * conv_states_dim_stride
                                + UInt32(state_idx) * conv_states_width_stride
                            )
                            val = conv_states.raw_load(state_offset)

                    var state_offset = (
                        UInt32(cache_idx) * conv_states_batch_stride
                        + UInt32(d) * conv_states_dim_stride
                        + UInt32(s) * conv_states_width_stride
                    )
                    conv_states.raw_store(state_offset, val)


def causal_conv1d_varlen_update_cpu[
    x_dtype: DType,
    weight_dtype: DType,
    bias_dtype: DType,
    output_dtype: DType,
    conv_state_dtype: DType,
    cache_seqlens_dtype: DType,
    conv_state_indices_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    state_len: Int,
    x: TileTensor[
        mut=False, x_dtype, ...
    ],  # Shape (batch, dim) or (batch, dim, seqlen)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (dim, width)
    bias: TileTensor[mut=False, bias_dtype, ...],  # Shape (dim,)
    conv_state: TileTensor[
        mut=True, conv_state_dtype, ...
    ],  # Shape (batch, dim, state_len)
    cache_seqlens: TileTensor[
        mut=False, cache_seqlens_dtype, ...
    ],  # Shape (batch,)
    conv_state_indices: TileTensor[
        mut=False, conv_state_indices_dtype, ...
    ],  # Shape (batch,)
    output: TileTensor[
        mut=True, output_dtype, ...
    ],  # Shape (batch, dim) or (batch, dim, seqlen)
    x_batch_stride: UInt32,
    x_dim_stride: UInt32,
    x_seqlen_stride: UInt32,
    weight_dim_stride: UInt32,
    weight_width_stride: UInt32,
    conv_state_batch_stride: UInt32,
    conv_state_dim_stride: UInt32,
    conv_state_seqlen_stride: UInt32,
    out_batch_stride: UInt32,
    out_dim_stride: UInt32,
    out_seqlen_stride: UInt32,
    silu_activation: Bool,
    pad_slot_id: Int32,
    has_conv_state_indices: Bool,
    has_cache_seqlens: Bool,
    has_bias: Bool,
):
    """Update function for causal conv1d decode.

    Updates the convolution state and computes output for decode steps.
    Supports circular buffer state management with cache_seqlens.
    """
    var width_minus_1 = width - 1

    for b in range(batch):
        # Check for padded entry
        if has_conv_state_indices:
            var state_idx_val = Int32(conv_state_indices.raw_load(b))
            if state_idx_val == pad_slot_id:
                continue

        # Determine actual batch index for conv_state
        var state_batch_idx = b
        if has_conv_state_indices:
            state_batch_idx = Int(conv_state_indices.raw_load(b))

        for d in range(dim):
            # Load bias
            var bias_val: Scalar[output_dtype] = 0
            if has_bias:
                bias_val = Scalar[output_dtype](bias.raw_load(d))

            # Load weights
            var weights = List[Scalar[weight_dtype]]()
            for w_idx in range(width):
                var weight_offset = (
                    UInt32(d) * weight_dim_stride
                    + UInt32(w_idx) * weight_width_stride
                )
                weights.append(weight.raw_load(weight_offset))

            for l in range(seqlen):
                # Gather input values from state and current x
                var input_vals = List[Scalar[x_dtype]]()
                for w_idx in range(width):
                    var input_val: Scalar[x_dtype] = 0
                    var rel_pos = (
                        w_idx - width_minus_1
                    )  # Ranges from -(width-1) to 0

                    if rel_pos + l < 0:
                        # Read from state
                        var state_pos: Int
                        if has_cache_seqlens:
                            # Circular buffer
                            var cache_seqlen = Int(cache_seqlens.raw_load(b))
                            state_pos = (
                                cache_seqlen + rel_pos + l + state_len
                            ) % state_len
                        else:
                            # Linear buffer: position in state
                            state_pos = width_minus_1 + rel_pos + l

                        if state_pos >= 0 and state_pos < state_len:
                            var state_offset = (
                                UInt32(state_batch_idx)
                                * conv_state_batch_stride
                                + UInt32(d) * conv_state_dim_stride
                                + UInt32(state_pos) * conv_state_seqlen_stride
                            )
                            input_val = Scalar[x_dtype](
                                conv_state.raw_load(state_offset)
                            )
                    else:
                        # Read from current x
                        var x_l = rel_pos + l
                        if x_l >= 0 and x_l < seqlen:
                            var x_offset = (
                                UInt32(b) * x_batch_stride
                                + UInt32(d) * x_dim_stride
                                + UInt32(x_l) * x_seqlen_stride
                            )
                            input_val = x.raw_load(x_offset)

                    input_vals.append(input_val)

                # Compute convolution
                var conv_sum = bias_val
                for w_idx in range(width):
                    conv_sum += Scalar[output_dtype](
                        input_vals[w_idx] * Scalar[x_dtype](weights[w_idx])
                    )

                # Apply activation
                var out_val = _apply_silu[output_dtype](
                    conv_sum, silu_activation
                )

                # Store output
                var out_offset = (
                    UInt32(b) * out_batch_stride
                    + UInt32(d) * out_dim_stride
                    + UInt32(l) * out_seqlen_stride
                )
                output.raw_store(out_offset, out_val)

            # Update state with new x values
            for l in range(seqlen):
                var x_offset = (
                    UInt32(b) * x_batch_stride
                    + UInt32(d) * x_dim_stride
                    + UInt32(l) * x_seqlen_stride
                )
                var x_val = x.raw_load(x_offset)

                var state_pos: Int
                if has_cache_seqlens:
                    # Circular buffer
                    var cache_seqlen = Int(cache_seqlens.raw_load(b))
                    state_pos = (cache_seqlen + l) % state_len
                else:
                    # Shift state left and add new value at end
                    if l == 0:
                        # Shift existing values
                        for s in range(state_len - seqlen):
                            var src_offset = (
                                UInt32(state_batch_idx)
                                * conv_state_batch_stride
                                + UInt32(d) * conv_state_dim_stride
                                + UInt32((s + seqlen))
                                * conv_state_seqlen_stride
                            )
                            var dst_offset = (
                                UInt32(state_batch_idx)
                                * conv_state_batch_stride
                                + UInt32(d) * conv_state_dim_stride
                                + UInt32(s) * conv_state_seqlen_stride
                            )
                            var val = conv_state.raw_load(src_offset)
                            conv_state.raw_store(dst_offset, val)
                    state_pos = state_len - seqlen + l

                var state_offset = (
                    UInt32(state_batch_idx) * conv_state_batch_stride
                    + UInt32(d) * conv_state_dim_stride
                    + UInt32(state_pos) * conv_state_seqlen_stride
                )
                conv_state.raw_store(
                    state_offset, Scalar[conv_state_dtype](x_val)
                )


# ============================================================================
# GPU Kernel Implementations
# ============================================================================


def causal_conv1d_varlen_states_gpu[
    x_dtype: DType,
    cu_seqlens_dtype: DType,
    states_dtype: DType,
    BLOCK_M: Int,
    BLOCK_N: Int,
    x_LT: TensorLayout,
    cu_seqlens_LT: TensorLayout,
    states_LT: TensorLayout,
    x_engine: TensorEngine,
    cu_seqlens_engine: TensorEngine,
    states_engine: TensorEngine,
](
    total_tokens: Int32,
    dim: Int32,
    batch: Int32,
    state_len: Int32,
    x: TileTensor[
        x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine
    ],  # Shape (total_tokens, dim)
    cu_seqlens: TileTensor[
        cu_seqlens_dtype,
        cu_seqlens_LT,
        MutUntrackedOrigin,
        Engine=cu_seqlens_engine,
    ],  # Shape (batch + 1,)
    states: TileTensor[
        states_dtype, states_LT, MutUntrackedOrigin, Engine=states_engine
    ],  # Shape (batch, dim, state_len)
    x_seqlen_stride: UInt32,
    x_dim_stride: UInt32,
    states_batch_stride: UInt32,
    states_dim_stride: UInt32,
    states_seqlen_stride: UInt32,
):
    """GPU kernel for extracting states from variable length sequences.

    Each thread block processes a tile of (BLOCK_M x BLOCK_N) elements.
    Grid dimensions: (ceildiv(dim, BLOCK_N), ceildiv(state_len, BLOCK_M), batch)

    Parameters:
        x_dtype: Data type of input.
        cu_seqlens_dtype: Data type of cumulative sequence lengths.
        states_dtype: Data type of output states.
        BLOCK_M: Tile size for sequence dimension.
        BLOCK_N: Tile size for channel dimension.
        x_LT: Layout type of input tensor.
        cu_seqlens_LT: Layout type of cumulative sequence lengths tensor.
        states_LT: Layout type of output states tensor.
        x_engine: Engine of input tensor.
        cu_seqlens_engine: Engine of cumulative sequence lengths tensor.
        states_engine: Engine of output states tensor.

    Args:
        total_tokens: Total number of tokens.
        dim: Number of channels.
        batch: Number of sequences.
        state_len: State length to extract.
        x: Input tensor.
        cu_seqlens: Cumulative sequence lengths.
        states: Output states tensor.
        x_seqlen_stride: Stride for sequence in x.
        x_dim_stride: Stride for dimension in x.
        states_batch_stride: Stride for batch in states.
        states_dim_stride: Stride for dimension in states.
        states_seqlen_stride: Stride for sequence in states.
    """
    var _total_tokens = Int(total_tokens)
    var _dim = Int(dim)
    var _batch = Int(batch)
    var _state_len = Int(state_len)
    var batch_idx = block_idx.z
    var block_row = block_idx.y
    var block_col = block_idx.x
    var tid_row = thread_idx.y
    var tid_col = thread_idx.x

    # Load sequence boundaries
    var end_idx = Int(cu_seqlens.raw_load(batch_idx + 1))
    var start_idx_seq = Int(cu_seqlens.raw_load(batch_idx))
    var start_idx = max(start_idx_seq, end_idx - _state_len)

    # Calculate row indices (processing from end backwards)
    var row = end_idx - (block_row * BLOCK_M + tid_row + 1)
    var col = block_col * BLOCK_N + tid_col

    # Load value from x if in valid range
    var val: Scalar[states_dtype] = 0
    if row >= start_idx and col < _dim:
        var x_offset = (
            UInt32(row) * x_seqlen_stride + UInt32(col) * x_dim_stride
        )
        val = Scalar[states_dtype](x.raw_load(x_offset))

    # Calculate state row index
    var states_row = _state_len - (block_row * BLOCK_M + tid_row + 1)

    # Store to states if in valid range
    if states_row >= 0 and col < _dim:
        var states_offset = (
            UInt32(batch_idx) * states_batch_stride
            + UInt32(col) * states_dim_stride
            + UInt32(states_row) * states_seqlen_stride
        )
        states.raw_store(states_offset, val)


def causal_conv1d_varlen_fwd_gpu[
    x_dtype: DType,
    weight_dtype: DType,
    bias_dtype: DType,
    output_dtype: DType,
    cu_seqlens_dtype: DType,
    cache_indices_dtype: DType,
    has_initial_state_dtype: DType,
    conv_states_dtype: DType,
    WIDTH: Int,
    BLOCK_DIM: Int,
    BLOCK_SEQ: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    bias_LT: TensorLayout,
    query_start_loc_LT: TensorLayout,
    cache_indices_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
    conv_states_LT: TensorLayout,
    output_LT: TensorLayout,
    x_engine: TensorEngine,
    weight_engine: TensorEngine,
    bias_engine: TensorEngine,
    query_start_loc_engine: TensorEngine,
    cache_indices_engine: TensorEngine,
    has_initial_state_engine: TensorEngine,
    conv_states_engine: TensorEngine,
    output_engine: TensorEngine,
](
    dim: Int32,
    total_seqlen: Int32,
    batch: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine],
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],
    bias: TileTensor[
        bias_dtype, bias_LT, MutUntrackedOrigin, Engine=bias_engine
    ],
    query_start_loc: TileTensor[
        cu_seqlens_dtype,
        query_start_loc_LT,
        MutUntrackedOrigin,
        Engine=query_start_loc_engine,
    ],
    cache_indices: TileTensor[
        cache_indices_dtype,
        cache_indices_LT,
        MutUntrackedOrigin,
        Engine=cache_indices_engine,
    ],
    has_initial_state: TileTensor[
        has_initial_state_dtype,
        has_initial_state_LT,
        MutUntrackedOrigin,
        Engine=has_initial_state_engine,
    ],
    conv_states: TileTensor[
        conv_states_dtype,
        conv_states_LT,
        MutUntrackedOrigin,
        Engine=conv_states_engine,
    ],
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],
    x_dim_stride: UInt32,
    x_seqlen_stride: UInt32,
    weight_dim_stride: UInt32,
    weight_width_stride: UInt32,
    out_dim_stride: UInt32,
    out_seqlen_stride: UInt32,
    conv_states_batch_stride: UInt32,
    conv_states_dim_stride: UInt32,
    conv_states_width_stride: UInt32,
    silu_activation: Int8,
    pad_slot_id: Int32,
    has_cache_indices: Int8,
    has_initial_state_flag: Int8,
    has_conv_states: Int8,
    has_bias: Int8,
):
    """GPU kernel for causal conv1d forward with variable length sequences.

    Grid: (batch, ceildiv(dim, BLOCK_DIM))
    Block: (BLOCK_DIM, BLOCK_SEQ)

    Each block processes BLOCK_DIM channels for one sequence.

    Note: silu_activation and flag parameters are Int8 (0 or 1) instead of Bool
    for DevicePassable compatibility on GPU.
    """
    var _dim = Int(dim)
    var _total_seqlen = Int(total_seqlen)
    var _batch = Int(batch)
    var batch_idx = block_idx.x
    var dim_block_idx = block_idx.y
    var tid = thread_idx.x

    var d = dim_block_idx * BLOCK_DIM + tid

    # Check for padding
    if has_cache_indices != 0:
        var cache_idx_val = Int32(cache_indices.raw_load(batch_idx))
        if cache_idx_val == pad_slot_id:
            return

    # Get sequence bounds
    var seq_start = Int(query_start_loc.raw_load(batch_idx))
    var seq_end = Int(query_start_loc.raw_load(batch_idx + 1))
    var seqlen = seq_end - seq_start

    if d >= _dim:
        return

    # Check for initial state
    var use_initial_state = False
    if has_initial_state_flag != 0:
        use_initial_state = Bool(has_initial_state.raw_load(batch_idx))

    # Get cache index
    var cache_idx: Int = batch_idx
    if has_cache_indices != 0:
        cache_idx = Int(cache_indices.raw_load(batch_idx))

    # Forward-path DRAM I/O owner. `load_x`/`store_out` address through the
    # caller's runtime dim/seqlen strides (channels_last-corrected by the
    # caller) rather than `Coord(d, s)`, matching the conv-state ring-buffer's
    # raw_load/raw_store pattern below -- see `VarlenConvIO`'s docstring.
    var io = VarlenConvIO(
        x,
        weight,
        bias,
        output,
        x_dim_stride,
        x_seqlen_stride,
        out_dim_stride,
        out_seqlen_stride,
    )

    # Load bias
    var bias_val: Scalar[output_dtype] = 0
    if has_bias != 0:
        bias_val = Scalar[output_dtype](io.load_bias(d))

    # Load weights into registers
    var weights = _channel_weights[weight_dtype, WIDTH](weight, d)

    comptime WIDTH_MINUS_1 = WIDTH - 1

    # Process sequence
    for l in range(seqlen):
        var conv_sum = bias_val

        # Gather inputs and compute convolution
        comptime for w_idx in range(WIDTH):
            var input_l = l - (WIDTH_MINUS_1 - w_idx)
            var input_val: Scalar[x_dtype] = 0

            if input_l >= 0:
                input_val = io.load_x(d, seq_start + input_l)
            elif use_initial_state and has_conv_states != 0:
                var state_idx = WIDTH_MINUS_1 + input_l
                if state_idx >= 0:
                    var state_offset = (
                        UInt32(cache_idx) * conv_states_batch_stride
                        + UInt32(d) * conv_states_dim_stride
                        + UInt32(state_idx) * conv_states_width_stride
                    )
                    input_val = Scalar[x_dtype](
                        conv_states.raw_load(state_offset)
                    )

            conv_sum += Scalar[output_dtype](
                input_val * Scalar[x_dtype](weights[w_idx])
            )

        # Apply activation
        var out_val = _apply_silu[output_dtype](conv_sum, silu_activation != 0)

        # Store output
        io.store_out(d, seq_start + l, out_val)

    # Update conv_states
    if has_conv_states != 0:
        comptime for s in range(WIDTH_MINUS_1):
            var src_l = seqlen - WIDTH_MINUS_1 + s
            var val: Scalar[conv_states_dtype] = 0

            if src_l >= 0:
                var x_offset = (
                    UInt32(d) * x_dim_stride
                    + UInt32((seq_start + src_l)) * x_seqlen_stride
                )
                val = Scalar[conv_states_dtype](x.raw_load(x_offset))
            elif use_initial_state:
                # A chunk shorter than WIDTH_MINUS_1 does not contain the whole
                # new state: the oldest entries have to come from the state
                # being continued, under the same negative-position mapping the
                # convolution above uses. Zero here would silently restart the
                # sequence on every decode step.
                #
                # This reads the pool it is writing, which is safe because the
                # index read simplifies to `seqlen + s` -- strictly AHEAD of the
                # `s` being written, and this loop runs `s` upwards. Reordering
                # it would turn the carry-over into a read of a just-written
                # entry.
                var state_idx = WIDTH_MINUS_1 + src_l
                if state_idx >= 0:
                    var prev_offset = (
                        UInt32(cache_idx) * conv_states_batch_stride
                        + UInt32(d) * conv_states_dim_stride
                        + UInt32(state_idx) * conv_states_width_stride
                    )
                    val = conv_states.raw_load(prev_offset)

            var state_offset = (
                UInt32(cache_idx) * conv_states_batch_stride
                + UInt32(d) * conv_states_dim_stride
                + UInt32(s) * conv_states_width_stride
            )
            conv_states.raw_store(state_offset, val)


# Outputs per steady-state trip of the seq-parallel prefill kernel below.
# The register sliding window reduces each output to one global load; the
# UNROLL-wide trip keeps that many independent loads in flight so the walk
# runs at bandwidth instead of at HBM-latency rate. 16 was measured on B200
# at the Inkling prefill shapes; other parts are untuned.
comptime _CONV1D_SEQPARALLEL_UNROLL: Int = 16


def causal_conv1d_varlen_fwd_seqparallel_gpu[
    x_dtype: DType,
    weight_dtype: DType,
    bias_dtype: DType,
    output_dtype: DType,
    cu_seqlens_dtype: DType,
    cache_indices_dtype: DType,
    has_initial_state_dtype: DType,
    conv_states_dtype: DType,
    WIDTH: Int,
    BLOCK_DIM: Int,
    TILE_SEQ: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    bias_LT: TensorLayout,
    query_start_loc_LT: TensorLayout,
    cache_indices_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
    conv_states_LT: TensorLayout,
    output_LT: TensorLayout,
    x_engine: TensorEngine,
    weight_engine: TensorEngine,
    bias_engine: TensorEngine,
    query_start_loc_engine: TensorEngine,
    cache_indices_engine: TensorEngine,
    has_initial_state_engine: TensorEngine,
    conv_states_engine: TensorEngine,
    output_engine: TensorEngine,
](
    dim_dev: Int32,
    total_seqlen_dev: Int32,
    batch_dev: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine],
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],
    bias: TileTensor[
        bias_dtype, bias_LT, MutUntrackedOrigin, Engine=bias_engine
    ],
    query_start_loc: TileTensor[
        cu_seqlens_dtype,
        query_start_loc_LT,
        MutUntrackedOrigin,
        Engine=query_start_loc_engine,
    ],
    cache_indices: TileTensor[
        cache_indices_dtype,
        cache_indices_LT,
        MutUntrackedOrigin,
        Engine=cache_indices_engine,
    ],
    has_initial_state: TileTensor[
        has_initial_state_dtype,
        has_initial_state_LT,
        MutUntrackedOrigin,
        Engine=has_initial_state_engine,
    ],
    conv_states: TileTensor[
        conv_states_dtype,
        conv_states_LT,
        MutUntrackedOrigin,
        Engine=conv_states_engine,
    ],
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],
    x_dim_stride: UInt32,
    x_seqlen_stride: UInt32,
    weight_dim_stride: UInt32,
    weight_width_stride: UInt32,
    out_dim_stride: UInt32,
    out_seqlen_stride: UInt32,
    conv_states_batch_stride: UInt32,
    conv_states_dim_stride: UInt32,
    conv_states_width_stride: UInt32,
    silu_activation: Int8,
    pad_slot_id: Int32,
    has_cache_indices: Int8,
    has_initial_state_flag: Int8,
    has_conv_states: Int8,
    has_bias: Int8,
):
    """GPU kernel for causal conv1d forward with variable length sequences,
    sequence-parallel prefill variant (NVIDIA B200/sm_100, generic elsewhere).

    Grid: (batch, ceildiv(dim, BLOCK_DIM), num_tiles_ub)
    Block: (BLOCK_DIM, 1)

    Each grid-(x,y,z) block handles one (sequence, channel-tile, seq-tile).
    The z-dimension tiles the sequence into TILE_SEQ-sized chunks so a long
    prefill sequence is spread across many blocks instead of walking the
    whole sequence serially in one thread (see `causal_conv1d_varlen_fwd_gpu`,
    which stays byte-identical and is reused verbatim for decode). Depthwise
    conv (WIDTH<=4) has no cross-position recurrence: the causal gather reads
    GLOBAL read-only `x` across tile boundaries, so tiles need no shared
    memory or cross-block synchronization. `conv_states` is written exactly
    once, by the tail tile of each sequence (`tile_end == seqlen`).

    Dispatched from `CausalConv1DVarlenFwd.launch_gpu` only when
    `total_seqlen > batch` (i.e. at least one multi-token prefill segment is
    present); pure decode (`total_seqlen == batch`) keeps using the serial
    kernel above unmodified. Per
    `Kernels/claude_kb` patterns/kv-buffer-pipeline-style host-vs-device
    tiling notes: `num_tiles_ub` is a safe host-side upper bound
    (`ceildiv(total_seqlen, TILE_SEQ)`; grid-z indexes each sequence's LOCAL
    tile and no sequence exceeds total_seqlen) that avoids a host-side
    max-reduction over ragged per-sequence lengths; blocks whose z-index
    exceeds a given sequence's actual tile count early-return.

    Note: silu_activation and flag parameters are Int8 (0 or 1) instead of Bool
    for DevicePassable compatibility on GPU.
    """
    # `Int` is not device-passable; widen the fixed-width args. Only `dim` is
    # read in this variant; the other two match the serial kernel's signature.
    var dim = Int(dim_dev)
    _ = total_seqlen_dev
    _ = batch_dev
    var batch_idx = block_idx.x
    var dim_block_idx = block_idx.y
    var tid = thread_idx.x

    var d = dim_block_idx * BLOCK_DIM + tid

    # Check for padding
    if has_cache_indices != 0:
        var cache_idx_val = Int32(cache_indices.raw_load(batch_idx))
        if cache_idx_val == pad_slot_id:
            return

    # Get sequence bounds
    var seq_start = Int(query_start_loc.raw_load(batch_idx))
    var seq_end = Int(query_start_loc.raw_load(batch_idx + 1))
    var seqlen = seq_end - seq_start

    # Grid-z tiling: each z-slice covers TILE_SEQ consecutive positions of
    # this sequence. Tile 0 is kept alive even for an empty sequence so it
    # can still reach the epilogue below and zero conv_states.
    var local_tile = Int(block_idx.z)
    var num_tiles_this_seq = ceildiv(seqlen, TILE_SEQ)
    if local_tile >= max(num_tiles_this_seq, 1):
        return
    var tile_start = local_tile * TILE_SEQ
    var tile_end = min(tile_start + TILE_SEQ, seqlen)

    if d >= dim:
        return

    # Check for initial state
    var use_initial_state = False
    if has_initial_state_flag != 0:
        use_initial_state = Bool(has_initial_state.raw_load(batch_idx))

    # Get cache index
    var cache_idx: Int = batch_idx
    if has_cache_indices != 0:
        cache_idx = Int(cache_indices.raw_load(batch_idx))

    # Load bias
    var bias_val: Scalar[output_dtype] = 0
    if has_bias != 0:
        bias_val = Scalar[output_dtype](bias.raw_load(d))

    # Load weights into registers
    var weights = _channel_weights[weight_dtype, WIDTH](weight, d)

    comptime WIDTH_MINUS_1 = WIDTH - 1
    comptime UNROLL = _CONV1D_SEQPARALLEL_UNROLL
    comptime TAP_LANES = _TAP_LANES[WIDTH]
    var tap_weights = weights.cast[x_dtype]()

    # Register sliding window over the WIDTH-1 inputs preceding the current
    # position, so the steady state costs one global load per output instead
    # of WIDTH. The window is preloaded once per tile: positions inside the
    # sequence come from global read-only x (tiles may overlap reads, never
    # writes), and negative positions fall back to the continuing sequence's
    # initial state (state slot WIDTH_MINUS_1 + pos of its pool entry).
    var win = Array[Scalar[x_dtype], WIDTH_MINUS_1](fill=0)
    comptime for i in range(WIDTH_MINUS_1):
        var pos = tile_start - WIDTH_MINUS_1 + i
        var v: Scalar[x_dtype] = 0
        if pos >= 0:
            v = x.raw_load(
                UInt32(d) * x_dim_stride
                + UInt32((seq_start + pos)) * x_seqlen_stride
            )
        elif use_initial_state and has_conv_states != 0:
            var state_idx = WIDTH_MINUS_1 + pos
            if state_idx >= 0:
                v = Scalar[x_dtype](
                    conv_states.raw_load(
                        UInt32(cache_idx) * conv_states_batch_stride
                        + UInt32(d) * conv_states_dim_stride
                        + UInt32(state_idx) * conv_states_width_stride
                    )
                )
        win[i] = v

    # One U-output trip. The steady state calls this with U=UNROLL so the U
    # loads share no dependencies and can all be in flight; the tile remainder
    # reuses it with U=1.
    def _emit_chunk[U: Int](tile_off: Int) {mut win, imm}:
        var l0 = tile_start + tile_off
        # Carried window followed by this trip's loads, so `taps[j]` is
        # position `l0 - WIDTH_MINUS_1 + j` throughout and output `u` folds
        # `taps[u ..< u + WIDTH]` against the taps.
        var taps = Array[Scalar[x_dtype], WIDTH_MINUS_1 + U](fill=0)
        comptime for i in range(WIDTH_MINUS_1):
            taps[i] = win[i]
        comptime for u in range(U):
            taps[WIDTH_MINUS_1 + u] = x.raw_load(
                UInt32(d) * x_dim_stride
                + UInt32((seq_start + l0 + u)) * x_seqlen_stride
            )

        comptime for u in range(U):
            var window = SIMD[x_dtype, TAP_LANES](0)
            comptime for w in range(WIDTH):
                window[w] = taps[u + w]
            var conv_sum = (
                bias_val
                + (window * tap_weights).reduce_add().cast[output_dtype]()
            )

            # Apply activation
            var out_val = _apply_silu[output_dtype](
                conv_sum, silu_activation != 0
            )

            # Store output
            var out_offset = (
                UInt32(d) * out_dim_stride
                + UInt32((seq_start + l0 + u)) * out_seqlen_stride
            )
            output.raw_store(out_offset, out_val)

        # Carry the last WIDTH-1 taps into the next trip.
        comptime for i in range(WIDTH_MINUS_1):
            win[i] = taps[U + i]

    vectorize[UNROLL](tile_end - tile_start, _emit_chunk)

    # Update conv_states exactly once, from the tail tile of this sequence.
    if has_conv_states != 0 and tile_end == seqlen:
        comptime for s in range(WIDTH_MINUS_1):
            var src_l = seqlen - WIDTH_MINUS_1 + s
            var val: Scalar[conv_states_dtype] = 0

            if src_l >= 0:
                var x_offset = (
                    UInt32(d) * x_dim_stride
                    + UInt32((seq_start + src_l)) * x_seqlen_stride
                )
                val = Scalar[conv_states_dtype](x.raw_load(x_offset))
            elif use_initial_state:
                # See the same carry-over in `causal_conv1d_varlen_fwd_gpu`: a
                # chunk shorter than WIDTH_MINUS_1 takes its oldest state
                # entries from the state being continued, and the index read is
                # `seqlen + s`, ahead of the `s` written here.
                var state_idx = WIDTH_MINUS_1 + src_l
                if state_idx >= 0:
                    var prev_offset = (
                        UInt32(cache_idx) * conv_states_batch_stride
                        + UInt32(d) * conv_states_dim_stride
                        + UInt32(state_idx) * conv_states_width_stride
                    )
                    val = conv_states.raw_load(prev_offset)

            var state_offset = (
                UInt32(cache_idx) * conv_states_batch_stride
                + UInt32(d) * conv_states_dim_stride
                + UInt32(s) * conv_states_width_stride
            )
            conv_states.raw_store(state_offset, val)


def causal_conv1d_varlen_update_gpu[
    x_dtype: DType,
    weight_dtype: DType,
    bias_dtype: DType,
    output_dtype: DType,
    conv_state_dtype: DType,
    cache_seqlens_dtype: DType,
    conv_state_indices_dtype: DType,
    WIDTH: Int,
    BLOCK_DIM: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    bias_LT: TensorLayout,
    conv_state_LT: TensorLayout,
    cache_seqlens_LT: TensorLayout,
    conv_state_indices_LT: TensorLayout,
    output_LT: TensorLayout,
    x_engine: TensorEngine,
    weight_engine: TensorEngine,
    bias_engine: TensorEngine,
    conv_state_engine: TensorEngine,
    cache_seqlens_engine: TensorEngine,
    conv_state_indices_engine: TensorEngine,
    output_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    state_len: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine],
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],
    bias: TileTensor[
        bias_dtype, bias_LT, MutUntrackedOrigin, Engine=bias_engine
    ],
    conv_state: TileTensor[
        conv_state_dtype,
        conv_state_LT,
        MutUntrackedOrigin,
        Engine=conv_state_engine,
    ],
    cache_seqlens: TileTensor[
        cache_seqlens_dtype,
        cache_seqlens_LT,
        MutUntrackedOrigin,
        Engine=cache_seqlens_engine,
    ],
    conv_state_indices: TileTensor[
        conv_state_indices_dtype,
        conv_state_indices_LT,
        MutUntrackedOrigin,
        Engine=conv_state_indices_engine,
    ],
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],
    x_batch_stride: UInt32,
    x_dim_stride: UInt32,
    x_seqlen_stride: UInt32,
    weight_dim_stride: UInt32,
    weight_width_stride: UInt32,
    conv_state_batch_stride: UInt32,
    conv_state_dim_stride: UInt32,
    conv_state_seqlen_stride: UInt32,
    out_batch_stride: UInt32,
    out_dim_stride: UInt32,
    out_seqlen_stride: UInt32,
    silu_activation: Int8,
    pad_slot_id: Int32,
    has_conv_state_indices: Int8,
    has_cache_seqlens: Int8,
    has_bias: Int8,
):
    """GPU kernel for causal conv1d update (decode step).

    Grid: (batch, ceildiv(dim, BLOCK_DIM))
    Block: (BLOCK_DIM,)

    Note: silu_activation and flag parameters are Int8 (0 or 1) instead of Bool
    for DevicePassable compatibility on GPU.
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _state_len = Int(state_len)
    var batch_idx = block_idx.x
    var dim_block_idx = block_idx.y
    var tid = thread_idx.x

    var d = dim_block_idx * BLOCK_DIM + tid

    # Check for padding
    if has_conv_state_indices != 0:
        var state_idx_val = Int32(conv_state_indices.raw_load(batch_idx))
        if state_idx_val == pad_slot_id:
            return

    if d >= _dim:
        return

    # Get state _batch index
    var state_batch_idx: Int = batch_idx
    if has_conv_state_indices != 0:
        state_batch_idx = Int(conv_state_indices.raw_load(batch_idx))

    # Load bias
    var bias_val: Scalar[output_dtype] = 0
    if has_bias != 0:
        bias_val = Scalar[output_dtype](bias.raw_load(d))

    # Load weights
    var weights = _channel_weights[weight_dtype, WIDTH](weight, d)

    comptime WIDTH_MINUS_1 = WIDTH - 1

    for l in range(_seqlen):
        # Get cache position
        var cache_offset = 0
        if has_cache_seqlens != 0:
            var cache_seqlen = Int(cache_seqlens.raw_load(batch_idx))
            cache_offset = cache_seqlen

        # Gather inputs and compute
        var conv_sum = bias_val

        comptime for w_idx in range(WIDTH):
            var rel_pos = w_idx - WIDTH_MINUS_1
            var input_val: Scalar[x_dtype] = 0

            if rel_pos + l < 0:
                # From state
                var state_pos: Int
                if has_cache_seqlens != 0:
                    state_pos = (
                        cache_offset + rel_pos + l + _state_len
                    ) % _state_len
                else:
                    state_pos = WIDTH_MINUS_1 + rel_pos + l

                if state_pos >= 0 and state_pos < _state_len:
                    var state_offset = (
                        UInt32(state_batch_idx) * conv_state_batch_stride
                        + UInt32(d) * conv_state_dim_stride
                        + UInt32(state_pos) * conv_state_seqlen_stride
                    )
                    input_val = Scalar[x_dtype](
                        conv_state.raw_load(state_offset)
                    )
            else:
                # From x
                var x_l = rel_pos + l
                if x_l >= 0 and x_l < _seqlen:
                    var x_offset = (
                        UInt32(batch_idx) * x_batch_stride
                        + UInt32(d) * x_dim_stride
                        + UInt32(x_l) * x_seqlen_stride
                    )
                    input_val = x.raw_load(x_offset)

            conv_sum += Scalar[output_dtype](
                input_val * Scalar[x_dtype](weights[w_idx])
            )
        # Apply activation
        var out_val = _apply_silu[output_dtype](conv_sum, silu_activation != 0)

        # Store output
        var out_offset = (
            UInt32(batch_idx) * out_batch_stride
            + UInt32(d) * out_dim_stride
            + UInt32(l) * out_seqlen_stride
        )
        output.raw_store(out_offset, out_val)

        # Update state
        var x_offset = (
            UInt32(batch_idx) * x_batch_stride
            + UInt32(d) * x_dim_stride
            + UInt32(l) * x_seqlen_stride
        )
        var x_val = x.raw_load(x_offset)

        var state_pos: Int
        if has_cache_seqlens != 0:
            state_pos = (cache_offset + l) % _state_len
        else:
            state_pos = _state_len - _seqlen + l

        var state_offset = (
            UInt32(state_batch_idx) * conv_state_batch_stride
            + UInt32(d) * conv_state_dim_stride
            + UInt32(state_pos) * conv_state_seqlen_stride
        )
        conv_state.raw_store(state_offset, Scalar[conv_state_dtype](x_val))
