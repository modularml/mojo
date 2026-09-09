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

from layout import TileTensor, row_major
from std.random import rand
from state_space.varlen_causal_conv1d import (
    causal_conv1d_varlen_fwd_cpu,
    causal_conv1d_varlen_update_cpu,
    causal_conv1d_varlen_states_cpu,
)
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index, IndexList


# Constants
comptime PAD_SLOT_ID: Int32 = -1


@always_inline
def silu_ref[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Reference SiLU implementation: x * sigmoid(x) = x / (1 + exp(-x))."""
    var x_f32 = x.cast[.float32]()
    var neg_x = -x_f32
    var exp_neg_x = exp(neg_x)
    var one = Float32(1.0)
    var sigmoid_x = one / (one + exp_neg_x)
    return (x_f32 * sigmoid_x).cast[dtype]()


def run_varlen_causal_conv1d_fwd[
    dtype: DType,
    activation: StaticString,
](
    batch: Int,
    dim: Int,
    seq_lengths: IndexList,
    width: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test varlen causal conv1d forward kernel against reference implementation.
    """
    # Calculate total_seqlen (sum of all sequence lengths)
    var total_seqlen = 0
    for i in range(batch):
        total_seqlen += seq_lengths[i]

    # Allocate host memory as TileTensors over their backing heaps.
    # x: (dim, total_seqlen) for varlen - sequences concatenated
    var x_heap = List(length=dim * total_seqlen, fill=Scalar[dtype](0))
    var x_tt = TileTensor(x_heap, row_major(dim, total_seqlen))

    # weight: (dim, width)
    var weight_heap = List(length=dim * width, fill=Scalar[dtype](0))
    var weight_tt = TileTensor(weight_heap, row_major(dim, width))

    # bias: (dim,)
    var bias_heap = List(length=dim, fill=Scalar[dtype](0))
    var bias_tt = TileTensor(
        bias_heap,
        row_major(
            dim,
        ),
    )

    # query_start_loc: (batch + 1,) - cumulative sequence lengths
    var query_start_loc_heap = List(length=batch + 1, fill=Int32(0))
    var query_start_loc_tt = TileTensor(
        query_start_loc_heap,
        row_major(
            batch + 1,
        ),
    )
    var cumsum = 0
    query_start_loc_tt.raw_store(0, Int32(0))
    for i in range(batch):
        cumsum += seq_lengths[i]
        query_start_loc_tt.raw_store(i + 1, Int32(cumsum))

    # cache_indices: (batch,) - identity mapping
    var cache_indices_heap = List(length=batch, fill=Int32(0))
    var cache_indices_tt = TileTensor(
        cache_indices_heap,
        row_major(
            batch,
        ),
    )
    for i in range(batch):
        cache_indices_tt.raw_store(i, Int32(i))

    # has_initial_state: (batch,) - all False
    var has_initial_state_heap = List(length=batch, fill=Scalar[.bool](False))
    var has_initial_state_tt = TileTensor(
        has_initial_state_heap,
        row_major(
            batch,
        ),
    )

    # conv_states: (batch, dim, width - 1)
    var state_len = width - 1
    var conv_states_heap = List(
        length=batch * dim * state_len, fill=Scalar[dtype](0)
    )
    var conv_states_tt = TileTensor(
        conv_states_heap,
        row_major(batch, dim, state_len),
    )

    # output: (dim, total_seqlen)
    var output_heap = List(length=dim * total_seqlen, fill=Scalar[dtype](0))
    var output_tt = TileTensor(output_heap, row_major(dim, total_seqlen))

    # reference output: (dim, total_seqlen)
    var output_ref_heap = List(length=dim * total_seqlen, fill=Scalar[dtype](0))
    var output_ref_tt = TileTensor(
        output_ref_heap, row_major(dim, total_seqlen)
    )

    # Initialize input data
    rand[dtype](x_tt._storage, dim * total_seqlen)
    rand[dtype](weight_tt._storage, dim * width)
    rand[dtype](bias_tt._storage, dim)

    var x_buf = x_tt
    var weight_buf = weight_tt
    var bias_buf = bias_tt
    var query_start_loc_buf = query_start_loc_tt
    var output_ref_buf = output_ref_tt

    # Strides for row-major layout
    var x_dim_stride: UInt32 = UInt32(total_seqlen)
    var x_seqlen_stride: UInt32 = 1
    var weight_dim_stride: UInt32 = UInt32(width)
    var weight_width_stride: UInt32 = 1
    var out_dim_stride: UInt32 = UInt32(total_seqlen)
    var out_seqlen_stride: UInt32 = 1
    var conv_states_batch_stride: UInt32 = UInt32(dim * state_len)
    var conv_states_dim_stride: UInt32 = UInt32(state_len)
    var conv_states_width_stride: UInt32 = 1

    var silu_activation = activation == "silu"

    # Test kernel
    causal_conv1d_varlen_fwd_cpu[
        dtype,
        dtype,
        dtype,
        dtype,
        DType.int32,
        DType.int32,
        DType.bool,
        dtype,
    ](
        dim,
        total_seqlen,
        width,
        batch,
        x_tt,
        weight_tt,
        bias_tt,
        query_start_loc_tt,
        cache_indices_tt,
        has_initial_state_tt,
        conv_states_tt,
        output_tt,
        x_dim_stride,
        x_seqlen_stride,
        weight_dim_stride,
        weight_width_stride,
        out_dim_stride,
        out_seqlen_stride,
        conv_states_batch_stride,
        conv_states_dim_stride,
        conv_states_width_stride,
        silu_activation,
        PAD_SLOT_ID,
        True,  # has_cache_indices
        True,  # has_initial_state_flag
        True,  # has_conv_states
        True,  # has_bias
    )

    # Reference implementation
    var width_minus_1: Int = width - 1
    for b in range(batch):
        var seq_start = Int(query_start_loc_buf.raw_load(b))
        var seq_end = Int(query_start_loc_buf.raw_load(b + 1))
        var seqlen = seq_end - seq_start

        for d in range(dim):
            var bias_val = bias_buf.raw_load(d)

            for l in range(seqlen):
                var conv_sum: Scalar[dtype] = bias_val

                for w_idx in range(width):
                    var input_l = l - (width_minus_1 - w_idx)
                    var input_val: Scalar[dtype] = Scalar[dtype](0.0)

                    if input_l >= 0:
                        var x_offset = (
                            UInt32(d) * x_dim_stride
                            + UInt32((seq_start + input_l)) * x_seqlen_stride
                        )
                        input_val = x_buf.raw_load(x_offset)

                    var weight_offset = (
                        UInt32(d) * weight_dim_stride
                        + UInt32(w_idx) * weight_width_stride
                    )
                    var weight_val = weight_buf.raw_load(weight_offset)
                    conv_sum = conv_sum + input_val * weight_val

                var out_val = conv_sum
                if silu_activation:
                    out_val = silu_ref[dtype](out_val)

                var out_offset = (
                    UInt32(d) * out_dim_stride
                    + UInt32((seq_start + l)) * out_seqlen_stride
                )
                output_ref_buf.raw_store(out_offset, out_val)

    # Compare results
    var flattened_size = dim * total_seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            output_tt._storage[i],
            output_ref_tt._storage[i],
            rtol=rtol,
        )


def run_varlen_causal_conv1d_update[
    dtype: DType,
    activation: StaticString,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    state_len: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test varlen causal conv1d update kernel against reference implementation.
    """
    # Allocate host memory as TileTensors over their backing heaps.
    # x: (batch, dim, seqlen)
    var x_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var x_tt2 = TileTensor(x_heap, row_major(batch, dim, seqlen))

    # weight: (dim, width)
    var weight_heap = List(length=dim * width, fill=Scalar[dtype](0))
    var weight_tt2 = TileTensor(weight_heap, row_major(dim, width))

    # bias: (dim,)
    var bias_heap = List(length=dim, fill=Scalar[dtype](0))
    var bias_tt2 = TileTensor(
        bias_heap,
        row_major(
            dim,
        ),
    )

    # conv_state: (batch, dim, state_len)
    var conv_state_heap = List(
        length=batch * dim * state_len, fill=Scalar[dtype](0)
    )
    var conv_state_tt2 = TileTensor(
        conv_state_heap,
        row_major(batch, dim, state_len),
    )

    # cache_seqlens: (batch,) - can be empty
    var cache_seqlens_heap = List(length=batch, fill=Int32(0))
    var cache_seqlens_tt = TileTensor(
        cache_seqlens_heap,
        row_major(
            batch,
        ),
    )

    # conv_state_indices: (batch,) - identity mapping
    var conv_state_indices_heap = List(length=batch, fill=Int32(0))
    var conv_state_indices_tt = TileTensor(
        conv_state_indices_heap,
        row_major(
            batch,
        ),
    )
    for i in range(batch):
        conv_state_indices_tt.raw_store(i, Int32(i))

    # output: (batch, dim, seqlen)
    var output_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var output_tt2 = TileTensor(output_heap, row_major(batch, dim, seqlen))

    # reference output: (batch, dim, seqlen)
    var output_ref_heap = List(
        length=batch * dim * seqlen, fill=Scalar[dtype](0)
    )
    var output_ref_tt = TileTensor(
        output_ref_heap, row_major(batch, dim, seqlen)
    )

    # Copy of conv_state for reference
    var conv_state_ref_heap = List(
        length=batch * dim * state_len, fill=Scalar[dtype](0)
    )
    var conv_state_ref_tt = TileTensor(
        conv_state_ref_heap,
        row_major(batch, dim, state_len),
    )

    # Initialize input data
    rand[dtype](x_tt2._storage, batch * dim * seqlen)
    rand[dtype](conv_state_tt2._storage, batch * dim * state_len)
    rand[dtype](weight_tt2._storage, dim * width)
    rand[dtype](bias_tt2._storage, dim)

    # Copy conv_state for reference
    for i in range(batch * dim * state_len):
        conv_state_ref_tt._storage[i] = conv_state_tt2._storage[i]

    var x_buf = x_tt2
    var weight_buf = weight_tt2
    var bias_buf = bias_tt2
    var cache_seqlens_buf = cache_seqlens_tt
    var output_ref_buf = output_ref_tt
    var conv_state_ref_buf = conv_state_ref_tt

    # Strides for row-major layout
    var x_batch_stride: UInt32 = UInt32(dim * seqlen)
    var x_dim_stride: UInt32 = UInt32(seqlen)
    var x_seqlen_stride: UInt32 = 1
    var weight_dim_stride: UInt32 = UInt32(width)
    var weight_width_stride: UInt32 = 1
    var conv_state_batch_stride: UInt32 = UInt32(dim * state_len)
    var conv_state_dim_stride: UInt32 = UInt32(state_len)
    var conv_state_seqlen_stride: UInt32 = 1
    var out_batch_stride: UInt32 = UInt32(dim * seqlen)
    var out_dim_stride: UInt32 = UInt32(seqlen)
    var out_seqlen_stride: UInt32 = 1

    var silu_activation = activation == "silu"

    # Test kernel
    causal_conv1d_varlen_update_cpu[
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        DType.int32,
        DType.int32,
    ](
        batch,
        dim,
        seqlen,
        width,
        state_len,
        x_tt2,
        weight_tt2,
        bias_tt2,
        conv_state_tt2,
        cache_seqlens_tt,
        conv_state_indices_tt,
        output_tt2,
        x_batch_stride,
        x_dim_stride,
        x_seqlen_stride,
        weight_dim_stride,
        weight_width_stride,
        conv_state_batch_stride,
        conv_state_dim_stride,
        conv_state_seqlen_stride,
        out_batch_stride,
        out_dim_stride,
        out_seqlen_stride,
        silu_activation,
        PAD_SLOT_ID,
        True,  # has_conv_state_indices
        True,  # has_cache_seqlens
        True,  # has_bias
    )

    # Reference implementation
    var width_minus_1: Int = width - 1
    for b in range(batch):
        var state_batch_idx = b

        for d in range(dim):
            var bias_val = bias_buf.raw_load(d)

            for l in range(seqlen):
                var conv_sum: Scalar[dtype] = bias_val

                for w_idx in range(width):
                    var rel_pos = w_idx - width_minus_1
                    var input_val: Scalar[dtype] = Scalar[dtype](0.0)

                    if rel_pos + l < 0:
                        # Read from state
                        var state_pos: Int
                        # has_cache_seqlens is True in our test, so use circular buffer
                        var cache_seqlen = Int(cache_seqlens_buf.raw_load(b))
                        state_pos = (
                            cache_seqlen + rel_pos + l + state_len
                        ) % state_len

                        if state_pos >= 0 and state_pos < state_len:
                            var state_offset = (
                                UInt32(state_batch_idx)
                                * conv_state_batch_stride
                                + UInt32(d) * conv_state_dim_stride
                                + UInt32(state_pos) * conv_state_seqlen_stride
                            )
                            input_val = conv_state_ref_buf.raw_load(
                                state_offset
                            )
                    else:
                        # Read from x
                        var x_l = rel_pos + l
                        if x_l >= 0 and x_l < seqlen:
                            var x_offset = (
                                UInt32(b) * x_batch_stride
                                + UInt32(d) * x_dim_stride
                                + UInt32(x_l) * x_seqlen_stride
                            )
                            input_val = x_buf.raw_load(x_offset)

                    var weight_offset = (
                        UInt32(d) * weight_dim_stride
                        + UInt32(w_idx) * weight_width_stride
                    )
                    var weight_val = weight_buf.raw_load(weight_offset)
                    conv_sum = conv_sum + input_val * weight_val

                var out_val = conv_sum
                if silu_activation:
                    out_val = silu_ref[dtype](out_val)

                var out_offset = (
                    UInt32(b) * out_batch_stride
                    + UInt32(d) * out_dim_stride
                    + UInt32(l) * out_seqlen_stride
                )
                output_ref_buf.raw_store(out_offset, out_val)

            # Update state with new x values
            # This matches the CPU implementation logic exactly
            for l in range(seqlen):
                var x_offset = (
                    UInt32(b) * x_batch_stride
                    + UInt32(d) * x_dim_stride
                    + UInt32(l) * x_seqlen_stride
                )
                var x_val = x_buf.raw_load(x_offset)

                var state_pos: Int
                # has_cache_seqlens is True in our test, so use circular buffer
                var cache_seqlen = Int(cache_seqlens_buf.raw_load(b))
                state_pos = (cache_seqlen + l) % state_len

                var state_offset = (
                    UInt32(state_batch_idx) * conv_state_batch_stride
                    + UInt32(d) * conv_state_dim_stride
                    + UInt32(state_pos) * conv_state_seqlen_stride
                )
                conv_state_ref_buf.raw_store(state_offset, x_val)

    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            output_tt2._storage[i],
            output_ref_tt._storage[i],
            rtol=rtol,
        )

    # Compare conv_state updates
    var conv_state_size = batch * dim * state_len
    for i in range(conv_state_size):
        assert_almost_equal(
            conv_state_tt2._storage[i],
            conv_state_ref_tt._storage[i],
            rtol=rtol,
        )


def run_varlen_causal_conv1d_states[
    dtype: DType,
](
    batch: Int,
    dim: Int,
    seq_lengths: IndexList,
    state_len: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test varlen causal conv1d states extraction kernel."""
    # Calculate total_tokens (sum of all sequence lengths)
    var total_tokens = 0
    for i in range(batch):
        total_tokens += seq_lengths[i]

    # Allocate host memory as TileTensors over their backing heaps.
    # x: (total_tokens, dim) - sequences concatenated
    var x_heap = List(length=total_tokens * dim, fill=Scalar[dtype](0))
    var x_tt3 = TileTensor(x_heap, row_major(total_tokens, dim))

    # cu_seqlens: (batch + 1,) - cumulative sequence lengths
    var cu_seqlens_heap = List(length=batch + 1, fill=Int32(0))
    var cu_seqlens_tt = TileTensor(
        cu_seqlens_heap,
        row_major(
            batch + 1,
        ),
    )
    var cumsum = 0
    cu_seqlens_tt.raw_store(0, Int32(0))
    for i in range(batch):
        cumsum += seq_lengths[i]
        cu_seqlens_tt.raw_store(i + 1, Int32(cumsum))

    # states: (batch, dim, state_len)
    var states_heap = List(
        length=batch * dim * state_len, fill=Scalar[dtype](0)
    )
    var states_tt = TileTensor(states_heap, row_major(batch, dim, state_len))

    # reference states: (batch, dim, state_len)
    var states_ref_heap = List(
        length=batch * dim * state_len, fill=Scalar[dtype](0)
    )
    var states_ref_tt = TileTensor(
        states_ref_heap, row_major(batch, dim, state_len)
    )

    # Initialize input data
    rand[dtype](x_tt3._storage, total_tokens * dim)

    var x_buf = x_tt3
    var cu_seqlens_buf = cu_seqlens_tt
    var states_ref_buf = states_ref_tt

    # Strides for row-major layout
    var x_seqlen_stride: UInt32 = UInt32(dim)
    var x_dim_stride: UInt32 = 1
    var states_batch_stride: UInt32 = UInt32(dim * state_len)
    var states_dim_stride: UInt32 = UInt32(state_len)
    var states_seqlen_stride: UInt32 = 1

    # Test kernel
    causal_conv1d_varlen_states_cpu[
        dtype,
        DType.int32,
        dtype,
    ](
        total_tokens,
        dim,
        batch,
        state_len,
        x_tt3,
        cu_seqlens_tt,
        states_tt,
        x_seqlen_stride,
        x_dim_stride,
        states_batch_stride,
        states_dim_stride,
        states_seqlen_stride,
    )

    # Reference implementation
    for b in range(batch):
        var end_idx = Int(cu_seqlens_buf.raw_load(b + 1))
        var start_idx_seq = Int(cu_seqlens_buf.raw_load(b))
        var start_idx = max(start_idx_seq, end_idx - state_len)
        var num_elements = end_idx - start_idx

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
                var val = x_buf.raw_load(x_offset)
                states_ref_buf.raw_store(states_offset, val)

    # Compare results
    var flattened_size = batch * dim * state_len
    for i in range(flattened_size):
        assert_almost_equal(
            states_tt._storage[i],
            states_ref_tt._storage[i],
            rtol=rtol,
        )


def run_conv_state_writeback[
    dtype: DType,
](batch: Int, dim: Int, width: Int, seqlen: Int) raises:
    """Assert the state a call leaves behind, not the output it returns.

    The contract is a sliding window: after consuming a chunk, the pool holds
    the last `width - 1` tokens of everything seen so far. When the chunk is
    SHORTER than that -- every decode step, where it is one token -- the older
    entries have to come from the state being continued, and only an assertion
    on the pool itself can tell that apart from zeros. The output of a single
    call cannot: it depends on the state read, which was always correct.

    The pool starts non-zero and `has_initial_state` is true, so this stands in
    for a request that has already been prefilled.
    """
    var state_len = width - 1
    var total_seqlen = batch * seqlen

    var x_heap = List(length=dim * total_seqlen, fill=Scalar[dtype](0))
    var x_tt = TileTensor(x_heap, row_major(dim, total_seqlen))
    var weight_heap = List(length=dim * width, fill=Scalar[dtype](0))
    var weight_tt = TileTensor(weight_heap, row_major(dim, width))
    var bias_heap = List(length=dim, fill=Scalar[dtype](0))
    var bias_tt = TileTensor(bias_heap, row_major(dim))
    var output_heap = List(length=dim * total_seqlen, fill=Scalar[dtype](0))
    var output_tt = TileTensor(output_heap, row_major(dim, total_seqlen))

    rand[dtype](x_tt._storage, dim * total_seqlen)
    rand[dtype](weight_tt._storage, dim * width)

    var query_start_loc_heap = List(length=batch + 1, fill=Int32(0))
    var query_start_loc_tt = TileTensor(
        query_start_loc_heap, row_major(batch + 1)
    )
    for i in range(batch + 1):
        query_start_loc_tt.raw_store(i, Int32(i * seqlen))

    var cache_indices_heap = List(length=batch, fill=Int32(0))
    var cache_indices_tt = TileTensor(cache_indices_heap, row_major(batch))
    for i in range(batch):
        cache_indices_tt.raw_store(i, Int32(i))

    var has_initial_state_heap = List(length=batch, fill=Scalar[.bool](True))
    var has_initial_state_tt = TileTensor(
        has_initial_state_heap, row_major(batch)
    )

    # A distinct value per (b, d, s) so a wrong entry names itself.
    var conv_states_heap = List(
        length=batch * dim * state_len, fill=Scalar[dtype](0)
    )
    var conv_states_tt = TileTensor(
        conv_states_heap, row_major(batch, dim, state_len)
    )
    var initial_heap = List(
        length=batch * dim * state_len, fill=Scalar[dtype](0)
    )
    var initial_tt = TileTensor(initial_heap, row_major(batch, dim, state_len))
    for b in range(batch):
        for d in range(dim):
            for s in range(state_len):
                var idx = (b * dim + d) * state_len + s
                var value = Scalar[dtype](1 + idx)
                conv_states_tt.raw_store(idx, value)
                initial_tt.raw_store(idx, value)

    causal_conv1d_varlen_fwd_cpu[
        dtype,
        dtype,
        dtype,
        dtype,
        DType.int32,
        DType.int32,
        DType.bool,
        dtype,
    ](
        dim,
        total_seqlen,
        width,
        batch,
        x_tt,
        weight_tt,
        bias_tt,
        query_start_loc_tt,
        cache_indices_tt,
        has_initial_state_tt,
        conv_states_tt,
        output_tt,
        UInt32(total_seqlen),  # x_dim_stride
        UInt32(1),  # x_seqlen_stride
        UInt32(width),  # weight_dim_stride
        UInt32(1),  # weight_width_stride
        UInt32(total_seqlen),  # out_dim_stride
        UInt32(1),  # out_seqlen_stride
        UInt32(dim * state_len),  # conv_states_batch_stride
        UInt32(state_len),  # conv_states_dim_stride
        UInt32(1),  # conv_states_width_stride
        False,  # silu_activation
        PAD_SLOT_ID,
        True,  # has_cache_indices
        True,  # has_initial_state_flag
        True,  # has_conv_states
        True,  # has_bias
    )

    # Expected: the last `state_len` of `initial ++ chunk`, per (b, d).
    for b in range(batch):
        for d in range(dim):
            for s in range(state_len):
                # Position of this state entry counted back from the chunk end:
                # `s - state_len` is negative, so it lands in the initial state
                # when the chunk is too short to cover it.
                var offset = seqlen - state_len + s
                var expected: Scalar[dtype]
                if offset >= 0:
                    expected = x_tt.raw_load(
                        UInt32(d) * UInt32(total_seqlen)
                        + UInt32(b * seqlen + offset)
                    )
                else:
                    expected = initial_tt.raw_load(
                        UInt32((b * dim + d) * state_len)
                        + UInt32(state_len + offset)
                    )
                assert_almost_equal(
                    conv_states_tt.raw_load(
                        UInt32((b * dim + d) * state_len) + UInt32(s)
                    ),
                    expected,
                    rtol=0.001,
                )


# =============================================================================
# Test functions for varlen causal conv1d forward
# =============================================================================


def test_varlen_causal_conv1d_fwd_equal_lengths() raises:
    """Test varlen causal conv1d forward with equal-length sequences."""
    run_varlen_causal_conv1d_fwd[.float32, "none"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=3
    )


def test_varlen_causal_conv1d_fwd_variable_lengths() raises:
    """Test varlen causal conv1d forward with variable-length sequences."""
    run_varlen_causal_conv1d_fwd[.float32, "none"](
        batch=3, dim=4, seq_lengths=Index(10, 6, 1), width=3
    )


def test_conv_state_writeback_chunk_shorter_than_width() raises:
    """A chunk shorter than `width - 1` must keep the state it did not replace.

    `seqlen=1` is the decode step, and `seqlen=2` with width 4 is the partial
    case between it and a chunk that supplies the whole new state.
    """
    run_conv_state_writeback[.float32](batch=2, dim=4, width=4, seqlen=1)
    run_conv_state_writeback[.float32](batch=2, dim=4, width=4, seqlen=2)
    run_conv_state_writeback[.float32](batch=3, dim=8, width=3, seqlen=1)


def test_conv_state_writeback_chunk_at_least_width() raises:
    """The same contract where the chunk does supply the whole state."""
    run_conv_state_writeback[.float32](batch=2, dim=4, width=4, seqlen=3)
    run_conv_state_writeback[.float32](batch=2, dim=4, width=4, seqlen=9)


def test_varlen_causal_conv1d_fwd_with_silu() raises:
    """Test varlen causal conv1d forward with SiLU activation."""
    run_varlen_causal_conv1d_fwd[.float32, "silu"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=3
    )


def test_varlen_causal_conv1d_fwd_various_widths() raises:
    """Test varlen causal conv1d forward with various kernel widths."""
    run_varlen_causal_conv1d_fwd[.float32, "none"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=2
    )
    run_varlen_causal_conv1d_fwd[.float32, "none"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=4
    )


# =============================================================================
# Test functions for varlen causal conv1d update
# =============================================================================


def test_varlen_causal_conv1d_update_basic() raises:
    """Test basic varlen causal conv1d update."""
    run_varlen_causal_conv1d_update[.float32, "none"](
        batch=2, dim=4, seqlen=1, width=3, state_len=4
    )


def test_varlen_causal_conv1d_update_with_silu() raises:
    """Test varlen causal conv1d update with SiLU activation."""
    run_varlen_causal_conv1d_update[.float32, "silu"](
        batch=2, dim=4, seqlen=1, width=3, state_len=4
    )


def test_varlen_causal_conv1d_update_seqlen_gt_1() raises:
    """Test varlen causal conv1d update with seqlen > 1."""
    run_varlen_causal_conv1d_update[.float32, "none"](
        batch=2, dim=4, seqlen=4, width=3, state_len=4
    )


# =============================================================================
# Test functions for varlen causal conv1d states
# =============================================================================


def test_varlen_causal_conv1d_states_basic() raises:
    """Test basic varlen causal conv1d states extraction."""
    run_varlen_causal_conv1d_states[.float32](
        batch=2, dim=4, seq_lengths=Index(8, 8), state_len=3
    )


def test_varlen_causal_conv1d_states_variable_lengths() raises:
    """Test varlen causal conv1d states with variable-length sequences."""
    run_varlen_causal_conv1d_states[.float32](
        batch=3, dim=4, seq_lengths=Index(10, 6, 1), state_len=3
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
