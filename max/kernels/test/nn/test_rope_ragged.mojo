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
from internal_utils import assert_almost_equal
from layout import Coord, TileTensor, row_major
from nn.rope import rope_ragged
from testdata.fused_qk_rope_goldens import (
    freqs_cis_table_input,
    q_input,
    q_out_golden,
)

from std.utils import IndexList


def test_rope_ragged[
    rope_dim: Int, dtype: DType
](ctx: DeviceContext) raises -> None:
    """Verifies fused_qk_rope against golden values computed with PyTorch."""
    comptime assert dtype == .float32, "goldens only for float32, currently"

    # Set up test hyperparameters.
    comptime batch_size = 2
    var start_positions: List[UInt32] = [0, 5]
    var lookup_table: List[UInt32] = [0, 1]
    comptime seq_len = 3
    comptime max_seq_len = 16
    comptime num_layers = 1

    def _max[dtype: DType](items: List[Scalar[dtype]]) -> Scalar[dtype]:
        assert len(items) > 0, "empty list in _max"
        var max_item = items[0]

        for i in range(1, len(items)):
            if items[i] > max_item:
                max_item = items[i]
        return max_item

    assert max_seq_len > (
        seq_len + Int(_max[.uint32](start_positions))
    ), "KV cache size smaller than sum of sequence length and start pos"
    comptime num_heads = 2
    comptime dim = 16
    comptime head_dim = dim // num_heads

    # Define layouts for all tensors
    comptime q_layout = row_major[batch_size * seq_len, num_heads, head_dim]()
    comptime input_row_offsets_layout = row_major[batch_size + 1]()
    comptime start_pos_layout = row_major[batch_size]()
    comptime freqs_cis_layout = row_major[max_seq_len, rope_dim]()

    # Create and initialize query tensor using HostBuffer + TileTensor
    var q_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        q_layout.static_product
    )
    ctx.synchronize()
    var q_buffer = q_input[dtype]()
    assert len(q_buffer) == batch_size * seq_len * dim, "invalid q_buffer init"

    # Copy data from golden buffer to host buffer
    for i in range(len(q_buffer)):
        q_host_buffer[i] = q_buffer[i]
    var q_tensor = TileTensor(q_host_buffer, q_layout)

    # Create input_row_offsets using HostBuffer + TileTensor
    var input_row_offsets_host_buffer = ctx.enqueue_create_host_buffer[.uint32](
        input_row_offsets_layout.static_product
    )
    ctx.synchronize()
    for i in range(batch_size):
        input_row_offsets_host_buffer[i] = UInt32(i * seq_len)
    input_row_offsets_host_buffer[batch_size] = batch_size * seq_len
    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_host_buffer, input_row_offsets_layout
    )

    # Create and init rotary matrix (frequencies as cos(x) + i*sin(x)).
    var freqs_cis_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        freqs_cis_layout.static_product
    )
    ctx.synchronize()
    var freqs_cis_table_buffer = freqs_cis_table_input[dtype]()
    assert (
        len(freqs_cis_table_buffer) == 2 * max_seq_len * head_dim
    ), "invalid freqs_cis_table init"

    # Copy the roped dimensions from the buffer to the host buffer
    for seq_idx in range(max_seq_len):
        for rope_idx in range(rope_dim):
            # Offset to last rope_dim elements in the original buffer
            var buffer_offset = (
                seq_idx * head_dim + (head_dim - rope_dim) + rope_idx
            )
            freqs_cis_host_buffer[
                seq_idx * rope_dim + rope_idx
            ] = freqs_cis_table_buffer[buffer_offset]
    var freqs_cis_tensor = TileTensor(freqs_cis_host_buffer, freqs_cis_layout)

    # Create and initialize golden outputs using HostBuffer + TileTensor
    var expected_q_out_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        q_layout.static_product
    )
    ctx.synchronize()
    var expected_q_out_buffer = q_out_golden[dtype]()
    assert len(expected_q_out_buffer) == len(
        q_buffer
    ), "invalid expected q out init"
    for i in range(len(expected_q_out_buffer)):
        expected_q_out_host_buffer[i] = expected_q_out_buffer[i]
    var expected_q_out_tensor = TileTensor(expected_q_out_host_buffer, q_layout)

    # Create output tensor using HostBuffer + TileTensor
    var q_out_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        q_layout.static_product
    )
    ctx.synchronize()
    # Initialize to zero
    for i in range(q_layout.static_product):
        q_out_host_buffer[i] = 0
    var q_out_tensor = TileTensor(q_out_host_buffer, q_layout)

    # Create start_pos tensor using HostBuffer + TileTensor
    var start_pos_host_buffer = ctx.enqueue_create_host_buffer[.uint32](
        start_pos_layout.static_product
    )
    ctx.synchronize()
    for i in range(len(start_positions)):
        start_pos_host_buffer[i] = start_positions[i]
    var start_pos_tensor = TileTensor(start_pos_host_buffer, start_pos_layout)

    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](idx: IndexList[3], val: SIMD[dtype, width]) {var q_out_tensor} -> None:
        q_out_tensor.store[width=width](Coord(idx), val)

    rope_ragged[
        dtype,
        dtype,
        interleaved=True,
        target=StaticString("cpu"),
    ](
        x=q_tensor,
        input_row_offsets=input_row_offsets_tensor,
        start_pos=start_pos_tensor,
        freqs_cis=freqs_cis_tensor,
        context=ctx,
        output_fn=output_fn,
    )

    # Compare output and expected query tensors.
    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            for head_idx in range(num_heads):
                # Calculate base offset for current head
                var base_offset = (
                    batch_idx * seq_len * dim  # batch offset
                    + seq_idx * dim  # sequence offset
                    + head_idx * head_dim  # head offset
                )
                # Verify unroped region: First (head_dim - rope_dim) elements should remain unchanged
                var unroped_len = head_dim - rope_dim
                assert_almost_equal(
                    q_out_host_buffer.as_span()[
                        base_offset : base_offset + unroped_len
                    ],
                    q_host_buffer.as_span()[
                        base_offset : base_offset + unroped_len
                    ],
                )

                # Verify roped region: Last rope_dim elements should match expected output
                var roped_offset = base_offset + (head_dim - rope_dim)
                assert_almost_equal(
                    q_out_host_buffer.as_span()[
                        roped_offset : roped_offset + rope_dim
                    ],
                    expected_q_out_host_buffer.as_span()[
                        roped_offset : roped_offset + rope_dim
                    ],
                )


def test_rope_ragged_rope_first[
    rope_dim: Int, dtype: DType
](ctx: DeviceContext) raises -> None:
    """Checks `rope_first` against an explicit slice -> RoPE -> concat.

    The Indexer fusion rewrites `concat(rope(x[..., :R]), x[..., R:])` into
    one `rope_first` call over the whole head, so running RoPE over a
    standalone `[..., R]` tensor (a full-head rope, no passthrough) must
    reproduce the roped columns exactly, and the tail must come through
    untouched.
    """
    comptime batch_size = 2
    comptime seq_len = 3
    comptime max_seq_len = 16
    comptime num_heads = 2
    comptime dim = 16
    comptime head_dim = dim // num_heads
    comptime assert rope_dim < head_dim, "rope_first needs a passthrough tail"
    var start_positions: List[UInt32] = [0, 5]

    comptime q_layout = row_major[batch_size * seq_len, num_heads, head_dim]()
    comptime prefix_layout = row_major[
        batch_size * seq_len, num_heads, rope_dim
    ]()
    comptime input_row_offsets_layout = row_major[batch_size + 1]()
    comptime start_pos_layout = row_major[batch_size]()
    comptime freqs_cis_layout = row_major[max_seq_len, rope_dim]()

    var q_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        q_layout.static_product
    )
    var prefix_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        prefix_layout.static_product
    )
    var full_out_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        q_layout.static_product
    )
    var prefix_out_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        prefix_layout.static_product
    )
    var freqs_cis_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        freqs_cis_layout.static_product
    )
    var input_row_offsets_host_buffer = ctx.enqueue_create_host_buffer[.uint32](
        input_row_offsets_layout.static_product
    )
    var start_pos_host_buffer = ctx.enqueue_create_host_buffer[.uint32](
        start_pos_layout.static_product
    )
    ctx.synchronize()

    var q_buffer = q_input[dtype]()
    assert len(q_buffer) == batch_size * seq_len * dim, "invalid q_buffer init"
    for i in range(len(q_buffer)):
        q_host_buffer[i] = q_buffer[i]

    # The leading `rope_dim` columns of every head, packed into their own
    # tensor -- the `mo.slice` the fusion consumes, materialized.
    for token_idx in range(batch_size * seq_len):
        for head_idx in range(num_heads):
            for col in range(rope_dim):
                prefix_host_buffer[
                    (token_idx * num_heads + head_idx) * rope_dim + col
                ] = q_host_buffer[
                    (token_idx * num_heads + head_idx) * head_dim + col
                ]

    var freqs_cis_table_buffer = freqs_cis_table_input[dtype]()
    for seq_idx in range(max_seq_len):
        for rope_idx in range(rope_dim):
            freqs_cis_host_buffer[
                seq_idx * rope_dim + rope_idx
            ] = freqs_cis_table_buffer[
                seq_idx * head_dim + (head_dim - rope_dim) + rope_idx
            ]

    for i in range(batch_size):
        input_row_offsets_host_buffer[i] = UInt32(i * seq_len)
    input_row_offsets_host_buffer[batch_size] = batch_size * seq_len
    for i in range(len(start_positions)):
        start_pos_host_buffer[i] = start_positions[i]
    for i in range(q_layout.static_product):
        full_out_host_buffer[i] = 0
    for i in range(prefix_layout.static_product):
        prefix_out_host_buffer[i] = 0

    var q_tensor = TileTensor(q_host_buffer, q_layout)
    var prefix_tensor = TileTensor(prefix_host_buffer, prefix_layout)
    var full_out_tensor = TileTensor(full_out_host_buffer, q_layout)
    var prefix_out_tensor = TileTensor(prefix_out_host_buffer, prefix_layout)
    var freqs_cis_tensor = TileTensor(freqs_cis_host_buffer, freqs_cis_layout)
    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_host_buffer, input_row_offsets_layout
    )
    var start_pos_tensor = TileTensor(start_pos_host_buffer, start_pos_layout)

    @always_inline
    def full_output_fn[
        width: SIMDLength, alignment: Int
    ](idx: IndexList[3], val: SIMD[dtype, width]) {var full_out_tensor} -> None:
        full_out_tensor.store[width=width](Coord(idx), val)

    @always_inline
    def prefix_output_fn[
        width: SIMDLength, alignment: Int
    ](idx: IndexList[3], val: SIMD[dtype, width]) {
        var prefix_out_tensor
    } -> None:
        prefix_out_tensor.store[width=width](Coord(idx), val)

    rope_ragged[
        dtype,
        dtype,
        interleaved=True,
        target=StaticString("cpu"),
        rope_first=True,
    ](
        x=q_tensor,
        input_row_offsets=input_row_offsets_tensor,
        start_pos=start_pos_tensor,
        freqs_cis=freqs_cis_tensor,
        context=ctx,
        output_fn=full_output_fn,
    )

    rope_ragged[
        dtype,
        dtype,
        interleaved=True,
        target=StaticString("cpu"),
    ](
        x=prefix_tensor,
        input_row_offsets=input_row_offsets_tensor,
        start_pos=start_pos_tensor,
        freqs_cis=freqs_cis_tensor,
        context=ctx,
        output_fn=prefix_output_fn,
    )

    for token_idx in range(batch_size * seq_len):
        for head_idx in range(num_heads):
            var head_offset = (token_idx * num_heads + head_idx) * head_dim
            var prefix_offset = (token_idx * num_heads + head_idx) * rope_dim

            assert_almost_equal(
                full_out_host_buffer.as_span()[
                    head_offset : head_offset + rope_dim
                ],
                prefix_out_host_buffer.as_span()[
                    prefix_offset : prefix_offset + rope_dim
                ],
            )
            assert_almost_equal(
                full_out_host_buffer.as_span()[
                    head_offset + rope_dim : head_offset + head_dim
                ],
                q_host_buffer.as_span()[
                    head_offset + rope_dim : head_offset + head_dim
                ],
            )


def main() raises -> None:
    with DeviceContext(api="cpu") as ctx:
        # Full head RoPE - this works correctly and is production ready
        print("Full head RoPE")
        test_rope_ragged[8, .float32](ctx)

        # TODO: This was failing for some reason, we don't actually need it for now.
        # Circle back and fix this.
        # Partial RoPE (last 4 elements of each head)
        print("Partial RoPE")
        test_rope_ragged[4, .float32](ctx)

        # Partial RoPE over the *leading* 4 elements of each head.
        print("Leading-half RoPE (rope_first)")
        test_rope_ragged_rope_first[4, .float32](ctx)
