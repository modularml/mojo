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
    position_ids_input,
    q_input,
    q_out_golden,
    q_out_golden_with_position_ids,
)

from std.utils import IndexList


def _test_rope_ragged_gpu_impl[
    rope_dim: Int, dtype: DType, has_position_ids: Bool
](ctx: DeviceContext) raises -> None:
    """Verifies rope_ragged GPU kernel against golden values computed with PyTorch.
    """
    comptime assert dtype == .float32, "goldens only for float32, currently"

    # Set up test hyperparameters - same as CPU test
    comptime batch_size = 2
    comptime seq_len = 3
    comptime max_seq_len = 16
    comptime num_heads = 2
    comptime dim = 16
    comptime head_dim = dim // num_heads

    # Define layouts for all tensors
    comptime q_layout = row_major[batch_size * seq_len, num_heads, head_dim]()
    comptime input_row_offsets_layout = row_major[batch_size + 1]()
    comptime start_pos_layout = row_major[batch_size]()
    comptime freqs_cis_layout = row_major[max_seq_len, rope_dim]()
    comptime position_ids_layout = row_major[1, batch_size * seq_len]()

    # ===== Step 1: Create all buffers =====
    # Query tensor buffers
    var q_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        q_layout.static_product
    )
    var q_device_buffer = ctx.enqueue_create_buffer[dtype](
        q_layout.static_product
    )

    # Input row offsets buffers
    var input_row_offsets_host_buffer = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](input_row_offsets_layout.static_product)
    var input_row_offsets_device_buffer = ctx.enqueue_create_buffer[
        DType.uint32
    ](input_row_offsets_layout.static_product)

    # Start position buffers
    var start_pos_host_buffer = ctx.enqueue_create_host_buffer[.uint32](
        start_pos_layout.static_product
    )
    var start_pos_device_buffer = ctx.enqueue_create_buffer[.uint32](
        start_pos_layout.static_product
    )

    # Frequency table buffers
    var freqs_cis_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        freqs_cis_layout.static_product
    )
    var freqs_cis_device_buffer = ctx.enqueue_create_buffer[dtype](
        freqs_cis_layout.static_product
    )

    # Position ids buffers
    var position_ids_host_buffer = ctx.enqueue_create_host_buffer[.uint32](
        position_ids_layout.static_product
    )
    var position_ids_device_buffer = ctx.enqueue_create_buffer[.uint32](
        position_ids_layout.static_product
    )

    # Output buffers
    var q_out_device_buffer = ctx.enqueue_create_buffer[dtype](
        q_layout.static_product
    )
    var q_out_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        q_layout.static_product
    )

    # Synchronize to ensure all buffers are created
    ctx.synchronize()

    # ===== Step 2: Fill host buffers with data =====
    # Fill query tensor
    var q_buffer = q_input[dtype]()
    for i in range(len(q_buffer)):
        q_host_buffer[i] = q_buffer[i]

    # Fill input row offsets: [0, seq_len, 2*seq_len] = [0, 3, 6]
    for i in range(batch_size):
        input_row_offsets_host_buffer[i] = UInt32(i * seq_len)
    input_row_offsets_host_buffer[batch_size] = batch_size * seq_len

    # Fill start positions (ignored when position_ids are passed)
    start_pos_host_buffer[0] = 0
    start_pos_host_buffer[1] = 5

    # Fill frequency table
    var freqs_cis_table_buffer = freqs_cis_table_input[dtype]()
    for seq_idx in range(max_seq_len):
        for rope_idx in range(rope_dim):
            # Offset to last rope_dim elements in the original buffer
            var buffer_offset = (
                seq_idx * head_dim + (head_dim - rope_dim) + rope_idx
            )
            freqs_cis_host_buffer[
                seq_idx * rope_dim + rope_idx
            ] = freqs_cis_table_buffer[buffer_offset]

    # Fill explicit position ids
    var position_ids_buffer = position_ids_input[.uint32]()
    for i in range(len(position_ids_buffer)):
        position_ids_host_buffer[i] = position_ids_buffer[i]

    # ===== Step 3: Copy all data to device =====
    ctx.enqueue_copy(q_device_buffer, q_host_buffer)
    ctx.enqueue_copy(
        input_row_offsets_device_buffer, input_row_offsets_host_buffer
    )
    ctx.enqueue_copy(start_pos_device_buffer, start_pos_host_buffer)
    ctx.enqueue_copy(freqs_cis_device_buffer, freqs_cis_host_buffer)
    ctx.enqueue_copy(position_ids_device_buffer, position_ids_host_buffer)

    # Synchronize to ensure all copies are complete before kernel execution
    ctx.synchronize()

    # ===== Step 4: Create TileTensor views =====
    var q_device_tensor = TileTensor(q_device_buffer, q_layout)
    var input_row_offsets_device_tensor = TileTensor(
        input_row_offsets_device_buffer, input_row_offsets_layout
    )
    var start_pos_device_tensor = TileTensor(
        start_pos_device_buffer, start_pos_layout
    )
    var freqs_cis_device_tensor = TileTensor(
        freqs_cis_device_buffer, freqs_cis_layout
    )
    var position_ids_device_tensor_static = TileTensor(
        position_ids_device_buffer, position_ids_layout
    )
    var position_ids_device_tensor = TileTensor[
        .uint32,
        type_of(position_ids_device_tensor_static).LayoutType,
        ImmutAnyOrigin,
    ](
        position_ids_device_tensor_static._storage.as_imm().unsafe_origin_cast[
            ImmutAnyOrigin
        ](),
        position_ids_device_tensor_static.layout,
    ).make_dynamic[
        DType.int64
    ]()

    var q_out_device_tensor = TileTensor(q_out_device_buffer, q_layout)

    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](idx: IndexList[3], val: SIMD[dtype, width]) {
        var q_out_device_tensor
    } -> None:
        q_out_device_tensor.store[width=width](Coord(idx), val)

    # Execute rope_ragged kernel on GPU
    comptime if has_position_ids:
        rope_ragged[
            dtype,
            dtype,
            interleaved=True,
            target=StaticString("gpu"),
        ](
            x=q_device_tensor.as_unsafe_any_origin(),
            input_row_offsets=input_row_offsets_device_tensor.as_unsafe_any_origin(),
            start_pos=start_pos_device_tensor.as_unsafe_any_origin(),
            freqs_cis=freqs_cis_device_tensor.as_unsafe_any_origin(),
            context=ctx,
            output_fn=output_fn,
            position_ids=position_ids_device_tensor,
        )
    else:
        rope_ragged[
            dtype,
            dtype,
            interleaved=True,
            target=StaticString("gpu"),
        ](
            x=q_device_tensor.as_unsafe_any_origin(),
            input_row_offsets=input_row_offsets_device_tensor.as_unsafe_any_origin(),
            start_pos=start_pos_device_tensor.as_unsafe_any_origin(),
            freqs_cis=freqs_cis_device_tensor.as_unsafe_any_origin(),
            context=ctx,
            output_fn=output_fn,
        )

    # Copy results back to host for validation
    ctx.enqueue_copy(q_out_host_buffer, q_out_device_buffer)
    ctx.synchronize()

    # Create expected output for validation using HostBuffer
    var expected_q_out_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        q_layout.static_product
    )
    ctx.synchronize()
    var expected_q_out_buffer: List[Scalar[dtype]]
    comptime if has_position_ids:
        expected_q_out_buffer = q_out_golden_with_position_ids[dtype]()
    else:
        expected_q_out_buffer = q_out_golden[dtype]()
    for i in range(len(expected_q_out_buffer)):
        expected_q_out_host_buffer[i] = expected_q_out_buffer[i]

    # Validate results - same logic as CPU test
    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            for head_idx in range(num_heads):
                # Calculate global token index and offsets
                var global_token_idx = batch_idx * seq_len + seq_idx

                # Calculate base offset for current head
                var base_offset = (
                    global_token_idx * num_heads * head_dim  # token offset
                    + head_idx * head_dim  # head offset
                )

                comptime if rope_dim == head_dim:
                    # Full RoPE case - compare entire output against golden
                    assert_almost_equal(
                        q_out_host_buffer.as_span()[
                            base_offset : base_offset + head_dim
                        ],
                        expected_q_out_host_buffer.as_span()[
                            base_offset : base_offset + head_dim
                        ],
                        atol=1e-4,
                    )
                else:
                    # Partial RoPE case - use same logic as original test
                    # Verify unroped region: Should remain unchanged from input
                    var unroped_len = head_dim - rope_dim
                    assert_almost_equal(
                        q_out_host_buffer.as_span()[
                            base_offset : base_offset + unroped_len
                        ],
                        q_host_buffer.as_span()[
                            base_offset : base_offset + unroped_len
                        ],
                        atol=1e-4,
                    )

                    # Verify roped region: Should match expected output
                    var roped_offset = base_offset + (head_dim - rope_dim)
                    assert_almost_equal(
                        q_out_host_buffer.as_span()[
                            roped_offset : roped_offset + rope_dim
                        ],
                        expected_q_out_host_buffer.as_span()[
                            roped_offset : roped_offset + rope_dim
                        ],
                        atol=1e-4,
                    )


def test_rope_ragged_gpu[
    rope_dim: Int, dtype: DType
](ctx: DeviceContext) raises -> None:
    _test_rope_ragged_gpu_impl[rope_dim, dtype, has_position_ids=False](ctx)


def test_rope_ragged_gpu_with_position_ids[
    rope_dim: Int, dtype: DType
](ctx: DeviceContext) raises -> None:
    _test_rope_ragged_gpu_impl[rope_dim, dtype, has_position_ids=True](ctx)


def test_rope_ragged_gpu_rope_first[
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
    var freqs_cis_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        freqs_cis_layout.static_product
    )
    var input_row_offsets_host_buffer = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](input_row_offsets_layout.static_product)
    var start_pos_host_buffer = ctx.enqueue_create_host_buffer[.uint32](
        start_pos_layout.static_product
    )
    var full_out_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        q_layout.static_product
    )
    var prefix_out_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        prefix_layout.static_product
    )

    var q_device_buffer = ctx.enqueue_create_buffer[dtype](
        q_layout.static_product
    )
    var prefix_device_buffer = ctx.enqueue_create_buffer[dtype](
        prefix_layout.static_product
    )
    var freqs_cis_device_buffer = ctx.enqueue_create_buffer[dtype](
        freqs_cis_layout.static_product
    )
    var input_row_offsets_device_buffer = ctx.enqueue_create_buffer[
        DType.uint32
    ](input_row_offsets_layout.static_product)
    var start_pos_device_buffer = ctx.enqueue_create_buffer[.uint32](
        start_pos_layout.static_product
    )
    var full_out_device_buffer = ctx.enqueue_create_buffer[dtype](
        q_layout.static_product
    )
    var prefix_out_device_buffer = ctx.enqueue_create_buffer[dtype](
        prefix_layout.static_product
    )
    ctx.synchronize()

    var q_buffer = q_input[dtype]()
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
    start_pos_host_buffer[0] = 0
    start_pos_host_buffer[1] = 5

    ctx.enqueue_copy(q_device_buffer, q_host_buffer)
    ctx.enqueue_copy(prefix_device_buffer, prefix_host_buffer)
    ctx.enqueue_copy(freqs_cis_device_buffer, freqs_cis_host_buffer)
    ctx.enqueue_copy(
        input_row_offsets_device_buffer, input_row_offsets_host_buffer
    )
    ctx.enqueue_copy(start_pos_device_buffer, start_pos_host_buffer)
    ctx.synchronize()

    var q_device_tensor = TileTensor(q_device_buffer, q_layout)
    var prefix_device_tensor = TileTensor(prefix_device_buffer, prefix_layout)
    var freqs_cis_device_tensor = TileTensor(
        freqs_cis_device_buffer, freqs_cis_layout
    )
    var input_row_offsets_device_tensor = TileTensor(
        input_row_offsets_device_buffer, input_row_offsets_layout
    )
    var start_pos_device_tensor = TileTensor(
        start_pos_device_buffer, start_pos_layout
    )
    var full_out_device_tensor = TileTensor(full_out_device_buffer, q_layout)
    var prefix_out_device_tensor = TileTensor(
        prefix_out_device_buffer, prefix_layout
    )

    @always_inline
    def full_output_fn[
        width: SIMDLength, alignment: Int
    ](idx: IndexList[3], val: SIMD[dtype, width]) {
        var full_out_device_tensor
    } -> None:
        full_out_device_tensor.store[width=width](Coord(idx), val)

    @always_inline
    def prefix_output_fn[
        width: SIMDLength, alignment: Int
    ](idx: IndexList[3], val: SIMD[dtype, width]) {
        var prefix_out_device_tensor
    } -> None:
        prefix_out_device_tensor.store[width=width](Coord(idx), val)

    rope_ragged[
        dtype,
        dtype,
        interleaved=True,
        target=StaticString("gpu"),
        rope_first=True,
    ](
        x=q_device_tensor.as_unsafe_any_origin(),
        input_row_offsets=input_row_offsets_device_tensor.as_unsafe_any_origin(),
        start_pos=start_pos_device_tensor.as_unsafe_any_origin(),
        freqs_cis=freqs_cis_device_tensor.as_unsafe_any_origin(),
        context=ctx,
        output_fn=full_output_fn,
    )

    rope_ragged[
        dtype,
        dtype,
        interleaved=True,
        target=StaticString("gpu"),
    ](
        x=prefix_device_tensor.as_unsafe_any_origin(),
        input_row_offsets=input_row_offsets_device_tensor.as_unsafe_any_origin(),
        start_pos=start_pos_device_tensor.as_unsafe_any_origin(),
        freqs_cis=freqs_cis_device_tensor.as_unsafe_any_origin(),
        context=ctx,
        output_fn=prefix_output_fn,
    )

    ctx.enqueue_copy(full_out_host_buffer, full_out_device_buffer)
    ctx.enqueue_copy(prefix_out_host_buffer, prefix_out_device_buffer)
    ctx.synchronize()

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
                atol=1e-4,
            )
            assert_almost_equal(
                full_out_host_buffer.as_span()[
                    head_offset + rope_dim : head_offset + head_dim
                ],
                q_host_buffer.as_span()[
                    head_offset + rope_dim : head_offset + head_dim
                ],
                atol=1e-4,
            )


def execute_rope_ragged_gpu(ctx: DeviceContext) raises -> None:
    """Execute GPU RoPE tests with different rope dimensions."""
    # Full head RoPE
    test_rope_ragged_gpu[8, DType.float32](ctx)
    test_rope_ragged_gpu_with_position_ids[8, DType.float32](ctx)

    # partial RoPE
    test_rope_ragged_gpu[4, DType.float32](ctx)
    test_rope_ragged_gpu_with_position_ids[4, DType.float32](ctx)

    # partial RoPE over the leading columns of each head
    test_rope_ragged_gpu_rope_first[4, DType.float32](ctx)


def main() raises:
    with DeviceContext() as ctx:
        execute_rope_ragged_gpu(ctx)
