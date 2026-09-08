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
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout.tile_layout import Layout as TileLayout
from std.memory import unsafe_memcpy
from nn.fused_qk_rope import fused_qk_rope_ragged
from testdata.fused_qk_rope_goldens import (
    freqs_cis_table_input,
    k_cache_input,
    k_out_golden,
    q_input,
    q_out_golden,
)

from std.utils import IndexList


def test_fused_qk_rope[
    rope_dim: Int, dtype: DType
](ctx: DeviceContext) raises -> None:
    """Verifies fused_qk_rope against golden values computed with PyTorch."""
    comptime assert dtype == .float32, "goldens only for float32, currently"

    # Set up test hyperparameters.
    comptime batch_size = 2
    comptime start_positions: List[UInt32] = [0, 5]
    comptime seq_len = 3
    comptime max_seq_len = 16
    comptime num_layers = 1
    # Small pages so batch 1's [5, 8) window straddles a page boundary.
    comptime page_size = 2
    comptime pages_per_seq = max_seq_len // page_size
    comptime num_paged_blocks = batch_size * pages_per_seq

    def _max[dtype: DType, items: List[Scalar[dtype]]]() -> Scalar[dtype]:
        comptime assert len(items) > 0, "empty list in _max"
        var items_dyn = materialize[items]()
        var max_item = items_dyn[0]
        for i in range(1, len(items_dyn)):
            if items_dyn[i] > max_item:
                max_item = items_dyn[i]
        return max_item

    comptime assert max_seq_len > (
        seq_len + Int(_max[.uint32, items=start_positions]())
    ), "KV cache size smaller than sum of sequence length and start pos"
    comptime num_heads = 2
    comptime dim = 16
    comptime head_dim = dim // num_heads

    # Create aliases for KV cache parameters.
    comptime kv_params = KVCacheStaticParams(
        num_heads=num_heads, head_size=head_dim
    )
    comptime block_shape = IndexList[6](
        num_paged_blocks, 2, num_layers, page_size, num_heads, head_dim
    )

    # Construct backing buffer and the KV cache itself (uses LayoutTensor).
    var kv_cache_block_buffer = List[Scalar[dtype]](
        length=block_shape.flattened_length(), fill=0
    )
    var kv_cache_block_ptr = kv_cache_block_buffer.unsafe_ptr()
    var kv_cache_block = LayoutTensor[dtype, Layout.row_major[6]()](
        kv_cache_block_ptr,
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )

    comptime lut_layout = Layout.row_major[2]()
    var lookup_table_buffer = List[UInt32](
        length=batch_size * pages_per_seq, fill=0
    )
    var lookup_table = LayoutTensor[.uint32, lut_layout](
        lookup_table_buffer.unsafe_ptr(),
        RuntimeLayout[lut_layout].row_major(
            IndexList[2](batch_size, pages_per_seq)
        ),
    )
    # Reverse the page pool so no sequence lands on an identity mapping.
    for batch_idx in range(batch_size):
        for page_idx in range(pages_per_seq):
            lookup_table[batch_idx, page_idx] = UInt32(
                num_paged_blocks - 1 - (batch_idx * pages_per_seq + page_idx)
            )

    # Initialize KV cache block buffer with golden values.
    var k_cache_input_buffer = k_cache_input[dtype]()
    var k_cache_input_buffer_ptr: MutPointer[
        k_cache_input_buffer.T, origin_of(k_cache_input_buffer)
    ] = k_cache_input_buffer.unsafe_ptr()
    var max_cache_len_in_batch = 0
    var start_positions_dyn = materialize[start_positions]()
    for batch_idx in range(batch_size):
        var start_pos = Int(start_positions_dyn[batch_idx])
        # Rows are contiguous only within a page, so seed one token at a time.
        for seq_idx in range(seq_len):
            var tok_idx = start_pos + seq_idx
            unsafe_memcpy(
                dest=kv_cache_block.ptr
                + kv_cache_block._offset(
                    IndexList[6](
                        Int(lookup_table[batch_idx, tok_idx // page_size]),
                        0,
                        0,
                        tok_idx % page_size,
                        0,
                        0,
                    )
                ),
                src=k_cache_input_buffer_ptr
                + ((batch_idx * seq_len + seq_idx) * dim),
                count=dim,
            )
        max_cache_len_in_batch = max(max_cache_len_in_batch, start_pos)

    # Create the actual KV cache type (uses LayoutTensor).
    var kv_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        blocks=kv_cache_block,
        cache_lengths=LayoutTensor[mut=False, .uint32, Layout(UNKNOWN_VALUE)](
            start_positions_dyn.unsafe_ptr(),
            RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                IndexList[1](len(start_positions_dyn))
            ),
        ),
        lookup_table=LayoutTensor[mut=False, .uint32, lut_layout](
            lookup_table.ptr,
            RuntimeLayout[lut_layout].row_major(
                IndexList[2](batch_size, pages_per_seq)
            ),
        ),
        max_seq_length=seq_len,
        max_cache_length=UInt32(max_cache_len_in_batch),
    )

    # Define layouts for TileTensor-based tensors.
    comptime q_layout = row_major[batch_size * seq_len, num_heads, head_dim]()
    comptime input_row_offsets_layout = row_major[batch_size + 1]()

    # Create and initialize query buffer.
    var q_buffer = q_input[dtype]()
    assert len(q_buffer) == batch_size * seq_len * dim, "invalid q_buffer init"

    # Create query tensor as a TileTensor view of the query buffer.
    var q = TileTensor(q_buffer, q_layout)

    # Create input_row_offsets tensor using TileTensor.
    var input_row_offsets_stack = Array[UInt32, batch_size + 1](
        fill_with=lambda (i: Int) -> UInt32: UInt32(i * seq_len)
    )
    var input_row_offsets = TileTensor(
        input_row_offsets_stack, input_row_offsets_layout
    )

    # Create and init rotary matrix (frequencies as cos(x) + i*sin(x)).
    var freqs_cis_table_buffer = freqs_cis_table_input[dtype]()
    assert (
        len(freqs_cis_table_buffer) == 2 * max_seq_len * head_dim
    ), "invalid freqs_cis_table init"
    # Create a TileTensor view into freqs_cis that only includes the roped dimensions.
    # Offset to last rope_dim elements.
    # Note: This tensor has non-row-major strides (head_dim, 1) to select every
    # rope_dim-th element from the original head_dim-strided buffer.
    comptime freqs_cis_layout = TileLayout(
        Coord(Idx[max_seq_len], Idx[rope_dim]),
        Coord(Idx[head_dim], Idx[1]),
    )
    var freqs_cis_table_buffer_ptr: MutPointer[
        freqs_cis_table_buffer.T, origin_of(freqs_cis_table_buffer)
    ] = freqs_cis_table_buffer.unsafe_ptr()
    var freqs_cis_table = TileTensor(
        freqs_cis_table_buffer_ptr + (head_dim - rope_dim),
        freqs_cis_layout,
    )

    # Create and initialize golden outputs.
    var expected_q_out_buffer = q_out_golden[dtype]()
    assert len(expected_q_out_buffer) == len(
        q_buffer
    ), "invalid expected q out init"
    var expected_q_out = TileTensor(expected_q_out_buffer, q_layout)
    var expected_k_out_buffer = k_out_golden[dtype]()
    assert (
        len(expected_k_out_buffer) == batch_size * seq_len * dim
    ), "invalid expected k out init"
    var expected_k_out_buffer_ptr: MutPointer[
        expected_k_out_buffer.T, origin_of(expected_k_out_buffer)
    ] = expected_k_out_buffer.unsafe_ptr()

    # Create output buffer and TileTensor.
    var q_out_buffer = List[Scalar[dtype]](length=len(q_buffer), fill=0)
    var q_out = TileTensor(q_out_buffer, q_layout)

    fused_qk_rope_ragged[
        kv_collection.CacheType, interleaved=True, target=StaticString("cpu")
    ](
        q_proj=q,
        input_row_offsets=input_row_offsets,
        kv_collection=kv_collection,
        freqs_cis=freqs_cis_table,
        position_ids=None,
        output=q_out,
        layer_idx=UInt32(0),
        context=ctx,
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
                assert_almost_equal(
                    q_out._storage + base_offset,
                    q._storage + base_offset,
                    head_dim - rope_dim,
                )

                # Verify roped region: Last rope_dim elements should match expected output
                var roped_offset = base_offset + (head_dim - rope_dim)
                assert_almost_equal(
                    q_out._storage + roped_offset,
                    expected_q_out._storage + roped_offset,
                    rope_dim,
                )

    # Compare output and expected key cache buffers.
    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            for head_idx in range(num_heads):
                # Calculate offsets for current position
                var tok_idx = Int(start_positions_dyn[batch_idx]) + seq_idx
                var cache_block_ptr = (
                    kv_cache_block.ptr
                    + kv_cache_block._offset(
                        IndexList[6](
                            Int(lookup_table[batch_idx, tok_idx // page_size]),
                            0,
                            0,
                            tok_idx % page_size,
                            head_idx,
                            0,
                        )
                    )
                )
                var seq_offset = seq_idx * dim + head_idx * head_dim
                var input_offset = batch_idx * seq_len * dim + seq_offset

                # Verify unroped region: Should match original input
                assert_almost_equal(
                    cache_block_ptr,
                    k_cache_input_buffer_ptr + input_offset,
                    head_dim - rope_dim,
                )

                # Verify roped region: Should match expected output
                var roped_offset = head_dim - rope_dim
                assert_almost_equal(
                    cache_block_ptr + roped_offset,
                    expected_k_out_buffer_ptr + input_offset + roped_offset,
                    rope_dim,
                )

    _ = q_out_buffer^
    _ = expected_q_out_buffer^
    _ = freqs_cis_table_buffer^
    _ = q_buffer^
    _ = k_cache_input_buffer^
    _ = kv_cache_block_buffer^
    _ = lookup_table_buffer^
    _ = start_positions_dyn^


def main() raises -> None:
    with DeviceContext(api="cpu") as ctx:
        # Full head RoPE
        test_fused_qk_rope[8, .float32](ctx)
        # Partial RoPE (last 4 elements of each head)
        test_fused_qk_rope[4, .float32](ctx)
