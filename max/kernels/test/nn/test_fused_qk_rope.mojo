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
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.memory import unsafe_memcpy
from nn.fused_qk_rope import fused_qk_rope
from testdata.fused_qk_rope_goldens import (
    freqs_cis_table_input,
    k_cache_input,
    k_out_golden,
    q_input,
    q_out_golden,
)

from std.utils import IndexList


def test_fused_qk_rope[dtype: DType](ctx: DeviceContext) raises -> None:
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

    # Construct backing buffer and the KV cache itself.
    var kv_cache_block_buffer = List[Scalar[dtype]](
        length=block_shape.flattened_length(), fill=0
    )
    var kv_cache_block_ptr = kv_cache_block_buffer.unsafe_ptr()
    var kv_cache_block = LayoutTensor[dtype, Layout.row_major[6]()](
        kv_cache_block_ptr,
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )

    # Initialize KV cache block buffer with golden values.
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

    var start_positions_dyn = materialize[start_positions]()
    var k_cache_input_buffer = k_cache_input[dtype]()
    var k_cache_input_buffer_ptr: MutPointer[
        k_cache_input_buffer.T, origin_of(k_cache_input_buffer)
    ] = k_cache_input_buffer.unsafe_ptr()
    var max_cache_len_in_batch = 0
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

    # Create the actual KV cache type.
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

    # Create and initialize query buffer.
    var q_buffer = q_input[dtype]()
    assert len(q_buffer) == batch_size * seq_len * dim, "invalid q_buffer init"

    # Create query tensor as a view of the query buffer.
    var q = TileTensor(
        q_buffer, row_major[batch_size, seq_len, num_heads, head_dim]()
    )

    # Create and init rotary matrix (frequencies as cos(x) + i*sin(x)).
    var freqs_cis_table_buffer = freqs_cis_table_input[dtype]()
    assert (
        len(freqs_cis_table_buffer) == 2 * max_seq_len * head_dim
    ), "invalid freqs_cis_table init"
    var freqs_cis_table = TileTensor(
        freqs_cis_table_buffer, row_major[max_seq_len, head_dim]()
    )

    # Create and initialize golden outputs.
    var expected_q_out_buffer = q_out_golden[dtype]()
    assert len(expected_q_out_buffer) == len(
        q_buffer
    ), "invalid expected q out init"
    var expected_q_out = TileTensor(expected_q_out_buffer, q.layout)
    var expected_k_out_buffer = k_out_golden[dtype]()
    assert (
        len(expected_k_out_buffer) == batch_size * seq_len * dim
    ), "invalid expected k out init"
    var expected_k_out_buffer_ptr: MutPointer[
        expected_k_out_buffer.T, origin_of(expected_k_out_buffer)
    ] = expected_k_out_buffer.unsafe_ptr()

    # Create output buffer.
    var q_out_buffer = List[Scalar[dtype]](length=len(q_buffer), fill=0)
    var q_out = TileTensor(q_out_buffer, q.layout).make_dynamic[.int64]()

    # Create valid_lengths buffer - all sequences have full seq_len valid
    var valid_lengths_buffer = List[UInt32](
        length=batch_size, fill=UInt32(seq_len)
    )
    var valid_lengths = TileTensor(
        valid_lengths_buffer,
        row_major(batch_size),
    )

    fused_qk_rope[
        kv_collection.CacheType, interleaved=True, target=StaticString("cpu")
    ](
        q_proj=q,
        kv_collection=kv_collection,
        freqs_cis=freqs_cis_table,
        layer_idx=UInt32(0),
        valid_lengths=valid_lengths,
        output=q_out,
        context=ctx,
    )

    # Compare output and expected query tensors.
    assert_almost_equal(
        q_out._storage, expected_q_out._storage, expected_q_out.num_elements()
    )

    # Compare output and expected key cache buffers.
    for batch_idx in range(batch_size):
        var start_pos = Int(start_positions_dyn[batch_idx])
        for seq_idx in range(seq_len):
            var tok_idx = start_pos + seq_idx
            assert_almost_equal(
                kv_cache_block.ptr
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
                expected_k_out_buffer_ptr
                + ((batch_idx * seq_len + seq_idx) * dim),
                # Number of elements in one token.
                dim,
            )

    _ = q_out_buffer^
    _ = expected_q_out_buffer^
    _ = freqs_cis_table_buffer^
    _ = q_buffer^
    _ = k_cache_input_buffer^
    _ = kv_cache_block_buffer^
    _ = valid_lengths_buffer^
    _ = lookup_table_buffer^
    _ = start_positions_dyn^


def main() raises -> None:
    with DeviceContext(api="cpu") as ctx:
        test_fused_qk_rope[.float32](ctx)
