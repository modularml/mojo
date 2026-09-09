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
"""Minimal unit test reproducing the Gemma4 attention shapes.

Matches max/tests/integration/architectures/gemma4:test_attention. The
integration test drives `flash_attention[ragged=True]` via the MAX graph
compiler with paged KV cache + CausalMask. Compiling those specializations on
AMD currently triggers an LLVM AMDGPU register-allocator assertion; this
self-contained test is intended as a fast repro so we don't have to run the
full Python pipeline to hit the crash.

Shapes (from max/tests/integration/architectures/gemma4/testdata/config.json):

| Variant | num_q_heads | num_kv_heads | head_dim | seq_len |
| ------- | ----------- | ------------ | -------- | ------- |
| Local   | 32          | 16           | 256      | 11      |
| Global  | 32          | 4            | 512      | 11      |

Both run with bfloat16, page_size=256, CausalMask, ragged inputs,
cache_len=0 (pure prefill).
"""

from std.math import ceildiv, rsqrt
from std.random import seed
from std.sys.defines import get_defined_int
from layout._utils import ManagedLayoutTensor
from max.gpu.host import DeviceContext
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from std.memory import unsafe_memset_zero
from kv_cache_test_utils import (
    assert_no_nan_inf,
    padded_lut_cols,
    random_distinct,
)
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalMask, MHAMask, SlidingWindowCausalMask

from std.utils import IndexList


def execute_ragged_paged_flash_attention[
    num_q_heads: Int,
    dtype: DType,
    kv_params: KVCacheStaticParams,
    mask_t: MHAMask,
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    num_layers: Int,
    layer_idx: Int,
    mask: mask_t,
    label: StaticString,
    ctx: DeviceContext,
) raises:
    comptime page_size = get_defined_int["page_size", 256]()

    var batch_size = len(valid_lengths)
    assert len(valid_lengths) == len(cache_lengths)

    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]

    comptime row_offsets_layout = Layout(UNKNOWN_VALUE)
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    comptime paged_lut_layout = Layout.row_major[2]()
    comptime kv_block_6d_layout = Layout.row_major[6]()

    var row_offsets_shape = IndexList[1](batch_size + 1)
    var cache_lengths_shape = IndexList[1](batch_size)
    var q_ragged_shape = IndexList[3](
        total_length, num_q_heads, kv_params.head_size
    )
    var output_shape = IndexList[3](
        total_length, num_q_heads, kv_params.head_size
    )

    var row_offsets_runtime_layout = RuntimeLayout[
        row_offsets_layout
    ].row_major(row_offsets_shape)
    var cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(cache_lengths_shape)
    var q_ragged_runtime_layout = RuntimeLayout[q_ragged_layout].row_major(
        q_ragged_shape
    )
    var output_runtime_layout = RuntimeLayout[output_layout].row_major(
        output_shape
    )

    var input_row_offsets = ManagedLayoutTensor[.uint32, row_offsets_layout](
        row_offsets_runtime_layout, ctx
    )
    var cache_lengths_managed = ManagedLayoutTensor[
        .uint32, cache_lengths_layout
    ](cache_lengths_runtime_layout, ctx)
    var q_ragged = ManagedLayoutTensor[dtype, q_ragged_layout](
        q_ragged_runtime_layout, ctx
    )
    var test_output = ManagedLayoutTensor[dtype, output_layout](
        output_runtime_layout, ctx
    )

    var input_row_offsets_host = input_row_offsets.tensor[update=False]()
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()

    var running_offset: UInt32 = 0
    for i in range(batch_size):
        input_row_offsets_host[i] = running_offset
        cache_lengths_host[i] = UInt32(cache_lengths[i])
        running_offset += UInt32(valid_lengths[i])
    input_row_offsets_host[batch_size] = running_offset

    random(q_ragged.tensor())

    var num_paged_blocks = (
        ceildiv(max_full_context_length, page_size) * batch_size + 2
    )

    var kv_block_paged_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    # Pad LUT inner dim to honor `PagedKVCache.populate`'s SIMD padding
    # invariant — see `padded_lut_cols`. Production
    # (`max/python/max/kv_cache/paged_kv_cache/cache_manager.py`) does
    # the same; tests that bypass the padding hit either a chunk-
    # alignment debug_assert or an OOB SIMD read into adjacent memory.
    var paged_lut_shape = IndexList[2](
        batch_size,
        padded_lut_cols(ceildiv(max_full_context_length, page_size)),
    )

    var kv_block_paged_runtime_layout = RuntimeLayout[
        kv_block_6d_layout
    ].row_major(kv_block_paged_shape)
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )

    var kv_block_paged = ManagedLayoutTensor[dtype, kv_block_6d_layout](
        kv_block_paged_runtime_layout, ctx
    )
    var paged_lut = ManagedLayoutTensor[.uint32, paged_lut_layout](
        paged_lut_runtime_layout, ctx
    )

    var kv_block_paged_tensor = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_paged.tensor[update=False]().ptr,
        kv_block_paged_runtime_layout,
    )
    random(kv_block_paged_tensor)

    var paged_lut_tensor = paged_lut.tensor[update=False]()
    # Sample one distinct paged block per page across the whole batch up
    # front, then hand them out in iteration order. Total pages needed is
    # <= num_paged_blocks by construction.
    var total_pages = 0
    for bs in range(batch_size):
        total_pages += ceildiv(cache_lengths[bs] + valid_lengths[bs], page_size)
    var paged_blocks = random_distinct(num_paged_blocks, total_pages)
    var page_pos = 0
    for bs in range(batch_size):
        var seq_len = cache_lengths[bs] + valid_lengths[bs]
        for block_idx in range(0, ceildiv(seq_len, page_size)):
            paged_lut_tensor[bs, block_idx] = UInt32(paged_blocks[page_pos])
            page_pos += 1

    var cache_lengths_lt = cache_lengths_managed.device_tensor()
    var kv_block_paged_lt = kv_block_paged.device_tensor()
    var paged_lut_lt = paged_lut.device_tensor()

    var kv_collection_paged_device = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        kv_block_paged_lt,
        cache_lengths_lt,
        paged_lut_lt,
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    var q_ragged_lt = q_ragged.device_tensor()
    var test_output_lt = test_output.device_tensor()

    flash_attention[ragged=True](
        test_output_lt,
        q_ragged_lt,
        kv_collection_paged_device.get_key_cache(layer_idx),
        kv_collection_paged_device.get_value_cache(layer_idx),
        mask,
        input_row_offsets.device_tensor(),
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )
    ctx.synchronize()
    assert_no_nan_inf(test_output, label)


def run_gemma4_suite[
    local_mask_t: MHAMask
](local_mask: local_mask_t, ctx: DeviceContext) raises:
    """Run the gemma4 prefill+decode suite. `local_mask` is applied to the
    local (sliding) layer; the global layer always uses `CausalMask()` to
    match production."""
    # Gemma4 local (sliding) attention: layer_idx=0.
    # num_q_heads=32, num_kv_heads=16, head_dim=256, seq_len=11 prefill.
    print("Gemma4 local: q_heads=32 kv_heads=16 head_dim=256 seq_len=11")
    var ce_seq_lens_local: List = [11]
    var ce_cache_sizes_local: List = [0]
    execute_ragged_paged_flash_attention[
        32,
        DType.bfloat16,
        KVCacheStaticParams(num_heads=16, head_size=256),
    ](
        ce_seq_lens_local,
        ce_cache_sizes_local,
        2,
        0,
        local_mask,
        "local_ce_bs1",
        ctx,
    )

    # Gemma4 global (full) attention: layer_idx=5.
    # num_q_heads=32, num_kv_heads=4, head_dim=512, seq_len=11 prefill.
    print("Gemma4 global: q_heads=32 kv_heads=4 head_dim=512 seq_len=11")
    var ce_seq_lens_global: List = [11]
    var ce_cache_sizes_global: List = [0]
    execute_ragged_paged_flash_attention[
        32,
        DType.bfloat16,
        KVCacheStaticParams(num_heads=4, head_size=512),
    ](
        ce_seq_lens_global,
        ce_cache_sizes_global,
        2,
        0,
        CausalMask(),
        "global_ce_bs1",
        ctx,
    )

    # Also try batch_size=4 to mirror a realistic serving scenario.
    print("Gemma4 local bs=4")
    var ce_seq_lens_local_bs4: List = [11, 11, 11, 11]
    var ce_cache_sizes_local_bs4: List = [0, 0, 0, 0]
    execute_ragged_paged_flash_attention[
        32,
        DType.bfloat16,
        KVCacheStaticParams(num_heads=16, head_size=256),
    ](
        ce_seq_lens_local_bs4,
        ce_cache_sizes_local_bs4,
        2,
        0,
        local_mask,
        "local_ce_bs4",
        ctx,
    )

    print("Gemma4 global bs=4")
    var ce_seq_lens_global_bs4: List = [11, 11, 11, 11]
    var ce_cache_sizes_global_bs4: List = [0, 0, 0, 0]
    execute_ragged_paged_flash_attention[
        32,
        DType.bfloat16,
        KVCacheStaticParams(num_heads=4, head_size=512),
    ](
        ce_seq_lens_global_bs4,
        ce_cache_sizes_global_bs4,
        2,
        0,
        CausalMask(),
        "global_ce_bs4",
        ctx,
    )

    # Decode (token-generation) shapes. MAX attention graphs compile both
    # CE and TG kernels; the LLVM crash was bisected to the decode
    # scaffolding commit so exercise these here too.
    print("Gemma4 local decode: seq_len=1 cache_len=512")
    var tg_seq_lens_local: List = [1]
    var tg_cache_sizes_local: List = [512]
    execute_ragged_paged_flash_attention[
        32,
        DType.bfloat16,
        KVCacheStaticParams(num_heads=16, head_size=256),
    ](
        tg_seq_lens_local,
        tg_cache_sizes_local,
        2,
        0,
        local_mask,
        "local_tg",
        ctx,
    )

    print("Gemma4 global decode: seq_len=1 cache_len=512")
    var tg_seq_lens_global: List = [1]
    var tg_cache_sizes_global: List = [512]
    execute_ragged_paged_flash_attention[
        32,
        DType.bfloat16,
        KVCacheStaticParams(num_heads=4, head_size=512),
    ](
        tg_seq_lens_global,
        tg_cache_sizes_global,
        2,
        0,
        CausalMask(),
        "global_tg",
        ctx,
    )


def main() raises:
    # Compile-time switches. `page_size` mirrors the production KV cache page
    # size; `mask_kind` selects the local-layer mask family
    # (0 = causal everywhere; 1 = SlidingWindowCausalMask[1024] for local
    # layers, CausalMask for global — the realistic Gemma-4 production
    # config per `sliding_window=1024`). The global layer always uses
    # CausalMask in either case.
    comptime mask_kind = get_defined_int["mask_kind", 0]()

    with DeviceContext() as ctx:
        seed(42)
        comptime if mask_kind == 0:
            run_gemma4_suite(CausalMask(), ctx)
        else:
            run_gemma4_suite(SlidingWindowCausalMask[1024](), ctx)
        print("PASS")
