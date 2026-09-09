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
"""Minimal repro for LLVM AMDGPU register-allocator assertion on gfx950.

Reproduces the `IntervalMap.h:639 "Overlapping insert"` assertion hit by
`//max/tests/integration/architectures/gemma4:test_attention::test_attention_global`.

Crashing combination (bisected to commit 97e211bb0 "MHA decode through
amd_structured/"):

  - gfx950 structured prefill path
  - depth = 512 (global Gemma4 head_dim)
  - CausalMask
  - bfloat16
  - sink = True

The gemma4 test never passes `sink_weights` at runtime, but MOGG's
`unswitch[call_flash_attention](Bool(sink_weights))` forces comptime
specialization of both `sink=True` and `sink=False` kernels, and the
`sink=True` instantiation is what trips the LLVM assertion during
`AMDGPURewriteAGPRCopyMFMA::eliminateSpillsOfReassignedVGPRs`.

The assertion only fires under LLVM builds with asserts enabled (bazel
`k8-dbg`). Release-LLVM builds (direct `mojo` CLI) silently produce code
that may or may not be correct — run this through `./bazelw test`.
"""

from std.math import ceildiv, rsqrt
from std.random import seed
from std.collections import OptionalReg
from layout._utils import ManagedLayoutTensor
from max.gpu.host import DeviceContext

from kv_cache_test_utils import random_distinct
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from nn.attention.gpu.mha import (
    MHADecodeDispatchMetadata,
    flash_attention,
)
from nn.kv_cache_ragged import generic_flash_attention_kv_cache_ragged
from nn.attention.mha_mask import CausalMask

from std.utils import IndexList


def execute_sink_prefill_repro[
    num_q_heads: Int, dtype: DType, kv_params: KVCacheStaticParams
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
) raises:
    comptime page_size = 256

    var batch_size = len(valid_lengths)
    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]

    comptime row_offsets_layout = Layout.row_major(UNKNOWN_VALUE)
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    comptime paged_lut_layout = Layout.row_major[2]()
    comptime kv_block_6d_layout = Layout.row_major[6]()
    comptime sink_weights_layout = Layout.row_major(UNKNOWN_VALUE)

    var row_offsets_shape = IndexList[1](batch_size + 1)
    var cache_lengths_shape = IndexList[1](batch_size)
    var q_ragged_shape = IndexList[3](
        total_length, num_q_heads, kv_params.head_size
    )
    var output_shape = IndexList[3](
        total_length, num_q_heads, kv_params.head_size
    )

    var row_offsets_rt = RuntimeLayout[row_offsets_layout].row_major(
        row_offsets_shape
    )
    var cache_lengths_rt = RuntimeLayout[cache_lengths_layout].row_major(
        cache_lengths_shape
    )
    var q_ragged_rt = RuntimeLayout[q_ragged_layout].row_major(q_ragged_shape)
    var output_rt = RuntimeLayout[output_layout].row_major(output_shape)

    var input_row_offsets = ManagedLayoutTensor[.uint32, row_offsets_layout](
        row_offsets_rt, ctx
    )
    var cache_lengths_managed = ManagedLayoutTensor[
        .uint32, cache_lengths_layout
    ](cache_lengths_rt, ctx)
    var q_ragged = ManagedLayoutTensor[dtype, q_ragged_layout](q_ragged_rt, ctx)
    var test_output = ManagedLayoutTensor[dtype, output_layout](output_rt, ctx)

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
    var paged_lut_shape = IndexList[2](
        batch_size, ceildiv(max_full_context_length, page_size)
    )

    var kv_block_paged_rt = RuntimeLayout[kv_block_6d_layout].row_major(
        kv_block_paged_shape
    )
    var paged_lut_rt = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )

    var kv_block_paged = ManagedLayoutTensor[dtype, kv_block_6d_layout](
        kv_block_paged_rt, ctx
    )
    var paged_lut = ManagedLayoutTensor[.uint32, paged_lut_layout](
        paged_lut_rt, ctx
    )

    var kv_block_paged_tensor = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_paged.tensor[update=False]().ptr,
        kv_block_paged_rt,
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

    # Sink weights: one per query head, per spec at [num_q_heads].
    var sink_weights_shape = IndexList[1](num_q_heads)
    var sink_weights_rt = RuntimeLayout[sink_weights_layout].row_major(
        sink_weights_shape
    )
    var sink_weights_managed = ManagedLayoutTensor[dtype, sink_weights_layout](
        sink_weights_rt, ctx
    )
    random(sink_weights_managed.tensor())

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
    var input_row_offsets_dev = input_row_offsets.device_tensor()
    var input_row_offsets_lt = LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ](
        input_row_offsets_dev.ptr,
        input_row_offsets_dev.runtime_layout,
    )

    # Match the dispatch_metadata that MOGG assembles from the graph inputs
    # in `_unmarshal_mha_decode_dispatch_metadata` — hard-coded for the
    # gemma4 prefill case.
    var decode_dispatch_metadata = MHADecodeDispatchMetadata(
        batch_size,
        max_prompt_length,
        0,
        max_full_context_length,
    )

    # Route through the exact MOGG-level entry point used by
    # `mo.mha.ragged.paged`. `_flash_attention_dispatch` inside runs
    # `unswitch[call_flash_attention](Bool(sink_weights))`, forcing BOTH
    # sink=True and sink=False specializations to compile — which is the
    # distinguishing property of the gemma4 graph compile.
    var ctx_ptr = ctx
    generic_flash_attention_kv_cache_ragged[
        target="gpu",
        mask_str="causal",
    ](
        q_ragged_lt,
        input_row_offsets_lt,
        kv_collection_paged_device,
        UInt32(layer_idx),
        rsqrt(Float32(kv_params.head_size)),
        test_output_lt,
        ctx_ptr,
        decode_dispatch_metadata,
    )
    ctx.synchronize()


def main() raises:
    with DeviceContext() as ctx:
        seed(42)

        # Gemma4 global shape: num_q_heads=32, num_kv_heads=4, head_dim=512,
        # seq_len=11 prefill, CausalMask, bf16, paged, ragged — this is the
        # exact specialization that crashes during AMDGPURewriteAGPRCopyMFMA.
        print("Gemma4 global sink=True prefill repro")
        var seq_lens: List = [11]
        var cache_sizes: List = [0]
        execute_sink_prefill_repro[
            32,
            DType.bfloat16,
            KVCacheStaticParams(num_heads=4, head_size=512),
        ](seq_lens, cache_sizes, 2, 0, ctx)

        print("PASS")
