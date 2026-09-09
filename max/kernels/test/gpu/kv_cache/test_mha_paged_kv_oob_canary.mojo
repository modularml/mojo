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

"""OOB canary stress test for paged flash attention.

Drives `flash_attention[ragged=True]` over the Gemma-4 production shapes
with KV-cache blocks NOT referenced by the LUT pre-filled with `+inf`.
Any spurious read into an unreferenced block produces `+inf` at QK^T,
which flows through softmax (`+inf - max = NaN`) to the output. The
post-kernel `assert_no_nan_inf` then names the failure with element
index — no need to know the exact OOB pattern in advance.

Targets KERN-2861: NaN at `page_size=128` on Gemma-4. Sweeps page_size
via `-D page_size=N` and mask kind via `-D mask_kind={0=causal,
1=sliding_1024}`. Iterates a small randomized seed range to exercise
multiple LUT permutations (different "used" block sets).
"""

from std.math import ceildiv, rsqrt
from std.random import seed
from std.sys.defines import get_defined_int
from std.utils import IndexList
from std.utils.numerics import max_or_inf

from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
from max.gpu.host import DeviceContext

from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from kv_cache_test_utils import assert_no_nan_inf, padded_lut_cols, randperm
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalMask, MHAMask, SlidingWindowCausalMask


def execute_oob_canary[
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
    comptime page_size = get_defined_int["page_size", 128]()

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

    # Random Q in a modest range so attention output stays finite for the
    # correct path.
    random(q_ragged.tensor())

    var num_paged_blocks = (
        ceildiv(max_full_context_length, page_size) * batch_size + 4
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
    # invariant — see `padded_lut_cols`. Trailing entries are
    # initialized below to a reserved sentinel block_idx so any future
    # OOB read into them deterministically hits a `+inf`-poisoned
    # block.
    var paged_lut_shape = IndexList[2](
        batch_size,
        padded_lut_cols(ceildiv(max_full_context_length, page_size)),
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

    # Random KV first; then we'll overwrite the unreferenced blocks below.
    var kv_block_paged_host = kv_block_paged.tensor[update=False]()
    var kv_block_paged_tensor = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_paged_host.ptr,
        kv_block_paged_rt,
    )
    random(kv_block_paged_tensor)

    # Build paged LUT, reserving the last block as a sentinel: it is never
    # assigned to a real sequence, so it stays unreferenced and falls into the
    # `+inf`-poisoning loop below. Trailing LUT padding entries point at the
    # sentinel so any spurious read into them deterministically hits a poisoned
    # block.
    var sentinel_block_idx = num_paged_blocks - 1
    var paged_lut_tensor = paged_lut.tensor[update=False]()
    var lut_padded_cols = padded_lut_cols(
        ceildiv(max_full_context_length, page_size)
    )

    # Sample distinct real blocks from the non-sentinel population
    # `[0, num_paged_blocks - 1)` via a random permutation, handing them out
    # in iteration order. After the assignment loop `page_pos` counts the
    # blocks we assigned, so the ones we did NOT assign are exactly the
    # permutation tail `block_perm[page_pos:]`. The canary fill below walks
    # that tail (plus the reserved sentinel, which sits outside the sampled
    # population), with no membership `Set` and no per-block `in` probe.
    var block_perm = randperm(num_paged_blocks - 1)

    var page_pos = 0
    for bs in range(batch_size):
        var seq_len = cache_lengths[bs] + valid_lengths[bs]
        var num_real_pages = ceildiv(seq_len, page_size)
        for block_idx in range(0, num_real_pages):
            paged_lut_tensor[bs, block_idx] = UInt32(block_perm[page_pos])
            page_pos += 1
        # Fill the trailing LUT padding with the poisoned sentinel.
        for col in range(num_real_pages, lut_padded_cols):
            paged_lut_tensor[bs, col] = UInt32(sentinel_block_idx)

    # Canary fill: every block we did NOT assign is filled with +inf. A
    # correct kernel never reads from these blocks (or reads through a mask
    # that zeros them out). A buggy kernel that reads here picks up `+inf`,
    # which flows through softmax to NaN at the output. The unassigned blocks
    # are the permutation tail plus the reserved sentinel.
    var inf_val = max_or_inf[dtype]()
    comptime block_size_elements_const = (
        2 * page_size * kv_params.num_heads * kv_params.head_size
    )
    var block_size_elements = num_layers * block_size_elements_const
    for i in range(page_pos, len(block_perm)):
        var offset = block_perm[i] * block_size_elements
        for j in range(block_size_elements):
            kv_block_paged_host.ptr[offset + j] = inf_val
    var sentinel_offset = sentinel_block_idx * block_size_elements
    for j in range(block_size_elements):
        kv_block_paged_host.ptr[sentinel_offset + j] = inf_val

    var cache_lengths_lt = cache_lengths_managed.device_tensor()
    var kv_block_paged_lt = kv_block_paged.device_tensor()
    var paged_lut_lt = paged_lut.device_tensor()

    var kv_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
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
        kv_collection.get_key_cache(layer_idx),
        kv_collection.get_value_cache(layer_idx),
        mask,
        input_row_offsets.device_tensor(),
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )
    ctx.synchronize()
    assert_no_nan_inf(test_output, label)


def run_canary_suite[
    local_mask_t: MHAMask
](local_mask: local_mask_t, ctx: DeviceContext) raises:
    """Run the gemma4 prefill+decode shapes with the OOB canary fill.

    `local_mask` applies to the local-layer call (head_dim=256,
    num_kv_heads=16) which uses sliding-window in production. The
    global-layer call (head_dim=512, num_kv_heads=4) always uses
    `CausalMask()` to match production.
    """
    # Local prefill (small CE).
    print("OOB canary local CE bs=1")
    var ce_lens: List = [11]
    var ce_caches: List = [0]
    execute_oob_canary[
        32,
        DType.bfloat16,
        KVCacheStaticParams(num_heads=16, head_size=256),
    ](ce_lens, ce_caches, 2, 0, local_mask, "local_ce_bs1", ctx)

    # Local prefill (bs=4) — exercises multi-batch LUT randomization.
    print("OOB canary local CE bs=4")
    var ce_lens_bs4: List = [11, 11, 11, 11]
    var ce_caches_bs4: List = [0, 0, 0, 0]
    execute_oob_canary[
        32,
        DType.bfloat16,
        KVCacheStaticParams(num_heads=16, head_size=256),
    ](ce_lens_bs4, ce_caches_bs4, 2, 0, local_mask, "local_ce_bs4", ctx)

    # Local decode with non-trivial cache_len: most blocks are assigned to the
    # cache, so unassigned canary blocks are scarce — but this is exactly the
    # gemma4 production decode shape and the failure mode under KERN-2861.
    print("OOB canary local TG bs=1 cache_len=512")
    var tg_lens: List = [1]
    var tg_caches: List = [512]
    execute_oob_canary[
        32,
        DType.bfloat16,
        KVCacheStaticParams(num_heads=16, head_size=256),
    ](tg_lens, tg_caches, 2, 0, local_mask, "local_tg", ctx)

    # Global prefill / decode: always CausalMask (matches production).
    print("OOB canary global CE bs=1")
    execute_oob_canary[
        32,
        DType.bfloat16,
        KVCacheStaticParams(num_heads=4, head_size=512),
    ](ce_lens, ce_caches, 2, 0, CausalMask(), "global_ce_bs1", ctx)

    print("OOB canary global TG bs=1 cache_len=512")
    execute_oob_canary[
        32,
        DType.bfloat16,
        KVCacheStaticParams(num_heads=4, head_size=512),
    ](tg_lens, tg_caches, 2, 0, CausalMask(), "global_tg", ctx)


def main() raises:
    # See `_GEMMA4_SHAPES_CONFIGS` in BUILD.bazel for the page_size /
    # mask_kind sweep. mask_kind: 0=causal, 1=sliding_1024.
    comptime mask_kind = get_defined_int["mask_kind", 1]()

    with DeviceContext() as ctx:
        # Iterate a few seeds so the LUT permutation (and therefore the
        # used/unused block split) varies. Catches mask-driven boundary
        # bugs that only manifest at specific physical block layouts.
        for s in range(4):
            seed(42 + s)
            print("=== seed=", 42 + s, " ===")
            comptime if mask_kind == 0:
                run_canary_suite(CausalMask(), ctx)
            else:
                run_canary_suite(SlidingWindowCausalMask[1024](), ctx)
        print("PASS")
