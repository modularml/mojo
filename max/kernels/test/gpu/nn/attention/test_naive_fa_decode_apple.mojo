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
"""Apple (Metal) split-K decode attention correctness check.

Ensures `naive_fa_decode_apple` matches the in-tree `mha_gpu_naive` reference on
a paged KV cache, for single-head, MHA, and GQA decode shapes with varied
per-sequence cache lengths. Apple silicon GPU (Metal) only; decode-only.

Mirrors the harness in
`max/kernels/test/gpu/kv_cache/test_batch_kv_cache_flash_attention_causal_mask.mojo`
(`ContinuousBatchingKVCacheCollection` + `mha_gpu_naive` with `CausalMask`). Both
kernels are invoked with the same args and `_use_valid_length=True` so the
reference honors per-sequence `cache_length`. bf16 storage / fp32 accumulation,
compared in the bf16 tolerance band.
"""

from std.collections import Set
from std.math import rsqrt
from std.random import random_ui64, seed

from max.gpu.host import DeviceContext
from std.sys import has_apple_gpu_accelerator
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from layout._fillers import random
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.gpu.apple.naive_fa_decode import naive_fa_decode_apple
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_operand import KVCacheMHAOperand
from std.testing import assert_almost_equal
from std.utils import Index, IndexList


def execute_decode_compare[
    num_q_heads: Int,
    kv_params: KVCacheStaticParams,
    dtype: DType = .bfloat16,
](
    cache_lengths: List[Int],
    max_seq_len_cache: Int,
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
    kernel_max_cache_size: Int = 0,
) raises -> Float64:
    """Return the max-abs error (fp32) between `naive_fa_decode_apple` and the
    `mha_gpu_naive` reference on one decode batch.
    """
    comptime group = num_q_heads // kv_params.num_heads
    comptime depth = kv_params.head_size
    comptime num_blocks = 32

    var batch_size = len(cache_lengths)
    debug_assert(
        batch_size < num_blocks,
        "batch_size larger than configured num_blocks",
    )

    # Decode: exactly one query token per sequence.
    comptime max_prompt_len = 1
    var max_context_len = 0
    for i in range(batch_size):
        max_context_len = max(
            max_context_len, cache_lengths[i] + max_prompt_len
        )

    # ---- q: padded layout [batch, max_prompt_len=1, num_q_heads, depth] ---- #
    comptime q_static_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_q_heads, depth
    )
    var q_shape = IndexList[4](batch_size, max_prompt_len, num_q_heads, depth)
    var q_runtime_layout = RuntimeLayout[q_static_layout].row_major(q_shape)
    var q = ManagedLayoutTensor[dtype, q_static_layout](q_runtime_layout, ctx)
    random(q.tensor())

    # ---- valid_length: one query token per sequence ([1, 1, ...]) -------- #
    var valid_lengths = ManagedLayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            Index(batch_size)
        ),
        ctx,
    )
    var valid_lengths_host = valid_lengths.tensor[update=False]()
    for i in range(batch_size):
        valid_lengths_host[i] = UInt32(max_prompt_len)

    # ---- output tensors (oracle + under-test), same padded layout -------- #
    var output_shape = IndexList[4](
        batch_size, max_prompt_len, num_q_heads, depth
    )
    var output_runtime_layout = RuntimeLayout[q_static_layout].row_major(
        output_shape
    )
    var ref_output = ManagedLayoutTensor[dtype, q_static_layout](
        output_runtime_layout, ctx
    )
    var test_output = ManagedLayoutTensor[dtype, q_static_layout](
        output_runtime_layout, ctx
    )

    # ---- per-sequence cache lengths (varied) ----------------------------- #
    var cache_lengths_managed = ManagedLayoutTensor[
        .uint32, Layout(UNKNOWN_VALUE)
    ](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(Index(batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])
    var cache_lengths_device = cache_lengths_managed.device_tensor()

    # ---- paged kv_block: [num_blocks, 2, num_layers, max_seq, kv_heads, d] - #
    comptime kv_block_static_layout = Layout.row_major[6]()
    var kv_block_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        max_seq_len_cache,
        kv_params.num_heads,
        depth,
    )
    var kv_block_runtime_layout = RuntimeLayout[
        kv_block_static_layout
    ].row_major(kv_block_shape)
    var kv_block = ManagedLayoutTensor[dtype, kv_block_static_layout](
        kv_block_runtime_layout, ctx
    )
    random(kv_block.tensor())

    # ---- lookup table: distinct random block index per sequence ---------- #
    var lookup_table = ManagedLayoutTensor[.uint32, Layout(UNKNOWN_VALUE)](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(Index(batch_size)),
        ctx,
    )
    var lookup_table_host = lookup_table.tensor[update=False]()
    var block_idx_set = Set[Int]()
    var idx = 0
    while len(block_idx_set) < batch_size:
        var randval = Int(random_ui64(0, num_blocks - 1))
        if randval in block_idx_set:
            continue
        block_idx_set.add(randval)
        lookup_table_host[idx] = UInt32(randval)
        idx += 1

    # ---- build the KV collection + pull K/V operands --------------------- #
    var q_tensor = q.device_tensor()
    var valid_lengths_tensor = valid_lengths.device_tensor()
    var ref_output_tensor = ref_output.device_tensor()
    var test_output_tensor = test_output.device_tensor()
    var kv_block_tensor = kv_block.device_tensor()
    var lookup_table_tensor = lookup_table.device_tensor()

    var kv_collection_device = ContinuousBatchingKVCacheCollection[
        dtype,
        kv_params,
    ](
        # `mha_gpu_naive`/`naive_fa_decode_apple` read both the `k` and `v` cache
        # views, which are disjoint kv_idx halves of one `blocks` buffer sharing its
        # origin, so the nested-origin exclusivity check rejects passing both.
        # Declare kv_block_tensor origin as UnsafeAnyOrigin` to opt out of exclusivity checking.
        kv_block_tensor.as_unsafe_any_origin(),
        cache_lengths_device,
        lookup_table_tensor,
        UInt32(max_prompt_len),
        UInt32(max_context_len),
    )
    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    var scale = rsqrt(Float32(depth))

    # ---- oracle: mha_gpu_naive KVCacheT overload (wraps the cache in a
    # `KVCacheMHAOperand` and forwards with `_use_valid_length=True`,
    # `_is_cache_length_accurate=False` — mha.mojo:5696-5751). For decode this
    # makes the new token attend `cache_length + cur_query_len` keys.
    mha_gpu_naive(
        q_tensor,
        k_cache_device,
        v_cache_device,
        CausalMask(),
        ref_output_tensor,
        valid_lengths_tensor,
        scale,
        batch_size,
        max_prompt_len,
        max_context_len,
        num_q_heads,
        depth,
        group,
        ctx,
    )

    # ---- under test: naive_fa_decode_apple (SAME args). It takes raw
    # `MHAOperand`s, so wrap the cache identically to the KVCacheT overload
    # above; flags must mirror that overload's forwarded values.
    var k_operand = KVCacheMHAOperand(k_cache_device)
    var v_operand = KVCacheMHAOperand(v_cache_device)
    # Lets a regression feed the kernel a tighter max_cache_size than the oracle,
    # to exercise the launcher's partition sizing.
    var test_max_cache_size = (
        max_context_len if kernel_max_cache_size == 0 else kernel_max_cache_size
    )
    naive_fa_decode_apple[
        _use_valid_length=True, _is_cache_length_accurate=False
    ](
        q_tensor,
        k_operand,
        v_operand,
        CausalMask(),
        test_output_tensor,
        valid_lengths_tensor,
        scale,
        batch_size,
        max_prompt_len,
        test_max_cache_size,
        num_q_heads,
        depth,
        group,
        ctx,
    )

    ctx.synchronize()

    # ---- compare in fp32, bf16 tolerance band ---------------------------- #
    var ref_out = ref_output.tensor()
    var test_out = test_output.tensor()
    var max_abs_err = Float64(0.0)
    for bs in range(batch_size):
        for h in range(num_q_heads):
            for hd in range(depth):
                var ref_val = ref_out[bs, 0, h, hd]
                var test_val = test_out[bs, 0, h, hd]
                var rv = ref_val.cast[.float64]()[0]
                var tv = test_val.cast[.float64]()[0]
                var err = abs(rv - tv)
                max_abs_err = max(max_abs_err, err)
                assert_almost_equal(
                    test_val,
                    ref_val,
                    atol=2e-2,
                    rtol=2e-2,
                )
    return max_abs_err


# kv_params: num_heads is the KV head count; head_size is depth.
comptime kv_params_single = KVCacheStaticParams(num_heads=1, head_size=64)
comptime kv_params_mha = KVCacheStaticParams(num_heads=8, head_size=64)
comptime kv_params_gqa = KVCacheStaticParams(num_heads=2, head_size=64)
# d128/d256 exercise EPL=4/8; 256 is the Qwen3.5 full-attention dim.
comptime kv_params_gqa_d128 = KVCacheStaticParams(num_heads=2, head_size=128)
comptime kv_params_mha_d256 = KVCacheStaticParams(num_heads=8, head_size=256)
comptime kv_params_gqa_d256 = KVCacheStaticParams(num_heads=2, head_size=256)


def run_all(ctx: DeviceContext) raises:
    # Varied per-sequence cache lengths: include a non-multiple of SplitSize=32
    # (70, 96) and a short one (33). page/seq cache budget = 1024.
    var cache_lengths: List[Int] = [128, 96, 70, 33]
    comptime max_seq = 1024

    print("=== single-head [num_q=1, num_kv=1, group=1] ===")
    var e0 = execute_decode_compare[1, kv_params_single](
        cache_lengths, max_seq, 1, 0, ctx
    )
    print("  PASS  max-abs-err:", e0)

    print("=== MHA [num_q=8, num_kv=8, group=1] ===")
    var e1 = execute_decode_compare[8, kv_params_mha](
        cache_lengths, max_seq, 1, 0, ctx
    )
    print("  PASS  max-abs-err:", e1)

    print("=== GQA [num_q=8, num_kv=2, group=4] ===")
    var e2 = execute_decode_compare[8, kv_params_gqa](
        cache_lengths, max_seq, 1, 0, ctx
    )
    print("  PASS  max-abs-err:", e2)

    # Second shape: include a length-1 cache (and a non-multiple of SplitSize).
    var cache_lengths_short: List[Int] = [1, 31, 33]
    print("=== GQA short/length-1 caches [1, 31, 33] ===")
    var e3 = execute_decode_compare[8, kv_params_gqa](
        cache_lengths_short, max_seq, 1, 0, ctx
    )
    print("  PASS  max-abs-err:", e3)

    # Partition off-by-one regression: a tight, mult-of-32 max_cache_size leaves
    # the launcher one partition short unless it accounts for cur_query_len.
    var cache_lengths_mult32: List[Int] = [128, 64, 96, 32]
    print("=== off-by-one regression (tight mult-of-32 max_cache_size) ===")
    var e4 = execute_decode_compare[8, kv_params_gqa](
        cache_lengths_mult32, max_seq, 1, 0, ctx, kernel_max_cache_size=128
    )
    print("  PASS  max-abs-err:", e4)

    print("=== GQA head_dim=128 [num_q=8, num_kv=2, group=4] ===")
    var e5 = execute_decode_compare[8, kv_params_gqa_d128](
        cache_lengths, max_seq, 1, 0, ctx
    )
    print("  PASS  max-abs-err:", e5)

    print("=== MHA head_dim=256 [num_q=8, num_kv=8, group=1] ===")
    var e6 = execute_decode_compare[8, kv_params_mha_d256](
        cache_lengths, max_seq, 1, 0, ctx
    )
    print("  PASS  max-abs-err:", e6)

    print("=== GQA head_dim=256 (Qwen3.5) [num_q=8, num_kv=2, group=4] ===")
    var e7 = execute_decode_compare[8, kv_params_gqa_d256](
        cache_lengths, max_seq, 1, 0, ctx
    )
    print("  PASS  max-abs-err:", e7)


def main() raises:
    comptime if not has_apple_gpu_accelerator():
        print("SKIP: naive_fa_decode_apple targets Apple silicon GPUs only")
        return
    seed(42)
    with DeviceContext() as ctx:
        run_all(ctx)
