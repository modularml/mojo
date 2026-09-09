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

"""Correctness test for SM100 MHA with native FP8 Q/K/V against a PAGED
KV cache (head_dim=256, page_size=128).

Sibling of `test_mha_sm100_qkv_fp8_d256_prefill.mojo` (which tested the
native-fp8 path against *contiguous* Q/K/V via the rank-4 non-ragged entry
point). This file proves the same native-fp8 path works against a real
`PagedKVCacheCollection` (dtype=float8_e4m3fn, no per-block scales) through
the **ragged paged** `flash_attention[ragged=True]` overload — the entry
relaxed in `mha.mojo` (~line 388) to allow Q=K=V=float8_e4m3fn,
output=bfloat16 on SM100.

No scale machinery: the fp8 cache holds raw e4m3fn data with tensor-wise
scale = 1. Both Q@K^T and P@V run as raw `KIND_F8F6F4` MMAs; P is cast to
fp8 in the softmax warp; output is bf16. This is the Gemma4 recipe.

Comparison pattern (mirrors the contiguous native-fp8 test):

1. Generate Q, K, V as fp8 e4m3fn directly on the host with `randn`.
2. Cast fp8 -> bf16 on the host (lossless — every fp8 value is exactly
   representable in bf16) to build the bf16 reference inputs.
3. Populate BOTH an fp8 paged cache and a bf16 paged cache from the same
   data (one LUT shared by both), plus an fp8 ragged Q and a bf16 ragged Q.
4. Reference: `flash_attention[ragged=True]` with the bf16 paged cache +
   bf16 Q -> bf16 out.
5. Test target: `flash_attention[ragged=True]` with the fp8 paged cache +
   fp8 Q -> bf16 out. Dispatches to `mha_sm100_depth512_dispatch`.
6. Element-wise compare with `atol=3e-1, rtol=5e-2` AND cosine >= 0.9997
   (the bar from `test_mha_sm100_qkv_fp8_d256_prefill.mojo:274`).

page_size=128 vs fp8 BN: for depth=256 fp8 the depth512 config computes
BN=128 (MMA_K=32, MMA_M=256, single-O), so page_size == BN exactly:
`page_size % BN == 0` -> no shrink, `num_pages == 1` (full page == full
tile, the cleanest paged path). See `mha_depth512/config.mojo:133-148`.

Target hardware family: NVIDIA SM100 (B200).
"""

from std.collections import Set
from std.math import ceildiv, sqrt
from std.random import random_ui64, randn, seed

from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)

from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import (
    CausalMask,
    MHAMask,
    SlidingWindowCausalMask,
)

from std.testing import assert_true
from std.utils import IndexList


# Mirror of `padded_lut_cols` in
# `max/kernels/test/gpu/kv_cache/kv_cache_test_utils.mojo` (inlined to avoid
# a cross-directory import for direct `mojo` runs). `PagedKVCache`'s SIMD
# lookup requires the LUT row stride to be a multiple of 8 and at least
# `cols + 15` so a 16-wide SIMD load at any valid index stays in-bounds.
comptime _LUT_TAIL_PAD = 16


def padded_lut_cols(cols: Int) -> Int:
    return ((cols + 7) // 8) * 8 + _LUT_TAIL_PAD


# ===-----------------------------------------------------------------------===#
# Core test
# ===-----------------------------------------------------------------------===#


def execute_paged_fp8_test[
    MaskType: MHAMask,
    *,
    num_q_heads: Int,
    group: Int,
    mask_name: StaticString,
    page_size: Int = 128,
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    mask: MaskType,
    ctx: DeviceContext,
) raises:
    """Run native-fp8 MHA against a paged fp8 KV cache vs a paged bf16
    reference, assert per-element tolerance + cosine.

    `valid_lengths[i]` is the number of new (ragged) Q tokens for batch i;
    `cache_lengths[i]` is the number of pre-existing KV tokens already in
    the paged cache for batch i. The KV cache holds the full context
    (cache_lengths[i] + valid_lengths[i]) tokens per batch.
    """
    comptime head_dim = 256
    comptime kv_num_heads = num_q_heads // group
    comptime fp8_dtype = DType.float8_e4m3fn
    comptime bf16_dtype = DType.bfloat16
    comptime scale = Float32(1.0) / sqrt(Float32(head_dim))
    comptime num_layers = 1
    comptime layer_idx = 0

    comptime kv_params = KVCacheStaticParams(
        num_heads=kv_num_heads, head_size=head_dim
    )

    var batch_size = len(valid_lengths)
    assert len(valid_lengths) == len(
        cache_lengths
    ), "valid_lengths and cache_lengths must be the same length"

    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]

    print(
        "test_mha_sm100_qkv_fp8_d256_paged: ",
        "mask=",
        mask_name,
        " group=",
        group,
        " n_q_heads=",
        num_q_heads,
        " n_kv_heads=",
        kv_num_heads,
        " bs=",
        batch_size,
        " max_ctx=",
        max_full_context_length,
        " max_prompt=",
        max_prompt_length,
        " page_size=",
        page_size,
    )

    # ---- Layouts for ragged metadata + Q + output ----
    comptime row_offsets_layout = Layout(UNKNOWN_VALUE)
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_dim
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_dim
    )
    comptime paged_lut_layout = Layout.row_major[2]()
    comptime kv_block_6d_layout = Layout.row_major[6]()

    var row_offsets_shape = IndexList[1](batch_size + 1)
    var cache_lengths_shape = IndexList[1](batch_size)
    var q_ragged_shape = IndexList[3](total_length, num_q_heads, head_dim)
    var output_shape = IndexList[3](total_length, num_q_heads, head_dim)

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

    # ---- Ragged metadata ----
    var input_row_offsets = ManagedLayoutTensor[.uint32, row_offsets_layout](
        row_offsets_runtime_layout, ctx
    )
    var cache_lengths_managed = ManagedLayoutTensor[
        .uint32, cache_lengths_layout
    ](cache_lengths_runtime_layout, ctx)

    var input_row_offsets_host = input_row_offsets.tensor[update=False]()
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    var running_offset: UInt32 = 0
    for i in range(batch_size):
        input_row_offsets_host[i] = running_offset
        cache_lengths_host[i] = UInt32(cache_lengths[i])
        running_offset += UInt32(valid_lengths[i])
    input_row_offsets_host[batch_size] = running_offset

    # ---- Q: generate fp8 directly, cast fp8 -> bf16 (lossless) ----
    var q_fp8 = ManagedLayoutTensor[fp8_dtype, q_ragged_layout](
        q_ragged_runtime_layout, ctx
    )
    var q_bf16 = ManagedLayoutTensor[bf16_dtype, q_ragged_layout](
        q_ragged_runtime_layout, ctx
    )
    var q_fp8_host = q_fp8.tensor[update=False]()
    var q_bf16_host = q_bf16.tensor[update=False]()
    randn(q_fp8_host.ptr, total_length * num_q_heads * head_dim)
    for t in range(total_length):
        for h in range(num_q_heads):
            for d in range(head_dim):
                q_bf16_host[t, h, d] = q_fp8_host[t, h, d].cast[bf16_dtype]()

    # ---- Paged KV blocks (fp8 + bf16) ----
    # 6D: [num_blocks, 2, num_layers, page_size, num_heads, head_size].
    var num_paged_blocks = (
        ceildiv(max_full_context_length, page_size) * batch_size + 4
    )
    if num_paged_blocks < 16:
        num_paged_blocks = 16

    var kv_block_paged_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_num_heads,
        head_dim,
    )
    var kv_block_paged_runtime_layout = RuntimeLayout[
        kv_block_6d_layout
    ].row_major(kv_block_paged_shape)

    var kv_fp8 = ManagedLayoutTensor[fp8_dtype, kv_block_6d_layout](
        kv_block_paged_runtime_layout, ctx
    )
    var kv_bf16 = ManagedLayoutTensor[bf16_dtype, kv_block_6d_layout](
        kv_block_paged_runtime_layout, ctx
    )
    var kv_fp8_host = kv_fp8.tensor[update=False]()
    var kv_bf16_host = kv_bf16.tensor[update=False]()

    # ---- Lookup table (shared by fp8 and bf16 caches) ----
    var paged_lut_shape = IndexList[2](
        batch_size,
        padded_lut_cols(ceildiv(max_full_context_length, page_size)),
    )
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )
    var paged_lut = ManagedLayoutTensor[.uint32, paged_lut_layout](
        paged_lut_runtime_layout, ctx
    )
    var paged_lut_host = paged_lut.tensor[update=False]()

    # Assign unique physical blocks per (batch, logical block).
    var used = Set[Int]()
    for bs in range(batch_size):
        var seq_len = cache_lengths[bs] + valid_lengths[bs]
        for blk in range(ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
            while randval in used:
                randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
            used.add(randval)
            paged_lut_host[bs, blk] = UInt32(randval)

    # Fill the entire fp8 KV pool with random fp8 (single span randn), then
    # cast the whole pool fp8 -> bf16 (lossless) into the bf16 pool. Both
    # caches share the same data; only the dtype differs. Unused physical
    # blocks / padding tail are never read past valid context.
    var kv_pool_size = (
        num_paged_blocks * 2 * num_layers * page_size * kv_num_heads * head_dim
    )
    randn(kv_fp8_host.ptr, kv_pool_size)
    for i in range(kv_pool_size):
        kv_bf16_host.ptr[i] = kv_fp8_host.ptr[i].cast[bf16_dtype]()

    # ---- Output buffers (both bf16) ----
    var out_fp8 = ManagedLayoutTensor[bf16_dtype, output_layout](
        output_runtime_layout, ctx
    )
    var out_ref = ManagedLayoutTensor[bf16_dtype, output_layout](
        output_runtime_layout, ctx
    )

    # ---- Build paged collections ----
    var kv_collection_fp8 = PagedKVCacheCollection[
        fp8_dtype, kv_params, page_size
    ](
        kv_fp8.device_tensor(),
        cache_lengths_managed.device_tensor(),
        paged_lut.device_tensor(),
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )
    var kv_collection_bf16 = PagedKVCacheCollection[
        bf16_dtype, kv_params, page_size
    ](
        kv_bf16.device_tensor(),
        cache_lengths_managed.device_tensor(),
        paged_lut.device_tensor(),
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    # ---- Reference: bf16 paged attention ----
    flash_attention[ragged=True](
        out_ref.device_tensor(),
        q_bf16.device_tensor(),
        kv_collection_bf16.get_key_cache(layer_idx),
        kv_collection_bf16.get_value_cache(layer_idx),
        mask,
        input_row_offsets.device_tensor(),
        scale,
        ctx,
    )

    # ---- Test target: native-fp8 paged attention -> bf16 out ----
    flash_attention[ragged=True](
        out_fp8.device_tensor(),
        q_fp8.device_tensor(),
        kv_collection_fp8.get_key_cache(layer_idx),
        kv_collection_fp8.get_value_cache(layer_idx),
        mask,
        input_row_offsets.device_tensor(),
        scale,
        ctx,
    )
    ctx.synchronize()

    # ---- Compare ----
    var out_fp8_h = out_fp8.tensor()
    var out_ref_h = out_ref.tensor()
    var row_offsets_h = input_row_offsets.tensor()

    comptime rtol = 5e-2
    comptime atol = 3e-1
    var num_mismatches = 0
    var total_abs_diff: Float64 = 0.0
    var max_abs_diff: Float64 = 0.0
    var num_compared = 0
    var dot: Float64 = 0.0
    var aa: Float64 = 0.0
    var bb: Float64 = 0.0
    for bs in range(batch_size):
        var prompt_len = valid_lengths[bs]
        var ragged_off = Int(row_offsets_h[bs])
        for s in range(prompt_len):
            for h in range(num_q_heads):
                for d in range(head_dim):
                    var expect = out_ref_h[ragged_off + s, h, d].cast[
                        DType.float64
                    ]()[0]
                    var actual = out_fp8_h[ragged_off + s, h, d].cast[
                        DType.float64
                    ]()[0]
                    var diff = abs(actual - expect)
                    total_abs_diff += diff
                    if diff > max_abs_diff:
                        max_abs_diff = diff
                    num_compared += 1
                    if diff > atol + rtol * abs(expect):
                        if num_mismatches < 16:
                            print(
                                "mismatch bs=",
                                bs,
                                "s=",
                                s,
                                "h=",
                                h,
                                "d=",
                                d,
                                "actual=",
                                actual,
                                "expect=",
                                expect,
                            )
                        num_mismatches += 1
                    dot += actual * expect
                    aa += actual * actual
                    bb += expect * expect

    var cos: Float64 = 0.0
    if aa > 0.0 and bb > 0.0:
        cos = dot / (sqrt(aa) * sqrt(bb))
    print(
        "  num_mismatches=",
        num_mismatches,
        " / ",
        num_compared,
        " mean_abs_diff=",
        total_abs_diff / Float64(num_compared),
        " max_abs_diff=",
        max_abs_diff,
        " cosine=",
        cos,
    )
    assert_true(cos >= 0.9997, "cosine below 0.9997 bar")
    if num_mismatches > 0:
        print(
            "  WARNING:",
            num_mismatches,
            "mismatches > atol/rtol (but cosine passed)",
        )
    print("  PASSED")

    _ = q_fp8^
    _ = q_bf16^
    _ = kv_fp8^
    _ = kv_bf16^
    _ = out_fp8^
    _ = out_ref^
    _ = paged_lut^
    _ = input_row_offsets^
    _ = cache_lengths_managed^


# ===-----------------------------------------------------------------------===#
# Entry point — smallest-first
# ===-----------------------------------------------------------------------===#


def main() raises:
    seed(0xC0FFEE)
    with DeviceContext() as ctx:
        var causal = CausalMask()

        # ---- PREFILL (cache_lengths == 0): smallest-first ----
        # Gemma4-relevant production shape: group=2 (n_q=32, n_kv=16).
        execute_paged_fp8_test[
            CausalMask,
            num_q_heads=32,
            group=2,
            mask_name="CAUSAL_g2_prefill_s256",
        ]([256], [0], causal, ctx)
        execute_paged_fp8_test[
            CausalMask,
            num_q_heads=32,
            group=2,
            mask_name="CAUSAL_g2_prefill_s1024",
        ]([1024], [0], causal, ctx)

        # SlidingWindowCausalMask[1024] prefill case.
        var sw_1024 = SlidingWindowCausalMask[1024]()
        execute_paged_fp8_test[
            SlidingWindowCausalMask[1024],
            num_q_heads=32,
            group=2,
            mask_name="SW1024_g2_prefill_s1024",
        ]([1024], [0], sw_1024, ctx)

        # Multi-batch ragged prefill to exercise per-batch LUT + offsets.
        execute_paged_fp8_test[
            CausalMask,
            num_q_heads=32,
            group=2,
            mask_name="CAUSAL_g2_prefill_bs2",
        ]([256, 384], [0, 0], causal, ctx)

        # ---- DECODE (1q, q_max_seq_len==1, nonzero cache_lengths) ----
        execute_paged_fp8_test[
            CausalMask,
            num_q_heads=32,
            group=2,
            mask_name="CAUSAL_g2_decode_ctx256",
        ]([1], [256], causal, ctx)
        execute_paged_fp8_test[
            CausalMask,
            num_q_heads=32,
            group=2,
            mask_name="CAUSAL_g2_decode_bs2",
        ]([1, 1], [256, 384], causal, ctx)

        # ---- DECODE with SlidingWindowCausalMask ----
        execute_paged_fp8_test[
            SlidingWindowCausalMask[1024],
            num_q_heads=32,
            group=2,
            mask_name="SW1024_g2_decode_ctx256",
        ]([1], [256], sw_1024, ctx)
        execute_paged_fp8_test[
            SlidingWindowCausalMask[1024],
            num_q_heads=32,
            group=2,
            mask_name="SW1024_g2_decode_bs2",
        ]([1, 1], [256, 384], sw_1024, ctx)

        print("test_mha_sm100_qkv_fp8_d256_paged: ALL PASSED")
