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

"""Regression test for the SM100 FA4 1Q attention-sink double-count bug.

Target hardware family: NVIDIA SM100 (B200).

## What this guards

`fa4_softmax` (`nn/attention/gpu/nvidia/sm100/softmax_warp.mojo`) adds the
attention-sink mass `exp2(sink - row_max)` to the softmax denominator
(`row_sum`). In the FA4 **1Q** path (`config.num_qo == 1`) BOTH warpgroups
cover the same Q rows but stride over disjoint halves of the K/V stream, then
LSE-combine their `row_sum`s into the final denominator. Without a warpgroup
guard the sink was added by *both* WGs and therefore counted twice in
`global_sum`, inflating the denominator and shrinking every output element.
gpt-oss-20b (head_dim=64, with sinks) failed logit verification because of
this (max-KL 0.43 vs 0.06). In the 2Q path the WGs own disjoint Q rows, so the
unguarded add is already correct.

## Why the oracle is independent

The pre-existing ragged/paged attention test only checks paged-vs-continuous
self-consistency — both sides run the SAME 1Q kernel, so the double-count
cancels and the bug is invisible. This test instead compares the FA4 paged
output against `mha_gpu_naive[..., sink=True]`, a structurally different
kernel (`_bmm0` + `_softmax_gpu` + `_bmm1`). Both kernels read the *same*
paged KV blocks and the *same* per-head sink weights, so the only thing that
differs is the attention math. A double-counted sink in FA4 then shows up as a
divergence from naive.

## Path selection (verified against the dispatch in `mha.mojo`)

`flash_attention[ragged=True]` on a paged cache sets
`is_token_generation = (max_prompt_len == 1) and not empty_cache()`
(`mha.mojo` ~L424). A true decode (`valid_length == 1` with a non-empty cache)
takes the SM100 FA4 `fa4_route` (`mha.mojo`) → `mha_sm100_2q_dispatch`, which
runs `fa4_softmax` for the 1Q sink path. To exercise that 1Q path we therefore
use a prefill with `max_prompt_len` in `(32, 128]`:

  * lower bound `> 32`: `max_prompt_len <= 32` now routes to the WS BM=32
    packed-TMEM datapath (`fa4_config_ws` -- also `fa4_softmax`, but the 8-way
    split + two-level combine, NOT the 1Q BM=128 peer-WG combine this cell
    guards). The `not sink` gate that used to keep sink shapes off WS was lifted
    in Phase B, so a short-prompt sink shape would otherwise land on WS.
  * `valid_length` in `(32, 128]` -> `is_token_generation == False` (prefill),
    and `max_prompt_len <= 128` so the FA4 dispatch heuristic
    (`sm100/dispatch.mojo`) selects `fa4_config_1q` (BM = 128).
  * a long cache (`cache_length` >> `2*BN`) so the per-WG K-tile count
    `T = ceil(num_keys / (2*BN)) >= 2`, i.e. both WGs are active and the
    LSE-combine actually runs. For depth=64/page_size=128, BN=128, so a cache
    of 512 gives num_keys >= 514 and T = 3.

  * `valid_length >= 256` -> `max_prompt_len > 128` selects `fa4_config_2q`
    (BM = 256), the disjoint-rows path that is correct with or without the fix.

If a future change reroutes these shapes to a different config -- the WS BM=32
route (`max_prompt_len <= 32`), the decode kernel (`max_prompt_len == 1`), or
`fa4_config_2q` (`max_prompt_len > 128`) -- the sink-on + 1Q cell stops being a
regression test for the 1Q BM=128 peer-WG combine; the bounds above are the
contract. (Phase C widens the WS route past 32 under a grid rule; C3 must
re-audit that this cell still lands on 1Q.)
"""

from std.collections import Set
from std.math import ceildiv, rsqrt
from std.random import rand, random_ui64, seed

from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import CausalMask, MHAMask

from std.utils import IndexList


# Mirror of `padded_lut_cols` in `kv_cache_test_utils.mojo` (inlined to avoid a
# cross-package bazel dep on the kv_cache test utilities). `PagedKVCache`'s
# SIMD populate path requires the LUT row stride to be a multiple of 8 and at
# least `cols + 15`; production allocates with this padding so tests must too.
comptime _LUT_TAIL_PAD = 16


def padded_lut_cols(cols: Int) -> Int:
    return ((cols + 7) // 8) * 8 + _LUT_TAIL_PAD


# ===-----------------------------------------------------------------------===#
# Core test
# ===-----------------------------------------------------------------------===#


def execute_1q_sink_test[
    num_q_heads: Int,
    kv_params: KVCacheStaticParams,
    mask_t: MHAMask,
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    use_sink: Bool,
    cell_name: StaticString,
    mask: mask_t,
    ctx: DeviceContext,
) raises -> Float32:
    """Run FA4 paged attention and `mha_gpu_naive` off the SAME paged KV
    cache and the SAME per-head sinks, return the max-abs output diff.

    `use_sink` toggles the sink path on both sides; `cell_name` labels the
    bisection cell in the diff print. The caller asserts on the returned diff
    so every bisection cell runs even when one fails.
    """
    comptime dtype = DType.bfloat16
    comptime page_size = 128
    comptime num_layers = 1
    comptime layer_idx = 0
    comptime head_size = kv_params.head_size
    comptime group = num_q_heads // kv_params.num_heads

    var batch_size = len(valid_lengths)

    # Dimensions.
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
        "  cell=",
        cell_name,
        " sink=",
        use_sink,
        " valid=",
        valid_lengths[0],
        " cache=",
        cache_lengths[0],
        " max_prompt_len=",
        max_prompt_length,
        " num_keys=",
        max_full_context_length,
        sep="",
    )

    # --- Layouts ---
    comptime row_offsets_layout = Layout(UNKNOWN_VALUE)
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_size
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_size
    )
    comptime paged_lut_layout = Layout.row_major[2]()
    comptime kv_block_6d_layout = Layout.row_major[6]()
    comptime sink_layout = Layout.row_major(UNKNOWN_VALUE)

    # --- Host metadata: row offsets + cache lengths ---
    var input_row_offsets = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var running_offset: UInt32 = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_offset
        cache_lengths_host[i] = UInt32(cache_lengths[i])
        running_offset += UInt32(valid_lengths[i])
    input_row_offsets[batch_size] = running_offset

    var input_row_offsets_dev = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var cache_lengths_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(input_row_offsets_dev, input_row_offsets)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)

    # --- Q (ragged: [total_length, num_q_heads, head_size]) ---
    var q_size = total_length * num_q_heads * head_size
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    var q_host_tt = LayoutTensor[dtype, q_ragged_layout](
        q_host.unsafe_ptr(),
        RuntimeLayout[q_ragged_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )
    random(q_host_tt)
    var q_dev = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q_dev, q_host)

    # --- Paged KV blocks ---
    var num_paged_blocks = (
        ceildiv(max_full_context_length, page_size) * batch_size + 4
    )
    var kv_block_paged_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        head_size,
    )
    var kv_block_size = (
        num_paged_blocks
        * 2
        * num_layers
        * page_size
        * kv_params.num_heads
        * head_size
    )
    var kv_block_host = ctx.enqueue_create_host_buffer[dtype](kv_block_size)
    var kv_block_host_tt = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_host.unsafe_ptr(),
        RuntimeLayout[kv_block_6d_layout].row_major(kv_block_paged_shape),
    )
    random(kv_block_host_tt)
    var kv_block_dev = ctx.enqueue_create_buffer[dtype](kv_block_size)
    ctx.enqueue_copy(kv_block_dev, kv_block_host)

    # --- Paged lookup table (unique random physical blocks per (bs, blk)) ---
    var lut_cols = padded_lut_cols(ceildiv(max_full_context_length, page_size))
    var paged_lut_shape = IndexList[2](batch_size, lut_cols)
    var paged_lut_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size * lut_cols
    )
    var paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        var seq_len = cache_lengths[bs] + valid_lengths[bs]
        for block_idx in range(ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
            paged_lut_set.add(randval)
            paged_lut_host[bs * lut_cols + block_idx] = UInt32(randval)
    var paged_lut_dev = ctx.enqueue_create_buffer[.uint32](
        batch_size * lut_cols
    )
    ctx.enqueue_copy(paged_lut_dev, paged_lut_host)

    # --- Per-head sink weights (random) ---
    # `num_q_heads` entries: `mha_gpu_naive`/`fa4_softmax` index sinks by the
    # query head, so the sink array length matches the q-head count.
    var sinks_host = ctx.enqueue_create_host_buffer[dtype](num_q_heads)
    if use_sink:
        # rand fills [0, 1); spread to ~[-2, 6) so the sink materially shifts
        # the denominator (making any double-count large) without dominating
        # it. Per-head values exercise the head-indexed sink lookup.
        rand(sinks_host.as_span())
        for h in range(num_q_heads):
            sinks_host[h] = (
                sinks_host[h].cast[.float32]() * Float32(8.0) - Float32(2.0)
            ).cast[dtype]()
    else:
        sinks_host.as_span().fill(Scalar[dtype](0))
    var sinks_dev = ctx.enqueue_create_buffer[dtype](num_q_heads)
    ctx.enqueue_copy(sinks_dev, sinks_host)

    # Device views. KV-cache collection requires `blocks` at MutAnyOrigin and
    # `cache_lengths`/`lookup_table` at ImmutAnyOrigin; the ragged FA4 path
    # wants `input_row_offsets` immutable. Bake the origins into the types.
    var input_row_offsets_lt = LayoutTensor[
        mut=False, .uint32, row_offsets_layout
    ](
        input_row_offsets_dev,
        RuntimeLayout[row_offsets_layout].row_major(
            IndexList[1](batch_size + 1)
        ),
    )
    var cache_lengths_lt = LayoutTensor[
        mut=False, .uint32, cache_lengths_layout
    ](
        cache_lengths_dev,
        RuntimeLayout[cache_lengths_layout].row_major(IndexList[1](batch_size)),
    )
    var paged_lut_lt = LayoutTensor[mut=False, .uint32, paged_lut_layout](
        paged_lut_dev,
        RuntimeLayout[paged_lut_layout].row_major(paged_lut_shape),
    )
    var kv_block_paged_lt = LayoutTensor[
        dtype,
        kv_block_6d_layout,
    ](
        kv_block_dev,
        RuntimeLayout[kv_block_6d_layout].row_major(kv_block_paged_shape),
    )
    var q_lt = LayoutTensor[mut=False, dtype, q_ragged_layout](
        q_dev,
        RuntimeLayout[q_ragged_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )
    var sinks_lt = LayoutTensor[mut=False, dtype, sink_layout](
        sinks_dev.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[sink_layout].row_major(IndexList[1](num_q_heads)),
    )

    var kv_collection = PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
    ](
        # `flash_attention`/`mha_gpu_naive` read both the `k` and `v` cache views,
        # which are disjoint kv_idx halves of one `blocks` buffer sharing its origin,
        # so the nested-origin exclusivity check rejects passing both. Declare the
        # kv_block_paged_lt origin as UnsafeAnyOrigin to opt the out of exclusivity checking.
        kv_block_paged_lt.as_unsafe_any_origin(),
        cache_lengths_lt,
        paged_lut_lt,
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    var k_cache = kv_collection.get_key_cache(layer_idx)
    var v_cache = kv_collection.get_value_cache(layer_idx)
    var scale = rsqrt(Float32(head_size))

    # --- FA4 paged output ---
    var test_out_size = total_length * num_q_heads * head_size
    var test_out_dev = ctx.enqueue_create_buffer[dtype](test_out_size)
    var test_out_lt = LayoutTensor[dtype, output_layout](
        test_out_dev.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )

    if use_sink:
        flash_attention[ragged=True, sink=True](
            test_out_lt,
            q_lt,
            k_cache,
            v_cache,
            mask,
            input_row_offsets_lt,
            scale,
            ctx,
            sink_weights=sinks_lt,
        )
    else:
        flash_attention[ragged=True](
            test_out_lt,
            q_lt,
            k_cache,
            v_cache,
            mask,
            input_row_offsets_lt,
            scale,
            ctx,
        )

    # --- Independent oracle: naive attention off the SAME paged cache ---
    var ref_out_dev = ctx.enqueue_create_buffer[dtype](test_out_size)
    var ref_out_lt = LayoutTensor[dtype, output_layout](
        ref_out_dev.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )

    if use_sink:
        mha_gpu_naive[ragged=True, sink=True](
            q_lt,
            k_cache,
            v_cache,
            mask,
            ref_out_lt,
            input_row_offsets_lt,
            scale,
            batch_size,
            max_prompt_length,
            max_full_context_length,
            num_q_heads,
            head_size,
            group,
            ctx,
            sinks_lt,
        )
    else:
        mha_gpu_naive[ragged=True](
            q_lt,
            k_cache,
            v_cache,
            mask,
            ref_out_lt,
            input_row_offsets_lt,
            scale,
            batch_size,
            max_prompt_length,
            max_full_context_length,
            num_q_heads,
            head_size,
            group,
            ctx,
        )

    ctx.synchronize()

    var test_out_host = ctx.enqueue_create_host_buffer[dtype](test_out_size)
    var ref_out_host = ctx.enqueue_create_host_buffer[dtype](test_out_size)
    ctx.enqueue_copy(test_out_host, test_out_dev)
    ctx.enqueue_copy(ref_out_host, ref_out_dev)
    ctx.synchronize()

    # Per-element compare + report max-abs diff for the cell.
    var max_abs_diff: Float32 = 0.0
    var argmax_idx = 0
    for i in range(test_out_size):
        var a = test_out_host[i].cast[.float32]()
        var b = ref_out_host[i].cast[.float32]()
        var d = abs(a - b)
        if d > max_abs_diff:
            max_abs_diff = d
            argmax_idx = i
    print(
        "    max-abs diff(FA4, naive) =",
        max_abs_diff,
        " at flat idx",
        argmax_idx,
    )

    _ = q_dev^
    _ = kv_block_dev^
    _ = paged_lut_dev^
    _ = input_row_offsets_dev^
    _ = cache_lengths_dev^
    _ = sinks_dev^
    _ = test_out_dev^
    _ = ref_out_dev^

    return max_abs_diff


# ===-----------------------------------------------------------------------===#
# Entry point
# ===-----------------------------------------------------------------------===#


def main() raises:
    # gpt-oss-20b shape: head_size=64, 64 q-heads, 8 kv-heads (group 8).
    comptime num_q_heads = 64
    comptime kv_params = KVCacheStaticParams(num_heads=8, head_size=64)

    with DeviceContext() as ctx:
        seed(0x5151)
        var causal = CausalMask()

        print("test_mha_sm100_1q_sink: gpt-oss-20b shape (hd=64, 64q/8kv)")

        # 2x2 bisection: {sink on, sink off} x {1Q (short prefill, long
        # cache), 2Q (long prefill)}. The sink-on + 1Q cell is the one that
        # regressed; the other three must pass with or without the fix.
        #
        # 1Q: valid in (32,128] (prefill, 32 < max_prompt_len <= 128 ->
        #     fa4_config_1q, BM=128; the `> 32` keeps it OFF the WS BM=32 route
        #     that now accepts sink after Phase B), long cache (512 >> 2*BN=256)
        #     so T>=2 and both WGs run the peer-WG LSE combine this cell guards.
        var valid_1q: List = [96]
        var cache_1q: List = [512]
        # 2Q: valid >= 256 (max_prompt_len>128 -> fa4_config_2q, BM=256).
        var valid_2q: List = [320]
        var cache_2q: List = [0]

        # Run all four cells before asserting so the bisection table is always
        # complete (one failing cell does not hide the other three).
        comptime atol = Float32(1e-2)
        var d_1q_on = execute_1q_sink_test[num_q_heads, kv_params](
            valid_1q, cache_1q, True, "1Q_sink_on", causal, ctx
        )
        var d_1q_off = execute_1q_sink_test[num_q_heads, kv_params](
            valid_1q, cache_1q, False, "1Q_sink_off", causal, ctx
        )
        var d_2q_on = execute_1q_sink_test[num_q_heads, kv_params](
            valid_2q, cache_2q, True, "2Q_sink_on", causal, ctx
        )
        var d_2q_off = execute_1q_sink_test[num_q_heads, kv_params](
            valid_2q, cache_2q, False, "2Q_sink_off", causal, ctx
        )

        print("=== bisection table (max-abs diff vs naive, atol=1e-2) ===")
        print(
            "  1Q sink_on :",
            d_1q_on,
            " ->",
            "PASS" if d_1q_on <= atol else "FAIL",
        )
        print(
            "  1Q sink_off:",
            d_1q_off,
            " ->",
            "PASS" if d_1q_off <= atol else "FAIL",
        )
        print(
            "  2Q sink_on :",
            d_2q_on,
            " ->",
            "PASS" if d_2q_on <= atol else "FAIL",
        )
        print(
            "  2Q sink_off:",
            d_2q_off,
            " ->",
            "PASS" if d_2q_off <= atol else "FAIL",
        )

        var n_fail = 0
        if d_1q_on > atol:
            n_fail += 1
        if d_1q_off > atol:
            n_fail += 1
        if d_2q_on > atol:
            n_fail += 1
        if d_2q_off > atol:
            n_fail += 1
        if n_fail > 0:
            raise Error(
                String(n_fail)
                + " bisection cell(s) exceeded atol=1e-2 vs naive"
            )

        print("test_mha_sm100_1q_sink: ALL PASSED")
