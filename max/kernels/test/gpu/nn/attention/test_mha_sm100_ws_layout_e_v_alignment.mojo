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

"""SM100 FA4 Layout-E (BM=64, MMA_M=64, m_pack=2) V-producer paged-LUT
alignment regression, at `page_size == config.BN` (256).

Target hardware family: NVIDIA SM100 (B200).

## The bug

`load_warp.mojo`'s `_produce_v_e` (Layout-E's per-partition V producer) walks
`kv_row_base + p * partition_keys` for `p` in `{0, 1}` (`partition_keys =
config.BN // config.m_pack == 128`), and hands each partition's base row to
`kv_lut.populate[partition_keys, base_alignment, ...]`. `base_alignment` there
was the TILE-level `MaskType.start_column_alignment[...]()` -- a promise about
`kv_row_base` (256-aligned for every admitted mask at `BN == 256`), not about
the `partition_keys`-row stride a partition sits at inside the tile.

At `page_size == 256 == config.BN`, `base_alignment % page_size == 0`, so
`PagedRowIndices.populate`'s `num_pages == 1` fast arm engages -- the one that
skips the `tok_in_block` divmod on the assumption that `base_kv_row` is already
page-aligned. That assumption holds for partition 0 (`p_base == kv_row_base`)
but not partition 1 (`p_base == kv_row_base + 128`, which is 128 mod 256, never
a multiple of 256). The fast arm silently returns `tok_in_block = 0` there
instead of 128, so partition 1's V load reads the first 128 rows of the SAME
physical page partition 0 already read, instead of its own back half.

This is reachable and silent in a production build: `supported()` admits every
`(depth, group, dtype)` cell in the shipping Layout-E grid at `page_size = 256`
(see the plan's host-side table), `kv_cache_page_size` has no validator
(`max/python/max/pipelines/kv_cache/config.py`), and with assertions off
`PagedRowIndices.populate`'s `debug_assert` on `base_kv_row % page_size == 0`
compiles out -- so this manifests as wrong keys, not a fault.

The fix (`load_warp.mojo`) is `v_e_base_alignment = gcd(base_alignment,
partition_keys)`, the same shape as the shared-key V-walk's
`v_sk_base_alignment` fix one section below it. `gcd(256, 128) == 128`, so
`128 % 256 != 0` takes `populate`'s general (`row_idx`, full divmod) arm
instead, which is correct for both partitions regardless of `p_base`'s
alignment.

## This test

`num_q_heads=64` / `kv_heads=8` (`group=8`), `head_size=64`, `page_size=256`,
`valid_length=8`. Per `test_mha_sm100_ws_bm32.mojo`'s module docstring:
Layout-G's `BM_eff() = 32/8 = 4` fails an 8-token prompt, so the auto route
falls through to Layout-E, whose `BM_eff() = 64/8 = 8` accepts it.

`cache_length=504` gives `num_keys=512`, two full 256-key WS tiles (`T=2`,
inside the `T <= 3` WS cap `test_mha_sm100_ws_bm32.mojo` documents), so BOTH
tiles' partition-1 V loads exercise the misaligned walk with real, distinct
per-page KV data -- not merely an OOB-zero-filled or single-tile corner case.
`CausalMask` at this cache/valid-length combination leaves nearly the whole
`[0, num_keys)` range visible to every query row, so both partitions of both
tiles are live.

Compared against `mha_gpu_naive` over the full `[0, num_keys)` range with the
same mask, at the bf16 floor used throughout this datapath's other tests
(`atol = 0.04`, `test_mha_sm100_ws_bm32.mojo`).
"""

from std.math import ceildiv, isnan, rsqrt
from std.random import seed
from std.utils import IndexList
from std.utils.numerics import nan

from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import CausalMask


comptime dtype = DType.bfloat16
comptime head_size = 64
comptime page_size = 256
comptime kv_params = KVCacheStaticParams(num_heads=8, head_size=head_size)
comptime num_q_heads = 64
comptime group = num_q_heads // kv_params.num_heads

comptime valid_length = 8
comptime cache_length = 504
comptime num_keys = cache_length + valid_length

# Inlined from `test_mha_sm100_ws_bm32.mojo`'s `_LUT_TAIL_PAD` /
# `padded_lut_cols`: `PagedRowIndices.populate`'s SIMD arm reads `num_pages`
# LUT columns starting at any valid `first_lut_idx` without clamping against
# the sequence's real page count, so the LUT needs tail padding beyond the
# last real page for that read to stay in-bounds.
comptime _LUT_TAIL_PAD = 16


def padded_lut_cols(cols: Int) -> Int:
    return ((cols + 7) // 8) * 8 + _LUT_TAIL_PAD


def main() raises:
    comptime row_offsets_layout = Layout(UNKNOWN_VALUE)
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime qo_layout = Layout.row_major(UNKNOWN_VALUE, num_q_heads, head_size)
    comptime paged_lut_layout = Layout.row_major[2]()
    comptime kv_block_layout = Layout.row_major[6]()
    comptime qo_shape = IndexList[3](valid_length, num_q_heads, head_size)
    comptime qo_size = valid_length * num_q_heads * head_size

    var scale = rsqrt(Float32(head_size))
    seed(0x5151)

    with DeviceContext() as ctx:
        var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](2)
        row_offsets_host[0] = 0
        row_offsets_host[1] = UInt32(valid_length)
        var row_offsets_dev = ctx.enqueue_create_buffer[.uint32](2)
        ctx.enqueue_copy(row_offsets_dev, row_offsets_host)
        var row_offsets = LayoutTensor[mut=False, .uint32, row_offsets_layout](
            row_offsets_dev,
            RuntimeLayout[row_offsets_layout].row_major(IndexList[1](2)),
        )

        var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](1)
        cache_lengths_host[0] = UInt32(cache_length)
        var cache_lengths_dev = ctx.enqueue_create_buffer[.uint32](1)
        ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
        var cache_lengths = LayoutTensor[
            mut=False, .uint32, cache_lengths_layout
        ](
            cache_lengths_dev,
            RuntimeLayout[cache_lengths_layout].row_major(IndexList[1](1)),
        )

        var q_host = ctx.enqueue_create_host_buffer[dtype](qo_size)
        random(
            LayoutTensor[dtype, qo_layout](
                q_host.unsafe_ptr(),
                RuntimeLayout[qo_layout].row_major(qo_shape),
            )
        )
        var q_dev = ctx.enqueue_create_buffer[dtype](qo_size)
        ctx.enqueue_copy(q_dev, q_host)
        var q = LayoutTensor[mut=False, dtype, qo_layout](
            q_dev, RuntimeLayout[qo_layout].row_major(qo_shape)
        )

        # One spare block, reserved as a NaN poison page (per
        # `test_mha_sm100_ws_bm32.mojo`): any TMA that reads past a
        # sequence's real pages -- rather than the misaligned-but-in-bounds
        # row this bug actually produces -- pulls non-finite data and fails
        # loudly instead of silently.
        var num_pages = ceildiv(num_keys, page_size)
        var num_blocks = num_pages + 1
        var poison_block = num_blocks - 1
        var kv_shape = IndexList[6](
            num_blocks, 2, 1, page_size, kv_params.num_heads, head_size
        )
        var kv_size = (
            num_blocks * 2 * page_size * kv_params.num_heads * head_size
        )
        var kv_host = ctx.enqueue_create_host_buffer[dtype](kv_size)
        random(
            LayoutTensor[dtype, kv_block_layout](
                kv_host.unsafe_ptr(),
                RuntimeLayout[kv_block_layout].row_major(kv_shape),
            )
        )
        var block_elems = 2 * page_size * kv_params.num_heads * head_size
        for i in range(block_elems):
            kv_host[poison_block * block_elems + i] = nan[dtype]()
        var kv_dev = ctx.enqueue_create_buffer[dtype](kv_size)
        ctx.enqueue_copy(kv_dev, kv_host)
        var kv_blocks = LayoutTensor[dtype, kv_block_layout](
            kv_dev, RuntimeLayout[kv_block_layout].row_major(kv_shape)
        )

        # Identity LUT (logical page `i` -> physical block `i`) -- this bug is
        # about the intra-page row offset (`tok_in_block`), not physical block
        # shuffling, so no permutation is needed. Tail columns point at the
        # poison block (see `padded_lut_cols`'s docstring above).
        var lut_cols = padded_lut_cols(num_pages)
        var lut_host = ctx.enqueue_create_host_buffer[.uint32](lut_cols)
        for i in range(lut_cols):
            lut_host[i] = UInt32(poison_block if i >= num_pages else i)
        var lut_dev = ctx.enqueue_create_buffer[.uint32](lut_cols)
        ctx.enqueue_copy(lut_dev, lut_host)
        var lut = LayoutTensor[mut=False, .uint32, paged_lut_layout](
            lut_dev,
            RuntimeLayout[paged_lut_layout].row_major(
                IndexList[2](1, lut_cols)
            ),
        )

        var kv_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
            kv_blocks.as_unsafe_any_origin(),
            cache_lengths,
            lut,
            UInt32(valid_length),
            UInt32(num_keys),
        )
        var k_cache = kv_collection.get_key_cache(0)
        var v_cache = kv_collection.get_value_cache(0)

        var test_dev = ctx.enqueue_create_buffer[dtype](qo_size)
        flash_attention[ragged=True](
            LayoutTensor[dtype, qo_layout](
                test_dev.unsafe_ptr(),
                RuntimeLayout[qo_layout].row_major(qo_shape),
            ),
            q,
            k_cache,
            v_cache,
            CausalMask(),
            row_offsets,
            scale,
            ctx,
        )

        var ref_dev = ctx.enqueue_create_buffer[dtype](qo_size)
        mha_gpu_naive[ragged=True](
            q,
            k_cache,
            v_cache,
            CausalMask(),
            LayoutTensor[dtype, qo_layout](
                ref_dev.unsafe_ptr(),
                RuntimeLayout[qo_layout].row_major(qo_shape),
            ),
            row_offsets,
            scale,
            1,
            valid_length,
            num_keys,
            num_q_heads,
            head_size,
            group,
            ctx,
        )

        var test_host = ctx.enqueue_create_host_buffer[dtype](qo_size)
        var ref_host = ctx.enqueue_create_host_buffer[dtype](qo_size)
        ctx.enqueue_copy(test_host, test_dev)
        ctx.enqueue_copy(ref_host, ref_dev)
        ctx.synchronize()

        var nan_count = 0
        var max_abs_diff = Float32(0)
        for i in range(qo_size):
            var got = test_host[i].cast[.float32]()
            if isnan(got):
                nan_count += 1
                continue
            var want = ref_host[i].cast[.float32]()
            max_abs_diff = max(max_abs_diff, abs(got - want))

        _ = q_dev^
        _ = kv_dev^
        _ = lut_dev^
        _ = row_offsets_dev^
        _ = cache_lengths_dev^
        _ = test_dev^
        _ = ref_dev^

        print("nan_count =", nan_count, " max-abs diff =", max_abs_diff)
        if nan_count > 0:
            raise Error("NaN x" + String(nan_count))
        if max_abs_diff > 0.04:
            raise Error("max-abs diff too large: " + String(max_abs_diff))
        print("PASSED")
