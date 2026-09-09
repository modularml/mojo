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
"""SM100 (B200) MLA prefill blockscale: large-cache_length finiteness sweep.

Companion to ``test_mla_prefill_blockscale_paged.mojo``. That test
exercises the SM100 dense FP8 MLA prefill kernel
(``mla_sm100_prefill_blockscale``) only at small ``num_keys`` with
``cache_length = 0`` (fresh self-attention) and a full naive reference.

This file targets the UNDER-tested axis: a small query (``seq_len``)
attending to a LARGE pre-existing cache (``cache_length`` up to 94715),
which is the production Kimi-K2.5 prefix-reuse geometry that triggers
the intermittent NaN-in-logits crash. The online-softmax accumulator
``li`` (rowsum) accumulates over ``num_keys = cache_length + seq_len``
keys; at ~94k keys it could overflow, or a num-partitions / block-count
limit could be hit, or an FP8 K_rope scale over a 94k range could
misbehave — any of which would surface as a non-finite output.

A full naive reference is infeasible at 94715 keys (the score matrix
alone is O(seq_len * 94715) per head and the reference attends
quadratically), so this test is finiteness-only: a NaN/Inf in the
attention output is never correct regardless of any reference. Small-
cache correctness lives in the companion reference test.

Geometry uses the REAL Kimi MLA dims: q_depth=192 (qk_nope 128 + rope
64), cache_depth=576 (kv_lora 512 + rope 64), kv_depth/v_head_dim=128.
``num_heads`` is a representative 16 (per-head softmax is independent of
head count). Per the paged-prefill geometry, ``cache_length(b)`` is the
kernel ``start_pos`` and the kernel attends to keys
``[0, cache_length + seq_len)``, so the physical paged K_rope cache and
the ragged K/V buffers are sized to ``cache_length + seq_len`` tokens.
"""

from std.math import ceildiv
from std.random import randn, seed
from std.sys import get_defined_int

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.memory import alloc
from std.utils.numerics import isinf, isnan
from nn.attention.mha_mask import CausalMask
from nn.attention.gpu.mla import flare_mla_prefill
from max.gpu.host.info import _is_sm10x_gpu
from std.utils.index import Index, IndexList
from std.testing import assert_equal

from _paged_prefill_test_utils import (
    CACHE_DEPTH,
    HEAD_DIM_GRAN,
    KV_NUM_HEADS,
    NUM_LAYERS,
    ROPE_DEPTH,
    SCALE_BLOCK_SIZE,
    fill_paged_block_scales,
    fill_paged_blocks_uniform,
    fill_uniform_lookup_table,
    lut_max_pages_per_batch,
    paged_block_elems,
    paged_scale_block_elems,
)


comptime PAGE_SIZE = get_defined_int["page_size", 256]()


def run_finite_check[
    qkv_type: DType,
    k_rope_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    kv_depth: Int,
    page_size: Int,
    batch_size: Int = 1,
](seq_len: Int, cache_length: Int, ctx: DeviceContext) raises -> Int:
    """Launches the blockscale MLA prefill with a large pre-existing cache and
    returns the number of non-finite output elements (must be 0)."""
    # The kernel attends to keys [0, cache_length + seq_len): size the
    # physical cache and ragged K/V to that total.
    var num_keys = cache_length + seq_len
    print(
        "  [large-cache] seq_len:",
        seq_len,
        " cache_length:",
        cache_length,
        " num_keys(total):",
        num_keys,
        " page_size:",
        page_size,
    )

    comptime scale = Float32(0.125)

    var q_size = batch_size * seq_len * num_heads * depth
    var k_size = batch_size * num_keys * num_heads * kv_depth
    var v_size = k_size
    var o_size = batch_size * seq_len * num_heads * kv_depth

    var q_ptr = alloc[Scalar[qkv_type]](q_size)
    var k_ptr = alloc[Scalar[qkv_type]](k_size)
    var v_ptr = alloc[Scalar[qkv_type]](v_size)
    var output_ptr = alloc[Scalar[output_type]](o_size)

    randn[qkv_type](q_ptr, q_size)
    randn[qkv_type](k_ptr, k_size)
    randn[qkv_type](v_ptr, v_size)

    # Row-offset tables. input_row_offsets indexes the query rows (seq_len);
    # cache_row_offsets indexes the ragged K/V rows (num_keys total).
    var input_row_offsets_host = alloc[UInt32](batch_size + 1)
    var cache_row_offsets_host = alloc[UInt32](batch_size + 1)
    for i in range(batch_size):
        input_row_offsets_host[i] = UInt32(i * seq_len)
        cache_row_offsets_host[i] = UInt32(i * num_keys)
    input_row_offsets_host[batch_size] = UInt32(batch_size * seq_len)
    cache_row_offsets_host[batch_size] = UInt32(batch_size * num_keys)

    var num_pages_per_batch = ceildiv(num_keys, page_size)
    var total_pages = batch_size * num_pages_per_batch
    var max_pages_per_batch = lut_max_pages_per_batch(num_keys, page_size)
    var lut_size = batch_size * max_pages_per_batch
    var block_elems = paged_block_elems(total_pages, page_size, CACHE_DEPTH)
    var scales_elems = paged_scale_block_elems(total_pages, page_size)

    var blocks_host = alloc[Scalar[k_rope_type]](block_elems)
    var scales_host = alloc[Float32](scales_elems)
    var cache_lengths_host = alloc[UInt32](batch_size)
    var lookup_table_host = alloc[UInt32](lut_size)

    fill_paged_blocks_uniform[k_rope_type](
        blocks_host, batch_size, num_keys, page_size
    )
    fill_paged_block_scales(scales_host, batch_size, num_keys, page_size)
    # cache_length(b) = the pre-existing prefix length (kernel start_pos).
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_length)
    fill_uniform_lookup_table(
        lookup_table_host,
        batch_size,
        num_keys,
        page_size,
        max_pages_per_batch,
    )

    var q_device_buf = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_buf = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_buf = ctx.enqueue_create_buffer[qkv_type](v_size)
    var output_device_buf = ctx.enqueue_create_buffer[output_type](o_size)
    var input_ro_buf = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var cache_ro_buf = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var blocks_device = ctx.enqueue_create_buffer[k_rope_type](block_elems)
    var scales_device = ctx.enqueue_create_buffer[.float32](scales_elems)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)

    # Zero-init the output so an un-written cell reads as a clean 0, not
    # uninitialized garbage that a finite-scan would mis-flag.
    output_device_buf.enqueue_fill(Scalar[output_type](0))

    ctx.enqueue_copy(q_device_buf, q_ptr)
    ctx.enqueue_copy(k_device_buf, k_ptr)
    ctx.enqueue_copy(v_device_buf, v_ptr)
    ctx.enqueue_copy(input_ro_buf, input_row_offsets_host)
    ctx.enqueue_copy(cache_ro_buf, cache_row_offsets_host)
    ctx.enqueue_copy(blocks_device, blocks_host)
    ctx.enqueue_copy(scales_device, scales_host)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    ctx.synchronize()

    var q_device = TileTensor(
        q_device_buf,
        row_major((batch_size * seq_len, Idx[num_heads], Idx[depth])),
    )
    var k_device = TileTensor(
        k_device_buf,
        row_major((batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var v_device = TileTensor(
        v_device_buf,
        row_major((batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var output_device = TileTensor(
        output_device_buf,
        row_major((batch_size * seq_len, Idx[num_heads], Idx[kv_depth])),
    )
    var input_ro_tt = TileTensor(input_ro_buf, row_major(batch_size + 1))
    var cache_ro_tt = TileTensor(cache_ro_buf, row_major(batch_size + 1))

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=CACHE_DEPTH, is_mla=True
    )
    var block_shape = IndexList[6](
        total_pages,
        1,
        NUM_LAYERS,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var scales_shape = IndexList[6](
        total_pages,
        1,
        NUM_LAYERS,
        page_size,
        kv_params.num_heads,
        HEAD_DIM_GRAN,
    )

    var blocks_lt = LayoutTensor[k_rope_type, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )
    var scales_lt = LayoutTensor[.float32, Layout.row_major[6]()](
        scales_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(scales_shape),
    )

    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_lt = LayoutTensor[.uint32, cl_layout](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )

    comptime lt_layout_2d = Layout.row_major[2]()
    var lookup_table_lt = LayoutTensor[.uint32, lt_layout_2d](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, max_pages_per_batch)
        ),
    )

    var kv_collection = PagedKVCacheCollection[
        k_rope_type,
        kv_params,
        page_size,
        scale_dtype_=DType.float32,
        quantization_granularity_=SCALE_BLOCK_SIZE,
    ](
        LayoutTensor[k_rope_type, Layout.row_major[6]()](
            blocks_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks_lt.runtime_layout.shape.value,
                blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, cl_layout](
            cache_lengths_lt.ptr,
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, lt_layout_2d](
            lookup_table_lt.ptr,
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(seq_len),
        UInt32(num_keys),
        LayoutTensor[.float32, Layout.row_major[6]()](
            scales_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                scales_lt.runtime_layout.shape.value,
                scales_lt.runtime_layout.stride.value,
            ),
        ),
    )

    var kv_cache = kv_collection.get_key_cache(0)
    var k_lt = k_device.to_layout_tensor()
    var v_lt = v_device.to_layout_tensor()

    print("    launching...")
    flare_mla_prefill[rank=3](
        output_device,
        q_device,
        k_lt,
        v_lt,
        kv_cache,
        CausalMask(),
        input_ro_tt,
        cache_ro_tt,
        scale,
        ctx,
        q_max_seq_len=seq_len,
    )
    ctx.synchronize()
    print("    kernel completed (no crash).")

    ctx.enqueue_copy(output_ptr, output_device_buf)
    ctx.synchronize()

    # Finite-check every kernel-written output element.
    var num_nonfinite = 0
    for i in range(o_size):
        var val = output_ptr.load(i).cast[.float32]()
        if isnan(val) or isinf(val):
            num_nonfinite += 1

    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    output_ptr.free()
    blocks_host.free()
    scales_host.free()
    cache_lengths_host.free()
    lookup_table_host.free()
    input_row_offsets_host.free()
    cache_row_offsets_host.free()

    _ = q_device_buf
    _ = k_device_buf
    _ = v_device_buf
    _ = output_device_buf
    _ = input_ro_buf
    _ = cache_ro_buf
    _ = blocks_device
    _ = scales_device
    _ = cache_lengths_device
    _ = lookup_table_device

    return num_nonfinite


def main() raises:
    with DeviceContext() as ctx:
        comptime if _is_sm10x_gpu(ctx.default_device_info):
            # Sweep cache_length over {0, 1k, 8k, 32k, 94715}. A small
            # seq_len keeps the query side cheap; the swept axis is the
            # cached prefix that the softmax accumulator sums over.
            comptime seq_len = 4
            var cache_lengths = [0, 1024, 8192, 32768, 94715]
            var first_break = -1
            for idx in range(len(cache_lengths)):
                var cl = cache_lengths[idx]
                seed(0)
                # cache_length == 0 is a fresh prefill of `seq_len` tokens;
                # use a larger seq_len there so the 0 case still exercises a
                # non-trivial accumulation.
                var sl = 256 if cl == 0 else seq_len
                var n = run_finite_check[
                    qkv_type=DType.bfloat16,
                    k_rope_type=DType.float8_e4m3fn,
                    output_type=DType.bfloat16,
                    depth=192,
                    num_heads=16,
                    kv_depth=128,
                    page_size=PAGE_SIZE,
                    batch_size=1,
                ](sl, cl, ctx)
                print(
                    "  >>> cache_length=",
                    cl,
                    " non-finite outputs =",
                    n,
                )
                if n > 0 and first_break < 0:
                    first_break = cl

            if first_break >= 0:
                print(
                    "FINITENESS BREAKS at cache_length =",
                    first_break,
                    "(instrument li/mi/scale next)",
                )
            else:
                print(
                    "ALL cache_length values FINITE up to 94715 (dense"
                    " attention clean — pivot to up-proj GEMM #1)"
                )
            # Hard assertion: a non-finite MLA prefill output at any swept
            # cache_length is a regression (first_break stays -1 when clean).
            assert_equal(
                first_break,
                -1,
                msg="MLA prefill output non-finite at a swept cache_length",
            )
