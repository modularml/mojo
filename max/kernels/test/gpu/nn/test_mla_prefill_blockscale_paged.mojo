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
"""Test MLA prefill with a paged FP8 K_rope cache (blockscale kernel).

Companion to ``test_mla_prefill_generic_paged.mojo``: exercises the
blockwise-FP8 SM100 MLA prefill kernel
(``mla_prefill_blockscale.mojo``) under the same partial-page configs.
Routing to ``mla_sm100_prefill_blockscale`` is automatic — the
``flare_mla_prefill[mla.mojo:1517]`` paged overload (the same one the
generic test uses) dispatches to blockscale via ``mla_sm100_prefill``
(``mla_prefill.mojo:58–103``) when ``KRopeType.dtype != KVType.dtype``.
Here we pass bf16 K/V (RaggedMHAOperand) and an FP8
``PagedKVCacheCollection`` for K_rope with integrated FP32 block-
scales (``scale_dtype_=DType.float32, quantization_granularity_=64``),
so the dispatch chooses blockscale.

Why this test exists
====================

The existing ``test_mla_prefill_blockwise_fp8.mojo`` covers only the
contiguous K_rope path (``LayoutTensorMHAOperand``) and so does not
exercise the kernel's paged sub-tile TMA loops at lines
643/731/850/955. The kernel-side ``debug_assert`` placed in those
loops fires whenever a partial-tile config tries to issue a TMA past
the sequence end (the silent partial-page bug — see
``docs/plans/sorted-sauteeing-snowglobe.md`` and
``test_mla_prefill_generic_paged.mojo`` for the full discovery
history).

Configs whose ``num_rope_pages`` loop hits a fully-OOB sub-page (and
therefore fail pre-fix): ``ps64_nk64``, ``ps32_nk96``,
``ps16_nk17``, ``ps16_nk100``. Configs where every issued sub-page
has a valid first row (and therefore pass both pre- and post-fix):
``ps256_nk256``, ``ps128_nk128``, ``ps128_nk256``, ``ps128_nk100``,
``ps64_nk256``, ``ps64_nk100``, ``ps32_nk100``.
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
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_operand import LayoutTensorMHAOperand
from nn.attention.gpu.mla import flare_mla_prefill
from std.testing import assert_almost_equal
from max.gpu.host.info import _is_sm10x_gpu
from std.utils.index import Index, IndexList

from _paged_prefill_test_utils import (
    CACHE_DEPTH,
    HEAD_DIM_GRAN,
    KV_NUM_HEADS,
    NUM_LAYERS,
    ROPE_DEPTH,
    SCALE_BLOCK_SIZE,
    extract_dequantized_k_rope_for_batch,
    fill_paged_block_scales,
    fill_paged_blocks_uniform,
    fill_uniform_lookup_table,
    lut_max_pages_per_batch,
    num_keys_to_test,
    paged_block_elems,
    paged_scale_block_elems,
)


# ===-----------------------------------------------------------------------===#
# Compile-time parameterisation. ``num_keys`` is iterated at runtime; only
# ``page_size`` requires a separate compilation since
# ``PagedKVCacheCollection[..., page_size]`` is parameterised on it.
# ===-----------------------------------------------------------------------===#

comptime PAGE_SIZE = get_defined_int["page_size", 256]()


# ===-----------------------------------------------------------------------===#
# Test scaffolding
# ===-----------------------------------------------------------------------===#


def run_test_paged_prefill_blockscale[
    qkv_type: DType,
    k_rope_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    kv_depth: Int,
    page_size: Int,
    batch_size: Int = 1,
](seq_len: Int, num_keys: Int, ctx: DeviceContext) raises:
    print(
        "test_mla_prefill_blockscale_paged",
        " batch_size:",
        batch_size,
        " seq_len:",
        seq_len,
        " num_keys:",
        num_keys,
        " page_size:",
        page_size,
        " qkv_type:",
        qkv_type,
        " k_rope_type:",
        k_rope_type,
    )

    comptime scale = Float32(0.125)

    # ------------------------------------------------------------------
    # Step 1: Allocate ragged Q, K, V on host (random init).
    # K and V stay in qkv_dtype (bf16); only K_rope is FP8.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 2: Build the row-offset tables.
    # ------------------------------------------------------------------
    var input_row_offsets_host = alloc[UInt32](batch_size + 1)
    var cache_row_offsets_host = alloc[UInt32](batch_size + 1)
    for i in range(batch_size):
        input_row_offsets_host[i] = UInt32(i * seq_len)
        cache_row_offsets_host[i] = UInt32(i * num_keys)
    input_row_offsets_host[batch_size] = UInt32(batch_size * seq_len)
    cache_row_offsets_host[batch_size] = UInt32(batch_size * num_keys)

    # ------------------------------------------------------------------
    # Step 3: Allocate paged K_rope blocks (FP8) and per-block scales
    # (FP32), plus the LUT and per-batch cache lengths. The red signal
    # for the partial-page bug is a `debug_assert` inside the kernel's
    # K_rope sub-tile loops; LUT padding values do not influence
    # whether it fires.
    # ------------------------------------------------------------------
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
    # For the SM100 paged path, ``cache_length(b)`` is interpreted as the
    # PRE-EXISTING cache length (start_pos in the kernel). For fresh
    # prefill (self-attention with no preceding tokens), this is 0; the
    # kernel then attends to keys ``[0, seq_len)`` which is where this
    # test places its K_rope data.
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(0)
    fill_uniform_lookup_table(
        lookup_table_host,
        batch_size,
        num_keys,
        page_size,
        max_pages_per_batch,
    )

    # ------------------------------------------------------------------
    # Step 4: Allocate device buffers and copy from host.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 5: Build TileTensors for the kernel call.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 6: Build the PagedKVCacheCollection for K_rope with
    # integrated blockwise scales (scale_dtype_=DType.float32,
    # quantization_granularity_=SCALE_BLOCK_SIZE). The scales live in a
    # 6D tensor with the same per-token layout as the FP8 blocks but
    # with the last axis = HEAD_DIM_GRAN block-scales.
    # ------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=CACHE_DEPTH, is_mla=True
    )
    var block_shape = IndexList[6](
        total_pages,
        1,  # kv_dim2 = 1 for is_mla
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
        UInt32(seq_len),  # max_seq_length
        UInt32(num_keys),  # max_cache_length
        # Pass the FP32 scales tensor.
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

    # ------------------------------------------------------------------
    # Step 7: Launch the kernel. The same `flare_mla_prefill` overload
    # the generic paged test uses; routing to blockscale happens
    # automatically because k_rope_type != qkv_type.
    # ------------------------------------------------------------------
    print("  Launching MLA prefill kernel (paged FP8 K_rope)...")

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
    print("  Kernel completed (no crash).")

    # ------------------------------------------------------------------
    # Step 8: Copy the kernel output back to host.
    # ------------------------------------------------------------------
    ctx.enqueue_copy(output_ptr, output_device_buf)
    ctx.synchronize()

    # ------------------------------------------------------------------
    # Step 9: Build the contiguous reference (K_ref, V_ref) in qkv_type
    # by extracting K_rope from the paged FP8 blocks AND dequantizing
    # with the per-token rope scale (block index 8). This mirrors the
    # blockscale kernel's CVT pipeline:
    # ``out_bf16 = fp8_val.cast[float32]() * scale``.
    # ------------------------------------------------------------------
    var ref_size = batch_size * num_keys * num_heads * depth
    var k_ref_host = alloc[Scalar[qkv_type]](ref_size)
    var v_ref_host = alloc[Scalar[qkv_type]](ref_size)
    var output_ref_host = alloc[Scalar[output_type]](
        batch_size * seq_len * num_heads * depth
    )

    # Dequantized rope window per batch (reused).
    var k_rope_one_batch = alloc[Scalar[qkv_type]](num_keys * ROPE_DEPTH)

    for b in range(batch_size):
        extract_dequantized_k_rope_for_batch[k_rope_type, qkv_type](
            blocks_host,
            scales_host,
            k_rope_one_batch,
            b,
            num_keys,
            page_size,
        )
        for s in range(num_keys):
            for h in range(num_heads):
                # First kv_depth elements: copy from K/V.
                var k_src_off = (
                    b * num_keys + s
                ) * num_heads * kv_depth + h * kv_depth
                var v_src_off = k_src_off
                var dst_base = (
                    b * num_keys + s
                ) * num_heads * depth + h * depth
                for d in range(kv_depth):
                    k_ref_host[dst_base + d] = k_ptr[k_src_off + d]
                    v_ref_host[dst_base + d] = v_ptr[v_src_off + d]
                # Last ROPE_DEPTH elements of each head: copy from the
                # dequantized rope (broadcast across heads). V_ref tail
                # is zero (V doesn't have a rope component).
                for d in range(ROPE_DEPTH):
                    k_ref_host[dst_base + kv_depth + d] = k_rope_one_batch[
                        s * ROPE_DEPTH + d
                    ]
                    v_ref_host[dst_base + kv_depth + d] = 0

    # ------------------------------------------------------------------
    # Step 10: Run the naive MHA reference on the contiguous K_ref/V_ref.
    # ------------------------------------------------------------------
    var k_ref_device_buf = ctx.enqueue_create_buffer[qkv_type](ref_size)
    var v_ref_device_buf = ctx.enqueue_create_buffer[qkv_type](ref_size)
    var output_ref_device_buf = ctx.enqueue_create_buffer[output_type](
        batch_size * seq_len * num_heads * depth
    )
    ctx.enqueue_copy(k_ref_device_buf, k_ref_host)
    ctx.enqueue_copy(v_ref_device_buf, v_ref_host)

    var q_device_rank4 = TileTensor(
        q_device_buf,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )
    var k_ref_device = TileTensor(
        k_ref_device_buf,
        row_major((batch_size, num_keys, Idx[num_heads], Idx[depth])),
    )
    var v_ref_device = TileTensor(
        v_ref_device_buf,
        row_major((batch_size, num_keys, Idx[num_heads], Idx[depth])),
    )
    var output_ref_device = TileTensor(
        output_ref_device_buf,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )

    var null_valid_length = LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
    ](
        None,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(Index(0)),
    )

    var k_ref_operand = LayoutTensorMHAOperand(
        k_ref_device.as_immut().as_unsafe_any_origin()
    )
    var v_ref_operand = LayoutTensorMHAOperand(
        v_ref_device.as_immut().as_unsafe_any_origin()
    )

    mha_gpu_naive[_is_cache_length_accurate=True](
        q_device_rank4.to_layout_tensor(),
        k_ref_operand,
        v_ref_operand,
        CausalMask(),
        output_ref_device.to_layout_tensor(),
        null_valid_length,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        1,  # group
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(output_ref_host, output_ref_device_buf)
    ctx.synchronize()

    # ------------------------------------------------------------------
    # Step 11: Compare the kernel output with the dequantized reference.
    # FP8 quantization noise + bf16 accumulation imprecision push the
    # tolerance higher than the bf16-only path. The existing contiguous
    # test (`test_mla_prefill_blockwise_fp8.mojo`) uses 2e-2 but with
    # large num_keys (120/1179) where noise averages out. Our small
    # partial-page configs (num_keys as small as 17) push noise much
    # higher; we adopt the same tolerance the paged FP8 decode test
    # uses (`test_mla_decode_blockwise_fp8.mojo:481–482`).
    # ------------------------------------------------------------------
    comptime atol: Float64 = 3e-1
    comptime rtol: Float64 = 5e-2
    var max_abs_err = Float64(0)
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(kv_depth):
                    var actual = output_ptr.load(
                        (b * seq_len + s) * num_heads * kv_depth
                        + h * kv_depth
                        + d
                    ).cast[.float64]()
                    var expect = output_ref_host.load(
                        ((b * seq_len + s) * num_heads + h) * depth + d
                    ).cast[.float64]()
                    var abs_err = abs(actual - expect)
                    if abs_err > max_abs_err:
                        max_abs_err = abs_err
                    if abs_err > atol:
                        print(
                            "mismatch at b=",
                            b,
                            " s=",
                            s,
                            " h=",
                            h,
                            " d=",
                            d,
                            " actual=",
                            actual,
                            " expect=",
                            expect,
                        )
                    assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

    print("  PASS, max_abs_err:", max_abs_err)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    output_ptr.free()
    k_ref_host.free()
    v_ref_host.free()
    output_ref_host.free()
    k_rope_one_batch.free()
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
    _ = k_ref_device_buf
    _ = v_ref_device_buf
    _ = output_ref_device_buf


def main() raises:
    with DeviceContext() as ctx:
        comptime if _is_sm10x_gpu(ctx.default_device_info):
            # Iterate over every ``num_keys`` in the shared list at
            # runtime; re-seed per ``num_keys`` for independent
            # reproducibility regardless of iteration order.
            for num_keys in num_keys_to_test():
                seed(0)
                # Single-batch baseline.
                run_test_paged_prefill_blockscale[
                    qkv_type=DType.bfloat16,
                    k_rope_type=DType.float8_e4m3fn,
                    output_type=DType.bfloat16,
                    depth=192,
                    num_heads=16,
                    kv_depth=128,
                    page_size=PAGE_SIZE,
                    batch_size=1,
                ](num_keys, num_keys, ctx)
                # Multi-batch — exercises cross-batch LUT layout: each
                # batch's pages occupy a distinct slice of the global
                # block array, so any incorrect LUT lookup past a
                # batch's valid page count would point to another
                # batch's data.
                run_test_paged_prefill_blockscale[
                    qkv_type=DType.bfloat16,
                    k_rope_type=DType.float8_e4m3fn,
                    output_type=DType.bfloat16,
                    depth=192,
                    num_heads=16,
                    kv_depth=128,
                    page_size=PAGE_SIZE,
                    batch_size=2,
                ](num_keys, num_keys, ctx)
