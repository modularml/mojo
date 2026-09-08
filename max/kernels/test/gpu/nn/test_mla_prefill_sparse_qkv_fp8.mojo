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

"""Numerical E2E test for the native-FP8 sparse MLA prefill kernel.

Mirrors `test_mla_prefill_sparse_kv_fp8.mojo` (SAME fp64 reference, SAME
verification gates and LOCKED tolerances) but drives
`mla_prefill_sparse_qkv_fp8`, where Q, K, V, and P are ALL FP8 e4m3 and QK^T /
P*V run natively via `tcgen05.mma.kind::f8f6f4` (no FP8->BF16 dequant).

Difference vs the KV-FP8 harness: Q is quantized to FP8 too (the native path
consumes FP8 Q, unlike the KV-FP8 path where Q stays BF16), and the fp64 oracle
consumes the SAME FP8-then-dequantized Q so Q quant noise is common-mode.  The
only non-common-mode approximation vs the oracle is P (softmax output) being FP8
instead of fp64 — the accuracy risk this test measures HONESTLY at the locked
tolerances.

Scope: `scale_block_size == 0` (unit-scale, the DSv3.2 production path) and
head <= 64 (cta_group=1).  head=128 (2-CTA f8f6f4) and tensorwise/blockwise
scale-fold are follow-ups.
"""

from std.math import ceildiv, exp2, sqrt
from std.math.constants import log2e
from std.memory import UnsafePointer, alloc
from std.random import randn, seed
from std.sys import has_nvidia_gpu_accelerator, size_of

from max.gpu import *
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
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
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils import (
    MLASparseConfig,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_qkv_fp8 import (
    mla_prefill_sparse_qkv_fp8,
)
from std.testing import assert_equal
from std.utils.index import Index, IndexList
from std.utils.numerics import isnan, min_or_neg_inf


# ===-----------------------------------------------------------------------===#
# Test constants (DSv3.2 absorbed / latent dims).
# ===-----------------------------------------------------------------------===#

comptime KV_LORA_RANK = 512
comptime QK_ROPE_HEAD_DIM = 64
comptime QK_DEPTH = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
comptime V_DEPTH = KV_LORA_RANK  # 512
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1

comptime SOFTMAX_SCALE_BASE_DIM = 192

comptime FP8_E4M3_MAX = Float32(448.0)


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


def _coprime_multiplier(n: Int) -> Int:
    """Find a multiplier coprime to n for deterministic token selection."""
    if n <= 1:
        return 1
    if _gcd(3, n) == 1:
        return 3
    if _gcd(5, n) == 1:
        return 5
    if _gcd(7, n) == 1:
        return 7
    if _gcd(11, n) == 1:
        return 11
    return 13


# ===-----------------------------------------------------------------------===#
# Host-side reference (identical algebra to test_mla_prefill_sparse_kv_fp8).
# ===-----------------------------------------------------------------------===#


def host_reference[
    q_type: DType,
](
    q_ptr: UnsafePointer[Scalar[q_type], _],
    kv_sparse_ptr: UnsafePointer[Scalar[q_type], _],
    output_ptr: UnsafePointer[mut=True, Scalar[q_type], _],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    topk: Int,
    qk_depth: Int,
    v_depth: Int,
    scale: Float32,
    valid_topk: Int = -1,
    sink_values: List[Float32] = [],
):
    var scale_log2e = Float64(scale) * Float64(log2e)
    var n_valid = topk if valid_topk == -1 else valid_topk
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                var bs = b * seq_len + s
                var q_base = bs * num_heads * qk_depth + h * qk_depth

                var mi = Float64(min_or_neg_inf[.float32]())
                var s_buf = alloc[Float64](n_valid)

                for k in range(n_valid):
                    var kv_base = (bs * topk + k) * qk_depth
                    var dot = Float64(0)
                    for d in range(qk_depth):
                        dot += (
                            q_ptr[q_base + d].cast[.float64]()
                            * kv_sparse_ptr[kv_base + d].cast[.float64]()
                        )
                    s_buf[k] = dot * scale_log2e
                    if s_buf[k] > mi:
                        mi = s_buf[k]

                var li = Float64(0)
                for k in range(n_valid):
                    s_buf[k] = exp2(s_buf[k] - mi)
                    li += s_buf[k]
                if len(sink_values) > 0:
                    var sink_arg = Float64(sink_values[h]) * Float64(log2e) - mi
                    if sink_arg > Float64(-1000.0):
                        li += exp2(sink_arg)
                for k in range(n_valid):
                    s_buf[k] = s_buf[k] / li

                var o_base = bs * num_heads * v_depth + h * v_depth
                for d in range(v_depth):
                    var acc = Float64(0)
                    for k in range(n_valid):
                        var kv_base = (bs * topk + k) * qk_depth
                        acc += (
                            s_buf[k]
                            * kv_sparse_ptr[kv_base + d].cast[.float64]()
                        )
                    output_ptr[o_base + d] = acc.cast[q_type]()

                s_buf.free()


# ===-----------------------------------------------------------------------===#
# Native-FP8 test driver.  scale_block_size is fixed to 0 (unit-scale).
# ===-----------------------------------------------------------------------===#


def run_test_prefill_sparse_qkv_fp8[
    num_heads: Int,
    topk: Int,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the latent
    # alone. The kernel's shared memory stays laid out for 576 either way.
    row_depth: Int = QK_DEPTH,
](
    name: StringLiteral,
    batch_size: Int,
    seq_len: Int,
    num_kv_tokens: Int,
    ctx: DeviceContext,
    *,
    valid_topk: Int = topk,
    atol: Float64 = 0.02,
    sink_values: List[Float32] = [],
    num_layers: Int = NUM_LAYERS,
    layer_idx: Int = 0,
    topk_lengths_override: Int = -1,
) raises:
    comptime scale_block_size = 0
    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " seq_len:",
        seq_len,
        " num_heads:",
        num_heads,
        " num_kv_tokens:",
        num_kv_tokens,
        " topk:",
        topk,
    )

    var scale = Float32(1.0) / sqrt(Float32(SOFTMAX_SCALE_BASE_DIM))
    comptime group = num_heads
    var total_q_tokens = batch_size * seq_len

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=row_depth, is_mla=True
    )
    comptime kv_dim2 = 1

    var total_pages = batch_size * ceildiv(num_kv_tokens, PAGE_SIZE)
    var max_pages_per_batch = ceildiv(num_kv_tokens, PAGE_SIZE)

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        num_layers,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * num_layers
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var page_stride_elems = (
        kv_dim2
        * num_layers
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var layer_stride_elems = (
        PAGE_SIZE * kv_params.num_heads * kv_params.head_size
    )

    # Page lookup table (coprime shuffle).
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = alloc[UInt32](lut_size)
    var page_offset = 0
    for bi in range(batch_size):
        var np = ceildiv(num_kv_tokens, PAGE_SIZE)
        var mult = _coprime_multiplier(np)
        for p in range(np):
            var shuffled_p = (p * mult + 1) % np
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np

    # Random BF16 KV, quantized to FP8 at unit scale (no-scale mode), then
    # dequantized for the fp64 oracle so quant noise is common-mode.
    var kv_total = batch_size * num_kv_tokens * row_depth
    var kv_bf16_host = alloc[BFloat16](kv_total)
    randn[.bfloat16](kv_bf16_host, kv_total, mean=0.0, standard_deviation=0.5)

    comptime scales_per_token = 1
    var total_phys_rows = total_pages * num_layers * PAGE_SIZE

    var blocks_fp8_host = alloc[Float8_e4m3fn](block_elems)
    if num_layers > 1:
        for i in range(block_elems):
            blocks_fp8_host[i] = Float8_e4m3fn(
                Float32(Int(i % 17) - 8) * Float32(0.5)
            )
    else:
        for i in range(block_elems):
            blocks_fp8_host[i] = Float8_e4m3fn(0)

    var kv_dequant_host = alloc[BFloat16](kv_total)

    for bi in range(batch_size):
        for t in range(num_kv_tokens):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var src_base = (bi * num_kv_tokens + t) * row_depth
            var base = (
                block_id * page_stride_elems
                + layer_idx * layer_stride_elems
                + tok_in_page * row_depth
            )
            # Unit scale: quantize with s == 1.0 (no dequant scale).
            for d in range(row_depth):
                var q_val = kv_bf16_host[src_base + d].cast[.float32]()
                var q_val_clamped = min(max(q_val, -FP8_E4M3_MAX), FP8_E4M3_MAX)
                var fp8_val = q_val_clamped.cast[.float8_e4m3fn]()
                blocks_fp8_host[base + d] = fp8_val
                kv_dequant_host[src_base + d] = fp8_val.cast[.bfloat16]()

    # Q: BF16 randn -> FP8 (unit scale) -> dequantized BF16 for the oracle.
    var q_elems = total_q_tokens * num_heads * row_depth
    var q_bf16_host = alloc[BFloat16](q_elems)
    randn[.bfloat16](q_bf16_host, q_elems, mean=0.0, standard_deviation=0.5)
    var q_fp8_host = alloc[Float8_e4m3fn](q_elems)
    var q_dequant_host = alloc[BFloat16](q_elems)
    for i in range(q_elems):
        var v = q_bf16_host[i].cast[.float32]()
        var v_clamped = min(max(v, -FP8_E4M3_MAX), FP8_E4M3_MAX)
        var fp8_v = v_clamped.cast[.float8_e4m3fn]()
        q_fp8_host[i] = fp8_v
        q_dequant_host[i] = fp8_v.cast[.bfloat16]()

    # Per-query token selection (coprime rotation).
    var selected_tokens = alloc[Int](total_q_tokens * topk)
    var sel_mult = _coprime_multiplier(num_kv_tokens)
    for bi in range(batch_size):
        for s in range(seq_len):
            var bs = bi * seq_len + s
            var rotation = s % num_kv_tokens
            for i in range(topk):
                selected_tokens[bs * topk + i] = (
                    (rotation + i) * sel_mult + 1
                ) % num_kv_tokens

    # Sparse KV ref from DEQUANTIZED values.
    var kv_sparse_size = total_q_tokens * topk * row_depth
    var kv_sparse = alloc[BFloat16](kv_sparse_size)
    for bi in range(batch_size):
        for s in range(seq_len):
            var bs = bi * seq_len + s
            for i in range(topk):
                var t = selected_tokens[bs * topk + i]
                var src_base = (bi * num_kv_tokens + t) * row_depth
                var dst_base = (bs * topk + i) * row_depth
                for d in range(row_depth):
                    kv_sparse[dst_base + d] = kv_dequant_host[src_base + d]

    # Host reference (consumes FP8-dequantized Q and KV -> common-mode noise).
    var out_elems = total_q_tokens * num_heads * V_DEPTH
    var ref_host = alloc[BFloat16](out_elems)
    host_reference[.bfloat16](
        q_dequant_host,
        kv_sparse,
        ref_host,
        batch_size,
        seq_len,
        num_heads,
        topk,
        row_depth,
        V_DEPTH,
        scale,
        valid_topk=valid_topk,
        sink_values=sink_values,
    )

    # Device buffers.
    var blocks_device = ctx.enqueue_create_buffer[.float8_e4m3fn](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_fp8_host)

    var cache_lengths_host = alloc[UInt32](batch_size)
    for bi in range(batch_size):
        cache_lengths_host[bi] = UInt32(num_kv_tokens)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_elems)
    ctx.enqueue_copy(q_device, q_fp8_host)

    var out_device = ctx.enqueue_create_buffer[.bfloat16](out_elems)

    ctx.synchronize()

    # Per-query gather4 indices (sentinel 0xFFFFFFFF for masked positions).
    var total_indices = total_q_tokens * topk
    var h_indices = alloc[UInt32](total_indices)
    for bi in range(batch_size):
        for s in range(seq_len):
            var bs = bi * seq_len + s
            for i in range(topk):
                if i < valid_topk:
                    var t = selected_tokens[bs * topk + i]
                    var page_idx = t // PAGE_SIZE
                    var tok_in_page = t % PAGE_SIZE
                    var block_id = Int(
                        lookup_table_host[bi * max_pages_per_batch + page_idx]
                    )
                    h_indices[bs * topk + i] = UInt32(
                        block_id * PAGE_SIZE + tok_in_page
                    )
                else:
                    h_indices[bs * topk + i] = UInt32(0xFFFFFFFF)

    var indices_device = ctx.enqueue_create_buffer[.uint32](total_indices)
    ctx.enqueue_copy(indices_device, h_indices)

    var reported_len = (
        valid_topk if topk_lengths_override < 0 else topk_lengths_override
    )
    var h_topk_lengths = alloc[UInt32](total_q_tokens)
    for i in range(total_q_tokens):
        h_topk_lengths[i] = UInt32(reported_len)
    var topk_lengths_device = ctx.enqueue_create_buffer[.uint32](total_q_tokens)
    ctx.enqueue_copy(topk_lengths_device, h_topk_lengths)

    ctx.synchronize()

    # FP8 PagedKVCacheCollection.
    var blocks_lt = LayoutTensor[.float8_e4m3fn, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
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
        DType.float8_e4m3fn, kv_params, PAGE_SIZE
    ](
        LayoutTensor[.float8_e4m3fn, Layout.row_major[6]()](
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
        UInt32(num_kv_tokens),
    )

    var kv_cache = kv_collection.get_key_cache(layer_idx)

    var q_tt = TileTensor(
        q_device,
        row_major((total_q_tokens, Idx[num_heads], Idx[row_depth])),
    )
    var out_tt = TileTensor(
        out_device,
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )
    var indices_tt = TileTensor(
        indices_device,
        row_major(total_indices),
    )
    var topk_lengths_tt = TileTensor(
        topk_lengths_device,
        row_major(total_q_tokens),
    )

    print("  Launching mla_prefill_sparse_qkv_fp8...")

    # head<=64 runs the single-CTA shared-KV tile at num_mbars_=4 (deepened from
    # 3: the kernel is KV-gather-latency bound -- NCU showed long_scoreboard the
    # dominant stall with DRAM at <1% -- so a deeper software-pipeline hides more
    # gather latency; measured 1.7-2.9% faster than depth-3 at production shapes,
    # same-session A/B, 24/24 correctness unchanged).  b_topk stays 64 here: the
    # native PV mn-major SW64 read only proves correct at a single SW64 atom
    # (see `_mma`'s NUM_PV_KHALVES comment in mla_prefill_sparse_qkv_fp8.mojo).
    #
    # head=128 runs the 2-CTA (cta_group=2) f8f6f4 tile at b_topk=128 (halves
    # the k-block count and cross-CTA handshakes vs b_topk=64) via the K-axis
    # P@V split (NUM_PV_KHALVES=2 BK=64 sub-MMAs per atom, mirroring the
    # proven SS_P0/SS_P1 key-half split in the dequant `SVMMAType`).
    # num_mbars=2 here (cut from 4) keeps num_mbars*b_topk constant (256, same
    # KV-staging footprint as the depth-4/b_topk=64 baseline), so the SMEM
    # carveout is unaffected by the b_topk bump.
    comptime cta_group = 2 if num_heads == 128 else 1
    comptime b_topk = 128 if cta_group == 2 else 64
    comptime num_mbars = 2 if cta_group == 2 else 4
    comptime config = MLASparseConfig[
        DType.bfloat16,
        b_topk_=b_topk,
        num_mbars_=num_mbars,
        cta_group_=cta_group,
    ](
        num_q_heads=num_heads,
        num_kv_heads=1,
        qk_depth=QK_DEPTH,
        input_qk_depth=row_depth,
        v_depth=V_DEPTH,
        indices_stride=topk,
        group=num_heads,
    )

    var attn_sink_ptr = Optional[UnsafePointer[Float32, ImmutAnyOrigin]](None)
    var sink_len = len(sink_values) if len(sink_values) > 0 else 1
    var sink_device = ctx.enqueue_create_buffer[.float32](sink_len)
    if len(sink_values) > 0:
        var sink_host = alloc[Float32](len(sink_values))
        for i in range(len(sink_values)):
            sink_host[i] = sink_values[i]
        ctx.enqueue_copy(sink_device, sink_host)
        ctx.synchronize()
        sink_host.free()
        attn_sink_ptr = Optional[UnsafePointer[Float32, ImmutAnyOrigin]](
            sink_device.unsafe_ptr().bitcast[Float32]().as_unsafe_any_origin()
        )

    mla_prefill_sparse_qkv_fp8[
        config=config,
        group=group,
        q_depth=row_depth,
        scale_block_size=scale_block_size,
    ](
        out_tt,
        q_tt,
        kv_cache,
        indices_tt,
        topk_lengths_tt,
        attn_sink_ptr,
        scale,
        Int32(topk),
        ctx,
    )

    ctx.synchronize()

    var out_host = alloc[BFloat16](out_elems)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    # Verification gates: IDENTICAL to test_mla_prefill_sparse_kv_fp8.mojo.
    # DO NOT loosen. Native FP8 adds Q and P quantization; some cases may
    # exceed these — report honestly rather than widen.
    var max_err = Float64(0)
    var max_actual = Float64(0)
    var num_nonzero = 0
    var total_checked = 0
    var nan_actual = 0
    var inf_actual = 0
    var nan_ref = 0
    var inf_ref = 0
    var sum_abs_err = Float64(0)
    var dot_ar = Float64(0)
    var norm_a = Float64(0)
    var norm_r = Float64(0)
    var n_finite = 0
    var n_err_gt_1em2 = 0
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(V_DEPTH):
                    var idx = (
                        b * seq_len * num_heads * V_DEPTH
                        + s * num_heads * V_DEPTH
                        + h * V_DEPTH
                        + d
                    )
                    var ref_val = ref_host[idx].cast[.float64]()
                    var actual_val = out_host[idx].cast[.float64]()
                    var err = abs(actual_val - ref_val)
                    if err > max_err:
                        max_err = err
                    if abs(actual_val) > max_actual:
                        max_actual = abs(actual_val)
                    if abs(actual_val) > 1e-6:
                        num_nonzero += 1
                    total_checked += 1
                    if actual_val != actual_val:
                        nan_actual += 1
                    elif abs(actual_val) > 1.0e300:
                        inf_actual += 1
                    elif ref_val == ref_val and abs(ref_val) <= 1.0e300:
                        sum_abs_err += err
                        if err > 1.0e-2:
                            n_err_gt_1em2 += 1
                        dot_ar += actual_val * ref_val
                        norm_a += actual_val * actual_val
                        norm_r += ref_val * ref_val
                        n_finite += 1
                    if ref_val != ref_val:
                        nan_ref += 1
                    elif abs(ref_val) > 1.0e300:
                        inf_ref += 1

    print(
        "  DIAG: max_err=",
        max_err,
        " max_abs_actual=",
        max_actual,
        " num_nonzero=",
        num_nonzero,
        "/",
        total_checked,
    )

    if nan_actual != 0 or inf_actual != 0 or nan_ref != 0 or inf_ref != 0:
        raise Error(
            "non-finite values detected: nan_actual="
            + String(nan_actual)
            + " inf_actual="
            + String(inf_actual)
            + " nan_ref="
            + String(nan_ref)
            + " inf_ref="
            + String(inf_ref)
        )

    comptime COS_MIN: Float64 = 0.99
    comptime MEAN_ERR_MAX: Float64 = 0.01
    var mean_abs_err = sum_abs_err / Float64(max(n_finite, 1))
    if mean_abs_err > MEAN_ERR_MAX:
        raise Error(
            "mean_abs_err exceeded tolerance: "
            + String(mean_abs_err)
            + " > "
            + String(MEAN_ERR_MAX)
        )
    var cos_denom = sqrt(norm_a) * sqrt(norm_r)
    var cosine: Float64
    if norm_a < 1e-12 and norm_r < 1e-12:
        cosine = 1.0
    elif cos_denom < 1e-12:
        cosine = 0.0
    else:
        cosine = dot_ar / cos_denom
    if cosine < COS_MIN:
        raise Error(
            "cosine similarity below tolerance: "
            + String(cosine)
            + " < "
            + String(COS_MIN)
        )

    var tail_frac = Float64(n_err_gt_1em2) / Float64(max(n_finite, 1))
    if tail_frac >= 0.0015:
        raise Error(
            "error-tail exceeded bound: "
            + String(n_err_gt_1em2)
            + " elements (|err| > 1e-2) = "
            + String(tail_frac * 100)
            + "% >= 0.15%"
        )

    if max_err > atol:
        raise Error(
            "max_err exceeded tolerance: "
            + String(max_err)
            + " > atol "
            + String(atol)
        )
    print(
        "  PASSED: max_err=",
        max_err,
        " mean_abs_err=",
        mean_abs_err,
        " cosine=",
        cosine,
        " n_err_gt_1e-2=",
        n_err_gt_1em2,
        " checked=",
        total_checked,
        " elements",
    )

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = indices_device
    _ = topk_lengths_device
    _ = sink_device

    blocks_fp8_host.free()
    kv_bf16_host.free()
    kv_dequant_host.free()
    lookup_table_host.free()
    cache_lengths_host.free()
    q_bf16_host.free()
    q_fp8_host.free()
    q_dequant_host.free()
    kv_sparse.free()
    selected_tokens.free()
    ref_host.free()
    out_host.free()
    h_indices.free()
    h_topk_lengths.free()


# ===-----------------------------------------------------------------------===#
# NoPE native-512 bit-identical differential (no fp64 host reference).
# ===-----------------------------------------------------------------------===#


def run_nope_native_512_prefill_fp8[
    num_heads: Int,
    topk: Int,
](
    ctx: DeviceContext,
    batch_size: Int,
    seq_len: Int,
    num_kv_tokens: Int,
    *,
    valid_topk: Int = topk,
    topk_lengths_override: Int = -1,
    name: StringLiteral = "",
) raises:
    """The native-FP8 sparse MLA prefill kernel takes a 512-wide NoPE row and
    answers identically.

    Same property as the decode oracle
    (`test_mla_decode_sparse_qkv_fp8.run_nope_native_512_fp8`) and as
    `run_test_prefill_sparse_qkv_fp8[row_depth=512]` above: a NoPE model
    stores the 512-wide latent alone, the kernel's shared memory stays laid
    out for the rotary-carrying 576 either way, and the extra 64 columns the
    narrower TMA no longer covers are zero-filled once at kernel entry. This
    runs one problem twice -- once with both operands zero-padded to 576,
    once at native 512 -- and requires the two GPU outputs to be
    **bit-identical**. Padding adds only exact zeros to a float32
    accumulation, so anything short of bit-identical means the narrow path
    read or wrote something the wide path did not.

    Unlike `run_test_prefill_sparse_qkv_fp8`, this builds no fp64 host
    reference: the zero-padded 576 run IS the reference. Both arms are GPU
    runs, so the cost is two kernel launches instead of an
    O(total_q_tokens * topk * row_depth) CPU loop, which is what makes the
    production-scale shapes below affordable.
    """
    print(
        "NoPE native 512 prefill (QKV FP8): ",
        name,
        " batch_size=",
        batch_size,
        " seq_len=",
        seq_len,
        " num_kv_tokens=",
        num_kv_tokens,
        " num_heads=",
        num_heads,
        " topk=",
        topk,
        " valid_topk=",
        valid_topk,
    )
    seed(0x5EED19)

    var scale = Float32(1.0) / sqrt(Float32(SOFTMAX_SCALE_BASE_DIM))
    comptime group = num_heads
    var total_q_tokens = batch_size * seq_len

    comptime kv_params_576 = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=QK_DEPTH, is_mla=True
    )
    comptime kv_params_512 = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=V_DEPTH, is_mla=True
    )
    comptime kv_dim2 = 1

    var total_pages = batch_size * ceildiv(num_kv_tokens, PAGE_SIZE)
    var max_pages_per_batch = ceildiv(num_kv_tokens, PAGE_SIZE)

    var block_shape_576 = IndexList[6](
        total_pages, kv_dim2, NUM_LAYERS, PAGE_SIZE, KV_NUM_HEADS, QK_DEPTH
    )
    var block_shape_512 = IndexList[6](
        total_pages, kv_dim2, NUM_LAYERS, PAGE_SIZE, KV_NUM_HEADS, V_DEPTH
    )
    var block_elems_576 = (
        total_pages * kv_dim2 * NUM_LAYERS * PAGE_SIZE * KV_NUM_HEADS * QK_DEPTH
    )
    var block_elems_512 = (
        total_pages * kv_dim2 * NUM_LAYERS * PAGE_SIZE * KV_NUM_HEADS * V_DEPTH
    )
    var page_stride_elems_576 = (
        kv_dim2 * NUM_LAYERS * PAGE_SIZE * KV_NUM_HEADS * QK_DEPTH
    )
    var page_stride_elems_512 = (
        kv_dim2 * NUM_LAYERS * PAGE_SIZE * KV_NUM_HEADS * V_DEPTH
    )

    # Page lookup table (coprime shuffle) -- shared by both arms, it only
    # depends on num_kv_tokens/batch_size.
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = alloc[UInt32](lut_size)
    var page_offset = 0
    for bi in range(batch_size):
        var np = ceildiv(num_kv_tokens, PAGE_SIZE)
        var mult = _coprime_multiplier(np)
        for p in range(np):
            var shuffled_p = (p * mult + 1) % np
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np

    var cache_lengths_host = alloc[UInt32](batch_size)
    for bi in range(batch_size):
        cache_lengths_host[bi] = UInt32(num_kv_tokens)

    # One latent KV, quantized once to FP8, laid out at both widths so the
    # two arms see exactly the same FP8 bytes over [0, 512). The 576 copy's
    # extra 64 columns per row are zero; the 512 copy has no tail at all.
    var kv_total = batch_size * num_kv_tokens * V_DEPTH
    var kv_bf16_host = alloc[BFloat16](kv_total)
    randn[.bfloat16](kv_bf16_host, kv_total, mean=0.0, standard_deviation=0.5)

    var blocks_576_host = alloc[Float8_e4m3fn](block_elems_576)
    var blocks_512_host = alloc[Float8_e4m3fn](block_elems_512)
    for i in range(block_elems_576):
        blocks_576_host[i] = Float8_e4m3fn(0)
    for i in range(block_elems_512):
        blocks_512_host[i] = Float8_e4m3fn(0)

    for bi in range(batch_size):
        for t in range(num_kv_tokens):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var src_base = (bi * num_kv_tokens + t) * V_DEPTH
            var base_576 = (
                block_id * page_stride_elems_576 + tok_in_page * QK_DEPTH
            )
            var base_512 = (
                block_id * page_stride_elems_512 + tok_in_page * V_DEPTH
            )
            for d in range(V_DEPTH):
                var q_val = kv_bf16_host[src_base + d].cast[.float32]()
                var q_val_clamped = min(max(q_val, -FP8_E4M3_MAX), FP8_E4M3_MAX)
                var fp8_val = q_val_clamped.cast[.float8_e4m3fn]()
                blocks_576_host[base_576 + d] = fp8_val
                blocks_512_host[base_512 + d] = fp8_val

    # Q at both widths, same FP8 bytes over [0, 512); 576's tail is zero.
    var q_elems_512 = total_q_tokens * num_heads * V_DEPTH
    var q_elems_576 = total_q_tokens * num_heads * QK_DEPTH
    var q_bf16_host = alloc[BFloat16](q_elems_512)
    randn[.bfloat16](q_bf16_host, q_elems_512, mean=0.0, standard_deviation=0.5)
    var q_512_host = alloc[Float8_e4m3fn](q_elems_512)
    var q_576_host = alloc[Float8_e4m3fn](q_elems_576)
    for row in range(total_q_tokens * num_heads):
        for d in range(V_DEPTH):
            var v = q_bf16_host[row * V_DEPTH + d].cast[.float32]()
            var v_clamped = min(max(v, -FP8_E4M3_MAX), FP8_E4M3_MAX)
            var fp8_v = v_clamped.cast[.float8_e4m3fn]()
            q_512_host[row * V_DEPTH + d] = fp8_v
            q_576_host[row * QK_DEPTH + d] = fp8_v
        for d in range(V_DEPTH, QK_DEPTH):
            q_576_host[row * QK_DEPTH + d] = Float8_e4m3fn(0)

    # Per-query token selection (coprime rotation) -- identical for both arms,
    # so the same physical rows get selected regardless of row_depth.
    var selected_tokens = alloc[Int](total_q_tokens * topk)
    var sel_mult = _coprime_multiplier(num_kv_tokens)
    for bi in range(batch_size):
        for s in range(seq_len):
            var bs = bi * seq_len + s
            var rotation = s % num_kv_tokens
            for i in range(topk):
                selected_tokens[bs * topk + i] = (
                    (rotation + i) * sel_mult + 1
                ) % num_kv_tokens

    # Per-query gather4 indices (sentinel 0xFFFFFFFF for masked positions) --
    # shared by both arms.
    var total_indices = total_q_tokens * topk
    var h_indices = alloc[UInt32](total_indices)
    for bi in range(batch_size):
        for s in range(seq_len):
            var bs = bi * seq_len + s
            for i in range(topk):
                if i < valid_topk:
                    var t = selected_tokens[bs * topk + i]
                    var page_idx = t // PAGE_SIZE
                    var tok_in_page = t % PAGE_SIZE
                    var block_id = Int(
                        lookup_table_host[bi * max_pages_per_batch + page_idx]
                    )
                    h_indices[bs * topk + i] = UInt32(
                        block_id * PAGE_SIZE + tok_in_page
                    )
                else:
                    h_indices[bs * topk + i] = UInt32(0xFFFFFFFF)

    var reported_len = (
        valid_topk if topk_lengths_override < 0 else topk_lengths_override
    )
    var h_topk_lengths = alloc[UInt32](total_q_tokens)
    for i in range(total_q_tokens):
        h_topk_lengths[i] = UInt32(reported_len)

    # Device buffers.
    var blocks_576_device = ctx.enqueue_create_buffer[.float8_e4m3fn](
        block_elems_576
    )
    ctx.enqueue_copy(blocks_576_device, blocks_576_host)
    var blocks_512_device = ctx.enqueue_create_buffer[.float8_e4m3fn](
        block_elems_512
    )
    ctx.enqueue_copy(blocks_512_device, blocks_512_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_576_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_elems_576)
    ctx.enqueue_copy(q_576_device, q_576_host)
    var q_512_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_elems_512)
    ctx.enqueue_copy(q_512_device, q_512_host)

    var indices_device = ctx.enqueue_create_buffer[.uint32](total_indices)
    ctx.enqueue_copy(indices_device, h_indices)
    var topk_lengths_device = ctx.enqueue_create_buffer[.uint32](total_q_tokens)
    ctx.enqueue_copy(topk_lengths_device, h_topk_lengths)

    var out_elems = total_q_tokens * num_heads * V_DEPTH
    var out_576_device = ctx.enqueue_create_buffer[.bfloat16](out_elems)
    var out_512_device = ctx.enqueue_create_buffer[.bfloat16](out_elems)

    ctx.synchronize()

    # Paged KV cache collections, one per row width.
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    comptime lt_layout_2d = Layout.row_major[2]()
    comptime blk_layout = Layout.row_major[6]()

    var cache_lengths_lt = LayoutTensor[.uint32, cl_layout](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )
    var lookup_table_lt = LayoutTensor[.uint32, lt_layout_2d](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, max_pages_per_batch)
        ),
    )

    var blocks_576_lt = LayoutTensor[.float8_e4m3fn, blk_layout](
        blocks_576_device.unsafe_ptr(),
        RuntimeLayout[blk_layout].row_major(block_shape_576),
    )
    var blocks_512_lt = LayoutTensor[.float8_e4m3fn, blk_layout](
        blocks_512_device.unsafe_ptr(),
        RuntimeLayout[blk_layout].row_major(block_shape_512),
    )

    var kv_collection_576 = PagedKVCacheCollection[
        DType.float8_e4m3fn, kv_params_576, PAGE_SIZE
    ](
        LayoutTensor[.float8_e4m3fn, blk_layout](
            blocks_576_lt.ptr,
            RuntimeLayout[blk_layout](
                blocks_576_lt.runtime_layout.shape.value,
                blocks_576_lt.runtime_layout.stride.value,
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
        UInt32(num_kv_tokens),
    )
    var kv_collection_512 = PagedKVCacheCollection[
        DType.float8_e4m3fn, kv_params_512, PAGE_SIZE
    ](
        LayoutTensor[.float8_e4m3fn, blk_layout](
            blocks_512_lt.ptr,
            RuntimeLayout[blk_layout](
                blocks_512_lt.runtime_layout.shape.value,
                blocks_512_lt.runtime_layout.stride.value,
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
        UInt32(num_kv_tokens),
    )

    var kv_cache_576 = kv_collection_576.get_key_cache(0)
    var kv_cache_512 = kv_collection_512.get_key_cache(0)

    var q_576_tt = TileTensor(
        q_576_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[QK_DEPTH])),
    )
    var q_512_tt = TileTensor(
        q_512_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )
    var out_576_tt = TileTensor(
        out_576_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )
    var out_512_tt = TileTensor(
        out_512_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )
    var indices_tt = TileTensor(
        indices_device.unsafe_ptr(), row_major(total_indices)
    )
    var topk_lengths_tt = TileTensor(
        topk_lengths_device.unsafe_ptr(), row_major(total_q_tokens)
    )

    comptime cta_group = 2 if num_heads == 128 else 1
    comptime b_topk = 128 if cta_group == 2 else 64
    comptime num_mbars = 2 if cta_group == 2 else 4
    comptime config_576 = MLASparseConfig[
        DType.bfloat16,
        b_topk_=b_topk,
        num_mbars_=num_mbars,
        cta_group_=cta_group,
    ](
        num_q_heads=num_heads,
        num_kv_heads=1,
        qk_depth=QK_DEPTH,
        input_qk_depth=QK_DEPTH,
        v_depth=V_DEPTH,
        indices_stride=topk,
        group=num_heads,
    )
    comptime config_512 = MLASparseConfig[
        DType.bfloat16,
        b_topk_=b_topk,
        num_mbars_=num_mbars,
        cta_group_=cta_group,
    ](
        num_q_heads=num_heads,
        num_kv_heads=1,
        qk_depth=QK_DEPTH,
        input_qk_depth=V_DEPTH,
        v_depth=V_DEPTH,
        indices_stride=topk,
        group=num_heads,
    )

    mla_prefill_sparse_qkv_fp8[
        config=config_576,
        group=group,
        q_depth=QK_DEPTH,
        scale_block_size=0,
    ](
        out_576_tt,
        q_576_tt,
        kv_cache_576,
        indices_tt,
        topk_lengths_tt,
        Optional[ImmPointer[Float32, ImmutAnyOrigin]](None),
        scale,
        Int32(topk),
        ctx,
    )
    ctx.synchronize()

    mla_prefill_sparse_qkv_fp8[
        config=config_512,
        group=group,
        q_depth=V_DEPTH,
        scale_block_size=0,
    ](
        out_512_tt,
        q_512_tt,
        kv_cache_512,
        indices_tt,
        topk_lengths_tt,
        Optional[ImmPointer[Float32, ImmutAnyOrigin]](None),
        scale,
        Int32(topk),
        ctx,
    )
    ctx.synchronize()

    var h576 = alloc[BFloat16](out_elems)
    var h512 = alloc[BFloat16](out_elems)
    ctx.enqueue_copy(h576, out_576_device)
    ctx.enqueue_copy(h512, out_512_device)
    ctx.synchronize()

    var mismatches = 0
    for i in range(out_elems):
        if isnan(h512[i].cast[.float64]()):
            raise Error("NaN in native-512 prefill QKV-FP8 output")
        if h576[i] != h512[i]:
            mismatches += 1
            if mismatches <= 5:
                print(
                    "  mismatch idx=",
                    i,
                    " padded576=",
                    h576[i],
                    " native512=",
                    h512[i],
                )
    if mismatches > 0:
        print("  FAILED: ", mismatches, "of", out_elems, "outputs differ")
        raise Error("native-512 prefill QKV-FP8 does not match padded 576")
    assert_equal(mismatches, 0)
    print("  PASSED: native 512 is bit-identical to zero-padded 576")

    _ = blocks_576_device
    _ = blocks_512_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_576_device
    _ = q_512_device
    _ = indices_device
    _ = topk_lengths_device
    _ = out_576_device
    _ = out_512_device

    blocks_576_host.free()
    blocks_512_host.free()
    kv_bf16_host.free()
    lookup_table_host.free()
    cache_lengths_host.free()
    q_bf16_host.free()
    q_512_host.free()
    q_576_host.free()
    selected_tokens.free()
    h_indices.free()
    h_topk_lengths.free()
    h576.free()
    h512.free()


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            # ---------------- NoPE native 512 ----------------
            # A NoPE model stores the latent alone. The kernel takes that row
            # directly -- narrower TMA, shared memory still laid out for 576 and
            # its trailing columns zero-filled -- and must still match the host
            # reference computed over the same 512-wide contraction.
            run_test_prefill_sparse_qkv_fp8[64, 64, row_depth=512](
                "b1_s32_h64_kv256_topk64_qkv_fp8_nope512", 1, 32, 256, ctx
            )
            run_test_prefill_sparse_qkv_fp8[64, 1024, row_depth=512](
                "b1_s8_h64_kv1024_topk1024_qkv_fp8_nope512", 1, 8, 1024, ctx
            )
            # GLM TP8 head count at the native row.
            run_test_prefill_sparse_qkv_fp8[8, 64, row_depth=512](
                "b1_s1_h8_kv256_topk64_qkv_fp8_nope512", 1, 1, 256, ctx
            )

            # ---------------- head=64 (cta_group=1) ----------------
            # Exact 1 k-block.
            run_test_prefill_sparse_qkv_fp8[64, 64](
                "b1_s32_h64_kv256_topk64_qkv_fp8_noscale", 1, 32, 256, ctx
            )
            # Exact 2 k-blocks (cross-block online-softmax state).
            run_test_prefill_sparse_qkv_fp8[64, 128](
                "b1_s32_h64_kv512_topk128_qkv_fp8_noscale", 1, 32, 512, ctx
            )
            # Deep 16 k-blocks.
            run_test_prefill_sparse_qkv_fp8[64, 1024](
                "b1_s8_h64_kv1024_topk1024_qkv_fp8_noscale", 1, 8, 1024, ctx
            )

            # ---------------- head=8 (GLM TP8) ----------------
            # Smallest grid (num_q_rows=1).
            run_test_prefill_sparse_qkv_fp8[8, 64](
                "b1_s1_h8_kv256_topk64_qkv_fp8_noscale", 1, 1, 256, ctx
            )
            run_test_prefill_sparse_qkv_fp8[8, 64](
                "b1_s32_h8_kv256_topk64_qkv_fp8_noscale", 1, 32, 256, ctx
            )
            # Multi-batch, exact 1 k-block.
            run_test_prefill_sparse_qkv_fp8[8, 64](
                "b4_s16_h8_kv256_topk64_qkv_fp8_noscale", 4, 16, 256, ctx
            )
            # Exact 2 k-blocks.
            run_test_prefill_sparse_qkv_fp8[8, 128](
                "b1_s32_h8_kv512_topk128_qkv_fp8_noscale", 1, 32, 512, ctx
            )
            # Multi-block, 4 k-blocks.
            run_test_prefill_sparse_qkv_fp8[8, 256](
                "b1_s32_h8_kv1024_topk256_qkv_fp8_noscale", 1, 32, 1024, ctx
            )
            # Ragged tail < B_TOPK.
            run_test_prefill_sparse_qkv_fp8[8, 64](
                "b1_s32_h8_kv256_topk64_valid40_qkv_fp8_noscale",
                1,
                32,
                256,
                ctx,
                valid_topk=40,
            )
            # Multi-block ragged (valid mid 2nd block).
            run_test_prefill_sparse_qkv_fp8[8, 128](
                "b1_s32_h8_kv512_topk128_valid96_qkv_fp8_noscale",
                1,
                32,
                512,
                ctx,
                valid_topk=96,
            )
            # All-invalid query -> O=0.
            run_test_prefill_sparse_qkv_fp8[8, 64](
                "b1_s32_h8_kv256_topk64_valid0_qkv_fp8_noscale",
                1,
                32,
                256,
                ctx,
                valid_topk=0,
            )
            # Deep 16 k-blocks.
            run_test_prefill_sparse_qkv_fp8[8, 1024](
                "b1_s8_h8_kv1024_topk1024_qkv_fp8_noscale", 1, 8, 1024, ctx
            )
            # Attention sink (finite) — device buffer exactly 8 Float32.
            var sink_h8_finite: List[Float32] = [
                0.5,
                1.0,
                1.5,
                2.0,
                0.5,
                1.0,
                1.5,
                2.0,
            ]
            run_test_prefill_sparse_qkv_fp8[8, 64](
                "b1_s32_h8_kv256_topk64_sink_finite_qkv_fp8_noscale",
                1,
                32,
                256,
                ctx,
                sink_values=sink_h8_finite,
            )
            # Production-regime: topk=2048 (32 k-blocks) with the indexer's
            # constant-length broadcast, heads {8, 64}.
            run_test_prefill_sparse_qkv_fp8[8, 2048](
                "b1_s8_h8_len2048_valid2048_qkv_fp8_noscale",
                1,
                8,
                2048,
                ctx,
                valid_topk=2048,
                topk_lengths_override=2048,
            )
            run_test_prefill_sparse_qkv_fp8[64, 2048](
                "b1_s8_h64_len2048_valid2048_qkv_fp8_noscale",
                1,
                8,
                2048,
                ctx,
                valid_topk=2048,
                topk_lengths_override=2048,
            )
            # h16 / h32: sub-64 padded generalization.
            run_test_prefill_sparse_qkv_fp8[16, 256](
                "b1_s32_h16_kv1024_topk256_qkv_fp8_noscale", 1, 32, 1024, ctx
            )
            run_test_prefill_sparse_qkv_fp8[32, 256](
                "b1_s32_h32_kv1024_topk256_qkv_fp8_noscale", 1, 32, 1024, ctx
            )

            # ---------------- head=128 (cta_group=2, 2-CTA f8f6f4) ----------
            # Ragged: only 64 valid keys (second B_TOPK block fully masked).
            run_test_prefill_sparse_qkv_fp8[128, 128](
                "b1_s32_h128_kv512_topk128_valid64_qkv_fp8_noscale",
                1,
                32,
                512,
                ctx,
                valid_topk=64,
            )
            # 2 k-blocks (topk 128 / B_TOPK 64).
            run_test_prefill_sparse_qkv_fp8[128, 128](
                "b1_s32_h128_kv512_topk128_qkv_fp8_noscale", 1, 32, 512, ctx
            )
            # Multi-batch, exact 1 k-block.
            run_test_prefill_sparse_qkv_fp8[128, 128](
                "b4_s16_h128_kv256_topk128_qkv_fp8_noscale", 4, 16, 256, ctx
            )
            # 2 k-blocks (cross-block online-softmax state).
            run_test_prefill_sparse_qkv_fp8[128, 256](
                "b1_s32_h128_kv512_topk256_qkv_fp8_noscale", 1, 32, 512, ctx
            )
            # Ragged tail < B_TOPK (valid mid 2nd block).
            run_test_prefill_sparse_qkv_fp8[128, 256](
                "b1_s32_h128_kv512_topk256_valid192_qkv_fp8_noscale",
                1,
                32,
                512,
                ctx,
                valid_topk=192,
            )
            # Production-regime: topk=2048 (32 k-blocks at B_TOPK=64).
            run_test_prefill_sparse_qkv_fp8[128, 2048](
                "b1_s8_h128_len2048_valid2048_qkv_fp8_noscale",
                1,
                8,
                2048,
                ctx,
                valid_topk=2048,
                topk_lengths_override=2048,
            )
            # Attention sink (finite), 128 heads.
            var sink_h128 = List[Float32]()
            for _i in range(128):
                sink_h128.append(Float32(1.0))
            run_test_prefill_sparse_qkv_fp8[128, 128](
                "b1_s32_h128_kv512_topk128_sink_finite_qkv_fp8_noscale",
                1,
                32,
                512,
                ctx,
                sink_values=sink_h128,
            )

            # ---------------- prime / unaligned shapes ----------------
            # Stress the ragged-tail masking and page-gather boundaries with
            # prime lengths that align to NEITHER B_TOPK (64) NOR PAGE_SIZE
            # (128): prime batch_size/seq_len (grid edges), prime num_kv_tokens
            # (partial last page + coprime shuffle), and prime valid_topk (mask
            # boundary landing mid-B_TOPK-block). Head count can't be prime
            # (must be a multiple of 8 in (0, 64] or 128), so it stays aligned.
            # cg1 (head=8): valid=101 -> 2 k-blocks, 37 valid in the 2nd.
            run_test_prefill_sparse_qkv_fp8[8, 128](
                "b3_s13_h8_kv521_topk128_valid101_qkv_fp8_prime_unaligned",
                3,
                13,
                521,
                ctx,
                valid_topk=101,
            )
            # cg2 (head=128, 2-CTA): valid=191 -> 3 k-blocks, 63 valid in the 3rd.
            run_test_prefill_sparse_qkv_fp8[128, 256](
                "b1_s7_h128_kv263_topk256_valid191_qkv_fp8_prime_unaligned",
                1,
                7,
                263,
                ctx,
                valid_topk=191,
            )

            # ---------------- NoPE native 512, production scale -----------
            # Same bit-identical differential as above, but skipping the fp64
            # host loop -- the zero-padded 576 run IS the reference -- which
            # is what makes these larger shapes affordable. Encodes a NoPE
            # deployment's real shapes: 8-way tensor parallelism (num_heads=8
            # per GPU, with a 64-head case kept too); a sparse indexer whose
            # raw top-k is 2048, whose 4-pooled selection with a boundary
            # tail is 2051 wide, and whose index buffer is tiled to 2176
            # columns by the engine. The kernel reads whole B_TOPK=64 blocks
            # of the index buffer, bounded by
            # `ceildiv(topk_lengths[row], B_TOPK)`
            # (mla_prefill_sparse_qkv_fp8.mojo:744-749) -- so the buffer
            # width itself (the comptime `topk` below) must stay
            # B_TOPK-aligned, and 2051 is realized as a ragged `valid_topk`
            # inside the 2176-wide buffer rather than as a buffer width on
            # its own.

            # Small warm-up at each head count / buffer width used below,
            # before trusting the expensive shapes.
            run_nope_native_512_prefill_fp8[8, 2048](
                ctx,
                batch_size=1,
                seq_len=4,
                num_kv_tokens=256,
                valid_topk=64,
                name="smoke_h8_topk2048_valid64",
            )
            run_nope_native_512_prefill_fp8[64, 2176](
                ctx,
                batch_size=1,
                seq_len=4,
                num_kv_tokens=256,
                valid_topk=64,
                name="smoke_h64_topk2176_valid64",
            )

            # Saturated at the raw top-k width (2048); chunked prefill of
            # 1024 new tokens split 8x128 (not a single 1xN chunk).
            run_nope_native_512_prefill_fp8[8, 2048](
                ctx,
                batch_size=8,
                seq_len=128,
                num_kv_tokens=4096,
                valid_topk=2048,
                topk_lengths_override=2048,
                name="prod_tp8_b8x128_topk2048_saturated",
            )
            # The pooled selection's tail-of-3 width (2051), ragged inside
            # the 2176-wide buffer; 2048 new tokens split 4x512.
            run_nope_native_512_prefill_fp8[8, 2176](
                ctx,
                batch_size=4,
                seq_len=512,
                num_kv_tokens=4096,
                valid_topk=2051,
                name="prod_tp8_b4x512_kpool_tail_topk2051",
            )
            # The engine-tiled buffer width (2176) fully saturated; a single
            # 4096-token chunk (the 1xN shape the two splits above don't
            # cover).
            run_nope_native_512_prefill_fp8[8, 2176](
                ctx,
                batch_size=1,
                seq_len=4096,
                num_kv_tokens=8192,
                valid_topk=2176,
                topk_lengths_override=2176,
                name="prod_tp8_b1x4096_engine_padded_topk2176",
            )
            # The other head count this kernel serves, at the same saturated
            # engine-tiled width.
            run_nope_native_512_prefill_fp8[64, 2176](
                ctx,
                batch_size=8,
                seq_len=128,
                num_kv_tokens=4096,
                valid_topk=2176,
                topk_lengths_override=2176,
                name="prod_h64_b8x128_engine_padded_topk2176",
            )

            # Prime and page-boundary num_kv_tokens (PAGE_SIZE=128): a short
            # cached context can't supply more keys than it holds, so the
            # real selection is naturally ragged here -- valid_topk tracks
            # num_kv_tokens rather than the full comptime buffer width.
            run_nope_native_512_prefill_fp8[8, 2048](
                ctx,
                batch_size=1,
                seq_len=8,
                num_kv_tokens=127,
                valid_topk=127,
                name="prod_page_boundary_kv127",
            )
            run_nope_native_512_prefill_fp8[8, 2048](
                ctx,
                batch_size=1,
                seq_len=8,
                num_kv_tokens=128,
                valid_topk=128,
                name="prod_page_boundary_kv128",
            )
            run_nope_native_512_prefill_fp8[8, 2048](
                ctx,
                batch_size=1,
                seq_len=8,
                num_kv_tokens=129,
                valid_topk=129,
                name="prod_page_boundary_kv129",
            )
            run_nope_native_512_prefill_fp8[8, 2048](
                ctx,
                batch_size=1,
                seq_len=8,
                num_kv_tokens=1021,
                valid_topk=1021,
                name="prod_page_boundary_kv1021",
            )
            run_nope_native_512_prefill_fp8[8, 2176](
                ctx,
                batch_size=1,
                seq_len=8,
                num_kv_tokens=2053,
                valid_topk=2053,
                name="prod_page_boundary_kv2053",
            )
        else:
            pass
