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

"""Numerical E2E test for MLA_SM100_Decode_Sparse_QKV_FP8.

Native FP8-Q+KV sparse decode: Q is float8_e4m3fn, eliminating the
FP8-to-BF16 convert warpgroup present in the BF16-Q path. Output is always
bfloat16 — the kernel dequantizes internally.

Layout tested here per KV row:
    [nope: 512 FP8 bytes] [rope: 64 FP8 bytes]  = 576 bytes total

Q generation: random BF16 values are cast to FP8 before being stored and
uploaded to the device, so the reference sees exactly the same quantized Q
values as the kernel.

This test covers the full feature matrix supported by the kernel:
  * base (NullMask, no extra features)
  * CausalMask
  * multi-token Q (q_max_seq_len > 1)
  * multi-batch (batch_size > 1)
  * variable per-batch topk (has_variable_topk)
  * attention sinks (has_attn_sink)
  * extra KV (has_extra_kv)
  * topk clamping (small cache with topk > actual_tokens)
  * read-once shared-index fold (shared_index=True)
"""

from std.math import ceildiv, exp, sqrt
from std.memory import alloc, bitcast
from std.random import randn, seed
from std.sys import has_nvidia_gpu_accelerator, size_of

from max.gpu import *
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.info import _is_sm10x_gpu
from max.gpu.host.nvidia.tma import TensorMapSwizzle
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
from nn.attention.mha_mask import CausalMask, NullMask
from nn.attention.mha_operand import KVCacheMHAOperand
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
    compute_mla_dispatch_scalars,
)
from std.utils.index import IndexList
from std.utils.numerics import isnan, min_or_neg_inf


# ===-----------------------------------------------------------------------===#
# Test constants
# ===-----------------------------------------------------------------------===#

# MLA dimensions (matching DeepSeek V3 production config).
comptime Q_DEPTH = 576  # Full Q depth: 512 nope + 64 rope
comptime V_DEPTH = 512  # Output depth (nope only)
comptime ROPE_DEPTH = 64  # Rope dimension
comptime PAGE_SIZE = 128  # Standard page size
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1  # MLA has 1 KV head

# ALL-FP8 KV layout: 512 (FP8 nope) + 64 (FP8 rope) = 576 FP8 bytes.
comptime KV_HEAD_SIZE = V_DEPTH + ROPE_DEPTH  # 576


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


def _coprime_multiplier(n: Int) -> Int:
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
# Host-side reference: FP8 Q (576) x BF16 K^T (576) -> P -> BF16 O
# ===-----------------------------------------------------------------------===#


def host_reference_with_attn_sink(
    q_ptr: Pointer[Float8_e4m3fn, _],
    k_bf16_ptr: Pointer[BFloat16, _],
    output_ptr: MutPointer[BFloat16, _],
    attn_sink_host: Pointer[Float32, _],
    batch_size: Int,
    num_heads: Int,
    num_keys: Int,
    depth: Int,
    v_depth: Int,
    scale: Float32,
):
    """Reference MLA with attn_sink correction (natural log domain).

    attn_sink is shape [num_heads_q]. The softmax denominator is adjusted:
      sum_exp += exp(attn_sink[h] - max_s)
    which captures the aggregate contribution of non-selected tokens.
    """
    for b in range(batch_size):
        for h in range(num_heads):
            var q_base = b * num_heads * depth + h * depth

            var max_s = Float64(min_or_neg_inf[.float32]())
            var s_buf = List(length=num_keys, fill=Float64(0))
            for k in range(num_keys):
                var k_base = b * num_keys * depth + k * depth
                var dot = Float64(0)
                for d in range(depth):
                    dot += (
                        q_ptr[q_base + d].cast[.float64]()
                        * k_bf16_ptr[k_base + d].cast[.float64]()
                    )
                s_buf[k] = dot * Float64(scale)
                if s_buf[k] > max_s:
                    max_s = s_buf[k]

            var attn_sink_val = Float64(attn_sink_host[h])
            if attn_sink_val > max_s:
                max_s = attn_sink_val

            var sum_exp = Float64(0)
            for k in range(num_keys):
                s_buf[k] = exp(s_buf[k] - max_s)
                sum_exp += s_buf[k]
            sum_exp += exp(attn_sink_val - max_s)

            for k in range(num_keys):
                s_buf[k] = s_buf[k] / sum_exp

            var o_base = b * num_heads * v_depth + h * v_depth
            for d in range(v_depth):
                var acc = Float64(0)
                for k in range(num_keys):
                    var k_base = b * num_keys * depth + k * depth
                    acc += s_buf[k] * k_bf16_ptr[k_base + d].cast[.float64]()
                output_ptr[o_base + d] = acc.cast[.bfloat16]()
            _ = s_buf^


def host_reference_varkeys(
    q_ptr: Pointer[Float8_e4m3fn, _],
    k_bf16_ptr: Pointer[BFloat16, _],
    output_ptr: MutPointer[BFloat16, _],
    batch_size: Int,
    num_heads: Int,
    num_keys_per_batch: List[Int],
    depth: Int,
    v_depth: Int,
    scale: Float32,
):
    """Reference MLA for variable-length per-batch sparse attention.

    K buffer is packed contiguously per batch: k_bf16_ptr contains
    sum_b num_keys_per_batch[b] * depth elements.
    """
    var k_offsets = List(length=batch_size, fill=Int(0))
    var running = 0
    for b in range(batch_size):
        k_offsets[b] = running
        running += num_keys_per_batch[b] * depth

    for b in range(batch_size):
        var num_keys = num_keys_per_batch[b]
        var k_base_offset = k_offsets[b]
        for h in range(num_heads):
            var q_base = b * num_heads * depth + h * depth

            var max_s = Float64(min_or_neg_inf[.float32]())
            var s_buf = List(length=num_keys, fill=Float64(0))
            for k in range(num_keys):
                var k_base = k_base_offset + k * depth
                var dot = Float64(0)
                for d in range(depth):
                    dot += (
                        q_ptr[q_base + d].cast[.float64]()
                        * k_bf16_ptr[k_base + d].cast[.float64]()
                    )
                s_buf[k] = dot * Float64(scale)
                if s_buf[k] > max_s:
                    max_s = s_buf[k]

            var sum_exp = Float64(0)
            for k in range(num_keys):
                s_buf[k] = exp(s_buf[k] - max_s)
                sum_exp += s_buf[k]
            for k in range(num_keys):
                s_buf[k] = s_buf[k] / sum_exp

            var o_base = b * num_heads * v_depth + h * v_depth
            for d in range(v_depth):
                var acc = Float64(0)
                for k in range(num_keys):
                    var k_base = k_base_offset + k * depth
                    acc += s_buf[k] * k_bf16_ptr[k_base + d].cast[.float64]()
                output_ptr[o_base + d] = acc.cast[.bfloat16]()
            _ = s_buf^
    _ = k_offsets^


# ===-----------------------------------------------------------------------===#
# Aggregate output verification (mirrors test_mla_prefill_sparse_qkv_fp8.mojo).
#
# A single element outside tolerance used to abort the whole suite, hiding the
# shape of a failure. This computes the same aggregate metrics the prefill
# harness reports -- cosine similarity, mean/max abs error, error-tail fraction,
# non-finite counts, non-zero fraction -- and PRINTS them for every case before
# gating, so a regression shows its full character (systemic vs a few outliers)
# in one run instead of dying on element 0.
# ===-----------------------------------------------------------------------===#


def verify_mla_output(
    name: StringLiteral,
    ref_ptr: Pointer[BFloat16, _],
    out_ptr: Pointer[BFloat16, _],
    num_elems: Int,
    atol: Float64 = 5e-2,
    mean_err_max: Float64 = 0.01,
    # Fraction of finite elements allowed to exceed |err| > 1e-2. Default matches
    # the prefill harness (0.15%); few-key cases (topk <= ~8, cache_len <= ~6)
    # pass a looser bound at their call site -- with so few softmax terms, FP8-P
    # quantization pushes a percent-ish of dims just past 1e-2 while the vector
    # stays cosine > 0.999 correct.
    tail_max: Float64 = 0.0015,
) raises:
    comptime COS_MIN = Float64(0.99)

    var max_err = Float64(0)
    var max_actual = Float64(0)
    var num_nonzero = 0
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

    for i in range(num_elems):
        var ref_val = ref_ptr[i].cast[.float64]()
        var actual_val = out_ptr[i].cast[.float64]()
        var err = abs(actual_val - ref_val)
        if err > max_err:
            max_err = err
        if abs(actual_val) > max_actual:
            max_actual = abs(actual_val)
        if abs(actual_val) > 1e-6:
            num_nonzero += 1
        if isnan(actual_val):
            nan_actual += 1
        elif abs(actual_val) > 1.0e300:
            inf_actual += 1
        elif not isnan(ref_val) and abs(ref_val) <= 1.0e300:
            sum_abs_err += err
            if err > 1.0e-2:
                n_err_gt_1em2 += 1
            dot_ar += actual_val * ref_val
            norm_a += actual_val * actual_val
            norm_r += ref_val * ref_val
            n_finite += 1
        if isnan(ref_val):
            nan_ref += 1
        elif abs(ref_val) > 1.0e300:
            inf_ref += 1

    var mean_abs_err = sum_abs_err / Float64(max(n_finite, 1))
    var cos_denom = sqrt(norm_a) * sqrt(norm_r)
    var cosine: Float64
    if norm_a < 1e-12 and norm_r < 1e-12:
        cosine = 1.0
    elif cos_denom < 1e-12:
        cosine = 0.0
    else:
        cosine = dot_ar / cos_denom
    var tail_frac = Float64(n_err_gt_1em2) / Float64(max(n_finite, 1))

    print(
        "  DIAG:",
        name,
        " cosine=",
        cosine,
        " max_err=",
        max_err,
        " mean_abs_err=",
        mean_abs_err,
        " |err|>1e-2=",
        n_err_gt_1em2,
        " tail=",
        tail_frac,
        " nan_actual=",
        nan_actual,
        " inf_actual=",
        inf_actual,
        " nonzero=",
        num_nonzero,
        "/",
        num_elems,
        " max_abs_actual=",
        max_actual,
    )

    # Gates apply AFTER the DIAG print so metrics are always visible on failure.
    if nan_actual != 0 or inf_actual != 0 or nan_ref != 0 or inf_ref != 0:
        raise Error(
            String("non-finite values in ")
            + String(name)
            + ": nan_actual="
            + String(nan_actual)
            + " inf_actual="
            + String(inf_actual)
            + " nan_ref="
            + String(nan_ref)
            + " inf_ref="
            + String(inf_ref)
        )
    if num_nonzero == 0 and num_elems > 0:
        raise Error(String("output is all-zero in ") + String(name))
    if mean_abs_err > mean_err_max:
        raise Error(
            String("mean_abs_err exceeded tolerance in ")
            + String(name)
            + ": "
            + String(mean_abs_err)
            + " > "
            + String(mean_err_max)
        )
    if cosine < COS_MIN:
        raise Error(
            String("cosine below tolerance in ")
            + String(name)
            + ": "
            + String(cosine)
            + " < "
            + String(COS_MIN)
        )
    if tail_frac >= tail_max:
        raise Error(
            String("error-tail exceeded bound in ")
            + String(name)
            + ": "
            + String(n_err_gt_1em2)
            + " elems (|err| > 1e-2) = "
            + String(tail_frac * 100)
            + "% >= "
            + String(tail_max * 100)
            + "%"
        )
    if max_err > atol:
        raise Error(
            String("max_err exceeded tolerance in ")
            + String(name)
            + ": "
            + String(max_err)
            + " > atol "
            + String(atol)
        )

    print(
        "  PASSED:",
        name,
        " cosine=",
        cosine,
        " max_err=",
        max_err,
        " checked=",
        num_elems,
        " elements",
    )


# ===-----------------------------------------------------------------------===#
# Core test function: native FP8-Q + all-FP8-KV sparse decode.
# Parametrized over mask type and shared-index fold via comptime flags.
# ===-----------------------------------------------------------------------===#


comptime LATENT_DEPTH = 512


def run_nope_native_512_fp8[
    num_heads: Int,
](
    ctx: DeviceContext,
    batch_size: Int,
    cache_len: Int,
    *,
    topk: Int = -1,
    q_max_seq_len: Int = 1,
    name: StringLiteral = "",
) raises:
    """The native-FP8 kernel takes a 512-wide NoPE row and answers identically.

    Same property as the BF16-KV oracle
    (`test_mla_decode_sparse_kv_bf16.run_nope_native_512`), on the backend a
    NoPE model actually runs: one problem, run once with both operands padded
    to 576 with a zero tail and once at their native 512, required
    **bit-identical**.

    This path is worth pinning separately because it reaches the tail through
    different machinery -- an INT64-packed gather into a linear staging buffer,
    then a re-swizzle warpgroup that permutes it into the SW64 MMA layout. The
    narrow row has to shorten the gather, the staging row stride and the
    re-swizzle's column count together; any one of them left wide would corrupt
    the operand rather than merely waste bandwidth.
    """
    comptime q_type = DType.float8_e4m3fn
    var num_keys = cache_len + q_max_seq_len
    var eff_topk = num_keys if topk < 0 else topk
    comptime scale = Float32(0.125)

    print(
        "NoPE native 512 (QKV FP8): ",
        name,
        " batch_size=",
        batch_size,
        " cache_len=",
        cache_len,
        " num_heads=",
        num_heads,
    )
    seed(0x5EED18)

    comptime kv_params_576 = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=Q_DEPTH, is_mla=True
    )
    comptime kv_params_512 = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=LATENT_DEPTH, is_mla=True
    )
    comptime kv_dim2 = 1

    var pages_per_batch = ceildiv(num_keys, PAGE_SIZE)
    var total_pages = batch_size * pages_per_batch
    var rows_meta = (
        total_pages * kv_dim2 * NUM_LAYERS * PAGE_SIZE * KV_NUM_HEADS
    )

    var block_shape_576 = IndexList[6](
        total_pages, kv_dim2, NUM_LAYERS, PAGE_SIZE, KV_NUM_HEADS, Q_DEPTH
    )
    var block_shape_512 = IndexList[6](
        total_pages, kv_dim2, NUM_LAYERS, PAGE_SIZE, KV_NUM_HEADS, LATENT_DEPTH
    )

    # One latent cache, quantized once, laid out at both widths so the two runs
    # see exactly the same FP8 bytes over [0, 512).
    var src = ctx.enqueue_create_host_buffer[.bfloat16](
        rows_meta * LATENT_DEPTH
    )
    randn(src.as_span(), mean=0.0, standard_deviation=0.5)
    var blocks_512_host = ctx.enqueue_create_host_buffer[q_type](
        rows_meta * LATENT_DEPTH
    )
    var blocks_576_host = ctx.enqueue_create_host_buffer[q_type](
        rows_meta * Q_DEPTH
    )
    for r in range(rows_meta):
        for d in range(LATENT_DEPTH):
            var v = src[r * LATENT_DEPTH + d].cast[q_type]()
            blocks_512_host[r * LATENT_DEPTH + d] = v
            blocks_576_host[r * Q_DEPTH + d] = v
        for d in range(LATENT_DEPTH, Q_DEPTH):
            blocks_576_host[r * Q_DEPTH + d] = Scalar[q_type](0)

    var lut_size = batch_size * pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for bi in range(batch_size):
        for p in range(pages_per_batch):
            lookup_table_host[bi * pages_per_batch + p] = UInt32(
                bi * pages_per_batch + p
            )
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_len)

    var total_q_tokens = batch_size * q_max_seq_len
    var q_rows = total_q_tokens * num_heads
    var q_src = ctx.enqueue_create_host_buffer[.bfloat16](q_rows * LATENT_DEPTH)
    randn(q_src.as_span(), mean=0.0, standard_deviation=0.5)
    var q_512_host = ctx.enqueue_create_host_buffer[q_type](
        q_rows * LATENT_DEPTH
    )
    var q_576_host = ctx.enqueue_create_host_buffer[q_type](q_rows * Q_DEPTH)
    for row in range(q_rows):
        for d in range(LATENT_DEPTH):
            var v = q_src[row * LATENT_DEPTH + d].cast[q_type]()
            q_512_host[row * LATENT_DEPTH + d] = v
            q_576_host[row * Q_DEPTH + d] = v
        for d in range(LATENT_DEPTH, Q_DEPTH):
            q_576_host[row * Q_DEPTH + d] = Scalar[q_type](0)

    # Indices are per QUERY TOKEN, not per batch: the kernel walks
    # `d_indices` with a stride of `topk` for each of the `total_q_tokens`
    # rows. These coincide only when `q_max_seq_len == 1`.
    var total_indices = total_q_tokens * eff_topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for bi in range(batch_size):
        for si in range(q_max_seq_len):
            var g = bi * q_max_seq_len + si
            for i in range(eff_topk):
                var page_idx = i // PAGE_SIZE
                var tok_in_page = i % PAGE_SIZE
                var block_id = Int(
                    lookup_table_host[bi * pages_per_batch + page_idx]
                )
                h_indices[g * eff_topk + i] = Int32(
                    block_id * PAGE_SIZE + tok_in_page
                )

    var out_size = total_q_tokens * num_heads * V_DEPTH

    var blocks_576_dev = ctx.enqueue_create_buffer[q_type](rows_meta * Q_DEPTH)
    ctx.enqueue_copy(blocks_576_dev, blocks_576_host)
    var blocks_512_dev = ctx.enqueue_create_buffer[q_type](
        rows_meta * LATENT_DEPTH
    )
    ctx.enqueue_copy(blocks_512_dev, blocks_512_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_576_dev = ctx.enqueue_create_buffer[q_type](q_rows * Q_DEPTH)
    ctx.enqueue_copy(q_576_dev, q_576_host)
    var q_512_dev = ctx.enqueue_create_buffer[q_type](q_rows * LATENT_DEPTH)
    ctx.enqueue_copy(q_512_dev, q_512_host)
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)
    var out_576 = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var out_512 = ctx.enqueue_create_buffer[.bfloat16](out_size)

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i * q_max_seq_len)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    comptime cl_layout = Layout(UNKNOWN_VALUE)
    comptime lt_layout_2d = Layout.row_major[2]()
    comptime blk_layout = Layout.row_major[6]()

    var cl_lt = LayoutTensor[mut=False, .uint32, cl_layout](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )
    var lt_lt = LayoutTensor[mut=False, .uint32, lt_layout_2d](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, pages_per_batch)
        ),
    )

    var kv_576 = PagedKVCacheCollection[q_type, kv_params_576, PAGE_SIZE](
        LayoutTensor[q_type, blk_layout](
            blocks_576_dev.unsafe_ptr(),
            RuntimeLayout[blk_layout].row_major(block_shape_576),
        ),
        cl_lt,
        lt_lt,
        UInt32(q_max_seq_len),
        UInt32(cache_len),
    )
    var kv_512 = PagedKVCacheCollection[q_type, kv_params_512, PAGE_SIZE](
        LayoutTensor[q_type, blk_layout](
            blocks_512_dev.unsafe_ptr(),
            RuntimeLayout[blk_layout].row_major(block_shape_512),
        ),
        cl_lt,
        lt_lt,
        UInt32(q_max_seq_len),
        UInt32(cache_len),
    )

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(), row_major(batch_size + 1)
    )
    var args_576 = MLADispatchScalarArgs[num_heads=num_heads, is_fp8_kv=True](
        batch_size, cache_len, q_max_seq_len, ctx
    )
    var args_512 = MLADispatchScalarArgs[num_heads=num_heads, is_fp8_kv=True](
        batch_size, cache_len, q_max_seq_len, ctx
    )

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
    ](
        TileTensor(
            out_576.unsafe_ptr(),
            row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
        ),
        TileTensor(
            q_576_dev.unsafe_ptr(),
            row_major((total_q_tokens, Idx[num_heads], Idx[Q_DEPTH])),
        ),
        kv_576.get_key_cache(0),
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        args_576.gpu_tile_tensor(),
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=eff_topk,
    )
    ctx.synchronize()

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, LATENT_DEPTH),
        ragged=True,
        sparse=True,
    ](
        TileTensor(
            out_512.unsafe_ptr(),
            row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
        ),
        TileTensor(
            q_512_dev.unsafe_ptr(),
            row_major((total_q_tokens, Idx[num_heads], Idx[LATENT_DEPTH])),
        ),
        kv_512.get_key_cache(0),
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        args_512.gpu_tile_tensor(),
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=eff_topk,
    )
    ctx.synchronize()

    var h576 = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    var h512 = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(h576, out_576)
    ctx.enqueue_copy(h512, out_512)
    ctx.synchronize()

    var mismatches = 0
    for i in range(out_size):
        if isnan(h512[i].cast[.float64]()):
            raise Error("NaN in native-512 QKV-FP8 decode output")
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
        print("  FAILED: ", mismatches, "of", out_size, "outputs differ")
        raise Error("native-512 QKV-FP8 decode does not match padded 576")
    print("  PASSED: native 512 is bit-identical to zero-padded 576")

    _ = args_576
    _ = args_512
    _ = blocks_576_dev
    _ = blocks_512_dev
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_576_dev
    _ = q_512_dev
    _ = d_indices_device
    _ = out_576
    _ = out_512
    _ = row_offsets_device


def run_test_sparse_qkv_fp8[
    kv_type: DType,  # float8_e4m3fn
    num_heads: Int,
    use_causal: Bool = False,
    # Read-once shared-index fold (KERN-3141).
    shared_index: Bool = False,
    # Physical gather order of the ONE shared set (shared_index only).
    # 0=identity, 1=reversed, 2=coprime permutation.
    order_mode: Int = 0,
    # Phase 5 seq_len=0: make the LAST batch a 0-length (empty) ragged sequence.
    empty_last_batch: Bool = False,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    ctx: DeviceContext,
    topk: Int,
    q_max_seq_len: Int = 1,
    forced_np: Int = 0,
    tail_max: Float64 = 0.0015,
) raises:
    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " cache_len:",
        cache_len,
        " num_heads:",
        num_heads,
        " topk:",
        topk,
        " q_max_seq_len:",
        q_max_seq_len,
        " causal:",
        use_causal,
    )

    var num_keys = cache_len + q_max_seq_len
    var total_q_tokens = batch_size * q_max_seq_len
    comptime scale = Float32(0.125)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=row_depth, is_mla=True
    )
    comptime kv_dim2 = 1

    var total_pages = batch_size * ceildiv(num_keys, PAGE_SIZE)
    var max_pages_per_batch = ceildiv(num_keys, PAGE_SIZE)

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    # Allocate KV cache on host, zero-initialized.
    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)

    # Generate random BF16 K data for nope (512) + rope (64) per token.
    var k_bf16_total = batch_size * num_keys * row_depth
    var k_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](k_bf16_total)
    randn(k_bf16_host.as_span(), mean=0.0, standard_deviation=0.5)

    var tok_stride = kv_params.head_size  # 576 FP8 slots

    # Build shuffled page table (exercises scatter).
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np = ceildiv(num_keys, PAGE_SIZE)
        var mult = _coprime_multiplier(np)
        for p in range(np):
            var shuffled_p = (p * mult + 1) % np
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_len)

    # Fill KV cache: nope (512) AND rope (64) both cast BF16 -> FP8.
    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    for bi in range(batch_size):
        for t in range(num_keys):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * tok_stride
            var k_base = bi * num_keys * row_depth + t * row_depth
            for d in range(row_depth):
                blocks_host[base + d] = k_bf16_host[k_base + d].cast[kv_type]()

    # Reference K: read back FP8 bytes as BF16 so the reference sees
    # exactly the same quantized values as the kernel.
    var k_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](k_bf16_total)
    for bi in range(batch_size):
        for t in range(num_keys):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * tok_stride
            var k_base = bi * num_keys * row_depth + t * row_depth
            for d in range(row_depth):
                k_ref_host[k_base + d] = blocks_host[base + d].cast[
                    DType.bfloat16
                ]()

    # Q tensor: generate BF16 randn, cast to FP8 for the kernel.
    var q_size = batch_size * q_max_seq_len * num_heads * row_depth
    var q_bf16_scratch = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_bf16_scratch.as_span(), mean=0.0, standard_deviation=0.5)
    var q_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    for i in range(q_size):
        q_host[i] = q_bf16_scratch[i].cast[.float8_e4m3fn]()

    # Select topk unique tokens PER QUERY TOKEN via deterministic permutation.
    var selected_tokens = List(length=total_q_tokens * topk, fill=Int(0))
    var mult = _coprime_multiplier(num_keys)
    var perm_mult = _coprime_multiplier(topk)
    for bi in range(batch_size):
        for s in range(q_max_seq_len):
            var g = bi * q_max_seq_len + s
            for i in range(topk):
                comptime if shared_index:
                    if order_mode == 1:
                        selected_tokens[g * topk + i] = topk - 1 - i
                    elif order_mode == 2:
                        selected_tokens[g * topk + i] = (i * perm_mult) % topk
                    else:
                        selected_tokens[g * topk + i] = i
                else:
                    selected_tokens[g * topk + i] = (
                        i * mult + 1 + s
                    ) % num_keys

    # Build sparse reference K buffer [total_q_tokens, topk, row_depth].
    var k_sparse_ref_size = total_q_tokens * topk * row_depth
    var k_sparse_ref = ctx.enqueue_create_host_buffer[.bfloat16](
        k_sparse_ref_size
    )
    for bi in range(batch_size):
        for s in range(q_max_seq_len):
            var g = bi * q_max_seq_len + s
            for i in range(topk):
                var t = selected_tokens[g * topk + i]
                var src_base = bi * num_keys * row_depth + t * row_depth
                var dst_base = g * topk * row_depth + i * row_depth
                for d in range(row_depth):
                    k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]

    var out_size = batch_size * q_max_seq_len * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)

    if use_causal:
        for b in range(batch_size):
            for s in range(q_max_seq_len):
                var g = b * q_max_seq_len + s
                var causal_limit = cache_len + s + 1
                for h in range(num_heads):
                    var q_base = (
                        b * q_max_seq_len * num_heads * row_depth
                        + s * num_heads * row_depth
                        + h * row_depth
                    )
                    var max_s = Float64(min_or_neg_inf[.float32]())
                    var s_buf = List(length=topk, fill=Float64(0))
                    var valid = List(length=topk, fill=False)
                    for i in range(topk):
                        var tok = selected_tokens[g * topk + i]
                        if tok >= causal_limit:
                            valid[i] = False
                            s_buf[i] = Float64(min_or_neg_inf[.float32]())
                            continue
                        valid[i] = True
                        var k_base = g * topk * row_depth + i * row_depth
                        var dot = Float64(0)
                        for d in range(row_depth):
                            dot += (
                                q_host[q_base + d].cast[.float64]()
                                * k_sparse_ref[k_base + d].cast[.float64]()
                            )
                        s_buf[i] = dot * Float64(scale)
                        if s_buf[i] > max_s:
                            max_s = s_buf[i]

                    var sum_exp = Float64(0)
                    for i in range(topk):
                        if not valid[i]:
                            s_buf[i] = Float64(0)
                            continue
                        s_buf[i] = exp(s_buf[i] - max_s)
                        sum_exp += s_buf[i]
                    if sum_exp > Float64(0):
                        for i in range(topk):
                            if valid[i]:
                                s_buf[i] = s_buf[i] / sum_exp

                    var o_base = (
                        b * q_max_seq_len * num_heads * V_DEPTH
                        + s * num_heads * V_DEPTH
                        + h * V_DEPTH
                    )
                    for d in range(V_DEPTH):
                        var acc = Float64(0)
                        for i in range(topk):
                            if not valid[i]:
                                continue
                            var k_base = g * topk * row_depth + i * row_depth
                            acc += (
                                s_buf[i]
                                * k_sparse_ref[k_base + d].cast[.float64]()
                            )
                        ref_host[o_base + d] = acc.cast[.bfloat16]()
                    _ = valid^
                    _ = s_buf^
    else:
        for b in range(batch_size):
            for s in range(q_max_seq_len):
                var g = b * q_max_seq_len + s
                for h in range(num_heads):
                    var q_base = (
                        b * q_max_seq_len * num_heads * row_depth
                        + s * num_heads * row_depth
                        + h * row_depth
                    )
                    var max_s = Float64(min_or_neg_inf[.float32]())
                    var s_buf = List(length=topk, fill=Float64(0))
                    for i in range(topk):
                        var k_base = g * topk * row_depth + i * row_depth
                        var dot = Float64(0)
                        for d in range(row_depth):
                            dot += (
                                q_host[q_base + d].cast[.float64]()
                                * k_sparse_ref[k_base + d].cast[.float64]()
                            )
                        s_buf[i] = dot * Float64(scale)
                        if s_buf[i] > max_s:
                            max_s = s_buf[i]

                    var sum_exp = Float64(0)
                    for i in range(topk):
                        s_buf[i] = exp(s_buf[i] - max_s)
                        sum_exp += s_buf[i]
                    for i in range(topk):
                        s_buf[i] = s_buf[i] / sum_exp

                    var o_base = (
                        b * q_max_seq_len * num_heads * V_DEPTH
                        + s * num_heads * V_DEPTH
                        + h * V_DEPTH
                    )
                    for d in range(V_DEPTH):
                        var acc = Float64(0)
                        for i in range(topk):
                            var k_base = g * topk * row_depth + i * row_depth
                            acc += (
                                s_buf[i]
                                * k_sparse_ref[k_base + d].cast[.float64]()
                            )
                        ref_host[o_base + d] = acc.cast[.bfloat16]()
                    _ = s_buf^

    # -----------------------------------------------------------------------
    # Copy to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var out_device = ctx.enqueue_create_buffer[.bfloat16](out_size)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build PagedKVCacheCollection on device
    # -----------------------------------------------------------------------
    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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

    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
        LayoutTensor[kv_type, Layout.row_major[6]()](
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
        UInt32(q_max_seq_len),
        UInt32(cache_len),
    )

    var kv_cache = kv_collection.get_key_cache(0)
    var kv_lut = KVCacheMHAOperand(kv_cache)

    # -----------------------------------------------------------------------
    # Build gather4 indices for the selected topk tokens.
    # -----------------------------------------------------------------------
    var total_indices = total_q_tokens * topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for bi in range(batch_size):
        for s in range(q_max_seq_len):
            var g = bi * q_max_seq_len + s
            for i in range(topk):
                var t = selected_tokens[g * topk + i]
                var page_idx = t // PAGE_SIZE
                var tok_in_page = t % PAGE_SIZE
                var block_id = Int(
                    lookup_table_host[bi * max_pages_per_batch + page_idx]
                )
                h_indices[g * topk + i] = Int32(
                    block_id * PAGE_SIZE + tok_in_page
                )

    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)
    ctx.synchronize()

    # Logical sparse indices, matching production: `selected_tokens` is a
    # score-sorted, non-monotonic permutation of logical positions, so this
    # feeds the causal-by-logical-position mask its expected input. Only
    # meaningful when use_causal.
    var h_logical_indices = ctx.enqueue_create_host_buffer[.int32](
        total_indices
    )
    for idx in range(total_indices):
        h_logical_indices[idx] = Int32(selected_tokens[idx])
    var logical_indices_device = ctx.enqueue_create_buffer[.int32](
        total_indices
    )
    ctx.enqueue_copy(logical_indices_device, h_logical_indices)
    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build TileTensors and call flare_mla_decoding.
    # -----------------------------------------------------------------------
    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[row_depth])),
    )

    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i * q_max_seq_len)
    if empty_last_batch and batch_size > 1:
        row_offsets_host[batch_size] = row_offsets_host[batch_size - 1]
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(batch_size + 1),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
        fold_shared_index=shared_index,
    ](batch_size, cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads,
        is_fp8_kv=True,
        half_sms=sm_count // 2,
        fold_shared_index=shared_index,
    ](batch_size, cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    var indices_stride = topk

    var _np_ovr = Optional[Int](forced_np) if forced_np > 0 else Optional[Int](
        None
    )

    print(
        "  Launching MLA sparse QKV_FP8 decode kernel...",
        " topk=",
        topk,
        " num_keys=",
        num_keys,
    )

    comptime if use_causal:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[.float8_e4m3fn](num_heads, row_depth),
            ragged=True,
            sparse=True,
            fold_shared_index=shared_index,
        ](
            out_tt,
            q_tt,
            kv_cache,
            CausalMask(),
            row_offsets_tt,
            scale,
            ctx,
            scalar_args_buf_tt,
            d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
                d_indices_device.unsafe_ptr()
            ),
            indices_stride=indices_stride,
            num_partitions_in=_np_ovr,
            logical_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
                logical_indices_device.unsafe_ptr()
            ),
        )
    else:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[.float8_e4m3fn](num_heads, row_depth),
            ragged=True,
            sparse=True,
            fold_shared_index=shared_index,
        ](
            out_tt,
            q_tt,
            kv_cache,
            NullMask(),
            row_offsets_tt,
            scale,
            ctx,
            scalar_args_buf_tt,
            d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
                d_indices_device.unsafe_ptr()
            ),
            indices_stride=indices_stride,
            num_partitions_in=_np_ovr,
        )

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Verify output.
    # -----------------------------------------------------------------------
    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    # `empty_last_batch` zero-fills the final ragged sequence; skip it by
    # verifying only the contiguous valid prefix (output is batch-major).
    var valid_batches = batch_size - 1 if empty_last_batch else batch_size
    var num_verify = valid_batches * q_max_seq_len * num_heads * V_DEPTH
    verify_mla_output(
        name,
        ref_host.unsafe_ptr(),
        out_host.unsafe_ptr(),
        num_verify,
        tail_max=tail_max,
    )

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = row_offsets_device
    _ = selected_tokens^


# ===-----------------------------------------------------------------------===#
# Variable-topk sparse test (has_variable_topk=True)
# ===-----------------------------------------------------------------------===#


def run_test_sparse_qkv_fp8_variable_topk[
    kv_type: DType,
    num_heads: Int,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    """Variable-topk per-batch with native FP8-Q + all-FP8-KV (576-byte row).

    Per-batch cache lengths and topk counts. indices_stride = max(topk).
    """
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)

    var max_topk = 0
    for bi in range(batch_size):
        if topk_per_batch[bi] > max_topk:
            max_topk = topk_per_batch[bi]

    print("test:", name, " batch_size:", batch_size, " num_heads:", num_heads)
    for i in range(batch_size):
        print(
            "  batch",
            i,
            ": cache_len=",
            cache_lengths[i],
            " topk=",
            topk_per_batch[i],
        )
    print("  indices_stride (max topk)=", max_topk)

    var max_cache_len = 0
    var total_pages = 0
    var num_keys_list = List[Int]()
    for i in range(batch_size):
        var cl = cache_lengths[i]
        if cl > max_cache_len:
            max_cache_len = cl
        var nk = cl + q_max_seq_len
        num_keys_list.append(nk)
        total_pages += ceildiv(nk, PAGE_SIZE)

    var max_num_keys = max_cache_len + q_max_seq_len
    var max_pages_per_batch = ceildiv(max_num_keys, PAGE_SIZE)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=row_depth, is_mla=True
    )
    comptime kv_dim2 = 1

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var tok_stride = kv_params.head_size

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np_bi

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * row_depth
    var k_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](total_k_elems)
    randn(k_bf16_host.as_span(), mean=0.0, standard_deviation=0.5)

    var k_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](total_k_elems)

    var k_offset = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        for t in range(nk):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = k_offset + t * row_depth
            for d in range(row_depth):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[.bfloat16]()
        k_offset += nk * row_depth

    # Q: generate BF16 randn, cast to FP8.
    var q_size = batch_size * num_heads * row_depth
    var q_bf16_scratch = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_bf16_scratch.as_span(), mean=0.0, standard_deviation=0.5)
    var q_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    for i in range(q_size):
        q_host[i] = q_bf16_scratch[i].cast[.float8_e4m3fn]()

    var total_indices = batch_size * max_topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for i in range(total_indices):
        h_indices[i] = Int32(0)
    var total_sparse_ref_elems = 0
    for bi in range(batch_size):
        total_sparse_ref_elems += topk_per_batch[bi] * row_depth
    var k_sparse_ref = ctx.enqueue_create_host_buffer[.bfloat16](
        total_sparse_ref_elems
    )

    var sparse_ref_offset = 0
    var k_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var topk_bi = topk_per_batch[bi]
        var mult = _coprime_multiplier(nk)
        for i in range(topk_bi):
            var t = (i * mult + 1) % nk
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * max_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = k_offset_src + t * row_depth
            var dst_base = sparse_ref_offset + i * row_depth
            for d in range(row_depth):
                k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]
        sparse_ref_offset += topk_bi * row_depth
        k_offset_src += nk * row_depth

    var sparse_num_keys_list = List[Int]()
    for bi in range(batch_size):
        sparse_num_keys_list.append(topk_per_batch[bi])

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    host_reference_varkeys(
        q_host.unsafe_ptr(),
        k_sparse_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        batch_size,
        num_heads,
        sparse_num_keys_list,
        row_depth,
        V_DEPTH,
        scale,
    )

    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    var topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    ctx.synchronize()

    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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
    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
        LayoutTensor[kv_type, Layout.row_major[6]()](
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
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[row_depth])),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(batch_size + 1),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    print("  Launching MLA sparse QKV_FP8 (variable topk)...")

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[.float8_e4m3fn](num_heads, row_depth),
        ragged=True,
        sparse=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf_tt,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=max_topk,
        topk_lengths=rebind[MutPointer[Int32, MutAnyOrigin]](
            topk_lengths_device.unsafe_ptr()
        ),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    verify_mla_output(
        name,
        ref_host.unsafe_ptr(),
        out_host.unsafe_ptr(),
        batch_size * num_heads * V_DEPTH,
    )

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = row_offsets_device


# ===-----------------------------------------------------------------------===#
# Attention sink sparse test (has_attn_sink=True)
# ===-----------------------------------------------------------------------===#


def run_test_sparse_qkv_fp8_attn_sink[
    kv_type: DType,
    num_heads: Int,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    ctx: DeviceContext,
    topk: Int,
) raises:
    """Native FP8-Q sparse MLA decode with attn_sink correction.

    attn_sink shape is [num_heads_q] per the MOGG fix.
    """
    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " cache_len:",
        cache_len,
        " num_heads:",
        num_heads,
        " topk:",
        topk,
    )

    comptime q_max_seq_len = 1
    var num_keys = cache_len + q_max_seq_len
    comptime scale = Float32(0.125)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=row_depth, is_mla=True
    )
    comptime kv_dim2 = 1

    var total_pages = batch_size * ceildiv(num_keys, PAGE_SIZE)
    var max_pages_per_batch = ceildiv(num_keys, PAGE_SIZE)

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    var k_bf16_total = batch_size * num_keys * row_depth
    var k_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](k_bf16_total)
    randn(k_bf16_host.as_span(), mean=0.0, standard_deviation=0.5)

    var tok_stride = kv_params.head_size
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np = ceildiv(num_keys, PAGE_SIZE)
        var mult = _coprime_multiplier(np)
        for p in range(np):
            var shuffled_p = (p * mult + 1) % np
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np

    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    for bi in range(batch_size):
        for t in range(num_keys):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * tok_stride
            var k_base = bi * num_keys * row_depth + t * row_depth
            for d in range(row_depth):
                blocks_host[base + d] = k_bf16_host[k_base + d].cast[kv_type]()

    var k_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](k_bf16_total)
    for bi in range(batch_size):
        for t in range(num_keys):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * tok_stride
            var k_base = bi * num_keys * row_depth + t * row_depth
            for d in range(row_depth):
                k_ref_host[k_base + d] = blocks_host[base + d].cast[
                    DType.bfloat16
                ]()

    var selected_tokens = List(length=batch_size * topk, fill=Int(0))
    for bi in range(batch_size):
        var mult = _coprime_multiplier(num_keys)
        for i in range(topk):
            selected_tokens[bi * topk + i] = (i * mult + 1) % num_keys

    var k_sparse_ref = ctx.enqueue_create_host_buffer[.bfloat16](
        batch_size * topk * row_depth
    )
    for bi in range(batch_size):
        for i in range(topk):
            var t = selected_tokens[bi * topk + i]
            var src = bi * num_keys * row_depth + t * row_depth
            var dst = bi * topk * row_depth + i * row_depth
            for d in range(row_depth):
                k_sparse_ref[dst + d] = k_ref_host[src + d]

    var attn_sink_host = ctx.enqueue_create_host_buffer[.float32](num_heads)
    for h in range(num_heads):
        attn_sink_host[h] = Float32(
            -1.0 + 3.0 * Float32(h) / Float32(num_heads)
        )

    # Q: generate BF16 randn, cast to FP8.
    var q_size = batch_size * num_heads * row_depth
    var q_bf16_scratch = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_bf16_scratch.as_span(), mean=0.0, standard_deviation=0.3)
    var q_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    for i in range(q_size):
        q_host[i] = q_bf16_scratch[i].cast[.float8_e4m3fn]()

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    host_reference_with_attn_sink(
        q_host.unsafe_ptr(),
        k_sparse_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        attn_sink_host.unsafe_ptr(),
        batch_size,
        num_heads,
        topk,
        row_depth,
        V_DEPTH,
        scale,
    )

    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_len)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var attn_sink_device = ctx.enqueue_create_buffer[.float32](num_heads)
    ctx.enqueue_copy(attn_sink_device, attn_sink_host)

    ctx.synchronize()

    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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
    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
        LayoutTensor[kv_type, Layout.row_major[6]()](
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
        UInt32(q_max_seq_len),
        UInt32(cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var total_indices = batch_size * topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for bi in range(batch_size):
        for i in range(topk):
            var t = selected_tokens[bi * topk + i]
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * topk + i] = Int32(block_id * PAGE_SIZE + tok_in_page)

    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)
    ctx.synchronize()

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[row_depth])),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(batch_size + 1),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    print("  Launching MLA sparse QKV_FP8 (attn_sink)...")

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[.float8_e4m3fn](num_heads, row_depth),
        ragged=True,
        sparse=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf_tt,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=topk,
        attn_sink_ptr=rebind[MutPointer[Float32, origin=MutAnyOrigin]](
            attn_sink_device.unsafe_ptr()
        ),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    verify_mla_output(
        name,
        ref_host.unsafe_ptr(),
        out_host.unsafe_ptr(),
        batch_size * num_heads * V_DEPTH,
    )

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = row_offsets_device
    _ = attn_sink_device
    _ = selected_tokens^


# ===-----------------------------------------------------------------------===#
# Extra KV sparse test (has_extra_kv=True)
# Both caches use the same 576-byte all-FP8 layout.
# ===-----------------------------------------------------------------------===#


def run_test_sparse_qkv_fp8_extra_kv[
    kv_type: DType,
    num_heads: Int,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    extra_cache_lengths: List[Int],
    extra_topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    """Two paged KV caches (main + always-attend). Both all-FP8, native FP8-Q.
    """
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)

    var max_topk = 0
    var max_extra_topk = 0
    for bi in range(batch_size):
        if topk_per_batch[bi] > max_topk:
            max_topk = topk_per_batch[bi]
        if extra_topk_per_batch[bi] > max_extra_topk:
            max_extra_topk = extra_topk_per_batch[bi]

    print("test:", name, " batch_size:", batch_size, " num_heads:", num_heads)
    for i in range(batch_size):
        print(
            "  batch",
            i,
            ": cache_len=",
            cache_lengths[i],
            " topk=",
            topk_per_batch[i],
            " extra_cache_len=",
            extra_cache_lengths[i],
            " extra_topk=",
            extra_topk_per_batch[i],
        )

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=row_depth, is_mla=True
    )
    comptime kv_dim2 = 1
    var tok_stride = kv_params.head_size
    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    # --- ORIGINAL cache ---
    var max_cache_len = 0
    var total_pages = 0
    var num_keys_list = List[Int]()
    for i in range(batch_size):
        var cl = cache_lengths[i]
        if cl > max_cache_len:
            max_cache_len = cl
        var nk = cl + q_max_seq_len
        num_keys_list.append(nk)
        total_pages += ceildiv(nk, PAGE_SIZE)
    var max_num_keys = max_cache_len + q_max_seq_len
    var max_pages_per_batch = ceildiv(max_num_keys, PAGE_SIZE)

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np_bi

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * row_depth
    var k_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](total_k_elems)
    randn(k_bf16_host.as_span(), mean=0.0, standard_deviation=0.5)
    var k_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](total_k_elems)

    var k_offset = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        for t in range(nk):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = k_offset + t * row_depth
            for d in range(row_depth):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[.bfloat16]()
        k_offset += nk * row_depth

    # --- EXTRA cache ---
    var max_extra_cache_len = 0
    var extra_total_pages = 0
    var extra_num_keys_list = List[Int]()
    for i in range(batch_size):
        var ecl = extra_cache_lengths[i]
        if ecl > max_extra_cache_len:
            max_extra_cache_len = ecl
        var enk = ecl + q_max_seq_len
        extra_num_keys_list.append(enk)
        extra_total_pages += ceildiv(enk, PAGE_SIZE)
    var max_extra_num_keys = max_extra_cache_len + q_max_seq_len
    var max_extra_pages_per_batch = ceildiv(max_extra_num_keys, PAGE_SIZE)

    var extra_block_shape = IndexList[6](
        extra_total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var extra_block_elems = (
        extra_total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var extra_blocks_host = ctx.enqueue_create_host_buffer[kv_type](
        extra_block_elems
    )
    for i in range(extra_block_elems):
        extra_blocks_host[i] = Scalar[kv_type](0)
    var extra_lut_size = batch_size * max_extra_pages_per_batch
    var extra_lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](
        extra_lut_size
    )
    for i in range(extra_lut_size):
        extra_lookup_table_host[i] = UInt32(0)
    var extra_page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(extra_num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            extra_lookup_table_host[
                bi * max_extra_pages_per_batch + p
            ] = UInt32(extra_page_offset + shuffled_p)
        extra_page_offset += np_bi

    var extra_cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size
    )
    for i in range(batch_size):
        extra_cache_lengths_host[i] = UInt32(extra_cache_lengths[i])

    var extra_total_k_elems = 0
    for bi in range(batch_size):
        extra_total_k_elems += extra_num_keys_list[bi] * row_depth
    var extra_k_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](
        extra_total_k_elems
    )
    randn(extra_k_bf16_host.as_span(), mean=0.0, standard_deviation=0.5)
    var extra_k_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](
        extra_total_k_elems
    )

    var ek_offset = 0
    for bi in range(batch_size):
        var enk = extra_num_keys_list[bi]
        for t in range(enk):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                extra_lookup_table_host[
                    bi * max_extra_pages_per_batch + page_idx
                ]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = ek_offset + t * row_depth
            for d in range(row_depth):
                var fp8_val = extra_k_bf16_host[k_base + d].cast[kv_type]()
                extra_blocks_host[base + d] = fp8_val
                extra_k_ref_host[k_base + d] = fp8_val.cast[.bfloat16]()
        ek_offset += enk * row_depth

    # Q: generate BF16 randn, cast to FP8.
    var q_size = batch_size * num_heads * row_depth
    var q_bf16_scratch = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_bf16_scratch.as_span(), mean=0.0, standard_deviation=0.5)
    var q_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    for i in range(q_size):
        q_host[i] = q_bf16_scratch[i].cast[.float8_e4m3fn]()

    # Build combined reference (main topk + extra topk per batch).
    var total_indices = batch_size * max_topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for i in range(total_indices):
        h_indices[i] = Int32(0)
    var extra_total_indices = batch_size * max_extra_topk
    var extra_h_indices = ctx.enqueue_create_host_buffer[.int32](
        extra_total_indices
    )
    for i in range(extra_total_indices):
        extra_h_indices[i] = Int32(0)
    var total_combined_ref_elems = 0
    for bi in range(batch_size):
        total_combined_ref_elems += (
            topk_per_batch[bi] + extra_topk_per_batch[bi]
        ) * row_depth
    var k_combined_ref = ctx.enqueue_create_host_buffer[.bfloat16](
        total_combined_ref_elems
    )

    var combined_ref_offset = 0
    var k_offset_src = 0
    var ek_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var topk_bi = topk_per_batch[bi]
        var mult = _coprime_multiplier(nk)
        for i in range(topk_bi):
            var t = (i * mult + 1) % nk
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * max_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = k_offset_src + t * row_depth
            var dst_base = combined_ref_offset + i * row_depth
            for d in range(row_depth):
                k_combined_ref[dst_base + d] = k_ref_host[src_base + d]
        combined_ref_offset += topk_bi * row_depth

        var enk = extra_num_keys_list[bi]
        var extra_topk_bi = extra_topk_per_batch[bi]
        var extra_mult = _coprime_multiplier(enk)
        for i in range(extra_topk_bi):
            var t = (i * extra_mult + 1) % enk
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                extra_lookup_table_host[
                    bi * max_extra_pages_per_batch + page_idx
                ]
            )
            extra_h_indices[bi * max_extra_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = ek_offset_src + t * row_depth
            var dst_base = combined_ref_offset + i * row_depth
            for d in range(row_depth):
                k_combined_ref[dst_base + d] = extra_k_ref_host[src_base + d]
        combined_ref_offset += extra_topk_bi * row_depth
        k_offset_src += nk * row_depth
        ek_offset_src += enk * row_depth

    var combined_num_keys_list = List[Int]()
    for bi in range(batch_size):
        combined_num_keys_list.append(
            topk_per_batch[bi] + extra_topk_per_batch[bi]
        )

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    host_reference_varkeys(
        q_host.unsafe_ptr(),
        k_combined_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        batch_size,
        num_heads,
        combined_num_keys_list,
        row_depth,
        V_DEPTH,
        scale,
    )

    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var extra_blocks_device = ctx.enqueue_create_buffer[kv_type](
        extra_block_elems
    )
    ctx.enqueue_copy(extra_blocks_device, extra_blocks_host)
    var extra_cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
        batch_size
    )
    ctx.enqueue_copy(extra_cache_lengths_device, extra_cache_lengths_host)
    var extra_lookup_table_device = ctx.enqueue_create_buffer[.uint32](
        extra_lut_size
    )
    ctx.enqueue_copy(extra_lookup_table_device, extra_lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[.bfloat16](out_size)

    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)
    var extra_d_indices_device = ctx.enqueue_create_buffer[.int32](
        extra_total_indices
    )
    ctx.enqueue_copy(extra_d_indices_device, extra_h_indices)

    var topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    var extra_topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](
        batch_size
    )
    for bi in range(batch_size):
        extra_topk_lengths_host[bi] = Int32(extra_topk_per_batch[bi])
    var extra_topk_lengths_device = ctx.enqueue_create_buffer[.int32](
        batch_size
    )
    ctx.enqueue_copy(extra_topk_lengths_device, extra_topk_lengths_host)

    ctx.synchronize()

    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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
    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
        LayoutTensor[kv_type, Layout.row_major[6]()](
            blocks_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[Layout.row_major[6]()](
                blocks_lt.runtime_layout.shape.value,
                blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, cl_layout](
            cache_lengths_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, lt_layout_2d](
            lookup_table_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var extra_blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
        extra_blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(extra_block_shape),
    )
    var extra_cache_lengths_lt = LayoutTensor[.uint32, cl_layout](
        extra_cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )
    var extra_lookup_table_lt = LayoutTensor[.uint32, lt_layout_2d](
        extra_lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, max_extra_pages_per_batch)
        ),
    )
    var extra_kv_collection = PagedKVCacheCollection[
        kv_type, kv_params, PAGE_SIZE
    ](
        LayoutTensor[kv_type, Layout.row_major[6]()](
            extra_blocks_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[Layout.row_major[6]()](
                extra_blocks_lt.runtime_layout.shape.value,
                extra_blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, cl_layout](
            extra_cache_lengths_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[cl_layout](
                extra_cache_lengths_lt.runtime_layout.shape.value,
                extra_cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, lt_layout_2d](
            extra_lookup_table_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[lt_layout_2d](
                extra_lookup_table_lt.runtime_layout.shape.value,
                extra_lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(q_max_seq_len),
        UInt32(max_extra_cache_len),
    )
    var extra_kv_cache = extra_kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[row_depth])),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(batch_size + 1),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    print("  Launching MLA sparse QKV_FP8 (extra KV)...")

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[.float8_e4m3fn](num_heads, row_depth),
        ragged=True,
        sparse=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf_tt,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=max_topk,
        topk_lengths=rebind[MutPointer[Int32, MutAnyOrigin]](
            topk_lengths_device.unsafe_ptr()
        ),
        extra_k=extra_kv_cache,
        extra_d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            extra_d_indices_device.unsafe_ptr()
        ),
        extra_indices_stride=max_extra_topk,
        extra_topk_lengths=rebind[MutPointer[Int32, MutAnyOrigin]](
            extra_topk_lengths_device.unsafe_ptr()
        ),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    verify_mla_output(
        name,
        ref_host.unsafe_ptr(),
        out_host.unsafe_ptr(),
        batch_size * num_heads * V_DEPTH,
    )

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = extra_blocks_device
    _ = extra_cache_lengths_device
    _ = extra_lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = extra_d_indices_device
    _ = topk_lengths_device
    _ = extra_topk_lengths_device
    _ = row_offsets_device


# ===-----------------------------------------------------------------------===#
# Topk-clamping sparse test (small caches with topk > actual_tokens)
# ===-----------------------------------------------------------------------===#


def run_test_sparse_qkv_fp8_topk_clamping[
    kv_type: DType,
    num_heads: Int,
    # Width of the KV row and the absorbed Q as the model stores them. 576
    # carries a rotary tail after the 512 latent; a NoPE model stores the
    # latent alone and the kernel takes that row directly.
    row_depth: Int = Q_DEPTH,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    ctx: DeviceContext,
    tail_max: Float64 = 0.0015,
) raises:
    """Topk clamping with native FP8-Q + all-FP8-KV layout.

    When topk_per_batch[b] > actual_tokens[b], the kernel clamps to actual
    tokens. Indices beyond effective_topk are filled with -1 to catch OOB reads.
    """
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)

    var max_topk = 0
    var effective_topk_list = List[Int]()
    for bi in range(batch_size):
        if topk_per_batch[bi] > max_topk:
            max_topk = topk_per_batch[bi]
        var actual_tokens = cache_lengths[bi] + q_max_seq_len
        var eff_topk = topk_per_batch[bi]
        if eff_topk > actual_tokens:
            eff_topk = actual_tokens
        effective_topk_list.append(eff_topk)

    print("test:", name, " batch_size:", batch_size, " num_heads:", num_heads)
    for i in range(batch_size):
        print(
            "  batch",
            i,
            ": cache_len=",
            cache_lengths[i],
            " topk=",
            topk_per_batch[i],
            " actual_tokens=",
            cache_lengths[i] + q_max_seq_len,
            " effective_topk=",
            effective_topk_list[i],
        )

    var max_cache_len = 0
    var total_pages = 0
    var num_keys_list = List[Int]()
    for i in range(batch_size):
        var cl = cache_lengths[i]
        if cl > max_cache_len:
            max_cache_len = cl
        var nk = cl + q_max_seq_len
        num_keys_list.append(nk)
        total_pages += ceildiv(nk, PAGE_SIZE)

    var max_num_keys = max_cache_len + q_max_seq_len
    var max_pages_per_batch = ceildiv(max_num_keys, PAGE_SIZE)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=row_depth, is_mla=True
    )
    comptime kv_dim2 = 1

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var tok_stride = kv_params.head_size

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np_bi

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * row_depth
    var k_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](total_k_elems)
    randn(k_bf16_host.as_span(), mean=0.0, standard_deviation=0.5)
    var k_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](total_k_elems)

    var k_offset = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        for t in range(nk):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = k_offset + t * row_depth
            for d in range(row_depth):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[.bfloat16]()
        k_offset += nk * row_depth

    # Q: generate BF16 randn, cast to FP8.
    var q_size = batch_size * num_heads * row_depth
    var q_bf16_scratch = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_bf16_scratch.as_span(), mean=0.0, standard_deviation=0.5)
    var q_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    for i in range(q_size):
        q_host[i] = q_bf16_scratch[i].cast[.float8_e4m3fn]()

    # d_indices: first effective_topk entries valid; rest are -1 sentinel.
    var total_indices = batch_size * max_topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for i in range(total_indices):
        h_indices[i] = Int32(-1)

    var total_sparse_ref_elems = 0
    for bi in range(batch_size):
        total_sparse_ref_elems += effective_topk_list[bi] * row_depth
    var k_sparse_ref = ctx.enqueue_create_host_buffer[.bfloat16](
        total_sparse_ref_elems
    )

    var sparse_ref_offset = 0
    var k_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var eff_topk = effective_topk_list[bi]
        var mult = _coprime_multiplier(nk) if nk > 1 else 1
        for i in range(eff_topk):
            var t: Int
            if nk == 1:
                t = 0
            else:
                t = (i * mult + 1) % nk
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * max_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = k_offset_src + t * row_depth
            var dst_base = sparse_ref_offset + i * row_depth
            for d in range(row_depth):
                k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]
        sparse_ref_offset += eff_topk * row_depth
        k_offset_src += nk * row_depth

    var sparse_num_keys_list = List[Int]()
    for bi in range(batch_size):
        sparse_num_keys_list.append(effective_topk_list[bi])

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    host_reference_varkeys(
        q_host.unsafe_ptr(),
        k_sparse_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        batch_size,
        num_heads,
        sparse_num_keys_list,
        row_depth,
        V_DEPTH,
        scale,
    )

    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    var topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])  # UNCLAMPED
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    ctx.synchronize()

    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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
    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
        LayoutTensor[kv_type, Layout.row_major[6]()](
            blocks_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks_lt.runtime_layout.shape.value,
                blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, cl_layout, _](
            cache_lengths_lt.ptr,
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, lt_layout_2d, _](
            lookup_table_lt.ptr,
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[row_depth])),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(batch_size + 1),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    print("  Launching MLA sparse QKV_FP8 (topk clamping)...")

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[.float8_e4m3fn](num_heads, row_depth),
        ragged=True,
        sparse=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf_tt,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=max_topk,
        topk_lengths=rebind[MutPointer[Int32, MutAnyOrigin]](
            topk_lengths_device.unsafe_ptr()
        ),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    verify_mla_output(
        name,
        ref_host.unsafe_ptr(),
        out_host.unsafe_ptr(),
        batch_size * num_heads * V_DEPTH,
        tail_max=tail_max,
    )

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = row_offsets_device


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            seed(42)

            # ===============================================================
            # NoPE native 512 on the production (native-FP8) decode backend.
            # ---------------------------------------------------------------
            # Breadth comes from differentials against the zero-tailed 576 run
            # rather than a host reference: both arms are GPU runs, so the
            # production shapes are affordable and the comparison is bit-exact.
            # ===============================================================
            run_nope_native_512_fp8[64](ctx, batch_size=1, cache_len=256)
            run_nope_native_512_fp8[64](ctx, batch_size=4, cache_len=2048)

            # Primes and the page-size boundary (PAGE_SIZE=128).
            run_nope_native_512_fp8[64](
                ctx, batch_size=1, cache_len=2, name="prime_cl2"
            )
            run_nope_native_512_fp8[64](
                ctx, batch_size=1, cache_len=7, name="prime_cl7"
            )
            run_nope_native_512_fp8[64](
                ctx, batch_size=3, cache_len=13, name="prime_cl13"
            )
            run_nope_native_512_fp8[64](
                ctx, batch_size=1, cache_len=97, name="prime_cl97"
            )
            run_nope_native_512_fp8[64](
                ctx, batch_size=1, cache_len=127, name="page_cl127"
            )
            run_nope_native_512_fp8[64](
                ctx, batch_size=1, cache_len=128, name="page_cl128"
            )
            run_nope_native_512_fp8[64](
                ctx, batch_size=1, cache_len=129, name="page_cl129"
            )
            run_nope_native_512_fp8[64](
                ctx, batch_size=2, cache_len=257, topk=64, name="prime_cl257"
            )

            # Production selection width, long context, and batching.
            run_nope_native_512_fp8[64](
                ctx,
                batch_size=1,
                cache_len=2048,
                topk=2048,
                name="prod_topk2048_saturating",
            )
            run_nope_native_512_fp8[64](
                ctx,
                batch_size=1,
                cache_len=8192,
                topk=2048,
                name="prod_cl8192_topk2048",
            )
            run_nope_native_512_fp8[64](
                ctx,
                batch_size=4,
                cache_len=4096,
                topk=2048,
                name="prod_b4_cl4096_topk2048",
            )
            # The real GLM decode step: TP=8 -> 8 heads, MTP=5 -> q_len 6.
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=1,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=6,
                name="prod_tp8_mtp6_topk2048",
            )
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=4,
                cache_len=2048,
                topk=2048,
                q_max_seq_len=6,
                name="prod_tp8_mtp6_b4",
            )

            # Production decode batch sweep at the deployed shape. Batch moves
            # the decode grid and the split-K decision, so the narrow row has
            # to hold across it, not only at the one batch the mechanism was
            # first proved on.
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=8,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=6,
                name="prod_tp8_mtp6_b8",
            )
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=16,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=6,
                name="prod_tp8_mtp6_b16",
            )
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=24,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=6,
                name="prod_tp8_mtp6_b24",
            )
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=32,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=6,
                name="prod_tp8_mtp6_b32",
            )
            # The selection width is not a round 2048. Pooling by 4 selects 512
            # pools of four contiguous tokens and always appends a tail of up to
            # three, so the row the kernel sees is 2051 wide; an engine that
            # tiles its index by 128 columns rounds that to 2176. Neither is a
            # multiple of the gather tile, so both run the remainder path at
            # production scale rather than at the toy sizes the prime cases use.
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=8,
                cache_len=8192,
                topk=2051,
                q_max_seq_len=6,
                name="prod_kpool_tail_topk2051",
            )
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=8,
                cache_len=8192,
                topk=2176,
                q_max_seq_len=6,
                name="prod_engine_padded_topk2176",
            )
            # Speculative depth sweep. The deployment recipe is MTP=5 (q_len 6)
            # but the engine varies draft depth at runtime and falls back to
            # q_len 1 with speculation off.
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=8,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=1,
                name="prod_tp8_qlen1_b8",
            )
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=8,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=2,
                name="prod_tp8_qlen2_b8",
            )
            run_nope_native_512_fp8[8](
                ctx,
                batch_size=8,
                cache_len=4096,
                topk=2048,
                q_max_seq_len=4,
                name="prod_tp8_qlen4_b8",
            )
            # Multi-token Q at smaller scale, and prime-on-prime.
            run_nope_native_512_fp8[64](
                ctx,
                batch_size=2,
                cache_len=512,
                topk=64,
                q_max_seq_len=4,
                name="qlen4_b2",
            )
            run_nope_native_512_fp8[64](
                ctx,
                batch_size=2,
                cache_len=1021,
                topk=509,
                name="prime_cl1021_topk509",
            )

            # =====================================================
            # Read-once shared-index MTP fold (KERN-3141), shared_index=True.
            # Exercises the fold branch with native FP8-Q.
            # =====================================================

            # NullMask, single tile.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=True](
                "shared_fold_b1_h8_cl256_topk64_seq6",
                1,
                256,
                ctx,
                topk=64,
                q_max_seq_len=6,
            )
            # Multi-tile shared list.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=True](
                "shared_fold_b2_h8_cl1024_topk160_seq6",
                2,
                1024,
                ctx,
                topk=160,
                q_max_seq_len=6,
            )
            # CausalMask fold.
            run_test_sparse_qkv_fp8[
                DType.float8_e4m3fn, 8, use_causal=True, shared_index=True
            ](
                "shared_fold_causal_b1_h8_cl64_topk70_seq6",
                1,
                64,
                ctx,
                topk=70,
                q_max_seq_len=6,
            )
            # Order audit: reversed gather order — NullMask output must be
            # invariant to gather order.
            run_test_sparse_qkv_fp8[
                DType.float8_e4m3fn, 8, shared_index=True, order_mode=1
            ](
                "shared_fold_null_reversed_b2_h8_cl256_topk64_seq6",
                2,
                256,
                ctx,
                topk=64,
                q_max_seq_len=6,
            )
            # Coprime-permuted gather order.
            run_test_sparse_qkv_fp8[
                DType.float8_e4m3fn, 8, shared_index=True, order_mode=2
            ](
                "shared_fold_null_permuted_b2_h8_cl512_topk160_seq6",
                2,
                512,
                ctx,
                topk=160,
                q_max_seq_len=6,
            )
            # Negative: per-position distinct sets (fold not selected).
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=False](
                "neg_distinct_sets_unfolded_b2_h8_cl512_topk64_seq6",
                2,
                512,
                ctx,
                topk=64,
                q_max_seq_len=6,
            )
            # Production shape: bs8, cache=2048, topk=2048.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 8, shared_index=True](
                "shared_fold_prod_b8_h8_cl2048_topk2048_seq6",
                8,
                2048,
                ctx,
                topk=2048,
                q_max_seq_len=6,
            )
            # seq_len=0: empty last batch under the fold.
            run_test_sparse_qkv_fp8[
                DType.float8_e4m3fn, 8, shared_index=True, empty_last_batch=True
            ](
                "shared_fold_seqlen0_b2_h8_cl256_topk64_seq6",
                2,
                256,
                ctx,
                topk=64,
                q_max_seq_len=6,
            )

            # =====================================================
            # Base variants: NullMask, no feature flags.
            # =====================================================

            # Minimal config: bs=1, h=16, cl=128, topk=8.
            # topk=8 is few enough keys that FP8-P softmax quantization pushes a
            # ~1% error tail past 1e-2 (cosine stays > 0.999); loosen tail bound.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_min_b1_h16_cl128_topk8",
                1,
                128,
                ctx,
                topk=8,
                tail_max=0.03,
            )

            # bs=1, h=16, cl=512, topk=64.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_b1_h16_cl512_topk64",
                1,
                512,
                ctx,
                topk=64,
            )

            # Production-ish: bs=1, h=128, cl=512, topk=64.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 128](
                "sparse_qkv_fp8_b1_h128_cl512_topk64",
                1,
                512,
                ctx,
                topk=64,
            )

            # Larger cache with split-K: bs=1, h=128, cl=2048, topk=64.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 128](
                "sparse_qkv_fp8_b1_h128_cl2048_topk64",
                1,
                2048,
                ctx,
                topk=64,
            )

            # =====================================================
            # Multi-batch (batch_size > 1).
            # =====================================================

            # bs=2, h=16, cl=128, topk=64.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_b2_h16_cl128_topk64",
                2,
                128,
                ctx,
                topk=64,
            )

            # bs=4, h=16, cl=256, topk=64.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_b4_h16_cl256_topk64",
                4,
                256,
                ctx,
                topk=64,
            )

            # =====================================================
            # Multi-token Q (speculative decode).
            # =====================================================

            # q_max_seq_len=2.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_b1_h16_cl256_topk64_seq2",
                1,
                256,
                ctx,
                topk=64,
                q_max_seq_len=2,
            )

            # q_max_seq_len=4.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_b1_h16_cl256_topk64_seq4",
                1,
                256,
                ctx,
                topk=64,
                q_max_seq_len=4,
            )

            # q_max_seq_len=8, multi-batch.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_b2_h16_cl256_topk64_seq8",
                2,
                256,
                ctx,
                topk=64,
                q_max_seq_len=8,
            )

            # =====================================================
            # CausalMask variant.
            # =====================================================

            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16, use_causal=True](
                "sparse_qkv_fp8_causal_b1_h16_cl256_topk64",
                1,
                256,
                ctx,
                topk=64,
            )

            # Multi-token Q with genuine sparsity (topk << num_keys) so the
            # score-sorted selection is not logical-position-sorted: exercises
            # causal masking by logical position rather than gather-slot order.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16, use_causal=True](
                "sparse_qkv_fp8_causal_multitoken_b2_h16_cl256_topk64_seq8",
                2,
                256,
                ctx,
                topk=64,
                q_max_seq_len=8,
            )

            # The logical-position mask reads its slots in batches of 8 and
            # picks up the remainder one at a time. A thread owns 32 slots, so
            # the cases above (topk a multiple of 32) only ever run whole
            # batches, and the topk=70 case above only ever runs the remainder:
            # the tail's shift base is carried out of the batch loop and no
            # shape yet reaches it with a nonzero carry. These land the
            # remainder at 3, 7 and 1 slots after a whole batch. Prime cache
            # lengths and q also keep the paged gather off its tile boundaries.
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16, use_causal=True](
                "sparse_qkv_fp8_causal_b1_h16_cl251_topk43",
                1,
                251,
                ctx,
                topk=43,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16, use_causal=True](
                "sparse_qkv_fp8_causal_multitoken_b2_h16_cl257_topk79_seq5",
                2,
                257,
                ctx,
                topk=79,
                q_max_seq_len=5,
            )
            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16, use_causal=True](
                "sparse_qkv_fp8_causal_multitoken_b3_h16_cl509_topk73_seq3",
                3,
                509,
                ctx,
                topk=73,
                q_max_seq_len=3,
            )

            # =====================================================
            # Variable per-batch topk (has_variable_topk=True).
            # =====================================================

            var vt_cls_2: List[Int] = [256, 128]
            var vt_topk_2: List[Int] = [64, 32]
            run_test_sparse_qkv_fp8_variable_topk[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_variable_topk_b2_h16",
                vt_cls_2,
                vt_topk_2,
                ctx,
            )

            var vt_cls_4: List[Int] = [256, 384, 128, 512]
            var vt_topk_4: List[Int] = [64, 128, 32, 64]
            run_test_sparse_qkv_fp8_variable_topk[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_variable_topk_b4_h16",
                vt_cls_4,
                vt_topk_4,
                ctx,
            )

            # =====================================================
            # Attention sink (has_attn_sink=True).
            # =====================================================

            run_test_sparse_qkv_fp8_attn_sink[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_attn_sink_b1_h16_cl128_topk64",
                1,
                128,
                ctx,
                topk=64,
            )

            run_test_sparse_qkv_fp8_attn_sink[.float8_e4m3fn, 128](
                "sparse_qkv_fp8_attn_sink_b1_h128_cl512_topk64",
                1,
                512,
                ctx,
                topk=64,
            )

            run_test_sparse_qkv_fp8_attn_sink[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_attn_sink_b4_h16_cl256_topk64",
                4,
                256,
                ctx,
                topk=64,
            )

            # =====================================================
            # Extra KV (has_extra_kv=True).
            # =====================================================

            var ek_cls_1: List[Int] = [256]
            var ek_topk_1: List[Int] = [64]
            var ek_ecls_1: List[Int] = [64]
            var ek_etopk_1: List[Int] = [64]
            run_test_sparse_qkv_fp8_extra_kv[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_extra_kv_b1_h16_topk64_extra64",
                ek_cls_1,
                ek_topk_1,
                ek_ecls_1,
                ek_etopk_1,
                ctx,
            )

            var ek_cls_2: List[Int] = [256, 384]
            var ek_topk_2: List[Int] = [64, 64]
            var ek_ecls_2: List[Int] = [64, 128]
            var ek_etopk_2: List[Int] = [64, 64]
            run_test_sparse_qkv_fp8_extra_kv[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_extra_kv_b2_h16_variable",
                ek_cls_2,
                ek_topk_2,
                ek_ecls_2,
                ek_etopk_2,
                ctx,
            )

            # =====================================================
            # Topk clamping (small caches).
            # =====================================================

            # First decode: cache_length=0, actual=1, topk=64 -> clamp to 1.
            var tc_cls_1: List[Int] = [0]
            var tc_topk_1: List[Int] = [64]
            run_test_sparse_qkv_fp8_topk_clamping[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_topk_clamp_first_exec_b1_h16",
                tc_cls_1,
                tc_topk_1,
                ctx,
            )

            # Small cache: cache_length=5, topk=64 -> clamp to 6.
            # Only 6 keys after clamping: FP8-P softmax quantization gives a ~3%
            # error tail past 1e-2 (cosine stays > 0.999); loosen tail bound.
            var tc_cls_2: List[Int] = [5]
            var tc_topk_2: List[Int] = [64]
            run_test_sparse_qkv_fp8_topk_clamping[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_topk_clamp_small_cache_b1_h16",
                tc_cls_2,
                tc_topk_2,
                ctx,
                tail_max=0.05,
            )

            # Mixed batch: cl=0 and cl=256.
            var tc_cls_3: List[Int] = [0, 256]
            var tc_topk_3: List[Int] = [64, 64]
            run_test_sparse_qkv_fp8_topk_clamping[.float8_e4m3fn, 16](
                "sparse_qkv_fp8_topk_clamp_mixed_b2_h16",
                tc_cls_3,
                tc_topk_3,
                ctx,
            )

            # =====================================================
            # NoPE native 512, the ragged and edge-case families.
            # ---------------------------------------------------------------
            # The differential above runs the narrow row at production scale,
            # but only on shapes that are uniform across the batch. These
            # re-run the ragged families -- per-batch topk and cache length,
            # attention sink, a second always-attend cache, and clamping on a
            # near-empty cache -- at 512 against the same host reference the
            # 576 row is held to, so the narrow row clears the whole bar and
            # not a friendly subset of it.
            #
            # The extra-KV case is the one that answers a specific question:
            # the tail is zeroed once at kernel entry, so a second cache that
            # wrote the tail afterwards would break that. It does not.
            # =====================================================

            run_test_sparse_qkv_fp8[.float8_e4m3fn, 16, row_depth=512](
                "nope512_qkv_fp8_b4_h16_cl256_topk64",
                4,
                256,
                ctx,
                topk=64,
            )
            run_test_sparse_qkv_fp8[
                .float8_e4m3fn, 16, use_causal=True, row_depth=512
            ](
                "nope512_qkv_fp8_causal_b3_h16_cl509_topk73_seq3",
                3,
                509,
                ctx,
                topk=73,
                q_max_seq_len=3,
            )

            var n512_vt_cls: List[Int] = [256, 384, 128, 512]
            var n512_vt_topk: List[Int] = [64, 128, 32, 64]
            run_test_sparse_qkv_fp8_variable_topk[
                .float8_e4m3fn, 16, row_depth=512
            ](
                "nope512_qkv_fp8_variable_topk_b4_h16",
                n512_vt_cls,
                n512_vt_topk,
                ctx,
            )

            run_test_sparse_qkv_fp8_attn_sink[
                .float8_e4m3fn, 16, row_depth=512
            ](
                "nope512_qkv_fp8_attn_sink_b4_h16_cl256_topk64",
                4,
                256,
                ctx,
                topk=64,
            )

            var n512_ek_cls: List[Int] = [256, 384]
            var n512_ek_topk: List[Int] = [64, 64]
            var n512_ek_ecls: List[Int] = [64, 128]
            var n512_ek_etopk: List[Int] = [64, 64]
            run_test_sparse_qkv_fp8_extra_kv[.float8_e4m3fn, 16, row_depth=512](
                "nope512_qkv_fp8_extra_kv_b2_h16_variable",
                n512_ek_cls,
                n512_ek_topk,
                n512_ek_ecls,
                n512_ek_etopk,
                ctx,
            )

            var n512_tc_cls: List[Int] = [0, 256]
            var n512_tc_topk: List[Int] = [64, 64]
            run_test_sparse_qkv_fp8_topk_clamping[
                .float8_e4m3fn, 16, row_depth=512
            ](
                "nope512_qkv_fp8_topk_clamp_mixed_b2_h16",
                n512_tc_cls,
                n512_tc_topk,
                ctx,
            )
