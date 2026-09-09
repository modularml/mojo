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
"""End-to-end correctness test for the Apple M5 MMA prefill kernels on PAGED KV.

Apple silicon GPU (Metal 4, `compute_capability == 5`) only.

Validates the paged-KV path of BOTH prefill kernels: a `PagedKVCacheCollection`
(`page_size > 0`) wrapped in `KVCacheMHAOperand`, run through the ragged prefill
launcher, compared against an **independent fp32 host attention reference**
(NOT a paged-vs-continuous self-comparison -- per
`Kernels/claude_kb/entries/patterns/amd-paged-vs-continuous-kv-test-not-independent.md`,
running the same kernel two ways shares any arithmetic bug; an fp32 host
reference catches shared-implementation bugs too).

`_run` drives the single `fa_prefill_apple` kernel (wide-threadgroup no-SMEM,
`num_simdgroups=16`) on paged KV against the independent fp32 reference.

The paged crux: a 16-row MMA sub-tile must not cross a page boundary. The kernel
resolves the page per 16-row sub-tile, valid iff `page_size % 16 == 0`. Coverage
exercises that contract directly: small page sizes (16, 32) so a single `Sk=32`
KV tile spans MULTIPLE pages, multiple pages per sequence, and a partial last
page (`num_keys` not a multiple of `page_size`). Plus ragged (mixed seq lens),
GQA, NullMask/CausalMask/SlidingWindowCausalMask, fp16/bf16, and depth 64/128.
"""

from std.collections import OptionalReg
from max.gpu.host import DeviceContext
from std.math import ceildiv, exp, sqrt
from std.memory import unsafe_memset_zero
from std.random import seed, shuffle
from std.sys import has_apple_gpu_accelerator

from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection

from layout import (
    UNKNOWN_VALUE,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
)
from layout._utils import ManagedLayoutTensor
from layout.tile_layout import row_major

from nn.attention.gpu.apple.fa_prefill import (
    fa_prefill_apple,
)
from nn.attention.mha_mask import (
    CausalMask,
    MHAMask,
    NullMask,
    SlidingWindowCausalMask,
)
from nn.attention.mha_operand import KVCacheMHAOperand

from std.utils import Index, IndexList


# Inlined from `test/gpu/kv_cache/kv_cache_test_utils.mojo` to keep this Apple
# test hermetic (no cross-dir source include in the BUILD rule).
comptime _LUT_TAIL_PAD = 16


def _padded_lut_cols(cols: Int) -> Int:
    """LUT row stride padded to a multiple of 8, >= cols + 15 (matches
    `PagedKVCache.populate`'s SIMD-chunk alignment). See `padded_lut_cols`.
    """
    return ((cols + 7) // 8) * 8 + _LUT_TAIL_PAD


def _random_distinct(n: Int, k: Int) -> List[Int]:
    """Sample `k` distinct integers from `[0, n)` (prefix of a shuffle)."""
    var perm: List[Int] = [i for i in range(n)]
    shuffle(perm)
    perm.shrink(k)
    return perm^


# ===-------------------------------------------------------------------=== #
# fp32 host attention reference (ragged: per-sequence seq_len == num_keys,
# cache_length == 0). Independent of the kernel -- this is the ground truth.
# ===-------------------------------------------------------------------=== #
def _host_attention_ragged[
    mask_kind: Int,  # 0 = null, 1 = causal, 2 = sliding-window
    window: Int = 0,
](
    q: List[Float32],  # ragged [total_tokens, num_heads, depth]
    k: List[Float32],  # ragged [total_tokens, kv_heads, depth] (per-batch keys)
    v: List[Float32],
    row_offsets: List[Int],  # length batch+1
    out_zero: List[Float32],
    num_heads: Int,
    kv_heads: Int,
    depth: Int,
    scale: Float32,
) -> List[Float32]:
    var result = out_zero.copy()
    var group = num_heads // kv_heads
    var batch = len(row_offsets) - 1
    for b in range(batch):
        var seq_start = row_offsets[b]
        var seq_len = row_offsets[b + 1] - seq_start
        # cache_length == 0 here, so num_keys == seq_len for this sequence.
        var num_keys = seq_len
        for h in range(num_heads):
            var kvh = h // group
            for qi in range(seq_len):
                var scores = List[Float32](length=num_keys, fill=0)
                var m = Float32(-3.0e38)
                for ki in range(num_keys):
                    var dot = Float32(0)
                    for d in range(depth):
                        var qv = q[
                            ((seq_start + qi) * num_heads + h) * depth + d
                        ]
                        var kv = k[
                            ((seq_start + ki) * kv_heads + kvh) * depth + d
                        ]
                        dot += qv * kv
                    var s = dot * scale
                    var visible = True
                    comptime if mask_kind == 1:
                        visible = ki <= qi
                    elif mask_kind == 2:
                        visible = ki <= qi and ki > qi - window
                    if not visible:
                        s = Float32(-3.0e38)
                    scores[ki] = s
                    m = max(m, s)
                var l = Float32(0)
                for ki in range(num_keys):
                    scores[ki] = exp(scores[ki] - m)
                    l += scores[ki]
                for d in range(depth):
                    var acc = Float32(0)
                    for ki in range(num_keys):
                        acc += (
                            scores[ki]
                            * v[((seq_start + ki) * kv_heads + kvh) * depth + d]
                        )
                    var denom = l if l > 0 else Float32(1)
                    result[((seq_start + qi) * num_heads + h) * depth + d] = (
                        acc / denom
                    )
    return result^


def _run[
    qkv_type: DType,
    page_size: Int,
    num_q_heads: Int,
    kv_params: KVCacheStaticParams,
    mask_t: MHAMask,
    mask_kind: Int,
    window: Int = 0,
](mask: mask_t, seq_lens: List[Int], ctx: DeviceContext,) raises:
    comptime depth = kv_params.head_size
    comptime kv_heads = kv_params.num_heads
    comptime group = num_q_heads // kv_heads
    comptime num_layers = 1
    comptime layer_idx = 0

    var batch_size = len(seq_lens)
    var scale = Float32(1.0) / sqrt(Float32(depth))

    # cache_length == 0 (pure prefill): each sequence attends exactly its own
    # seq_len keys.
    var total_length = 0
    var max_full_context = 0
    var max_prompt_len = 0
    for i in range(batch_size):
        total_length += seq_lens[i]
        max_full_context = max(max_full_context, seq_lens[i])
        max_prompt_len = max(max_prompt_len, seq_lens[i])

    print(
        "  paged",
        "dtype:",
        qkv_type,
        "page_size:",
        page_size,
        "depth:",
        depth,
        "q_heads:",
        num_q_heads,
        "kv_heads:",
        kv_heads,
        "batch:",
        batch_size,
        "total_tokens:",
        total_length,
        "mask_kind:",
        mask_kind,
    )

    # ---- ragged Q + row offsets ---------------------------------------- #
    var q_n = total_length * num_q_heads * depth
    var q_f = List[Float32](length=q_n, fill=0)
    var k_n = total_length * kv_heads * depth  # per-token keys (ragged)
    var k_f = List[Float32](length=k_n, fill=0)
    var v_f = List[Float32](length=k_n, fill=0)

    # Deterministic host master data.
    for i in range(q_n):
        q_f[i] = Float32(((i * 53 + 17) % 197) - 98) * 0.01
    for i in range(k_n):
        k_f[i] = Float32(((i * 31 + 7) % 211) - 105) * 0.01
        v_f[i] = Float32(((i * 91 + 13) % 173) - 86) * 0.01

    var row_offsets = List[Int]()
    var acc_off = 0
    for i in range(batch_size + 1):
        row_offsets.append(acc_off)
        if i < batch_size:
            acc_off += seq_lens[i]

    var ref_out = _host_attention_ragged[mask_kind, window](
        q_f,
        k_f,
        v_f,
        row_offsets,
        List[Float32](length=q_n, fill=0),
        num_q_heads,
        kv_heads,
        depth,
        scale,
    )

    # ---- ragged Q device tensor [total_tokens, num_q_heads, depth] ----- #
    comptime q_layout = Layout.row_major(UNKNOWN_VALUE, num_q_heads, depth)
    var q_managed = ManagedLayoutTensor[qkv_type, q_layout](
        RuntimeLayout[q_layout].row_major(
            IndexList[3](total_length, num_q_heads, depth)
        ),
        ctx,
    )
    var q_host = q_managed.tensor[update=False]()
    for t in range(total_length):
        for h in range(num_q_heads):
            for d in range(depth):
                q_host[t, h, d] = Scalar[qkv_type](
                    q_f[(t * num_q_heads + h) * depth + d]
                )

    # ---- ragged output tensor ------------------------------------------ #
    var o_managed = ManagedLayoutTensor[qkv_type, q_layout](
        RuntimeLayout[q_layout].row_major(
            IndexList[3](total_length, num_q_heads, depth)
        ),
        ctx,
    )

    # ---- input_row_offsets [batch+1] ----------------------------------- #
    comptime ro_layout = Layout(UNKNOWN_VALUE)
    var ro_managed = ManagedLayoutTensor[.uint32, ro_layout](
        RuntimeLayout[ro_layout].row_major(IndexList[1](batch_size + 1)),
        ctx,
    )
    var ro_host = ro_managed.tensor[update=False]()
    for i in range(batch_size + 1):
        ro_host[i] = UInt32(row_offsets[i])

    # ---- per-sequence cache_lengths (all 0: pure prefill) -------------- #
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cl_managed = ManagedLayoutTensor[.uint32, cl_layout](
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cl_host = cl_managed.tensor[update=False]()
    for i in range(batch_size):
        cl_host[i] = UInt32(0)

    # ---- paged KV blocks [num_pages, 2, num_layers, page_size, kv_heads,
    #      depth] + LUT [batch, padded_pages] ---------------------------- #
    var num_pages_per_batch = ceildiv(max_full_context, page_size)
    var total_pages = 0
    for i in range(batch_size):
        total_pages += ceildiv(seq_lens[i], page_size)
    # Pad block pool so distinct-block sampling has slack.
    var num_paged_blocks = total_pages + batch_size + 2

    comptime kv_block_layout = Layout.row_major[6]()
    var kv_block_shape = IndexList[6](
        num_paged_blocks, 2, num_layers, page_size, kv_heads, depth
    )
    var kv_block_managed = ManagedLayoutTensor[qkv_type, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(kv_block_shape), ctx
    )
    var kv_block_host = kv_block_managed.tensor[update=False]()
    # Zero-fill so OOB tail slots in a partial last page contribute nothing.
    var kv_block_elems = (
        num_paged_blocks * 2 * num_layers * page_size * kv_heads * depth
    )
    unsafe_memset_zero(kv_block_host.ptr, kv_block_elems)

    comptime lut_layout = Layout.row_major[2]()
    var max_pages = _padded_lut_cols(num_pages_per_batch)
    var lut_managed = ManagedLayoutTensor[.uint32, lut_layout](
        RuntimeLayout[lut_layout].row_major(
            IndexList[2](batch_size, max_pages)
        ),
        ctx,
    )
    var lut_host = lut_managed.tensor[update=False]()
    for i in range(batch_size):
        for p in range(max_pages):
            lut_host[i, p] = UInt32(0)

    # Assign distinct physical pages per (batch, page) and scatter the host K/V
    # master data into them. This is the SAME data the fp32 reference saw, just
    # physically permuted across pages -- so a page-indexing bug shows up.
    var pages = _random_distinct(num_paged_blocks, total_pages)
    var page_pos = 0
    # Per (batch, kv_idx) 6D strides.
    var s_kvidx = num_layers * page_size * kv_heads * depth
    var s_layer = page_size * kv_heads * depth
    var s_block = 2 * num_layers * page_size * kv_heads * depth
    var s_tok = kv_heads * depth
    for b in range(batch_size):
        var seq_start = row_offsets[b]
        var n_pages_b = ceildiv(seq_lens[b], page_size)
        for pblk in range(n_pages_b):
            var phys = pages[page_pos]
            page_pos += 1
            lut_host[b, pblk] = UInt32(phys)
            var n_in_page = min(page_size, seq_lens[b] - pblk * page_size)
            for tip in range(n_in_page):
                var global_tok = seq_start + pblk * page_size + tip
                for kvh in range(kv_heads):
                    for d in range(depth):
                        var src = (global_tok * kv_heads + kvh) * depth + d
                        kv_block_host[phys, 0, layer_idx, tip, kvh, d] = Scalar[
                            qkv_type
                        ](k_f[src])
                        kv_block_host[phys, 1, layer_idx, tip, kvh, d] = Scalar[
                            qkv_type
                        ](v_f[src])

    # ---- build the paged collection + operands ------------------------- #
    var kv_collection = PagedKVCacheCollection[qkv_type, kv_params, page_size](
        kv_block_managed.device_tensor().as_unsafe_any_origin(),
        cl_managed.device_tensor(),
        lut_managed.device_tensor(),
        UInt32(max_prompt_len),
        UInt32(max_full_context),
    )
    var k_cache = kv_collection.get_key_cache(layer_idx)
    var v_cache = kv_collection.get_value_cache(layer_idx)
    var k_op = KVCacheMHAOperand(k_cache)
    var v_op = KVCacheMHAOperand(v_cache)

    # ---- run the paged prefill kernel (ragged path, cache_length==0). No
    # sink (sink_weights defaults to None on both launchers). ------------ #
    fa_prefill_apple[
        ragged=True,
        sink=False,
        _use_valid_length=False,
        _is_cache_length_accurate=False,
    ](
        q_managed.device_tensor(),
        k_op,
        v_op,
        mask,
        o_managed.device_tensor(),
        ro_managed.device_tensor(),
        scale,
        batch_size,
        max_prompt_len,
        max_full_context,
        num_q_heads,
        depth,
        group,
        ctx,
    )
    ctx.synchronize()

    # ---- compare against the independent fp32 reference ---------------- #
    var o_out = o_managed.tensor()
    var atol = Float32(2e-2) if qkv_type == DType.bfloat16 else Float32(8e-3)
    var pass_ = True
    var max_err = Float32(0)
    for t in range(total_length):
        for h in range(num_q_heads):
            for d in range(depth):
                var got = o_out[t, h, d].cast[.float32]()[0]
                var exp_v = ref_out[(t * num_q_heads + h) * depth + d]
                var err = abs(got - exp_v)
                max_err = max(max_err, err)
                if err > atol * (1.0 + abs(exp_v)):
                    if pass_:
                        print(
                            "    FAIL t",
                            t,
                            "h",
                            h,
                            "d",
                            d,
                            "got",
                            got,
                            "exp",
                            exp_v,
                            "err",
                            err,
                        )
                    pass_ = False
    _ = q_managed^
    _ = o_managed^
    _ = ro_managed^
    _ = cl_managed^
    _ = kv_block_managed^
    _ = lut_managed^
    if not pass_:
        raise Error("FAILED (max_err=", max_err, ")")
    print("    PASS (max_err=", max_err, ")")


comptime kv_d64_h4 = KVCacheStaticParams(num_heads=4, head_size=64)
comptime kv_d64_h2 = KVCacheStaticParams(num_heads=2, head_size=64)
comptime kv_d128_h2 = KVCacheStaticParams(num_heads=2, head_size=128)


def _cases(ctx: DeviceContext) raises:
    """All `fa_prefill_apple` paged cases vs the independent fp32 reference."""
    print("== test_apple_fa_prefill_paged (paged MMA prefill vs fp32 host ref)")

    # --- page_size 16: a single Sk=32 KV tile spans TWO pages. The crux. ---
    # NullMask, fp16 + bf16, seq spanning many pages.
    _run[.float16, 16, 4, kv_d64_h4, NullMask, 0](NullMask(), [48], ctx)
    _run[.bfloat16, 16, 4, kv_d64_h4, NullMask, 0](NullMask(), [48], ctx)
    # CausalMask with page_size 16, seq=48 (3 pages).
    _run[.float16, 16, 4, kv_d64_h4, CausalMask, 1](CausalMask(), [48], ctx)

    # --- page_size 32 == Sk: each KV tile maps to exactly one page. ---
    _run[.float16, 32, 4, kv_d64_h4, CausalMask, 1](CausalMask(), [64], ctx)

    # --- partial last page: num_keys not a multiple of page_size. ---
    # seq=37, page_size=16 -> pages of 16,16,5 (last page 5/16 valid).
    _run[.float16, 16, 4, kv_d64_h4, CausalMask, 1](CausalMask(), [37], ctx)
    _run[.bfloat16, 16, 2, kv_d64_h2, NullMask, 0](NullMask(), [29], ctx)

    # --- ragged: mixed sequence lengths across the batch, multiple pages. ---
    _run[.float16, 16, 4, kv_d64_h4, CausalMask, 1](
        CausalMask(), [20, 48, 35], ctx
    )
    _run[.bfloat16, 32, 2, kv_d64_h2, CausalMask, 1](
        CausalMask(), [33, 17], ctx
    )

    # --- GQA: num_q_heads > kv_heads, paged. ---
    _run[.float16, 16, 8, kv_d64_h2, CausalMask, 1](CausalMask(), [40], ctx)

    # --- SlidingWindowCausalMask, paged. ---
    _run[.float16, 16, 2, kv_d64_h2, SlidingWindowCausalMask[16], 2, 16](
        SlidingWindowCausalMask[16](), [48], ctx
    )

    # --- depth 128, paged page_size 32. ---
    _run[.float16, 32, 2, kv_d128_h2, CausalMask, 1](
        CausalMask(), [40, 33], ctx
    )

    print("== all paged prefill cases PASS")


def test_apple_fa_prefill_paged(ctx: DeviceContext) raises:
    _cases(ctx)


def main() raises:
    comptime if not has_apple_gpu_accelerator():
        print("SKIP: fa_prefill_apple paged targets Apple silicon GPUs only")
        return
    seed(42)
    with DeviceContext() as ctx:
        if ctx.compute_capability() != 5:
            print("SKIP: Apple M5 required (16x16 simdgroup MMA)")
            return
        test_apple_fa_prefill_paged(ctx)
