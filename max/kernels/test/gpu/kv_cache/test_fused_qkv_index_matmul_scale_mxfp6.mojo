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
"""CDNA4 MXFP6 sibling of `test_fused_qkv_index_matmul_scale_mxfp8.mojo`.

Asserts the dual-cache 5-way fused op
(`generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4` at
MXFP6) agrees with running the 3-way fused MXFP6 op twice over the
same weight, split at the QKV/IndexQK boundary:

  1. `[Wq|Wk|Wv]` -> MAIN cache (K/V) + Q output.
  2. `[Wiq|Wik]`  -> INDEX cache (IndexK) + IndexQ output.

This is the only coverage the MXFP6 5-way path has: no MXFP6 checkpoint
quantizes the indexer projections (they stay BF16, so the model-side gate
rejects the fusion and the op is unreachable end to end). Without this test the
path would be entirely unexercised.

Unlike the MXFP8 test this compares within a TOLERANCE band rather than
bit-exactly. Every case here does in fact come out bit-exact (`max_abs_diff` 0.0
throughout), but that is not guaranteed by construction the way it is on SM100:
`mxfp6_block_scaled_matmul_amd` picks its split-K factor from
`_pick_num_splits`, which is keyed on N, so the concatenated matmul (N=1920) and
the two split matmuls (N=1280 and N=640) are free to reduce K in a different
order and move the odd output by a bf16 ULP. The band is 2 ULP for that. It
still gates the property under test: a mis-routed column or a cache slot written
twice differs by O(magnitude), not by an ULP. `max_abs_diff` is printed either
way, so a shape that stops being bit-exact is visible rather than silently
absorbed.

Run:
  bt-mi355 //max/kernels/test/gpu/kv_cache:test_fused_qkv_index_matmul_scale_mxfp6
"""

from std.math import ceildiv, isclose
from std.random import random_ui64, seed

from max.gpu.host import DeviceContext
from std.memory import unsafe_memset_zero
from std.testing import assert_equal

from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    UNKNOWN_VALUE,
)
from layout._utils import ManagedLayoutTensor
from linalg.mx_format import MXFormat
from linalg.fp6_utils import MXFP6_SF_VECTOR_SIZE
from nn.kv_cache_ragged import (
    generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4,
)

from std.utils import IndexList

from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable

comptime DATA_DTYPE = DType.uint8  # packed FP6, four codes per three bytes
comptime SCALE_DTYPE = DType.float8_e8m0fnu
comptime OUT_DTYPE = DType.bfloat16
comptime KV_DTYPE = DType.bfloat16
comptime SF_VECTOR_SIZE = MXFP6_SF_VECTOR_SIZE  # 32
comptime FP6_FORMAT = MXFormat.FP6_E2M3

comptime BF16_2ULP = 1.0 / 128.0

comptime HEAD_SIZE = 128
comptime NUM_Q_HEADS = 8
comptime MAIN_KV_HEADS = 1
comptime NUM_INDEX_HEADS = 4

comptime main_kv_params = KVCacheStaticParams(
    num_heads=MAIN_KV_HEADS, head_size=HEAD_SIZE
)
comptime index_kv_params = KVCacheStaticParams(
    num_heads=1, head_size=HEAD_SIZE, is_mla=True
)


def _fill_random_u8(t: LayoutTensor[mut=True, ...], n: Int, lo: Int, hi: Int):
    """Fill `n` bytes with random values in `[lo, hi]`.

    Writes through a byte view so the same helper serves the packed-FP6 uint8
    operands and the E8M0 scales, whose dtype has no integer conversion.
    """
    var bytes = t.ptr.bitcast[UInt8]()
    for i in range(n):
        bytes[i] = UInt8(Int(random_ui64(UInt64(lo), UInt64(hi))))


def execute_dual_cache_fused_mxfp6[
    hidden: Int = 256,
    fmt: MXFormat = FP6_FORMAT,
    rtol: Float64 = BF16_2ULP,
    atol: Float64 = 0.0,
](
    prompt_lens: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
) raises:
    """Assert the 5-way MXFP6 fused output matches the two 3-way MXFP6 ops."""
    comptime assert hidden % 128 == 0, "hidden must be a multiple of 128"
    comptime K_BYTES = (hidden * 6) // 8
    comptime scale_K = hidden // SF_VECTOR_SIZE

    comptime q_dim = NUM_Q_HEADS * HEAD_SIZE  # 1024
    comptime kv_dim = MAIN_KV_HEADS * HEAD_SIZE  # 128
    comptime iq_dim = NUM_INDEX_HEADS * HEAD_SIZE  # 512
    comptime ik_dim = HEAD_SIZE  # 128 (single MLA latent K head)

    comptime qkv_n = q_dim + 2 * kv_dim  # 1280
    comptime idx_n = iq_dim + ik_dim  # 640
    comptime n_total = qkv_n + idx_n  # 1920

    comptime assert qkv_n % 64 == 0 and idx_n % 64 == 0
    comptime assert n_total % 64 == 0

    var batch_size = len(prompt_lens)
    var cache_sizes = List[Int]()
    for _ in range(batch_size):
        cache_sizes.append(0)

    comptime num_paged_blocks = 128
    comptime page_size = 128

    comptime MainCollection = PagedKVCacheCollection[
        KV_DTYPE, main_kv_params, page_size, ...
    ]
    comptime IndexCollection = PagedKVCacheCollection[
        KV_DTYPE, index_kv_params, page_size, ...
    ]

    var clt = CacheLengthsTable.build(prompt_lens, cache_sizes, ctx)
    var total_length = clt.total_length
    var max_seq = clt.max_seq_length_batch
    var max_ctx = clt.max_full_context_length
    var input_row_offsets_tensor = clt.input_row_offsets.device_tensor()

    comptime hs_layout = Layout.row_major(UNKNOWN_VALUE, K_BYTES)
    var hs = ManagedLayoutTensor[DATA_DTYPE, hs_layout](
        RuntimeLayout[hs_layout].row_major(IndexList[2](total_length, K_BYTES)),
        ctx,
    )
    _fill_random_u8(hs.tensor[update=False](), total_length * K_BYTES, 0, 255)
    var hs_dev = hs.device_tensor()

    comptime w_layout = Layout.row_major(n_total, K_BYTES)
    var w = ManagedLayoutTensor[DATA_DTYPE, w_layout](ctx)
    _fill_random_u8(w.tensor[update=False](), n_total * K_BYTES, 0, 255)
    var w_dev = w.device_tensor()

    comptime input_sf_layout = Layout.row_major(UNKNOWN_VALUE, scale_K)
    var input_scale = ManagedLayoutTensor[SCALE_DTYPE, input_sf_layout](
        RuntimeLayout[input_sf_layout].row_major(
            IndexList[2](total_length, scale_K)
        ),
        ctx,
    )
    _fill_random_u8(
        input_scale.tensor[update=False](), total_length * scale_K, 125, 129
    )
    var input_scale_dev = input_scale.device_tensor()

    comptime weight_sf_layout = Layout.row_major(n_total, scale_K)
    var weight_scale = ManagedLayoutTensor[SCALE_DTYPE, weight_sf_layout](ctx)
    _fill_random_u8(
        weight_scale.tensor[update=False](), n_total * scale_K, 125, 129
    )
    var weight_scale_dev = weight_scale.device_tensor()

    comptime kv_block_layout = Layout.row_major[6]()
    var main_block_shape = IndexList[6](
        num_paged_blocks, 2, num_layers, page_size, MAIN_KV_HEADS, HEAD_SIZE
    )
    var main_blocks = ManagedLayoutTensor[KV_DTYPE, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(main_block_shape), ctx
    )
    var main_blocks_ref = ManagedLayoutTensor[KV_DTYPE, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(main_block_shape), ctx
    )
    var index_block_shape = IndexList[6](
        num_paged_blocks, 2, num_layers, page_size, 1, HEAD_SIZE
    )
    var index_blocks = ManagedLayoutTensor[KV_DTYPE, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(index_block_shape), ctx
    )
    var index_blocks_ref = ManagedLayoutTensor[KV_DTYPE, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(index_block_shape), ctx
    )

    var main_n0 = main_blocks.tensor[update=False]().runtime_layout.size()
    unsafe_memset_zero(main_blocks.tensor[update=False]().ptr, main_n0)
    unsafe_memset_zero(main_blocks_ref.tensor[update=False]().ptr, main_n0)
    var index_n0 = index_blocks.tensor[update=False]().runtime_layout.size()
    unsafe_memset_zero(index_blocks.tensor[update=False]().ptr, index_n0)
    unsafe_memset_zero(index_blocks_ref.tensor[update=False]().ptr, index_n0)

    var main_lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_sizes, max_ctx, num_paged_blocks, ctx
    )
    var index_lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_sizes, max_ctx, num_paged_blocks, ctx
    )

    var main_collection = MainCollection(
        main_blocks.device_tensor(),
        clt.cache_lengths.device_tensor(),
        main_lut.device_tensor(),
        UInt32(max_seq),
        UInt32(max_ctx),
    )
    var main_collection_ref = MainCollection(
        main_blocks_ref.device_tensor(),
        clt.cache_lengths.device_tensor(),
        main_lut.device_tensor(),
        UInt32(max_seq),
        UInt32(max_ctx),
    )
    var index_collection = IndexCollection(
        index_blocks.device_tensor(),
        clt.cache_lengths.device_tensor(),
        index_lut.device_tensor(),
        UInt32(max_seq),
        UInt32(max_ctx),
    )
    var index_collection_ref = IndexCollection(
        index_blocks_ref.device_tensor(),
        clt.cache_lengths.device_tensor(),
        index_lut.device_tensor(),
        UInt32(max_seq),
        UInt32(max_ctx),
    )

    comptime fused_q_layout = Layout.row_major(UNKNOWN_VALUE, q_dim)
    var fused_q_out = ManagedLayoutTensor[OUT_DTYPE, fused_q_layout](
        RuntimeLayout[fused_q_layout].row_major(
            IndexList[2](total_length, q_dim)
        ),
        ctx,
    )
    comptime fused_iq_layout = Layout.row_major(UNKNOWN_VALUE, iq_dim)
    var fused_iq_out = ManagedLayoutTensor[OUT_DTYPE, fused_iq_layout](
        RuntimeLayout[fused_iq_layout].row_major(
            IndexList[2](total_length, iq_dim)
        ),
        ctx,
    )
    var q_out = ManagedLayoutTensor[OUT_DTYPE, fused_q_layout](
        RuntimeLayout[fused_q_layout].row_major(
            IndexList[2](total_length, q_dim)
        ),
        ctx,
    )
    var iq_out = ManagedLayoutTensor[OUT_DTYPE, fused_iq_layout](
        RuntimeLayout[fused_iq_layout].row_major(
            IndexList[2](total_length, iq_dim)
        ),
        ctx,
    )

    generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        target="gpu",
        mx_format=fmt,
    ](
        hs_dev,
        input_row_offsets_tensor,
        w_dev,
        input_scale_dev,
        weight_scale_dev,
        Float32(1.0),
        main_collection,
        index_collection,
        UInt32(layer_idx),
        iq_dim,
        fused_q_out.device_tensor(),
        fused_iq_out.device_tensor(),
        ctx,
    )

    var w_qkv = LayoutTensor[DATA_DTYPE, Layout.row_major(qkv_n, K_BYTES)](
        w_dev.ptr,
        RuntimeLayout[Layout.row_major(qkv_n, K_BYTES)].row_major(
            IndexList[2](qkv_n, K_BYTES)
        ),
    )
    var ws_qkv = LayoutTensor[SCALE_DTYPE, Layout.row_major(qkv_n, scale_K)](
        weight_scale_dev.ptr,
        RuntimeLayout[Layout.row_major(qkv_n, scale_K)].row_major(
            IndexList[2](qkv_n, scale_K)
        ),
    )

    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        target="gpu",
        mx_format=fmt,
    ](
        hs_dev,
        input_row_offsets_tensor,
        w_qkv,
        input_scale_dev,
        ws_qkv,
        Float32(1.0),
        main_collection_ref,
        UInt32(layer_idx),
        q_out.device_tensor(),
        ctx,
    )

    var w_idx = LayoutTensor[DATA_DTYPE, Layout.row_major(idx_n, K_BYTES)](
        w_dev.ptr + qkv_n * K_BYTES,
        RuntimeLayout[Layout.row_major(idx_n, K_BYTES)].row_major(
            IndexList[2](idx_n, K_BYTES)
        ),
    )
    var ws_idx = LayoutTensor[SCALE_DTYPE, Layout.row_major(idx_n, scale_K)](
        weight_scale_dev.ptr + qkv_n * scale_K,
        RuntimeLayout[Layout.row_major(idx_n, scale_K)].row_major(
            IndexList[2](idx_n, scale_K)
        ),
    )

    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        target="gpu",
        mx_format=fmt,
    ](
        hs_dev,
        input_row_offsets_tensor,
        w_idx,
        input_scale_dev,
        ws_idx,
        Float32(1.0),
        index_collection_ref,
        UInt32(layer_idx),
        iq_out.device_tensor(),
        ctx,
    )

    ctx.synchronize()

    var fused_q_host = fused_q_out.tensor[update=True]()
    var fused_iq_host = fused_iq_out.tensor[update=True]()
    var q_host = q_out.tensor[update=True]()
    var iq_host = iq_out.tensor[update=True]()
    var main_host = main_blocks.tensor[update=True]()
    var main_ref_host = main_blocks_ref.tensor[update=True]()
    var index_host = index_blocks.tensor[update=True]()
    var index_ref_host = index_blocks_ref.tensor[update=True]()

    var rtol_f = Float32(rtol)
    var atol_f = Float32(atol)

    var q_mm = 0
    var q_maxdiff = Float32(0.0)
    var iq_mm = 0
    var iq_maxdiff = Float32(0.0)
    for m in range(total_length):
        for c in range(q_dim):
            var a = fused_q_host.ptr[m * q_dim + c].cast[.float32]()
            var b = q_host.ptr[m * q_dim + c].cast[.float32]()
            if not isclose(a, b, atol=Float64(atol_f), rtol=Float64(rtol_f)):
                q_mm += 1
            q_maxdiff = max(q_maxdiff, abs(a - b))
        for c in range(iq_dim):
            var a = fused_iq_host.ptr[m * iq_dim + c].cast[.float32]()
            var b = iq_host.ptr[m * iq_dim + c].cast[.float32]()
            if not isclose(a, b, atol=Float64(atol_f), rtol=Float64(rtol_f)):
                iq_mm += 1
            iq_maxdiff = max(iq_maxdiff, abs(a - b))
    print(
        "Q out: ",
        q_mm,
        " over-tol / ",
        total_length * q_dim,
        ", max_abs_diff=",
        q_maxdiff,
        "  IndexQ out: ",
        iq_mm,
        " over-tol / ",
        total_length * iq_dim,
        ", max_abs_diff=",
        iq_maxdiff,
        sep="",
    )
    assert_equal(q_mm, 0)
    assert_equal(iq_mm, 0)

    var main_n = main_host.runtime_layout.size()
    var main_mismatches = 0
    var main_max_diff = Float32(0.0)
    for i in range(main_n):
        var a = main_host.ptr[i].cast[.float32]()
        var b = main_ref_host.ptr[i].cast[.float32]()
        if not isclose(a, b, atol=Float64(atol_f), rtol=Float64(rtol_f)):
            main_mismatches += 1
        main_max_diff = max(main_max_diff, abs(a - b))

    var index_n = index_host.runtime_layout.size()
    var index_mismatches = 0
    var index_max_diff = Float32(0.0)
    for i in range(index_n):
        var a = index_host.ptr[i].cast[.float32]()
        var b = index_ref_host.ptr[i].cast[.float32]()
        if not isclose(a, b, atol=Float64(atol_f), rtol=Float64(rtol_f)):
            index_mismatches += 1
        index_max_diff = max(index_max_diff, abs(a - b))

    print(
        "main cache: ",
        main_mismatches,
        " over-tol / ",
        main_n,
        " slots, max_abs_diff=",
        main_max_diff,
        sep="",
    )
    print(
        "index cache: ",
        index_mismatches,
        " over-tol / ",
        index_n,
        " slots, max_abs_diff=",
        index_max_diff,
        sep="",
    )

    assert_equal(main_mismatches, 0)
    assert_equal(index_mismatches, 0)

    _ = clt^
    _ = main_lut^
    _ = index_lut^


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        var ce_lens = List[Int]()
        for _ in range(2):
            ce_lens.append(Int(random_ui64(8, 64)))
        execute_dual_cache_fused_mxfp6(ce_lens, 4, 1, ctx)

        var tg_lens = List[Int]()
        for _ in range(4):
            tg_lens.append(1)
        execute_dual_cache_fused_mxfp6(tg_lens, 4, 2, ctx)

        for tg_m in [1, 16, 64]:
            var tg_lens_small = List[Int]()
            for _ in range(tg_m):
                tg_lens_small.append(1)
            execute_dual_cache_fused_mxfp6[hidden=6144](
                tg_lens_small, 4, 2, ctx
            )

        var ce_lens_big = List[Int]()
        for _ in range(4):
            ce_lens_big.append(64)
        execute_dual_cache_fused_mxfp6[hidden=6144](ce_lens_big, 4, 1, ctx)

        # M=8 sits between the M=1 GEMV branch and the M=16 tile, the band the
        # MXFP8 sibling covers and this did not.
        var tg_lens_8 = List[Int]()
        for _ in range(8):
            tg_lens_8.append(1)
        execute_dual_cache_fused_mxfp6[hidden=6144](tg_lens_8, 4, 2, ctx)

        # M values that are not a multiple of any tile's BM, so the row-bounds
        # gate on the store path has to fire. Every M above is 1, 8, 16, 64 or
        # 256.
        var tg_lens_17 = List[Int]()
        for _ in range(17):
            tg_lens_17.append(1)
        execute_dual_cache_fused_mxfp6[hidden=6144](tg_lens_17, 4, 2, ctx)

        # M=130 crosses 128 with a tail. Two prompts rather than 130 single
        # tokens: `num_paged_blocks` is 128, so a token-per-prompt batch that
        # large exhausts the lookup table before reaching the kernel.
        var ce_lens_130 = List[Int]()
        ce_lens_130.append(65)
        ce_lens_130.append(65)
        execute_dual_cache_fused_mxfp6[hidden=6144](ce_lens_130, 4, 1, ctx)

        # Ragged prefill with unequal prompts: the row offsets are non-uniform
        # and the total is not a tile multiple, unlike the 4x64 case above.
        var ce_ragged = List[Int]()
        ce_ragged.append(7)
        ce_ragged.append(33)
        ce_ragged.append(1)
        ce_ragged.append(96)
        execute_dual_cache_fused_mxfp6[hidden=6144](ce_ragged, 4, 1, ctx)

        # A third hidden size. 256 and 6144 both divide 512; 3072 does not, so
        # it lands on a different K-tile count and split-K factor.
        var tg_lens_3072 = List[Int]()
        for _ in range(16):
            tg_lens_3072.append(1)
        execute_dual_cache_fused_mxfp6[hidden=3072](tg_lens_3072, 4, 2, ctx)

        # E3M2 is the other half of `fp6_format`; every case above is E2M3, so
        # the parameter had one of its two values covered. Same packing and the
        # same 32-element scale blocks, so it exercises the encoding only.
        var e3m2_tg = List[Int]()
        for _ in range(4):
            e3m2_tg.append(1)
        execute_dual_cache_fused_mxfp6[hidden=256, fmt=MXFormat.FP6_E3M2](
            e3m2_tg, 4, 2, ctx
        )
        var e3m2_ce = List[Int]()
        for _ in range(2):
            e3m2_ce.append(48)
        execute_dual_cache_fused_mxfp6[hidden=6144, fmt=MXFormat.FP6_E3M2](
            e3m2_ce, 4, 1, ctx
        )
    print("\n=== ALL TESTS PASSED ===\n")
