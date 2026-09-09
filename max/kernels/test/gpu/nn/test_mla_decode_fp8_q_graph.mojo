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

"""Equivalence of the two Q staging dtypes through the fused MLA decode branch.

Both arms run the whole branch from identical inputs, so what is compared is
the graph rather than one kernel. Exactness is not the bar: staging Q in FP8
rounds it, which is the point. The arms must also differ somewhere, since two
that agreed bit for bit would mean the parameter selected nothing.
"""

from std.math import ceildiv, sqrt
from std.sys import has_nvidia_gpu_accelerator

from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
from internal_utils.fp8_utils import cast_saturating
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from nn.attention.gpu.mla_graph import mla_decode_branch_fp8
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from std.utils import IndexList
from std.utils.numerics import isfinite

comptime KV_LATENT_DIM = 512
comptime ROPE_DEPTH = 64
comptime K_CACHE_DIM = KV_LATENT_DIM + ROPE_DEPTH
comptime QK_NOPE_HEAD_DIM = 128
comptime Q_HEAD_DIM = QK_NOPE_HEAD_DIM + ROPE_DEPTH
comptime V_HEAD_DIM = KV_LATENT_DIM
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1

comptime M_SCALE_GRAN = 1
comptime N_SCALE_GRAN = 128
comptime K_SCALE_GRAN = 128

# Agreement bounds. The error scales with the output's own magnitude, so the
# absolute-error bound is expressed as a fraction of it rather than fixed.
comptime COS_MIN = 0.99
comptime MEAN_ERR_FRAC = 0.05


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


def _coprime_multiplier(n: Int) -> Int:
    """Returns a stride that visits every position of a length-`n` ring.

    The sparse index list must name distinct keys; a stride sharing a factor
    with `n` would revisit a subset and leave most of the cache unread.

    Args:
        n: Ring length to be coprime with.

    Returns:
        A multiplier coprime with `n`.
    """
    if n <= 1:
        return 1
    for m in [3, 5, 7, 11]:
        if _gcd(m, n) == 1:
            return m
    return 13


def _run_arm[
    num_heads: Int,
    page_size: Int,
    fp8_q: Bool,
](
    batch_size: Int,
    cache_len: Int,
    q_len: Int,
    topk: Int,
    out_host: MutPointer[BFloat16, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Runs the fused decode branch once and copies its output to the host.

    Every input is a deterministic function of its index, so the two arms see
    identical inputs by construction rather than by seeding.

    Parameters:
        num_heads: Number of query heads.
        page_size: KV cache page size.
        fp8_q: Q staging dtype selector under test.

    Args:
        batch_size: Number of sequences.
        cache_len: Cached key positions per sequence.
        q_len: Query positions per sequence.
        topk: Sparse index count per query position.
        out_host: Destination for the arm's output.
        ctx: Device context.
    """
    comptime dtype = DType.bfloat16
    comptime fp8_dtype = DType.float8_e4m3fn
    comptime scale_dtype = DType.float32
    comptime scale = Float32(0.125)
    comptime epsilon = Float32(1e-6)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=K_CACHE_DIM, is_mla=True
    )

    var num_keys = cache_len + q_len
    var total_q_tokens = batch_size * q_len
    var max_pages_per_batch = ceildiv(num_keys, page_size)
    var total_pages = batch_size * max_pages_per_batch

    var block_shape = IndexList[6](
        total_pages,
        1,
        NUM_LAYERS,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = total_pages * NUM_LAYERS * page_size * kv_params.head_size
    # A zero cache makes the attention output identically zero, which turns
    # every comparison below into a check that cannot fail.
    var blocks_device = ctx.enqueue_create_buffer[fp8_dtype](block_elems)
    var blocks_host = ctx.enqueue_create_host_buffer[fp8_dtype](block_elems)
    for i in range(block_elems):
        blocks_host[i] = cast_saturating[fp8_dtype](
            Float32((i % 37) - 18) * 0.125
        )
    ctx.enqueue_copy(blocks_device, blocks_host)

    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for bi in range(batch_size):
        for p in range(max_pages_per_batch):
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                bi * max_pages_per_batch + p
            )
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_len)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var q_size = total_q_tokens * num_heads * Q_HEAD_DIM
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    for i in range(q_size):
        q_host[i] = Scalar[dtype](Float32((i % 97) - 48) * 0.05)
    var q_device = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var kv_in_size = total_q_tokens * K_CACHE_DIM
    var kv_in_host = ctx.enqueue_create_host_buffer[dtype](kv_in_size)
    for i in range(kv_in_size):
        kv_in_host[i] = Scalar[dtype](Float32((i % 61) - 30) * 0.03)
    var kv_in_device = ctx.enqueue_create_buffer[dtype](kv_in_size)
    ctx.enqueue_copy(kv_in_device, kv_in_host)

    var w_uk_size = num_heads * KV_LATENT_DIM * QK_NOPE_HEAD_DIM
    var w_uk_host = ctx.enqueue_create_host_buffer[fp8_dtype](w_uk_size)
    for i in range(w_uk_size):
        w_uk_host[i] = cast_saturating[fp8_dtype](Float32((i % 29) - 14) * 0.25)
    var w_uk_device = ctx.enqueue_create_buffer[fp8_dtype](w_uk_size)
    ctx.enqueue_copy(w_uk_device, w_uk_host)

    comptime w_uk_scale_n = KV_LATENT_DIM // N_SCALE_GRAN
    comptime w_uk_scale_k = ceildiv(QK_NOPE_HEAD_DIM, K_SCALE_GRAN)
    var w_uk_scale_size = num_heads * w_uk_scale_n * w_uk_scale_k
    var w_uk_scale_host = ctx.enqueue_create_host_buffer[scale_dtype](
        w_uk_scale_size
    )
    for i in range(w_uk_scale_size):
        w_uk_scale_host[i] = Float32(0.0125)
    var w_uk_scale_device = ctx.enqueue_create_buffer[scale_dtype](
        w_uk_scale_size
    )
    ctx.enqueue_copy(w_uk_scale_device, w_uk_scale_host)

    var w_uv_size = num_heads * V_HEAD_DIM * KV_LATENT_DIM
    var w_uv_host = ctx.enqueue_create_host_buffer[fp8_dtype](w_uv_size)
    for i in range(w_uv_size):
        w_uv_host[i] = cast_saturating[fp8_dtype](Float32((i % 23) - 11) * 0.25)
    var w_uv_device = ctx.enqueue_create_buffer[fp8_dtype](w_uv_size)
    ctx.enqueue_copy(w_uv_device, w_uv_host)

    comptime w_uv_scale_n = V_HEAD_DIM // N_SCALE_GRAN
    comptime w_uv_scale_k = KV_LATENT_DIM // K_SCALE_GRAN
    var w_uv_scale_size = num_heads * w_uv_scale_n * w_uv_scale_k
    var w_uv_scale_host = ctx.enqueue_create_host_buffer[scale_dtype](
        w_uv_scale_size
    )
    for i in range(w_uv_scale_size):
        w_uv_scale_host[i] = Float32(0.0125)
    var w_uv_scale_device = ctx.enqueue_create_buffer[scale_dtype](
        w_uv_scale_size
    )
    ctx.enqueue_copy(w_uv_scale_device, w_uv_scale_host)

    var freqs_size = num_keys * ROPE_DEPTH
    var freqs_host = ctx.enqueue_create_host_buffer[scale_dtype](freqs_size)
    for i in range(freqs_size):
        freqs_host[i] = Float32(0.5) if (i % 2 == 0) else Float32(0.25)
    var freqs_device = ctx.enqueue_create_buffer[scale_dtype](freqs_size)
    ctx.enqueue_copy(freqs_device, freqs_host)

    var gamma_host = ctx.enqueue_create_host_buffer[dtype](KV_LATENT_DIM)
    for i in range(KV_LATENT_DIM):
        gamma_host[i] = Scalar[dtype](1.0)
    var gamma_device = ctx.enqueue_create_buffer[dtype](KV_LATENT_DIM)
    ctx.enqueue_copy(gamma_device, gamma_host)

    var out_size = total_q_tokens * num_heads * V_HEAD_DIM
    var out_device = ctx.enqueue_create_buffer[dtype](out_size)
    ctx.enqueue_memset(out_device, 0)

    var total_indices = total_q_tokens * topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    var mult = _coprime_multiplier(num_keys)
    for bi in range(batch_size):
        for s in range(q_len):
            var g = bi * q_len + s
            for i in range(topk):
                var t = (i * mult + 1 + s) % num_keys
                var page_idx = t // page_size
                var tok_in_page = t % page_size
                var block_id = Int(
                    lookup_table_host[bi * max_pages_per_batch + page_idx]
                )
                h_indices[g * topk + i] = Int32(
                    block_id * page_size + tok_in_page
                )
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i * q_len)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    comptime blocks_layout = Layout.row_major[6]()
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    comptime lut_layout = Layout.row_major[2]()

    # Any-origins, matching how the graph runtime builds this collection.
    var kv_collection = PagedKVCacheCollection[fp8_dtype, kv_params, page_size](
        LayoutTensor[fp8_dtype, blocks_layout, MutAnyOrigin](
            blocks_device.unsafe_ptr().as_unsafe_any_origin(),
            RuntimeLayout[blocks_layout].row_major(block_shape),
        ),
        LayoutTensor[.uint32, cl_layout, ImmutAnyOrigin](
            cache_lengths_device.unsafe_ptr().as_unsafe_any_origin(),
            RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
        ),
        LayoutTensor[.uint32, lut_layout, ImmutAnyOrigin](
            lookup_table_device.unsafe_ptr().as_unsafe_any_origin(),
            RuntimeLayout[lut_layout].row_major(
                IndexList[2](batch_size, max_pages_per_batch)
            ),
        ),
        UInt32(q_len),
        UInt32(cache_len),
    )

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[Q_HEAD_DIM])),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[V_HEAD_DIM])),
    )
    var kv_in_tt = TileTensor(
        kv_in_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[K_CACHE_DIM])),
    )
    var freqs_tt = TileTensor(
        freqs_device.unsafe_ptr(),
        row_major((num_keys, Idx[ROPE_DEPTH])),
    )
    var gamma_tt = TileTensor(
        gamma_device.unsafe_ptr(), row_major(Idx[KV_LATENT_DIM])
    )
    var w_uk_tt = TileTensor(
        w_uk_device.unsafe_ptr(),
        row_major((Idx[num_heads], Idx[KV_LATENT_DIM], Idx[QK_NOPE_HEAD_DIM])),
    )
    var w_uk_scale_tt = TileTensor(
        w_uk_scale_device.unsafe_ptr(),
        row_major((Idx[num_heads], Idx[w_uk_scale_n], Idx[w_uk_scale_k])),
    )
    var w_uv_tt = TileTensor(
        w_uv_device.unsafe_ptr(),
        row_major((Idx[num_heads], Idx[V_HEAD_DIM], Idx[KV_LATENT_DIM])),
    )
    var w_uv_scale_tt = TileTensor(
        w_uv_scale_device.unsafe_ptr(),
        row_major((Idx[num_heads], Idx[w_uv_scale_n], Idx[w_uv_scale_k])),
    )
    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(), row_major(batch_size + 1)
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, cache_len, q_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    @__parameter
    @always_inline
    def kv_input_fn[width: Int](coords: IndexList[2]) -> SIMD[dtype, width]:
        comptime assert type_of(kv_in_tt).flat_rank == 2
        return kv_in_tt.load[width=width](Coord(coords))

    mla_decode_branch_fp8[
        m_scale_granularity=M_SCALE_GRAN,
        n_scale_granularity=N_SCALE_GRAN,
        k_scale_granularity=K_SCALE_GRAN,
        mask_str="causal",
        kv_input_fn=kv_input_fn,
        target="gpu",
        sparse_mla=True,
        fp8_q=fp8_q,
    ](
        out_tt,
        q_tt,
        row_offsets_tt,
        freqs_tt,
        gamma_tt,
        kv_collection,
        UInt32(0),
        scale,
        epsilon,
        w_uk_tt,
        w_uk_scale_tt,
        w_uv_tt,
        w_uv_scale_tt,
        scalar_args_buf_tt,
        ctx,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=topk,
    )
    ctx.synchronize()

    var out_host_buf = ctx.enqueue_create_host_buffer[dtype](out_size)
    ctx.enqueue_copy(out_host_buf, out_device)
    ctx.synchronize()
    for i in range(out_size):
        out_host[i] = out_host_buf[i]

    # The tensors above hold untracked pointers into these buffers, so nothing
    # keeps them alive to the launch on its own.
    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = kv_in_device
    _ = w_uk_device
    _ = w_uk_scale_device
    _ = w_uv_device
    _ = w_uv_scale_device
    _ = freqs_device
    _ = gamma_device
    _ = out_device
    _ = d_indices_device
    _ = row_offsets_device


def _compare[
    num_heads: Int, page_size: Int
](
    name: String,
    batch_size: Int,
    cache_len: Int,
    q_len: Int,
    topk: Int,
    ctx: DeviceContext,
) raises:
    """Asserts the two Q staging dtypes agree, and that they are not the same
    computation.

    Parameters:
        num_heads: Number of query heads.
        page_size: KV cache page size.

    Args:
        name: Case label reported on failure.
        batch_size: Number of sequences.
        cache_len: Cached key positions per sequence.
        q_len: Query positions per sequence.
        topk: Sparse index count per query position.
        ctx: Device context.
    """
    comptime dtype = DType.bfloat16
    var out_size = batch_size * q_len * num_heads * V_HEAD_DIM

    var a_buf = ctx.enqueue_create_host_buffer[dtype](out_size)
    var b_buf = ctx.enqueue_create_host_buffer[dtype](out_size)
    ctx.synchronize()

    _run_arm[num_heads, page_size, fp8_q=True](
        batch_size,
        cache_len,
        q_len,
        topk,
        a_buf.unsafe_ptr().as_unsafe_any_origin(),
        ctx,
    )
    _run_arm[num_heads, page_size, fp8_q=False](
        batch_size,
        cache_len,
        q_len,
        topk,
        b_buf.unsafe_ptr().as_unsafe_any_origin(),
        ctx,
    )

    var dot = Float64(0)
    var sq_a = Float64(0)
    var sq_b = Float64(0)
    var max_abs_err = Float64(0)
    var sum_abs_err = Float64(0)
    var nonfinite = 0
    var differing = 0
    for i in range(out_size):
        var a = Float64(a_buf[i])
        var b = Float64(b_buf[i])
        if not isfinite(a_buf[i]) or not isfinite(b_buf[i]):
            nonfinite += 1
        if a_buf[i] != b_buf[i]:
            differing += 1
        dot += a * b
        sq_a += a * a
        sq_b += b * b
        var e = abs(a - b)
        sum_abs_err += e
        if e > max_abs_err:
            max_abs_err = e

    var cosine = Float64(0)
    if sq_a > 0 and sq_b > 0:
        cosine = dot / sqrt(sq_a * sq_b)
    var mean_abs_err = sum_abs_err / Float64(out_size)
    var rms = sqrt(sq_a / Float64(out_size))

    # Metrics print BEFORE the gates so they are visible on failure.
    print(
        "FP8Q ",
        name,
        " n=",
        out_size,
        " cosine=",
        cosine,
        " max_abs_err=",
        max_abs_err,
        " mean_abs_err=",
        mean_abs_err,
        " rms=",
        rms,
        " differing=",
        differing,
        " nonfinite=",
        nonfinite,
    )

    if nonfinite != 0:
        raise Error(
            String("non-finite output in ")
            + name
            + ": "
            + String(nonfinite)
            + " elements"
        )
    # An all-zero output would satisfy every tolerance below.
    if sq_a == 0 or sq_b == 0:
        raise Error(String("an arm produced an all-zero output in ") + name)
    # Staging Q in FP8 rounds it, so the arms cannot agree bit for bit unless
    # one dtype was used twice and the parameter selects nothing.
    if differing == 0:
        raise Error(
            String("arms are bit-identical in ")
            + name
            + ", so the Q staging dtype had no effect"
        )
    # Rounding Q to FP8 perturbs the output by a fraction of its own
    # magnitude; the direction is what has to be preserved.
    if cosine < COS_MIN:
        raise Error(
            String("cosine below tolerance in ")
            + name
            + ": "
            + String(cosine)
            + " < "
            + String(COS_MIN)
        )
    if mean_abs_err > MEAN_ERR_FRAC * rms:
        raise Error(
            String("mean absolute error above tolerance in ")
            + name
            + ": "
            + String(mean_abs_err)
            + " > "
            + String(MEAN_ERR_FRAC)
            + " * rms "
            + String(rms)
        )


def main() raises:
    comptime if not has_nvidia_gpu_accelerator():
        return
    with DeviceContext() as ctx:
        comptime if not _is_sm10x_gpu(ctx.default_device_info):
            return

        # The shape the FP8-Q path is built for: speculative-decode verify with
        # a saturated top-k window.
        _compare[num_heads=8, page_size=128](
            "base", batch_size=8, cache_len=2042, q_len=6, topk=2048, ctx=ctx
        )
        # Below saturation: fewer keys exist than the index list can name, so
        # the gather clamps and the mask drops the overflow.
        _compare[num_heads=8, page_size=128](
            "short_cache",
            batch_size=4,
            cache_len=512,
            q_len=6,
            topk=2048,
            ctx=ctx,
        )
        # Prime cache length, and one that is not a multiple of the page size.
        _compare[num_heads=8, page_size=128](
            "prime", batch_size=3, cache_len=1021, q_len=6, topk=1024, ctx=ctx
        )
        _compare[num_heads=8, page_size=64](
            "prime_p64",
            batch_size=2,
            cache_len=613,
            q_len=5,
            topk=512,
            ctx=ctx,
        )
        # Single sequence, and a single query position: q_len=1 takes the
        # draft path rather than the verify path.
        _compare[num_heads=8, page_size=128](
            "batch1", batch_size=1, cache_len=997, q_len=6, topk=512, ctx=ctx
        )
        _compare[num_heads=8, page_size=128](
            "qlen1", batch_size=4, cache_len=1024, q_len=1, topk=512, ctx=ctx
        )
        # Head counts other than the target: the dispatch is shared.
        _compare[num_heads=16, page_size=128](
            "heads16", batch_size=2, cache_len=1024, q_len=4, topk=512, ctx=ctx
        )
