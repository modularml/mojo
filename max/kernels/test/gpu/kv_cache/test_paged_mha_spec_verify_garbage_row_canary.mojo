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
"""FP8-only cross-row leak in the SM100 FA4 dense-GQA kernel.

MiniMax-M3's speculative-decode verify forward hands the paged-MHA kernel a
query window `1 + num_draft_tokens_to_verify` wide even when a draft is
rejected (`prompt_tokens_for_context` in
`max/python/max/pipelines/kv_cache/paged_kv_cache/cache_manager.py` sizes the
window from `k`, not from how many drafts are eventually accepted), so rows
past the accepted prefix are frequently never a real committed token.
Causal masking means each output row is a pure function of its OWN query
vector -- row 0 must never be affected by what rows 1..S-1 contain.

This test poisons rows 1..S-1 with NaN in one run and ordinary random data in
another, holding row 0's query vector and the entire KV band byte-identical
across the two runs. Row 0's output must be bit-for-bit identical and must
not itself contain NaN.

CONFIRMED FAILING on SM100 (B200): fp8 e4m3 KV cache only, at
(num_q_heads=16, group=16, q_width in {6, 7, 8}), (num_q_heads=8, group=8,
q_width=5), AND -- critically -- at (num_q_heads=16, group=16, q_width=4),
MiniMax-M3's ACTUAL production shape (num_speculative_tokens=3), for two
independently-found random seeds at an ordinary prefix=1024. bf16 is
bit-exact clean at every (num_q_heads, group, q_width) combination tried.

The defect is DATA-DEPENDENT, not purely shape-gated: at a fixed
(num_q_heads=16, group=16, q_width=4, prefix=1024) shape, most random seeds
are clean but a fraction (roughly 1 in 6 of the first 24 tried) reproduce the
identical cross-row leak (a wrong finite value, never NaN itself). The same
data-dependence holds at q_width=5, and at fixed random data the failing
width set is not a simple `q_width * group` threshold (group=16 fails from
width 6 at the "SPEC"-seeded default; group=32 stays clean through width 8 at
every seed tried; group=8 fails only at width 5, with width 6-10 clean on
both sides of it, at the default seed) -- so a "clean" width/prefix
combination in one run is evidence of nothing beyond that run's own random
draw, not of the shape being safe. Notably width=8 at group=16 fills the
BM=128 MMA tile with ZERO padding rows (`fuse_gqa` packs `group` heads along
`MMA_M`), yet still fails -- ruling out uninitialized padding-row reads as
the mechanism. Bisection (poisoning exactly one never-committed row at a
time, not all of them) additionally localized the corrupted physical BM rows
to a FIXED offset: only the row range occupying physical BM rows [16, 32)
ever corrupts row 0, regardless of `group` (verified: seq=1 for group=16,
seq=2 for group=8 -- both land on physical rows 16-31). Combined with the
seed-sensitivity, this pointed at the online-softmax lazy-rescale gate in
`softmax_warp.mojo`, which adopted a new running max on a warp-wide ballot
rather than each thread's own predicate. Warp 0 spans physical BM rows
[0, 32), so a never-committed row shared that branch with row 0. fp8's
`rescale_threshold=-2` reaches the gate far more often than bf16's `-8`,
which is why only fp8 showed it.

This is NVIDIA SM100 vendor-specialized code
(`nn/attention/gpu/nvidia/sm100/dispatch.mojo`'s `mha_sm100_dispatch`), not
shared arch-neutral Mojo -- AMD reaches a different kernel for the same
shapes and was not measured here.
"""

from std.math import ceildiv, sqrt
from std.random import randn, seed
from std.testing import assert_equal, assert_true

from max.gpu.host import DeviceContext, HostBuffer
from std.utils import IndexList

from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from kv_cache_test_utils import padded_lut_cols
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalMask
from std.utils.numerics import nan

comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime LAYER_IDX = 0


def _fill_randn[
    dtype: DType
](buf: HostBuffer[dtype], n: Int, ctx: DeviceContext) raises:
    """Fills `buf` with standard normals, going through bf16 for narrow dtypes.
    """
    var tmp = ctx.enqueue_create_host_buffer[.bfloat16](n)
    ctx.synchronize()
    randn(tmp.as_span())
    for i in range(n):
        buf[i] = tmp[i].cast[dtype]()
    _ = tmp


def _run_paged_mha[
    *,
    dtype: DType,
    out_dtype: DType,
    head_dim: Int,
    num_q_heads: Int,
    group: Int,
](
    q_host: HostBuffer[dtype],  # [extend, num_q_heads, head_dim]
    kv_rand: HostBuffer[dtype],  # [total_keys, head_kv, head_dim] logical KV
    extend: Int,  # query rows for this sequence == max_prompt_length
    prefix: Int,  # cache_length
    total_keys: Int,  # prefix + extend
    ctx: DeviceContext,
) raises -> List[Float64]:
    """Runs `flash_attention[ragged=True]` for one sequence, O as float64.

    K and V share the same bytes (`kv_rand`) -- this test only cares about
    the cross-row leak, not a realistic K != V distribution.
    """
    comptime head_kv = num_q_heads // group
    comptime scale = Float32(1.0) / sqrt(Float32(head_dim))
    var num_pages = ceildiv(total_keys, PAGE_SIZE)
    var phys_tokens = num_pages * PAGE_SIZE

    var q_size = extend * num_q_heads * head_dim
    var kv_block_size = (
        num_pages * 2 * NUM_LAYERS * PAGE_SIZE * head_kv * head_dim
    )

    var kv_host = ctx.enqueue_create_host_buffer[dtype](kv_block_size)
    ctx.synchronize()
    for i in range(kv_block_size):
        kv_host[i] = Scalar[dtype](0)
    for tok in range(phys_tokens):
        var page = tok // PAGE_SIZE
        var off = tok % PAGE_SIZE
        var is_valid = tok < total_keys
        for kh in range(head_kv):
            for d in range(head_dim):
                var val = kv_rand[
                    (tok * head_kv + kh) * head_dim + d
                ] if is_valid else Scalar[dtype](0)
                var k_off = (
                    (
                        ((page * 2 + 0) * NUM_LAYERS + LAYER_IDX) * PAGE_SIZE
                        + off
                    )
                    * head_kv
                    * head_dim
                    + kh * head_dim
                    + d
                )
                var v_off = (
                    (
                        ((page * 2 + 1) * NUM_LAYERS + LAYER_IDX) * PAGE_SIZE
                        + off
                    )
                    * head_kv
                    * head_dim
                    + kh * head_dim
                    + d
                )
                kv_host[k_off] = val
                kv_host[v_off] = val

    var lut_cols = padded_lut_cols(num_pages)

    var q_dev = ctx.enqueue_create_buffer[dtype](q_size)
    var o_dev = ctx.enqueue_create_buffer[out_dtype](q_size)
    var kv_block_dev = ctx.enqueue_create_buffer[dtype](kv_block_size)
    var cl_dev = ctx.enqueue_create_buffer[.uint32](1)
    var lut_dev = ctx.enqueue_create_buffer[.uint32](lut_cols)
    var ro_dev = ctx.enqueue_create_buffer[.uint32](2)

    var cl_host = ctx.enqueue_create_host_buffer[.uint32](1)
    var lut_host = ctx.enqueue_create_host_buffer[.uint32](lut_cols)
    var ro_host = ctx.enqueue_create_host_buffer[.uint32](2)
    ctx.synchronize()
    cl_host[0] = UInt32(prefix)
    for p in range(lut_cols):
        lut_host[p] = UInt32(p) if p < num_pages else UInt32(0)
    ro_host[0] = 0
    ro_host[1] = UInt32(extend)

    ctx.enqueue_copy(dst_buf=q_dev, src_buf=q_host)
    ctx.enqueue_copy(dst_buf=kv_block_dev, src_buf=kv_host)
    ctx.enqueue_copy(dst_buf=cl_dev, src_buf=cl_host)
    ctx.enqueue_copy(dst_buf=lut_dev, src_buf=lut_host)
    ctx.enqueue_copy(dst_buf=ro_dev, src_buf=ro_host)

    comptime kv_block_layout = Layout.row_major[6]()
    var kv_block_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_dev,
        RuntimeLayout[kv_block_layout].row_major(
            IndexList[6](num_pages, 2, NUM_LAYERS, PAGE_SIZE, head_kv, head_dim)
        ),
    )
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cl_tensor = LayoutTensor[mut=False, .uint32, cl_layout](
        cl_dev, RuntimeLayout[cl_layout].row_major(IndexList[1](1))
    )
    comptime lut_layout = Layout.row_major[2]()
    var lut_tensor = LayoutTensor[mut=False, .uint32, lut_layout](
        lut_dev, RuntimeLayout[lut_layout].row_major(IndexList[2](1, lut_cols))
    )

    var kv_collection = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(num_heads=head_kv, head_size=head_dim),
        PAGE_SIZE,
    ](
        kv_block_tensor.as_unsafe_any_origin(),
        cl_tensor,
        lut_tensor,
        UInt32(extend),  # max_prompt_length -- the route selector
        UInt32(total_keys),  # max_full_context_length
    )

    comptime qo_layout = Layout.row_major(UNKNOWN_VALUE, num_q_heads, head_dim)
    var q_tensor = LayoutTensor[mut=False, dtype, qo_layout](
        q_dev,
        RuntimeLayout[qo_layout].row_major(
            IndexList[3](extend, num_q_heads, head_dim)
        ),
    )
    comptime o_layout = Layout.row_major(UNKNOWN_VALUE, num_q_heads, head_dim)
    var o_tensor = LayoutTensor[out_dtype, o_layout](
        o_dev,
        RuntimeLayout[o_layout].row_major(
            IndexList[3](extend, num_q_heads, head_dim)
        ),
    )
    comptime ro_layout = Layout(UNKNOWN_VALUE)
    var ro_tensor = LayoutTensor[mut=False, .uint32, ro_layout](
        ro_dev, RuntimeLayout[ro_layout].row_major(IndexList[1](2))
    )

    flash_attention[ragged=True](
        o_tensor,
        q_tensor,
        kv_collection.get_key_cache(LAYER_IDX),
        kv_collection.get_value_cache(LAYER_IDX),
        CausalMask(),
        ro_tensor,
        scale,
        ctx,
    )

    var o_host = ctx.enqueue_create_host_buffer[out_dtype](q_size)
    ctx.enqueue_copy(dst_buf=o_host, src_buf=o_dev)
    ctx.synchronize()

    var out = List[Float64](length=q_size, fill=Float64(0))
    for i in range(q_size):
        out[i] = o_host[i].cast[.float64]()
    _ = q_dev
    _ = o_dev
    _ = kv_block_dev
    _ = cl_dev
    _ = lut_dev
    _ = ro_dev
    return out^


def test_garbage_row_canary[
    *,
    dtype: DType,
    out_dtype: DType,
    head_dim: Int,
    num_q_heads: Int,
    group: Int,
](
    prefix: Int, q_width: Int, ctx: DeviceContext, rng_seed: Int = 0x5350_4543
) raises:
    """Layer G: NaN-poison the never-committed draft rows; row 0 must not move.

    `max_prompt_length` is handed to the kernel as `1 + num_draft_tokens`
    regardless of how many drafts are eventually accepted, so rows
    `1 .. q_width-1` are frequently never real committed tokens. This test
    poisons exactly those rows with NaN in one run and ordinary random data
    in another, holding row 0's Q and the whole KV band byte-identical
    across the two runs, and requires row 0's output to be bit-for-bit
    identical -- causal masking makes each output row a function of its own
    Q vector only, so any difference here is a cross-row leak (shared
    accumulator, shared SMEM tile, or a reduction that sums over Q rows
    instead of just KV), not a reduction-order artifact.

    `rng_seed` is exposed (default unchanged, "SPEC") because this defect is
    DATA-DEPENDENT, not purely shape-dependent -- see the `q_width=4` case in
    `main()`, which is clean at the default seed but fails at others.
    """
    comptime head_kv = num_q_heads // group
    var row_w = num_q_heads * head_dim
    var total_keys = prefix + q_width
    print(
        "  [Layer G garbage-row canary] dtype=",
        String(dtype),
        " prefix=",
        prefix,
        " q_width=",
        q_width,
        " group=",
        group,
        " seed=",
        rng_seed,
        sep="",
    )

    var q_clean = ctx.enqueue_create_host_buffer[dtype](q_width * row_w)
    var kv_rand = ctx.enqueue_create_host_buffer[dtype](
        total_keys * head_kv * head_dim
    )
    ctx.synchronize()
    seed(rng_seed)
    _fill_randn[dtype](q_clean, q_width * row_w, ctx)
    _fill_randn[dtype](kv_rand, total_keys * head_kv * head_dim, ctx)

    var q_poison = ctx.enqueue_create_host_buffer[dtype](q_width * row_w)
    ctx.synchronize()
    for i in range(row_w):
        q_poison[i] = q_clean[i]  # row 0 identical in both runs
    for i in range(row_w, q_width * row_w):
        q_poison[i] = nan[dtype]()  # rows 1..q_width-1: never-committed

    var o_clean = _run_paged_mha[
        dtype=dtype,
        out_dtype=out_dtype,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        group=group,
    ](q_clean, kv_rand, q_width, prefix, total_keys, ctx)
    var o_poison = _run_paged_mha[
        dtype=dtype,
        out_dtype=out_dtype,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        group=group,
    ](q_poison, kv_rand, q_width, prefix, total_keys, ctx)

    var mismatches = 0
    var any_nan = False
    for i in range(row_w):
        if o_clean[i] != o_poison[i]:
            mismatches += 1
        if o_poison[i] != o_poison[i]:  # NaN != NaN
            any_nan = True
    print(
        "     row0 mismatches=",
        mismatches,
        "/",
        row_w,
        "  row0_has_nan(poisoned run)=",
        any_nan,
        sep="",
    )
    assert_equal(
        mismatches,
        0,
        (
            "row 0 changed when trailing never-committed draft rows were"
            " NaN-poisoned -- cross-row leak in the kernel"
        ),
    )
    assert_true(
        not any_nan,
        (
            "row 0 contains NaN when trailing rows are poisoned -- the poison"
            " leaked into the committed row's output"
        ),
    )

    _ = q_clean
    _ = q_poison
    _ = kv_rand


def main() raises:
    with DeviceContext() as ctx:
        var failures = List[String]()

        # bf16 control: clean at every (num_q_heads, group) combination this
        # test exercises fp8 at, including the widths that fail for fp8.
        for w in range(2, 9):
            try:
                test_garbage_row_canary[
                    dtype=DType.bfloat16,
                    out_dtype=DType.bfloat16,
                    head_dim=128,
                    num_q_heads=16,
                    group=16,
                ](1024, w, ctx)
            except e:
                failures.append(String("bf16 group=16 w=") + String(w))
                print(
                    "     CANARY FAILED (bf16, group=16, w=",
                    w,
                    "): ",
                    e,
                    sep="",
                )

        # fp8, group=16: the confirmed-failing shape (MiniMax-M3's dense
        # attention layers at TP4, num_draft_tokens_to_verify up to 7).
        for w in range(2, 9):
            try:
                test_garbage_row_canary[
                    dtype=DType.float8_e4m3fn,
                    out_dtype=DType.bfloat16,
                    head_dim=128,
                    num_q_heads=16,
                    group=16,
                ](1024, w, ctx)
            except e:
                failures.append(String("fp8 group=16 w=") + String(w))
                print(
                    "     CANARY FAILED (fp8, group=16, w=", w, "): ", e, sep=""
                )

        # fp8, group=8: a second, independently-confirmed failing shape at a
        # different (num_q_heads, group) -- the failing width (5) is NOT a
        # `q_width * group` boundary shared with group=16's (6-8), which is
        # why both are kept as separate regression points rather than one
        # collapsed onto the other.
        try:
            test_garbage_row_canary[
                dtype=DType.float8_e4m3fn,
                out_dtype=DType.bfloat16,
                head_dim=128,
                num_q_heads=8,
                group=8,
            ](1024, 5, ctx)
        except e:
            failures.append(String("fp8 group=8 w=5"))
            print("     CANARY FAILED (fp8, group=8, w=5): ", e, sep="")

        # fp8, group=16, q_width=4: MiniMax-M3's ACTUAL production shape
        # (num_speculative_tokens=3 -> 1 + 3 draft rows). The leak here is
        # DATA-DEPENDENT, not purely shape-gated: the default seed above is
        # clean at this width, but these two independently-discovered seeds
        # reproduce the same cross-row leak at prefix=1024, a perfectly
        # ordinary (non-extreme) cache length -- i.e. this is not a
        # wide-width-only defect, and q_width=4 is not safe in production.
        for w4_seed in [8, 13]:
            try:
                test_garbage_row_canary[
                    dtype=DType.float8_e4m3fn,
                    out_dtype=DType.bfloat16,
                    head_dim=128,
                    num_q_heads=16,
                    group=16,
                ](1024, 4, ctx, rng_seed=w4_seed)
            except e:
                failures.append(
                    String("fp8 group=16 w=4 seed=") + String(w4_seed)
                )
                print(
                    "     CANARY FAILED (fp8, group=16, w=4, seed=",
                    w4_seed,
                    "): ",
                    e,
                    sep="",
                )

        assert_true(
            len(failures) == 0,
            "garbage-row canary failed for: " + String(failures),
        )
