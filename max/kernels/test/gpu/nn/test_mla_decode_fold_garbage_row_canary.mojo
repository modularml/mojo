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
"""Cross-row leak in the SM100 native-FP8 MLA decode MTP-fold kernels.

`flare_mla_decoding`'s speculative-decode M-fold (`fold_q=True`) packs
`q_len_fold` query tokens times `num_heads` into one BM tile so a whole
verify window shares a single kernel launch (`mla_decode_qkv_fp8.mojo`'s
`fold_q` docstring: "Whether speculative decoding folds multiple Q tokens
into the BM tile"). The dispatcher (`mla_decode_dispatch.mojo`) routes
`num_heads * q_len <= 32` to Layout G (`mla_decode_qkv_fp8_layout_g.mojo`,
BM=32) and `32 < num_heads * q_len <= 64` to Layout E
(`mla_decode_utils.mojo`, BM=64).

Both layouts map physical row `= lane_id & (BM-1)` to `(q_local, head_local)`
via `q_local = row // num_heads`, so a 32-lane warp covers MULTIPLE q_local
values whenever `num_heads < 32` -- e.g. num_heads=16 packs 2 query tokens
per warp. Their online-softmax lazy-rescale gate
(`if _vote_nvidia_helper(diff < rescale_threshold) != 0:`) is a warp-wide
ANY-ballot over this per-thread predicate, exactly the pattern fixed for the
SM100 FA4 dense-GQA kernel in `softmax_warp.mojo` (PR #98402): one row's
`diff` can force every row physically sharing its warp onto the adopt
branch, changing `scale_for_old_max`/`new_max` (and hence the O rescale
trajectory) for a row whose OWN `diff` did not warrant it.

Speculative decoding hands this kernel a verify window sized by the number
of drafts OFFERED, not the number eventually accepted, so `q_local >= 1`
rows are frequently never real committed tokens. Causal masking makes each
output row a pure function of its own query vector -- `q_local=0` (all
`num_heads` heads of it) must never move when `q_local >= 1` rows are
poisoned with NaN, holding `q_local=0`'s Q and the whole K/V band
byte-identical across the two runs.

CONFIRMED FAILING pre-fix on SM100 (B200), BOTH bf16 and fp8, at
(num_heads=16, group=16, prefix=1024): Layout G (seq_len=2, all 32 BM rows
share ONE warp with no partial-coupling headroom) reproduces the leak at
the DEFAULT bf16 seed and roughly a third of the first 24 fp8 seeds tried;
Layout E (seq_len=4, only adjacent q_local PAIRS share a warp) reproduces
at roughly half the first 24 fp8 seeds tried and a fraction of bf16 seeds.
Unlike the dense-GQA kernel (fp8-only in practice, since bf16's threshold
is 4x tighter in log2 space), Layout G's 100% row-coupling makes this
defect dtype-independent -- do not assume bf16 is safe from one clean run.

This is NVIDIA SM100 vendor-specialized code
(`nn/attention/gpu/nvidia/sm100/mla_decode_dispatch.mojo`), reachable by ANY
MLA model doing speculative decoding (DeepSeek-V3/V3.1, Kimi-K2, GLM-4.6,
and MiniMax-M2 if configured for MLA) -- not specific to MiniMax-M3, which
uses dense (non-MLA) attention and the already-fixed kernel.
"""

from std.math import ceildiv, sqrt
from std.random import randn, seed
from std.testing import assert_equal, assert_true

from max.gpu.host import DeviceContext, HostBuffer
from layout import Idx, LayoutTensor, RuntimeLayout, TileTensor, row_major
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_utils import MHAConfig
from std.utils.numerics import nan


def _fill_randn_fp8[
    dtype: DType
](buf: HostBuffer[dtype], n: Int, ctx: DeviceContext) raises:
    """Fills `buf` with standard normals, going through bf16 for fp8."""
    var tmp = ctx.enqueue_create_host_buffer[.bfloat16](n)
    ctx.synchronize()
    randn(tmp.as_span())
    for i in range(n):
        buf[i] = tmp[i].cast[dtype]()
    _ = tmp


def _run_mla_decode_fold[
    *,
    q_type: DType,
    kv_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int,
](
    q_host: HostBuffer[q_type],  # [1, seq_len, num_heads, depth]
    k_host: HostBuffer[kv_type],  # [1, num_keys, kv_num_heads, depth]
    seq_len: Int,
    num_keys: Int,
    ctx: DeviceContext,
) raises -> List[Float64]:
    """Runs `flare_mla_decoding` for one batch item, output as float64."""
    comptime v_depth = depth - 64
    comptime kv_num_heads = num_heads // group
    comptime scale = Float32(0.125)
    comptime batch_size = 1

    var q_size = batch_size * seq_len * num_heads * depth
    var k_size = batch_size * num_keys * kv_num_heads * depth
    var o_size = batch_size * seq_len * num_heads * v_depth

    var q_dev = ctx.enqueue_create_buffer[q_type](q_size)
    var k_dev = ctx.enqueue_create_buffer[kv_type](k_size)
    var o_dev = ctx.enqueue_create_buffer[output_type](o_size)
    ctx.enqueue_copy(dst_buf=q_dev, src_buf=q_host)
    ctx.enqueue_copy(dst_buf=k_dev, src_buf=k_host)

    var q_tt = TileTensor(
        q_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )
    var k_tt = TileTensor(
        k_dev,
        row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth])),
    )
    var out_tt = TileTensor(
        o_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[v_depth]))
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        _is_cache_length_accurate=True,
        is_fp8_kv=True,
    ](batch_size, num_keys, seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime config = MHAConfig[q_type](num_heads, depth)
    flare_mla_decoding[config=config](
        out_tt.as_unsafe_any_origin(),
        q_tt,
        k_tt,
        CausalMask(),
        scale,
        ctx,
        scalar_args_buf_tt,
    )
    ctx.synchronize()

    var o_host = ctx.enqueue_create_host_buffer[output_type](o_size)
    ctx.enqueue_copy(dst_buf=o_host, src_buf=o_dev)
    ctx.synchronize()

    var out = List[Float64](length=o_size, fill=Float64(0))
    for i in range(o_size):
        out[i] = o_host[i].cast[.float64]()
    _ = q_dev
    _ = k_dev
    _ = o_dev
    _ = mla_args
    return out^


def test_fold_garbage_row_canary[
    *,
    q_type: DType,
    kv_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int,
](
    prefix: Int, seq_len: Int, ctx: DeviceContext, rng_seed: Int = 0x5350_4543
) raises:
    """NaN-poison the never-committed draft tokens; q_local=0 must not move.

    `seq_len` is the MTP verify window (`1 + num_draft_tokens`) handed to
    `flare_mla_decoding` regardless of how many drafts are eventually
    accepted, so tokens `1 .. seq_len-1` are frequently never real committed
    tokens. This test poisons exactly those tokens (every head) with NaN in
    one run and ordinary random data in another, holding token 0's Q (every
    head) and the whole K band byte-identical across the two runs, and
    requires token 0's output to be bit-for-bit identical.
    """
    comptime row_w = num_heads * depth
    comptime v_row_w = num_heads * (depth - 64)
    comptime kv_num_heads = num_heads // group
    var num_keys = prefix + seq_len
    print(
        "  [MLA fold garbage-row canary] q_type=",
        String(q_type),
        " num_heads=",
        num_heads,
        " seq_len=",
        seq_len,
        " prefix=",
        prefix,
        " seed=",
        rng_seed,
        sep="",
    )

    var q_clean = ctx.enqueue_create_host_buffer[q_type](seq_len * row_w)
    var k_rand = ctx.enqueue_create_host_buffer[kv_type](
        num_keys * kv_num_heads * depth
    )
    ctx.synchronize()
    seed(rng_seed)
    _fill_randn_fp8[q_type](q_clean, seq_len * row_w, ctx)
    _fill_randn_fp8[kv_type](k_rand, num_keys * kv_num_heads * depth, ctx)

    var q_poison = ctx.enqueue_create_host_buffer[q_type](seq_len * row_w)
    ctx.synchronize()
    for i in range(row_w):
        q_poison[i] = q_clean[i]  # token 0 identical in both runs
    for i in range(row_w, seq_len * row_w):
        q_poison[i] = nan[q_type]()  # tokens 1..seq_len-1: never-committed

    var o_clean = _run_mla_decode_fold[
        q_type=q_type,
        kv_type=kv_type,
        output_type=output_type,
        depth=depth,
        num_heads=num_heads,
        group=group,
    ](q_clean, k_rand, seq_len, num_keys, ctx)
    var o_poison = _run_mla_decode_fold[
        q_type=q_type,
        kv_type=kv_type,
        output_type=output_type,
        depth=depth,
        num_heads=num_heads,
        group=group,
    ](q_poison, k_rand, seq_len, num_keys, ctx)

    var mismatches = 0
    var any_nan = False
    for i in range(v_row_w):
        if o_clean[i] != o_poison[i]:
            mismatches += 1
        if o_poison[i] != o_poison[i]:  # NaN != NaN
            any_nan = True
    print(
        "     token0 mismatches=",
        mismatches,
        "/",
        v_row_w,
        "  token0_has_nan(poisoned run)=",
        any_nan,
        sep="",
    )
    assert_equal(
        mismatches,
        0,
        (
            "token 0 changed when trailing never-committed draft tokens were"
            " NaN-poisoned -- cross-row leak in the kernel"
        ),
    )
    assert_true(
        not any_nan,
        (
            "token 0 contains NaN when trailing tokens are poisoned -- the"
            " poison leaked into the committed token's output"
        ),
    )

    _ = q_clean
    _ = q_poison
    _ = k_rand


def main() raises:
    with DeviceContext() as ctx:
        var failures = List[String]()

        # bf16, seq_len=2: Layout G (num_heads*seq_len == 32, BM=32 fold --
        # ALL 32 rows share one warp, no partial-coupling headroom). NOT a
        # dtype-scoped defect: unlike the dense-GQA kernel (fp8-only in
        # practice), Layout G's 100% row-coupling reproduces the leak at
        # bf16 too (confirmed at the default seed below before the fix).
        for s in range(24):
            try:
                test_fold_garbage_row_canary[
                    q_type=DType.bfloat16,
                    kv_type=DType.bfloat16,
                    output_type=DType.bfloat16,
                    depth=576,
                    num_heads=16,
                    group=16,
                ](1024, 2, ctx, rng_seed=s)
            except e:
                failures.append(
                    String("bf16 LayoutG seq_len=2 seed=") + String(s)
                )
                print(
                    "     CANARY FAILED (bf16, LayoutG, seq_len=2, seed=",
                    s,
                    "): ",
                    e,
                    sep="",
                )

        # bf16, seq_len=4: Layout E (num_heads*seq_len == 64, BM=64 fold) --
        # only ADJACENT q_local pairs share a warp there, so the leak is
        # rarer at bf16's looser threshold; still swept to the same 24
        # seeds rather than assumed clean from one passing run.
        for s in range(24):
            try:
                test_fold_garbage_row_canary[
                    q_type=DType.bfloat16,
                    kv_type=DType.bfloat16,
                    output_type=DType.bfloat16,
                    depth=576,
                    num_heads=16,
                    group=16,
                ](1024, 4, ctx, rng_seed=s)
            except e:
                failures.append(
                    String("bf16 LayoutE seq_len=4 seed=") + String(s)
                )
                print(
                    "     CANARY FAILED (bf16, LayoutE, seq_len=4, seed=",
                    s,
                    "): ",
                    e,
                    sep="",
                )

        # fp8, seq_len=2: Layout G (num_heads*seq_len == 32, BM=32 fold).
        for s in range(24):
            try:
                test_fold_garbage_row_canary[
                    q_type=DType.float8_e4m3fn,
                    kv_type=DType.float8_e4m3fn,
                    output_type=DType.bfloat16,
                    depth=576,
                    num_heads=16,
                    group=16,
                ](1024, 2, ctx, rng_seed=s)
            except e:
                failures.append(
                    String("fp8 LayoutG seq_len=2 seed=") + String(s)
                )
                print(
                    "     CANARY FAILED (fp8, LayoutG, seq_len=2, seed=",
                    s,
                    "): ",
                    e,
                    sep="",
                )

        # fp8, seq_len=4: Layout E (num_heads*seq_len == 64, BM=64 fold) --
        # the num_speculative_tokens=3 verify-window width (1 committed + 3
        # draft), the MLA analogue of MiniMax-M3's confirmed-failing dense
        # attention shape.
        for s in range(24):
            try:
                test_fold_garbage_row_canary[
                    q_type=DType.float8_e4m3fn,
                    kv_type=DType.float8_e4m3fn,
                    output_type=DType.bfloat16,
                    depth=576,
                    num_heads=16,
                    group=16,
                ](1024, 4, ctx, rng_seed=s)
            except e:
                failures.append(
                    String("fp8 LayoutE seq_len=4 seed=") + String(s)
                )
                print(
                    "     CANARY FAILED (fp8, LayoutE, seq_len=4, seed=",
                    s,
                    "): ",
                    e,
                    sep="",
                )

        assert_true(
            len(failures) == 0,
            "MLA fold garbage-row canary failed for: " + String(failures),
        )
