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
"""Cross-row leak in the SM100 depth=512 pair-CTA `fuse_gqa` prefill kernel.

`mha_sm100_depth512_dispatch` (`mha_depth512/dispatch.mojo`) fuses `group`
query heads that share one KV head into a single MMA (`fuse_gqa`), so
`per_thread_score_row = score_row + cta_rank*BM_eff + m_row // group`: groups
of `group` consecutive physical `m_row` values collapse onto ONE sequence
position, but distinct HEADS within that group, and -- whenever `group < 32`
-- distinct SEQUENCE POSITIONS, land in the same 32-lane warp. The online-
softmax lazy-rescale gate in `softmax_warp.mojo`'s main loop
(`if _vote_nvidia_helper(diff < rescale_threshold) != 0:`) is a warp-wide
ANY-ballot over this per-thread predicate -- the same pattern fixed for the
SM100 FA4 dense-GQA kernel in PR #98402 and for MLA decode's MTP-fold in this
PR. One row's growth can force a sibling row (a different head and/or a
different, causally-independent sequence position) onto the "adopt a new
running max" branch, substituting a real `exp2(diff)` correction the
sibling's OWN predicate did not ask for.

Unlike the fixed sites, this route is PREFILL-ONLY
(`mha_sm100_depth512_dispatch` is gated on `not is_token_generation`), so
there is no draft-vs-committed-token asymmetry to exploit for the canary.
Instead: hold one query position's Q byte-identical across two runs, poison
every OTHER query position sharing its physical warp with garbage (large
finite values, not NaN -- causal masking means a genuinely garbage-but-finite
sibling row is the realistic "this row's own data, not shared", and NaN would
propagate into intermediate reductions in ways that could mask rather than
prove the leak), and require the held-identical row's output not to move.

At `num_q_heads=32, group=8`, one warp (`m_row` 0..31) covers exactly 4
sequence positions (`m_row // group` in {0,1,2,3}) x 8 heads each. The last
BM tile of a `seq_len=1024` causal launch (`score_row=1008`) gives the
observed row (position 1008) the deepest key range (up to 1008 keys, 4
main-loop rescale opportunities at `BN=256`), maximizing the chance that some
poisoned sibling in {1009, 1010, 1011} triggers a rescale its own growth
would not have warranted for position 1008.

CONFIRMED FAILING pre-fix on SM100 (B200), fp8, `seq_len=num_keys=1024`,
`num_q_heads=32`, `group=8`: reproduces at a fraction of the first 24 seeds
tried. bf16's 4x looser log2 threshold (`rescale_threshold=-8` vs fp8's `-2`)
makes it rarer but not impossible -- swept 24 seeds at both dtypes rather
than assuming bf16 is safe from a single passing run.
"""

from std.math import sqrt
from std.random import randn, seed
from std.testing import assert_equal, assert_true

from max.gpu.host import DeviceContext, HostBuffer
from layout import Idx, TileTensor, row_major
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalMask
from std.utils.numerics import nan


def _run_d512_fuse_gqa[
    *,
    qkv_type: DType,
    output_type: DType,
    head_dim: Int,
    num_q_heads: Int,
    group: Int,
](
    q_host: HostBuffer[qkv_type],  # [1, seq_len, num_q_heads, head_dim]
    k_host: HostBuffer[qkv_type],  # [1, num_keys, kv_num_heads, head_dim]
    v_host: HostBuffer[qkv_type],  # [1, num_keys, kv_num_heads, head_dim]
    seq_len: Int,
    num_keys: Int,
    ctx: DeviceContext,
) raises -> List[Float64]:
    """Runs `flash_attention` (dispatches to the depth512 pair-CTA kernel)
    for one batch item, returning the output cast to float64."""
    comptime kv_num_heads = num_q_heads // group
    comptime scale = Float32(1.0) / sqrt(Float32(head_dim))
    comptime batch_size = 1

    var q_size = batch_size * seq_len * num_q_heads * head_dim
    var kv_size = batch_size * num_keys * kv_num_heads * head_dim
    var o_size = q_size

    var q_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var v_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var o_dev = ctx.enqueue_create_buffer[output_type](o_size)
    ctx.enqueue_copy(dst_buf=q_dev, src_buf=q_host)
    ctx.enqueue_copy(dst_buf=k_dev, src_buf=k_host)
    ctx.enqueue_copy(dst_buf=v_dev, src_buf=v_host)

    var q_tt = TileTensor(
        q_dev,
        row_major((batch_size, seq_len, Idx[num_q_heads], Idx[head_dim])),
    )
    var k_tt = TileTensor(
        k_dev,
        row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[head_dim])),
    )
    var v_tt = TileTensor(
        v_dev,
        row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[head_dim])),
    )
    var out_tt = TileTensor(
        o_dev,
        row_major((batch_size, seq_len, Idx[num_q_heads], Idx[head_dim])),
    )

    flash_attention(out_tt, q_tt, k_tt, v_tt, CausalMask(), scale, ctx)
    ctx.synchronize()

    var o_host = ctx.enqueue_create_host_buffer[output_type](o_size)
    ctx.enqueue_copy(dst_buf=o_host, src_buf=o_dev)
    ctx.synchronize()

    var out = List[Float64](length=o_size, fill=Float64(0))
    for i in range(o_size):
        out[i] = o_host[i].cast[.float64]()
    _ = q_dev
    _ = k_dev
    _ = v_dev
    _ = o_dev
    return out^


def test_d512_fuse_gqa_crossrow_canary[
    *,
    qkv_type: DType,
    output_type: DType,
](seq_len: Int, ctx: DeviceContext, rng_seed: Int = 0x5350_4543) raises:
    """Poison every sibling row in the observed row's warp; the observed
    row's output must not move.

    `num_q_heads=32, group=8, head_dim=512`: one warp covers exactly 4
    sequence positions (`m_row // group`). The observed row is the LAST
    position of the launch (`score_row = seq_len - 16`, i.e. `m_row=0`),
    which causal masking gives the deepest key range -- the most chances for
    the lazy-rescale gate to actually decide differently between rows.
    Every position sharing that warp (`observed+1 .. observed+3`) is
    poisoned with large-magnitude garbage in one run and held at the
    same clean random data as the control run in the other; every OTHER
    position in the launch (different warp / different cta_rank) is left
    identical across both runs since it cannot leak into the observed row
    through any path this canary is checking.
    """
    comptime head_dim = 512
    comptime num_q_heads = 32
    comptime group = 8
    comptime kv_num_heads = num_q_heads // group
    comptime row_w = num_q_heads * head_dim
    var num_keys = seq_len
    var observed = seq_len - 16  # score_row of the last BM tile, m_row=0
    print(
        "  [d512 fuse_gqa crossrow canary] qkv_type=",
        String(qkv_type),
        " seq_len=",
        seq_len,
        " observed_row=",
        observed,
        " seed=",
        rng_seed,
        sep="",
    )

    var q_clean = ctx.enqueue_create_host_buffer[qkv_type](seq_len * row_w)
    var k_rand = ctx.enqueue_create_host_buffer[qkv_type](
        num_keys * kv_num_heads * head_dim
    )
    var v_rand = ctx.enqueue_create_host_buffer[qkv_type](
        num_keys * kv_num_heads * head_dim
    )
    ctx.synchronize()
    seed(rng_seed)
    randn(q_clean.as_span())
    randn(k_rand.as_span())
    randn(v_rand.as_span())

    var q_poison = ctx.enqueue_create_host_buffer[qkv_type](seq_len * row_w)
    ctx.synchronize()
    for i in range(seq_len * row_w):
        q_poison[i] = q_clean[i]
    # Poison rows {observed+1, observed+2, observed+3} -- the other 3
    # sequence positions sharing `observed`'s physical warp under
    # group=8 (m_row // group in {0,1,2,3}). Large finite garbage, not
    # NaN: a NaN would poison intermediate cross-thread `exchange_reduce`
    # sums even along legitimate (non-buggy) paths, which would corrupt
    # the comparison rather than isolate the warp-vote leak.
    for r in range(observed + 1, observed + 4):
        for i in range(row_w):
            q_poison[r * row_w + i] = Scalar[qkv_type](30)

    var o_clean = _run_d512_fuse_gqa[
        qkv_type=qkv_type,
        output_type=output_type,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        group=group,
    ](q_clean, k_rand, v_rand, seq_len, num_keys, ctx)
    var o_poison = _run_d512_fuse_gqa[
        qkv_type=qkv_type,
        output_type=output_type,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        group=group,
    ](q_poison, k_rand, v_rand, seq_len, num_keys, ctx)

    var mismatches = 0
    var any_nan = False
    var base = observed * row_w
    for i in range(row_w):
        if o_clean[base + i] != o_poison[base + i]:
            mismatches += 1
        if o_poison[base + i] != o_poison[base + i]:  # NaN != NaN
            any_nan = True
    print(
        "     observed-row mismatches=",
        mismatches,
        "/",
        row_w,
        "  has_nan=",
        any_nan,
        sep="",
    )
    assert_equal(
        mismatches,
        0,
        (
            "observed row changed when sibling rows in its warp were"
            " poisoned -- cross-row leak in the kernel"
        ),
    )
    assert_true(
        not any_nan,
        "observed row contains NaN after poisoning sibling rows",
    )

    _ = q_clean
    _ = q_poison
    _ = k_rand
    _ = v_rand


def main() raises:
    with DeviceContext() as ctx:
        var failures = List[String]()

        for s in range(24):
            try:
                test_d512_fuse_gqa_crossrow_canary[
                    qkv_type=DType.float8_e4m3fn,
                    output_type=DType.bfloat16,
                ](1024, ctx, rng_seed=s)
            except e:
                failures.append(String("fp8 seed=") + String(s))
                print("     CANARY FAILED (fp8, seed=", s, "): ", e, sep="")

        for s in range(24):
            try:
                test_d512_fuse_gqa_crossrow_canary[
                    qkv_type=DType.bfloat16,
                    output_type=DType.bfloat16,
                ](1024, ctx, rng_seed=s)
            except e:
                failures.append(String("bf16 seed=") + String(s))
                print("     CANARY FAILED (bf16, seed=", s, "): ", e, sep="")

        assert_true(
            len(failures) == 0,
            "d512 fuse_gqa crossrow canary failed for: " + String(failures),
        )
