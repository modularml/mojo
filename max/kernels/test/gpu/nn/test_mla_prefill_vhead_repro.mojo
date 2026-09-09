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
"""CENG-282 repro: SM100 dense MLA prefill hardcodes v_head_dim == qk_nope.

The existing ``test_mla_prefill_generic_paged.mojo`` uses ONE ``kv_depth``
for the K-nope tensor, the V tensor, AND the output tensor, baking in
``qk_nope_head_dim == v_head_dim == 128``. That is exactly why it never
caught the bug.

This standalone repro DECOUPLES the two depths via the shared
``run_test_paged_prefill`` driver (in ``_paged_prefill_test_utils.mojo``,
called here with ``diagnostic_bands=True``):

  - ``nope_depth`` = width of the K-nope tensor = ``q_depth - 64`` (so
    Q@K^T and softmax are correct).
  - ``v_depth``    = width of the V tensor AND the output tensor.

The SM100 kernel derives ``ov_depth = nope_depth = q_depth - rope_depth``
and uses ``depth=nope_depth`` for the V TMA and ``BN=ov_depth`` for the
output store (``mla_prefill_generic.mojo:2513`` / ``:2472`` /
``mla_prefill_utils.mojo:159``). When ``v_depth != nope_depth`` the V
load is mis-strided and the output store writes the wrong width →
coherent-but-wrong output (or OOB if the real V buffer is narrower than
``nope_depth``).

Reference: naive MHA with K_ref = nope+rope (width q_depth), V_ref padded
to q_depth (first ``v_depth`` columns = real V, rest zero); compare the
first ``v_depth`` output columns.
"""

from std.random import seed
from std.sys import get_defined_int

from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu

from _paged_prefill_test_utils import run_test_paged_prefill


comptime PAGE_SIZE = get_defined_int["page_size", 128]()


def main() raises:
    """CENG-282 regression: SM100 dense MLA prefill must honor v_head_dim
    independently of qk_nope_head_dim.

    Runs four (nope_depth, v_depth) geometries at this binary's compile-time
    `page_size`, all of which must PASS post-fix:
      - CONTROL  nope=128, v=128  — DeepSeek (v == qk_nope); byte-identity guard.
      - GLM-min  nope=128, v=64   — v < qk_nope (minimized GLM testdata shape).
      - STRESS   nope=128, v=192  — v > qk_nope; exercises the split-KV path.
      - REAL GLM nope=192, v=256  — the real GLM-5.1-FP8 dims (q_depth=256);
        ov_depth=256 is too wide for the 2-O TMEM layout, so it routes to the
        single-O (num_qo=1) fallback. Pre-single-O this did not compile.
    Pre-fix, the v != nope cases produced coherent-but-wrong output (the kernel
    hardcoded the V/output width to nope_depth); the shared
    ``run_test_paged_prefill`` driver `assert_almost_equal`s against a
    naive-MHA reference and raises on mismatch. ``diagnostic_bands=True``
    prints the per-band error breakdown (lower band `d < nope_depth` vs upper
    band `d >= nope_depth`) before asserting.

    The REAL GLM section additionally sweeps seq_len across the single-O
    KV-tile boundaries (fresh prefill) and covers a cached-prefix shape
    (``num_keys >> seq_len`` with ``cache_length > 0``), so the single-O
    serial KV loop is exercised in the appended-after-a-prefix geometry the
    kernel hits in production — not only the ``seq_len == num_keys`` case.

    Beyond the base geometries, this test brings the decoupled-v / single-O
    path to dimensional rigor comparable to ``test_flash_attention.mojo``
    (adapted to MLA-prefill causal semantics — ``cache_length == num_keys -
    seq_len`` always holds, and MLA is per-head K/V so ``kv_heads == 1`` /
    ``group == 1``):
      - ``num_heads`` variety {1, 3, 128} on the single-O REAL GLM dims (16
        is the base). 128 is the production DP-attention config, run once
        with a cached prefix; 1/3 are cheap tiny/odd-seq shapes.
      - ``batch_size`` {2, 4} on single-O REAL GLM — exercises the
        cross-batch input/cache row-offset tables and per-batch paged LUT
        that ``batch_size == 1`` never touches. batch=4 uses a small seq to
        bound runtime.
      - Tiny / odd seq edges {1, 14, 15} and a tiny-Q x many-KV case
        (seq=14 over a long cached prefix), mirroring MHA's (14, 18) /
        (119, 200) — all on the base heads=16 single-O binary (no extra
        compile cost).
      - Multi-KV-tile coverage for the decoupled 2-O paths (v=64 < nope and
        v=192 > nope), which the (64, 64) rows only exercise as a single KV
        tile — reuses those binaries (no extra compile cost).
    The 2-O / DeepSeek (v == nope) rigor across page_size and num_keys is
    already owned by ``test_mla_prefill_generic_paged.mojo`` and is not
    duplicated here. Decode / seq==1-as-decode, attention sinks, GQA
    ``group != 1`` (MLA is per-head K/V), NullMask (MLA is causal), and
    ``seq_len > num_keys`` are not applicable to this path.

    Single-O register-spill guard (manual -- no automated assertion here).
    The REAL-GLM single-O section also exercises the ``num_reg_softmax = 208``
    tuning in ``mla_prefill_generic.mojo``, whose spill-freedom sits on a
    narrow, ptxas-version-dependent allocation island. That is a ptxas/SASS
    property with no lightweight in-tree check (the AOT binary is JIT'd with no
    static device code; ``_compile_code`` emits pre-ptxas PTX where the spill
    never appears; ``_dump_sass`` needs a GPU + CUDA Toolkit and would be
    brittle as the island moves). After any ptxas/toolchain bump, re-verify
    ZERO local ld/st sectors on the single-O ``mla_prefill_generic`` launch
    using the ncu recipe documented at that ``num_reg_softmax`` line.
    """
    with DeviceContext() as ctx:
        comptime if _is_sm10x_gpu(ctx.default_device_info):
            # CONTROL: matched nope == v == 128 (DeepSeek). Byte-identity guard.
            print("=== CONTROL matched nope=v=128 (q=192) ===")
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=192,
                num_heads=16,
                nope_depth=128,
                v_depth=128,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
            ](64, 64, ctx)
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=192,
                num_heads=16,
                nope_depth=128,
                v_depth=128,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
            ](128, 128, ctx)

            # GLM: v=64 < nope=128 (the production GLM shape).
            print("=== GLM nope=128 v=64 (q=192) ===")
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=192,
                num_heads=16,
                nope_depth=128,
                v_depth=64,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
            ](64, 64, ctx)
            # Multi-KV-tile + cached prefix for the v < nope 2-O path (the
            # (64, 64) row above is a single KV tile at cache=0). seq=128
            # queries appended after a 128-token prefix -> num_keys=256
            # (multi-tile), cache_length=128. Reuses the v=64 binary (no extra
            # compile cost) but exercises the split-KV accumulation + start_pos
            # causal offset that the single-tile row never reaches.
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=192,
                num_heads=16,
                nope_depth=128,
                v_depth=64,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
                cache_length=128,
            ](128, 256, ctx)

            # STRESS: v=192 > nope=128 (in-bounds; exercises the split-KV path).
            print("=== STRESS nope=128 v=192 (q=192) ===")
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=192,
                num_heads=16,
                nope_depth=128,
                v_depth=192,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
            ](64, 64, ctx)
            # Multi-KV-tile + cached prefix for the v > nope 2-O path (mirror
            # of the v=64 multi-tile row above). seq=128 over a 128-token
            # prefix -> num_keys=256, cache_length=128. Reuses the v=192
            # binary (no extra compile cost).
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=192,
                num_heads=16,
                nope_depth=128,
                v_depth=192,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
                cache_length=128,
            ](128, 256, ctx)

            # REAL GLM-5.1: qk_nope=192, q_depth=256, v=256. ov_depth=256 is too
            # wide for the 2-O TMEM layout (standard BN=0), so this routes to the
            # single-O (num_qo=1) fallback. Pre-single-O this would not compile.
            #
            # The single-O fallback aliases the second TMEM-O region onto the
            # first (`2*BN + 2*padded_ov > 512` for v=256), so it CANNOT run the
            # two-WG even/odd LSE-combine that the standard 1Q path uses; one
            # warp group folds every K-tile serially into the single O0. That
            # combine is only reached when a sequence spans MORE THAN ONE BN=64
            # KV tile, i.e. `num_keys > BN` (seq_len > 64). Historically this
            # test ran ONLY seq_len=64 (exactly one KV tile → the single-tile
            # fast path), so the multi-KV-tile single-O bug (CUDA illegal
            # address at seq_len >= 65) went uncaught. Cover seq_len that spans
            # 2 KV tiles (65, boundary), several KV tiles (128, 256), and the
            # 2-Q-tile boundary (129, 257) at both page sizes so CI catches this
            # class going forward. Keep the list short for CI runtime.
            print("=== REAL GLM nope=192 v=256 (q=256), single-O multi-KV ===")
            # seq_len sweep crossing the single-O KV-tile boundaries:
            #   64  -> 1 KV tile   (single-tile fast path; the only pre-fix case)
            #   65  -> 2 KV tiles  (the OOB boundary; was CUDA illegal-address)
            #   128 -> 2 KV tiles  (full second tile)
            #   129 -> 2 Q tiles x >= 3 KV tiles
            #   256 -> several KV tiles
            for seq in [64, 65, 128, 129, 256]:
                seed(0)
                run_test_paged_prefill[
                    qkv_type=DType.bfloat16,
                    k_rope_type=DType.bfloat16,
                    output_type=DType.bfloat16,
                    depth=256,
                    num_heads=16,
                    nope_depth=192,
                    v_depth=256,
                    page_size=PAGE_SIZE,
                    batch_size=1,
                    diagnostic_bands=True,
                ](seq, seq, ctx)

            # Tiny / odd single-O seq edges (fresh prefill; heads=16 so these
            # reuse the REAL GLM binary above — no extra compile cost). MLA
            # prefill still runs the prefill kernel at seq_len=1 (this is NOT
            # the decode path); the kernel grid is ceildiv(seq_len, 128), so
            # seq {1, 14, 15} all land in a single partial Q tile against a
            # single partial KV tile (T==1 single-O fast path). This mirrors
            # the MHA test's small-seq coverage (14, 15) adapted to the
            # decoupled-v single-O path.
            print("=== REAL GLM nope=192 v=256, tiny/odd single-O seq ===")
            for seq in [1, 14, 15]:
                seed(0)
                run_test_paged_prefill[
                    qkv_type=DType.bfloat16,
                    k_rope_type=DType.bfloat16,
                    output_type=DType.bfloat16,
                    depth=256,
                    num_heads=16,
                    nope_depth=192,
                    v_depth=256,
                    page_size=PAGE_SIZE,
                    batch_size=1,
                    diagnostic_bands=True,
                ](seq, seq, ctx)

            # Tiny-Q x many-KV single-O: 14 queries appended after a long
            # cached prefix -> num_keys=512 (8 single-O KV tiles),
            # cache_length=498 (== num_keys - seq_len). The queries fill one
            # partial Q tile but the single-O serial KV loop folds 8 tiles,
            # so this stresses the "few queries, many keys" cached-decode
            # geometry (mirrors MHA's (14, 18) / (119, 200) tiny-Q rows, at
            # the scale where the serial single-O accumulation runs). Reuses
            # the heads=16 binary (no extra compile cost).
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=256,
                num_heads=16,
                nope_depth=192,
                v_depth=256,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
                cache_length=498,
            ](14, 512, ctx)

            # REAL GLM single-O with a CACHED PREFIX (cache_length > 0) and a
            # larger fresh prefill. The rows above are all fresh prefill
            # (num_keys == seq_len, cache_length == 0): the queries sit at
            # positions [0, seq_len) over exactly seq_len KV tokens. A
            # seq_len == num_keys test never exercises the cached-decode
            # geometry the kernel uses in production (queries appended after a
            # long KV prefix), which stresses the single-O serial KV loop over
            # MANY more tiles than the query count and the start_pos causal
            # offset. Lock that regime in:
            #   * seq_len=128, num_keys=2048, cache_length=1920 — queries at
            #     [1920, 2048); ~32 KV tiles (num_keys >> seq_len). This is the
            #     cached-prefix single-O case the fresh-prefill rows miss.
            #   * seq_len=num_keys=512 — a larger fresh prefill (8 KV tiles),
            #     cheap for CI (the seq=4096 sweep point is validated offline).
            # The naive-MHA reference needs no change: it places query y at
            # score_row = y + (num_keys - seq_len) under CausalMask, matching
            # the kernel's start_pos = cache_length exactly.
            print(
                "=== REAL GLM nope=192 v=256, cached prefix + larger fresh ==="
            )
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=256,
                num_heads=16,
                nope_depth=192,
                v_depth=256,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
                cache_length=1920,
            ](128, 2048, ctx)
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=256,
                num_heads=16,
                nope_depth=192,
                v_depth=256,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
            ](512, 512, ctx)

            # Multi-Q-tile x many-KV-tile single-O: seq_len=512 (4 BM=128 Q
            # tiles) with a long cached prefix -> num_keys=2048 (16 KV tiles),
            # cache_length=1536 (== num_keys - seq_len). The rows above stress
            # many Q tiles OR many KV tiles but never both at once; this locks
            # the combined regime (each of several Q tiles folding a long
            # serial single-O KV loop) into CI. Validated across a broad
            # seq_len x cache sweep (480/480) during CENG-282 debugging.
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=256,
                num_heads=16,
                nope_depth=192,
                v_depth=256,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
                cache_length=1536,
            ](512, 2048, ctx)

            # Prime / non-tile-aligned single-O shapes: odd partial tiles in Q,
            # KV, AND a non-aligned start_pos (cache) offset, to shake out any
            # boundary bug the round-number cases miss. seq_len=521 (prime) =
            # 4*128 + 9 -> 5 Q tiles (partial 9); num_keys=1031 (prime) =
            # 16*64 + 7 -> 17 single-O KV tiles (partial 7); cache_length=510
            # (= num_keys - seq_len), a non-tile-aligned causal offset.
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=256,
                num_heads=16,
                nope_depth=192,
                v_depth=256,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
                cache_length=510,
            ](521, 1031, ctx)
            # seq_len=1031 (prime) = 8*128 + 7 -> 9 Q tiles, with a prime
            # num_keys=1553 = 24*64 + 17 -> 25 single-O KV tiles; cache=522
            # (= num_keys - seq_len).
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=256,
                num_heads=16,
                nope_depth=192,
                v_depth=256,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
                cache_length=522,
            ](1031, 1553, ctx)

            # ---------------------------------------------------------------
            # num_heads variety on the single-O REAL GLM dims (base is 16).
            # Each num_heads is a distinct kernel instantiation, so keep the
            # list short and each shape modest. MLA caches ONE KV head per
            # token regardless of num_heads (KV_NUM_HEADS=1), so num_heads
            # only scales the Q/O head loop; the single-O KV-tile geometry is
            # num_heads-independent (proven across heads {16, 128} in the
            # CENG-282 sweep). This mirrors the MHA test's num_heads coverage
            # {1, 2, 3, 24, 32}, restricted to shapes that hit single-O.
            # ---------------------------------------------------------------
            print("=== REAL GLM nope=192 v=256, num_heads variety ===")
            # heads=1: smallest head loop, multi-KV-tile fresh prefill
            # (seq=128 -> 2 KV tiles at BN=64) so the single-O serial P@V loop
            # runs, not just the T==1 fast path.
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=256,
                num_heads=1,
                nope_depth=192,
                v_depth=256,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
            ](128, 128, ctx)
            # heads=3: odd head count + odd seq (15) with a cached prefix
            # (num_keys=128, cache=113) so a few queries fold many single-O
            # KV tiles across an odd number of heads.
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=256,
                num_heads=3,
                nope_depth=192,
                v_depth=256,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
                cache_length=113,
            ](15, 128, ctx)
            # heads=128: the production DP-attention head count. Heaviest
            # single-O instantiation, so run exactly ONE modest shape — a
            # short prefill appended after a cached prefix (seq=64,
            # num_keys=512, cache=448 -> 8 single-O KV tiles). This is the
            # real per-head-K/V decode-after-prefill geometry at production
            # width.
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=256,
                num_heads=128,
                nope_depth=192,
                v_depth=256,
                page_size=PAGE_SIZE,
                batch_size=1,
                diagnostic_bands=True,
                cache_length=448,
            ](64, 512, ctx)

            # ---------------------------------------------------------------
            # batch_size > 1 on single-O REAL GLM. The base rows are all
            # batch_size=1, which never exercises the cross-batch
            # input_row_offsets / cache_row_offsets tables or the per-batch
            # paged lookup-table striding (each batch occupies a contiguous
            # page range). batch is a compile-time param -> a new
            # instantiation per value; keep the list short and seq small
            # (batch=4 smallest) to bound runtime. Mirrors the MHA test's
            # batch_size {1, 4} coverage.
            # ---------------------------------------------------------------
            print("=== REAL GLM nope=192 v=256, batch_size > 1 ===")
            # batch=2, multi-KV-tile fresh prefill (seq=128 -> 2 KV tiles);
            # two independent per-batch page ranges + row-offset tables.
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=256,
                num_heads=16,
                nope_depth=192,
                v_depth=256,
                page_size=PAGE_SIZE,
                batch_size=2,
                diagnostic_bands=True,
            ](128, 128, ctx)
            # batch=4 with a cached prefix (seq=64, num_keys=128, cache=64):
            # four per-batch cache_row_offsets rows + a non-zero start_pos,
            # small seq to keep the 4x work bounded.
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=256,
                num_heads=16,
                nope_depth=192,
                v_depth=256,
                page_size=PAGE_SIZE,
                batch_size=4,
                diagnostic_bands=True,
                cache_length=64,
            ](64, 128, ctx)
