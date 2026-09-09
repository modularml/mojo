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
"""Correctness tests for the block-wide bitonic sort top-k.

Tests `persistent_topk_block` (`topk_bitonic.mojo`) in isolation:
- Output indices are in *descending* score order (scores[idx[0]] ≥ … ≥ scores[idx[K-1]]).
- The set of selected indices exactly matches the CPU reference top-K set.
- Multiple independent batch rows are sorted correctly.
- Padding (N < 2048) does not introduce spurious -1 indices inside [0, K).
- Partial top-K (K < N) selects only the true K-largest elements.
- Duplicate scores are handled without producing duplicate indices.
"""

from std.collections import Set
from max.gpu.host import DeviceContext
from std.math import max
from std.memory import bitcast
from std.random import seed
from std.testing import assert_equal, assert_true
from layout import TileTensor, row_major

from nn.topk_bitonic import (
    PERSISTENT_TOPK_MAX_N,
    _phi,
    persistent_topk_block,
    persistent_topk_block_split,
)


# ===----------------------------------------------------------------------=== #
# CPU reference
# ===----------------------------------------------------------------------=== #


def _cpu_topk_set(scores: List[Float32], K: Int) -> Set[Int]:
    """CPU reference: return the set of indices of the K largest values."""
    var N = len(scores)
    var order = List[Int](capacity=N)
    for i in range(N):
        order.append(i)

    for i in range(min(K, N)):
        var best = i
        for j in range(i + 1, N):
            if scores[order[j]] > scores[order[best]]:
                best = j
        var tmp = order[i]
        order[i] = order[best]
        order[best] = tmp

    var result = Set[Int]()
    for i in range(K):
        result.add(order[i])
    return result^


# ===----------------------------------------------------------------------=== #
# Core test helper
# ===----------------------------------------------------------------------=== #


def _run_and_check(
    ctx: DeviceContext,
    scores_host: List[Float32],
    N: Int,
    K: Int,
    label: String,
) raises:
    """Run persistent_topk_block on a single row and compare to CPU reference.

    Verifies:
    1. All output indices are in [0, N) or -1 (no OOB).
    2. The output indices are in non-increasing score order.
    3. The set of output indices matches the CPU reference set.
    4. No duplicate indices appear in the output.
    """
    assert K <= PERSISTENT_TOPK_MAX_N, "K exceeds champion width"
    assert K <= N, "K must be <= N"
    assert len(scores_host) == N, "scores_host length mismatch"

    # GPU buffers: 1 row of N scores → 1 row of K indices.
    var scores_dev = ctx.enqueue_create_buffer[.float32](N)
    var idxs_dev = ctx.enqueue_create_buffer[.int32](K)
    idxs_dev.enqueue_fill(Int32(-2))  # sentinel to catch unwritten slots

    with scores_dev.map_to_host() as buf:
        for i in range(N):
            buf[i] = Float32(scores_host[i])

    persistent_topk_block(
        ctx,
        rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_dev.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
        N,
        K,
        total_seq_len=1,
    )
    ctx.synchronize()

    # Copy back and validate.
    var idxs_host = ctx.enqueue_create_host_buffer[.int32](K)
    ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
    ctx.synchronize()

    var seen = Set[Int]()
    for k in range(K):
        var idx = Int(idxs_host[k])

        # 1. In-bounds.
        assert_true(
            idx >= 0 and idx < N,
            String("[", label, "] idx[", k, "]=", idx, " is OOB for N=", N),
        )

        # 2. Descending order (scores are non-increasing).
        if k > 0:
            var prev = Int(idxs_host[k - 1])
            assert_true(
                scores_host[idx] <= scores_host[prev],
                String(
                    "[",
                    label,
                    "] order violation at k=",
                    k,
                    ": scores[",
                    idx,
                    "]=",
                    scores_host[idx],
                    " > scores[",
                    prev,
                    "]=",
                    scores_host[prev],
                ),
            )

        # 3. No duplicates.
        assert_true(
            not (idx in seen),
            String("[", label, "] duplicate index ", idx, " at k=", k),
        )
        seen.add(idx)

    # 4. Output set matches the CPU reference set.
    var ref_set = _cpu_topk_set(scores_host, K)

    # GPU set == reference set (ties may order arbitrarily within equal values).
    for k in range(K):
        var idx = Int(idxs_host[k])
        assert_true(
            idx in ref_set,
            String(
                "[",
                label,
                "] idx[",
                k,
                "]=",
                idx,
                " not in reference top-K set",
            ),
        )
    for ref_idx in ref_set:
        assert_true(
            ref_idx in seen,
            String(
                "[", label, "] reference idx ", ref_idx, " missing from output"
            ),
        )

    _ = scores_dev
    _ = idxs_dev
    _ = idxs_host


# ===----------------------------------------------------------------------=== #
# Test cases
# ===----------------------------------------------------------------------=== #


def test_full_sort_n2048(ctx: DeviceContext) raises:
    """K=N=2048 — full sort; the exact bottleneck shape from the issue."""
    comptime N = 2048
    comptime K = 2048
    seed(42)
    var scores = List[Float32](capacity=N)
    for i in range(N):
        # Unique values: score[i] = float(N - i) so index 0 should rank first.
        scores.append(Float32(N - i))
    _run_and_check(ctx, scores, N, K, "full_sort_n2048")
    print("PASS test_full_sort_n2048")


def test_random_full_sort_n2048(ctx: DeviceContext) raises:
    """K=N=2048 with random float32 scores (may have near-duplicates)."""
    comptime N = 2048
    comptime K = 2048

    # Use a deterministic pseudo-random sequence via a simple LCG.
    var a: UInt32 = 1664525
    var c: UInt32 = 1013904223
    var state: UInt32 = 0xDEADBEEF
    var scores = List[Float32](capacity=N)
    for _ in range(N):
        state = a * state + c
        # Map to [-10, 10]
        var f = Float32(Int32(state)) / Float32(2**31) * 10.0
        scores.append(f)

    _run_and_check(ctx, scores, N, K, "random_full_sort_n2048")
    print("PASS test_random_full_sort_n2048")


def test_partial_topk_k16(ctx: DeviceContext) raises:
    """K=16, N=2048 — sparse selection (matches MSA block-indexer k=16)."""
    comptime N = 2048
    comptime K = 16
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(i))  # scores[N-1] is the max → should be idx 0
    _run_and_check(ctx, scores, N, K, "partial_topk_k16")
    print("PASS test_partial_topk_k16")


def test_partial_topk_k1024(ctx: DeviceContext) raises:
    """K=1024, N=2048 — half-sort."""
    comptime N = 2048
    comptime K = 1024
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(i * 3 % 1000))  # non-trivial pattern
    _run_and_check(ctx, scores, N, K, "partial_topk_k1024")
    print("PASS test_partial_topk_k1024")


def test_small_n_padded(ctx: DeviceContext) raises:
    """N=64, K=16 — heavily padded (2048 - 64 = 1984 -inf slots)."""
    comptime N = 64
    comptime K = 16
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(N - i) * 0.5)
    _run_and_check(ctx, scores, N, K, "small_n_padded")
    print("PASS test_small_n_padded")


def test_n_equals_k_small(ctx: DeviceContext) raises:
    """N=K=32 — small full sort; all padded slots must be ignored."""
    comptime N = 32
    comptime K = 32
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(i * 7 % 97))  # scattered values
    _run_and_check(ctx, scores, N, K, "n_equals_k_small")
    print("PASS test_n_equals_k_small")


def test_n_equals_k_power_of_2(ctx: DeviceContext) raises:
    """N=K=512 — mid-size full sort; also a power of 2."""
    comptime N = 512
    comptime K = 512
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(i))
    _run_and_check(ctx, scores, N, K, "n_equals_k_512")
    print("PASS test_n_equals_k_power_of_2")


def test_duplicate_scores(ctx: DeviceContext) raises:
    """All scores identical — every index is a valid answer, no duplicates allowed.
    """
    comptime N = 256
    comptime K = 64
    var scores = List[Float32](capacity=N)
    for _ in range(N):
        scores.append(Float32(1.0))
    _run_and_check(ctx, scores, N, K, "duplicate_scores")
    print("PASS test_duplicate_scores")


def test_two_valued_scores(ctx: DeviceContext) raises:
    """Scores are 0.0 or 1.0 alternating — tests tie-breaking within a value class.
    """
    comptime N = 128
    comptime K = 32
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(1.0) if i % 2 == 0 else Float32(0.0))
    _run_and_check(ctx, scores, N, K, "two_valued_scores")
    print("PASS test_two_valued_scores")


def test_negative_scores(ctx: DeviceContext) raises:
    """All negative scores — top-K should select the least negative."""
    comptime N = 128
    comptime K = 16
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(
            Float32(-Float32(i + 1))
        )  # scores[0]=-1 is max, scores[127]=-128 is min
    _run_and_check(ctx, scores, N, K, "negative_scores")
    print("PASS test_negative_scores")


def test_multi_batch(ctx: DeviceContext) raises:
    """Multiple batch rows — each must be sorted independently and correctly."""
    comptime N = 256
    comptime K = 32
    comptime BATCH = 4

    var scores_dev = ctx.enqueue_create_buffer[.float32](BATCH * N)
    var idxs_dev = ctx.enqueue_create_buffer[.int32](BATCH * K)
    idxs_dev.enqueue_fill(Int32(-2))

    # Fill each row with a distinct pattern.
    with scores_dev.map_to_host() as buf:
        for b in range(BATCH):
            for i in range(N):
                # Row b: scores[i] = (b+1) * (N - i), so each row has the same
                # top-K structure (indices 0..K-1 are the top K).
                buf[b * N + i] = Float32((b + 1) * (N - i))

    persistent_topk_block(
        ctx,
        rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_dev.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
        N,
        K,
        total_seq_len=BATCH,
    )
    ctx.synchronize()

    var idxs_host = ctx.enqueue_create_host_buffer[.int32](BATCH * K)
    ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
    ctx.synchronize()

    for b in range(BATCH):
        # For each row, build the score array for validation.
        var row_scores = List[Float32](capacity=N)
        for i in range(N):
            row_scores.append(Float32((b + 1) * (N - i)))

        var seen = Set[Int]()
        for k in range(K):
            var idx = Int(idxs_host[b * K + k])
            assert_true(
                idx >= 0 and idx < N,
                String("multi_batch row ", b, " k=", k, " OOB idx=", idx),
            )
            assert_true(
                not (idx in seen),
                String(
                    "multi_batch row ", b, " duplicate idx=", idx, " at k=", k
                ),
            )
            seen.add(idx)

            # Descending order check.
            if k > 0:
                var prev = Int(idxs_host[b * K + k - 1])
                assert_true(
                    row_scores[idx] <= row_scores[prev],
                    String(
                        "multi_batch row ",
                        b,
                        " order violation at k=",
                        k,
                    ),
                )

        # Every index from [0, K) should appear (they are the K largest).
        for expected in range(K):
            assert_true(
                expected in seen,
                String(
                    "multi_batch row ", b, " missing expected idx ", expected
                ),
            )

    _ = scores_dev
    _ = idxs_dev
    _ = idxs_host
    print("PASS test_multi_batch")


def test_sorted_input_already_descending(ctx: DeviceContext) raises:
    """Input is already sorted descending — bitonic sort must not corrupt it."""
    comptime N = 2048
    comptime K = 64
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(N - i))
    _run_and_check(ctx, scores, N, K, "sorted_desc")
    print("PASS test_sorted_input_already_descending")


def test_sorted_input_ascending(ctx: DeviceContext) raises:
    """Input is sorted ascending — the reverse of the desired output."""
    comptime N = 2048
    comptime K = 64
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(i))  # scores[2047] is the max
    _run_and_check(ctx, scores, N, K, "sorted_asc")
    print("PASS test_sorted_input_ascending")


def test_single_element(ctx: DeviceContext) raises:
    """N=K=1 — degenerate case."""
    comptime N = 1
    comptime K = 1
    var scores = List[Float32](capacity=N)
    scores.append(Float32(42.0))
    _run_and_check(ctx, scores, N, K, "single_element")
    print("PASS test_single_element")


# ===----------------------------------------------------------------------=== #
# Streaming path (N > 2048) — the GLM 5.x long-context / prefill regime.
# ===----------------------------------------------------------------------=== #


def test_streaming_n16384_random(ctx: DeviceContext) raises:
    """N=16384 (8 tiles), K=2048 random — the long-context decode shape."""
    comptime N = 16384
    comptime K = 2048
    var a: UInt32 = 1664525
    var c: UInt32 = 1013904223
    var state: UInt32 = 0x12345678
    var scores = List[Float32](capacity=N)
    for _ in range(N):
        state = a * state + c
        scores.append(Float32(Int32(state)) / Float32(2**31) * 100.0)
    _run_and_check(ctx, scores, N, K, "streaming_n16384_random")
    print("PASS test_streaming_n16384_random")


def test_streaming_n16006_nonmultiple(ctx: DeviceContext) raises:
    """N=16006 (not a multiple of the 2048 tile), K=2048 — partial last tile."""
    comptime N = 16006
    comptime K = 2048
    var a: UInt32 = 22695477
    var c: UInt32 = 1
    var state: UInt32 = 0xCAFEBABE
    var scores = List[Float32](capacity=N)
    for _ in range(N):
        state = a * state + c
        scores.append(Float32(Int32(state)) / Float32(2**31) * 7.0)
    _run_and_check(ctx, scores, N, K, "streaming_n16006_nonmultiple")
    print("PASS test_streaming_n16006_nonmultiple")


def test_streaming_n163840_ascending(ctx: DeviceContext) raises:
    """N=163840 (80 tiles, GLM max context), K=2048 — the max-shape stress."""
    comptime N = 163840
    comptime K = 2048
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(i))
    _run_and_check(ctx, scores, N, K, "streaming_n163840_ascending")
    print("PASS test_streaming_n163840_ascending")


def test_streaming_masked_and_ties(ctx: DeviceContext) raises:
    """N=8192 with a masked (-1e30) prefix and heavy ties (causal-mask regime).
    """
    comptime N = 8192
    comptime K = 2048
    var scores = List[Float32](capacity=N)
    for i in range(N):
        if i < N // 2:
            scores.append(Float32(-1.0e30))
        else:
            scores.append(Float32(1.0) if (i % 3 == 0) else Float32(2.0))
    _run_and_check(ctx, scores, N, K, "streaming_masked_and_ties")
    print("PASS test_streaming_masked_and_ties")


# ===----------------------------------------------------------------------=== #
# Split path (`persistent_topk_block_split`) — the low-row / long-context
# decode regime that fans the streaming fold across `rows * S` blocks.
# ===----------------------------------------------------------------------=== #


def _check_topk_row(
    scores_row: List[Float32],
    idxs: List[Int],
    N: Int,
    K: Int,
    label: String,
    row: Int,
) raises:
    """Tie-robust top-K validation of one row.

    Verifies (1) indices in `[0, N)`, distinct, exactly `K`; (2) non-increasing
    score order; (3) every *non-selected* score is `<= min(selected scores)` —
    the exact top-K condition, which (unlike an exact index-set match) admits
    any valid tie-break at the K-th boundary.
    """
    var seen = Set[Int]()
    var min_sel = Float32.MAX
    for k in range(K):
        var idx = idxs[k]
        assert_true(
            idx >= 0 and idx < N,
            String("[", label, "] row ", row, " idx[", k, "]=", idx, " OOB"),
        )
        assert_true(
            not (idx in seen),
            String("[", label, "] row ", row, " duplicate idx=", idx),
        )
        seen.add(idx)
        if k > 0:
            var prev = idxs[k - 1]
            assert_true(
                scores_row[idx] <= scores_row[prev],
                String("[", label, "] row ", row, " order violation at k=", k),
            )
        min_sel = min(min_sel, scores_row[idx])

    assert_true(
        len(seen) == K,
        String("[", label, "] row ", row, " selected ", len(seen), " != K"),
    )
    for i in range(N):
        if not (i in seen):
            assert_true(
                scores_row[i] <= min_sel,
                String(
                    "[",
                    label,
                    "] row ",
                    row,
                    " non-selected idx ",
                    i,
                    " score ",
                    scores_row[i],
                    " exceeds selected min ",
                    min_sel,
                ),
            )


def _run_and_check_split(
    ctx: DeviceContext,
    scores_host: List[Float32],  # B * N flat, row-major
    N: Int,
    K: Int,
    B: Int,
    label: String,
) raises:
    """Run `persistent_topk_block_split` over `B` rows and validate each row."""
    assert K <= PERSISTENT_TOPK_MAX_N, "K exceeds champion width"
    assert len(scores_host) == B * N, "scores_host length mismatch"

    var scores_dev = ctx.enqueue_create_buffer[.float32](B * N)
    var idxs_dev = ctx.enqueue_create_buffer[.int32](B * K)
    idxs_dev.enqueue_fill(Int32(-2))

    with scores_dev.map_to_host() as buf:
        for i in range(B * N):
            buf[i] = Float32(scores_host[i])

    persistent_topk_block_split[ordered=True, deterministic=True](
        ctx,
        rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_dev.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
        N,
        K,
        total_seq_len=B,
    )
    ctx.synchronize()

    var idxs_host = ctx.enqueue_create_host_buffer[.int32](B * K)
    ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
    ctx.synchronize()

    for b in range(B):
        var row_scores = List[Float32](capacity=N)
        for i in range(N):
            row_scores.append(scores_host[b * N + i])
        var row_idxs = List[Int](capacity=K)
        for k in range(K):
            row_idxs.append(Int(idxs_host[b * K + k]))
        _check_topk_row(row_scores, row_idxs, N, K, label, b)

    _ = scores_dev
    _ = idxs_dev
    _ = idxs_host


def _lcg_scores(B: Int, N: Int, sd: UInt32, scale: Float32) -> List[Float32]:
    var a: UInt32 = 1664525
    var c: UInt32 = 1013904223
    var state = sd
    var scores = List[Float32](capacity=B * N)
    for _ in range(B * N):
        state = a * state + c
        scores.append(Float32(Int32(state)) / Float32(2**31) * scale)
    return scores^


def test_split_decode_long(ctx: DeviceContext) raises:
    """Rows=8, N=32769 (17 tiles, non-multiple), K=2048 — the decode-long shape.
    """
    comptime N = 32769
    comptime K = 2048
    comptime B = 8
    var scores = _lcg_scores(B, N, 0x1234ABCD, 100.0)
    _run_and_check_split(ctx, scores, N, K, B, "split_decode_long")
    print("PASS test_split_decode_long")


def test_split_decode_mtp(ctx: DeviceContext) raises:
    """Rows=16, N=8193 (5 tiles), K=2048 — the MTP decode shape."""
    comptime N = 8193
    comptime K = 2048
    comptime B = 16
    var scores = _lcg_scores(B, N, 0xCAFED00D, 50.0)
    _run_and_check_split(ctx, scores, N, K, B, "split_decode_mtp")
    print("PASS test_split_decode_mtp")


def test_split_max_context(ctx: DeviceContext) raises:
    """Rows=2, N=163840 (80 tiles, GLM max context), K=2048."""
    comptime N = 163840
    comptime K = 2048
    comptime B = 2
    var scores = _lcg_scores(B, N, 0xBADF00D5, 100.0)
    _run_and_check_split(ctx, scores, N, K, B, "split_max_context")
    print("PASS test_split_max_context")


def test_split_partial_k(ctx: DeviceContext) raises:
    """Rows=4, N=8193, K=512 — split with K < champion width."""
    comptime N = 8193
    comptime K = 512
    comptime B = 4
    var scores = _lcg_scores(B, N, 0x0FF1CE55, 30.0)
    _run_and_check_split(ctx, scores, N, K, B, "split_partial_k")
    print("PASS test_split_partial_k")


def test_split_masked_and_ties(ctx: DeviceContext) raises:
    """Rows=4, N=8193 with a masked prefix + heavy ties — the causal regime.

    The tie-robust check accepts any valid boundary tie-break, so the split's
    per-slice tie order need not match a specific reference permutation.
    """
    comptime N = 8193
    comptime K = 2048
    comptime B = 4
    var scores = List[Float32](capacity=B * N)
    for _b in range(B):
        for i in range(N):
            if i < N // 2:
                scores.append(Float32(-1.0e30))
            else:
                scores.append(Float32(1.0) if (i % 3 == 0) else Float32(2.0))
    _run_and_check_split(ctx, scores, N, K, B, "split_masked_and_ties")
    print("PASS test_split_masked_and_ties")


# The tree phase-2 reduces S partials with fan-in 5; these rows=8 shapes pin S
# (= min(num_tiles, ceildiv(2*sm_count, rows))) at values that exercise the
# reduction at several fan-out patterns: 6 (fan-in+1 edge), 8 (power of two),
# 13 (prime, uneven final group), 17 (non-power-of-two). N = num_tiles*2048 - 1
# keeps the last tile partial too.


def test_split_tree_s6(ctx: DeviceContext) raises:
    """Rows=8, N=12287 (6 tiles) -> S=6, one reduce round (6 -> 2) + final."""
    comptime N = 12287
    comptime K = 2048
    comptime B = 8
    var scores = _lcg_scores(B, N, 0x00516006, 80.0)
    _run_and_check_split(ctx, scores, N, K, B, "split_tree_s6")
    print("PASS test_split_tree_s6")


def test_split_tree_s8(ctx: DeviceContext) raises:
    """Rows=8, N=16383 (8 tiles) -> S=8, one reduce round (8 -> 2) + final."""
    comptime N = 16383
    comptime K = 2048
    comptime B = 8
    var scores = _lcg_scores(B, N, 0x00518008, 90.0)
    _run_and_check_split(ctx, scores, N, K, B, "split_tree_s8")
    print("PASS test_split_tree_s8")


def test_split_tree_s13(ctx: DeviceContext) raises:
    """Rows=8, N=26623 (13 tiles) -> S=13, reduce (13 -> 3) + final of 3."""
    comptime N = 26623
    comptime K = 2048
    comptime B = 8
    var scores = _lcg_scores(B, N, 0x00513013, 120.0)
    _run_and_check_split(ctx, scores, N, K, B, "split_tree_s13")
    print("PASS test_split_tree_s13")


# ===----------------------------------------------------------------------=== #
# Odd-N / odd-alignment coverage for the non-split kernels.
#
# The vectorized score load emits a 128-bit load only when the row base is
# 16B-aligned; odd `N` makes `token*N` non-aligned for most rows, so these
# multi-row odd-N shapes exercise the scalar fallback (a misaligned 128-bit
# load would fault). `test_split_decode_long` covers the same for the split
# kernels; these cover `persistent_topk_block`'s 2048 and streaming kernels,
# which every other test only drives at `total_seq_len=1` (row 0, always
# aligned).
# ===----------------------------------------------------------------------=== #


def _run_and_check_block_multirow(
    ctx: DeviceContext,
    scores_host: List[Float32],  # B * N flat, row-major
    N: Int,
    K: Int,
    B: Int,
    label: String,
) raises:
    """Run `persistent_topk_block` over `B` rows and validate each row."""
    assert K <= PERSISTENT_TOPK_MAX_N, "K exceeds champion width"
    assert len(scores_host) == B * N, "scores_host length mismatch"

    var scores_dev = ctx.enqueue_create_buffer[.float32](B * N)
    var idxs_dev = ctx.enqueue_create_buffer[.int32](B * K)
    idxs_dev.enqueue_fill(Int32(-2))
    with scores_dev.map_to_host() as buf:
        for i in range(B * N):
            buf[i] = Float32(scores_host[i])

    persistent_topk_block(
        ctx,
        rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_dev.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
        N,
        K,
        total_seq_len=B,
    )
    ctx.synchronize()

    var idxs_host = ctx.enqueue_create_host_buffer[.int32](B * K)
    ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
    ctx.synchronize()

    for b in range(B):
        var row_scores = List[Float32](capacity=N)
        for i in range(N):
            row_scores.append(scores_host[b * N + i])
        var row_idxs = List[Int](capacity=K)
        for k in range(K):
            row_idxs.append(Int(idxs_host[b * K + k]))
        _check_topk_row(row_scores, row_idxs, N, K, label, b)

    _ = scores_dev
    _ = idxs_dev
    _ = idxs_host


def test_block_odd_n_multirow(ctx: DeviceContext) raises:
    """N=1025 (odd, <=2048), rows=8, K=512 — 2048 kernel, misaligned bases.

    `token*1025` is non-16B-aligned for most rows, forcing the scalar-fallback
    score load in `_persistent_topk_2048_kernel`.
    """
    comptime N = 1025
    comptime K = 512
    comptime B = 8
    var scores = _lcg_scores(B, N, 0x0DD00001, 40.0)
    _run_and_check_block_multirow(ctx, scores, N, K, B, "block_odd_n_multirow")
    print("PASS test_block_odd_n_multirow")


def test_block_streaming_odd_n_multirow(ctx: DeviceContext) raises:
    """N=8193 (odd, >2048), rows=8, K=2048 — streaming kernel, misaligned bases.

    Drives `persistent_topk_block` (not the split launcher), so the streaming
    fold runs one block per row with odd `token*N` bases -> scalar-fallback
    score load. Complements `test_split_decode_long` (split path's odd-N loads).
    """
    comptime N = 8193
    comptime K = 2048
    comptime B = 8
    var scores = _lcg_scores(B, N, 0x0DD08193, 60.0)
    _run_and_check_block_multirow(ctx, scores, N, K, B, "block_streaming_odd_n")
    print("PASS test_block_streaming_odd_n_multirow")


# ===----------------------------------------------------------------------=== #
# `ordered=False`: the same set, without the ranking pass
# ===----------------------------------------------------------------------=== #


def _expected_topk_set(scores_row: List[Float32], K: Int) -> Set[Int]:
    """Exact top-`K` under the kernel's total order: descending score, then
    ascending column.

    Found by bisecting the order-preserving key rather than sorting, so it is
    O(N log R) and stays usable at the largest shapes -- a per-element rank would
    be O(N^2) and a sort drags in an ordering dependency this oracle exists to
    avoid. Ties are resolved exactly, which is the point: it pins which members of
    a plateau straddling `K` must be chosen, where `_cpu_topk_set` admits any of
    them.
    """
    var N = len(scores_row)
    var keys = List[UInt32](capacity=N)
    for i in range(N):
        keys.append(_phi(scores_row[i]))

    # Largest threshold `t` whose at-or-above count still reaches `K`.
    var lo = UInt32(0)
    var hi = UInt32(0xFFFFFFFF)
    while lo < hi:
        var mid = lo + (hi - lo) // 2 + 1
        var cnt = 0
        for i in range(N):
            if keys[i] >= mid:
                cnt += 1
        if cnt >= K:
            lo = mid
        else:
            hi = mid - 1
    var t = lo

    var res = Set[Int]()
    var n_gt = 0
    for i in range(N):
        if keys[i] > t:
            res.add(i)
            n_gt += 1
    # The rest come from the threshold value itself, lowest column first.
    var need = K - n_gt
    for i in range(N):
        if need == 0:
            break
        if keys[i] == t:
            res.add(i)
            need -= 1
    return res^


def _check_set_row(
    scores_row: List[Float32],
    idxs: List[Int],
    N: Int,
    K: Int,
    label: String,
    row: Int,
) raises:
    """Assert one row's indices are exactly the expected top-`K`, as a set.

    Order is deliberately not asserted here: `ordered=False` promises membership
    and determinism, and only shapes with a cheaper path take one -- the rest
    legitimately return the ordered result, which satisfies the weaker promise
    too. Ordering is checked separately, for the ordered contract.
    """
    var seen = Set[Int]()
    for k in range(K):
        var idx = idxs[k]
        assert_true(
            idx >= 0 and idx < N,
            String("[", label, "] row ", row, " idx[", k, "]=", idx, " OOB"),
        )
        assert_true(
            not (idx in seen),
            String("[", label, "] row ", row, " duplicate idx=", idx),
        )
        seen.add(idx)
    assert_equal(
        len(seen), K, String("[", label, "] row ", row, " wrong count")
    )

    var want = _expected_topk_set(scores_row, K)
    for c in want:
        assert_true(
            c in seen,
            String(
                "[",
                label,
                "] row ",
                row,
                " missing column ",
                c,
                " -- selection or tie-break differs from the oracle",
            ),
        )


def _check_score_valid_row(
    scores_row: List[Float32],
    idxs: List[Int],
    N: Int,
    K: Int,
    label: String,
    row: Int,
) raises:
    """Assert one row is *a* valid top-`K` by score, without pinning which one.

    Three things, which together are the whole contract and no more:

    1. `K` distinct columns, all in `[0, N)`;
    2. every column scoring **strictly above** the `K`-th largest is present --
       those are forced, no contract makes them optional;
    3. the rest tie the `K`-th largest exactly.

    Checking only 1 and 3 admits a row that drops higher-scoring columns outright,
    because the threshold is itself one of the selected scores -- so a row made
    entirely of columns tied at it satisfies "at least the K-th largest" while
    leaving out everything above. Which *tied* columns are chosen stays unasserted,
    because that is the part the mode really does leave free.
    """
    var seen = Set[Int]()
    for k in range(K):
        var idx = idxs[k]
        assert_true(
            idx >= 0 and idx < N,
            String("[", label, "] row ", row, " idx[", k, "]=", idx, " OOB"),
        )
        assert_true(
            not (idx in seen),
            String("[", label, "] row ", row, " duplicate idx=", idx),
        )
        seen.add(idx)
    assert_equal(
        len(seen), K, String("[", label, "] row ", row, " wrong count")
    )

    # The K-th largest score, taken from the oracle's own set.
    var want = _expected_topk_set(scores_row, K)
    var thresh = Float32.MAX
    for c in want:
        if scores_row[c] < thresh:
            thresh = scores_row[c]

    var n_at = 0
    for c in seen:
        assert_true(
            scores_row[c] >= thresh,
            String(
                "[",
                label,
                "] row ",
                row,
                " column ",
                c,
                " scores below the K-th largest",
            ),
        )
        if scores_row[c] == thresh:
            n_at += 1

    var n_above = 0
    for c in range(N):
        if scores_row[c] > thresh:
            n_above += 1
            assert_true(
                c in seen,
                String(
                    "[",
                    label,
                    "] row ",
                    row,
                    " column ",
                    c,
                    " scores above the K-th largest and was not selected",
                ),
            )
    assert_equal(
        n_at,
        K - n_above,
        String(
            "[",
            label,
            "] row ",
            row,
            " wrong count tied at the K-th largest",
        ),
    )


def _run_split[
    ordered: Bool = True, deterministic: Bool = True
](
    ctx: DeviceContext,
    scores_host: List[Float32],
    N: Int,
    K: Int,
    B: Int,
) raises -> List[Int]:
    """One launch of the given contract, returned flat.

    Parameterized rather than written once per contract: the contracts differ only
    in what they promise about order, so a launch that differed anywhere else would
    be comparing two things at once.
    """
    var scores_dev = ctx.enqueue_create_buffer[.float32](B * N)
    var idxs_dev = ctx.enqueue_create_buffer[.int32](B * K)
    idxs_dev.enqueue_fill(Int32(-2))
    with scores_dev.map_to_host() as buf:
        for i in range(B * N):
            buf[i] = Float32(scores_host[i])

    persistent_topk_block_split[ordered=ordered, deterministic=deterministic](
        ctx,
        rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_dev.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
        N,
        K,
        total_seq_len=B,
    )
    ctx.synchronize()

    var host = ctx.enqueue_create_host_buffer[.int32](B * K)
    ctx.enqueue_copy(dst_buf=host, src_buf=idxs_dev)
    ctx.synchronize()
    var out = List[Int](capacity=B * K)
    for i in range(B * K):
        out.append(Int(host[i]))
    _ = scores_dev
    _ = idxs_dev
    _ = host
    return out^


def _check_both(
    ctx: DeviceContext,
    scores_host: List[Float32],
    N: Int,
    K: Int,
    B: Int,
    label: String,
) raises:
    """The full battery on one input: both contracts, cross-checked.

    1. `ordered=False` selects exactly the oracle's set, every row.
    2. `ordered=True` selects the same set *and* is correctly ordered -- so the
       ordered path is gated at least as hard here as the new one.
    3. The two modes agree as sets, which is the claim that justifies the mode.
    4. `ordered=False` is byte-stable across launches, which an implementation
       that places tied keys by an atomic race cannot offer.
    5. `deterministic=False` -- which adopts that race deliberately -- returns *a*
       valid top-`K` by score on *both* of two launches. It is held to the
       strongest predicate that does not over-promise (see
       `_check_score_valid_row`): the columns above the threshold are forced, only
       the tied ones are free. The two launches are allowed to disagree, so that
       is reported rather than asserted -- forbidding it would forbid the mode.
    """
    assert K <= PERSISTENT_TOPK_MAX_N, "K exceeds champion width"
    assert len(scores_host) == B * N, "scores_host length mismatch"

    var unord = _run_split[False, True](ctx, scores_host, N, K, B)
    var ord = _run_split(ctx, scores_host, N, K, B)
    var unord_nd = _run_split[False, False](ctx, scores_host, N, K, B)
    var unord_nd2 = _run_split[False, False](ctx, scores_host, N, K, B)

    for b in range(B):
        var row_scores = List[Float32](capacity=N)
        for i in range(N):
            row_scores.append(scores_host[b * N + i])

        var u_row = List[Int](capacity=K)
        var o_row = List[Int](capacity=K)
        var r_row = List[Int](capacity=K)
        for k in range(K):
            u_row.append(unord[b * K + k])
            o_row.append(ord[b * K + k])
            r_row.append(unord_nd[b * K + k])

        _check_set_row(row_scores, u_row, N, K, String(label, ":unord"), b)
        _check_set_row(row_scores, o_row, N, K, String(label, ":ord"), b)
        var r2_row = List[Int](capacity=K)
        for k in range(K):
            r2_row.append(unord_nd2[b * K + k])
        _check_score_valid_row(
            row_scores, r_row, N, K, String(label, ":unord_nd"), b
        )
        _check_score_valid_row(
            row_scores, r2_row, N, K, String(label, ":unord_nd2"), b
        )

        # The ordered contract: non-increasing score, ties by ascending column.
        for k in range(1, K):
            var prev = o_row[k - 1]
            var cur = o_row[k]
            var sp = row_scores[prev]
            var sc = row_scores[cur]
            assert_true(
                sp > sc or (sp == sc and prev < cur),
                String(
                    "[",
                    label,
                    ":ord] row ",
                    b,
                    " order violation at k=",
                    k,
                    ": col ",
                    prev,
                    " then ",
                    cur,
                ),
            )

        # Same membership from both modes.
        var uset = Set[Int]()
        for k in range(K):
            uset.add(u_row[k])
        for k in range(K):
            assert_true(
                o_row[k] in uset,
                String(
                    "[",
                    label,
                    "] row ",
                    b,
                    " ordered picked column ",
                    o_row[k],
                    " that unordered did not",
                ),
            )

    var again = _run_split[False, True](ctx, scores_host, N, K, B)
    for i in range(B * K):
        assert_equal(
            unord[i],
            again[i],
            String("[", label, "] slot ", i, " differs between launches"),
        )


def test_unordered_resident_band(ctx: DeviceContext) raises:
    """The resident band, with lengths either side of the `bin_digit` crossover.
    """
    comptime K = 2048
    comptime B = 48
    for N in [2049, 2432, 2560, 4097, 8192]:
        var scores = _lcg_scores(B, N, 0x51EED000 + UInt32(N), 100.0)
        _check_both(ctx, scores, N, K, B, String("band_n", N))
    print("PASS test_unordered_resident_band")


def test_unord_nd_predicate_bites() raises:
    """The unord_nd predicate must reject the row its weaker form used to accept.

    A checker that cannot fail is not a gate, and this one silently could not: see
    `_check_score_valid_row` for the row that satisfies "at least the K-th largest"
    while omitting every column above it. Host-only, so it costs nothing to keep.
    """
    comptime N = 8
    comptime K = 3
    var sc = List[Float32](capacity=N)
    sc.append(Float32(10.0))
    for _ in range(N - 1):
        sc.append(Float32(5.0))

    # Columns tied at the threshold only, with the forced one left out.
    var bad = List[Int](capacity=K)
    bad.append(1)
    bad.append(2)
    bad.append(3)
    var rejected = False
    try:
        _check_score_valid_row(sc, bad, N, K, "selftest", 0)
    except:
        rejected = True
    assert_true(
        rejected,
        (
            "unord_nd predicate accepted a row that drops a column above the"
            " K-th largest"
        ),
    )

    # The same row with the forced column present must pass.
    var good = List[Int](capacity=K)
    good.append(0)
    good.append(1)
    good.append(2)
    _check_score_valid_row(sc, good, N, K, "selftest", 0)
    print("PASS test_unord_nd_predicate_bites")


def test_unordered_wide_payload_band(ctx: DeviceContext) raises:
    """The wide register payload: both its edges, both row regimes, maximal ties.

    A row narrow enough is held in registers when the rows under-fill the GPU and
    streamed otherwise, so every case here runs two different kernels across the
    two row counts. The lengths straddle each payload width's ceiling from both
    sides, and one of them is not a multiple of the vector width, which forces the
    unaligned tail of the load rather than the vectorised path.
    """
    comptime K = 2048
    for N in [8192, 8193, 16384, 16385, 16388]:
        for B in [8, 160]:
            var scores = _lcg_scores(B, N, 0x5EED0000 + UInt32(N), 100.0)
            _check_both(ctx, scores, N, K, B, String("wide_n", N, "_b", B))

    # Every column ties, so the plateau cut alone decides the answer and a
    # miscounted `n_above` shows up as a short row rather than a wrong score.
    comptime NEQ = 12288
    comptime BEQ = 8
    var eq = List[Float32](capacity=BEQ * NEQ)
    for _ in range(BEQ * NEQ):
        eq.append(Float32(1.5))
    _check_both(ctx, eq, NEQ, K, BEQ, "wide_all_equal")

    # `K` at both extremes, on a row the wide payload holds.
    for KK in [1, 2048]:
        var sc = _lcg_scores(8, 14336, 0xC0DE0000 + UInt32(KK), 100.0)
        _check_both(ctx, sc, 14336, KK, 8, String("wide_k", KK))
    print("PASS test_unordered_wide_payload_band")


def test_unordered_prime_lengths(ctx: DeviceContext) raises:
    """Prime `N`, so no tile, vector or bin width divides the row.

    8191 is also not a multiple of four, which is what forces the scalar tail of
    the vectorised load rather than the `float4` path.
    """
    comptime K = 2048
    comptime B = 24
    # The last two sit inside the wide payload's band, which the rest of this list
    # brackets without entering, and one of them just below its ceiling.
    for N in [2053, 3079, 4099, 6151, 8191, 12289, 16381]:
        var scores = _lcg_scores(B, N, 0x9E3779B1 + UInt32(N), 100.0)
        _check_both(ctx, scores, N, K, B, String("prime_n", N))
    print("PASS test_unordered_prime_lengths")


def test_unordered_k_edges(ctx: DeviceContext) raises:
    """`K` at its extremes, including 1, 2 and `K = N - 1`."""
    comptime B = 12
    comptime N = 4099
    for K in [1, 2, 17, 999, 1024, 2047, 2048]:
        var scores = _lcg_scores(B, N, 0xC0FFEE00 + UInt32(K), 50.0)
        _check_both(ctx, scores, N, K, B, String("kedge_k", K))
    print("PASS test_unordered_k_edges")


def test_unordered_row_counts(ctx: DeviceContext) raises:
    """Row counts around the SM count, which is what picks resident vs streaming.
    """
    comptime N = 4097
    comptime K = 2048
    for B in [1, 2, 47, 147, 200]:
        var scores = _lcg_scores(B, N, 0xB105F00D + UInt32(B), 100.0)
        _check_both(ctx, scores, N, K, B, String("rows_b", B))
    print("PASS test_unordered_row_counts")


def test_unordered_all_equal(ctx: DeviceContext) raises:
    """Every score identical, so the whole row is one tie plateau.

    The answer is then forced to columns `0 .. K-1`: ties go to the lower column,
    and "the K largest" alone would not pin it.
    """
    comptime N = 4097
    comptime K = 2048
    comptime B = 4
    var scores = List[Float32](capacity=B * N)
    for _ in range(B * N):
        scores.append(Float32(0.25))
    _check_both(ctx, scores, N, K, B, "all_equal")

    var got = _run_split[False, True](ctx, scores, N, K, B)
    for b in range(B):
        var seen = Set[Int]()
        for k in range(K):
            seen.add(got[b * K + k])
        for c in range(K):
            assert_true(
                c in seen,
                String(
                    "all_equal row ",
                    b,
                    ": column ",
                    c,
                    " must be selected when every score is equal",
                ),
            )
    print("PASS test_unordered_all_equal")


def test_unordered_plateau_straddles_k(ctx: DeviceContext) raises:
    """A plateau of equal scores spanning the `K` boundary.

    `0 .. K-201` score high, `K-200 .. K+399` share one middle score, the rest
    are low. The cut therefore lands *inside* the plateau, so only the tie-break
    decides which 200 of its 600 members are in.
    """
    comptime N = 4097
    comptime K = 2048
    comptime B = 4
    comptime hi_end = K - 200
    comptime plateau_end = K + 400
    var scores = List[Float32](capacity=B * N)
    for _ in range(B):
        for c in range(N):
            if c < hi_end:
                scores.append(Float32(10.0))
            elif c < plateau_end:
                scores.append(Float32(5.0))
            else:
                scores.append(Float32(-1.0))
    _check_both(ctx, scores, N, K, B, "plateau_straddle")

    var got = _run_split[False, True](ctx, scores, N, K, B)
    for b in range(B):
        var seen = Set[Int]()
        for k in range(K):
            seen.add(got[b * K + k])
        for c in range(K):
            assert_true(
                c in seen,
                String("plateau row ", b, ": column ", c, " must win"),
            )
        for c in range(K, plateau_end):
            assert_true(
                not (c in seen),
                String(
                    "plateau row ",
                    b,
                    ": column ",
                    c,
                    (
                        " ties the boundary but has a higher index, so it must"
                        " lose"
                    ),
                ),
            )
    print("PASS test_unordered_plateau_straddles_k")


def test_unordered_plateau_edges(ctx: DeviceContext) raises:
    """Plateaus that end exactly at `K`, and that start exactly at `K`.

    Both are off-by-one bait: the first needs the whole plateau in, the second
    needs all of it out.
    """
    comptime N = 4097
    comptime K = 2048
    comptime B = 4
    for start in [K - 300, K]:
        var scores = List[Float32](capacity=B * N)
        for _ in range(B):
            for c in range(N):
                if c < start:
                    scores.append(Float32(9.0))
                elif c < (K if start < K else K + 300):
                    scores.append(Float32(4.0))
                else:
                    scores.append(Float32(-2.0))
        _check_both(ctx, scores, N, K, B, String("plateau_edge", start))
    print("PASS test_unordered_plateau_edges")


def test_unordered_k1_all_equal(ctx: DeviceContext) raises:
    """`K = 1` on an all-equal row: the answer is exactly column 0."""
    comptime N = 3079
    comptime K = 1
    comptime B = 8
    var scores = List[Float32](capacity=B * N)
    for _ in range(B * N):
        scores.append(Float32(-7.5))
    _check_both(ctx, scores, N, K, B, "k1_all_equal")
    var got = _run_split[False, True](ctx, scores, N, K, B)
    for b in range(B):
        assert_equal(
            got[b],
            0,
            String("k1_all_equal row ", b, ": must pick column 0"),
        )
    print("PASS test_unordered_k1_all_equal")


def test_unordered_two_valued(ctx: DeviceContext) raises:
    """Two distinct scores only, so the digit has almost nothing to split on."""
    comptime N = 2560
    comptime K = 2048
    comptime B = 8
    var scores = List[Float32](capacity=B * N)
    for b in range(B):
        for c in range(N):
            scores.append(Float32(1.0) if (c + b) % 3 == 0 else Float32(0.0))
    _check_both(ctx, scores, N, K, B, "two_valued")
    print("PASS test_unordered_two_valued")


def test_unordered_masked_tail(ctx: DeviceContext) raises:
    """A masked tail at `-3.0e38` over 17 score levels, as the indexer produces.

    Rows differ in valid length, and the lowest ones have fewer valid columns
    than `K`, so the selection must reach into the masked region.
    """
    comptime N = 3071
    comptime K = 2048
    comptime B = 8
    var scores = List[Float32](capacity=B * N)
    for b in range(B):
        var valid = N - 150 * b
        for c in range(N):
            if c < valid:
                scores.append(Float32((c * 37 + b * 11) % 17) / Float32(16.0))
            else:
                scores.append(Float32(-3.0e38))
    _check_both(ctx, scores, N, K, B, "masked_tail")
    print("PASS test_unordered_masked_tail")


def test_unordered_negative_and_extremes(ctx: DeviceContext) raises:
    """All-negative rows, and rows mixing huge magnitudes with denormals.

    Exercises the sign-flip half of the key map and a very wide exponent range,
    which is where the rank's digit is weakest.
    """
    comptime N = 4099
    comptime K = 2048
    comptime B = 6
    var neg = List[Float32](capacity=B * N)
    for b in range(B):
        for c in range(N):
            neg.append(-Float32((c * 13 + b) % 1000) - Float32(1.0))
    _check_both(ctx, neg, N, K, B, "all_negative")

    var wide = List[Float32](capacity=B * N)
    for b in range(B):
        for c in range(N):
            var m = (c + b) % 6
            if m == 0:
                wide.append(Float32(3.0e38))
            elif m == 1:
                wide.append(Float32(-3.0e38))
            elif m == 2:
                wide.append(Float32(1.0e-38))
            elif m == 3:
                wide.append(Float32(-1.0e-38))
            elif m == 4:
                wide.append(Float32(0.0))
            else:
                wide.append(Float32(c))
    _check_both(ctx, wide, N, K, B, "wide_exponents")
    print("PASS test_unordered_negative_and_extremes")


def test_unordered_sorted_inputs(ctx: DeviceContext) raises:
    """Already-sorted rows, both directions.

    Descending input makes the answer the first `K` columns; ascending makes it
    the last `K`, in reverse. Both are degenerate for a histogram select.
    """
    comptime N = 6151
    comptime K = 2048
    comptime B = 6
    var desc = List[Float32](capacity=B * N)
    var asc = List[Float32](capacity=B * N)
    for _ in range(B):
        for c in range(N):
            desc.append(Float32(N - c))
            asc.append(Float32(c))
    _check_both(ctx, desc, N, K, B, "sorted_desc")
    _check_both(ctx, asc, N, K, B, "sorted_asc")
    print("PASS test_unordered_sorted_inputs")


def test_unordered_beyond_resident(ctx: DeviceContext) raises:
    """`N` past the narrow resident cap, over every select the contract can reach.

    Both row counts, because they instantiate different kernels: below the SM
    count a row this wide is held in registers by the wide-payload resident
    select, or streamed with prefetch once it is wider than that; above the SM
    count the occupancy variant streams it. The cheap contract has to land on
    exactly `K` in each. Long-context decode and prefill sit on opposite sides of
    that line.

    Two of the lengths straddle the wide payload's ceiling, so together they pin
    that dispatch edge: one column more moves the row off registers and onto the
    streaming select, and both sides must agree with each other and with
    themselves across launches.
    """
    comptime K = 2048
    for N in [8193, 16006, 16384, 16388, 32769]:
        for B in [8, 160]:
            var scores = _lcg_scores(B, N, 0xFEED0000 + UInt32(N), 100.0)
            _check_both(ctx, scores, N, K, B, String("beyond_n", N, "_b", B))
    print("PASS test_unordered_beyond_resident")


def test_unordered_small_n(ctx: DeviceContext) raises:
    """`N` at or below the champion width, which routes to the block kernel."""
    comptime B = 8
    for cfg in [(2048, 2048), (2048, 16), (512, 512), (64, 16), (1, 1)]:
        var N = cfg[0]
        var K = cfg[1]
        var scores = _lcg_scores(B, N, 0x0DDBA11 + UInt32(N + K), 20.0)
        _check_both(ctx, scores, N, K, B, String("small_", N, "_", K))
    print("PASS test_unordered_small_n")


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #


# ===----------------------------------------------------------------------=== #
# Bounded rows (`row_bounds`) — per-row live-column clamps, as driven by the
# MLA indexer under capture-frozen metadata where the row stride `N` is a
# worst-case bound far above any row's real length.
# ===----------------------------------------------------------------------=== #


def _run_and_check_bounded(
    ctx: DeviceContext,
    scores_host: List[Float32],  # B * N flat, poisoned at/past each bound
    N: Int,
    K: Int,
    B: Int,
    bounds: List[Int],
    label: String,
) raises:
    """Run the bounded split launcher and validate each row against its bound.

    The caller poisons columns at or past each row's bound with values larger
    than any live score, so a kernel that reads past a bound cannot pass: a
    poisoned column would displace a live selection and surface as an
    out-of-bound index.

    Verifies per row, with `E = min(K, bound)` expected live selections:
    1. Slots `[0, E)` hold the exact top-`E` of the row's first `bound`
       columns (distinct, in-bounds, non-increasing score order, tie-robust).
    2. Slots `[E, K)` are `-1`.
    """
    assert K <= PERSISTENT_TOPK_MAX_N, "K exceeds champion width"
    assert len(scores_host) == B * N, "scores_host length mismatch"
    assert len(bounds) == B, "bounds length mismatch"

    var scores_dev = ctx.enqueue_create_buffer[.float32](B * N)
    var idxs_dev = ctx.enqueue_create_buffer[.int32](B * K)
    idxs_dev.enqueue_fill(Int32(-2))  # sentinel to catch unwritten slots
    var bounds_dev = ctx.enqueue_create_buffer[.int32](B)

    with scores_dev.map_to_host() as buf:
        for i in range(B * N):
            buf[i] = Float32(scores_host[i])
    with bounds_dev.map_to_host() as buf:
        for b in range(B):
            buf[b] = Int32(bounds[b])

    persistent_topk_block_split[ordered=True, deterministic=True](
        ctx,
        rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_dev.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
        N,
        K,
        total_seq_len=B,
        row_bounds=Optional(
            rebind[ImmPointer[Int32, ImmutAnyOrigin]](bounds_dev.unsafe_ptr())
        ),
    )
    ctx.synchronize()

    var idxs_host = ctx.enqueue_create_host_buffer[.int32](B * K)
    ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
    ctx.synchronize()

    for b in range(B):
        var bound = bounds[b]
        var live = min(K, bound)

        for k in range(live, K):
            var idx = Int(idxs_host[b * K + k])
            assert_true(
                idx == -1,
                String(
                    "[",
                    label,
                    "] row ",
                    b,
                    " tail slot ",
                    k,
                    " = ",
                    idx,
                    " (expected -1 past bound ",
                    bound,
                    ")",
                ),
            )

        if live == 0:
            continue
        var row_scores = List[Float32](capacity=bound)
        for i in range(bound):
            row_scores.append(scores_host[b * N + i])
        var row_idxs = List[Int](capacity=live)
        for k in range(live):
            row_idxs.append(Int(idxs_host[b * K + k]))
        _check_topk_row(row_scores, row_idxs, bound, live, label, b)

    _ = scores_dev
    _ = idxs_dev
    _ = bounds_dev
    _ = idxs_host


def _poisoned_bounded_scores(
    B: Int, N: Int, bounds: List[Int], sd: UInt32
) -> List[Float32]:
    """LCG scores with columns at/past each row's bound poisoned to +1e30."""
    var scores = _lcg_scores(B, N, sd, 100.0)
    for b in range(B):
        for i in range(bounds[b], N):
            scores[b * N + i] = Float32(1.0e30)
    return scores^


def test_bounded_2048_kernel(ctx: DeviceContext) raises:
    """N=K=2048 — the single-block bitonic kernel with per-row bounds."""
    comptime N = 2048
    comptime K = 2048
    var bounds: List[Int] = [2048, 1024, 100, 1, 0]
    var B = len(bounds)
    var scores = _poisoned_bounded_scores(B, N, bounds, 0xB0DE2048)
    _run_and_check_bounded(ctx, scores, N, K, B, bounds, "bounded_2048")
    print("PASS test_bounded_2048_kernel")


def test_bounded_resident(ctx: DeviceContext) raises:
    """N=4096 (register-resident histsel, plain digit) with per-row bounds."""
    comptime N = 4096
    comptime K = 2048
    var bounds: List[Int] = [4096, 3000, 2048, 500, 1, 0]
    var B = len(bounds)
    var scores = _poisoned_bounded_scores(B, N, bounds, 0xB0DE4096)
    _run_and_check_bounded(ctx, scores, N, K, B, bounds, "bounded_resident")
    print("PASS test_bounded_resident")


def test_bounded_resident_bin_digit(ctx: DeviceContext) raises:
    """N=2560 < K + K//2 — the bin-digit resident instantiation, bounded."""
    comptime N = 2560
    comptime K = 2048
    var bounds: List[Int] = [2560, 2100, 800, 1, 0]
    var B = len(bounds)
    var scores = _poisoned_bounded_scores(B, N, bounds, 0xB0DE2560)
    _run_and_check_bounded(ctx, scores, N, K, B, bounds, "bounded_bin_digit")
    print("PASS test_bounded_resident_bin_digit")


def test_bounded_histsel_prefetch(ctx: DeviceContext) raises:
    """N=32769, few rows — the prefetching streaming histsel, bounded.

    This is the capture-frozen decode shape: the stride is far above every
    bound, including bounds below K (select-all rows) and at exactly K.
    """
    comptime N = 32769
    comptime K = 2048
    var bounds: List[Int] = [32769, 20000, 8192, 2048, 2047, 1000, 1, 0]
    var B = len(bounds)
    var scores = _poisoned_bounded_scores(B, N, bounds, 0xB0DE7E7C)
    _run_and_check_bounded(ctx, scores, N, K, B, bounds, "bounded_prefetch")
    print("PASS test_bounded_histsel_prefetch")


def test_bounded_histsel_fills_gpu(ctx: DeviceContext) raises:
    """Rows above the SM count — the non-prefetch histsel, bounded."""
    comptime N = 8193
    comptime K = 2048
    comptime B = 192
    var bounds = List[Int](capacity=B)
    for b in range(B):
        var m = b % 4
        if m == 0:
            bounds.append(N)
        elif m == 1:
            bounds.append(4096)
        elif m == 2:
            bounds.append(1500)
        else:
            bounds.append(0)
    var scores = _poisoned_bounded_scores(B, N, bounds, 0xB0DEF111)
    _run_and_check_bounded(ctx, scores, N, K, B, bounds, "bounded_fills_gpu")
    print("PASS test_bounded_histsel_fills_gpu")


def main() raises:
    with DeviceContext() as ctx:
        test_full_sort_n2048(ctx)
        test_random_full_sort_n2048(ctx)
        test_partial_topk_k16(ctx)
        test_partial_topk_k1024(ctx)
        test_small_n_padded(ctx)
        test_n_equals_k_small(ctx)
        test_n_equals_k_power_of_2(ctx)
        test_duplicate_scores(ctx)
        test_two_valued_scores(ctx)
        test_negative_scores(ctx)
        test_multi_batch(ctx)
        test_sorted_input_already_descending(ctx)
        test_sorted_input_ascending(ctx)
        test_single_element(ctx)
        test_streaming_n16384_random(ctx)
        test_streaming_n16006_nonmultiple(ctx)
        test_streaming_n163840_ascending(ctx)
        test_streaming_masked_and_ties(ctx)
        test_split_decode_long(ctx)
        test_split_decode_mtp(ctx)
        test_split_max_context(ctx)
        test_split_partial_k(ctx)
        test_split_masked_and_ties(ctx)
        test_split_tree_s6(ctx)
        test_split_tree_s8(ctx)
        test_split_tree_s13(ctx)
        test_block_odd_n_multirow(ctx)
        test_block_streaming_odd_n_multirow(ctx)
        test_unordered_resident_band(ctx)
        test_unord_nd_predicate_bites()
        test_unordered_wide_payload_band(ctx)
        test_unordered_prime_lengths(ctx)
        test_unordered_k_edges(ctx)
        test_unordered_row_counts(ctx)
        test_unordered_all_equal(ctx)
        test_unordered_plateau_straddles_k(ctx)
        test_unordered_plateau_edges(ctx)
        test_unordered_k1_all_equal(ctx)
        test_unordered_two_valued(ctx)
        test_unordered_masked_tail(ctx)
        test_unordered_negative_and_extremes(ctx)
        test_unordered_sorted_inputs(ctx)
        test_unordered_beyond_resident(ctx)
        test_unordered_small_n(ctx)
        test_bounded_2048_kernel(ctx)
        test_bounded_resident(ctx)
        test_bounded_resident_bin_digit(ctx)
        test_bounded_histsel_prefetch(ctx)
        test_bounded_histsel_fills_gpu(ctx)
    print("ALL TESTS PASSED")
