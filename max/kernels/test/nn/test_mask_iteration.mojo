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
from nn.attention.mha_mask import (
    CausalMask,
    ChunkedMask,
    SlidingWindowCausalMask,
    SlidingWindowNonCausalMask,
    ChunkedCausalMask,
    MaskStrategy,
    MHAMask,
    TileMaskStatus,
)
from std.utils.index import Index
from std.testing import assert_equal, assert_true


def compute_total_iters0[
    MaskType: MHAMask, //, BM: Int, BN: Int
](mask: MaskType, q_row: UInt32, end: UInt32) -> UInt32:
    comptime seq_id: UInt32 = 0
    var kv_row: UInt32 = 0
    while (
        mask.status(
            seq_id,
            Index[dtype=.int32](Int(q_row), Int(kv_row)),
            Index[dtype=.int32](BM, BN),
        )
        == TileMaskStatus.FULL_MASK
    ):
        kv_row += UInt32(BN)
    var iter_count: UInt32 = 0
    while True:
        kv_row += UInt32(BN)
        if kv_row >= end:
            break
        if (
            mask.status(
                seq_id,
                Index[dtype=.int32](Int(q_row), Int(kv_row)),
                Index[dtype=.int32](BM, BN),
            )
            == TileMaskStatus.FULL_MASK
        ):
            continue
        iter_count += 1
    return iter_count + 1


def compute_total_iters1[
    MaskType: MHAMask, //, BM: Int, BN: Int
](mask: MaskType, q_row: UInt32, end: UInt32) -> UInt32:
    comptime seq_id: UInt32 = 0
    var iter_count: UInt32 = 0
    var kv_row: UInt32 = 0
    while kv_row < end:
        iter_count += UInt32(
            Int(
                mask.status(
                    seq_id,
                    Index[dtype=.int32](Int(q_row), Int(kv_row)),
                    Index[dtype=.int32](BM, BN),
                )
                != TileMaskStatus.FULL_MASK
            )
        )
        kv_row += UInt32(BN)
    return iter_count


def status[
    MaskType: MHAMask, //, BM: Int, BN: Int
](mask: MaskType, q_row: UInt32, kv_row: UInt32) -> TileMaskStatus:
    return mask.status(
        UInt32(0),
        Index[dtype=.int32](q_row, kv_row),
        Index[dtype=.int32](BM, BN),
    )


def test_mask[
    MaskType: MHAMask, //, BM: Int, BN: Int, page_size: Int = 1
](mask: MaskType, q_row: UInt32, end: UInt32) raises:
    comptime seq_id: UInt32 = 0
    var kv_row: UInt32 = mask.start_column[BM, BN, page_size](seq_id, q_row)
    comptime mask_sets = MaskType.nonfull_sets[BM, BN]()
    comptime num_sets = len(mask_sets)
    var mask_ends = mask.masked_set_ends[BM=BM, BN=BN, page_size=page_size](
        seq_id, q_row, end
    )

    var ref_mask: TileMaskStatus
    if kv_row > 0:
        ref_mask = status[BM, BN](mask, q_row, kv_row - UInt32(BN))
        assert_equal(TileMaskStatus.FULL_MASK, ref_mask)
    var total_iters: UInt32 = 0
    for i in range(num_sets):
        var mask_status = mask_sets[i]
        var iters: UInt32 = (
            mask_ends[i] if i == 0 else mask_ends[i] - mask_ends[i - 1]
        )
        total_iters += iters
        for _ in range(iters):
            if kv_row >= end:
                print(
                    MaskType.name(), ": kv_row end iters =", kv_row, end, iters
                )
            assert_true(kv_row < end)
            ref_mask = status[BM, BN](mask, q_row, kv_row)
            if mask_status != ref_mask:
                print("mask_ends = [", end="")
                for i in range(num_sets):
                    if i > 0:
                        print(", ", end="")
                    print(mask_ends[i], end="")
                print("]")
                print("q_row num_keys kv_row =", q_row, end, kv_row)
                print(
                    "mask_status, ref_mask = ",
                    mask_status,
                    ", ",
                    ref_mask,
                    sep="",
                )
            assert_equal(mask_status, ref_mask)
            kv_row += UInt32(BN)
    if kv_row < end:
        ref_mask = status[BM, BN](mask, q_row, kv_row)
        assert_equal(TileMaskStatus.FULL_MASK, ref_mask)

    var calc_total_iter = mask.total_iters[BM, BN, page_size](
        seq_id, q_row, end
    )
    if total_iters != calc_total_iter:
        print("mask_ends = [", end="")
        for i in range(num_sets):
            if i > 0:
                print(", ", end="")
            print(mask_ends[i], end="")
        print("]")
        print("q_row =", q_row)
        print("num_keys =", end)
        print(
            "start_col =",
            mask.start_column[BM, BN, page_size](seq_id, q_row),
        )
        print("calc_total_iter =", calc_total_iter)
    assert_equal(
        total_iters, mask.total_iters[BM, BN, page_size](seq_id, q_row, end)
    )


def test_noncausal_strategies_bitmask[
    MaskType: MHAMask, //, BM: Int, BN: Int
](mask: MaskType) raises:
    # SlidingWindowNonCausalMask must mask every nonfull tile with BITMASK.
    # The final tile can straddle num_keys when num_keys % BN != 0; BITMASK's
    # mask_bits() folds in the col < num_keys clip, whereas a NO_MASK strategy
    # lets the comptime softmax path skip that clip and attend out-of-bounds
    # KV slots past the valid extent.
    comptime strategies = MaskType.mask_strategies[BM, BN]()
    for i in range(len(strategies)):
        assert_equal(
            Int(strategies[i]._value), Int(MaskStrategy.BITMASK._value)
        )


def test_no_mask_strategy_in_bounds[
    MaskType: MHAMask, //, BM: Int, BN: Int, page_size: Int = 1
](mask: MaskType, q_row: UInt32, num_keys: UInt32) raises:
    # A set whose STRATEGY is NO_MASK skips the `col < num_keys` clip, so every
    # tile it iterates must be fully in-bounds. Assert the cumulative end of
    # each non-empty NO_MASK-strategy set stays within num_keys. (Empty sets --
    # e.g. the interior set on a boundary-crossing block -- are skipped.)
    comptime seq_id: UInt32 = 0
    comptime strategies = MaskType.mask_strategies[BM, BN]()
    var start = mask.start_column[BM, BN, page_size](seq_id, q_row)
    var ends = mask.masked_set_ends[BM=BM, BN=BN, page_size=page_size](
        seq_id, q_row, num_keys
    )
    for i in range(len(strategies)):
        if strategies[i]._value == MaskStrategy.NO_MASK._value:
            var prev: UInt32 = ends[i - 1] if i > 0 else UInt32(0)
            if ends[i] > prev:
                assert_true(start + ends[i] * UInt32(BN) <= num_keys)


def main() raises:
    test_noncausal_strategies_bitmask[BM=128, BN=128](
        SlidingWindowNonCausalMask[16]()
    )
    test_noncausal_strategies_bitmask[BM=128, BN=64](
        SlidingWindowNonCausalMask[1024]()
    )

    # alias BM = 2
    # alias BN = 2
    comptime BM = 128
    comptime BN = 128
    comptime causal_mask = CausalMask()
    comptime sliding_mask16 = SlidingWindowCausalMask[16]()
    comptime sliding_mask1024 = SlidingWindowCausalMask[1024]()
    comptime noncausal_mask16 = SlidingWindowNonCausalMask[16]()
    comptime noncausal_mask1024 = SlidingWindowNonCausalMask[1024]()
    comptime noncausal_mask4096 = SlidingWindowNonCausalMask[4096]()
    comptime chunked_causal_mask = ChunkedCausalMask[256]()
    # Bare ChunkedMask with the 3-set partition gate satisfied
    # (W % BN == 0 and W >= BM). W == BM exercises the single-tile chunk;
    # larger W exercises a non-empty NO_MASK interior set.
    comptime chunked_mask128 = ChunkedMask[128]()
    comptime chunked_mask256 = ChunkedMask[256]()
    comptime chunked_mask512 = ChunkedMask[512]()
    # W < BM: every block straddles a chunk boundary, so the gate falls back to
    # the single PARTIAL set (which is exact here -- no tile fits in a chunk).
    comptime chunked_mask64 = ChunkedMask[64]()
    for num_keys in range(1, 8193):
        for q_row in range(num_keys):
            test_mask[BM=BM, BN=BN, page_size=1](
                causal_mask, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=512](
                causal_mask, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=1](
                sliding_mask16, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=512](
                sliding_mask16, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=1](
                sliding_mask1024, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=512](
                sliding_mask1024, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=1](
                noncausal_mask16, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=512](
                noncausal_mask16, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=1](
                noncausal_mask1024, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=512](
                noncausal_mask1024, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=1](
                noncausal_mask4096, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=512](
                noncausal_mask4096, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=1](
                chunked_mask128, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=512](
                chunked_mask128, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=1](
                chunked_mask256, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=512](
                chunked_mask256, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=1](
                chunked_mask512, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=512](
                chunked_mask512, UInt32(q_row), UInt32(num_keys)
            )
            # NO_MASK-strategy interior set must never straddle num_keys.
            test_no_mask_strategy_in_bounds[BM=BM, BN=BN, page_size=1](
                chunked_mask128, UInt32(q_row), UInt32(num_keys)
            )
            test_no_mask_strategy_in_bounds[BM=BM, BN=BN, page_size=512](
                chunked_mask128, UInt32(q_row), UInt32(num_keys)
            )
            test_no_mask_strategy_in_bounds[BM=BM, BN=BN, page_size=1](
                chunked_mask256, UInt32(q_row), UInt32(num_keys)
            )
            test_no_mask_strategy_in_bounds[BM=BM, BN=BN, page_size=512](
                chunked_mask512, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=1](
                chunked_mask64, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=512](
                chunked_mask64, UInt32(q_row), UInt32(num_keys)
            )
            var count0 = compute_total_iters0[BM=BM, BN=BN](
                chunked_causal_mask, UInt32(q_row), UInt32(num_keys)
            )
            var count1 = compute_total_iters1[BM=BM, BN=BN](
                chunked_causal_mask, UInt32(q_row), UInt32(num_keys)
            )
            var count2 = chunked_causal_mask.total_iters[
                BM=BM, BN=BN, page_size=1
            ](UInt32(0), UInt32(q_row), UInt32(num_keys))
            var count3 = chunked_causal_mask.total_iters[
                BM=BM, BN=BN, page_size=512
            ](UInt32(0), UInt32(q_row), UInt32(num_keys))
            if count0 != count1 or count0 != count2 or count0 != count3:
                print("q_row, num_keys =", q_row, num_keys)
                print(
                    "count0, count1, count2, count3 =",
                    count0,
                    count1,
                    count2,
                    count3,
                )
            assert_equal(count0, count1)
            assert_equal(count0, count2)
            assert_equal(count0, count3)
            # ChunkedCausalMask = OrMask[CausalMask, ChunkedMask] now exposes a
            # precise merged partition; validate it tile-by-tile against the
            # true combined status (and that no NO_MASK-strategy set straddles
            # num_keys).
            test_mask[BM=BM, BN=BN, page_size=1](
                chunked_causal_mask, UInt32(q_row), UInt32(num_keys)
            )
            test_mask[BM=BM, BN=BN, page_size=512](
                chunked_causal_mask, UInt32(q_row), UInt32(num_keys)
            )
            test_no_mask_strategy_in_bounds[BM=BM, BN=BN, page_size=1](
                chunked_causal_mask, UInt32(q_row), UInt32(num_keys)
            )
            test_no_mask_strategy_in_bounds[BM=BM, BN=BN, page_size=512](
                chunked_causal_mask, UInt32(q_row), UInt32(num_keys)
            )
