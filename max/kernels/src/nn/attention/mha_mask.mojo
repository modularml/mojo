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
"""Mask types and the `MHAMask` trait for multi-head attention kernels.

Defines the `MHAMask` trait and concrete implementations including
`CausalMask`, `NullMask`, `SlidingWindowCausalMask`, and `MaterializedMask`.
Masks encode which query-key pairs are visible and determine per-tile
iteration strategies used by prefill and decode kernels.
"""

from std.utils import StaticTuple
from std.math import align_down, iota, ceildiv
from std.sys import is_nvidia_gpu
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from std.collections import OptionalReg
from std.utils.index import IndexList, Index
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder

# ===-----------------------------------------------------------------------===#
# MaskName
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct MaskName(Equatable, Writable):
    """A canonical string name identifying a mask type."""

    var name: String

    comptime NULL = Self("null")
    comptime CAUSAL = Self("causal")
    comptime CHUNKED = Self("chunked")
    comptime SLIDING_WINDOW_CAUSAL = Self("sliding_window_causal")
    comptime SLIDING_WINDOW_NONCAUSAL = Self("sliding_window_noncausal")
    comptime MATERIALIZED = Self("materialized")
    comptime CHUNKED_CAUSAL = Self("chunked_causal")
    comptime CAUSAL_PADDING = Self("causal_padding")

    def write_to(self, mut writer: Some[Writer]):
        """Writes the mask name.

        Args:
            writer: The writer to write to.
        """
        writer.write_string(self.name)

    def __eq__(self, rhs: String) -> Bool:
        return self.name == rhs


# ===-----------------------------------------------------------------------===#
# TileMaskStatus
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct TileMaskStatus(
    Equatable,
    Identifiable,
    TrivialRegisterPassable,
    Writable,
):
    """A tile's masking status."""

    var status: UInt8

    # No element is masked.
    comptime NO_MASK = Self(0)

    # Some elements in the tile are masked.
    comptime PARTIAL_MASK = Self(1)

    # All elements in the tile are masked.
    comptime FULL_MASK = Self(3)

    # Unknown mask -- a further check needed.
    # This is used by masks not yet supporting
    # the predefined trip count information.
    comptime UNKNOWN_MASK = Self(4)

    def __eq__(self, rhs: Self) -> Bool:
        return self.status == rhs.status

    def __ne__(self, rhs: Self) -> Bool:
        return self.status != rhs.status

    def __is__(self, rhs: Self) -> Bool:
        return self.status == rhs.status

    def __and__(self, rhs: Self) -> Self:
        return Self(self.status & rhs.status)

    def __or__(self, rhs: Self) -> Self:
        return Self(self.status | rhs.status)

    def write_to(self, mut writer: Some[Writer]):
        if self is Self.NO_MASK:
            return writer.write("not masked")
        if self is Self.PARTIAL_MASK:
            return writer.write("partially masked")
        if self is Self.FULL_MASK:
            return writer.write("fully masked")
        writer.write("unknown mask")


@fieldwise_init
struct MaskStrategy(Equatable, TrivialRegisterPassable):
    """Bit-flag enum that selects the masking strategy for a tile iteration set.

    Strategies are combined with bitwise OR. `NO_MASK` skips masking
    entirely. `COMPUTED` calls the mask functor per element.
    `OUT_OF_BOUNDS` clips keys at `num_keys`. `BITMASK` reads a 32-bit
    column-visibility mask from `MHAMask.mask_bits()`, which subsumes the
    older triangular and OOB strategies for SM100 kernels.
    """

    var _value: Int32
    comptime NO_MASK = Self(0)
    """
    No mask is to be applied.
    """
    comptime COMPUTED = Self(4)
    """
    Mask where the call operator must be used to compute the masked value.
    """
    comptime OUT_OF_BOUNDS = Self(8)
    """
    Check if we are out of bounds, e.g. clip at `num_keys` after a
    `COMPUTED` mask. Used in combination with `COMPUTED` and as the
    kernel-hardcoded "no mask, just clip" form (e.g. hot-path optimizations
    in `softmax_warp` that bypass `BITMASK` when the runtime mask status
    is `NO_MASK`).
    """
    comptime BITMASK = Self(16)
    """
    Mask provides its own 32-bit visibility bitmask per 32-col batch via
    `MHAMask.mask_bits()`. Subsumes the historical `LOWER_TRIANGULAR` /
    `UPPER_TRIANGULAR` (and `OUT_OF_BOUNDS` for masks that fold it into
    `mask_bits`).
    """

    @always_inline
    def __and__(self, other: Self) -> Self:
        return {self._value & other._value}

    @always_inline
    def __or__(self, other: Self) -> Self:
        return {self._value | other._value}

    @always_inline
    def __contains__(self, other: Self) -> Bool:
        return (self._value | other._value) == self._value


# ===-----------------------------------------------------------------------===#
# MHAMask
# ===-----------------------------------------------------------------------===#


trait MHAMask(Copyable, DevicePassable, TrivialRegisterPassable):
    """The MHAMask trait describes masks for MHA kernels, such as the causal mask.
    """

    comptime apply_log2e_after_mask: Bool
    """
    Does the mask require `log2e` to be applied after the mask, or
    can it be fused with the scaling?
    """
    comptime mask_out_of_bound: Bool
    comptime mask_safe_out_of_bounds: Bool
    """
    Is the mask safe to read out of bounds?
    """
    comptime check_mask_during_decoding: Bool
    """
    Should we check the mask during decoding, or should we assume that it
    does not return `FULL_MASK`?
    """

    def mask[
        dtype: DType,
        width: SIMDLength,
        //,
        *,
        element_type: DType = .uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        """Return mask vector at given coordinates.

        The functor could capture an mask tensor and add to the score e.g. Replit.

        Parameters:
            dtype: The element type of the score vector.
            width: The SIMD width of the score vector.
            element_type: The integer type for index coordinates (inferred;
                defaults to `DType.uint32`).

        Args:
            coord: The coordinate tuple `(seq_id, head, q_idx, k_idx)`.
            score_vec: The score vector at `coord` of the score matrix.
        """
        ...

    def status[
        *, element_type: DType = .uint32
    ](
        self,
        seq_id: UInt32,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        """Given a tile's index range, return its masking status.

        `seq_id` identifies the sequence/batch this tile belongs to and is
        used by masks (e.g., `CausalPaddingMask`) whose status depends on
        per-sequence state. Implementations that don't need it should ignore
        it; the unused argument will be DCE'd.

        Parameters:
            element_type: The integer type for index coordinates (defaults
                to `DType.uint32`).

        Args:
            seq_id: The sequence/batch index.
            tile_offset: The `(row, col)` offset of the tile in the score
                matrix.
            tile_size: The `(height, width)` size of the tile in query rows
                and key columns.
        """
        ...

    def start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32) -> UInt32:
        """
        Returns the first column for which this mask does not return
        `TileMaskStatus.FULL_MASK`.
        This may not be a multiple of `BN`, in which case iterating using
        `start_column` and `masked_set_ends` will not necessarily produce
        the same set or number of iterations as iterating from
        `0` and checking `status` to skip.
        The return value of `total_iters` should be less than or equal to
        the number of non-skipped iterations.
        The practical consequence is that all warp group specializations
        within a kernel that loop over columns need to be in agreement.
        Either they all loop over all columns and check status to skip,
        or they loop using the `masked_set_ends`.

        Parameters:
            BM: Query tile height (number of query rows per block).
            BN: Key tile width (number of key columns per block).
            page_size: The KV cache page size in key columns (0 or 1 if
                unpaged).

        Args:
            seq_id: The sequence/batch index.
            row: The starting query-row index of the current query tile.
        """
        ...

    @staticmethod
    def start_column_alignment[BM: Int, BN: Int, page_size: Int]() -> Int:
        """
        The largest power of 2, capped at `BN`, that divides every
        `base_kv_row = start_column + k*BN` produced by BN-stride
        mask-driven iteration. Callers pass this directly as
        `base_alignment` to `PagedKVCache.populate`, which uses it to
        pick the largest legal SIMD chunk for its LUT vector load.

        Implementations must return a value that already divides `BN`
        (equivalently, the value must equal
        `gcd(natural_alignment, BN)`). For an implementation whose
        natural `start_column` alignment is a power of 2 less than or
        equal to `BN`, this is automatic. An implementation whose
        natural alignment doesn't divide `BN` must wrap its return in
        `gcd(..., BN)` itself.

        Parameters:
            BM: Query tile height (number of query rows per block).
            BN: Key tile width (number of key columns per block).
            page_size: The KV cache page size in key columns (0 or 1 if
                unpaged).
        """
        ...

    def total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        """
        The total number of column iterations for which this mask returns either
        `TileMaskStatus.NO_MASK' or 'TileMaskStatus.PARTIAL_MASK'.
        This is to be used by warp specializations that do not need to
        use `kv_row`.

        Parameters:
            BM: Query tile height (number of query rows per block).
            BN: Key tile width (number of key columns per block).
            page_size: The KV cache page size in key columns (0 or 1 if
                unpaged).

        Args:
            seq_id: The sequence/batch index.
            row: The starting query-row index of the current query tile.
            num_cols: The exclusive upper bound on key column indices.
        """
        ...

    @staticmethod
    def count_nonfull_sets(BM: Int, BN: Int) -> Int:
        """The number of blocks that are all partial-masks or not masked.

        Args:
            BM: Query tile height (number of query rows per block).
            BN: Key tile width (number of key columns per block).
        """
        ...

    def masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        """
        For each set of iterations in `nonfull_sets`, indicate the end idx
        belonging to that set (i.e., the last idx would be `end - 1`).
        Note that the final `masked_set_ends` may not necessarily equal
        `total_iters`, if we have `UNKNOWN_MASK`s.
        In case of `UNKNOWN_MASK`s, `masked_set_ends` with tile-skipping
        must be used to have the correct kv_row values at each iteration.

        Parameters:
            BM: Query tile height (number of query rows per block).
            BN: Key tile width (number of key columns per block).
            page_size: The KV cache page size in key columns (0 or 1 if
                unpaged).

        Args:
            seq_id: The sequence/batch index.
            row: The starting query-row index of the current query tile.
            num_cols: The exclusive upper bound on key column indices.
        """
        ...

    def last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        """Equivalent to `masked_set_ends[BM,BN,page_size](seq_id, row, num_cols)[-1]`.

        Parameters:
            BM: Query tile height (number of query rows per block).
            BN: Key tile width (number of key columns per block).
            page_size: The KV cache page size in key columns (0 or 1 if
                unpaged).

        Args:
            seq_id: The sequence/batch index.
            row: The starting query-row index of the current query tile.
            num_cols: The exclusive upper bound on key column indices.
        """
        ...

    @staticmethod
    def nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        """
        For each set of iterations that are either partially masked or not masked,
        this indicates the mask status.
        `UNKNOWN_MASK` here is an indicator meaning that we should check the status
        at runtime.
        It is semantically equivalent to `partial`, but with the optimization
        hint that it's worth checking on each iteration at runtime for
        `FULL_MASK` (in which case we can skip the tile) or `NO_MASK`
        (in which case we can unswitch and avoid masking in an inner loop).

        Parameters:
            BM: Query tile height (number of query rows per block).
            BN: Key tile width (number of key columns per block).
        """
        ...

    @staticmethod
    def mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        """
        For each set of iterations that are either partially masked or not masked,
        this indicates the `MaskStrategy` to use.

        Parameters:
            BM: Query tile height (number of query rows per block).
            BN: Key tile width (number of key columns per block).
        """
        ...

    def mask_bits(
        self,
        seq_id: UInt32,
        score_row: Int32,
        col_start: Int32,
        num_keys: Int32,
    ) -> UInt32:
        """Returns a 32-bit visibility bitmask for a column batch.

        Bit `i` (0..31) is 1 iff the column `col_start + i` is visible at
        (`seq_id`, `score_row`) for this mask. Called by SM100 `apply_mask`
        once per 32-col batch when the mask's `mask_strategies` returns
        `MaskStrategy.BITMASK`. Masks that don't use `BITMASK` should return
        `0xFFFF_FFFF` (no constraint); the result is unused in that case.

        Args:
            seq_id: Per-sequence batch index (e.g. for `CausalPaddingMask`).
            score_row: Global query row (the q index in the attention score).
            col_start: Global key index of bit 0 in this batch.
            num_keys: Kernel cache length (upper bound on visible keys).
        """
        ...

    @staticmethod
    def sliding_window_size() -> Int:
        """Returns the sliding window lower-bound offset, or 0 if unbounded.

        For `SlidingWindowCausalMask`, returns `window_size`. MLA decode
        kernels read this to recover the window size in places where the
        struct's parametric `window_size` is not accessible through the
        trait surface.
        """
        ...

    @staticmethod
    def name() -> String:
        ...


# ===-----------------------------------------------------------------------===#
# CausalMask
# ===-----------------------------------------------------------------------===#

# Finite mask sentinel. Must sit far below any physically reachable QK score
# (raw or scale*log2e-scaled) so it can never win the online-softmax row max
# and zero out a row's real columns, yet stay finite so a fully-masked row
# computes MASK_VALUE - MASK_VALUE = 0 instead of -inf - -inf = NaN. A power of
# two so float32 and bfloat16 hold it exactly and agree bit-for-bit.
comptime MASK_VALUE = -FloatLiteral(2**100)


@fieldwise_init
struct CausalMask(MHAMask, TrivialRegisterPassable):
    """MHA causal mask ensures a token is only affected by previous tokens."""

    comptime apply_log2e_after_mask: Bool = False
    comptime mask_out_of_bound: Bool = is_nvidia_gpu()
    comptime mask_safe_out_of_bounds: Bool = True
    comptime check_mask_during_decoding: Bool = False

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "CausalMask"

    @staticmethod
    def name() -> String:
        return "CausalMask"

    @always_inline
    def mask[
        dtype: DType,
        width: SIMDLength,
        //,
        *,
        element_type: DType = .uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime index_type = coord.element_type

        # coord[2] and coord[3] are the token index in query and key respectively.
        var q_idx = coord[2]
        var k_idx = coord[3]

        # coords[2] >= coords[3] ensures the current tokens is only affected by
        # itself and previous tokens.
        # TODO(KERN-782): MASK_VALUE should be -inf but softmax saturates with NaNs.
        var masked_score_vec = (
            SIMD[index_type, width](q_idx).ge(
                iota[index_type, width](Scalar[index_type](k_idx))
            )
        ).select(score_vec, MASK_VALUE)

        return masked_score_vec

    @always_inline
    def status[
        *, element_type: DType = .uint32
    ](
        self,
        seq_id: UInt32,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        # Consider tile corners
        #
        # 1
        # ^
        # C--------------D        A: (offset0,         offset1)
        # |              |        B: (offset0 + size0, offset1)
        # |              |        C: (offset0,         offset1 + size1)
        # |              |        D: (offset0 + size0, offset1 + size1)
        # A--------------B --> 0
        #
        # Key Points:
        #   * A is inside the tile but B, C, D are not.
        #   * If B is on or above the diagonal i.e. offset0 + size0 <= offset1
        #     the tile is fully masked.
        #   * If C is on or below the diagonal i.e. offset0 >= offset1 + size1 - 1
        #     the tile is not masked at all.

        # If false, the tile is not masked.
        var min_q_lt_max_k = (
            (tile_offset.data[0] + 1).lt(
                tile_offset.data[1] + tile_size.data[1]
            )
        ).cast[.uint8]()

        # If true, the tile is fully masked
        var max_q_lt_min_k = (
            (tile_offset.data[0] + tile_size.data[0]).le(tile_offset.data[1])
        ).cast[.uint8]()

        # Use 2 bits to represent:
        # (F, F) -> no mask
        # (T, F) -> partial mask
        # (T, T) -> full mask
        return TileMaskStatus(min_q_lt_max_k + (max_q_lt_min_k << 1))

    @always_inline
    def start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32) -> UInt32:
        # offset0 is row
        # offset1 is col
        # B is (row + BM, col) -- means fully masked (we skip)
        # C is (row, col + BN) -- means not masked
        # causal mask is
        return 0

    @staticmethod
    def start_column_alignment[BM: Int, BN: Int, page_size: Int]() -> Int:
        # `start_column` always returns 0, which is divisible by anything;
        # cap at BN.
        return BN

    @always_inline
    def total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        # Masked off when row < col
        # row + BM - 1 < BN * x
        # We want the smallest value of `x` such that this is true
        # NOTE: by not checking `num_cols`, this method assumes that
        # `seq_len <= cache_len`.
        # If `seq_len` (and thus `rows`) may be larger than
        # `cache_len` (and thus `num_cols`), this will return the wrong result.
        return ceildiv(min(row + UInt32(BM), num_cols), UInt32(BN))

    @staticmethod
    def count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 2

    @always_inline
    def last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return self.total_iters[BM, BN, page_size](seq_id, row, num_cols)

    @always_inline
    def masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        # Unmasked when row >= col
        # We want to find the maximum `x` such that
        # row >= x*BN - 1
        # Each col tile extends from col...col+BN-1,
        # so iter `i` (starting from 0) has maximum value
        # `i*BN + BN - 1 = (i+1)*BN - 1`.
        # Thus, iter `i` is fulle <= row if `row >= (i + 1)*BN - 1`.
        # `x`, the number of unmasked iters, is thus
        # x = i+1 = (row + 1) // BN
        var num_unmasked = (row + 1) // UInt32(BN)
        var partial_mask_end = self.total_iters[BM, BN, page_size](
            seq_id, row, num_cols
        )
        return {num_unmasked, partial_mask_end}

    @staticmethod
    def nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        return {TileMaskStatus.NO_MASK, TileMaskStatus.PARTIAL_MASK}

    @staticmethod
    def mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        return {MaskStrategy.NO_MASK, MaskStrategy.BITMASK}

    @always_inline
    def mask_bits(
        self,
        seq_id: UInt32,
        score_row: Int32,
        col_start: Int32,
        num_keys: Int32,
    ) -> UInt32:
        # Causal: bit i is 1 iff (col_start + i) <= score_row.
        # n_valid = max(1 + score_row - col_start, 0), clamped to 32.
        var n_valid: Int32 = max(1 + score_row - col_start, 0)
        return (
            (UInt32(1) << UInt32(n_valid)) - UInt32(1)
        ) if n_valid < 32 else UInt32(0xFFFF_FFFF)

    @staticmethod
    def sliding_window_size() -> Int:
        return 0


# ===-----------------------------------------------------------------------===#
# NullMask
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct NullMask(MHAMask, TrivialRegisterPassable):
    """Mask that's effectively a noop."""

    comptime apply_log2e_after_mask: Bool = False
    comptime mask_out_of_bound: Bool = True
    comptime mask_safe_out_of_bounds: Bool = True
    comptime check_mask_during_decoding: Bool = False

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "NullMask"

    @staticmethod
    def name() -> String:
        return "NullMask"

    @always_inline
    def mask[
        dtype: DType,
        width: SIMDLength,
        //,
        *,
        element_type: DType = .uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        return score_vec

    @always_inline
    def status[
        *, element_type: DType = .uint32
    ](
        self,
        seq_id: UInt32,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        # no mask
        return TileMaskStatus.NO_MASK

    @always_inline
    def start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32) -> UInt32:
        return 0

    @staticmethod
    def start_column_alignment[BM: Int, BN: Int, page_size: Int]() -> Int:
        # `start_column` always returns 0; cap at BN.
        return BN

    @always_inline
    def total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        """
        The total number of column iterations for which this mask returns either
        `TileMaskStatus.NO_MASK' or 'TileMaskStatus.PARTIAL_MASK'.

        Parameters:
            BM: Query tile height (number of query rows per block).
            BN: Key tile width (number of key columns per block).
            page_size: The KV cache page size in key columns (0 or 1 if
                unpaged).

        Args:
            seq_id: The sequence/batch index.
            row: The starting query-row index of the current query tile.
            num_cols: The exclusive upper bound on key column indices.
        """
        return ceildiv(num_cols, UInt32(BN))

    @always_inline
    def last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return self.total_iters[BM, BN, page_size](seq_id, row, num_cols)

    @staticmethod
    def count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 2

    @always_inline
    def masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        return {
            num_cols // UInt32(BN),
            self.total_iters[BM, BN, page_size](seq_id, row, num_cols),
        }

    @staticmethod
    def nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        return {TileMaskStatus.NO_MASK, TileMaskStatus.NO_MASK}

    @staticmethod
    def mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        return {MaskStrategy.NO_MASK, MaskStrategy.BITMASK}

    @always_inline
    def mask_bits(
        self,
        seq_id: UInt32,
        score_row: Int32,
        col_start: Int32,
        num_keys: Int32,
    ) -> UInt32:
        # NullMask: only the cache-length OOB cutoff. Bit i is 1 iff
        # (col_start + i) < num_keys.
        var n_valid_oob: Int32 = max(num_keys - col_start, 0)
        return (
            (UInt32(1) << UInt32(n_valid_oob)) - UInt32(1)
        ) if n_valid_oob < 32 else UInt32(0xFFFF_FFFF)

    @staticmethod
    def sliding_window_size() -> Int:
        return 0


# ===-----------------------------------------------------------------------===#
# ChunkedMask
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct ChunkedMask[local_window_size: Int](MHAMask, TrivialRegisterPassable):
    """Mask implementing Chunked attention.

    This groups the mask into chunks of size `local_window_size`.
    Considering the following case:
    - Q_len = 7
    - K_len = 10
    - local_window_size = 4

    The mask will be applied as follows:
        K > 0 1 2 3 4 5 6 7 8 9
        Q v x--------------------x
        0 | 1 1 1 1 0 0 0 0 0 0
        1 | 0 0 0 0 1 1 1 1 0 0
        2 | 0 0 0 0 1 1 1 1 0 0
        3 | 0 0 0 0 1 1 1 1 0 0
        4 | 0 0 0 0 1 1 1 1 0 0
        5 | 0 0 0 0 0 0 0 0 1 1
        6 | 0 0 0 0 0 0 0 0 1 1

    Parameters:
        local_window_size: The chunk size in tokens; positions are grouped
            into contiguous blocks of this width.
    """

    comptime apply_log2e_after_mask: Bool = False
    comptime mask_out_of_bound: Bool = True
    comptime mask_safe_out_of_bounds: Bool = True
    comptime check_mask_during_decoding: Bool = True

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "ChunkedMask"

    @staticmethod
    def name() -> String:
        return "ChunkedMask[" + String(Self.local_window_size) + "]"

    @always_inline
    def mask[
        dtype: DType,
        width: SIMDLength,
        //,
        *,
        element_type: DType = .uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime assert (
            width <= Self.local_window_size
        ), "SIMD width of chunked mask must be <= local window size"

        var k_start_idx = coord.data[3]
        var k_end_idx = k_start_idx + Scalar[element_type](width) - 1

        var q_chunk_idx = Int(
            coord.data[2] // Scalar[element_type](Self.local_window_size)
        )
        var k_start_chunk_idx = Int(k_start_idx) // Self.local_window_size
        var k_end_chunk_idx = Int(k_end_idx) // Self.local_window_size

        if q_chunk_idx == k_start_chunk_idx == k_end_chunk_idx:
            # fully unmasked, return the value
            return score_vec

        elif q_chunk_idx == k_start_chunk_idx or q_chunk_idx == k_end_chunk_idx:
            # partial mask
            var retval = score_vec
            var boundary = UInt32(
                (k_start_idx + Scalar[element_type](Self.local_window_size) - 1)
                // Scalar[element_type](Self.local_window_size)
            ) * UInt32(Self.local_window_size)

            var mask_val = SIMD[.bool, width](fill=False)
            var k_indices = k_start_idx.cast[.uint32]() + iota[.uint32, width]()
            if q_chunk_idx == k_start_chunk_idx:
                mask_val = k_indices.ge(boundary)
            elif q_chunk_idx == k_end_chunk_idx:
                mask_val = k_indices.lt(boundary)

            return mask_val.select(SIMD[dtype, width](MASK_VALUE), retval)

        # fully masked
        return SIMD[dtype, width](MASK_VALUE)

    @always_inline
    def status[
        *, element_type: DType = .uint32
    ](
        self,
        seq_id: UInt32,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        var q_start_window = tile_offset[0] // Self.local_window_size
        var q_end_window = (
            ceildiv(
                tile_offset[0] + tile_size[0],
                Self.local_window_size,
            )
            - 1
        )
        var k_start_window = tile_offset[1] // Self.local_window_size
        var k_end_window = (
            ceildiv(
                tile_offset[1] + tile_size[1],
                Self.local_window_size,
            )
            - 1
        )

        var overlapping_windows = (
            k_end_window >= q_start_window and q_end_window >= k_start_window
        )

        if q_start_window == k_start_window == k_end_window == q_end_window:
            return TileMaskStatus.NO_MASK
        elif overlapping_windows:
            return TileMaskStatus.PARTIAL_MASK
        else:
            return TileMaskStatus.FULL_MASK

    @always_inline
    def start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32) -> UInt32:
        # First column for which `row` is not masked is
        var col: UInt32 = (row // UInt32(Self.local_window_size)) * UInt32(
            Self.local_window_size
        )
        # Align down to a tile boundary so that iterators stepping
        # by BN or page_size land on multiples and never overshoot
        # num_keys.
        comptime align_to = BN if page_size <= 1 else min(page_size, BN)
        return align_down(col, UInt32(align_to))

    @staticmethod
    def start_column_alignment[BM: Int, BN: Int, page_size: Int]() -> Int:
        # Matches `align_to` in `start_column`.
        return BN if page_size <= 1 else min(page_size, BN)

    @always_inline
    def total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        var start_col = self.start_column[BM, BN, page_size](seq_id, row)
        # `end_col` is 1 past the last potentially-visible column: the end of
        # the chunk the last query row belongs to, clamped to the cache length.
        # The clamp matters for the NO_MASK partition below: `status()` is
        # chunk-only (ignores `num_keys`), so without it a `num_keys`-straddling
        # tile would be marked NO_MASK and iterated past the valid extent.
        var end_col = min(
            (1 + ((row + UInt32(BM) - 1) // UInt32(Self.local_window_size)))
            * UInt32(Self.local_window_size),
            num_cols,
        )
        return ceildiv(end_col - start_col, UInt32(BN))

    @staticmethod
    def count_nonfull_sets(BM: Int, BN: Int) -> Int:
        # When a query block can sit entirely inside one chunk (`W >= BM`) and
        # the chunk boundary is tile-aligned (`W % BN == 0`, so `start_column`
        # lands on the chunk start with no leading FULL tiles), a non-crossing
        # block's interior tiles are fully visible -> a NO_MASK middle set.
        # Boundary-straddling blocks leave that set empty (see
        # `masked_set_ends`).
        if Self.local_window_size % BN == 0 and Self.local_window_size >= BM:
            return 3
        else:
            return 1

    @always_inline
    def last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return self.total_iters[BM, BN, page_size](seq_id, row, num_cols)

    @always_inline
    def masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        var total = self.total_iters[BM, BN, page_size](seq_id, row, num_cols)
        comptime if (
            Self.local_window_size % BN == 0 and Self.local_window_size >= BM
        ):
            # `W % BN == 0` guarantees `start_column == c0` (the first row's
            # chunk start), so every offset below is relative to that chunk.
            comptime W = UInt32(Self.local_window_size)
            var c0 = (row // W) * W
            var crossing = ((row + UInt32(BM) - 1) // W) * W != c0
            if crossing:
                # Block spans two chunks (`BM <= W`): every tile is PARTIAL (its
                # columns are visible to only one of the two row groups), so the
                # whole range goes in the leading BITMASK set and both trailing
                # NO_MASK sets are empty.
                return {total, total, total}
            else:
                # Non-crossing: every chunk column is visible to every row.
                # Tiles fully below `num_keys` need no masking at all (set 1,
                # NO_MASK strategy); the single `num_keys`-straddling tile (the
                # ceil-vs-floor difference) needs only the OOB clip (set 2,
                # OUT_OF_BOUNDS). FLOOR division is what keeps set 1 strictly
                # in-bounds, so the NO_MASK strategy never reads past num_keys.
                var in_bounds_end = (min(c0 + W, num_cols) - c0) // UInt32(BN)
                return {UInt32(0), in_bounds_end, total}
        else:
            return {total}

    @staticmethod
    def nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        comptime if (
            Self.local_window_size % BN == 0 and Self.local_window_size >= BM
        ):
            return {
                TileMaskStatus.PARTIAL_MASK,
                TileMaskStatus.NO_MASK,
                TileMaskStatus.NO_MASK,
            }
        else:
            return {TileMaskStatus.PARTIAL_MASK}

    @staticmethod
    def mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        comptime if (
            Self.local_window_size % BN == 0 and Self.local_window_size >= BM
        ):
            # Leading set holds boundary-crossing tiles -> BITMASK (`mask_bits`
            # folds the chunk window and the OOB clip). The interior set is fully
            # chunk-visible AND bounded strictly below `num_keys` by
            # `masked_set_ends` -> NO_MASK (no clip, no select: the cheapest
            # path). Only the single `num_keys`-straddling tile remains, in the
            # trailing set -> OUT_OF_BOUNDS, which keeps the `col < num_keys`
            # clip. Splitting the straddle tile out is what makes the NO_MASK
            # interior safe for this non-causal mask (unlike
            # SlidingWindowCausalMask, causality can't bound the OOB columns).
            return {
                MaskStrategy.BITMASK,
                MaskStrategy.NO_MASK,
                MaskStrategy.OUT_OF_BOUNDS,
            }
        else:
            return {MaskStrategy.BITMASK}

    @always_inline
    def mask_bits(
        self,
        seq_id: UInt32,
        score_row: Int32,
        col_start: Int32,
        num_keys: Int32,
    ) -> UInt32:
        # `score_row` lives in chunk `[c_q, c_q + W)` where `W` is the chunk
        # size. Bit `i` (col `col_start + i`) is visible iff
        # `c_q <= col_start + i < c_q + W` and `col_start + i < num_keys`.
        # Build the chunk window as `high_mask ^ low_mask` (a contiguous run
        # of set bits) and AND in the OOB cutoff.
        comptime W: Int32 = Int32(Self.local_window_size)
        var c_q: Int32 = align_down(score_row, W)

        var lo: Int32 = max(min(c_q - col_start, Int32(32)), Int32(0))
        var hi: Int32 = max(min(c_q + W - col_start, Int32(32)), Int32(0))
        var n_oob: Int32 = max(min(num_keys - col_start, Int32(32)), Int32(0))

        var high_mask: UInt32 = (
            (UInt32(1) << UInt32(hi)) - UInt32(1)
        ) if hi < 32 else UInt32(0xFFFF_FFFF)
        var low_mask: UInt32 = (
            (UInt32(1) << UInt32(lo)) - UInt32(1)
        ) if lo < 32 else UInt32(0xFFFF_FFFF)
        var oob_mask: UInt32 = (
            (UInt32(1) << UInt32(n_oob)) - UInt32(1)
        ) if n_oob < 32 else UInt32(0xFFFF_FFFF)

        return (high_mask ^ low_mask) & oob_mask

    @staticmethod
    def sliding_window_size() -> Int:
        return 0


# ===-----------------------------------------------------------------------===#
# SlidingWindowCausalMask
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct SlidingWindowCausalMask[window_size: Int](
    MHAMask, TrivialRegisterPassable
):
    """Mask implementing Sliding Window attention.

    Considering the following case:
    - Q_len = 7
    - K_len = 7
    - window_size = 3

    The mask will be applied as follows:
        K > 0 1 2 3 4 5 6
        Q v x------------x
        0 | 1 0 0 0 0 0 0
        1 | 1 1 0 0 0 0 0
        2 | 1 1 1 0 0 0 0
        3 | 0 1 1 1 0 0 0
        4 | 0 0 1 1 1 0 0
        5 | 0 0 0 1 1 1 0
        6 | 0 0 0 0 1 1 1

    Parameters:
        window_size: The sliding window size in tokens; a query at position `q`
            attends only to keys in `[q - window_size + 1, q]`.
    """

    comptime apply_log2e_after_mask: Bool = False
    comptime mask_out_of_bound: Bool = True
    comptime mask_safe_out_of_bounds: Bool = True
    comptime check_mask_during_decoding: Bool = True

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "SlidingWindowCausalMask"

    @staticmethod
    def name() -> String:
        return "SlidingWindowCausalMask[" + String(Self.window_size) + "]"

    @always_inline
    def mask[
        dtype: DType,
        width: SIMDLength,
        *,
        element_type: DType = .uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime index_type = coord.element_type

        comptime assert (
            width <= Self.window_size
        ), "SIMD width of sliding window mask must be <= window size"

        var q_idx = coord[2]
        var k_idx = coord[3]

        # first, check if the query is after the key, this step is the same
        # as the causal mask
        var masked_score_vec = (
            SIMD[index_type, width](q_idx).ge(
                iota[index_type, width](Scalar[index_type](k_idx))
            )
        ).select(score_vec, MASK_VALUE)

        # second, check if the query is within the window size of the key
        # It looks like we will encounter a problem here when the q_idx is
        # smaller than k_idx, but this is not possible because of the causal mask
        # that we have applied.
        return (
            (
                SIMD[index_type, width](q_idx)
                - iota[index_type, width](Scalar[index_type](k_idx))
            )
            .lt(Scalar[index_type](Self.window_size))
            .select(masked_score_vec, SIMD[dtype, width](MASK_VALUE))
        )

    @always_inline
    def status[
        *, element_type: DType = .uint32
    ](
        self,
        seq_id: UInt32,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        # --- Check for FULL_MASK scenarios ---

        # Case 1: If the entire tile is too far to the right
        # (all query positions come before all key positions)
        var query_ends_before_keys_begin = (
            tile_offset.data[0] + tile_size.data[0] <= tile_offset.data[1]
        )

        # Case 2: If the entire tile is too far to the left
        # (all query positions are more than window_size away from all key positions)
        # Rewrite the inequality to use only addition so that we never subtract
        # `window_size` from an unsigned value (which can underflow).
        # Original condition:
        #     q_start - window_size + 1 >= k_start + k_size
        # is equivalent to:
        #     q_start + 1 >= k_start + k_size + window_size
        # where
        #     q_start = tile_offset[0]
        #     k_start = tile_offset[1]
        #     k_size  = tile_size[1]
        # Hence we compare two *added* terms, avoiding any risk of wrapping
        # around zero.

        var lhs = tile_offset.data[0] + 1
        var rhs = (
            tile_offset.data[1]
            + tile_size.data[1]
            + Scalar[element_type](Self.window_size)
        )
        var queries_too_far_ahead_of_keys = lhs >= rhs

        if query_ends_before_keys_begin or queries_too_far_ahead_of_keys:
            return TileMaskStatus.FULL_MASK

        # --- Check for NO_MASK scenario ---

        # Two conditions must BOTH be true for the tile to have no masks:

        # Condition 1: The earliest query position must be after the latest key position
        # (diagonal condition)
        var min_query_after_max_key = (
            tile_offset.data[0] >= tile_offset.data[1] + tile_size.data[1] - 1
        )

        # Condition 2: The latest query position must be within the window range of the
        # earliest key position
        var max_query_within_window_of_min_key = tile_offset.data[
            0
        ] + tile_size.data[0] - 1 < tile_offset.data[1] + Scalar[element_type](
            Self.window_size
        )

        if min_query_after_max_key and max_query_within_window_of_min_key:
            return TileMaskStatus.NO_MASK

        # If we reached here, some positions are masked and others aren't
        return TileMaskStatus.PARTIAL_MASK

    @always_inline
    def start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32) -> UInt32:
        var col: UInt32 = UInt32(
            max(Int32(row) - Int32(Self.window_size) + 1, 0)
        )

        # Align down to a tile boundary so that iterators stepping
        # by BN or page_size land on multiples and never overshoot
        # num_keys.
        comptime align_to = BN if page_size <= 1 else min(page_size, BN)
        return align_down(col, UInt32(align_to))

    @staticmethod
    def start_column_alignment[BM: Int, BN: Int, page_size: Int]() -> Int:
        # Matches `align_to` in `start_column`.
        return BN if page_size <= 1 else min(page_size, BN)

    @always_inline
    def total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        var start_col = self.start_column[BM, BN, page_size](seq_id, row)
        var end_col = min(row + UInt32(BM), num_cols)  # one past end
        return ceildiv(end_col - start_col, UInt32(BN))

    @staticmethod
    def count_nonfull_sets(BM: Int, BN: Int) -> Int:
        if ((Self.window_size) // BN) > ((BM + BN - 2) // BN):
            return 3
        else:
            return 1

    @always_inline
    def masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        var start_col = self.start_column[BM, BN, page_size](seq_id, row)
        # partial_exit_end_col = row + BM
        var partial_exit_end_col = min(row + UInt32(BM), num_cols)
        # partial's end uses `ceildiv` and unmasked uses floored division
        # Partials must cover the entire `BN` tile with an masked entry
        # The unmasked region can't handle a tile with any
        #
        # (unmasked_end_col - start_col) // BN -
        #     ceildiv(partial_entry_end_col - start_col, BN)
        #
        # = (unmasked_end_col - start_col) // BN -
        #       (partial_entry_end_col - start_col + BN - 1) // BN
        # = (row + 1 - row + window_size - 1) // BN -
        #       (row + BM - window_size - row + window_size - 1 + BN - 1) // BN
        # = (window_size) // BN - (BM + BN - 2) // BN
        #
        # Thus, we will have unmasked iters when
        # (window_size) // BN - (BM + BN - 2) // BN > 0
        #
        var end_tile = ceildiv(partial_exit_end_col - start_col, UInt32(BN))

        comptime if ((Self.window_size) // BN) > ((BM + BN - 2) // BN):
            # the partial entry region ends when row + BM - 1 is unmasked
            var partial_entry_end_col: UInt32 = row + UInt32(BM)
            if partial_entry_end_col > UInt32(Self.window_size):
                partial_entry_end_col -= UInt32(Self.window_size)
            else:
                partial_entry_end_col = 0
            var unmasked_end_col = row + 1
            return {
                ceildiv(partial_entry_end_col - start_col, UInt32(BN)),
                (unmasked_end_col - start_col) // UInt32(BN),
                end_tile,
            }
        else:
            return {end_tile}

    @always_inline
    def last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return self.total_iters[BM, BN, page_size](seq_id, row, num_cols)

    @staticmethod
    def nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        comptime if (((Self.window_size) // BN) - ((BM + BN - 2) // BN)) > 0:
            return {
                TileMaskStatus.PARTIAL_MASK,
                TileMaskStatus.NO_MASK,
                TileMaskStatus.PARTIAL_MASK,
            }
        else:
            return {
                TileMaskStatus.PARTIAL_MASK,
            }

    @staticmethod
    def mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        comptime if (((Self.window_size) // BN) - ((BM + BN - 2) // BN)) > 0:
            return {
                MaskStrategy.BITMASK,
                MaskStrategy.NO_MASK,
                MaskStrategy.BITMASK,
            }
        else:
            return {MaskStrategy.BITMASK}

    @always_inline
    def mask_bits(
        self,
        seq_id: UInt32,
        score_row: Int32,
        col_start: Int32,
        num_keys: Int32,
    ) -> UInt32:
        # Causal bound: bits in [0, n_valid) set, n_valid = max(1+score_row -
        # col_start, 0). Window bound: clear low (n_valid - window_size) bits
        # when positive. The combined pattern is the visible window. The same
        # body works for all three sets in the 3-set partition (the math
        # degenerates correctly for the all-set or all-clear edges).
        var n_valid: Int32 = max(1 + score_row - col_start, 0)
        var bits: UInt32 = (
            (UInt32(1) << UInt32(n_valid)) - UInt32(1)
        ) if n_valid < 32 else UInt32(0xFFFF_FFFF)
        var mask_off: Int32 = n_valid - Int32(Self.window_size)
        if mask_off > 0:
            bits &= (
                UInt32(0xFFFF_FFFF) << UInt32(mask_off)
            ) if mask_off < 32 else UInt32(0)
        return bits

    @staticmethod
    def sliding_window_size() -> Int:
        return Self.window_size


# ===-----------------------------------------------------------------------===#
# SlidingWindowNonCausalMask
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct SlidingWindowNonCausalMask[window_size: Int](
    MHAMask, TrivialRegisterPassable
):
    """Non-causal sliding-window attention mask.

    A `(q, k)` pair is visible iff `k + window_size > q`. Unlike
    `SlidingWindowCausalMask` there is no causal upper bound, so future keys
    (`k > q`) are always visible: a windowed context plus a bidirectional
    block. Used by windowed block-diffusion speculative-decode drafts (DFlash).

    Example with Q_len = K_len = 7, window_size = 3 (upper triangle all 1s,
    unlike `SlidingWindowCausalMask`):
        K > 0 1 2 3 4 5 6
        Q v x------------x
        0 | 1 1 1 1 1 1 1
        1 | 1 1 1 1 1 1 1
        2 | 1 1 1 1 1 1 1
        3 | 0 1 1 1 1 1 1
        4 | 0 0 1 1 1 1 1
        5 | 0 0 0 1 1 1 1
        6 | 0 0 0 0 1 1 1

    Parameters:
        window_size: The sliding window size in tokens; a query at position
            `q` attends only to keys at or after `q - window_size + 1`, with
            no upper bound (future keys are always visible).
    """

    comptime apply_log2e_after_mask: Bool = False
    comptime mask_out_of_bound: Bool = True
    comptime mask_safe_out_of_bounds: Bool = True
    comptime check_mask_during_decoding: Bool = True

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "SlidingWindowNonCausalMask"

    @staticmethod
    def name() -> String:
        return "SlidingWindowNonCausalMask[" + String(Self.window_size) + "]"

    @always_inline
    def mask[
        dtype: DType,
        width: SIMDLength,
        *,
        element_type: DType = .uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime index_type = coord.element_type

        comptime assert (
            width <= Self.window_size
        ), "SIMD width of sliding window mask must be <= window size"

        var q_idx = coord[2]
        var k_idx = coord[3]

        # Visible iff `k + window > q` (addition form is underflow-safe).
        return (
            (
                iota[index_type, width](Scalar[index_type](k_idx))
                + Scalar[index_type](Self.window_size)
            )
            .gt(SIMD[index_type, width](q_idx))
            .select(score_vec, SIMD[dtype, width](MASK_VALUE))
        )

    @always_inline
    def status[
        *, element_type: DType = .uint32
    ](
        self,
        seq_id: UInt32,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        # FULL_MASK iff the least-masked corner `(q0, k0+k_size-1)` is masked:
        # `q0 + 1 >= k0 + k_size + window_size` (addition form, underflow-safe).
        # No causal "too far right" case: future keys are always visible.
        var lhs = tile_offset.data[0] + 1
        var rhs = (
            tile_offset.data[1]
            + tile_size.data[1]
            + Scalar[element_type](Self.window_size)
        )
        if lhs >= rhs:
            return TileMaskStatus.FULL_MASK

        # NO_MASK iff the most-masked corner `(q0+q_size-1, k0)` is visible:
        # `k0 + window_size > q0 + q_size - 1`. No diagonal upper bound (unlike
        # causal): future keys are always visible.
        var max_query_within_window_of_min_key = (
            tile_offset.data[1] + Scalar[element_type](Self.window_size)
            > tile_offset.data[0] + tile_size.data[0] - 1
        )

        if max_query_within_window_of_min_key:
            return TileMaskStatus.NO_MASK

        return TileMaskStatus.PARTIAL_MASK

    @always_inline
    def start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32) -> UInt32:
        # Window lower bound `row - window_size + 1` (same as causal). Align
        # down so iterators stepping by BN/page_size never overshoot num_keys.
        var col: UInt32 = UInt32(
            max(Int32(row) - Int32(Self.window_size) + 1, 0)
        )
        comptime align_to = BN if page_size <= 1 else min(page_size, BN)
        return align_down(col, UInt32(align_to))

    @staticmethod
    def start_column_alignment[BM: Int, BN: Int, page_size: Int]() -> Int:
        # Matches `align_to` in `start_column`.
        return BN if page_size <= 1 else min(page_size, BN)

    @always_inline
    def total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        # Lower bound shifted by the window; no upper bound (iterate to
        # num_cols) since every tile from start_column on is non-FULL.
        var start_col = self.start_column[BM, BN, page_size](seq_id, row)
        return ceildiv(num_cols - start_col, UInt32(BN))

    @staticmethod
    def count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 2

    @always_inline
    def last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return self.total_iters[BM, BN, page_size](seq_id, row, num_cols)

    @always_inline
    def masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        # Monotonic scan from start_column: PARTIAL tiles (window edge) then
        # NO_MASK (above the band). PARTIAL run is `start_col <= k0 < thresh`,
        # thresh = row + BM - window_size, computed underflow-safe.
        var start_col = self.start_column[BM, BN, page_size](seq_id, row)
        var total = self.total_iters[BM, BN, page_size](seq_id, row, num_cols)

        var row_plus_bm: UInt32 = row + UInt32(BM)
        var partial_end: UInt32
        if row_plus_bm <= UInt32(Self.window_size):
            # Window covers the whole query block: no PARTIAL tiles.
            partial_end = 0
        else:
            var thresh: UInt32 = row_plus_bm - UInt32(Self.window_size)
            if thresh <= start_col:
                partial_end = 0
            else:
                var hi = min(thresh, num_cols)
                partial_end = ceildiv(hi - start_col, UInt32(BN))
        return {partial_end, total}

    @staticmethod
    def nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        # PARTIAL near the window edge, then NO_MASK above (mirror of causal).
        return {TileMaskStatus.PARTIAL_MASK, TileMaskStatus.NO_MASK}

    @staticmethod
    def mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        # The second set has NO_MASK *status* (no window-edge masking above
        # the band), but it still contains the final partial KV tile when
        # `num_cols % BN != 0`. Unlike causal masks — whose `mask()` hides
        # columns `>= num_keys` as future — nothing else bounds those
        # columns: with `MaskStrategy.NO_MASK` the comptime-set softmax
        # paths (sm100 FA4 and depth512) skip the `col < num_keys` clip and
        # attend to uninitialized slots in the tail of the last KV page.
        # (The runtime-status fallback paths use `OUT_OF_BOUNDS` for
        # NO_MASK-status tiles and never had this bug.) Use BITMASK for
        # both sets: `mask_bits()` folds the window lower bound and the
        # `num_keys` OOB clip into one visibility mask and computes
        # all-ones for interior tiles at a few scalar ops per 32 columns.
        return {MaskStrategy.BITMASK, MaskStrategy.BITMASK}

    @always_inline
    def mask_bits(
        self,
        seq_id: UInt32,
        score_row: Int32,
        col_start: Int32,
        num_keys: Int32,
    ) -> UInt32:
        # Visible iff `col + window > score_row` AND `col < num_keys` (OOB
        # clip). Clear the low `max(L - col_start, 0)` bits, L = score_row -
        # window + 1; no high-bit clear (no causal upper bound).
        var n_below: Int32 = max(
            score_row - Int32(Self.window_size) + 1 - col_start, 0
        )
        var low_mask: UInt32 = (
            UInt32(0xFFFF_FFFF) << UInt32(n_below)
        ) if n_below < 32 else UInt32(0)

        var n_oob: Int32 = max(num_keys - col_start, 0)
        var oob_mask: UInt32 = (
            (UInt32(1) << UInt32(n_oob)) - UInt32(1)
        ) if n_oob < 32 else UInt32(0xFFFF_FFFF)

        return low_mask & oob_mask

    @staticmethod
    def sliding_window_size() -> Int:
        return Self.window_size


# ===-----------------------------------------------------------------------===#
# CausalPaddingMask
# ===-----------------------------------------------------------------------===#


struct CausalPaddingMask[layout_: Layout, origin_: ImmOrigin](
    MHAMask, TrivialRegisterPassable
):
    """Causal mask combined with padding: a position (seq_id, head, q, k) is
    visible only when q >= k (causal) AND k < valid_lengths[seq_id] (padding).

    `valid_lengths` is a tensor of shape [num_seqs] with one uint32 value per
    sequence indicating the number of valid (non-padding) tokens.
    """

    comptime causal_mask: CausalMask = CausalMask()

    comptime apply_log2e_after_mask: Bool = False
    comptime mask_out_of_bound: Bool = is_nvidia_gpu()
    comptime mask_safe_out_of_bounds: Bool = True
    comptime check_mask_during_decoding: Bool = True

    var valid_lengths: LayoutTensor[.uint32, Self.layout_, Self.origin_]

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "CausalPaddingMask"

    @staticmethod
    def name() -> String:
        return "CausalPaddingMask"

    def __init__(
        out self,
        valid_lengths: LayoutTensor[.uint32, Self.layout_, Self.origin_],
    ):
        self.valid_lengths = valid_lengths

    @always_inline
    def mask[
        dtype: DType,
        width: SIMDLength,
        //,
        *,
        element_type: DType = .uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        # Apply causal mask first, then further mask padding positions.
        var causal_result = Self.causal_mask.mask(coord, score_vec)

        comptime index_type = coord.element_type
        var k_idx = coord[3]
        var valid_len = Scalar[index_type](Int(self.valid_lengths[coord[0]]))
        var k_positions = iota[index_type, width](Scalar[index_type](k_idx))

        return k_positions.lt(valid_len).select(causal_result, MASK_VALUE)

    @always_inline
    def status[
        *, element_type: DType = .uint32
    ](
        self,
        seq_id: UInt32,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        var causal_status = Self.causal_mask.status(
            seq_id, tile_offset, tile_size
        )

        # If the causal component alone says FULL_MASK, the tile is fully
        # masked regardless of padding.
        if causal_status == TileMaskStatus.FULL_MASK:
            return TileMaskStatus.FULL_MASK

        # Bring in the padding boundary now that we have seq_id.
        var valid_len = Scalar[element_type](
            Int(self.valid_lengths[Int(seq_id)])
        )

        # Tile fully masked when its first column is already past valid_len.
        if tile_offset.data[1] >= valid_len:
            return TileMaskStatus.FULL_MASK

        # NO_MASK only when causal alone is NO_MASK AND padding doesn't cut
        # the tile (last column is still strictly within valid_len).
        if (
            causal_status == TileMaskStatus.NO_MASK
            and tile_offset.data[1] + tile_size.data[1] <= valid_len
        ):
            return TileMaskStatus.NO_MASK

        return TileMaskStatus.PARTIAL_MASK

    @always_inline
    def start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32) -> UInt32:
        # Padding only chops the right end; `start_column` is the same as
        # `CausalMask`'s.
        return Self.causal_mask.start_column[BM, BN, page_size](seq_id, row)

    @staticmethod
    def start_column_alignment[BM: Int, BN: Int, page_size: Int]() -> Int:
        # Delegates to `CausalMask`, which always returns 0.
        return CausalMask.start_column_alignment[BM, BN, page_size]()

    @always_inline
    def total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        # Effective right edge is the min of the caller-supplied `num_cols`
        # (typically `cache_len`) and this sequence's padding cutoff.
        var valid_len = UInt32(Int(self.valid_lengths[Int(seq_id)]))
        var effective = min(num_cols, valid_len)
        return ceildiv(min(row + UInt32(BM), effective), UInt32(BN))

    @staticmethod
    def count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 2

    @always_inline
    def last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return self.total_iters[BM, BN, page_size](seq_id, row, num_cols)

    @always_inline
    def masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        # Tile `i` is fully unmasked iff its last column is both within the
        # causal-visible region (<= row) and within `effective`:
        #   (i+1)*BN <= row + 1  AND  (i+1)*BN <= effective
        # Combining via `min`:
        #   num_unmasked = min(row + 1, effective) // BN
        var valid_len = UInt32(Int(self.valid_lengths[Int(seq_id)]))
        var effective = min(num_cols, valid_len)
        var num_unmasked = min(row + 1, effective) // UInt32(BN)
        var total = self.total_iters[BM, BN, page_size](seq_id, row, num_cols)
        return {num_unmasked, total}

    @staticmethod
    def nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        return {TileMaskStatus.NO_MASK, TileMaskStatus.PARTIAL_MASK}

    @staticmethod
    def mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        # Partial tiles take the `BITMASK` path. `mask_bits` combines the
        # causal bound and the per-sequence padding cutoff into a single
        # 32-bit visibility pattern, so the kernel's per-element mask call
        # is unnecessary.
        return {MaskStrategy.NO_MASK, MaskStrategy.BITMASK}

    @always_inline
    def mask_bits(
        self,
        seq_id: UInt32,
        score_row: Int32,
        col_start: Int32,
        num_keys: Int32,
    ) -> UInt32:
        # Causal bits: low `n_valid` bits set,
        #   n_valid = max(1 + score_row - col_start, 0)
        # Padding bits: low `n_valid_pad` bits set,
        #   n_valid_pad = max(valid_lengths[seq_id] - col_start, 0)
        # Visible bits = causal_bits AND padding_bits.
        var n_valid: Int32 = max(1 + score_row - col_start, 0)
        var causal_bits: UInt32 = (
            (UInt32(1) << UInt32(n_valid)) - UInt32(1)
        ) if n_valid < 32 else UInt32(0xFFFF_FFFF)
        var valid_len: Int32 = Int32(Int(self.valid_lengths[Int(seq_id)]))
        var n_valid_pad: Int32 = max(valid_len - col_start, 0)
        var padding_bits: UInt32 = (
            (UInt32(1) << UInt32(n_valid_pad)) - UInt32(1)
        ) if n_valid_pad < 32 else UInt32(0xFFFF_FFFF)
        return causal_bits & padding_bits

    @staticmethod
    def sliding_window_size() -> Int:
        return 0


# ===-----------------------------------------------------------------------===#
# MaterializedMask
# ===-----------------------------------------------------------------------===#


@always_inline
def naively_compute_total_iters[
    MaskType: MHAMask, //, BM: Int, BN: Int
](mask: MaskType, seq_id: UInt32, q_row: UInt32, end: UInt32) -> UInt32:
    """Count the non-fully-masked KV tile iterations for a query row by linear scan.

    Walks every `BN`-wide column tile from `0` to `end`, calling
    `mask.status()` for each and counting tiles that are not
    `TileMaskStatus.FULL_MASK`. Intended as a reference fallback for mask
    types that do not implement a closed-form `total_iters()`.

    Parameters:
        MaskType: Concrete `MHAMask` implementation to query.
        BM: Query tile height (number of query rows per block).
        BN: Key tile width (number of key columns per block).

    Args:
        mask: The mask instance.
        seq_id: Sequence/batch index.
        q_row: Starting query-row index of the current query tile.
        end: Exclusive upper bound on key column indices.

    Returns:
        The number of KV tiles in `[0, end)` that are not fully masked.
    """

    var iter_count: UInt32 = 0
    var kv_row: UInt32 = 0
    while kv_row < end:
        iter_count += UInt32(
            Int(
                mask.status(
                    seq_id,
                    Index[dtype=DType.int32](Int(q_row), Int(kv_row)),
                    Index[dtype=DType.int32](BM, BN),
                )
                != TileMaskStatus.FULL_MASK
            )
        )
        kv_row += UInt32(BN)
    return iter_count


@always_inline
def naively_get_first_nonempty_mask_col[
    MaskType: MHAMask, //, BM: Int, BN: Int
](mask: MaskType, seq_id: UInt32, q_row: UInt32) -> UInt32:
    """Find the first KV tile column whose mask status is not `FULL_MASK`.

    Scans KV column tiles starting at 0 with stride `BN`, calling
    `mask.status()` until a tile that is not fully masked is found. Used as
    a fallback `start_column()` implementation for mask types that do not
    have a closed-form expression.

    Parameters:
        MaskType: Concrete `MHAMask` implementation to query.
        BM: Query tile height (number of query rows per block).
        BN: Key tile width (number of key columns per block).

    Args:
        mask: The mask instance.
        seq_id: Sequence/batch index.
        q_row: Starting query-row index of the current query tile.

    Returns:
        The column index of the first non-fully-masked KV tile, aligned to
        a multiple of `BN`.
    """

    var kv_row: UInt32 = 0
    while (
        mask.status(
            seq_id,
            Index[dtype=DType.int32](Int(q_row), Int(kv_row)),
            Index[dtype=DType.int32](BM, BN),
        )
        == TileMaskStatus.FULL_MASK
    ):
        kv_row += UInt32(BN)
    return kv_row


struct MaterializedMask[dtype_: DType, layout_: Layout, origin_: ImmOrigin](
    MHAMask, TrivialRegisterPassable
):
    """Mask that's backed by a materialized tensor.

    Parameters:
        dtype_: Element type of the backing mask tensor.
        layout_: Memory layout of the backing mask tensor.
        origin_: Origin (ownership/mutability qualifier) of the backing mask
            tensor.
    """

    comptime apply_log2e_after_mask: Bool = True
    comptime mask_out_of_bound: Bool = True
    comptime mask_safe_out_of_bounds: Bool = False
    comptime check_mask_during_decoding: Bool = True

    var mask_tensor: LayoutTensor[Self.dtype_, Self.layout_, Self.origin_]

    @__allow_legacy_any_origin_fields
    var start_pos: OptionalReg[
        LayoutTensor[.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ]
    var is_multiple_of_2: Bool

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "MaterializedMask"

    @staticmethod
    def name() -> String:
        return "MaterializedMask"

    def __init__(
        out self,
        mask_tensor: LayoutTensor[Self.dtype_, Self.layout_, Self.origin_],
        start_pos: OptionalReg[
            LayoutTensor[
                .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
            ]
        ] = None,
    ):
        comptime assert Self.layout_.rank() in (
            3,
            4,
        ), "Expected rank 3 or 4 for mask tensor"
        self.mask_tensor = mask_tensor
        self.start_pos = start_pos
        self.is_multiple_of_2 = (
            self.mask_tensor.dim[Self.layout_.rank() - 1]() % 2 == 0
        )

    @always_inline
    def get_start_pos(self, batch_idx: Int) -> Int:
        if self.start_pos:
            return Int(self.start_pos.value()[batch_idx])
        else:
            return (
                self.mask_tensor.dim[Self.layout_.rank() - 1]()
                - self.mask_tensor.dim[Self.layout_.rank() - 2]()
            )

    @always_inline
    def mask[
        dtype: DType,
        width: SIMDLength,
        //,
        *,
        element_type: DType = .uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime IndexListType = IndexList[
            Self.layout_.rank(), element_type=element_type
        ]
        var adjusted_coord: IndexListType

        var start_pos = self.get_start_pos(coord[0])

        comptime if Self.layout_.rank() == 3:
            adjusted_coord = IndexListType(
                coord[0], coord[2] - start_pos, coord[3]
            )
        else:
            adjusted_coord = IndexListType(
                coord[0], coord[1], coord[2] - start_pos, coord[3]
            )

        var retval = SIMD[dtype, width](MASK_VALUE)
        comptime rank = Self.layout_.rank()
        if adjusted_coord[rank - 2] < self.mask_tensor.dim[rank - 2]():
            if (
                adjusted_coord[rank - 1] + width
                <= self.mask_tensor.dim[rank - 1]()
                and self.is_multiple_of_2
            ):
                retval = self.mask_tensor.load[width=width](
                    adjusted_coord.canonicalize()
                ).cast[dtype]()
            elif adjusted_coord[rank - 1] < self.mask_tensor.dim[rank - 1]():
                for i in range(
                    min(
                        Int(width),
                        self.mask_tensor.dim[rank - 1]() - coord[3],
                    )
                ):
                    adjusted_coord[rank - 1] = coord[3] + i
                    retval[i] = self.mask_tensor.load[width=1](
                        adjusted_coord.canonicalize()
                    ).cast[dtype]()

        return score_vec + retval

    @always_inline
    def status[
        *, element_type: DType = .uint32
    ](
        self,
        seq_id: UInt32,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        # This is counter-intuitive but setting to `partial` ensures we
        # always read the values for the tensor.
        return TileMaskStatus.PARTIAL_MASK

    @always_inline
    def start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32) -> UInt32:
        return naively_get_first_nonempty_mask_col[BM, BN](self, seq_id, row)

    @staticmethod
    def start_column_alignment[BM: Int, BN: Int, page_size: Int]() -> Int:
        # `naively_get_first_nonempty_mask_col` steps by BN from 0.
        return BN

    @always_inline
    def total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return naively_compute_total_iters[BM, BN](self, seq_id, row, num_cols)

    @staticmethod
    def count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 1

    @always_inline
    def last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return ceildiv(num_cols, UInt32(BN))

    @always_inline
    def masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        return {
            self.last_masked_set_end[BM, BN, page_size](seq_id, row, num_cols)
        }

    @staticmethod
    def nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        return {TileMaskStatus.UNKNOWN_MASK}

    @staticmethod
    def mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        return {MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS}

    @always_inline
    def mask_bits(
        self,
        seq_id: UInt32,
        score_row: Int32,
        col_start: Int32,
        num_keys: Int32,
    ) -> UInt32:
        # `MaterializedMask` is an additive-bias mask: `mask()` returns
        # `score_vec + retval` (line ~1547), where `retval` is loaded from
        # an arbitrary-dtype tensor (e.g. ALiBi slopes, log-bias). A
        # 1-bit-per-key visibility mask cannot faithfully represent
        # arbitrary additive biases, so this mask permanently uses
        # `{COMPUTED | OUT_OF_BOUNDS}`. This body is unreachable;
        # returning all-ones is defensive only.
        return UInt32(0xFFFF_FFFF)

    @staticmethod
    def sliding_window_size() -> Int:
        return 0


@always_inline
def _supports_bitmask[M: MHAMask, BM: Int, BN: Int]() -> Bool:
    """Reports whether `M` can be safely routed through the SM100 BITMASK
    dispatch arm: no partition of its `mask_strategies` requires the
    `COMPUTED` per-element path. Pure `NO_MASK` partitions also qualify,
    since `mask_bits` returns a correct (all-ones or OOB-clipped) pattern
    for fully-visible tiles. Used by `AndMask` / `OrMask` to decide
    whether to advertise `BITMASK` based on the inner masks' strategies.
    """
    comptime strats = M.mask_strategies[BM, BN]()
    var ok: Bool = True
    comptime for i in range(len(strats)):
        if MaskStrategy.COMPUTED in strats[i]:
            ok = False
    return ok


@always_inline
def _nonfull_sets_known[M: MHAMask, BM: Int, BN: Int]() -> Bool:
    """Reports whether `M` publishes a statically-known iteration partition,
    i.e. no element of its `nonfull_sets` is `UNKNOWN_MASK`. Used by
    `AndMask` / `OrMask` to decide whether they can advertise a precise
    `PARTIAL_MASK` set (driving the cheaper known-partition consumer path
    that skips the per-tile runtime `status()` check) instead of falling
    back to `UNKNOWN_MASK`.
    """
    comptime sets = M.nonfull_sets[BM, BN]()
    var known: Bool = True
    comptime for i in range(len(sets)):
        if sets[i] == TileMaskStatus.UNKNOWN_MASK:
            known = False
    return known


@always_inline
def _both_multiset[T: MHAMask, S: MHAMask](BM: Int, BN: Int) -> Bool:
    """Reports whether both inner masks expose a multi-set (`>= 2`) known
    partition for `(BM, BN)`.

    Relies on the in-tree invariant that an `UNKNOWN_MASK` mask is always
    single-set (`count_nonfull_sets == 1`), so any mask whose
    `count_nonfull_sets >= 2` is a statically-known partition that contains a
    `NO_MASK` set. Hence `_both_multiset` ⟺ "both inners are known AND each has
    a `NO_MASK` band", exactly the precondition for an `OrMask` to expose a
    combined `NO_MASK` middle (intersection of two `NO_MASK` bands).

    This is callable from `count_nonfull_sets` (whose `BM`/`BN` are runtime
    `Int` arguments) because it takes `BM`/`BN` as arguments and only ever uses
    them in arithmetic / argument-position calls to the inners'
    `count_nonfull_sets`, never as parameters (which a runtime `Int` cannot
    satisfy).
    """
    return (
        T.count_nonfull_sets(BM, BN) >= 2 and S.count_nonfull_sets(BM, BN) >= 2
    )


@always_inline
def _child_nomask_cols[
    M: MHAMask, //, BM: Int, BN: Int, page_size: Int
](m: M, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
    UInt32, 2
]:
    """Returns a known child's contiguous `NO_MASK` run as an absolute
    `[lo_col, hi_col)` column interval (empty, `lo == hi == start_column`, when
    the child exposes no `NO_MASK` set). Adjacent `NO_MASK` sets (e.g.
    `NullMask`'s `{NO_MASK, NO_MASK}` or `ChunkedMask`'s
    `{PARTIAL, NO_MASK, NO_MASK}`) collapse into one run via first..last
    `NO_MASK` indices. Used by `OrMask` to intersect the two inners' `NO_MASK`
    bands. Assumes a `BN`-aligned `start_column` (the caller gates on this).
    """
    comptime sets = M.nonfull_sets[BM, BN]()
    var start_col = m.start_column[BM, BN, page_size](seq_id, row)
    var first_nm: Int = -1
    var last_nm: Int = -1
    for i in range(len(sets)):
        if sets[i] == TileMaskStatus.NO_MASK:
            if first_nm < 0:
                first_nm = i
            last_nm = i
    if first_nm < 0:
        return {start_col, start_col}
    var ends = m.masked_set_ends[BM, BN, page_size](seq_id, row, num_cols)
    var lo_rel: UInt32 = 0 if first_nm == 0 else UInt32(ends[first_nm - 1])
    var hi_rel: UInt32 = ends[last_nm]
    return {
        start_col + lo_rel * UInt32(BN),
        start_col + hi_rel * UInt32(BN),
    }


# ===-----------------------------------------------------------------------===#
# AndMask
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct AndMask[T: MHAMask, S: MHAMask, //, lhs: T, rhs: S](
    MHAMask, TrivialRegisterPassable
):
    """Mask that's the AND of two masks.
    If both masks mask off an element, the element is masked off."""

    comptime apply_log2e_after_mask: Bool = Self.T.apply_log2e_after_mask or Self.S.apply_log2e_after_mask
    comptime mask_out_of_bound: Bool = Self.T.mask_out_of_bound or Self.S.mask_out_of_bound
    comptime mask_safe_out_of_bounds: Bool = Self.T.mask_safe_out_of_bounds and Self.S.mask_safe_out_of_bounds
    comptime check_mask_during_decoding: Bool = Self.T.check_mask_during_decoding and Self.S.check_mask_during_decoding

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "AndMask"

    @staticmethod
    def name() -> String:
        return "AndMask[" + Self.T.name() + ", " + Self.S.name() + "]"

    @always_inline
    def mask[
        dtype: DType,
        width: SIMDLength,
        //,
        *,
        element_type: DType = .uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime if dtype == .bool or dtype.is_integral():
            return self.lhs.mask(coord, score_vec) & self.rhs.mask(
                coord, score_vec
            )

        else:
            return max(
                self.lhs.mask(coord, score_vec),
                self.rhs.mask(coord, score_vec),
            )

    @always_inline
    def status[
        *, element_type: DType = .uint32
    ](
        self,
        seq_id: UInt32,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        var lhs_status = self.lhs.status(seq_id, tile_offset, tile_size)
        var rhs_status = self.rhs.status(seq_id, tile_offset, tile_size)

        return lhs_status & rhs_status

    @always_inline
    def start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32) -> UInt32:
        return naively_get_first_nonempty_mask_col[BM, BN](self, seq_id, row)

    @staticmethod
    def start_column_alignment[BM: Int, BN: Int, page_size: Int]() -> Int:
        # `naively_get_first_nonempty_mask_col` steps by BN from 0.
        return BN

    @always_inline
    def total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return naively_compute_total_iters[BM, BN](self, seq_id, row, num_cols)

    @staticmethod
    def count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 1

    @always_inline
    def last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return ceildiv(num_cols, UInt32(BN))

    @always_inline
    def masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        return {
            self.last_masked_set_end[BM, BN, page_size](seq_id, row, num_cols)
        }

    @staticmethod
    def nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        # `AndMask` visibility is the *union* of the inners' visibility, so its
        # non-full region can be DISCONTIGUOUS (a `FULL` gap between two
        # non-overlapping operands) and its `NO_MASK` runs can sit anywhere --
        # neither is expressible as a fixed comptime set sequence the way
        # `OrMask`'s contiguous `{PARTIAL, NO_MASK, PARTIAL}` is (per-row
        # contiguity is a runtime property the trait surface can't prove at
        # comptime). Since `AndMask` is not instantiated by any kernel today,
        # we keep the always-safe `{UNKNOWN_MASK}` fallback (runtime per-tile
        # `status()` skip) rather than a partial/unsound merge. See `OrMask`
        # for the merge machinery to lift here once a contiguous `AndMask`
        # composition actually needs it.
        return {TileMaskStatus.UNKNOWN_MASK}

    @staticmethod
    def mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        # `AndMask` masks off an element iff BOTH inners mask it off,
        # i.e. union of visibility. The float branch is `max(lhs, rhs)`
        # (picks the less-masked operand) and `mask_bits` is `lhs | rhs`
        # (visible iff either inner says visible). We can route through
        # BITMASK iff both inners provide a real `mask_bits` (no `COMPUTED`
        # in any partition).
        comptime if (
            _supports_bitmask[Self.T, BM, BN]()
            and _supports_bitmask[Self.S, BM, BN]()
        ):
            return {MaskStrategy.BITMASK}
        else:
            return {MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS}

    @always_inline
    def mask_bits(
        self,
        seq_id: UInt32,
        score_row: Int32,
        col_start: Int32,
        num_keys: Int32,
    ) -> UInt32:
        # Bitwise OR of inners' visibility bits — visible iff either inner
        # says visible. Matches the float-branch `max(lhs, rhs)`
        # (`AndMask` = mask off iff both mask off = union of visibility).
        # Unreachable when `mask_strategies` falls back to
        # `COMPUTED | OUT_OF_BOUNDS`.
        return self.lhs.mask_bits(seq_id, score_row, col_start, num_keys) | (
            self.rhs.mask_bits(seq_id, score_row, col_start, num_keys)
        )

    @staticmethod
    def sliding_window_size() -> Int:
        # AND-of-visibility tightens: the tighter (smaller non-zero) window
        # dominates. 0 means unbounded.
        comptime l = Self.T.sliding_window_size()
        comptime r = Self.S.sliding_window_size()
        return r if l == 0 else (l if r == 0 else min(l, r))


# ===-----------------------------------------------------------------------===#
# OrMask
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct OrMask[T: MHAMask, S: MHAMask, //, lhs: T, rhs: S](
    MHAMask, TrivialRegisterPassable
):
    """Mask that's the OR of two masks.
    If either mask masks off an element, the element is masked off.

    Parameters:
        T: The type of the first (left) inner mask.
        S: The type of the second (right) inner mask.
        lhs: The first (left) inner mask instance.
        rhs: The second (right) inner mask instance.
    """

    comptime apply_log2e_after_mask: Bool = Self.T.apply_log2e_after_mask or Self.S.apply_log2e_after_mask
    comptime mask_out_of_bound: Bool = Self.T.mask_out_of_bound and Self.S.mask_out_of_bound
    comptime mask_safe_out_of_bounds: Bool = Self.T.mask_safe_out_of_bounds and Self.S.mask_safe_out_of_bounds
    comptime check_mask_during_decoding: Bool = Self.T.check_mask_during_decoding or Self.S.check_mask_during_decoding

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "OrMask"

    @staticmethod
    def name() -> String:
        return "OrMask[" + Self.T.name() + ", " + Self.S.name() + "]"

    @always_inline
    def mask[
        dtype: DType,
        width: SIMDLength,
        //,
        *,
        element_type: DType = .uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime if dtype == .bool or dtype.is_integral():
            return self.lhs.mask(coord, score_vec) | self.rhs.mask(
                coord, score_vec
            )
        else:
            return min(
                self.lhs.mask(coord, score_vec),
                self.rhs.mask(coord, score_vec),
            )

    @always_inline
    def status[
        *, element_type: DType = .uint32
    ](
        self,
        seq_id: UInt32,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        var lhs_status = self.lhs.status(seq_id, tile_offset, tile_size)
        var rhs_status = self.rhs.status(seq_id, tile_offset, tile_size)
        return lhs_status | rhs_status

    @always_inline
    def start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32) -> UInt32:
        # Closed form, so O(1) instead of the O(start_column / BN)
        # `status()`-stepping scan the generic fallback runs. That scan sits on
        # the HOST dispatch path for every caller that sizes split-K from mask
        # geometry (`sm100/dispatch.mojo`'s `_visible_keys`, `mha.mojo`'s FA2
        # decode correction), where a `ChunkedCausalMask` over a long cache cost
        # ~960 iterations per launch.
        #
        # WHY `max`: an `OrMask` masks an element off when EITHER child does, so
        # its visible region is the INTERSECTION of the children's. Each child's
        # non-`FULL_MASK` region is an interval `[start_i, end_i]`, so the
        # intersection begins at `max(start_i)` -- exactly what the scan finds
        # (`ChunkedCausalMask` = `Causal | Chunked` => `max(0, chunk_start)`).
        # Were a child's visible region ever NOT an interval, `max` is a LOWER
        # bound on the true first non-full column, which is the safe direction:
        # iteration starts earlier over a superset of the non-skipped tiles and
        # never skips a visible key. It is also better behaved than the scan when
        # the intersection is EMPTY -- the scan has no upper bound and walks past
        # `num_keys`, while this returns a finite value.
        #
        # `align_down` to BN keeps `start_column_alignment`'s promise below
        # intact: a child may align only to `min(page_size, BN)`, so a bare `max`
        # could return a non-BN-multiple while this type advertises BN. It also
        # makes the result bit-identical to the scan, which yields the first
        # non-full BN-*tile*, i.e. `align_down(first_visible, BN)` --
        # `align_down(align_down(x, page_size), BN) == align_down(x, BN)`
        # because BN is a multiple of `page_size` whenever `page_size < BN`.
        return align_down(
            max(
                self.lhs.start_column[BM, BN, page_size](seq_id, row),
                self.rhs.start_column[BM, BN, page_size](seq_id, row),
            ),
            UInt32(BN),
        )

    @staticmethod
    def start_column_alignment[BM: Int, BN: Int, page_size: Int]() -> Int:
        # `start_column` above aligns its result down to BN.
        return BN

    @always_inline
    def total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return naively_compute_total_iters[BM, BN](self, seq_id, row, num_cols)

    @staticmethod
    def count_nonfull_sets(BM: Int, BN: Int) -> Int:
        # 3 ({PARTIAL, NO_MASK, PARTIAL}) when both inners expose a `NO_MASK`
        # band (so a combined `NO_MASK` middle can exist); otherwise 1 (a
        # single `PARTIAL` set when both inners are known, else `UNKNOWN`).
        if _both_multiset[Self.T, Self.S](BM, BN):
            return 3
        else:
            return 1

    @always_inline
    def last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        comptime if (
            _nonfull_sets_known[Self.T, BM, BN]()
            and _nonfull_sets_known[Self.S, BM, BN]()
        ):
            # Tight, `start_column`-relative span (== `total_iters`, since the
            # `OrMask` non-full region is contiguous).
            return self.total_iters[BM, BN, page_size](seq_id, row, num_cols)
        else:
            return ceildiv(num_cols, UInt32(BN))

    @always_inline
    def masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        comptime if _both_multiset[Self.T, Self.S](BM, BN):
            # 3-set {PARTIAL, NO_MASK, PARTIAL}, cumulative and relative to
            # `start_column` (= the combined first non-FULL tile). The
            # `NO_MASK` middle is the intersection of the two inners' `NO_MASK`
            # bands, lifted into a common absolute-column frame and lowered
            # back to relative BN-tiles.
            var cs_col = self.start_column[BM, BN, page_size](seq_id, row)
            var total = self.total_iters[BM, BN, page_size](
                seq_id, row, num_cols
            )
            var ce_col = cs_col + total * UInt32(BN)
            var l_start = self.lhs.start_column[BM, BN, page_size](seq_id, row)
            var r_start = self.rhs.start_column[BM, BN, page_size](seq_id, row)
            # The absolute-column intersection is exact only when both inners'
            # start columns lie on the BN grid (offsets are multiples of BN).
            # This holds for every real composition (CausalMask starts at 0;
            # ChunkedMask's 3-set starts at a chunk boundary, a multiple of
            # BN). Otherwise leave the `NO_MASK` middle empty -> all PARTIAL.
            if (l_start % UInt32(BN) == 0) and (r_start % UInt32(BN) == 0):
                var l_nm = _child_nomask_cols[BM, BN, page_size](
                    self.lhs, seq_id, row, num_cols
                )
                var r_nm = _child_nomask_cols[BM, BN, page_size](
                    self.rhs, seq_id, row, num_cols
                )
                var nm_lo = max(l_nm[0], r_nm[0])
                var nm_hi = min(l_nm[1], r_nm[1])
                nm_lo = min(max(nm_lo, cs_col), ce_col)
                nm_hi = min(max(nm_hi, nm_lo), ce_col)
                return {
                    (nm_lo - cs_col) // UInt32(BN),
                    (nm_hi - cs_col) // UInt32(BN),
                    total,
                }
            else:
                return {UInt32(0), UInt32(0), total}
        else:
            comptime if (
                _nonfull_sets_known[Self.T, BM, BN]()
                and _nonfull_sets_known[Self.S, BM, BN]()
            ):
                # Single PARTIAL set spanning the (contiguous) intersection.
                return {
                    self.total_iters[BM, BN, page_size](seq_id, row, num_cols)
                }
            else:
                return {
                    self.last_masked_set_end[BM, BN, page_size](
                        seq_id, row, num_cols
                    )
                }

    @staticmethod
    def nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        # Precise partition when both inners are statically known. `OrMask`
        # visibility is the *intersection*, whose non-full region is contiguous
        # and whose `NO_MASK` band is `intersect(lhs_nm, rhs_nm)` (one
        # interval) -> the `{PARTIAL, NO_MASK, PARTIAL}` shape, possible only
        # when both inners expose a `NO_MASK` band (`_both_multiset`). When
        # only one (or neither) does but both are known, no combined `NO_MASK`
        # can exist, so a single `PARTIAL` set spanning the intersection
        # suffices. Fall back to `UNKNOWN_MASK` only when an inner is itself
        # unknown (e.g. `MaterializedMask`).
        comptime if _both_multiset[Self.T, Self.S](BM, BN):
            return {
                TileMaskStatus.PARTIAL_MASK,
                TileMaskStatus.NO_MASK,
                TileMaskStatus.PARTIAL_MASK,
            }
        else:
            comptime if (
                _nonfull_sets_known[Self.T, BM, BN]()
                and _nonfull_sets_known[Self.S, BM, BN]()
            ):
                return {TileMaskStatus.PARTIAL_MASK}
            else:
                return {TileMaskStatus.UNKNOWN_MASK}

    @staticmethod
    def mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        # `OrMask` masks off an element iff AT LEAST ONE inner masks it off,
        # i.e. intersection of visibility (`status = lhs | rhs`, float branch
        # `min(lhs, rhs)`, `mask_bits = lhs & rhs`). `ChunkedCausalMask =
        # OrMask[CausalMask, ChunkedMask]` relies on this.
        #
        # `edge` is the strategy for the (leading/trailing) PARTIAL sets:
        # BITMASK iff both inners provide a real `mask_bits` (no `COMPUTED`),
        # else the per-element `COMPUTED | OUT_OF_BOUNDS` path. The combined
        # `NO_MASK` middle is fully visible, so it needs no `mask_bits`/select
        # -- only `OUT_OF_BOUNDS` to keep the `col < num_keys` clip (the band
        # is `intersect(lhs_nm, rhs_nm)` and can inherit a `num_keys`-straddling
        # tile from an inner).
        comptime both_bm = _supports_bitmask[
            Self.T, BM, BN
        ]() and _supports_bitmask[Self.S, BM, BN]()
        comptime edge = MaskStrategy.BITMASK if both_bm else (
            MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS
        )
        comptime if _both_multiset[Self.T, Self.S](BM, BN):
            return {edge, MaskStrategy.OUT_OF_BOUNDS, edge}
        else:
            return {edge}

    @always_inline
    def mask_bits(
        self,
        seq_id: UInt32,
        score_row: Int32,
        col_start: Int32,
        num_keys: Int32,
    ) -> UInt32:
        # Bitwise AND of inners' visibility bits — visible iff both inners
        # say visible. Matches the float-branch `min(lhs, rhs)` (`OrMask` =
        # mask off iff either masks off = intersection of visibility).
        # Unreachable when `mask_strategies` falls back to
        # `COMPUTED | OUT_OF_BOUNDS`, but the body is unconditional for
        # trait satisfaction.
        return self.lhs.mask_bits(seq_id, score_row, col_start, num_keys) & (
            self.rhs.mask_bits(seq_id, score_row, col_start, num_keys)
        )

    @staticmethod
    def sliding_window_size() -> Int:
        # AND-of-visibility tightens: the tighter (smaller non-zero) window
        # dominates. 0 means unbounded.
        comptime l = Self.T.sliding_window_size()
        comptime r = Self.S.sliding_window_size()
        return r if l == 0 else (l if r == 0 else min(l, r))


# ===-----------------------------------------------------------------------===#
# ChunkedCausalMask
# ===-----------------------------------------------------------------------===#


@always_inline
def ChunkedCausalMask[
    local_window_size: Int
](out res: OrMask[CausalMask(), ChunkedMask[local_window_size]()]):
    """Mask implementing Chunked Causal attention for Llama4 models.

    This groups the mask into chunks of size `local_window_size` and performs causal
    attention within each local chunk. Considering the following case:
    - Q_len = 7
    - K_len = 10
    - start_pos = 3
    - local_window_size = 4

    The mask will be applied as follows:
        K > 0 1 2 3 4 5 6 7 8 9
        Q v x--------------------x
        0 | 1 1 1 1 0 0 0 0 0 0
        1 | 0 0 0 0 1 0 0 0 0 0
        2 | 0 0 0 0 1 1 0 0 0 0
        3 | 0 0 0 0 1 1 1 0 0 0
        4 | 0 0 0 0 1 1 1 1 0 0
        5 | 0 0 0 0 0 0 0 0 1 0
        6 | 0 0 0 0 0 0 0 0 1 1
    """
    res = {}


# ===-----------------------------------------------------------------------===#
# RelativeLogitsMask
# ===-----------------------------------------------------------------------===#


struct RelativeLogitsMask[
    V: MHAMask,
    //,
    visibility: V,
    dtype_: DType,
    layout_: Layout,
    origin_: Origin[mut=False],
](MHAMask, TrivialRegisterPassable):
    """Causal (optionally sliding-window) mask plus an additive relative-position bias.

    The bias is gathered by `rel_dist = q_pos - k_pos` from a `(tokens,
    heads, extent)` table and added on every visible position; distances
    outside `[0, extent)` carry no bias. `mask()` adds the bias before
    delegating the visibility select to `visibility.mask()`, so masked lanes
    come out as exactly `MASK_VALUE` (never `MASK_VALUE + bias`).

    Parameters:
        visibility: `CausalMask()` (global) or
            `SlidingWindowCausalMask[window_size]()` (local).
        dtype_: Element type of the bias tensor.
        layout_: Layout of the bias tensor, rank 3
            `(total_q_tokens, heads, extent)`.
        origin_: Origin of the bias tensor.
    """

    comptime window_size: Int = Self.V.sliding_window_size()
    """`0` means unbounded (plain causal), per this file's convention."""

    comptime apply_log2e_after_mask: Bool = True
    comptime mask_out_of_bound: Bool = is_nvidia_gpu()
    comptime mask_safe_out_of_bounds: Bool = True
    comptime check_mask_during_decoding: Bool = True

    var bias: LayoutTensor[Self.dtype_, Self.layout_, Self.origin_]
    """`(total_q_tokens, heads, extent)`, row `r` matching `q`'s ragged-flat row."""

    @__allow_legacy_any_origin_fields
    var cache_lengths: LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ]
    """Cached tokens before this call's new tokens."""

    @__allow_legacy_any_origin_fields
    var input_row_offsets: LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ]
    """Ragged row offset into `bias`/`q` for this call's new tokens, `(batch + 1,)`."""

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "RelativeLogitsMask"

    @staticmethod
    def name() -> String:
        return "RelativeLogitsMask[" + String(Self.window_size) + "]"

    def __init__(
        out self,
        bias: LayoutTensor[Self.dtype_, Self.layout_, Self.origin_],
        cache_lengths: LayoutTensor[
            .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ],
        input_row_offsets: LayoutTensor[
            .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ],
    ):
        comptime assert (
            Self.layout_.rank() == 3
        ), "Expected rank 3 (tokens, heads, extent) for the bias tensor"
        comptime visibility_name = Self.V.get_type_name()
        comptime assert (
            visibility_name == CausalMask.get_type_name()
            or visibility_name == SlidingWindowCausalMask[1].get_type_name()
        ), (
            "RelativeLogitsMask's visibility mask must be CausalMask or"
            " SlidingWindowCausalMask"
        )
        self.bias = bias
        self.cache_lengths = cache_lengths
        self.input_row_offsets = input_row_offsets

    @always_inline
    def _flat_row(self, seq_id: Int, q_idx_abs: Int) -> Int:
        # `q_idx_abs` is the query's absolute in-sequence position (cached
        # prefix + new tokens); subtract the cached length and add this
        # sequence's ragged offset to recover the global flat row into `bias`.
        var cache_len = Int(self.cache_lengths[seq_id])
        var row_offset = Int(self.input_row_offsets[seq_id])
        return row_offset + (q_idx_abs - cache_len)

    @always_inline
    def mask[
        dtype: DType,
        width: SIMDLength,
        //,
        *,
        element_type: DType = .uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        var q_idx = coord[2]
        var k_idx = coord[3]

        # Gather the bias by relative distance, zero outside [0, extent).
        var flat_row = self._flat_row(Int(coord[0]), Int(q_idx))
        var head = Int(coord[1])
        var extent = self.bias.dim[2]()
        if extent == 0:
            return Self.visibility.mask(coord, score_vec)
        var num_rows = self.bias.dim[0]()
        var row_in_bounds = 0 <= flat_row < num_rows
        # Clamp to an in-bounds row once; the gather below then has no
        # data-dependent skip on it.
        var safe_flat_row = flat_row if row_in_bounds else 0
        var bias_vec = SIMD[dtype, width](0)
        comptime for i in range(width):
            var rel_dist = Int(q_idx) - (Int(k_idx) + i)
            var dist_in_bounds = rel_dist >= 0 and rel_dist < extent
            # Clamp the distance so the gather always reads an in-bounds
            # entry; `valid` decides whether the value is used. Reading
            # unconditionally avoids a per-lane conditional store.
            var safe_rel_dist = max(0, min(rel_dist, extent - 1))
            var valid = dist_in_bounds and row_in_bounds
            # `bias[...]` returns a 1-element SIMD whose width is a
            # layout-size expression, not the literal `1`; `rebind` bridges
            # the two comptime spellings (not a numeric cast).
            var raw_bias = self.bias[safe_flat_row, head, safe_rel_dist].cast[
                dtype
            ]()
            bias_vec[i] = rebind[Scalar[dtype]](raw_bias) if valid else Scalar[
                dtype
            ](0)

        # Add the bias before the visibility select so masked lanes come out
        # as exactly `MASK_VALUE` (never `MASK_VALUE + bias`). The visibility
        # mask's own `mask()` is the select, so this can't drift from it.
        return Self.visibility.mask(coord, score_vec + bias_vec)

    @always_inline
    def status[
        *, element_type: DType = .uint32
    ](
        self,
        seq_id: UInt32,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        # `visibility` owns the FULL_MASK boundary; downgrade its NO_MASK
        # (skip `mask()`) to PARTIAL_MASK, since the bias must be added on
        # every visible tile.
        var inner = Self.visibility.status(seq_id, tile_offset, tile_size)
        return (
            TileMaskStatus.PARTIAL_MASK if inner
            == TileMaskStatus.NO_MASK else inner
        )

    @always_inline
    def start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32) -> UInt32:
        return Self.visibility.start_column[BM, BN, page_size](seq_id, row)

    @staticmethod
    def start_column_alignment[BM: Int, BN: Int, page_size: Int]() -> Int:
        return Self.V.start_column_alignment[BM, BN, page_size]()

    @always_inline
    def total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return Self.visibility.total_iters[BM, BN, page_size](
            seq_id, row, num_cols
        )

    @staticmethod
    def count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 1

    @always_inline
    def last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return self.total_iters[BM, BN, page_size](seq_id, row, num_cols)

    @always_inline
    def masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        return {
            self.last_masked_set_end[BM, BN, page_size](seq_id, row, num_cols)
        }

    @staticmethod
    def nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        # One contiguous PARTIAL band when `visibility` publishes a known
        # partition (the bias hits every visible tile); else forward UNKNOWN.
        comptime if _nonfull_sets_known[Self.V, BM, BN]():
            return {TileMaskStatus.PARTIAL_MASK}
        else:
            return {TileMaskStatus.UNKNOWN_MASK}

    @staticmethod
    def mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        return {MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS}

    @always_inline
    def mask_bits(
        self,
        seq_id: UInt32,
        score_row: Int32,
        col_start: Int32,
        num_keys: Int32,
    ) -> UInt32:
        # Unreachable: mask_strategies never advertises BITMASK.
        return UInt32(0xFFFF_FFFF)

    @staticmethod
    def sliding_window_size() -> Int:
        return Self.window_size
