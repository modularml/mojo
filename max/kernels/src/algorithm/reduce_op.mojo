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
"""Reduction monoid traits — the device-agnostic author surface.

A reduction algorithm is authored as a struct conforming to `ReduceOp`:
inlined state fields, `__init__` (identity), `accumulate[w]` (SIMD-tile
fold), `join` (sequential combine), and optionally
`join_parallel[R: Reducer]` (parallel combine). How participants
cooperate and how the parallel scope is defined live in the `Reducer`
impls.
"""

from std.math import exp, log
from std.sys.info import simd_width_of

from std.utils.numerics import max_finite, min_finite

from std.utils.static_tuple import StaticTuple


# ===-----------------------------------------------------------------------===#
# Reducer trait
# ===-----------------------------------------------------------------------===#


trait Reducer:
    """Parallel scalar reducer over the closed set of associative ops.

    Reduces a scalar across all participants in the current parallel
    scope; every participant receives the reduced value (like
    `MPI_Allreduce`).

    Concrete impls provide hardware-fast `sum` / `max` / `min` over a
    closed set of dtypes, plus a `generic` fallback that combines any
    `ReduceOp` via its `join` (slower; dtype-agnostic).
    """

    def sum[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns the sum of `val` across all participants.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The per-participant value.

        Returns:
            The reduced sum, broadcast to every participant.
        """
        ...

    def max[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns the maximum of `val` across all participants.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The per-participant value.

        Returns:
            The reduced maximum, broadcast to every participant.
        """
        ...

    def min[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns the minimum of `val` across all participants.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The per-participant value.

        Returns:
            The reduced minimum, broadcast to every participant.
        """
        ...

    def generic[S: ReduceOp](self, mut state: S):
        """Combines `state` across all participants using only `join` —
        the multi-field path for monoids whose state isn't decomposable
        into scalar `sum`/`max`/`min` (Welford, ArgMax, ArgMin, …).

        Impls pick by state size: small states (a few uint32 words) use
        a register-only warp-shuffle butterfly, larger states a shmem
        tree — both valid since `ReduceOp` is `TrivialRegisterPassable`.
        Monoids that *can* decompose (ReduceSum, ReduceMax,
        OnlineLogSumExp, …) instead call `reducer.sum`/`max`/`min`
        directly, which compile to one hardware `redux.sync` per field
        and are faster.

        Parameters:
            S: The monoid type being combined.

        Args:
            state: Per-participant state; on return, holds the combined
                value on every participant.
        """
        ...


# ===-----------------------------------------------------------------------===#
# ReduceOp trait
# ===-----------------------------------------------------------------------===#


trait ReduceOp(TrivialRegisterPassable):
    """Trait for an associative reduction monoid (identity +
    associative `join`).

    Required: `__init__` (identity), `accumulate[w]` (SIMD-tile fold),
    `join` (sequential combine). Optional: `join_parallel[R: Reducer]`
    (parallel combine across participants). The trait's default
    delegates to `reducer.generic` (only needs `join`); override to use
    a `Reducer`'s hardware-fast scalar primitives, or a field-wise
    algebraic decomposition that collapses many `join`s into a few
    scalar reductions.

    `TrivialRegisterPassable` because accumulators pass by value through
    the parallel reducer.

    `accumulate` carries a free `val_dtype` parameter because Mojo trait
    conformance treats a struct-param `dtype` and a trait-declared
    `comptime dtype` as distinct types; impls cast `val` to the
    accumulator dtype inline.

    Constraints:
        `join` must be associative. The default `join_parallel` also
        assumes commutativity (participants reduce in unspecified
        order); non-commutative monoids must override it.
    """

    comptime Single: ReduceOp
    """The width-1 form of this monoid (`Self` at `W == 1`), produced by
    `reduce` and consumed by the scalar broadcast constructor."""

    comptime width: Int
    """The SIMD width `W` of the state, exposed so the trait's default
    `reduce` / broadcast iterate the lanes generically."""

    def __init__(out self):
        """Returns the identity element."""
        ...

    def __init__(out self, s: Self.Single):
        """Broadcasts a width-1 result across all `W` lanes — the inverse of
        `reduce`, so the body can read `result.slice[w]` uniformly. Default:
        write every lane via `__setitem__`."""
        self = Self()
        comptime for j in range(Self.width):
            self[j] = s

    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        """Folds a SIMD tile into `self` **elementwise** — lane `j`
        folds `(val[j], idx[j])` — casting to the accumulator dtype
        internally.

        `idx[j]` is where `val[j]` sits along the reduce axis. The
        scaffolder builds it per tier — consecutive positions
        (`base + iota`) when the lanes are partials of one output, or a
        constant (`base`) when the lanes are independent output columns
        — so the monoid never learns its tier. Value-only monoids
        (`ReduceSum`, `ReduceMax`, `ReduceMin`, `ReduceProduct`,
        `OnlineLogSumExp`) ignore `idx`; index-tracking monoids
        (`ArgMax`, `ArgMin`) fold it directly. Defaults to a zero vector
        for callers that don't track positions.

        Parameters:
            val_dtype: The tile's source dtype.
            w: SIMD width of the tile.

        Args:
            val: The SIMD tile to fold.
            idx: Per-lane axis positions of `val`'s lanes (defaults to
                a zero vector).
        """
        ...

    def join(mut self, other: Self):
        """Combines `self` with `other` associatively (sequential).

        Args:
            other: The state to combine into `self`.
        """
        ...

    def __getitem__(self, j: Int) -> Self.Single:
        """Returns lane `j` of the state as a width-1 monoid."""
        ...

    def __setitem__(mut self, j: Int, s: Self.Single):
        """Writes the width-1 monoid `s` into lane `j` of the state."""
        ...

    def reduce(self) -> Self.Single:
        """Collapses the `W` lane-wise partials of one output into a single
        width-1 result.

        The scaffolder calls it once, immediately before `join_parallel`, and
        **only on cooperative tiers** (`W` lanes are partials of one output).
        On the tiled tier the `W` lanes are independent outputs, so both
        `reduce` and `join_parallel` are skipped and the per-lane results
        stand.

        Default: fold the lanes through `join` (the loop is empty at
        `W == 1`, so this is a branchless identity there). Override for a
        faster horizontal reduce (e.g. `acc.reduce_add()` for `ReduceSum`).
        """
        comptime assert Self.width > 0, "ReduceOp width must be positive"
        var acc = self[0]
        comptime for j in range(1, Self.width):
            acc.join(self[j])
        return acc

    @always_inline
    def join_parallel[R: Reducer](mut self, reducer: R):
        """Combines `self` across all participants via `reducer`,
        leaving the combined value on every participant (and, for SIMD
        states, every lane).

        Runs **after** `reduce`, so `self`'s lanes already hold a
        single within-participant result; this only does the
        cross-participant step. Default: `reducer.generic(self)` —
        correct for any commutative + associative monoid, and since
        `reduce` left all lanes equal, the generic field-wise combine
        yields the result on every lane (no separate broadcast).
        Override to dispatch to `reducer.sum`/`max`/`min` for a faster
        per-dtype path, splatting the scalar result back across lanes.

        Parameters:
            R: The parallel scalar reducer.

        Args:
            reducer: The reducer instance.
        """
        reducer.generic(self)

    @staticmethod
    @always_inline
    def pad[
        Wout: Int,
        Tout: DType,
        identity: Scalar[Tout],
        val_dtype: DType,
        w_in: Int,
    ](val: SIMD[val_dtype, w_in]) -> SIMD[Tout, Wout]:
        """Pads `val` to width `Wout`, casting `val_dtype -> Tout`
        and filling lanes `[w_in, Wout)` with `identity`.

        Lets a monoid's `accumulate[w]` body stay a single expression —
        no `comptime if w == Self.W` branching. For `w_in == Wout`
        returns the cast directly; for `w_in < Wout` builds an
        identity-filled vector and overwrites lanes `[0, w_in)` with the
        cast `val`.

        Precondition: `w_in <= Wout`. Bodies arrange this by computing
        `W = rowwise.pick_simd_width[...]` once across all monoids and
        dtypes involved, so the scaffolder's tile width never exceeds
        any monoid's `Self.W`.

        The default suffices for single-field SIMD monoids. Multi-field
        monoids (`ArgMax`, `ArgMin`) call `pad` twice — once per field —
        with each field's own identity.

        Parameters:
            Wout: Output SIMD width.
            Tout: Output element dtype.
            identity: Padding value for tail lanes.
            val_dtype: Input element dtype.
            w_in: Input SIMD width.

        Args:
            val: The partial-width input.

        Returns:
            A `SIMD[Tout, Wout]` with `val` in lanes `[0, w_in)`
            and `identity` in lanes `[w_in, Wout)`.
        """
        comptime if w_in == Wout:
            return rebind[SIMD[Tout, Wout]](val.cast[Tout]())
        var out = SIMD[Tout, Wout](identity)

        comptime for j in range(w_in):
            out[j] = val[j].cast[Tout]()
        return out


# ===-----------------------------------------------------------------------===#
# Online log-sum-exp monoid (softmax / log-softmax reduction half)
# ===-----------------------------------------------------------------------===#


struct OnlineLogSumExp[
    dtype: DType,
    W: Int = simd_width_of[dtype](),
](ReduceOp):
    """Online (flash-style) log-sum-exp monoid (Milakov & Gimelshein,
    2018) — the reduction half of softmax.

    Holds `{m, l}` — the sufficient statistics for
    `log(sum_i exp(x_i))`, with `m = max_i x_i` and
    `l = sum_i exp(x_i - m)`. `LSE = m + log(l)`. Downstream
    normalization (`exp(x - m) / l` for softmax, `x - m - log(l)` for
    log-softmax) lives in the kernel's pass-2 body.

    Single field per statistic: `m: SIMD[dtype, W]` and
    `l: SIMD[dtype, W]`. During accumulation each lane runs an
    independent flash state over its slice of the axis; `join_parallel`
    flash-combines the `W` lanes into lane 0. Bodies read `m[0]` and
    `l[0]`.

    Parameters:
        dtype: The accumulator dtype (must be floating-point).
        W: SIMD width of the per-lane flash accumulators.
            Defaults to the target's `simd_width_of[dtype]`.
    """

    comptime Single = OnlineLogSumExp[Self.dtype, 1]
    comptime width = Self.W

    var m: SIMD[Self.dtype, Self.W]
    """`W`-lane running maxima. Final scalar at `m[0]` post-`join_parallel`."""

    var l: SIMD[Self.dtype, Self.W]
    """`W`-lane running `sum exp(x - m)`. Final scalar at `l[0]`
    post-`join_parallel`."""

    @always_inline
    def __init__(out self):
        """Identity: every lane at `(-inf, 0)`."""
        comptime assert (
            Self.dtype.is_floating_point()
        ), "OnlineLogSumExp requires a floating-point dtype"
        self.m = SIMD[Self.dtype, Self.W](min_finite[Self.dtype]())
        self.l = SIMD[Self.dtype, Self.W](0)

    @always_inline
    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        """Folds a SIMD tile via lane-wise flash update. `val` is
        padded to width `Self.W` via `Self.pad` with `-inf` tail lanes;
        padded lanes contribute nothing because `exp(-inf - new_m) = 0`,
        and a lane-wise active-mask zeroes the correction factor where
        both `m` and `val` are `-inf` (avoids `exp(NaN)`).

        Parameters:
            val_dtype: The tile's source dtype.
            w: SIMD width of the tile.

        Args:
            val: The SIMD tile to fold.
            idx: Unused (LSE is index-agnostic).
        """
        comptime assert (
            Self.dtype.is_floating_point()
        ), "OnlineLogSumExp requires a floating-point dtype"
        comptime neg_inf = min_finite[Self.dtype]()
        var v_padded = Self.pad[Self.W, Self.dtype, neg_inf](val)
        # Early exit if the whole (padded) tile is `-inf`.
        if v_padded.reduce_max() <= neg_inf:
            return
        # Single-`exp` flash update. Of the two flash diffs
        # (`self.m - new_m`, `val - new_m`) exactly one is `0` per lane
        # — the side holding the new max — so its `exp` is `1`. Compute
        # `exp` only of the loser's diff (`lo - new_m`) and fold it onto
        # the correct side via a per-lane select. Halves the
        # transcendental count in the hot fold vs
        # `l*exp(self_diff) + exp(val_diff)`, which matters once the row
        # is compute-bound (narrow dtypes, wide rows).
        var new_m = max(self.m, v_padded)
        var lo = min(self.m, v_padded)
        var lane_active = new_m.gt(SIMD[Self.dtype, Self.W](neg_inf))
        # Inactive lanes (both `-inf`): force the diff to `-inf` so
        # `exp = 0` instead of `(-inf) - (-inf) = NaN`.
        var loser = lane_active.select(
            lo - new_m, SIMD[Self.dtype, Self.W](neg_inf)
        )
        var e = exp(loser)
        # `self` won → add the loser's contribution; `val` won → rescale
        # the running sum by it and add the new max's `exp(0) = 1`.
        var self_is_max = self.m.ge(v_padded)
        self.l = self_is_max.select(
            self.l + e, self.l * e + SIMD[Self.dtype, Self.W](1)
        )
        self.m = new_m

    @always_inline
    def join(mut self, other: Self):
        """Sequential combine: lane-wise flash combine.

        Args:
            other: The state to combine into `self`.
        """
        comptime assert (
            Self.dtype.is_floating_point()
        ), "OnlineLogSumExp requires a floating-point dtype"
        comptime neg_inf = min_finite[Self.dtype]()
        # Single-`exp` flash combine (same trick as `accumulate`): one
        # diff is `0`, so rescale only the loser's side.
        var new_m = max(self.m, other.m)
        var lo = min(self.m, other.m)
        var lane_active = new_m.gt(SIMD[Self.dtype, Self.W](neg_inf))
        var loser = lane_active.select(
            lo - new_m, SIMD[Self.dtype, Self.W](neg_inf)
        )
        var e = exp(loser)
        var self_is_max = self.m.ge(other.m)
        self.l = self_is_max.select(self.l + other.l * e, self.l * e + other.l)
        self.m = new_m

    @always_inline
    def __getitem__(self, j: Int) -> Self.Single:
        """Returns lane `j` as a width-1 monoid."""
        var r = Self.Single()
        r.m[0] = self.m[j]
        r.l[0] = self.l[j]
        return r

    @always_inline
    def __setitem__(mut self, j: Int, s: Self.Single):
        """Writes width-1 monoid `s` into lane `j`."""
        self.m[j] = s.m[0]
        self.l[j] = s.l[0]

    @always_inline
    def reduce(self) -> Self.Single:
        """Flash-combines the `W` lane partials into one `(m, l)` via the
        `reduce_max` / `reduce_add` intrinsics.

        Explicit horizontal reduce: global `max` of `m`, then the
        max-corrected sum `sum_j l[j] * exp(m[j] - lane_max)`. Lanes with
        `m[j] = -inf` carry `l[j] = 0`, so their corrected term is 0.

        Returns:
            A width-1 `OnlineLogSumExp` with the combined `(m, l)` in
            lane 0.
        """
        comptime assert (
            Self.dtype.is_floating_point()
        ), "OnlineLogSumExp requires a floating-point dtype"
        var r = Self.Single()
        var lane_max = self.m.reduce_max()
        var diff = self.m - SIMD[Self.dtype, Self.W](lane_max)
        r.m[0] = lane_max
        r.l[0] = (self.l * exp(diff)).reduce_add()
        return r

    @always_inline
    def join_parallel[R: Reducer](mut self, reducer: R):
        """Cross-thread flash combine: global `max` on `m[0]`,
        per-participant correction `l *= exp(m[0] - new_m)`, global
        `sum` on `l`, both splatted across all lanes. Runs after
        `reduce`, so `m[0]`/`l[0]` already hold the within-thread flash
        state.

        Parameters:
            R: The parallel scalar reducer.

        Args:
            reducer: The reducer instance.
        """
        comptime assert (
            Self.dtype.is_floating_point()
        ), "OnlineLogSumExp requires a floating-point dtype"
        comptime neg_inf = min_finite[Self.dtype]()
        comptime if Self.dtype in (
            DType.float16,
            DType.bfloat16,
            DType.float32,
        ):
            var global_m = reducer.max(self.m[0])
            var corrected_l = self.l[0]
            if global_m > neg_inf and self.m[0] > neg_inf:
                corrected_l *= exp(self.m[0] - global_m)
            self.m[0] = global_m
            self.l[0] = reducer.sum(corrected_l)
        else:
            reducer.generic(self)


# ===-----------------------------------------------------------------------===#
# Welford running-mean-variance monoid (layer_norm / group_norm reduction half)
# ===-----------------------------------------------------------------------===#


struct Welford[
    dtype: DType,
    W: Int = 1,
](ReduceOp):
    """Welford's online mean/variance monoid (Welford 1962, Chan et
    al. 1979 for the combine) — the reduction half of layer_norm /
    group_norm.

    Holds `{count, mean, M2}` — running sample count, mean, and sum of
    squared deviations from the mean — lane-wise across a width-`W` SIMD
    field. Variance is `M2 / count`. Combining two sub-states uses
    Chan's formula:

        n   = na + nb
        δ   = μb - μa
        μ   = μa + δ * nb / n
        M2  = M2_a + M2_b + δ² * na * nb / n

    `accumulate[W]` is **elementwise**: each call folds one new sample
    into every lane's Welford (Chan with `nb = 1`). Because every lane
    advances its count in lock-step, `1 / count` is a single **scalar**
    reciprocal — sidestepping the per-lane SIMD `fdiv` that serializes
    through the GPU divider — with no per-tile horizontal reduce.
    `reduce` Chan-combines the `W` lanes once at the end. On the tiled
    tier the lanes are independent output columns, so
    `reduce`/`join_parallel` are skipped and each lane's Welford stands
    as that column's statistics.

    Triple-field state: `count`, `mean`, `M2`, each a `SIMD[dtype, W]`.
    `join_parallel` cross-thread-combines through `reducer.generic` —
    Welford's combine isn't a hardware primitive.

    Parameters:
        dtype: The accumulator dtype (must be floating-point).
        W: SIMD width of the per-field storage. Defaults to `1` (scalar
            state). Cooperative tiers pass the body's SIMD width so each
            lane accumulates a strided slice of the row; the tiled tier
            passes `tile_width` so each lane is an output column.
    """

    comptime Single = Welford[Self.dtype, 1]
    comptime width = Self.W

    var count: SIMD[Self.dtype, Self.W]
    """Per-lane running sample count (uniform across lanes during
    accumulation). After `reduce`, all lanes hold the combined
    count."""

    var mean: SIMD[Self.dtype, Self.W]
    """Per-lane running mean. After `reduce`, all lanes hold the
    combined mean (cooperative); on the tiled tier each lane is its
    column's mean."""

    var M2: SIMD[Self.dtype, Self.W]
    """Per-lane running sum-of-squared-deviations-from-the-mean.
    Variance is `M2 / count`. After `reduce`, all lanes hold the
    combined `M2` (cooperative); on the tiled tier each lane is its
    column's `M2`."""

    @always_inline
    def __init__(out self):
        """Identity: every lane at `(0, 0, 0)`."""
        comptime assert (
            Self.dtype.is_floating_point()
        ), "Welford requires a floating-point dtype"
        self.count = SIMD[Self.dtype, Self.W](0)
        self.mean = SIMD[Self.dtype, Self.W](0)
        self.M2 = SIMD[Self.dtype, Self.W](0)

    @always_inline
    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        """Elementwise Welford fold: each lane folds one new sample
        into its running state (Chan with `nb = 1`).

        For a full tile (`w == W`) every lane advances its count in
        lock-step, so `1 / new_count` is a single scalar reciprocal and
        the SIMD work is one sub + two fmas — no horizontal reduce, no
        per-lane divide. A partial tile (`w < W`, the scalar tail of an
        odd row) masks the count increment to its first `w` lanes, which
        costs a per-lane reciprocal; tiers at `W > 1` only hit full tiles
        (the row divides the SIMD width), so this path is compiled but
        never executed there.

        Parameters:
            val_dtype: The tile's source dtype.
            w: SIMD width of the tile.

        Args:
            val: The SIMD tile to fold.
            idx: Unused (Welford is index-agnostic).
        """
        comptime assert (
            Self.dtype.is_floating_point()
        ), "Welford requires a floating-point dtype"
        var vW = Self.pad[Self.W, Self.dtype, Scalar[Self.dtype](0)](val)
        # `inc` is 1 in real lanes, 0 in tail-pad lanes (full tile: all 1s).
        var inc = Self.pad[Self.W, Self.dtype, Scalar[Self.dtype](0)](
            SIMD[Self.dtype, w](1)
        )
        var na = self.count
        var new_count = na + inc
        var delta = vW - self.mean
        var inv: SIMD[Self.dtype, Self.W]
        comptime if w == Self.W:
            # Full tile: count is uniform, so a single broadcast scalar
            # reciprocal — no per-lane SIMD divide.
            inv = SIMD[Self.dtype, Self.W](Scalar[Self.dtype](1) / new_count[0])
        else:
            # Partial tail: per-lane reciprocal masked to the first `w`
            # lanes (inc = 0 elsewhere). Never runs at `W > 1` (full
            # tiles only there).
            inv = inc / max(new_count, SIMD[Self.dtype, Self.W](1))
        self.mean = self.mean + delta * inv
        self.M2 = self.M2 + delta * delta * na * inv
        self.count = new_count

    @always_inline
    def join(mut self, other: Self):
        """Sequential combine of two Welford states via Chan's
        formula, lane-wise.

        Args:
            other: The state to combine into `self`.
        """
        comptime assert (
            Self.dtype.is_floating_point()
        ), "Welford requires a floating-point dtype"
        var na = self.count
        var nb = other.count
        var new_count = na + nb
        var safe_n = max(new_count, SIMD[Self.dtype, Self.W](1))
        var delta = other.mean - self.mean
        var n_ratio = nb / safe_n
        self.M2 = self.M2 + other.M2 + delta * delta * na * n_ratio
        self.mean = self.mean + delta * n_ratio
        self.count = new_count

    @always_inline
    def __getitem__(self, j: Int) -> Self.Single:
        """Returns lane `j` as a width-1 monoid."""
        var r = Self.Single()
        r.count[0] = self.count[j]
        r.mean[0] = self.mean[j]
        r.M2[0] = self.M2[j]
        return r

    @always_inline
    def __setitem__(mut self, j: Int, s: Self.Single):
        """Writes width-1 monoid `s` into lane `j`."""
        self.count[j] = s.count[0]
        self.mean[j] = s.mean[0]
        self.M2[j] = s.M2[0]

    @always_inline
    def join_parallel[R: Reducer](mut self, reducer: R):
        """Cross-thread combine via `reducer.generic` (Welford's combine
        isn't a hardware primitive), then broadcast the result from lane
        0 across all lanes so a body reads `(mean, M2, count).slice[w]`
        uniformly. Runs after `reduce`.

        Parameters:
            R: The parallel reducer.

        Args:
            reducer: The reducer instance.
        """
        reducer.generic(self)
        self.mean = SIMD[Self.dtype, Self.W](self.mean[0])
        self.M2 = SIMD[Self.dtype, Self.W](self.M2[0])
        self.count = SIMD[Self.dtype, Self.W](self.count[0])


# ===-----------------------------------------------------------------------===#
# Single-field reduction monoids
# ===-----------------------------------------------------------------------===#


struct ReduceSum[
    dtype: DType,
    W: Int = simd_width_of[dtype](),
](ReduceOp):
    """Sum reduction monoid: `(self, x) -> self + x`.

    Single-field state: `acc: SIMD[dtype, W]`. `accumulate` is one
    expression — `acc += Self.pad[...](val)`. Per-tile work is one
    lane-wise SIMD add; the horizontal collapse `SIMD[W] -> Scalar`
    happens once at `reduce`, with the final scalar in `acc[0]`. Bodies
    read `state.acc[0]`.

    `W` defaults to `simd_width_of[dtype]()`, target-aware (large on CPU,
    small on GPU). On the tiled tier the monoid runs with one lane per independent
    output column; the cross-lane `reduce` / `join_parallel` are skipped.

    Parameters:
        dtype: The accumulator dtype.
        W: SIMD width of the lane-wise accumulator. Defaults to the
            target's `simd_width_of[dtype]`.
    """

    comptime Single = ReduceSum[Self.dtype, 1]
    comptime width = Self.W

    var acc: SIMD[Self.dtype, Self.W]
    """Lane-wise SIMD accumulator. Per-tile work is one lane-wise add;
    `reduce` reduces to scalar (placed in `acc[0]`). Bodies read
    `acc[0]`."""

    @always_inline
    def __init__(out self):
        """Identity: `acc = 0_W`."""
        comptime assert (
            Self.dtype.is_numeric()
        ), "ReduceSum requires a numeric dtype"
        self.acc = SIMD[Self.dtype, Self.W](0)

    @always_inline
    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        """Folds a SIMD tile into `acc` lane-wise. Single expression;
        `Self.pad` lifts partial tiles to width `Self.W` with `0` tail
        lanes.

        Parameters:
            val_dtype: The tile's source dtype.
            w: SIMD width of the tile.

        Args:
            val: The SIMD tile to fold.
            idx: Unused (Sum is index-agnostic).
        """
        self.acc += Self.pad[
            Self.W,
            Self.dtype,
            Scalar[Self.dtype](0),
        ](val)

    @always_inline
    def join(mut self, other: Self):
        """Sequential combine: lane-wise add.

        Args:
            other: The state to combine into `self`.
        """
        self.acc += other.acc

    @always_inline
    def __getitem__(self, j: Int) -> Self.Single:
        """Returns lane `j` as a width-1 monoid."""
        var r = Self.Single()
        r.acc[0] = self.acc[j]
        return r

    @always_inline
    def __setitem__(mut self, j: Int, s: Self.Single):
        """Writes width-1 monoid `s` into lane `j`."""
        self.acc[j] = s.acc[0]

    @always_inline
    def reduce(self) -> Self.Single:
        """Sums the `W` lane partials via the `reduce_add` intrinsic.

        Overrides the default lane-fold so the vectorized horizontal add
        is emitted explicitly rather than reconstructed from a scalar
        chain by the optimizer (a measured CPU win; GPU-neutral).

        Returns:
            A width-1 `ReduceSum` holding the total in `acc[0]`.
        """
        var r = Self.Single()
        r.acc[0] = self.acc.reduce_add()
        return r

    @always_inline
    def join_parallel[R: Reducer](mut self, reducer: R):
        """Cross-thread combine via `reducer.sum`, splatting the scalar
        result across all lanes so bodies read `acc.slice[w]` uniformly.
        Runs after `reduce`, so `acc[0]` already holds the within-thread
        sum.

        Parameters:
            R: The parallel scalar reducer.

        Args:
            reducer: The reducer instance.
        """
        # The dtypes with a hardware-fast `reducer.sum`. This is deliberately
        # narrower than `Self.dtype.is_arithmetic()`: the latter also admits
        # float64 and the sub-word integers (u/int8, u/int16), which have no
        # fast reduce here and must take the `generic` path below.
        comptime if Self.dtype in (
            DType.float16,
            DType.bfloat16,
            DType.float32,
            DType.int32,
            DType.uint32,
            DType.int64,
            DType.uint64,
        ):
            self.acc[0] = reducer.sum(self.acc[0])
        else:
            # `reduce` left all lanes equal, so the generic field-wise
            # combine yields the cross-thread sum on every lane.
            reducer.generic(self)


# ===-----------------------------------------------------------------------===#
# `ReduceMax[dtype]` — running maximum
# ===-----------------------------------------------------------------------===#


struct ReduceMax[
    dtype: DType,
    W: Int = simd_width_of[dtype](),
](ReduceOp):
    """Max reduction monoid: `(self, x) -> max(self, x)`.

    Single-field state: `acc: SIMD[dtype, W]`. `accumulate` is one
    expression — `acc = max(acc, Self.pad[...](val))`. The horizontal
    collapse to scalar happens once at `reduce`, in `acc[0]`.

    Parameters:
        dtype: The accumulator dtype.
        W: SIMD width of the lane-wise accumulator. Defaults to the
            target's `simd_width_of[dtype]`.
    """

    comptime Single = ReduceMax[Self.dtype, 1]
    comptime width = Self.W

    var acc: SIMD[Self.dtype, Self.W]
    """Lane-wise SIMD accumulator. `reduce` reduces to scalar (placed
    in `acc[0]`); bodies read `acc[0]`."""

    @always_inline
    def __init__(out self):
        """Identity: `acc = MIN_W` (`False` for bool: max is logical OR)."""
        comptime assert (
            Self.dtype.is_numeric() or Self.dtype == .bool
        ), "ReduceMax requires a numeric or bool dtype"
        self.acc = SIMD[Self.dtype, Self.W](min_finite[Self.dtype]())

    @always_inline
    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        """Folds a SIMD tile into `acc` lane-wise. Partial tiles get
        identity-padded (`min_finite`) via `Self.pad`.

        Parameters:
            val_dtype: The tile's source dtype.
            w: SIMD width of the tile.

        Args:
            val: The SIMD tile to fold.
            idx: Unused (Max is index-agnostic).
        """
        self.acc = max(
            self.acc,
            Self.pad[Self.W, Self.dtype, min_finite[Self.dtype]()](val),
        )

    @always_inline
    def join(mut self, other: Self):
        """Sequential combine: lane-wise max.

        Args:
            other: The state to combine into `self`.
        """
        self.acc = max(self.acc, other.acc)

    @always_inline
    def __getitem__(self, j: Int) -> Self.Single:
        """Returns lane `j` as a width-1 monoid."""
        var r = Self.Single()
        r.acc[0] = self.acc[j]
        return r

    @always_inline
    def __setitem__(mut self, j: Int, s: Self.Single):
        """Writes width-1 monoid `s` into lane `j`."""
        self.acc[j] = s.acc[0]

    @always_inline
    def reduce(self) -> Self.Single:
        """Maxes the `W` lane partials via the `reduce_max` intrinsic.

        Overrides the default lane-fold so the vectorized horizontal max
        is emitted explicitly (a measured CPU win; GPU-neutral). `bool`
        reduces through `uint8` because the FP `reduce_max` intrinsic
        rejects `<N x i1>` on GPU.

        Returns:
            A width-1 `ReduceMax` holding the maximum in `acc[0]`.
        """
        var r = Self.Single()
        comptime if Self.dtype == .bool:
            r.acc = SIMD[Self.dtype, 1](
                self.acc.cast[.uint8]().reduce_max().cast[.bool]()
            )
        else:
            r.acc[0] = self.acc.reduce_max()
        return r

    @always_inline
    def join_parallel[R: Reducer](mut self, reducer: R):
        """Cross-thread combine via `reducer.max`, splatting the scalar
        result across all lanes so bodies read `acc.slice[w]`
        uniformly. Runs after `reduce`.

        Parameters:
            R: The parallel scalar reducer.

        Args:
            reducer: The reducer instance.
        """
        comptime if Self.dtype in (
            DType.float16,
            DType.bfloat16,
            DType.float32,
            DType.int32,
            DType.uint32,
            DType.int64,
            DType.uint64,
        ):
            self.acc[0] = reducer.max(self.acc[0])
        else:
            reducer.generic(self)


# ===-----------------------------------------------------------------------===#
# `ReduceMin[dtype]` — running minimum
# ===-----------------------------------------------------------------------===#


struct ReduceMin[
    dtype: DType,
    W: Int = simd_width_of[dtype](),
](ReduceOp):
    """Min reduction monoid: `(self, x) -> min(self, x)`.

    Single-field state: `acc: SIMD[dtype, W]`. `accumulate` is one
    expression — `acc = min(acc, Self.pad[...](val))`. The horizontal
    collapse to scalar happens once at `reduce`, in `acc[0]`.

    Parameters:
        dtype: The accumulator dtype.
        W: SIMD width of the lane-wise accumulator. Defaults to the
            target's `simd_width_of[dtype]`.
    """

    comptime Single = ReduceMin[Self.dtype, 1]
    comptime width = Self.W

    var acc: SIMD[Self.dtype, Self.W]
    """Lane-wise SIMD accumulator. `reduce` reduces to scalar (placed
    in `acc[0]`); bodies read `acc[0]`."""

    @always_inline
    def __init__(out self):
        """Identity: `acc = MAX_W` (`True` for bool: min is logical AND)."""
        comptime assert (
            Self.dtype.is_numeric() or Self.dtype == .bool
        ), "ReduceMin requires a numeric or bool dtype"
        self.acc = SIMD[Self.dtype, Self.W](max_finite[Self.dtype]())

    @always_inline
    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        """Folds a SIMD tile into `acc` lane-wise. Partial tiles get
        identity-padded (`max_finite`) via `Self.pad`.

        Parameters:
            val_dtype: The tile's source dtype.
            w: SIMD width of the tile.

        Args:
            val: The SIMD tile to fold.
            idx: Unused (Min is index-agnostic).
        """
        self.acc = min(
            self.acc,
            Self.pad[Self.W, Self.dtype, max_finite[Self.dtype]()](val),
        )

    @always_inline
    def join(mut self, other: Self):
        """Sequential combine: lane-wise min.

        Args:
            other: The state to combine into `self`.
        """
        self.acc = min(self.acc, other.acc)

    @always_inline
    def __getitem__(self, j: Int) -> Self.Single:
        """Returns lane `j` as a width-1 monoid."""
        var r = Self.Single()
        r.acc[0] = self.acc[j]
        return r

    @always_inline
    def __setitem__(mut self, j: Int, s: Self.Single):
        """Writes width-1 monoid `s` into lane `j`."""
        self.acc[j] = s.acc[0]

    @always_inline
    def reduce(self) -> Self.Single:
        """Mins the `W` lane partials via the `reduce_min` intrinsic.

        Overrides the default lane-fold so the vectorized horizontal min
        is emitted explicitly (a measured CPU win; GPU-neutral). `bool`
        reduces through `uint8` because the FP `reduce_min` intrinsic
        rejects `<N x i1>` on GPU.

        Returns:
            A width-1 `ReduceMin` holding the minimum in `acc[0]`.
        """
        var r = Self.Single()
        comptime if Self.dtype == .bool:
            r.acc = SIMD[Self.dtype, 1](
                self.acc.cast[.uint8]().reduce_min().cast[.bool]()
            )
        else:
            r.acc[0] = self.acc.reduce_min()
        return r

    @always_inline
    def join_parallel[R: Reducer](mut self, reducer: R):
        """Cross-thread combine via `reducer.min`, splatting the scalar
        result across all lanes. Runs after `reduce`.

        Parameters:
            R: The parallel scalar reducer.

        Args:
            reducer: The reducer instance.
        """
        comptime if Self.dtype in (
            DType.float16,
            DType.bfloat16,
            DType.float32,
            DType.int32,
            DType.uint32,
            DType.int64,
            DType.uint64,
        ):
            self.acc[0] = reducer.min(self.acc[0])
        else:
            reducer.generic(self)


# ===-----------------------------------------------------------------------===#
# `ReduceProduct[dtype]` — running product
# ===-----------------------------------------------------------------------===#


struct ReduceProduct[
    dtype: DType,
    W: Int = simd_width_of[dtype](),
](ReduceOp):
    """Product reduction monoid: `(self, x) -> self * x`.

    Single-field state: `acc: SIMD[dtype, W]`. `accumulate` is one
    expression — `acc *= Self.pad[...](val)`. The horizontal collapse to
    scalar happens once at `reduce`, in `acc[0]`. No hardware-fast
    `Reducer.product`, so `join_parallel` falls through to the generic
    combiner after the SIMD collapse.

    Parameters:
        dtype: The accumulator dtype.
        W: SIMD width of the lane-wise accumulator. Defaults to the
            target's `simd_width_of[dtype]`.
    """

    comptime Single = ReduceProduct[Self.dtype, 1]
    comptime width = Self.W

    var acc: SIMD[Self.dtype, Self.W]
    """Lane-wise SIMD accumulator. `reduce` reduces to scalar (placed
    in `acc[0]`); bodies read `acc[0]`."""

    @always_inline
    def __init__(out self):
        """Identity: `acc = 1_W`."""
        comptime assert (
            Self.dtype.is_numeric()
        ), "ReduceProduct requires a numeric dtype"
        self.acc = SIMD[Self.dtype, Self.W](1)

    @always_inline
    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        """Folds a SIMD tile into `acc` lane-wise. Partial tiles get
        identity-padded (`1`) via `Self.pad`.

        Parameters:
            val_dtype: The tile's source dtype.
            w: SIMD width of the tile.

        Args:
            val: The SIMD tile to fold.
            idx: Unused (Product is index-agnostic).
        """
        self.acc *= Self.pad[
            Self.W,
            Self.dtype,
            Scalar[Self.dtype](1),
        ](val)

    @always_inline
    def join(mut self, other: Self):
        """Sequential combine: lane-wise product.

        Args:
            other: The state to combine into `self`.
        """
        self.acc *= other.acc

    @always_inline
    def __getitem__(self, j: Int) -> Self.Single:
        """Returns lane `j` as a width-1 monoid."""
        var r = Self.Single()
        r.acc[0] = self.acc[j]
        return r

    @always_inline
    def __setitem__(mut self, j: Int, s: Self.Single):
        """Writes width-1 monoid `s` into lane `j`."""
        self.acc[j] = s.acc[0]

    @always_inline
    def reduce(self) -> Self.Single:
        """Multiplies the `W` lane partials via the `reduce_mul`
        intrinsic.

        Overrides the default lane-fold so the vectorized horizontal
        product is emitted explicitly rather than reconstructed from a
        scalar chain by the optimizer (a measured CPU win; GPU-neutral).

        Returns:
            A width-1 `ReduceProduct` holding the product in `acc[0]`.
        """
        var r = Self.Single()
        r.acc[0] = self.acc.reduce_mul()
        return r

    @always_inline
    def join_parallel[R: Reducer](mut self, reducer: R):
        """Cross-thread combine. No hardware-fast scalar `product`, so
        this goes through `reducer.generic`. Runs after `reduce`, so
        `acc[0]` already holds the within-thread product; reduce a
        width-1 `ReduceProduct` (1 uint32 word) so the combine takes the
        register shuffle-butterfly (`<= _WARP_SHUFFLE_MAX_WORDS`) rather
        than the shmem tree a wider state falls into, then splat the
        result back.

        Parameters:
            R: The parallel scalar reducer.

        Args:
            reducer: The reducer instance.
        """
        var s = ReduceProduct[Self.dtype, 1]()
        s.acc[0] = self.acc[0]
        reducer.generic(s)
        self.acc = SIMD[Self.dtype, Self.W](s.acc[0])


# ===-----------------------------------------------------------------------===#
# `ArgMax[dtype]` — running argmax (value + index)
# ===-----------------------------------------------------------------------===#


struct MinMax[
    dtype: DType,
    W: Int = simd_width_of[dtype](),
](ReduceOp):
    """Fused min+max reduction monoid: tracks both `min(self, x)` and
    `max(self, x)` in one state. Cuts the axis walk in half vs running
    separate `ReduceMin` + `ReduceMax`.

    Two SIMD fields — `min_acc` (running minimum) and `max_acc` (running
    maximum). `accumulate` is one expression per field; `reduce`
    reduces each into its lane 0.

    Parameters:
        dtype: The accumulator dtype.
        W: SIMD width of the lane-wise accumulators. Defaults to the
            target's `simd_width_of[dtype]`.
    """

    comptime Single = MinMax[Self.dtype, 1]
    comptime width = Self.W

    var min_acc: SIMD[Self.dtype, Self.W]
    """Lane-wise running minimum. Final scalar in `min_acc[0]`."""

    var max_acc: SIMD[Self.dtype, Self.W]
    """Lane-wise running maximum. Final scalar in `max_acc[0]`."""

    @always_inline
    def __init__(out self):
        """Identity: `min_acc = +inf_W`, `max_acc = -inf_W`."""
        comptime assert (
            Self.dtype.is_numeric()
        ), "MinMax requires a numeric dtype"
        self.min_acc = SIMD[Self.dtype, Self.W](max_finite[Self.dtype]())
        self.max_acc = SIMD[Self.dtype, Self.W](min_finite[Self.dtype]())

    @always_inline
    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        """Folds a SIMD tile into both accumulators. Each field uses
        its own identity padding via `Self.pad`: `+inf`/MAX for the
        min field, `-inf`/MIN for the max field, so padded lanes never
        affect the result.

        Parameters:
            val_dtype: The tile's source dtype.
            w: SIMD width of the tile.

        Args:
            val: The SIMD tile to fold.
            idx: Unused (min/max are index-agnostic).
        """
        var padded_min = Self.pad[Self.W, Self.dtype, max_finite[Self.dtype]()](
            val
        )
        var padded_max = Self.pad[Self.W, Self.dtype, min_finite[Self.dtype]()](
            val
        )
        self.min_acc = min(self.min_acc, padded_min)
        self.max_acc = max(self.max_acc, padded_max)

    @always_inline
    def join(mut self, other: Self):
        """Sequential combine: lane-wise min and max.

        Args:
            other: The state to combine into `self`.
        """
        self.min_acc = min(self.min_acc, other.min_acc)
        self.max_acc = max(self.max_acc, other.max_acc)

    @always_inline
    def __getitem__(self, j: Int) -> Self.Single:
        """Returns lane `j` as a width-1 monoid."""
        var r = Self.Single()
        r.min_acc[0] = self.min_acc[j]
        r.max_acc[0] = self.max_acc[j]
        return r

    @always_inline
    def __setitem__(mut self, j: Int, s: Self.Single):
        """Writes width-1 monoid `s` into lane `j`."""
        self.min_acc[j] = s.min_acc[0]
        self.max_acc[j] = s.max_acc[0]

    @always_inline
    def reduce(self) -> Self.Single:
        """Collapses each field's `W` lane partials via the `reduce_min`
        / `reduce_max` intrinsics.

        Overrides the default lane-fold so the vectorized horizontal
        min/max are emitted explicitly (a measured CPU win; GPU-neutral).

        Returns:
            A width-1 `MinMax` with the min in `min_acc[0]` and the max
            in `max_acc[0]`.
        """
        var r = Self.Single()
        r.min_acc[0] = self.min_acc.reduce_min()
        r.max_acc[0] = self.max_acc.reduce_max()
        return r

    @always_inline
    def join_parallel[R: Reducer](mut self, reducer: R):
        """Cross-thread combine via `reducer.min` / `reducer.max`,
        splatting each scalar result across its field's lanes. Runs
        after `reduce`.

        Parameters:
            R: The parallel scalar reducer.

        Args:
            reducer: The reducer instance.
        """
        comptime if Self.dtype in (
            DType.float16,
            DType.bfloat16,
            DType.float32,
            DType.int32,
            DType.uint32,
            DType.int64,
            DType.uint64,
        ):
            self.min_acc[0] = reducer.min(self.min_acc[0])
            self.max_acc[0] = reducer.max(self.max_acc[0])
        else:
            reducer.generic(self)


@always_inline
def _in_range_index(idx: Int64) -> Int64:
    """Folds the `Int64.MAX` identity index to an in-range value.

    `ArgMax`/`ArgMin` use `Int64.MAX` as the identity so a padded or empty
    lane loses every tie-break, but the identity survives whenever no
    candidate ever wins: an empty row, a row of pure identity values, or an
    all-NaN row (every ordered compare against NaN is false). Emitting it
    hands a downstream consumer a `2**63-1` index -- a `gather` on that index
    is an out-of-bounds device access, which turns a numerical problem into a
    dead worker. Report index 0 instead, which is the contract
    `nn/argmaxmin_gpu` already documents and tests for an all-NaN row, so the
    identity never escapes as a tensor value.
    """
    return 0 if idx == Int64.MAX else idx


@always_inline
def _argmax_identity[dtype: DType]() -> Scalar[dtype]:
    """Returns the ArgMax identity — `-inf` for floating dtypes, `MIN`
    for integer dtypes. Never wins against any finite value under the
    `>` rule."""
    comptime if dtype.is_floating_point():
        return min_finite[dtype]()
    else:
        return Scalar[dtype].MIN


@always_inline
def _argmin_identity[dtype: DType]() -> Scalar[dtype]:
    """Returns the ArgMin identity — `+inf` for floating dtypes, `MAX`
    for integer dtypes."""
    comptime if dtype.is_floating_point():
        return max_finite[dtype]()
    else:
        return Scalar[dtype].MAX


struct ArgMax[
    dtype: DType,
    W: Int = simd_width_of[dtype](),
](ReduceOp):
    """Argmax reduction monoid: tracks the (value, axis-index) pair with
    the largest value. Ties break to the lower index.

    Per-tile work is two `select`s into SIMD-wide
    `acc_values`/`acc_indices` (no per-tile horizontal reduce); the
    horizontal reduce + lane-find happens once at `reduce`. Partial
    tiles get identity-padded via `Self.pad` — `-inf` (or `MIN`) for
    values, `Int64.MAX` for indices — so padded lanes never win the
    tie-break.

    `W` defaults to `1` on GPU because ArgMax storage scales as
    `(value_size + 8) * W` per thread (int64 indices), and GPU's
    BLOCK_SIZE multiplier turns large `W` into severe register pressure.
    CPU defaults to `simd_width_of[dtype]` for full bulk-path SIMD
    parallelism.

    Parameters:
        dtype: The value dtype.
        W: SIMD width of the per-lane accumulator. Defaults to `1`
            on GPU (low register pressure) and `simd_width_of[dtype]`
            on CPU (full SIMD parallelism).
    """

    comptime Single = ArgMax[Self.dtype, 1]
    comptime width = Self.W

    var best: Scalar[Self.dtype]
    """Final scalar best value. Body reads this after `join_parallel`."""

    var best_idx: Int
    """Final scalar best index. Body reads this after `join_parallel`."""

    var acc_values: SIMD[Self.dtype, Self.W]
    """Per-lane running max during SIMD-wide tile accumulation."""

    var acc_indices: SIMD[.int64, Self.W]
    """Per-lane axis indices corresponding to `acc_values`."""

    @always_inline
    def __init__(out self):
        """Identity: scalars at `-inf`/`MIN` and `Int.MAX`; SIMD accs at
        the same identities. First real `(value, idx)` always wins under
        the lower-idx tie-break."""
        comptime assert (
            Self.dtype.is_numeric()
        ), "ArgMax requires a numeric dtype"
        self.best = _argmax_identity[Self.dtype]()
        self.best_idx = Int.MAX
        self.acc_values = SIMD[Self.dtype, Self.W](
            _argmax_identity[Self.dtype]()
        )
        self.acc_indices = SIMD[.int64, Self.W](Int64.MAX)

    @always_inline
    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        """Folds a SIMD tile via lane-wise compare-and-select. `val` and
        `idx` are padded to width `Self.W` via `Self.pad` — padded value
        lanes `-inf`/`MIN`, padded index lanes `Int64.MAX` — so they
        always lose to any real candidate.

        `le` (not `lt`) preserves the lower-idx tie-break: a tied value
        with a higher idx keeps the current (lower-idx) acc. `idx` is the
        scaffolder-built per-lane axis-position vector — the monoid never
        adds an `iota`.

        Parameters:
            val_dtype: The tile's source dtype.
            w: SIMD width of the tile.

        Args:
            val: The SIMD tile to fold.
            idx: Per-lane axis positions of `val`'s lanes.
        """
        var val_padded = Self.pad[
            Self.W,
            Self.dtype,
            _argmax_identity[Self.dtype](),
        ](val)
        var idx_padded = Self.pad[Self.W, .int64, Int64.MAX](idx)
        # Strict `gt` (not `!le`): the two agree for every ordered pair, but
        # for a NaN candidate `gt` is false, so the accumulator is kept and
        # the NaN is skipped. `le` took it, poisoning the accumulator and
        # making every later compare false. Skipping matches the contract
        # `nn/argmaxmin_gpu` already documents and tests (`TopK_2.insert`).
        var take = val_padded.gt(self.acc_values)
        self.acc_values = take.select(val_padded, self.acc_values)
        self.acc_indices = take.select(idx_padded, self.acc_indices)

    @always_inline
    def join(mut self, other: Self):
        """Sequential combine. Element-wise SIMD merge of the acc;
        scalar merge of the `(best, best_idx)` tail.

        Tie-symmetric: when values tie, the smaller index wins regardless
        of which side it came from — required for correctness under
        cross-thread tree reduction, where the pair-ordering isn't
        index-ordered.

        Args:
            other: The state to combine into `self`.
        """
        # `keep` = self wins: strictly greater value, or a tie with self
        # holding the lower index.
        var self_greater = self.acc_values.gt(other.acc_values)
        var tie = self.acc_values.eq(other.acc_values)
        var tie_self_lower_idx = tie & self.acc_indices.lt(other.acc_indices)
        var keep = self_greater | tie_self_lower_idx
        self.acc_values = keep.select(self.acc_values, other.acc_values)
        self.acc_indices = keep.select(self.acc_indices, other.acc_indices)
        if other.best > self.best or (
            other.best == self.best and other.best_idx < self.best_idx
        ):
            self.best = other.best
            self.best_idx = other.best_idx

    @always_inline
    def __getitem__(self, j: Int) -> Self.Single:
        """Returns lane `j` as a width-1 monoid."""
        var r = Self.Single()
        r.best = self.best
        r.best_idx = self.best_idx
        r.acc_values[0] = self.acc_values[j]
        r.acc_indices[0] = self.acc_indices[j]
        return r

    @always_inline
    def __setitem__(mut self, j: Int, s: Self.Single):
        """Writes width-1 monoid `s` into lane `j`."""
        self.acc_values[j] = s.acc_values[0]
        self.acc_indices[j] = s.acc_indices[0]
        self.best = s.best
        self.best_idx = s.best_idx

    @always_inline
    def reduce(self) -> Self.Single:
        """SIMD-tree collapse (faster than the default lane-fold for a
        value+index select): max value in lane 0, lowest index among the
        max lanes. `join_parallel` folds lane 0 into `(best, best_idx)`."""
        var acc_max = self.acc_values.reduce_max()
        var max_mask = self.acc_values.eq(SIMD[Self.dtype, Self.W](acc_max))
        var masked_idx = max_mask.select(
            self.acc_indices, SIMD[.int64, Self.W](Int64.MAX)
        )
        var r = Self.Single()
        r.best = self.best
        r.best_idx = self.best_idx
        r.acc_values[0] = acc_max
        r.acc_indices[0] = _in_range_index(masked_idx.reduce_min())
        return r

    @always_inline
    def join_parallel[R: Reducer](mut self, reducer: R):
        """Folds the within-thread acc at lane 0 into the scalar
        `(best, best_idx)`, resets the SIMD acc so the cross-thread
        `join` can't double-count it, combines `(best, best_idx)` across
        participants via `reducer.generic`, and leaves the winning index
        in `acc_indices[0]` so the body's emit reads
        `acc_indices.slice[tile_width]` uniformly (cooperative: lane 0;
        tiled: per-column, where `join_parallel` is skipped).

        At `W == 1` lane 0 is the running argmax directly; at `W > 1`
        `reduce` reduced it there first.

        Parameters:
            R: The parallel reducer.

        Args:
            reducer: The reducer instance.
        """
        var acc_max = self.acc_values[0]
        var acc_idx = Int(self.acc_indices[0])
        if acc_max > self.best or (
            acc_max == self.best and acc_idx < self.best_idx
        ):
            self.best = acc_max
            self.best_idx = acc_idx
        self.acc_values = SIMD[Self.dtype, Self.W](
            _argmax_identity[Self.dtype]()
        )
        self.acc_indices = SIMD[.int64, Self.W](Int64.MAX)
        reducer.generic(self)
        self.acc_indices[0] = _in_range_index(Int64(self.best_idx))


# ===-----------------------------------------------------------------------===#
# `ArgMin[dtype]` — running argmin (value + index)
# ===-----------------------------------------------------------------------===#


struct ArgMin[
    dtype: DType,
    W: Int = simd_width_of[dtype](),
](ReduceOp):
    """Argmin reduction monoid: mirrors `ArgMax` with `<`/`ge`
    comparison (see `ArgMax` for the algorithm + design notes).

    Parameters:
        dtype: The value dtype.
        W: SIMD width of the per-lane accumulator. Same GPU/CPU
            default rationale as `ArgMax`.
    """

    comptime Single = ArgMin[Self.dtype, 1]
    comptime width = Self.W

    var best: Scalar[Self.dtype]
    """Final scalar best value."""

    var best_idx: Int
    """Final scalar best index."""

    var acc_values: SIMD[Self.dtype, Self.W]
    """Per-lane running min during SIMD-wide tile accumulation."""

    var acc_indices: SIMD[.int64, Self.W]
    """Per-lane axis indices corresponding to `acc_values`."""

    @always_inline
    def __init__(out self):
        """Identity: `best = +inf`/`MAX`, `best_idx = Int.MAX`, SIMD
        accs at the same identity."""
        comptime assert (
            Self.dtype.is_numeric()
        ), "ArgMin requires a numeric dtype"
        self.best = _argmin_identity[Self.dtype]()
        self.best_idx = Int.MAX
        self.acc_values = SIMD[Self.dtype, Self.W](
            _argmin_identity[Self.dtype]()
        )
        self.acc_indices = SIMD[.int64, Self.W](Int64.MAX)

    @always_inline
    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        """Folds a SIMD tile via lane-wise compare-and-select. Mirrors
        `ArgMax.accumulate` with the `>=` rule: padded value lanes
        `+inf`/`MAX`, padded index lanes `Int64.MAX`, so they always
        lose to any real candidate. `ge` (not `gt`) preserves the
        lower-idx tie-break. `idx` is the scaffolder-built per-lane
        axis-position vector — the monoid never adds an `iota`.

        Parameters:
            val_dtype: The tile's source dtype.
            w: SIMD width of the tile.

        Args:
            val: The SIMD tile to fold.
            idx: Per-lane axis positions of `val`'s lanes.
        """
        var val_padded = Self.pad[
            Self.W,
            Self.dtype,
            _argmin_identity[Self.dtype](),
        ](val)
        var idx_padded = Self.pad[Self.W, .int64, Int64.MAX](idx)
        # Strict `lt` (not `!ge`) so a NaN candidate is skipped rather than
        # taken; see the note in `ArgMax.accumulate`.
        var take = val_padded.lt(self.acc_values)
        self.acc_values = take.select(val_padded, self.acc_values)
        self.acc_indices = take.select(idx_padded, self.acc_indices)

    @always_inline
    def join(mut self, other: Self):
        """Sequential combine. Element-wise SIMD merge of the acc;
        scalar merge of the `(best, best_idx)` tail. Tie-symmetric:
        when values tie, the smaller index wins. Mirrors `ArgMax.join`
        with `<` instead of `>`.

        Args:
            other: The state to combine into `self`.
        """
        # Tie-symmetric: `keep` = self wins (strictly smaller value,
        # or tie with self holding the lower index).
        var self_smaller = self.acc_values.lt(other.acc_values)
        var tie = self.acc_values.eq(other.acc_values)
        var tie_self_lower_idx = tie & self.acc_indices.lt(other.acc_indices)
        var keep = self_smaller | tie_self_lower_idx
        self.acc_values = keep.select(self.acc_values, other.acc_values)
        self.acc_indices = keep.select(self.acc_indices, other.acc_indices)
        if other.best < self.best or (
            other.best == self.best and other.best_idx < self.best_idx
        ):
            self.best = other.best
            self.best_idx = other.best_idx

    @always_inline
    def __getitem__(self, j: Int) -> Self.Single:
        """Returns lane `j` as a width-1 monoid."""
        var r = Self.Single()
        r.best = self.best
        r.best_idx = self.best_idx
        r.acc_values[0] = self.acc_values[j]
        r.acc_indices[0] = self.acc_indices[j]
        return r

    @always_inline
    def __setitem__(mut self, j: Int, s: Self.Single):
        """Writes width-1 monoid `s` into lane `j`."""
        self.acc_values[j] = s.acc_values[0]
        self.acc_indices[j] = s.acc_indices[0]
        self.best = s.best
        self.best_idx = s.best_idx

    @always_inline
    def reduce(self) -> Self.Single:
        """SIMD-tree collapse (faster than the default lane-fold for a
        value+index select): min value in lane 0, lowest index among the
        min lanes. `join_parallel` folds lane 0 into `(best, best_idx)`."""
        var acc_min = self.acc_values.reduce_min()
        var min_mask = self.acc_values.eq(SIMD[Self.dtype, Self.W](acc_min))
        var masked_idx = min_mask.select(
            self.acc_indices, SIMD[.int64, Self.W](Int64.MAX)
        )
        var r = Self.Single()
        r.best = self.best
        r.best_idx = self.best_idx
        r.acc_values[0] = acc_min
        r.acc_indices[0] = _in_range_index(masked_idx.reduce_min())
        return r

    @always_inline
    def join_parallel[R: Reducer](mut self, reducer: R):
        """Folds the within-thread acc at lane 0 into the scalar
        `(best, best_idx)`, resets the SIMD acc, combines across
        participants via `reducer.generic`, and leaves the winning index
        in `acc_indices[0]`. Mirrors `ArgMax.join_parallel` with the
        `<` tie-break. Works at any `W` (collapse pre-reduces to lane 0
        when `W > 1`).

        Parameters:
            R: The parallel reducer.

        Args:
            reducer: The reducer instance.
        """
        var acc_min = self.acc_values[0]
        var acc_idx = Int(self.acc_indices[0])
        if acc_min < self.best or (
            acc_min == self.best and acc_idx < self.best_idx
        ):
            self.best = acc_min
            self.best_idx = acc_idx
        self.acc_values = SIMD[Self.dtype, Self.W](
            _argmin_identity[Self.dtype]()
        )
        self.acc_indices = SIMD[.int64, Self.W](Int64.MAX)
        reducer.generic(self)
        self.acc_indices[0] = _in_range_index(Int64(self.best_idx))
