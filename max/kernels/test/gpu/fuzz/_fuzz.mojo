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
#
# Reusable kernel-fuzzing harness (see gpu-kernels-fuzzing-design.md, Slice 1).
#
# This is the generation + value-fill + result-reporting core shared by every
# per-kernel fuzz target. It is intentionally host-side only (fills run on host
# buffers, then the target copies to device), so it has no GPU dependency and
# can be unit-tested on CPU.
#
# What lives here (reusable across kernels):
#   - boundary_int: a boundary-aware integer generator (the core of shape fuzz).
#   - value-distribution fills incl. NaN/Inf/denormal/+-0 injection that
#     layout._fillers lacks today (the design's main new numerical content).
#   - Verdict constants + structured print helpers the Python orchestrator parses.
#
# What stays in each target (kernel-specific): the per-kernel CaseSpec and the
# run_one_case that allocates/launches/compares.

from std.math import cos, isfinite, log, min, pi, sin, sqrt
from std.random import random_float64, random_ui64
from std.sys import argv
from std.utils.numerics import inf, max_finite, min_finite, nan, neg_inf


# ===----------------------------------------------------------------------=== #
# Boundary-aware shape generation
# ===----------------------------------------------------------------------=== #


def boundary_int(lo: Int, hi: Int, tile: Int) -> Int:
    """Draw an int in [lo, hi], biased toward boundary classes around `tile`.

    Boundary-aware generation is the core of shape fuzzing: a bug at
    `n % tile == 1`, at a size-0/1 edge, or at a tile multiple is invisible to
    fixed-shape tests but is hit deliberately here. Roughly 3/4 of draws land
    on a boundary class (lo, lo+1, k*tile +- 1, hi, hi-1); the rest are uniform.

    Args:
        lo: Inclusive lower bound.
        hi: Inclusive upper bound.
        tile: The interesting modulus (e.g. an attention BN or a matmul block).

    Returns:
        An int in [lo, hi].
    """
    if hi <= lo:
        return lo
    var roll = Int(random_ui64(0, 15))
    if roll == 0:
        return lo
    if roll == 1:
        return lo + 1
    if roll == 2:
        return max(lo, min(hi, tile - 1))
    if roll == 3:
        return max(lo, min(hi, tile))
    if roll == 4:
        return max(lo, min(hi, tile + 1))
    if roll == 5:
        return max(lo, min(hi, 2 * tile - 1))
    if roll == 6:
        return max(lo, min(hi, 2 * tile))
    if roll == 7:
        return max(lo, min(hi, 3 * tile))
    if roll == 8:
        return max(lo, min(hi, 4 * tile - 1))
    if roll == 9:
        return max(lo, min(hi, 4 * tile))
    if roll == 10:
        return hi
    if roll == 11:
        return max(lo, hi - 1)
    # Uniform fallback.
    return Int(random_ui64(UInt64(lo), UInt64(hi)))


# ===----------------------------------------------------------------------=== #
# Value distributions
# ===----------------------------------------------------------------------=== #

comptime VD_UNIFORM = 0
comptime VD_NORMAL = 1
comptime VD_SPARSE = 2
comptime VD_LARGE = 3
comptime VD_ALL_EQUAL = 4
comptime VD_SPECIALS = 5
comptime NUM_VALUE_DISTS = 6


def value_dist_name(d: Int) -> String:
    """Returns the human-readable name of a value-distribution id."""
    if d == VD_NORMAL:
        return "normal"
    if d == VD_SPARSE:
        return "sparse"
    if d == VD_LARGE:
        return "large"
    if d == VD_ALL_EQUAL:
        return "all_equal"
    if d == VD_SPECIALS:
        return "specials"
    return "uniform"


def fill_uniform[
    dtype: DType
](
    span: Span[mut=True, Scalar[dtype], _],
    lo: Float64 = -1.0,
    hi: Float64 = 1.0,
):
    """Fills `span` with i.i.d. uniform values in [lo, hi)."""
    for i in range(len(span)):
        span[i] = random_float64(lo, hi).cast[dtype]()


def fill_normal[
    dtype: DType
](
    span: Span[mut=True, Scalar[dtype], _],
    mean: Float64 = 0.0,
    std: Float64 = 1.0,
):
    """Fills `span` with Gaussian(mean, std) values via Box-Muller."""
    var i = 0
    var n = len(span)
    while i < n:
        var u1 = max(1e-12, random_float64(0.0, 1.0))
        var u2 = random_float64(0.0, 1.0)
        var r = sqrt(-2.0 * log(u1))
        span[i] = (mean + std * r * cos(2.0 * pi * u2)).cast[dtype]()
        i += 1
        if i < n:
            span[i] = (mean + std * r * sin(2.0 * pi * u2)).cast[dtype]()
            i += 1


def fill_sparse[
    dtype: DType
](
    span: Span[mut=True, Scalar[dtype], _],
    density: Float64 = 0.1,
    lo: Float64 = -1.0,
    hi: Float64 = 1.0,
):
    """Fills `span` mostly with zeros; each element is nonzero w.p. `density`.

    Sparse inputs expose accumulation/empty-reduction bugs that dense uniform
    inputs hide (the "insensitive to uniform random" failure mode).
    """
    for i in range(len(span)):
        if random_float64(0.0, 1.0) < density:
            span[i] = random_float64(lo, hi).cast[dtype]()
        else:
            span[i] = Scalar[dtype](0)


def fill_large[dtype: DType](span: Span[mut=True, Scalar[dtype], _]):
    """Fills `span` with large-magnitude finite values (random sign).

    Provokes overflow / saturation in reductions and accumulators.
    """
    var mx = max_finite[dtype]().cast[.float64]()
    for i in range(len(span)):
        var v = random_float64(0.5, 1.0) * mx
        if random_ui64(0, 1) == 1:
            v = -v
        span[i] = v.cast[dtype]()


def fill_all_equal[
    dtype: DType
](span: Span[mut=True, Scalar[dtype], _], value: Float64 = 1.0):
    """Fills `span` with a single repeated value (degenerate reductions)."""
    var v = value.cast[dtype]()
    for i in range(len(span)):
        span[i] = v


def fill_with_specials[
    dtype: DType
](span: Span[mut=True, Scalar[dtype], _], density: Float64 = 0.25):
    """Fills `span` uniformly, injecting NaN/+-Inf/+-0/+-max at rate `density`.

    This is the special-value coverage the codebase lacks at the kernel level:
    it drives the finite-output / NaN-propagation contract checks.
    """
    var specials: List[Scalar[dtype]] = [
        nan[dtype](),
        inf[dtype](),
        neg_inf[dtype](),
        Scalar[dtype](0),
        -Scalar[dtype](0),
        max_finite[dtype](),
        min_finite[dtype](),
    ]
    for i in range(len(span)):
        if random_float64(0.0, 1.0) < density:
            span[i] = specials[Int(random_ui64(0, UInt64(len(specials) - 1)))]
        else:
            span[i] = random_float64(-1.0, 1.0).cast[dtype]()


def fill_by_dist[
    dtype: DType
](span: Span[mut=True, Scalar[dtype], _], dist: Int):
    """Dispatches to the fill for value-distribution id `dist`."""
    if dist == VD_NORMAL:
        fill_normal(span)
    elif dist == VD_SPARSE:
        fill_sparse(span)
    elif dist == VD_LARGE:
        fill_large(span)
    elif dist == VD_ALL_EQUAL:
        fill_all_equal(span)
    elif dist == VD_SPECIALS:
        fill_with_specials(span)
    else:
        fill_uniform(span)


# ===----------------------------------------------------------------------=== #
# Result reporting (machine-readable, parsed by the Python orchestrator)
# ===----------------------------------------------------------------------=== #

comptime VERDICT_PASS = "PASS"
comptime VERDICT_FAIL = "FAIL"
comptime VERDICT_INTERESTING = "INTERESTING"
comptime VERDICT_ERROR = "ERROR"


# ===----------------------------------------------------------------------=== #
# Numerical-correctness oracle helper
# ===----------------------------------------------------------------------=== #


def numeric_check[
    dtype: DType
](
    actual: Span[Scalar[dtype], _],
    expected: Span[Scalar[dtype], _],
    *,
    atol: Float64 = 1e-3,
    rtol: Float64 = 1e-2,
    allow_nonfinite: Bool = False,
) -> Bool:
    """Element-wise compares `actual` vs a (higher-precision) `expected`.

    Passing criterion mirrors the differential tests: `|a - e| <= atol +
    rtol*|e|`, plus a finite-contract check (a finite reference must not produce
    a non-finite kernel output). On failure prints a machine-readable
    `FUZZ_NUMERIC_FAIL ...` line (parsed by the orchestrator's `ref` oracle) and
    returns False; otherwise returns True.

    The default tolerances are deliberately generous (so legitimate
    reduction-order FP differences in a correct kernel do not cry wolf); a real
    wrong-answer is grossly out of band. Tightening per-dtype/per-reduction-depth
    is future work.

    Set `allow_nonfinite` when the kernel accumulates in a narrower type than
    the reference and the case's magnitudes put intermediate overflow out of
    reach of any ordering: saturating to +-Inf is IEEE-correct there, so the
    finite contract would be reporting the format, not a bug. It waives only
    the elements whose reference is finite and output is not; every other
    element still answers to `atol`/`rtol`.
    """
    var n = min(len(actual), len(expected))
    var max_abs = Float64(0)
    var max_rel = Float64(0)
    var n_bad = 0
    var worst = -1
    for i in range(n):
        var a = actual[i].cast[.float64]()
        var e = expected[i].cast[.float64]()
        # Finite contract: finite reference must yield a finite output.
        if isfinite(e) and not isfinite(a):
            if allow_nonfinite:
                continue
            n_bad += 1
            if worst < 0:
                worst = i
            continue
        var ad = abs(a - e)
        if ad > atol + rtol * abs(e):
            n_bad += 1
            var rel = ad / (abs(e) + 1e-30)
            if ad > max_abs:
                max_abs = ad
                worst = i
            if rel > max_rel:
                max_rel = rel
    if n_bad > 0:
        print(
            "FUZZ_NUMERIC_FAIL n_bad=",
            n_bad,
            "max_abs=",
            max_abs,
            "max_rel=",
            max_rel,
            "worst_idx=",
            worst,
        )
        return False
    return True


# ===----------------------------------------------------------------------=== #
# argv handling (shared by all targets; orchestrator drives `--<key> <value>`)
# ===----------------------------------------------------------------------=== #


def collect_args() -> List[String]:
    """Collects process argv into a List[String] for flag lookup."""
    var out = List[String]()
    for a in argv():
        out.append(String(a))
    return out^


def flag(args: List[String], name: String, default: String) -> String:
    """Returns the value following `name` in `args`, or `default`."""
    for i in range(len(args)):
        if args[i] == name and i + 1 < len(args):
            return args[i + 1]
    return default


def flag_int(args: List[String], name: String, default: Int) raises -> Int:
    """Returns the int value following `name` in `args`, or `default`."""
    var s = flag(args, name, "")
    if s.byte_length() == 0:
        return default
    return Int(s)
