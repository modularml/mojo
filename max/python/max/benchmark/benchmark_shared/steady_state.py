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

"""Steady-state auto-detection for benchmark runs.

Uses rolling MAD/median (median absolute deviation / median) on per-request
TTFT and TPOT to find the steady-state region:

- TTFT MAD/median detects the ramp-up boundary.
- TPOT MAD/median detects the ramp-down boundary.

When TPOT is absent across the run (prefill-only workloads), ramp-down
falls back to TTFT. Runs that never stabilize produce a descriptive
warning instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .request import RequestFuncOutput


def reject_metric_outliers(
    values: Sequence[float], k: float = 3.5
) -> list[bool]:
    """Return a keep-mask using the Iglewicz-Hoaglin modified z-score.

    The modified z-score is defined as ``0.6745 * (x - median) / MAD``
    where ``MAD = median(|x - median(x)|)``. A value is kept when its
    absolute modified z-score is at most ``k`` (default 3.5, the
    Iglewicz-Hoaglin recommendation).

    Edge cases: if there are fewer than 2 values, or ``MAD == 0`` (all
    values are identical), every value is kept (no rejection).

    Args:
        values: Per-request metric series (e.g. TTFTs in seconds).
        k: Modified-z-score rejection threshold (default 3.5).

    Returns:
        A list of booleans of the same length as ``values``.
        ``True`` means the value should be included; ``False`` means it
        is an outlier and should be dropped.
    """
    if len(values) < 2:
        return [True] * len(values)
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad == 0.0:
        return [True] * len(values)
    modified_z = 0.6745 * (arr - med) / mad
    return [bool(abs(mz) <= k) for mz in modified_z.tolist()]


DEFAULT_WINDOW_SIZE = 50
DEFAULT_TTFT_THRESHOLD = 0.5
DEFAULT_TPOT_THRESHOLD = 0.3
DEFAULT_SUSTAINED_COUNT = DEFAULT_WINDOW_SIZE // 2

DetectionMode = Literal["full", "ttft_only"]


@dataclass
class SteadyStateWindow:
    """Result of steady-state auto-detection on a benchmark run."""

    detected: bool
    """Whether a steady-state window was found."""
    start_index: int | None
    """Min original index in the steady-state window. None if not detected."""
    end_index: int | None
    """Max original index + 1 (exclusive) in the window. None if not detected.

    For multi-turn benchmarks where requests interleave across sessions, use
    ``steady_state_indices`` for the exact set of requests in the window.
    """
    steady_state_indices: list[int]
    """Original output indices of all requests in the steady-state window.

    Empty if not detected. Use this instead of start_index/end_index when
    requests may not be contiguous in dispatch order (e.g., multi-turn
    benchmarks).
    """
    total_requests: int
    """Number of valid requests considered for detection.

    Filters out cancelled, failed, requests missing timestamps or TPOT data.
    """
    steady_state_count: int
    """Number of requests in the detected window."""
    warning: str | None
    """Human-readable explanation when detection fails."""
    window_size: int
    """Rolling window size used for detection."""
    ttft_threshold: float
    """Threshold for TTFT stabilization."""
    tpot_threshold: float
    """Threshold for TPOT stabilization."""
    mode: DetectionMode = "full"
    """Detection mode.

    ``"full"`` uses TTFT for ramp-up and TPOT for ramp-down.
    ``"ttft_only"`` uses TTFT for both, when TPOT is absent
    (e.g., prefill-only workloads).
    """


def _rolling_mad_over_median(values: list[float], window: int) -> list[float]:
    """Compute rolling MAD/median over a sliding window.

    Returns one value per position starting at index (window - 1).
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    if n < window:
        return []

    # Create a (n - window + 1, window) view without copying data.
    shape = (n - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    medians = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - medians[:, np.newaxis]), axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(medians == 0, np.inf, mad / medians)

    return result.tolist()


def _find_first_stable_run(
    series: list[float], threshold: float, sustained_count: int
) -> int | None:
    """Find the first index where the series stays below threshold for sustained_count points."""
    run_length = 0
    for idx, val in enumerate(series):
        if val <= threshold:
            run_length += 1
            if run_length >= sustained_count:
                return idx - sustained_count + 1
        else:
            run_length = 0
    return None


def _find_last_stable_run(
    series: list[float], threshold: float, sustained_count: int
) -> int | None:
    """Find the last index where the series stays below threshold for sustained_count points (scanning from end)."""
    run_length = 0
    for idx in range(len(series) - 1, -1, -1):
        if series[idx] <= threshold:
            run_length += 1
            if run_length >= sustained_count:
                return idx + sustained_count - 1
        else:
            run_length = 0
    return None


def _empty_result(
    *,
    total_requests: int,
    warning: str | None,
    window_size: int,
    ttft_threshold: float,
    tpot_threshold: float,
    mode: DetectionMode = "full",
) -> SteadyStateWindow:
    return SteadyStateWindow(
        detected=False,
        start_index=None,
        end_index=None,
        steady_state_indices=[],
        total_requests=total_requests,
        steady_state_count=0,
        warning=warning,
        window_size=window_size,
        ttft_threshold=ttft_threshold,
        tpot_threshold=tpot_threshold,
        mode=mode,
    )


def detect_steady_state(
    outputs: Sequence[RequestFuncOutput],
    window_size: int = DEFAULT_WINDOW_SIZE,
    ttft_threshold: float = DEFAULT_TTFT_THRESHOLD,
    tpot_threshold: float = DEFAULT_TPOT_THRESHOLD,
    sustained_count: int = DEFAULT_SUSTAINED_COUNT,
    *,
    max_concurrency: int | None = None,
) -> SteadyStateWindow:
    """Detect the steady-state region within a benchmark run.

    Filters to valid requests (successful, non-cancelled, with timestamps
    and positive TTFT; non-empty TPOT in full mode), sorts by submit time,
    then uses rolling MAD/median to find where metrics stabilize.

    Falls back to a TTFT-only path when no request populated TPOT (e.g.,
    prefill-only workloads producing <=1 output token per request).

    Args:
        outputs: Request outputs in dispatch order.
        window_size: Number of requests in the rolling window.
        ttft_threshold: Threshold for TTFT stabilization.
        tpot_threshold: Threshold for TPOT stabilization.
        sustained_count: Consecutive points below threshold required
            to confirm stabilization.
        max_concurrency: Benchmark concurrency level. At ``1`` there is
            no queueing and detection is skipped.
    """
    # At concurrency=1 there's no ramp; pass len(outputs) so telemetry
    # doesn't misread the skip as zero valid requests.
    if max_concurrency == 1:
        return _empty_result(
            total_requests=len(outputs),
            warning=None,
            window_size=window_size,
            ttft_threshold=ttft_threshold,
            tpot_threshold=tpot_threshold,
        )

    # Only skip TPOT when no request populated it; otherwise keep the
    # full path so "too few valid" surfaces honestly instead of silently
    # masking a TPOT detection failure.
    min_required = window_size * 2
    mode: DetectionMode = (
        "full" if any(out.tpot for out in outputs) else "ttft_only"
    )
    require_tpot = mode == "full"
    indexed = [
        (i, out)
        for i, out in enumerate(outputs)
        if (
            out.success
            and not out.cancelled
            and out.request_submit_time is not None
            and out.ttft is not None
            and out.ttft > 0
            and (not require_tpot or out.tpot)
        )
    ]
    total = len(indexed)

    if total < min_required:
        if mode == "full":
            reason = (
                " Detection filters out requests that are cancelled, failed,"
                " or have no TPOT data. Prefill-only or short-output"
                " workloads (<=1 output token per request) will filter out"
                " entirely."
            )
        else:
            reason = (
                " TPOT was absent across the run, so detection ran in"
                " TTFT-only mode; the run has too few valid requests"
                " (cancelled, failed, or missing timestamps/TTFT are"
                " filtered out)."
            )
        return _empty_result(
            total_requests=total,
            warning=(
                f"Too few valid requests ({total} of {len(outputs)} total)"
                f" for steady-state detection (need at least {min_required})."
                f"{reason}"
            ),
            window_size=window_size,
            ttft_threshold=ttft_threshold,
            tpot_threshold=tpot_threshold,
            mode=mode,
        )

    result = _empty_result(
        total_requests=total,
        warning=None,
        window_size=window_size,
        ttft_threshold=ttft_threshold,
        tpot_threshold=tpot_threshold,
        mode=mode,
    )

    indexed.sort(key=lambda x: x[1].request_submit_time or 0.0)

    # Redundant `is not None` check kept to narrow `float | None` to
    # `float` for mypy; the filter above already guarantees ttft > 0.
    ttfts = [out.ttft for _, out in indexed if out.ttft is not None]
    ttft_mads = _rolling_mad_over_median(ttfts, window_size)
    if not ttft_mads:
        result.warning = "Could not compute rolling statistics."
        return result

    ramp_up_idx = _find_first_stable_run(
        ttft_mads, ttft_threshold, sustained_count
    )
    if ramp_up_idx is None:
        result.warning = (
            f"TTFT never stabilized below MAD/median threshold"
            f" {ttft_threshold:.2f}. The system may have been"
            f" overloaded for the entire run."
        )
        return result

    steady_start = ramp_up_idx

    if mode == "full":
        tpots = [float(np.mean(out.tpot)) for _, out in indexed]
        tpot_mads = _rolling_mad_over_median(tpots, window_size)
        if not tpot_mads:
            result.warning = "Could not compute rolling statistics."
            return result

        ramp_down_idx = _find_last_stable_run(
            tpot_mads, tpot_threshold, sustained_count
        )
        if ramp_down_idx is None:
            result.warning = (
                f"TPOT never stabilized below MAD/median threshold"
                f" {tpot_threshold:.2f}. The system may have been"
                f" overloaded for the entire run."
            )
            return result
        steady_end = ramp_down_idx + window_size
    else:
        # Reuse ttft_mads for ramp-down (TTFT can still destabilize at
        # end-of-run). Fall back to end-of-valid-set if no ramp-down is
        # found, so a stable run isn't rejected for lack of a ramp.
        ramp_down_idx = _find_last_stable_run(
            ttft_mads, ttft_threshold, sustained_count
        )
        steady_end = (
            ramp_down_idx + window_size if ramp_down_idx is not None else total
        )

    if steady_start >= steady_end:
        result.warning = (
            "Ramp-up and ramp-down regions overlap; no steady-state"
            " window found."
        )
        return result

    steady_count = steady_end - steady_start
    if steady_count < window_size:
        result.warning = (
            f"Steady-state window too small ({steady_count} requests,"
            f" minimum {window_size})."
        )
        return result

    ss_indices = [indexed[i][0] for i in range(steady_start, steady_end)]
    result.detected = True
    result.steady_state_indices = ss_indices
    result.start_index = min(ss_indices)
    result.end_index = max(ss_indices) + 1
    result.steady_state_count = steady_count
    return result
