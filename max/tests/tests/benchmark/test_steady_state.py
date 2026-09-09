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

"""Tests for steady-state auto-detection."""

from __future__ import annotations

import random

from max.benchmark.benchmark_shared.request import RequestFuncOutput
from max.benchmark.benchmark_shared.steady_state import (
    _rolling_mad_over_median,
    detect_steady_state,
    reject_metric_outliers,
)


def _make_output(
    submit_time: float,
    ttft: float,
    tpot: list[float],
    latency: float | None = None,
) -> RequestFuncOutput:
    """Create a minimal RequestFuncOutput for testing."""
    return RequestFuncOutput(
        success=True,
        latency=latency if latency is not None else ttft + sum(tpot),
        ttft=ttft,
        prompt_len=10,
        generated_text="test",
        itl=tpot,
        tpot=tpot,
        request_submit_time=submit_time,
    )


def test_rolling_mad_basic() -> None:
    """Rolling MAD/median over a constant series should be 0."""
    values = [5.0] * 10
    mads = _rolling_mad_over_median(values, window=5)
    assert len(mads) == 6
    for mad in mads:
        assert mad == 0.0


def test_rolling_mad_too_few_values() -> None:
    """Rolling MAD/median with fewer values than window returns empty."""
    assert _rolling_mad_over_median([1.0, 2.0], window=5) == []


def test_rolling_mad_heavy_tail() -> None:
    """MAD/median stays low on data with stable center but heavy tails."""
    # 90% at 1.0, 10% outliers at 10.0: stable central tendency, heavy tails
    values = ([1.0] * 45 + [10.0] * 5) * 4
    mads = _rolling_mad_over_median(values, window=50)
    assert len(mads) > 0
    assert all(m < 0.5 for m in mads)


def test_steady_state_stable_run() -> None:
    """A fully stable run should detect steady state covering most requests."""
    random.seed(99)
    n = 200
    outputs = []
    for i in range(n):
        outputs.append(
            _make_output(
                submit_time=float(i),
                ttft=0.05 + random.uniform(-0.003, 0.003),
                tpot=[0.02 + random.uniform(-0.001, 0.001) for _ in range(3)],
            )
        )

    result = detect_steady_state(
        outputs, window_size=20, ttft_threshold=0.5, tpot_threshold=0.3
    )

    assert result.detected is True
    assert result.start_index is not None
    assert result.end_index is not None
    assert result.warning is None
    assert result.steady_state_count >= n // 2


def test_steady_state_with_warmup_and_cooldown() -> None:
    """Detect steady state with noisy warmup and cooldown phases."""
    random.seed(123)
    outputs: list[RequestFuncOutput] = []

    # Warmup: wildly varying TTFT
    for i in range(60):
        outputs.append(
            _make_output(
                submit_time=float(i),
                ttft=random.uniform(0.1, 2.0),
                tpot=[0.02],
            )
        )

    # Steady state: stable metrics
    for i in range(60, 260):
        outputs.append(
            _make_output(
                submit_time=float(i),
                ttft=0.05 + random.uniform(-0.005, 0.005),
                tpot=[0.02 + random.uniform(-0.002, 0.002)],
            )
        )

    # Cooldown: wildly varying TPOT
    for i in range(260, 320):
        outputs.append(
            _make_output(
                submit_time=float(i),
                ttft=0.05,
                tpot=[random.uniform(0.01, 0.5)],
            )
        )

    result = detect_steady_state(
        outputs, window_size=30, ttft_threshold=0.15, tpot_threshold=0.15
    )

    assert result.detected is True
    assert result.start_index is not None
    assert result.end_index is not None
    assert result.start_index >= 30
    assert result.end_index <= 290


def test_steady_state_ttft_only_fallback_for_prefill_only_workload() -> None:
    """Prefill-only workloads produce ≤1 output token per request so TPOT
    is always empty. The full filter drops everything, but the TTFT-only
    fallback should still detect steady state using TTFT alone."""
    random.seed(7)
    n = 200
    outputs: list[RequestFuncOutput] = []

    # Warmup: noisy TTFT.
    for i in range(40):
        outputs.append(
            _make_output(
                submit_time=float(i),
                ttft=random.uniform(0.1, 1.5),
                tpot=[],  # Prefill-only: no inter-token latencies.
                latency=random.uniform(0.1, 1.5),
            )
        )
    # Steady: stable TTFT.
    for i in range(40, n):
        outputs.append(
            _make_output(
                submit_time=float(i),
                ttft=0.05 + random.uniform(-0.005, 0.005),
                tpot=[],
                latency=0.05,
            )
        )

    result = detect_steady_state(outputs, window_size=20, ttft_threshold=0.15)

    assert result.detected is True
    assert result.mode == "ttft_only"
    assert result.warning is None
    assert result.start_index is not None and result.end_index is not None
    # Ramp-up should fall within the noisy prefix; window extends to end.
    assert result.start_index >= 20
    assert result.end_index == n


def test_steady_state_ttft_only_fallback_requires_enough_data() -> None:
    """If even the TTFT-only filter yields too few requests, detection
    should still return a "too few" warning rather than crashing. In
    ttft_only mode TPOT wasn't filtered, so the warning should say the
    run has too few valid requests and name TTFT-only mode (not blame
    TPOT filtering, which only applies in full mode)."""
    outputs = [
        _make_output(submit_time=float(i), ttft=0.05, tpot=[])
        for i in range(10)
    ]

    result = detect_steady_state(outputs, window_size=20)

    assert result.detected is False
    assert result.mode == "ttft_only"
    assert result.warning is not None
    assert "Too few" in result.warning
    assert "TTFT-only" in result.warning


def test_steady_state_full_mode_preferred_over_fallback() -> None:
    """With full (TPOT-populated) data, the full-mode path should run and
    the result should carry mode="full". The fallback is only for the
    TPOT-empty case."""
    random.seed(11)
    n = 200
    outputs = [
        _make_output(
            submit_time=float(i),
            ttft=0.05 + random.uniform(-0.002, 0.002),
            tpot=[0.02 + random.uniform(-0.001, 0.001) for _ in range(3)],
        )
        for i in range(n)
    ]

    result = detect_steady_state(outputs, window_size=20)

    assert result.detected is True
    assert result.mode == "full"


def test_steady_state_concurrency_one_short_circuits() -> None:
    """At max_concurrency=1 there's no queueing to produce a ramp, so
    detection should skip silently (no warning, no detected window)."""
    # Provide a plausible stable run that would otherwise detect fine,
    # to prove the short-circuit is driven by concurrency and not data.
    n = 200
    outputs = [
        _make_output(submit_time=float(i), ttft=0.05, tpot=[0.02, 0.02, 0.02])
        for i in range(n)
    ]

    result = detect_steady_state(outputs, window_size=20, max_concurrency=1)

    assert result.detected is False
    assert result.warning is None
    assert result.start_index is None
    assert result.end_index is None
    assert result.steady_state_count == 0
    # mode stays at default "full" since no detection ran.
    assert result.mode == "full"
    # total_requests should reflect the input size; the filter didn't
    # run, so 0 would be misleading to downstream telemetry.
    assert result.total_requests == n


def test_steady_state_too_few_requests() -> None:
    """Too few requests should fail with a warning."""
    outputs = [
        _make_output(submit_time=float(i), ttft=0.05, tpot=[0.02])
        for i in range(10)
    ]

    result = detect_steady_state(outputs, window_size=20)

    assert result.detected is False
    assert result.warning is not None
    assert "Too few" in result.warning
    # Warning should name the TPOT filter / prefill-only workloads so
    # users know why detection skipped.
    assert "TPOT" in result.warning
    assert "prefill" in result.warning.lower()


def test_steady_state_never_stabilizes() -> None:
    """A run that never stabilizes should produce a warning."""
    random.seed(42)
    n = 200
    outputs = []
    for i in range(n):
        outputs.append(
            _make_output(
                submit_time=float(i),
                ttft=random.uniform(0.01, 10.0),
                tpot=[0.02],
            )
        )

    result = detect_steady_state(
        outputs,
        window_size=20,
        ttft_threshold=0.05,
    )

    assert result.detected is False
    assert result.warning is not None


def test_steady_state_no_timestamps() -> None:
    """Requests without timestamps should be filtered out gracefully."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.05,
            prompt_len=10,
            generated_text="test",
            itl=[0.02],
            tpot=[0.02],
            request_submit_time=None,
        )
        for _ in range(100)
    ]

    result = detect_steady_state(outputs, window_size=20)

    assert result.detected is False
    assert result.warning is not None
    assert "Too few" in result.warning


def test_steady_state_with_failed_requests() -> None:
    """Detection works correctly when failed requests are interspersed."""
    n = 200
    outputs: list[RequestFuncOutput] = []
    for i in range(n):
        if i % 10 == 5:
            # Intersperse failed requests
            outputs.append(
                RequestFuncOutput(
                    success=False,
                    latency=0.0,
                    ttft=0.0,
                    prompt_len=10,
                    generated_text="",
                    error="simulated failure",
                    request_submit_time=float(i),
                )
            )
        else:
            outputs.append(
                _make_output(
                    submit_time=float(i),
                    ttft=0.05,
                    tpot=[0.02, 0.02, 0.02],
                )
            )

    result = detect_steady_state(outputs, window_size=20)

    assert result.detected is True
    assert result.start_index is not None
    assert result.end_index is not None
    # Failed requests should be filtered out; total_requests < n
    assert result.total_requests < n
    assert result.total_requests == n - n // 10


def test_steady_state_indices_map_to_original_order() -> None:
    """Returned indices should reference the original outputs list."""
    n = 150
    outputs = [
        _make_output(submit_time=float(i), ttft=0.05, tpot=[0.02, 0.02, 0.02])
        for i in range(n)
    ]

    result = detect_steady_state(outputs, window_size=20)

    assert result.detected is True
    assert result.start_index is not None
    assert result.end_index is not None
    assert 0 <= result.start_index < result.end_index <= n
    ss_slice = outputs[result.start_index : result.end_index]
    assert len(ss_slice) == result.end_index - result.start_index
    for out in ss_slice:
        assert out.success


def test_reject_metric_outliers_keep_all_for_small_input() -> None:
    """With fewer than 2 values every element is kept."""
    assert reject_metric_outliers([]) == []
    assert reject_metric_outliers([0.5]) == [True]


def test_reject_metric_outliers_keep_all_when_mad_zero() -> None:
    """When all values are identical MAD==0 and nothing is rejected."""
    values = [0.05] * 20
    keep = reject_metric_outliers(values)
    assert all(keep)
    assert len(keep) == 20


def test_reject_metric_outliers_drops_extremes() -> None:
    """Clear outliers (modified z-score >> 3.5) are rejected.

    The MAD is only non-zero when the center of the distribution has spread.
    Use a mix of values with two different levels so the MAD is positive,
    then inject extreme outliers.
    """
    # Build a stable set with some natural spread (so MAD > 0).
    # Values alternate between 0.04 and 0.06 → median≈0.05, MAD≈0.01.
    center = [0.04, 0.06] * 9  # 18 values; median 0.05, MAD 0.01
    outliers = [100.0, -100.0]  # |mz| ≈ 0.6745 * 99.95 / 0.01 ≈ 6740 >> 3.5
    values = center + outliers
    keep = reject_metric_outliers(values)
    assert len(keep) == len(values)
    # The last two (outliers) must be rejected
    assert keep[-1] is False
    assert keep[-2] is False
    # All stable values must be kept (|mz| ≤ 0.6745)
    assert all(keep[:18])


def test_reject_metric_outliers_constant_0645_factor() -> None:
    """Modified z-score uses the 0.6745 Iglewicz-Hoaglin constant.

    With a known dataset the threshold at k=3.5 maps to a specific
    absolute deviation from the median. Verify the constant is correct by
    checking that a value just at the boundary is kept and one beyond is
    dropped.
    """
    import numpy as np

    # Build 10 values with median=0 and MAD=1.
    # Boundary: |mz| = 0.6745 * |x| / 1 = 3.5  =>  |x| = 3.5/0.6745 ≈ 5.189
    base = [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    # median = 0; MAD = median(|x|) = 1.0
    assert float(np.median(np.abs(np.array(base) - np.median(base)))) == 1.0

    threshold_val = 3.5 / 0.6745  # ≈ 5.189
    below = base + [threshold_val - 0.01]  # |mz| < 3.5 → keep
    above = base + [threshold_val + 0.01]  # |mz| > 3.5 → drop

    assert reject_metric_outliers(below)[-1] is True
    assert reject_metric_outliers(above)[-1] is False


def test_reject_metric_outliers_symmetric() -> None:
    """Rejection is symmetric around the median."""
    values = [0.0] * 10 + [10.0, -10.0]
    keep = reject_metric_outliers(values)
    # Both extremes should be rejected (or both kept; they're symmetric)
    assert keep[-1] == keep[-2]
