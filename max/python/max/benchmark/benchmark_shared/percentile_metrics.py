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

"""Lightweight percentile-metric types.

Carved out of :mod:`max.benchmark.benchmark_shared.metrics` so consumers
that just need the *type definitions* (e.g. dashboards deserialising
result rows from BigQuery) can import them without pulling in the
benchmark runner's heavy dependency tree (``max.serve``,
``max.profiler``, transformers, huggingface-hub, openai, etc.).

The full ``metrics`` module re-exports everything defined here, so
existing ``from max.benchmark.benchmark_shared.metrics import …``
imports keep working unchanged.

Imports here are kept to stdlib + ``pydantic``-free so the
``:percentile_metrics`` bazel target has a minimal dependency surface.
"""

from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

_LABEL_COL_WIDTH = 34
"""Width of the leading label column in single-line metric output."""

_STAT_COL_WIDTH = 14
"""Width of each ``name=value`` stat column in single-line metric output."""


def _is_finite_and_positive(value: float) -> bool:
    """Check that a numeric value is finite and positive."""
    return math.isfinite(value) and value > 0


ConfidenceLevel = Literal["high", "medium", "low", "insufficient_data"]


@dataclass
class ConfidenceInfo:
    """Confidence interval metadata for a metric."""

    ci_lower: float
    """Lower bound of the confidence interval (scaled units)."""
    ci_upper: float
    """Upper bound of the confidence interval (scaled units)."""
    ci_relative_width: float
    """Width of the CI as a fraction of the mean."""
    confidence: ConfidenceLevel
    """Classification based on ci_relative_width."""
    sample_size: int
    """Number of data points used to compute the CI."""


_T_CRITICAL_95: Mapping[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
    40: 2.021, 60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980,
}  # fmt: skip
_T_DF_KEYS = sorted(_T_CRITICAL_95.keys())


def finite_or_none(value: float | None) -> float | None:
    """NaN/Inf to None, so JSON output stays standard-parseable.

    Benchmark reports legitimately carry NaN exactly where something went
    wrong (a missing cell, an empty sample); ``json.dumps`` would emit a
    bare ``NaN`` token, which is not JSON — jq, ``JSON.parse``, and most
    non-Python consumers reject the whole document.
    """
    if value is None or not math.isfinite(value):
        return None
    return value


def json_safe(obj: object) -> object:
    """Recursively map NaN/Inf floats to None across a JSON-ready tree.

    Pair with ``json.dumps(..., allow_nan=False)`` so any non-finite value
    that slips past this scrub fails loudly at dump time instead of
    producing an unparseable file.
    """
    if isinstance(obj, float):
        return finite_or_none(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def t_critical_95(df: int) -> float:
    """Look up the 95% t critical value for given degrees of freedom."""
    if df >= 120:
        return 1.96
    for k in reversed(_T_DF_KEYS):
        if df >= k:
            return _T_CRITICAL_95[k]
    return _T_CRITICAL_95[1]


def compute_confidence_info(
    data: list[float], scaled_mean: float, scale_factor: float = 1.0
) -> ConfidenceInfo | None:
    """Compute a 95% t-interval for a metric from raw (unscaled) data.

    Returns:
        The interval and its high/medium/low/insufficient_data
        classification, or ``None`` when fewer than two samples exist or
        the scaled mean is not finite and positive.
    """
    n = len(data)
    if n < 2 or not _is_finite_and_positive(scaled_mean):
        return None

    t = t_critical_95(n - 1)
    se = statistics.stdev(data) * scale_factor / math.sqrt(n)
    margin = t * se
    ci_lower = scaled_mean - margin
    ci_upper = scaled_mean + margin
    ci_relative_width = (ci_upper - ci_lower) / scaled_mean

    confidence: ConfidenceLevel
    if n < 5:
        confidence = "insufficient_data"
    elif ci_relative_width <= 0.10:
        confidence = "high"
    elif ci_relative_width <= 0.20:
        confidence = "medium"
    else:
        confidence = "low"

    return ConfidenceInfo(
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_relative_width=ci_relative_width,
        confidence=confidence,
        sample_size=n,
    )


class Metrics(ABC):
    """Base class for all benchmark metric containers."""

    @abstractmethod
    def validate_metrics(self) -> tuple[bool, list[str]]:
        """Validate metric values are meaningful (not 0, NaN, inf, or negative).

        Returns:
            A ``(success, errors)`` tuple where *success* is ``True`` when all
            checks pass and *errors* is a list of human-readable descriptions
            of any failed checks.
        """
        ...


@dataclass
class PercentileMetrics(Metrics):
    """Container for percentile-based metrics."""

    mean: float
    std: float
    p50: float
    p90: float
    p95: float
    p99: float
    unit: str | None = None
    confidence_info: ConfidenceInfo | None = None

    def _format_single_line(self, label: str) -> str:
        """Render all stats on a single, table-aligned line.

        ``label`` is left-justified into a fixed-width leading column so the
        stat columns line up across successive metrics; an empty label omits
        that column entirely.
        """
        stats = (
            ("Mean", self.mean),
            ("Std", self.std),
            ("P50", self.p50),
            ("P90", self.p90),
            ("P95", self.p95),
            ("P99", self.p99),
        )
        cells = "".join(
            f"{f'{name}={value:.2f}':<{_STAT_COL_WIDTH}}"
            for name, value in stats
        )
        prefix = f"{label:<{_LABEL_COL_WIDTH}}" if label else ""
        return f"{prefix}{cells}".rstrip()

    def __str__(self) -> str:
        """Return a single-line, table-aligned representation of the metrics."""
        return self._format_single_line("")

    def format_with_prefix(self, prefix: str, unit: str | None = None) -> str:
        """Return single-line metrics labeled with a custom prefix and unit."""
        # Use passed unit, or fall back to self.unit
        effective_unit = unit or self.unit
        unit_suffix = f" ({effective_unit})" if effective_unit else ""
        return self._format_single_line(f"{prefix}{unit_suffix}")

    def to_flat_dict(self, name: str) -> dict[str, float]:
        """Flatten percentile stats into ``{"mean_{name}": v, ...}``.

        Note: emits ``median_{name}`` (not ``p50_{name}``) for the 50th
        percentile to preserve the legacy BigQuery column naming consumed by
        the SweepUploader path and benchmark-visibility dashboards. The
        dataclass field is named ``p50``; the legacy column name is kept
        here as the public flattening contract.
        """
        return {
            f"mean_{name}": self.mean,
            f"std_{name}": self.std,
            f"median_{name}": self.p50,
            f"p90_{name}": self.p90,
            f"p95_{name}": self.p95,
            f"p99_{name}": self.p99,
        }

    def confidence_to_flat_dict(self, name: str) -> dict[str, object]:
        """Flatten confidence-interval metadata into ``{"{name}_confidence": v, ...}``."""
        ci = self.confidence_info
        if ci is None:
            return {}
        return {
            f"{name}_ci_lower": ci.ci_lower,
            f"{name}_ci_upper": ci.ci_upper,
            f"{name}_ci_relative_width": ci.ci_relative_width,
            f"{name}_confidence": ci.confidence,
            f"{name}_sample_size": ci.sample_size,
        }

    def validate_metrics(self) -> tuple[bool, list[str]]:
        """Validate that the mean is finite and positive."""
        if not _is_finite_and_positive(self.mean):
            return False, [f"Invalid mean: {self.mean}"]
        return True, []
