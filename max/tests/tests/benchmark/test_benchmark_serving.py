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
"""Benchmark serving dev unit tests"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from max.benchmark.benchmark_serving import (
    _resolve_skip_counts,
    _seed_for_concurrency,
    parse_args,
)


def test_resolve_skip_counts() -> None:
    def rsc(
        orig_skip_first: int | None = None,
        orig_skip_last: int | None = None,
        request_rate: float = float("inf"),
        max_concurrency: int | None = None,
        ignore_first_turn_stats: bool = False,
        warmup_to_steady_state: bool = False,
    ) -> tuple[int, int]:
        return _resolve_skip_counts(
            orig_skip_first=orig_skip_first,
            orig_skip_last=orig_skip_last,
            request_rate=request_rate,
            max_concurrency=max_concurrency,
            ignore_first_turn_stats=ignore_first_turn_stats,
            warmup_to_steady_state=warmup_to_steady_state,
        )

    # finite rate: auto-None → 0; explicit values are preserved
    assert rsc(request_rate=2.0) == (0, 0)
    assert rsc(orig_skip_first=3, orig_skip_last=5, request_rate=2.0) == (3, 5)
    assert rsc(orig_skip_last=5, request_rate=2.0) == (0, 5)

    # infinite rate, max_concurrency > 1: auto-None → concurrency value
    assert rsc(max_concurrency=4) == (4, 4)
    assert rsc(orig_skip_first=2, max_concurrency=4) == (2, 4)
    assert rsc(orig_skip_last=2, max_concurrency=4) == (4, 2)
    assert rsc(orig_skip_first=2, orig_skip_last=3, max_concurrency=4) == (2, 3)

    # infinite rate, max_concurrency=1: sequential, no ramp-up → 0
    assert rsc(max_concurrency=1) == (0, 0)

    # infinite rate, max_concurrency=None: unbounded → 0
    assert rsc(max_concurrency=None) == (0, 0)

    # ignore_first_turn_stats resets skip_first unless warmup_to_steady_state
    assert rsc(max_concurrency=4, ignore_first_turn_stats=True) == (0, 4)
    assert rsc(
        max_concurrency=4,
        ignore_first_turn_stats=True,
        warmup_to_steady_state=True,
    ) == (4, 4)
    assert rsc(max_concurrency=4, ignore_first_turn_stats=False) == (4, 4)

    # ignore_first_turn_stats with skip_first=0 (falsy): no override
    assert rsc(request_rate=2.0, ignore_first_turn_stats=True) == (0, 0)


def test_seed_for_concurrency_is_deterministic() -> None:
    """Same (seed, max_concurrency) always derives the same sub-seed."""
    assert _seed_for_concurrency(24301, 4) == _seed_for_concurrency(24301, 4)


def test_seed_for_concurrency_differs_across_concurrency_levels() -> None:
    """Different concurrency levels get distinct sub-seeds from one base seed."""
    seeds = {_seed_for_concurrency(24301, mc) for mc in [1, 4, 6, 8, 16]}
    assert len(seeds) == 5


def test_parse_args_pre_parses_concurrency_and_request_rate_sweeps() -> None:
    """CLI strings for concurrency and request-rate sweeps become parsed lists."""
    cfg = parse_args(
        [
            "--model",
            "myorg/model",
            "--max-concurrency",
            "1,none,4",
            "--request-rate",
            "inf,2.5",
        ]
    )
    assert list(cfg.max_concurrency) == [1, None, 4]
    assert cfg.request_rate[0] == float("inf")
    assert cfg.request_rate[1] == pytest.approx(2.5)


def test_benchmark_serving_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the benchmark serving help function."""
    # Mock sys.argv to simulate running with --help flag
    test_args = ["benchmark_serving.py", "--help"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            parse_args()

        # Verify it exited with code 0 (success)
        assert excinfo.value.code == 0

        # Capture and verify the help output
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()
