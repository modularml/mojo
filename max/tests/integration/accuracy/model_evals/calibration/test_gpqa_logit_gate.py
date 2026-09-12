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
# ===---------------------------------------------------------------------=== #
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
# ===---------------------------------------------------------------------=== #
"""Unit tests for the GPQA logit-shift gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibration.gpqa_gate import (
    PARK_ACC,
    PARK_STOP,
    load_gate_hist,
    make_spec,
    score_results,
    score_spec,
    solve_repeats,
    subset_ids,
    subset_park_rates,
)
from calibration.gpqa_logit_gate import run_gate


def test_hist_and_subsets() -> None:
    hist = load_gate_hist()
    assert len(hist.q_trunc) == 198
    assert len(hist.noisy_ids) == 15
    assert len(hist.rare_ids) == 34
    assert len(hist.hot_rare_ids) == 12
    assert len(hist.ever_trunc_ids) == 49
    assert set(hist.rare_ids).isdisjoint(hist.noisy_ids)
    assert set(hist.hot_rare_ids) <= set(hist.rare_ids)
    assert subset_ids(hist, "noisy_15") == hist.noisy_ids
    assert len(subset_ids(hist, "plus_bucket")) == 27
    assert len(subset_ids(hist, "ever_trunc")) == 49


def test_park_matches_explorer() -> None:
    scored = score_spec(make_spec("bins", "noisy_15"))
    assert scored.stop_h == PARK_STOP
    assert scored.stop_r == PARK_STOP - 0.005
    assert scored.acc_h == PARK_ACC
    assert scored.acc_r == PARK_ACC - 0.01
    tight = score_spec(make_spec("bins", "noisy_15", delta_stop=0.002))
    assert tight.stop_h == PARK_STOP
    assert tight.stop_r == PARK_STOP - 0.002


def test_default_joint_has_stop_and_acc() -> None:
    scored = score_spec(make_spec("bins", "noisy_15"))
    assert scored.acc_cutoff is not None
    assert scored.stop_cutoff is not None
    assert scored.stop_cutoff >= 0
    assert scored.n_repeats == max(scored.n_repeats, 1)
    assert scored.cost == scored.n_repeats * 15


def test_bucket_and_per_prompt_catalog_rows() -> None:
    bins_bucket = score_spec(make_spec("bins", "plus_bucket"))
    prompt_49 = score_spec(make_spec("per_prompt", "ever_trunc"))
    assert len(bins_bucket.spec.prompt_ids) == 27
    assert len(prompt_49.spec.prompt_ids) == 49
    assert bins_bucket.cost < prompt_49.cost


def test_solve_repeats_toy_coin() -> None:
    solved = solve_repeats([0.10], [0.30], 0.05, 0.05)
    assert solved is not None
    assert solved.false_positive <= 0.05 + 1e-12
    assert solved.false_negative <= 0.05 + 1e-12


def _write_results(
    path: Path,
    n_rows: int,
    *,
    n_trunc: int,
    n_wrong: int,
    prompt_ids: list[int] | None = None,
) -> Path:
    ids = prompt_ids if prompt_ids is not None else list(range(15))
    n_ids = len(ids)
    lines = [
        json.dumps(
            {
                "prompt_index": ids[i % n_ids],
                "repeat_index": i // n_ids,
                "finish_reason": "length" if i < n_trunc else "stop",
                "correct": i >= n_wrong,
            }
        )
        for i in range(n_rows)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_need_a_metric() -> None:
    with pytest.raises(ValueError, match="need stop and/or accuracy"):
        make_spec("bins", "noisy_15", want_stop=False, want_acc=False)


def test_metric_flags_recount_repeats() -> None:
    both = score_spec(make_spec("bins", "noisy_15"))
    stop_only = score_spec(make_spec("bins", "noisy_15", want_acc=False))
    acc_only = score_spec(make_spec("bins", "noisy_15", want_stop=False))
    assert stop_only.n_repeats < both.n_repeats
    assert stop_only.acc_cutoff is None
    assert stop_only.stop_cutoff is not None
    assert acc_only.stop_cutoff is None
    assert acc_only.acc_cutoff is not None
    assert acc_only.n_repeats == both.n_repeats
    assert stop_only.cost == stop_only.n_repeats * 15
    assert acc_only.cost == acc_only.n_repeats * 15


def test_verdict_requires_both_metrics(tmp_path: Path) -> None:
    scored = score_spec(make_spec("bins", "noisy_15"))
    assert scored.stop_cutoff is not None
    assert scored.acc_cutoff is not None
    ok = score_results(
        _write_results(
            tmp_path / "pass.jsonl",
            scored.cost,
            n_trunc=0,
            n_wrong=0,
            prompt_ids=scored.spec.prompt_ids,
        ),
        scored,
    )
    assert ok.status == "pass"
    stop_fail = score_results(
        _write_results(
            tmp_path / "stop.jsonl",
            scored.cost,
            n_trunc=scored.stop_cutoff + 1,
            n_wrong=0,
            prompt_ids=scored.spec.prompt_ids,
        ),
        scored,
    )
    assert stop_fail.status == "fail"
    acc_fail = score_results(
        _write_results(
            tmp_path / "acc.jsonl",
            scored.cost,
            n_trunc=0,
            n_wrong=scored.acc_cutoff + 1,
            prompt_ids=scored.spec.prompt_ids,
        ),
        scored,
    )
    assert acc_fail.status == "fail"


def test_verdict_ignores_disabled_metric(tmp_path: Path) -> None:
    scored = score_spec(make_spec("bins", "noisy_15", want_acc=False))
    ok = score_results(
        _write_results(
            tmp_path / "acc_noise.jsonl",
            scored.cost,
            n_trunc=0,
            n_wrong=scored.cost,
            prompt_ids=scored.spec.prompt_ids,
        ),
        scored,
    )
    assert ok.status == "pass"
    assert ok.acc_cutoff is None
    stop_fail = score_results(
        _write_results(
            tmp_path / "stop.jsonl",
            scored.cost,
            n_trunc=(scored.stop_cutoff or 0) + 1,
            n_wrong=0,
            prompt_ids=scored.spec.prompt_ids,
        ),
        scored,
    )
    assert stop_fail.status == "fail"


def test_compare_only_report(tmp_path: Path) -> None:
    work = tmp_path / "gate"
    run_gate(
        work_dir=work, hist_mode="bins", subset="noisy_15", compare_only=True
    )
    report = (work / "REPORT.md").read_text()
    assert "## Selected" in report
    assert "## Verdict" in report
    assert "## Subset worlds" in report
    assert "## Configurations" in report
    assert "`bins:noisy_15`" in report
    assert "`per_prompt:plus_bucket`" in report


def test_compare_only_stop_only_recounts(tmp_path: Path) -> None:
    work = tmp_path / "stop"
    run_gate(
        work_dir=work,
        hist_mode="bins",
        subset="noisy_15",
        compare_only=True,
        want_acc=False,
    )
    selected = json.loads((work / "selected.json").read_text())
    both = score_spec(make_spec("bins", "noisy_15"))
    assert selected["want_acc"] is False
    assert selected["n_repeats"] < both.n_repeats
    assert selected["acc_cutoff"] is None
    assert selected["cost"] == selected["n_repeats"] * 15


def test_smoke_overrides_repeats_and_ids(tmp_path: Path) -> None:
    results = _write_results(
        tmp_path / "results.jsonl", 2, n_trunc=0, n_wrong=0
    )
    work = tmp_path / "smoke"
    run_gate(
        work_dir=work,
        hist_mode="bins",
        subset="noisy_15",
        results_jsonl=results,
        n_repeats=1,
        prompt_ids=[8, 12],
        include_catalog=False,
    )
    selected = json.loads((work / "selected.json").read_text())
    assert selected["prompt_ids"] == [8, 12]
    assert selected["n_repeats"] == 1
    assert selected["cost"] == 2
    report = (work / "REPORT.md").read_text()
    assert "✅ pass" in report
    assert "ids=[8, 12]" in report


def test_score_existing_jsonl(tmp_path: Path) -> None:
    scored = score_spec(make_spec("bins", "noisy_15"))
    results = _write_results(
        tmp_path / "results.jsonl",
        scored.cost,
        n_trunc=0,
        n_wrong=0,
        prompt_ids=scored.spec.prompt_ids,
    )
    work = tmp_path / "live"
    run_gate(
        work_dir=work,
        hist_mode="bins",
        subset="noisy_15",
        results_jsonl=results,
        model="MiniMaxAI/MiniMax-M3",
        base_url="http://127.0.0.1:8000/v1",
    )
    payload = json.loads((work / "verdict.json").read_text())
    assert payload["status"] == "pass"
    assert payload["acc_status"] == "pass"
    assert payload["stop_status"] == "pass"
    assert payload["acc_rate"] == 1.0
    assert payload["stop_rate"] == 1.0
    assert payload["model"] == "MiniMaxAI/MiniMax-M3"
    assert payload["base_url"] == "http://127.0.0.1:8000/v1"
    report = (work / "REPORT.md").read_text()
    assert "✅ pass" in report
    assert "MiniMaxAI/MiniMax-M3" in report
    assert "http://127.0.0.1:8000/v1" in report


def test_subset_park_rates_plus_bucket() -> None:
    acc_h, acc_r, stop_h, stop_r = subset_park_rates(
        make_spec("per_prompt", "plus_bucket")
    )
    assert acc_h is not None and acc_r is not None
    assert stop_h is not None and stop_r is not None
    assert abs(acc_h - 0.681) < 0.002
    assert abs(acc_r - 0.636) < 0.002
    assert abs(stop_h - 0.896) < 0.002
    assert abs(stop_r - 0.861) < 0.002


def test_report_splits_acc_and_stop_verdicts(tmp_path: Path) -> None:
    scored = score_spec(make_spec("per_prompt", "plus_bucket"))
    assert scored.acc_cutoff is not None
    results = _write_results(
        tmp_path / "mixed.jsonl",
        scored.cost,
        n_trunc=0,
        n_wrong=scored.acc_cutoff + 1,
        prompt_ids=scored.spec.prompt_ids,
    )
    work = tmp_path / "mixed"
    with pytest.raises(SystemExit) as exc:
        run_gate(
            work_dir=work,
            hist_mode="per_prompt",
            subset="plus_bucket",
            results_jsonl=results,
            include_catalog=False,
        )
    assert exc.value.code == 1
    payload = json.loads((work / "verdict.json").read_text())
    assert payload["status"] == "fail"
    assert payload["acc_status"] == "fail"
    assert payload["stop_status"] == "pass"
    report = (work / "REPORT.md").read_text()
    assert "❌ fail" in report
    assert "✅ pass" in report
