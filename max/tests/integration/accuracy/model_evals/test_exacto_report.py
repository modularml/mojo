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
"""Tests for the AutoExacto harness-output normalization."""

import json
from pathlib import Path
from typing import Any

import exacto_report
import pytest

# Shaped after a real `bench eval gpqa_diamond --log-format json` payload:
# 2 questions x 2 epochs, one question right and one wrong, one response
# truncated at the token cap.
OPENBENCH_LOG: dict[str, Any] = {
    "status": "success",
    "eval": {
        "task": "gpqa_diamond",
        "model": "openai-api/local/served-name",
        "model_base_url": "http://localhost:8000/v1",
        "dataset": {"name": "nmayorga7/gpqa_diamond", "samples": 198},
        "config": {"limit": 2, "epochs": 2, "epochs_reducer": ["mean"]},
        "packages": {"inspect_ai": "0.3.125"},
    },
    "plan": {"config": {"temperature": 0.5}},
    "results": {
        "total_samples": 4,
        "completed_samples": 4,
        "scores": [
            {
                "name": "mcq_scorer",
                "scored_samples": 2,
                "unscored_samples": 0,
                "metrics": {
                    "accuracy": {"value": 0.5},
                    "stderr": {"value": 0.25},
                    "std": {"value": 0.5},
                },
            }
        ],
    },
    "reductions": [
        {
            "scorer": "mcq_scorer",
            "samples": [
                {
                    "sample_id": 1,
                    "value": 1.0,
                    "answer": "B",
                    "explanation": "ok",
                },
                {
                    "sample_id": 2,
                    "value": 0.0,
                    "answer": "A",
                    "explanation": "no",
                },
            ],
        }
    ],
    "samples": [
        {
            "id": 1,
            "epoch": 1,
            "output": {
                "usage": {"output_tokens": 100},
                "choices": [{"stop_reason": "stop"}],
            },
        },
        {
            "id": 1,
            "epoch": 2,
            "output": {
                "usage": {"output_tokens": 200},
                "choices": [{"stop_reason": "stop"}],
            },
        },
        {
            "id": 2,
            "epoch": 1,
            "output": {
                "usage": {"output_tokens": 300},
                "choices": [{"stop_reason": "stop"}],
            },
        },
        {
            "id": 2,
            "epoch": 2,
            "output": {
                "usage": {"output_tokens": 400},
                "choices": [{"stop_reason": "length"}],
            },
        },
    ],
}


def test_rows_come_from_the_epoch_reduced_records() -> None:
    rows, _ = exacto_report.parse_openbench_log(OPENBENCH_LOG)
    # One row per question, not per request: 2 questions x 2 epochs = 4 samples.
    assert len(rows) == 2
    assert [r["sample_id"] for r in rows] == [1, 2]
    assert rows[0]["value"] == 1.0
    assert rows[0]["scorer"] == "mcq_scorer"


def test_summary_carries_the_reported_metrics() -> None:
    _, summary = exacto_report.parse_openbench_log(OPENBENCH_LOG)
    assert summary["accuracy"] == 0.5
    assert summary["stderr"] == 0.25
    assert summary["scored_samples"] == 2


def test_summary_token_stats_use_unreduced_samples() -> None:
    _, summary = exacto_report.parse_openbench_log(OPENBENCH_LOG)
    assert summary["mean_output_tokens"] == 250.0
    assert summary["p50_output_tokens"] == 250.0


def test_summary_reports_truncation_for_the_stop_ratio_floor() -> None:
    _, summary = exacto_report.parse_openbench_log(OPENBENCH_LOG)
    assert summary["finish_stop"] == 3
    assert summary["finish_length"] == 1
    assert summary["stop_ratio"] == 0.75


def test_summary_feeds_the_error_budget_guard() -> None:
    _, summary = exacto_report.parse_openbench_log(OPENBENCH_LOG)
    # dump_score() only enforces the budget when both are ints.
    assert isinstance(summary["total"], int)
    assert isinstance(summary["errors"], int)
    assert summary["errors"] == 0


def test_uncompleted_samples_count_as_errors() -> None:
    log = {**OPENBENCH_LOG, "results": {**OPENBENCH_LOG["results"]}}
    log["results"]["completed_samples"] = 1
    _, summary = exacto_report.parse_openbench_log(log)
    assert summary["errors"] == 3


def test_unscored_samples_are_not_lost_when_all_requests_completed() -> None:
    # An endpoint can answer every request and still leave samples unscoreable;
    # total - completed is 0 there, so the scorer's own count has to win.
    log = {**OPENBENCH_LOG, "results": {**OPENBENCH_LOG["results"]}}
    log["results"]["scores"] = [
        {**OPENBENCH_LOG["results"]["scores"][0], "unscored_samples": 2}
    ]
    _, summary = exacto_report.parse_openbench_log(log)
    assert summary["errors"] == 2


def test_summary_records_the_parity_critical_knobs() -> None:
    # A score is only comparable to OpenRouter's if these match their config.
    _, summary = exacto_report.parse_openbench_log(OPENBENCH_LOG)
    assert summary["temperature"] == 0.5
    assert summary["dataset_name"] == "nmayorga7/gpqa_diamond"
    assert summary["epochs"] == 2
    assert summary["epochs_reducer"] == ["mean"]
    assert summary["dataset_samples"] == 198


def test_empty_log_does_not_raise() -> None:
    rows, summary = exacto_report.parse_openbench_log({})
    assert rows == []
    assert summary["accuracy"] is None
    assert summary["stop_ratio"] is None
    assert summary["mean_output_tokens"] == 0.0


def test_score_line_omits_the_range_when_the_caller_gave_none() -> None:
    # No range is built in: provider spread is per-model, so a hardcoded one
    # would be wrong for every model but the one it came from.
    line = exacto_report.format_score_line("gpqa_diamond", {"accuracy": 0.7778})
    assert "0.7778" in line
    assert "published providers" not in line


def test_score_line_shows_a_caller_supplied_range_and_its_source() -> None:
    line = exacto_report.format_score_line(
        "gpqa_diamond",
        {
            "accuracy": 0.7778,
            "stderr": 0.0296,
            "reference": {
                "low": 0.842,
                "high": 0.908,
                "source": "z-ai/glm-5.3-flash",
            },
        },
    )
    assert "0.842-0.908" in line
    assert "z-ai/glm-5.3-flash" in line
    for label in ("IN_BAND", "BELOW", "ABOVE", "PASS", "FAIL"):
        assert label not in line


def test_score_line_handles_a_scoreless_run() -> None:
    assert "none" in exacto_report.format_score_line(
        "gpqa_diamond", {"accuracy": None}
    )


@pytest.mark.parametrize("raw", ["0.842,0.908", " 0.842 , 0.908 "])
def test_parse_reference_accepts_a_valid_range(raw: str) -> None:
    ref = exacto_report.parse_reference(raw, "src")
    assert ref == {"low": 0.842, "high": 0.908, "source": "src"}


def test_parse_reference_returns_none_when_unset() -> None:
    assert exacto_report.parse_reference(None, None) is None
    assert exacto_report.parse_reference("", "src") is None


@pytest.mark.parametrize(
    "raw", ["0.9", "a,b", "0.9,0.1", "-0.1,0.5", "0.5,1.5", "1,2,3"]
)
def test_parse_reference_rejects_bad_input(raw: str) -> None:
    with pytest.raises(SystemExit):
        exacto_report.parse_reference(raw, None)


def test_endpoint_meta_is_nested_not_merged_flat() -> None:
    # A field the server volunteers must never shadow a harness field, so the
    # metadata lands under its own key.
    _, summary = exacto_report.parse_openbench_log(OPENBENCH_LOG)
    exacto_report.merge_endpoint_meta(
        summary, {"accuracy": 0.99, "serve_config": "gemma4_31b_dflash"}
    )
    assert summary["accuracy"] == 0.5
    assert summary["endpoint"]["accuracy"] == 0.99
    assert summary["endpoint"]["serve_config"] == "gemma4_31b_dflash"


def test_endpoint_meta_absent_records_an_empty_dict() -> None:
    # An explicit empty dict, not a missing key: "we did not capture it" should
    # be visible in score.json rather than silently absent.
    _, summary = exacto_report.parse_openbench_log(OPENBENCH_LOG)
    exacto_report.merge_endpoint_meta(summary, None)
    assert summary["endpoint"] == {}


def test_stop_ratio_floor_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVAL_MIN_STOP_RATIO", raising=False)
    exacto_report._enforce_stop_ratio({"stop_ratio": 0.1})


def test_stop_ratio_floor_rejects_a_truncated_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVAL_MIN_STOP_RATIO", "0.9")
    with pytest.raises(SystemExit):
        exacto_report._enforce_stop_ratio(
            {"stop_ratio": 0.5, "finish_stop": 50, "finish_length": 50}
        )


def test_stop_ratio_floor_allows_a_clean_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVAL_MIN_STOP_RATIO", "0.9")
    exacto_report._enforce_stop_ratio(
        {"stop_ratio": 0.99, "finish_stop": 99, "finish_length": 1}
    )


def test_stop_ratio_floor_skipped_when_ratio_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # tau2 reports no stop_ratio; a floor set for GPQA must not fail it.
    monkeypatch.setenv("EVAL_MIN_STOP_RATIO", "0.9")
    exacto_report._enforce_stop_ratio({"stop_ratio": None})


def _row(value: int, answer: str, target: str) -> dict[str, Any]:
    return {
        "value": value,
        "answer": answer,
        "explanation": f"Extracted '{answer}' from response, target was '{target}'",
    }


def test_target_letters_read_from_both_correct_and_wrong_rows() -> None:
    rows = [_row(1, "B", "B"), _row(0, "A", "C"), _row(1, "D", "D")]
    assert exacto_report.target_letter_counts(rows) == {"B": 1, "C": 1, "D": 1}


def test_position_bias_detected_when_one_letter_dominates() -> None:
    # The real openbench gpqa_diamond shape: seed(0) before every shuffle, so
    # every question's correct answer is the same letter.
    rows = [_row(1, "B", "B") for _ in range(154)] + [
        _row(0, "A", "B") for _ in range(44)
    ]
    warning = exacto_report.answer_position_warning(rows)
    assert warning is not None
    assert "198" in warning and "'B'" in warning
    assert "position-contaminated" in warning


def test_no_position_warning_when_answers_are_spread() -> None:
    rows = [_row(1, L, L) for L in "ABCD" for _ in range(25)]
    assert exacto_report.answer_position_warning(rows) is None


def test_position_check_skipped_on_a_tiny_smoke_run() -> None:
    # A 4-question smoke run is degenerate by construction; do not cry wolf.
    rows = [_row(1, "B", "B") for _ in range(4)]
    assert exacto_report.answer_position_warning(rows) is None


def test_contamination_reaches_score_json_not_just_stdout(
    tmp_path: Path,
) -> None:
    # Regression: the counts were set after score.json was dumped, so the
    # warning printed but every downstream reader saw a clean run.
    rows = [_row(1, "B", "B") for _ in range(154)] + [
        _row(0, "A", "B") for _ in range(44)
    ]
    exacto_report.write_outputs(
        str(tmp_path), "gpqa_diamond", rows, {"accuracy": 0.7778}, "EXACTO_GPQA"
    )
    written = json.loads((tmp_path / "score.json").read_text())
    assert written["target_letter_counts"] == {"B": 198}


# Shaped after the CI run where every request 404'd: openbench interrupts, so
# it reports no counts at all and no score.
INTERRUPTED_LOG: dict[str, Any] = {
    "status": "error",
    "eval": {"task": "gpqa_diamond", "dataset": {}, "config": {}},
    "plan": {"config": {}},
    "results": {"scores": []},
    "reductions": [],
    "samples": [],
}


def test_zero_sample_run_fails_instead_of_reporting_none(
    tmp_path: Path,
) -> None:
    # Regression: the error budget cannot fire without counts, so this run
    # printed "none" and exited 0 while nothing had been measured.
    rows, summary = exacto_report.parse_openbench_log(INTERRUPTED_LOG)
    assert summary["accuracy"] is None
    with pytest.raises(SystemExit):
        exacto_report.write_outputs(
            str(tmp_path), "gpqa_diamond", rows, summary, "EXACTO_GPQA"
        )


def test_zero_sample_failure_still_leaves_artifacts(tmp_path: Path) -> None:
    # The score file has to survive the failure, or there is nothing to debug.
    rows, summary = exacto_report.parse_openbench_log(INTERRUPTED_LOG)
    with pytest.raises(SystemExit):
        exacto_report.write_outputs(
            str(tmp_path), "gpqa_diamond", rows, summary, "EXACTO_GPQA"
        )
    assert (tmp_path / "score.json").exists()


def test_a_scored_run_is_not_treated_as_empty() -> None:
    _, summary = exacto_report.parse_openbench_log(OPENBENCH_LOG)
    exacto_report._enforce_no_samples(summary)
