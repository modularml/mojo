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
"""Tests for the AutoExacto job-summary renderer."""

import io
import json
from pathlib import Path

import exacto_summary
import pytest


def test_notes_are_empty_for_a_clean_full_run() -> None:
    assert exacto_summary.notes({"limit": None}, "gpqa_diamond") == "—"


def test_notes_flag_a_question_subset() -> None:
    assert "partial" in exacto_summary.notes({"limit": 10}, "gpqa_diamond")


def test_notes_flag_a_tau2_task_subset_and_simulator() -> None:
    out = exacto_summary.notes(
        {
            "num_tasks": 2,
            "full_task_count": 50,
            "user_llm": "openai/served-name",
            "reference_user_llm": "gemini/gemini-2.5-flash",
        },
        "taubench_airline",
    )
    assert "2/50 tasks" in out
    assert "substitute user simulator" in out


def test_notes_accept_the_reference_simulator_on_a_full_task_set() -> None:
    out = exacto_summary.notes(
        {
            "num_tasks": 50,
            "full_task_count": 50,
            "user_llm": "gemini/gemini-2.5-flash",
            "reference_user_llm": "gemini/gemini-2.5-flash",
        },
        "taubench_airline",
    )
    assert out == "—"


def test_notes_cannot_judge_the_simulator_without_a_recorded_reference() -> (
    None
):
    # No reference recorded means we do not know what the reference was, so
    # do not invent a verdict about it.
    out = exacto_summary.notes(
        {"num_tasks": 50, "full_task_count": 50, "user_llm": "openai/whatever"},
        "taubench_airline",
    )
    assert "substitute user simulator" not in out


def test_notes_flag_answer_position_contamination() -> None:
    # The real openbench gpqa_diamond shape: every target is the same letter.
    out = exacto_summary.notes(
        {"target_letter_counts": {"B": 198}}, "gpqa_diamond"
    )
    assert "answer-position contaminated" in out


def test_notes_ignore_a_healthy_letter_spread() -> None:
    out = exacto_summary.notes(
        {"target_letter_counts": {"A": 50, "B": 50, "C": 50, "D": 48}},
        "gpqa_diamond",
    )
    assert out == "—"


def test_load_missing_file_returns_none() -> None:
    assert exacto_summary.load("/nonexistent/score.json") is None
    assert exacto_summary.load(None) is None


def test_load_invalid_json_returns_none(tmp_path: Path) -> None:
    bad = tmp_path / "score.json"
    bad.write_text("{not json")
    assert exacto_summary.load(str(bad)) is None


def test_load_non_object_json_returns_none(tmp_path: Path) -> None:
    listy = tmp_path / "score.json"
    listy.write_text("[1, 2, 3]")
    assert exacto_summary.load(str(listy)) is None


def test_render_marks_a_full_gpqa_run_in_band() -> None:
    out = io.StringIO()
    exacto_summary.render(
        {
            "gpqa": {
                "reference": {
                    "low": 0.842,
                    "high": 0.908,
                    "source": "some-model",
                },
                "accuracy": 0.88,
                "stderr": 0.02,
                "total": 1980,
                "limit": None,
                "dataset_name": "nmayorga7/gpqa_diamond",
                "temperature": 0.5,
                "epochs": 10,
                "epochs_reducer": ["mean"],
                "stop_ratio": 0.99,
            },
            "taubench": None,
        },
        out,
    )
    text = out.getvalue()
    assert "0.8800" in text
    assert "0.8800" in text
    assert "0.842-0.908" in text
    assert "some-model" in text
    assert "nmayorga7/gpqa_diamond" in text
    # The absent dataset still gets a row rather than vanishing.
    assert "tau2-bench airline" in text


def test_render_flags_a_substitute_user_simulator() -> None:
    out = io.StringIO()
    exacto_summary.render(
        {
            "gpqa": None,
            "taubench": {
                "accuracy": 0.76,
                "stderr": 0.06,
                "total": 50,
                "num_tasks": 50,
                "full_task_count": 50,
                "num_trials": 1,
                "user_llm": "openai/some-local-model",
                "reference_user_llm": "gemini/gemini-2.5-flash",
                "max_steps": 200,
                "hit_step_cap": 0,
                "harness_commit": "864350a8971a8f8ee9e7b8472e2edc380a806b0c",
            },
        },
        out,
    )
    text = out.getvalue()
    assert "0.7600" in text
    assert "substitute user simulator" in text
    # Provenance has to surface the simulator, since it decides comparability.
    assert "openai/some-local-model" in text


def test_render_treats_a_task_subset_as_partial() -> None:
    out = io.StringIO()
    exacto_summary.render(
        {
            "gpqa": None,
            "taubench": {
                "accuracy": 0.80,
                "total": 2,
                "num_tasks": 2,
                "full_task_count": 50,
                "num_trials": 1,
                "user_llm": "gemini/gemini-2.5-flash",
                "max_steps": 200,
                "hit_step_cap": 0,
                "harness_commit": "864350a",
            },
        },
        out,
    )
    assert "partial" in out.getvalue()


def test_render_with_no_scores_still_produces_a_table() -> None:
    out = io.StringIO()
    exacto_summary.render({"gpqa": None, "taubench": None}, out)
    text = out.getvalue()
    assert "OpenRouter AutoExacto gate" in text
    assert "GPQA-Diamond" in text
    # Nothing ran, so there is nothing to claim about comparability.
    assert "### Comparability" not in text


def test_main_writes_to_the_github_step_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = tmp_path / "score.json"
    score.write_text(
        json.dumps({"accuracy": 0.95, "total": 1980, "limit": None})
    )
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    exacto_summary.main(["--gpqa", str(score)])
    assert "0.9500" in summary.read_text()


def test_notes_cannot_judge_a_subset_without_the_split_size() -> None:
    # Nothing hardcodes the dataset size, so with no recorded split count there
    # is no basis to call a run partial.
    out = exacto_summary.notes({"num_tasks": 2}, "taubench_airline")
    assert "partial" not in out
