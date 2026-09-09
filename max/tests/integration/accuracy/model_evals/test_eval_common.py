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
"""Tests for the shared eval scaffolding."""

import json
from pathlib import Path
from typing import Any

import eval_common
import pytest

# An endpoint that rejects every request: the score is 0.0 not because the model
# was wrong but because nothing was measured.
ALL_ERRORED = {"accuracy": 0.0, "correct": 0, "total": 120, "errors": 120}

# A real AIME25 run whose only failures were gateway timeouts on the two
# longest-generating problems. Well inside the budget; must still score.
GATEWAY_TIMEOUTS = {
    "accuracy": 0.9021,
    "correct": 433,
    "total": 480,
    "errors": 3,
}


def test_error_budget_rejects_an_all_errored_run() -> None:
    with pytest.raises(SystemExit):
        eval_common.enforce_error_budget(ALL_ERRORED)


def test_error_budget_allows_incidental_errors() -> None:
    eval_common.enforce_error_budget(GATEWAY_TIMEOUTS)


@pytest.mark.parametrize(
    ("errors", "rejected"),
    [(9, False), (10, False), (11, True)],
)
def test_error_budget_boundary(errors: int, rejected: bool) -> None:
    summary = {"total": 100, "errors": errors}
    if rejected:
        with pytest.raises(SystemExit):
            eval_common.enforce_error_budget(summary)
    else:
        eval_common.enforce_error_budget(summary)


def test_error_budget_reads_the_judged_eval_key() -> None:
    """AA-Omniscience names its error count ``errored``, not ``errors``."""
    with pytest.raises(SystemExit):
        eval_common.enforce_error_budget({"total": 600, "errored": 600})


@pytest.mark.parametrize(
    "summary",
    [
        {"total": 600},  # no error count reported
        {"accuracy": 0.4},  # no total either (SciCode-style)
        {"total": 0, "errors": 0},  # nothing submitted
    ],
)
def test_error_budget_skips_summaries_it_cannot_judge(
    summary: dict[str, Any],
) -> None:
    eval_common.enforce_error_budget(summary)


def test_dump_score_writes_before_rejecting(tmp_path: Path) -> None:
    """A rejected run still leaves score.json behind to diagnose from."""
    out = tmp_path / "results"
    with pytest.raises(SystemExit):
        eval_common.dump_score(str(out), ALL_ERRORED)
    written = json.loads((out / "score.json").read_text())
    assert written == ALL_ERRORED


def test_gated_dataset_denial_fails_the_step() -> None:
    """A lane that cannot load its dataset must not report success."""

    def denied() -> list[dict[str, Any]]:
        raise RuntimeError("403 Client Error: gated repo, enable access")

    with pytest.raises(SystemExit) as excinfo:
        eval_common.load_gated(denied, label="GPQA", dataset_id="some/dataset")
    assert excinfo.value.code != 0


def test_gated_loader_passes_rows_through() -> None:
    rows = [{"q": 1}]
    assert (
        eval_common.load_gated(
            lambda: rows, label="GPQA", dataset_id="some/dataset"
        )
        is rows
    )


def test_gated_loader_propagates_unrelated_errors() -> None:
    """Only access denials are translated; real bugs keep their traceback."""

    def broken() -> list[dict[str, Any]]:
        raise ValueError("malformed row")

    with pytest.raises(ValueError):
        eval_common.load_gated(broken, label="GPQA", dataset_id="some/dataset")


# A run where a quarter of completed responses hit the token cap. Below any
# reasonable floor, but must pass while the gate is off (the default).
LOW_STOP = {
    "total": 100,
    "errors": 0,
    "finish_stop": 75,
    "finish_length": 25,
    "stop_ratio": 0.75,
}


def test_finish_stats_counts_and_ratio() -> None:
    rows = (
        [{"finish_reason": "stop"}] * 3
        + [{"finish_reason": "length"}]
        + [{"error": "boom"}]  # errored rows carry no finish reason
    )
    s = eval_common.finish_stats(rows)
    # These rows carry no ``correct`` key, so nothing counts as correct.
    assert s == {
        "finish_stop": 3,
        "finish_length": 1,
        "stop_ratio": 0.75,
        "correct_given_stop": 0,
        "accuracy_given_stop": 0.0,
        "correct_given_length": 0,
        "accuracy_given_length": 0.0,
    }


def test_finish_stats_with_nothing_completed() -> None:
    assert eval_common.finish_stats([{"error": "x"}])["stop_ratio"] is None


def test_stop_ratio_gate_is_off_by_default() -> None:
    eval_common.enforce_stop_ratio(LOW_STOP)


def test_stop_ratio_gate_rejects_below_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eval_common, "MIN_STOP_RATIO", 0.9)
    with pytest.raises(SystemExit):
        eval_common.enforce_stop_ratio(LOW_STOP)


def test_stop_ratio_gate_passes_at_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eval_common, "MIN_STOP_RATIO", 0.75)
    eval_common.enforce_stop_ratio(LOW_STOP)


def test_stop_ratio_gate_skips_summaries_without_the_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eval_common, "MIN_STOP_RATIO", 0.9)
    eval_common.enforce_stop_ratio({"total": 100, "errors": 0})
    eval_common.enforce_stop_ratio({"stop_ratio": None})


def test_exact_match_score_reports_stop_ratio() -> None:
    rows = [
        {"correct": True, "finish_reason": "stop", "completion_tokens": 10},
        {"correct": False, "finish_reason": "length", "completion_tokens": 99},
    ]
    s = eval_common.exact_match_score(rows, total=2, errors=0)
    assert (s["finish_stop"], s["finish_length"], s["stop_ratio"]) == (
        1,
        1,
        0.5,
    )


def _completed_accuracy(stats: dict[str, object]) -> float:
    """Recombines the two conditional accuracies into completed-row accuracy."""
    stop_ratio = stats["stop_ratio"]
    assert isinstance(stop_ratio, float)
    given_stop = stats["accuracy_given_stop"] or 0.0
    given_length = stats["accuracy_given_length"] or 0.0
    assert isinstance(given_stop, float) and isinstance(given_length, float)
    return given_stop * stop_ratio + given_length * (1 - stop_ratio)


def test_finish_stats_reports_the_accuracy_decomposition() -> None:
    """Completed-row accuracy splits by finish reason, weighted by stop_ratio.

    Reporting the factors rather than their sum keeps a shift in how often the
    model fails to terminate from reading as a change in answer quality.
    """
    results = [
        {"finish_reason": "stop", "correct": True},
        {"finish_reason": "stop", "correct": True},
        {"finish_reason": "stop", "correct": True},
        {"finish_reason": "stop", "correct": False},
        {"finish_reason": "length", "correct": False},
    ]
    stats = eval_common.finish_stats(results)
    assert stats["finish_stop"] == 4
    assert stats["finish_length"] == 1
    assert stats["stop_ratio"] == 0.8
    assert stats["correct_given_stop"] == 3
    assert stats["accuracy_given_stop"] == 0.75
    assert stats["correct_given_length"] == 0
    assert stats["accuracy_given_length"] == 0.0

    # Every row here completed, so completed-row accuracy is overall accuracy.
    overall = sum(1 for r in results if r["correct"]) / len(results)
    assert overall == pytest.approx(_completed_accuracy(stats))


def test_finish_stats_decomposition_excludes_errored_rows() -> None:
    """The product is completed-row accuracy, above accuracy over everything.

    :func:`eval_common.exact_match_score` keeps errored rows in the denominator
    of the headline ``accuracy`` so dropping the hardest problems cannot inflate
    it. Those rows carry no finish reason, so they are outside this product --
    the two agree only when nothing errored.
    """
    results = [
        {"finish_reason": "stop", "correct": True},
        {"finish_reason": "length", "correct": False},
        {"error": "boom", "correct": False},  # no finish reason
    ]
    stats = eval_common.finish_stats(results)
    # 1 correct of the 2 completed.
    assert _completed_accuracy(stats) == pytest.approx(0.5)

    summary = eval_common.exact_match_score(results, total=3, errors=1)
    assert summary["accuracy"] == pytest.approx(1 / 3)
    # The gap is exactly the completion factor the summary already reports.
    completion = summary["answered"] / summary["total"]
    assert summary["accuracy"] == pytest.approx(
        _completed_accuracy(stats) * completion
    )


def test_finish_stats_decomposition_survives_a_correct_truncated_row() -> None:
    """A ``length`` row that grades correct keeps the identity exact.

    Nothing makes truncation imply a wrong grade: the graders parse whatever
    content came back without consulting ``finish_reason``, and ``strip_think``
    only removes a ``<think>`` span that was actually closed. A response cut off
    mid-reasoning therefore reaches the parser as raw reasoning text, out of
    which ``gpqa_eval`` can read a standalone option letter and ``aime_eval``
    the last integer. Weighting both conditional accuracies keeps the
    decomposition exact when that happens, where a stop-only product would
    silently drop the row.
    """
    results = [
        {"finish_reason": "stop", "correct": True},
        {"finish_reason": "stop", "correct": False},
        {"finish_reason": "length", "correct": True},  # lucky parse
        {"finish_reason": "length", "correct": False},
    ]
    stats = eval_common.finish_stats(results)
    assert stats["accuracy_given_stop"] == 0.5
    assert stats["correct_given_length"] == 1
    assert stats["accuracy_given_length"] == 0.5

    overall = sum(1 for r in results if r["correct"]) / len(results)
    assert overall == pytest.approx(_completed_accuracy(stats))
    # The stop-side factor alone would have understated it by the dropped row.
    assert stats["accuracy_given_stop"] * stats["stop_ratio"] == pytest.approx(
        0.25
    )


def test_finish_stats_counts_recombine_exactly_despite_rounding() -> None:
    """The counts, unlike the reported ratios, carry no rounding error.

    ``accuracy_given_stop`` and ``stop_ratio`` are each rounded to four places,
    so a caller needing the identity to the last bit adds the counts instead.
    """
    results = [
        {"finish_reason": "stop", "correct": i % 3 == 0} for i in range(7)
    ]
    results += [{"finish_reason": "length", "correct": False}] * 2
    stats = eval_common.finish_stats(results)

    assert stats["accuracy_given_stop"] == 0.4286  # 3/7, rounded
    correct_completed = (
        stats["correct_given_stop"] + stats["correct_given_length"]
    )
    completed = stats["finish_stop"] + stats["finish_length"]
    assert correct_completed / completed == pytest.approx(3 / 9)
    # The rounded ratios only reconstruct it approximately.
    assert _completed_accuracy(stats) == pytest.approx(3 / 9, abs=1e-4)


def test_finish_stats_takes_an_eval_specific_correctness_reader() -> None:
    """Evals that grade into another field pass their own reader.

    AA-Omniscience grades into a four-way ``verdict`` and never sets
    ``correct``; without this its stopped-row metrics would read a missing key
    and report 0 on every run.
    """
    results = [
        {"finish_reason": "stop", "verdict": "CORRECT"},
        {"finish_reason": "stop", "verdict": "INCORRECT"},
        {"finish_reason": "stop", "verdict": "NOT_ATTEMPTED"},
        {"finish_reason": "length", "verdict": "NOT_ATTEMPTED"},
    ]
    stats = eval_common.finish_stats(
        results, is_correct=lambda r: r["verdict"] == "CORRECT"
    )
    assert stats["correct_given_stop"] == 1
    assert stats["accuracy_given_stop"] == 0.3333  # 1 of 3 stopped, rounded

    # The default reader finds no ``correct`` key on these rows at all.
    assert eval_common.finish_stats(results)["correct_given_stop"] == 0


def test_finish_stats_accuracy_given_stop_is_none_without_stopped_rows() -> (
    None
):
    """No stopped rows means the ratio has no denominator, not a zero."""
    stats = eval_common.finish_stats(
        [{"finish_reason": "length", "correct": False}]
    )
    assert stats["accuracy_given_stop"] is None
    assert stats["stop_ratio"] == 0.0
