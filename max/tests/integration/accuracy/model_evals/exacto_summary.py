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
"""Renders the OpenRouter AutoExacto job summary from per-dataset score.json.

Reports each score against the peer band rather than a fixed threshold, because
OpenRouter deranks on median + median-absolute-deviation across the providers
serving a given model: what matters is whether we cluster with peers.

Stdlib-only, like :mod:`collect_scores` — the CI summary step runs it with a
bare ``python3``, no bazel.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, TextIO

#: ``(dataset key, display name, metric, score.json path arg)``.
DATASETS = [
    ("gpqa_diamond", "GPQA-Diamond", "accuracy (MCQ)", "gpqa"),
    ("taubench_airline", "tau2-bench airline", "mean reward", "taubench"),
]


def load(path: str | None) -> dict[str, Any] | None:
    """Loads a ``score.json``, returning ``None`` when it is absent or invalid.

    A skipped or crashed dataset leaves no file, which is reported as a dash
    rather than failing the summary step.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def notes(score: dict[str, Any], dataset: str) -> str:
    """Lists factual caveats for one result, or a dash when there are none.

    Deliberately not a pass/fail verdict. OpenRouter deranks on a median and
    median absolute deviation across a model's providers, which we cannot
    compute, and the GPQA task's options are not reshuffled per question, so a
    verdict here would imply a precision the measurement does not have.
    """
    flags = []
    if score.get("limit"):
        flags.append("partial (question subset)")
    if dataset == "taubench_airline":
        # The runner counts the split, so this needs no knowledge of the
        # dataset size and stays right if upstream changes it.
        tasks = score.get("num_tasks")
        full = score.get("full_task_count")
        if isinstance(tasks, int) and isinstance(full, int) and tasks != full:
            flags.append(f"partial ({tasks}/{full} tasks)")
        # The reference simulator is recorded by the runner rather than named
        # here, so this needs no edit when OpenRouter changes theirs.
        reference_sim = score.get("reference_user_llm")
        if reference_sim and score.get("user_llm") != reference_sim:
            flags.append("substitute user simulator")
    counts = score.get("target_letter_counts")
    if isinstance(counts, dict) and counts:
        total = sum(counts.values())
        top = max(counts.values())
        if total and top / total > 0.5:
            flags.append("answer-position contaminated")
    return "; ".join(flags) if flags else "—"


def render(scores: dict[str, dict[str, Any] | None], out: TextIO) -> None:
    """Writes the Markdown summary for the loaded per-dataset scores."""
    out.write("## OpenRouter AutoExacto gate\n\n")
    model = os.environ.get("EXACTO_MODEL") or ""
    backend = os.environ.get("BACKEND") or ""
    if model:
        out.write(f"**Model:** `{model}`\n\n")
    if backend:
        out.write(f"**Endpoint under test:** {backend}\n\n")

    reference_sources: set[str] = set()
    out.write(
        "| Dataset | Metric | Score | ±Stderr | Published providers* | n | Notes |\n"
        "|---------|--------|-------|---------|----------------------|---|-------|\n"
    )
    for key, name, metric, arg in DATASETS:
        score = scores.get(arg)
        if score is None:
            out.write(f"| {name} | {metric} | — | — | — | — | not run |\n")
            continue
        accuracy = score.get("accuracy")
        stderr = score.get("stderr")
        reference = score.get("reference") or {}
        acc_text = (
            f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else "—"
        )
        err_text = f"{stderr:.4f}" if isinstance(stderr, (int, float)) else "—"
        low, high = reference.get("low"), reference.get("high")
        ref_text = (
            f"{low:.3f}-{high:.3f}"
            if isinstance(low, (int, float)) and isinstance(high, (int, float))
            else "—"
        )
        source = reference.get("source")
        if source:
            reference_sources.add(source)
        out.write(
            f"| {name} | {metric} | {acc_text} | {err_text} | {ref_text} | "
            f"{score.get('total') or '—'} | {notes(score, key)} |\n"
        )

    if reference_sources:
        out.write(
            "\n> \\* Published-provider range supplied by the caller, from "
            + ", ".join(sorted(reference_sources))
            + ". Provider spread is per-model.\n\n"
        )
    else:
        out.write(
            "\n> \\* No published-provider range was supplied, so none is "
            "shown. Provider spread is per-model; pass one with "
            "--reference-range if you have it.\n\n"
        )
    out.write(
        "> These scores are reported, not graded. OpenRouter deranks on a "
        "median and median absolute deviation across the providers serving a "
        "given model, which cannot be computed without every provider's score, "
        "so the published-providers column is context rather than a threshold. "
        "Read the Notes column before comparing anything.\n\n"
    )

    _render_provenance(scores, out)

    results_url = os.environ.get("RESULTS_URL")
    if results_url:
        out.write(
            f"### Artifacts\n\n- [autoexacto-eval-results]({results_url})\n\n"
        )


def _render_provenance(
    scores: dict[str, dict[str, Any] | None], out: TextIO
) -> None:
    """Writes the knobs that decide whether a score is comparable at all."""
    gpqa = scores.get("gpqa")
    tau = scores.get("taubench")
    if not gpqa and not tau:
        return
    out.write("### Comparability\n\n")
    if gpqa:
        out.write(
            f"- GPQA: dataset `{gpqa.get('dataset_name')}`, "
            f"temperature {gpqa.get('temperature')}, "
            f"epochs {gpqa.get('epochs')} "
            f"(reducer {gpqa.get('epochs_reducer')}), "
            f"stop_ratio {gpqa.get('stop_ratio')}\n"
        )
    if tau:
        out.write(
            f"- tau2: user simulator `{tau.get('user_llm')}`, "
            f"max_steps {tau.get('max_steps')}, "
            f"{tau.get('num_tasks')} tasks x {tau.get('num_trials')} trials, "
            f"{tau.get('hit_step_cap')} hit the step cap, "
            f"harness `{str(tau.get('harness_commit'))[:12]}`\n"
        )
    out.write("\n")


def main(argv: list[str] | None = None) -> None:
    """Renders the summary to ``$GITHUB_STEP_SUMMARY``, or stdout locally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpqa", help="Path to the GPQA score.json.")
    parser.add_argument("--taubench", help="Path to the tau2 score.json.")
    args = parser.parse_args(argv)

    scores = {"gpqa": load(args.gpqa), "taubench": load(args.taubench)}
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            render(scores, f)
    else:
        import sys

        render(scores, sys.stdout)


if __name__ == "__main__":
    main()
