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
"""Normalizes OpenRouter AutoExacto harness output into this suite's conventions.

OpenRouter gates provider endpoints on its AutoExacto benchmark — GPQA-Diamond
plus tau2-bench airline — and deranks endpoints that come out as statistical
outliers. Passing it is an exit criterion for a model bring-up, so we need to
measure ourselves the way OpenRouter measures us, which means running *their*
harnesses rather than our own equivalents: ``model_evals/gpqa_eval.py`` scores
the same dataset but with MiniMax's vendor prompt, so its numbers are not
comparable to the AutoExacto leaderboard.

Those harnesses write their own log formats, so this module translates them into
the ``results.jsonl`` / ``score.json`` / ``$GITHUB_ENV`` shape the rest of
``model_evals`` uses (see :mod:`eval_common`), which is what lets the existing
workflow job-summary machinery pick the scores up unchanged.

Parsing is pure — it takes an already-loaded log and returns rows plus a
summary — so it unit-tests without a harness, a server or a network.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from typing import Any

#: Datasets this module can normalize.
DATASETS = ("gpqa_diamond", "taubench_airline")

#: Share of questions that may share one correct-answer letter before the score
#: stops measuring knowledge. Four options, so chance is 0.25; anything past
#: half means a model biased toward that letter scores well for free.
MAX_TARGET_LETTER_SHARE = 0.5


def target_letter_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Counts how often each letter is the correct answer.

    A correctly-scored row's ``answer`` is the target; a wrong row names the
    target in its scorer explanation.
    """
    counts: dict[str, int] = {}
    for row in rows:
        target: str | None = None
        if row.get("value") == 1:
            answer = row.get("answer")
            target = answer if isinstance(answer, str) else None
        else:
            match = re.search(
                r"target was '([A-Z])'", str(row.get("explanation") or "")
            )
            target = match.group(1) if match else None
        if target:
            counts[target] = counts.get(target, 0) + 1
    return counts


def answer_position_warning(rows: list[dict[str, Any]]) -> str | None:
    """Reports when the correct answer sits in the same slot too often.

    openbench's gpqa_diamond calls ``random.seed(0)`` inside its per-record
    mapper before shuffling the options, so every question gets the same
    permutation and the correct answer is always the same letter. Measured on
    the full 198-question set: all 198 targets came back 'B'. The score then
    reflects the model's position bias as much as its knowledge, and a model
    that always answers that letter scores near 100%. Detected here rather than
    corrected, because correcting it would diverge from the methodology this
    runner exists to reproduce.

    Returns:
        A warning string, or ``None`` when the answer positions look spread.
    """
    counts = target_letter_counts(rows)
    total = sum(counts.values())
    if total < 10:
        return None
    letter, top = max(counts.items(), key=lambda kv: kv[1])
    share = top / total
    if share <= MAX_TARGET_LETTER_SHARE:
        return None
    detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return (
        f"::warning::{top}/{total} questions ({share:.0%}) have '{letter}' as the "
        f"correct answer ({detail}). The options are not being reshuffled per "
        f"question, so this score reflects the model's bias toward one answer "
        f"position as much as its knowledge, and a model that always answers "
        f"'{letter}' would score near 100%. Treat it as position-contaminated."
    )


def _metric(score: dict[str, Any], name: str) -> float | None:
    """Reads ``score.metrics.<name>.value``, tolerating a missing metric."""
    metric = (score.get("metrics") or {}).get(name)
    if not isinstance(metric, dict):
        return None
    value = metric.get("value")
    return float(value) if isinstance(value, (int, float)) else None


def parse_openbench_log(
    log: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Converts an openbench (Inspect) eval log into rows plus a summary.

    Reads the ``--log-format json`` payload openbench writes for
    ``bench eval gpqa_diamond``. Per-question rows come from ``reductions``,
    which holds one entry per question *after* the epoch reducer has averaged
    its epochs together — that reduced value is what the reported accuracy is
    computed from, so it is the honest per-question record. Token and
    finish-reason stats come from the unreduced ``samples`` list, since those
    are per request.

    Args:
        log: Parsed openbench JSON log.

    Returns:
        A ``(rows, summary)`` tuple. ``summary`` carries ``total`` and
        ``errors`` so :func:`eval_common.dump_score` can enforce the error
        budget, and ``stop_ratio`` so :func:`_enforce_stop_ratio` can
        enforce a truncation floor.
    """
    scores = (log.get("results") or {}).get("scores") or []
    score = scores[0] if scores else {}
    results = log.get("results") or {}

    rows: list[dict[str, Any]] = []
    for reduction in log.get("reductions") or []:
        for sample in reduction.get("samples") or []:
            rows.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "value": sample.get("value"),
                    "answer": sample.get("answer"),
                    "explanation": sample.get("explanation"),
                    "scorer": reduction.get("scorer"),
                }
            )

    output_tokens: list[int] = []
    finish_stop = 0
    finish_length = 0
    for sample in log.get("samples") or []:
        output = sample.get("output") or {}
        usage = output.get("usage") or {}
        tokens = usage.get("output_tokens")
        if isinstance(tokens, int):
            output_tokens.append(tokens)
        for choice in output.get("choices") or []:
            # Inspect normalizes the OpenAI finish_reason onto stop_reason.
            if choice.get("stop_reason") == "length":
                finish_length += 1
            else:
                finish_stop += 1

    completed = results.get("completed_samples")
    total = results.get("total_samples")
    # A sample that never produced a scoreable response is an error for budget
    # purposes: openbench counts it in total_samples but not completed_samples,
    # and an endpoint rejecting every request would otherwise score 0.0 and
    # report success.
    errors = (
        total - completed
        if isinstance(total, int) and isinstance(completed, int)
        else None
    )
    unscored = score.get("unscored_samples")
    if isinstance(errors, int) and isinstance(unscored, int):
        errors = max(errors, unscored)

    graded = finish_stop + finish_length
    eval_meta = log.get("eval") or {}
    config = eval_meta.get("config") or {}
    dataset = eval_meta.get("dataset") or {}

    summary: dict[str, Any] = {
        "accuracy": _metric(score, "accuracy"),
        "stderr": _metric(score, "stderr"),
        "std": _metric(score, "std"),
        "scorer": score.get("name"),
        "scored_samples": score.get("scored_samples"),
        "unscored_samples": unscored,
        "total": total,
        "completed": completed,
        "errors": errors,
        "mean_output_tokens": (
            statistics.fmean(output_tokens) if output_tokens else 0.0
        ),
        "p50_output_tokens": (
            statistics.median(output_tokens) if output_tokens else 0.0
        ),
        "finish_stop": finish_stop,
        "finish_length": finish_length,
        "stop_ratio": (finish_stop / graded) if graded else None,
        # Provenance: the knobs that make a score comparable to OpenRouter's.
        "harness": "openbench",
        "harness_status": log.get("status"),
        "task": eval_meta.get("task"),
        "dataset_name": dataset.get("name"),
        "dataset_samples": dataset.get("samples"),
        "model": eval_meta.get("model"),
        "base_url": eval_meta.get("model_base_url"),
        "epochs": config.get("epochs"),
        "epochs_reducer": config.get("epochs_reducer"),
        "limit": config.get("limit"),
        "temperature": ((log.get("plan") or {}).get("config") or {}).get(
            "temperature"
        ),
        "inspect_version": (eval_meta.get("packages") or {}).get("inspect_ai"),
    }
    return rows, summary


#: ``termination_reason`` values that mean the simulation never got a fair
#: chance — an API or harness failure, not the agent losing. ``max_steps`` is
#: deliberately absent: an agent that loops until the step cap has genuinely
#: failed the task, and OpenRouter scores that 0 rather than discarding it.
TAU2_ERROR_TERMINATIONS = frozenset(
    {"too_many_errors", "agent_error", "user_error"}
)


def parse_tau2_log(
    log: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Converts a tau2-bench results file into rows plus a summary.

    Reads the JSON ``tau2 run`` writes. The headline metric is the mean binary
    reward over every simulation — ``num_tasks x num_trials`` of them — where
    each reward is the conjunction of the database-state check and the
    communication checks, and a run that blows the step cap scores 0.

    Args:
        log: Parsed tau2 results JSON.

    Returns:
        A ``(rows, summary)`` tuple in the same shape as
        :func:`parse_openbench_log`.
    """
    info = log.get("info") or {}
    simulations = log.get("simulations") or []

    rows: list[dict[str, Any]] = []
    rewards: list[float] = []
    errors = 0
    hit_step_cap = 0
    for sim in simulations:
        reward_info = sim.get("reward_info") or {}
        reward = reward_info.get("reward")
        termination = sim.get("termination_reason")
        if isinstance(reward, (int, float)):
            rewards.append(float(reward))
        if termination in TAU2_ERROR_TERMINATIONS:
            errors += 1
        if termination == "max_steps":
            hit_step_cap += 1
        db_check = reward_info.get("db_check") or {}
        rows.append(
            {
                "task_id": sim.get("task_id"),
                "trial": sim.get("trial"),
                "reward": reward,
                "db_match": db_check.get("db_match"),
                "termination_reason": termination,
                "duration": sim.get("duration"),
                "agent_cost": sim.get("agent_cost"),
            }
        )

    accuracy = statistics.fmean(rewards) if rewards else None
    # Rewards are binary, so the sample stdev over them is the usual
    # proportion spread; report the standard error so a score can be compared
    # against the peer band without over-reading a small-n difference.
    if len(rewards) > 1:
        std: float | None = statistics.stdev(rewards)
        stderr: float | None = std / (len(rewards) ** 0.5)
    else:
        std = stderr = None

    agent_info = info.get("agent_info") or {}
    user_info = info.get("user_info") or {}
    env_info = info.get("environment_info") or {}

    summary: dict[str, Any] = {
        "accuracy": accuracy,
        "stderr": stderr,
        "std": std,
        "scorer": "tau2_reward",
        "scored_samples": len(rewards),
        "unscored_samples": len(simulations) - len(rewards),
        "total": len(simulations),
        "completed": len(rewards),
        "errors": errors,
        "hit_step_cap": hit_step_cap,
        # tau2 records per-simulation cost, not token counts, so the token
        # metrics the other evals report are not available here.
        "mean_output_tokens": 0.0,
        "p50_output_tokens": 0.0,
        "stop_ratio": None,
        # Provenance: the knobs that make a score comparable to OpenRouter's.
        "harness": "tau2-bench-verified",
        "domain": env_info.get("domain_name"),
        "num_tasks": len(log.get("tasks") or []),
        "num_trials": info.get("num_trials"),
        "max_steps": info.get("max_steps"),
        "max_errors": info.get("max_errors"),
        "seed": info.get("seed"),
        "agent_llm": agent_info.get("llm"),
        "agent_llm_args": agent_info.get("llm_args"),
        "user_llm": user_info.get("llm"),
        "user_llm_args": user_info.get("llm_args"),
        "harness_commit": info.get("git_commit"),
    }
    return rows, summary


def load_json(path: str) -> dict[str, Any]:
    """Loads a JSON harness log from ``path``."""
    with open(path) as f:
        return json.load(f)


def load_openbench_log(path: str) -> dict[str, Any]:
    """Loads an openbench JSON eval log from ``path``."""
    with open(path) as f:
        return json.load(f)


def format_score_line(dataset: str, summary: dict[str, Any]) -> str:
    """Renders the score, with a reference range only if the caller gave one.

    No range is built in. Provider spread is per-model, so a range that is
    hardcoded here would be wrong for every model except the one it came from;
    :func:`main` takes it as ``--reference-range`` instead.
    """
    accuracy = summary.get("accuracy")
    score_text = f"{accuracy:.4f}" if isinstance(accuracy, float) else "none"
    stderr = summary.get("stderr")
    stderr_text = f" +/-{stderr:.4f}" if isinstance(stderr, float) else ""
    reference = summary.get("reference") or {}
    low, high, source = (
        reference.get("low"),
        reference.get("high"),
        reference.get("source"),
    )
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        origin = f" on {source}" if source else ""
        reference_text = f" (published providers{origin}: {low:.3f}-{high:.3f})"
    else:
        reference_text = ""
    return f"{dataset}: {score_text}{stderr_text}{reference_text}"


# The writers below duplicate a slice of eval_common rather than importing it.
# This module is executed by the pinned third-party harness venv (openbench or
# tau2-bench), which does not have eval_common's openai/tqdm dependencies, so it
# has to stay standard-library only — the same constraint collect_scores.py
# documents. Key names are kept identical so downstream consumers (the workflow
# job summary, collect_scores) cannot tell the difference.

#: Mirrors ``eval_common.MAX_ERROR_RATE``.
DEFAULT_MAX_ERROR_RATE = 0.10


def write_outputs(
    out_dir: str,
    dataset: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    metric_prefix: str,
) -> None:
    """Writes ``results.jsonl`` + ``score.json`` and appends the CI metric keys.

    Mirrors :func:`eval_common.write_outputs` for a harness-normalized summary.

    Args:
        out_dir: Directory to create and write into.
        dataset: One of :data:`DATASETS`, used for the score line's label.
        rows: Per-question records.
        summary: Normalized summary from a ``parse_*`` function.
        metric_prefix: Prefix for the ``$GITHUB_ENV`` metric keys.
    """
    # Contamination is recorded before score.json is written, or the file ships
    # without it and every downstream reader (the job summary included) reports
    # a contaminated run as clean.
    position_warning = answer_position_warning(rows)
    if position_warning:
        summary["target_letter_counts"] = target_letter_counts(rows)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    with open(os.path.join(out_dir, "score.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(format_score_line(dataset, summary))
    if position_warning:
        print(position_warning)
    _enforce_no_samples(summary)
    _append_github_env(metric_prefix, summary)
    _enforce_error_budget(summary)
    _enforce_stop_ratio(summary)


def _append_github_env(metric_prefix: str, summary: dict[str, Any]) -> None:
    """Appends ``<PREFIX>_SCORE/_MEAN_TOKENS/_P50_TOKENS``, plus the verdict.

    A no-op when ``GITHUB_ENV`` is unset, matching
    :func:`eval_common.append_github_env`.
    """
    env_file = os.environ.get("GITHUB_ENV")
    if not env_file:
        return
    accuracy = summary.get("accuracy")
    with open(env_file, "a") as f:
        if isinstance(accuracy, (int, float)):
            f.write(f"{metric_prefix}_SCORE={accuracy:.4f}\n")
        f.write(
            f"{metric_prefix}_MEAN_TOKENS="
            f"{summary.get('mean_output_tokens') or 0:.0f}\n"
        )
        f.write(
            f"{metric_prefix}_P50_TOKENS="
            f"{summary.get('p50_output_tokens') or 0:.0f}\n"
        )
        stderr = summary.get("stderr")
        if isinstance(stderr, (int, float)):
            f.write(f"{metric_prefix}_STDERR={stderr:.4f}\n")


def _enforce_no_samples(summary: dict[str, Any]) -> None:
    """Exits nonzero when the harness scored nothing at all.

    The error budget cannot catch this: with zero samples the harness reports
    no counts, both guards return early, and the run exits 0 having printed
    ``none`` as the score. That is how a 404 against every request passed CI.
    """
    accuracy = summary.get("accuracy")
    if isinstance(accuracy, (int, float)):
        return
    status = summary.get("harness_status")
    print(
        "::error::the harness scored no samples "
        f"(status={status!r}, completed={summary.get('completed')!r}, "
        f"total={summary.get('total')!r}). Nothing was measured, so there is "
        "no score. Check the endpoint, the model name and the credential."
    )
    raise SystemExit(1)


def _enforce_error_budget(summary: dict[str, Any]) -> None:
    """Exits nonzero when too many samples errored to trust the score.

    Mirrors :func:`eval_common.enforce_error_budget`, including the
    ``EVAL_MAX_ERROR_RATE`` override, so an endpoint that rejects most requests
    cannot report a passing-looking 0.0.
    """
    total = summary.get("total")
    errors = summary.get("errors")
    if not isinstance(total, int) or not isinstance(errors, int) or total <= 0:
        return
    budget = float(
        os.environ.get("EVAL_MAX_ERROR_RATE") or DEFAULT_MAX_ERROR_RATE
    )
    if budget <= 0:
        return
    rate = errors / total
    if rate <= budget:
        return
    print(
        f"::error::{errors}/{total} samples errored ({rate:.1%}), above the "
        f"{budget:.0%} budget. Errors score as incorrect, so this run's score "
        f"reflects infrastructure, not model quality."
    )
    raise SystemExit(1)


def parse_reference(
    raw: str | None, source: str | None
) -> dict[str, Any] | None:
    """Parses a ``LOW,HIGH`` reference range, or returns ``None``.

    Args:
        raw: ``"0.842,0.908"``, or ``None`` when the caller gave no range.
        source: Free text naming where the range came from.

    Returns:
        A ``{"low", "high", "source"}`` dict, or ``None``.

    Raises:
        SystemExit: If ``raw`` is not two numbers in ``[0, 1]`` with low <= high.
    """
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise SystemExit(f"--reference-range must be 'LOW,HIGH', got {raw!r}")
    try:
        low, high = float(parts[0]), float(parts[1])
    except ValueError:
        raise SystemExit(
            f"--reference-range must be two numbers, got {raw!r}"
        ) from None
    if not 0.0 <= low <= high <= 1.0:
        raise SystemExit(
            f"--reference-range must satisfy 0 <= low <= high <= 1, got {raw!r}"
        )
    return {"low": low, "high": high, "source": source or None}


def merge_endpoint_meta(
    summary: dict[str, Any], meta: dict[str, Any] | None
) -> dict[str, Any]:
    """Records what was serving the model alongside the score.

    A score is only evidence if you can say what produced it. Quantization, KV
    cache dtype and speculative decoding all move accuracy (see the
    max-serve-evaluator eval-pitfalls notes), and none of them are visible in
    the harness log, so the runner probes the endpoint and passes what it found
    through here. Stored under ``endpoint`` rather than merged flat so a field
    the server volunteers can never shadow a harness field.
    """
    summary["endpoint"] = meta or {}
    return summary


def _enforce_stop_ratio(summary: dict[str, Any]) -> None:
    """Exits nonzero when too many responses hit the token cap to trust the run.

    Mirrors :func:`eval_common.enforce_stop_ratio`, including the
    ``EVAL_MIN_STOP_RATIO`` opt-in. Truncation silently under-measures any
    chain-of-thought benchmark, and openbench sets no ``max_tokens`` on
    gpqa_diamond, so the server's default decides whether a long reasoning
    trace ever reaches its answer line. Off unless a floor is set, because the
    right floor is dataset- and model-specific.
    """
    floor_env = os.environ.get("EVAL_MIN_STOP_RATIO") or ""
    if not floor_env:
        return
    ratio = summary.get("stop_ratio")
    if not isinstance(ratio, (int, float)):
        return
    floor = float(floor_env)
    if ratio >= floor:
        return
    print(
        f"::error::stop_ratio {ratio:.4f} is below the {floor:.4f} floor: "
        f"{summary.get('finish_length')} of "
        f"{(summary.get('finish_stop') or 0) + (summary.get('finish_length') or 0)}"
        " completed responses hit the token cap instead of stopping, so this "
        "score reflects truncation rather than model quality."
    )
    raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    """Normalizes a harness log into this suite's eval artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harness",
        required=True,
        choices=["openbench", "tau2"],
        help="Which harness produced --log.",
    )
    parser.add_argument(
        "--log", required=True, help="Harness log to read (openbench: JSON)."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
        help="Dataset key, selecting the reference range to report.",
    )
    parser.add_argument(
        "--out-dir", required=True, help="Where to write results."
    )
    parser.add_argument(
        "--metric-prefix",
        required=True,
        help="Prefix for the $GITHUB_ENV metric keys (e.g. EXACTO_GPQA).",
    )
    parser.add_argument(
        "--full-task-count",
        type=int,
        help=(
            "How many tasks the chosen split holds, so a subset run is "
            "identifiable without anything hardcoding the dataset size."
        ),
    )
    parser.add_argument(
        "--reference-user-llm",
        help=(
            "The simulator OpenRouter pins, recorded so downstream readers can "
            "tell a parity run from a substitute without hardcoding the name."
        ),
    )
    parser.add_argument(
        "--reference-range",
        help=(
            "Optional 'LOW,HIGH' range published providers land in for THIS "
            "model, shown next to the score as context. Provider spread is "
            "per-model, so there is deliberately no default."
        ),
    )
    parser.add_argument(
        "--reference-source",
        help="Where --reference-range came from, e.g. an OpenRouter model slug.",
    )
    parser.add_argument(
        "--endpoint-meta",
        help="JSON file describing what served the model, recorded verbatim.",
    )
    args = parser.parse_args(argv)

    log = load_json(args.log)
    if args.harness == "openbench":
        rows, summary = parse_openbench_log(log)
    else:
        rows, summary = parse_tau2_log(log)
    merge_endpoint_meta(
        summary, load_json(args.endpoint_meta) if args.endpoint_meta else None
    )
    summary["reference"] = parse_reference(
        args.reference_range, args.reference_source
    )
    if args.reference_user_llm:
        summary["reference_user_llm"] = args.reference_user_llm
    if args.full_task_count:
        summary["full_task_count"] = args.full_task_count
    write_outputs(args.out_dir, args.dataset, rows, summary, args.metric_prefix)


if __name__ == "__main__":
    main()
