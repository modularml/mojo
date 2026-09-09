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
"""Shared scaffolding for the de-embedded dataset evals.

The individual eval modules (``aime_eval``, ``gpqa_eval``, ``hle_eval``,
``aa_lcr_eval``, ``aa_omniscience_eval``) were de-embedded from
``minimaxM3NonAgenticTextDatasetEval.yaml`` inline heredocs. This module
factors out the scaffolding they share — client construction, ``<think>``
stripping, chat-kwargs assembly, subsampling, parallel execution, exact-match
scoring, output writing, gated-dataset skipping, MCQ parsing, and LLM-judge
helpers — so each eval only holds its dataset-specific logic.

Everything here is pure or takes an injected client, so the evals unit-test
without a server or network.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI
from tqdm import tqdm

# Root identity preamble that MiniMax prepends to every request in their own
# dataset evals (verified byte-identical across sampled production traces).
# Model-specific; kept as a CLI default so an eval can target other models by
# overriding it (or passing an empty string to drop the ``root`` turn).
DEFAULT_ROOT_PREAMBLE = (
    "Your model version is MiniMax-M3, developed by MiniMax. "
    "Knowledge cutoff: January 2026. Founded in early 2022, MiniMax is "
    "a global AI foundation model company committed to advancing the "
    "frontiers of AI towards AGI."
)
# MiniMax's own traces use the generic assistant system prompt.
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_YESNO_FIELD_RE = re.compile(r"correct\s*:\s*\**\s*(yes|no)")
_YESNO_BARE_RE = re.compile(r"\b(yes|no)\b")


class ChatClient(Protocol):
    """Minimal structural type for the OpenAI chat-completions client.

    Declared so the eval drivers can be exercised with a fake client in tests.
    """

    @property
    def chat(self) -> Any: ...


@dataclass
class GenParams:
    """Generation and sampling parameters for a single request.

    ``top_p`` and ``seed`` are ``Optional``: ``None`` omits the field entirely
    (server default). Judges pass ``top_p=None`` since they send no ``top_p``.
    """

    max_tokens: int = 98304
    temperature: float = 1.0
    top_p: float | None = 0.95
    seed: int | None = None


def make_client(base_url: str, api_key: str | None = None) -> OpenAI:
    """Builds the OpenAI-compatible client used by every eval.

    No client timeout: a request must only ever end by the server hitting
    ``max_tokens``, never a client-side deadline (which would drop the hardest,
    longest-reasoning problems and inflate accuracy). ``max_retries=0``: a retry
    opens a new request while the original generation keeps running and holding
    KV cache, piling the server up toward OOM.

    Args:
        base_url: Server base URL; ``/v1`` is appended.
        api_key: Credential for an authenticated endpoint. Falls back to
            ``$OPENAI_API_KEY``, then to ``"dummy"`` — a local MAX Serve
            ignores it, but an authenticated deployment 401s without it.
    """
    return OpenAI(
        base_url=base_url.rstrip("/") + "/v1",
        api_key=api_key or os.environ.get("OPENAI_API_KEY") or "dummy",
        timeout=None,
        max_retries=0,
    )


def strip_think(content: str | None) -> str:
    """Removes any ``<think>...</think>`` span and surrounding whitespace.

    Args:
        content: Raw assistant message content. ``None`` (returned by some
            backends when a response is truncated mid-reasoning) is treated as
            empty rather than raising.

    Returns:
        The think-stripped, stripped text.
    """
    return _THINK_RE.sub("", content or "").strip()


def build_chat_kwargs(
    model: str, messages: Sequence[Mapping[str, Any]], params: GenParams
) -> dict[str, Any]:
    """Assembles ``chat.completions.create`` kwargs.

    ``messages`` is typed permissively (``Mapping`` values are ``Any``) so both
    the text evals' plain ``str`` content and the multimodal evals' list-of-parts
    content type-check through the same helper. Omits ``top_p`` when ``None``
    (judges send no ``top_p``) and ``seed`` when ``None`` (server default).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
    }
    if params.top_p is not None:
        kwargs["top_p"] = params.top_p
    if params.seed is not None:
        kwargs["seed"] = params.seed
    return kwargs


def subsample(
    dataset: list[dict[str, Any]], sample_size: int | None
) -> list[dict[str, Any]]:
    """Takes ``sample_size`` items evenly across the dataset (``None`` = all)."""
    if not sample_size:
        return dataset
    step = max(1, len(dataset) // sample_size)
    return dataset[::step][:sample_size]


def make_repeat_samples(
    dataset: list[dict[str, Any]], repeats: int
) -> list[tuple[int, int, dict[str, Any]]]:
    """Expands ``dataset`` into ``(repeat_index, prompt_index, item)`` tuples.

    Every output row can then be tied back to its problem and repeat regardless
    of completion order.
    """
    return [
        (rep, i, item)
        for rep in range(repeats)
        for i, item in enumerate(dataset)
    ]


def select_rows(
    dataset: list[dict[str, Any]],
    sample_size: int | None,
    row_ids: str | None,
) -> list[tuple[int, dict[str, Any]]]:
    """Selects which rows of ``dataset`` to evaluate, keeping original indices.

    Pass at most one of ``sample_size`` (an evenly-strided subset) or
    ``row_ids`` (explicit comma-separated indices into the full dataset,
    order and duplicates preserved) — passing both is an error. Neither set
    returns the full dataset. Every row comes back as
    ``(original_index, item)`` so it stays traceable to the same dataset row
    regardless of which selection switch was used — e.g. to re-run exactly
    the rows that looked wrong last time via ``row_ids``.

    Raises:
        ValueError: ``sample_size`` and ``row_ids`` are both set, or a
            ``row_ids`` index is out of range.
    """
    if sample_size and row_ids:
        raise ValueError(
            "sample_size and row_ids are mutually exclusive — set only one."
        )
    if row_ids:
        ids = [int(x) for x in row_ids.split(",") if x.strip()]
        bad = [i for i in ids if not (0 <= i < len(dataset))]
        if bad:
            raise ValueError(
                f"row_ids {bad} out of range for dataset of size {len(dataset)}"
            )
        return [(i, dataset[i]) for i in ids]
    if sample_size:
        step = max(1, len(dataset) // sample_size)
        return list(enumerate(dataset))[::step][:sample_size]
    return list(enumerate(dataset))


def expand_repeats(
    indexed_dataset: list[tuple[int, dict[str, Any]]], repeats: int
) -> list[tuple[int, int, dict[str, Any]]]:
    """Expands ``(index, item)`` pairs into ``(repeat_index, index, item)``.

    Like :func:`make_repeat_samples`, but for a dataset already narrowed by
    :func:`select_rows`, so ``index`` stays the original full-dataset index
    instead of a position in the narrowed subset.
    """
    return [
        (rep, idx, item)
        for rep in range(repeats)
        for idx, item in indexed_dataset
    ]


def run_parallel(
    items: list[Any],
    fn: Callable[[Any], dict[str, Any]],
    on_error: Callable[[Any, Exception], dict[str, Any]],
    workers: int,
    desc: str,
    out_dir: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Runs ``fn`` over ``items`` on a thread pool with a tqdm progress bar.

    On error, ``on_error(item, exc)`` builds a row that is RECORDED (never
    dropped) so dropping the hardest items can't inflate accuracy.

    When ``out_dir`` is set, each row is written to ``results.jsonl`` and
    flushed as soon as its future resolves, so a killed/crashed run still
    leaves every completed sample on disk rather than losing everything
    because the file would otherwise only be written after the slowest
    request finishes.

    Returns:
        A ``(results, errors)`` tuple.
    """
    results: list[dict[str, Any]] = []
    errors = 0
    out_f = None
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_f = open(os.path.join(out_dir, "results.jsonl"), "w")
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fn, it): it for it in items}
            for fut in tqdm(as_completed(futures), total=len(items), desc=desc):
                it = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    errors += 1
                    print(f"Error: {e}")
                    r = on_error(it, e)
                results.append(r)
                if out_f is not None:
                    out_f.write(json.dumps(r) + "\n")
                    out_f.flush()
    finally:
        if out_f is not None:
            out_f.close()
    return results, errors


def token_stats(results: list[dict[str, Any]]) -> tuple[float, float]:
    """Computes ``(mean, p50)`` output tokens over non-error, tokened rows."""
    otoks = [
        r["completion_tokens"]
        for r in results
        if "completion_tokens" in r and "error" not in r
    ]
    mean_output_tokens = round(statistics.mean(otoks), 1) if otoks else 0.0
    p50_output_tokens = round(statistics.median(otoks), 1) if otoks else 0.0
    return mean_output_tokens, p50_output_tokens


def truncation_stats(
    results: list[dict[str, Any]],
) -> tuple[int, float, float]:
    """Counts cap-truncated rows and token stats over the finished rows.

    A row is truncated when generation hit the token cap
    (``finish_reason == "length"``). On a reasoning model these are typically
    runaway-reasoning samples that never emitted a visible answer; they score
    wrong and dominate the mean token count, so reporting them separately lets
    a mean-vs-reference gap be attributed to the tail rather than to typical
    generations.
    """
    truncated_rows = [r for r in results if r.get("finish_reason") == "length"]
    finished = [r for r in results if r.get("finish_reason") != "length"]
    mean_finished, p50_finished = token_stats(finished)
    return len(truncated_rows), mean_finished, p50_finished


def _graded_correct(row: dict[str, Any]) -> bool:
    """Reads the correctness flag most evals record directly on the row."""
    return bool(row.get("correct"))


def finish_stats(
    results: list[dict[str, Any]],
    is_correct: Callable[[dict[str, Any]], bool] = _graded_correct,
) -> dict[str, Any]:
    """Counts finish reasons and splits accuracy by how generation ended.

    ``stop_ratio`` is the share of completed responses that finished with
    ``stop`` rather than by hitting the token cap (``length``), mirroring how
    the vendor reference reports it. Rows that errored have no finish reason
    and stay out of the ratio.

    ``accuracy_given_stop`` and ``accuracy_given_length`` are accuracy over the
    stopped and the truncated rows, so accuracy over the completed rows factors
    exactly::

        completed = accuracy_given_stop   * stop_ratio
                  + accuracy_given_length * (1 - stop_ratio)

    Reporting the factors rather than only their sum keeps a shift in how often
    the model fails to terminate from reading as a regression in answer
    quality: on a reasoning model at a high token cap the runaway tail is
    concentrated on a handful of prompts and moves far more easily than quality
    does, so the two move independently and a single number cannot say which
    one did.

    ``accuracy_given_length`` is normally ``0.0``, because a response truncated
    mid-reasoning carries no parseable answer. That holds because of what the
    endpoint returns, though, not because of anything enforced here -- the
    graders parse whatever content came back without consulting
    ``finish_reason``, so a truncated response that happens to contain a
    parseable answer does grade correct. Reporting the term measures that per
    run instead of assuming it, and leaves the identity above exact either way.

    Errored rows sit outside it entirely. :func:`exact_match_score` keeps them
    in the denominator of the headline ``accuracy`` so that dropping the hardest
    problems cannot inflate it, so the two agree only on an error-free run.
    Otherwise ``accuracy`` is the completed-row figure scaled by the
    ``answered / total`` completion factor, whose terms that summary reports
    alongside it.

    The reported ratios are rounded, so reconstruct from ``correct_given_stop``
    and ``correct_given_length`` when the identity has to hold to the last bit.

    Args:
        results: The graded rows, including any that errored.
        is_correct: Reads whether a row was graded correct. Defaults to the
            ``correct`` flag; evals that grade into some other field (such as
            AA-Omniscience's four-way ``verdict``) pass their own reader rather
            than duplicating the grade onto the row.

    Returns:
        The finish-reason counts, ``stop_ratio``, and the correct-count and
        accuracy conditioned on each finish reason. An accuracy is ``None``
        when its finish reason never occurred, so that "no such rows" stays
        distinguishable from "those rows all graded wrong".
    """
    stop_rows = [r for r in results if r.get("finish_reason") == "stop"]
    length_rows = [r for r in results if r.get("finish_reason") == "length"]
    stop = len(stop_rows)
    length = len(length_rows)
    completed = stop + length
    correct_given_stop = sum(1 for r in stop_rows if is_correct(r))
    correct_given_length = sum(1 for r in length_rows if is_correct(r))
    return {
        "finish_stop": stop,
        "finish_length": length,
        "stop_ratio": round(stop / completed, 4) if completed else None,
        "correct_given_stop": correct_given_stop,
        "accuracy_given_stop": (
            round(correct_given_stop / stop, 4) if stop else None
        ),
        "correct_given_length": correct_given_length,
        "accuracy_given_length": (
            round(correct_given_length / length, 4) if length else None
        ),
    }


def exact_match_score(
    results: list[dict[str, Any]], total: int, errors: int
) -> dict[str, Any]:
    """Computes the accuracy + token-stats summary over every submitted sample.

    Errors/timeouts count as incorrect (they stay in ``results`` as rows with an
    ``error`` key) so dropping the hardest problems cannot inflate accuracy.
    """
    correct = sum(1 for r in results if r.get("correct"))
    accuracy = correct / total if total else 0.0
    mean_output_tokens, p50_output_tokens = token_stats(results)
    truncated, mean_finished, p50_finished = truncation_stats(results)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "answered": total - errors,
        "errors": errors,
        "mean_output_tokens": mean_output_tokens,
        "p50_output_tokens": p50_output_tokens,
        "truncated": truncated,
        "stop_ratio": round((total - errors - truncated) / (total - errors), 4)
        if (total - errors) > 0
        else 0.0,
        "mean_output_tokens_finished": mean_finished,
        "p50_output_tokens_finished": p50_finished,
        **finish_stats(results),
    }


def dump_jsonl(out_dir: str, results: list[dict[str, Any]]) -> None:
    """Writes ``results.jsonl`` (one JSON object per line) into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in results))


#: Fraction of samples that may error before the score stops meaning anything.
#: Errors count as incorrect, so an endpoint that rejects every request scores
#: 0.0 and, without this, reports success. Override for a deliberately lossy
#: run; 0 disables the check.
MAX_ERROR_RATE = float(os.environ.get("EVAL_MAX_ERROR_RATE") or 0.10)


def enforce_error_budget(summary: dict[str, Any]) -> None:
    """Exits nonzero when too many samples errored to trust the score.

    Hooked into :func:`dump_score` so it covers every eval that writes its
    summary there. Summaries reporting no error count are left alone rather
    than guessed at, which is also how SciCode escapes it: it writes
    ``score.json`` itself and discards the count :func:`run_parallel` returns.
    """
    total = summary.get("total")
    errors = summary.get("errors", summary.get("errored"))
    if not isinstance(total, int) or not isinstance(errors, int) or total <= 0:
        return
    if MAX_ERROR_RATE <= 0:
        return
    rate = errors / total
    if rate <= MAX_ERROR_RATE:
        return
    print(
        f"::error::{errors}/{total} samples errored ({rate:.1%}), above the "
        f"{MAX_ERROR_RATE:.0%} budget. Errors score as incorrect, so this run's "
        f"score reflects infrastructure, not model quality."
    )
    raise SystemExit(1)


#: Minimum share of completed responses that finish with ``stop`` rather than
#: the token cap. Nothing sets this yet; floors are dataset-specific, so each
#: dataset opts in via its workflow env. Parsed at import so typos fail fast.
_MIN_STOP_RATIO_ENV = os.environ.get("EVAL_MIN_STOP_RATIO") or ""
MIN_STOP_RATIO = float(_MIN_STOP_RATIO_ENV) if _MIN_STOP_RATIO_ENV else None


def enforce_stop_ratio(summary: dict[str, Any]) -> None:
    """Exits nonzero when too many responses hit the token cap to trust the run.

    Off unless ``EVAL_MIN_STOP_RATIO`` is set, and skipped for summaries that
    report no usable ``stop_ratio``.
    """
    if MIN_STOP_RATIO is None:
        return
    ratio = summary.get("stop_ratio")
    if not isinstance(ratio, (int, float)):
        return
    floor = MIN_STOP_RATIO
    if ratio >= floor:
        return
    print(
        f"::error::stop_ratio {ratio:.4f} is below the {floor:.4f} floor: "
        f"{summary.get('finish_length')} of "
        f"{summary.get('finish_stop', 0) + summary.get('finish_length', 0)} "
        f"completed responses hit the token cap instead of stopping."
    )
    raise SystemExit(1)


def dump_score(out_dir: str, summary: dict[str, Any]) -> None:
    """Writes ``score.json`` (pretty-printed) into ``out_dir``.

    Writes before enforcing the error budget so a rejected run still leaves its
    artifacts behind for diagnosis.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "score.json"), "w") as f:
        json.dump(summary, f, indent=2)
    enforce_error_budget(summary)
    enforce_stop_ratio(summary)


def append_github_env(
    metric_prefix: str, score: float, mean_tokens: float, p50_tokens: float
) -> None:
    """Appends the ``<PREFIX>_SCORE/_MEAN_TOKENS/_P50_TOKENS`` CI metric keys.

    A no-op when ``GITHUB_ENV`` is unset (local runs), matching the historical
    inline behavior.
    """
    env_file = os.environ.get("GITHUB_ENV")
    if not env_file:
        return
    with open(env_file, "a") as f:
        f.write(f"{metric_prefix}_SCORE={score:.4f}\n")
        f.write(f"{metric_prefix}_MEAN_TOKENS={mean_tokens:.0f}\n")
        f.write(f"{metric_prefix}_P50_TOKENS={p50_tokens:.0f}\n")


def write_outputs(
    out_dir: str,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    metric_prefix: str,
    label: str | None = None,
) -> None:
    """Writes ``results.jsonl`` + ``score.json`` and appends CI env metrics.

    The standard-eval convenience for exact-match-style summaries (produced by
    :func:`exact_match_score`). ``label`` defaults to ``metric_prefix`` and only
    affects the human-readable stdout line.
    """
    label = label or metric_prefix
    dump_jsonl(out_dir, results)
    dump_score(out_dir, summary)
    print(
        f"{label}: {summary['accuracy']:.4f} "
        f"({summary['correct']}/{summary['total']}, {summary['errors']} "
        f"errors/timeouts) mean_output_tokens={summary['mean_output_tokens']} "
        f"p50_output_tokens={summary['p50_output_tokens']}"
    )
    append_github_env(
        metric_prefix,
        summary["accuracy"],
        summary["mean_output_tokens"],
        summary["p50_output_tokens"],
    )


def is_hf_access_error(exc: BaseException) -> bool:
    """Reports whether ``exc`` (or its cause chain) is a gated-repo access error."""
    chain = ""
    e: BaseException | None = exc
    while e is not None:
        chain += str(e)
        e = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
    return (
        "gated" in chain.lower()
        or "403" in chain
        or "enable access" in chain.lower()
    )


def load_gated(
    loader: Callable[[], list[dict[str, Any]]], *, label: str, dataset_id: str
) -> list[dict[str, Any]]:
    """Runs ``loader`` for a gated dataset, failing the eval on access denial.

    Exits nonzero rather than skipping: the suite qualifies a release branch on
    all its datasets having run, so a lane that silently disappears reads as a
    pass when nothing was measured. Fix the token's gated-repo access instead.
    Other errors propagate.
    """
    try:
        return loader()
    except Exception as e:
        if is_hf_access_error(e):
            print(
                f"::error::{label} could not run — HF token does not have "
                f"access to {dataset_id} (gated). Visit "
                f"https://huggingface.co/datasets/{dataset_id} "
                f"to request access and enable gated-repo permissions on your token."
            )
            raise SystemExit(1) from None
        raise


def stable_seed(key: str) -> int:
    """Derives a reproducible 32-bit seed from ``key`` via md5.

    Python's built-in ``hash()`` on ``str`` is salted by ``PYTHONHASHSEED`` and
    varies run-to-run, so it cannot drive a reproducible shuffle; md5 can.
    """
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF


def parse_mcq_letter(raw: str, labels: str) -> str:
    """Extracts the chosen multiple-choice letter from a think-stripped answer.

    Takes the first standalone label stated after the last answer cue ("the
    answer is ...", "correct choice ..."); falls back to the last standalone
    label on the final non-empty line, then anywhere. Avoids grabbing a letter
    out of a word like "choices"/"definitely" or from a trailing distractor
    mention ("A, not B").

    Args:
        raw: Think-stripped response text.
        labels: The option letters, e.g. ``"ABCD"``.

    Returns:
        The uppercased chosen letter, or ``""`` when nothing parseable is found.
    """
    cls = f"[{''.join(labels)}]"
    cues = list(
        re.finditer(r"\b(?:answer|correct choice)\b", raw, re.IGNORECASE)
    )
    if cues:
        m = re.search(rf"\b({cls})\b", raw[cues[-1].end() :])
        if m:
            return m.group(1).upper()
    last_line = next((l for l in reversed(raw.splitlines()) if l.strip()), "")
    matches = re.findall(rf"\b({cls})\b", last_line) or re.findall(
        rf"\b({cls})\b", raw
    )
    return matches[-1].upper() if matches else ""


def parse_yes_no_verdict(text: str) -> bool:
    """Parses an LLM judge verdict into a boolean.

    Prefers the official ``correct: yes/no`` field; falls back to the last bare
    ``yes``/``no`` token. Defaults to ``False`` (incorrect) when nothing is
    parseable.
    """
    lowered = text.lower()
    m = _YESNO_FIELD_RE.search(lowered)
    if m:
        return m.group(1) == "yes"
    tokens = _YESNO_BARE_RE.findall(lowered)
    return bool(tokens) and tokens[-1] == "yes"


def judge(
    client: ChatClient,
    model: str,
    prompt: str,
    seed: int | None = None,
    max_tokens: int = 16384,
) -> str:
    """Runs a single-turn judge request and returns its think-stripped text.

    Uses ``temperature=0.0`` and no ``top_p``. ``max_tokens`` budgets a
    reasoning judge to finish thinking and still emit its verdict.
    """
    params = GenParams(
        max_tokens=max_tokens, temperature=0.0, top_p=None, seed=seed
    )
    resp = client.chat.completions.create(
        **build_chat_kwargs(
            model, [{"role": "user", "content": prompt}], params
        )
    )
    return strip_think(resp.choices[0].message.content or "")


def self_judge(
    client: ChatClient,
    model: str,
    prompt: str,
    seed: int | None = None,
    max_tokens: int = 16384,
) -> bool:
    """Runs :func:`judge` and parses a yes/no verdict from its response."""
    return parse_yes_no_verdict(
        judge(client, model, prompt, seed=seed, max_tokens=max_tokens)
    )
