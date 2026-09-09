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
"""AIME25 math-competition eval against an OpenAI-compatible server.

De-embedded from ``minimaxM3NonAgenticTextDatasetEval.yaml`` so the eval logic
lives in a tested, locally-runnable module instead of an inline YAML heredoc.
The scoring core (:func:`grade`) and the driver (:func:`run_eval`) are pure and
take an injected client, so they unit-test without a server or network. Shared
scaffolding lives in :mod:`eval_common`.

Run locally against a server on ``localhost:8000``::

    ./bazelw run //max/tests/integration/accuracy/model_evals:aime_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --sample-size 2 --seed 0 --out-dir /tmp/aime25

Results stream to ``<out-dir>/results.jsonl`` as each request completes, so a
killed/crashed run keeps everything already finished. To debug specific
problems (e.g. re-run exactly the ones that errored or looked wrong last
time), pass ``--row-ids`` instead of ``--sample-size``::

    ./bazelw run //max/tests/integration/accuracy/model_evals:aime_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --row-ids 3,17,42 --out-dir /tmp/aime25-debug
"""

from __future__ import annotations

import re
from typing import Any

import click
from datasets import load_dataset
from eval_common import (
    DEFAULT_ROOT_PREAMBLE,
    DEFAULT_SYSTEM_PROMPT,
    ChatClient,
    GenParams,
    build_chat_kwargs,
    exact_match_score,
    expand_repeats,
    make_client,
    run_parallel,
    select_rows,
    strip_think,
    write_outputs,
)

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_INT_RE = re.compile(r"-?\d+")


def grade(content: str | None, expected: object) -> tuple[bool, str]:
    """Grades one AIME response against its expected integer answer.

    Prefers a ``\\boxed{...}`` answer and falls back to the last integer in the
    text. Comparison is integer-valued so ``"042"`` matches ``42`` and trailing
    recap numbers in the reasoning do not cause false negatives.

    Args:
        content: Raw assistant message content (may include a ``<think>`` span
            or be ``None``).
        expected: The dataset's expected answer (stringified before parsing).

    Returns:
        A ``(correct, predicted)`` tuple, where ``predicted`` is the extracted
        answer string (empty when nothing parseable was found).
    """
    raw = strip_think(content)
    boxed = _BOXED_RE.findall(raw)
    ints = _INT_RE.findall(boxed[-1] if boxed else raw)
    predicted = ints[-1] if ints else ""
    exp_ints = _INT_RE.findall(str(expected))
    correct = (
        bool(ints) and bool(exp_ints) and int(predicted) == int(exp_ints[-1])
    )
    return correct, predicted


def build_messages(
    problem: str, root_preamble: str, system_prompt: str
) -> list[dict[str, str]]:
    """Builds the chat messages for one problem.

    The ``root`` preamble turn is omitted when ``root_preamble`` is empty, so a
    non-MiniMax target can run without it.
    """
    messages: list[dict[str, str]] = []
    if root_preamble:
        messages.append({"role": "root", "content": root_preamble})
    messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": "Return your final response within \\boxed{}. "
            + problem,
        }
    )
    return messages


def load_aime_dataset() -> list[dict[str, Any]]:
    """Loads the ``math-ai/aime25`` set, falling back to the ``train`` split."""
    try:
        return list(load_dataset("math-ai/aime25", split="test"))
    except Exception:
        return list(load_dataset("math-ai/aime25", split="train"))


def infer(
    client: ChatClient,
    model: str,
    item: tuple[int, int, dict[str, Any]],
    params: GenParams,
    root_preamble: str,
    system_prompt: str,
) -> dict[str, Any]:
    """Runs and grades a single (repeat, problem) sample."""
    repeat_index, prompt_index, sample = item
    problem = sample["problem"]
    expected = str(sample["answer"]).strip()
    messages = build_messages(problem, root_preamble, system_prompt)
    resp = client.chat.completions.create(
        **build_chat_kwargs(model, messages, params)
    )
    choice = resp.choices[0]
    correct, predicted = grade(choice.message.content, expected)
    return {
        "prompt_index": prompt_index,
        "repeat_index": repeat_index,
        "problem": problem[:120],
        "expected": expected,
        "predicted": predicted,
        "correct": correct,
        "finish_reason": choice.finish_reason,
        "completion_tokens": (
            resp.usage.completion_tokens if resp.usage else 0
        ),
    }


def run_eval(
    client: ChatClient,
    indexed_dataset: list[tuple[int, dict[str, Any]]],
    model: str,
    repeats: int,
    workers: int,
    params: GenParams,
    root_preamble: str,
    system_prompt: str,
    out_dir: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Runs AIME25 over ``indexed_dataset`` with ``repeats``.

    Each ``(repeat, problem)`` pair is submitted independently so a row can be
    tied back to its problem and repeat regardless of completion order. A
    failed/timed-out request is recorded as an incorrect row, never dropped.
    When ``out_dir`` is set, results stream to ``results.jsonl`` as they
    complete, so a crash mid-run doesn't lose already-finished samples.

    Args:
        client: OpenAI-compatible chat client (real or fake).
        indexed_dataset: ``(original_index, problem)`` pairs, already narrowed
            by :func:`eval_common.select_rows`.
        model: Served model name to request.
        repeats: Repeats per problem (``temperature`` makes each independent).
        workers: Max concurrent in-flight requests.
        params: Generation/sampling parameters.
        root_preamble: MiniMax root turn (empty to omit).
        system_prompt: System turn content.
        out_dir: When set, stream results to ``<out_dir>/results.jsonl``.

    Returns:
        A ``(results, score)`` tuple.
    """
    samples = expand_repeats(indexed_dataset, repeats)
    print(
        f"AIME25: evaluating {len(indexed_dataset)} problems x {repeats} "
        f"repeats = {len(samples)} total samples"
    )

    def fn(item: tuple[int, int, dict[str, Any]]) -> dict[str, Any]:
        return infer(client, model, item, params, root_preamble, system_prompt)

    def on_error(
        item: tuple[int, int, dict[str, Any]], exc: Exception
    ) -> dict[str, Any]:
        rep, pi, p = item
        return {
            "prompt_index": pi,
            "repeat_index": rep,
            "problem": p["problem"][:120],
            "error": str(exc),
            "correct": False,
        }

    results, errors = run_parallel(
        samples, fn, on_error, workers, "AIME25", out_dir=out_dir
    )
    return results, exact_match_score(results, len(samples), errors)


@click.command()
@click.option(
    "--base-url",
    required=True,
    help="Server base URL, e.g. http://localhost:8000",
)
@click.option("--model", required=True, help="Served model name to request.")
@click.option(
    "--sample-size",
    type=int,
    default=None,
    help="Max problems (evenly sampled). Empty = full dataset. Applied "
    "before repeats. Mutually exclusive with --row-ids.",
)
@click.option(
    "--row-ids",
    default=None,
    help="Explicit comma-separated dataset row indices to evaluate, e.g. "
    "'3,17,42' (indices into the full dataset, order/duplicates preserved). "
    "Mutually exclusive with --sample-size — use this to re-run exactly the "
    "problems that errored or looked wrong last time.",
)
@click.option("--repeats", type=int, default=16, help="Repeats per problem.")
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Per-request seed for reproducibility (omitted when unset).",
)
@click.option(
    "--workers", type=int, default=16, help="Max concurrent requests."
)
@click.option(
    "--out-dir", default="/tmp/aime25-results", help="Output directory."
)
@click.option("--max-tokens", type=int, default=98304, show_default=True)
@click.option("--temperature", type=float, default=1.0, show_default=True)
@click.option("--top-p", type=float, default=0.95, show_default=True)
@click.option(
    "--root-preamble",
    default=DEFAULT_ROOT_PREAMBLE,
    help="Root identity turn (empty to omit; default is MiniMax-M3's).",
)
@click.option(
    "--system-prompt", default=DEFAULT_SYSTEM_PROMPT, show_default=True
)
@click.option(
    "--metric-prefix",
    default="AIME25",
    help="Prefix for the GITHUB_ENV metric keys the job summary reads.",
)
def main(
    base_url: str,
    model: str,
    sample_size: int | None,
    row_ids: str | None,
    repeats: int,
    seed: int | None,
    workers: int,
    out_dir: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    root_preamble: str,
    system_prompt: str,
    metric_prefix: str,
) -> None:
    """Runs AIME25 against a running OpenAI-compatible server and scores it."""
    client = make_client(base_url)
    indexed_dataset = select_rows(load_aime_dataset(), sample_size, row_ids)
    params = GenParams(
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed
    )
    results, summary = run_eval(
        client,
        indexed_dataset,
        model,
        repeats,
        workers,
        params,
        root_preamble,
        system_prompt,
        out_dir=out_dir,
    )
    write_outputs(out_dir, results, summary, metric_prefix)


if __name__ == "__main__":
    main()
