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
"""GPQA-diamond PhD-level multiple-choice eval against an OpenAI-compatible server.

De-embedded from ``minimaxM3NonAgenticTextDatasetEval.yaml`` so the eval logic
lives in a tested, locally-runnable module instead of an inline YAML heredoc.
The scoring core (:func:`grade`), the deterministic option shuffle
(:func:`prepare_options`), and the driver (:func:`run_eval`) take pure inputs or
an injected client, so they unit-test without a server or network. Shared
scaffolding lives in :mod:`eval_common`.

GPQA-diamond is gated on HuggingFace; the eval prints a ``::warning::`` and
exits 0 when the HF token lacks access.

Run locally against a server on ``localhost:8000``::

    ./bazelw run //max/tests/integration/accuracy/model_evals:gpqa_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --sample-size 2 --out-dir /tmp/gpqa

Results stream to ``<out-dir>/results.jsonl`` as each request completes, so a
killed/crashed run keeps everything already finished. To debug specific rows
(e.g. re-run exactly the ones that errored or looked wrong last time), pass
``--row-ids`` instead of ``--sample-size``::

    ./bazelw run //max/tests/integration/accuracy/model_evals:gpqa_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --row-ids 3,17,42 --out-dir /tmp/gpqa-debug
"""

from __future__ import annotations

import random
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
    load_gated,
    make_client,
    parse_mcq_letter,
    run_parallel,
    select_rows,
    stable_seed,
    strip_think,
    write_outputs,
)

# MiniMax's own GPQA traces use the generic assistant system prompt and put the
# multiple-choice / "Answer: A/B/C/D" instruction in the user turn ahead of the
# question and options.
GPQA_INSTRUCTION = (
    "Answer the following multiple choice question. The last line of "
    "your response should be in the following format: "
    "'Answer: A/B/C/D' (e.g. 'Answer: A'). "
)


def prepare_options(
    sample: dict[str, Any], seed: int | None = None
) -> tuple[list[str], str]:
    """Deterministically shuffles the four options and returns the answer letter.

    Uses a stable per-question seed (:func:`eval_common.stable_seed`) so the
    shuffle is reproducible run-to-run (unlike ``PYTHONHASHSEED``-salted
    ``hash()``). When ``seed`` is set, it's folded into the shuffle key so
    changing the seed also reorders the options — useful for checking that a
    correct answer isn't an artifact of its position.

    Returns:
        A ``(shuffled_options, correct_letter)`` tuple.
    """
    options = [
        sample["Correct Answer"],
        sample["Incorrect Answer 1"],
        sample["Incorrect Answer 2"],
        sample["Incorrect Answer 3"],
    ]
    question = sample["Question"]
    key = f"{seed}:{question}" if seed is not None else question
    rng = random.Random(stable_seed(key))
    rng.shuffle(options)
    correct_letter = "ABCD"[options.index(sample["Correct Answer"])]
    return options, correct_letter


def grade(content: str | None, correct_letter: str) -> tuple[bool, str]:
    """Grades one GPQA response against its correct option letter.

    Returns:
        A ``(correct, predicted)`` tuple, where ``predicted`` is the extracted
        A-D letter (empty when nothing parseable was found).
    """
    predicted = parse_mcq_letter(strip_think(content), "ABCD")
    return predicted == correct_letter, predicted


def build_messages(
    prompt: str, root_preamble: str, system_prompt: str
) -> list[dict[str, str]]:
    """Builds the chat messages for one question (root turn omitted when empty)."""
    messages: list[dict[str, str]] = []
    if root_preamble:
        messages.append({"role": "root", "content": root_preamble})
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def infer(
    client: ChatClient,
    model: str,
    item: tuple[int, int, dict[str, Any]],
    params: GenParams,
    root_preamble: str,
    system_prompt: str,
) -> dict[str, Any]:
    """Runs and grades a single (repeat, question) sample."""
    repeat_index, prompt_index, sample = item
    question = sample["Question"]
    options, correct_letter = prepare_options(sample, params.seed)
    options_str = "\n".join(
        f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)
    )
    prompt = f"{GPQA_INSTRUCTION}\n\n{question}\n\n{options_str}"
    messages = build_messages(prompt, root_preamble, system_prompt)
    resp = client.chat.completions.create(
        **build_chat_kwargs(model, messages, params)
    )
    choice = resp.choices[0]
    correct, predicted = grade(choice.message.content, correct_letter)
    return {
        "prompt_index": prompt_index,
        "repeat_index": repeat_index,
        "question": question[:120],
        "correct_letter": correct_letter,
        "predicted": predicted,
        "correct": correct,
        "finish_reason": choice.finish_reason,
        "completion_tokens": (
            resp.usage.completion_tokens if resp.usage else 0
        ),
    }


def load_gpqa_dataset() -> list[dict[str, Any]]:
    """Loads the gated ``idavidrein/gpqa`` diamond split, skipping on no access."""
    return load_gated(
        lambda: list(
            load_dataset("idavidrein/gpqa", "gpqa_diamond", split="train")
        ),
        label="GPQA-diamond",
        dataset_id="idavidrein/gpqa",
    )


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
    """Runs GPQA-diamond over ``indexed_dataset`` with ``repeats``.

    Each ``(repeat, question)`` pair is submitted independently. A failed/
    timed-out request is recorded as an incorrect row, never dropped. When
    ``out_dir`` is set, results stream to ``results.jsonl`` as they complete,
    so a crash mid-run doesn't lose already-finished samples.

    Returns:
        A ``(results, score)`` tuple.
    """
    samples = expand_repeats(indexed_dataset, repeats)
    print(
        f"GPQA-diamond: evaluating {len(indexed_dataset)} questions x "
        f"{repeats} repeats = {len(samples)} total samples"
    )

    def fn(item: tuple[int, int, dict[str, Any]]) -> dict[str, Any]:
        return infer(client, model, item, params, root_preamble, system_prompt)

    def on_error(
        item: tuple[int, int, dict[str, Any]], exc: Exception
    ) -> dict[str, Any]:
        rep, qi, q = item
        return {
            "prompt_index": qi,
            "repeat_index": rep,
            "question": q["Question"][:120],
            "error": str(exc),
            "correct": False,
        }

    results, errors = run_parallel(
        samples, fn, on_error, workers, "GPQA-diamond", out_dir=out_dir
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
    help="Max questions (evenly sampled). Empty = full dataset. Applied "
    "before repeats. Mutually exclusive with --row-ids.",
)
@click.option(
    "--row-ids",
    default=None,
    help="Explicit comma-separated dataset row indices to evaluate, e.g. "
    "'3,17,42' (indices into the full dataset, order/duplicates preserved). "
    "Mutually exclusive with --sample-size — use this to re-run exactly the "
    "rows that errored or looked wrong last time.",
)
@click.option("--repeats", type=int, default=5, help="Repeats per question.")
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
    "--out-dir", default="/tmp/gpqa-results", help="Output directory."
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
    default="GPQA",
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
    """Runs GPQA-diamond against a running OpenAI-compatible server and scores it."""
    client = make_client(base_url)
    indexed_dataset = select_rows(load_gpqa_dataset(), sample_size, row_ids)
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
    write_outputs(
        out_dir, results, summary, metric_prefix, label="GPQA-diamond"
    )


if __name__ == "__main__":
    main()
