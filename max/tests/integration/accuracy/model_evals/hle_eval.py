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
"""HLE (Humanity's Last Exam) eval against an OpenAI-compatible server.

De-embedded from ``minimaxM3NonAgenticTextDatasetEval.yaml`` so the eval logic
lives in a tested, locally-runnable module instead of an inline YAML heredoc.
The model elicits a marked answer, then an LLM self-judge (the same served
model, no external judge key in CI) grades it yes/no. The driver
(:func:`run_eval`) takes an injected client, so it unit-tests without a server
or network. Shared scaffolding lives in :mod:`eval_common`.

HLE is gated on HuggingFace; the eval prints a ``::warning::`` and exits 0 when
the HF token lacks access. Image-based questions are skipped (text-only eval).

Run locally against a server on ``localhost:8000``::

    ./bazelw run //max/tests/integration/accuracy/model_evals:hle_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --sample-size 2 --out-dir /tmp/hle

Results stream to ``<out-dir>/results.jsonl`` as each request completes, so a
killed/crashed run keeps everything already finished. To debug specific
questions (e.g. re-run exactly the ones that errored or looked wrong last
time), pass ``--row-ids`` instead of ``--sample-size``::

    ./bazelw run //max/tests/integration/accuracy/model_evals:hle_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --row-ids 3,17,42 --out-dir /tmp/hle-debug
"""

from __future__ import annotations

from typing import Any

import click
from datasets import load_dataset
from eval_common import (
    DEFAULT_ROOT_PREAMBLE,
    ChatClient,
    GenParams,
    append_github_env,
    build_chat_kwargs,
    dump_jsonl,
    dump_score,
    finish_stats,
    load_gated,
    make_client,
    run_parallel,
    select_rows,
    self_judge,
    strip_think,
    token_stats,
    truncation_stats,
)

# Official HLE format + LLM judge: elicit a marked answer, then have the judge
# extract and compare it. Grades multipleChoice and exactMatch uniformly (the
# old regex/exact-match grader never matched the real answer_type values and
# scored ~0). This is the system turn, not a format template — its braces are
# literal.
HLE_SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)
# Official HLE judge prompt (structured: extracted_final_answer, reasoning,
# correct: yes/no, confidence). No external judge key in CI, so the model
# self-judges (like AA-Omniscience); approximate.
JUDGE_PROMPT = (
    "Judge whether the following [response] to [question] is correct or not "
    "based on the precise and unambiguous [correct_answer] below.\n"
    "[question]: {question}\n"
    "[response]: {response}\n"
    "extracted_final_answer: ...\n"
    "[correct_answer]: {correct_answer}\n"
    "reasoning: ...\n"
    "correct: yes/no\n"
    "confidence: 0%-100%"
)


def build_messages(
    question: str, root_preamble: str, system_prompt: str
) -> list[dict[str, str]]:
    """Builds the chat messages for one question (root turn omitted when empty)."""
    messages: list[dict[str, str]] = []
    if root_preamble:
        messages.append({"role": "root", "content": root_preamble})
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return messages


def infer(
    client: ChatClient,
    model: str,
    item: tuple[int, dict[str, Any]],
    params: GenParams,
    root_preamble: str,
    system_prompt: str,
) -> dict[str, Any]:
    """Runs one question then self-judges the answer against the reference."""
    prompt_index, sample = item
    question = sample["question"]
    reference = sample["answer"].strip()
    messages = build_messages(question, root_preamble, system_prompt)
    resp = client.chat.completions.create(
        **build_chat_kwargs(model, messages, params)
    )
    prediction = strip_think(resp.choices[0].message.content)
    correct = self_judge(
        client,
        model,
        JUDGE_PROMPT.format(
            question=question, response=prediction, correct_answer=reference
        ),
        seed=params.seed,
    )
    return {
        "prompt_index": prompt_index,
        "question": question[:120],
        "reference": reference,
        "predicted": prediction[:200],
        "verdict": "yes" if correct else "no",
        "correct": correct,
        "finish_reason": resp.choices[0].finish_reason,
        # Count only the model-under-test's answer call; the judge call is NOT
        # counted toward output-token stats.
        "completion_tokens": (
            resp.usage.completion_tokens if resp.usage else 0
        ),
        "reasoning_tokens": (
            resp.usage.completion_tokens_details.reasoning_tokens
            if resp.usage and resp.usage.completion_tokens_details
            else None
        ),
    }


def load_hle_dataset() -> list[dict[str, Any]]:
    """Loads the gated ``cais/hle`` test split; skips image-based questions."""
    ds = load_gated(
        lambda: list(load_dataset("cais/hle", split="test")),
        label="HLE",
        dataset_id="cais/hle",
    )
    return [s for s in ds if not s.get("image")]


def score(results: list[dict[str, Any]], total: int) -> dict[str, Any]:
    """Computes accuracy + token stats over the full dataset.

    Scored over ``total`` (the full dataset) so errors/timeouts count as
    incorrect.
    """
    correct = sum(1 for r in results if r.get("correct"))
    accuracy = correct / total if total else 0.0
    mean_output_tokens, p50_output_tokens = token_stats(results)
    truncated, mean_finished, p50_finished = truncation_stats(results)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "mean_output_tokens": mean_output_tokens,
        "p50_output_tokens": p50_output_tokens,
        "truncated": truncated,
        "mean_output_tokens_finished": mean_finished,
        "p50_output_tokens_finished": p50_finished,
        **finish_stats(results),
    }


def run_eval(
    client: ChatClient,
    indexed_dataset: list[tuple[int, dict[str, Any]]],
    model: str,
    workers: int,
    params: GenParams,
    root_preamble: str,
    system_prompt: str,
    out_dir: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Runs HLE over ``indexed_dataset`` (one pass) and returns results + score.

    A failed/timed-out request is recorded as an incorrect row, never dropped.
    When ``out_dir`` is set, results stream to ``results.jsonl`` as they
    complete, so a crash mid-run doesn't lose already-finished samples.

    Returns:
        A ``(results, score)`` tuple.
    """
    print(f"HLE: evaluating {len(indexed_dataset)} questions")

    def fn(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        return infer(client, model, item, params, root_preamble, system_prompt)

    def on_error(
        item: tuple[int, dict[str, Any]], exc: Exception
    ) -> dict[str, Any]:
        prompt_index, sample = item
        return {
            "prompt_index": prompt_index,
            "question": sample["question"][:120],
            "error": str(exc),
            "correct": False,
        }

    results, _errors = run_parallel(
        indexed_dataset, fn, on_error, workers, "HLE", out_dir=out_dir
    )
    return results, score(results, len(indexed_dataset))


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
    help="Max questions (evenly sampled). Empty = full dataset. Mutually "
    "exclusive with --row-ids.",
)
@click.option(
    "--row-ids",
    default=None,
    help="Explicit comma-separated dataset row indices to evaluate, e.g. "
    "'3,17,42' (indices into the full dataset, order/duplicates preserved). "
    "Mutually exclusive with --sample-size — use this to re-run exactly the "
    "questions that errored or looked wrong last time.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Per-request seed for reproducibility (omitted when unset).",
)
@click.option(
    "--workers", type=int, default=16, help="Max concurrent requests."
)
@click.option("--out-dir", default="/tmp/hle-results", help="Output directory.")
@click.option("--max-tokens", type=int, default=128000, show_default=True)
@click.option("--temperature", type=float, default=1.0, show_default=True)
@click.option("--top-p", type=float, default=0.95, show_default=True)
@click.option(
    "--root-preamble",
    default=DEFAULT_ROOT_PREAMBLE,
    help="Root identity turn (empty to omit; default is MiniMax-M3's).",
)
@click.option("--system-prompt", default=HLE_SYSTEM_PROMPT, show_default=True)
@click.option(
    "--metric-prefix",
    default="HLE",
    help="Prefix for the GITHUB_ENV metric keys the job summary reads.",
)
def main(
    base_url: str,
    model: str,
    sample_size: int | None,
    row_ids: str | None,
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
    """Runs HLE against a running OpenAI-compatible server and scores it."""
    client = make_client(base_url)
    indexed_dataset = select_rows(load_hle_dataset(), sample_size, row_ids)
    params = GenParams(
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed
    )
    results, summary = run_eval(
        client,
        indexed_dataset,
        model,
        workers,
        params,
        root_preamble,
        system_prompt,
        out_dir=out_dir,
    )
    dump_jsonl(out_dir, results)
    dump_score(out_dir, summary)
    print(
        f"HLE: {summary['accuracy']:.4f} "
        f"({summary['correct']}/{summary['total']}) "
        f"mean_output_tokens={summary['mean_output_tokens']} "
        f"p50_output_tokens={summary['p50_output_tokens']}"
    )
    append_github_env(
        metric_prefix,
        summary["accuracy"],
        summary["mean_output_tokens"],
        summary["p50_output_tokens"],
    )


if __name__ == "__main__":
    main()
