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
"""AA-LCR (Artificial Analysis Long-Context Reasoning) eval.

De-embedded from ``minimaxM3NonAgenticTextDatasetEval.yaml`` so the eval logic
lives in a tested, locally-runnable module instead of an inline YAML heredoc.
100 hard questions, each reasoning across a document set averaging ~100k input
tokens. The dataset rows do NOT embed document text — it is published separately
as ``extracted_text/AA-LCR_extracted-text.zip`` in the same HF repo (PDFs
already parsed to ``.txt``). We rebuild each question's context from that zip —
reproducible, no fragile runtime PDF fetch. An LLM self-judge (the same served
model, no external judge key in CI) grades each answer yes/no.

The context builder (:func:`build_doc_index`/:func:`build_context`) is pure and
the driver (:func:`run_eval`) takes an injected client and a prebuilt document
index, so they unit-test without a server, network, or the real zip. Shared
scaffolding lives in :mod:`eval_common`.

Run locally against a server on ``localhost:8000``::

    ./bazelw run //max/tests/integration/accuracy/model_evals:aa_lcr_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --sample-size 2 --out-dir /tmp/aa-lcr

Results stream to ``<out-dir>/results.jsonl`` as each request completes, so a
killed/crashed run keeps everything already finished. To debug specific
questions (e.g. re-run exactly the ones that errored or looked wrong last
time), pass ``--row-ids`` instead of ``--sample-size``::

    ./bazelw run //max/tests/integration/accuracy/model_evals:aa_lcr_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --row-ids 3,17,42 --out-dir /tmp/aa-lcr-debug
"""

from __future__ import annotations

import unicodedata
import zipfile
from typing import Any

import click
from datasets import load_dataset
from eval_common import (
    ChatClient,
    GenParams,
    build_chat_kwargs,
    exact_match_score,
    expand_repeats,
    make_client,
    run_parallel,
    select_rows,
    self_judge,
    strip_think,
    write_outputs,
)
from huggingface_hub import hf_hub_download

AA_LCR_SYSTEM = (
    "You are an expert analyst. Answer the question using only the "
    "provided documents. Reason carefully across all documents, then "
    "state your final answer on the last line after 'Answer:'. If the "
    "answer is a ranked list, give it in the requested order."
)
# metric `correct,all`: an LLM judge grades each answer correct/incorrect
# (answers range from single values to ranked lists, so neither exact nor
# substring match works). No external judge key in CI, so the model under test
# self-judges (same approach as the HLE and AA-Omniscience steps).
JUDGE_PROMPT = (
    "Judge whether the [response] correctly answers the [question] based "
    "on the [correct_answer] below. The correct answer may be a single "
    "value, a set of entities, or a ranked list; the response is correct "
    "if it conveys the same answer (same items, and same order when an "
    "order is requested), allowing for paraphrase and formatting "
    "differences. Minor extra explanation is fine.\n"
    "[question]: {question}\n"
    "[response]: {response}\n"
    "[correct_answer]: {correct_answer}\n\n"
    "Reply with exactly one line:\n"
    "correct: yes/no"
)


def real_name(name: str) -> str:
    """Recovers the true UTF-8 member name from a cp437-decoded zip entry.

    ``zipfile`` decodes non-UTF8-flagged member names as cp437, mangling the
    many non-ASCII document filenames (em-dashes, curly quotes, ş).
    """
    try:
        return name.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def norm(s: str) -> str:
    """NFC-normalizes and strips a filename so both sides of a lookup match."""
    return unicodedata.normalize("NFC", s).strip()


def build_doc_index(zip_path: str) -> dict[tuple[str, str], str]:
    """Reads every ``.txt`` document from the context zip into a lookup map.

    Read single-threaded here: ``zipfile.ZipFile`` is not thread-safe (all
    reads share one file position), but the workers run concurrently.
    Pre-loading also decompresses each unique document only once instead of up
    to ``repeats`` times.

    Returns:
        A ``(document_set_id, normalized_basename) -> text`` map.
    """
    doc_text: dict[tuple[str, str], str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for entry in zf.namelist():
            if not entry.endswith(".txt"):
                continue
            # lcr/<Category>/<docset>/<file>.txt
            parts = real_name(entry).split("/")
            if len(parts) >= 3:
                doc_text[(parts[-2], norm(parts[-1]))] = zf.read(entry).decode(
                    "utf-8", "replace"
                )
    return doc_text


def build_context(
    sample: dict[str, Any], doc_text: dict[tuple[str, str], str]
) -> tuple[str, list[str]]:
    """Assembles a question's document context from the prebuilt index.

    Returns:
        A ``(context_text, missing_filenames)`` tuple.
    """
    docset = sample["document_set_id"]
    filenames = [
        f.strip()
        for f in sample["data_source_filenames"].split(";")
        if f.strip()
    ]
    docs: list[str] = []
    missing: list[str] = []
    for fn in filenames:
        text = doc_text.get((docset, norm(fn)))
        if text is None:
            missing.append(fn)
            continue
        docs.append(f"=== Document: {fn} ===\n{text}")
    return "\n\n".join(docs), missing


def download_context_zip() -> str:
    """Downloads the AA-LCR extracted-text zip and returns its local path."""
    return hf_hub_download(
        "ArtificialAnalysis/AA-LCR",
        "extracted_text/AA-LCR_extracted-text.zip",
        repo_type="dataset",
    )


def infer(
    client: ChatClient,
    model: str,
    item: tuple[int, int, dict[str, Any]],
    params: GenParams,
    system_prompt: str,
    doc_text: dict[tuple[str, str], str],
) -> dict[str, Any]:
    """Runs one long-context question then self-judges against the reference."""
    repeat_index, prompt_index, sample = item
    question = sample["question"]
    reference = str(sample["answer"]).strip()
    context, missing = build_context(sample, doc_text)
    prompt = f"{context}\n\n=== Question ===\n{question}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat.completions.create(
        **build_chat_kwargs(model, messages, params)
    )
    choice = resp.choices[0]
    # Count only the model-under-test's answer call; the judge call is NOT
    # counted toward output-token stats.
    answer_tokens = resp.usage.completion_tokens if resp.usage else 0
    prediction = strip_think(choice.message.content or "")
    correct = self_judge(
        client,
        model,
        JUDGE_PROMPT.format(
            question=question,
            response=prediction[:4000],
            correct_answer=reference,
        ),
        seed=params.seed,
    )
    return {
        "question_id": sample["question_id"],
        "repeat_index": repeat_index,
        "prompt_index": prompt_index,
        "document_set_id": sample["document_set_id"],
        "question": question[:200],
        "reference": reference,
        "prediction": prediction[:500],
        "verdict": "yes" if correct else "no",
        "correct": correct,
        "missing_docs": missing,
        "finish_reason": choice.finish_reason,
        "completion_tokens": answer_tokens,
    }


def run_eval(
    client: ChatClient,
    indexed_dataset: list[tuple[int, dict[str, Any]]],
    model: str,
    repeats: int,
    workers: int,
    params: GenParams,
    system_prompt: str,
    doc_text: dict[tuple[str, str], str],
    out_dir: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Runs AA-LCR over ``indexed_dataset`` with ``repeats``.

    A failed/timed-out request is recorded as an incorrect row, never dropped.
    When ``out_dir`` is set, results stream to ``results.jsonl`` as they
    complete, so a crash mid-run doesn't lose already-finished samples.

    Returns:
        A ``(results, score)`` tuple.
    """
    samples = expand_repeats(indexed_dataset, repeats)
    print(
        f"AA-LCR: evaluating {len(indexed_dataset)} questions x {repeats} "
        f"repeats = {len(samples)} total samples (self-judge)"
    )

    def fn(item: tuple[int, int, dict[str, Any]]) -> dict[str, Any]:
        return infer(client, model, item, params, system_prompt, doc_text)

    def on_error(
        item: tuple[int, int, dict[str, Any]], exc: Exception
    ) -> dict[str, Any]:
        rep, qi, q = item
        return {
            "question_id": q["question_id"],
            "repeat_index": rep,
            "prompt_index": qi,
            "document_set_id": q["document_set_id"],
            "question": q["question"][:200],
            "error": str(exc),
            "correct": False,
        }

    results, errors = run_parallel(
        samples, fn, on_error, workers, "AA-LCR", out_dir=out_dir
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
    "questions that errored or looked wrong last time.",
)
@click.option("--repeats", type=int, default=5, help="Repeats per question.")
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Per-request seed for reproducibility (omitted when unset).",
)
@click.option(
    "--workers",
    type=int,
    default=4,
    help="Max concurrent requests (modest: ~100k-token contexts).",
)
@click.option(
    "--out-dir", default="/tmp/aa-lcr-results", help="Output directory."
)
@click.option("--max-tokens", type=int, default=30000, show_default=True)
@click.option("--temperature", type=float, default=1.0, show_default=True)
@click.option("--top-p", type=float, default=0.95, show_default=True)
@click.option("--system-prompt", default=AA_LCR_SYSTEM, show_default=True)
@click.option(
    "--metric-prefix",
    default="AA_LCR",
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
    system_prompt: str,
    metric_prefix: str,
) -> None:
    """Runs AA-LCR against a running OpenAI-compatible server and scores it."""
    client = make_client(base_url)
    doc_text = build_doc_index(download_context_zip())
    indexed_dataset = select_rows(
        list(load_dataset("ArtificialAnalysis/AA-LCR", split="test")),
        sample_size,
        row_ids,
    )
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
        system_prompt,
        doc_text,
        out_dir=out_dir,
    )
    write_outputs(out_dir, results, summary, metric_prefix, label="AA-LCR")


if __name__ == "__main__":
    main()
