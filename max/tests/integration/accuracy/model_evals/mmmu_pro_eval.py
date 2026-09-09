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
"""MMMU-Pro multimodal multiple-choice eval against an OpenAI-compatible server.

De-embedded from ``minimaxM3MultiModalDatasetEval.yaml`` so the eval logic lives
in a locally-runnable module instead of an inline YAML heredoc. Runs the
``standard (10 options)`` and/or ``vision`` configs (separate flags), scores each
by exact letter match, and writes the per-config ``score.json`` plus the overall
``summary.json`` and the ``MMMU_PRO_*`` CI metric keys the job summary reads.
Shared scaffolding (client construction, ``<think>`` stripping, chat-kwargs
assembly, parallel execution) lives in :mod:`eval_common`; the multi-choice
answer parser is kept local because it is richer than the shared MCQ helper.

Run locally against a server on ``localhost:8000``::

    ./bazelw run //max/tests/integration/accuracy/model_evals:mmmu_pro_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --run-standard --run-vision --limit 4 --out-dir /tmp/mmmu-pro-results

Results stream to ``<out-dir>/<config>/results.jsonl`` as each request
completes, so a killed/crashed run keeps everything already finished. To
debug specific samples (e.g. re-run exactly the ones that errored or looked
wrong last time), pass ``--row-ids`` instead of ``--limit``::

    ./bazelw run //max/tests/integration/accuracy/model_evals:mmmu_pro_eval -- \\
        --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \\
        --run-vision --row-ids 3,17,42 --out-dir /tmp/mmmu-pro-debug
"""

from __future__ import annotations

import ast
import base64
import glob
import io
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import click
from datasets import load_dataset
from eval_common import (
    ChatClient,
    GenParams,
    build_chat_kwargs,
    dump_score,
    make_client,
    run_parallel,
    select_rows,
    strip_think,
    token_stats,
    truncation_stats,
)
from PIL import Image


def img_to_b64(img: Image.Image) -> str:
    """Encodes a PIL image as a base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def img_part(img: Image.Image) -> dict[str, Any]:
    """Wraps a PIL image as an OpenAI ``image_url`` content part (data URI)."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{img_to_b64(img)}"},
    }


def parse_multi_choice(
    response: str, all_choices: list[str], index2ans: dict[str, str]
) -> str:
    """Extracts the chosen letter (official MMMU-Pro ``parse_multi_choice_response``).

    Restricts to the sample's actual option letters, prefers the text after the
    last "Answer:" cue, then matches ``(X)`` / ``X.`` / ``X`` forms, then falls
    back to matching the option *text*, tie-breaking on the LAST occurrence. No
    ``raw[:1]`` garbage fallback: an unparseable response scores wrong (empty),
    never noise. Robust to M3's long answer-first reasoning, which the old
    ``[A-J]`` + first-char parser mis-graded.
    """
    if not response or not response.strip():
        return ""
    # Normalize markdown emphasis so "**A**"/"`A`" expose a delimiter.
    resp = re.sub(r"[*`#]", " ", response)
    for ch in [",", ".", "!", "?", ";", ":", "'"]:
        resp = resp.strip(ch)
    # Anchor on the final committed answer when a cue is present.
    cues = list(
        re.finditer(
            r"(?:final answer|answer|correct choice|correct option)\s*:?\s*",
            resp,
            re.IGNORECASE,
        )
    )
    if cues:
        resp = resp[cues[-1].end() :]
    resp = " " + resp + " "
    candidates: list[str] = []
    ans_with_brack = False
    index_ans = True
    for c in all_choices:
        if f"({c})" in resp:
            candidates.append(c)
            ans_with_brack = True
    if not candidates:
        for c in all_choices:
            if f" {c} " in resp:
                candidates.append(c)
    if not candidates:
        for c in all_choices:
            if f"{c}." in resp or f"{c})" in resp or f"{c}:" in resp:
                candidates.append(c)
    if not candidates and len(resp.split()) > 5:
        for c, ans in index2ans.items():
            if ans and ans.lower() in resp.lower():
                candidates.append(c)
                index_ans = False
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    # Multiple candidates: take the last occurrence in the response.
    starts = []
    for c in candidates:
        if index_ans:
            key = f"({c})" if ans_with_brack else f" {c} "
            starts.append(resp.rfind(key))
        else:
            starts.append(resp.lower().rfind(index2ans[c].lower()))
    return candidates[starts.index(max(starts))]


def build_content(
    sample: dict[str, Any], is_vision: bool
) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    """Builds the user content parts and the grading maps for one sample.

    Returns a ``(content, all_choices, index2ans)`` tuple. For the vision config
    the whole question + options are rendered into a single image; for the
    standard config each image is interleaved at its ``<image N>`` token position
    (official ``replace_images_tokens`` / ``make_interleave_content``) and the
    per-sample letter→text map is built for text-fallback grading.
    """
    content: list[dict[str, Any]] = []
    if is_vision:
        content.append(img_part(sample["image"]))
        content.append({"type": "text", "text": "Answer with the letter only."})
        all_choices: list[str] = [chr(65 + i) for i in range(10)]
        return content, all_choices, {}

    q = sample["question"]
    placed: set[int] = set()
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            content.append({"type": "text", "text": buf})
        buf = ""

    for piece in re.split(r"(<image\s+\d+>)", q):
        m = re.fullmatch(r"<image\s+(\d+)>", piece)
        if m:
            idx = int(m.group(1))
            img = sample.get(f"image_{idx}")
            if img is not None:
                flush()
                content.append(img_part(img))
                placed.add(idx)
        else:
            buf += piece
    flush()
    # Build options and the per-sample letter->text map for grading.
    # MMMU-Pro stores `options` as a stringified Python list (e.g.
    # "['opt a', 'opt b', ...]"); parse it to a real list first, otherwise
    # enumerate() walks the string CHARACTER by character and feeds the model
    # garbage one-char choices.
    raw_options = sample["options"]
    if isinstance(raw_options, str):
        raw_options = ast.literal_eval(raw_options)
    all_choices = []
    index2ans: dict[str, str] = {}
    opt_lines: list[str] = []
    for i, opt in enumerate(raw_options):
        if not opt:
            continue
        letter = chr(65 + i)
        all_choices.append(letter)
        index2ans[letter] = str(opt)
        opt_lines.append(f"{letter}. {opt}")
    # Any non-null image not referenced inline (e.g. tokens in options) is
    # appended so no visual context is dropped.
    for i in range(1, 8):
        img = sample.get(f"image_{i}")
        if img is not None and i not in placed:
            content.append(img_part(img))
    content.append(
        {
            "type": "text",
            "text": "\n".join(opt_lines) + "\n\nAnswer with the letter only.",
        }
    )
    return content, all_choices, index2ans


def infer(
    client: ChatClient,
    model: str,
    sample: dict[str, Any],
    prompt_index: int,
    is_vision: bool,
    params: GenParams,
) -> dict[str, Any]:
    """Runs and grades a single MMMU-Pro sample."""
    content, all_choices, index2ans = build_content(sample, is_vision)
    resp = client.chat.completions.create(
        **build_chat_kwargs(
            model, [{"role": "user", "content": content}], params
        )
    )
    choice = resp.choices[0]
    raw = strip_think(choice.message.content)
    predicted = parse_multi_choice(raw, all_choices, index2ans)
    completion_tokens = resp.usage.completion_tokens if resp.usage else 0
    # Derive finish_reason: use the API field when available, fall back to
    # detecting max-token truncation via completion_tokens hitting the cap.
    finish_reason = choice.finish_reason
    if not finish_reason:
        finish_reason = (
            "length" if completion_tokens >= params.max_tokens else "stop"
        )
    return {
        "prompt_index": prompt_index,
        "id": sample["id"],
        "predicted": predicted,
        "ground_truth": sample["answer"],
        "raw": raw[:800],
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
    }


def load_mmmu_pro(hf_config: str) -> Any:
    """Loads one MMMU-Pro config, retrying through a corrupt shared HF cache.

    The shared runner HF cache races under concurrent jobs and can delete a
    ``*.incomplete``/``tmp_*`` blob mid-download (``FileNotFoundError`` during
    ``_chmod_and_move``). Purge partials and force a clean re-fetch on retry.
    """
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            return load_dataset(
                "MMMU/MMMU_Pro",
                hf_config,
                split="test",
                download_mode="force_redownload" if attempt else None,
            )
        except (FileNotFoundError, OSError) as e:
            last_err = e
            for pat in ("*.incomplete", "tmp_*"):
                for inc in glob.glob(
                    os.path.expanduser(
                        "~/.cache/huggingface/hub/datasets--MMMU--MMMU_Pro/blobs/"
                        + pat
                    )
                ):
                    try:
                        os.remove(inc)
                    except OSError:
                        pass
            print(
                f"  {hf_config}: load_dataset failed "
                f"(attempt {attempt + 1}/5): {e}"
            )
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not load MMMU_Pro {hf_config}: {last_err}")


def run_config(
    client: ChatClient,
    model: str,
    hf_config: str,
    out_dir: Path,
    limit: int | None,
    row_ids: str | None,
    params: GenParams,
    workers: int,
) -> dict[str, Any]:
    """Runs one MMMU-Pro config and writes its ``results.jsonl`` + ``score.json``.

    Results stream to ``<out_dir>/results.jsonl`` as each request completes
    (:func:`eval_common.run_parallel`'s ``out_dir``), so a crash mid-run
    doesn't lose already-finished samples.

    Returns a ``{"accuracy", "output_tokens"}`` dict; the output-token list is
    returned (not just its mean) so the caller can combine token lists across
    configs for the overall stats.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    indexed_dataset = select_rows(
        list(load_mmmu_pro(hf_config)), limit, row_ids
    )
    is_vision = hf_config == "vision"

    def fn(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, sample = item
        return infer(client, model, sample, idx, is_vision, params)

    def on_error(
        item: tuple[int, dict[str, Any]], exc: Exception
    ) -> dict[str, Any]:
        idx, _ = item
        # Count a failed/timed-out request as wrong rather than dropping it (the
        # sentinel ground truth can never match a real answer).
        return {
            "prompt_index": idx,
            "predicted": "",
            "ground_truth": "__ERROR__",
            "error": str(exc),
        }

    results, errors = run_parallel(
        indexed_dataset, fn, on_error, workers, hf_config, out_dir=str(out_dir)
    )

    out_dir.joinpath("results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results)
    )
    # Score over the selected rows so errors count as incorrect.
    total = len(indexed_dataset)
    correct = sum(1 for r in results if r["predicted"] == r["ground_truth"])
    accuracy = correct / total if total else 0.0
    otoks = [
        r["completion_tokens"]
        for r in results
        if "completion_tokens" in r and "error" not in r
    ]
    mean_output_tokens, p50_output_tokens = token_stats(results)
    truncated, _mean_finished, _p50_finished = truncation_stats(results)
    non_error_count = sum(1 for r in results if "error" not in r)
    stop_ratio = (
        round((non_error_count - truncated) / non_error_count, 4)
        if non_error_count
        else 0.0
    )
    dump_score(
        str(out_dir),
        {
            "config": hf_config,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": errors,
            "mean_output_tokens": mean_output_tokens,
            "p50_output_tokens": p50_output_tokens,
            "truncated": truncated,
            "stop_ratio": round(stop_ratio, 4),
        },
    )
    print(
        f"{hf_config}: {accuracy:.4f} ({correct}/{total}) "
        f"mean_out_tok={mean_output_tokens:.1f} p50_out_tok={p50_output_tokens:.1f} "
        f"stop_ratio={stop_ratio:.4f}"
    )
    return {
        "accuracy": accuracy,
        "output_tokens": otoks,
        "stop_ratio": stop_ratio,
        "non_error_count": non_error_count,
    }


@click.command()
@click.option(
    "--base-url",
    required=True,
    help="Server base URL, e.g. http://localhost:8000",
)
@click.option("--model", required=True, help="Served model name to request.")
@click.option(
    "--run-standard",
    is_flag=True,
    default=False,
    help="Run the standard (10-option) config.",
)
@click.option(
    "--run-vision", is_flag=True, default=False, help="Run the vision config."
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Max samples per config (evenly sampled). Empty = full dataset. "
    "Mutually exclusive with --row-ids.",
)
@click.option(
    "--row-ids",
    default=None,
    help="Explicit comma-separated dataset row indices to evaluate, e.g. "
    "'3,17,42' (indices into the full per-config dataset, order/duplicates "
    "preserved). Applied identically to --run-standard and --run-vision when "
    "both are set. Mutually exclusive with --limit — use this to re-run "
    "exactly the samples that errored or looked wrong last time.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Per-request seed for reproducibility (omitted when unset).",
)
@click.option(
    "--workers", type=int, default=32, help="Max concurrent requests."
)
@click.option(
    "--out-dir", default="/tmp/mmmu-pro-results", help="Output directory."
)
@click.option("--max-tokens", type=int, default=65536, show_default=True)
@click.option("--temperature", type=float, default=1.0, show_default=True)
@click.option("--top-p", type=float, default=0.95, show_default=True)
@click.option(
    "--metric-prefix",
    default="MMMU_PRO",
    help="Prefix for the GITHUB_ENV metric keys the job summary reads.",
)
@click.option(
    "--baseline",
    type=float,
    default=0.754,
    show_default=True,
    help="Reference accuracy for the PASSED gate (MiniMax-M3's published "
    "score). Override when scoring a different model so PASSED reflects "
    "that model's bar instead of MiniMax-M3's.",
)
def main(
    base_url: str,
    model: str,
    run_standard: bool,
    run_vision: bool,
    limit: int | None,
    row_ids: str | None,
    seed: int | None,
    workers: int,
    out_dir: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    metric_prefix: str,
    baseline: float,
) -> None:
    """Runs MMMU-Pro against a running OpenAI-compatible server and scores it."""
    client = make_client(base_url)
    params = GenParams(
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed
    )
    out_root = Path(out_dir)

    results_by_config: dict[str, dict[str, Any]] = {}
    if run_standard:
        results_by_config["standard"] = run_config(
            client,
            model,
            "standard (10 options)",
            out_root / "standard",
            limit,
            row_ids,
            params,
            workers,
        )
    if run_vision:
        results_by_config["vision"] = run_config(
            client,
            model,
            "vision",
            out_root / "vision",
            limit,
            row_ids,
            params,
            workers,
        )

    scores = {k: v["accuracy"] for k, v in results_by_config.items()}
    # Overall is the mean over whichever configs actually ran.
    overall = sum(scores.values()) / len(scores) if scores else 0.0
    # Combine the per-config token lists (NOT the per-config means) for overall.
    all_otoks = [
        t for v in results_by_config.values() for t in v["output_tokens"]
    ]
    mean_output_tokens = (
        round(statistics.mean(all_otoks), 1) if all_otoks else 0.0
    )
    p50_output_tokens = (
        round(statistics.median(all_otoks), 1) if all_otoks else 0.0
    )
    # Overall stop ratio: weighted by non-error sample count per config.
    total_stopped = sum(
        v["stop_ratio"] * v["non_error_count"]
        for v in results_by_config.values()
    )
    total_non_error = sum(
        v["non_error_count"] for v in results_by_config.values()
    )
    overall_stop_ratio = (
        round(total_stopped / total_non_error, 4) if total_non_error else 0.0
    )
    threshold = baseline * 0.98
    passed = bool(scores) and overall >= threshold
    summary = {
        **scores,
        "overall": overall,
        "mean_output_tokens": mean_output_tokens,
        "p50_output_tokens": p50_output_tokens,
        "stop_ratio": overall_stop_ratio,
        "baseline": baseline,
        "threshold": threshold,
        "passed": passed,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_root / "summary.json", "w"), indent=2)
    print(
        f"MMMU-Pro overall: {overall:.4f} "
        f"mean_out_tok={mean_output_tokens:.1f} p50_out_tok={p50_output_tokens:.1f} "
        f"stop_ratio={overall_stop_ratio:.4f}"
    )

    env_file = os.environ.get("GITHUB_ENV")
    if env_file:
        with open(env_file, "a") as f:
            if "standard" in scores:
                f.write(f"{metric_prefix}_STANDARD={scores['standard']:.4f}\n")
            if "vision" in scores:
                f.write(f"{metric_prefix}_VISION={scores['vision']:.4f}\n")
            f.write(f"{metric_prefix}_OVERALL={overall:.4f}\n")
            f.write(f"{metric_prefix}_MEAN_TOKENS={mean_output_tokens:.0f}\n")
            f.write(f"{metric_prefix}_P50_TOKENS={p50_output_tokens:.0f}\n")
            f.write(f"{metric_prefix}_STOP_RATIO={overall_stop_ratio:.4f}\n")
            f.write(f"{metric_prefix}_PASSED={'true' if passed else 'false'}\n")


if __name__ == "__main__":
    main()
