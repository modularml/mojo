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

"""Video-MMMU evaluator for an OpenAI-compatible MAX Serve endpoint.

Runs the Video-MMMU benchmark (Adaptation/Comprehension/Perception) against any
served multimodal model that exposes the OpenAI chat-completions API. The server
owns mp4 decoding and frame sampling; this harness only submits each clip (as
base64 or a URL) plus the official Video-MMMU prompt and grades the response
rule-based, so it is model-agnostic.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.parse
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import click
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm

HF_TOKEN = os.environ.get("HF_TOKEN")

# Offline mode reads the already-cached raw parquet + extracted mp4s from the
# HF hub cache instead of resolving the (gated) lmms-lab/VideoMMMU repo. Enabled
# by MMMU_OFFLINE=1 or automatically when no HF_TOKEN is present. The measured
# workload is byte-identical to the online path: same canonical parquet rows
# (same question IDs / counts) and the same extracted clips — only the source of
# the files changes. Gated so CI's online path is preserved.
MMMU_OFFLINE = os.environ.get("MMMU_OFFLINE") == "1" or not HF_TOKEN

# Root of the local HF hub cache holding the raw VideoMMMU snapshot.
HF_HUB_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
VIDEOMMMU_REPO_DIR = os.path.join(HF_HUB_CACHE, "datasets--lmms-lab--VideoMMMU")

# Local HTTP port used only for "url" input mode when no external base URL is
# supplied (internal detail, not a user-facing knob).
VIDEO_HTTP_PORT = 8199

# Video-MMMU stores the lecture clips as per-domain zip archives in the dataset
# repo (Art/Business/Engineering/Humanities/Medicine/Science.zip), NOT as bytes
# embedded in the parquet rows. Each row only carries id/question/options/answer/
# question_type; the clip is a separate <id>.mp4 with the question image baked in
# as the final frame. Download + unzip the archives, then resolve each row's clip
# by id. Mirrors lmms-eval's videommmu task (dataset_kwargs: video, cache_dir).
VIDEO_ROOT = os.path.expanduser("~/.cache/huggingface/videommmu_videos")


@dataclass(frozen=True)
class EvalConfig:
    """Per-run configuration threaded through the inference workers."""

    client: OpenAI
    model: str
    video_index: dict[str, str]
    video_input_mode: str
    video_server_base: str
    max_tokens: int
    temperature: float
    top_p: float
    # Server-side frame-sampling hints attached to the video part when set.
    # Some heads honor them (e.g. MiniMax-M3: fps=1.0, max_frames=512); others
    # ignore them (e.g. gemma4 samples a fixed frame count). Omitted when None.
    video_fps: float | None
    video_max_frames: int | None


# ---- Official Video-MMMU prompts (lmms-eval _default_template_yaml) ----
PRE_PROMPT = "You should watch and learn the video content. Then apply what you learned to "
MCQ_PROMPT = "answer the following multi-choice question. The image for this question is at the end of the video.\n"
OPEN_PROMPT = "answer the following open-ended question. The image for this question is at the end of the video.\n"
PC_PROMPT = "\nPlease ignore the Quiz question in last frame of the video."
# Format-constraint post-prompts. The official methodology pairs the bare
# prompt with an LLM judge; we grade rule-based, so (like the MMMU-Pro step
# in this workflow) we constrain the answer line for reliable parsing.
MCQ_POST = "\n\nAnswer with the letter only."
OPEN_POST = "\n\nEnd your response with your final answer on the last line."


def fetch_zip(name: str) -> str:
    # Returns a local path to a valid zip, re-downloading past a corrupt or
    # missing cache blob on retry.
    last = None
    for attempt in range(3):
        try:
            zp = hf_hub_download(
                "lmms-lab/VideoMMMU",
                name,
                repo_type="dataset",
                token=HF_TOKEN,
                force_download=attempt > 0,
            )
            with zipfile.ZipFile(zp):  # validate central directory
                pass
            return zp
        except (FileNotFoundError, OSError, zipfile.BadZipFile) as e:
            last = e
            print(
                f"  {name}: fetch/validate failed "
                f"(attempt {attempt + 1}/3): {e}"
            )
    raise RuntimeError(f"could not fetch {name}: {last}")


def index_videos() -> dict[str, str]:
    # Index every extracted clip by basename so resolution doesn't depend on
    # the archive's folder layout.
    video_index = {}
    for root, _, files in os.walk(VIDEO_ROOT):
        for fn in files:
            if fn.endswith(".mp4"):
                video_index[fn] = os.path.join(root, fn)
    return video_index


def prepare_videos() -> dict[str, str]:
    if MMMU_OFFLINE:
        # Clips are already extracted under VIDEO_ROOT (identical bytes to the
        # online path); just index them. No repo resolution, no download.
        video_index = index_videos()
        if not video_index:
            raise RuntimeError(
                f"offline mode: no extracted mp4s under {VIDEO_ROOT}"
            )
        print(f"Indexed {len(video_index)} lecture videos (offline)")
        return video_index
    # Download + extract the per-domain lecture archives, then index every clip
    # by basename so we don't depend on the archive's folder layout.
    os.makedirs(VIDEO_ROOT, exist_ok=True)
    print("Downloading Video-MMMU lecture archives...")
    # Discover the per-domain zips by name (no blob download), then fetch each
    # individually with retries. snapshot_download(allow_patterns=["*.zip"]) was
    # flaky here — a partially-downloaded archive left a dangling symlink that
    # glob still listed, so ZipFile() died with FileNotFoundError. Fetching per
    # file lets us force a re-download past a corrupt/missing cache blob.
    zip_names = sorted(
        os.path.basename(f)
        for f in list_repo_files(
            "lmms-lab/VideoMMMU", repo_type="dataset", token=HF_TOKEN
        )
        if f.endswith(".zip")
        and os.path.basename(f) != "question_only_videos.zip"
    )
    print(f"  domain archives: {zip_names}")

    for name in zip_names:
        marker = os.path.join(VIDEO_ROOT, "." + name + ".extracted")
        if os.path.exists(marker):
            continue
        print(f"  fetching + extracting {name}...")
        with zipfile.ZipFile(fetch_zip(name)) as z:
            z.extractall(VIDEO_ROOT)
        open(marker, "w").close()

    video_index = index_videos()
    print(f"Indexed {len(video_index)} lecture videos")
    return video_index


def start_local_video_server() -> str:
    # Host the already-extracted clips over http so the server
    # exercises its URL download path (resolve_image_from_url)
    # instead of decoding embedded base64. ThreadingHTTPServer (the
    # http.server CLI default) keeps the parallel fetches from serializing.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(VIDEO_HTTP_PORT),
            "--bind",
            "127.0.0.1",
            "--directory",
            VIDEO_ROOT,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        try:
            with socket.create_connection(
                ("127.0.0.1", VIDEO_HTTP_PORT), timeout=0.5
            ):
                break
        except OSError:
            time.sleep(0.1)
    else:
        proc.terminate()
        raise RuntimeError("local video HTTP server did not come up")
    return f"http://127.0.0.1:{VIDEO_HTTP_PORT}"


def video_to_url(path: str, server_base: str) -> str:
    # Reference the clip by URL (relative to the hosting root) so the server
    # downloads and decodes it, same as base64 mode.
    rel = urllib.parse.quote(os.path.relpath(path, VIDEO_ROOT))
    return f"{server_base}/{rel}"


def video_to_data_uri(path: str) -> str:
    # Submit the raw clip; the server decodes the mp4 and samples frames itself
    # (the sampling policy is architecture-specific and owned by the served
    # model's pipeline). Video-MMMU appends the Adaptation question image as the
    # final frame; models that keep the last frame retain the question image.
    with open(path, "rb") as fh:
        data = fh.read()
    return "data:video/mp4;base64," + base64.b64encode(data).decode()


def letters_for(n: int) -> list[str]:
    return [chr(65 + i) for i in range(n)]


def parse_options(options: Sequence[object]) -> str:
    # Mirrors lmms-eval videommmu parse_options.
    letters = letters_for(len(options))
    if all(
        str(o).startswith(f"{l}.")
        for o, l in zip(options, letters, strict=False)
    ):
        return "\n".join(str(o) for o in options)
    return "\n".join(
        f"{l}. {o}" for l, o in zip(letters, options, strict=False)
    )


def is_mcq_doc(doc: dict[str, Any]) -> bool:
    return doc.get("question_type") == "multiple-choice" or bool(
        doc.get("options")
    )


def build_prompt(doc: dict[str, Any], config: str) -> str:
    # Mirrors videommmu_doc_to_text_{adaptation,perception_comprehension}.
    question = doc.get("question", "")
    options = list(doc.get("options") or [])
    if config == "Adaptation":
        if is_mcq_doc(doc):
            pre = PRE_PROMPT + MCQ_PROMPT
            return f"{pre}{question}\n{parse_options(options)}{MCQ_POST}"
        return f"{PRE_PROMPT}{OPEN_PROMPT}{question}{OPEN_POST}"
    # Perception / Comprehension: no pre-prompt; ignore the final-frame quiz.
    return f"{question}\n{parse_options(options)}{PC_PROMPT}{MCQ_POST}"


def clean_content(text: str) -> str:
    # Reasoning is handled SERVER-SIDE by the served model's reasoning parser,
    # not here: when a reasoning parser is active, chain-of-thought is split out
    # of the OpenAI `content` field into `reasoning`/`reasoning_content`, so
    # `content` is already the committed answer. (Running a model with its
    # reasoning parser disabled can leave un-delimited reasoning in `content`
    # that CANNOT be reliably separated client-side — so don't.)
    #
    # Tool-call blocks are best-effort defensive cleanup: some models (e.g.
    # gemma4) keep `<|tool_call> ... <tool_call|>` markers through
    # detokenization and the server only parses them into `tool_calls` when the
    # request carries `tools` (this eval sends none). A stray tool-call block
    # would otherwise pollute answer extraction. The regex is a no-op for models
    # that don't emit these markers.
    #
    # `<think> ... </think>` reasoning blocks are stripped for models served
    # without a reasoning parser (e.g. MiniMax-M3 in the multimodal-eval
    # workflow), which leave the delimited CoT in `content`. No-op for models
    # whose server-side parser already removed it.
    if not text:
        return ""
    text = re.sub(r"<\|tool_call>.*?<tool_call\|>", " ", text, flags=re.DOTALL)
    # Unterminated tool call (hit max_tokens mid-call): drop from the opener on.
    text = re.sub(r"<\|tool_call>.*$", " ", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    return text.strip()


# ---- Answer extraction ----
# NOTE: lmms-eval's extract_mcq_answer assumes short outputs and takes the
# last-occurrence letter. Robust to answer-first reasoning.
def extract_mcq_answer(response: str, choices: Sequence[str]) -> str:
    if not response or not response.strip():
        return ""
    cset = "".join(choices)
    cues = list(
        re.finditer(r"\b(?:answer|correct choice)\b", response, re.IGNORECASE)
    )
    if cues:
        m = re.search(rf"\b([{cset}])\b", response[cues[-1].end() :])
        if m:
            return m.group(1).upper()
    last_line = next(
        (l for l in reversed(response.splitlines()) if l.strip()), ""
    )
    matches = re.findall(rf"\b([{cset}])\b", last_line) or re.findall(
        rf"\b([{cset}])\b", response
    )
    return matches[-1].upper() if matches else ""


def grade_open(response: str, gold: object) -> tuple[str, bool]:
    # Grade open-ended on the FINAL answer only (the model is instructed to
    # end with it): match any accepted gold by 2-dp number equality, else
    # case-insensitive substring. Anchoring on the final answer avoids
    # matching intermediate values anywhere in the long reasoning.
    cues = list(
        re.finditer(r"\b(?:final answer|answer)\b", response, re.IGNORECASE)
    )
    final = (
        response[cues[-1].end() :]
        if cues
        else next((l for l in reversed(response.splitlines()) if l.strip()), "")
    )
    final = final.strip().lower()
    if not final:
        return "", False
    nums = re.findall(r"-?\d+(?:\.\d+)?", final.replace(",", ""))
    golds = gold if isinstance(gold, (list, tuple)) else [gold]
    for g in golds:
        g = str(g).strip().lower()
        try:
            gv = round(float(g.replace(",", "")), 2)
            if any(round(float(n), 2) == gv for n in nums):
                return final[:80], True
        except ValueError:
            if g and g in final:
                return final[:80], True
    return final[:80], False


def infer(cfg: EvalConfig, item: tuple[dict[str, Any], str]) -> dict[str, Any]:
    doc, config = item
    qid = doc["id"]
    options = list(doc.get("options") or [])
    answer = str(doc.get("answer", "")).strip()
    is_mcq = is_mcq_doc(doc)

    video_path = cfg.video_index.get(qid + ".mp4")
    if not video_path:
        return {
            "id": qid,
            "config": config,
            "answer": answer,
            "predicted": "",
            "correct": False,
            "error": f"video not found for id {qid}",
        }
    try:
        video_uri = (
            video_to_url(video_path, cfg.video_server_base)
            if cfg.video_input_mode == "url"
            else video_to_data_uri(video_path)
        )
    except Exception as e:
        return {
            "id": qid,
            "config": config,
            "answer": answer,
            "predicted": "",
            "correct": False,
            "error": f"video read: {e}",
        }

    video_url: dict[str, object] = {"url": video_uri}
    if cfg.video_fps is not None:
        video_url["fps"] = cfg.video_fps
    if cfg.video_max_frames is not None:
        video_url["max_frames"] = cfg.video_max_frames
    content = [
        {"type": "video_url", "video_url": video_url},
        {"type": "text", "text": build_prompt(doc, config)},
    ]
    # `video_url` is a MAX-Serve extension content part that the OpenAI SDK's
    # typed message params don't model, so cast the payload past the type check.
    messages = cast(
        "list[ChatCompletionMessageParam]",
        [{"role": "user", "content": content}],
    )

    # Long videos can exceed the server's max_total_pixels budget at the
    # default 672 long-side tier: the vendor manual specifies error-on-exceed,
    # while MiniMax's own API downscales and answers, so those samples are
    # answerable. On that specific 400, retry at lower long-side tiers
    # (multiples of 28; 504 covers overages to ~1.78x, 336 to ~4x) via the
    # per-video max_long_side_pixel knob instead of structurally failing the
    # sample. Byte-cap (413-style "maximum allowed size") rejections are not
    # retried; resolution does not change encoded size.
    resp = None
    last_err: Exception | None = None
    used_tier: int | None = None
    for tier in (None, 504, 336):
        if tier is not None:
            video_url["max_long_side_pixel"] = tier
        try:
            resp = cfg.client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
            used_tier = tier
            break
        except Exception as e:
            last_err = e
            if "exceeds max_total_pixels" in str(e):
                continue
            break
    if resp is None:
        return {
            "id": qid,
            "config": config,
            "answer": answer,
            "predicted": "",
            "correct": False,
            "error": f"request: {last_err}",
        }

    # The server's reasoning parser (if any) already removed chain-of-thought
    # from `content`; just drop any leaked tool-call block.
    raw = clean_content(resp.choices[0].message.content or "")
    labels = letters_for(len(options))
    if is_mcq:
        predicted = extract_mcq_answer(raw, choices=labels)
        # gold may be a letter ("B") or the option text — normalize to a letter.
        gold = answer.upper()
        if not (len(gold) == 1 and gold in labels):
            gold = next(
                (
                    l
                    for l, o in zip(labels, options, strict=False)
                    if str(o).strip().lower() == answer.strip().lower()
                ),
                gold,
            )
        correct = bool(predicted) and predicted == gold
    else:
        predicted, correct = grade_open(raw, doc.get("answer"))
    result = {
        "id": qid,
        "config": config,
        "answer": answer,
        "predicted": str(predicted),
        "raw": raw[:800],
        "correct": correct,
        "completion_tokens": (
            resp.usage.completion_tokens if resp.usage else 0
        ),
    }
    if used_tier is not None:
        # Answered after downscaling; recorded so score interpretation can
        # distinguish full-resolution answers from downscaled ones.
        result["video_max_long_side"] = used_tier
    return result


def local_snapshot_dir() -> str:
    # Resolve the single cached VideoMMMU snapshot dir under the hub cache
    # without touching the network (no list_repo_files / repo resolution).
    snaps = sorted(
        glob.glob(os.path.join(VIDEOMMMU_REPO_DIR, "snapshots", "*"))
    )
    if not snaps:
        raise RuntimeError(
            f"no cached VideoMMMU snapshot under {VIDEOMMMU_REPO_DIR}; "
            "offline mode requires the raw parquet already in the hub cache"
        )
    # Newest snapshot wins if more than one is present.
    return snaps[-1]


def load_subset_offline(config: str, limit: int | None) -> Dataset:
    # Read the raw per-config parquet straight from the cached snapshot. The
    # parquet's embedded schema is `Sequence` (parseable by datasets==2.21.0),
    # and reading it as the "parquet" builder bypasses both the gated repo and
    # the List-typed processed `.arrow` cache. Rows (ids/counts/answers) are
    # byte-identical to the canonical online dataset.
    files = sorted(
        glob.glob(os.path.join(local_snapshot_dir(), config, "*.parquet"))
    )
    if not files:
        raise RuntimeError(
            f"no cached parquet for config {config} under "
            f"{local_snapshot_dir()}"
        )
    subset = load_dataset("parquet", data_files={"test": files}, split="test")
    assert isinstance(subset, Dataset)
    return _subsample(subset, limit)


def load_subset(config: str, limit: int | None) -> Dataset:
    if MMMU_OFFLINE:
        return load_subset_offline(config, limit)
    # Retry the parquet metadata load: HF Hub can race and leave a vanished
    # *.incomplete temp blob (FileNotFoundError during _chmod_and_move).
    # Purge partials and force a clean re-fetch on retry.
    last = None
    for attempt in range(4):
        try:
            subset = load_dataset(
                "lmms-lab/VideoMMMU",
                config,
                split="test",
                token=HF_TOKEN,
                download_mode="force_redownload" if attempt else None,
            )
            assert isinstance(subset, Dataset)
            break
        except (FileNotFoundError, OSError) as e:
            last = e
            for inc in glob.glob(
                os.path.expanduser(
                    "~/.cache/huggingface/hub/datasets--lmms-lab--VideoMMMU/"
                    "blobs/*.incomplete"
                )
            ):
                try:
                    os.remove(inc)
                except OSError:
                    pass
            print(
                f"  {config}: load_dataset failed "
                f"(attempt {attempt + 1}/4): {e}"
            )
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"could not load {config}: {last}")

    return _subsample(subset, limit)


def _subsample(subset: Dataset, limit: int | None) -> Dataset:
    if limit is not None and limit < len(subset):
        # Evenly spaced subsample, not the first N.
        m = len(subset)
        subset = subset.select([i * m // limit for i in range(limit)])
    return subset


@click.command()
@click.option(
    "--base-url",
    default="http://127.0.0.1:8000/v1",
    show_default=True,
    help="OpenAI-compatible base URL of the served model (include /v1).",
)
@click.option(
    "--model",
    "-m",
    required=True,
    help="Model name to send in the chat-completions request.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Cap on samples per config, for quick smoke runs.",
)
@click.option(
    "--video-input-mode",
    type=click.Choice(["base64", "url"]),
    default="base64",
    show_default=True,
    help="How to hand clips to the server: embed bytes, or pass an http URL.",
)
@click.option(
    "--video-base-url",
    default="",
    help="External base URL hosting the clips (url mode). Empty starts a "
    "local server.",
)
@click.option(
    "--workers",
    type=int,
    default=4,
    show_default=True,
    help="Parallel inference workers (server-side video decode is heavy).",
)
@click.option(
    "--max-tokens",
    type=int,
    default=32768,
    show_default=True,
    help="max_tokens per request.",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    show_default=True,
)
@click.option(
    "--top-p",
    type=float,
    default=0.95,
    show_default=True,
)
@click.option(
    "--video-fps",
    type=float,
    default=None,
    help="Frame-sampling rate hint on the video part. Omitted when unset; "
    "honored only by heads that support it (e.g. MiniMax-M3 uses 1.0).",
)
@click.option(
    "--video-max-frames",
    type=int,
    default=None,
    help="Max frames hint on the video part. Omitted when unset (e.g. "
    "MiniMax-M3 uses 512).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("/tmp/mmmu-video-results"),
    show_default=True,
    help="Directory for results.jsonl and summary.json.",
)
def main(
    base_url: str,
    model: str,
    limit: int | None,
    video_input_mode: str,
    video_base_url: str,
    workers: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    video_fps: float | None,
    video_max_frames: int | None,
    output_dir: Path,
) -> None:
    """Run Video-MMMU against an OpenAI-compatible endpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # No client timeout: a request must only ever end by the server
    # hitting max_tokens, never by a client-side deadline.
    # An unconditional api_key would shadow OPENAI_API_KEY and 401 against an
    # authenticated endpoint.
    client = OpenAI(
        base_url=base_url.rstrip("/"),
        api_key=os.environ.get("OPENAI_API_KEY") or "dummy",
        timeout=None,
    )

    video_index = prepare_videos()

    video_server_base = ""
    if video_input_mode == "url":
        video_server_base = (
            video_base_url.rstrip("/") or start_local_video_server()
        )
        print(f"MMMU-Video: URL input mode, hosting base {video_server_base}")
    else:
        print("MMMU-Video: base64 input mode")

    cfg = EvalConfig(
        client=client,
        model=model,
        video_index=video_index,
        video_input_mode=video_input_mode,
        video_server_base=video_server_base,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        video_fps=video_fps,
        video_max_frames=video_max_frames,
    )

    items: list[tuple[dict[str, Any], str]] = []
    for config in ("Adaptation", "Comprehension", "Perception"):
        subset = load_subset(config, limit)
        items.extend((doc, config) for doc in subset)
        print(f"  {config}: {len(subset)} questions")
    print(f"MMMU-Video: {len(items)} questions total")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(infer, cfg, it): it for it in items}
        for fut in tqdm(
            as_completed(futures), total=len(items), desc="MMMU-Video"
        ):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"Error: {e}")
                results.append(
                    {"predicted": "", "correct": False, "error": str(e)}
                )

    (output_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results)
    )

    correct = sum(1 for r in results if r.get("correct"))
    errors = sum(1 for r in results if r.get("error"))
    total = len(results)
    accuracy = correct / total if total else 0.0
    # Errored samples count as incorrect in `accuracy` (never dropped); the
    # ex-errors view makes any structural ceiling (e.g. server media-cap
    # rejections) visible alongside it.
    answerable = total - errors
    accuracy_excluding_errors = correct / answerable if answerable else 0.0
    # Output (completion) token stats over SUCCESSFUL samples only.
    _otoks = [
        r["completion_tokens"]
        for r in results
        if "completion_tokens" in r and "error" not in r
    ]
    mean_output_tokens = round(statistics.mean(_otoks), 1) if _otoks else 0.0
    p50_output_tokens = round(statistics.median(_otoks), 1) if _otoks else 0.0
    print(
        f"MMMU-Video: {accuracy:.4f} ({correct}/{total}, {errors} errors; "
        f"excluding errors {accuracy_excluding_errors:.4f}) "
        f"mean_out_tok={mean_output_tokens:.1f} p50_out_tok={p50_output_tokens:.1f}"
    )

    summary = {
        "accuracy": accuracy,
        "accuracy_excluding_errors": accuracy_excluding_errors,
        "correct": correct,
        "total": total,
        "errors": errors,
        "mean_output_tokens": mean_output_tokens,
        "p50_output_tokens": p50_output_tokens,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # collect_scores.py (the suite-level collector) only globs for score.json;
    # writing it here — identical to summary.json — is what puts MMMU-Video in
    # the full-accuracy-eval suite's consolidated results.json and score table.
    (output_dir / "score.json").write_text(json.dumps(summary, indent=2))

    # Fail on a broken harness (no successful inferences / mostly errors) so
    # data or plumbing regressions surface loudly instead of silently scoring
    # 0.0.
    if total == 0 or errors == total:
        print(
            "::error::MMMU-Video produced no successful inferences — "
            "harness/data failure"
        )
        sys.exit(1)
    if errors > total * 0.5:
        print(
            f"::error::MMMU-Video errored on {errors}/{total} samples (>50%) "
            "— likely infra/data failure"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
