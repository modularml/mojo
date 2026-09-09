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
from __future__ import annotations

"""
Shared eval-runner library for smoke tests.

Drives an lm_eval task against an OpenAI-compatible chat/completions endpoint
and post-processes the results. The endpoint URL is supplied by the caller, so
this module is agnostic to whether it's backed by a single max serve, a DI
proxy, or anything else.

Also exposes a small set of utilities (HF token check, GPU detection,
GitHub-output writer) that are shared between the smoke-test entrypoints.
"""

import json
import logging
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from subprocess import DEVNULL, TimeoutExpired, check_call, check_output
from tempfile import TemporaryDirectory
from typing import Any

import requests

logger = logging.getLogger(__name__)

DUMMY_2X2_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP8zwACTGCSAQANHQEDgslx/wAAAABJRU5ErkJggg=="
)
IMAGE_PROMPT = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Say 'hello image'"},
        {"type": "image_url", "image_url": {"url": DUMMY_2X2_IMAGE}},
    ],
}
TEXT_PROMPT = {"role": "user", "content": "Say: 'hello world'"}

TEXT_TASK = "gsm8k_cot_llama"
VISION_TASK = "chartqa"

EvalResults = dict[str, Any]
EvalSamples = list[dict[str, Any]]

# Prometheus field where each framework exposes its generated-token count.
_COMPLETION_TOKEN_METRICS = (
    "maxserve_num_output_tokens_total",  # max / max-ci
    "vllm:generation_tokens_total",  # vllm
    "sglang:generation_tokens_total",  # sglang
)

_SPEC_DECODE_ACCEPTANCE_LENGTH_METRIC = (
    "maxserve_spec_decode_avg_acceptance_length"
)


def _inside_bazel() -> bool:
    return os.getenv("BUILD_WORKSPACE_DIRECTORY") is not None


def validate_hf_token() -> None:
    if os.getenv("HF_TOKEN") is None:
        raise ValueError(
            "Environment variable `HF_TOKEN` must be set. "
            "See https://www.notion.so/modularai/HuggingFace-Access-Token-29d1044d37bb809fbe70e37428faf9da"
        )


def resolve_canonical_repo_id(repo_id: str) -> str:
    """HF disk cache is case-sensitive, so do what we can to avoid issues"""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return repo_id
    try:
        r = requests.get(
            f"https://huggingface.co/api/models/{repo_id}",
            headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
            timeout=(5, 10),
        )
        r.raise_for_status()
        return r.json()["id"]
    except Exception as e:
        logger.warning("Failed repo id lookup for %s: %s", repo_id, e)
        return repo_id


@cache
def get_gpu_name_and_count() -> tuple[str, int]:
    """Returns the name and number of the available GPUs, e.g. ('MI300', 2)"""
    amd = ["amd-smi", "static", "--json"]
    nv = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    try:  # AMD path
        env = os.environ.copy()
        if _inside_bazel():
            # Workaround to make amd-smi work inside bazel.
            for k in list(env):
                if k.startswith("PYTHON") or "RUNFILES" in k:
                    del env[k]
        result = check_output(amd, text=True, stderr=DEVNULL, env=env)
        data = json.loads(result.strip())["gpu_data"]
        name, count = data[0]["asic"]["market_name"], len(data)
    except Exception:
        try:  # Nvidia path
            lines = (
                check_output(nv, text=True, stderr=DEVNULL).strip().split("\n")
            )
            name, count = lines[0].strip(), len(lines)
        except Exception:
            logger.warning("nvidia-smi and amd-smi both failed")
            return "N/A", 0
    return name, count


def safe_model_name(model: str) -> str:
    return model.replace("/", "__")


def write_github_output(key: str, value: str) -> None:
    path = os.getenv("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{key}={value}\n")


def test_single_request(
    url: str, model: str, task: str, disable_timeouts: bool
) -> None:
    is_vision = task == VISION_TASK
    m = [IMAGE_PROMPT if is_vision else TEXT_PROMPT]

    # Initial req can be slow for huge models
    connect_timeout, read_timeout = (
        (None, None) if disable_timeouts else (30, 180)
    )
    r = requests.post(
        url,
        json={"model": model, "messages": m, "max_tokens": 8},
        timeout=(connect_timeout, read_timeout),
    )
    r.raise_for_status()
    resp = r.json()["choices"][0]["message"]["content"]
    logger.info(f"Test single request OK. Response: {resp}")


def call_eval(
    url: str,
    model: str,
    task: str,
    *,
    max_concurrent: int,
    num_questions: int,
    disable_timeouts: bool,
    metrics_url: str | None = None,
    model_alias: str | None = None,
    lm_eval_metadata: str | None = None,
) -> tuple[EvalResults, EvalSamples]:
    extra_gen_kwargs = ""
    # model_alias carries the original recipe key (e.g. "nvidia/Kimi-K2.5-NVFP4__internal")
    # when model is a local path that doesn't contain the model name keywords.
    check_name = (model_alias or model).casefold()
    is_reasoning_model = any(
        kw in check_name
        for kw in (
            "academic-ds",
            "deepseek-r1",
            "deepseek-v3",
            "gemma-4",
            "gpt-oss",
            "inkling",
            "internvl3_5",
            "qwen3",
            "kimi-k2",
            "minimax-m2",
            "minimax-m3",
            "nemotron",
            "step-3.5",
            "glm-5",
        )
    )
    # Reasoning models needs extra tokens for .. reasoning
    if is_reasoning_model:
        extra_gen_kwargs = ",max_gen_toks=4096"

    # GPT-OSS sometimes gets stuck in a reasoning loop. To ensure consistency
    # in CI, we add a repetition penalty which helps prevent the loop
    if "gpt-oss" in model.casefold():
        extra_gen_kwargs = extra_gen_kwargs + ",repetition_penalty=1.1"

    interpreter = sys.executable if _inside_bazel() else ".venv-eval/bin/python"

    model_args: dict[str, str] = {
        "model": model,
        "base_url": url,
        "num_concurrent": str(max_concurrent),
        "max_retries": "1",
        "timeout": "86400" if disable_timeouts else "600",
    }

    include_path = str(Path(__file__).parent.resolve() / "tasks")
    with TemporaryDirectory() as tempdir:
        eval_cmd = [
            "lm_eval",
            f"--tasks={task}",
            "--model=local-chat-completions",
            "--log_samples",
            f"--model_args={','.join(f'{k}={v}' for k, v in model_args.items())}",
            "--apply_chat_template",
            f"--output_path={tempdir}",
            f"--limit={num_questions}",
            "--seed=42",
            f"--gen_kwargs=seed=42,temperature=0{extra_gen_kwargs}",
            f"--include_path={include_path}",
            "--fewshot_as_multiturn",
        ]

        # Passed verbatim to lm-eval; merged into each task's config metadata to
        # parameterize it at runtime (e.g. babilong context length).
        if lm_eval_metadata is not None:
            eval_cmd.append(f"--metadata={lm_eval_metadata}")

        args = [interpreter, "-m", *eval_cmd]
        logger.info(f"Running eval with:\n {' '.join(args)}")
        eval_timeout = None if disable_timeouts else 600
        tokens_before = (
            get_num_tokens_generated(metrics_url) if metrics_url else None
        )
        spec_decode_totals_before = (
            get_spec_decode_acceptance_totals(metrics_url)
            if metrics_url
            else None
        )
        try:
            check_call(args, timeout=eval_timeout)
        except TimeoutExpired:
            raise RuntimeError(
                f"Evals did not finish within the expected timeout={eval_timeout}s. "
                "You can pass --disable-timeouts to opt-out of this."
            ) from None
        tokens_after = (
            get_num_tokens_generated(metrics_url) if metrics_url else None
        )
        spec_decode_totals_after = (
            get_spec_decode_acceptance_totals(metrics_url)
            if metrics_url
            else None
        )

        results, samples = parse_eval_results(Path(tempdir))
        results["num_tokens_generated"] = (
            tokens_after - tokens_before
            if tokens_before is not None and tokens_after is not None
            else None
        )
        results["spec_decode_acceptance_length"] = (
            _spec_decode_acceptance_length_delta(
                spec_decode_totals_before, spec_decode_totals_after
            )
        )
        return results, samples


def get_num_tokens_generated(metrics_url: str) -> int | None:
    """Reads the generated-token count from Prometheus, or None if missing."""
    try:
        r = requests.get(metrics_url, timeout=(5, 30))
        r.raise_for_status()
    except requests.RequestException:
        return None

    for metric in _COMPLETION_TOKEN_METRICS:
        total = 0.0
        found = False
        for line in r.text.splitlines():
            if not line or line.startswith("#"):
                continue
            # A sample line is ``name{labels} value`` or ``name value``.
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name != metric:
                continue
            total += float(line.rsplit(" ", 1)[1])
            found = True
        if found:
            return int(total)

    return None


def get_spec_decode_acceptance_totals(
    metrics_url: str,
) -> tuple[float, float] | None:
    """Reads the cumulative (sum, count) of the spec-decode acceptance-length
    histogram from Prometheus."""
    try:
        r = requests.get(metrics_url, timeout=(5, 30))
        r.raise_for_status()
    except requests.RequestException:
        return None

    total_sum = 0.0
    total_count = 0.0
    for line in r.text.splitlines():
        if not line or line.startswith("#"):
            continue
        # A sample line is ``name{labels} value`` or ``name value``.
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if not name.startswith(_SPEC_DECODE_ACCEPTANCE_LENGTH_METRIC):
            continue
        # Skips the ``_bucket`` rows of the same family.
        if name.endswith("_sum"):
            total_sum += float(line.rsplit(" ", 1)[1])
        elif name.endswith("_count"):
            total_count += float(line.rsplit(" ", 1)[1])

    return total_sum, total_count


def _spec_decode_acceptance_length_delta(
    before: tuple[float, float] | None,
    after: tuple[float, float] | None,
) -> float | None:
    """Computes the windowed acceptance length from two cumulative
    (sum, count) snapshots, or None if either snapshot is missing or the
    count didn't advance (guards the division)."""
    if before is None or after is None:
        return None
    sum_before, count_before = before
    sum_after, count_after = after
    count_delta = count_after - count_before
    if count_delta <= 0:
        return None
    return (sum_after - sum_before) / count_delta


def parse_eval_results(loc: Path) -> tuple[EvalResults, EvalSamples]:
    samples = []
    for line in open(next(loc.glob("**/samples*.jsonl")), encoding="utf-8"):
        samples.append(json.loads(line))

    results = json.loads(next(loc.glob("**/results*.json")).read_text("utf-8"))

    return results, samples


@dataclass
class EvalSummary:
    gpu_name: str
    gpu_count: int
    startup_time_seconds: float
    eval_task: str
    task_type: str
    accuracy: float
    accuracy_stderr: float
    total_evaluation_time_seconds: float
    task_hash: str
    num_tokens_generated: int | None = None
    spec_decode_acceptance_length: float | None = None


def build_eval_summary(
    results: Sequence[Mapping[str, Any]],
    startup_time_seconds: float,
) -> list[EvalSummary]:
    """
    Extract the metrics from the eval results and build a summary for each task.
    """
    summaries = []

    for result in results:
        task = next(iter(result["results"].keys()))
        metrics = result["results"][task]
        total_secs = float(result["total_evaluation_time_seconds"])

        if VISION_TASK in task:
            accuracy = metrics["relaxed_accuracy,none"]
            accuracy_stderr = metrics["relaxed_accuracy_stderr,none"]
            task_type = "vision"
        elif task == TEXT_TASK:
            accuracy = metrics["exact_match,flexible-extract"]
            accuracy_stderr = metrics["exact_match_stderr,flexible-extract"]
            task_type = "text"
        elif "babilong" in task:
            accuracy = metrics["acc,none"]
            accuracy_stderr = metrics["acc_stderr,none"]
            task_type = "long-context"
        else:
            raise ValueError(f"Unknown task: {task}")

        gpu_name, gpu_count = get_gpu_name_and_count()
        summaries.append(
            EvalSummary(
                gpu_name=gpu_name,
                gpu_count=gpu_count,
                startup_time_seconds=round(startup_time_seconds, 2),
                eval_task=task,
                task_type=task_type,
                accuracy=accuracy,
                accuracy_stderr=accuracy_stderr,
                total_evaluation_time_seconds=total_secs,
                task_hash=result["task_hashes"][task],
                num_tokens_generated=result.get("num_tokens_generated"),
                spec_decode_acceptance_length=result.get(
                    "spec_decode_acceptance_length"
                ),
            )
        )

    return summaries


def print_samples(samples: EvalSamples, print_cot: bool) -> None:
    """
    Print question and the model's responses to each sample
    Assumes 'resps' is [[str]] (one decode) and GSM8K uses 'question',
    ChartQA uses 'query'.
    """
    for item in samples:
        doc = item.get("doc", {})
        question = doc.get("question") or doc.get("query")

        filt = item["filtered_resps"]
        extracted = filt[0] if isinstance(filt, list) and filt else str(filt)

        status = "✅" if item["exact_match"] == 1.0 else "❌"
        prefix_q = "🧮" if "question" in doc else "📊"

        logger.info(f"{prefix_q} {question}")
        if print_cot:
            logger.info(f"🤖💭 {item['resps'][0][0]}")
        logger.info(f"{status} {extracted}")


def write_results(
    path: Path,
    summary: list[EvalSummary],
    results: list[EvalResults],
    all_samples: list[EvalSamples],
    tasks: list[str],
) -> None:
    summary_file = path / "eval_metrics.json"
    summary_json = json.dumps([asdict(s) for s in summary], indent=2)
    summary_file.write_text(summary_json, encoding="utf-8")
    for result, samples, task in zip(results, all_samples, tasks, strict=True):
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        result_fp = path / f"results_{task}_{timestamp}.json"
        result_fp.write_text(json.dumps(result, indent=2), encoding="utf-8")

        samples_fp = path / f"samples_{task}_{timestamp}.jsonl"
        with open(samples_fp, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
