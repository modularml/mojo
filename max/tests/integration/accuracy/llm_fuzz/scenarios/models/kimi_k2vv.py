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
"""Kimi K2 Vendor Verifier (K2VV) — tool call trigger and schema accuracy benchmark.

Runs MoonshotAI's official K2VV dataset against the endpoint and measures:
  1. ToolCall-Trigger F1: does the deployment trigger tool calls in the same
     situations as the official Moonshot API?
  2. Schema Accuracy: when tool calls are triggered, do the JSON arguments
     validate against the tool's schema?

Dataset is downloaded on each run from MoonshotAI's CDN (~13MB compressed).

Modes (controlled via --k2vv-mode):
  quick  — 500 randomly sampled requests (default)
  full   — all 2,000 requests

Reference: https://github.com/MoonshotAI/K2-Vendor-Verifier
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import random
import shutil
import tarfile
import tempfile
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import jsonschema

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig
    from validator_client import ValidatorClient

_DATASET_URL = "https://statics.moonshot.cn/k2vv/tool-calls.tar.gz"
_QUICK_SAMPLE_COUNT = 500
_RANDOM_SEED = 42

# Thresholds from K2VV README
_F1_THRESHOLD = 0.73
_SCHEMA_ACCURACY_THRESHOLD = 0.95


def _download_and_extract() -> tuple[list[str], dict[str, list[str]]]:
    """Download K2VV dataset to a temp dir, return (sample_lines, {model: result_lines})."""
    tmp = tempfile.mkdtemp(prefix="k2vv-")
    tar_path = pathlib.Path(tmp) / "tool-calls.tar.gz"

    print(f"  Downloading K2VV dataset ({_DATASET_URL})...")
    urllib.request.urlretrieve(_DATASET_URL, str(tar_path))

    print("  Extracting...")
    with tarfile.open(str(tar_path), "r:gz") as tf:
        for member in tf.getmembers():
            basename = pathlib.Path(member.name).name
            if not basename or basename.startswith("._"):
                continue
            member.name = basename
            tf.extract(member, path=tmp, filter="data")

    tar_path.unlink()

    samples_path = pathlib.Path(tmp) / "samples.jsonl"
    if not samples_path.exists():
        shutil.rmtree(tmp, ignore_errors=True)
        raise FileNotFoundError("samples.jsonl not found in downloaded archive")

    sample_lines = samples_path.read_text().splitlines()

    ref_lines: dict[str, list[str]] = {}
    for name in (
        "kimi-k2-thinking_results.jsonl",
        "kimi-k2-0905-preview_results.jsonl",
    ):
        p = pathlib.Path(tmp) / name
        if p.exists():
            ref_lines[name] = p.read_text().splitlines()

    shutil.rmtree(tmp, ignore_errors=True)
    print(
        f"  Downloaded {len(sample_lines)} samples, {sum(len(v) for v in ref_lines.values())} reference results"
    )
    return sample_lines, ref_lines


def _parse_samples(lines: list[str]) -> list[dict[str, Any]]:
    """Parse JSONL lines into sample dicts with data_index."""
    samples = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append({"data_index": i, "request": json.loads(line)})
        except json.JSONDecodeError:
            continue
    return samples


def _parse_reference(ref_lines: dict[str, list[str]]) -> dict[int, str]:
    """Parse reference results into {data_index: finish_reason}. Prefers thinking variant."""
    refs: dict[int, str] = {}
    for name in (
        "kimi-k2-thinking_results.jsonl",
        "kimi-k2-0905-preview_results.jsonl",
    ):
        lines = ref_lines.get(name)
        if not lines:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                idx = d.get("data_index")
                fr = d.get("finish_reason")
                if idx is not None and fr is not None:
                    refs[idx] = fr
            except json.JSONDecodeError:
                continue
        if refs:
            break
    return refs


def _schema_map(
    tools: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index a tool list by function name → parameters schema."""
    return {
        defn["function"]["name"]: defn["function"]["parameters"]
        for defn in tools
        if "function" in defn and "name" in defn["function"]
    }


def _validate_tool_call(
    tool_call: dict[str, Any], tools: list[dict[str, Any]]
) -> tuple[bool, str]:
    """Check whether a tool call's arguments match its declared schema.

    The K2VV benchmark scores schema-conformance: the model named a tool
    and emitted arguments; the arguments must be parseable JSON and must
    satisfy the schema that was given to the model. The metric is
    described in MoonshotAI's public K2VV README.

    Returns ``(valid, detail)``. ``detail`` is empty on success and a
    one-line description of the failure otherwise.
    """
    fn_name = tool_call.get("function", {}).get("name", "")
    if not fn_name:
        return False, "tool call has no function name"

    schemas_by_name = _schema_map(tools)
    arg_schema = schemas_by_name.get(fn_name)
    if arg_schema is None:
        return False, f"no schema found for function '{fn_name}'"

    raw_args = tool_call["function"].get("arguments", "")
    parsed_args: object
    if isinstance(raw_args, str):
        try:
            parsed_args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            return False, f"'{fn_name}': arguments are not JSON: {exc}"
    else:
        parsed_args = raw_args

    try:
        jsonschema.validate(instance=parsed_args, schema=arg_schema)
    except jsonschema.ValidationError as exc:
        return False, f"'{fn_name}': {exc.message}"
    return True, ""


def _compute_f1(
    predictions: list[str], references: list[str]
) -> dict[str, Any]:
    """Compute tool call trigger precision, recall, and F1."""
    tp = fp = fn = tn = 0
    for pred, ref in zip(predictions, references, strict=False):
        pred_tc = pred == "tool_calls"
        ref_tc = ref == "tool_calls"
        if pred_tc and ref_tc:
            tp += 1
        elif pred_tc and not ref_tc:
            fp += 1
        elif not pred_tc and ref_tc:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _send_one_sync(
    validator: ValidatorClient, request: dict[str, Any]
) -> tuple[str, list[dict[str, Any]] | None, str]:
    """Send a single K2VV request synchronously.

    Returns ``(finish_reason, tool_calls, content)``.
    """
    messages = request.get("messages", [])
    tools = request.get("tools")
    max_tokens = request.get("max_tokens", 16000)
    temperature = request.get("temperature", 0.6)
    kwargs = {"temperature": temperature, "max_tokens": max_tokens}

    if tools:
        resp = validator.tc_chat(
            messages, tools, model=None, tool_choice="auto", **kwargs
        )
        choice = resp.choices[0]
        finish_reason = choice.finish_reason or "stop"
        content = choice.message.content or ""
        raw_tcs = choice.message.tool_calls
        tool_calls = None
        if raw_tcs:
            tool_calls = [
                {
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in raw_tcs
            ]
    else:
        resp = validator.chat(messages, **kwargs)
        finish_reason = resp.choices[0].finish_reason or "stop"
        content = resp.choices[0].message.content or ""
        tool_calls = None

    return finish_reason, tool_calls, content


@register_scenario
class KimiK2VV(BaseScenario):
    name = "kimi_k2vv"
    description = (
        "K2 Vendor Verifier -- tool call trigger F1 and schema accuracy "
        "against MoonshotAI reference (quick: 500, full: 2K requests)"
    )
    tags = ["model:kimi-k2.5", "k2vv", "benchmark"]
    requires_validator = True
    model_filter = "kimi-k2.5"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        v = config.validator
        if not v:
            results.append(
                self.make_result(
                    self.name,
                    "setup",
                    Verdict.ERROR,
                    detail="No validator client available",
                )
            )
            return results

        loop = asyncio.get_running_loop()

        # Determine mode from config (set by --k2vv-mode flag in fuzz.py)
        k2vv_mode = getattr(config, "k2vv_mode", "quick")

        # Download dataset
        try:
            sample_lines, ref_lines = await loop.run_in_executor(
                None, _download_and_extract
            )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "download",
                    Verdict.ERROR,
                    detail=f"Failed to download dataset: {e}",
                    error=str(e),
                )
            )
            return results

        # Parse all samples and references
        all_samples = _parse_samples(sample_lines)
        reference = _parse_reference(ref_lines)

        if not all_samples:
            results.append(
                self.make_result(
                    self.name,
                    "setup",
                    Verdict.ERROR,
                    detail="No valid samples in dataset",
                )
            )
            return results

        # Select subset based on mode
        if k2vv_mode == "full":
            samples = all_samples
        else:
            rng = random.Random(_RANDOM_SEED)
            count = min(_QUICK_SAMPLE_COUNT, len(all_samples))
            samples = rng.sample(all_samples, count)

        results.append(
            self.make_result(
                self.name,
                "dataset_loaded",
                Verdict.PASS,
                detail=f"mode={k2vv_mode}, {len(samples)}/{len(all_samples)} samples, {len(reference)} reference results",
            )
        )

        # Run requests concurrently
        concurrency = min(getattr(config, "max_concurrency", 10), 10)

        predictions: list[str] = []
        ref_finish_reasons: list[str] = []
        schema_total = 0
        schema_valid = 0
        schema_failures: list[str] = []
        errors = 0
        diagnostics: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        def _run_sample(sample: dict[str, Any]) -> dict[str, Any]:
            """Run one K2VV sample and return a structured result dict."""
            idx = sample["data_index"]
            req = sample["request"]
            tools = req.get("tools", [])
            tool_names = [t["function"]["name"] for t in tools] if tools else []

            try:
                finish_reason, tool_calls, content = _send_one_sync(v, req)
            except Exception as e:
                return {
                    "data_index": idx,
                    "finish_reason": "error",
                    "tool_calls": None,
                    "tool_names": tool_names,
                    "content_preview": "",
                    "schema_ok": True,
                    "schema_errors": [],
                    "error": str(e),
                }

            schema_ok = True
            schema_errors: list[str] = []
            if finish_reason == "tool_calls" and tool_calls:
                for tc in tool_calls:
                    valid, detail = _validate_tool_call(tc, tools)
                    if not valid:
                        schema_ok = False
                        schema_errors.append(detail)

            # Extract last user message for diagnostics (FP/FN analysis)
            msgs = req.get("messages", [])
            user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
            user_preview = (user_msgs[-1][:300]) if user_msgs else ""

            return {
                "data_index": idx,
                "finish_reason": finish_reason,
                "tool_calls": tool_calls,
                "tool_names": tool_names,
                "content_preview": (content or "")[:300],
                "user_message_preview": user_preview,
                "schema_ok": schema_ok,
                "schema_errors": schema_errors,
                "error": "",
            }

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_run_sample, s): s["data_index"] for s in samples
            }
            done = 0
            total = len(futures)
            for future in as_completed(futures):
                done += 1
                if done % 50 == 0 or done == total:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"\r  K2VV progress: {done}/{total} ({elapsed:.0f}s)",
                        end="",
                        flush=True,
                    )

                r = future.result()
                idx = r["data_index"]

                if r["error"]:
                    errors += 1
                    diagnostics.append({**r, "mismatch_type": "error"})
                    continue

                finish_reason = r["finish_reason"]
                predictions.append(finish_reason)
                ref_fr = reference.get(idx)
                ref_finish_reasons.append(ref_fr or finish_reason)

                # Track schema accuracy
                if finish_reason == "tool_calls":
                    schema_total += 1
                    if r["schema_ok"]:
                        schema_valid += 1
                    else:
                        schema_failures.extend(
                            f"sample {idx}: {d}" for d in r["schema_errors"]
                        )

                # Classify mismatches for diagnostics
                pred_tc = finish_reason == "tool_calls"
                ref_tc = (ref_fr == "tool_calls") if ref_fr else pred_tc

                if pred_tc != ref_tc or not r["schema_ok"]:
                    mismatch_type = (
                        "false_positive"
                        if pred_tc and not ref_tc
                        else "false_negative"
                        if not pred_tc and ref_tc
                        else "schema_failure"
                    )
                    diagnostics.append(
                        {
                            **r,
                            "mismatch_type": mismatch_type,
                            "ref_finish_reason": ref_fr,
                            "called_functions": [
                                tc["function"]["name"]
                                for tc in (r["tool_calls"] or [])
                            ],
                        }
                    )

        print()  # newline after progress
        elapsed_sec = time.perf_counter() - t0

        total_sent = len(predictions)
        if total_sent == 0:
            results.append(
                self.make_result(
                    self.name,
                    "no_responses",
                    Verdict.FAIL,
                    detail=f"All {errors} requests failed",
                )
            )
            return results

        # ToolCall-Trigger F1
        f1_stats = _compute_f1(predictions, ref_finish_reasons)
        f1 = f1_stats["f1"]

        f1_verdict = Verdict.PASS if f1 >= _F1_THRESHOLD else Verdict.FAIL
        results.append(
            self.make_result(
                self.name,
                "toolcall_trigger_f1",
                f1_verdict,
                elapsed_ms=elapsed_sec * 1000,
                detail=(
                    f"F1={f1:.3f} (threshold={_F1_THRESHOLD:.2f}) | "
                    f"precision={f1_stats['precision']:.3f} recall={f1_stats['recall']:.3f} | "
                    f"TP={f1_stats['tp']} FP={f1_stats['fp']} FN={f1_stats['fn']} TN={f1_stats['tn']}"
                ),
            )
        )

        # Schema Accuracy
        schema_acc = 0.0
        if schema_total > 0:
            schema_acc = schema_valid / schema_total
            schema_verdict = (
                Verdict.PASS
                if schema_acc >= _SCHEMA_ACCURACY_THRESHOLD
                else Verdict.FAIL
            )
            detail = f"accuracy={schema_acc:.3f} ({schema_valid}/{schema_total}) (threshold={_SCHEMA_ACCURACY_THRESHOLD:.2f})"
            if schema_failures:
                detail += " | failures: " + "; ".join(schema_failures[:10])
            results.append(
                self.make_result(
                    self.name,
                    "schema_accuracy",
                    schema_verdict,
                    detail=detail,
                )
            )
        else:
            results.append(
                self.make_result(
                    self.name,
                    "schema_accuracy",
                    Verdict.INTERESTING,
                    detail="No tool calls triggered -- cannot measure schema accuracy",
                )
            )

        # Summary
        overall = (
            Verdict.PASS
            if f1_verdict == Verdict.PASS
            and (schema_total == 0 or schema_acc >= _SCHEMA_ACCURACY_THRESHOLD)
            else Verdict.FAIL
        )
        results.append(
            self.make_result(
                self.name,
                "k2vv_summary",
                overall,
                elapsed_ms=elapsed_sec * 1000,
                detail=f"{total_sent} requests in {elapsed_sec:.1f}s | F1={f1:.3f} | schema={schema_valid}/{schema_total} | errors={errors}",
            )
        )

        # Write diagnostics for mismatch analysis
        self._write_diagnostics(diagnostics, f1, schema_acc)

        return results

    @staticmethod
    def _write_diagnostics(
        diagnostics: list[dict[str, Any]],
        f1: float,
        schema_acc: float,
    ) -> None:
        """Write mismatch diagnostics to a JSONL file for F1 analysis."""
        if not diagnostics:
            return

        counts = Counter(d["mismatch_type"] for d in diagnostics)

        log_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = log_dir / f"k2vv-diagnostics-{ts}.jsonl"

        diagnostics.sort(
            key=lambda d: (d.get("mismatch_type", ""), d["data_index"])
        )

        with open(path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "event": "diagnostics_summary",
                        "total_mismatches": len(diagnostics),
                        "false_positives": counts.get("false_positive", 0),
                        "false_negatives": counts.get("false_negative", 0),
                        "schema_failures": counts.get("schema_failure", 0),
                        "errors": counts.get("error", 0),
                        "f1": f1,
                        "schema_accuracy": schema_acc,
                    }
                )
                + "\n"
            )
            for d in diagnostics:
                f.write(
                    json.dumps(
                        {
                            "data_index": d["data_index"],
                            "mismatch_type": d.get("mismatch_type", "unknown"),
                            "our_finish_reason": d.get("finish_reason"),
                            "ref_finish_reason": d.get("ref_finish_reason"),
                            "tool_names_available": d.get("tool_names", []),
                            "called_functions": d.get("called_functions", []),
                            "tool_calls": d.get("tool_calls"),
                            "schema_errors": d.get("schema_errors", []),
                            "content_preview": d.get("content_preview", ""),
                            "user_message_preview": d.get(
                                "user_message_preview", ""
                            ),
                            "error": d.get("error", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(
            f"  Diagnostics: {path.name} "
            f"({len(diagnostics)} mismatches: "
            f"{counts.get('false_positive', 0)} FP, "
            f"{counts.get('false_negative', 0)} FN, "
            f"{counts.get('schema_failure', 0)} schema, "
            f"{counts.get('error', 0)} errors)"
        )
