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
"""Consolidates per-benchmark ``score.json`` files into one suite result.

The full-accuracy-eval orchestrator downloads every artifact of the run into
one directory (one subdirectory per artifact) and runs this over it. Each eval
module writes ``score.json`` into its own results directory (see
``eval_common.write_outputs``), so a suite run yields a tree like::

    <artifacts-dir>/
      aime25-non-agentic-text-eval-results/score.json
      scicode-scicode-eval-results/score.json
      agent-class-eval-results/swe-results/score.json
      ...

Output is a single ``results.json`` plus a Markdown table (appended to
``$GITHUB_STEP_SUMMARY`` when set, stdout otherwise). Benchmarks listed via
``--expect`` but absent from the tree are reported as ``missing`` rather than
silently dropped, so a partially failed suite run stays visible.

Stdlib-only on purpose: the collector runs on a bare CPU runner without bazel.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def strip_suffix(name: str) -> str:
    for suffix in ("-eval-results", "-results"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def benchmark_name(
    score_path: Path, artifacts_dir: Path, prefixes: list[str]
) -> tuple[str, str]:
    """Returns (group, benchmark) for a score.json path.

    Group is the artifact directory (first path component). Benchmark is the
    score file's parent directory with the conventional ``-results`` suffix
    stripped (``aime25-results`` -> ``aime25``) — except when the parent IS
    the artifact directory: upload-artifact v4 puts a single uploaded
    directory's contents at the artifact root whenever exactly one directory
    matched. In that case the orchestrator's per-call artifact prefix is the
    only reliable name (each call runs one benchmark), so a matching prefix
    names the benchmark (``aime25-`` -> ``aime25``); without a match, fall
    back to the artifact name with the ``-results`` suffix stripped.
    """
    rel = score_path.relative_to(artifacts_dir)
    group = rel.parts[0] if len(rel.parts) > 1 else ""
    if len(rel.parts) == 2:
        for prefix in prefixes:
            if prefix and group.startswith(prefix):
                return group, prefix.rstrip("-")
        return group, strip_suffix(group)
    return group, strip_suffix(score_path.parent.name)


# Every headline-metric key any eval writes, ordered most- to least-specific.
# A benchmark whose headline metric is a composite of raw accuracy has to be
# named ahead of ``accuracy``, or the raw value wins and the suite table reports
# a number the benchmark is not scored on: AA-Omniscience writes both, and its
# real metric is ``50 + (accuracy - hallucination_rate) * 50``, which sits 10-16
# points above raw accuracy and turns parity into an apparent regression. A
# benchmark whose key is missing here is worse off still — it reports no score
# at all, which is what SciCode (``pass_at_1``) did until this list grew.
SCORE_KEYS = ("omniscience_score", "accuracy", "score", "pass_at_1")


def extract_score(summary: dict[str, Any]) -> float | None:
    for key in SCORE_KEYS:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def collect(
    artifacts_dir: Path, prefixes: list[str] | None = None
) -> list[dict[str, Any]]:
    rows = []
    for score_path in sorted(artifacts_dir.rglob("score.json")):
        try:
            summary = json.loads(score_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: unreadable {score_path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(summary, dict):
            print(f"WARNING: non-object {score_path}", file=sys.stderr)
            continue
        group, bench = benchmark_name(score_path, artifacts_dir, prefixes or [])
        rows.append(
            {
                "benchmark": bench,
                "group": group,
                "score": extract_score(summary),
                "errors": summary.get("errors"),
                "total": summary.get("total"),
                "mean_output_tokens": summary.get("mean_output_tokens"),
                "p50_output_tokens": summary.get("p50_output_tokens"),
                "stop_ratio": summary.get("stop_ratio"),
                "path": str(score_path.relative_to(artifacts_dir)),
                "summary": summary,
            }
        )
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def compare(
    rows: list[dict[str, Any]], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    """Joins this run's rows with a baseline results.json by benchmark name.

    Benchmarks present on only one side get a row with the other side's score
    as None, so an A/B with a failed leg cannot silently narrow the
    comparison.
    """
    base_rows = {r["benchmark"]: r for r in baseline.get("benchmarks", [])}
    names = [r["benchmark"] for r in rows]
    names += [n for n in base_rows if n not in names]
    joined = []
    ours = {r["benchmark"]: r for r in rows}
    for name in names:
        score = (ours.get(name) or {}).get("score")
        base_score = (base_rows.get(name) or {}).get("score")
        delta = (
            score - base_score
            if score is not None and base_score is not None
            else None
        )
        joined.append(
            {
                "benchmark": name,
                "score": score,
                "baseline_score": base_score,
                "delta": delta,
            }
        )
    return joined


def comparison_table(
    joined: list[dict[str, Any]], baseline_meta: dict[str, Any]
) -> str:
    label = baseline_meta.get("backend") or baseline_meta.get("run_id") or ""
    lines = [
        f"## Comparison vs baseline {label}".rstrip(),
        "",
        "| Benchmark | Score | Baseline | Delta |",
        "|---|---|---|---|",
    ]
    for r in joined:
        delta = r["delta"]
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        lines.append(
            f"| {r['benchmark']} | {fmt(r['score'])} "
            f"| {fmt(r['baseline_score'])} | {delta_str} |"
        )
    return "\n".join(lines) + "\n"


def markdown_table(rows: list[dict[str, Any]], missing: list[str]) -> str:
    lines = [
        (
            "| Benchmark | Group | Score | Errors | Total | Mean tokens | p50"
            " tokens | Stop ratio |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['benchmark']} | {r['group']} | {fmt(r['score'])} |"
            f" {fmt(r['errors'])} | {fmt(r['total'])} |"
            f" {fmt(r['mean_output_tokens'])} | {fmt(r['p50_output_tokens'])} |"
            f" {fmt(r.get('stop_ratio'))} |"
        )
    for name in missing:
        lines.append(f"| {name} | | MISSING | | | | | |")
    return "\n".join(lines) + "\n"


def extra_sections(rows: list[dict[str, Any]]) -> str:
    """Renders benchmark-specific detail sections beyond the fixed table.

    A row's ``summary`` can carry two optional keys that don't fit the
    table's fixed columns: ``unexpected_failures`` (a list of test IDs that
    failed and are not on a known-failures allowlist — the provider verifier
    is the first producer) and ``pass_at_10`` (a metric-name -> {score,
    reference, delta, ok} mapping — also the provider verifier's batch
    verification). Rendered as extra Markdown sections keyed by benchmark
    name so a run's consolidated summary surfaces them instead of leaving
    them buried in results.json.
    """
    lines = []
    for r in rows:
        summary = r.get("summary") or {}
        unexpected = summary.get("unexpected_failures")
        if unexpected:
            lines.append(f"\n### {r['benchmark']}: unexpected test failures\n")
            lines.append("| Test |\n|---|")
            lines.extend(f"| `{t}` |" for t in unexpected)
            lines.append("")

        pass_at_10 = summary.get("pass_at_10")
        if pass_at_10:
            lines.append(f"\n### {r['benchmark']}: pass@10 metrics\n")
            lines.append("| Metric | Score | Reference | Delta | Result |")
            lines.append("|---|---|---|---|---|")
            for name, m in pass_at_10.items():
                icon = "✅" if m.get("ok") else "❌"
                lines.append(
                    f"| {name} | {fmt(m.get('score'))} |"
                    f" {fmt(m.get('reference'))} | {fmt(m.get('delta'))} |"
                    f" {icon} |"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--expect",
        default="",
        help=(
            "Comma-separated benchmark names that must be present; absent "
            "ones are reported as missing instead of dropped."
        ),
    )
    parser.add_argument(
        "--meta",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Run metadata embedded verbatim in results.json (repeatable).",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help=(
            "Known per-call artifact-name prefix (repeatable; pass the "
            "orchestrator's artifact_prefix values). A root-level score.json "
            "in an artifact matching a prefix is named by that prefix."
        ),
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help=(
            "results.json of a baseline suite run (e.g. the other leg of a "
            "MAX-vs-vLLM A/B); adds a per-benchmark delta section."
        ),
    )
    args = parser.parse_args()

    rows = collect(args.artifacts_dir, args.prefix)
    found = {r["benchmark"] for r in rows}
    expected = [n.strip() for n in args.expect.split(",") if n.strip()]
    missing = [n for n in expected if n not in found]

    meta = dict(kv.split("=", 1) for kv in args.meta if "=" in kv)
    result = {"run": meta, "benchmarks": rows, "missing": missing}

    comparison_md = ""
    if args.compare_to is not None:
        baseline = json.loads(args.compare_to.read_text())
        joined = compare(rows, baseline)
        result["comparison"] = {
            "baseline_run": baseline.get("run", {}),
            "benchmarks": joined,
        }
        comparison_md = "\n" + comparison_table(joined, baseline.get("run", {}))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    table = markdown_table(rows, missing) + comparison_md + extra_sections(rows)
    header = "# Full accuracy eval: consolidated scores\n\n"
    if meta:
        header += (
            "".join(f"**{k}:** `{v}`  \n" for k, v in sorted(meta.items()))
            + "\n"
        )
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(header + table)
    print(header + table)
    print(f"Wrote {args.out} ({len(rows)} benchmarks, {len(missing)} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
