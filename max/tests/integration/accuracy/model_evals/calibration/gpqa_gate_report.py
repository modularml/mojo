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
# ===---------------------------------------------------------------------=== #
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
# ===---------------------------------------------------------------------=== #
"""Markdown report for the GPQA logit-shift gate (GitHub job summary)."""

from __future__ import annotations

from pathlib import Path

from calibration.gpqa_gate import (
    LiveVerdict,
    ScoredGate,
    load_gate_hist,
    rate_from_k,
    score_catalog,
    subset_park_rates,
)

# v1 plus_bucket 60+90 pool (n=4050 subset / 29700 full).
V1_PLUS_ACC = 0.6630
V1_PLUS_STOP = 0.8923


def pct(rate: float | None) -> str:
    return "-" if rate is None else f"{100.0 * rate:.2f}%"


def _mark(status: str | None) -> str:
    if status == "pass":
        return "✅ pass"
    if status == "fail":
        return "❌ fail"
    if status == "error":
        return "⚠️ error"
    return "skipped"


def _k(value: int | None) -> str:
    return "-" if value is None else str(value)


def _catalog_rows(
    rows: list[ScoredGate], selected: ScoredGate | None
) -> list[ScoredGate]:
    if rows:
        return rows
    if selected is None:
        return score_catalog()
    spec = selected.spec
    return score_catalog(
        alpha=spec.alpha,
        beta=spec.beta,
        delta_stop=spec.delta_stop,
        delta_acc=spec.delta_acc,
        want_stop=spec.want_stop,
        want_acc=spec.want_acc,
    )


def _md_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    right: set[int] | None = None,
) -> list[str]:
    aligns = [
        "---:" if right and i in right else "---" for i in range(len(headers))
    ]
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
        *[("| " + " | ".join(row) + " |") for row in rows],
    ]


def write_report(
    work_dir: Path,
    rows: list[ScoredGate],
    selected: ScoredGate | None,
    live: LiveVerdict | None,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> Path:
    """Write the job-summary markdown to work_dir/REPORT.md."""
    acc_h = acc_r = stop_h = stop_r = acc_pass = stop_pass = None
    if selected is not None:
        acc_h, acc_r, stop_h, stop_r = subset_park_rates(selected.spec)
        acc_pass = rate_from_k(selected.acc_cutoff, selected.cost)
        stop_pass = rate_from_k(selected.stop_cutoff, selected.cost)
    if live is not None and live.status != "error":
        if live.acc_rate_cutoff is not None:
            acc_pass = live.acc_rate_cutoff
        if live.stop_rate_cutoff is not None:
            stop_pass = live.stop_rate_cutoff
    if live is None:
        headline = "compare-only"
        rationale = None
        acc_status = stop_status = None
    else:
        headline = _mark(live.status)
        rationale = live.rationale
        acc_status, stop_status = live.acc_status, live.stop_status
    endpoint = base_url or (live.base_url if live is not None else None)
    model_id = model or (live.model if live is not None else None)
    acc_threshold = pct(live.acc_rate_cutoff if live else acc_pass)
    stop_threshold = pct(live.stop_rate_cutoff if live else stop_pass)
    designed_acc_k = selected.acc_cutoff if selected is not None else None
    designed_stop_k = selected.stop_cutoff if selected is not None else None
    n_ids = len(selected.spec.prompt_ids) if selected is not None else 0
    v1_acc = v1_stop = None
    if selected is not None and n_ids:
        if selected.spec.subset == "plus_bucket":
            v1_acc, v1_stop = V1_PLUS_ACC, V1_PLUS_STOP
        else:
            hist = load_gate_hist()
            v1_acc = (
                sum(hist.q_acc[i] for i in selected.spec.prompt_ids) / n_ids
            )
            v1_stop = (
                1.0
                - sum(hist.q_trunc[i] for i in selected.spec.prompt_ids) / n_ids
            )
    v1_label = (
        "v1 plus_bucket"
        if selected is None or selected.spec.subset == "plus_bucket"
        else "v1"
    )
    meta = [
        f"- endpoint: `{endpoint or '(not set)'}`",
        f"- model: `{model_id or '(not set)'}`",
    ]
    lines = [
        "# GPQA logit-gate report",
        "",
        *meta,
        "",
    ]
    if selected is not None:
        lines.extend(
            ["## Selected", "", *_selected_bullets(selected, rows), ""]
        )
    lines.extend(
        [
            "## Verdict",
            "",
            f"**{headline}**  (job fails if either enabled metric fails)",
            "",
            *([rationale, ""] if rationale else []),
            *_md_table(
                ["Metric", "Observed", "Pass threshold", "k", "S", "Verdict"],
                [
                    [
                        "Accuracy",
                        pct(live.acc_rate if live else None),
                        f"≥ {acc_threshold}",
                        _k(live.acc_cutoff if live else designed_acc_k),
                        "-" if live is None else str(live.n_wrong),
                        _mark(acc_status),
                    ],
                    [
                        "Stop ratio",
                        pct(live.stop_rate if live else None),
                        f"≥ {stop_threshold}",
                        _k(live.stop_cutoff if live else designed_stop_k),
                        "-" if live is None else str(live.n_trunc),
                        _mark(stop_status),
                    ],
                ],
                right={1, 2, 3, 4},
            ),
            "",
            "## Subset worlds",
            "",
            *_md_table(
                ["World", "Acc", "Stop ratio"],
                [
                    ["Healthy (H)", pct(acc_h), pct(stop_h)],
                    [f"**{v1_label}**", pct(v1_acc), pct(v1_stop)],
                    ["Pass threshold", pct(acc_pass), pct(stop_pass)],
                    ["Regressed (R)", pct(acc_r), pct(stop_r)],
                ],
                right={1, 2},
            ),
            "",
            "## Configurations",
            "",
            *_md_table(
                [
                    "Config",
                    "n",
                    "m",
                    "Cost",
                    "Stop SNR",
                    "Acc SNR",
                    "k_stop",
                    "k_acc",
                ],
                [
                    [
                        f"`{cfg.spec.name}`",
                        str(len(cfg.spec.prompt_ids)),
                        str(cfg.n_repeats),
                        f"**{cfg.cost}**",
                        f"{cfg.stop_snr:.3f}",
                        f"{cfg.acc_snr:.3f}",
                        _k(cfg.stop_cutoff),
                        _k(cfg.acc_cutoff),
                    ]
                    for cfg in _catalog_rows(rows, selected)
                ],
                right={1, 2, 3, 4, 5, 6, 7},
            ),
            "",
        ]
    )
    path = work_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _selected_bullets(
    selected: ScoredGate, rows: list[ScoredGate]
) -> list[str]:
    lines = [
        f"- `{selected.spec.name}` ids={selected.spec.prompt_ids}",
        "- metrics: "
        + ", ".join(
            name
            for name, on in (
                ("stop ratio", selected.spec.want_stop),
                ("accuracy", selected.spec.want_acc),
            )
            if on
        ),
    ]
    if selected.spec.want_stop:
        lines.append(
            f"- stop ratio park {selected.stop_h:.4%} → {selected.stop_r:.4%} "
            f"(H pinned, R = H - {selected.spec.delta_stop:.2%})"
        )
    if selected.spec.want_acc:
        lines.append(
            f"- acc park {selected.acc_h:.4%} → {selected.acc_r:.4%} "
            f"(H pinned, R = H - {selected.spec.delta_acc:.2%})"
        )
    designed = next(
        (cfg for cfg in rows if cfg.spec.name == selected.spec.name), None
    )
    if designed is not None and (
        designed.n_repeats != selected.n_repeats
        or designed.spec.prompt_ids != selected.spec.prompt_ids
    ):
        lines.append(
            f"- override: designed n={len(designed.spec.prompt_ids)} "
            f"m={designed.n_repeats} cost={designed.cost}; "
            f"smoke n={len(selected.spec.prompt_ids)} m={selected.n_repeats} "
            f"cost={selected.cost}"
        )
    lines.append(
        f"- n={len(selected.spec.prompt_ids)} m={selected.n_repeats} "
        f"cost={selected.cost} stop ratio k={_k(selected.stop_cutoff)} "
        f"acc k={_k(selected.acc_cutoff)}"
    )
    return lines
