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
"""CLI, eval wrapper, and report for the GPQA logit-shift gate."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import click

from calibration.gpqa_gate import (
    DEFAULT_CAP,
    DEFAULT_DELTA_ACC,
    DEFAULT_DELTA_STOP,
    HIST_MODES,
    SUBSETS,
    LiveVerdict,
    ScoredGate,
    make_spec,
    score_catalog,
    score_results,
    score_spec,
)

GPQA_EVAL = "//max/tests/integration/accuracy/model_evals:gpqa_eval"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _parse_row_ids(raw: str) -> list[int]:
    ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not ids:
        raise click.UsageError("--row-ids is empty")
    return ids


def scored_to_dict(cfg: ScoredGate) -> dict[str, object]:
    spec = cfg.spec
    return {
        "name": spec.name,
        "hist_mode": spec.hist_mode,
        "subset": spec.subset,
        "prompt_ids": list(spec.prompt_ids),
        "n_subset": len(spec.prompt_ids),
        "n_repeats": cfg.n_repeats,
        "cost": cfg.cost,
        "alpha": spec.alpha,
        "beta": spec.beta,
        "delta_stop": spec.delta_stop,
        "delta_acc": spec.delta_acc,
        "want_stop": spec.want_stop,
        "want_acc": spec.want_acc,
        "stop_cutoff": cfg.stop_cutoff,
        "acc_cutoff": cfg.acc_cutoff,
        "stop_fp": cfg.stop_fp,
        "stop_fn": cfg.stop_fn,
        "acc_fp": cfg.acc_fp,
        "acc_fn": cfg.acc_fn,
        "stop_snr": cfg.stop_snr,
        "acc_snr": cfg.acc_snr,
        "base_stop": cfg.base_stop,
        "stop_h": cfg.stop_h,
        "stop_r": cfg.stop_r,
        "base_acc": cfg.base_acc,
        "acc_h": cfg.acc_h,
        "acc_r": cfg.acc_r,
    }


def print_catalog(rows: list[ScoredGate]) -> None:
    if not rows:
        return
    click.echo(
        f"{'name':<28} {'n':>4} {'m':>5} {'cost':>7} {'stopSNR':>8} {'accSNR':>8}"
    )
    for cfg in rows:
        click.echo(
            f"{cfg.spec.name:<28} {len(cfg.spec.prompt_ids):>4} "
            f"{cfg.n_repeats:>5} {cfg.cost:>7} {cfg.stop_snr:>8.3f} "
            f"{cfg.acc_snr:>8.3f}"
        )


def run_gpqa_eval(
    *,
    base_url: str,
    model: str,
    prompt_ids: list[int],
    n_repeats: int,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(_repo_root() / "bazelw"),
        "run",
        "--ui_event_filters=-info",
        "--noshow_progress",
        "--curses=no",
        GPQA_EVAL,
        "--",
        "--base-url",
        base_url,
        "--model",
        model,
        "--row-ids",
        ",".join(str(i) for i in prompt_ids),
        "--repeats",
        str(n_repeats),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=_repo_root(), check=True)
    results = out_dir / "results.jsonl"
    if not results.is_file():
        raise FileNotFoundError(f"gpqa_eval did not write {results}")
    return results


def _svg(
    path: Path, title: str, points: list[tuple[str, float, float]], ylabel: str
) -> None:
    xs = [max(p[1], 1) for p in points]
    ys = [p[2] for p in points]
    pad, w, h = 56, 640, 320
    xmax, ymax = max(xs) * 1.15, (max(ys) or 1.0) * 1.25
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="{w / 2}" y="20" text-anchor="middle" font-size="14">{title}</text>',
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - 24}" y2="{h - pad}" stroke="#333"/>',
        f'<line x1="{pad}" y1="36" x2="{pad}" y2="{h - pad}" stroke="#333"/>',
        f'<text x="14" y="{h / 2}" font-size="11" transform="rotate(-90 14 {h / 2})">{ylabel}</text>',
    ]
    for name, x, y in points:
        px = pad + x / xmax * (w - pad - 24)
        py = (h - pad) - y / ymax * (h - pad - 36)
        body.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="#2563eb"/>'
        )
        body.append(
            f'<text x="{px + 7:.1f}" y="{py - 6:.1f}" font-size="10">{name}</text>'
        )
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n")


def write_report(
    work_dir: Path,
    rows: list[ScoredGate],
    selected: ScoredGate | None,
    live: LiveVerdict | None,
) -> Path:
    plots = work_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_rows = rows or ([selected] if selected is not None else [])
    if plot_rows:
        _svg(
            plots / "cost_vs_snr.svg",
            "Cost vs stop SNR",
            [(c.spec.name, float(c.cost), c.stop_snr) for c in plot_rows],
            "stop SNR",
        )
    if live is not None and selected is not None and live.status != "error":
        observed: list[tuple[str, float, float]] = []
        if selected.stop_cutoff is not None:
            observed.extend(
                [
                    ("stop S", 1.0, float(live.n_trunc)),
                    ("stop k", 2.0, float(selected.stop_cutoff)),
                ]
            )
        if selected.acc_cutoff is not None:
            observed.extend(
                [
                    ("acc S", 3.0, float(live.n_wrong)),
                    ("acc k", 4.0, float(selected.acc_cutoff)),
                ]
            )
        if observed:
            _svg(
                plots / "observed_s.svg",
                "Observed S vs cutoff",
                observed,
                "count",
            )
    if live is None:
        metrics = []
        if selected is None or selected.spec.want_stop:
            metrics.append("a 0.5pp stop drop (98.5% → 98.0%)")
        if selected is None or selected.spec.want_acc:
            metrics.append("a 1pp accuracy drop (92.5% → 91.5%)")
        joint = (
            "Joint m is max of the enabled solvers. "
            if selected is None
            or (selected.spec.want_stop and selected.spec.want_acc)
            else "Repeats are sized from the enabled metric only. "
        )
        verdict, rationale = (
            "compare-only",
            (
                "No live eval. Default is bins x noisy_15 detecting "
                + " and ".join(metrics)
                + f" at 1% caps. {joint}"
                "plus_bucket adds the 12 rares at ≥3%; ever_trunc is all 49."
            ),
        )
    else:
        mark = {"pass": "✅ pass", "fail": "❌ fail", "error": "⚠️ error"}
        verdict, rationale = mark[live.status], live.rationale
    lines = [
        "# GPQA logit-gate report",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        rationale,
        "",
        "## Configurations",
        "",
        "| Config | n | m | Cost | Stop SNR | Acc SNR | k_stop | k_acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def _k(value: int | None) -> str:
        return "-" if value is None else str(value)

    table_rows = rows or ([selected] if selected is not None else [])
    for cfg in table_rows:
        lines.append(
            f"| `{cfg.spec.name}` | {len(cfg.spec.prompt_ids)} | {cfg.n_repeats} | "
            f"**{cfg.cost}** | {cfg.stop_snr:.3f} | {cfg.acc_snr:.3f} | "
            f"{_k(cfg.stop_cutoff)} | {_k(cfg.acc_cutoff)} |"
        )
    if selected is not None:
        selected_lines = [
            "",
            "## Selected",
            "",
            f"- `{selected.spec.name}` ids={selected.spec.prompt_ids}",
            "- metrics: "
            + ", ".join(
                name
                for name, on in (
                    ("stop", selected.spec.want_stop),
                    ("accuracy", selected.spec.want_acc),
                )
                if on
            ),
        ]
        if selected.spec.want_stop:
            selected_lines.append(
                f"- stop park {selected.stop_h:.4%} → {selected.stop_r:.4%} "
                f"(H pinned, R = H - {selected.spec.delta_stop:.2%})"
            )
        if selected.spec.want_acc:
            selected_lines.append(
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
            selected_lines.append(
                f"- override: designed n={len(designed.spec.prompt_ids)} "
                f"m={designed.n_repeats} cost={designed.cost}; "
                f"smoke n={len(selected.spec.prompt_ids)} m={selected.n_repeats} "
                f"cost={selected.cost}"
            )
        selected_lines.extend(
            [
                f"- n={len(selected.spec.prompt_ids)} m={selected.n_repeats} "
                f"cost={selected.cost} stop k={_k(selected.stop_cutoff)} "
                f"acc k={_k(selected.acc_cutoff)}",
                "",
            ]
        )
        lines.extend(selected_lines)
    lines.extend(["## Plots", ""])
    if (plots / "cost_vs_snr.svg").exists():
        lines.append("- [Cost vs stop SNR](plots/cost_vs_snr.svg)")
    if (plots / "observed_s.svg").exists():
        lines.append("- [Observed S vs cutoff](plots/observed_s.svg)")
    path = work_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_gate(
    *,
    work_dir: Path,
    hist_mode: str,
    subset: str,
    compare_only: bool = False,
    base_url: str | None = None,
    model: str | None = None,
    results_jsonl: Path | None = None,
    alpha: float = DEFAULT_CAP,
    beta: float = DEFAULT_CAP,
    delta_stop: float = DEFAULT_DELTA_STOP,
    delta_acc: float = DEFAULT_DELTA_ACC,
    want_stop: bool = True,
    want_acc: bool = True,
    n_repeats: int | None = None,
    prompt_ids: list[int] | None = None,
    include_catalog: bool = True,
) -> LiveVerdict | None:
    if not want_stop and not want_acc:
        raise click.UsageError("need --stop and/or --acc")
    work_dir.mkdir(parents=True, exist_ok=True)
    rows = (
        []
        if not include_catalog
        else score_catalog(
            alpha=alpha,
            beta=beta,
            delta_stop=delta_stop,
            delta_acc=delta_acc,
            want_stop=want_stop,
            want_acc=want_acc,
        )
    )
    (work_dir / "catalog.json").write_text(
        json.dumps([scored_to_dict(c) for c in rows], indent=2) + "\n"
    )
    selected = score_spec(
        make_spec(
            hist_mode,
            subset,
            alpha=alpha,
            beta=beta,
            delta_stop=delta_stop,
            delta_acc=delta_acc,
            want_stop=want_stop,
            want_acc=want_acc,
            prompt_ids=prompt_ids,
        ),
        n_repeats=n_repeats,
    )
    print_catalog(rows or [selected])
    (work_dir / "selected.json").write_text(
        json.dumps(scored_to_dict(selected), indent=2) + "\n"
    )
    live: LiveVerdict | None = None
    if not compare_only:
        if results_jsonl is not None:
            results = results_jsonl
        else:
            if not base_url or not model:
                raise click.UsageError("live eval needs --base-url and --model")
            results = run_gpqa_eval(
                base_url=base_url,
                model=model,
                prompt_ids=selected.spec.prompt_ids,
                n_repeats=selected.n_repeats,
                out_dir=work_dir / "eval",
            )
        live = score_results(results, selected)
        (work_dir / "verdict.json").write_text(
            json.dumps(asdict(live), indent=2) + "\n"
        )
        click.echo(f"[{live.status}] {live.rationale}")
    report = write_report(work_dir, rows, selected, live)
    click.echo(f"Wrote report: {report}")
    if live is not None and live.status == "fail":
        raise SystemExit(1)
    if live is not None and live.status == "error":
        raise SystemExit(2)
    return live


@click.command()
@click.option("--list", "list_configs", is_flag=True)
@click.option("--compare-only", is_flag=True)
@click.option(
    "--hist",
    "hist_mode",
    type=click.Choice(HIST_MODES),
    default="bins",
    show_default=True,
)
@click.option(
    "--subset",
    type=click.Choice(SUBSETS),
    default="noisy_15",
    show_default=True,
)
@click.option("--base-url", default=None)
@click.option("--model", default=None)
@click.option("--work-dir", type=click.Path(path_type=Path), default=None)
@click.option(
    "--results-jsonl",
    type=click.Path(path_type=Path, exists=True),
    default=None,
)
@click.option("--alpha", type=float, default=DEFAULT_CAP, show_default=True)
@click.option("--beta", type=float, default=DEFAULT_CAP, show_default=True)
@click.option(
    "--delta-stop", type=float, default=DEFAULT_DELTA_STOP, show_default=True
)
@click.option(
    "--delta-acc", type=float, default=DEFAULT_DELTA_ACC, show_default=True
)
@click.option("--stop/--no-stop", "want_stop", default=True, show_default=True)
@click.option("--acc/--no-acc", "want_acc", default=True, show_default=True)
@click.option(
    "--repeats", type=int, default=None, help="Override designed m (smoke)."
)
@click.option(
    "--row-ids", default=None, help="Comma-separated prompt indexes (smoke)."
)
@click.option(
    "--catalog/--no-catalog", "include_catalog", default=True, show_default=True
)
def main(
    list_configs: bool,
    compare_only: bool,
    hist_mode: str,
    subset: str,
    base_url: str | None,
    model: str | None,
    work_dir: Path | None,
    results_jsonl: Path | None,
    alpha: float,
    beta: float,
    delta_stop: float,
    delta_acc: float,
    want_stop: bool,
    want_acc: bool,
    repeats: int | None,
    row_ids: str | None,
    include_catalog: bool,
) -> None:
    """Runs the GPQA stop and/or accuracy logit gate."""
    if not want_stop and not want_acc:
        raise click.UsageError("need --stop and/or --acc")
    prompt_ids = _parse_row_ids(row_ids) if row_ids is not None else None
    if list_configs:
        print_catalog(score_catalog(want_stop=want_stop, want_acc=want_acc))
        return
    if work_dir is None:
        raise click.UsageError("--work-dir is required unless --list")
    live = results_jsonl is not None or (
        base_url is not None and model is not None
    )
    run_gate(
        work_dir=work_dir,
        hist_mode=hist_mode,
        subset=subset,
        compare_only=compare_only or not live,
        base_url=base_url,
        model=model,
        results_jsonl=results_jsonl,
        alpha=alpha,
        beta=beta,
        delta_stop=delta_stop,
        delta_acc=delta_acc,
        want_stop=want_stop,
        want_acc=want_acc,
        n_repeats=repeats,
        prompt_ids=prompt_ids,
        include_catalog=include_catalog,
    )


if __name__ == "__main__":
    main()
