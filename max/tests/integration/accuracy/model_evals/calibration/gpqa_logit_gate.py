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
"""CLI, eval wrapper, and report for the GPQA logit-shift gate."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, replace
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
    rate_from_k,
    score_catalog,
    score_results,
    score_spec,
    subset_park_rates,
)
from calibration.gpqa_gate_report import write_report

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
    acc_h, acc_r, stop_h, stop_r = subset_park_rates(spec)
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
        "subset_acc_h": acc_h,
        "subset_acc_r": acc_r,
        "subset_stop_h": stop_h,
        "subset_stop_r": stop_r,
        "acc_rate_cutoff": rate_from_k(cfg.acc_cutoff, cfg.cost),
        "stop_rate_cutoff": rate_from_k(cfg.stop_cutoff, cfg.cost),
    }


def print_catalog(rows: list[ScoredGate]) -> None:
    if not rows:
        return
    click.echo(
        f"{'name':<28} {'n':>4} {'m':>5} {'cost':>7} "
        f"{'stopSNR':>8} {'accSNR':>8}"
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
        live = replace(
            score_results(results, selected),
            model=model,
            base_url=base_url,
        )
        (work_dir / "verdict.json").write_text(
            json.dumps(asdict(live), indent=2) + "\n"
        )
        click.echo(f"[{live.status}] {live.rationale}")
    report = write_report(
        work_dir, rows, selected, live, model=model, base_url=base_url
    )
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
    default="per_prompt",
    show_default=True,
)
@click.option(
    "--subset",
    type=click.Choice(SUBSETS),
    default="plus_bucket",
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
    "--catalog/--no-catalog",
    "include_catalog",
    default=True,
    show_default=True,
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
