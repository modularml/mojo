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
"""Dataset prep and score extraction for the OmniDocBench eval.

OmniDocBench is scored by a third-party harness (``pdf_validation.py``) that
runs from its own pinned checkout under Python 3.11, so unlike the other
dataset evals it is driven by a shell runner
(``run_omnidocbench_local.sh``) rather than a single Python module. This holds
the pieces of that flow worth testing, de-embedded from the CI-workflow
heredocs they used to live in: dataset download, subsetting, empty-prediction
detection, and score extraction.

Each function is exposed as a subcommand so the runner can call it, and stays
pure enough (filesystem in, filesystem out) to unit-test without a server.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any

from huggingface_hub import snapshot_download

DATASET_REPO = "opendatalab/OmniDocBench"

# ``notebook_metric_summary.overall_notebook`` is on a 0-100 scale and averages
# text Edit_dist, display-formula CDM, and table TEDS. The harness leaves it
# null when any component is missing (src/runtime/eval_report.py) — most often
# CDM, which needs the LaTeX render toolchain.
SUMMARY_KEY = "notebook_metric_summary"
OVERALL_KEY = "overall_notebook"


def image_stem(image_path: str) -> str:
    """Returns the prediction stem for a ground-truth ``image_path``.

    Predictions are one ``<stem>.md`` per image, so the stem is the join key
    between the ground truth and the prediction directory.
    """
    return os.path.splitext(os.path.basename(image_path))[0]


def download_dataset(
    local_dir: str, token: str | None = None, attempts: int = 5
) -> int:
    """Downloads the ground-truth JSON + page images, returning the GT count.

    Retries with linear backoff: the dataset is several GB of individually
    fetched images, and an unauthenticated puller reliably trips HTTP 429
    partway through. ``snapshot_download`` resumes, so each attempt only picks
    up what is still missing.

    Raises:
        RuntimeError: Every attempt failed.
    """
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            snapshot_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=["OmniDocBench.json", "images/*"],
                token=token,
            )
            break
        except Exception as e:  # any transport error is worth retrying
            last_err = e
            print(
                f"  download failed (attempt {attempt + 1}/{attempts}): "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            if attempt < attempts - 1:
                time.sleep(15 * (attempt + 1))
    else:
        raise RuntimeError(
            f"could not download {DATASET_REPO} after {attempts} attempts: "
            f"{last_err}"
        )

    with open(os.path.join(local_dir, "OmniDocBench.json")) as f:
        return len(json.load(f))


def missing_images(local_dir: str) -> list[str]:
    """Returns ground-truth images absent from ``local_dir``.

    The JSON is one small file among thousands of images, so it lands early: a
    partial download leaves it present while images are still missing. Checking
    it as a sentinel would silently score a truncated dataset, so completeness
    is measured against what the ground truth actually references.
    """
    gt_path = os.path.join(local_dir, "OmniDocBench.json")
    if not os.path.exists(gt_path):
        return ["OmniDocBench.json"]
    with open(gt_path) as f:
        gt = json.load(f)
    image_dir = os.path.join(local_dir, "images")
    return [
        name
        for name in (
            os.path.basename(s.get("page_info", {}).get("image_path", ""))
            for s in gt
        )
        if name and not os.path.exists(os.path.join(image_dir, name))
    ]


def select_images(image_dir: str, limit: int | None) -> list[str]:
    """Takes ``limit`` image paths evenly strided across ``image_dir``.

    Even striding (rather than a head slice) keeps a subset spread across the
    dataset's page types instead of concentrating on whichever prefix sorts
    first. ``None`` returns every image.
    """
    images = sorted(
        p
        for p in glob.glob(os.path.join(image_dir, "*"))
        if p.lower().endswith((".jpg", ".png"))
    )
    if not limit:
        return images
    step = max(1, len(images) // limit)
    return images[::step][:limit]


def filter_ground_truth(
    gt: list[dict[str, Any]], stems: set[str]
) -> list[dict[str, Any]]:
    """Keeps only the ground-truth samples whose image is in ``stems``.

    The scorer walks the ground truth, not the prediction directory, so a page
    with no prediction scores zero instead of being skipped. Predicting a
    subset therefore has to narrow the ground truth by the same set, or the
    unpredicted remainder drags the score toward zero.
    """
    return [
        sample
        for sample in gt
        if image_stem(sample.get("page_info", {}).get("image_path", ""))
        in stems
    ]


def find_empty_predictions(pred_dir: str, images: list[str]) -> list[str]:
    """Returns the images whose prediction is missing or zero-length.

    Upstream's ``gpt_4o_inf.py`` swallows request failures and writes an empty
    ``.md``, which scores zero rather than erroring — indistinguishable from a
    page the model genuinely had nothing to say about, so these get retried.
    """
    empty = []
    for image in images:
        pred = os.path.join(pred_dir, image_stem(image) + ".md")
        if not os.path.exists(pred) or os.path.getsize(pred) == 0:
            empty.append(image)
    return empty


def link_images(images: list[str], link_dir: str) -> int:
    """Rebuilds ``link_dir`` as symlinks to ``images``, returning the count.

    Upstream's inference script takes a directory, not a file list, so both
    subsetting and retrying work by pointing it at a directory of symlinks.
    Stale links are cleared first so a retry round can't re-run a page that
    succeeded in the previous one.
    """
    os.makedirs(link_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(link_dir, "*")):
        os.remove(stale)
    for image in images:
        os.symlink(
            os.path.abspath(image),
            os.path.join(link_dir, os.path.basename(image)),
        )
    return len(images)


def _summary(summary_path: str) -> dict[str, Any]:
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path) as f:
        return json.load(f).get(SUMMARY_KEY, {}) or {}


def extract_score(summary_path: str) -> float | None:
    """Reads the harness's overall score, or ``None`` when it produced none."""
    raw = _summary(summary_path).get(OVERALL_KEY)
    return None if raw is None else float(raw)


def extract_metrics(summary_path: str) -> dict[str, Any]:
    """Returns the per-component notebook metrics behind the overall score."""
    return _summary(summary_path).get("metrics", {}) or {}


def cdm_silently_zero(metrics: dict[str, Any]) -> bool:
    """Reports a zero CDM that the other metrics contradict.

    CDM identifies formula tokens by rendering each in a distinct color and
    recovering bounding boxes by color, so a render toolchain too old to
    preserve that color (ImageMagick 6.9.11 on Ubuntu 22.04 writes the render
    out as grayscale) scores every sample 0 without raising. The failure is
    invisible in isolation — 0.0 looks like a real score — but it drags the
    overall down by ~29 points, so it must not pass silently. A run whose text
    and table metrics are healthy cannot legitimately have scored 0 on every
    formula.
    """
    cdm = metrics.get("display_formula_CDM") or {}
    if not cdm.get("page_denominator") or cdm.get("notebook_value") != 0.0:
        return False
    teds = (metrics.get("table_TEDS") or {}).get("notebook_value")
    text = (metrics.get("text_block_Edit_dist") or {}).get("notebook_value")
    healthy_tables = isinstance(teds, (int, float)) and teds > 50
    healthy_text = isinstance(text, (int, float)) and text < 0.5
    return healthy_tables or healthy_text


def metrics_without_samples(metrics: dict[str, Any]) -> list[str]:
    """Names the components the run had nothing to score.

    A component with a zero page denominator nulls the overall score. On a
    subset run that is expected — four sampled pages may contain no display
    formula at all — and says nothing about whether the metric works.
    """
    return sorted(
        name
        for name, m in metrics.items()
        if not (m or {}).get("page_denominator")
    )


def build_score_json(
    score: float | None,
    baseline: float,
    total: int,
    partial: bool,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds the ``score.json`` payload for the suite collector.

    ``score`` is renamed to ``accuracy`` because ``collect_scores.py`` reads
    that key first; ``partial`` marks a subset run whose denominator differs
    from the published baseline's, so a smoke number is never mistaken for a
    comparable one. The component metrics ride along so a null overall can be
    diagnosed from the artifact alone.
    """
    passed = score is not None and score >= baseline * 0.98 and not partial
    return {
        "accuracy": score,
        "total": total,
        "baseline": baseline,
        "threshold": baseline * 0.98,
        "passed": passed,
        "partial": partial,
        "metrics": metrics or {},
    }


def _cmd_download(args: argparse.Namespace) -> int:
    count = download_dataset(args.local_dir, os.environ.get("HF_TOKEN"))
    absent = missing_images(args.local_dir)
    if absent:
        print(
            f"error: {len(absent)} of {count} ground-truth images are still "
            f"missing after download, e.g. {absent[:3]}",
            file=sys.stderr,
        )
        return 1
    print(f"Downloaded {count} ground truth samples and images")
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    """Selects the images to run and writes the matching ground truth."""
    images = select_images(args.image_dir, args.limit)
    if not images:
        print(f"error: no images found in {args.image_dir}", file=sys.stderr)
        return 1
    link_images(images, args.link_dir)
    with open(args.gt) as f:
        gt = json.load(f)
    if args.limit:
        stems = {image_stem(p) for p in images}
        gt = filter_ground_truth(gt, stems)
    with open(args.gt_out, "w") as f:
        json.dump(gt, f)
    print(f"{len(images)} images, {len(gt)} ground truth samples")
    return 0


def _cmd_find_empty(args: argparse.Namespace) -> int:
    images = select_images(args.link_dir, None)
    empty = find_empty_predictions(args.pred_dir, images)
    if empty:
        link_images(empty, args.retry_dir)
    print(len(empty))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    score = extract_score(args.summary)
    metrics = extract_metrics(args.summary)
    payload = build_score_json(
        score, args.baseline, args.total, args.partial, metrics
    )
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    if cdm_silently_zero(metrics):
        print(
            "::warning::display_formula_CDM scored 0 on every sample while the "
            "text and table metrics look healthy — the LaTeX/ImageMagick render "
            "stack is almost certainly too old. CDM is one of the three terms "
            "averaged into the overall, so at CI's CDM of ~84.8 this costs the "
            "overall roughly 28 points.\n"
            "  Known-good (CI): Ubuntu 24.04, ImageMagick 6.9.12, TeX Live 2023.\n"
            "  A CDM of 0 here is an environment problem, not a model result.",
            file=sys.stderr,
        )

    if score is not None:
        label = " (PARTIAL — subset run, not comparable to the baseline)"
        print(f"OmniDocBench: {score:.2f}{label if args.partial else ''}")
        return 0

    def component_lines() -> str:
        return "\n".join(
            f"    {name}: {(m or {}).get('notebook_value')} "
            f"(pages={(m or {}).get('page_denominator')})"
            for name, m in sorted(metrics.items())
        )

    unsampled = metrics_without_samples(metrics)
    # A subset that happens to contain none of a component's elements nulls the
    # overall score without anything being wrong; only report it as a failure
    # when the run was supposed to cover the whole set.
    if args.partial and unsampled:
        print(
            "OmniDocBench: no overall score — this subset contained no samples "
            f"for {', '.join(unsampled)}, so the harness cannot average an "
            "overall. Expected on a small --limit; components that did score:\n"
            f"{component_lines()}"
        )
        return 0

    print(
        f"OmniDocBench: no overall score — the harness left {OVERALL_KEY} null."
        + (
            f"\n  Components with no samples: {', '.join(unsampled)}."
            if unsampled
            else "\n  All components had samples, so a metric failed outright "
            "— check the scoring log for LaTeX/ImageMagick errors."
        )
        + (f"\n  Component metrics:\n{component_lines()}" if metrics else ""),
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="Fetch the ground truth JSON + images.")
    p.add_argument("--local-dir", required=True)
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("prepare", help="Pick images and subset ground truth.")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--link-dir", required=True)
    p.add_argument("--gt-out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=_cmd_prepare)

    p = sub.add_parser("find-empty", help="Count/relink empty predictions.")
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--link-dir", required=True)
    p.add_argument("--retry-dir", required=True)
    p.set_defaults(func=_cmd_find_empty)

    p = sub.add_parser("score", help="Extract the score into score.json.")
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--baseline", type=float, default=89.5)
    p.add_argument("--total", type=int, default=0)
    p.add_argument("--partial", action="store_true")
    p.set_defaults(func=_cmd_score)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
