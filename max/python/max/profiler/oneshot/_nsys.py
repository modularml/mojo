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

"""Nsight Systems re-exec and ``cuda_gpu_kern_sum`` rendering."""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
from typing import TextIO, TypedDict

from ._backend import _CPROFILE_DUMP_PATH_ENV_VAR, _NSYS_CHILD_ENV_VAR
from ._format import _format_ns

logger = logging.getLogger("max.profiler.oneshot")


class KernelRow(TypedDict):
    """One row of the ``cuda_gpu_kern_sum`` summary."""

    name: str
    total_ns: float
    instances: int


def _reexec_under_nsys(
    output_path: str, *, cprofile_dump_path: str | None = None
) -> int:
    """Re-launch the current process under ``nsys profile``.

    ``cprofile_dump_path``, if given, is propagated to the child via env so
    the child writes its cProfile stats there instead of rendering them
    inline. The parent then renders that file after nsys finishes writing
    the ``.nsys-rep`` — see :func:`maybe_reexec_under_nsys` for the
    ordering rationale.

    Returns the child process exit code so the caller can propagate it.
    """
    cmd = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--output",
        output_path,
        "--force-overwrite=true",
        sys.executable,
        *sys.argv,
    ]
    env = os.environ.copy()
    env[_NSYS_CHILD_ENV_VAR] = "1"
    if cprofile_dump_path is not None:
        env[_CPROFILE_DUMP_PATH_ENV_VAR] = cprofile_dump_path
    # Default to ``detailed`` so the runtime emits Python-level NVTX markers
    # (the ``Tracer("inference")`` span and the per-op markers in the MAX
    # stack). Without this, ``is_profiling_enabled()`` returns False and
    # every Tracer / @traced span is a no-op, leaving the .nsys-rep with no
    # NVTX correlation. Respect a value the user already exported.
    env.setdefault("MODULAR_ENABLE_PROFILING", "detailed")
    logger.info("Launching under nsys: %s", " ".join(cmd))
    return subprocess.run(cmd, env=env, check=False).returncode


def render_nsys_kernel_summary(
    nsys_rep_path: str, top_n: int, out: TextIO
) -> bool:
    """Run ``nsys stats`` against a report and render a top-N kernel table.

    Returns True if a non-empty table was rendered.
    """
    try:
        result = subprocess.run(
            [
                "nsys",
                "stats",
                # Force re-derivation of the .sqlite — newer nsys (>= 2026.2)
                # refuses to re-use a stale export and otherwise just prints
                # usage instead of running the report.
                "--force-export=true",
                "--report",
                "cuda_gpu_kern_sum",
                "--format",
                "csv",
                nsys_rep_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.warning("nsys stats failed: %s", e)
        return False

    rows = _parse_nsys_csv(result.stdout)
    if not rows:
        return False

    total_ns = sum(r["total_ns"] for r in rows)
    rows.sort(key=lambda r: r["total_ns"], reverse=True)
    rows = rows[:top_n]

    out.write(
        f"\n=== Top {len(rows)} GPU kernels "
        f"({sum(r['total_ns'] for r in rows) / max(total_ns, 1) * 100:.1f}% "
        f"of GPU time, {_format_ns(total_ns).strip()} total) ===\n"
    )
    out.write(f"{'%':>6}  {'total':>10}  {'calls':>8}  kernel\n")
    for r in rows:
        pct = r["total_ns"] / max(total_ns, 1) * 100
        out.write(
            f"{pct:>5.1f}%  {_format_ns(r['total_ns'])}  "
            f"{r['instances']:>8}  {r['name']}\n"
        )
    return True


def _parse_nsys_csv(text: str) -> list[KernelRow]:
    """Parse a ``cuda_gpu_kern_sum`` CSV from ``nsys stats``.

    The exact column set is stable across recent nsys versions but we look up
    columns by name to be defensive.
    """
    # nsys stats CSV is preceded by section headers; find the row that starts
    # with "Time" or contains a "Total Time" column header.
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.startswith("**")
    ]
    if not lines:
        return []

    reader = csv.reader(lines)
    header: list[str] | None = None
    rows: list[KernelRow] = []
    for row in reader:
        if header is None:
            # First row that mentions a total-time column is the header.
            joined = ",".join(row).lower()
            if "total time" in joined or "time (%)" in joined:
                header = [c.strip() for c in row]
            continue
        if len(row) != len(header):
            continue
        record = dict(zip(header, row, strict=False))
        try:
            total_ns = _coerce_ns(record)
            instances = int(record.get("Instances", "0").replace(",", "") or 0)
            name = record.get("Name") or record.get("Kernel Name") or ""
        except (KeyError, ValueError):
            continue
        if not name:
            continue
        rows.append(
            {"name": name.strip(), "total_ns": total_ns, "instances": instances}
        )
    return rows


def _coerce_ns(record: dict[str, str]) -> float:
    """Extract total-time-in-ns from an nsys stats row."""
    for key in ("Total Time (ns)", "Total Time", "Sum (ns)"):
        if key in record:
            raw = record[key].replace(",", "").strip()
            if not raw:
                continue
            return float(raw)
    raise KeyError("no total-time column found")
