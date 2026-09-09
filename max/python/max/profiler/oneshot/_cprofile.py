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

"""Python/CPU ``cProfile`` rendering with bazel-aware filename trimming."""

from __future__ import annotations

import cProfile
import logging
import marshal
import os
from typing import TextIO, TypeAlias

from ._format import _format_ns

logger = logging.getLogger("max.profiler.oneshot")

# Shape of ``cProfile.Profile.stats`` and the output of ``marshal.load`` on a
# file written by ``prof.dump_stats``: key is ``(filename, lineno, funcname)``,
# value is ``(call_count, recursive_call_count, total_time, cumulative_time,
# callers_dict)``. The time fields are float at runtime but typeshed declares
# them as int on ``cProfile.Profile.stats``; we match typeshed so callers can
# pass ``prof.stats`` directly, and cast to float when actually using the
# values.
_Label: TypeAlias = tuple[str, int, str]
_CProfileStats: TypeAlias = dict[
    _Label,
    tuple[int, int, int, int, dict[_Label, tuple[int, int, int, int]]],
]


def _strip_bazel_path(filename: str) -> str:
    """Strip bazel-cache and runfiles prefixes to leave a readable suffix.

    cProfile reports absolute filenames such as
    ``/home/.../bazel-out/.../<bin>.runfiles/_main/max/python/.../generate.py``
    or ``/.../external/rules_python.../python3.13/asyncio/base_events.py``.
    Returns the human-relevant suffix (e.g. ``max/python/.../generate.py``
    or ``asyncio/base_events.py``).
    """
    # Bazel runfiles: ".../<bin>.runfiles/<repo>/<actual path>". Strip the
    # ``.runfiles/`` prefix and the repo segment (``_main``, an external
    # repo name, etc.).
    rf = filename.rfind(".runfiles/")
    if rf >= 0:
        tail = filename[rf + len(".runfiles/") :]
        slash = tail.find("/")
        if slash >= 0:
            return tail[slash + 1 :]
    # site-packages installs.
    sp = filename.rfind("/site-packages/")
    if sp >= 0:
        return filename[sp + len("/site-packages/") :]
    # External Python stdlib paths under bazel: ``.../python3.13/asyncio/...``.
    py_idx = filename.rfind("/python3.")
    if py_idx >= 0:
        tail = filename[py_idx + 1 :]
        slash = tail.find("/")
        if slash >= 0:
            return tail[slash + 1 :]
    return filename


def _render_cprofile(prof: cProfile.Profile, top_n: int, out: TextIO) -> None:
    """Render a top-N table of Python functions by exclusive (own) time.

    Sums ``tottime`` (time in the function excluding callees) across all
    profiled functions to get a denominator, then ranks. This matches the
    semantics of the GPU kernel summary above and avoids the
    cumulative-time double-counting that affects recursive functions.
    """
    # Populate ``prof.stats`` from the underlying ``_lsprof`` data. Read
    # ``prof.stats`` directly rather than going through ``pstats.Stats``
    # because typeshed's ``pstats.Stats`` stub omits the (public, documented)
    # ``.stats`` attribute, while ``cProfile.Profile.stats`` is typed.
    prof.create_stats()
    _render_cprofile_stats(prof.stats, top_n, out)


def _render_cprofile_from_dump(path: str, top_n: int, out: TextIO) -> bool:
    """Load a ``prof.dump_stats``-format file and render the top-N table.

    Returns True if the file existed and was rendered. The companion writer
    is :func:`cProfile.Profile.dump_stats`, used from the nsys child so the
    parent can render its output after nsys has finished writing the
    ``.nsys-rep``. Missing or malformed dumps are logged at WARNING and
    return False rather than crashing the parent.
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            stats = marshal.load(f)
    except (OSError, EOFError, ValueError) as e:
        logger.warning("Failed to load cProfile dump %s: %s", path, e)
        return False
    _render_cprofile_stats(stats, top_n, out)
    return True


def _render_cprofile_stats(
    stats: _CProfileStats, top_n: int, out: TextIO
) -> None:
    """Render a top-N table from a populated cProfile ``stats`` dict."""
    # Each row carries (label, calls_str, tottime). ``calls_str`` follows the
    # standard pstats convention: ``cc/nc`` when primitive and total counts
    # differ (i.e. the function recurses), otherwise just ``nc``.
    rows: list[tuple[str, str, float]] = []
    for func, (cc, nc, tt, _ct, _callers) in stats.items():
        filename, lineno, funcname = func
        path = _strip_bazel_path(filename)
        if funcname.startswith("<") and funcname.endswith(">"):
            label = f"{path}:{lineno}{funcname}"
        else:
            label = f"{path}:{lineno}({funcname})"
        calls_str = f"{cc}/{nc}" if cc != nc else str(nc)
        rows.append((label, calls_str, float(tt)))

    if not rows:
        out.write("\n=== Python/CPU functions: no samples captured ===\n")
        return

    total_s = sum(r[2] for r in rows)
    rows.sort(key=lambda r: r[2], reverse=True)
    rows = rows[:top_n]
    shown_s = sum(r[2] for r in rows)
    pct_shown = shown_s / max(total_s, 1e-12) * 100

    out.write(
        f"\n=== Top {len(rows)} Python/CPU functions "
        f"({pct_shown:.1f}% of CPU time, "
        f"{_format_ns(total_s * 1e9).strip()} total) ===\n"
    )
    out.write(f"{'%':>6}  {'total':>10}  {'calls':>8}  function\n")
    for name, calls_str, row_tt in rows:
        pct = row_tt / max(total_s, 1e-12) * 100
        out.write(
            f"{pct:>5.1f}%  {_format_ns(row_tt * 1e9)}  {calls_str:>8}  {name}\n"
        )
