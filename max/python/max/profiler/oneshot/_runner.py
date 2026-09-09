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

"""Top-level orchestrators that wire the one-shot profiler into CLI entrypoints.

CLI handlers wire this up in two places:

1. Call :func:`maybe_reexec_under_nsys` at the very top of the click
   handler, before any model state is loaded. If Nsight Systems (``nsys``)
   is on ``PATH`` and an NVIDIA GPU is present, the current process is
   re-launched under
   ``nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop``.
   The parent waits for the child, renders a top-N kernel summary from the
   resulting ``.nsys-rep`` via ``nsys stats``, and ``sys.exit``s.
2. Drive the timed region with a :class:`OneShotCapture` handle:
   :meth:`OneShotCapture.start` right before the inference call,
   :meth:`OneShotCapture.end_and_finalize` *after* the surrounding metrics
   context exits. Splitting start and end into explicit method calls lets
   the caller place the capture-end point outside the lexical scope of
   the metrics context, so the normal generate-stats output prints before
   nsys begins writing the ``.nsys-rep``.

When ``nsys`` is not installed or no NVIDIA GPU is detected, a friendly
note is printed and the cProfile output alone is shown.

Output ordering (under nsys): ``cudaProfilerStop`` fires *after* the
generate stats are printed, so the order on the terminal is:
generate output → generate metrics → ``Capture range ended`` →
nsys ``Generating ...`` → cProfile table → GPU kernel table. The child
dumps its cProfile stats to ``profile_output + ".cprofile"`` so the
parent can render them after nsys finishes writing the ``.nsys-rep``.
"""

from __future__ import annotations

import cProfile
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import _GeneratorContextManager, contextmanager
from typing import TextIO

from ._backend import (
    _CPROFILE_DUMP_PATH_ENV_VAR,
    ProfileBackend,
    _is_under_nsys,
    detect_backend,
)
from ._cprofile import _render_cprofile, _render_cprofile_from_dump
from ._cuda import _cuda_profiler_region
from ._nsys import _reexec_under_nsys, render_nsys_kernel_summary

logger = logging.getLogger("max.profiler.oneshot")


def _cprofile_dump_path(profile_output: str) -> str:
    """Return the sibling file the nsys child uses for its cProfile dump."""
    return profile_output + ".cprofile"


def inject_trace_flags(args: Sequence[str], profile_output: str) -> list[str]:
    """Inject ``--trace`` + ``--trace-file <profile_output>`` into ``args``.

    Used by ``max benchmark --profile`` to translate the user-friendly
    ``--profile`` flag into the underlying ``--trace`` / ``--trace-file``
    flags accepted by ``benchmark_serving``. Any user-supplied
    ``--trace-file`` is stripped first so the rewritten argv has exactly
    one canonical path. ``--trace`` is added only when not already present.
    """
    out: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--trace-file":
            skip_next = True
            continue
        if arg.startswith("--trace-file="):
            continue
        out.append(arg)
    if "--trace" not in out:
        out.append("--trace")
    out.extend(["--trace-file", profile_output])
    return out


def default_profile_output() -> str:
    """Return the default ``.nsys-rep`` path for ``--profile``."""
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace:
        return os.path.join(workspace, "max-profile.nsys-rep")
    return os.path.abspath("max-profile.nsys-rep")


def maybe_reexec_under_nsys(
    profile_output: str | None = None,
    *,
    top_n: int = 15,
    backend: ProfileBackend | None = None,
    out: TextIO | None = None,
) -> None:
    """Re-exec the current process under ``nsys profile`` if appropriate.

    Call this as early as possible in the CLI handler — *before* any model
    loading, weight download, or compile work — so the parent process does
    no wasted work before the child takes over.

    Behavior:

    - If we are already running inside the nsys-spawned child (detected via
      env var), return immediately so the caller proceeds normally.
    - If GPU capture is unavailable (no nsys, no NVIDIA GPU, no libcudart),
      print a one-line explanation and return so the caller falls through
      to a ``profiled_region`` cProfile-only run.
    - Otherwise, spawn ``nsys profile`` with the same argv, wait for the
      child to exit, render the kernel summary from the resulting
      ``.nsys-rep``, and ``sys.exit`` with the child's exit code. This
      function does **not** return in that case.
    """
    out = out or sys.stdout
    backend = backend or detect_backend()
    if _is_under_nsys():
        return
    if not backend.can_capture_gpu:
        _write_no_capture_reason(backend, out)
        return

    profile_output = profile_output or default_profile_output()
    cprofile_dump = _cprofile_dump_path(profile_output)
    rc = _reexec_under_nsys(profile_output, cprofile_dump_path=cprofile_dump)
    # Render the deferred cProfile table (dumped by the child) *before* the
    # nsys kernel summary, so the order is:
    #   <normal generate output>
    #   <nsys "Generating .nsys-rep" lines>
    #   <cProfile top-N>
    #   <GPU kernel top-N>
    _render_cprofile_from_dump(cprofile_dump, top_n, out)
    try:
        os.remove(cprofile_dump)
    except OSError:
        pass
    if os.path.exists(profile_output):
        rendered = render_nsys_kernel_summary(profile_output, top_n, out)
        if not rendered:
            out.write(
                "\nNo GPU kernel data found in profile "
                f"({profile_output}). The capture window may have been "
                "empty.\n"
            )
        out.write(f"\nFull profile saved to: {profile_output}\n")
        out.write(f"Open with: nsys-ui {profile_output}\n")
    else:
        out.write(
            f"\n--profile: nsys exited with rc={rc} without writing "
            f"{profile_output}. No kernel summary produced.\n"
        )
    sys.exit(rc)


def _write_no_capture_reason(backend: ProfileBackend, out: TextIO) -> None:
    """Tell the user why GPU capture was skipped, before any work runs."""
    if not backend.nsys_available:
        out.write(
            "\nNote: Nsight Systems (`nsys`) not found on PATH; only "
            "Python/CPU profile will be captured. Install Nsight Systems "
            "and re-run for GPU kernel breakdown.\n"
        )
    elif not backend.nvidia_gpu_present:
        out.write(
            "\nNote: no NVIDIA GPU detected; only Python/CPU profile will "
            "be captured.\n"
        )
    elif not backend.cudart_loadable:
        out.write(
            "\nNote: libcudart could not be loaded; only Python/CPU "
            "profile will be captured.\n"
        )


@contextmanager
def profiled_region(
    *,
    top_n: int = 15,
    backend: ProfileBackend | None = None,
    out: TextIO | None = None,
) -> Iterator[None]:
    """Bracket a region of code with the in-process profiling layers.

    Always wraps the region with a ``cProfile.Profile()`` so a Python/CPU
    summary is printed on exit. When running inside an nsys-spawned child
    (set up via :func:`maybe_reexec_under_nsys`), additionally calls
    ``cudaProfilerStart``/``Stop`` so the nsys capture is bounded to just
    this region — excluding warmup, model compile, and weight load time.

    The cuda-profiler stop fires *before* anything is rendered to ``out``,
    so the "Capture range ended in the application" line emitted by nsys
    appears immediately after the timed region's own output rather than
    after a cProfile table. Under nsys the cProfile data is dumped to a
    sibling file (path read from
    :data:`max.profiler.oneshot._backend._CPROFILE_DUMP_PATH_ENV_VAR`) and
    rendered later by :func:`maybe_reexec_under_nsys` in the parent — see
    that function's docstring for the full ordering rationale.

    The "why no GPU capture" explanation is emitted earlier, by
    :func:`maybe_reexec_under_nsys` in the CLI flow.
    """
    out = out or sys.stdout
    backend = backend or detect_backend()
    prof = cProfile.Profile()

    under_nsys = _is_under_nsys()
    cuda_region = _cuda_profiler_region() if under_nsys else _null_context()
    prof.enable()
    try:
        with cuda_region:
            yield
        # cudaProfilerStop has fired here — "Capture range ended in the
        # application" is already on stderr before any of our rendering.
    finally:
        prof.disable()
        _finalize_cprofile(prof, top_n, out)


@contextmanager
def _null_context() -> Iterator[None]:
    yield


def _finalize_cprofile(prof: cProfile.Profile, top_n: int, out: TextIO) -> None:
    """Dump (under nsys) or render the cProfile output.

    Under nsys, the parent reads the dump after ``nsys profile`` finishes
    writing the ``.nsys-rep`` and renders it from there. Outside nsys, the
    table is printed inline. On dump-failure we fall back to inline
    rendering — better noisy output than no profile at all.
    """
    dump_path = os.environ.get(_CPROFILE_DUMP_PATH_ENV_VAR)
    if dump_path:
        try:
            prof.dump_stats(dump_path)
            return
        except OSError as e:
            logger.warning(
                "Failed to dump cProfile stats to %s: %s; rendering inline",
                dump_path,
                e,
            )
    _render_cprofile(prof, top_n, out)


class OneShotCapture:
    """Imperative one-shot profile capture: start, then end_and_finalize.

    Designed for when the capture-start and capture-end points do not share
    a single lexical scope. The CLI ``max generate --profile`` flow uses
    this to start the capture *after* model load/warmup (inside the
    ``TextGenerationMetrics`` context) and to end it *after* the metrics
    context exits, so the metrics report prints to the terminal before
    nsys begins emitting its ``.nsys-rep`` file-writing messages.

    Lifecycle:

    - :meth:`start` calls ``cudaProfilerStart`` (under nsys) and
      enables ``cProfile``. Safe to call multiple times; subsequent calls
      are no-ops.
    - :meth:`end_and_finalize` calls ``cudaProfilerStop``, disables
      ``cProfile``, and either dumps the cProfile data to disk (under
      nsys, for the parent to render) or renders it inline. Idempotent;
      callers must invoke it (typically from a ``finally`` block).
    """

    def __init__(
        self,
        *,
        top_n: int = 15,
        out: TextIO | None = None,
    ) -> None:
        self._top_n = top_n
        self._out = out or sys.stdout
        self._cprofile = cProfile.Profile()
        self._under_nsys = _is_under_nsys()
        self._cuda_cm: _GeneratorContextManager[None] | None = None
        self._started = False
        self._ended = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._under_nsys:
            self._cuda_cm = _cuda_profiler_region()
            # __enter__ runs cudaProfilerStart.
            self._cuda_cm.__enter__()
        self._cprofile.enable()

    def end_and_finalize(self) -> None:
        if self._ended or not self._started:
            self._ended = True
            return
        self._ended = True
        self._cprofile.disable()
        if self._cuda_cm is not None:
            # __exit__ runs cudaProfilerStop. nsys begins writing the
            # ``.nsys-rep`` immediately after this returns, so anything
            # the caller wanted to print *first* must already be flushed.
            try:
                self._cuda_cm.__exit__(None, None, None)
            except Exception:
                logger.exception("cudaProfilerStop failed")
            self._cuda_cm = None
        _finalize_cprofile(self._cprofile, self._top_n, self._out)
