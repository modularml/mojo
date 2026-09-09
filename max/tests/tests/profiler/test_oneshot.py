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

"""Tests for the rudimentary one-shot profiler."""

from __future__ import annotations

import cProfile
import io
import os
import pathlib
import unittest.mock

import pytest
from max.profiler.oneshot import (
    OneShotCapture,
    ProfileBackend,
    inject_trace_flags,
    maybe_reexec_under_nsys,
    profiled_region,
)
from max.profiler.oneshot._cprofile import _strip_bazel_path
from max.profiler.oneshot._nsys import _parse_nsys_csv

# A trimmed but realistic ``nsys stats --report cuda_gpu_kern_sum --format csv``
# excerpt. Real output is preceded by section headers; the parser is meant to
# skip those, which we verify by including a stray header line.
GOLDEN_NSYS_CSV = """\
** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
54.3,108600000,256,424218.7,420000.0,400000.0,500000.0,12000.0,matmul_gpu_kernel<bfloat16>
22.8,45600000,128,356250.0,350000.0,340000.0,400000.0,8000.0,flash_attention_kernel
12.4,24800000,512,48437.5,48000.0,45000.0,55000.0,1500.0,layernorm_kernel
10.5,21000000,256,82031.2,80000.0,75000.0,100000.0,5000.0,rope_kernel
"""


def test_parse_nsys_csv_golden() -> None:
    rows = _parse_nsys_csv(GOLDEN_NSYS_CSV)
    assert len(rows) == 4
    assert rows[0]["name"] == "matmul_gpu_kernel<bfloat16>"
    assert rows[0]["total_ns"] == 108_600_000
    assert rows[0]["instances"] == 256
    # Sum should be the total across all kernels.
    assert sum(r["total_ns"] for r in rows) == 200_000_000


def test_parse_nsys_csv_empty() -> None:
    assert _parse_nsys_csv("") == []
    assert _parse_nsys_csv("** Section header only\n") == []


def test_parse_nsys_csv_with_commas_in_numbers() -> None:
    csv_with_thousands = """\
Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
100.0,"1,234,567","42",29395.6,29000.0,28000.0,32000.0,1000.0,kernel_with_thousands
"""
    rows = _parse_nsys_csv(csv_with_thousands)
    assert len(rows) == 1
    assert rows[0]["total_ns"] == 1_234_567
    assert rows[0]["instances"] == 42


def test_profiled_region_runs_cprofile_when_not_under_nsys() -> None:
    """Outside the nsys child, profiled_region still emits a cProfile table."""
    backend = ProfileBackend(
        nsys_available=False, nvidia_gpu_present=False, cudart_loadable=False
    )
    out = io.StringIO()
    called = []
    with profiled_region(backend=backend, out=out):
        called.append(True)
        # Some real work so cProfile has something to report.
        sum(range(1000))
    assert called == [True]
    assert "Python/CPU functions" in out.getvalue()


def test_profiled_region_dumps_cprofile_when_env_var_set(
    tmp_path: pathlib.Path,
) -> None:
    """When the dump-path env var is set, cProfile data is written to disk
    instead of being rendered inline. This is how the nsys child defers
    cProfile rendering to the parent."""
    backend = ProfileBackend(
        nsys_available=False, nvidia_gpu_present=False, cudart_loadable=False
    )
    dump_path = str(tmp_path / "out.cprofile")
    out = io.StringIO()
    with unittest.mock.patch.dict(
        os.environ,
        {"_MAX_PROFILE_CPROFILE_DUMP_PATH": dump_path},
        clear=False,
    ):
        with profiled_region(backend=backend, out=out):
            sum(range(1000))
    # No inline rendering.
    assert "Python/CPU functions" not in out.getvalue()
    # The dump file exists and is loadable via pstats / marshal.
    assert os.path.exists(dump_path)
    import marshal

    with open(dump_path, "rb") as f:
        stats = marshal.load(f)
    assert isinstance(stats, dict)
    assert stats  # at least one entry


def test_render_cprofile_from_dump_missing_file(
    tmp_path: pathlib.Path,
) -> None:
    """Missing dump files are reported as not-rendered, not exceptions."""
    from max.profiler.oneshot._cprofile import _render_cprofile_from_dump

    out = io.StringIO()
    rendered = _render_cprofile_from_dump(
        str(tmp_path / "nope.cprofile"), top_n=5, out=out
    )
    assert rendered is False
    assert out.getvalue() == ""


def test_render_cprofile_from_dump_roundtrips_a_real_profile(
    tmp_path: pathlib.Path,
) -> None:
    """A round-trip from prof.dump_stats → _render_cprofile_from_dump
    produces the same Python/CPU table format as the inline renderer."""
    from max.profiler.oneshot._cprofile import _render_cprofile_from_dump

    prof = cProfile.Profile()
    prof.enable()
    for _ in range(5):
        sum(range(100))
    prof.disable()
    dump_path = str(tmp_path / "out.cprofile")
    prof.dump_stats(dump_path)

    out = io.StringIO()
    rendered = _render_cprofile_from_dump(dump_path, top_n=5, out=out)
    assert rendered is True
    text = out.getvalue()
    assert "Python/CPU functions" in text
    assert "function" in text


def test_maybe_reexec_explains_no_nsys() -> None:
    """When nsys is unavailable, maybe_reexec_under_nsys prints the hint."""
    backend = ProfileBackend(
        nsys_available=False, nvidia_gpu_present=False, cudart_loadable=True
    )
    out = io.StringIO()
    maybe_reexec_under_nsys(backend=backend, out=out)
    assert "Nsight Systems" in out.getvalue()


def test_maybe_reexec_explains_no_gpu() -> None:
    """nsys installed but no NVIDIA GPU prints the no-GPU hint."""
    backend = ProfileBackend(
        nsys_available=True, nvidia_gpu_present=False, cudart_loadable=True
    )
    out = io.StringIO()
    maybe_reexec_under_nsys(backend=backend, out=out)
    assert "no NVIDIA GPU detected" in out.getvalue()


def test_maybe_reexec_explains_no_libcudart() -> None:
    """nsys + GPU present but libcudart missing prints the libcudart hint."""
    backend = ProfileBackend(
        nsys_available=True, nvidia_gpu_present=True, cudart_loadable=False
    )
    out = io.StringIO()
    maybe_reexec_under_nsys(backend=backend, out=out)
    assert "libcudart could not be loaded" in out.getvalue()


def test_inject_trace_flags_empty_args() -> None:
    out = inject_trace_flags([], "/tmp/p.nsys-rep")
    assert out == ["--trace", "--trace-file", "/tmp/p.nsys-rep"]


def test_inject_trace_flags_passthrough_args() -> None:
    out = inject_trace_flags(["--model", "m"], "/tmp/p.nsys-rep")
    assert out == ["--model", "m", "--trace", "--trace-file", "/tmp/p.nsys-rep"]


def test_inject_trace_flags_dedups_existing_trace() -> None:
    """User-supplied ``--trace`` is not duplicated."""
    out = inject_trace_flags(["--model", "m", "--trace"], "/tmp/p.nsys-rep")
    assert out.count("--trace") == 1
    assert out[-2:] == ["--trace-file", "/tmp/p.nsys-rep"]


def test_inject_trace_flags_strips_user_trace_file_space() -> None:
    """``--trace-file PATH`` (space form) is stripped along with its value."""
    out = inject_trace_flags(
        [
            "--model",
            "m",
            "--trace-file",
            "/tmp/old.nsys-rep",
            "--num-prompts",
            "8",
        ],
        "/tmp/new.nsys-rep",
    )
    # Old path is gone; new path is at the end.
    assert "/tmp/old.nsys-rep" not in out
    assert out[-2:] == ["--trace-file", "/tmp/new.nsys-rep"]
    # The unrelated `--num-prompts 8` survives in order.
    assert "--num-prompts" in out
    assert "8" in out
    assert out.index("--num-prompts") < out.index("8")


def test_inject_trace_flags_strips_user_trace_file_equals() -> None:
    """``--trace-file=PATH`` (equals form) is stripped."""
    out = inject_trace_flags(
        ["--trace-file=/tmp/old.nsys-rep"], "/tmp/new.nsys-rep"
    )
    assert "--trace-file=/tmp/old.nsys-rep" not in out
    assert out == ["--trace", "--trace-file", "/tmp/new.nsys-rep"]


def test_maybe_reexec_no_op_when_already_under_nsys() -> None:
    """Inside the nsys child, the helper must do nothing and return."""
    backend = ProfileBackend(
        nsys_available=True, nvidia_gpu_present=True, cudart_loadable=True
    )
    with unittest.mock.patch.dict(
        os.environ,
        {"_MAX_PROFILE_UNDER_NSYS": "1"},
        clear=False,
    ):
        # Returns normally — does not call _reexec_under_nsys, does not exit.
        maybe_reexec_under_nsys(backend=backend)


def test_maybe_reexec_no_op_when_no_gpu_capture() -> None:
    """No NVIDIA GPU or no nsys → return without re-exec or exit."""
    backend = ProfileBackend(
        nsys_available=False, nvidia_gpu_present=False, cudart_loadable=False
    )
    maybe_reexec_under_nsys(backend=backend)


def test_maybe_reexec_runs_child_and_exits() -> None:
    """When GPU-capable, we spawn nsys, render summary, and sys.exit."""
    backend = ProfileBackend(
        nsys_available=True, nvidia_gpu_present=True, cudart_loadable=True
    )
    out = io.StringIO()

    with (
        unittest.mock.patch(
            "max.profiler.oneshot._runner._reexec_under_nsys", return_value=0
        ) as m_reexec,
        unittest.mock.patch(
            "max.profiler.oneshot._runner.os.path.exists", return_value=False
        ),
        pytest.raises(SystemExit) as ei,
    ):
        maybe_reexec_under_nsys("/tmp/test.nsys-rep", backend=backend, out=out)
    assert ei.value.code == 0
    # The parent forwards the cProfile dump path so the child writes its
    # cProfile stats there instead of rendering them inline.
    m_reexec.assert_called_once_with(
        "/tmp/test.nsys-rep",
        cprofile_dump_path="/tmp/test.nsys-rep.cprofile",
    )
    # When nsys didn't write the .nsys-rep, the user is told.
    assert "without writing" in out.getvalue()


def test_maybe_reexec_renders_cprofile_dump_before_nsys_summary(
    tmp_path: pathlib.Path,
) -> None:
    """The parent renders the child's cProfile dump before the nsys table.

    The cProfile output should follow the nsys "Generating .nsys-rep" lines
    (which the child emits at exit) and precede the GPU kernel summary, so
    the --profile output sits cleanly after the normal generate output.
    """
    backend = ProfileBackend(
        nsys_available=True, nvidia_gpu_present=True, cudart_loadable=True
    )
    nsys_rep = str(tmp_path / "p.nsys-rep")
    cprofile_dump = nsys_rep + ".cprofile"

    # Pre-seed a real cProfile dump so the parent has something to render.
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(5):
        sum(range(100))
    prof.disable()
    prof.dump_stats(cprofile_dump)

    nsys_csv = (
        "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),"
        "Max (ns),StdDev (ns),Name\n"
        "100.0,1000000,4,250000.0,250000.0,250000.0,250000.0,0.0,my_kernel\n"
    )

    class _NsysStatsResult:
        returncode = 0
        stdout = nsys_csv

    out = io.StringIO()
    with (
        unittest.mock.patch(
            "max.profiler.oneshot._runner._reexec_under_nsys", return_value=0
        ),
        # Pretend the .nsys-rep exists so the kernel summary renders.
        unittest.mock.patch(
            "max.profiler.oneshot._runner.os.path.exists", return_value=True
        ),
        unittest.mock.patch(
            "max.profiler.oneshot._nsys.subprocess.run",
            return_value=_NsysStatsResult(),
        ),
        pytest.raises(SystemExit),
    ):
        maybe_reexec_under_nsys(nsys_rep, backend=backend, out=out)
    text = out.getvalue()
    assert "Python/CPU functions" in text
    assert "Top 1 GPU kernels" in text
    # cProfile table must precede the GPU kernel table.
    assert text.index("Python/CPU functions") < text.index("Top 1 GPU kernels")
    # The parent cleans up the dump file after consuming it.
    assert not os.path.exists(cprofile_dump)


def test_reexec_env_defaults_modular_enable_profiling() -> None:
    """The nsys child env defaults to MODULAR_ENABLE_PROFILING=detailed so
    Tracer/@traced spans actually emit NVTX ranges."""
    from max.profiler.oneshot._nsys import _reexec_under_nsys

    captured: dict[str, dict[str, str]] = {}

    class _Result:
        returncode = 0

    def fake_run(cmd: list[str], env: dict[str, str], check: bool) -> _Result:
        captured["env"] = env
        return _Result()

    with (
        unittest.mock.patch(
            "max.profiler.oneshot._nsys.subprocess.run", side_effect=fake_run
        ),
        unittest.mock.patch.dict(
            os.environ,
            {},  # start without MODULAR_ENABLE_PROFILING set
            clear=True,
        ),
    ):
        rc = _reexec_under_nsys("/tmp/p.nsys-rep")
    assert rc == 0
    assert captured["env"]["MODULAR_ENABLE_PROFILING"] == "detailed"
    assert captured["env"]["_MAX_PROFILE_UNDER_NSYS"] == "1"


def test_reexec_env_respects_user_modular_enable_profiling() -> None:
    """If the user already exported MODULAR_ENABLE_PROFILING, we keep it."""
    from max.profiler.oneshot._nsys import _reexec_under_nsys

    captured: dict[str, dict[str, str]] = {}

    class _Result:
        returncode = 0

    def fake_run(cmd: list[str], env: dict[str, str], check: bool) -> _Result:
        captured["env"] = env
        return _Result()

    with (
        unittest.mock.patch(
            "max.profiler.oneshot._nsys.subprocess.run", side_effect=fake_run
        ),
        unittest.mock.patch.dict(
            os.environ,
            {"MODULAR_ENABLE_PROFILING": "on"},
            clear=True,
        ),
    ):
        _reexec_under_nsys("/tmp/p.nsys-rep")
    assert captured["env"]["MODULAR_ENABLE_PROFILING"] == "on"


def test_strip_bazel_path_runfiles() -> None:
    assert (
        _strip_bazel_path(
            "/home/u/.cache/bazel/_bazel_u/abc/execroot/_main/bazel-out/"
            "k8-dbg/bin/max/python/max/_entrypoints/pipelines.runfiles/_main/"
            "max/python/max/_entrypoints/cli/generate.py"
        )
        == "max/python/max/_entrypoints/cli/generate.py"
    )


def test_strip_bazel_path_python_stdlib() -> None:
    assert (
        _strip_bazel_path(
            "/home/u/.cache/bazel/_bazel_u/abc/external/"
            "rules_python++python+python_3_13_x86_64-unknown-linux-gnu/"
            "lib/python3.13/asyncio/base_events.py"
        )
        == "asyncio/base_events.py"
    )


def test_strip_bazel_path_passthrough() -> None:
    # Plain path with no bazel markers is returned unchanged.
    assert _strip_bazel_path("/usr/lib/foo.py") == "/usr/lib/foo.py"


def test_render_nsys_kernel_summary_returns_false_on_empty() -> None:
    """An empty CSV from nsys stats yields no table and the function reports
    nothing was rendered."""
    from max.profiler.oneshot import render_nsys_kernel_summary

    class _Result:
        returncode = 0
        stdout = ""

    out = io.StringIO()
    with unittest.mock.patch(
        "max.profiler.oneshot._nsys.subprocess.run", return_value=_Result()
    ):
        rendered = render_nsys_kernel_summary("/tmp/p.nsys-rep", 10, out)
    assert rendered is False
    assert out.getvalue() == ""


def test_render_nsys_kernel_summary_renders_single_row() -> None:
    """A one-row CSV produces a header line and a 100%-of-total entry."""
    from max.profiler.oneshot import render_nsys_kernel_summary

    csv = (
        "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),"
        "Max (ns),StdDev (ns),Name\n"
        "100.0,1000000,4,250000.0,250000.0,250000.0,250000.0,0.0,my_kernel\n"
    )

    class _Result:
        returncode = 0
        stdout = csv

    out = io.StringIO()
    with unittest.mock.patch(
        "max.profiler.oneshot._nsys.subprocess.run", return_value=_Result()
    ):
        rendered = render_nsys_kernel_summary("/tmp/p.nsys-rep", 10, out)
    text = out.getvalue()
    assert rendered is True
    # Header line and single-row entry.
    assert "Top 1 GPU kernels" in text
    assert "100.0%" in text
    assert "my_kernel" in text


def test_render_cprofile_handles_no_samples() -> None:
    """A profile with no recorded calls prints the "no samples" line."""
    from max.profiler.oneshot._cprofile import _render_cprofile

    prof = cProfile.Profile()
    out = io.StringIO()
    _render_cprofile(prof, 10, out)
    assert "no samples captured" in out.getvalue()


def test_render_cprofile_renders_top_n() -> None:
    """A profile with real work renders a Python/CPU table."""
    from max.profiler.oneshot._cprofile import _render_cprofile

    prof = cProfile.Profile()
    prof.enable()
    # Run something that cProfile will record.
    for _ in range(5):
        sum(range(100))
    prof.disable()

    out = io.StringIO()
    _render_cprofile(prof, 5, out)
    text = out.getvalue()
    assert "Python/CPU functions" in text
    # Header row of the table.
    assert "function" in text


def test_oneshot_capture_renders_cprofile_when_not_under_nsys() -> None:
    """Outside the nsys child, ``end_and_finalize`` prints a cProfile table.

    This is the inline-render path used when ``nsys`` isn't available; the
    nsys-child path (env var set → dump to file) is exercised by
    ``test_oneshot_capture_dumps_when_env_var_set``.
    """
    out = io.StringIO()
    capture = OneShotCapture(top_n=5, out=out)
    capture.start()
    sum(range(1000))
    capture.end_and_finalize()
    assert "Python/CPU functions" in out.getvalue()


def test_oneshot_capture_end_is_idempotent() -> None:
    """Calling ``end_and_finalize`` twice prints exactly one table."""
    out = io.StringIO()
    capture = OneShotCapture(top_n=5, out=out)
    capture.start()
    sum(range(100))
    capture.end_and_finalize()
    capture.end_and_finalize()
    # Header appears exactly once.
    assert out.getvalue().count("Python/CPU functions") == 1


def test_oneshot_capture_end_without_start_is_a_no_op() -> None:
    """``end_and_finalize`` without a prior ``start`` does nothing.

    Lets the generate flow safely call ``end_and_finalize`` from a
    ``finally`` even when ``start`` never ran (e.g. an exception during
    model load before the timed region).
    """
    out = io.StringIO()
    capture = OneShotCapture(top_n=5, out=out)
    capture.end_and_finalize()
    assert out.getvalue() == ""


def test_oneshot_capture_dumps_when_env_var_set(
    tmp_path: pathlib.Path,
) -> None:
    """Under the nsys child, ``OneShotCapture`` writes the cProfile dump
    instead of rendering inline."""
    out = io.StringIO()
    dump_path = str(tmp_path / "out.cprofile")
    with unittest.mock.patch.dict(
        os.environ,
        {"_MAX_PROFILE_CPROFILE_DUMP_PATH": dump_path},
        clear=False,
    ):
        capture = OneShotCapture(top_n=5, out=out)
        capture.start()
        sum(range(1000))
        capture.end_and_finalize()
    assert "Python/CPU functions" not in out.getvalue()
    assert os.path.exists(dump_path)


def test_profiled_region_propagates_user_exceptions() -> None:
    """An exception raised inside the region is not swallowed; the cProfile
    summary still prints because we hit the finally branch."""
    backend = ProfileBackend(
        nsys_available=False, nvidia_gpu_present=False, cudart_loadable=False
    )
    out = io.StringIO()
    with pytest.raises(RuntimeError, match="boom"):
        with profiled_region(backend=backend, out=out):
            raise RuntimeError("boom")
    assert "Python/CPU functions" in out.getvalue()
