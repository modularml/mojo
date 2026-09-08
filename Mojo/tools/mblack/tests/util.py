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

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   tests/util.py
#
# ===----------------------------------------------------------------------=== #

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import mblack
from mblack.debug import DebugVisitor
from mblack.mode import TargetVersion
from mblack.output import diff, err, out

PYTHON_SUFFIX = ".mojo"
ALLOWED_SUFFIXES = (PYTHON_SUFFIX, ".pyi", ".out", ".diff")

THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR / "data"
PROJECT_ROOT = THIS_DIR.parent
EMPTY_LINE = "# EMPTY LINE WITH WHITESPACE" + " (this comment will be removed)"
DETERMINISTIC_HEADER = "[Deterministic header]"

PY36_VERSIONS = {
    TargetVersion.PY36,
    TargetVersion.PY37,
    TargetVersion.PY38,
    TargetVersion.PY39,
}

DEFAULT_MODE = mblack.Mode()
ff = partial(mblack.format_file_in_place, mode=DEFAULT_MODE, fast=True)
fs = partial(mblack.format_str, mode=DEFAULT_MODE)


def _assert_format_equal(expected: str, actual: str) -> None:
    if actual != expected and not os.environ.get("SKIP_AST_PRINT"):
        bdv: DebugVisitor[Any]
        out("Expected tree:", fg="green")
        try:
            exp_node = mblack.lib2to3_parse(expected)
            bdv = DebugVisitor()
            list(bdv.visit(exp_node))
        except Exception as ve:
            err(str(ve))
        out("Actual tree:", fg="red")
        try:
            exp_node = mblack.lib2to3_parse(actual)
            bdv = DebugVisitor()
            list(bdv.visit(exp_node))
        except Exception as ve:
            err(str(ve))

    if actual != expected:
        out(diff(expected, actual, "expected", "actual"))

    assert actual == expected


# The mode users get when running `mojo format`:
#   - `mojo format` hardcodes --preview -t mojo (see mojo-format.cpp)
#   - bazel //:format and //:lint pick up preview=true from pyproject.toml
MOJO_MODE = mblack.Mode(
    target_versions={mblack.TargetVersion.MOJO},
    preview=True,
    is_mojo=True,
)

# The mode users get when running `mblack -t mojo` without --preview.
MOJO_MODE_NO_PREVIEW = mblack.Mode(
    target_versions={mblack.TargetVersion.MOJO},
    is_mojo=True,
)


def mojo_format_str(source: str, *, mode: mblack.Mode = MOJO_MODE) -> str:
    """Format Mojo source and return the result."""
    return mblack.format_str(source, mode=mode)


# Enabled by passing `--validate-with-mojo-build` to pytest (see conftest.py).
# When True, `assert_mojo_format` also runs `mojo build` on each sample to
# verify it's valid Mojo. Off by default: it's slow (a `mojo build` per sample)
# and many samples in this suite don't compile in isolation.
VALIDATE_WITH_MOJO_BUILD = False


def _require_mojo_binary() -> str:
    """Returns a path to the ``mojo`` binary, or raises with a diagnostic.

    Prefers ``MODULAR_MOJO_MAX_DRIVER_PATH`` when set (the surrounding
    test environment, e.g. bazel, may point it at a pre-built binary).
    Otherwise falls back to ``PATH``. Relative env-var values resolve
    against cwd via ``os.path.abspath``.

    Raises:
        AssertionError: If no usable ``mojo`` binary can be found.
    """
    env_path = os.environ.get("MODULAR_MOJO_MAX_DRIVER_PATH", "")
    if env_path:
        resolved = env_path if os.path.isabs(env_path) else os.path.abspath(env_path)
        if os.access(resolved, os.X_OK):
            return resolved
        detail = "does not exist" if not os.path.exists(resolved) else "is not executable"
        raise AssertionError(
            f"MODULAR_MOJO_MAX_DRIVER_PATH is set to {env_path!r} but "
            f"the resolved path {resolved!r} {detail}."
        )
    which = shutil.which("mojo")
    if which is None:
        raise AssertionError(
            "--validate-with-mojo-build needs `mojo` available: set "
            "MODULAR_MOJO_MAX_DRIVER_PATH or put `mojo` on PATH."
        )
    return which


def _truncate_for_error(text: str, *, head: int = 40, tail: int = 40) -> str:
    """Truncates ``text`` to at most ``head + tail`` lines, with an elision marker."""
    lines = text.splitlines(keepends=True)
    if len(lines) <= head + tail:
        return text
    omitted = len(lines) - head - tail
    return "".join(
        lines[:head] + [f"... <{omitted} lines omitted> ...\n"] + lines[-tail:]
    )


def _assert_mojo_compiles(source: str, *, label: str) -> None:
    """Compiles ``source`` with ``mojo build`` and fails if it's rejected.

    A trivial ``def main(): pass`` is appended if missing.

    Args:
        source: The Mojo source to compile.
        label: A short tag (e.g. ``"source"`` or ``"expected"``) used in
            error messages to identify which sample was rejected.

    Raises:
        AssertionError: If no ``mojo`` binary is available, or if
            ``mojo build`` rejects or times out on the sample.
    """
    mojo = _require_mojo_binary()
    build_source = source
    if "def main(" not in build_source:
        build_source = build_source.rstrip("\n") + "\n\n\ndef main():\n    pass\n"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "sample.mojo"
        src.write_text(build_source)
        try:
            result = subprocess.run(
                [mojo, "build", str(src), "-o", str(Path(tmp) / "sample")],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as e:
            raise AssertionError(
                f"`mojo build` timed out on the {label} sample after "
                f"{e.timeout}s:\n"
                f"--- {label} ---\n{_truncate_for_error(build_source)}"
            ) from e
        if result.returncode != 0:
            # Show `build_source` (what mojo actually saw) so the stderr
            # line numbers line up with what the reader is looking at.
            parts = [
                f"`mojo build` rejected the {label} sample:",
                f"--- {label} ---",
                _truncate_for_error(build_source),
            ]
            if result.stdout:
                parts += ["--- stdout ---", result.stdout]
            parts += ["--- stderr ---", result.stderr]
            raise AssertionError("\n".join(parts))


def assert_mojo_format(
    source: str,
    expected: str,
    *,
    mode: mblack.Mode = MOJO_MODE,
    fast: bool = False,
) -> None:
    """Convenience function to check that mblack formats mojo code as expected.

    When the source is reformatted, also verify that re-formatting the output
    is a no-op (idempotency).

    When the pytest flag `--validate-with-mojo-build` is set, also compile
    `source` with `mojo build` to verify it is valid Mojo.
    """
    if VALIDATE_WITH_MOJO_BUILD:
        # Only compile source to halve compile time. This ensures that the test
        # inputs are valid Mojo, with a small risk that the outputs are not.
        _assert_mojo_compiles(source, label="source")
    actual = mojo_format_str(source, mode=mode)
    _assert_format_equal(expected, actual)
    if not fast and source != expected:
        mblack.assert_stable(source, actual, mode=mode)


def assert_format(
    source: str,
    expected: str,
    mode: mblack.Mode = DEFAULT_MODE,
    *,
    fast: bool = False,
    minimum_version: Optional[Tuple[int, int]] = None,
) -> None:
    """Convenience function to check that Black formats as expected.

    You can pass @minimum_version if you're passing code with newer syntax to guard
    safety guards so they don't just crash with a SyntaxError. Please note this is
    separate from TargetVerson Mode configuration.
    """
    actual = mblack.format_str(source, mode=mode)
    _assert_format_equal(expected, actual)
    # It's not useful to run safety checks if we're expecting no changes anyway. The
    # assertion right above will raise if reality does actually make changes. This just
    # avoids wasted CPU cycles.
    if not fast and source != expected:
        # Unfortunately the AST equivalence check relies on the built-in ast module
        # being able to parse the code being formatted. This doesn't always work out
        # when checking modern code on older versions.
        if minimum_version is None or sys.version_info >= minimum_version:
            mblack.assert_equivalent(source, actual)
        mblack.assert_stable(source, actual, mode=mode)


def dump_to_stderr(*output: str) -> str:
    return "\n" + "\n".join(output) + "\n"


class BlackBaseTestCase(unittest.TestCase):
    def assertFormatEqual(self, expected: str, actual: str) -> None:
        _assert_format_equal(expected, actual)


def get_base_dir(data: bool) -> Path:
    return DATA_DIR if data else PROJECT_ROOT


def all_data_cases(subdir_name: str, data: bool = True) -> List[str]:
    cases_dir = get_base_dir(data) / subdir_name
    assert cases_dir.is_dir()
    return [case_path.stem for case_path in cases_dir.iterdir()]


def get_case_path(
    subdir_name: str, name: str, data: bool = True, suffix: str = PYTHON_SUFFIX
) -> Path:
    """Get case path from name"""
    case_path = get_base_dir(data) / subdir_name / name
    if not name.endswith(ALLOWED_SUFFIXES):
        case_path = case_path.with_suffix(suffix)
    assert case_path.is_file(), f"{case_path} is not a file."
    return case_path


def read_data(subdir_name: str, name: str, data: bool = True) -> Tuple[str, str]:
    """read_data('test_name') -> 'input', 'output'"""
    return read_data_from_file(get_case_path(subdir_name, name, data))


def read_data_from_file(file_name: Path) -> Tuple[str, str]:
    with open(file_name, "r", encoding="utf8") as test:
        lines = test.readlines()

    # We added a proprietary header to our files, so we need to skip it.
    if lines[1].startswith("# Copyright (c)"):
        lines = lines[21:]

    _input: List[str] = []
    _output: List[str] = []
    result = _input
    for line in lines:
        line = line.replace(EMPTY_LINE, "")
        if line.rstrip() == "# output":
            result = _output
            continue

        result.append(line)
    if _input and not _output:
        # If there's no output marker, treat the entire file as already pre-formatted.
        _output = _input[:]
    return "".join(_input).strip() + "\n", "".join(_output).strip() + "\n"


@contextmanager
def change_directory(path: Path) -> Iterator[None]:
    """Context manager to temporarily chdir to a different directory."""
    previous_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous_dir)
