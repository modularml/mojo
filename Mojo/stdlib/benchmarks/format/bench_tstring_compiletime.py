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
"""Compile-time stress benchmark for `TString`.

Generates a Mojo file containing many `print(t"...")` statements, each with a
distinct format string. Every distinct format string forces its own
compile-time specialization of `TString[format_string, *Ts]`, and its own run
of the comptime format-string encoder inside `write_to()`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_PATTERNS = (
    'print(t"stress literal {tag} with no fields")',
    'print(t"stress literal {tag} value={{a}}")',
    'print(t"stress {tag} a={{a}} c={{c}}")',
    'print(t"stress {tag} a={{a}} c={{c}} e={{e}}")',
)


def generate_source(count: int) -> str:
    """Generates Mojo source with `count` distinct t-string print statements.

    Args:
        count: How many distinct t-string literals to emit.

    Returns:
        The full text of a `main()`-only Mojo file.
    """
    lines = [
        '    var a = "abcdef"',
        "    var c = [1, 2, 3, 4, 5, 6]",
        "    var e = 4567890",
        "",
    ]
    for i in range(count):
        tag = f"{i:05d}"
        lines.append("    " + _PATTERNS[i % len(_PATTERNS)].format(tag=tag))
    body = "\n".join(lines)
    return f"def main():\n{body}\n"


def _find_mojo() -> str:
    mojo = shutil.which("mojo")
    if mojo is None:
        sys.exit("error: `mojo` not found on PATH")
    return mojo


def run_timed_build(
    mojo_bin: str, source_path: Path, object_path: Path, log_path: Path
) -> float:
    """Compiles `source_path` with timing flags and writes the report to `log_path`.

    Uses a fresh `MODULAR_CACHE_DIR` per call: a warm cache serves work from
    cache instead of recompiling it, which silently drops it out of the
    timing report.

    Args:
        mojo_bin: Path to the `mojo` binary.
        source_path: The `.mojo` file to compile.
        object_path: Where to write the compiled object.
        log_path: Where to write the raw `--mlir-timing`/`--llvm-timing` report.

    Returns:
        Wall-clock seconds the build took.
    """
    with tempfile.TemporaryDirectory(prefix="tstring-cache-") as cache_dir:
        env = os.environ.copy()
        env["MODULAR_CACHE_DIR"] = cache_dir
        start = time.monotonic()
        result = subprocess.run(
            [
                mojo_bin,
                "build",
                "--emit=object",
                "--mlir-timing",
                "--llvm-timing",
                str(source_path),
                "-o",
                str(object_path),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        elapsed = time.monotonic() - start

    log_path.write_text(result.stderr)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(f"mojo build failed with exit code {result.returncode}")
    return elapsed


def report_binary_size(object_path: Path) -> None:
    """Prints a section-size breakdown of `object_path` via llvm-size/size."""
    tool = shutil.which("llvm-size") or shutil.which("size")
    if tool is None:
        print(
            "(skipped binary-size report: neither llvm-size nor size found on"
            " PATH)"
        )
        return
    # The tool writes straight to the shared stdout fd instead of going
    # through Python's buffer, so drain ours first to keep the order.
    sys.stdout.flush()
    subprocess.run([tool, str(object_path)], check=False)
    print(f"object file size on disk: {object_path.stat().st_size:,} bytes")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="number of distinct t-string literals to generate (default: 1000)",
    )
    args = parser.parse_args()

    mojo_bin = _find_mojo()

    scratch = Path(tempfile.mkdtemp(prefix="bench-tstring-compiletime-"))
    try:
        source_path = scratch / "generated.mojo"
        source_path.write_text(generate_source(args.count))

        object_path = scratch / "generated.o"
        log_fd, log_name = tempfile.mkstemp(
            prefix="bench-tstring-compiletime-", suffix=".log"
        )
        os.close(log_fd)
        log_path = Path(log_name)

        print(f"compiling {args.count} distinct t-string literals...")
        elapsed = run_timed_build(mojo_bin, source_path, object_path, log_path)
        print(f"wall clock: {elapsed:.2f}s")
        print(f"timing log: {log_path}")

        report_binary_size(object_path)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
