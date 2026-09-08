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
"""What a `debug_assert` or an `abort` costs to compile, by message shape.

Generates a Mojo file with `--count` call sites of one shape, compiles it, and
reports wall clock and object size. Nothing runs; no GPU needed.
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
from typing import NamedTuple, TextIO

# The sites are plain `debug_assert(...)`, whose own `assert_mode` defaults to
# "none", so only `all` turns them on. Any other level reports a table of
# zeroes that looks like a measurement.
_ASSERT_MODE = "all"

# Builds per cell. The fastest is the estimator; three is enough to show the
# spread without tripling an already slow sweep.
_REPS = 3

_TARGETS = ("host", "sm_90a", "sm_100a", "gfx950", "apple-m4")

# Sum of the report column widths in `main`.
_TABLE_WIDTH = 55

_SITES = {
    "none": "    if idx + {i} >= n:\n        dst[unsafe_offset=1] = Byte({i})",
    "bare": "    debug_assert(idx + {i} < n)",
    "literal": '    debug_assert(idx + {i} < n, "index {tag} out of bounds")',
    "formatted": (
        '    debug_assert(idx + {i} < n, "oob {tag}: ", idx + {i}, " >= ", n)'
    ),
    "tstring": (
        '    debug_assert(idx + {i} < n, t"oob {tag}: {{idx + {i}}} >= {{n}}")'
    ),
    # The `assert` statement form. It takes at most one message expression
    # (see `parseAssertStmt`), so there is no variadic counterpart to
    # `formatted` here -- a t-string is how a value reaches the message.
    "stmt_bare": "    assert idx + {i} < n",
    "stmt_literal": '    assert idx + {i} < n, "index {tag} out of bounds"',
    "stmt_tstring": (
        '    assert idx + {i} < n, t"oob {tag}: {{idx + {i}}} >= {{n}}"'
    ),
    "abort_literal": (
        '    if idx + {i} >= n:\n        abort("index {tag} out of bounds")'
    ),
    "abort_tstring": (
        '    if idx + {i} >= n:\n        abort(t"oob {tag}: {{idx + {i}}} >='
        ' {{n}}")'
    ),
}


class Cell(NamedTuple):
    """One measured cell. `size` is `None` when the build failed."""

    shape: str
    sites: int
    wall_min: float
    wall_max: float
    size: int | None


def site_counts(arg: str) -> list[int]:
    """Parses `--count` into a list of positive site counts."""
    try:
        counts = [int(part) for part in arg.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated list of integers, got {arg!r}"
        ) from None
    if any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError(
            f"expected positive site counts, got {arg!r}"
        )
    return counts


def generate_source(shape: str, count: int, target: str) -> str:
    """Returns Mojo source with `count` sites of `shape`, built for `target`.

    The sites sit in a function that `compile_info` cross-compiles, and `main`
    prints the assembly that comes back so it survives into the object. Every
    condition mentions its own index, which stops the compiler folding the
    sites into one.
    """
    imports = ["from std.compile import compile_info"]
    if shape.startswith("abort_"):
        imports.append("from std.os import abort")
    if target != "host":
        imports.insert(0, "from std._gpu.host import get_gpu_target")

    sites = "\n".join(
        _SITES[shape].format(i=i, tag=f"{i:03d}") for i in range(count)
    )
    if target == "host":
        target_arg = ""
    else:
        target_arg = f'\n                target = get_gpu_target["{target}"](),'

    import_block = "\n".join(imports)

    return f"""{import_block}


def sites(dst: MutPointer[Byte, MutUntrackedOrigin], n: Int):
    var idx = Int(dst[unsafe_offset=0])
{sites}
    dst[unsafe_offset=0] = Byte(idx)


def main():
    print(
        String(
            compile_info[
                sites,
                emission_kind="asm",{target_arg}
            ]().asm
        )
    )
"""


def build(
    mojo: str, source: Path, obj: Path, log: TextIO
) -> tuple[float, int | None]:
    """Compiles `source`, appending its timing report to `log`.

    Returns the wall seconds and the object size, or `None` for a build that
    failed. The cache directory is fresh every time: a warm cache serves the
    work from cache and reports a compile that never happened.
    """
    with tempfile.TemporaryDirectory(prefix="assert-cache-") as cache:
        env = dict(os.environ, MODULAR_CACHE_DIR=cache)
        start = time.monotonic()
        result = subprocess.run(
            # -D goes before the file; after it, mojo ignores it silently.
            [
                mojo,
                "build",
                "--emit",
                "object",
                "--mlir-timing",
                "-D",
                f"ASSERT={_ASSERT_MODE}",
                str(source),
                "-o",
                str(obj),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        wall = time.monotonic() - start

    log.write(result.stderr)
    if result.returncode != 0:
        first = next(
            (
                line.strip()
                for line in result.stderr.splitlines()
                if "error:" in line
            ),
            "build failed",
        )
        print(f"  {first}", file=sys.stderr)
        return wall, None
    return wall, obj.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--count",
        type=site_counts,
        default=[1, 8, 32],
        help="call sites per kernel",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=_TARGETS + ("all",),
        help="target to compile for",
    )
    args = parser.parse_args()

    mojo = shutil.which("mojo")
    if mojo is None:
        sys.exit("error: `mojo` not found on PATH")

    targets = args.target or ["host"]
    if "all" in targets:
        targets = list(_TARGETS)

    work = Path(tempfile.mkdtemp(prefix="bench_debug_assert_"))
    source, obj = work / "sites.mojo", work / "sites.o"
    log_path = work.parent / f"{work.name}.log"

    rows: dict[str, list[Cell]] = {}
    try:
        with open(log_path, "w") as log:
            for target in targets:
                rows[target] = []
                for shape in _SITES:
                    for count in args.count:
                        print(f"[{target}] {shape} x{count}", file=sys.stderr)
                        source.write_text(generate_source(shape, count, target))
                        log.write(f"\n===== {target} {shape} x{count} =====\n")
                        walls: list[float] = []
                        size: int | None = None
                        for _ in range(_REPS):
                            wall, size = build(mojo, source, obj, log)
                            walls.append(wall)
                            if size is None:
                                break
                        rows[target].append(
                            Cell(shape, count, min(walls), max(walls), size)
                        )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print()
    for target, cells in rows.items():
        print(f"target: {target}")
        print("-" * _TABLE_WIDTH)
        print(
            f"{'shape':<15}{'sites':>6}{'wall min':>11}"
            f"{'wall max':>11}{'obj':>12}"
        )
        for shape, count, low, high, size in cells:
            obj_cell = f"{size:,}" if size is not None else "FAILED"
            print(
                f"{shape:<15}{count:>6}{low:>10.2f}s{high:>10.2f}s{obj_cell:>12}"
            )
        print("-" * _TABLE_WIDTH)
        print()
    print(f"timing reports: {log_path}")
    failed = sum(
        1 for cells in rows.values() for cell in cells if cell.size is None
    )
    if failed:
        print(f"error: {failed} cell(s) failed to build", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
