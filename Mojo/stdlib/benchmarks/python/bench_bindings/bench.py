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
"""Per-call FFI overhead microbenchmark for the Python -> Mojo bindings.

Measures the time CPython spends crossing into Mojo and back for a trivial
no-op and a two-int add, across each binding path exposed by
`mojo_module.mojo`. See README.md and modular/modular#6521 for context and
methodology.

Methodology mirrors the bug report:
- ``timeit.repeat(stmt, number=2_000_000, repeat=7)`` for each variant.
- Report the **minimum** time-per-call across repeats: of the standard
  microbench summary statistics, the min is the closest estimate of true
  per-call cost; means are contaminated by transient scheduler / cache /
  other-process noise.
- Always run baselines (``py_noop``, ``py_add``, ``1 + 2``) alongside so
  drift in the host (CPython upgrades, frequency scaling, etc.) is visible.

For stable numbers, pin the process to a single core::

    taskset -c 2 python3 bench.py
"""

from __future__ import annotations

import os
import sys
import timeit

# `mojo_module.so` is dropped next to this script by Bazel runfiles; make sure
# we can import it whether running under bazel or invoked directly from a
# build output directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mojo_module  # type: ignore[import-not-found]

ITERATIONS = 2_000_000
REPEATS = 7


# Pure-Python baselines: same shape as the Mojo functions so we can read the
# Mojo numbers against a "Python -> Python call" floor.
def py_noop(x):  # noqa: ANN001, ANN201
    return x


def py_add(a, b):  # noqa: ANN001, ANN201
    return a + b


def _measure(label: str, stmt: str, globals_: dict[str, object]) -> None:
    times = timeit.repeat(
        stmt, number=ITERATIONS, repeat=REPEATS, globals=globals_
    )
    min_ns = min(times) / ITERATIONS * 1e9
    max_ns = max(times) / ITERATIONS * 1e9
    print(f"{label:<44} {min_ns:>10.2f} ns / call   ({max_ns:>7.2f} ns max)")


def _sanity_check() -> None:
    """Fail fast if the binding is wired up wrong, so we don't publish noise."""
    assert mojo_module.noop_def(1) == 1
    assert mojo_module.noop_raw(1) == 1
    assert mojo_module.noop_raw_fastcall(1) == 1
    assert mojo_module.add_def(1, 2) == 3
    assert mojo_module.add_raw(1, 2) == 3
    assert mojo_module.add_raw_fastcall(1, 2) == 3


def main() -> int:
    _sanity_check()

    g: dict[str, object] = {
        "noop_def": mojo_module.noop_def,
        "noop_raw": mojo_module.noop_raw,
        "noop_raw_fastcall": mojo_module.noop_raw_fastcall,
        "add_def": mojo_module.add_def,
        "add_raw": mojo_module.add_raw,
        "add_raw_fastcall": mojo_module.add_raw_fastcall,
        "py_noop": py_noop,
        "py_add": py_add,
    }

    print(
        f"# Per-call FFI overhead "
        f"(ITERATIONS={ITERATIONS:,}, REPEATS={REPEATS}, min of repeats)\n"
    )
    print(f"{'Variant':<54} {'min':>10}            {'max':>10}")
    print("-" * 94)

    # High-level `def_function` path: regression target.
    _measure(
        "Python -> Mojo  noop_def(x)             [def_function/FASTCALL]",
        "noop_def(1)",
        g,
    )
    _measure(
        "Python -> Mojo  add_def(1, 2)           [def_function/FASTCALL]",
        "add_def(1, 2)",
        g,
    )

    # Low-level def_py_c_function paths: hand-written wrappers at each
    # calling convention, lower bound for that convention.
    _measure(
        "Python -> Mojo  noop_raw(x)             [def_py_c_function/VARARGS]",
        "noop_raw(1)",
        g,
    )
    _measure(
        "Python -> Mojo  add_raw(1, 2)           [def_py_c_function/VARARGS]",
        "add_raw(1, 2)",
        g,
    )
    _measure(
        "Python -> Mojo  noop_raw_fastcall(x)    [def_py_c_function/FASTCALL]",
        "noop_raw_fastcall(1)",
        g,
    )
    _measure(
        "Python -> Mojo  add_raw_fastcall(1, 2)  [def_py_c_function/FASTCALL]",
        "add_raw_fastcall(1, 2)",
        g,
    )

    # Pure-Python baselines.
    _measure("Python -> Python py_noop(x)", "py_noop(1)", g)
    _measure("Python -> Python py_add(1, 2)", "py_add(1, 2)", g)

    # timeit floor: no call at all.
    _measure("Python builtin: 1 + 2  (no call)", "1 + 2", g)

    return 0


if __name__ == "__main__":
    sys.exit(main())
