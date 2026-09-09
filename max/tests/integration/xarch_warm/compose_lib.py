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
"""Merge logic for the eager-GC warm compose step (CLI in compose.py).

Unions N per-family warm fragments into one consumer cache dir, equivalent in
shape to the old unified output (so consumers are unchanged): the union of
every fragment's ``*.mef`` and manifest entries, the merged envelope (CPU
fragments give host_arch/cpu_target/toolchain, accelerator fragments give
gpu/device_count), and the warm stamp from the fragment with the most
accelerators. Pure stdlib (in its own library so the unit test can import it
without also owning compose.py).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

_MANIFEST_NAME = "manifest.json"
# Mirrors gc_compile._cache_dir(): <MODULAR_DERIVED_PATH>/cache/.max_cache.
_CACHE_SUBDIR = Path("cache") / ".max_cache"
# Mirrors gc_compile._WARM_STAMP_NAME.
_WARM_STAMP_NAME = "eager_gc_warm_stamp.json"


def _copy_mefs(sources: Iterable[Path], out: Path) -> None:
    """Copy every per-slot ``*.mef`` from *sources* into *out* under its bare
    name (the manifest references MEFs by bare name relative to
    MODULAR_DERIVED_PATH, so they must sit alongside it)."""
    for src in sources:
        for mef in sorted(src.glob("*.mef")):
            shutil.copy2(mef, out / mef.name)


def _accel_count(context: str) -> int:
    """Leading ``accelerators=N`` of a stamp context, or -1 if absent/malformed."""
    head = context.partition(";")[0]
    try:
        return int(head.removeprefix("accelerators="))
    except ValueError:
        return -1


def merge(
    warm_dirs: list[Path], out: Path
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Union every fragment's MEFs + manifest entries into *out*, merge their
    envelopes, and copy the stamp from the fragment with the most accelerators
    (GPU stamp beats a CPU ``accelerators=0`` stamp). Returns (envelope,
    entries)."""
    _copy_mefs(warm_dirs, out)
    envelope: dict[str, object] = {}
    entries: list[dict[str, str]] = []
    best_src: Path | None = None
    best_accel = 0
    for warm_dir in warm_dirs:
        manifest = json.loads((warm_dir / _MANIFEST_NAME).read_text())
        # CPU and accelerator fragments contribute disjoint envelope keys
        # (host_arch/cpu_target/toolchain vs gpu/device_count), so update is safe.
        envelope.update(manifest["envelope"])
        entries += manifest["entries"]
        stamp_path = warm_dir / _CACHE_SUBDIR / _WARM_STAMP_NAME
        if stamp_path.exists():
            context = json.loads(stamp_path.read_text()).get("context", "")
            count = _accel_count(context)
            # First stamped fragment wins; a later one only on a higher count.
            if best_src is None or count > best_accel:
                best_accel, best_src = count, warm_dir
    if best_src is not None:
        shutil.copytree(
            best_src / _CACHE_SUBDIR, out / _CACHE_SUBDIR, dirs_exist_ok=True
        )
    (out / _MANIFEST_NAME).write_text(
        json.dumps({"envelope": envelope, "entries": entries}, indent=2)
    )
    return envelope, entries
