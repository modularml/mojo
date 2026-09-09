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
"""Unit tests for compose_lib.merge (pure stdlib; runs on any platform)."""

from __future__ import annotations

import json
from pathlib import Path

from max.tests.integration.xarch_warm import compose_lib

_STAMP = Path("cache/.max_cache/eager_gc_warm_stamp.json")


def _fragment(
    root: Path,
    name: str,
    envelope: dict[str, object],
    entries: list[dict[str, str]],
    accel: int,
) -> Path:
    """Write a fake warm-fragment dir: a MEF per entry, a manifest, a stamp."""
    frag = root / name
    (frag / _STAMP.parent).mkdir(parents=True)
    for entry in entries:
        (frag / entry["mef"]).write_text(name)  # stand-in MEF payload
    (frag / "manifest.json").write_text(
        json.dumps({"envelope": envelope, "entries": entries})
    )
    (frag / _STAMP).write_text(
        json.dumps(
            {"context": f"accelerators={accel};cpu=x86_64;accel=sm_100a"}
        )
    )
    return frag


def test_merge_unions_and_picks_gpu_stamp(tmp_path: Path) -> None:
    cpu = _fragment(
        tmp_path,
        "cpu_matmul",
        {
            "host_arch": "x86_64",
            "cpu_target": "cpu=x86-64-v3",
            "toolchain": {"mode": "asserted"},
        },
        [{"family": "matmul", "device_class": "cpu", "mef": "matmul_cpu.mef"}],
        accel=0,
    )
    accel = _fragment(
        tmp_path,
        "accel_matmul",
        {"gpu": {"arch": "sm_100a"}, "device_count": 4},
        [
            {
                "family": "matmul",
                "device_class": "gpu:0",
                "mef": "matmul_slot_0.mef",
            }
        ],
        accel=4,
    )
    out = tmp_path / "out"
    out.mkdir()

    envelope, entries = compose_lib.merge([cpu, accel], out)

    # union of MEFs
    assert {p.name for p in out.glob("*.mef")} == {
        "matmul_cpu.mef",
        "matmul_slot_0.mef",
    }
    # merged envelope carries both CPU and GPU bits
    assert envelope["host_arch"] == "x86_64"
    assert envelope["gpu"] == {"arch": "sm_100a"}
    assert envelope["device_count"] == 4
    # union of entries
    assert len(entries) == 2
    # GPU stamp (accelerators=4) wins over the CPU stamp (accelerators=0)
    stamp = json.loads((out / _STAMP).read_text())
    assert stamp["context"].startswith("accelerators=4;")


def test_merge_cpu_only_keeps_zero_stamp(tmp_path: Path) -> None:
    frag_a = _fragment(
        tmp_path,
        "cpu_a",
        {
            "host_arch": "x86_64",
            "cpu_target": "cpu=x86-64-v3",
            "toolchain": {"mode": "asserted"},
        },
        [{"family": "matmul", "device_class": "cpu", "mef": "matmul_cpu.mef"}],
        accel=0,
    )
    frag_b = _fragment(
        tmp_path,
        "cpu_b",
        {
            "host_arch": "x86_64",
            "cpu_target": "cpu=x86-64-v3",
            "toolchain": {"mode": "asserted"},
        },
        [{"family": "unary", "device_class": "cpu", "mef": "unary_cpu.mef"}],
        accel=0,
    )
    out = tmp_path / "out"
    out.mkdir()

    envelope, entries = compose_lib.merge([frag_a, frag_b], out)

    assert "gpu" not in envelope  # CPU-only manifest
    assert len(entries) == 2
    stamp = json.loads((out / _STAMP).read_text())
    assert stamp["context"].startswith("accelerators=0;")
