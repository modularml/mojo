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
"""Producer: warm ONLY the CPU slot of the eager-interpreter GC sweep.

Sets no virtual accelerator, so the builder's real (GPU-less, ``cfg=exec``)
device set is just the CPU slot. Writes ``matmul_cpu.mef`` / ``unary_cpu.mef``,
a CPU manifest (envelope with no ``gpu`` key, so ``manifest_adoptable`` admits
it only on a GPU-less box), and an ``accelerators=0`` warm stamp.

This action is config-independent per host arch: it takes only ``--cpu-target``
(no GPU ``--target``), so its Bazel action is byte-identical across every x86
GPU/CPU lane and the remote cache shares it, the ~20-min CPU sweep runs once
per host arch instead of once per lane. aarch64 is a valid CPU target here; the
x86 guard is GPU-mode only and lives in ``warm_accelerators``.
"""

from __future__ import annotations

import os
import platform
import sys

import click
from max.driver import set_virtual_cpu_target
from max.tests.integration.xarch_warm import warm_lib


@click.command()
@click.option(
    "--cpu-target",
    required=True,
    help="Virtual host-CPU target descriptor, e.g. "
    "'triple=x86_64-unknown-linux-gnu;cpu=x86-64-v3'.",
)
@click.option(
    "--family",
    default=None,
    help="Warm only this GC family (a GC_FAMILIES name); default warms all.",
)
def main(cpu_target: str, family: str | None) -> None:
    # Pin the host-CPU target (the warmed CPU kernels' compile key includes it).
    # No virtual accelerator: the real GPU-less worker presents only the CPU
    # slot, so the sweep and stamp are CPU-only.
    set_virtual_cpu_target(cpu_target)

    derived = os.environ["MODULAR_DERIVED_PATH"]
    entries = warm_lib.export_slots(
        derived,
        include_cpu=True,
        include_accelerators=False,
        only_family=family,
    )
    # Unset MODULAR_DERIVED_PATH means the warm wasn't pinned to the output dir,
    # so nothing consumable was produced, hard-fail the build action.
    if not warm_lib.write_stamp():
        print(
            "XARCH_WARM_CPU: MODULAR_DERIVED_PATH unset; cache not pinned",
            file=sys.stderr,
        )
        sys.exit(1)
    # CPU manifest: no 'gpu'/'device_count' key, so manifest_adoptable admits it
    # only on a GPU-less box. This is the final manifest the CPU consumer reads.
    envelope: dict[str, object] = {
        "host_arch": platform.machine(),
        "cpu_target": cpu_target,
        "toolchain": {"mode": "asserted"},
    }
    warm_lib.write_manifest(envelope, entries)


if __name__ == "__main__":
    main()
