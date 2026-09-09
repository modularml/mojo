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
"""Producer: warm ONLY the accelerator slots of the eager-interpreter GC sweep.

Sets the virtual device (api/arch from ``--target``, count =
``WARM_DEVICE_COUNT``) so a GPU-less builder presents that many virtual
accelerators of the requested arch, then compiles + exports one MEF per
accelerator slot. It does NOT compile the CPU slot, that is ``warm_cpu``'s
job, and keeping it out of here is the whole point of the split: the CPU sweep
must run once per host arch, not once per (lane-keyed) GPU action.

Writes ``matmul_slot_{i}.mef`` / ``unary_slot_{i}.mef``, an accel manifest
FRAGMENT (only the gpu envelope bits this action owns, ``gpu.arch`` +
``device_count``, plus the gpu entries; ``compose`` merges it with
``warm_cpu``'s manifest), and the GPU-context warm stamp
(``accelerators=WARM_DEVICE_COUNT;...;accel=<arch>``).
"""

from __future__ import annotations

import os
import platform
import sys

import click
from max.driver import (
    set_virtual_cpu_target,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.tests.integration.xarch_warm import warm_lib
from max.tests.integration.xarch_warm.warm_config import WARM_DEVICE_COUNT


@click.command()
@click.option(
    "--target",
    required=True,
    help="Virtual GPU target as 'api:arch', e.g. 'cuda:sm_100a'.",
)
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
def main(target: str, cpu_target: str, family: str | None) -> None:
    device_api, _, device_arch = target.partition(":")
    if not device_api or not device_arch:
        raise click.BadParameter(
            "must be 'api:arch', e.g. 'cuda:sm_100a'", param_hint="--target"
        )

    # Warm the GPU sweep on x86 only: the sole GPU eager consumer lane is the
    # x86 + B200 runner, so an aarch64-warmed GPU cache has no validated
    # consumer. Abort cleanly rather than emit an unexercised artifact. This
    # guard is GPU-mode only, aarch64 (Graviton/m7g) IS a valid CPU-only
    # target in warm_cpu.
    if platform.machine() not in ("x86_64", "amd64"):
        print(
            f"XARCH_WARM_ACCELERATORS: ABANDONED, builder arch "
            f"{platform.machine()!r} is not x86; the only eager GPU consumer "
            "lane is x86, so an aarch64 GPU warm has no validated consumer.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Pin the host-CPU target (the graphs' compile key includes it) and present
    # WARM_DEVICE_COUNT virtual accelerators of the requested arch, EARLY --
    # before the first _interpreter_ops import, whose device set is frozen from
    # accelerator_count() at import; a GPU-less worker thus presents the virtual
    # accelerators. The CPU slot is intentionally left to warm_cpu.
    set_virtual_cpu_target(cpu_target)
    set_virtual_device_api(device_api)
    set_virtual_device_target_arch(device_arch)
    set_virtual_device_count(WARM_DEVICE_COUNT)

    derived = os.environ["MODULAR_DERIVED_PATH"]
    entries = warm_lib.export_slots(
        derived,
        include_cpu=False,
        include_accelerators=True,
        only_family=family,
    )
    # Unset MODULAR_DERIVED_PATH means the warm wasn't pinned to the output dir,
    # so nothing consumable was produced, hard-fail the build action.
    if not warm_lib.write_stamp():
        print(
            "XARCH_WARM_ACCELERATORS: MODULAR_DERIVED_PATH unset; cache not "
            "pinned",
            file=sys.stderr,
        )
        sys.exit(1)
    # Fragment: only the gpu envelope bits this action owns, the arch a
    # consumer must match + the warmed slot count (a ceiling). compose adds
    # host_arch/cpu_target/toolchain from warm_cpu's manifest.
    envelope: dict[str, object] = {
        "gpu": {"arch": device_arch},
        "device_count": WARM_DEVICE_COUNT,
    }
    warm_lib.write_manifest(envelope, entries)
    # The device classes below never include "cpu", the dedup guarantee.


if __name__ == "__main__":
    main()
