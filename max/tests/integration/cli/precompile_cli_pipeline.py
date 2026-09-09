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

"""CPU producer for the GPU CLI tests' compiled graphs.

Runs ``max warm-cache`` under MAX's virtual-device knobs so a GPU-less build
action compiles every graph one pipeline configuration needs for a GPU arch,
exporting each to ``--out`` for the consuming test to initialize.
"""

from __future__ import annotations

import click
from _cli_pipeline_flags import FLAG_SETS, pipeline_flags
from max._entrypoints import pipelines
from max.driver import set_virtual_cpu_target


@click.command()
@click.option("--out", "out_path", required=True, help="Directory to write.")
@click.option("--target", required=True, help="GPU target 'api:arch'.")
@click.option(
    "--cpu-target", required=True, help="Host-CPU codegen descriptor."
)
@click.option(
    "--flag-set",
    required=True,
    type=click.Choice(FLAG_SETS),
    help="The pipeline configuration to compile for.",
)
def main(out_path: str, target: str, cpu_target: str, flag_set: str) -> None:
    # Pins host codegen to a baseline the consuming GPU host can execute rather
    # than this build worker's own CPU. ``warm-cache --target`` sets the
    # remaining virtual-device knobs itself.
    set_virtual_cpu_target(cpu_target)
    pipelines.main(
        [
            "warm-cache",
            "--target",
            target,
            "--export-mefs",
            out_path,
            *pipeline_flags(flag_set),
        ]
    )


if __name__ == "__main__":
    main()
