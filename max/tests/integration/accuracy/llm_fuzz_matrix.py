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
"""Generate the GitHub Actions matrix for the llm-fuzz Ad Hoc workflow."""

# /// script
# dependencies = ["click>=8,<9"]
# ///

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Final

import click
from pipeline_matrix import PipelineEntry, entries_to_matrix, filter_entries

CONFIGS_DIR = Path("max/tests/integration/accuracy/llm_fuzz/configs")
SMOKE_TEST_PIPELINE = "nvidia/Kimi-K2.7-Code-NVFP4-ep-tp-dflash"


PIPELINES: Final[list[PipelineEntry]] = [
    PipelineEntry(
        pipeline="nvidia/Kimi-K2.7-Code-NVFP4-ep-tp-dflash",
        model_path="nvidia/Kimi-K2.7-Code-NVFP4",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=90,
    ),
    PipelineEntry(
        pipeline="google/gemma-4-31B-it",
        model_path="google/gemma-4-31B-it",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=60,
    ),
    PipelineEntry(
        pipeline="minimax/MiniMax-M3-MXFP8-ep-tp",
        model_path="MiniMaxAI/MiniMax-M3-MXFP8",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=90,
    ),
    # DP=2 + tiered KV offload — mirrors the production M3 serving recipe and
    # is the only config that exercises the overlap-scheduler staging race
    # (heterogeneous_batch_tool_call_invariance scenario).
    PipelineEntry(
        pipeline="minimax/MiniMax-M3-MXFP8-ep-dp",
        model_path="MiniMaxAI/MiniMax-M3-MXFP8",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=90,
    ),
    PipelineEntry(
        pipeline="amd/MiniMax-M3-MXFP4-ep-tp",
        model_path="amd/MiniMax-M3-MXFP4",
        runner="modrunner-mi355-4x",
        gpu_flag="--devices gpu:0,1,2,3",
        instance_type="bm.gpu.mi355x.4",
        timeout=90,
    ),
    # MXFP8 is vendor-agnostic (unlike MXFP4, which needs an AMD-repacked
    # checkpoint), so this reuses the same model_path as the B200 MXFP8
    # entries above; only the runner/topology differ. Mirrors the production
    # smoke-test recipe at
    # max_private/minimax_m3/recipes/mxfp8_tp4dp2ep8_8x_mi355.yaml.
    PipelineEntry(
        pipeline="amd/MiniMax-M3-MXFP8-ep-tp",
        model_path="MiniMaxAI/MiniMax-M3-MXFP8",
        runner="modrunner-mi355-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.mi355x.8",
        timeout=90,
    ),
    PipelineEntry(
        pipeline="nvidia/GLM-5.2-NVFP4-ep-tp-mtp",
        model_path="nvidia/GLM-5.2-NVFP4",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=90,
    ),
]


def _apply_local_model_path(
    entries: list[PipelineEntry],
    local_model_path: str,
    local_model_runner: str,
    local_model_instance_type: str,
) -> list[PipelineEntry]:
    """Reroute every entry to the runner that hosts the local model mount.

    Local model paths only resolve on the runner that has the volume
    mount; any dispatch that names a local path must land there
    regardless of the entry's default runner. The runner label and
    instance type are supplied by the caller (the workflow) so the
    matrix file itself stays free of deployment-specific labels.
    """
    return [
        replace(
            entry,
            runner=local_model_runner,
            instance_type=local_model_instance_type,
            model_path=local_model_path,
        )
        for entry in entries
    ]


def generate_matrix(
    event_name: str,
    selected_pipeline: str,
    base_ref: str | None,
    local_model_path: str,
    local_model_runner: str = "",
    local_model_instance_type: str = "",
) -> list[PipelineEntry]:
    """Return the filtered list of pipeline entries for the GH Actions matrix."""
    final = filter_entries(
        PIPELINES,
        event_name=event_name,
        selected_pipeline=selected_pipeline,
        base_ref=base_ref,
        configs_dir=CONFIGS_DIR,
        smoke_test_pipeline=SMOKE_TEST_PIPELINE,
    )

    if local_model_path:
        if not local_model_runner or not local_model_instance_type:
            print(
                "::error::--local-model-path requires --local-model-runner"
                " and --local-model-instance-type",
                file=sys.stderr,
            )
            sys.exit(1)
        final = _apply_local_model_path(
            final,
            local_model_path,
            local_model_runner,
            local_model_instance_type,
        )

    return final


@click.command()
@click.option(
    "--event-name",
    required=True,
    type=click.Choice(["pull_request", "schedule", "workflow_dispatch"]),
)
@click.option("--selected-pipeline", default="")
@click.option("--base-ref", default=None)
@click.option(
    "--local-model-path",
    default="",
    help=(
        "Local model-weights path (e.g. /mnt/local/data/...)."
        " When non-empty, every matrix entry is rerouted to the runner"
        " supplied via --local-model-runner and its model_path is"
        " overridden."
    ),
)
@click.option(
    "--local-model-runner",
    default="",
    help=(
        "Runner label that hosts the local model-weights mount. Required"
        " when --local-model-path is set."
    ),
)
@click.option(
    "--local-model-instance-type",
    default="",
    help=(
        "Instance type for the local-model-runner. Required when"
        " --local-model-path is set."
    ),
)
def main(
    event_name: str,
    selected_pipeline: str,
    base_ref: str | None,
    local_model_path: str,
    local_model_runner: str,
    local_model_instance_type: str,
) -> None:
    """Generate the GitHub Actions matrix for the llm-fuzz Ad Hoc workflow."""
    final = generate_matrix(
        event_name,
        selected_pipeline,
        base_ref,
        local_model_path,
        local_model_runner,
        local_model_instance_type,
    )
    click.echo(json.dumps(entries_to_matrix(final)))


if __name__ == "__main__":
    main()
