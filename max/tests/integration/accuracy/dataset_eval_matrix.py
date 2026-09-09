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
"""Generate the GitHub Actions matrix for Pipeline Dataset Evaluation."""

# /// script
# dependencies = ["click>=8,<9"]
# ///

from __future__ import annotations

import json
from pathlib import Path

import click
from pipeline_matrix import PipelineEntry, entries_to_matrix, filter_entries

CONFIGS_DIR = Path("max/tests/integration/accuracy/dataset_eval_configs")
# Switch to llama once llama is fixed in dataset eval.
SMOKE_TEST_PIPELINE = "sentence-transformers/all-mpnet-base-v2"


PIPELINES: list[PipelineEntry] = [
    PipelineEntry(
        pipeline="meta-llama/Meta-Llama-3-8B-Instruct",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="OpenGVLab/InternVL3-8B-Instruct",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="modularai/Llama-3.1-8B-Instruct-GGUF",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="sentence-transformers/all-mpnet-base-v2",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="LiquidAI/LFM2.5-1.2B-Instruct",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        # LFM2 requires batch_size=1 (SSM/conv state can't be batched across
        # sequences in the same way as pure-attention models), so all evaluation
        # tasks run sequentially rather than in parallel batches.  3 hours
        # instead of the usual 2 hours accounts for that serialization overhead.
        timeout=180,
    ),
    PipelineEntry(
        pipeline="Qwen/Qwen2.5-VL-3B-Instruct",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="Qwen/Qwen2.5-VL-7B-Instruct",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="OpenGVLab/InternVL3-38B-Instruct",
        runner="modrunner-b200-2x",
        gpu_flag="--devices gpu:0,1",
        instance_type="bm.gpu.b200.2",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="Qwen/Qwen2.5-VL-32B-Instruct",
        runner="modrunner-b200-2x",
        gpu_flag="--devices gpu:0,1",
        instance_type="bm.gpu.b200.2",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="RedHatAI/gemma-3-27b-it-FP8-dynamic",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="google/gemma-4-26B-A4B-it",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="google/gemma-4-31B-it",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="nvidia/Gemma-4-26B-A4B-NVFP4",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="nvidia/Gemma-4-31B-IT-NVFP4",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic",
        runner="modrunner-b200-4x",
        gpu_flag="--devices gpu:0,1,2,3",
        instance_type="bm.gpu.b200.4",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="deepseek-ai/DeepSeek-R1-longbench-v2",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=390,  # 6.5 hours
    ),
    PipelineEntry(
        pipeline="deepseek-ai/DeepSeek-V3.1-Terminus",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=90,  # 1.5 hours
    ),
    PipelineEntry(
        pipeline="nvidia/DeepSeek-V3.1-NVFP4-longbench-v2",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=390,  # 6.5 hours
    ),
    PipelineEntry(
        pipeline="nvidia/DeepSeek-V3.1-NVFP4-fp8kv-longbench-v2",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=390,  # 6.5 hours
    ),
    PipelineEntry(
        pipeline="nvidia/Kimi-K2.7-Code-NVFP4-ep-tp-eagle3-longbench-v2",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=390,  # 6.5 hours
    ),
]


def generate_matrix(
    event_name: str,
    selected_pipeline: str,
    base_ref: str | None,
) -> list[PipelineEntry]:
    """Return the filtered list of pipeline entries for the GH Actions matrix."""
    return filter_entries(
        PIPELINES,
        event_name=event_name,
        selected_pipeline=selected_pipeline,
        base_ref=base_ref,
        configs_dir=CONFIGS_DIR,
        smoke_test_pipeline=SMOKE_TEST_PIPELINE,
    )


@click.command()
@click.option(
    "--event-name",
    required=True,
    type=click.Choice(["pull_request", "schedule", "workflow_dispatch"]),
)
@click.option("--selected-pipeline", default="")
@click.option("--base-ref", default=None)
def main(event_name: str, selected_pipeline: str, base_ref: str | None) -> None:
    """Generate the GitHub Actions matrix for Pipeline Dataset Evaluation."""
    final = generate_matrix(event_name, selected_pipeline, base_ref)
    click.echo(json.dumps(entries_to_matrix(final)))


if __name__ == "__main__":
    main()
