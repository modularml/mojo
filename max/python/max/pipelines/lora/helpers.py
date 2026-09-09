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

"""Shared LoRA helpers used by the serving path."""

from __future__ import annotations

from max.pipelines.modeling.types.pipeline import (
    Pipeline,
    PipelineInputsType,
    PipelineOutputType,
)

from .modulev3 import LoRAManagerV3


def get_lora_manager(
    pipeline: Pipeline[PipelineInputsType, PipelineOutputType],
) -> LoRAManagerV3 | None:
    """Returns the LoRA manager from the pipeline if LoRA is enabled."""
    manager: LoRAManagerV3 | None = None

    if hasattr(pipeline, "_pipeline_model"):
        manager = pipeline._pipeline_model._lora_manager
    elif hasattr(pipeline, "speech_lm_pipeline"):
        manager = pipeline.speech_lm_pipeline._pipeline_model._lora_manager
    elif hasattr(pipeline, "pipeline_model"):
        manager = pipeline.pipeline_model._lora_manager

    return manager
