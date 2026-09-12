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
"""Pins the task resolution ``warm-cache`` relies on.

``warm-cache`` resolves an architecture's task before retrieving its pipeline.
Taking ``retrieve``'s text-generation default instead sends a diffusion
architecture down a branch that reads the main model's HuggingFace config,
and diffusion pipelines have no main model, so warming their cache failed
before compiling anything.
"""

from __future__ import annotations

import max.pipelines  # noqa: F401  (registers the built-in architectures)
import pytest
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.modeling.types.task import PipelineTask


@pytest.mark.parametrize(
    "architecture_name",
    ["Flux2Pipeline", "Flux2KleinPipeline"],
)
def test_diffusion_architecture_resolves_to_pixel_generation(
    architecture_name: str,
) -> None:
    assert (
        PIPELINE_REGISTRY.retrieve_pipeline_task(architecture_name)
        is PipelineTask.PIXEL_GENERATION
    )


def test_text_architecture_still_resolves_to_text_generation() -> None:
    assert (
        PIPELINE_REGISTRY.retrieve_pipeline_task("LlamaForCausalLM")
        is PipelineTask.TEXT_GENERATION
    )
