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
"""Placeholder pipeline model for the DFlash2 standalone draft architecture.

The DFlash2 draft model (registered HuggingFace architecture name
``DFlash2DraftModel``) is never invoked as a standalone pipeline — it
borrows the target's ``embed_tokens`` and ``lm_head`` and is always loaded
and executed through the fused DFlash2 pipeline. This placeholder exists
solely so MAX's architecture registry can resolve the draft's
``architectures[0]`` during ``PipelineConfig`` validation when a DFlash2
recipe is used.
"""

from __future__ import annotations

from max.pipelines.lib import (
    ModelInputs,
    ModelOutputs,
)

from ..llama3.model import LlamaModelBase


class DFlash2Qwen3_5Model(LlamaModelBase):
    """Placeholder pipeline model for the DFlash2 draft architecture.

    See module docstring. ``execute`` raises because the draft is only ever
    run via the unified pipeline.
    """

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise NotImplementedError(
            "DFlash2Qwen3_5Model is a placeholder for architecture-registry"
            " lookup. The DFlash2 draft is run through the fused DFlash2"
            " pipeline."
        )
