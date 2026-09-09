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
"""Placeholder pipeline model for the standalone DSpark draft architecture.

Speculators-format DSpark drafts (registered HuggingFace architecture name
``DSparkDraftModel``) are never invoked as a standalone pipeline — they are
always loaded and executed through the target's unified pipeline model
(e.g. ``UnifiedDSparkGemma4_31BModel``). This placeholder exists solely so
MAX's architecture registry can resolve the draft's ``architectures[0]``
during ``PipelineConfig`` validation.
"""

from __future__ import annotations

from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.pipeline_model import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
)


class DSparkDraftPlaceholderModel(PipelineModel[TextContext]):
    """Placeholder for the draft-only registration; never constructed."""

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise NotImplementedError(
            "Speculators-format DSpark drafts only run inside a unified"
            " pipeline model."
        )
