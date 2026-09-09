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
"""Shared model-side boilerplate for the unified spec-decode pipelines."""

from __future__ import annotations

import numpy as np
from max.driver import Buffer, Device
from max.engine import Model

from ..interfaces.pipeline_model import ModelInputs, UnifiedEagleOutputs

__all__ = ["_UnifiedSpecDecodeModelMixin"]


class _UnifiedSpecDecodeModelMixin:
    """Shared execute / _next_seed for unified spec-decode models."""

    # Provided by the concrete PipelineModel this mixin is combined with.
    model: Model
    devices: list[Device]

    # Dflash-style block drafters derive the draft width from the draft
    # checkpoint at model construction and expose it here instead of
    # rewriting SpeculativeConfig; None means the config already carries
    # the value.
    resolved_num_speculative_tokens: int | None = None

    @property
    def _spec_decode_model(self) -> Model:
        # self.model by default; Kimi wrappers override to self.language_model.
        return self.model

    def execute(self, model_inputs: ModelInputs) -> UnifiedEagleOutputs:
        model_outputs = self._spec_decode_model.execute(*model_inputs.buffers)
        if len(model_outputs) not in (3, 4):
            raise RuntimeError(
                f"{type(self).__name__} graph returned {len(model_outputs)} "
                "outputs; expected 3 (num_accepted_draft_tokens, next_tokens, "
                "next_draft_tokens) or 4 (+ next_draft_probs_full, under "
                "draft_proposal='sampled')."
            )
        return UnifiedEagleOutputs(
            num_accepted_draft_tokens=model_outputs[0],
            next_tokens=model_outputs[1],
            next_draft_tokens=model_outputs[2],
            next_draft_probs_full=model_outputs[3]
            if len(model_outputs) == 4
            else None,
        )

    def _next_seed(self) -> Buffer:
        self._seed_counter = getattr(self, "_seed_counter", 0) + 1
        return Buffer.from_numpy(
            np.array([self._seed_counter], dtype=np.uint64)
        ).to(self.devices[0])
