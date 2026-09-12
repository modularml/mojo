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
"""Defines the ModernBERT pipeline model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
)
from max.pipelines.modeling.dataprocessing import collate_batch

from .graph import build_graph
from .model_config import ModernBertConfig


@dataclass
class ModernBertInputs(ModelInputs):
    """Input tensors for the ModernBERT model."""

    next_tokens_batch: Buffer
    attention_mask: Buffer


class ModernBertPipelineModel(PipelineModel[TextContext]):
    """ModernBERT graph-backed pipeline model."""

    model_config_cls: ClassVar[type[ModernBertConfig]] = ModernBertConfig

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        self.model = self.load_model(session)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, ModernBertInputs)
        model_outputs = self.model.execute(
            model_inputs.next_tokens_batch,
            model_inputs.attention_mask,
        )
        assert isinstance(model_outputs[0], Buffer)
        return ModelOutputs(logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModernBertInputs:
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]
        tokens = [ctx.tokens.active for ctx in context_batch]

        pad_value = getattr(self.huggingface_config, "pad_token_id", 0)
        next_tokens_batch, _ = collate_batch(
            tokens,
            pad_value=pad_value,
            batch_size=len(tokens),
        )
        attention_mask = (next_tokens_batch != pad_value).astype(np.float32)

        return ModernBertInputs(
            next_tokens_batch=Buffer.from_numpy(next_tokens_batch).to(
                self.devices[0]
            ),
            attention_mask=Buffer.from_numpy(attention_mask).to(
                self.devices[0]
            ),
        )

    def load_model(self, session: InferenceSession) -> Model:
        with CompilationTimer("model") as timer:
            if self.adapter:
                state_dict = self.adapter(dict(self.weights.items()))
            else:
                state_dict = {
                    key: value.data() for key, value in self.weights.items()
                }
            config = ModernBertConfig.initialize(self.pipeline_config)
            graph = build_graph(config, state_dict)
            timer.mark_build_complete()
            model = session.load(graph, weights_registry=state_dict)

        return model
