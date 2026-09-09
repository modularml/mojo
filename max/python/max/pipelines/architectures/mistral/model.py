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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.nn.layer import Module
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphPipelineModelWithKVCache,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from transformers import AutoConfig
from typing_extensions import override

from .batch_processor import MistralBatchProcessor
from .distributed_mistral import DistributedMistral
from .mistral import Mistral
from .model_config import MistralConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class MistralInputs(ModelInputs):
    """A class representing inputs for the Mistral model.

    This class encapsulates the input tensors required for the Mistral model execution:
    - tokens: A tensor containing the input token IDs
    - input_row_offsets: A tensor containing the offsets for each row in the ragged input sequence
    - return_n_logits: A tensor containing the number of expected token logits.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""
    return_n_logits: Buffer


class MistralModel(GraphPipelineModelWithKVCache[TextContext]):
    model_config_cls: ClassVar[type[Any]] = MistralConfig
    batch_processor_cls: ClassVar[type[MistralBatchProcessor]] = (
        MistralBatchProcessor
    )

    model: Model
    """Compiled and initialized model ready for inference."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        self.model = self.load_model(session)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Runs the graph."""
        assert isinstance(model_inputs, MistralInputs)

        curr_kv_cache_inputs = model_inputs.kv_cache_inputs
        assert curr_kv_cache_inputs is not None

        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs.flatten(),
        )
        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(model_outputs)

    @override
    def load_model(self, session: InferenceSession) -> Model:
        if self.pipeline_config.model.enable_echo:
            raise ValueError(
                "Mistral model does not currently implement enable echo."
            )
        return super().load_model(session)

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights are currently supported in Mistral models."
            )
        return super()._load_state_dict()

    @override
    def _hf_config_for_weights(self) -> AutoConfig | None:
        return getattr(
            self.huggingface_config, "text_config", self.huggingface_config
        )

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> MistralConfig:
        del state_dict
        text_config = self._hf_config_for_weights()
        assert text_config is not None
        model_config = MistralConfig.initialize_from_config(
            self.pipeline_config, text_config, max_seq_len=self.max_seq_len
        )
        model_config.return_logits = self.return_logits
        return model_config

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: MistralConfig,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        assert self.batch_processor is not None
        graph_inputs = tuple(
            self.batch_processor.get_symbolic_inputs(
                kv_params=self.kv_params,
                device_refs=self.device_refs,
            )
        )

        # Build Graph
        nn_model: Module
        if len(self.devices) > 1:
            nn_model = DistributedMistral(model_config)
            nn_model.load_state_dict(
                state_dict,
                weight_alignment=1,
                strict=False,  # TODO(MODELS-551) vision tower weights not used
            )
            weights_registry = nn_model.state_dict()

            with Graph("mistral", input_types=[*graph_inputs]) as graph:
                tokens, input_row_offsets, return_n_logits, *variadic_args = (
                    graph.inputs
                )

                # Multi-GPU passes a signal buffer per device: unmarshal these.
                signal_buffers = [
                    v.buffer for v in variadic_args[: len(self.devices)]
                ]

                # Unmarshal the remaining arguments, which are for KV cache.
                kv_caches_per_dev = self._unflatten_kv_inputs(
                    variadic_args[len(self.devices) :]
                )

                outputs = nn_model(
                    tokens.tensor,
                    signal_buffers,
                    kv_caches_per_dev,
                    return_n_logits.tensor,
                    input_row_offsets.tensor,
                )

                graph.output(*outputs)
                return graph, weights_registry

        nn_model = Mistral(model_config)
        nn_model.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=False,  # TODO(MODELS-551) vision tower weights not used
        )
        weights_registry = nn_model.state_dict()

        with Graph("mistral", input_types=graph_inputs) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)
            outputs = nn_model(
                tokens.tensor,
                kv_collections[0],
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )
            graph.output(*outputs)
            return graph, weights_registry
