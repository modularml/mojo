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

from typing import Any, Literal

from max.driver import Device
from max.engine import InferenceSession
from max.graph import Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.utils import parse_state_dict_from_weights
from typing_extensions import override

from ..llama3.model import LlamaModelBase
from .eagle_llama3 import EagleLlama3
from .model_config import Llama3Config


class EagleLlama3Model(LlamaModelBase):
    """EAGLE Llama3 draft model pipeline implementation."""

    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"

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
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.LAST,
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
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise NotImplementedError(
            "Cannot directly execute EagleLlama3Model. You should use the UnifiedEagleLlama3Model instead."
        )

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        draft_model: MAXModelConfig | None = self.pipeline_config.draft_model
        assert draft_model is not None
        draft_hf_config = draft_model.huggingface_config
        assert draft_hf_config is not None
        return parse_state_dict_from_weights(
            self.pipeline_config,
            self.weights,
            self.adapter,
            hf_config=draft_hf_config,
        )

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Llama3Config:
        draft_model: MAXModelConfig | None = self.pipeline_config.draft_model
        assert draft_model is not None
        draft_hf_config = draft_model.huggingface_config
        assert draft_hf_config is not None
        model_config = Llama3Config.initialize_from_config(
            self.pipeline_config, draft_hf_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=draft_hf_config,
            state_dict=state_dict,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )
        return model_config

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Llama3Config,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        assert len(self.devices) == 1, "EAGLE only supports single device"

        single_model: EagleLlama3 = EagleLlama3(model_config)

        single_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,  # We don't use the input layer norm and output layer norm
        )
        weights_registry = single_model.state_dict()

        with Graph(
            "eagle_llama3",
            input_types=single_model.input_types(
                self.kv_params,
                needs_hidden_state_input=True,
            ),
        ) as graph:
            (
                tokens,
                input_row_offsets,
                return_n_logits,
                hidden_states,
                *kv_cache_inputs,
            ) = graph.inputs

            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

            outputs = single_model(
                tokens.tensor,
                kv_collections[0],
                return_n_logits.tensor,
                input_row_offsets.tensor,
                hidden_states.tensor,
            )
            graph.output(*outputs)
            return graph, weights_registry
