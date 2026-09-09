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
"""Implements the DeepseekV2 nn.model (ModuleV3)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

from max.driver import Buffer, Device, DeviceSpec
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.nn.kv_cache import (
    KVCacheParamInterface,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    ModuleV3PipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.memory_estimation import MemoryPlan
from transformers import AutoConfig
from typing_extensions import override

from .batch_processor import DeepseekV2ModuleV3BatchProcessor
from .deepseekV2 import DeepseekV2
from .model_config import DeepseekV2Config

logger = logging.getLogger("max.pipelines")


@dataclass
class DeepseekV2Inputs(ModelInputs):
    """Inputs for the DeepseekV2 model."""

    tokens: Buffer
    input_row_offsets: Buffer

    return_n_logits: Buffer = field(kw_only=True)


class DeepseekV2Model(
    LogProbabilitiesMixin, ModuleV3PipelineModelWithKVCache[TextContext]
):
    model_config_cls: ClassVar[type[Any]] = DeepseekV2Config
    batch_processor_cls: ClassVar[type[DeepseekV2ModuleV3BatchProcessor]] = (
        DeepseekV2ModuleV3BatchProcessor
    )

    model: Callable[..., Any]

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
        return_logits: ReturnLogits = ReturnLogits.ALL,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        if pipeline_config.model.device_specs[0] == DeviceSpec.cpu():
            raise ValueError("DeepseekV2 currently only supported on gpu.")

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

        self.model = self.load_model()

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, DeepseekV2Inputs)

        curr_kv_cache_inputs = model_inputs.kv_cache_inputs
        assert curr_kv_cache_inputs is not None
        model_outputs = self.model(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            model_inputs.input_row_offsets,
            *curr_kv_cache_inputs.flatten(),
        )
        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
            )
        return ModelOutputs(
            logits=cast(Buffer, model_outputs[0].driver_tensor),
            next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        return DeepseekV2Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights supported in DeepseekV2."
            )
        return super()._load_state_dict()

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        del state_dict
        model_config = DeepseekV2Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.max_batch_context_length = (
            self.planned_max_batch_total_tokens
            or model_config.max_batch_context_length
        )
        return model_config

    @override
    def _instantiate_module(self, model_config: Any) -> Any:
        nn_model = DeepseekV2(model_config, self.kv_params)
        nn_model.to(self.devices[0])
        return nn_model
