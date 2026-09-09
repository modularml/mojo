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
from typing import Any, ClassVar, Literal, cast

import numpy as np
from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.graph.weights import Weights, WeightsAdapter
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
from max.pipelines.lora import LoRATargetModule

from .batch_processor import Llama3ModuleV3BatchProcessor
from .llama3 import Llama3
from .lora import LLAMA3_LORA_TARGETS
from .model_config import Llama3Config

logger = logging.getLogger("max.pipelines")


@dataclass
class Llama3Inputs(ModelInputs):
    """A class representing inputs for the Llama3 model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        if isinstance(self.input_row_offsets, np.ndarray):
            input_row_offsets = Buffer.from_numpy(self.input_row_offsets).to(
                self.tokens.device
            )
        else:
            input_row_offsets = self.input_row_offsets
        return (
            self.tokens,
            self.return_n_logits,
            input_row_offsets,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
            *self.lora_buffers,
        )


class Llama3Model(
    LogProbabilitiesMixin,
    ModuleV3PipelineModelWithKVCache[TextContext],
):
    """Llama3 pipeline model using the ModuleV3 API."""

    model_config_cls: ClassVar[type[Any]] = Llama3Config
    batch_processor_cls: ClassVar[type[Llama3ModuleV3BatchProcessor]] = (
        Llama3ModuleV3BatchProcessor
    )

    config_class: type[Any] = Llama3Config
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    attention_bias: bool = False

    #: Serve LoRA via the ModuleV3 adapters-as-inputs path (LoRAManagerV3).
    lora_modulev3: ClassVar[bool] = True
    #: The projections ModuleV3 LoRA wraps: the fused qkv (one adapter per
    #: q/k/v) and o_proj.
    lora_targets: ClassVar[tuple[LoRATargetModule, ...]] = LLAMA3_LORA_TARGETS

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
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
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
        self.model = self.load_model()

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        model_config = self.config_class.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )
        return model_config

    def _instantiate_module(self, model_config: Any) -> Any:
        nn_model = Llama3(model_config, self.kv_params)
        nn_model.to(self.devices[0])
        return nn_model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        model_inputs = cast(Llama3Inputs, model_inputs)
        model_outputs = self.model(*model_inputs.buffers)

        has_offsets = self.return_logits in (
            ReturnLogits.VARIABLE,
            ReturnLogits.ALL,
        )
        has_hidden_states = self.return_hidden_states != ReturnHiddenStates.NONE

        if has_offsets and has_hidden_states:
            assert len(model_outputs) == 4
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
                hidden_states=cast(Buffer, model_outputs[3].driver_tensor),
            )
        elif has_offsets:
            assert len(model_outputs) == 3
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
            )
        elif has_hidden_states:
            assert len(model_outputs) == 2
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                hidden_states=cast(Buffer, model_outputs[1].driver_tensor),
            )
        else:
            assert len(model_outputs) == 1
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
            )
