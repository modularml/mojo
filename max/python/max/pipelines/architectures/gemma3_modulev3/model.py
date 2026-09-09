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
from typing import Any, ClassVar, cast

from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.experimental.sharding import DeviceMesh
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
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

from .batch_processor import Gemma3ModuleV3BatchProcessor
from .gemma3 import Gemma3
from .model_config import Gemma3Config

logger = logging.getLogger("max.pipelines")


@dataclass
class Gemma3Inputs(ModelInputs):
    """A class representing inputs for the Gemma3 model (ModuleV3).

    This class encapsulates the input tensors required for the Gemma3 model
    execution.
    """

    tokens: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: Buffer
    """Tensor containing the offsets for each row in the ragged input
    sequence."""

    return_n_logits: Buffer
    """Number of logits to return."""


class Gemma3Model(
    LogProbabilitiesMixin,
    ModuleV3PipelineModelWithKVCache[TextContext],
):
    """A Gemma3 pipeline model for text generation using the ModuleV3 API.

    This class integrates the Gemma3 architecture with the MAX Engine pipeline
    infrastructure using the V3 eager compilation API.
    """

    model_config_cls: ClassVar[type[Any]] = Gemma3Config
    batch_processor_cls: ClassVar[type[Gemma3ModuleV3BatchProcessor]] = (
        Gemma3ModuleV3BatchProcessor
    )

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
        # Detect multimodal models by presence of text_config
        self._is_multimodal = hasattr(self.huggingface_config, "text_config")

        self.model = self.load_model()

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return Gemma3Config.get_num_layers(huggingface_config)

    def _hf_config_for_weights(self) -> AutoConfig | None:
        if self._is_multimodal:
            return self.huggingface_config.text_config
        return self.huggingface_config

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        text_config = (
            self.huggingface_config.text_config
            if self._is_multimodal
            else self.huggingface_config
        )
        model_config = Gemma3Config.initialize_from_config(
            self.pipeline_config, text_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=text_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )
        return model_config

    def _instantiate_module(self, model_config: Any) -> Any:
        n_devices = len(self.devices)
        mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))
        nn_model = Gemma3(model_config, self.kv_params)
        nn_model.to(mesh)
        return nn_model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Gemma3 model with the prepared inputs."""
        assert isinstance(model_inputs, Gemma3Inputs)
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
        else:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
            )
