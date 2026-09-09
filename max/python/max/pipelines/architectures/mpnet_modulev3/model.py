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
"""Defines the MPNet V3 pipeline model.

Implementation is based on MPNetModel from the transformers library,
using the V3 eager API (max.experimental.nn).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.graph.buffer_utils import cast_tensor_to
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    ModuleV3PipelineModel,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan

from .batch_processor import MPNetModuleV3BatchProcessor
from .graph import MPNetModel
from .model_config import MPNetConfig

logger = logging.getLogger("max.pipelines")

PAD_VALUE = 1


@dataclass
class MPNetInputs(ModelInputs):
    """Input tensors for the MPNet model."""

    next_tokens_batch: Buffer
    attention_mask: Buffer


class MPNetPipelineModel(ModuleV3PipelineModel[TextContext]):
    model_config_cls: ClassVar[type[MPNetConfig]] = MPNetConfig
    batch_processor_cls: ClassVar[type[MPNetModuleV3BatchProcessor]] = (
        MPNetModuleV3BatchProcessor
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
        self.model = self.load_model()

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, MPNetInputs)
        model_outputs = self.model(
            model_inputs.next_tokens_batch, model_inputs.attention_mask
        )
        result = model_outputs[0].driver_tensor
        assert isinstance(result, Buffer)
        return ModelOutputs(logits=result)

    def _create_model_config(self, state_dict: dict[str, Any]) -> MPNetConfig:
        del state_dict
        return self.arch_config_as(MPNetConfig)

    def _prepare_state_dict(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> dict[str, Any]:
        del model_config
        # Cast weights to match the model's configured dtype (e.g. float32
        # safetensor weights -> bfloat16 when default_encoding is bfloat16).
        # V3 compile() requires exact dtype matching unlike V2 load_state_dict.
        target_dtype = self.dtype
        cast_state_dict: dict[str, Any] = {}
        for key, value in state_dict.items():
            buf = (
                Buffer.from_dlpack(value)
                if not isinstance(value, Buffer)
                else value
            )
            if buf.dtype != target_dtype and buf.dtype.is_float():
                buf = cast_tensor_to(buf, target_dtype)
            cast_state_dict[key] = buf
        return cast_state_dict

    def _instantiate_module(self, model_config: Any) -> MPNetModel:
        assert isinstance(model_config, MPNetConfig)
        nn_model = MPNetModel(model_config)
        nn_model.to(self.devices[0])
        return nn_model
