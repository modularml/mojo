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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.sharding import (
    DeviceMesh,
)
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    ModuleV3MultiGraphPipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from transformers import AutoConfig

from .batch_processor import Gemma3MultiModalModuleV3BatchProcessor
from .model_config import Gemma3ForConditionalGenerationConfig
from .vision_model.gemma3multimodal import (
    Gemma3LanguageModel,
    Gemma3VisionModel,
)
from .weight_adapters import (
    convert_safetensor_language_state_dict,
    convert_safetensor_vision_state_dict,
)

logger = logging.getLogger("max.pipelines")


@dataclass
class Gemma3MultiModalModelInputs(ModelInputs):
    """Inputs for the Gemma3 multimodal model (V3)."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    pixel_values: Buffer | None = None
    image_token_indices: Buffer | None = None

    @property
    def has_vision_inputs(self) -> bool:
        return self.pixel_values is not None


class Gemma3MultiModalModelV3(
    ModuleV3MultiGraphPipelineModelWithKVCache[TextAndVisionContext],
):
    """Gemma 3 multimodal pipeline model using the ModuleV3 API."""

    model_config_cls: ClassVar[type[Any]] = Gemma3ForConditionalGenerationConfig
    batch_processor_cls: ClassVar[
        type[Gemma3MultiModalModuleV3BatchProcessor]
    ] = Gemma3MultiModalModuleV3BatchProcessor

    language_model: Callable[..., Any]
    vision_model: Callable[..., Any] | None

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
        self._max_batch_size = max_batch_size
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            memory_plan=memory_plan,
        )

        self.vision_model, self.language_model = self.load_model()

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return Gemma3ForConditionalGenerationConfig.get_num_layers(
            huggingface_config
        )

    def _load_state_dict(self) -> dict[str, Any]:
        assert self._max_batch_size, "Expected max_batch_size to be set"

        weights_dict = dict(self.weights.items())
        self._language_weights_dict = convert_safetensor_language_state_dict(
            weights_dict
        )
        self._vision_weights_dict = convert_safetensor_vision_state_dict(
            weights_dict
        )
        return {k: v.data() for k, v in weights_dict.items()}

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> Gemma3ForConditionalGenerationConfig:
        assert self._max_batch_size, "Expected max_batch_size to be set"

        model_config = Gemma3ForConditionalGenerationConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )
        self.config = model_config
        return model_config

    def _compile_vision_model(
        self,
        model_config: Gemma3ForConditionalGenerationConfig,
        state_dict: dict[str, Any],
    ) -> Callable[..., Any]:
        device_ref = DeviceRef.from_device(self.devices[0])

        # ---- Build and compile vision model ----
        with F.lazy():
            vision_nn = Gemma3VisionModel(model_config)
            vision_nn.to(self.devices[0])

        pixel_values_type = TensorType(
            DType.bfloat16,
            shape=[
                "batch_size",
                3,
                model_config.vision_config.image_size,
                model_config.vision_config.image_size,
            ],
            device=device_ref,
        )

        return vision_nn.compile(
            pixel_values_type,
            weights=state_dict,
        )

    def _compile_language_model(
        self,
        model_config: Gemma3ForConditionalGenerationConfig,
        state_dict: dict[str, Any],
    ) -> Callable[..., Any]:
        device_ref = DeviceRef.from_device(self.devices[0])
        n_devices = len(self.devices)

        mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))

        # ---- Build and compile language model ----
        with F.lazy():
            language_nn = Gemma3LanguageModel(model_config, self.kv_params)
            language_nn.to(mesh)

        assert isinstance(
            self._batch_processor, Gemma3MultiModalModuleV3BatchProcessor
        )
        language_input_types = (
            self._batch_processor.get_language_symbolic_inputs(
                kv_params=self.kv_params,
                device_ref=device_ref,
                hidden_size=model_config.text_config.hidden_size,
            )
        )

        return language_nn.compile(
            *language_input_types,
            weights=state_dict,
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        model_inputs = cast(Gemma3MultiModalModelInputs, model_inputs)

        image_embeddings: Buffer
        image_token_indices: Buffer
        if model_inputs.has_vision_inputs:
            assert self.vision_model is not None
            assert model_inputs.pixel_values is not None

            vision_output = self.vision_model(model_inputs.pixel_values)
            image_embeddings = cast(Buffer, vision_output[0].driver_tensor)

            assert model_inputs.image_token_indices is not None
            image_token_indices = model_inputs.image_token_indices
        else:
            assert isinstance(
                self._batch_processor, Gemma3MultiModalModuleV3BatchProcessor
            )
            image_embeddings = self._batch_processor.empty_image_embeddings()
            image_token_indices = (
                self._batch_processor.empty_image_token_indices()
            )

        kv_cache_inputs = model_inputs.kv_cache_inputs
        assert kv_cache_inputs is not None

        model_outputs = self.language_model(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            model_inputs.input_row_offsets,
            image_embeddings,
            image_token_indices,
            *kv_cache_inputs.flatten(),
        )

        def _to_buffer(t: Tensor) -> Buffer:
            """Extracts a Buffer from a potentially distributed Tensor."""
            if t.is_distributed:
                return cast(Buffer, t.local_shards[0].driver_tensor)
            return cast(Buffer, t.driver_tensor)

        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=_to_buffer(model_outputs[1]),
                next_token_logits=_to_buffer(model_outputs[0]),
                logit_offsets=_to_buffer(model_outputs[2]),
            )
        else:
            return ModelOutputs(
                logits=_to_buffer(model_outputs[0]),
                next_token_logits=_to_buffer(model_outputs[0]),
            )
