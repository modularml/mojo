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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from max.driver import Buffer, Device, DLPackArray
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, Module, TensorType
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    MultiGraphPipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.profiler import traced

from .batch_processor import PixtralBatchProcessor
from .model_config import PixtralConfig
from .pixtral import PixtralLanguage, PixtralVision

logger = logging.getLogger("max.pipelines")


@dataclass
class PixtralInputs(ModelInputs):
    """Holds inputs for the Pixtral model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    # Vision inputs — ragged tensor of pre-extracted patches from all images.
    pixel_patches: Buffer | None = None
    vision_attention_mask: Buffer | None = None
    vision_position_ids: Buffer | None = None
    image_token_indices: Buffer | None = None

    @property
    def has_vision_inputs(self) -> bool:
        return self.pixel_patches is not None


class PixtralModel(MultiGraphPipelineModelWithKVCache[TextAndVisionContext]):
    """Pixtral pipeline model with separate vision and language graphs."""

    model_config_cls: ClassVar[type[Any]] = PixtralConfig
    batch_processor_cls: ClassVar[type[PixtralBatchProcessor]] = (
        PixtralBatchProcessor
    )

    vision_model: Model | None
    language_model: Model

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

        if self.pipeline_config.model.enable_echo:
            raise ValueError(
                "Pixtral model does not currently implement enable echo."
            )

        assert self._max_batch_size, "Expected max_batch_size to be set"

        if len(self.devices) > 1:
            raise NotImplementedError(
                "Pixtral does not support distributed inference"
            )

        self.vision_model, self.language_model = self.load_model(session)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, PixtralInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Pixtral requires KV cache inputs"
        )

        # Process vision inputs if present.
        if model_inputs.has_vision_inputs:
            assert self.vision_model is not None
            assert model_inputs.pixel_patches is not None
            assert model_inputs.vision_attention_mask is not None
            assert model_inputs.vision_position_ids is not None
            assert model_inputs.image_token_indices is not None

            vision_outputs = self.vision_model.execute(
                model_inputs.pixel_patches,
                model_inputs.vision_attention_mask,
                model_inputs.vision_position_ids,
            )
            assert isinstance(vision_outputs[0], Buffer)
            image_embeddings = vision_outputs[0]
            image_token_indices = model_inputs.image_token_indices
        else:
            assert isinstance(self.batch_processor, PixtralBatchProcessor)
            image_embeddings = self.batch_processor.empty_image_embeddings()
            image_token_indices = (
                self.batch_processor.empty_image_token_indices()
            )

        # Execute language model with text and image embeddings.
        language_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            image_embeddings,
            image_token_indices,
            *model_inputs.kv_cache_inputs.flatten(),
        )

        if len(language_outputs) == 3:
            assert isinstance(language_outputs[0], Buffer)
            assert isinstance(language_outputs[1], Buffer)
            assert isinstance(language_outputs[2], Buffer)
            return ModelOutputs(
                next_token_logits=language_outputs[0],
                logits=language_outputs[1],
                logit_offsets=language_outputs[2],
            )
        else:
            assert isinstance(language_outputs[0], Buffer)
            return ModelOutputs(
                next_token_logits=language_outputs[0],
                logits=language_outputs[0],
            )

    def _vision_graph_input_types(
        self, patch_dim: int
    ) -> Sequence[TensorType | BufferType]:
        return (
            TensorType(
                DType.float32,
                shape=["total_patches", patch_dim],
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.float32,
                shape=[1, 1, "total_patches", "total_patches"],
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.int64,
                shape=["total_patches"],
                device=DeviceRef.GPU(),
            ),
        )

    def _language_graph_input_types(self) -> Sequence[TensorType | BufferType]:
        device_ref = DeviceRef.from_device(self.devices[0])
        return (
            TensorType(DType.int64, shape=["total_seq_len"], device=device_ref),
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=device_ref,
            ),
            TensorType(
                DType.int64,
                shape=["return_n_logits"],
                device=DeviceRef.CPU(),
            ),
            TensorType(
                self.dtype,
                shape=[
                    "num_image_tokens",
                    self.huggingface_config.text_config.hidden_size,
                ],
                device=device_ref,
            ),
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=device_ref,
            ),
            *self.kv_params.flattened_kv_inputs(),
        )

    @traced
    def _build_vision_graph(
        self,
        config: PixtralConfig,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        patch_dim = self._patch_dim
        with Graph(
            "pixtral_vision",
            input_types=self._vision_graph_input_types(patch_dim),
            module=module,
        ) as graph:
            vision_nn = PixtralVision(config)
            vision_nn.load_state_dict(
                state_dict, weight_alignment=1, strict=True
            )

            pixel_patches, attention_mask, position_ids = graph.inputs
            output = vision_nn(
                pixel_patches.tensor,
                attention_mask.tensor,
                position_ids.tensor,
            )
            graph.output(output)
            return graph, vision_nn.state_dict()

    @traced
    def _build_language_graph(
        self,
        config: PixtralConfig,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        with Graph(
            "pixtral_language",
            input_types=self._language_graph_input_types(),
            module=module,
        ) as graph:
            language_nn = PixtralLanguage(config)
            language_nn.load_state_dict(
                state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=True,
            )

            (
                tokens,
                input_row_offsets,
                return_n_logits,
                image_embeddings,
                image_token_indices,
                *kv_cache_inputs,
            ) = graph.inputs

            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)
            outputs = language_nn(
                tokens=tokens.tensor,
                kv_collection=kv_collections[0],
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets.tensor,
                image_embeddings=image_embeddings.tensor,
                image_token_indices=image_token_indices.tensor,
            )
            graph.output(*outputs)
            return graph, language_nn.state_dict()

    def _load_state_dict(self) -> dict[str, Any]:
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "Only safetensors weights are currently supported in Pixtral."
            )
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, self.weights, self.adapter
        )

        vision_state_dict: dict[str, WeightData] = {}
        language_state_dict: dict[str, WeightData] = {}
        for k, v in state_dict.items():
            if k.startswith(("vision_encoder.", "multi_modal_projector.")):
                if k.startswith("vision_encoder."):
                    new_key = k.replace("vision_encoder.", "", 1)
                    vision_state_dict[new_key] = v
                else:
                    vision_state_dict[k] = v
            elif k.startswith("language_model."):
                language_state_dict[k] = v

        self._vision_weights_dict = vision_state_dict
        self._language_weights_dict = language_state_dict
        return state_dict

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> PixtralConfig:
        del state_dict
        vision_config = self.huggingface_config.vision_config
        self._patch_dim = (
            vision_config.num_channels
            * vision_config.patch_size
            * vision_config.patch_size
        )

        model_config = PixtralConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.return_logits = self.return_logits
        return model_config
