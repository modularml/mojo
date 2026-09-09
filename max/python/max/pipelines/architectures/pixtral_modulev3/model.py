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
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
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
    ModuleV3MultiGraphPipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.utils import (
    parse_state_dict_from_weights,
)

from .batch_processor import PixtralModuleV3BatchProcessor
from .model_config import PixtralConfig
from .pixtral import PixtralLanguage, PixtralVision

logger = logging.getLogger("max.pipelines")


@dataclass
class PixtralInputs(ModelInputs):
    """Holds inputs for the Pixtral model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    # Vision inputs - ragged tensor of pre-extracted patches from all images
    pixel_patches: Buffer | None = None
    vision_attention_mask: Buffer | None = None
    vision_position_ids: Buffer | None = None
    image_token_indices: Buffer | None = None

    @property
    def has_vision_inputs(self) -> bool:
        """Returns true iff this includes vision model inputs."""
        return self.pixel_patches is not None


class PixtralModel(
    ModuleV3MultiGraphPipelineModelWithKVCache[TextAndVisionContext]
):
    """The overall interface to the Pixtral model."""

    model_config_cls: ClassVar[type[Any]] = PixtralConfig
    batch_processor_cls: ClassVar[type[PixtralModuleV3BatchProcessor]] = (
        PixtralModuleV3BatchProcessor
    )

    vision_model: Callable[..., Any] | None
    """Compiled vision model (encoder + projector) for a ragged batch of images."""

    language_model: Callable[..., Any]
    """Compiled language model with multimodal embedding merge."""

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

        self.vision_model, self.language_model = self.load_model()

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        assert isinstance(model_inputs, PixtralInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Pixtral has KV cache inputs, but none were provided"
        )

        # Process vision inputs: single call for all images in the batch
        if model_inputs.has_vision_inputs:
            assert self.vision_model is not None
            vision_output = self.vision_model(
                model_inputs.pixel_patches,
                model_inputs.vision_attention_mask,
                model_inputs.vision_position_ids,
            )
            image_embeddings = cast(Buffer, vision_output[0].driver_tensor)
            image_token_indices = model_inputs.image_token_indices
        else:
            assert isinstance(
                self.batch_processor, PixtralModuleV3BatchProcessor
            )
            image_embeddings = self.batch_processor.empty_image_embeddings()
            image_token_indices = (
                self.batch_processor.empty_image_token_indices()
            )

        model_outputs = self.language_model(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            image_embeddings,
            image_token_indices,
            *model_inputs.kv_cache_inputs.flatten(),
        )

        if len(model_outputs) == 3:
            return ModelOutputs(
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
            )
        else:
            return ModelOutputs(
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logits=cast(Buffer, model_outputs[0].driver_tensor),
            )

    def _load_state_dict(self) -> dict[str, Any]:
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights are currently supported in Pixtral models."
            )
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, self.weights, self.adapter
        )

        vision_state_dict: dict[str, WeightData] = {}
        language_state_dict: dict[str, WeightData] = {}
        for k, v in state_dict.items():
            if k.startswith(("vision_encoder.", "multi_modal_projector.")):
                # Remap vision_encoder.X -> X since PixtralVision owns
                # these components directly (no VisionEncoder sub-module).
                if k.startswith("vision_encoder."):
                    new_key = k.replace("vision_encoder.", "", 1)
                    # patch_conv is a Tensor param, not a Linear submodule,
                    # so strip the .weight suffix from the key.
                    if new_key == "patch_conv.weight":
                        new_key = "patch_conv"
                    vision_state_dict[new_key] = v
                else:
                    # multi_modal_projector.* stays as-is
                    vision_state_dict[k] = v
            elif k.startswith("language_model."):
                language_state_dict[k] = v

        # Validate that expected vision weight keys are present after remapping.
        expected_vision_prefixes = {
            "patch_conv",
            "layer_norm.",
            "patch_positional_embedding.",
            "transformer.",
            "multi_modal_projector.",
        }
        for key in vision_state_dict:
            if not any(
                key == prefix or key.startswith(prefix)
                for prefix in expected_vision_prefixes
            ):
                logger.warning(
                    "Unexpected vision weight key after remapping: %s", key
                )

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

    def _compile_vision_model(
        self,
        model_config: PixtralConfig,
        state_dict: dict[str, Any],
    ) -> Callable[..., Any]:
        patch_dim = self._patch_dim

        # ---- Build and compile vision model ----
        with F.lazy(), default_dtype(model_config.dtype):
            vision_nn = PixtralVision(model_config)
            vision_nn.to(self.devices[0])

        pixel_patches_type = TensorType(
            DType.float32,
            shape=["total_patches", patch_dim],
            device=DeviceRef.GPU(),
        )
        attention_mask_type = TensorType(
            DType.float32,
            shape=[1, 1, "total_patches", "total_patches"],
            device=DeviceRef.GPU(),
        )
        position_ids_type = TensorType(
            DType.int64,
            shape=["total_patches"],
            device=DeviceRef.GPU(),
        )

        return vision_nn.compile(
            pixel_patches_type,
            attention_mask_type,
            position_ids_type,
            weights=state_dict,
        )

    def _compile_language_model(
        self,
        model_config: PixtralConfig,
        state_dict: dict[str, Any],
    ) -> Callable[..., Any]:
        device_ref = DeviceRef.from_device(self.devices[0])

        # ---- Build and compile language model ----
        with F.lazy(), default_dtype(model_config.dtype):
            language_nn = PixtralLanguage(model_config)
            language_nn.kv_params = self.kv_params
            language_nn.to(self.devices[0])

        input_ids_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=DeviceRef.GPU()
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        image_embeddings_type = TensorType(
            model_config.dtype,
            shape=["total_image_tokens", model_config.hidden_size],
            device=DeviceRef.GPU(),
        )
        image_token_indices_type = TensorType(
            DType.int32,
            shape=["num_image_token_indices"],
            device=device_ref,
        )

        kv_inputs = self.kv_params.flattened_kv_inputs()

        return language_nn.compile(
            input_ids_type,
            input_row_offsets_type,
            return_n_logits_type,
            image_embeddings_type,
            image_token_indices_type,
            *kv_inputs,
            weights=state_dict,
        )
