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
"""Idefics3 pipeline model (ModuleV3).

Handles compilation and execution of both the vision and language models
using the V3 eager API (``F.lazy()`` + ``compile()``).
"""

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
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
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
from max.pipelines.weights.weight_loading import (
    auto_cast_weights_from_env,
)

from .batch_processor import Idefics3ModuleV3BatchProcessor
from .model_config import Idefics3Config
from .text_model.idefics3_text import Idefics3Language
from .vision_model.idefics3_vision import Idefics3VisionModel
from .weight_adapters import (
    convert_idefics3_language_model_state_dict,
    convert_idefics3_vision_model_state_dict,
)

logger = logging.getLogger("max.pipelines")


def _assert_image_embeddings_invariant(
    image_embeddings: Buffer, image_token_indices: Buffer
) -> None:
    """Validates that image embeddings count matches image token indices count."""
    embed_count = image_embeddings.shape[0]
    indices_count = image_token_indices.shape[0]

    if embed_count != indices_count:
        logger.error(
            f"[CRITICAL] Vision embedding count ({embed_count}) "
            f"!= image token indices count ({indices_count})."
        )

    assert embed_count == indices_count, (
        f"Vision embedding shape mismatch: {embed_count} embeddings "
        f"but {indices_count} indices."
    )


@dataclass
class Idefics3Inputs(ModelInputs):
    """Inputs for the Idefics3 model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    # Vision inputs
    pixel_values: Buffer | None = None
    image_token_indices: Buffer | None = None

    @property
    def has_vision_inputs(self) -> bool:
        return self.pixel_values is not None


class Idefics3Model(
    ModuleV3MultiGraphPipelineModelWithKVCache[TextAndVisionContext]
):
    """An Idefics3 pipeline model using the ModuleV3 API."""

    model_config_cls: ClassVar[type[Any]] = Idefics3Config
    batch_processor_cls: ClassVar[type[Idefics3ModuleV3BatchProcessor]] = (
        Idefics3ModuleV3BatchProcessor
    )

    vision_model: Callable[..., Any] | None
    """The compiled vision model."""

    language_model: Callable[..., Any]
    """The compiled language model."""

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

        self.vision_model, self.language_model = self.load_model()
        self.image_token_id = self.huggingface_config.image_token_id

    def _load_state_dict(self) -> dict[str, Any]:
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "Idefics3 currently only supports safetensors weights"
            )

        weights_dict = dict(self.weights.items())
        self._language_weights_dict = (
            convert_idefics3_language_model_state_dict(weights_dict)
        )
        self._vision_weights_dict = convert_idefics3_vision_model_state_dict(
            weights_dict
        )
        return {}

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> Idefics3Config:
        del state_dict

        idefics3_config = Idefics3Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        idefics3_config.finalize(
            huggingface_config=self.huggingface_config,
            llm_state_dict=self._language_weights_dict,
            return_logits=self.return_logits,
        )
        return idefics3_config

    def _compile_vision_model(
        self,
        config: Idefics3Config,
        state_dict: dict[str, Any],
    ) -> Callable[..., Any]:
        """Build and compile the vision model using F.lazy()."""
        image_size = config.vision_config.image_size

        pixel_values_type = TensorType(
            DType.bfloat16,
            shape=["batch_size", 3, image_size, image_size],
            device=DeviceRef.GPU(),
        )

        with F.lazy(), default_dtype(config.vision_config.dtype):
            nn_vision = Idefics3VisionModel(config.vision_config)
            nn_vision.to(self.devices[0])

        compiled = nn_vision.compile(
            pixel_values_type,
            weights=state_dict,
            auto_cast=auto_cast_weights_from_env(),
        )

        return compiled

    def _compile_language_model(
        self,
        config: Idefics3Config,
        state_dict: dict[str, Any],
    ) -> Callable[..., Any]:
        """Build and compile the language model using F.lazy()."""
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)

        assert isinstance(self._batch_processor, Idefics3ModuleV3BatchProcessor)
        language_input_types = (
            self._batch_processor.get_language_symbolic_inputs(
                kv_params=self.kv_params,
                device_ref=device_ref,
                hidden_size=self.huggingface_config.text_config.hidden_size,
                embedding_dtype=self.dtype,
            )
        )

        with F.lazy(), default_dtype(config.text_config.dtype):
            nn_language = Idefics3Language(
                config.text_config,
                config.image_token_id,
                self.kv_params,
            )
            nn_language.to(self.devices[0])

        compiled = nn_language.compile(
            *language_input_types,
            weights=state_dict,
        )

        return compiled

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the Idefics3 model."""
        assert model_inputs.kv_cache_inputs is not None, (
            "Idefics3 requires KV cache inputs"
        )
        assert isinstance(model_inputs, Idefics3Inputs)

        # Process vision inputs if present.
        image_embeddings: Buffer
        image_token_indices: Buffer
        if model_inputs.has_vision_inputs:
            assert self.vision_model is not None
            assert model_inputs.pixel_values is not None
            assert model_inputs.image_token_indices is not None

            # Execute vision model: pixel_values -> image_embeddings.
            # V3 compiled model returns Tensor, not Buffer.
            vision_output = self.vision_model(model_inputs.pixel_values)
            image_embeddings = cast(Buffer, vision_output.driver_tensor)
            image_token_indices = model_inputs.image_token_indices

            _assert_image_embeddings_invariant(
                image_embeddings, image_token_indices
            )
        else:
            assert isinstance(
                self._batch_processor, Idefics3ModuleV3BatchProcessor
            )
            image_embeddings = self._batch_processor.empty_image_embeddings()
            image_token_indices = (
                self._batch_processor.empty_image_token_indices()
            )

        # Execute language model.
        language_outputs = self.language_model(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            image_embeddings,
            image_token_indices,
            *model_inputs.kv_cache_inputs.flatten(),
        )

        # Unpack outputs (V3 returns Tensor objects with .driver_tensor).
        if self.return_logits in (ReturnLogits.VARIABLE, ReturnLogits.ALL):
            return ModelOutputs(
                next_token_logits=cast(
                    Buffer, language_outputs[0].driver_tensor
                ),
                logits=cast(Buffer, language_outputs[1].driver_tensor),
                logit_offsets=cast(Buffer, language_outputs[2].driver_tensor),
            )
        else:
            return ModelOutputs(
                next_token_logits=cast(
                    Buffer, language_outputs[0].driver_tensor
                ),
                logits=cast(Buffer, language_outputs[0].driver_tensor),
            )
