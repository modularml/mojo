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

from .batch_processor import Idefics3BatchProcessor
from .model_config import Idefics3Config
from .text_model.idefics3_text import Idefics3LanguageModel
from .vision_model.idefics3_vision import Idefics3VisionModel
from .weight_adapters import (
    convert_idefics3_language_model_state_dict,
    convert_idefics3_vision_model_state_dict,
)

logger = logging.getLogger("max.pipelines")


def _assert_image_embeddings_invariant(
    image_embeddings: Buffer, image_token_indices: Buffer
) -> None:
    """Validates that image embeddings count matches image token indices count.

    This prevents out-of-bounds access during scatter operations where image
    embeddings are placed at specific token positions.

    Args:
        image_embeddings: Single tensor of image embeddings
        image_token_indices: Single tensor of image token indices

    Raises:
        AssertionError: If embedding count doesn't match indices count
    """
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
    """A class representing inputs for the Idefics3 model."""

    tokens: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: Buffer
    """Tensor containing the offsets for each row in the ragged input sequence."""

    return_n_logits: Buffer
    """Number of logits to return, used by speculative decoding for example."""

    # Vision inputs
    pixel_values: Buffer | None = None
    """Pixel values for vision inputs."""

    image_token_indices: Buffer | None = None
    """Pre-computed indices of image tokens in the input sequence."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


class Idefics3Model(MultiGraphPipelineModelWithKVCache[TextAndVisionContext]):
    """An Idefics3 pipeline model for multimodal text generation."""

    model_config_cls: ClassVar[type[Any]] = Idefics3Config
    batch_processor_cls: ClassVar[type[Idefics3BatchProcessor]] = (
        Idefics3BatchProcessor
    )

    vision_model: Model | None
    """The compiled vision model for processing images."""

    language_model: Model
    """The compiled language model for text generation."""

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

        self.image_token_id = self.huggingface_config.image_token_id

        self.vision_model, self.language_model = self.load_model(session)

    def _load_state_dict(self) -> dict[str, Any]:
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "Idefics3 currently only supports safetensors weights"
            )

        # Get processed state dict for language and vision models.
        # NOTE: use weights_dict to mean WeightData, and state dict to mean
        # DLPack arrays, since state dict is overloaded.
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

    def _build_vision_graph(
        self,
        config: Idefics3Config,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the vision model graph for processing images."""
        # Define input types for the vision model
        # Use static dimensions from the vision config
        image_size = config.vision_config.image_size

        # Expect pre-extracted patches from the tokenizer.
        pixel_values_type = TensorType(
            DType.bfloat16,
            shape=[
                "batch_size",
                3,
                image_size,
                image_size,
            ],
            # Expect the input on device 0.
            device=DeviceRef.GPU(),
        )

        # Initialize graph with input types
        with Graph(
            "idefics3_vision", input_types=[pixel_values_type], module=module
        ) as graph:
            # Build vision model architecture.
            vision_model = Idefics3VisionModel(
                config.vision_config,
                dtype=self.dtype,
                device=DeviceRef.from_device(self.devices[0]),
            )
            vision_model.load_state_dict(
                state_dict=state_dict,
                weight_alignment=1,
                strict=True,
            )

            # Unpack inputs.
            (pixel_values,) = graph.inputs

            # Execute vision model: pixel_values -> image_embeddings.
            image_embeddings = vision_model(pixel_values.tensor)

            # Set graph outputs.
            graph.output(image_embeddings)

            return graph, vision_model.state_dict()

    def _language_graph_input_types(self) -> Sequence[TensorType | BufferType]:
        # Generate DeviceRef.
        device_ref = DeviceRef.from_device(self.devices[0])

        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        # Construct Graph Inputs
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )

        # Add image embeddings type - one per device, can be empty for text-only inputs
        image_embeddings_type = TensorType(
            self.dtype,
            shape=[
                "num_image_tokens",
                self.huggingface_config.text_config.hidden_size,
            ],
            device=device_ref,
        )

        # Add image token indices type
        image_token_indices_type = TensorType(
            DType.int32, shape=["total_image_tokens"], device=device_ref
        )

        return (
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            image_embeddings_type,
            image_token_indices_type,
            *self.kv_params.flattened_kv_inputs(),
        )

    def _build_language_graph(
        self,
        config: Idefics3Config,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the language model graph for text generation with image embeddings."""
        # Initialize graph with input types.
        with Graph(
            "idefics3_language",
            input_types=self._language_graph_input_types(),
            module=module,
        ) as graph:
            # Build language model architecture.
            language_model = Idefics3LanguageModel(
                config.text_config,
                config.image_token_id,
                self.dtype,
                DeviceRef.from_device(self.devices[0]),
            )
            language_model.load_state_dict(
                state_dict=state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=True,
            )

            # Unpack inputs
            (
                tokens,
                input_row_offsets,
                return_n_logits,
                image_embeddings,
                image_token_indices,
                *variadic_args,
            ) = graph.inputs

            # Unmarshal the remaining arguments, which are for KV cache
            kv_cache = self._unflatten_kv_inputs(variadic_args)

            # Execute language model: text + image embeddings -> logits
            outputs = language_model(
                tokens=tokens.tensor,
                kv_collection=kv_cache[0],
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets.tensor,
                image_embeddings=image_embeddings.tensor,
                image_token_indices=image_token_indices.tensor,
            )

            graph.output(*outputs)

            return graph, language_model.state_dict()

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Idefics3 model with the prepared inputs."""
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
            vision_outputs = self.vision_model.execute(
                model_inputs.pixel_values,
            )

            assert isinstance(vision_outputs[0], Buffer)

            image_embeddings = vision_outputs[0]
            image_token_indices = model_inputs.image_token_indices

            _assert_image_embeddings_invariant(
                image_embeddings, image_token_indices
            )
        else:
            # Initialize empty tensors for text-only mode.
            assert isinstance(self.batch_processor, Idefics3BatchProcessor)
            image_embeddings = self.batch_processor.empty_image_embeddings()
            image_token_indices = (
                self.batch_processor.empty_image_token_indices()
            )

        # Prepare KV cache inputs as list of tensors
        assert model_inputs.kv_cache_inputs is not None, (
            "Idefics3 has KV cache inputs, but none were provided"
        )

        # Execute language model with text and image embeddings
        language_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            image_embeddings,
            image_token_indices,
            *model_inputs.kv_cache_inputs.flatten(),
        )

        # Return model outputs based on what the language model returns
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
