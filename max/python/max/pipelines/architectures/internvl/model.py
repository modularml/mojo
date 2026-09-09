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
from max.graph import DeviceRef, Graph, Module, TensorType, Type
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.nn.comm import Signals
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    MultiGraphPipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan

from .batch_processor import InternVLBatchProcessor
from .internvl import InternVLLanguageModel, InternVLVisionModel
from .model_config import InternVLConfig
from .tokenizer import _get_image_context_token_id
from .weight_adapters import (
    convert_internvl_language_model_state_dict,
    convert_internvl_vision_model_state_dict,
)

logger = logging.getLogger("max.pipelines")


@dataclass
class InternVLInputs(ModelInputs):
    """A class representing inputs for the InternVL model."""

    tokens: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: list[Buffer]
    """Per-device tensors containing the offsets for each row in the ragged
    input sequence.
    """

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer
    """Number of logits to return, used by speculative decoding for example."""

    # Vision inputs.
    pixel_values: list[Buffer] | None = None
    """Pixel values for vision inputs."""

    image_token_indices: list[Buffer] | None = None
    """Per-device pre-computed indices of image tokens in the input sequence."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


def assert_image_embeddings_invariant(
    image_embeddings: Sequence[Buffer], image_token_indices: Sequence[Buffer]
) -> None:
    # Check for shape mismatch that causes scatter_nd OOB access.
    for i, (embed, indices) in enumerate(
        zip(image_embeddings, image_token_indices, strict=True)
    ):
        embed_count = embed.shape[0]
        indices_count = indices.shape[0]
        if embed_count != indices_count:
            logger.error(
                f"[CRITICAL] Device {i}: Vision embedding count ({embed_count}) "
                f"!= image token indices count ({indices_count})."
            )
        assert embed_count == indices_count, (
            f"Vision embedding shape mismatch on device {i}: {embed_count} embeddings "
            f"but {indices_count} indices."
        )


class InternVLModel(
    AlwaysSignalBuffersMixin,
    MultiGraphPipelineModelWithKVCache[TextAndVisionContext],
):
    """An InternVL pipeline model for multimodal text generation."""

    model_config_cls: ClassVar[type[Any]] = InternVLConfig
    batch_processor_cls: ClassVar[type[InternVLBatchProcessor]] = (
        InternVLBatchProcessor
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

        self.vision_model, self.language_model = self.load_model(session)

    def _load_state_dict(self) -> dict[str, Any]:
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "InternVL currently only supports safetensors weights"
            )

        # Get processed state dict for language and vision models.
        # NOTE: use weights_dict to mean WeightData, and state dict to mean
        # DLPack arrays, since state dict is overloaded.
        weights_dict = dict(self.weights.items())
        self._language_weights_dict = (
            convert_internvl_language_model_state_dict(weights_dict)
        )
        self._vision_weights_dict = convert_internvl_vision_model_state_dict(
            weights_dict
        )
        return {}

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> InternVLConfig:
        del state_dict

        internvl_config = InternVLConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        internvl_config.finalize(
            huggingface_config=self.huggingface_config,
            llm_state_dict=self._language_weights_dict,
            vision_state_dict=self._vision_weights_dict,
            dtype=self.dtype,
            return_logits=self.return_logits,
        )
        return internvl_config

    def _build_vision_graph(
        self,
        config: InternVLConfig,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the vision model graph for processing images."""
        # Define input types for the vision model
        # Use static dimensions from the vision config
        image_size = config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        # Calculate number of patches in each dimension
        height_patches = image_size // patch_size
        width_patches = image_size // patch_size

        # Expect pre-extracted patches from the tokenizer.
        # Use bfloat16 to match the tokenizer's output.
        pixel_values_types = [
            TensorType(
                DType.bfloat16,
                shape=[
                    "batch_size",
                    height_patches,
                    width_patches,
                    3,
                    patch_size,
                    patch_size,
                ],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        # Create signal types for distributed communication
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        # Initialize graph with input types
        with Graph(
            "internvl_vision",
            input_types=[*pixel_values_types, *signals.input_types()],
            module=module,
        ) as graph:
            # Build vision model architecture.
            vision_model = InternVLVisionModel(config)
            vision_model.load_state_dict(
                state_dict=state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=True,
            )

            # Unpack inputs (one per device).
            pixel_values = [
                inp.tensor for inp in graph.inputs[: len(self.devices)]
            ]

            # Extract signal buffers (one per device).
            signal_buffers = [
                inp.buffer for inp in graph.inputs[len(self.devices) :]
            ]

            # Execute vision model: pixel_values -> image_embeddings.
            image_embeddings = vision_model(pixel_values, signal_buffers)

            # Set graph outputs.
            graph.output(*image_embeddings)

            return graph, vision_model.state_dict()

    def _language_graph_input_types(self) -> tuple[Type[Any], ...]:
        # Generate DeviceRef.
        device_ref = DeviceRef.from_device(self.devices[0])

        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        kv_inputs = self.kv_params.get_symbolic_inputs()

        # Construct Graph Inputs
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_types = [
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        # Add image embeddings type - one per device, can be empty for text-only inputs
        image_embeddings_types = [
            TensorType(
                self.dtype,
                shape=[
                    "num_image_tokens",
                    self.huggingface_config.llm_config.hidden_size,
                ],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        # Add image token indices type
        image_token_indices_types = [
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        # Flatten kv types for each device
        flattened_kv_types = kv_inputs.flatten()

        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        return (
            tokens_type,
            return_n_logits_type,
            *input_row_offsets_types,
            *image_embeddings_types,
            *image_token_indices_types,
            *signals.input_types(),
            *flattened_kv_types,
        )

    def _build_language_graph(
        self,
        config: InternVLConfig,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the language model graph for text generation with image embeddings."""
        # Initialize graph with input types.
        with Graph(
            "internvl_language",
            input_types=self._language_graph_input_types(),
            module=module,
        ) as graph:
            image_context_token_id = _get_image_context_token_id(
                self.huggingface_config
            )
            language_model = InternVLLanguageModel(
                config, image_context_token_id
            )
            language_model.load_state_dict(
                state_dict=state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
            )

            # Unpack inputs.
            tokens, return_n_logits, *variadic_args = graph.inputs

            # Extract input_row_offsets (one per device).
            input_row_offsets = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract image embeddings (one per device).
            image_embeddings = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract image token indices.
            image_token_indices = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Multi-GPU passes a signal buffer per device: unmarshal these.
            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]

            # Unmarshal the remaining arguments, which are for KV cache.
            kv_cache = self._unflatten_kv_inputs(
                variadic_args[len(self.devices) :]
            )

            # Execute language model: text + image embeddings -> logits
            outputs = language_model(
                tokens=tokens.tensor,
                signal_buffers=signal_buffers,
                kv_collections=kv_cache,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets,
                image_embeddings=image_embeddings,
                image_token_indices=image_token_indices,
            )

            graph.output(*outputs)

            return graph, language_model.state_dict()

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the InternVL model with the prepared inputs."""
        assert model_inputs.kv_cache_inputs is not None, (
            "InternVL requires KV cache inputs"
        )
        assert isinstance(model_inputs, InternVLInputs)

        # Process vision inputs if present.
        image_embeddings: list[Buffer]
        image_token_indices: list[Buffer]
        if model_inputs.has_vision_inputs:
            assert self.vision_model is not None
            assert model_inputs.pixel_values is not None
            assert model_inputs.image_token_indices is not None

            # Execute vision model: pixel_values -> image_embeddings.
            vision_outputs = self.vision_model.execute(
                *model_inputs.pixel_values, *model_inputs.signal_buffers
            )
            assert len(vision_outputs) == len(self.devices)

            image_embeddings = [
                output
                for output in vision_outputs
                if isinstance(output, Buffer)
            ]
            image_token_indices = model_inputs.image_token_indices

            assert_image_embeddings_invariant(
                image_embeddings, image_token_indices
            )
        else:
            # Initialize empty tensors for text-only mode.
            assert isinstance(self.batch_processor, InternVLBatchProcessor)
            image_embeddings = self.batch_processor.empty_image_embeddings()
            image_token_indices = (
                self.batch_processor.empty_image_token_indices()
            )

        # Prepare KV cache inputs as list of tensors
        assert model_inputs.kv_cache_inputs

        # Execute language model with text and image embeddings
        language_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *model_inputs.input_row_offsets,
            *image_embeddings,
            *image_token_indices,
            *model_inputs.signal_buffers,
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
