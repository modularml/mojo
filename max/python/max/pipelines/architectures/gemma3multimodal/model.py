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
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, DLPackArray
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, Module, TensorType, Type
from max.graph.weights import WeightData, Weights, WeightsAdapter
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
from transformers import AutoConfig

from .batch_processor import Gemma3MultiModalBatchProcessor
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
    """A class representing inputs for the Gemma3 multi modal model.

    This class encapsulates the input tensors required for the Gemma3 multi
    modal model, for text and vision processing.

    Args:
        tokens: Input token IDs.
        input_row_offsets: Input row offsets (ragged tensors).
        return_n_logits: Number of logits to return.
        signal_buffers: Device buffers for distributed communication.
        kv_cache_inputs: Inputs for the KV cache.
        pixel_values: Raw pixel values for vision inputs. Defaults to ``None``.
        image_token_indices: Pre-computed indices of image tokens. Defaults to
            ``None``.
    """

    tokens: npt.NDArray[np.integer[Any]] | Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: npt.NDArray[np.integer[Any]] | list[Buffer]
    """Tensor containing the offsets for each row in the ragged input sequence,
    or the attention mask for the padded input sequence. For distributed execution,
    this can be a list of tensors, one per device."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer
    """Number of logits to return, used by speculative decoding for example."""

    pixel_values: list[Buffer] | None = None
    """Raw pixel values for vision inputs: [batch, channels, height, width]."""

    image_token_indices: list[Buffer] | None = None
    """Pre-computed indices of image tokens in the input sequence."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


class Gemma3_MultiModalModel(
    AlwaysSignalBuffersMixin,
    MultiGraphPipelineModelWithKVCache[TextAndVisionContext],
):
    """Gemma 3 multimodal pipeline model for text generation.

    This class integrates the Gemma 3 multimodal architecture with the MAX
    pipeline infrastructure, handling model loading, KV cache management, and
    input preparation for inference.

    Args:
        pipeline_config: The configuration settings for the entire pipeline.
        session: The MAX inference session managing the runtime.
        huggingface_config: The configuration loaded from HuggingFace
            (:obj:`transformers.AutoConfig`).
        devices: A list of MAX devices (:obj:`max.driver.Device`) to
            run the model on.
        kv_cache_config: Configuration settings for the Key-Value cache
            (:obj:`max.pipelines.max_config.KVCacheConfig`).
        weights: The model weights (:obj:`max.graph.weights.Weights`).
        adapter: An optional adapter to modify weights before loading
            (:obj:`max.graph.weights.WeightsAdapter`).
        return_logits: The number of top logits to return from the model
            execution.
    """

    model_config_cls: ClassVar[type[Any]] = Gemma3ForConditionalGenerationConfig
    batch_processor_cls: ClassVar[type[Gemma3MultiModalBatchProcessor]] = (
        Gemma3MultiModalBatchProcessor
    )

    language_model: Model
    """The compiled and initialized MAX Engine model ready for inference."""

    vision_model: Model | None
    """The compiled and initialized MAX Engine vision model ready for inference."""

    # The vision and text towers are in the same weights file, but are in
    # separate models, so load_state_dict will naturally be loading subsets in
    # each case.
    _strict_state_dict_loading = True

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

        # signal_buffers are provided by AlwaysSignalBuffersMixin as a cached_property
        # to avoid GPU memory allocation during compile-only mode (cross-compilation).
        # Force initialization here to ensure buffers are ready before model execution,
        # preventing potential race conditions in multi-GPU scenarios.
        _ = self.signal_buffers

        self.vision_model, self.language_model = self.load_model(session)

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Gets the number of hidden layers from the HuggingFace configuration."""
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

    def _language_model_input_types(
        self, config: Gemma3ForConditionalGenerationConfig
    ) -> Sequence[TensorType | BufferType]:
        """Prepare the Tensor input types that our language graph will work with"""
        device_ref = DeviceRef.from_device(self.devices[0])
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

        image_embeddings_types = [
            TensorType(
                DType.bfloat16,
                shape=[
                    "num_image_tokens",
                    config.text_config.hidden_size,
                ],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        image_token_indices_types = [
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

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
            *self.kv_params.flattened_kv_inputs(),
        )

    def _build_language_graph(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the language model with our input types and graph"""
        with Graph(
            getattr(self.huggingface_config, "model_type", "Gemma3"),
            input_types=self._language_model_input_types(config),
            module=module,
        ) as graph:
            language_model = Gemma3LanguageModel(config)
            language_model.load_state_dict(
                state_dict,
                weight_alignment=1,
                strict=self._strict_state_dict_loading,
            )

            # Unpack inputs following InternVL pattern
            tokens, return_n_logits, *variadic_args = graph.inputs

            # Extract input_row_offsets (one per device)
            input_row_offsets = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract image embeddings (one per device).
            image_embeddings = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            image_token_indices = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract signal buffers (one per device)
            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract KV cache inputs from the unified {sliding, global} tree.
            kv_cache_local, kv_cache_global = (
                self.kv_params.unflatten_basic_kv_tree(iter(variadic_args))
            )

            outputs = language_model(
                tokens=tokens.tensor,
                signal_buffers=signal_buffers,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets,
                sliding_kv_collections=kv_cache_local,
                global_kv_collections=kv_cache_global,
                image_embeddings=image_embeddings,
                image_token_indices=image_token_indices,
            )
            graph.output(*outputs)
        return graph, language_model.state_dict()

    def _vision_model_input_types(
        self, config: Gemma3ForConditionalGenerationConfig
    ) -> list[Type[Any]]:
        """Build the vision model graph for processing images."""
        pixel_values_types = [
            TensorType(
                DType.bfloat16,
                shape=[
                    "batch_size",
                    3,
                    config.vision_config.image_size,
                    config.vision_config.image_size,
                ],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        # Create signal types for distributed communication
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )
        return [*pixel_values_types, *signals.input_types()]

    def _build_vision_graph(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the vision model with our input types and graph"""
        vision_graph_name = (
            getattr(self.huggingface_config, "model_type", "Gemma3") + "_vision"
        )
        with Graph(
            vision_graph_name,
            input_types=self._vision_model_input_types(config),
            module=module,
        ) as graph:
            vision_model = Gemma3VisionModel(
                config,
                device=DeviceRef.from_device(self.devices[0]),
            )

            vision_model.load_state_dict(
                state_dict=state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=self._strict_state_dict_loading,
            )

            pixel_values = [
                inp.tensor for inp in graph.inputs[: len(self.devices)]
            ]

            signal_buffers = [
                inp.buffer for inp in graph.inputs[len(self.devices) :]
            ]

            image_embeddings = vision_model(pixel_values, signal_buffers)

            graph.output(*image_embeddings)

            return graph, vision_model.state_dict()

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """If required, execute the vision model, then continue to execute the
        language model.  Either pass through image embeddings or create an empty
        placeholder."""
        model_inputs = cast(Gemma3MultiModalModelInputs, model_inputs)

        input_row_offsets = model_inputs.input_row_offsets

        image_embeddings: list[Buffer]
        image_token_indices: list[Buffer]
        if model_inputs.has_vision_inputs:
            assert self.vision_model is not None
            assert model_inputs.pixel_values is not None

            # Execute vision model: patched pixel_values -> image_embeddings.
            vision_outputs = self.vision_model.execute(
                *model_inputs.pixel_values, *model_inputs.signal_buffers
            )
            assert len(vision_outputs) == len(self.devices)

            image_embeddings = [
                output
                for output in vision_outputs
                if isinstance(output, Buffer)
            ]
            assert model_inputs.image_token_indices is not None
            image_token_indices = model_inputs.image_token_indices
        else:
            assert isinstance(
                self._batch_processor, Gemma3MultiModalBatchProcessor
            )
            image_embeddings = self._batch_processor.empty_image_embeddings()
            image_token_indices = (
                self._batch_processor.empty_image_token_indices()
            )

        assert model_inputs.kv_cache_inputs

        model_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *input_row_offsets,
            *image_embeddings,
            *image_token_indices,
            *model_inputs.signal_buffers,
            *model_inputs.kv_cache_inputs.flatten(),
        )

        if len(model_outputs) == 3:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                logits=model_outputs[0],
                next_token_logits=model_outputs[0],
            )
