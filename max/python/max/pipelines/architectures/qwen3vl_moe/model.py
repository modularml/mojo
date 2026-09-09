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
from dataclasses import dataclass, field
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Type,
)
from max.graph import (
    Module as GraphModule,
)
from max.graph.buffer_utils import cast_tensors_to
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.nn.comm import Signals
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.layer import Module
from max.nn.transformer import ReturnLogits
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    MultiGraphPipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan

from .batch_processor import Qwen3VLMoeBatchProcessor
from .context import Qwen3VLTextAndVisionContext
from .model_config import Qwen3VLConfig
from .qwen3vl import Qwen3VL
from .weight_adapters import convert_qwen3vl_model_state_dict

logger = logging.getLogger("max.pipelines")


@dataclass
class Qwen3VLInputs(ModelInputs):
    """A class representing inputs for the Qwen3VL model.

    This class encapsulates the input tensors required for the Qwen3VL model execution,
    including both text and vision inputs. Vision inputs are optional and can be None
    for text-only processing.
    """

    tokens: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: list[Buffer]
    """Per-device tensors containing the offsets for each row in the ragged input sequence."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    decoder_position_ids: Buffer
    """3D RoPE position IDs for the decoder."""

    return_n_logits: Buffer
    """Number of logits to return, used by speculative decoding for example."""

    kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] = field(
        kw_only=True
    )
    """KV cache inputs for the model."""

    image_token_indices: list[Buffer] | None = None
    """Per-device pre-computed multimodal merge indices for the image embeddings.

    These are the locations of the image_token_id in the inputs fed to the model.

    Some indices may be negative, which means that they are ignored by the multimodal merge."""

    # Vision inputs.
    pixel_values: list[Buffer] | None = None
    """Pixel values for vision inputs."""

    vision_position_ids: list[Buffer] | None = None
    """Vision rotary position IDs per device."""

    weights: list[Buffer] | None = None
    """Bilinear interpolation weights for vision position embeddings per device."""

    indices: list[Buffer] | None = None
    """Bilinear interpolation indices for vision position embeddings per device."""

    max_grid_size: list[Buffer] | None = None
    """Maximum grid size for vision inputs per device."""

    cu_seqlens: list[Buffer] | None = None
    """Cumulative sequence lengths for full attention per device."""

    max_seqlen: list[Buffer] | None = None
    """Maximum sequence length for full attention for vision inputs per device."""

    grid_thw: list[Buffer] | None = None
    """Grid dimensions (temporal, height, width) for each image/video, shape (n_images, 3) per device."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


class Qwen3VLModel(
    AlwaysSignalBuffersMixin,
    MultiGraphPipelineModelWithKVCache[Qwen3VLTextAndVisionContext],
):
    """A Qwen3VL pipeline model for multimodal text generation."""

    model_config_cls: ClassVar[type[Any]] = Qwen3VLConfig
    batch_processor_cls: ClassVar[type[Qwen3VLMoeBatchProcessor]] = (
        Qwen3VLMoeBatchProcessor
    )

    vision_model: Model | None
    """The compiled vision model for processing images."""

    language_model: Model
    """The compiled language model for text generation."""

    model_config: Qwen3VLConfig | None
    """The Qwen3VL model configuration."""

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

        self.model_config = None
        self._session = session  # reuse for on-device casts

        self.vision_model, self.language_model = self.load_model(session)

    def _load_state_dict(self) -> dict[str, Any]:
        # Validate SafetensorWeights requirement
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "Qwen3VL currently only supports safetensors weights"
            )

        # Get processed state dict
        if self.adapter:
            model_state_dict = self.adapter(
                dict(self.weights.items()),
            )
        else:
            # Use the weight adapter to convert Qwen3VL checkpoint format
            model_state_dict = convert_qwen3vl_model_state_dict(
                dict(self.weights.items())
            )

        vision_state_dict: dict[str, WeightData] = {}
        llm_state_dict: dict[str, WeightData] = {}
        for key, value in model_state_dict.items():
            if key.startswith("vision_encoder."):
                vision_state_dict[key] = value
            elif key.startswith("language_model."):
                llm_state_dict[key] = value
            else:
                raise ValueError(
                    f"Key: {key} is not part of the vision or language model"
                )

        self._vision_weights_dict = vision_state_dict
        self._language_weights_dict = llm_state_dict
        return model_state_dict

    def _create_model_config(
        self, model_state_dict: dict[str, WeightData]
    ) -> Qwen3VLConfig:
        # Generate Qwen3VL config from HuggingFace config
        qwen3vl_config = Qwen3VLConfig.initialize_from_config(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            max_seq_len=self.max_seq_len,
        )
        qwen3vl_config.finalize(
            huggingface_config=self.huggingface_config,
            llm_state_dict=self._language_weights_dict,
            vision_state_dict=self._vision_weights_dict,
            return_logits=self.return_logits,
        )
        self.model_config = qwen3vl_config

        # Use the local non-optional variable to satisfy typing.
        self.model: Module = Qwen3VL(qwen3vl_config)
        self.model.load_state_dict(
            model_state_dict, weight_alignment=1, strict=True
        )
        return qwen3vl_config

    def _build_vision_graph(
        self,
        config: Qwen3VLConfig,
        state_dict: dict[str, WeightData],
        module: GraphModule,
    ) -> tuple[Graph, dict[str, WeightData]]:
        """Build the vision model graph for processing images."""
        del state_dict
        assert isinstance(config, Qwen3VLConfig)
        assert isinstance(self.model, Qwen3VL)
        vision_encoder = self.model.vision_encoder

        # Define vision graph input types - one per device
        pixel_values_types = [
            TensorType(
                DType.float32,
                shape=["vision_seq_len", vision_encoder.patch_embed.patch_dim],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        weights_types = [
            TensorType(
                DType.float32,
                shape=[4, "vision_seq_len", 1],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        indices_types = [
            TensorType(
                DType.int64,
                shape=[4, "vision_seq_len"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        vision_rot_pos_ids_types = [
            TensorType(
                DType.int32,
                shape=["vision_seq_len", 2],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        max_grid_size_types = [
            TensorType(
                DType.int32,
                shape=[],
                device=DeviceRef.CPU(),
            )
            for _ in self.devices
        ]

        grid_thw_types = [
            TensorType(
                DType.int64,
                shape=["n_images", 3],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        cu_seqlens_types = [
            TensorType(
                DType.uint32,
                shape=["n_seqlens"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        max_seqlen_types = [
            TensorType(
                DType.uint32,
                shape=[1],
                device=DeviceRef.CPU(),
            )
            for _ in self.devices
        ]

        # Create signal types for distributed communication
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        # Build the vision graph
        with Graph(
            "qwen3vl_vision",
            input_types=(
                *pixel_values_types,
                *weights_types,
                *indices_types,
                *vision_rot_pos_ids_types,
                *max_grid_size_types,
                *grid_thw_types,
                *cu_seqlens_types,
                *max_seqlen_types,
                *signals.input_types(),
            ),
            module=module,
        ) as graph:
            # Extract inputs
            all_inputs = graph.inputs
            n_devices = len(self.devices)

            pixel_values_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            weights_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            indices_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            rot_pos_ids_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            max_grid_size_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            grid_thw_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            cu_seqlens_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            max_seqlen_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            signal_buffers = [inp.buffer for inp in all_inputs]

            # Execute vision transformer
            image_embeddings, deepstack_image_embeddings = vision_encoder(
                pixel_values=pixel_values_list,
                idxs=indices_list,
                weights=weights_list,
                grid_thw=grid_thw_list,
                rot_pos_ids=rot_pos_ids_list,
                max_grid_size=max_grid_size_list,
                cu_seqlens=cu_seqlens_list,
                max_seqlen=max_seqlen_list,
                signal_buffers=signal_buffers,
            )
            # Ensure we have a valid output
            assert image_embeddings is not None, (
                "Vision encoder must return a valid output"
            )

            graph.output(
                *[
                    *image_embeddings,
                    *[
                        item
                        for sublist in deepstack_image_embeddings
                        for item in sublist
                    ],
                ]
            )

            return graph, self._vision_weights_dict

    def _language_graph_input_types(self) -> tuple[Type[Any], ...]:
        """Generate input types for the language model graph."""
        device_ref = DeviceRef.from_device(self.devices[0])

        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        kv_inputs = self.kv_params.get_symbolic_inputs()

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
        assert self.model_config is not None, "Model config must be initialized"
        n_deepstack_layers = len(
            self.model_config.vision_config.deepstack_visual_indexes
        )
        # shape: (vision_seq_len // (spatial_merge_size**2), out_hidden_size)
        image_embeddings_types = [
            TensorType(
                self.dtype,
                shape=[
                    "vision_merged_seq_len",
                    self.model_config.llm_config.hidden_size,
                ],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]
        n_deepstack_layers = len(
            self.model_config.vision_config.deepstack_visual_indexes
        )
        deepstack_embeddings_types = []
        for _ in range(n_deepstack_layers):
            for device in self.devices:
                deepstack_embeddings_types.append(
                    TensorType(
                        self.dtype,
                        shape=[
                            "vision_merged_seq_len",
                            self.model_config.llm_config.hidden_size,
                        ],
                        device=DeviceRef.from_device(device),
                    )
                )

        # Add image token indices type - one per device
        image_token_indices_types = [
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        # Add decoder position IDs type
        position_ids_type = TensorType(
            DType.int64,
            shape=[3, "total_seq_len"],
            device=DeviceRef.CPU(),
        )

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
            *deepstack_embeddings_types,
            *image_token_indices_types,
            position_ids_type,
            *signals.input_types(),
            *flattened_kv_types,
        )

    def _build_language_graph(
        self,
        config: Qwen3VLConfig,
        state_dict: dict[str, WeightData],
        module: GraphModule,
    ) -> tuple[Graph, dict[str, WeightData]]:
        """Build the language model graph for text generation with image embeddings."""
        del state_dict
        assert isinstance(config, Qwen3VLConfig)
        assert isinstance(self.model, Qwen3VL)
        language_model = self.model.language_model
        assert language_model is not None, "Language model must be initialized"

        # Get the number of deepstack layers
        num_deepstack_layers = len(
            config.vision_config.deepstack_visual_indexes
        )

        # Get input types from the helper method
        input_types = self._language_graph_input_types()

        with Graph(
            "qwen3vl_moe_language", input_types=input_types, module=module
        ) as graph:
            # Extract inputs
            (
                input_ids,
                return_n_logits,
                *variadic_args,
            ) = graph.inputs

            # Extract input_row_offsets (one per device)
            input_row_offsets = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract image embeddings (one per device)
            image_embeddings = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract deepstack visual embeddings (they come after KV cache inputs)
            # Structure: list[list[Buffer]] where outer is per layer, inner is per device
            deepstack_visual_embeds: list[list[TensorValue]] = []
            for layer_idx in range(num_deepstack_layers):
                layer_start = layer_idx * len(self.devices)
                layer_embeds = [
                    variadic_args[layer_start + device_idx].tensor
                    for device_idx in range(len(self.devices))
                ]
                deepstack_visual_embeds.append(layer_embeds)
            # Update variadic_args after extracting deepstack embeddings
            num_deepstack_inputs = num_deepstack_layers * len(self.devices)
            variadic_args = variadic_args[num_deepstack_inputs:]

            # Extract image token indices (one per device)
            image_token_indices = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract position_ids
            position_ids = variadic_args[0].tensor
            variadic_args = variadic_args[1:]

            # Extract signal buffers (one per device)
            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Calculate how many KV cache inputs there are
            kv_inputs = self.kv_params.get_symbolic_inputs()
            flattened_kv_types = kv_inputs.flatten()
            num_kv_inputs = len(flattened_kv_types)

            # Extract KV cache inputs (they come after signal buffers in the graph)
            kv_inputs_flat = variadic_args[:num_kv_inputs]
            kv_collections = self._unflatten_kv_inputs(kv_inputs_flat)
            variadic_args = variadic_args[num_kv_inputs:]

            # Execute language model: text + image embeddings -> logits
            outputs = language_model(
                tokens=input_ids.tensor,
                return_n_logits=return_n_logits.tensor,
                image_embeddings=image_embeddings,
                image_token_indices=image_token_indices,
                position_ids=position_ids,
                mrope_section=config.mrope_section,
                kv_collections=kv_collections,
                input_row_offsets=input_row_offsets,
                signal_buffers=signal_buffers,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )

            graph.output(*outputs)

        return graph, self._language_weights_dict

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Qwen3VL model with the prepared inputs."""
        assert isinstance(model_inputs, Qwen3VLInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Qwen3VL requires KV cache inputs"
        )

        # Process vision inputs if present
        image_embeddings: list[Buffer]
        deepstack_image_embeddings: list[Buffer]
        if model_inputs.has_vision_inputs:
            assert self.vision_model is not None
            assert model_inputs.image_token_indices is not None
            assert model_inputs.pixel_values is not None
            assert model_inputs.vision_position_ids is not None
            assert model_inputs.weights is not None
            assert model_inputs.indices is not None
            assert model_inputs.max_grid_size is not None
            assert model_inputs.cu_seqlens is not None
            assert model_inputs.max_seqlen is not None
            assert model_inputs.grid_thw is not None

            assert self.model_config is not None, (
                "Model config must be initialized"
            )
            n_deepstack_layers = len(
                self.model_config.vision_config.deepstack_visual_indexes
            )
            # Execute vision model: pixel_values -> image_embeddings
            vision_outputs = self.vision_model.execute(
                *model_inputs.pixel_values,
                *model_inputs.weights,
                *model_inputs.indices,
                *model_inputs.vision_position_ids,
                *model_inputs.max_grid_size,
                *model_inputs.grid_thw,
                *model_inputs.cu_seqlens,
                *model_inputs.max_seqlen,
                *model_inputs.signal_buffers,
            )
            assert len(vision_outputs) == len(self.devices) * (
                1 + n_deepstack_layers
            )
            # assert image embeddings and deepstack image embeddings have the same hidden size
            for output in vision_outputs:
                assert isinstance(output, Buffer)
                assert (
                    output.shape[1]
                    == self.huggingface_config.text_config.hidden_size
                )

            # Extract image embeddings (first len(self.devices) outputs)
            n_devices = len(self.devices)
            image_embeddings = [
                output
                for output in vision_outputs[:n_devices]
                if isinstance(output, Buffer)
            ]

            deepstack_image_embeddings = [
                output
                for output in vision_outputs[n_devices:]
                if isinstance(output, Buffer)
            ]

            image_embeddings = cast_tensors_to(
                image_embeddings, self.dtype, self._session
            )
            deepstack_image_embeddings = cast_tensors_to(
                deepstack_image_embeddings, self.dtype, self._session
            )

            image_token_indices = model_inputs.image_token_indices

            # The size of scatter indices must match the number of image embeddings.
            assert (
                image_token_indices[0].shape[0] == image_embeddings[0].shape[0]
            )

            # Normalize index dtypes to match the language graph contract.
            image_token_indices = cast_tensors_to(
                image_token_indices, DType.int32, self._session
            )
        else:
            assert isinstance(self._batch_processor, Qwen3VLMoeBatchProcessor)
            image_embeddings, deepstack_image_embeddings = (
                self._batch_processor.empty_image_embeddings()
            )
            image_token_indices = (
                self._batch_processor.empty_image_token_indices()
            )

        # Prepare KV cache inputs as list of tensors
        assert model_inputs.kv_cache_inputs
        kv_cache_inputs_list = list(model_inputs.kv_cache_inputs.flatten())

        # Execute language model with text and image embeddings and deepstack features
        # deepstack_image_embeddings Structure: [layer0_device0, layer0_device1, ..., layer1_device0, layer1_device1, ...]

        language_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *model_inputs.input_row_offsets,
            *image_embeddings,
            *deepstack_image_embeddings,
            *image_token_indices,
            model_inputs.decoder_position_ids,
            *model_inputs.signal_buffers,
            *kv_cache_inputs_list,
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
