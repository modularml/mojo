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

from max._core.engine import Model
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.graph import Module as GraphModule
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
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    MultiGraphPipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.profiler import traced

from .batch_processor import Qwen2_5VLBatchProcessor
from .model_config import Qwen2_5VLConfig
from .qwen2_5vl import Qwen2_5VL

logger = logging.getLogger("max.pipelines")


@dataclass
class Qwen2_5VLInputs(ModelInputs):
    """A class representing inputs for the Qwen2.5VL model.

    This class encapsulates the input tensors required for the Qwen2.5VL model execution,
    including both text and vision inputs. Vision inputs are optional and can be None
    for text-only processing."""

    tokens: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: list[Buffer]
    """Per-device tensors containing the offsets for each row in the ragged input sequence."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    position_ids: Buffer
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

    window_index: list[Buffer] | None = None
    """Window indices for vision attention mechanism."""

    vision_position_ids: list[Buffer] | None = None
    """1D RoPE position IDs for the visual inputs."""

    max_grid_size: list[Buffer] | None = None
    """Maximum grid size for vision inputs."""

    cu_seqlens: list[Buffer] | None = None
    """Cumulative sequence lengths for full attention."""

    cu_window_seqlens: list[Buffer] | None = None
    """Cumulative window sequence lengths for window attention."""

    max_seqlen: list[Buffer] | None = None
    """Maximum sequence length for full attention for vision inputs."""

    max_window_seqlen: list[Buffer] | None = None
    """Maximum sequence length for window attention for vision inputs."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


class Qwen2_5VLModel(
    AlwaysSignalBuffersMixin,
    MultiGraphPipelineModelWithKVCache[TextAndVisionContext],
):
    """A Qwen2.5VL pipeline model for multimodal text generation."""

    batch_processor_cls: ClassVar[type[Qwen2_5VLBatchProcessor]] = (
        Qwen2_5VLBatchProcessor
    )

    model_config_cls: ClassVar[type[Any]] = Qwen2_5VLConfig

    vision_model: Model | None
    """The compiled vision model for processing images."""

    language_model: Model
    """The compiled language model for text generation."""

    model_config: Qwen2_5VLConfig | None
    """The Qwen2.5VL model configuration."""

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

        gpu0 = devices[0]
        if gpu0.is_host:
            raise ValueError("Qwen2.5VL currently only supports GPU devices")

        self.vision_model, self.language_model = self.load_model(session)

    def _load_state_dict(self) -> dict[str, Any]:
        # Get LLM weights dictionary. Needed before model config generation
        # because we need to know if word embeddings are tied or not.
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "Qwen2.5VL currently only supports safetensors weights"
            )
        if self.adapter:
            model_state_dict = self.adapter(
                dict(self.weights.items()),
            )
        else:
            model_state_dict = {
                key: value.data() for key, value in self.weights.items()
            }
        # Get state dict for the vision encoder
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
    ) -> Qwen2_5VLConfig:
        # Generate Qwen2.5VL config from HuggingFace config
        qwen2_5vl_config = Qwen2_5VLConfig.initialize_from_config(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            max_seq_len=self.max_seq_len,
        )
        qwen2_5vl_config.finalize(
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
            llm_state_dict=self._language_weights_dict,
            vision_state_dict=self._vision_weights_dict,
            return_logits=self.return_logits,
        )
        self.model_config = qwen2_5vl_config

        assert self.model_config is not None, "Model config must be initialized"
        self.model: Module = Qwen2_5VL(self.model_config)
        self.model.load_state_dict(model_state_dict, strict=True)
        return qwen2_5vl_config

    @traced
    def load_model(
        self, session: InferenceSession
    ) -> tuple[Model | None, Model]:
        """Override: incompatible tower graph capture signature.

        ``_build_*`` is ``(module) -> Graph`` (not the base
        ``(config, state_dict, module) -> (Graph, registry)``), because
        graphs are captured from a pre-instantiated ``Qwen2_5VL`` module after
        ``load_state_dict`` in ``_create_model_config``. Registry keys come
        from the tower splits in ``_load_state_dict``, not from per-tower
        ``nn.state_dict()`` returns.
        """
        state_dict = self._load_state_dict()
        self._create_model_config(state_dict)

        with CompilationTimer("vision + language model") as timer:
            graph_module = GraphModule()

            vision_graph = self._build_vision_graph(module=graph_module)
            language_graph = self._build_language_graph(module=graph_module)
            timer.mark_build_complete()

            models = session.load_all(
                graph_module,
                weights_registry={
                    **self._vision_weights_dict,
                    **self._language_weights_dict,
                },
            )

        vision_model = models[vision_graph.name]
        language_model = models[language_graph.name]
        return vision_model, language_model

    def _build_vision_graph(  # type: ignore[override]
        self, module: GraphModule | None = None
    ) -> Graph:
        """Build the vision model graph for processing images.

        Now supports multi-GPU processing for the vision encoder.
        """

        # Create Qwen2.5VL model and vision encoder
        assert isinstance(self.model, Qwen2_5VL)
        vision_encoder = self.model.vision_encoder
        # Define vision graph input types - one per device
        # vision_seq_len is the number of patches in all images and videos in the request
        pixel_values_types = [
            TensorType(
                DType.bfloat16,
                shape=["vision_seq_len", vision_encoder.patch_embed.patch_dim],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        rot_pos_ids_types = [
            TensorType(
                DType.int64,
                shape=["vision_seq_len", 2],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        window_index_types = [
            TensorType(
                DType.int64,
                shape=["window_seq_len"],
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
            for device in self.devices
        ]

        # Create signal types for distributed communication
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        cu_seqlens_types = [
            TensorType(
                DType.uint32,
                shape=["n_seqlens"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        cu_window_seqlens_types = [
            TensorType(
                DType.uint32,
                shape=["n_window_seqlens"],
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

        max_window_seqlen_types = [
            TensorType(
                DType.uint32,
                shape=[1],
                device=DeviceRef.CPU(),
            )
            for _ in self.devices
        ]

        # Build the vision graph
        with Graph(
            "qwen2_5vl_vision",
            input_types=(
                *pixel_values_types,
                *rot_pos_ids_types,
                *window_index_types,
                *cu_seqlens_types,
                *cu_window_seqlens_types,
                *max_seqlen_types,
                *max_window_seqlen_types,
                *max_grid_size_types,
                *signals.input_types(),
            ),
            module=module,
        ) as graph:
            # Extract inputs
            all_inputs = graph.inputs
            n_devices = len(self.devices)

            pixel_values_list = [inp.tensor for inp in all_inputs[:n_devices]]
            rot_pos_ids_list = [
                inp.tensor for inp in all_inputs[n_devices : 2 * n_devices]
            ]
            window_index_list = [
                inp.tensor for inp in all_inputs[2 * n_devices : 3 * n_devices]
            ]
            cu_seqlens_list = [
                inp.tensor for inp in all_inputs[3 * n_devices : 4 * n_devices]
            ]
            cu_window_seqlens_list = [
                inp.tensor for inp in all_inputs[4 * n_devices : 5 * n_devices]
            ]
            max_seqlen_list = [
                inp.tensor for inp in all_inputs[5 * n_devices : 6 * n_devices]
            ]
            max_window_seqlen_list = [
                inp.tensor for inp in all_inputs[6 * n_devices : 7 * n_devices]
            ]
            max_grid_size_list = [
                inp.tensor for inp in all_inputs[7 * n_devices : 8 * n_devices]
            ]
            signal_buffers = [inp.buffer for inp in all_inputs[8 * n_devices :]]

            vision_outputs = vision_encoder(
                pixel_values=pixel_values_list,
                rot_pos_ids=rot_pos_ids_list,
                window_index=window_index_list,
                cu_seqlens=cu_seqlens_list,
                cu_window_seqlens=cu_window_seqlens_list,
                max_seqlen=max_seqlen_list,
                max_window_seqlen=max_window_seqlen_list,
                max_grid_size=max_grid_size_list,
                signal_buffers=signal_buffers,
            )

            # Ensure we have a valid output
            assert vision_outputs is not None, (
                "Vision encoder must return a valid output"
            )

            graph.output(*vision_outputs)

        return graph

    def _build_language_graph(  # type: ignore[override]
        self, module: GraphModule | None = None
    ) -> Graph:
        """Build the language model graph for text generation with image embeddings."""

        assert isinstance(self.model, Qwen2_5VL)
        language_model = self.model.language_model

        # Generate DeviceRef
        device_ref = DeviceRef.from_device(self.devices[0])

        input_ids_type = TensorType(
            DType.int64,
            shape=["total_seq_len"],
            device=device_ref,
        )
        return_n_logits_type = TensorType(
            DType.int64,
            shape=["return_n_logits"],
            device=DeviceRef.CPU(),
        )
        # Create input_row_offsets_type for each device
        input_row_offsets_types = [
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )
        assert self.model_config is not None, "Model config must be initialized"

        # Add image embeddings type - one per device, can be empty for text-only inputs
        image_embeddings_types = [
            TensorType(
                self.dtype,
                shape=[
                    "vision_seq_len",
                    self.model_config.llm_config.hidden_size,
                ],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        # Add image token indices type - one per device
        image_token_indices_types = [
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        position_ids_type = TensorType(
            DType.uint32,
            shape=[len(self.model_config.mrope_section), "total_seq_len"],
            device=device_ref,
        )

        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = kv_inputs.flatten()

        with Graph(
            "qwen2_5vl_language",
            input_types=(
                input_ids_type,
                return_n_logits_type,
                *input_row_offsets_types,
                *image_embeddings_types,
                *image_token_indices_types,
                position_ids_type,
                *signals.input_types(),
                *flattened_kv_types,
            ),
            module=module,
        ) as graph:
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

            # Unmarshal the remaining arguments, which are for KV cache.
            variadic_args = variadic_args[len(self.devices) :]
            kv_collections = self._unflatten_kv_inputs(variadic_args)

            # Execute language model: text + image embeddings -> logits
            outputs = language_model(
                tokens=input_ids.tensor,
                return_n_logits=return_n_logits.tensor,
                image_embeddings=image_embeddings,
                image_token_indices=image_token_indices,
                position_ids=position_ids,
                signal_buffers=signal_buffers,
                kv_collections=kv_collections,
                input_row_offsets=input_row_offsets,
                mrope_section=self.model_config.mrope_section,
            )

            graph.output(*outputs)

        return graph

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Qwen2.5VL model with the prepared inputs."""
        assert isinstance(model_inputs, Qwen2_5VLInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Qwen2.5VL requires KV cache inputs"
        )

        # Process vision inputs if present
        image_embeddings: list[Buffer]

        if model_inputs.has_vision_inputs:
            assert self.vision_model is not None
            assert model_inputs.image_token_indices is not None
            assert model_inputs.pixel_values is not None
            assert model_inputs.vision_position_ids is not None
            assert model_inputs.window_index is not None
            assert model_inputs.cu_seqlens is not None
            assert model_inputs.cu_window_seqlens is not None
            assert model_inputs.max_seqlen is not None
            assert model_inputs.max_window_seqlen is not None
            assert model_inputs.max_grid_size is not None

            # Execute vision model: pixel_values -> image_embeddings (multi-GPU)

            vision_outputs = self.vision_model.execute(
                *model_inputs.pixel_values,
                *model_inputs.vision_position_ids,
                *model_inputs.window_index,
                *model_inputs.cu_seqlens,
                *model_inputs.cu_window_seqlens,
                *model_inputs.max_seqlen,
                *model_inputs.max_window_seqlen,
                *model_inputs.max_grid_size,
                *model_inputs.signal_buffers,
            )

            # Extract image embeddings from vision outputs (one per device)
            assert len(vision_outputs) == len(self.devices)
            image_embeddings = [
                output
                for output in vision_outputs
                if isinstance(output, Buffer)
            ]
            image_embeddings = cast_tensors_to(
                image_embeddings, self.dtype, self._session
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
            assert isinstance(self._batch_processor, Qwen2_5VLBatchProcessor)
            image_embeddings = self._batch_processor.empty_image_embeddings()
            image_token_indices = (
                self._batch_processor.empty_image_token_indices()
            )

        # Execute language model with text and image embeddings
        language_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *model_inputs.input_row_offsets,
            *image_embeddings,
            *image_token_indices,
            model_inputs.position_ids,
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
