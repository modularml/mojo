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
"""Gemma4 ModuleV3 pipeline model (text + vision)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar, cast

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.gemma4.batch_vision_inputs import (
    VisionRawInputs,
    create_empty_embeddings,
    pack_uncached_images,
)
from max.pipelines.architectures.gemma4.context import Gemma4Context
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)
from max.pipelines.context import ImageMetadata
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    ModuleV3MultiGraphPipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from transformers import AutoConfig

from .batch_processor import Gemma4ModuleV3BatchProcessor
from .gemma4 import Gemma4
from .inputs import Gemma4Inputs
from .vision_model.vision_model import Gemma4VisionModel
from .weight_adapters import (
    convert_language_state_dict_for_module,
    convert_vision_state_dict_for_module,
)


class Gemma4Model(
    LogProbabilitiesMixin,
    ModuleV3MultiGraphPipelineModelWithKVCache[Gemma4Context],
):
    """A Gemma4 pipeline model (ModuleV3): eager language + vision towers.

    Vision is driven by the pipeline's ``VisionEncoderCache`` through the
    :class:`SupportsVisionEncoding` protocol. Text-only checkpoints
    (``model_type`` ``gemma4_unified``, so ``vision_config is None``) compile
    the language tower only and take the empty-embeddings path.
    """

    model_config_cls: ClassVar[type[Any]] = Gemma4ForConditionalGenerationConfig
    batch_processor_cls: ClassVar[type[Gemma4ModuleV3BatchProcessor]] = (
        Gemma4ModuleV3BatchProcessor
    )

    language_model: Callable[..., Any]
    """The compiled eager language tower."""

    vision_model: Callable[..., Any] | None
    """The compiled eager vision tower, or ``None`` for text-only
    checkpoints."""

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

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return huggingface_config.text_config.num_hidden_layers

    def _hf_config_for_weights(self) -> AutoConfig | None:
        return self.huggingface_config.text_config

    def _load_state_dict(self) -> dict[str, Any]:
        weights_dict = dict(self.weights.items())
        self._language_weights_dict = convert_language_state_dict_for_module(
            weights_dict
        )
        self._vision_weights_dict = convert_vision_state_dict_for_module(
            weights_dict
        )
        return {k: v.data() for k, v in weights_dict.items()}

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> Gemma4ForConditionalGenerationConfig:
        model_config = (
            Gemma4ForConditionalGenerationConfig.initialize_from_config(
                self.pipeline_config,
                self.huggingface_config,
                max_seq_len=self.max_seq_len,
            )
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )
        self.config = model_config
        return model_config

    def _compile_vision_model(  # type: ignore[override]
        self,
        model_config: Gemma4ForConditionalGenerationConfig,
        state_dict: dict[str, Any],
    ) -> Callable[..., Any] | None:
        """Compiles the vision tower, or ``None`` when text-only.

        The override widens the base hook's return type: ``load_model``
        already types the vision tower as optional, hence the override
        ignore.

        The multi-graph base does not open the ``F.lazy()`` / ``default_dtype``
        build context that the single-graph base does, so each compile hook
        owns it.
        """
        if model_config.vision_config is None:
            return None

        device_ref = DeviceRef.from_device(self.devices[0])
        with F.lazy(), default_dtype(model_config.dtype):
            vision_nn = Gemma4VisionModel(model_config)
            vision_nn.to(self.devices[0])

        patch_dim = 3 * model_config.vision_config.patch_size**2
        input_types = (
            TensorType(
                model_config.unquantized_dtype,
                shape=["total_patches", patch_dim],
                device=device_ref,
            ),
            TensorType(
                DType.int32, shape=["total_patches", 2], device=device_ref
            ),
            TensorType(
                DType.uint32, shape=["num_images_plus_1"], device=device_ref
            ),
            TensorType(
                DType.int32,
                shape=["num_pooled_tokens", "max_pool_patches"],
                device=device_ref,
            ),
            TensorType(DType.uint32, shape=[], device=DeviceRef.CPU()),
        )
        return vision_nn.compile(*input_types, weights=state_dict)

    def _compile_language_model(
        self,
        model_config: Gemma4ForConditionalGenerationConfig,
        state_dict: dict[str, Any],
    ) -> Callable[..., Any]:
        n_devices = len(self.devices)
        mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))
        with F.lazy(), default_dtype(model_config.dtype):
            language_nn = Gemma4(model_config, self.kv_params, mesh)
            language_nn.to(mesh)

        assert isinstance(self._batch_processor, Gemma4ModuleV3BatchProcessor)
        input_types = self._batch_processor.get_language_symbolic_inputs(
            kv_params=self.kv_params,
            device_ref=DeviceRef.from_device(self.devices[0]),
            hidden_size=model_config.text_config.hidden_size,
            # Must match the vision tower's output dtype, which is also what
            # empty_vision_embeddings() allocates.
            embedding_dtype=model_config.unquantized_dtype,
        )
        return language_nn.compile(*input_types, weights=state_dict)

    # --- SupportsVisionEncoding ---

    def pack_vision_inputs(
        self,
        selection: Sequence[tuple[Gemma4Context, Sequence[ImageMetadata]]],
        devices: list[Device],
    ) -> VisionRawInputs | None:
        """Packs the pipeline-selected uncached image pixels to device."""
        assert self.config.vision_config is not None
        return pack_uncached_images(
            selection,
            devices,
            self.config.vision_config.pooling_kernel_size,
            self.config.unquantized_dtype,
        )

    def vision_execute(
        self,
        selection: Sequence[tuple[Gemma4Context, Sequence[ImageMetadata]]],
        devices: list[Device],
        packed: VisionRawInputs | None,
    ) -> VisionEncodeResult:
        """Runs the vision encoder on the pixels packed by
        :meth:`pack_vision_inputs`."""
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(self.devices)
            )
        assert self.vision_model is not None, (
            "This checkpoint is served text-only (no vision encoder is"
            " loaded); image and video inputs are not supported."
        )
        out = self.vision_model(
            packed.patches_flat[0],
            packed.pixel_position_ids[0],
            packed.cu_seqlens[0],
            packed.pool_gather_index[0],
            packed.max_seq_len,
        )
        first = out[0] if isinstance(out, (tuple, list)) else out
        return VisionEncodeResult(
            embeddings=[cast(Buffer, first.driver_tensor)]
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for cached / text-only batches.

        Cached: this is hit on every text-only / decode step, so it must not
        allocate per call.
        """
        if not hasattr(self, "_cached_empty_embeddings"):
            self._cached_empty_embeddings = create_empty_embeddings(
                devices,
                self.huggingface_config.text_config.hidden_size,
                self.config.unquantized_dtype,
            )
        return self._cached_empty_embeddings

    # --- execute ---

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Gemma4 language tower with the prepared inputs."""
        assert isinstance(model_inputs, Gemma4Inputs)
        kv_cache_inputs = model_inputs.kv_cache_inputs
        assert kv_cache_inputs is not None
        # The pipeline's VisionEncoderCache always finalizes these, falling
        # back to empty_vision_embeddings() for text-only / decode steps.
        assert len(model_inputs.vision_embeddings) == 1
        assert len(model_inputs.vision_scatter_indices) == 1

        model_outputs = self.language_model(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            model_inputs.input_row_offsets,
            model_inputs.vision_embeddings[0],
            model_inputs.vision_scatter_indices[0],
            *kv_cache_inputs.flatten(),
        )
        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
            )
        else:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
            )
