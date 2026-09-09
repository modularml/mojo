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
"""Input batching for Gemma3 multimodal ModuleV3 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.buffer_utils import cast_dlpack_to
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessor,
    modulev3_gemma_multimodal_language_symbolic_inputs,
    process_ragged_kv_outputs,
    ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelOutputs
from max.pipelines.lib.vision_batching import (
    VisionStacker,
    create_empty_image_embeddings_single,
    create_empty_image_token_indices_single,
)

from .model_config import Gemma3ForConditionalGenerationConfig

if TYPE_CHECKING:
    from .model import Gemma3MultiModalModelInputs


class Gemma3MultiModalModuleV3BatchProcessor(
    BatchProcessor[TextAndVisionContext, "Gemma3MultiModalModelInputs"]
):
    """Ragged batching with vision support for Gemma3 multimodal ModuleV3 models."""

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        runtime: Any,
    ) -> None:
        super().__init__(config, runtime)
        self._gemma_config = config
        self._stacker = VisionStacker()
        self._cached_empty_embeddings: Buffer | None = None
        self._cached_empty_indices: Buffer | None = None

    def empty_image_embeddings(self) -> Buffer:
        """Zero-row image embedding buffer for text-only language-model decode."""
        if self._cached_empty_embeddings is None:
            hf_config = self.runtime.pipeline_config.model.huggingface_config
            self._cached_empty_embeddings = (
                create_empty_image_embeddings_single(
                    self.runtime.devices[0],
                    hf_config.text_config.hidden_size,
                )
            )
        return self._cached_empty_embeddings

    def empty_image_token_indices(self) -> Buffer:
        """Zero-length scatter indices for text-only decode."""
        if self._cached_empty_indices is None:
            self._cached_empty_indices = (
                create_empty_image_token_indices_single(self.runtime.devices[0])
            )
        return self._cached_empty_indices

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        return ragged_kv_symbolic_inputs(
            kv_params=kv_params,
            device_refs=device_refs,
            include_signal_buffers=False,
        )

    def get_language_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_ref: DeviceRef,
        hidden_size: int,
    ) -> list[TensorType | BufferType]:
        """Symbolic inputs for the ModuleV3 language-model ``compile()`` call."""
        return modulev3_gemma_multimodal_language_symbolic_inputs(
            kv_params=kv_params,
            device_ref=device_ref,
            hidden_size=hidden_size,
        )

    def _prepare_vision_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
        device0: Device,
    ) -> Buffer | None:
        """Batch pixel_values for vision processing."""
        images = []
        for context in context_batch:
            for img in context.next_images:
                images.append(img.pixel_values)

        if not images:
            return None

        final_images = self._stacker.stack(images)
        return cast_dlpack_to(
            final_images, DType.float32, DType.bfloat16, device0
        )

    def _batch_image_token_indices(
        self,
        context_batch: Sequence[TextAndVisionContext],
        device0: Device,
    ) -> Buffer:
        """Batch image token indices from multiple contexts."""
        indices_and_offsets = []
        batch_offset = 0

        for ctx in context_batch:
            input_ids = ctx.tokens.active
            special_image_token_mask = (
                input_ids == self._gemma_config.image_token_index
            )
            indices = np.where(special_image_token_mask)[0]

            if len(indices) > 0:
                indices_and_offsets.append(indices + batch_offset)

            batch_offset += ctx.tokens.active_length

        if not indices_and_offsets:
            return Buffer.zeros(shape=[0], dtype=DType.int32).to(device0)

        np_indices = np.concatenate(indices_and_offsets).astype(
            np.int32, copy=False
        )
        return Buffer.from_numpy(np_indices).to(device0)

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Gemma3MultiModalModelInputs:
        from .model import Gemma3MultiModalModelInputs

        if len(replica_batches) > 1:
            raise ValueError(
                "Gemma3MultiModalModuleV3BatchProcessor does not support DP>1"
            )

        context_batch = replica_batches[0]
        device0 = self.runtime.devices[0]
        assert kv_cache_inputs is not None

        input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(device0)

        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])
        pixel_values = self._prepare_vision_inputs(context_batch, device0)
        image_token_indices = self._batch_image_token_indices(
            context_batch, device0
        )

        return Gemma3MultiModalModelInputs(
            tokens=Buffer.from_numpy(tokens).to(device0),
            input_row_offsets=input_row_offsets,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            pixel_values=pixel_values,
            image_token_indices=image_token_indices,
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )
