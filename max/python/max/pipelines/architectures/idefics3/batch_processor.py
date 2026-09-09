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
"""Input batching for Idefics3 multimodal pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.buffer_utils import cast_dlpack_to
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessor,
    BatchProcessorRuntime,
    process_ragged_kv_outputs,
    ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelOutputs
from max.pipelines.lib.vision_batching import (
    VisionStacker,
    create_empty_image_embeddings_single,
    create_empty_image_token_indices_single,
)

if TYPE_CHECKING:
    from .model import Idefics3Inputs


class Idefics3BatchProcessor(
    BatchProcessor[TextAndVisionContext, "Idefics3Inputs"]
):
    """Ragged batching with optional vision inputs for Idefics3 models."""

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        self._stacker = VisionStacker()
        hf_config = runtime.pipeline_config.model.huggingface_config
        self._image_token_id: int = hf_config.image_token_id
        self._devices = list(runtime.devices)
        self._cached_empty_embeddings: Buffer | None = None
        self._cached_empty_indices: Buffer | None = None

    def empty_image_embeddings(self) -> Buffer:
        """Zero-row image embedding buffer for text-only language-model decode."""
        if self._cached_empty_embeddings is None:
            hf_config = self.runtime.pipeline_config.model.huggingface_config
            self._cached_empty_embeddings = (
                create_empty_image_embeddings_single(
                    self._devices[0],
                    hf_config.text_config.hidden_size,
                )
            )
        return self._cached_empty_embeddings

    def empty_image_token_indices(self) -> Buffer:
        """Zero-length scatter indices for text-only decode."""
        if self._cached_empty_indices is None:
            self._cached_empty_indices = (
                create_empty_image_token_indices_single(self._devices[0])
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

    def _prepare_vision_inputs(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> Buffer | None:
        """Batch pixel_values from all contexts for the vision encoder."""
        images = []
        for context in context_batch:
            for img in context.next_images:
                images.append(img.pixel_values)
        if not images:
            return None
        final_images = self._stacker.stack(images)
        return cast_dlpack_to(
            final_images, DType.float32, DType.bfloat16, self._devices[0]
        )

    def _batch_image_token_indices(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> Buffer | None:
        """Collect flat image-token positions across the batched sequence."""
        indices_and_offsets = []
        batch_offset = 0
        for ctx in context_batch:
            input_ids = ctx.tokens.active
            special_image_token_mask = input_ids == self._image_token_id
            indices = np.where(special_image_token_mask)[0].tolist()
            indices_and_offsets.append([idx + batch_offset for idx in indices])
            batch_offset += ctx.tokens.active_length
        if not indices_and_offsets:
            return None
        np_indices = np.concatenate(indices_and_offsets).astype(
            np.int32, copy=False
        )
        return Buffer.from_numpy(np_indices).to(self._devices[0])

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Idefics3Inputs:
        from .model import Idefics3Inputs

        if len(replica_batches) > 1:
            raise ValueError("Idefics3BatchProcessor does not support DP>1")

        context_batch = replica_batches[0]
        device0 = self._devices[0]

        pixel_values = self._prepare_vision_inputs(context_batch)
        image_token_indices = self._batch_image_token_indices(context_batch)

        input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(device0)

        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])
        input_ids = Buffer.from_numpy(tokens).to(device0)

        return Idefics3Inputs(
            tokens=input_ids,
            input_row_offsets=input_row_offsets,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            pixel_values=pixel_values,
            kv_cache_inputs=kv_cache_inputs,
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
