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
"""Input batching for InternVL pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
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
    create_empty_image_embeddings,
    create_empty_image_token_indices,
)

from .tokenizer import IMAGE_NDIMS

if TYPE_CHECKING:
    from .model import InternVLInputs


class InternVLBatchProcessor(
    BatchProcessor[TextAndVisionContext, "InternVLInputs"]
):
    """Ragged batching with optional vision inputs for InternVL models."""

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        self._stacker = VisionStacker()
        self._devices = list(runtime.devices)
        self._signal_buffers = list(runtime.signal_buffers)
        self._cached_empty_embeddings: list[Buffer] | None = None
        self._cached_empty_indices: list[Buffer] | None = None

    def empty_image_embeddings(self) -> list[Buffer]:
        """Per-device zero-row embeddings for text-only language-model decode."""
        if self._cached_empty_embeddings is None:
            hf_config = self.runtime.pipeline_config.model.huggingface_config
            self._cached_empty_embeddings = create_empty_image_embeddings(
                self._devices,
                hf_config.llm_config.hidden_size,
            )
        return self._cached_empty_embeddings

    def empty_image_token_indices(self) -> list[Buffer]:
        """Per-device zero-length scatter indices for text-only decode."""
        if self._cached_empty_indices is None:
            self._cached_empty_indices = create_empty_image_token_indices(
                self._devices
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
            include_signal_buffers=len(device_refs) > 1,
        )

    def _prepare_vision_inputs(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> list[Buffer] | None:
        """Batch pixel values from contexts that need vision encoding."""
        images = []
        for context in context_batch:
            if context.needs_vision_encoding:
                next_images = context.next_images
                if len(next_images) != 1:
                    raise ValueError(
                        "InternVL only supports one image per request"
                    )
                image = next_images[0].pixel_values
                if len(image.shape) != IMAGE_NDIMS:
                    raise ValueError(
                        "InternVL vision model expects image shape to be "
                        "[num_patches, height_patches, width_patches, "
                        "channels, patch_size, patch_size]"
                    )
                for patch_group in image:
                    images.append(patch_group)
        if not images:
            return None
        final_images = self._stacker.stack(images)
        tensor = Buffer.from_numpy(final_images)
        if final_images.dtype == np.uint16:
            tensor = tensor.view(DType.bfloat16, tensor.shape)
        return [tensor.to(dev) for dev in self._devices]

    def _batch_image_token_indices(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> list[Buffer] | None:
        """Collect image-token indices from context extra_model_args."""
        indices_and_offsets = []
        batch_offset = 0
        for ctx in context_batch:
            if "image_token_indices" in ctx.extra_model_args:
                indices = ctx.extra_model_args["image_token_indices"]
                indices_and_offsets.append(indices + batch_offset)
            batch_offset += ctx.tokens.active_length
        if not indices_and_offsets:
            return None
        np_indices = np.concatenate(indices_and_offsets).astype(
            np.int32, copy=False
        )
        return [Buffer.from_numpy(np_indices).to(dev) for dev in self._devices]

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InternVLInputs:
        from .model import InternVLInputs

        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]

        pixel_values = self._prepare_vision_inputs(context_batch)

        input_row_offsets_host = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            ),
        )
        input_row_offsets = [
            input_row_offsets_host.to(dev) for dev in self._devices
        ]

        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])
        input_ids = Buffer.from_numpy(tokens).to(self._devices[0])

        image_token_indices = self._batch_image_token_indices(context_batch)

        return InternVLInputs(
            tokens=input_ids,
            input_row_offsets=input_row_offsets,
            signal_buffers=self._signal_buffers,
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
