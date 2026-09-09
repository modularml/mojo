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
"""Input batching for Pixtral pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer
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
    create_empty_image_embeddings_single,
    create_empty_image_token_indices_single,
)

if TYPE_CHECKING:
    from .model import PixtralInputs


class PixtralBatchProcessor(
    BatchProcessor[TextAndVisionContext, "PixtralInputs"]
):
    """Ragged batching with vision patch extraction for Pixtral models."""

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
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

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> PixtralInputs:
        from .model import PixtralInputs

        if len(replica_batches) > 1:
            raise ValueError("PixtralBatchProcessor does not support DP>1")

        context_batch = replica_batches[0]
        device0 = self.runtime.devices[0]
        hf_config = self.runtime.pipeline_config.model.huggingface_config

        input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(device0)

        tokens = np.ascontiguousarray(
            np.concatenate([ctx.tokens.active for ctx in context_batch])
        )
        input_ids = Buffer.from_numpy(tokens).to(device0)

        patch_size = hf_config.vision_config.patch_size
        image_token_index = hf_config.image_token_index
        max_patches_per_side = hf_config.vision_config.image_size // patch_size

        all_patches: list[np.ndarray] = []
        all_position_ids: list[np.ndarray] = []
        patch_counts: list[int] = []
        indices_parts: list[np.ndarray] = []
        batch_offset = 0

        for ctx in context_batch:
            if ctx.needs_vision_encoding:
                for img_data in ctx.next_images:
                    image = np.ascontiguousarray(img_data.pixel_values)
                    C, H, W = image.shape
                    n_h = H // patch_size
                    n_w = W // patch_size
                    n_patches = n_h * n_w

                    # Extract patches: [C, H, W] -> [n_patches, C*p*p]
                    patches = image.reshape(C, n_h, patch_size, n_w, patch_size)
                    patches = patches.transpose(1, 3, 0, 2, 4)
                    patches = patches.reshape(
                        n_patches, C * patch_size * patch_size
                    )
                    all_patches.append(patches.astype(np.float32))

                    # Position IDs for 2D RoPE.
                    row_ids = np.repeat(np.arange(n_h), n_w)
                    col_ids = np.tile(np.arange(n_w), n_h)
                    pos_ids = row_ids * max_patches_per_side + col_ids
                    all_position_ids.append(pos_ids.astype(np.int64))
                    patch_counts.append(n_patches)

            # Find image token positions in this context's active tokens.
            active_tokens = ctx.tokens.active
            image_positions = np.where(active_tokens == image_token_index)[0]
            if len(image_positions) > 0:
                indices_parts.append(
                    (image_positions + batch_offset).astype(np.int32)
                )
            batch_offset += ctx.tokens.active_length

        pixel_patches: Buffer | None = None
        vision_attention_mask: Buffer | None = None
        vision_position_ids: Buffer | None = None
        image_token_indices: Buffer | None = None

        if all_patches:
            pixel_patches = Buffer.from_numpy(np.concatenate(all_patches)).to(
                device0
            )
            vision_position_ids = Buffer.from_numpy(
                np.concatenate(all_position_ids)
            ).to(device0)

            # Block-diagonal attention mask.
            # NOTE: This is a dense N x N mask where N = total patches across
            # all images. For multi-image batches this scales O(n^2) in memory
            # and should be replaced with a sparse or per-image scheme.
            total_patches = sum(patch_counts)
            # TODO(KERN-782): fill_val should be -inf but softmax saturates.
            fill_val = -10000.0
            mask = np.full(
                (1, 1, total_patches, total_patches),
                fill_val,
                dtype=np.float32,
            )
            offset = 0
            for count in patch_counts:
                mask[0, 0, offset : offset + count, offset : offset + count] = (
                    0.0
                )
                offset += count
            vision_attention_mask = Buffer.from_numpy(mask).to(device0)

        if indices_parts:
            image_token_indices = Buffer.from_numpy(
                np.concatenate(indices_parts)
            ).to(device0)

        return PixtralInputs(
            tokens=input_ids,
            input_row_offsets=input_row_offsets,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            pixel_patches=pixel_patches,
            vision_attention_mask=vision_attention_mask,
            vision_position_ids=vision_position_ids,
            image_token_indices=image_token_indices,
            kv_cache_inputs=kv_cache_inputs,
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )
