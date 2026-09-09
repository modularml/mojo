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
"""Implements the Kimi LM."""

from __future__ import annotations

from typing import Any

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings

from ...deepseekV3.deepseekV3 import (
    DeepseekV3,
)


class KimiK2_5MoEDecoder(DeepseekV3):
    subgraph_layer_prefix: str = "language_model.layers"

    def __call__(  # type: ignore[override]
        self,
        tokens: TensorValue,
        image_embeddings: list[TensorValue],
        image_token_indices: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        ep_inputs: list[Value[Any]] | None = None,
        eplb_counter_buffers_per_layer: list[list[BufferValue]] | None = None,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens, signal_buffers)

        # Merge image embeddings into text embeddings
        h = [
            merge_multimodal_embeddings(
                inputs_embeds=h_device,
                multimodal_embeddings=image_embeddings_device,
                image_token_indices=image_token_indices_device,
            )
            for h_device, image_embeddings_device, image_token_indices_device in zip(
                h,
                image_embeddings,
                image_token_indices,
                strict=True,
            )
        ]

        input_row_offsets_per_dev = list(
            ops.distributed_broadcast(input_row_offsets, signal_buffers)
        )
        return self._process_hidden_states(
            h,
            signal_buffers,
            kv_collections,
            return_n_logits,
            input_row_offsets_per_dev,
            host_input_row_offsets,
            data_parallel_splits,
            batch_context_lengths,
            ep_inputs,
            eplb_counter_buffers_per_layer,
        )

    def input_types(
        self,
        kv_params: KVCacheParamInterface,
    ) -> tuple[TensorType | BufferType, ...]:
        all_input_types: tuple[TensorType | BufferType, ...] = (
            super().input_types(kv_params)
        )

        image_embeddings_types = [
            TensorType(
                DType.bfloat16,
                shape=[
                    "vision_merged_seq_len",
                    self.config.hidden_size,
                ],
                device=DeviceRef.from_device(device),
            )
            for device in self.config.devices
        ]

        image_token_indices_types = [
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(device),
            )
            for device in self.config.devices
        ]
        tokens_type, remaining_input_types = (
            all_input_types[0],
            all_input_types[1:],
        )
        return (
            tokens_type,
            *image_embeddings_types,
            *image_token_indices_types,
        ) + remaining_input_types
