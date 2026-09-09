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
"""Input batching for Qwen3VL MoE pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessor,
    BatchProcessorRuntime,
    process_ragged_kv_outputs,
    ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelOutputs
from max.pipelines.lib.vlm_utils import compute_multimodal_merge_indices
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from max.profiler import Tracer

from .context import Qwen3VLTextAndVisionContext, VisionEncodingData
from .model_config import Qwen3VLConfig

if TYPE_CHECKING:
    from .model import Qwen3VLInputs


class Qwen3VLMoeBatchProcessor(
    BatchProcessor[Qwen3VLTextAndVisionContext, "Qwen3VLInputs"]
):
    """Ragged batching with optional vision inputs for Qwen3VL MoE models."""

    _cached_empty_embeddings: tuple[list[Buffer], list[Buffer]] | None
    _cached_empty_indices: list[Buffer] | None

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        self._cached_empty_embeddings = None
        self._cached_empty_indices = None

    def empty_image_embeddings(self) -> tuple[list[Buffer], list[Buffer]]:
        """Returns empty image and deepstack embedding buffers for text-only decode."""
        if self._cached_empty_embeddings is None:
            hf_config = self.runtime.pipeline_config.model.huggingface_config
            quantization_encoding = _select_quantization_encoding(
                self.runtime.pipeline_config.model,
                Qwen3VLConfig.DEFAULT_ENCODING,
            )
            dtype = supported_encoding_dtype(quantization_encoding)
            hidden_size = hf_config.text_config.hidden_size
            n_deepstack_layers = len(
                hf_config.vision_config.deepstack_visual_indexes
            )
            image_embeddings = Buffer.zeros(
                shape=[0, hidden_size],
                dtype=dtype,
            ).to(self.runtime.devices)
            deepstack_image_embeddings = [
                tensor
                for _ in range(n_deepstack_layers)
                for tensor in Buffer.zeros(
                    shape=[0, hidden_size],
                    dtype=dtype,
                ).to(self.runtime.devices)
            ]
            self._cached_empty_embeddings = (
                image_embeddings,
                deepstack_image_embeddings,
            )
        return self._cached_empty_embeddings

    def empty_image_token_indices(self) -> list[Buffer]:
        """Returns per-device zero-length scatter indices for text-only decode."""
        if self._cached_empty_indices is None:
            self._cached_empty_indices = Buffer.zeros(
                shape=[0],
                dtype=DType.int32,
            ).to(self.runtime.devices)
        return self._cached_empty_indices

    def _mrope_section_length(self) -> int:
        hf_config = self.runtime.pipeline_config.model.huggingface_config
        rope_scaling = (
            getattr(hf_config.text_config, "rope_scaling", None) or {}
        )
        mrope_section = rope_scaling.get("mrope_section")
        if mrope_section is None:
            return 3
        return len(mrope_section)

    def _batch_image_token_indices(
        self, context_batch: Sequence[Qwen3VLTextAndVisionContext]
    ) -> list[Buffer]:
        """Batch image token indices across contexts into a per-device buffer list.

        Args:
            context_batch: Contexts that may contain image token indices.

        Returns:
            Per-device buffers containing multimodal merge indices.
        """
        np_image_token_indices = compute_multimodal_merge_indices(context_batch)
        return Buffer.from_numpy(np_image_token_indices).to(
            self.runtime.devices
        )

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        return ragged_kv_symbolic_inputs(
            kv_params=kv_params,
            device_refs=device_refs,
            include_signal_buffers=True,
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[Qwen3VLTextAndVisionContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Qwen3VLInputs:
        """Prepares the initial inputs for the first execution pass of the Qwen3VL model."""
        from .model import Qwen3VLInputs

        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]

        if kv_cache_inputs is None:
            raise ValueError("KV Cache Inputs must be provided")

        devices = self.runtime.devices
        n_rope_sections = self._mrope_section_length()

        # Gather all vision data from contexts that need vision encoding
        vision_datas: list[VisionEncodingData] = []
        for ctx in context_batch:
            assert isinstance(ctx, Qwen3VLTextAndVisionContext), (
                f"Expected Qwen3VLTextAndVisionContext, got {type(ctx).__name__}"
            )
            if ctx.needs_vision_encoding:
                assert ctx.vision_data is not None, (
                    "vision_data must be present when needs_vision_encoding is True"
                )
                vision_datas.append(ctx.vision_data)
        any_needs_vision_encoding = len(vision_datas) > 0

        # Prepare Inputs Needed Regardless of Images
        with Tracer("prepare_input_ids"):
            input_ids = Buffer.from_numpy(
                np.concatenate([ctx.tokens.active for ctx in context_batch])
            ).to(devices[0])

        with Tracer("prepare_input_row_offsets"):
            input_row_offsets_host = Buffer.from_numpy(
                np.cumsum(
                    [0] + [ctx.tokens.active_length for ctx in context_batch],
                    dtype=np.uint32,
                ),
            )
            input_row_offsets = [
                input_row_offsets_host.to(dev) for dev in devices
            ]

        with Tracer("prepare_decoder_position_ids"):
            decoder_position_ids_list = []
            for ctx in context_batch:
                ctx_decoder_position_ids = ctx.decoder_position_ids
                if ctx.needs_vision_encoding and ctx_decoder_position_ids.shape[
                    1
                ] == len(ctx.tokens):
                    decoder_position_ids_list.append(
                        ctx_decoder_position_ids[
                            :,
                            ctx.tokens.processed_length : ctx.tokens.current_position,
                        ]
                    )
                else:
                    context_seq_length = ctx.tokens.active_length
                    temp_pos_ids = np.tile(
                        np.arange(context_seq_length).reshape(1, 1, -1),
                        (n_rope_sections, 1, 1),
                    )
                    delta = ctx.tokens.processed_length + ctx.rope_delta
                    temp_position_ids = (temp_pos_ids + delta).squeeze(1)
                    decoder_position_ids_list.append(temp_position_ids)

            decoder_position_ids = Buffer.from_numpy(
                np.concatenate(decoder_position_ids_list, axis=1).astype(
                    np.int64
                )
            )

        # Batch image token indices
        with Tracer("prepare_image_token_indices"):
            image_token_indices = self._batch_image_token_indices(context_batch)

        if not any_needs_vision_encoding:
            return Qwen3VLInputs(
                tokens=input_ids,
                input_row_offsets=input_row_offsets,
                signal_buffers=list(self.runtime.signal_buffers),
                decoder_position_ids=decoder_position_ids,
                return_n_logits=Buffer.from_numpy(
                    np.array([return_n_logits], dtype=np.int64)
                ),
                kv_cache_inputs=kv_cache_inputs,
                image_token_indices=image_token_indices,
                pixel_values=None,
                vision_position_ids=None,
                weights=None,
                indices=None,
                max_grid_size=None,
                cu_seqlens=None,
                max_seqlen=None,
                grid_thw=None,
            )

        # Prepare vision inputs
        pixel_values_list = [
            vision_data.concatenated_pixel_values
            for vision_data in vision_datas
        ]
        pixel_values = Buffer.from_numpy(
            np.concatenate(pixel_values_list).astype(np.float32)
        ).to(devices)

        weights = Buffer.from_numpy(
            np.concatenate(
                [vision_data.weights for vision_data in vision_datas], axis=1
            ).astype(np.float32)
        ).to(devices)

        indices = Buffer.from_numpy(
            np.concatenate(
                [vision_data.indices for vision_data in vision_datas], axis=1
            )
        ).to(devices)

        vision_position_ids_list = [
            vision_data.vision_position_ids for vision_data in vision_datas
        ]
        vision_position_ids = Buffer.from_numpy(
            np.concatenate(vision_position_ids_list).astype(np.int32)
        ).to(devices)

        grid_thw_list = [
            vision_data.image_grid_thw for vision_data in vision_datas
        ]
        grid_thw = Buffer.from_numpy(
            np.concatenate(grid_thw_list).astype(np.int64)
        ).to(devices)

        max_grid_size_value = max(
            vision_data.max_grid_size.item() for vision_data in vision_datas
        )
        max_grid_size = [
            Buffer.from_numpy(np.array(max_grid_size_value, dtype=np.int32))
            for _ in devices
        ]

        cu_seqlens_list = []
        offset = 0
        for vision_data in vision_datas:
            seqlens = vision_data.cu_seqlens
            adjusted = seqlens.copy()
            adjusted[1:] += offset
            cu_seqlens_list.append(adjusted[1:])
            offset = adjusted[-1]

        cu_seqlens = Buffer.from_numpy(
            np.concatenate(
                [np.array([0], dtype=np.uint32), *cu_seqlens_list]
            ).astype(np.uint32)
        ).to(devices)

        max_seqlen_value = max(
            vision_data.max_seqlen.item() for vision_data in vision_datas
        )
        max_seqlen = [
            Buffer.from_numpy(np.array([max_seqlen_value], dtype=np.uint32))
            for _ in devices
        ]

        return Qwen3VLInputs(
            tokens=input_ids,
            input_row_offsets=input_row_offsets,
            signal_buffers=list(self.runtime.signal_buffers),
            decoder_position_ids=decoder_position_ids,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            image_token_indices=image_token_indices,
            pixel_values=pixel_values,
            vision_position_ids=vision_position_ids,
            weights=weights,
            indices=indices,
            max_grid_size=max_grid_size,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            grid_thw=grid_thw,
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )
