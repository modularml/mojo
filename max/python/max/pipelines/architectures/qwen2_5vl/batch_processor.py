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
"""Input batching for Qwen2.5VL pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.nn.parallel import ParallelArrayOps
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

from .context import Qwen2_5VLTextAndVisionContext, VisionEncodingData
from .model_config import Qwen2_5VLConfig
from .nn.data_processing import get_rope_index

if TYPE_CHECKING:
    from .model import Qwen2_5VLInputs


class Qwen2_5VLBatchProcessor(
    BatchProcessor[Qwen2_5VLTextAndVisionContext, "Qwen2_5VLInputs"]
):
    """Ragged batching with optional vision inputs for Qwen2.5VL models."""

    _parallel_ops: ParallelArrayOps
    _cached_empty_embeddings: list[Buffer] | None
    _cached_empty_indices: list[Buffer] | None

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        gpu0 = runtime.devices[0]
        if gpu0.is_host:
            raise ValueError("Qwen2.5VL currently only supports GPU devices")
        self._parallel_ops = ParallelArrayOps(accelerator=gpu0, max_workers=24)
        self._cached_empty_embeddings = None
        self._cached_empty_indices = None

    def empty_image_embeddings(self) -> list[Buffer]:
        """Returns per-device zero-length image embedding buffers for text-only decode."""
        if self._cached_empty_embeddings is None:
            hf_config = self.runtime.pipeline_config.model.huggingface_config
            quantization_encoding = _select_quantization_encoding(
                self.runtime.pipeline_config.model,
                Qwen2_5VLConfig.DEFAULT_ENCODING,
            )
            dtype = supported_encoding_dtype(quantization_encoding)
            self._cached_empty_embeddings = Buffer.zeros(
                shape=[0, hf_config.text_config.hidden_size],
                dtype=dtype,
            ).to(self.runtime.devices)
        return self._cached_empty_embeddings

    def empty_image_token_indices(self) -> list[Buffer]:
        """Returns per-device zero-length scatter indices for text-only decode."""
        if self._cached_empty_indices is None:
            self._cached_empty_indices = Buffer.zeros(
                shape=[0],
                dtype=DType.int32,
            ).to(self.runtime.devices)
        return self._cached_empty_indices

    def _batch_image_token_indices(
        self, context_batch: Sequence[Qwen2_5VLTextAndVisionContext]
    ) -> list[Buffer]:
        """Batch image token indices across contexts into per-device buffers.

        Args:
            context_batch: Contexts that may contain image token indices.

        Returns:
            Per-device buffers containing multimodal merge indices.
        """
        np_image_token_indices = compute_multimodal_merge_indices(context_batch)
        return Buffer.from_numpy(np_image_token_indices).to(
            self.runtime.devices
        )

    @staticmethod
    def prepare_decoder_position_ids(
        context_batch: Sequence[Qwen2_5VLTextAndVisionContext],
        devices: list[Device],
    ) -> Buffer:
        """Prepare decoder position IDs for a batch of contexts.

        This function computes position IDs for decoder tokens, handling three cases:
        1. Vision encoding with pre-computed position IDs (use stored values)
        2. Vision encoding requiring recomputation (after preemption)
        3. Text-only generation (simple arange with offset)

        Optimized implementation: pre-allocates output array and writes directly,
        avoiding concatenation overhead for better performance.

        Args:
            context_batch: Sequence of Qwen2.5VL contexts to process
            devices: List of devices to place the output tensor on

        Returns:
            Buffer containing decoder position IDs with shape [n_rope_sections, total_seq_len]
        """
        # Optimize concatenation: pre-allocate output array and write directly
        # Calculate total output size first
        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)
        n_rope_sections = 3  # Fixed for Qwen2.5VL

        # Fast path for single context: avoid concatenation overhead
        if len(context_batch) == 1:
            ctx = context_batch[0]
            ctx_decoder_position_ids = ctx.decoder_position_ids

            if ctx.needs_vision_encoding and ctx_decoder_position_ids.shape[
                1
            ] == len(ctx.tokens):
                result_array = ctx_decoder_position_ids[
                    :, ctx.tokens.processed_length : ctx.tokens.current_position
                ].astype(np.uint32, copy=False)
            elif ctx.needs_vision_encoding:
                # Recompute decoder_position_ids using get_rope_index
                spatial_merge_size = ctx.spatial_merge_size
                image_token_id = ctx.image_token_id
                video_token_id = ctx.video_token_id
                vision_start_token_id = ctx.vision_start_token_id
                tokens_per_second = ctx.tokens_per_second
                image_grid_thw = (
                    ctx.vision_data.image_grid_thw
                    if ctx.vision_data is not None
                    else None
                )

                attention_mask = np.ones((1, len(ctx.tokens)), dtype=np.float32)

                temp_position_ids, rope_delta_array = get_rope_index(
                    spatial_merge_size=spatial_merge_size,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                    vision_start_token_id=vision_start_token_id,
                    tokens_per_second=tokens_per_second,
                    input_ids=ctx.tokens[: len(ctx.tokens)].reshape(1, -1),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=None,
                    second_per_grid_ts=None,
                    attention_mask=attention_mask,
                )
                temp_position_ids = temp_position_ids.squeeze(1)

                ctx.rope_delta = int(rope_delta_array.item())

                result_array = temp_position_ids[
                    :, ctx.tokens.processed_length : ctx.tokens.current_position
                ].astype(np.uint32, copy=False)
            else:
                context_seq_length = ctx.tokens.active_length
                temp_position_ids = np.arange(context_seq_length)
                temp_position_ids = temp_position_ids.reshape(1, 1, -1)
                temp_position_ids = np.tile(temp_position_ids, (3, 1, 1))
                delta = ctx.tokens.processed_length + ctx.rope_delta
                temp_position_ids = temp_position_ids + delta
                result_array = temp_position_ids.squeeze(1).astype(
                    np.uint32, copy=False
                )

            decoder_position_ids = Buffer.from_numpy(result_array).to(
                devices[0]
            )
            return decoder_position_ids

        # Multi-context path: pre-allocate output and write directly
        out_array = np.empty(
            (n_rope_sections, total_seq_len), dtype=np.uint32, order="C"
        )

        write_offset = 0

        for ctx in context_batch:
            ctx_decoder_position_ids = ctx.decoder_position_ids
            active_len = ctx.tokens.active_length

            if ctx.needs_vision_encoding and ctx_decoder_position_ids.shape[
                1
            ] == len(ctx.tokens):
                src_slice = ctx_decoder_position_ids[
                    :, ctx.tokens.processed_length : ctx.tokens.current_position
                ]
                out_array[:, write_offset : write_offset + active_len] = (
                    src_slice.astype(np.uint32, copy=False)
                )
            elif ctx.needs_vision_encoding:
                spatial_merge_size = ctx.spatial_merge_size
                image_token_id = ctx.image_token_id
                video_token_id = ctx.video_token_id
                vision_start_token_id = ctx.vision_start_token_id
                tokens_per_second = ctx.tokens_per_second
                image_grid_thw = (
                    ctx.vision_data.image_grid_thw
                    if ctx.vision_data is not None
                    else None
                )

                attention_mask = np.ones((1, len(ctx.tokens)), dtype=np.float32)

                temp_position_ids, rope_delta_array = get_rope_index(
                    spatial_merge_size=spatial_merge_size,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                    vision_start_token_id=vision_start_token_id,
                    tokens_per_second=tokens_per_second,
                    input_ids=ctx.tokens[: len(ctx.tokens)].reshape(1, -1),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=None,
                    second_per_grid_ts=None,
                    attention_mask=attention_mask,
                )
                temp_position_ids = temp_position_ids.squeeze(1)

                ctx.rope_delta = int(rope_delta_array.item())

                src_slice = temp_position_ids[
                    :, ctx.tokens.processed_length : ctx.tokens.current_position
                ]
                out_array[:, write_offset : write_offset + active_len] = (
                    src_slice.astype(np.uint32, copy=False)
                )
            else:
                context_seq_length = ctx.tokens.active_length
                temp_position_ids = np.arange(context_seq_length)
                temp_position_ids = temp_position_ids.reshape(1, 1, -1)
                temp_position_ids = np.tile(temp_position_ids, (3, 1, 1))
                delta = ctx.tokens.processed_length + ctx.rope_delta
                temp_position_ids = temp_position_ids + delta
                temp_position_ids = temp_position_ids.squeeze(1)
                out_array[:, write_offset : write_offset + active_len] = (
                    temp_position_ids.astype(np.uint32, copy=False)
                )

            write_offset += active_len

        decoder_position_ids = Buffer.from_numpy(out_array).to(devices[0])

        return decoder_position_ids

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
        replica_batches: Sequence[Sequence[Qwen2_5VLTextAndVisionContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Qwen2_5VLInputs:
        """Prepares the initial inputs for the first execution pass of the Qwen2.5VL model."""
        from .model import Qwen2_5VLInputs

        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]

        if kv_cache_inputs is None:
            raise ValueError("KV Cache Inputs must be provided")

        devices = self.runtime.devices

        # Gather all vision data from contexts that need vision encoding
        vision_datas: list[VisionEncodingData] = []
        for ctx in context_batch:
            assert isinstance(ctx, Qwen2_5VLTextAndVisionContext), (
                f"Expected Qwen2_5VLTextAndVisionContext, got {type(ctx).__name__}"
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
            input_row_offsets = np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
            input_row_offsets_tensors = Buffer.from_numpy(input_row_offsets).to(
                devices
            )

        with Tracer("prepare_decoder_position_ids"):
            decoder_position_ids = self.prepare_decoder_position_ids(
                context_batch, devices
            )

        with Tracer("prepare_image_token_indices"):
            image_token_indices = self._batch_image_token_indices(context_batch)

        if not any_needs_vision_encoding:
            return Qwen2_5VLInputs(
                tokens=input_ids,
                input_row_offsets=input_row_offsets_tensors,
                position_ids=decoder_position_ids,
                signal_buffers=list(self.runtime.signal_buffers),
                return_n_logits=Buffer.from_numpy(
                    np.array([return_n_logits], dtype=np.int64)
                ),
                kv_cache_inputs=kv_cache_inputs,
                image_token_indices=image_token_indices,
                pixel_values=None,
                window_index=None,
                vision_position_ids=None,
                max_grid_size=None,
                cu_seqlens=None,
                cu_window_seqlens=None,
                max_seqlen=None,
                max_window_seqlen=None,
            )

        with Tracer("preparing_pixel_values"):
            pixel_values_list = [
                vision_data.concatenated_pixel_values
                for vision_data in vision_datas
            ]
            pixel_values_tensor = self._parallel_ops.concatenate(
                pixel_values_list
            )

            if pixel_values_tensor.dtype == DType.uint16:
                pixel_values_tensor = pixel_values_tensor.view(
                    DType.bfloat16, pixel_values_tensor.shape
                )

            pixel_values = pixel_values_tensor.to(devices)

        with Tracer("preparing_window_index"):
            window_index_parts: list[npt.NDArray[np.int64]] = []
            index_offset = 0
            for ctx in context_batch:
                if ctx.needs_vision_encoding:
                    assert ctx.vision_data is not None
                    per_ctx_index = ctx.vision_data.window_index.astype(
                        np.int64
                    )
                    window_index_parts.append(per_ctx_index + index_offset)
                    index_offset += int(per_ctx_index.shape[0])
            window_index_np = np.concatenate(window_index_parts, axis=0)
            window_index_tensor = Buffer.from_numpy(window_index_np)
            window_index = window_index_tensor.to(devices)

        with Tracer("preparing_vision_position_ids"):
            vision_position_ids_list = [
                vision_data.vision_position_ids.astype(np.int64)
                for vision_data in vision_datas
            ]
            vision_position_ids_tensor = self._parallel_ops.concatenate(
                vision_position_ids_list
            )

            vision_position_ids = vision_position_ids_tensor.to(devices)

        with Tracer("preparing_max_grid_size"):
            max_grid_size_value = max(
                vision_data.max_grid_size.item() for vision_data in vision_datas
            )
            max_grid_size_tensor = Buffer.from_numpy(
                np.array(max_grid_size_value, dtype=np.int32)
            )
            max_grid_size = [max_grid_size_tensor for _ in devices]

        with Tracer("preparing_cu_seqlens"):
            cu_seqlens_list = []
            offset = 0
            for vision_data in vision_datas:
                seqlens = vision_data.cu_seqlens
                adjusted = seqlens.copy()
                adjusted[1:] += offset
                cu_seqlens_list.append(adjusted[1:])
                offset = adjusted[-1]

            cu_seqlens_tensor = Buffer.from_numpy(
                np.concatenate(
                    [np.array([0], dtype=np.uint32), *cu_seqlens_list]
                ).astype(np.uint32)
            )
            cu_seqlens = cu_seqlens_tensor.to(devices)

        with Tracer("preparing_cu_window_seqlens"):
            cu_window_seqlens_parts: list[npt.NDArray[np.uint32]] = []
            offset = 0
            for vision_data in vision_datas:
                seqlens_unique = vision_data.cu_window_seqlens_unique.astype(
                    np.uint32
                )
                cu_window_seqlens_parts.append(
                    (seqlens_unique[1:] + offset).astype(np.uint32)
                )
                offset = offset + seqlens_unique[-1]

            cu_window_seqlens_np = np.concatenate(
                [np.array([0], dtype=np.uint32), *cu_window_seqlens_parts]
            ).astype(np.uint32)
            cu_window_seqlens_unique_tensor = Buffer.from_numpy(
                cu_window_seqlens_np
            )
            cu_window_seqlens = cu_window_seqlens_unique_tensor.to(devices)

        with Tracer("preparing_max_seqlen"):
            max_seqlen_value = max(
                vision_data.max_seqlen.item() for vision_data in vision_datas
            )
            max_seqlen_tensor = Buffer.from_numpy(
                np.array([max_seqlen_value], dtype=np.uint32)
            )
            max_seqlen = [max_seqlen_tensor for _ in devices]

        with Tracer("preparing_max_window_seqlen"):
            window_max_seqlen_value = max(
                vision_data.window_max_seqlen.item()
                for vision_data in vision_datas
            )
            window_max_seqlen_tensor = Buffer.from_numpy(
                np.array([window_max_seqlen_value], dtype=np.uint32)
            )
            max_window_seqlen = [window_max_seqlen_tensor for _ in devices]

        return Qwen2_5VLInputs(
            tokens=input_ids,
            input_row_offsets=input_row_offsets_tensors,
            signal_buffers=list(self.runtime.signal_buffers),
            position_ids=decoder_position_ids,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            image_token_indices=image_token_indices,
            pixel_values=pixel_values,
            window_index=window_index,
            vision_position_ids=vision_position_ids,
            max_grid_size=max_grid_size,
            cu_seqlens=cu_seqlens,
            cu_window_seqlens=cu_window_seqlens,
            max_seqlen=max_seqlen,
            max_window_seqlen=max_window_seqlen,
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )
