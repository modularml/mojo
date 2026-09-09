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
"""Input batching for Inkling pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from max.driver import Buffer, DevicePinnedBuffer
from max.dtype import DType
from max.engine import Model
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    SingleReplicaRaggedBatchProcessor,
    single_replica_context_batch,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelInputs
from max.pipelines.lib.vision_batching import (
    create_empty_image_embeddings_single,
    create_empty_image_token_indices_single,
)
from max.pipelines.lib.vlm_utils import compute_multimodal_merge_indices

from .model_config import InklingConfig
from .state_cache import InklingConvStateCache


@dataclass
class InklingInputs(ModelInputs):
    """Ragged token inputs plus the convolution-state pool addressing."""

    tokens: Buffer
    input_row_offsets: Buffer
    positions: Buffer
    return_n_logits: Buffer

    image_embeddings: Buffer
    image_indices: Buffer
    """Token-stream row each vision row replaces; negative entries are skipped."""

    signal_buffers: list[Buffer]
    slot_idx: list[Buffer]
    conv_pools: list[Buffer]
    """Per device, one pool per convolution site per layer, mutated in place."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        return (
            self.tokens,
            self.input_row_offsets,
            self.positions,
            self.return_n_logits,
            self.image_embeddings,
            self.image_indices,
            *self.signal_buffers,
            *self.kv_cache_inputs.flatten(),
            *self.slot_idx,
            *self.conv_pools,
        )


class InklingBatchProcessor(
    SingleReplicaRaggedBatchProcessor[TextAndVisionContext, InklingInputs]
):
    """Ragged batching with Inkling's convolution slots and vision operands."""

    def __init__(
        self, config: ArchConfig, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        assert isinstance(config, InklingConfig)
        self._hidden_size = config.text_config.hidden_size
        self._dtype = config.dtype
        self._state_cache: InklingConvStateCache | None = None
        self._vision_model: Model | None = None
        self._conv_pools: list[Buffer] = []
        self._signal_buffers = list(runtime.signal_buffers)
        self._return_n_logits_buffers: dict[int, Buffer] = {}
        self._no_images: tuple[Buffer, Buffer] | None = None

    def bind_runtime_state(
        self,
        state_cache: InklingConvStateCache | None,
        vision_model: Model,
    ) -> None:
        """Hands over what only exists once the model is compiled and loaded."""
        self._state_cache = state_cache
        self._vision_model = vision_model
        if state_cache is None:
            return
        self._conv_pools = [
            pool
            for device_idx in range(len(self.runtime.devices))
            for pool in state_cache.pools(device_idx)
        ]

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InklingInputs:
        context_batch = single_replica_context_batch(
            replica_batches, processor_name=type(self).__qualname__
        )
        state_cache = self._state_cache
        assert state_cache is not None

        request_ids = [context.request_id for context in context_batch]
        for request_id in request_ids:
            state_cache.claim(request_id)
        slot_idx = state_cache.slot_idx_for(request_ids)

        tokens, input_row_offsets, positions = self._stage_token_inputs(
            context_batch
        )

        # Reusing one buffer per distinct value keeps graph-capture replay
        # from recopying an input that never changes.
        return_n_logits_buffer = self._return_n_logits_buffers.get(
            return_n_logits
        )
        if return_n_logits_buffer is None:
            return_n_logits_buffer = Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            )
            self._return_n_logits_buffers[return_n_logits] = (
                return_n_logits_buffer
            )

        image_embeddings, image_indices = self._image_operands(context_batch)

        return InklingInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            positions=positions,
            return_n_logits=return_n_logits_buffer,
            image_embeddings=image_embeddings,
            image_indices=image_indices,
            signal_buffers=self._signal_buffers,
            slot_idx=slot_idx,
            conv_pools=self._conv_pools,
            kv_cache_inputs=kv_cache_inputs,
        )

    def _stage_token_inputs(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> tuple[Buffer, Buffer, Buffer]:
        # Host staging is fresh per step so overlap writes can't clobber an
        # in-flight copy; device destinations are reused for graph replay.
        device0 = self.runtime.devices[0]
        lengths = np.fromiter(
            (context.tokens.active_length for context in context_batch),
            dtype=np.int64,
            count=len(context_batch),
        )
        total_seq_len = int(lengths.sum())
        pinned = not device0.is_host
        host_buffer_cls: type[Buffer | DevicePinnedBuffer] = (
            DevicePinnedBuffer if pinned else Buffer
        )
        host_tokens: Buffer = host_buffer_cls(
            dtype=DType.int64, shape=(total_seq_len,), device=device0
        )
        host_row_offsets: Buffer = host_buffer_cls(
            dtype=DType.uint32, shape=(len(context_batch) + 1,), device=device0
        )
        host_positions: Buffer = host_buffer_cls(
            dtype=DType.uint32, shape=(total_seq_len,), device=device0
        )

        offsets = np.cumsum([0, *lengths], dtype=np.int64)
        host_row_offsets.to_numpy()[:] = offsets
        if total_seq_len:
            np.concatenate(
                [context.tokens.active for context in context_batch],
                out=host_tokens.to_numpy(),
            )
        # One ramp shifted per sequence covers the whole ragged batch.
        first_positions = (
            np.fromiter(
                (context.tokens.current_position for context in context_batch),
                dtype=np.int64,
                count=len(context_batch),
            )
            - lengths
        )
        host_positions.to_numpy()[:] = np.arange(
            total_seq_len, dtype=np.int64
        ) + np.repeat(first_positions - offsets[:-1], lengths)

        if not pinned:
            return host_tokens, host_row_offsets, host_positions

        staged = []
        for name, host in (
            ("ragged_input_tokens", host_tokens),
            ("ragged_input_row_offsets", host_row_offsets),
            ("token_positions", host_positions),
        ):
            device_buffer = self._device_input_allocator.alloc(
                name=name,
                dtype=host.dtype,
                shape=tuple(host.shape),
                device=device0,
            )
            device_buffer.inplace_copy_from(host)
            staged.append(device_buffer)
        return staged[0], staged[1], staged[2]

    def _empty_image_operands(self) -> tuple[Buffer, Buffer]:
        """Zero-row operands for a batch with no images to encode."""
        if self._no_images is None:
            device0 = self.runtime.devices[0]
            self._no_images = (
                create_empty_image_embeddings_single(
                    device0, self._hidden_size, self._dtype
                ),
                create_empty_image_token_indices_single(device0),
            )
        return self._no_images

    def _image_operands(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> tuple[Buffer, Buffer]:
        """Vision-tower rows for this batch, and the rows they replace."""
        # The whole batch feeds the helper, so the row offsets it returns cover
        # the ragged token stream. A graph-capture warmup probe batches bare
        # TextContexts, which is why the attribute is read defensively.
        indices = compute_multimodal_merge_indices(context_batch)
        if indices.size == 0:
            return self._empty_image_operands()
        blocks = [
            image.pixel_values
            for context in context_batch
            if getattr(context, "needs_vision_encoding", False)
            for image in context.images
        ]
        assert self._vision_model is not None
        device0 = self.runtime.devices[0]
        embeddings = self._vision_model.execute(
            Buffer.from_numpy(np.concatenate(blocks)).to(device0)
        )[0]
        assert isinstance(embeddings, Buffer)
        return embeddings, Buffer.from_numpy(indices).to(device0)
