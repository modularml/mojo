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
"""Input batching for the unified MTP Gemma4 pipeline model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer, DevicePinnedBuffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessor,
    BatchProcessorRuntime,
    ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelOutputs

from ..gemma4.context import Gemma4Context

if TYPE_CHECKING:
    from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
    from .model import UnifiedMTPGemma4Inputs


class UnifiedMTPGemma4BatchProcessor(
    BatchProcessor[Gemma4Context, "UnifiedMTPGemma4Inputs"]
):
    """Ragged batching with signal buffers + optional vision for unified MTP.

    Prepares :class:`UnifiedMTPGemma4Inputs` for each forward pass.
    ``draft_tokens`` and sampling buffers are left as ``None`` and filled
    in by the overlap pipeline after this method returns.
    """

    _config: Gemma4ForConditionalGenerationConfig | None = None

    def __init__(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)

    def bind_model_state(
        self,
        *,
        config: Gemma4ForConditionalGenerationConfig,
    ) -> None:
        """Wire the model config from ``load_model``."""
        self._config = config

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
        replica_batches: Sequence[Sequence[Gemma4Context]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> UnifiedMTPGemma4Inputs:
        """Prepare batch inputs for a UnifiedMTPGemma4 forward pass.

        Args:
            replica_batches: One inner list per DP replica containing the
                :class:`Gemma4Context` objects for that shard.
            kv_cache_inputs: Optional KV cache inputs (may be ``None`` during
                compilation warm-up).
            return_n_logits: Number of per-token logit rows to return.

        Returns:
            :class:`UnifiedMTPGemma4Inputs` with tokens, row offsets, signal
            buffers, optional image/video inputs, and ``draft_tokens=None``
            (set later by the overlap pipeline).
        """
        from .model import UnifiedMTPGemma4Inputs

        context_batch = [ctx for batch in replica_batches for ctx in batch]
        devices = self.runtime.devices
        device0 = devices[0]
        pinned = not device0.is_host

        batch_size = len(context_batch)
        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)

        buffer_type = DevicePinnedBuffer if pinned else Buffer
        host_tokens = buffer_type(
            dtype=DType.int64, shape=(total_seq_len,), device=device0
        )
        host_row_offsets = buffer_type(
            dtype=DType.uint32,
            shape=(batch_size + 1,),
            device=device0,
        )

        np.concatenate(
            [ctx.tokens.active for ctx in context_batch],
            out=host_tokens.to_numpy(),
        )
        device_tokens = host_tokens.to(device0)

        np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
            out=host_row_offsets.to_numpy(),
        )
        device_row_offsets = host_row_offsets.to(device0)

        host_input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        )

        return_n_logits_buf = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        data_parallel_splits = Buffer.from_numpy(
            np.array([0, batch_size], dtype=np.int64)
        )

        batch_context_lengths = [
            Buffer.zeros(shape=[1], dtype=DType.int32)
            for _ in range(len(self.runtime.devices))
        ]

        return UnifiedMTPGemma4Inputs(
            tokens=device_tokens,
            input_row_offsets=device_row_offsets,
            host_input_row_offsets=host_input_row_offsets,
            return_n_logits=return_n_logits_buf,
            data_parallel_splits=data_parallel_splits,
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
            batch_context_lengths=batch_context_lengths,
            draft_tokens=None,
            structured_output=self.runtime.pipeline_config.needs_bitmask_constraints,
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        assert isinstance(outputs[0], Buffer)
        return ModelOutputs(logits=outputs[0])
