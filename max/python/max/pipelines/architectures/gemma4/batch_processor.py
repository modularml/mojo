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
"""Input batching for Gemma4 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer, DevicePinnedBuffer, copy_pinned_to_destinations
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessor,
    process_ragged_kv_outputs,
    ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelOutputs
from max.profiler import traced

from .context import Gemma4Context
from .model_config import Gemma4ForConditionalGenerationConfig

if TYPE_CHECKING:
    from .model import Gemma3MultiModalModelInputs


class Gemma4BatchProcessor(
    BatchProcessor[Gemma4Context, "Gemma3MultiModalModelInputs"]
):
    """Ragged batching with optional vision inputs for Gemma4 models."""

    _config: Gemma4ForConditionalGenerationConfig | None = None

    def bind_model_state(
        self,
        *,
        config: Gemma4ForConditionalGenerationConfig,
    ) -> None:
        """Wire the model config from ``load_model``.

        Images go through the pipeline-owned ``VisionEncoderCache``; this
        processor only builds tokens/offsets and video inputs.

        Args:
            config: Fully-initialised Gemma4 model configuration.
        """
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

    @traced
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[Gemma4Context]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Gemma3MultiModalModelInputs:
        """Prepare inputs for the first execution pass."""
        from .model import Gemma3MultiModalModelInputs

        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")
        context_batch = replica_batches[0]

        assert kv_cache_inputs is not None

        devices = self.runtime.devices
        dev = devices[0]
        pinned = not dev.is_host

        batch_size = len(context_batch)
        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)

        # Fresh pinned host staging every step (never reused) so the next
        # overlap step's host writes can't clobber the in-flight H2D copy.
        # Device buffers are cached and reused so captured graphs replay in
        # place.
        host_buffer_cls = DevicePinnedBuffer if pinned else Buffer
        host_tokens: Buffer = host_buffer_cls(
            dtype=DType.int64, shape=(total_seq_len,), device=dev
        )
        host_row_offsets: Buffer = host_buffer_cls(
            dtype=DType.uint32, shape=(batch_size + 1,), device=dev
        )

        device_tokens = self._device_input_allocator.alloc(
            name="ragged_input_tokens",
            dtype=DType.int64,
            shape=(total_seq_len,),
            device=dev,
        )
        device_row_offsets = [
            self._device_input_allocator.alloc(
                name="ragged_input_row_offsets",
                dtype=DType.uint32,
                shape=(batch_size + 1,),
                device=device,
            )
            for device in devices
        ]

        return_n_logits_buf = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        row_offsets_np = host_row_offsets.to_numpy()
        np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
            out=row_offsets_np,
        )

        tokens_np = host_tokens.to_numpy()
        if context_batch:
            np.concatenate(
                [ctx.tokens.active for ctx in context_batch],
                out=tokens_np,
            )

        copy_pinned_to_destinations(host_tokens, [device_tokens])
        copy_pinned_to_destinations(host_row_offsets, device_row_offsets)

        return Gemma3MultiModalModelInputs(
            tokens=device_tokens,
            input_row_offsets=device_row_offsets,
            return_n_logits=return_n_logits_buf,
            signal_buffers=list(self.runtime.signal_buffers),
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
