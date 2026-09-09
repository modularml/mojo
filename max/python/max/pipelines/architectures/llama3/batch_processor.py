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
"""Input batching for Llama 3 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer, Device, DevicePinnedBuffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    RaggedBatchProcessor,
    process_ragged_kv_outputs,
    ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelOutputs
from max.pipelines.lib.utils import compute_data_parallel_splits
from max.support.algorithm import flatten2d

if TYPE_CHECKING:
    from .model import Llama3Inputs


class Llama3BatchProcessor(RaggedBatchProcessor[TextContext, "Llama3Inputs"]):
    """Ragged batching with pinned host buffers and optional DP / LoRA."""

    def _stage_ragged_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        device0: Device,
    ) -> tuple[Buffer, Buffer, Buffer]:
        """Stages ragged tokens/offsets into cached device buffers.

        Fresh pinned host staging is allocated every step (never reused) so the
        next overlap step's host writes can't clobber the in-flight H2D copy.
        Destination device buffers are cached and reused so captured graphs
        replay in place.

        Returns:
            ``(device_tokens, device_row_offsets, host_row_offsets)``; the host
            offsets are also used by DP/LoRA callers.
        """
        batch_size = len(context_batch)
        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)
        pinned = not device0.is_host

        host_buffer_cls = DevicePinnedBuffer if pinned else Buffer
        host_tokens: Buffer = host_buffer_cls(
            dtype=DType.int64, shape=(total_seq_len,), device=device0
        )
        host_row_offsets: Buffer = host_buffer_cls(
            dtype=DType.uint32, shape=(batch_size + 1,), device=device0
        )

        np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
            out=host_row_offsets.to_numpy(),
        )
        if context_batch:
            np.concatenate(
                [ctx.tokens.active for ctx in context_batch],
                out=host_tokens.to_numpy(),
            )

        if not pinned:
            # On host there is no separate device memory; the graph reads the
            # host buffers directly.
            return host_tokens, host_row_offsets, host_row_offsets

        device_tokens = self._device_input_allocator.alloc(
            name="ragged_input_tokens",
            dtype=DType.int64,
            shape=(total_seq_len,),
            device=device0,
        )
        device_row_offsets = self._device_input_allocator.alloc(
            name="ragged_input_row_offsets",
            dtype=DType.uint32,
            shape=(batch_size + 1,),
            device=device0,
        )
        device_tokens.inplace_copy_from(host_tokens)
        device_row_offsets.inplace_copy_from(host_row_offsets)
        return device_tokens, device_row_offsets, host_row_offsets

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

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Llama3Inputs:
        from .model import Llama3Inputs

        dp = self.runtime.pipeline_config.model.data_parallel_degree
        if len(replica_batches) != dp:
            raise ValueError(
                "Number of replica batches must match data parallel degree"
            )

        context_batch = flatten2d(replica_batches)
        device0 = self.runtime.devices[0]

        device_tokens, device_row_offsets, _ = self._stage_ragged_token_inputs(
            context_batch, device0
        )

        return_n_logits_tensor = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        if dp > 1:
            data_parallel_splits = Buffer.from_numpy(
                compute_data_parallel_splits(replica_batches)
            )
        else:
            data_parallel_splits = None

        return Llama3Inputs(
            tokens=device_tokens,
            input_row_offsets=device_row_offsets,
            return_n_logits=return_n_logits_tensor,
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
            data_parallel_splits=data_parallel_splits,
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )


class Llama3EpBatchProcessor(Llama3BatchProcessor):
    """Llama3 batching extended with EP MoE communication buffers."""

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        self._ep_comm_initializer: EPCommInitializer | None = None

    def bind_ep_comm_initializer(
        self, initializer: EPCommInitializer | None
    ) -> None:
        """Wires EP buffers created during model ``load_model``."""
        self._ep_comm_initializer = initializer

    def _ep_inputs(self) -> tuple[Buffer, ...]:
        if self._ep_comm_initializer is None:
            return ()
        return tuple(self._ep_comm_initializer.model_inputs())

    def _host_input_row_offsets_for_dp(
        self, host_row_offsets: Buffer, dp: int
    ) -> Buffer | None:
        return host_row_offsets if dp > 1 else None

    def _prepare_ep_moe_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        return_n_logits: int,
    ) -> tuple[
        Buffer,
        Buffer,
        Buffer,
        Buffer | None,
        tuple[Buffer, ...],
        Buffer | None,
    ]:
        dp = self.runtime.pipeline_config.model.data_parallel_degree
        if len(replica_batches) != dp:
            raise ValueError(
                "Number of replica batches must match data parallel degree"
            )

        context_batch = flatten2d(replica_batches)
        device0 = self.runtime.devices[0]

        device_tokens, device_row_offsets, host_row_offsets = (
            self._stage_ragged_token_inputs(context_batch, device0)
        )

        return_n_logits_tensor = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        if dp > 1:
            data_parallel_splits = Buffer.from_numpy(
                compute_data_parallel_splits(replica_batches)
            )
        else:
            data_parallel_splits = None

        return (
            device_tokens,
            device_row_offsets,
            return_n_logits_tensor,
            data_parallel_splits,
            self._ep_inputs(),
            self._host_input_row_offsets_for_dp(host_row_offsets, dp),
        )
