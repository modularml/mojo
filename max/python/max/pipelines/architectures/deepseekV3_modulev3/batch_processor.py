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
"""Input batching for DeepseekV3 ModuleV3 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    build_single_replica_ragged_token_arrays,
    modulev3_ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.utils import compute_data_parallel_splits

from ..deepseekV2_modulev3.batch_processor import (
    DeepseekV2ModuleV3BatchProcessor,
)

if TYPE_CHECKING:
    from .model import DeepseekV3Inputs


class DeepseekV3ModuleV3BatchProcessor(DeepseekV2ModuleV3BatchProcessor):
    """Ragged batching and DP/EP support for DeepseekV3 ModuleV3 models."""

    def __init__(
        self, config: ArchConfig, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        self._ep_comm_initializer: EPCommInitializer | None = None
        dp_degree = runtime.pipeline_config.model.data_parallel_degree
        self._batch_context_lengths = [
            Buffer.zeros(shape=[1], dtype=DType.int32)
            for _ in range(max(1, dp_degree))
        ]
        self._kv_cache_page_size = (
            runtime.pipeline_config.model.kv_cache.kv_cache_page_size
        )

    def bind_ep_comm_initializer(
        self, initializer: EPCommInitializer | None
    ) -> None:
        """Wires EP buffers created during model ``load_model``."""
        self._ep_comm_initializer = initializer

    def _ep_inputs(self) -> tuple[Buffer, ...]:
        if self._ep_comm_initializer is None:
            return ()
        return tuple(self._ep_comm_initializer.model_inputs())

    @property
    def _dp_degree(self) -> int:
        return self.runtime.pipeline_config.model.data_parallel_degree

    def _update_batch_context_lengths(
        self, replica_batches: Sequence[Sequence[TextContext]]
    ) -> list[Buffer]:
        """Writes each DP replica's page-aligned KV context length in place."""
        page_size = self._kv_cache_page_size

        def align_length(length: int) -> int:
            return (length + page_size - 1) // page_size * page_size

        for i, batch in enumerate(replica_batches):
            self._batch_context_lengths[i].to_numpy()[0] = sum(
                align_length(ctx.tokens.current_position) for ctx in batch
            )

        return self._batch_context_lengths

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
        extra_input_types: Sequence[TensorType | BufferType] = (),
    ) -> list[TensorType | BufferType]:
        """Returns ModuleV3 symbolic inputs plus DP splits and EP buffer types."""
        inputs: list[TensorType | BufferType] = list(
            modulev3_ragged_kv_symbolic_inputs(
                kv_params=kv_params,
                device_refs=device_refs,
            )
        )

        # Host batch context lengths, inserted right after input_row_offsets to
        # match DeepseekV3.forward's positional signature
        # Each data parallel replica should get a different context length
        # buffer.
        dp_degree = self._dp_degree
        assert dp_degree > 0
        for i in range(dp_degree):
            inputs.insert(
                3 + i,
                TensorType(DType.int32, shape=[1], device=DeviceRef.CPU()),
            )

        # Under data parallelism the CPU split boundaries and int64 row offsets
        # follow the per-replica context lengths, matching the order
        # DeepseekV3.forward peels them off its variadic args.
        if dp_degree > 1:
            inputs.insert(
                3 + dp_degree,
                TensorType(
                    DType.int64,
                    shape=[dp_degree + 1],
                    device=DeviceRef.CPU(),
                ),
            )
            inputs.insert(
                4 + dp_degree,
                TensorType(
                    DType.int64,
                    shape=["input_row_offsets_len"],
                    device=DeviceRef.CPU(),
                ),
            )
        inputs.extend(extra_input_types)
        return inputs

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer],
    ) -> DeepseekV3Inputs:
        from .model import DeepseekV3Inputs

        return DeepseekV3Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
            batch_context_lengths=self._batch_context_lengths,
            ep_inputs=self._ep_inputs(),
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DeepseekV3Inputs:
        """Packs the ragged batch.

        Under data parallelism all replicas' requests are concatenated into one
        device-0 batch plus the CPU ``data_parallel_splits`` boundaries the
        forward splits it by.
        """
        from .model import DeepseekV3Inputs

        assert kv_cache_inputs is not None
        device0 = self.runtime.devices[0]
        dp_degree = self._dp_degree
        if dp_degree > 1 and len(replica_batches) != dp_degree:
            raise ValueError(
                f"data parallelism expects {dp_degree} replica batches, "
                f"got {len(replica_batches)}."
            )
        # An empty replica contributes a zero-width boundary; the in-graph
        # split still hands it a 0-token shard.
        context_batch = [ctx for batch in replica_batches for ctx in batch]
        if context_batch:
            tokens_np, offsets_np = build_single_replica_ragged_token_arrays(
                context_batch
            )
        else:
            tokens_np = np.empty(0, dtype=np.int64)
            offsets_np = np.zeros(1, dtype=np.uint32)

        data_parallel_splits = None
        input_row_offsets_i64 = None
        if dp_degree > 1:
            data_parallel_splits = Buffer.from_numpy(
                compute_data_parallel_splits(replica_batches)
            )
            input_row_offsets_i64 = Buffer.from_numpy(
                offsets_np.astype(np.int64)
            )

        return DeepseekV3Inputs(
            tokens=Buffer.from_numpy(tokens_np.astype(np.int64)).to(device0),
            input_row_offsets=Buffer.from_numpy(
                offsets_np.astype(np.uint32)
            ).to(device0),
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            batch_context_lengths=self._update_batch_context_lengths(
                replica_batches
            ),
            data_parallel_splits=data_parallel_splits,
            input_row_offsets_i64=input_row_offsets_i64,
            ep_inputs=self._ep_inputs(),
        )
