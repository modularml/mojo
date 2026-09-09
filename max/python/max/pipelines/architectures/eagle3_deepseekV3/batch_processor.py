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
"""Input batching for Eagle3 + DeepseekV3 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.architectures.deepseekV3.batch_processor import (
    DeepseekV3BatchProcessor,
)
from max.pipelines.architectures.deepseekV3.model import DeepseekV3Inputs
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces import ArchConfig, BatchProcessorRuntime

if TYPE_CHECKING:
    from .mha_pipeline import Eagle3MHADeepseekV3Inputs
    from .model import Eagle3DeepseekV3Inputs


class _Eagle3DeepseekV3BatchProcessorBase(DeepseekV3BatchProcessor):
    """Shared Eagle3 batching: DeepseekV3 inputs plus seed and draft slot."""

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        self._seed_counter: int = 0

    def _next_seed(self) -> Buffer:
        """Returns a monotonically advancing ``uint64[1]`` seed on device 0."""
        self._seed_counter += 1
        return Buffer.from_numpy(
            np.array([self._seed_counter], dtype=np.uint64)
        ).to(self.runtime.devices[0])

    def _prepare_deepseek_base(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None,
        return_n_logits: int,
    ) -> DeepseekV3Inputs:
        return super().prepare_initial_token_inputs(
            replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )


class Eagle3DeepseekV3BatchProcessor(_Eagle3DeepseekV3BatchProcessorBase):
    """Ragged batching for the Eagle3 + DeepseekV3 unified (MLA-draft) model."""

    def prepare_initial_token_inputs(  # type: ignore[override]
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Eagle3DeepseekV3Inputs:
        from .model import Eagle3DeepseekV3Inputs

        base = self._prepare_deepseek_base(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        return Eagle3DeepseekV3Inputs(
            tokens=base.tokens,
            input_row_offsets=base.input_row_offsets,
            host_input_row_offsets=base.host_input_row_offsets,
            batch_context_lengths=base.batch_context_lengths,
            signal_buffers=base.signal_buffers,
            kv_cache_inputs=base.kv_cache_inputs,
            return_n_logits=base.return_n_logits,
            data_parallel_splits=base.data_parallel_splits,
            ep_inputs=base.ep_inputs,
            draft_tokens=None,
            seed=self._next_seed(),
            structured_output=self.runtime.pipeline_config.needs_bitmask_constraints,
        )


class Eagle3MHADeepseekV3BatchProcessor(_Eagle3DeepseekV3BatchProcessorBase):
    """Ragged batching for the Eagle3 MHA-draft + DeepseekV3 unified model."""

    def prepare_initial_token_inputs(  # type: ignore[override]
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Eagle3MHADeepseekV3Inputs:
        from .mha_pipeline import Eagle3MHADeepseekV3Inputs

        base = self._prepare_deepseek_base(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        return Eagle3MHADeepseekV3Inputs(
            tokens=base.tokens,
            input_row_offsets=base.input_row_offsets,
            host_input_row_offsets=base.host_input_row_offsets,
            batch_context_lengths=base.batch_context_lengths,
            signal_buffers=base.signal_buffers,
            kv_cache_inputs=base.kv_cache_inputs,
            return_n_logits=base.return_n_logits,
            data_parallel_splits=base.data_parallel_splits,
            ep_inputs=base.ep_inputs,
            draft_tokens=None,
            seed=self._next_seed(),
            structured_output=self.runtime.pipeline_config.needs_bitmask_constraints,
        )
