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
"""Input batching for Mamba pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.batch_processor import (
    RaggedBatchProcessor,
    process_ragged_kv_outputs,
    ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelOutputs

if TYPE_CHECKING:
    from .model import MambaModelInputs
    from .ssm_cache import SSMStateCache


class MambaBatchProcessor(
    RaggedBatchProcessor[TextContext, "MambaModelInputs"]
):
    """Ragged batching extended with per-request SSM state slots."""

    _ssm_cache: SSMStateCache | None = None

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

    def bind_ssm_cache(self, ssm_cache: SSMStateCache) -> None:
        """Wires the SSM state cache created during model ``__init__``."""
        self._ssm_cache = ssm_cache

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> MambaModelInputs:
        from .model import MambaModelInputs

        assert self._ssm_cache is not None, (
            "Mamba SSM cache must be bound before prepare_initial_token_inputs()"
        )

        if len(replica_batches) != 1:
            raise ValueError("Mamba does not support DP>1")

        context_batch = replica_batches[0]
        request_ids = [ctx.request_id for ctx in context_batch]

        for rid in request_ids:
            self._ssm_cache.claim(rid)

        device0 = self.runtime.devices[0]

        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )
        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        tokens_buf = Buffer.from_numpy(tokens).to(device0)
        offsets_buf = Buffer.from_numpy(input_row_offsets).to(device0)
        n_logits_buf = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        has_existing_states = any(
            self._ssm_cache.contains(rid)
            and self._ssm_cache.has_valid_state(rid)
            for rid in request_ids
        )

        if has_existing_states:
            layer_states = self._ssm_cache.get_states(request_ids)
            inputs = MambaModelInputs(
                tokens_buf,
                offsets_buf,
                n_logits_buf,
                is_prefill=False,
                layer_states=layer_states,
                request_ids=request_ids,
            )
        else:
            inputs = MambaModelInputs(
                tokens_buf,
                offsets_buf,
                n_logits_buf,
                is_prefill=True,
                request_ids=request_ids,
            )
        inputs.kv_cache_inputs = kv_cache_inputs
        return inputs

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )
