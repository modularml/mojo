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
"""Input batching for Olmo3 pipeline models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.driver import Buffer
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.batch_processor import (
    SingleReplicaRaggedBatchProcessor,
    modulev3_ragged_kv_symbolic_inputs,
)

if TYPE_CHECKING:
    from .model import Olmo3Inputs


class Olmo3BatchProcessor(
    SingleReplicaRaggedBatchProcessor[TextContext, "Olmo3Inputs"]
):
    """Single-replica ragged KV batching for Olmo3 (no signal buffers)."""

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns ModuleV3 compile input order (tokens, return_n_logits, offsets, *kv)."""
        return modulev3_ragged_kv_symbolic_inputs(
            kv_params=kv_params,
            device_refs=device_refs,
        )

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None,
        signal_buffers: list[Buffer],
    ) -> Olmo3Inputs:
        from .model import Olmo3Inputs

        del signal_buffers
        return Olmo3Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            kv_cache_inputs=kv_cache_inputs,
        )
