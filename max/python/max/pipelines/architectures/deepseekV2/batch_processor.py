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
"""Input batching for DeepseekV2 pipeline models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.driver import Buffer
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.batch_processor import (
    SingleReplicaRaggedBatchProcessor,
    ragged_kv_symbolic_inputs,
)

if TYPE_CHECKING:
    from .model import DeepseekV2Inputs


class DeepseekV2BatchProcessor(
    SingleReplicaRaggedBatchProcessor[TextContext, "DeepseekV2Inputs"]
):
    """Single-replica ragged KV batching for DeepseekV2.

    Signal buffers are included only when more than one device is present.
    ``return_n_logits`` is placed on the first device (GPU) to match the
    DeepseekV2 graph's input type definition.
    """

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

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None,
        signal_buffers: list[Buffer],
    ) -> DeepseekV2Inputs:
        from .model import DeepseekV2Inputs

        return DeepseekV2Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            signal_buffers=signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits.to(self.runtime.devices[0]),
        )
