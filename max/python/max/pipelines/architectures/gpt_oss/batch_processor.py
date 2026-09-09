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
"""Input batching for GPT OSS pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.batch_processor import (
    SingleReplicaRaggedBatchProcessor,
    build_single_replica_ragged_token_arrays,
    single_replica_context_batch,
)

if TYPE_CHECKING:
    from .model import GptOssInputs


class GptOssBatchProcessor(
    SingleReplicaRaggedBatchProcessor[TextContext, "GptOssInputs"]
):
    """Single-replica ragged KV batching for GPT OSS with per-device row offsets."""

    _include_signal_buffers = True

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> GptOssInputs:
        from .model import GptOssInputs

        context_batch = single_replica_context_batch(
            replica_batches,
            processor_name=type(self).__qualname__,
        )
        tokens_np, offsets_np = build_single_replica_ragged_token_arrays(
            context_batch
        )
        tokens = Buffer.from_numpy(tokens_np).to(self.runtime.devices[0])
        input_row_offsets_per_dev = [
            Buffer.from_numpy(offsets_np).to(device)
            for device in self.runtime.devices
        ]
        return GptOssInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets_per_dev,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
        )
