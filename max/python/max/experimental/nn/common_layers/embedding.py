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

"""Vocab-parallel embedding for multi-device tensor parallelism."""

from __future__ import annotations

import math

from max.experimental import functional as F
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.embedding import Embedding
from max.experimental.sharding import (
    NamedMapping,
    Partial,
    PlacementMapping,
)
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, DimLike, TensorValue, ops


class VocabParallelEmbedding(Embedding):
    """An embedding whose vocabulary is sharded across devices.

    On a single device this behaves identically to
    :class:`~max.experimental.nn.embedding.Embedding`.  On a multi-device
    mesh the vocabulary dimension (axis 0) is split so each device holds
    ``ceil(vocab_size / n)`` rows.  A lookup gathers from the local shard,
    masks out-of-range indices, and all-reduces the results.
    """

    def __init__(
        self, vocab_size: DimLike, *, dim: DimLike, tp_axis: str | None = None
    ) -> None:
        super().__init__(vocab_size, dim=dim)
        if tp_axis is None:
            tp_axis = TP
        self.weight._mapping = NamedMapping(self.weight.mesh, (tp_axis, None))

    def forward(self, indices: Tensor) -> Tensor:
        """Gather the embeddings for the input indices."""
        if not self.weight.is_distributed:
            return F.gather(self.weight, indices, axis=0)
        return self._vocab_parallel_gather(indices)

    def _vocab_parallel_gather(self, indices: Tensor) -> Tensor:
        """Per-shard gather with masking and all-reduce."""
        mesh = self.weight.mesh
        n = mesh.num_devices
        vocab_size = int(self.weight.shape[0])
        shard_size = math.ceil(vocab_size / n)

        weight_shards = [TensorValue(w) for w in self.weight.local_shards]

        if indices.is_distributed:
            idx_shards = [TensorValue(i) for i in indices.local_shards]
        else:
            idx_tv = TensorValue(indices)
            idx_shards = [
                ops.transfer_to(idx_tv, DeviceRef.from_device(mesh.devices[i]))
                for i in range(n)
            ]

        results = []
        for i in range(n):
            vocab_start = shard_size * i
            vocab_end = min(shard_size * (i + 1), vocab_size)

            in_range = ops.logical_and(
                idx_shards[i] >= vocab_start, idx_shards[i] < vocab_end
            )
            local_idx = (idx_shards[i] - vocab_start) * in_range

            gathered = ops.gather(weight_shards[i], local_idx, axis=0)
            mask = ops.cast(ops.unsqueeze(in_range, -1), gathered.dtype)
            results.append(gathered * mask)

        partial = Tensor.from_shard_values(
            results, PlacementMapping(mesh, (Partial(),))
        )
        return F.allreduce_sum(partial)
