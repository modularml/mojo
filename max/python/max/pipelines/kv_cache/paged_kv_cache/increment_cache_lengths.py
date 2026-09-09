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

"""Builds computation graphs for incrementing cache lengths in ragged tensor operations."""

from __future__ import annotations

from max.graph import BufferValue, TensorValue, ops
from max.nn.kv_cache.data_parallelism_utils import split_into_groups


def increment_cache_lengths_from_counts(
    batch_increments: TensorValue,
    data_parallel_splits: TensorValue,
    cache_lengths: list[TensorValue],
    signal_buffers: list[BufferValue] | None,
) -> list[TensorValue]:
    """Adds per-request cache-length increments to each device's cache lengths.

    This helper works directly from batch-aligned per-request increments
    instead of reconstructing ragged row offsets. To preserve empty-shard
    behavior, it broadcasts the full increment vector and slices each replica's
    local range after the broadcast.

    Args:
        batch_increments: Per-request cache-length increments on device 0, shape
            ``[batch]``.
        data_parallel_splits: DP split boundaries, shape ``[dp+1]``, on CPU.
        cache_lengths: Current cache lengths per device.
        signal_buffers: Signal buffers for multi-device comm (``None`` for
            single device).

    Returns:
        Updated cache lengths per device.
    """
    dp = int(data_parallel_splits.shape[0]) - 1
    n_devices = len(cache_lengths)
    replica_groups = split_into_groups(list(range(n_devices)), dp)

    if signal_buffers is not None:
        lengths_all = ops.distributed_broadcast(
            batch_increments, signal_buffers
        )
    else:
        lengths_all = [batch_increments]

    outputs = []
    for replica_idx, device_indices in enumerate(replica_groups):
        if replica_idx + 1 >= dp:
            end_idx = None
        else:
            end_idx = data_parallel_splits[replica_idx + 1]

        for gpu_idx in device_indices:
            cache_length = cache_lengths[gpu_idx]
            assert isinstance(cache_length, TensorValue)
            local_lengths = ops.slice_tensor(
                lengths_all[gpu_idx],
                [
                    (
                        slice(data_parallel_splits[replica_idx], end_idx),
                        f"length_split_{gpu_idx}",
                    )
                ],
            )
            increment_amount = local_lengths.cast(cache_length.dtype).rebind(
                cache_length.shape
            )
            outputs.append(cache_length + increment_amount)

    return outputs
