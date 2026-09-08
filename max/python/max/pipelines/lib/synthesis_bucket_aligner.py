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
"""Cache-length bucketing for device-graph synthesis serving.

Device-graph synthesis records kernels into a driver ``DeviceGraph`` lazily
inside ``execute`` and keys the recorded graph on symbolic dims, device, input
addresses, and host-input *contents* read at enqueue time. Decode attention
dispatch metadata (e.g. the 4-int MHA buffer
``[batch_size, q, num_partitions, max_cache_valid_length]``) is a host input
whose contents increment every decode step, which would force a fresh device
graph per step (a ~3.3x decode cost measured on the 12B gsm8k run).

This aligner reuses the same bucketing policy as the record/replay
:class:`~max.pipelines.lib.graph_capture.ServeGraphCaptureRunner` -- the
grain-256 probe boundaries from
:meth:`KVCacheParamInterface.graph_capture_probe_cache_lengths` and a snap-up
``bisect`` -- but none of its capture/replay mechanism. Snapping the runtime
cache length to a recorded boundary keeps dispatch-metadata buffer contents
stable across consecutive decode steps, so the driver ``DeviceGraphCache`` hits
instead of rebuilding. The first step in a new bucket pays the build; later
steps in that bucket hit.

Unlike graph capture, synthesis builds lazily per bucket, so this aligner holds
no captured graph state, no model handle, and no buffer pool -- only the sorted
boundaries and the query width to validate against.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import replace

from max.nn.kv_cache import BatchCharacteristics, KVCacheParamInterface

logger = logging.getLogger("max.pipelines")


class SynthesisBucketAligner:
    """Snaps decode cache length to recorded boundaries for graph synthesis.

    Reuses the KV-cache probe-length policy (grain 256 for MHA, unioned across
    children for multi-cache trees like Gemma 4's ``{sliding, full}``) from the
    record/replay path, without any capture/replay coupling. The bucket
    boundaries are computed once at construction; :meth:`align` is a pure CPU
    ``bisect`` lookup on the hot path with no resolver kernel op.
    """

    def __init__(
        self,
        *,
        kv_params: KVCacheParamInterface,
        max_cache_length_upper_bound: int,
        num_speculative_tokens: int = 0,
    ) -> None:
        if max_cache_length_upper_bound < 1:
            raise ValueError(
                "Device-graph synthesis bucketing requires a positive decode "
                "max-cache length upper bound."
            )
        self._kv_params = kv_params
        self._max_cache_length_upper_bound = max_cache_length_upper_bound

        # The query (prompt) width every decode forward runs at. Speculative
        # verify runs at q = 1 + num_speculative_tokens; plain decode at q = 1.
        self._q_max_seq_len = num_speculative_tokens + 1

        # Bucket boundaries shared with graph capture: grain 256 for MHA,
        # unioned across children for multi-attention caches. Snap targets are
        # exactly the probed lengths, so a snapped length is guaranteed to map
        # to one dispatch-metadata buffer contents the driver has recorded.
        probe_lengths = self._kv_params.graph_capture_probe_cache_lengths(
            self._max_cache_length_upper_bound, self._q_max_seq_len
        )
        self._recorded_cache_lengths: list[int] = sorted(set(probe_lengths))
        if not self._recorded_cache_lengths:
            raise ValueError(
                "Device-graph synthesis bucketing produced no recorded "
                "cache lengths; KV cache params returned no probe lengths."
            )

    @property
    def max_cache_length_upper_bound(self) -> int:
        """The upper bound passed to ``runtime_inputs(max_cache_length=...)``."""
        return self._max_cache_length_upper_bound

    @property
    def recorded_cache_lengths(self) -> list[int]:
        """Sorted cache-length bucket boundaries the aligner snaps up to."""
        return self._recorded_cache_lengths

    def _bucket_cache_length(self, cache_length: int) -> int:
        """Rounds a runtime cache length up to the nearest recorded boundary.

        The recorded boundaries are exactly the probed lengths, so the snapped
        value resolves to dispatch metadata the driver has (or will, lazily)
        recorded a device graph for.
        """
        if cache_length > self._recorded_cache_lengths[-1]:
            raise RuntimeError(
                f"Cache length {cache_length} exceeds the largest bucket "
                f"boundary {self._recorded_cache_lengths[-1]}"
            )
        idx = bisect.bisect_left(self._recorded_cache_lengths, cache_length)
        return self._recorded_cache_lengths[idx]

    def align(
        self, characteristics: BatchCharacteristics
    ) -> BatchCharacteristics:
        """Aligns real batch characteristics to a synthesis bucket.

        Buckets ``characteristics.max_cache_valid_length`` up to a recorded
        boundary, yielding aligned characteristics for the
        ``(batch_size, q, aligned_cache_length)`` shape (a pure CPU bucketing --
        no resolver kernel op). The caller passes the returned
        :class:`BatchCharacteristics` to ``KVCacheManager.runtime_inputs`` so
        dispatch metadata is prepared for the aligned length, which keeps the
        host dispatch-metadata buffer contents stable across decode steps in
        the same bucket.

        Args:
            characteristics: The batch's real (upper-bound) characteristics.
                For data parallelism this is the per-replica maximum, since
                every replica must record the identical device graph.

        Returns:
            The aligned characteristics.

        Raises:
            RuntimeError: If ``q_max_seq_len`` differs from the captured value
                or the cache length exceeds the largest bucket boundary.
        """
        if characteristics.max_prompt_length != self._q_max_seq_len:
            raise RuntimeError(
                f"q_max_seq_len={characteristics.max_prompt_length} != "
                f"{self._q_max_seq_len}; only q_max_seq_len="
                f"{self._q_max_seq_len} decode shapes are bucketed."
            )
        return replace(
            characteristics,
            max_cache_valid_length=self._bucket_cache_length(
                characteristics.max_cache_valid_length
            ),
        )
