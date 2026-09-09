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
"""Centralized graph capture runner for overlap serving.

Flow:
- Model worker creates the runner and executes pre-ready warmup.
- Warmup probes a set of cache lengths (``KVCacheParams.graph_capture_probe_
  cache_lengths``); for each ``(batch_size, q, cache_length)`` it resolves the
  dispatch metadata (via ``KVCacheParams.resolve_attn_key``), captures one
  device graph per distinct key, and records the
  ``BatchCharacteristics -> GraphKey`` mapping plus the set of recorded cache
  lengths.
- Attention dispatch metadata is prepared exactly once, by the KV cache
  manager's ``runtime_inputs(batch_characteristics=...)`` -- the same code path
  used at replay -- so capture and replay agree by construction.
- Serving replays by bucketing the runtime cache length up to a recorded length
  and looking up the recorded ``GraphKey`` (a pure CPU lookup; no resolver
  kernel op on the hot path). ``q_max_seq_len`` must be ``1 + w`` for one of the
  captured verify widths ``w``; any other value raises ``RuntimeError``.
"""

from __future__ import annotations

import bisect
import logging
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import replace

from max._core.driver import _release_buffers_to_borrowed
from max.driver import Buffer, batch_inplace_copy
from max.engine import Model
from max.experimental.nn.module import CompiledModel
from max.nn.kv_cache import BatchCharacteristics, KVCacheParamInterface
from max.nn.kv_cache.utils import AttnKeyInterface, MultiAttnKey
from max.profiler import traced
from tqdm import tqdm

from .interfaces import ModelInputs, ModelOutputs, UnifiedEagleOutputs

logger = logging.getLogger("max.pipelines")


# Captured device graphs are keyed by
GraphEntry = tuple[tuple[Buffer, ...], ModelOutputs]
# Builds warmup model inputs for a ``(batch_size, batch_characteristics)`` pair.
# The characteristics drive the KV manager so dispatch metadata is prepared for
# the probed cache length.
WarmupModelInputs = Callable[
    [int, BatchCharacteristics], AbstractContextManager[ModelInputs]
]


def _release_graph_capture_outputs_to_borrowed(
    outputs: ModelOutputs,
) -> ModelOutputs:
    """Returns graph-capture warmup outputs as borrowed wrappers.

    The returned buffers continue to point at the same storage, but later
    captures or replays may overwrite that memory.
    """
    buffer_field_names: list[str] = []
    buffers: list[Buffer] = []
    for field_name in (
        "logits",
        "next_token_logits",
        "logit_offsets",
        "hidden_states",
        "num_accepted_draft_tokens",
        "next_tokens",
        "next_draft_tokens",
        "next_draft_probs_full",
    ):
        value = getattr(outputs, field_name, None)
        if isinstance(value, Buffer):
            buffer_field_names.append(field_name)
            buffers.append(value)

    if not buffers:
        return outputs

    released_buffers = _release_buffers_to_borrowed(buffers)
    return replace(
        outputs,
        **dict(zip(buffer_field_names, released_buffers, strict=True)),
    )


def _pack_model_graph_key(key: AttnKeyInterface) -> int:
    """Maps a capture key to a uint64 for the C++ capture layer."""
    return hash(key) & 0xFFFFFFFFFFFFFFFF


class ServeGraphCaptureRunner:
    """Central owner for serve-time graph capture state."""

    def __init__(
        self,
        *,
        model: Model,
        kv_params: KVCacheParamInterface,
        warmup_model_inputs: WarmupModelInputs,
        max_cache_length_upper_bound: int,
        max_batch_size: int,
        num_speculative_tokens: int = 0,
        verify_widths: Sequence[int] | None = None,
        width_lookup: Sequence[int] | None = None,
    ) -> None:
        self._model = model
        self._warmup_model_inputs = warmup_model_inputs
        self._num_speculative_tokens = num_speculative_tokens
        if max_cache_length_upper_bound < 1:
            raise ValueError(
                "Decode graph capture requires a positive decode "
                "max-cache length upper bound."
            )
        self._max_cache_length_upper_bound = max_cache_length_upper_bound
        if max_batch_size < 1:
            raise ValueError(
                "Device graph capture requires a positive decode capture "
                "batch-size upper bound."
            )
        self._max_batch_size = max_batch_size

        # Dispatch resolution + probe lengths live on the KV cache params.
        self._kv_params = kv_params
        self._is_spec_decode = num_speculative_tokens > 0

        widths = sorted(set(verify_widths or (num_speculative_tokens,)))
        for width in widths:
            if not 0 <= width <= num_speculative_tokens:
                raise ValueError(
                    f"Verify width {width} is outside [0, "
                    f"{num_speculative_tokens}]: a step cannot verify more "
                    "drafts than it carries."
                )
        self._verify_widths = widths
        # ``batch_size -> verify width``. When set, only the width a batch size
        # resolves to is reachable, so only that one is probed.
        self._width_lookup = width_lookup
        if width_lookup is not None:
            for batch_size in range(1, self._max_batch_size + 1):
                scheduled = width_lookup[min(batch_size, len(width_lookup) - 1)]
                if scheduled not in widths:
                    raise ValueError(
                        f"Batch size {batch_size} resolves to verify width "
                        f"{scheduled}, which is not among the captured widths "
                        f"{widths}."
                    )
        # Block drafts (DFlash) run at q=num_draft_tokens_per_step; autoregressive
        # drafts (eagle/mtp) run at q=1.
        self._draft_q_at_capture = kv_params.num_draft_tokens_per_step

        self.graph_entries: dict[AttnKeyInterface, GraphEntry] = {}
        # Maps a probed ``(batch_size, q, cache_length)`` to the captured graph.
        # Many cache lengths can map to one ``GraphKey`` (one captured graph).
        self._records: dict[BatchCharacteristics, AttnKeyInterface] = {}
        # Sorted, distinct cache lengths recorded during capture (the snap set).
        self._recorded_cache_lengths: list[int] = []

    def release_graph(self, key: AttnKeyInterface) -> None:
        """Releases a single captured graph and its working memory.

        Drops the runner's entry for ``key`` (input + output buffer handles)
        and asks the engine to release the underlying device graph. Safe to
        call when ``key`` is not currently captured: the runner-side ``pop``
        becomes a no-op and the engine-side release is itself a no-op for
        unknown keys.
        """
        self.graph_entries.pop(key, None)
        self._model.release_captured_graph(_pack_model_graph_key(key))

    def _probe_verify_widths(self, batch_size: int) -> list[int]:
        """Returns the verify widths to capture for ``batch_size``.

        Without a schedule any width is reachable at any batch size, so all of
        them are probed. Here we pin one width per batch size.
        """
        if self._width_lookup is None:
            return self._verify_widths
        index = min(batch_size, len(self._width_lookup) - 1)
        return [self._width_lookup[index]]

    def _resolve_graph_key(
        self, batch_size: int, cache_length: int, q_max_seq_len: int
    ) -> AttnKeyInterface:
        """Resolves the ``GraphKey`` for a ``(batch_size, q, cache_length)`` shape.

        Resolves the verify-width dispatch metadata tree and, under speculative
        decoding, the draft-width tree via the KV cache params, then folds them
        into one capture key. Each tree is a leaf :class:`AttnKey` or a
        ``MultiAttnKey`` mirroring the cache tree. Calls the
        resolver kernel op, so it is used at warmup only.
        """
        children: dict[str, AttnKeyInterface] = {}
        children["verify"] = self._kv_params.resolve_attn_key(
            batch_size, q_max_seq_len, cache_length
        )
        if self._is_spec_decode:
            # Block drafts (DFlash) run the draft at q=num_draft_tokens_per_step;
            # autoregressive drafts (eagle/mtp) run at q=1. Resolve the draft
            # dispatch key at that width so the captured-graph identity matches
            # the shape actually executed.
            children["draft"] = self._kv_params.resolve_attn_key(
                batch_size, self._draft_q_at_capture, cache_length
            )
        return MultiAttnKey.from_dict(children)

    @traced
    def warmup_pre_ready(self) -> None:
        """Captures decode buckets before the worker becomes ready."""
        logger.info(
            "Pre-capturing overlap device graphs for decode batch sizes [1..%d] "
            "at verify widths %s with num_steps=1.",
            self._max_batch_size,
            self._verify_widths,
        )
        probe_lengths: set[int] = set()
        for width in self._verify_widths:
            probe_lengths.update(
                self._kv_params.graph_capture_probe_cache_lengths(
                    self._max_cache_length_upper_bound, width + 1
                )
            )
        recorded_lengths: set[int] = set()
        # Capture largest-first so peak allocations
        # happen up front and oversized configs fail fast.
        probes = sorted(
            (
                (width, batch_size)
                for batch_size in range(1, self._max_batch_size + 1)
                for width in self._probe_verify_widths(batch_size)
            ),
            reverse=True,
        )
        for q_max_seq_len, batch_size in tqdm(
            [(width + 1, batch_size) for width, batch_size in probes],
            desc="Capturing device graph shapes",
        ):
            for cache_length in sorted(probe_lengths, reverse=True):
                recorded_lengths.add(cache_length)
                graph_key = self._resolve_graph_key(
                    batch_size, cache_length, q_max_seq_len
                )
                # Record every probed length so replay can bucket to any of
                # them; many lengths share one captured graph.
                self._records[
                    BatchCharacteristics(
                        batch_size=batch_size,
                        max_prompt_length=q_max_seq_len,
                        max_cache_valid_length=cache_length,
                    )
                ] = graph_key
                if graph_key in self.graph_entries:
                    continue

                # Prepare dispatch metadata once, via the same KV-manager path
                # used at replay (``runtime_inputs(batch_characteristics=...)``).
                batch_characteristics = BatchCharacteristics(
                    batch_size=batch_size,
                    max_prompt_length=q_max_seq_len,
                    max_cache_valid_length=cache_length,
                )
                with self._warmup_model_inputs(
                    batch_size, batch_characteristics
                ) as model_inputs:
                    input_buffers = model_inputs.buffers
                    if isinstance(self._model, CompiledModel):
                        output_buffers = self._model.engine_model.capture(
                            _pack_model_graph_key(graph_key),
                            *input_buffers,
                            *self._model.signal_buffers,
                        )
                    else:
                        output_buffers = self._model.capture(
                            _pack_model_graph_key(graph_key), *input_buffers
                        )
                    if not self._is_spec_decode:
                        outputs = ModelOutputs(*output_buffers)
                    else:
                        if len(output_buffers) not in (3, 4):
                            raise RuntimeError(
                                "spec-decode graph capture returned "
                                f"{len(output_buffers)} outputs; expected 3 "
                                "(num_accepted_draft_tokens, next_tokens, "
                                "next_draft_tokens) or 4 (+ "
                                "next_draft_probs_full, under "
                                "draft_proposal='sampled')."
                            )
                        outputs = UnifiedEagleOutputs(
                            num_accepted_draft_tokens=output_buffers[0],
                            next_tokens=output_buffers[1],
                            next_draft_tokens=output_buffers[2],
                            next_draft_probs_full=output_buffers[3]
                            if len(output_buffers) == 4
                            else None,
                        )
                    # Graph-capture warmup keeps many output handles alive. Drop
                    # Python-side ownership so later captures can reuse the same
                    # memory-manager-backed storage.
                    outputs = _release_graph_capture_outputs_to_borrowed(
                        outputs
                    )
                    self.graph_entries[graph_key] = (input_buffers, outputs)

        self._recorded_cache_lengths = sorted(recorded_lengths)
        logger.info(
            "Captured %d distinct device graphs from %d probe records at "
            "verify widths %s (%d recorded cache lengths).",
            len(self.graph_entries),
            len(self._records),
            self._verify_widths,
            len(self._recorded_cache_lengths),
        )

        if hasattr(self._model, "_await_device_graphs"):
            logger.info(
                "Awaiting remaining device graph instantiation threads."
            )
            t0 = time.perf_counter()
            self._model._await_device_graphs()
            logger.info(
                "Device graph instantiation complete in %.3fs.",
                time.perf_counter() - t0,
            )

        logger.info(
            "Overlap device graph pre-capture complete for decode batch sizes "
            "[1..%d] with num_steps=1.",
            self._max_batch_size,
        )

    def _bucket_cache_length(self, cache_length: int) -> int:
        """Rounds a runtime cache length up to the nearest recorded length.

        The recorded lengths are exactly the cache lengths captured during
        warmup, so the snapped value is guaranteed to have a captured graph.
        """
        if not self._recorded_cache_lengths:
            raise RuntimeError(
                "No recorded cache lengths; warmup_pre_ready must run before "
                "replay."
            )
        if cache_length > self._recorded_cache_lengths[-1]:
            raise RuntimeError(
                f"Cache length {cache_length} exceeds the largest captured length {self._recorded_cache_lengths[-1]}"
            )
        idx = bisect.bisect_left(self._recorded_cache_lengths, cache_length)
        return self._recorded_cache_lengths[idx]

    def align(
        self, characteristics: BatchCharacteristics
    ) -> BatchCharacteristics:
        """Aligns real batch characteristics to a captured graph.

        Buckets ``characteristics.max_cache_valid_length`` up to a recorded
        length, yielding aligned characteristics for the
        ``(batch_size, q, aligned_cache_length)`` shape (a pure CPU bucketing --
        no resolver kernel op). The caller passes the returned
        :class:`~max.nn.kv_cache.BatchCharacteristics` to
        ``KVCacheManager.runtime_inputs`` so the dispatch metadata is prepared
        for the aligned length, and to :meth:`replay`, which looks up the
        captured ``GraphKey`` for those characteristics.

        Args:
            characteristics: The batch's real (upper-bound) characteristics.
                For data parallelism this is the per-replica maximum, since
                every replica must replay the identical captured graph.

        Returns:
            The aligned characteristics.

        Raises:
            RuntimeError: If ``q_max_seq_len`` matches no captured verify width
                or the cache length exceeds the largest captured length.
        """
        verify_width = characteristics.max_prompt_length - 1
        if verify_width not in self._verify_widths:
            raise RuntimeError(
                f"q_max_seq_len={characteristics.max_prompt_length} implies "
                f"verify width {verify_width}, which is not captured; "
                f"captured widths are {self._verify_widths}."
            )
        aligned = replace(
            characteristics,
            max_cache_valid_length=self._bucket_cache_length(
                characteristics.max_cache_valid_length
            ),
        )
        return aligned

    @traced
    def replay(
        self,
        *,
        model_inputs: ModelInputs,
        batch_characteristics: BatchCharacteristics,
        debug_verify_replay: bool = False,
        debug_verify_model_inputs: ModelInputs | None = None,
    ) -> ModelOutputs:
        """Replays the captured graph identified by ``batch_characteristics``.

        ``batch_characteristics`` comes from :meth:`align`. ``model_inputs`` must
        already carry dispatch metadata prepared for the aligned characteristics
        (via ``runtime_inputs(batch_characteristics=...)``); this method only
        copies the inputs into the captured replay buffers and replays.
        """
        input_buffers = model_inputs.buffers
        replay_graph_key = self._records.get(batch_characteristics)
        if replay_graph_key is None:
            raise RuntimeError(
                f"No captured device graph for {batch_characteristics}. "
                f"Available keys are {list(self._records.keys())}."
            )

        packed_model_graph_key = _pack_model_graph_key(replay_graph_key)
        captured_inputs, outputs = self.graph_entries[replay_graph_key]

        # Refresh captured inputs. Host-resident destinations copy inline; the
        # rest go into one batched call, which the driver splits into one
        # submit per destination device (cuMemcpyBatchAsync on CUDA 12.8+,
        # sequential fallback otherwise).
        dsts: list[Buffer] = []
        srcs: list[Buffer] = []
        for src_value, dst_value in zip(
            input_buffers, captured_inputs, strict=True
        ):
            if dst_value.device.is_host:
                dst_value.inplace_copy_from(src_value)
                continue
            assert src_value.device == dst_value.device, (
                "Graph-capture replay refresh must be a same-device copy "
                "(single-stream ordering is the correctness premise); "
                f"got src {src_value.device} -> dst {dst_value.device}."
            )
            dsts.append(dst_value)
            srcs.append(src_value)

        batch_inplace_copy(dsts, srcs)

        if debug_verify_replay:
            verify_inputs = debug_verify_model_inputs or model_inputs
            self._model.debug_verify_replay(
                packed_model_graph_key,
                *verify_inputs.buffers,
            )

        if isinstance(self._model, CompiledModel):
            self._model.engine_model.replay(
                packed_model_graph_key,
                *captured_inputs,
                *self._model.signal_buffers,
            )
        else:
            self._model.replay(packed_model_graph_key, *captured_inputs)
        return outputs
