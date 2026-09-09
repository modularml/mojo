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

"""Reference-counted LRU cache for vision encoder outputs.

Stores per-image encoder embeddings so the vision encoder runs once per
unique image, regardless of how many chunks or requests reference it.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt
from max.driver import (
    Buffer,
    Device,
    DevicePinnedBuffer,
    batch_inplace_copy,
    copy_pinned_to_destinations,
)
from max.dtype import DType
from max.pipelines.context import (
    ImageMetadata,
    TextAndVisionContext,
    TextContext,
    VLMContextType,
)
from max.pipelines.lib.vlm_utils import (
    compute_multimodal_merge_indices,
    compute_windowed_merge_indices,
)
from max.pipelines.request import RequestID
from max.profiler import traced

if TYPE_CHECKING:
    from max.pipelines.lib.interfaces.pipeline_model import ModelInputs

logger = logging.getLogger("max.pipelines")

DEFAULT_VISION_CACHE_BLOCK_TOKENS = 128
"""Rows per fixed-size block in the block-mode vision encoder cache.

Sized so a typical image (a few hundred merged tokens) spans a handful of
blocks while a video spans many; per-entry internal fragmentation is at most
one block.
"""


@dataclass(frozen=True)
class VisionCachePlan:
    """Resolved block-mode vision encoder cache reservation.

    Produced by memory estimation when a block budget is reserved and
    consumed by pipeline construction, which hands it to
    :class:`VisionEncoderCache` so the block pool allocates exactly what
    was reserved.
    """

    bytes_per_device: int
    """Bytes reserved on each device for its shard of the pool, rounded
    down to whole blocks. Total cache capacity is this value times the
    device count.
    """

    hidden_size: int
    """Width of one cached embedding row."""

    dtype: DType
    """Element type of one cached embedding row."""


def concat_device_buffers(bufs: list[Buffer]) -> Buffer:
    """Concatenate 2D Buffers along dim 0 on device.

    Each buffer must have shape ``[n_rows_i, hidden]`` on the same device
    and with the same dtype. Allocates a single output buffer
    ``[sum(n_rows_i), hidden]`` and copies each input slice into it via
    ``inplace_copy_from``.

    Used both internally by the vision encoder cache (per-image splits
    re-assembled into a batch-shaped output) and by VLM model code that
    runs the vision encoder in multiple chunks and needs to concat the
    per-chunk outputs back into a single per-device tensor before handing
    off to ``prepare_vision_outputs``.
    """
    assert len(bufs) > 0, "concat_device_buffers requires at least one buffer"
    first = bufs[0]
    hidden = int(first.shape[1])
    dtype = first.dtype
    device = first.device
    for b in bufs[1:]:
        assert b.dtype == dtype, (
            f"concat_device_buffers: dtype mismatch ({b.dtype} vs {dtype})"
        )
        assert b.device == device, (
            f"concat_device_buffers: device mismatch ({b.device} vs {device})"
        )
        assert int(b.shape[1]) == hidden, (
            f"concat_device_buffers: dim-1 mismatch "
            f"({int(b.shape[1])} vs {hidden})"
        )
    total_rows = sum(int(b.shape[0]) for b in bufs)
    out = Buffer(
        shape=[total_rows, hidden],
        dtype=dtype,
        device=device,
    )
    offset = 0
    for b in bufs:
        n = int(b.shape[0])
        out[offset : offset + n, :].inplace_copy_from(b)
        offset += n
    return out


def _owned_row_slice(src: Buffer, start: int, count: int) -> Buffer:
    """Copy ``src[start:start + count, :]`` into a freshly allocated Buffer.

    An owned copy (not a view) so a cache entry does not pin the variable-size
    vision-encoder output buffer, avoiding GPU allocator fragmentation.
    """
    slot = Buffer.zeros(
        shape=[count, int(src.shape[1])],
        dtype=src.dtype,
        device=src.device,
    )
    slot.inplace_copy_from(src[start : start + count, :])
    return slot


class _VisionBlockPool:
    """Fixed-size block storage backing the vision encoder cache.

    Storage is sharded: block ``b`` lives only on device ``b % n_devices``,
    as one preallocated shard Buffer per device, allocated at construction
    from the memory planner's row spec. A cached entry is stored once
    across the shards and gathered to every device's output on a hit.
    Capacity is a byte budget: an entry maps to a span of blocks, so a
    video spans many blocks and an image a few.
    """

    def __init__(
        self,
        budget_bytes_per_device: int,
        block_tokens: int,
        hidden_size: int,
        dtype: DType,
        devices: Sequence[Device],
    ) -> None:
        assert budget_bytes_per_device > 0
        assert block_tokens > 0
        row_bytes = hidden_size * dtype.size_in_bytes
        blocks_per_shard = budget_bytes_per_device // (block_tokens * row_bytes)
        if blocks_per_shard <= 0:
            raise ValueError(
                f"Vision cache byte budget {budget_bytes_per_device} cannot "
                f"fit one {block_tokens}-token block "
                f"({block_tokens * row_bytes} bytes) per device."
            )
        self._block_tokens = block_tokens
        self._hidden_size = hidden_size
        self._dtype = dtype
        self._num_blocks = int(blocks_per_shard) * len(devices)
        self._pools = [
            Buffer(
                shape=[int(blocks_per_shard) * block_tokens, hidden_size],
                dtype=dtype,
                device=device,
            )
            for device in devices
        ]
        self._free: list[int] = list(range(self._num_blocks))

    @property
    def block_tokens(self) -> int:
        return self._block_tokens

    @property
    def num_blocks(self) -> int:
        """Total blocks in the pool."""
        return self._num_blocks

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def blocks_for(self, num_tokens: int) -> int:
        """Blocks needed to hold ``num_tokens`` rows (ceiling division)."""
        return -(-num_tokens // self._block_tokens)

    def matches(self, reference: Buffer) -> bool:
        """Whether a buffer's rows match the pool's planner-provided spec."""
        return (
            int(reference.shape[1]) == self._hidden_size
            and reference.dtype == self._dtype
        )

    def allocate(self, n: int) -> list[int] | None:
        """Take ``n`` free blocks, or return None (no side effects) if short."""
        if n == 0:
            return []
        if n > len(self._free):
            return None
        ids = self._free[-n:]
        del self._free[-n:]
        return ids

    def free(self, block_ids: list[int]) -> None:
        self._free.extend(block_ids)

    def host_device_index(self, block_id: int) -> int:
        """Index of the device whose shard holds ``block_id``."""
        return block_id % len(self._pools)

    def _spans(
        self, block_ids: Sequence[int], row_lo: int, row_hi: int
    ) -> Iterator[tuple[int, int, int, int]]:
        """Yield ``(device_idx, pool_row, chunk_rows, slice_row)`` per block.

        ``pool_row`` is within the hosting device's shard; ``slice_row`` is
        relative to ``row_lo``.
        """
        bt = self._block_tokens
        n = len(self._pools)
        slice_row = 0
        for i in range(row_lo // bt, -(-row_hi // bt)):
            lo = max(row_lo, i * bt)
            hi = min(row_hi, (i + 1) * bt)
            b = block_ids[i]
            yield b % n, (b // n) * bt + (lo - i * bt), hi - lo, slice_row
            slice_row += hi - lo

    def write_rows(
        self,
        block_ids: Sequence[int],
        src: Sequence[Buffer],
        start: int,
        num_tokens: int,
    ) -> None:
        """Copy rows ``[start, start + num_tokens)`` into the given blocks.

        Every model's vision encode returns one output copy per device, so
        each block is written from its hosting device's local ``src`` —
        stores never cross devices.
        """
        assert self.matches(src[0]), (
            "vision encoder output does not match the arch config's "
            "get_vision_cache_row_spec"
        )
        dsts: list[Buffer] = []
        srcs: list[Buffer] = []
        for dev, base, chunk, row in self._spans(block_ids, 0, num_tokens):
            dsts.append(self._pools[dev][base : base + chunk, :])
            srcs.append(src[dev][start + row : start + row + chunk, :])
        batch_inplace_copy(dsts, srcs)

    def copy_out(
        self,
        out: Sequence[Buffer],
        out_row: int,
        block_ids: Sequence[int],
        row_lo: int,
        row_hi: int,
    ) -> None:
        """Gather entry rows ``[row_lo, row_hi)`` to every ``out[d]``.

        Rows on another device's shard arrive as peer copies; all pairs go
        in one batched submission per destination device.
        """
        pairs = list(self._spans(block_ids, row_lo, row_hi))
        dsts: list[Buffer] = []
        srcs: list[Buffer] = []
        for o in out:
            for dev, base, chunk, row in pairs:
                dsts.append(o[out_row + row : out_row + row + chunk, :])
                srcs.append(self._pools[dev][base : base + chunk, :])
        batch_inplace_copy(dsts, srcs)

    def rows_view(self, block_id: int, row_lo: int, row_hi: int) -> Buffer:
        """Block-relative zero-copy view of rows ``[row_lo, row_hi)``.

        The view lives on the block's hosting device
        (:meth:`host_device_index`).
        """
        n = len(self._pools)
        base = (block_id // n) * self._block_tokens
        return self._pools[block_id % n][base + row_lo : base + row_hi, :]


@dataclass
class VisionEncodeResult:
    """A model's vision-encoder output for the uncached images of a batch."""

    embeddings: list[Buffer]
    """Per-device encoder output, each ``[total_tokens, hidden]``.

    Rows are ordered context-major, image-minor.
    """

    per_image_token_counts: list[int] | None = None
    """Explicit per-image token counts, in row order.

    ``None`` lets the driver derive them from placeholder spans. A model whose
    span differs from its emitted row count must set this.
    """


PackedVisionInputsT = TypeVar("PackedVisionInputsT")


@runtime_checkable
class SupportsVisionEncoding(Protocol[PackedVisionInputsT]):
    """A pipeline model that encodes images in two declared steps.

    Caching is the driver's job: the model packs and encodes, the cache
    stores and assembles. ``PackedVisionInputsT`` is the model's packed-input
    type, carried from prep to encode.
    """

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> PackedVisionInputsT | None:
        """Pack the batch's selected cache-miss pixels to device, during prep.

        Optional: returning ``None`` defers packing to :meth:`vision_execute`.
        Runs in the prep phase so the host-to-device copy overlaps the prior
        batch. ``selection`` is the ``(context, miss-images)`` pairs from
        :meth:`VisionEncoderCache.select`.
        """
        ...

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: PackedVisionInputsT | None,
    ) -> VisionEncodeResult:
        """Run the vision encoder over the batch's selected cache-miss images.

        Uses ``packed`` when :meth:`pack_vision_inputs` packed, otherwise packs
        from ``selection`` inline. Returns per-device embeddings, context-major
        then image-minor; caching is the cache's job.
        """
        ...

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row embedding buffers for text-only/cached batches.

        The cache assembles from these when no image is encoded this step;
        the model owns the hidden size and dtype.
        """
        ...


@runtime_checkable
class SupportsPooledVisionMetrics(Protocol):
    """A pipeline model that owns its vision/video encoder cache internally.

    Most models encode images through :class:`SupportsVisionEncoding`, so the
    pipeline owns the :class:`VisionEncoderCache` and drains its metrics
    directly. A model that instead builds and drives its own cache (e.g. to
    handle video, which :class:`SupportsVisionEncoding` doesn't cover)
    implements this protocol so the pipeline's ``batch_vision_metrics``/
    ``batch_video_metrics`` can still reach it.
    """

    def pop_vision_metrics(self) -> VisionEncoderMetrics | None:
        """Returns and clears this model's per-batch image encoder metrics."""
        ...

    def pop_video_metrics(self) -> VideoEncoderMetrics | None:
        """Returns and clears this model's per-batch video encoder metrics."""
        ...


def derive_counts_from_spans(
    selection: Sequence[tuple[VLMContextType, Sequence[ImageMetadata]]],
) -> list[int]:
    """Per-image token counts from the cache's selection, in row order.

    Walks the ``(context, miss-images)`` pairs from
    :meth:`VisionEncoderCache.select` — so the counts equal the encoder's
    emitted rows by construction, in every mode (including a disabled cache
    under chunked prefill, where ``ctx.images`` would include already-processed
    images the encoder did not emit rows for).

    Args:
        selection: The ``(context, miss-images)`` pairs to encode this step.

    Returns:
        One token count per encoded image, in row order.
    """
    return [
        img.end_idx - img.start_idx
        for _ctx, miss_images in selection
        for img in miss_images
    ]


def validate_vision_encode_counts(
    per_image_token_counts: Sequence[int],
    embeddings: Sequence[Buffer],
) -> None:
    """Raise if the per-image counts don't sum to the encoder's row count.

    Guards the per-image cache split against a model whose placeholder span
    doesn't match its emitted rows.

    Args:
        per_image_token_counts: Per-image token counts, in row order.
        embeddings: Per-device encoder output; row count is from device 0.

    Raises:
        ValueError: On a count/row mismatch.
    """
    if not embeddings:
        return
    total_rows = int(embeddings[0].shape[0])
    total_counts = int(sum(per_image_token_counts))
    if total_counts != total_rows:
        raise ValueError(
            f"Vision encoder emitted {total_rows} row(s) but per-image token "
            f"counts sum to {total_counts}. The encoder must emit exactly one "
            "row per placeholder token; a model whose placeholder span "
            "includes non-scatter-target tokens must return explicit "
            "per_image_token_counts."
        )


@dataclass
class VisionEncoderCacheEntry:
    """Cached vision encoder output for a single image.

    Exactly one of ``embeddings`` / ``block_ids`` is set: owned per-device
    buffers in entry-count mode (and for transient or oversized-fallback
    entries), or a span of pool blocks in block mode.
    """

    embeddings: list[Buffer] | None
    """Per-device embeddings, each shape [num_tokens, hidden_size]."""

    num_tokens: int
    """Number of merged image tokens this entry covers."""

    ref_count: int = 0
    """Number of active requests referencing this entry."""

    block_ids: list[int] | None = None
    """Pool blocks holding this entry's rows, in row order (block mode)."""


@dataclass
class VisionEncoderMetrics:
    """Per-iteration vision encoder statistics for one batch.

    Populated by :class:`VisionEncoderCache` during batch preparation and
    surfaced by the scheduler in its per-iteration log so that vision
    encoder cost is attributed separately from the language model forward
    pass.
    """

    num_images_total: int = 0
    """Images referenced by vision requests in this batch (hits + misses)."""

    num_images_encoded: int = 0
    """Images the vision encoder actually ran on this batch (cache misses)."""

    num_images_cached: int = 0
    """Images served from the vision encoder cache this batch (cache hits)."""

    num_patches_encoded: int = 0
    """Input image patches fed to the vision encoder this batch."""

    num_tokens_encoded: int = 0
    """Merged vision tokens produced by the encoder this batch."""

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of images served from cache (0.0 when no images)."""
        if self.num_images_total == 0:
            return 0.0
        return self.num_images_cached / self.num_images_total


@dataclass
class VideoEncoderMetrics:
    """Per-iteration video encoder statistics for one batch.

    Populated by a pipeline model that caches video clips (by content hash,
    same LRU semantics as :class:`VisionEncoderCache`) and surfaced via
    :class:`SupportsPooledVisionMetrics` so video encoder cost is attributed
    separately from the language model forward pass, mirroring
    :class:`VisionEncoderMetrics`.
    """

    num_clips_total: int = 0
    """Clips referenced by video requests in this batch (hits + misses)."""

    num_clips_encoded: int = 0
    """Clips the video encoder actually ran on this batch (cache misses)."""

    num_clips_cached: int = 0
    """Clips served from the video encoder cache this batch (cache hits)."""

    frame_counts: list[int] = field(default_factory=list)
    """Sampled frame count per newly-encoded clip this batch (cache misses
    only), one entry per clip so callers can observe the distribution rather
    than only a batch-level total."""

    num_tokens_encoded: int = 0
    """Merged video tokens produced by the encoder this batch."""

    encoding_time_ms: float = 0.0
    """Wall-clock time spent running the video encoder this batch."""

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of clips served from cache (0.0 when no clips)."""
        if self.num_clips_total == 0:
            return 0.0
        return self.num_clips_cached / self.num_clips_total


class VisionEncoderCache(Generic[VLMContextType]):
    """Reference-counted LRU cache for vision encoder outputs.

    Stores per-image encoder embeddings so the vision encoder runs once
    per unique image, regardless of how many chunks or requests
    reference it. Storage is a byte budget carved into fixed-size blocks
    (:class:`_VisionBlockPool`): an entry maps to a span of blocks, a video
    spans many and an image a few, and LRU eviction frees whole entries on
    block demand.

    Typical usage is pipeline-owned, via the seam::

        result = cache.run_vision_encode(model, replica_batches, devices)
        cache.finalize_vision_inputs(model, model_inputs, devices, result)
    """

    def __init__(
        self,
        plan: VisionCachePlan | None = None,
        devices: Sequence[Device] | None = None,
        block_tokens: int = DEFAULT_VISION_CACHE_BLOCK_TOKENS,
    ) -> None:
        """Initializes the cache, allocating its block pool up front.

        Args:
            plan: Resolved block reservation from memory estimation;
                ``None`` disables caching.
            devices: Devices to allocate one pool shard on each. Required
                when a plan is given.
            block_tokens: Rows per fixed-size block.

        Raises:
            ValueError: If a plan is given without ``devices``, or its
                budget cannot fit one block.
        """
        self._cache: OrderedDict[int, VisionEncoderCacheEntry] = OrderedDict()
        self._request_refs: defaultdict[RequestID, set[int]] = defaultdict(set)

        self._empty_indices_cache: list[Buffer] | None = None
        self._scatter_buffers: dict[int, list[Buffer]] = {}

        self._pool: _VisionBlockPool | None = None
        if plan is not None:
            if devices is None:
                raise ValueError(
                    "An enabled vision encoder cache requires devices."
                )
            self._pool = _VisionBlockPool(
                plan.bytes_per_device,
                block_tokens,
                plan.hidden_size,
                plan.dtype,
                devices,
            )
        self._n_devices = len(devices) if devices is not None else 1

        # Per-batch vision encoder metrics, populated during batch
        # preparation and drained by the scheduler once per iteration via
        # ``pop_metrics``. ``None`` when the most recent batch did no vision
        # work (e.g. a text-only or decode step).
        self._batch_metrics: VisionEncoderMetrics | None = None

    @traced
    def lookup(self, image_hash: int) -> VisionEncoderCacheEntry | None:
        """Look up a cached entry by image hash, refreshing LRU order.

        A falsy hash (``0``, the sentinel for an image/video with no
        content hash) is treated as a miss, so callers don't need to guard
        the call themselves.
        """
        if not image_hash:
            return None
        entry = self._cache.get(image_hash)
        if entry is not None:
            self._cache.move_to_end(image_hash)
        return entry

    @property
    def enabled(self) -> bool:
        """Whether caching is enabled (a positive block byte budget)."""
        return self._pool is not None

    @traced
    def insert(
        self,
        image_hash: int,
        embeddings: list[Buffer],
        num_tokens: int,
    ) -> VisionEncoderCacheEntry:
        """Insert a new cache entry. Returns existing entry if already cached.

        When the cache is disabled (zero byte budget), creates a transient
        entry without storing it.
        """
        if image_hash in self._cache:
            self._cache.move_to_end(image_hash)
            return self._cache[image_hash]
        entry = VisionEncoderCacheEntry(
            embeddings=embeddings,
            num_tokens=num_tokens,
        )
        if not self.enabled:
            return entry
        block_entry = self._store_blocks(embeddings, 0, num_tokens)
        if block_entry is not None:
            entry = block_entry
        self._cache[image_hash] = entry
        return entry

    @traced
    def acquire(self, request_id: RequestID, image_hash: int) -> None:
        """Increment ref count for a (request, image) pair."""
        refs = self._request_refs[request_id]
        if image_hash in refs:
            return  # already acquired for this request
        entry = self._cache.get(image_hash)
        if entry is not None:
            entry.ref_count += 1
        refs.add(image_hash)

    def _release_ref(self, request_id: RequestID, image_hash: int) -> None:
        """Release one (request, image) ref, if held."""
        refs = self._request_refs.get(request_id)
        if refs is None or image_hash not in refs:
            return
        refs.discard(image_hash)
        entry = self._cache.get(image_hash)
        if entry is not None:
            entry.ref_count = max(0, entry.ref_count - 1)

    @traced
    def release_request(self, request_id: RequestID) -> None:
        """Release all cache refs held by a request.

        Also reconciles occupancy with capacity. Storing never fails: when
        the pool's blocks are all ref-held, an entry falls back to an owned
        buffer outside the pool. Without draining here, that overshoot
        would linger until the next cache miss reaches the store path --
        indefinitely under cache-hit-only or text-only traffic. Draining on
        release bounds it to the lifetime of the requests that caused it.
        """
        for h in self._request_refs.pop(request_id, set()):
            entry = self._cache.get(h)
            if entry is not None:
                entry.ref_count = max(0, entry.ref_count - 1)
        for key in list(self._cache.keys()):
            entry = self._cache[key]
            if entry.ref_count == 0 and entry.embeddings is not None:
                del self._cache[key]

    def _evict_lru(self) -> bool:
        """Evict the least-recently-used entry with ref_count == 0."""
        for key in list(self._cache.keys()):
            entry = self._cache[key]
            if entry.ref_count == 0:
                if entry.block_ids is not None:
                    assert self._pool is not None
                    self._pool.free(entry.block_ids)
                del self._cache[key]
                return True
        return False

    def _evict_until_free(self, needed: int) -> bool:
        """Evict zero-ref LRU entries until ``needed`` pool blocks are free."""
        assert self._pool is not None
        if needed > self._pool.num_blocks:
            return False
        while self._pool.num_free_blocks < needed:
            if not self._evict_lru():
                return False
        return True

    def _store_blocks(
        self, src: Sequence[Buffer], start: int, count: int
    ) -> VisionEncoderCacheEntry | None:
        """Store rows ``[start, start + count)`` of ``src`` into pool blocks.

        Returns None when the entry cannot fit — larger than the whole pool,
        or not enough blocks reclaimable behind referenced entries — and
        callers fall back to an owned entry so an active image is never
        dropped.
        TODO(SERVOPT-1530): once the scheduler gates admission on
        ``blocks_needed``, make this a hard error instead of a fallback.
        """
        pool = self._pool
        assert pool is not None
        needed = pool.blocks_for(count)
        if not self._evict_until_free(needed):
            logger.warning(
                "Vision cache cannot free %d block(s) for a %d-token entry "
                "(%d total, %d free); storing an owned buffer outside the "
                "block pool.",
                needed,
                count,
                pool.num_blocks,
                pool.num_free_blocks,
            )
            return None
        block_ids = pool.allocate(needed)
        assert block_ids is not None
        pool.write_rows(block_ids, src, start, count)
        return VisionEncoderCacheEntry(
            embeddings=None, num_tokens=count, block_ids=block_ids
        )

    @staticmethod
    def _ensure_image_hashes(
        ctx: TextAndVisionContext,
    ) -> None:
        """Assert that all images have pre-computed hashes.

        The tokenizer must compute image_hash when vision caching is
        enabled.
        """
        for img in ctx.images:
            if img.image_hash is None:
                raise ValueError(
                    "image_hash must be set by the tokenizer when "
                    "vision caching is enabled"
                )

    @traced
    def get_uncached_contexts(
        self,
        context_batch: Sequence[VLMContextType],
    ) -> list[VLMContextType]:
        """Return contexts that have at least one uncached image.

        Contexts where every image is already cached get their refs
        acquired and are excluded.  For partial hits (some cached, some
        not), refs for the cached images are acquired immediately and
        the context is returned.

        Refs for images whose tokens are fully processed are released
        here: their embeddings live on in the KV cache, so an entry only
        needs to survive until its consuming chunk completes.

        Callers can check ``self.lookup(img.image_hash)`` to distinguish
        cached from uncached images within the returned contexts.

        Raises ``ValueError`` if any image is missing its hash.
        """
        uncached_contexts: list[VLMContextType] = []

        for ctx in context_batch:
            if not getattr(ctx, "needs_vision_encoding", False):
                continue

            if not self.enabled:
                uncached_contexts.append(ctx)
                continue

            self._ensure_image_hashes(ctx)

            cached_in_ctx: list[int] = []
            has_uncached = False

            for img in ctx.images:
                assert img.image_hash is not None
                if img.end_idx <= ctx.tokens.processed_length:
                    # Evictable: these tokens now live in the KV cache.
                    self._release_ref(ctx.request_id, img.image_hash)
                    continue
                if self.lookup(img.image_hash) is not None:
                    cached_in_ctx.append(img.image_hash)
                else:
                    has_uncached = True

            for h in cached_in_ctx:
                self.acquire(ctx.request_id, h)
            if has_uncached:
                uncached_contexts.append(ctx)

        return uncached_contexts

    def pop_metrics(self) -> VisionEncoderMetrics | None:
        """Return the metrics for the most recent batch and reset them.

        Returns ``None`` when the most recent batch preparation did no
        vision encoding (text-only or decode step). Intended to be called
        once per scheduler iteration.
        """
        metrics = self._batch_metrics
        self._batch_metrics = None
        return metrics

    @property
    def total_num_blocks(self) -> int:
        """Total pool blocks (0 in entry-count mode or before lazy init)."""
        return self._pool.num_blocks if self._pool is not None else 0

    @property
    def num_free_or_evictable_blocks(self) -> int:
        """Blocks free now plus blocks held by unreferenced entries.

        The admission-probe view for a vision-aware scheduler: an allocation
        of at most this many blocks is guaranteed to succeed.
        """
        if self._pool is None:
            return 0
        evictable = sum(
            len(entry.block_ids)
            for entry in self._cache.values()
            if entry.ref_count == 0 and entry.block_ids is not None
        )
        return self._pool.num_free_blocks + evictable

    def blocks_needed(self, images: Sequence[ImageMetadata]) -> int:
        """Pool blocks required to cache the not-yet-resident ``images``.

        A pure probe: touches neither LRU order nor ref counts.
        """
        if self._pool is None:
            return 0
        return sum(
            self._pool.blocks_for(img.end_idx - img.start_idx)
            for img in images
            if img.image_hash is None or img.image_hash not in self._cache
        )

    @traced
    def _cache_and_split(
        self,
        vision_outputs: list[Buffer],
        per_image_token_counts: list[int],
        image_hashes: list[int],
        request_ids: list[RequestID],
    ) -> None:
        """Split concatenated encoder output per-image and store each in cache.

        Args:
            vision_outputs: Per-device tensors, each [total_tokens, hidden].
            per_image_token_counts: Number of tokens per image.
            image_hashes: Content hash per image.
            request_ids: Request ID per image.
        """
        offset = 0
        for count, img_hash, req_id in zip(
            per_image_token_counts, image_hashes, request_ids, strict=True
        ):
            start = offset
            offset += count
            # acquire for it, but still advance past its tokens in the encoder
            # output (this method only populates the cache for future reuse;
            # the current forward uses the encoder output directly, so skipping
            # is output-neutral).
            if not img_hash:
                continue
            if img_hash in self._cache:
                self._cache.move_to_end(img_hash)
            else:
                entry = self._store_blocks(vision_outputs, start, count)
                if entry is None:
                    entry = VisionEncoderCacheEntry(
                        embeddings=[
                            _owned_row_slice(dev_tensor, start, count)
                            for dev_tensor in vision_outputs
                        ],
                        num_tokens=count,
                    )
                self._cache[img_hash] = entry
            self.acquire(req_id, img_hash)

    @traced
    def prepare_vision_outputs(
        self,
        context_batch: Sequence[VLMContextType],
        uncached_contexts: Sequence[VLMContextType],
        uncached_images: Sequence[Sequence[ImageMetadata]],
        vision_embeds: list[Buffer],
        per_image_token_counts: list[int],
        n_devices: int,
        empty_embeddings: list[Buffer],
    ) -> tuple[list[Buffer], npt.NDArray[np.int32]]:
        """Store encoder output, assemble embeddings, and compute scatter indices.

        Only images not already in the cache are expected in
        *vision_embeds*.  Images that were already cached (partial hits)
        are skipped automatically.

        Args:
            context_batch: Full batch of contexts (cached + uncached).
            uncached_contexts: Subset from ``get_uncached_contexts``.
            uncached_images: The cache-miss images per ``uncached_contexts``
                entry (the single source of the encode selection), aligned with
                the concatenation order of *vision_embeds*.
            vision_embeds: Per-device encoder output for uncached images.
            per_image_token_counts: Tokens per uncached image, matching
                the concatenation order of *vision_embeds*.
            n_devices: Number of devices.
            empty_embeddings: Empty per-device buffers for text-only batches.

        Returns:
            ``(embeddings, indices)`` — per-device buffers and a 1-D
            int32 scatter-index array.
        """
        metrics = VisionEncoderMetrics()
        for ctx in context_batch:
            if not getattr(ctx, "needs_vision_encoding", False):
                continue
            for img in ctx.images:
                metrics.num_images_total += 1
                if (
                    self.enabled
                    and img.image_hash is not None
                    and self.lookup(img.image_hash) is not None
                ):
                    metrics.num_images_cached += 1
        if metrics.num_images_total > 0:
            for miss_images in uncached_images:
                for img in miss_images:
                    metrics.num_images_encoded += 1
                    metrics.num_patches_encoded += int(
                        img.pixel_values.shape[0]
                    )
                    metrics.num_tokens_encoded += img.end_idx - img.start_idx
            self._batch_metrics = metrics
        else:
            self._batch_metrics = None

        if not self.enabled:
            embeddings = (
                vision_embeds if uncached_contexts else empty_embeddings
            )
            indices = compute_multimodal_merge_indices(context_batch)
            return embeddings, indices

        if not per_image_token_counts:
            # All images cached or text-only — assemble from cache,
            # no sync or slicing needed.
            embeddings = self._assemble_embeddings(
                context_batch, n_devices, empty_embeddings
            )
            indices = compute_windowed_merge_indices(context_batch)
            return embeddings, indices

        # Record an event and synchronize so the vision encoder output
        # is visible before we slice or cache it.
        for buf in vision_embeds:
            if not buf.is_host:
                buf.device.default_queue.record_event().synchronize()

        hashes: list[int] = []
        req_ids: list[RequestID] = []
        all_uncached = True
        for ctx, miss_images in zip(
            uncached_contexts, uncached_images, strict=True
        ):
            for img in miss_images:
                assert img.image_hash is not None
                hashes.append(img.image_hash)
                req_ids.append(ctx.request_id)
            if len(miss_images) != len(ctx.images):
                all_uncached = False  # a partial hit in this context

        self._cache_and_split(
            vision_embeds, per_image_token_counts, hashes, req_ids
        )

        # A straddling image is encoded in full but indexed only in-window,
        # so it must take the assembly path.
        n_vision = sum(
            1
            for ctx in context_batch
            if getattr(ctx, "needs_vision_encoding", False)
        )
        fully_in_window = all(
            ctx.tokens.processed_length <= img.start_idx
            and img.end_idx <= ctx.tokens.current_position
            for ctx in uncached_contexts
            for img in ctx.images
        )
        if (
            len(uncached_contexts) == n_vision
            and all_uncached
            and fully_in_window
        ):
            embeddings = vision_embeds
        else:
            embeddings = self._assemble_embeddings(
                context_batch, n_devices, empty_embeddings
            )

        indices = compute_windowed_merge_indices(context_batch)
        return embeddings, indices

    @traced
    def _assemble_embeddings(
        self,
        context_batch: Sequence[VLMContextType],
        n_devices: int,
        empty_embeddings: list[Buffer],
    ) -> list[Buffer]:
        """Build the active-window image_embeddings tensor from cache.

        Emits the rows whose placeholder positions fall inside each
        context's window ``[processed_length, current_position)``, ordered
        to match
        :func:`~max.pipelines.lib.vlm_utils.compute_windowed_merge_indices`.
        Call after ``_cache_and_split()`` so in-window images are resident.

        Returns:
            Per-device buffers, each ``[window_image_tokens, hidden_size]``.
        """
        spans: list[tuple[VisionEncoderCacheEntry, int, int]] = []
        for ctx in context_batch:
            if not getattr(ctx, "needs_vision_encoding", False):
                continue
            win_lo = ctx.tokens.processed_length
            win_hi = ctx.tokens.current_position
            positions = np.asarray(ctx.image_token_indices)
            for img in ctx.images:
                if img.end_idx <= win_lo or img.start_idx >= win_hi:
                    continue
                img_lo = int(np.searchsorted(positions, img.start_idx))
                img_positions = positions[img_lo : img_lo + img.embedding_rows]
                row_lo = int(np.searchsorted(img_positions, win_lo))
                row_hi = int(np.searchsorted(img_positions, win_hi))
                if row_lo == row_hi:
                    continue
                assert img.image_hash is not None
                entry = self.lookup(img.image_hash)
                assert entry is not None, (
                    f"Active in-window image {img.image_hash} not in cache"
                )
                spans.append((entry, row_lo, row_hi))

        if not spans:
            return empty_embeddings

        # The entry's ref count keeps these rows resident while the
        # forward reads them.
        if len(spans) == 1:
            entry, row_lo, row_hi = spans[0]
            if entry.embeddings is not None:
                if row_lo == 0 and row_hi == entry.num_tokens:
                    return entry.embeddings
                return [e[row_lo:row_hi, :] for e in entry.embeddings]
            assert entry.block_ids is not None
            assert self._pool is not None
            bt = self._pool.block_tokens
            if row_lo // bt == (row_hi - 1) // bt:
                block_id = entry.block_ids[row_lo // bt]
                lo = row_lo % bt
                view = self._pool.rows_view(
                    block_id, lo, lo + (row_hi - row_lo)
                )
                if n_devices == 1:
                    return [view]
                host = self._pool.host_device_index(block_id)
                outs: list[Buffer] = []
                dsts: list[Buffer] = []
                for d, base in enumerate(empty_embeddings):
                    if d == host:
                        outs.append(view)
                        continue
                    buf = Buffer(
                        shape=[row_hi - row_lo, int(base.shape[1])],
                        dtype=base.dtype,
                        device=base.device,
                    )
                    outs.append(buf)
                    dsts.append(buf)
                batch_inplace_copy(dsts, [view] * len(dsts))
                return outs

        total_rows = sum(hi - lo for _, lo, hi in spans)
        out = [
            Buffer(
                shape=[total_rows, int(base.shape[1])],
                dtype=base.dtype,
                device=base.device,
            )
            for base in empty_embeddings
        ]
        row = 0
        for entry, row_lo, row_hi in spans:
            count = row_hi - row_lo
            if entry.embeddings is not None:
                for d in range(n_devices):
                    out[d][row : row + count, :].inplace_copy_from(
                        entry.embeddings[d][row_lo:row_hi, :]
                    )
            else:
                assert entry.block_ids is not None
                assert self._pool is not None
                self._pool.copy_out(out, row, entry.block_ids, row_lo, row_hi)
            row += count
        return out

    @traced
    def select(
        self, context_batch: Sequence[VLMContextType]
    ) -> list[tuple[VLMContextType, list[ImageMetadata]]]:
        """Select contexts to encode, each paired with its cache-miss images.

        Computes the cache-miss set once (over ``ctx.next_images_in_window``),
        acquires refs for already-cached images immediately (so a hit can't be
        evicted between selection and assembly), and returns each selected
        context paired with its miss images. Every downstream consumer reads
        that same returned selection: the model's pack/encode steps, the counts
        (:func:`derive_counts_from_spans`), and the store/split
        (``prepare_vision_outputs``, which records the batch metrics).

        Only misses overlapping the active window are selected, so each
        encoder forward is bounded by the scheduler's chunked-prefill window:
        an image fully ahead of the window is encoded by the iteration whose
        chunk covers it; this iteration its rows are zero-filled and its
        scatter positions OOB-masked.

        ``get_uncached_contexts`` scans ``ctx.images`` (all images) rather than
        ``next_images`` to decide which contexts are candidates, so a context
        can be a candidate solely because of an uncached image ahead of the
        window (e.g. a later chunk's image) while every in-window image is a
        cache hit. Such contexts have an empty miss set and are excluded from
        the returned selection: there is nothing to encode for them this
        iteration, and assembly reads their in-window hits from the cache and
        zero-fills out-of-window images from ``context_batch`` directly.
        Fully-processed images (in ``ctx.images`` but not ``next_images``)
        are skipped by that scan and their refs released — their embeddings
        live on in the KV cache, and assembly zero-fills their rows if the
        entry is later evicted.
        """
        uncached = self.get_uncached_contexts(context_batch)
        selection: list[tuple[VLMContextType, list[ImageMetadata]]] = []
        for ctx in uncached:
            misses = [
                img
                for img in ctx.next_images_in_window
                if img.image_hash is None or self.lookup(img.image_hash) is None
            ]
            if misses:
                selection.append((ctx, misses))
        return selection

    @traced
    def cache_vision_embeddings(
        self,
        context_batch: Sequence[VLMContextType],
        selection: Sequence[tuple[VLMContextType, Sequence[ImageMetadata]]],
        encode_result: VisionEncodeResult,
        empty_embeddings: list[Buffer],
    ) -> tuple[list[Buffer], npt.NDArray[np.int32]]:
        """Resolve/validate token counts, then store and assemble embeddings.

        Uses ``encode_result.per_image_token_counts`` when set, else derives
        them from placeholder spans (skipping images already resident).

        Returns:
            ``(embeddings, scatter_indices)`` — per-device buffers and a 1-D
            int32 scatter-index array.
        """
        counts = encode_result.per_image_token_counts
        if counts is None:
            counts = derive_counts_from_spans(selection)
        validate_vision_encode_counts(counts, encode_result.embeddings)
        result = self.prepare_vision_outputs(
            context_batch=context_batch,
            uncached_contexts=[ctx for ctx, _ in selection],
            uncached_images=[list(miss) for _, miss in selection],
            vision_embeds=encode_result.embeddings,
            per_image_token_counts=counts,
            n_devices=self._n_devices,
            empty_embeddings=empty_embeddings,
        )
        return result

    @traced
    def run_vision_encode(
        self,
        model: SupportsVisionEncoding[PackedVisionInputsT],
        replica_batches: Sequence[Sequence[VLMContextType]],
        devices: list[Device],
    ) -> tuple[list[Buffer], npt.NDArray[np.int32]] | None:
        """Drive one batch's vision encode + cache assembly (pipeline-owned).

        Runs between prep and the language forward: select the uncached images,
        pack them (prep already ran, so this is the pixel h2d), run the encoder,
        then store and assemble. Returns ``(embeddings, scatter_indices)`` for a
        batch that has vision this step, or ``None`` for a decode / text-only
        step (the model uses its own empties).
        """
        context_batch = [ctx for replica in replica_batches for ctx in replica]
        if not any(ctx.needs_vision_encoding for ctx in context_batch):
            return None
        empty = model.empty_vision_embeddings(devices)
        selection = self.select(context_batch)
        if selection:
            packed = model.pack_vision_inputs(selection, devices)
            result = model.vision_execute(selection, devices, packed)
        else:
            result = VisionEncodeResult(embeddings=empty)
        return self.cache_vision_embeddings(
            context_batch, selection, result, empty
        )

    @traced
    def finalize_vision_inputs(
        self,
        model: SupportsVisionEncoding[PackedVisionInputsT],
        model_inputs: ModelInputs,
        devices: list[Device],
        vision_result: tuple[list[Buffer], npt.NDArray[np.int32]] | None,
    ) -> None:
        """Sets the ABI-facing vision-merge inputs on ``model_inputs``.

        The single place empties-vs-real is decided: after this call the base
        ``vision_embeddings`` / ``vision_scatter_indices`` fields hold the
        per-device buffers the language graph declares, and
        ``model_inputs.buffers`` is packable. Must run on every prepared batch
        of a vision-capable model — including graph-capture warmup, which
        packs ``.buffers`` without going through ``execute()``.

        Args:
            model: The vision-capable pipeline model (owns the empty
                embeddings, whose hidden size and dtype are model-specific).
            model_inputs: The prepared inputs, finalized in place.
            devices: The pipeline's devices.
            vision_result: What :meth:`run_vision_encode` returned for this
                batch — an ``(embeddings, merge_indices)`` pair of assembled
                per-device embedding buffers and the host merge-index array
                to copy to each device. ``None`` means nothing was encoded
                this step (a decode or text-only batch), so the fields fall
                back to the model's cached zero-row empties.
        """
        if vision_result is None:
            model_inputs.vision_embeddings = model.empty_vision_embeddings(
                devices
            )
            model_inputs.vision_scatter_indices = self._empty_indices(devices)
            return
        embeddings, scatter_np = vision_result
        model_inputs.vision_embeddings = embeddings
        if len(scatter_np) == 0:
            model_inputs.vision_scatter_indices = self._empty_indices(devices)
        else:
            model_inputs.vision_scatter_indices = self._scatter_to_devices(
                scatter_np, devices
            )

    def _empty_indices(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-length merge-index buffers.

        Cached: hit on every decode / text-only step, so it must not allocate
        per call.
        """
        if self._empty_indices_cache is None:
            self._empty_indices_cache = [
                Buffer.zeros(shape=[0], dtype=DType.int32).to(dev)
                for dev in devices
            ]
        return self._empty_indices_cache

    @traced
    def _scatter_to_devices(
        self, scatter_np: npt.NDArray[np.int32], devices: list[Device]
    ) -> list[Buffer]:
        """Copy merge indices to each device.

        Allocates a fresh pinned host buffer every call and never reuses it
        across calls: under the overlap scheduler a reused pinned buffer would
        be clobbered by the next step's host write while the current step's
        asynchronous H2D copy is still reading it. The per-device destination
        buffers are cached by index count and reused (never pinned).
        """
        dev = devices[0]
        n = len(scatter_np)
        host_buffer_cls = DevicePinnedBuffer if not dev.is_host else Buffer
        host: Buffer = host_buffer_cls(
            dtype=DType.int32, shape=(n,), device=dev
        )

        device_bufs = self._scatter_buffers.get(n)
        if device_bufs is None:
            device_bufs = [
                Buffer(shape=(n,), dtype=DType.int32, device=d) for d in devices
            ]
            self._scatter_buffers[n] = device_bufs

        host.to_numpy()[:] = scatter_np.astype(np.int32)
        copy_pinned_to_destinations(host, device_bufs)
        return device_bufs


def as_vision_context_batches(
    replica_batches: Sequence[Sequence[TextContext]],
) -> list[list[TextAndVisionContext]]:
    """Narrow generic pipeline batches to vision contexts (runtime-checked).

    The text-generation pipelines are generic over ``TextContext``; the vision
    drive only runs for VLM models, whose contexts are all
    ``TextAndVisionContext``. Asserts that invariant rather than casting it.
    """
    narrowed: list[list[TextAndVisionContext]] = []
    for batch in replica_batches:
        vision_batch: list[TextAndVisionContext] = []
        for ctx in batch:
            assert isinstance(ctx, TextAndVisionContext)
            vision_batch.append(ctx)
        narrowed.append(vision_batch)
    return narrowed
