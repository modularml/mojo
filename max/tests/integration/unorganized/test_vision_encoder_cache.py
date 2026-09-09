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

"""Unit tests for VisionEncoderCache."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.pipelines.context import (
    GenerationStatus,
    GrammarEnforcementSnapshot,
    GrammarMatcher,
    ImageMetadata,
    LogProbabilities,
    SamplingParams,
    SpecDecodingState,
    TextAndVisionContext,
    TextGenerationOutput,
)
from max.pipelines.context.context import TokenBuffer
from max.pipelines.context.eos_tracking import EOSTracker
from max.pipelines.lib.interfaces.pipeline_model import ModelInputs
from max.pipelines.lib.vision_encoder_cache import (
    SupportsVisionEncoding,
    VideoEncoderMetrics,
    VisionCachePlan,
    VisionEncoderCache,
    VisionEncodeResult,
    derive_counts_from_spans,
    validate_vision_encode_counts,
)
from max.pipelines.lib.vlm_utils import (
    compute_multimodal_merge_indices,
    compute_windowed_merge_indices,
)
from max.pipelines.request import RequestID


def _make_buffer(rows: int, cols: int = 4) -> Buffer:
    """Create a host Buffer with deterministic data."""
    arr = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    return Buffer.from_numpy(arr)


def _make_image_meta(
    start: int, end: int, image_hash: int | None = None
) -> ImageMetadata:
    """Create an ImageMetadata with minimal pixel data."""
    return ImageMetadata(
        start_idx=start,
        end_idx=end,
        pixel_values=np.zeros((1, 3), dtype=np.float32),
        image_hash=image_hash,
    )


def _make_token_buffer(
    total_length: int, processed_length: int = 0
) -> TokenBuffer:
    """Create a TokenBuffer of *total_length* tokens with *processed_length* already processed.

    The active window covers ``[processed_length, total_length)``.
    """
    buf = TokenBuffer(np.zeros(total_length, dtype=np.int64))
    if processed_length > 0:
        # skip_processing advances the active-window start, which sets processed_length.
        buf.skip_processing(processed_length)
    return buf


class FakeContext:
    """Test context implementing TextAndVisionContext."""

    def __init__(
        self,
        request_id: RequestID,
        images: list[ImageMetadata] | None = None,
        needs_vision: bool = True,
        image_token_indices: npt.NDArray[np.int32] | None = None,
        processed_length: int = 0,
        active_length: int = 0,
        next_images: list[ImageMetadata] | None = None,
    ) -> None:
        self._eos_tracker = EOSTracker()
        self._request_id = request_id
        self.images: list[ImageMetadata] = images or []
        self._next_images_override = next_images
        self._needs_vision = needs_vision
        self.status = GenerationStatus.ACTIVE
        self.image_token_indices: npt.NDArray[np.int32] = (
            image_token_indices
            if image_token_indices is not None
            else np.empty(0, dtype=np.int32)
        )
        total_length = processed_length + active_length
        self.tokens: TokenBuffer = _make_token_buffer(
            max(total_length, 1), processed_length
        )
        self.cached_prefix_length: int | None = None
        self.in_reasoning_phase: bool = False
        self.grammar_enforced: bool = False
        self.tools_forced: bool = False
        self.requires_structured_output_flag: bool = False

    @property
    def request_id(self) -> RequestID:
        return self._request_id

    @property
    def image_idx(self) -> int:
        return 0 if self._needs_vision else len(self.images)

    @property
    def needs_vision_encoding(self) -> bool:
        return self._needs_vision

    @property
    def next_images(self) -> list[ImageMetadata]:
        if not self._needs_vision:
            return []
        if self._next_images_override is not None:
            return self._next_images_override
        return self.images

    @property
    def next_images_in_window(self) -> list[ImageMetadata]:
        return self.next_images

    def compute_image_aligned_idx(self, idx: int) -> int:
        return idx

    @property
    def eos_tracker(self) -> EOSTracker:
        return self._eos_tracker

    @property
    def max_length(self) -> int | None:
        return None

    def reset(self) -> None:
        pass

    def compute_num_available_steps(self, max_seq_len: int) -> int:
        return 0

    @property
    def min_tokens(self) -> int:
        return 0

    @property
    def log_probabilities(self) -> int:
        return 0

    @property
    def log_probabilities_echo(self) -> bool:
        return False

    def get_min_token_logit_mask(
        self, num_steps: int
    ) -> list[npt.NDArray[np.int32]]:
        return []

    def advance_token_buffer(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        pass

    def advance_fsm(self, token: int) -> bool:
        return False

    def update(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        pass

    def update_with_future_token(self) -> None:
        pass

    def realize_future_token(
        self, new_token: int, log_probabilities: LogProbabilities | None = None
    ) -> None:
        pass

    @property
    def matcher(self) -> GrammarMatcher | None:
        return None

    @property
    def json_schema(self) -> str | None:
        return None

    @property
    def grammar(self) -> str | None:
        return None

    def set_matcher(self, matcher: GrammarMatcher) -> None:
        pass

    def set_tool_region(
        self,
        start_token_ids: list[int] | None,
        end_token_ids: list[int] | None,
    ) -> None:
        pass

    def set_thinking_region(
        self,
        start_token_ids: list[int] | None,
        end_token_ids: list[int] | None,
    ) -> None:
        pass

    def update_enforcement_state(self, token: int) -> bool:
        return False

    def snapshot_grammar_state(self) -> GrammarEnforcementSnapshot:
        return GrammarEnforcementSnapshot(
            in_thinking_region=False,
            grammar_enforced=False,
            tool_calling_match_buffer=[],
            thinking_match_buffer=[],
        )

    def restore_grammar_state(
        self, snapshot: GrammarEnforcementSnapshot
    ) -> None:
        pass

    @property
    def sampling_params(self) -> SamplingParams:
        return SamplingParams()

    @property
    def is_initial_prompt(self) -> bool:
        return True

    def to_generation_output(self) -> TextGenerationOutput:
        raise NotImplementedError

    @property
    def spec_decoding_state(self) -> SpecDecodingState:
        return SpecDecodingState()

    @property
    def is_done(self) -> bool:
        return self.status.is_done


def _as_vlm_batch(
    contexts: list[FakeContext],
) -> list[TextAndVisionContext]:
    """Cast FakeContext test doubles for VLM cache APIs."""
    return cast(list[TextAndVisionContext], contexts)


def _as_selection(
    selection: list[tuple[FakeContext, list[ImageMetadata]]],
) -> list[tuple[TextAndVisionContext, list[ImageMetadata]]]:
    """Cast (FakeContext, miss-images) pairs for VLM cache APIs."""
    return cast(
        list[tuple[TextAndVisionContext, list[ImageMetadata]]], selection
    )


def _compute_merge_indices(
    contexts: list[FakeContext],
) -> npt.NDArray[np.int32]:
    """Call merge-index helper with FakeContext test doubles."""
    return compute_multimodal_merge_indices(_as_vlm_batch(contexts))


def _ref_count(
    cache: VisionEncoderCache[TextAndVisionContext], image_hash: int
) -> int:
    """Helper to get ref_count, asserting entry exists."""
    entry = cache.lookup(image_hash)
    assert entry is not None, f"Expected cache entry for {image_hash:#x}"
    return entry.ref_count


def _make_cache() -> VisionEncoderCache[TextAndVisionContext]:
    """Create a cache for testing.

    Tiny blocks with a budget far larger than any test's data, so entries
    are block-backed and nothing is ever evicted unintentionally.
    """
    return VisionEncoderCache(
        plan=VisionCachePlan(
            bytes_per_device=1024 * 1024,
            hidden_size=4,
            dtype=DType.float32,
        ),
        devices=[CPU()],
        block_tokens=4,
    )


def _miss_images(
    cache: VisionEncoderCache[TextAndVisionContext],
    uncached_contexts: list[TextAndVisionContext],
) -> list[list[ImageMetadata]]:
    """Cache-miss images per context — the selection prepare_vision_outputs needs."""
    return [
        [
            img
            for img in ctx.images
            if img.image_hash is None or cache.lookup(img.image_hash) is None
        ]
        for ctx in uncached_contexts
    ]


def _make_cache_sized(
    n_entries: int,
) -> VisionEncoderCache[TextAndVisionContext]:
    """Create a cache with room for exactly ``n_entries`` one-token entries.

    Callers insert 1-token hidden-4 float32 buffers (16 bytes/row); with
    ``block_tokens=1`` each insert takes exactly one 16-byte block, so
    entry-count LRU semantics carry over to block mode. ``n_entries=0``
    yields a zero budget, i.e. a disabled cache.
    """
    if n_entries == 0:
        return VisionEncoderCache()
    return VisionEncoderCache(
        plan=VisionCachePlan(
            bytes_per_device=n_entries * 16,
            hidden_size=4,
            dtype=DType.float32,
        ),
        devices=[CPU()],
        block_tokens=1,
    )


def _entry_rows(
    cache: VisionEncoderCache[TextAndVisionContext], image_hash: int
) -> npt.NDArray[np.float32]:
    """Materialize a cached entry's device-0 rows from its storage.

    Block entries are read out of their pool span; owned-fallback entries
    return their buffer's rows.
    """
    entry = cache.lookup(image_hash)
    assert entry is not None
    if entry.embeddings is not None:
        return entry.embeddings[0].to_numpy()
    assert entry.block_ids is not None
    pool = cache._pool
    assert pool is not None
    parts: list[npt.NDArray[np.float32]] = []
    row = 0
    for block_id in entry.block_ids:
        chunk = min(pool.block_tokens, entry.num_tokens - row)
        parts.append(pool.rows_view(block_id, 0, chunk).to_numpy())
        row += chunk
    return np.concatenate(parts)


def test_insert_and_lookup() -> None:
    cache = _make_cache()
    buf = _make_buffer(10)
    cache.insert(0xABC, [buf], 10)
    entry = cache.lookup(0xABC)
    assert entry is not None
    assert entry.num_tokens == 10
    assert entry.embeddings is None
    assert entry.block_ids is not None
    np.testing.assert_array_equal(_entry_rows(cache, 0xABC), buf.to_numpy())


def test_lookup_miss() -> None:
    cache = _make_cache()
    assert cache.lookup(0xDEAD) is None


def test_insert_idempotent() -> None:
    cache = _make_cache()
    buf1 = _make_buffer(10)
    buf2 = _make_buffer(20)
    entry1 = cache.insert(0xABC, [buf1], 10)
    entry2 = cache.insert(0xABC, [buf2], 20)
    assert entry1 is entry2
    assert entry1.num_tokens == 10


def test_lookup_refreshes_lru() -> None:
    cache = _make_cache_sized(2)
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.insert(0x2, [_make_buffer(1)], 1)
    cache.lookup(0x1)  # refresh
    cache.insert(0x3, [_make_buffer(1)], 1)
    assert cache.lookup(0x1) is not None
    assert cache.lookup(0x2) is None  # evicted
    assert cache.lookup(0x3) is not None


def test_evicts_oldest_unreferenced() -> None:
    cache = _make_cache_sized(2)
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.insert(0x2, [_make_buffer(1)], 1)
    cache.insert(0x3, [_make_buffer(1)], 1)
    assert cache.lookup(0x1) is None
    assert cache.lookup(0x2) is not None


def test_eviction_skips_referenced_entries() -> None:
    cache = _make_cache_sized(2)
    req = RequestID("r1")
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.acquire(req, 0x1)
    cache.insert(0x2, [_make_buffer(1)], 1)
    cache.insert(0x3, [_make_buffer(1)], 1)
    assert cache.lookup(0x1) is not None  # protected
    assert cache.lookup(0x2) is None  # evicted
    assert cache.lookup(0x3) is not None


def test_no_eviction_when_all_referenced() -> None:
    cache = _make_cache_sized(2)
    req = RequestID("r1")
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.acquire(req, 0x1)
    cache.insert(0x2, [_make_buffer(1)], 1)
    cache.acquire(req, 0x2)
    cache.insert(0x3, [_make_buffer(1)], 1)
    assert cache.lookup(0x1) is not None
    assert cache.lookup(0x2) is not None
    assert cache.lookup(0x3) is not None


def test_release_evicts_over_capacity_entries() -> None:
    """A 3-image request over-fills a 2-block pool, so the third entry is
    stored as an owned buffer outside the pool. Completing the request must
    drain that overshoot immediately, not at the next cache miss; the
    pool-resident entries stay."""
    cache = _make_cache_sized(2)
    req = RequestID("r1")
    for h in (0x1, 0x2, 0x3):
        cache.insert(h, [_make_buffer(1)], 1)
        cache.acquire(req, h)
    cache.release_request(req)
    assert len(cache._cache) == 2
    assert cache.lookup(0x1) is not None
    assert cache.lookup(0x2) is not None
    assert cache.lookup(0x3) is None


def test_release_drain_skips_entries_held_by_other_requests() -> None:
    cache = _make_cache_sized(1)
    r1 = RequestID("r1")
    r2 = RequestID("r2")
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.acquire(r1, 0x1)
    cache.acquire(r2, 0x1)
    cache.insert(0x2, [_make_buffer(1)], 1)
    cache.acquire(r1, 0x2)
    cache.release_request(r1)
    # 0x1 is still ref-held by r2 and must survive even though it is the
    # LRU entry; the zero-ref over-capacity entry 0x2 is the one drained.
    assert cache.lookup(0x1) is not None
    assert cache.lookup(0x2) is None


def test_acquire_increments_ref() -> None:
    cache = _make_cache()
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.acquire(RequestID("r1"), 0x1)
    assert _ref_count(cache, 0x1) == 1


def test_acquire_idempotent_per_request() -> None:
    cache = _make_cache()
    cache.insert(0x1, [_make_buffer(1)], 1)
    req = RequestID("r1")
    cache.acquire(req, 0x1)
    cache.acquire(req, 0x1)
    assert _ref_count(cache, 0x1) == 1


def test_multiple_requests_increment_separately() -> None:
    cache = _make_cache()
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.acquire(RequestID("r1"), 0x1)
    cache.acquire(RequestID("r2"), 0x1)
    assert _ref_count(cache, 0x1) == 2


def test_release_decrements_ref() -> None:
    cache = _make_cache()
    cache.insert(0x1, [_make_buffer(1)], 1)
    req = RequestID("r1")
    cache.acquire(req, 0x1)
    cache.release_request(req)
    assert _ref_count(cache, 0x1) == 0


def test_release_unknown_request_is_noop() -> None:
    cache = _make_cache()
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.release_request(RequestID("unknown"))
    assert _ref_count(cache, 0x1) == 0


def test_release_does_not_go_negative() -> None:
    cache = _make_cache()
    cache.insert(0x1, [_make_buffer(1)], 1)
    req = RequestID("r1")
    cache.acquire(req, 0x1)
    cache.release_request(req)
    cache.release_request(req)  # double release
    assert _ref_count(cache, 0x1) == 0


def test_get_uncached_all_cached() -> None:
    cache = _make_cache()
    cache.insert(0xA, [_make_buffer(5)], 5)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 5, image_hash=0xA)],
    )
    misses = cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert len(misses) == 0
    assert _ref_count(cache, 0xA) == 1


def test_get_uncached_returns_miss() -> None:
    cache = _make_cache()
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 5, image_hash=0xB)],
    )
    misses = cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert len(misses) == 1
    assert misses[0] is ctx


def test_get_uncached_partial_miss() -> None:
    """If one image is cached but another isn't, context is a miss.

    The cached image should have its ref acquired immediately, and
    only the uncached hash should appear in the returned set.
    """
    cache = _make_cache()
    cache.insert(0xA, [_make_buffer(5)], 5)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 5, image_hash=0xA),
            _make_image_meta(5, 10, image_hash=0xB),
        ],
    )
    misses = cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert len(misses) == 1
    # The cached image should already have its ref acquired.
    assert _ref_count(cache, 0xA) == 1


def test_prepare_partial_hit_only_encodes_uncached() -> None:
    """With a partial hit, only the uncached image is encoded and stored."""
    cache = _make_cache()
    hidden = 4

    # Pre-cache image A.
    buf_a = Buffer.from_numpy(np.ones((2, hidden), dtype=np.float32) * 1.0)
    cache.insert(0xA, [buf_a], 2)

    # Context has cached image A and uncached image B.
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 2, image_hash=0xA),
            _make_image_meta(2, 5, image_hash=0xB),
        ],
        image_token_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        processed_length=0,
        active_length=8,
    )
    uncached = cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert len(uncached) == 1

    # Simulate encoding ONLY image B (3 tokens).
    vision_embeds = [
        Buffer.from_numpy(np.ones((3, hidden), dtype=np.float32) * 2.0)
    ]
    result, _indices = cache.prepare_vision_outputs(
        context_batch=_as_vlm_batch([ctx]),
        uncached_contexts=uncached,
        uncached_images=_miss_images(cache, uncached),
        vision_embeds=vision_embeds,
        per_image_token_counts=[3],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    # Should assemble: image A (2 rows of 1.0) then image B (3 rows of 2.0).
    arr = result[0].to_numpy()
    assert arr.shape == (5, hidden)
    np.testing.assert_allclose(arr[:2], 1.0)
    np.testing.assert_allclose(arr[2:], 2.0)

    # Both images should have refs acquired.
    assert _ref_count(cache, 0xA) == 1
    assert _ref_count(cache, 0xB) == 1


def test_prepare_partial_hit_multi_context() -> None:
    """Partial hit in one context, full miss in another."""
    cache = _make_cache()
    hidden = 4

    # Pre-cache image A.
    buf_a = Buffer.from_numpy(np.ones((2, hidden), dtype=np.float32) * 1.0)
    cache.insert(0xA, [buf_a], 2)

    # ctx1: partial hit (A cached, B uncached).
    ctx1 = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 2, image_hash=0xA),
            _make_image_meta(2, 5, image_hash=0xB),
        ],
        image_token_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        processed_length=0,
        active_length=6,
    )
    # ctx2: full miss (C uncached).
    ctx2 = FakeContext(
        request_id=RequestID("r2"),
        images=[_make_image_meta(0, 4, image_hash=0xC)],
        image_token_indices=np.array([0, 1, 2, 3], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    uncached = cache.get_uncached_contexts(_as_vlm_batch([ctx1, ctx2]))
    assert len(uncached) == 2
    # Image A in ctx1 should already have its ref.
    assert _ref_count(cache, 0xA) == 1

    # Encode only B (3 tokens) and C (4 tokens).
    vision_embeds = [
        Buffer.from_numpy(np.ones((7, hidden), dtype=np.float32) * 2.0)
    ]
    result, _indices = cache.prepare_vision_outputs(
        context_batch=_as_vlm_batch([ctx1, ctx2]),
        uncached_contexts=uncached,
        uncached_images=_miss_images(cache, uncached),
        vision_embeds=vision_embeds,
        per_image_token_counts=[3, 4],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    # Assembled: A(2) + B(3) + C(4) = 9 rows.
    arr = result[0].to_numpy()
    assert arr.shape == (9, hidden)
    np.testing.assert_allclose(arr[:2], 1.0)  # image A from cache
    np.testing.assert_allclose(arr[2:], 2.0)  # images B, C from encoder


def test_get_uncached_skips_non_vision() -> None:
    cache = _make_cache()
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[],
        needs_vision=False,
    )
    misses = cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert len(misses) == 0


def test_none_hash_raises() -> None:
    """Missing image_hash raises ValueError when cache is enabled."""
    cache = _make_cache()
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 5, image_hash=None)],
    )
    with pytest.raises(ValueError):
        cache.get_uncached_contexts(_as_vlm_batch([ctx]))


def test_none_hash_allowed_when_disabled() -> None:
    """Missing image_hash is fine when cache is disabled."""
    cache = _make_cache_sized(0)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 5, image_hash=None)],
    )
    misses = cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert len(misses) == 1


def test__cache_and_split_stores_per_image() -> None:
    cache = _make_cache()
    hidden = 4
    total = _make_buffer(8, hidden)
    cache._cache_and_split(
        vision_outputs=[total],
        per_image_token_counts=[3, 5],
        image_hashes=[0xA, 0xB],
        request_ids=[RequestID("r1"), RequestID("r1")],
    )
    entry_a = cache.lookup(0xA)
    entry_b = cache.lookup(0xB)
    assert entry_a is not None and entry_a.num_tokens == 3
    assert entry_b is not None and entry_b.num_tokens == 5
    assert entry_a.embeddings is None and entry_a.block_ids is not None
    assert entry_b.embeddings is None and entry_b.block_ids is not None
    emb_a = _entry_rows(cache, 0xA)
    emb_b = _entry_rows(cache, 0xB)
    assert emb_a.shape == (3, hidden)
    assert emb_b.shape == (5, hidden)
    np.testing.assert_array_equal(emb_a, total.to_numpy()[:3])
    np.testing.assert_array_equal(emb_b, total.to_numpy()[3:8])


def test__cache_and_split_acquires_refs() -> None:
    cache = _make_cache()
    total = _make_buffer(5, 4)
    cache._cache_and_split(
        vision_outputs=[total],
        per_image_token_counts=[5],
        image_hashes=[0xA],
        request_ids=[RequestID("r1")],
    )
    assert _ref_count(cache, 0xA) == 1


def test__cache_and_split_none_hash_not_cached() -> None:
    cache = _make_cache()
    total = _make_buffer(5, 4)
    cache._cache_and_split(
        vision_outputs=[total],
        per_image_token_counts=[5],
        image_hashes=[None],  # type: ignore[list-item]
        request_ids=[RequestID("r1")],
    )
    assert len(cache._cache) == 0


def test__cache_and_split_zero_hash_not_cached() -> None:
    """0 is the no-content-hash sentinel; lookup() treats it as a miss, so a
    0-hash entry is never retrievable and must not be cached."""
    cache = _make_cache()
    total = _make_buffer(5, 4)
    cache._cache_and_split(
        vision_outputs=[total],
        per_image_token_counts=[5],
        image_hashes=[0],
        request_ids=[RequestID("r1")],
    )
    assert len(cache._cache) == 0
    assert cache.lookup(0) is None


def test__cache_and_split_skips_zero_hash_keeps_offset() -> None:
    """A 0 (no-content) hash is skipped, but its tokens still advance the split
    offset so later real images get the correct slice."""
    cache = _make_cache()
    hidden = 4
    total = _make_buffer(9, hidden)  # 2 (A) + 3 (zero) + 4 (B)
    cache._cache_and_split(
        vision_outputs=[total],
        per_image_token_counts=[2, 3, 4],
        image_hashes=[0xA, 0, 0xB],
        request_ids=[RequestID("r1"), RequestID("r1"), RequestID("r1")],
    )
    # Only the two real hashes are cached; the 0-hash range is skipped.
    assert cache.lookup(0) is None
    assert len(cache._cache) == 2
    entry_a = cache.lookup(0xA)
    entry_b = cache.lookup(0xB)
    assert entry_a is not None and entry_a.num_tokens == 2
    assert entry_b is not None and entry_b.num_tokens == 4
    emb_a = _entry_rows(cache, 0xA)
    emb_b = _entry_rows(cache, 0xB)
    # A takes rows [0:2]; B takes rows [5:9] -- the offset advanced past the
    # 3 skipped zero-hash rows, so B is NOT [2:6].
    np.testing.assert_array_equal(emb_a, total.to_numpy()[0:2])
    np.testing.assert_array_equal(emb_b, total.to_numpy()[5:9])


def test_assemble_concatenates_in_order() -> None:
    cache = _make_cache()
    hidden = 4
    buf_a = _make_buffer(3, hidden)
    buf_b = _make_buffer(5, hidden)
    cache.insert(0xA, [buf_a], 3)
    cache.insert(0xB, [buf_b], 5)

    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 3, image_hash=0xA),
            _make_image_meta(3, 8, image_hash=0xB),
        ],
        image_token_indices=np.arange(8, dtype=np.int32),
        active_length=8,
    )
    empty = [_make_buffer(0, hidden)]
    result = cache._assemble_embeddings(
        _as_vlm_batch([ctx]), n_devices=1, empty_embeddings=empty
    )
    arr = result[0].to_numpy()
    assert arr.shape == (8, hidden)
    np.testing.assert_array_equal(arr[:3], buf_a.to_numpy())
    np.testing.assert_array_equal(arr[3:], buf_b.to_numpy())


def _make_sharded_cache(
    n_devices: int, block_tokens: int = 4
) -> VisionEncoderCache[TextAndVisionContext]:
    """A cache whose pool is sharded across ``n_devices`` host devices."""
    return VisionEncoderCache(
        plan=VisionCachePlan(
            bytes_per_device=1024,
            hidden_size=4,
            dtype=DType.float32,
        ),
        devices=[CPU() for _ in range(n_devices)],
        block_tokens=block_tokens,
    )


def test_sharded_pool_places_blocks_round_robin() -> None:
    cache = _make_sharded_cache(2)
    pool = cache._pool
    assert pool is not None
    assert pool.num_blocks % 2 == 0
    for b in range(pool.num_blocks):
        assert pool.host_device_index(b) == b % 2


def test_sharded_store_spans_shards_and_gathers_identically() -> None:
    cache = _make_sharded_cache(2)
    buf = _make_buffer(10)
    cache.insert(0xA, [buf, buf], 10)
    entry = cache.lookup(0xA)
    assert entry is not None
    assert entry.block_ids is not None
    pool = cache._pool
    assert pool is not None
    hosts = {pool.host_device_index(b) for b in entry.block_ids}
    assert hosts == {0, 1}
    out = [
        Buffer(shape=[10, 4], dtype=DType.float32, device=CPU())
        for _ in range(2)
    ]
    pool.copy_out(out, 0, entry.block_ids, 0, 10)
    for o in out:
        np.testing.assert_array_equal(o.to_numpy(), buf.to_numpy())


def test_sharded_gather_slice_across_shard_boundary() -> None:
    cache = _make_sharded_cache(2)
    buf = _make_buffer(10)
    cache.insert(0xA, [buf, buf], 10)
    entry = cache.lookup(0xA)
    assert entry is not None
    assert entry.block_ids is not None
    pool = cache._pool
    assert pool is not None
    out = [
        Buffer(shape=[5, 4], dtype=DType.float32, device=CPU())
        for _ in range(2)
    ]
    pool.copy_out(out, 0, entry.block_ids, 2, 7)
    for o in out:
        np.testing.assert_array_equal(o.to_numpy(), buf.to_numpy()[2:7])


def test_sharded_rows_view_lives_on_host_shard() -> None:
    cache = _make_sharded_cache(2)
    buf = _make_buffer(4)
    cache.insert(0xA, [buf, buf], 4)
    entry = cache.lookup(0xA)
    assert entry is not None
    assert entry.block_ids is not None
    pool = cache._pool
    assert pool is not None
    (block_id,) = entry.block_ids
    view = pool.rows_view(block_id, 0, 4)
    np.testing.assert_array_equal(view.to_numpy(), buf.to_numpy())


def test_sharded_assemble_single_span_all_devices() -> None:
    cache = _make_sharded_cache(2)
    hidden = 4
    buf = _make_buffer(3, hidden)
    cache.insert(0xA, [buf, buf], 3)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 3, image_hash=0xA)],
        image_token_indices=np.arange(3, dtype=np.int32),
        active_length=3,
    )
    empty = [_make_buffer(0, hidden) for _ in range(2)]
    result = cache._assemble_embeddings(
        _as_vlm_batch([ctx]), n_devices=2, empty_embeddings=empty
    )
    assert len(result) == 2
    for r in result:
        np.testing.assert_array_equal(r.to_numpy(), buf.to_numpy())


def test_sharded_assemble_multi_span_all_devices() -> None:
    cache = _make_sharded_cache(2)
    hidden = 4
    buf_a = _make_buffer(3, hidden)
    buf_b = _make_buffer(5, hidden)
    cache.insert(0xA, [buf_a, buf_a], 3)
    cache.insert(0xB, [buf_b, buf_b], 5)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 3, image_hash=0xA),
            _make_image_meta(3, 8, image_hash=0xB),
        ],
        image_token_indices=np.arange(8, dtype=np.int32),
        active_length=8,
    )
    empty = [_make_buffer(0, hidden) for _ in range(2)]
    result = cache._assemble_embeddings(
        _as_vlm_batch([ctx]), n_devices=2, empty_embeddings=empty
    )
    assert len(result) == 2
    for r in result:
        arr = r.to_numpy()
        np.testing.assert_array_equal(arr[:3], buf_a.to_numpy())
        np.testing.assert_array_equal(arr[3:], buf_b.to_numpy())


def test_assemble_returns_empty_when_no_vision() -> None:
    cache = _make_cache()
    ctx = FakeContext(request_id=RequestID("r1"), images=[], needs_vision=False)
    empty = [_make_buffer(0, 4)]
    result = cache._assemble_embeddings(
        _as_vlm_batch([ctx]), n_devices=1, empty_embeddings=empty
    )
    assert result is empty


def test_assemble_multi_context_ordering() -> None:
    """Embeddings concatenated in context order."""
    cache = _make_cache()
    hidden = 4
    buf_a = Buffer.from_numpy(np.ones((2, hidden), dtype=np.float32) * 1.0)
    buf_b = Buffer.from_numpy(np.ones((3, hidden), dtype=np.float32) * 2.0)
    cache.insert(0xA, [buf_a], 2)
    cache.insert(0xB, [buf_b], 3)

    ctx1 = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 2, image_hash=0xA)],
        image_token_indices=np.arange(2, dtype=np.int32),
        active_length=2,
    )
    ctx2 = FakeContext(
        request_id=RequestID("r2"),
        images=[_make_image_meta(0, 3, image_hash=0xB)],
        image_token_indices=np.arange(3, dtype=np.int32),
        active_length=3,
    )
    empty = [_make_buffer(0, hidden)]
    result = cache._assemble_embeddings(
        _as_vlm_batch([ctx1, ctx2]), n_devices=1, empty_embeddings=empty
    )
    arr = result[0].to_numpy()
    assert arr.shape == (5, hidden)
    np.testing.assert_allclose(arr[:2], 1.0)
    np.testing.assert_allclose(arr[2:], 2.0)


def test_cross_request_dedup() -> None:
    """Two requests with the same image share one cache entry."""
    cache = _make_cache()
    cache.insert(0xABC, [_make_buffer(5)], 5)
    r1, r2 = RequestID("r1"), RequestID("r2")
    ctx1 = FakeContext(
        request_id=r1,
        images=[_make_image_meta(0, 5, image_hash=0xABC)],
    )
    ctx2 = FakeContext(
        request_id=r2,
        images=[_make_image_meta(0, 5, image_hash=0xABC)],
    )
    misses = cache.get_uncached_contexts(_as_vlm_batch([ctx1, ctx2]))
    assert len(misses) == 0
    assert _ref_count(cache, 0xABC) == 2

    cache.release_request(r1)
    assert _ref_count(cache, 0xABC) == 1
    cache.release_request(r2)
    assert _ref_count(cache, 0xABC) == 0


def test_end_to_end_chunked_prefill() -> None:
    """Simulates the full 2-chunk prefill workflow."""
    cache = _make_cache()
    hidden = 4
    req = RequestID("request-1")

    # Chunk 1: cache miss → encode → store
    ctx = FakeContext(
        request_id=req,
        images=[_make_image_meta(100, 400, image_hash=0xABC)],
        image_token_indices=np.arange(100, 400, dtype=np.int32),
        active_length=400,
    )
    misses = cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert len(misses) == 1

    vision_output = _make_buffer(300, hidden)
    cache._cache_and_split(
        vision_outputs=[vision_output],
        per_image_token_counts=[300],
        image_hashes=[0xABC],
        request_ids=[req],
    )

    empty = [_make_buffer(0, hidden)]
    embeds1 = cache._assemble_embeddings(
        _as_vlm_batch([ctx]), n_devices=1, empty_embeddings=empty
    )
    assert embeds1[0].to_numpy().shape == (300, hidden)

    # Chunk 2: cache hit → no encoding
    misses = cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert len(misses) == 0

    embeds2 = cache._assemble_embeddings(
        _as_vlm_batch([ctx]), n_devices=1, empty_embeddings=empty
    )
    np.testing.assert_array_equal(embeds2[0].to_numpy(), embeds1[0].to_numpy())

    cache.release_request(req)
    assert _ref_count(cache, 0xABC) == 0
    assert cache.lookup(0xABC) is not None


def test_prepare_all_uncached_fast_path() -> None:
    """When every vision context is uncached, returns encoder output directly."""
    cache = _make_cache()
    hidden = 4
    req = RequestID("r1")
    ctx = FakeContext(
        request_id=req,
        images=[_make_image_meta(0, 3, image_hash=0xA)],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    uncached = cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert len(uncached) == 1

    vision_embeds = [_make_buffer(3, hidden)]
    result, indices = cache.prepare_vision_outputs(
        context_batch=_as_vlm_batch([ctx]),
        uncached_contexts=uncached,
        uncached_images=_miss_images(cache, uncached),
        vision_embeds=vision_embeds,
        per_image_token_counts=[3],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    assert result is vision_embeds
    assert _ref_count(cache, 0xA) == 1
    np.testing.assert_array_equal(indices, [0, 1, 2])


def test_prepare_mixed_hits() -> None:
    """When some contexts are cached and others are not, assembles from cache."""
    cache = _make_cache()
    hidden = 4

    buf_a = Buffer.from_numpy(np.ones((2, hidden), dtype=np.float32) * 1.0)
    cache.insert(0xA, [buf_a], 2)

    ctx1 = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 2, image_hash=0xA)],
        image_token_indices=np.array([0, 1], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    ctx2 = FakeContext(
        request_id=RequestID("r2"),
        images=[_make_image_meta(0, 3, image_hash=0xB)],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    uncached = cache.get_uncached_contexts(_as_vlm_batch([ctx1, ctx2]))
    # ctx1 is a hit (all images cached), ctx2 is a miss.
    assert len(uncached) == 1
    assert uncached[0].request_id == RequestID("r2")

    vision_embeds = [
        Buffer.from_numpy(np.ones((3, hidden), dtype=np.float32) * 2.0)
    ]
    result, _indices = cache.prepare_vision_outputs(
        context_batch=_as_vlm_batch([ctx1, ctx2]),
        uncached_contexts=uncached,
        uncached_images=_miss_images(cache, uncached),
        vision_embeds=vision_embeds,
        per_image_token_counts=[3],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    # Should assemble: image A (2 rows of 1.0) then image B (3 rows of 2.0).
    arr = result[0].to_numpy()
    assert arr.shape == (5, hidden)
    np.testing.assert_allclose(arr[:2], 1.0)
    np.testing.assert_allclose(arr[2:], 2.0)


def test_prepare_all_cached() -> None:
    """When every context is already cached, returns assembled embeddings."""
    cache = _make_cache()
    hidden = 4
    buf = Buffer.from_numpy(np.ones((4, hidden), dtype=np.float32) * 3.0)
    cache.insert(0xC, [buf], 4)

    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 4, image_hash=0xC)],
        image_token_indices=np.array([0, 1, 2, 3], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    uncached = cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert len(uncached) == 0

    empty = [_make_buffer(0, hidden)]
    result, _indices = cache.prepare_vision_outputs(
        context_batch=_as_vlm_batch([ctx]),
        uncached_contexts=uncached,
        uncached_images=_miss_images(cache, uncached),
        vision_embeds=empty,
        per_image_token_counts=[],
        n_devices=1,
        empty_embeddings=empty,
    )
    arr = result[0].to_numpy()
    assert arr.shape == (4, hidden)
    np.testing.assert_allclose(arr, 3.0)


def test_disabled_cache_never_hits() -> None:
    """With a zero byte budget, every vision context is always uncached."""
    cache = _make_cache_sized(0)
    assert not cache.enabled

    hidden = 4
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 3, image_hash=0xA)],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )

    # First call: miss as expected.
    uncached = cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert len(uncached) == 1

    # Simulate encoding and caching.
    vision_embeds = [_make_buffer(3, hidden)]
    cache.prepare_vision_outputs(
        context_batch=_as_vlm_batch([ctx]),
        uncached_contexts=uncached,
        uncached_images=_miss_images(cache, uncached),
        vision_embeds=vision_embeds,
        per_image_token_counts=[3],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )

    # Second call with same image: still a miss — nothing was stored.
    uncached2 = cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert len(uncached2) == 1
    assert cache.lookup(0xA) is None


def test_disabled_cache_insert_returns_transient_entry() -> None:
    """Insert on a disabled cache returns a valid entry but doesn't store it."""
    cache = _make_cache_sized(0)
    buf = _make_buffer(5, 4)
    entry = cache.insert(0xBEEF, [buf], 5)
    assert entry.num_tokens == 5
    assert entry.embeddings == [buf]
    # Not stored in the cache.
    assert cache.lookup(0xBEEF) is None


def test_merge_indices_single_context_no_offset() -> None:
    """Indices start from 0 when processed_length is 0."""
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(2, 5, image_hash=0xA)],
        image_token_indices=np.array([2, 3, 4], dtype=np.int32),
        processed_length=0,
        active_length=10,
    )
    indices = _compute_merge_indices([ctx])
    np.testing.assert_array_equal(indices, [2, 3, 4])


def test_merge_indices_accounts_for_processed_length() -> None:
    """Indices in already-processed tokens become OOB sentinels."""
    # Prompt: [0..9, IMG, IMG, IMG, IMG, 14..19]  (20 tokens total)
    # Image at positions 10-13.  processed_length=12 means tokens 0-11 done.
    # Active window is tokens 12-19 (active_length=8).
    # Indices 10 and 11 are in processed region → OOB.
    # Indices 12 → offset 0, 13 → offset 1 within the active window.
    oob = np.iinfo(np.int32).min
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(10, 14, image_hash=0xA)],
        image_token_indices=np.array([10, 11, 12, 13], dtype=np.int32),
        processed_length=12,
        active_length=8,
    )
    indices = _compute_merge_indices([ctx])
    np.testing.assert_array_equal(indices, [oob, oob, 0, 1])


def test_merge_indices_batch_offsets() -> None:
    """Active-token offsets accumulate across contexts in a batch."""
    ctx1 = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 2, image_hash=0xA)],
        image_token_indices=np.array([0, 1], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    ctx2 = FakeContext(
        request_id=RequestID("r2"),
        images=[_make_image_meta(0, 3, image_hash=0xB)],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,
        active_length=7,
    )
    indices = _compute_merge_indices([ctx1, ctx2])
    # ctx1 indices: [0, 1]  (offset 0)
    # ctx2 indices: [0, 1, 2] + 5 (ctx1.active_length) = [5, 6, 7]
    np.testing.assert_array_equal(indices, [0, 1, 5, 6, 7])


def test_merge_indices_skips_non_vision_contexts() -> None:
    """Non-vision contexts contribute to offset but produce no indices."""
    ctx_text = FakeContext(
        request_id=RequestID("r1"),
        needs_vision=False,
        active_length=10,
    )
    ctx_vision = FakeContext(
        request_id=RequestID("r2"),
        images=[_make_image_meta(0, 2, image_hash=0xA)],
        image_token_indices=np.array([0, 1], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    indices = _compute_merge_indices([ctx_text, ctx_vision])
    # text context contributes 10 tokens of offset.
    np.testing.assert_array_equal(indices, [10, 11])


def test_merge_indices_empty_batch() -> None:
    """Empty batch returns empty array."""
    indices = compute_multimodal_merge_indices([])
    assert indices.shape == (0,)
    assert indices.dtype == np.int32


def test_merge_indices_beyond_active_are_oob() -> None:
    """Indices beyond active_length must be OOB, not passed through."""
    oob = np.iinfo(np.int32).min
    # Image spans positions 2-7, but active window is only 0-4 (active_length=5).
    # Indices 5, 6, 7 are beyond active and must be OOB.
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(2, 8, image_hash=0xA)],
        image_token_indices=np.array([2, 3, 4, 5, 6, 7], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    indices = _compute_merge_indices([ctx])
    np.testing.assert_array_equal(indices, [2, 3, 4, oob, oob, oob])


def test_merge_indices_beyond_active_no_cross_contamination() -> None:
    """Beyond-active indices must not land in another request's token range."""
    oob = np.iinfo(np.int32).min
    # ctx1: vision request with image spanning beyond its active window.
    ctx1 = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(2, 8, image_hash=0xA)],
        image_token_indices=np.array([2, 3, 4, 5, 6, 7], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    # ctx2: text-only decode request contributing 3 tokens.
    ctx2 = FakeContext(
        request_id=RequestID("r2"),
        needs_vision=False,
        active_length=3,
    )
    indices = _compute_merge_indices([ctx1, ctx2])
    # Only indices 2,3,4 are valid. 5,6,7 must be OOB — NOT 5+0=5, 6+0=6, 7+0=7
    # which would land in ctx2's token range [5, 8).
    np.testing.assert_array_equal(indices, [2, 3, 4, oob, oob, oob])
    # Verify no index falls in ctx2's range.
    valid = indices[indices != oob]
    assert all(v < 5 for v in valid)


def test_prepare_vision_outputs_returns_embeddings_and_indices() -> None:
    """prepare_vision_outputs returns both embeddings and scatter indices."""
    cache = _make_cache()
    hidden = 4
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 3, image_hash=0xA)],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,
        active_length=10,
    )
    uncached = cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert len(uncached) == 1

    vision_embeds = [_make_buffer(3, hidden)]
    embeddings, indices = cache.prepare_vision_outputs(
        context_batch=_as_vlm_batch([ctx]),
        uncached_contexts=uncached,
        uncached_images=_miss_images(cache, uncached),
        vision_embeds=vision_embeds,
        per_image_token_counts=[3],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    assert embeddings is vision_embeds
    np.testing.assert_array_equal(indices, [0, 1, 2])


def test_prepare_vision_outputs_chunked_prefill() -> None:
    """A straddling image is stored in full but assembled window-bounded."""
    cache = _make_cache()
    hidden = 4

    # 6 image tokens at positions 4-9, processed_length=6 (first 6 done),
    # active window is positions 6-15 (active_length=10).
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(4, 10, image_hash=0xA)],
        image_token_indices=np.array([4, 5, 6, 7, 8, 9], dtype=np.int32),
        processed_length=6,
        active_length=10,
    )
    uncached = cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    vision_embeds = [_make_buffer(6, hidden)]
    embeddings, indices = cache.prepare_vision_outputs(
        context_batch=_as_vlm_batch([ctx]),
        uncached_contexts=uncached,
        uncached_images=_miss_images(cache, uncached),
        vision_embeds=vision_embeds,
        per_image_token_counts=[6],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    # Positions 4,5 are already processed: their rows are omitted, not
    # OOB-masked. Rows pair with indices 0-3 of the active window.
    np.testing.assert_array_equal(indices, [0, 1, 2, 3])
    arr = embeddings[0].to_numpy()
    np.testing.assert_array_equal(arr, vision_embeds[0].to_numpy()[2:6])
    # The full entry is stored for reuse by later chunks and requests.
    entry = cache.lookup(0xA)
    assert entry is not None and entry.num_tokens == 6


# ---------------------------------------------------------------------------
# Per-iteration metrics (pop_metrics) tests
# ---------------------------------------------------------------------------


def test_pop_metrics_none_for_text_only() -> None:
    cache = _make_cache()
    ctx = FakeContext(request_id=RequestID("r1"), images=[])
    cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert cache.pop_metrics() is None


def test_pop_metrics_counts_hits_and_misses() -> None:
    cache = _make_cache()
    cache.insert(0xA, [_make_buffer(2)], 2)  # pre-cache image A -> hit
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 2, image_hash=0xA),  # cached hit, 2 tokens
            _make_image_meta(2, 5, image_hash=0xB),  # miss, 3 tokens, 1 patch
        ],
        image_token_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        processed_length=0,
        active_length=8,
    )
    batch = _as_vlm_batch([ctx])
    uncached = cache.get_uncached_contexts(batch)
    cache.prepare_vision_outputs(
        context_batch=batch,
        uncached_contexts=uncached,
        uncached_images=_miss_images(cache, uncached),
        vision_embeds=[_make_buffer(3)],
        per_image_token_counts=[3],
        n_devices=1,
        empty_embeddings=[_make_buffer(0)],
    )
    m = cache.pop_metrics()
    assert m is not None
    assert m.num_images_total == 2
    assert m.num_images_cached == 1
    assert m.num_images_encoded == 1
    assert m.num_patches_encoded == 1  # only the miss is encoded
    assert m.num_tokens_encoded == 3  # 5 - 2
    assert m.cache_hit_rate == 0.5
    # pop resets the accumulator.
    assert cache.pop_metrics() is None


def test_pop_metrics_disabled_cache_counts_all_as_encoded() -> None:
    cache = _make_cache_sized(0)  # caching disabled
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 5, image_hash=0xA)],
        image_token_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        processed_length=0,
        active_length=8,
    )
    batch = _as_vlm_batch([ctx])
    uncached = cache.get_uncached_contexts(batch)
    cache.prepare_vision_outputs(
        context_batch=batch,
        uncached_contexts=uncached,
        uncached_images=_miss_images(cache, uncached),
        vision_embeds=[_make_buffer(5)],
        per_image_token_counts=[5],
        n_devices=1,
        empty_embeddings=[_make_buffer(0)],
    )
    m = cache.pop_metrics()
    assert m is not None
    assert m.num_images_total == 1
    assert m.num_images_encoded == 1
    assert m.num_images_cached == 0
    assert m.num_patches_encoded == 1
    assert m.num_tokens_encoded == 5


def test_video_encoder_metrics_defaults_are_zero() -> None:
    m = VideoEncoderMetrics()
    assert m.num_clips_total == 0
    assert m.num_clips_encoded == 0
    assert m.num_clips_cached == 0
    assert m.frame_counts == []
    assert m.num_tokens_encoded == 0
    assert m.encoding_time_ms == 0.0
    assert m.cache_hit_rate == 0.0


def test_video_encoder_metrics_cache_hit_rate() -> None:
    m = VideoEncoderMetrics(
        num_clips_total=4,
        num_clips_encoded=1,
        num_clips_cached=3,
        frame_counts=[16],
        num_tokens_encoded=64,
        encoding_time_ms=12.5,
    )
    assert m.cache_hit_rate == 0.75


def test_derive_counts_from_spans_single_context() -> None:
    """Per-image count is the placeholder span of the selected images."""
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 3, image_hash=0xA),
            _make_image_meta(3, 8, image_hash=0xB),
        ],
    )
    counts = derive_counts_from_spans(
        _as_selection([(ctx, list(ctx.images))])  # both selected to encode
    )
    assert counts == [3, 5]


def test_derive_counts_from_spans_context_major_image_minor() -> None:
    """Counts follow context-major, image-minor order (the embeddings order)."""
    ctx1 = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 2, image_hash=0xA)],
    )
    ctx2 = FakeContext(
        request_id=RequestID("r2"),
        images=[
            _make_image_meta(0, 4, image_hash=0xB),
            _make_image_meta(4, 7, image_hash=0xC),
        ],
    )
    counts = derive_counts_from_spans(
        _as_selection([(ctx1, list(ctx1.images)), (ctx2, list(ctx2.images))])
    )
    assert counts == [2, 4, 3]


def test_derive_counts_from_spans_uses_selection_not_all_images() -> None:
    """Counts follow the manager's selection, not every image in the context."""
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 2, image_hash=0xA),  # not selected
            _make_image_meta(2, 7, image_hash=0xB),  # selected -> counted
        ],
    )
    counts = derive_counts_from_spans(
        _as_selection([(ctx, [ctx.images[1]])])  # only B selected to encode
    )
    assert counts == [5]


def test_derive_counts_from_spans_empty_batch() -> None:
    assert derive_counts_from_spans([]) == []


def test_validate_counts_matches_rows() -> None:
    """No raise when the summed counts equal the encoder row count."""
    embeds = [_make_buffer(8, 4)]  # 8 rows
    validate_vision_encode_counts([3, 5], embeds)


def test_validate_counts_mismatch_raises() -> None:
    """A row/count mismatch is a hard error, not a silent reshape."""
    embeds = [_make_buffer(8, 4)]  # 8 rows
    with pytest.raises(ValueError):
        validate_vision_encode_counts([3, 4], embeds)  # sums to 7 != 8


def test_validate_counts_empty_embeddings_is_noop() -> None:
    """An empty (no-encode) batch validates trivially."""
    validate_vision_encode_counts([], [])


def test_derive_then_validate_roundtrip() -> None:
    """Counts derived from the selection validate against the encoder output."""
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 3, image_hash=0xA),
            _make_image_meta(3, 8, image_hash=0xB),
        ],
    )
    counts = derive_counts_from_spans(_as_selection([(ctx, list(ctx.images))]))
    embeds = [_make_buffer(sum(counts), 4)]  # 8 rows
    validate_vision_encode_counts(counts, embeds)


def test_supports_vision_encoding_runtime_checkable() -> None:
    """``SupportsVisionEncoding`` matches the two-step pack/execute contract."""

    class _Encoder:
        def pack_vision_inputs(
            self, uncached: object, devices: object
        ) -> object:
            return None

        def vision_execute(
            self, uncached: object, devices: object, packed: object
        ) -> VisionEncodeResult:
            return VisionEncodeResult(embeddings=[])

        def empty_vision_embeddings(self, devices: object) -> list[Buffer]:
            return []

    class _OnlyPacks:
        def pack_vision_inputs(
            self, uncached: object, devices: object
        ) -> object:
            return None

    class _NotAnEncoder:
        pass

    assert isinstance(_Encoder(), SupportsVisionEncoding)
    assert not isinstance(_OnlyPacks(), SupportsVisionEncoding)
    assert not isinstance(_NotAnEncoder(), SupportsVisionEncoding)


def test_vision_encode_result_defaults() -> None:
    """``per_image_token_counts`` defaults to ``None`` (driver derives)."""
    result = VisionEncodeResult(embeddings=[_make_buffer(3, 4)])
    assert result.per_image_token_counts is None
    assert len(result.embeddings) == 1


def _make_layer_buffer(rows: int, cols: int, base: int) -> Buffer:
    """A [rows, cols] host Buffer whose values start at *base* (distinct layers)."""
    arr = (base + np.arange(rows * cols, dtype=np.float32)).reshape(rows, cols)
    return Buffer.from_numpy(arr)


def _make_manager(
    budget_bytes_per_device: int = 1024 * 1024, n_devices: int = 1
) -> VisionEncoderCache[TextAndVisionContext]:
    """Create a manager for GPU-free testing (host buffers, one device).

    The default budget dwarfs every test's data, so nothing is evicted;
    pass ``budget_bytes_per_device=0`` for a disabled cache.
    """
    if budget_bytes_per_device == 0:
        return VisionEncoderCache()
    return VisionEncoderCache(
        plan=VisionCachePlan(
            bytes_per_device=budget_bytes_per_device,
            hidden_size=4,
            dtype=DType.float32,
        ),
        devices=[CPU() for _ in range(n_devices)],
        block_tokens=4,
    )


def test_manager_select_returns_miss_images_and_acquires_hits() -> None:
    """``select`` returns (miss context, miss images) pairs and acquires hits."""
    manager = _make_manager()
    manager.insert(0xA, [_make_buffer(2)], num_tokens=2)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 2, image_hash=0xA),  # hit
            _make_image_meta(2, 5, image_hash=0xB),  # miss
        ],
    )
    selection = manager.select(_as_vlm_batch([ctx]))
    assert [c for c, _ in selection] == [ctx]  # has a miss -> selected
    assert [img.image_hash for _, miss in selection for img in miss] == [0xB]
    assert _ref_count(manager, 0xA) == 1


def test_manager_select_skips_fully_cached_context() -> None:
    """A context whose every image is cached is not returned, but refs held."""
    manager = _make_manager()
    manager.insert(0xA, [_make_buffer(2)], num_tokens=2)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 2, image_hash=0xA)],
    )
    selection = manager.select(_as_vlm_batch([ctx]))
    assert selection == []
    assert _ref_count(manager, 0xA) == 1


def test_manager_select_excludes_contexts_with_no_window_misses() -> None:
    """A context whose only misses are ahead of the window is not selected.

    Regression test for MXSERV-330: a cache hit inside the window plus an
    uncached image ahead of it produced a selection entry with an empty
    miss list, which reached the model encode path and tripped its
    non-empty assertion.
    """
    manager = _make_manager()
    manager.insert(0xA, [_make_buffer(2)], num_tokens=2)
    hit = _make_image_meta(0, 2, image_hash=0xA)
    ahead = _make_image_meta(10, 14, image_hash=0xB)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[hit, ahead],
        next_images=[hit],
    )
    selection = manager.select(_as_vlm_batch([ctx]))
    assert selection == []
    assert _ref_count(manager, 0xA) == 1


def test_manager_cache_vision_embeddings_stores_and_assembles() -> None:
    """``cache_vision_embeddings`` derives counts, stores, and assembles."""
    manager = _make_manager()
    hidden = 4
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 3, image_hash=0xA)],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    selection = manager.select(_as_vlm_batch([ctx]))
    result = VisionEncodeResult(embeddings=[_make_buffer(3, hidden)])
    _embeddings, indices = manager.cache_vision_embeddings(
        context_batch=_as_vlm_batch([ctx]),
        selection=selection,
        encode_result=result,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    np.testing.assert_array_equal(indices, [0, 1, 2])
    entry = manager.lookup(0xA)
    assert entry is not None and entry.num_tokens == 3


def test_manager_cache_vision_embeddings_validates_before_caching() -> None:
    """A row/count mismatch raises before the cache is written."""
    manager = _make_manager()
    hidden = 4
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 3, image_hash=0xA)],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    selection = manager.select(_as_vlm_batch([ctx]))
    result = VisionEncodeResult(
        embeddings=[_make_buffer(5, hidden)], per_image_token_counts=[3]
    )
    with pytest.raises(ValueError):
        manager.cache_vision_embeddings(
            context_batch=_as_vlm_batch([ctx]),
            selection=selection,
            encode_result=result,
            empty_embeddings=[_make_buffer(0, hidden)],
        )
    assert manager.lookup(0xA) is None


def test_manager_release_drops_refs() -> None:
    """``release`` mirrors the KV release path: drops this request's refs."""
    manager = _make_manager()
    manager.insert(0xA, [_make_buffer(2)], num_tokens=2)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 2, image_hash=0xA)],
    )
    manager.select(_as_vlm_batch([ctx]))
    assert _ref_count(manager, 0xA) == 1
    manager.release_request(RequestID("r1"))
    assert _ref_count(manager, 0xA) == 0


def test_manager_disabled_when_zero_budget() -> None:
    """A zero byte budget disables the cache (manager reflects it)."""
    manager = _make_manager(budget_bytes_per_device=0)
    assert not manager.enabled


class _FakeEncodeModel:
    """Minimal SupportsVisionEncoding double recording pack/execute calls."""

    def __init__(self, hidden: int = 4, pack_result: object = "PACKED") -> None:
        self.hidden = hidden
        self._pack_result = pack_result
        self.calls: list[tuple[str, object]] = []

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> object:
        self.calls.append(("pack", self._pack_result))
        return self._pack_result

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: object,
    ) -> VisionEncodeResult:
        self.calls.append(("vision_execute", packed))
        counts = [
            img.end_idx - img.start_idx
            for _ctx, miss_images in selection
            for img in miss_images
        ]
        rows = sum(counts)
        return VisionEncodeResult(
            embeddings=[_make_buffer(rows, self.hidden)],
            per_image_token_counts=counts,
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        return [_make_buffer(0, self.hidden)]


_FAKE_DEVICES: list[Device] = []


def test_run_vision_encode_text_only_returns_none() -> None:
    """A batch with no vision contexts skips the encoder entirely."""
    manager = _make_manager()
    model = _FakeEncodeModel()
    ctx = FakeContext(request_id=RequestID("r1"), needs_vision=False)
    out = manager.run_vision_encode(
        model, [_as_vlm_batch([ctx])], _FAKE_DEVICES
    )
    assert out is None
    assert model.calls == []  # neither pack nor encode ran


def test_run_vision_encode_all_hits_assembles_from_cache() -> None:
    """All-cache-hits: assemble from cache, no pack/encode."""
    manager = _make_manager()
    model = _FakeEncodeModel()
    manager.insert(0xA, [_make_buffer(3, 4)], num_tokens=3)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 3, image_hash=0xA)],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,
        active_length=3,
    )
    out = manager.run_vision_encode(
        model, [_as_vlm_batch([ctx])], _FAKE_DEVICES
    )
    assert out is not None
    embeds, scatter = out
    assert model.calls == []  # nothing to encode
    assert int(embeds[0].shape[0]) == 3  # assembled from the resident entry
    np.testing.assert_array_equal(scatter, [0, 1, 2])


def test_run_vision_encode_passes_packed_none_through() -> None:
    """``pack_vision_inputs`` returning None is passed through to execute."""
    manager = _make_manager()
    model = _FakeEncodeModel(pack_result=None)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 3, image_hash=0xB)],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,
        active_length=3,
    )
    out = manager.run_vision_encode(
        model, [_as_vlm_batch([ctx])], _FAKE_DEVICES
    )
    assert out is not None
    assert ("pack", None) in model.calls
    assert ("vision_execute", None) in model.calls  # packed=None reached it
    assert manager.lookup(0xB) is not None  # encoded + cached


def test_derive_disabled_cache_chunked_prefill_no_false_raise() -> None:
    """Disabled cache + chunked prefill: counts follow the selection, not all images.

    Regression: deriving over ``ctx.images`` filtered by cache lookups would,
    with the cache disabled, count already-processed images the encoder emitted
    no rows for and falsely raise in validation.
    """
    manager = _make_manager(budget_bytes_per_device=0)
    assert not manager.enabled
    img_a = _make_image_meta(0, 2, image_hash=0xA)
    img_b = _make_image_meta(2, 7, image_hash=0xB)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[img_a, img_b],  # A already processed in a prior chunk
        next_images=[img_b],  # only B is in this chunk
        image_token_indices=np.array([2, 3, 4, 5, 6], dtype=np.int32),
        processed_length=2,
        active_length=5,
    )
    selection = manager.select(_as_vlm_batch([ctx]))
    assert [c for c, _ in selection] == [ctx]
    assert [img.image_hash for _, miss in selection for img in miss] == [0xB]
    result = VisionEncodeResult(embeddings=[_make_buffer(5, 4)])
    embeds, _indices = manager.cache_vision_embeddings(
        context_batch=_as_vlm_batch([ctx]),
        selection=selection,
        encode_result=result,
        empty_embeddings=[_make_buffer(0, 4)],
    )
    assert int(embeds[0].shape[0]) == 5


def test_derive_enabled_partial_hit_counts_match_selection() -> None:
    """Enabled cache, partial hit: counts cover only the freshly-encoded image."""
    manager = _make_manager()
    manager.insert(0xA, [_make_buffer(2, 4)], num_tokens=2)  # A resident
    img_a = _make_image_meta(0, 2, image_hash=0xA)
    img_b = _make_image_meta(2, 7, image_hash=0xB)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[img_a, img_b],
        next_images=[img_a, img_b],  # both in-chunk, but A is a cache hit
        image_token_indices=np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32),
        processed_length=0,
        active_length=7,
    )
    selection = manager.select(_as_vlm_batch([ctx]))
    assert [img.image_hash for _, miss in selection for img in miss] == [0xB]
    result = VisionEncodeResult(embeddings=[_make_buffer(5, 4)])  # B only
    embeds, _indices = manager.cache_vision_embeddings(
        context_batch=_as_vlm_batch([ctx]),
        selection=selection,
        encode_result=result,
        empty_embeddings=[_make_buffer(0, 4)],
    )
    assert int(embeds[0].shape[0]) == 7  # A (2, from cache) + B (5)


def test_prepare_vision_outputs_splits_by_passed_selection() -> None:
    """The store follows the caller's ``uncached_images``, not a re-derivation
    over ``ctx.images`` — so hashes stay aligned with the encoder rows and the
    strict-zip split can't diverge."""
    cache = _make_cache()
    hidden = 4
    img_a = _make_image_meta(0, 2, image_hash=0xA)
    img_b = _make_image_meta(2, 5, image_hash=0xB)
    cache.insert(0xA, [_make_buffer(2, hidden)], num_tokens=2)  # A resident
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[img_a, img_b],
        image_token_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    embeddings, _ = cache.prepare_vision_outputs(
        context_batch=_as_vlm_batch([ctx]),
        uncached_contexts=_as_vlm_batch([ctx]),
        uncached_images=[[img_b]],
        vision_embeds=[_make_buffer(3, hidden)],
        per_image_token_counts=[3],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    entry_b = cache.lookup(0xB)
    assert entry_b is not None and entry_b.num_tokens == 3  # B stored
    assert int(embeddings[0].shape[0]) == 5  # A(2, cached) + B(3) assembled


def test_assemble_skips_evicted_prior_image() -> None:
    """A prefix-hit prior image evicted from the vision cache must not
    crash assembly. Its tokens are all outside the active window, so it
    contributes no rows and no indices."""
    cache = _make_cache()
    hidden = 4
    img_a = _make_image_meta(0, 2, image_hash=0xA)
    img_b = _make_image_meta(2, 5, image_hash=0xB)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[img_a, img_b],
        next_images=[img_b],
        # Realistic: placeholder tokens span BOTH images (A at 0-1, B at 2-4).
        image_token_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        processed_length=2,  # A fully processed (end_idx 2); B in the window
        active_length=3,
    )
    # A is NOT inserted -> evicted. Encoder output covers only B (3 rows).
    vision_embeds = [
        Buffer.from_numpy(np.full((3, hidden), 7.0, dtype=np.float32))
    ]
    embeddings, indices = cache.prepare_vision_outputs(
        context_batch=_as_vlm_batch([ctx]),
        uncached_contexts=_as_vlm_batch([ctx]),
        uncached_images=[[img_b]],
        vision_embeds=vision_embeds,
        per_image_token_counts=[3],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    arr = embeddings[0].to_numpy()
    # One embedding row per merge index; A contributes neither.
    assert arr.shape == (3, hidden)
    assert len(indices) == arr.shape[0]
    np.testing.assert_allclose(arr, 7.0)
    np.testing.assert_array_equal(indices, [0, 1, 2])
    # B is cached; A is never fabricated into the cache.
    assert cache.lookup(0xB) is not None
    assert cache.lookup(0xA) is None


def _make_block_cache(
    num_blocks: int = 8,
    block_tokens: int = 4,
    hidden: int = 4,
    n_devices: int = 1,
) -> VisionEncoderCache[TextAndVisionContext]:
    """Block-mode cache whose pool holds exactly ``num_blocks`` blocks.

    Budget is sized for float32 rows of width ``hidden`` (what
    ``_make_buffer`` produces).
    """
    budget = num_blocks * block_tokens * hidden * 4
    return VisionEncoderCache(
        plan=VisionCachePlan(
            bytes_per_device=budget,
            hidden_size=hidden,
            dtype=DType.float32,
        ),
        devices=[CPU() for _ in range(n_devices)],
        block_tokens=block_tokens,
    )


def test_block_pool_allocates_at_construction() -> None:
    cache = _make_block_cache(num_blocks=8, block_tokens=4)
    assert cache.total_num_blocks == 8
    entry = cache.insert(0xA, [_make_buffer(6)], 6)
    assert cache.total_num_blocks == 8
    assert entry.embeddings is None
    assert entry.block_ids is not None and len(entry.block_ids) == 2


def test_block_store_roundtrip_multi_block() -> None:
    """Values survive the split into blocks and the gather back out."""
    cache = _make_block_cache(num_blocks=8, block_tokens=4)
    hidden = 4
    total = _make_buffer(12, hidden)
    cache._cache_and_split(
        vision_outputs=[total],
        per_image_token_counts=[3, 9],
        image_hashes=[0xA, 0xB],
        request_ids=[RequestID("r1"), RequestID("r1")],
    )
    entry_a = cache.lookup(0xA)
    entry_b = cache.lookup(0xB)
    assert entry_a is not None and entry_a.block_ids is not None
    assert entry_b is not None and len(entry_b.block_ids or []) == 3
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 3, image_hash=0xA),
            _make_image_meta(3, 12, image_hash=0xB),
        ],
        image_token_indices=np.arange(12, dtype=np.int32),
        active_length=12,
    )
    result = cache._assemble_embeddings(
        _as_vlm_batch([ctx]),
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    np.testing.assert_array_equal(result[0].to_numpy(), total.to_numpy())


def test_block_store_roundtrip_multi_device() -> None:
    cache = _make_block_cache(num_blocks=4, block_tokens=4, n_devices=2)
    hidden = 4
    buf = _make_buffer(6, hidden)
    cache.insert(0xA, [buf, buf], 6)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 6, image_hash=0xA)],
        image_token_indices=np.arange(6, dtype=np.int32),
        active_length=6,
    )
    result = cache._assemble_embeddings(
        _as_vlm_batch([ctx]),
        n_devices=2,
        empty_embeddings=[_make_buffer(0, hidden), _make_buffer(0, hidden)],
    )
    assert len(result) == 2
    for r in result:
        np.testing.assert_array_equal(r.to_numpy(), buf.to_numpy())


def test_block_fragmentation_rounds_up_to_whole_blocks() -> None:
    cache = _make_block_cache(num_blocks=4, block_tokens=4)
    cache.insert(0xA, [_make_buffer(5)], 5)
    assert cache._pool is not None
    assert cache._pool.num_free_blocks == 2
    assert cache.num_free_or_evictable_blocks == 4


def test_block_eviction_frees_blocks_on_demand() -> None:
    """Inserting past capacity evicts zero-ref LRU entries until it fits."""
    cache = _make_block_cache(num_blocks=4, block_tokens=4)
    cache.insert(0xA, [_make_buffer(8)], 8)
    cache.insert(0xB, [_make_buffer(8)], 8)
    cache.insert(0xC, [_make_buffer(12)], 12)
    assert cache.lookup(0xA) is None
    assert cache.lookup(0xB) is None
    entry_c = cache.lookup(0xC)
    assert entry_c is not None and entry_c.block_ids is not None


def test_block_eviction_respects_refcounts_owned_fallback() -> None:
    """When every block is pinned, the new entry falls back to an owned
    buffer outside the pool rather than evicting a referenced entry."""
    cache = _make_block_cache(num_blocks=4, block_tokens=4)
    req = RequestID("r1")
    cache.insert(0xA, [_make_buffer(8)], 8)
    cache.acquire(req, 0xA)
    cache.insert(0xB, [_make_buffer(8)], 8)
    cache.acquire(req, 0xB)
    buf_c = _make_buffer(4)
    entry_c = cache.insert(0xC, [buf_c], 4)
    assert entry_c.block_ids is None
    assert entry_c.embeddings is not None
    entry_a = cache.lookup(0xA)
    assert entry_a is not None and entry_a.block_ids is not None
    ctx = FakeContext(
        request_id=RequestID("r2"),
        images=[_make_image_meta(0, 4, image_hash=0xC)],
        image_token_indices=np.arange(4, dtype=np.int32),
        active_length=4,
    )
    result = cache._assemble_embeddings(
        _as_vlm_batch([ctx]), n_devices=1, empty_embeddings=[_make_buffer(0, 4)]
    )
    np.testing.assert_array_equal(result[0].to_numpy(), buf_c.to_numpy())


def test_block_entry_larger_than_pool_falls_back() -> None:
    cache = _make_block_cache(num_blocks=2, block_tokens=4)
    entry = cache.insert(0xA, [_make_buffer(12)], 12)
    assert entry.block_ids is None
    assert entry.embeddings is not None
    assert cache.lookup(0xA) is not None


def test_block_budget_below_one_block_raises() -> None:
    with pytest.raises(ValueError):
        VisionEncoderCache(
            plan=VisionCachePlan(
                bytes_per_device=8,
                hidden_size=4,
                dtype=DType.float32,
            ),
            devices=[CPU()],
            block_tokens=4,
        )


def test_block_single_block_assembly_returns_pool_view() -> None:
    """A one-image batch whose entry fits in one block assembles zero-copy:
    the returned buffer aliases the pool."""
    cache = _make_block_cache(num_blocks=4, block_tokens=8)
    buf = _make_buffer(5, 4)
    cache.insert(0xA, [buf], 5)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 5, image_hash=0xA)],
        image_token_indices=np.arange(5, dtype=np.int32),
        active_length=5,
    )
    batch = _as_vlm_batch([ctx])
    empty = [_make_buffer(0, 4)]
    result = cache._assemble_embeddings(
        batch, n_devices=1, empty_embeddings=empty
    )
    np.testing.assert_array_equal(result[0].to_numpy(), buf.to_numpy())
    marker = Buffer.from_numpy(np.full((5, 4), 7.0, dtype=np.float32))
    result[0].inplace_copy_from(marker)
    again = cache._assemble_embeddings(
        batch, n_devices=1, empty_embeddings=empty
    )
    np.testing.assert_allclose(again[0].to_numpy(), 7.0)


def test_block_multi_block_assembly_copies() -> None:
    """A multi-block entry cannot be a contiguous view, so assembly copies."""
    cache = _make_block_cache(num_blocks=4, block_tokens=4)
    buf = _make_buffer(6, 4)
    cache.insert(0xA, [buf], 6)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 6, image_hash=0xA)],
        image_token_indices=np.arange(6, dtype=np.int32),
        active_length=6,
    )
    batch = _as_vlm_batch([ctx])
    empty = [_make_buffer(0, 4)]
    result = cache._assemble_embeddings(
        batch, n_devices=1, empty_embeddings=empty
    )
    np.testing.assert_array_equal(result[0].to_numpy(), buf.to_numpy())
    zeros = Buffer.from_numpy(np.zeros((6, 4), dtype=np.float32))
    result[0].inplace_copy_from(zeros)
    again = cache._assemble_embeddings(
        batch, n_devices=1, empty_embeddings=empty
    )
    np.testing.assert_array_equal(again[0].to_numpy(), buf.to_numpy())


def test_block_partial_hit_assembly() -> None:
    """Second batch mixes a resident block entry with fresh encoder output."""
    cache = _make_block_cache(num_blocks=8, block_tokens=4)
    hidden = 4
    first = Buffer.from_numpy(np.ones((4, hidden), dtype=np.float32))
    cache._cache_and_split(
        vision_outputs=[first],
        per_image_token_counts=[4],
        image_hashes=[0xA],
        request_ids=[RequestID("r1")],
    )
    img_a = _make_image_meta(0, 4, image_hash=0xA)
    img_c = _make_image_meta(4, 8, image_hash=0xC)
    ctx = FakeContext(
        request_id=RequestID("r2"),
        images=[img_a, img_c],
        next_images=[img_a, img_c],
        image_token_indices=np.arange(8, dtype=np.int32),
        active_length=8,
    )
    vision_embeds = [
        Buffer.from_numpy(np.full((4, hidden), 3.0, dtype=np.float32))
    ]
    embeddings, indices = cache.prepare_vision_outputs(
        context_batch=_as_vlm_batch([ctx]),
        uncached_contexts=_as_vlm_batch([ctx]),
        uncached_images=[[img_c]],
        vision_embeds=vision_embeds,
        per_image_token_counts=[4],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    arr = embeddings[0].to_numpy()
    assert arr.shape == (8, hidden)
    np.testing.assert_allclose(arr[:4], 1.0)
    np.testing.assert_allclose(arr[4:], 3.0)
    assert len(indices) == 8


def test_block_assemble_skips_evicted_prior_image() -> None:
    """Block-mode variant: the evicted prior image contributes no rows."""
    cache = _make_block_cache(num_blocks=8, block_tokens=4)
    hidden = 4
    img_a = _make_image_meta(0, 2, image_hash=0xA)
    img_b = _make_image_meta(2, 5, image_hash=0xB)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[img_a, img_b],
        next_images=[img_b],
        image_token_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        processed_length=2,
        active_length=3,
    )
    vision_embeds = [
        Buffer.from_numpy(np.full((3, hidden), 7.0, dtype=np.float32))
    ]
    embeddings, indices = cache.prepare_vision_outputs(
        context_batch=_as_vlm_batch([ctx]),
        uncached_contexts=_as_vlm_batch([ctx]),
        uncached_images=[[img_b]],
        vision_embeds=vision_embeds,
        per_image_token_counts=[3],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    arr = embeddings[0].to_numpy()
    assert arr.shape == (3, hidden)
    np.testing.assert_allclose(arr, 7.0)
    assert len(indices) == arr.shape[0]
    np.testing.assert_array_equal(indices, [0, 1, 2])


def test_blocks_needed_probe_is_pure() -> None:
    cache = _make_block_cache(num_blocks=8, block_tokens=4)
    imgs = [
        _make_image_meta(0, 3, image_hash=0xA),
        _make_image_meta(3, 12, image_hash=0xB),
    ]
    assert cache.blocks_needed(imgs) == 1 + 3
    cache.insert(0xA, [_make_buffer(3)], 3)
    assert cache.blocks_needed(imgs) == 3
    assert len(cache._request_refs) == 0


def test_processed_image_ref_released() -> None:
    """A fully-processed image's ref drops during the selection walk."""
    cache = _make_block_cache(num_blocks=8, block_tokens=4)
    req = RequestID("r1")
    cache.insert(0xA, [_make_buffer(4)], 4)
    cache.insert(0xB, [_make_buffer(4)], 4)
    cache.acquire(req, 0xA)
    cache.acquire(req, 0xB)
    assert cache.num_free_or_evictable_blocks == 6
    img_a = _make_image_meta(0, 4, image_hash=0xA)
    img_b = _make_image_meta(4, 8, image_hash=0xB)
    ctx = FakeContext(
        request_id=req,
        images=[img_a, img_b],
        processed_length=4,
        active_length=4,
    )
    cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    assert _ref_count(cache, 0xA) == 0
    assert _ref_count(cache, 0xB) == 1
    assert cache.num_free_or_evictable_blocks == 7


def test_released_processed_entry_is_evictable_midrequest() -> None:
    """Block pressure may reclaim a processed image while its request runs."""
    cache = _make_block_cache(num_blocks=2, block_tokens=4)
    req = RequestID("r1")
    cache.insert(0xA, [_make_buffer(4)], 4)
    cache.acquire(req, 0xA)
    img_a = _make_image_meta(0, 4, image_hash=0xA)
    ctx = FakeContext(
        request_id=req,
        images=[img_a],
        needs_vision=True,
        processed_length=4,
        active_length=1,
    )
    cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    cache.insert(0xC, [_make_buffer(8)], 8)
    assert cache.lookup(0xA) is None
    assert cache.lookup(0xC) is not None


def test_release_request_after_processed_release_is_noop() -> None:
    cache = _make_block_cache(num_blocks=8, block_tokens=4)
    req = RequestID("r1")
    cache.insert(0xA, [_make_buffer(4)], 4)
    cache.acquire(req, 0xA)
    img_a = _make_image_meta(0, 4, image_hash=0xA)
    ctx = FakeContext(
        request_id=req, images=[img_a], processed_length=4, active_length=1
    )
    cache.get_uncached_contexts(_as_vlm_batch([ctx]))
    cache.release_request(req)
    assert _ref_count(cache, 0xA) == 0


def test_block_free_or_evictable_through_ref_cycle() -> None:
    cache = _make_block_cache(num_blocks=8, block_tokens=4)
    req = RequestID("r1")
    cache.insert(0xA, [_make_buffer(8)], 8)
    assert cache.num_free_or_evictable_blocks == 8
    cache.acquire(req, 0xA)
    assert cache.num_free_or_evictable_blocks == 6
    cache.release_request(req)
    assert cache.num_free_or_evictable_blocks == 8


def test_assemble_missing_active_image_still_raises() -> None:
    """A missing image whose tokens are in the active window is a genuine
    _cache_and_split bug and must still fail loudly, not get silent filler."""
    cache = _make_cache()
    hidden = 4
    img_a = _make_image_meta(0, 3, image_hash=0xA)  # active, never stored
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[img_a],
        next_images=[img_a],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,  # in the active window, not prior
        active_length=3,
    )
    with pytest.raises(AssertionError, match="Active in-window image"):
        cache.prepare_vision_outputs(
            context_batch=_as_vlm_batch([ctx]),
            uncached_contexts=_as_vlm_batch([ctx]),
            uncached_images=[[]],
            vision_embeds=[_make_buffer(0, hidden)],
            per_image_token_counts=[],
            n_devices=1,
            empty_embeddings=[_make_buffer(0, hidden)],
        )


class _FakeVisionModel:
    """The ``SupportsVisionEncoding`` surface ``finalize_vision_inputs`` uses."""

    def __init__(self, hidden: int = 4) -> None:
        self._hidden = hidden
        self._empties: list[Buffer] | None = None

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> None:
        return None

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: None,
    ) -> VisionEncodeResult:
        return VisionEncodeResult(
            embeddings=self.empty_vision_embeddings(devices)
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        if self._empties is None:
            self._empties = [_make_buffer(0, self._hidden) for _ in devices]
        return self._empties


def test_finalize_vision_inputs_sets_empties() -> None:
    """``vision_result=None`` (decode / text-only / graph-capture warmup)
    sets the base vision fields to the model's empty embeddings and
    zero-length indices, making them packable into ``.buffers``."""
    cache = _make_cache()
    model = _FakeVisionModel()
    inputs = ModelInputs()
    devices: list[Device] = [CPU()]

    cache.finalize_vision_inputs(model, inputs, devices, None)

    assert len(inputs.vision_embeddings) == 1
    assert inputs.vision_embeddings[0].shape[0] == 0
    assert len(inputs.vision_scatter_indices) == 1
    assert tuple(inputs.vision_scatter_indices[0].shape) == (0,)

    # The empty index buffers are cached across steps (decode hot path).
    first = inputs.vision_scatter_indices[0]
    cache.finalize_vision_inputs(model, inputs, devices, None)
    assert inputs.vision_scatter_indices[0] is first


def test_finalize_vision_inputs_sets_real_embeddings() -> None:
    """A vision-encode result sets the assembled embeddings and copies the
    merge indices to per-device buffers."""
    cache = _make_cache()
    model = _FakeVisionModel()
    inputs = ModelInputs()
    devices: list[Device] = [CPU()]
    embeds = [_make_buffer(3, 4)]
    scatter = np.array([0, 1, 2], dtype=np.int32)

    cache.finalize_vision_inputs(model, inputs, devices, (embeds, scatter))

    assert inputs.vision_embeddings is embeds
    assert len(inputs.vision_scatter_indices) == 1
    np.testing.assert_array_equal(
        inputs.vision_scatter_indices[0].to_numpy(), scatter
    )


def test_finalize_vision_inputs_empty_scatter_uses_empties() -> None:
    """A vision result with zero merge indices falls back to the cached
    zero-length index buffers instead of staging an empty copy."""
    cache = _make_cache()
    model = _FakeVisionModel()
    inputs = ModelInputs()
    devices: list[Device] = [CPU()]
    embeds = model.empty_vision_embeddings(devices)

    cache.finalize_vision_inputs(
        model, inputs, devices, (embeds, np.empty(0, dtype=np.int32))
    )

    assert inputs.vision_embeddings is embeds
    assert tuple(inputs.vision_scatter_indices[0].shape) == (0,)


# Window-bounded assembly: rows and indices are bounded by each context's
# active window instead of covering every image with OOB sentinels.


def test_windowed_merge_indices_dense_no_sentinels() -> None:
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(4, 10, image_hash=0xA)],
        image_token_indices=np.array([4, 5, 6, 7, 8, 9], dtype=np.int32),
        processed_length=6,
        active_length=10,
    )
    indices = compute_windowed_merge_indices(_as_vlm_batch([ctx]))
    np.testing.assert_array_equal(indices, [0, 1, 2, 3])


def test_windowed_merge_indices_batch_offsets_skip_non_vision() -> None:
    text = FakeContext(
        request_id=RequestID("r0"),
        images=[],
        needs_vision=False,
        active_length=3,
    )
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(1, 3, image_hash=0xB)],
        image_token_indices=np.array([1, 2], dtype=np.int32),
        processed_length=0,
        active_length=4,
    )
    indices = compute_windowed_merge_indices(_as_vlm_batch([text, ctx]))
    np.testing.assert_array_equal(indices, [4, 5])


def test_windowed_merge_indices_empty_when_no_window_overlap() -> None:
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 4, image_hash=0xA)],
        image_token_indices=np.arange(4, dtype=np.int32),
        processed_length=4,
        active_length=2,
    )
    indices = compute_windowed_merge_indices(_as_vlm_batch([ctx]))
    assert indices.size == 0


def test_assemble_slices_trailing_edge_straddle() -> None:
    """An image extending past the window contributes only its leading rows."""
    cache = _make_cache()
    hidden = 4
    buf = _make_buffer(6, hidden)
    cache.insert(0xA, [buf], 6)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 6, image_hash=0xA)],
        image_token_indices=np.arange(6, dtype=np.int32),
        processed_length=0,
        active_length=4,
    )
    result = cache._assemble_embeddings(
        _as_vlm_batch([ctx]),
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    np.testing.assert_array_equal(result[0].to_numpy(), buf.to_numpy()[:4])


def test_assemble_skips_ahead_of_window_image() -> None:
    """An image entirely beyond the window contributes nothing, so the
    in-window image assembles through the single-span zero-copy path."""
    cache = _make_cache()
    hidden = 4
    buf_a = _make_buffer(2, hidden)
    cache.insert(0xA, [buf_a], 2)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 2, image_hash=0xA),
            _make_image_meta(5, 8, image_hash=0xB),
        ],
        image_token_indices=np.array([0, 1, 5, 6, 7], dtype=np.int32),
        processed_length=0,
        active_length=3,
    )
    result = cache._assemble_embeddings(
        _as_vlm_batch([ctx]),
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    entry_a = cache.lookup(0xA)
    assert entry_a is not None
    assert result[0].shape == (2, hidden)
    np.testing.assert_array_equal(result[0].to_numpy(), buf_a.to_numpy())


def test_assemble_multi_context_distinct_windows() -> None:
    cache = _make_cache()
    hidden = 4
    buf_a = Buffer.from_numpy(np.ones((4, hidden), dtype=np.float32))
    buf_b = Buffer.from_numpy(np.full((4, hidden), 2.0, dtype=np.float32))
    cache.insert(0xA, [buf_a], 4)
    cache.insert(0xB, [buf_b], 4)
    ctx1 = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 4, image_hash=0xA)],
        image_token_indices=np.arange(4, dtype=np.int32),
        processed_length=2,
        active_length=2,
    )
    ctx2 = FakeContext(
        request_id=RequestID("r2"),
        images=[_make_image_meta(0, 4, image_hash=0xB)],
        image_token_indices=np.arange(4, dtype=np.int32),
        processed_length=0,
        active_length=3,
    )
    batch = _as_vlm_batch([ctx1, ctx2])
    result = cache._assemble_embeddings(
        batch, n_devices=1, empty_embeddings=[_make_buffer(0, hidden)]
    )
    arr = result[0].to_numpy()
    assert arr.shape == (5, hidden)
    np.testing.assert_allclose(arr[:2], 1.0)
    np.testing.assert_allclose(arr[2:], 2.0)
    indices = compute_windowed_merge_indices(batch)
    np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4])


def test_block_assemble_slices_window_across_blocks() -> None:
    """A window slice spanning a block boundary takes the ranged copy path."""
    cache = _make_block_cache(num_blocks=4, block_tokens=4)
    hidden = 4
    buf = _make_buffer(8, hidden)
    cache.insert(0xA, [buf], 8)
    other = _make_buffer(2, hidden)
    cache.insert(0xB, [other], 2)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 8, image_hash=0xA),
            _make_image_meta(8, 10, image_hash=0xB),
        ],
        image_token_indices=np.arange(10, dtype=np.int32),
        processed_length=2,
        active_length=8,
    )
    result = cache._assemble_embeddings(
        _as_vlm_batch([ctx]),
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    arr = result[0].to_numpy()
    assert arr.shape == (8, hidden)
    np.testing.assert_array_equal(arr[:6], buf.to_numpy()[2:8])
    np.testing.assert_array_equal(arr[6:], other.to_numpy())


def test_block_assemble_single_block_window_view() -> None:
    """A sub-block window slice of a one-block entry stays zero-copy."""
    cache = _make_block_cache(num_blocks=4, block_tokens=8)
    hidden = 4
    buf = _make_buffer(5, hidden)
    cache.insert(0xA, [buf], 5)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 5, image_hash=0xA)],
        image_token_indices=np.arange(5, dtype=np.int32),
        processed_length=1,
        active_length=3,
    )
    result = cache._assemble_embeddings(
        _as_vlm_batch([ctx]),
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    np.testing.assert_array_equal(result[0].to_numpy(), buf.to_numpy()[1:4])
    marker = Buffer.from_numpy(np.full((3, hidden), 9.0, dtype=np.float32))
    result[0].inplace_copy_from(marker)
    again = cache._assemble_embeddings(
        _as_vlm_batch([ctx]),
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    np.testing.assert_allclose(again[0].to_numpy(), 9.0)
