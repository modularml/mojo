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

"""Byte-budgeted cache for preprocessed vision inputs."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypedDict, TypeVar

import numpy as np

from .pipeline_runtime_config import PipelineRuntimeConfig

__all__ = ["VisionPreprocessCache"]

_T = TypeVar("_T")


def _root_buffer(array: np.ndarray) -> np.ndarray:
    """The array whose buffer ``array`` keeps alive.

    A view pins the whole buffer it looks into, not just its own window, so the
    budget is charged for what the entry really retains. For an array that owns
    its data -- what a payload should hold, see
    :meth:`VisionPreprocessCache.get_or_preprocess` -- this is the array
    itself.
    """
    root = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    return root


def _freeze_and_size(payload: object) -> int:
    """Freezes every array a payload retains and returns their total bytes.

    Freezing matters because one payload is handed to every request that
    reuses the media: an in-place write by any one of them would otherwise
    corrupt a later request's pixels silently, so this converts that into a
    loud failure. Applied on the miss path too, so a hit and a miss return
    payloads that behave identically.

    Walks tuples, lists and dict values, since a payload is normally a tuple of
    a few arrays plus scalars or small metadata. Only arrays are counted: the
    metadata a processor returns alongside them is orders of magnitude smaller
    than the pixels, and sizing it would mean walking arbitrary objects.

    A view's base is frozen as well as the view. Marking only the view
    read-only leaves the buffer writable through whatever still holds the base,
    so a payload sliced out of a batched processor call would stay silently
    mutable -- the corruption this exists to prevent, in the one case where the
    caller demonstrably has another reference.

    Args:
        payload: The preprocessed payload to freeze and measure.

    Returns:
        Host bytes the payload's arrays retain, counting a buffer once even
        when several of the payload's arrays view into it.
    """
    total = 0
    seen: set[int] = set()
    stack: list[object] = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, np.ndarray):
            item.flags.writeable = False
            root = _root_buffer(item)
            root.flags.writeable = False
            if id(root) not in seen:
                seen.add(id(root))
                total += int(root.nbytes)
        elif isinstance(item, (tuple, list)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
    return total


class _PickledState(TypedDict):
    """What a pickled cache carries into the process that unpickles it.

    A known, fixed set of keys, so it is spelled out rather than left as a
    loose mapping: the two values have different types, and the reader relies
    on that.
    """

    max_bytes: int
    idle_seconds: float


@dataclass
class _Entry(Generic[_T]):
    """One cached payload, the host bytes it retains, and when it was used."""

    value: _T
    nbytes: int
    last_used: float


class VisionPreprocessCache(Generic[_T]):
    """LRU cache of preprocessed media payloads, bounded by total bytes.

    Model-agnostic: a tokenizer supplies a content key and a function that
    preprocesses one media item, and :meth:`get_or_preprocess` does the rest.
    Everything that is easy to get wrong per model -- freezing the arrays that
    are about to be shared, charging the budget for what a payload really
    retains, evicting, reclaiming on idle, and pickling into the model worker
    -- lives here rather than in each architecture.

    Keyed on the same raw-encoded-bytes digest that
    :class:`~max.pipelines.lib.vision_encoder_cache.VisionEncoderCache` uses,
    so both caches hit and miss together for a given image.

    This sits *upstream* of the vision encoder cache: it is consulted in the
    tokenizer, before preprocessing, whereas the encoder cache is consulted in
    the model worker after preprocessing has already run. A hit therefore skips
    the resize, rescale and patchify -- work the encoder cache cannot avoid no
    matter how often it hits.

    For images the decode itself is not saved on the serving path, because the
    API server already decodes every image once at admission and hands the
    tokenizer the decoded image; offline callers, which pass raw bytes through
    to the tokenizer, save the decode too. For video, which is never decoded at
    admission, a hit skips the whole decode.

    Bounded by bytes rather than by entry count (unlike
    :class:`~max.pipelines.lib.utils.BoundedCache`) because a preprocessed
    entry's size tracks the resized image area: a thumbnail and a full-budget
    image differ by more than an order of magnitude, so an entry count bounds
    host memory far too loosely to be a safe default.

    Args:
        max_bytes: Host-memory budget for cached payloads. ``0`` disables the
            cache, in which case :meth:`put` is a no-op and :meth:`get` always
            misses.
        idle_seconds: Drop an entry once it has gone this long without being
            used, so a burst of traffic does not hold host memory for the rest
            of the process's life. Swept lazily on the next lookup or insert,
            which needs no thread and no timer: the reclaim exists to stop the
            cache holding memory it is not earning, and a process that has
            stopped touching it is not competing for that memory either. ``0``
            keeps entries until the budget evicts them.
        clock: Monotonic seconds source, for tests to age entries without
            sleeping.
    """

    def __init__(
        self,
        max_bytes: int,
        *,
        idle_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_bytes = max(0, max_bytes)
        self._idle_seconds = max(0.0, idle_seconds)
        self._clock = clock
        self._cache: OrderedDict[int, _Entry[_T]] = OrderedDict()
        self._total_bytes = 0
        # Preprocessing may be dispatched to worker threads, so guard the
        # ordered dict rather than relying on the caller running single
        # threaded under the event loop.
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @classmethod
    def for_images(
        cls, runtime: PipelineRuntimeConfig
    ) -> VisionPreprocessCache[_T]:
        """Builds the preprocessed-image cache this deployment configured."""
        return cls(
            runtime.max_vision_preprocess_cache_bytes,
            idle_seconds=runtime.max_media_preprocess_cache_idle_seconds,
        )

    @classmethod
    def for_videos(
        cls, runtime: PipelineRuntimeConfig
    ) -> VisionPreprocessCache[_T]:
        """Builds the preprocessed-video cache this deployment configured.

        Budgeted separately from :meth:`for_images` because a video entry is an
        order of magnitude larger than an image one, so a shared budget would
        let a single video evict many images.
        """
        return cls(
            runtime.max_video_preprocess_cache_bytes,
            idle_seconds=runtime.max_media_preprocess_cache_idle_seconds,
        )

    @property
    def enabled(self) -> bool:
        """Whether caching is enabled (``max_bytes > 0``)."""
        return self._max_bytes > 0

    @property
    def idle_seconds(self) -> float:
        """How long an entry may go unused before it is reclaimed (``0``: never)."""
        return self._idle_seconds

    @property
    def total_bytes(self) -> int:
        """Host bytes currently retained by cached payloads."""
        return self._total_bytes

    @property
    def hits(self) -> int:
        """Lookups served from the cache."""
        return self._hits

    @property
    def misses(self) -> int:
        """Lookups that had to preprocess."""
        return self._misses

    def __len__(self) -> int:
        return len(self._cache)

    def __getstate__(self) -> _PickledState:
        """Pickles as an empty cache, carrying only the configuration.

        The tokenizer that owns this cache is pickled into the spawned model
        worker, because the pipeline factory captures it (see
        ``PIPELINE_REGISTRY.retrieve_factory``). A :class:`threading.Lock`
        cannot be pickled, so without this the whole server fails to start.

        Dropping the entries is not merely a workaround, it is the correct
        semantics: this cache is process-local. The worker preprocesses
        nothing -- it is handed already-preprocessed tensors -- so a copied
        entry would be dead weight there, and each process must own its own
        lock regardless.
        """
        return {
            "max_bytes": self._max_bytes,
            "idle_seconds": self._idle_seconds,
        }

    def __setstate__(self, state: _PickledState) -> None:
        """Restores an empty cache with a fresh lock in the new process."""
        self._max_bytes = state["max_bytes"]
        self._idle_seconds = state["idle_seconds"]
        self._clock = time.monotonic
        self._cache = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: int) -> _T | None:
        """Look up a payload by content key, refreshing LRU order.

        Also sweeps idle entries, because lookups are what a cache that is
        working does: a conversation resending its image hits every turn and
        never inserts, so sweeping only on insert would never reclaim anything
        on the workload this cache exists for.

        The entry being looked up is refreshed *before* the sweep, so a request
        can never be answered by preprocessing something this same call just
        threw away. An entry past its deadline that a request wants is served,
        not treated as a miss: the payload is a pure function of the content
        key, so it cannot be stale, and discarding one we still hold only to
        preprocess the identical bytes again would spend CPU to free nothing.
        The hit resets the clock, which is the point -- the entry turned out
        not to be idle.
        """
        dropped: list[_Entry[_T]] = []
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._cache.move_to_end(key)
                entry.last_used = self._clock()
                self._hits += 1
            else:
                self._misses += 1
            self._collect_locked(dropped)
        # Released the lock first on purpose: see _collect_locked.
        del dropped
        return entry.value if entry is not None else None

    def put(self, key: int, value: _T, nbytes: int) -> None:
        """Insert a payload, evicting least-recently-used entries to fit.

        A payload larger than the whole budget is dropped rather than cached,
        so one oversized image cannot flush every useful entry.

        Args:
            key: The content key to key on.
            value: The preprocessed payload to retain.
            nbytes: Host bytes ``value`` retains, used against the budget.
        """
        if not self.enabled or nbytes > self._max_bytes:
            return
        dropped: list[_Entry[_T]] = []
        with self._lock:
            existing = self._cache.pop(key, None)
            if existing is not None:
                self._total_bytes -= existing.nbytes
                dropped.append(existing)
            self._collect_locked(dropped)
            while self._cache and self._total_bytes + nbytes > self._max_bytes:
                _, evicted = self._cache.popitem(last=False)
                self._total_bytes -= evicted.nbytes
                dropped.append(evicted)
            self._cache[key] = _Entry(
                value=value, nbytes=nbytes, last_used=self._clock()
            )
            self._total_bytes += nbytes
        # Released the lock first on purpose: see _collect_locked.
        del dropped

    def get_or_preprocess(
        self, key: int | None, preprocess: Callable[[], _T]
    ) -> _T:
        """Returns the cached payload for ``key``, else preprocesses and caches.

        The one entry point a tokenizer needs. On a miss it runs
        ``preprocess``, freezes every array in the result, charges the budget
        for what those arrays retain, and stores it -- unless the payload is
        larger than the whole budget, which :meth:`put` drops rather than cache
        so that one oversized item cannot flush every useful entry. Such a
        payload is still returned, and still frozen, so a caller cannot tell
        from the result whether it was retained.

        Two requests for the same uncached item may both preprocess it: this
        deliberately holds no lock across ``preprocess``, since serializing on
        one would make every miss wait behind an unrelated one. The duplicate
        insert is accounted for correctly.

        The payload's arrays should own their data. A view pins the whole
        buffer it looks into, so caching one out of a batched processor call
        keeps that entire batch alive; copy it (``np.ascontiguousarray``)
        before returning it from ``preprocess`` if the processor slices a
        shared buffer.

        Args:
            key: The item's content digest, or ``None`` when the caller has
                none -- no caching is enabled, so ``preprocess`` runs and its
                result is returned untouched, exactly as it would without a
                cache.
            preprocess: Preprocesses the one media item, called only on a miss.

        Returns:
            The preprocessed payload, from the cache when it was there.
        """
        if key is None or not self.enabled:
            return preprocess()

        cached = self.get(key)
        if cached is not None:
            return cached

        value = preprocess()
        self.put(key, value, _freeze_and_size(value))
        return value

    def collect(self) -> int:
        """Drops entries unused for ``idle_seconds``, returning bytes freed.

        Runs on every lookup and every insert, so no caller has to schedule it.
        A no-op when ``idle_seconds`` is ``0``.
        """
        dropped: list[_Entry[_T]] = []
        with self._lock:
            freed = self._collect_locked(dropped)
        # Released the lock first on purpose: see _collect_locked.
        del dropped
        return freed

    def clear(self) -> int:
        """Drops every entry, returning the bytes freed."""
        with self._lock:
            freed = self._total_bytes
            # Hand the entries off and install a fresh dict, so the payloads
            # are freed after the lock is released rather than under it.
            dropped = self._cache
            self._cache = OrderedDict()
            self._total_bytes = 0
        del dropped
        return freed

    def _collect_locked(self, dropped: list[_Entry[_T]]) -> int:
        """Drops expired entries. Caller holds the lock.

        Costs one comparison when nothing has expired, and amortizes to O(1)
        per entry over the cache's life, since an entry is looked at here
        exactly once -- when it is dropped. LRU order is last-used order, so
        the oldest entries sit at the front and the first live one ends the
        sweep; nothing walks the entries behind it.

        Evicted entries are handed to ``dropped`` rather than released here,
        because releasing them is the expensive part and it must not happen
        under the lock: unmapping a full 20 GiB budget's worth of pixels
        measures 41ms, which every other thread's lookup would wait on. The
        bookkeeping itself is under a millisecond. Callers drop ``dropped``
        once they are outside the ``with`` block.

        Args:
            dropped: Collects the evicted entries, keeping them alive until the
                caller releases the lock.

        Returns:
            Host bytes freed.
        """
        if self._idle_seconds <= 0:
            return 0
        deadline = self._clock() - self._idle_seconds
        freed = 0
        while self._cache:
            key, entry = next(iter(self._cache.items()))
            if entry.last_used > deadline:
                break
            del self._cache[key]
            self._total_bytes -= entry.nbytes
            freed += entry.nbytes
            dropped.append(entry)
        return freed
