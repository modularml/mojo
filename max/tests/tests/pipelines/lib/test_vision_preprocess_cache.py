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

"""Tests for the byte-budgeted preprocessed-vision cache."""

from __future__ import annotations

import gc
import pickle
import threading
import weakref

import numpy as np
import numpy.typing as npt
import pytest
from max.pipelines.lib.vision_preprocess_cache import VisionPreprocessCache


class _FakeClock:
    """A monotonic clock a test advances by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestVisionPreprocessCache:
    def test_hit_returns_the_stored_payload(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(1024)
        cache.put(7, "seven", 10)

        assert cache.get(7) == "seven"
        assert cache.hits == 1
        assert cache.misses == 0

    def test_miss_returns_none_and_counts(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(1024)

        assert cache.get(7) is None
        assert cache.misses == 1

    def test_disabled_cache_never_retains(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(0)
        cache.put(7, "seven", 10)

        assert not cache.enabled
        assert cache.get(7) is None
        assert len(cache) == 0
        assert cache.total_bytes == 0

    def test_evicts_least_recently_used_to_fit_budget(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(100)
        cache.put(1, "a", 40)
        cache.put(2, "b", 40)

        # Touch 1 so 2 becomes the least recently used.
        assert cache.get(1) == "a"

        cache.put(3, "c", 40)

        assert cache.get(2) is None
        assert cache.get(1) == "a"
        assert cache.get(3) == "c"
        assert cache.total_bytes == 80

    def test_budget_is_tracked_in_bytes_not_entries(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(100)
        for key in range(10):
            cache.put(key, "small", 10)
        assert len(cache) == 10
        assert cache.total_bytes == 100

        # One large payload displaces as many small ones as it needs.
        cache.put(99, "large", 50)
        assert cache.total_bytes <= 100
        assert cache.get(99) == "large"

    def test_payload_larger_than_budget_is_dropped_not_cached(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(100)
        cache.put(1, "a", 60)
        cache.put(2, "oversized", 101)

        # The oversized payload must not flush the entries that do fit.
        assert cache.get(2) is None
        assert cache.get(1) == "a"
        assert cache.total_bytes == 60

    def test_reinsert_does_not_double_count_bytes(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(100)
        cache.put(1, "a", 40)
        cache.put(1, "a2", 30)

        assert cache.get(1) == "a2"
        assert len(cache) == 1
        assert cache.total_bytes == 30

    def test_concurrent_writers_keep_the_budget_consistent(self) -> None:
        # Preprocessing may run in worker threads, so the accounting has to
        # hold up without the caller serializing access.
        cache: VisionPreprocessCache[int] = VisionPreprocessCache(1000)
        barrier = threading.Barrier(8)

        def hammer(worker: int) -> None:
            barrier.wait()
            for i in range(100):
                key = worker * 100 + i
                cache.put(key, key, 10)
                cache.get(key)

        threads = [threading.Thread(target=hammer, args=(w,)) for w in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert cache.total_bytes <= 1000
        assert cache.total_bytes == 10 * len(cache)

    def test_survives_pickling_as_an_empty_cache(self) -> None:
        """The owning tokenizer is pickled into the spawned model worker.

        A ``threading.Lock`` cannot be pickled, so before this the whole
        server failed to start for every VLM. The cache is process-local:
        it must come back empty, usable, and with the budget intact.
        """
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(1024)
        cache.put(7, "seven", 10)
        assert len(cache) == 1

        revived: VisionPreprocessCache[str] = pickle.loads(pickle.dumps(cache))

        assert revived.enabled
        assert len(revived) == 0
        assert revived.total_bytes == 0
        assert revived.get(7) is None

        # Usable in the new process: a fresh lock was installed, not shared.
        revived.put(7, "seven", 10)
        assert revived.get(7) == "seven"
        assert revived.total_bytes == 10

    def test_disabled_cache_survives_pickling_disabled(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(0)

        revived: VisionPreprocessCache[str] = pickle.loads(pickle.dumps(cache))

        assert not revived.enabled
        revived.put(1, "one", 1)
        assert revived.get(1) is None

    def test_pickling_carries_the_idle_timeout(self) -> None:
        """Reclaim-on-idle must survive into a process that inserts entries.

        The budget alone came across before, which would have left a revived
        cache retaining entries until the budget evicted them.
        """
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(
            1024, idle_seconds=60.0
        )

        revived: VisionPreprocessCache[str] = pickle.loads(pickle.dumps(cache))

        assert revived.idle_seconds == 60.0
        revived.put(1, "one", 10)
        assert revived.total_bytes == 10


class TestGetOrPreprocess:
    """The one entry point a tokenizer uses, hoisted out of the models."""

    @staticmethod
    def _payload(fill: float) -> tuple[npt.NDArray[np.float32], int]:
        return np.full((4, 3), fill, dtype=np.float32), 7

    def test_miss_preprocesses_and_hit_reuses(self) -> None:
        cache: VisionPreprocessCache[tuple[npt.NDArray[np.float32], int]] = (
            VisionPreprocessCache(1 << 20)
        )
        calls = 0

        def preprocess() -> tuple[npt.NDArray[np.float32], int]:
            nonlocal calls
            calls += 1
            return self._payload(float(calls))

        first = cache.get_or_preprocess(1, preprocess)
        second = cache.get_or_preprocess(1, preprocess)

        assert calls == 1
        assert first is second
        assert first[0][0][0] == 1.0

    def test_none_key_never_caches(self) -> None:
        """A caller with no digest gets today's uncached behavior exactly."""
        cache: VisionPreprocessCache[tuple[npt.NDArray[np.float32], int]] = (
            VisionPreprocessCache(1 << 20)
        )
        calls = 0

        def preprocess() -> tuple[npt.NDArray[np.float32], int]:
            nonlocal calls
            calls += 1
            return self._payload(float(calls))

        first = cache.get_or_preprocess(None, preprocess)
        cache.get_or_preprocess(None, preprocess)

        assert calls == 2
        assert len(cache) == 0
        # Uncached payloads stay writable: freezing is for the shared ones.
        first[0][0][0] = 99.0

    def test_disabled_cache_preprocesses_every_time(self) -> None:
        cache: VisionPreprocessCache[tuple[npt.NDArray[np.float32], int]] = (
            VisionPreprocessCache(0)
        )
        calls = 0

        def preprocess() -> tuple[npt.NDArray[np.float32], int]:
            nonlocal calls
            calls += 1
            return self._payload(float(calls))

        cache.get_or_preprocess(1, preprocess)
        cache.get_or_preprocess(1, preprocess)

        assert calls == 2

    def test_arrays_are_frozen_on_the_miss_path_too(self) -> None:
        """One payload is handed to every request that reuses the media.

        An in-place write by any one of them would corrupt a later request's
        pixels silently, so this must be a loud failure -- and identically so
        whether the caller got a hit or the miss that populated the entry.
        """
        cache: VisionPreprocessCache[tuple[npt.NDArray[np.float32], int]] = (
            VisionPreprocessCache(1 << 20)
        )

        missed = cache.get_or_preprocess(1, lambda: self._payload(1.0))
        hit = cache.get_or_preprocess(1, lambda: self._payload(2.0))

        with pytest.raises(ValueError):
            missed[0][0][0] = 99.0
        with pytest.raises(ValueError):
            hit[0][0][0] = 99.0

    def test_sizes_every_array_a_nested_payload_retains(self) -> None:
        """Models return their own payload shapes, so the walk is structural."""
        cache: VisionPreprocessCache[object] = VisionPreprocessCache(1 << 20)
        frames = [np.zeros((2, 3), dtype=np.float32) for _ in range(3)]
        ids = np.zeros((2, 2), dtype=np.int32)

        cache.get_or_preprocess(1, lambda: (frames, {"ids": ids}, 4, "meta"))

        # 3 * 2*3*4 bytes of frames + 2*2*4 bytes of ids. The int and the str
        # are not sized: metadata is orders of magnitude below the pixels.
        assert cache.total_bytes == 72 + 16
        assert not frames[0].flags.writeable
        assert not ids.flags.writeable

    def test_a_view_is_charged_for_the_buffer_it_pins(self) -> None:
        """A slice of a batch keeps the whole batch alive, so charge for it."""
        cache: VisionPreprocessCache[npt.NDArray[np.float32]] = (
            VisionPreprocessCache(1 << 20)
        )
        batch = np.zeros((100, 10), dtype=np.float32)

        cache.get_or_preprocess(1, lambda: batch[:10])

        assert cache.total_bytes == batch.nbytes

    def test_freezing_a_view_also_freezes_the_buffer_behind_it(self) -> None:
        """Marking only the view read-only leaves the payload mutable.

        A model that slices a batched processor call still holds the batch, and
        a write through it would reach into the cached payload -- exactly the
        corruption the freeze exists to prevent, and invisible when only the
        view is marked.
        """
        cache: VisionPreprocessCache[npt.NDArray[np.float32]] = (
            VisionPreprocessCache(1 << 20)
        )
        batch = np.zeros((100, 10), dtype=np.float32)

        cached = cache.get_or_preprocess(1, lambda: batch[:10])

        assert not cached.flags.writeable
        with pytest.raises(ValueError):
            batch[0, 0] = 42.0

    def test_a_buffer_two_arrays_share_is_counted_once(self) -> None:
        cache: VisionPreprocessCache[object] = VisionPreprocessCache(1 << 20)
        batch = np.zeros((100, 10), dtype=np.float32)

        cache.get_or_preprocess(1, lambda: (batch[:10], batch[10:20]))

        assert cache.total_bytes == batch.nbytes

    def test_a_failed_preprocess_caches_nothing(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(1 << 20)

        def preprocess() -> str:
            raise RuntimeError("decode failed")

        with pytest.raises(RuntimeError):
            cache.get_or_preprocess(1, preprocess)

        assert len(cache) == 0
        assert cache.total_bytes == 0


class TestReclaimOnIdle:
    """Entries a conversation stopped using must stop costing host memory."""

    def test_collect_drops_entries_past_the_idle_timeout(self) -> None:
        clock = _FakeClock()
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(
            1024, idle_seconds=60.0, clock=clock
        )
        cache.put(1, "one", 10)

        clock.advance(59.0)
        assert cache.collect() == 0
        assert len(cache) == 1

        clock.advance(2.0)
        assert cache.collect() == 10
        assert len(cache) == 0
        assert cache.total_bytes == 0

    def test_use_refreshes_the_idle_timeout(self) -> None:
        clock = _FakeClock()
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(
            1024, idle_seconds=60.0, clock=clock
        )
        cache.put(1, "one", 10)
        cache.put(2, "two", 10)

        clock.advance(40.0)
        assert cache.get(1) == "one"
        clock.advance(30.0)

        # 1 was used 30s ago; 2 has been idle for 70s.
        assert cache.collect() == 10
        assert cache.get(1) == "one"
        assert cache.get(2) is None

    def test_collect_stops_at_the_first_live_entry(self) -> None:
        """The sweep is O(expired), not O(size): LRU order is last-used order."""
        clock = _FakeClock()
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(
            10_000, idle_seconds=60.0, clock=clock
        )
        # 50 entries a second apart, so none has reached the timeout yet and
        # the inserts themselves sweep nothing.
        for key in range(50):
            cache.put(key, "payload", 10)
            clock.advance(1.0)
        assert len(cache) == 50

        clock.advance(20.0)

        # Entry `key` was last used 70 - key seconds ago, so 0..10 have reached
        # the 60s timeout and 11..49 have not.
        assert cache.collect() == 110
        assert len(cache) == 39
        assert cache.get(10) is None
        assert cache.get(11) == "payload"

    def test_zero_idle_seconds_keeps_entries_until_the_budget_evicts(
        self,
    ) -> None:
        clock = _FakeClock()
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(
            1024, idle_seconds=0.0, clock=clock
        )
        cache.put(1, "one", 10)

        clock.advance(10_000.0)

        assert cache.collect() == 0
        assert cache.get(1) == "one"

    def test_an_insert_reclaims_what_expired_before_it(self) -> None:
        """A live entry is not evicted to make room for what a sweep would free."""
        clock = _FakeClock()
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(
            100, idle_seconds=60.0, clock=clock
        )
        cache.put(1, "one", 40)
        clock.advance(61.0)
        cache.put(2, "two", 40)

        # Entry 1 expired, so entry 3 fits alongside 2 rather than displacing
        # it: without the sweep, 40 + 40 + 40 would have evicted 2 to fit 3.
        cache.put(3, "three", 40)

        assert cache.get(1) is None
        assert cache.get(2) == "two"
        assert cache.get(3) == "three"
        assert cache.total_bytes == 80

    def test_a_lookup_reclaims_what_expired(self) -> None:
        """A hit-only workload must still reclaim.

        A conversation resending its image hits every turn and never inserts,
        so sweeping only on insert would never reclaim anything on the very
        workload this cache exists for.
        """
        clock = _FakeClock()
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(
            1024, idle_seconds=60.0, clock=clock
        )
        cache.put(1, "one", 10)
        cache.put(2, "two", 10)

        clock.advance(61.0)

        # A lookup for something else entirely sweeps both.
        assert cache.get(99) is None
        assert len(cache) == 0
        assert cache.total_bytes == 0

    def test_a_hit_is_refreshed_before_the_sweep_can_drop_it(self) -> None:
        """A request must never be answered by reprocessing what it just lost.

        The entry is past its deadline, so a sweep that ran first would drop it
        and turn a hit into a miss -- CPU spent to free nothing, since the miss
        immediately re-inserts the identical payload.
        """
        clock = _FakeClock()
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(
            1024, idle_seconds=60.0, clock=clock
        )
        cache.put(1, "one", 10)
        cache.put(2, "two", 10)

        clock.advance(61.0)

        # Entry 1 is expired, and asking for it serves it anyway.
        assert cache.get(1) == "one"
        assert cache.hits == 1
        # The same call still reclaimed the entry nobody asked for.
        assert cache.get(2) is None
        assert cache.total_bytes == 10

    def test_a_reclaimed_payload_is_actually_released(self) -> None:
        """Deferring the frees past the lock must not keep them alive.

        The sweep hands evicted entries to a list so the unmapping happens
        outside the lock; holding that list a moment too long -- or stashing it
        on the cache -- would turn the reclaim into a leak that ``total_bytes``
        would still report as freed.
        """
        clock = _FakeClock()
        cache: VisionPreprocessCache[npt.NDArray[np.float32]] = (
            VisionPreprocessCache(1 << 20, idle_seconds=60.0, clock=clock)
        )
        payload = np.zeros((64, 16), dtype=np.float32)
        witness = weakref.ref(payload)
        cache.put(1, payload, payload.nbytes)
        del payload

        assert witness() is not None
        clock.advance(61.0)
        assert cache.collect() == 64 * 16 * 4

        gc.collect()
        assert witness() is None, "the reclaimed payload is still referenced"

    def test_clear_releases_its_payloads_too(self) -> None:
        clock = _FakeClock()
        cache: VisionPreprocessCache[npt.NDArray[np.float32]] = (
            VisionPreprocessCache(1 << 20, idle_seconds=60.0, clock=clock)
        )
        payload = np.zeros((64, 16), dtype=np.float32)
        witness = weakref.ref(payload)
        cache.put(1, payload, payload.nbytes)
        del payload

        cache.clear()

        gc.collect()
        assert witness() is None

    def test_reclaim_needs_no_background_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reclaim is lazy by construction: no timer, no thread, no teardown.

        Worth pinning, because the alternative -- a reaper thread sweeping on a
        timer -- is what this deliberately does not do, and a thread would be
        easy to reintroduce without noticing. Forbidding thread construction
        rather than counting live threads, so an unrelated pool started by the
        test runner cannot make this pass or fail by accident.
        """

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("the cache must not start a thread")

        monkeypatch.setattr(threading, "Thread", forbidden)

        clock = _FakeClock()
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(
            1024, idle_seconds=60.0, clock=clock
        )
        cache.put(1, "one", 10)
        cache.get(1)
        clock.advance(61.0)
        cache.put(2, "two", 10)
        assert cache.collect() == 0
        cache.clear()

        assert cache.get(1) is None

    def test_clear_drops_everything(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(1024)
        cache.put(1, "one", 10)
        cache.put(2, "two", 20)

        assert cache.clear() == 30
        assert len(cache) == 0
        assert cache.total_bytes == 0
