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

"""Utilities for the CPython garbage-collector and the host allocator."""

from __future__ import annotations

import ctypes
import gc
import logging
import time
from collections import Counter
from typing import Any

logger = logging.getLogger("max.serve")


class GCDebugger:
    """Times CPython GC collections and logs slow pauses.

    Registered as a ``gc.callbacks`` hook. CPython invokes the callback once
    with ``phase="start"`` immediately before a collection and once with
    ``phase="stop"`` immediately after, on the same thread that triggered the
    collection. Because a collection cannot be re-entrant (the GIL is held for
    its whole duration), a single start timestamp is sufficient to measure the
    pause.
    """

    def __init__(self, top_objects: int = 0) -> None:
        """Initializes the debugger.

        Args:
            top_objects: When greater than zero, log the ``top_objects`` most
                common live object types in the generation being collected.
                This walks every tracked object in that generation and is
                itself expensive, so leave it at ``0`` unless actively
                hunting for what is filling the heap.
        """
        self._top_objects = top_objects
        self._start_time_ns: int = time.monotonic_ns()
        self._gc_top_collected_objects: dict[str, int] | None = None
        self._gc_num_objects: int = -1

    def __call__(self, phase: str, info: dict[str, int]) -> None:
        """
        Handles a GC event (e.g. GC start or GC finish)
        """
        generation = info.get("generation")
        if generation is None:
            return
        if phase == "start":
            # Before GC started, record GC start time
            # and top collected objects
            self._start_time_ns = time.monotonic_ns()
            if self._top_objects > 0:
                # gc.get_objects() walks the entire heap and is expensive, so
                # only do it when we're interested in the top objects.
                objects = gc.get_objects(generation)
                self._gc_num_objects = len(objects)
                self._gc_top_collected_objects = self._top_object_types(objects)
        elif phase == "stop":
            # After GC finished, Record GC elapsed time and
            # optionally top collected objects
            elapsed_ms = (time.monotonic_ns() - self._start_time_ns) / 1e6
            logger.info(
                "GC generation %d took %.3fms to complete. "
                "Collected %s objects (out of %s).%s",
                generation,
                elapsed_ms,
                str(info.get("collected", "?")),
                self._gc_num_objects if self._top_objects > 0 else "?",
                (
                    f" Top collected objects: \n{self._gc_top_collected_objects}"
                    if self._gc_top_collected_objects and self._top_objects > 0
                    else ""
                ),
            )

    def _top_object_types(self, objects: list[Any]) -> dict[str, int]:
        """Returns a histogram of the most common live object types.

        Walks every object tracked in ``generation`` and tallies type names.
        Expensive; only called when ``top_objects > 0``.
        """
        counter: Counter[str] = Counter(
            type(obj).__qualname__ for obj in objects
        )
        return dict(counter.most_common(self._top_objects))


def install_gc_debugger(
    *, enabled: bool, top_objects: int = 0
) -> GCDebugger | None:
    """Installs the GC pause debugger if ``enabled``.

    Intended to be called exactly once per worker process, after logging has
    been configured. Safe to call when disabled (returns ``None``).

    Args:
        enabled: Whether to attach the GC callback at all.
        top_objects: Number of top live object types to log per collection
            (``0`` disables the expensive heap walk).

    Returns:
        The installed :class:`GCDebugger`, or ``None`` if disabled.
    """
    if not enabled:
        return None

    debugger = GCDebugger(top_objects=top_objects)
    gc.callbacks.append(debugger)
    logger.info("GC debug instrumentation enabled.")
    return debugger


def freeze_gc_heap() -> int:
    """Freezes all currently-tracked objects so GC never rescans them.

    Intended to run once in the model worker after model init + graph-capture
    warmup, before serving. The long-lived startup objects (weights, KV-cache
    metadata, graph-capture state -- on the order of 10M+ objects for Kimi) are
    promoted to the oldest generation and then moved into CPython's permanent
    "frozen" set via :func:`gc.freeze`. Subsequent collections scan only
    objects allocated after this point, which drastically cuts the
    stop-the-world pauses.

    See MXSERV-232 and https://github.com/vllm-project/vllm/pull/44363.

    Returns:
        The number of objects now in the permanent (frozen) generation.
    """
    # Promote survivors down through every generation before freezing so the
    # frozen set captures everything currently live (mirrors the vLLM PR).
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    gc.freeze()
    return gc.get_freeze_count()


def _release_free_host_memory() -> bool:
    """Returns free host-allocator pages to the OS, if the platform supports it.

    No-ops when ``malloc_trim`` is not in the process's symbol scope, which
    covers macOS and musl. Under a jemalloc or tcmalloc ``LD_PRELOAD`` the
    symbol still resolves from glibc but finds nothing to release, since those
    allocators do not strand memory this way to begin with.

    Returns:
        Whether the trim actually ran.
    """
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
    except (OSError, AttributeError):
        logger.debug("malloc_trim unavailable; skipping host memory trim.")
        return False

    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int

    start_s = time.monotonic()
    malloc_trim(0)
    logger.debug(
        "Released free host memory in %.2fs.", time.monotonic() - start_s
    )
    return True
