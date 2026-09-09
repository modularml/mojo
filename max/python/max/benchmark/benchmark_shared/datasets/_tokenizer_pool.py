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

"""Context-managed spawn process pool for slow tokenizer work.

Some benchmark tokenizers (Kimi K2.5) are pure-Python and hold the GIL on
every `encode`/`decode` call. The dataset-generation code paths perform many
thousands of these calls per benchmark, so we offload them to a process pool.
Workers re-load the tokenizer once each via `_init_encoder` and then service
encode/decode/session-build tasks for the lifetime of the `TokenizerPool`.

Use as a context manager so the pool is closed and joined deterministically
on exit. Worker processes are created lazily on the first encode/decode/map
call, so constructing a `TokenizerPool` you never use (e.g. when a dry-run
test mocks out the dataset method) is free.
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing as mp
import os
from collections.abc import Callable, Iterable
from typing import TypeVar

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

_T = TypeVar("_T")
_R = TypeVar("_R")

# Per-worker tokenizer instance, populated by `_init_encoder` when each
# spawned worker process starts. Reads via `worker_tokenizer()` from inside
# worker callables (e.g. the multiturn session builder).
_WORKER_TOK: PreTrainedTokenizerBase | None = None

# Architectures resolved once in the parent and forwarded to each worker via
# `_init_encoder`, so `_default_loader` can skip the redundant (and noisy)
# per-worker `AutoConfig` Hub lookup. `None` means "resolve in the worker".
_WORKER_ARCHITECTURES: list[str] | None = None


def _default_loader(
    name_or_path: str,
    model_max_length: int | None,
    trust_remote_code: bool,
    revision: str | None,
) -> PreTrainedTokenizerBase:
    """Production worker loader: re-loads the real tokenizer by name."""
    from max.benchmark.benchmark_shared.utils import get_tokenizer

    return get_tokenizer(
        name_or_path,
        revision=revision,
        model_max_length=model_max_length,
        trust_remote_code=trust_remote_code,
        architectures=_WORKER_ARCHITECTURES,
    )


_LoaderFn = Callable[
    [str, "int | None", bool, "str | None"], PreTrainedTokenizerBase
]


def _init_encoder(
    name_or_path: str,
    model_max_length: int | None,
    trust_remote_code: bool,
    revision: str | None,
    architectures: list[str] | None,
    loader: _LoaderFn,
    log_queue: mp.Queue[logging.LogRecord],
    log_level: int,
) -> None:
    """Worker initializer: wire logs back to the parent, then build a tokenizer."""
    # The standard multiprocessing-logging pattern: workers push records
    # into a queue; a `QueueListener` thread in the parent dispatches them
    # through the parent's real handlers. See Python docs, "Logging to a
    # single file from multiple processes".
    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    global _WORKER_ARCHITECTURES
    _WORKER_ARCHITECTURES = architectures
    global _WORKER_TOK
    try:
        _WORKER_TOK = loader(
            name_or_path, model_max_length, trust_remote_code, revision
        )
    except Exception as exc:
        # A worker's tokenizer load can fail under HF_HUB_OFFLINE (e.g. a
        # remote-code file this worker's cache race didn't pick up in time).
        # The pool's worker-handler thread already tolerates a dead worker
        # by spawning a replacement while survivors drain the task queue, so
        # let this worker exit -- but via `SystemExit`, which
        # `BaseProcess._bootstrap` handles without dumping a raw traceback
        # to stderr, unlike a plain exception escaping the initializer.
        logging.getLogger(__name__).warning(
            "Worker failed to load tokenizer %r, exiting worker: %s",
            name_or_path,
            exc,
        )
        raise SystemExit(1) from None


def _encode_len(text: str) -> int:
    assert _WORKER_TOK is not None
    return len(_WORKER_TOK.encode(text, add_special_tokens=False))


def _encode_ids(text: str) -> list[int]:
    """Worker callable: full `tokenizer.encode(text)` (ids, not lengths).

    Pass to `pool.map(_encode_ids, texts)` when callers need the actual
    token IDs and not just the count.
    """
    assert _WORKER_TOK is not None
    return _WORKER_TOK.encode(text)


def _decode_text(ids: list[int]) -> str:
    assert _WORKER_TOK is not None
    return _WORKER_TOK.decode(ids, skip_special_tokens=False)


def worker_tokenizer() -> PreTrainedTokenizerBase:
    """Return the tokenizer loaded inside the current worker process.

    Worker callables that need direct tokenizer access (e.g. the multiturn
    session builder) call this to read the worker-local tokenizer instance.
    """
    assert _WORKER_TOK is not None
    return _WORKER_TOK


class TokenizerPool:
    """Spawn process pool bound to a specific tokenizer.

    Use as a context manager so the pool is closed cleanly on exit:

        with TokenizerPool(tokenizer) as pool:
            lens = pool.encode_lens(texts)
            sessions = pool.map(_build_session, args_list)

    The tokenizer is captured at construction; mismatch is structurally
    impossible because callers receive only the pool and read
    `pool.tokenizer` for parent-side work.

    Worker processes are spawned lazily on the first encode/decode/map
    call. Constructing a pool you never use is free.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        loader: _LoaderFn | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        # Make sure to always leave at least one core free for the main thread.
        # If the core count is unknown, limit to 1 core only.
        cpu_count = os.cpu_count()
        if cpu_count and cpu_count > 1:
            cpu_limit = cpu_count - 1
        else:
            cpu_limit = 1

        # Tests (and other constrained environments) can set the
        # default worker count via env var.
        env_count = os.environ.get("MAX_BENCHMARK_MAX_TOKENIZER_THREADS")
        if env_count and env_count.isdigit():
            workers = int(env_count)
        else:
            # Default to 16 cores if no value requested.
            workers = 16
        self._workers = min(workers, cpu_limit)
        self._loader = loader if loader is not None else _default_loader
        self._pool: mp.pool.Pool | None = None
        self._log_listener: logging.handlers.QueueListener | None = None

    def __enter__(self) -> TokenizerPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the pool and wait for workers to exit cleanly."""
        if self._pool is None:
            return
        # `close()` stops the pool from accepting new tasks; `join()` waits
        # for in-flight tasks to drain and workers to exit. Unlike
        # `terminate()`, this gives workers a chance to release any
        # Rayon/transformers C-extension state without leaking.
        self._pool.close()
        self._pool.join()
        self._pool = None
        if self._log_listener is not None:
            self._log_listener.stop()
            self._log_listener = None

    def _ensure_pool(self) -> mp.pool.Pool:
        if self._pool is None:
            ctx = mp.get_context("spawn")
            # Standard multiprocessing-logging plumbing: workers push log
            # records into the queue via a `QueueHandler`; a listener
            # thread in the parent dispatches them through the parent's
            # real handlers. See Python docs, `logging.handlers.QueueListener`.
            log_queue: mp.Queue[logging.LogRecord] = ctx.Queue(-1)
            root = logging.getLogger()
            self._log_listener = logging.handlers.QueueListener(
                log_queue, *root.handlers, respect_handler_level=True
            )
            self._log_listener.start()
            revision = getattr(self.tokenizer, "_resolved_revision", None)
            # Forward the parent's resolved architectures so workers skip the
            # redundant (and, when the Hub is unreachable, noisy) AutoConfig
            # probe. `None` leaves each worker to resolve them itself.
            architectures = getattr(
                self.tokenizer, "_resolved_architectures", None
            )
            # huggingface_hub caches HF_HUB_OFFLINE at import time, so set it
            # before spawning workers (not inside the initializer — too late).
            # Restore the parent's env afterward so we don't leak this global
            # into the rest of the program.
            prev_hf_offline = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                self._pool = ctx.Pool(
                    processes=self._workers,
                    initializer=_init_encoder,
                    initargs=(
                        self.tokenizer.name_or_path,
                        self.tokenizer.model_max_length,
                        True,
                        revision,
                        architectures,
                        self._loader,
                        log_queue,
                        root.level,
                    ),
                )
            finally:
                if prev_hf_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev_hf_offline
        return self._pool

    def _chunksize(self, n: int) -> int:
        return max(1, n // (self._workers * 4))

    def encode_lens(self, texts: list[str]) -> list[int]:
        """Encode each text and return only its token count.

        Workload is fanned out across the pool's workers.
        """
        return self.map(_encode_len, texts)

    def decode_texts(self, id_lists: list[list[int]]) -> list[str]:
        """Decode each token-id list to text.

        Workload is fanned out across the pool's workers.
        """
        return self.map(_decode_text, id_lists)

    def map(self, fn: Callable[[_T], _R], items: Iterable[_T]) -> list[_R]:
        """Run `fn` across the pool for every element of `items`.

        Used by coarser per-task workers (e.g. session builders) that
        access their tokenizer via `worker_tokenizer()`.
        """
        items_list = list(items)
        if not items_list:
            return []
        pool = self._ensure_pool()
        return pool.map(
            fn, items_list, chunksize=self._chunksize(len(items_list))
        )
