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

"""Async structured-output grammar gate for the batch constructor.

Building a grammar matcher can take seconds for complex schemas; run inline
on the shared decode thread it stalls every co-batched request. The gate
builds matchers on worker threads while the batch constructor holds the
request out of CE admission. All gate methods run on the scheduler thread;
only the build runs on workers, and both grammar backends release the GIL
during it.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor

from max.pipelines.context import GrammarMatcher
from max.pipelines.context.context import TextContext
from max.pipelines.lib.pipeline_variants.utils import (
    StructuredOutputHelper,
    get_structured_output_helper,
)
from max.pipelines.modeling.types import RequestID
from max.serve.telemetry.metrics import METRICS

logger = logging.getLogger("max.serve")


class AsyncGrammarGate:
    """Builds grammar matchers off-thread and tracks per-request readiness."""

    def __init__(self, structured_output: StructuredOutputHelper) -> None:
        self._structured_output = structured_output
        self._executor: ThreadPoolExecutor | None = None
        self._futures: dict[RequestID, Future[GrammarMatcher]] = {}

    @classmethod
    def create(cls, pipeline: object) -> AsyncGrammarGate | None:
        """Returns a gate, or ``None`` without enabled structured output."""
        helper = get_structured_output_helper(pipeline)
        if helper is None or not helper.enabled or helper.backend is None:
            return None
        return cls(helper)

    def _wants_precompile(self, ctx: TextContext) -> bool:
        """Whether to build this request's matcher off-thread.

        Mirrors ``update_context``'s checks: a request it would reject must
        not get a matcher pre-installed, or the rejection is skipped.
        """
        if ctx.matcher is not None:
            return False
        helper = self._structured_output
        if ctx.grammar:
            return (
                not ctx.requires_structured_output_flag
                or helper.enable_response_format_schema
            )
        if ctx.json_schema is not None:
            return helper.enable_response_format_schema
        return False

    def submit(self, ctx: TextContext) -> None:
        """Starts the off-thread matcher build if needed. Idempotent."""
        if not self._wants_precompile(ctx):
            return
        if ctx.request_id in self._futures:
            return
        if self._executor is None:
            # ~cpu/2 workers, matching vLLM's structured-output pool.
            max_workers = max(1, (os.cpu_count() or 2) // 2)
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="grammar-compile",
            )
        self._futures[ctx.request_id] = self._executor.submit(
            self._timed_build_matcher, ctx.grammar, ctx.json_schema
        )

    def _timed_build_matcher(
        self, grammar: str | None, json_schema: str | None
    ) -> GrammarMatcher:
        """Runs on a worker thread; records the build's own duration.

        Submitted at admission time, this build is meant to overlap
        prefill's round trip rather than land on the handoff's critical
        path -- this metric is how to tell whether a given build actually
        won that race.
        """
        start = time.monotonic()
        try:
            return self._structured_output.build_matcher(grammar, json_schema)
        finally:
            METRICS.structured_output_grammar_build_time(
                (time.monotonic() - start) * 1000
            )

    def is_ready(self, ctx: TextContext) -> bool:
        """Whether a request's build has finished. Non-blocking.

        Nothing submitted reports ready, so unconstrained requests are never
        held. A failed build also reports ready; see :meth:`install_ready`.
        """
        future = self._futures.get(ctx.request_id)
        return future is None or future.done()

    def install_ready(self, ctx: TextContext) -> str | None:
        """Installs a finished build's matcher on the context.

        Returns a user-facing error message when the build failed (logged);
        the caller must fail the request instead of admitting it. ``None``
        on success, including when nothing was submitted.
        """
        future = self._futures.pop(ctx.request_id, None)
        if future is None:
            return None
        assert future.done()
        try:
            matcher = future.result()
        except Exception as e:
            logger.error(
                "Grammar build failed for request %s; failing the request.",
                ctx.request_id,
                exc_info=True,
            )
            backend = self._structured_output.backend
            assert backend is not None
            return (
                f"Grammar provided in request cannot be compiled. "
                f"From {backend.name}: {str(e).strip()}"
            )
        self._structured_output.install_matcher(ctx, matcher)
        return None

    def release(self, request_id: RequestID) -> None:
        """Drops a request's outstanding build, if any. Idempotent."""
        future = self._futures.pop(request_id, None)
        if future is not None:
            future.cancel()
