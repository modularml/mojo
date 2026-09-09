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
"""Tests for the async structured-output grammar gate."""

import threading
import time
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache.kv_connector import (
    BlockCount,
    CompletedTransfer,
)
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    GrammarBackend,
)
from max.pipelines.lib.pipeline_variants.utils import StructuredOutputHelper
from max.pipelines.modeling.types import RequestID
from max.serve.scheduler.batch_constructor.grammar_gate import AsyncGrammarGate
from max.serve.scheduler.batch_constructor.text_batch_constructor import (
    TextBatchConstructor,
)
from max.serve.scheduler.config import TokenGenerationSchedulerConfig

SCHEMA = '{"type": "object"}'


def create_mock_kv_cache() -> Mock:
    cache = Mock()
    cache.max_seq_len = 2048
    cache.page_size = 16
    cache.get_total_num_pages = Mock(return_value=128)
    cache.get_free_blocks_pct = Mock(return_value=0.5)
    cache.alloc = Mock(return_value=CompletedTransfer.load())
    cache.claim = Mock()
    cache.release = Mock()
    cache.contains = Mock(return_value=False)
    cache.pending_transfers_exist = Mock(return_value=False)
    cache.block_count = Mock(return_value=BlockCount(free=100, total=100))
    return cache


class _StubMatcher:
    """Sentinel matcher; the gate never calls into it."""

    def try_consume_tokens(self, tokens: list[int]) -> int:
        return len(tokens)

    def is_accepting(self) -> bool:
        return False

    def is_stopped(self) -> bool:
        return False

    def get_error(self) -> str | None:
        return None

    def get_grammar_warnings(self) -> Any:
        return None

    def deep_copy(self) -> "_StubMatcher":
        return self


class _ControlledBackend(GrammarBackend[Any]):
    """Backend stub whose build can block on ``release_build`` or fail."""

    name = "stub"

    def __init__(self) -> None:
        self.release_build = threading.Event()
        self.release_build.set()
        self.fail_build = False
        self.compile_json_schema_calls = 0
        self.create_matcher_calls = 0

    def compile_json_schema(self, json_schema: str) -> str:
        self.compile_json_schema_calls += 1
        return f"compiled:{json_schema}"

    def create_matcher(self, grammar: Any) -> _StubMatcher:
        self.create_matcher_calls += 1
        assert self.release_build.wait(timeout=10.0), "test deadlock"
        if self.fail_build:
            raise ValueError("stub backend refuses this grammar")
        return _StubMatcher()

    def validate_grammar(self, grammar: Any) -> None:
        pass

    def allocate_token_bitmask(
        self, batch_size: int, vocab_size: int
    ) -> np.ndarray:
        return np.zeros((batch_size, (vocab_size + 31) // 32), dtype=np.int32)

    def fill_next_token_bitmask(
        self, matcher: Any, bitmask: np.ndarray, index: int
    ) -> None:
        pass


def make_helper(
    backend: _ControlledBackend, enable_response_format_schema: bool = True
) -> StructuredOutputHelper:
    return StructuredOutputHelper(
        enabled=True,
        enable_response_format_schema=enable_response_format_schema,
        vocab_size=128,
        backend=backend,
    )


def make_gated_pipeline(helper: StructuredOutputHelper) -> Mock:
    pipeline = Mock()
    pipeline.release = Mock()
    pipeline._structured_output = helper
    return pipeline


def make_batch_constructor(pipeline: Mock) -> TextBatchConstructor:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )
    return TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=create_mock_kv_cache(),
    )


def make_context(
    json_schema: str | None = None, grammar: str | None = None
) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
        max_length=100,
        json_schema=json_schema,
        grammar=grammar,
    )


def wait_until_ready(
    gate: AsyncGrammarGate, ctx: TextContext, timeout_s: float = 10.0
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if gate.is_ready(ctx):
            return
        time.sleep(0.005)
    raise AssertionError("grammar build did not finish in time")


def test_gated_request_held_then_promoted_with_matcher() -> None:
    backend = _ControlledBackend()
    bc = make_batch_constructor(make_gated_pipeline(make_helper(backend)))
    assert bc._grammar_gate is not None
    backend.release_build.clear()

    ctx = make_context(json_schema=SCHEMA)
    bc.enqueue_new_request(ctx)

    assert bc.contains(ctx.request_id)
    assert ctx.request_id in bc._grammar_pending
    assert ctx.request_id in bc.all_ce_reqs
    inputs = bc.construct_batch()
    assert len(inputs.batches[0]) == 0
    assert ctx.request_id not in bc.replicas[0].ce_reqs

    backend.release_build.set()
    wait_until_ready(bc._grammar_gate, ctx)
    inputs = bc.construct_batch()
    assert len(inputs.batches[0]) == 1
    assert inputs.batches[0][0].request_id == ctx.request_id
    assert isinstance(ctx.matcher, _StubMatcher)
    assert ctx.request_id not in bc._grammar_pending


def test_grammar_build_records_duration_metric() -> None:
    backend = _ControlledBackend()
    bc = make_batch_constructor(make_gated_pipeline(make_helper(backend)))
    assert bc._grammar_gate is not None

    with patch(
        "max.serve.scheduler.batch_constructor.grammar_gate.METRICS"
    ) as mock_metrics:
        ctx = make_context(json_schema=SCHEMA)
        bc.enqueue_new_request(ctx)
        wait_until_ready(bc._grammar_gate, ctx)

    assert mock_metrics.structured_output_grammar_build_time.called
    (ms,) = mock_metrics.structured_output_grammar_build_time.call_args.args
    assert ms >= 0


def test_submit_grammar_build_starts_build_before_enqueue() -> None:
    """DI's decode scheduler calls ``submit_grammar_build`` at initial
    admission, before the request ever reaches ``enqueue_new_request``
    (which fires today only after prefill's KV transfer completes). The
    build must start immediately, and ``enqueue_new_request``'s own submit
    call later must be a no-op re-check rather than a second compile.
    """
    backend = _ControlledBackend()
    bc = make_batch_constructor(make_gated_pipeline(make_helper(backend)))
    assert bc._grammar_gate is not None
    backend.release_build.clear()

    ctx = make_context(json_schema=SCHEMA)
    bc.submit_grammar_build(ctx)
    assert ctx.request_id in bc._grammar_gate._futures
    assert not bc._grammar_gate.is_ready(ctx)

    backend.release_build.set()
    wait_until_ready(bc._grammar_gate, ctx)

    bc.enqueue_new_request(ctx)

    assert ctx.request_id in bc.replicas[0].ce_reqs
    assert isinstance(ctx.matcher, _StubMatcher)
    assert backend.compile_json_schema_calls == 1, (
        "enqueue_new_request's later submit must not trigger a second build"
    )


def test_release_grammar_build_purges_future() -> None:
    """DI's decode scheduler calls ``release_grammar_build`` for a request
    it drops (cancellation, TTL eviction, prefill rejection) before the
    request ever reaches ``enqueue_new_request``/``release_request``. The
    outstanding future must actually be removed from the gate's tracking,
    not merely orphaned.
    """
    backend = _ControlledBackend()
    bc = make_batch_constructor(make_gated_pipeline(make_helper(backend)))
    assert bc._grammar_gate is not None
    backend.release_build.clear()

    ctx = make_context(json_schema=SCHEMA)
    bc.submit_grammar_build(ctx)
    assert ctx.request_id in bc._grammar_gate._futures

    bc.release_grammar_build(ctx.request_id)
    assert ctx.request_id not in bc._grammar_gate._futures

    backend.release_build.set()


def test_unconstrained_request_is_never_gated() -> None:
    backend = _ControlledBackend()
    backend.release_build.clear()  # would deadlock if a build were submitted
    bc = make_batch_constructor(make_gated_pipeline(make_helper(backend)))

    ctx = make_context()
    bc.enqueue_new_request(ctx)

    assert ctx.request_id in bc.replicas[0].ce_reqs
    assert ctx.request_id not in bc._grammar_pending
    assert backend.create_matcher_calls == 0


def test_pipeline_without_structured_output_gets_no_gate() -> None:
    pipeline = Mock()
    pipeline.release = Mock()
    bc = make_batch_constructor(pipeline)

    assert bc._grammar_gate is None
    ctx = make_context(json_schema=SCHEMA)
    bc.enqueue_new_request(ctx)
    assert ctx.request_id in bc.replicas[0].ce_reqs


def test_release_while_gated_purges_future() -> None:
    backend = _ControlledBackend()
    backend.release_build.clear()
    pipeline = make_gated_pipeline(make_helper(backend))
    bc = make_batch_constructor(pipeline)
    assert bc._grammar_gate is not None

    ctx = make_context(json_schema=SCHEMA)
    bc.enqueue_new_request(ctx)
    assert bc.contains(ctx.request_id)

    bc.release_request(ctx.request_id)
    backend.release_build.set()

    assert not bc.contains(ctx.request_id)
    assert ctx.request_id not in bc._grammar_gate._futures
    pipeline.release.assert_called_once_with(ctx.request_id)


def test_failed_build_fails_request_without_admitting() -> None:
    backend = _ControlledBackend()
    backend.fail_build = True
    pipeline = make_gated_pipeline(make_helper(backend))
    bc = make_batch_constructor(pipeline)
    assert bc._grammar_gate is not None

    with patch(
        "max.serve.scheduler.batch_constructor.grammar_gate.METRICS"
    ) as mock_metrics:
        ctx = make_context(json_schema=SCHEMA)
        bc.enqueue_new_request(ctx)
        wait_until_ready(bc._grammar_gate, ctx)

    assert mock_metrics.structured_output_grammar_build_time.called

    inputs = bc.construct_batch()
    assert len(inputs.batches[0]) == 0
    assert ctx.matcher is None
    assert not bc.contains(ctx.request_id)
    failed = bc.take_grammar_failed()
    assert [req_id for req_id, _ in failed] == [ctx.request_id]
    assert "stub backend refuses this grammar" in failed[0][1]
    assert bc.take_grammar_failed() == []
    pipeline.release.assert_called_once_with(ctx.request_id)


def test_schema_request_skips_gate_when_flag_disabled() -> None:
    backend = _ControlledBackend()
    backend.release_build.clear()
    helper = make_helper(backend, enable_response_format_schema=False)
    bc = make_batch_constructor(make_gated_pipeline(helper))

    # Pre-installing a matcher would skip update_context's rejection.
    ctx = make_context(json_schema=SCHEMA)
    bc.enqueue_new_request(ctx)

    assert ctx.request_id in bc.replicas[0].ce_reqs
    assert backend.create_matcher_calls == 0
    assert ctx.matcher is None


def test_request_with_existing_matcher_skips_gate() -> None:
    backend = _ControlledBackend()
    backend.release_build.clear()
    bc = make_batch_constructor(make_gated_pipeline(make_helper(backend)))

    # E.g. a preempted request re-entering CE: reset() preserves the matcher.
    ctx = make_context(json_schema=SCHEMA)
    matcher = _StubMatcher()
    ctx.set_matcher(matcher)
    bc.enqueue_new_request(ctx)

    assert ctx.request_id in bc.replicas[0].ce_reqs
    assert ctx.matcher is matcher
    assert backend.create_matcher_calls == 0


def test_tool_grammar_request_is_gated() -> None:
    backend = _ControlledBackend()
    bc = make_batch_constructor(make_gated_pipeline(make_helper(backend)))
    assert bc._grammar_gate is not None

    ctx = make_context(grammar="root ::= object")
    bc.enqueue_new_request(ctx)
    wait_until_ready(bc._grammar_gate, ctx)
    inputs = bc.construct_batch()

    assert len(inputs.batches[0]) == 1
    assert isinstance(ctx.matcher, _StubMatcher)
    assert backend.compile_json_schema_calls == 0


def test_build_matcher_dispatch() -> None:
    backend = _ControlledBackend()
    helper = make_helper(backend)

    helper.build_matcher("root ::= object", None)
    assert backend.compile_json_schema_calls == 0
    assert backend.create_matcher_calls == 1

    helper.build_matcher(None, SCHEMA)
    assert backend.compile_json_schema_calls == 1
    assert backend.create_matcher_calls == 2
