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
"""Tests for pipeline_variants/utils.py."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import max.pipelines.lib.pipeline_variants.structured_output_backend as _sob
import numpy as np
import numpy.typing as npt
import pytest
from max import _xgrammar as xgrammar
from max.pipelines.context import (
    FUTURE_TOKEN,
    EOSTracker,
    GenerationStatus,
    GrammarMatcher,
    StructuredOutputRegionDelimiters,
    TextContext,
    TokenBuffer,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    GrammarBackend,
    XgrammarBackend,
)
from max.pipelines.lib.pipeline_variants.utils import (
    CommittedSpanSnapshot,
    StructuredOutputHelper,
    build_response,
    update_spec_decode_context_and_prepare_responses,
)
from max.pipelines.lib.tool_parsing import StructuralTagToolParser, register
from max.pipelines.modeling.types import ParsedToolCall, RequestID


class _RecordingMatcher(GrammarMatcher):
    """Minimal GrammarMatcher that records the tokens fed to it."""

    def __init__(self) -> None:
        self.consumed: list[list[int]] = []

    def try_consume_tokens(self, tokens: list[int]) -> int:
        self.consumed.append(list(tokens))
        return len(tokens)

    def is_accepting(self) -> bool:
        return True

    def is_stopped(self) -> bool:
        return False

    def get_error(self) -> str | None:
        return None

    def get_grammar_warnings(self) -> Any:
        return None

    def deep_copy(self) -> _RecordingMatcher:
        # Speculative walks (Part 2) use a copy, never the original.
        return _RecordingMatcher()


class _NoopBackend(GrammarBackend[Any]):
    """GrammarBackend stub so Part 2's fills don't touch llguidance."""

    name = "noop"

    def compile_json_schema(self, json_schema: str) -> Any:
        return None

    def create_matcher(self, grammar: Any) -> GrammarMatcher:
        return _RecordingMatcher()

    def validate_grammar(self, grammar: Any) -> None:
        return None

    def allocate_token_bitmask(
        self, batch_size: int, vocab_size: int
    ) -> npt.NDArray[np.int32]:
        return np.zeros((batch_size, (vocab_size + 31) // 32), dtype=np.int32)

    def fill_next_token_bitmask(
        self,
        matcher: GrammarMatcher,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        pass


class _DeadMatcher(_RecordingMatcher):
    """Stopped without accepting: llguidance's state after a rejected token.

    Its mask is all-zero, since no token can continue the grammar.
    """

    def is_accepting(self) -> bool:
        return False

    def is_stopped(self) -> bool:
        return True

    def deep_copy(self) -> _DeadMatcher:
        return _DeadMatcher()


class _CompletedMatcher(_RecordingMatcher):
    """Stopped *and* accepting: a satisfied grammar, whose mask allows EOS."""

    def is_stopped(self) -> bool:
        return True

    def deep_copy(self) -> _CompletedMatcher:
        return _CompletedMatcher()


class _FillCountingBackend(_NoopBackend):
    """Backend whose fill is observable, so a skipped fill can be asserted on."""

    def __init__(self) -> None:
        self.fills = 0

    def fill_next_token_bitmask(
        self,
        matcher: GrammarMatcher,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        self.fills += 1
        # Any non--1 value marks the slot as "the backend wrote here".
        bitmask[index, :] = 0


def create_text_context(prompt_len: int, max_length: int) -> TextContext:
    """Create a TextContext for testing."""
    tokens = np.arange(prompt_len, dtype=np.int64)
    return TextContext(
        request_id=RequestID(),
        max_length=max_length,
        tokens=TokenBuffer(tokens),
    )


def advance_to_processed(ctx: TextContext) -> None:
    """Advance context so prompt tokens are marked as processed.

    After this call, processed_length equals the original token count.
    """
    ctx.update_with_future_token()
    ctx.realize_future_token(new_token=99, log_probabilities=None)


@register("_test_section_parser")
class _SectionParser(StructuralTagToolParser):
    """Minimal section-wrapped parser for region-tag derivation tests."""

    SECTION_BEGIN = "<sec>"
    SECTION_END = "</sec>"

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        return []


@register("_test_to_eos_parser")
class _ToEosParser(_SectionParser):
    """Section parser that opts into enforcement-to-EOS."""

    ENFORCE_TOOL_REGION_TO_EOS = True


class TestGetToolRegionTags:
    """``StructuredOutputHelper._get_tool_region_tags``."""

    def test_section_parser_returns_section_pair(self) -> None:
        tags = StructuredOutputHelper._get_tool_region_tags(
            "_test_section_parser"
        )
        assert tags == ("<sec>", "</sec>")

    def test_to_eos_parser_has_no_end_tag(self) -> None:
        """With no end tag, enforcement never flips off: the completed
        grammar masks everything but EOS, so the turn ends with its single
        tool-call section (e.g. MiniMax-M3)."""
        tags = StructuredOutputHelper._get_tool_region_tags(
            "_test_to_eos_parser"
        )
        assert tags == ("<sec>", None)


class TestBuildResponse:
    """Tests for build_response function."""

    def test_marks_maximum_length_when_at_limit(self) -> None:
        """Context is marked MAXIMUM_LENGTH when at the boundary."""
        max_seq_len = 100
        # Create context with 99 tokens
        ctx = create_text_context(prompt_len=99, max_length=max_seq_len)

        # Advance to processed state: processed_length = 99
        advance_to_processed(ctx)
        assert ctx.tokens.processed_length == 99

        # current_length = 99 + 1 = 100 >= 100 → MAXIMUM_LENGTH
        build_response([ctx], max_seq_len=max_seq_len)

        assert ctx.status == GenerationStatus.MAXIMUM_LENGTH

    def test_does_not_mark_when_below_limit(self) -> None:
        """Context is not marked when there's room for growth."""
        max_seq_len = 100
        ctx = create_text_context(prompt_len=50, max_length=max_seq_len)

        # Advance to processed state: processed_length = 50
        advance_to_processed(ctx)
        assert ctx.tokens.processed_length == 50

        # current_length = 50 + 1 = 51 < 100 → not done
        build_response([ctx], max_seq_len=max_seq_len)

        assert ctx.status != GenerationStatus.MAXIMUM_LENGTH

    def test_does_not_reserve_speculative_slack(self) -> None:
        """``build_response`` no longer early-stops for speculative growth.

        It marks ``MAXIMUM_LENGTH`` only when there is no room for even one
        more token. Reserving ``num_speculative_tokens + 1`` worst-case slack
        here truncated output up to ``num_speculative_tokens`` tokens short of
        the cap; the commit loop in
        ``update_spec_decode_context_and_prepare_responses`` is now what keeps
        a step from overshooting, so a near-limit context stays live.
        """
        max_seq_len = 100

        # processed_length = 96, current_length = 97. Even with 3 spec tokens
        # in flight (worst-case growth 4), build_response must not stop here:
        # 97 < 100 → still live.
        ctx = create_text_context(prompt_len=96, max_length=max_seq_len)
        advance_to_processed(ctx)
        assert ctx.tokens.processed_length == 96

        build_response([ctx], max_seq_len=max_seq_len)

        assert ctx.status != GenerationStatus.MAXIMUM_LENGTH

    def test_respects_per_request_max_length(self) -> None:
        """Per-request max_length is respected when lower than global."""
        global_max_seq_len = 100
        per_request_max = 50

        # Create context with per-request limit of 50
        ctx = create_text_context(prompt_len=49, max_length=per_request_max)
        advance_to_processed(ctx)
        assert ctx.tokens.processed_length == 49

        # current_length = 49 + 1 = 50 >= 50 → MAXIMUM_LENGTH
        build_response([ctx], max_seq_len=global_max_seq_len)

        assert ctx.status == GenerationStatus.MAXIMUM_LENGTH


class TestSpecDecodeStopsExactlyAtPerRequestCap:
    """Regression for CENG-827.

    With Eagle spec decoding on, ``build_response`` reserved a full
    ``num_speculative_tokens + 1`` worst-case chunk of slack against the
    *per-request* cap (``context.max_length``), so a request was marked
    ``MAXIMUM_LENGTH`` up to ``num_speculative_tokens`` tokens before it
    actually reached its cap -- the final (possibly partial) accept chunk
    that would have landed exactly on the cap never got a chance to run.
    That slack must only be reserved against the hard model/KV limit
    (``max_seq_len``), never against the per-request cap.
    """

    def test_stops_exactly_at_cap_on_non_chunk_aligned_boundary(self) -> None:
        prompt_len = 10
        num_speculative_tokens = 3
        max_gen_tokens = 20
        # Hard model/KV limit is far above the per-request cap, so only the
        # per-request cap should ever gate termination in this test.
        max_seq_len = 10_000

        ctx = create_text_context(
            prompt_len=prompt_len, max_length=prompt_len + max_gen_tokens
        )

        # Phase 1: overlap scheduler always appends a placeholder future
        # token before a request's first spec-decode verify step.
        ctx.update_with_future_token()

        # Per-step accepted-draft counts (out of 3 drafts). Each full cycle
        # (a placeholder future token followed by its verify step) commits
        # num_accept + 1 tokens (accepted drafts + bonus token) -- these sum
        # to exactly max_gen_tokens (20). The step sizes -- 4, 4, 4, 4, 2, 2
        # -- don't line up on a fixed 4-token (num_speculative_tokens + 1)
        # chunk grid, so the cap is reached by a partial (2-token) chunk,
        # not a full one: exactly the case the per-token accept loop in
        # ``update_spec_decode_context_and_prepare_responses`` truncates to
        # land precisely on the cap -- if ``build_response`` lets that final
        # step run at all.
        accept_counts = [3, 3, 3, 3, 1, 1]
        assert all(c <= num_speculative_tokens for c in accept_counts)
        assert sum(c + 1 for c in accept_counts) == max_gen_tokens

        for num_accept in accept_counts:
            if ctx.is_done:
                # The cap was already reached (e.g. by a placeholder future
                # token landing exactly on it); the scheduler would not have
                # driven a further step for this request.
                break
            update_spec_decode_context_and_prepare_responses(
                draft_tokens=np.array([[101, 102, 103]], dtype=np.int32),
                next_draft_tokens=np.array([[201, 202, 203]], dtype=np.int32),
                num_accepted_draft_tokens=np.array(
                    [num_accept], dtype=np.int32
                ),
                next_tokens=np.array([999], dtype=np.int32),
                context_batch=[ctx],
                max_seq_len=max_seq_len,
            )
            if ctx.is_done:
                break
            # Mirrors the overlap scheduler: only a still-active request gets
            # a placeholder future token for its next verify step.
            ctx.update_with_future_token()

        assert ctx.tokens.generated_length == max_gen_tokens, (
            f"expected exactly {max_gen_tokens} generated tokens, got "
            f"{ctx.tokens.generated_length}"
        )
        assert ctx.status == GenerationStatus.MAXIMUM_LENGTH

    def test_stops_exactly_at_max_seq_len(self) -> None:
        """The model/KV limit truncates a straddling accept chunk.

        A request whose per-request cap coincides with ``max_seq_len`` (i.e. it
        fills the whole model context) must have its commit capped exactly at
        ``max_seq_len``. This is the KV-safety case Option 3 relies on: the step
        may over-speculate past ``max_seq_len`` into the pool's
        ``num_draft_tokens`` slack, but the committed length must land on
        ``max_seq_len`` so the next step never exceeds the pool.

        (``max_length`` can never exceed ``max_seq_len`` at runtime -- the
        registry clamps it and ``upper_bounded_default`` enforces the invariant
        -- so the binding case is exactly ``max_length == max_seq_len``.)
        """
        prompt_len = 10
        num_speculative_tokens = 3
        # Per-request cap coincides with the model/KV limit.
        max_seq_len = 30
        max_length = max_seq_len
        expected_generated = max_seq_len - prompt_len  # 20

        ctx = create_text_context(prompt_len=prompt_len, max_length=max_length)
        ctx.update_with_future_token()

        # Step sizes 4,4,4,4,2,2 sum to 20 and cross max_seq_len on a partial
        # (2-token) chunk, exercising mid-chunk truncation.
        accept_counts = [3, 3, 3, 3, 1, 1]
        assert all(c <= num_speculative_tokens for c in accept_counts)
        assert sum(c + 1 for c in accept_counts) == expected_generated

        for num_accept in accept_counts:
            if ctx.is_done:
                break
            update_spec_decode_context_and_prepare_responses(
                draft_tokens=np.array([[101, 102, 103]], dtype=np.int32),
                next_draft_tokens=np.array([[201, 202, 203]], dtype=np.int32),
                num_accepted_draft_tokens=np.array(
                    [num_accept], dtype=np.int32
                ),
                next_tokens=np.array([999], dtype=np.int32),
                context_batch=[ctx],
                max_seq_len=max_seq_len,
            )
            if ctx.is_done:
                break
            ctx.update_with_future_token()

        assert ctx.tokens.generated_length == expected_generated, (
            f"expected exactly {expected_generated} generated tokens, got "
            f"{ctx.tokens.generated_length}"
        )
        assert len(ctx.tokens) == max_seq_len
        assert ctx.status == GenerationStatus.MAXIMUM_LENGTH


class TestTokensForConsume:
    """``StructuredOutputHelper._tokens_for_consume``.

    On the conditional-enforcement flip-on (``tool_choice=auto``), the fresh
    matcher must consume the whole tool-call start marker, not just the token
    that completed it — otherwise multi-token / namespace-prefixed markers
    (e.g. MiniMax-M3's ``NS<tool_call>``) reject and enforcement falls open.
    """

    @staticmethod
    def _helper(start_token_ids: list[int]) -> StructuredOutputHelper:
        return StructuredOutputHelper(
            tool_call_region_delimiters=StructuredOutputRegionDelimiters(
                start_token_ids=start_token_ids,
                end_token_ids=[999],
            )
        )

    def test_flip_on_feeds_full_multitoken_marker(self) -> None:
        # M3-style NS<tool_call> = two tokens; flip-on feeds both.
        helper = self._helper([200058, 200052])
        assert helper._tokens_for_consume(200052, was_enforced=False) == [
            200058,
            200052,
        ]

    def test_already_enforced_feeds_single_token(self) -> None:
        helper = self._helper([200058, 200052])
        assert helper._tokens_for_consume(77, was_enforced=True) == [77]

    def test_single_token_marker_is_noop(self) -> None:
        # Single-token markers (e.g. Kimi) feed just the token even on flip-on.
        helper = self._helper([42])
        assert helper._tokens_for_consume(42, was_enforced=False) == [42]

    def test_no_delimiters_feeds_single_token(self) -> None:
        helper = StructuredOutputHelper()
        assert helper._tokens_for_consume(5, was_enforced=False) == [5]


class TestAdvanceFsmAndComputeBitmasks:
    """``StructuredOutputHelper.advance_fsm_and_compute_bitmasks``.

    A row that can't be attributed to a producing-batch slot (absent, or reset
    to an initial prompt after enqueue) must be degraded to the all-valid -1
    reset rather than raising -- a raise blanket-unconstrains the whole batch.
    """

    @staticmethod
    def _decoding_ctx() -> TextContext:
        """A continuing (non-initial-prompt) unconstrained decode row."""
        ctx = create_text_context(prompt_len=4, max_length=128)
        # Mirror handle_prefill_response on the decode engine: applying the
        # first generated token makes generated_length > 0 and clears
        # is_initial_prompt -- exactly the state of a KV-transferred row.
        ctx.update(new_token=99)
        assert ctx.tokens.generated_length > 0
        assert not ctx.is_initial_prompt
        return ctx

    @staticmethod
    def _empty_bitmask(num_rows: int, num_positions: int) -> np.ndarray:
        # Packed int32 bitmask, [rows, K+1, ceil(vocab/32)]. Vocab is small;
        # the callback resets each row to -1 before any fill, so the only
        # requirement is a writable rectangle of the right outer shape.
        return np.zeros((num_rows, num_positions, 1), dtype=np.int32)

    def test_row_absent_from_producing_batch_degrades_not_raises(self) -> None:
        """A row absent from the producing batch is degraded, not raised on."""
        helper = StructuredOutputHelper(enabled=True, vocab_size=16)

        producer = self._decoding_ctx()
        transferred = self._decoding_ctx()

        producing_batch = [producer]
        consuming_batch = [transferred, producer]

        # Producing-batch-shaped spec-decode arrays: K=1 draft token.
        accepted = np.zeros((1, 1), dtype=np.int64)
        num_accepted = np.zeros((1,), dtype=np.int64)
        bonus = np.full((1,), 99, dtype=np.int64)
        next_draft = np.zeros((1, 1), dtype=np.int64)
        # Sentinel: every reached row is reset to -1, so a row still holding 7
        # afterwards was never reached (the loop aborted early).
        bitmask_out = self._empty_bitmask(len(consuming_batch), num_positions=2)
        bitmask_out[:] = 7

        helper.advance_fsm_and_compute_bitmasks(
            context_batch=producing_batch,
            accepted_draft_tokens=accepted,
            num_accepted=num_accepted,
            bonus_tokens=bonus,
            next_draft_tokens=next_draft,
            bitmask_out=bitmask_out,
            output_context_batch=consuming_batch,
        )

        # No raise; the degraded row and the trailing row are both reached.
        assert (bitmask_out == -1).all()

    def test_row_preempted_in_flight_degrades_not_raises(self) -> None:
        """Regression: a row reset (preempted) after enqueue is degraded, not
        raised on, so the rest of the batch keeps its constraints."""
        helper = StructuredOutputHelper(enabled=True, vocab_size=16)

        survivor = self._decoding_ctx()
        preempted = self._decoding_ctx()

        producing_batch = [preempted, survivor]
        consuming_batch = [preempted, survivor]

        # Mirror an in-flight preemption: the scheduler resets the context
        # (requeuing it to context encoding) after the callback was enqueued.
        preempted.reset()
        assert preempted.is_initial_prompt

        accepted = np.zeros((2, 1), dtype=np.int64)
        num_accepted = np.zeros((2,), dtype=np.int64)
        bonus = np.full((2,), 99, dtype=np.int64)
        next_draft = np.zeros((2, 1), dtype=np.int64)
        bitmask_out = self._empty_bitmask(len(consuming_batch), num_positions=2)
        bitmask_out[:] = 7

        helper.advance_fsm_and_compute_bitmasks(
            context_batch=producing_batch,
            accepted_draft_tokens=accepted,
            num_accepted=num_accepted,
            bonus_tokens=bonus,
            next_draft_tokens=next_draft,
            bitmask_out=bitmask_out,
            output_context_batch=consuming_batch,
        )

        # No raise; the preempted row and the survivor row are both reached.
        assert (bitmask_out == -1).all()

    def test_preempted_in_flight_row_matcher_not_advanced(self) -> None:
        """Regression: a constrained row reset (preempted) after enqueue must
        NOT have its matcher advanced in Part 1.

        ``reset()`` rewinds the token buffer (dropping the in-flight committed
        token) but preserves ``ctx.matcher``. Advancing the matcher through that
        dropped token would leave it one token ahead of the sequence the row
        re-primes from on resume -- a silent desync that yields schema-shaped
        but invalid output. The matcher must be left untouched.
        """
        helper = StructuredOutputHelper(
            enabled=True, vocab_size=16, backend=_NoopBackend()
        )

        matcher = _RecordingMatcher()
        preempted = self._decoding_ctx()
        preempted.set_matcher(matcher)
        preempted.grammar_enforced = True

        producing_batch = [preempted]
        consuming_batch = [preempted]

        # Mirror an in-flight preemption after the callback was enqueued.
        preempted.reset()
        assert preempted.is_initial_prompt
        assert preempted.matcher is matcher  # reset preserves the matcher

        accepted = np.zeros((1, 1), dtype=np.int64)
        num_accepted = np.zeros((1,), dtype=np.int64)
        # A non-EOS committed (bonus) token that, absent the skip, the matcher
        # would be advanced through.
        bonus = np.full((1,), 5, dtype=np.int64)
        next_draft = np.zeros((1, 1), dtype=np.int64)
        bitmask_out = self._empty_bitmask(len(consuming_batch), num_positions=2)
        bitmask_out[:] = 7

        helper.advance_fsm_and_compute_bitmasks(
            context_batch=producing_batch,
            accepted_draft_tokens=accepted,
            num_accepted=num_accepted,
            bonus_tokens=bonus,
            next_draft_tokens=next_draft,
            bitmask_out=bitmask_out,
            output_context_batch=consuming_batch,
        )

        # The preempted row's matcher was never advanced (Part 1 skipped it),
        # and its bitmask row was left unconstrained (Part 2 skipped it).
        assert matcher.consumed == []
        assert (bitmask_out == -1).all()

    def test_continuing_row_matcher_is_advanced(self) -> None:
        """Control: a constrained row NOT preempted still has its matcher
        advanced through the committed token -- the skip is preemption-only."""
        helper = StructuredOutputHelper(
            enabled=True, vocab_size=16, backend=_NoopBackend()
        )

        matcher = _RecordingMatcher()
        ctx = self._decoding_ctx()
        ctx.set_matcher(matcher)
        ctx.grammar_enforced = True
        assert not ctx.is_initial_prompt

        producing_batch = [ctx]
        consuming_batch = [ctx]

        accepted = np.zeros((1, 1), dtype=np.int64)
        num_accepted = np.zeros((1,), dtype=np.int64)
        bonus = np.full((1,), 5, dtype=np.int64)
        next_draft = np.zeros((1, 1), dtype=np.int64)
        bitmask_out = self._empty_bitmask(len(consuming_batch), num_positions=2)

        helper.advance_fsm_and_compute_bitmasks(
            context_batch=producing_batch,
            accepted_draft_tokens=accepted,
            num_accepted=num_accepted,
            bonus_tokens=bonus,
            next_draft_tokens=next_draft,
            bitmask_out=bitmask_out,
            output_context_batch=consuming_batch,
        )

        # Part 1 advanced the original matcher through the bonus token.
        assert matcher.consumed == [[5]]

    def test_all_consumer_rows_present_in_producing_batch_ok(self) -> None:
        """Control: steady decode->decode, every consumer row attributable.

        When no row was admitted from outside the producing batch (the
        aggregated steady-state path), the callback attributes every consumer
        row and does not assert.
        """
        helper = StructuredOutputHelper(enabled=True, vocab_size=16)

        row_a = self._decoding_ctx()
        row_b = self._decoding_ctx()

        producing_batch = [row_a, row_b]
        consuming_batch = [row_a, row_b]

        accepted = np.zeros((2, 1), dtype=np.int64)
        num_accepted = np.zeros((2,), dtype=np.int64)
        bonus = np.full((2,), 99, dtype=np.int64)
        next_draft = np.zeros((2, 1), dtype=np.int64)
        bitmask_out = self._empty_bitmask(len(consuming_batch), num_positions=2)

        helper.advance_fsm_and_compute_bitmasks(
            context_batch=producing_batch,
            accepted_draft_tokens=accepted,
            num_accepted=num_accepted,
            bonus_tokens=bonus,
            next_draft_tokens=next_draft,
            bitmask_out=bitmask_out,
            output_context_batch=consuming_batch,
        )

        # Unconstrained rows: callback resets every row to all-valid (-1).
        assert (bitmask_out == -1).all()

    def test_snapshot_count_mismatch_degrades_not_raises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A short snapshot list must not raise mid-Part-1.

        Defensive only: _build_bitmask_callback captures from the same batch it
        passes, so the async caller cannot produce a mismatch today.

        The caller already set fsm_advanced_by_callback, so this batch's later
        sync will not advance these matchers: any row Part 1 does not reach
        stays permanently behind its request's tokens.
        """
        helper = StructuredOutputHelper(
            enabled=True, vocab_size=16, backend=_NoopBackend()
        )

        matcher_a = _RecordingMatcher()
        matcher_b = _RecordingMatcher()
        row_a = self._decoding_ctx()
        row_b = self._decoding_ctx()
        for ctx, matcher in ((row_a, matcher_a), (row_b, matcher_b)):
            ctx.set_matcher(matcher)
            ctx.grammar_enforced = True

        producing_batch = [row_a, row_b]

        accepted = np.zeros((2, 1), dtype=np.int64)
        num_accepted = np.zeros((2,), dtype=np.int64)
        bonus = np.full((2,), 5, dtype=np.int64)
        next_draft = np.zeros((2, 1), dtype=np.int64)
        bitmask_out = self._empty_bitmask(len(producing_batch), num_positions=2)

        with caplog.at_level(logging.ERROR, logger="max.pipelines"):
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=producing_batch,
                accepted_draft_tokens=accepted,
                num_accepted=num_accepted,
                bonus_tokens=bonus,
                next_draft_tokens=next_draft,
                bitmask_out=bitmask_out,
                committed_span_snapshots=CommittedSpanSnapshot.capture([row_a]),
            )

        assert "committed-span snapshots" in caplog.text
        # Both matchers advanced: the fallback read covered the whole batch.
        assert matcher_a.consumed == [[5]]
        assert matcher_b.consumed == [[5]]


class _RaisingBackend(GrammarBackend[Any]):
    """GrammarBackend stub whose compiles always raise, to exercise the
    validator's exception translation."""

    name = "boom"

    def compile_json_schema(self, json_schema: Any) -> Any:
        raise ValueError("cannot compile schema")

    def create_matcher(self, grammar: Any) -> GrammarMatcher:
        raise ValueError("cannot compile grammar")

    def validate_grammar(self, grammar: Any) -> None:
        return None

    def allocate_token_bitmask(
        self, batch_size: int, vocab_size: int
    ) -> npt.NDArray[np.int32]:
        raise NotImplementedError

    def fill_next_token_bitmask(
        self,
        matcher: GrammarMatcher,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        raise NotImplementedError


class TestDeadMatcherLeavesSlotUnconstrained:
    """``StructuredOutputHelper._fill_slot_unless_matcher_dead``.

    A matcher stopped without accepting has erred and can allow no token, so
    its mask is all-zero. Handing that row to the sampler is worse than
    dropping the constraint: the masked-out fill is finite (so a fully-masked
    row degrades to a uniform draw rather than NaN), which turns the row into a
    uniform draw over the whole vocabulary -- padded tail included. The slot
    must stay at its all-valid ``-1`` reset instead.
    """

    @staticmethod
    def _constrained_ctx(matcher: GrammarMatcher) -> TextContext:
        ctx = create_text_context(prompt_len=4, max_length=128)
        ctx.update(new_token=99)
        ctx.set_matcher(matcher)
        ctx.grammar_enforced = True
        return ctx

    @staticmethod
    def _run(
        helper: StructuredOutputHelper, ctx: TextContext
    ) -> npt.NDArray[np.int32]:
        bitmask_out = np.zeros((1, 2, 1), dtype=np.int32)
        helper.advance_fsm_and_compute_bitmasks(
            context_batch=[ctx],
            accepted_draft_tokens=np.zeros((1, 1), dtype=np.int64),
            num_accepted=np.zeros((1,), dtype=np.int64),
            bonus_tokens=np.full((1,), 5, dtype=np.int64),
            next_draft_tokens=np.zeros((1, 1), dtype=np.int64),
            bitmask_out=bitmask_out,
            output_context_batch=[ctx],
        )
        return bitmask_out

    def test_dead_matcher_leaves_every_slot_all_valid(self) -> None:
        backend = _FillCountingBackend()
        helper = StructuredOutputHelper(
            enabled=True, vocab_size=16, backend=backend
        )
        ctx = self._constrained_ctx(_DeadMatcher())

        bitmask_out = self._run(helper, ctx)

        assert backend.fills == 0
        assert (bitmask_out == -1).all()

    def test_completed_matcher_is_still_applied(self) -> None:
        """Control: a stopped *accepting* matcher is a satisfied grammar.

        Its EOS-only mask is correct and must still constrain the row -- the
        guard keys on the error state, not on being stopped. A backend that
        can't safely compute that mask for a stopped matcher (xgrammar)
        guards this itself; see
        ``test_fill_slot_after_stop_token_accepted_does_not_crash`` in
        ``test_structured_output_backend.py``.
        """
        backend = _FillCountingBackend()
        helper = StructuredOutputHelper(
            enabled=True, vocab_size=16, backend=backend
        )
        ctx = self._constrained_ctx(_CompletedMatcher())

        bitmask_out = self._run(helper, ctx)

        assert backend.fills > 0
        assert (bitmask_out == 0).any()

    def test_report_is_latched_per_request(self) -> None:
        """The error state persists, so the log must not repeat every step."""
        backend = _FillCountingBackend()
        helper = StructuredOutputHelper(
            enabled=True, vocab_size=16, backend=backend
        )
        ctx = self._constrained_ctx(_DeadMatcher())

        assert not ctx.grammar_state.dead_matcher_reported
        self._run(helper, ctx)
        assert ctx.grammar_state.dead_matcher_reported

        # The latch survives the speculative walk's snapshot/restore, so a
        # second step reports nothing further while still skipping the fill.
        self._run(helper, ctx)
        assert backend.fills == 0


class TestGrammarCompileFailure:
    """The worker owns the only compile, so it is what turns a compile
    failure into the InputError the API server returns as a 400."""

    def _helper(self, backend: GrammarBackend[Any]) -> StructuredOutputHelper:
        return StructuredOutputHelper(
            enabled=True,
            enable_response_format_schema=True,
            vocab_size=128,
            backend=backend,
        )

    def test_uncompilable_schema_raises_input_error(self) -> None:
        ctx = create_text_context(prompt_len=4, max_length=100)
        ctx.json_schema = '{"type": "object"}'
        bitmask = np.zeros((1, 4), dtype=np.int32)
        with pytest.raises(InputError, match="boom"):
            self._helper(_RaisingBackend()).update_context(ctx, bitmask, 0)

    def test_uncompilable_tool_grammar_raises_input_error(self) -> None:
        ctx = create_text_context(prompt_len=4, max_length=100)
        ctx.grammar = "<grammar>"
        bitmask = np.zeros((1, 4), dtype=np.int32)
        with pytest.raises(InputError, match="boom"):
            self._helper(_RaisingBackend()).update_context(ctx, bitmask, 0)

    def test_unsatisfiable_schema_is_rejected(self) -> None:
        """llguidance's matcher fails open on unsatisfiable schemas, so
        build_matcher's validate_grammar call is what rejects them."""

        class _UnsatisfiableBackend(_NoopBackend):
            def validate_grammar(self, grammar: Any) -> None:
                raise ValueError("Unsatisfiable schema")

        ctx = create_text_context(prompt_len=4, max_length=100)
        ctx.json_schema = '{"anyOf": [false]}'
        bitmask = np.zeros((1, 4), dtype=np.int32)
        with pytest.raises(InputError, match="Unsatisfiable"):
            self._helper(_UnsatisfiableBackend()).update_context(
                ctx, bitmask, 0
            )


class TestSpecialTokenIdsForMarkers:
    """special_token_ids_for_markers resolves single-token markers to ids and
    skips markers with no single-token vocabulary form."""

    def test_resolves_single_token_markers_and_skips_unknown(self) -> None:
        class _FakeTokenizer:
            unk_token_id = 0

            def convert_tokens_to_ids(self, token: str) -> int:
                return {"<arg_value>": 7, "</arg_value>": 8}.get(
                    token, self.unk_token_id
                )

        ids = _sob.special_token_ids_for_markers(
            ("<arg_value>", "</arg_value>", "<no_single_token_form>"),
            _FakeTokenizer(),
        )
        assert ids == {7, 8}


class TestXgrammarCacheBound:
    """xgrammar's compiled-grammar cache is bounded (vLLM's VLLM_XGRAMMAR_CACHE_MB
    analog) so a long-running server can't grow it without limit."""

    def test_default_limit_is_512_mb(self) -> None:
        assert _sob._xgrammar_cache_limit_bytes() == 512 * 1024 * 1024

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODULAR_XGRAMMAR_CACHE_MB", "64")
        assert _sob._xgrammar_cache_limit_bytes() == 64 * 1024 * 1024

    def test_compiler_honors_byte_limit(self) -> None:
        vocab = [f"tok{i}".encode() for i in range(16)]
        tokenizer_info = xgrammar.TokenizerInfo(
            vocab, vocab_type=xgrammar.VocabType.RAW, stop_token_ids=[0]
        )
        limit = 8 * 1024 * 1024
        compiler = xgrammar.GrammarCompiler(
            tokenizer_info, max_memory_bytes=limit
        )
        assert compiler._impl.cache_limit_bytes == limit


class _FillRecordingBackend(_NoopBackend):
    """Backend that stamps a ``0`` sentinel into every slot it fills.

    ``advance_fsm_and_compute_bitmasks`` resets each output row to ``-1``
    (all-valid / unconstrained) before filling. A slot the fill path touches
    is stamped ``0`` here, so after the call a slot still holding ``-1`` was
    left unconstrained and a slot holding ``0`` was constrained.
    """

    name = "fill-recording"

    def fill_next_token_bitmask(
        self,
        matcher: GrammarMatcher,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        bitmask[index, :] = 0


class TestCommittedInteriorEosTerminates:
    """``StructuredOutputHelper.advance_fsm_and_compute_bitmasks`` Part 1: the
    async callback permanently advances the real ``ctx`` through the committed
    span ``accepted_drafts[:num_accepted] + [bonus]``, and it must stop at the
    FIRST EOS-class token so spec decode does not diverge from non-spec.

    Generation stops at the first EOS-class committed token: non-spec decode
    terminates on ``generated[-1]`` one token at a time, and
    ``update_spec_decode_context_and_prepare_responses`` truncates the committed
    span at that same token. Tokens after it are never emitted, so the matcher
    must not advance through them -- doing so would desync the matcher on
    phantom tokens and disable enforcement for a continuation that does not
    exist.
    """

    @staticmethod
    def _constrained_ctx() -> TextContext:
        ctx = create_text_context(prompt_len=4, max_length=128)
        ctx.update(new_token=99)
        ctx.set_matcher(_RecordingMatcher())
        # ``tool_choice=required`` enforces from the first token throughout.
        ctx.grammar_enforced = True
        # ``<eos>`` (1) and ``<end_of_turn>`` (7) are both stop ids.
        ctx.eos_tracker.eos_token_ids = {1, 7}
        return ctx

    def test_interior_eos_terminates_and_stops_matcher_walk(self) -> None:
        """An interior committed EOS ends generation (the span is truncated
        there), so enforcement flips off and the matcher must not advance
        through the tokens after it. Spec decode matches non-spec: the first
        EOS-class token is terminal."""
        helper = StructuredOutputHelper(
            enabled=True, vocab_size=16, backend=_FillRecordingBackend()
        )

        ctx = self._constrained_ctx()
        matcher = ctx.matcher
        assert isinstance(matcher, _RecordingMatcher)

        # Committed span: accepted draft [1] (=<eos>, interior) + bonus 8.
        # Generation stops at the interior <eos>; bonus 8 is truncated and
        # never reaches the matcher. num_accepted=1.
        accepted = np.array([[1]], dtype=np.int64)
        num_accepted = np.array([1], dtype=np.int64)
        bonus = np.array([8], dtype=np.int64)
        next_draft = np.array([[5, 6, 8]], dtype=np.int64)
        bitmask_out = np.zeros((1, 4, 1), dtype=np.int32)

        helper.advance_fsm_and_compute_bitmasks(
            context_batch=[ctx],
            accepted_draft_tokens=accepted,
            num_accepted=num_accepted,
            bonus_tokens=bonus,
            next_draft_tokens=next_draft,
            bitmask_out=bitmask_out,
            output_context_batch=[ctx],
        )

        # Generation ends at the first EOS: enforcement flips off.
        assert not ctx.grammar_enforced, (
            "interior EOS is terminal; enforcement should be disabled"
        )
        # The matcher walk stops at the EOS: the truncated post-EOS token is
        # never fed to the matcher.
        assert matcher.consumed == [], matcher.consumed

    def test_terminal_eos_still_disables_enforcement(self) -> None:
        """Control: a terminal committed EOS (generation actually ending) still
        disables enforcement, and the content token before it is advanced into
        the matcher first."""
        helper = StructuredOutputHelper(
            enabled=True, vocab_size=16, backend=_FillRecordingBackend()
        )

        ctx = self._constrained_ctx()

        # Committed span: accepted draft [8] (content) + bonus 1 (=<eos>,
        # terminal). is_eos_from_tokens sees generated[-1]=<eos> -> done.
        accepted = np.array([[8]], dtype=np.int64)
        num_accepted = np.array([1], dtype=np.int64)
        bonus = np.array([1], dtype=np.int64)
        next_draft = np.array([[5, 6, 8]], dtype=np.int64)
        bitmask_out = np.zeros((1, 4, 1), dtype=np.int32)

        helper.advance_fsm_and_compute_bitmasks(
            context_batch=[ctx],
            accepted_draft_tokens=accepted,
            num_accepted=num_accepted,
            bonus_tokens=bonus,
            next_draft_tokens=next_draft,
            bitmask_out=bitmask_out,
            output_context_batch=[ctx],
        )

        assert not ctx.grammar_enforced, (
            "terminal EOS should disable enforcement (generation ended)"
        )

    def test_multi_token_eos_sequence_terminates_the_walk(self) -> None:
        """A committed span completing a multi-token stop ``eos_sequence``
        terminates the request at the sequence's final token, matching the
        truncation path (which checks ``is_eos_from_tokens``, not just single
        ids). Enforcement flips off and the matcher does not advance past the
        completed sequence -- the case a bare ``token in eos_token_ids`` check
        would miss."""
        helper = StructuredOutputHelper(
            enabled=True, vocab_size=16, backend=_FillRecordingBackend()
        )

        ctx = self._constrained_ctx()
        matcher = ctx.matcher
        assert isinstance(matcher, _RecordingMatcher)
        # A two-token stop sequence [5, 6]. None of 5/6/8 is a single-id EOS
        # ({1, 7}), so only the completed sequence can end the request here.
        ctx.eos_tracker.eos_sequences = [[5, 6]]

        # Committed span: accepted [5, 6] completes the stop sequence at token
        # 6; bonus 8 is truncated and must not reach the matcher.
        accepted = np.array([[5, 6]], dtype=np.int64)
        num_accepted = np.array([2], dtype=np.int64)
        bonus = np.array([8], dtype=np.int64)
        next_draft = np.array([[5, 6, 8]], dtype=np.int64)
        bitmask_out = np.zeros((1, 4, 1), dtype=np.int32)

        helper.advance_fsm_and_compute_bitmasks(
            context_batch=[ctx],
            accepted_draft_tokens=accepted,
            num_accepted=num_accepted,
            bonus_tokens=bonus,
            next_draft_tokens=next_draft,
            bitmask_out=bitmask_out,
            output_context_batch=[ctx],
        )

        # Only the completed stop sequence can flip enforcement off here, so
        # this proves the sequence check fired (a single-id check would not).
        assert not ctx.grammar_enforced, (
            "multi-token stop sequence did not terminate the matcher walk"
        )
        # The truncated post-sequence token never reaches the matcher.
        assert [8] not in matcher.consumed, matcher.consumed


class TestSpecDecodeEosOutranksLengthCap:
    """A commit that both ends the sequence and fills the cap reports EOS.

    ``TextContext.update`` gives EOS precedence over the length cap with an
    ``if``/``elif``. The speculative commit loop applied the cap
    unconditionally afterwards, so a request whose EOS landed exactly on its
    last allowed position was reported ``MAXIMUM_LENGTH``. That misreports a
    completed request as truncated, and only ever on the speculative path --
    which is precisely the stop-vs-length signal used to compare speculative
    decoding against plain decode.
    """

    @staticmethod
    def _run_one_step(bonus_token: int) -> TextContext:
        prompt_len = 10
        max_gen_tokens = 4  # one full accept chunk: 3 drafts + bonus
        max_length = prompt_len + max_gen_tokens

        ctx = create_text_context(prompt_len=prompt_len, max_length=max_length)
        ctx.eos_tracker.eos_token_ids = {1}
        ctx.update_with_future_token()

        update_spec_decode_context_and_prepare_responses(
            draft_tokens=np.array([[101, 102, 103]], dtype=np.int32),
            next_draft_tokens=np.array([[201, 202, 203]], dtype=np.int32),
            num_accepted_draft_tokens=np.array([3], dtype=np.int32),
            next_tokens=np.array([bonus_token], dtype=np.int32),
            context_batch=[ctx],
            max_seq_len=10_000,
        )
        assert ctx.tokens.current_position == max_length, (
            "test must land the final commit exactly on the cap"
        )
        return ctx

    def test_eos_on_the_final_allowed_position_reports_eos(self) -> None:
        ctx = self._run_one_step(bonus_token=1)
        assert ctx.status == GenerationStatus.END_OF_SEQUENCE, (
            "a sequence that ended on EOS must not be reported as truncated "
            "just because the EOS filled the last allowed position"
        )

    def test_non_eos_on_the_final_allowed_position_reports_max_length(
        self,
    ) -> None:
        ctx = self._run_one_step(bonus_token=999)
        assert ctx.status == GenerationStatus.MAXIMUM_LENGTH


class TestSpeculativeBitmaskWindowWidth:
    """``StructuredOutputHelper._speculatively_fill_bitmask_window`` must accept
    a window wider than the drafts justify.

    ``num_positions`` is the graph's static bitmask width
    (``num_speculative_tokens + 1``) and cannot shrink per step, but the
    realized draft width can: at the prefill->decode boundary the batch
    verifies zero drafts, so ``compute_speculative_bitmasks`` hands this helper
    a full-width window and an empty draft row. The window arrives
    pre-initialized to ``-1`` (unconstrained), so slots the drafts never reach
    are already correct. Only a window too *narrow* for the drafts is a caller
    bug.
    """

    @staticmethod
    def _run(
        drafts: list[int], num_positions: int
    ) -> tuple[TextContext, npt.NDArray[np.int32]]:
        helper = StructuredOutputHelper(
            enabled=True, vocab_size=32, backend=_FillRecordingBackend()
        )
        ctx = create_text_context(prompt_len=4, max_length=128)
        ctx.update(new_token=99)
        ctx.set_matcher(_RecordingMatcher())
        ctx.grammar_enforced = True

        window = np.full((num_positions, 1), -1, dtype=np.int32)
        helper._speculatively_fill_bitmask_window(
            ctx,
            drafts=np.array(drafts, dtype=np.int64),
            bitmask_window=window,
        )
        return ctx, window

    @staticmethod
    def _constrained(window: npt.NDArray[np.int32]) -> list[bool]:
        """Per-slot: True where the backend wrote (0), False where still -1."""
        return [bool((row == 0).all()) for row in window]

    def test_zero_drafts_constrain_only_the_first_slot(self) -> None:
        """Prefill->decode boundary: zero realized drafts, static 4-slot
        window. Slot 0 still carries the current matcher state; the rest keep
        their unconstrained reset."""
        _, window = self._run(drafts=[], num_positions=4)

        assert self._constrained(window) == [True, False, False, False]

    def test_partial_drafts_leave_the_unreached_tail_alone(self) -> None:
        """Two realized drafts in a static 4-slot window constrain slots 0..2;
        slot 3 has no draft to advance into."""
        ctx, window = self._run(drafts=[3, 4], num_positions=4)

        assert self._constrained(window) == [True, True, True, False]
        # The walk runs on a deep copy and unwinds, so the request's own
        # matcher and enforcement state are untouched.
        assert isinstance(ctx.matcher, _RecordingMatcher)
        assert ctx.matcher.consumed == []
        assert ctx.grammar_enforced

    def test_exact_width_constrains_every_slot(self) -> None:
        """Regression guard: the steady-state case where the drafts exactly
        fill the window is unchanged."""
        _, window = self._run(drafts=[3, 4, 5], num_positions=4)

        assert self._constrained(window) == [True, True, True, True]

    def test_more_drafts_than_the_window_holds_raises(self) -> None:
        """A window too narrow for the drafts is a real caller bug -- silently
        truncating would drop constraints the sampler needs."""
        with pytest.raises(ValueError, match="bitmask window"):
            self._run(drafts=[3, 4, 5, 6], num_positions=4)


def _xgrammar_backend() -> XgrammarBackend:
    """A real xgrammar backend over a tiny RAW vocab.

    Built directly rather than through from_tokenizer_delegate: the FSM-position
    assertions below need a vocab small enough to compare whole bitmasks.
    """
    vocab = [chr(c) for c in range(32, 127)] + ["<eos>"]
    tokenizer_info = xgrammar.TokenizerInfo(
        vocab,
        vocab_type=xgrammar.VocabType.RAW,
        stop_token_ids=[len(vocab) - 1],
    )
    return XgrammarBackend(xgrammar.GrammarCompiler(tokenizer_info))


# Token ids in _xgrammar_backend's RAW vocab (chr(32 + id)).
_OBJ_OPEN = ord("{") - 32
_QUOTE = ord('"') - 32
_A = ord("a") - 32
_B = ord("b") - 32
_VOCAB_SIZE = 96

# The only string this schema admits is {"ab":<int>}, so every prefix has a
# distinct allowed-token set -- which is what makes "the matcher advanced
# exactly this far" an assertable value rather than a guess.
_SINGLE_KEY_SCHEMA = {
    "type": "object",
    "properties": {"ab": {"type": "integer"}},
    "required": ["ab"],
    "additionalProperties": False,
}


class TestCommittedSpanSnapshot:
    """CommittedSpanSnapshot + Part 1's use of it.

    advance_fsm_and_compute_bitmasks runs on an AsyncRT worker while the
    main thread realizes the same batch's committed tokens into ctx.tokens
    and appends the next FUTURE_TOKEN placeholder. Reading the token buffer
    on the worker cannot distinguish "history before the committed span" from
    "history including it": both end in a placeholder. The snapshot is taken on
    the main thread at enqueue time, so the FSM advance must be unaffected by
    any of those mutations.
    """

    @staticmethod
    def _matcher(backend: XgrammarBackend, prefix: list[int]) -> GrammarMatcher:
        matcher = backend.create_matcher(
            backend.compile_json_schema(_SINGLE_KEY_SCHEMA)
        )
        assert matcher.try_consume_tokens(prefix) == len(prefix)
        return matcher

    @classmethod
    def _mask(
        cls, backend: XgrammarBackend, matcher: GrammarMatcher | None
    ) -> npt.NDArray[np.int32]:
        assert matcher is not None
        mask = backend.allocate_token_bitmask(1, _VOCAB_SIZE)
        backend.fill_next_token_bitmask(matcher, mask, 0)
        return mask

    @classmethod
    def _enqueue_state_ctx(
        cls, backend: XgrammarBackend, eos_sequence: list[int]
    ) -> TextContext:
        """A constrained row in the state it holds when the callback is enqueued.

        One realized generated token (the opening brace, already consumed by
        the matcher) followed by the unrealized placeholder the previous step
        appended.
        """
        ctx = create_text_context(prompt_len=4, max_length=128)
        # Built, not assigned into: EOSTracker derives its stop-sequence
        # lookback in model_post_init, and first_eos_offset needs that lookback
        # to see the pre-span history at all.
        ctx.eos_tracker = EOSTracker(eos_sequences=[eos_sequence])
        ctx.set_matcher(cls._matcher(backend, [_OBJ_OPEN]))
        ctx.grammar_enforced = True
        ctx.update_with_future_token()
        ctx.realize_future_token(_OBJ_OPEN)
        ctx.update_with_future_token()
        assert ctx.tokens.generated.tolist() == [_OBJ_OPEN, FUTURE_TOKEN]
        return ctx

    @staticmethod
    def _advance(
        helper: StructuredOutputHelper,
        ctx: TextContext,
        committed: list[int],
        snapshots: Sequence[CommittedSpanSnapshot],
    ) -> npt.NDArray[np.int32]:
        """Run the callback body over a one-row batch; returns the bitmask."""
        *accepted, bonus = committed
        bitmask_out = np.zeros((1, 2, 3), dtype=np.int32)
        helper.advance_fsm_and_compute_bitmasks(
            context_batch=[ctx],
            accepted_draft_tokens=np.array([accepted], dtype=np.int64),
            num_accepted=np.array([len(accepted)], dtype=np.int64),
            bonus_tokens=np.array([bonus], dtype=np.int64),
            next_draft_tokens=np.zeros((1, 1), dtype=np.int64),
            bitmask_out=bitmask_out,
            output_context_batch=[ctx],
            committed_span_snapshots=snapshots,
        )
        return bitmask_out

    def test_capture_drops_placeholders_and_copies(self) -> None:
        """The snapshot is the realized prefix, and later realization of the
        placeholder must not reach back into it -- realize_future_token
        overwrites the context's token array in place."""
        backend = _xgrammar_backend()
        ctx = self._enqueue_state_ctx(backend, [_OBJ_OPEN, _QUOTE, _A])

        (snapshot,) = CommittedSpanSnapshot.capture([ctx])
        assert snapshot.prior_generated.tolist() == [_OBJ_OPEN]

        ctx.realize_future_token(_QUOTE)
        assert snapshot.prior_generated.tolist() == [_OBJ_OPEN]

    @pytest.mark.parametrize(
        ("eos_sequences", "keep"),
        [
            ([], 0),
            ([[_A]], 0),
            ([[_A, _B]], 1),
            ([[_A, _B], [_A, _B, _QUOTE]], 2),
        ],
    )
    def test_capture_is_bounded_by_eos_lookback(
        self, eos_sequences: list[list[int]], keep: int
    ) -> None:
        """A long context must not cost a whole-history copy per row.

        first_eos_offset reads only the last eos_sequence_lookback
        pre-span tokens, so that is all the snapshot needs to carry -- nothing
        at all for a request whose stop sequences are single tokens.
        """
        ctx = create_text_context(prompt_len=4, max_length=256)
        ctx.eos_tracker = EOSTracker(eos_sequences=eos_sequences)
        realized = list(range(50))
        for token in realized:
            ctx.update_with_future_token()
            ctx.realize_future_token(token)
        ctx.update_with_future_token()
        assert ctx.tokens.generated_length == len(realized) + 1

        (snapshot,) = CommittedSpanSnapshot.capture([ctx])

        # The tail immediately preceding the span: realized tokens only, with
        # the trailing placeholder excluded.
        expected = realized[len(realized) - keep :] if keep else []
        assert snapshot.prior_generated.tolist() == expected

    @pytest.mark.parametrize("mutate", ["none", "realize", "realize_and_fill"])
    def test_fsm_advance_is_independent_of_main_thread_mutation(
        self, mutate: str
    ) -> None:
        """Regression: the same execute() that enqueues the callback goes on to
        realize the committed span and append the next placeholder. Whatever it
        has done by the time the worker runs, the FSM advance must land where
        the enqueue-time history says it should."""
        backend = _xgrammar_backend()
        helper = StructuredOutputHelper(
            enabled=True, vocab_size=_VOCAB_SIZE, backend=backend
        )
        # Stop sequence {"a straddles the span boundary: it starts at the token
        # already in the buffer and completes at committed[1].
        ctx = self._enqueue_state_ctx(backend, [_OBJ_OPEN, _QUOTE, _A])
        committed = [_QUOTE, _A, _B]

        snapshots = CommittedSpanSnapshot.capture([ctx])

        if mutate != "none":
            ctx.realize_future_token(committed[0])
        if mutate == "realize_and_fill":
            for token in committed[1:]:
                ctx.advance_token_buffer(token)
            ctx.update_with_future_token()

        bitmask_out = self._advance(helper, ctx, committed, snapshots)

        # Generation ends inside the span, so enforcement is off and the
        # matcher stopped one token in: it consumed committed[0] and nothing
        # after it.
        assert not ctx.grammar_enforced
        expected = self._mask(
            backend, self._matcher(backend, [_OBJ_OPEN, _QUOTE])
        )
        over_advanced = self._mask(
            backend, self._matcher(backend, [_OBJ_OPEN, _QUOTE, _A])
        )
        assert (self._mask(backend, ctx.matcher) == expected).all()
        assert not (expected == over_advanced).all()
        # Enforcement is off, so Part 2 leaves the row unconstrained.
        assert (bitmask_out == -1).all()

    @pytest.mark.parametrize("eos_offset", [0, 1])
    def test_eos_offset_index_comes_from_snapshot_history(
        self, eos_offset: int
    ) -> None:
        """The index enforcement is cleared at is decided by the snapshot, not
        by whether a placeholder happens to be sitting at the end of the buffer.

        Both stop sequences below start at the token already in the buffer, so
        only the pre-span history distinguishes "ends at committed[0]" from
        "ends at committed[1]".
        """
        backend = _xgrammar_backend()
        helper = StructuredOutputHelper(
            enabled=True, vocab_size=_VOCAB_SIZE, backend=backend
        )
        committed = [_QUOTE, _A, _B]
        eos_sequence = [_OBJ_OPEN, *committed[: eos_offset + 1]]
        ctx = self._enqueue_state_ctx(backend, eos_sequence)

        snapshots = CommittedSpanSnapshot.capture([ctx])

        # Main thread, mid-flight: span realized, next placeholder appended.
        ctx.realize_future_token(committed[0])
        for token in committed[1:]:
            ctx.advance_token_buffer(token)
        ctx.update_with_future_token()

        self._advance(helper, ctx, committed, snapshots)

        assert not ctx.grammar_enforced
        expected = self._mask(
            backend,
            self._matcher(backend, [_OBJ_OPEN, *committed[:eos_offset]]),
        )
        assert (self._mask(backend, ctx.matcher) == expected).all()
