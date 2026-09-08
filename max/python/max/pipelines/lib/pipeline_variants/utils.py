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

"""Provides utility functions for computing allowed generation steps in pipeline variants."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.pipelines.context import (
    FUTURE_TOKEN,
    GenerationStatus,
    GrammarMatcher,
    LogProbabilities,
    StructuredOutputRegionDelimiters,
    TextGenerationContextType,
    TextGenerationOutput,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    GrammarBackend,
    LlguidanceBackend,
    make_grammar_backend,
)
from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    get_parser_cls,
)
from max.pipelines.lib.utils import upper_bounded_default
from max.pipelines.modeling.types import RequestID
from max.pipelines.sampling import DEFAULT_STRUCTURED_OUTPUT_BACKEND
from max.profiler import Tracer, traced
from max.support.math import ceildiv
from transformers import AutoConfig

if TYPE_CHECKING:
    from max.pipelines.modeling.types import PipelineTokenizer

logger = logging.getLogger("max.pipelines")


def _count_token_subsequence(
    content: Sequence[int], special_tags: Sequence[int]
) -> int:
    """Counts non-overlapping occurrences of ``special_tags`` in ``content``.

    Used only on the matcher-rejection diagnostic path to count how many
    tool-call section markers were already committed. ``special_tags`` is the
    section-begin (or -end) token-id sequence — a single token for most
    parsers, so this is effectively a count. Runs O(len(content)); acceptable
    because it fires only when a rejection has already occurred.
    """
    width = len(special_tags)
    if width == 0:
        return 0
    tag_ids = list(special_tags)
    count = 0
    i = 0
    last_start = len(content) - width
    while i <= last_start:
        if list(content[i : i + width]) == tag_ids:
            count += 1
            i += width
        else:
            i += 1
    return count


def calculate_num_steps(
    context: TextGenerationContextType,
    num_steps: int,
    max_seq_len: int,
) -> int:
    """Compute the number of generation steps allowed for a context.

    The value is clamped by the remaining capacity with respect to
    the model's configured ``max_seq_len``.

    Args:
        context: The context whose sequence length constraints apply.
        num_steps: Desired number of steps to attempt.
        max_seq_len: The maximum allowed sequence length for the model.

    Returns:
        The number of steps to execute for this context (>= 1).

    Raises:
        ValueError: If the current request length is already >= ``max_seq_len``.
    """
    num_available_steps = context.compute_num_available_steps(max_seq_len)

    if num_available_steps <= 0:
        raise ValueError(
            f"Request {context.request_id} length ({len(context.tokens)}) is larger than or equal to the configured max_length ({max_seq_len})"
        )

    return min(num_available_steps, num_steps)


def build_response(
    context_batch: list[TextGenerationContextType],
    max_seq_len: int,
) -> dict[RequestID, TextGenerationOutput]:
    """Build response from updated contexts.

    Marks a context ``MAXIMUM_LENGTH`` only when it has no room for even one
    more token. Callers that can append more than one token per step (e.g.
    speculative decoding) are responsible for not overshooting the cap when
    they commit tokens; see
    :func:`update_spec_decode_context_and_prepare_responses`.

    Args:
        context_batch: The list of context objects.
        max_seq_len: The maximum sequence length.

    Returns:
        Dictionary mapping request IDs to TextGenerationOutput objects.
    """
    res: dict[RequestID, TextGenerationOutput] = {}

    for context in context_batch:
        context_max_length = upper_bounded_default(
            upper_bound=max_seq_len, default=context.max_length
        )

        # Mark done only when there is no room for even one more token. The
        # per-step commit loop is responsible for not overshooting this cap.
        # An already-terminated request keeps its status: EOS outranks the
        # cap, matching ``TextContext.update``'s precedence.
        current_length = context.tokens.processed_length + 1
        if current_length >= context_max_length and not context.status.is_done:
            context.status = GenerationStatus.MAXIMUM_LENGTH

        output = context.to_generation_output()
        if output.tokens:
            res[context.request_id] = output

    return res


@traced
def update_context_and_prepare_responses(
    generated_tokens_host: npt.NDArray[np.int32],
    flat_batch: list[TextGenerationContextType],
    batch_log_probabilities: list[list[LogProbabilities | None]] | None = None,
    enable_log_probs: bool = False,
    overwrite_future: bool = False,
) -> dict[RequestID, TextGenerationOutput]:
    """Updates context objects and prepares response objects after generation.

    Args:
        generated_tokens_host: Array of generated tokens on the host, indexed
            as [batch, 1] (single step).
        flat_batch: List of generation contexts, one per request, matching
            batch dimension.
        batch_log_probabilities: List of per-step log probability outputs (or
            None), each entry is a list per batch for that step.
        enable_log_probs: Whether to include log probability data in outputs.
        overwrite_future: Whether to overwrite future tokens in the context.

    Returns:
        A dictionary mapping request IDs to their respective generation outputs.
    """
    res: dict[RequestID, TextGenerationOutput] = {}
    for batch_index, context in enumerate(flat_batch):
        # Convert to a Python scalar to improve serialization performance.
        next_token = int(generated_tokens_host[batch_index, 0])

        # Get Log probs if needed.
        log_probs: LogProbabilities | None = None
        if enable_log_probs:
            assert batch_log_probabilities is not None
            if batch_log_probabilities:
                log_probs_for_step = batch_log_probabilities[0]
                if log_probs_for_step and batch_index < len(log_probs_for_step):
                    log_probs = log_probs_for_step[batch_index]

        if overwrite_future:
            # If generated_length is still 0, then there is no placeholder
            # future token. This is possible due to chunked prefill or preemption.
            if context.tokens.generated_length:
                context.realize_future_token(
                    new_token=next_token, log_probabilities=log_probs
                )
        else:
            context.advance_token_buffer(
                new_token=next_token, log_probabilities=log_probs
            )
            context.advance_fsm(next_token)

        # Only add the output if there are tokens to return.
        # It is possible that there are no generated tokens due to chunked prefill.
        output = context.to_generation_output()
        if output.tokens:
            res[context.request_id] = output

    return res


@traced
def update_spec_decode_context_and_prepare_responses(
    draft_tokens: npt.NDArray[np.int32],
    next_draft_tokens: npt.NDArray[np.int32],
    num_accepted_draft_tokens: npt.NDArray[np.int32],
    next_tokens: npt.NDArray[np.int32],
    context_batch: list[TextGenerationContextType],
    max_seq_len: int,
    think_start_token_id: int | None = None,
    think_end_token_id: int | None = None,
    skip_fsm_advance: bool = False,
) -> dict[RequestID, TextGenerationOutput]:
    """Updates context objects and prepares response objects after speculative decoding.

    When both boundary ids are provided, also toggles
    ``ctx.in_reasoning_phase`` from the just-committed tokens, in commit
    order so a ``<think>...</think>`` pair within one accept set ends
    correctly.

    Args:
        draft_tokens: Draft tokens verified this batch.
        next_draft_tokens: Next batch's draft tokens.
        num_accepted_draft_tokens: Count of accepted draft tokens per request.
        next_tokens: Bonus tokens per request.
        context_batch: List of generation contexts.
        max_seq_len: Maximum sequence length.
        think_start_token_id: Token ID that starts a reasoning phase.
        think_end_token_id: Token ID that ends a reasoning phase.
        skip_fsm_advance: When True, skip FSM advancement because a CUDA host
            callback already advanced the FSM. Token buffer is still updated.
    """
    num_draft_tokens_to_verify = draft_tokens.shape[1]
    num_speculative_tokens = next_draft_tokens.shape[1]

    assert num_accepted_draft_tokens.shape == (len(context_batch),)
    assert next_tokens.shape == (len(context_batch),)
    assert next_draft_tokens.shape == (
        len(context_batch),
        num_speculative_tokens,
    )
    assert all(
        num_accept <= num_draft_tokens_to_verify
        for num_accept in num_accepted_draft_tokens
    )

    track_phase = (
        think_start_token_id is not None and think_end_token_id is not None
    )

    # Handle chunked prefill case where there are no future tokens.
    for batch_idx, ctx in enumerate(context_batch):
        if not ctx.tokens.generated_length:
            continue

        # A spec-decode step can append up to num_speculative_tokens + 1
        # tokens at once, which may cross max_seq_len. The KV pool carries
        # num_draft_tokens slack beyond max_seq_len so the forward pass is
        # safe, but committing past the limit would emit out-of-context
        # tokens and push the next step past the pool. Cap the commit at the
        # limit and mark the request done.
        context_max_length = upper_bounded_default(
            upper_bound=max_seq_len, default=ctx.max_length
        )

        maybe_accepted_draft_tokens: list[int] = draft_tokens[
            batch_idx
        ].tolist()
        num_accept = num_accepted_draft_tokens[batch_idx]
        tokens = maybe_accepted_draft_tokens[:num_accept]
        tokens += [next_tokens[batch_idx]]
        num_committed = 0
        for i, token in enumerate(tokens):
            if i == 0:
                ctx.realize_future_token(token)
                if ctx.matcher is not None and not skip_fsm_advance:
                    ctx.advance_fsm(token)
            elif ctx.is_done:
                break
            else:
                if skip_fsm_advance and ctx.matcher is not None:
                    ctx.advance_token_buffer(token)
                else:
                    ctx.update(token)

            num_committed = i + 1
            if ctx.tokens.current_position >= context_max_length:
                # Same EOS-outranks-cap precedence as build_response above.
                if not ctx.status.is_done:
                    ctx.status = GenerationStatus.MAXIMUM_LENGTH
                break

        if track_phase:
            # Only the committed prefix reached the buffer; tokens dropped by
            # the length cap must not toggle the reasoning phase.
            for token in tokens[:num_committed]:
                if token == think_start_token_id:
                    ctx.in_reasoning_phase = True
                elif token == think_end_token_id:
                    ctx.in_reasoning_phase = False

        ctx.spec_decoding_state.maybe_accepted_draft_tokens = []
        if not ctx.is_done:
            ctx.spec_decoding_state.draft_tokens_to_verify = next_draft_tokens[
                batch_idx
            ].tolist()

    result = build_response(
        context_batch=context_batch,
        max_seq_len=max_seq_len,
    )

    # Clear draft tokens for contexts that won't be processed further.
    for ctx in context_batch:
        if ctx.is_done:
            ctx.spec_decoding_state.draft_tokens_to_verify = []

    return result


def get_rope_theta(config: AutoConfig) -> float:
    """Gets rope_theta from a HuggingFace config, compatible with transformers v4 and v5.

    Transformers v5 moved rope_theta into config.rope_parameters["rope_theta"].
    This function checks rope_parameters first, then falls back to config.rope_theta.
    """
    rope_params = getattr(config, "rope_parameters", None)
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        return rope_params["rope_theta"]

    return config.rope_theta


@dataclass(frozen=True)
class CommittedSpanSnapshot:
    """Token-buffer state of one producing-batch row, captured at enqueue time.

    StructuredOutputHelper.advance_fsm_and_compute_bitmasks runs on an
    AsyncRT worker while the main thread realizes the same batch's tokens into
    ctx.tokens, so a worker-side read cannot tell "history before the
    committed span" from "history including it" -- both end in a
    FUTURE_TOKEN placeholder. Capturing on the main thread, before the
    callback is enqueued, is what makes the answer unambiguous.
    """

    prior_generated: npt.NDArray[np.int64]
    """Generated tokens preceding the committed span, placeholder-free.

    Only the last EOSTracker.eos_sequence_lookback of them, since
    that is all EOSTracker.first_eos_offset reads -- a bounded tail per
    row, whatever the context length. Empty when the request configures
    no multi-token stop sequence, which is the common case.

    A copy, not a view: realize_future_token overwrites placeholder slots
    of the context's token array in place.
    """

    @classmethod
    def capture(
        cls, context_batch: Sequence[TextGenerationContextType]
    ) -> list[CommittedSpanSnapshot]:
        """Captures one snapshot per row, in context_batch order."""
        snapshots = []
        for ctx in context_batch:
            keep = ctx.eos_tracker.eos_sequence_lookback
            gen = ctx.tokens.generated
            end = len(gen)
            if end > 0 and int(gen[-1]) == FUTURE_TOKEN:
                end -= 1
            snapshots.append(
                cls(
                    prior_generated=np.array(
                        gen[max(0, end - keep) : end], dtype=np.int64
                    )
                )
            )
        return snapshots


@dataclass
class StructuredOutputHelper:
    """Helper for structured output (constrained decoding) in text generation pipelines.

    Encapsulates grammar compilation and bitmask management, consolidating
    shared logic between TextGenerationPipeline and OverlapTextGenerationPipeline.

    Constrained decoding is used when:
    1. Feature enabled via feature flag (--enable-structured-output)
    2. Tool calling enforced via grammar (tool_choice=required, named function, or auto (conditional enforcement))
    """

    enabled: bool = False
    """Whether constrained decoding is available (tokenizer info initialized)."""
    enable_response_format_schema: bool = False
    """Whether user-provided json_schema is allowed."""
    vocab_size: int | None = None
    """Vocabulary size from the tokenizer, or None if disabled."""
    backend: GrammarBackend[Any] | None = field(default=None, repr=False)
    """Pluggable grammar backend (llguidance by default)."""
    tool_call_region_delimiters: StructuredOutputRegionDelimiters | None = None
    """Token sequences for tool call boundaries (conditional enforcement)."""
    # Serialises access to per-context ``ctx.matcher`` between the async
    # FSM-advance host callback and the synchronous spec-decode bitmask
    # path; concurrent calls into llguidance's ``LLInterpreter`` trip a
    # ``RuntimeError: Already borrowed`` and kill the worker coroutine.
    _matcher_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )

    def __post_init__(self) -> None:
        if self.enabled and self.backend is None:
            self.backend = LlguidanceBackend(None)

    @staticmethod
    def _get_tool_region_tags(
        tool_parser_name: str | None,
    ) -> tuple[str | None, str | None]:
        """Extract tool-calling *region* start/end tags from a registered parser.

        These tags control when ``GrammarEnforcementState`` toggles
        ``grammar_enforced``.  The start tag tells the state machine
        when to begin constraining (for ``tool_choice=auto`` where
        enforcement doesn't start immediately).  The end tag, if
        present, tells it when to stop.

        For **section-wrapped** parsers (e.g. Kimi K2.5, DeepSeek V3)
        that define ``SECTION_BEGIN``/``SECTION_END``, the outer
        section pair is returned — enforcement spans all tool calls.

        For **flat** parsers (e.g. Gemma 4) that only define
        ``CALL_BEGIN``/``CALL_END``, only ``CALL_BEGIN`` is returned
        as the start trigger; the end is ``None``.  ``CALL_END`` is a
        per-call delimiter, not a region boundary — using it would
        prematurely disable enforcement between consecutive tool calls,
        causing the model to generate unconstrained tokens before the
        next call.  With no end tag the grammar itself governs what
        follows each ``CALL_END`` (another call, or a terminal like
        ``<|tool_response>``).

        Args:
            tool_parser_name: Name of the registered tool parser, or None.

        Returns:
            A (start, end) pair of region tags.
        """
        parser_cls = get_parser_cls(tool_parser_name)
        if parser_cls is None:
            return (None, None)

        if not (
            isinstance(parser_cls, type)
            and issubclass(parser_cls, StructuralTagToolParser)
        ):
            return (None, None)

        if parser_cls.SECTION_BEGIN and parser_cls.SECTION_END:
            # Parsers that opt into enforcement-to-EOS get no end tag:
            # enforcement stays on after the section closes, so the
            # completed grammar masks everything but EOS and the turn
            # ends with its single section (e.g. MiniMax-M3).
            if parser_cls.ENFORCE_TOOL_REGION_TO_EOS:
                return (parser_cls.SECTION_BEGIN, None)
            return (parser_cls.SECTION_BEGIN, parser_cls.SECTION_END)
        if parser_cls.CALL_BEGIN:
            return (parser_cls.CALL_BEGIN, None)

        return (None, None)

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
        enable_structured_output: bool,
        tool_parser_name: str | None = None,
        backend_name: str | None = None,
        any_whitespace: bool | None = None,
    ) -> StructuredOutputHelper:
        """Create a helper from a tokenizer.

        Args:
            tokenizer: A pipeline tokenizer with a HuggingFace delegate attribute.
            enable_structured_output: Whether structured output is enabled
                (e.g. to constrain to response format json_schema).
            tool_parser_name: Name of the registered tool parser. Used to extract
                structural tags for tool call start/end markers.
            backend_name: Structured-output backend to use. ``None`` (the
                default, i.e. an unresolved ``SamplingConfig``) falls back to
                ``"xgrammar"``.
            any_whitespace: Whether ``response_format`` grammars accept
                whitespace between JSON tokens. ``None`` (an unresolved
                ``SamplingConfig``) falls back to ``False`` (compact JSON).

        Returns:
            A configured StructuredOutputHelper instance.

            Note: Constrained decoding is used when tool calling grammar is forced or enable_structured_output=True.
        """
        if not hasattr(tokenizer, "delegate"):
            return cls(enabled=False)
        tokenizer_delegate = tokenizer.delegate
        vocab_size = len(tokenizer_delegate)

        backend = make_grammar_backend(
            backend_name or DEFAULT_STRUCTURED_OUTPUT_BACKEND,
            tokenizer_delegate,
            vocab_size,
            tool_parser_name=tool_parser_name,
            stop_token_ids=tokenizer.eos_token_ids,
            any_whitespace=bool(any_whitespace),
        )

        # Extract structural tags from tool parser if available
        tool_start, tool_end = cls._get_tool_region_tags(tool_parser_name)

        # Tokenize start/end tags to get token ID sequences
        tool_call_region_delimiters: StructuredOutputRegionDelimiters | None = (
            None
        )
        if tool_start is not None or tool_end is not None:
            start_token_ids: list[int] | None = None
            end_token_ids: list[int] | None = None
            if tool_start is not None:
                start_token_ids = tokenizer_delegate.encode(
                    tool_start, add_special_tokens=False
                )
            if tool_end is not None:
                end_token_ids = tokenizer_delegate.encode(
                    tool_end, add_special_tokens=False
                )
            tool_call_region_delimiters = StructuredOutputRegionDelimiters(
                start_token_ids=start_token_ids,
                end_token_ids=end_token_ids,
            )

        return cls(
            enabled=True,
            enable_response_format_schema=enable_structured_output,
            vocab_size=vocab_size,
            backend=backend,
            tool_call_region_delimiters=tool_call_region_delimiters,
        )

    def build_matcher(
        self, grammar: str | None, json_schema: str | None
    ) -> GrammarMatcher:
        """Builds a grammar matcher without touching context state.

        Safe to call from any thread: it reads only the immutable backend,
        which releases the GIL during the expensive step. ``grammar`` takes
        precedence over ``json_schema``, matching :meth:`update_context`.
        """
        assert self.backend is not None
        if grammar:
            return self.backend.create_matcher(grammar)
        assert json_schema is not None
        compiled = self.backend.compile_json_schema(json_schema)
        # No-op on xgrammar, which rejects unsatisfiable schemas at compile
        # time; llguidance fails open without it.
        self.backend.validate_grammar(compiled)
        return self.backend.create_matcher(compiled)

    def install_matcher(
        self, context: TextGenerationContextType, matcher: GrammarMatcher
    ) -> None:
        """Installs a built matcher on a context.

        Sets the tool region for grammar requests; cheap enough for the
        decode thread.
        """
        context.set_matcher(matcher)
        if context.grammar:
            self.set_context_tool_region(context)

    def update_context(
        self,
        context: TextGenerationContextType,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        """Update context and bitmask for structured output.

        If a grammar is present, it is used directly. Otherwise,
        if a json_schema is present and no matcher is set, this compiles a
        grammar matcher and installs it on the context, then fills the
        per-request token bitmask.

        tool-call grammars (with any ``tool_choice`` setting) work
        regardless of ``enable_response_format_schema`` — the
        ``--enable-structured-output`` flag only gates user-supplied JSON
        schemas (via ``response_format``). Requests carrying a user schema
        set ``requires_structured_output_flag=True`` and are rejected when
        the flag is off.

        Args:
            context: Request context to update.
            bitmask: Preallocated bitmask buffer; updated in-place.
            index: Position in the bitmask for this request.

        Raises:
            InputError: If a JSON schema is provided but structured output is
                not enabled, or if constrained decoding is not available.
        """
        # Check for grammar first (e.g., tool call grammars from tool_choice=required)
        if context.grammar and context.matcher is None:
            if not self.enabled:
                raise InputError(
                    "grammar provided but constrained decoding is not available."
                )

            # ``--enable-structured-output`` gates user-supplied schemas (passed
            # via ``response_format``), not tool grammars.
            # Pure tool-call grammars work regardless of the flag. The combined
            # tool+schema case sets ``requires_structured_output_flag=True``.
            if (
                context.requires_structured_output_flag
                and not self.enable_response_format_schema
            ):
                raise InputError(
                    "response_format with a JSON schema requires "
                    "--enable-structured-output. Drop response_format to use "
                    "tool-call constraints only, or pass the flag to allow "
                    "schema-constrained responses."
                )

            assert self.backend is not None
            try:
                with Tracer("tool_grammar_compile"):
                    matcher = self.build_matcher(context.grammar, None)
                self.install_matcher(context, matcher)
            except Exception as e:
                raise InputError(
                    f"Grammar provided in request cannot be compiled. "
                    f"From {self.backend.name}: {e}"
                ) from e

        # Fall back to json_schema if no grammar
        # json_schema requires enable_response_format_schema (--enable-structured-output flag)
        elif context.json_schema is not None and context.matcher is None:
            if not self.enable_response_format_schema:
                raise InputError(
                    "json_schema provided but structured output is not enabled. "
                    "Pass --enable-structured-output to enable this feature."
                )

            assert self.backend is not None
            try:
                matcher = self.build_matcher(None, context.json_schema)
                self.install_matcher(context, matcher)
            except Exception as e:
                raise InputError(
                    f"JSON schema provided in request cannot be compiled to "
                    f"valid grammar. Update your JSON schema to produce valid "
                    f"structured output. From {self.backend.name}: {e}"
                ) from e

        if context.matcher:
            # Fill the bitmask for this context.
            self.fill_bitmask(context, bitmask, index)

    def allocate_bitmask(
        self,
        batch_size: int,
    ) -> npt.NDArray[np.int32]:
        """Allocate a token bitmask for the given batch size.

        Args:
            batch_size: Number of requests in the batch.

        Returns:
            A bitmask array of shape [batch_size, ceil(vocab_size/32)].

        Raises:
            ValueError: If vocab_size is not set.
        """
        if self.vocab_size is None:
            raise ValueError("vocab_size must be set to allocate bitmask")
        assert self.backend is not None
        return self.backend.allocate_token_bitmask(batch_size, self.vocab_size)

    def fill_bitmask(
        self,
        context: TextGenerationContextType,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        """Fill the bitmask for a context's matcher.

        Only fills the bitmask when the context has a matcher AND
        grammar_enforced is True. For conditional enforcement
        (tool_choice=auto), the bitmask is left unconstrained until
        the tool call start token is detected, and a matcher with no valid
        continuation leaves it unconstrained too (see
        :meth:`_fill_slot_unless_matcher_dead`).

        Args:
            context: Request context with a matcher.
            bitmask: Bitmask buffer to update in-place.
            index: Position in the bitmask for this request.
        """
        if context.matcher and context.grammar_enforced:
            self._fill_slot_unless_matcher_dead(
                context, context.matcher, bitmask, index
            )

    def set_context_tool_region(
        self,
        context: TextGenerationContextType,
    ) -> None:
        """Set the tool_region on context's grammar state if conditional enforcement.

        Called after setting the matcher to configure conditional enforcement
        for tool_choice=auto scenarios.

        Args:
            context: Request context with grammar state.
        """
        if self.tool_call_region_delimiters is not None:
            end_token_ids = self.tool_call_region_delimiters.end_token_ids
            if context.grammar_state.has_json_schema:
                end_token_ids = None
            context.set_tool_region(
                start_token_ids=self.tool_call_region_delimiters.start_token_ids,
                end_token_ids=end_token_ids,
            )

    def _tokens_for_consume(self, token: int, was_enforced: bool) -> list[int]:
        """Tokens to feed the matcher for one conditional-enforcement step.

        Mirrors ``TextGenerationContext._tokens_for_consume`` for the async
        spec-decode paths: on the enforcement flip-on (``was_enforced`` was
        False), feed the whole start marker rather than just the token that
        completed it, so multi-token / namespace-prefixed markers (e.g.
        MiniMax-M3's ``NS<tool_call>``) align with the grammar's start rule
        instead of rejecting into fail-open. Single-token markers have
        ``start_token_ids == [token]``, so this is a no-op for them.
        """
        delims = self.tool_call_region_delimiters
        if not was_enforced and delims and delims.start_token_ids:
            return list(delims.start_token_ids)
        return [token]

    def _fill_slot_unless_matcher_dead(
        self,
        ctx: TextGenerationContextType,
        matcher: GrammarMatcher,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        """Fill one bitmask slot, unless the matcher has no valid continuation.

        A matcher that is stopped *without* accepting has erred: llguidance
        leaves it stopped-and-not-accepting once a token is rejected, and
        ``fill_next_token_bitmask`` then writes an all-zero row. That row offers
        the sampler no candidate at all, and because the masked-out fill is
        finite by design -- so a fully-masked row degrades to a uniform draw
        rather than NaN -- the draw spreads across the whole vocabulary,
        including any padded tail the model masked to ``-inf``. Leaving the slot
        at its ``-1`` reset is strictly better: the request keeps generating
        from the model's own distribution instead of a uniform one, and the
        grammar was already unsatisfiable for this request.

        A stopped matcher that *is* accepting is a completed grammar, not an
        error. Its mask allows only EOS, which is correct and still applied --
        a backend whose native fill call can't tolerate a stopped matcher
        (see :meth:`XgrammarBackend.fill_next_token_bitmask`) is responsible
        for computing that EOS-only mask itself, since llguidance's real,
        frequently-hit case here is a normal successful completion, not an
        error state.

        Args:
            ctx: The request context, for the report latch and diagnostics.
            matcher: The matcher to read the mask from (a speculative copy on
                the spec-decode path, the request's own matcher otherwise).
            bitmask: The ``[rows, packed_vocab]`` bitmask to write into.
            index: Row of ``bitmask`` to fill.
        """
        if matcher.is_stopped() and not matcher.is_accepting():
            if not ctx.grammar_state.dead_matcher_reported:
                ctx.grammar_state.dead_matcher_reported = True
                logger.error(
                    "Matcher for request %s is stopped without accepting, so "
                    "it has no valid next token; leaving its bitmask "
                    "unconstrained for the rest of the request. "
                    "matcher_errors=%s matcher_warnings=%s",
                    ctx.request_id,
                    matcher.get_error(),
                    matcher.get_grammar_warnings(),
                )
            return
        assert self.backend is not None
        self.backend.fill_next_token_bitmask(matcher, bitmask, index)

    def _speculatively_fill_bitmask_window(
        self,
        ctx: TextGenerationContextType,
        drafts: npt.NDArray[np.int64],
        bitmask_window: npt.NDArray[np.int32],
    ) -> None:
        """Advance enforcement state through drafts, filling per-slot bitmasks.

        A draft that flips enforcement on mid-window causes downstream
        slots to be constrained: e.g. a ``</think>`` draft exits the
        thinking region, so the slot immediately after it gets a filled
        bitmask instead of staying unconstrained. The matcher is walked
        on a deep copy (never mutated), and enforcement state is restored
        at the end, so committed-token processing on the next batch
        replays the same transitions from a clean state.

        Out-of-vocab drafts stop the speculative advance and leave any
        remaining slots unconstrained; they are not treated as errors.

        Args:
            ctx: The request context.
            drafts: Candidate draft tokens for the next batch.
            bitmask_window: ``[num_positions, packed_vocab]`` with
                ``num_positions >= len(drafts)+1``, pre-initialized to
                ``-1`` (unconstrained). Slot 0 is the position
                immediately after the committed tokens; slot ``i+1`` is
                the position after consuming ``drafts[i]``. Slots stay
                ``-1`` wherever grammar enforcement is off, where the
                matcher has no valid continuation (see
                :meth:`_fill_slot_unless_matcher_dead`).
        """
        assert ctx.matcher is not None
        fsm_snap = ctx.snapshot_grammar_state()

        # The window keeps the graph's static width while the realized draft
        # width can shrink (a prefill->decode batch verifies zero drafts), so a
        # wider window is normal; unreached slots keep their -1. Too narrow is a
        # caller bug worth raising on.
        if drafts.shape[0] + 1 > bitmask_window.shape[0]:
            raise ValueError(
                f"bitmask window has {bitmask_window.shape[0]} slots but "
                f"{drafts.shape[0]} drafts were passed; the window must hold "
                "at least one slot per draft plus the bonus slot."
            )
        # Speculatively consume drafts on a throwaway copy of the matcher.
        # LLMatcher.rollback() is not a perfect inverse when the consumed
        # span crosses a grammar rule/repetition boundary — e.g.
        # ``<|tool_call_begin|>`` can cause issues for rollback. Bypass this
        # issue by taking a deep copy instead.
        matcher_copy = ctx.matcher.deep_copy()

        # Slot 0: state immediately after committed tokens.
        if ctx.grammar_enforced:
            self._fill_slot_unless_matcher_dead(
                ctx, matcher_copy, bitmask_window[0, :].reshape(1, -1), 0
            )

        vocab_size = self.vocab_size or 0
        for i in range(drafts.shape[0]):
            draft_token = int(drafts[i])
            if draft_token < 0 or draft_token >= vocab_size:
                break

            # EOS-class tokens are not part of the grammar — they signal end of
            # generation. Skip the matcher so it stays in a clean terminal
            # state. ``restore_grammar_state`` undoes this transient flip.
            # Drafts past EOS are pointless (the request ended), so exit the
            # loop and leave remaining slots unconstrained.
            if draft_token in ctx.eos_tracker.eos_token_ids:
                ctx.grammar_enforced = False
                break

            consumed = False
            was_enforced = ctx.grammar_enforced
            if ctx.update_enforcement_state(draft_token):
                tokens = self._tokens_for_consume(draft_token, was_enforced)
                if matcher_copy.try_consume_tokens(tokens) == len(tokens):
                    consumed = True
                else:
                    break

            if consumed or ctx.grammar_enforced:
                self._fill_slot_unless_matcher_dead(
                    ctx,
                    matcher_copy,
                    bitmask_window[i + 1, :].reshape(1, -1),
                    0,
                )

        ctx.restore_grammar_state(fsm_snap)

    def _rejection_diagnostics(
        self,
        ctx: TextGenerationContextType,
        committed_tokens: list[int],
        committed_idx: int,
    ) -> str:
        """Best-effort extra state for the matcher-rejection error log.

        Runs only on the (rare) rejection path and is fully guarded so a
        diagnostic failure can never crash the async worker thread. Surfaces
        whether the rejection landed in the middle of a tool call (a desync
        signature) versus at a clean grammar boundary:

        * ``matcher_accepting=False`` means the matcher was mid-structure
          (inside a call header / args), not at a stoppable boundary.
        * ``open_sections>0`` means more ``<|tool_calls_section_begin|>`` than
          ``...section_end|>`` are committed, i.e. an open tool-call section.
        * ``committed_token_ids`` is this spec-decode step's accepted-drafts +
          bonus token, as raw token IDs, so the exact desyncing batch can be
          reconstructed offline against the tokenizer.

        Only token IDs are logged (no decoded text), so no model output text
        reaches the logs; reconstruct decoded forms after the fact.
        """
        try:
            matcher = ctx.matcher
            snapshot = ctx.snapshot_grammar_state()

            # "Inside an open tool-call section": section-begins minus
            # section-ends committed so far.
            delims = self.tool_call_region_delimiters
            open_sections = -1
            if (
                delims is not None
                and delims.start_token_ids
                and delims.end_token_ids
            ):
                generated = [int(t) for t in ctx.tokens.generated]
                open_sections = _count_token_subsequence(
                    generated, delims.start_token_ids
                ) - _count_token_subsequence(generated, delims.end_token_ids)

            return (
                f"reject_idx={committed_idx}/{len(committed_tokens)} "
                f"matcher_accepting="
                f"{matcher.is_accepting() if matcher is not None else '?'} "
                f"matcher_stopped="
                f"{matcher.is_stopped() if matcher is not None else '?'} "
                f"enforced={ctx.grammar_enforced} "
                f"tools_forced={ctx.tools_forced} "
                f"in_thinking_region={snapshot.in_thinking_region} "
                f"open_sections={open_sections} "
                f"committed_token_ids={list(committed_tokens)}"
            )
        except Exception as e:
            return f"<diagnostics unavailable: {e!r}>"

    @traced
    def advance_fsm_and_compute_bitmasks(
        self,
        context_batch: list[TextGenerationContextType],
        accepted_draft_tokens: npt.NDArray[np.int64],
        num_accepted: npt.NDArray[np.int64],
        bonus_tokens: npt.NDArray[np.int64],
        next_draft_tokens: npt.NDArray[np.int64],
        bitmask_out: npt.NDArray[np.int32],
        output_context_batch: list[TextGenerationContextType] | None = None,
        committed_span_snapshots: Sequence[CommittedSpanSnapshot] | None = None,
    ) -> None:
        """Advance FSM through accepted tokens, then compute bitmasks for the next batch.

        Combines FSM advancement (Part 1) with bitmask computation (Part 2) for
        use in a CUDA host callback. Must NOT call any CUDA APIs.

        Part 1 permanently advances the FSM of every ``context_batch`` request
        through its committed tokens (accepted draft tokens followed by the
        bonus token). This mirrors what sync_and_process_outputs would do for
        structured output, and is independent of the output row order. A row
        preempted in flight (``reset()`` after enqueue) is skipped: ``reset()``
        rewinds its token buffer but preserves its matcher, so advancing
        through the dropped committed token would desync the matcher from the
        sequence it re-primes on resume.

        Part 2 owns the **entire** ``output_context_batch`` rectangle and
        writes each row's speculative bitmask **in that batch's row order**, by
        speculatively advancing the already-advanced matcher through the next
        batch's draft tokens and rolling back. Every row is first reset to -1
        (all valid), then each constrained row is filled from its request's
        ``next_draft_tokens``. This runs only from a callback enqueued when the
        whole current batch verifies drafts, so every consumer row continues
        from ``context_batch`` (the scheduler routes fresh/resumed requests
        through the cold-start prime path instead; see the caller
        ``_enqueue_prev_bitmask_callback`` and ``_assign_bitmask_inputs``). A
        row preempted in flight (``reset()`` after enqueue) is degraded to the
        all-valid -1 reset rather than raising, which would blanket-reset the
        whole rectangle and unconstrain every other row. Writing directly in
        the consumer's row order, with no second writer on the main thread, is
        what lets the model graph consume the bitmask without an on-device
        gather and without a host wait.

        Args:
            context_batch: Requests whose FSM is advanced (the producing batch).
                Indexes ``accepted_draft_tokens`` / ``num_accepted`` /
                ``bonus_tokens`` / ``next_draft_tokens``.
            accepted_draft_tokens: Draft tokens verified this batch, shape [batch, K].
            num_accepted: Count of accepted draft tokens per request, shape [batch].
            bonus_tokens: Bonus (target) tokens per request, shape [batch].
            next_draft_tokens: Draft tokens for the next batch, shape
                [batch, K], at the configured draft depth. Trimmed to the
                consuming step's verify width, which may be narrower.
            bitmask_out: Packed int32 bitmask output, shape
                [len(output_context_batch), verify_width+1, packed_vocab],
                where ``verify_width`` is how many drafts the consuming step
                verifies. Every row is reset to -1 (unconstrained) before
                filling.
            output_context_batch: Requests in the consuming batch's row order.
                Defaults to ``context_batch`` when the batch did not change.
            committed_span_snapshots: Per-context_batch-row token history
                captured on the main thread at enqueue time. Required from an
                async caller, whose ctx.tokens reads would otherwise race
                the main thread's token realization; a synchronous caller may
                omit it and have it captured here. A count mismatch is logged
                and falls back to a worker-side read, never raised on.
        """
        if output_context_batch is None:
            output_context_batch = context_batch
        if committed_span_snapshots is None:
            committed_span_snapshots = CommittedSpanSnapshot.capture(
                context_batch
            )
        elif len(committed_span_snapshots) != len(context_batch):
            # Fall back to reading the buffer rather than raising: the batch is
            # already marked FSM-advanced, so a raise would leave every row this
            # loop had not reached stuck behind its request's emitted tokens.
            # Reading here races the main thread's writes: the offset stays
            # exact for a row with no multi-token stop sequence, whose history
            # goes unread, and may be wrong for any other row.
            logger.error(
                "Async FSM advance got %d committed-span snapshots for %d "
                "rows; falling back to a worker-side read. The end-of-sequence "
                "offset may be wrong for a row whose stop sequence straddles "
                "the newly committed tokens. Capture the snapshots from the "
                "same batch, before enqueuing.",
                len(committed_span_snapshots),
                len(context_batch),
            )
            committed_span_snapshots = CommittedSpanSnapshot.capture(
                context_batch
            )
        # This method runs on an AsyncRT worker thread. The main thread
        # may try to access the same ``ctx.matcher`` via
        # ``compute_speculative_bitmasks`` for the next iter while this
        # callback is still in flight; without serialisation llguidance
        # raises ``RuntimeError: Already borrowed`` and the worker dies.
        # See the comment on ``_matcher_lock``.
        with self._matcher_lock:
            # Part 1: permanently advance every producing-batch matcher
            # through its committed tokens. Order-independent of the output
            # batch, so it always covers all of ``context_batch`` — which is
            # what keeps the batch-level ``skip_fsm_advance`` contract intact
            # for the producing batch's later sync.
            for ctx_idx, ctx in enumerate(context_batch):
                if (
                    ctx.matcher is None
                    or ctx.is_initial_prompt
                    or ctx._is_padding_ctx
                ):
                    continue

                # Advance the enforcement state machine through committed
                # tokens, one at a time so special tokens (e.g. tool-call
                # structural tags) can flip grammar enforcement mid-sequence.
                n_accepted = int(num_accepted[ctx_idx])
                bonus_token = int(bonus_tokens[ctx_idx])
                committed_tokens = [
                    int(accepted_draft_tokens[ctx_idx, j])
                    for j in range(n_accepted)
                ]
                committed_tokens.append(bonus_token)
                eos_offset = ctx.eos_tracker.first_eos_offset(
                    committed_span_snapshots[ctx_idx].prior_generated,
                    committed_tokens,
                )
                for committed_idx, token in enumerate(committed_tokens):
                    # Generation stops at the first terminating token; tokens
                    # after it are never emitted, so disable enforcement and
                    # stop rather than advancing the matcher through them.
                    if committed_idx == eos_offset:
                        ctx.grammar_enforced = False
                        break
                    was_enforced = ctx.grammar_enforced
                    if not ctx.update_enforcement_state(token):
                        continue
                    # On the enforcement flip-on, feed the matcher the whole
                    # start marker (multi-token / NS-prefixed markers like
                    # M3's NS<tool_call>), not just the completing token.
                    tokens = self._tokens_for_consume(token, was_enforced)
                    if ctx.matcher.try_consume_tokens(tokens) == len(tokens):
                        continue
                    # ``role`` distinguishes a rejection on the bonus
                    # token (sampled by target *with* bitmask, so a
                    # rejection here usually means a bitmask/matcher
                    # desync) from a rejection on an accepted draft
                    # (produced by the draft model and verified by
                    # target, where rejection more often reflects the
                    # target sampling outside the matcher's allowed
                    # set on a draft slot the speculative walk did
                    # not constrain).
                    role = (
                        "bonus"
                        if committed_idx == len(committed_tokens) - 1
                        else f"accepted_draft[{committed_idx}]"
                    )
                    logger.error(
                        "Async matcher rejected %d token(s) ending at %d "
                        "(request %s, role=%s); disabling enforcement "
                        "for the rest of the request. "
                        "matcher_errors=%s matcher_warnings=%s %s",
                        len(tokens),
                        token,
                        ctx.request_id,
                        role,
                        ctx.matcher.get_error(),
                        ctx.matcher.get_grammar_warnings(),
                        self._rejection_diagnostics(
                            ctx, committed_tokens, committed_idx
                        ),
                    )
                    ctx.grammar_enforced = False

            # Part 2: write each output row's speculative bitmask in the
            # consuming batch's row order. ``rid_to_src`` maps a request id to
            # its slot in the producing batch's ``next_draft_tokens``.
            rid_to_src = {
                ctx.request_id: i for i, ctx in enumerate(context_batch)
            }
            for out_idx, ctx in enumerate(output_context_batch):
                # The callback owns every consumer row. Reset to -1 (all valid)
                # up front so an unconstrained continuing row needs no further
                # work and no row is ever left holding a previous iteration's
                # bitmask for the next iter's in-graph H2D to copy.
                bitmask_out[out_idx, :, :] = -1
                # The callback is enqueued only when the whole current batch
                # verifies drafts, so every consumer row should continue from
                # the producing batch. But the callback runs on an AsyncRT
                # worker and holds live references to these contexts: between
                # its enqueue and its execution the scheduler can preempt a row
                # (``reset()`` to an initial prompt, requeuing it to
                # context-encoding) when KV pages run short. Degrade such a row
                # to the all-valid -1 reset above and ``continue`` rather than
                # raising -- a raise propagates to the callback's except and
                # blanket-resets the *whole* rectangle to -1, unconstraining
                # every other (correctly continuing) request in the batch.
                if ctx._is_padding_ctx:
                    continue
                src = rid_to_src.get(ctx.request_id)
                if src is None:
                    logger.error(
                        "bitmask callback: row %s absent from the producing "
                        "batch -- a scheduler change admitted a new or resumed "
                        "row into a verify batch without a synchronous fill. "
                        "Leaving this row unconstrained for this step.",
                        ctx.request_id,
                    )
                    continue
                if ctx.is_initial_prompt:
                    # Preempted in flight: its token is dropped and it re-primes
                    # on resume, so -1 is the correct don't-care. Debug-only to
                    # avoid per-row log spam on the hot path under KV pressure.
                    logger.debug(
                        "bitmask callback: row %s was preempted in flight; "
                        "leaving it unconstrained for this step.",
                        ctx.request_id,
                    )
                    continue
                if ctx.matcher is None:
                    # Continuing unconstrained row: all-valid, no fill needed.
                    continue
                # A draft that flips enforcement on mid-window causes
                # downstream slots to be constrained.
                # The producing batch drafted the configured depth, but the
                # consuming step may verify fewer, and the window is sized from
                # what it verifies. Trim to the window's draft slots so a
                # narrower consumer cannot overrun it.
                self._speculatively_fill_bitmask_window(
                    ctx,
                    drafts=next_draft_tokens[src, : bitmask_out.shape[1] - 1],
                    bitmask_window=bitmask_out[out_idx],
                )

    @traced
    def compute_speculative_bitmasks(
        self,
        context_batch: list[TextGenerationContextType],
        draft_tokens: npt.NDArray[np.int64],
        num_positions: int,
    ) -> npt.NDArray[np.int32]:
        """Compute speculative bitmasks for structured output in spec decode.

        For each draft position i, the bitmask at position i contains valid
        tokens given the FSM state after consuming draft[0:i-1]. The last
        position (num_positions - 1) is for the bonus token.

        This method speculatively advances the FSM through draft tokens to
        compute bitmasks, then rolls back to restore the original state.

        The bitmask is returned packed (1 bit per token, 32 tokens per int32
        word); the GPU acceptance sampler unpacks and applies it in one fused
        pass, so this method never unpacks to bool.

        Args:
            context_batch: List of generation contexts.
            draft_tokens: Draft tokens to verify, shape
                [batch, verify_width].
            num_positions: Number of bitmask positions
                (``verify_width + 1``, including bonus).

        Returns:
            Packed int32 bitmask array of shape
            ``[batch_size, num_positions, ceil(vocab_size / 32)]``. ``-1`` (all
            bits set) means all tokens are valid.
        """
        if self.vocab_size is None:
            raise ValueError("vocab_size must be set for speculative bitmasks")

        batch_size = len(context_batch)
        packed_vocab_size = ceildiv(self.vocab_size, 32)

        # Check if any context has structured output
        has_structured_output = any(
            ctx.json_schema is not None
            or ctx.matcher is not None
            or ctx.grammar is not None
            for ctx in context_batch
        )

        if not has_structured_output:
            # Fast path: all unconstrained, return all-valid packed bitmask
            # (-1 = all bits set).
            return np.full(
                (batch_size, num_positions, packed_vocab_size),
                -1,
                dtype=np.int32,
            )

        assert self.backend is not None
        packed_bitmask = self.backend.allocate_token_bitmask(
            batch_size * num_positions, self.vocab_size
        )
        packed_vocab_size = packed_bitmask.shape[1]
        packed_bitmask = packed_bitmask.reshape(
            batch_size, num_positions, packed_vocab_size
        )

        # Serialise against the async FSM-advance host callback
        # (``advance_fsm_and_compute_bitmasks``). Both paths touch the
        # same ``ctx.matcher`` LLInterpreter; concurrent access trips
        # llguidance's "Already borrowed" Rust panic and kills the
        # worker. See the comment on ``_matcher_lock``.
        with self._matcher_lock:
            # Initialize matchers for contexts with json_schema or grammar
            for ctx in context_batch:
                needs_matcher = ctx.matcher is None and (
                    ctx.json_schema is not None or ctx.grammar is not None
                )
                if needs_matcher:
                    self.update_context(
                        ctx,
                        packed_bitmask[0, 0, :].reshape(
                            1, -1
                        ),  # Dummy, will be overwritten
                        index=0,
                    )

            # Fill bitmasks for each context. ``packed_bitmask`` is
            # initialized to -1 (all bits set = all tokens valid), so the
            # helper only needs to write slots where the FSM is enforced.
            for ctx_idx, ctx in enumerate(context_batch):
                if not ctx.matcher:
                    continue
                self._speculatively_fill_bitmask_window(
                    ctx,
                    drafts=draft_tokens[ctx_idx],
                    bitmask_window=packed_bitmask[ctx_idx],
                )
        # Return the packed int32 bitmask directly; the GPU acceptance sampler
        # unpacks and applies it in a single fused pass.
        return packed_bitmask


def get_structured_output_helper(
    pipeline: object,
) -> StructuredOutputHelper | None:
    """Returns the pipeline's structured-output helper, if it exposes one."""
    helper = getattr(pipeline, "_structured_output", None)
    return helper if isinstance(helper, StructuredOutputHelper) else None
