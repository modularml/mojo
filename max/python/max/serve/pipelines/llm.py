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

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic, cast

from max.pipelines.context import (
    BaseContextType,
    GenerationStatus,
    LogProbabilities,
    TextAndVisionContext,
    TextContext,
    TextGenerationOutput,
)
from max.pipelines.lib import reasoning
from max.pipelines.modeling.types import (
    EmbeddingsGenerationOutput,
    PipelineOutputType,
    PipelineTokenizer,
    ReasoningParser,
    RequestType,
    TextGenerationRequest,
)
from max.profiler import Tracer
from max.serve.pipelines.incremental_detokenizer import (
    BufferedDetokenizer,
    create_buffered_detokenizer,
)
from max.serve.telemetry.common import request_trace_ctx
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch, record_ms
from max.serve.worker_interface import ModelWorkerProxy
from max.serve.worker_interface.lora_queue import LoRAQueue
from opentelemetry import propagate as otel_propagate

logger = logging.getLogger("max.serve")


@dataclass(frozen=True)
class TokenGeneratorOutput:
    """Output from token generation - can contain a chunk of tokens.

    When yielded from next_token_chunk(), contains combined decoded text from
    all tokens in a single scheduler response. The chunk size equals
    len(response.tokens) from the model worker.

    Unless otherwise indicated, statistics and containers do not consider
    prompt or reasoning tokens.
    """

    status: GenerationStatus
    # Combined decoded text from all tokens in this chunk
    decoded_tokens: str | None = None
    # Combined decoded text from all reasoning tokens in this chunk
    decoded_reasoning_tokens: str | None = None
    # Number of tokens in this chunk (1 for single token, N for chunk)
    token_count: int = 1
    # TODO: (MODELS-1118) determine whether to include logprobs for reasoning tokens in the response delta
    token_log_probabilities: list[float] | None = None
    top_log_probabilities: list[dict[str, float]] | None = None
    prompt_token_count: int | None = None
    cached_token_count: int | None = None
    reasoning_token_count: int | None = None
    stop_sequence: str | None = None
    batch_id: int | None = None
    """Monotonic forward-pass counter from the scheduler that produced this
    chunk. Used to correlate API-side OTel spans with model-worker spans."""


def _merge_outputs(chunks: list[TokenGeneratorOutput]) -> TokenGeneratorOutput:
    """Combine several consecutive TokenGeneratorOutputs into one.

    Decoded text concatenates directly because the detokenizer is stateful
    across the whole request, so consecutive decoded pieces reassemble the
    exact stream text. status/stop_sequence come from the last chunk; the
    first-chunk fields take the first non-None value.
    """
    assert chunks
    content_parts = [c.decoded_tokens for c in chunks if c.decoded_tokens]
    reasoning_parts = [
        c.decoded_reasoning_tokens for c in chunks if c.decoded_reasoning_tokens
    ]
    token_log_probs = [
        p
        for c in chunks
        if c.token_log_probabilities
        for p in c.token_log_probabilities
    ]
    top_log_probs = [
        p
        for c in chunks
        if c.top_log_probabilities
        for p in c.top_log_probabilities
    ]

    def _first_not_none(attr: str) -> Any:
        for c in chunks:
            v = getattr(c, attr)
            if v is not None:
                return v
        return None

    stop_sequence = None
    for c in chunks:
        if c.stop_sequence is not None:
            stop_sequence = c.stop_sequence

    return TokenGeneratorOutput(
        status=chunks[-1].status,
        decoded_tokens="".join(content_parts) if content_parts else None,
        decoded_reasoning_tokens="".join(reasoning_parts)
        if reasoning_parts
        else None,
        token_count=sum(c.token_count for c in chunks),
        token_log_probabilities=token_log_probs or None,
        top_log_probabilities=top_log_probs or None,
        prompt_token_count=_first_not_none("prompt_token_count"),
        cached_token_count=_first_not_none("cached_token_count"),
        reasoning_token_count=sum(c.reasoning_token_count or 0 for c in chunks)
        or None,
        stop_sequence=stop_sequence,
        batch_id=chunks[-1].batch_id,
    )


def _inject_trace_carrier(context: BaseContextType) -> None:
    """Serialize the current request's OTel trace context onto ``context``.

    ``context`` crosses into the model-worker process by value (pickled onto
    the request queue), so the ambient ``request_trace_ctx`` -- populated by
    the route handler from the inbound request's W3C headers -- can't follow
    it there directly. Inject it into a plain string-dict carrier instead,
    which the scheduler re-``extract``s to parent its phase spans under the
    caller's trace. A no-op for context types that don't carry a
    ``trace_carrier`` field (only ``TextContext`` does).
    """
    if not isinstance(context, TextContext):
        return
    carrier: dict[str, str] = {}
    otel_propagate.inject(carrier, context=request_trace_ctx.get())
    if carrier:
        context.trace_carrier = carrier


def _apply_stop_truncation(
    merged: TokenGeneratorOutput,
) -> TokenGeneratorOutput:
    """Drop the stop string and anything after it from a flushed chunk's
    content. The coalescer buffers, so a matched stop string is normally
    wholly present in the merged text and located exactly by find().

    Accepted limitation: a stop string split across an already-emitted prior
    chunk boundary cannot be retroactively trimmed. This matches every
    streaming implementation and is unchanged from today's behavior.
    """
    if merged.stop_sequence is None or not merged.decoded_tokens:
        return merged
    idx = merged.decoded_tokens.find(merged.stop_sequence)
    if idx < 0:
        return merged  # straddled an already-emitted chunk (rare) — accepted limitation
    return replace(merged, decoded_tokens=merged.decoded_tokens[:idx] or None)


async def _coalesce_chunks(
    source: AsyncGenerator[TokenGeneratorOutput, None],
    min_chunk_tokens: int,
) -> AsyncGenerator[TokenGeneratorOutput, None]:
    """Buffer streamed token chunks to a minimum size and drop empty deltas.

    Flush rules:
    - first visible chunk flushes as soon as it has text (TTFT unaffected);
    - a terminal chunk (status.is_done) always flushes;
    - the token floor only counts once there is non-empty text, so empty
      deltas (partial multi-byte char / skipped special token) are buffered
      until real text arrives and never emitted alone.
    """
    buffer: list[TokenGeneratorOutput] = []
    buffered_tokens = 0
    first_emitted = False
    async for chunk in source:
        buffer.append(chunk)
        buffered_tokens += chunk.token_count + (
            chunk.reasoning_token_count or 0
        )
        merged = _merge_outputs(buffer)
        has_text = bool(merged.decoded_tokens) or bool(
            merged.decoded_reasoning_tokens
        )
        must_flush = chunk.status.is_done
        reached_floor = has_text and (
            not first_emitted or buffered_tokens >= min_chunk_tokens
        )
        if must_flush or reached_floor:
            yield _apply_stop_truncation(merged)  # Part B
            buffer, buffered_tokens, first_emitted = [], 0, True
    if buffer:
        yield _apply_stop_truncation(_merge_outputs(buffer))


class BasePipeline(Generic[BaseContextType, RequestType, PipelineOutputType]):
    def __init__(
        self,
        model_name: str,
        tokenizer: PipelineTokenizer[BaseContextType, Any, RequestType],
        model_worker: ModelWorkerProxy[BaseContextType, PipelineOutputType],
        lora_queue: LoRAQueue | None = None,
    ) -> None:
        self.logger = logging.getLogger(
            self.__class__.__module__ + "." + self.__class__.__qualname__
        )
        # This logger is too verbose to expose to end users. Disable propagation to the root logger by default.
        self.debug_logging = self.logger.isEnabledFor(logging.DEBUG)

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.lora_queue = lora_queue
        self.model_worker = model_worker


class TokenGeneratorPipeline(
    BasePipeline[
        TextAndVisionContext | TextContext,
        TextGenerationRequest,
        TextGenerationOutput,
    ]
):
    """Base class for LLM text generation pipelines."""

    def __init__(
        self,
        *args,
        reasoning_parser_name: str | None = None,
        min_chunk_tokens: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reasoning_parser_name = reasoning_parser_name
        self._min_chunk_tokens = max(1, min_chunk_tokens)

    async def _reasoning_parser(self) -> ReasoningParser | None:
        if self._reasoning_parser_name is None:
            return None
        return await reasoning.create(
            self._reasoning_parser_name, self.tokenizer
        )

    async def _top_log_probs(
        self,
        log_prob: LogProbabilities,
        skip_special_tokens: bool,
    ) -> list[dict[str, float]]:
        top_log_probabilities = []
        for top_token_log_probs in log_prob.top_log_probabilities:
            decoded_log_probs = {}
            for token_id, value in top_token_log_probs.items():
                decoded_log_probs[
                    await self.tokenizer.decode(
                        token_id, skip_special_tokens=skip_special_tokens
                    )
                ] = value
            top_log_probabilities.append(decoded_log_probs)

        return top_log_probabilities

    async def next_token_chunk(
        self, request: TextGenerationRequest
    ) -> AsyncGenerator[TokenGeneratorOutput, None]:
        """Tokenizes and submits ``request``, returning a token-chunk generator.

        Awaiting this coroutine tokenizes the request and hands it off to the
        model worker. A failure during that submission (e.g. a tokenization
        error or a dead worker) raises here, before the generator is returned,
        so the caller can surface it as an error response before any streaming
        headers are sent.

        Iterating the returned generator streams token chunks aligned with
        scheduler responses. Each chunk contains all tokens from a single model
        worker response. Benefits:
        - Single tokenizer.decode() call per chunk instead of per token
        - Callers can amortize Pydantic/SSE overhead across the chunk
        """
        itl = StopWatch()
        # TTFT runs from the arrival timestamp stamped by the HTTP middleware,
        # so it covers body parse, validation and media resolution rather than
        # starting here at pipeline entry. Offline callers leave timestamp_ns
        # at its 0 default; those fall back to pipeline entry.
        ttft_sw = StopWatch(start_ns=request.timestamp_ns or None)
        total_sw = StopWatch()
        decode_sw = StopWatch()
        decode_elapsed_ms = 0.0
        num_generated_tokens = 0
        self.logger.debug(
            "%s: Started: Elapsed: %0.2f ms",
            request.request_id,
            total_sw.elapsed_ms,
        )

        # Always skip special tokens in decoded output
        # (EOS tokens like <|im_end|> should not appear in the text response)
        skip_special_tokens = True

        # Track whether we've yielded the first chunk (for TTFT metric)
        first_chunk_yielded = False

        # For reasoning models, we assume that there is always a reasoning span at the very start
        # We do not support multiple reasoning spans per response
        # We also do not support reasoning spans that are not at the very start of the response
        # This is consistent with vLLM
        # TODO: (MODELS-1115) assume that the reasoning tokens are at the start of the reasoning section
        reasoning_parser = await self._reasoning_parser()
        if reasoning_parser is not None:
            reasoning_parser.reset()
        is_still_reasoning = reasoning_parser is not None

        # Count this request as awaiting admission to the model worker: it has
        # been accepted by the API server but is still API-side (tokenization /
        # pre-submit). Decremented once the handoff to the worker succeeds (or
        # fails) below, so a persistently high gauge points at an API-server
        # backlog rather than the scheduler queue.
        self.model_worker.note_awaiting_admission(1)

        try:
            with record_ms(METRICS.input_time):
                context = await self.tokenizer.new_context(request)
            _inject_trace_carrier(context)

            # Create buffered detokenizers for proper UTF-8 handling.
            # These handle multi-byte UTF-8 sequences that span multiple tokens,
            # such as emojis like 😊 which require 4 bytes and may be split
            # across multiple tokens. Without buffered detokenization, each
            # token decoded separately would produce replacement characters (�).
            # See SERVSYS-1032 and MXSERV-61 for details.
            content_detokenizer: BufferedDetokenizer = (
                create_buffered_detokenizer(
                    self.tokenizer,
                    context.tokens.prompt,
                    skip_special_tokens=skip_special_tokens,
                )
            )
            reasoning_detokenizer: BufferedDetokenizer = (
                create_buffered_detokenizer(
                    self.tokenizer,
                    context.tokens.prompt,
                    skip_special_tokens=skip_special_tokens,
                )
            )

            if is_still_reasoning:
                # Check if reasoning was disabled in the prompt. Use the
                # parser's prompt-aware decision so multi-turn prompts (which
                # legitimately contain ``</think>`` from prior assistant
                # turns) don't false-trigger "reasoning already ended".
                assert reasoning_parser is not None
                is_still_reasoning = reasoning_parser.will_reason_after_prompt(
                    cast(Sequence[int], context.tokens.prompt)
                )

            # Suppress reasoning classification only when constrained decoding
            # will actually constrain the model from the first token with no
            # way to suspend it for reasoning. Two escape hatches keep
            # reasoning live:
            #
            #   1. ``grammar_enforced=False`` on a context that has a grammar
            #      (tool_choice=auto): the grammar is compiled but the bitmask
            #      is gated until a tool-call start token is seen, so the model
            #      can reason freely up to that point.
            #   2. A configured thinking region (thinking_region_delimiters): GrammarEnforcementState
            #      suspends grammar during ``<think>...</think>``.
            #
            # Note that when reasoning classification is disabled reasoning
            # tokens are routed to the content field, not reasoning.
            grammar_will_constrain_from_start = (
                context.grammar and context.grammar_enforced
            ) or context.json_schema
            has_thinking_region = (
                hasattr(context, "grammar_state")
                and context.grammar_state.thinking_region_delimiters is not None
            )
            if (
                is_still_reasoning
                and grammar_will_constrain_from_start
                and not has_thinking_region
            ):
                is_still_reasoning = False

            has_stop_sequences = bool(context.eos_tracker.eos_stop_strings)

            # Hand the request off to the model worker. Awaiting the submit
            # performs the handoff (e.g. the zmq put), so a failure here — for
            # example a dead worker — raises before the generator is returned,
            # letting the caller respond with an HTTP error before streaming
            # headers are sent.
            response_stream = await self.model_worker.stream(
                context.request_id, context
            )
        except BaseException:
            # Balance the awaiting-admission counter if we never reached a
            # successful handoff (tokenization failed or the submit raised).
            self.model_worker.note_awaiting_admission(-1)
            raise

        # Handoff succeeded: the request is no longer awaiting admission.
        self.model_worker.note_awaiting_admission(-1)

        async def _generate() -> AsyncGenerator[TokenGeneratorOutput, None]:
            nonlocal \
                decode_elapsed_ms, \
                num_generated_tokens, \
                first_chunk_yielded, \
                is_still_reasoning
            try:
                with record_ms(METRICS.output_time):
                    async for responses, batch_id in response_stream:
                        assert isinstance(responses, list)
                        assert len(responses) > 0
                        assert isinstance(responses[0], TextGenerationOutput)
                        response = TextGenerationOutput.merge(responses)

                        num_generated_tokens += len(response.tokens)

                        tokens: list[int] | None = response.tokens
                        token_log_probs = response.log_probabilities
                        reasoning_tokens = None
                        reasoning_text_formatter = None

                        if reasoning_parser is not None:
                            # Always run the parser, even when we weren't seeded
                            # into reasoning. This lets architectures like Gemma
                            # 4 — which can emit
                            # ``<|channel>thought\n...<channel|>`` mid-stream
                            # regardless of enable_thinking — detect those
                            # reasoning sections dynamically rather than leaking
                            # them as content.
                            parsed = reasoning_parser.stream(
                                response.tokens,
                                is_currently_reasoning=is_still_reasoning,
                            )
                            reasoning_span = parsed.span
                            is_still_reasoning = parsed.is_still_reasoning
                            reasoning_text_formatter = (
                                parsed.reasoning_text_formatter
                            )
                            tokens = (
                                reasoning_span.extract_content(response.tokens)
                                or None
                            )
                            if response.log_probabilities is not None:
                                token_log_probs = (
                                    reasoning_span.extract_content(
                                        response.log_probabilities
                                    )
                                    or None
                                )
                            reasoning_tokens = (
                                reasoning_span.extract_reasoning(
                                    response.tokens
                                )
                                or None
                            )

                        if tokens is None and reasoning_tokens is None:
                            # If the status is not done and there were no
                            # tokens, this indicates that the chunk contained
                            # only stripped tokens, such as reasoning
                            # delimiters. In this case, hold off on yielding a
                            # chunk.
                            if response.final_status.is_done:
                                # This terminal chunk may be the only one the
                                # route ever sees (e.g. max_tokens=1 stopping
                                # on a think-start delimiter), so it must carry
                                # the prompt token count or the final usage
                                # chunk reports prompt_tokens=0 (CLIN-1523).
                                # cached_token_count is likewise a one-time
                                # per-request report, so mirror the first-chunk
                                # gating below and only emit it if no earlier
                                # chunk already carried it.
                                #
                                # response.tokens (pre-strip) still holds the
                                # delimiter token(s) that extract_content/
                                # extract_reasoning stripped down to nothing.
                                # They were genuinely generated and consumed
                                # the token budget, so they must still be
                                # billed -- as reasoning tokens, since that's
                                # what they are -- or a max_tokens=1 request
                                # that stops on a bare think-start delimiter
                                # reports completion_tokens=0 (CENG-932).
                                yield TokenGeneratorOutput(
                                    status=response.final_status,
                                    token_count=0,
                                    reasoning_token_count=len(response.tokens)
                                    if response.tokens
                                    else 0,
                                    prompt_token_count=context.tokens.prompt_length,
                                    cached_token_count=response.num_cached_tokens
                                    if not first_chunk_yielded
                                    else None,
                                    batch_id=batch_id,
                                )
                            continue

                        token_count = len(tokens) if tokens is not None else 0
                        reasoning_token_count = (
                            len(reasoning_tokens)
                            if reasoning_tokens is not None
                            else 0
                        )

                        with Tracer(
                            f"tokenizer.decode_chunk({token_count + reasoning_token_count} toks)"
                        ):
                            # Decode tokens using the buffered detokenizer which
                            # handles multi-byte UTF-8 sequences across chunks.
                            decoded_tokens = (
                                await content_detokenizer.decode(tokens)
                                if tokens
                                else None
                            )
                            decoded_reasoning_tokens = (
                                await reasoning_detokenizer.decode(
                                    reasoning_tokens
                                )
                                if reasoning_tokens
                                else None
                            )

                        if (
                            reasoning_text_formatter
                            and decoded_reasoning_tokens
                        ):
                            decoded_reasoning_tokens = reasoning_text_formatter(
                                decoded_reasoning_tokens
                            )

                        # Check for stop sequences if configured (EOSTracker)
                        status = response.final_status
                        stop_sequence_match = None
                        if has_stop_sequences and decoded_tokens is not None:
                            with Tracer("eos_tracker.is_eos_from_string"):
                                if (
                                    stop_sequence_match
                                    := context.eos_tracker.is_eos_from_string(
                                        decoded_tokens
                                    )
                                ):
                                    status = GenerationStatus.END_OF_SEQUENCE
                                    self.model_worker.cancel(request.request_id)

                        # Collect log probability values if present (still
                        # per-token). Does not consider reasoning tokens.
                        token_log_prob_values: list[float] | None = None
                        top_token_log_prob_values: (
                            list[dict[str, float]] | None
                        ) = None
                        if token_log_probs is not None:
                            token_log_prob_values = []
                            top_token_log_prob_values = []
                            for log_prob in token_log_probs:
                                with Tracer("collect_log_probs"):
                                    token_probs = (
                                        log_prob.token_log_probabilities
                                    )
                                    top_probs = await self._top_log_probs(
                                        log_prob, skip_special_tokens
                                    )
                                    token_log_prob_values.extend(token_probs)
                                    top_token_log_prob_values.extend(top_probs)

                        # Record metrics - one TTFT/ITL per chunk
                        is_first_chunk = not first_chunk_yielded
                        if is_first_chunk:
                            METRICS.ttft(ttft_sw.elapsed_ms)
                            decode_sw.reset()
                            first_chunk_yielded = True
                        else:
                            METRICS.itl(itl.elapsed_ms)
                            decode_elapsed_ms = decode_sw.elapsed_ms
                        itl.reset()

                        yield TokenGeneratorOutput(
                            status=status,
                            decoded_tokens=decoded_tokens,
                            decoded_reasoning_tokens=decoded_reasoning_tokens,
                            token_count=token_count,
                            token_log_probabilities=token_log_prob_values,
                            top_log_probabilities=top_token_log_prob_values,
                            prompt_token_count=context.tokens.prompt_length,
                            cached_token_count=response.num_cached_tokens
                            if is_first_chunk
                            else None,
                            reasoning_token_count=reasoning_token_count,
                            stop_sequence=stop_sequence_match,
                            batch_id=batch_id,
                        )
            finally:
                if first_chunk_yielded and num_generated_tokens > 1:
                    METRICS.time_per_output_token(
                        decode_elapsed_ms / (num_generated_tokens - 1)
                    )
                if self.debug_logging:
                    self.logger.debug(
                        "%s: Completed: Elapsed: %0.2f ms",
                        request.request_id,
                        total_sw.elapsed_ms,
                    )

        if self._min_chunk_tokens > 1:
            return _coalesce_chunks(_generate(), self._min_chunk_tokens)
        return _generate()

    async def all_tokens(
        self, request: TextGenerationRequest
    ) -> list[TokenGeneratorOutput]:
        """Generates all token chunks for the provided request."""
        generator = await self.next_token_chunk(request)
        return [chunk async for chunk in generator]

    async def encode(
        self, request: TextGenerationRequest
    ) -> EmbeddingsGenerationOutput:
        """Generates embedded outputs for the provided request."""
        total_sw = StopWatch()
        self.logger.debug(  # noqa: PLE1206 (maybe FIXME)
            "%s [%d]: Started: Elapsed: %0.2f ms",
            request.request_id,
            total_sw.elapsed_ms,
        )

        try:
            with record_ms(METRICS.input_time):
                context = await self.tokenizer.new_context(request)
            _inject_trace_carrier(context)

            with record_ms(METRICS.output_time):
                # For embeddings tasks, the model worker runs an EmbeddingsPipeline which
                # returns EmbeddingsGenerationOutput. The EngineQueue correctly deserializes
                # this based on the model_worker_interface pipeline_task.
                response_stream = await self.model_worker.stream(
                    request.request_id, context
                )
                async for responses, _batch_id in response_stream:
                    for response in responses:
                        # At runtime, response should be EmbeddingsGenerationOutput for embeddings tasks
                        # Cast to handle the generic type parameter mismatch
                        if isinstance(response, EmbeddingsGenerationOutput):
                            return response
                        self.logger.error(
                            f"Unexpected response type for embeddings task: {type(response).__name__}, "
                            f"expected EmbeddingsGenerationOutput. Response: {response}"
                        )
                        raise RuntimeError(
                            f"Expected EmbeddingsGenerationOutput for embeddings task but got "
                            f"{type(response).__name__}. This may indicate a mismatch between "
                            f"the API server pipeline task and the model worker pipeline."
                        )

                raise RuntimeError(
                    f"No embeddings were generated for request {request.request_id}"
                )
        finally:
            if self.debug_logging:
                self.logger.debug(
                    "%s: Completed: Elapsed: %0.2f ms",
                    request.request_id,
                    total_sw.elapsed_ms,
                )
