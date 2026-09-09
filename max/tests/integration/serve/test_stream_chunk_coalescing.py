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

"""Pure unit tests for the streamed-chunk coalescer (``_coalesce_chunks``,
``_merge_outputs``) and the stop-sequence leak fix (``_apply_stop_truncation``)
in ``max.serve.pipelines.llm``.

No server, no model worker -- these exercise the module-level helpers
directly against hand-built ``TokenGeneratorOutput`` sequences.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, AsyncIterator

import pytest
from max.pipelines.context import GenerationStatus
from max.serve.config import Settings
from max.serve.pipelines.llm import (
    TokenGeneratorOutput,
    _apply_stop_truncation,
    _coalesce_chunks,
    _merge_outputs,
)


def _chunk(
    *,
    status: GenerationStatus = GenerationStatus.ACTIVE,
    decoded_tokens: str | None = None,
    decoded_reasoning_tokens: str | None = None,
    token_count: int = 1,
    reasoning_token_count: int | None = None,
    stop_sequence: str | None = None,
    prompt_token_count: int | None = None,
    cached_token_count: int | None = None,
    token_log_probabilities: list[float] | None = None,
    top_log_probabilities: list[dict[str, float]] | None = None,
) -> TokenGeneratorOutput:
    return TokenGeneratorOutput(
        status=status,
        decoded_tokens=decoded_tokens,
        decoded_reasoning_tokens=decoded_reasoning_tokens,
        token_count=token_count,
        token_log_probabilities=token_log_probabilities,
        top_log_probabilities=top_log_probabilities,
        prompt_token_count=prompt_token_count,
        cached_token_count=cached_token_count,
        reasoning_token_count=reasoning_token_count,
        stop_sequence=stop_sequence,
    )


async def _source(
    chunks: list[TokenGeneratorOutput],
) -> AsyncGenerator[TokenGeneratorOutput, None]:
    for chunk in chunks:
        yield chunk


async def _collect(
    gen: AsyncIterator[TokenGeneratorOutput],
) -> list[TokenGeneratorOutput]:
    return [chunk async for chunk in gen]


# --------------------------------------------------------------------------- #
# _merge_outputs
# --------------------------------------------------------------------------- #


def test_merge_outputs_field_semantics() -> None:
    """status/stop_sequence come from the last chunk with a value; the
    first-chunk-only fields (prompt/cached token counts) take the first
    non-None value; text concatenates; counts sum."""
    chunks = [
        _chunk(
            decoded_tokens="hel",
            token_count=2,
            prompt_token_count=10,
            cached_token_count=3,
            reasoning_token_count=1,
            token_log_probabilities=[-0.1, -0.2],
            top_log_probabilities=[{"a": -0.1}, {"b": -0.2}],
        ),
        _chunk(
            decoded_tokens="lo",
            token_count=1,
            reasoning_token_count=2,
            stop_sequence="STOP",
            token_log_probabilities=[-0.3],
            top_log_probabilities=[{"c": -0.3}],
        ),
        _chunk(
            status=GenerationStatus.END_OF_SEQUENCE,
            token_count=0,
        ),
    ]
    merged = _merge_outputs(chunks)
    assert merged.decoded_tokens == "hello"
    assert merged.status == GenerationStatus.END_OF_SEQUENCE
    assert merged.token_count == 3
    assert merged.prompt_token_count == 10
    assert merged.cached_token_count == 3
    assert merged.reasoning_token_count == 3
    assert merged.stop_sequence == "STOP"
    assert merged.token_log_probabilities == [-0.1, -0.2, -0.3]
    assert merged.top_log_probabilities == [
        {"a": -0.1},
        {"b": -0.2},
        {"c": -0.3},
    ]


def test_merge_outputs_all_empty_content_is_none() -> None:
    """When no chunk in the group has content, the merged text is None
    rather than an empty string."""
    chunks = [
        _chunk(decoded_tokens=None, token_count=0),
        _chunk(decoded_tokens=None, token_count=0),
    ]
    merged = _merge_outputs(chunks)
    assert merged.decoded_tokens is None
    assert merged.decoded_reasoning_tokens is None


def test_merge_outputs_reasoning_concatenates_separately() -> None:
    """Reasoning and content text streams merge independently."""
    chunks = [
        _chunk(decoded_reasoning_tokens="think ", token_count=0),
        _chunk(decoded_reasoning_tokens="more", token_count=0),
        _chunk(decoded_tokens="answer", token_count=1),
    ]
    merged = _merge_outputs(chunks)
    assert merged.decoded_reasoning_tokens == "think more"
    assert merged.decoded_tokens == "answer"


# --------------------------------------------------------------------------- #
# _coalesce_chunks
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_coalesce_flushes_at_floor() -> None:
    """With min_chunk_tokens=3, the first chunk flushes immediately (TTFT),
    then subsequent chunks buffer until the 3-token floor is met."""
    chunks = [
        _chunk(decoded_tokens="a", token_count=1),
        _chunk(decoded_tokens="b", token_count=1),
        _chunk(decoded_tokens="c", token_count=1),
        _chunk(decoded_tokens="d", token_count=1),
        _chunk(
            decoded_tokens="e",
            token_count=1,
            status=GenerationStatus.END_OF_SEQUENCE,
        ),
    ]
    out = await _collect(_coalesce_chunks(_source(chunks), 3))
    # chunk 1: "a" (first-visible-chunk early flush)
    # chunks 2-4: "bcd" (floor of 3 reached)
    # chunk 5: "e" (terminal flush, even though only 1 token buffered)
    assert [c.decoded_tokens for c in out] == ["a", "bcd", "e"]
    assert out[-1].status == GenerationStatus.END_OF_SEQUENCE


@pytest.mark.asyncio
async def test_coalesce_never_emits_empty_deltas_alone() -> None:
    """Empty deltas (e.g. stripped special tokens) are buffered until real
    text arrives and never emitted as their own chunk."""
    chunks = [
        _chunk(decoded_tokens=None, token_count=1),  # empty delta
        _chunk(decoded_tokens=None, token_count=1),  # empty delta
        _chunk(decoded_tokens="hello", token_count=1),
    ]
    out = await _collect(_coalesce_chunks(_source(chunks), 5))
    assert len(out) == 1
    assert out[0].decoded_tokens == "hello"
    assert out[0].token_count == 3


@pytest.mark.asyncio
async def test_coalesce_first_visible_chunk_flushes_below_floor() -> None:
    """Even with a large floor, the very first chunk with text flushes
    immediately so TTFT is unaffected."""
    chunks = [
        _chunk(decoded_tokens="first", token_count=1),
    ]
    out = await _collect(_coalesce_chunks(_source(chunks), 100))
    assert len(out) == 1
    assert out[0].decoded_tokens == "first"


@pytest.mark.asyncio
async def test_coalesce_terminal_chunk_always_flushes_trailing_text() -> None:
    """A terminal (is_done) chunk flushes immediately even if the buffered
    token count is below the floor, so trailing text is never dropped."""
    chunks = [
        _chunk(decoded_tokens="first", token_count=1),
        _chunk(
            decoded_tokens="tail",
            token_count=1,
            status=GenerationStatus.MAXIMUM_LENGTH,
        ),
    ]
    out = await _collect(_coalesce_chunks(_source(chunks), 10))
    assert [c.decoded_tokens for c in out] == ["first", "tail"]
    assert out[-1].status == GenerationStatus.MAXIMUM_LENGTH


@pytest.mark.asyncio
async def test_coalesce_stop_sequence_flushes_and_terminates() -> None:
    """A stop-sequence match sets status to END_OF_SEQUENCE, which forces an
    immediate flush (must_flush), and the coalescer applies stop truncation
    to the emitted chunk."""
    chunks = [
        _chunk(decoded_tokens="first", token_count=1),
        _chunk(
            decoded_tokens="hello STOP world",
            token_count=1,
            status=GenerationStatus.END_OF_SEQUENCE,
            stop_sequence="STOP",
        ),
    ]
    out = await _collect(_coalesce_chunks(_source(chunks), 10))
    assert len(out) == 2
    assert out[0].decoded_tokens == "first"
    assert out[1].decoded_tokens == "hello "
    assert "STOP" not in (out[1].decoded_tokens or "")
    assert out[1].status == GenerationStatus.END_OF_SEQUENCE


@pytest.mark.asyncio
async def test_coalesce_reasoning_stream_coalesces() -> None:
    """Reasoning-only chunks coalesce the same way as content chunks, using
    the combined content+reasoning token count against the floor. The first
    visible chunk (reasoning counts as visible text) still flushes early for
    TTFT; subsequent reasoning chunks buffer to the floor."""
    chunks = [
        _chunk(
            decoded_reasoning_tokens="think",
            token_count=0,
            reasoning_token_count=1,
        ),
        _chunk(
            decoded_reasoning_tokens=" more",
            token_count=0,
            reasoning_token_count=1,
        ),
        _chunk(
            decoded_reasoning_tokens=" done",
            token_count=0,
            reasoning_token_count=1,
            status=GenerationStatus.END_OF_SEQUENCE,
        ),
    ]
    out = await _collect(_coalesce_chunks(_source(chunks), 3))
    assert [c.decoded_reasoning_tokens for c in out] == ["think", " more done"]


@pytest.mark.asyncio
async def test_coalesce_leftover_buffer_flushed_at_stream_end() -> None:
    """If the source ends without a terminal-status chunk, any remaining
    buffered content is still flushed (defensive; the real generator always
    terminates with a done status, but the coalescer must not drop text)."""
    chunks = [
        _chunk(decoded_tokens="first", token_count=1),
        _chunk(decoded_tokens="stuck", token_count=1),
    ]
    out = await _collect(_coalesce_chunks(_source(chunks), 10))
    assert [c.decoded_tokens for c in out] == ["first", "stuck"]


# --------------------------------------------------------------------------- #
# _apply_stop_truncation (Part B: stop-leak fix)
# --------------------------------------------------------------------------- #


def test_stop_truncation_drops_stop_and_after() -> None:
    merged = _chunk(decoded_tokens="hello STOP world", stop_sequence="STOP")
    truncated = _apply_stop_truncation(merged)
    assert truncated.decoded_tokens == "hello "
    assert "STOP" not in truncated.decoded_tokens
    assert "world" not in truncated.decoded_tokens


def test_stop_truncation_noop_when_absent() -> None:
    """A straddled stop string (idx < 0, the accepted limitation) leaves the
    chunk unchanged rather than mangling it."""
    merged = _chunk(decoded_tokens="abc", stop_sequence="STOP")
    unchanged = _apply_stop_truncation(merged)
    assert unchanged.decoded_tokens == "abc"


def test_stop_truncation_noop_without_stop_sequence() -> None:
    merged = _chunk(decoded_tokens="abc", stop_sequence=None)
    unchanged = _apply_stop_truncation(merged)
    assert unchanged.decoded_tokens == "abc"


def test_stop_truncation_never_touches_reasoning_text() -> None:
    """Stop matching is content-only -- reasoning text is never truncated."""
    merged = _chunk(
        decoded_tokens="hello STOP world",
        decoded_reasoning_tokens="thinking STOP still",
        stop_sequence="STOP",
    )
    truncated = _apply_stop_truncation(merged)
    assert truncated.decoded_tokens == "hello "
    assert truncated.decoded_reasoning_tokens == "thinking STOP still"


# --------------------------------------------------------------------------- #
# Settings.stream_min_chunk_tokens
# --------------------------------------------------------------------------- #


def test_settings_stream_min_chunk_tokens_default_is_one() -> None:
    if "MAX_SERVE_STREAM_MIN_CHUNK_TOKENS" in os.environ:
        del os.environ["MAX_SERVE_STREAM_MIN_CHUNK_TOKENS"]
    settings = Settings()
    assert settings.stream_min_chunk_tokens == 1


def test_settings_stream_min_chunk_tokens_env_override() -> None:
    os.environ["MAX_SERVE_STREAM_MIN_CHUNK_TOKENS"] = "8"
    try:
        settings = Settings()
        assert settings.stream_min_chunk_tokens == 8
    finally:
        del os.environ["MAX_SERVE_STREAM_MIN_CHUNK_TOKENS"]
