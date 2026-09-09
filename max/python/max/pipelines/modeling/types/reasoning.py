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

"""Interfaces for interacting with the reasoning section of model output."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

from max.pipelines.request import RequestType

from .tokenizer import PipelineTokenizer, TokenizerEncoded, UnboundContextType

_T = TypeVar("_T")


@runtime_checkable
class ReasoningPipelineTokenizer(
    PipelineTokenizer[UnboundContextType, TokenizerEncoded, RequestType],
    Protocol[UnboundContextType, TokenizerEncoded, RequestType],
):
    """:class:`PipelineTokenizer` that exposes its reasoning-delimiter token ids.

    Implemented by architecture-specific tokenizers that drive a reasoning
    parser (Gemma 4, Kimi K2.5, MiniMax M2). The tokenizer resolves the
    delimiter ids once at construction and exposes them as instance
    attributes so callers — for example
    :class:`~max.pipelines.lib.pipeline_variants.overlap_text_generation.OverlapTextGenerationPipeline`'s
    thinking-mode temperature scaling — can read them directly without
    re-encoding ``<think>``/``</think>`` or depending on the reasoning
    parser registry.
    """

    @property
    def reasoning_start_token_id(self) -> int:
        """The token id that opens a reasoning span (e.g. ``<|channel>``)."""
        ...

    @property
    def reasoning_end_token_id(self) -> int:
        """The token id that closes a reasoning span (e.g. ``<channel|>``)."""
        ...


class ReasoningSpan:
    """Identifies a reasoning span within a token ID sequence.

    A reasoning span is a contiguous span of tokens near the start of a reasoning
    model's output. In streaming mode with multiple chunks, the reasoning section
    may consist of one or more reasoning spans across multiple initial chunks.

    Tracks both the delimited reasoning span (including delimiter tokens like
    ``<think>`` and ``</think>``) and the reasoning span (excluding
    delimiters). Uses standard Python slice semantics: ``[start, end)``.

    Args:
        reasoning_with_delimiters: The full span including delimiter tokens.
        reasoning: The span excluding delimiter tokens. Must be contained
            within ``reasoning_with_delimiters``.
    """

    def __init__(
        self,
        reasoning_with_delimiters: tuple[int, int],
        reasoning: tuple[int, int],
    ) -> None:
        delimited_start, delimited_end = reasoning_with_delimiters
        reasoning_start, reasoning_end = reasoning
        assert delimited_start <= delimited_end
        assert reasoning_start <= reasoning_end
        assert delimited_start <= reasoning_start
        assert delimited_end >= reasoning_end
        self._reasoning_with_delimiters = reasoning_with_delimiters
        self._reasoning = reasoning

    def extract_content(self, seq: Sequence[_T]) -> list[_T]:
        """Extracts the non-reasoning elements from a sequence.

        Args:
            seq: The sequence from which to extract non-reasoning elements.

        Returns:
            The elements outside the delimited reasoning span.
        """
        delimited_start, delimited_end = self._reasoning_with_delimiters
        return list(seq[:delimited_start]) + list(seq[delimited_end:])

    def extract_reasoning(self, seq: Sequence[_T]) -> list[_T]:
        """Extracts the reasoning elements from a sequence.

        Args:
            seq: The sequence from which to extract reasoning elements.

        Returns:
            The elements within the reasoning span, excluding delimiters.
        """
        reasoning_start, reasoning_end = self._reasoning
        return list(seq[reasoning_start:reasoning_end])


@dataclass(frozen=True)
class ParsedReasoningDelta:
    """Result of applying reasoning parsing to a streaming delta chunk."""

    span: ReasoningSpan
    """The ReasoningSpan identifying the reasoning portion of the chunk."""
    is_still_reasoning: bool
    """Whether the reasoning section is still active."""
    reasoning_text_formatter: Callable[[str], str | None] | None = field(
        default=None
    )
    """Optional callback to post-process decoded reasoning text.

    Returns the formatted text, or ``None`` if the text should be ignored.
    """


class ReasoningParser(ABC):
    """Parser for identifying reasoning spans in model output."""

    REASONING_START: ClassVar[str | None] = None
    """Text delimiter that opens a reasoning span (e.g. ``"<think>"``).

    Subclasses declare their delimiters here and resolve token ids from them,
    so each model's delimiters are written down exactly once. Consumers that
    work in the text domain rather than the token domain read them from here
    instead of restating them.

    Declare both delimiters or neither. ``None`` means this parser has no text
    form at all, so a text-domain consumer cannot bound a reasoning span and
    will leave reasoning in the assistant's content; declaring only one is a
    bug, since a span needs both ends. A parser whose chat template prefills
    the opening delimiter still declares it -- whether a given turn emits it is
    a property of the request, not of the parser.
    """

    REASONING_END: ClassVar[str | None] = None
    """Text delimiter that closes a reasoning span (e.g. ``"</think>"``).

    See :attr:`REASONING_START` for the declaration contract.
    """

    @abstractmethod
    def stream(
        self,
        delta_token_ids: Sequence[int],
        is_currently_reasoning: bool = True,
    ) -> ParsedReasoningDelta:
        r"""Identifies a reasoning span within a streaming delta chunk.

        Args:
            delta_token_ids: The token IDs of the incremental streaming chunk.
            is_currently_reasoning: Whether the stream was already inside a
                reasoning span at the start of this chunk. When ``True``
                (the default, for backward compatibility), the parser
                treats the chunk as continuing reasoning unless/until it
                finds an end delimiter. When ``False``, the parser only
                enters reasoning if it actually finds a start delimiter in
                this chunk — letting callers feed every chunk through and
                catch mid-stream reasoning sections (e.g. Gemma 4 emitting
                ``<|channel>thought\n...<channel|>`` even when reasoning
                wasn't pre-seeded).

        Returns:
            A :class:`ParsedReasoningDelta` containing the reasoning span,
            whether reasoning is still active, and an optional formatter for
            decoded reasoning text.
        """
        ...

    def will_reason_after_prompt(
        self,
        prompt_token_ids: Sequence[int],
    ) -> bool:
        """Predicts whether the model will emit reasoning after this prompt.

        Called once at turn initiation to seed the streaming reasoning
        state machine and decide whether grammar enforcement should be
        suspended for the first generated tokens.

        The default implementation delegates to :meth:`stream`, which
        scans left-to-right and returns ``is_still_reasoning``.
        Architectures should override this when they have a more
        reliable signal (e.g., a dedicated think-enable token).

        Args:
            prompt_token_ids: The full prompt token id sequence.

        Returns:
            ``True`` if the model will start with reasoning tokens;
            ``False`` otherwise.
        """
        return self.stream(prompt_token_ids).is_still_reasoning

    def reset(self) -> None:
        """Resets per-request state.

        Called at the start of each request to clear any internal state
        accumulated during a prior request.
        """
        return

    @classmethod
    @abstractmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> ReasoningParser:
        """Constructs a reasoning parser from a tokenizer.

        Args:
            tokenizer: The :class:`~max.pipelines.modeling.types.PipelineTokenizer` to use
                for resolving reasoning delimiter token IDs.

        Returns:
            A new :class:`ReasoningParser` instance.
        """
        ...

    @classmethod
    @abstractmethod
    async def reasoning_end_token_id(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> int | None:
        """Returns the single-token ID that closes a reasoning span.

        Used by callers that need to detect end-of-reasoning without
        instantiating the full parser (e.g., grammar-region setup in the
        tokenizer). Implementations should resolve their architecture's
        end-marker string (``</think>``, ``<channel|>``, etc.) via
        :func:`max.pipelines.lib.tokenizer.convert_token_to_id`.

        Args:
            tokenizer: The :class:`~max.pipelines.modeling.types.PipelineTokenizer` used
                for token-id resolution.

        Returns:
            The token ID that marks end-of-reasoning, or ``None`` if the
            architecture's end marker doesn't tokenize to a single ID.
        """
        ...
