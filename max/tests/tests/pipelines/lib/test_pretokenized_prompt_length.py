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
"""A pre-tokenized prompt must be length-checked like a string one.

``encode`` used to bound only the ``str`` arm, so a ``Sequence[int]`` prompt --
what ``openai_routes`` forwards for a ``/v1/completions`` token array, or
whenever the orchestrator sets ``prompt_tokens`` for KV cache-aware routing --
was admitted at any length. ``new_context`` then derives
``max_length = len(token_ids) + max_tokens_to_generate(...)``, and
``max_tokens_to_generate`` floors the remaining budget at zero, so an
over-length prompt yields a context whose ``max_length`` *is* the prompt
length. The speculative-decoding response path refuses such a context via
``upper_bounded_default`` rather than clamping it, which killed the model
worker on a request that should have been a 400.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib.pipeline_variants.utils import build_response
from max.pipelines.lib.tokenizer import TextTokenizer
from max.pipelines.modeling.types import RequestID, TextGenerationRequest

MAX_LENGTH = 4096


class _StubDelegate:
    """The slice of a HuggingFace tokenizer these paths actually touch.

    A pre-tokenized prompt never reaches the delegate; it is needed for
    ``tokenizer_vocab_size`` and to encode a string prompt.
    """

    def __len__(self) -> int:
        return 128256

    def encode(self, prompt: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) % 128 for c in prompt]


def _text_tokenizer(max_length: int | None = MAX_LENGTH) -> TextTokenizer:
    """Builds a ``TextTokenizer`` without loading a HuggingFace checkpoint.

    ``__init__`` downloads a tokenizer; the paths under test read only the
    attributes set here.
    """
    tokenizer = TextTokenizer.__new__(TextTokenizer)
    tokenizer.delegate = _StubDelegate()
    tokenizer.max_length = max_length
    tokenizer._eos_token_ids = {0}
    return tokenizer


def _request(token_ids: list[int]) -> TextGenerationRequest:
    """A chat request carrying the orchestrator's pre-tokenized prompt.

    Mirrors ``openai_routes`` when ``prompt_tokens`` is set: the token ids go
    in ``prompt`` and ``messages`` is left empty.
    """
    return TextGenerationRequest(
        request_id=RequestID(),
        model_name="test-model",
        prompt=token_ids,
        messages=[],
    )


def _encode(tokenizer: Any, prompt: Any) -> npt.NDArray[np.integer[Any]]:
    return asyncio.run(tokenizer.encode(prompt))


class TestPretokenizedPromptIsBounded:
    """``encode`` bounds a token-id prompt exactly as it bounds a string."""

    def test_over_length_token_ids_are_rejected(self) -> None:
        prompt = list(range(MAX_LENGTH + 1))

        with pytest.raises(PromptTooLongError) as excinfo:
            _encode(_text_tokenizer(), prompt)

        assert excinfo.value.num_tokens == MAX_LENGTH + 1
        assert excinfo.value.max_length == MAX_LENGTH

    def test_token_ids_at_the_limit_are_accepted_unchanged(self) -> None:
        prompt = list(range(MAX_LENGTH))

        encoded = _encode(_text_tokenizer(), prompt)

        assert np.array_equal(encoded, np.array(prompt))

    def test_string_and_token_id_prompts_agree_at_the_boundary(self) -> None:
        """The two arms must not disagree about what is admissible."""
        tokenizer = _text_tokenizer()

        with pytest.raises(PromptTooLongError) as from_text:
            _encode(tokenizer, "a" * (MAX_LENGTH + 1))
        with pytest.raises(PromptTooLongError) as from_ids:
            _encode(tokenizer, list(range(MAX_LENGTH + 1)))

        assert from_text.value.num_tokens == from_ids.value.num_tokens
        assert from_text.value.max_length == from_ids.value.max_length

    def test_unbounded_tokenizer_still_accepts_any_length(self) -> None:
        """``max_length=None`` means no bound, for either prompt shape."""
        prompt = list(range(MAX_LENGTH * 2))

        encoded = _encode(_text_tokenizer(max_length=None), prompt)

        assert np.array_equal(encoded, np.array(prompt))


class TestAdmittedContextNeverBreaksBuildResponse:
    """The invariant the crashloop violated.

    Every context ``new_context`` hands back must satisfy
    ``context.max_length <= max_seq_len``, because the response path raises --
    and takes the model worker with it -- when it does not.
    """

    @pytest.mark.parametrize(
        "prompt_len",
        [1, MAX_LENGTH - 1, MAX_LENGTH, MAX_LENGTH + 1, MAX_LENGTH * 2],
    )
    def test_admitted_context_max_length_is_within_the_model_bound(
        self, prompt_len: int
    ) -> None:
        tokenizer = _text_tokenizer()

        try:
            context = asyncio.run(
                tokenizer.new_context(_request(list(range(prompt_len))))
            )
        except PromptTooLongError:
            assert prompt_len > MAX_LENGTH
            return

        assert context.max_length == MAX_LENGTH
        build_response([context], max_seq_len=MAX_LENGTH)

    def test_over_length_context_would_kill_build_response(self) -> None:
        """Pins the consequence, so the admission guard cannot be dropped.

        This is the crash the stage-1 shadow deployment hit: a context whose
        ``max_length`` exceeds the model bound makes the response path raise
        instead of clamping.
        """
        over_length = MAX_LENGTH + 1
        context = TextContext(
            request_id=RequestID(),
            max_length=over_length,
            tokens=TokenBuffer(np.arange(over_length, dtype=np.int64)),
        )

        with pytest.raises(ValueError) as excinfo:
            build_response([context], max_seq_len=MAX_LENGTH)

        assert str(excinfo.value) == (
            f"default value provided ({over_length}) exceeds the upper bound "
            f"({MAX_LENGTH})"
        )
