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
"""Tests for AudioGenerationTokenizer.

The prompt tokenizer is faked, so these run without a checkpoint or network
access: what is under test is how a request becomes an AudioContext, not what
any particular vocabulary encodes a prompt to.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import pytest
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib.audio_tokenizer import AudioGenerationTokenizer
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.modeling.types import RequestID
from max.pipelines.request import OpenResponsesRequest
from max.pipelines.request.open_responses import OpenResponsesRequestBody
from max.pipelines.request.provider_options import (
    AudioProviderOptions,
    ProviderOptions,
)

DEFAULT_DURATION = 45.0
DEFAULT_STEPS = 25
EOS_TOKEN_ID = 2


class _FakeDelegate:
    """A prompt tokenizer, with one token per word and no files to load."""

    def __init__(self) -> None:
        self.eos_token_id: int | None = EOS_TOKEN_ID
        self.encoded: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.encoded.append((text, add_special_tokens))
        return [10 + index for index in range(len(text.split()))]


class _FakeAutoTokenizer:
    """Stands in for transformers' AutoTokenizer in the module under test."""

    def __init__(self) -> None:
        self.delegate = _FakeDelegate()
        self.kwargs: dict[str, Any] | None = None
        self.raise_on_load = False

    def from_pretrained(self, model_path: str, **kwargs: Any) -> _FakeDelegate:
        if self.raise_on_load:
            raise OSError("no such repository")
        self.kwargs = {"model_path": model_path, **kwargs}
        return self.delegate


class _Tokenizer(AudioGenerationTokenizer):
    """An architecture tokenizer: a prompt format, and a negative prompt."""

    def assemble_prompt(self, description: str, lyrics: str | None) -> str:
        if lyrics is None:
            return f"<style> {description}"
        return f"<style> {description} <lyrics> {lyrics}"

    def unconditional_ids(
        self, token_ids: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64] | None:
        return np.zeros_like(token_ids)


class _UnguidedTokenizer(_Tokenizer):
    """An architecture that does not guide, so leaves unconditional_ids alone."""

    def unconditional_ids(
        self, token_ids: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64] | None:
        return super(_Tokenizer, self).unconditional_ids(token_ids)


@pytest.fixture
def auto_tokenizer(monkeypatch: pytest.MonkeyPatch) -> _FakeAutoTokenizer:
    fake = _FakeAutoTokenizer()
    monkeypatch.setattr("max.pipelines.lib.audio_tokenizer.AutoTokenizer", fake)
    return fake


def _build(
    tokenizer_class: type[_Tokenizer] = _Tokenizer,
    **kwargs: Any,
) -> _Tokenizer:
    kwargs.setdefault("default_audio_duration", DEFAULT_DURATION)
    kwargs.setdefault("default_num_inference_steps", DEFAULT_STEPS)
    return tokenizer_class(
        model_path="fake/music-model",
        pipeline_config=cast(PipelineConfig, Mock()),
        **kwargs,
    )


def _request(
    prompt: str = "a slow jazz ballad with upright bass",
    seed: int | None = None,
    **audio_options: Any,
) -> OpenResponsesRequest:
    body = OpenResponsesRequestBody(
        model="fake-music-model",
        input=prompt,
        seed=seed,
        provider_options=ProviderOptions(
            audio=AudioProviderOptions(**audio_options)
        ),
    )
    return OpenResponsesRequest(request_id=RequestID(), body=body)


def test_the_delegate_is_loaded_from_the_named_subfolder(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    tokenizer = _build(
        subfolder="text_encoder_tokenizer",
        revision="abc123",
        trust_remote_code=True,
        max_length=77,
    )

    assert tokenizer.model_path == "fake/music-model"
    assert tokenizer.max_length == 77
    assert auto_tokenizer.kwargs == {
        "model_path": "fake/music-model",
        "subfolder": "text_encoder_tokenizer",
        "revision": "abc123",
        "trust_remote_code": True,
    }


def test_a_tokenizer_that_will_not_load_says_what_to_check(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    auto_tokenizer.raise_on_load = True

    with pytest.raises(ValueError, match="trust-remote-code"):
        _build(subfolder="tokenizer")


def test_eos_token_ids_come_from_the_prompt_tokenizer(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    assert _build().eos_token_ids == {EOS_TOKEN_ID}


def test_a_tokenizer_without_an_eos_token_reports_none(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    auto_tokenizer.delegate.eos_token_id = None

    assert _build().eos_token_ids == set()


def test_audio_requests_carry_a_prompt_not_chat_messages(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    assert _build().expects_content_wrapping is False


@pytest.mark.asyncio
async def test_the_prompt_is_assembled_by_the_architecture(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    """Prompt assembly is the checkpoint's contract, so the subclass owns it."""
    tokenizer = _build()

    await tokenizer.new_context(
        _request(prompt="a slow jazz ballad", lyrics="[verse] hello")
    )

    assert auto_tokenizer.delegate.encoded == [
        ("<style> a slow jazz ballad <lyrics> [verse] hello", False)
    ]


@pytest.mark.asyncio
async def test_a_request_without_lyrics_assembles_without_them(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    tokenizer = _build()

    await tokenizer.new_context(_request(prompt="a slow jazz ballad"))

    assert auto_tokenizer.delegate.encoded == [
        ("<style> a slow jazz ballad", False)
    ]


@pytest.mark.asyncio
async def test_request_options_reach_the_context(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    tokenizer = _build()

    context = await tokenizer.new_context(
        _request(
            seed=7,
            audio_duration=12.5,
            steps=8,
            guidance_scale=3.5,
            audio_format="wav",
        )
    )

    assert context.audio_duration == 12.5
    assert context.num_inference_steps == 8
    assert context.guidance_scale == 3.5
    assert context.audio_format == "wav"
    assert context.seed == 7


@pytest.mark.asyncio
async def test_unset_options_fall_back_to_the_checkpoint_defaults(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    """The architecture's numbers, not any the framework picked."""
    tokenizer = _build()

    context = await tokenizer.new_context(_request())

    assert context.audio_duration == DEFAULT_DURATION
    assert context.num_inference_steps == DEFAULT_STEPS
    # Left to the model, which commonly guides its stages differently.
    assert context.guidance_scale is None


@pytest.mark.asyncio
async def test_the_context_carries_the_request_identity(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    tokenizer = _build()
    request = _request()

    context = await tokenizer.new_context(request)

    assert context.request_id == request.request_id
    assert context.model_name == "fake-music-model"
    # "<style>" plus the seven words of the prompt.
    np.testing.assert_array_equal(
        context.tokens.array, np.array([10, 11, 12, 13, 14, 15, 16, 17])
    )


@pytest.mark.asyncio
async def test_a_guiding_architecture_gets_negative_tokens(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    tokenizer = _build()

    context = await tokenizer.new_context(_request())

    assert context.negative_tokens is not None
    np.testing.assert_array_equal(
        context.negative_tokens.array, np.zeros(8, dtype=np.int64)
    )


@pytest.mark.asyncio
async def test_an_unguided_architecture_has_no_negative_tokens(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    tokenizer = _build(_UnguidedTokenizer)

    context = await tokenizer.new_context(_request())

    assert context.negative_tokens is None


@pytest.mark.asyncio
async def test_an_empty_prompt_is_refused(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    """A generative audio model has nothing to render from silence."""
    tokenizer = _build()

    with pytest.raises(ValueError, match="non-empty description"):
        await tokenizer.new_context(_request(prompt="   "))


@pytest.mark.asyncio
async def test_a_prompt_over_the_limit_is_refused(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    tokenizer = _build(max_length=3)

    with pytest.raises(PromptTooLongError):
        await tokenizer.new_context(_request())


@pytest.mark.asyncio
async def test_a_prompt_at_the_limit_is_accepted(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    # "<style> a slow jazz ballad" is five words, so five tokens.
    tokenizer = _build(max_length=5)

    context = await tokenizer.new_context(_request(prompt="a slow jazz ballad"))

    assert context.tokens.array.size == 5


@pytest.mark.asyncio
async def test_decoding_is_not_offered(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    """Audio generation returns samples, so there is no text to decode."""
    tokenizer = _build()

    with pytest.raises(NotImplementedError):
        await tokenizer.decode(np.array([1, 2, 3], dtype=np.int64))


@pytest.mark.asyncio
async def test_the_base_class_leaves_prompt_assembly_unimplemented(
    auto_tokenizer: _FakeAutoTokenizer,
) -> None:
    tokenizer = _build(cast(Any, AudioGenerationTokenizer))

    with pytest.raises(NotImplementedError):
        await tokenizer.new_context(_request())
