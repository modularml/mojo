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
"""Tests for how AudioGenerationPipeline drives an executor and shapes responses.

The executor here is a stand-in, so these cover the pipeline's own contract --
one waveform per request, in the batch's order, at the executor's sample rate
-- without compiling a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from max.pipelines.audio import AudioGenerationPipeline
from max.pipelines.context import AudioContext, TokenBuffer
from max.pipelines.modeling.types import AudioGenerationInputs
from max.pipelines.request.open_responses import OutputAudioContent

SAMPLE_RATE = 44100


@dataclass
class _FakeOutputs:
    waveform: npt.NDArray[np.float32]


class _FakeExecutor:
    """Returns a fixed waveform, and records what it was asked to prepare."""

    def __init__(self, waveform: npt.NDArray[np.float32]) -> None:
        self._waveform = waveform
        self.prepared: list[AudioContext] = []

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def prepare_inputs(self, contexts: list[AudioContext]) -> Any:
        self.prepared = contexts
        return contexts

    def execute(self, inputs: Any) -> _FakeOutputs:
        return _FakeOutputs(waveform=self._waveform)


def _pipeline(
    waveform: npt.NDArray[np.float32],
) -> tuple[AudioGenerationPipeline[AudioContext], _FakeExecutor]:
    """Builds the pipeline around a fake executor, skipping compilation.

    ``__init__`` loads devices and compiles the architecture, neither of which
    these tests are about, so the executor is installed directly.
    """
    pipeline: AudioGenerationPipeline[AudioContext] = object.__new__(
        AudioGenerationPipeline
    )
    executor = _FakeExecutor(waveform)
    pipeline._executor = executor
    return pipeline, executor


def _context(**kwargs: Any) -> AudioContext:
    """A context carrying the numbers a tokenizer would have resolved."""
    kwargs.setdefault("audio_duration", 10.0)
    kwargs.setdefault("num_inference_steps", 4)
    return AudioContext(
        tokens=TokenBuffer(array=np.arange(8, dtype=np.int64)), **kwargs
    )


def _waveform(
    channels: int = 2, samples: int = 64, batch: int = 1
) -> npt.NDArray[np.float32]:
    one = np.tile(
        np.linspace(-1.0, 1.0, samples, dtype=np.float32), (channels, 1)
    )
    return np.stack([one] * batch)


def test_execute_returns_one_audio_response() -> None:
    waveform = _waveform()
    pipeline, executor = _pipeline(waveform)
    context = _context(audio_format="wav")

    responses = pipeline.execute(
        AudioGenerationInputs(batch={context.request_id: context})
    )

    assert list(responses) == [context.request_id]
    response = responses[context.request_id]
    assert response.is_done
    # Audio has no tokens to count, so usage reports what was generated in
    # the audio details and leaves the token counters at zero.
    assert response.usage is not None
    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0
    assert response.usage.total_tokens == 0
    details = response.usage.audio_generation_details
    assert details is not None
    assert details.num_samples == 64
    assert details.channels == 2
    assert details.sample_rate == SAMPLE_RATE
    assert details.steps == context.num_inference_steps

    content = response.output[0]
    assert isinstance(content, OutputAudioContent)
    assert content.sample_rate == SAMPLE_RATE
    assert content.format == "wav"
    assert content.samples is not None
    np.testing.assert_array_equal(content.samples, waveform[0])

    # The context reached the executor rather than being reshaped en route.
    assert executor.prepared == [context]


def test_execute_of_an_empty_batch_does_nothing() -> None:
    pipeline, _ = _pipeline(_waveform())

    assert pipeline.execute(AudioGenerationInputs(batch={})) == {}


def test_execute_rejects_a_mismatched_waveform_count() -> None:
    pipeline, _ = _pipeline(_waveform(batch=2))
    context = _context()

    with pytest.raises(ValueError, match="expected 1, got 2"):
        pipeline.execute(
            AudioGenerationInputs(batch={context.request_id: context})
        )


def test_execute_rejects_a_waveform_without_a_batch_axis() -> None:
    pipeline, _ = _pipeline(_waveform()[0])
    context = _context()

    with pytest.raises(ValueError, match="channels, samples"):
        pipeline.execute(
            AudioGenerationInputs(batch={context.request_id: context})
        )


def test_prepare_batch_rejects_more_than_one_request() -> None:
    pipeline, _ = _pipeline(_waveform())
    first, second = _context(), _context()

    with pytest.raises(ValueError, match="not supported yet"):
        pipeline.prepare_batch(
            {first.request_id: first, second.request_id: second}
        )


def test_max_batch_size_is_one() -> None:
    pipeline, _ = _pipeline(_waveform())

    assert pipeline.max_batch_size == 1
