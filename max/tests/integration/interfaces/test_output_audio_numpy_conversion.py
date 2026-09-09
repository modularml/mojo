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
"""Tests for numpy to OutputAudioContent conversion."""

import numpy as np
import pytest
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.context.status import GenerationStatus
from max.pipelines.modeling.types.utils import (
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)
from max.pipelines.request import RequestID
from max.pipelines.request.open_responses import (
    AudioGenerationDetails,
    OutputAudioContent,
    Usage,
)

SAMPLE_RATE = 44100


def _waveform(channels: int = 2, samples: int = 128) -> np.ndarray:
    return np.tile(
        np.linspace(-1.0, 1.0, samples, dtype=np.float32), (channels, 1)
    )


def test_output_audio_from_numpy_samples() -> None:
    waveform = _waveform()
    content = OutputAudioContent.from_numpy_samples(
        waveform, sample_rate=SAMPLE_RATE, format="wav"
    )

    assert content.type == "output_audio"
    assert content.sample_rate == SAMPLE_RATE
    assert content.num_samples == 128
    assert content.format == "wav"
    # The waveform rides raw so that the route can encode it into a
    # container; nothing is base64-encoded at this layer.
    assert content.audio_data is None
    assert content.samples is not None
    np.testing.assert_array_equal(content.samples, waveform)


def test_output_audio_from_numpy_samples_mono() -> None:
    content = OutputAudioContent.from_numpy_samples(
        _waveform(channels=1, samples=64), sample_rate=16000
    )

    assert content.samples is not None
    assert content.samples.shape == (1, 64)
    assert content.num_samples == 64
    assert content.format is None


def test_output_audio_from_numpy_samples_rejects_dtype() -> None:
    with pytest.raises(ValueError, match="float32"):
        OutputAudioContent.from_numpy_samples(
            _waveform().astype(np.float64), sample_rate=SAMPLE_RATE
        )


def test_output_audio_from_numpy_samples_rejects_rank() -> None:
    with pytest.raises(ValueError, match="2D waveform"):
        OutputAudioContent.from_numpy_samples(
            _waveform()[0], sample_rate=SAMPLE_RATE
        )


def test_output_audio_from_numpy_samples_rejects_sample_rate() -> None:
    with pytest.raises(ValueError, match="Sample rate"):
        OutputAudioContent.from_numpy_samples(_waveform(), sample_rate=0)


def test_generation_output_with_audio() -> None:
    request_id = RequestID()
    output = GenerationOutput(
        request_id=request_id,
        final_status=GenerationStatus.END_OF_SEQUENCE,
        output=[
            OutputAudioContent.from_numpy_samples(
                _waveform(), sample_rate=SAMPLE_RATE, format="wav"
            )
        ],
    )

    assert output.is_done
    assert output.request_id == request_id
    assert len(output.output) == 1


def test_generation_output_with_audio_survives_msgpack() -> None:
    """The worker sends responses over ZMQ, so the union has to round trip."""
    waveform = _waveform()
    output = GenerationOutput(
        request_id=RequestID(),
        final_status=GenerationStatus.END_OF_SEQUENCE,
        output=[
            OutputAudioContent.from_numpy_samples(
                waveform, sample_rate=SAMPLE_RATE, format="wav"
            )
        ],
    )

    decoded = msgpack_numpy_decoder(GenerationOutput)(
        msgpack_numpy_encoder()(output)
    )

    content = decoded.output[0]
    assert isinstance(content, OutputAudioContent)
    assert content.sample_rate == SAMPLE_RATE
    assert content.samples is not None
    np.testing.assert_array_equal(content.samples, waveform)


def test_audio_generation_details_describe_the_audio_produced() -> None:
    details = AudioGenerationDetails.from_waveform(
        _waveform(channels=2, samples=SAMPLE_RATE * 2),
        sample_rate=SAMPLE_RATE,
        steps=30,
    )

    assert details.channels == 2
    assert details.num_samples == SAMPLE_RATE * 2
    assert details.sample_rate == SAMPLE_RATE
    assert details.duration_seconds == 2.0
    assert details.steps == 30


def test_audio_generation_details_reject_a_one_dimensional_waveform() -> None:
    with pytest.raises(ValueError, match="waveform, got shape"):
        AudioGenerationDetails.from_waveform(
            _waveform()[0], sample_rate=SAMPLE_RATE, steps=1
        )


def test_audio_generation_details_reject_a_zero_sample_rate() -> None:
    with pytest.raises(ValueError, match="Sample rate"):
        AudioGenerationDetails.from_waveform(
            _waveform(), sample_rate=0, steps=1
        )


def test_audio_usage_survives_msgpack() -> None:
    """The usage a caller is billed on rides the same response over ZMQ."""
    waveform = _waveform()
    usage = Usage(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        audio_generation_details=AudioGenerationDetails.from_waveform(
            waveform, sample_rate=SAMPLE_RATE, steps=30
        ),
    )
    output = GenerationOutput(
        request_id=RequestID(),
        final_status=GenerationStatus.END_OF_SEQUENCE,
        output=[
            OutputAudioContent.from_numpy_samples(
                waveform, sample_rate=SAMPLE_RATE, format="wav"
            )
        ],
        usage=usage,
    )

    decoded = msgpack_numpy_decoder(GenerationOutput)(
        msgpack_numpy_encoder()(output)
    )

    assert decoded.usage == usage
