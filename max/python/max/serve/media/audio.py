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
"""WAV encoding for generated audio."""

from __future__ import annotations

import io
import wave

import numpy as np
import numpy.typing as npt

WAV_MEDIA_TYPE = "audio/wav"
"""MIME type of the payload :func:`encode_wav_bytes` returns."""

_INT16_PEAK = 32767
"""Largest magnitude a 16-bit sample can carry."""


def encode_wav_bytes(
    samples: npt.NDArray[np.float32],
    sample_rate: int,
) -> bytes:
    """Encodes a float waveform as a 16-bit PCM WAV payload.

    16-bit is what the format's ubiquitous reader path expects and what every
    client can play; a generative model's own noise floor sits well above the
    quantization step, so the width costs nothing audible.

    Args:
        samples: ``[channels, samples]`` float array with values in
            ``[-1, 1]``. Values outside that range are clipped, which is what
            a decoder would do to them anyway.
        sample_rate: Sample rate of the waveform, in hertz.

    Returns:
        The complete WAV file, header included.

    Raises:
        ValueError: If the waveform is not 2D, has no channels or no samples,
            or the sample rate is not positive.
    """
    if samples.ndim != 2:
        raise ValueError(
            f"Expected a [channels, samples] waveform, got shape "
            f"{samples.shape}."
        )
    channels, frames = samples.shape
    if channels < 1 or frames < 1:
        raise ValueError(
            f"Cannot encode an empty waveform, got shape {samples.shape}."
        )
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}.")

    # WAV interleaves channels per frame, so the transpose is the format's
    # layout rather than a copy for convenience.
    interleaved = np.clip(samples.T, -1.0, 1.0) * _INT16_PEAK
    payload = interleaved.astype("<i2").tobytes()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as container:
        container.setnchannels(channels)
        container.setsampwidth(2)
        container.setframerate(sample_rate)
        container.writeframes(payload)
    return buffer.getvalue()
