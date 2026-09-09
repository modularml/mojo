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

"""Renders a song from a caption and lyrics with MiniMax-Music3.

The song is a JSON file holding the two texts the model conditions on: a
caption describing how the music should sound, and lyrics for it to sing. A
duration and a seed come along so that a run is reproducible.

There are two ways to reach the model and this script does both, because the
request is the same either way. In process, it loads the checkpoint here and
calls the pipeline directly. With ``--server`` it posts to a running
``max serve`` instead, which is worth doing when you plan to render more than
one song: the server keeps its compiled graphs between requests.

Usage:

.. code-block:: bash

    # In process, using the bundled song:
    python generate_song.py --out song.wav

    # Against a running server, for a 30 s excerpt:
    python generate_song.py --out song.wav --server http://localhost:8000 \\
        --duration 30

    # Your own song, and a check of the joins between denoising windows:
    python generate_song.py --song my_song.json --out song.wav --check-seams
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import requests
from max.driver import DeviceSpec
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.audio import AudioGenerationPipeline
from max.pipelines.lib import PipelineArgs
from max.pipelines.modeling.types import (
    AudioGenerationInputs,
    PipelineTask,
    RequestID,
)
from max.pipelines.request import OpenResponsesRequest
from max.pipelines.request.open_responses import OutputAudioContent
from seam_check import report_seams, seam_offsets

DEFAULT_MODEL = "MiniMaxAI/MiniMax-Music3"
DEFAULT_SONG = Path(__file__).resolve().parent / "songs" / "dream_pop.json"
DEFAULT_STEPS = 30

# Minutes of GPU work for a full-length song, and the first request after a
# cold start pays for compiling the graphs on top of that.
SERVER_TIMEOUT_S = 3600.0


@dataclass(frozen=True)
class Song:
    """The two texts the model conditions on, plus how to render them."""

    caption: str
    """Describes the music: genre, tempo, key, voice, arrangement."""

    lyrics: str
    """What the model sings, with ``[verse]``-style tags between sections."""

    duration: float
    """Length of the render in seconds."""

    seed: int
    """Fixes the sampling, so the same song file renders the same audio."""

    @classmethod
    def from_json(cls, path: Path) -> Song:
        """Reads a song file, whose lyrics are a list of lines.

        Args:
            path: JSON file with ``caption``, ``lyrics``, ``duration`` and
                ``seed`` fields.

        Returns:
            The parsed song.
        """
        data = json.loads(path.read_text())
        return cls(
            caption=data["caption"],
            lyrics="\n".join(data["lyrics"]),
            duration=float(data["duration"]),
            seed=int(data["seed"]),
        )


def read_wav(source: io.BytesIO | str) -> tuple[npt.NDArray[np.float32], int]:
    """Reads 16-bit PCM into a ``(channels, samples)`` float array.

    Args:
        source: An open WAV stream, or a path to one.

    Returns:
        The waveform in ``[-1, 1]``, and its sample rate in hertz.
    """
    with wave.open(source, "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    return (pcm.reshape(-1, channels).T / 32768.0).astype(np.float32), rate


def write_wav(path: Path, samples: npt.NDArray[np.float32], rate: int) -> None:
    """Writes a ``(channels, samples)`` float waveform as 16-bit PCM."""
    pcm = (np.clip(samples.T, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(pcm.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


def build_request(
    model: str, song: Song, duration: float, steps: int
) -> OpenResponsesRequest:
    """Builds the OpenResponses request the audio pipeline consumes.

    The caption is the prompt and the lyrics ride in the audio provider
    options, which is the split the API draws between what the model is asked
    for and what a particular modality needs to hear.
    """
    return OpenResponsesRequest.model_validate(
        {
            "request_id": RequestID(),
            "body": {
                "model": model,
                "input": song.caption,
                "seed": song.seed,
                "provider_options": {
                    "audio": {
                        "lyrics": song.lyrics,
                        "audio_duration": duration,
                        "steps": steps,
                    }
                },
            },
        }
    )


def render_in_process(
    model: str, request: OpenResponsesRequest
) -> tuple[npt.NDArray[np.float32], int]:
    """Loads the model here and renders one request.

    Args:
        model: A Hugging Face repository ID or a local checkpoint directory.
        request: The request to render.

    Returns:
        The waveform as ``(channels, samples)``, and its sample rate.
    """
    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(
            model_path=model, device_specs=[DeviceSpec.accelerator()]
        )
    )

    tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(
        config, task=PipelineTask.AUDIO_GENERATION
    )
    assert isinstance(pipeline, AudioGenerationPipeline)

    context = asyncio.run(tokenizer.new_context(request))
    outputs = pipeline.execute(
        AudioGenerationInputs(batch={request.request_id: context})
    )

    audio = outputs[request.request_id].output[0]
    assert isinstance(audio, OutputAudioContent)
    assert audio.samples is not None and audio.sample_rate is not None
    return audio.samples, audio.sample_rate


def served_model(server: str) -> str:
    """Asks a running server which model it is serving."""
    response = requests.get(f"{server}/v1/models", timeout=30.0)
    response.raise_for_status()
    return str(response.json()["data"][0]["id"])


def render_on_server(
    server: str, model: str, song: Song, duration: float, steps: int
) -> tuple[npt.NDArray[np.float32], int]:
    """Renders one song through a running server's speech endpoint.

    ``/v1/audio/speech`` is OpenAI's schema, where ``input`` is the text to
    voice and ``instructions`` is how it should sound -- so for a model that
    sings, the lyrics and the caption swap places relative to the request the
    pipeline takes in process.

    Args:
        server: Base URL of the server, such as ``http://localhost:8000``.
        model: The model ID the server reports.
        song: The song to render.
        duration: Length of the render in seconds.
        steps: Denoising steps per window.

    Returns:
        The waveform as ``(channels, samples)``, and its sample rate.
    """
    response = requests.post(
        f"{server}/v1/audio/speech",
        json={
            "model": model,
            "input": song.lyrics,
            "instructions": song.caption,
            "audio_duration": duration,
            "steps": steps,
            "seed": song.seed,
        },
        timeout=SERVER_TIMEOUT_S,
    )
    response.raise_for_status()
    return read_wav(io.BytesIO(response.content))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--song", type=Path, default=DEFAULT_SONG, help="a song JSON file"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("song.wav"), help="where to write"
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"repository ID or checkpoint path (default {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--server", default=None, help="render through a running max serve"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="overrides the song's"
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--check-seams",
        action="store_true",
        help="measure the joins between denoising windows after rendering",
    )
    parser.add_argument(
        "--analyze",
        type=Path,
        default=None,
        help="check the seams of an existing WAV instead of rendering; "
        "needs the --duration it was rendered at",
    )
    args = parser.parse_args()

    if args.analyze is not None:
        if args.duration is None:
            parser.error(
                "--analyze needs the --duration the WAV was rendered at"
            )
        model = args.model or DEFAULT_MODEL
        samples, rate = read_wav(str(args.analyze))
        offsets = seam_offsets(model, args.duration)
        return 0 if report_seams(samples, rate, offsets) else 1

    song = Song.from_json(args.song)
    if args.seed is not None:
        song = Song(song.caption, song.lyrics, song.duration, args.seed)
    duration = args.duration if args.duration is not None else song.duration

    model = args.model
    if model is None:
        model = served_model(args.server) if args.server else DEFAULT_MODEL

    print(
        f"{args.song.stem} at seed {song.seed}: {duration:.0f} s, "
        f"{args.steps} denoising steps per window, model {model}",
        flush=True,
    )

    started = time.perf_counter()
    if args.server:
        samples, rate = render_on_server(
            args.server, model, song, duration, args.steps
        )
    else:
        samples, rate = render_in_process(
            model, build_request(model, song, duration, args.steps)
        )
    elapsed = time.perf_counter() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(args.out, samples, rate)
    seconds = samples.shape[-1] / rate
    print(
        f"wrote {args.out}: {seconds:.1f} s of audio in {elapsed:.1f} s "
        f"({elapsed / seconds:.2f}x realtime)"
    )

    if args.check_seams:
        return (
            0
            if report_seams(samples, rate, seam_offsets(model, duration))
            else 1
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
