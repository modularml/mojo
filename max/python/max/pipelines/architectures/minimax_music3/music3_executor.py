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
"""The executor: a prompt in, a waveform out, in three stages.

MiniMax Music 3 is four networks in a row. An autoregressive pair -- a 17 GiB
Qwen3 and a small depth decoder -- turns the prompt into per-frame hidden states
at 25 Hz; a flow-matching DiT turns those into latents at ~86 Hz; a convolutional
decoder turns those into 44.1 kHz stereo. Nothing about that is unusual.

What is unusual is that the four do not fit on the device at once. In bfloat16
their weights come to about 23.6 GiB against an A10G's 22.5, and the measured
autoregressive pair alone is 17.96 GiB resident with 4.53 free -- which the DiT
cannot join. So this executor is written around dropping each stage before
building the next (M0d: a dropped ``Model`` returns its weight memory to MAX's
pool, and the next stage allocates out of that pool). The cost is that staged
mode rebuilds every stage on every request: minutes on a cold compilation cache,
and measured at 77 s for a 12 s clip once it is warm, against 48.6 s for the
reference. It is what makes the pipeline run at all on a 22 GiB card, and it is
skipped entirely on a device that can hold everything.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from max.driver import Accelerator, Device, load_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.tensor import Tensor
from max.pipelines.audio import AudioExecutorOutputs
from max.pipelines.context import AudioContext
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_executor import PipelineExecutor
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.modeling.base import TensorStruct
from max.profiler import traced

from . import denoise
from .autoregressive import (
    DepthDecoderStage,
    Generator,
    LanguageModelStage,
)
from .checkpoint import Checkpoint
from .diffusion import DiffusionStage
from .vocode import ChunkedVocoder
from .weight_adapters import convert_vocoder_state, embed_text

logger = logging.getLogger("max.pipelines")

DTYPE = DType.bfloat16
"""The port's only supported encoding. float32 does not fit any stage but the
vocoder, and 8-bit weights were never gated against the reference."""

# Headroom for activations on top of the weights, measured: the DiT peaks at 5.7
# GiB on a 689-latent window and the decoder at 4.4 on a 160-frame one. A device
# that cannot hold every component's weights plus this much has to stage.
ACTIVATION_HEADROOM = 6 * 2**30


@dataclass(frozen=True)
class MiniMaxMusic3Inputs(TensorStruct):
    """One request, stored as tensors.

    The prompt arrives already embedded rather than as token ids, which keeps
    the model's 200000-row embedding table off the device. See
    :func:`embed_text`.
    """

    prompt: Tensor
    """The conditional prompt embeddings and the classifier-free guidance
    prompt embeddings, laid end to end. Shape is
    ``(2 * prompt_length, hidden_size)``.
    """

    prompt_length: Tensor
    """The number of tokens in each prompt. A 1-element int64 tensor. Both
    rows share one length."""

    max_frames: Tensor
    """Upper bound on generated frames, 1-element int64. The model may stop
    earlier, and usually does."""

    num_inference_steps: Tensor
    """Euler steps per denoising window, 1-element int64."""

    guidance_scale: Tensor
    """Classifier-free guidance scale for the denoiser, 1-element float32."""

    seed: Tensor
    """RNG seed for both the code sampling and the denoising noise,
    1-element int64."""


def _scalar(value: int) -> Tensor:
    return Tensor.from_dlpack(np.array([value], dtype=np.int64))


def _scalar_float(value: float) -> Tensor:
    return Tensor.from_dlpack(np.array([value], dtype=np.float32))


def _read(tensor: Tensor) -> int:
    return int(tensor.to_numpy().reshape(-1)[0])


def _read_float(tensor: Tensor) -> float:
    return float(tensor.to_numpy().reshape(-1)[0])


class MiniMaxMusic3Executor(
    PipelineExecutor[AudioContext, MiniMaxMusic3Inputs, AudioExecutorOutputs]
):
    """Turns a text prompt and lyrics into music, one request at a time."""

    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
        runtime_config: PipelineRuntimeConfig,
    ) -> None:
        self._checkpoint = Checkpoint(manifest)
        self.config = self._checkpoint.config
        self._session = session
        self._runtime_config = runtime_config

        device = load_devices(manifest["transformer"].device_specs)[0]
        if not isinstance(device, Accelerator):
            raise ValueError(
                "MiniMax Music 3 needs a GPU: its autoregressive half alone is "
                "18 GiB of weights and 300 sequential frame steps."
            )
        self._device: Accelerator = device

        self._staged = self._must_stage(self._checkpoint, device)
        self._language: LanguageModelStage | None = None
        self._generator: Generator | None = None
        self._diffusion: DiffusionStage | None = None
        self._vocoder: ChunkedVocoder | None = None

    @staticmethod
    def _must_stage(checkpoint: Checkpoint, device: Device) -> bool:
        """Whether the components have to take turns on this device."""
        weights = checkpoint.weight_bytes()
        # A card that reports no free memory is reporting nothing, not a full
        # card, so fall back to its capacity as the memory estimator does.
        free = int(device.stats["free_memory"]) or int(
            device.stats["total_memory"]
        )
        staged = weights + ACTIVATION_HEADROOM > free
        logger.info(
            "MiniMax Music 3: %.1f GiB of weights against %.1f GiB free, "
            "%s residency.",
            weights / 2**30,
            free / 2**30,
            "staged" if staged else "whole-model",
        )
        if staged:
            logger.warning(
                "MiniMax Music 3 is staging its components, which rebuilds "
                "each stage per request. The first generation on this device "
                "spends minutes compiling; later ones replay that from the "
                "compilation cache."
            )
        return staged

    @property
    def sample_rate(self) -> int:
        """The vocoder's sample rate of 44100 Hz."""
        return self.config.sampling_rate

    # -- PipelineExecutor interface ------------------------------------------

    @traced(message="MiniMaxMusic3Executor.prepare_inputs")
    def prepare_inputs(
        self, contexts: list[AudioContext]
    ) -> MiniMaxMusic3Inputs:
        """Embeds the prompt pair and settles the generation's size.

        Raises:
            ValueError: If the batch is not exactly one request, or if the
                context carries no guidance prompt.
        """
        if len(contexts) != 1:
            raise ValueError(
                "MiniMax Music 3 generates one request at a time, got "
                f"{len(contexts)}."
            )
        context = contexts[0]
        if context.negative_tokens is None:
            raise ValueError(
                "MiniMax Music 3 always generates with classifier-free "
                "guidance, so the request needs the masked prompt its "
                "tokenizer produces alongside the real one."
            )

        conditional = np.asarray(context.tokens.all, dtype=np.int64)
        unconditional = np.asarray(context.negative_tokens.all, dtype=np.int64)
        if conditional.shape != unconditional.shape:
            raise ValueError(
                "the guided prompts must be the same length, got "
                f"{conditional.shape} and {unconditional.shape}"
            )
        prompt_length = int(conditional.shape[-1])

        return MiniMaxMusic3Inputs(
            prompt=embed_text(
                self._checkpoint.weights("language_model"),
                np.stack([conditional, unconditional]),
                DTYPE,
            ),
            prompt_length=_scalar(prompt_length),
            max_frames=_scalar(self._frame_budget(context, prompt_length)),
            num_inference_steps=_scalar(context.num_inference_steps),
            guidance_scale=_scalar_float(
                context.guidance_scale
                if context.guidance_scale is not None
                else denoise.GUIDANCE_SCALE
            ),
            seed=_scalar(
                context.seed if context.seed is not None else 0,
            ),
        )

    @traced(message="MiniMaxMusic3Executor.execute")
    def execute(self, inputs: MiniMaxMusic3Inputs) -> AudioExecutorOutputs:
        """Runs the language model, diffusion, and vocoder stages, returning
        the finished stereo waveform.
        """
        seed = _read(inputs.seed)
        conditioning = self._generate(inputs, seed)
        chunks = self._denoise(
            conditioning,
            _read(inputs.num_inference_steps),
            seed,
            _read_float(inputs.guidance_scale),
        )
        waveform = self._vocode(chunks)
        # ``(1, channels, samples)``: one request's audio, as a batch of one.
        return AudioExecutorOutputs(waveform=Tensor.from_dlpack(waveform[None]))

    # -- the three stages ----------------------------------------------------

    @traced(message="MiniMaxMusic3Executor.generate")
    def _generate(
        self, inputs: MiniMaxMusic3Inputs, seed: int
    ) -> npt.NDArray[np.float32]:
        """Sample frames autoregressively, returning their conditioning."""
        generator = self._autoregressive()
        generation = generator.generate(
            inputs.prompt.to(self._device),
            _read(inputs.prompt_length),
            _read(inputs.max_frames),
            rng=np.random.default_rng(seed),
        )
        # ``(frames, width)`` from the loop, and the condition encoder wants
        # ``(batch, frames, width)``.
        return np.ascontiguousarray(
            generation.conditioning[None], dtype=np.float32
        )

    @traced(message="MiniMaxMusic3Executor.denoise")
    def _denoise(
        self,
        conditioning: npt.NDArray[np.float32],
        num_steps: int,
        seed: int,
        guidance_scale: float,
    ) -> list[npt.NDArray[np.float32]]:
        """Flow-match every window of the frame timeline into latents."""
        stage = self._diffusion_stage()
        windows, conditions = stage.conditions(conditioning)
        channels = self.config.transformer.in_channels
        # Drawn here rather than inside the loop so that a window's noise
        # depends only on the seed and the plan, not on how many Euler steps
        # ran before it.
        draw = np.random.default_rng(seed)
        noise = [
            draw.standard_normal(
                (1, channels, window.latents), dtype=np.float32
            )
            for window in windows
        ]
        return stage.denoise(
            windows,
            conditions,
            noise,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )

    @traced(message="MiniMaxMusic3Executor.vocode")
    def _vocode(
        self, chunks: list[npt.NDArray[np.float32]]
    ) -> npt.NDArray[np.float32]:
        """Decode each window and crop the overlaps away."""
        vocoder = self._vocoder_stage()
        waveforms = [vocoder.decode(chunk) for chunk in chunks]
        return denoise.crop(waveforms, self.config.vocoder.hop_length)

    # -- residency -----------------------------------------------------------

    def _frame_budget(self, context: AudioContext, prompt_length: int) -> int:
        """Frames the request asks for, capped by what the cache can hold.

        The autoregressive model's positions are the prompt's tokens and then
        one per frame, so a long prompt costs frames rather than failing at the
        last one.
        """
        asked = int(context.audio_duration * self.config.frame_rate)
        # One position spare: the loop samples one frame more than it keeps.
        headroom = (
            self.config.language_model.max_position_embeddings
            - prompt_length
            - 1
        )
        if headroom <= 0:
            raise ValueError(
                f"a {prompt_length}-token prompt leaves no room for audio in "
                f"{self.config.language_model.max_position_embeddings} "
                "positions"
            )
        if asked > headroom:
            logger.warning(
                "MiniMax Music 3: %.0fs of audio would need %d frames, but "
                "this prompt leaves room for %d. Generating %.0fs.",
                context.audio_duration,
                asked,
                headroom,
                headroom / self.config.frame_rate,
            )
        return max(1, min(asked, headroom))

    def _autoregressive(self) -> Generator:
        """The global model and the depth decoder, compiled and resident."""
        if self._generator is None:
            self._evict(keep="autoregressive")
            language_weights = self._checkpoint.weights("language_model")
            language = LanguageModelStage(
                self._session,
                self._device,
                self.config.language_model,
                language_weights,
                dtype=DTYPE,
                max_length=self.config.language_model.max_position_embeddings,
                recipe=self.config.sampling,
            )
            depth = DepthDecoderStage(
                self._device,
                self.config.depth_decoder,
                self._checkpoint.weights("rvq_depth_decoder"),
                language_weights,
                self.config.language_model,
                dtype=DTYPE,
                recipe=self.config.sampling,
            )
            self._language = language
            self._generator = Generator(
                language, depth, self.config.language_model
            )
        return self._generator

    def _diffusion_stage(self) -> DiffusionStage:
        """The condition encoder and the DiT."""
        if self._diffusion is None:
            self._evict(keep="diffusion")
            self._diffusion = DiffusionStage(
                self._device,
                self._checkpoint.weights("condition_encoder"),
                self._checkpoint.weights("transformer"),
                condition_config=self.config.condition_encoder,
                transformer_config=self.config.transformer,
                dtype=DTYPE,
            )
        return self._diffusion

    def _vocoder_stage(self) -> ChunkedVocoder:
        """The waveform decoder, which stays float32 -- it is small, and it is
        the one component whose rounding would be audible rather than diffuse."""
        if self._vocoder is None:
            self._evict(keep="vocoder")
            self._vocoder = ChunkedVocoder(
                self.config.vocoder,
                convert_vocoder_state(self._checkpoint.weights("vocoder")),
                self._device,
            )
        return self._vocoder

    def _evict(self, *, keep: str) -> None:
        """Drop every stage but *keep*, when the device cannot hold them all.

        The collection is explicit because the pool only recycles what Python
        has already released, and the next stage's first allocation is what
        needs the room.
        """
        if not self._staged:
            return
        if keep != "autoregressive" and self._generator is not None:
            # The cache's blocks are the manager's, not the model's, so they
            # have to be handed back before the manager itself is dropped.
            if self._language is not None:
                self._language.release()
            self._language = None
            self._generator = None
        if keep != "diffusion":
            self._diffusion = None
        if keep != "vocoder":
            self._vocoder = None
        gc.collect()
