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
"""Host-side driver that decodes latents of any length through one fixed graph.

Two reasons the decoder cannot simply be compiled at the clip's length.

Memory: activations cost about 28 MiB per latent frame, so a single 689-frame
denoising window peaks at 22.1 GiB of a 22.5 GiB A10G -- it fits, but nothing
else does, and a 60 s song would need eleven times that.

Shape specialization: a graph compiled per clip length recompiles for every new
duration, and compiling this decoder takes minutes.

Chunking is exact rather than approximate because the decoder is fully
convolutional with a measured receptive field of 19.6 latent frames
(``probe/probe_vocoder_chunk.py``): decode a window with ``overlap_frames`` of
real context on each side, keep the middle, and the result is bit-identical to
decoding the whole sequence.

What is *not* interchangeable is padding the latent sequence with zero frames to
make a window fit. Every convolution carries a bias, so a zero input frame
produces ``bias`` rather than zero, and those values propagate inward through the
receptive field -- measured at 1.3e-1 relative error on a 64-frame clip, four
orders of magnitude worse than the port's own 1.3e-5. The two true ends therefore
have to coincide with a window edge, where the convolutions' internal padding
supplies real zeros at every layer. Sequences shorter than one window get their
own graph instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import CompiledModel
from max.experimental.tensor import Tensor, default_dtype

from .components.vocoder import Vocoder
from .model_config import VocoderConfig

# 19.6 frames measured end to end, so 12 would do; 16 leaves margin for the
# dilation growth if the config's residual stack ever changes, at a cost of one
# extra frame of compute per 8 kept.
DEFAULT_OVERLAP_FRAMES = 16
# 128 kept frames puts the peak near 4.4 GiB and wastes 25% of the compute on
# context. Larger windows waste less and cost more memory, roughly linearly.
DEFAULT_CHUNK_FRAMES = 128


class ChunkedVocoder:
    """Compiles the decoder once and slides it over a latent sequence."""

    def __init__(
        self,
        config: VocoderConfig,
        weights: dict[str, np.ndarray],
        device: Device,
        *,
        dtype: DType = DType.float32,
        chunk_frames: int = DEFAULT_CHUNK_FRAMES,
        overlap_frames: int = DEFAULT_OVERLAP_FRAMES,
    ) -> None:
        self.config = config
        self.weights = weights
        self.dtype = dtype
        self.device = device
        self.chunk_frames = chunk_frames
        self.overlap_frames = overlap_frames
        self.window_frames = chunk_frames + 2 * overlap_frames
        # One module, compiled at as many lengths as the clip needs. The
        # parameters are declared but never materialized under `F.lazy()`;
        # `compile` binds the real values.
        with F.lazy(), default_dtype(dtype):
            self.vocoder = Vocoder(config).to(device)
        self._models: dict[int, CompiledModel[..., Any]] = {}
        self.model = self._model_for(self.window_frames)

    def _model_for(self, frames: int) -> CompiledModel[..., Any]:
        """Compile the decoder for a latent length, memoized."""
        if frames not in self._models:
            self._models[frames] = self.vocoder.compile(
                *self.vocoder.input_types(frames), weights=self.weights
            )
        return self._models[frames]

    def _plan(self, frames: int) -> list[tuple[int, int, int]]:
        """Lay out the windows covering ``frames``.

        Returns:
            One ``(window_start, keep_offset, keep_frames)`` triple per window,
            in latent frames. Every window lies wholly inside the sequence, so no
            window ever sees a padded frame; the first and last are pinned to the
            sequence edges, where the convolutions' own padding is correct.
        """
        overlap, chunk, window = (
            self.overlap_frames,
            self.chunk_frames,
            self.window_frames,
        )
        plan = []
        position = 0
        while position < frames:
            start = min(max(position - overlap, 0), frames - window)
            keep = min(chunk, frames - position)
            plan.append((start, position - start, keep))
            position += keep
        return plan

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Decodes latents into a stereo waveform.

        Args:
            latents: Shape ``(1, latent_channels, frames)``, channel-first as the
                denoiser produces them.

        Returns:
            Shape ``(2, frames * hop_length)`` in ``[-1, 1]``.

        Raises:
            ValueError: If ``latents`` is not a single batch row.
        """
        if latents.shape[0] != 1:
            raise ValueError(f"expected batch 1, got {latents.shape[0]}")
        latents = np.ascontiguousarray(latents, dtype=np.float32)
        frames = latents.shape[2]
        hop = self.config.hop_length

        def decode_window(
            model: CompiledModel[..., Any], window: np.ndarray
        ) -> np.ndarray:
            waveform = model(Tensor.from_dlpack(window).to(self.device))
            return waveform.to_numpy()[0]

        # Too short to hold a window, so there is nothing to slide: one graph at
        # the exact length is both correct and cheaper.
        if frames <= self.window_frames:
            return decode_window(self._model_for(frames), latents)

        pieces = []
        for start, keep_offset, keep in self._plan(frames):
            window = np.ascontiguousarray(
                latents[:, :, start : start + self.window_frames]
            )
            wave = decode_window(self.model, window)
            pieces.append(
                wave[:, keep_offset * hop : (keep_offset + keep) * hop]
            )
        return np.concatenate(pieces, axis=-1)
