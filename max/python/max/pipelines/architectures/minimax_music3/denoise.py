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
"""Flow-matching denoising over overlapping windows of the latent timeline.

The autoregressive stage produces per-frame hidden states at 25 Hz. Those are
denoised in 200-frame windows with a 100-frame hop, which is a 50% overlap, and
neighbouring windows are made to agree on their shared boundary by a blend that
runs *inside* the denoising loop rather than after it: at every Euler step the
leading latents of the current window are pulled toward the previous window's
trailing latents, on a schedule that starts at pure noise and ends at the
previous window's values exactly.

That is the whole reason this file exists rather than a for-loop at the call
site. Three coordinate systems have to line up -- frames at 25 Hz, latents at
~86 Hz, and samples at 44.1 kHz -- and the overlap is expressed in a different
one of them at each of the three places it appears: the hop in frames, the blend
and carry in latents, the final crop in samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# Windows of the frame timeline, and their hop. A 100-frame hop into a 200-frame
# window is a 50% overlap.
CHUNK_FRAMES = 200
CHUNK_HOP = 100

# The hop is ~344.5 latent frames, but only the leading 172 of the overlap are
# blended; the rest of each window is denoised freely and then discarded by the
# crop below. The carry a window hands forward is its latents over
# ``[length - 344, length - 172)``, which is where the next window begins.
OVERLAP_LATENTS = 172
CARRY_BACK_LATENTS = 2 * OVERLAP_LATENTS

# Crops applied to each window's *decoded waveform*, in latent frames. Every
# window after the first drops its leading 86, and every window before the last
# drops its trailing 258, so the kept spans tile the song exactly once.
CROP_LEFT_LATENTS = 86
CROP_RIGHT_LATENTS = CARRY_BACK_LATENTS - CROP_LEFT_LATENTS

# The blend never quite reaches pure previous-window values at t = 0, which keeps
# the first step from being handed a noiseless input.
BLEND_EPSILON = 1e-6

# The reference pipeline's sampling defaults.
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 1.7


def window_starts(num_frames: int) -> list[int]:
    """Frame index at which each denoising window starts.

    A song shorter than one window is a single window. Otherwise the last window
    is allowed to be ragged rather than padded -- the stop is ``num_frames -
    CHUNK_HOP``, so a window is only started if at least one hop of new material
    remains.
    """
    if num_frames <= CHUNK_FRAMES:
        return [0]
    return list(range(0, num_frames - CHUNK_HOP, CHUNK_HOP))


def euler_schedule(num_steps: int) -> tuple[npt.NDArray[np.float32], ...]:
    """Build the flow-matching time steps and their Euler increments.

    The scheduler is configured with ``invert_sigmas``, so the sigmas are
    generated descending from 1 and then flipped, which for this configuration
    (``shift = 1``, ``num_train_timesteps = 1``, no dynamic shifting) leaves the
    times ascending from 0 with a uniform ``1 / num_steps`` increment. The
    arithmetic is spelled out rather than shortcut to that closed form, because
    the closed form is a property of the config and not of the scheduler.

    Returns:
        The times at which the velocity is evaluated, ``(num_steps,)``, and the
        increment to take after each, also ``(num_steps,)``. Time 0 is pure
        noise and 1 is data.
    """
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps).astype(np.float32)
    sigmas = 1.0 - sigmas
    # The terminal sigma, which supplies the last step's increment.
    padded = np.concatenate([sigmas, np.ones(1, np.float32)])
    return sigmas, np.diff(padded)


@dataclass
class Window:
    """One denoising window's place in the three coordinate systems."""

    frame_start: int
    frame_end: int
    latents: int
    """Length of this window on the latent timeline."""
    overlap: int
    """Leading latents shared with the previous window, blended and then
    overwritten by it. Zero for the first window."""


def plan(num_frames: int, latent_length: object) -> list[Window]:
    """Lay out the windows for a song of ``num_frames`` frames.

    Args:
        num_frames: Length of the autoregressive output on the 25 Hz frame grid.
        latent_length: Maps a frame count to the latent count the condition
            encoder will resample it to.

    Returns:
        The windows, in order.
    """
    windows = []
    carry = 0
    for start in window_starts(num_frames):
        end = min(start + CHUNK_FRAMES, num_frames)
        length = int(latent_length(end - start))  # type: ignore[operator]
        windows.append(Window(start, end, length, min(carry, length)))
        # What this window will hand forward, which may be nothing if it is
        # shorter than the carry span.
        back = max(0, length - CARRY_BACK_LATENTS)
        carry = max(back, length - OVERLAP_LATENTS) - back
    return windows


def blend(
    noise: npt.NDArray[np.float32],
    previous: npt.NDArray[np.float32],
    time: float,
) -> npt.NDArray[np.float32]:
    """Interpolate the overlap from its own initial noise toward the previous
    window's latents.

    At ``time = 0`` this is the noise the window was seeded with, and at
    ``time = 1`` it is the previous window's latents, so re-running it before
    every Euler step drags the two windows into agreement over the course of the
    denoising rather than seaming them afterwards.
    """
    return (1.0 - (1.0 - BLEND_EPSILON) * time) * noise + time * previous


def crop(
    waveforms: list[npt.NDArray[np.float32]], hop_length: int
) -> npt.NDArray[np.float32]:
    """Trim each window's decoded waveform to its share and concatenate.

    The windows overlap by ``CARRY_BACK_LATENTS``; the split point is placed
    ``CROP_LEFT_LATENTS`` into the overlap, so each interior window keeps the
    span between its predecessor's and successor's boundaries.
    """
    pieces = []
    for index, waveform in enumerate(waveforms):
        left = 0 if index == 0 else CROP_LEFT_LATENTS * hop_length
        right = (
            0
            if index == len(waveforms) - 1
            else CROP_RIGHT_LATENTS * hop_length
        )
        pieces.append(waveform[..., left : waveform.shape[-1] - right])
    return np.clip(np.concatenate(pieces, axis=-1), -1.0, 1.0)
