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
"""The window plan and the three coordinate systems it has to keep aligned.

MiniMax Music 3 counts time three ways -- frames at 25 Hz, latents at ~86 Hz and
samples at 44.1 kHz -- and the overlap between denoising windows is expressed in
a different one at each of the three places it appears: the hop in frames, the
blend and carry in latents, the crop in samples. An off-by-one in any of them
moves the output by a latent frame or a whole window, which is audible as a
seam but not visible in any per-component numerical check.

These are pure arithmetic over the released checkpoint's rates, so they need no
GPU, no weights and no network -- and they pin the end-to-end lengths the
hardware gates produce, from the 12 s request every gate uses (529408 samples,
two windows) up to a full song (7727104 samples, 43 windows), where a carry
error accumulates instead of cancelling.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.pipelines.architectures.minimax_music3 import denoise
from max.pipelines.architectures.minimax_music3.model_config import (
    ConditionEncoderConfig,
    VocoderConfig,
)

# 12 s at the released 25 Hz frame rate, the duration every hardware gate uses.
FRAMES_12S = 300
SAMPLES_12S = 529408

# Full-length renders, as ``(frames, windows, samples)``. Twelve seconds is two
# windows, where a carry error either cancels or is obvious; these are the
# regime where it accumulates window after window, and where a ragged final
# window pays for every earlier one. Each sample count is what the pipeline
# actually wrote to a WAV for that duration, so the arithmetic below is pinned
# against real output rather than against itself.
FULL_LENGTH_RENDERS = [
    (1500, 14, 2649088),  # 60 s
    (4125, 41, 7285760),  # 2:45
    (4375, 43, 7727104),  # 2:55
]


@pytest.fixture
def condition() -> ConditionEncoderConfig:
    return ConditionEncoderConfig()


@pytest.fixture
def vocoder() -> VocoderConfig:
    return VocoderConfig()


def test_latent_length_truncates(condition: ConditionEncoderConfig) -> None:
    """A frame is 3.4453125 latents, and the reference truncates the product."""
    assert condition.latent_length(200) == 689  # not 690
    assert condition.latent_length(1) == 3
    assert condition.latent_length(0) == 1  # never degenerate


def test_frame_rate_is_25hz(condition: ConditionEncoderConfig) -> None:
    assert condition.frame_rate == 25.0


def test_hop_length_is_the_upsampling_product(vocoder: VocoderConfig) -> None:
    assert vocoder.hop_length == 8 * 8 * 4 * 2 == 512


def test_short_song_is_one_window() -> None:
    """Under a full window there is nothing to overlap."""
    assert denoise.window_starts(1) == [0]
    assert denoise.window_starts(denoise.CHUNK_FRAMES) == [0]


def test_window_starts_split_as_soon_as_a_window_overflows() -> None:
    """The last window is ragged rather than padded. One frame past a full
    window is enough to open a second one, which then covers the tail."""
    assert denoise.window_starts(201) == [0, 100]
    assert denoise.window_starts(300) == [0, 100]
    assert denoise.window_starts(301) == [0, 100, 200]


def test_plan_for_12s(condition: ConditionEncoderConfig) -> None:
    windows = denoise.plan(FRAMES_12S, condition.latent_length)
    assert [(w.frame_start, w.frame_end) for w in windows] == [
        (0, 200),
        (100, 300),
    ]
    assert [w.latents for w in windows] == [689, 689]
    # The first window shares nothing; the second inherits exactly the carry.
    assert [w.overlap for w in windows] == [0, denoise.OVERLAP_LATENTS]


def test_plan_clamps_overlap_to_a_short_final_window(
    condition: ConditionEncoderConfig,
) -> None:
    """A window shorter than the carry cannot blend more than it has."""
    windows = denoise.plan(250, condition.latent_length)
    assert len(windows) == 2
    assert windows[1].latents == condition.latent_length(150) == 516
    assert windows[1].overlap <= windows[1].latents


def test_crop_tiles_the_song_exactly_once(
    condition: ConditionEncoderConfig, vocoder: VocoderConfig
) -> None:
    """The gate's sample count, reproduced from the plan alone.

    This is the whole chain -- 25 Hz to ~86 Hz to 44.1 kHz, with a crop between
    every pair of windows -- and 529408 is what the end-to-end gate and both
    HTTP endpoints returned for a 12 s request.
    """
    windows = denoise.plan(FRAMES_12S, condition.latent_length)
    hop = vocoder.hop_length
    decoded = [
        np.zeros((2, window.latents * hop), np.float32) for window in windows
    ]
    assert denoise.crop(decoded, hop).shape == (2, SAMPLES_12S)


@pytest.mark.parametrize(("frames", "windows", "samples"), FULL_LENGTH_RENDERS)
def test_crop_tiles_a_full_length_song(
    frames: int,
    windows: int,
    samples: int,
    condition: ConditionEncoderConfig,
    vocoder: VocoderConfig,
) -> None:
    """The same chain as above, at the window counts a real song reaches.

    One channel rather than two, because the crop is channel-agnostic and the
    largest case is already 58 MB of zeros.
    """
    plan = denoise.plan(frames, condition.latent_length)
    assert len(plan) == windows
    hop = vocoder.hop_length
    decoded = [
        np.zeros((1, window.latents * hop), np.float32) for window in plan
    ]
    assert denoise.crop(decoded, hop).shape == (1, samples)


def test_seams_land_where_the_crop_constants_put_them(
    condition: ConditionEncoderConfig, vocoder: VocoderConfig
) -> None:
    """Where one window's audio gives way to the next, in samples.

    The joins are invisible in the output's length: shifting a latent from the
    right crop to the left one moves every seam and leaves the sample count
    exactly where it was, so the tests above would all still pass. These are the
    offsets an out-of-tree listening check measures for discontinuities, written
    as literals so that moving them has to be deliberate.
    """
    plan = denoise.plan(4125, condition.latent_length)
    kept = [
        window.latents
        - (0 if index == 0 else denoise.CROP_LEFT_LATENTS)
        - (0 if index == len(plan) - 1 else denoise.CROP_RIGHT_LATENTS)
        for index, window in enumerate(plan)
    ]
    # A full window is 689 latents: the first keeps all but its right crop, the
    # interior ones lose both, and the ragged last one loses only its left.
    assert kept[0] == 431
    assert set(kept[1:-1]) == {345}
    assert kept[-1] == 344

    seams = np.cumsum(kept)[:-1] * vocoder.hop_length
    assert len(seams) == len(plan) - 1 == 40
    assert seams[0] == 220672
    assert set(np.diff(seams)) == {176640}


def test_crop_of_one_window_keeps_everything(vocoder: VocoderConfig) -> None:
    waveform = np.zeros((2, 689 * vocoder.hop_length), np.float32)
    assert denoise.crop([waveform], vocoder.hop_length).shape == waveform.shape


def test_crop_clips_to_the_representable_range(vocoder: VocoderConfig) -> None:
    """The decoder can overshoot; a WAV writer's cast cannot."""
    waveform = np.array([[-3.0, 0.5, 3.0]], np.float32)
    cropped = denoise.crop([waveform], vocoder.hop_length)
    np.testing.assert_array_equal(cropped, [[-1.0, 0.5, 1.0]])


def test_crop_neither_repeats_nor_drops_a_sample(
    vocoder: VocoderConfig,
) -> None:
    """Every kept sample, in order, with none duplicated across a join.

    Each window carries a rising ramp inside its own band of the output range,
    so a sample can be traced both to the window it came from and to its
    position within that window. Constant-valued windows can only show that the
    order is right; distinct values also show that the spans meet exactly, which
    is the failure a listener would hear as a click.
    """
    hop = vocoder.hop_length
    length = 689 * hop
    rise = np.arange(length, dtype=np.float32) / length * 0.6
    pieces = [
        (np.float32(base) + rise).reshape(1, -1) for base in (-0.9, -0.3, 0.3)
    ]

    left = denoise.CROP_LEFT_LATENTS * hop
    right = length - denoise.CROP_RIGHT_LATENTS * hop
    expected = np.concatenate(
        [pieces[0][0, :right], pieces[1][0, left:right], pieces[2][0, left:]]
    )
    np.testing.assert_array_equal(denoise.crop(pieces, hop)[0], expected)


def test_euler_schedule_runs_from_noise_to_data() -> None:
    times, increments = denoise.euler_schedule(30)
    assert times.shape == increments.shape == (30,)
    # Time 0 is pure noise, and the increments carry the state to 1.
    assert times[0] == pytest.approx(0.0)
    np.testing.assert_allclose(increments, 1 / 30, rtol=1e-6)
    assert times[-1] + increments[-1] == pytest.approx(1.0)


def test_euler_schedule_is_uniform_at_any_step_count() -> None:
    for steps in (1, 2, 7, 50):
        times, increments = denoise.euler_schedule(steps)
        # float32, so the tolerance is the dtype's rather than the schedule's.
        np.testing.assert_allclose(increments, 1 / steps, rtol=1e-5)
        assert times[-1] + increments[-1] == pytest.approx(1.0)


def test_blend_ends_on_the_previous_window() -> None:
    """At t=0 the overlap is the window's own noise; at t=1 it is the previous
    window's latents, up to the epsilon that keeps the first step noisy."""
    noise = np.array([1.0, 1.0], np.float32)
    previous = np.array([-1.0, 3.0], np.float32)

    np.testing.assert_allclose(denoise.blend(noise, previous, 0.0), noise)
    np.testing.assert_allclose(
        denoise.blend(noise, previous, 1.0),
        previous + denoise.BLEND_EPSILON * noise,
        rtol=1e-6,
    )


def test_blend_is_monotone_between_its_endpoints() -> None:
    noise = np.zeros(1, np.float32)
    previous = np.ones(1, np.float32)
    values = [
        float(denoise.blend(noise, previous, t)[0])
        for t in np.linspace(0, 1, 11)
    ]
    assert values == sorted(values)
