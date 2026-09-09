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

"""Measures the joins between MiniMax-Music3's denoising windows.

The model denoises 8 s windows that overlap by half, blends them where they
meet and crops the result, so a three-minute song is dozens of windows joined
at dozens of seams. A join that went wrong sounds like a click (the waveform
jumps between two independently decoded signals) or a lurch in level (the two
windows disagree about how loud the song is), and both are cheap to look for
because the offsets follow from the model's constants rather than from the
audio.

Neither measurement has an absolute scale, and music is full of transients and
dynamics, so a seam is only suspicious relative to the song's own behaviour.
The baseline is thousands of arbitrary offsets in the same render; a seam is
reported as a percentile of that, and the verdict comes from how many seams
clear a high quantile against how many chance predicts.

These are host-side statistics over a finished waveform, which is what numpy
is for -- nothing here runs on the accelerator.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download
from max.pipelines.architectures.minimax_music3 import denoise
from max.pipelines.architectures.minimax_music3.model_config import (
    ConditionEncoderConfig,
    VocoderConfig,
)

BASELINE_DRAWS = 5000
EXCEEDANCE_QUANTILE = 99.0
SIGNIFICANCE = 0.05


def _checkpoint_root(model: str) -> Path:
    """Finds the checkpoint holding the component configs.

    Args:
        model: A local checkpoint directory, or a Hugging Face repository ID
            whose two relevant config files are fetched (not its weights).

    Returns:
        A directory containing ``condition_encoder/`` and ``vocoder/``.
    """
    local = Path(model)
    if local.is_dir():
        return local
    configs = [
        Path(hf_hub_download(model, f"{component}/config.json"))
        for component in ("condition_encoder", "vocoder")
    ]
    return configs[0].parents[1]


def seam_offsets(model: str, duration: float) -> list[int]:
    """Predicts the sample offsets where one window is joined to the next.

    Derived from the window plan and the crop rule rather than from the audio:
    the first window keeps everything but its right crop, and each later one
    keeps its length less both crops, so the joins fall on the running total of
    kept spans. Reading the checkpoint's own configs is what makes these a
    prediction -- if the crop arithmetic disagreed with the constants, the
    seams would not be where this looks.

    Args:
        model: The checkpoint that rendered the audio.
        duration: The duration that was asked for, in seconds.

    Returns:
        One offset per join, in samples.
    """
    root = _checkpoint_root(model)
    encoder = ConditionEncoderConfig.from_pretrained(root)
    vocoder = VocoderConfig.from_pretrained(root)

    windows = denoise.plan(
        int(duration * encoder.frame_rate), encoder.latent_length
    )
    offsets, total = [], 0
    for index, window in enumerate(windows):
        kept = window.latents
        kept -= 0 if index == 0 else denoise.CROP_LEFT_LATENTS
        kept -= 0 if index == len(windows) - 1 else denoise.CROP_RIGHT_LATENTS
        total += kept
        if index != len(windows) - 1:
            offsets.append(total * vocoder.hop_length)
    return offsets


def jumps_at(
    diffs: npt.NDArray[np.float32], offsets: npt.NDArray[np.intp]
) -> npt.NDArray[np.float32]:
    """Largest one-sample step in the immediate neighbourhood of each offset.

    A concatenation artefact is a discontinuity in the waveform itself, so the
    statistic is the first difference. The neighbourhood is a few samples wide
    because the seam joins two independently decoded signals and the worst step
    need not land exactly on the boundary sample.
    """
    return diffs[offsets[:, None] + np.arange(-4, 4)].max(axis=1)


def level_steps_at(
    energy: npt.NDArray[np.float64], offsets: npt.NDArray[np.intp], width: int
) -> npt.NDArray[np.float64]:
    """Absolute RMS ratio across each offset, in decibels.

    Catches the other seam failure: no click, but the two windows disagreeing
    about how loud the song is, which reads as a lurch in level. ``energy`` is
    a prefix sum of squares, so a window's RMS is a subtraction rather than a
    pass over the samples -- which is what makes thousands of offsets cheap.
    """
    quiet = 1e-9
    before = np.sqrt((energy[offsets] - energy[offsets - width]) / width)
    after = np.sqrt((energy[offsets + width] - energy[offsets]) / width)
    return np.abs(20.0 * np.log10((after + quiet) / (before + quiet)))


def binomial_tail(count: int, trials: int, probability: float) -> float:
    """Chance of at least ``count`` exceedances if seams were ordinary points.

    By construction a fraction ``probability`` of arbitrary offsets clears the
    quantile, so a handful of seams clearing it is expected rather than
    alarming. This is what turns a count into a verdict.
    """
    return float(
        sum(
            math.comb(trials, k)
            * probability**k
            * (1 - probability) ** (trials - k)
            for k in range(count, trials + 1)
        )
    )


def report_statistic(
    label: str,
    unit: str,
    seams: npt.NDArray[np.float64],
    baseline: npt.NDArray[np.float64],
    local_median: float,
) -> bool:
    """Prints one statistic's seam values against the baseline, and judges it.

    Returns:
        Whether the seams cleared the quantile more often than chance explains.
    """

    def rank(value: float) -> float:
        """Where ``value`` falls in the baseline, as a percentile."""
        return float((baseline < value).mean() * 100.0)

    threshold = float(np.percentile(baseline, EXCEEDANCE_QUANTILE))
    over = int((seams > threshold).sum())
    expected = len(seams) * (1.0 - EXCEEDANCE_QUANTILE / 100.0)
    chance = binomial_tail(over, len(seams), 1.0 - EXCEEDANCE_QUANTILE / 100.0)
    print(
        f"  {label:16s} seams median {np.median(seams):.4f}{unit} "
        f"(p{rank(float(np.median(seams))):.0f} of baseline), "
        f"worst {seams.max():.4f}{unit} (p{rank(float(seams.max())):.1f})"
    )
    print(
        f"  {'':16s} baseline p{EXCEEDANCE_QUANTILE:.0f} "
        f"{threshold:.4f}{unit}, max {baseline.max():.4f}{unit}; "
        f"local control median {local_median:.4f}{unit}"
    )
    print(
        f"  {'':16s} seams past baseline p{EXCEEDANCE_QUANTILE:.0f}: "
        f"{over} of {len(seams)} ({expected:.1f} expected by chance, "
        f"p={chance:.3f})"
    )
    return chance < SIGNIFICANCE


def report_seams(
    waveform: npt.NDArray[np.float32], rate: int, offsets: list[int]
) -> bool:
    """Judges the seams against the same measurements taken anywhere else.

    Args:
        waveform: The rendered audio, as ``(channels, samples)``.
        rate: Its sample rate in hertz.
        offsets: Predicted seam offsets, from :func:`seam_offsets`.

    Returns:
        True when the seams look like ordinary points in the song.
    """
    mono = waveform.astype(np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=0)
    if not offsets or offsets[-1] >= mono.size:
        print(f"seam offsets {offsets[:3]}... do not fit {mono.size} samples")
        return False

    width = rate // 40  # 25 ms, short enough to be local to the seam.
    seams = np.asarray(offsets, dtype=np.intp)
    diffs = np.abs(np.diff(mono))
    energy = np.concatenate(([0.0], np.cumsum(mono.astype(np.float64) ** 2)))

    # Seeded, because a verdict that changes between two runs of the same
    # numbers is not a verdict. Offsets within a level-step window of a seam
    # are dropped so the baseline cannot contain what it controls for.
    rng = np.random.default_rng(0)
    margin = width + 8
    draws = rng.integers(
        margin, mono.size - margin, BASELINE_DRAWS, dtype=np.intp
    )
    draws = draws[np.abs(draws[:, None] - seams).min(axis=1) > width]

    # A third of a window past each seam: same musical neighbourhood, so this
    # holds content roughly fixed where the arbitrary draws do not.
    stride = int(seams[1] - seams[0]) if len(seams) > 1 else rate
    local = seams[seams + stride // 3 + margin < mono.size] + stride // 3

    print(
        f"\n{len(seams)} seams at "
        f"{[round(offset / rate, 2) for offset in offsets[:4]]}... s, "
        f"calibrated against {draws.size} arbitrary offsets"
    )
    flagged = report_statistic(
        "one-sample jump",
        "",
        jumps_at(diffs, seams).astype(np.float64),
        jumps_at(diffs, draws).astype(np.float64),
        float(np.median(jumps_at(diffs, local))),
    )
    flagged |= report_statistic(
        "25 ms level step",
        " dB",
        level_steps_at(energy, seams, width),
        level_steps_at(energy, draws, width),
        float(np.median(level_steps_at(energy, local, width))),
    )

    worst = int(np.argmax(jumps_at(diffs, seams)))
    print(
        f"  {'':16s} worst seam is #{worst + 1} at {seams[worst] / rate:.2f}s"
    )
    print(
        "  VERDICT: seams stand out from the song's own behaviour"
        if flagged
        else "  verdict: seams are inside the song's own distribution"
    )
    return not flagged
