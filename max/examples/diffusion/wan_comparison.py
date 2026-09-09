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

"""Wan T2V performance comparison: diffusers (PyTorch) vs MAX.

Runs Wan text-to-video generation with both backends, performs warmup runs,
benchmarks with split preprocessing/execution timings, and prints a
side-by-side summary.  Supports Wan 2.1/2.2 T2V models (including MoE
variants) via the `--model` argument.  Also includes 1-frame (text-to-image)
configs in the iteration set.  Each iteration uses a different
`(height, width, num_frames, num_inference_steps)` tuple and cycles through
a fixed set of prompts with different sequence lengths to stress test
eager-mode recompilation and text-encoder performance.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import gc
import io
import json
import logging
import os
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from max.driver import DeviceSpec
from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
from max.pipelines.architectures.wan.context import WanContext
from max.pipelines.architectures.wan.tokenizer import WanTokenizer
from max.pipelines.diffusion.interface import DiffusionPipeline
from max.pipelines.diffusion.pipeline import PixelGenerationPipeline
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.modeling.types import (
    PipelineTask,
    PixelGenerationInputs,
    RequestID,
)
from max.pipelines.request import OpenResponsesRequest
from max.pipelines.request.open_responses import (
    OpenResponsesRequestBody,
    OutputImageContent,
    OutputVideoContent,
)
from max.pipelines.request.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
    VideoProviderOptions,
)
from PIL import Image

# Varying-input configurations for systematic memory/performance testing.
# Each tuple is `(height, width, num_frames, num_inference_steps)`.
# Ordered from largest to smallest estimated memory footprint so the
# biggest allocation lands first while the memory manager has maximum
# headroom, allowing more configurations to succeed before OOM.
#
# B200 memory budget (192 GB HBM3e, ~185 GB usable):
#   - DiT weights (14B, bf16) x 2 experts (MoE dual-load): ~56 GB
#   - Text encoder (UMT5):      ~9.4 GB
#   - VAE weights:               ~0.5 GB
#   - Framework/compile overhead: ~6 GB
#   - Workspace (shared per expert): scales with seq_len, x 2 experts
#   - Scheduler latent state:     scales with latent volume (modest)
#   - VAE decode peak:            scales with pixel volume
#
# With combined-blocks compilation (1 workspace per expert, not 40):
#   480p/49f   → ~74 GB   (seq_len ~20k)
#   480p/81f   → ~76 GB   (seq_len ~33k)
#   720p/49f   → ~77 GB   (seq_len ~47k)
#   720p/81f   → ~80 GB   (seq_len ~76k)
#
# Default negative prompt from the official Wan repo
# (`wan/configs/shared_config.py` → `sample_neg_prompt`).  The model was
# trained/tuned with this as the "empty" negative; omitting it produces
# visibly over-saturated, jittery, low-detail output.
WAN_DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"  # noqa: RUF001
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"  # noqa: RUF001
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"  # noqa: RUF001
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"  # noqa: RUF001
)

# Reused for the "with vs without Wan default negative prompt" A/B: slot 3
# runs the default negative, slot 7 forces an empty negative.  Keep the two
# in lockstep so the only varying input is the negative prompt.
_FORGE_PROMPT = (
    "A grizzled blacksmith hammers a white-hot sword blank on a"
    " blackened anvil in a shadowy stone-walled forge, sparks"
    " erupting in bright orange arcs with every strike. His"
    " soot-streaked hands grip the hammer and iron tongs with"
    " practiced precision; the glowing billet hisses as he plunges"
    " it briefly into a water trough, steam boiling up around him."
    " The camera holds a steady chest-level medium shot, locked on"
    " the anvil, framing just above his shoulders and just below the"
    " hearth. A roaring forge fire radiates warm rim light across"
    " his leather apron; the background falls into near-black void."
    " High-contrast chiaroscuro firelight, saturated amber and"
    " charcoal palette, 50mm lens shallow depth of field,"
    " ember-laden atmosphere, Game-of-Thrones-grade gritty realism,"
    " 35mm film, photorealistic, 24fps."
)

# Flat list of benchmark iterations.  Each entry is a dict with all params
# needed to build a `BenchmarkRequest`.  `negative_prompt` semantics:
#
#   * ``None`` — use the CLI ``--negative-prompt`` (defaults to
#     ``WAN_DEFAULT_NEGATIVE_PROMPT``).
#   * ``""``   — force an empty negative prompt regardless of the CLI flag.
#   * any str  — use this literal negative prompt.
#
# Slots 3 and 7 are the forge A/B: same prompt, same config, only the
# negative prompt differs.  Keep those two in lockstep.
REQUESTS: list[dict[str, Any]] = [
    # 0 — golden retriever portrait (image)
    {
        "height": 1152,
        "width": 2048,
        "num_frames": 1,
        "num_inference_steps": 20,
        "prompt": (
            "A muscular golden retriever stands knee-deep in shallow"
            " turquoise surf at low tide, body angled slightly toward the"
            " camera with head turned to the horizon. Salt-soaked fur clings"
            " to her flanks, a single water droplet caught on her chin."
            " Foam laps gently around her paws without breaking. The camera"
            " holds a steady low-angle three-quarter portrait, framing her"
            " against an empty sunset sky. Warm golden-hour rim light rakes"
            " across her coat; a distant headland softens into creamy"
            " bokeh. Amber and teal palette, volumetric sea haze, 85mm"
            " portrait lens with gentle anamorphic flares, cinematic"
            " beach-commercial grade, 35mm film emulation, photorealistic"
            " still."
        ),
        "negative_prompt": None,
    },
    # 1 — Shibuya neon portrait (image)
    {
        "height": 1536,
        "width": 2048,
        "num_frames": 1,
        "num_inference_steps": 20,
        "prompt": (
            "A young woman in a glossy red vinyl raincoat stands paused"
            " under a translucent umbrella on a rain-slicked Shibuya"
            " backstreet just after midnight, looking up at towering kanji"
            " neon above her. Her face is lit by pink and cyan glow from"
            " the reflected signs. Wet pavement mirrors the entire neon"
            " lattice, creating a second inverted city beneath her feet."
            " Warm window interiors bleed over cold asphalt. The camera"
            " holds a steady waist-level portrait, slightly wide framing"
            " with lampposts flanking her silhouette. Cyberpunk"
            " teal-and-magenta split-tone grade, anamorphic lens flares,"
            " 50mm compression, shallow depth of field, Blade Runner-"
            "inspired, photorealistic still."
        ),
        "negative_prompt": None,
    },
    # 2 — lotus macro still life (image)
    {
        "height": 2048,
        "width": 2048,
        "num_frames": 1,
        "num_inference_steps": 20,
        "prompt": (
            "A single pristine white lotus flower floats on a glass-still"
            " black pond at first light, fully bloomed and petals fanned"
            " wide to reveal a golden pollen-dusted core. A single perfect"
            " dew droplet rests at the tip of the topmost petal. Drifting"
            " mist hovers just above the water surface, softened by a low"
            " dawn sun. The camera holds a locked overhead macro"
            " composition, the flower centered against concentric"
            " lily-pad ripples. Soft diffused dawn light, pale gold and"
            " jade palette, volumetric rays piercing thin mist, 100mm"
            " macro lens, razor-sharp focus on the dew drop, creamy bokeh"
            " background, meditative sacred mood, high-end nature"
            " documentary photography, photorealistic still."
        ),
        "negative_prompt": None,
    },
    # 3 — forge (A/B pair with slot 7, uses CLI negative prompt)
    {
        "height": 1280,
        "width": 720,
        "num_frames": 49,
        "num_inference_steps": 50,
        "prompt": _FORGE_PROMPT,
        "negative_prompt": None,
    },
    # 4 — rocket launch (480p portrait, 121f)
    {
        "height": 832,
        "width": 480,
        "num_frames": 121,
        "num_inference_steps": 50,
        "prompt": (
            "At first light a colossal rocket ignites on its tropical"
            " launchpad, erupting a billowing white column of steam and"
            " orange flame that roars outward across scorched concrete."
            " The rocket trembles, strains, then lifts off, accelerating"
            " upward with a searing plume trailing behind it. The camera"
            " starts low at ground level looking up, tilts steeply upward"
            " following the ascent, then cranes alongside the rocket's"
            " body as it punches through low clouds into the pink and gold"
            " dawn sky. Heat distortion ripples against the rocket's skin."
            " Volumetric exhaust plumes, atmospheric haze, rim-lit"
            " contrails, 65mm anamorphic lens, IMAX-grade deep contrast,"
            " epic sci-fi realism, 24fps photorealistic."
        ),
        "negative_prompt": None,
    },
    # 5 — F1 at Monaco (480p landscape, 121f)
    {
        "height": 480,
        "width": 832,
        "num_frames": 121,
        "num_inference_steps": 50,
        "prompt": (
            "A Formula 1 car barrels into a wet Monaco hairpin at full"
            " throttle, carbon-fiber bodywork streaked with rain, rear tires"
            " kicking up dense rooster tails of mist behind it. The driver"
            " countersteers on the edge of grip as the chassis skitters"
            " sideways through the apex. Armco barriers and tricolor"
            " trackside signage flash past in strobing streaks, reflected"
            " across the glossy livery. The camera holds a locked trackside"
            " pan, tight on the car as it clips the apex, then whips with"
            " the chassis through the corner exit. Overcast Monte Carlo"
            " daylight diffuses silver off wet tarmac; bright neon team"
            " colors pop against grey stone walls. Anamorphic 100mm"
            " telephoto with motion blur, desaturated teal-and-red sport"
            " broadcast grade, 35mm film emulation, photorealistic, 24fps."
        ),
        "negative_prompt": None,
    },
    # 6 — medieval castle (no negative prompt)
    {
        "height": 720,
        "width": 1280,
        "num_frames": 49,
        "num_inference_steps": 50,
        "prompt": (
            "A sprawling stone medieval castle perches on a mist-shrouded"
            " granite hilltop at the break of dawn. Thick cold fog rolls"
            " through the valley below and slowly retreats to reveal"
            " towers, crenellations, and weathered battlements as light"
            " intensifies. A flock of ravens bursts from the highest spire"
            " in a wide sweep of black wings. The camera begins with a wide"
            " establishing crane shot high above the mist, then slowly"
            " descends and pushes forward toward the iron gate. Deep blue"
            " dawn gradually warms into amber golden light; volumetric rays"
            " pierce the retreating fog. Epic and majestic, Lord-of-the-"
            "Rings-grade cinematic landscape, 18mm wide-angle anamorphic,"
            " 4K, photorealistic, 24fps."
        ),
        "negative_prompt": "",
    },
    # 7 — forge (A/B pair with slot 3, forces empty negative prompt)
    {
        "height": 1280,
        "width": 720,
        "num_frames": 49,
        "num_inference_steps": 50,
        "prompt": _FORGE_PROMPT,
        "negative_prompt": "",
    },
]


class _Tee:
    """Fan writes out to multiple streams — used with
    ``contextlib.redirect_stdout`` to mirror the summary block to a file
    while still printing to the terminal."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@dataclass
class GPUMemorySnapshot:
    """A point-in-time snapshot of GPU memory usage."""

    allocated_gb: float
    reserved_gb: float
    free_gb: float
    total_gb: float

    @staticmethod
    def capture() -> GPUMemorySnapshot | None:
        """Capture current GPU memory state. Returns None if unavailable."""
        if not torch.cuda.is_available():
            return None
        try:
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            props = torch.cuda.get_device_properties(0)
            total = props.total_mem / (1024**3)
            free = total - reserved
            return GPUMemorySnapshot(
                allocated_gb=allocated,
                reserved_gb=reserved,
                free_gb=free,
                total_gb=total,
            )
        except Exception:
            return None

    @staticmethod
    def capture_nvidia_smi() -> GPUMemorySnapshot | None:
        """Capture GPU memory via nvidia-smi (works even without PyTorch
        holding the memory)."""
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            parts = out.strip().split(",")
            used_mb = float(parts[0].strip())
            free_mb = float(parts[1].strip())
            total_mb = float(parts[2].strip())
            return GPUMemorySnapshot(
                allocated_gb=used_mb / 1024,
                reserved_gb=used_mb / 1024,
                free_gb=free_mb / 1024,
                total_gb=total_mb / 1024,
            )
        except Exception:
            return None

    def __str__(self) -> str:
        return (
            f"alloc={self.allocated_gb:.1f}GB"
            f" reserved={self.reserved_gb:.1f}GB"
            f" free={self.free_gb:.1f}GB"
            f" total={self.total_gb:.1f}GB"
        )


def _compute_theoretical_memory(
    height: int,
    width: int,
    num_frames: int,
    *,
    dual_moe: bool = True,
) -> dict[str, float | str]:
    """Compute theoretical memory estimates for a given video config.

    Returns a dict of component names to estimated GB.

    With ``dual_moe=True`` (default), accounts for both high-noise and
    low-noise expert transformers being resident on GPU simultaneously,
    each with their own compiled model and shared workspace.
    """
    # VAE compression.
    vae_scale_s = 8
    vae_scale_t = 4
    latent_h = 2 * (height // (vae_scale_s * 2))
    latent_w = 2 * (width // (vae_scale_s * 2))
    adjusted = max(1, num_frames)
    if adjusted > 1:
        remainder = (adjusted - 1) % vae_scale_t
        if remainder != 0:
            adjusted += vae_scale_t - remainder
    latent_frames = (adjusted - 1) // vae_scale_t + 1

    # Patch embedding: patch_size = (1, 2, 2).
    seq_len = latent_frames * (latent_h // 2) * (latent_w // 2)

    # Latent tensor sizes.
    latent_elements = 1 * 16 * latent_frames * latent_h * latent_w
    latent_f32_gb = latent_elements * 4 / (1024**3)

    # DiT weights: 14B params x 2 bytes (bf16) ~ 28.1 GB per expert.
    num_experts = 2 if dual_moe else 1
    dit_weights_gb = 28.1 * num_experts

    # Workspace: with combined-blocks compilation, each expert has one
    # shared workspace.  Workspace ≈ peak single-block intermediates.
    # hidden_state [1, seq_len, 5120] bf16 + FFN [1, seq_len, 13824] bf16
    # + Q/K/V for self-attention (flash attn, ~2x hidden for Q+output).
    hidden_gb = seq_len * 5120 * 2 / (1024**3)
    ffn_gb = seq_len * 13824 * 2 / (1024**3)
    attn_gb = seq_len * 5120 * 2 * 2 / (1024**3)
    workspace_per_expert_gb = hidden_gb + ffn_gb + attn_gb
    workspace_gb = workspace_per_expert_gb * num_experts

    # Scheduler state: 5 copies of latent in f32.
    scheduler_gb = 5 * latent_f32_gb

    # VAE decode: output tensor [1, 3, frames, H, W] f32.
    vae_output_gb = 1 * 3 * num_frames * height * width * 4 / (1024**3)

    # RoPE: [seq_len, 128] f32 x 2 (cos + sin).
    rope_gb = 2 * seq_len * 128 * 4 / (1024**3)

    return {
        "dit_weights_bf16": dit_weights_gb,
        "text_encoder_bf16": 9.4,
        "vae_weights": 0.5,
        "workspace_peak": workspace_gb,
        "scheduler_latent_state": scheduler_gb,
        "vae_decode_output": vae_output_gb,
        "rope_embeddings": rope_gb,
        "framework_overhead": 6.0,
        "seq_len": float(seq_len),
        "latent_shape": f"[1,16,{latent_frames},{latent_h},{latent_w}]",
    }


@dataclass
class BenchmarkRequest:
    """All parameters for a single benchmark iteration."""

    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    prompt: str
    negative_prompt: str = ""
    # When True, this request is part of the no-negative-prompt comparison
    # set.  CLI overrides (`--negative-prompt`, shape overrides) must not
    # inject a negative prompt into it.
    force_no_negative: bool = False

    @property
    def config_key(self) -> tuple[int, int, int, int]:
        return (
            self.height,
            self.width,
            self.num_frames,
            self.num_inference_steps,
        )


def _build_requests(cli_negative_prompt: str) -> list[BenchmarkRequest]:
    """Materialize `REQUESTS` into `BenchmarkRequest`s.

    Per-entry ``negative_prompt`` of ``None`` uses the CLI default; ``""``
    forces an empty negative; any other string is used verbatim.
    """
    return [
        BenchmarkRequest(
            height=r["height"],
            width=r["width"],
            num_frames=r["num_frames"],
            num_inference_steps=r["num_inference_steps"],
            prompt=r["prompt"],
            negative_prompt=(
                cli_negative_prompt
                if r["negative_prompt"] is None
                else r["negative_prompt"]
            ),
            force_no_negative=r["negative_prompt"] == "",
        )
        for r in REQUESTS
    ]


@dataclass
class TimingResult:
    """Collected timings for a single backend."""

    preprocess_durations: list[float] = field(default_factory=list)
    execute_durations: list[float] = field(default_factory=list)
    total_durations: list[float] = field(default_factory=list)
    peak_memory_gb: list[float] = field(default_factory=list)
    errors: list[tuple[int, str, str]] = field(default_factory=list)
    """(iteration_index, config_desc, error_message) for failed iterations."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Wan T2V performance: diffusers vs MAX."
    )
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        help=(
            "HuggingFace model ID to use (e.g. "
            "'Wan-AI/Wan2.1-T2V-14B-Diffusers', "
            "'Wan-AI/Wan2.2-T2V-A14B-Diffusers')."
        ),
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--guidance-scale-2",
        type=float,
        default=None,
        help=(
            "Secondary guidance scale for low-noise expert (MoE models). "
            "When set, the pipeline uses dual guidance scales."
        ),
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=0,
        help="Number of warmup passes over all requests (not timed) per backend.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help=(
            "Number of timed iterations to run (1 to number of prompts). "
            "Each iteration uses a different prompt/config pair. "
            "Defaults to the full prompt list."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not set, a random seed is generated.",
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable torch.compile on the diffusers pipeline components "
            "(default on). Pass --no-torch-compile to run eager."
        ),
    )
    parser.add_argument(
        "--skip-diffusers",
        action="store_true",
        help="Skip the diffusers (PyTorch) run.",
    )
    parser.add_argument(
        "--skip-max",
        action="store_true",
        help="Skip the MAX run.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help=(
            "Override the number of frames for all requests. "
            "If not set, uses the per-config defaults."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override the height for all requests.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override the width for all requests.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override the number of inference steps for all requests.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=WAN_DEFAULT_NEGATIVE_PROMPT,
        help=(
            "Negative prompt for all requests. Defaults to the official Wan "
            "negative prompt (from `wan/configs/shared_config.py`). Pass an "
            "empty string to disable."
        ),
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=None,
        help=(
            "Flow-matching shift (`flow_shift`) for the diffusers scheduler. "
            "Official recipes: 12.0 for Wan2.2 T2V-A14B, 5.0 for Wan2.1 T2V "
            "720p, 3.0 for Wan2.1 T2V 480p. If unset, uses whatever ships in "
            "the HF scheduler config."
        ),
    )
    parser.add_argument(
        "--weight-path",
        type=str,
        action="append",
        default=None,
        help="Path(s) to model weight files. Can be specified multiple times.",
    )
    parser.add_argument(
        "--quantization-encoding",
        type=str,
        default=None,
        choices=["float32", "bfloat16"],
        help="Weight encoding type.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Skip saving generated video frames to disk.",
    )
    parser.add_argument(
        "--model-override",
        type=str,
        action="append",
        default=None,
        help=(
            "Per-component overrides in 'component.field=value' format. "
            "Repeatable. Example: "
            "'transformer.quantization_encoding=bfloat16'."
        ),
    )
    parser.add_argument(
        "--single-prompt",
        type=str,
        default=None,
        help=(
            "Run a single prompt instead of the varied prompt set. "
            "Useful for quick testing."
        ),
    )
    return parser.parse_args(argv)


def _build_warmup_requests(
    requests: list[BenchmarkRequest],
    num_warmups: int,
) -> list[BenchmarkRequest]:
    """Return warmup requests: the full request list repeated *num_warmups* times."""
    return requests * num_warmups


def _print_gpu_info() -> None:
    """Print GPU name, VRAM, and CUDA build version."""
    if not torch.cuda.is_available():
        print("  GPU: not available (CPU mode)")
        return
    name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_memory / (1024**3)
    cuda_version = torch.version.cuda or "N/A"
    print(f"  GPU            : {name}")
    print(f"  VRAM           : {total_gb:.1f} GB")
    print(f"  CUDA (build)   : {cuda_version}")


def _truncate_prompt(prompt: str, max_len: int = 40) -> str:
    """Return a display-friendly truncated prompt."""
    if len(prompt) <= max_len:
        return prompt
    return prompt[: max_len - 1] + "…"


def _normalize_frames(frames: np.ndarray) -> np.ndarray:
    """Normalize frames to (T, H, W, C) uint8."""
    if frames.ndim == 5:
        # (B, C, T, H, W) -> take first batch element -> (C, T, H, W)
        frames = frames[0]
    # Check last dim first: if it's 1/3, the data is already (T, H, W, C).
    # Only then fall back to interpreting shape[0] as the channel dim (which
    # would otherwise mis-trigger on (N=1, H, W, C) 1-frame outputs).
    if frames.ndim == 4 and frames.shape[-1] in (1, 3):
        pass  # already (T, H, W, C)
    elif frames.ndim == 4 and frames.shape[0] in (1, 3):
        frames = np.transpose(frames, (1, 2, 3, 0))
    elif frames.ndim == 4:
        frames = np.transpose(frames, (0, 2, 3, 1))

    if frames.dtype in (np.float32, np.float64):
        frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    elif frames.dtype == np.float16:
        frames = np.clip(frames.astype(np.float32) * 255.0, 0, 255).astype(
            np.uint8
        )
    return frames


def _save_output(
    frames: np.ndarray,
    output_dir: str,
    prefix: str,
    fps: int = 16,
) -> str:
    """Save frames as a PNG (single frame) or MP4 (multiple frames).

    Args:
        frames: Video frames, shape (T, H, W, C) or (T, C, H, W) uint8/float.
        output_dir: Directory to save the output in.
        prefix: Filename prefix.
        fps: Frames per second for the output video (multi-frame only).

    Returns:
        The saved filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames = _normalize_frames(frames)

    if frames.shape[0] == 1:
        frame = frames[0]
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame.squeeze(-1)
        fname = f"{prefix}.png"
        Image.fromarray(frame).save(os.path.join(output_dir, fname))
        return fname

    import av

    fname = f"{prefix}.mp4"
    path = os.path.join(output_dir, fname)

    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"

    for frame_data in frames:
        video_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    return fname


def _get_vae_dtype(args: argparse.Namespace) -> torch.dtype:
    """Determine VAE dtype from --model-override flags.

    Defaults to bf16 for apples-to-apples comparison with MAX.  If the user
    passes ``--model-override vae.quantization_encoding=float32``, the VAE
    falls back to fp32.
    """
    if args.model_override:
        for override in args.model_override:
            if override.startswith("vae.quantization_encoding="):
                encoding = override.split("=", 1)[1].lower()
                if encoding in ("float32", "fp32"):
                    return torch.float32
    return torch.bfloat16


def _load_diffusers_pipeline(model_id: str, args: argparse.Namespace) -> Any:
    """Load Wan pipeline via diffusers with optimized attention."""
    import diffusers
    from diffusers import AutoencoderKLWan

    vae_dtype = _get_vae_dtype(args)

    # Load VAE separately so we control its dtype, ensuring an
    # apples-to-apples comparison with the MAX backend.
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=vae_dtype
    )
    pipe = diffusers.WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda")

    # The HF checkpoint only ships `shared.weight` for the UMT5 text encoder
    # but its `config.json` sets `tie_word_embeddings=false`, so transformers
    # randomly initializes `encoder.embed_tokens.weight` on load.  Force the
    # tie to recover correct text conditioning — without this every prompt
    # produces garbled output.
    pipe.text_encoder.encoder.embed_tokens.weight = (
        pipe.text_encoder.shared.weight
    )

    if args.shift is not None:
        pipe.scheduler = pipe.scheduler.from_config(
            pipe.scheduler.config, flow_shift=args.shift
        )

    if args.torch_compile:
        # Compile key components for performance.
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            pipe.text_encoder = torch.compile(
                pipe.text_encoder,
                mode="max-autotune",
                fullgraph=False,
            )
        # Use max-autotune-no-cudagraphs instead of max-autotune for the
        # transformer.  max-autotune enables CUDA graphs whose output
        # buffers get reused across replays, but the diffusers denoising
        # loop reads the previous output after the next transformer call,
        # hitting a stale-tensor error.  This mode keeps Triton kernel
        # autotuning while disabling CUDA graph capture.
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
        )
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae = torch.compile(
                pipe.vae, mode="max-autotune", fullgraph=True
            )
    return pipe


def _list_output_dir(output_dir: str | None, label: str) -> None:
    """Log the sorted contents of the output dir so it's obvious whether
    saves from the prior phase actually landed and whether anything is
    being clobbered between phases."""
    if output_dir is None or not os.path.isdir(output_dir):
        return
    entries = sorted(os.listdir(output_dir))
    print(f"\n  Output dir contents {label} ({output_dir}):")
    if not entries:
        print("    (empty)")
        return
    for name in entries:
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"    {name:60s} {size_mb:7.2f} MB")
        else:
            print(f"    {name}/")


def _torch_pipe_kwargs(
    req: BenchmarkRequest,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Kwargs for `diffusers.WanPipeline.__call__`.

    Kept in one place so warmup and timed loops pass identical args, and
    so parity with the MAX request body built in `_build_max_inputs` is
    verifiable at a glance.  `guidance_scale_2` is only forwarded when
    set — diffusers raises if it's passed on a single-expert Wan 2.1
    pipeline (no `boundary_ratio` configured).
    """
    kwargs: dict[str, Any] = {
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt or None,
        "height": req.height,
        "width": req.width,
        "num_frames": req.num_frames,
        "num_inference_steps": req.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }
    if args.guidance_scale_2 is not None:
        kwargs["guidance_scale_2"] = args.guidance_scale_2
    return kwargs


def _video_to_numpy(output: Any) -> np.ndarray:
    """Extract numpy video frames from diffusers output.

    Diffusers WanPipeline returns output.frames as a list of lists of
    PIL Images, or sometimes as a tensor. Normalize to (T, H, W, C) float32.
    """
    frames = output.frames
    if isinstance(frames, torch.Tensor):
        return frames.cpu().float().numpy()
    if isinstance(frames, np.ndarray):
        return frames.astype(np.float32)
    # frames is list[list[PIL.Image]]
    if isinstance(frames, list) and isinstance(frames[0], list):
        pil_frames = frames[0]  # First batch element.
        return np.stack(
            [np.array(f).astype(np.float32) / 255.0 for f in pil_frames],
            axis=0,
        )
    if isinstance(frames, list) and isinstance(frames[0], Image.Image):
        return np.stack(
            [np.array(f).astype(np.float32) / 255.0 for f in frames],
            axis=0,
        )
    raise TypeError(f"Unexpected diffusers output type: {type(frames)}")


def run_diffusers(
    args: argparse.Namespace,
    requests: list[BenchmarkRequest],
    output_dir: str | None = None,
) -> TimingResult:
    """Benchmark Wan T2V through diffusers. Returns split timings."""
    print("\n=== Diffusers (PyTorch) backend (text-to-video) ===", flush=True)
    print(f"Loading pipeline ({args.model})...", flush=True)
    pipe = _load_diffusers_pipeline(args.model, args)

    warmup_requests = _build_warmup_requests(requests, args.num_warmups)
    for i, req in enumerate(warmup_requests):
        print(
            f"  warmup {i + 1}/{len(warmup_requests)}"
            f" ({req.width}x{req.height}, {req.num_frames}f,"
            f" {req.num_inference_steps} steps,"
            f" prompt={_truncate_prompt(req.prompt)!r})",
            flush=True,
        )
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        pipe(**_torch_pipe_kwargs(req, args, generator))
        torch.cuda.synchronize()

    result = TimingResult()
    n = len(requests)
    for i, req in enumerate(requests):
        config_desc = (
            f"{req.width}x{req.height}, {req.num_frames}f,"
            f" {req.num_inference_steps} steps"
        )
        theory = _compute_theoretical_memory(
            req.height, req.width, req.num_frames
        )
        print(
            f"  iter {i + 1}/{n}"
            f" ({config_desc},"
            f" prompt={_truncate_prompt(req.prompt)!r})",
            flush=True,
        )
        print(
            f"    theoretical: seq_len={int(theory['seq_len']):,}"
            f"  latent={theory['latent_shape']}"
            f"  est_total="
            f"{sum(v for k, v in theory.items() if isinstance(v, float) and k != 'seq_len'):.1f}GB",
            flush=True,
        )

        mem_before = GPUMemorySnapshot.capture()
        if mem_before:
            print(f"    gpu before : {mem_before}", flush=True)

        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

        try:
            # Full pipeline timing (diffusers doesn't easily split preprocess/execute).
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = pipe(**_torch_pipe_kwargs(req, args, generator))
            torch.cuda.synchronize()
            t_total = time.perf_counter() - t0

            peak_gb = torch.cuda.max_memory_allocated(0) / (1024**3)
            mem_after = GPUMemorySnapshot.capture()
            print(f"    gpu after  : {mem_after}")
            print(f"    peak alloc : {peak_gb:.1f} GB")

            result.preprocess_durations.append(0.0)
            result.execute_durations.append(t_total)
            result.total_durations.append(t_total)
            result.peak_memory_gb.append(peak_gb)

            if output_dir is not None:
                try:
                    frames = _video_to_numpy(output)
                    prefix = (
                        f"iter{i}_torch_{req.width}x{req.height}"
                        f"_{req.num_frames}f_{req.num_inference_steps}s"
                    )
                    fname = _save_output(frames, output_dir, prefix)
                    full_path = os.path.join(output_dir, fname)
                    size_mb = (
                        os.path.getsize(full_path) / (1024 * 1024)
                        if os.path.exists(full_path)
                        else -1.0
                    )
                    print(f"    saved: {fname} ({size_mb:.2f} MB)")
                except Exception as e:
                    print(f"    WARNING: Failed to save output: {e}")
                    print(f"    traceback:\n{traceback.format_exc()}")
        except Exception as e:
            error_msg = str(e)
            mem_at_error = GPUMemorySnapshot.capture()
            print(f"    ERROR: {error_msg}")
            if mem_at_error:
                print(f"    gpu at error: {mem_at_error}")
            print(f"    traceback:\n{traceback.format_exc()}")
            result.errors.append((i, config_desc, error_msg))
            # Clear GPU memory and continue to next iteration.
            gc.collect()
            torch.cuda.empty_cache()
            continue

    # Free GPU memory before MAX runs.
    del pipe
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()

    return result


def _load_max_pipeline(
    args: argparse.Namespace,
) -> tuple[PixelGenerationPipeline[WanContext], WanTokenizer, PipelineConfig]:
    """Load Wan pipeline via MAX."""
    model_id = args.model

    manifest = ModelManifest.from_model_path(
        model_id,
        device_specs=[DeviceSpec.accelerator()],
    )
    if args.weight_path:
        manifest = manifest.with_override(
            "transformer",
            weight_path=[Path(p) for p in args.weight_path],
        )
    if args.quantization_encoding:
        manifest = manifest.with_override(
            "transformer",
            quantization_encoding=args.quantization_encoding,
        )

    # Apply flexible per-component overrides from --model-override.
    if args.model_override:
        from pydantic import TypeAdapter

        for override in args.model_override:
            dot_pos = override.find(".")
            if dot_pos < 1:
                raise ValueError(
                    f"Invalid --model-override format: {override!r}. "
                    f"Expected 'component.field=value'."
                )
            eq_pos = override.find("=", dot_pos)
            if eq_pos < dot_pos + 2:
                raise ValueError(
                    f"Invalid --model-override format: {override!r}. "
                    f"Expected 'component.field=value'."
                )
            component = override[:dot_pos]
            field_name = override[dot_pos + 1 : eq_pos]
            raw_value = override[eq_pos + 1 :]

            if field_name not in MAXModelConfig.model_fields:
                raise ValueError(
                    f"Unknown MAXModelConfig field: {field_name!r}. "
                    f"Valid fields: "
                    f"{sorted(MAXModelConfig.model_fields.keys())}"
                )
            if component not in manifest:
                raise ValueError(
                    f"Component {component!r} not found in manifest. "
                    f"Available: {list(manifest.keys())}"
                )

            field_info = MAXModelConfig.model_fields[field_name]
            adapter: TypeAdapter[Any] = TypeAdapter(field_info.annotation)
            try:
                parsed = json.loads(raw_value)
            except (json.JSONDecodeError, ValueError):
                parsed = raw_value
            value = adapter.validate_python(parsed)
            manifest = manifest.with_override(component, **{field_name: value})

    config = PipelineConfig(
        models=manifest,
        runtime=PipelineRuntimeConfig(),
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        config.models.main_architecture_name,
        task=PipelineTask.PIXEL_GENERATION,
    )
    assert arch is not None, (
        f"No architecture found in MAX registry for {model_id}."
    )

    tokenizer = WanTokenizer(
        model_path=model_id,
        pipeline_config=config,
        subfolder="tokenizer",
        max_length=512,
    )

    pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
    pipeline = PixelGenerationPipeline[WanContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
    )
    return pipeline, tokenizer, config


async def _build_max_inputs(
    args: argparse.Namespace,
    tokenizer: WanTokenizer,
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    steps: int,
    negative_prompt: str = "",
) -> tuple[Any, Any]:
    """Build MAX pipeline inputs for a single video generation."""
    body = OpenResponsesRequestBody(
        model=args.model,
        input=prompt,
        seed=args.seed,
        provider_options=ProviderOptions(
            image=ImageProviderOptions(
                height=height,
                width=width,
                steps=steps,
                guidance_scale=args.guidance_scale,
                negative_prompt=negative_prompt or None,
            ),
            video=VideoProviderOptions(
                negative_prompt=negative_prompt or None,
                height=height,
                width=width,
                num_frames=num_frames,
                steps=steps,
                guidance_scale_2=args.guidance_scale_2,
            ),
        ),
    )
    request = OpenResponsesRequest(request_id=RequestID(), body=body)
    context = await tokenizer.new_context(request)
    inputs = PixelGenerationInputs[WanContext](
        batch={context.request_id: context}
    )
    return inputs, context


def run_max(
    args: argparse.Namespace,
    requests: list[BenchmarkRequest],
    output_dir: str | None = None,
) -> TimingResult:
    """Benchmark Wan T2V through MAX with split preprocess/execute timings."""
    print("\n=== MAX backend (text-to-video) ===", flush=True)
    print(f"Loading pipeline ({args.model})...", flush=True)
    pipeline, tokenizer, _config = _load_max_pipeline(args)

    warmup_requests = _build_warmup_requests(requests, args.num_warmups)
    for i, req in enumerate(warmup_requests):
        print(
            f"  warmup {i + 1}/{len(warmup_requests)}"
            f" ({req.width}x{req.height}, {req.num_frames}f,"
            f" {req.num_inference_steps} steps,"
            f" prompt={_truncate_prompt(req.prompt)!r})",
            flush=True,
        )
        inputs, _ = asyncio.run(
            _build_max_inputs(
                args,
                tokenizer,
                req.prompt,
                req.height,
                req.width,
                req.num_frames,
                req.num_inference_steps,
                req.negative_prompt,
            )
        )
        pipeline.execute(inputs)

    result = TimingResult()
    n = len(requests)
    for i, req in enumerate(requests):
        config_desc = (
            f"{req.width}x{req.height}, {req.num_frames}f,"
            f" {req.num_inference_steps} steps"
        )
        theory = _compute_theoretical_memory(
            req.height, req.width, req.num_frames
        )
        print(
            f"  iter {i + 1}/{n}"
            f" ({config_desc},"
            f" prompt={_truncate_prompt(req.prompt)!r})",
            flush=True,
        )
        print(
            f"    theoretical: seq_len={int(theory['seq_len']):,}"
            f"  latent={theory['latent_shape']}"
            f"  est_total="
            f"{sum(v for k, v in theory.items() if isinstance(v, float) and k != 'seq_len'):.1f}GB",
            flush=True,
        )

        mem_before = GPUMemorySnapshot.capture_nvidia_smi()
        if mem_before:
            print(f"    gpu before : {mem_before}", flush=True)

        try:
            # Preprocessing: tokenization + context + input building
            t0 = time.perf_counter()
            inputs, context = asyncio.run(
                _build_max_inputs(
                    args,
                    tokenizer,
                    req.prompt,
                    req.height,
                    req.width,
                    req.num_frames,
                    req.num_inference_steps,
                    req.negative_prompt,
                )
            )
            t_preprocess = time.perf_counter() - t0

            # Model execution
            t1 = time.perf_counter()
            outputs = pipeline.execute(inputs)
            t_execute = time.perf_counter() - t1

            mem_after = GPUMemorySnapshot.capture_nvidia_smi()
            if mem_after:
                print(f"    gpu after  : {mem_after}")
                result.peak_memory_gb.append(mem_after.allocated_gb)
            else:
                result.peak_memory_gb.append(0.0)

            result.preprocess_durations.append(t_preprocess)
            result.execute_durations.append(t_execute)
            result.total_durations.append(t_preprocess + t_execute)

            # Save output frames (outside timing)
            if output_dir is not None:
                output = outputs[context.request_id]
                output = asyncio.run(tokenizer.postprocess(output))
                if output.output:
                    frames = []
                    for content in output.output:
                        if (
                            isinstance(content, OutputImageContent)
                            and content.image_data
                        ):
                            image_bytes = base64.b64decode(content.image_data)
                            img = Image.open(io.BytesIO(image_bytes))
                            frames.append(np.array(img))
                        elif (
                            isinstance(content, OutputVideoContent)
                            and content.frames is not None
                        ):
                            frames.extend(list(content.frames))
                    if frames:
                        prefix = (
                            f"iter{i}_max_{req.width}x{req.height}"
                            f"_{req.num_frames}f_{req.num_inference_steps}s"
                        )
                        stacked = np.stack(frames)
                        fname = _save_output(stacked, output_dir, prefix)
                        full_path = os.path.join(output_dir, fname)
                        size_mb = (
                            os.path.getsize(full_path) / (1024 * 1024)
                            if os.path.exists(full_path)
                            else -1.0
                        )
                        print(f"    saved: {fname} ({size_mb:.2f} MB)")
        except Exception as e:
            error_msg = str(e)
            mem_at_error = GPUMemorySnapshot.capture_nvidia_smi()
            print(f"    ERROR: {error_msg}")
            if mem_at_error:
                print(f"    gpu at error: {mem_at_error}")
            print(f"    traceback:\n{traceback.format_exc()}")
            result.errors.append((i, config_desc, error_msg))
            gc.collect()
            continue

    return result


def _print_summary(label: str, result: TimingResult) -> None:
    """Print timing summary for one backend."""
    n = len(result.total_durations)
    print(f"  {label}:")
    print(f"    iterations : {n} succeeded, {len(result.errors)} failed")

    if n == 0:
        if result.errors:
            print("    All iterations failed:")
            for idx, desc, msg in result.errors:
                print(f"      iter {idx}: [{desc}] {msg}")
        return

    if result.preprocess_durations and any(
        d > 0 for d in result.preprocess_durations
    ):
        mean_pp = statistics.mean(result.preprocess_durations)
        std_pp = statistics.stdev(result.preprocess_durations) if n > 1 else 0.0
        print(f"    preprocess : {mean_pp:8.2f} s  (std {std_pp:.2f} s)")

    mean_ex = statistics.mean(result.execute_durations)
    std_ex = statistics.stdev(result.execute_durations) if n > 1 else 0.0
    print(f"    execute    : {mean_ex:8.2f} s  (std {std_ex:.2f} s)")

    mean_tot = statistics.mean(result.total_durations)
    std_tot = statistics.stdev(result.total_durations) if n > 1 else 0.0
    mn = min(result.total_durations)
    mx = max(result.total_durations)
    print(f"    total      : {mean_tot:8.2f} s  (std {std_tot:.2f} s)")
    print(f"    min        : {mn:8.2f} s")
    print(f"    max        : {mx:8.2f} s")

    if result.peak_memory_gb:
        peak = max(result.peak_memory_gb)
        print(f"    peak GPU   : {peak:8.1f} GB")

    if result.errors:
        print(f"    FAILED iterations ({len(result.errors)}):")
        for idx, desc, msg in result.errors:
            print(f"      iter {idx}: [{desc}] {msg}")


def _print_per_iteration(
    label: str,
    result: TimingResult,
    requests: list[BenchmarkRequest],
) -> None:
    """Print per-iteration breakdown with config/prompt details."""
    print(f"\n  {label} per-iteration breakdown:")

    # Build a set of failed iteration indices for quick lookup.
    failed_indices = {idx for idx, _, _ in result.errors}

    header = (
        f"    {'Config':>22s}"
        f"  {'Neg':>4s}"
        f"  {'Prompt':>42s}"
        f"  {'Preprocess':>10s}  {'Execute':>10s}  {'Total':>10s}"
        f"  {'Peak GPU':>8s}  {'Status':>8s}"
    )
    print(header)

    success_idx = 0
    for i, req in enumerate(requests):
        config_str = (
            f"{req.width:4d}x{req.height:<4d} {req.num_frames:3d}f"
            f" {req.num_inference_steps:2d}s"
        )
        neg_str = (
            "no"
            if (req.force_no_negative or not req.negative_prompt)
            else "yes"
        )
        if i in failed_indices:
            error_msg = next(msg for idx, _, msg in result.errors if idx == i)
            short_err = (
                error_msg[:30] + "…" if len(error_msg) > 30 else error_msg
            )
            line = (
                f"    {config_str:>22s}"
                f"  {neg_str:>4s}"
                f"  {_truncate_prompt(req.prompt, 42):>42s}"
                f"  {'---':>10s}  {'---':>10s}  {'---':>10s}"
                f"  {'---':>8s}  FAILED"
            )
            print(line)
            print(f"      error: {short_err}")
        else:
            pp = (
                result.preprocess_durations[success_idx]
                if result.preprocess_durations
                else 0.0
            )
            ex = result.execute_durations[success_idx]
            tot = result.total_durations[success_idx]
            mem = (
                f"{result.peak_memory_gb[success_idx]:.1f}GB"
                if result.peak_memory_gb
                else "N/A"
            )
            line = (
                f"    {config_str:>22s}"
                f"  {neg_str:>4s}"
                f"  {_truncate_prompt(req.prompt, 42):>42s}"
                f"  {pp:10.2f}  {ex:10.2f}  {tot:10.2f}"
                f"  {mem:>8s}  OK"
            )
            print(line)
            success_idx += 1


def _write_prompts_manifest(
    path: str,
    args: argparse.Namespace,
    requests: list[BenchmarkRequest],
) -> None:
    """Write a full manifest documenting every parameter fed into both the
    diffusers (PyTorch) backend and the MAX backend for this run.

    The goal is end-to-end reproducibility from this file alone: it records
    resolved CLI values (overridden and default), per-iteration shapes and
    prompts, and the concrete kwargs / request-body fields each backend
    receives — including MAX/diffusers provider-option defaults we do *not*
    override.
    """
    vae_dtype = _get_vae_dtype(args)
    # torch.compile settings applied in `_load_diffusers_pipeline`.
    torch_compile_modes = {
        "text_encoder": ("max-autotune", False),
        "transformer": ("max-autotune-no-cudagraphs", True),
        "vae": ("max-autotune", True),
    }

    def _fmt(v: Any) -> str:
        if v is None:
            return "None"
        if isinstance(v, str) and not v:
            return "'' (empty)"
        return repr(v) if isinstance(v, str) else str(v)

    with open(path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("Wan T2V comparison — run manifest\n")
        f.write(f"written: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 72 + "\n\n")

        # --- Reproducibility -------------------------------------------
        f.write("[Reproducibility]\n")
        f.write(f"  command          : {shlex.join(sys.argv)}\n")
        f.write(f"  seed             : {args.seed}\n")
        f.write(f"  reproduce with   : --seed {args.seed}\n\n")

        # --- CLI args (all, resolved) ----------------------------------
        f.write("[CLI args — resolved values]\n")
        cli_fields: list[tuple[str, Any]] = [
            ("--model", args.model),
            ("--guidance-scale", args.guidance_scale),
            ("--guidance-scale-2", args.guidance_scale_2),
            ("--num-warmups", args.num_warmups),
            ("--num-iterations", args.num_iterations),
            ("--seed", args.seed),
            ("--torch-compile", args.torch_compile),
            ("--skip-diffusers", args.skip_diffusers),
            ("--skip-max", args.skip_max),
            ("--height (shape override)", args.height),
            ("--width (shape override)", args.width),
            ("--num-frames (shape override)", args.num_frames),
            ("--num-inference-steps (override)", args.num_inference_steps),
            ("--negative-prompt (CLI default)", args.negative_prompt),
            ("--shift (scheduler flow_shift)", args.shift),
            ("--weight-path", args.weight_path),
            ("--quantization-encoding", args.quantization_encoding),
            ("--no-output", args.no_output),
            ("--model-override", args.model_override),
            ("--single-prompt", args.single_prompt),
        ]
        for name, value in cli_fields:
            if (
                isinstance(value, str)
                and name == "--negative-prompt (CLI default)"
                and len(value) > 80
            ):
                f.write(
                    f"  {name:36s}: {_truncate_prompt(value, 80)}"
                    f"  (len={len(value)})\n"
                )
            else:
                f.write(f"  {name:36s}: {_fmt(value)}\n")
        f.write("\n")

        # --- PyTorch / diffusers ---------------------------------------
        f.write("[diffusers (PyTorch) — WanPipeline]\n")
        f.write("  Pipeline load (_load_diffusers_pipeline):\n")
        f.write("    AutoencoderKLWan.from_pretrained(\n")
        f.write(f"      {args.model!r},\n")
        f.write("      subfolder='vae',\n")
        f.write(f"      torch_dtype={vae_dtype},\n")
        f.write("    )\n")
        f.write("    diffusers.WanPipeline.from_pretrained(\n")
        f.write(f"      {args.model!r},\n")
        f.write("      vae=<loaded above>,\n")
        f.write("      torch_dtype=torch.bfloat16,  # hardcoded\n")
        f.write("      low_cpu_mem_usage=True,       # hardcoded\n")
        f.write("    ).to('cuda')\n")
        f.write("  Post-load patches:\n")
        f.write(
            "    text_encoder.encoder.embed_tokens.weight"
            " = text_encoder.shared.weight\n"
        )
        f.write(
            "      (HF UMT5 tie hack — checkpoint only ships `shared.weight`"
            " but config has tie_word_embeddings=false)\n"
        )
        if args.shift is not None:
            f.write(
                f"    scheduler = scheduler.from_config("
                f"flow_shift={args.shift})\n"
            )
        else:
            f.write(
                "    scheduler flow_shift: (HF config default, unchanged)\n"
            )
        f.write("  torch.compile:\n")
        if args.torch_compile:
            for name, (mode, fullgraph) in torch_compile_modes.items():
                f.write(
                    f"    {name:13s}: mode={mode!r}, fullgraph={fullgraph}\n"
                )
        else:
            f.write("    disabled (eager mode, --no-torch-compile)\n")
        f.write("  Per-call kwargs passed to pipe(**kwargs):\n")
        f.write("    prompt               : <per-iter>\n")
        f.write("    negative_prompt      : <per-iter, or None if empty>\n")
        f.write("    height               : <per-iter>\n")
        f.write("    width                : <per-iter>\n")
        f.write("    num_frames           : <per-iter>\n")
        f.write("    num_inference_steps  : <per-iter>\n")
        f.write(f"    guidance_scale       : {args.guidance_scale}\n")
        if args.guidance_scale_2 is not None:
            f.write(f"    guidance_scale_2     : {args.guidance_scale_2}\n")
        else:
            f.write(
                "    guidance_scale_2     : (not forwarded — raises on"
                " single-expert Wan2.1)\n"
            )
        f.write(
            "    generator            :"
            f" torch.Generator('cpu').manual_seed({args.seed})\n"
        )
        f.write(
            "  WanPipeline.__call__ args NOT passed (diffusers defaults used):\n"
        )
        f.write(
            "    num_videos_per_prompt, max_sequence_length, output_type,\n"
            "    return_dict, callback_on_step_end,"
            " callback_on_step_end_tensor_inputs,\n"
            "    attention_kwargs, latents, prompt_embeds,"
            " negative_prompt_embeds\n"
        )
        f.write("\n")

        # --- MAX --------------------------------------------------------
        f.write("[MAX — PixelGenerationPipeline]\n")
        f.write("  Pipeline load (_load_max_pipeline):\n")
        f.write(
            "    ModelManifest.from_model_path("
            f"{args.model!r}, device_specs=[DeviceSpec.accelerator()])\n"
        )
        f.write(f"    weight_path override      : {_fmt(args.weight_path)}\n")
        f.write(
            "    quantization_encoding ovr :"
            f" {_fmt(args.quantization_encoding)}\n"
        )
        f.write(
            f"    model_override list       : {_fmt(args.model_override)}\n"
        )
        f.write(
            "    PipelineConfig(models=manifest,"
            " runtime=PipelineRuntimeConfig())\n"
        )
        f.write("    WanTokenizer:\n")
        f.write(f"      model_path     : {args.model!r}\n")
        f.write("      subfolder      : 'tokenizer'\n")
        f.write("      max_length     : 512\n")
        f.write(
            "  MAX-side scheduler shift (WanTokenizer._select_wan_flow_shift):\n"
        )
        f.write(
            "    if scheduler.flow_shift is set and != 1.0 → use it verbatim\n"
        )
        f.write("    else linearly interpolate by pixel count:\n")
        f.write("      480p (480*832 = 399_360 px) → flow_shift = 3.0\n")
        f.write("      720p (720*1280 = 921_600 px) → flow_shift = 5.0\n")
        f.write("  Per-call OpenResponsesRequestBody:\n")
        f.write(f"    model                : {args.model}\n")
        f.write("    input (=prompt)      : <per-iter>\n")
        f.write(f"    seed                 : {args.seed}\n")
        f.write("    provider_options.image (ImageProviderOptions):\n")
        f.write("      height             : <per-iter>\n")
        f.write("      width              : <per-iter>\n")
        f.write("      steps              : <per-iter>\n")
        f.write(f"      guidance_scale     : {args.guidance_scale}\n")
        f.write("      negative_prompt    : <per-iter, or None if empty>\n")
        f.write("    provider_options.video (VideoProviderOptions):\n")
        f.write("      height             : <per-iter>\n")
        f.write("      width              : <per-iter>\n")
        f.write("      num_frames         : <per-iter>\n")
        f.write("      steps              : <per-iter>\n")
        f.write(f"      guidance_scale_2   : {_fmt(args.guidance_scale_2)}\n")
        f.write("      negative_prompt    : <per-iter, or None if empty>\n")
        f.write(
            "  Provider-option defaults we do NOT override (used as shipped):\n"
        )
        f.write("    ImageProviderOptions:\n")
        f.write("      secondary_prompt          : None\n")
        f.write("      secondary_negative_prompt : None\n")
        f.write("      true_cfg_scale            : 1.0\n")
        f.write("      num_images                : 1\n")
        f.write("      residual_threshold        : None\n")
        f.write("      strength                  : 0.6  (unused for t2v)\n")
        f.write("      cfg_normalization         : False\n")
        f.write("      cfg_truncation            : 1.0\n")
        f.write("      output_format             : 'jpeg'\n")
        f.write("    VideoProviderOptions:\n")
        f.write("      frames_per_second         : None\n")
        f.write("\n")

        # --- Negative prompt (full text) --------------------------------
        f.write("[Negative prompt — CLI default, full text]\n")
        if args.negative_prompt:
            f.write(f"  length  : {len(args.negative_prompt)} chars\n")
            f.write(f"  value   : {args.negative_prompt}\n")
        else:
            f.write("  value   : (empty)\n")
        f.write("\n")

        # --- Iterations (full per-iter spec) ----------------------------
        f.write("[Iterations]\n")
        f.write(f"  count    : {len(requests)}\n")
        for i, req in enumerate(requests):
            if req.force_no_negative:
                neg_label = "forced-empty (slot opts out)"
            elif not req.negative_prompt:
                neg_label = "empty"
            elif req.negative_prompt == args.negative_prompt:
                neg_label = "CLI default (see above)"
            else:
                neg_label = "literal override"
            f.write(
                f"\n  iter {i}:\n"
                f"    height             : {req.height}\n"
                f"    width              : {req.width}\n"
                f"    num_frames         : {req.num_frames}\n"
                f"    num_inference_steps: {req.num_inference_steps}\n"
                f"    negative_prompt    : {neg_label}\n"
            )
            if (
                neg_label == "literal override"
                and req.negative_prompt is not None
            ):
                f.write(f"    negative_prompt text: {req.negative_prompt}\n")
            f.write(f"    prompt:\n      {req.prompt}\n")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args(argv)

    # Resolve seed: generate a random one if not explicitly provided.
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    print(f"  seed             : {args.seed}")
    print("  (reproduce with: --seed", args.seed, ")")

    # Build the benchmark request list.
    if args.single_prompt:
        first = REQUESTS[0]
        requests = [
            BenchmarkRequest(
                height=args.height or first["height"],
                width=args.width or first["width"],
                num_frames=args.num_frames or first["num_frames"],
                num_inference_steps=(
                    args.num_inference_steps or first["num_inference_steps"]
                ),
                prompt=args.single_prompt,
                negative_prompt=args.negative_prompt,
            )
        ]
    else:
        requests = _build_requests(args.negative_prompt)
        # Apply per-request CLI overrides if specified.  Negative prompt
        # behavior is preserved: `force_no_negative` rows stay empty.
        if (
            args.height
            or args.width
            or args.num_frames
            or args.num_inference_steps
        ):
            requests = [
                BenchmarkRequest(
                    height=args.height or req.height,
                    width=args.width or req.width,
                    num_frames=args.num_frames or req.num_frames,
                    num_inference_steps=(
                        args.num_inference_steps or req.num_inference_steps
                    ),
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    force_no_negative=req.force_no_negative,
                )
                for req in requests
            ]

    # Limit the number of timed iterations if requested.
    if args.num_iterations is not None:
        if args.num_iterations < 1 or args.num_iterations > len(requests):
            raise ValueError(
                f"--num-iterations must be between 1 and {len(requests)}"
                f" (number of prompts), got {args.num_iterations}"
            )
        requests = requests[: args.num_iterations]

    # Create output directory for saved frames (unless --no-output).
    output_dir: str | None = None
    if not args.no_output:
        output_dir = os.path.abspath("wan_comparison")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        print(f"  Saving frames to: {output_dir}/")

    # Write a prompts.txt manifest documenting every param we feed to
    # both backends, so a run can be reproduced end-to-end from this file.
    if output_dir is not None:
        prompts_path = os.path.join(output_dir, "prompts.txt")
        _write_prompts_manifest(prompts_path, args, requests)
        print(f"  Saved prompt manifest: {prompts_path}")

    diffusers_result: TimingResult | None = None
    max_result: TimingResult | None = None

    if not args.skip_diffusers:
        try:
            diffusers_result = run_diffusers(args, requests, output_dir)
        except Exception as e:
            print(f"ERROR running diffusers: {e}", file=sys.stderr)
            traceback.print_exc()
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        _list_output_dir(output_dir, "after torch phase")

    if not args.skip_max:
        _list_output_dir(output_dir, "before MAX phase")
        try:
            max_result = run_max(args, requests, output_dir)
        except Exception as e:
            print(f"ERROR running MAX: {e}", file=sys.stderr)
            traceback.print_exc()
        _list_output_dir(output_dir, "after MAX phase")

    # Mirror the summary block to `summary.txt` in the output dir (if any)
    # so results can be retrieved later.  Without --no-output the file
    # lands next to the generated frames.
    summary_path: str | None = None
    summary_file: Any = None
    if output_dir is not None:
        summary_path = os.path.join(output_dir, "summary.txt")
        summary_file = open(summary_path, "w")
        summary_ctx: contextlib.AbstractContextManager[Any] = (
            contextlib.redirect_stdout(_Tee(sys.stdout, summary_file))
        )
    else:
        summary_ctx = contextlib.nullcontext()

    try:
        with summary_ctx:
            # Summary header
            print("\n" + "=" * 60)
            model_name = (
                args.model.rsplit("/", 1)[-1]
                if "/" in args.model
                else args.model
            )
            print(
                f"{model_name} Text-to-Video Performance Comparison"
                f" — {datetime.now():%Y-%m-%d}"
            )
            print("=" * 60)
            _print_gpu_info()
            print(f"  model            : {args.model}")
            print(f"  seed             : {args.seed}")
            print("  mode             : Text-to-Video")
            print(f"  guidance scale   : {args.guidance_scale}")
            if args.guidance_scale_2 is not None:
                print(f"  guidance scale 2 : {args.guidance_scale_2}")
            print(f"  warmup runs      : {args.num_warmups}")
            print(f"  timed iterations : {len(requests)}")
            print()
            print("  Torch config:")
            compile_mode = (
                "torch.compile (max-autotune)"
                if args.torch_compile
                else "eager (no compile)"
            )
            print(f"    mode           : {compile_mode}")
            print("    dtype          : BF16")
            print()
            print("  MAX config:")
            max_dtype = (
                args.quantization_encoding.upper()
                if args.quantization_encoding
                else "BF16"
            )
            print(f"    dtype          : {max_dtype}")
            print()

            # Print theoretical memory estimates for all configs.
            print("  Theoretical memory estimates per config:")
            print(
                f"    {'Config':>24s}  {'Seq Len':>10s}"
                f"  {'Latent Shape':>24s}"
                f"  {'Workspace':>9s}  {'Sched St':>9s}  {'VAE Dec':>8s}"
                f"  {'Est Total':>10s}"
            )
            for req in requests:
                t = _compute_theoretical_memory(
                    req.height, req.width, req.num_frames
                )
                total_est = sum(
                    v
                    for k, v in t.items()
                    if isinstance(v, float) and k != "seq_len"
                )
                cfg_label = (
                    f"{req.width}x{req.height} {req.num_frames}f"
                    f" {req.num_inference_steps}s"
                )
                print(
                    f"    {cfg_label:>24s}"
                    f"  {int(t['seq_len']):>10,}"
                    f"  {t['latent_shape']:>24s}"
                    f"  {t['workspace_peak']:>8.1f}G"
                    f"  {t['scheduler_latent_state']:>8.2f}G"
                    f"  {t['vae_decode_output']:>7.02f}G"
                    f"  {total_est:>9.1f}G"
                )
            print()

            if diffusers_result:
                _print_summary("Diffusers (PyTorch)", diffusers_result)
            if max_result:
                _print_summary("MAX", max_result)

            if diffusers_result:
                _print_per_iteration(
                    "Diffusers (PyTorch)",
                    diffusers_result,
                    requests,
                )
            if max_result:
                _print_per_iteration("MAX", max_result, requests)

            if diffusers_result and max_result:
                d_mean = statistics.mean(diffusers_result.total_durations)
                m_mean = statistics.mean(max_result.total_durations)
                speedup = d_mean / m_mean if m_mean > 0 else float("inf")
                per_iter_speedups = [
                    d / m if m > 0 else float("inf")
                    for d, m in zip(
                        diffusers_result.total_durations,
                        max_result.total_durations,
                        strict=True,
                    )
                ]
                min_speedup = min(per_iter_speedups)
                max_speedup = max(per_iter_speedups)
                print()
                print("  Speedup (MAX vs Diffusers):")
                print(f"    avg: {speedup:.2f}x")
                print(f"    min: {min_speedup:.2f}x")
                print(f"    max: {max_speedup:.2f}x")

            print()
    finally:
        if summary_file is not None:
            summary_file.close()

    if summary_path is not None:
        print(f"  Saved summary to: {summary_path}")

    return 0


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    raise SystemExit(main())
