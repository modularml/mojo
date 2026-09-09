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

"""FLUX.2 performance comparison: diffusers (PyTorch) vs MAX.

Runs FLUX.2 image generation with both backends, performs warmup runs,
benchmarks with split preprocessing/execution timings, and prints a
side-by-side summary. Supports FLUX.2-dev, FLUX.2-klein, and other registered
FLUX.2 variants via the `--model` argument.  Supports both text-to-image and
image-to-image modes. Each iteration uses a different
`(height, width, num_inference_steps)` tuple and cycles through a fixed set of
prompts with different sequence lengths to stress test eager-mode recompilation
and text-encoder performance.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import io
import json
import logging
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
import traceback
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import diffusers
import torch
from max.driver import DeviceSpec
from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
from max.pipelines.context import PixelContext
from max.pipelines.diffusion.config import (
    DenoisingCacheSettings,
    resolve_denoising_cache,
)
from max.pipelines.diffusion.interface import DiffusionPipeline
from max.pipelines.diffusion.pipeline import PixelGenerationPipeline
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.modeling.types import (
    PipelineTask,
    PixelGenerationInputs,
    RequestID,
)
from max.pipelines.request import OpenResponsesRequest
from max.pipelines.request.open_responses import (
    InputImageContent,
    InputTextContent,
    OpenResponsesRequestBody,
    OutputImageContent,
    UserMessage,
)
from max.pipelines.request.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
)
from PIL import Image

# Varying-input configurations for recompilation stress testing.
# Each tuple is `(height, width, num_inference_steps)`.
VARIED_CONFIGS: list[tuple[int, int, int]] = [
    (1024, 1024, 50),
    (768, 1360, 40),
    (1360, 768, 30),
    (512, 512, 50),
    (1024, 768, 28),
    (768, 768, 50),
    (1360, 1024, 40),
    (512, 1024, 30),
    (1024, 512, 28),
    (768, 1024, 50),
    (256, 256, 15),
]

# Fixed prompts of varying sequence lengths for text-encoder stress testing.
# Ordered roughly short → long so iteration logs are easy to follow.
VARIED_PROMPTS: list[str] = [
    (
        "Black cat hiding behind a watermelon slice, professional studio"
        " shot, bright red and turquoise background with summer mystery vibe"
    ),
    (
        "Dog wrapped in white towel after bath, photographed with direct"
        " flash and high exposure, fur wet details sharply visible, editorial"
        " raw portrait, cinematic harsh flash lighting, intimate humorous"
        " documentary style"
    ),
    (
        "A small red propeller plane banking sharply between massive jungle"
        " trees in a bright anime style, with midday sun illuminating lush"
        " green foliage and waterfalls cascading in the background."
    ),
    (
        "A businessman in a charcoal grey suit resting his arms on a bamboo"
        " railing at a secluded beach in the Philippines, cigarette glowing"
        " between his lips, illustrated in a vintage Japanese woodblock print"
        " style with soft pastel tones, calm turquoise waters, and a hazy"
        " afternoon sky."
    ),
    (
        "A group of baby penguins in a trampoline park, having the time of"
        " their lives, 80s vintage photo"
    ),
]

# Curated image-to-image benchmark pairs.
# Each entry is (url, prompt, label) or (url, prompt, label, (out_h, out_w)).
# When the optional 4th element is provided, the output is generated at that
# size instead of the input image's native dimensions.
# Images are chosen at non-standard dimensions (not 1024x1024) to stress test
# the pipeline's handling of varied input resolutions and aspect ratios.
# All images are from Unsplash (free license) or Wikimedia Commons (public domain).
IMG2IMG_BENCHMARKS: list[
    tuple[str, str, str] | tuple[str, str, str, tuple[int, int]]
] = [
    (
        # 3:2 wide landscape — tests style transfer with depth and color gradients
        "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=1920",
        (
            "Same mountain valley scene transformed into a Japanese woodblock"
            " print with ukiyo-e style, soft pastel colors, stylized clouds"
            " and water"
        ),
        "landscape_3x2_1920x1280",
        (768, 1024),
    ),
    (
        # 2:3 tall portrait — tests fine detail preservation on animal subject
        "https://images.unsplash.com/photo-1633722715463-d30f4f325e24?w=1280",
        (
            "A regal golden retriever wearing an ornate Elizabethan ruff collar"
            " and royal crown, sitting on a velvet throne, Renaissance oil"
            " painting style, rich warm colors, Rembrandt lighting, gold leaf"
            " accents, 16th century Dutch master painting"
        ),
        "golden_retriever_2x3_853x1280",
        (1024, 768),
    ),
    (
        # 16:9 wide urban — tests cyberpunk style transfer on neon-lit street
        "https://images.unsplash.com/photo-1542051841857-5f90071e7989?w=1800",
        (
            "A futuristic cyberpunk city street in the year 2199, towering"
            " holographic advertisements, flying cars in the background,"
            " rain-slicked neon-lit pavement, Blade Runner aesthetic, cinematic"
            " atmosphere, volumetric fog, ultra detailed, concept art"
        ),
        "tokyo_cyberpunk_16x9_1800x1013",
        (768, 1360),
    ),
    (
        # 2:3 tall painting (Mona Lisa) — tests style domain transfer
        (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/"
            "Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/"
            "960px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"
        ),
        (
            "Same composition and pose but as a modern photograph,"
            " photorealistic skin, contemporary clothing, DSLR shallow"
            " depth of field"
        ),
        "mona_lisa_2x3_800x1200",
        (1024, 768),
    ),
    (
        # 3:2 small landscape — tests low-resolution input with fantasy transformation
        "https://images.pexels.com/photos/354089/pexels-photo-354089.jpeg?w=768",
        (
            "A dark fantasy castle fortress perched on a cliff edge, a massive"
            " dragon coiled around the tallest tower breathing fire into a"
            " stormy sky, lightning crashing in the background, Lord of the"
            " Rings style, epic wide-angle shot, dramatic clouds, matte"
            " painting, cinematic lighting"
        ),
        "castle_dragon_3x2_768x512",
        (512, 512),
    ),
    (
        # 2:3 tall portrait — tests stylized character transformation from photo
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=1280",
        (
            "Same person as a 3D Fortnite character skin, Unreal Engine render,"
            " stylized cartoon proportions, bright cel-shaded lighting,"
            " legendary rarity glow, vibrant colors, battle royale hero pose,"
            " clean game-ready character art"
        ),
        "fortnite_character_2x3_853x1280",
        (1024, 768),
    ),
    (
        # 16:9 wide jungle — tests game environment style transfer
        "https://images.unsplash.com/photo-1564460549828-f0219a31bf90?w=1800",
        (
            "Same jungle scene as a Fortnite battle royale map location,"
            " stylized 3D game environment, vibrant saturated colors, cartoon"
            " foliage, loot chests glowing in the clearing, Unreal Engine"
            " render, bright sunny lighting"
        ),
        "fortnite_jungle_16x9_1800x1013",
        (768, 1360),
    ),
    (
        # Close-up cable knit texture — tests garment transfer to new scene
        "https://provinceofcanada.com/cdn/shop/files/Lead-Oatmeal-Cable-Knit-1S8A8479_b3695431-70c7-4cc8-848a-79124f227d62.jpg?v=1762373283&width=1280",
        (
            "A middle-aged man laughing at a dimly lit cocktail bar wearing this"
            " exact camel cable-knit crewneck sweater, candid editorial photograph,"
            " shallow depth of field, warm tungsten lighting, shot on medium"
            " format film, GQ magazine fashion spread, natural expression"
        ),
        "cableknit_bar_3x4_768x1024",
        (1024, 768),
    ),
]

# Cache directory for downloaded benchmark images.
_IMG2IMG_CACHE_DIR = Path(tempfile.gettempdir()) / "flux2_benchmark_images"


def _download_benchmark_image(url: str, label: str) -> Image.Image:
    """Download a benchmark image, caching it on disk to avoid re-downloads."""
    _IMG2IMG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _IMG2IMG_CACHE_DIR / f"{label}.png"
    if cache_path.exists():
        img = Image.open(cache_path).convert("RGB")
        print(
            f"  Loaded cached benchmark image: {label} ({img.width}x{img.height})"
        )
        return img

    print(f"  Downloading benchmark image: {label} ...")
    req = urllib.request.Request(
        url, headers={"User-Agent": "flux2-benchmark/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.save(cache_path)
    print(f"  Downloaded: {label} ({img.width}x{img.height})")
    return img


@dataclass
class BenchmarkRequest:
    """All parameters for a single benchmark iteration."""

    height: int
    width: int
    num_inference_steps: int
    prompt: str
    image: Image.Image | None = None
    label: str = ""
    image_data_uri: str | None = None

    @property
    def config_key(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.num_inference_steps)


def _build_txt2img_requests() -> list[BenchmarkRequest]:
    """Build one request per prompt, cycling through VARIED_CONFIGS."""
    return [
        BenchmarkRequest(
            height=VARIED_CONFIGS[i % len(VARIED_CONFIGS)][0],
            width=VARIED_CONFIGS[i % len(VARIED_CONFIGS)][1],
            num_inference_steps=VARIED_CONFIGS[i % len(VARIED_CONFIGS)][2],
            prompt=VARIED_PROMPTS[i],
        )
        for i in range(len(VARIED_PROMPTS))
    ]


def _build_img2img_requests(
    benchmark_items: list[Img2ImgBenchmarkItem],
) -> list[BenchmarkRequest]:
    """Build one request per img2img benchmark item, using image dimensions."""
    requests = []
    for i, item in enumerate(benchmark_items):
        cfg = VARIED_CONFIGS[i % len(VARIED_CONFIGS)]
        data_uri = _image_to_data_uri(item.image)
        if item.output_size is not None:
            out_h, out_w = item.output_size
        else:
            out_h, out_w = item.image.height, item.image.width
        requests.append(
            BenchmarkRequest(
                height=out_h,
                width=out_w,
                num_inference_steps=cfg[2],
                prompt=item.prompt,
                image=item.image,
                label=item.label,
                image_data_uri=data_uri,
            )
        )
    return requests


@dataclass
class TimingResult:
    """Collected timings for a single backend."""

    preprocess_durations: list[float] = field(default_factory=list)
    execute_durations: list[float] = field(default_factory=list)
    total_durations: list[float] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FLUX.2 performance: diffusers vs MAX."
    )
    parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.2-dev",
        help=(
            "HuggingFace model ID to use (e.g. "
            "'black-forest-labs/FLUX.2-dev', "
            "'black-forest-labs/FLUX.2-klein-4B')."
        ),
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=0,
        help="Number of warmup passes over all requests (not timed) per backend.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not set, a random seed is generated.",
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
        "--first-block-caching",
        action="store_true",
        help="Enable first-block caching (FBC) for the MAX diffusion pipeline.",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=0.08,
        help=(
            "Residual threshold for step-cache early stopping in the "
            "denoising loop (default: 0.08)."
        ),
    )
    parser.add_argument(
        "--image-to-image",
        action="store_true",
        help=(
            "Run image-to-image mode instead of text-to-image. Uses "
            "--input-image if provided, otherwise downloads curated "
            "benchmark images at varied dimensions."
        ),
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help=(
            "Path to an input image file for image-to-image mode. "
            "If --image-to-image is set but no path is provided, "
            "curated benchmark images are downloaded instead."
        ),
    )
    parser.add_argument(
        "--taylorseer",
        action="store_true",
        help="Enable TaylorSeer cache optimization for the MAX pipeline.",
    )
    parser.add_argument(
        "--taylorseer-cache-interval",
        type=int,
        default=None,
        help="Steps between full computations for TaylorSeer (model default if unset).",
    )
    parser.add_argument(
        "--taylorseer-warmup-steps",
        type=int,
        default=None,
        help="Warmup steps for TaylorSeer factor gathering (model default if unset).",
    )
    parser.add_argument(
        "--taylorseer-max-order",
        type=int,
        default=None,
        choices=[1, 2],
        help="Taylor expansion order: 1=linear, 2=quadratic (model default if unset).",
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
        choices=[
            "float32",
            "bfloat16",
            "q4_k",
            "q4_0",
            "q6_k",
            "float8_e4m3fn",
            "float4_e2m1fnx2",
            "gptq",
        ],
        help="Weight encoding type (e.g., 'float4_e2m1fnx2' for NVFP4).",
    )
    parser.add_argument(
        "--prefer-module-v3",
        action="store_true",
        help=(
            "Whether to prefer the eager API architecture over the graph API "
            "architecture. When set, the server uses the eager API architecture "
            "when available and falls back to the graph API architecture."
        ),
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Skip saving generated images to disk.",
    )
    parser.add_argument(
        "--model-override",
        type=str,
        action="append",
        default=None,
        help=(
            "Per-component overrides in 'component.field=value' format. "
            "Repeatable. Example: "
            "'transformer.quantization_encoding=float4_e2m1fnx2'."
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


@dataclass
class Img2ImgBenchmarkItem:
    """A single image-to-image benchmark: input image, prompt, and label."""

    image: Image.Image
    prompt: str
    label: str
    output_size: tuple[int, int] | None = None  # (height, width) override


def _load_input_image(args: argparse.Namespace) -> Image.Image:
    """Load an input image for image-to-image mode from --input-image.

    Only used when `--input-image` is explicitly provided.
    """
    img = Image.open(args.input_image).convert("RGB")
    print(
        f"  Loaded input image: {args.input_image} ({img.width}x{img.height})"
    )
    return img


def _load_benchmark_items() -> list[Img2ImgBenchmarkItem]:
    """Download and return all curated img2img benchmark items."""
    items = []
    for entry in IMG2IMG_BENCHMARKS:
        url, prompt, label = entry[0], entry[1], entry[2]
        output_size = entry[3] if len(entry) > 3 else None
        try:
            img = _download_benchmark_image(url, label)
        except Exception as e:
            print(f"  WARNING: Failed to download {label} ({e}), skipping.")
            continue
        items.append(
            Img2ImgBenchmarkItem(
                image=img,
                prompt=prompt,
                label=label,
                output_size=output_size,
            )
        )
    return items


def _image_to_data_uri(img: Image.Image) -> str:
    """Encode a PIL image as a JPEG base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _images_to_jpeg_base64(images: list[Any]) -> list[str]:
    """Convert PIL images to JPEG-encoded base64 strings.

    This mirrors the post-processing that MAX performs internally
    (numpy → PIL → JPEG → base64) so that timing comparisons are fair.
    """
    result = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="jpeg")
        result.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return result


def _save_image_with_retry(
    img: Image.Image,
    path: str,
    max_retries: int = 3,
    delay: float = 1.0,
) -> None:
    """Save a PIL image to disk, retrying on transient I/O errors."""
    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img.save(path)
            return
        except OSError as e:
            if attempt < max_retries - 1:
                print(
                    f"    WARNING: save failed ({e}), retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                print(
                    f"    ERROR: save failed after {max_retries} attempts: {e}"
                )
                raise


def _build_image_filename(
    backend: str,
    width: int,
    height: int,
    steps: int,
    iteration: int,
    args: argparse.Namespace,
) -> str:
    """Build a descriptive image filename encoding all relevant settings."""
    parts = [
        f"iter{iteration}",
        f"output_{backend}_{(args.quantization_encoding or 'bf16')}_seed{args.seed}_{width}x{height}_{steps}steps",
    ]
    if args.first_block_caching:
        parts.append(f"fbc_thresh{args.residual_threshold}")
    if args.taylorseer:
        interval = args.taylorseer_cache_interval or "default"
        warmup = args.taylorseer_warmup_steps or "default"
        order = args.taylorseer_max_order or "default"
        parts.append(f"taylorseer_i{interval}_w{warmup}_o{order}")
    return "_".join(parts) + ".png"


def _load_diffusers_pipeline(model_id: str) -> Any:
    """Load FLUX.2 pipeline via diffusers with optimized attention.

    Uses DiffusionPipeline.from_pretrained which auto-detects the pipeline
    class from the model's model_index.json.  If the required class (e.g.
    Flux2KleinPipeline) is not available in the installed diffusers version,
    the error is re-raised with an actionable message.
    """
    try:
        pipe = diffusers.DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to("cuda")
    except AttributeError as exc:
        raise RuntimeError(
            f"Failed to load {model_id}: {exc}. "
            "The installed diffusers version may not support this model's "
            "pipeline class. Try upgrading diffusers or use --skip-diffusers "
            "to benchmark only the MAX backend."
        ) from exc
    pipe.transformer.set_attention_backend("native")
    # Some text encoders (Klein's Qwen3, FLUX.2-dev's Mistral3) use a
    # threading.Lock in transformers' output_capturing.py which
    # torch.compile(fullgraph=True) cannot trace.  Fall back to
    # fullgraph=False for all text encoders so the lock causes a graph
    # break instead of a hard error while still compiling the rest.
    text_encoder_fullgraph = False
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder = torch.compile(
            pipe.text_encoder,
            mode="max-autotune",
            fullgraph=text_encoder_fullgraph,
        )
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2 = torch.compile(
            pipe.text_encoder_2, mode="max-autotune", fullgraph=True
        )
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae = torch.compile(pipe.vae, mode="max-autotune", fullgraph=True)
    return pipe


def run_diffusers(
    args: argparse.Namespace,
    requests: list[BenchmarkRequest],
    output_dir: str | None = None,
) -> TimingResult:
    """Benchmark FLUX.2 through diffusers. Returns split timings.

    Preprocessing is measured as text encoding (encode_prompt). Execution
    is the remainder: latent prep, denoising loop, VAE decode, and
    JPEG + base64 encoding (to match MAX's post-processing).
    """
    is_img2img = any(req.image is not None for req in requests)
    mode = "image-to-image" if is_img2img else "text-to-image"
    print(f"\n=== Diffusers (PyTorch) backend ({mode}) ===")
    print(f"Loading pipeline ({args.model})...")
    pipe = _load_diffusers_pipeline(args.model)

    # Detect whether the pipeline supports split encode_prompt + prompt_embeds.
    # FLUX.2-dev (Flux2Pipeline) does; Klein (Flux2KleinPipeline) may not.
    has_encode_prompt = hasattr(pipe, "encode_prompt")

    def _make_pipe_kwargs(
        prompt_or_embeds: Any,
        req: BenchmarkRequest,
        generator: torch.Generator,
    ) -> dict[str, Any]:
        """Build kwargs dict for the diffusers pipeline call."""
        if has_encode_prompt and not isinstance(prompt_or_embeds, str):
            kwargs: dict[str, Any] = {"prompt_embeds": prompt_or_embeds}
        else:
            kwargs = {"prompt": prompt_or_embeds}
        kwargs.update(
            {
                "num_inference_steps": req.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "generator": generator,
            }
        )
        if req.image is not None:
            kwargs["image"] = req.image
        else:
            kwargs["height"] = req.height
            kwargs["width"] = req.width
        return kwargs

    def _encode_prompt(prompt: str) -> Any:
        """Encode a prompt if supported, otherwise return the raw string."""
        if has_encode_prompt:
            prompt_embeds, _text_ids = pipe.encode_prompt(
                prompt=prompt, device=pipe._execution_device
            )
            return prompt_embeds
        return prompt

    # Warm up using the same split path (encode_prompt + `pipe(prompt_embeds=)`)
    # that we use for timed iterations. If we warmed up with `pipe(prompt=...)`
    # instead, then `torch.compile` would recompile when it first sees the
    # `prompt_embeds` call signature during timed runs, adding ~50s of overhead.
    warmup_requests = _build_warmup_requests(requests, args.num_warmups)
    for i, req in enumerate(warmup_requests):
        print(
            f"  warmup {i + 1}/{len(warmup_requests)}"
            f" ({req.width}x{req.height}, {req.num_inference_steps} steps,"
            f" prompt={_truncate_prompt(req.prompt)!r})"
            + (f" [{req.label}]" if req.label else "")
        )
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        encoded = _encode_prompt(req.prompt)
        output = pipe(**_make_pipe_kwargs(encoded, req, generator))
        _images_to_jpeg_base64(output.images)
        torch.cuda.synchronize()

    result = TimingResult()
    n = len(requests)
    for i, req in enumerate(requests):
        print(
            f"  iter {i + 1}/{n}"
            f" ({req.width}x{req.height}, {req.num_inference_steps} steps,"
            f" prompt={_truncate_prompt(req.prompt)!r})"
            + (f" [{req.label}]" if req.label else "")
        )
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

        # Preprocessing: text encoding
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        encoded = _encode_prompt(req.prompt)
        torch.cuda.synchronize()
        t_preprocess = time.perf_counter() - t0

        # Execution: latent prep + denoising loop + VAE decode + JPEG encode
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        output = pipe(**_make_pipe_kwargs(encoded, req, generator))
        _images_to_jpeg_base64(output.images)
        torch.cuda.synchronize()
        t_execute = time.perf_counter() - t1

        result.preprocess_durations.append(t_preprocess)
        result.execute_durations.append(t_execute)
        result.total_durations.append(t_preprocess + t_execute)

        # Save output and input reference images (outside timing)
        if output_dir is not None and output.images:
            fname = _build_image_filename(
                "torch",
                req.width,
                req.height,
                req.num_inference_steps,
                i,
                args,
            )
            _save_image_with_retry(
                output.images[0], os.path.join(output_dir, fname)
            )
            print(f"    saved: {fname}")
            if req.image is not None:
                ref_fname = f"iter{i}_input_{req.label}.png"
                ref_path = os.path.join(output_dir, ref_fname)
                if not os.path.exists(ref_path):
                    _save_image_with_retry(req.image, ref_path)
                    print(f"    saved input: {ref_fname}")

    # Free GPU memory before MAX runs.  torch.compile caches and
    # torch._dynamo state can hold tens of GBs on the GPU even after
    # deleting the pipeline.  Reset them so the MAX cache sees the full
    # device memory.
    del pipe
    del output, encoded
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()

    return result


def _load_max_pipeline(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """Load FLUX.2 pipeline via MAX."""
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

    arch = PIPELINE_REGISTRY.retrieve_architecture(
        manifest.main_architecture_name,
        prefer_module_v3=args.prefer_module_v3,
        task=PipelineTask.PIXEL_GENERATION,
    )
    assert arch is not None, (
        f"No architecture found in MAX registry for {model_id}."
    )

    # Fill unset denoising-cache fields from the architecture's defaults.
    config = PipelineConfig(
        models=manifest,
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=args.prefer_module_v3,
            denoising_cache=resolve_denoising_cache(
                DenoisingCacheSettings(
                    first_block_caching=args.first_block_caching,
                    taylorseer=args.taylorseer,
                    taylorseer_cache_interval=args.taylorseer_cache_interval,
                    taylorseer_warmup_steps=args.taylorseer_warmup_steps,
                    taylorseer_max_order=args.taylorseer_max_order,
                ),
                arch.denoising_cache_defaults,
                arch_name=arch.name,
            ),
        ),
    )

    max_length = None
    if "tokenizer" in config.models:
        max_length = 512  # Flux2-specific override

    tokenizer_cls = cast(type[PixelGenerationTokenizer], arch.tokenizer_cls)
    tokenizer = tokenizer_cls(
        model_path=model_id,
        pipeline_config=config,
        subfolder="tokenizer",
        max_length=max_length,
    )

    pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
    )
    return pipeline, tokenizer, config


async def _build_max_inputs(
    args: argparse.Namespace,
    tokenizer: Any,
    prompt: str,
    height: int,
    width: int,
    steps: int,
    input_image_data_uri: str | None = None,
) -> tuple[Any, Any]:
    """Build MAX pipeline inputs for a single generation.

    When input_image_data_uri is provided the request is constructed with
    a `UserMessage` containing both text and image content, triggering the
    image-to-image code path.
    """
    if input_image_data_uri is not None:
        # Image-to-image: wrap prompt + image in a UserMessage.
        request_input: Any = [
            UserMessage(
                role="user",
                content=[
                    InputTextContent(type="input_text", text=prompt),
                    InputImageContent(
                        type="input_image",
                        image_url=input_image_data_uri,
                    ),
                ],
            )
        ]
    else:
        request_input = prompt

    body = OpenResponsesRequestBody(
        model=args.model,
        input=request_input,
        seed=args.seed,
        provider_options=ProviderOptions(
            image=ImageProviderOptions(
                height=height,
                width=width,
                steps=steps,
                guidance_scale=args.guidance_scale,
            )
        ),
    )
    request = OpenResponsesRequest(request_id=RequestID(), body=body)
    context = await tokenizer.new_context(request)
    inputs = PixelGenerationInputs[PixelContext](
        batch={context.request_id: context}
    )
    return inputs, context


def run_max(
    args: argparse.Namespace,
    requests: list[BenchmarkRequest],
    output_dir: str | None = None,
) -> TimingResult:
    """Benchmark FLUX.2 through MAX with split preprocess/execute timings."""
    is_img2img = any(req.image is not None for req in requests)
    mode = "image-to-image" if is_img2img else "text-to-image"
    print(f"\n=== MAX backend ({mode}) ===")
    print(f"Loading pipeline ({args.model})...")
    pipeline, tokenizer, _config = _load_max_pipeline(args)

    warmup_requests = _build_warmup_requests(requests, args.num_warmups)
    for i, req in enumerate(warmup_requests):
        print(
            f"  warmup {i + 1}/{len(warmup_requests)}"
            f" ({req.width}x{req.height}, {req.num_inference_steps} steps,"
            f" prompt={_truncate_prompt(req.prompt)!r})"
            + (f" [{req.label}]" if req.label else "")
        )
        inputs, _ = asyncio.run(
            _build_max_inputs(
                args,
                tokenizer,
                req.prompt,
                req.height,
                req.width,
                req.num_inference_steps,
                req.image_data_uri,
            )
        )
        pipeline.execute(inputs)

    result = TimingResult()
    n = len(requests)
    for i, req in enumerate(requests):
        print(
            f"  iter {i + 1}/{n}"
            f" ({req.width}x{req.height}, {req.num_inference_steps} steps,"
            f" prompt={_truncate_prompt(req.prompt)!r})"
            + (f" [{req.label}]" if req.label else "")
        )

        # Preprocessing: tokenization + context + input building
        t0 = time.perf_counter()
        inputs, context = asyncio.run(
            _build_max_inputs(
                args,
                tokenizer,
                req.prompt,
                req.height,
                req.width,
                req.num_inference_steps,
                req.image_data_uri,
            )
        )
        t_preprocess = time.perf_counter() - t0

        # Model execution
        t1 = time.perf_counter()
        outputs = pipeline.execute(inputs)
        t_execute = time.perf_counter() - t1

        result.preprocess_durations.append(t_preprocess)
        result.execute_durations.append(t_execute)
        result.total_durations.append(t_preprocess + t_execute)

        # Save output and input reference images (outside timing)
        if output_dir is not None:
            output = outputs[context.request_id]
            output = asyncio.run(tokenizer.postprocess(output))
            if output.output:
                for img_content in output.output:
                    if (
                        isinstance(img_content, OutputImageContent)
                        and img_content.image_data
                    ):
                        image_bytes = base64.b64decode(img_content.image_data)
                        img = Image.open(io.BytesIO(image_bytes))
                        fname = _build_image_filename(
                            "max",
                            req.width,
                            req.height,
                            req.num_inference_steps,
                            i,
                            args,
                        )
                        _save_image_with_retry(
                            img, os.path.join(output_dir, fname)
                        )
                        print(f"    saved: {fname}")
                        break  # Save only the first image per iteration
            if req.image is not None:
                ref_fname = f"iter{i}_input_{req.label}.png"
                ref_path = os.path.join(output_dir, ref_fname)
                if not os.path.exists(ref_path):
                    _save_image_with_retry(req.image, ref_path)
                    print(f"    saved input: {ref_fname}")

    return result


def _print_summary(label: str, result: TimingResult) -> None:
    """Print timing summary for one backend."""
    n = len(result.total_durations)
    print(f"  {label}:")
    print(f"    iterations : {n}")

    if result.preprocess_durations:
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


def _print_per_iteration(
    label: str,
    result: TimingResult,
    requests: list[BenchmarkRequest],
) -> None:
    """Print per-iteration breakdown with config/prompt details."""
    has_images = any(req.image is not None for req in requests)
    print(f"\n  {label} per-iteration breakdown:")
    header = f"    {'Config':>15s}"
    if has_images:
        header += f"  {'Input Image':>32s}"
    header += (
        f"  {'Prompt':>42s}"
        f"  {'Preprocess':>10s}  {'Execute':>10s}  {'Total':>10s}"
    )
    print(header)
    for i, req in enumerate(requests):
        pp = (
            result.preprocess_durations[i]
            if result.preprocess_durations
            else 0.0
        )
        ex = result.execute_durations[i]
        tot = result.total_durations[i]
        line = (
            f"    {req.width:4d}x{req.height:<4d} {req.num_inference_steps:2d}s"
        )
        if has_images:
            line += f"  {req.label:>32s}"
        line += (
            f"  {_truncate_prompt(req.prompt, 42):>42s}"
            f"  {pp:10.2f}  {ex:10.2f}  {tot:10.2f}"
        )
        print(line)


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

    # Auto-enable image-to-image mode when --input-image is provided.
    if args.input_image and not args.image_to_image:
        args.image_to_image = True

    # Build the benchmark request list based on mode.
    if args.image_to_image:
        if args.input_image:
            # Single user-provided image: run through all prompts with it.
            input_image = _load_input_image(args)
            data_uri = _image_to_data_uri(input_image)
            requests = [
                BenchmarkRequest(
                    height=VARIED_CONFIGS[i % len(VARIED_CONFIGS)][0],
                    width=VARIED_CONFIGS[i % len(VARIED_CONFIGS)][1],
                    num_inference_steps=VARIED_CONFIGS[i % len(VARIED_CONFIGS)][
                        2
                    ],
                    prompt=VARIED_PROMPTS[i],
                    image=input_image,
                    label=Path(args.input_image).stem,
                    image_data_uri=data_uri,
                )
                for i in range(len(VARIED_PROMPTS))
            ]
        else:
            print("\n  Downloading curated img2img benchmark images...")
            benchmark_items = _load_benchmark_items()
            print(
                f"  Loaded {len(benchmark_items)} benchmark image+prompt pairs"
            )
            requests = _build_img2img_requests(benchmark_items)
    else:
        requests = _build_txt2img_requests()

    # Create output directory for saved images (unless --no-output).
    output_dir: str | None = None
    if not args.no_output:
        output_dir = os.path.abspath("flux_comparison")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        print(f"  Saving images to: {output_dir}/")

    # Write a prompts.txt manifest so each iteration's prompt is easy to look up.
    if output_dir is not None:
        prompts_path = os.path.join(output_dir, "prompts.txt")
        with open(prompts_path, "w") as f:
            f.write(f"seed: {args.seed}\n")
            f.write(f"reproduce with: --seed {args.seed}\n\n")
            for i, req in enumerate(requests):
                f.write(
                    f"iter {i}: {req.width}x{req.height},"
                    f" {req.num_inference_steps} steps\n"
                )
                if req.label:
                    f.write(f"  input: {req.label}\n")
                f.write(f"  prompt: {req.prompt}\n\n")
        print(f"  Saved prompt manifest: {prompts_path}")

    diffusers_result: TimingResult | None = None
    max_result: TimingResult | None = None

    if not args.skip_diffusers:
        try:
            diffusers_result = run_diffusers(args, requests, output_dir)
        except Exception as e:
            print(f"ERROR running diffusers: {e}", file=sys.stderr)
            traceback.print_exc()
            # Ensure GPU memory is freed so the MAX run can proceed.
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if not args.skip_max:
        try:
            max_result = run_max(args, requests, output_dir)
        except Exception as e:
            print(f"ERROR running MAX: {e}", file=sys.stderr)
            traceback.print_exc()

    # Summary header
    is_img2img = any(req.image is not None for req in requests)
    mode = "Image-to-Image" if is_img2img else "Text-to-Image"
    print("\n" + "=" * 60)
    model_name = (
        args.model.rsplit("/", 1)[-1] if "/" in args.model else args.model
    )
    print(
        f"{model_name} {mode} Performance Comparison — {datetime.now():%Y-%m-%d}"
    )
    print("=" * 60)
    _print_gpu_info()
    print(f"  model            : {args.model}")
    print(f"  seed             : {args.seed}")
    print(f"  mode             : {mode}")
    if is_img2img:
        img_requests = [r for r in requests if r.image is not None]
        unique_labels = dict.fromkeys(r.label for r in img_requests)
        if len(unique_labels) == 1:
            r = img_requests[0]
            print(
                f"  input image      : {r.label}"
                f" ({r.image.width}x{r.image.height})"  # type: ignore[union-attr]
            )
        else:
            print(
                f"  input images     : {len(unique_labels)} curated"
                " benchmark pairs (varied dimensions)"
            )
            for r in img_requests:
                if r.label in unique_labels:
                    del unique_labels[r.label]
                    print(
                        f"    {r.label:30s}  {r.image.width}x{r.image.height}"  # type: ignore[union-attr]
                    )
    print("  prompts          : varied (seq-length stress test)")
    print("  inputs           : varied (recompilation stress test)")
    print(f"  guidance scale   : {args.guidance_scale}")
    print(f"  enable FBC       : {args.first_block_caching}")
    print(f"  residual thresh  : {args.residual_threshold}")
    print(f"  taylorseer       : {args.taylorseer}")
    if args.taylorseer:
        print(
            f"  ts interval      : {args.taylorseer_cache_interval or 'model-default'}"
        )
        print(
            f"  ts warmup        : {args.taylorseer_warmup_steps or 'model-default'}"
        )
        print(
            f"  ts max order     : {args.taylorseer_max_order or 'model-default'}"
        )
    print(f"  warmup runs      : {args.num_warmups}")
    print(f"  timed iterations : {len(requests)}")
    print()
    print("  Torch config:")
    print("    mode           : torch.compile (max-autotune)")
    print("    dtype          : BF16")
    print()
    print("  MAX config:")
    max_dtype = (
        args.quantization_encoding.upper()
        if args.quantization_encoding
        else "BF16"
    )
    print(f"    dtype          : {max_dtype}")
    caching_parts: list[str] = []
    if args.first_block_caching:
        caching_parts.append(
            f"first block cache (threshold: {args.residual_threshold})"
        )
    if args.taylorseer:
        caching_parts.append("taylorseer")
    print(f"    caching        : {', '.join(caching_parts) or 'none'}")
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
    return 0


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    raise SystemExit(main())
