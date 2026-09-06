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

"""Simple offline video generation example using diffusion models.

Usage:
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=100 \
    ./bazelw run //max/examples/diffusion:simple_offline_video_generation -- \
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --prompt "A cat playing piano" \
        --output output.mp4

Note: MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=100 is required
for 720p+ resolutions with symbolic seq_len block graphs.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Any, cast

# Suppress noisy HTTP/download logs
import warnings

for _logger_name in ("httpx", "huggingface_hub", "hf_xet", "urllib3"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import numpy as np
from max.driver import DeviceSpec
from max.examples.diffusion.profiler import profile_execute
from max.interfaces import (
    PipelineTask,
    RequestID,
)
from max.interfaces.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
    VideoProviderOptions,
)
from max.interfaces.request import OpenResponsesRequest
from max.interfaces.request.open_responses import (
    OpenResponsesRequestBody,
)
from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
from max.pipelines.core import PixelContext
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.interfaces import DiffusionPipeline
from max.pipelines.lib.pipeline_variants.pixel_generation import (
    PixelGenerationPipeline,
)

logging.basicConfig(
    level=logging.INFO, format="%(name)s %(levelname)s %(message)s"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate videos with a diffusion model.",
    )
    parser.add_argument("--model", required=True, help="Model identifier.")
    parser.add_argument(
        "--prompt", required=True, help="Text prompt for video generation."
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, blurry, distorted, deformed, ugly, bad, poor, worst quality",
        help="Negative prompt to guide what NOT to generate.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Video height in pixels. Auto-computed from input image if omitted.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Video width in pixels. Auto-computed from input image if omitted.",
    )
    parser.add_argument(
        "--num-frames", type=int, default=81, help="Number of video frames."
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=40,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--guidance-scale-2",
        type=float,
        default=3.0,
        help="Secondary guidance scale for low-noise expert (MoE models).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output video filename.",
    )
    parser.add_argument(
        "--fps", type=int, default=16, help="Frames per second."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum length of tokenizer.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of iterations to run (for benchmarking).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=False,
        help="Run a small warmup (480x832, 17 frames, 2 steps) before timed runs.",
    )
    parser.add_argument(
        "--lora-repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID for LoRA weights.",
    )
    parser.add_argument(
        "--lora-subfolder",
        type=str,
        default=None,
        help="Subfolder in the LoRA repo containing safetensors files.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA strength multiplier for primary transformer.",
    )
    parser.add_argument(
        "--lora-scale-2",
        type=float,
        default=None,
        help="LoRA strength for transformer_2. Defaults to --lora-scale.",
    )
    parser.add_argument(
        "--lora-weight-name",
        type=str,
        default=None,
        help="Single LoRA safetensors filename (for non-MoE models).",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="Path or URL to input image for I2V (image-to-video) generation.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="*",
        default=None,
        help="Multiple WxHxF specs (e.g. 1280x720x81 720x1280x81). "
        "Runs each resolution sequentially in the same process for "
        "recompilation testing. Overrides --width/--height/--num-frames.",
    )

    args = parser.parse_args(argv)
    assert args.prompt, "Prompt must be a non-empty string."

    # --resolutions overrides --height/--width/--num-frames;
    # skip auto-compute and validation here (handled in generate_video).
    if not args.resolutions:
        if args.height is None or args.width is None:
            if args.input_image:
                from PIL import Image

                if args.input_image.startswith(("http://", "https://")):
                    import io
                    import urllib.request

                    with urllib.request.urlopen(args.input_image) as _resp:
                        img = Image.open(io.BytesIO(_resp.read()))
                else:
                    img = Image.open(args.input_image)
                aspect_ratio = img.height / img.width
                max_area = 720 * 1280
                mod_value = 16
                h = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                w = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
                if args.height is None:
                    args.height = h
                if args.width is None:
                    args.width = w
                print(
                    f"Auto-computed resolution: {args.width}x{args.height}"
                    f" (from {img.size[0]}x{img.size[1]})"
                )
            else:
                if args.height is None:
                    args.height = 720
                if args.width is None:
                    args.width = 1280

        assert args.height > 0, "Height must be positive."
        assert args.width > 0, "Width must be positive."
        assert args.num_frames > 0, "num-frames must be positive."
    assert args.num_inference_steps > 0, "num-inference-steps must be positive."
    assert args.guidance_scale > 0.0, "guidance-scale must be positive."
    return args


def save_video(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    """Encode frames to mp4 using PyAV (no system ffmpeg needed)."""
    import av
    import av.video

    if not frames:
        print("ERROR: No frames to save")
        return

    h, w = frames[0].shape[:2]
    container = av.open(output_path, mode="w")
    stream: av.video.VideoStream = container.add_stream("libx264", rate=fps)  # type: ignore[assignment]
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.codec_context.options = {"crf": "18", "preset": "medium"}

    for arr in frames:
        frame = av.VideoFrame.from_ndarray(arr.astype(np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"Video saved to: {output_path}")


def _video_frames_from_raw_output(images: np.ndarray) -> list[np.ndarray]:
    """Convert raw pipeline video output [B, C, T, H, W] to uint8 RGB frames."""
    if images.ndim != 5:
        raise ValueError(
            f"Expected video output with rank 5 [B, C, T, H, W], got {images.shape}."
        )

    video = (images[0] * 0.5 + 0.5).clip(min=0.0, max=1.0)
    video = np.transpose(video, (1, 2, 3, 0))
    video = (video * 255.0).round().astype(np.uint8, copy=False)
    return [video[t] for t in range(video.shape[0])]


def _build_request_body(
    args: argparse.Namespace,
) -> OpenResponsesRequestBody:
    return OpenResponsesRequestBody(
        model=args.model,
        input=args.prompt,
        seed=args.seed,
        provider_options=ProviderOptions(
            image=ImageProviderOptions(
                negative_prompt=args.negative_prompt,
                guidance_scale=args.guidance_scale,
            ),
            video=VideoProviderOptions(
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                steps=args.num_inference_steps,
                num_frames=args.num_frames,
                frames_per_second=getattr(args, "fps", 16),
                guidance_scale_2=getattr(args, "guidance_scale_2", None),
            ),
        ),
    )


def _load_pipeline(
    args: argparse.Namespace,
) -> tuple[PixelGenerationTokenizer, PixelGenerationPipeline[PixelContext]]:
    """Load tokenizer and pipeline from args."""
    print(f"Loading model: {args.model}")

    model_kwargs: dict[str, Any] = {
        "model_path": args.model,
        "device_specs": [DeviceSpec.accelerator()],
    }
    if getattr(args, "quantization_encoding", None):
        model_kwargs["quantization_encoding"] = args.quantization_encoding
    config = PipelineConfig(
        model=MAXModelConfig(**model_kwargs),
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        config.model.huggingface_weight_repo,
        task=PipelineTask.PIXEL_GENERATION,
    )
    assert arch is not None, "No matching diffusion architecture found."

    # I2V auto-routing.
    if arch.name == "WanPipeline" and getattr(args, "input_image", None):
        i2v_arch = PIPELINE_REGISTRY.architectures.get(
            "WanImageToVideoPipeline"
        )
        if i2v_arch is not None:
            print(
                "I2V auto-routing: WanPipeline -> WanI2VPipeline"
                " (--input-image provided)"
            )
            arch = i2v_arch

    # Inject LoRA config.
    diffusers_config = config.model.diffusers_config
    if getattr(args, "lora_repo_id", None) and diffusers_config is not None:
        lora_cfg: dict[str, Any] = {
            "repo_id": args.lora_repo_id,
            "subfolder": getattr(args, "lora_subfolder", "") or "",
            "scale": getattr(args, "lora_scale", 1.0),
            "scale_2": (
                args.lora_scale_2
                if getattr(args, "lora_scale_2", None) is not None
                else getattr(args, "lora_scale", 1.0)
            ),
        }
        if getattr(args, "lora_weight_name", None):
            lora_cfg["filenames"] = [args.lora_weight_name]
        diffusers_config["lora"] = lora_cfg

    # Tokenizer setup.
    # Diffusers pads tokens to 512 but the pipeline trims embeddings to
    # embed_seq_len (226 for Wan) before cross-attention.
    max_length = getattr(args, "max_length", None) or 512
    tokenizer = arch.tokenizer(
        model_path=args.model,
        pipeline_config=config,
        subfolder="tokenizer",
        max_length=max_length,
    )

    # Pipeline setup.
    assert issubclass(arch.pipeline_model, DiffusionPipeline), (
        f"Architecture does not implement DiffusionPipeline: "
        f"{arch.pipeline_model}"
    )
    pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
    )

    print("Initialization complete.")
    return tokenizer, pipeline


def _parse_resolution(spec: str) -> tuple[int, int, int]:
    """Parse 'WxHxF' or 'WxH' spec."""
    parts = spec.split("x")
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    if len(parts) == 2:
        return int(parts[0]), int(parts[1]), 81
    raise ValueError(f"Invalid resolution spec: {spec}. Use WxHxF or WxH.")


async def generate_video(args: argparse.Namespace) -> None:
    # Build list of (width, height, num_frames) to run
    if args.resolutions:
        res_list = [_parse_resolution(r) for r in args.resolutions]
    else:
        res_list = [(args.width, args.height, args.num_frames)]

    # Use first resolution for pipeline loading
    args.width, args.height, args.num_frames = res_list[0]
    tokenizer, pipeline = _load_pipeline(args)

    # Optional warmup with small resolution
    if args.warmup:
        import time as _time

        print("Warmup (480x832, 17 frames, 2 steps)...")
        saved = (args.width, args.height, args.num_frames, args.num_inference_steps)
        args.width, args.height, args.num_frames = 832, 480, 17
        args.num_inference_steps = 2
        t0 = _time.perf_counter()
        await _run_single(args, tokenizer, pipeline, 0, 1)
        print(f"Warmup done: {_time.perf_counter() - t0:.1f}s")
        args.width, args.height, args.num_frames, args.num_inference_steps = saved

    for res_idx, (w, h, nf) in enumerate(res_list):
        args.width, args.height, args.num_frames = w, h, nf
        print(f"\n--- Resolution {res_idx + 1}/{len(res_list)}: {w}x{h}, {nf} frames ---")
        await _run_single(args, tokenizer, pipeline, res_idx, len(res_list))


async def _run_single(
    args: argparse.Namespace,
    tokenizer: PixelGenerationTokenizer,
    pipeline: PixelGenerationPipeline,  # type: ignore[type-arg]
    res_idx: int,
    total_res: int,
) -> None:
    # Create request.
    print(f"Generating video for prompt: '{args.prompt}'")
    print(
        f"Parameters: {args.height}x{args.width}, {args.num_frames} frames, "
        f"steps={args.num_inference_steps}, guidance={args.guidance_scale}"
    )

    body = _build_request_body(args)
    request = OpenResponsesRequest(request_id=RequestID(), body=body)

    # Create context.
    input_image = None
    if args.input_image:
        from PIL import Image

        image_src = args.input_image
        if image_src.startswith(("http://", "https://")):
            import io
            import urllib.request

            with urllib.request.urlopen(image_src) as resp:
                input_image = Image.open(io.BytesIO(resp.read())).convert("RGB")
        else:
            input_image = Image.open(image_src).convert("RGB")
        print(
            f"Input image: {image_src}"
            f" ({input_image.size[0]}x{input_image.size[1]})"
        )
    # Execute (with profiling).
    print("Running diffusion model...")
    num_iterations = args.num_iterations
    with profile_execute(pipeline, enabled=True) as prof:
        for i in range(num_iterations):
            if num_iterations > 1:
                print(f"Running inference {i + 1} of {num_iterations}")
            context = await tokenizer.new_context(
                request, input_image=input_image
            )
            model_inputs = pipeline._pipeline_model.prepare_inputs(context)
            model_outputs = pipeline._pipeline_model.execute(model_inputs)

    prof.report(unit="ms")

    if not isinstance(model_outputs.images, np.ndarray):
        raise TypeError("Expected raw numpy video output from the pipeline.")

    frames = _video_frames_from_raw_output(model_outputs.images)
    if not frames:
        print("ERROR: No frames generated")
        return

    # Save output.
    output_path = args.output
    if total_res > 1:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_{args.width}x{args.height}x{args.num_frames}{ext}"
    print(f"Saving {len(frames)} frames as video to {output_path}")
    save_video(frames, output_path, args.fps)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        asyncio.run(generate_video(args))
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)
    raise SystemExit(main())
