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

"""Simple offline pixel generation example using diffusion models.

This module demonstrates end-to-end pixel generation using:
- OpenResponsesRequest: Create generation requests with prompts
- PixelGenerationTokenizer: Tokenize prompts and prepare model context
- PixelGenerationPipeline: Execute the diffusion model to generate pixels

Usage:
    ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
        --model black-forest-labs/FLUX.2-dev \
        --prompt "A cat in a garden"

    # NVFP4 quantized model (two-repo layout: base model + quantized weights):
    ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
        --model black-forest-labs/FLUX.2-dev \
        --weight-path black-forest-labs/FLUX.2-dev-NVFP4/flux2-dev-nvfp4.safetensors \
        --quantization-encoding float4_e2m1fnx2 \
        --prompt "A cat in a garden"

    ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
        --model black-forest-labs/FLUX.2-dev \
        --prompt "A cat in a garden" \
        --prefer-module-v3
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from max.driver import DeviceSpec
from max.examples.diffusion.profiler import profile_execute
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
from max.pipelines.lib.pipeline_executor import PipelineExecutor
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.modeling.types import (
    PipelineTask,
    PixelGenerationInputs,
    RequestID,
)
from max.pipelines.request import OpenResponsesRequest
from max.pipelines.request.open_responses import (
    InputFileContent,
    InputImageContent,
    InputTextContent,
    InputVideoContent,
    OpenResponsesRequestBody,
    OutputImageContent,
    UserMessage,
)
from max.pipelines.request.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
)
from PIL import Image

_FLUX2_ARCH_NAMES = {
    "Flux2Pipeline",
    "Flux2KleinPipeline",
}
QWEN_IMAGE_ARCH_NAMES = {
    "QwenImagePipeline",
    "QwenImageEditPipeline",
    "QwenImageEditPlusPipeline",
}
QWEN_IMAGE_EDIT_ARCH_NAMES = {
    "QwenImageEditPipeline",
    "QwenImageEditPlusPipeline",
}
QWEN_DEFAULT_GUIDANCE_SCALE = 1.0
QWEN_DEFAULT_TRUE_CFG_SCALE = 4.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the pixel generation example.

    Args:
        argv: Optional explicit list of argument strings. If None, arguments
            are read from sys.argv[1:].

    Returns:
        An argparse.Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate images with a diffusion model.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Identifier of the model to use for generation (e.g., black-forest-labs/FLUX.2-dev).",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt describing the image to generate.",
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
        help="Weight encoding type (e.g., 'bfloat16', 'float4_e2m1fnx2' for NVFP4).",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt to guide what NOT to generate.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of generated image in pixels. None uses model's native resolution.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width of generated image in pixels. None uses model's native resolution.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps. More steps = higher quality but slower.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help=(
            "Guidance scale for classifier-free guidance. "
            "If omitted, defaults to 1.0 for QwenImage family and 3.5 otherwise."
        ),
    )
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=None,
        help=(
            "True classifier-free guidance scale. "
            "If omitted, defaults to 4.0 for QwenImage family when negative prompt is provided, "
            "and 1.0 otherwise."
        ),
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.6,
        help="Image-to-image strength in (0, 1]. Ignored for text-to-image.",
    )
    parser.add_argument(
        "--cfg-normalization",
        action="store_true",
        help="Enable CFG output renormalization when supported by the selected model.",
    )
    parser.add_argument(
        "--cfg-truncation",
        type=float,
        default=1.0,
        help="CFG truncation threshold in normalized time when supported by the selected model (> 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output filename for the generated image.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum length of tokenizer",
    )
    parser.add_argument(
        "--secondary-max-length",
        type=int,
        default=None,
        help="Maximum length of secondary tokenizer",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        action="append",
        default=None,
        help="Input image for image-to-image generation. Can be specified multiple times.",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Profile timings of the pipeline.",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=0,
        help=(
            "Number of warmup iterations to run before the timed execution. "
            "Use >=1 to pre-compile JIT graphs and obtain steady-state timings."
        ),
    )
    parser.add_argument(
        "--num-profile-iterations",
        type=int,
        default=3,
        help="Number of iterations to run for profiling.",
    )
    parser.add_argument(
        "--first-block-caching",
        action="store_true",
        help="Enable first-block step cache optimization.",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=None,
        help="Relative-difference threshold for step cache.",
    )
    parser.add_argument(
        "--taylorseer",
        action="store_true",
        help="Enable TaylorSeer cache optimization.",
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
        "--prefer-module-v3",
        action="store_true",
        help="Prefer the ModuleV3 implementation when the selected model provides one.",
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

    args = parser.parse_args(argv)

    # Validate arguments
    assert args.prompt, "Prompt must be a non-empty string."
    if args.height is not None:
        assert args.height > 0, "Height must be a positive integer."
    if args.width is not None:
        assert args.width > 0, "Width must be a positive integer."
    assert args.num_inference_steps > 0, (
        "num-inference-steps must be a positive integer."
    )
    if args.guidance_scale is not None:
        assert args.guidance_scale > 0.0, "guidance-scale must be positive."
    if args.true_cfg_scale is not None:
        assert args.true_cfg_scale > 0.0, "true-cfg-scale must be positive."
    assert 0.0 < args.strength <= 1.0, "strength must be in (0, 1]."
    assert args.cfg_truncation > 0.0, "cfg-truncation must be positive."
    if args.residual_threshold is not None:
        assert args.residual_threshold >= 0.0, (
            "residual-threshold must be non-negative."
        )
    if (
        hasattr(args, "taylorseer_cache_interval")
        and args.taylorseer_cache_interval is not None
    ):
        assert args.taylorseer_cache_interval >= 1, (
            "taylorseer-cache-interval must be >= 1."
        )
    if (
        hasattr(args, "taylorseer_warmup_steps")
        and args.taylorseer_warmup_steps is not None
    ):
        assert args.taylorseer_warmup_steps >= 1, (
            "taylorseer-warmup-steps must be >= 1."
        )

    return args


def save_image(image_data: str, output_path: str) -> None:
    """Save base64-encoded image data to a file.

    Args:
        image_data: Base64-encoded image data string
        output_path: Path where the image should be saved

    Raises:
        ImportError: If PIL is not available
    """
    try:
        from PIL import Image

        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image.save(output_path)
        print(f"Image saved to: {output_path}")
    except ImportError:
        print("WARNING: PIL not available, cannot save image")
        print(f"Base64 data length: {len(image_data)} chars")


def load_image_as_data_uri(image_path: str | None) -> str | None:
    """Load an image from a file and convert to base64 data URI.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 data URI string, or None if no path provided.
    """
    if image_path is None:
        return None

    # Load image
    image = Image.open(image_path)

    # Convert to bytes
    buffer = BytesIO()
    image_format = image.format or "PNG"
    image.save(buffer, format=image_format)
    image_bytes = buffer.getvalue()

    # Encode as base64
    base64_data = base64.b64encode(image_bytes).decode("utf-8")

    # Determine MIME type
    mime_type = f"image/{image_format.lower()}"

    # Return as data URI
    return f"data:{mime_type};base64,{base64_data}"


async def generate_image(args: argparse.Namespace) -> None:
    """Main generation logic.

    Args:
        args: Parsed command-line arguments
    """
    print(f"Loading model: {args.model}")

    # Step 1: Initialize pipeline configuration
    # Use from_model_path to get full diffusers component expansion.
    manifest = ModelManifest.from_model_path(
        args.model,
        device_specs=[DeviceSpec.accelerator()],
    )

    # Apply legacy single-component overrides for backward compat.
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
        "No matching diffusion architecture found for the provided model."
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

    # Step 2: Initialize the tokenizer
    # The tokenizer handles prompt encoding and context preparation
    models = config.models
    has_tokenizer_2 = "tokenizer_2" in models
    max_length = args.max_length
    secondary_max_length = args.secondary_max_length
    if max_length is None and "tokenizer" in models:
        if arch.name in _FLUX2_ARCH_NAMES or arch.name == "ZImagePipeline":
            max_length = 512
        elif arch.name in QWEN_IMAGE_ARCH_NAMES:
            max_length = 512
        else:
            # Load tokenizer_config.json directly — tokenizer subfolders
            # don't carry a HuggingFace model config.json.
            from huggingface_hub import hf_hub_download

            tok_cfg_path = hf_hub_download(
                repo_id=args.model,
                filename="tokenizer/tokenizer_config.json",
            )
            with open(tok_cfg_path) as f:
                tok_cfg = json.load(f)
            max_length = tok_cfg.get("model_max_length", None)
        print(f"Using max length: {max_length} for tokenizer")

    if secondary_max_length is None and has_tokenizer_2:
        from huggingface_hub import hf_hub_download

        tok2_cfg_path = hf_hub_download(
            repo_id=args.model,
            filename="tokenizer_2/tokenizer_config.json",
        )
        with open(tok2_cfg_path) as f:
            tok2_cfg = json.load(f)
        secondary_max_length = tok2_cfg.get("model_max_length", None)
        print(
            f"Using secondary max length: {secondary_max_length} for tokenizer_2"
        )

    tokenizer_cls = cast(type[PixelGenerationTokenizer], arch.tokenizer_cls)
    tokenizer = tokenizer_cls(
        model_path=args.model,
        pipeline_config=config,
        subfolder="tokenizer",  # Tokenizer is in a subfolder for diffusion models
        max_length=max_length,
        subfolder_2="tokenizer_2" if has_tokenizer_2 else None,
        secondary_max_length=secondary_max_length if has_tokenizer_2 else None,
    )

    # Step 3: Initialize the pipeline
    # The pipeline executes the diffusion model. PixelGenerationPipeline
    # accepts either a DiffusionPipeline (ModuleV3) or a PipelineExecutor
    # (ModuleV2) subclass, so the type check here mirrors that.
    if msg := getattr(arch.pipeline_model, "not_implemented_message", None):
        raise NotImplementedError(msg)
    if not issubclass(
        arch.pipeline_model, (DiffusionPipeline, PipelineExecutor)
    ):
        raise TypeError(
            "Selected architecture does not implement DiffusionPipeline or "
            f"PipelineExecutor: {arch.pipeline_model}"
        )
    pipeline_model = cast(
        "type[DiffusionPipeline | PipelineExecutor[Any, Any, Any]]",
        arch.pipeline_model,
    )
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
    )

    print(f"Generating image for prompt: '{args.prompt}'")

    # Step 4: Create an OpenResponsesRequest
    # Load input images if provided and convert to data URIs
    input_image_data_uris: list[str] = []
    if args.input_image:
        for img_path in args.input_image:
            uri = load_image_as_data_uri(img_path)
            if uri is not None:
                input_image_data_uris.append(uri)

    is_qwen_image_edit_family = arch.name in QWEN_IMAGE_EDIT_ARCH_NAMES
    guidance_scale = args.guidance_scale
    if guidance_scale is None:
        guidance_scale = (
            QWEN_DEFAULT_GUIDANCE_SCALE if is_qwen_image_edit_family else 3.5
        )

    true_cfg_scale = args.true_cfg_scale
    if true_cfg_scale is None:
        if is_qwen_image_edit_family and args.negative_prompt is not None:
            true_cfg_scale = QWEN_DEFAULT_TRUE_CFG_SCALE
        else:
            true_cfg_scale = 1.0

    # Create request with structured message if images are provided
    if input_image_data_uris:
        # Image-to-image: Use structured message with InputImageContent + InputTextContent.
        image_content_items: list[
            InputTextContent
            | InputImageContent
            | InputFileContent
            | InputVideoContent
        ] = [
            InputImageContent(
                type="input_image",
                image_url=uri,
            )
            for uri in input_image_data_uris
        ]
        image_content_items.append(
            InputTextContent(
                type="input_text",
                text=args.prompt,
            )
        )
        body = OpenResponsesRequestBody(
            model=args.model,
            input=[
                UserMessage(
                    role="user",
                    content=image_content_items,
                )
            ],
            seed=args.seed,
            provider_options=ProviderOptions(
                image=ImageProviderOptions(
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    steps=args.num_inference_steps,
                    guidance_scale=guidance_scale,
                    true_cfg_scale=true_cfg_scale,
                    strength=args.strength,
                    cfg_normalization=args.cfg_normalization,
                    cfg_truncation=args.cfg_truncation,
                )
            ),
        )
    else:
        # Text-to-image: Use simple string prompt
        body = OpenResponsesRequestBody(
            model=args.model,
            input=args.prompt,
            seed=args.seed,
            provider_options=ProviderOptions(
                image=ImageProviderOptions(
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    steps=args.num_inference_steps,
                    guidance_scale=guidance_scale,
                    true_cfg_scale=true_cfg_scale,
                    strength=args.strength,
                    cfg_normalization=args.cfg_normalization,
                    cfg_truncation=args.cfg_truncation,
                )
            ),
        )

    request = OpenResponsesRequest(request_id=RequestID(), body=body)

    print(
        "Parameters: "
        f"steps={args.num_inference_steps}, guidance={guidance_scale}, true_cfg={true_cfg_scale}, "
        f"cfg_norm={args.cfg_normalization}, cfg_trunc={args.cfg_truncation}"
    )

    # Step 5: Create a PixelContext object from the request
    # The tokenizer handles prompt tokenization, timestep scheduling,
    # latent initialization, and all other preprocessing
    # Image is now extracted from the message content automatically
    context = await tokenizer.new_context(request)

    print(
        f"Context created: {context.height}x{context.width}, {context.num_inference_steps} steps"
    )
    if args.first_block_caching:
        rdt_info = (
            f", rdt={args.residual_threshold}"
            if args.residual_threshold is not None
            else ""
        )
        print(f"First-block caching enabled{rdt_info}.")
    if args.taylorseer:
        order_info = (
            f"order={args.taylorseer_max_order}"
            if args.taylorseer_max_order is not None
            else "order=model-default"
        )
        interval_info = (
            f"interval={args.taylorseer_cache_interval}"
            if args.taylorseer_cache_interval is not None
            else "interval=model-default"
        )
        warmup_info = (
            f"warmup={args.taylorseer_warmup_steps}"
            if args.taylorseer_warmup_steps is not None
            else "warmup=model-default"
        )
        print(
            f"TaylorSeer enabled: {order_info}, {interval_info}, {warmup_info}."
        )

    # Step 6: Prepare inputs for the pipeline
    # Create a batch with a single context
    inputs = PixelGenerationInputs[PixelContext](
        batch={context.request_id: context}
    )

    # Step 6-1: Warmup — run before profiling or timed execution so that JIT
    # compilation completes and steady-state performance can be measured.
    if args.num_warmups > 0:
        body_warmup = OpenResponsesRequestBody(
            model=args.model,
            input=args.prompt,
            seed=args.seed,
            provider_options=ProviderOptions(
                image=ImageProviderOptions(
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    steps=args.num_inference_steps,
                    guidance_scale=guidance_scale,
                    true_cfg_scale=true_cfg_scale,
                    strength=args.strength,
                    cfg_normalization=args.cfg_normalization,
                    cfg_truncation=args.cfg_truncation,
                )
            ),
        )
        request_warmup = OpenResponsesRequest(
            request_id=RequestID(), body=body_warmup
        )
        warmup_image = (
            Image.open(args.input_image[0]) if args.input_image else None
        )
        context_warmup = await tokenizer.new_context(
            request_warmup, input_image=warmup_image
        )
        inputs_warmup = PixelGenerationInputs[PixelContext](
            batch={context_warmup.request_id: context_warmup}
        )
        for i in range(args.num_warmups):
            print(f"Running warmup {i + 1} of {args.num_warmups}")
            pipeline.execute(inputs_warmup)
        print("Warmup complete")

    # Step 7: Execute the pipeline
    print("Running diffusion model...")
    if args.profile_timings:
        with profile_execute(pipeline) as prof:
            for i in range(args.num_profile_iterations):
                print(
                    f"Running inference {i + 1} of {args.num_profile_iterations}"
                )
                outputs = pipeline.execute(inputs)
        prof.report(unit="ms")
    else:
        start_time = time.perf_counter()
        outputs = pipeline.execute(inputs)
        elapsed = time.perf_counter() - start_time
        print(f"Generation took {elapsed:.3f}s")

    # Step 8: Get the output for our request
    output = outputs[context.request_id]
    output = await tokenizer.postprocess(output)

    # Check if generation completed successfully
    if not output.is_done:
        print(f"WARNING: Generation status: {output.final_status}")
        return

    print("Generation complete!")

    # Step 9: Extract and save images from OutputImageContent
    # The output now contains a list of OutputImageContent objects with base64-encoded images
    if not output.output:
        print("ERROR: No images generated")
        return

    # Save each generated image
    for idx, image_content in enumerate(output.output):
        # Narrow type for mypy - we expect OutputImageContent for pixel generation
        if not isinstance(image_content, OutputImageContent):
            print(
                f"ERROR: Expected OutputImageContent, got {type(image_content)}"
            )
            continue

        # Determine output filename
        if len(output.output) > 1:
            # Multiple images: add index to filename
            base_name, ext = os.path.splitext(args.output)
            output_path = f"{base_name}_{idx}{ext}"
        else:
            output_path = args.output

        # Save the image
        if image_content.image_data:
            save_image(image_content.image_data, output_path)
        elif image_content.image_url:
            print(f"Image available at URL: {image_content.image_url}")
        else:
            print("ERROR: No image data or URL in output")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the pixel generation example.

    Args:
        argv: Optional explicit list of argument strings. If None, arguments
            are read from sys.argv[1:].

    Returns:
        Process exit code. 0 indicates success.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args(argv)

    try:
        asyncio.run(generate_image(args))
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
