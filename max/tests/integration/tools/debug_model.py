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

from __future__ import annotations

import json
import sys

# Standard library
from pathlib import Path
from typing import Any, cast, get_args

# 3rd-party
import click
import torch
from max import pipelines
from max._entrypoints.cli.config import DevicesOptionType
from max.pipelines.lib.device_specs import (
    _default_device_specs,
    device_specs_from_normalized_device_handle,
    normalize_device_specs_input,
)
from max.tests.integration.tools.debugging_utils import run_debug_model
from max.tests.integration.tools.hf_config_overrides import (
    create_layer_overrides,
)
from run_models import Flake

# This is far from a universal standard, but this is the closest to a standard
# that I could find: BSD-derived programs sometimes use exit codes from
# "sysexits.h", which defines this exit code as "temp failure; user is invited
# to retry".  debug_model will emit this if it detects a failure is
# likely caused by a network flake and could be resolved by a retry.
EX_TEMPFAIL = 75


@click.command()
@click.option(
    "--framework",
    "framework_name",
    type=click.Choice(["max", "torch", "vllm"]),
    default="max",
    help="Framework to run pipeline with",
)
@click.option(
    "--pipeline",
    "pipeline_name",
    type=str,
    required=True,
    help="Pipeline to run. Must be a valid Transformers model path or the key to an existing pipeline oracle.",
)
@click.option(
    "--device",
    "--devices",
    "device_type",
    type=DevicesOptionType(),
    default=None,
    help="Type of device to run pipeline with. Default is to use the first available GPU.",
)
@click.option(
    "--encoding",
    "encoding_name",
    type=click.Choice(get_args(pipelines.SupportedEncoding)),
    required=False,
    help="Quantization encoding to run pipeline with.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory to write intermediate tensors. If omitted, tensors are printed to the console.",
)
@click.option(
    "--max-batch-size",
    "max_batch_size",
    type=int,
    default=None,
    help="The maximum batch size to use when evaluating the model.",
)
@click.option(
    "--log-hf-downloads",
    "log_hf_downloads",
    is_flag=True,
    default=False,
    help="Log HuggingFace file downloads for MAX and Torch models.",
)
@click.option(
    "--num-steps",
    "num_steps",
    type=int,
    default=1,
    help="The number of steps to run the model for (default: 1).",
)
@click.option(
    "--prompt",
    "prompt",
    type=str,
    required=False,
    help="Override the default TEXT prompt (plain text only). For multimodal inputs pass images via --image. If omitted, uses the pipeline's first default prompt.",
)
@click.option(
    "--image",
    "images",
    type=str,
    multiple=True,
    required=False,
    help="Image URL or path for multimodal pipelines. Can be passed multiple times.",
)
@click.option(
    "--hf-config-overrides",
    "hf_config_overrides",
    type=str,
    default=None,
    help="JSON dict of overrides applied to HuggingFace AutoConfig fields.",
)
@click.option(
    "--num-hidden-layers",
    "num_hidden_layers",
    type=str,
    default="1",
    help="Number of hidden layers to use (default: 1). Pass 'all' to use all layers.",
)
@click.option(
    "--prefer-module-v3",
    "prefer_module_v3",
    is_flag=True,
    default=False,
    help=(
        "Use the ModuleV3 (eager-API) architecture variant when available. "
        "Instructs the pipeline registry to look up '<arch>_ModuleV3' instead "
        "of '<arch>'. Has no effect on non-MAX frameworks."
    ),
)
def main(
    device_type: str | list[int] | None,
    framework_name: str,
    pipeline_name: str,
    encoding_name: pipelines.SupportedEncoding | None,
    output_path: Path,
    max_batch_size: int | None,
    log_hf_downloads: bool,
    num_steps: int,
    prompt: str | None,
    images: tuple[str, ...] | None,
    hf_config_overrides: str | None,
    num_hidden_layers: str,
    prefer_module_v3: bool,
) -> None:
    if "gemma3" in pipeline_name:
        # Running into dynamo error:
        # https://huggingface.co/google/gemma-3-4b-it/discussions/51
        torch._dynamo.config.disable = True

    # Validate num_hidden_layers input
    if num_hidden_layers != "all":
        try:
            int(num_hidden_layers)
        except ValueError as e:
            raise click.UsageError(
                f"--num-hidden-layers must be a positive integer or 'all', got: {num_hidden_layers}"
            ) from e

    # Parse user-provided config overrides
    parsed_overrides: dict[str, Any] = {}
    if hf_config_overrides:
        try:
            parsed = json.loads(hf_config_overrides)
            if not isinstance(parsed, dict):
                raise ValueError("hf_config_overrides must be a JSON object.")
            parsed_overrides = cast(dict[str, Any], parsed)
        except Exception as e:
            raise click.UsageError(
                f"Invalid --hf-config-overrides JSON: {e}"
            ) from e

    # Create layer overrides and merge with user overrides (user overrides take precedence)
    layer_overrides = create_layer_overrides(num_hidden_layers, pipeline_name)
    final_overrides = {**layer_overrides, **parsed_overrides} or None

    try:
        run_debug_model(
            device_specs=(
                _default_device_specs()
                if device_type is None
                else device_specs_from_normalized_device_handle(
                    normalize_device_specs_input(device_type)
                )
            ),
            framework_name=framework_name,
            pipeline_name=pipeline_name,
            encoding_name=encoding_name,
            output_path=output_path,
            max_batch_size=max_batch_size,
            log_hf_downloads=log_hf_downloads,
            num_steps=num_steps,
            prompt=prompt,
            images=images,
            hf_config_overrides=final_overrides,
            prefer_module_v3=prefer_module_v3,
        )
    except Flake:
        sys.exit(EX_TEMPFAIL)


if __name__ == "__main__":
    main()
