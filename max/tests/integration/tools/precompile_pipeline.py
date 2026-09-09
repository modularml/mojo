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

"""Pre-compile a single pipeline using the same code path as verify_pipelines.

This script uses PIPELINE_ORACLES.create_max_pipeline() to compile a model,
ensuring the resulting cache artifacts are identical to what the verification
tests produce. This allows the compile step to run on CPU (with virtual
devices) and the resulting cache to be reused by GPU tests.
"""

from __future__ import annotations

import logging
from typing import cast

import click
from create_pipelines import PIPELINE_ORACLES, GenericOracle
from max._entrypoints.cli.entrypoint import configure_cli_logging
from max.driver import (
    calculate_virtual_device_count_from_cli,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.pipelines.lib.device_specs import (
    device_specs_from_normalized_device_handle,
    normalize_device_specs_input,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from max.serve.config import parse_api_and_target_arch

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--pipeline", required=True, help="Pipeline name (key in PIPELINE_ORACLES)"
)
@click.option(
    "--encoding",
    required=True,
    help="Quantization encoding (e.g. bfloat16, float32)",
)
@click.option(
    "--devices", required=True, help="Devices string (e.g. gpu, gpu:0,1)"
)
@click.option(
    "--target", required=True, help="Target API and arch (e.g. cuda:sm_90a)"
)
def main(pipeline: str, encoding: str, devices: str, target: str) -> None:
    """Pre-compile a model, matching the verify_pipelines code path."""
    configure_cli_logging(level="INFO")

    api, target_arch = parse_api_and_target_arch(target)
    set_virtual_device_api(api)
    set_virtual_device_target_arch(target_arch)
    normalized = normalize_device_specs_input(devices)
    set_virtual_device_count(
        calculate_virtual_device_count_from_cli(normalized)
    )

    device_specs = device_specs_from_normalized_device_handle(normalized)

    if pipeline in PIPELINE_ORACLES:
        oracle = PIPELINE_ORACLES[pipeline]
    else:
        oracle = GenericOracle(model_path=pipeline)

    logger.info(
        "Compiling for target: %s (%s) using virtual devices", api, target_arch
    )

    oracle.create_max_pipeline(
        encoding=cast(SupportedEncoding, encoding), device_specs=device_specs
    )

    logger.info("Precompile complete.")


if __name__ == "__main__":
    main()
