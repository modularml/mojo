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

import functools
import sys
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

import huggingface_hub
import requests
import torch
from create_pipelines import (
    ImageGenerationOracle,
    MaxPipelineAndTokenizer,
    PipelineOracle,
    TorchModelAndDataProcessor,
    VLLMPipeline,
)
from max import driver, pipelines
from max.pipelines.diffusion.pipeline import (
    PixelGenerationPipeline,
)
from max.pipelines.modeling.types import PipelineTask
from test_common import evaluate, evaluate_embeddings, torch_utils, vllm_utils
from test_common.evaluate import ModelOutput
from typing_extensions import ParamSpec


@contextmanager
def maybe_log_hf_downloads(enable_logging: bool):  # noqa: ANN201
    """Context manager that conditionally logs HuggingFace file downloads."""
    if not enable_logging:
        yield
        return

    original_hf_hub_download = huggingface_hub.hf_hub_download

    def logged_hf_hub_download(*args, **kwargs):  # noqa: ANN202
        repo_id = kwargs.get("repo_id") or (
            args[0] if len(args) > 0 else "unknown"
        )
        filename = kwargs.get("filename") or (
            args[1] if len(args) > 1 else "unknown"
        )
        print(f"Accessing {filename} from {repo_id}")
        result = original_hf_hub_download(*args, **kwargs)
        print(f"-> Located at: {result}\n")
        return result

    huggingface_hub.hf_hub_download = logged_hf_hub_download
    try:
        yield
    finally:
        huggingface_hub.hf_hub_download = original_hf_hub_download


class Flake(Exception):
    """A failure has occurred that appears to be of a temporary nature.

    It is likely that retrying the operation would succeed.
    """


_ParamsT = ParamSpec("_ParamsT")
_ReturnT = TypeVar("_ReturnT")


def _detect_hf_flakes(
    inner: Callable[_ParamsT, _ReturnT],
) -> Callable[_ParamsT, _ReturnT]:
    """Decorator to exit with a distinct status on Hugging Face flake."""

    def is_client_error(exc: requests.RequestException) -> bool:
        if not isinstance(exc, requests.HTTPError):
            return False
        if exc.response is None:
            return False
        # 429 is transient rate limiting, not a permanent client fault.
        if exc.response.status_code == 429:
            return False
        # 4xx status codes indicate client error.
        return 400 <= exc.response.status_code < 500

    def get_all_exceptions_in_chain(
        exc: Exception,
    ) -> list[Exception]:
        """Gets all exceptions in the exception chain."""
        to_visit = [exc]
        visited = set()
        all_exceptions = []

        while to_visit:
            current_exc = to_visit.pop(0)

            if id(current_exc) in visited:
                continue
            visited.add(id(current_exc))

            all_exceptions.append(current_exc)

            cause = current_exc.__cause__
            if cause is not None and isinstance(cause, Exception):
                to_visit.append(cause)

            context = current_exc.__context__
            if context is not None and isinstance(context, Exception):
                to_visit.append(context)

        return all_exceptions

    @functools.wraps(inner)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        try:
            return inner(*args, **kwargs)
        except Exception as exc:
            request_exceptions = [
                e
                for e in get_all_exceptions_in_chain(exc)
                if isinstance(e, requests.RequestException)
            ]
            for req_exc in request_exceptions:
                if (
                    req_exc.request is not None
                    and req_exc.request.url is not None
                    and "huggingface.co" in req_exc.request.url
                    and not is_client_error(req_exc)
                ):
                    # This is probably a Hugging Face flake.
                    print(
                        "Seems like a Hugging Face flake has occurred:",
                        file=sys.stderr,
                    )
                    traceback.print_exc()
                    print(
                        "-- End of Hugging Face flake traceback --",
                        file=sys.stderr,
                    )
                    raise Flake("Hugging Face API flake detected") from exc
            raise

    return wrapper


def get_max_default_encoding(
    pipeline_oracle: PipelineOracle,
    pipeline_name: str,
    device_specs: list[driver.DeviceSpec],
) -> pipelines.SupportedEncoding:
    """Determine a suitable default encoding for MAX given the pipeline and devices.

    Preference order:
    1) device_encoding_map on the pipeline oracle for the current device type
    2) architecture default encoding derived from the model repo
    """
    # Get trust_remote_code from pipeline_oracle if available
    trust_remote_code = getattr(pipeline_oracle, "config_params", {}).get(
        "trust_remote_code", False
    )
    model_config = pipelines.MAXModelConfig(
        model_path=pipeline_name, trust_remote_code=trust_remote_code
    )
    arch_name = model_config.architecture_name
    if arch_name is None:
        raise ValueError("Model architecture not yet supported by MAX.")
    arch = pipelines.PIPELINE_REGISTRY.retrieve_architecture(arch_name)
    if arch is None:
        raise ValueError("Model architecture not yet supported by MAX.")

    # Prefer encoding from device_encoding_map if available
    device_encoding_map = getattr(pipeline_oracle, "device_encoding_map", None)
    if device_encoding_map:
        # Determine device type from device_specs
        device_type = device_specs[0].device_type if device_specs else "default"
        # Normalize "default" to "gpu" (default typically means GPU when available)
        if device_type == "default":
            device_type = "gpu"

        # Get encodings for this device type
        encodings = device_encoding_map.get(device_type)
        if encodings and len(encodings) > 0:
            return encodings[0]
        # Fall back to architecture default
        return arch.default_encoding
    # Fall back to architecture default if no device_encoding_map
    return arch.default_encoding


def run_max_model(
    *,
    task: PipelineTask,
    max_pipeline_and_tokenizer: MaxPipelineAndTokenizer,
    inputs: list[Any],
    num_steps: int,
    evaluation_batch_size: int | list[int],
    reference: list[ModelOutput] | None,
    generate_logprobs: bool = False,
) -> Any:
    if task == PipelineTask.TEXT_GENERATION:
        assert isinstance(
            max_pipeline_and_tokenizer.pipeline,
            pipelines.TextGenerationPipelineInterface,
        )
        results = evaluate.run_model(
            max_pipeline_and_tokenizer.pipeline,
            max_pipeline_and_tokenizer.tokenizer,
            requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
            batch_size=evaluation_batch_size,
            reference=reference,
            generate_logprobs=generate_logprobs,
        )
    elif task == PipelineTask.EMBEDDINGS_GENERATION:
        assert isinstance(
            max_pipeline_and_tokenizer.pipeline,
            pipelines.EmbeddingsPipeline,
        )
        if not isinstance(evaluation_batch_size, int):
            raise ValueError(
                "Data parallel mode not supported for embeddings generation."
            )
        results = evaluate_embeddings.encode(
            max_pipeline_and_tokenizer.pipeline,
            max_pipeline_and_tokenizer.tokenizer,
            prompts=(inp.prompt for inp in inputs),
            batch_size=evaluation_batch_size,
        )
    elif task == PipelineTask.PIXEL_GENERATION:
        assert isinstance(
            max_pipeline_and_tokenizer.pipeline,
            PixelGenerationPipeline,
        )
        results = evaluate.run_pixel_generation(
            max_pipeline_and_tokenizer.pipeline,
            max_pipeline_and_tokenizer.tokenizer,
            requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
        )
    else:
        raise ValueError(f"Evaluating task {task} is not supported.")
    return results


def get_torch_device(device_specs: list[driver.DeviceSpec]) -> torch.device:
    """Return the primary torch.device to use based on the provided device specs."""
    if device_specs[0].device_type == "cpu":
        return torch.device("cpu")
    if device_specs[0].device_type == "gpu":
        return torch.device("cuda:0")
    if device_specs[0].device_type == "default":
        return (
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    return torch.device("cpu")


def run_torch_model(
    *,
    pipeline_oracle: PipelineOracle,
    torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
    device: torch.device,
    inputs: list[Any],
    num_steps: int,
    generate_logprobs: bool = False,
) -> Any:
    if pipeline_oracle.task == PipelineTask.TEXT_GENERATION:
        results = pipeline_oracle.run_torch_text_generation(
            torch_pipeline_and_tokenizer=torch_pipeline_and_tokenizer,
            device=device,
            num_steps=num_steps,
            inputs=inputs,
            generate_logprobs=generate_logprobs,
        )
    elif pipeline_oracle.task == PipelineTask.EMBEDDINGS_GENERATION:
        # Get pool_embeddings from oracle config if it has config_params (GenericOracle)
        pool_embeddings = getattr(pipeline_oracle, "config_params", {}).get(
            "pool_embeddings", False
        )
        results = torch_utils.run_embeddings_generation(
            model=torch_pipeline_and_tokenizer.model,
            data_processor=torch_pipeline_and_tokenizer.data_processor,
            device=device,
            prompts=(inp.prompt for inp in inputs),
            pool_embeddings=pool_embeddings,
        )
    elif pipeline_oracle.task == PipelineTask.PIXEL_GENERATION:
        assert isinstance(pipeline_oracle, ImageGenerationOracle)
        results = pipeline_oracle.run_torch_image_generation(
            torch_pipeline_and_tokenizer=torch_pipeline_and_tokenizer,
            device=device,
            num_steps=num_steps,
            inputs=inputs,
        )
    else:
        raise ValueError(
            f"Evaluating task {pipeline_oracle.task} is not supported."
        )

    return results


def run_vllm_model(
    *,
    pipeline_oracle: PipelineOracle,
    vllm_pipeline: VLLMPipeline,
    inputs: list[Any],
    num_steps: int,
    max_batch_size: int | None = None,
) -> Any:
    """Runs the model using the vLLM engine.

    NOTE: Unlike the Torch runner, this execution path treats the model as a
    black box. It does not support `TorchPrintHook` or intermediate layer
    inspection for debugging. This is not by choice; it's due to some
    limitations in the vLLM API.
    """
    if pipeline_oracle.task == PipelineTask.TEXT_GENERATION:
        # Note: vllm_pipeline is just a config object.
        # The heavy lifting happens inside vllm_utils.
        results = vllm_utils.run_text_generation(
            model_path=vllm_pipeline.model_path,
            textgen_requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
            encoding_name=vllm_pipeline.encoding,
            trust_remote_code=vllm_pipeline.trust_remote_code,
            max_batch_size=max_batch_size,
            tensor_parallel_size=vllm_pipeline.tensor_parallel_size,
            extra_kwargs=vllm_pipeline.extra_kwargs,
            mm_data_key=vllm_pipeline.mm_data_key,
        )
    else:
        raise NotImplementedError(
            f"Task {pipeline_oracle.task} not yet supported for vLLM runner."
        )

    return results
