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

import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

# Standard library
from abc import ABC, abstractmethod
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import diffusers

# 3rd-party
import huggingface_hub
import torch
import transformers
import transformers.integrations.peft as peft_integration
from idefics3 import torch_utils as idefics3_torch_utils
from internvl import torch_utils as internvl_torch_utils
from max import driver, pipelines
from max.pipelines import TextGenerationPipelineInterface
from max.pipelines.architectures.internvl.tokenizer import InternVLProcessor
from max.pipelines.architectures.qwen3.text_encoder import (
    Qwen3TextEncoderKleinModel,
)
from max.pipelines.diffusion.config import DenoisingCacheSettings
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.modeling.types import PipelineTask, PipelineTokenizer
from peft.peft_model import PeftModel
from qwen2_5vl import generate_utils as qwen2_5vl_utils
from qwen3vl import generate_utils as qwen3vl_utils
from test_common import test_data, torch_utils
from test_common.test_data import (
    WAN_PIXEL_GENERATION_T2I,
    MockTextGenerationRequest,
)


# This is required since the presence of peft changes
# code-path that our `transformer` pipelines take.
# More specifically, when `weights_path` are present in an oracle,
# the `UNUSED` value for the model path is used to try and query
# for LoRA specific config. There isn't a good way to disable this path
# since it is always taken when `peft` is available in the env.
@contextmanager
def disable_peft() -> Generator[None, None, None]:
    original_is_peft_available = peft_integration.is_peft_available
    peft_integration.is_peft_available = lambda: False
    try:
        yield
    finally:
        peft_integration.is_peft_available = original_is_peft_available


ENCODING_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "gptq": torch.float16,
    "q4_k": torch.float32,
    "q4_0": torch.float32,
    "q6_k": torch.float32,
}


@dataclass
class MaxPipelineAndTokenizer:
    """An instantiated MAX pipeline and pieces necessary to run it."""

    pipeline: (
        TextGenerationPipelineInterface[Any] | pipelines.EmbeddingsPipeline
    )
    tokenizer: PipelineTokenizer[Any, Any, Any]


@dataclass
class TorchModelAndDataProcessor:
    """An instantiated Torch model and pieces necessary to run it."""

    model: transformers.PreTrainedModel
    data_processor: (
        transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast
        | transformers.MllamaProcessor
        | transformers.PixtralProcessor
        | InternVLProcessor
    )


# Local cache root for model weights synced from S3 (see
# `_sync_s3_model_to_local`).
_S3_MODEL_CACHE_ROOT = os.path.expanduser("~/.cache/modular/s3-models")


def _sync_s3_model_to_local(model_path: str) -> str:
    """Resolve a model path to a local directory, syncing from S3 if needed.

    If ``model_path`` is an ``s3://bucket/prefix`` URI, mirror that prefix to a
    local cache directory with ``aws s3 sync`` and return the local path;
    otherwise return ``model_path`` unchanged. The sync is idempotent (only
    changed objects are transferred) and uses the ambient AWS credentials
    (the standard AWS credential chain, e.g. an ``AWS_PROFILE`` with an active
    ``aws sso login`` session).

    Args:
        model_path: A local path, Hugging Face repo id, or ``s3://`` URI.

    Returns:
        A local filesystem path (the cache dir for ``s3://`` URIs, otherwise
        the input unchanged).
    """
    if not model_path.startswith("s3://"):
        return model_path
    rel = model_path[len("s3://") :].strip("/")
    local_dir = os.path.join(_S3_MODEL_CACHE_ROOT, rel)
    os.makedirs(local_dir, exist_ok=True)
    print(
        f"Syncing model weights from s3://{rel} to {local_dir} ...",
        flush=True,
    )
    subprocess.run(["aws", "s3", "sync", f"s3://{rel}", local_dir], check=True)
    return local_dir


@dataclass
class VLLMPipeline:
    """Configuration to run a vLLM pipeline.

    We do not instantiate the LLM engine here to avoid CUDA context initialization
    in the main process.
    """

    model_path: str
    trust_remote_code: bool = False
    encoding: pipelines.SupportedEncoding | None = None
    tensor_parallel_size: int = 1
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    mm_data_key: str = "image"


class PipelineOracle(ABC):
    """Knows about a kind of pipeline.

    Can provide information about that pipeline, and create other objects
    necessary to run the model.
    """

    model_path: str
    """ID of the Hugging Face repository."""

    task: PipelineTask = PipelineTask.TEXT_GENERATION
    default_batch_size: int | list[int] | None = None

    @property
    @abstractmethod
    def device_encoding_map(self) -> dict[str, list[str]] | None:
        """A dict where the key are the supported device types, and the
        values are lists of supported encodings.

        Example:
            {
                "cpu": ["float32"],
                "gpu": ["bfloat16"]
            }
        """
        raise NotImplementedError

    @abstractmethod
    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        """Instantiate a MAX pipeline for the given encoding/device."""
        raise NotImplementedError

    @abstractmethod
    def create_torch_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device: torch.device | str,
    ) -> TorchModelAndDataProcessor:
        """Instantiate a Torch pipeline for the given encoding/device."""
        raise NotImplementedError

    def create_vllm_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device_specs: list[driver.DeviceSpec],
    ) -> VLLMPipeline:
        """Instantiate a vLLM pipeline config."""
        # Use tensor parallelism across all GPU devices
        gpu_count = sum(1 for d in device_specs if d.device_type == "gpu")
        return VLLMPipeline(
            model_path=self.model_path,
            trust_remote_code=getattr(self, "trust_remote_code", False),
            encoding=encoding,
            tensor_parallel_size=max(1, gpu_count),
        )

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        """Input requests for the model.

        By default, creates text-only requests from test data. Multimodal pipelines
        should override this to include images.
        """
        return test_data.DEFAULT_TEXT_ONLY

    @property
    def use_cache(self) -> bool:
        """Whether to use the KV cache, for HF transformers models only."""
        return True

    def run_torch_text_generation(
        self,
        *,
        torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
        device: torch.device,
        num_steps: int,
        inputs: list[Any],
        generate_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        """Run text generation using the standard torch_utils implementation.

        Can be overridden by subclasses that need custom preprocessing logic.
        """
        return torch_utils.run_text_generation(
            model=torch_pipeline_and_tokenizer.model,
            data_processor=torch_pipeline_and_tokenizer.data_processor,
            device=device,
            textgen_requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
            use_cache=self.use_cache,
            generate_logprobs=generate_logprobs,
        )


def _create_vision_max_pipeline(
    model_path: str,
    encoding: pipelines.SupportedEncoding,
    device_specs: list[driver.DeviceSpec],
    *,
    max_length: int = 8192,
    trust_remote_code: bool = True,
    device_memory_utilization: float | None = None,
    enable_chunked_prefill: bool | None = None,
) -> MaxPipelineAndTokenizer:
    """Shared MAX pipeline construction for vision oracles."""
    if device_memory_utilization is not None:
        kv_cache = pipelines.KVCacheConfig(
            device_memory_utilization=device_memory_utilization,
        )
    else:
        kv_cache = pipelines.KVCacheConfig()
    config = pipelines.PipelineArgs(
        device_specs=device_specs,
        quantization_encoding=encoding,
        model_path=model_path,
        trust_remote_code=trust_remote_code,
        max_length=max_length,
        kv_cache=kv_cache,
        runtime=pipelines.PipelineRuntimeConfig(
            enable_chunked_prefill=(
                enable_chunked_prefill
                if enable_chunked_prefill is not None
                else True
            ),
        ),
    )
    tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(
        PipelineConfig.from_args(config)
    )
    assert isinstance(pipeline, pipelines.TextGenerationPipelineInterface)
    return MaxPipelineAndTokenizer(pipeline, tokenizer)


class InternVLPipelineOracle(PipelineOracle):
    """Pipeline oracle for InternVL3 architectures."""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path
        self.trust_remote_code = True

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {
            "gpu": ["bfloat16"],
        }

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        """Input requests for InternVL."""
        return (
            test_data.DEFAULT_TEXT_ONLY + test_data.INTERNVL_INSTRUCT_REQUESTS
        )

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        return _create_vision_max_pipeline(
            self.model_path,
            encoding,
            device_specs,
            # TODO(GEX-2365): Handle this in model memory estimation.
            device_memory_utilization=0.8,
        )

    def create_torch_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device: torch.device,
    ) -> TorchModelAndDataProcessor:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        config = transformers.AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        processor = InternVLProcessor(tokenizer, config)
        model = transformers.AutoModel.from_pretrained(
            self.model_path,
            config=config,
            device_map=device,
            torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding] if encoding else None,
            trust_remote_code=True,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    def run_torch_text_generation(
        self,
        *,
        torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
        device: torch.device,
        num_steps: int,
        inputs: list[Any],
        generate_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        """Run text generation using InternVL-specific preprocessing logic."""
        return internvl_torch_utils.run_text_generation(
            model=torch_pipeline_and_tokenizer.model,
            processor=torch_pipeline_and_tokenizer.data_processor,
            device=device,
            textgen_requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
            generate_logprobs=generate_logprobs,
            # Omit `use_cache` since the InternVL code hardcodes it.
        )


class Idefics3PipelineOracle(PipelineOracle):
    """Pipeline oracle for Idefics3 architectures."""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {
            "gpu": ["bfloat16"],
        }

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        """Input requests for Idefics3."""

        return (
            test_data.DEFAULT_TEXT_ONLY + test_data.IDEFICS3_INSTRUCT_REQUESTS
        )

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        return _create_vision_max_pipeline(
            self.model_path,
            encoding,
            device_specs,
            # TODO(GEX-2365): Handle this in model memory estimation.
            device_memory_utilization=0.8,
        )

    def create_torch_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device: torch.device,
    ) -> TorchModelAndDataProcessor:
        config = transformers.AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        processor = transformers.AutoProcessor.from_pretrained(self.model_path)
        # Use AutoModelForImageTextToText instead of AutoModel for Idefics3
        # (transformers 5.12 removed AutoModelForVision2Seq).
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            config=config,
            device_map=device,
            torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding] if encoding else None,
            trust_remote_code=True,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    def run_torch_text_generation(
        self,
        *,
        torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
        device: torch.device,
        num_steps: int,
        inputs: list[Any],
        generate_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        """Run text generation using Idefics3-specific preprocessing logic."""

        return idefics3_torch_utils.run_text_generation(
            model=torch_pipeline_and_tokenizer.model,
            data_processor=torch_pipeline_and_tokenizer.data_processor,
            device=device,
            textgen_requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
            use_cache=self.use_cache,
            generate_logprobs=generate_logprobs,
        )


class Qwen2_5VLPipelineOracle(PipelineOracle):
    """Pipeline oracle for Qwen2.5VL architectures."""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {
            "gpu": ["bfloat16"],
        }

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        """Input requests for Qwen2.5VL."""
        # Torch model tries to return EOT for the default long text prompt,
        # so add another bullet point to get it to generate more tokens.
        long_prompt = test_data.LONG_TEXT_PROMPT + "\n    * "
        text_only_prompts = [long_prompt] + list(test_data.SHORT_TEXT_PROMPTS)
        text_only_requests = [
            MockTextGenerationRequest.text_only(prompt)
            for prompt in text_only_prompts
        ]
        return qwen2_5vl_utils.INSTRUCT_REQUESTS + text_only_requests

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        return _create_vision_max_pipeline(
            self.model_path,
            encoding,
            device_specs,
            # Chunked prefill is not supported for image prompts.
            # (technically, this script doesn't go through the scheduler so
            # it's not a problem, but it's a good idea to disable it anyway.)
            enable_chunked_prefill=False,
        )

    def create_torch_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device: torch.device,
    ) -> TorchModelAndDataProcessor:
        config = transformers.AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        processor = transformers.AutoProcessor.from_pretrained(self.model_path)
        model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            config=config,
            device_map=device,
            # Qwen2.5VL 32B uses float32 for the vision model, and bfloat16 for the language model
            # So, we don't set the encoding dtype for the torch model
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    def run_torch_text_generation(
        self,
        *,
        torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
        device: torch.device,
        num_steps: int,
        inputs: list[Any],
        generate_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        """Run text generation using Qwen2.5VL-specific preprocessing logic."""

        return qwen2_5vl_utils.run_text_generation(
            model=torch_pipeline_and_tokenizer.model,
            data_processor=torch_pipeline_and_tokenizer.data_processor,
            device=device,
            textgen_requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
            use_cache=self.use_cache,
            generate_logprobs=generate_logprobs,
        )


class Qwen3VLPipelineOracle(PipelineOracle):
    """Pipeline oracle for Qwen3VL architectures."""

    def __init__(
        self,
        model_path: str,
        torch_model_path: str | None = None,
        device_encoding_map: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        # A quantized checkpoint transformers cannot load names its bf16 source
        # here, so the torch reference is the model MAX's weights were
        # quantized from; the quantization error lands in the tolerances.
        self.torch_model_path = torch_model_path or model_path
        self._device_encoding_map = device_encoding_map or {"gpu": ["bfloat16"]}
        self.trust_remote_code = True

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return self._device_encoding_map

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        """Input requests for Qwen3VL."""
        # Torch model tries to return EOT for the default long text prompt,
        # so add another bullet point to get it to generate more tokens.
        long_prompt = test_data.LONG_TEXT_PROMPT + "\n    * "
        text_only_prompts = [long_prompt] + list(test_data.SHORT_TEXT_PROMPTS)
        text_only_requests = [
            MockTextGenerationRequest.text_only(prompt)
            for prompt in text_only_prompts
        ]
        return qwen3vl_utils.INSTRUCT_REQUESTS + text_only_requests

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        return _create_vision_max_pipeline(
            self.model_path,
            encoding,
            device_specs,
            # Chunked prefill is not supported for image prompts.
            # (technically, this script doesn't go through the scheduler so
            # it's not a problem, but it's a good idea to disable it anyway.)
            enable_chunked_prefill=False,
        )

    def create_torch_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device: torch.device,
    ) -> TorchModelAndDataProcessor:
        config = transformers.AutoConfig.from_pretrained(
            self.torch_model_path, trust_remote_code=True
        )
        processor = transformers.AutoProcessor.from_pretrained(
            self.torch_model_path, trust_remote_code=True
        )
        # Sub-byte and FP8 encodings cannot be a torch default dtype, and the
        # reference checkpoint holds unquantized weights anyway, so the torch
        # side computes in bfloat16.
        if encoding in ("float8_e4m3fn", "float4_e2m1fnx2"):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = (
                ENCODING_TO_TORCH_DTYPE[encoding] if encoding else None
            )
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            self.torch_model_path,
            config=config,
            device_map=device,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    def run_torch_text_generation(
        self,
        *,
        torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
        device: torch.device,
        num_steps: int,
        inputs: list[Any],
        generate_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        """Run text generation using Qwen3VL-specific preprocessing logic."""

        return qwen3vl_utils.run_text_generation(
            model=torch_pipeline_and_tokenizer.model,
            data_processor=torch_pipeline_and_tokenizer.data_processor,
            device=device,
            textgen_requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
            use_cache=self.use_cache,
            generate_logprobs=generate_logprobs,
        )


class PixtralPipelineOracle(PipelineOracle):
    def __init__(self) -> None:
        super().__init__()
        self.model_path = "mistral-experimental/pixtral-12b"
        self.max_length = 8192

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        """Input requests for Pixtral model."""
        return test_data.PIXTRAL_REQUESTS

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {
            "gpu": ["bfloat16"],
        }

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        # TODO (AIPIPE-234): Implement MAX pipeline generation for Pixtral.
        config = pipelines.PipelineArgs(
            device_specs=device_specs,
            quantization_encoding=encoding,
            model_path=self.model_path,
            max_length=self.max_length,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(
            PipelineConfig.from_args(config)
        )

        assert isinstance(pipeline, pipelines.TextGenerationPipelineInterface)
        return MaxPipelineAndTokenizer(pipeline, tokenizer)

    def create_torch_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device: torch.device,
    ) -> TorchModelAndDataProcessor:
        processor = transformers.AutoProcessor.from_pretrained(self.model_path)
        config = transformers.AutoConfig.from_pretrained(self.model_path)
        model = transformers.LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            config=config,
            device_map=device,
            torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding] if encoding else None,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)


class _KimiK2_5BaseOracle(PipelineOracle):
    """Shared scaffolding for Kimi-K2.5-NVFP4 oracle variants.

    The Kimi K2.5 MAX pipeline config (8x EP/DP, 4096 ctx, NVFP4, etc.) and
    the "torch goldens not practical at 1T params" stance are identical
    across the multimodal upstream checkpoint and any text-only DeepseekV3
    conversion. Subclasses provide their own ``inputs`` and override the
    vLLM mm-specific kwargs via ``_vllm_extra_kwargs`` / ``_vllm_mm_data_key``.
    """

    # vLLM defaults match VLLMPipeline's own defaults (no mm passthrough).
    # The multimodal subclass overrides these.
    _vllm_extra_kwargs: Mapping[str, Any] = {}
    _vllm_mm_data_key: str = "image"

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path
        self.trust_remote_code = True

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {"gpu": ["float4_e2m1fnx2"]}

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        config = pipelines.PipelineArgs.from_flat_kwargs(
            device_specs=device_specs,
            quantization_encoding=encoding,
            model_path=self.model_path,
            max_length=4096,
            trust_remote_code=self.trust_remote_code,
            max_batch_input_tokens=4096,
            ep_size=8,
            data_parallel_degree=8,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(
            PipelineConfig.from_args(config)
        )
        assert isinstance(pipeline, TextGenerationPipelineInterface)
        return MaxPipelineAndTokenizer(pipeline, tokenizer)

    def create_torch_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device: torch.device | str,
    ) -> TorchModelAndDataProcessor:
        raise NotImplementedError(
            "Kimi K2.5 is 1T params (MoE) — torch golden generation is not"
            " practical. Use --framework vllm instead."
        )

    def create_vllm_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device_specs: list[driver.DeviceSpec],
    ) -> VLLMPipeline:
        gpu_count = sum(1 for d in device_specs if d.device_type == "gpu")
        return VLLMPipeline(
            model_path=self.model_path,
            trust_remote_code=self.trust_remote_code,
            encoding=encoding,
            tensor_parallel_size=max(1, gpu_count),
            extra_kwargs=dict(self._vllm_extra_kwargs),
            mm_data_key=self._vllm_mm_data_key,
        )


class KimiK2_5PipelineOracle(_KimiK2_5BaseOracle):
    """Pipeline oracle for Kimi K2.5 multimodal architectures (vLLM only).

    Kimi K2.5 is a 1T-parameter MoE model.
    """

    _vllm_extra_kwargs = {
        "mm_encoder_tp_mode": "data",
        "limit_mm_per_prompt": {"vision_chunk": 1},
    }
    _vllm_mm_data_key = "vision_chunk"

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        text_only_requests = [
            MockTextGenerationRequest.text_only(prompt)
            for prompt in test_data.SHORT_TEXT_PROMPTS
        ]
        return test_data.KIMIK2_5_REQUESTS + text_only_requests


class KimiK2_6PipelineOracle(KimiK2_5PipelineOracle):
    """Pipeline oracle for Kimi-K2.6-NVFP4 (vLLM only).

    K2.6 reuses the Kimi-K2.5 MAX architecture — same TEXT+IMAGE modalities,
    parsers, and tokenizer (see ``architectures/kimik2_5/arch.py``, which lists
    K2.6 as an example repo). It therefore inherits K2.5's vLLM golden setup
    verbatim: multimodal (``KIMIK2_5_REQUESTS``) + text inputs and the vision
    ``_vllm_extra_kwargs`` (``mm_encoder_tp_mode`` / ``limit_mm_per_prompt`` on
    the ``vision_chunk`` mm-data key).

    The MAX pipeline runs in TP+EP mode (``data_parallel_degree=1``,
    ``ep_size=8``, ``ep_use_allreduce=True``) to match the served K2.6
    deployment, rather than the K2.5 base oracle's DP+EP (``dp=8``) layout. This
    create_max_pipeline override is adapted from PR #86438.
    """

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        config = pipelines.PipelineArgs.from_flat_kwargs(
            device_specs=device_specs,
            quantization_encoding=encoding,
            model_path=self.model_path,
            max_length=4096,
            trust_remote_code=self.trust_remote_code,
            max_batch_input_tokens=4096,
            ep_size=8,
            data_parallel_degree=1,
            ep_use_allreduce=True,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(
            PipelineConfig.from_args(config)
        )
        assert isinstance(pipeline, TextGenerationPipelineInterface)
        return MaxPipelineAndTokenizer(pipeline, tokenizer)


class KimiK2_7PipelineOracle(KimiK2_5PipelineOracle):
    """Pipeline oracle for Kimi-K2.7-Code-NVFP4 (vLLM only).

    K2.7-Code reuses the Kimi-K2.5 MAX architecture — confirmed via the HF
    config (``architectures: KimiK25ForConditionalGeneration``, ``vision_config``
    present), same as K2.5 and K2.6. It therefore inherits K2.5's vLLM golden
    setup verbatim: multimodal (``KIMIK2_5_REQUESTS``) + text inputs and the
    vision ``_vllm_extra_kwargs`` (``mm_encoder_tp_mode`` / ``limit_mm_per_prompt``
    on the ``vision_chunk`` mm-data key).

    The MAX pipeline runs in TP+EP mode (``data_parallel_degree=1``,
    ``ep_size=8``, ``ep_use_allreduce=True``) to match the served K2.7-Code
    deployment (same layout as K2.6), rather than the K2.5 base oracle's DP+EP
    (``dp=8``) layout.
    """

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        config = pipelines.PipelineArgs.from_flat_kwargs(
            device_specs=device_specs,
            quantization_encoding=encoding,
            model_path=self.model_path,
            max_length=4096,
            trust_remote_code=self.trust_remote_code,
            max_batch_input_tokens=4096,
            ep_size=8,
            data_parallel_degree=1,
            ep_use_allreduce=True,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(
            PipelineConfig.from_args(config)
        )
        assert isinstance(pipeline, TextGenerationPipelineInterface)
        return MaxPipelineAndTokenizer(pipeline, tokenizer)


class KimiK2_5DeepseekV3PipelineOracle(_KimiK2_5BaseOracle):
    """Oracle for the text-only DeepseekV3 conversion of Kimi-K2.5-NVFP4.

    The checkpoint is the original ``nvidia/Kimi-K2.5-NVFP4`` weights with the
    vision stack stripped, MTP layer dropped, and key prefixes rewritten so
    the file loads via the standard DeepseekV3 pipeline. Underlying tensor
    bytes are bit-identical to the upstream Kimi NVFP4 checkpoint, so MAX
    DeepseekV3 logits should match vLLM's at NVFP4 quantization noise levels.
    """

    # Bump back to the default NUM_STEPS once EP-MoE + full-logits-return
    # mode no longer trips CUDA_ERROR_ILLEGAL_ADDRESS in
    # ``max/kernels/src/shmem/ep_comm.mojo``. With ``num_steps=10`` the run
    # aborts during the second decode step.
    num_steps = 1

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        return [
            MockTextGenerationRequest.text_only(prompt)
            for prompt in test_data.SHORT_TEXT_PROMPTS
        ]


class AmdKimiK2_5MXFP4PipelineOracle(PipelineOracle):
    """Pipeline oracle for AMD MXFP4 Kimi K2.5."""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path
        self.trust_remote_code = True

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {"gpu": ["float4_e2m1fnx2"]}

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        text_only_requests = [
            MockTextGenerationRequest.with_messages(
                prompt=prompt,
                messages=[{"role": "user", "content": prompt}],
                is_multimodal=False,
            )
            for prompt in test_data.SHORT_TEXT_PROMPTS
        ]
        return test_data.KIMIK2_5_REQUESTS + text_only_requests

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        gpu_count = max(
            1, sum(1 for d in device_specs if d.device_type == "gpu")
        )
        config = pipelines.PipelineArgs.from_flat_kwargs(
            device_specs=device_specs,
            quantization_encoding=encoding,
            model_path=self.model_path,
            max_length=4096,
            trust_remote_code=self.trust_remote_code,
            max_batch_input_tokens=4096,
            ep_size=gpu_count,
            data_parallel_degree=gpu_count,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(
            PipelineConfig.from_args(config)
        )
        assert isinstance(pipeline, TextGenerationPipelineInterface)
        return MaxPipelineAndTokenizer(pipeline, tokenizer)

    def create_torch_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device: torch.device | str,
    ) -> TorchModelAndDataProcessor:
        raise NotImplementedError(
            "Kimi K2.5 is 1T params (MoE) — torch golden generation is not"
            " practical. Use --framework vllm instead."
        )

    def create_vllm_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device_specs: list[driver.DeviceSpec],
    ) -> VLLMPipeline:
        gpu_count = sum(1 for d in device_specs if d.device_type == "gpu")
        # vLLM's kimi_k25 plugin exposes its multimodal input under the
        # vision_chunk modality, so the image must be keyed and limited under
        # vision_chunk to reach the model.
        return VLLMPipeline(
            model_path=self.model_path,
            trust_remote_code=self.trust_remote_code,
            encoding=encoding,
            tensor_parallel_size=max(1, gpu_count),
            extra_kwargs={
                "mm_encoder_tp_mode": "data",
                "limit_mm_per_prompt": {"vision_chunk": 1},
            },
            mm_data_key="vision_chunk",
        )


class KimiK2_5DeepseekV3LocalPathPipelineOracle(
    KimiK2_5DeepseekV3PipelineOracle
):
    """DeepseekV3 Kimi-K2.5-NVFP4 oracle loading weights from a local path.

    Identical to :class:`KimiK2_5DeepseekV3PipelineOracle` in every respect
    except that ``model_path`` is an absolute filesystem path (weights are
    pre-staged on a dedicated runner).
    """

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        config = pipelines.PipelineArgs(
            model_path=self.model_path,
            quantization_encoding=encoding,
            device_specs=device_specs,
            max_length=4096,
            trust_remote_code=self.trust_remote_code,
            data_parallel_degree=8,
            runtime=pipelines.PipelineRuntimeConfig(
                max_batch_input_tokens=4096,
                ep_size=8,
            ),
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(
            PipelineConfig.from_args(config)
        )
        assert isinstance(pipeline, TextGenerationPipelineInterface)
        return MaxPipelineAndTokenizer(pipeline, tokenizer)


class GenericOracle(PipelineOracle):
    def __init__(
        self,
        *,
        model_path: str,
        torch_model_path: str | None = None,
        device_encoding_map: dict[str, list[str]] | None = None,
        weight_path_map: dict[str, str] | None = None,
        config_params: dict[str, Any] = {},  # noqa: B006
        prompts: list[str] | None = None,
        apply_chat_template: bool = False,
        use_cache: bool = True,
        auto_model_cls: Any = transformers.AutoModelForCausalLM,
        auto_processor_cls: Any = transformers.AutoTokenizer,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
        batch_size: int | list[int] | None = None,
        add_bos_token: bool | None = None,
    ) -> None:
        self.model_path = model_path
        # A quantized checkpoint transformers cannot load names its bf16 source
        # here, so the torch reference is the model MAX's weights were
        # quantized from. Same tokenizer, same prompts, and the quantization
        # error lands in the tolerances.
        self.torch_model_path = torch_model_path or model_path
        # Memoized local path: an ``s3://`` model_path is synced to a local
        # cache dir on first use (see `_local_model_path`).
        self._resolved_model_path: str | None = None
        self._device_encoding_map = device_encoding_map
        self._weight_path_map = weight_path_map
        self.config_params = config_params
        self._prompts = prompts
        self._apply_chat_template = apply_chat_template
        self.auto_model_cls = auto_model_cls
        self.auto_processor_cls = auto_processor_cls
        self.task = task
        self._use_cache = use_cache
        self.default_batch_size = batch_size
        self.add_bos_token = add_bos_token
        self.trust_remote_code = config_params.get("trust_remote_code", False)

    @property
    def device_encoding_map(self) -> dict[str, list[str]] | None:
        return self._device_encoding_map

    def _local_model_path(self) -> str:
        """Return a local model directory, syncing from S3 once if needed.

        For an ``s3://`` ``model_path`` the weights are mirrored to a local
        cache dir on first call and reused thereafter; for any other path the
        value is returned unchanged.
        """
        if self._resolved_model_path is None:
            self._resolved_model_path = _sync_s3_model_to_local(self.model_path)
        return self._resolved_model_path

    def weight_path(self, encoding: pipelines.SupportedEncoding) -> str | None:
        if self._weight_path_map and encoding in self._weight_path_map:
            return self._weight_path_map[encoding]
        return None

    def _parse_weight_path(self, weight_path: str) -> tuple[str, str]:
        """Parse weight path into (repo_id, filename)."""
        path_pieces = weight_path.split("/")
        weight_repo_id = f"{path_pieces[0]}/{path_pieces[1]}"
        weight_filename = "/".join(path_pieces[2:])
        return weight_repo_id, weight_filename

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        model_path = self._local_model_path()
        weight_path = self.weight_path(encoding) if encoding else None

        weight_filename: str | None = None
        weight_repo_id: str | None = None
        if weight_path:
            weight_repo_id, weight_filename = self._parse_weight_path(
                weight_path
            )

        # Defer resolution so we can set _weights_repo_id before
        # validation runs.  Without this, construction-time resolution
        # would look for weight files in the model repo (meta-llama)
        # instead of the weights repo (bartowski).
        config_kwargs = {
            "task": self.task,
            "device_specs": device_specs or None,
            "quantization_encoding": encoding,
            "model_path": model_path,
            "weight_path": [] if weight_path is None else [weight_filename],
            **self.config_params,
        }
        config = pipelines.PipelineArgs.from_flat_kwargs(**config_kwargs)
        if weight_repo_id and weight_repo_id != model_path:
            # MAXModelConfig.from_pipeline_args(config) rebuilds a fresh
            # MAXModelConfig on every call, so writing through it is
            # silently discarded -- set the PipelineArgs private attr
            # directly instead.
            config._weights_repo_id = weight_repo_id
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(
            PipelineConfig.from_args(config), task=self.task
        )
        assert isinstance(
            pipeline,
            pipelines.TextGenerationPipelineInterface
            | pipelines.EmbeddingsPipeline,
        )
        if self.add_bos_token is not None and hasattr(tokenizer, "delegate"):
            # transformers v5 stopped honoring add_bos_token from
            # tokenizer_config.json for some tokenizers, so raw-prompt encoding
            # drops the leading BOS the reference goldens were generated with.
            tokenizer.delegate.add_bos_token = self.add_bos_token
        return MaxPipelineAndTokenizer(pipeline, tokenizer)

    def create_torch_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device: torch.device,
    ) -> TorchModelAndDataProcessor:
        model_path = (
            self.torch_model_path
            if self.torch_model_path != self.model_path
            else self._local_model_path()
        )
        processor = self.auto_processor_cls.from_pretrained(
            model_path,
            trust_remote_code=self.trust_remote_code,
        )
        weight_path = self.weight_path(encoding) if encoding else None
        if weight_path:
            config_path = Path(
                huggingface_hub.hf_hub_download(
                    repo_id=self.model_path,
                    filename="config.json",
                )
            )
            weight_repo_id, weight_filename = self._parse_weight_path(
                weight_path
            )
            downloaded_weight_path = Path(
                huggingface_hub.hf_hub_download(
                    repo_id=weight_repo_id,
                    filename=weight_filename,
                )
            )
            config = transformers.AutoConfig.from_pretrained(config_path)

            with disable_peft():
                model = self.auto_model_cls.from_pretrained(
                    "UNUSED",
                    config=config,
                    gguf_file=str(downloaded_weight_path),
                    device_map=device,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding]
                    if encoding
                    else None,
                )
        else:
            # Sub-byte and FP8 encodings cannot be a torch default dtype,
            # and a bf16 reference model has no quantized weights to load
            # anyway, so the torch side computes in bfloat16.
            if encoding in ("float8_e4m3fn", "float4_e2m1fnx2"):
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = (
                    ENCODING_TO_TORCH_DTYPE[encoding] if encoding else None
                )
            model = self.auto_model_cls.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch_dtype,
            )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        prompts = self._prompts or test_data.DEFAULT_PROMPTS
        if self._apply_chat_template:
            # Wrap each prompt in a chat message so the MAX tokenizer applies
            # the model's chat template, matching a templated reference golden.
            return [
                MockTextGenerationRequest.with_messages(
                    prompt=prompt,
                    messages=[{"role": "user", "content": prompt}],
                    is_multimodal=False,
                )
                for prompt in prompts
            ]
        return [
            MockTextGenerationRequest.text_only(prompt=prompt)
            for prompt in prompts
        ]

    @property
    def use_cache(self) -> bool:
        return self._use_cache


class ComponentModelOracle(GenericOracle):
    """Oracle for verifying a MAX ComponentModel (e.g. a text encoder) against
    a HuggingFace reference.

    The MAX path runs the ComponentModel directly, bypassing the standard
    text-generation pipeline. The Torch path uses the inherited GenericOracle
    loader and routes through ``run_text_encode``, which captures the
    pre-norm hidden state via a hook on ``model.model.norm``.

    ``padded_length`` controls input padding for both sides. Set to an
    integer to pad each prompt to that length via ``apply_chat_template`` +
    tokenizer ``padding="max_length"``; set to ``None`` to tokenize each
    prompt at its natural length with an all-ones attention mask.
    """

    def __init__(
        self,
        *,
        component_model_class: type,
        padded_length: int | None = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.component_model_class = component_model_class
        self.padded_length = padded_length

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        raise NotImplementedError(
            "ComponentModelOracle does not use the standard MAX pipeline; "
            "run_max_text_encoder is dispatched directly."
        )


class LoRAOracle(PipelineOracle):
    """Oracle for models with LoRA adapters."""

    def __init__(
        self,
        *,
        model_path: str,
        lora_repo_id: str,
        device_encoding_map: dict[str, list[str]],
        config_params: dict[str, Any] = {},  # noqa: B006
        prompts: list[str] | None = None,
        use_cache: bool = True,
    ) -> None:
        """Initialize LoRA oracle.

        Args:
            model_path: Path to the base model
            target_modules: Target modules for LoRA (qkvo, gate, down, up, etc.)
            lora_adapter_path: Path to the LoRA adapter (if None, creates test adapter)
            device_encoding_map: Device to encoding mapping
            config_params: Additional config parameters
            prompts: Custom prompts to use
            use_cache: Whether to use cache
            lora_rank: LoRA rank parameter
        """
        self.model_path = model_path
        self.lora_repo_id = lora_repo_id
        self._device_encoding_map = device_encoding_map
        self.config_params = config_params
        self._prompts = prompts
        self._use_cache = use_cache
        self.lora_rank = -1
        self._adapter_path: str | None = None

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return self._device_encoding_map

    def _get_shared_adapter(self) -> str:
        if self._adapter_path is None:
            original_adapter_path = huggingface_hub.snapshot_download(
                repo_id=self.lora_repo_id,
            )

            # Copy the adapter to /tmp/ to avoid modifying the original
            tmp_dir = tempfile.mkdtemp(prefix="lora_adapter_", dir="/tmp")
            self._adapter_path = os.path.join(
                tmp_dir, os.path.basename(original_adapter_path)
            )
            shutil.copytree(original_adapter_path, self._adapter_path)

            # Fix adapter config for compatibility with older PEFT versions
            config_path = Path(self._adapter_path) / "adapter_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

                self.lora_rank = config["r"]
                unsupported_fields = [
                    "corda_config",
                    "eva_config",
                    "exclude_modules",
                    "lora_bias",
                    "qalora_group_size",
                    "target_parameters",
                    "trainable_token_indices",
                    "use_qalora",
                ]
                for field in unsupported_fields:
                    if field in config:
                        del config[field]

                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

        assert self._adapter_path is not None
        return self._adapter_path

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        """Create MAX pipeline with LoRA adapter."""

        lora_path = self._get_shared_adapter()

        config = pipelines.PipelineArgs.from_flat_kwargs(
            device_specs=device_specs,
            quantization_encoding=encoding,
            model_path=self.model_path,
            enable_lora=True,
            lora_paths=[lora_path],
            max_num_loras=1,
            max_lora_rank=self.lora_rank,
            enable_prefix_caching=False,
            trust_remote_code=True,
            **self.config_params,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(
            PipelineConfig.from_args(config)
        )

        assert isinstance(pipeline, pipelines.TextGenerationPipeline)
        assert pipeline._pipeline_model._lora_manager is not None
        pipeline._pipeline_model._lora_manager.activate_adapter(lora_path)
        return MaxPipelineAndTokenizer(pipeline, tokenizer)

    def create_torch_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device: torch.device,
    ) -> TorchModelAndDataProcessor:
        """Create PyTorch pipeline with LoRA adapter using PEFT."""

        # Load base model
        lora_path = self._get_shared_adapter()

        processor = transformers.AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding] if encoding else None,
        )

        model = PeftModel.from_pretrained(model, lora_path, "lora")
        model.set_adapter("lora")

        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        prompts = self._prompts or test_data.DEFAULT_PROMPTS
        return [
            MockTextGenerationRequest(
                prompt=prompt,
                images=[],
                messages=[],
                is_multimodal=False,
                model_name=self._get_shared_adapter(),
            )
            for prompt in prompts
        ]

    @property
    def use_cache(self) -> bool:
        return self._use_cache


class ImageGenerationOracle(PipelineOracle):
    """Pipeline oracle for FLUX image generation."""

    num_steps: int
    """Number of denoising steps."""

    config_params: dict[str, Any]
    """Additional config parameters (e.g. prefer_module_v3)."""

    def __init__(
        self,
        model_path: str = "black-forest-labs/FLUX.2-dev-t2i-bfloat16-v2",
        num_steps: int = 50,
        requests: list[Any] = test_data.DEFAULT_PIXEL_GENERATION,
        config_params: dict[str, Any] = {},  # noqa: B006
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.task = PipelineTask.PIXEL_GENERATION
        self.num_steps = num_steps
        self._inputs = requests
        self.config_params = config_params

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {
            "gpu": ["bfloat16"],
        }

    @property
    def inputs(self) -> list[Any]:
        """Input prompts for image generation."""
        return self._inputs

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        """Create MAX FLUX pixel generation pipeline."""

        prefer_module_v3 = self.config_params.get("prefer_module_v3", False)
        transformer_weight_path = self.config_params.get(
            "transformer_weight_path"
        )

        models = ModelManifest.from_model_path(
            self.model_path,
            device_specs=device_specs,
        )
        if transformer_weight_path:
            # Two-repo layout: the transformer's weights live in a separate
            # single-file checkpoint (e.g. FLUX.2-dev + FLUX.2-dev-NVFP4)
            # while the rest of the pipeline keeps the base repo.
            models = models.with_override(
                "transformer",
                weight_path=[Path(p) for p in transformer_weight_path],
                quantization_encoding=encoding,
            )

        args_kwargs: dict[str, Any] = {}

        # Optional denoising-cache overrides (e.g. TaylorSeer / FBCache).
        denoising_cache = self.config_params.get("denoising_cache")
        if denoising_cache is not None:
            args_kwargs["denoising_cache"] = DenoisingCacheSettings(
                **denoising_cache
            )

        config = pipelines.PipelineArgs(
            models=models,
            runtime=pipelines.PipelineRuntimeConfig(
                prefer_module_v3=prefer_module_v3
            ),
            **args_kwargs,
        )

        # retrieve resolves the manifest and picks the tokenizer/executor
        # from the arch registry, like production serving.
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(
            PipelineConfig.from_args(config), task=self.task
        )

        return MaxPipelineAndTokenizer(
            pipeline=pipeline,  # type: ignore
            tokenizer=tokenizer,
        )

    def create_torch_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device: torch.device,
    ) -> TorchModelAndDataProcessor:
        """Create diffusers FLUX pipeline."""

        # Load the exact pipeline class from model config instead of relying on
        # auto-pipeline resolution.
        pipeline = diffusers.DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=ENCODING_TO_TORCH_DTYPE.get(encoding, torch.bfloat16),  # type: ignore
        )
        pipeline = pipeline.to(device)

        # Return pipeline as "model" and None as data_processor (not needed for diffusers)
        return TorchModelAndDataProcessor(
            model=pipeline,
            data_processor=None,
        )

    def run_torch_image_generation(
        self,
        *,
        torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
        device: torch.device,
        num_steps: int,
        inputs: list[Any],
    ) -> list[dict[str, Any]]:
        """Run image generation using diffusers FLUX."""

        return torch_utils.run_image_generation(
            pipeline=torch_pipeline_and_tokenizer.model,
            device=device,
            requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
        )


class WanGenerationOracle(ImageGenerationOracle):
    """Pipeline oracle for Wan T2V single-frame (image) generation."""

    def __init__(
        self,
        model_path: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        num_steps: int = 20,
        requests: list[Any] = WAN_PIXEL_GENERATION_T2I,
    ) -> None:
        super().__init__(
            model_path=model_path,
            num_steps=num_steps,
            requests=requests,
        )

    def create_max_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        """Create MAX Wan pixel generation pipeline."""
        models = ModelManifest.from_model_path(
            self.model_path,
            device_specs=device_specs,
        )
        config = pipelines.PipelineArgs(models=models)
        # retrieve resolves the manifest and picks the tokenizer/executor
        # from the arch registry (see ImageGenerationOracle).
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(
            PipelineConfig.from_args(config), task=self.task
        )
        return MaxPipelineAndTokenizer(
            pipeline=pipeline,  # type: ignore
            tokenizer=tokenizer,
        )

    def create_torch_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device: torch.device,
    ) -> TorchModelAndDataProcessor:
        """Load Wan diffusers pipeline with embed_tokens weight tie applied."""
        import diffusers
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(
            self.model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        pipeline = diffusers.WanPipeline.from_pretrained(
            self.model_path,
            vae=vae,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)

        # The HF checkpoint only ships shared.weight for the UMT5 text encoder,
        # but config.json sets tie_word_embeddings=false so transformers randomly
        # initializes encoder.embed_tokens.weight on load.  Force the tie to
        # recover correct text conditioning.
        pipeline.text_encoder.encoder.embed_tokens.weight = (
            pipeline.text_encoder.shared.weight
        )

        return TorchModelAndDataProcessor(model=pipeline, data_processor=None)

    def run_torch_image_generation(
        self,
        *,
        torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
        device: torch.device,
        num_steps: int,
        inputs: list[Any],
    ) -> list[dict[str, Any]]:
        """Run single-frame generation using diffusers WanPipeline."""
        return torch_utils.run_wan_image_generation(
            pipeline=torch_pipeline_and_tokenizer.model,
            requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
        )


class NemotronHOracle(GenericOracle):
    """Oracle for Nemotron-H (hybrid Mamba-2) ``trust_remote_code`` references.

    NVIDIA's bundled ``modeling_nemotron_h.py`` imports ``rmsnorm_fn`` from
    ``mamba_ssm.ops.triton.layernorm_gated`` unconditionally, and its
    ``MambaRMSNormGated.forward`` calls it even on the pure-torch
    ``torch_forward`` path. ``mamba_ssm`` is a CUDA/Triton package that does not
    build on non-CUDA hosts; since ``is_mamba_2_ssm_available()`` already gates
    on CUDA torch (so the SSD/conv fast path stays off), only this gated-norm op
    needs a torch stand-in. Install a pure-torch grouped gated-RMSNorm shim
    before loading the torch reference (no-op if ``mamba_ssm`` is really
    installed, so CUDA hosts are unaffected).
    """

    @staticmethod
    def _install_mamba_ssm_shim() -> None:
        if importlib.util.find_spec("mamba_ssm") is not None:
            return

        def rmsnorm_fn(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None = None,
            z: torch.Tensor | None = None,
            eps: float = 1e-6,
            group_size: int | None = None,
            norm_before_gate: bool = False,
            **_kwargs: object,
        ) -> torch.Tensor:
            if norm_before_gate:
                raise NotImplementedError(
                    "rmsnorm_fn shim only supports norm_before_gate=False"
                )
            in_dtype = x.dtype
            xf = x.to(torch.float32)
            if z is not None:
                xf = xf * torch.nn.functional.silu(z.to(torch.float32))
            d = xf.shape[-1]
            gs = int(group_size) if group_size is not None else d
            lead = xf.shape[:-1]
            xg = xf.reshape(*lead, d // gs, gs)
            xg = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
            out = weight * xg.reshape(*lead, d).to(in_dtype)
            return out if bias is None else out + bias

        def _mk(name: str) -> types.ModuleType:
            module = types.ModuleType(name)
            # transformers' is_mamba_2_ssm_available() -> _is_package_available
            # -> importlib.util.find_spec(name) raises if __spec__ is None, so
            # give each shim module a benign spec.
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            return module

        # Populate each shim module's namespace via its (writable) __dict__:
        # a plain `module.attr = x` trips mypy ("Module has no attribute"),
        # while setattr(module, "attr", x) trips ruff B010 — __dict__ avoids both.
        lng = _mk("mamba_ssm.ops.triton.layernorm_gated")
        lng.__dict__["rmsnorm_fn"] = rmsnorm_fn
        triton = _mk("mamba_ssm.ops.triton")
        triton.__dict__["layernorm_gated"] = lng
        ops = _mk("mamba_ssm.ops")
        ops.__dict__["triton"] = triton
        root = _mk("mamba_ssm")
        root.__dict__["ops"] = ops
        for name, module in (
            ("mamba_ssm", root),
            ("mamba_ssm.ops", ops),
            ("mamba_ssm.ops.triton", triton),
            ("mamba_ssm.ops.triton.layernorm_gated", lng),
        ):
            sys.modules[name] = module

    def create_torch_pipeline(
        self,
        *,
        encoding: pipelines.SupportedEncoding | None,
        device: torch.device,
    ) -> TorchModelAndDataProcessor:
        self._install_mamba_ssm_shim()
        return super().create_torch_pipeline(encoding=encoding, device=device)


PIPELINE_ORACLES: Mapping[str, PipelineOracle] = {
    "allenai/OLMo-1B-hf": GenericOracle(
        model_path="allenai/OLMo-1B-hf",
        config_params={"max_length": 1024},
        device_encoding_map={"cpu": ["float32"], "gpu": ["float32"]},
    ),
    "google/gemma-4-26B-A4B-it": GenericOracle(
        model_path="google/gemma-4-26B-A4B-it",
        config_params={
            "max_batch_size": 128,
        },
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "google/gemma-4-31B-it": GenericOracle(
        model_path="google/gemma-4-31B-it",
        config_params={
            "max_batch_size": 128,
        },
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "microsoft/Phi-3.5-mini-instruct": GenericOracle(
        model_path="microsoft/Phi-3.5-mini-instruct",
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "microsoft/phi-4": GenericOracle(
        model_path="microsoft/phi-4",
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": GenericOracle(
        model_path="LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        config_params={
            "max_length": 1024,
            "max_batch_size": 128,  # TODO(E2EOPT-48): Remove batch size override.
            "trust_remote_code": True,
        },
        device_encoding_map={"cpu": ["float32"], "gpu": ["float32"]},
    ),
    "LiquidAI/LFM2.5-1.2B-Instruct": GenericOracle(
        model_path="LiquidAI/LFM2.5-1.2B-Instruct",
        config_params={"trust_remote_code": True},
        device_encoding_map={
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": GenericOracle(
        model_path="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        # BF16 weights use ~278/288 GB on MI355X; cap context + batch to fit KV cache in remaining ~10 GB.
        config_params={"max_length": 8192, "max_batch_size": 16},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "meta-llama/Meta-Llama-3-8B-Instruct": GenericOracle(
        model_path="meta-llama/Meta-Llama-3-8B-Instruct",
        weight_path_map={
            "q4_k": "bartowski/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            "float32": "bartowski/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-fp32.gguf",
        },
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["float32", "bfloat16"],
            "cpu": ["float32", "q4_k"],
        },
    ),
    "meta-llama/Llama-3.1-8B-Instruct": GenericOracle(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        weight_path_map={
            "q4_k": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "float32": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-f32.gguf",
        },
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["float32", "bfloat16"],
            "cpu": ["float32", "q4_k"],
        },
    ),
    "meta-llama/Llama-3.1-8B-Instruct-data-parallel": GenericOracle(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        config_params={
            "max_length": 512,
            "data_parallel_degree": 2,
            "enable_prefix_caching": False,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
        },
        # prompts=test_data.DEFAULT_PROMPTS[1:2] + test_data.DEFAULT_PROMPTS[1:3],
        # Run the 4 text prompts with batch sizes 1, 2, 1.
        batch_size=[1, 2, 1],
    ),
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-float8-static": GenericOracle(
        model_path="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["float8_e4m3fn"],
        },
    ),
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic": GenericOracle(
        model_path="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["float8_e4m3fn"],
        },
    ),
    "nvidia/Llama-3.1-8B-Instruct-NVFP4": GenericOracle(
        model_path="nvidia/Llama-3.1-8B-Instruct-NVFP4",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["float4_e2m1fnx2"]},
    ),
    "nvidia/Llama-3.1-405B-Instruct-NVFP4": GenericOracle(
        model_path="nvidia/Llama-3.1-405B-Instruct-NVFP4",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["float4_e2m1fnx2"]},
    ),
    "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic": GenericOracle(
        model_path="RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic",
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["float8_e4m3fn"],
        },
    ),
    "modularai/Llama-3.1-405B-Instruct-autofp8": GenericOracle(
        model_path="modularai/Llama-3.1-405B-Instruct-autofp8",
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["float8_e4m3fn"],
        },
    ),
    "meta-llama/Llama-3.2-1B": GenericOracle(
        model_path="meta-llama/Llama-3.2-1B",
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["bfloat16"],
        },
    ),
    "meta-llama/Llama-3.3-70B-Instruct": GenericOracle(
        model_path="meta-llama/Llama-3.3-70B-Instruct",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "allenai/OLMo-2-0425-1B": GenericOracle(
        model_path="allenai/OLMo-2-0425-1B",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["float32"],
            "cpu": ["float32"],
        },
    ),
    "allenai/OLMo-2-0425-1B-Instruct": GenericOracle(
        model_path="allenai/OLMo-2-0425-1B-Instruct",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "allenai/OLMo-2-0425-1B-RLVR1": GenericOracle(
        model_path="allenai/OLMo-2-0425-1B-RLVR1",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "allenai/OLMo-2-1124-7B": GenericOracle(
        model_path="allenai/OLMo-2-1124-7B",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["float32"],
            "cpu": ["float32"],
        },
    ),
    "allenai/OLMo-2-1124-7B-Instruct": GenericOracle(
        model_path="allenai/OLMo-2-1124-7B-Instruct",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "allenai/OLMo-2-1124-13B-Instruct": GenericOracle(
        model_path="allenai/OLMo-2-1124-13B-Instruct",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "allenai/OLMo-2-0325-32B-Instruct": GenericOracle(
        model_path="allenai/OLMo-2-0325-32B-Instruct",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "allenai/Olmo-3-7B-Instruct": GenericOracle(
        model_path="allenai/Olmo-3-7B-Instruct",
        config_params={
            "max_length": 32768,
            "prefer_module_v3": True,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
        },
    ),
    "mistralai/Mistral-Nemo-Instruct-2407": GenericOracle(
        model_path="mistralai/Mistral-Nemo-Instruct-2407",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": GenericOracle(
        model_path="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
        auto_model_cls=transformers.AutoModelForImageTextToText,
    ),
    "OpenGVLab/InternVL3-1B-Instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-1B-Instruct"
    ),
    "OpenGVLab/InternVL3-8B-Instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-8B-Instruct"
    ),
    "OpenGVLab/InternVL3-14B-Instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-14B-Instruct"
    ),
    "OpenGVLab/InternVL3-38B-Instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-38B-Instruct"
    ),
    "OpenGVLab/InternVL3-78B-Instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-78B-Instruct"
    ),
    "OpenGVLab/InternVL3_5-8B-Instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3_5-8B-Instruct"
    ),
    "HuggingFaceM4/Idefics3-8B-Llama3": Idefics3PipelineOracle(
        "HuggingFaceM4/Idefics3-8B-Llama3"
    ),
    "mistral-experimental/pixtral-12b": PixtralPipelineOracle(),
    "Qwen/Qwen2.5-7B-Instruct": GenericOracle(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "Qwen/Qwen2.5-VL-3B-Instruct": Qwen2_5VLPipelineOracle(
        "Qwen/Qwen2.5-VL-3B-Instruct"
    ),
    "Qwen/Qwen2.5-VL-7B-Instruct": Qwen2_5VLPipelineOracle(
        "Qwen/Qwen2.5-VL-7B-Instruct"
    ),
    "Qwen/Qwen2.5-VL-32B-Instruct": Qwen2_5VLPipelineOracle(
        "Qwen/Qwen2.5-VL-32B-Instruct"
    ),
    # Qwen2.VL-FP8
    "allenai/olmOCR-2-7B-1025-FP8": Qwen2_5VLPipelineOracle(
        "allenai/olmOCR-2-7B-1025-FP8"
    ),
    "Qwen/Qwen3-VL-30B-A3B-Instruct": Qwen3VLPipelineOracle(
        "Qwen/Qwen3-VL-30B-A3B-Instruct"
    ),
    "Qwen/Qwen3-VL-4B-Instruct": Qwen3VLPipelineOracle(
        "Qwen/Qwen3-VL-4B-Instruct"
    ),
    "Qwen/Qwen3-VL-4B-Instruct-FP8": Qwen3VLPipelineOracle(
        "Qwen/Qwen3-VL-4B-Instruct-FP8",
        device_encoding_map={"gpu": ["float8_e4m3fn"]},
    ),
    "Qwen/Qwen3-4B-text-encoder": ComponentModelOracle(
        model_path="Qwen/Qwen3-4B",
        component_model_class=Qwen3TextEncoderKleinModel,
        padded_length=512,
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "Qwen/Qwen3-8B": GenericOracle(
        model_path="Qwen/Qwen3-8B",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "Qwen/Qwen3-32B": GenericOracle(
        model_path="Qwen/Qwen3-32B",
        config_params={"max_length": 512, "max_batch_size": 1},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "Qwen/Qwen3-30B-A3B": GenericOracle(
        model_path="Qwen/Qwen3-30B-A3B",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": GenericOracle(
        model_path="Qwen/Qwen3-30B-A3B-Instruct-2507",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8": GenericOracle(
        model_path="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["float8_e4m3fn"]},
    ),
    "Qwen/Qwen3.8-27B": GenericOracle(
        model_path="Qwen/Qwen3.8-27B",
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    # Same checkpoint through its image path. Qwen3.5 shares Qwen3-VL's
    # processor and `AutoModelForImageTextToText` entry, so the Qwen3VL oracle
    # drives it unchanged. The text-only entry above cannot stand in: with no
    # image to splice, M-RoPE's three axes collapse onto the flat token index
    # and the multi-axis positions go unexercised.
    "Qwen/Qwen3.8-27B-vision": Qwen3VLPipelineOracle("Qwen/Qwen3.8-27B"),
    "RadixArk/Qwen3.8-27B-NVFP4": GenericOracle(
        model_path="RadixArk/Qwen3.8-27B-NVFP4",
        torch_model_path="Qwen/Qwen3.8-27B",
        device_encoding_map={"gpu": ["float4_e2m1fnx2"]},
    ),
    "RadixArk/Qwen3.8-27B-NVFP4-vision": Qwen3VLPipelineOracle(
        "RadixArk/Qwen3.8-27B-NVFP4",
        torch_model_path="Qwen/Qwen3.8-27B",
        device_encoding_map={"gpu": ["float4_e2m1fnx2"]},
    ),
    "HuggingFaceTB/SmolLM2-135M": GenericOracle(
        model_path="HuggingFaceTB/SmolLM2-135M",
        config_params={
            "max_length": 512,
        },
        prompts=[p[:502] for p in test_data.DEFAULT_PROMPTS],
        device_encoding_map={
            "cpu": ["float32", "q4_k", "q4_0", "q6_k", "gptq"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "HuggingFaceTB/SmolLM2-360M-Instruct": LoRAOracle(
        model_path="HuggingFaceTB/SmolLM2-360M-Instruct",
        lora_repo_id="fausap/peft-smollm2-lora-gtx1660",
        config_params={
            "max_length": 2048,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
        },
    ),
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic-BF16-LoRA": LoRAOracle(
        model_path="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
        lora_repo_id="FinGPT/fingpt-mt_llama3-8b_lora",
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["float8_e4m3fn"],
        },
    ),
    "sentence-transformers/all-mpnet-base-v2": GenericOracle(
        model_path="sentence-transformers/all-mpnet-base-v2",
        # Maximum length accepted by MPNet tokenizer is 512.
        config_params={"max_length": 512, "pool_embeddings": False},
        prompts=[p[:502] for p in test_data.DEFAULT_PROMPTS],
        auto_model_cls=transformers.AutoModel,
        task=PipelineTask.EMBEDDINGS_GENERATION,
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32"],
        },
    ),
    "Qwen/Qwen3-Embedding-0.6B": GenericOracle(
        model_path="Qwen/Qwen3-Embedding-0.6B",
        config_params={"max_length": 8192, "pool_embeddings": True},
        auto_model_cls=transformers.AutoModel,
        task=PipelineTask.EMBEDDINGS_GENERATION,
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "Qwen/Qwen3-Embedding-4B": GenericOracle(
        model_path="Qwen/Qwen3-Embedding-4B",
        config_params={"max_length": 8192, "pool_embeddings": True},
        auto_model_cls=transformers.AutoModel,
        task=PipelineTask.EMBEDDINGS_GENERATION,
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "Qwen/Qwen3-Embedding-8B": GenericOracle(
        model_path="Qwen/Qwen3-Embedding-8B",
        config_params={"max_length": 8192, "pool_embeddings": True},
        auto_model_cls=transformers.AutoModel,
        task=PipelineTask.EMBEDDINGS_GENERATION,
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    # GPTQ llama with perm_idx
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4": GenericOracle(
        model_path="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        auto_model_cls=transformers.AutoModelForCausalLM,
        device_encoding_map={
            "cpu": ["float32", "q4_k", "q4_0", "q6_k", "gptq"],
            "gpu": ["float32", "bfloat16", "gptq"],
        },
    ),
    # GPTQ llama without perm_idx
    "kaitchup/DeepSeek-R1-Distill-Llama-8B-AutoRound-GPTQ-4bit": GenericOracle(
        model_path="kaitchup/DeepSeek-R1-Distill-Llama-8B-AutoRound-GPTQ-4bit",
        auto_model_cls=transformers.AutoModelForCausalLM,
        device_encoding_map={
            "cpu": ["float32", "q4_k", "q4_0", "q6_k", "gptq"],
            "gpu": ["float32", "bfloat16", "gptq"],
        },
    ),
    "google/gemma-3-1b-it": GenericOracle(
        model_path="google/gemma-3-1b-it",
        config_params={"max_length": 8192, "trust_remote_code": True},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "google/gemma-3-27b-it": GenericOracle(
        model_path="google/gemma-3-27b-it",
        config_params={"max_length": 8192, "trust_remote_code": True},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "google/gemma-3-12b-it": GenericOracle(
        model_path="google/gemma-3-12b-it",
        config_params={"max_length": 8192},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "RedHatAI/gemma-3-27b-it-FP8-dynamic": GenericOracle(
        model_path="RedHatAI/gemma-3-27b-it-FP8-dynamic",
        config_params={"max_length": 8192, "trust_remote_code": True},
        device_encoding_map={"gpu": ["float8_e4m3fn"]},
    ),
    "deepseek-ai/DeepSeek-V2-Lite-Chat": GenericOracle(
        model_path="deepseek-ai/DeepSeek-V2-Lite-Chat",
        config_params={"max_length": 516, "trust_remote_code": True},
        device_encoding_map={"gpu": ["bfloat16"]},
        prompts=[prompt[:1500] for prompt in test_data.DEFAULT_PROMPTS],
        # upstream modeling_deepsek.py uses a deprecated transformers function
        use_cache=False,
    ),
    "kathywu95/deepseek-v3-small-random": GenericOracle(
        model_path="kathywu95/deepseek-v3-small-random",
        config_params={
            "max_length": 516,
            "trust_remote_code": False,
        },
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "kathywu95/deepseek-v3-small-random-fp8": GenericOracle(
        model_path="kathywu95/deepseek-v3-small-random-fp8",
        config_params={
            "max_length": 516,
            "trust_remote_code": False,
            "use_subgraphs": False,
        },
        device_encoding_map={"gpu": ["float8_e4m3fn"]},
        add_bos_token=True,
    ),
    "deepseek-ai/DeepSeek-V3.1-Terminus": GenericOracle(
        model_path="deepseek-ai/DeepSeek-V3.1-Terminus",
        config_params={
            "max_length": 516,
            "trust_remote_code": False,
            "max_batch_input_tokens": 512,
            "ep_size": 8,
            "data_parallel_degree": 8,
        },
        device_encoding_map={"gpu": ["float8_e4m3fn"]},
        add_bos_token=True,
    ),
    "deepseek-ai/DeepSeek-V3.2-Exp": GenericOracle(
        model_path="deepseek-ai/DeepSeek-V3.2-Exp",
        config_params={
            "max_length": 516,
            "trust_remote_code": False,
            "max_batch_input_tokens": 512,
            "ep_size": 8,
            "data_parallel_degree": 1,
            # Match `--kv-cache-format float8_e4m3fn` (FP8 KV avoids dtype mismatches / hangs).
            "kv_cache_format": "float8_e4m3fn",
        },
        device_encoding_map={"gpu": ["float8_e4m3fn"]},
        add_bos_token=True,
    ),
    "nvidia/Kimi-K2.7-Code-NVFP4": KimiK2_7PipelineOracle(
        "nvidia/Kimi-K2.7-Code-NVFP4"
    ),
    # Trimmed MiniMax-M3 (dense-only layers) for logit verification. The prompts
    # are pinned and apply_chat_template is off so the input_ids match those the
    # reference logits were generated from.
    "minimax/minimax3-dense-only": GenericOracle(
        model_path="s3://modular-kadabra-weights/minimax-m3-3L-fp8",
        config_params={"trust_remote_code": True},
        device_encoding_map={"gpu": ["float8_e4m3fn"]},
        prompts=[
            "The capital of France is",
            "Quantum computing is",
        ],
        apply_chat_template=False,
    ),
    # MiniMax-M3-MXFP8, compared against goldens pregenerated
    # from the mm-sglang MiniMax-M3 fork. Uses the short logit-verification
    # prompts (the long prompt is excluded) and apply_chat_template is off so
    # the input_ids match the raw tokens the reference logits were generated
    # from. Config mirrors the proven 8-GPU MXFP8 recipe
    # (max_private/minimax_m3/recipes/mxfp8_8x_b200.yaml): ep_size=8 builds the
    # sparse-attention indexer's multi-KV branch, data_parallel_degree=1 avoids
    # the DP-attention indexer kernel crash, and chunked prefill + prefix
    # caching keep the EP / FP8 paths on their known-good code path.
    # trust_remote_code is off so MAX's registered MiniMaxM3VLConfig is used.
    # The MAX side needs the private arch registered: run via
    # //max_private/minimax_m3:verify_minimax_m3.
    "MiniMaxAI/MiniMax-M3-MXFP8": GenericOracle(
        model_path="MiniMaxAI/MiniMax-M3-MXFP8",
        config_params={
            "max_length": 1024,
            "trust_remote_code": False,
            "data_parallel_degree": 1,
            "ep_size": 8,
            "max_batch_input_tokens": 4096,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "device_memory_utilization": 0.65,
        },
        device_encoding_map={"gpu": ["float8_e4m3fn"]},
        prompts=list(test_data.SHORT_TEXT_PROMPTS),
        apply_chat_template=False,
    ),
    "HKUSTAudio/Llasa-8B": GenericOracle(
        model_path="HKUSTAudio/Llasa-8B",
        config_params={
            "max_length": 2048,
            "trust_remote_code": False,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
        },
        # TTS-specific prompts formatted according to the HF model card.
        prompts=[
            "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>Hello, this is a test of the Llasa text-to-speech system.<|TEXT_UNDERSTANDING_END|>",
            "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>The quick brown fox jumps over the lazy dog.<|TEXT_UNDERSTANDING_END|>",
            "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>Good morning! How are you today?<|TEXT_UNDERSTANDING_END|>",
            "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>In a hole in the ground there lived a hobbit.<|TEXT_UNDERSTANDING_END|>",
        ],
        use_cache=True,
    ),
    "black-forest-labs/FLUX.2-dev-t2i": ImageGenerationOracle(
        "black-forest-labs/FLUX.2-dev",
    ),
    "black-forest-labs/FLUX.2-dev-i2i": ImageGenerationOracle(
        "black-forest-labs/FLUX.2-dev",
        requests=test_data.FLUX2_PIXEL_GENERATION_I2I,
    ),
    "black-forest-labs/FLUX.2-klein-4B": ImageGenerationOracle(
        "black-forest-labs/FLUX.2-klein-4B",
        num_steps=4,
    ),
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": WanGenerationOracle(
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    ),
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8": NemotronHOracle(
        # A pre-dequantized local bf16 checkpoint can be substituted via
        # NEMOTRON_H_MODEL_PATH (there is no Apple FP8 matmul kernel, so on
        # Metal the FP8 path yields all-zero logits; bring up on bf16-at-load).
        model_path=os.environ.get(
            "NEMOTRON_H_MODEL_PATH", "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"
        ),
        # Cap the context and batch so the paged KV cache AND the per-slot SSM
        # state pool fit: the config default max_position_embeddings=262144
        # sizes an ~80 GB KV buffer, and the SSM pool is
        # max_batch_size x n_mamba x nheads x head_dim x dstate x fp32
        # (~40 GB at 512 slots), so keep both small on a single Apple GPU.
        config_params={
            "max_length": 8192,
            "max_batch_size": 8,
            "trust_remote_code": True,
            "device_memory_utilization": 0.9,
        },
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
}
