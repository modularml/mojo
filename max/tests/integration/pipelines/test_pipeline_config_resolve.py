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

"""Tests for the full PipelineConfig construction-time resolution flow.

All tests use fake local repos (config.json + weight files) and mock
SupportedArchitecture instances registered in PIPELINE_REGISTRY.
No network access required.
"""

import dataclasses
import json
import logging
import os
import pickle
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from max.driver import DeviceSpec
from max.graph import DeviceRef
from max.graph.weights import WeightsFormat
from max.pipelines import PIPELINE_REGISTRY, PipelineArgs, PipelineConfig
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.config import KVCacheConfig
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import MAXModelConfig, MemoryEstimator
from max.pipelines.lib.config import SpeculativeConfig
from max.pipelines.lib.config.model_config import (
    _device_specs_for_encoding,
    _resolve_weights_and_encoding,
    _select_dtype_cast,
    _select_quantization_encoding,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.registry import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask
from max.pipelines.sampling import SamplingConfig
from test_common.fake_weights import (
    write_fake_safetensors,
    write_mixed_safetensors,
)
from test_common.pipeline_model_dummy import (
    DUMMY_GEMMA_ARCH,
    DUMMY_LLAMA_ARCH,
    DummyLlamaArchConfig,
    DummyLlamaPipelineModel,
    DummyTextTokenizer,
)
from test_common.registry import prepare_registry

GPU_DEVICE_SPEC = DeviceSpec(id=0, device_type="gpu")
CPU_DEVICE_SPEC = DeviceSpec(id=0, device_type="cpu")


# ---------------------------------------------------------------------------
# Helpers — local repo with config.json
# ---------------------------------------------------------------------------

_LLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "num_hidden_layers": 2,
    "rope_theta": 10000.0,
    "max_position_embeddings": 2048,
    "intermediate_size": 11008,
    "vocab_size": 32000,
    "rms_norm_eps": 1e-5,
}

_GEMMA_CONFIG = {
    "architectures": ["Gemma3ForCausalLM"],
    "model_type": "gemma3",
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "num_hidden_layers": 2,
    "rope_theta": 10000.0,
    "max_position_embeddings": 2048,
    "intermediate_size": 11008,
    "vocab_size": 32000,
    "rms_norm_eps": 1e-5,
    "head_dim": 128,
}


def _make_local_repo(
    tmpdir: str,
    hf_config: dict[str, Any] | None = None,
    safetensors_files: dict[str, dict[str, str]] | None = None,
    gguf_files: list[str] | None = None,
) -> str:
    """Create a local repo directory with config.json and fake weight files.

    Args:
        tmpdir: Root temp directory.
        hf_config: HuggingFace config dict to write as config.json.
            Defaults to _LLAMA_CONFIG.
        safetensors_files: Mapping of relative path to {tensor_name: dtype}.
        gguf_files: List of relative GGUF filenames to create as empty files.

    Returns:
        The repo root path.
    """
    config = hf_config if hf_config is not None else _LLAMA_CONFIG
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(config, f)

    if safetensors_files:
        for rel_path, tensors in safetensors_files.items():
            full_path = os.path.join(tmpdir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if len(tensors) == 1:
                _, dtype = next(iter(tensors.items()))
                write_fake_safetensors(full_path, dtype=dtype)
            else:
                write_mixed_safetensors(full_path, tensors)
    if gguf_files:
        for rel_path in gguf_files:
            full_path = os.path.join(tmpdir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            open(full_path, "w").close()
    return tmpdir


# ---------------------------------------------------------------------------
# Mock context manager
# ---------------------------------------------------------------------------


@contextmanager
def _pipeline_resolve_mocks(
    weight_path_return: tuple[list[Path], str | None] = ([], None),
    num_devices: int = 1,
) -> Iterator[None]:
    """Patches external dependencies for the full config resolution flow.

    Mocks external I/O and hardware while leaving the real resolution
    logic intact:
    - load_devices — avoid GPU probes
    - WeightPathParser.parse — avoid network
    - validate_hf_repo_access — avoid network
    - MemoryEstimator — avoid real memory estimation
    - accelerator_api — avoid CUDA probes
    - PagedMemoryPlanner — avoid activation memory estimation
    """
    mock_devices = [DeviceRef.GPU()] * num_devices

    with (
        patch(
            "max.pipelines.lib.config.model_config.WeightPathParser.parse",
            return_value=weight_path_return,
        ),
        patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"),
        patch(
            "max.pipelines.lib.memory_estimation.load_devices",
            return_value=mock_devices,
        ),
        patch.object(
            MemoryEstimator,
            "plan_from_sizes",
            side_effect=lambda pipeline_config, model_config, *a, **kw: (
                MemoryPlan(
                    planned_max_batch_size=1,
                    footprint=0,
                    planned_max_length=model_config.max_length,
                    device_specs=tuple(model_config.device_specs),
                    planned_max_batch_total_tokens=pipeline_config.runtime.max_batch_total_tokens,
                )
            ),
        ),
        patch(
            "max.pipelines.lib.config.config.accelerator_api",
            return_value="cpu",
        ),
        patch.object(
            PagedMemoryPlanner,
            "estimate_activation_memory",
            return_value=0,
        ),
    ):
        yield


def _model(config: PipelineConfig) -> MAXModelConfig:
    """Return the main model config, asserting it is not None."""
    assert config.model is not None
    return config.model


def _plan_memory(config: PipelineConfig) -> PipelineConfig:
    """Run the registry's memory-planning step against the config.

    Convenience wrapper for tests that exercise planning directly rather
    than going through PIPELINE_REGISTRY.retrieve_factory(). Returns the
    config that planning ran against.
    """
    task = (
        config.task
        if config.task != PipelineTask.UNDEFINED
        else PipelineTask.TEXT_GENERATION
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        architecture_name=config.models.main_architecture_name,
        prefer_module_v3=config.runtime.prefer_module_v3,
        task=task,
    )
    if arch is None:
        raise ValueError(
            f"MAX-optimized architecture not available for"
            f" '{config.models.main_architecture_name}'."
            " Please file a request at https://modul.ar/request to add this"
            " model architecture to MAX."
        )
    # from_args already resolved the encoding, making this write-back a
    # no-op; directly-constructed configs (a few tests below) still need the
    # effective encoding mirrored onto the field for assertions.
    resolved_encoding = _select_quantization_encoding(
        _model(config), arch.default_encoding
    )
    if _model(config).quantization_encoding != resolved_encoding:
        # Both are immutable, so rebuild rather than assign.
        config = config.model_copy(
            update={
                "models": config.models.with_override(
                    "main", quantization_encoding=resolved_encoding
                )
            }
        )
    MemoryEstimator.plan(config, arch)
    return config


def _make_pipeline_config(
    model_path: str,
    device_specs: list[DeviceSpec] | None = None,
    weight_path: list[Path] | None = None,
    max_length: int | None = 512,
    max_batch_size: int = 1,
    pipeline_task: Any = None,
    **model_kwargs: Any,
) -> PipelineConfig:
    """Create a PipelineConfig via ``from_args``, the construction boundary.

    Construction resolves ``quantization_encoding`` and ``weight_path``
    against the registered architecture, so the architecture under test must
    be registered before calling this.
    """
    if device_specs is None:
        device_specs = [GPU_DEVICE_SPEC]
    return PipelineConfig.from_args(
        PipelineArgs(
            model_path=model_path,
            device_specs=device_specs,
            weight_path=weight_path or [],
            max_length=max_length,
            runtime=PipelineRuntimeConfig(
                max_batch_size=max_batch_size,
            ),
            task=pipeline_task or PipelineTask.UNDEFINED,
            **model_kwargs,
        )
    )


# ---------------------------------------------------------------------------
# Category A: Architecture Lookup + Encoding Resolution (Happy Path)
# ---------------------------------------------------------------------------


class TestArchitectureEncodingResolution:
    """Tests that verify the full resolve chain for common encoding scenarios."""

    @prepare_registry
    def test_resolve_bf16_safetensors_llama(self) -> None:
        """BF16 safetensors with LlamaForCausalLM architecture."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir, safetensors_files={"model.safetensors": {"w": "BF16"}}
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            assert _model(config).quantization_encoding == "bfloat16"
            assert any(
                "model.safetensors" in str(p)
                for p in _model(config).weight_path
            )

    @prepare_registry
    def test_resolve_gguf_q4_0_llama(self) -> None:
        """Q4_0 GGUF with LlamaForCausalLM architecture."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, gguf_files=["model-Q4_0.gguf"])
            config = _make_pipeline_config(
                tmpdir, device_specs=[CPU_DEVICE_SPEC]
            )
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            assert _model(config).quantization_encoding == "q4_0"
            assert any(
                "model-Q4_0.gguf" in str(p) for p in _model(config).weight_path
            )

    @prepare_registry
    def test_resolve_fp8_safetensors_llama(self) -> None:
        """FP8 safetensors with LlamaForCausalLM architecture."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={"model.safetensors": {"w": "F8_E4M3"}},
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            assert _model(config).quantization_encoding == "float8_e4m3fn"

    @prepare_registry
    def test_resolve_f32_on_gpu_uses_arch_default(self) -> None:
        """F32 safetensors on GPU: architecture default_encoding is used.

        Construction infers float32 from the file, but the
        architecture-level validation may fall back to the arch default
        encoding when reconciling file encoding with device capabilities.
        """
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir, safetensors_files={"model.safetensors": {"w": "F32"}}
            )
            config = _make_pipeline_config(
                tmpdir, device_specs=[GPU_DEVICE_SPEC]
            )
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            # The encoding should be resolved (either float32 or bfloat16)
            # and must be in the architecture's supported_encodings.
            model = _model(config)
            assert (
                model.quantization_encoding
                in DUMMY_LLAMA_ARCH.supported_encodings
            )

    @prepare_registry
    def test_resolve_f32_on_cpu_stays_f32(self) -> None:
        """F32 safetensors on CPU should stay F32."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir, safetensors_files={"model.safetensors": {"w": "F32"}}
            )
            config = _make_pipeline_config(
                tmpdir, device_specs=[CPU_DEVICE_SPEC]
            )
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            assert _model(config).quantization_encoding == "float32"


# ---------------------------------------------------------------------------
# Category B: Architecture Default Encoding Fallback
# ---------------------------------------------------------------------------


class TestDefaultEncodingFallback:
    """Tests that the architecture's default_encoding is used as a fallback."""

    @prepare_registry
    def test_default_encoding_used_when_ambiguous_on_cpu(self) -> None:
        """A mixed-dtype checkpoint on CPU resolves to the architecture default.

        The checkpoint's own weight dtype (bfloat16) is GPU-only, so inference
        must not select it for a CPU target; resolution falls back to the
        CPU-valid architecture default (float32). This holds regardless of
        whether weight_path defaults have been discovered yet, so every
        consumer resolves the same value.
        """
        from max.graph.weights import WeightsFormat
        from max.pipelines.context import TextContext
        from max.pipelines.modeling.types import PipelineTask

        # Create an architecture with default_encoding="float32" compatible with CPU
        cpu_arch = SupportedArchitecture(
            name="LlamaForCausalLM",
            task=PipelineTask.TEXT_GENERATION,
            example_repo_ids=["test/model"],
            default_encoding="float32",
            supported_encodings={"float32", "bfloat16"},
            pipeline_model=DummyLlamaPipelineModel,
            tokenizer=DummyTextTokenizer,
            context_type=TextContext,
            multi_gpu_supported=True,
            default_weights_format=WeightsFormat.safetensors,
            config=DummyLlamaArchConfig,
        )
        PIPELINE_REGISTRY.register(cpu_arch)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mixed BF16+F32 on CPU: bfloat16 is GPU-only, so resolution can't
            # pick it and falls back to the arch default (float32).
            _make_local_repo(
                tmpdir,
                safetensors_files={
                    "model.safetensors": {
                        "weight": "BF16",
                        "bias": "F32",
                    }
                },
            )
            config = _make_pipeline_config(
                tmpdir, device_specs=[CPU_DEVICE_SPEC]
            )
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            assert _model(config).quantization_encoding == "float32"


# ---------------------------------------------------------------------------
# Category C: Encoding Validation Against supported_encodings
# ---------------------------------------------------------------------------


class TestEncodingValidation:
    """Tests that unsupported encodings are rejected."""

    @prepare_registry
    def test_reject_encoding_not_in_supported_encodings(self) -> None:
        """Q4_K GGUF should be rejected when arch doesn't support q4_k.

        Encoding resolution runs at construction, so the error fires in
        ``from_args``.
        """
        # DUMMY_LLAMA_ARCH intentionally excludes q4_k from supported_encodings
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, gguf_files=["model-Q4_K_M.gguf"])
            with pytest.raises(ValueError, match="not supported by MAX engine"):
                _make_pipeline_config(tmpdir, device_specs=[CPU_DEVICE_SPEC])

    @prepare_registry
    def test_explicit_unsupported_encoding_rejected(self) -> None:
        """Explicitly setting an unsupported encoding raises at construction."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir, safetensors_files={"model.safetensors": {"w": "BF16"}}
            )
            with pytest.raises(ValueError, match="not supported by MAX engine"):
                _make_pipeline_config(
                    tmpdir,
                    device_specs=[CPU_DEVICE_SPEC],
                    quantization_encoding="q4_k",
                )

    @prepare_registry
    def test_encoding_incompatible_with_devices_rejected(self) -> None:
        """A GPU-only encoding on a CPU target raises at construction."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir, safetensors_files={"model.safetensors": {"w": "BF16"}}
            )
            with pytest.raises(
                ValueError,
                match="not compatible with the selected device type 'cpu'",
            ):
                _make_pipeline_config(
                    tmpdir,
                    device_specs=[CPU_DEVICE_SPEC],
                    quantization_encoding="bfloat16",
                )


# ---------------------------------------------------------------------------
# Category D: Architecture Not Found
# ---------------------------------------------------------------------------


class TestArchitectureNotFound:
    """Tests for missing or unknown architectures."""

    @prepare_registry
    def test_unknown_architecture_raises(self) -> None:
        """Unknown architecture in config.json raises at construction."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            unknown_config = dict(_LLAMA_CONFIG)
            unknown_config["architectures"] = ["UnknownModelForCausalLM"]
            _make_local_repo(
                tmpdir,
                hf_config=unknown_config,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            with (
                _pipeline_resolve_mocks(),
                pytest.raises(
                    ValueError,
                    match="No architecture found for UnknownModelForCausalLM",
                ),
            ):
                _make_pipeline_config(tmpdir)

    @prepare_registry
    def test_missing_config_json_raises(self) -> None:
        """Missing config.json should raise an error."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write weight files but no config.json
            write_fake_safetensors(os.path.join(tmpdir, "model.safetensors"))
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks(), pytest.raises(Exception):
                config = _plan_memory(config)


# ---------------------------------------------------------------------------
# Category E: Multi-GPU Validation
# ---------------------------------------------------------------------------


class TestMultiGPUValidation:
    """Tests for multi-GPU support validation."""

    @prepare_registry
    def test_multi_gpu_rejected_for_unsupported_arch(self) -> None:
        """Architecture without multi_gpu_supported should reject 2 GPUs."""
        # DUMMY_GEMMA_ARCH has multi_gpu_supported=False
        PIPELINE_REGISTRY.register(DUMMY_GEMMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                hf_config=_GEMMA_CONFIG,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            two_gpus = [
                DeviceSpec(id=0, device_type="gpu"),
                DeviceSpec(id=1, device_type="gpu"),
            ]
            # Multi-GPU support is validated at construction.
            with (
                _pipeline_resolve_mocks(num_devices=2),
                pytest.raises(
                    ValueError,
                    match="Multiple GPU inference is currently not supported",
                ),
            ):
                _make_pipeline_config(tmpdir, device_specs=two_gpus)

    @prepare_registry
    def test_multi_gpu_allowed_for_supported_arch(self) -> None:
        """Architecture with multi_gpu_supported should allow 2 GPUs."""
        # DUMMY_LLAMA_ARCH has multi_gpu_supported=True
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            two_gpus = [
                DeviceSpec(id=0, device_type="gpu"),
                DeviceSpec(id=1, device_type="gpu"),
            ]
            config = _make_pipeline_config(tmpdir, device_specs=two_gpus)
            with _pipeline_resolve_mocks(num_devices=2):
                config = _plan_memory(config)
            assert _model(config).quantization_encoding == "bfloat16"


# ---------------------------------------------------------------------------
# Category F: RoPE Type Resolution
# ---------------------------------------------------------------------------


class TestRopeTypeResolution:
    """Tests for RoPE type resolution from architecture defaults."""

    @prepare_registry
    def test_rope_type_preserved_through_resolution(self) -> None:
        """A user-set rope_type survives resolution untouched.

        Resolution no longer writes an architecture default onto
        ``model.rope_type`` (the arch default is applied by each
        architecture's ``ArchConfig.initialize``); the field is now purely the
        user override, so an explicit value must pass through unchanged.
        """
        PIPELINE_REGISTRY.register(DUMMY_GEMMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                hf_config=_GEMMA_CONFIG,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            config = _make_pipeline_config(tmpdir, rope_type="neox")
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            assert _model(config).rope_type == "neox"

    @prepare_registry
    def test_rope_type_unset_stays_none(self) -> None:
        """An unset rope_type stays None after resolution (no arch default)."""
        PIPELINE_REGISTRY.register(DUMMY_GEMMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                hf_config=_GEMMA_CONFIG,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            assert _model(config).rope_type is None


class TestStructuredOutputBackendResolution:
    """Architecture default for ``sampling.structured_output_backend``."""

    @prepare_registry
    def test_backend_resolved_from_architecture(self) -> None:
        """Arch's default_structured_output_backend applies when user is unset."""
        arch = dataclasses.replace(
            DUMMY_GEMMA_ARCH, default_structured_output_backend="xgrammar"
        )
        PIPELINE_REGISTRY.register(arch)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                hf_config=_GEMMA_CONFIG,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            assert config.sampling.structured_output_backend == "xgrammar"

    @prepare_registry
    def test_explicit_backend_value_wins(self) -> None:
        """An explicit user backend is never overridden by the arch default."""
        arch = dataclasses.replace(
            DUMMY_GEMMA_ARCH, default_structured_output_backend="xgrammar"
        )
        PIPELINE_REGISTRY.register(arch)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                hf_config=_GEMMA_CONFIG,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            # Constructing with the field set records it in model_fields_set.
            config = _make_pipeline_config(
                tmpdir,
                sampling=SamplingConfig(structured_output_backend="llguidance"),
            )
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            assert config.sampling.structured_output_backend == "llguidance"


class TestStructuredOutputAnyWhitespaceResolution:
    """Architecture default for ``sampling.structured_output_any_whitespace``."""

    @prepare_registry
    def test_any_whitespace_resolved_from_architecture(self) -> None:
        """Arch's default_structured_output_any_whitespace applies when unset."""
        arch = dataclasses.replace(
            DUMMY_GEMMA_ARCH, default_structured_output_any_whitespace=True
        )
        PIPELINE_REGISTRY.register(arch)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                hf_config=_GEMMA_CONFIG,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            assert config.sampling.structured_output_any_whitespace is True

    @prepare_registry
    def test_explicit_any_whitespace_value_wins(self) -> None:
        """An explicit user value is never overridden by the arch default."""
        arch = dataclasses.replace(
            DUMMY_GEMMA_ARCH, default_structured_output_any_whitespace=True
        )
        PIPELINE_REGISTRY.register(arch)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                hf_config=_GEMMA_CONFIG,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            config = _make_pipeline_config(
                tmpdir,
                sampling=SamplingConfig(structured_output_any_whitespace=False),
            )
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            assert config.sampling.structured_output_any_whitespace is False

    @prepare_registry
    def test_any_whitespace_defaults_to_compact(self) -> None:
        """With no user value and no arch default, resolution pins False.

        False (compact JSON) is today's behavior and the Gemma-4 runaway
        mitigation (0c57a6bd331); the global default must not drift.
        """
        PIPELINE_REGISTRY.register(DUMMY_GEMMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                hf_config=_GEMMA_CONFIG,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            assert config.sampling.structured_output_any_whitespace is False


# ---------------------------------------------------------------------------
# Category G: Cache Dtype Resolution
# ---------------------------------------------------------------------------


class TestCacheDtypeResolution:
    """Tests that cache dtype is set based on quantization encoding."""

    @prepare_registry
    def test_cache_dtype_bf16_for_bf16_encoding(self) -> None:
        """BF16 encoding should result in bfloat16 cache dtype."""
        from max.dtype import DType
        from max.pipelines.kv_cache import cache_dtype_for_encoding

        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            assert (
                cache_dtype_for_encoding(
                    _model(config).quantization_encoding,
                    _model(config).kv_cache.kv_cache_format,
                )
                == DType.bfloat16
            )


# ---------------------------------------------------------------------------
# Category H: Weight Path Discovery Through Full Pipeline
# ---------------------------------------------------------------------------


class TestWeightPathDiscovery:
    """Tests for weight file discovery through the full resolve chain."""

    @prepare_registry
    def test_sharded_safetensors_discovered(self) -> None:
        """Multiple sharded safetensors should all be discovered."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={
                    "model-00001-of-00002.safetensors": {"w": "BF16"},
                    "model-00002-of-00002.safetensors": {"w": "BF16"},
                },
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            paths = sorted(str(p) for p in _model(config).weight_path)
            assert paths == [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ]

    @prepare_registry
    def test_safetensors_preferred_over_gguf(self) -> None:
        """When both formats exist, safetensors should be preferred."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
                gguf_files=["model-Q4_0.gguf"],
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            paths = [str(p) for p in _model(config).weight_path]
            assert paths == ["model.safetensors"]


# ---------------------------------------------------------------------------
# Category I: Required Arguments Enforcement
# ---------------------------------------------------------------------------


class TestRequiredArguments:
    """Tests that architecture required_arguments override user config."""

    @prepare_registry
    def test_required_arguments_override_user_config(self) -> None:
        """Architecture required_arguments should override conflicting config values."""
        from max.pipelines.context import TextContext
        from max.pipelines.modeling.types import PipelineTask

        arch_with_required = SupportedArchitecture(
            name="LlamaForCausalLM",
            task=PipelineTask.TEXT_GENERATION,
            example_repo_ids=["test/model"],
            default_encoding="bfloat16",
            supported_encodings={"bfloat16", "float32"},
            pipeline_model=DummyLlamaPipelineModel,
            tokenizer=DummyTextTokenizer,
            context_type=TextContext,
            multi_gpu_supported=True,
            default_weights_format=DUMMY_LLAMA_ARCH.default_weights_format,
            config=DummyLlamaArchConfig,
            required_arguments={"enable_prefix_caching": False},
        )
        PIPELINE_REGISTRY.register(arch_with_required)

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            # Pass a value that conflicts with the required argument;
            # construction applies the architecture override.
            config = _make_pipeline_config(
                tmpdir, kv_cache=KVCacheConfig(enable_prefix_caching=True)
            )
            assert _model(config).kv_cache.enable_prefix_caching is False
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            # Registry-phase resolution must not undo the construction-time
            # override.
            assert _model(config).kv_cache.enable_prefix_caching is False


# ---------------------------------------------------------------------------
# Category J: DGC suppressed for embedding task on shared arch name (QUA-484)
# ---------------------------------------------------------------------------


class TestDGCTaskDisambiguation:
    """DGC must not be auto-enabled when the arch name is shared between
    TEXT_GENERATION and EMBEDDINGS_GENERATION and the pipeline task is
    EMBEDDINGS_GENERATION.

    Regression test for QUA-484: Qwen3ForCausalLM is registered for both
    tasks; removing it from the DGC disable list incorrectly enabled DGC
    for the embedding model because the no-task lookup returned the
    text-gen arch (registered first), passing the task eligibility check.
    """

    @prepare_registry
    def test_dgc_not_enabled_for_embedding_task(self) -> None:
        """Construction with task=EMBEDDINGS_GENERATION must not auto-enable DGC."""

        shared_name = "SharedArchForCausalLM"
        text_gen_arch = SupportedArchitecture(
            name=shared_name,
            task=PipelineTask.TEXT_GENERATION,
            example_repo_ids=["test/text-model"],
            default_encoding="bfloat16",
            supported_encodings={"bfloat16", "float32"},
            pipeline_model=DummyLlamaPipelineModel,
            tokenizer=DummyTextTokenizer,
            context_type=TextContext,
            multi_gpu_supported=True,
            default_weights_format=WeightsFormat.safetensors,
            config=DummyLlamaArchConfig,
        )
        embedding_arch = SupportedArchitecture(
            name=shared_name,
            task=PipelineTask.EMBEDDINGS_GENERATION,
            example_repo_ids=["test/embed-model"],
            default_encoding="bfloat16",
            supported_encodings={"bfloat16", "float32"},
            pipeline_model=DummyLlamaPipelineModel,
            tokenizer=DummyTextTokenizer,
            context_type=TextContext,
            multi_gpu_supported=True,
            default_weights_format=WeightsFormat.safetensors,
            config=DummyLlamaArchConfig,
        )
        # Register text-gen first (mirrors Qwen3ForCausalLM registration order)
        PIPELINE_REGISTRY.register(text_gen_arch)
        PIPELINE_REGISTRY.register(embedding_arch)

        hf_config = dict(_LLAMA_CONFIG)
        hf_config["architectures"] = [shared_name]

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                hf_config=hf_config,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            # Patch the accelerator probe around construction so the False
            # comes from the task gating, not the host's accelerator.
            with patch(
                "max.pipelines.lib.config.config.accelerator_api",
                return_value="cuda",
            ):
                config = _make_pipeline_config(
                    tmpdir,
                    max_batch_size=4,
                    pipeline_task=PipelineTask.EMBEDDINGS_GENERATION,
                )
            assert config.runtime.device_graph_capture is False

    @prepare_registry
    def test_dgc_enabled_for_text_gen_task(self) -> None:
        """Without a task (text-gen default), DGC auto-enables when eligible."""

        shared_name = "SharedArchForCausalLM"
        text_gen_arch = SupportedArchitecture(
            name=shared_name,
            task=PipelineTask.TEXT_GENERATION,
            example_repo_ids=["test/text-model"],
            default_encoding="bfloat16",
            supported_encodings={"bfloat16", "float32"},
            pipeline_model=DummyLlamaPipelineModel,
            tokenizer=DummyTextTokenizer,
            context_type=TextContext,
            multi_gpu_supported=True,
            default_weights_format=WeightsFormat.safetensors,
            config=DummyLlamaArchConfig,
        )
        embedding_arch = SupportedArchitecture(
            name=shared_name,
            task=PipelineTask.EMBEDDINGS_GENERATION,
            example_repo_ids=["test/embed-model"],
            default_encoding="bfloat16",
            supported_encodings={"bfloat16", "float32"},
            pipeline_model=DummyLlamaPipelineModel,
            tokenizer=DummyTextTokenizer,
            context_type=TextContext,
            multi_gpu_supported=True,
            default_weights_format=WeightsFormat.safetensors,
            config=DummyLlamaArchConfig,
        )
        PIPELINE_REGISTRY.register(text_gen_arch)
        PIPELINE_REGISTRY.register(embedding_arch)

        hf_config = dict(_LLAMA_CONFIG)
        hf_config["architectures"] = [shared_name]

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                hf_config=hf_config,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            # Overlap-scheduler/DGC resolution happens at construction, so
            # the accelerator probe must be patched around from_args.
            with patch(
                "max.pipelines.lib.config.config.accelerator_api",
                return_value="cuda",
            ):
                config = _make_pipeline_config(tmpdir, max_batch_size=4)
            assert config.runtime.device_graph_capture is True


# ---------------------------------------------------------------------------
# Category K: Chat Template Wiring Through retrieve_tokenizer()
# ---------------------------------------------------------------------------


class TestChatTemplateWiring:
    """``PipelineConfig.model.chat_template`` (a ``Path``) must reach the
    tokenizer.

    Regression coverage for the ``registry.py`` call sites that read
    ``pipeline_config.model.chat_template`` and pass it through
    ``_retrieve_chat_template()`` when building the tokenizer.
    """

    @prepare_registry
    def test_chat_template_path_reaches_tokenizer(self) -> None:
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir, safetensors_files={"model.safetensors": {"w": "BF16"}}
            )
            template_file = Path(tmpdir) / "custom_template.jinja"
            template_file.write_text("{{ messages }}")
            config = _make_pipeline_config(tmpdir, chat_template=template_file)

            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
                PIPELINE_REGISTRY.retrieve_tokenizer(config)

        assert DummyTextTokenizer.init_kwargs["chat_template"] == (
            "{{ messages }}"
        )


class TestCpuOnlyEncodingDeviceHandling:
    """CPU-only encodings (GGUF q4) on GPU devices downcast to CPU at
    construction; directly-constructed configs keep their raw fields."""

    @prepare_registry
    def test_gguf_q4_direct_construction_keeps_raw_devices(self) -> None:
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, gguf_files=["model-Q4_0.gguf"])
            with patch(
                "max.pipelines.lib.device_specs.scan_available_devices",
                return_value=[GPU_DEVICE_SPEC],
            ):
                config = PipelineConfig(
                    models=ModelManifest(
                        {
                            "main": MAXModelConfig(
                                model_path=tmpdir, max_length=512
                            )
                        }
                    ),
                    runtime=PipelineRuntimeConfig(max_batch_size=1),
                )
            model = _model(config)
            assert model.device_specs == [GPU_DEVICE_SPEC]

            with _pipeline_resolve_mocks():
                config = _plan_memory(config)

            assert _model(config).quantization_encoding == "q4_0"
            # Directly-constructed configs skip construction-time resolution;
            # nothing downcasts their devices anymore.
            assert _model(config).device_specs == [GPU_DEVICE_SPEC]

    @prepare_registry
    def test_gguf_q4_on_explicit_gpu_downcasts_at_construction(self) -> None:
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, gguf_files=["model-Q4_0.gguf"])
            config = _make_pipeline_config(
                tmpdir, device_specs=[GPU_DEVICE_SPEC]
            )
            # from_args applies the CPU downcast at construction.
            assert _model(config).device_specs == [DeviceSpec.cpu()]
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
            assert _model(config).device_specs == [DeviceSpec.cpu()]


def test_downcast_free_function() -> None:
    gpu = [GPU_DEVICE_SPEC]
    assert _device_specs_for_encoding(gpu, "q4_k") == [DeviceSpec.cpu()]
    # GPU-capable encodings are never downcast.
    assert _device_specs_for_encoding(gpu, "bfloat16") == [GPU_DEVICE_SPEC]
    # CPU devices are already valid for CPU-only encodings.
    assert _device_specs_for_encoding([CPU_DEVICE_SPEC], "q4_k") == [
        CPU_DEVICE_SPEC
    ]


@prepare_registry
def test_construction_downcast_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The CPU-downcast warning fires once, at construction; re-populating
    an already-downcast config does not warn again.
    """
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_local_repo(tmpdir, gguf_files=["model-Q4_0.gguf"])
        with caplog.at_level(logging.WARNING, logger="max.pipelines"):
            config = _make_pipeline_config(
                tmpdir, device_specs=[GPU_DEVICE_SPEC]
            )
        downcast_warnings = [
            r for r in caplog.records if "Switching device_specs" in r.message
        ]
        assert len(downcast_warnings) == 1
        assert _model(config).device_specs == [DeviceSpec.cpu()]

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="max.pipelines"):
            _resolve_weights_and_encoding(
                _model(config),
                default_encoding=DUMMY_LLAMA_ARCH.default_encoding,
                supported_encodings=DUMMY_LLAMA_ARCH.supported_encodings,
                default_weights_format=DUMMY_LLAMA_ARCH.default_weights_format,
            )
        assert _model(config).device_specs == [DeviceSpec.cpu()]
        assert not [
            r for r in caplog.records if "Switching device_specs" in r.message
        ]


class TestMemoryPlanDevices:
    """The memory plan carries the resolved devices it was computed against."""

    def _retrieve_arch(self, config: PipelineConfig) -> Any:
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            architecture_name=config.models.main_architecture_name,
            prefer_module_v3=config.runtime.prefer_module_v3,
            task=PipelineTask.TEXT_GENERATION,
        )
        assert arch is not None
        return arch

    @prepare_registry
    def test_memory_plan_carries_resolved_devices(self) -> None:
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir, safetensors_files={"model.safetensors": {"w": "BF16"}}
            )
            config = _make_pipeline_config(tmpdir)
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
                plan = MemoryEstimator.plan(config, self._retrieve_arch(config))
            assert plan.device_specs == (GPU_DEVICE_SPEC,)
            # The plan crosses the model-worker process boundary inside the
            # pipeline factory; it must stay picklable.
            pickle.dumps(plan)

    @prepare_registry
    def test_memory_plan_devices_reflect_cpu_downcast(self) -> None:
        """For a q4 GGUF model the plan carries the construction-downcast
        CPU set."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, gguf_files=["model-Q4_0.gguf"])
            config = _make_pipeline_config(
                tmpdir, device_specs=[GPU_DEVICE_SPEC]
            )
            with _pipeline_resolve_mocks():
                config = _plan_memory(config)
                plan = MemoryEstimator.plan(config, self._retrieve_arch(config))
            assert plan.device_specs == (DeviceSpec.cpu(),)


# ---------------------------------------------------------------------------
# Category L: Construction-time resolution
# ---------------------------------------------------------------------------


class TestConstructionResolution:
    """``PipelineConfig.from_args`` resolves ``quantization_encoding``,
    ``weight_path``, and the effective ``device_specs`` against the
    registered architecture; the registry phase leaves them untouched.

    The dummy architectures must be registered *before* ``from_args`` is
    called: construction reads the shared ``ARCH_LOOKUP`` table that
    ``prepare_registry`` resets.
    """

    @staticmethod
    def _from_args(
        model_path: str,
        device_specs: list[DeviceSpec] | None = None,
        **kwargs: Any,
    ) -> PipelineConfig:
        return PipelineConfig.from_args(
            PipelineArgs(
                model_path=model_path,
                device_specs=device_specs or [GPU_DEVICE_SPEC],
                **kwargs,
            )
        )

    @staticmethod
    def _resolve_via_registry(config: PipelineConfig) -> tuple[Any, Any]:
        """Looks up the archs with the same selection inputs the registry uses."""
        task = (
            config.task
            if config.task != PipelineTask.UNDEFINED
            else PipelineTask.TEXT_GENERATION
        )
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            architecture_name=config.models.main_architecture_name,
            prefer_module_v3=config.runtime.prefer_module_v3,
            task=task,
        )
        assert arch is not None
        draft_arch = None
        if config.draft_model is not None:
            draft_arch = PIPELINE_REGISTRY.retrieve_architecture(
                architecture_name=config.draft_model.architecture_name,
                prefer_module_v3=config.runtime.prefer_module_v3,
            )
            assert draft_arch is not None
        return arch, draft_arch

    def _assert_resolve_preserves(
        self, config: PipelineConfig
    ) -> tuple[Any, Any]:
        """Runs the registry arch lookup and asserts the constructed values survive."""
        model = _model(config)
        constructed_encoding = model.quantization_encoding
        constructed_paths = list(model.weight_path)
        assert constructed_encoding is not None
        assert constructed_paths
        arch, draft_arch = self._resolve_via_registry(config)
        assert model.quantization_encoding == constructed_encoding
        assert model.weight_path == constructed_paths
        return arch, draft_arch

    @prepare_registry
    def test_explicit_weight_path(self) -> None:
        """An explicit weight_path is kept — discovery never runs."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={
                    "model.safetensors": {"w": "BF16"},
                    "other.safetensors": {"w": "BF16"},
                },
            )
            config = self._from_args(
                tmpdir, weight_path=[Path("model.safetensors")]
            )
            assert _model(config).weight_path == [Path("model.safetensors")]
            self._assert_resolve_preserves(config)

    @prepare_registry
    def test_discovery_sharded_safetensors(self) -> None:
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={
                    "model-00001-of-00002.safetensors": {"w": "BF16"},
                    "model-00002-of-00002.safetensors": {"w": "BF16"},
                },
            )
            config = self._from_args(tmpdir)
            assert sorted(str(p) for p in _model(config).weight_path) == [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ]
            self._assert_resolve_preserves(config)

    @prepare_registry
    def test_safetensors_preferred_over_gguf(self) -> None:
        """Format preference (arch default is gguf; only safetensors match)."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
                gguf_files=["model-Q4_0.gguf"],
            )
            config = self._from_args(tmpdir)
            assert _model(config).weight_path == [Path("model.safetensors")]
            self._assert_resolve_preserves(config)

    @prepare_registry
    def test_f32_to_bf16_cast_fallback(self) -> None:
        """fp32 checkpoint on GPU: encoding casts to bf16, files stay f32.

        The recorded cast bookkeeping must also survive later re-derivation:
        weight adapters call ``_select_quantization_encoding`` /
        ``_select_dtype_cast`` at load time, after ``weight_path`` is
        populated, and must still see (bfloat16, float32->bfloat16).
        """
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir, safetensors_files={"model.safetensors": {"w": "F32"}}
            )
            config = self._from_args(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            model = _model(config)
            assert model.quantization_encoding == "bfloat16"
            assert model.weight_path == [Path("model.safetensors")]
            arch, _ = self._assert_resolve_preserves(config)
            assert (
                _select_quantization_encoding(model, arch.default_encoding)
                == "bfloat16"
            )
            assert _select_dtype_cast(model, arch.default_encoding) == (
                "float32",
                "bfloat16",
            )

    @prepare_registry
    def test_multi_encoding_repo_default_tiebreak(self) -> None:
        """Ambiguous multi-encoding repo on CPU falls to the arch default."""
        cpu_arch = dataclasses.replace(
            DUMMY_LLAMA_ARCH,
            default_encoding="float32",
            supported_encodings={"float32", "bfloat16"},
            default_weights_format=WeightsFormat.safetensors,
        )
        PIPELINE_REGISTRY.register(cpu_arch)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={
                    "model.safetensors": {"weight": "BF16", "bias": "F32"}
                },
            )
            config = self._from_args(tmpdir, device_specs=[CPU_DEVICE_SPEC])
            assert _model(config).quantization_encoding == "float32"
            self._assert_resolve_preserves(config)

    @prepare_registry
    def test_draft_model_resolved_at_construction(self) -> None:
        """The draft model resolves with its own architectures[0].

        The target is Gemma3 (outside the unified spec-decode override
        mapping) so the target arch name survives construction; a Llama
        target would be rewritten to UnifiedEagleLlama3ForCausalLM.
        """
        PIPELINE_REGISTRY.register(DUMMY_GEMMA_ARCH)
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with (
            tempfile.TemporaryDirectory() as target_dir,
            tempfile.TemporaryDirectory() as draft_dir,
        ):
            _make_local_repo(
                target_dir,
                hf_config=_GEMMA_CONFIG,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            _make_local_repo(
                draft_dir,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            config = self._from_args(
                target_dir,
                draft_model=MAXModelConfig(
                    model_path=draft_dir, device_specs=[GPU_DEVICE_SPEC]
                ),
                speculative=SpeculativeConfig(speculative_method="mtp"),
            )
            draft = config.draft_model
            assert draft is not None
            assert draft.quantization_encoding == "bfloat16"
            assert draft.weight_path == [Path("model.safetensors")]
            self._assert_resolve_preserves(config)
            assert draft.quantization_encoding == "bfloat16"
            assert draft.weight_path == [Path("model.safetensors")]

    @prepare_registry
    def test_draft_model_device_downcast_at_construction(self) -> None:
        """A CPU-only draft encoding downcasts only the draft's devices."""
        PIPELINE_REGISTRY.register(DUMMY_GEMMA_ARCH)
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with (
            tempfile.TemporaryDirectory() as target_dir,
            tempfile.TemporaryDirectory() as draft_dir,
        ):
            _make_local_repo(
                target_dir,
                hf_config=_GEMMA_CONFIG,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            _make_local_repo(draft_dir, gguf_files=["model-Q4_0.gguf"])
            config = self._from_args(
                target_dir,
                draft_model=MAXModelConfig(
                    model_path=draft_dir, device_specs=[GPU_DEVICE_SPEC]
                ),
                speculative=SpeculativeConfig(speculative_method="mtp"),
            )
            draft = config.draft_model
            assert draft is not None
            assert draft.quantization_encoding == "q4_0"
            assert draft.device_specs == [DeviceSpec.cpu()]
            assert _model(config).device_specs == [GPU_DEVICE_SPEC]

    @prepare_registry
    def test_max_length_resolved_at_construction(self) -> None:
        """With --max-length unset, construction runs each architecture's
        sequence-length policy once: the main config carries its own
        checkpoint bound, the draft carries the draft's, and the args keep
        recording the raw user intent. The draft clamp is planning-only, so
        the main value is not lowered here."""
        PIPELINE_REGISTRY.register(DUMMY_GEMMA_ARCH)
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with (
            tempfile.TemporaryDirectory() as target_dir,
            tempfile.TemporaryDirectory() as draft_dir,
        ):
            _make_local_repo(
                target_dir,
                hf_config=_GEMMA_CONFIG,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            _make_local_repo(
                draft_dir,
                hf_config={**_LLAMA_CONFIG, "max_position_embeddings": 1024},
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            args = PipelineArgs(
                model_path=target_dir,
                device_specs=[GPU_DEVICE_SPEC],
                draft_model=MAXModelConfig(
                    model_path=draft_dir, device_specs=[GPU_DEVICE_SPEC]
                ),
                speculative=SpeculativeConfig(speculative_method="mtp"),
            )
            config = PipelineConfig.from_args(args)
            assert args.max_length is None
            assert _model(config).max_length == 2048
            draft = config.draft_model
            assert draft is not None
            assert draft.max_length == 1024

    @prepare_registry
    def test_max_length_over_checkpoint_bound_rejected(self) -> None:
        """A user max_length above a bounded architecture's checkpoint limit
        is rejected at construction, where the policy now runs."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            # _LLAMA_CONFIG caps max_position_embeddings at 2048.
            with pytest.raises(ValueError, match="exceeds the upper bound"):
                self._from_args(tmpdir, max_length=4096)

    @prepare_registry
    def test_unknown_arch_rejected_at_construction(self) -> None:
        """A determinable but unregistered architecture fails construction."""
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            unknown_config = dict(_LLAMA_CONFIG)
            unknown_config["architectures"] = ["UnknownModelForCausalLM"]
            _make_local_repo(
                tmpdir,
                hf_config=unknown_config,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            with pytest.raises(
                ValueError,
                match="No architecture found for UnknownModelForCausalLM",
            ):
                self._from_args(tmpdir)

    @prepare_registry
    def test_spec_decode_target_override_before_resolution(self) -> None:
        """The unified spec-decode target override runs in from_args, so
        construction resolves the overridden architecture (regression guard
        for the pre-override arch being resolved instead; see #88511)."""
        unified_arch = dataclasses.replace(
            DUMMY_LLAMA_ARCH, name="UnifiedMTPDeepseekV3ForCausalLM"
        )
        PIPELINE_REGISTRY.register(unified_arch)
        hf_config = dict(_LLAMA_CONFIG)
        hf_config["architectures"] = ["DeepseekV3ForCausalLM"]
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                hf_config=hf_config,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            config = self._from_args(
                tmpdir,
                speculative=SpeculativeConfig(speculative_method="mtp"),
            )
            assert (
                config.models.main_architecture_name
                == "UnifiedMTPDeepseekV3ForCausalLM"
            )
            assert _model(config).quantization_encoding == "bfloat16"
            assert _model(config).weight_path == [Path("model.safetensors")]
            self._assert_resolve_preserves(config)

    @prepare_registry
    def test_custom_architectures_imported_at_construction(self) -> None:
        """runtime.custom_architectures modules register before the
        construction-time arch lookup, so from_args resolves against them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = os.path.join(tmpdir, "my_custom_arch_mxf517.py")
            with open(module_path, "w") as f:
                f.write(
                    "import dataclasses\n"
                    "from test_common.pipeline_model_dummy import DUMMY_LLAMA_ARCH\n"
                    "ARCHITECTURES = [dataclasses.replace("
                    "DUMMY_LLAMA_ARCH, name='MyCustomForCausalLM')]\n"
                )
            repo_dir = os.path.join(tmpdir, "repo")
            os.makedirs(repo_dir)
            hf_config = dict(_LLAMA_CONFIG)
            hf_config["architectures"] = ["MyCustomForCausalLM"]
            _make_local_repo(
                repo_dir,
                hf_config=hf_config,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
            )
            config = self._from_args(
                repo_dir,
                runtime=PipelineRuntimeConfig(
                    custom_architectures=[f"{tmpdir}:my_custom_arch_mxf517"]
                ),
            )
            assert _model(config).quantization_encoding == "bfloat16"
            assert _model(config).weight_path == [Path("model.safetensors")]
