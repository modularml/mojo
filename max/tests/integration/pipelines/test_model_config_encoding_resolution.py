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

"""Tests for MAXModelConfig encoding inference and weight path resolution.

All tests use fake local safetensors/GGUF repos with no network access.

Encoding/weight-path resolution for LLM models happens at construction:
``PipelineConfig.from_args`` calls ``_resolve_weights_and_encoding()``
once the architecture is known, which selects the effective encoding and
discovers weight files. Most tests below drive that resolver (or the pure
``_select_quantization_encoding()`` helper) directly rather than going
through the full ``PipelineConfig``/registry machinery, to keep the setup
narrow.
"""

import json
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, get_args
from unittest.mock import patch

import pytest
from max.driver import DeviceSpec
from max.graph.weights import WeightsFormat
from max.pipelines.lib import MAXModelConfig
from max.pipelines.lib.config.model_config import (
    _resolve_weights_and_encoding,
    _select_quantization_encoding,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from max.pipelines.weights.hf_utils import (
    HuggingFaceRepo,
)
from max.pipelines.weights.hf_utils import (
    generate_local_model_path as _real_generate_local_model_path,
)
from test_common.fake_weights import (
    write_fake_safetensors,
    write_mixed_safetensors,
)

_DEFAULT_ENCODING: SupportedEncoding = "bfloat16"

GPU_DEVICE_SPEC = DeviceSpec(id=0, device_type="gpu")
CPU_DEVICE_SPEC = DeviceSpec(id=0, device_type="cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_local_repo(
    tmpdir: str,
    safetensors_files: dict[str, dict[str, str]] | None = None,
    gguf_files: list[str] | None = None,
) -> str:
    """Create a local repo directory with fake weight files.

    Args:
        tmpdir: Root temp directory.
        safetensors_files: Mapping of relative path to {tensor_name: dtype}.
            If the dict has one entry, uses write_fake_safetensors for simplicity.
        gguf_files: List of relative GGUF filenames to create as empty files.

    Returns:
        The repo root path.
    """
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


@contextmanager
def _resolve_mocks(
    weight_path_return: tuple[list[Path], str | None] = ([], None),
) -> Iterator[None]:
    """Context manager that patches external dependencies for resolve().

    Args:
        weight_path_return: Return value for WeightPathParser.parse.
    """
    with (
        patch(
            "max.pipelines.lib.config.model_config.WeightPathParser.parse",
            return_value=weight_path_return,
        ),
        patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"),
    ):
        yield


def _make_config(
    model_path: str,
    device_specs: list[DeviceSpec] | None = None,
    weight_path: list[Path] | None = None,
    **kwargs,
) -> MAXModelConfig:
    """Create a MAXModelConfig for testing."""
    if device_specs is None:
        device_specs = [GPU_DEVICE_SPEC]
    return MAXModelConfig(
        model_path=model_path,
        device_specs=device_specs,
        weight_path=weight_path or [],
        **kwargs,
    )


def _resolve_encoding(
    config: MAXModelConfig,
    default_encoding: SupportedEncoding = _DEFAULT_ENCODING,
) -> MAXModelConfig:
    """Resolves quantization_encoding the way a consumer (an
    ``ArchConfig.initialize``) does: via the pure
    :func:`_select_quantization_encoding` helper. Returns a copy carrying the
    result so tests can assert on ``config.quantization_encoding``.
    """
    encoding = _select_quantization_encoding(config, default_encoding)
    return config.model_copy(update={"quantization_encoding": encoding})


def _resolve_encoding_and_weight_path(
    config: MAXModelConfig,
    default_encoding: SupportedEncoding = _DEFAULT_ENCODING,
    default_weights_format: WeightsFormat = WeightsFormat.safetensors,
) -> MAXModelConfig:
    """Resolves both encoding and weight_path via
    :func:`_resolve_weights_and_encoding`, the resolver
    ``PipelineConfig.from_args`` runs, and applies the results the way
    from_args does so the assertions below can read them off the config.
    Every encoding is treated as architecture-supported so these tests
    exercise inference/discovery, not the supported-encodings gate.
    """
    encoding, dtype_cast, weight_path, device_specs = (
        _resolve_weights_and_encoding(
            config,
            default_encoding=default_encoding,
            supported_encodings=set(get_args(SupportedEncoding)),
            default_weights_format=default_weights_format,
        )
    )
    update: dict[str, Any] = {
        "quantization_encoding": encoding,
        "device_specs": device_specs,
    }
    if not config.weight_path:
        update["weight_path"] = weight_path
    replaced = config.model_copy(update=update)
    replaced._resolved_dtype_cast = dtype_cast
    return replaced


# ---------------------------------------------------------------------------
# Category A: Single-Encoding Repos — Encoding Inference
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _offline_hf_construction() -> Iterator[None]:
    """Keep ``MAXModelConfig`` construction offline (CI runs
    ``HF_HUB_OFFLINE=1``): ``__init__`` eagerly builds the HuggingFace repo
    handles. Real cached repos resolve normally; uncached/placeholder repos
    get a fake path.
    """

    def _gen(repo_id: str, revision: str) -> str:
        try:
            return _real_generate_local_model_path(repo_id, revision)
        except Exception:
            return f"/fake/cache/{repo_id}"

    with (
        patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"),
        patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
        patch(
            "max.pipelines.weights.hf_utils.generate_local_model_path",
            side_effect=_gen,
        ),
    ):
        yield


class TestSingleEncodingInference:
    """Tests for encoding inference from repos with a single encoding."""

    def test_infer_encoding_single_bf16_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "BF16"}})
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            config = _resolve_encoding(config)
            assert config.quantization_encoding == "bfloat16"

    def test_infer_encoding_single_f32_on_gpu_casts_to_bf16(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "F32"}})
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            config = _resolve_encoding(config)
            assert config.quantization_encoding == "bfloat16"

    def test_infer_encoding_single_f32_on_cpu_stays_f32(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "F32"}})
            config = _make_config(tmpdir, device_specs=[CPU_DEVICE_SPEC])
            config = _resolve_encoding(config)
            assert config.quantization_encoding == "float32"

    def test_infer_encoding_single_fp8_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "F8_E4M3"}})
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            config = _resolve_encoding(config)
            assert config.quantization_encoding == "float8_e4m3fn"

    def test_infer_encoding_single_fp4_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "U8"}})
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            config = _resolve_encoding(config)
            assert config.quantization_encoding == "float4_e2m1fnx2"

    def test_infer_encoding_gguf_q4_k_from_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, gguf_files=["model-Q4_K_M.gguf"])
            config = _make_config(tmpdir, device_specs=[CPU_DEVICE_SPEC])
            config = _resolve_encoding(config)
            assert config.quantization_encoding == "q4_k"


# ---------------------------------------------------------------------------
# Category B: Mixed-Encoding Safetensors — Core Stress Tests
# ---------------------------------------------------------------------------


class TestMixedEncodingInference:
    """Tests for encoding inference from repos with mixed-encoding safetensors."""

    def test_mixed_fp4_fp8_bf16_selects_fp4(self) -> None:
        """FP4 should win when all three quantized types are present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {
                        "quant_weight": "U8",
                        "scale": "F8_E4M3",
                        "norm": "BF16",
                    }
                },
            )
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            config = _resolve_encoding(config)
            assert config.quantization_encoding == "float4_e2m1fnx2"

    def test_unknown_dtypes_reported_once_per_dtype(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unmappable dtype warns once per scan, not once per tensor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model-00001-of-00002.safetensors": {
                        "layers.0.weight": "BF16",
                        "layers.0.original_shape": "I64",
                        "layers.1.original_shape": "I64",
                    },
                    "model-00002-of-00002.safetensors": {
                        "layers.2.weight": "BF16",
                        "layers.2.original_shape": "I64",
                        "layers.2.g_idx": "I32",
                    },
                },
            )
            repo = HuggingFaceRepo(repo_id=tmpdir)
            with caplog.at_level(logging.WARNING, logger="max.pipelines"):
                assert "bfloat16" in repo.supported_encodings

        messages = sorted(
            record.getMessage()
            for record in caplog.records
            if "unknown dtype" in record.getMessage()
        )
        # Four unmappable tensors across two shards, two distinct dtypes.
        assert len(messages) == 2
        assert "I32" in messages[0] and "(1 tensor," in messages[0]
        assert "I64" in messages[1] and "(3 tensors," in messages[1]
        # Shard read order is glob order, so only the shared suffix is stable.
        assert "original_shape" in messages[1]

    def test_mixed_bf16_and_f32_selects_bf16_on_gpu(self) -> None:
        """On GPU, bf16 should be preferred over f32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {
                        "weight": "BF16",
                        "bias": "F32",
                    }
                },
            )
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            config = _resolve_encoding(config)
            assert config.quantization_encoding == "bfloat16"

    def test_mixed_bf16_and_f32_ambiguous_on_cpu(self) -> None:
        """On CPU, multiple non-quantized encodings are ambiguous, so
        architecture-level resolution falls back to the architecture's
        declared default_encoding rather than guessing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {
                        "weight": "BF16",
                        "bias": "F32",
                    }
                },
            )
            config = _make_config(tmpdir, device_specs=[CPU_DEVICE_SPEC])
            # Use a default_encoding not present in the repo's file, so a
            # match proves it came from the fallback, not real inference.
            config = _resolve_encoding(config, default_encoding="float8_e4m3fn")
            assert config.quantization_encoding == "float8_e4m3fn"

    def test_sharded_fp8_with_bf16_first_shard(self) -> None:
        """FP8 must be detected even when first shard is BF16-only norms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    # Shard 3: norms/embeddings only (BF16) - may sort first
                    "model-00003-of-00003.safetensors": {
                        "model.norm.weight": "BF16",
                    },
                    # Shard 1: FP8 quantized weights
                    "model-00001-of-00003.safetensors": {
                        "model.layers.0.self_attn.q_proj.weight": "F8_E4M3",
                        "model.layers.0.input_layernorm.weight": "BF16",
                    },
                    # Shard 2: more FP8 weights
                    "model-00002-of-00003.safetensors": {
                        "model.layers.1.self_attn.q_proj.weight": "F8_E4M3",
                    },
                },
            )
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            config = _resolve_encoding(config)
            assert config.quantization_encoding == "float8_e4m3fn"

    def test_gptq_detected_from_local_config_json(self) -> None:
        """gptq should be detected from config.json for local repos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "U8"}})
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"quantization_config": {"quant_method": "gptq"}}, f)
            repo = HuggingFaceRepo(repo_id=tmpdir)
            assert "gptq" in repo.supported_encodings


# ---------------------------------------------------------------------------
# Category C: Weight Path Resolution
# ---------------------------------------------------------------------------


class TestWeightPathResolution:
    """Tests for weight file discovery during architecture-level validation."""

    def test_resolve_weight_path_sharded_safetensors(self) -> None:
        """Sharded safetensors should all be discovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model-00001-of-00002.safetensors": {"w": "BF16"},
                    "model-00002-of-00002.safetensors": {"w": "BF16"},
                },
            )
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            config = _resolve_encoding_and_weight_path(config)
            paths = sorted(str(p) for p in config.weight_path)
            assert paths == [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ]

    def test_prefers_safetensors_over_gguf(self) -> None:
        """When both formats exist, safetensors should be preferred."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                safetensors_files={"model.safetensors": {"w": "BF16"}},
                gguf_files=["model-Q4_K_M.gguf"],
            )
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            config = _resolve_encoding_and_weight_path(config)
            paths = [str(p) for p in config.weight_path]
            assert paths == ["model.safetensors"]

    def test_falls_back_to_gguf_when_only_format(self) -> None:
        """GGUF files should be discovered when no safetensors exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, gguf_files=["model-Q4_K_M.gguf"])
            config = _make_config(tmpdir, device_specs=[CPU_DEVICE_SPEC])
            config = _resolve_encoding_and_weight_path(config)
            paths = [str(p) for p in config.weight_path]
            assert paths == ["model-Q4_K_M.gguf"]

    def test_dtype_cast_fallback_finds_f32_files_for_bf16(self) -> None:
        """On GPU, encoding casts to bf16 but weight files match the original f32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "F32"}})
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            config = _resolve_encoding_and_weight_path(config)
            assert config.quantization_encoding == "bfloat16"
            paths = [str(p) for p in config.weight_path]
            assert paths == ["model.safetensors"]

    def test_explicit_weight_path_skips_discovery(self) -> None:
        """Explicit weight_path should not be overwritten by discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {"w": "BF16"},
                    "other.safetensors": {"w": "BF16"},
                },
            )
            explicit = [Path("model.safetensors")]
            config = _make_config(
                tmpdir, device_specs=[GPU_DEVICE_SPEC], weight_path=explicit
            )
            config = _resolve_encoding_and_weight_path(config)
            assert config.weight_path == [Path("model.safetensors")]

    def test_consolidated_safetensors_excluded(self) -> None:
        """consolidated.safetensors should be excluded when sharded files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "consolidated.safetensors": {"w": "BF16"},
                    "model-00001-of-00002.safetensors": {"w": "BF16"},
                    "model-00002-of-00002.safetensors": {"w": "BF16"},
                },
            )
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            config = _resolve_encoding_and_weight_path(config)
            paths = sorted(str(p) for p in config.weight_path)
            assert "consolidated.safetensors" not in paths
            assert len(paths) == 2


# ---------------------------------------------------------------------------
# Category D: Encoding from Explicit Weight Path
# ---------------------------------------------------------------------------


class TestEncodingFromExplicitWeightPath:
    """Tests for encoding inference when weight_path is explicitly provided."""

    def test_encoding_from_gguf_filename_in_weight_path(self) -> None:
        """GGUF encoding should be inferred from the filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, gguf_files=["model-Q4_K_M.gguf"])
            explicit = [Path("model-Q4_K_M.gguf")]
            config = _make_config(
                tmpdir, device_specs=[CPU_DEVICE_SPEC], weight_path=explicit
            )
            config = _resolve_encoding(config)
            assert config.quantization_encoding == "q4_k"

    def test_encoding_from_remote_safetensors_via_repo(self) -> None:
        """For remote safetensors, encoding is inferred from the repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "BF16"}})
            # Simulate remote: weight_path points to a non-local file (a
            # relative path, not the absolute path under tmpdir), so
            # resolution falls through to the repo-based encoding_for_file
            # lookup instead of the local-file branch.
            explicit = [Path("model.safetensors")]
            config = _make_config(
                tmpdir, device_specs=[GPU_DEVICE_SPEC], weight_path=explicit
            )
            config = _resolve_encoding(config)
            assert config.quantization_encoding == "bfloat16"

    def test_encoding_from_local_safetensors_with_name_hint(self) -> None:
        """Encoding should be parsed from filename when a hint is present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = os.path.join(tmpdir, "model-bf16.safetensors")
            write_fake_safetensors(fp, dtype="BF16")
            explicit = [Path(fp)]
            config = _make_config(
                tmpdir, device_specs=[GPU_DEVICE_SPEC], weight_path=explicit
            )
            config = _resolve_encoding(config)
            assert config.quantization_encoding == "bfloat16"


# ---------------------------------------------------------------------------
# Category E: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Tests that encoding inference and weight path resolution are deterministic.

    These resolve encoding/weight_path multiple times with fresh
    MAXModelConfig instances to exercise the set→list conversion in
    supported_encodings.
    """

    def test_deterministic_mixed_fp4_bf16(self) -> None:
        """FP4+BF16+F32 mixed file must always resolve to fp4."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {
                        "data_weight": "U8",
                        "norm_weight": "BF16",
                        "bias": "F32",
                    }
                },
            )
            results = []
            for _ in range(50):
                config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
                config = _resolve_encoding(config)
                results.append(config.quantization_encoding)
            assert all(r == "float4_e2m1fnx2" for r in results), (
                f"Non-deterministic results: {set(results)}"
            )

    def test_deterministic_mixed_fp8_bf16_f32(self) -> None:
        """FP8+BF16+F32 mixed file must always resolve to fp8."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {
                        "data_weight": "F8_E4M3",
                        "norm_weight": "BF16",
                        "bias": "F32",
                    }
                },
            )
            results = []
            for _ in range(50):
                config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
                config = _resolve_encoding(config)
                results.append(config.quantization_encoding)
            assert all(r == "float8_e4m3fn" for r in results), (
                f"Non-deterministic results: {set(results)}"
            )

    def test_deterministic_weight_path_sharded(self) -> None:
        """Sharded weight path resolution must be stable across runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model-00001-of-00004.safetensors": {"w": "BF16"},
                    "model-00002-of-00004.safetensors": {"w": "BF16"},
                    "model-00003-of-00004.safetensors": {"w": "BF16"},
                    "model-00004-of-00004.safetensors": {"w": "BF16"},
                },
            )
            results = []
            for _ in range(10):
                config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
                config = _resolve_encoding_and_weight_path(config)
                results.append(sorted(str(p) for p in config.weight_path))
            assert all(r == results[0] for r in results), (
                f"Non-deterministic weight paths: {results}"
            )


# ---------------------------------------------------------------------------
# Category F: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases in resolve()."""

    def test_empty_model_path_is_constructible(self) -> None:
        """A model-less config constructs and passes repo-access validation.

        ``validate_repo_access`` only checks a *specified* remote repo, so a
        placeholder config (no model_path, no weight_path) is a no-op rather
        than an error -- requiring a model to run is enforced later, during
        architecture resolution.
        """
        with _resolve_mocks():
            config = _make_config("", device_specs=[CPU_DEVICE_SPEC])
        config.validate_repo_access()  # should not raise

    def test_corrupt_safetensors_suppresses_exception(self) -> None:
        """A corrupt safetensors header must not raise during resolution.

        HuggingFaceRepo._detect_safetensors_encodings_from_files() catches
        per-file parse errors internally and just skips the file, so the
        repo reports zero supported encodings -- same as a genuinely empty
        repo. Architecture-level resolution treats that as "no signal" and
        falls back to default_encoding rather than raising.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_path = os.path.join(tmpdir, "model.safetensors")
            with open(corrupt_path, "wb") as f:
                # Write truncated header (only 4 bytes instead of 8).
                f.write(b"\x00\x00\x00\x00")
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            config = _resolve_encoding(config)
            assert config.quantization_encoding == _DEFAULT_ENCODING

    def test_no_weight_files_in_repo(self) -> None:
        """An empty repo resolves to the architecture's default_encoding.

        No weight files means no signal either way, so resolution falls
        back to default_encoding rather than raising -- weight_path
        discovery itself is not attempted (it would raise for a repo with
        no matching files), matching this test's original scope.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir, device_specs=[GPU_DEVICE_SPEC])
            config = _resolve_encoding(config)
            assert config.weight_path == []
            assert config.quantization_encoding == _DEFAULT_ENCODING

    def test_encoding_for_file_honors_preferred_encoding(self) -> None:
        """encoding_for_file should return preferred_encoding when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {
                        "weight": "U8",
                        "norm": "BF16",
                    }
                },
            )
            repo = HuggingFaceRepo(repo_id=tmpdir)
            result = repo.encoding_for_file(
                "model.safetensors", preferred_encoding="bfloat16"
            )
            assert result == "bfloat16"

    def test_encoding_for_file_without_preferred_uses_priority(self) -> None:
        """Without preferred_encoding, priority should pick fp4 over bf16."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(
                tmpdir,
                {
                    "model.safetensors": {
                        "weight": "U8",
                        "norm": "BF16",
                    }
                },
            )
            repo = HuggingFaceRepo(repo_id=tmpdir)
            result = repo.encoding_for_file("model.safetensors")
            assert result == "float4_e2m1fnx2"


# ---------------------------------------------------------------------------
# Category G: PipelineArgs write-through regressions
# ---------------------------------------------------------------------------


class TestPipelineArgsWriteThroughRegressions:
    """`MAXModelConfig.from_pipeline_args()` rebuilds a fresh `MAXModelConfig`
    on every call, unlike `PipelineConfig.model`, which returns the live
    stored object. Code migrated from the old `PipelineConfig` API kept the
    `config.model.foo = x` write-through pattern, which is silently discarded
    against `PipelineArgs`. These regression tests guard the fixed call
    sites: writes must go through `PipelineArgs`'s own fields/private attrs.
    """

    def test_weights_repo_id_must_be_set_on_args_not_model_property(
        self,
    ) -> None:
        """Regression test (QUA-729/QUA-730): MAXModelConfig.from_pipeline_args()
        rebuilds a fresh MAXModelConfig on every call, so
        `MAXModelConfig.from_pipeline_args(config)._weights_repo_id = x` (the
        pattern GenericOracle used, mirroring the old PipelineConfig API
        where `.model` returned the live stored object) is silently
        discarded. Setting `config._weights_repo_id` directly on the
        PipelineArgs instance is the only way it survives to
        `MAXModelConfig.from_pipeline_args()`, which is what a cross-repo
        weights setup (e.g. a bartowski GGUF repo supplying weights for a
        meta-llama config repo) depends on to find weight files in the right
        repo instead of the (potentially gated, GGUF-less) model repo.
        """
        from max.pipelines.lib import MAXModelConfig, PipelineArgs

        args = PipelineArgs(model_path="meta-llama/Meta-Llama-3-8B-Instruct")

        # The buggy pattern: mutating the returned object's copy is a no-op.
        MAXModelConfig.from_pipeline_args(
            args
        )._weights_repo_id = "bartowski/Meta-Llama-3-8B-Instruct-GGUF"
        assert (
            MAXModelConfig.from_pipeline_args(args).huggingface_weight_repo_id
            == "meta-llama/Meta-Llama-3-8B-Instruct"
        )

        # The fix: set the private attr on PipelineArgs itself.
        args._weights_repo_id = "bartowski/Meta-Llama-3-8B-Instruct-GGUF"
        assert (
            MAXModelConfig.from_pipeline_args(args).huggingface_weight_repo_id
            == "bartowski/Meta-Llama-3-8B-Instruct-GGUF"
        )

    def test_multi_component_manifest_accepts_nested_runtime_kwarg(
        self,
    ) -> None:
        """`PipelineArgs` accepts nested `runtime=PipelineRuntimeConfig(...)`.

        Nested is the canonical shape shared with recipes and
        `PipelineConfig`; runtime knobs like `prefer_module_v3` live on the
        `runtime` sub-config, and denoising-cache settings on the top-level
        `denoising_cache` field, including for multi-component (diffusion)
        manifests.
        """
        from max.pipelines.diffusion.config import DenoisingCacheSettings
        from max.pipelines.lib import (
            ModelManifest,
            PipelineArgs,
            PipelineRuntimeConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_local_repo(tmpdir, {"model.safetensors": {"w": "BF16"}})
            models = ModelManifest(
                {
                    "transformer": _make_config(
                        tmpdir, device_specs=[GPU_DEVICE_SPEC]
                    )
                }
            )

            args = PipelineArgs(
                models=models,
                runtime=PipelineRuntimeConfig(prefer_module_v3=True),
                denoising_cache=DenoisingCacheSettings(taylorseer=True),
            )
            assert args.runtime.prefer_module_v3 is True
            assert args.denoising_cache.taylorseer is True
