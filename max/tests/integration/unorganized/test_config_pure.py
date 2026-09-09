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

import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, PropertyMock, patch

import huggingface_hub
import pytest
from max._entrypoints.cli.config import parse_task_flags
from max.driver import DeviceSpec, accelerator_count
from max.dtype import DType
from max.graph.weights import WeightsFormat
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.context import SamplingParamsGenerationConfigDefaults
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import (
    KVCacheConfig,
    LoRAConfig,
    MAXModelConfig,
    PipelineArgs,
    PipelineConfig,
    PipelineRuntimeConfig,
    SamplingConfig,
)
from max.pipelines.lib.config.config import (
    _apply_speculative_target_architecture,
    _resolve_default_reasoning_parser,
    _resolve_default_structured_output_backend,
    _resolve_default_tool_parser,
    _resolve_overlap_and_device_graph_capture,
)
from max.pipelines.lib.config.model_config import (
    _build_model_config,
    _infer_weight_path,
    _select_dtype_cast,
    _select_quantization_encoding,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.modeling.config_enums import SupportedEncoding
from max.pipelines.modeling.types.task import PipelineTask
from max.pipelines.speculative.config import SpeculativeConfig
from test_common.mocks import (
    mock_hf_repo_access,
    mock_pipeline_config_resolve,
    mock_plan_from_sizes,
)
from test_common.pipeline_model_dummy import DUMMY_GEMMA_ARCH, DUMMY_LLAMA_ARCH
from test_common.registry import prepare_registry

# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #

requires_hf_network = pytest.mark.skipif(
    os.environ.get("HF_HUB_OFFLINE", "0") == "1",
    reason="Verifies weight files against live HuggingFace; presubmit runs "
    "offline, the HF workflow covers this (SERVOPT-900)",
)


def _serve_optimization_arch(
    name: str,
    *,
    supports_overlap_scheduler: bool = True,
    supports_device_graph_capture: bool = True,
    task: PipelineTask = PipelineTask.TEXT_GENERATION,
) -> SimpleNamespace:
    """Minimal architecture stub for serve-optimization resolution tests."""
    return SimpleNamespace(
        name=name,
        task=task,
        supports_overlap_scheduler=supports_overlap_scheduler,
        supports_device_graph_capture=supports_device_graph_capture,
    )


# ===----------------------------------------------------------------------=== #
# Tests for utility methods
# ===----------------------------------------------------------------------=== #


class TestClickFlagParsing:
    """Test suite for the click flag parsing."""

    def test_parse_task_flags(self) -> None:
        """Test parsing of task flags."""
        flags = parse_task_flags(("flag1=value1", "flag2=value2"))
        assert flags == {"flag1": "value1", "flag2": "value2"}

    def test_parse_task_flags_with_dash_prefix(self) -> None:
        """Test parsing of task flags with dash prefix."""
        with pytest.raises(
            ValueError,
            match="Flag must be in format 'flag_name=flag_value', got: --flag3=value3",
        ):
            parse_task_flags(("flag1=value1", "flag2=value2", "--flag3=value3"))

    def test_parse_task_flags_with_space_in_value(self) -> None:
        """Test parsing of task flags with space in value."""
        with pytest.raises(
            ValueError,
            match="Flag must be in format 'flag_name=flag_value', got: flag3 value3",
        ):
            parse_task_flags(("flag1=value1", "flag2=value2", "flag3 value3"))

    def test_parse_task_flags_with_dash_in_flag_name(self) -> None:
        """Test parsing of task flags with dash in flag name."""

        # flag-3 is converted to flag_3
        flags = parse_task_flags(
            ("flag1=value1", "flag2=value2", "flag-3=value3")
        )
        assert flags == {
            "flag1": "value1",
            "flag2": "value2",
            "flag_3": "value3",
        }


class TestPipelineConfigUtilityMethods:
    """Test suite for the refactored utility methods in PipelineConfig."""

    @mock_pipeline_config_resolve
    def test_lora_config_built_when_enabled(self) -> None:
        """LoRA flags build a LoRAConfig when ``enable_lora`` is set."""
        config = PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(
                model_path="test/model",
                enable_lora=True,
                lora_paths=["/path/to/lora1", "/path/to/lora2"],
                max_lora_rank=32,
                enable_prefix_caching=False,
            )
        )

        assert config.lora is not None
        assert config.lora.lora_paths == [
            "/path/to/lora1",
            "/path/to/lora2",
        ]
        assert config.lora.max_lora_rank == 32

    @mock_pipeline_config_resolve
    def test_lora_config_absent_without_enable_lora(self) -> None:
        """LoRA flags alone don't enable LoRA -- only ``enable_lora`` does.

        The CLI supplies a default for every LoRA flag, so their presence
        cannot be read as intent.
        """
        config = PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(
                model_path="test/model",
                max_lora_rank=32,
                max_num_loras=10,
            )
        )
        assert config.lora is None

    @mock_pipeline_config_resolve
    def test_sampling_flags_route_through_from_args(self) -> None:
        """Flat sampling flags land on the built config's sampling."""
        config = PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(
                model_path="test/model",
                enable_structured_output=True,
                enable_penalties=True,
            )
        )

        assert config.sampling is not None
        assert config.sampling.enable_structured_output is True
        assert config.sampling.enable_penalties is True

    @mock_pipeline_config_resolve
    def test_from_args_sampling_with_echo_enabled(self) -> None:
        """``enable_echo`` forces variable logits on the built sampling."""
        config = PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(
                model_path="test/model",
                enable_echo=True,
                enable_min_tokens=True,
            )
        )

        assert config.sampling is not None
        assert config.sampling.enable_min_tokens is True
        assert config.sampling.enable_variable_logits is True

    @mock_pipeline_config_resolve
    def test_unmatched_flat_kwargs_raise(self) -> None:
        """Flat kwargs that route to no config field are rejected."""
        with pytest.raises(ValueError, match="Unmatched kwargs"):
            PipelineArgs.from_flat_kwargs(
                model_path="test/model",
                unknown_param="value",
            )

    @mock_pipeline_config_resolve
    def test_integration_full_config_initialization(
        self,
    ) -> None:
        """Test full integration of all utility methods during config initialization."""
        kwargs = {
            "model_path": "test/model",
            "max_batch_size": 4,
            # LoRA config
            "enable_lora": True,
            "lora_paths": ["/lora1", "/lora2"],
            "max_lora_rank": 64,
            # Draft model config
            "draft_model_path": "/draft/model",
            "draft_quantization_encoding": "float32",
            # Sampling config
            "enable_structured_output": True,
            # Model config with KV cache
            "quantization_encoding": "bfloat16",
            "kv_cache_page_size": 512,
            # LoRA rejects prefix caching at construction.
            "enable_prefix_caching": False,
        }

        config = PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(**kwargs)
        )

        # Should have created all configs correctly
        assert config.runtime.max_batch_size == 4

        # LoRA config
        assert config.lora is not None
        assert config.lora.lora_paths == ["/lora1", "/lora2"]
        assert config.lora.max_lora_rank == 64

        # Draft model config
        assert config.draft_model is not None
        assert config.draft_model.model_path == "/draft/model"
        assert config.draft_model.quantization_encoding == "float32"

        # Sampling config
        assert config.sampling.enable_structured_output is True
        assert config.sampling.enable_penalties is False

        # Model config with KV cache
        assert config.model.quantization_encoding == "bfloat16"
        assert config.model.kv_cache.kv_cache_page_size == 512

    @mock_pipeline_config_resolve
    def test_kv_cache_config_dtype(
        self,
    ) -> None:
        """Test that the KVCache dtype is set correctly."""
        kwargs = {
            "model_path": "trl-internal-testing/tiny-random-LlamaForCausalLM",
            # Draft model config
            "draft_model_path": "/draft/model",
            "draft_quantization_encoding": "float8_e4m3fn",
            # Model config with KV cache
            "quantization_encoding": "float4_e2m1fnx2",
            "kv_cache_page_size": 512,
        }

        config = PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(**kwargs)
        )
        assert config.model.quantization_encoding == "float4_e2m1fnx2"

        assert config.draft_model is not None
        assert config.draft_model.quantization_encoding == "float8_e4m3fn"

        # The KV cache dtype is derived from the quantization encoding on demand.
        assert (
            cache_dtype_for_encoding(
                config.model.quantization_encoding,
                config.model.kv_cache.kv_cache_format,
            )
            == DType.bfloat16
        )
        assert (
            cache_dtype_for_encoding(
                config.draft_model.quantization_encoding,
                config.draft_model.kv_cache.kv_cache_format,
            )
            == DType.bfloat16
        )

    @mock_pipeline_config_resolve
    def test_denoising_cache_survives_runtime_kwargs(self) -> None:
        """CLI ``--taylorseer`` flags reach ``runtime.denoising_cache``."""
        kwargs = {
            "model_path": "test/model",
            # DenoisingCacheConfig fields (--taylorseer etc.)
            "taylorseer": True,
            "taylorseer_cache_interval": 5,
            "taylorseer_warmup_steps": 4,
            "taylorseer_max_order": 1,
            # PipelineRuntimeConfig field that triggers runtime reconstruction
            "max_batch_size": 4,
        }

        config = PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(**kwargs)
        )

        assert config.runtime.max_batch_size == 4
        assert config.runtime.denoising_cache.taylorseer is True
        assert config.runtime.denoising_cache.taylorseer_cache_interval == 5
        assert config.runtime.denoising_cache.taylorseer_warmup_steps == 4
        assert config.runtime.denoising_cache.taylorseer_max_order == 1

    @mock_pipeline_config_resolve
    def test_first_block_caching_survives_runtime_kwargs(self) -> None:
        """CLI ``--first-block-caching`` reaches ``runtime.denoising_cache``."""
        kwargs = {
            "model_path": "test/model",
            "first_block_caching": True,
            "max_batch_size": 4,
        }

        config = PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(**kwargs)
        )

        assert config.runtime.max_batch_size == 4
        assert config.runtime.denoising_cache.first_block_caching is True


class TestNeedsBitmaskConstraints:
    """Tests for the ``PipelineConfig.needs_bitmask_constraints`` property.

    The property drives whether the bitmask path is compiled into the
    sampler graph, the unified Eagle graph, and the D2H pinned buffer.
    Tool-call grammars are server-generated when a tool parser is
    configured, so the bitmask path must wire in for that case even
    without ``--enable-structured-output``.
    """

    @mock_pipeline_config_resolve
    @pytest.mark.parametrize(
        "enable_structured_output,tool_parser,enable_tool_call_constrained_decode,expected",
        [
            # No structured output, no parser: never needs the bitmask path.
            (False, None, True, False),
            (False, None, False, False),
            # User structured output on: always needs it, regardless of the
            # tool-call flag.
            (True, None, True, True),
            (True, None, False, True),
            # Parser configured + tool-call constrained decode on (default):
            # bitmask path wires in for server-generated tool grammars.
            (False, "kimik2_5", True, True),
            (True, "kimik2_5", True, True),
            # Parser configured but tool-call constrained decode disabled: the
            # parser still parses output, but no grammar/bitmask on its account.
            (False, "kimik2_5", False, False),
            # ...unless user structured output independently requires it.
            (True, "kimik2_5", False, True),
        ],
    )
    def test_truth_table(
        self,
        enable_structured_output: bool,
        tool_parser: str | None,
        enable_tool_call_constrained_decode: bool,
        expected: bool,
    ) -> None:
        config = PipelineConfig(
            models=ModelManifest(
                {"main": MAXModelConfig(model_path="test/model")}
            ),
            sampling=SamplingConfig(
                enable_structured_output=enable_structured_output,
                enable_tool_call_constrained_decode=enable_tool_call_constrained_decode,
            ),
            runtime=PipelineRuntimeConfig(tool_parser=tool_parser),
        )
        assert config.needs_bitmask_constraints is expected


class TestSpeculativeArchitectureOverride:
    """Tests for ``_apply_speculative_target_architecture``.

    The override must rewrite ``model.huggingface_config.architectures[0]`` to
    the unified spec-decode target arch. The registry applies it *before*
    resolving ``arch`` so memory estimation / scheduler / parser resolution all
    see the overridden arch (regression guard for #88511).
    """

    @staticmethod
    def _make_config(
        target_arch: str,
        *,
        speculative: bool = True,
        is_dflash: bool = False,
        draft_arch: str | None = None,
    ) -> SimpleNamespace:
        """Build a minimal stand-in exposing the attrs the method reads."""
        model = SimpleNamespace(
            huggingface_config=SimpleNamespace(architectures=[target_arch])
        )
        draft_model = None
        if draft_arch is not None:
            draft_model = SimpleNamespace(
                huggingface_config=SimpleNamespace(architectures=[draft_arch])
            )
        spec = (
            SimpleNamespace(is_dflash=lambda: is_dflash)
            if speculative
            else None
        )
        manifest = {"main": model}
        if draft_model is not None:
            manifest["draft"] = draft_model
        return SimpleNamespace(speculative=spec, manifest=manifest)

    @staticmethod
    def _resolved_arch(cfg: SimpleNamespace) -> str:
        _apply_speculative_target_architecture(cfg.speculative, cfg.manifest)
        return cfg.manifest["main"].huggingface_config.architectures[0]

    def test_deepseek_mtp_no_draft(self) -> None:
        """DeepseekV3 + no draft (NextN baked in) -> unified MTP arch."""
        cfg = self._make_config("DeepseekV3ForCausalLM", draft_arch=None)
        assert self._resolved_arch(cfg) == "UnifiedMTPDeepseekV3ForCausalLM"

    def test_deepseek_eagle3_draft(self) -> None:
        cfg = self._make_config(
            "DeepseekV3ForCausalLM", draft_arch="Eagle3DeepseekV2ForCausalLM"
        )
        assert self._resolved_arch(cfg) == "Eagle3DeepseekV3ForCausalLM"

    def test_deepseek_unrecognized_draft_raises(self) -> None:
        cfg = self._make_config(
            "DeepseekV3ForCausalLM", draft_arch="LlamaForCausalLM"
        )
        with pytest.raises(
            ValueError, match="Unrecognized draft architecture for DeepseekV3"
        ):
            self._resolved_arch(cfg)

    def test_llama_eagle(self) -> None:
        cfg = self._make_config("LlamaForCausalLM")
        assert self._resolved_arch(cfg) == "UnifiedEagleLlama3ForCausalLM"

    def test_llama_dflash(self) -> None:
        cfg = self._make_config("LlamaForCausalLM", is_dflash=True)
        assert self._resolved_arch(cfg) == "UnifiedDflashLlama3ForCausalLM"

    def test_gemma4_mtp(self) -> None:
        cfg = self._make_config(
            "Gemma4ForConditionalGeneration",
            draft_arch="Gemma4AssistantForCausalLM",
        )
        assert self._resolved_arch(cfg) == "UnifiedMTPGemma4ForCausalLM"

    def test_gemma4_unified_dspark(self) -> None:
        cfg = self._make_config(
            "Gemma4UnifiedForConditionalGeneration",
            is_dflash=True,
            draft_arch="Gemma4DSparkModel",
        )
        assert self._resolved_arch(cfg) == "UnifiedDSparkGemma4_12BForCausalLM"

    def test_gemma4_unified_without_dspark_draft_is_noop(self) -> None:
        cfg = self._make_config(
            "Gemma4UnifiedForConditionalGeneration", draft_arch=None
        )
        assert (
            self._resolved_arch(cfg) == "Gemma4UnifiedForConditionalGeneration"
        )

    def test_no_speculative_is_noop(self) -> None:
        cfg = self._make_config(
            "DeepseekV3ForCausalLM", speculative=False, draft_arch=None
        )
        assert self._resolved_arch(cfg) == "DeepseekV3ForCausalLM"

    def test_inkling_mtp_no_draft(self) -> None:
        cfg = self._make_config(
            "InklingForConditionalGeneration", draft_arch=None
        )
        cfg.manifest["main"].huggingface_config.mtp_config = SimpleNamespace(
            num_nextn_predict_layers=8
        )
        assert (
            self._resolved_arch(cfg)
            == "UnifiedMTPInklingForConditionalGeneration"
        )

    def test_inkling_without_mtp_config_is_noop(self) -> None:
        cfg = self._make_config(
            "InklingForConditionalGeneration", draft_arch=None
        )
        assert self._resolved_arch(cfg) == "InklingForConditionalGeneration"

    def test_inkling_mtp_dict_config_no_draft(self) -> None:
        cfg = self._make_config(
            "InklingForConditionalGeneration", draft_arch=None
        )
        cfg.manifest["main"].huggingface_config.mtp_config = {
            "num_nextn_predict_layers": 8
        }
        assert (
            self._resolved_arch(cfg)
            == "UnifiedMTPInklingForConditionalGeneration"
        )


class TestDraftModelDefaultsInheritance:
    """Tests that draft model inherits certain defaults from the target model."""

    @mock_hf_repo_access
    def test_apply_draft_model_defaults_inherits_trust_remote_code(
        self,
    ) -> None:
        """_apply_draft_model_defaults inherits trust_remote_code from target."""
        target_kwargs: dict[str, Any] = {"trust_remote_code": True}
        draft_kwargs: dict[str, Any] = {"model_path": "test/draft"}

        PipelineArgs._apply_draft_model_defaults(draft_kwargs, target_kwargs)

        assert draft_kwargs["trust_remote_code"] is True

    @mock_hf_repo_access
    def test_apply_draft_model_defaults_does_not_inherit_false_trust_remote_code(
        self,
    ) -> None:
        """_apply_draft_model_defaults does not inherit trust_remote_code=False."""
        target_kwargs: dict[str, Any] = {"trust_remote_code": False}
        draft_kwargs: dict[str, Any] = {"model_path": "test/draft"}

        PipelineArgs._apply_draft_model_defaults(draft_kwargs, target_kwargs)

        # trust_remote_code should not be added when target has False
        assert "trust_remote_code" not in draft_kwargs

    @mock_hf_repo_access
    def test_apply_draft_model_defaults_preserves_explicit_trust_remote_code(
        self,
    ) -> None:
        """Explicit draft trust_remote_code is not overridden."""
        target_kwargs: dict[str, Any] = {"trust_remote_code": True}
        draft_kwargs: dict[str, Any] = {
            "model_path": "test/draft",
            "trust_remote_code": False,
        }

        PipelineArgs._apply_draft_model_defaults(draft_kwargs, target_kwargs)

        # Explicit False should be preserved
        assert draft_kwargs["trust_remote_code"] is False

    @mock_hf_repo_access
    def test_apply_draft_model_defaults_inherits_device_specs(self) -> None:
        """_apply_draft_model_defaults inherits device_specs from target."""
        target_devices = [DeviceSpec.cpu()]
        target_kwargs: dict[str, Any] = {"device_specs": target_devices}
        draft_kwargs: dict[str, Any] = {"model_path": "test/draft"}

        PipelineArgs._apply_draft_model_defaults(draft_kwargs, target_kwargs)

        assert draft_kwargs["device_specs"] == target_devices

    @mock_hf_repo_access
    def test_apply_draft_model_defaults_preserves_explicit_device_specs(
        self,
    ) -> None:
        """Explicit draft device_specs is not overridden."""
        target_devices = [DeviceSpec.cpu()]
        draft_devices = [DeviceSpec.accelerator()]
        target_kwargs: dict[str, Any] = {"device_specs": target_devices}
        draft_kwargs: dict[str, Any] = {
            "model_path": "test/draft",
            "device_specs": draft_devices,
        }

        PipelineArgs._apply_draft_model_defaults(draft_kwargs, target_kwargs)

        assert draft_kwargs["device_specs"] == draft_devices

    @mock_hf_repo_access
    def test_apply_draft_model_defaults_inherits_data_parallel_degree(
        self,
    ) -> None:
        """_apply_draft_model_defaults inherits data_parallel_degree from target."""
        target_kwargs: dict[str, Any] = {"data_parallel_degree": 8}
        draft_kwargs: dict[str, Any] = {"model_path": "test/draft"}

        PipelineArgs._apply_draft_model_defaults(draft_kwargs, target_kwargs)

        assert draft_kwargs["data_parallel_degree"] == 8

    @mock_hf_repo_access
    def test_apply_draft_model_defaults_preserves_explicit_data_parallel_degree(
        self,
    ) -> None:
        """Explicit draft data_parallel_degree is not overridden."""
        target_kwargs: dict[str, Any] = {"data_parallel_degree": 8}
        draft_kwargs: dict[str, Any] = {
            "model_path": "test/draft",
            "data_parallel_degree": 4,
        }

        PipelineArgs._apply_draft_model_defaults(draft_kwargs, target_kwargs)

        assert draft_kwargs["data_parallel_degree"] == 4

    @mock_hf_repo_access
    def test_apply_draft_model_defaults_does_not_inherit_quantization_encoding(
        self,
    ) -> None:
        """_apply_draft_model_defaults does NOT inherit quantization_encoding.

        EAGLE3 and other draft models typically use bfloat16 regardless of
        the target model's quantization. The draft model should auto-detect
        its encoding from its weights, not inherit from target.
        """
        target_kwargs: dict[str, Any] = {
            "quantization_encoding": "float4_e2m1fnx2"
        }
        draft_kwargs: dict[str, Any] = {"model_path": "test/draft"}

        PipelineArgs._apply_draft_model_defaults(draft_kwargs, target_kwargs)

        # quantization_encoding should NOT be inherited
        assert "quantization_encoding" not in draft_kwargs


class TestDraftModelQuantizationEncoding:
    """Tests that draft model quantization_encoding is independent from target."""

    _MODEL = "trl-internal-testing/tiny-random-LlamaForCausalLM"

    @staticmethod
    def _run_speculative_memory_resolution(
        config: PipelineConfig,
        *,
        draft_max_seq_len: int = 131072,
        draft_encoding: SupportedEncoding = "bfloat16",
    ) -> None:
        """Run _validate_speculative_model_configs with mocked internals.

        Sets the target encoding to ``"float8_e4m3fn"`` and defaults the
        draft encoding to ``draft_encoding`` up front, mirroring the
        construction-time resolution (auto-detection from weights).

        Args:
            config: The pipeline config to resolve.
            draft_max_seq_len: Value returned by the draft arch config's
                ``get_max_seq_len()``.  Defaults to a large value so the
                clamping path is *not* exercised unless explicitly requested.
            draft_encoding: Encoding to set on the draft model during
                architecture resolution (simulating auto-detection).
        """
        mock_draft_arch_config = Mock()
        mock_draft_arch_config.get_max_seq_len.return_value = draft_max_seq_len

        mock_arch = Mock()
        mock_arch.pipeline_model.estimate_weights_size.return_value = 0
        mock_arch.config.initialize.return_value = mock_draft_arch_config

        # Encodings settle at construction, so set them up front.
        models = config.models.with_override(
            "main", quantization_encoding="float8_e4m3fn"
        )
        assert config.draft_model is not None
        if config.draft_model.quantization_encoding is None:
            models = models.with_override(
                "draft", quantization_encoding=draft_encoding
            )
        config = config.model_copy(update={"models": models})

        with patch.object(
            PipelineConfig,
            "_validate_model_config_against_arch",
        ):
            config._validate_speculative_model_configs(
                target_arch=mock_arch, draft_arch=mock_arch
            )


# float32 safetensors for a repo that ships no bfloat16 files, mirroring a
# checkpoint like ``nvidia/Kimi-K2.6-Eagle3``.
_F32_SAFETENSORS = [
    Path("model-00001-of-00002.safetensors"),
    Path("model-00002-of-00002.safetensors"),
]


def _make_f32_only_repo() -> Mock:
    """Build a fake ``HuggingFaceRepo`` whose only weights are float32."""

    def files_for_encoding(
        encoding: SupportedEncoding,
        weights_format: WeightsFormat | None = None,
    ) -> dict[WeightsFormat, list[Path]]:
        if encoding == "float32":
            return {WeightsFormat.safetensors: list(_F32_SAFETENSORS)}
        return {}

    repo = Mock()
    repo.repo_id = "test/f32-only"
    repo.repo_type = "online"
    repo.supported_encodings = ["float32"]
    repo.files_for_encoding = Mock(side_effect=files_for_encoding)
    # A float32-only repo reports float32 regardless of the preferred encoding.
    repo.encoding_for_file = Mock(return_value="float32")
    return repo


class TestFloat32WeightFallbackScoping:
    """Regression tests for the float32 -> 16-bit weight-path fallback.

    ``_infer_weight_path`` falls back to a repo's float32 safetensors
    when a float16/bfloat16 graph has no matching files. That fallback is
    scoped to diffuser sub-components (``subfolder`` set); it must NOT fire
    for architecture-validated models (LLMs, speculative-decoding draft
    models), where eagerly binding ``weight_path`` to the float32 checkpoint
    makes the given-encoding validation flip ``quantization_encoding`` to
    float32 and drop the requested bfloat16.

    Regression guard for KERN-3167: the NVFP4 Kimi-K2.6 Eagle recipes
    configure a bfloat16 draft model whose HF repo ships only float32
    safetensors; the unscoped fallback flipped it to float32, which the Eagle3
    architecture does not support (``quantization_encoding of 'float32' not
    supported by MAX engine``).
    """

    @mock_hf_repo_access
    def test_draft_model_bf16_encoding_preserved_over_f32_only_repo(
        self,
    ) -> None:
        """An LLM/draft model keeps bfloat16 (cast from float32), not float32.

        Requested bfloat16, repo has only float32 safetensors, no
        ``subfolder``. The best-effort pass must not bind ``weight_path``, so
        the given-encoding resolution casts float32 -> bfloat16 (preserving
        the requested encoding) instead of flipping to float32.
        """
        config = MAXModelConfig(
            model_path="nvidia/Kimi-K2.6-Eagle3",
            quantization_encoding="bfloat16",
        )
        assert config.subfolder is None

        with (
            patch.object(
                MAXModelConfig,
                "huggingface_weight_repo",
                new_callable=PropertyMock,
                return_value=_make_f32_only_repo(),
            ),
            patch(
                "max.pipelines.lib.config.model_config.supported_encoding_supported_on",
                return_value=True,
            ),
        ):
            # Best-effort (pre-architecture) pass must not bind weight_path.
            assert _infer_weight_path(config, "bfloat16", None) == []

            # Architecture-level given-encoding resolution.
            encoding = _select_quantization_encoding(config, "bfloat16")
            cast_from, cast_to = _select_dtype_cast(config, "bfloat16")

        # The requested bfloat16 is preserved; the float32 weights are cast at
        # load time, recorded in the dtype-cast bookkeeping.
        assert encoding == "bfloat16"
        assert cast_from == "float32"
        assert cast_to == "bfloat16"

    @mock_hf_repo_access
    def test_no_given_encoding_f32_only_repo_casts_to_bfloat16(self) -> None:
        """Architecture-level resolution alone still casts f32 -> bf16.

        No ``quantization_encoding`` given, repo has only float32 weights,
        no ``subfolder``. Calls ``_select_quantization_encoding`` directly to
        verify the no-given-encoding path applies the float32 -> bfloat16 GPU
        cast. Regression guard for a model whose repo ships only float32
        weights: without this cast it would silently run in float32 on GPU
        instead of the expected bfloat16.
        """
        config = MAXModelConfig(model_path="test/f32-only")
        assert config.quantization_encoding is None

        with (
            patch.object(
                MAXModelConfig,
                "huggingface_weight_repo",
                new_callable=PropertyMock,
                return_value=_make_f32_only_repo(),
            ),
            patch(
                "max.pipelines.lib.config.model_config.supported_encoding_supported_on",
                return_value=True,
            ),
        ):
            encoding = _select_quantization_encoding(config, "bfloat16")
            cast_from, cast_to = _select_dtype_cast(config, "bfloat16")

        assert encoding == "bfloat16"
        assert cast_from == "float32"
        assert cast_to == "bfloat16"

    @mock_hf_repo_access
    def test_diffuser_subcomponent_f32_fallback_still_resolves(self) -> None:
        """A diffuser sub-component (``subfolder`` set) still gets the fallback.

        This is the mixed-precision FLUX.2 case the fallback was added for: a
        bfloat16 component whose checkpoint ships float32 safetensors. The
        fallback must still resolve ``weight_path`` to the float32 files while
        leaving the requested bfloat16 encoding in place.
        """
        config = MAXModelConfig(
            model_path="black-forest-labs/FLUX.2-dev",
            subfolder="text_encoder",
            quantization_encoding="bfloat16",
        )

        with patch.object(
            MAXModelConfig,
            "huggingface_weight_repo",
            new_callable=PropertyMock,
            return_value=_make_f32_only_repo(),
        ):
            resolved_weight_path = _infer_weight_path(config, "bfloat16", None)

        assert resolved_weight_path == _F32_SAFETENSORS
        assert config.quantization_encoding == "bfloat16"


@prepare_registry
@mock_plan_from_sizes
def test_validate_model_path__bad_repo_provided() -> None:
    # This test requires a HF call to check that this repo is not valid.
    # The config layer's model builder is what loads the repo; plain
    # construction records the fields without touching the network.
    with pytest.raises(Exception):
        _build_model_config(MAXModelConfig, model_path="bert-base-asdfasdf")


def test_config_init__raises_with_no_model_path() -> None:
    # A weight path with no model path and no derivable repo is an error
    # when the config layer builds the model.
    with pytest.raises(ValueError):
        _build_model_config(MAXModelConfig, weight_path=[Path("file.gguf")])


@requires_hf_network
@prepare_registry
def test_config_post_init__with_weight_path_but_no_model_path() -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": _build_model_config(
                    MAXModelConfig,
                    weight_path=[
                        Path(
                            "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-q4_0.gguf"
                        )
                    ],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )

    assert config.model.model_path == "modularai/Llama-3.1-8B-Instruct-GGUF"
    assert config.model.weight_path == [Path("llama-3.1-8b-instruct-q4_0.gguf")]


@requires_hf_network
@prepare_registry
@mock_plan_from_sizes
def test_config_post_init__other_repo_weights(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": _build_model_config(
                    MAXModelConfig,
                    model_path=llama_3_1_8b_instruct_local_path,
                    weight_path=[
                        Path(
                            "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-q4_0.gguf"
                        )
                    ],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )

    assert (
        config.model._weights_repo_id == "modularai/Llama-3.1-8B-Instruct-GGUF"
    )
    assert config.model.weight_path == [Path("llama-3.1-8b-instruct-q4_0.gguf")]


@requires_hf_network
def test_config_init__reformats_with_str_weights_path(
    modular_ai_llama_3_1_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    # We expect this to convert the string.
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=modular_ai_llama_3_1_local_path,
                    weight_path=[
                        Path(
                            "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-q4_0.gguf"
                        )
                    ],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )

    assert isinstance(config.model.weight_path, list)
    assert len(config.model.weight_path) == 1
    assert isinstance(config.model.weight_path[0], Path)


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
def test_validate_model_path__correct_repo_id_provided(
    modular_ai_llama_3_1_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=modular_ai_llama_3_1_local_path,
                    quantization_encoding="bfloat16",
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )

    assert config.model.model_path == modular_ai_llama_3_1_local_path


@requires_hf_network
@prepare_registry
@mock_plan_from_sizes
def test_config__test_incompatible_quantization_encoding(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    """Arch-dependent encoding validation runs on the ``from_args`` path."""
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    with pytest.raises(ValueError, match="'q4_k' not supported by MAX engine"):
        # This should raise: the dummy Llama arch does not support q4_k.
        PipelineConfig.from_args(
            PipelineArgs(
                model_path=llama_3_1_8b_instruct_local_path,
                quantization_encoding="q4_k",
                weight_path=[
                    Path(
                        "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-f32.gguf"
                    )
                ],
                max_length=1,
                runtime=PipelineRuntimeConfig(
                    max_batch_size=1,
                    prefer_module_v3=True,
                ),
            )
        )

    # This should not raise, as float32 == f32.
    PipelineConfig.from_args(
        PipelineArgs(
            model_path=llama_3_1_8b_instruct_local_path,
            quantization_encoding="float32",
            weight_path=[
                Path(
                    "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-f32.gguf"
                )
            ],
            max_length=1,
            runtime=PipelineRuntimeConfig(
                max_batch_size=1,
                prefer_module_v3=True,
            ),
        )
    )


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
@prepare_registry
@mock_plan_from_sizes
def test_config__test_quantization_encoding_with_dtype_casting(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # Float32 <-> bfloat16 casting is always enabled,
    # so this should succeed by casting bfloat16 weights to float32.
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    quantization_encoding="float32",
                    max_length=1,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            max_batch_size=1,
            prefer_module_v3=True,
        ),
    )
    assert (
        cache_dtype_for_encoding(
            config.model.quantization_encoding,
            config.model.kv_cache.kv_cache_format,
        )
        == DType.float32
    )


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
@prepare_registry
@mock_plan_from_sizes
def test_config__test_quantization_encoding_with_dtype_casting2(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # This should pass, because the flag also supports casting bfloat16 weights
    # to float32.
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    quantization_encoding="float32",
                    max_length=1,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            max_batch_size=1,
            prefer_module_v3=True,
        ),
    )
    assert (
        cache_dtype_for_encoding(
            config.model.quantization_encoding,
            config.model.kv_cache.kv_cache_format,
        )
        == DType.float32
    )


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
@prepare_registry
@mock_plan_from_sizes
def test_config__test_quantization_encoding_with_dtype_casting3(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # This should not raise, as float32 <-> bfloat16 casting is always enabled
    # and the quantization encoding is set to bfloat16.
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    quantization_encoding="bfloat16",
                    max_length=1,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            max_batch_size=1,
            prefer_module_v3=True,
        ),
    )
    assert (
        cache_dtype_for_encoding(
            config.model.quantization_encoding,
            config.model.kv_cache.kv_cache_format,
        )
        == DType.bfloat16
    )


@pytest.mark.skip(
    "TODO: This test is failing due to some int vs. MagicMock mismatch"
)
@prepare_registry
@mock_plan_from_sizes
def test_config__test_retrieve_factory_with_known_architecture(
    modular_ai_llama_3_1_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    config = PipelineArgs(
        model_path=modular_ai_llama_3_1_local_path,
        quantization_encoding="bfloat16",
        max_length=1,
        runtime=PipelineRuntimeConfig(
            max_batch_size=1,
            prefer_module_v3=True,
        ),
    )

    PIPELINE_REGISTRY.retrieve_factory(PipelineConfig.from_args(config))


@prepare_registry
@mock_plan_from_sizes
@requires_hf_network
def test_config__test_retrieve_factory_with_unsupported_model_path(
    gemma_3_1b_it_local_path: str,
) -> None:
    # Construction leaves unregistered architectures alone; the registry
    # rejects them when the pipeline factory is retrieved.
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=gemma_3_1b_it_local_path, max_length=1
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            max_batch_size=1,
            prefer_module_v3=True,
        ),
    )

    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # Should raise an error since HuggingFace fallback is removed.
    with pytest.raises(ValueError, match="No architecture found for"):
        PIPELINE_REGISTRY.retrieve_factory(config)


class LimitedPickler(pickle.Unpickler):
    """A custom Unpickler class that checks for transformer modules."""

    def find_class(self, module: str, name: str) -> type:
        if module.startswith("transformers"):
            raise AssertionError(
                "Tried to unpickle class from transformers module, raising an "
                "error because this may break in serving."
            )
        return super().find_class(module, name)


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
def test_config_is_picklable(
    tmp_path: Path, modular_ai_llama_3_1_local_path: str
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=modular_ai_llama_3_1_local_path,
                    quantization_encoding="bfloat16",
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )

    config.model._huggingface_config = None

    pickle_path = tmp_path / "config.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(config, f)

    with open(pickle_path, "rb") as f:
        limited_pickler = LimitedPickler(f)
        loaded_config = limited_pickler.load()

    assert loaded_config == config


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
@prepare_registry
def test_config__validates_supported_device(
    modular_ai_llama_3_1_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # Valid device/encoding combinations.
    _ = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=modular_ai_llama_3_1_local_path,
                    device_specs=[DeviceSpec.cpu()],
                    quantization_encoding="float32",
                    max_length=1,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )

    if accelerator_count() == 0:
        with pytest.raises(ValueError):
            _ = PipelineConfig(
                models=ModelManifest(
                    {
                        "main": MAXModelConfig(
                            model_path=modular_ai_llama_3_1_local_path,
                            device_specs=[DeviceSpec.accelerator()],
                            quantization_encoding="float32",
                            max_length=1,
                        )
                    }
                ),
                runtime=PipelineRuntimeConfig(
                    prefer_module_v3=True,
                ),
            )
    else:
        _ = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=modular_ai_llama_3_1_local_path,
                        device_specs=[DeviceSpec.accelerator()],
                        quantization_encoding="bfloat16",
                        max_length=1,
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(
                prefer_module_v3=True,
            ),
        )

    with pytest.raises(
        ValueError, match="not compatible with the selected device type 'cpu'"
    ):
        # Invalid device/encoding combinations.
        PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=modular_ai_llama_3_1_local_path,
                        device_specs=[DeviceSpec.cpu()],
                        quantization_encoding="bfloat16",
                        max_length=1,
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(
                prefer_module_v3=True,
            ),
        )


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
@prepare_registry
def test_config__validates_lora_configuration(
    llama_3_1_8b_instruct_local_path: str, llama_3_1_8b_lora_local_path: str
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # Test LoRA configuration with valid config
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    device_specs=[DeviceSpec.accelerator()],
                    quantization_encoding="bfloat16",
                    kv_cache=KVCacheConfig(enable_prefix_caching=False),
                    max_length=1,
                )
            }
        ),
        lora=LoRAConfig(
            enable_lora=True, lora_paths=[llama_3_1_8b_lora_local_path]
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )
    assert config.lora is not None
    assert config.lora.lora_paths[0] == llama_3_1_8b_lora_local_path
    assert config.lora.max_lora_rank == 16
    assert config.lora.max_num_loras == 1


@prepare_registry
@mock_plan_from_sizes
@requires_hf_network
def test_config__validates_lora_only_supported_for_llama(
    gemma_3_1b_it_local_path: str,
) -> None:
    """Test that LoRA validation fails for non-Llama models."""

    PIPELINE_REGISTRY.register(DUMMY_GEMMA_ARCH, allow_override=True)

    # Test that enabling LoRA on a non-Llama model raises ValueError
    with pytest.raises(
        ValueError,
        match=r"LoRA is not currently supported for architecture.*LoRA support is currently only available for Llama-3\.x models",
    ):
        _ = PipelineConfig.from_args(
            PipelineArgs(
                model_path=gemma_3_1b_it_local_path,
                device_specs=[DeviceSpec.accelerator()],
                quantization_encoding="bfloat16",
                kv_cache=KVCacheConfig(enable_prefix_caching=False),
                max_length=1,
                lora=LoRAConfig(
                    enable_lora=True, lora_paths=["/some/lora/path"]
                ),
                runtime=PipelineRuntimeConfig(
                    prefer_module_v3=True,
                ),
            )
        )


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
@prepare_registry
@mock_plan_from_sizes
def test_config__validates_lora_works_for_llama(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    """Test that LoRA validation passes for Llama models."""
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    device_specs=[DeviceSpec.accelerator()],
                    quantization_encoding="bfloat16",
                    kv_cache=KVCacheConfig(enable_prefix_caching=False),
                    max_length=1,
                )
            }
        ),
        lora=LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"]),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )

    # Verify LoRA config was created successfully
    assert config.lora is not None
    assert config.lora.enable_lora is True
    assert config.lora.lora_paths == ["/some/lora/path"]


@prepare_registry
@mock_plan_from_sizes
@requires_hf_network
def test_config__validates_lora_incompatible_with_prefix_caching(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    """Test that LoRA and prefix caching cannot be enabled together."""
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # Test that enabling both LoRA and prefix caching raises ValueError
    with pytest.raises(
        ValueError,
        match=r"LoRA is not compatible with prefix caching\. Please disable prefix caching by using the --no-enable-prefix-caching flag\.",
    ):
        _ = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=llama_3_1_8b_instruct_local_path,
                        device_specs=[DeviceSpec.accelerator()],
                        quantization_encoding="bfloat16",
                        kv_cache=KVCacheConfig(enable_prefix_caching=True),
                        max_length=1,
                    )
                }
            ),
            lora=LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"]),
            runtime=PipelineRuntimeConfig(
                prefer_module_v3=True,
            ),
        )


@prepare_registry
@mock_plan_from_sizes
@requires_hf_network
@pytest.mark.skipif(
    accelerator_count() > 1, reason="Test requires single GPU or CPU"
)
def test_config__validates_lora_single_device_only(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    device_specs=[DeviceSpec.accelerator()],
                    quantization_encoding="bfloat16",
                    kv_cache=KVCacheConfig(enable_prefix_caching=False),
                    max_length=1,
                )
            }
        ),
        lora=LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"]),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )
    assert config.lora is not None
    assert config.lora.enable_lora is True


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
@prepare_registry
@mock_plan_from_sizes
@pytest.mark.skipif(
    accelerator_count() < 2, reason="Test requires multiple GPUs"
)
def test_config__validates_lora_fails_with_multiple_devices(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    with pytest.raises(
        ValueError,
        match=r"LoRA is currently not supported with the number of devices > 1\.",
    ):
        _ = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=llama_3_1_8b_instruct_local_path,
                        device_specs=[
                            DeviceSpec.accelerator(),
                            DeviceSpec.accelerator(),
                        ],
                        quantization_encoding="bfloat16",
                        kv_cache=KVCacheConfig(enable_prefix_caching=False),
                        max_length=1,
                    )
                }
            ),
            lora=LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"]),
            runtime=PipelineRuntimeConfig(
                prefer_module_v3=True,
            ),
        )

    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    device_specs=[
                        DeviceSpec.accelerator(),
                        DeviceSpec.accelerator(),
                    ],
                    quantization_encoding="bfloat16",
                    max_length=1,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )
    assert config.lora is None


def test_manifest_discovers_diffusion_components() -> None:
    """Test that ModelManifest discovers components for a diffusion pipeline."""
    from transformers import PretrainedConfig

    diffusion_model = "hf-internal-testing/tiny-stable-diffusion-torch"

    # Manifest discovery reads the real per-component configs, so offline
    # runs need the snapshot in the local HF cache (cache contents vary by
    # CI runner); online lanes always cover this test.
    if huggingface_hub.constants.HF_HUB_OFFLINE:
        try:
            huggingface_hub.snapshot_download(
                repo_id=diffusion_model, local_files_only=True
            )
        except huggingface_hub.errors.LocalEntryNotFoundError:
            pytest.skip(
                f"{diffusion_model} is not in the local HF cache and "
                "HF_HUB_OFFLINE is enabled"
            )

    manifest = ModelManifest.from_model_path(diffusion_model)

    # ModelManifest should have discovered per-component configs.
    expected_components = ["vae", "unet", "text_encoder"]
    for component in expected_components:
        assert component in manifest, (
            f"manifest should contain {component} component"
        )
        # Each component should have a valid huggingface_config.
        assert isinstance(
            manifest[component].huggingface_config, PretrainedConfig
        )

    # Metadata should contain the pipeline class name.
    assert "_class_name" in manifest.metadata
    assert "StableDiffusion" in manifest.metadata["_class_name"]


class TestSamplingConfig:
    """Test suite for SamplingConfig."""

    def test_from_generation_config_sampling_defaults_with_repetition_penalty(
        self,
    ) -> None:
        """Test that enable_penalties is True when repetition_penalty is set to non-default value."""
        # Create sampling defaults with repetition_penalty=1.05
        sampling_defaults = SamplingParamsGenerationConfigDefaults(
            repetition_penalty=1.05
        )

        # Create SamplingConfig from the defaults
        sampling_config = (
            SamplingConfig.from_generation_config_sampling_defaults(
                sampling_defaults
            )
        )

        # Assert that enable_penalties is True
        assert sampling_config.enable_penalties is True

    def test_from_generation_config_sampling_defaults_with_default_repetition_penalty(
        self,
    ) -> None:
        """Test that enable_penalties is False when repetition_penalty is at default value."""
        # Create sampling defaults with repetition_penalty=1.0 (default)
        sampling_defaults = SamplingParamsGenerationConfigDefaults(
            repetition_penalty=1.0
        )

        # Create SamplingConfig from the defaults
        sampling_config = (
            SamplingConfig.from_generation_config_sampling_defaults(
                sampling_defaults
            )
        )

        # Assert that enable_penalties is False (since 1.0 is the default)
        assert sampling_config.enable_penalties is False

    def test_from_generation_config_sampling_defaults_without_penalties(
        self,
    ) -> None:
        """Test that enable_penalties is False when no penalty parameters are set."""
        # Create sampling defaults without any penalty parameters
        sampling_defaults = SamplingParamsGenerationConfigDefaults(
            temperature=0.7, top_k=50
        )

        # Create SamplingConfig from the defaults
        sampling_config = (
            SamplingConfig.from_generation_config_sampling_defaults(
                sampling_defaults
            )
        )

        # Assert that enable_penalties is False
        assert sampling_config.enable_penalties is False


@mock_pipeline_config_resolve
@pytest.mark.parametrize(
    (
        "arch_name,supports_overlap_scheduler,supports_device_graph_capture,"
        "max_batch_size,force,is_cuda,expected_device_graph_capture"
    ),
    [
        ("LlamaForCausalLM", True, True, 16, False, True, True),
        ("DeepseekV2ForCausalLM", False, False, 16, False, True, False),
        ("DeepseekV3ForCausalLM", True, True, 16, False, True, True),
        ("DeepseekV32ForCausalLM", True, True, 16, False, True, True),
        ("DeepseekV3ForCausalLMNextN", True, True, 16, False, True, True),
        ("KimiK25ForConditionalGeneration", True, True, 16, False, True, True),
        ("UnifiedEagleLlama3ForCausalLM", True, True, 16, False, True, True),
        ("LlamaForCausalLM", True, True, 16, False, False, False),
        ("LlamaForCausalLM", True, True, None, False, True, True),
        ("LlamaForCausalLM", True, True, 16, True, True, False),
        ("SomeOtherArchitecture", True, True, 16, False, True, True),
    ],
)
def test_validate_and_resolve_overlap_scheduler__auto_enable_device_graph_capture(
    arch_name: str,
    supports_overlap_scheduler: bool,
    supports_device_graph_capture: bool,
    max_batch_size: int | None,
    force: bool,
    is_cuda: bool,
    expected_device_graph_capture: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mock architecture_name so we don't reach out to HF for the config.
    monkeypatch.setattr(
        MAXModelConfig, "architecture_name", property(lambda self: arch_name)
    )
    arch = _serve_optimization_arch(
        arch_name,
        supports_overlap_scheduler=supports_overlap_scheduler,
        supports_device_graph_capture=supports_device_graph_capture,
    )
    monkeypatch.setattr(
        "max.pipelines.lib.config.config.accelerator_api",
        Mock(return_value="cuda" if is_cuda else "cpu"),
    )

    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            force=force,
            max_batch_size=max_batch_size,
        ),
    )
    device_graph_capture, enable_overlap_scheduler = (
        _resolve_overlap_and_device_graph_capture(
            config.runtime, config.sampling, config.lora, config.model, arch
        )
    )

    assert device_graph_capture is expected_device_graph_capture
    if expected_device_graph_capture:
        assert enable_overlap_scheduler is True


@mock_pipeline_config_resolve
def test_validate_and_resolve_overlap_scheduler__no_auto_enable_for_non_text_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embeddings (and other non-text-generation) architectures must not
    auto-enable the overlap scheduler or device graph capture.

    Both features only support ``PipelineTask.TEXT_GENERATION`` (see
    ``get_pipeline_for_task`` in registry.py). An embeddings architecture with
    default ``supports_* = True`` (e.g. ``MPNetForMaskedLM``) would otherwise
    be incorrectly auto-enabled and crash pipeline construction. Regression
    test for QUA-460.
    """
    arch_name = "MPNetForMaskedLM"
    arch = _serve_optimization_arch(
        arch_name,
        task=PipelineTask.EMBEDDINGS_GENERATION,
    )

    monkeypatch.setattr(
        MAXModelConfig, "architecture_name", property(lambda self: arch_name)
    )
    monkeypatch.setattr(
        "max.pipelines.lib.config.config.accelerator_api",
        Mock(return_value="cuda"),
    )

    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(max_batch_size=16),
    )
    device_graph_capture, enable_overlap_scheduler = (
        _resolve_overlap_and_device_graph_capture(
            config.runtime, config.sampling, config.lora, config.model, arch
        )
    )

    assert device_graph_capture is False
    assert enable_overlap_scheduler is False


@mock_pipeline_config_resolve
def test_validate_and_resolve_overlap_scheduler__no_device_graph_capture_for_prefill_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Device graph capture is not supported for prefill-only workers."""
    arch_name = "LlamaForCausalLM"
    monkeypatch.setattr(
        MAXModelConfig, "architecture_name", property(lambda self: arch_name)
    )
    arch = _serve_optimization_arch(arch_name)
    monkeypatch.setattr(
        "max.pipelines.lib.config.config.accelerator_api",
        Mock(return_value="cuda"),
    )

    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            max_batch_size=16,
            pipeline_role="prefill_only",
        ),
    )
    device_graph_capture, enable_overlap_scheduler = (
        _resolve_overlap_and_device_graph_capture(
            config.runtime, config.sampling, config.lora, config.model, arch
        )
    )

    # Overlap scheduling should be auto-enabled for prefill_only.
    assert enable_overlap_scheduler is True
    # But device graph capture should NOT be auto-enabled.
    assert device_graph_capture is False


@mock_pipeline_config_resolve
def test_validate_and_resolve_overlap_scheduler__auto_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Auto-enable overlap scheduler for architectures that declare support.
    for arch_name in (
        "LlamaForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "DeepseekV3ForCausalLMNextN",
    ):
        arch = _serve_optimization_arch(arch_name)
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path="test/model",
                        device_specs=[DeviceSpec.accelerator()],
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(),
        )
        _device_graph_capture, enable_overlap_scheduler = (
            _resolve_overlap_and_device_graph_capture(
                config.runtime, config.sampling, config.lora, config.model, arch
            )
        )
        assert enable_overlap_scheduler is True

    # Architectures that opt out of overlap scheduler are not auto-enabled.
    deepseek_v2 = _serve_optimization_arch(
        "DeepseekV2ForCausalLM",
        supports_overlap_scheduler=False,
        supports_device_graph_capture=False,
    )
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(),
    )
    _device_graph_capture, enable_overlap_scheduler = (
        _resolve_overlap_and_device_graph_capture(
            config.runtime,
            config.sampling,
            config.lora,
            config.model,
            deepseek_v2,
        )
    )
    assert enable_overlap_scheduler is False

    # Don't override if the device is CPU
    llama_arch = _serve_optimization_arch("LlamaForCausalLM")
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.cpu()],
                )
            }
        ),
    )
    _device_graph_capture, enable_overlap_scheduler = (
        _resolve_overlap_and_device_graph_capture(
            config.runtime,
            config.sampling,
            config.lora,
            config.model,
            llama_arch,
        )
    )
    assert enable_overlap_scheduler is False

    # Don't override if variable logits are enabled
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        sampling=SamplingConfig(enable_variable_logits=True),
    )
    _device_graph_capture, enable_overlap_scheduler = (
        _resolve_overlap_and_device_graph_capture(
            config.runtime,
            config.sampling,
            config.lora,
            config.model,
            llama_arch,
        )
    )
    assert enable_overlap_scheduler is False

    # Auto-enable for DI pipeline roles (prefill_only, decode_only)
    for role in ("prefill_only", "decode_only"):
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path="test/model",
                        device_specs=[DeviceSpec.accelerator()],
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(pipeline_role=role),
        )
        _device_graph_capture, enable_overlap_scheduler = (
            _resolve_overlap_and_device_graph_capture(
                config.runtime,
                config.sampling,
                config.lora,
                config.model,
                llama_arch,
            )
        )
        assert enable_overlap_scheduler is True

    # Don't auto-enable for unknown architectures that keep default support.
    other_arch = _serve_optimization_arch("SomeOtherArchitecture")
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
    )
    _device_graph_capture, enable_overlap_scheduler = (
        _resolve_overlap_and_device_graph_capture(
            config.runtime,
            config.sampling,
            config.lora,
            config.model,
            other_arch,
        )
    )
    assert enable_overlap_scheduler is True

    # Explicitly opt-out architectures stay disabled.
    disabled_arch = SimpleNamespace(
        name="SomeDisabledArchitecture",
        task=PipelineTask.TEXT_GENERATION,
        supports_overlap_scheduler=False,
        supports_device_graph_capture=False,
    )
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
    )
    _device_graph_capture, enable_overlap_scheduler = (
        _resolve_overlap_and_device_graph_capture(
            config.runtime,
            config.sampling,
            config.lora,
            config.model,
            disabled_arch,
        )
    )
    assert enable_overlap_scheduler is False


@mock_pipeline_config_resolve
def test_validate_and_resolve_overlap_scheduler__validate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mock architecture_name so we don't reach out to HF for the config.
    monkeypatch.setattr(
        MAXModelConfig,
        "architecture_name",
        property(lambda self: "SomeArchitecture"),
    )

    # Allow user to manually enable overlap scheduler
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(enable_overlap_scheduler=True),
    )
    _device_graph_capture, enable_overlap_scheduler = (
        _resolve_overlap_and_device_graph_capture(
            config.runtime, config.sampling, config.lora, config.model, None
        )
    )
    assert enable_overlap_scheduler is True

    # Error out if user tries to enable overlap scheduler on CPU
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.cpu()],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(enable_overlap_scheduler=True),
    )
    with pytest.raises(ValueError):
        _resolve_overlap_and_device_graph_capture(
            config.runtime, config.sampling, config.lora, config.model, None
        )

    # prefill_only with overlap scheduler is now allowed (experimental).
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            pipeline_role="prefill_only",
            enable_overlap_scheduler=True,
        ),
    )
    _device_graph_capture, enable_overlap_scheduler = (
        _resolve_overlap_and_device_graph_capture(
            config.runtime, config.sampling, config.lora, config.model, None
        )
    )
    assert enable_overlap_scheduler is True

    # Error out if user tries to enable overlap scheduler with variable logits
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        sampling=SamplingConfig(enable_variable_logits=True),
        runtime=PipelineRuntimeConfig(enable_overlap_scheduler=True),
    )
    with pytest.raises(ValueError):
        _resolve_overlap_and_device_graph_capture(
            config.runtime, config.sampling, config.lora, config.model, None
        )


@prepare_registry
@mock_pipeline_config_resolve
@pytest.mark.parametrize(
    "num_speculative_tokens",
    [1, 2, 5],
    ids=["1_spec_token", "2_spec_tokens", "5_spec_tokens"],
)
def test_auto_device_graph_capture_eagle_gating(
    num_speculative_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eagle arch auto-enables graph capture for any num_speculative_tokens.

    Device graphs support num_speculative_tokens > 1 since #83956, so the
    old <= 1 gate no longer exists.
    """
    monkeypatch.setattr(MAXModelConfig, "huggingface_model_repo", Mock())
    arch = SimpleNamespace(
        name="UnifiedEagleLlama3ForCausalLM",
        task=PipelineTask.TEXT_GENERATION,
        supports_overlap_scheduler=True,
        supports_device_graph_capture=True,
    )
    monkeypatch.setattr(
        "max.pipelines.lib.config.config.accelerator_api",
        Mock(return_value="cuda"),
    )

    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        speculative=SpeculativeConfig(
            speculative_method="eagle",
            num_speculative_tokens=num_speculative_tokens,
        ),
        runtime=PipelineRuntimeConfig(max_batch_size=16),
    )
    device_graph_capture, _enable_overlap_scheduler = (
        _resolve_overlap_and_device_graph_capture(
            config.runtime, config.sampling, config.lora, config.model, arch
        )
    )

    assert device_graph_capture is True


@prepare_registry
@mock_pipeline_config_resolve
def test_resolve_default_reasoning_parser__applies_arch_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the user did not set runtime.reasoning_parser and the resolved
    architecture declares a default, the default is applied."""
    arch = SimpleNamespace(
        name="KimiK25ForConditionalGeneration",
        reasoning_parser="kimik2_5",
    )

    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        runtime=PipelineRuntimeConfig(),
    )
    assert config.runtime.reasoning_parser is None

    assert _resolve_default_reasoning_parser(config.runtime, arch) == "kimik2_5"


@prepare_registry
@mock_pipeline_config_resolve
def test_resolve_default_reasoning_parser__user_value_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit runtime.reasoning_parser value is never overwritten,
    even when the architecture declares a different default."""
    arch = SimpleNamespace(
        name="KimiK25ForConditionalGeneration",
        reasoning_parser="kimik2_5",
    )

    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        runtime=PipelineRuntimeConfig(reasoning_parser="user_choice"),
    )

    assert (
        _resolve_default_reasoning_parser(config.runtime, arch) == "user_choice"
    )


@prepare_registry
@mock_pipeline_config_resolve
def test_resolve_default_reasoning_parser__no_arch_default_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the architecture does not declare a default reasoning parser
    (or no architecture is found), runtime.reasoning_parser stays None."""
    arch_without_default = SimpleNamespace(
        name="LlamaForCausalLM",
        reasoning_parser=None,
    )

    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        runtime=PipelineRuntimeConfig(),
    )

    assert (
        _resolve_default_reasoning_parser(config.runtime, arch_without_default)
        is None
    )
    assert _resolve_default_reasoning_parser(config.runtime, None) is None


@prepare_registry
@mock_pipeline_config_resolve
def test_resolve_default_tool_parser__applies_arch_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When runtime.tool_parser is unset and architecture declares a default,
    the default is applied."""
    arch = SimpleNamespace(
        name="KimiK25ForConditionalGeneration",
        tool_parser="kimik2_5",
    )

    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        runtime=PipelineRuntimeConfig(),
    )
    assert config.runtime.tool_parser is None

    assert (
        _resolve_default_tool_parser(config.runtime, config.model, arch)
        == "kimik2_5"
    )


@prepare_registry
@mock_pipeline_config_resolve
def test_resolve_default_tool_parser__user_value_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit runtime.tool_parser value is never overwritten."""
    arch = SimpleNamespace(
        name="KimiK25ForConditionalGeneration",
        tool_parser="kimik2_5",
    )

    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        runtime=PipelineRuntimeConfig(tool_parser="user_choice"),
    )

    assert (
        _resolve_default_tool_parser(config.runtime, config.model, arch)
        == "user_choice"
    )


@prepare_registry
@mock_pipeline_config_resolve
def test_resolve_default_tool_parser__no_arch_default_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If architecture has no default tool parser, runtime value stays None."""
    arch_without_default = SimpleNamespace(
        name="LlamaForCausalLM",
        tool_parser=None,
    )

    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        runtime=PipelineRuntimeConfig(),
    )

    assert (
        _resolve_default_tool_parser(
            config.runtime, config.model, arch_without_default
        )
        is None
    )
    assert (
        _resolve_default_tool_parser(config.runtime, config.model, None) is None
    )


@pytest.mark.parametrize("sentinel", ["none", "None", "NONE", "nOnE"])
@prepare_registry
@mock_pipeline_config_resolve
def test_resolve_default_reasoning_parser__none_sentinel_disables(
    monkeypatch: pytest.MonkeyPatch,
    sentinel: str,
) -> None:
    """Passing the case-insensitive ``"none"`` sentinel explicitly disables
    the reasoning parser, overriding the architecture default and normalizing
    the runtime value to ``None``."""
    arch = SimpleNamespace(
        name="KimiK25ForConditionalGeneration",
        reasoning_parser="kimik2_5",
    )

    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        runtime=PipelineRuntimeConfig(reasoning_parser=sentinel),
    )

    assert _resolve_default_reasoning_parser(config.runtime, arch) is None


@pytest.mark.parametrize("sentinel", ["none", "None", "NONE", "nOnE"])
@prepare_registry
@mock_pipeline_config_resolve
def test_resolve_default_tool_parser__none_sentinel_disables(
    monkeypatch: pytest.MonkeyPatch,
    sentinel: str,
) -> None:
    """Passing the case-insensitive ``"none"`` sentinel explicitly disables
    the tool parser, overriding the architecture default (including callable
    defaults) and normalizing the runtime value to ``None``."""
    arch = SimpleNamespace(
        name="KimiK25ForConditionalGeneration",
        tool_parser="kimik2_5",
    )

    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        runtime=PipelineRuntimeConfig(tool_parser=sentinel),
    )

    assert (
        _resolve_default_tool_parser(config.runtime, config.model, arch) is None
    )


# ===----------------------------------------------------------------------=== #
# Tests for _resolve_default_structured_output_backend
# ===----------------------------------------------------------------------=== #


def _backend_arch(default: str | None) -> SimpleNamespace:
    """Minimal architecture stub for structured-output-backend resolution."""
    return SimpleNamespace(
        name="DummyForCausalLM", default_structured_output_backend=default
    )


@mock_hf_repo_access
def test_resolve_backend__unset_normal_arch_defaults_to_xgrammar() -> None:
    """Unset + an arch with no backend preference resolves to the global
    default ``xgrammar``."""
    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
    )
    assert config.sampling.structured_output_backend is None  # unset sentinel

    assert (
        _resolve_default_structured_output_backend(
            config.sampling, _backend_arch(default=None)
        )
        == "xgrammar"
    )


@mock_hf_repo_access
def test_resolve_backend__unset_pinned_arch_uses_arch_default() -> None:
    """Unset + an arch that pins ``llguidance`` (e.g. Gemma 3 / MiniMax-M2)
    resolves to the arch default."""
    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
    )

    assert (
        _resolve_default_structured_output_backend(
            config.sampling, _backend_arch(default="llguidance")
        )
        == "llguidance"
    )


@mock_hf_repo_access
def test_resolve_backend__explicit_xgrammar_overrides_pinned_arch() -> None:
    """Regression: an explicit ``xgrammar`` on a ``llguidance``-pinned arch is
    honored, not silently overwritten. This is the precedence bug this fix
    closes (explicit ``xgrammar`` equalled the old hardcoded default)."""
    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        sampling=SamplingConfig(structured_output_backend="xgrammar"),
    )

    assert (
        _resolve_default_structured_output_backend(
            config.sampling, _backend_arch(default="llguidance")
        )
        == "xgrammar"
    )


@mock_hf_repo_access
def test_resolve_backend__explicit_llguidance_on_normal_arch_is_honored() -> (
    None
):
    """An explicit ``llguidance`` on a model with no arch preference is
    honored over the global ``xgrammar`` default."""
    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        sampling=SamplingConfig(structured_output_backend="llguidance"),
    )

    assert (
        _resolve_default_structured_output_backend(
            config.sampling, _backend_arch(default=None)
        )
        == "llguidance"
    )


@mock_hf_repo_access
def test_resolve_backend__unset_no_arch_defaults_to_xgrammar() -> None:
    """Unset + ``arch=None`` exercises the unconditional global fallback."""
    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
    )

    assert (
        _resolve_default_structured_output_backend(config.sampling, None)
        == "xgrammar"
    )


@mock_hf_repo_access
def test_from_args__unset_backend_preserves_none_sentinel() -> None:
    """Regression: ``PipelineArgs`` with no ``--structured-output-backend``
    must carry the ``None`` sentinel into the built ``PipelineConfig``.

    Before the fix, ``PipelineArgs.structured_output_backend`` defaulted to a
    hardcoded ``"llguidance"`` string, so ``from_args`` produced a
    ``SamplingConfig`` that already looked like an explicit user choice. That
    short-circuited ``_resolve_default_structured_output_backend`` and the
    global ``xgrammar`` default (and any arch pin) was never reached."""
    args = PipelineArgs(model_path="test/model")
    assert args.sampling.structured_output_backend is None

    with patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"):
        config = PipelineConfig.from_args(args)

    assert config.sampling.structured_output_backend is None


@mock_hf_repo_access
def test_from_args__unset_backend_resolves_to_xgrammar() -> None:
    """End-to-end guard for the reported bug: a model launched without an
    explicit backend and no arch pin ends up on ``xgrammar``, not
    ``llguidance``."""
    args = PipelineArgs(model_path="test/model")

    with patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"):
        config = PipelineConfig.from_args(args)
    assert (
        _resolve_default_structured_output_backend(
            config.sampling, _backend_arch(default=None)
        )
        == "xgrammar"
    )


@mock_hf_repo_access
def test_from_args__explicit_backend_is_preserved() -> None:
    """An explicit ``--structured-output-backend`` value survives
    ``from_args`` and wins over resolution."""
    args = PipelineArgs(
        model_path="test/model",
        sampling=SamplingConfig(structured_output_backend="llguidance"),
    )

    with patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"):
        config = PipelineConfig.from_args(args)
    assert config.sampling.structured_output_backend == "llguidance"

    assert (
        _resolve_default_structured_output_backend(
            config.sampling, _backend_arch(default=None)
        )
        == "llguidance"
    )
