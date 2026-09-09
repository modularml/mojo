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
"""Tests for MAXConfig interface."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from max.config import ConfigFileModel, MAXBaseModel
from max.dtype import DType
from max.nn.kv_cache import KVConnectorType
from max.pipelines.lib import (
    KVCacheConfig,
    LoRAConfig,
    ProfilingConfig,
    SamplingConfig,
)
from pydantic import Field, ValidationError


class TestConfig(ConfigFileModel):
    """Test ConfigFileModel class for unit testing."""

    test_field: str = Field(default="default_value")
    test_int: int = Field(default=42)
    test_bool: bool = Field(default=True)
    test_inf: float = Field(default=float("inf"))


class _BaseModelForEqTest(MAXBaseModel):
    a: int = 1
    b: str = "x"


class _ConfigFileModelForEqTest(ConfigFileModel):
    x: int = 1
    # Ensure `section_name` does not participate in equality.
    section_name: str | None = Field(default=None, exclude=True)


class TestMAXConfigInterface:
    """Test suite for MAXConfig base interface."""

    def test_config_instantiation_with_defaults(self) -> None:
        """Test that MAXConfig classes can be instantiated with defaults."""
        config = TestConfig()
        assert config.test_field == "default_value"
        assert config.test_int == 42
        assert config.test_bool is True


class TestMAXConfigFileLoading:
    """Test suite for YAML MAXConfig file loading functionality."""

    def test_load_individual_config_file(self) -> None:
        """Test loading from individual MAXConfig file."""
        config_data = {
            "test_field": "loaded_value",
            "test_int": 100,
            "test_bool": False,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config_data, f)
            f.flush()  # Ensure data is written to disk
            config = TestConfig(config_file=f.name)
            assert config.test_field == "loaded_value"
            assert config.test_int == 100
            assert config.test_bool is False

    def test_load_full_config_file(self) -> None:
        """Test loading from full MAXConfig file with multiple sections."""
        config_data = {
            "name": "test_full_config",
            "description": "A test full config",
            "version": "1.0",
            "test_config": {
                "test_field": "full_value",
                "test_int": 200,
                "test_inf": "inf",
            },
            "other_section": {
                "other_field": "other_value",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config_data, f)
            f.flush()  # Ensure data is written to disk
            config = TestConfig(config_file=f.name, section_name="test_config")
            assert config.test_field == "full_value"
            assert config.test_int == 200
            assert config.test_bool is True  # Default value.
            assert config.test_inf == float("inf")

    def test_load_config_with_enums(self) -> None:
        """Test loading MAXConfig file with enum values."""
        config_data = {
            "name": "test_enum_config",
            "description": "A test config with enums",
            "version": "1.0",
            "kv_cache_config": {
                "kv_cache_page_size": 256,
                "kv_connector_config": {"type": "tiered"},
            },
            "profiling_config": {
                "gpu_profiling": "on",
            },
            "sampling_config": {
                "in_dtype": "float16",
                "out_dtype": "bfloat16",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config_data, f)
            f.flush()  # Ensure data is written to disk
            # Test KVCacheConfig enum loading.
            kv_config = KVCacheConfig(
                config_file=f.name, section_name="kv_cache_config"
            )
            assert kv_config.kv_cache_page_size == 256
            assert kv_config.kv_connector_config.type == KVConnectorType.tiered

            # Test ProfilingConfig enum loading.
            profiling_config = ProfilingConfig(
                config_file=f.name, section_name="profiling_config"
            )
            assert profiling_config.gpu_profiling == "on"

            # Test SamplingConfig enum loading.
            sampling_config = SamplingConfig(
                config_file=f.name, section_name="sampling_config"
            )
            assert isinstance(sampling_config.in_dtype, DType)
            assert sampling_config.in_dtype == DType.float16
            assert sampling_config.out_dtype == DType.bfloat16


class TestPydanticBaseModelEquality:
    def test_max_base_model_structural_equality(self) -> None:
        assert _BaseModelForEqTest(a=1, b="x") == _BaseModelForEqTest(
            a=1, b="x"
        )
        assert _BaseModelForEqTest(a=1, b="x") != _BaseModelForEqTest(
            a=2, b="x"
        )
        assert _BaseModelForEqTest(a=1, b="x") != _BaseModelForEqTest(
            a=1, b="y"
        )

    def test_max_base_model_notimplemented_for_other_types(self) -> None:
        obj = _BaseModelForEqTest(a=1, b="x")
        assert obj.__eq__(123) is NotImplemented
        assert obj.__ne__(123) is NotImplemented

    def test_config_file_model_section_name_excluded_from_equality(
        self,
    ) -> None:
        assert _ConfigFileModelForEqTest(
            x=1, section_name="a"
        ) == _ConfigFileModelForEqTest(x=1, section_name="b")

    def test_config_file_model_config_file_participates_in_equality(
        self,
    ) -> None:
        # Construct a trivial config file so the validator can open it.
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.yaml"
            cfg_path.write_text("{}", encoding="utf-8")

            a = _ConfigFileModelForEqTest(x=1)
            b = _ConfigFileModelForEqTest(x=1, config_file=str(cfg_path))
            assert a != b


class TestBuiltinConfigClasses:
    """Test suite for built-in MAXConfig classes."""

    def test_kv_cache_config(self) -> None:
        """Test KVCacheConfig functionality."""
        config = KVCacheConfig()

        # Test default values.
        assert hasattr(config, "kv_cache_page_size")

        # Test section name.
        assert config._config_file_section_name == "kv_cache_config"

    def test_sampling_config(self) -> None:
        """Test SamplingConfig functionality."""
        config = SamplingConfig()

        # Test default values.
        assert hasattr(config, "in_dtype")
        assert hasattr(config, "out_dtype")

        # Test section name.
        assert config._config_file_section_name == "sampling_config"

    def test_profiling_config(self) -> None:
        """Test ProfilingConfig functionality."""
        config = ProfilingConfig()

        # Test default values.
        assert hasattr(config, "gpu_profiling")

        # Test section name.
        assert config._config_file_section_name == "profiling_config"

    def test_lora_config(self) -> None:
        """Test LoRAConfig functionality."""
        config = LoRAConfig()

        # Test default values.
        assert hasattr(config, "enable_lora")
        assert hasattr(config, "lora_paths")
        assert hasattr(config, "max_lora_rank")

        # Test section name.
        assert config._config_file_section_name == "lora_config"


class TestProfilingConfigEnv:
    def test_env_enables_profiling_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODULAR_ENABLE_PROFILING", "on")
        config = ProfilingConfig()
        assert config.gpu_profiling == "on"

    def test_env_invalid_value_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODULAR_ENABLE_PROFILING", "bad-value")
        with pytest.raises(ValueError, match="gpu_profiling must be one of"):
            ProfilingConfig()

    def test_explicit_value_ignores_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODULAR_ENABLE_PROFILING", "bad-value")
        config = ProfilingConfig(gpu_profiling="on")
        assert config.gpu_profiling == "on"


class TestFrozenLeafConfigs:
    """LoRAConfig and ProfilingConfig are immutable after construction."""

    def test_lora_config_rejects_field_assignment(self) -> None:
        config = LoRAConfig()
        with pytest.raises(ValidationError):
            config.enable_lora = True  # type: ignore[misc]

    def test_profiling_config_rejects_field_assignment(self) -> None:
        config = ProfilingConfig()
        with pytest.raises(ValidationError):
            config.gpu_profiling = "on"  # type: ignore[misc]

    def test_lora_config_model_copy_update(self) -> None:
        config = LoRAConfig()
        updated = config.model_copy(update={"max_lora_rank": 32})
        assert updated.max_lora_rank == 32
        assert config.max_lora_rank == 16

    def test_profiling_config_model_copy_update(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MODULAR_ENABLE_PROFILING", raising=False)
        config = ProfilingConfig()
        updated = config.model_copy(update={"gpu_profiling": "detailed"})
        assert updated.gpu_profiling == "detailed"
        assert config.gpu_profiling == "off"


class _StrictModel(MAXBaseModel):
    count: int = 0
    name: str = "default"


class _LaxConfigModel(ConfigFileModel):
    count: int = 0
    ratio: float = 1.0
    enabled: bool = False
    label: str = "default"


class TestModelConfigStrictness:
    """Verify extra='forbid' and strict/lax behavior on MAXBaseModel vs ConfigFileModel."""

    def test_base_model_rejects_extra_fields(self) -> None:
        with pytest.raises(
            ValidationError, match="Extra inputs are not permitted"
        ):
            _StrictModel(count=1, name="ok", unknown_field="boom")  # type: ignore[call-arg]

    def test_base_model_rejects_wrong_types(self) -> None:
        with pytest.raises(ValidationError):
            _StrictModel(count="not-an-int")  # type: ignore[arg-type]

    def test_base_model_accepts_correct_types(self) -> None:
        m = _StrictModel(count=5, name="hello")
        assert m.count == 5
        assert m.name == "hello"

    def test_config_file_model_rejects_extra_fields(self) -> None:
        with pytest.raises(
            ValidationError, match="Extra inputs are not permitted"
        ):
            _LaxConfigModel(count=1, bogus="nope")  # type: ignore[call-arg]

    def test_config_file_model_coerces_str_to_int(self) -> None:
        m = _LaxConfigModel(count="42")  # type: ignore[arg-type]
        assert m.count == 42

    def test_config_file_model_coerces_str_to_float(self) -> None:
        m = _LaxConfigModel(ratio="3.14")  # type: ignore[arg-type]
        assert m.ratio == pytest.approx(3.14)

    def test_config_file_model_coerces_int_to_float(self) -> None:
        m = _LaxConfigModel(ratio=2)
        assert m.ratio == 2.0
        assert isinstance(m.ratio, float)

    def test_config_file_model_coerces_str_to_bool(self) -> None:
        m = _LaxConfigModel(enabled="true")  # type: ignore[arg-type]
        assert m.enabled is True

    def test_config_file_model_accepts_correct_types(self) -> None:
        m = _LaxConfigModel(count=7, ratio=0.5, enabled=True, label="test")
        assert m.count == 7
        assert m.ratio == 0.5
        assert m.enabled is True
        assert m.label == "test"
