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
"""Benchmark config utility functions unit tests"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from max.benchmark.benchmark_serving import (
    _resolve_seed,
    main_with_parsed_args,
    parse_args,
)
from max.benchmark.benchmark_shared.config import (
    DEFAULT_BENCHMARK_SEED,
    ServingBenchmarkConfig,
)


class TestServingSweepFields:
    """Test that sweep/workload-related fields exist on ServingBenchmarkConfig."""

    def test_sweep_fields_on_serving_config(self) -> None:
        """Test that sweep-related fields exist on ServingBenchmarkConfig with correct defaults."""
        config = ServingBenchmarkConfig()

        assert config.workload_config is None
        assert config.log_dir is None
        assert config.dry_run is False
        assert config.upload_results is False
        assert config.benchmark_sha is None
        assert config.cluster_information_path is None
        assert config.benchmark_config_name is None
        assert config.metadata == []
        assert config.latency_percentiles == "50,90,95,99"
        assert config.num_iters == 1
        assert config.num_prompts_multiplier is None
        assert config.flush_prefix_cache is True
        assert list(config.max_concurrency) == [None]
        assert list(config.request_rate) == [float("inf")]

    def test_sweep_config_field_metadata(self) -> None:
        """Test that sweep-related fields have proper json_schema_extra metadata."""
        model_fields = ServingBenchmarkConfig.model_fields

        for name, field_info in model_fields.items():
            extra = field_info.json_schema_extra
            if extra is None or not isinstance(extra, dict):
                continue
            if name == "workload_config":
                assert "group" in extra
                assert extra["group"] == "Workload Configuration"
            elif name in [
                "upload_results",
                "benchmark_sha",
                "cluster_information_path",
                "benchmark_config_name",
            ]:
                assert "group" in extra
                assert extra["group"] == "Result Upload Configuration"
            elif name in [
                "num_iters",
                "flush_prefix_cache",
                "num_prompts_multiplier",
            ]:
                assert "group" in extra
                assert extra["group"] == "Sweep Configuration"
            elif name == "max_concurrency":
                assert "group" in extra
                assert extra["group"] == "Request Configuration"
            elif name == "request_rate":
                assert "group" in extra
                assert extra["group"] == "Traffic Control"


# ===----------------------------------------------------------------------=== #
# ConfigFileModel / cyclopts config-file loading tests
# ===----------------------------------------------------------------------=== #


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)


class TestServingConfigFileLoading:
    """Tests for ``--config-file`` loading with ServingBenchmarkConfig.

    These verify that the cyclopts/ConfigFileModel approach works correctly
    after removal of the legacy argparse infrastructure.

    Config YAML format notes
    ------------------------
    ``ServingBenchmarkConfig`` uses ``section_name = "benchmark_config"`` as
    its default, but that default is NOT visible to the ``model_validator``
    because it runs before Pydantic applies defaults.  The section is only
    extracted when ``section_name`` is explicitly supplied at construction time.

    Two valid file formats therefore exist:

    * **Flat** (no section wrapper) — works without passing ``section_name``::

          model: myorg/model
          host: 10.0.0.1

    * **Sectioned** — requires ``section_name="benchmark_config"``::

          benchmark_config:
              model: myorg/model
              host: 10.0.0.1
    """

    # ------------------------------------------------------------------
    # Flat YAML (no section wrapper)
    # ------------------------------------------------------------------

    def test_flat_yaml_loads_model_and_host(self, tmp_path: Path) -> None:
        """Flat YAML values are applied to ServingBenchmarkConfig."""
        cfg_path = tmp_path / "serving.yaml"
        _write_yaml(cfg_path, {"model": "myorg/llama", "host": "10.0.0.1"})

        config = ServingBenchmarkConfig(config_file=str(cfg_path))
        assert config.model == "myorg/llama"
        assert config.host == "10.0.0.1"

    def test_flat_yaml_applies_port_and_backend(self, tmp_path: Path) -> None:
        """Numeric and string fields from flat YAML are applied correctly."""
        cfg_path = tmp_path / "serving.yaml"
        _write_yaml(
            cfg_path, {"model": "myorg/llama", "port": 9000, "backend": "vllm"}
        )

        config = ServingBenchmarkConfig(config_file=str(cfg_path))
        assert config.port == 9000
        assert config.backend == "vllm"

    def test_flat_yaml_with_tempfile(self) -> None:
        """config_file works with a real NamedTemporaryFile."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({"model": "tmp/model", "port": 8080}, f)
            tmp_name = f.name

        config = ServingBenchmarkConfig(config_file=tmp_name)
        assert config.model == "tmp/model"
        assert config.port == 8080

    # ------------------------------------------------------------------
    # Sectioned YAML (benchmark_config section, explicit section_name)
    # ------------------------------------------------------------------

    def test_sectioned_yaml_with_explicit_section_name(
        self, tmp_path: Path
    ) -> None:
        """Sectioned YAML is parsed correctly when section_name is explicit."""
        cfg_path = tmp_path / "serving.yaml"
        _write_yaml(
            cfg_path,
            {
                "benchmark_config": {
                    "model": "myorg/llama",
                    "host": "10.0.0.1",
                    "port": 9000,
                },
                "other_section": {"ignored": True},
            },
        )

        config = ServingBenchmarkConfig(
            config_file=str(cfg_path), section_name="benchmark_config"
        )
        assert config.model == "myorg/llama"
        assert config.host == "10.0.0.1"
        assert config.port == 9000

    def test_missing_section_raises_key_error(self, tmp_path: Path) -> None:
        """A KeyError is raised when the requested section is absent."""
        cfg_path = tmp_path / "config.yaml"
        _write_yaml(cfg_path, {"other_section": {"model": "x"}})

        with pytest.raises(KeyError, match="benchmark_config"):
            ServingBenchmarkConfig(
                config_file=str(cfg_path), section_name="benchmark_config"
            )

    def test_non_dict_yaml_raises_type_error(self, tmp_path: Path) -> None:
        """A TypeError is raised when the YAML root is not a mapping."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("- item1\n- item2\n")

        with pytest.raises(TypeError, match="mapping"):
            ServingBenchmarkConfig(config_file=str(cfg_path))

    # ------------------------------------------------------------------
    # Precedence: explicit kwargs beat config file
    # ------------------------------------------------------------------

    def test_explicit_kwarg_beats_flat_yaml(self, tmp_path: Path) -> None:
        """An explicitly supplied kwarg takes precedence over flat YAML."""
        cfg_path = tmp_path / "serving.yaml"
        _write_yaml(
            cfg_path,
            {"model": "file/model", "host": "10.0.0.1", "port": 9000},
        )

        config = ServingBenchmarkConfig(
            config_file=str(cfg_path), model="cli/model"
        )
        assert config.model == "cli/model"
        assert config.host == "10.0.0.1"  # still from file
        assert config.port == 9000  # still from file

    def test_multiple_overrides_beat_config_file(self, tmp_path: Path) -> None:
        """Multiple explicit kwargs all override the corresponding file values."""
        cfg_path = tmp_path / "serving.yaml"
        _write_yaml(
            cfg_path, {"model": "file/model", "host": "1.2.3.4", "port": 9000}
        )

        config = ServingBenchmarkConfig(
            config_file=str(cfg_path),
            model="cli/model",
            host="localhost",
        )
        assert config.model == "cli/model"
        assert config.host == "localhost"
        assert config.port == 9000  # not overridden → from file

    # ------------------------------------------------------------------
    # model_fields_set semantics
    # ------------------------------------------------------------------

    def test_model_fields_set_includes_both_cli_and_file_fields(
        self, tmp_path: Path
    ) -> None:
        """Both explicit kwargs and config-file fields appear in model_fields_set.

        ``ConfigFileModel.load_config_file`` is a ``model_validator(mode="before")``
        that injects file values into the input dict before Pydantic validation.
        Pydantic therefore treats them as explicitly supplied, so both the CLI
        kwarg (``model``) and the file value (``host``) end up in
        ``model_fields_set``.
        """
        cfg_path = tmp_path / "serving.yaml"
        _write_yaml(cfg_path, {"model": "file/model", "host": "10.0.0.1"})

        config = ServingBenchmarkConfig(
            config_file=str(cfg_path), model="cli/model"
        )
        # Both the explicit kwarg and the file-sourced field are tracked.
        assert "model" in config.model_fields_set
        assert "host" in config.model_fields_set
        # The CLI value still wins over the file value.
        assert config.model == "cli/model"


# ===----------------------------------------------------------------------=== #
# Workload YAML max-concurrency Tests
# ===----------------------------------------------------------------------=== #


@pytest.mark.usefixtures("offline_dryrun_mocks")
class TestWorkloadMaxConcurrency:
    """Tests that max-concurrency from a workload YAML is applied correctly."""

    def test_workload_max_concurrency_applied_when_cli_not_set(
        self, tmp_path: Path
    ) -> None:
        """Workload YAML max-concurrency is used when the caller did not set one."""
        workload = tmp_path / "workload.yaml"
        workload.write_text("max-concurrency: 1\nnum-prompts: 10\n")

        config = ServingBenchmarkConfig(
            model="HuggingFaceTB/SmolLM2-135M",
            workload_config=str(workload),
            dry_run=True,
        )

        results = list(main_with_parsed_args(config))
        assert len(results) == 1
        assert results[0].max_concurrency == 1

    def test_cli_max_concurrency_beats_workload_yaml(
        self, tmp_path: Path
    ) -> None:
        """Explicitly supplied max_concurrency takes precedence over workload YAML."""
        workload = tmp_path / "workload.yaml"
        workload.write_text("max-concurrency: 1\nnum-prompts: 10\n")

        config = ServingBenchmarkConfig(
            model="HuggingFaceTB/SmolLM2-135M",
            workload_config=str(workload),
            max_concurrency=[4],
            dry_run=True,
        )

        results = list(main_with_parsed_args(config))
        assert len(results) == 1
        assert results[0].max_concurrency == 4


# ===----------------------------------------------------------------------=== #
# Seed handling (PERF-2587)
# ===----------------------------------------------------------------------=== #


class TestSeedDefaultAndOverride:
    """The seed is pinned by default; ``none`` opts into a fresh random draw."""

    def test_default_seed_is_pinned(self) -> None:
        """An unset seed defaults to the fixed constant, not ``None``."""
        assert ServingBenchmarkConfig().seed == DEFAULT_BENCHMARK_SEED

    def test_explicit_int_is_kept(self) -> None:
        """An explicit integer seed is preserved verbatim."""
        assert ServingBenchmarkConfig(seed=1234).seed == 1234

    def test_explicit_none_is_kept(self) -> None:
        """A ``null`` seed (Python ``None``) is preserved as ``None``."""
        assert ServingBenchmarkConfig(seed=None).seed is None

    def test_parse_args_default(self) -> None:
        """``parse_args`` with no ``--seed`` yields the pinned default."""
        assert parse_args(["--model", "m"]).seed == DEFAULT_BENCHMARK_SEED

    def test_parse_args_int(self) -> None:
        """``--seed 7`` parses to the integer 7."""
        assert parse_args(["--model", "m", "--seed", "7"]).seed == 7

    @pytest.mark.parametrize("value", ["none", "None", "NONE"])
    def test_cli_none_maps_to_random(self, value: str) -> None:
        """``--seed none`` (any case) parses to ``None`` (draw a random seed)."""
        assert parse_args(["--model", "m", "--seed", value]).seed is None


class TestSeedConfigFileLoading:
    """Seed loaded from a YAML config/workload file."""

    def test_yaml_null_maps_to_random(self, tmp_path: Path) -> None:
        """YAML ``seed: null`` loads as ``None`` (draw a random seed)."""
        cfg_path = tmp_path / "serving.yaml"
        cfg_path.write_text("model: myorg/llama\nseed: null\n")

        assert ServingBenchmarkConfig(config_file=str(cfg_path)).seed is None

    def test_yaml_string_none_maps_to_random(self, tmp_path: Path) -> None:
        """YAML ``seed: none`` (string) loads as ``None`` via the validator."""
        cfg_path = tmp_path / "serving.yaml"
        cfg_path.write_text("model: myorg/llama\nseed: none\n")

        assert ServingBenchmarkConfig(config_file=str(cfg_path)).seed is None

    def test_yaml_int_is_kept(self, tmp_path: Path) -> None:
        """YAML ``seed: 1234`` loads as the integer 1234."""
        cfg_path = tmp_path / "serving.yaml"
        _write_yaml(cfg_path, {"model": "myorg/llama", "seed": 1234})

        assert ServingBenchmarkConfig(config_file=str(cfg_path)).seed == 1234

    def test_yaml_unset_uses_pinned_default(self, tmp_path: Path) -> None:
        """A YAML file that omits ``seed`` uses the pinned default."""
        cfg_path = tmp_path / "serving.yaml"
        _write_yaml(cfg_path, {"model": "myorg/llama"})

        config = ServingBenchmarkConfig(config_file=str(cfg_path))
        assert config.seed == DEFAULT_BENCHMARK_SEED


class TestResolveSeed:
    """``_resolve_seed`` draws a concrete seed only when one was not pinned."""

    def test_random_seed_is_drawn(self) -> None:
        """A ``None`` seed is resolved to a concrete int in ``[0, 10000)``."""
        config = ServingBenchmarkConfig(model="m", seed=None)
        _resolve_seed(config)
        assert isinstance(config.seed, int)
        assert 0 <= config.seed < 10000

    def test_pinned_seed_is_unchanged(self) -> None:
        """A pinned seed is left exactly as supplied."""
        config = ServingBenchmarkConfig(model="m", seed=1234)
        _resolve_seed(config)
        assert config.seed == 1234


# ===----------------------------------------------------------------------=== #
# extra_body validator
# ===----------------------------------------------------------------------=== #


class TestExtraBodyValidator:
    """``ServingBenchmarkConfig.extra_body`` accepts dict / inline JSON / file."""

    # Covers the shapes the merge must preserve verbatim: a top-level array
    # (with a tricky ``}``), a scalar, and a nested object.
    _NESTED = {
        "stop": ["}"],
        "max_tokens": 15,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }

    @staticmethod
    def _build_with_extra(value: object) -> ServingBenchmarkConfig:
        """Build a config routing ``value`` through the ``extra_body`` validator.

        ``extra_body`` is typed ``dict[str, Any]`` (its resolved form), but the
        CLI/file forms feed it a ``str``. ``model_validate`` accepts ``Any``, so
        this exercises the same validator without a type-checker complaint at the
        call site (mirroring how cyclopts hands the raw token to the model).
        """
        return ServingBenchmarkConfig.model_validate(
            {"model": "m", "extra_body": value}
        )

    def test_dict_passthrough_preserves_nested(self) -> None:
        """A mapping is stored verbatim with nested objects/arrays intact."""
        config = ServingBenchmarkConfig(model="m", extra_body=self._NESTED)
        assert config.extra_body == self._NESTED

    def test_inline_json_string(self) -> None:
        """An inline JSON string parses into the equivalent mapping."""
        config = self._build_with_extra(json.dumps(self._NESTED))
        assert config.extra_body == self._NESTED

    def test_file_path(self, tmp_path: Path) -> None:
        """A path to a YAML/JSON file is read and parsed."""
        payload = tmp_path / "payload.yaml"
        _write_yaml(payload, self._NESTED)
        config = self._build_with_extra(str(payload))
        assert config.extra_body == self._NESTED

    def test_missing_file_rejected(self) -> None:
        """A path that does not exist surfaces a clear, dual-cause error."""
        with pytest.raises(ValueError, match="not a readable file path"):
            self._build_with_extra("/nonexistent/payload.yaml")
