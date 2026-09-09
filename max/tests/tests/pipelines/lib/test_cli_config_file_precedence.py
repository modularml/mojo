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
"""Tests for Click config file precedence and recipe path resolution."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner
from max._entrypoints.cli.config import config_to_flag, pipeline_config_options
from max.config import ConfigFileModel
from max.config.config_file_model import _resolve_config_file
from max.driver import DeviceSpec
from max.nn.kv_cache.cache_params import KVConnectorType
from max.pipelines.lib import PipelineArgs, PipelineConfig
from pydantic import Field
from pytest import MonkeyPatch


@pytest.fixture(autouse=True)
def _skip_repo_access_check() -> Iterator[None]:
    """These precedence tests use placeholder repos, so skip the HF network
    calls that ``MAXModelConfig`` construction now runs: the existence check
    (via both the ``model_config`` and ``hf_utils`` references) and the eager
    ``load_huggingface_config``."""
    with (
        patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"),
        patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
        patch(
            "max.pipelines.weights.hf_utils.generate_local_model_path",
            side_effect=lambda repo_id, revision=None: f"/fake/cache/{repo_id}",
        ),
        patch("huggingface_hub.file_exists", return_value=False),
        # architectures=None keeps the architecture name undeterminable, so
        # construction-time resolution skips instead of rejecting the
        # placeholder repo as an unknown architecture.
        patch(
            "max.pipelines.lib.config.model_config.load_huggingface_config",
            return_value=MagicMock(architectures=None),
        ),
    ):
        yield


class _TestConfig(ConfigFileModel):
    model_path: str = Field(default="")
    device_graph_capture: bool = Field(default=False)


def _make_cli() -> click.Command:
    @click.command()
    @config_to_flag(_TestConfig)
    def cli(**config_kwargs: Any) -> None:
        config = _TestConfig(**config_kwargs)
        click.echo(f"{config.model_path}|{config.device_graph_capture}")

    return cli


def _make_pipeline_cli() -> click.Command:
    @click.command()
    @pipeline_config_options
    def cli(**config_kwargs: Any) -> None:
        click.echo(
            "|".join(
                key
                for key in ("device_specs", "draft_device_specs")
                if key in config_kwargs
            )
        )

    return cli


def test_config_file_overrides_click_defaults(tmp_path: Path) -> None:
    """Config file values win over Click defaults (case 2)."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model_path: test-model\ndevice_graph_capture: true\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        _make_cli(),
        ["--config-file", str(config_path)],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "test-model|True"


def test_absent_fields_keep_pydantic_defaults(tmp_path: Path) -> None:
    """Fields absent from both CLI and config file get Pydantic defaults (case 1)."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_path: from-file\n", encoding="utf-8")
    result = CliRunner().invoke(
        _make_cli(),
        ["--config-file", str(config_path)],
    )
    assert result.exit_code == 0, result.output
    # device_graph_capture not in config file -> Pydantic default (False).
    assert result.output.strip() == "from-file|False"


def test_cli_args_override_config_file(tmp_path: Path) -> None:
    """Explicit CLI args override config file values (case 3)."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model_path: from-file\ndevice_graph_capture: true\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        _make_cli(),
        ["--config-file", str(config_path), "--model-path", "from-cli"],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "from-cli|True"


def test_implicit_devices_do_not_override_config(
    monkeypatch: MonkeyPatch,
) -> None:
    """Absent --devices leaves device_specs to config or model defaults."""
    monkeypatch.setattr(
        "max._entrypoints.cli.config.DevicesOptionType.device_specs",
        staticmethod(lambda devices: [devices]),
    )

    result = CliRunner().invoke(_make_pipeline_cli(), ["--config-file", "x"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == ""


def test_implicit_devices_use_default_without_config(
    monkeypatch: MonkeyPatch,
) -> None:
    """Absent --devices lets MAXModelConfig use its Pydantic default."""
    monkeypatch.setattr(
        "max._entrypoints.cli.config.DevicesOptionType.device_specs",
        staticmethod(lambda devices: [devices]),
    )

    result = CliRunner().invoke(_make_pipeline_cli(), [])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == ""


def test_explicit_devices_still_override_config_file(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "max._entrypoints.cli.config.DevicesOptionType.device_specs",
        staticmethod(lambda devices: [devices]),
    )

    result = CliRunner().invoke(
        _make_pipeline_cli(),
        ["--config-file", "x", "--devices", "gpu:0"],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "device_specs|draft_device_specs"


def test_explicit_devices_inherited_by_draft_devices(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "max._entrypoints.cli.config.DevicesOptionType.device_specs",
        staticmethod(lambda devices: [devices]),
    )

    result = CliRunner().invoke(_make_pipeline_cli(), ["--devices", "gpu:0"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "device_specs|draft_device_specs"


def test_unset_num_speculative_tokens_reaches_config_as_none() -> None:
    """A ``--num-speculative-tokens`` the user never passed reaches the
    constructed config as ``None``, so each speculative method resolves its
    own default (eagle/mtp use 2; DSpark/DFlash drafters derive the trained
    width from the draft checkpoint), while an explicit value is honored."""

    @click.command()
    @pipeline_config_options
    def cli(**kwargs: Any) -> None:
        config = PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(**kwargs)
        )
        assert config.speculative is not None
        click.echo(f"{config.speculative.num_speculative_tokens}")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--model-path", "fake/model", "--speculative-method", "dflash"],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "None"

    result = runner.invoke(
        cli,
        [
            "--model-path",
            "fake/model",
            "--speculative-method",
            "dflash",
            "--num-speculative-tokens",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "5"

    result = runner.invoke(
        cli,
        ["--model-path", "fake/model", "--speculative-method", "eagle"],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "2"


def test_resolve_config_file_passthrough_for_plain_paths() -> None:
    """Non-prefixed paths are returned unchanged."""
    assert _resolve_config_file("/tmp/my_config.yaml") == "/tmp/my_config.yaml"
    assert _resolve_config_file("relative/path.yaml") == "relative/path.yaml"


def test_resolve_config_file_resolves_builtin_recipe() -> None:
    """Paths with the max/pipelines/architectures/ prefix resolve to the package."""
    resolved = _resolve_config_file(
        "max/pipelines/architectures/llama3_modulev3/recipes/llama32_1b.yaml"
    )
    assert Path(resolved).is_file()
    assert resolved.endswith(
        "pipelines/architectures/llama3_modulev3/recipes/llama32_1b.yaml"
    )


def test_resolve_config_file_raises_for_missing_recipe() -> None:
    """Non-existent recipe under the prefix raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Built-in recipe not found"):
        _resolve_config_file(
            "max/pipelines/architectures/no_such/recipes/missing.yaml"
        )


def test_cli_overrides_yaml_recipe_values(tmp_path: Path) -> None:
    """Regression: CLI flags must override values set in a YAML recipe,
    covering both the ``model:`` section (``--devices``,
    ``--data-parallel-degree``) and the ``runtime:`` section
    (``--ep-size``)."""
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "model:\n"
        "  model_path: fake/model\n"
        "  device_specs: [0, 1, 2, 3, 4, 5, 6, 7]\n"
        "  data_parallel_degree: 8\n"
        "runtime:\n"
        "  ep_size: 8\n",
        encoding="utf-8",
    )

    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(
            config_file=str(config_path),
            device_specs=[DeviceSpec(i, "gpu") for i in range(4)],
            data_parallel_degree=4,
            ep_size=4,
        )
    )

    main = config.models["main"]
    assert len(main.device_specs) == 4
    assert main.data_parallel_degree == 4
    assert config.runtime.ep_size == 4


def test_model_path_override_preserves_yaml_recipe_fields(
    tmp_path: Path,
) -> None:
    """Regression: --model-path must merge into (not replace) a YAML-loaded
    model manifest, preserving other recipe-set fields like device_specs."""
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "model:\n"
        "  model_path: fake/original-model\n"
        "  device_specs: [0, 1, 2, 3, 4, 5, 6, 7]\n"
        "  data_parallel_degree: 2\n"
        "runtime:\n"
        "  ep_size: 8\n",
        encoding="utf-8",
    )

    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(
            config_file=str(config_path),
            model_path="fake/override-model",
        )
    )

    main = config.models["main"]
    assert main.model_path == "fake/override-model"
    # The rest of the recipe must survive the model_path override.
    assert len(main.device_specs) == 8
    assert main.data_parallel_degree == 2
    assert config.runtime.ep_size == 8


def test_kv_cache_flag_preserves_yaml_recipe_kv_cache_fields(
    tmp_path: Path,
) -> None:
    """A KV-cache CLI flag must merge into (not replace) a YAML-loaded
    ``kv_cache``, preserving other recipe-set fields such as
    ``device_memory_utilization``."""
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "model:\n"
        "  model_path: fake/model\n"
        "  kv_cache:\n"
        "    device_memory_utilization: 0.8\n"
        "    enable_dp_cross_replica_prefix_copy: false\n"
        "    kv_connector_config:\n"
        "      type: tiered\n",
        encoding="utf-8",
    )

    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(
            config_file=str(config_path),
            kv_connector_config={
                "type": "rust_tiered",
                "host_offload_max_gb": 1200,
            },
        )
    )

    kv = config.models["main"].kv_cache
    # The CLI flag overrides the connector fields it names...
    assert kv.kv_connector_config.type == KVConnectorType.rust_tiered
    assert kv.kv_connector_config.host_offload_max_gb == 1200
    # ...but the recipe's other kv_cache fields survive the override.
    assert kv.device_memory_utilization == 0.8


def test_kv_cache_flag_via_cli_preserves_config_file_fields(
    tmp_path: Path,
) -> None:
    """Same as above, but through Click so the default-stripping step the
    flat-kwargs path skips is exercised for the exact reported invocation."""
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "model:\n"
        "  model_path: fake/model\n"
        "  kv_cache:\n"
        "    device_memory_utilization: 0.8\n"
        "    enable_dp_cross_replica_prefix_copy: false\n"
        "    kv_connector_config:\n"
        "      type: tiered\n",
        encoding="utf-8",
    )

    @click.command()
    @pipeline_config_options
    def cli(**kwargs: Any) -> None:
        kv = (
            PipelineConfig.from_args(PipelineArgs.from_flat_kwargs(**kwargs))
            .models["main"]
            .kv_cache
        )
        click.echo(
            f"{kv.kv_connector_config.type.value}"
            f"|{kv.device_memory_utilization}"
            f"|{kv.enable_dp_cross_replica_prefix_copy}"
        )

    result = CliRunner().invoke(
        cli,
        [
            "--config-file",
            str(config_path),
            "--kv-connector-config",
            '{"type": "rust_tiered", "host_offload_max_gb": 1200}',
        ],
    )
    assert result.exit_code == 0, result.output
    # CLI overrides the connector; the recipe's other kv_cache fields survive.
    assert result.output.strip() == "rust_tiered|0.8|False"


def test_speculative_flag_preserves_yaml_recipe_speculative_fields(
    tmp_path: Path,
) -> None:
    """A speculative CLI flag must merge into (not be dropped by) a YAML-loaded
    ``speculative`` config: --num-speculative-tokens overrides the recipe value
    while the recipe's ``speculative_method`` survives (regression, MXF-594)."""
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "model:\n"
        "  model_path: fake/model\n"
        "speculative:\n"
        "  speculative_method: mtp\n"
        "  num_speculative_tokens: 3\n",
        encoding="utf-8",
    )

    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(
            config_file=str(config_path),
            num_speculative_tokens=2,
        )
    )

    assert config.speculative is not None
    assert config.speculative.num_speculative_tokens == 2
    assert config.speculative.speculative_method == "mtp"


def test_speculative_flag_via_cli_preserves_config_file_fields(
    tmp_path: Path,
) -> None:
    """Same as above, but through Click so the default-stripping step the
    flat-kwargs path skips is exercised for the exact reported invocation."""
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "model:\n"
        "  model_path: fake/model\n"
        "speculative:\n"
        "  speculative_method: mtp\n"
        "  num_speculative_tokens: 3\n",
        encoding="utf-8",
    )

    @click.command()
    @pipeline_config_options
    def cli(**kwargs: Any) -> None:
        spec = PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(**kwargs)
        ).speculative
        assert spec is not None
        click.echo(f"{spec.speculative_method}|{spec.num_speculative_tokens}")

    result = CliRunner().invoke(
        cli,
        [
            "--config-file",
            str(config_path),
            "--num-speculative-tokens",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "mtp|2"


def test_speculative_flag_alone_does_not_enable_speculative() -> None:
    """A speculative tuning flag without a method leaves speculative disabled.

    Only ``--speculative-method`` turns speculative decoding on; the other
    flags merely tune it.
    """
    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(
            model_path="fake/model",
            num_speculative_tokens=2,
        )
    )
    assert config.speculative is None


def test_recipe_speculative_intact_without_speculative_flags(
    tmp_path: Path,
) -> None:
    """A recipe-only speculative config is preserved verbatim when no
    speculative CLI flag is passed."""
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "model:\n"
        "  model_path: fake/model\n"
        "speculative:\n"
        "  speculative_method: mtp\n"
        "  num_speculative_tokens: 3\n",
        encoding="utf-8",
    )

    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(config_file=str(config_path))
    )

    assert config.speculative is not None
    assert config.speculative.speculative_method == "mtp"
    assert config.speculative.num_speculative_tokens == 3


def test_model_path_override_warns_about_mismatched_recipe(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """--model-path differing from the recipe's model_path logs a warning."""
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "model:\n  model_path: fake/original-model\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(
                config_file=str(config_path),
                model_path="fake/override-model",
            )
        )

    assert any(
        "fake/original-model" in rec.message
        and "fake/override-model" in rec.message
        for rec in caplog.records
    )


def test_model_path_matching_recipe_does_not_warn(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """--model-path matching the recipe's own model_path is a no-op, not a
    mismatch, so it should not warn."""
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "model:\n  model_path: fake/same-model\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(
                config_file=str(config_path),
                model_path="fake/same-model",
            )
        )

    assert caplog.records == []


_KV_CACHE_RECIPE = (
    "model:\n"
    "  model_path: fake/model\n"
    "  kv_cache:\n"
    "    device_memory_utilization: 0.8\n"
    "    kv_connector_config:\n"
    "      type: tiered\n"
    "      host_offload_max_gb: 1500\n"
    "      disk_offload_max_gb: 8192\n"
    "      disk_offload_dir: /kv-offload\n"
)


def _write_kv_recipe(tmp_path: Path) -> Path:
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(_KV_CACHE_RECIPE, encoding="utf-8")
    return config_path


def test_kv_cache_dict_flag_merges_into_recipe_field(tmp_path: Path) -> None:
    """A dict-valued KV-cache CLI flag merges field-wise into the recipe's
    value rather than replacing it.

    This matters most for the connector ``type``: it lives inside
    ``kv_connector_config``, so a wholesale replacement would reset it to
    ``null`` and silently disable KV offloading whenever an operator tuned one
    sizing knob from the command line."""
    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(
            config_file=str(_write_kv_recipe(tmp_path)),
            kv_connector_config={
                "host_offload_max_gb": 50,
                "disk_offload_max_gb": 50,
            },
        )
    )

    kv = config.models["main"].kv_cache
    assert kv.device_memory_utilization == 0.8
    # The CLI wins for the fields it names...
    assert kv.kv_connector_config.host_offload_max_gb == 50
    assert kv.kv_connector_config.disk_offload_max_gb == 50
    # ...and the recipe's unnamed connector fields survive, including the type.
    assert kv.kv_connector_config.type is KVConnectorType.tiered
    assert kv.kv_connector_config.disk_offload_dir == "/kv-offload"


def test_kv_cache_scalar_flag_preserves_recipe_dict_field(
    tmp_path: Path,
) -> None:
    """A scalar KV-cache CLI flag leaves the recipe's dict-valued
    ``kv_connector_config`` (and other unnamed fields) untouched."""
    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(
            config_file=str(_write_kv_recipe(tmp_path)),
            device_memory_utilization=0.5,
        )
    )

    kv = config.models["main"].kv_cache
    assert kv.device_memory_utilization == 0.5
    assert kv.kv_connector_config.type is KVConnectorType.tiered
    assert kv.kv_connector_config is not None
    assert kv.kv_connector_config.host_offload_max_gb == 1500
    assert kv.kv_connector_config.disk_offload_max_gb == 8192
    assert kv.kv_connector_config.disk_offload_dir == "/kv-offload"


def test_kv_cache_flags_without_recipe_apply_to_defaults() -> None:
    """Without a recipe, KV-cache CLI flags land on top of KVCacheConfig
    defaults (the pre-existing behavior the merge must not change)."""
    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(
            model={"model_path": "fake/model"},
            kv_connector_config={"host_offload_max_gb": 50},
        )
    )

    kv = config.models["main"].kv_cache
    assert kv.kv_connector_config.host_offload_max_gb == 50
    assert kv.kv_connector_config.type is KVConnectorType.null
    assert kv.device_memory_utilization == 0.9


def test_recipe_kv_cache_intact_without_kv_flags(tmp_path: Path) -> None:
    """With no KV-cache CLI flags, the recipe's kv_cache section is loaded
    verbatim."""
    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(
            config_file=str(_write_kv_recipe(tmp_path)),
        )
    )

    kv = config.models["main"].kv_cache
    assert kv.kv_connector_config.type is KVConnectorType.tiered
    assert kv.device_memory_utilization == 0.8
    assert kv.kv_connector_config.host_offload_max_gb == 1500
    assert kv.kv_connector_config.disk_offload_max_gb == 8192
    assert kv.kv_connector_config.disk_offload_dir == "/kv-offload"


def test_config_file_with_builtin_recipe_prefix() -> None:
    """ConfigFileModel resolves and opens a built-in recipe via the prefix path."""
    import yaml

    resolved = _resolve_config_file(
        "max/pipelines/architectures/llama3_modulev3/recipes/llama32_1b.yaml"
    )
    with open(resolved) as f:
        data = yaml.safe_load(f)
    assert data["model"]["model_path"] == "meta-llama/Llama-3.2-1B-Instruct"


def test_synthetic_acceptance_rejected_with_constrained_decoding() -> None:
    """Synthetic acceptance ignores token bitmasks, so a config where
    constrained decoding can fire must be rejected at resolution instead of
    silently serving unenforced grammars.

    Invoked unbound on a stand-in exposing only what the validator reads
    (the pattern used for the arch-rewrite tests): ``needs_bitmask_constraints``
    is a property on the real config, a plain attribute here.
    """
    validate = (
        PipelineConfig._validate_synthetic_acceptance_with_constrained_decoding
    )

    def _cfg(
        synthetic_rate: float | None, needs_bitmask: bool
    ) -> SimpleNamespace:
        return SimpleNamespace(
            speculative=SimpleNamespace(
                synthetic_acceptance_rate=synthetic_rate
            ),
            needs_bitmask_constraints=needs_bitmask,
        )

    validate(_cfg(None, True))  # type: ignore[arg-type]
    validate(_cfg(0.8, False))  # type: ignore[arg-type]
    validate(SimpleNamespace(speculative=None))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="synthetic_acceptance_rate"):
        validate(_cfg(0.8, True))  # type: ignore[arg-type]
