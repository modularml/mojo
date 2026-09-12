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
    _load_workload_yaml,
    _resolve_seed,
    main_with_parsed_args,
    parse_args,
)
from max.benchmark.benchmark_shared.config import (
    DEFAULT_BENCHMARK_SEED,
    ServingBenchmarkConfig,
)
from max.benchmark.benchmark_shared.datasets import DistributionParameter
from max.benchmark.benchmark_shared.datasets.all import (
    _resolve_agentic_tool_profiles,
)
from pydantic import ValidationError


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

    def test_file_beside_a_workload_resolves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A path in a workload YAML is relative to that file, not to cwd."""
        _write_yaml(tmp_path / "payload.yaml", self._NESTED)
        workload = tmp_path / "workload.yaml"
        workload.write_text("extra-body: payload.yaml\nnum-prompts: 4\n")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        args = ServingBenchmarkConfig(model="m", workload_config=str(workload))
        _load_workload_yaml(args)
        assert args.extra_body == self._NESTED

    def test_inline_json_in_a_workload_is_not_a_path(
        self, tmp_path: Path
    ) -> None:
        """Path resolution must not touch a value that is the content itself."""
        workload = tmp_path / "workload.yaml"
        workload.write_text(
            f"extra-body: '{json.dumps(self._NESTED)}'\nnum-prompts: 4\n"
        )
        args = ServingBenchmarkConfig(model="m", workload_config=str(workload))
        _load_workload_yaml(args)
        assert args.extra_body == self._NESTED


# ---------------------------------------------------------------------------
# --agentic-tool-profiles validation
# ---------------------------------------------------------------------------

_PROFILES: dict[str, Any] = {"tools": [{"input-len": "10", "output-len": "5"}]}


def _fitted_config(
    *,
    agentic_tool_profiles: dict[str, Any] | None = None,
    agentic_rounds_per_turn: DistributionParameter | None = None,
) -> ServingBenchmarkConfig:
    """A fitted multiturn run, the only shape that reaches the agent loop."""
    return ServingBenchmarkConfig(
        model="m",
        dataset_name="instruct-coder",
        fit_distributions=True,
        num_chat_sessions=4,
        agentic_tool_profiles=agentic_tool_profiles,
        agentic_rounds_per_turn=agentic_rounds_per_turn,
    )


def test_agentic_tool_profiles_accepts_every_spelling(tmp_path: Path) -> None:
    """A mapping, an inline JSON string and a file path resolve alike."""
    profiles = {"tools": [{"input-len": "10", "output-len": "5"}]}
    config = tmp_path / "tools.yaml"
    _write_yaml(config, profiles)
    for value in (profiles, json.dumps(profiles), str(config)):
        # model_validate takes Any, so the str forms reach the validator the
        # way cyclopts hands it the raw token.
        args = ServingBenchmarkConfig.model_validate(
            {"model": "m", "agentic_tool_profiles": value}
        )
        assert args.agentic_tool_profiles == profiles


def test_agentic_tool_profiles_missing_file_is_reported() -> None:
    """A path that does not exist names the field, not a bare parse error."""
    with pytest.raises(
        ValueError, match=r"agentic_tool_profiles .* not a readable file path"
    ):
        ServingBenchmarkConfig.model_validate(
            {"model": "m", "agentic_tool_profiles": "/nonexistent/tools.yaml"}
        )


def test_agentic_absent_by_default() -> None:
    assert (
        _resolve_agentic_tool_profiles(ServingBenchmarkConfig(model="m"))
        is None
    )


@pytest.mark.parametrize(
    ("tool_profiles", "rounds_per_turn"),
    [(_PROFILES, None), (None, 3)],
    ids=["tools-without-rounds", "rounds-without-tools"],
)
def test_agentic_flags_must_be_set_together(
    tool_profiles: dict[str, Any] | None,
    rounds_per_turn: DistributionParameter | None,
) -> None:
    args = _fitted_config(
        agentic_tool_profiles=tool_profiles,
        agentic_rounds_per_turn=rounds_per_turn,
    )
    with pytest.raises(ValueError, match="must be set together"):
        _resolve_agentic_tool_profiles(args)


@pytest.mark.parametrize(
    ("dataset_name", "fit_distributions", "num_chat_sessions"),
    [
        ("instruct-coder", False, 4),
        # Any dataset off the fitted list takes the same branch.
        ("random", True, 4),
        # The agent loop lives on the multiturn path; a single-turn run would
        # otherwise accept the flags and silently send plain requests.
        ("instruct-coder", True, None),
    ],
)
def test_agentic_needs_a_run_that_reaches_the_builder(
    dataset_name: str, fit_distributions: bool, num_chat_sessions: int | None
) -> None:
    """Only a fitted multiturn run on the three datasets assembles it."""
    args = ServingBenchmarkConfig(
        model="m",
        dataset_name=dataset_name,
        fit_distributions=fit_distributions,
        num_chat_sessions=num_chat_sessions,
        agentic_tool_profiles=_PROFILES,
        agentic_rounds_per_turn=3,
    )
    with pytest.raises(ValueError, match="needs --fit-distributions with"):
        _resolve_agentic_tool_profiles(args)


def test_agentic_resolves_when_fully_specified() -> None:
    args = _fitted_config(
        agentic_tool_profiles=_PROFILES,
        agentic_rounds_per_turn="NB(33,0.3)",
    )
    tools = _resolve_agentic_tool_profiles(args)
    assert tools is not None and len(tools) == 1


# ---------------------------------------------------------------------------
# The agent loop in a workload YAML
# ---------------------------------------------------------------------------

_WORKLOAD_BASE = (
    "dataset-name: instruct-coder\n"
    "fit-distributions: true\n"
    "num-chat-sessions: 4\n"
    "agentic-rounds-per-turn: '3'\n"
)

_NESTED = """agentic-tool-profiles:
  tools:
    - weight: 5
      input-len: 'N(30,20)'
      output-len: 'N(25,8)'
"""
_QUOTED = (
    "agentic-tool-profiles:"
    ' \'{"tools":[{"weight":5,"input-len":"N(30,20)",'
    '"output-len":"N(25,8)"}]}\'\n'
)
_BY_FILENAME = "agentic-tool-profiles: agentic_tools.yaml\n"
_BY_FILENAME_NO_EXT = "agentic-tool-profiles: agentic_tools\n"


def _workload(tmp_path: Path, profiles: str) -> Path:
    """A workload file, with a tool file beside it for the by-filename form."""
    body = (
        "tools:\n"
        "  - weight: 5\n"
        "    input-len: N(30,20)\n"
        "    output-len: N(25,8)\n"
    )
    (tmp_path / "agentic_tools.yaml").write_text(body)
    # A tool file need not be named for its format to be found.
    (tmp_path / "agentic_tools").write_text(body)
    workload = tmp_path / "workload.yaml"
    workload.write_text(_WORKLOAD_BASE + profiles)
    return workload


@pytest.mark.parametrize(
    "profiles",
    [_NESTED, _QUOTED, _BY_FILENAME, _BY_FILENAME_NO_EXT],
    ids=[
        "nested-mapping",
        "quoted-string",
        "by-filename",
        "by-filename-no-extension",
    ],
)
def test_workload_yaml_spellings_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profiles: str
) -> None:
    """Three ways to write one tool resolve to that one tool.

    Run from an unrelated directory, so the by-filename form is resolved
    against the workload file rather than the caller's cwd.
    """
    workload = _workload(tmp_path, profiles)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    args = ServingBenchmarkConfig(model="m", workload_config=str(workload))
    _load_workload_yaml(args)
    tools = _resolve_agentic_tool_profiles(args)

    assert tools is not None
    assert [(t.weight, t.input_len, t.output_len) for t in tools] == [
        (5.0, "N(30,20)", "N(25,8)")
    ]


# ===----------------------------------------------------------------------=== #
# General image-mixing flags
# ===----------------------------------------------------------------------=== #


class TestImageFlags:
    """--image-fraction is mutually exclusive with --random-image-*."""

    def test_defaults_allow_construction(self) -> None:
        config = ServingBenchmarkConfig()
        assert config.image_fraction == 0.0
        assert config.image_turn == "first"

    def test_conflicts_with_random_image_count(self) -> None:
        with pytest.raises(ValueError, match="cannot be combined"):
            ServingBenchmarkConfig(image_fraction=0.1, random_image_count=1)

    def test_conflicts_with_random_image_size(self) -> None:
        with pytest.raises(ValueError, match="cannot be combined"):
            ServingBenchmarkConfig(
                image_fraction=0.1, random_image_size="512,512"
            )

    def test_rejects_unknown_image_turn(self) -> None:
        """Rejected by the ImageTurn literal itself, so there is no hand-rolled
        check to keep in sync with the type.

        Goes through ``model_validate`` rather than the constructor: mypy now
        rejects a bad literal statically, which is the point of the change, so
        the runtime check has to come in through the untyped parsing entry
        point the CLI itself uses.
        """
        with pytest.raises(ValidationError, match="image_turn"):
            ServingBenchmarkConfig.model_validate(
                {"image_fraction": 0.1, "image_turn": "middle"}
            )

    def test_accepts_every_image_turn_value(self) -> None:
        for value in ("first", "last", "every"):
            config = ServingBenchmarkConfig(
                image_fraction=0.1, image_turn=value
            )
            assert config.image_turn == value

    def test_rejects_out_of_range_fraction(self) -> None:
        """The description promises 0.0-1.0; -0.1 used to slip past `> 0` and
        silently produce a text-only run."""
        for bad in (-0.1, 1.5, float("nan"), float("inf")):
            with pytest.raises(ValidationError):
                ServingBenchmarkConfig(image_fraction=bad)

    def test_rejects_pixel_generation_task(self) -> None:
        with pytest.raises(ValueError, match="does not apply"):
            ServingBenchmarkConfig(
                image_fraction=0.1, benchmark_task="text-to-image"
            )

    def test_rejects_text_only_endpoint(self) -> None:
        """A text-only driver would drop the images while stats still count them."""
        with pytest.raises(ValueError, match="image-capable endpoint"):
            ServingBenchmarkConfig(
                image_fraction=0.1, endpoint="/v1/completions"
            )

    def test_rejects_responses_endpoint(self) -> None:
        """/v1/responses only routes to OpenResponsesRequestDriver for
        pixel-generation tasks, and that driver takes only
        PixelGenerationRequestFuncInput, so it never carries input images."""
        with pytest.raises(ValueError, match="image-capable endpoint"):
            ServingBenchmarkConfig(image_fraction=0.1, endpoint="/v1/responses")

    def test_allows_chat_completions_endpoint(self) -> None:
        config = ServingBenchmarkConfig(
            image_fraction=0.1, endpoint="/v1/chat/completions"
        )
        assert config.image_fraction == 0.1

    def test_accepts_distribution_strings(self) -> None:
        config = ServingBenchmarkConfig(
            image_fraction=0.2,
            image_count="DU(1,3)",
            image_long_side="U(256,1024)",
            image_aspect_ratio=1.0,
            image_turn="last",
        )
        assert config.image_count == "DU(1,3)"
        assert config.image_long_side == "U(256,1024)"
        assert config.image_turn == "last"
