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
"""Tests for the flat-kwargs -> PipelineArgs -> PipelineConfig path.

The serve entrypoint parses CLI flags into ``PipelineArgs``
(``from_flat_kwargs``) and later constructs the config for the model worker
(``PipelineConfig.from_args``). Any runtime field dropped along that path is
silently reset to its default in the worker, so CLI flags appear accepted but
never take effect.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import click
import cyclopts
import pytest
import yaml
from cyclopts import Parameter
from max._entrypoints.cli.config import pipeline_config_options
from max.config import ConfigFileModel
from max.pipelines.lib import PipelineArgs, PipelineConfig
from max.pipelines.lib.config.model_config import MAXModelConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_args import (
    _FLAT_KWARG_SUBTREES,
    _SHARED_CONFIG_FIELDS,
)
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.speculative.config import SpeculativeConfig
from pydantic import ValidationError

# Consumed by the CLI, or by ``from_flat_kwargs`` before its unmatched-kwargs
# check, so they need no field to route to.
_CLI_ONLY_FLAGS = frozenset(
    {"devices", "draft_devices", "config_file", "models"}
)


def test_every_cli_flag_routes_to_a_known_destination() -> None:
    """Every flag the CLI generates must have somewhere to land.

    The CLI's list of flattened config models and ``_FLAT_KWARG_SUBTREES``
    are both maintained by hand; drift leaves a flag with no destination.
    """

    @click.command()
    @pipeline_config_options
    def cli(**kwargs) -> None: ...

    flags = {param.name for param in cli.params if param.name}

    destinations = set(PipelineArgs.model_fields)
    for _path, config_class in _FLAT_KWARG_SUBTREES:
        destinations |= set(config_class.model_fields)
    destinations |= {f"draft_{field}" for field in MAXModelConfig.model_fields}

    unroutable = flags - destinations - _CLI_ONLY_FLAGS
    assert not unroutable, (
        f"CLI flags with no routing destination: {sorted(unroutable)}. "
        "Register the owning config model in _FLAT_KWARG_SUBTREES "
        "(max/python/max/pipelines/lib/pipeline_args.py) so these flags "
        "reconcile with a config file's nested section."
    )


def test_from_args_threads_runtime_flags() -> None:
    args = PipelineArgs(
        runtime=PipelineRuntimeConfig(execute_empty_batches=True)
    )
    config = PipelineConfig.from_args(args)
    assert config.runtime.execute_empty_batches is True


def test_runtime_flags_survive_flat_kwargs_path() -> None:
    # Non-default values for the fields this test guards, spelled the way
    # the CLI passes them (flat).
    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(execute_empty_batches=True)
    )
    assert config.runtime.execute_empty_batches is True


def test_sampling_flags_survive_flat_kwargs_path() -> None:
    # gen-mef passes enable_structured_output as a flat kwarg at
    # construction; dropping it on this path would silently export a
    # grammar-incapable sampler MEF.
    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(enable_structured_output=True)
    )
    assert config.sampling.enable_structured_output is True


def test_empty_models_kwarg_is_not_a_manifest_override() -> None:
    # The CLI generates a --models flag from PipelineConfig's models field,
    # so every invocation carries an empty-manifest default. It must not be
    # taken as an explicit manifest override, or every CLI serve/generate
    # produces an empty-manifest config (regression: smoke tests failed at
    # startup with "Cannot determine architecture name: manifest is empty").
    args = PipelineArgs.from_flat_kwargs(models={}, max_batch_size=2)
    assert args._manifest_override is None
    assert PipelineArgs(models=ModelManifest())._manifest_override is None


# One case per generated flag, so new flags are covered the day they land: the
# file value survives, an explicit CLI value wins, a sibling stays untouched.

# Fields the matrix does not drive, and why.
_MATRIX_SKIP = {
    "kv_connector_config": "dict-valued: a merge policy, not precedence",
    "denoising_cache": "a subtree, driven via its own dotted path",
    "enable_lora": "enables its subtree; presence is the signal",
    "first_block_caching": (
        "mutually exclusive with taylorseer, the sibling the matrix would "
        "pair it with"
    ),
    "speculative_method": "enables its subtree; presence is the signal",
}


def _values(annotation: Any) -> list[Any]:
    """Two distinct scalar values, or ``[]`` if none can be derived."""
    if get_origin(annotation) in (Union, UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        annotation = inner[0] if len(inner) == 1 else annotation
    if annotation is bool:
        return [True, False]
    if annotation is int:
        return [3, 7]
    if annotation is float:
        return [0.25, 0.5]
    if annotation is str:
        return ["alpha", "beta"]
    if get_origin(annotation) is Literal:
        return list(get_args(annotation))[:2]
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return list(annotation)[:2]
    return []


# Subtrees PipelineArgs drops unless their enabling field is set.
_ENABLERS: dict[str, dict[str, Any]] = {
    "lora": {"enable_lora": True},
    "speculative": {
        "speculative_method": _values(
            SpeculativeConfig.model_fields["speculative_method"].annotation
        )[0]
    },
}


def _cases() -> list[tuple[str, type[ConfigFileModel], str, Any, Any]]:
    cases: list[tuple[str, type[ConfigFileModel], str, Any, Any]] = []
    for path, config_class in _FLAT_KWARG_SUBTREES:
        for field, info in config_class.model_fields.items():
            if field in _SHARED_CONFIG_FIELDS or field in _MATRIX_SKIP:
                continue
            values = _values(info.annotation)
            if len(values) < 2:
                continue
            if values[0] == info.get_default(call_default_factory=True):
                # Else the config-file assertion just reads the default.
                values.reverse()
            try:  # Keep only values the owning model accepts.
                for value in values:
                    config_class(**{field: value})
            except ValidationError:
                continue
            cases.append((path, config_class, field, *values))
    return cases


_CASES = _cases()


# A neighbour per subtree, to prove an override leaves it alone.
_SIBLINGS: dict[str, list[tuple[str, Any]]] = {}
for _path, _cls, _field, _from_file, _ in _CASES:
    _SIBLINGS.setdefault(_path, []).append((_field, _from_file))


def test_precedence_matrix_covers_every_subtree() -> None:
    """A subtree contributing no case would silently lose all its coverage."""
    assert {path for path, *_ in _CASES} == {
        path for path, _cls in _FLAT_KWARG_SUBTREES
    }, (
        "subtrees with no case: "
        f"{sorted({p for p, _c in _FLAT_KWARG_SUBTREES} - {p for p, *_ in _CASES})}"
    )


@pytest.mark.parametrize(
    ("path", "config_class", "field", "from_file", "from_cli"),
    _CASES,
    ids=[f"{path}.{field}" for path, _c, field, _a, _b in _CASES],
)
def test_config_file_value_survives_and_cli_overrides_it(
    tmp_path: Path,
    path: str,
    config_class: type[ConfigFileModel],
    field: str,
    from_file: Any,
    from_cli: Any,
) -> None:
    def as_yaml(value: Any) -> Any:
        return value.value if isinstance(value, Enum) else value

    sibling = next(((f, v) for f, v in _SIBLINGS[path] if f != field), None)
    section: dict[str, Any] = dict(_ENABLERS.get(path, {}))
    section[field] = as_yaml(from_file)
    if sibling is not None:
        section[sibling[0]] = as_yaml(sibling[1])
    nested: Any = section
    for part in reversed(path.split(".")):
        nested = {part: nested}
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(nested), encoding="utf-8")

    def routed(read: str, /, **cli: Any) -> Any:
        target: Any = PipelineArgs.from_flat_kwargs(
            config_file=str(recipe), **cli
        )
        for part in path.split("."):
            target = getattr(target, part)
        return getattr(target, read)

    def stored(read: str, value: Any) -> Any:
        return getattr(config_class(**{read: value}), read)

    assert stored(field, from_file) != stored(field, from_cli), (
        "candidates are indistinguishable once stored, so neither assertion "
        "below can observe routing"
    )
    assert routed(field) == stored(field, from_file)
    assert routed(field, **{field: from_cli}) == stored(field, from_cli)
    if sibling is not None:
        assert routed(sibling[0], **{field: from_cli}) == stored(
            sibling[0], sibling[1]
        ), "a CLI flag reset a sibling field set in the config file"


_FROM_ARGS_REBUILT = ("profiling", "runtime", "sampling")

_FROM_ARGS_CASES = [
    (path, config_class, field, value)
    for path, config_class, field, value, _ in _CASES
    if path in _FROM_ARGS_REBUILT
]


@pytest.mark.parametrize(
    ("path", "config_class", "field", "value"),
    _FROM_ARGS_CASES,
    ids=[f"{path}.{field}" for path, _c, field, _v in _FROM_ARGS_CASES],
)
def test_from_args_preserves_every_explicitly_set_field(
    path: str, config_class: type[ConfigFileModel], field: str, value: Any
) -> None:
    """Every explicitly-set field must survive ``PipelineConfig.from_args``.

    Regression guard for the kadabra-v5 incident (ENABLE-2881): ``from_args``
    rebuilt ``SamplingConfig`` from a hand-picked field list that omitted
    ``enable_tool_call_constrained_decode``, so production served with the
    flag reset to its default and ``tool_choice="required"`` was
    grammar-forced despite ``--no-enable-tool-call-constrained-decode``.
    Sweeping ``model_fields`` covers future fields the day they are added.
    """
    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(**{field: value})
    )
    expected = getattr(config_class(**{field: value}), field)
    assert getattr(getattr(config, path), field) == expected, (
        f"{path}.{field} was set via CLI flat kwargs but reset by "
        "PipelineConfig.from_args — the worker would serve with the default"
    )


def test_tool_call_constrained_decode_flag_reaches_worker_config() -> None:
    """The exact field production lost; kept explicit so the incident's
    reproducer survives even if the matrix's value derivation changes."""
    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(enable_tool_call_constrained_decode=False)
    )
    assert config.sampling.enable_tool_call_constrained_decode is False


def test_pipeline_args_surface_is_frozen() -> None:
    """Top-level fields are fixed at construction; assignment raises."""
    args = PipelineArgs(max_length=128)
    field = "max_length"
    with pytest.raises(ValidationError, match="frozen"):
        setattr(args, field, 256)


def test_model_copy_preserves_model_fields_set() -> None:
    """Pins the ``model_copy`` invariant internal callers depend on: updated
    keys are marked set and untouched fields stay unset, because
    ``PipelineConfig.from_args`` rebuilds sub-configs from ``model_fields_set``.
    """
    args = PipelineArgs(max_length=128)
    copied = args.model_copy(update={"max_length": 512})
    assert copied.max_length == 512
    assert args.max_length == 128
    assert "max_length" in copied.model_fields_set
    assert "model_path" not in copied.model_fields_set


def test_private_attrs_stay_assignable() -> None:
    """Callers set ``_weights_repo_id`` post-construction; frozen only
    guards declared fields, not private attributes.
    """
    args = PipelineArgs()
    args._weights_repo_id = "org/weights-repo"
    assert args._weights_repo_id == "org/weights-repo"


def test_cyclopts_flat_binding_constructs_a_new_instance() -> None:
    """The cascade CLI binds PipelineArgs with ``Parameter(name="*")`` and a
    default instance; cyclopts must build a fresh instance from parsed flags
    (the constructor path) rather than mutating the default in place.
    """
    app = cyclopts.App(name="probe", help_formatter="plain")
    default_args = PipelineArgs()
    received: list[PipelineArgs] = []

    @app.default
    def probe(
        pipeline_args: Annotated[
            PipelineArgs, Parameter(name="*")
        ] = default_args,
    ) -> None:
        received.append(pipeline_args)

    try:
        app(["--model-path", "org/some-model", "--max-length", "1234"])
    except SystemExit as e:  # cyclopts may exit(0) after a command
        if e.code:
            raise

    (args,) = received
    assert args is not default_args
    assert args.model_path == "org/some-model"
    assert args.max_length == 1234
    assert default_args.model_path == ""
    assert default_args.max_length is None
