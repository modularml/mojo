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


"""Utilities for working with Config objects in Click."""

from __future__ import annotations

import functools
import inspect
import json
import pathlib
from collections.abc import Callable
from dataclasses import MISSING
from enum import Enum
from pathlib import Path
from types import SimpleNamespace, UnionType
from typing import (
    Any,
    Literal,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import click
from max.driver import DeviceSpec
from max.pipelines.diffusion.config import DenoisingCacheSettings
from max.pipelines.lib import (
    KVCacheConfig,
    LoRAConfig,
    MAXModelConfig,
    PipelineConfig,
    PipelineRuntimeConfig,
    ProfilingConfig,
    SamplingConfig,
    SpeculativeConfig,
)
from pydantic import BaseModel
from typing_extensions import ParamSpec

from .device_options import DevicesOptionType

VALID_CONFIG_TYPES = [
    str,
    bool,
    Enum,
    Path,
    DeviceSpec,
    int,
    float,
    dict,
    BaseModel,
]

_P = ParamSpec("_P")
_R = TypeVar("_R")


class JSONType(click.ParamType):
    """Click parameter type for JSON input."""

    name = "json"

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> Any:
        # Already-structured values need no parsing: a dict from a config file,
        # or a Pydantic model when the option's default is a model instance
        # rather than None.
        if not isinstance(value, str):
            return value
        # Auto-detect file path vs inline JSON/YAML
        if not value.lstrip().startswith(("{", "[")):
            path = Path(value)
            if path.is_file():
                import yaml

                with open(path) as f:
                    return yaml.safe_load(f)
            self.fail(f"Not valid JSON and file not found: {value}", param, ctx)
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            self.fail(f"Invalid JSON: {e}", param, ctx)


def get_interior_type(type_hint: type | str | Any) -> type[Any]:
    interior_args = set(get_args(type_hint)) - {type(None)}
    if len(interior_args) > 1:
        raise ValueError(
            "Parsing does not currently supported Union type, with more than"
            f" one non-None type: {type_hint}"
        )

    return get_args(type_hint)[0]


def is_optional(type_hint: type | str | Any) -> bool:
    return get_origin(type_hint) in (Union, UnionType) and type(
        None
    ) in get_args(type_hint)


def is_flag(field_type: Any) -> bool:
    if field_type is bool:
        return True
    return is_optional(field_type) and get_interior_type(field_type) is bool


def validate_field_type(field_type: Any) -> bool:
    if is_optional(field_type):
        test_type = get_args(field_type)[0]
    else:
        test_type = field_type

    if get_origin(test_type) is list:
        test_type = get_interior_type(test_type)

    if get_origin(test_type) is dict:
        return True

    if get_origin(test_type) is Literal:
        return True

    for valid_type in VALID_CONFIG_TYPES:
        if valid_type == test_type:
            return True

        if get_origin(valid_type) is None and inspect.isclass(test_type):
            if issubclass(test_type, valid_type):
                return True
    raise ValueError(f"type '{test_type}' not supported in config.")


def get_field_type(field_type: Any) -> Any:
    validate_field_type(field_type)

    # Get underlying core field type, is Optional or list.
    if is_optional(field_type):
        field_type = get_interior_type(field_type)
    if get_origin(field_type) is list:
        field_type = get_interior_type(field_type)

    # Update the field_type to be format specific.
    if field_type == Path:
        field_type = click.Path(path_type=pathlib.Path)
    elif (
        get_origin(field_type) is dict
        or field_type is dict
        or (inspect.isclass(field_type) and issubclass(field_type, BaseModel))
    ):
        field_type = JSONType()
    elif get_origin(field_type) is Literal:
        field_type = click.Choice(
            [str(v) for v in get_args(field_type)], case_sensitive=False
        )
    elif inspect.isclass(field_type):
        if issubclass(field_type, Enum):
            field_type = click.Choice(list(field_type), case_sensitive=False)

    return field_type


def get_default(dataclass_field: Any) -> Any:
    if dataclass_field.default_factory != MISSING:
        default = dataclass_field.default_factory()
    elif dataclass_field.default != MISSING:
        default = dataclass_field.default
    else:
        default = None

    return default


def is_multiple(field_type: Any) -> bool:
    return get_origin(field_type) is list


def get_normalized_flag_names(
    dataclass_field: Any, field_type: Any
) -> tuple[str, ...]:
    normalized_name = dataclass_field.name.lower().replace("_", "-")

    if dataclass_field.name == "model_path":
        return ("--model", "--model-path", dataclass_field.name)

    if is_flag(field_type):
        return (f"--{normalized_name}/--no-{normalized_name}",)

    return (f"--{normalized_name}",)


def create_click_option(
    help_for_fields: dict[str, str],
    dataclass_field: Any,
    field_type: Any,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    # Get Help text.
    help_text = help_for_fields.get(dataclass_field.name)

    # Get help field.
    return click.option(
        *get_normalized_flag_names(dataclass_field, field_type),
        show_default=False,  # Many strings include default already, and True breaks Sphinx docs
        help=help_text,
        is_flag=is_flag(field_type),
        default=get_default(dataclass_field),
        multiple=is_multiple(field_type),
        type=get_field_type(field_type),
    )


def get_fields_from_pydantic_model(
    cls: type[BaseModel],
) -> list[SimpleNamespace]:
    """Get fields from a Pydantic model.

    Returns a list of Field objects compatible with dataclass Field API.
    """

    # Try to import PydanticUndefined to detect unset defaults
    try:
        from pydantic import PydanticUndefined
    except ImportError:
        try:
            from pydantic_core import PydanticUndefined
        except ImportError:
            # Fallback: create a unique sentinel if we can't import it
            PydanticUndefined = type("PydanticUndefined", (), {})()

    pydantic_fields = []
    model_fields = cls.model_fields

    for field_name, field_info in model_fields.items():
        # Create a simple object that mimics dataclass Field.
        field_obj = SimpleNamespace()
        field_obj.name = field_name

        # Extract default value from Pydantic FieldInfo
        # In Pydantic v2, field_info.default is PydanticUndefined if not set,
        # otherwise it's the actual default value (which can be None)
        default_value = getattr(field_info, "default", PydanticUndefined)
        if default_value is not PydanticUndefined:
            field_obj.default = default_value
        else:
            field_obj.default = MISSING

        # Extract default_factory
        # In Pydantic v2, default_factory is None if not set, otherwise it's the factory
        default_factory = getattr(field_info, "default_factory", None)
        if default_factory is not None:
            field_obj.default_factory = default_factory
        else:
            field_obj.default_factory = MISSING

        pydantic_fields.append(field_obj)

    return pydantic_fields


def get_config_skip_fields(cls: type[BaseModel]) -> set[str]:
    """Return config fields that should not be exposed as CLI flags."""
    skip_fields = {
        "device_specs",
        "in_dtype",
        "out_dtype",
    }
    if cls is PipelineConfig:
        skip_fields.update(
            {
                "model",
                "draft_model",
                "sampling",
                "profiling",
                "lora",
                "speculative",
                "runtime",
            }
        )
    elif cls is PipelineRuntimeConfig:
        # Skip fields that are still duplicated on PipelineConfig to avoid
        # generating duplicate CLI flags.  As fields are migrated from
        # PipelineConfig to PipelineRuntimeConfig and removed from
        # PipelineConfig, they will automatically start being generated
        # from PipelineRuntimeConfig instead.
        skip_fields.update(
            field_name
            for field_name in cls.model_fields
            if field_name in PipelineConfig.model_fields
        )
        # denoising_cache is a nested model — individual flags are
        # generated by @config_to_flag(DenoisingCacheSettings).
        skip_fields.add("denoising_cache")
    elif cls is MAXModelConfig:
        skip_fields.add("kv_cache")
    skip_fields.update(
        field_name
        for field_name in cls.model_fields
        if field_name.startswith("_")
    )
    return skip_fields


def config_to_flag(
    cls: type[BaseModel], prefix: str | None = None
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    options: list[tuple[str, Callable[..., Any]]] = []
    param_names: set[str] = set()
    help_text = {
        field_name: field_info.description
        for field_name, field_info in cls.model_fields.items()
        if field_info.description
    }
    field_types = get_type_hints(cls)
    skip_fields = get_config_skip_fields(cls)

    for _field in get_fields_from_pydantic_model(cls):
        # Skip config fields that are not exposed in CLI.
        if _field.name in skip_fields:
            continue

        original_name = _field.name
        field_type = field_types[original_name]

        if prefix:
            # Create a copy of the help text with the prefixed name
            new_name = f"{prefix}_{original_name}"

            if original_name in help_text:
                help_text[new_name] = help_text[original_name]

            # Create a new Field with the modified name by copying all attributes
            from copy import copy

            modified_field = copy(_field)
            modified_field.name = new_name
            new_option = create_click_option(
                help_text, modified_field, field_type
            )
            param_names.add(new_name)
        else:
            new_name = original_name
            new_option = create_click_option(help_text, _field, field_type)
            param_names.add(original_name)
        options.append((new_name, new_option))

    def apply_flags(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            ctx = click.get_current_context(silent=True)
            if ctx is not None and _config_file_is_set(ctx, kwargs):
                kwargs = _strip_default_params(ctx, kwargs, param_names)  # type: ignore[assignment]
            return func(*args, **kwargs)

        existing = {p.name for p in getattr(wrapped, "__click_params__", [])}
        for name, option in reversed(options):
            if name in existing:
                continue
            wrapped = option(wrapped)
            existing.add(name)
        return wrapped

    return apply_flags


def _config_file_is_set(ctx: click.Context, params: dict[str, Any]) -> bool:
    source = ctx.get_parameter_source("config_file")
    if source is None or source is click.core.ParameterSource.DEFAULT:
        return False
    return params.get("config_file") is not None


def _strip_default_params(
    ctx: click.Context,
    params: dict[str, Any],
    names: set[str],
) -> dict[str, Any]:
    # Strip Click defaults for config_to_flag fields so that
    # ConfigFileModel.load_config_file can fill them from the YAML.
    # Safe because Click defaults originate from Pydantic field defaults,
    # so for fields absent from the config file Pydantic recovers the
    # same default.
    return {
        name: value
        for name, value in params.items()
        if not (
            name in names
            and ctx.get_parameter_source(name)
            is click.core.ParameterSource.DEFAULT
        )
    }


def pipeline_config_options(func: Callable[_P, _R]) -> Callable[_P, _R]:
    # The order of these decorators must be preserved - ie. PipelineConfig
    # must be applied only after KVCacheConfig, ProfilingConfig etc.
    @config_to_flag(PipelineConfig)
    @config_to_flag(PipelineRuntimeConfig)
    @config_to_flag(MAXModelConfig)
    @config_to_flag(MAXModelConfig, prefix="draft")
    @config_to_flag(KVCacheConfig)
    @config_to_flag(DenoisingCacheSettings)
    @config_to_flag(LoRAConfig)
    @config_to_flag(ProfilingConfig)
    @config_to_flag(SamplingConfig)
    @config_to_flag(SpeculativeConfig)
    @click.option(
        "--devices",
        is_flag=False,
        type=DevicesOptionType(),
        show_default=False,
        default=None,
        help=(
            "Whether to run the model on CPU (``--devices=cpu``), GPU (``--devices=gpu``),"
            " every visible GPU (``--devices=gpu:all``), or a list of GPUs"
            " (``--devices=gpu:0,1``). An ID value can be provided optionally to"
            " indicate the device ID to target. If not provided, the model or"
            " config default is used."
        ),
    )
    @click.option(
        "--draft-devices",
        is_flag=False,
        type=DevicesOptionType(),
        show_default=False,
        default=None,
        help=(
            "Devices for the draft model in speculative decoding. "
            "If not provided, inherits from ``--devices``. "
            "Accepts the same format as ``--devices``."
        ),
    )
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        def is_device_handle(value: Any) -> TypeGuard[str | list[int]]:
            return isinstance(value, str) or (
                isinstance(value, list)
                and all(isinstance(x, int) for x in value)
            )

        # Remove the options from kwargs and replace with unified device_specs.
        devices = kwargs.pop("devices")
        draft_devices = kwargs.pop("draft_devices")
        assert devices is None or is_device_handle(devices)
        # Inherit draft_devices from devices if the target devices were explicit.
        if draft_devices is None and devices is not None:
            draft_devices = devices
        assert draft_devices is None or is_device_handle(draft_devices)

        # Enable virtual device mode if target is set.
        # This must happen BEFORE calling device_specs() which does validation.
        if kwargs.get("target"):
            from max.driver import (
                calculate_virtual_device_count_from_cli,
                set_virtual_device_api,
                set_virtual_device_count,
                set_virtual_device_target_arch,
            )
            from max.serve.config import parse_api_and_target_arch

            api, target_arch = parse_api_and_target_arch(
                cast(str, kwargs["target"])
            )
            set_virtual_device_api(api)
            set_virtual_device_target_arch(target_arch)
            target_devices = [] if devices is None else devices
            target_draft_devices = (
                [] if draft_devices is None else draft_devices
            )

            virtual_count = calculate_virtual_device_count_from_cli(
                target_devices, target_draft_devices
            )
            set_virtual_device_count(virtual_count)

        # The type ignores are necessary because "devices" is a str, but in
        # device_specs() we accept them as a DeviceHandle.
        if devices is not None:
            kwargs["device_specs"] = DevicesOptionType.device_specs(devices)  # type: ignore[arg-type]
        if draft_devices is not None:
            kwargs["draft_device_specs"] = DevicesOptionType.device_specs(
                draft_devices  # type: ignore[arg-type]
            )

        return func(*args, **kwargs)

    return wrapper


def sampling_params_options(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @click.option(
        "--top-k",
        is_flag=False,
        type=int,
        show_default=False,
        default=None,
        help="Limits the sampling to the K most probable tokens. This defaults to 255. For greedy sampling, set to 1.",
    )
    @click.option(
        "--top-p",
        is_flag=False,
        type=float,
        show_default=False,
        default=None,
        help="Only use the tokens whose cumulative probability is within the top_p threshold. This applies to the top_k tokens.",
    )
    @click.option(
        "--min-p",
        is_flag=False,
        type=float,
        show_default=False,
        default=None,
        help="Float that represents the minimum probability for a token to be considered, relative to the probability of the most likely token. Must be in [0, 1]. Set to 0 to disable this.",
    )
    @click.option(
        "--temperature",
        is_flag=False,
        type=float,
        show_default=False,
        default=None,
        help="Controls the randomness of the model's output; higher values produce more diverse responses.",
    )
    @click.option(
        "--frequency-penalty",
        is_flag=False,
        type=float,
        show_default=False,
        default=None,
        help="The frequency penalty to apply to the model's output. A positive value will penalize new tokens based on their frequency in the generated text.",
    )
    @click.option(
        "--presence-penalty",
        is_flag=False,
        type=float,
        show_default=False,
        default=None,
        help="The presence penalty to apply to the model's output. A positive value will penalize new tokens that have already appeared in the generated text at least once.",
    )
    @click.option(
        "--repetition-penalty",
        is_flag=False,
        type=float,
        show_default=False,
        default=None,
        help="The repetition penalty to apply to the model's output. Values > 1 will penalize new tokens that have already appeared in the generated text at least once.",
    )
    @click.option(
        "--max-new-tokens",
        is_flag=False,
        type=int,
        show_default=False,
        default=None,
        help="Maximum number of new tokens to generate during a single inference pass of the model.",
    )
    @click.option(
        "--min-new-tokens",
        is_flag=False,
        type=int,
        show_default=False,
        default=None,
        help="Minimum number of tokens to generate in the response.",
    )
    @click.option(
        "--ignore-eos",
        is_flag=True,
        show_default=False,
        default=None,
        help="If True, the response will ignore the EOS token, and continue to generate until the max tokens or a stop string is hit.",
    )
    @click.option(
        "--stop",
        is_flag=False,
        type=str,
        show_default=False,
        default=None,
        multiple=True,
        help="A list of detokenized sequences that can be used as stop criteria when generating a new sequence. Can be specified multiple times.",
    )
    @click.option(
        "--stop-token-ids",
        is_flag=False,
        type=str,
        show_default=False,
        default=None,
        help="A list of token ids that are used as stopping criteria when generating a new sequence. Comma-separated integers.",
    )
    @click.option(
        "--detokenize/--no-detokenize",
        is_flag=True,
        show_default=False,
        default=None,
        help="Whether to detokenize the output tokens into text.",
    )
    @click.option(
        "--seed",
        is_flag=False,
        type=int,
        show_default=False,
        default=None,
        help="Seed for the random number generator.",
    )
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return func(*args, **kwargs)

    return wrapper


def parse_task_flags(task_flags: tuple[str, ...]) -> dict[str, str]:
    """Parse task flags into a dictionary.

    The flags must be in the format `flag_name=flag_value`.

    This requires that the task flags are:
    1. Passed and interpreted as strings, including their values.
    2. Be passed as a list of strings via explicit --task-arg flags. For example:
        --task-arg=flag1=value1 --task-arg=flag2=value2

    Args:
        task_flags: A tuple of task flags.

    Returns:
        A dictionary of parsed flag values.
    """
    flags = {}
    for flag in task_flags:
        if "=" not in flag or flag.startswith("--"):
            raise ValueError(
                f"Flag must be in format 'flag_name=flag_value', got: {flag}"
            )

        flag_name, flag_value = flag.split("=", 1)
        flags[flag_name.replace("-", "_")] = flag_value
    return flags
