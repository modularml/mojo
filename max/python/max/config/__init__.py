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
"""MAX config classes."""

from __future__ import annotations

import argparse
import enum
import logging
import types
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import (
    Any,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml
from pydantic import BaseModel, model_validator

from .base_model import MAXBaseModel
from .config_file_model import ConfigFileModel

logger = logging.getLogger("max.pipelines")

T = TypeVar("T", bound="MAXConfig")

STRINGLY_TYPED_BASIC_MAPPING: Mapping[str, type] = {
    "bool": bool,
    "float": float,
    "int": int,
    "str": str,
}

MAX_CONFIG_METADATA_FIELDS: Mapping[str, type[Any]] = {
    "name": str,
    "description": str,
    "version": str,
    "depends_on": str,
}


def _resolve_enum_type(
    enum_name: str, config_class: type[MAXConfig]
) -> type[enum.Enum] | None:
    """Dynamically resolve an enum type by name.

    This allows the base class to resolve enum types without importing them
    directly, making the system more modular.

    Args:
        enum_name: The name of the enum type to resolve.
        config_class: The config class requesting the enum resolution.

    Returns:
        The resolved enum type, or None if it cannot be resolved.
    """
    # Check if the class has a custom enum mapping
    if hasattr(config_class, "_get_enum_mapping"):
        enum_mapping = config_class._get_enum_mapping()
        if enum_name in enum_mapping:
            return enum_mapping[enum_name]

    return None


# TODO: I believe we have some utils like this in our entrypoint code. We should
# move and consolidate them.
def _is_union_type(origin: Any) -> bool:
    """Check if the given origin represents a union type (Union or Python 3.10+ UnionType).

    Args:
        origin: The result of get_origin(field_type)

    Returns:
        True if the origin represents a union type, False otherwise.
    """
    # Check for traditional Union syntax
    is_union = origin is Union

    if not is_union and origin is not None:
        # Check for Python 3.10+ types.UnionType
        if hasattr(types, "UnionType"):
            is_union = origin is types.UnionType
        if not is_union:
            # Fallback: check by name
            origin_name = getattr(origin, "__name__", str(origin))
            is_union = "UnionType" in origin_name

    return is_union


def _get_argparse_type_and_action(
    field_type: Any, field_default: Any
) -> tuple[Any, str | type[argparse.Action] | None]:
    """Determine the appropriate argparse type and action for a field type.

    This reuses the same type analysis logic as convert_max_config_value but
    returns argparse-compatible type and action parameters.

    Args:
        field_type: The field type to analyze
        field_default: The default value for the field

    Returns:
        Tuple of (type_func, action) where:
        - type_func: The type function to pass to add_argument() or None for action args
        - action: The action string for add_argument() or None for typed args
    """

    # Get the origin and args for generic types (e.g., int | None -> Union, (int, NoneType))
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Handle Optional types (which are Union[T1, T2, ...] or Python 3.10+ UnionType)
    if _is_union_type(origin):
        # Check if this is Optional[T] (Union[T, None])
        if len(args) == 2 and type(None) in args:
            # This is Optional[T], get the non-None type
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return _get_argparse_type_and_action(non_none_type, field_default)
        else:
            # Complex Union - default to string
            return str, None
    elif origin is list:
        if args:
            # For List[T], the parser should convert each element to T
            element_type = args[0]
            if element_type in (int, float, str):
                return element_type, None
            elif isinstance(element_type, type) and issubclass(
                element_type, enum.Enum
            ):
                return str, None  # Enums converted from strings
            else:
                return str, None
        else:
            return str, None

    # Handle basic types
    if field_type in (int, float, str):
        return field_type, None

    # Handle enum types
    elif isinstance(field_type, type) and issubclass(field_type, enum.Enum):
        return str, None  # Enums are parsed as strings then converted

    # Handle boolean conversion
    elif field_type is bool:
        return None, argparse.BooleanOptionalAction

    # Handle modern list syntax (list[T]) - fallback for types not caught above
    if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
        if hasattr(field_type, "__args__") and field_type.__args__:
            element_type = field_type.__args__[0]
            if element_type in (int, float, str):
                return element_type, None
            elif isinstance(element_type, type) and issubclass(
                element_type, enum.Enum
            ):
                return str, None  # Enums converted from strings
            else:
                return str, None
        else:
            return str, None

    # Default to string for unknown types
    return str, None


# TODO: config_class is now passed in here. This function should just be a
# class method of MAXConfig.
def convert_max_config_value(
    config_class: type[MAXConfig], value: Any, field_type: Any, field_name: str
) -> Any:
    """Converts a config value to the appropriate type.

    Handles enums, Optional types, Union types, lists, and basic types.

    Args:
        config_class: The MAXConfig class requesting the conversion (required for enum resolution).
        value: The value from the configuration file.
        field_type: The expected type of the field.
        field_name: The name of the field (for error messages).

    Returns:
        The converted value.

    Raises:
        ValueError: If the value cannot be converted.
    """
    # Handle None values
    if value is None:
        return None

    # Get the origin and args for generic types (e.g., int | None -> Union, (int, NoneType))
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Handle Optional types (which are Union[T1, T2, ...] or Python 3.10+ UnionType)
    if _is_union_type(origin):
        # Check if this is Optional[T] (Union[T, None])
        if len(args) == 2 and type(None) in args:
            # This is Optional[T], get the non-None type
            non_none_type = args[0] if args[1] is type(None) else args[1]
            if value is None:
                return None
            # Recursively convert using the non-None type
            return convert_max_config_value(
                config_class=config_class,
                value=value,
                field_type=non_none_type,
                field_name=field_name,
            )
        else:
            # This is a more complex Union, try each type until one works
            last_error = None
            for arg_type in args:
                try:
                    return convert_max_config_value(
                        config_class=config_class,
                        value=value,
                        field_type=arg_type,
                        field_name=field_name,
                    )
                except (ValueError, TypeError) as e:
                    last_error = e
                    continue
            # If none of the union types worked, raise the last error
            if last_error:
                raise ValueError(
                    f"Cannot convert '{value}' to any type in {field_type}: {last_error}"
                )
            else:
                raise ValueError(f"Cannot convert '{value}' to {field_type}")
    # Handle List types
    elif origin is list:
        if not isinstance(value, list):
            raise ValueError(
                f"Expected list for field '{field_name}', got {type(value)}"
            )

        # If we have type args, convert each element
        if args:
            element_type = args[0]
            return [
                convert_max_config_value(
                    config_class=config_class,
                    value=item,
                    field_type=element_type,
                    field_name=f"{field_name}[{i}]",
                )
                for i, item in enumerate(value)
            ]
        else:
            # No type information, return as-is
            return value

    # Handle string-typed fields.
    if isinstance(field_type, str):
        # Resolve enum type dynamically
        resolved_enum_type = _resolve_enum_type(field_type, config_class)
        if resolved_enum_type:
            field_type = resolved_enum_type
        elif field_type in STRINGLY_TYPED_BASIC_MAPPING:
            field_type = STRINGLY_TYPED_BASIC_MAPPING[field_type]

    # Handle Literal types — validate the value is one of the allowed literals
    if origin is Literal:
        if value not in args:
            raise ValueError(
                f"Invalid value '{value}' for field '{field_name}'. "
                f"Allowed values: {', '.join(repr(a) for a in args)}"
            )
        return value

    # Handle basic types
    if field_type in (int, float, str):
        try:
            return field_type(value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert '{value}' to {field_type.__name__} for field '{field_name}': {e}"
            ) from e
    # Handle enum types
    elif isinstance(field_type, type) and issubclass(field_type, enum.Enum):
        return _convert_enum_value(
            value=value,
            enum_type=field_type,
            field_name=field_name,
        )
    # Handle boolean conversion
    elif field_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower_value = value.lower()
            if lower_value in ("true", "1", "yes", "on"):
                return True
            elif lower_value in ("false", "0", "no", "off"):
                return False
            else:
                raise ValueError(
                    f"Cannot convert '{value}' to bool for field '{field_name}': "
                    f"expected 'true'/'false', '1'/'0', 'yes'/'no', or 'on'/'off'"
                )
        try:
            # Handle numeric values (0/1)
            if isinstance(value, int | float):
                return bool(value)
            else:
                # Try to convert to int first, then to bool
                return bool(int(value))
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert '{value}' to bool for field '{field_name}': {e}"
            ) from e

    # Handle modern list syntax (list[T]) - fallback for types not caught above
    if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
        # This handles list[str], list[int], etc. in Python 3.9+
        if hasattr(field_type, "__args__") and field_type.__args__:
            element_type = field_type.__args__[0]
            if not isinstance(value, list):
                raise ValueError(
                    f"Expected list for field '{field_name}', got {type(value)}"
                )
            return [
                convert_max_config_value(
                    config_class=config_class,
                    value=item,
                    field_type=element_type,
                    field_name=f"{field_name}[{i}]",
                )
                for i, item in enumerate(value)
            ]
        else:
            # No type information, return as-is
            return value if isinstance(value, list) else [value]

    raise ValueError(f"Unexpected field type: {field_type}")


def _convert_enum_value(
    value: Any, enum_type: type[enum.Enum], field_name: str
) -> enum.Enum:
    """Convert a value to an enum type (case-insensitive).

    Args:
        value: The value to convert.
        enum_type: The enum type to convert to.
        field_name: The field name for error messages.

    Returns:
        The converted enum value.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if isinstance(value, enum_type):
        return value
    elif isinstance(value, str):
        value_casefolded = value.casefold()
        # Try to get the enum by name first (case-insensitive)
        for enum_member in enum_type:
            if enum_member.name.casefold() == value_casefolded:
                return enum_member

    try:
        return enum_type(int(value))
    except ValueError:
        pass

    # If we get here, the conversion failed
    valid_names = [e.name for e in enum_type]
    valid_values = [e.value for e in enum_type]
    raise ValueError(
        f"Invalid enum value '{value}' for {field_name}. "
        f"Valid names: {valid_names}, valid values: {valid_values}"
    )


def get_default_max_config_file_section_name(
    config_class: type[MAXConfig],
) -> str:
    """Gets the default section name for a MAXConfig class.

    Args:
        config_class: The config class.

    Returns:
        The default section name.

    Raises:
        ValueError: If the config class doesn't define a _config_file_section_name attribute.
    """
    section_name = getattr(config_class, "_config_file_section_name", None)
    if section_name is None:
        raise ValueError(
            f"Config class {config_class.__name__} must define a '_config_file_section_name' class attribute. "
            f"This attribute specifies the expected configuration section name for this config class."
        )
    return section_name


def deep_merge_max_configs(
    base_config: dict[str, Any], child_config: dict[str, Any]
) -> dict[str, Any]:
    """Deep merge two MAXConfig configuration dictionaries.

    Args:
        base_config: The base MAXConfig configuration dictionary.
        child_config: The child MAXConfig configuration dictionary that takes precedence.

    Returns:
        Merged MAXConfig configuration dictionary with deep merging of nested MAXConfig dictionaries.
    """
    merged = base_config.copy()

    for key, value in child_config.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            # Both are MAXConfig dictionaries, merge them recursively.
            merged[key] = deep_merge_max_configs(merged[key], value)
        else:
            # Child value takes precedence (either not a dict or key doesn't exist in base).
            merged[key] = value

    return merged


def resolve_max_config_inheritance(
    config_dict: dict[str, Any],
    config_class: type[MAXConfig],
    config_file_path: Path | None = None,
) -> dict[str, Any]:
    """Resolves configuration inheritance by loading base config and merging.

    Args:
        config_dict: The current configuration dictionary.
        config_class: The config class being loaded.
        config_file_path: Path to the config file being loaded (used to resolve relative paths).

    Returns:
        Merged configuration dictionary with inheritance resolved.
    """
    depends_on = config_dict.get("depends_on")
    if not depends_on:
        return config_dict

    # Try to load the base config
    try:
        base_config_path = Path(depends_on)

        # If the path is relative and we have a config file path, resolve it relative to the config file's directory
        if not base_config_path.is_absolute() and config_file_path is not None:
            base_config_path = config_file_path.parent / base_config_path

        if not base_config_path.exists():
            raise FileNotFoundError(
                f"Base configuration file not found: {base_config_path}"
            )

        logger.info(f"Loading base configuration from {base_config_path}")

        # Load base config
        with open(base_config_path, encoding="utf-8") as f:
            base_config_dict = yaml.safe_load(f)

        # Keep this assert to help with type checking
        if not isinstance(base_config_dict, dict):
            raise ValueError(
                f"Base configuration file {base_config_path} must contain a dictionary at the top level"
            )

        # Recursively resolve inheritance in base config
        base_config_dict = resolve_max_config_inheritance(
            config_dict=base_config_dict,
            config_class=config_class,
            config_file_path=base_config_path,
        )

        # Merge base config with current config (current takes precedence)
        return deep_merge_max_configs(base_config_dict, config_dict)

    except (FileNotFoundError, ValueError):
        # Re-raise FileNotFoundError and ValueError exceptions (these are intentional validation errors)
        raise
    except Exception as e:
        logger.warning(f"Failed to load base configuration '{depends_on}': {e}")
        return config_dict


def _extract_max_config_data(
    config_dict: dict[str, Any],
    config_class: type[MAXConfig],
    section_name: str | None = None,
    config_file_path: Path | None = None,
) -> dict[str, Any]:
    """Extract config data for a specific MAXConfig class from MAXConfig files.

    Args:
        config_dict: The loaded YAML configuration dictionary.
        config_class: The config class we're extracting data for.
        section_name: Optional specific section name to look for.

    Returns:
        Configuration data for the specific config class.

    Raises:
        ValueError: If the appropriate config section cannot be found.
    """
    # Check if this looks like a full MAXConfig file
    # Full MAXConfig files have metadata fields like name, description, version, etc.
    has_metadata = any(
        field in config_dict for field in MAX_CONFIG_METADATA_FIELDS
    )

    # Get class fields to help determine if this is an individual config
    class_fields = {field.name for field in fields(config_class)}

    # Check if the top-level keys are mostly class fields (individual config)
    matching_fields = set(config_dict.keys()) & class_fields

    if not has_metadata and len(matching_fields) > 0:
        # This looks like a partial MAXConfig file
        logger.debug(
            f"Detected partial MAXConfig file for {config_class.__name__}"
        )
        return config_dict

    # This looks like a full MAXConfig file
    logger.debug(f"Detected full MAXConfig file for {config_class.__name__}")

    section_name = section_name or get_default_max_config_file_section_name(
        config_class
    )

    # Handle inheritance via depends_on
    config_dict = resolve_max_config_inheritance(
        config_dict=config_dict,
        config_class=config_class,
        config_file_path=config_file_path,
    )

    # Look for the appropriate config section
    if section_name in config_dict:
        section_data = config_dict[section_name]
        if isinstance(section_data, dict):
            return section_data
        else:
            raise ValueError(
                f"Section '{section_name}' must be a dictionary, got {type(section_data)}"
            )
    else:
        # Section not found - list available sections for helpful error message
        available_sections = [
            key
            for key in config_dict
            if key not in MAX_CONFIG_METADATA_FIELDS
            and isinstance(config_dict[key], dict)
        ]
        raise ValueError(
            f"Section '{section_name}' not found in configuration file. "
            f"Available sections: {available_sections}. "
            f"Use section_name parameter to specify a different section."
        )


@dataclass
class MAXConfig:
    """Abstract base class for all MAX configs.

    There are some invariants that :obj:`MAXConfig` classes should follow:
    - All config classes should be dataclasses.
    - All config classes should have a :obj:`help()` method that returns a dictionary of config
    options and their descriptions.
    - All config classes dataclass fields should have default values, and hence
    can be trivially initialized via :obj:`cls()`.
    - All config classes should be frozen (except :obj:`KVCacheConfig` for now), to
    avoid accidental modification of config objects.
    - All config classes must have mutually exclusive dataclass fields among
    themselves.
    - All config classes must define a `_config_file_section_name` class attribute specifying
    their expected configuration section name.
    """

    @staticmethod
    @abstractmethod
    def help() -> dict[str, str]:
        """Returns a dictionary of config options and their descriptions."""
        ...

    @staticmethod
    def get_default_field_choices() -> dict[str, list[str]]:
        """Get default valid choices for fields that have constrained values.

        Returns:
            Dictionary mapping field names to their valid choices.
        """
        return {}

    @classmethod
    def get_default_required_fields(cls) -> set[str]:
        """Get default required fields for the config."""
        return set()

    @classmethod
    def _get_enum_mapping(cls) -> Mapping[str, type[enum.Enum]]:
        """Get the enum mapping for this config class.

        This method automatically collects enum mappings from all parent classes
        in the inheritance hierarchy, creating a union of all enums.

        Subclasses can override this method to provide their own enum mappings
        without requiring the base class to import all possible enum types.

        Returns:
            A mapping from string names to enum types, including all enums
            from parent classes in the inheritance hierarchy.
        """
        # Start with the current class's enum mapping
        enum_mapping = {}

        # Get the Method Resolution Order (MRO) to traverse the inheritance hierarchy
        # This includes the current class and all its parent classes
        for base_class in cls.__mro__:
            # Skip the current class (cls) as we'll handle it separately
            if base_class is cls:
                continue

            # Check if the base class has a _get_enum_mapping method
            if hasattr(base_class, "_get_enum_mapping"):
                try:
                    # Get the enum mapping from the parent class
                    parent_enum_mapping = base_class._get_enum_mapping()
                    if parent_enum_mapping:
                        # Merge the parent's enum mapping into our current mapping
                        # Later mappings (from more derived classes) take precedence
                        enum_mapping.update(parent_enum_mapping)
                except Exception as e:
                    # Log warning but continue to avoid breaking the system
                    logger.warning(
                        f"Failed to get enum mapping from {base_class.__name__}: {e}"
                    )

        # Now add the current class's enum mapping (this takes precedence)
        # This allows subclasses to override parent enum mappings if needed
        if hasattr(cls, "_get_enum_mapping_impl"):
            try:
                current_enum_mapping = cls._get_enum_mapping_impl()
                if current_enum_mapping:
                    enum_mapping.update(current_enum_mapping)
            except Exception as e:
                logger.warning(
                    f"Failed to get enum mapping from {cls.__name__}: {e}"
                )

        return enum_mapping

    @classmethod
    def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
        """Internal implementation method for enum mapping.

        Subclasses should override this method to provide their own enum mappings.
        The public _get_enum_mapping method will automatically collect enums from
        all parent classes and merge them with this implementation.

        Returns:
            A mapping from string names to enum types for this specific class.
        """
        return {}

    @classmethod
    def from_config_file(
        cls: type[T],
        config_path: str | Path,
        section_name: str | None = None,
    ) -> T:
        """Loads configuration from a YAML file.

        Supports both individual config files and comprehensive multi-config
        files. For comprehensive files, automatically detects the appropriate
        section based on class name.

        Args:
            config_path: The path to the YAML configuration file.
            section_name: The section name for comprehensive configs.
                Auto-detected if ``None``. Defaults to ``None``.

        Returns:
            A config instance with parameters loaded from the file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the configuration is invalid.

        Define a config subclass, then load its values from a YAML file:

        .. code-block:: python

            from dataclasses import dataclass
            from pathlib import Path
            from tempfile import TemporaryDirectory

            from max.config import MAXConfig

            @dataclass
            class MyCacheConfig(MAXConfig):
                page_size: int = 128

                @staticmethod
                def help() -> dict[str, str]:
                    return {"page_size": "Number of tokens per KV cache page."}

            with TemporaryDirectory() as tmp_dir:
                config_path = Path(tmp_dir) / "my_cache.yaml"
                config_path.write_text("page_size: 256")
                config = MyCacheConfig.from_config_file(config_path)

        .. invisible-code-block: python

            assert config.page_size == 256
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        # Load the YAML file.
        try:
            with open(config_path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(
                f"Failed to load configuration from {config_path}: {e}"
            ) from e

        # Ensure we have a proper dict[str, Any] type for mypy.
        if not isinstance(config_dict, dict):
            raise ValueError(
                "Configuration file must contain a dictionary at the top level"
            )

        # Extract the config data for this MAXConfig class from the config file.
        config_data = _extract_max_config_data(
            config_dict=config_dict,
            config_class=cls,
            section_name=section_name,
            config_file_path=config_path,
        )

        # Get the dataclass fields for this MAXConfig class.
        class_fields = {field.name: field for field in fields(cls)}

        # Resolve type hints in case of string annotations (from __future__ import annotations)
        try:
            type_hints = get_type_hints(cls)
        except (NameError, AttributeError):
            type_hints = {}

        # Filter config_data to only include valid fields for this MAXConfig class
        filtered_config = {}
        invalid_keys = []

        for key, value in config_data.items():
            if key in class_fields:
                field = class_fields[key]
                # Use resolved type hint if available, otherwise fall back to field.type
                field_type = type_hints.get(key, field.type)
                try:
                    # Handle type conversion for specific types
                    converted_value = convert_max_config_value(
                        config_class=cls,
                        value=value,
                        field_type=field_type,
                        field_name=key,
                    )
                    filtered_config[key] = converted_value
                except Exception as e:
                    raise ValueError(
                        f"Invalid value for field '{key}': {e}"
                    ) from e
            else:
                invalid_keys.append(key)

        if invalid_keys:
            logger.warning(
                f"Ignoring unknown configuration keys for {cls.__name__}: {invalid_keys}"
            )

        # Create and return the config instance
        try:
            return cls(**filtered_config)
        except Exception as e:
            raise ValueError(
                f"Failed to create {cls.__name__} instance: {e}"
            ) from e

    def _get_group_description(
        self, group_name: str, group_fields: list[Any]
    ) -> str | None:
        """Get the description for an argument group.

        Args:
            group_name: The name of the group.
            group_fields: List of dataclass field objects in this group.

        Returns:
            The group description if found, None otherwise.
        """
        # TODO: I think the group_description should live somewhere else more global
        # and not defined in an individual config field. This isn't a great design
        # but maybe it's ok for now until a follow up PR.
        # Look for group_description in any field's metadata
        for field_obj in group_fields:
            group_description = field_obj.metadata.get("group_description")
            if group_description:
                return group_description
        return None

    def _add_field_as_argument(
        self,
        parser_or_group: argparse.ArgumentParser | argparse._ArgumentGroup,
        field_obj: Any,
        type_hints: dict[str, Any],
        choices_provider: dict[str, list[str]],
        required_params: set[str],
    ) -> None:
        """Add a single field as an argument to a parser or argument group.

        Args:
            parser_or_group: The ArgumentParser or ArgumentGroup to add the argument to.
            field_obj: The dataclass field object.
            type_hints: Resolved type hints for the class.
            choices_provider: Dictionary mapping field names to their valid choices.
            required_params: Set of field names that should be marked as required.
        """
        # Skip internal fields
        if field_obj.name.startswith("_"):
            return

        field_name = field_obj.name.replace("_", "-")
        arg_name = f"--{field_name}"

        # Use resolved type hint if available, otherwise fall back to field.type
        field_type = type_hints.get(field_obj.name, field_obj.type)

        # Use helper function to determine argparse parameters
        arg_type, action = _get_argparse_type_and_action(
            field_type, field_obj.default
        )

        # Build argument kwargs
        # Use the actual config value, not the field default
        field_value = getattr(self, field_obj.name)

        # Handle enum types specially - argparse expects string values for enums
        if (
            hasattr(field_value, "value")
            and hasattr(field_value, "__class__")
            and issubclass(field_value.__class__, enum.Enum)
        ):
            # For enums, use the string value as default but we'll need to convert back
            arg_kwargs = {
                "default": field_value.value
                if field_value
                else field_obj.default
            }
        else:
            arg_kwargs = {"default": field_value}

        # Apply choices from choices_provider
        if field_obj.name in choices_provider:
            arg_kwargs["choices"] = choices_provider[field_obj.name]

        # Add help from the config class if available
        if hasattr(self, "help"):
            help_dict = self.help()
            if field_obj.name in help_dict:
                arg_kwargs["help"] = help_dict[field_obj.name]

        # Mark as required if specified in required_params
        # The required_params set is already calculated by the caller (e.g., smart required fields logic)
        if field_obj.name in required_params:
            arg_kwargs["required"] = True

        # Add argument with appropriate type and action
        if action:
            # Boolean fields with action
            arg_kwargs["action"] = action
            parser_or_group.add_argument(arg_name, **arg_kwargs)
        elif get_origin(field_type) is list:
            # List fields need nargs
            arg_kwargs.update({"type": arg_type, "nargs": "*"})
            parser_or_group.add_argument(arg_name, **arg_kwargs)
        else:
            # Regular typed fields
            arg_kwargs["type"] = arg_type
            parser_or_group.add_argument(arg_name, **arg_kwargs)

    def cli_arg_parsers(
        self,
        choices_provider: dict[str, list[str]] | None = None,
        description: str | None = None,
        formatter_class: type[argparse.HelpFormatter] | None = None,
        required_params: set[str] | None = None,
    ) -> argparse.ArgumentParser:
        """Creates an ``ArgumentParser`` populated with all config fields.

        Builds a parser with ``add_argument()`` calls for each field, using
        the loaded config values as defaults. Arguments are automatically
        grouped by their ``group`` metadata from field definitions. The
        parser's ``parse_args()`` method is wrapped to convert parsed string
        values back to their proper types (for example, enum objects) using
        :class:`MAXConfig` type conversion logic.

        Args:
            choices_provider: A dictionary mapping field names to their valid
                choices. Allows external code to specify choices for specific
                fields. Defaults to ``None``.
            description: A description for the argument parser. Defaults to
                ``None``.
            formatter_class: A formatter class for the argument parser,
                forwarded to the ``argparse.ArgumentParser`` constructor.
                Defaults to ``None``.
            required_params: A set of field names that should be marked as
                required in the argument parser, regardless of their default
                values. Defaults to ``None``.

        Returns:
            A configured ``ArgumentParser`` with an enhanced ``parse_args()``
            method that:

            - Uses loaded config values as argument defaults.
            - Converts parsed values to proper types (enums and similar).
            - Groups arguments by field metadata for better organization.
            - Maintains compatibility with standard ``argparse`` usage.

        Build a parser from a config instance and parse a list of arguments,
        restricting a field to a set of valid choices:

        .. code-block:: python

            from dataclasses import dataclass

            from max.config import MAXConfig

            @dataclass
            class MyServerConfig(MAXConfig):
                backend: str = "modular"

                @staticmethod
                def help() -> dict[str, str]:
                    return {"backend": "Serving backend to use."}

            config = MyServerConfig()
            parser = config.cli_arg_parsers(
                choices_provider={"backend": ["modular", "vllm"]},
            )
            args = parser.parse_args(["--backend", "vllm"])

        .. invisible-code-block: python

            assert args.backend == "vllm"
        """

        # Create parser
        additional_argument_parser_args: dict[str, Any] = {}

        if formatter_class is not None:
            additional_argument_parser_args["formatter_class"] = formatter_class

        parser = argparse.ArgumentParser(
            description=description, **additional_argument_parser_args
        )
        choices_provider = choices_provider or self.get_default_field_choices()
        # Use provided required_params if it's not None, otherwise use default required fields
        required_params = (
            required_params
            if required_params is not None
            else self.get_default_required_fields()
        )

        # Resolve type hints in case of string annotations (from __future__ import annotations)
        try:
            type_hints = get_type_hints(self.__class__)
        except (NameError, AttributeError):
            type_hints = {}

        # Group fields by their 'group' metadata
        groups: dict[str, list[Any]] = {}
        ungrouped_fields: list[Any] = []

        for field_obj in fields(self):
            if field_obj.name.startswith("_"):
                continue

            group_name = field_obj.metadata.get("group")
            if group_name is not None:
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append(field_obj)
            else:
                ungrouped_fields.append(field_obj)

        # Create argument groups
        for group_name, group_fields in groups.items():
            group_description = self._get_group_description(
                group_name, group_fields
            )
            group = parser.add_argument_group(group_name, group_description)

            # Add arguments to this group
            for field_obj in group_fields:
                self._add_field_as_argument(
                    group,
                    field_obj,
                    type_hints,
                    choices_provider,
                    required_params,
                )

        # Add ungrouped fields to main parser
        for field_obj in ungrouped_fields:
            self._add_field_as_argument(
                parser, field_obj, type_hints, choices_provider, required_params
            )

        # Create a wrapper class to override parse_args method
        class MAXConfigArgumentParser(argparse.ArgumentParser):
            def __init__(
                self,
                parser_instance: argparse.ArgumentParser,
                config_instance: MAXConfig,
            ):
                # Copy all attributes from the original parser
                self.__dict__.update(parser_instance.__dict__)
                self._config_instance = config_instance
                self._original_parse_args = parser_instance.parse_args

            def parse_args(  # type: ignore[override]  # noqa: ANN202
                self,
                args: list[str] | None = None,
                namespace: argparse.Namespace | None = None,
            ):
                # Parse with the original method
                parsed_namespace = self._original_parse_args(args, namespace)

                # Convert string values back to proper types using MAXConfig conversion logic
                converted_dict = {}
                class_fields = {
                    field.name: field for field in fields(self._config_instance)
                }

                for attr_name in dir(parsed_namespace):
                    if attr_name.startswith("_"):
                        continue

                    parsed_value = getattr(parsed_namespace, attr_name)

                    if attr_name in class_fields:
                        field = class_fields[attr_name]
                        try:
                            # Use the same conversion logic as from_config_file
                            converted_value = convert_max_config_value(
                                config_class=self._config_instance.__class__,
                                value=parsed_value,
                                field_type=field.type,
                                field_name=attr_name,
                            )
                            converted_dict[attr_name] = converted_value
                        except Exception:
                            # If conversion fails, keep the parsed value
                            converted_dict[attr_name] = parsed_value
                    else:
                        # Unknown fields remain as-is
                        converted_dict[attr_name] = parsed_value

                return argparse.Namespace(**converted_dict)

        # Return the wrapped parser
        return MAXConfigArgumentParser(parser, self)


all = [
    "MAXBaseModel",
    "ConfigFileModel",
]
