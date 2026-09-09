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
"""ConfigFileModel for Pydantic-based config classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ConfigDict, Field, model_validator

from .base_model import MAXBaseModel

_RECIPE_PREFIX = "max/pipelines/architectures/"


def _resolve_config_file(config_file: str) -> str:
    """Resolve a config file path, handling built-in recipe prefixes.

    Paths starting with ``max/pipelines/architectures/`` are resolved relative
    to the installed ``max`` package so that users and CI can reference bundled
    recipe YAML files without hard-coding a repo root or relying on the current
    working directory.

    All other paths are returned unchanged (resolved by the caller's cwd as
    before).
    """
    if not config_file.startswith(_RECIPE_PREFIX):
        return config_file

    suffix = config_file[len(_RECIPE_PREFIX) :]
    # Navigate from this file (max/config/config_file_model.py) up to the max
    # package root, then down into pipelines/architectures.  This avoids
    # importing max.pipelines.architectures which would be a circular dep.
    max_pkg_dir = Path(__file__).parent.parent
    resolved = max_pkg_dir / "pipelines" / "architectures" / suffix
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Built-in recipe not found: {config_file} (resolved to {resolved})"
        )
    return str(resolved)


def _deep_merge(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``; ``override`` wins at leaves.

    Nested mappings are merged key by key, so a partial override (e.g. a single
    CLI flag deep in a subtree) keeps the sibling values from ``base`` instead of
    replacing the whole subtree. A shallow ``base | override`` would drop those
    siblings — e.g. ``--load.max-concurrency`` would wipe the rest of the config
    file's ``benchmark`` object.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


class ConfigFileModel(MAXBaseModel):
    """Base class for models that can load configuration from a file.

    This class provides functionality for Pydantic-based config classes to load
    configuration from YAML files. Config classes should inherit from this class
    to enable config file support.

    Example:
        ```python
        from max.config import ConfigFileModel
        from pydantic import Field

        class MyConfig(ConfigFileModel):
            value: int = Field(default=1)

        # Can be used with --config-file config.yaml
        config = MyConfig(config_file="config.yaml")
        ```
    """

    # Config models receive values from text sources (environment variables,
    # CLI arguments, quoted YAML strings) that rely on Pydantic's lax coercion
    # (e.g. "123" -> int, "1.5" -> float).  Override the strict=True default
    # inherited from MAXBaseModel so these conversions continue to work.
    model_config = ConfigDict(strict=False)

    config_file: str | None = None
    """Path to the configuration file."""

    # TODO: This is very similar to _config_file_section_name in MAXConfig.
    # We'll deprecate that and use this instead in the future once we've fully
    # migrated to cyclopts for our CLI bindings.
    section_name: str | None = Field(default=None, exclude=True)
    """Optional section name for comprehensive/multi-section config files.

    If not provided, values are loaded from the YAML top-level (treating the file
    as an "individual config" file).
    """

    @model_validator(mode="before")
    @classmethod
    def load_config_file(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Load configuration from YAML file if config_file is provided.

        This validator runs before Pydantic validation. Cyclopts processes config
        sources in order: CLI args are parsed first, then env vars (from Env config
        source) are applied. When this validator runs, `data` already contains CLI
        args and env vars merged together.

        To achieve the correct precedence (CLI > Config File > Env Vars > Defaults),
        we need to separate CLI args from env vars. However, since cyclopts merges
        them before validation, we approximate by:
        1. Loading config file values
        2. Merging with data (CLI args + env vars), where data takes precedence

        This results in: CLI args > Env vars > Config file > Defaults

        Note: The README documents the desired precedence, but due to cyclopts'
        architecture, config files cannot override env vars while still allowing
        CLI args to override everything. The actual precedence is:
        1. CLI arguments (highest)
        2. Environment variables
        3. Config files
        4. Defaults (lowest)

        Args:
            data: Dictionary of data to validate, may contain 'config_file' key.
                  This dict already contains CLI args and env vars merged by cyclopts.
        Returns:
            Dictionary with config file values merged in if config_file was provided.
        """
        if (config_file := data.get("config_file")) is not None:
            config_file = _resolve_config_file(config_file)
            with open(config_file) as f:
                loaded_data = yaml.safe_load(f) or {}

            if not isinstance(loaded_data, dict):
                raise TypeError(
                    f"Configuration file must contain a mapping at the root, got {type(loaded_data)}"
                )

            explicit_section_name = data.get("section_name")
            if explicit_section_name is not None:
                # Caller explicitly requested a section.
                section_data = loaded_data.get(explicit_section_name)
                if section_data is None:
                    available_sections = [
                        key
                        for key, value in loaded_data.items()
                        if isinstance(value, dict)
                    ]
                    raise KeyError(
                        f"Section '{explicit_section_name}' not found in configuration file. "
                        f"Available sections: {available_sections}."
                    )
                if not isinstance(section_data, dict):
                    raise TypeError(
                        f"Section '{explicit_section_name}' must be a mapping, got {type(section_data)}"
                    )
                loaded_data = section_data

            # Merge: config file values are loaded, then overridden by CLI args +
            # env vars. The merge is recursive so a partial CLI override (e.g. one
            # flag under a nested subtree) keeps the config file's sibling values
            # instead of replacing the whole subtree.
            # Note: Due to cyclopts processing order, env vars override config files.
            data = _deep_merge(loaded_data, data)
        return data
