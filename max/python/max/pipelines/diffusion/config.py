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

"""Denoising-cache settings, resolved config, and architecture defaults.

CLI and ``PipelineArgs`` carry :class:`DenoisingCacheSettings`. Construction
fills that bag from architecture :class:`TaylorSeerDefaults` into a frozen
:class:`DenoisingCacheConfig` on ``PipelineRuntimeConfig``. Runtime cache
execution lives in :mod:`max.pipelines.diffusion.cache`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from max.config import ConfigFileModel
from pydantic import ConfigDict, Field, model_validator

__all__ = [
    "DEFAULT_DENOISING_CACHE_CONFIG",
    "GENERIC_TAYLORSEER_DEFAULTS",
    "DenoisingCacheConfig",
    "DenoisingCacheSettings",
    "TaylorSeerDefaults",
    "resolve_denoising_cache",
]


class DenoisingCacheSettings(ConfigFileModel):
    """User denoising-cache settings. Every field is optional.

    Construction fills unset fields from the architecture's TaylorSeer
    defaults, producing :class:`DenoisingCacheConfig`.
    """

    model_config = ConfigDict(frozen=True)

    first_block_caching: bool | None = Field(
        default=None,
        description=(
            "Enable First-Block Cache (FBCache) for step-cache denoising. "
            "When enabled, the transformer skips remaining blocks if the "
            "first-block residual is similar to the previous step."
        ),
    )

    taylorseer: bool | None = Field(
        default=None,
        description=(
            "Enable TaylorSeer cache optimization. Uses Taylor series "
            "prediction to skip full transformer passes on certain "
            "denoising steps."
        ),
    )

    taylorseer_cache_interval: int | None = Field(
        default=None,
        description=(
            "Steps between full TaylorSeer computations. "
            "None uses the model-specific default (typically 5)."
        ),
    )

    taylorseer_warmup_steps: int | None = Field(
        default=None,
        description=(
            "Number of warmup steps before TaylorSeer prediction begins. "
            "None uses the model-specific default (typically 4)."
        ),
    )

    taylorseer_max_order: int | None = Field(
        default=None,
        description=(
            "Taylor expansion order (1 or 2). Higher order uses second "
            "derivatives for more accurate prediction. "
            "None uses the model-specific default (typically 1)."
        ),
    )

    @model_validator(mode="after")
    def _validate_cache_mode(self) -> DenoisingCacheSettings:
        if self.taylorseer and self.first_block_caching:
            raise ValueError(
                "TaylorSeer and first-block caching are mutually exclusive; "
                "enable only one (--taylorseer OR --first-block-caching)."
            )
        if (
            self.taylorseer_cache_interval is not None
            and self.taylorseer_cache_interval < 1
        ):
            raise ValueError("taylorseer_cache_interval must be >= 1.")
        if (
            self.taylorseer_warmup_steps is not None
            and self.taylorseer_warmup_steps < 1
        ):
            raise ValueError("taylorseer_warmup_steps must be >= 1.")
        if self.taylorseer_max_order is not None and (
            self.taylorseer_max_order not in (1, 2)
        ):
            raise ValueError("taylorseer_max_order must be 1 or 2.")
        return self


class DenoisingCacheConfig(ConfigFileModel):
    """Resolved denoising-cache config. Every field is required.

    Built from :class:`DenoisingCacheSettings` plus architecture defaults.
    """

    model_config = ConfigDict(frozen=True)

    first_block_caching: bool = Field(
        description="Whether First-Block Cache (FBCache) is enabled.",
    )

    taylorseer: bool = Field(
        description="Whether the TaylorSeer cache optimization is enabled.",
    )

    taylorseer_cache_interval: int = Field(
        description="Steps between full TaylorSeer computations.",
    )

    taylorseer_warmup_steps: int = Field(
        description=(
            "Number of warmup steps before TaylorSeer prediction begins."
        ),
    )

    taylorseer_max_order: Literal[1, 2] = Field(
        description=(
            "Taylor expansion order. Higher order uses second derivatives "
            "for more accurate prediction."
        ),
    )

    @model_validator(mode="after")
    def _validate_cache_mode(self) -> DenoisingCacheConfig:
        if self.taylorseer and self.first_block_caching:
            raise ValueError(
                "TaylorSeer and first-block caching are mutually exclusive; "
                "enable only one (--taylorseer OR --first-block-caching)."
            )
        if self.taylorseer_cache_interval < 1:
            raise ValueError("taylorseer_cache_interval must be >= 1.")
        if self.taylorseer_warmup_steps < 1:
            raise ValueError("taylorseer_warmup_steps must be >= 1.")
        return self


DEFAULT_DENOISING_CACHE_CONFIG = DenoisingCacheConfig(
    first_block_caching=False,
    taylorseer=False,
    taylorseer_cache_interval=5,
    taylorseer_warmup_steps=9,
    taylorseer_max_order=1,
)
"""Benign complete default: caching disabled, generic TaylorSeer tuning."""


@dataclass(frozen=True)
class TaylorSeerDefaults:
    """Architecture-declared TaylorSeer tuning. User-set fields always win."""

    cache_interval: int
    warmup_steps: int
    max_order: Literal[1, 2]


GENERIC_TAYLORSEER_DEFAULTS = TaylorSeerDefaults(
    cache_interval=5, warmup_steps=9, max_order=1
)
"""Generic TaylorSeer tuning for architectures without model-specific numbers."""


def resolve_denoising_cache(
    settings: DenoisingCacheSettings,
    defaults: TaylorSeerDefaults | None,
    arch_name: str | None = None,
) -> DenoisingCacheConfig:
    """Fill unset settings from architecture defaults.

    User-set fields win. Unset booleans become ``False``. Remaining unset
    tuning uses the complete default. Enabling TaylorSeer without resolvable
    tuning raises here.

    Args:
        settings: User denoising-cache settings.
        defaults: Architecture TaylorSeer defaults, if any.
        arch_name: Architecture name for error messages.

    Raises:
        ValueError: TaylorSeer is enabled but its tuning cannot be resolved.
    """
    cache_interval = settings.taylorseer_cache_interval
    warmup_steps = settings.taylorseer_warmup_steps
    max_order_value = settings.taylorseer_max_order
    if defaults is not None:
        if cache_interval is None:
            cache_interval = defaults.cache_interval
        if warmup_steps is None:
            warmup_steps = defaults.warmup_steps
        if max_order_value is None:
            max_order_value = defaults.max_order

    unresolved = [
        name
        for name, value in (
            ("taylorseer_cache_interval", cache_interval),
            ("taylorseer_warmup_steps", warmup_steps),
            ("taylorseer_max_order", max_order_value),
        )
        if value is None
    ]
    # First-block caching does not read these knobs.
    if settings.taylorseer and unresolved:
        arch_label = (
            f"architecture {arch_name}" if arch_name else "the architecture"
        )
        raise ValueError(
            f"Cannot enable TaylorSeer: {', '.join(unresolved)} unset and "
            f"{arch_label} declares no TaylorSeer defaults. Set the fields "
            "explicitly."
        )

    benign = DEFAULT_DENOISING_CACHE_CONFIG
    if max_order_value is None:
        max_order_value = benign.taylorseer_max_order
    # Settings and TaylorSeerDefaults both validate membership in {1, 2};
    # the explicit branches narrow the int to the published Literal type.
    max_order: Literal[1, 2]
    if max_order_value == 1:
        max_order = 1
    elif max_order_value == 2:
        max_order = 2
    else:
        raise ValueError("taylorseer_max_order must be 1 or 2.")
    return DenoisingCacheConfig(
        first_block_caching=settings.first_block_caching or False,
        taylorseer=settings.taylorseer or False,
        taylorseer_cache_interval=(
            cache_interval
            if cache_interval is not None
            else benign.taylorseer_cache_interval
        ),
        taylorseer_warmup_steps=(
            warmup_steps
            if warmup_steps is not None
            else benign.taylorseer_warmup_steps
        ),
        taylorseer_max_order=max_order,
    )
