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


"""Operations on the role-to-config mapping behind a model manifest.

``PipelineConfig.from_args`` stages models in a plain dict while it
resolves them, and builds the immutable :class:`ModelManifest` once at
the end. These helpers work on that mapping, so both the staging window
and the manifest itself can use them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from max.pipelines.lib.config.model_config import (
    MAXModelConfig,
    _build_model_config,
)

_WEIGHT_IDENTITY_FIELDS = frozenset({"model_path", "weight_path"})


def updated_component(
    base: MAXModelConfig, **field_overrides: Any
) -> MAXModelConfig:
    """Returns ``base`` with the given field overrides applied.

    Overriding ``model_path``/``weight_path`` rebuilds the config so the
    weight-path identity re-resolves (a copy would keep the stale derived
    identity, and an external org/repo/file path would 404); other
    overrides use ``model_copy``. Loaded seeds carry over unless their
    source field changed.
    """
    if _WEIGHT_IDENTITY_FIELDS.isdisjoint(field_overrides):
        return base.model_copy(update=field_overrides)
    data = {**base.__dict__, **field_overrides}
    if "model_path" not in field_overrides:
        data["_huggingface_config"] = getattr(base, "_huggingface_config", None)
    if "weight_path" not in field_overrides:
        data["_weights_repo_id"] = getattr(base, "_weights_repo_id", None)
    return _build_model_config(type(base), **data)


def architecture_name_for(
    models: Mapping[str, MAXModelConfig], metadata: Mapping[str, Any]
) -> str:
    """Returns the main architecture class name for a models mapping.

    Non-diffusion mappings (a ``"main"`` key) read the HuggingFace
    config's ``architectures[0]``; diffusion mappings read
    ``metadata["_class_name"]``.

    Raises:
        ValueError: If the architecture name cannot be determined.
    """
    if "main" in models:
        arch_name = models["main"].architecture_name
        if arch_name:
            return arch_name
        raise ValueError(
            f"Cannot determine architecture name for main model "
            f"{models['main'].model_path!r}: HuggingFace config has "
            f"no 'architectures' field."
        )
    if not models:
        raise ValueError(
            "Cannot determine architecture name: manifest is empty."
        )
    class_name = metadata.get("_class_name")
    if class_name:
        return class_name
    any_config = next(iter(models.values()))
    raise ValueError(
        f"Cannot determine architecture name for diffusion model "
        f"{any_config.model_path!r}: metadata has no "
        f"'_class_name' field."
    )
