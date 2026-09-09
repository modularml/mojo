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
"""Pipeline-side helpers for loading weights into experimental Modules.

The framework (``max.experimental.nn``) exposes auto-casting as an explicit
``auto_cast`` argument. Pipelines that want to defer the decision to a
deployment-time env var route through these helpers so the framework stays
env-agnostic.
"""

from __future__ import annotations

__all__ = ["AUTO_CAST_ENV_VAR", "auto_cast_weights_from_env"]

import os

AUTO_CAST_ENV_VAR = "MODULAR_AUTO_CAST_WEIGHTS"


def auto_cast_weights_from_env() -> bool:
    """Returns the pipeline default for ``Module`` auto-cast, read from env.

    Reads ``MODULAR_AUTO_CAST_WEIGHTS``; defaults to ``True`` when unset.
    Accepted values are ``true``/``false``/``1``/``0``/``yes``/``no``/
    ``on``/``off`` (case-insensitive). Any other value raises
    :class:`ValueError` so a typo surfaces immediately rather than silently
    falling back to a default.
    """
    raw = os.environ.get(AUTO_CAST_ENV_VAR, "true").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"{AUTO_CAST_ENV_VAR} must be a boolean "
        f"(true/false/1/0/yes/no/on/off); got {raw!r}."
    )
