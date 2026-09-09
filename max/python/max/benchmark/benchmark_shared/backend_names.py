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

"""Canonical inference backend / framework name literals for benchmarks.

These name strings are the single cross-package source of truth for the
benchmarking stack: the serving load generator
(:mod:`max.benchmark.benchmark_shared.config`) and the config generator's
manifest catalogs (:mod:`config_generator.catalogs.frameworks`) both import
them from here instead of re-declaring the list.

This module intentionally imports nothing beyond :mod:`typing` so that
lightweight, MAX-free consumers (such as the config generator's ``:catalogs``
library) can depend on it without pulling in the heavy ``benchmark_shared``
transitive deps -- mirroring the ``percentile_metrics`` carve-out in this same
package.

The offline benchmark engine deliberately keeps its own three-member
``Backend`` enum in ``utils/benchmarking/engine/shared/config.py``: that module
is copied verbatim into the isolated vLLM/SGLang subprocess venvs, which have
no ``max`` package installed and therefore cannot import anything from here.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

# Locally launched inference frameworks that the config generator fans
# manifests out over.
FrameworkName = Literal["atom", "mach", "modular", "sglang", "trtllm", "vllm"]

# Server backends the load generator can target. Superset of ``FrameworkName``
# that adds the externally managed ``mcloud`` endpoint (not a locally launched
# framework). Composed from ``FrameworkName`` so the framework names are spelled
# exactly once.
BaseBackend = FrameworkName | Literal["mcloud"]

# Full backend selector: every ``BaseBackend`` plus the ``-chat`` endpoint
# variants routed to the load generator's chat-completions drivers. ``mach`` and
# ``mcloud`` have no ``-chat`` variant. Composed from ``BaseBackend`` so no name
# is repeated.
Backend = (
    BaseBackend
    | Literal[
        "atom-chat",
        "modular-chat",
        "sglang-chat",
        "trtllm-chat",
        "vllm-chat",
    ]
)


class BackendEnum(str, Enum):
    """Enum form of the benchmark backend names.

    The ``str``-enum counterpart to the :data:`Backend` / :data:`BaseBackend` /
    :data:`FrameworkName` literals, for call sites that need a runtime value to
    branch on or serialize rather than a ``Literal`` used only for validation. It
    lives here (not in the engine) so the serving stack can share the same
    vocabulary. The offline benchmark engine uses it for backend dispatch and
    subprocess IPC.

    Members mirror :data:`FrameworkName`; the ``mcloud`` server (externally
    managed, nothing to launch) is omitted. The engine currently launches
    ``modular`` in-process and ``vllm`` / ``sglang`` in subprocess venvs; the
    remaining members round out the vocabulary and raise a clear error until a
    runner is wired up.
    """

    MODULAR = "modular"
    VLLM = "vllm"
    SGLANG = "sglang"
    ATOM = "atom"
    MACH = "mach"
    TRTLLM = "trtllm"
