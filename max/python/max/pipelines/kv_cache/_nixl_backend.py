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

"""The one validator for ``MODULAR_NIXL_TRANSFER_BACKEND`` values.

Shared by every consumer of the variable — the KV transfer engine and the dKV
connector — so a typo fails loudly at the first read with the accepted set,
under one rule, no matter which consumer reads it first. Only validation is
shared: the default for an *unset* variable belongs to each caller (the
transfer engine assumes ``"ucx"``, the dKV connector auto-selects).

A leaf, stdlib-only module for the same reason as ``_nixl_plugin_deps``: the
dKV connector cannot depend on the full ``kv_cache`` library without forming
a cycle.
"""

from __future__ import annotations

from typing import Literal

NixlBackendType = Literal["ucx", "libfabric", "uccl"]

NIXL_BACKEND_ENV_VAR = "MODULAR_NIXL_TRANSFER_BACKEND"

# Must stay in lockstep with `Backend::ALL` in dkv/dkv-memxfer/src/config.rs:
# on the dKV connector path this validator runs in front of the Rust parse,
# so a transport listed there but not here fails model load even though Rust
# accepts it — with an error naming this (stale) set. dkv-connector-pyo3's
# test suite asserts the extension accepts every name listed here.
SUPPORTED_NIXL_BACKENDS: set[NixlBackendType] = {"ucx", "libfabric", "uccl"}


def validate_nixl_backend(raw: str) -> NixlBackendType:
    """Validates a NIXL transfer backend name against the supported set.

    Case-insensitive: the Modular-facing convention is lowercase, but NIXL's
    own plugin names are uppercase and deployments set both spellings.

    ``auto`` is deliberately not accepted: it is the auto-select *sentinel*,
    not a backend, and stripping it out is each reader's job. The dKV
    connector maps it to ``None`` before validating (mirroring the Rust
    ``BackendSelection`` parse), while the transfer engine has no auto mode,
    so for it ``auto`` is as unsupported as any typo.

    Args:
        raw: The backend name to validate, as read from the environment.

    Returns:
        The matching supported backend, normalized to lowercase.

    Raises:
        ValueError: If ``raw`` is not a supported backend, naming the value
            and the accepted set.
    """
    normalized = raw.strip().lower()
    for backend in SUPPORTED_NIXL_BACKENDS:
        if backend == normalized:
            return backend
    raise ValueError(
        f"Unsupported NIXL transfer backend {raw!r} "
        f"(set via {NIXL_BACKEND_ENV_VAR}). "
        f"Supported backends: {sorted(SUPPORTED_NIXL_BACKENDS)}"
    )
