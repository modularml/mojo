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

"""Unit tests for the shared ``MODULAR_NIXL_TRANSFER_BACKEND`` validator.

One validator serves both consumers of the variable (the KV transfer engine
and the dKV connector), so these tests pin the accepted set, the
normalization, and the error shape once for both.
"""

from __future__ import annotations

import pytest
from max.pipelines.kv_cache._nixl_backend import (
    NIXL_BACKEND_ENV_VAR,
    SUPPORTED_NIXL_BACKENDS,
    validate_nixl_backend,
)
from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
    _get_nixl_backend_type,
)


@pytest.mark.parametrize("backend", sorted(SUPPORTED_NIXL_BACKENDS))
def test_supported_backends_validate_in_either_case(backend: str) -> None:
    """Both live spellings normalize to the lowercase Modular-facing name.

    Uppercase is not hypothetical: NIXL's own plugin names are uppercase and
    ab-bench deploys ``UCX``, the exact shape behind the CLIN-1722 outage.
    """
    assert validate_nixl_backend(backend) == backend
    assert validate_nixl_backend(backend.upper()) == backend


def test_surrounding_whitespace_is_stripped() -> None:
    assert validate_nixl_backend(" ucx\n") == "ucx"


def test_unknown_backend_fails_naming_the_variable_and_accepted_set() -> None:
    """A typo names the value, the variable to fix, and what it may be."""
    with pytest.raises(ValueError) as excinfo:
        validate_nixl_backend("ucx_cuda")

    message = str(excinfo.value)
    assert "ucx_cuda" in message
    assert NIXL_BACKEND_ENV_VAR in message
    assert "libfabric" in message and "uccl" in message


def test_auto_is_not_a_backend() -> None:
    """``auto`` stays rejected here: the validator names backends only.

    The auto-select *sentinel* is each reader's concern — the dKV connector
    maps it to ``None`` before validating, and the transfer engine has no
    auto mode at all, so for it ``auto`` is as wrong as any typo.
    """
    with pytest.raises(ValueError, match="auto"):
        validate_nixl_backend("auto")


def test_transfer_engine_defaults_to_ucx_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine's read-with-default keeps its own semantics.

    Only validation is shared: the transfer engine assumes ``"ucx"`` when the
    variable is unset, while the dKV connector maps unset to ``None``
    (auto-select) — see ``_nixl_backend_override`` in its connector module.
    """
    monkeypatch.delenv(NIXL_BACKEND_ENV_VAR, raising=False)

    assert _get_nixl_backend_type() == "ucx"
