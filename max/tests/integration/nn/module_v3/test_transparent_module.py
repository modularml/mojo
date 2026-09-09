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
"""Tests for :class:`max.experimental.nn.TransparentModule`."""

from __future__ import annotations

import numpy as np
import pytest
from max.dtype import DType
from max.experimental import random
from max.experimental.nn import Linear, Module, TransparentModule
from max.experimental.tensor import Tensor, default_dtype


class _Transparent(TransparentModule[[Tensor], Tensor]):
    """A container that omits its own name from its children's paths."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.a = Linear(in_dim, out_dim)
        self.b = Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.a(x) + self.b(x)


class _Opaque(Module[[Tensor], Tensor]):
    """Same structure, but keeps its name (a plain Module)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.a = Linear(in_dim, out_dim)
        self.b = Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.a(x) + self.b(x)


class _Parent(Module[[Tensor], Tensor]):
    def __init__(self, *, transparent: bool) -> None:
        self.inner = (_Transparent if transparent else _Opaque)(4, 4)
        self.out = Linear(4, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.inner(x))


def test_transparent_child_omits_its_name_from_param_paths() -> None:
    names = {name for name, _ in _Parent(transparent=True).parameters}
    assert names == {
        "a.weight",
        "a.bias",
        "b.weight",
        "b.bias",
        "out.weight",
        "out.bias",
    }


def test_opaque_child_keeps_its_name() -> None:
    names = {name for name, _ in _Parent(transparent=False).parameters}
    assert names == {
        "inner.a.weight",
        "inner.a.bias",
        "inner.b.weight",
        "inner.b.bias",
        "out.weight",
        "out.bias",
    }


def test_transparent_child_still_discoverable_as_descendant() -> None:
    descendants = dict(_Parent(transparent=True).descendants)
    assert "inner" in descendants
    assert "a" in descendants and "b" in descendants
    assert "inner.a" not in descendants


def test_apply_to_parameters_matches_parameters_paths() -> None:
    parent = _Parent(transparent=True)
    visited: set[str] = set()

    def record(name: str, t: Tensor) -> Tensor:
        visited.add(name)
        return t

    parent.apply_to_parameters(record)
    assert visited == {name for name, _ in parent.parameters}


def test_load_state_dict_with_omitted_names() -> None:
    with default_dtype(DType.float32):
        parent = _Parent(transparent=True)
    state = {name: Tensor.zeros(t.shape) for name, t in parent.parameters}
    parent.load_state_dict(state, strict=True)
    for _, t in parent.parameters:
        shape = tuple(int(d) for d in t.shape)
        assert np.array_equal(t.to_numpy(), np.zeros(shape))


class _ExposesA(TransparentModule[[Tensor], Tensor]):
    def __init__(self) -> None:
        self.a = Linear(4, 4)

    def forward(self, x: Tensor) -> Tensor:
        return self.a(x)


class _Conflict(Module[[Tensor], Tensor]):
    """A real child ``a`` alongside a transparent child that also exposes ``a``."""

    def __init__(self) -> None:
        self.a = Linear(4, 4)
        self.passthrough = _ExposesA()

    def forward(self, x: Tensor) -> Tensor:
        return self.a(x) + self.passthrough(x)


def test_colliding_paths_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate parameter path"):
        dict(_Conflict().parameters)


class _LocalParamTransparent(TransparentModule[[Tensor], Tensor]):
    """A transparent module that holds a local parameter of its own."""

    def __init__(self) -> None:
        self.weight = random.normal([4, 4])

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class _LocalParamParent(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        self.inner = _LocalParamTransparent()

    def forward(self, x: Tensor) -> Tensor:
        return self.inner(x)


def test_transparent_module_local_param_drops_its_name() -> None:
    # A transparent module's own local parameter is named at the parent level
    # (its ``inner`` attr dropped); parameters and apply_to_parameters agree.
    parent = _LocalParamParent()
    assert {name for name, _ in parent.parameters} == {"weight"}

    visited: set[str] = set()

    def record(name: str, t: Tensor) -> Tensor:
        visited.add(name)
        return t

    parent.apply_to_parameters(record)
    assert visited == {"weight"}


class _OpaqueTransparent(TransparentModule[[Tensor], Tensor]):
    """A TransparentModule with transparency switched off (opaque)."""

    name_transparent = False

    def __init__(self) -> None:
        self.weight = random.normal([4, 4])

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class _OpaqueParent(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        self.inner = _OpaqueTransparent()

    def forward(self, x: Tensor) -> Tensor:
        return self.inner(x)


def test_transparency_can_be_switched_off() -> None:
    names = {name for name, _ in _OpaqueParent().parameters}
    assert names == {"inner.weight"}
