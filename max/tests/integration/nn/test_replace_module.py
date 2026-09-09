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

"""Tests for ``Module.replace_module``."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.layer import Module
from max.nn.linear import Linear

_DEVICE = DeviceRef.CPU()


class _Parent(Module):
    def __init__(self, child: Module) -> None:
        super().__init__()
        self.child = child

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError


def _linear() -> Linear:
    return Linear(4, 4, dtype=DType.float32, device=_DEVICE)


def test_replace_module_swaps_registered_child() -> None:
    parent = _Parent(_linear())
    new = _linear()
    parent.replace_module("child", new)
    assert parent.child is new
    assert parent.sublayers["child"] is new


def test_replace_module_unknown_name_raises() -> None:
    parent = _Parent(_linear())
    with pytest.raises(KeyError):
        parent.replace_module("missing", _linear())
