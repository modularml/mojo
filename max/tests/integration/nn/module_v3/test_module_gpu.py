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
"""Accelerator-only tests for max.nn.Module (device placement and `to`)."""

from __future__ import annotations

import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental.nn.module import (
    Module,
    PinnedDeviceTensor,
    module_dataclass,
)
from max.experimental.tensor import Tensor


@module_dataclass
class SubModule(Module[[Tensor], Tensor]):
    b: Tensor
    eps: float = 1e-5

    def forward(self, x: Tensor) -> Tensor:
        return x + self.b


@module_dataclass
class TestModule(Module[[Tensor], Tensor]):
    a: Tensor
    sub: SubModule

    def forward(self, x: Tensor) -> Tensor:
        return self.sub(x) + self.a


@pytest.mark.skipif(not accelerator_count(), reason="requires accelerator")
def test_to() -> None:
    module = TestModule(a=Tensor(1), sub=SubModule(b=Tensor(2)))
    assert all(t.device == Accelerator() for _, t in module.parameters)
    assert module.to(CPU()) is module
    assert all(t.device == CPU() for _, t in module.parameters)


@pytest.mark.skipif(not accelerator_count(), reason="requires accelerator")
def test_pinned_device_tensor_unchanged_by_to() -> None:
    """`PinnedDeviceTensor` fields are not moved by `Module.to`."""

    @module_dataclass
    class ScaledModule(Module[[Tensor], Tensor]):
        weight: Tensor
        scale: PinnedDeviceTensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = ScaledModule(
        weight=Tensor.ones([3, 3]),
        scale=Tensor.full([], 1.0, dtype=DType.float32),
    )
    original_scale_device = module.scale.device
    module.to(Accelerator())

    assert module.weight.device == Accelerator()
    assert module.scale.device == original_scale_device


@pytest.mark.skipif(not accelerator_count(), reason="requires accelerator")
def test_pinned_device_tensor_in_child_module() -> None:
    """`PinnedDeviceTensor` fields in child modules are not moved by `Module.to`."""

    @module_dataclass
    class Inner(Module[[Tensor], Tensor]):
        weight: Tensor
        scale: PinnedDeviceTensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    @module_dataclass
    class Outer(Module[[Tensor], Tensor]):
        inner: Inner

        def forward(self, x: Tensor) -> Tensor:
            return self.inner(x)

    module = Outer(
        inner=Inner(
            weight=Tensor.ones([3, 3]),
            scale=Tensor.full([], 2.0, dtype=DType.float32),
        )
    )
    original_scale_device = module.inner.scale.device
    module.to(Accelerator())

    assert module.inner.weight.device == Accelerator()
    assert module.inner.scale.device == original_scale_device


@pytest.mark.skipif(not accelerator_count(), reason="requires accelerator")
def test_pinned_device_tensor_inherited() -> None:
    """`PinnedDeviceTensor` annotations are respected through inheritance."""

    @module_dataclass
    class Base(Module[[Tensor], Tensor]):
        weight: Tensor
        scale: PinnedDeviceTensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    @module_dataclass
    class Child(Base):
        pass

    module = Child(
        weight=Tensor.ones([3, 3]),
        scale=Tensor.full([], 1.0, dtype=DType.float32),
    )
    original_scale_device = module.scale.device
    module.to(Accelerator())

    assert module.weight.device == Accelerator()
    assert module.scale.device == original_scale_device
