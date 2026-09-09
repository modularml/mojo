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
"""Tests for PrintHook on v3 (`max.experimental.nn`) modules."""

from __future__ import annotations

import pytest
from max.experimental import random
from max.experimental.nn.module import Module, module_dataclass
from max.experimental.tensor import Tensor, TensorType, defaults
from max.nn.hooks.print_hook import PrintHook


@module_dataclass
class InnerModule(Module[[Tensor], Tensor]):
    def forward(self, x: Tensor) -> Tensor:
        return x * 2


@module_dataclass
class OuterModule(Module[[Tensor], Tensor]):
    inner_1: InnerModule
    inner_2: InnerModule

    def forward(self, x: Tensor) -> Tensor:
        return self.inner_2(self.inner_1(x))


def _build_model() -> OuterModule:
    return OuterModule(inner_1=InnerModule(), inner_2=InnerModule())


def test_named_print_hook_v3(
    capfd: pytest.CaptureFixture,  # type: ignore[type-arg]
) -> None:
    """`name_layers` dispatches to the v3 path and the hook fires per module."""
    print_hook = PrintHook()
    model = _build_model()

    # `name_layers` detects a v3 Module, names every layer, and wraps each
    # module's `forward` so the hook fires during tracing.
    print_hook.name_layers(model)

    dtype, device = defaults()
    input_type = TensorType(dtype, [3, 3], device=device)
    compiled = model.compile(input_type)

    # The print ops are emitted when the compiled graph runs.
    compiled(random.uniform([3, 3]))

    print_hook.remove()
    del print_hook  # Trigger print_hook.summarize()

    # The printed tensors use the qualified names created by `name_layers`.
    captured = capfd.readouterr()
    assert "model-input" in captured.out
    assert "model-output" in captured.out
    assert "model.inner_1-input" in captured.out
    assert "model.inner_1-output" in captured.out
    assert "model.inner_2-input" in captured.out
    assert "model.inner_2-output" in captured.out


def test_print_hook_v3_remove_restores_forward() -> None:
    """`remove` drops the per-instance `forward`, restoring the class method."""
    print_hook = PrintHook()
    model = _build_model()

    print_hook.name_layers(model)
    # Every module gets an instance-level `forward` shadowing the class method.
    assert "forward" in vars(model)
    assert "forward" in vars(model.inner_1)
    assert "forward" in vars(model.inner_2)

    print_hook.remove()

    # The shadow is gone, so attribute lookup falls back to the class method.
    assert "forward" not in vars(model)
    assert "forward" not in vars(model.inner_1)
    assert "forward" not in vars(model.inner_2)

    # The module still computes correctly after restoration.
    result = model(Tensor.ones([2, 2]))
    assert result.shape == [2, 2]


def test_name_layers_double_call_does_not_double_wrap() -> None:
    """Calling `name_layers` twice wraps each module's `forward` only once."""
    print_hook = PrintHook()
    model = _build_model()

    print_hook.name_layers(model)
    wrapped_forward = vars(model.inner_1)["forward"]

    print_hook.name_layers(model)
    # Idempotent: the second pass must not re-wrap an already-wrapped module.
    assert vars(model.inner_1)["forward"] is wrapped_forward

    print_hook.remove()


def test_name_layers_rejects_unsupported_type() -> None:
    """`name_layers` rejects inputs that are neither v2 Layer nor v3 Module."""
    print_hook = PrintHook()
    with pytest.raises(TypeError, match=r"V2.*Layer.*V3.*Module"):
        print_hook.name_layers(object())  # type: ignore[arg-type]
    print_hook.remove()
