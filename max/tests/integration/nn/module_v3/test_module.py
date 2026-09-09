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
"""Tests for max.nn.Module."""

from __future__ import annotations

import logging
import re
import weakref

import pytest
from max import driver
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random
from max.experimental.nn._compile_utils import (
    prepare_weight_for_parameter,
    prepare_weights_registry,
)
from max.experimental.nn.module import (
    Module,
    module_dataclass,
)
from max.experimental.sharding import (
    DeviceMesh,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.sharding.types import DistributedTensorType
from max.experimental.tensor import Tensor, TensorType, defaults
from max.experimental.testing import assert_all_close


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


@module_dataclass
class SuperModule(Module[[Tensor], Tensor]):
    mod: TestModule


@pytest.fixture
def test_module():  # noqa: ANN201
    return TestModule(
        a=Tensor(1),
        sub=SubModule(b=Tensor(2)),
    )


@pytest.fixture
def lazy_test_module():  # noqa: ANN201
    with F.lazy():
        return TestModule(
            a=Tensor(1),
            sub=SubModule(b=Tensor(2)),
        )


@pytest.fixture
def super_module(test_module: TestModule):  # noqa: ANN201
    return SuperModule(mod=test_module)


def test_module_dataclass() -> None:
    @module_dataclass
    class Test(Module[..., None]):
        a: int
        b: int = 0

    assert repr(Test(2)) == "Test(a=2)"
    assert repr(Test(1, 3)) == "Test(a=1, b=3)"


def test_module_repr(test_module: TestModule) -> None:
    assert "TestModule" in repr(test_module)
    assert "SubModule" in repr(test_module)
    assert "a=Tensor" in repr(test_module)
    assert "b=Tensor" in repr(test_module)
    # eps is the default value, shouldn't be present
    assert "eps=" not in repr(test_module)

    sub = SubModule(b=Tensor(2), eps=1e-6)

    assert "SubModule" in repr(sub)
    assert "b=Tensor" in repr(sub)
    assert "eps=" in repr(sub)


def test_module_custom_repr() -> None:
    class Linear(Module[..., None]):
        weight: Tensor
        bias: Tensor | int

        def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
            self.weight = Tensor.zeros([out_dim, in_dim])
            self.bias = Tensor.zeros([out_dim]) if bias else 0

        def __rich_repr__(self):
            out_dim, in_dim = self.weight.shape
            bias = isinstance(self.bias, Tensor)
            yield "in_dim", in_dim
            yield "out_dim", out_dim
            yield "bias", bias, True

    l1 = Linear(2, 2)
    assert repr(l1) == "Linear(in_dim=Dim(2), out_dim=Dim(2))"

    l2 = Linear(3, 1, bias=False)
    assert repr(l2) == "Linear(in_dim=Dim(3), out_dim=Dim(1), bias=False)"


def test_module_decomposition(test_module: TestModule) -> None:
    test_module_2 = TestModule(a=Tensor(1), sub=test_module.sub)
    assert test_module_2.sub is test_module.sub
    assert dict(test_module_2.children) == dict(test_module.children)


def test_module_decomposition_call(test_module: TestModule) -> None:
    x = Tensor(1)
    assert test_module.sub.b.item() == 2
    assert test_module.sub(x).item() == 3


def test_module_forward(test_module: TestModule) -> None:
    x = Tensor(1)
    # __call__ invokes forward, so both should produce the same result
    assert test_module.forward(x).item() == test_module(x).item()


def test_module_local_parameters(test_module: TestModule) -> None:
    assert dict(test_module.local_parameters) == {"a": test_module.a}
    assert dict(test_module.sub.local_parameters) == {"b": test_module.sub.b}


def test_module_parameters(test_module: TestModule) -> None:
    assert dict(test_module.parameters) == {
        "a": test_module.a,
        "sub.b": test_module.sub.b,
    }

    assert dict(test_module.sub.parameters) == {"b": test_module.sub.b}


def test_module_children(
    test_module: TestModule, super_module: SuperModule
) -> None:
    assert dict(super_module.children) == {"mod": test_module}
    assert dict(test_module.children) == {"sub": test_module.sub}
    assert dict(test_module.sub.children) == {}


def test_module_descendants(
    test_module: TestModule, super_module: SuperModule
) -> None:
    assert super_module.mod is test_module
    assert dict(super_module.descendants) == {
        "mod": test_module,
        "mod.sub": test_module.sub,
    }
    assert dict(super_module.mod.descendants) == {"sub": super_module.mod.sub}
    assert dict(test_module.sub.descendants) == {}


def test_apply_to_local_parameters(test_module: TestModule) -> None:
    a = test_module.a
    b = test_module.sub.b

    test_module.apply_to_local_parameters(lambda _, t: t + 1)
    # Applied to a
    assert test_module.a.item() == (a + 1).item()
    # Not applied to submodule
    assert test_module.sub.b.item() == b.item()


def test_apply_to_parameters(test_module: TestModule) -> None:
    a = test_module.a
    b = test_module.sub.b

    test_module.apply_to_parameters(lambda _, t: t + 1)
    # Applied to a
    assert test_module.a.item() == (a + 1).item()
    # Also applied to submodule
    assert test_module.sub.b.item() == (b + 1).item()


def test_apply_to_parameters__qualified_names(test_module: TestModule) -> None:
    names = set()
    expected = dict(test_module.parameters).keys()

    def lookup(name: str, tensor: Tensor):  # noqa: ANN202
        names.add(name)
        return tensor

    test_module.apply_to_parameters(lookup)
    assert expected == names


def test_map_parameters(test_module: TestModule) -> None:
    a = test_module.a
    b = test_module.sub.b

    m2 = test_module.map_parameters(lambda _, t: t + 1)
    # Test parameters were mapped
    assert m2.a.item() == (a + 1).item()
    assert m2.sub.b.item() == (b + 1).item()
    # Not updated in the original module
    assert test_module.a.item() == a.item()
    assert test_module.sub.b.item() == b.item()


def test_load_state_simple_dict(test_module: TestModule) -> None:
    weights = {
        "a": Tensor(5),
        "sub.b": Tensor(6),
    }
    test_module.load_state(lambda name, _: weights[name])
    assert test_module.a.item() == 5
    assert test_module.sub.b.item() == 6


def test_load_state_simple_dict_lookup_failure(test_module: TestModule) -> None:
    weights: dict[str, Tensor] = {}
    # No guarantee on the resulting state here!
    with pytest.raises(KeyError):
        test_module.load_state(lambda name, _: weights[name])


def test_load_state_name_remapping(test_module: TestModule) -> None:
    def remap_name(name: str):  # noqa: ANN202
        name = re.sub(r"\bsub\.", "feed_forward.", name)
        return name

    weights = {
        "a": Tensor(5),
        "feed_forward.b": Tensor(6),
    }

    test_module.load_state(lambda name, _: weights[remap_name(name)])
    assert test_module.a.item() == 5
    assert test_module.sub.b.item() == 6


def test_load_state_dict(test_module: TestModule) -> None:
    weights = {
        "a": Tensor(5),
        "sub.b": Tensor(6),
    }
    test_module.load_state_dict(weights)
    assert test_module.a.item() == 5
    assert test_module.sub.b.item() == 6


def test_load_state_dict_strict(test_module: TestModule) -> None:
    weights = {
        "a": Tensor(5),
        "sub.b": Tensor(6),
        "extra": Tensor(7),
    }
    with pytest.raises(ValueError):
        test_module.load_state_dict(weights)


def test_load_state_dict_nonstrict(test_module: TestModule) -> None:
    weights = {
        "a": Tensor(5),
        "sub.b": Tensor(6),
        "extra": Tensor(7),
    }
    test_module.load_state_dict(weights, strict=False)
    assert test_module.a.item() == 5
    assert test_module.sub.b.item() == 6


def test_load_state_dict_dtype_mismatch() -> None:
    """Test that load_state_dict raises ValueError for dtype mismatch."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    # Create module with float32 weight
    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))

    # Try to load int32 weights - should fail
    weights = {"weight": Tensor.zeros([3, 3], dtype=DType.int32)}

    with pytest.raises(ValueError, match="not assignable"):
        module.load_state_dict(weights)


def test_load_state_dict_shape_mismatch() -> None:
    """Test that load_state_dict raises ValueError for shape mismatch."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    # Create module with [3, 3] weight
    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))

    # Try to load [4, 4] weights - should fail
    weights = {"weight": Tensor.zeros([4, 4], dtype=DType.float32)}

    with pytest.raises(ValueError, match="not assignable"):
        module.load_state_dict(weights)


def test_load_state_dict_dtype_and_shape_mismatch() -> None:
    """Test that load_state_dict raises ValueError when both dtype and shape mismatch."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    # Create module with float32 [3, 3] weight
    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))

    # Try to load int32 [4, 4] weights - should fail
    weights = {"weight": Tensor.zeros([4, 4], dtype=DType.int32)}

    with pytest.raises(ValueError, match="not assignable"):
        module.load_state_dict(weights)


def test_load_state_dict_valid_types() -> None:
    """Test that load_state_dict succeeds when dtype and shape match."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    # Create module with float32 [3, 3] weight initialized to zeros
    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))

    # Load matching weights with ones - should succeed
    weights = {"weight": Tensor.ones([3, 3], dtype=DType.float32)}
    module.load_state_dict(weights)

    # Verify the weights were loaded (first element should be 1.0, not 0.0)
    assert module.weight[0, 0].item() == 1.0


_AUTO_CAST_LOGGER = "max.experimental.nn._compile_utils"


def _auto_cast_logs(
    caplog: pytest.LogCaptureFixture,
) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == _AUTO_CAST_LOGGER and "auto-cast" in r.getMessage()
    ]


def test_load_state_dict_safe_cast_float32_to_bfloat16(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """float32 -> bfloat16 auto-casts with a summary log flagged lossy."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.bfloat16))
    weights = {"weight": Tensor.ones([3, 3], dtype=DType.float32)}

    with caplog.at_level(logging.WARNING, logger=_AUTO_CAST_LOGGER):
        module.load_state_dict(weights, auto_cast=True)

    assert module.weight.dtype == DType.bfloat16
    assert module.weight[0, 0].item() == 1.0
    # Narrowing cast must be flagged so users can tell precision was lost.
    logs = _auto_cast_logs(caplog)
    assert len(logs) == 1
    assert "precision loss" in logs[0].getMessage()


def test_load_state_dict_safe_cast_bfloat16_to_float32(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """bfloat16 -> float32 auto-casts with a summary log (lossless)."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))
    weights = {"weight": Tensor.ones([3, 3], dtype=DType.bfloat16)}

    with caplog.at_level(logging.WARNING, logger=_AUTO_CAST_LOGGER):
        module.load_state_dict(weights, auto_cast=True)

    assert module.weight.dtype == DType.float32
    assert module.weight[0, 0].item() == 1.0
    # Widening cast: must not be flagged as lossy.
    logs = _auto_cast_logs(caplog)
    assert len(logs) == 1
    assert "precision loss" not in logs[0].getMessage()


def test_load_state_dict_no_warning_when_dtypes_match(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Matching-dtype loads must not emit an auto-cast log."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))
    weights = {"weight": Tensor.ones([3, 3], dtype=DType.float32)}

    with caplog.at_level(logging.WARNING, logger=_AUTO_CAST_LOGGER):
        module.load_state_dict(weights)

    assert not _auto_cast_logs(caplog)


def test_load_state_dict_safe_cast_summary_aggregates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A single summary log aggregates counts across all cast parameters."""

    @module_dataclass
    class TwoWeightModule(Module[[Tensor], Tensor]):
        a: Tensor
        b: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.a + self.b

    module = TwoWeightModule(
        a=Tensor.zeros([3, 3], dtype=DType.bfloat16),
        b=Tensor.zeros([3, 3], dtype=DType.bfloat16),
    )
    weights = {
        "a": Tensor.ones([3, 3], dtype=DType.float32),
        "b": Tensor.ones([3, 3], dtype=DType.float32),
    }

    with caplog.at_level(logging.WARNING, logger=_AUTO_CAST_LOGGER):
        module.load_state_dict(weights, auto_cast=True)

    logs = _auto_cast_logs(caplog)
    assert len(logs) == 1
    assert "2 parameter(s)" in logs[0].getMessage()


def test_load_state_dict_default_does_not_auto_cast() -> None:
    """``auto_cast`` defaults to False; safe-cast pairs still raise unless opted in."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.bfloat16))
    weights = {"weight": Tensor.ones([3, 3], dtype=DType.float32)}

    with pytest.raises(ValueError, match="not assignable"):
        module.load_state_dict(weights)


def test_load_state_dict_auto_cast_false_arg_disables(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Passing ``auto_cast=False`` reverts to hard-fail on dtype mismatch."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.bfloat16))
    weights = {"weight": Tensor.ones([3, 3], dtype=DType.float32)}

    with caplog.at_level(logging.WARNING, logger=_AUTO_CAST_LOGGER):
        with pytest.raises(ValueError, match="not assignable"):
            module.load_state_dict(weights, auto_cast=False)
    # No auto-cast log line should fire when the feature is disabled.
    assert not _auto_cast_logs(caplog)


def test_compile_with_weights_auto_cast_false_arg_disables() -> None:
    """Passing ``auto_cast=False`` to ``compile`` raises on dtype mismatch."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.bfloat16))
    _, device = defaults()
    input_type = TensorType(DType.bfloat16, [3, 3], device=device)
    weights = {"weight": Tensor.ones([3, 3], dtype=DType.float32)}

    with pytest.raises(ValueError, match="not assignable"):
        module.compile(input_type, weights=weights, auto_cast=False)


def test_load_state_dict_unwhitelisted_float_dtype_still_raises() -> None:
    """Float dtypes outside the safe set (e.g. float16) still hard-fail."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))
    weights = {"weight": Tensor.zeros([3, 3], dtype=DType.float16)}

    with pytest.raises(ValueError, match="not assignable"):
        module.load_state_dict(weights)


def test_compile(test_module: TestModule) -> None:
    dtype, device = defaults()
    type = TensorType(dtype, ["batch", "n"], device=device)
    compiled = test_module.compile(type)

    input = random.uniform([3, 3])
    result_eager = test_module(input)
    result_compiled = compiled(input)

    assert all((result_eager == result_compiled)._values())


def test_compile_with_weights_shape_mismatch() -> None:
    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))
    dtype, device = defaults()
    type = TensorType(dtype, [3, 3], device=device)
    weights = {
        "weight": Tensor.zeros([4, 4], dtype=DType.float32),
    }

    with pytest.raises(ValueError, match="not assignable"):
        module.compile(type, weights=weights)


def test_compile_with_weights_dtype_mismatch() -> None:
    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))
    dtype, device = defaults()
    type = TensorType(dtype, [3, 3], device=device)
    weights = {
        "weight": Tensor.zeros([3, 3], dtype=DType.int32),
    }

    with pytest.raises(ValueError, match="not assignable"):
        module.compile(type, weights=weights)


def test_compile_with_weights_safe_cast(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """compile(weights=...) auto-casts safe dtypes and logs a summary."""

    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.bfloat16))
    _, device = defaults()
    input_type = TensorType(DType.bfloat16, [3, 3], device=device)
    weights = {"weight": Tensor.ones([3, 3], dtype=DType.float32)}

    with caplog.at_level(logging.WARNING, logger=_AUTO_CAST_LOGGER):
        module.compile(input_type, weights=weights, auto_cast=True)

    logs = _auto_cast_logs(caplog)
    assert len(logs) == 1
    assert "precision loss" in logs[0].getMessage()


def test_compile_with_weights_missing_parameter_raises() -> None:
    @module_dataclass
    class SimpleModule(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.weight

    module = SimpleModule(weight=Tensor.zeros([3, 3], dtype=DType.float32))
    dtype, device = defaults()
    type = TensorType(dtype, [3, 3], device=device)

    with pytest.raises(KeyError, match="is missing"):
        module.compile(type, weights={})


def test_compile_with_weights(lazy_test_module: TestModule) -> None:
    test_module = lazy_test_module
    dtype, device = defaults()
    type = TensorType(dtype, ["batch", "n"], device=device)

    parameters = weakref.WeakValueDictionary(test_module.parameters)

    weights = {
        name: driver.Buffer.zeros(
            [int(d) for d in param.shape], param.dtype, param.device
        )
        for name, param in test_module.parameters
    }

    assert not any(param.real for param in parameters.values())
    assert not any(param.real for _, param in test_module.parameters)

    compiled = test_module.compile(type, weights=weights)

    assert not any(param.real for param in parameters.values())
    assert not any(param.real for _, param in test_module.parameters)

    input = Tensor(storage=driver.Buffer.zeros([3, 3], dtype, device))
    _ = compiled(input)

    assert not any(param.real for param in parameters.values())
    assert not any(param.real for _, param in test_module.parameters)


def test_compile_with_weights_never_realized(
    lazy_test_module: TestModule,
) -> None:
    test_module = lazy_test_module
    dtype, device = defaults()
    type = TensorType(dtype, ["batch", "n"], device=device)

    parameters = weakref.WeakValueDictionary(test_module.parameters)

    weights = {
        name: Tensor.zeros_like(param.type)
        for name, param in test_module.parameters
    }

    assert not any(param.real for param in parameters.values())
    assert not any(param.real for _, param in test_module.parameters)

    compiled = test_module.compile(type, weights=weights)

    assert not any(param.real for param in parameters.values())
    assert not any(param.real for _, param in test_module.parameters)

    input = random.uniform([3, 3])
    _ = compiled(input)

    assert not any(param.real for param in parameters.values())
    assert not any(param.real for _, param in test_module.parameters)


# ---------------------------------------------------------------------------
# Sharding and transferring provided weights inside the compiled graph.
#
# Covers the weight-loading path that shards/transfers a single-device weight
# for a distributed parameter in-graph, and the preserved path where an
# already-sharded weight passes through untouched. A two-way mesh on CPU
# matches the trace shape and numerics of a real two-GPU mesh.
# ---------------------------------------------------------------------------

_F32 = DType.float32
_MESH = DeviceMesh(devices=(CPU(), CPU()), mesh_shape=(2,), axis_names=("tp",))
_REPLICATED = PlacementMapping(_MESH, (Replicated(),))
_COLUMN = PlacementMapping(_MESH, (Sharded(1),))
_ROW = PlacementMapping(_MESH, (Sharded(0),))


def cpu_tensor(*shape: int) -> Tensor:
    return Tensor.zeros(list(shape), dtype=_F32, device=CPU())


def test_prepare_weight_single_device_for_distributed_needs_transfer() -> None:
    """A single-device weight for a distributed parameter defers the transfer.

    The prepared tensor stays single-device (the transfer happens in-graph),
    and ``transfer_needed`` is True.
    """
    param = F.transfer_to(cpu_tensor(4, 8), _COLUMN)
    weight = cpu_tensor(4, 8)

    prepared, cast_record, transfer_needed = prepare_weight_for_parameter(
        "w", weight, param, auto_cast=False
    )

    assert transfer_needed
    assert cast_record is None
    assert not prepared.is_distributed


def test_prepare_weight_matching_distribution_no_transfer() -> None:
    """An already-sharded weight matching the parameter's mapping is untouched."""
    param = F.transfer_to(cpu_tensor(4, 8), _COLUMN)
    weight = F.transfer_to(cpu_tensor(4, 8), _COLUMN)

    prepared, _, transfer_needed = prepare_weight_for_parameter(
        "w", weight, param, auto_cast=False
    )

    assert not transfer_needed
    assert prepared.is_distributed
    assert prepared.mapping == param.mapping


def test_prepare_weight_incompatible_distribution_raises() -> None:
    """A sharded weight whose mapping differs from the parameter's is rejected."""
    param = F.transfer_to(cpu_tensor(4, 8), _COLUMN)
    weight = F.transfer_to(cpu_tensor(4, 8), _ROW)

    with pytest.raises(ValueError, match="incompatible distribution"):
        prepare_weight_for_parameter("w", weight, param, auto_cast=False)


def test_prepare_weight_non_distributed_no_transfer() -> None:
    """A single-device weight for a single-device parameter never transfers."""
    param = cpu_tensor(4, 8)
    weight = cpu_tensor(4, 8)

    prepared, _, transfer_needed = prepare_weight_for_parameter(
        "w", weight, param, auto_cast=False
    )

    assert not transfer_needed
    assert not prepared.is_distributed


def test_load_state_dict_single_device_weight_keeps_distribution() -> None:
    """Loading a single-device weight into a distributed parameter keeps it distributed."""
    module = SubModule(b=F.transfer_to(cpu_tensor(4, 8), _COLUMN))
    assert module.b.is_distributed
    assert module.b.mapping == _COLUMN

    module.load_state_dict({"b": cpu_tensor(4, 8)})

    assert module.b.is_distributed
    assert module.b.mapping == _COLUMN


def test_prepare_weights_registry_registers_plain_name() -> None:
    """The transfer path registers the whole unsharded weight under its name.

    Sharding is deferred to the graph, so the registry holds one plain-named
    entry (not per-shard entries) and the weight is recorded for in-graph
    transfer.
    """
    param = F.transfer_to(cpu_tensor(4, 8), _COLUMN)
    weight = cpu_tensor(4, 8)

    registry, to_transfer = prepare_weights_registry(
        {"w": weight}, [("w", param)], auto_cast=False
    )

    assert set(registry) == {"w"}
    w = registry["w"]
    assert isinstance(w, Tensor)
    assert list(w.shape) == [4, 8]
    assert set(to_transfer) == {"w"}


def test_prepare_weights_registry_presharded_registers_shard_keys() -> None:
    """An already-sharded weight is registered per-shard with no in-graph transfer."""
    param = F.transfer_to(cpu_tensor(4, 8), _COLUMN)
    weight = F.transfer_to(cpu_tensor(4, 8), _COLUMN)

    registry, to_transfer = prepare_weights_registry(
        {"w": weight}, [("w", param)], auto_cast=False
    )

    assert set(registry) == {"w._shard.0", "w._shard.1"}
    assert not to_transfer


def test_prepare_weights_registry_non_distributed_passthrough() -> None:
    """Single-device parameters pass through unchanged with no transfer."""
    param = cpu_tensor(4, 8)
    weight = cpu_tensor(4, 8)

    registry, to_transfer = prepare_weights_registry(
        {"w": weight}, [("w", param)], auto_cast=False
    )

    assert set(registry) == {"w"}
    assert not to_transfer


@module_dataclass
class _ColumnParallelLinear(Module[[Tensor], Tensor]):
    """Column-parallel matmul, all-gathered back to a replicated output."""

    w: Tensor  # [D, H], column-parallel

    def forward(self, x: Tensor) -> Tensor:  # x replicated [batch, D]
        return F.transfer_to(x @ self.w, _REPLICATED)


def test_compile_shards_single_device_weight_in_graph() -> None:
    """A single-device weight provided for a sharded parameter is sharded and
    transferred inside the compiled graph, and produces correct numerics.

    Loading the same weight pre-sharded must produce identical results,
    confirming the in-graph transfer matches the previously-eager behavior.
    """
    random.set_seed(0)
    w = random.normal([4, 8], dtype=_F32, device=CPU())
    x = random.normal([2, 4], dtype=_F32, device=CPU())
    # Eager (non-compile) reference. Flattened to 1-D so `assert_all_close`
    # reduces over every element rather than only the last axis.
    expected = (x @ w).reshape([16])

    input_type = DistributedTensorType(
        _F32, ["batch", 4], _MESH, (Replicated(),)
    )
    replicated_x = F.transfer_to(x, _REPLICATED)

    # Placeholder distributed parameter; the real weight arrives via compile().
    module = _ColumnParallelLinear(w=F.transfer_to(cpu_tensor(4, 8), _COLUMN))

    # Single-device weight: sharded and transferred in-graph. `materialize`
    # gathers the replicated result back to a single local tensor to compare.
    compiled = module.compile(input_type, weights={"w": w})
    single_device = compiled(replicated_x)
    assert single_device.placements == (Replicated(),)
    single_device_local = single_device.materialize().reshape([16])
    assert_all_close(expected, single_device_local, rtol=1e-4, atol=1e-4)

    # Pre-sharded weight: preserved path, must match.
    compiled_presharded = module.compile(
        input_type,
        weights={"w": F.transfer_to(w, _COLUMN)},
    )
    presharded = compiled_presharded(replicated_x)
    presharded_local = presharded.materialize().reshape([16])
    assert_all_close(
        single_device_local, presharded_local, rtol=1e-6, atol=1e-6
    )


# ---------------------------------------------------------------------------
# Module.compile() with custom_extensions
# ---------------------------------------------------------------------------

import os
from pathlib import Path


@pytest.fixture
def kernel_verification_ops_path() -> Path:
    raw = os.environ.get("MODULAR_KERNEL_VERIFICATION_OPS_PATH")
    if raw is None:
        pytest.skip("MODULAR_KERNEL_VERIFICATION_OPS_PATH not set")
    return Path(raw)


def test_compile_with_custom_extensions(
    kernel_verification_ops_path: Path,
) -> None:
    """Module.compile() loads custom kernels so F.custom works during tracing."""

    @module_dataclass
    class CustomAddModule(Module[[Tensor], Tensor]):
        bias: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return F.custom(
                "my_add",
                device=x.device,
                values=[x, self.bias],
                out_types=[x.type],
            )[0]

    device = CPU()
    dtype = DType.float32
    module = CustomAddModule(bias=Tensor.ones([64], dtype=dtype, device=device))
    input_type = TensorType(dtype, [64], device=device)

    compiled = module.compile(
        input_type,
        custom_extensions=[kernel_verification_ops_path],
    )

    x = Tensor.ones([64], dtype=dtype, device=device)
    result = compiled(x)
    assert result.shape == [64]
    assert result.dtype == dtype


@pytest.mark.parametrize(
    ("kernel_name", "parameters"),
    [
        ("op_with_int_parameter", {"IntParameter": 42}),
        ("op_with_static_string_parameter", {"StringParameter": "hello"}),
    ],
    ids=["int_param", "static_string_param"],
)
def test_compile_with_custom_extensions_struct_params(
    kernel_verification_ops_path: Path,
    kernel_name: str,
    parameters: dict[str, int | str],
) -> None:
    """Module.compile() discovers struct-level parameters on custom kernels.

    Without custom_extensions on compile(), struct parameters like
    ``[IntParameter: Int]`` or ``[StringParameter: StaticString]`` were
    not discovered during graph-tracing validation.
    """

    @module_dataclass
    class StructParamModule(Module[[Tensor], Tensor]):
        _kernel_name: str
        _parameters: dict[str, int | str]

        def forward(self, x: Tensor) -> Tensor:
            return F.custom(
                self._kernel_name,
                device=x.device,
                values=[x],
                out_types=[x.type],
                parameters=self._parameters,
            )[0]

    device = CPU()
    dtype = DType.float32
    module = StructParamModule(_kernel_name=kernel_name, _parameters=parameters)
    input_type = TensorType(dtype, [64], device=device)

    compiled = module.compile(
        input_type,
        custom_extensions=[kernel_verification_ops_path],
    )

    x = Tensor.ones([64], dtype=dtype, device=device)
    result = compiled(x)
    assert result.shape == [64]
    assert result.dtype == dtype
