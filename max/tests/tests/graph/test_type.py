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
"""Tests type factories and accessors."""

import re

import pytest
from conftest import static_dims
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from max import mlir
from max._core.dialects import builtin, m, mo, mosh
from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Dim,
    Graph,
    StaticDim,
    SymbolicDim,
    TensorType,
    _ChainType,
    _OpaqueType,
)
from max.graph.type import FilterLayout, Type


@given(dim=...)
def test_static_dim(dim: int) -> None:
    assume(-(2**63) <= dim < 2**63)
    assert StaticDim(dim).dim == dim


@given(i=...)
def test_static_dim__equals_dim_value(i: int) -> None:
    assume(-(2**63) <= i < 2**63)
    dim = StaticDim(i)
    assert isinstance(dim, Dim)
    assert dim == i
    assert dim == dim  # noqa: PLR0124


@given(i=...)
def test_static_dim__compares_to_dim_value(i: int) -> None:
    assume(-(2**63) <= i < 2**63)
    dim = StaticDim(i)
    assert isinstance(dim, Dim)
    assert i <= dim < i + 1


@given(dim=st.integers(min_value=2**63))
def test_static_dim_too_big(dim: int) -> None:
    with pytest.raises(ValueError):
        StaticDim(dim)


@given(numerator=static_dims())
def test_static_dim__division_by_zero(numerator: StaticDim) -> None:
    with pytest.raises(ZeroDivisionError):
        _ = numerator // 0


def test_algebraic_dim_simplify_and_comparison(
    mlir_context: mlir.Context,
) -> None:
    assert 4 * Dim("x") + 4 == (Dim("x") + 1) * 4
    assert 4 * Dim("x") // 5 != Dim(4) // 5 * "x"
    assert 0 == Dim(4) // 5 * "x"
    assert -Dim("x") - 4 == -(Dim("x") + 4)


def test_dims_print_reasonably(mlir_context: mlir.Context) -> None:
    assert str(Dim(23)) == "23"
    assert str(Dim("test")) == "test"
    assert str((Dim("x") + "y" - 4) // 5) == "(x + y + -4) // 5"

    assert repr(Dim(23)) == "Dim(23)"
    assert repr(Dim("test")) == "Dim('test')"
    assert (
        repr((Dim("x") + "y" - 4) // 5)
        == "(Dim('x') + Dim('y') + Dim(-4)) // Dim(5)"
    )


# TODO(MSDK-695): less restrictive dim names
@given(
    name=st.text(
        alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z"))
    )
)
def test_symbolic_dim(name: str) -> None:
    assume(name != "")
    dim = SymbolicDim(name)
    assert isinstance(dim, Dim)
    assert dim == name
    assert dim == dim  # noqa: PLR0124


# TODO(MSDK-695): less restrictive dim names
@given(name=st.text())
def test_symbolic_dim_invalid(name: str) -> None:
    assume(not re.match(r"^[a-zA-Z_]\w*$", name))
    with pytest.raises(ValueError):
        SymbolicDim(name)


def test_symbolic_dim_to_int_error() -> None:
    """Checks the error message when creating an int from a SymbolicDim."""
    with pytest.raises(
        TypeError, match="conversions only supported for static dims"
    ):
        int(SymbolicDim("x"))


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(tensor_type=...)
def test_tensor_type_to_mlir(
    mlir_context: mlir.Context, tensor_type: TensorType
) -> None:
    assert tensor_type == TensorType.from_mlir(tensor_type.to_mlir())


def test_tensor_type(mlir_context: mlir.Context) -> None:
    """Tests tensor type creation."""
    t = TensorType(DType.float32, [3, "x"], DeviceRef.CPU())
    assert t == Type.from_mlir(t.to_mlir())
    assert t.dtype == DType.float32
    assert t.shape == [Dim(3), Dim("x")]
    assert t.device == DeviceRef.CPU()

    tt = t.to_mlir()
    assert tt == mo.TensorType(
        mosh.ShapeAttr(
            [Dim(3).to_mlir(), Dim("x").to_mlir()],
            mosh.ShapeType(),
        ),
        DType.float32,
        m.DeviceRefAttr("cpu", 0),
    )


def test_tensor_type__negative_dim(mlir_context: mlir.Context) -> None:
    with pytest.raises(TypeError, match="dimensions must be non-negative"):
        TensorType(DType.float32, [-1], device=DeviceRef.CPU())


def test_tensor_type_with_device(mlir_context: mlir.Context) -> None:
    """Tests tensor type creation."""
    device_type = DeviceRef.GPU(id=2)
    tensor_type = TensorType(DType.float32, shape=[3], device=device_type)
    assert TensorType.from_mlir(tensor_type.to_mlir()) == tensor_type
    assert tensor_type.to_mlir().device_ref == device_type.to_mlir()


def test_tensor_type_layout(mlir_context: mlir.Context) -> None:
    t = TensorType(DType.float32, ["r", "s", "f", "c"], DeviceRef.CPU())
    t_copy = Type.from_mlir(t.to_mlir())
    t._layout = FilterLayout.RSCF
    t2 = Type.from_mlir(t.to_mlir())
    assert t._layout == FilterLayout.RSCF
    assert t2._layout == FilterLayout.RSCF  # type: ignore
    assert t == t2
    assert t != t_copy


def test_tensor_type_layout_default(mlir_context: mlir.Context) -> None:
    t = TensorType(DType.float32, ["r", "s", "f", "c"], DeviceRef.CPU())
    assert t._layout is None


def test_opaque_type(mlir_context: mlir.Context) -> None:
    """Tests opaque type creation and properties."""
    opaque = mo.OpaqueType(
        builtin.StringAttr("custom_type"), builtin.DictionaryAttr()
    )
    assert isinstance(opaque, mo.OpaqueType)
    assert not isinstance(opaque, mo.TensorType)
    assert opaque.symbol.value == "custom_type"


@given(opaque_type=...)
def test_opaque_type_to_mlir(
    mlir_context: mlir.Context, opaque_type: _OpaqueType
) -> None:
    assert opaque_type == _OpaqueType.from_mlir(opaque_type.to_mlir())


def test_type_checking(mlir_context: mlir.Context) -> None:
    """Tests type checking functions."""
    dtype = DType.float32
    dim = Dim(3).to_mlir()
    tensor_type = mo.TensorType(
        mosh.ShapeAttr([dim], mosh.ShapeType()),
        dtype,
        m.DeviceRefAttr("cpu", 0),
    )
    opaque_type = mo.OpaqueType(
        builtin.StringAttr("custom_type"), builtin.DictionaryAttr()
    )
    buffer_type = mo.BufferType(
        mosh.ShapeAttr([dim], mosh.ShapeType()),
        dtype,
        m.DeviceRefAttr("cpu", 0),
    )
    chain_type = mo.ChainType()

    assert isinstance(tensor_type, mo.TensorType)
    assert not isinstance(tensor_type, mo.OpaqueType)
    assert not isinstance(tensor_type, mo.BufferType)
    assert not isinstance(tensor_type, mo.ChainType)

    assert isinstance(opaque_type, mo.OpaqueType)
    assert not isinstance(opaque_type, mo.TensorType)
    assert not isinstance(opaque_type, mo.BufferType)
    assert not isinstance(opaque_type, mo.ChainType)

    assert isinstance(buffer_type, mo.BufferType)
    assert not isinstance(buffer_type, mo.OpaqueType)
    assert not isinstance(buffer_type, mo.TensorType)
    assert not isinstance(buffer_type, mo.ChainType)

    assert isinstance(chain_type, mo.ChainType)
    assert not isinstance(chain_type, mo.TensorType)
    assert not isinstance(chain_type, mo.OpaqueType)
    assert not isinstance(chain_type, mo.BufferType)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(buffer_type=...)
def test_buffer_mlir_roundtrip(
    mlir_context: mlir.Context, buffer_type: BufferType
) -> None:
    assert buffer_type == BufferType.from_mlir(buffer_type.to_mlir())


def test_buffer_type(mlir_context: mlir.Context) -> None:
    """Tests buffer type creation."""
    t = BufferType(DType.float32, [3, "x"], DeviceRef.CPU())
    assert t == Type.from_mlir(t.to_mlir())
    assert t.dtype == DType.float32
    assert t.shape == [Dim(3), Dim("x")]
    assert t.device == DeviceRef.CPU()

    tt = t.to_mlir()
    assert tt == mo.BufferType(
        mosh.ShapeAttr(
            [Dim(3).to_mlir(), Dim("x").to_mlir()],
            mosh.ShapeType(),
        ),
        DType.float32,
        m.DeviceRefAttr("cpu", 0),
    )


def test_chain_type(mlir_context: mlir.Context) -> None:
    assert _ChainType() == _ChainType.from_mlir(_ChainType().to_mlir())


def test_invalid_dimension(mlir_context: mlir.Context) -> None:
    with pytest.raises(TypeError):
        _ = TensorType(
            DType.bfloat16, [-7095393036038990704], device=DeviceRef.CPU()
        )


@pytest.mark.skip("GEX-1918")
def test_GEX_1918(mlir_context: mlir.Context) -> None:
    with pytest.raises(ValueError):
        _ = Dim(2**63 - 1) * 2
    with pytest.raises(ValueError):
        _ = Dim(2**63 - 1) + 1


def test_MAXPLAT_148(mlir_context: mlir.Context) -> None:
    with pytest.raises(TypeError):
        Graph(
            "MAXPLAT-148",
            input_types=[
                TensorType(DType.float32, [-1, 2], device=DeviceRef.CPU())
            ],
        )


def test_device_type(mlir_context: mlir.Context) -> None:
    """Tests Device type."""
    host = DeviceRef.CPU(0)
    cuda0 = DeviceRef.GPU(0)
    cuda1 = DeviceRef.GPU(1)
    cuda1_2 = DeviceRef.GPU(1)
    assert cuda0 != cuda1 != host
    assert cuda0 != cuda1_2 != host
    assert cuda0 != DeviceRef.CPU()
    assert cuda1 == cuda1_2


def test_type_hashing(mlir_context: mlir.Context) -> None:
    lhs = TensorType(DType.float32, [7, 2], device=DeviceRef.CPU())
    rhs = TensorType(DType.float32, [7, 2], device=DeviceRef.CPU())

    assert lhs.to_mlir() == rhs.to_mlir()
    assert hash(lhs) == hash(rhs)
    assert hash(lhs.to_mlir() == rhs.to_mlir())


def test_tensor_type_hashing_layout(mlir_context: mlir.Context) -> None:
    lhs = TensorType(
        DType.float32, [7, 2], device=DeviceRef.CPU(), _layout=FilterLayout.RSCF
    )
    rhs = TensorType(
        DType.float32, [7, 2], device=DeviceRef.CPU(), _layout=FilterLayout.RSCF
    )

    assert lhs == rhs
    assert hash(lhs) == hash(rhs)
