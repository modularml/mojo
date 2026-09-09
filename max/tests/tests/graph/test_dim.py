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
"""Tests for Dim algebra operations."""

from collections.abc import Callable

import numpy as np
import pytest
from max._core.dialects import builtin, kgen
from max.dtype import DType
from max.graph import AlgebraicDim, Dim, Graph, StaticDim, SymbolicDim, ops
from max.graph.type import DeviceRef


class TestStaticDimAlgebraNoContext:
    """Tests for static Dim algebra outside of a Graph context.

    Static dim arithmetic should work without an MLIR context by computing
    results in pure Python.
    """

    def test_static_dim_add(self) -> None:
        """Static dims can be added outside Graph context."""
        result = Dim(5) + Dim(10)
        assert isinstance(result, StaticDim)
        assert int(result) == 15

    def test_static_dim_add_int(self) -> None:
        """Static dims can be added to ints outside Graph context."""
        result = Dim(5) + 10
        assert isinstance(result, StaticDim)
        assert int(result) == 15

    def test_static_dim_radd_int(self) -> None:
        """Ints can be added to static dims outside Graph context."""
        result = 10 + Dim(5)
        assert isinstance(result, StaticDim)
        assert int(result) == 15

    def test_static_dim_sub(self) -> None:
        """Static dims can be subtracted outside Graph context."""
        result = Dim(10) - Dim(3)
        assert isinstance(result, StaticDim)
        assert int(result) == 7

    def test_static_dim_mul(self) -> None:
        """Static dims can be multiplied outside Graph context."""
        result = Dim(5) * Dim(4)
        assert isinstance(result, StaticDim)
        assert int(result) == 20

    def test_static_dim_mul_int(self) -> None:
        """Static dims can be multiplied by ints outside Graph context."""
        result = Dim(5) * 4
        assert isinstance(result, StaticDim)
        assert int(result) == 20

    def test_static_dim_floordiv(self) -> None:
        """Static dims can be floor divided outside Graph context."""
        result = Dim(10) // Dim(3)
        assert isinstance(result, StaticDim)
        assert int(result) == 3

    def test_static_dim_floordiv_int(self) -> None:
        """Static dims can be floor divided by ints outside Graph context."""
        result = Dim(10) // 3
        assert isinstance(result, StaticDim)
        assert int(result) == 3

    def test_static_dim_rfloordiv_int(self) -> None:
        """Ints can be floor divided by static dims outside Graph context."""
        result = 12 // Dim(5)
        assert isinstance(result, StaticDim)
        assert int(result) == 2

    def test_static_dim_rfloordiv_zero_division_raises(self) -> None:
        """Reverse floor division by zero static dim raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            _ = 12 // Dim(0)

    def test_static_dim_neg(self) -> None:
        """Static dims can be negated outside Graph context."""
        result = -Dim(5)
        assert isinstance(result, StaticDim)
        assert int(result) == -5

    def test_static_dim_complex_expression(self) -> None:
        """Complex static dim expressions work outside Graph context."""
        # (5 + 3) * 2 - 4 // 2 = 16 - 2 = 14
        result = (Dim(5) + Dim(3)) * 2 - Dim(4) // 2
        assert isinstance(result, StaticDim)
        assert int(result) == 14

    def test_static_dim_zero_division_raises(self) -> None:
        """Floor division by zero raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            Dim(10) // 0

        with pytest.raises(ZeroDivisionError):
            Dim(10) // Dim(0)


class TestSymbolicDimAlgebraDefaultContext:
    """Symbolic dim algebra works with the default context active."""

    def test_symbolic_dim_add(self) -> None:
        result = Dim("batch") + 1
        assert isinstance(result, AlgebraicDim)
        assert "batch" in str(result)

    def test_symbolic_dim_mul(self) -> None:
        result = Dim("seq_len") * 2
        assert isinstance(result, AlgebraicDim)
        assert "seq_len" in str(result)

    def test_symbolic_dim_floordiv(self) -> None:
        result = Dim("dim") // 4
        assert isinstance(result, AlgebraicDim)
        assert "dim" in str(result)

    def test_algebraic_dim_reused_between_graphs(self) -> None:
        with Graph("test_reuse_1"):
            dim = Dim("batch") + 1

        with Graph("test_reuse_2"):
            result = dim + 2
            assert isinstance(result, AlgebraicDim)
            assert "batch" in str(result)


class TestDimAlgebraInsideGraphContext:
    """Tests for Dim algebra inside a Graph context.

    All dim arithmetic should work inside a Graph context.
    """

    def test_static_dim_add_inside_graph(self) -> None:
        """Static dims can be added inside Graph context."""
        with Graph("test"):
            result = Dim(5) + Dim(10)
            assert isinstance(result, StaticDim)
            assert int(result) == 15

    def test_symbolic_dim_add_inside_graph(self) -> None:
        """Symbolic dims can be added inside Graph context."""
        with Graph("test"):
            result = Dim("batch") + Dim(10)
            assert isinstance(result, AlgebraicDim)
            assert "batch" in str(result)

    def test_symbolic_dim_mul_inside_graph(self) -> None:
        """Symbolic dims can be multiplied inside Graph context."""
        with Graph("test"):
            result = Dim("dim") * 4
            assert isinstance(result, AlgebraicDim)
            assert "dim" in str(result)

    def test_symbolic_dim_floordiv_inside_graph(self) -> None:
        """Symbolic dims can be floor divided inside Graph context."""
        with Graph("test"):
            result = Dim("dim") // 4
            assert isinstance(result, AlgebraicDim)
            assert "dim" in str(result)

    def test_mixed_symbolic_static_inside_graph(self) -> None:
        """Mixed symbolic and static dims work inside Graph context."""
        with Graph("test"):
            result = (Dim("batch") + 1) * 2
            assert isinstance(result, AlgebraicDim)


class TestDimTypes:
    """Tests for Dim type construction and properties."""

    def test_dim_from_int_is_static(self) -> None:
        """Dim from int creates StaticDim."""
        d = Dim(5)
        assert isinstance(d, StaticDim)
        assert int(d) == 5

    def test_dim_from_str_is_symbolic(self) -> None:
        """Dim from str creates SymbolicDim."""
        d = Dim("batch")
        assert isinstance(d, SymbolicDim)
        assert d.name == "batch"

    def test_dim_from_dim_returns_same(self) -> None:
        """Dim from Dim returns the same object."""
        original = Dim(5)
        result = Dim(original)
        assert result is original

    def test_static_dim_int_conversion(self) -> None:
        """StaticDim can be converted to int."""
        d = StaticDim(42)
        assert int(d) == 42

    def test_symbolic_dim_int_conversion_raises(self) -> None:
        """SymbolicDim cannot be converted to int."""
        d = SymbolicDim("batch")
        with pytest.raises(TypeError, match="static dims"):
            int(d)

    def test_symbolic_dim_from_symbolic_dim_returns_same(self) -> None:
        """SymbolicDim from SymbolicDim returns the same object."""
        original = SymbolicDim("batch")
        result = SymbolicDim(original)
        assert result is original

    def test_algebraic_dim_from_algebraic_dim_returns_same(self) -> None:
        """AlgebraicDim from AlgebraicDim returns the same object."""
        original = AlgebraicDim(
            kgen.ParamOperatorAttr(
                kgen.POC.add, [Dim("batch").to_mlir(), Dim(10).to_mlir()]
            )
        )
        result = AlgebraicDim(original)
        assert result is original

    def test_static_dim_from_static_dim_returns_same(self) -> None:
        """StaticDim from StaticDim returns the same object."""
        original = StaticDim(5)
        result = StaticDim(original)
        assert result is original

    @pytest.mark.parametrize("to_type", [StaticDim, SymbolicDim, AlgebraicDim])
    @pytest.mark.parametrize(
        "from_value",
        [
            SymbolicDim("batch"),
            AlgebraicDim(
                kgen.ParamOperatorAttr(
                    kgen.POC.add, [Dim("batch").to_mlir(), Dim(10).to_mlir()]
                )
            ),
            StaticDim(3),
        ],
    )
    def test_constructing_dim_subclass_from_different_dim_subclass(
        self, to_type: type, from_value: Dim
    ) -> None:
        """Constructing a dim subclass from a different dim subclass raises"""
        if isinstance(from_value, to_type):
            pytest.skip("Not considering construction from the same subclass")
        with pytest.raises(TypeError):
            to_type(from_value)

    def test_constructing_algebraic_dim_from_invalid_type(self) -> None:
        """Constructing an AlgebraicDim from an invalid type raises."""
        with pytest.raises(
            TypeError,
            match=r"AlgebraicDim.__init__ only accepts kgen.ParamOperatorAttr or AlgebraicDim, got StaticDim",
        ):
            AlgebraicDim(StaticDim(5))  # type: ignore

    def test_constructing_symbolic_dim_from_invalid_type(self) -> None:
        """Constructing a SymbolicDim from an invalid type raises."""
        with pytest.raises(
            TypeError,
            match=r"SymbolicDim.__init__ only accepts str, kgen.ParamDeclRefAttr, or SymbolicDim, got StaticDim",
        ):
            SymbolicDim(StaticDim(5))  # type: ignore

    def test_constructing_static_dim_from_invalid_type(self) -> None:
        """Constructing a StaticDim from an invalid type raises."""
        with pytest.raises(
            TypeError,
            match=r"StaticDim.__init__ only accepts int, builtin.IntegerAttr, or StaticDim, got SymbolicDim",
        ):
            StaticDim(SymbolicDim("batch"))  # type: ignore

    def test_constructing_static_dim_from_str_raises(self) -> None:
        """Constructing a StaticDim from a str raises rather than creating a SymbolicDim."""
        with pytest.raises(
            TypeError,
            match=r"StaticDim.__init__ only accepts int, builtin.IntegerAttr, or StaticDim, got str",
        ):
            StaticDim("5")  # type: ignore

    def test_constructing_symbolic_dim_from_int_raises(self) -> None:
        """Constructing a SymbolicDim from an int raises rather than creating a StaticDim."""
        with pytest.raises(
            TypeError,
            match=r"SymbolicDim.__init__ only accepts str, kgen.ParamDeclRefAttr, or SymbolicDim, got int",
        ):
            SymbolicDim(5)  # type: ignore

    def test_constructing_algebraic_dim_from_int_raises(self) -> None:
        """Constructing an AlgebraicDim from an int raises rather than creating a StaticDim."""
        with pytest.raises(
            TypeError,
            match=r"AlgebraicDim.__init__ only accepts kgen.ParamOperatorAttr or AlgebraicDim, got int",
        ):
            AlgebraicDim(5)  # type: ignore

    @pytest.mark.parametrize(
        "creator,attr,expected_type,check",
        [
            (
                Dim,
                builtin.IntegerAttr(
                    builtin.IntegerType(64, builtin.SignednessSemantics.signed),
                    5,
                ),
                StaticDim,
                lambda r: int(r) == 5,
            ),
            (
                Dim,
                kgen.ParamDeclRefAttr(
                    "batch",
                    builtin.IntegerType(64, builtin.SignednessSemantics.signed),
                ),
                SymbolicDim,
                lambda r: r.name == "batch",
            ),
            (
                Dim,
                kgen.ParamOperatorAttr(
                    kgen.POC.add, [Dim("batch").to_mlir(), Dim(10).to_mlir()]
                ),
                AlgebraicDim,
                lambda r: (
                    r.attr.opcode == kgen.POC.add
                    and r.attr.operands[0] == Dim("batch").to_mlir()
                    and r.attr.operands[1] == Dim(10).to_mlir()
                ),
            ),
            (
                StaticDim,
                builtin.IntegerAttr(
                    builtin.IntegerType(64, builtin.SignednessSemantics.signed),
                    5,
                ),
                StaticDim,
                lambda r: int(r) == 5,
            ),
            (
                SymbolicDim,
                kgen.ParamDeclRefAttr(
                    "batch",
                    builtin.IntegerType(64, builtin.SignednessSemantics.signed),
                ),
                SymbolicDim,
                lambda r: r.name == "batch",
            ),
            (
                AlgebraicDim,
                kgen.ParamOperatorAttr(
                    kgen.POC.add,
                    [Dim("batch").to_mlir(), Dim(10).to_mlir()],
                ),
                AlgebraicDim,
                lambda r: (
                    r.attr.opcode == kgen.POC.add
                    and r.attr.operands[0] == Dim("batch").to_mlir()
                    and r.attr.operands[1] == Dim(10).to_mlir()
                ),
            ),
            (
                StaticDim.from_mlir,
                builtin.IntegerAttr(
                    builtin.IntegerType(64, builtin.SignednessSemantics.signed),
                    5,
                ),
                StaticDim,
                lambda r: int(r) == 5,
            ),
            (
                SymbolicDim.from_mlir,
                kgen.ParamDeclRefAttr(
                    "batch",
                    builtin.IntegerType(64, builtin.SignednessSemantics.signed),
                ),
                SymbolicDim,
                lambda r: r.name == "batch",
            ),
            (
                AlgebraicDim.from_mlir,
                kgen.ParamOperatorAttr(
                    kgen.POC.add,
                    [Dim("batch").to_mlir(), Dim(10).to_mlir()],
                ),
                AlgebraicDim,
                lambda r: (
                    r.attr.opcode == kgen.POC.add
                    and r.attr.operands[0] == Dim("batch").to_mlir()
                    and r.attr.operands[1] == Dim(10).to_mlir()
                ),
            ),
        ],
    )
    def test_creating_dim_from_attr_returns_correct_type(
        self,
        creator: Callable[[builtin.TypedAttr], Dim],
        attr: builtin.TypedAttr,
        expected_type: type,
        check: Callable[[Dim], bool],
    ) -> None:
        """Creating dim from valid attr returns correct Dim subclass."""
        result = creator(attr)
        assert isinstance(result, expected_type)
        assert check(result)

    @pytest.mark.parametrize(
        "creator,attr,match",
        [
            (
                AlgebraicDim,
                builtin.IntegerAttr(
                    builtin.IntegerType(64, builtin.SignednessSemantics.signed),
                    5,
                ),
                r"AlgebraicDim.__init__ only accepts kgen.ParamOperatorAttr or AlgebraicDim, got IntegerAttr",
            ),
            (
                StaticDim,
                kgen.ParamDeclRefAttr(
                    "batch",
                    builtin.IntegerType(64, builtin.SignednessSemantics.signed),
                ),
                r"StaticDim.__init__ only accepts int, builtin.IntegerAttr, or StaticDim, got ParamDeclRefAttr",
            ),
            (
                SymbolicDim,
                kgen.ParamOperatorAttr(
                    kgen.POC.add,
                    [Dim("batch").to_mlir(), Dim(10).to_mlir()],
                ),
                r"SymbolicDim.__init__ only accepts str, kgen.ParamDeclRefAttr, or SymbolicDim, got ParamOperatorAttr",
            ),
            (
                AlgebraicDim.from_mlir,
                builtin.IntegerAttr(
                    builtin.IntegerType(64, builtin.SignednessSemantics.signed),
                    5,
                ),
                r"AlgebraicDim.from_mlir only accepts kgen.ParamOperatorAttr, got IntegerAttr",
            ),
            (
                StaticDim.from_mlir,
                kgen.ParamDeclRefAttr(
                    "batch",
                    builtin.IntegerType(64, builtin.SignednessSemantics.signed),
                ),
                r"StaticDim.from_mlir only accepts builtin.IntegerAttr, got ParamDeclRefAttr",
            ),
            (
                SymbolicDim.from_mlir,
                kgen.ParamOperatorAttr(
                    kgen.POC.add,
                    [Dim("batch").to_mlir(), Dim(10).to_mlir()],
                ),
                r"SymbolicDim.from_mlir only accepts kgen.ParamDeclRefAttr, got ParamOperatorAttr",
            ),
        ],
    )
    def test_creating_dim_with_invalid_attr_raises(
        self,
        creator: type,
        attr: object,
        match: str,
    ) -> None:
        """Creating dim from invalid attr raises TypeError."""
        with pytest.raises(TypeError, match=match):
            creator(attr)


class TestDimTensorValueInterop:
    """Tests that Dim defers to TensorValue for unsupported operand types.

    When a Dim is on the LHS of an arithmetic op with a TensorValue on the
    RHS, Dim should return NotImplemented so Python falls through to
    TensorValue.__radd__ (etc.), which already handles Dim operands.
    """

    def test_add_dim_lhs(self) -> None:
        """Dim.__add__: Dim(5) + tv -> TensorValue via TensorValue.__radd__."""
        with Graph("test"):
            tv = ops.constant(
                np.array(3, dtype=np.int64), DType.int64, device=DeviceRef.CPU()
            )
            result = Dim(5) + tv
            assert isinstance(result, type(tv))

    def test_add_dim_rhs(self) -> None:
        """TensorValue.__add__: tv + Dim(5) -> TensorValue directly."""
        with Graph("test"):
            tv = ops.constant(
                np.array(3, dtype=np.int64), DType.int64, device=DeviceRef.CPU()
            )
            result = tv + Dim(5)
            assert isinstance(result, type(tv))

    def test_add_symbolic_dim_lhs(self) -> None:
        """Dim.__add__: SymbolicDim + tv -> TensorValue."""
        with Graph("test"):
            tv = ops.constant(
                np.array(3, dtype=np.int64), DType.int64, device=DeviceRef.CPU()
            )
            result = Dim("batch") + tv
            assert isinstance(result, type(tv))

    def test_sub_dim_lhs(self) -> None:
        """Dim.__sub__: Dim(10) - tv -> TensorValue via TensorValue.__rsub__."""
        with Graph("test"):
            tv = ops.constant(
                np.array(3, dtype=np.int64), DType.int64, device=DeviceRef.CPU()
            )
            result = Dim(10) - tv
            assert isinstance(result, type(tv))

    def test_sub_dim_rhs(self) -> None:
        """TensorValue.__sub__: tv - Dim(3) -> TensorValue directly."""
        with Graph("test"):
            tv = ops.constant(
                np.array(10, dtype=np.int64),
                DType.int64,
                device=DeviceRef.CPU(),
            )
            result = tv - Dim(3)
            assert isinstance(result, type(tv))

    def test_mul_dim_lhs(self) -> None:
        """Dim.__mul__: Dim(5) * tv -> TensorValue via TensorValue.__rmul__."""
        with Graph("test"):
            tv = ops.constant(
                np.array(3, dtype=np.int64), DType.int64, device=DeviceRef.CPU()
            )
            result = Dim(5) * tv
            assert isinstance(result, type(tv))

    def test_mul_dim_rhs(self) -> None:
        """TensorValue.__mul__: tv * Dim(5) -> TensorValue directly."""
        with Graph("test"):
            tv = ops.constant(
                np.array(3, dtype=np.int64), DType.int64, device=DeviceRef.CPU()
            )
            result = tv * Dim(5)
            assert isinstance(result, type(tv))

    def test_floordiv_dim_lhs(self) -> None:
        """Dim.__floordiv__: Dim(10) // tv -> TensorValue via TensorValue.__rfloordiv__."""
        with Graph("test"):
            tv = ops.constant(
                np.array(3, dtype=np.int64), DType.int64, device=DeviceRef.CPU()
            )
            result = Dim(10) // tv
            assert isinstance(result, type(tv))

    def test_floordiv_dim_rhs(self) -> None:
        """TensorValue.__floordiv__: tv // Dim(3) -> TensorValue directly."""
        with Graph("test"):
            tv = ops.constant(
                np.array(10, dtype=np.int64),
                DType.int64,
                device=DeviceRef.CPU(),
            )
            result = tv // Dim(3)
            assert isinstance(result, type(tv))
