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

from typing import Any

import numpy as np
import pytest
import torch.utils.dlpack
from conftest import broadcast_shapes, broadcastable_tensor_types
from hypothesis import assume, event, given
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType
from max.graph.ops import div, floor


@given(tensor_type=...)
def test_div__same_type(tensor_type: TensorType) -> None:
    """Test div with same tensor types."""
    with Graph("div", input_types=[tensor_type, tensor_type]) as graph:
        op = div(graph.inputs[0].tensor, graph.inputs[1].tensor)
        # For integer types, division promotes to float64 for true division
        if tensor_type.dtype.is_integral():
            assert op.dtype == DType.float64
        else:
            assert op.type == tensor_type


@given(tensor_type=...)
def test_div__same_type__operator(tensor_type: TensorType) -> None:
    """Test div operator (/) with same tensor types."""
    with Graph("div", input_types=[tensor_type, tensor_type]) as graph:
        op = graph.inputs[0] / graph.inputs[1]  # type: ignore
        # For integer types, division promotes to float64 for true division
        if tensor_type.dtype.is_integral():
            assert op.dtype == DType.float64
        else:
            assert op.type == tensor_type


@given(d1=..., d2=..., shape=...)
def test_div__promoted_dtype__operator(
    d1: DType, d2: DType, shape: list[Dim]
) -> None:
    """Test div operator with dtype promotion."""
    assume(d1 != d2)
    t1 = TensorType(d1, shape, device=DeviceRef.CPU())
    t2 = TensorType(d2, shape, device=DeviceRef.CPU())
    with Graph("div", input_types=[t1, t2]) as graph:
        i0, i1 = graph.inputs
        try:
            result = i0 / i1  # type: ignore
            if d1.is_integral() and d2.is_integral():
                assert result.dtype == DType.float64
            elif d1.is_float() or d2.is_float():
                assert result.dtype.is_float()

            result2 = i1 / i0  # type: ignore
            assert result.dtype == result2.dtype
            event("types promote")
        except ValueError as e:
            assert "Unsafe cast" in str(e)
            event("types don't promote")


@given(types=broadcastable_tensor_types(2))
def test_div__broadcast__operator(types: list[TensorType]) -> None:
    """Test div operator with broadcasting."""
    t1, t2 = types
    broadcast_shape = broadcast_shapes(t1.shape, t2.shape)
    with Graph("div", input_types=[t1, t2]) as graph:
        i0, i1 = graph.inputs
        assert (i0 / i1).shape == broadcast_shape  # type: ignore
        assert (i1 / i0).shape == broadcast_shape  # type: ignore


@given(tensor_type=...)
def test_div__python_int__operator(tensor_type: TensorType) -> None:
    """Test div operator with Python int."""
    with Graph("div", input_types=[tensor_type, tensor_type]) as graph:
        try:
            op = graph.inputs[0] / 2  # type: ignore
            if tensor_type.dtype.is_integral():
                assert op.dtype == DType.float64  # type: ignore
            else:
                assert op.type == tensor_type  # type: ignore
        except ValueError:
            # Some type combinations may not be supported
            pass


@given(tensor_type=...)
def test_div__mismatched_devices(tensor_type: TensorType) -> None:
    device = DeviceRef.GPU(1)
    assume(tensor_type.device != device)
    other_type = TensorType(tensor_type.dtype, tensor_type.shape, device)
    with Graph("div", input_types=[tensor_type, other_type]) as graph:
        tensor, other = graph.inputs
        with pytest.raises(ValueError, match="same device"):
            _ = tensor.tensor / other.tensor


def build_models(lhs_dtype: DType, rhs_dtype: DType) -> Graph:
    lhs_type = TensorType(lhs_dtype, [], device=DeviceRef.CPU())
    rhs_type = TensorType(rhs_dtype, [], device=DeviceRef.CPU())
    with Graph("div_and_floor", input_types=[lhs_type, rhs_type]) as g:
        q = div(g.inputs[0].tensor, g.inputs[1].tensor)
        g.output(q, floor(q))
    return g


def run_models(
    model: Any,
    lhs_val: float,
    rhs_val: float,
    lhs_dtype: DType,
    rhs_dtype: DType,
    cpu: Device,
) -> tuple[float, float]:
    lhs = Buffer.from_numpy(np.array(lhs_val, dtype=lhs_dtype.to_numpy())).to(
        cpu
    )
    rhs = Buffer.from_numpy(np.array(rhs_val, dtype=rhs_dtype.to_numpy())).to(
        cpu
    )
    out_div, out_floor = model(lhs, rhs)

    def to_scalar(x: Any) -> float:
        if isinstance(x, Buffer):
            return x.to_numpy().item()
        return torch.utils.dlpack.from_dlpack(x).numpy().item()

    return to_scalar(out_div), to_scalar(out_floor)


@pytest.mark.parametrize(
    "lhs_dtype,rhs_dtype,cases",
    [
        (
            DType.int32,
            DType.int32,
            [(7, 3), (7, -3), (-7, 3), (-7, -3), (1, 2), (0, 5)],
        ),
        (DType.int64, DType.int64, [(15, 4), (np.iinfo(np.int64).min, -1)]),
        (
            DType.float32,
            DType.float32,
            [(7.5, 2.5), (1.0, 3.0), (-0.0, 3.0)],
        ),
        (DType.float64, DType.float64, [(7.0, 3.0), (1.0, 3.0)]),
    ],
)
def test_div_and_floordiv_match_python(
    lhs_dtype: DType,
    rhs_dtype: DType,
    cases: list[tuple[int | float, int | float]],
) -> None:
    cpu = CPU()
    session = InferenceSession(devices=[cpu])
    graph = build_models(lhs_dtype, rhs_dtype)
    model = session.load(graph)

    for lhs_val, rhs_val in cases:
        py_div = lhs_val / rhs_val
        py_floor = lhs_val // rhs_val

        max_div, max_floor = run_models(
            model, lhs_val, rhs_val, lhs_dtype, rhs_dtype, cpu
        )

        if DType.is_float(lhs_dtype) or DType.is_float(rhs_dtype):
            if lhs_dtype == DType.float64 or rhs_dtype == DType.float64:
                rtol, atol = 1e-12, 0.0
            else:
                rtol, atol = 1e-6, 1e-7
            np.testing.assert_allclose(
                max_div,
                py_div,
                rtol=rtol,
                atol=atol,
                err_msg=f"div({lhs_val}, {rhs_val}) mismatch",
            )
            np.testing.assert_allclose(
                max_floor,
                py_floor,
                rtol=rtol,
                atol=atol,
                err_msg=f"floor(div({lhs_val}, {rhs_val})) mismatch",
            )
        else:
            np.testing.assert_equal(max_div, py_div)
            np.testing.assert_equal(max_floor, py_floor)
