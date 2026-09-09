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
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.graph.ops import div, floor, floor_div


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


def build_floor_div_model(lhs_dtype: DType, rhs_dtype: DType) -> Graph:
    lhs_type = TensorType(lhs_dtype, [], device=DeviceRef.CPU())
    rhs_type = TensorType(rhs_dtype, [], device=DeviceRef.CPU())
    with Graph("floor_div", input_types=[lhs_type, rhs_type]) as g:
        g.output(floor_div(g.inputs[0].tensor, g.inputs[1].tensor))
    return g


@pytest.mark.parametrize(
    "dtype,cases",
    [
        # Signed integers exercise the floor correction: truncation toward zero
        # would give -3 (not -4) for -7 // 2, so this pins the signed path.
        (DType.int32, [(7, 2), (-7, 2), (7, -2), (-7, -2), (-6, 3), (0, 5)]),
        # Unsigned skips the correction (fast path); operands stay non-negative.
        (DType.uint32, [(7, 2), (18, 5)]),
        # Float operands route through floor(div(...)).
        (DType.float32, [(7.5, 2.0), (-7.5, 2.0)]),
    ],
)
def test_floor_div_matches_python(
    dtype: DType,
    cases: list[tuple[int | float, int | float]],
) -> None:
    """``ops.floor_div`` must match Python ``//`` across sign and dtype."""
    cpu = CPU()
    session = InferenceSession(devices=[cpu])
    model = session.load(build_floor_div_model(dtype, dtype))

    for lhs_val, rhs_val in cases:
        lhs = Buffer.from_numpy(np.array(lhs_val, dtype=dtype.to_numpy())).to(
            cpu
        )
        rhs = Buffer.from_numpy(np.array(rhs_val, dtype=dtype.to_numpy())).to(
            cpu
        )
        out = model(lhs, rhs)[0]
        got = (
            out.to_numpy().item()
            if isinstance(out, Buffer)
            else torch.utils.dlpack.from_dlpack(out).numpy().item()
        )
        np.testing.assert_equal(
            got, lhs_val // rhs_val, err_msg=f"floor_div({lhs_val}, {rhs_val})"
        )
