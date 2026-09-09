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
"""Validates the row_mean_of_squares graph op against a float64 reference.

Mirrors the kernel-level Mojo test (test_row_mean_of_squares.mojo) but exercises
the full graph-op path: ops.custom -> mo.reduce.row_mean_of_squares ->
row_mean_of_squares_gpu. The reference is computed in float64 so the only error
is the input's representational error (bf16) plus the f32 accumulation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import row_mean_of_squares

# (rows, cols): decode (small M), prefill (large M), odd-N tail, large-N.
_SHAPES = [
    (16, 1536),
    (16, 256),
    (512, 1536),
    (2048, 256),
    (16, 1537),
    (4, 8192),
]


def _run(rows: int, cols: int, dtype: DType, rtol: float, atol: float) -> None:
    device = Accelerator(0)
    device_ref = DeviceRef.GPU()

    torch_dtype = {
        DType.bfloat16: torch.bfloat16,
        DType.float32: torch.float32,
    }[dtype]

    x = torch.randn((rows, cols), dtype=torch_dtype, device="cpu")

    with Graph(
        "row_mean_of_squares",
        input_types=(TensorType(dtype, [rows, cols], device=device_ref),),
    ) as graph:
        (x_val,) = graph.inputs
        out = row_mean_of_squares(x_val.tensor)
        graph.output(out)

    session = InferenceSession(devices=[device])
    compiled = session.load(graph)
    (result,) = compiled.execute(Buffer.from_dlpack(x).to(device))

    max_out = torch.from_dlpack(result).to("cpu").to(torch.float64).numpy()
    assert max_out.shape == (rows, 1)

    # float64 reference: square/accumulate in f64 over the f64-promoted input.
    x_f64 = x.to(torch.float64).numpy()
    ref = (x_f64 * x_f64).mean(axis=-1, keepdims=True)

    np.testing.assert_allclose(max_out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("rows,cols", _SHAPES)
def test_row_mean_of_squares_bfloat16(rows: int, cols: int) -> None:
    _run(rows, cols, DType.bfloat16, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("rows,cols", [(16, 1536), (16, 256), (16, 1537)])
def test_row_mean_of_squares_float32(rows: int, cols: int) -> None:
    _run(rows, cols, DType.float32, rtol=1e-6, atol=1e-6)
