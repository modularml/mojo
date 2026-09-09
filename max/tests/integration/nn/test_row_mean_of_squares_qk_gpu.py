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
"""Validates the fused row_mean_of_squares_qk graph op against a float64 ref.

Mirrors the kernel-level Mojo test (test_row_mean_of_squares_qk.mojo) but
exercises the full graph-op path: ops.custom ->
mo.reduce.row_mean_of_squares_qk -> row_mean_of_squares_qk_gpu. The reference is
computed in float64 so the only error is the input's representational error
(bf16) plus the f32 accumulation. Also checks equivalence with two separate
row_mean_of_squares calls + concat.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import row_mean_of_squares_qk

# (rows, q_cols, k_cols): decode (small M), prefill (large M), odd-N tail,
# wide-N grid-stride, equal widths.
_SHAPES = [
    (16, 1536, 256),
    (512, 1536, 256),
    (2048, 4096, 512),
    (16, 1536, 257),
    (4, 8192, 8192),
    (16, 512, 512),
]


def _run(
    rows: int,
    q_cols: int,
    k_cols: int,
    dtype: DType,
    rtol: float,
    atol: float,
) -> None:
    device = Accelerator(0)
    device_ref = DeviceRef.GPU()

    torch_dtype = {
        DType.bfloat16: torch.bfloat16,
        DType.float32: torch.float32,
    }[dtype]

    q = torch.randn((rows, q_cols), dtype=torch_dtype, device="cpu")
    k = torch.randn((rows, k_cols), dtype=torch_dtype, device="cpu")

    with Graph(
        "row_mean_of_squares_qk",
        input_types=(
            TensorType(dtype, [rows, q_cols], device=device_ref),
            TensorType(dtype, [rows, k_cols], device=device_ref),
        ),
    ) as graph:
        q_val, k_val = graph.inputs
        out = row_mean_of_squares_qk(q_val.tensor, k_val.tensor)
        graph.output(out)

    session = InferenceSession(devices=[device])
    compiled = session.load(graph)
    (result,) = compiled.execute(
        Buffer.from_dlpack(q).to(device), Buffer.from_dlpack(k).to(device)
    )

    max_out = torch.from_dlpack(result).to("cpu").to(torch.float64).numpy()
    assert max_out.shape == (rows, 2)

    # float64 reference: square/accumulate in f64 over the f64-promoted inputs.
    q_f64 = q.to(torch.float64).numpy()
    k_f64 = k.to(torch.float64).numpy()
    ref = np.concatenate(
        [
            (q_f64 * q_f64).mean(axis=-1, keepdims=True),
            (k_f64 * k_f64).mean(axis=-1, keepdims=True),
        ],
        axis=-1,
    )

    np.testing.assert_allclose(max_out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("rows,q_cols,k_cols", _SHAPES)
def test_row_mean_of_squares_qk_bfloat16(
    rows: int, q_cols: int, k_cols: int
) -> None:
    _run(rows, q_cols, k_cols, DType.bfloat16, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "rows,q_cols,k_cols", [(16, 1536, 256), (16, 1537, 257)]
)
def test_row_mean_of_squares_qk_float32(
    rows: int, q_cols: int, k_cols: int
) -> None:
    _run(rows, q_cols, k_cols, DType.float32, rtol=1e-6, atol=1e-6)
