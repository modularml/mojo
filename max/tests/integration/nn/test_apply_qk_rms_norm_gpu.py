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
"""Validates the fused apply_qk_rms_norm graph op against a float64 reference.

Mirrors the kernel-level Mojo test (test_apply_qk_rms_norm.mojo) but exercises
the full graph-op path: ops.custom -> mo.norm.apply_qk_rms_norm ->
apply_qk_rms_norm_gpu. The reference is computed in float64 so the only error is
the inputs' representational error (bf16) plus the output downcast. Also
cross-checks against the unfused graph (cast/rsqrt/mul) this op replaces.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import apply_qk_rms_norm

_EPS = 1e-6

# (rows, q_cols, k_cols): decode (small M), prefill (large M), odd-N tail,
# wide-N grid-stride, equal widths. Q/K cols are MiniMax-M2.7 TP4 per-rank
# slices of 6144/1024.
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
    gamma_q = torch.randn((q_cols,), dtype=torch.float32, device="cpu")
    gamma_k = torch.randn((k_cols,), dtype=torch.float32, device="cpu")

    # Realistic variance statistics: per-row mean of squares (as produced by
    # row_mean_of_squares_qk upstream), computed in float32.
    qk_var = torch.stack(
        [
            (q.to(torch.float32) ** 2).mean(dim=-1),
            (k.to(torch.float32) ** 2).mean(dim=-1),
        ],
        dim=-1,
    ).contiguous()

    with Graph(
        "apply_qk_rms_norm",
        input_types=(
            TensorType(dtype, [rows, q_cols], device=device_ref),
            TensorType(dtype, [rows, k_cols], device=device_ref),
            TensorType(DType.float32, [rows, 2], device=device_ref),
            TensorType(DType.float32, [q_cols], device=device_ref),
            TensorType(DType.float32, [k_cols], device=device_ref),
        ),
    ) as graph:
        q_val, k_val, var_val, gq_val, gk_val = graph.inputs
        q_out, k_out = apply_qk_rms_norm(
            q_val.tensor,
            k_val.tensor,
            var_val.tensor,
            gq_val.tensor,
            gk_val.tensor,
            _EPS,
        )
        # Reference (unfused) path: the exact ops the kernel replaces, so the
        # graph also asserts semantic equivalence with the pre-fusion form.
        q_var = ops.slice_tensor(var_val.tensor, [slice(None), slice(0, 1)])
        k_var = ops.slice_tensor(var_val.tensor, [slice(None), slice(1, 2)])
        qf = ops.cast(q_val.tensor, DType.float32) * ops.rsqrt(q_var + _EPS)
        kf = ops.cast(k_val.tensor, DType.float32) * ops.rsqrt(k_var + _EPS)
        q_ref = ops.cast(qf * gq_val.tensor, dtype)
        k_ref = ops.cast(kf * gk_val.tensor, dtype)
        graph.output(q_out, k_out, q_ref, k_ref)

    session = InferenceSession(devices=[device])
    compiled = session.load(graph)
    q_res, k_res, q_ref_res, k_ref_res = compiled.execute(
        Buffer.from_dlpack(q).to(device),
        Buffer.from_dlpack(k).to(device),
        Buffer.from_dlpack(qk_var).to(device),
        Buffer.from_dlpack(gamma_q).to(device),
        Buffer.from_dlpack(gamma_k).to(device),
    )

    q_max = torch.from_dlpack(q_res).to("cpu").to(torch.float64).numpy()
    k_max = torch.from_dlpack(k_res).to("cpu").to(torch.float64).numpy()
    assert q_max.shape == (rows, q_cols)
    assert k_max.shape == (rows, k_cols)

    # float64 reference: scale and gamma-multiply in f64 over f64-promoted in.
    q_f64 = q.to(torch.float64).numpy()
    k_f64 = k.to(torch.float64).numpy()
    var_f64 = qk_var.to(torch.float64).numpy()
    gq_f64 = gamma_q.to(torch.float64).numpy()
    gk_f64 = gamma_k.to(torch.float64).numpy()
    rs_q = 1.0 / np.sqrt(var_f64[:, 0:1] + _EPS)
    rs_k = 1.0 / np.sqrt(var_f64[:, 1:2] + _EPS)
    q_expected = (q_f64 * rs_q) * gq_f64
    k_expected = (k_f64 * rs_k) * gk_f64

    np.testing.assert_allclose(q_max, q_expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(k_max, k_expected, rtol=rtol, atol=atol)

    # The fused op must match the unfused graph it replaces. They share the
    # float grouping and dtype; any residual gap is GPU rsqrt lowering (~1 ULP),
    # so compare with the same dtype tolerance rather than bit-for-bit.
    q_ref = torch.from_dlpack(q_ref_res).to("cpu").to(torch.float64).numpy()
    k_ref = torch.from_dlpack(k_ref_res).to("cpu").to(torch.float64).numpy()
    np.testing.assert_allclose(q_max, q_ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(k_max, k_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("rows,q_cols,k_cols", _SHAPES)
def test_apply_qk_rms_norm_bfloat16(
    rows: int, q_cols: int, k_cols: int
) -> None:
    _run(rows, q_cols, k_cols, DType.bfloat16, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "rows,q_cols,k_cols", [(16, 1536, 256), (16, 1537, 257)]
)
def test_apply_qk_rms_norm_float32(rows: int, q_cols: int, k_cols: int) -> None:
    _run(rows, q_cols, k_cols, DType.float32, rtol=1e-6, atol=1e-6)
