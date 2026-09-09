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
"""Mixed bf16-A x fp32-B router-gate GEMV op (KERN-3098).

Covers the graph custom op (`mo.router.gate.mixed.gemv`) shape/dtype contract
and numerics, including the tiny-M GEMV and large-M matmul fallback paths.
"""

from __future__ import annotations

import pytest
import torch
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import _router_gate_mixed_gemv
from torch.utils.dlpack import from_dlpack

# Router-gate shape (KERN-3219): N experts, K hidden.
N = 128
K = 6144


def _build_op_graph(m: int) -> Graph:
    with Graph(
        "router_gate_mixed_gemv",
        input_types=(
            TensorType(DType.bfloat16, [m, K], device=DeviceRef.GPU()),
            TensorType(DType.float32, [N, K], device=DeviceRef.GPU()),
        ),
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        graph.output(_router_gate_mixed_gemv(a, b))
    return graph


# m<=16 = the original mixed-GEMV band, 17-64 = the widened band (formerly
# fallback), 512 = still the large-M fallback.
@pytest.mark.parametrize("m", [2, 16, 17, 24, 32, 64, 512])
def test_router_gate_mixed_gemv_op(
    gpu_session: InferenceSession, m: int
) -> None:
    """The op returns fp32 [M, N] and is numerically equal to cast-A @ B.T on
    both the tiny-M fused route and the large-M cast + fp32-matmul fallback."""
    torch.manual_seed(0)
    a = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.float32, device="cuda")

    compiled = gpu_session.load(_build_op_graph(m))
    got = from_dlpack(compiled.execute(a, b)[0])

    assert tuple(got.shape) == (m, N)
    assert got.dtype == torch.float32

    # Reference: widen bf16 A to fp32 (lossless) then fp32 GEMV — exactly the
    # cast + fp32-GEMV chain the fused op replaces.
    ref = a.to(torch.float32) @ b.t()
    # Exact top-1 routing (the property the MoE gate depends on).
    assert torch.equal(got.argmax(dim=1), ref.argmax(dim=1))
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)


def test_router_gate_mixed_gemv_validation() -> None:
    """The wrapper rejects wrong dtypes and dynamic N/K at graph build."""
    with Graph(
        "router_gate_mixed_gemv_validation",
        input_types=(
            TensorType(DType.float32, [8, K], device=DeviceRef.GPU()),
            TensorType(DType.bfloat16, [N, K], device=DeviceRef.GPU()),
            TensorType(DType.bfloat16, [8, K], device=DeviceRef.GPU()),
            TensorType(DType.float32, ["N", "K"], device=DeviceRef.GPU()),
        ),
    ) as graph:
        a_f32, b_bf16, a_bf16, b_dyn = (v.tensor for v in graph.inputs)
        # fp32 activation rejected (must be bf16).
        try:
            _router_gate_mixed_gemv(a_f32, b_bf16)
        except ValueError:
            pass
        else:
            raise AssertionError("fp32 hidden states must be rejected")
        # bf16 weight rejected (must be fp32).
        try:
            _router_gate_mixed_gemv(a_bf16, b_bf16)
        except ValueError:
            pass
        else:
            raise AssertionError("bf16 gate weights must be rejected")
        # Dynamic N/K rejected (must be static).
        try:
            _router_gate_mixed_gemv(a_bf16, b_dyn)
        except ValueError:
            pass
        else:
            raise AssertionError("dynamic gate shapes must be rejected")
