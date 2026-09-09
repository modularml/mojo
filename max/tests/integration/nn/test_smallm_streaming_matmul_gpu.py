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
"""MI355X small-M streaming matmul op over a CPU-preshuffled weight.

Covers the graph custom op (`mo.smallm.streaming.matmul`) contract and, by
comparing against a plain matmul on the ORIGINAL weight, the layout parity
between the numpy `preshuffle_smallm_b` and the Mojo reader kernel.
"""

from __future__ import annotations

import pytest
import torch
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import smallm_streaming_matmul
from max.pipelines.weights.smallm_preshuffle import preshuffle_smallm_b
from torch.utils.dlpack import from_dlpack

N = 512
K = 2048


def _build_op_graph(m: int) -> Graph:
    with Graph(
        "smallm_streaming_matmul",
        input_types=(
            TensorType(DType.bfloat16, [m, K], device=DeviceRef.GPU()),
            TensorType(DType.bfloat16, [N, K], device=DeviceRef.GPU()),
            TensorType(DType.bfloat16, [N, K], device=DeviceRef.GPU()),
        ),
    ) as graph:
        a, b_shuffled, b = (v.tensor for v in graph.inputs)
        graph.output(smallm_streaming_matmul(a, b_shuffled, b))
    return graph


# 1..16 = register-A band, 17..32 = the paired-column-tile band, and
# above 32 the op falls back to generic matmul dispatch at execute time.
@pytest.mark.parametrize("m", [1, 2, 16, 17, 32, 64, 128, 200])
def test_smallm_streaming_matmul_op(
    gpu_session: InferenceSession, m: int
) -> None:
    """The op over the numpy-shuffled weight equals a @ w.T on the original."""
    torch.manual_seed(0)
    a = torch.randn(m, K, dtype=torch.bfloat16, device="cuda") * 0.5
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.5

    w_shuffled = (
        torch.from_numpy(
            preshuffle_smallm_b(w.cpu().view(torch.uint16).numpy())
        )
        .view(torch.bfloat16)
        .cuda()
    )

    compiled = gpu_session.load(_build_op_graph(m))
    got = from_dlpack(compiled.execute(a, w_shuffled, w)[0])

    assert tuple(got.shape) == (m, N)
    assert got.dtype == torch.bfloat16

    # fp32 accumulate, bf16 out — same as the kernel's epilogue.
    ref = (a.to(torch.float32) @ w.to(torch.float32).t()).to(torch.bfloat16)
    torch.testing.assert_close(
        got.to(torch.float32), ref.to(torch.float32), rtol=2e-2, atol=1e-2
    )


def test_smallm_streaming_matmul_validation() -> None:
    """The wrapper rejects unaligned shapes and wrong dtypes at graph build."""
    with Graph(
        "smallm_validation",
        input_types=(
            TensorType(DType.bfloat16, [4, K], device=DeviceRef.GPU()),
            TensorType(DType.bfloat16, [N + 8, K], device=DeviceRef.GPU()),
            TensorType(DType.bfloat16, [N, K + 32], device=DeviceRef.GPU()),
            TensorType(DType.float32, [N, K], device=DeviceRef.GPU()),
        ),
    ) as graph:
        a, b_badn, b_badk, b_f32 = (v.tensor for v in graph.inputs)
        with pytest.raises(ValueError, match="N % 16"):
            smallm_streaming_matmul(a, b_badn, b_badn)
        with pytest.raises(ValueError, match="N % 16"):
            smallm_streaming_matmul(a, b_badk, b_badk)
        with pytest.raises(ValueError):
            smallm_streaming_matmul(a, b_f32, b_f32)
        graph.output(a)
