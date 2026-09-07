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

"""One graph exercising several MAP fusers together, end to end.

``reshape(relu(reshape(matmul(A, transpose(B))) + broadcast(C))) + D``
collapses into a single kernel under the legacy pipeline: matmul's own
epilogue absorbs the transpose, both reshapes, the broadcast, and both
adds. Not env-gated -- the new MAP-dialect system doesn't yet fuse this
graph as aggressively; Commit 7's exclusion list will pick it up.
"""

from __future__ import annotations

import re

import numpy as np
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops


def _assert_canonical_fusion(model: Model) -> None:
    """Checks that the whole graph fused into a single kernel."""
    summaries = model.kernel_summaries
    assert any(
        re.search(
            r"mo\.matmul.*mo\.static\.broadcast_to.*mo\.add.*mo\.relu.*mo\.add",
            s,
        )
        for s in summaries
    ), summaries


def _reference(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> np.ndarray:
    mm = a @ b.T
    r1 = mm.reshape([-1])
    s = r1 + c
    relu = np.maximum(s, 0)
    r2 = relu.reshape(mm.shape)
    return r2 + d


def test_static_shapes(session: InferenceSession) -> None:
    with Graph(
        "canonical_fused_graph_static",
        input_types=[
            TensorType(DType.float32, [2, 3], device=DeviceRef.CPU()),
            TensorType(DType.float32, [4, 3], device=DeviceRef.CPU()),
            TensorType(DType.float32, [1], device=DeviceRef.CPU()),
            TensorType(DType.float32, [2, 4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b, c, d = (v.tensor for v in graph.inputs)
        mm = ops.matmul(a, ops.transpose(b, 0, 1))
        r1 = mm.reshape([-1])
        s = r1 + c.broadcast_to(r1.shape)
        relu = ops.relu(s)
        r2 = relu.reshape(mm.shape)
        graph.output(r2 + d)

    model = session.load(graph)
    _assert_canonical_fusion(model)

    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(4, 3).astype(np.float32)
    c_np = np.random.randn(1).astype(np.float32)
    d_np = np.random.randn(2, 4).astype(np.float32)
    (out,) = model.execute(a_np, b_np, c_np, d_np)
    np.testing.assert_allclose(
        out.to_numpy(),
        _reference(a_np, b_np, c_np, d_np),
        rtol=1e-4,
        atol=1e-4,
    )


def test_dynamic_shapes(session: InferenceSession) -> None:
    """Same graph, symbolic dims on `A`/`B`/`D` (`m`, `n`, `k`).

    `C` stays a plain size-1 broadcast source in both variants: its role is
    to be stretched to whatever `A @ B.T` flattens to, not to carry a named
    dim of its own.
    """
    with Graph(
        "canonical_fused_graph_dynamic",
        input_types=[
            TensorType(DType.float32, ["m", "k"], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["n", "k"], device=DeviceRef.CPU()),
            TensorType(DType.float32, [1], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["m", "n"], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b, c, d = (v.tensor for v in graph.inputs)
        mm = ops.matmul(a, ops.transpose(b, 0, 1))
        r1 = mm.reshape([-1])
        s = r1 + c.broadcast_to(r1.shape)
        relu = ops.relu(s)
        r2 = relu.reshape(mm.shape)
        graph.output(r2 + d)

    model = session.load(graph)
    _assert_canonical_fusion(model)

    a_np = np.random.randn(5, 6).astype(np.float32)
    b_np = np.random.randn(7, 6).astype(np.float32)
    c_np = np.random.randn(1).astype(np.float32)
    d_np = np.random.randn(5, 7).astype(np.float32)
    (out,) = model.execute(a_np, b_np, c_np, d_np)
    np.testing.assert_allclose(
        out.to_numpy(),
        _reference(a_np, b_np, c_np, d_np),
        rtol=1e-4,
        atol=1e-4,
    )
