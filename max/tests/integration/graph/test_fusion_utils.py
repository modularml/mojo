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

"""Tests for `fusion_utils.run_and_verify_fusion`.

These test the helper's own regex matching, not the graph compiler's fusion
behavior -- that's what the fusion test suites and lit tests are for. A
single-op graph (a lone `mo.add`, nothing to fuse with) is enough: its
compiled kernel summary is just that op's own name, with no group wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest
from fusion_utils import run_and_verify_fusion
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def _build_add_graph() -> Graph:
    with Graph(
        "single_add",
        input_types=[
            TensorType(DType.float32, [4], device=DeviceRef.CPU()),
            TensorType(DType.float32, [4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        graph.output(ops.add(a, b))
    return graph


def test_matching_pattern_returns_result(session: InferenceSession) -> None:
    """A pattern matching the summary returns the executed result."""
    graph = _build_add_graph()
    a_np = np.random.randn(4).astype(np.float32)
    b_np = np.random.randn(4).astype(np.float32)
    (out,) = run_and_verify_fusion(session, graph, a_np, b_np, fused=r"mo\.add")
    np.testing.assert_allclose(out, a_np + b_np, rtol=1e-5, atol=1e-5)


def test_non_matching_pattern_raises(session: InferenceSession) -> None:
    """A pattern that matches no summary fails the call, not the test body."""
    graph = _build_add_graph()
    a_np = np.random.randn(4).astype(np.float32)
    b_np = np.random.randn(4).astype(np.float32)
    with pytest.raises(AssertionError, match="expected a kernel summary"):
        run_and_verify_fusion(session, graph, a_np, b_np, fused=r"mo\.mul")
