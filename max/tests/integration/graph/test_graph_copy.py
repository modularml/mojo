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
"""Tests for Graph.copy."""

from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def _asm(g: Graph) -> str:
    return g._module.asm(assume_verified=True, use_local_scope=True)


def _build_add_graph() -> Graph:
    with Graph(
        "copy_test",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.CPU())],
    ) as g:
        (x,) = g.inputs
        g.output(ops.add(x, x))
    return g


def test_copy_is_deep_and_equal() -> None:
    g = _build_add_graph()
    copied = g.copy()
    assert copied._module is not g._module
    assert _asm(copied) == _asm(g)
    assert copied._kernel_library is g._kernel_library


def test_copy_is_executable(session: InferenceSession) -> None:
    """The copy compiles and runs independently of the original graph object."""
    copied = _build_add_graph().copy()
    model = session.load(copied)
    ones = Buffer(DType.float32, [4])
    for i in range(4):
        ones[i] = 1.0
    (out,) = model(ones)
    assert out is not None
    assert [out[i].item() for i in range(4)] == [2.0] * 4
