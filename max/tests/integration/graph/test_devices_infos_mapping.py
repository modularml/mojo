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
"""Tests for mo.device_info_mapping attribute population on the module op."""

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def test_device_info_mapping_contains_cpu() -> None:
    """CPU entry must always be present regardless of available accelerators."""
    t = TensorType(DType.float32, [4], DeviceRef.CPU())
    with Graph("test_graph", input_types=[t, t]) as graph:
        x, y = (v.tensor for v in graph.inputs)
        graph.output(ops.add(x, y))

    module = graph._mlir_op.block.owner
    assert "mo.device_info_mapping" in module.attributes
    attr_str = str(module.attributes["mo.device_info_mapping"])
    assert '#M.device_info<"cpu", "cpu", "unknown", "cpu">' in attr_str
