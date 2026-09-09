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
"""test the max.graph python bindings."""

from conftest import GraphBuilder, float_dtypes, tensor_types
from hypothesis import given
from max.graph import TensorType
from max.graph.ops import ceil


@given(tensor_type=tensor_types(dtypes=float_dtypes()))
def test_ceil_same_type(
    graph_builder: GraphBuilder, tensor_type: TensorType
) -> None:
    """ceil preserves the input shape, dtype, and device."""
    with graph_builder(input_types=[tensor_type]) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        op = ceil(x)
        assert op.type == tensor_type
        graph.output(op)
