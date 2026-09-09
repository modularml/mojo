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
"""Ops to help with debugging."""

from __future__ import annotations

from max._core.dialects import mo

from ..graph import Graph
from ..type import DeviceRef, _ChainType
from ..value import TensorValue


def print(value: str | TensorValue, label: str = "debug_tensor") -> None:
    """Prints the value of a tensor or a string during graph execution.

    This function is used to output the current value of a tensor and is
    primarily used for debugging purposes within the context of the Max
    Engine and its graph execution framework. This is particularly useful to
    verify the intermediate results of your computations are as expected.

    By printing the tensor values, you can visualize the data flowing through the
    graph, which helps in understanding how the operations are transforming
    the data.

    Pass a ``label`` to identify which tensor's value is being printed,
    especially when there are multiple print statements in a complex graph.

    .. code-block:: python

        import numpy as np
        from max.driver import CPU
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        input_type = TensorType(DType.float32, (2,), device=DeviceRef.CPU())
        with Graph("print_example", input_types=(input_type, input_type)) as graph:
            lhs, rhs = graph.inputs
            out = ops.add(lhs, rhs)
            ops.print(out, label="addition_output")
            graph.output(out)

        model = InferenceSession(devices=[CPU()]).load(graph)
        result = model.execute(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        )[0]
        # During execution, prints:
        # addition_output = tensor([[4.0000, 6.0000]], dtype=f32, shape=[2])

    .. invisible-code-block: python

        np.testing.assert_allclose(result.to_numpy(), [4.0, 6.0])

    Args:
        value: The value to print. Can be either a string or a TensorValue.
        label: A label to identify the printed value. Defaults to
          ``debug_tensor``.
    """
    in_chain = Graph.current.device_chains[DeviceRef.CPU()]
    out_chain_type = _ChainType()

    if isinstance(value, str):
        output = Graph.current._add_op_generated(
            mo.DebugPrintOp,
            out_chain=out_chain_type,
            in_chain=in_chain,
            value=value,
            label=label,
        )[0]
    else:
        output = Graph.current._add_op_generated(
            mo.DebugTensorPrintOp,
            out_chain=out_chain_type,
            in_chain=in_chain,
            input=value,
            label=label,
        )[0]
    Graph.current.device_chains[DeviceRef.CPU()] = output
