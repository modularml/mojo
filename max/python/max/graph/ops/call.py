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
"""Op implementation for calling a graph."""

from __future__ import annotations

from typing import Any

from max.mlir.dialects import mo

from ..graph import Graph
from ..type import DeviceRef, _ChainType
from ..value import Value, _ChainValue


def call(graph: Graph, *args: Value[Any], prefix: str = "") -> list[Value[Any]]:
    """Calls a previously defined graph with the provided arguments.

    Use this function to invoke a subgraph built with
    :meth:`~max.graph.Graph.add_subgraph` or
    :meth:`~max.nn.Module.build_subgraph`. The primary benefit is that the
    compiler processes the subgraph definition once, which reduces compile
    time significantly for models with repeated blocks.

    Examples:
        Call a subgraph and forward its outputs to the parent graph:

        .. code-block:: python

            from max.dtype import DType
            from max.graph import Graph, ops
            from max.graph.type import TensorType, DeviceRef

            input_type = TensorType(DType.float32, [10], DeviceRef.CPU())

            with Graph("main", input_types=[input_type]) as graph:
                with graph.add_subgraph(
                    "add_one", input_types=[input_type]
                ) as sub:
                    x = sub.inputs[0].tensor
                    one = ops.constant(1, DType.float32, device=DeviceRef.CPU())
                    sub.output(ops.elementwise.add(x, one))

                result = ops.call(sub, graph.inputs[0])
                graph.output(*result)

        Call a shared subgraph for each layer of a model, resolving
        different weights at each call site with ``prefix``. Build the
        subgraph once from the first layer, then call it once per layer,
        passing the matching weight prefix so each call resolves that
        layer's weights:

        .. code-block:: python

            import numpy as np
            from max.driver import CPU
            from max.dtype import DType
            from max.engine import InferenceSession
            from max.graph import DeviceRef, Graph, TensorType, ops
            from max.nn import LayerList, Linear, Module

            class Model(Module):
                def __init__(self):
                    super().__init__()
                    self.layers = LayerList([
                        Linear(2, 2, DType.float32, DeviceRef.CPU()),
                        Linear(2, 2, DType.float32, DeviceRef.CPU()),
                    ])

                def __call__(self, x):
                    return x

            # Two layers with distinct weights: layer 0 doubles its input,
            # layer 1 negates its input.
            weights = {
                "layers.0.weight": np.array(
                    [[2.0, 0.0], [0.0, 2.0]], dtype=np.float32
                ),
                "layers.1.weight": np.array(
                    [[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32
                ),
            }

            model = Model()
            # load_state_dict assigns each weight its fully-qualified name,
            # which build_subgraph strips with weight_prefix to create
            # placeholder weights.
            model.load_state_dict(weights)

            input_type = TensorType(DType.float32, [1, 2], DeviceRef.CPU())
            with Graph("shared_block", input_types=[input_type]) as graph:
                x = graph.inputs[0].tensor

                # Build the subgraph once from the first layer.
                subgraph = model.layers[0].build_subgraph(
                    "linear_block",
                    inputs=[x],
                    weight_prefix="layers.0.",
                )

                # Call it once per layer, resolving that layer's weights.
                out0 = ops.call(subgraph, x, prefix="layers.0.")
                out1 = ops.call(subgraph, out0[0].tensor, prefix="layers.1.")
                graph.output(out1[0].tensor)

            session = InferenceSession(devices=[CPU()])
            compiled = session.load(graph, weights_registry=weights)

            result = compiled.execute(
                np.array([[3.0, 5.0]], dtype=np.float32)
            )[0].to_numpy()
            # Layer 0 doubles, then layer 1 negates: 3 -> 6 -> -6, 5 -> 10 -> -10.
            # result: [[-6. -10.]]

        .. invisible-code-block: python

            np.testing.assert_allclose(result, [[-6.0, -10.0]])

    Args:
        graph: The subgraph to call.
        *args: Arguments to pass to the subgraph. Must match the subgraph's
            input types, excluding the chain value (handled internally).
        prefix: A string prepended to all weight names when the subgraph is
            invoked. Use this to distinguish repeated calls to the same
            subgraph. For example, if a transformer block references a weight
            named ``attention.wq``, calling with ``prefix="layers.3."``
            resolves it to ``layers.3.attention.wq`` in the weights registry.
            Leave empty if the subgraph contains no placeholder weights.

    Returns:
        A list of :class:`~max.graph.Value` objects representing the
        subgraph's outputs, excluding any internal chain values.
    """
    # Get the current graph context
    current_graph = Graph.current
    call_args = list(args)  # mutable so we can add a chain
    # Be careful, input_types are type[Value], output_types are Type
    input_types = [type(input) for input in graph.inputs]
    output_types = list(graph.output_types)

    # Mostly leave type checking up to the op builder.
    # We can do some basic type checking to improve error messages,
    # but for instance can't check forward shape propagation correctness.
    if len(call_args) != len(input_types):
        raise ValueError(
            f"Expected {len(input_types)} args to call to {graph.name}, got {len(call_args)}. "
            f"\n    {graph.name}{tuple(input_types)}"
        )

    # Plumb chains through the call: the callee declared a specific set of
    # device chains in its signature, so we pull the corresponding chain
    # values from the caller's ``device_chains`` and add matching chain
    # result types. Caller's and callee's chain sets may differ, so we
    # can't use ``device_chains.pack``/``unpack`` here.
    chain_devices: tuple[DeviceRef, ...] = ()
    if graph._has_chain_input:
        chain_devices = tuple(graph.device_chains)
        call_args.extend(
            current_graph.device_chains[device] for device in chain_devices
        )
        output_types.extend(_ChainType() for _ in chain_devices)

    # TODO: migrate to _add_op_generated(mo.CallOp, ...). Blocked on
    # max._core.dialects.builtin.FlatSymbolRefAttr having no Python
    # constructor bound (see builtin.cpp); the legacy mo.call_ factory
    # builds it for us from the str callee.
    call_results = current_graph._add_op(
        mo.call_,
        callee=graph.name,
        results=output_types,
        operands=call_args,
        prefix=prefix,
    )

    if not chain_devices:
        return call_results

    chain_count = len(chain_devices)
    for device, chain in zip(
        chain_devices, call_results[-chain_count:], strict=True
    ):
        assert isinstance(chain, _ChainValue)
        current_graph.device_chains[device] = chain
    return call_results[:-chain_count]
