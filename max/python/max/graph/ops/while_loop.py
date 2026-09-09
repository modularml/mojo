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
"""Op implementation for while loop."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from max._core import Value as _Value
from max.mlir.dialects import mo

from ..graph import Graph, GraphBlock
from ..type import DeviceRef
from ..value import BufferValue, TensorValue, Value
from .support import as_iterable, as_values


def while_loop(
    initial_values: Iterable[Value[Any]] | Value[Any],
    predicate: Callable[..., TensorValue],
    body: Callable[..., Value[Any] | Iterable[Value[Any]]],
) -> list[TensorValue]:
    """Repeatedly executes a body function while a predicate holds.

    Both the predicate and body take the same number and types of
    arguments as the initial values. The predicate must return a single
    boolean scalar tensor of type :attr:`~max.dtype.DType.bool` that controls
    loop continuation. The body must return updated values matching the types
    of the initial value(s).

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()

        def predicate(x):
            return x < 10

        def body(x):
            return x + 1

        with Graph("while_loop_example") as graph:
            x = ops.constant(0, DType.int32, device=device)
            graph.output(*ops.while_loop(x, predicate, body))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    .. invisible-code-block: python

        assert int(result.to_numpy()) == 10

    Args:
        initial_values: The initial values for the loop arguments. Must be
            non-empty.
        predicate: A callable that takes the loop arguments and returns a
            boolean scalar tensor of type :attr:`~max.dtype.DType.bool`
            determining loop continuation.
        body: A callable that takes the loop arguments and returns updated
            values matching the types of ``initial_values``.

    Returns:
        The output values from the final loop iteration.

    Raises:
        ValueError: If ``initial_values`` is empty.
        NotImplementedError: If any initial value is a buffer rather than a
            tensor. Buffer operations are not currently supported.
    """
    initial_values = (
        list(initial_values)
        if isinstance(initial_values, Iterable)
        else [initial_values]
    )
    if not initial_values:
        raise ValueError("While loops must have at least one iteration value.")
    if any(isinstance(arg, BufferValue) for arg in initial_values):
        raise NotImplementedError(
            "Buffer operations are currently not supported in while loops"
        )

    graph = Graph.current

    # Loop-carried values are: the user's initial values, then the per-device
    # chains (including the host chain at ``DeviceRef.CPU()``). The same
    # shape is what the cond/body blocks receive as block arguments and what
    # they yield.
    carried = graph.device_chains.pack(initial_values)
    carried_types = [v.type for v in carried]

    def adopt_block_args(args: Iterable[Any]) -> list[Value[Any]]:
        """Rebind graph chain state from the block's arguments.

        Returns the user-visible loop variables.
        """
        all_args = [Value.from_mlir(_Value._from_cmlir(a)) for a in args]
        return graph.device_chains.unpack(all_args)

    def while_condition_terminator(args: list[Any]):  # noqa: ANN202
        """Build ``mo.while.condition`` from a flat ``[cond, *yielded]`` arg list.

        ``mo.while.condition`` takes its first operand as the condition and
        the rest as yielded values, but ``GraphBlock.output`` passes
        everything as one flat list.
        """
        condition, *yielded = args
        return mo.WhileConditionOp(condition, yielded)

    def fold_new_chains(live_on_entry: set[DeviceRef]) -> None:
        new_devices = set(graph.device_chains) - live_on_entry
        graph.device_chains.merge_for(new_devices)
        for device in new_devices:
            del graph.device_chains[device]

    with GraphBlock(arg_types=carried_types) as pred_block:
        live_on_entry = set(graph.device_chains)
        loop_vars = adopt_block_args(pred_block.arguments)
        condition = predicate(*loop_vars)
        if not condition.device.is_cpu():
            raise ValueError(
                "The predicate for `ops.while_loop` must reside on CPU,"
                f" but got a tensor on {condition.device}. Transfer it"
                " explicitly with `ops.transfer_to(pred, CPU())`."
            )
        fold_new_chains(live_on_entry)
        pred_block.output(
            *graph.device_chains.pack([condition, *loop_vars]),
            terminator=while_condition_terminator,
        )

    user_carry_types = [v.type for v in initial_values]
    with GraphBlock(arg_types=carried_types) as body_block:
        live_on_entry = set(graph.device_chains)
        loop_vars = adopt_block_args(body_block.arguments)
        body_results = as_values(
            as_iterable(body(*loop_vars)), user_carry_types
        )
        fold_new_chains(live_on_entry)
        body_block.output(*graph.device_chains.pack(body_results))

    # The body's output types are what mo.while yields (loop vars + chains);
    # the condition block's first operand is the bool, so its output_types
    # would be wrong here.
    results = graph._add_op(
        mo.while_,
        body_block.output_types,
        carried,
        cond_block=pred_block.mlir_block,
        body_block=body_block.mlir_block,
    )
    return graph.device_chains.unpack(results)
