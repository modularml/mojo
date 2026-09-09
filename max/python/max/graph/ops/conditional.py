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
"""Op implementation for conditional."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from max.mlir.dialects import mo

from ..graph import Graph, GraphBlock
from ..type import Type
from ..value import TensorValue, TensorValueLike, Value
from .support import as_iterable, as_values


def cond(
    pred: TensorValueLike,
    out_types: Iterable[Type[Any]] | None,
    then_fn: Callable[
        [],
        Iterable[Value[Any] | TensorValueLike]
        | Value[Any]
        | TensorValueLike
        | None,
    ],
    else_fn: Callable[
        [],
        Iterable[Value[Any] | TensorValueLike]
        | Value[Any]
        | TensorValueLike
        | None,
    ],
) -> list[TensorValue]:
    """Conditionally executes one of two branches based on a boolean predicate.

    Selects between the ``then_fn`` and ``else_fn`` branches based on the
    runtime value of ``pred``. Both branches must return the same number and
    types of values as specified by ``out_types``. Buffer mutations within
    a branch are tracked automatically through the chain mechanism.

    The predicate is evaluated at runtime to determine which branch to run.
    Both branches are compiled, but only the selected branch executes.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()

        def then_fn():
            return ops.constant(1, DType.int32, device=device)

        def else_fn():
            return ops.constant(0, DType.int32, device=device)

        with Graph("cond_example") as graph:
            pred = ops.constant(True, DType.bool, device=device)
            graph.output(
                *ops.cond(
                    pred,
                    [TensorType(DType.int32, [], device=device)],
                    then_fn,
                    else_fn,
                )
            )

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    .. invisible-code-block: python

        assert int(result.to_numpy()) == 1  # pred is True

    Args:
        pred: A boolean scalar tensor of type :attr:`~max.dtype.DType.bool`
            determining which branch to execute.
        out_types: The expected output types for both branches. Use :obj:`None`
            for branches that do not return values (such as buffer mutations).
        then_fn: A callable executed when ``pred`` is ``True``. Must return
            values matching ``out_types`` if ``out_types`` is not :obj:`None`.
        else_fn: A callable executed when ``pred`` is ``False``. Must return
            values matching ``out_types`` if ``out_types`` is not :obj:`None`.

    Returns:
        The output values from the executed branch, or an empty list when
        ``out_types`` is :obj:`None`.

    Raises:
        ValueError: If the branches return different numbers of results or if
            result types don't match ``out_types``.
    """
    pred = TensorValue(pred)
    if not pred.device.is_cpu():
        raise ValueError(
            "The predicate for `ops.cond` must reside on CPU, but got a"
            f" tensor on {pred.device}. Transfer it explicitly with"
            " `ops.transfer_to(pred, CPU())`."
        )

    out_types_list = list(out_types) if out_types is not None else []

    graph = Graph.current

    def _build_branch(block: GraphBlock, fn: Callable[..., Any]) -> None:
        # Snapshot the device set on entry so we can detect new device
        # chains introduced inside the branch and fold them into the host
        # chain before yielding — otherwise the yield's chain count would
        # exceed what the surrounding scope expects.
        live_on_entry = set(graph.device_chains)
        results = as_values(as_iterable(fn()), out_types_list)
        new_devices = set(graph.device_chains) - live_on_entry
        graph.device_chains.merge_for(new_devices)
        for device in new_devices:
            del graph.device_chains[device]
        block.output(*graph.device_chains.pack(results))

    with GraphBlock() as then_block:
        _build_branch(then_block, then_fn)
    with GraphBlock() as else_block:
        _build_branch(else_block, else_fn)

    # Both blocks are fully populated. Now create the mo.if op with them
    # attached; verification runs normally and recursively re-verifies the
    # yields against the real parent.
    results = graph._add_op(
        mo.if_,
        pred,
        then_block.output_types,
        then_block=then_block.mlir_block,
        else_block=else_block.mlir_block,
    )

    return graph.device_chains.unpack(results)
