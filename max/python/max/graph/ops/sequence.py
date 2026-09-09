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
"""Op implementation for mo.sequence (side-stream execution)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from max._core import Block as _CBlock
from max._core import OpBuilder
from max._core import Value as _CValue
from max._core import graph as _graph
from max._core.dialects import kgen
from max._core.dialects.mo import SequenceOp, YieldOp

from ..graph import Graph, _location
from ..type import TensorType
from ..value import TensorValue, TensorValueLike, Value

__all__ = ["side_stream"]


def side_stream(
    inputs: Sequence[TensorValueLike],
    body_fn: Callable[..., TensorValue | Iterable[TensorValue]],
    *,
    result_types: Sequence[TensorType],
    stream_id: int = 1,
) -> list[TensorValue]:
    """Run a block of ops on a side device stream via ``mo.sequence``.

    The body executes on the device stream selected by ``stream_id`` (0 is the
    default stream), overlapping independent work on the main stream. Inputs map
    1:1 to the body's block arguments and the body returns one value per
    ``result_types`` entry. The graph compiler binds the whole body to a
    side-stream device-context view and inserts the cross-stream synchronization
    at the boundary, so callers never manage streams or events directly.

    Args:
        inputs: Values passed into the body, mapped 1:1 to block arguments.
        body_fn: Callable receiving one value per input and returning one value
            per ``result_types`` entry.
        result_types: The body's output types (one per result).
        stream_id: Device stream to run the body on (0 is the default stream).

    Returns:
        One ``TensorValue`` per ``result_types`` entry.
    """
    in_vals = [TensorValue(v) for v in inputs]
    out_types = list(result_types)

    graph = Graph.current
    seq_result_types = [t.to_mlir() for t in out_types]

    # Defer verification until the body has a yield terminator.
    with graph._pause_verification():
        with _location() as loc:
            builder = OpBuilder(_CBlock._from_cmlir(graph._current_block).end)
            SequenceOp(
                builder,
                loc,
                inputs=[v._mlir_value for v in in_vals],  # type: ignore[misc]
                result_types=seq_result_types,
                stream_id=stream_id,
            )

        # The typed-core builder doesn't return an OpView, so look it up.
        seq_op = _graph.last_operation(graph._current_block).opview
        body_block = seq_op.bodyRegion.blocks[0]

        with graph._block(body_block):
            block_args = [
                Value.from_mlir(_CValue._from_cmlir(arg))
                for arg in body_block.arguments
            ]
            body_result = body_fn(*block_args)
            if isinstance(body_result, TensorValue):
                body_result = [body_result]
            else:
                body_result = list(body_result)

            if len(body_result) != len(out_types):
                raise ValueError(
                    f"side_stream body yielded {len(body_result)} tensor(s), "
                    f"expected {len(out_types)}"
                )

            graph._add_op_generated(
                YieldOp,
                operands=body_result,
                parameters=kgen.ParameterExprArrayAttr([]),
            )

    graph._verify_op(seq_op)

    return [
        Value.from_mlir(_CValue._from_cmlir(seq_op.results[i])).tensor
        for i in range(len(out_types))
    ]
