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
"""Op implementation for mo.parallel."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

from max._core import Block as _CBlock
from max._core import OpBuilder
from max._core import Value as _CValue
from max._core import graph as _graph
from max._core.dialects import kgen
from max._core.dialects.mo import (
    BundleType,
    ChainType,
    ParallelOp,
    TensorBundleOp,
    TensorUnbundleOp,
    YieldOp,
)

from ..graph import Graph, _location
from ..value import (
    BufferValue,
    BufferValueLike,
    TensorType,
    TensorValue,
    TensorValueLike,
    Value,
    _ChainValue,
)


def parallel(
    inputs: Sequence[Sequence[TensorValueLike]],
    body_fn: Callable[
        ...,
        TensorValue
        | Iterable[TensorValue]
        | tuple[TensorValue | Iterable[TensorValue], _ChainValue],
    ],
    *,
    buffers: Iterable[BufferValueLike] | None = None,
    chain: _ChainValue | None = None,
    result_types: Sequence[Sequence[TensorType]],
) -> list[list[TensorValue]] | tuple[list[list[TensorValue]], _ChainValue]:
    """Execute a function in parallel for each launch via ``mo.parallel``.

    Each input bundle holds one ``TensorValue`` per launch.  All bundles
    must have the same launch count.  The body receives one representative
    ``TensorValue`` per input bundle (typed like the bundle's first launch)
    and yields one ``TensorValue`` per output bundle; the runtime
    re-dispatches the body across all launches.

    When ``buffers`` are provided (e.g. signal buffers for bundled
    collectives), the body receives an additional ``BufferValue`` argument
    after the input-bundle representatives.  Buffers are flat (one per
    launch) and not bundled.

    When ``chain`` is provided, the parallel region is sequenced relative
    to prior ops and the returned ``out_chain`` represents completion of
    all parallel launches.  The in-chain enters the body through a trailing
    ``_ChainValue`` argument (after the bundle representatives and the
    buffer), and the body must thread it and return the resulting out-chain
    as the second element of a ``(tensors, out_chain)`` tuple.

    Args:
        inputs: Per-bundle, per-launch tensors.  Each inner sequence is
            one bundle's launches; all bundles must share the same launch
            count and per-launch device labels.
        body_fn: Callable receiving one ``TensorValue`` per input bundle
            (then optionally one ``BufferValue`` for ``buffers`` and one
            ``_ChainValue`` for ``chain``).  A chainless body returns one
            ``TensorValue`` per output bundle; a chained body returns
            ``(tensors, out_chain)``.
        buffers: Optional per-launch buffer values (one per launch).
        chain: Optional chain value for sequencing.
        result_types: Per-output-bundle, per-launch result types.

    Returns:
        ``[[t0, t1, ...], ...]`` per output bundle; if ``chain`` is
        provided, returns ``(results, out_chain)``.
    """
    bundle_inputs: list[list[TensorValue]] = [
        [TensorValue(v) for v in bundle] for bundle in inputs
    ]
    if not bundle_inputs:
        raise ValueError("parallel requires at least one input bundle")
    num_launches = len(bundle_inputs[0])
    if num_launches == 0:
        raise ValueError("each input bundle must have at least one launch")
    for i, b in enumerate(bundle_inputs[1:], start=1):
        if len(b) != num_launches:
            raise ValueError(
                f"input bundle {i} has {len(b)} launches; bundle 0 has "
                f"{num_launches}"
            )

    output_bundles: list[Sequence[TensorType]] = list(result_types)
    for i, ob in enumerate(output_bundles):
        if len(ob) != num_launches:
            raise ValueError(
                f"result_types[{i}] has {len(ob)} types; expected "
                f"{num_launches} (one per launch)"
            )

    buffer_inputs: list[BufferValue] | None = None
    if buffers is not None:
        buffer_inputs = [BufferValue(v) for v in buffers]
        if len(buffer_inputs) != num_launches:
            raise ValueError(
                f"buffers length ({len(buffer_inputs)}) must match "
                f"launch count ({num_launches})"
            )
        if chain is None:
            raise ValueError("chain is required when buffers are provided")

    if chain is not None and buffer_inputs is None:
        raise ValueError("buffers are required when chain is provided")

    graph = Graph.current

    parallel_result_types: list[BundleType | ChainType] = [
        BundleType([t.to_mlir() for t in b]) for b in output_bundles
    ]
    if chain is not None:
        parallel_result_types.append(ChainType())

    # Defer verification until the body has a yield terminator.
    with graph._pause_verification():
        with _location() as loc:
            builder = OpBuilder(_CBlock._from_cmlir(graph._current_block).end)
            bundle_values = [
                TensorBundleOp(
                    builder,
                    loc,
                    [t._mlir_value for t in b],  # type: ignore[misc]
                ).results[0]
                for b in bundle_inputs
            ]
            if chain is not None:
                assert buffer_inputs is not None
                ParallelOp(
                    builder,
                    loc,
                    inputs=bundle_values,
                    buffers=[b._mlir_value for b in buffer_inputs],  # type: ignore[misc]
                    in_chain=chain._mlir_value,
                    result_types=parallel_result_types,
                )
            else:
                ParallelOp(
                    builder,
                    loc,
                    inputs=bundle_values,
                    result_types=parallel_result_types,
                )

        # The typed-core builder doesn't return an OpView, so look it up.
        parallel_op = _graph.last_operation(graph._current_block).opview
        body_block = parallel_op.bodyRegion.blocks[0]

        with graph._block(body_block):
            block_args = [
                Value.from_mlir(_CValue._from_cmlir(arg))
                for arg in body_block.arguments
            ]
            # The body block has one representative arg per input bundle,
            # followed by a trailing !mo.chain arg when the parallel is chained.
            # The in-chain enters the body through that arg, the body threads it
            # through its buffer ops, and yields the result as the terminator's
            # last operand.
            n_bundles = len(bundle_inputs)
            bundle_args = block_args[:n_bundles]
            chain_arg = block_args[n_bundles] if chain is not None else None

            call_args: list[Value[Any]] = list(bundle_args)
            if buffer_inputs is not None:
                call_args.append(buffer_inputs[0])
            if chain_arg is not None:
                call_args.append(chain_arg)
            body_result = body_fn(*call_args)

            # A chained body returns ``(tensors, out_chain)``; a chainless body
            # returns just ``tensors``.
            out_chain_value: Value[Any] | None = None
            if chain is not None:
                # A chained body always yields a (tensors, out_chain) pair.
                assert isinstance(body_result, tuple) and len(body_result) == 2
                body_tensors, out_chain_value = body_result
            else:
                # A chainless body returns the tensors directly; the chained
                # 2-tuple form is excluded by the chain-is-None contract.
                body_tensors = cast(
                    "TensorValue | Iterable[TensorValue]", body_result
                )

            if isinstance(body_tensors, TensorValue):
                output_tensors = [body_tensors]
            else:
                output_tensors = list(body_tensors)

            if len(output_tensors) != len(output_bundles):
                raise ValueError(
                    f"parallel body yielded {len(output_tensors)} tensor(s), "
                    f"expected {len(output_bundles)} (one per output bundle)"
                )

            yield_operands: list[Value[Any]] = list(output_tensors)
            if out_chain_value is not None:
                yield_operands.append(out_chain_value)

            graph._add_op_generated(
                YieldOp,
                operands=yield_operands,
                parameters=kgen.ParameterExprArrayAttr([]),
            )

    graph._verify_op(parallel_op)

    tensor_results: list[list[TensorValue]] = []
    for i in range(len(output_bundles)):
        unbundled = graph._add_op_generated(
            TensorUnbundleOp,
            _CValue._from_cmlir(parallel_op.results[i]),
        )
        tensor_results.append([v.tensor for v in unbundled])

    if chain is not None:
        out_chain = _ChainValue(_CValue._from_cmlir(parallel_op.results[-1]))  # type: ignore[arg-type]
        return tensor_results, out_chain

    return tensor_results
