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

"""MO graph interpreter for eager execution.

This module provides an interpreter for MO (Modular Operation) graphs that can
execute operations directly without going through the full compilation pipeline.
This is useful for eager mode execution where compilation latency needs to be
minimized.

The interpreter walks through the MO graph in topological order and dispatches
each operation to an appropriate handler. Handlers can implement operations
using NumPy or by calling into Mojo kernels.

Example usage:
    from max import _interpreter

    outputs = _interpreter.execute(graph, input_buffers)
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from max import _core
from max._core.dialects import builtin, kgen, mo, mosh
from max.driver import Buffer
from max.graph import Graph

try:
    # Importing registers all op handlers as a side effect.
    from ._interpreter_ops import (  # type: ignore[import-not-found]
        lookup_handler,
    )
except Exception as _e:
    import os as _os
    import subprocess as _sp
    import sys as _sys
    import traceback as _tb

    _pkg = _os.path.dirname(_os.path.abspath(__file__))
    _ops_dir = _os.path.join(_pkg, "_interpreter_ops")
    try:
        _ops_files = sorted(
            f"{e.name} ({e.stat().st_size}B)" for e in _os.scandir(_ops_dir)
        )
    except OSError:
        _ops_files = ["(unable to list)"]
    _ldd_lines = []
    for _so in _os.scandir(_ops_dir) if _os.path.isdir(_ops_dir) else []:
        if not _so.name.endswith(".so"):
            continue
        try:
            _r = _sp.run(
                ["ldd", _so.path], capture_output=True, text=True, timeout=5
            )
            _missing = [
                l.strip() for l in _r.stdout.splitlines() if "not found" in l
            ]
            if _missing:
                _ldd_lines.append(f"    {_so.name}: {_missing}")
        except Exception:
            pass
    _ldd_out = (
        "\n  ldd missing deps:\n" + "\n".join(_ldd_lines) if _ldd_lines else ""
    )
    print(
        f"max._interpreter: failed to import _interpreter_ops:"
        f" {type(_e).__name__}: {_e}\n"
        f"  agent: {_os.environ.get('BUILDKITE_AGENT_NAME', '(not set)')}\n"
        f"  build: {_os.environ.get('BUILDKITE_BUILD_ID', '(not set)')}"
        f" job: {_os.environ.get('BUILDKITE_JOB_ID', '(not set)')}\n"
        f"  package dir: {_pkg}\n"
        f"  _interpreter_ops contents: {_ops_files}"
        f"{_ldd_out}\n"
        f"{_tb.format_exc()}",
        file=_sys.stderr,
    )
    raise

# Type alias for interpreter slots
InterpreterSlots = dict[Any, Buffer | None]

# Op names that always require the compiled execution path.
# Name-based matching is used (like _is_dispatchable and the handler
# name-fallback) because nanobind may create different class objects
# for the same MLIR op.
_COMPILATION_REQUIRED_OP_NAMES: tuple[str, ...] = ("CustomOp",)


def _validate_inputs(graph: Graph, inputs: Sequence[Buffer]) -> None:
    """Validate input buffers match graph expectations.

    Args:
        graph: The graph being executed.
        inputs: Input buffers provided by caller.

    Raises:
        ValueError: If inputs don't match graph expectations.
    """
    graph_inputs = list(graph.inputs)
    if len(graph_inputs) != len(inputs):
        raise ValueError(
            f"Expected {len(graph_inputs)} inputs, got {len(inputs)}"
        )
    # TODO(EMF-93): Add dtype/shape validation once we have more complete
    # tensor type extraction. The MO type system provides dtype and
    # shape_attr but extracting static shapes requires handling
    # symbolic dimensions.


def can_execute(graph: Graph, max_ops: int | None = None) -> bool:
    """Check whether the interpreter can handle this graph.

    Scans the graph for ops that require compilation (e.g. ``CustomOp``)
    and for ops without a registered handler.  Optionally enforces a
    maximum dispatchable-op count so that large graphs still go through
    the graph compiler where fusion is beneficial.

    Args:
        graph: The graph to check.
        max_ops: If set, the maximum number of dispatchable ops the
            interpreter will accept.  Graphs exceeding this threshold
            return ``False`` so the graph compiler can apply fusion
            optimizations.  ``None`` means no limit.

    Returns:
        True if the interpreter can handle the graph, False otherwise.
    """
    module = graph._module
    dispatchable_count = 0
    for op in _walk_ops(module):
        if isinstance(op, mo.OutputOp):
            continue
        if type(op).__name__ in _COMPILATION_REQUIRED_OP_NAMES:
            return False
        if lookup_handler(op) is None:
            return False
        dispatchable_count += 1
        if max_ops is not None and dispatchable_count > max_ops:
            return False
    return True


def execute(
    graph: Graph,
    inputs: Sequence[Buffer],
) -> Sequence[Buffer | None]:
    """Execute an MO graph and return output buffers.

    Args:
        graph: The finalized MO graph to execute.
        inputs: Input buffers corresponding to graph.inputs.

    Returns:
        List of output buffers.

    Raises:
        ValueError: If inputs don't match graph expectations.
        RuntimeError: If output value was not computed.
        NotImplementedError: If an operation has no handler.
    """
    # Create a new interpreter slots dictionary for this execution.
    slots: InterpreterSlots = {}

    # Validate inputs before execution
    _validate_inputs(graph, inputs)

    # Map graph inputs to their buffers
    for graph_input, buffer in zip(graph.inputs, inputs, strict=True):
        slots[graph_input._mlir_value] = buffer

    # Walk ops in the graph body and dispatch
    module = graph._module
    output_op = None
    for op in _walk_ops(module):
        if isinstance(op, mo.OutputOp):
            output_op = op
        else:
            _dispatch_op(op, slots)

    # Collect outputs from the mo.output terminator
    if output_op is None:
        raise RuntimeError("Graph has no output terminator")
    outputs = []
    for operand in output_op.operands:
        try:
            outputs.append(slots[operand])
        except RuntimeError as e:
            raise RuntimeError(f"Output value not computed: {operand}") from e
    return outputs


def _walk_ops(module: builtin.ModuleOp) -> Iterator[_core.Operation]:
    """Walk operations in a valid execution order.

    Args:
        module: The MLIR module operation.

    Returns:
        Generator of dispatchable operations in execution order.
    """

    # MO graphs have the structure:
    # builtin.module -> mo.graph -> Region 0 -> Block 0 -> operations
    # SSA form guarantees operations are already in valid execution order.
    for top_level_op in module.body:
        if isinstance(top_level_op, mo.GraphOp):
            block = top_level_op.regions[0].front
            for op in block:
                if _is_dispatchable(op):
                    yield op


def _is_dispatchable(op: _core.Operation) -> bool:
    """Check if an operation should be dispatched or collected.

    Skip function definitions and other structural ops.
    OutputOp is included so we can extract outputs from it.

    Args:
        op: The operation to check.

    Returns:
        True if the operation should be processed, False otherwise.
    """
    skip_types = (
        mo.ChainCreateOp,  # Sequencing (interpreter executes sequentially)
        kgen.ParamDeclareOp,  # Shape parameter declarations
        mosh.ParamFromValueOp,  # Records values into params (not needed)
    )
    if isinstance(op, skip_types):
        return False

    # TODO(EMF-104): Check type for these
    skip_names = (
        "ParamConstantOp",  # Constant parameter declarations
    )

    return type(op).__name__ not in skip_names


def _dispatch_op(op: _core.Operation, slots: dict[Any, Buffer | None]) -> None:
    """Dispatch a single MO operation to its handler.

    Args:
        op: The operation to dispatch.
        slots: The interpreter slots.

    Raises:
        NotImplementedError: If no handler exists for the operation.
    """
    # Check handler registry
    if (handler := lookup_handler(op)) is not None:
        # Operation.operands returns OpOperand, use .value to get the Value.
        # Use .get() with default None for chain values (ChainCreateOp is
        # skipped, so chain values are not stored in slots)
        input_buffers = [slots.get(operand.value) for operand in op.operands]
        outputs = handler(op, input_buffers)
    else:
        raise NotImplementedError(f"No handler for op: {type(op).__name__}")

    # Store outputs
    for result, output_buf in zip(op.results, outputs, strict=True):
        slots[result] = output_buf
