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
"""Transformation passes over a MAX Graph."""

from typing import Any

from max import _core as mlir
from max._core.dialects import builtin, kgen, mo
from max.graph import Graph, Type, Value
from max.graph.graph import _location


def _graph_block(graph: Graph) -> mlir.Block:
    op = mlir.Operation._from_cmlir(graph._mlir_op)
    assert isinstance(op, mo.GraphOp)
    return op.regions[0].front


def _fixup_graph(graph: Graph) -> None:
    """Updates the mo.GraphOp and Graph objects to match the internal graph
    structure.

    Assumes that the graph block definition is correct, and infers the remaining
    properties from there:
        - Graph.inputs
        - Graph._params
        - op.input_parameters
        - op.result_parameters
        - op.function_type
        - op.signature
        - op metadata: argument names
    """
    op = mlir.Operation._from_cmlir(graph._mlir_op)
    assert isinstance(op, mo.GraphOp)
    block = op.regions[0].front

    with graph:
        # - use block.arguments as the source of truth for inputs
        inputs = [Value.from_mlir(arg) for arg in block.arguments]
        if isinstance(output_op := block[-1], mo.OutputOp):
            results = [Value.from_mlir(o) for o in output_op.operands]
        else:
            results = []

        # - reset graph.inputs
        graph.inputs = inputs

        # - reset op.input_parameters
        input_params = {
            dim: None
            for input in inputs
            for dim in getattr(input.type, "parameters", ())
        }
        si64 = kgen.SIMDType(1, kgen._KGENDType.get_int(64, True))
        op.input_parameters = kgen.ParamDeclArrayAttr(
            [kgen.ParamDeclAttr(str(dim), si64) for dim in input_params]
        )
        # - update graph._params
        graph._params.update(input_params)
        # - update argument names
        op.discardable_attributes["argument_names"] = builtin.ArrayAttr(
            [builtin.StringAttr(f"input{i}") for i in range(len(graph.inputs))]
        )

        # Parameters already declared by a body op (e.g. data-dependent
        # output dims like NMS's num_selected) must not be redeclared as
        # graph result parameters: KGEN allows exactly one declaration per
        # scope.
        body_declared = {
            decl.name.value
            for body_op in block
            for decl in getattr(body_op, "output_param_decls", ())
        }
        result_params = {
            dim: None
            for result in results
            for dim in getattr(result.type, "parameters", ())
            if dim not in input_params and str(dim) not in body_declared
        }
        op.result_parameters = kgen.ParamDeclArrayAttr(
            [kgen.ParamDeclAttr(str(dim), si64) for dim in result_params]
        )

        # - reset op.function_type
        op.function_type = builtin.FunctionType(  # type: ignore
            [input.type.to_mlir() for input in inputs],
            [result.type.to_mlir() for result in results],
        )
        # - reset op.signature
        op.signature = kgen.FuncTypeGeneratorType([], op.function_type)  # type: ignore


def add_input(graph: Graph, type: Type[Any]) -> Value[Any]:
    """Adds a new input to an existing graph.

    Args:
        graph: The graph to which to add the new input
        type: The type of the new input to add
    Returns:
        The Value associated with the new input
    """
    block = _graph_block(graph)

    with graph:
        block.add_argument(type.to_mlir(), _location())

    _fixup_graph(graph)
    return graph.inputs[-1]


def remove_unused_arguments(graph: Graph) -> None:
    """Removes any unused arguments from the graph.

    Args:
        graph: The graph on which to apply the pass
    """
    block = _graph_block(graph)

    # reverse so indices don't during iteration+mutation
    for i, arg in reversed(list(enumerate(block.arguments))):
        if not arg.num_uses:
            block.erase_argument(i)

    _fixup_graph(graph)
