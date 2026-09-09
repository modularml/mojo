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
"""Tests for `max.experimental._passes`."""

from collections.abc import Iterator

import pytest
from max import _core
from max._core.dialects import mo, rmo
from max.driver import CPU
from max.dtype import DType
from max.engine.api import InferenceSession
from max.experimental import _passes, support
from max.graph import Graph, TensorType, ops


@pytest.fixture
def session() -> Iterator[InferenceSession]:
    yield support._session()


def test_add_input(session: InferenceSession) -> None:
    type = TensorType(DType.float32, ["a", "b"], CPU())
    with Graph("test_add_input", input_types=[]) as graph:
        graph.output()

    # Basic checks: the updated graph has the right input,
    #  still compiles, and the resulting model
    _passes.add_input(graph, type)
    assert len(graph.inputs) == 1
    assert graph.inputs[0].type == type
    model = session.load(graph)
    assert len(model.input_metadata) == 1


def test_fixup_does_not_redeclare_body_op_params(
    session: InferenceSession,
) -> None:
    """A parameter declared by a body op (a data-dependent output dim) must
    not be redeclared as a graph result parameter: KGEN permits exactly one
    declaration per scope, so a redeclaration breaks both legalization and
    compilation of the graph."""
    input_type = TensorType(DType.float32, [4], CPU())
    with Graph("nonzero", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        # ops.nonzero declares its data-dependent output dim on the op.
        graph.output(ops.nonzero(x.tensor, "nnz"))

    _passes.remove_unused_arguments(graph)

    op = _core.Operation._from_cmlir(graph._mlir_op)
    assert isinstance(op, mo.GraphOp)
    declared = {
        str(decl.name).strip('"') for decl in op.result_parameters or ()
    }
    assert "nnz" not in declared

    # The graph must still legalize (twice -- the pass is idempotent) and
    # compile.
    _core.lower(graph._module, [rmo.passes.LegalizeRMOOps()])
    _core.lower(graph._module, [rmo.passes.LegalizeRMOOps()])
    model = session.load(graph)
    assert len(model.input_metadata) == 1


def test_remove_unused_arguments(session: InferenceSession) -> None:
    type_a = TensorType(DType.float32, ["a"], CPU())
    type_b = TensorType(DType.float32, ["b"], CPU())
    with Graph("test_add_input", input_types=[type_a, type_b]) as graph:
        _, b = graph.inputs
        graph.output(b)

    # Basic checks: the updated graph has the right input,
    #  still compiles, and the resulting model
    _passes.remove_unused_arguments(graph)
    assert len(graph.inputs) == 1
    assert graph.inputs[0].type == type_b

    model = session.load(graph)
    assert len(model.input_metadata) == 1
