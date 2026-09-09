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
from typing import NoReturn

from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops


def test_conditional_no_results() -> None:
    with Graph("conditional", input_types=()) as graph:
        cond = ops.constant(True, dtype=DType.bool, device=DeviceRef.CPU())

        def then_fn() -> None:
            ops.print("then")

        def else_fn() -> None:
            ops.print("else")

        ops.cond(cond, None, then_fn, else_fn)
        graph.output()

    # Verify both branches are present in MLIR
    mlir_str = str(graph._mlir_op)
    assert "then" in mlir_str
    assert "else" in mlir_str


def test_conditional_with_results() -> None:
    # Test conditional with return values
    with Graph("conditional_with_returns", input_types=()) as graph:
        cond = ops.constant(True, dtype=DType.bool, device=DeviceRef.CPU())

        def then_fn():  # noqa: ANN202
            return ops.constant(1, DType.int32, device=DeviceRef.CPU())

        def else_fn():  # noqa: ANN202
            return ops.constant(0, DType.int32, device=DeviceRef.CPU())

        result = ops.cond(
            cond,
            [TensorType(DType.int32, shape=[], device=DeviceRef.CPU())],
            then_fn,
            else_fn,
        )
        graph.output(result[0])

    assert "1" in str(graph._mlir_op)
    assert "0" in str(graph._mlir_op)


def test_conditional_type_check() -> None:
    # Test type checking between branches
    with Graph("conditional_type_check", input_types=()) as graph:
        cond = ops.constant(False, dtype=DType.bool, device=DeviceRef.CPU())

        def then_fn():  # noqa: ANN202
            return ops.constant(1.0, DType.float32, device=DeviceRef.CPU())

        def else_fn():  # noqa: ANN202
            return ops.constant(0, DType.int32, device=DeviceRef.CPU())

        try:
            ops.cond(
                cond,
                [TensorType(DType.float32, shape=[], device=DeviceRef.CPU())],
                then_fn,
                else_fn,
            )
        except TypeError as e:
            assert "Results don't match expected types" in str(e)

        graph.output()

    graph._mlir_op.verify()


def test_conditional_with_raising() -> None:
    with Graph("conditional_with_chain", input_types=()) as graph:
        chain = graph.device_chains[DeviceRef.CPU()]
        cond = ops.constant(True, dtype=DType.bool, device=DeviceRef.CPU())

        def then_fn() -> None:
            return

        def else_fn() -> NoReturn:
            raise Exception("else")

        try:
            ops.cond(cond, None, then_fn, else_fn)
        except Exception as e:
            assert "else" in str(e)

        assert graph.device_chains[DeviceRef.CPU()] == chain
        graph.output()
    graph._mlir_op.verify()


def test_conditional_device_chains_scoped() -> None:
    """Tests that staging ops.cond restores device chains."""
    # Set up two-device inputs and signals so branches may update per-device chains.
    t0 = TensorType(DType.float32, [4], device=DeviceRef.GPU(0))
    t1 = TensorType(DType.float32, [4], device=DeviceRef.GPU(1))
    sb0 = BufferType(DType.int64, [1], device=DeviceRef.GPU(0))
    sb1 = BufferType(DType.int64, [1], device=DeviceRef.GPU(1))

    with Graph(
        "cond_device_chains_scoped", input_types=[t0, t1, sb0, sb1]
    ) as graph:
        x0, x1, s0, s1 = graph.inputs
        x0_t, x1_t = x0.tensor, x1.tensor
        s0_b, s1_b = s0.buffer, s1.buffer

        # Touch device_chains to materialize entries and snapshot ids.
        ch0_before = graph.device_chains[DeviceRef.GPU(0)]
        ch1_before = graph.device_chains[DeviceRef.GPU(1)]
        id0, id1 = id(ch0_before), id(ch1_before)

        pred = ops.constant(True, dtype=DType.bool, device=DeviceRef.CPU())

        def then_fn() -> None:
            # This updates per-device chains inside the cond.
            _ = ops.allreduce.sum([x0_t, x1_t], [s0_b, s1_b])

        def else_fn() -> None:
            return None

        ops.cond(pred, None, then_fn, else_fn)
        # After cond staging, device chains must be restored.
        assert id(graph.device_chains[DeviceRef.GPU(0)]) != id0
        assert id(graph.device_chains[DeviceRef.GPU(1)]) != id1
        graph.output()


def test_conditional_fresh_device_in_branch() -> None:
    t0 = TensorType(DType.bool, [], device=DeviceRef.GPU(0))

    with Graph(
        "test_conditional_fresh_device_in_branch",
        input_types=[t0],
    ) as graph:
        x0 = graph.inputs[0].tensor

        pred = ops.constant(True, dtype=DType.bool, device=DeviceRef.CPU())

        def then_fn() -> None:
            return None

        def else_fn() -> None:
            _ = ops.transfer_to(x0, device=DeviceRef.GPU(1))

        ops.cond(pred, None, then_fn, else_fn)

        graph.output()

    print(graph)


def test_conditional_fresh_device_in_branch_in_subgraph() -> None:
    "Tests that fresh devices inside a conditional don't leak outside of the conditional"
    t0 = TensorType(DType.bool, [], device=DeviceRef.CPU())
    t1 = TensorType(DType.bool, [], device=DeviceRef.GPU(0))

    with Graph(
        "test_conditional_fresh_device_in_branch_in_subgraph",
        input_types=[t0, t1],
    ) as main_graph:
        with main_graph.add_subgraph(
            "subgraph", input_types=[t0, t1]
        ) as subgraph:
            x0 = subgraph.inputs[0].tensor
            x1 = subgraph.inputs[1].tensor

            def then_fn() -> None:
                return None

            def else_fn() -> None:
                _ = ops.transfer_to(x1, device=DeviceRef.GPU(1))

            ops.cond(x0, None, then_fn, else_fn)

            subgraph.output()

        x0 = main_graph.inputs[0].tensor
        x1 = main_graph.inputs[1].tensor
        ops.call(subgraph, x0, x1)
        main_graph.output()


def test_conditional_existing_device_in_branch_in_subgraph() -> None:
    "Tests that conditionals work with subgraphs that have device chains pre-allocated"
    t0 = TensorType(DType.bool, [], device=DeviceRef.CPU())
    t1 = TensorType(DType.bool, [], device=DeviceRef.GPU(0))

    with Graph(
        "test_conditional_fresh_device_in_branch_in_subgraph",
        input_types=[t0, t1],
    ) as main_graph:
        with main_graph.add_subgraph(
            "subgraph", input_types=[t0, t1], devices=[t1.device]
        ) as subgraph:
            x0 = subgraph.inputs[0].tensor
            x1 = subgraph.inputs[1].tensor

            def then_fn() -> None:
                return None

            def else_fn() -> None:
                _ = ops.transfer_to(x1, device=DeviceRef.GPU(1))

            ops.cond(x0, None, then_fn, else_fn)

            subgraph.output()

        x0 = main_graph.inputs[0].tensor
        x1 = main_graph.inputs[1].tensor
        ops.call(subgraph, x0, x1)
        main_graph.output()
