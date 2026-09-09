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
"""Test while loop operations."""

from typing import NoReturn

from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue, ops


def test_while_loop_basic() -> None:
    """Test basic while loop functionality."""
    with Graph(
        "while_loop_basic",
        input_types=[TensorType(DType.int32, [], device=DeviceRef.CPU())],
    ) as graph:
        x = graph.inputs[0]

        def pred(x: TensorValue) -> TensorValue:
            import sys

            print(f"{pred=}", file=sys.stderr)
            return x < 10

        def body(x: TensorValue) -> TensorValue:
            return x + 1

        results = ops.while_loop(x, pred, body)
        graph.output(results[0])

    # Verify MLIR contains while op and expected structure
    mlir_str = str(graph)
    assert "mo.while" in mlir_str
    assert " do " in mlir_str


def test_while_loop_multiple_args() -> None:
    """Test while loop with multiple arguments."""
    with Graph(
        "while_loop_multiple_args",
        input_types=[
            TensorType(DType.int32, [], device=DeviceRef.CPU()),
            TensorType(DType.int32, [], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, y = graph.inputs

        def pred(x: TensorValue, y: TensorValue) -> TensorValue:
            return x < 10 and y < 10

        def body(x: TensorValue, y: TensorValue) -> list[TensorValue]:
            return [x + 1, y + 1]

        results = ops.while_loop((x, y), pred, body)
        graph.output(results[0], results[1])

    # Verify MLIR contains while op with multiple args
    mlir_str = str(graph)
    assert "mo.while" in mlir_str
    assert " do " in mlir_str


def test_while_loop_empty_init() -> None:
    """Test while loop with empty initial values raises error."""
    with Graph("while_loop_empty_init", input_types=()):
        try:
            ops.while_loop([], lambda: True, list)  # type: ignore
        except ValueError as e:
            assert "While loops must have at least one iteration value" in str(
                e
            )


def test_while_loop_type_check() -> None:
    """Test type checking in while loop."""
    with Graph(
        "while_loop_type_check",
        input_types=[TensorType(DType.int32, [], device=DeviceRef.CPU())],
    ) as graph:
        x = graph.inputs[0]

        def pred(x: TensorValue) -> TensorValue:
            return x < 10

        def body(x: TensorValue) -> TensorValue:
            # Return wrong type
            return ops.cast(x + 1, DType.float32)

        try:
            ops.while_loop(x, pred, body)
        except TypeError as e:
            assert "Results don't match expected types" in str(e)

        graph.output()

    graph._mlir_op.verify()


def test_while_loop_with_raising() -> None:
    with Graph(
        "while_loop_with_raising",
        input_types=[TensorType(DType.int32, [], device=DeviceRef.CPU())],
    ) as graph:
        x = graph.inputs[0]
        chain = graph.device_chains[DeviceRef.CPU()]

        def pred(x: TensorValue) -> TensorValue:
            return x < 10

        def body(x: TensorValue) -> NoReturn:
            raise Exception("raising")

        try:
            ops.while_loop(x, pred, body)
        except Exception as e:
            assert "raising" in str(e)

        assert graph.device_chains[DeviceRef.CPU()] == chain
        graph.output()
    graph._mlir_op.verify()


def test_while_loop_with_pred_block_chain_mutation() -> None:
    with Graph(
        "while_loop_with_pred_block_chain_mutation",
        input_types=[TensorType(DType.int32, [], device=DeviceRef.CPU())],
    ) as graph:
        x = graph.inputs[0]

        def pred(x: TensorValue) -> TensorValue:
            # print mutates the chain
            x.print()
            return x < 10

        def body(x: TensorValue) -> TensorValue:
            return x + 1

        try:
            ops.while_loop(x, pred, body)
        except Exception as e:
            assert "Chain mutation detected" in str(e)

        graph.output()
    graph._mlir_op.verify()


def test_while_loop_device_chains_scoped() -> None:
    # Ensures per-device chain mutations in loop blocks don't leak.
    t0 = TensorType(DType.float32, [4], device=DeviceRef.GPU(0))
    t1 = TensorType(DType.float32, [4], device=DeviceRef.GPU(1))
    sb0 = BufferType(DType.int64, [1], device=DeviceRef.GPU(0))
    sb1 = BufferType(DType.int64, [1], device=DeviceRef.GPU(1))

    with Graph(
        "while_device_chains_scoped", input_types=[t0, t1, sb0, sb1]
    ) as graph:
        x0, x1, s0, s1 = graph.inputs
        x0_t, x1_t = x0.tensor, x1.tensor
        s0_b, s1_b = s0.buffer, s1.buffer

        # Materialize and snapshot device chains.
        ch0_before = graph.device_chains[DeviceRef.GPU(0)]
        ch1_before = graph.device_chains[DeviceRef.GPU(1)]
        id0, id1 = id(ch0_before), id(ch1_before)

        def pred(_: TensorValue, __: TensorValue) -> TensorValue:
            return ops.constant(True, dtype=DType.bool, device=DeviceRef.CPU())

        def body(a: TensorValue, b: TensorValue) -> list[TensorValue]:
            # Advance device chains inside loop body via allreduce.
            _ = ops.allreduce.sum([x0_t, x1_t], [s0_b, s1_b])
            return [a, b]

        ops.while_loop([x0_t, x1_t], pred, body)
        # device chains must be restored after staging the loop regions
        assert id(graph.device_chains[DeviceRef.GPU(0)]) != id0
        assert id(graph.device_chains[DeviceRef.GPU(1)]) != id1
        graph.output()
