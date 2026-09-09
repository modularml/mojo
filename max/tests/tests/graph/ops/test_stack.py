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
"""ops.stack tests."""

import pytest
from conftest import GraphBuilder, new_axes, shapes, static_dims, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, StaticDim, TensorType, ops


@given(
    type=st.shared(tensor_types(), key="type"),
    # Stack is currently "slow" for larger lists because of MLIR interop.
    # Stack currently fails on an empty list.
    stack_size=st.integers(min_value=1, max_value=20),
    axis=new_axes(st.shared(tensor_types(), key="type")),
)
def test_stack(type: TensorType, stack_size: int, axis: int) -> None:
    with Graph("stack", input_types=[type] * stack_size) as graph:
        out = ops.stack((v.tensor for v in graph.inputs), axis)
        target_shape = list(type.shape)
        target_shape.insert(
            axis + type.rank + 1 if axis < 0 else axis, StaticDim(stack_size)
        )
        assert out.shape == target_shape
        graph.output(out)


def test_stack_error_with_empty_list() -> None:
    with Graph("stack", input_types=[]) as graph:
        with pytest.raises(
            ValueError, match="Expected at least one value to stack"
        ):
            ops.stack(v.tensor for v in graph.inputs)


def test_stack_error_with_different_ranks() -> None:
    with Graph(
        "stack",
        input_types=[
            TensorType(
                shape=[1, 2], dtype=DType.float32, device=DeviceRef.CPU()
            ),
            TensorType(
                shape=[1, 2, 3], dtype=DType.float32, device=DeviceRef.CPU()
            ),
        ],
    ) as graph:
        with pytest.raises(
            ValueError, match="All inputs to stack must be the same rank"
        ):
            ops.stack(v.tensor for v in graph.inputs)


def test_stack_error_with_different_dtypes() -> None:
    with Graph(
        "stack",
        input_types=[
            TensorType(
                shape=[1, 2], dtype=DType.float32, device=DeviceRef.CPU()
            ),
            TensorType(shape=[1, 2], dtype=DType.int32, device=DeviceRef.CPU()),
        ],
    ) as graph:
        with pytest.raises(
            ValueError, match="All inputs to stack must have the same dtype"
        ):
            ops.stack(v.tensor for v in graph.inputs)


def test_stack_error_with_different_devices() -> None:
    with Graph(
        "stack",
        input_types=[
            TensorType(
                shape=[1, 2], dtype=DType.float32, device=DeviceRef.CPU()
            ),
            TensorType(
                shape=[1, 2], dtype=DType.float32, device=DeviceRef.GPU()
            ),
        ],
    ) as graph:
        with pytest.raises(
            ValueError, match="Input values must be on the same device"
        ):
            ops.stack(v.tensor for v in graph.inputs)


shared_static_dim = st.shared(static_dims())


@given(
    input_types=st.lists(
        tensor_types(
            dtypes=st.shared(st.from_type(DType)),
            shapes=shapes(),
        ),
        min_size=2,
    )
)
def test_stack_error_with_many_different_shapes(
    graph_builder: GraphBuilder,
    input_types: list[TensorType],
) -> None:
    # Using a list comprehension to check if there are different ranks
    assume(len({len(t.shape) for t in input_types}) > 1)
    with graph_builder(input_types=input_types) as graph:
        with pytest.raises(
            ValueError, match="All inputs to stack must be the same rank"
        ):
            ops.stack(v.tensor for v in graph.inputs)


@given(
    x_type=tensor_types(shapes=shapes(max_rank=2, min_rank=2)),
    y_type=tensor_types(shapes=shapes(max_rank=2, min_rank=2)),
)
def test_stack_error_with_many_different_dtypes(
    graph_builder: GraphBuilder,
    x_type: TensorType,
    y_type: TensorType,
) -> None:
    assume(x_type.dtype != y_type.dtype)
    with graph_builder(input_types=[x_type, y_type]) as graph:
        with pytest.raises(
            ValueError, match="All inputs to stack must have the same dtype"
        ):
            ops.stack(v.tensor for v in graph.inputs)


shared_shapes_rank_gt_0 = st.shared(shapes(min_rank=1))


def invalid_axes(rank: int):  # noqa: ANN201
    lower_bound, upper_bound = -(rank + 1), rank
    return st.one_of(
        st.integers(max_value=lower_bound - 1),
        st.integers(min_value=upper_bound + 1),
    )


@given(
    base_type=tensor_types(shapes=shared_shapes_rank_gt_0),
    invalid_axis=shared_shapes_rank_gt_0.flatmap(
        lambda shape: invalid_axes(shape.rank)
    ),
)
def test_stack_error_with_axis_out_of_bounds(
    graph_builder: GraphBuilder,
    base_type: TensorType,
    invalid_axis: int,
) -> None:
    with graph_builder(input_types=[base_type]) as graph:
        with pytest.raises(IndexError, match="Axis out of range"):
            ops.stack([graph.inputs[0].tensor], axis=invalid_axis)
