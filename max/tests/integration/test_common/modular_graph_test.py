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

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, Dim, Graph, StaticDim, SymbolicDim, TensorType
from test_common.graph_utils import (
    are_all_tensor_values_iterable as _are_all_tensor_values_iterable,
)

MAX_INPUT_MAGNITUDE = 1e5
MIN_SHAPE_DIM = 1
MAX_SHAPE_DIM = 16

ACCURACY_RTOL = 1e-2
ACCURACY_ATOL = 1e-8


# TODO: Swap all imports of this to use test_common.graph_utils directly.
are_all_tensor_values = _are_all_tensor_values_iterable


def elements(
    dtype: np.dtype, max_magnitude: float | None = None, **kwargs
) -> st.SearchStrategy[Any]:
    if max_magnitude is None:
        max_magnitude = MAX_INPUT_MAGNITUDE
    kwargs.setdefault("max_value", max_magnitude)

    if "min_value" not in kwargs:
        if np.issubdtype(dtype, np.integer):
            kwargs.setdefault("min_value", 0)
        else:
            kwargs.setdefault("min_value", -max_magnitude)

    return nps.from_dtype(
        dtype,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        **kwargs,
    )


@st.composite
def shapes(
    draw: st.DrawFn,
    tensor_type: TensorType | BufferType,
    static_dims: Mapping[str, int] = {},
) -> tuple[int, ...]:
    """Defines a strategy for generating a concrete shape from a TensorType.

    Args:
        draw: Used with @st.composite to write type hints.
        tensor_type: Input tensor type.
        static_dims: Map of symbolic dimension name to static dimensions. This
            is used when there are dimensions in the test input arrays that must
            be hardcoded. For example, if the input has dims ["a", "b", "c"],
            and "c" must be "a" + "b".

    Returns:
        Shape tuple
    """

    def draw_dim(dim: Dim) -> int:
        if isinstance(dim, SymbolicDim):
            if (static := static_dims.get(dim.name)) is not None:
                return static

            sides = st.integers(
                min_value=MIN_SHAPE_DIM, max_value=MAX_SHAPE_DIM
            )
            # shared ensures symbolic dimensions match
            return draw(st.shared(sides, key=dim.name))
        elif isinstance(dim, StaticDim):
            return int(dim)
        raise NotImplementedError

    return tuple(draw_dim(dim) for dim in tensor_type.shape)


def arrays(
    tensor_type: TensorType | BufferType,
    static_dims: Mapping[str, int] = {},
    **kwargs,
) -> st.SearchStrategy[Buffer]:
    if tensor_type.dtype == DType.bfloat16:
        tensor_type = TensorType(
            DType.float32, tensor_type.shape, device=tensor_type.device
        )
        return arrays(tensor_type, static_dims=static_dims, **kwargs).map(
            lambda t: Buffer.from_dlpack(
                torch.from_dlpack(t).to(torch.bfloat16)
            )
        )

    dtype = tensor_type.dtype.to_numpy()
    return nps.arrays(
        dtype,
        shapes(tensor_type, static_dims),
        elements=elements(dtype, **kwargs),
    ).map(Buffer.from_dlpack)


def given_input_types(
    input_types: Iterable[TensorType | BufferType],
    static_dims: Mapping[str, int] = {},
    provided_inputs: Mapping[int, np.ndarray | Buffer] = {},
    max_magnitude: float | None = None,
    **kwargs,
) -> Callable[[Callable[[tuple[Buffer, ...]], None]], Callable[[], None]]:
    input_arrays = []
    for i, input_type in enumerate(input_types):
        if i in provided_inputs:
            input_arrays.append(st.just(Buffer.from_dlpack(provided_inputs[i])))
        elif max_magnitude is not None:
            input_arrays.append(
                arrays(
                    input_type,
                    static_dims=static_dims,
                    max_magnitude=max_magnitude,
                    **kwargs,
                )
            )
        else:
            input_arrays.append(
                arrays(input_type, static_dims=static_dims, **kwargs)
            )

        # Move things to the right device
        input_arrays[i] = input_arrays[i].map(
            lambda t: t.to(input_type.device.to_device())  # noqa: B023
        )

    return given(st.tuples(*input_arrays))


def execute(model: Model, inputs: Sequence[np.ndarray | Buffer]) -> Buffer:
    tensor_inputs = [
        t if isinstance(t, Buffer) else Buffer.from_dlpack(t) for t in inputs
    ]
    results = model.execute(*tensor_inputs)
    assert results
    assert isinstance(results[0], Buffer)
    return results[0]


def modular_graph_test(
    session: InferenceSession,
    graph: Graph,
    *,
    static_dims: Mapping[str, int] = {},
    provided_inputs: Mapping[int, np.ndarray | Buffer] = {},
    hypothesis_settings: settings | None = None,
    max_magnitude: float | None = None,
    **kwargs,
) -> Callable[
    [
        Callable[
            [
                Callable[[Sequence[Buffer]], Buffer],
                Sequence[Buffer],
                Sequence[torch.Tensor],
            ],
            None,
        ]
    ],
    Callable[[], None],
]:
    def decorator(
        test_fn: Callable[
            [
                Callable[[Sequence[Buffer]], Buffer],
                Sequence[Buffer],
                Sequence[torch.Tensor],
            ],
            None,
        ],
    ) -> Callable[[], None]:
        model = session.load(graph)

        input_types = []
        for input in graph.inputs:
            assert isinstance(input.type, TensorType | BufferType)

            input_types.append(input.type)

        @settings(suppress_health_check=[HealthCheck.data_too_large])
        @given_input_types(
            input_types,
            static_dims=static_dims,
            provided_inputs=provided_inputs,
            max_magnitude=max_magnitude,
            **kwargs,
        )
        def test_correctness(inputs: Sequence[Buffer]) -> None:
            model_execute = functools.partial(execute, model)

            torch_inputs = [torch.from_dlpack(t) for t in inputs]
            test_fn(model_execute, inputs, torch_inputs)

        if hypothesis_settings is not None:
            test_correctness = hypothesis_settings(test_correctness)
        test_correctness()
        return test_correctness

    return decorator
