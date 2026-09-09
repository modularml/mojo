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
"""Tests for transformer sequential layers helper in max.nn.transformer."""

from __future__ import annotations

import unittest.mock
from collections.abc import Sequence
from typing import Any

import pytest
from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.kv_cache import KVCacheInputsPerDevice, PagedCacheValues
from max.nn.layer import Module
from max.nn.transformer import forward_sequential_layers

TENSOR_T = TensorType(DType.float32, (1, 8), DeviceRef.CPU())
IDX_T = TensorType(DType.uint32, (), DeviceRef.CPU())
KV_BLOCKS_T = BufferType(DType.float32, (2, 4), DeviceRef.CPU())
KV_1D_T = TensorType(DType.int64, (2,), DeviceRef.CPU())


class _IdentityLayer(Module):
    """Identity layer that records how many times it was called."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.last_input: Value[Any] | None = None

    def __call__(self, x: list[Value[Any]]) -> list[Value[Any]]:
        self.call_count += 1
        self.last_input = x[0]
        return x


class _ScalarAndListLayer(Module):
    """Layer with a scalar first arg and list second arg, like real layers."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.received_idx: TensorValue | None = None

    def __call__(
        self, layer_idx: TensorValue, xs: list[TensorValue]
    ) -> list[TensorValue]:
        self.call_count += 1
        assert isinstance(layer_idx, TensorValue), (
            f"layer_idx should be TensorValue, got {type(layer_idx)}"
        )
        self.received_idx = layer_idx
        return xs


def _inputs_for_layer(
    idx: int, h: list[TensorValue]
) -> list[Value[Any] | Sequence[Value[Any]]]:
    return [h]


def _mixed_inputs_for_layer(
    idx: int, h: list[TensorValue]
) -> list[Value[Any] | Sequence[Value[Any]]]:
    return [
        ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
        list(h),
    ]


@pytest.mark.parametrize("subgraph_layer_groups", [None, []])
def test_direct_call_path_no_groups(
    subgraph_layer_groups: list[list[int]] | None,
) -> None:
    """Test direct layer invocation when subgraph_layer_groups is None or empty."""
    with Graph(
        "test", input_types=[TensorType(DType.float32, (1, 8), DeviceRef.CPU())]
    ) as main_graph:
        layers = [_IdentityLayer() for _ in range(3)]
        initial_hidden_states = [main_graph.inputs[0].tensor]

        result = forward_sequential_layers(
            layers,
            inputs_for_layer=_inputs_for_layer,
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
            subgraph_layer_groups=subgraph_layer_groups,
            initial_hidden_states=initial_hidden_states,
        )

        for layer in layers:
            assert layer.call_count == 1

        assert "mo.call @" not in str(main_graph)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].type == TENSOR_T


def test_subgraph_path_single_group() -> None:
    """Test subgraph path with a single group containing 3 layers."""
    with Graph(
        "test", input_types=[TensorType(DType.float32, (1, 8), DeviceRef.CPU())]
    ) as main_graph:
        layers = [_IdentityLayer() for _ in range(3)]
        initial_hidden_states = [main_graph.inputs[0].tensor]

        result = forward_sequential_layers(
            layers,
            inputs_for_layer=_inputs_for_layer,
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
            subgraph_layer_groups=[[0, 1, 2]],
            initial_hidden_states=initial_hidden_states,
        )

        assert layers[0].call_count == 1
        assert layers[1].call_count == 0
        assert layers[2].call_count == 0

        assert str(main_graph).count("callee = @transformer_block_0") == 3

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].type == TENSOR_T


@pytest.mark.parametrize("use_subgraphs", [False, True])
def test_state_threading_between_layers(use_subgraphs: bool) -> None:
    """Test that hidden states are correctly threaded through layers."""
    with Graph(
        "test", input_types=[TensorType(DType.float32, (1, 8), DeviceRef.CPU())]
    ) as main_graph:
        layers = [_IdentityLayer() for _ in range(3)]
        initial_hidden_states = [main_graph.inputs[0].tensor]

        recorded_calls: list[tuple[int, list[TensorValue]]] = []

        def on_layer_output(idx: int, h: list[TensorValue]) -> None:
            recorded_calls.append((idx, h))

        result = forward_sequential_layers(
            layers,
            inputs_for_layer=_inputs_for_layer,
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
            subgraph_layer_groups=[[0, 1, 2]] if use_subgraphs else None,
            initial_hidden_states=initial_hidden_states,
            on_layer_output=on_layer_output,
        )

        assert len(recorded_calls) == 3
        assert [idx for idx, _ in recorded_calls] == [0, 1, 2]

        for _, h in recorded_calls:
            assert len(h) == 1

        assert result == recorded_calls[-1][1]

        if not use_subgraphs:
            assert layers[1].last_input is layers[0].last_input
            assert layers[2].last_input is layers[1].last_input


@pytest.mark.parametrize(
    "use_subgraphs",
    [
        False,
        True,
    ],
)
def test_on_layer_output_callback(
    use_subgraphs: bool,
) -> None:
    """Test on_layer_output callback is invoked correctly."""
    on_layer_output = unittest.mock.MagicMock()
    with Graph(
        "test", input_types=[TensorType(DType.float32, (1, 8), DeviceRef.CPU())]
    ) as main_graph:
        layers = [_IdentityLayer() for _ in range(3)]
        initial_hidden_states = [main_graph.inputs[0].tensor]

        forward_sequential_layers(
            layers,
            inputs_for_layer=_inputs_for_layer,
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
            subgraph_layer_groups=[[0, 1, 2]] if use_subgraphs else None,
            initial_hidden_states=initial_hidden_states,
            on_layer_output=on_layer_output,
        )

        assert on_layer_output.call_count == 3

        assert [c.args[0] for c in on_layer_output.call_args_list] == [0, 1, 2]

        for call in on_layer_output.call_args_list:
            h = call.args[1]
            assert len(h) == 1


@pytest.mark.parametrize("use_subgraphs", [False, True])
def test_mixed_scalar_and_list_args(use_subgraphs: bool) -> None:
    """Test that scalar args (like layer_idx) are passed as scalars, not lists."""
    with Graph(
        "test", input_types=[TensorType(DType.float32, (1, 8), DeviceRef.CPU())]
    ) as main_graph:
        layers = [_ScalarAndListLayer() for _ in range(3)]
        initial_hidden_states = [main_graph.inputs[0].tensor]

        result = forward_sequential_layers(
            layers,
            inputs_for_layer=_mixed_inputs_for_layer,
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
            subgraph_layer_groups=[[0, 1, 2]] if use_subgraphs else None,
            initial_hidden_states=initial_hidden_states,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].type == TENSOR_T

        if not use_subgraphs:
            for layer in layers:
                assert layer.call_count == 1
                assert isinstance(layer.received_idx, TensorValue)
                assert layer.received_idx.type == IDX_T


class _KVCollectionLayer(Module):
    """Layer receiving hidden states plus a ``list[PagedCacheValues]``."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.received_kv: list[PagedCacheValues] | None = None

    def __call__(
        self, xs: list[TensorValue], kv_collections: list[PagedCacheValues]
    ) -> list[TensorValue]:
        self.call_count += 1
        self.received_kv = kv_collections
        assert isinstance(kv_collections, list)
        for kv in kv_collections:
            assert isinstance(kv, KVCacheInputsPerDevice), (
                f"expected reconstructed PagedCacheValues, got {type(kv)}"
            )
        return xs


@pytest.mark.parametrize("use_subgraphs", [False, True])
def test_structured_kv_collection_input(use_subgraphs: bool) -> None:
    """A ``list[PagedCacheValues]`` element crosses the subgraph boundary and
    is reconstructed as structured ``PagedCacheValues`` before the block runs."""
    with Graph(
        "test",
        input_types=[
            TENSOR_T,
            KV_BLOCKS_T,
            KV_1D_T,
            KV_1D_T,
            KV_1D_T,
            KV_1D_T,
        ],
    ) as main_graph:
        h0 = main_graph.inputs[0].tensor
        kv_collections = [
            PagedCacheValues(
                kv_blocks=main_graph.inputs[1].buffer,
                cache_lengths=main_graph.inputs[2].tensor,
                lookup_table=main_graph.inputs[3].tensor,
                max_prompt_length=main_graph.inputs[4].tensor,
                max_cache_length=main_graph.inputs[5].tensor,
            )
        ]

        layers = [_KVCollectionLayer() for _ in range(2)]

        def inputs_for_layer(idx: int, h: list[TensorValue]) -> list[Any]:
            return [h, kv_collections]

        result = forward_sequential_layers(
            layers,
            inputs_for_layer=inputs_for_layer,
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
            subgraph_layer_groups=[[0, 1]] if use_subgraphs else None,
            initial_hidden_states=[h0],
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].type == TENSOR_T

        # The representative layer (layer 0) always runs at graph-build time and
        # must have received a reconstructed structured PagedCacheValues.
        assert layers[0].call_count == 1
        assert layers[0].received_kv is not None
        assert isinstance(layers[0].received_kv[0], KVCacheInputsPerDevice)

        if use_subgraphs:
            # Subgraph built once from layer 0, invoked once per layer.
            assert layers[1].call_count == 0
            assert str(main_graph).count("callee = @transformer_block_0") == 2
        else:
            assert layers[1].call_count == 1
            # Direct path forwards the exact same collection object.
            assert layers[1].received_kv is kv_collections
