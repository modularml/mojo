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

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.torch import max_dtype_to_torch
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.mlir import StringAttr
from max.nn.kernels import (
    fused_qkv_ragged_matmul,
    matmul_k_cache_ragged,
    matmul_kv_cache_ragged,
)
from max.nn.kv_cache import (
    KVCacheParams,
    MHAKVCacheParams,
    PagedCacheValues,
)
from test_common.modular_graph_test import modular_graph_test
from test_common.simple_kv_cache import (
    block_ids_for_batch,
    paged_kv_cache_inputs,
)
from torch.utils.dlpack import from_dlpack


def _dump_k_cache_to_torch_tensor(
    params: KVCacheParams,
    kv_blocks: Buffer,
    block_ids: Sequence[int],
    seq_len: int,
) -> torch.Tensor:
    """
    Returns a torch tensor of the shape [seq_len, num_layers, n_heads, head_dim]

    This should only be used for testing purposes.
    """
    torch_dtype = max_dtype_to_torch(params.dtype)
    page_size = params.page_size

    # [total_num_pages, kv_dim, num_layers, page_size, n_heads, head_dim]
    device_buffer_torch = from_dlpack(kv_blocks).to(torch_dtype).cpu()

    # Keys live at index 0 of the kv dim. [total_num_pages, num_layers,
    # page_size, n_heads, head_dim]
    device_buffer_torch = device_buffer_torch[:, 0, :, :, :, :]

    # [seq_len, num_layers, n_heads, head_dim]
    res = torch.empty(
        (
            seq_len,
            params.num_layers,
            params.n_kv_heads_per_device,
            params.head_dim,
        ),
        dtype=torch_dtype,
    )

    for start_idx in range(0, seq_len, page_size):
        end_idx = min(start_idx + page_size, seq_len)

        block_id = block_ids[start_idx // page_size]

        # [num_layers, page_size, n_heads, head_dim]
        block_torch = device_buffer_torch[block_id, :]

        for token_idx in range(start_idx, end_idx):
            res[token_idx, :, :, :] = block_torch[
                :, token_idx % page_size, :, :
            ]

    return res


def test_fused_qkv_ragged_matmul(session: InferenceSession) -> None:
    num_q_heads = 32
    kv_params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.CPU()],
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)
    input_type = TensorType(
        DType.float32,
        ["total_seq_len", num_q_heads * kv_params.head_dim],
        device=DeviceRef.CPU(),
    )
    wqkv_type = TensorType(
        DType.float32,
        [
            num_q_heads * kv_params.head_dim,
            (num_q_heads + 2 * (kv_params.n_kv_heads)) * kv_params.head_dim,
        ],
        device=DeviceRef.CPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        [
            "input_row_offsets_len",
        ],
        device=DeviceRef.CPU(),
    )

    def construct() -> Graph:
        with Graph(
            "call_ragged_qkv_matmul",
            input_types=[
                input_type,
                input_row_offsets_type,
                wqkv_type,
                *kv_params.flattened_kv_inputs(),
            ],
        ) as g:
            (
                input,
                input_row_offsets,
                wqkv,
                blocks,
                cache_lengths,
                lookup_table,
                max_prompt_length,
                max_cache_length,
                _attention_dispatch_metadata,
            ) = g.inputs
            layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())

            kv_collection = PagedCacheValues(
                blocks.buffer,
                cache_lengths.tensor,
                lookup_table.tensor,
                max_prompt_length.tensor,
                max_cache_length.tensor,
            )
            result = fused_qkv_ragged_matmul(
                kv_params,
                input.tensor,
                input_row_offsets.tensor,
                wqkv.tensor,
                kv_collection,
                layer_idx,
                num_q_heads,
            )
            g.output(result)
        return g

    g = construct()

    input_row_offsets = Buffer(
        DType.uint32,
        [batch_size + 1],
    )
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[i] = running_sum
    kv_runtime_inputs = paged_kv_cache_inputs(
        kv_params, prompt_lens, total_num_pages=8
    )
    assert kv_runtime_inputs.attention_dispatch_metadata is not None

    @modular_graph_test(
        session,
        g,
        static_dims={
            "total_seq_len": total_seq_len,
            "input_row_offsets_len": len(prompt_lens) + 1,
        },
        provided_inputs={
            1: input_row_offsets,
            3: kv_runtime_inputs.kv_blocks,
            4: kv_runtime_inputs.cache_lengths,
            5: kv_runtime_inputs.lookup_table,
            6: kv_runtime_inputs.max_prompt_length,
            7: kv_runtime_inputs.max_cache_length,
            8: kv_runtime_inputs.attention_dispatch_metadata,
        },
    )
    def test_runs_without_nan(
        execute: Callable[[Sequence[Buffer]], Buffer],
        inputs: Sequence[Buffer],
        torch_inputs: Sequence[torch.Tensor],
    ) -> None:
        inputs = list(inputs)
        result = execute(inputs).to_numpy()
        assert np.all(np.isfinite(result))


@dataclass(frozen=True)
class MatmulKVRaggedModel:
    """Model containing a single matmul KV ragged op."""

    kv_params: KVCacheParams
    """Hyperparameters describing this instance of the KV cache."""

    layer_idx: int
    """Layer index of the KV cache collection."""

    def __call__(
        self,
        hidden_states: TensorValue,
        input_row_offsets: TensorValue,
        weight: TensorValue,
        *kv_inputs: TensorValue,
    ) -> None:
        """Stages a graph consisting of a matmul KV cache ragged custom op.

        This contains both the matmul KV cache ragged custom op and a "fetch"
        op to get a KVCacheCollection.
        """
        matmul_kv_cache_ragged(
            self.kv_params,
            hidden_states,
            input_row_offsets,
            weight,
            kv_collection=PagedCacheValues(
                kv_blocks=kv_inputs[0].buffer,
                cache_lengths=kv_inputs[1].tensor,
                lookup_table=kv_inputs[2].tensor,
                max_prompt_length=kv_inputs[3].tensor,
                max_cache_length=kv_inputs[4].tensor,
            ),
            layer_idx=ops.constant(
                self.layer_idx, DType.uint32, device=DeviceRef.CPU()
            ),
        )


@pytest.mark.parametrize(
    "dtype",
    [
        DType.float32,
        # TODO(bduke): support converting to torch tensor from bfloat16 driver
        # tensor.
        # DType.bfloat16,
    ],
)
def test_matmul_kv_ragged(session: InferenceSession, dtype: DType) -> None:
    """Tests the matmul_kv_cache_ragged custom op."""
    # Set up hyperparameters for the test.
    torch_dtype = {
        DType.float32: torch.float32,
        DType.bfloat16: torch.bfloat16,
    }[dtype]
    num_q_heads = 32
    kv_params = MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.CPU()],
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)

    # Set MLIR types for the graph.
    hidden_state_type = TensorType(
        dtype,
        ["total_seq_len", num_q_heads * kv_params.head_dim],
        device=DeviceRef.CPU(),
    )
    wkv_type = TensorType(
        dtype,
        [
            (2 * (kv_params.n_kv_heads)) * kv_params.head_dim,
            num_q_heads * kv_params.head_dim,
        ],
        device=DeviceRef.CPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        ["input_row_offsets_len"],
        device=DeviceRef.CPU(),
    )

    # Stage the fetch op + custom matmul KV cache ragged op graph.
    graph = Graph(
        "matmul_kv_cache_ragged",
        forward=MatmulKVRaggedModel(kv_params, layer_idx=0),
        input_types=[
            hidden_state_type,
            input_row_offsets_type,
            wkv_type,
            *kv_params.flattened_kv_inputs(),
        ],
    )

    # Compile and init the model.
    model = session.load(graph)

    # Compute input row offsets for ragged tensors.
    input_row_offsets = Buffer(DType.uint32, [batch_size + 1])
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[i] = running_sum
    kv_inputs = paged_kv_cache_inputs(kv_params, prompt_lens, total_num_pages=8)
    kv_blocks = kv_inputs.kv_blocks
    # First check that the KV cache was zeroed out on initialization.
    assert not kv_blocks.to_numpy().any()

    hidden_states = torch.randn(
        size=[total_seq_len, num_q_heads * kv_params.head_dim],
        dtype=torch_dtype,
    )
    wkv = torch.randn(size=wkv_type.shape.static_dims, dtype=torch_dtype)
    model(hidden_states, input_row_offsets, wkv, *kv_inputs.flatten())

    # Check that the matmul wrote output to the KV cache.
    assert kv_blocks.to_numpy().any()


@dataclass(frozen=True)
class MatmulKRaggedModel:
    """Model containing a single matmul KV ragged op."""

    kv_params: KVCacheParams
    """Hyperparameters describing this instance of the KV cache."""

    layer_idx: int
    """Layer index of the KV cache collection."""

    def __call__(
        self,
        hidden_states: TensorValue,
        input_row_offsets: TensorValue,
        weight: TensorValue,
        *kv_inputs: TensorValue,
    ) -> None:
        """Stages a graph consisting of a matmul KV cache ragged custom op.

        This contains both the matmul KV cache ragged custom op and a "fetch"
        op to get a KVCacheCollection.
        """
        matmul_k_cache_ragged(
            self.kv_params,
            hidden_states,
            input_row_offsets,
            weight,
            kv_collection=PagedCacheValues(
                kv_blocks=kv_inputs[0].buffer,
                cache_lengths=kv_inputs[1].tensor,
                lookup_table=kv_inputs[2].tensor,
                max_prompt_length=kv_inputs[3].tensor,
                max_cache_length=kv_inputs[4].tensor,
            ),
            layer_idx=ops.constant(
                self.layer_idx, DType.uint32, device=DeviceRef.CPU()
            ),
        )


@pytest.mark.parametrize("dtype", [DType.float32])
def test_matmul_k_ragged(session: InferenceSession, dtype: DType) -> None:
    """Tests the matmul_k_cache_ragged custom op."""
    # Set up hyperparameters for the test.
    page_size = 128
    torch_dtype = {
        DType.float32: torch.float32,
        DType.bfloat16: torch.bfloat16,
    }[dtype]
    num_q_heads = 32
    kv_params = MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        page_size=page_size,
        devices=[DeviceRef.CPU()],
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)

    # Set MLIR types for the graph.
    hidden_state_type = TensorType(
        dtype,
        ["total_seq_len", num_q_heads * kv_params.head_dim],
        device=DeviceRef.CPU(),
    )
    wk_type = TensorType(
        dtype,
        [
            kv_params.n_kv_heads * kv_params.head_dim,
            num_q_heads * kv_params.head_dim,
        ],
        device=DeviceRef.CPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        ["input_row_offsets_len"],
        device=DeviceRef.CPU(),
    )
    graph = Graph(
        "matmul_k_cache_ragged",
        forward=MatmulKRaggedModel(kv_params, layer_idx=0),
        input_types=[
            hidden_state_type,
            input_row_offsets_type,
            wk_type,
            *kv_params.flattened_kv_inputs(),
        ],
    )

    # Compile and init the model.
    model = session.load(graph)

    # Compute input row offsets for ragged tensors.
    input_row_offsets = Buffer(DType.uint32, [batch_size + 1])
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum
    kv_inputs = paged_kv_cache_inputs(kv_params, prompt_lens, total_num_pages=8)

    hidden_states = torch.randn(
        size=[total_seq_len, num_q_heads * kv_params.head_dim],
        dtype=torch_dtype,
    )
    wk = torch.randn(size=wk_type.shape.static_dims, dtype=torch_dtype)
    model(hidden_states, input_row_offsets, wk, *kv_inputs.flatten())

    ref_results = hidden_states @ wk.T

    block_ids = block_ids_for_batch(prompt_lens, kv_params.page_size)
    for batch_idx in range(batch_size):
        k_cache = _dump_k_cache_to_torch_tensor(
            kv_params,
            kv_inputs.kv_blocks,
            block_ids[batch_idx],
            prompt_lens[batch_idx],
        )

        # Calculate starting position for this batch
        seq_start = (
            int(np.cumsum(prompt_lens[:batch_idx])) if batch_idx != 0 else 0
        )
        seq_len = prompt_lens[batch_idx]

        expected = ref_results[seq_start : (seq_start + seq_len), :]

        torch.testing.assert_close(
            k_cache.reshape([seq_len, -1]),
            expected,
            rtol=5e-4,
            atol=5e-4,
        )


@pytest.mark.parametrize(
    "dtype",
    [DType.float32, DType.bfloat16],
)
def test_matmul_kv_cache_ragged_chains(dtype: DType) -> None:
    """Tests that staging matmul_kv_cache_ragged threads chains."""
    # Set up hyperparameters for the test.
    num_q_heads = 32
    kv_params = MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.CPU()],
    )

    # Set MLIR types for the graph.
    hidden_state_type = TensorType(
        dtype,
        ["total_seq_len", num_q_heads * kv_params.head_dim],
        device=DeviceRef.CPU(),
    )
    wkv_type = TensorType(
        dtype,
        [
            (2 * (kv_params.n_kv_heads)) * kv_params.head_dim,
            num_q_heads * kv_params.head_dim,
        ],
        device=DeviceRef.CPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        ["input_row_offsets_len"],
        device=DeviceRef.CPU(),
    )

    # Stage the fetch op + custom matmul KV cache ragged op graph.
    graph = Graph(
        "matmul_kv_cache_ragged",
        forward=MatmulKVRaggedModel(kv_params, layer_idx=0),
        input_types=[
            hidden_state_type,
            input_row_offsets_type,
            wkv_type,
            *kv_params.flattened_kv_inputs(),
        ],
    )
    matmul_kv_cache_op = [
        op
        for op in graph._mlir_op.regions[0].blocks[0].operations
        if op.name == "mo.custom"
        and "kv_matmul" in StringAttr(op.attributes["symbol"]).value
    ][0]
    assert len(matmul_kv_cache_op.results) == 1
    assert "!mo.chain" in str(matmul_kv_cache_op.results[-1].type)

    matmul_args = matmul_kv_cache_op.operands
    assert "!mo.chain" in str(matmul_args[-1].type)
