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
"""Test pipelines ragged attention layer."""

import math
from collections.abc import Callable
from functools import partial

import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.torch import torch_dtype_to_max
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
)
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import (
    cross_attention_ragged,
    flash_attention_gpu,
    flash_attention_ragged_gpu,
)
from max.nn.kv_cache import MHAKVCacheParams, PagedCacheValues
from test_common.modular_graph_test import are_all_tensor_values
from torch.nn.functional import scaled_dot_product_attention

TORCH_DTYPE = torch.bfloat16


def causal_max_flash_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    dtype = torch_dtype_to_max(q.dtype)
    _batch, _q_seq_len, nheads, head_dim = q.shape

    # Graph types.
    q_type = TensorType(
        dtype,
        shape=["batch", "q_seq_len", nheads, head_dim],
        device=DeviceRef.GPU(),
    )
    kv_type = TensorType(
        dtype,
        shape=["batch", "kv_seq_len", nheads, head_dim],
        device=DeviceRef.GPU(),
    )

    session = InferenceSession(devices=[Accelerator()])

    # Stage ops.

    # Construct and compile the MAX graph flash attention.
    graph = Graph(
        "flash_attn",
        forward=partial(
            flash_attention_gpu,
            scale=math.sqrt(1.0 / head_dim),
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
        ),
        input_types=[
            q_type,
            kv_type,
            kv_type,
        ],
    )

    # Compile model.
    model = session.load(graph)

    # Execute.
    output = model.execute(q.detach(), k.detach(), v.detach())[0]
    assert isinstance(output, Buffer)
    return torch.from_dlpack(output)


@pytest.mark.parametrize(
    "q_seqlen,k_seqlen",
    [
        (128, 128),
        # TODO(KERN-1634): support num_keys != seq_len.
        # (2, 3),
    ],
)
def test_causal_flash_attention_gpu(q_seqlen: int, k_seqlen: int) -> None:
    head_dim = 128
    batch_size = 1
    nheads = 6
    nheads_k = 6
    torch_device = "cuda"
    torch_dtype = torch.bfloat16

    q_shape = (batch_size, q_seqlen, nheads, head_dim)
    kv_shape = (batch_size, k_seqlen, nheads_k, head_dim)

    q = torch.randn(q_shape, device=torch_device, dtype=torch_dtype)
    k = torch.randn(kv_shape, device=torch_device, dtype=torch_dtype)
    v = torch.randn(kv_shape, device=torch_device, dtype=torch_dtype)

    out_max = causal_max_flash_attn(q, k, v).squeeze()
    out_flash_attn = (
        scaled_dot_product_attention(
            q.to(torch_device).permute(0, 2, 1, 3),
            k.to(torch_device).permute(0, 2, 1, 3),
            v.to(torch_device).permute(0, 2, 1, 3),
            is_causal=True,
            scale=math.sqrt(1.0 / head_dim),
        )
        .permute(0, 2, 1, 3)
        .squeeze()
    )

    torch.testing.assert_close(out_max, out_flash_attn, rtol=1e-2, atol=2e-2)


@pytest.fixture
def compute_ragged_max_flash_attn(
    gpu_session: InferenceSession,
    n_heads: int,
    head_dim: int,
) -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    dtype = torch_dtype_to_max(TORCH_DTYPE)
    # total_seq_len, nheads, head_dim = q_ragged.shape

    # Graph types.
    qkv_type = TensorType(
        dtype,
        shape=["total_seq_len", n_heads, head_dim],
        device=DeviceRef.GPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], DeviceRef.GPU()
    )
    max_seq_len_type = TensorType(
        DType.uint32, shape=[1], device=DeviceRef.CPU()
    )

    # Construct and compile the MAX graph flash attention.
    def construct() -> Graph:
        with Graph(
            "ragged_flash_attn",
            input_types=[
                qkv_type,
                qkv_type,
                qkv_type,
                input_row_offsets_type,
                max_seq_len_type,
            ],
        ) as g:
            assert are_all_tensor_values(g.inputs)
            q, k, v, input_row_offsets, max_seq_len = g.inputs

            result = flash_attention_ragged_gpu(
                q,
                k,
                v,
                input_row_offsets,
                max_seq_len,
                mask_variant=MHAMaskVariant.NULL_MASK,
                scale=math.sqrt(1.0 / head_dim),
            )
            g.output(result)
        return g

    graph = construct()

    # Compile model.
    model = gpu_session.load(graph)

    def execute(
        q_ragged: torch.Tensor,
        k_ragged: torch.Tensor,
        v_ragged: torch.Tensor,
        input_row_offsets: torch.Tensor,
        max_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        # Execute.
        output = model.execute(
            q_ragged.detach(),
            k_ragged.detach(),
            v_ragged.detach(),
            input_row_offsets.detach(),
            max_seq_len.detach(),
        )[0]
        return torch.from_dlpack(output)

    return execute


@pytest.mark.parametrize(
    "seq_lengths, n_heads, head_dim",
    [
        ([64, 64, 64, 16, 16, 4], 16, 128),  # Variable length sequences
        ([100], 32, 128),  # Single sequence
        ([256, 64], 16, 80),  # MiniMax M3 vision attention
    ],
)
def test_ragged_flash_attention_gpu(
    seq_lengths: list[int],
    n_heads: int,
    head_dim: int,
    compute_ragged_max_flash_attn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
) -> None:
    torch_device = "cuda"
    torch_dtype = TORCH_DTYPE

    # Create input_row_offsets (prefix sum of sequence lengths)
    input_row_offsets = [0] + list(
        torch.cumsum(torch.tensor(seq_lengths), dim=0).tolist()
    )
    total_seq_len = input_row_offsets[-1]

    # Generate ragged tensors
    q_ragged = torch.randn(
        total_seq_len,
        n_heads,
        head_dim,
        dtype=torch_dtype,
        device=torch_device,
    )
    k_ragged = torch.randn(
        total_seq_len,
        n_heads,
        head_dim,
        dtype=torch_dtype,
        device=torch_device,
    )
    v_ragged = torch.randn(
        total_seq_len,
        n_heads,
        head_dim,
        dtype=torch_dtype,
        device=torch_device,
    )

    input_row_offsets_tensor = torch.tensor(
        input_row_offsets, dtype=torch.uint32, device=torch_device
    )

    # Run ragged flash attention
    max_seq_len = max(seq_lengths)

    out_ragged = compute_ragged_max_flash_attn(
        q_ragged,
        k_ragged,
        v_ragged,
        input_row_offsets_tensor,
        torch.tensor([max_seq_len], dtype=torch.uint32, device="cpu"),
    )

    # Run reference attention for each batch element separately
    reference_outputs = []
    for i in range(len(seq_lengths)):
        start_idx = input_row_offsets[i]
        end_idx = input_row_offsets[i + 1]
        q_single = q_ragged[start_idx:end_idx].unsqueeze(0)
        k_single = k_ragged[start_idx:end_idx].unsqueeze(0)
        v_single = v_ragged[start_idx:end_idx].unsqueeze(0)

        out_single = scaled_dot_product_attention(
            q_single.permute(0, 2, 1, 3),
            k_single.permute(0, 2, 1, 3),
            v_single.permute(0, 2, 1, 3),
            scale=math.sqrt(1.0 / head_dim),
        ).permute(0, 2, 1, 3)
        reference_outputs.append(out_single.squeeze(0))

    # Compare ragged output with reference output
    start_idx = 0
    for i, seq_len in enumerate(seq_lengths):
        end_idx = start_idx + seq_len
        ragged_slice = out_ragged[start_idx:end_idx]
        reference_slice = reference_outputs[i]

        torch.testing.assert_close(
            ragged_slice, reference_slice, rtol=1e-2, atol=2e-2
        )
        start_idx = end_idx


def test_flash_attention_ragged_gpu_rejects_max_seq_len_on_gpu() -> None:
    """max_seq_len must be on CPU; passing a GPU tensor should raise."""
    n_heads = 4
    head_dim = 64
    dtype = DType.bfloat16

    qkv_type = TensorType(
        dtype,
        shape=["total_seq_len", n_heads, head_dim],
        device=DeviceRef.GPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], DeviceRef.GPU()
    )
    # Wrong: max_seq_len on GPU instead of CPU
    max_seq_len_type = TensorType(
        DType.uint32, shape=[1], device=DeviceRef.GPU()
    )

    with pytest.raises(ValueError, match=r"max_seq_len.*device"):
        with Graph(
            "bad_ragged_flash_attn",
            input_types=[
                qkv_type,
                qkv_type,
                qkv_type,
                input_row_offsets_type,
                max_seq_len_type,
            ],
        ) as g:
            assert are_all_tensor_values(g.inputs)
            q, k, v, input_row_offsets, max_seq_len = g.inputs
            flash_attention_ragged_gpu(
                q,
                k,
                v,
                input_row_offsets,
                max_seq_len,
                mask_variant=MHAMaskVariant.NULL_MASK,
                scale=math.sqrt(1.0 / head_dim),
            )


def test_cross_attention_ragged_rejects_q_max_seq_len_on_gpu() -> None:
    """q_max_seq_len must be on CPU; passing a GPU tensor should raise."""
    n_kv_heads = 4
    head_dim = 64
    num_layers = 1
    page_size = 128
    dtype = DType.bfloat16

    kv_params = MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        devices=[DeviceRef.GPU()],
        page_size=page_size,
    )

    input_type = TensorType(
        dtype, ["total_seq_len", n_kv_heads, head_dim], DeviceRef.GPU()
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["batch_plus_one"], DeviceRef.GPU()
    )
    kv_blocks_type = BufferType(
        dtype, ["blocks", 2, page_size, n_kv_heads, head_dim], DeviceRef.GPU()
    )
    cache_lengths_type = TensorType(DType.uint32, ["batch"], DeviceRef.GPU())
    lookup_table_type = TensorType(
        DType.uint32, ["batch", "max_pages"], DeviceRef.GPU()
    )
    max_prompt_length_type = TensorType(DType.uint32, [1], DeviceRef.CPU())
    max_cache_length_type = TensorType(DType.uint32, [1], DeviceRef.CPU())
    layer_idx_type = TensorType(DType.uint32, [1], DeviceRef.CPU())
    kv_input_row_offsets_type = TensorType(
        DType.uint32, ["kv_batch_plus_one"], DeviceRef.GPU()
    )
    # Wrong: q_max_seq_len on GPU instead of CPU
    q_max_seq_len_type = TensorType(DType.uint32, [1], DeviceRef.GPU())

    with pytest.raises(ValueError, match=r"q_max_seq_len.*device"):
        with Graph(
            "bad_cross_attn",
            input_types=[
                input_type,
                input_row_offsets_type,
                kv_blocks_type,
                cache_lengths_type,
                lookup_table_type,
                max_prompt_length_type,
                max_cache_length_type,
                layer_idx_type,
                kv_input_row_offsets_type,
                q_max_seq_len_type,
            ],
        ) as g:
            (
                inp,
                input_row_offsets,
                kv_blocks,
                cache_lengths,
                lookup_table,
                max_prompt_length,
                max_cache_length,
                layer_idx,
                kv_input_row_offsets,
                q_max_seq_len,
            ) = g.inputs

            assert isinstance(inp, TensorValue)
            assert isinstance(input_row_offsets, TensorValue)
            assert isinstance(kv_blocks, BufferValue)
            assert isinstance(cache_lengths, TensorValue)
            assert isinstance(lookup_table, TensorValue)
            assert isinstance(max_prompt_length, TensorValue)
            assert isinstance(max_cache_length, TensorValue)
            assert isinstance(layer_idx, TensorValue)
            assert isinstance(kv_input_row_offsets, TensorValue)
            assert isinstance(q_max_seq_len, TensorValue)

            kv_collection = PagedCacheValues(
                kv_blocks=kv_blocks,
                cache_lengths=cache_lengths,
                lookup_table=lookup_table,
                max_prompt_length=max_prompt_length,
                max_cache_length=max_cache_length,
            )

            cross_attention_ragged(
                kv_params=kv_params,
                input=inp,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                mask_variant=MHAMaskVariant.CAUSAL_MASK,
                kv_input_row_offsets=kv_input_row_offsets,
                q_max_seq_len=q_max_seq_len,
                scale=math.sqrt(1.0 / head_dim),
            )
