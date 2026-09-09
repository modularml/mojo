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
"""Unit tests for kv_cache_ragged_2m_iadd kernel."""

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.torch import max_dtype_to_torch
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import kv_cache_ragged_2m_iadd
from max.nn.kv_cache import (
    KVCacheParams,
    MHAKVCacheParams,
    PagedCacheValues,
)
from test_common.simple_kv_cache import (
    block_ids_for_batch,
    paged_kv_cache_inputs,
)
from torch.utils.dlpack import from_dlpack

DTYPE = DType.bfloat16
TORCH_DTYPE = torch.bfloat16
RTOL = 1e-3
ATOL = 5e-3


def rand_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype = TORCH_DTYPE,
    seed: int = 1234,
) -> torch.Tensor:
    """Generate random tensor on CPU."""
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=dtype) * 0.02


def to_max_tensor(tensor: torch.Tensor, device: Device) -> Buffer:
    """Convert CPU torch tensor to MAX tensor on device."""
    return Buffer.from_dlpack(tensor).to(device)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> None:
    """Compare tensors, moving to CPU for comparison."""
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=rtol, atol=atol
    )


class KeyOrValue:
    KEY = 0
    VALUE = 1


def dump_kv_cache_to_torch(
    kv_params: KVCacheParams,
    kv_blocks: Buffer,
    seq_lens: list[int],
    key_or_value: int,
) -> list[torch.Tensor]:
    """Extract K or V cache contents for each sequence in batch."""
    torch_dtype = max_dtype_to_torch(kv_params.dtype)
    device_buffer_torch = from_dlpack(kv_blocks).to(torch_dtype).cpu()
    device_buffer_torch = device_buffer_torch[:, key_or_value, :, :, :, :]
    page_size = kv_params.page_size

    results = []
    blocks = block_ids_for_batch(seq_lens, page_size)
    for req_blocks, seq_len in zip(blocks, seq_lens, strict=True):
        result = torch.empty(
            seq_len,
            kv_params.n_kv_heads_per_device,
            kv_params.head_dim,
            dtype=torch_dtype,
        )

        for start_idx in range(0, seq_len, page_size):
            end_idx = min(start_idx + page_size, seq_len)
            block_id = req_blocks[start_idx // page_size]
            block_torch = device_buffer_torch[block_id, 0]

            for token_idx in range(start_idx, end_idx):
                result[token_idx] = block_torch[token_idx % page_size]

        results.append(result.reshape(seq_len, -1))

    return results


def run_kv_cache_2m_iadd(
    kv_lora_output: torch.Tensor,
    prompt_lens: list[int],
    n_kv_heads: int,
    head_dim: int,
    lora_end_idx: int,
    page_size: int = 128,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Run kv_cache_ragged_2m_iadd and return updated K/V cache contents.

    Args:
        kv_lora_output: [2*lora_end_idx, kv_dim] LoRA output with interleaved K/V
        prompt_lens: list of sequence lengths for each batch item
        n_kv_heads: number of KV heads
        head_dim: head dimension
        lora_end_idx: number of LoRA tokens
        page_size: KV cache page size

    Returns:
        (k_caches, v_caches): lists of K and V tensors per batch item
    """
    device = Accelerator(0)
    session = InferenceSession(devices=[device, CPU()])
    device_ref = DeviceRef.from_device(device)

    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)
    kv_dim = n_kv_heads * head_dim

    kv_params = MHAKVCacheParams(
        dtype=DTYPE,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=1,
        page_size=page_size,
        devices=[device_ref],
    )

    # The cache starts zeroed, which iadd needs since it adds to what is
    # already there.
    kv_runtime_inputs = paged_kv_cache_inputs(
        kv_params, prompt_lens, total_num_pages=16
    )

    kv_symbolic_inputs = kv_params.flattened_kv_inputs()

    with Graph(
        "kv_cache_2m_iadd_test",
        input_types=[
            TensorType(DTYPE, [2 * lora_end_idx, kv_dim], device=device_ref),
            TensorType(DType.uint32, [batch_size + 1], device=device_ref),
            TensorType(DType.int64, [lora_end_idx], device=DeviceRef.CPU()),
            TensorType(DType.int64, [1], device=DeviceRef.CPU()),
            *kv_symbolic_inputs,
        ],
    ) as graph:
        kv_input, row_offsets, end_idx, batch_len, *kv_inputs = graph.inputs
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())

        kv_collection = PagedCacheValues(
            kv_blocks=kv_inputs[0].buffer,
            cache_lengths=kv_inputs[1].tensor,
            lookup_table=kv_inputs[2].tensor,
            max_prompt_length=kv_inputs[3].tensor,
            max_cache_length=kv_inputs[4].tensor,
        )

        kv_cache_ragged_2m_iadd(
            kv_params=kv_params,
            a=kv_input.tensor,
            kv_collection=kv_collection,
            input_row_offsets=row_offsets.tensor,
            lora_end_idx=end_idx.tensor,
            batch_seq_len=batch_len.tensor,
            layer_idx=layer_idx,
        )
        graph.output(ops.constant(0, DType.int32, device=DeviceRef.CPU()))

    compiled = session.load(graph)

    # Prepare inputs
    input_row_offsets = np.array(
        [sum(prompt_lens[:i]) for i in range(batch_size + 1)], dtype=np.uint32
    )

    lora_end_idx_arr = np.zeros(lora_end_idx, dtype=np.int64)
    if lora_end_idx > 0:
        lora_end_idx_arr[0] = lora_end_idx

    batch_seq_len_arr = np.array([total_seq_len], dtype=np.int64)

    compiled.execute(
        to_max_tensor(kv_lora_output, device),
        Buffer.from_numpy(input_row_offsets).to(device),
        Buffer.from_numpy(lora_end_idx_arr),
        Buffer.from_numpy(batch_seq_len_arr),
        *kv_runtime_inputs.flatten(),
    )

    k_caches = dump_kv_cache_to_torch(
        kv_params, kv_runtime_inputs.kv_blocks, prompt_lens, KeyOrValue.KEY
    )
    v_caches = dump_kv_cache_to_torch(
        kv_params, kv_runtime_inputs.kv_blocks, prompt_lens, KeyOrValue.VALUE
    )

    return k_caches, v_caches


def verify_kv_cache_2m_iadd(
    prompt_lens: list[int],
    num_lora_seqs: int,
    n_kv_heads: int = 8,
    head_dim: int = 64,
    seed: int = 42,
) -> None:
    """
    Verify kv_cache_ragged_2m_iadd correctness.

    Args:
        prompt_lens: List of sequence lengths
        num_lora_seqs: Number of sequences with LoRA (from the start)
        n_kv_heads: Number of KV heads
        head_dim: Head dimension
        seed: Random seed
    """
    kv_dim = n_kv_heads * head_dim
    lora_end_idx = sum(prompt_lens[:num_lora_seqs])

    kv_lora_output = rand_tensor((2 * lora_end_idx, kv_dim), seed=seed)

    k_caches, v_caches = run_kv_cache_2m_iadd(
        kv_lora_output, prompt_lens, n_kv_heads, head_dim, lora_end_idx
    )

    # Verify LoRA sequences have correct values
    k_offset = 0
    for i in range(num_lora_seqs):
        seq_len = prompt_lens[i]
        k_expected = kv_lora_output[k_offset : k_offset + seq_len]
        v_expected = kv_lora_output[
            lora_end_idx + k_offset : lora_end_idx + k_offset + seq_len
        ]

        assert_close(k_caches[i], k_expected)
        assert_close(v_caches[i], v_expected)
        assert k_caches[i].abs().sum() > 0, (
            f"K cache[{i}] should not be all zeros"
        )
        assert v_caches[i].abs().sum() > 0, (
            f"V cache[{i}] should not be all zeros"
        )

        k_offset += seq_len

    # Verify non-LoRA sequences have zeros
    for i in range(num_lora_seqs, len(prompt_lens)):
        assert torch.allclose(
            k_caches[i], torch.zeros_like(k_caches[i]), atol=1e-6
        ), f"K cache[{i}] for non-LoRA should be zeros"
        assert torch.allclose(
            v_caches[i], torch.zeros_like(v_caches[i]), atol=1e-6
        ), f"V cache[{i}] for non-LoRA should be zeros"


@pytest.mark.parametrize(
    "prompt_lens, num_lora_seqs",
    [
        # Single sequence, all LoRA
        ([16], 1),
        # Two equal sequences, all LoRA
        ([8, 8], 2),
        # Two unequal sequences, all LoRA
        ([10, 20], 2),
        # Three sequences, all LoRA
        ([100, 50, 60], 3),
        # Two sequences, one LoRA (first only)
        ([500, 16], 1),
        # Three sequences, two LoRA (first two)
        ([23, 76, 12], 2),
    ],
)
def test_kv_cache_2m_iadd(prompt_lens: list[int], num_lora_seqs: int) -> None:
    """Test kv_cache_ragged_2m_iadd with various configurations."""
    verify_kv_cache_2m_iadd(
        prompt_lens=prompt_lens, num_lora_seqs=num_lora_seqs
    )
