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
"""Unit tests for sgmv_qkv_lora_kernel."""

import numpy as np
import numpy.typing as npt
import pytest
import torch
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.torch import max_dtype_to_torch
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import sgmv_qkv_lora_kernel
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


def calc_input_row_offsets(seq_lens: list[int]) -> npt.NDArray[np.uint32]:
    """Calculate cumulative input row offsets from sequence lengths."""
    return np.cumsum([0] + seq_lens, dtype=np.uint32)


def torch_sgmv_qkv_lora_ref(
    input_tensor: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b_q: torch.Tensor,
    lora_b_kv: torch.Tensor,
    lora_ids: npt.NDArray[np.int32],
    offsets: npt.NDArray[np.uint32],
    lora_end_idx: int,
    max_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference implementation for sgmv_qkv_lora_kernel.

    Computes Q, K, V LoRA outputs:
    1. Shrink: input @ lora_a.T -> [3, M, rank] (Q, K, V intermediate)
    2. Expand Q: q_intermediate @ lora_b_q.T -> Q output
    3. Expand KV: kv_intermediate @ lora_b_kv.T -> K, V outputs

    Args:
        input_tensor: [M, hidden_dim]
        lora_a: [num_adapters, 3*rank, hidden_dim] - QKV shrink weights
        lora_b_q: [num_adapters, q_dim, rank] - Q expand weights
        lora_b_kv: [2*num_adapters, kv_dim, rank] - KV expand weights (K then V)
        lora_ids: [num_groups] adapter IDs
        offsets: [num_groups + 1] token boundaries
        lora_end_idx: number of LoRA tokens
        max_rank: maximum rank

    Returns:
        (q_output, k_output, v_output)
    """
    combined_rank = lora_a.shape[1]
    rank = combined_rank // 3
    num_adapters = lora_a.shape[0]
    M = input_tensor.shape[0]
    q_dim = lora_b_q.shape[1]
    kv_dim = lora_b_kv.shape[1]

    # QKV shrink -> [3, lora_end_idx, rank]
    qkv_shrink = torch.zeros(
        3,
        lora_end_idx,
        max_rank,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    for g in range(len(lora_ids)):
        start = offsets[g]
        end = min(offsets[g + 1], lora_end_idx)
        adapter_id = lora_ids[g]
        if adapter_id >= 0 and end > start:
            # [end-start, hidden] @ [3*rank, hidden].T -> [end-start, 3*rank]
            flat = input_tensor[start:end] @ lora_a[adapter_id].T
            # Reshape to [end-start, 3, rank] then transpose
            reshaped = flat.reshape(end - start, 3, rank).permute(1, 0, 2)
            qkv_shrink[:, start:end, :rank] = reshaped

    # Q expand -> [M, q_dim]
    q_output = torch.zeros(
        M, q_dim, dtype=input_tensor.dtype, device=input_tensor.device
    )
    q_intermediate = qkv_shrink[0, :, :rank]  # [lora_end_idx, rank]

    for g in range(len(lora_ids)):
        start = offsets[g]
        end = min(offsets[g + 1], lora_end_idx)
        adapter_id = lora_ids[g]
        if adapter_id >= 0 and end > start:
            q_output[start:end] = (
                q_intermediate[start:end] @ lora_b_q[adapter_id].T
            )

    # KV expand -> K [lora_end_idx, kv_dim], V [lora_end_idx, kv_dim]
    k_output = torch.zeros(
        lora_end_idx,
        kv_dim,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    v_output = torch.zeros(
        lora_end_idx,
        kv_dim,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    k_intermediate = qkv_shrink[1, :, :rank]  # [lora_end_idx, rank]
    v_intermediate = qkv_shrink[2, :, :rank]  # [lora_end_idx, rank]

    for g in range(len(lora_ids)):
        start = offsets[g]
        end = min(offsets[g + 1], lora_end_idx)
        adapter_id = lora_ids[g]
        if adapter_id >= 0 and end > start:
            # K uses first half of lora_b_kv, V uses second half
            k_output[start:end] = (
                k_intermediate[start:end] @ lora_b_kv[adapter_id].T
            )
            v_output[start:end] = (
                v_intermediate[start:end]
                @ lora_b_kv[num_adapters + adapter_id].T
            )

    return q_output, k_output, v_output


def run_sgmv_qkv_lora_kernel(
    input_tensor: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b_q: torch.Tensor,
    lora_b_kv: torch.Tensor,
    seq_lens: list[int],
    lora_ids: npt.NDArray[np.int32],
    grouped_offsets: npt.NDArray[np.uint32],
    lora_end_idx: int,
    n_kv_heads: int,
    head_dim: int,
    max_rank: int,
    page_size: int = 128,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """
    Run sgmv_qkv_lora_kernel and return Q output and updated K/V cache contents.

    Args:
        input_tensor: [M, hidden_dim] input (all sequences, LoRA + non-LoRA)
        lora_a: [num_adapters, 3*rank, hidden_dim] QKV shrink weights
        lora_b_q: [num_adapters, q_dim, rank] Q expand weights
        lora_b_kv: [2*num_adapters, kv_dim, rank] KV expand weights
        seq_lens: list of sequence lengths (all sequences)
        lora_ids: [num_lora_groups] adapter IDs (only LoRA groups)
        grouped_offsets: [num_lora_groups + 1] offsets (only LoRA groups)
        lora_end_idx: number of LoRA tokens
        n_kv_heads: number of KV heads
        head_dim: head dimension
        max_rank: maximum LoRA rank
        page_size: KV cache page size

    Returns:
        (q_output, k_caches, v_caches)
    """
    device = Accelerator(0)
    session = InferenceSession(devices=[device, CPU()])
    device_ref = DeviceRef.from_device(device)

    M = input_tensor.shape[0]
    hidden_dim = input_tensor.shape[1]
    num_adapters = lora_a.shape[0]
    combined_rank = lora_a.shape[1]
    q_dim = lora_b_q.shape[1]
    kv_dim = lora_b_kv.shape[1]

    total_seq_len = sum(seq_lens)

    # input_row_offsets contains all sequences
    input_row_offsets = calc_input_row_offsets(seq_lens)

    # Build the fused LoRA-B weight the single-launch kernel consumes:
    # [num_adapters, q_dim + 2*kv_dim, rank], rows ordered Q | K | V per adapter.
    # K uses lora_b_kv[g], V uses lora_b_kv[num_adapters + g] (matching the
    # torch reference oracle).
    fused_b = torch.cat(
        [
            lora_b_q,
            lora_b_kv[:num_adapters],
            lora_b_kv[num_adapters:],
        ],
        dim=1,
    )

    kv_params = MHAKVCacheParams(
        dtype=DTYPE,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=1,
        page_size=page_size,
        devices=[device_ref],
    )

    # The cache starts zeroed, which the comparison below relies on.
    kv_runtime_inputs = paged_kv_cache_inputs(
        kv_params, seq_lens, total_num_pages=32
    )

    kv_symbolic_inputs = kv_params.get_symbolic_inputs().inputs[0]

    with Graph(
        "sgmv_qkv_lora_kernel_test",
        input_types=[
            TensorType(DTYPE, [M, hidden_dim], device=device_ref),
            TensorType(
                DTYPE,
                [num_adapters, combined_rank, hidden_dim],
                device=device_ref,
            ),
            TensorType(
                DTYPE,
                [num_adapters, q_dim + 2 * kv_dim, max_rank],
                device=device_ref,
            ),
            TensorType(DType.int32, ["lora_ids"], device=device_ref),
            TensorType(
                DType.uint32, ["lora_grouped_offsets"], device=device_ref
            ),
            TensorType(DType.uint32, ["input_row_offsets"], device=device_ref),
            TensorType(DType.int64, ["lora_end"], device=DeviceRef.CPU()),
            TensorType(DType.int64, [1], device=DeviceRef.CPU()),
            *kv_symbolic_inputs.flatten(),
        ],
    ) as graph:
        (
            x,
            a_in,
            b_in,
            ids,
            grouped_offs,
            row_offs,
            end_idx,
            batch_len,
            *kv_inputs,
        ) = graph.inputs

        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())

        kv_collection = PagedCacheValues(
            kv_blocks=kv_inputs[0].buffer,
            cache_lengths=kv_inputs[1].tensor,
            lookup_table=kv_inputs[2].tensor,
            max_prompt_length=kv_inputs[3].tensor,
            max_cache_length=kv_inputs[4].tensor,
        )

        q_out = sgmv_qkv_lora_kernel(
            input=x.tensor,
            lora_a=a_in.tensor,
            lora_b=b_in.tensor,
            lora_ids=ids.tensor,
            input_row_offsets=row_offs.tensor,
            lora_grouped_offsets=grouped_offs.tensor,
            lora_end_idx=end_idx.tensor,
            batch_seq_len=batch_len.tensor,
            kv_collection=kv_collection,
            kv_params=kv_params,
            layer_idx=layer_idx,
            q_dim=q_dim,
            kv_dim=kv_dim,
            max_lora_seq_len=max(seq_lens),
            max_rank=max_rank,
        )
        graph.output(q_out)

    compiled = session.load(graph)

    # Prepare lora_end_idx array
    lora_end_idx_arr = np.zeros(lora_end_idx, dtype=np.int64)
    if lora_end_idx > 0:
        lora_end_idx_arr[0] = lora_end_idx

    batch_seq_len_arr = np.array([total_seq_len], dtype=np.int64)

    result = compiled.execute(
        to_max_tensor(input_tensor, device),
        to_max_tensor(lora_a, device),
        to_max_tensor(fused_b, device),
        Buffer.from_numpy(lora_ids.astype(np.int32)).to(device),
        Buffer.from_numpy(grouped_offsets.astype(np.uint32)).to(device),
        Buffer.from_numpy(input_row_offsets.astype(np.uint32)).to(device),
        Buffer.from_numpy(lora_end_idx_arr),
        Buffer.from_numpy(batch_seq_len_arr),
        *kv_runtime_inputs.flatten(),
    )

    q_output = from_dlpack(result[0])

    k_caches = dump_kv_cache_to_torch(
        kv_params, kv_runtime_inputs.kv_blocks, seq_lens, KeyOrValue.KEY
    )
    v_caches = dump_kv_cache_to_torch(
        kv_params, kv_runtime_inputs.kv_blocks, seq_lens, KeyOrValue.VALUE
    )

    return q_output, k_caches, v_caches


def verify_sgmv_qkv_lora_kernel(
    seq_lens: list[int],
    lora_ids: list[int],
    num_adapters: int,
    hidden_dim: int = 256,
    q_dim: int = 256,
    kv_dim: int = 64,
    rank: int = 8,
    seed: int = 42,
) -> None:
    """
    Verify sgmv_qkv_lora_kernel correctness for a given configuration.

    Args:
        seq_lens: List of sequence lengths (all sequences, LoRA first then non-LoRA)
        lora_ids: List of adapter IDs for LoRA sequences only (-1 excluded).
                  Length determines how many sequences have LoRA applied.
        num_adapters: Number of distinct LoRA adapters
        hidden_dim: Hidden dimension
        q_dim: Q output dimension
        kv_dim: KV output dimension
        rank: LoRA rank
        seed: Random seed for reproducibility
    """
    n_kv_heads = 1
    head_dim = kv_dim

    num_lora_seqs = len(lora_ids)
    lora_seq_lens = seq_lens[:num_lora_seqs]
    nolora_seq_lens = seq_lens[num_lora_seqs:]

    lora_end_idx = sum(lora_seq_lens)
    M = sum(seq_lens)

    input_tensor = rand_tensor((M, hidden_dim), seed=seed)
    lora_a = rand_tensor((num_adapters, 3 * rank, hidden_dim), seed=seed + 1)
    lora_b_q = rand_tensor((num_adapters, q_dim, rank), seed=seed + 2)
    lora_b_kv = rand_tensor((2 * num_adapters, kv_dim, rank), seed=seed + 3)

    lora_ids_arr = np.array(lora_ids, dtype=np.int32)
    grouped_offsets = calc_input_row_offsets(lora_seq_lens)

    q_actual, k_caches, v_caches = run_sgmv_qkv_lora_kernel(
        input_tensor,
        lora_a,
        lora_b_q,
        lora_b_kv,
        seq_lens,
        lora_ids_arr,
        grouped_offsets,
        lora_end_idx,
        n_kv_heads,
        head_dim,
        max_rank=rank,
    )

    q_expected, k_expected, v_expected = torch_sgmv_qkv_lora_ref(
        input_tensor,
        lora_a,
        lora_b_q,
        lora_b_kv,
        lora_ids_arr,
        grouped_offsets,
        lora_end_idx,
        max_rank=rank,
    )

    if lora_end_idx > 0:
        assert_close(q_actual[:lora_end_idx], q_expected[:lora_end_idx])
        assert q_actual[:lora_end_idx].abs().sum() > 0, (
            "Q output should be non-zero"
        )

    k_offset = 0
    for i, seq_len in enumerate(lora_seq_lens):
        assert_close(
            k_caches[i],
            k_expected[k_offset : k_offset + seq_len],
            rtol=RTOL,
            atol=ATOL,
        )
        assert_close(
            v_caches[i],
            v_expected[k_offset : k_offset + seq_len],
            rtol=RTOL,
            atol=ATOL,
        )
        assert k_caches[i].abs().sum() > 0, f"K cache[{i}] should be non-zero"
        assert v_caches[i].abs().sum() > 0, f"V cache[{i}] should be non-zero"
        k_offset += seq_len

    # Verify non-LoRA sequences have zeros in KV cache
    for i, _ in enumerate(nolora_seq_lens):
        cache_idx = num_lora_seqs + i
        assert torch.allclose(
            k_caches[cache_idx],
            torch.zeros_like(k_caches[cache_idx]),
            atol=1e-6,
        ), f"K cache[{cache_idx}] for non-LoRA should be zeros"
        assert torch.allclose(
            v_caches[cache_idx],
            torch.zeros_like(v_caches[cache_idx]),
            atol=1e-6,
        ), f"V cache[{cache_idx}] for non-LoRA should be zeros"


@pytest.mark.parametrize(
    "seq_lens, lora_ids, num_adapters",
    [
        # Single sequence with single adapter
        ([16], [0], 1),
        # Multiple sequences, same adapter
        ([8, 12, 10], [0, 0, 0], 1),
        # Multiple sequences, different adapters
        ([8, 12, 10, 14], [0, 1, 2, 3], 4),
        # 3 LoRA + 3 non-LoRA, single adapter
        ([8, 10, 12, 8, 10, 12], [0, 0, 0], 1),
        # 4 LoRA + 2 non-LoRA, multiple adapters
        ([8, 10, 12, 14, 8, 10], [0, 1, 2, 1], 3),
        # 1 LoRA + 3 non-LoRA
        ([16, 8, 10, 12], [0], 1),
        # 5 LoRA + 3 non-LoRA
        ([8, 10, 12, 8, 10, 12, 8, 10], [0, 1, 0, 1, 2], 3),
    ],
)
def test_sgmv_qkv_lora_kernel(
    seq_lens: list[int],
    lora_ids: list[int],
    num_adapters: int,
) -> None:
    """Test sgmv_qkv_lora_kernel with various configurations."""
    verify_sgmv_qkv_lora_kernel(
        seq_lens=seq_lens,
        lora_ids=lora_ids,
        num_adapters=num_adapters,
    )
