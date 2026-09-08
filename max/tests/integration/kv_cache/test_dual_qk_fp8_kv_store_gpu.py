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
"""In-place K update through a mixed-dtype dual paged KV cache (B200 / SM100).

MiniMax-M3 sparse layers fuse the main-GQA and lightning-indexer per-head
RMSNorm+RoPE + in-place paged K-cache update into one GPU launch
(``fused_dual_qk_rms_norm_rope_ragged``, reaching
``mo.fused_qk_rms_norm_rope.ragged.paged.dual``). The main GQA K cache can use
a scale-free FP8 e4m3 dtype with *no* per-block dequant scales while Q and the
indexer's K cache stay BF16, so the op's two paged caches have independent
dtypes.

This isolates the dual-launch update: it pre-populates identical raw
(un-normalized) K entries into a BF16 cache pair and a mixed FP8-main /
BF16-index cache pair, runs the op on both, and asserts the FP8-stored main-K
output matches the BF16 baseline within FP8 rounding (cosine), plus
finite/non-zero, while the index-K cache and both Q outputs -- which never
depend on the main cache's dtype -- are bit-exact between the two runs. Whole
-stack accuracy (FP8-sparse vs BF16 logits on a real checkpoint) is a
separate, higher-level gate.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import fused_dual_qk_rms_norm_rope_ragged
from max.nn.kv_cache import MHAKVCacheParams, PagedCacheValues
from max.nn.kv_cache.input_types import KVCacheInputsPerDevice
from test_common.graph_utils import is_b100_b200
from test_common.simple_kv_cache import paged_kv_cache_inputs

_KVRuntimeInputs = KVCacheInputsPerDevice[Buffer, Buffer]

_HEAD_DIM = 128
_MAIN_Q_HEADS = 4
_MAIN_KV_HEADS = 1
_INDEX_Q_HEADS = 1
_INDEX_KV_HEADS = 1
_PAGE_SIZE = 128
_PROMPT_LENS = [10, 30]
_LAYER_IDX = 0
_EPSILON = 1e-6

_TORCH_DTYPE = {
    DType.bfloat16: torch.bfloat16,
    DType.float8_e4m3fn: torch.float8_e4m3fn,
}


def _make_freqs_cis(max_seq: int, head_dim: int) -> torch.Tensor:
    """Interleaved (cos, sin) RoPE table ``[max_seq, head_dim]``, fp32."""
    pos = torch.arange(max_seq, dtype=torch.float32)
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (10000.0**exponent)
    angles = torch.outer(pos, inv_freq)
    freqs = torch.empty(max_seq, head_dim, dtype=torch.float32)
    freqs[:, 0::2] = torch.cos(angles)
    freqs[:, 1::2] = torch.sin(angles)
    return freqs


def _kv_params(dtype: DType, n_kv_heads: int) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=n_kv_heads,
        head_dim=_HEAD_DIM,
        num_layers=1,
        page_size=_PAGE_SIZE,
        devices=[DeviceRef.GPU()],
    )


def _token_positions(
    lut_host: npt.NDArray[np.uint32], cache_lengths_host: npt.NDArray[np.uint32]
) -> list[tuple[int, int]]:
    """Returns ``(block, in_page)`` per token, in the same order raw K is
    seeded / Q rows are laid out (batch-major, then token)."""
    positions = []
    for bs, length in enumerate(_PROMPT_LENS):
        cache_len = int(cache_lengths_host[bs])
        for t in range(length):
            ctx_pos = cache_len + t
            block = int(lut_host[bs, ctx_pos // _PAGE_SIZE])
            in_page = ctx_pos % _PAGE_SIZE
            positions.append((block, in_page))
    return positions


def _seed_raw_k(
    kv_rt: _KVRuntimeInputs,
    cache_dtype: DType,
    num_heads: int,
    raw_k_ground_truth: torch.Tensor,
    device: Accelerator,
) -> list[tuple[int, int]]:
    """Overwrites ``kv_rt.kv_blocks`` with known raw (un-normalized) K entries.

    ``raw_k_ground_truth`` is ``[total_seq_len, num_heads, head_dim]`` fp32,
    already fp8-e4m3-representable so storing it as either bf16 or fp8 is
    lossless -- isolating the comparison to this op's own store-cast rather
    than to divergent raw-K rounding between the two cache dtypes.

    Returns the per-token ``(block, in_page)`` positions for later readback.
    """
    lut_host = kv_rt.lookup_table.to_numpy()
    cache_lengths_host = kv_rt.cache_lengths.to_numpy()
    positions = _token_positions(lut_host, cache_lengths_host)

    blocks_host = torch.from_dlpack(kv_rt.kv_blocks).cpu().clone()
    raw_k = raw_k_ground_truth.to(_TORCH_DTYPE[cache_dtype])
    for tok, (block, in_page) in enumerate(positions):
        blocks_host[block, 0, _LAYER_IDX, in_page, :num_heads, :] = raw_k[tok]

    seeded_buf = Buffer.from_dlpack(blocks_host).to(device)
    kv_rt.kv_blocks.inplace_copy_from(seeded_buf)
    return positions


def _extract_tokens(
    kv_rt: _KVRuntimeInputs, positions: list[tuple[int, int]], num_heads: int
) -> torch.Tensor:
    """Reads back the per-token K entries at ``positions``, as fp32."""
    blocks_host = torch.from_dlpack(kv_rt.kv_blocks).to(torch.float32).cpu()
    out = torch.empty(len(positions), num_heads, _HEAD_DIM, dtype=torch.float32)
    for tok, (block, in_page) in enumerate(positions):
        out[tok] = blocks_host[block, 0, _LAYER_IDX, in_page, :num_heads, :]
    return out


def _run_dual(
    main_cache_dtype: DType,
    index_cache_dtype: DType,
    q_main: torch.Tensor,
    q_index: torch.Tensor,
    raw_main_k: torch.Tensor,
    raw_index_k: torch.Tensor,
    gammas: dict[str, torch.Tensor],
    freqs_cis: torch.Tensor,
    device: Accelerator,
    session: InferenceSession,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs one dual QK RMSNorm+RoPE + in-place paged K update.

    Returns ``(q_main_out, q_index_out, main_k_out, index_k_out)``, all fp32
    on CPU; the K tensors are per-token (``[total_seq_len, num_heads,
    head_dim]``), extracted via each run's own page-table assignment.
    """
    main_kv_params = _kv_params(main_cache_dtype, _MAIN_KV_HEADS)
    index_kv_params = _kv_params(index_cache_dtype, _INDEX_KV_HEADS)
    total_seq_len = sum(_PROMPT_LENS)

    q_main_type = TensorType(
        DType.bfloat16,
        [total_seq_len, _MAIN_Q_HEADS, _HEAD_DIM],
        device=DeviceRef.GPU(),
    )
    q_index_type = TensorType(
        DType.bfloat16,
        [total_seq_len, _INDEX_Q_HEADS, _HEAD_DIM],
        device=DeviceRef.GPU(),
    )
    iro_type = TensorType(DType.uint32, ["iro_len"], device=DeviceRef.GPU())
    freqs_type = TensorType(
        DType.float32, [freqs_cis.shape[0], _HEAD_DIM], device=DeviceRef.GPU()
    )
    gamma_type = TensorType(DType.bfloat16, [_HEAD_DIM], device=DeviceRef.GPU())

    with Graph(
        f"dual_qk_rope_store_{main_cache_dtype}_{index_cache_dtype}",
        input_types=[
            q_main_type,
            q_index_type,
            iro_type,
            freqs_type,
            gamma_type,  # q_main_gamma
            gamma_type,  # k_main_gamma
            gamma_type,  # q_index_gamma
            gamma_type,  # k_index_gamma
            *main_kv_params.flattened_kv_inputs(),
            *index_kv_params.flattened_kv_inputs(),
        ],
    ) as graph:
        (
            q_main_in,
            q_index_in,
            iro_in,
            freqs_in,
            gamma_main_q_in,
            gamma_main_k_in,
            gamma_index_q_in,
            gamma_index_k_in,
            *kv_inputs,
        ) = graph.inputs

        num_main_kv_inputs = len(main_kv_params.flattened_kv_inputs())
        (
            main_blocks,
            main_cache_lengths,
            main_lookup,
            main_max_p,
            main_max_c,
            *_main_rest,
        ) = kv_inputs[:num_main_kv_inputs]
        (
            index_blocks,
            index_cache_lengths,
            index_lookup,
            index_max_p,
            index_max_c,
            *_index_rest,
        ) = kv_inputs[num_main_kv_inputs:]

        q_main_out, q_index_out = fused_dual_qk_rms_norm_rope_ragged(
            main_kv_params,
            index_kv_params,
            q_main_in.tensor,
            q_index_in.tensor,
            iro_in.tensor,
            PagedCacheValues(
                main_blocks.buffer,
                main_cache_lengths.tensor,
                main_lookup.tensor,
                main_max_p.tensor,
                main_max_c.tensor,
            ),
            PagedCacheValues(
                index_blocks.buffer,
                index_cache_lengths.tensor,
                index_lookup.tensor,
                index_max_p.tensor,
                index_max_c.tensor,
            ),
            q_main_gamma=gamma_main_q_in.tensor,
            k_main_gamma=gamma_main_k_in.tensor,
            q_index_gamma=gamma_index_q_in.tensor,
            k_index_gamma=gamma_index_k_in.tensor,
            freqs_cis=freqs_in.tensor,
            main_epsilon=_EPSILON,
            index_epsilon=_EPSILON,
            layer_idx=ops.constant(
                _LAYER_IDX, DType.uint32, device=DeviceRef.CPU()
            ),
            weight_offset=0.0,
            interleaved=True,
        )
        graph.output(q_main_out, q_index_out)

    model = session.load(graph)

    main_kv_rt = paged_kv_cache_inputs(
        main_kv_params, _PROMPT_LENS, total_num_pages=8
    )
    index_kv_rt = paged_kv_cache_inputs(
        index_kv_params, _PROMPT_LENS, total_num_pages=8
    )

    main_positions = _seed_raw_k(
        main_kv_rt, main_cache_dtype, _MAIN_KV_HEADS, raw_main_k, device
    )
    index_positions = _seed_raw_k(
        index_kv_rt, index_cache_dtype, _INDEX_KV_HEADS, raw_index_k, device
    )

    offsets = np.zeros(len(_PROMPT_LENS) + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(_PROMPT_LENS)

    q_main_buf = Buffer.from_dlpack(q_main).to(device)
    q_index_buf = Buffer.from_dlpack(q_index).to(device)
    iro_buf = Buffer.from_dlpack(torch.from_numpy(offsets)).to(device)
    freqs_buf = Buffer.from_dlpack(freqs_cis).to(device)
    gamma_main_q_buf = Buffer.from_dlpack(gammas["main_q"]).to(device)
    gamma_main_k_buf = Buffer.from_dlpack(gammas["main_k"]).to(device)
    gamma_index_q_buf = Buffer.from_dlpack(gammas["index_q"]).to(device)
    gamma_index_k_buf = Buffer.from_dlpack(gammas["index_k"]).to(device)

    q_main_result, q_index_result = model.execute(
        q_main_buf,
        q_index_buf,
        iro_buf,
        freqs_buf,
        gamma_main_q_buf,
        gamma_main_k_buf,
        gamma_index_q_buf,
        gamma_index_k_buf,
        *main_kv_rt.flatten(),
        *index_kv_rt.flatten(),
    )

    main_k_out = _extract_tokens(main_kv_rt, main_positions, _MAIN_KV_HEADS)
    index_k_out = _extract_tokens(index_kv_rt, index_positions, _INDEX_KV_HEADS)
    q_main_out_host = torch.from_dlpack(q_main_result).to(torch.float32).cpu()
    q_index_out_host = torch.from_dlpack(q_index_result).to(torch.float32).cpu()

    return q_main_out_host, q_index_out_host, main_k_out, index_k_out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a1, b1 = a.reshape(-1), b.reshape(-1)
    return float(torch.dot(a1, b1) / (a1.norm() * b1.norm()).clamp_min(1e-12))


def _fp8_roundtrip(shape: tuple[int, ...]) -> torch.Tensor:
    """Small-magnitude fp32 values that round-trip losslessly through e4m3.

    Used for the "ground truth" raw K so storing it as either bf16 or fp8 is
    exact -- any divergence measured downstream is this op's own store-cast,
    not raw-K rounding that differs between the two cache dtypes.
    """
    return (torch.randn(shape) * 0.1).to(torch.float8_e4m3fn).to(torch.float32)


def test_dual_qk_fp8_main_cache_matches_bf16() -> None:
    """Mixed FP8-main/BF16-index dual QK RMSNorm+RoPE matches all-BF16.

    Regression for MiniMax-M3 FP8 KV: ``fused_dual_qk_rms_norm_rope_ragged``
    must accept a main-K cache dtype that differs from Q / the index-K cache
    (previously rejected by a compile-time assert requiring all three to
    match). The FP8-stored main-K must reproduce the BF16-stored main-K
    (modulo FP8 rounding, not garbage / zeros / NaN); the index-K cache and
    both Q outputs must be completely unaffected by the main cache's dtype.
    """
    if not is_b100_b200():
        pytest.skip("Native (scale-free) FP8 KV store requires B200 (SM100)")

    device = Accelerator()
    session = InferenceSession(devices=[device])

    total_seq_len = sum(_PROMPT_LENS)
    torch.manual_seed(0)
    # Small magnitudes keep Q well inside the e4m3 range too, so the RoPE'd,
    # saturating-cast main-K output stays representable.
    q_main = (torch.randn(total_seq_len, _MAIN_Q_HEADS, _HEAD_DIM) * 0.1).to(
        torch.bfloat16
    )
    q_index = (torch.randn(total_seq_len, _INDEX_Q_HEADS, _HEAD_DIM) * 0.1).to(
        torch.bfloat16
    )
    raw_main_k = _fp8_roundtrip((total_seq_len, _MAIN_KV_HEADS, _HEAD_DIM))
    raw_index_k = _fp8_roundtrip((total_seq_len, _INDEX_KV_HEADS, _HEAD_DIM))
    gammas = {
        name: torch.ones(_HEAD_DIM, dtype=torch.bfloat16)
        for name in ("main_q", "main_k", "index_q", "index_k")
    }
    freqs_cis = _make_freqs_cis(_PAGE_SIZE, _HEAD_DIM)

    q_main_bf16, q_index_bf16, main_k_bf16, index_k_bf16 = _run_dual(
        DType.bfloat16,
        DType.bfloat16,
        q_main,
        q_index,
        raw_main_k,
        raw_index_k,
        gammas,
        freqs_cis,
        device,
        session,
    )
    q_main_mixed, q_index_mixed, main_k_fp8, index_k_mixed = _run_dual(
        DType.float8_e4m3fn,
        DType.bfloat16,
        q_main,
        q_index,
        raw_main_k,
        raw_index_k,
        gammas,
        freqs_cis,
        device,
        session,
    )

    assert torch.isfinite(main_k_fp8).all(), (
        "FP8 main-K cache has non-finite values"
    )
    assert main_k_fp8.abs().sum() > 0, "FP8 main-K cache was not written"

    cos = _cosine(main_k_fp8, main_k_bf16)
    assert cos >= 0.99, f"FP8 vs BF16 main-K cosine {cos:.4f} < 0.99"

    # The index-K cache is BF16 in both configs; it must be completely
    # unaffected by the main cache's dtype.
    torch.testing.assert_close(index_k_mixed, index_k_bf16, rtol=0, atol=0)
    # Q never depends on K cache dtype.
    torch.testing.assert_close(q_main_mixed, q_main_bf16, rtol=0, atol=0)
    torch.testing.assert_close(q_index_mixed, q_index_bf16, rtol=0, atol=0)
