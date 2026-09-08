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
"""Runtime K/V write into a scale-free FP8 paged KV cache (B200 / SM100).

A block-sparse (MSA) attention layer can use an FP8 e4m3 KV cache with *no*
per-block dequant scales (``quantized_kv_cache == False``,
``is_fp8_kv_dtype == True``). The MSA attention kernel reads that cache; the
K/V *write* is a BF16 QKV projection stored into FP8 paged blocks via
``rope_split_store_ragged`` (the ``mo.rope_split_store.ragged.paged`` op reached
through ``fused_qk_rms_norm_rope_ragged``), whose ``qkv_dtype`` and
``cache_dtype`` are independent -- the store casts to e4m3. The single-dtype
matmul-store ops (``matmul_kv_cache_ragged``, ``fused_qkv_ragged_matmul``)
cannot do this, so the rope-store path is the one used for a scale-free FP8
cache.

This isolates the write: it runs the identical store into a BF16 cache and an
FP8 cache and asserts the FP8-stored K/V match the BF16 baseline within FP8
rounding (cosine), plus finite/non-zero. Whole-stack accuracy (FP8-sparse vs
BF16 logits on a real checkpoint) is a separate, higher-level gate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import rope_split_store_ragged
from max.nn.kv_cache import MHAKVCacheParams, PagedCacheValues
from test_common.graph_utils import is_b100_b200
from test_common.simple_kv_cache import paged_kv_cache_inputs

_NUM_Q_HEADS = 64
_N_KV_HEADS = 8
_HEAD_DIM = 128
_PAGE_SIZE = 128
_PROMPT_LENS = [10, 30]


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


def _run_store(
    cache_dtype: DType,
    qkv: torch.Tensor,
    freqs_cis: torch.Tensor,
    device: Accelerator,
    session: InferenceSession,
) -> torch.Tensor:
    """Store one BF16 QKV projection into a paged cache of ``cache_dtype``.

    Returns the raw KV blocks ``[pages, 2, layers, page_size, heads, head_dim]``
    read back as fp32.
    """
    kv_params = MHAKVCacheParams(
        dtype=cache_dtype,
        n_kv_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
        num_layers=1,
        page_size=_PAGE_SIZE,
        devices=[DeviceRef.GPU()],
    )
    total_seq_len = sum(_PROMPT_LENS)
    qkv_dim = (_NUM_Q_HEADS + 2 * _N_KV_HEADS) * _HEAD_DIM

    qkv_type = TensorType(
        DType.bfloat16, [total_seq_len, qkv_dim], device=DeviceRef.GPU()
    )
    iro_type = TensorType(DType.uint32, ["iro_len"], device=DeviceRef.GPU())
    freqs_type = TensorType(
        DType.float32, [freqs_cis.shape[0], _HEAD_DIM], device=DeviceRef.GPU()
    )

    with Graph(
        f"rope_store_{cache_dtype}",
        input_types=[
            qkv_type,
            iro_type,
            freqs_type,
            *kv_params.flattened_kv_inputs(),
        ],
    ) as graph:
        (
            qkv_in,
            iro_in,
            freqs_in,
            blocks,
            cache_lengths,
            lookup,
            max_p,
            max_c,
            *_,
        ) = graph.inputs
        roped_q = rope_split_store_ragged(
            kv_params,
            qkv_in.tensor,
            iro_in.tensor,
            freqs_in.tensor,
            kv_collection=PagedCacheValues(
                blocks.buffer,
                cache_lengths.tensor,
                lookup.tensor,
                max_p.tensor,
                max_c.tensor,
            ),
            layer_idx=ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
            n_heads=_NUM_Q_HEADS,
            interleaved=True,
        )
        graph.output(roped_q)

    model = session.load(graph)

    offsets = np.zeros(len(_PROMPT_LENS) + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(_PROMPT_LENS)
    kv_inputs = paged_kv_cache_inputs(
        kv_params, _PROMPT_LENS, total_num_pages=8
    )

    qkv_buf = Buffer.from_dlpack(qkv).to(device)
    iro_buf = Buffer.from_dlpack(torch.from_numpy(offsets)).to(device)
    freqs_buf = Buffer.from_dlpack(freqs_cis).to(device)
    model.execute(qkv_buf, iro_buf, freqs_buf, *kv_inputs.flatten())

    return torch.from_dlpack(kv_inputs.kv_blocks).to(torch.float32).cpu()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a1, b1 = a.reshape(-1), b.reshape(-1)
    return float(torch.dot(a1, b1) / (a1.norm() * b1.norm()).clamp_min(1e-12))


def test_fp8_kv_store_matches_bf16() -> None:
    """A BF16 QKV projection stored into a scale-free FP8 cache is correct.

    Regression for FP8-sparse KV: the FP8 store must reproduce the BF16
    store's K/V (modulo FP8 rounding), not garbage / zeros / NaN.
    """
    if not is_b100_b200():
        pytest.skip("Native (scale-free) FP8 KV store requires B200 (SM100)")

    # Scale-free FP8: dtype fp8 but no quantization config -> not the scaled
    # quant KV cache, matching what the MSA kernel expects.
    fp8_params = MHAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        n_kv_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
        num_layers=1,
        page_size=_PAGE_SIZE,
        devices=[DeviceRef.GPU()],
    )
    assert fp8_params.is_fp8_kv_dtype
    assert not fp8_params.quantized_kv_cache
    assert fp8_params.kvcache_quant_config is None

    device = Accelerator()
    session = InferenceSession(devices=[device])

    total_seq_len = sum(_PROMPT_LENS)
    qkv_dim = (_NUM_Q_HEADS + 2 * _N_KV_HEADS) * _HEAD_DIM
    torch.manual_seed(0)
    # Small magnitudes keep values well inside the e4m3 range.
    qkv = torch.randn(total_seq_len, qkv_dim, dtype=torch.bfloat16) * 0.1
    freqs_cis = _make_freqs_cis(_PAGE_SIZE, _HEAD_DIM)

    bf16_blocks = _run_store(DType.bfloat16, qkv, freqs_cis, device, session)
    fp8_blocks = _run_store(
        DType.float8_e4m3fn, qkv, freqs_cis, device, session
    )

    assert torch.isfinite(fp8_blocks).all(), (
        "FP8 KV cache has non-finite values"
    )
    assert fp8_blocks.abs().sum() > 0, "FP8 KV cache was not written"

    cos = _cosine(fp8_blocks, bf16_blocks)
    assert cos >= 0.99, f"FP8 vs BF16 KV store cosine {cos:.4f} < 0.99"
