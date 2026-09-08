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
"""Bit-exactness of ``mo.msa.attention.ragged.paged.mxfp8`` (MI355 / gfx950).

The MXFP8 MSA attention op replaces the pair ``mo.msa.attention.ragged.paged``
+ ``mo.quantize.dynamic.block.scaled`` that MiniMax-M3 runs to feed its MXFP8
o_proj, fusing the quantize into the split-K combine on decode (KERN-3384).
Its contract is byte equality with that pair -- the activations feed a matmul,
so the fusion must be a dispatch win and nothing else.

One graph runs both paths on identical inputs and this test compares every
FP8 byte and every E8M0 scale byte. The cases pick the op's three runtime
routes via ``max_prompt_length`` -- single-token decode (split-K, the fused
kernel), speculative decode (packed draft tokens, also split-K), and prefill
(BF16 attention then the stock quantize, issued from inside the op) -- and
within decode they span the partition counts production resolves, including
the partial-staging-chunk counts (2 and 4) where the fused accumulator's
update count has to match the reference's exactly.

The paged cache is hand-built rather than going through
``PagedKVCacheManager``: both paths read the same blocks, so the cache
contents only need to be identical, not meaningful. Kernel-level adversarial
coverage (ties, dead blocks, outliers) lives in
``Kernels/test/msa/test_msa_amd_splitk_reduce_quant.mojo``; this test owns the
graph-op plumbing: operand order, output layouts, and route selection.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.kernels import (
    msa_sparse_attention_ragged,
    msa_sparse_attention_ragged_mxfp8,
    quantize_dynamic_block_scaled,
)
from max.nn.kv_cache import MHAKVCacheParams, PagedCacheValues

# MiniMax-M3 TP4 attention shape: 16 query heads over a single KV head,
# head_dim 128, 128-token pages, 16 gathered blocks per token.
_NUM_Q_HEADS = 16
_N_KV_HEADS = 1
_HEAD_DIM = 128
_PAGE_SIZE = 128
_SPARSE_BLOCK_SIZE = 128
_TOPK = 16
_SF_VECTOR_SIZE = 32
_NUM_LAYERS = 1
_ROW_WIDTH = _NUM_Q_HEADS * _HEAD_DIM


def _to_uint8(buf: Buffer) -> np.ndarray:
    """Reads a device buffer back as raw bytes (FP8 dtypes lack host views)."""
    return buf.view(DType.uint8).to_numpy().reshape(-1)


def _run_case(
    session: InferenceSession,
    device: Accelerator,
    *,
    batch: int,
    q_per_seq: int,
    cache_len: int,
) -> None:
    """Runs both paths for one route and asserts byte equality.

    ``q_per_seq`` selects the op's runtime route through
    ``max_prompt_length``: 1 decode, 2-8 speculative decode, >8 prefill.
    """
    rows = batch * q_per_seq
    keys_per_seq = cache_len + q_per_seq
    pages_per_seq = -(-keys_per_seq // _PAGE_SIZE)
    total_pages = batch * pages_per_seq
    scale_cols = _ROW_WIDTH // _SF_VECTOR_SIZE

    gpu = DeviceRef.GPU()
    graph = Graph(
        f"msa_mxfp8_vs_pair_q{q_per_seq}",
        input_types=[
            TensorType(
                DType.bfloat16, [rows, _NUM_Q_HEADS, _HEAD_DIM], device=gpu
            ),
            TensorType(DType.uint32, [batch + 1], device=gpu),
            TensorType(DType.uint32, [batch + 1], device=gpu),
            TensorType(DType.uint32, [1], device=DeviceRef.CPU()),
            BufferType(
                DType.bfloat16,
                [
                    total_pages,
                    2,
                    _NUM_LAYERS,
                    _PAGE_SIZE,
                    _N_KV_HEADS,
                    _HEAD_DIM,
                ],
                device=gpu,
            ),
            TensorType(DType.uint32, [batch], device=gpu),
            TensorType(DType.uint32, [batch, pages_per_seq], device=gpu),
            TensorType(DType.uint32, [1], device=DeviceRef.CPU()),
            TensorType(DType.uint32, [1], device=DeviceRef.CPU()),
            TensorType(DType.int64, [2], device=gpu),
            TensorType(DType.int32, [_N_KV_HEADS, rows, _TOPK], device=gpu),
        ],
    )
    with graph:
        (
            q_in,
            iro_in,
            cro_in,
            tcl_in,
            blocks_in,
            cl_in,
            lut_in,
            max_p_in,
            max_c_in,
            scalar_args_in,
            d_indices_in,
        ) = graph.inputs

        kv_params = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=_N_KV_HEADS,
            head_dim=_HEAD_DIM,
            num_layers=_NUM_LAYERS,
            page_size=_PAGE_SIZE,
            devices=[gpu],
        )
        kv_collection = PagedCacheValues(
            blocks_in.buffer,
            cl_in.tensor,
            lut_in.tensor,
            max_p_in.tensor,
            max_c_in.tensor,
            attention_dispatch_metadata=scalar_args_in.tensor,
        )
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        scale = float(_HEAD_DIM**-0.5)

        ref_bf16 = msa_sparse_attention_ragged(
            kv_params=kv_params,
            input=q_in.tensor,
            input_row_offsets=iro_in.tensor,
            cache_row_offsets=cro_in.tensor,
            total_context_length=tcl_in.tensor,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            block_indices=d_indices_in.tensor,
            group=_NUM_Q_HEADS // _N_KV_HEADS,
            topk=_TOPK,
            sparse_block_size=_SPARSE_BLOCK_SIZE,
            scale=scale,
        )
        ref_fp8, ref_scales = quantize_dynamic_block_scaled(
            ops.reshape(ref_bf16, shape=[rows, _ROW_WIDTH]),
            sf_vector_size=_SF_VECTOR_SIZE,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
        fused_fp8, fused_scales = msa_sparse_attention_ragged_mxfp8(
            kv_params=kv_params,
            input=q_in.tensor,
            input_row_offsets=iro_in.tensor,
            cache_row_offsets=cro_in.tensor,
            total_context_length=tcl_in.tensor,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            block_indices=d_indices_in.tensor,
            group=_NUM_Q_HEADS // _N_KV_HEADS,
            topk=_TOPK,
            sparse_block_size=_SPARSE_BLOCK_SIZE,
            scale=scale,
        )

        graph.output(ref_fp8, ref_scales, fused_fp8, fused_scales)

    model = session.load(graph)

    torch.manual_seed(rows)
    q = (torch.randn(rows, _NUM_Q_HEADS, _HEAD_DIM) * 0.5).to(torch.bfloat16)
    # K/V page contents only have to be identical for both paths, not
    # meaningful: no store ran, both ops read the same bytes. Generated in
    # BF16 rather than via an FP32 tensor because the larger-batch cases would
    # otherwise build a multi-hundred-MB host temporary.
    blocks = (
        torch.randn(
            total_pages,
            2,
            _NUM_LAYERS,
            _PAGE_SIZE,
            _N_KV_HEADS,
            _HEAD_DIM,
            dtype=torch.bfloat16,
        )
        * 0.5
    )

    iro = np.arange(batch + 1, dtype=np.uint32) * q_per_seq
    cro = np.arange(batch + 1, dtype=np.uint32) * keys_per_seq
    tcl = np.array([batch * keys_per_seq], dtype=np.uint32)
    cache_lengths = np.full(batch, cache_len, dtype=np.uint32)
    lut = np.arange(total_pages, dtype=np.uint32).reshape(batch, pages_per_seq)
    max_p = np.array([q_per_seq], dtype=np.uint32)
    max_c = np.array([cache_len], dtype=np.uint32)
    scalar_args = np.array([batch, keys_per_seq], dtype=np.int64)
    # Logical block ids within each sequence; the indexer pads with -1 when a
    # sequence has fewer than topk selectable blocks (mask_unselected).
    d_idx = np.full((_N_KV_HEADS, rows, _TOPK), -1, dtype=np.int32)
    for j in range(min(pages_per_seq, _TOPK)):
        d_idx[:, :, j] = j

    outs = model.execute(
        Buffer.from_dlpack(q).to(device),
        Buffer.from_numpy(iro).to(device),
        Buffer.from_numpy(cro).to(device),
        Buffer.from_numpy(tcl),
        Buffer.from_dlpack(blocks).to(device),
        Buffer.from_numpy(cache_lengths).to(device),
        Buffer.from_numpy(lut).to(device),
        Buffer.from_numpy(max_p),
        Buffer.from_numpy(max_c),
        Buffer.from_numpy(scalar_args).to(device),
        Buffer.from_numpy(d_idx).to(device),
    )
    ref_fp8_b, ref_scales_b, fused_fp8_b, fused_scales_b = (
        outs[0],
        outs[1],
        outs[2],
        outs[3],
    )
    assert isinstance(ref_fp8_b, Buffer)
    assert isinstance(ref_scales_b, Buffer)
    assert isinstance(fused_fp8_b, Buffer)
    assert isinstance(fused_scales_b, Buffer)

    ref_data = _to_uint8(ref_fp8_b)
    fused_data = _to_uint8(fused_fp8_b)
    ref_sc = _to_uint8(ref_scales_b)
    fused_sc = _to_uint8(fused_scales_b)

    assert ref_data.size == rows * _ROW_WIDTH
    assert ref_sc.size == rows * scale_cols
    # A silently dead kernel would pass a pure equality check.
    assert int(ref_data.astype(np.int64).sum()) > 0, (
        "reference output is all zero"
    )

    data_mismatch = int((ref_data != fused_data).sum())
    scale_mismatch = int((ref_sc != fused_sc).sum())
    assert data_mismatch == 0, (
        f"q_per_seq={q_per_seq}: {data_mismatch}/{ref_data.size} FP8 bytes"
        " differ from the unfused pair"
    )
    assert scale_mismatch == 0, (
        f"q_per_seq={q_per_seq}: {scale_mismatch}/{ref_sc.size} E8M0"
        " scales differ from the unfused pair"
    )


@pytest.mark.skipif(
    accelerator_api() != "hip",
    reason="mo.msa.attention.ragged.paged.mxfp8 is AMD (gfx950) only",
)
def test_msa_mxfp8_matches_unfused_pair() -> None:
    """The fused op is byte-identical to attention + quantize on all routes."""
    device = Accelerator()
    session = InferenceSession(devices=[device])

    # The decode cases below pin all three partition counts the fused route
    # runs at in production. `np` depends only on `rows = batch * q_per_seq`,
    # because the heuristic's `num_keys` is always `topk * page_size` (2048)
    # -- not the cache length (`decode.mojo`, `max_cache_valid_length`).
    # Tracing `get_mha_decoding_max_num_partitions[16, 16]` for 1 kv head:
    # `split_floor = 128`, `pages = 16`, `ctas_per_partition = rows`, and
    # single-kv-head MHA takes the two-wave target `min(2 * 256 / rows, 16)`
    # (rows < 256) or the high-occupancy `ceildiv(2048, 256 * rows // 64)`
    # (rows >= 256). Hence 16 at rows 4, 4 at rows 128, 2 at rows 256.
    #
    # np in {2, 4} is the case that matters: the staged accumulate covers 8
    # partitions per chunk, so only a partial chunk can expose an accumulator
    # that runs more updates than `mha_splitk_reduce` does.
    _run_case(session, device, batch=4, q_per_seq=1, cache_len=2047)  # np 16
    _run_case(session, device, batch=128, q_per_seq=1, cache_len=2047)  # np 4
    _run_case(session, device, batch=256, q_per_seq=1, cache_len=2047)  # np 2
    # Speculative decode: 4 packed draft tokens/seq; real causal masking
    # against the trailing selected block, still the split-K route.
    _run_case(session, device, batch=2, q_per_seq=4, cache_len=2043)
    # Prefill (> MAX_SPEC_DRAFT): BF16 attention into scratch, then the stock
    # quantize inside the op.
    _run_case(session, device, batch=2, q_per_seq=32, cache_len=0)
