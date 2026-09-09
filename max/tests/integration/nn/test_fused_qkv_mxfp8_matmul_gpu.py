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
"""Numeric test for the fused MXFP8 QKV-matmul-with-KV-write kernel.

Exercises the ``mo.fused_qkv_matmul.ragged.paged.scale.mxfp8`` path (and its
``.amd`` CDNA4 sibling, which differs only in taking rank-2 rather than rank-5
E8M0 scales) that ``quantized_fused_qkv_matmul`` uses for ``QuantFormat.MXFP8``:
the activation and the concatenated QKV weight are quantized to
``float8_e4m3fn`` with E8M0 block scales, the fused matmul runs, the Q
projection is returned, and K/V are written in place into a paged KV cache.

It checks two things:

1. The returned Q projection against an fp32 reference of the un-quantized
   ``a @ wqkv.T``.
2. The K/V cache contents against a bf16 reference path. Both paths write
   through the same paged-cache store epilogue, so the raw ``kv_blocks``
   buffers are directly comparable element-wise without decoding the paged
   layout. This is what confirms K and V land in the right slots.

Tolerances absorb the MXFP8 round-trip but are tight enough to catch a wrong
layout or wrong kernel.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.kernels import (
    _fused_qkv_index_ragged_matmul_scaled_mxfp8,
    _fused_qkv_ragged_matmul_scaled_mxfp8,
    dynamic_block_scaled_matmul_amd,
    fused_qkv_ragged_matmul,
    quantize_dynamic_block_scaled,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParams,
    MHAKVCacheParams,
    MLAKVCacheParams,
    PagedCacheValues,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.quant_ops import quantized_fused_qkv_index_matmul
from test_common.graph_utils import is_b100_b200
from test_common.simple_kv_cache import paged_kv_cache_inputs


def _skip_if_not_supported() -> None:
    if accelerator_count() == 0:
        pytest.skip("No GPU available for MXFP8 fused-QKV test")
    if accelerator_api() == "hip":
        pytest.skip("MXFP8 block-scaled MMA only supports NVIDIA GPUs")
    if not is_b100_b200():
        pytest.skip("MXFP8 block-scaled MMA requires B100 or B200 (SM100)")


def _skip_if_no_fused_qkv_mxfp8() -> None:
    """The 3-way fused QKV has both an SM100 and a CDNA4 op behind one wrapper."""
    if accelerator_count() == 0:
        pytest.skip("No GPU available for MXFP8 fused-QKV test")
    if accelerator_api() != "hip" and not is_b100_b200():
        pytest.skip(
            "MXFP8 block-scaled MMA requires B100/B200 (SM100) or CDNA4"
        )


def _skip_if_not_amd() -> None:
    if accelerator_count() == 0:
        pytest.skip("No GPU available for the CDNA4 stacked QKV+index test")
    if accelerator_api() != "hip":
        pytest.skip("The stacked QKV+index path is the CDNA4 path")


def _mxfp8_quant_config() -> QuantConfig:
    """MXFP8 dynamic-activation-quant config, as MiniMax-M3 builds for attention."""
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, 32),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, 32),
        ),
        mlp_quantized_layers={0},
        attn_quantized_layers={0},
        format=QuantFormat.MXFP8,
    )


def _cosine_and_rel_l2(out: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    out_flat = out.reshape(-1).astype(np.float32)
    ref_flat = ref.reshape(-1).astype(np.float32)
    cos = float(
        np.dot(out_flat, ref_flat)
        / (np.linalg.norm(out_flat) * np.linalg.norm(ref_flat) + 1e-12)
    )
    rel = float(
        np.linalg.norm(out_flat - ref_flat) / (np.linalg.norm(ref_flat) + 1e-12)
    )
    return cos, rel


def _make_cache(
    kv_params: KVCacheParams, seq_len: int
) -> KVCacheInputsPerDevice[Buffer, Buffer]:
    """Allocate a single request's KV cache and return its runtime inputs."""
    return paged_kv_cache_inputs(kv_params, [seq_len], total_num_pages=8)


def _make_cache_batch(
    kv_params: KVCacheParams, prompt_lens: list[int]
) -> KVCacheInputsPerDevice[Buffer, Buffer]:
    """Allocate one request per prompt length and return the runtime inputs."""
    return paged_kv_cache_inputs(kv_params, prompt_lens, total_num_pages=32)


def _build_qkv_value(
    *,
    is_mxfp8: bool,
    a: TensorValue,
    wqkv: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    kv_params: KVCacheParams,
    num_heads: int,
) -> TensorValue:
    """Q projection for either the fused MXFP8 path or the bf16 reference."""
    if not is_mxfp8:
        return fused_qkv_ragged_matmul(
            kv_params,
            input=a,
            input_row_offsets=input_row_offsets,
            wqkv=wqkv,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=num_heads,
        )

    a_q, a_scales = quantize_dynamic_block_scaled(
        a,
        sf_vector_size=32,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )
    w_q, w_scales = quantize_dynamic_block_scaled(
        wqkv,
        sf_vector_size=32,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )
    return _fused_qkv_ragged_matmul_scaled_mxfp8(
        kv_params,
        input=a_q,
        input_row_offsets=input_row_offsets,
        wqkv=w_q,
        kv_collection=kv_collection,
        layer_idx=layer_idx,
        n_heads=num_heads,
        input_scale=a_scales,
        weight_scale=w_scales,
    )


def _run_path(
    *,
    is_mxfp8: bool,
    a_np: np.ndarray,
    wqkv_np: np.ndarray,
    seq_len: int,
    num_heads: int,
    kv_params: KVCacheParams,
    device: Accelerator,
    device_ref: DeviceRef,
    session: InferenceSession,
) -> tuple[np.ndarray, np.ndarray]:
    """Build, run one QKV path; return (Q output, KV cache blocks)."""
    hidden = a_np.shape[1]
    qkv_dim = wqkv_np.shape[0]
    kv_symbolic = kv_params.get_symbolic_inputs().inputs[0]

    with Graph(
        f"qkv_{'mxfp8' if is_mxfp8 else 'bf16'}",
        input_types=[
            TensorType(
                DType.bfloat16, shape=(seq_len, hidden), device=device_ref
            ),
            TensorType(DType.uint32, shape=(2,), device=device_ref),
            TensorType(
                DType.bfloat16, shape=(qkv_dim, hidden), device=device_ref
            ),
            *kv_symbolic.flatten(),
        ],
    ) as graph:
        layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())
        (
            a,
            input_row_offsets,
            wqkv,
            blocks,
            cache_lengths,
            lookup_table,
            max_prompt_length,
            max_cache_length,
            *_rest,
        ) = graph.inputs
        kv_collection = PagedCacheValues(
            blocks.buffer,
            cache_lengths.tensor,
            lookup_table.tensor,
            max_prompt_length.tensor,
            max_cache_length.tensor,
        )
        q_out = _build_qkv_value(
            is_mxfp8=is_mxfp8,
            a=a.tensor,
            wqkv=wqkv.tensor,
            input_row_offsets=input_row_offsets.tensor,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            kv_params=kv_params,
            num_heads=num_heads,
        )
        graph.output(q_out)

    model = session.load(graph)
    kv_runtime = _make_cache(kv_params, seq_len)

    a_buf = Buffer.from_dlpack(torch.from_numpy(a_np).to(torch.bfloat16)).to(
        device
    )
    wqkv_buf = Buffer.from_dlpack(
        torch.from_numpy(wqkv_np).to(torch.bfloat16)
    ).to(device)
    row_offsets_buf = Buffer.from_dlpack(
        torch.tensor([0, seq_len], dtype=torch.uint32)
    ).to(device)

    (out_buf,) = model.execute(
        a_buf, row_offsets_buf, wqkv_buf, *kv_runtime.flatten()
    )
    q_out_np = torch.from_dlpack(out_buf).to(torch.float32).cpu().numpy()
    # The cache is bf16, which numpy can't represent, so read it through torch.
    kv_blocks_np = (
        torch.from_dlpack(kv_runtime.kv_blocks).to(torch.float32).cpu().numpy()
    )
    return q_out_np, kv_blocks_np


# MiniMax-M3-shaped GQA with the head count scaled down to keep the test light.
# K (hidden) must stay a multiple of 128, the rank-5 SF K-group size.
@pytest.mark.parametrize(
    "label,seq_len,num_heads,num_kv_heads,head_dim,hidden",
    [
        ("prefill", 96, 16, 4, 128, 768),
        ("decode", 1, 16, 4, 128, 768),
        # Production M3 dense-layer shape at TP=4 (N=2304, K=6144). The short-K
        # cases above leave CDNA4's split-K dispatch unqualified (3 BK-tiles,
        # below its 2-tiles-per-split floor); K=6144 gives 24 tiles -> 12
        # splits, which is what routes the KV-scatter epilogue through the
        # split-K reduce kernel instead of the matmul's own store path.
        ("decode_splitk", 1, 16, 1, 128, 6144),
    ],
)
def test_fused_qkv_mxfp8_matmul(
    label: str,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden: int,
) -> None:
    _skip_if_no_fused_qkv_mxfp8()

    qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim

    rng = np.random.default_rng(0)
    # Small-magnitude inputs keep values inside the E4M3 dynamic range so block
    # scaling is well-conditioned.
    a_np = (rng.standard_normal((seq_len, hidden)) * 0.1).astype(np.float32)
    wqkv_np = (rng.standard_normal((qkv_dim, hidden)) * 0.1).astype(np.float32)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        page_size=128,
        n_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=1,
        devices=[device_ref],
    )

    q_mxfp8, kv_mxfp8 = _run_path(
        is_mxfp8=True,
        a_np=a_np,
        wqkv_np=wqkv_np,
        seq_len=seq_len,
        num_heads=num_heads,
        kv_params=kv_params,
        device=device,
        device_ref=device_ref,
        session=session,
    )
    _q_ref, kv_ref = _run_path(
        is_mxfp8=False,
        a_np=a_np,
        wqkv_np=wqkv_np,
        seq_len=seq_len,
        num_heads=num_heads,
        kv_params=kv_params,
        device=device,
        device_ref=device_ref,
        session=session,
    )

    q_dim = num_heads * head_dim
    q_host_ref = (a_np @ wqkv_np.T)[:, :q_dim]
    q_cos, q_rel = _cosine_and_rel_l2(q_mxfp8, q_host_ref)
    # Unwritten cache slots are zero in both buffers, so they do not distort
    # the cosine.
    kv_cos, kv_rel = _cosine_and_rel_l2(kv_mxfp8, kv_ref)

    print(
        f"\n=== fused_qkv_mxfp8 {label} "
        f"(S={seq_len}, H={num_heads}, KV={num_kv_heads}, D={head_dim}, "
        f"K={hidden}) ===\n"
        f"  Q   cosine / rel-L2 : {q_cos:.5f} / {q_rel:.5f}\n"
        f"  K/V cosine / rel-L2 : {kv_cos:.5f} / {kv_rel:.5f}",
        flush=True,
    )

    assert q_mxfp8.shape == (seq_len, q_dim)
    assert q_cos > 0.99, f"{label}: Q cosine {q_cos:.5f} too low"
    assert q_rel < 0.1, f"{label}: Q rel-L2 {q_rel:.5f} too high"
    assert np.any(kv_mxfp8 != 0.0), f"{label}: MXFP8 KV cache is all zeros"
    assert kv_cos > 0.99, f"{label}: K/V cosine {kv_cos:.5f} too low"
    assert kv_rel < 0.1, f"{label}: K/V rel-L2 {kv_rel:.5f} too high"

    # Dense-layer FP8 KV-cache store (the write op a block-sparse MSA FP8 KV
    # uses for its dense layers): the same MXFP8 projection stored into a
    # scale-free FP8 cache must match the BF16-cache store within FP8 rounding.
    # Checked on the prefill shape only, reusing the MXFP8 K/V computed above as
    # the BF16 baseline so no extra reference graph is compiled.
    if label == "prefill":
        fp8_params = MHAKVCacheParams(
            dtype=DType.float8_e4m3fn,
            page_size=128,
            n_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_layers=1,
            devices=[device_ref],
        )
        assert fp8_params.is_fp8_kv_dtype
        assert not fp8_params.quantized_kv_cache
        _q_fp8, kv_fp8 = _run_path(
            is_mxfp8=True,
            a_np=a_np,
            wqkv_np=wqkv_np,
            seq_len=seq_len,
            num_heads=num_heads,
            kv_params=fp8_params,
            device=device,
            device_ref=device_ref,
            session=session,
        )
        fp8_cos, _ = _cosine_and_rel_l2(kv_fp8, kv_mxfp8)
        assert np.all(np.isfinite(kv_fp8)), "FP8 KV cache has non-finite values"
        assert np.any(kv_fp8 != 0.0), "FP8 KV cache is all zeros"
        assert fp8_cos > 0.99, f"FP8 vs BF16 K/V cosine {fp8_cos:.5f} too low"


def test_fused_qkv_index_mxfp8_matmul_fp8_main_cache() -> None:
    """The 5-way sparse (QKV + index-QK) store into an FP8 main cache.

    Exercises the dual-cache fused op the sparse block-sparse layers use. The
    GEMM outputs the BF16 Q/IndexQ scratch and the epilogue saturating-casts
    K/V into the main cache (index cache stays BF16). Runs the op
    with a BF16 and an FP8 main cache and asserts the FP8 K/V match the BF16
    reference within FP8 tolerance, while the Q/IndexQ output (unaffected by the
    cache dtype) is bit-identical.
    """
    _skip_if_not_supported()

    seq_len = 96
    n_heads, n_kv_heads, head_dim = 16, 4, 128
    # num_index_heads must be an MLA-dispatch-supported value (8/16/32/64/128);
    # this store-only test doesn't run MLA decode, but the index cache's
    # runtime inputs bind dispatch metadata eagerly.
    num_index_heads, idx_head_dim = 16, 128
    hidden = 768
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    iq_dim = num_index_heads * idx_head_dim
    ik_dim = idx_head_dim
    n_total = q_dim + 2 * kv_dim + iq_dim + ik_dim

    rng = np.random.default_rng(0)
    a_np = (rng.standard_normal((seq_len, hidden)) * 0.1).astype(np.float32)
    wqkv_np = (rng.standard_normal((n_total, hidden)) * 0.1).astype(np.float32)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])

    def _run(
        main_dtype: DType,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the 5-way op with a main cache of ``main_dtype`` (index BF16).

        Returns ``(Q, IndexQ, main-cache K/V blocks)`` as fp32.
        """
        main_params = MHAKVCacheParams(
            dtype=main_dtype,
            page_size=128,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_layers=1,
            devices=[device_ref],
        )
        index_params = MLAKVCacheParams(
            dtype=DType.bfloat16,
            page_size=128,
            head_dim=idx_head_dim,
            num_layers=1,
            devices=[device_ref],
            num_q_heads=num_index_heads,
        )

        main_sym = main_params.get_symbolic_inputs().inputs[0]
        index_sym = index_params.get_symbolic_inputs().inputs[0]
        n_main = len(main_sym.flatten())

        with Graph(
            f"qkv_index_mxfp8_{main_dtype}_main_cache",
            input_types=[
                TensorType(
                    DType.bfloat16, (seq_len, hidden), device=device_ref
                ),
                TensorType(DType.uint32, (2,), device=device_ref),
                TensorType(
                    DType.bfloat16, (n_total, hidden), device=device_ref
                ),
                *main_sym.flatten(),
                *index_sym.flatten(),
            ],
        ) as graph:
            a, iro, wqkv, *rest = graph.inputs
            main_in, index_in = rest[:n_main], rest[n_main:]
            layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())
            a_q, a_scales = quantize_dynamic_block_scaled(
                a.tensor,
                sf_vector_size=32,
                scales_type=DType.float8_e8m0fnu,
                out_type=DType.float8_e4m3fn,
            )
            w_q, w_scales = quantize_dynamic_block_scaled(
                wqkv.tensor,
                sf_vector_size=32,
                scales_type=DType.float8_e8m0fnu,
                out_type=DType.float8_e4m3fn,
            )
            main_kv = PagedCacheValues(
                main_in[0].buffer,
                main_in[1].tensor,
                main_in[2].tensor,
                main_in[3].tensor,
                main_in[4].tensor,
            )
            index_kv = PagedCacheValues(
                index_in[0].buffer,
                index_in[1].tensor,
                index_in[2].tensor,
                index_in[3].tensor,
                index_in[4].tensor,
            )
            q, index_q = _fused_qkv_index_ragged_matmul_scaled_mxfp8(
                main_params,
                index_params,
                input=a_q,
                input_row_offsets=iro.tensor,
                wqkv=w_q,
                kv_collection=main_kv,
                index_kv_collection=index_kv,
                layer_idx=layer_idx,
                n_heads=n_heads,
                num_index_heads=num_index_heads,
                idx_head_dim=idx_head_dim,
                input_scale=a_scales,
                weight_scale=w_scales,
            )
            graph.output(q, index_q)

        model = session.load(graph)
        main_rt = _make_cache(main_params, seq_len)
        index_rt = _make_cache(index_params, seq_len)

        a_buf = Buffer.from_dlpack(
            torch.from_numpy(a_np).to(torch.bfloat16)
        ).to(device)
        wqkv_buf = Buffer.from_dlpack(
            torch.from_numpy(wqkv_np).to(torch.bfloat16)
        ).to(device)
        iro_buf = Buffer.from_dlpack(
            torch.tensor([0, seq_len], dtype=torch.uint32)
        ).to(device)

        q_buf, iq_buf = model.execute(
            a_buf, iro_buf, wqkv_buf, *main_rt.flatten(), *index_rt.flatten()
        )
        q_np = torch.from_dlpack(q_buf).to(torch.float32).cpu().numpy()
        iq_np = torch.from_dlpack(iq_buf).to(torch.float32).cpu().numpy()
        main_kv_np = (
            torch.from_dlpack(main_rt.kv_blocks).to(torch.float32).cpu().numpy()
        )
        return q_np, iq_np, main_kv_np

    q_fp8, iq_fp8, kv_fp8 = _run(DType.float8_e4m3fn)
    q_bf16, iq_bf16, kv_bf16 = _run(DType.bfloat16)

    kv_cos, kv_rel = _cosine_and_rel_l2(kv_fp8, kv_bf16)
    print(
        f"\n=== fused_qkv_index_mxfp8 fp8-main-cache store ===\n"
        f"  K/V cosine / rel-L2 : {kv_cos:.5f} / {kv_rel:.5f}",
        flush=True,
    )

    assert np.all(np.isfinite(kv_fp8)), "FP8 main cache has non-finite values"
    assert np.any(kv_fp8 != 0.0), "FP8 main cache was not written"
    # Q and IndexQ come from the BF16 scratch, unaffected by the cache dtype.
    np.testing.assert_array_equal(q_fp8, q_bf16)
    np.testing.assert_array_equal(iq_fp8, iq_bf16)
    # K/V differ only by the FP8 store rounding.
    assert kv_cos > 0.99, f"FP8 vs BF16 K/V cosine {kv_cos:.5f} too low"


@pytest.mark.parametrize(
    "prompt_lens",
    [[96], [1], [1, 1, 1, 1], [3, 17], [200]],
    ids=["single", "decode", "decode_batch", "ragged", "page_crossing"],
)
def test_fused_qkv_index_mxfp8_matmul_amd_stacked(
    prompt_lens: list[int],
) -> None:
    """CDNA4's stacked 5-way QKV+index GEMM matches five separate projections.

    The CDNA4 block-scaled matmul has no epilogue hook, so
    ``quantized_fused_qkv_index_matmul`` runs one GEMM over the stacked
    ``[Wq | Wk | Wv | Wiq | Wik]`` weight and places K/V/IndexK with standalone
    paged stores. Both paths quantize the same activation the same way, so the
    stacked GEMM has to agree with five band-sliced GEMMs on Q, IndexQ and both
    caches -- which is what pins each column band to the right destination.

    Not asserted bit-exact: ``N`` drives the AMD tile and split-K dispatch, so
    the stacked GEMM reduces K in a different order than the narrow per-band
    GEMMs do.
    """
    _skip_if_not_amd()

    seq_len = sum(prompt_lens)
    n_heads, n_kv_heads, head_dim = 16, 4, 128
    # As in the SM100 test above: an MLA-dispatch-supported index head count,
    # since the index cache binds dispatch metadata when its inputs are built.
    num_index_heads, idx_head_dim = 16, 128
    hidden = 768
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    iq_dim = num_index_heads * idx_head_dim
    ik_dim = idx_head_dim
    n_total = q_dim + 2 * kv_dim + iq_dim + ik_dim

    rng = np.random.default_rng(0)
    a_np = (rng.standard_normal((seq_len, hidden)) * 0.1).astype(np.float32)
    wqkv_np = (rng.standard_normal((n_total, hidden)) * 0.1).astype(np.float32)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])

    main_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        page_size=128,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=1,
        devices=[device_ref],
    )
    index_params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        page_size=128,
        head_dim=idx_head_dim,
        num_layers=1,
        devices=[device_ref],
        num_q_heads=num_index_heads,
    )

    def _run(
        stacked: bool,
        pre: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run one path; returns (Q, IndexQ, main blocks, index blocks) as fp32.

        `pre` is the `(fp8, e8m0)` pair a producer epilogue would hand the op:
        "same" is what the op would compute itself, "foreign" unrelated rows.
        """
        main_sym = main_params.get_symbolic_inputs().inputs[0]
        index_sym = index_params.get_symbolic_inputs().inputs[0]
        n_main = len(main_sym.flatten())

        name = "stacked" if stacked else "separate"
        if pre is not None:
            name += f"_pre_{pre}"
        with Graph(
            f"qkv_index_mxfp8_amd_{name}",
            input_types=[
                TensorType(
                    DType.bfloat16, (seq_len, hidden), device=device_ref
                ),
                TensorType(
                    DType.uint32,
                    (len(prompt_lens) + 1,),
                    device=device_ref,
                ),
                TensorType(
                    DType.bfloat16, (n_total, hidden), device=device_ref
                ),
                *main_sym.flatten(),
                *index_sym.flatten(),
            ],
        ) as graph:
            a, iro, wqkv, *rest = graph.inputs
            main_in, index_in = rest[:n_main], rest[n_main:]
            layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())
            main_kv = PagedCacheValues(
                main_in[0].buffer,
                main_in[1].tensor,
                main_in[2].tensor,
                main_in[3].tensor,
                main_in[4].tensor,
            )
            index_kv = PagedCacheValues(
                index_in[0].buffer,
                index_in[1].tensor,
                index_in[2].tensor,
                index_in[3].tensor,
                index_in[4].tensor,
            )
            # On AMD this returns rank-2 [N, K // 32] E8M0 scales -- the
            # checkpoint layout, which the CDNA4 matmul consumes uninterleaved.
            w_q, w_scales = quantize_dynamic_block_scaled(
                wqkv.tensor,
                sf_vector_size=32,
                scales_type=DType.float8_e8m0fnu,
                out_type=DType.float8_e4m3fn,
            )
            if stacked:
                pre_pair = None
                if pre is not None:
                    # "foreign" borrows the first `seq_len` weight rows: right
                    # shape and dtype, unrelated values.
                    pre_src = (
                        a.tensor if pre == "same" else wqkv.tensor[:seq_len]
                    )
                    pre_pair = quantize_dynamic_block_scaled(
                        pre_src,
                        sf_vector_size=32,
                        scales_type=DType.float8_e8m0fnu,
                        out_type=DType.float8_e4m3fn,
                    )
                q, index_q = quantized_fused_qkv_index_matmul(
                    kv_params=main_params,
                    index_kv_params=index_params,
                    x=a.tensor,
                    wqkv=w_q,
                    kv_collection=main_kv,
                    index_kv_collection=index_kv,
                    layer_idx=layer_idx,
                    input_row_offsets=iro.tensor,
                    n_heads=n_heads,
                    num_index_heads=num_index_heads,
                    idx_head_dim=idx_head_dim,
                    quant_config=_mxfp8_quant_config(),
                    weight_scale=w_scales,
                    prequantized=pre_pair,
                )
            else:
                a_q, a_scales = quantize_dynamic_block_scaled(
                    a.tensor,
                    sf_vector_size=32,
                    scales_type=DType.float8_e8m0fnu,
                    out_type=DType.float8_e4m3fn,
                )
                projections = []
                start = 0
                for width in (q_dim, kv_dim, kv_dim, iq_dim, ik_dim):
                    stop = start + width
                    projections.append(
                        dynamic_block_scaled_matmul_amd(
                            a_q,
                            w_q[start:stop],
                            a_scales,
                            w_scales[start:stop],
                            out_type=DType.bfloat16,
                        )
                    )
                    start = stop
                q, k, v, index_q, index_k = projections
                store_k_cache_ragged(
                    main_kv,
                    ops.reshape(k, [seq_len, n_kv_heads, head_dim]),
                    iro.tensor,
                    layer_idx,
                )
                store_v_cache_ragged(
                    main_kv,
                    ops.reshape(v, [seq_len, n_kv_heads, head_dim]),
                    iro.tensor,
                    layer_idx,
                )
                store_k_cache_ragged(
                    index_kv,
                    ops.reshape(index_k, [seq_len, 1, idx_head_dim]),
                    iro.tensor,
                    layer_idx,
                )
            graph.output(q, index_q)

        model = session.load(graph)
        main_rt = _make_cache_batch(main_params, prompt_lens)
        index_rt = _make_cache_batch(index_params, prompt_lens)

        a_buf = Buffer.from_dlpack(
            torch.from_numpy(a_np).to(torch.bfloat16)
        ).to(device)
        wqkv_buf = Buffer.from_dlpack(
            torch.from_numpy(wqkv_np).to(torch.bfloat16)
        ).to(device)
        iro_buf = Buffer.from_dlpack(
            torch.tensor(np.cumsum([0] + prompt_lens), dtype=torch.uint32)
        ).to(device)

        q_buf, iq_buf = model.execute(
            a_buf, iro_buf, wqkv_buf, *main_rt.flatten(), *index_rt.flatten()
        )

        # The caches are bf16, which numpy can't represent, so read through torch.
        def to_np(buf: Buffer) -> np.ndarray:
            return torch.from_dlpack(buf).to(torch.float32).cpu().numpy()

        return (
            to_np(q_buf),
            to_np(iq_buf),
            to_np(main_rt.kv_blocks),
            to_np(index_rt.kv_blocks),
        )

    q_st, iq_st, main_st, index_st = _run(stacked=True)
    q_sep, iq_sep, main_sep, index_sep = _run(stacked=False)

    q_cos, q_rel = _cosine_and_rel_l2(q_st, q_sep)
    iq_cos, iq_rel = _cosine_and_rel_l2(iq_st, iq_sep)
    main_cos, main_rel = _cosine_and_rel_l2(main_st, main_sep)
    index_cos, index_rel = _cosine_and_rel_l2(index_st, index_sep)
    print(
        f"\n=== fused_qkv_index_mxfp8 CDNA4 stacked vs separate ===\n"
        f"  Q       cosine / rel-L2 : {q_cos:.6f} / {q_rel:.6f}\n"
        f"  IndexQ  cosine / rel-L2 : {iq_cos:.6f} / {iq_rel:.6f}\n"
        f"  main KV cosine / rel-L2 : {main_cos:.6f} / {main_rel:.6f}\n"
        f"  indexK  cosine / rel-L2 : {index_cos:.6f} / {index_rel:.6f}",
        flush=True,
    )

    # A mis-routed band writes another projection's values, which collapses the
    # cosine long before it reaches these thresholds.
    assert q_cos > 0.9999, f"Q cosine {q_cos:.6f} too low"
    assert iq_cos > 0.9999, f"IndexQ cosine {iq_cos:.6f} too low"
    assert main_cos > 0.9999, f"main K/V cosine {main_cos:.6f} too low"
    assert index_cos > 0.9999, f"IndexK cosine {index_cos:.6f} too low"
    assert np.any(main_st != 0.0), "stacked path left the main cache unwritten"
    assert np.any(index_st != 0.0), (
        "stacked path left the index cache unwritten"
    )

    # Two arms: byte-equality against the op's own quantize is the layout gate,
    # but an op ignoring the pair and requantizing `x` would pass it too -- so
    # arm 2 feeds unrelated rows and requires the outputs to MOVE.
    q_same, iq_same, main_same, index_same = _run(stacked=True, pre="same")
    np.testing.assert_array_equal(q_same, q_st)
    np.testing.assert_array_equal(iq_same, iq_st)
    np.testing.assert_array_equal(main_same, main_st)
    np.testing.assert_array_equal(index_same, index_st)

    q_fgn, iq_fgn, main_fgn, index_fgn = _run(stacked=True, pre="foreign")
    for name, got, ref in (
        ("Q", q_fgn, q_st),
        ("IndexQ", iq_fgn, iq_st),
        ("main KV", main_fgn, main_st),
        ("IndexK", index_fgn, index_st),
    ):
        assert not np.array_equal(got, ref), (
            f"{name} unchanged when `prequantized` carried unrelated rows: the"
            " op is quantizing `x` itself and ignoring the pair"
        )
