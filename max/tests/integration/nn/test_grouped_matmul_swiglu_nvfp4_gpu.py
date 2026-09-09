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

"""Byte-exact equivalence between the fused SwiGLU+NVFP4 grouped matmul
kernel (`grouped_matmul_swiglu_nvfp4`) and the chained reference
(`grouped_matmul_block_scaled` -> `fused_silu_quantized`) at the MAX graph
level.

The kernel guarantees byte-identical output to the chained reference under
its default ``match_bf16=True`` setting when the caller pre-permutes ``W``
and ``b_scales`` on the N axis with ``sigma(2i)=i, sigma(2i+1)=D+i`` (where
``D = moe_dim``, ``N = 2D``). This test exercises that contract end-to-end
through MAX Python.

PATH A (chained, concat-layout W):
    bf16 = grouped_matmul_block_scaled(hidden, W_A, a_scales, b_scales_A, ...)
    packed_A, sf_A = fused_silu_quantized(bf16, ..., input_scales=s)
PATH B (fused, sigma-permuted W):
    packed_B, sf_B = grouped_matmul_swiglu_nvfp4(
        hidden, W_B, a_scales, b_scales_B, ..., c_input_scales=1/s,
    )

Assert ``np.array_equal(packed_A, packed_B)`` and
``np.array_equal(sf_A, sf_B)``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.comm.ep.ep_kernels import fused_silu_quantized
from max.nn.kernels import (
    block_scales_interleave,
    grouped_matmul_block_scaled,
    grouped_matmul_blocked_swiglu,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from torch.utils.dlpack import from_dlpack


def _make_nvfp4_quant_config() -> QuantConfig:
    """Builds an NVFP4 QuantConfig matching production MoE settings."""
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, 16),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=(1, 16),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.NVFP4,
    )


def _random_uint8(
    shape: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _random_e4m3fn_safe(
    shape: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """Random ``float8_e4m3fn`` bytes with the NaN bit pattern masked out.

    ``float8_e4m3fn`` has a single NaN encoding (``S 1111 111`` -> bytes
    ``0x7F`` and ``0xFF``). Replace those with ``+0`` so intermediate
    computations stay finite and byte-comparison stays well-defined.
    """
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
    arr[(arr & 0x7F) == 0x7F] = 0
    return arr


def _sigma_permute_n(x: np.ndarray, d: int) -> np.ndarray:
    """Apply sigma(2i)=i, sigma(2i+1)=D+i on the N axis (axis 1)."""
    assert x.shape[1] == 2 * d
    out = np.empty_like(x)
    out[:, 0::2] = x[:, :d]
    out[:, 1::2] = x[:, d:]
    return out


def _to_buffer(arr: np.ndarray, device: Accelerator, dtype: DType) -> Buffer:
    """Copy a numpy array to a device buffer with the requested MAX dtype."""
    return (
        Buffer.from_dlpack(torch.from_numpy(arr.copy())).view(dtype).to(device)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Fused SwiGLU+NVFP4 grouped matmul kernel is SM100-only.",
)
def test_grouped_matmul_swiglu_nvfp4_equiv() -> None:
    """Verify fused path matches chained path byte-exactly."""
    rng = np.random.default_rng(1234)

    # Problem shape. M is a multiple of 128 (SF_MN_GROUP_SIZE); each expert
    # owns M/E tokens. K is a multiple of 64 (SF_K_GROUP_SIZE).
    E = 2
    M = 128
    D = 128
    K = 256
    num_active = E
    K_groups = K // 16  # NVFP4_SF_VECTOR_SIZE = 16
    sf_dim_0 = M // 128 + num_active  # per-expert tail-pad

    # Random NVFP4 packed weights and activations (any uint8 byte is a
    # valid NVFP4 pair; OOB bits = arbitrary NVFP4 values).
    hidden = _random_uint8((M, K // 2), rng)
    gate_packed = _random_uint8((E, D, K // 2), rng)
    up_packed = _random_uint8((E, D, K // 2), rng)

    # Concat-layout (path A): per-expert [gate_slab; up_slab].
    w_a = np.concatenate([gate_packed, up_packed], axis=1)
    # Sigma-permuted (path B): per-expert row-interleaved [g0,u0,g1,u1,...].
    w_b = _sigma_permute_n(w_a, D)

    # Pre-interleave (rank-3) b_scales per expert: [E, 2D, K_groups].
    # block_scales_interleave inside the graph lifts each per-expert slab
    # to the rank-5 tcgen05 layout the kernel expects.
    gate_b_scales = _random_e4m3fn_safe((E, D, K_groups), rng)
    up_b_scales = _random_e4m3fn_safe((E, D, K_groups), rng)
    b_scales_a_pre = np.concatenate([gate_b_scales, up_b_scales], axis=1)
    b_scales_b_pre = _sigma_permute_n(b_scales_a_pre, D)

    # a_scales already in rank-5 tcgen05 layout (no interleave needed).
    a_scales = _random_e4m3fn_safe((sf_dim_0, K_groups // 4, 32, 4, 4), rng)

    expert_start = np.array([0, M // 2, M], dtype=np.uint32)
    a_scale_offsets = np.array([0, 1], dtype=np.uint32)
    expert_ids = np.array([0, 1], dtype=np.int32)
    expert_scales = np.array([1.0, 1.0], dtype=np.float32)
    usage_stats = np.array([M // 2, num_active], dtype=np.uint32)
    raw_input_scales = np.array([0.5, 0.5], dtype=np.float32)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])

    input_types: list[TensorType] = [
        TensorType(DType.uint8, (M, K // 2), device=device_ref),  # hidden
        TensorType(DType.uint8, (E, 2 * D, K // 2), device=device_ref),  # w_a
        TensorType(DType.uint8, (E, 2 * D, K // 2), device=device_ref),  # w_b
        TensorType(
            DType.float8_e4m3fn,
            (sf_dim_0, K_groups // 4, 32, 4, 4),
            device=device_ref,
        ),  # a_scales
        TensorType(
            DType.float8_e4m3fn, (E, 2 * D, K_groups), device=device_ref
        ),  # b_scales_a_pre
        TensorType(
            DType.float8_e4m3fn, (E, 2 * D, K_groups), device=device_ref
        ),  # b_scales_b_pre
        TensorType(DType.uint32, (3,), device=device_ref),  # expert_start
        TensorType(DType.uint32, (E,), device=device_ref),  # a_scale_offsets
        TensorType(DType.int32, (E,), device=device_ref),  # expert_ids
        TensorType(DType.float32, (E,), device=device_ref),  # expert_scales
        TensorType(DType.uint32, (2,), device=cpu_ref),  # usage_stats
        TensorType(
            DType.float32, (num_active,), device=device_ref
        ),  # raw_input_scales
    ]

    quant_config = _make_nvfp4_quant_config()

    with Graph("swiglu_nvfp4_equiv", input_types=input_types) as graph:
        (
            hidden_t,
            w_a_t,
            w_b_t,
            a_scales_t,
            b_scales_a_pre_t,
            b_scales_b_pre_t,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            expert_scales_t,
            usage_stats_t,
            raw_input_scales_t,
        ) = (inp.tensor for inp in graph.inputs)

        # Lift pre-interleave [E, 2D, K_groups] to rank-6 tcgen05 layout via
        # per-expert block_scales_interleave (mirrors
        # ``_interleave_nvfp4_scales`` in quant_strategy.py).
        b_scales_a = ops.stack(
            [
                block_scales_interleave(s.reshape([2 * D, K_groups]))
                for s in ops.split(b_scales_a_pre_t, [1] * E, axis=0)
            ],
            axis=0,
        )
        b_scales_b = ops.stack(
            [
                block_scales_interleave(s.reshape([2 * D, K_groups]))
                for s in ops.split(b_scales_b_pre_t, [1] * E, axis=0)
            ],
            axis=0,
        )

        # PATH A: chained two-kernel reference on concat-layout weights.
        bf16_out = grouped_matmul_block_scaled(
            hidden_t,
            w_a_t,
            a_scales_t,
            b_scales_a,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            expert_scales_t,
            usage_stats_t,
        )
        packed_a, sf_a = fused_silu_quantized(
            bf16_out,
            expert_start_t,
            quant_config,
            DType.uint8,
            input_scales=raw_input_scales_t,
            scales_offsets=a_scale_offsets_t,
        )

        # PATH B: fused kernel on sigma-permuted weights. The fused kernel
        # consumes ``1/raw_input_scales`` directly (the chained
        # ``fused_silu_quantized`` does this inversion internally).
        inv_input_scales = (
            ops.constant(1.0, DType.float32, device=device_ref)
            / raw_input_scales_t
        )
        packed_b, sf_b = grouped_matmul_blocked_swiglu(
            hidden_t,
            w_b_t,
            a_scales_t,
            b_scales_b,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            usage_stats_t,
            expert_scales=expert_scales_t,
            c_input_scales=inv_input_scales,
        )

        graph.output(packed_a, sf_a, packed_b, sf_b)

    compiled = session.load(graph)

    # Copy all inputs to device. Use uint8 view for float8_e4m3fn so dlpack
    # can transit the raw bytes without dtype handshake hassles.
    def _buf(
        arr: np.ndarray, dtype: DType, dev: Accelerator | None = None
    ) -> Buffer:
        target = dev if dev is not None else device
        view = Buffer.from_dlpack(torch.from_numpy(arr.copy()))
        if dtype != DType.uint8 and arr.dtype == np.uint8:
            view = view.view(dtype)
        return view.to(target)

    # CPU buffer for usage_stats (matches cpu_ref TensorType).
    usage_stats_cpu = Buffer.from_dlpack(torch.from_numpy(usage_stats.copy()))

    result = compiled.execute(
        _buf(hidden, DType.uint8),
        _buf(w_a, DType.uint8),
        _buf(w_b, DType.uint8),
        _buf(a_scales, DType.float8_e4m3fn),
        _buf(b_scales_a_pre, DType.float8_e4m3fn),
        _buf(b_scales_b_pre, DType.float8_e4m3fn),
        _buf(expert_start, DType.uint32),
        _buf(a_scale_offsets, DType.uint32),
        _buf(expert_ids, DType.int32),
        _buf(expert_scales, DType.float32),
        usage_stats_cpu,
        _buf(raw_input_scales, DType.float32),
    )

    packed_a_np = from_dlpack(result[0]).cpu().numpy()
    sf_a_np = from_dlpack(result[1]).cpu().view(torch.uint8).numpy()
    packed_b_np = from_dlpack(result[2]).cpu().numpy()
    sf_b_np = from_dlpack(result[3]).cpu().view(torch.uint8).numpy()

    assert packed_a_np.shape == packed_b_np.shape, (
        f"packed shape mismatch: A={packed_a_np.shape} vs B={packed_b_np.shape}"
    )
    assert sf_a_np.shape == sf_b_np.shape, (
        f"sf shape mismatch: A={sf_a_np.shape} vs B={sf_b_np.shape}"
    )
    assert np.array_equal(packed_a_np, packed_b_np), (
        "Fused NVFP4 packed output differs from chained reference; the "
        "sigma-permutation contract is broken (see kernel docstring)."
    )
    assert np.array_equal(sf_a_np, sf_b_np), (
        "Fused SF tile differs from chained reference; the sigma-permutation "
        "contract is broken (see kernel docstring)."
    )
