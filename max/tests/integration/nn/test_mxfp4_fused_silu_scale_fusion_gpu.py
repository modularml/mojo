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

"""Graph-level A/B tests for the MXFP4 EP A-scale slot-fusion matmul reader.

Each runs a grouped MXFP4 matmul two ways on a single GPU and asserts the
outputs are byte-identical:

- ``test_fused_silu_down_matmul_ab``: down-proj ``fused_silu_quantized`` ->
  ``grouped_dynamic_block_scaled_matmul_amd`` (KS64).
- ``test_up_proj_dispatch_matmul_ab``: KS224 up-proj reader fed by the slot
  scales ``ep_wait`` writes.

With ``padded_m`` inflated past the runtime per-expert stride and >1 expert,
the A/B diverges if the matmul reads A-scale slots at the runtime stride
instead of the graph-const ``a_scales_max_padded_m`` — the producer/consumer
single source of truth.

NOTE: AMD / CDNA-only (preb kernel targets gfx9 CDNA4 / MI355).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.comm.ep.ep_kernels import fused_silu_quantized
from max.nn.kernels import (
    block_scaled_preshuffle_b_5d,
    block_scaled_preshuffle_grouped_scale_4d,
    grouped_dynamic_block_scaled_matmul_amd,
    quantize_dynamic_block_scaled_mxfp4,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)


def _make_mxfp4_quant_config() -> QuantConfig:
    """MXFP4 QuantConfig matching EP down-proj production settings."""
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 32),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, 32),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.MXFP4,
    )


@pytest.mark.skipif(
    accelerator_api() != "hip",
    reason="MXFP4 preb fused-silu scale fusion is AMD CDNA-only",
)
@pytest.mark.parametrize(
    "tokens_per_expert, intermediate, n_out, padded_m",
    [
        # `padded_m` (graph-const slot stride the fused path writes with) is
        # deliberately > align_up(max tokens/expert, 32) with >1 expert, so a
        # matmul that read slots at the *runtime* stride instead of `padded_m`
        # would read the wrong slot for experts >= 1 and the A/B would diverge.
        # intermediate must be a multiple of 512 (preb path requires packed
        # K = K//2 >= 256 and divisible by 256).
        ([10, 8], 512, 128, 64),
        ([5, 0, 12], 512, 256, 96),
    ],
)
def test_fused_silu_down_matmul_ab(
    tokens_per_expert: list[int],
    intermediate: int,
    n_out: int,
    padded_m: int,
) -> None:
    """Down-proj stride-consistency guard (the real matmul A/B).

    Runs the down-proj ``fused_silu -> grouped_matmul`` two ways and asserts the
    matmul outputs match:
      REF:   raw scales + standalone preshuffle (runtime slot stride).
      FUSED: fused_silu writes slots directly (graph-const ``padded_m`` stride);
             matmul told ``a_scales_preshuffled=True, a_scales_max_padded_m``.
    Same SILU input -> identical scale *values* + identical down_in tokens, so
    the only difference is the A-scale slot layout/stride; outputs must be
    identical. With ``padded_m`` > runtime stride and >1 expert, this fails if
    the matmul ignores ``a_scales_max_padded_m`` and uses the runtime stride.
    """
    n_experts = len(tokens_per_expert)
    assert intermediate % 512 == 0 and n_out % 16 == 0
    K = intermediate
    K_bytes = K // 2
    K_scales = K // 32
    total_tokens = sum(tokens_per_expert)
    max_tok = max(tokens_per_expert)
    aligned_runtime = (max_tok + 31) // 32 * 32
    assert padded_m % 32 == 0 and padded_m >= max_tok
    assert padded_m > aligned_runtime, "test must inflate padded_m past runtime"

    rng = np.random.default_rng(7)
    gate_up_torch = torch.from_numpy(
        rng.standard_normal((total_tokens, K * 2)).astype(np.float32)
    ).to(torch.bfloat16)

    row_offsets_np = np.zeros(n_experts + 1, dtype=np.uint32)
    for i, n in enumerate(tokens_per_expert):
        row_offsets_np[i + 1] = row_offsets_np[i] + n
    row_offsets_torch = torch.from_numpy(row_offsets_np)

    b_torch = torch.from_numpy(
        rng.integers(0, 256, size=(n_experts, n_out, K_bytes), dtype=np.uint8)
    )
    # Uniform b_scales = E8M0 byte 127 (== 2^0 == 1.0): all-equal bytes are
    # layout-invariant (no B-scale preshuffle needed) and finite (no NaN), so
    # the A/B difference is purely the A-scale slot read.
    b_scales_torch = torch.full(
        (n_experts, n_out, K_scales), 127, dtype=torch.uint8
    )
    expert_ids_torch = torch.arange(n_experts, dtype=torch.int32)
    usage_torch = torch.from_numpy(
        np.array([max_tok, n_experts], dtype=np.uint32)
    )

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])
    quant_config = _make_mxfp4_quant_config()

    input_types = [
        TensorType(DType.bfloat16, (total_tokens, K * 2), device=device_ref),
        TensorType(DType.uint32, (n_experts + 1,), device=device_ref),
        TensorType(DType.uint8, (n_experts, n_out, K_bytes), device=device_ref),
        TensorType(
            DType.float8_e8m0fnu,
            (n_experts, n_out, K_scales),
            device=device_ref,
        ),
        TensorType(DType.int32, (n_experts,), device=device_ref),
        TensorType(DType.uint32, (2,), device=cpu_ref),
    ]

    with Graph("fused_silu_down_matmul_ab", input_types=input_types) as graph:
        gate_up_t = graph.inputs[0].tensor
        row_offsets_t = graph.inputs[1].tensor
        b_pre = block_scaled_preshuffle_b_5d(graph.inputs[2].tensor)
        b_scales_t = graph.inputs[3].tensor
        expert_ids_t = graph.inputs[4].tensor
        usage_t = graph.inputs[5].tensor

        # Same SILU input both ways: identical down_in + identical scale values.
        down_in, raw_scales = fused_silu_quantized(
            gate_up_t, row_offsets_t, quant_config, DType.uint8, max_padded_M=0
        )
        down_in2, slot_scales = fused_silu_quantized(
            gate_up_t,
            row_offsets_t,
            quant_config,
            DType.uint8,
            max_padded_M=padded_m,
        )

        # estimated_total_m must be a host (CPU) scalar.
        est_m = ops.constant(total_tokens, dtype=DType.uint32, device=cpu_ref)
        c_ref = grouped_dynamic_block_scaled_matmul_amd(
            down_in,
            b_pre,
            raw_scales,
            b_scales_t,
            row_offsets_t,
            expert_ids_t,
            usage_t,
            estimated_total_m=est_m,
            preshuffled_b=True,
            a_scales_preshuffled=False,
        )
        c_fused = grouped_dynamic_block_scaled_matmul_amd(
            down_in2,
            b_pre,
            slot_scales,
            b_scales_t,
            row_offsets_t,
            expert_ids_t,
            usage_t,
            estimated_total_m=est_m,
            preshuffled_b=True,
            a_scales_preshuffled=True,
            a_scales_max_padded_m=padded_m,
        )
        # Cast to f32: numpy from_dlpack can't read bfloat16.
        graph.output(c_ref.cast(DType.float32), c_fused.cast(DType.float32))

    compiled = session.load(graph)
    cpu = Accelerator.cpu()
    c_ref_buf, c_fused_buf = compiled.execute(
        Buffer.from_dlpack(gate_up_torch).to(device),
        Buffer.from_dlpack(row_offsets_torch).to(device),
        Buffer.from_dlpack(b_torch).to(device),
        Buffer.from_dlpack(b_scales_torch)
        .view(DType.float8_e8m0fnu)
        .to(device),
        Buffer.from_dlpack(expert_ids_torch).to(device),
        Buffer.from_dlpack(usage_torch),
    )
    c_ref_np = np.from_dlpack(c_ref_buf.to(cpu))
    c_fused_np = np.from_dlpack(c_fused_buf.to(cpu))

    np.testing.assert_allclose(
        c_fused_np,
        c_ref_np,
        rtol=0,
        atol=0,
        err_msg=(
            "Down-proj matmul output diverged between the fused (slot-stride="
            f"{padded_m}) and non-fused paths for tokens_per_expert="
            f"{tokens_per_expert} — the matmul likely read A-scale slots at the"
            " runtime stride instead of a_scales_max_padded_m."
        ),
    )


@pytest.mark.skipif(
    accelerator_api() != "hip",
    reason="MXFP4 preb up-proj scale fusion is AMD CDNA-only",
)
@pytest.mark.parametrize(
    "tokens_per_expert, hidden, n_out, padded_m",
    [
        # KS224 up-proj: hidden=7168 -> K_SCALES=224. `padded_m` is deliberately
        # > align_up(max tokens/expert, 32) with >1 expert, so a matmul reading
        # slots at the runtime stride instead of `padded_m` would read the wrong
        # slot for experts >= 1 and the A/B would diverge. n_out (gate+up width)
        # must be a multiple of 16; hidden//2 must be divisible by 256.
        ([10, 8], 7168, 256, 64),
        ([5, 0, 12], 7168, 512, 96),
    ],
)
def test_up_proj_dispatch_matmul_ab(
    tokens_per_expert: list[int],
    hidden: int,
    n_out: int,
    padded_m: int,
) -> None:
    """Stride-consistency guard for the KS224 up-proj reader.

    The up-proj A-scale is produced by ``ep_wait`` (in slot layout when fused).
    This A/B isolates the *matmul reader* contract that ``ep_wait`` feeds, with
    no EP comm machinery (single GPU):
      REF:   raw dispatched scales + matmul ``a_scales_preshuffled=False``
             (internal standalone preshuffle, runtime slot stride).
      FUSED: raw scales -> standalone preshuffle with the graph-const
             ``padded_m`` stride (== what ``ep_wait`` writes) -> matmul told
             ``a_scales_preshuffled=True, a_scales_max_padded_m=padded_m``.
    Same quantized tokens + identical scale *values* both ways, so the only
    difference is the A-scale slot layout/stride; outputs must be identical.
    With ``padded_m`` > runtime stride and >1 expert, this fails if the matmul
    ignores ``a_scales_max_padded_m`` and uses the runtime stride
    (producer/consumer single source of truth, mirrored for the up-proj K=7168).
    """
    n_experts = len(tokens_per_expert)
    K = hidden
    assert (K // 2) % 256 == 0 and n_out % 16 == 0
    K_bytes = K // 2
    K_scales = K // 32
    total_tokens = sum(tokens_per_expert)
    max_tok = max(tokens_per_expert)
    aligned_runtime = (max_tok + 31) // 32 * 32
    assert padded_m % 32 == 0 and padded_m >= max_tok
    assert padded_m > aligned_runtime, "test must inflate padded_m past runtime"

    rng = np.random.default_rng(11)
    # Dispatched activation tokens (bf16) — quantized to MXFP4 row-major, as the
    # dispatch producer does before the up-proj matmul.
    act_torch = torch.from_numpy(
        rng.standard_normal((total_tokens, K)).astype(np.float32)
    ).to(torch.bfloat16)

    row_offsets_np = np.zeros(n_experts + 1, dtype=np.uint32)
    for i, n in enumerate(tokens_per_expert):
        row_offsets_np[i + 1] = row_offsets_np[i] + n
    row_offsets_torch = torch.from_numpy(row_offsets_np)

    b_torch = torch.from_numpy(
        rng.integers(0, 256, size=(n_experts, n_out, K_bytes), dtype=np.uint8)
    )
    # Uniform b_scales (E8M0 byte 127 == 1.0): layout-invariant + finite, so the
    # A/B difference is purely the A-scale slot read.
    b_scales_torch = torch.full(
        (n_experts, n_out, K_scales), 127, dtype=torch.uint8
    )
    expert_ids_torch = torch.arange(n_experts, dtype=torch.int32)
    usage_torch = torch.from_numpy(
        np.array([max_tok, n_experts], dtype=np.uint32)
    )

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])

    input_types = [
        TensorType(DType.bfloat16, (total_tokens, K), device=device_ref),
        TensorType(DType.uint32, (n_experts + 1,), device=device_ref),
        TensorType(DType.uint8, (n_experts, n_out, K_bytes), device=device_ref),
        TensorType(
            DType.float8_e8m0fnu,
            (n_experts, n_out, K_scales),
            device=device_ref,
        ),
        TensorType(DType.int32, (n_experts,), device=device_ref),
        TensorType(DType.uint32, (2,), device=cpu_ref),
    ]

    with Graph("up_proj_dispatch_matmul_ab", input_types=input_types) as graph:
        act_t = graph.inputs[0].tensor
        row_offsets_t = graph.inputs[1].tensor
        b_pre = block_scaled_preshuffle_b_5d(graph.inputs[2].tensor)
        b_scales_t = graph.inputs[3].tensor
        expert_ids_t = graph.inputs[4].tensor
        usage_t = graph.inputs[5].tensor

        # One quantization → identical tokens + identical raw scale values.
        a_tokens, raw_scales = quantize_dynamic_block_scaled_mxfp4(act_t)

        # FUSED slot scales: standalone preshuffle with the build-time `padded_m`
        # stride — exactly what `ep_wait` writes under `fuse_a_scale_preshuffle`.
        # The preshuffle op requires its scalar operands on the host CPU.
        padded_m_scalar = ops.constant(
            padded_m, dtype=DType.uint32, device=cpu_ref
        )
        n_active_scalar = ops.constant(
            n_experts, dtype=DType.uint32, device=cpu_ref
        )
        slot_scales = block_scaled_preshuffle_grouped_scale_4d(
            raw_scales,
            row_offsets_t,
            padded_m_scalar,
            n_active_scalar,
            num_experts=n_experts,
        )

        est_m = ops.constant(total_tokens, dtype=DType.uint32, device=cpu_ref)
        c_ref = grouped_dynamic_block_scaled_matmul_amd(
            a_tokens,
            b_pre,
            raw_scales,
            b_scales_t,
            row_offsets_t,
            expert_ids_t,
            usage_t,
            estimated_total_m=est_m,
            preshuffled_b=True,
            a_scales_preshuffled=False,
        )
        c_fused = grouped_dynamic_block_scaled_matmul_amd(
            a_tokens,
            b_pre,
            slot_scales,
            b_scales_t,
            row_offsets_t,
            expert_ids_t,
            usage_t,
            estimated_total_m=est_m,
            preshuffled_b=True,
            a_scales_preshuffled=True,
            a_scales_max_padded_m=padded_m,
        )
        graph.output(c_ref.cast(DType.float32), c_fused.cast(DType.float32))

    compiled = session.load(graph)
    cpu = Accelerator.cpu()
    c_ref_buf, c_fused_buf = compiled.execute(
        Buffer.from_dlpack(act_torch).to(device),
        Buffer.from_dlpack(row_offsets_torch).to(device),
        Buffer.from_dlpack(b_torch).to(device),
        Buffer.from_dlpack(b_scales_torch)
        .view(DType.float8_e8m0fnu)
        .to(device),
        Buffer.from_dlpack(expert_ids_torch).to(device),
        Buffer.from_dlpack(usage_torch),
    )
    c_ref_np = np.from_dlpack(c_ref_buf.to(cpu))
    c_fused_np = np.from_dlpack(c_fused_buf.to(cpu))

    np.testing.assert_allclose(
        c_fused_np,
        c_ref_np,
        rtol=0,
        atol=0,
        err_msg=(
            "Up-proj matmul output diverged between the fused (slot-stride="
            f"{padded_m}) and non-fused paths for tokens_per_expert="
            f"{tokens_per_expert} — the matmul likely read A-scale slots at the"
            " runtime stride instead of a_scales_max_padded_m."
        ),
    )
