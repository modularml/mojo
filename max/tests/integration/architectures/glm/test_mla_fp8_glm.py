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
"""Graph-level test for FP8 MLA at GLM-5.1-FP8 head dims.

GLM-5.1-FP8 has `qk_nope_head_dim = 192`, `v_head_dim = 256`. The per-head
row count `Dn + Dv = 448` is not a multiple of the 128-row FP8 scale block,
so the FP8 MLA path consumes per-head B-scale at granularity 64 via the
load-time gather in :class:`LatentAttentionWithRopeFp8`, and the SM100 kernel
runs with `BK = 64`.

The test builds the FP8 layer at GLM dims, runs a forward pass on B200 through
`mla_prefill_decode_graph`, and compares to a BF16 reference layer fed the
*same effective weights* (`BF16_reference = dequant_fp8(weight, scale)`).
"""

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType, ops
from max.graph.weights import WeightData
from max.kv_cache import PagedKVCacheManager
from max.nn.attention.multi_latent_attention import LatentAttentionWithRope
from max.nn.attention.multi_latent_attention_fp8 import (
    LatentAttentionWithRopeFp8,
)
from max.nn.kv_cache import KVCacheParams, MLAKVCacheParams
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack

# GLM-5.1-FP8 head dims. Per-head row count Dn + Dv = 448 is not a
# multiple of 128, so the FP8 MLA gather runs at granularity 64.
QK_NOPE_HEAD_DIM = 192
V_HEAD_DIM = 256
QK_ROPE_HEAD_DIM = 64
# H must be in {8, 16, 32, 64, 128} for the MLA decode dispatch metadata.
H_HEADS = 8
Q_LORA_RANK = 256
KV_LORA_RANK = 512
HIDDEN_SIZE = 256
# SEQ_LEN <= MLA_DECODE_MAX_SEQ_LEN (=6) forces the decode branch.
SEQ_LEN = 1
WEIGHT_SCALE = 192.0

# FP8↔BF16 tolerance: both layers see the same effective floats (the BF16
# layer is fed the FP8-dequantized weights), so disagreement is bounded by
# accumulation order and the activation-side FP8 dynamic-scaling rounding
# inside the kernel.
ATOL = 0.05


def _quantize_blockwise_fp8(
    weight: torch.Tensor, block: tuple[int, int] = (128, 128)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 weight to FP8 e4m3fn with per-(Bm, Bk)-block scales."""
    bm, bk = block
    m, k = weight.shape
    nbm = (m + bm - 1) // bm
    nbk = (k + bk - 1) // bk
    fp8_max = 448.0
    quantized = torch.zeros_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.zeros(nbm, nbk, dtype=torch.float32)
    for i in range(nbm):
        for j in range(nbk):
            sm, em = i * bm, min((i + 1) * bm, m)
            sk, ek = j * bk, min((j + 1) * bk, k)
            block_w = weight[sm:em, sk:ek]
            scale = max(block_w.abs().max().item(), 1e-6) / fp8_max
            scales[i, j] = scale
            scaled = (block_w / scale).clamp(-fp8_max, fp8_max)
            quantized[sm:em, sk:ek] = scaled.to(torch.float8_e4m3fn)
    return quantized, scales


def _dequantize_blockwise_fp8(
    quantized: torch.Tensor, scales: torch.Tensor, block: tuple[int, int]
) -> torch.Tensor:
    """`w_real[i,j] = w_fp8[i,j] * scale[i//Bm, j//Bk]`."""
    bm, bk = block
    m, k = quantized.shape
    f32 = quantized.float()
    scale_full = scales.repeat_interleave(bm, dim=0).repeat_interleave(
        bk, dim=1
    )
    return (f32 * scale_full[:m, :k]).to(torch.bfloat16)


def _make_attention_weights(seed: int) -> dict[str, torch.Tensor]:
    """Generate small random BF16 attention weights at GLM head dims."""
    torch.manual_seed(seed)
    qk_head_dim = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
    cache_head_dim = KV_LORA_RANK + QK_ROPE_HEAD_DIM
    return {
        "q_a_proj.weight": torch.randn(
            Q_LORA_RANK, HIDDEN_SIZE, dtype=torch.bfloat16
        )
        / WEIGHT_SCALE,
        "q_a_layernorm.weight": torch.ones(Q_LORA_RANK, dtype=torch.bfloat16),
        "q_b_proj.weight": torch.randn(
            H_HEADS * qk_head_dim, Q_LORA_RANK, dtype=torch.bfloat16
        )
        / WEIGHT_SCALE,
        "kv_a_layernorm.weight": torch.ones(KV_LORA_RANK, dtype=torch.bfloat16),
        "kv_a_proj_with_mqa.weight": torch.randn(
            cache_head_dim, HIDDEN_SIZE, dtype=torch.bfloat16
        )
        / WEIGHT_SCALE,
        "kv_b_proj.weight": torch.randn(
            H_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
            KV_LORA_RANK,
            dtype=torch.bfloat16,
        )
        / WEIGHT_SCALE,
        "o_proj.weight": torch.randn(
            HIDDEN_SIZE, H_HEADS * V_HEAD_DIM, dtype=torch.bfloat16
        )
        / WEIGHT_SCALE,
    }


def _quantize_to_fp8_state_dict(
    bf16_weights: dict[str, torch.Tensor],
) -> tuple[dict[str, WeightData], dict[str, torch.Tensor]]:
    """Quantize all matmul weights to FP8 and return:

    * `fp8_state_dict` for `LatentAttentionWithRopeFp8.load_state_dict`
      (weights + `.weight_scale` entries).
    * `bf16_equivalents` mapping each weight name to the BF16-reconstructed
      value (FP8-dequantized at granularity (128, 128)); this is what the
      reference BF16 layer will see, so the two paths share identical
      effective floats.
    """
    fp8_dict: dict[str, WeightData] = {}
    bf16_equiv: dict[str, torch.Tensor] = {}
    for key, weight in bf16_weights.items():
        if (
            "layernorm" in key
            or weight.dim() != 2
            or weight.numel() <= 128 * 128
        ):
            fp8_dict[key] = WeightData(
                weight, key, DType.bfloat16, Shape(weight.shape)
            )
            bf16_equiv[key] = weight
            continue
        q, s = _quantize_blockwise_fp8(weight, block=(128, 128))
        fp8_dict[key] = WeightData(
            Buffer.from_dlpack(q.view(torch.uint8)).view(DType.float8_e4m3fn),
            key,
            DType.float8_e4m3fn,
            Shape(q.shape),
        )
        fp8_dict[f"{key}_scale"] = WeightData(
            s, f"{key}_scale", DType.float32, Shape(s.shape)
        )
        bf16_equiv[key] = _dequantize_blockwise_fp8(q, s, (128, 128))
    return fp8_dict, bf16_equiv


def _make_rope() -> DeepseekYarnRotaryEmbedding:
    scaling = DeepseekYarnRopeScalingParams(
        scaling_factor=40.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        mscale_all_dim=1.0,
    )
    return DeepseekYarnRotaryEmbedding(
        dim=QK_ROPE_HEAD_DIM,
        n_heads=H_HEADS,
        theta=10000.0,
        max_seq_len=4096,
        scaling_params=scaling,
    )


def _kv_params() -> KVCacheParams:
    return MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        num_layers=1,
        devices=[DeviceRef.GPU()],
        page_size=128,
        num_q_heads=H_HEADS,
    )


def _fp8_quant_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 128),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=(128, 128),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.BLOCKSCALED_FP8,
    )


def _run_layer(
    layer: LatentAttentionWithRope | LatentAttentionWithRopeFp8,
    kv_params: KVCacheParams,
    input_tensor: torch.Tensor,
    device0: Accelerator,
) -> torch.Tensor:
    """Build a Graph that calls `layer` once, compile, and execute."""
    rope = _make_rope()
    layer.rope = rope
    session = InferenceSession(devices=[device0])
    hidden_state_type = TensorType(
        DType.bfloat16, ["total_seq_len", HIDDEN_SIZE], DeviceRef.GPU()
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], DeviceRef.GPU()
    )

    with Graph(
        "GLM_FP8_MLA",
        input_types=(
            hidden_state_type,
            input_row_offsets_type,
            *kv_params.flattened_kv_inputs(),
        ),
    ) as graph:
        hidden_states = graph.inputs[0].tensor
        input_row_offsets = graph.inputs[1].tensor
        kv_collection = (
            kv_params.get_symbolic_inputs()
            .unflatten(iter(graph.inputs[2:]))
            .inputs[0]
        )
        result = layer(
            ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
            hidden_states,
            kv_collection,
            freqs_cis=rope.freqs_cis,
            input_row_offsets=input_row_offsets,
        )
        graph.output(result)

    compiled = session.load(graph, weights_registry=layer.state_dict())
    kv_manager = PagedKVCacheManager(
        params=kv_params, total_num_pages=8, session=session, max_batch_size=4
    )
    ctx = create_text_context(np.empty(SEQ_LEN))
    kv_manager.claim(ctx)
    kv_manager.alloc(ctx)
    kv_inputs = kv_manager.runtime_inputs_for_leaf([[ctx]]).inputs[0]
    row_offsets_buf = Buffer(DType.uint32, [2])
    row_offsets_buf[0] = 0
    row_offsets_buf[1] = SEQ_LEN

    input_dev = (
        Buffer.from_dlpack(input_tensor.view(torch.float16))
        .view(DType.bfloat16)
        .to(device0)
    )
    max_output = compiled.execute(
        input_dev, row_offsets_buf.to(device0), *kv_inputs.flatten()
    )
    return from_dlpack(max_output[0]).to(torch.bfloat16).to("cpu")


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="MLA FP8 kernel only supports NVIDIA"
)
def test_fp8_mla_matches_bf16_reference() -> None:
    """FP8 MLA at GLM dims produces the same forward output as a BF16
    reference layer that consumes the same FP8-dequantized weights.
    """
    bf16_weights = _make_attention_weights(seed=2758)
    fp8_state, bf16_equiv = _quantize_to_fp8_state_dict(bf16_weights)

    device0 = Accelerator(0)
    kv_params = _kv_params()
    quant_config = _fp8_quant_config()

    fp8_layer = LatentAttentionWithRopeFp8(
        rope=_make_rope(),
        num_attention_heads=H_HEADS,
        num_key_value_heads=1,
        hidden_size=HIDDEN_SIZE,
        kv_params=kv_params,
        quant_config=quant_config,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        devices=[DeviceRef.GPU()],
    )
    fp8_layer.load_state_dict(fp8_state, strict=True)

    bf16_layer = LatentAttentionWithRope(
        rope=_make_rope(),
        num_attention_heads=H_HEADS,
        num_key_value_heads=1,
        hidden_size=HIDDEN_SIZE,
        kv_params=kv_params,
        dtype=DType.bfloat16,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        devices=[DeviceRef.GPU()],
    )
    bf16_layer.load_state_dict(bf16_equiv, strict=True)

    torch.manual_seed(7)
    input_tensor = torch.randn(SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)

    fp8_out = _run_layer(fp8_layer, kv_params, input_tensor, device0)
    bf16_out = _run_layer(bf16_layer, kv_params, input_tensor, device0)

    assert torch.isfinite(fp8_out.float()).all(), "FP8 output non-finite"
    diff = (fp8_out.float() - bf16_out.float()).abs()
    max_abs = float(diff.max())
    print(f"max abs diff = {max_abs:.6f} (atol={ATOL})")
    assert max_abs <= ATOL, (
        f"FP8 MLA diverged from BF16 reference: max abs diff = {max_abs}"
    )
