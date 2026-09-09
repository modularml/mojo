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

import numpy as np
import pytest
import torch
from max._core.engine import PrintStyle
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType, ops
from max.graph.weights import WeightData
from max.nn.attention.multi_latent_attention_fp8 import (
    LatentAttentionWithRopeFp8,
)
from max.nn.kv_cache import MLAKVCacheParams
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
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from torch_reference.configuration_deepseek import DeepseekV2Config
from torch_reference.modeling_deepseek import DeepseekV2Attention

RTOL = 0.006
ATOL = 0.006


def _quantize_attention_weights_to_fp8(
    attention_weights: dict[str, torch.Tensor],
) -> dict[str, WeightData]:
    """Quantize bfloat16 attention weights to FP8 format with block-wise scaling.

    Quantizes all weights except layer norm weights (which remain in bfloat16).
    Uses block-wise dynamic scaling with [128, 128] blocks as per DeepSeek V3.

    The weights are wrapped as WeightData because MAX Buffer.from_dlpack() does
    not support float8.

    Args:
        attention_weights: Dictionary of bfloat16 attention weights

    Returns:
        Dictionary with FP8 quantized weights and corresponding scales
    """
    quantized_weights = {}

    # Layer norm weights should remain in bfloat16
    layernorm_keys = ["kv_a_layernorm", "kv_a_proj_layernorm", "q_a_layernorm"]

    for key, weight in attention_weights.items():
        # Keep layer norm weights in bfloat16
        if any(ln_key in key for ln_key in layernorm_keys):
            quantized_weights[key] = WeightData(
                weight, key, DType.bfloat16, Shape(weight.shape)
            )
            continue

        # Quantize linear layer weights to FP8 with block-wise scaling
        if (
            weight.dim() == 2 and weight.numel() > 128 * 128
        ):  # Only quantize large matrices
            fp8_weight, weight_scales = _quantize_weight_blockwise_fp8(
                weight, block_size=[128, 128]
            )
            quantized_weights[key] = WeightData(
                Buffer.from_dlpack(fp8_weight.view(torch.uint8)).view(
                    DType.float8_e4m3fn
                ),
                key,
                DType.float8_e4m3fn,
                Shape(fp8_weight.shape),
            )
            quantized_weights[f"{key}_scale"] = WeightData(
                weight_scales,
                f"{key}_scale",
                DType.float32,
                Shape(weight_scales.shape),
            )
        else:
            # Keep small weights or biases in bfloat16
            quantized_weights[key] = WeightData(
                weight, key, DType.bfloat16, Shape(weight.shape)
            )

    return quantized_weights


def _quantize_weight_blockwise_fp8(
    weight: torch.Tensor, block_size: list[int] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to FP8 using block-wise dynamic scaling.

    Args:
        weight: Input weight tensor [M, N] in bfloat16
        block_size: Block dimensions [block_m, block_n] for scaling

    Returns:
        Tuple of (quantized_weight, scales) where:
        - quantized_weight: FP8 E4M3 quantized weights
        - scales: bfloat16 scale factors per block
    """
    if block_size is None:
        block_size = [128, 128]

    M, N = weight.shape
    block_m, block_n = block_size

    # Calculate number of blocks
    num_blocks_m = (M + block_m - 1) // block_m
    num_blocks_n = (N + block_n - 1) // block_n

    # Initialize outputs
    quantized_weight = torch.zeros_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.zeros(num_blocks_m, num_blocks_n, dtype=torch.float32)

    # FP8 E4M3 max finite value
    fp8_max = 448.0

    for i in range(num_blocks_m):
        for j in range(num_blocks_n):
            # Extract block
            start_m = i * block_m
            end_m = min((i + 1) * block_m, M)
            start_n = j * block_n
            end_n = min((j + 1) * block_n, N)

            block = weight[start_m:end_m, start_n:end_n]

            # Compute dynamic scale for this block
            max_val = torch.abs(block).max().item()
            scale_factor = (
                min(max_val, 1200.0) / fp8_max
            )  # 1200.0 is scale upper bound

            if scale_factor == 0:
                scale_factor = 1.0  # Avoid division by zero

            scales[i, j] = scale_factor

            # Quantize block
            scaled_block = block / scale_factor
            clamped_block = torch.clamp(scaled_block, -fp8_max, fp8_max)
            quantized_weight[start_m:end_m, start_n:end_n] = clamped_block.to(
                torch.float8_e4m3fn
            )

    return quantized_weight, scales


def generate_torch_outputs(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    layer = DeepseekV2Attention(config=config, layer_idx=0).to(torch.bfloat16)
    layer.load_state_dict(attention_weights)
    torch_output = layer(input_tensor, attention_mask=attention_mask)
    return torch_output[0]


def generate_max_outputs_fp8(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    use_prefill: bool = True,
    prefill_buffer_size: int = 16384,
) -> torch.Tensor:
    """Generate MAX outputs using FP8 MLA implementation with quantized weights.

    Args:
        config: DeepSeek V2 configuration
        input_tensor: Input tensor
        attention_weights: Attention layer weights (bfloat16)
        use_prefill: Whether to use prefill mode
        prefill_buffer_size: Buffer size for prefill

    Returns:
        Output tensor from the FP8 attention layer
    """
    # Quantize bfloat16 weights to FP8 (except layer norm weights)
    quantized_weights = _quantize_attention_weights_to_fp8(attention_weights)

    device0 = Accelerator(0)
    session = InferenceSession(devices=[device0])
    session.set_debug_print_options(style=PrintStyle.COMPACT)

    assert config.rope_scaling is not None
    scaling_params = DeepseekYarnRopeScalingParams(
        scaling_factor=config.rope_scaling["factor"],
        original_max_position_embeddings=config.rope_scaling[
            "original_max_position_embeddings"
        ],
        beta_fast=config.rope_scaling["beta_fast"],
        beta_slow=config.rope_scaling["beta_slow"],
        mscale=config.rope_scaling["mscale"],
        mscale_all_dim=config.rope_scaling["mscale_all_dim"],
    )

    rope = DeepseekYarnRotaryEmbedding(
        dim=config.qk_rope_head_dim,
        n_heads=config.num_attention_heads,
        theta=config.rope_theta,
        max_seq_len=config.max_position_embeddings,
        scaling_params=scaling_params,
    )

    kv_params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=config.num_hidden_layers,
        devices=[DeviceRef.GPU()],
        page_size=128,
        num_q_heads=config.num_attention_heads,
    )

    # Create FP8 configuration with block-wise dynamic scaling [128, 128]
    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.DYNAMIC,
        dtype=DType.float32,
        block_size=(1, 128),
    )

    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float32,
        block_size=(128, 128),
    )

    quant_config = QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.BLOCKSCALED_FP8,
    )

    latent_attention = LatentAttentionWithRopeFp8(
        rope=rope,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        hidden_size=config.hidden_size,
        kv_params=kv_params,
        quant_config=quant_config,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        devices=[DeviceRef.GPU()],
        buffer_size=prefill_buffer_size,
    )
    latent_attention.load_state_dict(quantized_weights, strict=True)

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )

    hidden_state_type = TensorType(
        DType.bfloat16, ["total_seq_len", config.hidden_size], DeviceRef.GPU()
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], DeviceRef.GPU()
    )

    def construct() -> Graph:
        with Graph(
            "LatentAttentionWithRope",
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

            result = latent_attention(
                ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
                hidden_states,
                kv_collection,
                freqs_cis=rope.freqs_cis,
                input_row_offsets=input_row_offsets,
            )
            graph.output(result)
        return graph

    g = construct()
    compiled = session.load(g, weights_registry=latent_attention.state_dict())
    batch_size = 1
    total_tokens = input_tensor.shape[1]
    prompt_lens = [total_tokens] if use_prefill else [1]

    batch = []
    for i in range(batch_size):
        context = create_text_context(np.empty(prompt_lens[i]))
        kv_manager.claim(context)
        batch.append(context)

    input_row_offsets = Buffer(DType.uint32, [batch_size + 1])
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum

    if not use_prefill:
        all_outputs = []
        for tok_idx in range(total_tokens):
            for ctx in batch:
                kv_manager.alloc(ctx)
            kv_inputs = kv_manager.runtime_inputs_for_leaf([batch]).inputs[0]
            input_tensor_device = (
                Buffer.from_numpy(
                    input_tensor[:, tok_idx, :].view(torch.float16).numpy()
                )
                .view(DType.bfloat16)
                .to(device0)
            )
            max_output = compiled.execute(
                input_tensor_device,
                input_row_offsets.to(device0),
                *kv_inputs.flatten(),
            )

            for ctx in batch:
                ctx.update(42)

            for ctx in batch:
                kv_manager.step(ctx)
            torch_output = from_dlpack(max_output[0]).to(torch.bfloat16)
            all_outputs.append(torch_output[:, None, :].to("cpu"))
        return torch.concat(all_outputs, dim=1)

    for ctx in batch:
        kv_manager.alloc(ctx)
    kv_inputs = kv_manager.runtime_inputs_for_leaf([batch]).inputs[0]
    input_tensor_device = (
        Buffer.from_numpy(input_tensor[0, :, :].view(torch.float16).numpy())
        .view(DType.bfloat16)
        .to(device0)
    )
    max_output = compiled.execute(
        input_tensor_device, input_row_offsets.to(device0), *kv_inputs.flatten()
    )
    torch_output = from_dlpack(max_output[0]).to(torch.bfloat16).to("cpu")
    return torch_output[None, :, :]


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="MLA kernel only supports Nvidia GPUs"
)
def test_latent_attention_prefill(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> None:
    max_output = generate_max_outputs_fp8(
        config, input_tensor, attention_weights, use_prefill=True
    )

    torch_output = generate_torch_outputs(
        config, input_tensor, attention_mask, attention_weights
    )
    torch.testing.assert_close(
        torch_output.squeeze(0), max_output.squeeze(0), rtol=RTOL, atol=ATOL
    )


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="MLA kernel only supports Nvidia GPUs"
)
def test_latent_attention_decode(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> None:
    max_output = generate_max_outputs_fp8(
        config, input_tensor, attention_weights, use_prefill=False
    )
    torch_output = generate_torch_outputs(
        config, input_tensor, attention_mask, attention_weights
    )
    torch.testing.assert_close(torch_output, max_output, rtol=RTOL, atol=ATOL)
