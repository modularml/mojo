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

import json
import os
import typing
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from max._core.engine import PrintStyle
from max.driver import Accelerator, Buffer, accelerator_architecture_name
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.attention.multi_latent_attention import LatentAttentionWithRope
from max.nn.kv_cache import MLAKVCacheParams
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from torch_reference.configuration_deepseek import DeepseekV2Config

"""
Fixtures for DeepseekV2 tests, including config, generated input tensors, and dummy weights.
"""

WEIGHT_STDDEV = 0.001


def _is_b200() -> bool:
    """Check if running on B200 (sm_100) GPU."""
    try:
        return accelerator_architecture_name() == "sm_100"
    except Exception:
        return False


@pytest.fixture(
    params=[
        pytest.param(DType.bfloat16, id="bf16"),
        pytest.param(
            DType.float8_e4m3fn,
            id="fp8",
            marks=pytest.mark.skipif(
                not _is_b200(),
                reason="FP8 KV cache only supported on B200 (sm_100)",
            ),
        ),
    ]
)
def kv_dtype(request: pytest.FixtureRequest) -> DType:
    """Fixture that provides KV cache dtype."""
    return request.param


@pytest.fixture
def config() -> DeepseekV2Config:
    config = DeepseekV2Config()
    path = os.environ["PIPELINES_TESTDATA"]
    config_path = Path(path) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    config.update(data)
    return config


def _generate_latent_attention_max_outputs(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    use_prefill: bool = True,
    prefill_buffer_size: int = 16384,
    kv_dtype: DType = DType.bfloat16,
) -> torch.Tensor:
    attention_weights = {k: v for k, v in attention_weights.items()}

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
        dtype=kv_dtype,
        head_dim=576,
        num_layers=config.num_hidden_layers,
        devices=[DeviceRef.GPU()],
        page_size=128,
        num_q_heads=config.num_attention_heads,
    )

    latent_attention = LatentAttentionWithRope(
        rope=rope,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        hidden_size=config.hidden_size,
        kv_params=kv_params,
        dtype=DType.bfloat16,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        devices=[DeviceRef.GPU()],
        buffer_size=prefill_buffer_size,
    )
    latent_attention.load_state_dict(attention_weights)

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
            kv_collection = kv_params.unflatten_kv_inputs(
                iter(graph.inputs[2:])
            ).inputs[0]

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
            kv_inputs = kv_manager.runtime_inputs([batch])
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
    kv_inputs = kv_manager.runtime_inputs([batch])
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


@pytest.fixture
def generate_latent_attention_max_outputs(
    kv_dtype: DType,
) -> typing.Callable[..., torch.Tensor]:
    return partial(_generate_latent_attention_max_outputs, kv_dtype=kv_dtype)


@pytest.fixture
def input_tensor(
    config: DeepseekV2Config,
    seq_len: int = 7,
    batch_size: int = 1,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)  # Set fixed seed for reproducibility
    return torch.randn(
        batch_size,
        seq_len,
        config.hidden_size,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def input_tensor_rope(
    config: DeepseekV2Config,
    seq_len: int = 7,
    batch_size: int = 1,
    seed: int = 1234,
) -> torch.Tensor:
    torch.manual_seed(seed)  # Set fixed seed for reproducibility

    # x: [bs, num_attention_heads, seq_len, head_size]

    return torch.randn(
        batch_size,
        config.num_attention_heads,
        seq_len,
        config.qk_rope_head_dim,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def attention_mask(
    seq_len: int = 7,
    batch_size: int = 1,
) -> torch.Tensor:
    # Create causal mask where future tokens can't attend to past tokens
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    causal_mask = torch.zeros(
        1, batch_size, seq_len, seq_len, dtype=torch.bfloat16
    )
    causal_mask.masked_fill_(mask, float("-inf")).to(torch.bfloat16)
    return causal_mask


@pytest.fixture
def dummy_moe_weight(
    config: DeepseekV2Config, seed: int = 1234
) -> torch.Tensor:
    """
    Fixture to create dummy weights for an MLP layer.
    Returns tensors in bfloat16 format.
    """
    torch.manual_seed(seed)  # Set fixed seed for reproducibility
    n_experts = (
        config.n_routed_experts if config.n_routed_experts is not None else 64
    )
    return (
        torch.randn(n_experts, config.hidden_size, dtype=torch.bfloat16)
        * WEIGHT_STDDEV
    )


@pytest.fixture
def shared_expert_weights(config: DeepseekV2Config) -> dict[str, torch.Tensor]:
    """Create dummy weights for shared experts"""
    torch.manual_seed(42)  # For reproducibility
    assert isinstance(config.moe_intermediate_size, int)
    assert isinstance(config.n_shared_experts, int)
    shared_experts_intermediate_size = (
        config.moe_intermediate_size * config.n_shared_experts
    )
    expert = {
        "down_proj.weight": torch.randn(
            config.hidden_size,
            shared_experts_intermediate_size,
            dtype=torch.bfloat16,
        )
        * WEIGHT_STDDEV,
        "gate_proj.weight": torch.randn(
            shared_experts_intermediate_size,
            config.hidden_size,
            dtype=torch.bfloat16,
        )
        * WEIGHT_STDDEV,
        "up_proj.weight": torch.randn(
            shared_experts_intermediate_size,
            config.hidden_size,
            dtype=torch.bfloat16,
        )
        * WEIGHT_STDDEV,
    }
    return expert


@pytest.fixture
def expert_weights(config: DeepseekV2Config) -> list[dict[str, torch.Tensor]]:
    """Create dummy weights for individual experts"""
    experts = []
    n_experts = (
        config.n_routed_experts if config.n_routed_experts is not None else 64
    )
    for i in range(n_experts):
        torch.manual_seed(i)  # For reproducibility
        expert = {
            "down_proj.weight": torch.randn(
                config.hidden_size,
                config.moe_intermediate_size,
                dtype=torch.bfloat16,
            )
            * WEIGHT_STDDEV,
            "gate_proj.weight": torch.randn(
                config.moe_intermediate_size,
                config.hidden_size,
                dtype=torch.bfloat16,
            )
            * WEIGHT_STDDEV,
            "up_proj.weight": torch.randn(
                config.moe_intermediate_size,
                config.hidden_size,
                dtype=torch.bfloat16,
            )
            * WEIGHT_STDDEV,
        }
        experts.append(expert)
    return experts


@pytest.fixture
def attention_weights(config: DeepseekV2Config) -> dict[str, torch.Tensor]:
    """Create dummy weights for DeepseekV2Attention module"""
    torch.manual_seed(42)  # For reproducibility

    weight_scale = 192.0  # so that we won't get overflow in the attention layer

    weights = {}

    # Query projection weights
    if config.q_lora_rank is not None:
        weights["q_a_proj.weight"] = (
            torch.randn(
                config.q_lora_rank,
                config.hidden_size,
                dtype=torch.bfloat16,
            )
            / weight_scale
        )
        weights["q_a_layernorm.weight"] = torch.ones(
            config.q_lora_rank, dtype=torch.bfloat16
        )
        weights["q_b_proj.weight"] = (
            torch.randn(
                config.num_attention_heads
                * (config.qk_nope_head_dim + config.qk_rope_head_dim),
                config.q_lora_rank,
                dtype=torch.bfloat16,
            )
            / weight_scale
        )
    else:
        weights["q_proj.weight"] = (
            torch.randn(
                config.num_attention_heads
                * (config.qk_nope_head_dim + config.qk_rope_head_dim),
                config.hidden_size,
                dtype=torch.bfloat16,
            )
            / weight_scale
        )

    # Key-value projection weights
    weights["kv_a_proj_with_mqa.weight"] = (
        torch.randn(
            config.kv_lora_rank + config.qk_rope_head_dim,
            config.hidden_size,
            dtype=torch.bfloat16,
        )
        / weight_scale
    )
    weights["kv_a_layernorm.weight"] = torch.ones(
        config.kv_lora_rank, dtype=torch.bfloat16
    )
    weights["kv_b_proj.weight"] = (
        torch.randn(
            config.num_attention_heads
            * (config.qk_nope_head_dim + config.v_head_dim),
            config.kv_lora_rank,
            dtype=torch.bfloat16,
        )
        / weight_scale
    )

    # Output projection weights
    weights["o_proj.weight"] = (
        torch.randn(
            config.hidden_size,
            config.num_attention_heads * config.v_head_dim,
            dtype=torch.bfloat16,
        )
        / weight_scale
    )

    return weights
