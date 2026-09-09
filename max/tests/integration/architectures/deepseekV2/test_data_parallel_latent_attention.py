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

import typing

import numpy as np
import pytest
import torch
from max._core.engine import PrintStyle
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.attention.multi_latent_attention import (
    DataParallelLatentAttentionWithRope,
)
from max.nn.kv_cache import MLAKVCacheParams
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from torch_reference.configuration_deepseek import DeepseekV2Config


def generate_latent_attention_max_outputs_dp(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    use_prefill: bool = True,
    prefill_buffer_size: int = 16384,
) -> torch.Tensor:
    """Run DataParallelLatentAttentionWithRope in DP mode with a single device
    to exercise the DP call path."""
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
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=config.num_hidden_layers,
        devices=[DeviceRef.GPU()],
        page_size=128,
        num_q_heads=config.num_attention_heads,
    )

    dp_attention = DataParallelLatentAttentionWithRope(
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
    dp_attention.load_state_dict(attention_weights)

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
            "DataParallelLatentAttentionWithRope",
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
            out_list = dp_attention(
                ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
                xs=[hidden_states],
                signal_buffers=[],
                kv_collections=[kv_collection],
                freqs_cis=[rope.freqs_cis],
                input_row_offsets=[input_row_offsets],
            )
            graph.output(out_list[0])
        return graph

    g = construct()
    compiled = session.load(g, weights_registry=dp_attention.state_dict())
    batch_size = 1
    total_tokens = input_tensor.shape[1]
    prompt_lens = [total_tokens] if use_prefill else [1]

    batch = []
    for _ in range(batch_size):
        context = create_text_context(np.empty(prompt_lens[0]))
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


@pytest.mark.skip("MODELS-1039: times out on B200")
def test_data_parallel_latent_attention_prefill_matches_single(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    generate_latent_attention_max_outputs: typing.Callable[..., torch.Tensor],
) -> None:
    single = generate_latent_attention_max_outputs(
        config, input_tensor, attention_weights, use_prefill=True
    )
    dp = generate_latent_attention_max_outputs_dp(
        config, input_tensor, attention_weights, use_prefill=True
    )
    torch.testing.assert_close(single, dp, rtol=5e-4, atol=5e-4)


@pytest.mark.skip("MODELS-1039: times out on B200")
def test_data_parallel_latent_attention_decode_matches_single(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    generate_latent_attention_max_outputs: typing.Callable[..., torch.Tensor],
) -> None:
    single = generate_latent_attention_max_outputs(
        config, input_tensor, attention_weights, use_prefill=False
    )
    dp = generate_latent_attention_max_outputs_dp(
        config, input_tensor, attention_weights, use_prefill=False
    )
    torch.testing.assert_close(single, dp, rtol=5e-4, atol=5e-4)
