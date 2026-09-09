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


import copy

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kv_cache import MHAKVCacheParams, PagedCacheValues
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.gemma3.layers.attention import (
    Gemma3Attention as MaxGemma3Attention,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3RotaryEmbedding,
)

MAX_SEQ_LEN = 1152


@pytest.fixture
def input_tensor(text_config: Gemma3TextConfig) -> torch.Tensor:
    torch.manual_seed(42)
    return (
        torch.randn(1, 11, text_config.hidden_size)
        .to(torch.bfloat16)
        .to("cuda")
    )


def _get_position_embeddings(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    use_global_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates rotary position embeddings based on the input tensor shape."""
    seq_len = input_tensor.shape[1]
    position_ids = torch.arange(
        seq_len, dtype=torch.long, device="cuda"
    ).unsqueeze(0)

    rope_params = getattr(text_config, "rope_parameters", None)
    if isinstance(rope_params, dict) and "sliding_attention" in rope_params:
        # v5: single embedding handles both layer types natively
        rotary_emb = Gemma3RotaryEmbedding(config=text_config).to("cuda")
        layer_type = (
            "full_attention" if use_global_rope else "sliding_attention"
        )
        cos, sin = rotary_emb(input_tensor, position_ids, layer_type=layer_type)
    else:
        # v4: need separate embedding with hacked config for local rope
        if use_global_rope:
            rotary_emb = Gemma3RotaryEmbedding(config=text_config).to("cuda")
        else:
            config = copy.deepcopy(text_config)
            config.rope_theta = config.rope_local_base_freq
            config.rope_scaling = {"rope_type": "default"}
            rotary_emb = Gemma3RotaryEmbedding(config=config).to("cuda")
        cos, sin = rotary_emb(input_tensor, position_ids)

    return cos.to(torch.bfloat16).to("cuda"), sin.to(torch.bfloat16).to("cuda")


def _causal_attention_mask(seq_len: int) -> torch.Tensor:
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda"),
        diagonal=1,
    )
    attention_mask = torch.zeros(
        1, 1, seq_len, seq_len, dtype=torch.bfloat16, device="cuda"
    )
    attention_mask = attention_mask.masked_fill(
        causal_mask[None, None, :, :], torch.finfo(torch.bfloat16).min
    )
    return attention_mask


@torch.no_grad()
def generate_torch_outputs(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    layer_idx: int,
) -> torch.Tensor:
    """Generates the outputs of the MAX and PyTorch attention layers.

    `layer_idx` affects whether the local or global `RoPE` is used. When
    `layer_idx % 6 == 5`, the global `RoPE` is used. Otherwise, the local `RoPE`
    is used.
    """
    layer = (
        Gemma3Attention(
            text_config,
            layer_idx=layer_idx,
        )
        .to(torch.bfloat16)
        .to("cuda")
    )

    for name, param in layer.named_parameters():
        param.data = attention_weights[name].to(torch.bfloat16).to("cuda")

    attention_mask = _causal_attention_mask(input_tensor.shape[1])
    use_global_rope = layer_idx % 6 == 5
    position_embeddings = _get_position_embeddings(
        text_config, input_tensor, use_global_rope
    )

    return layer(input_tensor, position_embeddings, attention_mask)[0]


def generate_max_outputs(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
    layer_idx: int,
) -> torch.Tensor:
    """Runs the MAX Llama4 attention layer.

    Returns the outputs:
    1) Layer with rope
    2) Attention without rope but with attention tuning

    `layer_idx` affects whether the local or global `RoPE` is used. When
    `layer_idx % 6 == 5`, the global `RoPE` is used. Otherwise, the local `RoPE`
    is used.
    """
    is_gpu = isinstance(device, Accelerator)
    input_tensor = input_tensor.cuda() if is_gpu else input_tensor.cpu()
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()
    input_seq_len = input_tensor.shape[1]

    state_dict = {
        weight_name: value.cpu()
        for weight_name, value in attention_weights.items()
    }

    kv_params = MHAKVCacheParams(
        dtype=dtype,
        devices=[device_ref],
        n_kv_heads=text_config.num_key_value_heads,
        head_dim=text_config.head_dim,
        num_layers=text_config.num_hidden_layers,
        page_size=256,
    )

    session = InferenceSession(devices=[Accelerator(0)])

    rope_params = getattr(text_config, "rope_parameters", None)
    if isinstance(rope_params, dict) and "full_attention" in rope_params:
        rope_theta = rope_params["full_attention"]["rope_theta"]
        rope_local_base_freq = rope_params["sliding_attention"]["rope_theta"]
    else:
        rope_theta = text_config.rope_theta
        rope_local_base_freq = text_config.rope_local_base_freq

    attention = MaxGemma3Attention(
        rope_global=Llama3RotaryEmbedding(
            text_config.hidden_size,
            text_config.num_attention_heads,
            rope_theta,
            MAX_SEQ_LEN,
            interleaved=False,
            head_dim=text_config.head_dim,
        ),
        rope_local=Llama3RotaryEmbedding(
            text_config.hidden_size,
            text_config.num_attention_heads,
            rope_local_base_freq,
            MAX_SEQ_LEN,
            interleaved=False,
            head_dim=text_config.head_dim,
        ),
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        hidden_size=text_config.hidden_size,
        kv_params=kv_params,
        dtype=dtype,
        devices=[device_ref],
        layer_idx=layer_idx,
        is_sliding=bool((layer_idx + 1) % text_config.sliding_window_pattern),
    )
    attention.load_state_dict(state_dict)

    # Set up blank KV cache.
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )

    # Construct input types.
    input_type = TensorType(
        dtype,
        ["total_seq_len", text_config.hidden_size],
        device=device_ref,
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=device_ref
    )
    flattened_kv_types = kv_params.flattened_kv_inputs()

    # Build graph.
    with Graph(
        "Gemma3Attention",
        input_types=(
            input_type,
            input_row_offsets_type,
            *flattened_kv_types,
        ),
    ) as graph:
        inputs, input_row_offsets, *kv_cache = graph.inputs
        kv_collection: PagedCacheValues = kv_params.unflatten_kv_inputs(
            iter(kv_cache)
        ).inputs[0]

        graph.output(
            attention(
                inputs.tensor,
                kv_collection,
                input_row_offsets=input_row_offsets.tensor,
            )
        )

    compiled = session.load(graph, weights_registry=attention.state_dict())

    # Set up cache inputs and call the compiled model.
    batch = [create_text_context(np.empty(input_seq_len))]
    kv_manager.claim(batch[0])
    kv_manager.alloc(batch[0])
    kv_runtime_inputs = kv_manager.runtime_inputs_for_leaf([batch]).inputs[0]
    assert kv_runtime_inputs.attention_dispatch_metadata is not None

    output = compiled.execute(
        Buffer.from_dlpack(input_tensor[0]).to(device),
        Buffer.from_numpy(np.array([0, input_seq_len], dtype=np.uint32)).to(
            device
        ),
        kv_runtime_inputs.kv_blocks.to(device),
        kv_runtime_inputs.cache_lengths.to(device),
        kv_runtime_inputs.lookup_table.to(device),
        kv_runtime_inputs.max_prompt_length,
        kv_runtime_inputs.max_cache_length,
        kv_runtime_inputs.attention_dispatch_metadata,
    )[0]

    return output


def test_attention_local_rope(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> None:
    torch_output = generate_torch_outputs(
        text_config, input_tensor, attention_weights, layer_idx=0
    )

    max_output = generate_max_outputs(
        text_config,
        input_tensor,
        attention_weights,
        DType.bfloat16,
        Accelerator(),
        layer_idx=0,
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(torch.bfloat16),
        from_dlpack(max_output).to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )


def test_attention_global_rope(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> None:
    torch_output = generate_torch_outputs(
        text_config,
        input_tensor,
        attention_weights,
        layer_idx=5,
    )

    max_output = generate_max_outputs(
        text_config,
        input_tensor,
        attention_weights,
        DType.bfloat16,
        Accelerator(),
        layer_idx=5,
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(torch.bfloat16),
        from_dlpack(max_output).to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )
