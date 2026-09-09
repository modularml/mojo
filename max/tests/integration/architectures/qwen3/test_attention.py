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


import math

import max.driver as md
import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.torch import torch_dtype_to_max
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import masked_flash_attention_gpu
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.qwen3.layers.attention import (
    Qwen3Attention as MaxQwen3Attention,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RotaryEmbedding,
)

MAX_SEQ_LEN = 1024


@pytest.fixture
def input_tensor(text_config: Qwen3Config) -> torch.Tensor:
    torch.manual_seed(42)
    # https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json
    # 2048 per Qwen3-1.7B Hidden Size in config.json (5120 if you want to test the 32B attention)
    return torch.randn(1, 11, 2048).to(torch.bfloat16).to("cuda")


def _get_position_embeddings(
    text_config: Qwen3Config,
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates rotary position embeddings based on the input tensor shape."""
    seq_len = input_tensor.shape[1]
    rotary_emb = Qwen3RotaryEmbedding(config=text_config, device="cuda")
    position_ids = torch.arange(
        seq_len, dtype=torch.long, device="cuda"
    ).unsqueeze(0)
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
    text_config: Qwen3Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    layer = (
        Qwen3Attention(
            text_config,
            layer_idx=0,
        )
        .to(torch.bfloat16)
        .to("cuda")
    )

    for name, param in layer.named_parameters():
        param.data = attention_weights[name].to(torch.bfloat16).to("cuda")

    attention_mask = _causal_attention_mask(input_tensor.shape[1])
    position_embeddings = _get_position_embeddings(text_config, input_tensor)

    return layer(input_tensor, position_embeddings, attention_mask)[0]


def generate_max_outputs(
    text_config: Qwen3Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Runs the MAX Qwen3 attention layer.

    Returns the outputs:
    1) Layer with rope
    2) Attention without rope but with attention tuning
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
        n_kv_heads=text_config.num_key_value_heads,
        head_dim=text_config.head_dim,
        num_layers=text_config.num_hidden_layers,
        devices=[device_ref],
    )

    session = InferenceSession(devices=[Accelerator(0)])

    rope = Llama3RotaryEmbedding(
        text_config.hidden_size,
        text_config.num_attention_heads,
        text_config.rope_theta,
        MAX_SEQ_LEN,
        interleaved=False,
        head_dim=text_config.head_dim,
    )
    attention = MaxQwen3Attention(
        rope=rope,
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        hidden_size=text_config.hidden_size,
        kv_params=kv_params,
        dtype=dtype,
        devices=[device_ref],
        layer_idx=0,
    )
    attention.load_state_dict(state_dict)

    # Set up blank KV cache.
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )
    assert isinstance(kv_manager, PagedKVCacheManager)

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
        "Qwen3Attention",
        input_types=(
            input_type,
            input_row_offsets_type,
            *flattened_kv_types,
        ),
    ) as graph:
        inputs, input_row_offsets, *kv_cache = graph.inputs
        kv_collection = kv_params.unflatten_kv_inputs(iter(kv_cache)).inputs[0]

        graph.output(
            attention(
                ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
                inputs.tensor,
                kv_collection,
                freqs_cis=rope.freqs_cis,
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


def test_attention(
    text_config: Qwen3Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> None:
    # TODO: Remove this once we figure out the attention error on AMD GPUs.
    if md.accelerator_api() != "cuda":
        pytest.skip("NVIDIA GPUs are required for this test.")

    torch_output = generate_torch_outputs(
        text_config, input_tensor, attention_weights
    )

    max_output = generate_max_outputs(
        text_config=text_config,
        input_tensor=input_tensor,
        attention_weights=attention_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(torch.bfloat16),
        from_dlpack(max_output).to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )


def _materialized_attention_mask(
    token_mask: torch.Tensor,
) -> torch.Tensor:
    seq_len = token_mask.shape[0]
    mask = torch.full(
        (1, seq_len, seq_len),
        -10000.0,
        dtype=torch.float32,
        device=token_mask.device,
    )
    for row in range(seq_len):
        for col in range(seq_len):
            if bool(token_mask[col]) and col <= row:
                mask[0, row, col] = 0.0
    return mask


def _masked_flash_attention_max(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    dtype = torch_dtype_to_max(q.dtype)
    _batch, q_seq_len, nheads, head_dim = q.shape
    kv_seq_len = k.shape[1]

    q_type = TensorType(
        dtype,
        shape=["batch", q_seq_len, nheads, head_dim],
        device=DeviceRef.GPU(),
    )
    kv_type = TensorType(
        dtype,
        shape=["batch", kv_seq_len, nheads, head_dim],
        device=DeviceRef.GPU(),
    )
    mask_type = TensorType(
        DType.float32,
        shape=["batch", q_seq_len, kv_seq_len],
        device=DeviceRef.GPU(),
    )

    session = InferenceSession(devices=[Accelerator()])
    with Graph(
        "masked_flash_attention_gpu",
        input_types=[q_type, kv_type, kv_type, mask_type],
    ) as graph:
        q_in, k_in, v_in, mask_in = graph.inputs
        graph.output(
            masked_flash_attention_gpu(
                q_in.tensor,
                k_in.tensor,
                v_in.tensor,
                mask_in.tensor,
                scale=math.sqrt(1.0 / head_dim),
            )
        )

    model = session.load(graph)
    output = model.execute(
        q.detach(),
        k.detach(),
        v.detach(),
        mask.detach(),
    )[0]
    assert isinstance(output, Buffer)
    return torch.from_dlpack(output)


def test_masked_flash_attention_gpu_matches_naive() -> None:
    if md.accelerator_api() != "cuda":
        pytest.skip("NVIDIA GPUs are required for this test.")

    batch_size = 1
    seq_len = 7
    nheads = 4
    head_dim = 32
    scale = math.sqrt(1.0 / head_dim)
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(
        (batch_size, seq_len, nheads, head_dim), device=device, dtype=dtype
    )
    k = torch.randn(
        (batch_size, seq_len, nheads, head_dim), device=device, dtype=dtype
    )
    v = torch.randn(
        (batch_size, seq_len, nheads, head_dim), device=device, dtype=dtype
    )
    token_mask = torch.tensor(
        [True, True, True, True, False, False, False],
        device=device,
    )
    materialized_mask = _materialized_attention_mask(token_mask)

    out_max = _masked_flash_attention_max(q, k, v, materialized_mask)

    q_ref = q.permute(0, 2, 1, 3).to(torch.float32)
    k_ref = k.permute(0, 2, 1, 3).to(torch.float32)
    v_ref = v.permute(0, 2, 1, 3).to(torch.float32)
    attn_scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * scale
    attn_scores = attn_scores + materialized_mask.unsqueeze(1)
    attn_probs = torch.softmax(attn_scores, dim=-1)
    out_ref = torch.matmul(attn_probs, v_ref).permute(0, 2, 1, 3)

    torch.testing.assert_close(
        out_max.to(torch.float32),
        out_ref,
        rtol=1e-2,
        atol=2e-2,
    )
