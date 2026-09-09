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
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.torch import max_dtype_to_torch
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.olmo2.layers.attention import (
    Olmo2Attention as MaxOlmo2Attention,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from transformers.models.olmo2.modeling_olmo2 import Olmo2RotaryEmbedding
from transformers.models.olmo2.modular_olmo2 import Olmo2Attention, Olmo2Config

# Max position embeddings for OLMo2-7B
# Based on OLMo2 configuration
MAX_SEQ_LEN = 4096


@pytest.fixture
def input_tensor(text_config: Olmo2Config) -> torch.Tensor:
    torch.manual_seed(39)
    # Use the hidden size from the config
    return torch.randn(1, 11, 4096).to(torch.float32).to("cuda")


def _get_position_embeddings(
    text_config: Olmo2Config,
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates rotary position embeddings based on the input tensor shape."""

    seq_len = input_tensor.shape[1]
    rotary_emb = Olmo2RotaryEmbedding(config=text_config, device="cuda")
    position_ids = torch.arange(
        seq_len, dtype=torch.long, device="cuda"
    ).unsqueeze(0)
    cos, sin = rotary_emb(input_tensor, position_ids)
    return cos.to(torch.float32).to("cuda"), sin.to(torch.float32).to("cuda")


def _causal_attention_mask(seq_len: int) -> torch.Tensor:
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda"),
        diagonal=1,
    )
    attention_mask = torch.zeros(
        1, 1, seq_len, seq_len, dtype=torch.float32, device="cuda"
    )
    attention_mask = attention_mask.masked_fill(
        causal_mask[None, None, :, :], torch.finfo(torch.float32).min
    )
    return attention_mask


@torch.no_grad()
def generate_torch_outputs(
    text_config: Olmo2Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    layer = (
        Olmo2Attention(
            text_config,
            layer_idx=0,
        )
        .to(torch.float32)
        .to("cuda")
    )

    for name, param in layer.named_parameters():
        param.data = attention_weights[name].to(torch.float32).to("cuda")

    attention_mask = _causal_attention_mask(input_tensor.shape[1])
    position_embeddings = _get_position_embeddings(text_config, input_tensor)

    return layer(input_tensor, position_embeddings, attention_mask)[0]


def generate_max_outputs(
    text_config: Olmo2Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Runs the MAX Olmo2 attention layer.
    Returns the outputs:
    1) Layer with rope
    2) Attention without rope but with attention tuning
    """
    is_gpu = isinstance(device, Accelerator)
    input_tensor = input_tensor.cuda() if is_gpu else input_tensor.cpu()
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()
    input_seq_len = input_tensor.shape[1]

    state_dict = {
        weight_name: value.to(max_dtype_to_torch(dtype)).cpu()
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

    assert text_config.head_dim == 128
    print("inja inja text_config.head_dim")
    print(text_config.head_dim)

    attention = MaxOlmo2Attention(
        rope=Llama3RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=get_rope_theta(text_config),
            max_seq_len=MAX_SEQ_LEN,
            interleaved=False,
            head_dim=text_config.head_dim,
        ),
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
        total_num_pages=1,
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
        "Olmo2Attention",
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
                freqs_cis=attention.rope.freqs_cis,
                input_row_offsets=input_row_offsets.tensor,
            )
        )

    compiled = session.load(graph, weights_registry=attention.state_dict())

    # Set up cache inputs and call the compiled model.
    batch = [create_text_context(np.empty(input_seq_len))]
    kv_manager.claim(batch[0])
    kv_manager.alloc(batch[0])
    kv_runtime_inputs = kv_manager.runtime_inputs([batch])

    output = compiled.execute(
        Buffer.from_dlpack(input_tensor[0]).to(device),
        Buffer.from_numpy(np.array([0, input_seq_len], dtype=np.uint32)).to(
            device
        ),
        *kv_runtime_inputs.flatten(),
    )[0]

    return output


def test_attention(
    text_config: Olmo2Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> None:
    torch_output = generate_torch_outputs(
        text_config, input_tensor, attention_weights
    )

    max_output = generate_max_outputs(
        text_config=text_config,
        input_tensor=input_tensor,
        attention_weights=attention_weights,
        dtype=DType.float32,
        device=Accelerator(),
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(torch.float32),
        from_dlpack(max_output).to(torch.float32),
        rtol=0.01,
        atol=0.01,
    )
