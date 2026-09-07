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
"""Test AttentionWithRope layer with FP8 quantization on AMD GPUs."""

import numpy as np
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType, ops
from max.graph.weights import WeightData
from max.nn import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
)
from max.nn.attention.attention_with_rope import AttentionWithRope
from max.nn.kv_cache import KVCacheParams, MHAKVCacheParams, PagedCacheValues
from max.nn.quant_config import WeightScaleSpec
from max.nn.rotary_embedding import RotaryEmbedding
from test_common.simple_kv_cache import paged_kv_cache_inputs


def _create_fp8_weights(
    num_heads: int,
    num_kv_heads: int,
    hidden_size: int,
    head_dim: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create FP8 weights with negative zeros injected at every second position."""
    torch.manual_seed(seed)

    # Generate random weights
    q_weight = torch.randn(
        num_heads * head_dim, hidden_size, dtype=torch.float32
    )
    k_weight = torch.randn(
        num_kv_heads * head_dim, hidden_size, dtype=torch.float32
    )
    v_weight = torch.randn(
        num_kv_heads * head_dim, hidden_size, dtype=torch.float32
    )
    o_weight = torch.randn(
        hidden_size, num_heads * head_dim, dtype=torch.float32
    )

    # Inject negative zero values - set every second value to negative zero
    negative_zero = torch.tensor(-0.0, dtype=torch.float32)

    for weight in [q_weight, k_weight, v_weight, o_weight]:
        weight_flat = weight.view(-1)
        weight_flat[1::2] = negative_zero

    # Convert to FP8
    return (
        q_weight.to(torch.float8_e4m3fn),
        k_weight.to(torch.float8_e4m3fn),
        v_weight.to(torch.float8_e4m3fn),
        o_weight.to(torch.float8_e4m3fn),
    )


def _create_kv_params(num_kv_heads: int, head_dim: int) -> MHAKVCacheParams:
    """Describe the KV cache page layout for the attention layer."""
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        page_size=128,
        n_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=1,
        devices=[DeviceRef.GPU()],
    )


def _create_attention_state_dict(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    quant_config: QuantConfig,
    num_heads: int,
    num_kv_heads: int,
    hidden_size: int,
    head_dim: int,
) -> dict[str, WeightData]:
    """Create state dict for AttentionWithRope with FP8 weights and scales."""
    state_dict: dict[str, WeightData] = {}

    # Common weight entries for all projections
    weights_info = [
        ("q_proj", q_weight, num_heads * head_dim),
        ("k_proj", k_weight, num_kv_heads * head_dim),
        ("v_proj", v_weight, num_kv_heads * head_dim),
        ("o_proj", o_weight, hidden_size),
    ]

    for proj_name, weight, out_dim in weights_info:
        # Add weight
        state_dict[f"{proj_name}.weight"] = WeightData(
            Buffer.from_dlpack(weight.view(torch.uint8)).view(
                DType.float8_e4m3fn
            ),
            f"{proj_name}.weight",
            DType.float8_e4m3fn,
            Shape(weight.shape),
        )

        # Add weight scale based on granularity
        if quant_config.weight_scale.granularity == ScaleGranularity.TENSOR:
            # Static scaling - single scale value
            scale_tensor = torch.tensor([1.0], dtype=torch.float32)
            scale_shape = Shape([1])
        else:  # ROWWISE
            # Dynamic scaling - per-row scales
            scale_tensor = torch.ones(out_dim, 1, dtype=torch.float32)
            scale_shape = Shape(scale_tensor.shape)

        state_dict[f"{proj_name}.weight_scale"] = WeightData(
            Buffer.from_dlpack(scale_tensor),
            f"{proj_name}.weight_scale",
            DType.float32,
            scale_shape,
        )

        # Add input scale only for static scaling
        if quant_config.input_scale.origin == ScaleOrigin.STATIC:
            input_scale_tensor = torch.tensor([1.0], dtype=torch.float32)
            state_dict[f"{proj_name}.input_scale"] = WeightData(
                Buffer.from_dlpack(input_scale_tensor),
                f"{proj_name}.input_scale",
                DType.float32,
                Shape([1]),
            )

    return state_dict


def _build_and_execute_attention_graph(
    attention: AttentionWithRope,
    rope: RotaryEmbedding,
    kv_params: KVCacheParams,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    device: Accelerator,
    gpu_session: InferenceSession,
    graph_name: str,
) -> torch.Tensor:
    """Build graph, execute model, and return results."""
    kv_symbolic_inputs = kv_params.get_symbolic_inputs().inputs[0]
    dispatch_metadata_symbol = kv_symbolic_inputs.attention_dispatch_metadata
    assert dispatch_metadata_symbol is not None

    # Prepare input data
    np.random.seed(42)
    input_data = np.random.randn(batch_size * seq_len, hidden_size).astype(
        np.float32
    )

    if batch_size == 1:
        input_row_offsets_data = np.array([0, seq_len], dtype=np.uint32)
    else:
        input_row_offsets_data = np.array(
            [0, seq_len, seq_len * 2], dtype=np.uint32
        )

    with Graph(
        graph_name,
        input_types=[
            TensorType(
                DType.bfloat16,
                shape=("seq_len", hidden_size),
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.uint32,
                shape=["row_offsets_length"],
                device=DeviceRef.GPU(),
            ),
            *kv_symbolic_inputs.flatten(),
        ],
    ) as graph:
        freqs_cis = rope.freqs_cis
        layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

        (
            x,
            input_row_offsets,
            blocks,
            cache_lengths,
            lookup_table,
            max_prompt_length,
            max_cache_length,
            attention_dispatch_metadata,
        ) = graph.inputs

        kv_collection = PagedCacheValues(
            blocks.buffer,
            cache_lengths.tensor,
            lookup_table.tensor,
            max_prompt_length.tensor,
            max_cache_length.tensor,
            attention_dispatch_metadata=attention_dispatch_metadata.tensor,
        )
        output = attention(
            layer_idx=layer_idx.tensor,
            x=x.tensor,
            kv_collection=kv_collection,
            freqs_cis=freqs_cis,
            input_row_offsets=input_row_offsets.tensor,
        )

        graph.output(output)

    model = gpu_session.load(graph, weights_registry=attention.state_dict())

    # Prepare tensors for execution
    input_tensor = Buffer.from_dlpack(
        torch.from_numpy(input_data).to(torch.bfloat16)
    ).to(device)
    input_row_offsets_tensor = Buffer.from_dlpack(
        torch.from_numpy(input_row_offsets_data)
    ).to(device)

    kv_runtime_inputs = paged_kv_cache_inputs(
        kv_params, [seq_len] * batch_size, total_num_pages=8
    )
    assert kv_runtime_inputs.attention_dispatch_metadata is not None

    result = model.execute(
        input_tensor,
        input_row_offsets_tensor,
        *kv_runtime_inputs.flatten(),
    )[0]

    return torch.from_dlpack(result)


def test_attention_with_rope_fp8_amd_static(
    gpu_session: InferenceSession,
) -> None:
    """Test AttentionWithRope applies AMD FP8 conversion with static scaling."""

    # Configuration for static scaling
    quant_config = QuantConfig(
        format=QuantFormat.COMPRESSED_TENSORS_FP8,
        input_scale=InputScaleSpec(
            dtype=DType.float32,
            granularity=ScaleGranularity.TENSOR,
            origin=ScaleOrigin.STATIC,
        ),
        weight_scale=WeightScaleSpec(
            dtype=DType.float32,
            granularity=ScaleGranularity.TENSOR,
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
    )

    # Test parameters
    batch_size = 1
    seq_len = 4
    hidden_size = 128
    num_heads = 4
    num_kv_heads = 4
    head_dim = hidden_size // num_heads

    device = Accelerator(0)
    head_dim = hidden_size // num_heads

    # Set up KV cache and rope
    rope = RotaryEmbedding(
        dim=hidden_size,
        n_heads=num_heads,
        theta=10000.0,
        max_seq_len=seq_len * 2,
    )

    kv_params = _create_kv_params(num_kv_heads, head_dim)

    # Create AttentionWithRope layer with quant_config
    attention = AttentionWithRope(
        rope=rope,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=hidden_size,
        kv_params=kv_params,
        devices=[DeviceRef.GPU()],
        dtype=DType.float8_e4m3fn,
        quant_config=quant_config,
    )

    # Create weights with negative zeros
    q_weight, k_weight, v_weight, o_weight = _create_fp8_weights(
        num_heads, num_kv_heads, hidden_size, head_dim
    )

    # Create and load state dict using helper function
    state_dict = _create_attention_state_dict(
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        quant_config,
        num_heads,
        num_kv_heads,
        hidden_size,
        head_dim,
    )
    attention.load_state_dict(state_dict)

    # Execute the test
    result_torch = _build_and_execute_attention_graph(
        attention,
        rope,
        kv_params,
        batch_size,
        seq_len,
        hidden_size,
        device,
        gpu_session,
        "test_attention_fp8_amd_static",
    )

    assert torch.isfinite(result_torch).all(), (
        "Output should be finite (no NaN or Inf from FP8 conversion)"
    )


def test_attention_with_rope_fp8_amd_dynamic(
    gpu_session: InferenceSession,
) -> None:
    """Test AttentionWithRope applies AMD FP8 conversion with dynamic scaling."""

    # Configuration for dynamic scaling
    quant_config = QuantConfig(
        format=QuantFormat.FBGEMM_FP8,
        input_scale=InputScaleSpec(
            dtype=DType.float32,
            granularity=ScaleGranularity.COLWISE,
            origin=ScaleOrigin.DYNAMIC,
        ),
        weight_scale=WeightScaleSpec(
            dtype=DType.float32,
            granularity=ScaleGranularity.ROWWISE,
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
    )

    # Test parameters
    batch_size = 2
    seq_len = 8
    hidden_size = 256
    num_heads = 8
    num_kv_heads = 8

    device = Accelerator(0)
    head_dim = hidden_size // num_heads

    # Set up KV cache and rope
    rope = RotaryEmbedding(
        dim=hidden_size,
        n_heads=num_heads,
        theta=10000.0,
        max_seq_len=seq_len * 2,
    )

    kv_params = _create_kv_params(num_kv_heads, head_dim)

    # Create AttentionWithRope layer with dynamic quant_config
    attention = AttentionWithRope(
        rope=rope,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=hidden_size,
        kv_params=kv_params,
        devices=[DeviceRef.GPU()],
        dtype=DType.float8_e4m3fn,
        quant_config=quant_config,
    )

    # Create weights with negative zeros
    q_weight, k_weight, v_weight, o_weight = _create_fp8_weights(
        num_heads, num_kv_heads, hidden_size, head_dim
    )

    # Create and load state dict using helper function
    state_dict = _create_attention_state_dict(
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        quant_config,
        num_heads,
        num_kv_heads,
        hidden_size,
        head_dim,
    )
    attention.load_state_dict(state_dict)

    # Execute the test
    result_torch = _build_and_execute_attention_graph(
        attention,
        rope,
        kv_params,
        batch_size,
        seq_len,
        hidden_size,
        device,
        gpu_session,
        "test_attention_fp8_amd_dynamic",
    )

    assert torch.isfinite(result_torch).all(), (
        "Output should be finite (no NaN or Inf from FP8 conversion)"
    )
