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

"""Tests for Qwen3VL-MoE text attention layer."""

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.torch import max_dtype_to_torch
from max.graph import DeviceRef, Dim, Graph, TensorType, ops
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.linear import Linear
from max.pipelines import KVCacheConfig
from max.pipelines.architectures.qwen3vl_moe.nn.text_attention import (
    Qwen3VLMoEDecoderAttentionWithRope,
)
from max.pipelines.architectures.qwen3vl_moe.nn.text_rotary import (
    Qwen3VLTextRotaryEmbedding,
)
from max.pipelines.kv_cache import PagedKVCacheManager, load_kv_manager
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextAttention as HFQwen3VLMoeTextAttention,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextRotaryEmbedding as HFQwen3VLMoeTextRotaryEmbedding,
)
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader

# Looser tolerances for BF16 comparisons.
DECODE_RTOL = 2e-2
DECODE_ATOL = 3e-2

MAX_SEQ_LEN = 2048


def _build_input_row_offsets(seq_lens: list[int]) -> torch.Tensor:
    """Build ragged offsets [0, L0, L0+L1, ...] as uint32."""
    offsets = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    offsets[1:] = torch.tensor(seq_lens, dtype=torch.int32).cumsum(0)
    return offsets.to(torch.uint32)


def _flatten_sequences(sequences: list[torch.Tensor]) -> torch.Tensor:
    """Flatten a list of [Li, H] tensors into [sum(Li), H]."""
    return torch.cat(sequences, dim=0)


def generate_text_attention_weights(
    text_config: dict,  # type: ignore[type-arg]
) -> dict[str, torch.Tensor]:
    """Generate text attention weights for Qwen3VL-MoE.

    For testing, we make q_norm.weight and k_norm.weight all ones so that
    HF and MAX use the same RMSNorm gamma.
    """
    torch.manual_seed(42)

    # TODO: Refactor these as a conftest, similar to how Olmo2 and Gemma3 do it.
    Q_PROJ_STD = 0.034377
    K_PROJ_STD = 0.031698
    V_PROJ_STD = 0.031565
    O_PROJ_STD = 0.035518

    hidden_size = text_config["hidden_size"]
    num_heads = text_config["num_attention_heads"]
    num_kv_heads = text_config["num_key_value_heads"]
    head_dim = text_config["head_dim"]

    q_dim = head_dim * num_heads
    kv_dim = head_dim * num_kv_heads

    weights = {
        "q_proj.weight": (
            torch.randn(q_dim, hidden_size, dtype=torch.bfloat16) * Q_PROJ_STD
        ),
        "k_proj.weight": (
            torch.randn(kv_dim, hidden_size, dtype=torch.bfloat16) * K_PROJ_STD
        ),
        "v_proj.weight": (
            torch.randn(kv_dim, hidden_size, dtype=torch.bfloat16) * V_PROJ_STD
        ),
        "o_proj.weight": (
            torch.randn(hidden_size, q_dim, dtype=torch.bfloat16) * O_PROJ_STD
        ),
        # Set `q_norm` and `k_norm` gamma to one so HF and MAX agree.
        "q_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
        "k_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
    }
    return weights


@torch.no_grad()
def generate_qwen3_torch_outputs(
    sequences: list[torch.Tensor],
    attention_weights: dict[str, torch.Tensor],
    text_config: dict,  # type: ignore[type-arg]
    device: torch.device,
) -> torch.Tensor:
    """Generate reference outputs using HF Qwen3VL-MoE attention.

    We avoid HF's padding/mask complexity by running each sequence separately
    with batch=1 and concatenating the results.
    """
    hf_config = Qwen3VLMoeTextConfig(**text_config)
    # Use eager implementation for deterministic testing.
    hf_config._attn_implementation = "eager"

    hf_attn = (
        HFQwen3VLMoeTextAttention(hf_config, layer_idx=0)
        .to(device)
        .to(torch.bfloat16)
    )
    hf_attn.load_state_dict(attention_weights, strict=True)
    hf_attn.eval()

    hf_rope = (
        HFQwen3VLMoeTextRotaryEmbedding(config=hf_config)
        .to(device)
        .to(torch.bfloat16)
    )
    hf_rope.eval()

    hidden_size = text_config["hidden_size"]
    num_heads = text_config["num_attention_heads"]
    head_dim = text_config["head_dim"]

    outputs = []
    for seq in sequences:
        seq_len = seq.shape[0]
        assert seq.shape[1] == hidden_size

        # HF Qwen3 attention expects [batch, seq, hidden_size]
        hidden_states = seq.unsqueeze(0).to(device=device, dtype=torch.bfloat16)

        # Position IDs: [batch, seq_len]
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)

        # HF rotary embedding expects [batch, seq_len, num_heads, head_dim]
        dummy_input = torch.zeros(
            1,
            seq_len,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        cos, sin = hf_rope(dummy_input, position_ids)

        # Causal attention mask as additive mask: [batch, 1, seq_len, seq_len]
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )

        # Build additive mask: 0 for allowed, -1e9 for disallowed.
        attention_mask = torch.zeros(
            seq_len, seq_len, dtype=torch.float32, device=device
        )
        attention_mask = attention_mask.masked_fill(~causal_mask, float("-1e9"))
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        hf_out = hf_attn(
            hidden_states=hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
        )

        if isinstance(hf_out, tuple):
            hf_out = hf_out[0]

        outputs.append(hf_out.squeeze(0))  # [Li, H]

    return torch.cat(outputs, dim=0)  # [sum(Li), H]


def generate_qwen3_max_outputs(
    sequences: list[torch.Tensor],
    attention_weights: dict[str, torch.Tensor],
    qwen3_config: dict,  # type: ignore[type-arg]
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Generate outputs using MAX Qwen3VL-MoE text attention implementation.

    This builds a small MAX graph that applies Qwen3VLMoEDecoderAttentionWithRope
    to a ragged batch represented by flattened input and input_row_offsets.
    """
    assert isinstance(device, Accelerator), (
        "Qwen3VL-MoE attention tests expect an Accelerator (GPU) device"
    )

    torch_device = torch.device("cuda")

    sequences = [seq.to(torch_device) for seq in sequences]
    flat_input = _flatten_sequences(sequences)  # [T, H]
    seq_lens = [seq.shape[0] for seq in sequences]
    _total_seq_len, hidden_size = flat_input.shape

    text_config = qwen3_config["text_config"]
    n_heads = text_config["num_attention_heads"]
    head_dim = text_config["head_dim"]
    num_kv_heads = text_config["num_key_value_heads"]

    state_dict = {
        weight_name: value.to(max_dtype_to_torch(dtype)).cpu()
        for weight_name, value in attention_weights.items()
    }

    kv_cache_config = KVCacheConfig()
    kv_params = MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=1,
        page_size=kv_cache_config.kv_cache_page_size,
        enable_prefix_caching=kv_cache_config.enable_prefix_caching,
        devices=[DeviceRef.GPU()],
        data_parallel_degree=1,
    )

    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.GPU()

    rope = Qwen3VLTextRotaryEmbedding(
        dim=text_config["hidden_size"],
        n_heads=text_config["num_attention_heads"],
        theta=text_config["rope_theta"],
        max_seq_len=text_config["max_position_embeddings"],
        dtype=DType.bfloat16,
        # Embedding yields pair-wise interleaved ([cos0,sin0,...]); the layer repacks and
        # calls the kernel with interleaved=False (contiguous halves) for HF parity.
        mrope_section=text_config.get("rope_scaling", {}).get(
            "mrope_section", [24, 20, 20]
        ),
        head_dim=text_config["head_dim"],
        interleaved=False,
        scaling_params=None,
    )

    attention = Qwen3VLMoEDecoderAttentionWithRope(
        rope=rope,
        num_attention_heads=n_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=hidden_size,
        kv_params=kv_params,
        devices=[device_ref],
        dtype=dtype,
        linear_cls=Linear,
        has_bias=False,  # Generated weights do not include biases
        rms_norm_eps=text_config.get("rms_norm_eps", 1e-6),
    )
    attention.load_state_dict(state_dict, strict=True)

    kv_manager = load_kv_manager(
        params=kv_params,
        max_batch_size=len(seq_lens),
        max_seq_len=MAX_SEQ_LEN,
        available_cache_memory=30 * 1024 * 1024,
        session=session,
        is_di_enabled=False,
        model_name="FAKE",
    )
    assert isinstance(kv_manager, PagedKVCacheManager)

    input_type = TensorType(
        dtype, [Dim("total_seq_len"), hidden_size], device=device_ref
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=[Dim("row_offsets_len")], device=device_ref
    )

    flattened_kv_types = kv_params.flattened_kv_inputs()

    input_row_offsets = _build_input_row_offsets(seq_lens).to(torch_device)

    with Graph(
        "Qwen3VLMoETextAttention",
        input_types=(
            input_type,
            input_row_offsets_type,
            *flattened_kv_types,
        ),
    ) as graph:
        x, input_row_offsets_input, *kv_cache = graph.inputs

        kv_collection = kv_params.unflatten_kv_inputs(iter(kv_cache)).inputs[0]

        output = attention(
            layer_idx=ops.constant(0, DType.uint32, DeviceRef.CPU()),
            x=x.tensor,
            kv_collection=kv_collection,
            freqs_cis=rope.freqs_cis,
            input_row_offsets=input_row_offsets_input.tensor,
        )
        graph.output(output)

    compiled = session.load(graph, weights_registry=attention.state_dict())

    batch = [
        create_text_context(np.zeros(seq_len, dtype=np.int32))
        for seq_len in seq_lens
    ]
    for context in batch:
        kv_manager.claim(context)
        kv_manager.alloc(context)

    kv_cache_runtime = kv_manager.runtime_inputs_for_leaf([batch]).inputs[0]
    assert kv_cache_runtime.attention_dispatch_metadata is not None

    result = compiled.execute(
        Buffer.from_dlpack(flat_input.to(torch_device)).to(device),
        Buffer.from_dlpack(input_row_offsets.to(torch_device)).to(device),
        *kv_cache_runtime.flatten(),
    )
    max_tensor = result[0]
    return from_dlpack(max_tensor)


@pytest.mark.parametrize("seq_len", [16, 32])
def test_qwen3vl_moe_text_attention_single_sequence(seq_len: int) -> None:
    """HF vs MAX Qwen3VL-MoE text attention on a single contiguous sequence."""
    torch.manual_seed(42)

    config_loader = get_config_loader()

    qwen3_config = config_loader.create_qwen3vl_config(ConfigNames.QWEN3VL_30B)
    text_config = qwen3_config["text_config"]
    hidden_size = text_config["hidden_size"]

    text_attention_weights = generate_text_attention_weights(text_config)

    torch_device = torch.device("cuda")

    # Single sequence [L, H]
    seq = torch.randn(
        seq_len, hidden_size, dtype=torch.bfloat16, device=torch_device
    )
    sequences = [seq]

    torch_output = generate_qwen3_torch_outputs(
        sequences=sequences,
        attention_weights=text_attention_weights,
        text_config=text_config,
        device=torch_device,
    )

    max_output = generate_qwen3_max_outputs(
        sequences=sequences,
        attention_weights=text_attention_weights,
        qwen3_config=qwen3_config,
        dtype=DType.bfloat16,
        device=Accelerator(),
    )

    expected_shape = (seq_len, hidden_size)
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )

    # Compare only the last token (decode-style behavior).
    torch_last = torch_output[-1:, :]  # [1, H]
    max_last = max_output[-1:, :]  # [1, H]

    assert_tensors_close(
        torch_last,
        max_last,
        rtol=DECODE_RTOL,
        atol=DECODE_ATOL,
        message="Qwen3VL-MoE text attention last-token outputs do not match",
    )

    # Basic sanity: no NaNs/Infs anywhere in MAX output
    assert torch.all(torch.isfinite(max_output)), (
        "MAX output contains NaNs or infs"
    )

    del seq, sequences, torch_output, max_output, text_attention_weights
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "seq_lens",
    [
        [16],  # single short sequence
        [32],  # single longer sequence
        [8, 24],  # two sequences, different lengths
        [4, 4, 4, 4],  # multiple short sequences
    ],
)
def test_qwen3vl_moe_text_attention_ragged_smoke(seq_lens: list[int]) -> None:
    """Smoke test for ragged Qwen3VL-MoE attention on MAX.

    We only check that the MAX implementation runs on ragged batches, returns
    the right shape, and produces finite values. Numerical equality to HF is
    covered in the simpler single-sequence test above.
    """
    torch.manual_seed(42)

    config_loader = get_config_loader()
    qwen3_config = config_loader.create_qwen3vl_config(ConfigNames.QWEN3VL_30B)
    text_config = qwen3_config["text_config"]
    hidden_size = text_config["hidden_size"]

    text_attention_weights = generate_text_attention_weights(text_config)

    torch_device = torch.device("cuda")
    sequences = [
        torch.randn(L, hidden_size, dtype=torch.bfloat16, device=torch_device)
        for L in seq_lens
    ]

    max_output = generate_qwen3_max_outputs(
        sequences=sequences,
        attention_weights=text_attention_weights,
        qwen3_config=qwen3_config,
        dtype=DType.bfloat16,
        device=Accelerator(),
    )

    expected_shape = (sum(seq_lens), hidden_size)
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )
    assert torch.all(torch.isfinite(max_output)), (
        "MAX output contains NaNs or infs"
    )

    del sequences, max_output, text_attention_weights
    torch.cuda.empty_cache()


@pytest.mark.parametrize("seq_len", [16])
def test_qwen3vl_moe_text_attention_single_sequence_full(seq_len: int) -> None:
    """HF vs MAX full-sequence comparison in prefill mode.

    Both HF and MAX see the entire sequence in a single call; this avoids
    having to manually manage KV cache lengths in the test while still
    validating the attention math end-to-end.
    """
    torch.manual_seed(42)

    config_loader = get_config_loader()
    qwen3_config = config_loader.create_qwen3vl_config(ConfigNames.QWEN3VL_30B)
    text_config = qwen3_config["text_config"]
    hidden_size = text_config["hidden_size"]

    text_attention_weights = generate_text_attention_weights(text_config)
    torch_device = torch.device("cuda")

    # Single sequence [L, H]
    seq = torch.randn(
        seq_len, hidden_size, dtype=torch.bfloat16, device=torch_device
    )
    sequences = [seq]

    torch_output = generate_qwen3_torch_outputs(
        sequences=sequences,
        attention_weights=text_attention_weights,
        text_config=text_config,
        device=torch_device,
    )

    max_output = generate_qwen3_max_outputs(
        sequences=sequences,
        attention_weights=text_attention_weights,
        qwen3_config=qwen3_config,
        dtype=DType.bfloat16,
        device=Accelerator(),
    )

    assert max_output.shape == torch_output.shape

    assert_tensors_close(
        torch_output,
        max_output,
        rtol=DECODE_RTOL,
        atol=DECODE_ATOL,
        message="Qwen3VL-MoE full-sequence prefill outputs do not match",
    )

    assert torch.all(torch.isfinite(max_output)), (
        "MAX full-sequence output contains NaNs or infs"
    )
