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

"""Fixtures for GLM-5.1 (GlmMoeDsa) layer integration tests."""

from __future__ import annotations

import json
import os
import typing
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from max._core.engine import PrintStyle
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.attention.multi_latent_attention import LatentAttentionWithRope
from max.nn.kv_cache import MLAKVCacheParams
from max.nn.rotary_embedding import RotaryEmbedding
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from test_common.graph_utils import is_b100_b200
from torch.utils.dlpack import from_dlpack
from torch_reference.configuration_glm import GlmMoeDsaConfig

WEIGHT_STDDEV = 0.001


@pytest.fixture(
    params=[
        pytest.param(DType.bfloat16, id="bf16"),
        pytest.param(
            DType.float8_e4m3fn,
            id="fp8",
            marks=pytest.mark.skipif(
                not is_b100_b200(),
                reason="FP8 KV cache only supported on B100/B200 (sm_100)",
            ),
        ),
    ]
)
def kv_dtype(request: pytest.FixtureRequest) -> DType:
    return request.param


def _finalize_config(cfg: GlmMoeDsaConfig) -> None:
    """Recompute derived fields after :meth:`PretrainedConfig.update`."""
    cfg.qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    # HF ``GlmMoeDsaRotaryEmbedding`` reads ``config.head_dim`` (not hidden_size // n_heads).
    cfg.head_dim = cfg.qk_rope_head_dim
    cfg.num_local_experts = cfg.n_routed_experts
    if getattr(cfg, "rope_parameters", None) is None:
        cfg.rope_parameters = {
            "rope_type": "default",
            "rope_theta": getattr(cfg, "rope_theta", 10000.0),
        }
    cfg.rope_theta = cfg.rope_parameters.get("rope_theta", 10000.0)
    if getattr(cfg, "indexer_types", None) is None:
        cfg.indexer_types = ["full"] * cfg.num_hidden_layers


@pytest.fixture
def config() -> GlmMoeDsaConfig:
    cfg = GlmMoeDsaConfig()
    path = os.environ["PIPELINES_TESTDATA"]
    config_path = Path(path) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    cfg.update(data)
    _finalize_config(cfg)
    # Layer tests use the eager MLA path (no flash-mla).
    cfg._attn_implementation = "eager"
    return cfg


import math

from max.dtype import DType
from max.graph import TensorValue, Weight, ops
from max.nn.layer import Module


class NaiveGlmMlaAttention(Module):
    """HF-style dense MLA without flare ``mla_*_graph`` kernels."""

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        rope: RotaryEmbedding,
        *,
        device: DeviceRef,
        dtype: DType = DType.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self.rope = rope
        self.device = device
        self.dtype = dtype

        self.n_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.scale = math.sqrt(1.0 / self.qk_head_dim)

        assert self.q_lora_rank is not None

        self.q_a_proj = Weight(
            "q_a_proj.weight",
            dtype,
            (self.q_lora_rank, self.hidden_size),
            device=device,
        )
        self.q_a_layernorm = Weight(
            "q_a_layernorm.weight",
            dtype,
            (self.q_lora_rank,),
            device=device,
        )
        self.q_b_proj = Weight(
            "q_b_proj.weight",
            dtype,
            (self.n_heads * self.qk_head_dim, self.q_lora_rank),
            device=device,
        )
        self.kv_a_proj_with_mqa = Weight(
            "kv_a_proj_with_mqa.weight",
            dtype,
            (self.kv_lora_rank + self.qk_rope_head_dim, self.hidden_size),
            device=device,
        )
        self.kv_a_layernorm = Weight(
            "kv_a_layernorm.weight",
            dtype,
            (self.kv_lora_rank,),
            device=device,
        )
        self.kv_b_proj = Weight(
            "kv_b_proj.weight",
            dtype,
            (
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                self.kv_lora_rank,
            ),
            device=device,
        )
        self.o_proj = Weight(
            "o_proj.weight",
            dtype,
            (self.hidden_size, self.n_heads * self.v_head_dim),
            device=device,
        )

    def _rms_norm(self, x: TensorValue, gamma: TensorValue) -> TensorValue:
        weight = gamma.cast(x.dtype).to(x.device)
        return ops.rms_norm(
            x,
            weight,
            epsilon=1e-6,
            weight_offset=0.0,
            multiply_before_cast=False,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
    ) -> TensorValue:
        """Forward pass with padded ``[batch, seq, hidden]`` activations."""
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        q = hidden_states @ ops.transpose(self.q_a_proj, 0, 1)
        q = self._rms_norm(q, self.q_a_layernorm)
        q = q @ ops.transpose(self.q_b_proj, 0, 1)
        q = ops.reshape(
            q, (batch_size, seq_len, self.n_heads, self.qk_head_dim)
        )

        q_nope, q_pe = ops.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1
        )
        q_pe = self.rope(q_pe)
        q = ops.concat((q_nope, q_pe), axis=-1)
        q = ops.transpose(q, 1, 2)

        compressed_kv = hidden_states @ ops.transpose(
            self.kv_a_proj_with_mqa, 0, 1
        )
        k_compressed, k_pe = ops.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            axis=-1,
        )
        k_compressed = self._rms_norm(k_compressed, self.kv_a_layernorm)

        kv_expanded = k_compressed @ ops.transpose(self.kv_b_proj, 0, 1)
        kv_expanded = ops.reshape(
            kv_expanded,
            (
                batch_size,
                seq_len,
                self.n_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ),
        )
        k_nope, v = ops.split(
            kv_expanded,
            [self.qk_nope_head_dim, self.v_head_dim],
            axis=-1,
        )

        k_pe = ops.reshape(
            k_pe, (batch_size, seq_len, 1, self.qk_rope_head_dim)
        )
        k_pe = self.rope(k_pe)
        k_pe = ops.broadcast_to(
            k_pe, (batch_size, seq_len, self.n_heads, self.qk_rope_head_dim)
        )

        k = ops.concat((k_nope, k_pe), axis=-1)
        k = ops.transpose(k, 1, 2)
        v = ops.transpose(v, 1, 2)

        scores = q @ ops.transpose(k, 2, 3)
        scores = ops.cast(scores, DType.float32)
        scores = scores * ops.constant(
            self.scale, dtype=DType.float32, device=scores.device
        )

        mask = attention_mask
        if mask.rank == 4:
            mask = mask[0, 0]
        elif mask.rank == 3:
            mask = mask[0]
        scores = scores + ops.cast(mask, DType.float32).to(scores.device)

        attn_probs = ops.softmax(scores, axis=-1)
        attn_probs = ops.cast(attn_probs, q.dtype)
        attn_out = attn_probs @ v

        attn_out = ops.transpose(attn_out, 1, 2)
        attn_out = ops.reshape(
            attn_out, (batch_size, seq_len, self.n_heads * self.v_head_dim)
        )
        return attn_out @ ops.transpose(self.o_proj, 0, 1)


def _generate_mla_max_outputs_naive(
    config: GlmMoeDsaConfig,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    use_prefill: bool = True,
    kv_dtype: DType = DType.bfloat16,
) -> torch.Tensor:
    """Dense MLA via graph matmul/softmax (``NaiveGlmMlaAttention``).

    ``use_prefill`` is accepted for API compatibility; both modes run a full
    sequence forward to match the torch eager reference.
    """
    del use_prefill, kv_dtype

    mla_weights = {
        k: v
        for k, v in attention_weights.items()
        if not k.startswith("indexer.")
    }

    device0 = Accelerator(0)
    device_ref = DeviceRef.GPU()
    session = InferenceSession(devices=[device0])
    session.set_debug_print_options(style=PrintStyle.COMPACT)

    rope = RotaryEmbedding(
        dim=config.qk_rope_head_dim,
        n_heads=config.num_attention_heads,
        theta=config.rope_theta,
        max_seq_len=config.max_position_embeddings,
        head_dim=config.qk_rope_head_dim,
        interleaved=False,
    )
    mla = NaiveGlmMlaAttention(config, rope, device=device_ref)
    mla.load_state_dict(mla_weights)

    batch_size, seq_len, hidden_size = input_tensor.shape
    hidden_type = TensorType(
        DType.bfloat16, [batch_size, seq_len, hidden_size], device_ref
    )
    mask_type = TensorType(
        DType.bfloat16,
        list(attention_mask.shape),
        device_ref,
    )

    graph = Graph(
        "NaiveGlmMlaAttention",
        mla,
        input_types=[hidden_type, mask_type],
    )
    compiled = session.load(graph, weights_registry=mla.state_dict())

    hidden_device = (
        Buffer.from_numpy(input_tensor.view(torch.float16).numpy())
        .view(DType.bfloat16)
        .to(device0)
    )
    mask_device = (
        Buffer.from_numpy(attention_mask.view(torch.float16).numpy())
        .view(DType.bfloat16)
        .to(device0)
    )
    max_output = compiled.execute(hidden_device, mask_device)
    return from_dlpack(max_output[0]).to(torch.bfloat16).to("cpu")


def _mla_kv_head_dim(config: GlmMoeDsaConfig) -> int:
    """KV cache head dim for MLA; flare decode requires 576 (512 + 64 on GLM)."""
    head_dim = config.kv_lora_rank + config.qk_rope_head_dim
    assert head_dim == 576, (
        f"GPU MLA kernels require head_dim=576, got {head_dim} "
        f"(kv_lora_rank={config.kv_lora_rank}, qk_rope_head_dim={config.qk_rope_head_dim})"
    )
    return head_dim


def _generate_mla_max_outputs(
    config: GlmMoeDsaConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    use_prefill: bool = True,
    prefill_buffer_size: int = 4096,
    kv_dtype: DType = DType.bfloat16,
) -> torch.Tensor:
    attention_weights = {k: v for k, v in attention_weights.items()}
    device0 = Accelerator(0)
    session = InferenceSession(devices=[device0])
    session.set_debug_print_options(style=PrintStyle.COMPACT)

    # HF reference uses split-half RoPE (``rotate_half``), not interleaved layout.
    rope = RotaryEmbedding(
        dim=config.qk_rope_head_dim,
        n_heads=config.num_attention_heads,
        theta=config.rope_theta,
        max_seq_len=config.max_position_embeddings,
        head_dim=config.qk_rope_head_dim,
        interleaved=False,
    )

    kv_params = MLAKVCacheParams(
        dtype=kv_dtype,
        head_dim=_mla_kv_head_dim(config),
        num_layers=config.num_hidden_layers,
        devices=[DeviceRef.GPU()],
        page_size=128,
        num_q_heads=config.num_attention_heads,
    )

    mla_weights = {
        k: v
        for k, v in attention_weights.items()
        if not k.startswith("indexer.")
    }
    mla = LatentAttentionWithRope(
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
        graph_mode="auto" if use_prefill else "decode",
    )
    mla.load_state_dict(mla_weights)

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
            result = mla(
                ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
                hidden_states,
                kv_collection,
                freqs_cis=rope.freqs_cis,
                input_row_offsets=input_row_offsets,
            )
            graph.output(result)
        return graph

    g = construct()
    compiled = session.load(g, weights_registry=mla.state_dict())
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
def generate_mla_max_outputs(
    kv_dtype: DType,
) -> typing.Callable[..., torch.Tensor]:
    return partial(_generate_mla_max_outputs_naive, kv_dtype=kv_dtype)


@pytest.fixture
def input_tensor(
    config: GlmMoeDsaConfig,
    seq_len: int = 7,
    batch_size: int = 1,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(
        batch_size,
        seq_len,
        config.hidden_size,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def input_tensor_rope(
    config: GlmMoeDsaConfig,
    seq_len: int = 7,
    batch_size: int = 1,
    seed: int = 1234,
) -> torch.Tensor:
    torch.manual_seed(seed)
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
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    causal_mask = torch.zeros(
        1, batch_size, seq_len, seq_len, dtype=torch.bfloat16
    )
    causal_mask.masked_fill_(mask, float("-inf")).to(torch.bfloat16)
    return causal_mask


@pytest.fixture
def gate_weight(config: GlmMoeDsaConfig, seed: int = 1234) -> torch.Tensor:
    torch.manual_seed(seed)
    n_experts = config.n_routed_experts or 16
    return (
        torch.randn(n_experts, config.hidden_size, dtype=torch.bfloat16)
        * WEIGHT_STDDEV
    )


@pytest.fixture
def gate_up_proj(config: GlmMoeDsaConfig) -> torch.Tensor:
    """Fused gate+up weights for :class:`~torch_reference.modeling_glm.GlmMoeDsaNaiveMoe`."""
    n_experts = config.n_routed_experts or 16
    fused_dim = 2 * config.moe_intermediate_size
    weights = []
    for i in range(n_experts):
        torch.manual_seed(i)
        weights.append(
            torch.randn(
                fused_dim,
                config.hidden_size,
                dtype=torch.bfloat16,
            )
            * WEIGHT_STDDEV
        )
    return torch.stack(weights, dim=0)


@pytest.fixture
def down_proj(config: GlmMoeDsaConfig) -> torch.Tensor:
    n_experts = config.n_routed_experts or 16
    weights = []
    for i in range(n_experts):
        torch.manual_seed(i + 1000)
        weights.append(
            torch.randn(
                config.hidden_size,
                config.moe_intermediate_size,
                dtype=torch.bfloat16,
            )
            * WEIGHT_STDDEV
        )
    return torch.stack(weights, dim=0)


@pytest.fixture
def shared_expert_weights(config: GlmMoeDsaConfig) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    assert isinstance(config.moe_intermediate_size, int)
    assert isinstance(config.n_shared_experts, int)
    shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
    return {
        "down_proj.weight": torch.randn(
            config.hidden_size,
            shared_intermediate,
            dtype=torch.bfloat16,
        )
        * WEIGHT_STDDEV,
        "gate_proj.weight": torch.randn(
            shared_intermediate,
            config.hidden_size,
            dtype=torch.bfloat16,
        )
        * WEIGHT_STDDEV,
        "up_proj.weight": torch.randn(
            shared_intermediate,
            config.hidden_size,
            dtype=torch.bfloat16,
        )
        * WEIGHT_STDDEV,
    }


@pytest.fixture
def expert_weights(config: GlmMoeDsaConfig) -> list[dict[str, torch.Tensor]]:
    experts = []
    n_experts = (
        config.n_routed_experts if config.n_routed_experts is not None else 16
    )
    for i in range(n_experts):
        torch.manual_seed(i)
        experts.append(
            {
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
        )
    return experts


@pytest.fixture
def attention_weights(config: GlmMoeDsaConfig) -> dict[str, torch.Tensor]:
    """Dummy weights for SparseLatentAttentionWithRope + Indexer."""
    torch.manual_seed(42)
    weight_scale = 192.0
    weights: dict[str, torch.Tensor] = {}

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
    weights["o_proj.weight"] = (
        torch.randn(
            config.hidden_size,
            config.num_attention_heads * config.v_head_dim,
            dtype=torch.bfloat16,
        )
        / weight_scale
    )

    weights["indexer.wq_b.weight"] = (
        torch.randn(
            config.index_n_heads * config.index_head_dim,
            config.q_lora_rank,
            dtype=torch.bfloat16,
        )
        / weight_scale
    )
    weights["indexer.wk.weight"] = (
        torch.randn(
            config.index_head_dim,
            config.hidden_size,
            dtype=torch.bfloat16,
        )
        / weight_scale
    )
    weights["indexer.k_norm.weight"] = torch.ones(
        config.index_head_dim, dtype=torch.bfloat16
    )
    weights["indexer.weights_proj.weight"] = (
        torch.randn(
            config.index_n_heads,
            config.hidden_size,
            dtype=torch.bfloat16,
        )
        / weight_scale
    )

    return weights
