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

"""Gemma3Attention harness for the model test bed.

Supports benchmarking and correctness testing of the Gemma3Attention layer
used by Gemma3 models (e.g. google/gemma-3-1b-it). Correctness testing
compares against HuggingFace's Gemma3Attention implementation.

Static params:
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float (default 1000000.0)
    qk_norm_eps: float (default 1e-6)
    sliding_window_pattern: int (default 6)
    local_window_size: int (default 1024)
    layer_idx: int (default 0)
    total_num_pages: int (optional, default 8192)

Dynamic params (per shape):
    batch_size: int
    seq_len: int
    ctx_len: int (default 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# torch is a caller-supplied dep, see BUILD.bazel
import torch  # type: ignore[import-not-found]
from max.driver import DLPackArray
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.gemma3.layers.attention import Gemma3Attention
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention as HFGemma3Attention,
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding

from testbed.harnesses.ragged_attention_harness import (
    AttentionDynamicParams,
    AttentionStaticParams,
    RaggedAttentionHarness,
)
from testbed.registry import register_harness


@dataclass
class Gemma3AttentionStaticParams(AttentionStaticParams):
    rope_theta: float = 1000000.0
    qk_norm_eps: float = 1e-6
    sliding_window_pattern: int = 6
    local_window_size: int = 1024
    layer_idx: int = 0


@register_harness("gemma3_attention")
class Gemma3AttentionHarness(
    RaggedAttentionHarness[Gemma3AttentionStaticParams]
):
    """Harness for Gemma3Attention layers.

    Supports correctness testing by comparing against HuggingFace's
    Gemma3Attention. Correctness is only supported for prefill shapes
    (ctx_len=0) with batch_size=1 and global attention layers.
    """

    @staticmethod
    def static_params_type() -> type:
        return Gemma3AttentionStaticParams

    @property
    def name(self) -> str:
        p = self.static_params
        layer_idx = p.layer_idx
        sliding_window_pattern = p.sliding_window_pattern
        is_local = bool((layer_idx + 1) % sliding_window_pattern)
        suffix = "local" if is_local else "global"
        return f"gemma3_attention_{suffix}"

    def build_graph(self) -> tuple[Graph, dict[str, DLPackArray]]:
        p = self.static_params
        hidden_size = p.hidden_size
        n_heads = p.n_heads
        n_kv_heads = p.n_kv_heads
        head_dim = p.head_dim
        max_seq_len = p.max_seq_len
        rope_theta = p.rope_theta
        qk_norm_eps = p.qk_norm_eps
        sliding_window_pattern = p.sliding_window_pattern
        local_window_size = p.local_window_size
        layer_idx = p.layer_idx

        kv_params = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_layers=1,
            devices=[DeviceRef.GPU()],
        )

        rope_global = Llama3RotaryEmbedding(
            dim=hidden_size,
            n_heads=n_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            interleaved=False,
        )
        rope_local = Llama3RotaryEmbedding(
            dim=hidden_size,
            n_heads=n_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            interleaved=False,
        )

        layer = Gemma3Attention(
            rope_global=rope_global,
            rope_local=rope_local,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            layer_idx=layer_idx,
            is_sliding=bool((layer_idx + 1) % sliding_window_pattern),
            dtype=DType.bfloat16,
            devices=[DeviceRef.GPU()],
            has_bias=False,
            qk_norm_eps=qk_norm_eps,
            local_window_size=local_window_size,
        )

        # Per-projection STDs from gemma-3-1b-it checkpoint for realistic
        # weight distributions that exercise production numerical paths.
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim

        weights: dict[str, torch.Tensor] = {
            "q_proj.weight": torch.randn(
                q_dim, hidden_size, dtype=torch.bfloat16
            )
            * 0.0284,
            "k_proj.weight": torch.randn(
                kv_dim, hidden_size, dtype=torch.bfloat16
            )
            * 0.0309,
            "v_proj.weight": torch.randn(
                kv_dim, hidden_size, dtype=torch.bfloat16
            )
            * 0.0309,
            "o_proj.weight": torch.randn(
                hidden_size, q_dim, dtype=torch.bfloat16
            )
            * 0.0237,
            # Gemma3RMSNorm uses weight_offset=1.0, so the effective scale
            # is (1.0 + weight). Initialize with realistic STDs.
            "q_norm.weight": torch.randn(head_dim, dtype=torch.bfloat16) * 0.68,
            "k_norm.weight": torch.randn(head_dim, dtype=torch.bfloat16)
            * 0.793,
        }

        layer.load_state_dict(weights)
        self._torch_weights = weights

        device_ref = DeviceRef.GPU()
        input_type = TensorType(
            dtype=DType.bfloat16,
            shape=["total_seq_len", hidden_size],
            device=device_ref,
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        flattened_kv_types = kv_params.flattened_kv_inputs()

        with Graph(
            "Gemma3Attention",
            input_types=(
                input_type,
                input_row_offsets_type,
                *flattened_kv_types,
            ),
        ) as graph:
            inputs, input_row_offsets, *kv_cache = graph.inputs
            kv_collection = kv_params.unflatten_kv_inputs(
                iter(kv_cache)
            ).inputs[0]
            graph.output(
                layer(
                    inputs.tensor,
                    kv_collection,
                    input_row_offsets=input_row_offsets.tensor,
                )
            )

        return graph, layer.state_dict()

    # ------------------------------------------------------------------ #
    # Correctness support (prefill with ctx_len=0, batch_size=1)
    # ------------------------------------------------------------------ #

    def torch_reference_layer(
        self, device: str = "cuda"
    ) -> tuple[HFGemma3Attention, dict[str, Any]]:
        p = self.static_params
        layer_idx = p.layer_idx
        sliding_window_pattern = p.sliding_window_pattern

        if not hasattr(self, "_torch_weights"):
            raise RuntimeError(
                "build_and_compile must be called before torch_reference_layer"
            )

        # Build layer_types list: every sliding_window_pattern-th layer is
        # global ("full_attention"), the rest are sliding.
        num_layers = layer_idx + 1
        layer_types = []
        for i in range(num_layers):
            if (i + 1) % sliding_window_pattern == 0:
                layer_types.append("full_attention")
            else:
                layer_types.append("sliding_attention")

        config = Gemma3TextConfig(
            hidden_size=p.hidden_size,
            num_attention_heads=p.n_heads,
            num_key_value_heads=p.n_kv_heads,
            head_dim=p.head_dim,
            max_position_embeddings=p.max_seq_len,
            rms_norm_eps=p.qk_norm_eps,
            query_pre_attn_scalar=p.head_dim,
            attn_implementation="eager",
            num_hidden_layers=num_layers,
            layer_types=layer_types,
            rope_theta=p.rope_theta,
            sliding_window=p.local_window_size,
            # Provide required fields that won't affect attention.
            vocab_size=262144,
            intermediate_size=p.hidden_size,
        )

        layer = HFGemma3Attention(config, layer_idx=layer_idx).to(
            device=device, dtype=torch.bfloat16
        )

        # Load matching weights. HF Gemma3Attention uses Gemma3RMSNorm for
        # q_norm/k_norm. Gemma3RMSNorm also applies weight_offset=1.0
        # (effective scale = 1 + weight), so we can load weights directly.
        for name, param in layer.named_parameters():
            if name in self._torch_weights:
                param.data = self._torch_weights[name].to(
                    device=device, dtype=torch.bfloat16
                )

        self._hf_config = config
        return layer

    def prepare_torch_inputs(
        self,
        execute_args: list[Any],
        dynamic_params: AttentionDynamicParams,
        device: str = "cuda",
    ) -> list[Any]:
        p = self.static_params
        seq_len = dynamic_params.seq_len
        ctx_len = dynamic_params.ctx_len
        if ctx_len != 0:
            raise NotImplementedError(
                "Correctness testing only supports prefill (ctx_len=0)"
            )

        hidden_states = (
            torch.from_dlpack(execute_args[0])
            .reshape(1, seq_len, p.hidden_size)
            .to(device=device, dtype=torch.bfloat16)
        )

        layer_idx = p.layer_idx
        swp = p.sliding_window_pattern
        use_local = bool((layer_idx + 1) % swp)
        layer_type = "sliding_attention" if use_local else "full_attention"
        rotary_emb = Gemma3RotaryEmbedding(config=self._hf_config).to(device)
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)
        cos, sin = rotary_emb(
            hidden_states, position_ids, layer_type=layer_type
        )
        position_embeddings = (cos, sin)

        # Causal attention mask.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        attention_mask = torch.zeros(
            1, 1, seq_len, seq_len, dtype=torch.bfloat16, device=device
        )
        attention_mask = attention_mask.masked_fill(
            causal_mask[None, None, :, :],
            torch.finfo(torch.bfloat16).min,
        )

        return [hidden_states, position_embeddings, attention_mask]
