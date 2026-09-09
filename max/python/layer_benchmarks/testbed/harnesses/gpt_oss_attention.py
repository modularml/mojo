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

"""GptOssAttention harness for the model test bed.

Supports benchmarking and correctness testing of the GptOssAttention layer
used by GPT-OSS models (e.g. openai/gpt-oss-120b). Correctness testing
compares against HuggingFace's GptOssAttention implementation.

Static params:
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float (default 150000.0)
    has_bias: bool (default True)
    layer_type: str (default "full_attention")
    local_window_size: int (default 128)
    rope_factor: float (default 32.0)
    rope_beta_fast: float (default 32.0)
    rope_beta_slow: float (default 1.0)
    rope_original_max_pos: int (default 4096)
    rope_truncate: bool (default False)
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
from max.nn.rotary_embedding import YarnRotaryEmbedding, YarnScalingParams
from max.pipelines.architectures.gpt_oss.layers.attention import GptOssAttention
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention as HFGptOssAttention,
)
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

from testbed.harnesses.ragged_attention_harness import (
    AttentionDynamicParams,
    AttentionStaticParams,
    RaggedAttentionHarness,
)
from testbed.registry import register_harness


@dataclass
class GptOssAttentionStaticParams(AttentionStaticParams):
    has_bias: bool = True
    layer_type: str = "full_attention"
    local_window_size: int = 128
    rope_factor: float = 32.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_original_max_pos: int = 4096
    rope_truncate: bool = False
    rope_theta: float = 150000.0


@register_harness("gpt_oss_attention")
class GptOssAttentionHarness(
    RaggedAttentionHarness[GptOssAttentionStaticParams]
):
    """Harness for GptOssAttention layers.

    Supports correctness testing by comparing against HuggingFace's
    GptOssAttention. Correctness is only supported for prefill shapes
    (ctx_len=0) with batch_size=1 and layer_type="full_attention".
    """

    @staticmethod
    def static_params_type() -> type:
        return GptOssAttentionStaticParams

    @property
    def name(self) -> str:
        layer_type = self.static_params.layer_type
        suffix = "sliding" if layer_type == "sliding_attention" else "full"
        return f"gpt_oss_attention_{suffix}"

    def _yarn_scaling_params(self) -> YarnScalingParams:
        p = self.static_params
        return YarnScalingParams(
            factor=p.rope_factor,
            beta_fast=p.rope_beta_fast,
            beta_slow=p.rope_beta_slow,
            original_max_position_embeddings=p.rope_original_max_pos,
            truncate=p.rope_truncate,
        )

    def build_graph(self) -> tuple[Graph, dict[str, DLPackArray]]:
        p = self.static_params
        hidden_size = p.hidden_size
        n_heads = p.n_heads
        n_kv_heads = p.n_kv_heads
        head_dim = p.head_dim
        max_seq_len = p.max_seq_len
        rope_theta = p.rope_theta
        has_bias = p.has_bias
        layer_type = p.layer_type
        local_window_size = p.local_window_size

        kv_params = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_layers=1,
            devices=[DeviceRef.GPU()],
        )

        rope = YarnRotaryEmbedding(
            dim=hidden_size,
            n_heads=n_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            interleaved=False,
            scaling_params=self._yarn_scaling_params(),
        )

        layer = GptOssAttention(
            rope=rope,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            layer_idx=0,
            layer_type=layer_type,
            dtype=DType.bfloat16,
            devices=[DeviceRef.GPU()],
            has_bias=has_bias,
            local_window_size=local_window_size,
        )

        # Generate random weights with small std for numerical stability.
        std = 0.02
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim

        weights: dict[str, torch.Tensor] = {
            "q_proj.weight": torch.randn(
                q_dim, hidden_size, dtype=torch.bfloat16
            )
            * std,
            "k_proj.weight": torch.randn(
                kv_dim, hidden_size, dtype=torch.bfloat16
            )
            * std,
            "v_proj.weight": torch.randn(
                kv_dim, hidden_size, dtype=torch.bfloat16
            )
            * std,
            "o_proj.weight": torch.randn(
                hidden_size, q_dim, dtype=torch.bfloat16
            )
            * std,
            # Initialize sinks to zeros (no sink effect by default).
            "sinks": torch.zeros(n_heads, dtype=torch.bfloat16),
        }

        if has_bias:
            weights["q_proj.bias"] = (
                torch.randn(q_dim, dtype=torch.bfloat16) * std
            )
            weights["k_proj.bias"] = (
                torch.randn(kv_dim, dtype=torch.bfloat16) * std
            )
            weights["v_proj.bias"] = (
                torch.randn(kv_dim, dtype=torch.bfloat16) * std
            )
            weights["o_proj.bias"] = (
                torch.randn(hidden_size, dtype=torch.bfloat16) * std
            )

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
            "GptOssAttention",
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
    # Correctness support (prefill with ctx_len=0, full_attention only)
    # ------------------------------------------------------------------ #

    def torch_reference_layer(
        self, device: str = "cuda"
    ) -> tuple[HFGptOssAttention, dict[str, Any]]:
        p = self.static_params
        if p.layer_type == "sliding_attention":
            raise NotImplementedError(
                "Correctness testing not yet supported for sliding_attention"
            )
        if not hasattr(self, "_torch_weights"):
            raise RuntimeError(
                "build_and_compile must be called before torch_reference_layer"
            )

        config = GptOssConfig(
            hidden_size=p.hidden_size,
            num_attention_heads=p.n_heads,
            num_key_value_heads=p.n_kv_heads,
            head_dim=p.head_dim,
            max_position_embeddings=p.max_seq_len,
            rope_theta=p.rope_theta,
            attention_bias=p.has_bias,
            sliding_window=p.local_window_size,
            rope_scaling={
                "rope_type": "yarn",
                "factor": p.rope_factor,
                "beta_fast": p.rope_beta_fast,
                "beta_slow": p.rope_beta_slow,
                "original_max_position_embeddings": p.rope_original_max_pos,
                "truncate": p.rope_truncate,
            },
            attn_implementation="eager",
            # Provide required fields that won't affect attention.
            vocab_size=201088,
            num_hidden_layers=1,
            intermediate_size=p.hidden_size,
            layer_types=["full_attention"],
        )

        layer = HFGptOssAttention(config, layer_idx=0).to(
            device=device, dtype=torch.bfloat16
        )

        # Load matching weights.
        for name, param in layer.named_parameters():
            if name in self._torch_weights:
                param.data = self._torch_weights[name].to(
                    device=device, dtype=torch.bfloat16
                )

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

        config = GptOssConfig(
            hidden_size=p.hidden_size,
            num_attention_heads=p.n_heads,
            num_key_value_heads=p.n_kv_heads,
            head_dim=p.head_dim,
            max_position_embeddings=p.max_seq_len,
            rope_theta=p.rope_theta,
            rope_scaling={
                "rope_type": "yarn",
                "factor": p.rope_factor,
                "beta_fast": p.rope_beta_fast,
                "beta_slow": p.rope_beta_slow,
                "original_max_position_embeddings": p.rope_original_max_pos,
                "truncate": p.rope_truncate,
            },
        )
        rotary_emb = GptOssRotaryEmbedding(config=config, device=device)
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)
        cos, sin = rotary_emb(hidden_states, position_ids)
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
