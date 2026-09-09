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

"""Qwen3Attention harness for the model test bed.

Supports benchmarking and correctness testing of the Qwen3Attention layer
used by Qwen3 models (e.g. Qwen/Qwen3-8B). Correctness testing compares
against HuggingFace's Qwen3Attention implementation.

Static params:
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float (default 1000000.0)
    qk_norm_eps: float (default 1e-6)
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
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.rotary_embedding import RotaryEmbedding
from max.pipelines.architectures.qwen3.layers.attention import Qwen3Attention
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention as HFQwen3Attention,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from testbed.harnesses.ragged_attention_harness import (
    AttentionDynamicParams,
    AttentionStaticParams,
    RaggedAttentionHarness,
)
from testbed.registry import register_harness


@dataclass
class Qwen3AttentionStaticParams(AttentionStaticParams):
    rope_theta: float = 1000000.0
    qk_norm_eps: float = 1e-6


@register_harness("qwen3_attention")
class Qwen3AttentionHarness(RaggedAttentionHarness[Qwen3AttentionStaticParams]):
    """Harness for Qwen3Attention layers.

    Supports correctness testing by comparing against HuggingFace's
    Qwen3Attention. Correctness is only supported for prefill shapes
    (ctx_len=0) with batch_size=1.
    """

    @staticmethod
    def static_params_type() -> type:
        return Qwen3AttentionStaticParams

    @property
    def name(self) -> str:
        return "qwen3_attention"

    def build_graph(self) -> tuple[Graph, dict[str, DLPackArray]]:
        p = self.static_params
        hidden_size = p.hidden_size
        n_heads = p.n_heads
        n_kv_heads = p.n_kv_heads
        head_dim = p.head_dim
        max_seq_len = p.max_seq_len
        rope_theta = p.rope_theta
        qk_norm_eps = p.qk_norm_eps

        kv_params = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_layers=1,
            devices=[DeviceRef.GPU()],
        )

        rope = RotaryEmbedding(
            dim=hidden_size,
            n_heads=n_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            interleaved=False,
        )

        layer = Qwen3Attention(
            rope=rope,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            layer_idx=0,
            dtype=DType.bfloat16,
            devices=[DeviceRef.GPU()],
            has_bias=False,
            qk_norm_eps=qk_norm_eps,
        )

        # Per-projection STDs from Qwen3-1.7B checkpoint for realistic
        # weight distributions.
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim

        weights: dict[str, torch.Tensor] = {
            "q_proj.weight": torch.randn(
                q_dim, hidden_size, dtype=torch.bfloat16
            )
            * 0.0344,
            "k_proj.weight": torch.randn(
                kv_dim, hidden_size, dtype=torch.bfloat16
            )
            * 0.0317,
            "v_proj.weight": torch.randn(
                kv_dim, hidden_size, dtype=torch.bfloat16
            )
            * 0.0316,
            "o_proj.weight": torch.randn(
                hidden_size, q_dim, dtype=torch.bfloat16
            )
            * 0.0355,
            "q_norm.weight": torch.randn(head_dim, dtype=torch.bfloat16)
            * 1.859,
            "k_norm.weight": torch.randn(head_dim, dtype=torch.bfloat16)
            * 0.789,
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
            "Qwen3Attention",
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
            layer_idx_const = ops.constant(
                0, DType.uint32, device=DeviceRef.CPU()
            )
            graph.output(
                layer(
                    layer_idx_const,
                    inputs.tensor,
                    kv_collection,
                    freqs_cis=rope.freqs_cis,
                    input_row_offsets=input_row_offsets.tensor,
                )
            )

        return graph, layer.state_dict()

    # ------------------------------------------------------------------ #
    # Correctness support (prefill with ctx_len=0, batch_size=1)
    # ------------------------------------------------------------------ #

    def torch_reference_layer(
        self, device: str = "cuda"
    ) -> tuple[HFQwen3Attention, dict[str, Any]]:
        p = self.static_params
        if not hasattr(self, "_torch_weights"):
            raise RuntimeError(
                "build_and_compile must be called before torch_reference_layer"
            )

        config = Qwen3Config(
            hidden_size=p.hidden_size,
            num_attention_heads=p.n_heads,
            num_key_value_heads=p.n_kv_heads,
            head_dim=p.head_dim,
            max_position_embeddings=p.max_seq_len,
            rope_theta=p.rope_theta,
            rms_norm_eps=p.qk_norm_eps,
            attn_implementation="eager",
            # Required but won't affect attention.
            vocab_size=151936,
            num_hidden_layers=1,
            intermediate_size=p.hidden_size,
        )

        layer = HFQwen3Attention(config, layer_idx=0).to(
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

        config = Qwen3Config(
            hidden_size=p.hidden_size,
            num_attention_heads=p.n_heads,
            num_key_value_heads=p.n_kv_heads,
            head_dim=p.head_dim,
            max_position_embeddings=p.max_seq_len,
            rope_theta=p.rope_theta,
            rms_norm_eps=p.qk_norm_eps,
        )
        rotary_emb = Qwen3RotaryEmbedding(config=config, device=device)
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
