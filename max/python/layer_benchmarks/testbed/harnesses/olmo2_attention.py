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

"""Olmo2Attention harness for the model test bed.

Supports benchmarking and correctness testing of the Olmo2Attention layer
used by OLMo2 models (e.g. allenai/OLMo-2-7B). Correctness testing compares
against HuggingFace's Olmo2Attention implementation.

Static params:
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float (default 500000.0)
    rms_norm_eps: float (default 1e-5)
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
from max.pipelines.architectures.olmo2.layers.attention import Olmo2Attention
from transformers.models.olmo2.configuration_olmo2 import Olmo2Config
from transformers.models.olmo2.modeling_olmo2 import (
    Olmo2Attention as HFOlmo2Attention,
)
from transformers.models.olmo2.modeling_olmo2 import Olmo2RotaryEmbedding

from testbed.harnesses.ragged_attention_harness import (
    AttentionDynamicParams,
    AttentionStaticParams,
    RaggedAttentionHarness,
)
from testbed.registry import register_harness


@dataclass
class Olmo2AttentionStaticParams(AttentionStaticParams):
    rms_norm_eps: float = 1e-5


@register_harness("olmo2_attention")
class Olmo2AttentionHarness(RaggedAttentionHarness[Olmo2AttentionStaticParams]):
    """Harness for Olmo2Attention layers.

    Supports correctness testing by comparing against HuggingFace's
    Olmo2Attention. Correctness is only supported for prefill shapes
    (ctx_len=0) with batch_size=1.
    """

    @staticmethod
    def static_params_type() -> type:
        return Olmo2AttentionStaticParams

    @property
    def name(self) -> str:
        return "olmo2_attention"

    def build_graph(self) -> tuple[Graph, dict[str, DLPackArray]]:
        p = self.static_params
        hidden_size = p.hidden_size
        n_heads = p.n_heads
        n_kv_heads = p.n_kv_heads
        head_dim = p.head_dim
        max_seq_len = p.max_seq_len
        rope_theta = p.rope_theta
        rms_norm_eps = p.rms_norm_eps

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
            interleaved=False,
        )

        layer = Olmo2Attention(
            rope=rope,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            layer_idx=0,
            dtype=DType.bfloat16,
            devices=[DeviceRef.GPU()],
            has_bias=False,
            rms_norm_eps=rms_norm_eps,
        )

        # Per-projection STDs and means from OLMo-2-7B checkpoint for
        # realistic weight distributions.
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim

        weights: dict[str, torch.Tensor] = {
            "q_proj.weight": torch.randn(
                q_dim, hidden_size, dtype=torch.bfloat16
            )
            * 0.0155,
            "k_proj.weight": torch.randn(
                kv_dim, hidden_size, dtype=torch.bfloat16
            )
            * 0.0150,
            "v_proj.weight": torch.randn(
                kv_dim, hidden_size, dtype=torch.bfloat16
            )
            * 0.0188,
            "o_proj.weight": torch.randn(
                hidden_size, q_dim, dtype=torch.bfloat16
            )
            * 0.0185,
            # QK norm weights (RMSNorm on full projection dimension)
            # with realistic mean + std from checkpoint.
            "q_norm.weight": torch.randn(q_dim, dtype=torch.bfloat16) * 0.400
            + 0.678,
            "k_norm.weight": torch.randn(kv_dim, dtype=torch.bfloat16) * 0.402
            + 0.661,
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
            "Olmo2Attention",
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
    ) -> tuple[HFOlmo2Attention, dict[str, Any]]:
        p = self.static_params
        if not hasattr(self, "_torch_weights"):
            raise RuntimeError(
                "build_and_compile must be called before torch_reference_layer"
            )

        config = Olmo2Config(
            hidden_size=p.hidden_size,
            num_attention_heads=p.n_heads,
            num_key_value_heads=p.n_kv_heads,
            head_dim=p.head_dim,
            max_position_embeddings=p.max_seq_len,
            rope_theta=p.rope_theta,
            attention_bias=False,
            rms_norm_eps=p.rms_norm_eps,
            attn_implementation="eager",
            # Provide required fields that won't affect attention.
            vocab_size=50304,
            num_hidden_layers=1,
            intermediate_size=p.hidden_size,
        )

        layer = HFOlmo2Attention(config, layer_idx=0).to(
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

        config = Olmo2Config(
            hidden_size=p.hidden_size,
            num_attention_heads=p.n_heads,
            num_key_value_heads=p.n_kv_heads,
            head_dim=p.head_dim,
            max_position_embeddings=p.max_seq_len,
            rope_theta=p.rope_theta,
            rms_norm_eps=p.rms_norm_eps,
        )
        rotary_emb = Olmo2RotaryEmbedding(config=config, device=device)
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
