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

"""Gemma4Attention harness for the model test bed.

Supports benchmarking of the Gemma4Attention layer used by Gemma4 models.
No HuggingFace Gemma4 model is available yet, so correctness testing is
not supported — torch_reference_layer and prepare_torch_inputs raise
NotImplementedError.

Static params:
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    n_global_kv_heads: int (default same as n_kv_heads)
    head_dim: int
    global_head_dim: int (default same as head_dim)
    max_seq_len: int
    rope_theta: float (default 1000000.0)
    qk_norm_eps: float (default 1e-6)
    local_window_size: int (default 1024)
    is_sliding: bool (default True)
    attention_k_eq_v: bool (default False)
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
from max.pipelines.architectures.gemma4.layers.attention import Gemma4Attention

from testbed.harnesses.ragged_attention_harness import (
    AttentionDynamicParams,
    AttentionStaticParams,
    RaggedAttentionHarness,
)
from testbed.registry import register_harness


@dataclass
class Gemma4AttentionStaticParams(AttentionStaticParams):
    rope_theta: float = 1000000.0
    qk_norm_eps: float = 1e-6
    local_window_size: int = 1024
    is_sliding: bool = True
    attention_k_eq_v: bool = False
    n_global_kv_heads: int | None = None
    global_head_dim: int | None = None


@register_harness("gemma4_attention")
class Gemma4AttentionHarness(
    RaggedAttentionHarness[Gemma4AttentionStaticParams]
):
    """Harness for Gemma4Attention layers.

    Benchmarking only — no HuggingFace reference model is available for
    correctness testing.
    """

    @staticmethod
    def static_params_type() -> type:
        return Gemma4AttentionStaticParams

    @property
    def name(self) -> str:
        is_sliding = self.static_params.is_sliding
        suffix = "local" if is_sliding else "global"
        return f"gemma4_attention_{suffix}"

    def build_graph(self) -> tuple[Graph, dict[str, DLPackArray]]:
        p = self.static_params
        hidden_size = p.hidden_size
        n_heads = p.n_heads
        n_kv_heads = p.n_kv_heads
        n_global_kv_heads = p.n_global_kv_heads or n_kv_heads
        head_dim = p.head_dim
        global_head_dim = p.global_head_dim or head_dim
        max_seq_len = p.max_seq_len
        rope_theta = p.rope_theta
        qk_norm_eps = p.qk_norm_eps
        local_window_size = p.local_window_size
        is_sliding = p.is_sliding
        attention_k_eq_v = p.attention_k_eq_v

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
            head_dim=global_head_dim,
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

        layer = Gemma4Attention(
            rope_global=rope_global,
            rope_local=rope_local,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            num_global_key_value_heads=n_global_kv_heads,
            attention_k_eq_v=attention_k_eq_v,
            hidden_size=hidden_size,
            kv_params=kv_params,
            global_head_dim=global_head_dim,
            layer_idx=0,
            layer_idx_in_cache=0,
            is_sliding=is_sliding,
            dtype=DType.bfloat16,
            devices=[DeviceRef.GPU()],
            has_bias=False,
            qk_norm_eps=qk_norm_eps,
            local_window_size=local_window_size,
        )

        std = 0.02
        q_dim = n_heads * head_dim
        # kv_dim depends on sliding vs global
        effective_kv_heads = n_kv_heads if is_sliding else n_global_kv_heads
        kv_dim = effective_kv_heads * head_dim
        has_v_proj = not (attention_k_eq_v and not is_sliding)

        weights: dict[str, torch.Tensor] = {}
        if has_v_proj:
            weights["q_proj.weight"] = (
                torch.randn(q_dim, hidden_size, dtype=torch.bfloat16) * std
            )
            weights["k_proj.weight"] = (
                torch.randn(kv_dim, hidden_size, dtype=torch.bfloat16) * std
            )
            weights["v_proj.weight"] = (
                torch.randn(kv_dim, hidden_size, dtype=torch.bfloat16) * std
            )
        else:
            weights["q_proj.weight"] = (
                torch.randn(q_dim, hidden_size, dtype=torch.bfloat16) * std
            )
            weights["k_proj.weight"] = (
                torch.randn(kv_dim, hidden_size, dtype=torch.bfloat16) * std
            )

        weights["o_proj.weight"] = (
            torch.randn(hidden_size, q_dim, dtype=torch.bfloat16) * std
        )
        # Per-head QK norm weights.
        weights["q_norm.weight"] = (
            torch.randn(head_dim, dtype=torch.bfloat16) * 0.5
        )
        weights["k_norm.weight"] = (
            torch.randn(head_dim, dtype=torch.bfloat16) * 0.5
        )

        layer.load_state_dict(weights)

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
            "Gemma4Attention",
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

    def torch_reference_layer(
        self, device: str = "cuda"
    ) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError(
            "No HuggingFace Gemma4 model available for correctness testing"
        )

    def prepare_torch_inputs(
        self,
        execute_args: list[Any],
        dynamic_params: AttentionDynamicParams,
        device: str = "cuda",
    ) -> list[Any]:
        raise NotImplementedError(
            "No HuggingFace Gemma4 model available for correctness testing"
        )
