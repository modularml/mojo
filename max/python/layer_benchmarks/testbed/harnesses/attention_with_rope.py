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

"""Unified AttentionWithRope harness supporting bf16, fp8, and fp4 dtypes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# torch is a caller-supplied dep, see BUILD.bazel
import torch  # type: ignore[import-not-found]
from max.driver import Buffer, DLPackArray
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.nn import AttentionWithRope, RotaryEmbedding
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaRotaryEmbedding,
)

from testbed.harnesses.ragged_attention_harness import (
    AttentionDynamicParams,
    AttentionStaticParams,
    RaggedAttentionHarness,
)
from testbed.registry import register_harness

_VARIANTS = {"bf16", "fp8", "fp4"}


def wrap_fp8(tensor: torch.Tensor, name: str) -> WeightData:
    """Wrap a float8_e4m3fn torch tensor as a MAX WeightData."""
    return WeightData(
        Buffer.from_dlpack(tensor.cpu().view(torch.uint8)).view(
            DType.float8_e4m3fn
        ),
        name,
        DType.float8_e4m3fn,
        Shape(tensor.shape),
    )


def _make_fp8_quant_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.COLWISE,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float8_e4m3fn,
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.ROWWISE,
            dtype=DType.float32,
        ),
        format=QuantFormat.FBGEMM_FP8,
        mlp_quantized_layers={0},
        attn_quantized_layers={0},
        embedding_output_dtype=DType.bfloat16,
    )


def _make_nvfp4_quant_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, 16),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=(1, 16),
        ),
        format=QuantFormat.NVFP4,
        mlp_quantized_layers={0},
        attn_quantized_layers={0},
        embedding_output_dtype=DType.bfloat16,
    )


def _make_fp8_weights(name: str, out_dim: int, in_dim: int) -> dict[str, Any]:
    weight = torch.randn(out_dim, in_dim, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    return {
        f"{name}.weight": wrap_fp8(weight, f"{name}.weight"),
        f"{name}.weight_scale": torch.rand((out_dim, 1), dtype=torch.float32)
        * 1e-3,
    }


def _make_fp4_weights(name: str, out_dim: int, in_dim: int) -> dict[str, Any]:
    packed_k = in_dim // 2
    ws_cols = math.ceil(packed_k / 8)
    weight = torch.randint(0, 255, (out_dim, packed_k), dtype=torch.uint8)
    ws = (torch.rand(out_dim, ws_cols, dtype=torch.float32) * 100 + 50).to(
        torch.float8_e4m3fn
    )
    return {
        f"{name}.weight": weight,
        f"{name}.weight_scale": wrap_fp8(ws, f"{name}.weight_scale"),
        f"{name}.weight_scale_2": torch.rand((), dtype=torch.float32) * 1e-4,
        f"{name}.input_scale": torch.rand((), dtype=torch.float32) * 1e-3,
    }


@dataclass
class AttentionWithRopeStaticParams(AttentionStaticParams):
    dtype: str = "bf16"
    _fuse_rope_and_store: bool = True


@register_harness("attention_with_rope")
class AttentionWithRopeHarness(
    RaggedAttentionHarness[AttentionWithRopeStaticParams]
):
    """Unified harness for AttentionWithRope layers (bf16, fp8, fp4).

    Supports correctness testing for bf16 dtype by comparing against
    HuggingFace's LlamaAttention. Correctness is only supported for
    prefill shapes (ctx_len=0) with batch_size=1.

    Static params:
        hidden_size: int
        n_heads: int
        n_kv_heads: int
        head_dim: int
        max_seq_len: int
        rope_theta: float (default 500000.0)
        dtype: str — "bf16" (default), "fp8", or "fp4"
        total_num_pages: int (optional, default 1024)

    Dynamic params (per shape):
        batch_size: int
        seq_len: int
        ctx_len: int (default 0)
    """

    @staticmethod
    def static_params_type() -> type:
        return AttentionWithRopeStaticParams

    @property
    def name(self) -> str:
        return f"attention_with_rope_{self.static_params.dtype}"

    def build_graph(self) -> tuple[Graph, dict[str, DLPackArray]]:
        p = self.static_params
        dtype = p.dtype
        if dtype not in _VARIANTS:
            raise ValueError(
                f"Unknown attention dtype '{dtype}', "
                f"expected one of {_VARIANTS}"
            )

        hidden_size = p.hidden_size
        n_heads = p.n_heads
        n_kv_heads = p.n_kv_heads
        head_dim = p.head_dim
        max_seq_len = p.max_seq_len
        rope_theta = p.rope_theta

        kv_params = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_layers=1,
            devices=[DeviceRef.GPU()],
        )

        # Variant-specific: quant config, layer dtype, stacked_qkv, rope kwargs.
        quant_config: QuantConfig | None = None
        rope_kwargs: dict[str, Any] = {}

        if dtype == "bf16":
            layer_dtype = DType.bfloat16
            stacked_qkv = False
        elif dtype == "fp8":
            layer_dtype = DType.float8_e4m3fn
            stacked_qkv = False
            quant_config = _make_fp8_quant_config()
            rope_kwargs["head_dim"] = head_dim
        elif dtype == "fp4":
            layer_dtype = DType.uint8
            stacked_qkv = False
            quant_config = _make_nvfp4_quant_config()

        rope = RotaryEmbedding(
            hidden_size,
            n_heads,
            rope_theta,
            max_seq_len,
            interleaved=False,
            **rope_kwargs,
        )

        fuse_rope = p._fuse_rope_and_store

        layer_kwargs: dict[str, Any] = {
            "rope": rope,
            "num_attention_heads": n_heads,
            "num_key_value_heads": n_kv_heads,
            "hidden_size": hidden_size,
            "kv_params": kv_params,
            "stacked_qkv": stacked_qkv,
            "has_bias": False,
            "dtype": layer_dtype,
            "devices": [DeviceRef.GPU()],
            "_fuse_rope_and_store": fuse_rope,
        }
        if quant_config is not None:
            layer_kwargs["quant_config"] = quant_config

        layer = AttentionWithRope(**layer_kwargs)

        # Variant-specific: weight generation.
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim

        if dtype == "bf16":
            # Use small std (~0.02) to match realistic weight distributions.
            # Large random weights cause softmax saturation and bf16 overflow,
            # making MAX vs torch comparison unreliable.
            std = 0.02
            weights = {
                "q_proj.weight": (
                    torch.randn(q_dim, hidden_size, dtype=torch.bfloat16) * std
                ),
                "k_proj.weight": (
                    torch.randn(kv_dim, hidden_size, dtype=torch.bfloat16) * std
                ),
                "v_proj.weight": (
                    torch.randn(kv_dim, hidden_size, dtype=torch.bfloat16) * std
                ),
                "o_proj.weight": (
                    torch.randn(hidden_size, q_dim, dtype=torch.bfloat16) * std
                ),
            }
            layer.load_state_dict(weights)
            # Keep weights for torch correctness comparison.
            self._torch_weights = weights
        elif dtype == "fp8":
            state: dict[str, Any] = {}
            state.update(_make_fp8_weights("q_proj", q_dim, hidden_size))
            state.update(_make_fp8_weights("k_proj", kv_dim, hidden_size))
            state.update(_make_fp8_weights("v_proj", kv_dim, hidden_size))
            state.update(_make_fp8_weights("o_proj", hidden_size, q_dim))
            layer.load_state_dict(state)
        elif dtype == "fp4":
            state = {}
            state.update(_make_fp4_weights("q_proj", q_dim, hidden_size))
            state.update(_make_fp4_weights("k_proj", kv_dim, hidden_size))
            state.update(_make_fp4_weights("v_proj", kv_dim, hidden_size))
            state.update(_make_fp4_weights("o_proj", hidden_size, q_dim))
            layer.load_state_dict(state)

        # Graph building is identical across dtypes.
        graph_name = {
            "bf16": "AttentionWithRope",
            "fp8": "AttentionWithRopeFP8",
            "fp4": "AttentionWithRopeFP4",
        }[dtype]

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
            graph_name,
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
            layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
            graph.output(
                layer(
                    layer_idx,
                    inputs.tensor,
                    kv_collection,
                    freqs_cis=rope.freqs_cis,
                    input_row_offsets=input_row_offsets.tensor,
                )
            )

        return graph, layer.state_dict()

    # ------------------------------------------------------------------ #
    # Correctness support (bf16 only, prefill with ctx_len=0)
    # ------------------------------------------------------------------ #

    def torch_reference_layer(
        self, device: str = "cuda"
    ) -> tuple[LlamaAttention, dict[str, Any]]:
        p = self.static_params
        if p.dtype != "bf16":
            raise NotImplementedError(
                f"Correctness testing not yet supported for dtype='{p.dtype}'"
            )
        if not hasattr(self, "_torch_weights"):
            raise RuntimeError(
                "build_and_compile must be called before torch_reference_layer"
            )

        config = LlamaConfig(
            hidden_size=p.hidden_size,
            num_attention_heads=p.n_heads,
            num_key_value_heads=p.n_kv_heads,
            head_dim=p.head_dim,
            max_position_embeddings=p.max_seq_len,
            rope_theta=p.rope_theta,
            attn_implementation="eager",
        )

        layer = LlamaAttention(config, layer_idx=0).to(
            device=device, dtype=torch.bfloat16
        )
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

        # Reshape flat input to (1, seq_len, hidden_size) for HF.
        hidden_states = (
            torch.from_dlpack(execute_args[0])
            .reshape(1, seq_len, p.hidden_size)
            .to(device=device, dtype=torch.bfloat16)
        )

        # Position embeddings via HF's LlamaRotaryEmbedding.
        config = LlamaConfig(
            hidden_size=p.hidden_size,
            num_attention_heads=p.n_heads,
            num_key_value_heads=p.n_kv_heads,
            head_dim=p.head_dim,
            max_position_embeddings=p.max_seq_len,
            rope_theta=p.rope_theta,
        )
        rotary_emb = LlamaRotaryEmbedding(config=config, device=device)
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
