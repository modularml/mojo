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
"""Idefics3 vision attention layers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef, TensorValue
from max.nn.attention.multihead_attention import MultiheadAttention
from max.nn.linear import Linear


class Idefics3VisionAttention(MultiheadAttention):
    """Idefics3 vision multi-head attention layer.

    Multi-headed attention from 'Attention Is All You Need' paper,
    adapted for Idefics3 vision transformer component.

    From vision_config:
    - hidden_size: 1152
    - num_attention_heads: 16
    - head_dim: 1152 // 16 = 72
    - scale: 72^(-0.5) ≈ 0.1178
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType = DType.bfloat16,
    ) -> None:
        """Initialize Idefics3 vision attention layer.

        Args:
            hidden_size: The dimension of the hidden states (embed_dim).
                From config: 1152
            num_attention_heads: The number of attention heads.
                From config: 16
            devices: Device(s) to place the weights and run the computation.
                If multiple devices provided, uses distributed computation.
            dtype: DType of the QKV and output projection weights.

        Raises:
            ValueError: If hidden_size is not divisible by num_attention_heads.
        """
        # Validate that embed_dim is divisible by num_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads "
                f"(got `hidden_size`: {hidden_size} and `num_attention_heads`: "
                f"{num_attention_heads})."
            )

        head_dim = hidden_size // num_attention_heads
        # Scale factor for attention: 1/sqrt(head_dim)
        scale = head_dim ** (-0.5)

        devices_list = list(devices) if devices else []

        super().__init__(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            devices=devices_list or None,
            dtype=dtype,
            scale=scale,
            qkv_has_bias=True,  # Idefics3 uses bias in QKV projections
            o_proj_has_bias=True,  # Idefics3 uses bias in output projection
            stacked_qkv=False,  # Use efficient stacked QKV computation
        )

        self.is_causal = False  # Vision attention is not causal

        # Override the output projection with PyTorch-compatible naming
        self._init_pytorch_compatible_weights(dtype)

    def _init_pytorch_compatible_weights(self, dtype: DType) -> None:
        """Initialize output projection with PyTorch-compatible naming."""
        # Replace o_proj with out_proj to match PyTorch naming
        self.out_proj = Linear(
            in_dim=self.embed_dim,
            out_dim=self.embed_dim,
            has_bias=self.o_proj_has_bias,
            dtype=dtype,
            device=self.devices[0],
        )
        # Remove the original o_proj to avoid conflicts
        if hasattr(self, "o_proj"):
            delattr(self, "o_proj")

    def _forward_single(self, x: TensorValue, **kwargs) -> TensorValue:
        """Single-device forward pass with PyTorch-compatible naming.

        Override to use out_proj instead of o_proj.
        """
        # Compute QKV
        q, k, v = self._compute_qkv(x)

        # Apply attention
        attn_out = self._apply_attention(q, k, v, **kwargs)

        # Output projection using PyTorch-compatible naming
        return self.out_proj(attn_out)

    def __call__(  # type: ignore[override]
        self,
        x: TensorValue,
        signal_buffers: None = None,
        **kwargs: Any,
    ) -> TensorValue:
        """Forward pass of Idefics3 vision attention.

        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_size].
            signal_buffers: Not used in vision attention (set to None).
            **kwargs: Additional arguments, including optional attention_mask.

        Returns:
            attention_output: Output tensor of shape [batch_size, seq_length, hidden_size].
        """
        # Extract attention_mask from kwargs if provided
        attention_mask = kwargs.get("attention_mask")

        # Use our custom _forward_single method for the core attention computation
        attn_output = self._forward_single(x, attention_mask=attention_mask)

        return attn_output
