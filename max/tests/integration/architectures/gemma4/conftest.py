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
"""Shared torch reference implementations for Gemma4 tests."""

# ruff: noqa

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils.generic import maybe_autocast

import json
import math
import os
from pathlib import Path

from max.driver import Accelerator, Device
from max.dtype import DType
from max.engine import InferenceSession
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

if TYPE_CHECKING:
    from max.pipelines.architectures.gemma4.model_config import (
        Gemma4TextConfig,
    )

# Values from the gemma-4-31B-it config.json text_config section.
TEXT_HIDDEN_SIZE = 5376
TEXT_INTERMEDIATE_SIZE = 21504
TEXT_HIDDEN_ACTIVATION = "gelu_pytorch_tanh"
TEXT_SLIDING_WINDOW = 1024
TEXT_NUM_ATTENTION_HEADS = 32
TEXT_NUM_KEY_VALUE_HEADS = 16
TEXT_HEAD_DIM = 256
TEXT_RMS_NORM_EPS = 1e-6
TEXT_NUM_HIDDEN_LAYERS = 60
TEXT_LAYER_TYPES = [
    "sliding_attention" if (i + 1) % 6 else "full_attention"
    for i in range(TEXT_NUM_HIDDEN_LAYERS)
]

# Global (full-attention) RoPE parameters from the text_config rope_parameters.
TEXT_GLOBAL_HEAD_DIM = 512
TEXT_GLOBAL_ROPE_THETA = 1000000.0
TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR = 0.25

TEXT_NUM_GLOBAL_KEY_VALUE_HEADS = 4
TEXT_ATTENTION_K_EQ_V = True
TEXT_VOCAB_SIZE = 262144
TEXT_FINAL_LOGIT_SOFTCAPPING = 30.0
TEXT_TIE_WORD_EMBEDDINGS = True
TEXT_SLIDING_WINDOW_ROPE_THETA = 10000.0

VISION_HIDDEN_SIZE = 1152
VISION_DEFAULT_OUTPUT_LENGTH = 280
VISION_POOLING_KERNEL_SIZE = 3
VISION_HEAD_DIM = 72

# Values used for vision patch-embedder unit tests.  Kept small for speed.
VISION_EMBED_HIDDEN_SIZE = 64
VISION_PATCH_SIZE = 4
VISION_POSITION_EMBEDDING_SIZE = 16
VISION_NUM_HEADS = 4
VISION_RMS_NORM_EPS = 1e-6

# Values from the gg-hf-gg/gemma-4-26B-A4B-it config.json text_config section.
MOE_TEXT_NUM_EXPERTS = 128
MOE_TEXT_TOP_K_EXPERTS = 8
MOE_TEXT_MOE_INTERMEDIATE_SIZE = 704
MOE_TEXT_HIDDEN_SIZE = 2816
MOE_TEXT_INTERMEDIATE_SIZE = 2112
MOE_TEXT_HIDDEN_ACTIVATION = "gelu_pytorch_tanh"
MOE_TEXT_RMS_NORM_EPS = 1e-6


@pytest.fixture()
def gemma4_text_config() -> PreTrainedConfig:
    """Minimal PreTrainedConfig matching gemma-4-31B-it text_config values."""
    config = PreTrainedConfig()
    config.hidden_size = TEXT_HIDDEN_SIZE
    config.sliding_window = TEXT_SLIDING_WINDOW
    config.num_attention_heads = TEXT_NUM_ATTENTION_HEADS
    config.num_key_value_heads = TEXT_NUM_KEY_VALUE_HEADS
    config.head_dim = TEXT_HEAD_DIM
    config.num_hidden_layers = TEXT_NUM_HIDDEN_LAYERS
    config.layer_types = TEXT_LAYER_TYPES
    return config


def _compute_proportional_rope_parameters(
    config: PreTrainedConfig,
    device: torch.device,
    layer_type: str,
    head_dim_key: str = "head_dim",
) -> tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with proportional RoPE.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`): The proportion of the embedding dimension
                to apply rotary positional encoding, e.g., [0.0, 0.25, 0.5, 0.75, 1.0]. Unlike other RoPE functions
                that use this parameter, proportional RoPE will always return an encoding that is the size of
                `head_dim`.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = (
        config.rope_parameters[layer_type]
        if layer_type is not None
        else config.rope_parameters
    )

    head_dim = (
        getattr(config, head_dim_key, None)
        or config.hidden_size // config.num_attention_heads
    )
    base = rope_parameters_dict["rope_theta"]
    factor = rope_parameters_dict.get("factor", 1.0)
    rope_proportion = rope_parameters_dict.get("partial_rotary_factor", 1.0)

    attention_factor = 1.0  # Unused in this type of RoPE

    rope_angles = int(rope_proportion * head_dim // 2)

    inv_freq_rotated = 1.0 / (
        base
        ** (
            torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / head_dim
        )
    )

    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat(
            (
                inv_freq_rotated,
                torch.zeros(nope_angles, dtype=torch.float32, device=device),
            ),
            dim=0,
        )
    else:
        inv_freq = inv_freq_rotated

    inv_freq /= factor
    return inv_freq, attention_factor


# This maps the "rope_type" string field in rope config to the corresponding function to compute the RoPE parameters
# from the model config. You can append new {'rope_type': callable} pairs to this rope_parameters to enable custom RoPE
# parameterizations, as long as the callable has the same signature.
ROPE_INIT_FUNCTIONS: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {
    # "linear": _compute_linear_scaling_rope_parameters,
    # "dynamic": _compute_dynamic_ntk_parameters,
    # "yarn": _compute_yarn_parameters,
    # "longrope": _compute_longrope_parameters,
    # "llama3": _compute_llama3_parameters,
    "proportional": _compute_proportional_rope_parameters,
}


def torch_compute_proportional_rope_inv_freqs(
    head_dim: int,
    theta: float,
    partial_rotary_factor: float,
) -> torch.Tensor:
    """Reference implementation of proportional RoPE inverse frequencies.

    Ported from ``_compute_proportional_rope_parameters`` in HuggingFace
    transformers' ``modeling_rope_utils.py``.
    """
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    nope_angles = head_dim // 2 - rope_angles
    inv_freq_rotated = 1.0 / (
        theta
        ** (
            torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float()
            / head_dim
        )
    )
    if nope_angles > 0:
        return torch.cat(
            [inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)]
        )
    return inv_freq_rotated


def _torch_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _torch_apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Standard 1D RoPE (half-rotate convention)."""
    return (x * cos) + (_torch_rotate_half(x) * sin)


def torch_apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    ndim: int,
) -> torch.Tensor:
    """Reference implementation of multi-dimensional RoPE.

    Ported from HuggingFace Gemma4's ``apply_multidimensional_rope``.
    Splits ``head_dim`` into ``ndim`` equal parts, applies independent 1D RoPE
    per spatial dimension, then concatenates.
    """
    num_input_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))

    split_sizes = [num_rotated_channels_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    y_parts = [
        _torch_apply_rotary_pos_emb(x_parts[k], cos_parts[k], sin_parts[k])
        for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


def token_type_ids_mask_function(
    token_type_ids: torch.Tensor | None,
    image_group_ids: torch.Tensor | None,
) -> Callable[..., bool] | None:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """
    # Do not return an additional mask in this case
    if token_type_ids is None:
        return None

    def inner_mask(
        batch_idx: int, head_idx: int, q_idx: int, kv_idx: int
    ) -> bool:
        # If it's 1 for both query and key/value, we are in an image block
        # NOTE: static cache shape goes beyond input seq length, while token_type_ids.shape[1] == input seq length
        # Since vmap doesn't support `if statement` we workaround it with `torch.where`
        assert image_group_ids is not None

        safe_q_idx = torch.where(q_idx < token_type_ids.shape[1], q_idx, 0)
        safe_kv_idx = torch.where(kv_idx < token_type_ids.shape[1], kv_idx, 0)

        token_type_ids_at_q_idx = token_type_ids[batch_idx, safe_q_idx]
        token_type_ids_at_q_idx = torch.where(
            q_idx < token_type_ids.shape[1], token_type_ids_at_q_idx, 0
        )

        token_type_ids_at_kv_idx = token_type_ids[batch_idx, safe_kv_idx]
        token_type_ids_at_kv_idx = torch.where(
            kv_idx < token_type_ids.shape[1], token_type_ids_at_kv_idx, 0
        )

        image_group_ids_at_q_idx = image_group_ids[batch_idx, safe_q_idx]
        image_group_ids_at_q_idx = torch.where(
            q_idx < image_group_ids.shape[1], image_group_ids_at_q_idx, -1
        )

        image_group_ids_at_kv_idx = image_group_ids[batch_idx, safe_kv_idx]
        image_group_ids_at_kv_idx = torch.where(
            kv_idx < image_group_ids.shape[1], image_group_ids_at_kv_idx, -1
        )

        is_image_block = (token_type_ids_at_q_idx == 1) & (
            token_type_ids_at_kv_idx == 1
        )
        same_image_block = image_group_ids_at_q_idx == image_group_ids_at_kv_idx

        # This is bidirectional attention whenever we are dealing with image tokens
        return is_image_block & same_image_block

    return inner_mask


class TorchGemma4VisionPatchEmbedder(nn.Module):
    """Copy-pasted HF Gemma4VisionPatchEmbedder for test reference comparison.

    Ported from ``transformers.models.gemma4.modeling_gemma4``.

    Accepts batched inputs (``batch, num_patches, ...``).  Set
    ``padding_positions`` to all-False for tests that have no padding.
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        position_embedding_size: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.position_embedding_size = position_embedding_size

        self.input_proj = nn.Linear(
            3 * self.patch_size**2, self.hidden_size, bias=False
        )
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size)
        )

    def _position_embeddings(
        self,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare patch positions map for matmul with position embedding table."""
        import torch.nn.functional as F

        clamped_positions = pixel_position_ids.clamp(min=0)
        one_hot = F.one_hot(
            clamped_positions, num_classes=self.position_embedding_size
        )
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        position_embeddings = one_hot @ self.position_embedding_table
        position_embeddings = position_embeddings.sum(dim=1)
        position_embeddings = torch.where(
            padding_positions.unsqueeze(-1), 0.0, position_embeddings
        )
        return position_embeddings

    def _patch_projection(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Project patches into model space as [batch, num_patches, hidden_size]."""
        patches = 2 * (pixel_values - 0.5)
        return self.input_proj(patches.to(self.input_proj.weight.dtype))

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self._patch_projection(pixel_values)
        position_embeddings = self._position_embeddings(
            pixel_position_ids, padding_positions
        )
        return hidden_states + position_embeddings


class TorchGemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        # Use torch.pow() (over torch.sqrt() or torch.rsqrt()) to address compiler differences between Torch and JAX
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)


class TorchGemma4MLP(nn.Module):
    """Copy-pasted HF Gemma4MLP for test reference comparison."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, hidden_activation: str
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TorchGemma4TextDecoderLayer(nn.Module):
    """Copy-pasted HF Gemma4TextDecoderLayer for test reference comparison.

    Attention is passed as a constructor argument so that tests can inject a
    stub.  When ``with_layer_scalar=True`` (default, matches text config), a
    ``layer_scalar`` buffer is created and applied.  Vision layers pass
    ``with_layer_scalar=False`` to omit it.

    When ``enable_moe_block=True``, the feedforward section splits into a
    shared MLP branch and a routed MoE branch (each with its own pre/post
    norms), matching the HuggingFace ``Gemma4TextDecoderLayer``.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        mlp: nn.Module,
        hidden_size: int,
        rms_norm_eps: float,
        with_layer_scalar: bool = True,
        enable_moe_block: bool = False,
        router: nn.Module | None = None,
        moe: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.mlp = mlp
        self.input_layernorm = TorchGemma4RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = TorchGemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.pre_feedforward_layernorm = TorchGemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.post_feedforward_layernorm = TorchGemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.with_layer_scalar = with_layer_scalar
        if with_layer_scalar:
            self.layer_scalar = nn.parameter.Buffer(torch.ones(1))

        self.enable_moe_block = enable_moe_block
        if self.enable_moe_block:
            assert router is not None and moe is not None
            self.router = router
            self.moe = moe
            self.post_feedforward_layernorm_1 = TorchGemma4RMSNorm(
                hidden_size, eps=rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = TorchGemma4RMSNorm(
                hidden_size, eps=rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = TorchGemma4RMSNorm(
                hidden_size, eps=rms_norm_eps
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, **kwargs)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states

        if self.enable_moe_block:
            hidden_states_1 = self.pre_feedforward_layernorm(hidden_states)
            hidden_states_1 = self.mlp(hidden_states_1)
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states_1)

            hidden_states_flat = hidden_states.reshape(
                -1, hidden_states.shape[-1]
            )
            top_k_weights, top_k_index = self.router(hidden_states_flat)
            hidden_states_2 = self.pre_feedforward_layernorm_2(
                hidden_states_flat
            )
            hidden_states_2 = self.moe(
                hidden_states_2, top_k_index, top_k_weights
            )
            hidden_states_2 = hidden_states_2.reshape(hidden_states.shape)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            hidden_states = hidden_states_1 + hidden_states_2
        else:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.with_layer_scalar:
            hidden_states *= self.layer_scalar

        return (hidden_states,)


class TorchGemma4VisionPooler(nn.Module):
    """Copy-pasted HF Gemma4VisionPooler for test reference comparison."""

    def __init__(self, hidden_size: int, default_output_length: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.default_output_length = default_output_length
        self.root_hidden_size = self.hidden_size**0.5

    def _avg_pool_by_positions(
        self,
        x: torch.Tensor,
        patch_positions: torch.Tensor,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = x.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {x.shape} to {length}: "
                f"{k=}^2 times {length=} must be {input_seq_len}."
            )

        clamped_positions = patch_positions.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = (
            nn.functional.one_hot(kernel_idxs.long(), length).float()
            / k_squared
        )
        output = weights.transpose(1, 2).to(x.dtype) @ x
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output, mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        patch_positions: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_length is None:
            output_length = self.default_output_length

        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens (requested {output_length})"
                f" than there are patches ({hidden_states.shape[1]})."
            )

        if hidden_states.shape[1] == output_length:
            mask = padding_positions
        else:
            hidden_states, mask = self._avg_pool_by_positions(
                hidden_states, patch_positions, output_length
            )

        hidden_states *= self.root_hidden_size
        return hidden_states, mask


class TorchGemma4TextScaledWordEmbedding(nn.Embedding):
    """Copy-pasted HF Gemma4TextScaledWordEmbedding for test reference."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer(
            "embed_scale", torch.tensor(embed_scale), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(
            self.weight.dtype
        )


class TorchGemma4TextModel(nn.Module):
    """Copy-pasted HF Gemma4TextModel for test reference comparison.

    Simplified: no gradient checkpointing, no per-layer inputs, attention is
    injected via ``attn_factory`` so tests can stub it.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        intermediate_size: int,
        hidden_activation: str,
        rms_norm_eps: float,
        layer_types: list[str],
        attn_factory: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layer_types = layer_types

        self.embed_tokens = TorchGemma4TextScaledWordEmbedding(
            vocab_size,
            hidden_size,
            padding_idx=0,
            embed_scale=hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [
                TorchGemma4TextDecoderLayer(
                    self_attn=attn_factory(layer_idx=i),
                    mlp=TorchGemma4MLP(
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        hidden_activation=hidden_activation,
                    ),
                    hidden_size=hidden_size,
                    rms_norm_eps=rms_norm_eps,
                    with_layer_scalar=True,
                )
                for i in range(num_hidden_layers)
            ]
        )
        self.norm = TorchGemma4RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Simplified forward: embed → layers → norm. No KV cache."""
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma4ClippableLinear(nn.Linear):
    def __init__(
        self,
        config: PreTrainedConfig,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}
        use_clipping = getattr(config, "use_clipped_linears", False)
        if use_clipping:
            self.input_min = nn.parameter.Buffer(
                torch.tensor(-float("inf"), **factory_kwargs)
            )
            self.input_max = nn.parameter.Buffer(
                torch.tensor(float("inf"), **factory_kwargs)
            )
            self.output_min = nn.parameter.Buffer(
                torch.tensor(-float("inf"), **factory_kwargs)
            )
            self.output_max = nn.parameter.Buffer(
                torch.tensor(float("inf"), **factory_kwargs)
            )
        else:
            self.input_min = None
            self.input_max = None
            self.output_min = None
            self.output_max = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_min and self.input_max:
            x = torch.clamp(x, self.input_min, self.input_max)

        x = super().forward(x)

        if self.output_min and self.output_max:
            x = torch.clamp(x, self.output_min, self.output_max)
        return x


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        x (`torch.Tensor`): The tensor to embed.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_torch_rotate_half(x) * sin)


def _apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_torch_rotate_half(x) * sin)


def _apply_2d_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply 2-D multidimensional RoPE (non-batched).

    Args:
        x: ``[total_patches, num_heads, head_dim]``
        cos: ``[total_patches, head_dim]`` — first half is dim-0 freqs
            (repeated for rotate-half convention), second half is dim-1.
        sin: same shape as cos.

    Returns:
        Tensor with same shape as ``x``.
    """
    head_dim = x.shape[-1]
    x_parts = x.split(head_dim // 2, dim=-1)  # each [p, N, H//2]
    cos_parts = cos.split(head_dim // 2, dim=-1)  # each [p, H//2]
    sin_parts = sin.split(head_dim // 2, dim=-1)  # each [p, H//2]
    y_parts = [
        _apply_rotary_pos_emb(
            x_parts[k], cos_parts[k], sin_parts[k], unsqueeze_dim=1
        )
        for k in range(2)
    ]
    return torch.cat(y_parts, dim=-1)


class TorchGemma4VisionAttention(nn.Module):
    """Reference Gemma4VisionAttention for test comparison.

    Ported from ``transformers.models.gemma4.modeling_gemma4``.

    Operates on flat (non-batched) ``[total_patches, hidden_size]`` inputs,
    matching the MAX ragged representation.  Uses eager scaled dot-product
    attention with ``scale=1.0`` and no causal mask (bidirectional).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        proj_out = num_heads * head_dim

        self.q_proj = nn.Linear(hidden_size, proj_out, bias=False)
        self.k_proj = nn.Linear(hidden_size, proj_out, bias=False)
        self.v_proj = nn.Linear(hidden_size, proj_out, bias=False)
        self.o_proj = nn.Linear(proj_out, hidden_size, bias=False)

        # Q/K norms: learned scale with weight_offset=1 (Gemma4 convention).
        self.q_norm = TorchGemma4RMSNorm(
            head_dim, eps=rms_norm_eps, with_scale=True
        )
        self.k_norm = TorchGemma4RMSNorm(
            head_dim, eps=rms_norm_eps, with_scale=True
        )
        # V norm: bare normalisation with no learnable scale.
        self.v_norm = TorchGemma4RMSNorm(
            head_dim, eps=rms_norm_eps, with_scale=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute vision self-attention.

        Args:
            hidden_states: ``[total_patches, hidden_size]``
            position_embeddings: ``(cos, sin)`` each ``[total_patches, head_dim]``
                in HF format — first/second ``head_dim // 2`` hold dim-0/dim-1
                frequencies (each repeated for the rotate-half convention).

        Returns:
            ``[total_patches, hidden_size]``
        """
        n = hidden_states.shape[0]
        cos, sin = position_embeddings

        # Project and reshape to [total_patches, num_heads, head_dim].
        xq = self.q_proj(hidden_states).view(n, self.num_heads, self.head_dim)
        xk = self.k_proj(hidden_states).view(n, self.num_heads, self.head_dim)
        xv = self.v_proj(hidden_states).view(n, self.num_heads, self.head_dim)

        # Per-head RMS norms.
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xv = self.v_norm(xv)

        # 2-D multidimensional RoPE (ported from HF apply_multidimensional_rope).
        xq = _apply_2d_multidimensional_rope(xq, cos, sin)
        xk = _apply_2d_multidimensional_rope(xk, cos, sin)

        # Transpose to [num_heads, total_patches, head_dim] for matmul.
        xq = xq.transpose(0, 1)
        xk = xk.transpose(0, 1)
        xv = xv.transpose(0, 1)

        # Bidirectional attention, scale=1.0.
        # Upcast to float32 for softmax (matches HF eager_attention_forward),
        # then cast back to xq.dtype before the output matmul.
        attn = torch.matmul(xq, xk.transpose(-2, -1))  # scale=1.0
        attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(
            xq.dtype
        )
        out = torch.matmul(attn, xv)

        # Reshape back to [total_patches, num_heads * head_dim].
        out = out.transpose(0, 1).reshape(n, -1)
        return self.o_proj(out)


class TorchGemma4TextExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts, 2 * self.intermediate_dim, self.hidden_dim
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_experts, self.hidden_dim, self.intermediate_dim
            )
        )
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(
                current_state, self.gate_up_proj[expert_idx]
            ).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(
                current_hidden_states, self.down_proj[expert_idx]
            )
            current_hidden_states = (
                current_hidden_states
                * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )

        return final_hidden_states


class TorchGemma4TextRouter(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size**-0.5
        self.eps = config.rms_norm_eps

        self.norm = TorchGemma4RMSNorm(
            self.hidden_size, eps=self.eps, with_scale=False
        )
        self.proj = nn.Linear(
            config.hidden_size, config.num_experts, bias=False
        )
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        expert_scores = self.proj(hidden_states)  # [B*S, E]
        router_probabilities = nn.functional.softmax(expert_scores, dim=-1)

        # topk returns both values (probabilities) and indices directly
        top_k_weights, top_k_index = torch.topk(
            router_probabilities,
            k=self.config.top_k_experts,
            dim=-1,
        )  # both [B*S, K]

        # Normalize the top-k weights so they sum to 1 per token
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)

        # Apply per-expert scale directly to the weights
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        return top_k_weights, top_k_index


class Gemma4RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(
        self,
        config: PreTrainedConfig,
        device: str | torch.device | None = None,
        layer_type: str | None = None,
    ):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.layer_types = set(config.layer_types)
        self.rope_init_fns: dict[
            str, Callable[..., tuple[torch.Tensor, float]]
        ] = {}
        self.rope_type: dict[str, str] = {}

        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue

            if (rope_type := rope_params["rope_type"]) != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            else:
                rope_init_fn = self.compute_default_rope_parameters

            self.rope_init_fns[layer_type] = rope_init_fn
            self.rope_type[layer_type] = rope_type

            rope_init_fn_kwargs = {"device": device, "layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

            curr_inv_freq, curr_attention_scaling = rope_init_fn(
                self.config, **rope_init_fn_kwargs
            )
            self.register_buffer(
                f"{layer_type}_inv_freq", curr_inv_freq, persistent=False
            )
            self.register_buffer(
                f"{layer_type}_original_inv_freq",
                curr_inv_freq.clone(),
                persistent=False,
            )
            setattr(
                self, f"{layer_type}_attention_scaling", curr_attention_scaling
            )

    @staticmethod
    def compute_default_rope_parameters(
        config: PreTrainedConfig,
        device: torch.device,
        layer_type: str,
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            layer_type (`str`):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, attention_factor

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )

        if position_ids.ndim == 2:
            # Standard 1D positions: [batch, seq_len]
            inv_freq_expanded = (
                inv_freq[None, :, None]
                .float()
                .expand(position_ids.shape[0], -1, 1)
                .to(x.device)
            )
            position_ids_expanded = position_ids[:, None, :].float()

            with maybe_autocast(device_type=device_type, enabled=False):
                freqs = (
                    inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * attention_scaling
                sin = emb.sin() * attention_scaling
        else:
            # Multidimensional positions (e.g., vision 2D): [batch, num_patches, ndim]
            # The reference implementation computes RoPE frequencies INDEPENDENTLY
            # for each spatial dimension using the partitioned head_dim (head_dim // ndim),
            # so both x and y dimensions get identical frequency ranges.
            # This is different from splitting the global inv_freq between dimensions.
            ndim = position_ids.shape[-1]
            head_dim_per_dim = (inv_freq.shape[0] * 2) // ndim
            rope_theta = self.config.rope_parameters[layer_type]["rope_theta"]
            all_embs = []

            for d in range(ndim):
                # Recompute inv_freq independently for this dimension using partitioned head_dim
                dim_inv_freq = 1.0 / (
                    rope_theta
                    ** (
                        torch.arange(
                            0,
                            head_dim_per_dim,
                            2,
                            device=x.device,
                            dtype=torch.float,
                        )
                        / head_dim_per_dim
                    )
                )
                dim_inv_freq_expanded = (
                    dim_inv_freq[None, :, None]
                    .float()
                    .expand(position_ids.shape[0], -1, 1)
                )
                dim_positions = position_ids[:, :, d]  # [batch, num_patches]
                dim_positions_expanded = dim_positions[:, None, :].float()

                with maybe_autocast(device_type=device_type, enabled=False):
                    dim_freqs = (
                        dim_inv_freq_expanded.float()
                        @ dim_positions_expanded.float()
                    ).transpose(1, 2)
                    # Each dimension gets its own duplicated freqs for the half-rotate convention
                    dim_emb = torch.cat(
                        (dim_freqs, dim_freqs), dim=-1
                    )  # [batch, seq, 2*freq_per_dim]
                    all_embs.append(dim_emb)

            with maybe_autocast(device_type=device_type, enabled=False):
                # Concatenate per-dimension embeddings: [batch, seq, ndim * 2 * freq_per_dim]
                emb = torch.cat(all_embs, dim=-1)
                cos = emb.cos() * attention_scaling
                sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim
    )


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Gemma4TextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PreTrainedConfig, layer_idx: int):
        super().__init__()
        self.layer_type = (
            config.layer_types[layer_idx]
            if hasattr(config, "layer_types")
            else None
        )
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = (
            config.global_head_dim
            if not self.is_sliding and config.global_head_dim
            else config.head_dim
        )
        self.use_alternative_attention = (
            config.attention_k_eq_v and not self.is_sliding
        )
        num_key_value_heads = (
            config.num_global_key_value_heads
            if self.use_alternative_attention
            else config.num_key_value_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // num_key_value_heads
        )
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = config.use_bidirectional_attention != "all"

        # Shared kv cache
        first_kv_shared_layer_idx = self.config.num_hidden_layers - getattr(
            self.config, "num_kv_shared_layers", 0
        )
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            # For shared layers, find the last non-shared layer of the same type before sharing starts
            self.kv_shared_layer_index = (
                len(prev_layers)
                - 1
                - prev_layers[::-1].index(config.layer_types[layer_idx])
            )
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            # For non-shared layers, store full-length kv if this is the last non-shared layer of its type
            self.store_full_length_kv = layer_idx == len(
                prev_layers
            ) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])

        self.q_norm = TorchGemma4RMSNorm(
            dim=self.head_dim, eps=config.rms_norm_eps
        )
        self.k_norm = TorchGemma4RMSNorm(
            dim=self.head_dim, eps=config.rms_norm_eps
        )
        self.v_norm = TorchGemma4RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, with_scale=False
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = (
            nn.Linear(
                config.hidden_size,
                num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
            )
            if not self.use_alternative_attention
            else None
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: FlashAttentionKwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(
            query_states, cos, sin, unsqueeze_dim=2
        )
        query_states = query_states.transpose(1, 2)

        # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer
        if self.is_kv_shared_layer and past_key_values is not None:
            key_states, value_states = past_key_values.shared_layers[
                self.kv_shared_layer_index
            ]
            # Device of past layer may be different from current one
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = (
                self.v_proj(hidden_states).view(hidden_shape)
                if self.v_proj is not None
                else key_states
            )

            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(
                key_states, cos, sin, unsqueeze_dim=2
            )
            key_states = key_states.transpose(1, 2)

            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            if not self.is_kv_shared_layer:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx
                )
            if self.store_full_length_kv:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[self.layer_idx] = (
                    key_states,
                    value_states,
                )

        attention_interface: Callable[
            ..., tuple[torch.Tensor, torch.Tensor | None]
        ] = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class TorchGemma4MultimodalEmbedder(nn.Module):
    """Copy-pasted HF Gemma4MultimodalEmbedder for test reference comparison."""

    def __init__(
        self,
        multimodal_hidden_size: int,
        text_hidden_size: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.embedding_projection = nn.Linear(
            multimodal_hidden_size, text_hidden_size, bias=False
        )
        self.embedding_pre_projection_norm = TorchGemma4RMSNorm(
            multimodal_hidden_size, eps=rms_norm_eps, with_scale=False
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)
