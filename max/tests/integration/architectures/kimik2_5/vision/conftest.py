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
"""Shared torch reference implementations for Kimi K2.5 tests."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _torch_get_1d_sincos_pos_embed(embed_dim: int, t_size: int) -> np.ndarray:
    """1D sincos positional embedding. Returns (t_size, embed_dim)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    grid_t = np.arange(t_size, dtype=np.float32)
    out = np.einsum("m,d->md", grid_t, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _torch_get_rope_shape(
    org: torch.Tensor, interpolation_mode: str, shape: tuple[int, int]
) -> torch.Tensor:
    """Interpolate 2D pos grid (H, W, C) to (h, w), return (h*w, C)."""
    x = org.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=shape, mode=interpolation_mode)
    return x.squeeze(0).permute(1, 2, 0).flatten(0, 1)


class TorchPosEmb(nn.Module):
    """Torch reference for Learnable2DInterpPosEmbDivided_fixed."""

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        time_embed = _torch_get_1d_sincos_pos_embed(dim, num_frames)
        self.register_buffer(
            "time_weight",
            torch.from_numpy(time_embed).float().unsqueeze(1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames
            if (h, w) == (self.weight.shape[0], self.weight.shape[1]):
                pos_emb_2d = self.weight.flatten(0, 1)
            else:
                pos_emb_2d = _torch_get_rope_shape(
                    self.weight, self.interpolation_mode, (h, w)
                )
            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1)
                    + self.time_weight[0:t]
                )
            pos_embs.append(pos_emb_3d.reshape(-1, self.dim))
        return x + torch.cat(pos_embs, dim=0)


class TorchPatchEmbed(nn.Module):
    """Torch reference for MoonVision3dPatchEmbed (proj + pos_emb)."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 64,
        pos_emb_width: int = 64,
        pos_emb_time: int = 4,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=has_bias,
        )
        self.pos_emb = TorchPosEmb(
            height=pos_emb_height,
            width=pos_emb_width,
            num_frames=pos_emb_time,
            dim=out_dim,
        )

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).view(x.size(0), -1)
        return self.pos_emb(x, grid_thws)


class TorchMLP2(nn.Module):
    """PyTorch reference for the MLP2 layer (non-gated MLP with gelu_tanh)."""

    def __init__(
        self, dim: tuple[int, int, int], has_bias: bool = False
    ) -> None:
        super().__init__()
        self.up_proj = nn.Linear(dim[0], dim[1], bias=has_bias)
        self.down_proj = nn.Linear(dim[1], dim[2], bias=has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = F.gelu(x, approximate="tanh")
        return self.down_proj(x)


class TorchRope2DPosEmbRepeated(nn.Module):
    """Nearly verbatim copy of the Kimi-K2.5 reference ``Rope2DPosEmbRepeated``."""

    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        theta_base: float = 10000,
    ):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(
        self, grid_thws: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis",
                self._precompute_freqs_cis(device),
                persistent=False,
            )

        shapes = grid_thws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width
            for t, h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in shapes
            ],
            dim=0,
        )
        return freqs_cis


def _torch_apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _torch_eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_row_offsets: torch.Tensor,
) -> torch.Tensor:
    seq_length = q.shape[0]
    attention_mask = torch.zeros(
        [1, seq_length, seq_length], device=q.device, dtype=torch.bool
    )
    for i in range(1, len(input_row_offsets)):
        attention_mask[
            ...,
            input_row_offsets[i - 1] : input_row_offsets[i],
            input_row_offsets[i - 1] : input_row_offsets[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight += attention_mask
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(
        q.dtype
    )

    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


class TorchEncoderBlock(nn.Module):
    """PyTorch reference for a single vision encoder layer."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = hidden_dim // num_heads

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.wo = nn.Linear(hidden_dim, hidden_dim)

        self.mlp = TorchMLP2(
            dim=(hidden_dim, mlp_dim, hidden_dim), has_bias=True
        )

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        input_row_offsets: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xqkv = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = _torch_apply_rope(xq, xk, rope_freqs_cis)

        attn_out = _torch_eager_attention(
            xq, xk, xv, input_row_offsets=input_row_offsets
        )

        attn_out = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        x: torch.Tensor,
        input_row_offsets: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm0(x)

        x = self.attention_qkvpacked(x, input_row_offsets, rope_freqs_cis)
        x = residual + x

        residual = x
        x = self.norm1(x)
        x = self.mlp(x)
        x = residual + x

        return x


class TorchPatchMergerMLP(nn.Module):
    """PyTorch reference for PatchMergerMLP.

    Math matches HuggingFace reference (modeling_kimi_k25.py PatchMergerMLP):
    pre_norm over last dim, reshape to (N, mm_hidden_size * kH * kW), then
    Linear -> GELU -> Linear to decoder_hidden_size.
    """

    def __init__(
        self,
        mm_hidden_size: int,
        decoder_hidden_size: int,
        merge_kernel_size: tuple[int, int],
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        input_dim = mm_hidden_size * merge_kernel_size[0] * merge_kernel_size[1]
        self.pre_norm = nn.LayerNorm(mm_hidden_size, eps=eps)
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, decoder_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = F.gelu(x)
        return self.linear2(x)


class TorchEncoder(nn.Module):
    """PyTorch reference for the full vision encoder."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        num_layers: int,
        rope_max_height: int,
        rope_max_width: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.rope_2d = TorchRope2DPosEmbRepeated(
            hidden_dim // num_heads, rope_max_height, rope_max_width, rope_theta
        )
        self.blocks = nn.ModuleList(
            [
                TorchEncoderBlock(num_heads, hidden_dim, mlp_dim)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        input_row_offsets: torch.Tensor,
        grid_thws: torch.Tensor,
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws, device=x.device)
        for block in self.blocks:
            x = block(x, input_row_offsets, rope_freqs_cis)
        x = self.norm(x)
        return x
