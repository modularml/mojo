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

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import Model
from max.graph.buffer_utils import cast_dlpack_to
from max.graph.weights import WeightData, Weights
from max.profiler import Tracer

# Weight key remapping from diffusers -> MAX module naming
_KEY_REMAP = [
    (".attn1.to_out.0.", ".attn1.to_out."),
    (".attn2.to_out.0.", ".attn2.to_out."),
    (".ffn.net.0.proj.", ".ffn.proj."),
    (".ffn.net.2.", ".ffn.linear_out."),
    # Image embedder GELU FFN: diffusers nested structure → flat Linear layers
    ("image_embedder.ff.net.0.proj.", "image_embedder.ff_proj."),
    ("image_embedder.ff.net.2.", "image_embedder.ff_out."),
]

# Keys to skip (non-persistent buffers computed at runtime)
_SKIP_PREFIXES = ("rope.freqs_cos", "rope.freqs_sin")


def _remap_state_dict(
    weights: Weights,
    target_dtype: DType = DType.bfloat16,
) -> dict[str, Any]:
    """Remap diffusers weight keys to MAX module naming, permute Conv3d,
    and cast weights to target dtype.

    Some WAN checkpoints store weights as float32 (A14B), others as
    bfloat16 (5B). We cast all to target_dtype to match the module
    parameter declarations.
    """
    state_dict: dict[str, Any] = {}

    # First pass: collect all weights with key remapping.
    raw_dict: dict[str, Any] = {}
    for key, value in weights.items():
        if any(key.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue

        new_key = key
        for old, new in _KEY_REMAP:
            new_key = new_key.replace(old, new)

        tensor = value.data()

        # Conv3d weight permutation for patch_embedding
        # Diffusers: [F, C, D, H, W] (PyTorch FCDHW)
        # MAX Conv3d(permute=False): [D, H, W, C, F] (QRSCF)
        if new_key == "patch_embedding.weight" and len(tensor.shape) == 5:
            buf = tensor.to_buffer() if hasattr(tensor, "to_buffer") else tensor
            t_f32 = cast_dlpack_to(buf, tensor.dtype, DType.float32, CPU())
            permuted: WeightData | np.ndarray = np.ascontiguousarray(
                np.from_dlpack(t_f32).transpose(2, 3, 4, 1, 0)
            )
            raw_dict[new_key] = permuted
        else:
            raw_dict[new_key] = tensor

    # Second pass: fuse attn2.to_k + attn2.to_v into attn2.to_kv
    fused_keys: set[str] = set()
    for key in list(raw_dict.keys()):
        if ".attn2.to_k." in key:
            k_key = key
            v_key = key.replace(".attn2.to_k.", ".attn2.to_v.")
            kv_key = key.replace(".attn2.to_k.", ".attn2.to_kv.")
            if v_key in raw_dict:
                k_data = raw_dict[k_key]
                v_data = raw_dict[v_key]
                k_buf = (
                    k_data.to_buffer()
                    if hasattr(k_data, "to_buffer")
                    else k_data
                )
                v_buf = (
                    v_data.to_buffer()
                    if hasattr(v_data, "to_buffer")
                    else v_data
                )
                k_f32 = cast_dlpack_to(
                    k_buf, k_data.dtype, DType.float32, CPU()
                )
                v_f32 = cast_dlpack_to(
                    v_buf, v_data.dtype, DType.float32, CPU()
                )
                k_np = np.from_dlpack(k_f32)
                v_np = np.from_dlpack(v_f32)
                kv_np = np.ascontiguousarray(
                    np.concatenate([k_np, v_np], axis=0)
                )
                state_dict[kv_key] = kv_np
                fused_keys.add(k_key)
                fused_keys.add(v_key)

    for key, tensor in raw_dict.items():
        if key not in fused_keys:
            state_dict[key] = tensor

    cpu_device = CPU()
    for key, tensor in state_dict.items():
        if isinstance(tensor, WeightData):
            src_dtype = tensor.dtype
            dlpack_obj = tensor.to_buffer()
        else:
            src_dtype = DType.float32
            dlpack_obj = tensor
        state_dict[key] = cast_dlpack_to(
            dlpack_obj, src_dtype, target_dtype, cpu_device
        )

    return state_dict


def _get_1d_rotary_pos_embed_np(
    dim: int,
    pos: np.ndarray,
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1D rotary position embeddings (numpy, for eager RoPE)."""
    freq_exponent = np.arange(0, dim, 2, dtype=np.float64) / dim
    freqs = 1.0 / (theta**freq_exponent)
    angles = np.outer(pos.astype(np.float64), freqs)
    cos_emb = np.cos(angles).astype(np.float32)
    sin_emb = np.sin(angles).astype(np.float32)
    cos_emb = np.repeat(cos_emb, 2, axis=1)
    sin_emb = np.repeat(sin_emb, 2, axis=1)
    return cos_emb, sin_emb


@lru_cache(maxsize=8)
def _compute_wan_rope_cached(
    num_frames: int,
    height: int,
    width: int,
    patch_size: tuple[int, int, int],
    head_dim: int,
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 3D RoPE cos/sin arrays for Wan transformer (cached by resolution)."""
    p_t, p_h, p_w = patch_size
    ppf = num_frames // p_t
    pph = height // p_h
    ppw = width // p_w

    d_h = (head_dim // 3 // 2) * 2
    d_w = d_h
    d_t = head_dim - d_h - d_w

    cos_t, sin_t = _get_1d_rotary_pos_embed_np(d_t, np.arange(ppf), theta)
    cos_h, sin_h = _get_1d_rotary_pos_embed_np(d_h, np.arange(pph), theta)
    cos_w, sin_w = _get_1d_rotary_pos_embed_np(d_w, np.arange(ppw), theta)

    cos_t = np.broadcast_to(cos_t[:, None, None, :], (ppf, pph, ppw, d_t))
    sin_t = np.broadcast_to(sin_t[:, None, None, :], (ppf, pph, ppw, d_t))
    cos_h = np.broadcast_to(cos_h[None, :, None, :], (ppf, pph, ppw, d_h))
    sin_h = np.broadcast_to(sin_h[None, :, None, :], (ppf, pph, ppw, d_h))
    cos_w = np.broadcast_to(cos_w[None, None, :, :], (ppf, pph, ppw, d_w))
    sin_w = np.broadcast_to(sin_w[None, None, :, :], (ppf, pph, ppw, d_w))

    rope_cos = np.concatenate([cos_t, cos_h, cos_w], axis=-1)
    rope_sin = np.concatenate([sin_t, sin_h, sin_w], axis=-1)

    seq_len = ppf * pph * ppw
    rope_cos = np.ascontiguousarray(rope_cos.reshape(seq_len, head_dim))
    rope_sin = np.ascontiguousarray(rope_sin.reshape(seq_len, head_dim))
    return rope_cos, rope_sin


class BlockLevelModel:
    """Executes transformer forward pass as pre -> combined blocks -> post.

    All transformer blocks are compiled into a single ``Model`` graph,
    so the runtime allocates one shared workspace.
    """

    def __init__(
        self,
        pre: Model,
        post: Model,
        *,
        combined_blocks: Model,
    ) -> None:
        self.pre = pre
        self.post = post
        self.combined_blocks = combined_blocks

    def __call__(
        self,
        hidden_states: Buffer,
        timestep: Buffer,
        encoder_hidden_states: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
        i2v_condition: Buffer | None = None,
    ) -> Buffer:
        pre_args = [hidden_states, timestep, encoder_hidden_states]
        if i2v_condition is not None:
            pre_args.append(i2v_condition)
        with Tracer("dit_pre"):
            pre_out = self.pre.execute(*pre_args)
            hs, temb, timestep_proj, text_emb = (
                pre_out[0],
                pre_out[1],
                pre_out[2],
                pre_out[3],
            )
        with Tracer("dit_blocks"):
            block_out = self.combined_blocks.execute(
                hs, text_emb, timestep_proj, rope_cos, rope_sin
            )
            hs = block_out[0]
        with Tracer("dit_post"):
            post_out = self.post.execute(hs, temb, spatial_shape)
        return post_out[0]
