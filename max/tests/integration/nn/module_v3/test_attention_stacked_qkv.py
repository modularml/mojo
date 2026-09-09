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
"""QKV projection tests for ModuleV3 ``AttentionWithRope``."""

from __future__ import annotations

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.experimental.nn import TransparentModule
from max.experimental.nn.common_layers.attention import AttentionWithRope
from max.experimental.nn.common_layers.linear import QKVLinear
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.tensor import Tensor, default_dtype
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams

HEAD_DIM = 16
N_HEADS = 4
N_KV_HEADS = 2  # GQA: q_dim != kv_dim.
HIDDEN_SIZE = 64
Q_WEIGHT_DIM = HEAD_DIM * N_HEADS
KV_WEIGHT_DIM = HEAD_DIM * N_KV_HEADS
OUT_DIM = Q_WEIGHT_DIM + 2 * KV_WEIGHT_DIM
MAX_SEQ_LEN = 32
TOKENS = 8


def _make_attention(
    *,
    stacked_qkv: bool = False,
    clip_qkv: float | None = None,
    has_bias: bool = False,
) -> AttentionWithRope:
    rope = RotaryEmbedding(
        dim=HIDDEN_SIZE,
        n_heads=N_HEADS,
        theta=10000.0,
        max_seq_len=MAX_SEQ_LEN,
        device=CPU(),
    )
    kv_params = MHAKVCacheParams(
        dtype=DType.float32,
        head_dim=HEAD_DIM,
        n_kv_heads=N_KV_HEADS,
        num_layers=1,
        devices=[DeviceRef.CPU()],
    )
    with default_dtype(DType.float32):
        attn = AttentionWithRope(
            rope=rope,
            num_attention_heads=N_HEADS,
            num_key_value_heads=N_KV_HEADS,
            hidden_size=HIDDEN_SIZE,
            kv_params=kv_params,
            layer_idx=0,
            has_bias=has_bias,
            stacked_qkv=stacked_qkv,
            clip_qkv=clip_qkv,
        )
    return attn.to(CPU())


def test_unfused_exposes_native_projection_names() -> None:
    attn = _make_attention(stacked_qkv=False)
    assert isinstance(attn.qkv_proj, QKVLinear)
    assert isinstance(attn.qkv_proj, TransparentModule)
    assert attn.qkv_proj.name_transparent is True
    names = {name for name, _ in attn.parameters}
    assert "q_proj.weight" in names
    assert "k_proj.weight" in names
    assert "v_proj.weight" in names
    assert "o_proj.weight" in names
    assert not any(n.startswith("qkv_proj.") for n in names)


def test_stacked_exposes_fused_projection_name() -> None:
    attn = _make_attention(stacked_qkv=True)
    assert isinstance(attn.qkv_proj, QKVLinear)
    assert attn.qkv_proj.name_transparent is False
    names = {name for name, _ in attn.parameters}
    assert "qkv_proj.weight" in names
    assert "q_proj.weight" not in names


def test_projection_parity_unfused() -> None:
    """``qkv_proj(x)`` equals a single matmul over the concatenated q||k||v."""
    attn = _make_attention(stacked_qkv=False)
    qkv = attn.qkv_proj
    rng = np.random.default_rng(0)
    q = rng.standard_normal((Q_WEIGHT_DIM, HIDDEN_SIZE)).astype(np.float32)
    k = rng.standard_normal((KV_WEIGHT_DIM, HIDDEN_SIZE)).astype(np.float32)
    v = rng.standard_normal((KV_WEIGHT_DIM, HIDDEN_SIZE)).astype(np.float32)
    qkv.q_proj.weight = Tensor(q, device=CPU())
    qkv.k_proj.weight = Tensor(k, device=CPU())
    qkv.v_proj.weight = Tensor(v, device=CPU())

    x_np = rng.standard_normal((TOKENS, HIDDEN_SIZE)).astype(np.float32)
    out = qkv(Tensor(x_np, device=CPU())).to_numpy()
    expected = x_np @ np.concatenate([q, k, v], axis=0).T
    assert out.shape == (TOKENS, OUT_DIM)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_clip_qkv_folds_and_commutes_with_concat() -> None:
    """``wqkv`` clamps the fused weight; equals concatenating clamped q/k/v."""
    clip = 0.05
    attn = _make_attention(stacked_qkv=False, clip_qkv=clip)
    qkv = attn.qkv_proj
    rng = np.random.default_rng(1)
    q = rng.standard_normal((Q_WEIGHT_DIM, HIDDEN_SIZE)).astype(np.float32)
    k = rng.standard_normal((KV_WEIGHT_DIM, HIDDEN_SIZE)).astype(np.float32)
    v = rng.standard_normal((KV_WEIGHT_DIM, HIDDEN_SIZE)).astype(np.float32)
    qkv.q_proj.weight = Tensor(q, device=CPU())
    qkv.k_proj.weight = Tensor(k, device=CPU())
    qkv.v_proj.weight = Tensor(v, device=CPU())

    wqkv = attn.wqkv.to_numpy()
    # clamp(concat) == concat(clamp q, clamp k, clamp v).
    clamped_concat = np.clip(np.concatenate([q, k, v], axis=0), -clip, clip)
    concat_clamped = np.concatenate(
        [
            np.clip(q, -clip, clip),
            np.clip(k, -clip, clip),
            np.clip(v, -clip, clip),
        ],
        axis=0,
    )
    np.testing.assert_array_equal(wqkv, clamped_concat)
    np.testing.assert_array_equal(wqkv, concat_clamped)
