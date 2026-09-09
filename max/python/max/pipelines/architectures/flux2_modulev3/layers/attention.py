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
"""FLUX.2 ModuleV3 attention and feed-forward layers.

Mojo kernels (``rope_ragged_with_position_ids``, ``flash_attention_gpu``) are
reused from ``max.nn.kernels`` via :class:`~max.graph.TensorValue` unwrap;
they take graph values directly and the experimental ``Tensor`` exposes
``__tensorvalue__`` and :meth:`Tensor.from_graph_value` as the unwrap /
rewrap pair.
"""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu, rope_ragged_with_position_ids


def _apply_flux2_qk_rope(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply FLUX.2 rotary embeddings to ``query`` and ``key``.

    Reuses the legacy ``rope_ragged_with_position_ids`` Mojo kernel.  The
    experimental tensors are unwrapped to :class:`TensorValue` for the
    kernel call and rewrapped on the way out.

    Args:
        query: Query tensor of shape ``[B, S, H, D]``.
        key: Key tensor of shape ``[B, S, H, D]``.
        cos: Rotary cosine of shape ``[S, D]``, repeat-interleaved.
        sin: Rotary sine of shape ``[S, D]``, repeat-interleaved.

    Returns:
        Tuple ``(query, key)`` with the same shapes and dtypes as the
        inputs.
    """
    batch_size = query.shape[0]
    seq_len = query.shape[1]
    num_heads = query.shape[2]
    head_dim = query.shape[3]

    query_ragged = query.reshape([batch_size * seq_len, num_heads, head_dim])
    key_ragged = key.reshape([batch_size * seq_len, num_heads, head_dim])

    # Convert repeat-interleaved ([cos, cos], [sin, sin]) to [cos, sin]
    # pairs.  The kernel expects packed (cos, sin) pairs along the last
    # axis; the legacy implementation does the same conversion.
    cos_pairs = cos.reshape([cos.shape[0], cos.shape[1] // 2, 2])[..., 0]
    sin_pairs = sin.reshape([sin.shape[0], sin.shape[1] // 2, 2])[..., 0]
    freqs_cis = F.stack([cos_pairs, sin_pairs], axis=-1).reshape(
        [cos.shape[0], cos.shape[1]]
    )

    position_ids = F.range(
        0, seq_len, 1, dtype=DType.uint32, device=query.device
    )
    # broadcast_to instead of tile: tile has no GPU kernel and forces a
    # CPU round-trip. broadcast_to expands [1, seq_len] -> [batch_size,
    # seq_len] entirely on GPU.
    position_ids = F.broadcast_to(
        position_ids.unsqueeze(0), [batch_size, seq_len]
    )
    position_ids = position_ids.reshape([batch_size * seq_len])

    query_tv = query_ragged.__tensorvalue__()
    key_tv = key_ragged.__tensorvalue__()
    freqs_cis_tv = freqs_cis.__tensorvalue__()
    position_ids_tv = position_ids.__tensorvalue__()

    query_out_tv = rope_ragged_with_position_ids(
        query_tv,
        freqs_cis_tv,
        position_ids_tv,
        interleaved=True,
    )
    key_out_tv = rope_ragged_with_position_ids(
        key_tv,
        freqs_cis_tv,
        position_ids_tv,
        interleaved=True,
    )

    query_out = Tensor.from_graph_value(query_out_tv).reshape(
        [batch_size, seq_len, num_heads, head_dim]
    )
    key_out = Tensor.from_graph_value(key_out_tv).reshape(
        [batch_size, seq_len, num_heads, head_dim]
    )
    return query_out, key_out


def _flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float,
) -> Tensor:
    """Run ``flash_attention_gpu`` with a NULL mask.

    Bridges the experimental ``Tensor`` API to the legacy Mojo kernel
    via :class:`TensorValue` unwrap / rewrap.
    """
    query_tv = query.__tensorvalue__()
    key_tv = key.__tensorvalue__()
    value_tv = value.__tensorvalue__()
    out_tv = flash_attention_gpu(
        query_tv,
        key_tv,
        value_tv,
        mask_variant=MHAMaskVariant.NULL_MASK,
        scale=scale,
    )
    return Tensor.from_graph_value(out_tv)


class Flux2SwiGLU(Module[[Tensor], Tensor]):
    """SwiGLU activation: chunks input in half and gates with ``silu``.

    Stateless; no parameters.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Return ``silu(x1) * x2`` where ``x1, x2 = chunk(x, 2, -1)``."""
        x1, x2 = F.chunk(x, chunks=2, axis=-1)
        return F.silu(x1) * x2


class Flux2FeedForward(Module[[Tensor], Tensor]):
    """Two-Linear feed-forward block with SwiGLU activation.

    Matches the legacy
    :class:`max.pipelines.architectures.flux2.layers.flux2_attention.Flux2FeedForward`
    single-device path.  Sharding/quantization wiring is out of scope
    for the single-GPU + BF16 first port.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 3.0,
        inner_dim: int | None = None,
        bias: bool = False,
    ) -> None:
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.dim = dim
        self.dim_out = dim_out
        self.inner_dim = inner_dim
        self.has_bias = bias

        # ``inner_dim * 2`` because the SwiGLU activation chunks the
        # output in half along the channel axis.
        self.linear_in = Linear(dim, inner_dim * 2, bias=bias)
        self.act_fn = Flux2SwiGLU()
        self.linear_out = Linear(inner_dim, dim_out, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply ``linear_in -> SwiGLU -> linear_out``."""
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x


class Flux2Modulation(
    Module[[Tensor], tuple[tuple[Tensor, Tensor, Tensor], ...]]
):
    """Projects ``temb`` into ``mod_param_sets`` modulation triples.

    Each triple is ``(shift, scale, gate)``; FLUX.2 uses two triples
    per dual-stream block (image + text) and one triple per single-stream
    block.
    """

    def __init__(
        self,
        dim: int,
        *,
        mod_param_sets: int = 2,
        bias: bool = False,
    ) -> None:
        self.mod_param_sets = mod_param_sets
        self.dim = dim
        self.linear = Linear(dim, dim * 3 * mod_param_sets, bias=bias)

    def forward(
        self, temb: Tensor
    ) -> tuple[tuple[Tensor, Tensor, Tensor], ...]:
        """Return ``mod_param_sets`` (shift, scale, gate) tuples.

        Args:
            temb: Time / guidance embedding, shape ``[B, dim]`` or
                ``[B, 1, dim]``.

        Returns:
            Tuple of length ``mod_param_sets``; each element is a triple
            ``(shift, scale, gate)`` with shape ``[B, 1, dim]``.
        """
        mod = self.linear(F.silu(temb))
        if len(mod.shape) == 2:
            mod = mod.unsqueeze(1)
        mod_params = mod.split(
            [self.dim] * (3 * self.mod_param_sets),
            axis=-1,
        )
        return tuple(
            (mod_params[3 * i], mod_params[3 * i + 1], mod_params[3 * i + 2])
            for i in range(self.mod_param_sets)
        )


class Flux2Attention(Module[..., object]):
    """Dual-stream joint image+text attention with optional encoder projections.

    Mirrors the legacy
    :class:`max.pipelines.architectures.flux2.layers.flux2_attention.Flux2Attention`
    single-device path.  When ``added_kv_proj_dim`` is set, separate
    Q/K/V/out projections are created for the text stream; otherwise
    the layer runs single-stream.

    The returned tuple is ``(hidden_out, encoder_out)`` in the dual-stream
    case and a single ``Tensor`` otherwise.  Single-device only;
    tensor-parallel sharding is out of scope here.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
    ) -> None:
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        out_dim = out_dim if out_dim is not None else query_dim

        self.to_q = Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = Linear(query_dim, self.inner_dim, bias=bias)
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        # Legacy stores the single output projection inside a
        # ``LayerList`` so checkpoints carry ``to_out.0.weight``.  Match
        # that layout via a single-element ``ModuleList`` here.
        self.to_out = ModuleList[Linear](
            [Linear(self.inner_dim, out_dim, bias=out_bias)]
        )

        self.norm_added_q: RMSNorm | None
        self.norm_added_k: RMSNorm | None
        self.add_q_proj: Linear | None
        self.add_k_proj: Linear | None
        self.add_v_proj: Linear | None
        self.to_add_out: Linear | None

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)
            proj_bias = False if added_proj_bias is None else added_proj_bias
            self.add_q_proj = Linear(
                added_kv_proj_dim, self.inner_dim, bias=proj_bias
            )
            self.add_k_proj = Linear(
                added_kv_proj_dim, self.inner_dim, bias=proj_bias
            )
            self.add_v_proj = Linear(
                added_kv_proj_dim, self.inner_dim, bias=proj_bias
            )
            self.to_add_out = Linear(self.inner_dim, query_dim, bias=out_bias)
        else:
            self.norm_added_q = None
            self.norm_added_k = None
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None
            self.to_add_out = None

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> object:
        """Apply joint image+text attention.

        Args:
            hidden_states: Image tokens of shape ``[B, S_img, D]``.
            encoder_hidden_states: Optional text tokens of shape
                ``[B, S_txt, D_enc]``.  When provided (and
                ``added_kv_proj_dim`` is set), enables dual-stream mode.
            image_rotary_emb: Optional ``(cos, sin)`` rotary embeddings
                applied jointly to Q/K.

        Returns:
            Tensor ``[B, S_img, out_dim]`` if ``encoder_hidden_states``
            is None; otherwise a tuple ``(hidden_out, encoder_out)``.
        """
        batch_size = hidden_states.shape[0]
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        seq_len = query.shape[1]
        query = query.reshape([batch_size, seq_len, self.heads, self.head_dim])
        key = key.reshape([batch_size, seq_len, self.heads, self.head_dim])
        value = value.reshape([batch_size, seq_len, self.heads, self.head_dim])

        query = self.norm_q(query)
        key = self.norm_k(key)

        if (
            encoder_hidden_states is not None
            and self.added_kv_proj_dim is not None
        ):
            if (
                self.add_q_proj is None
                or self.add_k_proj is None
                or self.add_v_proj is None
            ):
                raise ValueError("Encoder projections are not initialized")
            encoder_query = self.add_q_proj(encoder_hidden_states)
            encoder_key = self.add_k_proj(encoder_hidden_states)
            encoder_value = self.add_v_proj(encoder_hidden_states)
            encoder_seq_len = encoder_query.shape[1]
            encoder_query = encoder_query.reshape(
                [batch_size, encoder_seq_len, self.heads, self.head_dim]
            )
            encoder_key = encoder_key.reshape(
                [batch_size, encoder_seq_len, self.heads, self.head_dim]
            )
            encoder_value = encoder_value.reshape(
                [batch_size, encoder_seq_len, self.heads, self.head_dim]
            )
            if self.norm_added_q is None or self.norm_added_k is None:
                raise ValueError("Encoder normalizations not initialized")
            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = F.concat([encoder_query, query], axis=1)
            key = F.concat([encoder_key, key], axis=1)
            value = F.concat([encoder_value, value], axis=1)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query, key = _apply_flux2_qk_rope(query, key, cos, sin)

        scale = 1.0 / (self.head_dim**0.5)
        hidden_states = _flash_attention(query, key, value, scale=scale)

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.reshape(
            [batch_size, seq_len, self.inner_dim]
        )
        hidden_states = hidden_states.cast(query.dtype)

        if encoder_hidden_states is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_out = hidden_states[:, :encoder_seq_len, :]
            hidden_out = hidden_states[:, encoder_seq_len:, :]

            hidden_out = self.to_out[0](hidden_out)
            if self.to_add_out is None:
                raise ValueError("Encoder output projection is not initialized")
            encoder_out = self.to_add_out(encoder_out)
            return hidden_out, encoder_out

        return self.to_out[0](hidden_states)


class Flux2ParallelSelfAttention(Module[..., Tensor]):
    """Single-stream parallel self-attention + MLP block.

    Mirrors the legacy
    :class:`max.pipelines.architectures.flux2.layers.flux2_attention.Flux2ParallelSelfAttention`
    single-device path: a single fused ``[Q | K | V | gate | up]``
    input projection feeds both the attention head and the SwiGLU MLP
    in parallel, and a single fused ``[attn_out | mlp_out]`` output
    projection produces the residual.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
        mlp_ratio: float = 4.0,
        mlp_mult_factor: int = 2,
    ) -> None:
        if mlp_mult_factor != 2:
            raise ValueError(
                "Flux2ParallelSelfAttention only supports "
                f"mlp_mult_factor=2 (SwiGLU expects two chunks); got "
                f"{mlp_mult_factor}"
            )
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        out_dim = out_dim if out_dim is not None else query_dim

        self.mlp_hidden_dim = int(query_dim * mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor

        fused_dim = self.inner_dim * 3 + self.mlp_hidden_dim * mlp_mult_factor
        self.to_qkv_mlp_proj = Linear(query_dim, fused_dim, bias=bias)
        self.mlp_act_fn = Flux2SwiGLU()
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.to_out = Linear(
            self.inner_dim + self.mlp_hidden_dim, out_dim, bias=out_bias
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Apply parallel self-attention and MLP, return fused residual."""
        del attention_mask  # not supported; kept to match legacy signature.

        fused = self.to_qkv_mlp_proj(hidden_states)
        qkv_dim = self.inner_dim * 3
        mlp_dim = self.mlp_hidden_dim * self.mlp_mult_factor
        qkv, mlp_hidden_states = fused.split([qkv_dim, mlp_dim], axis=-1)

        query, key, value = F.chunk(qkv, 3, axis=-1)
        query = query.reshape(
            [query.shape[0], query.shape[1], self.heads, self.head_dim]
        )
        key = key.reshape(
            [key.shape[0], key.shape[1], self.heads, self.head_dim]
        )
        value = value.reshape(
            [value.shape[0], value.shape[1], self.heads, self.head_dim]
        )

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query, key = _apply_flux2_qk_rope(query, key, cos, sin)

        hidden_states = _flash_attention(
            query, key, value, scale=1.0 / (self.head_dim**0.5)
        )

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.reshape(
            [batch_size, seq_len, self.inner_dim]
        )
        hidden_states = hidden_states.cast(query.dtype)

        mlp_hidden_states = self.mlp_act_fn(mlp_hidden_states)
        hidden_states = F.concat([hidden_states, mlp_hidden_states], axis=-1)
        return self.to_out(hidden_states)
