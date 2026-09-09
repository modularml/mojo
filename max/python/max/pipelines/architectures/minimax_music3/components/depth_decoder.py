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
"""The local transformer that predicts a frame's seven residual RVQ codebooks.

Every audio frame carries eight codes: one semantic code, sampled by the global
language model, and seven residual ones sampled here, each conditioned on the
codes already chosen. So this runs eight times per frame -- 200 times a second of
audio -- over a sequence that is never longer than eight tokens.

Its per-step hidden states are not a by-product: concatenated with the global
model's, they *are* the conditioning the diffusion stage denoises against.

Two things about the shape of the computation. It is **cacheless**: the reference
re-runs the whole prefix at every depth step rather than keeping a KV cache, which
for a sequence of at most eight tokens costs less than the cache bookkeeping would.
And attention is **causal**, which is not a detail that cancels out even though
only the last position is read -- the earlier positions feed the next layer.
"""

from __future__ import annotations

import math

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Embedding, Linear, Module, ModuleList, RMSNorm
from max.experimental.tensor import Tensor

from ..model_config import DepthDecoderConfig

# The reference normalizes in float32 and rounds *before* applying the weight,
# which is `ops.rms_norm`'s -- and so the experimental `RMSNorm`'s -- default.
NORM_EPS = 1e-6


class Attention(Module[[Tensor], Tensor]):
    """Causal self-attention over the depth sequence.

    Written out of primitives rather than dispatched to the fused kernel: the
    sequence is at most eight tokens, so the launch and mask bookkeeping cost
    more than the arithmetic, and the scores are cheap to take in float32.
    """

    def __init__(self, dim: int, heads: int) -> None:
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = Linear(dim, dim, bias=False)
        self.to_k = Linear(dim, dim, bias=False)
        self.to_v = Linear(dim, dim, bias=False)
        self.to_out = Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq = int(x.shape[0]), int(x.shape[1])
        shape = [batch, seq, self.heads, self.head_dim]

        def heads_first(projection: Tensor) -> Tensor:
            return projection.reshape(shape).transpose(1, 2)

        query = heads_first(self.to_q(x)).cast(DType.float32)
        key = heads_first(self.to_k(x)).cast(DType.float32)
        value = heads_first(self.to_v(x))

        scores = query @ F.transpose(key, -1, -2) / math.sqrt(self.head_dim)
        # Additive rather than a select, so the softmax's own max subtraction
        # handles the masked positions. `exclude` inverts the band, so the
        # infinities land strictly above the diagonal and the rest is zero.
        mask = F.band_part(
            F.full(
                [seq, seq],
                float("-inf"),
                dtype=DType.float32,
                device=x.device,
            ),
            num_upper=0,
            exclude=True,
        )
        scores = scores + mask
        weights = F.softmax(scores).cast(x.dtype)

        attended = F.transpose(weights @ value, 1, 2)
        return self.to_out(
            F.reshape(attended, [batch, seq, self.heads * self.head_dim])
        )


class DecoderBlock(Module[[Tensor], Tensor]):
    """Pre-norm attention and a SwiGLU feed-forward, both residual."""

    def __init__(self, dim: int, heads: int, intermediate_size: int) -> None:
        self.input_layernorm = RMSNorm(dim, eps=NORM_EPS)
        self.attn = Attention(dim, heads)
        self.post_attention_layernorm = RMSNorm(dim, eps=NORM_EPS)
        self.gate_proj = Linear(dim, intermediate_size, bias=False)
        self.up_proj = Linear(dim, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.input_layernorm(x))
        normed = self.post_attention_layernorm(x)
        gated = F.silu(self.gate_proj(normed)) * self.up_proj(normed)
        return x + self.down_proj(gated)


class DepthDecoder(Module[..., Tensor]):
    """Predicts one frame's residual codebooks, one step at a time.

    Owns both of the embedding tables a frame's codes are looked up in, and the
    shared projection they pass through. The semantic one belongs to the global
    model's vocabulary rather than to this checkpoint, and it lives here because
    this is where it is read: nothing else on the device needs it, and having
    both lookups in one place is what lets the frame's feedback embedding be
    computed without a round trip to the host.
    """

    def __init__(
        self, config: DepthDecoderConfig, semantic_vocab_size: int
    ) -> None:
        """Builds the decoder and its tables.

        Args:
            config: This checkpoint's shape.
            semantic_vocab_size: Rows of the global model's embedding table
                holding semantic codes. Only that slice is loaded, so a code is
                its own row index and the checkpoint's offset disappears.
        """
        self.config = config
        residual_codebooks = config.num_codebooks - 1

        self.audio_embeddings = Embedding(
            config.audio_vocab_size * residual_codebooks,
            dim=config.hidden_size,
        )
        self.semantic_embeddings = Embedding(
            semantic_vocab_size, dim=config.hidden_size
        )
        self.projection = Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.pos_embedding = Embedding(
            config.max_position_embeddings, dim=config.hidden_size
        )
        self.layers = ModuleList(
            [
                DecoderBlock(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.intermediate_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=NORM_EPS)
        self.audio_heads = ModuleList(
            [
                Linear(config.hidden_size, config.audio_vocab_size, bias=False)
                for _ in range(residual_codebooks)
            ]
        )

    @property
    def dtype(self) -> DType:
        """The dtype the decoder computes in, taken from its parameters."""
        return self.projection.weight.dtype

    def embed_codes(self, codes: Tensor) -> Tensor:
        """Embeds residual codes ``c1..cn``, one row per depth position.

        The table holds all seven codebooks end to end, so a code at depth
        ``j`` is offset by ``j * audio_vocab_size``. Getting the offset wrong
        reads a different codebook's vectors and still produces audio.

        Args:
            codes: ``(batch, n)`` integer codes, in depth order from ``c1``.

        Returns:
            ``(batch, n, hidden_size)`` embeddings.
        """
        offsets = (
            F.arange(
                0,
                codes.shape[1],
                dtype=DType.int32,
                device=codes.device,
            )
            * self.config.audio_vocab_size
        )
        return self.audio_embeddings(codes + offsets)

    def sequence(self, hidden: Tensor, codes: Tensor) -> Tensor:
        """Builds the inputs to one frame's depth sequence.

        Position 0 carries the global model's state for the frame and position
        ``k`` carries code ``c(k-1)``, so position ``k``'s output is what
        predicts ``c(k)``. A sequence built from the frame's first
        ``num_codebooks - 1`` codes is therefore exactly long enough to predict
        the last one, and because attention is causal the positions past the step
        being read do not reach it -- the loop can leave them zero and take one
        step per call off a single fixed-length graph.

        Args:
            hidden: ``(batch, hidden_size)`` from the global model.
            codes: ``(batch, num_codebooks - 1)`` holding ``c0..c6``. Entries
                past the step being decoded are ignored, but every entry is
                still gathered, so all of them must be in range.

        Returns:
            ``(batch, num_codebooks, hidden_size)`` projected inputs.
        """
        return self.projection(
            F.concat(
                [
                    F.unsqueeze(hidden, 1),
                    self.semantic_embeddings(codes[:, :1]),
                    self.embed_codes(codes[:, 1:]),
                ],
                axis=1,
            )
        )

    def feedback(self, codes: Tensor) -> Tensor:
        """Embeds a finished frame into the vector the global model reads next.

        A frame's codes describe one sound together, so their embeddings are
        summed rather than sequenced, then scaled by ``num_codebooks ** -0.5``
        to put the sum back on the scale of the global model's other inputs.

        Args:
            codes: ``(batch, num_codebooks)`` holding ``c0..c7``.

        Returns:
            ``(batch, 1, hidden_size)``, the next step's input embedding.
        """
        summed = self.semantic_embeddings(codes[:, :1]) + F.sum(
            self.embed_codes(codes[:, 1:]), axis=1
        )
        scale = F.constant(
            self.config.num_codebooks**-0.5,
            summed.dtype,
            device=summed.device,
        )
        return summed * scale

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        """Runs the depth sequence and normalizes every position's output.

        Args:
            inputs_embeds: ``(batch, steps, hidden_size)``, already projected.

        Returns:
            Normalized hidden states, shaped like ``inputs_embeds``. The last
            step's is what feeds the next codebook's head.
        """
        positions = self.pos_embedding(
            F.arange(
                0,
                inputs_embeds.shape[1],
                dtype=DType.int32,
                device=inputs_embeds.device,
            )
        )
        x = inputs_embeds + F.unsqueeze(positions, 0)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
