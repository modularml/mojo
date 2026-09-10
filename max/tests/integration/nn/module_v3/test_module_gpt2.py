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
"""Tests for max.nn.Module."""

from __future__ import annotations

import asyncio
from typing import Generic, TypeVar

import pytest
from max.driver import CPU, Accelerator, Device, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import (
    Embedding,
    Linear,
    Module,
    Sequential,
    module_dataclass,
)
from max.experimental.tensor import Tensor, TensorType
from max.graph import DeviceRef, Dim, DimLike


# Tricky: because this function adds two dimensions, it needs
#   a valid MLIR context and so the whole function has to be
#   wrapped in @F.functional.
@F.functional
def causal_mask(  # noqa: ANN201
    sequence_length: DimLike,
    num_tokens: DimLike,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
):
    n = Dim(sequence_length) + num_tokens
    mask = Tensor(float("-inf"), dtype=dtype, device=device)
    mask = F.broadcast_to(mask, shape=(sequence_length, n))
    return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)


class MultiHeadAttention(Module[[Tensor], Tensor]):
    def __init__(
        self,
        in_dim: DimLike,
        out_dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = False,
    ):
        assert out_dim % num_heads == 0, (
            "out_dim must be divisible by num_heads"
        )
        # Reduce the projection dim to match desired output dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.query = Linear(in_dim, out_dim, bias=qkv_bias)
        self.key = Linear(in_dim, out_dim, bias=qkv_bias)
        self.value = Linear(in_dim, out_dim, bias=qkv_bias)
        self.out_proj = Linear(out_dim, out_dim)

    @property
    def in_dim(self):  # noqa: ANN201
        return self.query.in_dim

    @property
    def out_dim(self):  # noqa: ANN201
        return self.query.out_dim

    def forward(self, x: Tensor) -> Tensor:
        b, num_tokens, _in_dim = x.shape

        keys = self.key(x)  # Shape: (b, num_tokens, out_dim)
        queries = self.query(x)
        values = self.value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, out_dim) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.reshape((b, num_tokens, self.num_heads, self.head_dim))
        values = values.reshape((b, num_tokens, self.num_heads, self.head_dim))
        queries = queries.reshape(
            (b, num_tokens, self.num_heads, self.head_dim)
        )

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)

        # Use the mask to fill attention scores
        # Mask is -inf for masked tokens, 0 for unmasked tokens
        mask = causal_mask(num_tokens, 0, dtype=x.dtype, device=x.device)
        attn_scores += mask

        # TODO(GEX-2632): Support Dim.__pow__
        attn_weights = F.softmax(attn_scores / int(self.head_dim) ** 0.5)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.out_dim = self.num_heads * self.head_dim
        context_vec = context_vec.reshape((b, num_tokens, self.out_dim))
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class LayerNorm(Module[[Tensor], Tensor]):
    def __init__(self, dim: DimLike, *, eps: float = 1e-5):
        self.eps = eps
        self.scale = Tensor.ones([dim])
        self.shift = Tensor.zeros([dim])

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x, gamma=self.scale, beta=self.shift, epsilon=self.eps
        )


@module_dataclass
class GeLU(Module[[Tensor], Tensor]):
    approximate: str = "tanh"

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x, approximate=self.approximate)


class FeedForward(Module[[Tensor], Tensor]):
    def __init__(self, dim: int):
        self.layers = Sequential(
            Linear(dim, 4 * dim),
            GeLU(),
            Linear(4 * dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


M = TypeVar("M", bound=Module[[Tensor], Tensor])


@module_dataclass
class Highway(Module[[Tensor], Tensor], Generic[M]):
    norm: LayerNorm
    layer: M

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layer(self.norm(x))


class TransformerBlock(Module[[Tensor], Tensor]):
    attention: Highway[MultiHeadAttention]
    feed_forward: Highway[FeedForward]

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = False,
    ):
        self.attention = Highway(
            LayerNorm(dim),
            MultiHeadAttention(dim, dim, num_heads, qkv_bias=qkv_bias),
        )
        self.feed_forward = Highway(LayerNorm(dim), FeedForward(dim))

    def forward(self, x: Tensor) -> Tensor:
        return Sequential(
            self.attention,
            self.feed_forward,
        )(x)


class GPTModel(Module[[Tensor], Tensor]):
    def __init__(
        self,
        vocab_size: DimLike,
        dim: int,
        num_layers: int,
        num_heads: int,
        context_length: DimLike,
        *,
        qkv_bias: bool = False,
    ):
        self.token_embedding = Embedding(vocab_size, dim=dim)
        self.positional_embedding = Embedding(context_length, dim=dim)

        self.layers = Sequential(
            *(
                TransformerBlock(
                    dim,
                    num_heads,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            )
        )

        self.norm = LayerNorm(dim)
        self.out_head = Linear(dim, vocab_size, bias=False)

    @property
    def vocab_size(self) -> Dim:
        return self.token_embedding.vocab_size

    @property
    def dim(self) -> Dim:
        return self.token_embedding.dim

    def forward(self, in_idx: Tensor) -> Tensor:
        tok_embeds = self.token_embedding(in_idx)
        pos_embeds = self.positional_embedding(Tensor.range_like(in_idx))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        return Sequential(
            self.layers,
            self.norm,
            self.out_head,
        )(x)


@pytest.fixture
def device():  # noqa: ANN201
    yield Accelerator() if accelerator_count() else CPU()


@pytest.fixture
def gpt_model(device: Device):  # noqa: ANN201
    with F.lazy():
        model = GPTModel(
            dim=64,
            num_layers=2,
            num_heads=2,
            vocab_size=50257,
            context_length=1024,
            qkv_bias=True,
        ).to(target=device)

    yield model


def test_gpt2_repr(gpt_model: GPTModel) -> None:
    _ = repr(gpt_model)


def test_gpt2_eager(gpt_model: GPTModel, device: Device) -> None:
    input = Tensor.zeros([1, 1], dtype=DType.int64, device=device)

    # Eagerly use model, submodule
    results = gpt_model(input)
    embedded_tokens = gpt_model.token_embedding(input)

    assert isinstance(results, Tensor)
    assert results.shape == [1, 1, gpt_model.token_embedding.vocab_size]
    assert isinstance(embedded_tokens, Tensor)
    assert embedded_tokens.shape == [1, 1, gpt_model.token_embedding.dim]

    asyncio.run(results.realize)
    asyncio.run(embedded_tokens.realize)

    assert results.real
    assert embedded_tokens.real


def test_gpt2_compiled(gpt_model: GPTModel, device: Device) -> None:
    # Compile to inference graph
    token_type = TensorType(
        DType.int64, ("batch", "seqlen"), device=DeviceRef.from_device(device)
    )
    compiled = gpt_model.compile(token_type)
    input = Tensor.zeros([1, 1], dtype=DType.int64, device=device)
    results = compiled(input)
    assert isinstance(results, Tensor)
    assert results.real
    assert results.shape == [1, 1, gpt_model.token_embedding.vocab_size]
