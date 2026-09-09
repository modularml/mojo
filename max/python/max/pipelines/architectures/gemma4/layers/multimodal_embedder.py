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

"""Gemma4 Multimodal Embedder: projects vision (or audio) tokens into LM space."""

from __future__ import annotations

from collections.abc import Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue
from max.nn.layer import Module
from max.nn.linear import Linear
from max.pipelines.architectures.gemma4.layers.rms_norm import Gemma4RMSNorm


class Gemma4MultimodalEmbedder(Module):
    """Projects multimodal soft tokens into language-model hidden space.

    Mirrors the HuggingFace ``Gemma4MultimodalEmbedder``:

    """

    def __init__(
        self,
        multimodal_hidden_size: int,
        text_hidden_size: int,
        dtype: DType,
        device: DeviceRef,
        eps: float = 1e-6,
    ) -> None:
        """Initializes Gemma4MultimodalEmbedder.

        Args:
            multimodal_hidden_size: Hidden size of the multimodal encoder
                (vision or audio). Corresponds to
                ``getattr(multimodal_config, "output_proj_dims",
                multimodal_config.hidden_size)`` in the HF reference.
            text_hidden_size: Hidden size of the language model (i.e.
                ``text_config.hidden_size``).
            dtype: Weight and computation dtype.
            device: Device on which the linear projection runs.
            eps: Epsilon for the RMS normalization.
        """
        super().__init__()
        self.multimodal_hidden_size = multimodal_hidden_size
        self.text_hidden_size = text_hidden_size
        self.dtype = dtype
        self.eps = eps

        self.embedding_projection = Linear(
            in_dim=multimodal_hidden_size,
            out_dim=text_hidden_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            dim=multimodal_hidden_size,
            dtype=dtype,
            eps=eps,
            with_weight=False,
        )

    def __call__(self, inputs_embeds: TensorValue) -> TensorValue:
        """Projects multimodal embeddings into language-model space.

        Args:
            inputs_embeds: Soft token embeddings from the multimodal encoder,
                shape ``[seq_len, multimodal_hidden_size]``.

        Returns:
            Projected embeddings with shape
            ``[seq_len, text_hidden_size]``.
        """
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normed.cast(self.dtype))

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.embedding_projection.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self.embedding_projection.weight.sharding_strategy = strategy
        self.embedding_pre_projection_norm.sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma4MultimodalEmbedder]:
        assert self.sharding_strategy

        embedding_projection_shards = self.embedding_projection.weight.shard(
            devices
        )
        embedding_pre_projection_norm_shards = (
            self.embedding_pre_projection_norm.shard(devices)
        )

        shards = []
        for (
            device,
            emb_proj_weight_shard,
            emb_pre_proj_norm_weight_shard,
        ) in zip(
            devices,
            embedding_projection_shards,
            embedding_pre_projection_norm_shards,
            strict=False,
        ):
            sharded = Gemma4MultimodalEmbedder(
                self.multimodal_hidden_size,
                self.text_hidden_size,
                dtype=self.dtype,
                device=device,
                eps=self.eps,
            )

            sharded.embedding_projection.weight = emb_proj_weight_shard
            sharded.embedding_pre_projection_norm = (
                emb_pre_proj_norm_weight_shard
            )

            shards.append(sharded)

        return shards
