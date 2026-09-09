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

from collections.abc import Iterable

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.graph.weight import ShardingStrategy
from max.nn.layer import Module
from max.nn.linear import Linear

from ..model_config import Gemma4ForConditionalGenerationConfig


class Gemma4VisionPatchEmbedder(Module):
    """Patchify + positional embedding for the Gemma4 SigLIP vision encoder.

    Takes pre-extracted flat patches and (x, y) position IDs, normalises pixel
    values from [0, 1] to [-1, 1], projects them via a linear layer, then adds
    2-D position embeddings computed by an embedding-table lookup.
    """

    def __init__(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        # Use unquantized_dtype (not config.dtype) so quantized models still
        # embed patches in a real float dtype.
        self.dtype = config.unquantized_dtype

        vision_cfg = config.vision_config
        assert vision_cfg is not None
        hidden_size = vision_cfg.hidden_size
        patch_size = vision_cfg.patch_size
        self.hidden_size = hidden_size
        self.position_embedding_size = vision_cfg.position_embedding_size

        self.input_proj = Linear(
            3 * patch_size**2,
            hidden_size,
            dtype=self.dtype,
            device=self.device,
            has_bias=False,
        )

        self.position_embedding_table = Weight(
            "position_embedding_table",
            dtype=self.dtype,
            shape=(2, self.position_embedding_size, hidden_size),
            device=self.device,
        )

    def __call__(
        self,
        patches_flat: TensorValue,
        pixel_position_ids: TensorValue,
    ) -> TensorValue:
        """Embed flat patches into hidden space with 2-D position information.

        Args:
            patches_flat: Raw pixel patches, shape
                ``[total_patches, 3 * patch_size²]``, values in ``[0, 1]``.
            pixel_position_ids: Integer (x, y) coordinates for each patch,
                shape ``[total_patches, 2]``, dtype int32.

        Returns:
            Patch embeddings of shape ``[total_patches, hidden_size]``.
        """
        # 1. Normalise pixel values from [0, 1] to [-1, 1].
        patches = ops.cast(patches_flat, self.dtype)
        patches = patches * ops.constant(
            2.0, self.dtype, device=self.device
        ) - ops.constant(1.0, self.dtype, device=self.device)

        # 2. Linear projection: [total_patches, 3*ps²] → [total_patches, hidden]
        hidden = self.input_proj(patches)

        # 3. Position embeddings via embedding-table lookup.
        #    position_embedding_table: [2, pos_emb_size, hidden_size]
        #    Reshape to [2 * pos_emb_size, hidden_size] so we can use
        #    ops.gather with offset indices for the y dimension.
        table_flat = ops.reshape(
            self.position_embedding_table,
            [2 * self.position_embedding_size, self.hidden_size],
        )

        x_ids = ops.cast(pixel_position_ids[:, 0], DType.int64)
        y_ids = ops.cast(pixel_position_ids[:, 1], DType.int64) + ops.constant(
            self.position_embedding_size, DType.int64, device=self.device
        )

        emb_x = ops.gather(table_flat, x_ids, axis=0)
        emb_y = ops.gather(table_flat, y_ids, axis=0)
        position_emb = emb_x + emb_y

        return hidden + position_emb

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.input_proj.weight.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self.input_proj.weight.sharding_strategy = strategy
        self.position_embedding_table.sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma4VisionPatchEmbedder]:
        assert self.sharding_strategy

        input_proj_shards = self.input_proj.weight.shard(devices)
        position_embedding_table_shards = self.position_embedding_table.shard(
            devices
        )

        shards = []
        for device, input_proj_shard, pos_emb_shard in zip(
            devices,
            input_proj_shards,
            position_embedding_table_shards,
            strict=True,
        ):
            sharded = Gemma4VisionPatchEmbedder(self.config, device)
            sharded.input_proj.weight = input_proj_shard
            sharded.position_embedding_table = pos_emb_shard
            shards.append(sharded)

        return shards
