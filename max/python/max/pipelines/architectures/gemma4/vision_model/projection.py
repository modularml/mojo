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

from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.graph.ops import avg_pool2d
from max.nn.layer import Module, Shardable

from ..layers.rms_norm import Gemma4RMSNorm
from ..model_config import Gemma4ForConditionalGenerationConfig


class Gemma4MultiModalProjector(Module, Shardable):
    """Projects vision encoder outputs to text embedding space."""

    def __init__(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        device: DeviceRef | None = None,
    ):
        """Prepare the normalisation and projection weights based on config"""
        super().__init__()

        self.config = config
        assert config.vision_config is not None
        self.device = device if device is not None else config.devices[0]

        vision_dtype = config.unquantized_dtype

        self.mm_input_projection_weight = Weight(
            "mm_input_projection_weight",
            dtype=vision_dtype,
            shape=(
                config.vision_config.hidden_size,
                config.text_config.hidden_size,
            ),
            device=self.device,
        )

        self.mm_soft_emb_norm = Gemma4RMSNorm(
            config.vision_config.hidden_size,
            eps=config.vision_config.rms_norm_eps,
            dtype=vision_dtype,
        )

        self.patches_per_image = 280  # TODO: Hardcoded

        self.kernel_size = config.vision_config.pooling_kernel_size

    def __call__(self, vision_outputs: TensorValue) -> TensorValue:
        """Process vision outputs through pooling, normalisation, and a
        projection weight"""
        batch_size, _, seq_length = vision_outputs.shape

        transposed_vision_outputs = vision_outputs.transpose(1, 2)

        reshaped_vision_outputs = ops.reshape(
            transposed_vision_outputs,
            [
                batch_size,
                seq_length,
                self.patches_per_image,
                self.patches_per_image,
            ],
        )

        # reshape to 0 2 3 1 NHWL (or NHWC) for avg pool
        reshaped_vision_outputs = ops.permute(
            reshaped_vision_outputs, [0, 2, 3, 1]
        )
        pooled_vision_outputs = avg_pool2d(
            input=reshaped_vision_outputs,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=self.kernel_size,
        )

        pooled_vision_outputs = ops.permute(pooled_vision_outputs, [0, 3, 1, 2])
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = (
            normed_vision_outputs @ self.mm_input_projection_weight
        )

        image_hidden_states = ops.flatten(
            projected_vision_outputs, start_dim=0, end_dim=1
        )

        return image_hidden_states

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.mm_input_projection_weight.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self.mm_input_projection_weight.sharding_strategy = strategy
        self.mm_soft_emb_norm.weight.sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma4MultiModalProjector]:
        assert self.sharding_strategy

        projection_weight_shards = self.mm_input_projection_weight.shard(
            devices
        )
        norm_weight_shards = self.mm_soft_emb_norm.weight.shard(devices)

        shards = []
        for device, proj_weight_shard, norm_weight_shard in zip(
            devices,
            projection_weight_shards,
            norm_weight_shards,
            strict=True,
        ):
            sharded = Gemma4MultiModalProjector(self.config, device)

            sharded.mm_input_projection_weight = proj_weight_shard
            sharded.mm_soft_emb_norm.weight = norm_weight_shard

            shards.append(sharded)

        return shards
