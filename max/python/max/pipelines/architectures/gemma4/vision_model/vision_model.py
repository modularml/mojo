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

from collections.abc import Sequence

from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Weight,
)
from max.nn.layer import Module
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    compute_vision_freqs_cis,
)

from ..layers.multimodal_embedder import Gemma4MultimodalEmbedder
from ..model_config import Gemma4ForConditionalGenerationConfig
from .embedding import Gemma4VisionPatchEmbedder
from .encoding import Gemma4VisionEncoder
from .pooling import Gemma4VisionPooler


class Gemma4VisionModel(Module):
    """Vision tower for the Gemma4 multimodal model.

    Processes a ragged batch of flat image patches through:

    1. ``patch_embedder`` — pixel normalisation, linear projection, 2-D
       position embeddings.
    2. ``encoder`` — stack of ``Gemma4VisionEncoderLayer`` blocks with 2-D
       multidimensional RoPE and ragged flash attention.
    3. ``pooler`` — sparse average pooling to ``image_seq_length`` tokens
       per image using a pre-computed weight matrix passed as a graph input.
    """

    def __init__(
        self, config: Gemma4ForConditionalGenerationConfig, device: DeviceRef
    ) -> None:
        super().__init__()
        self.config = config
        self.device = config.devices[0]
        vision_config = config.vision_config
        assert vision_config is not None
        self.patch_embedder = Gemma4VisionPatchEmbedder(
            config, device=self.device
        )
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(
            vision_config.hidden_size, vision_config.pooling_kernel_size
        )
        self.rope_theta = vision_config.rope_theta
        self.head_dim = vision_config.head_dim

        # In the reference, the multimodal embedder is in the layer above
        # (Gemma4Model), but the image features includes this projection,
        # so it is included here.
        vision_dtype = config.unquantized_dtype
        self.embed_vision = Gemma4MultimodalEmbedder(
            vision_config.hidden_size,
            config.text_config.hidden_size,
            vision_dtype,
            self.device,
            vision_config.rms_norm_eps,
        )
        self.standardize = vision_config.standardize

        if self.standardize:
            self.std_bias = Weight(
                "std_bias",
                vision_dtype,
                shape=[vision_config.hidden_size],
                device=self.device,
            )
            self.std_scale = Weight(
                "std_scale",
                vision_dtype,
                shape=[vision_config.hidden_size],
                device=self.device,
            )
            self.std_bias.sharding_strategy = ShardingStrategy.replicate(
                len(config.devices)
            )
            self.std_scale.sharding_strategy = ShardingStrategy.replicate(
                len(config.devices)
            )
            self.std_bias_shards = self.std_bias.shard(config.devices)
            self.std_scale_shards = self.std_scale.shard(config.devices)

        if len(config.devices) > 1:
            self.patch_embedder.sharding_strategy = ShardingStrategy.replicate(
                len(config.devices)
            )
            self.encoder.sharding_strategy = ShardingStrategy.replicate(
                len(config.devices)
            )
            self.embed_vision.sharding_strategy = ShardingStrategy.replicate(
                len(config.devices)
            )
            self.patch_embedder_shards = self.patch_embedder.shard(
                config.devices
            )
            self.encoder_shards = self.encoder.shard(config.devices)
            # pooler is stateless so sharding is a no_op
            self.pooler_shards = [self.pooler] * len(config.devices)
            self.embed_vision_shards = self.embed_vision.shard(config.devices)
        else:
            self.patch_embedder_shards = [self.patch_embedder]
            self.encoder_shards = [self.encoder]
            self.pooler_shards = [self.pooler]
            self.embed_vision_shards = [self.embed_vision]

    def __call__(
        self,
        patches_flat: Sequence[TensorValue],
        pixel_position_ids: Sequence[TensorValue],
        cu_seqlens: Sequence[TensorValue],
        pool_gather_index: Sequence[TensorValue],
        max_seq_len: TensorValue,
    ) -> list[TensorValue]:
        """Process packed image patches through the full vision tower.

        Args:
            patches_flat: Packed flat patch pixels,
                shape ``[total_patches, 3 * patch_size²]``.
            pixel_position_ids: Integer (x, y) coordinates,
                shape ``[total_patches, 2]``, dtype int32.
            cu_seqlens: Cumulative sequence lengths
                (image boundaries), shape ``[num_images + 1]``, dtype uint32.
            pool_gather_index: Per-output-bin patch indices for sparse average
                pooling, shape ``[num_pooled_tokens, k**2]``, dtype int32.
            max_seq_len: Maximum patches per image (scalar uint32, CPU).

        Returns:
            Projected image embeddings, shape
            ``[num_images * image_seq_length, text_hidden_size]``.
        """
        hidden_states_list = [
            patch_embedder(patches, pixel_positions)
            for patch_embedder, patches, pixel_positions in zip(
                self.patch_embedder_shards,
                patches_flat,
                pixel_position_ids,
                strict=True,
            )
        ]

        freqs_cis_list = [
            compute_vision_freqs_cis(
                pixel_position_ids=pos_ids,
                head_dim=self.head_dim,
                ndim=2,
                theta=self.rope_theta,
                dtype=self.config.unquantized_dtype,
                device=pos_ids.device,  # or hidden_states.device
            )
            for pos_ids in pixel_position_ids
        ]
        encoded_list = [
            encoder(
                hidden_states,
                freqs_cis,
                cusl,
                max_seq_len,
            )
            for encoder, hidden_states, freqs_cis, cusl in zip(
                self.encoder_shards,
                hidden_states_list,
                freqs_cis_list,
                cu_seqlens,
                strict=True,
            )
        ]

        pooled_list = [
            pooler(encoded, pgi)
            for pooler, encoded, pgi in zip(
                self.pooler_shards,
                encoded_list,
                pool_gather_index,
                strict=True,
            )
        ]

        if self.standardize:
            # `pooled` may be float32 (the pooler defers its float16 downcast);
            # match the params to its dtype (a no-op otherwise).
            pooled_list = [
                (pooled - std_bias.cast(pooled.dtype))
                * std_scale.cast(pooled.dtype)
                for std_bias, std_scale, pooled in zip(
                    self.std_bias_shards,
                    self.std_scale_shards,
                    pooled_list,
                    strict=False,
                )
            ]

        return [
            embed_vision(pooled)
            for embed_vision, pooled in zip(
                self.embed_vision_shards, pooled_list, strict=False
            )
        ]

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """Build the input type list for the vision model graph.

        The vision model receives four device tensors plus one CPU scalar.

        * ``patches_flat``        — ``[total_patches, 3 * patch_size²]``, bf16
        * ``pixel_position_ids``  — ``[total_patches, 2]``, int32
        * ``cu_seqlens``          — ``[num_images + 1]``, uint32
        * ``pool_gather_index``   — ``[num_pooled_tokens, max_pool_patches]``, int32
        * ``max_seq_len``         — scalar uint32, on CPU (one shared tensor)
        """
        vision_config = self.config.vision_config
        assert vision_config is not None
        patch_dim = 3 * vision_config.patch_size**2
        devices = self.config.devices

        patches_flat_types = [
            TensorType(
                self.config.unquantized_dtype,
                shape=["total_patches", patch_dim],
                device=device,
            )
            for device in devices
        ]
        pixel_position_ids_types = [
            TensorType(
                DType.int32,
                shape=["total_patches", 2],
                device=device,
            )
            for device in devices
        ]
        cu_seqlens_types = [
            TensorType(
                DType.uint32,
                shape=["num_images_plus_1"],
                device=device,
            )
            for device in devices
        ]
        pool_gather_index_types = [
            TensorType(
                DType.int32,
                shape=["num_pooled_tokens", "max_pool_patches"],
                device=device,
            )
            for device in devices
        ]
        max_seq_len_type = TensorType(
            DType.uint32,
            shape=[],
            device=DeviceRef.CPU(),
        )

        return (
            *patches_flat_types,
            *pixel_position_ids_types,
            *cu_seqlens_types,
            *pool_gather_index_types,
            max_seq_len_type,
        )
