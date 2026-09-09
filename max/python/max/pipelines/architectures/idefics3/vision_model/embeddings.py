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

"""Vision embeddings for IDEFICS3 model."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.conv import Conv2d
from max.nn.embedding import Embedding
from max.nn.layer import Module

from ..model_config import Idefics3VisionConfig


class Idefics3VisionEmbeddings(Module):
    """Vision embeddings for Idefics3 with patch embeddings and simplified position encoding.

    This converts images into patch embeddings and adds position embeddings.
    Unlike the PyTorch version, we simplify position ID generation since it's
    always just a sequential range from 0 to num_patches-1.

    The modifications are adapted from [Patch n' Pack: NaViT, a Vision Transformer
    for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304) which
    allows treating images in their native aspect ratio and without the need to
    resize them to the same fixed size.
    """

    def __init__(
        self,
        config: Idefics3VisionConfig,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.GPU(),  # noqa: B008
    ) -> None:
        """Initialize the Idefics3 vision embeddings.

        Args:
            config: Vision configuration object containing hidden_size, image_size,
                patch_size, num_channels, and other vision parameters.
            dtype: Data type for the weights.
            device: Device to place the weights on.
        """
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.dtype = dtype

        # Patch embedding using Conv2d with stride=patch_size (equivalent to patch extraction)
        # TODO: use a more optimized implementation of conv2d
        self.patch_embedding = Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,  # "valid" padding
            has_bias=True,
            dtype=dtype,
            device=device,
        )

        # Position embedding table
        self.position_embedding = Embedding(
            vocab_size=self.num_positions,
            hidden_dim=self.embed_dim,
            dtype=dtype,
            device=device,
        )

    def __call__(
        self,
        pixel_values: TensorValue,
        patch_attention_mask: TensorValue | None,
    ):
        """Forward pass of vision embeddings.

        Args:
            pixel_values: Input images of shape [batch_size, channels, height, width].
            patch_attention_mask: Attention mask of shape [batch_size, num_patches_h, num_patches_w].

        Returns:
            Embeddings of shape [batch_size, num_patches, hidden_size].
        """

        batch_size = pixel_values.shape[0]
        max_im_h = pixel_values.shape[2]
        max_im_w = pixel_values.shape[3]

        # Convert input from NCHW to NHWC format for MAX Conv2d
        # pixel_values: [batch_size, channels, height, width] -> [batch_size, height, width, channels]
        pixel_values_nhwc = ops.permute(pixel_values, [0, 2, 3, 1])

        # Apply patch embedding (Conv2d with stride=patch_size extracts patches)
        # Output will be in NHWC format: [batch_size, out_height, out_width, out_channels]
        patch_embeds_nhwc = self.patch_embedding(pixel_values_nhwc)

        # Convert output back to NCHW format: [batch_size, out_channels, out_height, out_width]
        patch_embeds = ops.permute(patch_embeds_nhwc, [0, 3, 1, 2])

        # Flatten spatial dimensions and transpose to [batch_size, num_patches, embed_dim]
        # patch_embeds shape: [batch_size, embed_dim, num_patches_h, num_patches_w]
        embeddings = ops.flatten(
            patch_embeds, start_dim=2
        )  # [batch_size, embed_dim, num_patches]
        embeddings = ops.transpose(
            embeddings, 1, 2
        )  # [batch_size, num_patches, embed_dim]

        max_nb_patches_h = max_im_h // self.patch_size
        max_nb_patches_w = max_im_w // self.patch_size
        total_patches = max_nb_patches_h * max_nb_patches_w

        # Create position IDs: [0, 1, 2, ..., total_patches-1] for each batch
        # Generate 2D tensor with shape [batch_size, total_patches]
        position_ids = ops.range(
            start=0,
            stop=self.num_patches,
            step=1,
            out_dim=total_patches,
            device=DeviceRef.GPU(),
            dtype=DType.int32,
        )  # [total_patches]
        position_ids = ops.unsqueeze(position_ids, 0)  # [1, total_patches]
        position_ids = ops.tile(
            position_ids, [batch_size, 1]
        )  # [batch_size, total_patches]

        # Get position embeddings for the position IDs
        position_embeds = self.position_embedding(
            position_ids
        )  # [batch_size, total_patches, embed_dim]

        # Add position embeddings to patch embeddings
        embeddings = embeddings + position_embeds

        return embeddings
