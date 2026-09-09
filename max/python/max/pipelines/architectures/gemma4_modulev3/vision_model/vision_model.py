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

"""Gemma4 vision tower for the ModuleV3 API."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    compute_vision_freqs_cis as _compute_vision_freqs_cis,
)
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)

from .embedding import Gemma4VisionPatchEmbedder
from .encoding import Gemma4VisionEncoder
from .multimodal_embedder import Gemma4MultimodalEmbedder
from .pooling import Gemma4VisionPooler

# The 2-D RoPE frequency helper is gemma4-specific, so it is wrapped here
# instead of being registered in common_layers/functional_kernels.py.
compute_vision_freqs_cis = F.functional(_compute_vision_freqs_cis)


class Gemma4VisionModel(Module[..., Tensor]):
    """patch_embedder -> 2-D-RoPE encoder -> pooler -> (standardize) -> embed_vision.

    Port of the graph arch's Gemma4VisionModel, single-device.
    Attribute names match the graph tree so the vision weight adapter's keys
    load unchanged.
    """

    def __init__(self, config: Gemma4ForConditionalGenerationConfig) -> None:
        super().__init__()
        vision_config = config.vision_config
        assert vision_config is not None
        self.rope_theta = vision_config.rope_theta
        self.head_dim = vision_config.head_dim
        self.dtype = config.unquantized_dtype

        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(
            vision_config.hidden_size, vision_config.pooling_kernel_size
        )
        self.embed_vision = Gemma4MultimodalEmbedder(
            vision_config.hidden_size,
            config.text_config.hidden_size,
            self.dtype,
            eps=vision_config.rms_norm_eps,
        )
        self.standardize = vision_config.standardize
        if self.standardize:
            self.std_bias = Tensor.zeros([vision_config.hidden_size])
            self.std_scale = Tensor.ones([vision_config.hidden_size])

    def forward(
        self,
        patches_flat: Tensor,
        pixel_position_ids: Tensor,
        cu_seqlens: Tensor,
        pool_gather_index: Tensor,
        max_seq_len: Tensor,
    ) -> Tensor:
        hidden = self.patch_embedder(patches_flat, pixel_position_ids)

        freqs_cis = compute_vision_freqs_cis(
            pixel_position_ids,
            head_dim=self.head_dim,
            ndim=2,
            theta=self.rope_theta,
            dtype=self.dtype,
            device=DeviceRef.from_device(pixel_position_ids.device),
        )

        encoded = self.encoder(hidden, freqs_cis, cu_seqlens, max_seq_len)
        pooled = self.pooler(encoded, pool_gather_index)
        if self.standardize:
            pooled = (
                pooled - self.std_bias.cast(pooled.dtype)
            ) * self.std_scale.cast(pooled.dtype)
        return self.embed_vision(pooled)
