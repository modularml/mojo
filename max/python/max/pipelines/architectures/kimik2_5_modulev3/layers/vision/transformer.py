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

"""Vision transformer for Kimi K2.5."""

from __future__ import annotations

from typing import Any

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.sharding.action import ActionSet
from max.experimental.sharding.cost import force_replicated_action_set
from max.experimental.sharding.types import TensorLayout
from max.experimental.tensor import Tensor
from max.nn.kernels import tpool_patch_merger as _tpool_patch_merger

from ...model_config import VisionConfig
from .encoder import Encoder
from .patch_embedding import PatchEmbedding
from .patch_merger import PatchMergerMLP


def _tpool_patch_merger_rule(
    input: TensorLayout, grid_thws: TensorLayout, *extras: Any
) -> ActionSet:
    """Replicated on the token stream and the grid table.

    Only the leading tensor operands are declared; the ``kH``/``kW`` merge
    extents and the host ``max_h``/``max_w`` scalars trail them and pass
    through the dispatcher untouched.
    """
    return force_replicated_action_set(input, grid_thws)


tpool_patch_merger = F.functional(
    _tpool_patch_merger, rule=_tpool_patch_merger_rule
)


class Transformer(
    Module[[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, ...]]
):
    """Full vision transformer.

    Composes patch embedding, the encoder stack, spatial-temporal
    patch merging via ``tpool_patch_merger``, and a learned
    :obj:`PatchMergerMLP` projection.

    Args:
        config: :obj:`VisionConfig` instance from which all architecture
            parameters are derived.
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.merge_kernel_size = (
            config.merge_kernel_size[0],
            config.merge_kernel_size[1],
        )

        self.patch_embed = PatchEmbedding(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            hidden_size=config.vt_hidden_size,
            init_pos_emb_height=config.init_pos_emb_height,
            init_pos_emb_width=config.init_pos_emb_width,
            init_pos_emb_time=config.init_pos_emb_time,
            dtype=config.dtype,
            has_bias=config.has_bias,
        )
        self.encoder = Encoder(
            num_heads=config.vt_num_attention_heads,
            hidden_dim=config.vt_hidden_size,
            mlp_dim=config.vt_intermediate_size,
            num_layers=config.vt_num_hidden_layers,
            rope_max_height=config.rope_max_height,
            rope_max_width=config.rope_max_width,
            rope_theta=config.rope_theta,
            has_bias=config.has_bias,
        )
        self.patch_merger = PatchMergerMLP(
            mm_hidden_size=config.mm_hidden_size,
            hidden_size=config.text_hidden_size,
            merge_kernel_size=self.merge_kernel_size,
            eps=config.projector_ln_eps,
        )

    def forward(
        self,
        pixel_values: Tensor,
        grid_thws: Tensor,
        input_row_offsets: Tensor,
        max_seq_len: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, ...]:
        """Full vision transformer forward pass.

        Args:
            pixel_values: (n_patches, in_channels, patch_size, patch_size) NCHW
                pixel patches.
            grid_thws: (n_videos, 3) temporal, height, width per video, int64.
            input_row_offsets: Cumulative sequence lengths of shape
                (batch_size + 1,), dtype uint32.
            max_seq_len: Maximum sequence length, shape (1,), dtype uint32.
            position_ids: 1-D int tensor of flat grid indices for RoPE, int64.

        Returns:
            Merged patch tensor of shape ``(sum_i(H_i * W_i), hidden_dim)``.
        """
        h = self.patch_embed(pixel_values, grid_thws)
        h = self.encoder(h, input_row_offsets, max_seq_len, position_ids)

        kH, kW = self.merge_kernel_size

        # Compute max_h and max_w on a single device, because the value will
        # be moved to the host for the kernel.
        device_0_grid_thw = grid_thws.local_shards[0]
        max_h = F.reshape(
            F.max(device_0_grid_thw[:, 1], axis=0).cast(DType.int32), []
        ).to(CPU())
        max_w = F.reshape(
            F.max(device_0_grid_thw[:, 2], axis=0).cast(DType.int32), []
        ).to(CPU())

        h = tpool_patch_merger(
            h,
            grid_thws,
            kH=kH,
            kW=kW,
            max_h=max_h,
            max_w=max_w,
        )

        # PatchMergerMLP expects (n_spatial, kH*kW, D); kernel returns (n_merged, D).
        # n_merged is a dynamic dim, so rebind first to assert it is divisible by
        # kH*kW before reshaping (as suggested by the MAX error message).
        merge_k = kH * kW
        h = F.rebind(h, [(h.shape[0] // merge_k) * merge_k, h.shape[1]])
        h = h.reshape([h.shape[0] // merge_k, merge_k, h.shape[1]])
        h = self.patch_merger(h)
        return (h,)
