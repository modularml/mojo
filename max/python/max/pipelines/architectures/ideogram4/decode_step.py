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
"""Fused decode-step module for the Ideogram 4 pipeline.

Combines Ideogram-specific per-channel latent de-normalization + 2x2 unpatch
with the FLUX.2 VAE decoder forward pass into one compiled graph. Accepts
packed latents ``(B, S, 128)`` where ``S = grid_h * grid_w`` and produces an
NHWC uint8 image in ``[0, 255]``.
"""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType

from ..autoencoders_modulev3.vae import Decoder
from .latent_norm import latent_scale_array, latent_shift_array


class Ideogram4DecodeStep(Module[..., Tensor]):
    """Fused latent-denorm + unpatch + VAE decode: packed latents -> image.

    The 128 packed channels split as ``(patch_h=2, patch_w=2, ae_channels)``
    with ``ae_channels`` fastest, matching the torch reference unpatch
    ``z.view(B, gh, gw, 2, 2, C).permute(0, 5, 1, 3, 2, 4)``.
    """

    def __init__(self, decoder: Decoder) -> None:
        super().__init__()
        self.decoder = decoder

    def input_types(self) -> tuple[TensorType, ...]:
        num_channels = self.decoder.in_channels * 4  # 32 * 4 = 128
        dtype = self.decoder.dtype
        device = self.decoder.device
        assert dtype is not None, "Decoder dtype must be set before compilation"
        assert device is not None, (
            "Decoder device must be set before compilation"
        )
        return (
            TensorType(
                dtype, shape=["batch", "seq", num_channels], device=device
            ),
            # Shape carriers: lengths encode grid_h / grid_w as symbolic dims.
            TensorType(DType.float32, shape=["grid_h"], device=DeviceRef.CPU()),
            TensorType(DType.float32, shape=["grid_w"], device=DeviceRef.CPU()),
        )

    def forward(
        self,
        latents_bsc: Tensor,
        h_carrier: Tensor,
        w_carrier: Tensor,
    ) -> Tensor:
        batch = latents_bsc.shape[0]
        c = latents_bsc.shape[2]
        gh = h_carrier.shape[0]
        gw = w_carrier.shape[0]
        ae_c = self.decoder.in_channels

        device = self.decoder.device
        dtype = self.decoder.dtype
        assert device is not None and dtype is not None

        # Per-channel de-normalization: z * scale + shift, on packed (B, S, 128).
        scale = Tensor(latent_scale_array(), device=device).cast(dtype)
        shift = Tensor(latent_shift_array(), device=device).cast(dtype)
        latents_bsc = F.rebind(latents_bsc, [batch, gh * gw, c])
        latents_bsc = latents_bsc * scale + shift

        # Unpatch: (B, gh*gw, 128) -> (B, ae_c, gh*2, gw*2).
        latents = F.reshape(latents_bsc, (batch, gh, gw, 2, 2, ae_c))
        latents = F.permute(latents, (0, 5, 1, 3, 2, 4))
        latents = F.reshape(latents, (batch, ae_c, gh * 2, gw * 2))

        decoded = self.decoder(latents, None)  # (B, 3, H, W)

        # Postprocess: [-1, 1] -> [0, 255] uint8, NCHW -> NHWC, to CPU.
        decoded = F.cast(decoded, DType.float32)
        decoded = (decoded + 1.0) * 127.5
        decoded = F.max(decoded, 0.0)
        decoded = F.min(decoded, 255.0)
        decoded = F.round(decoded)
        decoded = F.permute(decoded, (0, 2, 3, 1))
        return F.cast(decoded, DType.uint8).to(CPU())
