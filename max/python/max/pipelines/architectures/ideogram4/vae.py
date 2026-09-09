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
"""Ideogram 4 VAE component.

Reuses the FLUX.2 ``AutoencoderKLFlux2`` decoder architecture but decodes via
a fused graph that applies Ideogram's per-channel latent de-normalization and
2x2 unpatch (no BatchNorm), then postprocesses to NHWC uint8.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.profiler import traced

from ..autoencoders_modulev3.autoencoder_kl_flux2 import AutoencoderKLFlux2
from ..autoencoders_modulev3.model import BaseAutoencoderModel
from ..autoencoders_modulev3.model_config import AutoencoderKLFlux2Config
from .decode_step import Ideogram4DecodeStep


class Ideogram4VAEModel(BaseAutoencoderModel):
    """ComponentModel wrapping the Ideogram 4 (FLUX.2-arch) VAE decoder."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        **kwargs: Any,
    ) -> None:
        # The repo's VAE encoder is fp8, so the manifest reports an fp8
        # encoding, but only the bf16 decoder is used for image generation.
        # Pin the compute dtype to bf16 regardless of the manifest encoding.
        super().__init__(
            config=config,
            encoding="bfloat16",
            devices=devices,
            weights=weights,
            config_class=AutoencoderKLFlux2Config,
            autoencoder_class=AutoencoderKLFlux2,
            **kwargs,
        )

    @traced(message="Ideogram4VAEModel.load_model")
    def load_model(self) -> None:
        """Compile the fused latent-denorm + unpatch + decode graph."""
        target_dtype = self.config.dtype
        # Only the decoder + post-quant conv are needed for decoding. The VAE
        # encoder weights are fp8 and unused here; skip them before any cast so
        # we never attempt an unsupported fp8 cast on CPU. The decoder weights
        # are stored in bf16, so a plain ``astype`` to the compute dtype works.
        fused_weights: dict[str, Any] = {}
        for key, value in self.weights.items():
            if key.startswith("decoder."):
                module_key = key
            elif key.startswith("post_quant_conv."):
                module_key = f"decoder.{key}"
            else:
                continue
            weight_data = value.data()
            if (
                weight_data.dtype != target_dtype
                and weight_data.dtype.is_float()
                and target_dtype.is_float()
            ):
                weight_data = weight_data.astype(target_dtype)
            fused_weights[module_key] = weight_data

        with F.lazy():
            autoencoder = AutoencoderKLFlux2(self.config)
            decode_step = Ideogram4DecodeStep(decoder=autoencoder.decoder)
            decode_step.to(self.devices[0])
            self.model = decode_step.compile(
                *decode_step.input_types(), weights=fused_weights
            )

    @traced(message="Ideogram4VAEModel.decode_packed")
    def decode_packed(
        self, latents_bsc: Tensor, grid_h: int, grid_w: int
    ) -> Tensor:
        """Decode packed ``(B, S, 128)`` latents to an NHWC uint8 image.

        The grid dimensions are conveyed as 1-D CPU shape-carrier tensors whose
        *lengths* (not contents) encode ``grid_h`` / ``grid_w`` as symbolic
        graph dims, so one compiled graph handles any resolution.
        """
        h_carrier = Tensor(storage=Buffer.from_dlpack(_shape_carrier(grid_h)))
        w_carrier = Tensor(storage=Buffer.from_dlpack(_shape_carrier(grid_w)))
        return self.model(latents_bsc, h_carrier, w_carrier)


def _shape_carrier(n: int) -> np.ndarray:
    return np.ascontiguousarray(np.empty(int(n), dtype=np.float32))
