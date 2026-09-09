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
"""Fused FLUX.2 Denoiser sub-``Module`` for the ModuleV3 executor."""

from __future__ import annotations

from typing import Any

from max.driver import DeviceSpec, load_devices
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph.weights import WeightData
from max.pipelines.architectures.flux2.model_config import Flux2Config
from max.pipelines.architectures.flux2.nvfp4_weight_adapter import (
    convert_nvfp4_state_dict,
)
from max.pipelines.architectures.flux2.weight_adapters import (
    _split_stacked_qkv,
)
from max.pipelines.lib.weight_loader import WeightLoader, precompute
from max.pipelines.modeling.config_enums import SupportedEncoding

from ..transformer import Flux2Transformer2DModel


class Denoiser(Module[..., Tensor]):
    """Fused FLUX.2 denoiser: concat + transformer + Euler step.

    Mirrors the legacy
    :class:`max.pipelines.architectures.flux2.components.denoiser.DenoiseStep`
    body inside a single :class:`~max.experimental.nn.Module`: concatenates
    optional image latents onto noise latents along the sequence axis,
    runs the FLUX.2 transformer, slices the predicted noise to the noise
    sequence length, and applies a single float32 Euler update.  The
    enclosing :class:`FLUXModule` drives the per-step ``timestep`` / ``dt``
    gather and ``ops.while_loop`` carry.

    Single-device + BF16 only in this commit; multi-device (``Allreduce``
    / ``Signals``) and the NVFP4 quantization-aware Linear path are
    follow-up commits.
    """

    def __init__(
        self,
        huggingface_config: dict[str, Any],
        quantization_encoding: SupportedEncoding | None,
        device_specs: list[DeviceSpec],
    ) -> None:
        encoding: SupportedEncoding = quantization_encoding or "bfloat16"
        devices = load_devices(device_specs)

        flux2_config = Flux2Config.initialize_from_config(
            huggingface_config, encoding, devices
        )

        self._dtype: DType = flux2_config.dtype
        self.transformer = Flux2Transformer2DModel(flux2_config)
        self.to(devices[0])

    def forward(
        self,
        latents: Tensor,
        image_latents: Tensor,
        text_embeddings: Tensor,
        timestep: Tensor,
        dt: Tensor,
        guidance: Tensor,
        latent_image_ids: Tensor,
        text_ids: Tensor,
    ) -> Tensor:
        """One fused FLUX.2 denoise step.

        Args:
            latents: Current noise latents ``[B, S, C]``.
            image_latents: Optional packed image latents ``[B, S_img, C]``
                (zero-seq-length for text-to-image; the concat below is a
                no-op in that case).
            text_embeddings: Encoder embeddings
                ``[B, S_txt, joint_attention_dim]``.
            timestep: Per-batch sigma ``[B]`` (float32).
            dt: Step delta ``[1]`` (float32).
            guidance: Per-batch guidance scale ``[B]`` (float32).
            latent_image_ids: Latent position IDs ``[B, S, 4]`` int64.
            text_ids: Text position IDs ``[B, S_txt, 4]`` int64.

        Returns:
            Updated noise latents ``[B, S, C]`` in the transformer dtype.
        """
        # Concat image latents onto noise latents along the sequence
        # axis.  For text-to-image, ``image_latents`` has shape
        # ``(B, 0, C)`` so the concat is a no-op and the produced
        # tensor is shape-equivalent to ``latents``.  The matching
        # ``image_latent_ids`` concat is omitted here: the ModuleV3
        # pipeline does not yet plumb img2img reference IDs, so
        # ``latent_image_ids`` already carries every position ID the
        # transformer needs.  Img2img support is a follow-up commit.
        latents_concat = F.concat([latents, image_latents], axis=1)

        noise_pred = self.transformer(
            latents_concat,
            text_embeddings,
            timestep,
            latent_image_ids,
            text_ids,
            guidance,
        )

        # Slice off any extra image tokens; in single-image-input mode
        # (no img2img) ``latents_concat`` is shape-equivalent to
        # ``latents``, so the slice is identity.
        num_tokens = latents.shape[1]
        noise_pred_sliced = noise_pred[:, :num_tokens, :]

        # Euler update in float32 for numerical stability, matching the
        # legacy ``DenoiseStep`` Euler path.
        latents_dtype = latents.dtype
        latents_f32 = latents.cast(DType.float32)
        noise_pred_f32 = noise_pred_sliced.cast(DType.float32)
        return (latents_f32 + dt * noise_pred_f32).cast(latents_dtype)

    @staticmethod
    def adapt_loader(loader: WeightLoader) -> WeightLoader:
        """Resolve this Module's queries to HF FLUX.2 transformer keys.

        Reuses the legacy :func:`adapt_weights` logic (NVFP4 layout
        conversion + stacked-QKV split), expressed as an eager loader
        transform: the source transformer checkpoint is materialised as
        a group and translated up front, because both transforms reshape
        the namespace (weight + ``weight_scale`` pairing, fused-QKV
        split) and so don't line up key-for-key.  The NVFP4 branch is
        gated on the presence of ``*.weight_scale`` keys in the source,
        and the stacked-QKV branch on ``.attn.qkv_proj.`` /
        ``.attn.add_qkv_proj.`` infixes.  Both detections are pure key
        inspection -- no ``quant_config`` argument is needed at this
        layer.

        Output keys are re-prefixed under ``transformer.`` to match the
        Module hierarchy (``Denoiser.transformer`` ->
        :class:`Flux2Transformer2DModel`).
        """

        def transform(
            state_dict: dict[str, WeightData],
        ) -> dict[str, WeightData]:
            if any(k.endswith(".weight_scale") for k in state_dict):
                state_dict = convert_nvfp4_state_dict(state_dict)
            if any(
                ".attn.qkv_proj." in k or ".attn.add_qkv_proj." in k
                for k in state_dict
            ):
                state_dict = _split_stacked_qkv(state_dict)
            return {f"transformer.{k}": v for k, v in state_dict.items()}

        return precompute(loader, list(loader.keys()), transform)
