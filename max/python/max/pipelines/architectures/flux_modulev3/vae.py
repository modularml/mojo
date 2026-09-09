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
"""Standalone unified FLUX VAE ``Module``.

Reference implementation of the
*single-Module-with-two-forward-methods-and-shared-weights* pattern
called out in MXF-460.  The same :class:`Vae` instance owns both the
encoder and decoder *plus* a single pair of per-channel batch-norm
statistics, and exposes the two halves as :meth:`Vae.encode` and
:meth:`Vae.decode` rather than a single ``forward``.  Each method is
invoked at a different point in a containing pipeline's ``forward``
(encoder up front on the input image, decoder at the end on the
denoised latents); both halves land in the same traced graph, so the
shared BN parameters live on one Module by construction rather than
being duplicated across two siblings.

This package is self-contained: every supporting block, layer, and
config class is local to ``flux_modulev3``.
"""

from __future__ import annotations

from typing import Any

from max.driver import CPU, DeviceSpec, load_devices
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.weight_loader import WeightLoader, rename

from .components import Decoder, Encoder
from .model_config import AutoencoderKLFlux2Config


class Vae(Module[..., Tensor]):
    """FLUX VAE owning encoder + decoder + shared BN statistics.

    Wraps the FLUX VAE :class:`Encoder` and :class:`Decoder` plus the
    surrounding pre/post-processing.  The two halves share one
    ``bn_mean`` / ``bn_var`` pair -- the same checkpoint values
    normalize the encoder output and denormalize the decoder input --
    so they live on this Module directly instead of being duplicated
    across two sibling Modules.

    The Module is *not* compiled standalone.  Both :meth:`encode` and
    :meth:`decode` are designed to be invoked from a containing
    Module's ``forward`` so they land in a single traced graph along
    with the shared BN parameters; calling :meth:`forward` directly
    raises ``NotImplementedError`` to surface that contract.

    Built directly from a HuggingFace VAE config dict, quantization
    encoding, and device specs.  Places itself on ``device_specs[0]``
    so the caller only needs to wrap construction in
    :func:`max.experimental.functional.lazy` and drive compilation
    on the parent Module.

    Weight loading is decoupled from construction.  The static
    :meth:`adapt_loader` wraps a source
    :class:`~max.pipelines.lib.weight_loader.WeightLoader` so the
    Module's parameter queries resolve against the HuggingFace VAE
    checkpoint namespace, unioning the legacy encoder + decoder
    adapters; the shared BN buffers absorb the ``bn.running_mean`` /
    ``bn.running_var`` keys.
    """

    def __init__(
        self,
        huggingface_config: dict[str, Any],
        quantization_encoding: SupportedEncoding | None,
        device_specs: list[DeviceSpec],
    ) -> None:
        encoding: SupportedEncoding = quantization_encoding or "bfloat16"
        devices = load_devices(device_specs)
        device_ref = DeviceRef.from_device(devices[0])

        vae_config = AutoencoderKLFlux2Config.generate(
            huggingface_config, encoding, devices
        )

        patch_h, patch_w = vae_config.patch_size

        self._dtype: DType = vae_config.dtype
        self._device_ref: DeviceRef = device_ref
        self._batch_norm_eps: float = vae_config.batch_norm_eps
        self._latent_channels: int = vae_config.latent_channels
        self._patch_h: int = patch_h
        self._patch_w: int = patch_w
        self._num_channels: int = vae_config.latent_channels * patch_h * patch_w

        self.encoder = Encoder(
            in_channels=vae_config.in_channels,
            out_channels=vae_config.latent_channels,
            down_block_types=tuple(vae_config.down_block_types),
            block_out_channels=tuple(vae_config.block_out_channels),
            layers_per_block=vae_config.layers_per_block,
            norm_num_groups=vae_config.norm_num_groups,
            act_fn=vae_config.act_fn,
            double_z=True,
            mid_block_add_attention=vae_config.mid_block_add_attention,
            use_quant_conv=vae_config.use_quant_conv,
            device=device_ref,
            dtype=vae_config.dtype,
        )

        self.decoder = Decoder(
            in_channels=vae_config.latent_channels,
            out_channels=vae_config.out_channels,
            up_block_types=tuple(vae_config.up_block_types),
            block_out_channels=tuple(vae_config.block_out_channels),
            layers_per_block=vae_config.layers_per_block,
            norm_num_groups=vae_config.norm_num_groups,
            act_fn=vae_config.act_fn,
            mid_block_add_attention=vae_config.mid_block_add_attention,
            use_post_quant_conv=vae_config.use_post_quant_conv,
            device=device_ref,
            dtype=vae_config.dtype,
        )

        # Shared per-channel batch-norm statistics used by both
        # :meth:`encode` and :meth:`decode`.  Loaded from
        # ``bn.running_mean`` / ``bn.running_var`` in the HF VAE
        # checkpoint (see :meth:`adapt_loader`).
        self.bn_mean = Tensor.zeros(
            [self._num_channels], dtype=vae_config.dtype
        )
        self.bn_var = Tensor.zeros([self._num_channels], dtype=vae_config.dtype)

        self.to(devices[0])

    @property
    def num_channels(self) -> int:
        """Latent channel count exposed to the parent after patchify.

        The encoder emits ``latent_channels`` mean channels which are
        then patchified by ``patch_size = (patch_h, patch_w)`` into
        ``latent_channels * patch_h * patch_w`` packed channels.  This
        is the value the parent's denoiser expects to see on its input
        latents.
        """
        return self._num_channels

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        # This Module is invoked through :meth:`encode` and
        # :meth:`decode` from a parent Module's ``forward``; both
        # methods land in the parent's traced graph.  Compiling this
        # Module standalone is unsupported -- pick one of the explicit
        # method entry points instead.
        raise NotImplementedError(
            "Vae has two entry points (encode/decode) and is not "
            "compiled standalone; call vae.encode(...) or "
            "vae.decode(...) from a parent Module's forward."
        )

    def encode(self, input_image: Tensor) -> Tensor:
        """Encode a raw uint8 image into packed latent tokens.

        Args:
            input_image: ``(H, W, 3)`` uint8.  ``H`` and ``W`` may be
                zero, in which case every downstream tensor carries
                zero spatial dims through to the
                ``(1, 0, num_channels)`` output.

        Returns:
            ``(1, packed_h * packed_w, num_channels)`` in the model
            dtype, where ``packed_h = H / (vae_scale * patch_h)`` and
            ``packed_w = W / (vae_scale * patch_w)``.
        """
        # Preprocess: (H, W, 3) uint8 -> (1, 3, H, W) model dtype in [-1, 1].
        image = input_image.cast(DType.float32)
        image = image / 127.5 - 1.0
        image = image.permute([2, 0, 1])
        image = image.unsqueeze(0)
        image = image.cast(self._dtype)

        # VAE encode -> (1, 2 * latent_channels, H/8, W/8) mean|logvar.
        moments = self.encoder(image)

        batch = moments.shape[0]
        h = moments.shape[2]
        w = moments.shape[3]
        c = self._latent_channels

        # Extract the mode (mean half of the mean|logvar concatenation).
        mean = moments[:, :c, :, :]

        # Patchify by (patch_h, patch_w): (1, C, H, W) ->
        # (1, C * patch_h * patch_w, H / patch_h, W / patch_w).  The
        # rebind asserts H and W (the post-encoder spatial dims) are
        # divisible by patch_h / patch_w, which lets the reshape below
        # typecheck under symbolic shapes.
        ph = self._patch_h
        pw = self._patch_w
        h2 = h // ph
        w2 = w // pw
        mean = F.rebind(mean, [batch, c, h2 * ph, w2 * pw])
        mean = mean.reshape((batch, c, h2, ph, w2, pw))
        mean = mean.permute([0, 1, 3, 5, 2, 4])
        mean = mean.reshape((batch, self._num_channels, h2, w2))

        # Learnable BN: (latents - bn_mean) / sqrt(bn_var + eps).
        bn_mean = self.bn_mean.reshape((1, self._num_channels, 1, 1))
        bn_std = (self.bn_var + self._batch_norm_eps) ** 0.5
        bn_std = bn_std.reshape((1, self._num_channels, 1, 1))
        latents = (mean - bn_mean) / bn_std

        # Pack: (1, num_channels, H_p, W_p) -> (1, H_p * W_p, num_channels).
        latents = latents.reshape((batch, self._num_channels, h2 * w2))
        latents = latents.permute([0, 2, 1])

        return latents

    def decode(
        self,
        latents_bsc: Tensor,
        h_carrier: Tensor,
        w_carrier: Tensor,
    ) -> Tensor:
        """Decode packed latents into an ``(B, H, W, 3)`` uint8 image on CPU.

        Args:
            latents_bsc: Packed latents ``(B, packed_h * packed_w, C*4)``
                in the model dtype.
            h_carrier: Shape carrier of length ``packed_h``; content
                unused.
            w_carrier: Shape carrier of length ``packed_w``; content
                unused.

        Returns:
            Decoded uint8 image on CPU, shape ``(B, packed_h * vae_scale
            * patch_h, packed_w * vae_scale * patch_w, 3)``.
        """
        batch = latents_bsc.shape[0]
        c = latents_bsc.shape[2]
        h = h_carrier.shape[0]
        w = w_carrier.shape[0]

        # Reshape packed (B, S, C) -> spatial (B, H, W, C); the rebind
        # asserts ``S == H * W`` so the symbolic-shape reshape typechecks.
        latents_bsc = F.rebind(latents_bsc, [batch, h * w, c])
        latents_bhwc = latents_bsc.reshape((batch, h, w, c))

        # (B, H, W, C) -> (B, C, H, W).
        latents = latents_bhwc.permute([0, 3, 1, 2])

        # BN denormalization: x = x * sqrt(var + eps) + mean.
        bn_mean_r = self.bn_mean.reshape((1, self._num_channels, 1, 1))
        bn_std = (self.bn_var + self._batch_norm_eps) ** 0.5
        bn_std = bn_std.reshape((1, self._num_channels, 1, 1))
        latents = latents * bn_std + bn_mean_r

        # Unpatchify by (patch_h, patch_w): (B, C, H, W) ->
        # (B, C / (patch_h * patch_w), H * patch_h, W * patch_w).
        ph = self._patch_h
        pw = self._patch_w
        patch_area = ph * pw
        latents = latents.reshape((batch, c // patch_area, ph, pw, h, w))
        latents = latents.permute([0, 1, 4, 2, 5, 3])
        latents = latents.reshape((batch, c // patch_area, h * ph, w * pw))

        # VAE decode: (B, C / patch_area, H * patch_h, W * patch_w)
        # -> (B, 3, H_pixels, W_pixels).
        decoded = self.decoder(latents)

        # (B, 3, H, W) -> (B, H, W, 3).
        decoded = decoded.permute([0, 2, 3, 1])

        # Normalize [-1, 1] -> [0, 255] uint8.  Upcast to float32 for the
        # multiply so bf16 rounding doesn't shift pixel values; round
        # before the truncating cast so we don't bias every pixel down by
        # ~0.5.  Matches the diffusers AutoencoderKL post-processor.
        decoded = decoded.cast(DType.float32)
        decoded = decoded * 0.5 + 0.5
        decoded = F.max(decoded, 0.0)
        decoded = F.min(decoded, 1.0)
        decoded = decoded * 255.0
        decoded = F.round(decoded)
        decoded = decoded.cast(DType.uint8)
        # ``Tensor.to`` takes a ``Device``-like (Device / DeviceMesh /
        # DeviceMapping), not a ``DeviceRef``; use the concrete CPU
        # device from ``max.driver``.
        return decoded.to(CPU())

    def input_types(self) -> tuple[TensorType, ...]:
        """Encoder input type, exposed to the parent's ``input_types``.

        Returns the single ``(H, W, 3)`` uint8 ``TensorType`` consumed
        by :meth:`encode`.  The decoder is invoked from within the
        parent's ``forward`` with tensors managed by the parent (final
        latents from the denoise loop plus the ``h_carrier`` /
        ``w_carrier`` shape signals already declared at the parent
        level), so it has no public input_types of its own here.
        """
        return (
            TensorType(
                DType.uint8,
                shape=["image_height", "image_width", 3],
                device=self._device_ref,
            ),
        )

    @staticmethod
    def adapt_loader(loader: WeightLoader) -> WeightLoader:
        """Resolve this Module's parameter queries to HF VAE keys.

        Inverse of the legacy key mapping, unioning the encoder and
        decoder adapters because both halves live on a single Module:

        * ``encoder.quant_conv.*`` -> ``quant_conv.*``
        * ``encoder.*`` -> ``encoder.*``
        * ``decoder.post_quant_conv.*`` -> ``post_quant_conv.*``
        * ``decoder.*`` -> ``decoder.*``
        * ``bn_mean`` -> ``bn.running_mean``
        * ``bn_var`` -> ``bn.running_var``

        Pure per-key query translation: the Module only queries the
        parameters it declares, so checkpoint-only keys are never
        resolved.  Float32 BN statistics from the HF checkpoint are cast
        to the Module's parameter dtype by the loader's ``auto_cast``
        mode (MAX pipelines opt in by default via
        ``MODULAR_AUTO_CAST_WEIGHTS=true``; see
        :func:`max.pipelines.modeling.weights.weight_loading.auto_cast_weights_from_env`).
        Without ``auto_cast`` the loader is strict and dtype mismatch
        raises.
        """

        def to_source(name: str) -> str:
            if name == "bn_mean":
                return "bn.running_mean"
            if name == "bn_var":
                return "bn.running_var"
            if name.startswith("encoder.quant_conv."):
                return name.removeprefix("encoder.")
            if name.startswith("decoder.post_quant_conv."):
                return name.removeprefix("decoder.")
            return name

        return rename(loader, to_source)
