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

"""VAE wrapper component for WanExecutor."""

from __future__ import annotations

import logging

import numpy as np
from max.driver import Buffer, load_devices
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.weights import load_weights
from max.pipelines.lib.config.model_config import (
    _resolve_component_encoding_and_weights,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.profiler import Tracer, traced

from ...autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWanModel,
    _buffer_to_numpy_f32,
    _numpy_f32_to_buffer,
)

logger = logging.getLogger("max.pipelines")


class VaeWrapper:
    """Thin wrapper around AutoencoderKLWanModel with denormalization.

    Holds the existing VAE ComponentModel directly and provides:
    - ``decode``: denormalize latents + VAE decode
    - ``encode_i2v_condition``: encode image for I2V conditioning

    The denormalization graph (``latents * std + mean`` + cast) is compiled
    once at init time.
    """

    @traced(message="VaeWrapper.__init__")
    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
    ) -> None:
        self._session = session

        vae_config_entry = manifest["vae"]
        config_dict = vae_config_entry.huggingface_config.to_dict()
        resolved_encoding, resolved_weight_path = (
            _resolve_component_encoding_and_weights(vae_config_entry)
        )
        encoding = resolved_encoding or "bfloat16"
        devices = load_devices(vae_config_entry.device_specs)

        # Load the VAE using the existing ComponentModel.
        paths = vae_config_entry.resolved_weight_paths(resolved_weight_path)
        weights = load_weights(paths)
        self._vae = AutoencoderKLWanModel(
            config=config_dict,
            encoding=encoding,
            devices=devices,
            weights=weights,
            session=session,
        )

        self._device = devices[0]
        self._dtype = self._vae.config.dtype

        # Precompute denormalization constants.
        z_dim = int(self._vae.config.z_dim)
        mean_arr = np.asarray(
            self._vae.config.latents_mean, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        std_arr = np.asarray(
            self._vae.config.latents_std, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        self._vae_mean_buf = Buffer.from_numpy(mean_arr).to(self._device)
        self._vae_std_buf = Buffer.from_numpy(std_arr).to(self._device)

        # Compile denormalization graph.
        self._denorm_model = self._compile_denorm(z_dim)

        # Compile GPU postprocess graph (scale, clip, cast uint8, permute).
        self._postprocess_model = self._compile_postprocess(
            int(self._vae.config.out_channels)
        )

        # VAE scale factors.
        self.scale_factor_temporal = int(
            getattr(self._vae.config, "scale_factor_temporal", 4) or 4
        )
        self.scale_factor_spatial = int(
            getattr(self._vae.config, "scale_factor_spatial", 8) or 8
        )

    @traced(message="VaeWrapper.decode")
    def decode(
        self,
        latents: Buffer,
        num_frames: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Denormalize latents and decode through VAE.

        Args:
            latents: Latents in f32, shape ``(B, C, T, H, W)``.
            num_frames: Target number of output frames.
            height: Target output height.
            width: Target output width.

        Returns:
            Decoded output as uint8 numpy array.

            - Multi-frame video requests return ``(B, C, T, H, W)`` so the
              pixel-generation pipeline can emit ``OutputVideoContent``.
            - Single-frame requests return ``(B, H, W, C)`` so they continue
              through the image path.
        """
        logger.info("Decoding Wan output")
        # Denormalize: latents * std + mean, cast to model dtype.
        with Tracer("vae_denormalize"):
            denorm_result = self._denorm_model.execute(
                latents, self._vae_std_buf, self._vae_mean_buf
            )
            denorm_latents = denorm_result[0]

        # Decode through VAE (framewise, has internal "wan_vae_decode" trace).
        with Tracer("vae_decode_5d"):
            decoded_video = self._vae.decode_5d(denorm_latents)

        # GPU post-processing: scale to [0, 1], clip, scale to [0, 255], cast
        # to uint8, permute to (B, T, H, W, C), and transfer to CPU. Running
        # these on device means the DtoH DMA moves uint8 (1 byte/elem)
        # instead of bf16 (2 bytes/elem).
        with Tracer("vae_decode_postprocess"):
            decoded_u8_buf = self._postprocess_model.execute(decoded_video)[0]
            decoded_np = np.from_dlpack(decoded_u8_buf)
            target_num_frames = min(decoded_np.shape[1], num_frames)
            decoded_np = decoded_np[:, :target_num_frames, :height, :width, :]

            if target_num_frames > 1:
                video_batch = np.transpose(decoded_np, (0, 4, 1, 2, 3))
                return np.ascontiguousarray(video_batch)

            # Preserve image behavior for single-frame requests.
            image_batch = decoded_np[:, 0, :, :, :]
            return np.ascontiguousarray(image_batch)

    @traced(message="VaeWrapper.encode_i2v_condition")
    def encode_i2v_condition(
        self,
        image: np.ndarray,
        latent_shape: tuple[int, ...],
        num_frames: int,
        height: int,
        width: int,
    ) -> Buffer:
        """Encode input image into I2V condition tensor.

        Creates a ``[B, vae_t + z_dim, T_l, H_l, W_l]`` condition tensor
        by encoding the image via VAE, normalizing, and concatenating
        with a temporal mask.

        Args:
            image: Input image as numpy array (HWC or CHW format).
            latent_shape: Shape of the noise latents ``(B, C, T, H, W)``.
            num_frames: Number of video frames.
            height: Target pixel height.
            width: Target pixel width.

        Returns:
            Condition tensor on device in model dtype.
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        logger.info("Preparing I2V condition")

        # Normalize to [-1, 1] float32, shape [1, 3, H, W].
        image_f32 = image.astype(np.float32) / 127.5 - 1.0
        if image_f32.ndim == 3:
            image_f32 = image_f32.transpose(2, 0, 1)[np.newaxis]

        batch_size = int(latent_shape[0])

        # Resize image to target resolution if needed.
        if image_f32.shape[2] != height or image_f32.shape[3] != width:
            import PIL.Image

            pil_img = PIL.Image.fromarray(
                ((image_f32[0].transpose(1, 2, 0) + 1.0) * 127.5)
                .clip(0, 255)
                .astype(np.uint8)
            )
            pil_img = pil_img.resize(
                (width, height), PIL.Image.Resampling.LANCZOS
            )
            image_f32 = (
                np.array(pil_img).astype(np.float32) / 127.5 - 1.0
            ).transpose(2, 0, 1)[np.newaxis]

        # VAE encode.
        enc_latent = self._vae.encode_zero_padded_video_condition(
            image_f32,
            batch_size=batch_size,
            num_frames=num_frames,
        )
        latent_cond_np = _buffer_to_numpy_f32(enc_latent)

        expected_t = int(latent_shape[2])
        if latent_cond_np.shape[2] != expected_t:
            raise ValueError(
                "VAE encode temporal shape mismatch for I2V condition: "
                f"got {latent_cond_np.shape[2]}, expected {expected_t} "
                f"for num_frames={num_frames}."
            )

        expected_h = int(latent_shape[3])
        expected_w = int(latent_shape[4])
        if (
            latent_cond_np.shape[3] != expected_h
            or latent_cond_np.shape[4] != expected_w
        ):
            raise ValueError(
                "VAE encode spatial shape mismatch for I2V condition: "
                f"got {latent_cond_np.shape[3:5]}, expected "
                f"({expected_h}, {expected_w})."
            )

        # Normalize encoded latents.
        z_dim = self._vae.config.z_dim
        mean = np.array(
            self._vae.config.latents_mean, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        inv_std = 1.0 / np.array(
            self._vae.config.latents_std, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        latent_cond_np = (latent_cond_np - mean) * inv_std

        # Build temporal mask [B, 1, num_frames, H_l, W_l].
        h_latent = latent_cond_np.shape[3]
        w_latent = latent_cond_np.shape[4]

        mask = np.zeros(
            (batch_size, 1, num_frames, h_latent, w_latent),
            dtype=np.float32,
        )
        mask[:, :, 0, :, :] = 1.0  # First frame is conditioned.

        vae_t = self.scale_factor_temporal
        first_mask = np.repeat(mask[:, :, 0:1, :, :], vae_t, axis=2)
        mask_expanded = np.concatenate(
            [first_mask, mask[:, :, 1:, :, :]], axis=2
        )
        mask_expanded = mask_expanded.reshape(
            batch_size, -1, vae_t, h_latent, w_latent
        )
        mask_expanded = mask_expanded.transpose(0, 2, 1, 3, 4)

        # Concat: [mask, latent_condition] -> [B, vae_t+z_dim, T_l, H_l, W_l].
        condition = np.concatenate(
            [mask_expanded, latent_cond_np], axis=1
        ).astype(np.float32)

        return _numpy_f32_to_buffer(condition, self._dtype, self._device)

    # -- Internal helpers ---------------------------------------------------

    def _compile_denorm(self, z_dim: int) -> Model:
        """Compile the VAE latent denormalization graph."""
        model_dtype = self._dtype
        input_types = [
            TensorType(
                DType.float32,
                ["batch", z_dim, "f", "h", "w"],
                device=self._device,
            ),
            TensorType(DType.float32, [1, z_dim, 1, 1, 1], device=self._device),
            TensorType(DType.float32, [1, z_dim, 1, 1, 1], device=self._device),
        ]
        with Graph("wan_denorm", input_types=input_types) as g:
            latents, std, mean = (v.tensor for v in g.inputs)
            result = ops.cast(latents * std + mean, model_dtype)
            g.output(result)
        return self._session.load(g)

    def _compile_postprocess(self, out_channels: int) -> Model:
        """Compile the VAE decoder postprocess graph.

        Takes a decoded ``(B, C, T, H, W)`` tensor in the VAE dtype and
        produces a host-side ``(B, T, H, W, C)`` uint8 tensor: upcast to
        f32 for precision, scale from ``[-1, 1]`` to ``[0, 255]``, clip,
        cast to uint8, permute, and transfer to CPU. Fusing the transfer
        into the graph means the DtoH DMA moves 1 byte/elem instead of 2.
        """
        input_types = [
            TensorType(
                self._dtype,
                ["batch", out_channels, "t", "h", "w"],
                device=self._device,
            ),
        ]
        with Graph("wan_vae_postprocess", input_types=input_types) as g:
            x = g.inputs[0].tensor
            # Upcast to f32 before the *255 so bf16 rounding doesn't shift
            # pixel values; matches the flux2 VAE decoder precision path.
            # Round before the uint8 cast so the truncating cast doesn't bias
            # every pixel down by ~0.5; diffusers' image processor does
            # `(x * 255).round().astype(uint8)`.
            x = ops.cast(x, DType.float32)
            x = x * 0.5 + 0.5
            x = ops.max(x, 0.0)
            x = ops.min(x, 1.0)
            x = x * 255.0
            x = ops.round(x)
            x = ops.cast(x, DType.uint8)
            # (B, C, T, H, W) -> (B, T, H, W, C).
            x = ops.permute(x, [0, 2, 3, 4, 1])
            x = ops.transfer_to(x, DeviceRef.CPU())
            g.output(x)
        return self._session.load(g)
