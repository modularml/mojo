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

"""QwenImage diffusion pipeline.

Key differences from Flux2Pipeline:
- True CFG with two forward passes (positive + negative prompts)
- No guidance embedding (timestep only, not timestep+guidance)
- Latent normalization via latents_mean/latents_std instead of BatchNorm
- Text encoder returns last hidden state (not multiple layers)
- 3D position IDs (T, H, W) instead of 4D (T, H, W, L)
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.graph import TensorType, TensorValue, ops
from max.graph.ops import shape_to_tensor
from max.pipelines.context import PixelContext, TokenBuffer
from max.pipelines.diffusion.interface import DiffusionPipeline, max_compile
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.profiler import Tracer, traced

from ..autoencoders.autoencoder_kl_qwen_image import AutoencoderKLQwenImageModel
from ..qwen2_5vl.encoder import Qwen25VLEncoderModel
from .model import QwenImageTransformerModel


@dataclass(kw_only=True)
class QwenImageModelInputs:
    """QwenImage-specific model inputs.

    QwenImage is not guidance-distilled — use ``--true-cfg-scale``
    (not ``--guidance-scale``) to control classifier-free guidance.
    """

    tokens: TokenBuffer
    """Primary encoder token buffer."""

    tokens_2: TokenBuffer | None = None
    """Secondary encoder token buffer (for dual-encoder models)."""

    negative_tokens: TokenBuffer | None = None
    """Negative prompt tokens for the primary encoder."""

    negative_tokens_2: TokenBuffer | None = None
    """Negative prompt tokens for the secondary encoder."""

    timesteps: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed denoising timestep schedule."""

    sigmas: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed sigma schedule for denoising."""

    latents: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Initial latent noise tensor."""

    latent_image_ids: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Latent image positional identifiers."""

    true_cfg_scale: float = 1.0
    """True CFG scale (enabled when > 1.0 with negative prompt)."""

    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1

    @classmethod
    def from_context(cls, context: PixelContext) -> "QwenImageModelInputs":
        """Build QwenImageModelInputs from a PixelContext."""
        return cls(
            tokens=context.tokens,
            tokens_2=context.tokens_2,
            negative_tokens=context.negative_tokens,
            negative_tokens_2=context.negative_tokens_2,
            timesteps=context.timesteps,
            sigmas=context.sigmas,
            latents=context.latents,
            latent_image_ids=context.latent_image_ids,
            true_cfg_scale=context.true_cfg_scale,
            width=context.width,
            height=context.height,
            num_inference_steps=context.num_inference_steps,
            num_images_per_prompt=context.num_images_per_prompt,
        )


@dataclass
class QwenImagePipelineOutput:
    """Container for QwenImage pipeline results."""

    images: np.ndarray | list[np.ndarray | Buffer]


@dataclass
class QwenImageCache:
    """Runtime cache for reusable Qwen image buffers."""

    sigmas: dict[str, Buffer] = field(default_factory=dict)
    text_ids: dict[str, Buffer] = field(default_factory=dict)
    shape_carriers: dict[int, Buffer] = field(default_factory=dict)
    cfg_scales: dict[float, Buffer] = field(default_factory=dict)
    latent_image_ids: dict[tuple[int, int, int], Buffer] = field(
        default_factory=dict
    )
    prompt_tokens: dict[tuple[int, ...], Buffer] = field(default_factory=dict)


class QwenImagePipeline(DiffusionPipeline):
    """Diffusion pipeline for QwenImage text-to-image generation.

    Wires together:
    - Qwen2.5-VL text encoder
    - QwenImage transformer denoiser (60 dual-stream blocks)
    - QwenImage 3D VAE (with latents_mean/std normalization)
    """

    vae: AutoencoderKLQwenImageModel
    text_encoder: Qwen25VLEncoderModel
    transformer: QwenImageTransformerModel

    components = {
        "vae": AutoencoderKLQwenImageModel,
        "text_encoder": Qwen25VLEncoderModel,
        "transformer": QwenImageTransformerModel,
    }

    def init_remaining_components(self) -> None:
        """Initialize derived attributes that depend on loaded components."""
        # QwenImage VAE uses dim_mult [1,2,4,4] with 3 downsample stages
        # Spatial scale factor = 2^3 = 8
        self.vae_scale_factor = 8

        self._compile_runtime_helpers()
        self._compile_cfg_fastpath_helpers()
        self.cache: QwenImageCache = QwenImageCache()

    def prepare_inputs(self, context: PixelContext) -> QwenImageModelInputs:
        """Convert a PixelContext into QwenImageModelInputs."""
        return QwenImageModelInputs.from_context(context)

    def _compile_runtime_helpers(self) -> None:
        """Compile the core runtime helper graphs used by QwenImage."""
        device = self.transformer.devices[0]
        self.cached_patchify_and_pack = max_compile(
            self._patchify_and_pack,
            input_types=[
                TensorType(
                    DType.float32,
                    shape=["batch", "channels", "height", 2, "width", 2],
                    device=device,
                ),
            ],
        )

        self.cached_prepare_scheduler = max_compile(
            self.prepare_scheduler,
            input_types=[
                TensorType(
                    DType.float32,
                    shape=["num_sigmas"],
                    device=device,
                ),
            ],
        )

        dtype = self.transformer.config.dtype
        packed_channels = self.transformer.config.in_channels
        self.cached_scheduler_step = max_compile(
            self.scheduler_step,
            input_types=[
                TensorType(
                    dtype, shape=["batch", "seq", "channels"], device=device
                ),
                TensorType(
                    dtype,
                    shape=["batch", "pred_seq", "channels"],
                    device=device,
                ),
                TensorType(DType.float32, shape=[1], device=device),
            ],
        )

        z_dim = 16  # VAE latent channels
        self.cached_postprocess_latents = max_compile(
            self._postprocess_latents,
            input_types=[
                TensorType(
                    dtype,
                    shape=["batch", "height", "width", packed_channels],
                    device=device,
                ),
                TensorType(dtype, shape=[z_dim], device=device),
                TensorType(dtype, shape=[z_dim], device=device),
            ],
        )

        self.cached_cfg_blend = max_compile(
            self._cfg_blend,
            input_types=[
                TensorType(
                    dtype,
                    shape=["batch", "seq", "channels"],
                    device=device,
                ),
                TensorType(
                    dtype,
                    shape=["batch", "seq", "channels"],
                    device=device,
                ),
                TensorType(DType.float32, shape=[1], device=device),
            ],
        )

        self.cached_reshape_latents = max_compile(
            self._reshape_latents,
            input_types=[
                TensorType(
                    dtype,
                    shape=["batch", "seq", packed_channels],
                    device=device,
                ),
                TensorType(DType.float32, shape=["packed_h"], device=CPU()),
                TensorType(DType.float32, shape=["packed_w"], device=CPU()),
            ],
        )

        hidden_size = self.text_encoder.config.hidden_size
        text_dtype = self.text_encoder.config.dtype
        text_device = self.text_encoder.devices[0]
        self.cached_trim_prompt_embeddings = max_compile(
            self._trim_prompt_embeddings,
            input_types=[
                TensorType(
                    text_dtype,
                    shape=["seq", hidden_size],
                    device=text_device,
                )
            ],
        )

        def duplicate_batch(value: TensorValue) -> TensorValue:
            return ops.concat([value, value], axis=0)

        self.cached_duplicate_prompt_embeddings = max_compile(
            duplicate_batch,
            input_types=[
                TensorType(
                    text_dtype,
                    shape=[1, "trimmed_seq_len", hidden_size],
                    device=text_device,
                )
            ],
        )

    def _compile_cfg_fastpath_helpers(self) -> None:
        """Compile the small helper graphs used by the CFG fast path."""

        def duplicate_batch(value: TensorValue) -> TensorValue:
            return ops.concat([value, value], axis=0)

        def concat_batch_pair(
            first_value: TensorValue,
            second_value: TensorValue,
        ) -> TensorValue:
            return ops.concat([first_value, second_value], axis=0)

        def split_cfg_predictions(
            batched_predictions: TensorValue,
        ) -> tuple[TensorValue, TensorValue]:
            positive_prediction = ops.slice_tensor(
                batched_predictions,
                [slice(0, 1), slice(None), slice(None)],
            )
            negative_prediction = ops.slice_tensor(
                batched_predictions,
                [slice(1, 2), slice(None), slice(None)],
            )
            return positive_prediction, negative_prediction

        text_dtype = self.text_encoder.config.dtype
        text_device = self.text_encoder.devices[0]
        hidden_size = self.text_encoder.config.hidden_size
        self.cached_concat_cfg_prompt_embeddings = max_compile(
            concat_batch_pair,
            input_types=[
                TensorType(
                    text_dtype,
                    shape=[1, "trimmed_seq_len", hidden_size],
                    device=text_device,
                ),
                TensorType(
                    text_dtype,
                    shape=[1, "trimmed_seq_len", hidden_size],
                    device=text_device,
                ),
            ],
        )

        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        packed_channels = self.transformer.config.in_channels
        self.cached_duplicate_cfg_latents = max_compile(
            duplicate_batch,
            input_types=[
                TensorType(
                    dtype,
                    shape=[1, "seq", packed_channels],
                    device=device,
                )
            ],
        )

        self.cached_concat_cfg_ids = max_compile(
            concat_batch_pair,
            input_types=[
                TensorType(
                    DType.int64,
                    shape=[1, "seq", 3],
                    device=device,
                ),
                TensorType(
                    DType.int64,
                    shape=[1, "seq", 3],
                    device=device,
                ),
            ],
        )

        self.cached_duplicate_cfg_timesteps = max_compile(
            duplicate_batch,
            input_types=[TensorType(dtype, shape=[1], device=device)],
        )

        self.cached_duplicate_cfg_ids = max_compile(
            duplicate_batch,
            input_types=[
                TensorType(
                    DType.int64,
                    shape=[1, "seq", 3],
                    device=device,
                )
            ],
        )

        self.cached_split_cfg_predictions = max_compile(
            split_cfg_predictions,
            input_types=[
                TensorType(
                    dtype,
                    shape=[2, "seq", packed_channels],
                    device=device,
                )
            ],
        )

    def _cfg_blend(
        self,
        cond_pred: TensorValue,
        uncond_pred: TensorValue,
        cfg_scale: TensorValue,
    ) -> TensorValue:
        scale = ops.cast(cfg_scale, cond_pred.dtype)
        return uncond_pred + scale * (cond_pred - uncond_pred)

    # Number of chat template prefix tokens to drop from encoder output.
    # Matches diffusers' prompt_template_encode_start_idx = 34.
    PROMPT_TEMPLATE_DROP_IDX = 34

    def _trim_prompt_embeddings(
        self, hidden_states: TensorValue
    ) -> TensorValue:
        trimmed = ops.slice_tensor(
            hidden_states,
            [slice(self.PROMPT_TEMPLATE_DROP_IDX, None), slice(None)],
        )
        return ops.unsqueeze(trimmed, 0)

    def prepare_prompt_embeddings(
        self,
        tokens: TokenBuffer,
        num_images_per_prompt: int = 1,
    ) -> Buffer:
        """Create prompt embeddings from tokens.

        QwenImage uses the last hidden state from the text encoder (layer -1).
        The tokens include a chat template prefix (~34 tokens) that must be
        dropped from the encoder output to match diffusers' behavior.
        """
        device = self.text_encoder.devices[0]
        text_input_ids_np = np.asarray(tokens.array).flatten()
        token_key = tuple(int(token) for token in text_input_ids_np.tolist())
        if token_key not in self.cache.prompt_tokens:
            self.cache.prompt_tokens[token_key] = Buffer.from_dlpack(
                np.ascontiguousarray(text_input_ids_np)
            ).to(device)
        token_buf = self.cache.prompt_tokens[token_key]

        hidden_states_all = self.text_encoder(token_buf)
        hs_buf = hidden_states_all[-1]

        trimmed = self.cached_trim_prompt_embeddings(hs_buf)
        if num_images_per_prompt == 1:
            return trimmed
        if num_images_per_prompt == 2:
            return self.cached_duplicate_prompt_embeddings(trimmed)

        hs_cpu = hs_buf.to(CPU())
        if self.text_encoder.config.dtype == DType.bfloat16:
            hs_u16 = np.from_dlpack(
                hs_cpu.view(dtype=DType.uint16, shape=hs_cpu.shape)
            )
            hs_np = (hs_u16.astype(np.uint32) << 16).view(np.float32)
        else:
            hs_np = np.from_dlpack(hs_cpu).astype(np.float32)

        hs_np = hs_np[self.PROMPT_TEMPLATE_DROP_IDX :]
        hs_np = hs_np[np.newaxis, :, :]

        if num_images_per_prompt != 1:
            hs_np = np.broadcast_to(
                hs_np,
                (num_images_per_prompt, hs_np.shape[1], hs_np.shape[2]),
            ).copy()

        if self.text_encoder.config.dtype == DType.bfloat16:
            result_u16 = float32_to_bfloat16_as_uint16(
                np.ascontiguousarray(hs_np)
            )
            buf = Buffer.from_numpy(result_u16).to(device)
            return buf.view(dtype=DType.bfloat16, shape=hs_np.shape)

        return Buffer.from_numpy(np.ascontiguousarray(hs_np)).to(device)

    @staticmethod
    def _prepare_text_ids(
        batch_size: int,
        seq_len: int,
        device: Device,
        max_vid_index: int = 0,
    ) -> Buffer:
        """Create 3D text position IDs in (T, H, W) format.

        QwenImage text tokens use positions [max_vid_index, max_vid_index+1, ...]
        for all 3 axes (matching diffusers scale_rope=True convention).
        """
        tok_positions = np.arange(seq_len, dtype=np.int64) + max_vid_index
        coords = np.stack(
            [tok_positions, tok_positions, tok_positions], axis=-1
        )
        text_ids = np.broadcast_to(
            coords[np.newaxis, :, :],
            (batch_size, coords.shape[0], coords.shape[1]),
        ).copy()
        return Buffer.from_dlpack(text_ids).to(device)

    def _reshape_latents(
        self,
        latents_bsc: TensorValue,
        h_carrier: TensorValue,
        w_carrier: TensorValue,
    ) -> TensorValue:
        batch = latents_bsc.shape[0]
        h = h_carrier.shape[0]
        w = w_carrier.shape[0]
        channels = latents_bsc.shape[2]
        latents_bsc = ops.rebind(latents_bsc, [batch, h * w, channels])
        return ops.reshape(latents_bsc, (batch, h, w, channels))

    def _get_shape_carriers(
        self, h_latent: int, w_latent: int
    ) -> tuple[Buffer, Buffer]:
        for n in (h_latent, w_latent):
            if n not in self.cache.shape_carriers:
                self.cache.shape_carriers[n] = Buffer.from_dlpack(
                    np.empty(n, dtype=np.float32)
                )
        return (
            self.cache.shape_carriers[h_latent],
            self.cache.shape_carriers[w_latent],
        )

    def decode_latents(
        self,
        latents: Buffer,
        height: int,
        width: int,
        output_type: Literal["np", "latent"] = "np",
    ) -> np.ndarray | Buffer:
        """Decode packed latents into an image array."""
        if output_type == "latent":
            return latents

        h_latent = height // (self.vae_scale_factor * 2)
        w_latent = width // (self.vae_scale_factor * 2)

        latents_mean = self.vae.latents_mean_tensor
        latents_std = self.vae.latents_std_tensor
        if latents_mean is None or latents_std is None:
            raise ValueError("VAE latents_mean/latents_std not loaded.")

        h_carrier, w_carrier = self._get_shape_carriers(h_latent, w_latent)
        latents_bhwc = self.cached_reshape_latents(
            latents, h_carrier, w_carrier
        )

        latents_decoded = self.cached_postprocess_latents(
            latents_bhwc, latents_mean, latents_std
        )

        decoded = self.vae.decode(latents_decoded)
        return self._image_to_flat_hwc(self._to_numpy(decoded))

    def _postprocess_latents(
        self,
        latents_bhwc: TensorValue,
        latents_mean: TensorValue,
        latents_std: TensorValue,
    ) -> TensorValue:
        """Unpatchify and denormalize latents for VAE decoding."""
        batch = latents_bhwc.shape[0]
        h = latents_bhwc.shape[1]
        w = latents_bhwc.shape[2]
        c = latents_bhwc.shape[3]
        z_dim = c // 4  # 16

        # Permute (B, H, W, C) -> (B, C, H, W)
        latents = ops.permute(latents_bhwc, [0, 3, 1, 2])

        # Unpatchify first: (B, C, H, W) -> (B, z_dim, H*2, W*2)
        latents = ops.reshape(latents, (batch, z_dim, 2, 2, h, w))
        latents = ops.permute(latents, [0, 1, 4, 2, 5, 3])
        latents = ops.reshape(latents, (batch, z_dim, h * 2, w * 2))

        # Then denormalize using latents_mean/std (shape [z_dim])
        mean_r = ops.reshape(latents_mean, (1, z_dim, 1, 1))
        std_r = ops.reshape(latents_std, (1, z_dim, 1, 1))
        latents = latents * std_r + mean_r

        return latents

    def _to_numpy(self, image: Any) -> np.ndarray:
        cpu_image = image.to(CPU())
        try:
            return np.from_dlpack(cpu_image).astype(np.float32)
        except (RuntimeError, TypeError):
            # bfloat16 not supported by numpy, cast via v1 Tensor
            from max.experimental.tensor import Tensor as _Tensor

            if isinstance(cpu_image, _Tensor):
                return np.from_dlpack(cpu_image.cast(DType.float32)).astype(
                    np.float32
                )
            # Buffer bfloat16: wrap in v1 Tensor to cast
            t = _Tensor(storage=cpu_image)
            return np.from_dlpack(t.cast(DType.float32)).astype(np.float32)

    @staticmethod
    def _image_to_flat_hwc(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        while img.ndim > 3:
            img = img.squeeze(0)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img.astype(np.float32, copy=False)

    def preprocess_latents(
        self,
        latents: npt.NDArray[np.float32],
        latent_image_ids: npt.NDArray[np.float32],
    ) -> tuple[Buffer, Buffer]:
        latents_np = np.asarray(latents)
        batch = latents_np.shape[0]
        c = latents_np.shape[1]
        h = latents_np.shape[2]
        w = latents_np.shape[3]
        latents_6d = latents_np.reshape(batch, c, h // 2, 2, w // 2, 2)
        latents_6d_buf = Buffer.from_dlpack(
            np.ascontiguousarray(latents_6d)
        ).to(self.transformer.devices[0])
        latents_packed = self.cached_patchify_and_pack(latents_6d_buf)

        latent_image_ids_int64 = np.asarray(latent_image_ids, dtype=np.int64)
        ids_key = (
            int(latent_image_ids_int64.shape[0]),
            int(latent_image_ids_int64.shape[1]),
            int(latent_image_ids_int64.shape[2]),
        )
        if ids_key not in self.cache.latent_image_ids:
            self.cache.latent_image_ids[ids_key] = Buffer.from_dlpack(
                latent_image_ids_int64
            ).to(self.transformer.devices[0])
        latent_image_ids_buf = self.cache.latent_image_ids[ids_key]
        return latents_packed, latent_image_ids_buf

    def _patchify_and_pack(self, latents: TensorValue) -> TensorValue:
        """Patchify (B,C,H,W)->(B,C*4,H//2,W//2) then pack to (B,H//2*W//2,C*4)."""
        latents = ops.cast(latents, self.transformer.config.dtype)
        batch = latents.shape[0]
        c = latents.shape[1]
        h2 = latents.shape[2]
        w2 = latents.shape[4]

        latents = ops.permute(latents, [0, 1, 3, 5, 2, 4])
        latents = ops.reshape(latents, (batch, c * 4, h2, w2))

        c4 = c * 4
        latents = ops.reshape(latents, (batch, c4, h2 * w2))
        latents = ops.permute(latents, [0, 2, 1])

        return latents

    def scheduler_step(
        self,
        latents: TensorValue,
        noise_pred: TensorValue,
        dt: TensorValue,
    ) -> TensorValue:
        """Apply a single Euler update step."""
        num_noise_tokens = shape_to_tensor([latents.shape[1]])
        latents_sliced = ops.slice_tensor(
            latents,
            [
                slice(None),
                (slice(0, num_noise_tokens), "num_tokens"),
                slice(None),
            ],
        )
        noise_pred_sliced = ops.slice_tensor(
            noise_pred,
            [
                slice(None),
                (slice(0, num_noise_tokens), "num_tokens"),
                slice(None),
            ],
        )
        latents_dtype = latents_sliced.dtype
        latents_sliced = ops.cast(latents_sliced, DType.float32)
        latents_sliced = latents_sliced + dt * noise_pred_sliced
        return ops.cast(latents_sliced, latents_dtype)

    def prepare_scheduler(
        self, sigmas: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Precompute timesteps and dt values from sigmas."""
        sigmas_curr = ops.slice_tensor(sigmas, [slice(0, -1)])
        sigmas_next = ops.slice_tensor(sigmas, [slice(1, None)])
        all_dt = sigmas_next - sigmas_curr
        all_timesteps = ops.cast(sigmas_curr, self.transformer.config.dtype)
        return all_timesteps, all_dt

    @traced
    def execute(  # type: ignore[override]
        self,
        model_inputs: QwenImageModelInputs,
        output_type: Literal["np", "latent"] = "np",
    ) -> QwenImagePipelineOutput:
        """Run the QwenImage denoising loop and decode outputs.

        Supports true classifier-free guidance with separate positive and
        negative prompt forward passes.
        """
        # Phase 1: prompt and latent preparation.
        prompt_embeds = self.prepare_prompt_embeddings(
            tokens=model_inputs.tokens,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
        )
        batch_size = int(prompt_embeds.shape[0])
        device = self.transformer.devices[0]

        latents, latent_image_ids = self.preprocess_latents(
            model_inputs.latents, model_inputs.latent_image_ids
        )

        h_latent = model_inputs.height // (self.vae_scale_factor * 2)
        w_latent = model_inputs.width // (self.vae_scale_factor * 2)
        max_vid_index = max(h_latent // 2, w_latent // 2)
        text_ids_key = (
            f"{batch_size}_{int(prompt_embeds.shape[1])}_{max_vid_index}"
        )
        if text_ids_key not in self.cache.text_ids:
            self.cache.text_ids[text_ids_key] = self._prepare_text_ids(
                batch_size=batch_size,
                seq_len=int(prompt_embeds.shape[1]),
                device=device,
                max_vid_index=max_vid_index,
            )
        text_ids = self.cache.text_ids[text_ids_key]

        # Phase 2: CFG setup.
        do_true_cfg = (
            model_inputs.true_cfg_scale > 1.0
            and model_inputs.negative_tokens is not None
        )
        negative_prompt_embeds: Buffer | None = None
        negative_text_ids: Buffer | None = None
        cfg_scale_buf: Buffer | None = None
        batched_prompt_embeds: Buffer | None = None
        batched_text_ids: Buffer | None = None
        batched_latent_image_ids: Buffer | None = None
        if do_true_cfg and model_inputs.negative_tokens is not None:
            negative_prompt_embeds = self.prepare_prompt_embeddings(
                tokens=model_inputs.negative_tokens,
                num_images_per_prompt=model_inputs.num_images_per_prompt,
            )
            negative_text_ids_key = f"{batch_size}_{int(negative_prompt_embeds.shape[1])}_{max_vid_index}"
            if negative_text_ids_key not in self.cache.text_ids:
                self.cache.text_ids[negative_text_ids_key] = (
                    self._prepare_text_ids(
                        batch_size=batch_size,
                        seq_len=int(negative_prompt_embeds.shape[1]),
                        device=device,
                        max_vid_index=max_vid_index,
                    )
                )
            negative_text_ids = self.cache.text_ids[negative_text_ids_key]

            cfg_scale = float(model_inputs.true_cfg_scale)
            if cfg_scale not in self.cache.cfg_scales:
                self.cache.cfg_scales[cfg_scale] = Buffer.from_dlpack(
                    np.array([cfg_scale], dtype=np.float32)
                ).to(device)
            cfg_scale_buf = self.cache.cfg_scales[cfg_scale]

            if (
                batch_size == 1
                and prompt_embeds.shape[1] == negative_prompt_embeds.shape[1]
            ):
                batched_prompt_embeds = (
                    self.cached_concat_cfg_prompt_embeddings(
                        prompt_embeds,
                        negative_prompt_embeds,
                    )
                )
                batched_text_ids = self.cached_concat_cfg_ids(
                    text_ids,
                    negative_text_ids,
                )
                batched_latent_image_ids = self.cached_duplicate_cfg_ids(
                    latent_image_ids
                )

        # Phase 3: scheduler setup.
        sigmas_key = (
            f"{model_inputs.num_inference_steps}_{int(latents.shape[1])}"
        )
        if sigmas_key not in self.cache.sigmas:
            self.cache.sigmas[sigmas_key] = Buffer.from_dlpack(
                model_inputs.sigmas
            ).to(device)
        sigmas = self.cache.sigmas[sigmas_key]

        with Tracer("prepare_scheduler"):
            all_timesteps, all_dts = self.cached_prepare_scheduler(sigmas)
            timesteps_seq = all_timesteps.driver_tensor
            dts_seq = all_dts.driver_tensor

        # Phase 4: denoising loop.
        with Tracer("denoising_loop"):
            for i in range(model_inputs.num_inference_steps):
                with Tracer(f"denoising_step_{i}"):
                    timestep = timesteps_seq[i : i + 1]
                    dt = dts_seq[i : i + 1]

                    if (
                        batched_prompt_embeds is not None
                        and batched_text_ids is not None
                        and batched_latent_image_ids is not None
                        and cfg_scale_buf is not None
                    ):
                        with Tracer("transformer_cfg"):
                            batched_predictions = self.transformer(
                                self.cached_duplicate_cfg_latents(latents),
                                batched_prompt_embeds,
                                self.cached_duplicate_cfg_timesteps(timestep),
                                batched_latent_image_ids,
                                batched_text_ids,
                            )[0]
                        positive_prediction, negative_prediction = (
                            self.cached_split_cfg_predictions(
                                batched_predictions
                            )
                        )
                        with Tracer("cfg_blend"):
                            noise_pred = self.cached_cfg_blend(
                                positive_prediction,
                                negative_prediction,
                                cfg_scale_buf,
                            )
                    else:
                        with Tracer("transformer_pos"):
                            noise_pred = self.transformer(
                                latents,
                                prompt_embeds,
                                timestep,
                                latent_image_ids,
                                text_ids,
                            )[0]

                        if (
                            do_true_cfg
                            and negative_prompt_embeds is not None
                            and negative_text_ids is not None
                            and cfg_scale_buf is not None
                        ):
                            with Tracer("transformer_neg"):
                                negative_prediction = self.transformer(
                                    latents,
                                    negative_prompt_embeds,
                                    timestep,
                                    latent_image_ids,
                                    negative_text_ids,
                                )[0]
                            with Tracer("cfg_blend"):
                                noise_pred = self.cached_cfg_blend(
                                    noise_pred,
                                    negative_prediction,
                                    cfg_scale_buf,
                                )

                    with Tracer("scheduler_step"):
                        latents = self.cached_scheduler_step(
                            latents,
                            noise_pred,
                            dt,
                        )

        # Phase 5: decode outputs.
        image_list: list[np.ndarray | Buffer] = []
        if batch_size == 1:
            image_list.append(
                self.decode_latents(
                    latents,
                    model_inputs.height,
                    model_inputs.width,
                    output_type=output_type,
                )
            )
        else:
            latents_np = np.from_dlpack(latents.to(CPU())).astype(np.float32)
            for batch_index in range(batch_size):
                batch_latents = Buffer.from_dlpack(
                    np.ascontiguousarray(
                        latents_np[batch_index : batch_index + 1]
                    )
                ).to(device)
                image_list.append(
                    self.decode_latents(
                        batch_latents,
                        model_inputs.height,
                        model_inputs.width,
                        output_type=output_type,
                    )
                )

        return QwenImagePipelineOutput(images=image_list)
