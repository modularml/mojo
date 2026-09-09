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

"""QwenImage edit diffusion pipeline.

Key differences from QwenImagePipeline:
- Multimodal prompt encoding when edit images are present
- VAE image-conditioning path that concatenates condition latents to noise
- True CFG with two forward passes (positive + negative prompts)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.graph import TensorType, TensorValue, ops
from max.pipelines.context import PixelContext, TokenBuffer
from max.pipelines.diffusion.interface import DiffusionPipeline, max_compile
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.pipelines.lib.config.model_config import (
    _resolve_component_encoding_and_weights,
    _select_quantization_encoding,
)
from max.profiler import Tracer, traced

from ..autoencoders.autoencoder_kl_qwen_image import AutoencoderKLQwenImageModel
from ..qwen2_5vl.encoder import (
    Qwen25VLEncoderModel,
    Qwen25VLMultimodalEncoderModel,
)
from ..qwen_image.arch import QwenImageArchConfig
from .model import QwenImageEditTransformerModel


@dataclass(kw_only=True)
class QwenImageEditModelInputs:
    """QwenImage-edit-specific model inputs.

    For image editing the recommended usage is
    ``--guidance-scale 1.0 --true-cfg-scale 4.0``.
    ``guidance_scale`` is unused (model is not guidance-distilled);
    ``true_cfg_scale`` drives the two-pass CFG behavior.
    """

    tokens: TokenBuffer
    tokens_2: TokenBuffer | None = None
    negative_tokens: TokenBuffer | None = None
    negative_tokens_2: TokenBuffer | None = None
    timesteps: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    sigmas: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    latents: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    latent_image_ids: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    true_cfg_scale: float = 1.0
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 1.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1
    input_images: list[npt.NDArray[np.uint8]] | None = None
    prompt_images: list[npt.NDArray[np.uint8]] | None = None
    vae_condition_images: list[npt.NDArray[np.uint8]] | None = None

    @classmethod
    def from_context(cls, context: PixelContext) -> "QwenImageEditModelInputs":
        """Build QwenImageEditModelInputs from a PixelContext."""
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
            input_images=context.input_images,
            prompt_images=context.prompt_images,
            vae_condition_images=context.vae_condition_images,
        )


@dataclass
class QwenImageEditPipelineOutput:
    """Container for QwenImage edit pipeline results."""

    images: np.ndarray | list[np.ndarray | Buffer]


@dataclass
class QwenImageEditCache:
    """Runtime cache for reusable edit-path buffers."""

    sigmas: dict[str, Buffer] = field(default_factory=dict)
    text_ids: dict[str, Buffer] = field(default_factory=dict)
    shape_carriers: dict[int, Buffer] = field(default_factory=dict)
    cfg_scales: dict[float, Buffer] = field(default_factory=dict)
    noise_token_counts: dict[int, Buffer] = field(default_factory=dict)
    condition_image_ids: dict[tuple[int, int, int], Buffer] = field(
        default_factory=dict
    )
    latent_image_ids: dict[tuple[int, int, int], Buffer] = field(
        default_factory=dict
    )
    prompt_tokens: dict[tuple[int, ...], Buffer] = field(default_factory=dict)


class QwenImageEditPipeline(DiffusionPipeline):
    """Diffusion pipeline for QwenImage image editing.

    Wires together:
    - Qwen2.5-VL prompt encoder
    - QwenImage edit transformer denoiser
    - QwenImage 3D VAE (with latents_mean/std normalization)
    - Image-conditioning path (VAE encode -> normalize -> patchify -> concat)
    """

    vae: AutoencoderKLQwenImageModel
    text_encoder: Qwen25VLEncoderModel
    transformer: QwenImageEditTransformerModel

    components = {
        "vae": AutoencoderKLQwenImageModel,
        "text_encoder": Qwen25VLEncoderModel,
        "transformer": QwenImageEditTransformerModel,
    }

    # NOTE:
    # `prompt_encoder` is intentionally not part of `components`.
    #
    # QwenImageEdit needs a multimodal prompt path that reuses the already-loaded
    # `text_encoder` and layers a vision encoder + prompt/image merge logic on top.
    # That makes it closer to an edit-specific helper than an independent pipeline
    # submodel. Keeping it out of `components` avoids adding special loading rules
    # to the shared DiffusionPipeline base just for this dependency shape.
    prompt_encoder: Qwen25VLMultimodalEncoderModel | None = None
    _prompt_encoder_config: dict[str, Any] | None = None
    _prompt_encoder_weight_paths: list[Path] | None = None

    def init_remaining_components(self) -> None:
        """Initialize derived attributes that depend on loaded components."""
        self.vae_scale_factor = 8

        self._compile_runtime_helpers()
        self.cache: QwenImageEditCache = QwenImageEditCache()

        te_config = self.pipeline_config.models.get("text_encoder")
        if te_config is not None:
            self._prompt_encoder_config = te_config.huggingface_config.to_dict()
            _, te_weight_path = _resolve_component_encoding_and_weights(
                te_config
            )
            self._prompt_encoder_weight_paths = (
                self._get_component_weight_paths(te_config, te_weight_path)
            )

    def _compile_runtime_helpers(self) -> None:
        """Compile the helper graphs used by the edit pipeline runtime."""

        def duplicate_batch(value: TensorValue) -> TensorValue:
            return ops.concat([value, value], axis=0)

        def concat_sequence_pair(
            left: TensorValue,
            right: TensorValue,
        ) -> TensorValue:
            return ops.concat([left, right], axis=1)

        device = self.transformer.devices[0]
        self.cached_patchify_and_pack = max_compile(
            self._patchify_and_pack,
            input_types=[
                TensorType(
                    DType.float32,
                    shape=["batch", "channels", "height", 2, "width", 2],
                    device=device,
                )
            ],
        )

        self.cached_prepare_scheduler = max_compile(
            self.prepare_scheduler,
            input_types=[
                TensorType(
                    DType.float32,
                    shape=["num_sigmas"],
                    device=device,
                )
            ],
        )

        dtype = self.transformer.config.dtype
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
                TensorType(
                    DType.int64,
                    shape=["batch", "seq", 3],
                    device=device,
                ),
            ],
        )

        packed_channels = self.transformer.config.in_channels
        self.cached_postprocess_latents = max_compile(
            self._postprocess_latents,
            input_types=[
                TensorType(
                    dtype,
                    shape=["batch", "height", "width", packed_channels],
                    device=device,
                ),
                TensorType(dtype, shape=[16], device=device),
                TensorType(dtype, shape=[16], device=device),
            ],
        )

        self.cached_cfg_blend = max_compile(
            self._cfg_blend,
            input_types=[
                TensorType(
                    dtype, shape=["batch", "seq", "channels"], device=device
                ),
                TensorType(
                    dtype, shape=["batch", "seq", "channels"], device=device
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

        text_dtype = self.text_encoder.config.dtype
        text_device = self.text_encoder.devices[0]
        hidden_size = self.text_encoder.config.hidden_size
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

        vae_dtype = self.vae.config.dtype
        vae_device = self.vae.devices[0]
        self.cached_reshape_vae_latents = max_compile(
            self._reshape_vae_latents,
            input_types=[
                TensorType(
                    vae_dtype,
                    shape=["batch", "channels", "height", "width"],
                    device=vae_device,
                )
            ],
        )

        z_dim = self.vae.config.z_dim
        self.cached_normalize_and_pack_image_latent = max_compile(
            self._normalize_and_pack_image_latent,
            input_types=[
                TensorType(
                    vae_dtype,
                    shape=["batch", z_dim, "height", 2, "width", 2],
                    device=vae_device,
                ),
                TensorType(vae_dtype, shape=[z_dim], device=vae_device),
                TensorType(vae_dtype, shape=[z_dim], device=vae_device),
            ],
        )

        self.cached_concat_image_latents = max_compile(
            self.concat_image_latents,
            input_types=[
                TensorType(
                    dtype, shape=["batch", "seq", "channels"], device=device
                ),
                TensorType(
                    dtype, shape=["batch", "img_seq", "channels"], device=device
                ),
                TensorType(
                    DType.int64, shape=["batch", "seq", 3], device=device
                ),
                TensorType(
                    DType.int64, shape=["batch", "img_seq", 3], device=device
                ),
            ],
        )

        self.cached_concat_image_sequences = max_compile(
            concat_sequence_pair,
            input_types=[
                TensorType(
                    dtype, shape=["batch", "seq", "channels"], device=device
                ),
                TensorType(
                    dtype, shape=["batch", "img_seq", "channels"], device=device
                ),
            ],
        )

        self.cached_concat_image_ids = max_compile(
            concat_sequence_pair,
            input_types=[
                TensorType(
                    DType.int64, shape=["batch", "seq", 3], device=device
                ),
                TensorType(
                    DType.int64, shape=["batch", "img_seq", 3], device=device
                ),
            ],
        )

        self.cached_duplicate_condition_latents = max_compile(
            duplicate_batch,
            input_types=[
                TensorType(
                    dtype,
                    shape=[1, "img_seq", packed_channels],
                    device=device,
                )
            ],
        )

        self.cached_duplicate_condition_ids = max_compile(
            duplicate_batch,
            input_types=[
                TensorType(DType.int64, shape=[1, "img_seq", 3], device=device)
            ],
        )

        self.cached_extract_noise_latents = max_compile(
            self._extract_noise_latents,
            input_types=[
                TensorType(
                    dtype,
                    shape=["batch", "seq", packed_channels],
                    device=device,
                ),
                TensorType(DType.int64, shape=[1], device=CPU()),
            ],
        )

    def _init_prompt_encoder(self) -> None:
        if self.prompt_encoder is not None:
            return

        # NOTE:
        # This is a local assembly step, not a normal ComponentModel load.
        #
        # The edit prompt encoder depends on the already-instantiated
        # `self.text_encoder`, reuses the text-encoder weight set, and adds the
        # Qwen2.5-VL vision path needed for multimodal prompt encoding. If we
        # tried to model it as a regular pipeline component, the shared loader
        # would need special-case dependency wiring for "component B depends on
        # loaded component A", which is more confusing than keeping the assembly
        # here in the edit pipeline.
        from max.graph.weights import load_weights

        if self._prompt_encoder_config is None:
            raise ValueError("prompt encoder config is not initialized")
        if self._prompt_encoder_weight_paths is None:
            raise ValueError("prompt encoder weight paths are not initialized")

        from transformers import AutoTokenizer

        first_config = next(iter(self.pipeline_config.models.values()))
        tokenizer = AutoTokenizer.from_pretrained(
            first_config.model_path,
            subfolder="tokenizer",
        )

        encoding = _select_quantization_encoding(
            first_config, QwenImageArchConfig.DEFAULT_ENCODING
        )
        self.prompt_encoder = Qwen25VLMultimodalEncoderModel(
            text_encoder=self.text_encoder,
            config=self._prompt_encoder_config,
            encoding=encoding,
            devices=self.devices,
            weights=load_weights(self._prompt_encoder_weight_paths),
            session=self.session,
            tokenizer=tokenizer,
        )

    def _get_prompt_encoder(self) -> Qwen25VLMultimodalEncoderModel:
        # NOTE:
        # We only need the multimodal prompt path when edit images are present.
        # Text-only prompt encoding stays on `self.text_encoder`, so we avoid
        # paying the extra vision-side setup cost unless the request actually
        # uses image conditioning.
        if self.prompt_encoder is None:
            self._init_prompt_encoder()
            if self.prompt_encoder is None:
                raise ValueError("failed to initialize prompt_encoder")
        return self.prompt_encoder

    def _encode_prompt(
        self,
        *,
        tokens: TokenBuffer,
        prompt_images: list[npt.NDArray[np.uint8]],
        num_images_per_prompt: int,
        prompt_encoder: Qwen25VLMultimodalEncoderModel | None,
    ) -> Buffer:
        if prompt_images:
            assert prompt_encoder is not None
            return prompt_encoder.encode(
                tokens=tokens,
                images=prompt_images,
                num_images_per_prompt=num_images_per_prompt,
            )

        return self.prepare_prompt_embeddings(
            tokens=tokens,
            num_images_per_prompt=num_images_per_prompt,
        )

    @staticmethod
    def _resolve_condition_images(
        model_inputs: QwenImageEditModelInputs,
    ) -> tuple[list[npt.NDArray[np.uint8]], list[npt.NDArray[np.uint8]]]:
        prompt_images = (
            model_inputs.prompt_images or model_inputs.input_images or []
        )
        vae_condition_images = (
            model_inputs.vae_condition_images or model_inputs.input_images or []
        )
        return prompt_images, vae_condition_images

    def _prepare_negative_prompt_embeddings(
        self,
        *,
        model_inputs: QwenImageEditModelInputs,
        prompt_images: list[npt.NDArray[np.uint8]],
        prompt_encoder: Qwen25VLMultimodalEncoderModel | None,
    ) -> Buffer | None:
        if (
            model_inputs.true_cfg_scale <= 1.0
            or model_inputs.negative_tokens is None
        ):
            return None

        return self._encode_prompt(
            tokens=model_inputs.negative_tokens,
            prompt_images=prompt_images,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
            prompt_encoder=prompt_encoder,
        )

    def _prepare_condition_latents(
        self,
        *,
        vae_condition_images: list[npt.NDArray[np.uint8]],
        batch_size: int,
        device: Device,
    ) -> tuple[Buffer | None, Buffer | None]:
        if not vae_condition_images:
            return None, None

        image_bufs = [
            self._numpy_image_to_buffer(image) for image in vae_condition_images
        ]
        return self.prepare_image_latents(
            images=image_bufs,
            batch_size=batch_size,
            device=device,
        )

    def _prepare_text_ids_for_embeddings(
        self,
        *,
        embeddings: Buffer,
        batch_size: int,
        device: Device,
        max_vid_index: int,
    ) -> Buffer:
        seq_len = embeddings.shape[1]
        cache_key = f"{batch_size}_{seq_len}_{max_vid_index}"
        if cache_key not in self.cache.text_ids:
            self.cache.text_ids[cache_key] = self._prepare_text_ids(
                batch_size, seq_len, device, max_vid_index
            )
        return self.cache.text_ids[cache_key]

    def prepare_inputs(self, context: PixelContext) -> QwenImageEditModelInputs:
        """Convert a PixelContext into QwenImageEditModelInputs."""
        return QwenImageEditModelInputs.from_context(context)

    def _patchify_and_pack(self, latents: TensorValue) -> TensorValue:
        """(B,C,H//2,2,W//2,2) → (B, H//2*W//2, C*4)"""
        latents = ops.cast(latents, self.transformer.config.dtype)
        batch = latents.shape[0]
        c = latents.shape[1]
        h2 = latents.shape[2]
        w2 = latents.shape[4]
        latents = ops.permute(latents, [0, 1, 3, 5, 2, 4])
        latents = ops.reshape(latents, (batch, c * 4, h2, w2))
        latents = ops.reshape(latents, (batch, c * 4, h2 * w2))
        return ops.permute(latents, [0, 2, 1])

    def _postprocess_latents(
        self,
        latents_bhwc: TensorValue,
        latents_mean: TensorValue,
        latents_std: TensorValue,
    ) -> TensorValue:
        """Unpatchify (B,H,W,C*4) → (B,z_dim,H*2,W*2) and denormalize."""
        batch = latents_bhwc.shape[0]
        h = latents_bhwc.shape[1]
        w = latents_bhwc.shape[2]
        c = latents_bhwc.shape[3]
        z_dim = c // 4
        latents = ops.permute(latents_bhwc, [0, 3, 1, 2])
        latents = ops.reshape(latents, (batch, z_dim, 2, 2, h, w))
        latents = ops.permute(latents, [0, 1, 4, 2, 5, 3])
        latents = ops.reshape(latents, (batch, z_dim, h * 2, w * 2))
        mean_r = ops.reshape(latents_mean, (1, z_dim, 1, 1))
        std_r = ops.reshape(latents_std, (1, z_dim, 1, 1))
        return latents * std_r + mean_r

    def _normalize_and_pack_image_latent(
        self,
        image_latents: TensorValue,
        latents_mean: TensorValue,
        latents_std: TensorValue,
    ) -> TensorValue:
        """Normalize VAE output, then patchify+pack to (B, seq, C*4)."""
        batch = image_latents.shape[0]
        c = image_latents.shape[1]
        h2 = image_latents.shape[2]
        w2 = image_latents.shape[4]
        mean_r = ops.reshape(latents_mean, (1, c, 1, 1))
        std_r = ops.reshape(latents_std, (1, c, 1, 1))
        raw = ops.reshape(image_latents, (batch, c, h2 * 2, w2 * 2))
        raw = (raw - mean_r) / std_r
        packed = ops.reshape(raw, (batch, c, h2, 2, w2, 2))
        packed = ops.permute(packed, [0, 1, 3, 5, 2, 4])
        packed = ops.reshape(packed, (batch, c * 4, h2, w2))
        packed = ops.reshape(packed, (batch, c * 4, h2 * w2))
        return ops.permute(packed, [0, 2, 1])

    def _cfg_blend(
        self,
        cond_pred: TensorValue,
        uncond_pred: TensorValue,
        cfg_scale: TensorValue,
    ) -> TensorValue:
        scale = ops.cast(cfg_scale, cond_pred.dtype)
        return uncond_pred + scale * (cond_pred - uncond_pred)

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

    def _reshape_vae_latents(self, x: TensorValue) -> TensorValue:
        x = ops.rebind(
            x,
            [
                x.shape[0],
                x.shape[1],
                (x.shape[2] // 2) * 2,
                (x.shape[3] // 2) * 2,
            ],
        )
        return ops.reshape(
            x,
            (x.shape[0], x.shape[1], x.shape[2] // 2, 2, x.shape[3] // 2, 2),
        )

    def concat_image_latents(
        self,
        latents: TensorValue,
        image_latents: TensorValue,
        latent_image_ids: TensorValue,
        image_latent_ids: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        return (
            ops.concat([latents, image_latents], axis=1),
            ops.concat([latent_image_ids, image_latent_ids], axis=1),
        )

    def _extract_noise_latents(
        self, latents: TensorValue, num_noise_tokens: TensorValue
    ) -> TensorValue:
        return ops.slice_tensor(
            latents,
            [
                slice(None),
                (slice(0, num_noise_tokens), "noise_tokens"),
                slice(None),
            ],
        )

    def scheduler_step(
        self,
        latents: TensorValue,
        noise_pred: TensorValue,
        dt: TensorValue,
        img_ids: TensorValue,
    ) -> TensorValue:
        """Single Euler step that updates only the noise-token prefix."""
        lat_dtype = latents.dtype
        updated_latents = ops.cast(latents, DType.float32)
        noise_pred = ops.rebind(
            noise_pred,
            [latents.shape[0], latents.shape[1], latents.shape[2]],
        )
        updated_latents = updated_latents + dt * noise_pred
        updated_latents = ops.cast(updated_latents, lat_dtype)

        token_types = img_ids[:, :, 0]
        is_condition_token = ops.not_equal(
            token_types,
            ops.constant(0, DType.int64, device=token_types.device),
        )
        condition_token_mask = ops.broadcast_to(
            ops.unsqueeze(is_condition_token, -1),
            latents.shape,
        )
        return ops.where(condition_token_mask, latents, updated_latents)

    def prepare_scheduler(
        self, sigmas: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Precompute timesteps and dt values from sigmas."""
        sigmas_curr = ops.slice_tensor(sigmas, [slice(0, -1)])
        sigmas_next = ops.slice_tensor(sigmas, [slice(1, None)])
        return (
            ops.cast(sigmas_curr, self.transformer.config.dtype),
            sigmas_next - sigmas_curr,
        )

    # ── prompt encoding ───────────────────────────────────────────────────

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

    # ── position ID helpers ───────────────────────────────────────────────

    @staticmethod
    def _prepare_text_ids(
        batch_size: int, seq_len: int, device: Device, max_vid_index: int = 0
    ) -> Buffer:
        """Create 3D text position IDs in (T, H, W) format."""
        tok_positions = np.arange(seq_len, dtype=np.int64) + max_vid_index
        coords = np.stack(
            [tok_positions, tok_positions, tok_positions], axis=-1
        )
        text_ids = np.broadcast_to(
            coords[np.newaxis, :, :],
            (batch_size, coords.shape[0], coords.shape[1]),
        ).copy()
        return Buffer.from_dlpack(text_ids).to(device)

    @staticmethod
    def _prepare_image_ids(
        batch_size: int, height: int, width: int, device: Device
    ) -> Buffer:
        """Create 3D image position IDs in (T, H, W) format."""
        t_coords = np.zeros((height, width), dtype=np.int64)
        h_c = np.arange(height, dtype=np.int64) - (height - height // 2)
        w_c = np.arange(width, dtype=np.int64) - (width - width // 2)
        h_coords, w_coords = np.meshgrid(h_c, w_c, indexing="ij")
        coords = np.stack([t_coords, h_coords, w_coords], axis=-1).reshape(
            -1, 3
        )
        image_ids = np.broadcast_to(
            coords[np.newaxis, :, :],
            (batch_size, coords.shape[0], coords.shape[1]),
        ).copy()
        return Buffer.from_dlpack(image_ids).to(device)

    @staticmethod
    def _prepare_condition_image_ids(
        batch_size: int,
        height: int,
        width: int,
        device: Device,
        image_index: int = 0,
    ) -> Buffer:
        """Condition-image IDs with T=image_index+1 (noise tokens use T=0).

        For multi-image editing each condition image needs a distinct T
        coordinate so the transformer can distinguish them via RoPE:
        noise → T=0, first image → T=1, second image → T=2, etc.
        """
        t_coords = np.full((height, width), image_index + 1, dtype=np.int64)
        h_c = np.arange(height, dtype=np.int64) - (height - height // 2)
        w_c = np.arange(width, dtype=np.int64) - (width - width // 2)
        h_coords, w_coords = np.meshgrid(h_c, w_c, indexing="ij")
        coords = np.stack([t_coords, h_coords, w_coords], axis=-1).reshape(
            -1, 3
        )
        condition_image_ids = np.broadcast_to(
            coords[np.newaxis, :, :],
            (batch_size, coords.shape[0], coords.shape[1]),
        ).copy()
        return Buffer.from_dlpack(condition_image_ids).to(device)

    def _get_condition_image_ids(
        self, height: int, width: int, device: Device, image_index: int = 0
    ) -> Buffer:
        cache_key = (height, width, image_index)
        if cache_key not in self.cache.condition_image_ids:
            self.cache.condition_image_ids[cache_key] = (
                self._prepare_condition_image_ids(
                    1,
                    height,
                    width,
                    device,
                    image_index=image_index,
                )
            )
        return self.cache.condition_image_ids[cache_key]

    # ── latent preprocessing ──────────────────────────────────────────────

    def preprocess_latents(
        self,
        latents: npt.NDArray[np.float32],
        latent_image_ids: npt.NDArray[np.float32],
    ) -> tuple[Buffer, Buffer]:
        latents_np = np.asarray(latents)
        b, c, h, w = latents_np.shape
        latents_6d = latents_np.reshape(b, c, h // 2, 2, w // 2, 2)
        device = self.transformer.devices[0]
        latents_packed = self.cached_patchify_and_pack(
            Buffer.from_dlpack(np.ascontiguousarray(latents_6d)).to(device)
        )
        ids_key = (b, h, w)
        if ids_key not in self.cache.latent_image_ids:
            self.cache.latent_image_ids[ids_key] = Buffer.from_dlpack(
                np.asarray(latent_image_ids, dtype=np.int64)
            ).to(device)
        ids_buf = self.cache.latent_image_ids[ids_key]
        return latents_packed, ids_buf

    # ── image conditioning ────────────────────────────────────────────────

    def _numpy_image_to_buffer(self, image: npt.NDArray[np.uint8]) -> Buffer:
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        img_array = (image.astype(np.float32) / 127.5) - 1.0
        img_array = np.ascontiguousarray(
            np.expand_dims(np.transpose(img_array, (2, 0, 1)), 0)
        )
        vae_dtype = self.vae.config.dtype
        device = self.vae.devices[0]
        if vae_dtype == DType.bfloat16:
            u16 = float32_to_bfloat16_as_uint16(img_array)
            buf = Buffer.from_numpy(u16).to(device)
            return buf.view(dtype=DType.bfloat16, shape=img_array.shape)
        if vae_dtype == DType.float16:
            img_array = img_array.astype(np.float16)
        return Buffer.from_dlpack(img_array).to(device)

    def _encode_single_image(
        self,
        image: Buffer,
        device: Device,
        image_index: int = 0,
    ) -> tuple[Buffer, Buffer]:
        latents_mean = self.vae.latents_mean_tensor
        latents_std = self.vae.latents_std_tensor
        if latents_mean is None or latents_std is None:
            raise ValueError("VAE latents_mean/latents_std are required.")

        raw_latents = self.vae.encode(image.to(device))
        _, _, raw_h, raw_w = raw_latents.shape

        latents_6d = self.cached_reshape_vae_latents(raw_latents)
        image_latents = self.cached_normalize_and_pack_image_latent(
            latents_6d, latents_mean, latents_std
        )
        image_ids = self._get_condition_image_ids(
            raw_h // 2,
            raw_w // 2,
            device,
            image_index=image_index,
        )
        return image_latents, image_ids

    def prepare_image_latents(
        self, images: list[Buffer], batch_size: int, device: Device
    ) -> tuple[Buffer, Buffer]:
        all_latents: list[Buffer] = []
        all_ids: list[Buffer] = []
        for idx, img in enumerate(images):
            lat, ids = self._encode_single_image(img, device, image_index=idx)
            all_latents.append(lat)
            all_ids.append(ids)

        if len(all_latents) == 1:
            image_latents, image_ids = all_latents[0], all_ids[0]
        else:
            image_latents = all_latents[0]
            for image_latent in all_latents[1:]:
                image_latents = self.cached_concat_image_sequences(
                    image_latents,
                    image_latent,
                )
            image_ids = all_ids[0]
            for ids in all_ids[1:]:
                image_ids = self.cached_concat_image_ids(image_ids, ids)

        if batch_size > 1:
            if batch_size == 2:
                return (
                    self.cached_duplicate_condition_latents(image_latents),
                    self.cached_duplicate_condition_ids(image_ids),
                )
            lat_np = np.from_dlpack(image_latents.to(CPU()))
            image_latents = Buffer.from_dlpack(
                np.broadcast_to(
                    lat_np,
                    (batch_size, lat_np.shape[1], lat_np.shape[2]),
                ).copy()
            ).to(device)
            ids_np = np.from_dlpack(image_ids.to(CPU()))
            image_ids = Buffer.from_dlpack(
                np.broadcast_to(
                    ids_np,
                    (batch_size, ids_np.shape[1], ids_np.shape[2]),
                ).copy()
            ).to(device)

        return image_latents, image_ids

    # ── decode ────────────────────────────────────────────────────────────

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

    def _to_numpy(self, image: Any) -> np.ndarray:
        cpu_image = image.to(CPU())
        try:
            return np.from_dlpack(cpu_image).astype(np.float32)
        except (RuntimeError, TypeError):
            from max.experimental.tensor import Tensor as _Tensor

            if isinstance(cpu_image, _Tensor):
                return np.from_dlpack(cpu_image.cast(DType.float32)).astype(
                    np.float32
                )
            return np.from_dlpack(
                _Tensor(storage=cpu_image).cast(DType.float32)
            ).astype(np.float32)

    @staticmethod
    def _image_to_flat_hwc(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        while img.ndim > 3:
            img = img.squeeze(0)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img.astype(np.float32, copy=False)

    # ── main execute ──────────────────────────────────────────────────────

    @traced
    def execute(  # type: ignore[override]
        self,
        model_inputs: QwenImageEditModelInputs,
        output_type: Literal["np", "latent"] = "np",
    ) -> QwenImageEditPipelineOutput:
        """Run the QwenImageEdit denoising loop and decode outputs."""
        device = self.transformer.devices[0]

        # Phase 1: prompt, latent, and conditioning preparation.
        prompt_images, vae_condition_images = self._resolve_condition_images(
            model_inputs
        )
        has_images = bool(prompt_images)
        prompt_encoder = self._get_prompt_encoder() if has_images else None

        prompt_embeds = self._encode_prompt(
            tokens=model_inputs.tokens,
            prompt_images=prompt_images,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
            prompt_encoder=prompt_encoder,
        )
        batch_size = int(prompt_embeds.shape[0])

        do_true_cfg = model_inputs.true_cfg_scale > 1.0
        negative_prompt_embeds = self._prepare_negative_prompt_embeddings(
            model_inputs=model_inputs,
            prompt_images=prompt_images,
            prompt_encoder=prompt_encoder,
        )

        latents, latent_image_ids = self.preprocess_latents(
            model_inputs.latents, model_inputs.latent_image_ids
        )
        noise_token_count_value = int(latents.shape[1])
        if noise_token_count_value not in self.cache.noise_token_counts:
            self.cache.noise_token_counts[noise_token_count_value] = (
                Buffer.from_numpy(
                    np.array([noise_token_count_value], dtype=np.int64)
                )
            )
        noise_token_count = self.cache.noise_token_counts[
            noise_token_count_value
        ]

        image_latents, image_latent_ids = self._prepare_condition_latents(
            vae_condition_images=vae_condition_images,
            batch_size=batch_size,
            device=device,
        )

        h_latent = model_inputs.height // (self.vae_scale_factor * 2)
        w_latent = model_inputs.width // (self.vae_scale_factor * 2)
        max_vid_index = max(h_latent // 2, w_latent // 2)

        text_ids = self._prepare_text_ids_for_embeddings(
            embeddings=prompt_embeds,
            batch_size=batch_size,
            device=device,
            max_vid_index=max_vid_index,
        )

        # Phase 2: classifier-free guidance setup.
        negative_text_ids: Buffer | None = None
        if do_true_cfg and negative_prompt_embeds is not None:
            negative_text_ids = self._prepare_text_ids_for_embeddings(
                embeddings=negative_prompt_embeds,
                batch_size=batch_size,
                device=device,
                max_vid_index=max_vid_index,
            )

        # Phase 3: scheduler and loop-invariant inputs.
        num_inference_steps = model_inputs.num_inference_steps
        sigmas_key = f"{num_inference_steps}_{latents.shape[1]}"
        if sigmas_key not in self.cache.sigmas:
            self.cache.sigmas[sigmas_key] = Buffer.from_dlpack(
                model_inputs.sigmas
            ).to(device)
        with Tracer("prepare_scheduler"):
            all_timesteps, all_dts = self.cached_prepare_scheduler(
                self.cache.sigmas[sigmas_key]
            )
            timesteps_seq = all_timesteps.driver_tensor
            dts_seq = all_dts.driver_tensor

        cfg_scale_buf: Buffer | None = None
        if do_true_cfg:
            cfg_scale = float(model_inputs.true_cfg_scale)
            if cfg_scale not in self.cache.cfg_scales:
                self.cache.cfg_scales[cfg_scale] = Buffer.from_dlpack(
                    np.array([cfg_scale], dtype=np.float32)
                ).to(device)
            cfg_scale_buf = self.cache.cfg_scales[cfg_scale]

        latents_in = latents
        ids_in = latent_image_ids
        if image_latents is not None and image_latent_ids is not None:
            latents_in, ids_in = self.cached_concat_image_latents(
                latents,
                image_latents,
                latent_image_ids,
                image_latent_ids,
            )

        # Phase 4: denoising loop.
        with Tracer("denoising_loop"):
            for i in range(num_inference_steps):
                with Tracer(f"denoising_step_{i}"):
                    timestep = timesteps_seq[i : i + 1]
                    dt = dts_seq[i : i + 1]

                    with Tracer("transformer_pos"):
                        noise_pred = self.transformer(
                            latents_in,
                            prompt_embeds,
                            timestep,
                            ids_in,
                            text_ids,
                        )[0]

                    if (
                        do_true_cfg
                        and negative_prompt_embeds is not None
                        and negative_text_ids is not None
                        and cfg_scale_buf is not None
                    ):
                        with Tracer("transformer_neg"):
                            noise_pred_uncond = self.transformer(
                                latents_in,
                                negative_prompt_embeds,
                                timestep,
                                ids_in,
                                negative_text_ids,
                            )[0]
                        with Tracer("cfg_blend"):
                            noise_pred = self.cached_cfg_blend(
                                noise_pred, noise_pred_uncond, cfg_scale_buf
                            )

                    with Tracer("scheduler_step"):
                        latents_in = self.cached_scheduler_step(
                            latents_in,
                            noise_pred,
                            dt,
                            ids_in,
                        )

            latents = self.cached_extract_noise_latents(
                latents_in, noise_token_count
            )

        # Phase 5: decode outputs.
        image_list = []
        if batch_size == 1:
            image_list.append(
                self.decode_latents(
                    latents,
                    model_inputs.height,
                    model_inputs.width,
                    output_type,
                )
            )
        else:
            lat_np = self._to_numpy(latents)
            for b in range(batch_size):
                latents_b = Buffer.from_dlpack(
                    np.ascontiguousarray(lat_np[b : b + 1])
                ).to(device)
                image_list.append(
                    self.decode_latents(
                        latents_b,
                        model_inputs.height,
                        model_inputs.width,
                        output_type,
                    )
                )

        return QwenImageEditPipelineOutput(images=image_list)
