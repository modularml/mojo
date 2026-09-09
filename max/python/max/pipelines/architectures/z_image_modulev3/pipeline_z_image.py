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
"""Z-Image diffusion pipeline (ModuleV3).

Wires together the Qwen3 text encoder, Z-Image transformer denoiser, and
standard AutoencoderKL VAE. Follows the shared diffusion-pipeline structure
used by the image generation architectures in this directory, including
tracing, module docstrings, and flat weight path assignment.
"""

from __future__ import annotations

import hashlib
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.pipelines.context import PixelContext, TokenBuffer
from max.pipelines.diffusion.cache import DenoisingCacheState
from max.pipelines.diffusion.interface import DiffusionPipeline, max_compile
from max.profiler import Tracer, traced

from ..autoencoders import AutoencoderKLModel
from ..qwen3_modulev3.text_encoder import Qwen3TextEncoderZImageModel
from .model import ZImageTransformerModel

_DEVICE_TENSOR_FIELDS = frozenset(
    {
        "tokens_tensor",
        "negative_tokens_tensor",
        "txt_ids_tensor",
        "img_ids_tensor",
        "negative_txt_ids_tensor",
        "negative_img_ids_tensor",
        "input_image_tensor",
        "latents_tensor",
        "sigmas_tensor",
        "h_carrier",
        "w_carrier",
    }
)


def _validate_z_image_context(context: PixelContext) -> None:
    """Fail fast before device uploads."""
    if context.latents.size == 0:
        raise ValueError(
            "ZImagePipeline requires non-empty latents in PixelGenerationContext."
        )
    for name in ("latent_image_ids", "sigmas", "timesteps"):
        if not hasattr(context, name):
            raise TypeError(
                f"ZImagePipeline requires PixelGenerationContext with attribute "
                f"{name!r} (e.g. max.pipelines.context.PixelContext); "
                f"{type(context).__name__} has no {name!r}."
            )
        arr = getattr(context, name)
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            raise ValueError(
                f"ZImagePipeline requires non-empty {name} in PixelGenerationContext."
            )


@dataclass(kw_only=True)
class ZImageModelInputs:
    """Z-Image execution inputs with device tensors and host metadata.

    Scalar and NumPy fields are populated from :class:`PixelGenerationContext`
    (concrete type :class:`~max.pipelines.context.PixelContext`) via
    :meth:`kwargs_from_context`; ``latents_tensor``, ``sigmas_tensor``, and
    shape carriers are required device tensors supplied by
    :meth:`ZImagePipeline.prepare_inputs` in one shot.
    """

    # -- inherited from the former PixelModelInputs base class --
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

    guidance: npt.NDArray[np.float32] | None = None
    """Optional precomputed guidance tensor."""

    true_cfg_scale: float = 1.0
    """True CFG scale (enabled when > 1.0 with negative prompt)."""

    num_warmup_steps: int = 0
    """Number of scheduler warmup steps."""

    input_image: npt.NDArray[np.uint8] | None = None
    """Optional HWC uint8 numpy array for image-to-image generation."""

    strength: float = 0.6
    """Image-to-image denoising strength in (0, 1]."""

    cfg_normalization: bool = False
    """Whether to apply Z-Image CFG renormalization."""

    cfg_truncation: float = 1.0
    """Z-Image CFG truncation threshold in normalized time."""

    # -- Z-Image-specific fields --
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1
    explicit_negative_prompt: bool = False
    do_cfg: bool = False
    tokens_tensor: Tensor
    negative_tokens_tensor: Tensor | None = None
    txt_ids_tensor: Tensor
    img_ids_tensor: Tensor
    negative_txt_ids_tensor: Tensor | None = None
    negative_img_ids_tensor: Tensor | None = None
    input_image_tensor: Tensor | None = None
    latents_tensor: Tensor
    sigmas_tensor: Tensor
    h_carrier: Tensor
    w_carrier: Tensor

    @classmethod
    def kwargs_from_context(cls, context: PixelContext) -> dict[str, Any]:
        """Build kwargs for all fields except device tensors."""
        kwargs: dict[str, Any] = {}
        for dataclass_field in fields(cls):
            name = dataclass_field.name
            if name in _DEVICE_TENSOR_FIELDS:
                continue
            if not hasattr(context, name):
                continue
            v = getattr(context, name)
            if v is None:
                if dataclass_field.default is not MISSING:
                    kwargs[name] = dataclass_field.default
                elif dataclass_field.default_factory is not MISSING:
                    kwargs[name] = dataclass_field.default_factory()
                else:
                    kwargs[name] = None
            else:
                kwargs[name] = v
        return kwargs

    def __post_init__(self) -> None:
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError(
                f"height must be a positive int. Got {self.height!r}"
            )
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError(
                f"width must be a positive int. Got {self.width!r}"
            )
        if (
            not isinstance(self.num_inference_steps, int)
            or self.num_inference_steps <= 0
        ):
            raise ValueError(
                f"num_inference_steps must be a positive int. Got {self.num_inference_steps!r}"
            )
        if self.sigmas.size == 0:
            raise ValueError(
                "ZImagePipeline requires non-empty sigmas in context."
            )
        if self.latent_image_ids.size == 0:
            raise ValueError(
                "ZImagePipeline requires non-empty latent image ids in context."
            )


@dataclass
class ZImagePipelineOutput:
    """Container for Z-Image pipeline results."""

    images: np.ndarray | Tensor
    """Decoded image data as a NumPy array or MAX Tensor.

    The format depends on decode path and ``output_type``.
    """


class ZImagePipeline(DiffusionPipeline):
    """Diffusion pipeline for Z-Image generation (Qwen3 + transformer + VAE)."""

    unprefixed_weight_component = "transformer"
    default_num_inference_steps = 50
    default_residual_threshold = 0.06

    vae: AutoencoderKLModel
    text_encoder: Qwen3TextEncoderZImageModel
    transformer: ZImageTransformerModel

    components = {
        "vae": AutoencoderKLModel,
        "text_encoder": Qwen3TextEncoderZImageModel,
        "transformer": ZImageTransformerModel,
    }

    @traced(message="ZImagePipeline.init_remaining_components")
    def init_remaining_components(self) -> None:
        """Initialize derived attributes and compiled subgraphs."""
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )

        self.build_preprocess_latents()
        self.build_prepare_scheduler()
        self.build_scheduler_step()
        self.build_decode_latents()

        self._init_cache_state(
            dtype=self.transformer.config.dtype,
            device=self.transformer.devices[0],
        )

        self._cached_text_ids: dict[str, Tensor] = {}
        self._cached_sigmas: dict[str, Tensor] = {}
        self._cached_img_ids: dict[str, Tensor] = {}
        self._cached_img_ids_base_np: dict[str, np.ndarray] = {}
        self._cached_shape_carriers: dict[int, Tensor] = {}
        self._cached_timesteps_batched: dict[str, Tensor] = {}
        self._cached_timesteps_host: dict[str, np.ndarray] = {}
        self._cached_prompt_token_tensors: dict[str, Tensor] = {}
        self._cached_prompt_padding: dict[str, Tensor] = {}

    @traced(message="ZImagePipeline.prepare_inputs")
    def prepare_inputs(
        self,
        context: PixelContext,
    ) -> ZImageModelInputs:
        """Convert a :class:`PixelGenerationContext` into model inputs with device tensors."""
        _validate_z_image_context(context)
        kwargs = ZImageModelInputs.kwargs_from_context(context)
        device = self.transformer.devices[0]
        text_device = self.text_encoder.devices[0]

        # Keep host NumPy fields aligned with the tensors we upload.
        kwargs["latents"] = np.asarray(context.latents)
        kwargs["sigmas"] = np.asarray(context.sigmas)
        kwargs["latent_image_ids"] = np.asarray(context.latent_image_ids)

        latents_np = np.ascontiguousarray(kwargs["latents"])
        latent_h = int(latents_np.shape[-2])
        latent_w = int(latents_np.shape[-1])
        packed_h = int(latent_h // 2)
        packed_w = int(latent_w // 2)
        image_seq_len = int(np.asarray(context.latent_image_ids).shape[-2])

        tokens_np = self._select_tokens_for_text_encoder(
            context.tokens.array, context.mask
        )
        tokens_tensor = self._token_tensor_from_numpy(tokens_np, text_device)
        txt_ids_tensor, img_ids_tensor = self._prepare_conditioning_ids(
            text_seq_len=int(tokens_np.shape[0]),
            image_seq_len=image_seq_len,
            latent_image_ids=np.asarray(context.latent_image_ids),
            height=int(context.height),
            width=int(context.width),
            device=device,
        )

        negative_tokens_tensor: Tensor | None = None
        negative_txt_ids_tensor: Tensor | None = None
        negative_img_ids_tensor: Tensor | None = None
        if context.negative_tokens is not None:
            negative_tokens_np = self._select_tokens_for_text_encoder(
                context.negative_tokens.array,
                context.negative_mask,
            )
            negative_tokens_tensor = self._token_tensor_from_numpy(
                negative_tokens_np, text_device
            )
            if context.explicit_negative_prompt:
                (
                    negative_txt_ids_tensor,
                    negative_img_ids_tensor,
                ) = self._prepare_conditioning_ids(
                    text_seq_len=int(negative_tokens_np.shape[0]),
                    image_seq_len=image_seq_len,
                    latent_image_ids=np.asarray(context.latent_image_ids),
                    height=int(context.height),
                    width=int(context.width),
                    device=device,
                )
        do_cfg = (
            float(context.guidance_scale) > 0.0
            and negative_tokens_tensor is not None
        )

        input_image_tensor: Tensor | None = None
        if context.input_image is not None:
            input_image_tensor = self._image_to_tensor(
                image=np.ascontiguousarray(
                    context.input_image.astype(np.uint8, copy=False)
                ),
                batch_size=int(context.num_images_per_prompt),
                dtype=self.vae.config.dtype,
            )

        latents_tensor = Tensor(
            storage=Buffer.from_dlpack(latents_np).to(device)
        )

        for n in (packed_h, packed_w):
            if n not in self._cached_shape_carriers:
                carrier = np.ascontiguousarray(np.empty(n, dtype=np.float32))
                self._cached_shape_carriers[n] = Tensor(
                    storage=Buffer.from_dlpack(carrier).to(device)
                )
        h_carrier = self._cached_shape_carriers[packed_h]
        w_carrier = self._cached_shape_carriers[packed_w]

        num_steps = int(context.num_inference_steps)
        sigmas_key = f"sigmas::{num_steps}::{latent_h}x{latent_w}"
        if sigmas_key in self._cached_sigmas:
            sigmas_tensor = self._cached_sigmas[sigmas_key]
        else:
            sigmas_tensor = Tensor(
                storage=Buffer.from_dlpack(
                    np.ascontiguousarray(context.sigmas)
                ).to(device)
            )
            self._cached_sigmas[sigmas_key] = sigmas_tensor

        return ZImageModelInputs(
            **kwargs,
            do_cfg=do_cfg,
            tokens_tensor=tokens_tensor,
            negative_tokens_tensor=negative_tokens_tensor,
            txt_ids_tensor=txt_ids_tensor,
            img_ids_tensor=img_ids_tensor,
            negative_txt_ids_tensor=negative_txt_ids_tensor,
            negative_img_ids_tensor=negative_img_ids_tensor,
            input_image_tensor=input_image_tensor,
            latents_tensor=latents_tensor,
            sigmas_tensor=sigmas_tensor,
            h_carrier=h_carrier,
            w_carrier=w_carrier,
        )

    def create_cache_state(
        self,
        batch_size: int,
        seq_len: int,
        transformer_config: Any,
        text_seq_len: int = 0,
    ) -> DenoisingCacheState:
        """Allocate FBCache / Taylor tensors using Z-Image output layout."""
        cfg = transformer_config
        residual_dim, output_dim = cfg.fbcache_dims()
        state = DenoisingCacheState()

        if self.cache_config.first_block_caching:
            assert self._fbc is not None
            fbc_state = self._fbc.create_state(
                batch_size, seq_len, residual_dim, output_dim
            )
            state.prev_residual = fbc_state.prev_residual
            state.prev_output = fbc_state.prev_output

        if self.cache_config.taylorseer:
            assert self._taylorseer is not None
            ts_state = self._taylorseer.create_state(
                batch_size, seq_len, output_dim
            )
            state.taylor_factor_0 = ts_state.factor_0
            state.taylor_factor_1 = ts_state.factor_1
            state.taylor_factor_2 = ts_state.factor_2

        return state

    def run_transformer(
        self,
        cache_state: DenoisingCacheState,
        **kwargs: Any,
    ) -> tuple[Tensor, ...]:
        return self.transformer(
            kwargs["latents"],
            kwargs["prompt_embeds"],
            kwargs["timestep"],
            kwargs["img_ids"],
            kwargs["txt_ids"],
            prev_residual=cache_state.prev_residual,
            prev_output=cache_state.prev_output,
        )

    def build_preprocess_latents(self) -> None:
        device = self.transformer.devices[0]
        self.__dict__["_pack_latents_from_6d"] = max_compile(
            self._pack_latents_from_6d,
            input_types=[
                TensorType(
                    DType.float32,
                    shape=[
                        "batch",
                        "channels",
                        "height",
                        2,
                        "width",
                        2,
                    ],
                    device=device,
                ),
            ],
        )

    def build_prepare_scheduler(self) -> None:
        self.__dict__["prepare_scheduler"] = max_compile(
            self.prepare_scheduler,
            input_types=[
                TensorType(
                    DType.float32,
                    shape=["num_sigmas"],
                    device=self.transformer.devices[0],
                ),
            ],
        )

    def build_scheduler_step(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        self.__dict__["scheduler_step"] = max_compile(
            self.scheduler_step,
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

    def build_decode_latents(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        self.__dict__["_postprocess_latents"] = max_compile(
            self._postprocess_latents,
            input_types=[
                TensorType(
                    dtype,
                    shape=[
                        "batch",
                        "half_h",
                        "half_w",
                        2,
                        2,
                        "ch_4",
                    ],
                    device=device,
                ),
            ],
        )

    @staticmethod
    def _pack_latents(latents: Tensor) -> Tensor:
        batch_size, num_channels, height, width = map(int, latents.shape)
        latents = F.reshape(
            latents,
            (
                batch_size,
                num_channels,
                height // 2,
                2,
                width // 2,
                2,
            ),
        )
        latents = F.permute(latents, (0, 2, 4, 3, 5, 1))
        latents = F.reshape(
            latents,
            (
                batch_size,
                (height // 2) * (width // 2),
                num_channels * 4,
            ),
        )
        return latents

    @staticmethod
    def _pack_latents_from_6d(latents: Tensor) -> Tensor:
        batch_size = latents.shape[0]
        num_channels = latents.shape[1]
        height = latents.shape[2]
        width = latents.shape[4]
        latents = F.permute(latents, (0, 2, 4, 3, 5, 1))
        latents = F.reshape(
            latents,
            (
                batch_size,
                height * width,
                num_channels * 4,
            ),
        )
        return latents

    @staticmethod
    def _unpack_latents(
        latents: Tensor,
        height: int,
        width: int,
        vae_scale_factor: int,
    ) -> Tensor:
        batch_size = int(latents.shape[0])
        ch_size = int(latents.shape[2])

        height = 2 * (height // (vae_scale_factor * 2))
        width = 2 * (width // (vae_scale_factor * 2))

        h2 = height // 2
        w2 = width // 2
        latents = F.reshape(
            latents,
            (batch_size, h2, w2, 2, 2, ch_size // 4),
        )
        latents = F.permute(latents, (0, 5, 1, 3, 2, 4))
        latents = F.reshape(
            latents,
            (batch_size, ch_size // 4, height, width),
        )
        return latents

    @traced(message="ZImagePipeline.prepare_prompt_embeddings")
    def prepare_prompt_embeddings(
        self,
        tokens: Tensor,
        num_images_per_prompt: int,
    ) -> Tensor:
        """Encode prompt tokens into text embeddings."""
        prompt_embeds = self.text_encoder(tokens)
        if prompt_embeds.rank == 2:
            prompt_embeds = F.unsqueeze(prompt_embeds, axis=0)
        elif prompt_embeds.rank != 3:
            raise ValueError(
                f"Unexpected prompt_embeds rank={prompt_embeds.rank}; expected 2 or 3."
            )
        if num_images_per_prompt > 1:
            batch = int(prompt_embeds.shape[0])
            seq = int(prompt_embeds.shape[1])
            hidden = int(prompt_embeds.shape[2])
            if batch == 1:
                prompt_embeds = F.broadcast_to(
                    prompt_embeds,
                    [num_images_per_prompt, seq, hidden],
                )
            else:
                prompt_embeds = F.concat(
                    [prompt_embeds] * num_images_per_prompt,
                    axis=0,
                )

        return prompt_embeds

    @staticmethod
    def _select_tokens_for_text_encoder(
        tokens: np.ndarray,
        mask: np.ndarray | None,
    ) -> np.ndarray:
        if tokens.ndim == 2:
            tokens = tokens[0]

        selected_tokens = tokens
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[0]
            if mask.shape[0] != tokens.shape[0]:
                raise ValueError(
                    f"ZImage mask length mismatch: mask={mask.shape[0]}, "
                    f"tokens={tokens.shape[0]}."
                )
            selected_mask = mask.astype(np.bool_, copy=False)
            if not np.any(selected_mask):
                raise ValueError("ZImage mask cannot exclude all tokens.")
            if not np.all(selected_mask):
                selected_tokens = tokens[selected_mask]

        return np.ascontiguousarray(
            selected_tokens.astype(np.int64, copy=False)
        )

    def _token_tensor_from_numpy(
        self,
        tokens: np.ndarray,
        device: Device,
    ) -> Tensor:
        token_digest = hashlib.sha1(tokens.tobytes()).hexdigest()
        token_key = (
            f"prompt_tokens::{tokens.shape[0]}::{token_digest}::{device}"
        )
        if token_key in self._cached_prompt_token_tensors:
            return self._cached_prompt_token_tensors[token_key]

        text_input_ids = Tensor(storage=Buffer.from_dlpack(tokens).to(device))
        self._cached_prompt_token_tensors[token_key] = text_input_ids
        return text_input_ids

    def _prepare_conditioning_ids(
        self,
        text_seq_len: int,
        image_seq_len: int,
        latent_image_ids: np.ndarray,
        height: int,
        width: int,
        device: Device,
    ) -> tuple[Tensor, Tensor]:
        text_seq_len_padded = text_seq_len + (-text_seq_len % 32)
        img_ids_base_key = f"img_ids_base::{image_seq_len}_{height}x{width}"
        if img_ids_base_key in self._cached_img_ids_base_np:
            img_ids_base_np = self._cached_img_ids_base_np[img_ids_base_key]
        else:
            img_ids_base_np = np.asarray(latent_image_ids, dtype=np.int64)
            if img_ids_base_np.ndim == 3:
                img_ids_base_np = img_ids_base_np[0]
            img_ids_base_np = np.ascontiguousarray(img_ids_base_np)
            self._cached_img_ids_base_np[img_ids_base_key] = img_ids_base_np

        img_ids_key = (
            f"img_ids::{text_seq_len_padded}_{image_seq_len}_{height}x{width}"
        )
        if img_ids_key in self._cached_img_ids:
            img_ids_tensor = self._cached_img_ids[img_ids_key]
        else:
            img_ids_np = img_ids_base_np.copy()
            img_ids_np[:, 0] = img_ids_np[:, 0] + text_seq_len_padded + 1
            img_ids_tensor = Tensor(
                storage=Buffer.from_dlpack(np.ascontiguousarray(img_ids_np)).to(
                    device
                )
            )
            self._cached_img_ids[img_ids_key] = img_ids_tensor

        text_ids_key = f"text_ids::{text_seq_len}"
        if text_ids_key in self._cached_text_ids:
            txt_ids_tensor = self._cached_text_ids[text_ids_key]
        else:
            txt_ids_tensor = self._prepare_text_ids(text_seq_len, device)
            self._cached_text_ids[text_ids_key] = txt_ids_tensor

        return txt_ids_tensor, img_ids_tensor

    def _align_prompt_seq_len(
        self,
        embeds: Tensor,
        target_seq_len: int,
    ) -> Tensor:
        cur_len = int(embeds.shape[1])
        if cur_len == target_seq_len:
            return embeds
        if cur_len > target_seq_len:
            return embeds[:, :target_seq_len, :]

        pad_len = target_seq_len - cur_len
        pad_key = (
            f"prompt_pad::{int(embeds.shape[0])}::{pad_len}::"
            f"{int(embeds.shape[2])}::{embeds.dtype}::{embeds.device}"
        )
        if pad_key in self._cached_prompt_padding:
            pad = self._cached_prompt_padding[pad_key]
        else:
            pad = Tensor(
                storage=Buffer.zeros(
                    (int(embeds.shape[0]), pad_len, int(embeds.shape[2])),
                    embeds.dtype,
                    device=embeds.device,
                )
            )
            self._cached_prompt_padding[pad_key] = pad
        return F.concat([embeds, pad], axis=1)

    @staticmethod
    def _prepare_text_ids(
        seq_len: int,
        device: Device,
    ) -> Tensor:
        """Build text position IDs in [1, 2, ..., seq_len] format."""
        text_ids = np.zeros((seq_len, 3), dtype=np.int64)
        text_ids[:, 0] = np.arange(1, seq_len + 1, dtype=np.int64)
        return Tensor(
            storage=Buffer.from_dlpack(np.ascontiguousarray(text_ids)).to(
                device
            )
        )

    @traced(message="ZImagePipeline.decode_latents")
    def decode_latents(
        self,
        latents: Tensor,
        h_carrier: Tensor,
        w_carrier: Tensor,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Tensor | np.ndarray:
        """Decode packed latents into image output."""
        latent_h = int(h_carrier.shape[0]) * 2
        latent_w = int(w_carrier.shape[0]) * 2
        if output_type == "latent":
            return latents

        batch_size = int(latents.shape[0])
        ch_size = int(latents.shape[2])
        latents = F.reshape(
            latents,
            (
                batch_size,
                latent_h // 2,
                latent_w // 2,
                2,
                2,
                ch_size // 4,
            ),
        )

        latents = self._postprocess_latents(latents)
        decoded: Tensor = self.vae.decode(latents)
        return self._to_numpy(decoded)

    def _postprocess_latents(self, latents: Tensor) -> Tensor:
        batch_size = latents.shape[0]
        half_h = latents.shape[1]
        half_w = latents.shape[2]
        c_quarter = latents.shape[5]

        latents = F.permute(latents, (0, 5, 1, 3, 2, 4))
        latents = F.reshape(
            latents, (batch_size, c_quarter, half_h * 2, half_w * 2)
        )
        latents = (latents / float(self.vae.config.scaling_factor)) + float(
            self.vae.config.shift_factor or 0.0
        )
        return latents

    @staticmethod
    def _to_numpy(image: Tensor) -> np.ndarray:
        cpu_image: Tensor = image.cast(DType.float32).to(CPU())
        return np.from_dlpack(cpu_image)

    @staticmethod
    def _vector_norm_per_sample(x: Tensor) -> Tensor:
        """Compute per-sample norm for [B, S, C] embeddings."""
        squared = x * x
        squared = F.sum(squared, axis=2)
        squared = F.sum(squared, axis=1)
        return F.sqrt(squared + 1e-12)

    @classmethod
    def _apply_cfg_renormalization(
        cls,
        pos: Tensor,
        pred: Tensor,
        cfg_normalization: bool,
    ) -> Tensor:
        if not cfg_normalization:
            return pred

        ori_pos_norm = cls._vector_norm_per_sample(pos)
        new_pos_norm = cls._vector_norm_per_sample(pred)
        while ori_pos_norm.rank > 1:
            ori_pos_norm = F.squeeze(ori_pos_norm, axis=-1)
        while new_pos_norm.rank > 1:
            new_pos_norm = F.squeeze(new_pos_norm, axis=-1)
        max_new_norm = ori_pos_norm
        # Avoid divide-by-zero and clip only when required.
        safe_new_norm = F.where(
            new_pos_norm > 1e-12,
            new_pos_norm,
            1e-12,
        )
        ratio = max_new_norm / safe_new_norm
        ratio = F.where(new_pos_norm > max_new_norm, ratio, 1.0)
        ratio = F.unsqueeze(F.unsqueeze(ratio, 1), 2)
        return pred * ratio

    @staticmethod
    def scheduler_step(
        latents: Tensor,
        noise_pred: Tensor,
        dt: Tensor,
    ) -> Tensor:
        latents_dtype = latents.dtype
        latents = latents.cast(DType.float32)
        latents = latents + dt * noise_pred
        latents = latents.cast(latents_dtype)
        return latents

    @staticmethod
    def prepare_scheduler(sigmas: Tensor) -> tuple[Tensor, Tensor]:
        """Precompute denoising timesteps and step deltas."""
        sigmas_curr = F.slice_tensor(sigmas, [slice(0, -1)])
        sigmas_next = F.slice_tensor(sigmas, [slice(1, None)])
        all_dt = sigmas_next - sigmas_curr
        all_timesteps = sigmas_curr.cast(DType.float32)
        return all_timesteps, all_dt

    @traced(message="ZImagePipeline.preprocess_latents")
    def preprocess_latents(self, latents: Tensor, dtype: DType) -> Tensor:
        """Patchify and pack latents before denoising."""
        with Tracer("host_to_device_latents"):
            latents = latents.to(self.transformer.devices[0]).cast(
                DType.float32
            )

        with Tracer("patchify_and_pack"):
            batch, channels, height, width = map(int, latents.shape)
            latents = F.reshape(
                latents,
                (
                    batch,
                    channels,
                    height // 2,
                    2,
                    width // 2,
                    2,
                ),
            )
            latents = self._pack_latents_from_6d(latents)

        return latents.cast(dtype)

    def _image_to_tensor(
        self,
        image: np.ndarray,
        batch_size: int,
        dtype: DType,
    ) -> Tensor:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected input image shape [H, W, 3], got {image.shape}."
            )

        height, width, _ = image.shape
        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0 or width % vae_scale != 0:
            raise ValueError(
                f"Input image dimensions must be divisible by {vae_scale}, "
                f"got {height}x{width}."
            )

        image_f32 = image.astype(np.float32) / 127.5 - 1.0
        image_chw = np.transpose(image_f32, (2, 0, 1))
        image_bchw = np.expand_dims(image_chw, axis=0)
        image_bchw = np.ascontiguousarray(image_bchw)

        image_tensor = Tensor(
            storage=Buffer.from_dlpack(image_bchw).to(self.vae.devices[0])
        ).cast(dtype)
        if batch_size > 1:
            image_tensor = F.broadcast_to(
                image_tensor,
                [batch_size, 3, height, width],
            )
        return image_tensor

    @traced(message="ZImagePipeline.prepare_img2img_latents")
    def prepare_img2img_latents(
        self,
        noise_latents: Tensor,
        image_tensor: Tensor,
        sigmas: Tensor,
    ) -> Tensor:
        noise_latents = noise_latents.to(self.transformer.devices[0])

        encoder_output = self.vae.encode(
            image_tensor,
            return_dict=True,
        )
        posterior = (
            encoder_output["latent_dist"]
            if isinstance(encoder_output, dict)
            else encoder_output
        )
        if not hasattr(posterior, "mode"):
            raise ValueError("VAE encoder output must expose mode().")

        image_latents: Tensor = posterior.mode()
        image_latents = (
            image_latents - float(self.vae.config.shift_factor or 0.0)
        ) * float(self.vae.config.scaling_factor)
        image_latents = image_latents.to(self.transformer.devices[0]).cast(
            noise_latents.dtype
        )

        sigma = sigmas[0]
        latents = sigma * noise_latents + (1.0 - sigma) * image_latents
        return latents.cast(noise_latents.dtype)

    def _prepare_timestep_broadcast(
        self,
        timesteps: np.ndarray,
        device: Device,
        cache_key: str | None = None,
    ) -> tuple[Tensor, np.ndarray]:
        transformed_timesteps = (1.0 - timesteps).astype(np.float32, copy=False)

        if cache_key is None:
            num_timesteps = int(transformed_timesteps.shape[0])
            first_t = (
                float(transformed_timesteps[0]) if num_timesteps > 0 else 0.0
            )
            last_t = (
                float(transformed_timesteps[-1]) if num_timesteps > 0 else 0.0
            )
            cache_key = (
                f"timesteps::{num_timesteps}::{first_t:.8f}::{last_t:.8f}"
            )

        if (
            cache_key in self._cached_timesteps_batched
            and cache_key in self._cached_timesteps_host
        ):
            return (
                self._cached_timesteps_batched[cache_key],
                self._cached_timesteps_host[cache_key],
            )

        transformed_timesteps = np.ascontiguousarray(transformed_timesteps)

        timesteps_tensor = Tensor(
            storage=Buffer.from_dlpack(transformed_timesteps).to(device)
        )
        self._cached_timesteps_batched[cache_key] = timesteps_tensor
        self._cached_timesteps_host[cache_key] = transformed_timesteps
        return timesteps_tensor, transformed_timesteps

    def _prepare_scheduler_inputs(
        self,
        model_inputs: ZImageModelInputs,
        sigmas: Tensor,
        device: Device,
    ) -> tuple[Any, Any, np.ndarray]:
        _, all_dts = self.prepare_scheduler(sigmas)
        dts_seq: Any = all_dts
        if hasattr(dts_seq, "driver_tensor"):
            dts_seq = dts_seq.driver_tensor

        timesteps = model_inputs.timesteps
        num_timesteps = timesteps.shape[0]
        timesteps_key = (
            f"timesteps::{num_timesteps}::{model_inputs.height}x"
            f"{model_inputs.width}::{int(model_inputs.input_image is not None)}::"
            f"{model_inputs.num_inference_steps}::{model_inputs.strength:.4f}::"
            f"{float(getattr(self, '_scheduler_shift', 1.0)):.4f}"
        )
        timesteps_seq, transformed_timesteps = self._prepare_timestep_broadcast(
            timesteps=timesteps,
            device=device,
            cache_key=timesteps_key,
        )
        timesteps_seq_any: Any = timesteps_seq
        if hasattr(timesteps_seq_any, "driver_tensor"):
            timesteps_seq_any = timesteps_seq_any.driver_tensor

        return timesteps_seq_any, dts_seq, transformed_timesteps

    @traced(message="ZImagePipeline.execute")
    def execute(  # type: ignore[override]
        self,
        model_inputs: ZImageModelInputs,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> ZImagePipelineOutput:
        """Run the Z-Image denoising loop and decode outputs."""

        # 1) Encode prompt embeddings.
        with Tracer("prepare_prompt_embeddings"):
            prompt_embeds = self.prepare_prompt_embeddings(
                tokens=model_inputs.tokens_tensor,
                num_images_per_prompt=model_inputs.num_images_per_prompt,
            )

            negative_prompt_embeds: Tensor | None = None
            if (
                model_inputs.do_cfg
                and model_inputs.negative_tokens_tensor is not None
            ):
                negative_prompt_embeds = self.prepare_prompt_embeddings(
                    tokens=model_inputs.negative_tokens_tensor,
                    num_images_per_prompt=model_inputs.num_images_per_prompt,
                )
                if not model_inputs.explicit_negative_prompt:
                    negative_prompt_embeds = self._align_prompt_seq_len(
                        negative_prompt_embeds,
                        int(prompt_embeds.shape[1]),
                    )

        dtype = prompt_embeds.dtype
        latents = model_inputs.latents_tensor
        sigmas = model_inputs.sigmas_tensor
        h_carrier = model_inputs.h_carrier
        w_carrier = model_inputs.w_carrier

        timesteps: np.ndarray = model_inputs.timesteps
        batch_size = int(prompt_embeds.shape[0])
        num_timesteps = timesteps.shape[0]
        if num_timesteps < 1:
            raise ValueError("No timesteps were provided for denoising.")

        # 2) Prepare latents and conditioning tensors.
        device = self.transformer.devices[0]
        img_ids = model_inputs.img_ids_tensor
        txt_ids = model_inputs.txt_ids_tensor

        if model_inputs.input_image_tensor is not None:
            latents = self.prepare_img2img_latents(
                noise_latents=latents,
                image_tensor=model_inputs.input_image_tensor,
                sigmas=sigmas,
            )
        latents = self.preprocess_latents(latents, dtype)

        image_seq_len = int(latents.shape[1])
        cache_pos = self.create_cache_state(
            batch_size,
            image_seq_len,
            self.transformer.config,
        )
        cache_neg = (
            self.create_cache_state(
                batch_size,
                image_seq_len,
                self.transformer.config,
            )
            if model_inputs.do_cfg
            else None
        )

        # 3) Prepare scheduler tensors.
        with Tracer("prepare_scheduler"):
            timesteps_seq, dts_seq, transformed_timesteps = (
                self._prepare_scheduler_inputs(
                    model_inputs=model_inputs,
                    sigmas=sigmas,
                    device=device,
                )
            )

        cfg_active: np.ndarray | None = None
        if model_inputs.do_cfg:
            if model_inputs.cfg_truncation <= 1.0:
                cfg_active = (
                    transformed_timesteps <= model_inputs.cfg_truncation
                )
            else:
                cfg_active = np.ones(num_timesteps, dtype=np.bool_)

        # 4) Denoising loop.
        with Tracer("denoising_loop"):
            for i in range(num_timesteps):
                with Tracer(f"denoising_step_{i}"):
                    timestep = timesteps_seq[i : i + 1]
                    apply_cfg = bool(
                        model_inputs.do_cfg
                        and cfg_active is not None
                        and cfg_active[i]
                    )
                    current_guidance_scale = (
                        model_inputs.guidance_scale if apply_cfg else 0.0
                    )

                    with Tracer("transformer"):
                        noise_pred = self.run_denoising_step(
                            step=i,
                            cache_state=cache_pos,
                            device=device,
                            latents=latents,
                            prompt_embeds=prompt_embeds,
                            timestep=timestep,
                            img_ids=img_ids,
                            txt_ids=txt_ids,
                        )

                    if apply_cfg:
                        assert negative_prompt_embeds is not None
                        assert cache_neg is not None
                        neg_img_ids = img_ids
                        neg_txt_ids = txt_ids
                        if model_inputs.explicit_negative_prompt:
                            assert (
                                model_inputs.negative_img_ids_tensor is not None
                            )
                            assert (
                                model_inputs.negative_txt_ids_tensor is not None
                            )
                            neg_img_ids = model_inputs.negative_img_ids_tensor
                            neg_txt_ids = model_inputs.negative_txt_ids_tensor
                        with Tracer("cfg_transformer"):
                            neg_noise_pred = self.run_denoising_step(
                                step=i,
                                cache_state=cache_neg,
                                device=device,
                                latents=latents,
                                prompt_embeds=negative_prompt_embeds,
                                timestep=timestep,
                                img_ids=neg_img_ids,
                                txt_ids=neg_txt_ids,
                            )
                        pos_noise_pred = noise_pred
                        noise_delta = F.sub(noise_pred, neg_noise_pred)
                        noise_pred = F.add(
                            pos_noise_pred,
                            F.mul(noise_delta, current_guidance_scale),
                        )
                        noise_pred = self._apply_cfg_renormalization(
                            pos_noise_pred,
                            noise_pred,
                            model_inputs.cfg_normalization,
                        )

                    with Tracer("scheduler_step"):
                        noise_pred = F.mul(noise_pred, -1.0)
                        dt = dts_seq[i : i + 1]
                        latents = self.scheduler_step(latents, noise_pred, dt)

        with Tracer("decode_outputs"):
            outputs = self.decode_latents(
                latents,
                h_carrier,
                w_carrier,
                output_type=output_type,
            )

        return ZImagePipelineOutput(images=outputs)
