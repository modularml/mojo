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
"""Flux2 Klein executor (ModuleV2, Graph API).

Mirrors the V3 Klein pipeline's three-graph CFG structure on top of the
Graph API stack used by :class:`Flux2Executor`:

    1. run the raw denoise graph with the positive prompt,
    2. optionally run the same graph again with the negative prompt,
    3. optionally blend via a compiled ``cfg_combine`` graph,
    4. apply one compiled Euler scheduler step.

CFG-off takes the single-forward path (one transformer launch per
step); V3 Klein behaves the same way. TaylorSeer caching is supported
on both streams when enabled via the runtime cache config; distilled
checkpoints disable CFG regardless of request inputs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, load_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.tensor import Tensor
from max.graph import DeviceRef
from max.graph import Module as GraphModule
from max.graph.weights import load_weights
from max.pipelines.architectures.qwen3.text_encoder import (
    Qwen3TextEncoderKleinModel,
)
from max.pipelines.context import PixelContext
from max.pipelines.diffusion.cache import (
    TaylorSeerBufferState,
    TaylorSeerCache,
)
from max.pipelines.diffusion.config import DenoisingCacheConfig
from max.pipelines.lib import float32_array_to_buffer
from max.pipelines.lib.compiled_component import CompiledComponent
from max.pipelines.lib.config.model_config import (
    _resolve_component_encoding_and_weights,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_executor import PipelineExecutor
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.modeling.base import TensorStruct
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from max.profiler import Tracer, traced
from typing_extensions import Self

from .components import (
    CfgCombineComponent,
    DenoiseCompute,
    DenoisePredict,
    ImageEncoder,
    VaeDecoder,
)
from .flux2_executor import Flux2ExecutorOutputs

logger = logging.getLogger("max.pipelines")


@dataclass(frozen=True)
class Flux2KleinExecutorInputs(TensorStruct):
    """Structured inputs for Flux2 Klein execution.

    Mirrors the base Flux2 input shape and adds negative-prompt fields
    plus a pre-built ``guidance_scale`` scalar buffer for CFG. All
    fields are Buffer/Tensor to satisfy :class:`TensorStruct`; CFG
    activation is signaled by ``guidance_scale is not None``.
    """

    tokens: Buffer
    """Positive-prompt token IDs for the text encoder, shape ``(S,)``."""

    text_ids: Buffer
    """Positive-prompt text position IDs, shape ``(1, S, 4)`` int64."""

    attention_bias: Buffer
    """Pre-built additive attention bias for the positive prompt, shape
    ``(1, 1, S, S)`` float32."""

    latents: Buffer
    """Packed latent noise tensor, shape ``(B, seq, C*4)``."""

    latent_image_ids: Buffer
    """Latent positional identifiers, shape ``(B, seq, 4)`` int64."""

    timesteps: Buffer
    """Precomputed timesteps, shape ``(num_steps,)`` float32."""

    dts: Buffer
    """Precomputed step deltas, shape ``(num_steps,)`` float32."""

    guidance: Buffer
    """Per-step transformer guidance embedding scalar (broadcast), shape
    ``(B,)`` float32. Distinct from ``guidance_scale`` which governs CFG."""

    image_seq_len: Buffer
    """Packed image sequence length as a 1-element int64 tensor."""

    h_carrier: Buffer
    """Shape carrier of length ``packed_h``; content is never read."""

    w_carrier: Buffer
    """Shape carrier of length ``packed_w``; content is never read."""

    height: Buffer
    """Output image height in pixels as a 1-element int64 tensor."""

    width: Buffer
    """Output image width in pixels as a 1-element int64 tensor."""

    num_inference_steps: Buffer
    """Number of denoising steps as a 1-element int64 tensor."""

    num_images_per_prompt: Buffer
    """Number of images to generate per prompt as a 1-element int64 tensor."""

    input_image: Buffer | None = None
    """Input image for image-to-image generation, shape ``(H, W, C)`` uint8."""

    negative_tokens: Buffer | None = None
    """Negative-prompt token IDs, shape ``(S',)``. ``None`` when no
    negative prompt was supplied."""

    negative_text_ids: Buffer | None = None
    """Negative-prompt text position IDs, shape ``(1, S', 4)`` int64."""

    negative_attention_bias: Buffer | None = None
    """Pre-built additive attention bias for the negative prompt,
    shape ``(1, 1, S', S')`` float32."""

    guidance_scale: Buffer | None = None
    """Scalar float32 CFG scale on device. Presence is the sole signal
    that CFG is active for this request."""

    _CPU_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "num_inference_steps",
            "num_images_per_prompt",
            "height",
            "width",
            "image_seq_len",
            "h_carrier",
            "w_carrier",
        }
    )

    _TEXT_ENCODER_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "tokens",
            "attention_bias",
            "negative_tokens",
            "negative_attention_bias",
        }
    )

    def to(
        self,
        transformer_device: Device,
        text_encoder_device: Device | None = None,
    ) -> Self:
        """Transfer GPU-bound tensors to their target devices.

        Text-encoder inputs (tokens + attention bias, positive and
        negative) go to ``text_encoder_device``; all other device-bound
        tensors go to ``transformer_device``. If ``text_encoder_device``
        is ``None``, both routes collapse to ``transformer_device``.
        """
        text_device = (
            text_encoder_device
            if text_encoder_device is not None
            else transformer_device
        )
        updates: dict[str, Any] = {}
        for f in fields(self):
            if f.name in self._CPU_FIELDS:
                continue
            val = getattr(self, f.name)
            if isinstance(val, (Tensor, Buffer)):
                target = (
                    text_device
                    if f.name in self._TEXT_ENCODER_FIELDS
                    else transformer_device
                )
                updates[f.name] = val.to(target)
        return replace(self, **updates)


class Flux2KleinExecutor(
    PipelineExecutor[
        PixelContext, Flux2KleinExecutorInputs, Flux2ExecutorOutputs
    ]
):
    """Flux2 Klein pipeline executor with classifier-free guidance."""

    default_num_inference_steps: int = 28

    _DEFAULT_VAE_SCALE_FACTOR: int = 8

    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
        runtime_config: PipelineRuntimeConfig,
    ) -> None:
        self._manifest = manifest
        self._session = session
        self._runtime_config = runtime_config
        self._cache_config: DenoisingCacheConfig = (
            runtime_config.denoising_cache
        )

        vae_config = (
            manifest["vae"].huggingface_config.to_dict()
            if "vae" in manifest
            else {}
        )
        block_out_channels = vae_config.get("block_out_channels", None)
        self._vae_scale_factor = (
            2 ** (len(block_out_channels) - 1)
            if block_out_channels
            else self._DEFAULT_VAE_SCALE_FACTOR
        )

        transformer_config = manifest["transformer"]
        resolved_transformer_encoding, _ = (
            _resolve_component_encoding_and_weights(transformer_config)
        )
        encoding = resolved_transformer_encoding or "bfloat16"
        self._model_dtype: DType = (
            DType.bfloat16
            if encoding == "float4_e2m1fnx2"
            else supported_encoding_dtype(encoding)
        )
        if len(transformer_config.device_specs) != 1:
            raise ValueError(
                "FLUX.2-Klein is only supported on a single device"
            )
        model_devices = load_devices(transformer_config.device_specs)
        self._model_device: Device = model_devices[0]
        self._in_channels: int = 128
        self._is_distilled: bool = bool(
            manifest.metadata.get("is_distilled", False)
        )

        text_encoder_entry = manifest["text_encoder"]
        text_encoder_devices = load_devices(text_encoder_entry.device_specs)
        self._text_encoder_device: Device = text_encoder_devices[0]
        resolved_te_encoding, resolved_te_weight_path = (
            _resolve_component_encoding_and_weights(text_encoder_entry)
        )
        self.text_encoder = Qwen3TextEncoderKleinModel(
            config=text_encoder_entry.huggingface_config.to_dict(),
            encoding=resolved_te_encoding or "bfloat16",
            devices=text_encoder_devices,
            weights=load_weights(
                text_encoder_entry.resolved_weight_paths(
                    resolved_te_weight_path
                )
            ),
            session=session,
        )

        # Build Klein's CompiledComponent graphs into one shared Module so
        # we can compile them all together via session.load_all (MODELS-1440).
        # Note: ``self.text_encoder`` (Qwen3TextEncoderKleinModel) is not a
        # CompiledComponent and continues to manage its own compile.
        self._graphs_module = GraphModule()

        self.image_encoder = ImageEncoder(
            manifest, session, graphs_module=self._graphs_module
        )
        self.decoder = VaeDecoder(
            manifest, session, graphs_module=self._graphs_module
        )

        self.denoise_compute = DenoiseCompute(
            manifest, session, graphs_module=self._graphs_module
        )
        self.denoise_predict = DenoisePredict(
            manifest,
            session,
            dtype=self._model_dtype,
            device=self._model_device,
            graphs_module=self._graphs_module,
        )
        self.cfg_combiner = CfgCombineComponent(
            manifest,
            session,
            dtype=self._model_dtype,
            device=DeviceRef.from_device(self._model_device),
            graphs_module=self._graphs_module,
        )

        components: list[CompiledComponent] = [
            self.image_encoder,
            self.decoder,
            self.denoise_compute,
            self.denoise_predict,
            self.cfg_combiner,
        ]

        combined_registry: dict[str, Any] = {}
        for component in components:
            for key, value in component._pending_weights.items():
                if key in combined_registry:
                    raise RuntimeError(
                        f"FLUX2 Klein load_all: weight key {key!r} appears "
                        f"in multiple components; rename one to disambiguate."
                    )
                combined_registry[key] = value

        graph_names = [c._pending_graph_name for c in components]
        logger.info(
            "Compiling FLUX2 Klein graphs via session.load_all "
            "(%d graphs: %s)...",
            len(graph_names),
            ", ".join(repr(n) for n in graph_names if n is not None),
        )
        t0 = time.perf_counter()
        with Tracer("Flux2KleinExecutor.compile_load_all"):
            models = session.load_all(
                self._graphs_module, weights_registry=combined_registry
            )
        elapsed = time.perf_counter() - t0
        logger.info(
            "Compiled FLUX2 Klein graphs via session.load_all "
            "(%d graphs) in %.2fs",
            len(models),
            elapsed,
        )

        for component in components:
            component._attach_compiled_model(models)

        self._taylor_cache: TaylorSeerCache | None = None
        if self._cache_config.taylorseer:
            self._taylor_cache = TaylorSeerCache(
                config=self._cache_config,
                dtype=self._model_dtype,
                device=self._model_device,
                session=session,
            )

    @traced(message="Flux2KleinExecutor.prepare_inputs")
    def prepare_inputs(
        self, contexts: list[PixelContext]
    ) -> Flux2KleinExecutorInputs:
        if len(contexts) != 1:
            raise ValueError(
                "Flux2KleinExecutor currently supports batch_size=1. "
                f"Got {len(contexts)} contexts."
            )
        context = contexts[0]

        if context.latents.size == 0:
            raise ValueError(
                "Flux2KleinExecutor requires non-empty latents in PixelContext"
            )
        if context.latent_image_ids.size == 0:
            raise ValueError(
                "Flux2KleinExecutor requires non-empty latent_image_ids "
                "in PixelContext"
            )
        if context.sigmas.size == 0:
            raise ValueError(
                "Flux2KleinExecutor requires non-empty sigmas in PixelContext"
            )

        latent_h = context.height // self._vae_scale_factor
        latent_w = context.width // self._vae_scale_factor
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        image_seq_len = packed_h * packed_w

        tokens_np = context.tokens.array
        if tokens_np.ndim == 2:
            if tokens_np.shape[0] != 1:
                raise ValueError(
                    "Flux2KleinExecutor expects batch_size=1 for 2D tokens."
                )
            tokens_np = tokens_np[0]
        tokens = Buffer.from_dlpack(tokens_np)
        text_ids = Buffer.from_dlpack(context.text_ids)
        attention_bias = self._build_attention_bias(context.mask, tokens_np)

        latents = self._patchify_and_pack(context.latents)
        latent_image_ids = Buffer.from_dlpack(context.latent_image_ids)
        timesteps, dts = self._prepare_scheduler(context.sigmas)

        guidance = Buffer.from_dlpack(
            np.full(
                [context.num_images_per_prompt],
                context.guidance_scale,
                dtype=np.float32,
            )
        )

        h_carrier = Buffer.from_dlpack(np.empty(packed_h, dtype=np.float32))
        w_carrier = Buffer.from_dlpack(np.empty(packed_w, dtype=np.float32))

        height = Buffer.from_dlpack(np.array([context.height], dtype=np.int64))
        width = Buffer.from_dlpack(np.array([context.width], dtype=np.int64))
        num_inference_steps = Buffer.from_dlpack(
            np.array([context.num_inference_steps], dtype=np.int64)
        )
        num_images_per_prompt = Buffer.from_dlpack(
            np.array([context.num_images_per_prompt], dtype=np.int64)
        )
        image_seq_len_buf = Buffer.from_dlpack(
            np.array([image_seq_len], dtype=np.int64)
        )

        input_image: Buffer | None = None
        if context.input_image is not None:
            input_image = Buffer.from_dlpack(context.input_image)

        negative_tokens: Buffer | None = None
        negative_text_ids: Buffer | None = None
        negative_attention_bias: Buffer | None = None
        guidance_scale_buf: Buffer | None = None

        has_negative = context.negative_tokens is not None
        want_cfg = context.guidance_scale > 1.0
        if self._is_distilled and want_cfg and context.explicit_negative_prompt:
            logger.warning(
                "Guidance scale %s is ignored for distilled Klein models.",
                context.guidance_scale,
            )
        enable_cfg = has_negative and want_cfg and not self._is_distilled
        if enable_cfg:
            assert context.negative_tokens is not None
            negative_tokens_np = context.negative_tokens.array
            if negative_tokens_np.ndim == 2:
                if negative_tokens_np.shape[0] != 1:
                    raise ValueError(
                        "Flux2KleinExecutor expects batch_size=1 for 2D "
                        "negative tokens."
                    )
                negative_tokens_np = negative_tokens_np[0]
            negative_tokens = Buffer.from_dlpack(negative_tokens_np)
            if context.negative_text_ids.size > 0:
                negative_text_ids = Buffer.from_dlpack(
                    context.negative_text_ids
                )
            negative_attention_bias = self._build_attention_bias(
                context.negative_mask, negative_tokens_np
            )
            guidance_scale_buf = Buffer.from_dlpack(
                np.array(context.guidance_scale, dtype=np.float32)
            )
        elif (
            not has_negative
            and want_cfg
            and not self._is_distilled
            and context.explicit_negative_prompt
        ):
            logger.warning(
                "CFG requested (guidance_scale=%s) but no negative prompt "
                "was supplied; running without CFG.",
                context.guidance_scale,
            )

        return Flux2KleinExecutorInputs(
            tokens=tokens,
            text_ids=text_ids,
            attention_bias=attention_bias,
            latents=latents,
            latent_image_ids=latent_image_ids,
            timesteps=timesteps,
            dts=dts,
            guidance=guidance,
            image_seq_len=image_seq_len_buf,
            h_carrier=h_carrier,
            w_carrier=w_carrier,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            input_image=input_image,
            negative_tokens=negative_tokens,
            negative_text_ids=negative_text_ids,
            negative_attention_bias=negative_attention_bias,
            guidance_scale=guidance_scale_buf,
        )

    @traced(message="Flux2KleinExecutor.execute")
    def execute(self, inputs: Flux2KleinExecutorInputs) -> Flux2ExecutorOutputs:
        inputs = inputs.to(
            transformer_device=self._model_device,
            text_encoder_device=self._text_encoder_device,
        )

        do_cfg = inputs.guidance_scale is not None

        prompt_embeds = self._encode_prompt(
            inputs.tokens, inputs.attention_bias
        )

        negative_prompt_embeds: Buffer | None = None
        if do_cfg:
            assert inputs.negative_tokens is not None
            assert inputs.negative_attention_bias is not None
            negative_prompt_embeds = self._encode_prompt(
                inputs.negative_tokens, inputs.negative_attention_bias
            )

        if inputs.input_image is not None:
            image_latents, image_latent_ids = self.image_encoder(
                inputs.input_image
            )
        else:
            image_latents = self._empty_image_latents()
            image_latent_ids = self._empty_image_latent_ids()

        latents = self._run_denoising_loop(
            latents=inputs.latents,
            image_latents=image_latents,
            image_latent_ids=image_latent_ids,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            text_ids=inputs.text_ids,
            negative_text_ids=inputs.negative_text_ids,
            latent_image_ids=inputs.latent_image_ids,
            timesteps=inputs.timesteps,
            dts=inputs.dts,
            guidance=inputs.guidance,
            guidance_scale=inputs.guidance_scale,
            num_inference_steps=inputs.num_inference_steps,
            do_cfg=do_cfg,
        )

        images = self.decoder(latents, inputs.h_carrier, inputs.w_carrier)
        return Flux2ExecutorOutputs(images=images)

    @traced(message="Flux2KleinExecutor.encode_prompt")
    def _encode_prompt(self, tokens: Buffer, attention_bias: Buffer) -> Buffer:
        """Execute the compiled Qwen3 text encoder with a pre-built bias."""
        return self.text_encoder.encode_with_bias(tokens, attention_bias)

    @traced(message="Flux2KleinExecutor.run_denoising_loop")
    def _run_denoising_loop(
        self,
        latents: Buffer,
        image_latents: Buffer,
        image_latent_ids: Buffer,
        prompt_embeds: Buffer,
        negative_prompt_embeds: Buffer | None,
        text_ids: Buffer,
        negative_text_ids: Buffer | None,
        latent_image_ids: Buffer,
        timesteps: Buffer,
        dts: Buffer,
        guidance: Buffer,
        guidance_scale: Buffer | None,
        num_inference_steps: Buffer,
        do_cfg: bool,
    ) -> Buffer:
        num_steps: int = np.from_dlpack(num_inference_steps).item()  # type: ignore[assignment]

        state_pos: TaylorSeerBufferState | None = None
        state_neg: TaylorSeerBufferState | None = None
        if self._taylor_cache is not None:
            batch_size, seq_len, output_dim = latents.shape
            state_pos = self._taylor_cache.create_state(
                batch_size, seq_len, output_dim
            )
            if do_cfg:
                state_neg = self._taylor_cache.create_state(
                    batch_size, seq_len, output_dim
                )

        for i in range(num_steps):
            timestep_i = timesteps[i : i + 1]
            dt_i = dts[i : i + 1]

            noise_pred = self._stream_noise_pred(
                step=i,
                latents=latents,
                image_latents=image_latents,
                prompt_embeds=prompt_embeds,
                timestep=timestep_i,
                guidance=guidance,
                latent_image_ids=latent_image_ids,
                image_latent_ids=image_latent_ids,
                text_ids=text_ids,
                state=state_pos,
            )

            if do_cfg:
                assert negative_prompt_embeds is not None
                assert negative_text_ids is not None
                assert guidance_scale is not None
                neg_noise_pred = self._stream_noise_pred(
                    step=i,
                    latents=latents,
                    image_latents=image_latents,
                    prompt_embeds=negative_prompt_embeds,
                    timestep=timestep_i,
                    guidance=guidance,
                    latent_image_ids=latent_image_ids,
                    image_latent_ids=image_latent_ids,
                    text_ids=negative_text_ids,
                    state=state_neg,
                )
                noise_pred = self.cfg_combiner(
                    noise_pred, neg_noise_pred, guidance_scale
                )

            latents = self.denoise_predict(latents, noise_pred, dt_i)

        return latents

    def _stream_noise_pred(
        self,
        *,
        step: int,
        latents: Buffer,
        image_latents: Buffer,
        prompt_embeds: Buffer,
        timestep: Buffer,
        guidance: Buffer,
        latent_image_ids: Buffer,
        image_latent_ids: Buffer,
        text_ids: Buffer,
        state: TaylorSeerBufferState | None,
    ) -> Buffer:
        """Return ``noise_pred`` for one CFG stream, honoring TaylorSeer.

        When the TaylorSeer cache schedule says to skip, returns a
        Taylor-predicted noise; otherwise runs the transformer and
        updates the cache in-place.
        """
        if (
            self._taylor_cache is not None
            and state is not None
            and self._taylor_cache.should_skip(step)
        ):
            return self._taylor_cache.predict(state, step)

        noise_pred = self.denoise_compute(
            latents,
            image_latents,
            prompt_embeds,
            timestep,
            guidance,
            latent_image_ids,
            image_latent_ids,
            text_ids,
        )
        if self._taylor_cache is not None and state is not None:
            self._taylor_cache.update(state, noise_pred, step)
        return noise_pred

    @staticmethod
    def _build_attention_bias(
        mask: npt.NDArray[np.bool_] | None,
        tokens_np: npt.NDArray[np.int64],
    ) -> Buffer:
        """Build a causal + padding additive bias Buffer from an optional mask.

        Reuses the same static helper as the V3 Klein path so positive
        and negative prompts share bias semantics with the encoder.
        """
        seq_len = int(tokens_np.shape[0])
        attention_mask_np = (
            np.asarray(mask)
            if mask is not None
            else np.ones((seq_len,), dtype=np.bool_)
        )
        bias_np = (
            Qwen3TextEncoderKleinModel.attention_bias_from_attention_mask_array(
                attention_mask_np, expected_seq_len=seq_len
            )
        )
        return Buffer.from_dlpack(np.ascontiguousarray(bias_np))

    def _patchify_and_pack(
        self,
        latents: npt.NDArray[np.float32],
    ) -> Buffer:
        """Patchify ``(B, C, H, W)`` -> ``(B, H//2 * W//2, C*4)`` latents."""
        arr = latents
        b, c, h, w = arr.shape
        h2, w2 = h // 2, w // 2
        arr = arr.reshape(b, c, h2, 2, w2, 2)
        arr = arr.transpose(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h2, w2)
        arr = arr.reshape(b, c * 4, h2 * w2).transpose(0, 2, 1)
        arr = np.ascontiguousarray(arr)
        return float32_array_to_buffer(
            arr, dtype=self._model_dtype, device=self._model_device
        )

    def _empty_image_latents(self) -> Buffer:
        """Zero-seq image latent placeholder for text-to-image."""
        return float32_array_to_buffer(
            np.zeros((1, 0, self._in_channels), dtype=np.float32),
            dtype=self._model_dtype,
            device=self._model_device,
        )

    def _empty_image_latent_ids(self) -> Buffer:
        """Zero-seq image latent-ID placeholder for text-to-image."""
        return Buffer.from_dlpack(np.zeros((1, 0, 4), dtype=np.int64)).to(
            self._model_device
        )

    def _prepare_scheduler(
        self,
        sigmas: npt.NDArray[np.float32],
    ) -> tuple[Buffer, Buffer]:
        """Precompute ``(timesteps, dts)`` from a sigma schedule."""
        timesteps = np.ascontiguousarray(sigmas[:-1])
        dts = np.ascontiguousarray(sigmas[1:] - sigmas[:-1])
        return (
            Buffer.from_dlpack(timesteps),
            Buffer.from_dlpack(dts),
        )
