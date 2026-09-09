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
"""Flux2 single-``Module`` model (ModuleV3 scaffold)."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device, load_devices
from max.dtype import DType
from max.experimental import functional
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.pipelines.architectures.flux2.flux2_executor import (
    Flux2ExecutorOutputs,
)
from max.pipelines.architectures.flux_modulev3 import Vae
from max.pipelines.context import PixelContext
from max.pipelines.lib import float32_array_to_buffer
from max.pipelines.lib.config.model_config import (
    _resolve_component_encoding_and_weights,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.weight_loader import WeightLoader, swap_prefix
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from max.profiler import traced

from .components import Denoiser, TextEncoder
from .flux2_inputs import Flux2ModuleV3Inputs

logger = logging.getLogger("max.pipelines")


class FLUXModule(Module[..., tuple[Tensor, Tensor, Tensor]]):
    """Single-``Module`` implementation of the FLUX.2 pipeline.

    Expresses the entire FLUX.2 forward pass -- text encode, optional
    image encode, denoising loop, and VAE decode -- inside a single
    :class:`~max.experimental.nn.Module` tree, with sub-Modules placed
    on the device specified by their corresponding manifest entry (for
    example, ``text_encoder`` on
    ``manifest["text_encoder"].device_specs[0]``).

    No graph construction, weight binding, or session compilation lives
    on this class.  The caller (typically
    :class:`~max.pipelines.diffusion.pipeline.PixelGenerationPipeline`)
    is responsible for:

    1. Wrapping construction in :func:`max.experimental.functional.lazy`.
    2. Composing ``manifest.loader()`` through
       :func:`~max.pipelines.lib.weight_loader.adapt_module_loader` and
       materialising the Module's declared parameters into a
       compile-ready ``weights`` mapping (the walker descends into every
       ``HasLoaderAdapter`` sub-Module automatically).
    3. Calling :meth:`~max.experimental.nn.Module.compile` with
       :meth:`input_types` and the materialised weights.
    4. Driving the resulting ``CompiledModel`` against
       :meth:`prepare_inputs` and :meth:`from_outputs` for each batch.
    """

    # Fallback VAE scale factor when not derivable from the manifest.
    _DEFAULT_VAE_SCALE_FACTOR: int = 8

    def __init__(self, manifest: ModelManifest) -> None:
        self._manifest = manifest

        # Derive VAE scale factor from manifest config, falling back to 8.
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
        # Latent channel count for the pre-patchify noise tensor.  The
        # 2x2 patchify done in :meth:`_patchify_and_pack` produces a
        # ``num_channels = latent_channels * 4`` sequence that the
        # denoiser consumes.
        self._latent_channels: int = int(vae_config.get("latent_channels", 32))

        # Extract transformer config for input-staging dtype.
        transformer_config = manifest["transformer"]
        resolved_transformer_encoding, _ = (
            _resolve_component_encoding_and_weights(transformer_config)
        )
        encoding = resolved_transformer_encoding or "bfloat16"
        # For NVFP4, weights are stored as FP4 but compute stays bfloat16.
        self._model_dtype: DType = (
            DType.bfloat16
            if encoding == "float4_e2m1fnx2"
            else supported_encoding_dtype(encoding)
        )

        # Sub-Modules: built up component-by-component.  Each component
        # receives its own ``device_specs`` from the corresponding
        # manifest entry and is responsible for placing itself on the
        # appropriate device inside its own constructor.  Their
        # devices are also cached on this Module so :meth:`prepare_inputs`
        # can stage Buffers onto the matching device before the
        # compiled graph runs.
        text_encoder_config = manifest["text_encoder"]
        self._text_encoder_device: Device = load_devices(
            text_encoder_config.device_specs
        )[0]
        resolved_te_encoding, _ = _resolve_component_encoding_and_weights(
            text_encoder_config
        )
        self.text_encoder = TextEncoder(
            huggingface_config=text_encoder_config.huggingface_config.to_dict(),
            quantization_encoding=resolved_te_encoding,
            device_specs=text_encoder_config.device_specs,
        )

        vae_manifest_config = manifest["vae"]
        self._vae_device: Device = load_devices(
            vae_manifest_config.device_specs
        )[0]
        resolved_vae_encoding, _ = _resolve_component_encoding_and_weights(
            vae_manifest_config
        )
        self.vae = Vae(
            huggingface_config=vae_manifest_config.huggingface_config.to_dict(),
            quantization_encoding=resolved_vae_encoding,
            device_specs=vae_manifest_config.device_specs,
        )
        transformer_config = manifest["transformer"]
        self._transformer_device: Device = load_devices(
            transformer_config.device_specs
        )[0]
        self.denoiser = Denoiser(
            huggingface_config=transformer_config.huggingface_config.to_dict(),
            quantization_encoding=resolved_transformer_encoding,
            device_specs=transformer_config.device_specs,
        )

    # -- Module forward + compile surface -------------------------------------

    def forward(
        self,
        tokens: Tensor,
        input_image: Tensor,
        latents: Tensor,
        num_inference_steps: Tensor,
        h_carrier: Tensor,
        w_carrier: Tensor,
        timesteps: Tensor,
        dts: Tensor,
        guidance: Tensor,
        text_ids: Tensor,
        latent_image_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through the Module tree.

        Runs the text encoder and VAE image encoder unconditionally,
        threads the caller-supplied ``latents`` through the denoising
        loop via :func:`~max.graph.ops.while_loop` for exactly
        ``num_inference_steps`` iterations.

        Initial noise is pre-generated on CPU by :meth:`prepare_inputs`
        using ``np.random.RandomState(seed).standard_normal()``,
        matching the legacy V2 executor.  Sampling in the graph (via
        ``ops.random``) would diverge from V2 because MAX's in-graph
        RNG produces different values than numpy's ``RandomState`` for
        the same seed, breaking V2/V3 accuracy parity.

        ``h_carrier`` and ``w_carrier`` carry the latent grid shape:
        ``packed_h = h_carrier.shape[0]``,
        ``packed_w = w_carrier.shape[0]``.  Their contents are
        irrelevant -- only their dimensions are read.

        ``timesteps`` and ``dts`` carry the full sigma schedule; the
        loop body gathers a single element from each per iteration
        using the carried ``step`` index.  ``guidance``, ``text_ids``,
        and ``latent_image_ids`` are loop-invariant denoiser conditioning
        captured by closure.

        Returns:
            ``(text_embeddings, image_latents, final_latents)``.
        """
        text_embeddings = self.text_encoder(tokens)
        image_latents = self.vae.encode(input_image)

        # Use ``guidance``'s symbolic ``"batch"`` dim as the canonical
        # batch dim for the whole pipeline.  The text encoder and VAE
        # encoder both produce hardcoded ``batch=1`` outputs (their
        # ``unsqueeze(0)`` adds a literal batch dim), so rebind both
        # onto the symbolic ``"batch"`` here -- this matches the
        # broadcast that happens inside :class:`Flux2TimestepGuidanceEmbeddings`
        # (timestep emb is ``[1, embed_dim]``, guidance emb is
        # ``[batch, embed_dim]``; their sum is ``[batch, embed_dim]``),
        # so every downstream tensor that touches modulation params plus
        # the concat with ``image_latents`` inside the denoiser sees a
        # consistent batch dim.  Single-context-only today; the symbolic
        # dim resolves to ``num_images_per_prompt`` at runtime, which
        # :meth:`prepare_inputs` enforces to be 1.
        #
        # The VAE encoder runs unconditionally even for text-to-image
        # requests: the pipeline feeds a ``(0, 0, 3)`` placeholder image
        # whose zero spatial dims propagate through ``group_norm_gpu``,
        # ``conv_gpu``, and ``flash_attention`` (all hardened with
        # zero-size early-return guards) and end up as a
        # ``(1, 0, num_channels)`` packed-latent output that the
        # denoiser concats onto its noise latents as a no-op.
        batch = guidance.shape[0]
        text_embeddings = F.rebind(
            text_embeddings,
            [batch, text_embeddings.shape[1], text_embeddings.shape[2]],
        )
        image_latents = F.rebind(
            image_latents,
            [batch, image_latents.shape[1], image_latents.shape[2]],
        )
        # The encoder lives on ``self._vae_device``; the denoiser's
        # ``F.concat([latents, image_latents], axis=1)`` runs on
        # ``self._transformer_device``.  Move latents across.
        image_latents = image_latents.to(self._transformer_device)

        # Use the caller-supplied initial latents.  Pre-generated on
        # CPU by :meth:`prepare_inputs` using
        # ``np.random.RandomState(seed).standard_normal()`` and then
        # patchified + packed to match the V2 executor's noise shape
        # and distribution.  The ``rebind`` pins the batch dim to the
        # canonical ``batch`` (from ``guidance``) and the sequence dim
        # to the symbolic ``image_seq`` declared in :meth:`input_types`
        # so MOGG's Mojo-kernel lowering pass can resolve dynamic dims
        # via a single ``ParamDeclRef`` rather than a compound expression.
        seq = latent_image_ids.shape[1]
        initial_latents = F.rebind(
            latents,
            [batch, seq, self.vae.num_channels],
        )

        # Denoising loop.  Every tensor the body or predicate touches is
        # threaded through the carry explicitly -- closure-captured
        # TensorValues don't survive MOGG->MGP lowering when a Mojo
        # kernel inside the loop body (flash-attention, RoPE) tries to
        # resolve their symbolic dims (the lowering pass crashes in
        # ``getTensorDynamicDimensions`` because the dim was defined in
        # the parent graph scope, not the loop region).  Step lives on
        # CPU so the predicate's bool scalar lands on CPU as
        # ``ops.while_loop`` requires.  Per-iteration ``timestep`` and
        # ``dt`` are gathered from the schedule arrays (on the
        # transformer device) using ``step`` transferred to the same
        # device.  ``functional.while_loop`` exposes a Tensor-only
        # surface: predicate and body take and return experimental
        # Tensors; the wrapper handles the underlying TensorValue
        # plumbing.
        zero_step = Tensor.zeros([1], dtype=DType.int64, device=CPU())

        def predicate(
            latents: Tensor,
            step: Tensor,
            num_inference_steps: Tensor,
            text_embeddings: Tensor,
            image_latents: Tensor,
            guidance: Tensor,
            text_ids: Tensor,
            latent_image_ids: Tensor,
            timesteps: Tensor,
            dts: Tensor,
        ) -> Tensor:
            return step[0] < num_inference_steps[0]

        def body(
            latents: Tensor,
            step: Tensor,
            num_inference_steps: Tensor,
            text_embeddings: Tensor,
            image_latents: Tensor,
            guidance: Tensor,
            text_ids: Tensor,
            latent_image_ids: Tensor,
            timesteps: Tensor,
            dts: Tensor,
        ) -> list[Tensor]:
            step_device = step.to(self._transformer_device)
            timestep = F.gather(timesteps, step_device, axis=0)
            dt = F.gather(dts, step_device, axis=0)
            updated = self.denoiser(
                latents,
                image_latents,
                text_embeddings,
                timestep,
                dt,
                guidance,
                latent_image_ids,
                text_ids,
            )
            return [
                updated,
                step + 1,
                num_inference_steps,
                text_embeddings,
                image_latents,
                guidance,
                text_ids,
                latent_image_ids,
                timesteps,
                dts,
            ]

        results = functional.while_loop(
            initial_values=[
                initial_latents,
                zero_step,
                num_inference_steps,
                text_embeddings,
                image_latents,
                guidance,
                text_ids,
                latent_image_ids,
                timesteps,
                dts,
            ],
            predicate=predicate,
            body=body,
        )
        # ``functional.while_loop`` returns experimental Tensors; the
        # wrapper handles the underlying TensorValue plumbing on both
        # the callback and result sides.
        final_latents = results[0]

        image = self.vae.decode(final_latents, h_carrier, w_carrier)

        return text_embeddings, image_latents, image

    def input_types(self) -> tuple[TensorType, ...]:
        """Input tensor types for compilation, sourced from sub-Modules."""
        transformer_ref = DeviceRef.from_device(self._transformer_device)
        return (
            *self.text_encoder.input_types(),
            *self.vae.input_types(),
            # latents: pre-generated initial noise, shape
            # ``(batch, image_seq, num_channels)`` in the model dtype.
            # Pre-generated on CPU by :meth:`prepare_inputs` using
            # ``np.random.RandomState(seed).standard_normal()`` to match
            # the V2 executor's noise distribution.
            TensorType(
                self._model_dtype,
                ["batch", "image_seq", self.denoiser.transformer.in_channels],
                device=transformer_ref,
            ),
            # num_inference_steps (int64, [1]) on CPU for the predicate.
            TensorType(DType.int64, [1], device=DeviceRef.CPU()),
            # h_carrier, w_carrier: shape carriers on the transformer device.
            TensorType(DType.float32, ["packed_h"], device=transformer_ref),
            TensorType(DType.float32, ["packed_w"], device=transformer_ref),
            # timesteps, dts: full sigma schedule on the transformer
            # device; the loop body gathers one element per iteration.
            TensorType(DType.float32, ["num_steps"], device=transformer_ref),
            TensorType(DType.float32, ["num_steps"], device=transformer_ref),
            # guidance: per-batch guidance scale, loop-invariant.
            TensorType(DType.float32, ["batch"], device=transformer_ref),
            # text_ids, latent_image_ids: position IDs on the transformer
            # device, loop-invariant.
            TensorType(
                DType.int64,
                ["batch", "text_seq", 4],
                device=transformer_ref,
            ),
            TensorType(
                DType.int64,
                ["batch", "image_seq", 4],
                device=transformer_ref,
            ),
        )

    # -- Pipeline I/O contract -----------------------------------------------

    @traced(message="prepare_inputs")
    def prepare_inputs(
        self, contexts: list[PixelContext]
    ) -> Flux2ModuleV3Inputs:
        if len(contexts) != 1:
            raise ValueError(
                "FLUXModule currently supports batch_size=1. "
                f"Got {len(contexts)} contexts."
            )
        context = contexts[0]

        if context.latent_image_ids.size == 0:
            raise ValueError(
                "FLUXModule requires non-empty latent_image_ids in PixelContext"
            )
        if context.sigmas.size == 0:
            raise ValueError(
                "FLUXModule requires non-empty sigmas in PixelContext"
            )
        if context.seed is None:
            raise ValueError(
                "FLUXModule requires a seed on PixelContext; "
                "pass `seed` on the request body so the graph can sample "
                "initial noise deterministically."
            )

        latent_h = context.height // self._vae_scale_factor
        latent_w = context.width // self._vae_scale_factor
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        image_seq_len = packed_h * packed_w

        tokens = Buffer.from_dlpack(context.tokens.array).to(
            self._text_encoder_device
        )
        # ``text_ids`` and ``latent_image_ids`` are denoiser conditioning
        # consumed inside the loop body, so they land on the transformer
        # device alongside the schedule and guidance buffers.
        text_ids = Buffer.from_dlpack(context.text_ids).to(
            self._transformer_device
        )
        # Generate initial noise on CPU matching V2's executor:
        # ``np.random.RandomState(seed).standard_normal()`` produces
        # the bit-identical noise the V2 path uses (and matches the
        # legacy tokenizer's ``_randn_tensor``).  The MAX in-graph RNG
        # produces different values for the same seed, which is why
        # the previous ``random.gaussian`` path broke V2/V3 parity.
        latents = self._patchify_and_pack(
            self._sample_latents(context),
        )
        latent_image_ids = Buffer.from_dlpack(context.latent_image_ids).to(
            self._transformer_device
        )
        timesteps, dts = self._prepare_scheduler(context.sigmas)
        timesteps = timesteps.to(self._transformer_device)
        dts = dts.to(self._transformer_device)

        guidance = Buffer.from_dlpack(
            np.full(
                [context.num_images_per_prompt],
                context.guidance_scale,
                dtype=np.float32,
            )
        ).to(self._transformer_device)

        # ``h_carrier`` / ``w_carrier`` give the graph the latent grid
        # shape via their dimensions; placed on the transformer device
        # because ``initial_latents`` is sampled there.
        h_carrier = Buffer.from_dlpack(np.empty(packed_h, dtype=np.float32)).to(
            self._transformer_device
        )
        w_carrier = Buffer.from_dlpack(np.empty(packed_w, dtype=np.float32)).to(
            self._transformer_device
        )

        height = Buffer.from_dlpack(np.array([context.height], dtype=np.int64))
        width = Buffer.from_dlpack(np.array([context.width], dtype=np.int64))
        # ``num_inference_steps`` stays on CPU because the while-loop
        # predicate compares CPU bool tensors.
        num_inference_steps = Buffer.from_dlpack(
            np.array([context.num_inference_steps], dtype=np.int64)
        )
        num_images_per_prompt = Buffer.from_dlpack(
            np.array([context.num_images_per_prompt], dtype=np.int64)
        )
        image_seq_len_buf = Buffer.from_dlpack(
            np.array([image_seq_len], dtype=np.int64)
        )

        # Always stage an input_image buffer on the VAE's device.  When
        # the request is text-only the placeholder has shape ``(0, 0, 3)``
        # so the encoder runs without contributing to the denoiser.
        if context.input_image is not None:
            input_image_array = context.input_image
        else:
            input_image_array = np.empty((0, 0, 3), dtype=np.uint8)
        input_image = Buffer.from_dlpack(input_image_array).to(self._vae_device)

        return Flux2ModuleV3Inputs(
            tokens=tokens,
            text_ids=text_ids,
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
        )

    def from_outputs(self, buffers: list[Buffer]) -> Flux2ExecutorOutputs:
        """Translate compiled-call output buffers into pipeline-facing outputs.

        The compiled :class:`~max.experimental.nn.Module` returns
        ``(text_embeddings, image_latents, image)`` where ``image`` is
        a decoded ``(B, H, W, 3)`` uint8 buffer on CPU.  Only the last
        output is surfaced; the intermediates exist for diagnostic
        purposes today and will be dropped once the pipeline is fully
        verified.
        """
        return Flux2ExecutorOutputs(images=buffers[-1])

    @staticmethod
    def adapt_loader(loader: WeightLoader) -> WeightLoader:
        """Rebase the ``transformer`` role onto the ``denoiser`` attribute.

        The :func:`~max.pipelines.lib.weight_loader.adapt_module_loader`
        walker scopes each sub-Module's adapter by its attribute path.
        The ``transformer.*`` manifest role does not line up with the
        denoiser sub-Module's attribute name (:attr:`FLUXModule.denoiser`),
        so we swap the prefix here.  :func:`swap_prefix` keeps ``keys``
        faithful to the outward ``denoiser.`` namespace so the denoiser's
        own adapter can enumerate its source keys; the denoiser then maps
        each key under ``transformer.`` to match ``Denoiser.transformer``
        (:class:`Flux2Transformer2DModel`).

        The ``vae.*`` role lands directly on :attr:`FLUXModule.vae`
        without alias surgery: a single
        :class:`~max.pipelines.architectures.flux_modulev3.vae.Vae`
        Module owns both encoder and decoder, with the shared BN
        parameters shared by construction rather than duplicated across
        sibling Modules.

        Renaming reuses the same ``WeightData`` handles -- no underlying
        weight bytes are copied.
        """
        return swap_prefix(loader, outer="denoiser", inner="transformer")

    # -- Input staging helpers ------------------------------------------------

    def _prepare_scheduler(
        self,
        sigmas: npt.NDArray[np.float32],
    ) -> tuple[Buffer, Buffer]:
        """Precompute ``(timesteps, dts)`` buffers from the sigma schedule."""
        timesteps = np.ascontiguousarray(sigmas[:-1])
        dts = np.ascontiguousarray(sigmas[1:] - sigmas[:-1])
        return (
            Buffer.from_dlpack(timesteps),
            Buffer.from_dlpack(dts),
        )

    def _sample_latents(self, context: PixelContext) -> npt.NDArray[np.float32]:
        """Sample initial Gaussian noise on CPU using the request seed.

        Prefers the latents already cooked by the FLUX.2 tokenizer
        (``context.latents``) so V3 inherits bit-identical noise from
        V2's :class:`Flux2Tokenizer` path.  Falls back to seeding a
        ``np.random.RandomState`` locally when the tokenizer hasn't
        populated the field (defensive: every production call site
        currently sets it).
        """
        if context.latents.size != 0:
            return np.ascontiguousarray(context.latents)

        latent_h = context.height // self._vae_scale_factor
        latent_w = context.width // self._vae_scale_factor
        shape = (
            context.num_images_per_prompt,
            self._latent_channels,
            latent_h,
            latent_w,
        )
        rng = np.random.RandomState(context.seed)
        return rng.standard_normal(shape).astype(np.float32)

    def _patchify_and_pack(self, latents: npt.NDArray[np.float32]) -> Buffer:
        """Patchify and pack raw latents for the transformer.

        Mirrors :meth:`Flux2Executor._patchify_and_pack`: reshapes
        ``(B, C, H, W)`` -> ``(B, H//2 * W//2, C*4)`` via 2x2
        patchification followed by sequence packing, then casts to
        the model dtype and uploads to the transformer device.
        """
        arr = latents
        b, c, h, w = arr.shape
        h2, w2 = h // 2, w // 2
        arr = arr.reshape(b, c, h2, 2, w2, 2)
        arr = arr.transpose(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h2, w2)
        arr = arr.reshape(b, c * 4, h2 * w2).transpose(0, 2, 1)
        arr = np.ascontiguousarray(arr)
        return float32_array_to_buffer(
            arr,
            dtype=self._model_dtype,
            device=self._transformer_device,
        )
