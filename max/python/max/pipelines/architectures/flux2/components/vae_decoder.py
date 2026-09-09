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

"""Graph 4: VAE decoder component for Flux2Executor."""

from __future__ import annotations

from typing import Any

import numpy as np
from max.driver import Buffer, load_devices
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight, ops
from max.graph import Module as GraphModule
from max.graph.weights import Weights, load_weights
from max.nn.layer import Module
from max.pipelines.lib.compiled_component import CompiledComponent
from max.pipelines.lib.config.model_config import (
    _resolve_component_encoding_and_weights,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.profiler import traced

from ...autoencoders.vae_flux2 import Decoder
from ...autoencoders_modulev3.model_config import AutoencoderKLFlux2Config


class PostprocessAndDecode(Module):
    """Fused BN-denorm + unpatchify + VAE decode in a single compiled graph.

    Accepts packed latents in ``(B, S, C)`` shape where ``S = latent_h * latent_w``.
    Spatial dimensions are conveyed via two 1-D shape-carrier tensors whose
    *lengths* encode ``latent_h`` and ``latent_w`` as symbolic graph Dims,
    so a single compiled graph handles any spatial size without recompilation.

    Input:  ``(B, S, C)`` model dtype
    Output: ``(B, H_pixels, W_pixels, 3)`` uint8 on CPU
    """

    def __init__(
        self,
        decoder: Decoder,
        batch_norm_eps: float,
        num_channels: int,
        device: DeviceRef,
        dtype: DType,
        latents_in_dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self._batch_norm_eps = batch_norm_eps
        self._num_channels = num_channels
        self._device = device
        self._dtype = dtype
        # Dtype of the incoming latents.  In a uniform-precision pipeline this
        # equals the VAE compute dtype, but a mixed-precision assembly (e.g. an
        # NVFP4 transformer whose compute dtype is bfloat16 feeding an
        # float32 VAE) produces latents in a different dtype than the VAE
        # expects.  We accept latents at this dtype and cast to ``dtype`` as
        # the first graph op; the cast is a no-op when they match.
        self._latents_in_dtype = latents_in_dtype or dtype

        # Named with a ``decoder_`` prefix so the FQN doesn't collide with
        # ``PreprocessAndEncode``'s ``encoder_bn_*`` when both components
        # are compiled together via ``session.load_all`` (MODELS-1440).
        self.decoder_bn_mean = Weight(
            name="decoder_bn_mean",
            dtype=dtype,
            shape=[num_channels],
            device=DeviceRef.CPU(),
        )
        self.decoder_bn_var = Weight(
            name="decoder_bn_var",
            dtype=dtype,
            shape=[num_channels],
            device=DeviceRef.CPU(),
        )

    def __call__(
        self,
        latents_bsc: TensorValue,
        h_carrier: TensorValue,
        w_carrier: TensorValue,
    ) -> TensorValue:
        # Reconcile the incoming latents dtype with the VAE compute dtype.
        # No-op when they already match (uniform-precision pipeline); widens
        # bfloat16 -> float32 for a mixed-precision assembly (NVFP4 transformer
        # feeding a float32 VAE).
        if latents_bsc.dtype != self._dtype:
            latents_bsc = ops.cast(latents_bsc, self._dtype)

        batch = latents_bsc.shape[0]
        c = latents_bsc.shape[2]

        # Extract spatial dims from carrier shapes (symbolic Dims).
        h = h_carrier.shape[0]
        w = w_carrier.shape[0]

        # Reshape packed (B, S, C) -> spatial (B, H, W, C).
        latents_bsc = ops.rebind(latents_bsc, [batch, h * w, c])
        latents_bhwc = ops.reshape(latents_bsc, (batch, h, w, c))

        # (B, H, W, C) -> (B, C, H, W)
        latents = ops.permute(latents_bhwc, [0, 3, 1, 2])

        # BN denormalization: x = x * sqrt(var + eps) + mean
        bn_mean = self.decoder_bn_mean.to(self._device)
        bn_var = self.decoder_bn_var.to(self._device)
        bn_mean_r = ops.reshape(bn_mean, (1, self._num_channels, 1, 1))
        bn_var_r = ops.reshape(bn_var, (1, self._num_channels, 1, 1))
        bn_std = ops.sqrt(
            bn_var_r
            + ops.constant(
                self._batch_norm_eps, self._dtype, device=self._device
            )
        )
        latents = latents * bn_std + bn_mean_r

        # Unpatchify: (B, C, H, W) -> (B, C//4, H*2, W*2)
        latents = ops.reshape(latents, (batch, c // 4, 2, 2, h, w))
        latents = ops.permute(latents, [0, 1, 4, 2, 5, 3])
        latents = ops.reshape(latents, (batch, c // 4, h * 2, w * 2))

        # VAE decode.
        decoded = self.decoder(latents, None)

        # (B, C, H, W) -> (B, H, W, C)
        decoded = ops.permute(decoded, [0, 2, 3, 1])

        # Normalize [-1, 1] -> [0, 255] and cast to uint8.
        # Upcast to float32 for the normalization + scaling so that the
        # x255 multiplication doesn't amplify bfloat16 rounding errors
        # into +-1-2 pixel differences.  This matches the precision path
        # used by diffusers (AutoencoderKL outputs float32 by default).
        # Round before the uint8 cast so the truncating cast doesn't bias
        # every pixel down by ~0.5; diffusers' image processor does
        # `(x * 255).round().astype(uint8)`.
        decoded = ops.cast(decoded, DType.float32)
        decoded = decoded * 0.5 + 0.5
        decoded = ops.max(decoded, 0.0)
        decoded = ops.min(decoded, 1.0)
        decoded = decoded * 255.0
        decoded = ops.round(decoded)

        return ops.transfer_to(ops.cast(decoded, DType.uint8), DeviceRef.CPU())

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self._latents_in_dtype,
                shape=["batch", "seq", self._num_channels],
                device=self._device,
            ),
            # Shape carriers: lengths encode latent_h / latent_w.
            TensorType(
                DType.float32, shape=["latent_h"], device=DeviceRef.CPU()
            ),
            TensorType(
                DType.float32, shape=["latent_w"], device=DeviceRef.CPU()
            ),
        )


class VaeDecoder(CompiledComponent):
    """Graph 4: BN-denormalize + unpatchify + VAE decode -> uint8 images.

    Encapsulates the full lifecycle: config extraction from the manifest,
    weight loading and key adaptation, Module construction, graph
    compilation, and runtime execution.

    Output shape: ``(B, H_pixels, W_pixels, 3)`` uint8 on CPU.
    """

    _model: Model

    @traced(message="VaeDecoder.__init__")
    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
        *,
        graphs_module: GraphModule | None = None,
        latents_in_dtype: DType | None = None,
    ) -> None:
        super().__init__(manifest, session, graphs_module=graphs_module)

        config = manifest["vae"]
        config_dict = config.huggingface_config.to_dict()
        resolved_encoding, resolved_weight_path = (
            _resolve_component_encoding_and_weights(config)
        )
        encoding = resolved_encoding or "bfloat16"
        devices = load_devices(config.device_specs)

        vae_config = AutoencoderKLFlux2Config.generate(
            config_dict, encoding, devices
        )

        dtype = vae_config.dtype
        device = vae_config.device

        # Build Graph API Decoder.
        decoder = Decoder(
            in_channels=vae_config.latent_channels,
            out_channels=vae_config.out_channels,
            up_block_types=tuple(vae_config.up_block_types),
            block_out_channels=tuple(vae_config.block_out_channels),
            layers_per_block=vae_config.layers_per_block,
            norm_num_groups=vae_config.norm_num_groups,
            act_fn=vae_config.act_fn,
            norm_type="group",
            mid_block_add_attention=vae_config.mid_block_add_attention,
            use_post_quant_conv=vae_config.use_post_quant_conv,
            device=device,
            dtype=dtype,
        )

        num_channels = vae_config.latent_channels * 4
        fused = PostprocessAndDecode(
            decoder=decoder,
            batch_norm_eps=vae_config.batch_norm_eps,
            num_channels=num_channels,
            device=device,
            dtype=dtype,
            latents_in_dtype=latents_in_dtype,
        )

        # Load and adapt weights.
        paths = config.resolved_weight_paths(resolved_weight_path)
        weights = load_weights(paths)
        state_dict = self._adapt_state_dict(weights, fused.raw_state_dict())

        # Validate BN stats are present and non-trivial.  Missing or
        # all-zero stats collapse the BN denormalization to near-zero,
        # producing washed-out images.
        for bn_key in ("decoder_bn_mean", "decoder_bn_var"):
            if bn_key not in state_dict:
                raise ValueError(
                    f"VaeDecoder: BatchNorm stat {bn_key!r} not found in "
                    "VAE weights.  The checkpoint must contain "
                    "'bn.running_mean' and 'bn.running_var'."
                )
            bn_data = state_dict[bn_key]
            # np.from_dlpack doesn't support bfloat16, so cast to float32
            # before validation.
            if hasattr(bn_data, "astype"):
                bn_arr = np.from_dlpack(bn_data.astype(DType.float32))
            else:
                bn_arr = np.from_dlpack(bn_data)
            if np.all(bn_arr == 0):
                raise ValueError(
                    f"VaeDecoder: {bn_key!r} is all zeros — BN "
                    "denormalization will collapse output to near-zero.  "
                    "Check that the correct VAE weights are loaded."
                )

        fused.load_state_dict(state_dict, weight_alignment=1)

        # Build and compile graph.
        with Graph(
            "vae_decode",
            input_types=fused.input_types(),
            module=self._graphs_module,
        ) as graph:
            outputs = fused(*(v.tensor for v in graph.inputs))
            graph.output(outputs)

        self._load_graph(graph, weights_registry=fused.state_dict())

    @traced(message="VaeDecoder.__call__")
    def __call__(
        self,
        latents: Buffer,
        h_carrier: Buffer,
        w_carrier: Buffer,
    ) -> Buffer:
        """Decode denoised packed latents into a uint8 image tensor.

        Args:
            latents: Denoised packed latents, shape ``(B, seq, C*4)``.
            h_carrier: Shape carrier of length ``packed_h``.
            w_carrier: Shape carrier of length ``packed_w``.

        Returns:
            Decoded images, shape ``(B, H, W, C)`` uint8 on CPU.
        """
        result = self._model.execute(latents, h_carrier, w_carrier)
        return result[0] if isinstance(result, (list, tuple)) else result

    @staticmethod
    def _adapt_state_dict(
        weights: Weights,
        raw_state_dict: dict[str, Weight],
    ) -> dict[str, Any]:
        """Adapt HuggingFace VAE weights to the fused Module hierarchy.

        Key mapping:
        - ``decoder.*`` -> ``decoder.*``
        - ``post_quant_conv.*`` -> ``decoder.post_quant_conv.*``
        - ``bn.running_mean`` -> ``decoder_bn_mean``
        - ``bn.running_var`` -> ``decoder_bn_var``

        Casts each weight to the dtype expected by the corresponding
        Weight in the module's raw_state_dict.
        """
        state_dict: dict[str, Any] = {}
        for key, value in weights.items():
            weight_data = value.data()

            # Map checkpoint key to module key.
            if key.startswith("decoder."):
                module_key = key
            elif key.startswith("post_quant_conv."):
                module_key = f"decoder.{key}"
            elif key == "bn.running_mean":
                module_key = "decoder_bn_mean"
            elif key == "bn.running_var":
                module_key = "decoder_bn_var"
            else:
                continue

            # Cast to the dtype the module Weight expects.
            target_weight = raw_state_dict.get(module_key)
            if target_weight is not None:
                target_dtype = target_weight.dtype
                if weight_data.dtype != target_dtype:
                    if weight_data.dtype.is_float() and target_dtype.is_float():
                        weight_data = weight_data.astype(target_dtype)

            state_dict[module_key] = weight_data

        return state_dict
