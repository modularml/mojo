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

"""Graph 2: VAE image encoder component for Flux2Executor."""

from __future__ import annotations

from typing import Any

import numpy as np
from max.driver import Buffer, Device, load_devices
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

from ...autoencoders.vae_flux2 import Encoder
from ...autoencoders_modulev3.model_config import AutoencoderKLFlux2Config


class PreprocessAndEncode(Module):
    """Fused preprocess + VAE encode + mode extraction + patchify + BN normalize + pack.

    Accepts a uint8 HWC image, preprocesses it in-graph (cast to float32,
    scale to [-1, 1], permute HWC->CHW, unsqueeze batch, cast to model
    dtype), then runs the VAE encoder, extracts the mean (mode),
    patchifies with 2x2 patches, applies BatchNorm normalization, and
    packs into sequence format for the transformer.

    Input:  ``(H, W, in_channels)`` uint8
    Output: ``(1, H/16 * W/16, C*4)`` model dtype
    """

    def __init__(
        self,
        encoder: Encoder,
        batch_norm_eps: float,
        num_channels: int,
        latent_channels: int,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self._batch_norm_eps = batch_norm_eps
        self._num_channels = num_channels
        self._latent_channels = latent_channels
        self._device = device
        self._dtype = dtype

        # Named with an ``encoder_`` prefix so the FQN doesn't collide with
        # ``PostprocessAndDecode``'s ``decoder_bn_*`` when both components
        # are compiled together via ``session.load_all`` (MODELS-1440).
        self.encoder_bn_mean = Weight(
            name="encoder_bn_mean",
            dtype=dtype,
            shape=[num_channels],
            device=DeviceRef.CPU(),
        )
        self.encoder_bn_var = Weight(
            name="encoder_bn_var",
            dtype=dtype,
            shape=[num_channels],
            device=DeviceRef.CPU(),
        )

    def __call__(self, image: TensorValue) -> TensorValue:
        # Preprocess: (H, W, C) uint8 -> (1, C, H, W) model-dtype [-1, 1].
        image = ops.cast(image, DType.float32)
        image = image / ops.constant(127.5, DType.float32, device=self._device)
        image = image - ops.constant(1.0, DType.float32, device=self._device)
        image = ops.permute(image, [2, 0, 1])  # HWC -> CHW
        image = ops.unsqueeze(image, 0)  # (1, C, H, W)
        image = ops.cast(image, self._dtype)

        # VAE encode -> (B, 2*C, H/8, W/8) mean|logvar concatenated.
        encoder_moments = self.encoder(image)

        batch = encoder_moments.shape[0]
        full_c = encoder_moments.shape[1]
        c = full_c // 2
        h = encoder_moments.shape[2]
        w = encoder_moments.shape[3]

        # Extract mode (first half of channels = mean).
        mean = encoder_moments[:, :c, :, :]

        # Patchify: (B, C, H, W) -> (B, C, H//2, 2, W//2, 2)
        mean = ops.rebind(mean, [batch, c, (h // 2) * 2, (w // 2) * 2])
        latents_6d = ops.reshape(mean, (batch, c, h // 2, 2, w // 2, 2))
        h2 = latents_6d.shape[2]
        w2 = latents_6d.shape[4]

        # (B, C, H', 2, W', 2) -> (B, C, 2, 2, H', W') -> (B, C*4, H', W')
        latents = ops.permute(latents_6d, [0, 1, 3, 5, 2, 4])
        latents = ops.reshape(latents, (batch, c * 4, h2, w2))

        # BN normalize: (latents - mean) / sqrt(var + eps)
        bn_mean = self.encoder_bn_mean.to(self._device)
        bn_var = self.encoder_bn_var.to(self._device)
        bn_mean_r = ops.reshape(bn_mean, (1, self._num_channels, 1, 1))
        bn_var_r = ops.reshape(bn_var, (1, self._num_channels, 1, 1))
        bn_std = ops.sqrt(
            bn_var_r
            + ops.constant(
                self._batch_norm_eps, self._dtype, device=self._device
            )
        )
        latents = (latents - bn_mean_r) / bn_std

        # Pack: (B, C*4, H', W') -> (B, H'*W', C*4)
        num_ch = latents.shape[1]
        latents = ops.reshape(latents, (batch, num_ch, h2 * w2))
        latents = ops.permute(latents, [0, 2, 1])

        return latents

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                DType.uint8,
                shape=[
                    "image_height",
                    "image_width",
                    self.encoder.in_channels,
                ],
                device=self._device,
            ),
        )


class ImageEncoder(CompiledComponent):
    """Graph 2: VAE encode + BN-normalize + patchify + pack.

    Encapsulates the full lifecycle: config extraction from the manifest,
    weight loading and key adaptation, Module construction, graph
    compilation, and runtime execution.

    Output shapes:
        - ``image_latents``: ``(1, img_seq, C*4)`` model dtype
        - ``image_latent_ids``: ``(1, img_seq, 4)`` int64
    """

    _model: Model
    _device: Device
    _vae_scale_factor: int
    _in_channels: int

    @traced(message="ImageEncoder.__init__")
    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
        *,
        graphs_module: GraphModule | None = None,
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
        self._device = devices[0]
        self._in_channels = vae_config.in_channels
        self._vae_scale_factor = 2 ** (len(vae_config.block_out_channels) - 1)

        # Build Graph API encoder.
        encoder = Encoder(
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
            device=device,
            dtype=dtype,
        )

        num_channels = vae_config.latent_channels * 4
        fused = PreprocessAndEncode(
            encoder=encoder,
            batch_norm_eps=vae_config.batch_norm_eps,
            num_channels=num_channels,
            latent_channels=vae_config.latent_channels,
            device=device,
            dtype=dtype,
        )

        # Load and adapt weights.
        paths = config.resolved_weight_paths(resolved_weight_path)
        weights = load_weights(paths)
        state_dict = self._adapt_state_dict(weights, fused.raw_state_dict())
        fused.load_state_dict(state_dict, weight_alignment=1)

        # Build and compile graph.
        with Graph(
            "image_encode",
            input_types=fused.input_types(),
            module=self._graphs_module,
        ) as graph:
            outputs = fused(*(v.tensor for v in graph.inputs))
            graph.output(outputs)

        self._load_graph(graph, weights_registry=fused.state_dict())

    @traced(message="ImageEncoder.__call__")
    def __call__(self, input_image: Buffer) -> tuple[Buffer, Buffer]:
        """Encode a raw image into packed latents and position IDs.

        Args:
            input_image: Raw input image, shape ``(H, W, C)`` uint8.

        Returns:
            A tuple of ``(image_latents, image_latent_ids)`` where:
            - ``image_latents`` has shape ``(1, img_seq, C*4)``
            - ``image_latent_ids`` has shape ``(1, img_seq, 4)`` int64
        """
        # Transfer uint8 image to device; preprocessing happens in-graph.
        image_buf = input_image.to(self._device)

        # Execute fused graph: preprocess + encode + mode + patchify + BN + pack.
        result = self._model.execute(image_buf)
        image_latents = (
            result[0] if isinstance(result, (list, tuple)) else result
        )

        # Derive spatial dims for image IDs.
        h_pixels = input_image.shape[0]
        w_pixels = input_image.shape[1]
        packed_h = h_pixels // (self._vae_scale_factor * 2)
        packed_w = w_pixels // (self._vae_scale_factor * 2)

        image_latent_ids = self._build_image_ids(packed_h, packed_w).to(
            self._device
        )

        return image_latents, image_latent_ids

    @staticmethod
    def _build_image_ids(
        packed_h: int,
        packed_w: int,
        scale: int = 10,
    ) -> Buffer:
        """Create 4D image position IDs in (T, H, W, L) format.

        For image patches: T=scale, H=arange(packed_h),
        W=arange(packed_w), L=0.

        Returns:
            Buffer of shape ``(1, packed_h * packed_w, 4)`` int64.
        """
        t_coords = np.full((packed_h, packed_w), scale, dtype=np.int64)
        h_coords, w_coords = np.meshgrid(
            np.arange(packed_h, dtype=np.int64),
            np.arange(packed_w, dtype=np.int64),
            indexing="ij",
        )
        l_coords = np.zeros((packed_h, packed_w), dtype=np.int64)

        coords = np.stack(
            [t_coords, h_coords, w_coords, l_coords], axis=-1
        )  # (packed_h, packed_w, 4)
        coords = coords.reshape(1, -1, 4)  # (1, seq, 4)
        return Buffer.from_dlpack(np.ascontiguousarray(coords))

    @staticmethod
    def _adapt_state_dict(
        weights: Weights,
        raw_state_dict: dict[str, Weight],
    ) -> dict[str, Any]:
        """Adapt HuggingFace VAE weights to the fused Module hierarchy.

        Key mapping:
        - ``encoder.*`` -> ``encoder.*``
        - ``quant_conv.*`` -> ``encoder.quant_conv.*``
        - ``bn.running_mean`` -> ``encoder_bn_mean``
        - ``bn.running_var`` -> ``encoder_bn_var``

        Casts each weight to the dtype expected by the corresponding
        Weight in the module's raw_state_dict (e.g. GroupNorm affine
        params stay float32 even when the model dtype is bfloat16).
        """
        state_dict: dict[str, Any] = {}
        for key, value in weights.items():
            weight_data = value.data()

            # Map checkpoint key to module key.
            if key.startswith("encoder."):
                adapted = key.removeprefix("encoder.")
                module_key = f"encoder.{adapted}"
            elif key.startswith("quant_conv."):
                module_key = f"encoder.{key}"
            elif key == "bn.running_mean":
                module_key = "encoder_bn_mean"
            elif key == "bn.running_var":
                module_key = "encoder_bn_var"
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
