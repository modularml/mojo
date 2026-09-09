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

"""Graph 3a: Transformer-only denoise compute component for Flux2Executor.

Used when TaylorSeer is enabled.  Returns the raw ``noise_pred`` so that
TaylorSeer can cache and predict it on skipped steps.  The Euler
scheduler step is handled separately by :class:`DenoisePredict`.
"""

from __future__ import annotations

from typing import Any

from max.driver import Buffer, load_devices
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.graph import Module as GraphModule
from max.graph.weights import load_weights
from max.nn.comm import Signals
from max.nn.layer import Module
from max.pipelines.lib.compiled_component import CompiledComponent
from max.pipelines.lib.config.model_config import (
    _resolve_component_encoding_and_weights,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.profiler import traced

from ..flux2 import Flux2Transformer2DModel
from ..model_config import Flux2Config
from ..weight_adapters import (
    adapt_weights,
    parse_nvfp4_quantization_metadata,
    verify_int8_quantization_consistency,
)


class DenoiseComputeStep(Module):
    """Concat + transformer forward + slice — no Euler step.

    Same as :class:`~.denoiser.DenoiseStep` but returns ``noise_pred``
    instead of updated latents.  The ``dt`` input is not needed.

    Input:  8 tensors (see :meth:`input_types`)
    Output: ``(B, image_seq_len, C)`` model dtype
    """

    def __init__(
        self,
        transformer: Flux2Transformer2DModel,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self._dtype = dtype
        self._device = device

    def __call__(
        self,
        latents: TensorValue,
        image_latents: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        guidance: TensorValue,
        latent_image_ids: TensorValue,
        image_latent_ids: TensorValue,
        txt_ids: TensorValue,
        *,
        signal_buffers: list[BufferValue] | None = None,
    ) -> TensorValue:
        # Concat image latents for img2img (no-op when img_seq=0).
        latents_concat = ops.concat([latents, image_latents], axis=1)
        latent_image_ids_concat = ops.concat(
            [latent_image_ids, image_latent_ids], axis=1
        )

        # Keep timestep and guidance as float32 here.  The transformer
        # multiplies by 1000.0 *before* casting to model dtype
        # (flux2.py:628-629), so leaving them in float32 avoids bf16
        # rounding on the raw sigma — matching the diffusers precision
        # path.

        # Transformer forward.
        (noise_pred,) = self.transformer(
            latents_concat,
            encoder_hidden_states,
            timestep,
            latent_image_ids_concat,
            txt_ids,
            guidance,
            signal_buffers=signal_buffers,
        )

        # Slice noise_pred to latents.shape[1] tokens (discard image
        # predictions in img2img case).
        num_tokens = ops.shape_to_tensor([latents.shape[1]])
        noise_pred_sliced = ops.slice_tensor(
            noise_pred,
            [
                slice(None),
                (slice(0, num_tokens), "num_tokens"),
                slice(None),
            ],
        )

        return noise_pred_sliced

    def input_types(self) -> tuple[TensorType, ...]:
        in_channels = self.transformer.in_channels
        joint_attention_dim = self.transformer.joint_attention_dim
        return (
            # latents: (B, image_seq_len, C)
            TensorType(
                self._dtype,
                shape=["batch", "image_seq_len", in_channels],
                device=self._device,
            ),
            # image_latents: (B, img_seq, C) — zero-seq for t2i
            TensorType(
                self._dtype,
                shape=["batch", "img_seq", in_channels],
                device=self._device,
            ),
            # encoder_hidden_states: (B, text_seq_len, joint_attention_dim)
            TensorType(
                self._dtype,
                shape=["batch", "text_seq_len", joint_attention_dim],
                device=self._device,
            ),
            # timestep: (B,) float32 — transformer casts after x1000
            TensorType(DType.float32, shape=["batch"], device=self._device),
            # guidance: (B,) float32 — transformer casts after x1000
            TensorType(DType.float32, shape=["batch"], device=self._device),
            # latent_image_ids: (B, image_seq_len, 4)
            TensorType(
                DType.int64,
                shape=["batch", "image_seq_len", 4],
                device=self._device,
            ),
            # image_latent_ids: (B, img_seq, 4) — zero-seq for t2i
            TensorType(
                DType.int64,
                shape=["batch", "img_seq", 4],
                device=self._device,
            ),
            # txt_ids: (B, text_seq_len, 4)
            TensorType(
                DType.int64,
                shape=["batch", "text_seq_len", 4],
                device=self._device,
            ),
        )


class DenoiseCompute(CompiledComponent):
    """Graph 3a: Concat + transformer forward (no Euler step).

    Used when TaylorSeer is enabled.  Returns ``noise_pred`` so it can
    be cached for Taylor prediction on skipped steps.

    Output shape: ``(B, image_seq_len, C)`` model dtype.
    """

    _model: Model
    _signal_buffers: list[Buffer]

    @traced(message="DenoiseCompute.__init__")
    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
        *,
        graphs_module: GraphModule | None = None,
    ) -> None:
        super().__init__(manifest, session, graphs_module=graphs_module)

        config = manifest["transformer"]
        config_dict = config.huggingface_config.to_dict()
        resolved_encoding, resolved_weight_path = (
            _resolve_component_encoding_and_weights(config)
        )
        encoding = resolved_encoding or "bfloat16"
        devices = load_devices(config.device_specs)

        transformer_config = Flux2Config.initialize_from_config(
            config_dict, encoding, devices
        )

        dtype = transformer_config.dtype
        device = transformer_config.devices[0]
        device_refs = transformer_config.devices

        # Load weights and adapt for NVFP4 / stacked-QKV checkpoints.
        paths = config.resolved_weight_paths(resolved_weight_path)
        weights = load_weights(paths)
        raw_state_dict = {key: value.data() for key, value in weights.items()}
        raw_state_dict = adapt_weights(
            raw_state_dict, transformer_config.quant_config
        )
        # ``weights`` caches every materialized source buffer in its shared
        # ``_st_weight_map`` and is unused past this point. Drop it so the
        # int8 W8A8 path (which quantizes the bf16 Linears away in
        # ``adapt_weights``) actually releases the original bf16 buffers rather
        # than keeping stale refs alongside the int8 weights.
        del weights

        nvfp4_layers_bfl = parse_nvfp4_quantization_metadata(paths)
        if nvfp4_layers_bfl:
            transformer_config = transformer_config.model_copy(
                update={"nvfp4_layers_bfl": nvfp4_layers_bfl}
            )

        has_guidance_embedder = any(
            "time_guidance_embed.guidance_embedder." in k
            for k in raw_state_dict
        )
        if not has_guidance_embedder and transformer_config.guidance_embeds:
            transformer_config = transformer_config.model_copy(
                update={"guidance_embeds": False}
            )

        # Build transformer and compute-only module.
        transformer = Flux2Transformer2DModel(transformer_config)
        compute = DenoiseComputeStep(
            transformer=transformer,
            dtype=dtype,
            device=device,
        )

        # Prefix with "transformer." for the module hierarchy.
        state_dict: dict[str, Any] = {
            f"transformer.{key}": value for key, value in raw_state_dict.items()
        }
        # For int8 W8A8, reconcile the RTN-quantized weights against the
        # model's int8 Linears so a whitelist/resolve drift fails here with a
        # named layer, not later as a cryptic dtype error in the matmul op.
        qc = transformer_config.quant_config
        if qc is not None and qc.is_int8_w8a8:
            verify_int8_quantization_consistency(compute, state_dict)
        compute.load_state_dict(state_dict, weight_alignment=1)

        # Build and compile graph. When running multi-device, append
        # ``Signals`` buffer types so the transformer's allreduces have
        # peer-to-peer scratch space; on a single device the graph is
        # unchanged from the pre-multi-device build.
        tensor_types = compute.input_types()
        input_types: list[TensorType | BufferType] = list(tensor_types)
        if len(device_refs) > 1:
            signals = Signals(devices=device_refs)
            input_types.extend(signals.input_types())
            self._signal_buffers = signals.buffers()
        else:
            self._signal_buffers = []

        with Graph(
            "denoise_compute",
            input_types=input_types,
            module=self._graphs_module,
        ) as graph:
            inputs = list(graph.inputs)
            tensor_inputs = inputs[: len(tensor_types)]
            buffer_inputs = inputs[len(tensor_types) :]
            outputs = compute(
                *(v.tensor for v in tensor_inputs),
                signal_buffers=[v.buffer for v in buffer_inputs],
            )
            graph.output(outputs)

        self._load_graph(graph, weights_registry=compute.state_dict())

    @traced(message="DenoiseCompute.__call__")
    def __call__(
        self,
        latents: Buffer,
        image_latents: Buffer,
        encoder_hidden_states: Buffer,
        timestep: Buffer,
        guidance: Buffer,
        latent_image_ids: Buffer,
        image_latent_ids: Buffer,
        txt_ids: Buffer,
    ) -> Buffer:
        """Execute the transformer-only compute step.

        Args:
            latents: Current latent state, shape ``(B, seq, C)``.
            image_latents: Packed image latents, shape ``(B, img_seq, C)``.
                Zero-seq-length for text-to-image.
            encoder_hidden_states: Text embeddings, shape
                ``(B, text_seq, D)``.
            timestep: Current sigma, shape ``(B,)``.
            guidance: Guidance scale, shape ``(B,)``.
            latent_image_ids: Latent position IDs, shape
                ``(B, seq, 4)`` int64.
            image_latent_ids: Image latent position IDs, shape
                ``(B, img_seq, 4)`` int64.
            txt_ids: Text position IDs, shape ``(B, text_seq, 4)`` int64.

        Returns:
            noise_pred, shape ``(B, seq, C)``.
        """
        result = self._model.execute(
            latents,
            image_latents,
            encoder_hidden_states,
            timestep,
            guidance,
            latent_image_ids,
            image_latent_ids,
            txt_ids,
            *self._signal_buffers,
        )
        return result[0] if isinstance(result, (list, tuple)) else result
