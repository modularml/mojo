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

"""Graph 3a-FBC: First-Block-Cache denoise compute component for Flux2Executor.

Used when ``--first-block-caching`` is enabled.  Analogous to
:class:`~.denoise_compute.DenoiseCompute` (returns raw ``noise_pred``; the
Euler scheduler step stays in :class:`~.denoise_predict.DenoisePredict`), but
inserts a First-Block-Cache (FBCache) conditional inside the transformer:

1. Run the preamble + double-stream block 0.
2. Compute the block-0 image-stream residual and compare its relative diff
   against the previous step's residual (passed in as a graph input).
3. Via ``ops.cond``:
   - **skip** (residual similar): reuse ``prev_output`` — the remaining
     blocks + postamble are not executed.
   - **compute** (residual changed): run blocks 1..N + single-stream blocks
     + postamble to produce a fresh ``noise_pred``.

The graph returns ``(new_residual, noise_pred)``.  The host loop threads
``new_residual``/``noise_pred`` into the next step's ``prev_residual``/
``prev_output`` (mirroring the generic ``run_denoising_step`` in
``diffusion/taylorseer.py``).  The relative-diff formula matches
``diffusion.cache._can_use_fbcache`` exactly; the predicate is transferred to
CPU for ``ops.cond`` (which requires a CPU predicate).
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
)

# Matches diffusion.cache._can_use_fbcache.
_FBCACHE_EPS: float = 1e-9


class DenoiseComputeFBCacheStep(Module):
    """Concat + transformer with a First-Block-Cache conditional.

    Same inputs as :class:`~.denoise_compute.DenoiseComputeStep` plus three
    FBCache tensors (``prev_residual``, ``prev_output``, ``residual_threshold``)
    and two outputs (``new_residual``, ``noise_pred``).
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

    def _relative_diff_lt_threshold(
        self,
        residual: TensorValue,
        prev_residual: TensorValue,
        residual_threshold: TensorValue,
    ) -> TensorValue:
        """Return a scalar bool: relative L1 diff < threshold.

        Mirrors ``diffusion.cache._can_use_fbcache``: mean over the last
        axis, then mean over all rows, then ``mean_diff / (mean_prev + eps)``.
        """
        # mean over last axis -> (B, seq, 1), then over remaining -> scalar.
        diff = ops.abs(residual - prev_residual)
        mean_diff_rows = ops.mean(diff, axis=-1)  # (B, seq, 1)
        mean_prev_rows = ops.mean(ops.abs(prev_residual), axis=-1)
        # Flatten and reduce to a single scalar via mean over axis 0 of the
        # flattened tensor.
        mean_diff = ops.mean(mean_diff_rows.reshape([-1]), axis=0)  # (1,)
        mean_prev = ops.mean(mean_prev_rows.reshape([-1]), axis=0)  # (1,)
        eps = ops.constant(
            _FBCACHE_EPS, mean_prev.dtype, device=mean_prev.device
        )
        relative_diff = mean_diff / (mean_prev + eps)
        rdt = residual_threshold.cast(relative_diff.dtype)
        pred = relative_diff < rdt  # (1,) bool
        return ops.squeeze(pred, 0)  # scalar bool

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
        prev_residual: TensorValue,
        prev_output: TensorValue,
        residual_threshold: TensorValue,
        *,
        signal_buffers: list[BufferValue] | None = None,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        # Concat image latents for img2img (no-op when img_seq=0).
        latents_concat = ops.concat([latents, image_latents], axis=1)
        latent_image_ids_concat = ops.concat(
            [latent_image_ids, image_latent_ids], axis=1
        )

        # Preamble + block 0.
        preamble = self.transformer.forward_preamble(
            latents_concat,
            encoder_hidden_states,
            timestep,
            latent_image_ids_concat,
            guidance,
            txt_ids,
            signal_buffers=signal_buffers,
        )
        # Image stream before block 0 (device 0), for the residual delta.
        img_before = preamble.hidden_states_d[0]
        state_after_first = self.transformer.run_first_block(preamble)
        img_after = state_after_first[1][0]  # hidden_states_d[0] after block 0

        # Number of image tokens to keep from the (image, text) sequence.
        # ``latents.shape[1]`` is the original latent seq (excludes any img2img
        # image_latents concatenated above), matching the sliced noise_pred
        # and the ``prev_residual`` graph-input shape ``[B, image_seq_len, D]``.
        num_tokens = ops.shape_to_tensor([latents.shape[1]])

        def _slice_seq(x: TensorValue) -> TensorValue:
            return ops.slice_tensor(
                x,
                [
                    slice(None),
                    (slice(0, num_tokens), "num_tokens"),
                    slice(None),
                ],
            )

        # FBCache residual: block-0 delta on the image tokens only.  The
        # slice labels the seq dim ``num_tokens``; rebind it to the
        # ``prev_residual`` seq dim (``image_seq_len``) since they are equal
        # at runtime (``latents.shape[1] == image_seq_len`` for t2i, and the
        # host allocates prev_residual at exactly ``latents.shape[1]``).
        first_block_residual = ops.rebind(
            _slice_seq(img_after) - _slice_seq(img_before),
            prev_residual.shape,
        )

        use_fbcache = self._relative_diff_lt_threshold(
            first_block_residual, prev_residual, residual_threshold
        )
        # ops.cond requires the predicate on CPU.
        use_fbcache_cpu = ops.transfer_to(use_fbcache, DeviceRef.CPU())

        def then_fn() -> tuple[TensorValue, TensorValue]:
            # Skip remaining blocks + postamble: reuse prev_output.
            return (first_block_residual, prev_output)

        def else_fn() -> tuple[TensorValue, TensorValue]:
            hidden_states_d = self.transformer.run_remaining_blocks(
                preamble, state_after_first
            )
            noise_pred = self.transformer.forward_postamble(
                preamble, hidden_states_d
            )
            # Rebind the sliced seq dim to ``prev_output``'s so both cond
            # branches yield the same output type.
            noise_pred = ops.rebind(_slice_seq(noise_pred), prev_output.shape)
            return (first_block_residual, noise_pred)

        residual_type = TensorType(
            first_block_residual.dtype,
            shape=first_block_residual.shape,
            device=first_block_residual.device,
        )
        output_type = TensorType(
            prev_output.dtype,
            shape=prev_output.shape,
            device=prev_output.device,
        )
        result = ops.cond(
            use_fbcache_cpu,
            [residual_type, output_type],
            then_fn,
            else_fn,
        )
        # Also surface the (CPU) skip predicate so the host loop can count
        # how many steps actually reused the cache without an extra device
        # read-back of the residuals.  True == the remaining blocks were
        # skipped this step.
        return (result[0], result[1], use_fbcache_cpu)

    def input_types(self) -> tuple[TensorType, ...]:
        in_channels = self.transformer.in_channels
        joint_attention_dim = self.transformer.joint_attention_dim
        inner_dim = self.transformer.inner_dim
        out_channels = self.transformer.out_channels
        patch_size = self.transformer.patch_size
        output_dim = patch_size * patch_size * out_channels
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
            # prev_residual: (B, image_seq_len, inner_dim) — block-0 residual
            TensorType(
                self._dtype,
                shape=["batch", "image_seq_len", inner_dim],
                device=self._device,
            ),
            # prev_output: (B, image_seq_len, output_dim) — cached noise_pred
            TensorType(
                self._dtype,
                shape=["batch", "image_seq_len", output_dim],
                device=self._device,
            ),
            # residual_threshold: scalar float32 (runtime-tunable, no recompile)
            TensorType(DType.float32, shape=[], device=self._device),
        )


class DenoiseComputeFBCache(CompiledComponent):
    """Graph 3a-FBC: transformer with a First-Block-Cache conditional.

    Returns ``(new_residual, noise_pred)``.  The Euler step is handled by
    :class:`~.denoise_predict.DenoisePredict`, exactly as in the TaylorSeer
    split path.
    """

    _model: Model
    _signal_buffers: list[Buffer]

    @traced(message="DenoiseComputeFBCache.__init__")
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

        # Build transformer and FBCache compute module.
        transformer = Flux2Transformer2DModel(transformer_config)
        compute = DenoiseComputeFBCacheStep(
            transformer=transformer,
            dtype=dtype,
            device=device,
        )

        # Prefix with "transformer." for the module hierarchy.
        state_dict: dict[str, Any] = {
            f"transformer.{key}": value for key, value in raw_state_dict.items()
        }
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
            "denoise_compute_fbcache",
            input_types=input_types,
            module=self._graphs_module,
        ) as graph:
            inputs = list(graph.inputs)
            tensor_inputs = inputs[: len(tensor_types)]
            buffer_inputs = inputs[len(tensor_types) :]
            new_residual, noise_pred, used_cache = compute(
                *(v.tensor for v in tensor_inputs),
                signal_buffers=[v.buffer for v in buffer_inputs],
            )
            graph.output(new_residual, noise_pred, used_cache)

        self._load_graph(graph, weights_registry=compute.state_dict())

    @traced(message="DenoiseComputeFBCache.__call__")
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
        prev_residual: Buffer,
        prev_output: Buffer,
        residual_threshold: Buffer,
    ) -> tuple[Buffer, Buffer, Buffer]:
        """Execute the FBCache transformer step.

        Returns:
            ``(new_residual, noise_pred, used_cache)`` where ``new_residual``
            is the block-0 image-stream residual for this step, ``noise_pred``
            is either freshly computed or the reused ``prev_output``, and
            ``used_cache`` is a scalar CPU bool that is ``True`` when the
            remaining transformer blocks were skipped (cache reused).
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
            prev_residual,
            prev_output,
            residual_threshold,
            *self._signal_buffers,
        )
        return (result[0], result[1], result[2])
