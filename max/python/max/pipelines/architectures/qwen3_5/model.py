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

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from max.driver import Buffer, Device, DLPackArray, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    Module,
    TensorType,
    TensorValue,
)
from max.graph.buffer_utils import cast_tensors_to
from max.nn.comm import Signals
from max.pipelines.context import ImageMetadata
from max.pipelines.lib import (
    CompilationTimer,
    ModelInputs,
    ModelOutputs,
)
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.pipelines.modeling.types import RequestID
from max.profiler import traced

from ..llama3.model import Llama3Inputs, LlamaModelBase
from ..qwen3vl_moe.context import Qwen3VLTextAndVisionContext
from .batch_processor import Qwen3_5BatchProcessor
from .model_config import Qwen3_5Config
from .qwen3_5 import Qwen3_5
from .state_cache import GatedDeltaNetStateCache
from .vision_packing import Qwen3_5VisionInputs, pack_uncached_images

logger = logging.getLogger("max.pipelines")


@dataclass
class Qwen3_5Inputs(Llama3Inputs):
    """Inputs for Qwen3.5, including the linear-attention state pools.

    Image embeddings come from the pipeline-driven encoder cache on the base
    ``vision_embeddings`` / ``vision_scatter_indices`` fields.
    """

    slot_idx: list[Buffer] | None = None
    """Per-device ``[B]`` uint32 slot indices into the linear-attention pools."""

    conv_pools: list[Buffer] | None = None
    """Device-major mutable conv pools, ``[max_slots, conv_dim, K-1]``."""

    recurrent_pools: list[Buffer] | None = None
    """Device-major mutable recurrent pools, ``[max_slots, nv, KD, VD]``."""

    request_ids: list[RequestID] | None = None
    """Request IDs for this batch, used to manage per-request state cache slots."""

    decoder_position_ids: Buffer | None = None
    """``[3, total_seq_len]`` M-RoPE positions, one column per active token.

    Present only when the graph was built with M-RoPE wired in; see
    ``Qwen3_5.mrope_enabled``."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        slot_idx_inputs: tuple[Buffer, ...] = tuple(self.slot_idx or ())
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
            *slot_idx_inputs,
            *(self.conv_pools or ()),
            *(self.recurrent_pools or ()),
            # Set by the pipeline's vision seam (``finalize_vision_inputs``)
            # on every prepared batch, empties included.
            *self.vision_embeddings,
            *self.vision_scatter_indices,
            *(
                ()
                if self.decoder_position_ids is None
                else (self.decoder_position_ids,)
            ),
        )


# Scale tensors carry the calibration a quantized checkpoint cannot be read
# without. A dropped one is not a degradation, it is a different model.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# Everything the architecture deliberately does not load. `mtp.*` is the
# speculative-decoding head, which the weight adapter drops.
_UNUSED_PREFIXES = ("mtp.",)


def _check_weights_match(expected: set[str], provided: set[str]) -> None:
    """Fails the load when the checkpoint and the graph disagree on weights.

    ``load_state_dict(strict=False)`` drops both directions of mismatch
    without a word, so a quantized checkpoint whose 995 scale tensors go
    unconsumed loads clean and emits garbage. This is the gate that turns
    that into a startup error.

    Args:
        expected: Weight names the built graph will look up.
        provided: Weight names the adapted checkpoint supplies.

    Raises:
        ValueError: If a weight the graph needs is absent, or a scale tensor
            the checkpoint supplies is not consumed.
    """
    missing = sorted(expected - provided)
    if missing:
        raise ValueError(
            f"Qwen3.5 checkpoint is missing {len(missing)} weight(s) the model "
            f"requires: {missing[:20]}"
            + (f" (+{len(missing) - 20} more)" if len(missing) > 20 else "")
        )

    unused = provided - expected
    unconsumed_scales = sorted(
        k
        for k in unused
        if k.endswith(_SCALE_SUFFIXES) and not k.startswith(_UNUSED_PREFIXES)
    )
    if unconsumed_scales:
        raise ValueError(
            f"Qwen3.5 checkpoint supplies {len(unconsumed_scales)} "
            "quantization scale tensor(s) that no layer consumes, so those "
            "weights would be read at the wrong precision: "
            f"{unconsumed_scales[:20]}"
            + (
                f" (+{len(unconsumed_scales) - 20} more)"
                if len(unconsumed_scales) > 20
                else ""
            )
        )

    if unused:
        logger.info(
            "Qwen3.5 load_state_dict: %d unused checkpoint keys: %s",
            len(unused),
            sorted(unused)[:20],
        )


class Qwen3_5Model(AlwaysSignalBuffersMixin, LlamaModelBase):
    """Qwen3.5 pipeline model implementation.

    Supports the hybrid linear/full attention architecture with KV cache
    for full attention layers and conv/recurrent states for linear layers.
    """

    model_config_cls: ClassVar[type[Any]] = Qwen3_5Config
    batch_processor_cls: ClassVar[type[Qwen3_5BatchProcessor]] = (
        Qwen3_5BatchProcessor
    )

    model: Model
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    attention_bias: bool = False
    state_dict: dict[str, Any]

    # Vision model (None for text-only checkpoints)
    vision_model: Model | None = None
    _vision_state_dict: dict[str, DLPackArray] | None = None
    _nn_model: Any = None
    _session: InferenceSession | None = None

    # Model dtype and hidden size (set during graph build, used for empty buffers)
    _hidden_size: int = 0
    _model_dtype: DType = DType.bfloat16
    # State-pool storage dtype. Deliberately separate from _model_dtype: the
    # vision empties keep the compute dtype even when the pools are fp32.
    _state_dtype: DType = DType.bfloat16
    # Per-request state bytes the config budgeted, checked against what the
    # pools actually allocate. See the assertion at the state-cache build.
    _accounted_state_bytes: int = 0

    # Linear attention state dimensions (set during graph build)
    _num_linear_layers: int = 0
    _conv_dim: int = 0
    _conv_kernel_size: int = 0
    _num_v_heads: int = 0
    _key_head_dim: int = 0
    _value_head_dim: int = 0

    # Whether the built graph takes M-RoPE positions; set during graph build.
    _mrope_enabled: bool = False

    # Per-request state cache for the linear-attention pools.
    _state_cache: GatedDeltaNetStateCache | None = None

    # Zero-row vision embeddings for decode / text-only steps, so buffers()
    # always has the right input count. Cached: see empty_vision_embeddings.
    _empty_vision_embeddings: list[Buffer] | None = None

    @traced
    def load_model(self, session: InferenceSession) -> Model:
        self._session = session

        self._input_row_offsets_prealloc: Buffer | None = None
        self._slot_idx_prealloc: list[Buffer] | None = None
        max_batch_size = self.max_batch_size
        assert max_batch_size is not None, (
            "max_batch_size must be set in runtime config"
        )
        if not is_virtual_device_mode():
            self._input_row_offsets_prealloc = Buffer.from_numpy(
                np.arange(
                    max_batch_size + 1,
                    dtype=np.uint32,
                )
            ).to(self.devices[0])

        with CompilationTimer("model") as timer:
            module = Module()
            state_dict = self._load_state_dict()
            language_graph = self._build_language_graph(
                state_dict, module=module
            )
            assert self._vision_state_dict is not None
            vision_graph = self._build_vision_graph(module=module)
            timer.mark_build_complete()
            models = session.load_all(
                module,
                weights_registry={
                    **self.state_dict,
                    **self._vision_state_dict,
                },
            )
            model = models[language_graph.name]
            self.vision_model = models[vision_graph.name]

        # Initialize per-request state cache for linear attention layers.
        # _num_linear_layers is populated by _build_graph, so this and the
        # slot-idx prealloc must run after it.
        if self._num_linear_layers > 0 and not is_virtual_device_mode():
            # The value heads are split across devices, so the recorded
            # dimensions are already per-device shard widths.
            self._state_cache = GatedDeltaNetStateCache(
                num_layers=self._num_linear_layers,
                conv_dim=self._conv_dim,
                conv_kernel_size=self._conv_kernel_size,
                num_v_heads=self._num_v_heads,
                key_head_dim=self._key_head_dim,
                value_head_dim=self._value_head_dim,
                max_slots=max_batch_size,
                devices=self.devices,
                dtype=self._state_dtype,
            )
            # Memory planning sizes the pools from `Qwen3_5Config.state_dtype`
            # and the unsharded geometry; the cache is built from per-device
            # shard widths. Every device holds one shard, so the two must
            # reconcile exactly. They have diverged before -- by reading the
            # encoding's storage dtype instead of the pool dtype -- and the
            # symptom was an inflated batch size that OOMed at load, far from
            # the cause.
            allocated = self._state_cache.bytes_per_slot * len(self.devices)
            assert allocated == self._accounted_state_bytes, (
                "Qwen3.5 state pools allocate "
                f"{allocated} B per request but memory planning budgeted "
                f"{self._accounted_state_bytes} B. The pool dtype "
                f"({self._state_dtype}) and the accounted dtype must agree."
            )
            self._slot_idx_prealloc = [
                Buffer(
                    shape=[max_batch_size],
                    dtype=DType.uint32,
                    device=device,
                )
                for device in self.devices
            ]

        if (
            self._batch_processor is not None
            and self._state_cache is not None
            and self._slot_idx_prealloc is not None
        ):
            bind = getattr(self._batch_processor, "bind_prepare_state", None)
            if bind is not None:
                bind(
                    state_cache=self._state_cache,
                    slot_idx_prealloc=self._slot_idx_prealloc,
                    mrope_enabled=self._mrope_enabled,
                )

        return model

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[Qwen3VLTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> Qwen3_5VisionInputs | None:
        """Pack the pipeline-selected cache-miss images to device.

        Runs in the pipeline's prep-ahead window so the host-to-device copy
        overlaps the prior batch.
        """
        return pack_uncached_images(selection, devices)

    def vision_execute(
        self,
        selection: Sequence[
            tuple[Qwen3VLTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: Qwen3_5VisionInputs | None,
    ) -> VisionEncodeResult:
        """Run the vision encoder on the images ``pack_vision_inputs`` packed.

        Returns embeddings only; the pipeline derives per-image token counts
        from its selection, which match because the tokenizer emits exactly
        one placeholder per merged patch.
        """
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        assert self.vision_model is not None
        assert self._session is not None
        vision_outputs = self.vision_model.execute(
            packed.pixel_values,
            packed.weights,
            packed.indices,
            packed.vision_position_ids,
            packed.max_grid_size,
            packed.grid_thw,
            packed.cu_seqlens,
            packed.max_seqlen,
            *self.signal_buffers,
        )
        assert isinstance(vision_outputs[0], Buffer)
        embeddings = cast_tensors_to(
            [vision_outputs[0]], self._model_dtype, self._session
        )[0]
        # The hidden state is replicated across devices, so every replica
        # merges the same embeddings.
        return VisionEncodeResult(
            embeddings=[embeddings.to(device) for device in devices]
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for cached / text-only batches.

        Cached: hit on every text-only / decode step, so it must not allocate
        per call, and graph-capture replay only skips an input refresh for an
        identical buffer object.
        """
        if self._empty_vision_embeddings is None:
            self._empty_vision_embeddings = [
                Buffer.zeros(
                    shape=[0, self._hidden_size], dtype=self._model_dtype
                ).to(device)
                for device in devices
            ]
        return self._empty_vision_embeddings

    def _build_vision_graph(self, module: Module) -> Graph:
        """Build the vision encoder graph for processing images."""
        assert isinstance(self._nn_model, Qwen3_5), (
            "_build_vision_graph called before _build_graph"
        )
        vision_encoder = self._nn_model.vision_encoder
        assert vision_encoder is not None, (
            "_build_vision_graph called but no vision encoder"
        )

        patch_dim = vision_encoder.patch_embed.patch_dim

        # Input types - one per device (currently single-device only; see arch.py)
        pixel_values_types = [
            TensorType(
                DType.float32,
                shape=["vision_seq_len", patch_dim],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]
        weights_types = [
            TensorType(
                DType.float32,
                shape=[4, "vision_seq_len", 1],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]
        indices_types = [
            TensorType(
                DType.int64,
                shape=[4, "vision_seq_len"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]
        rot_pos_ids_types = [
            TensorType(
                DType.int32,
                shape=["vision_seq_len", 2],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]
        max_grid_size_types = [
            TensorType(DType.int32, shape=[], device=DeviceRef.CPU())
            for _ in self.devices
        ]
        grid_thw_types = [
            TensorType(
                DType.int64,
                shape=["n_images", 3],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]
        cu_seqlens_types = [
            TensorType(
                DType.uint32,
                shape=["n_seqlens"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]
        max_seqlen_types = [
            TensorType(DType.uint32, shape=[1], device=DeviceRef.CPU())
            for _ in self.devices
        ]

        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        with Graph(
            "qwen3_5_vision",
            input_types=(
                *pixel_values_types,
                *weights_types,
                *indices_types,
                *rot_pos_ids_types,
                *max_grid_size_types,
                *grid_thw_types,
                *cu_seqlens_types,
                *max_seqlen_types,
                *signals.input_types(),
            ),
            module=module,
        ) as graph:
            all_inputs = graph.inputs
            n = len(self.devices)

            pixel_values_list = [inp.tensor for inp in all_inputs[:n]]
            weights_list = [inp.tensor for inp in all_inputs[n : 2 * n]]
            indices_list = [inp.tensor for inp in all_inputs[2 * n : 3 * n]]
            rot_pos_ids_list = [inp.tensor for inp in all_inputs[3 * n : 4 * n]]
            max_grid_size_list = [
                inp.tensor for inp in all_inputs[4 * n : 5 * n]
            ]
            grid_thw_list = [inp.tensor for inp in all_inputs[5 * n : 6 * n]]
            cu_seqlens_list = [inp.tensor for inp in all_inputs[6 * n : 7 * n]]
            max_seqlen_list = [inp.tensor for inp in all_inputs[7 * n : 8 * n]]
            signal_buffers = [inp.buffer for inp in all_inputs[8 * n :]]

            # Qwen3.5 does not use deepstack (intermediate visual features
            # injected at multiple LM depths) — that is a Qwen3VL-MoE feature.
            image_embeddings, _ = vision_encoder(
                pixel_values=pixel_values_list,
                idxs=indices_list,
                weights=weights_list,
                grid_thw=grid_thw_list,
                rot_pos_ids=rot_pos_ids_list,
                max_grid_size=max_grid_size_list,
                cu_seqlens=cu_seqlens_list,
                max_seqlen=max_seqlen_list,
                signal_buffers=signal_buffers,
            )
            assert image_embeddings is not None

            graph.output(*image_embeddings)
            return graph

    def _build_language_graph(
        self,
        state_dict: dict[str, Any],
        module: Module,
    ) -> Graph:
        full_state_dict = state_dict

        model_config = Qwen3_5Config.initialize_from_config(
            self.pipeline_config,
            self.huggingface_config,
            max_seq_len=self.max_seq_len,
        )
        model_config.finalize(
            huggingface_config=Qwen3_5Config._get_text_config(
                self.huggingface_config
            ),
            state_dict=full_state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )

        # finalize() reads tie_word_embeddings from the text sub-config,
        # which inherits PretrainedConfig's default of True.  The correct
        # value lives on the top-level config.
        model_config.tie_word_embeddings = getattr(
            self.huggingface_config, "tie_word_embeddings", False
        )
        nn_model = Qwen3_5(model_config)

        graph_inputs = nn_model.input_types(self.kv_params)

        _check_weights_match(
            expected=set(nn_model.raw_state_dict().keys()),
            provided=set(full_state_dict.keys()),
        )

        nn_model.load_state_dict(
            full_state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )

        # Split processed state dict into vision and LM parts.
        # Vision keys keep their "vision_encoder." prefix because the graph
        # resolves weights relative to nn_model (the root), so the registry
        # must match those fully-qualified paths.
        processed = nn_model.state_dict()
        vision_prefix = "vision_encoder."
        self._vision_state_dict = {
            k: v for k, v in processed.items() if k.startswith(vision_prefix)
        } or None
        self.state_dict = {
            k: v
            for k, v in processed.items()
            if not k.startswith(vision_prefix)
        }
        # Keep a reference so _build_vision_graph can access vision_encoder
        self._nn_model = nn_model

        # Save dimensions for state buffer allocation and empty-buffer creation
        # Per-device shard widths: the tensor-parallel split is by head, so
        # the state pools follow the value heads onto their own device.
        num_devices = len(self.devices)
        self._num_linear_layers = len(nn_model.linear_layer_indices)
        self._conv_dim = nn_model._conv_dim // num_devices
        self._conv_kernel_size = nn_model._conv_kernel_size
        self._num_v_heads = nn_model._num_v_heads // num_devices
        self._key_head_dim = nn_model._key_head_dim
        self._value_head_dim = nn_model._value_head_dim
        self._hidden_size = model_config.hidden_size
        self._model_dtype = model_config.compute_dtype
        self._state_dtype = model_config.state_dtype
        self._accounted_state_bytes = model_config._per_request_state_bytes()

        has_vision = nn_model.vision_encoder is not None
        num_linear_layers = self._num_linear_layers
        # Vision adds image_embeddings + image_token_indices, per device.
        vision_input_count = 2 * num_devices if has_vision else 0
        # M-RoPE adds one shared [3, total_seq_len] positions tensor.
        self._mrope_enabled = nn_model.mrope_enabled
        position_ids_count = 1 if nn_model.mrope_enabled else 0

        with Graph(
            "qwen3_5",
            input_types=graph_inputs,
            module=module,
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *variadic_args = (
                graph.inputs
            )

            # Extract signal buffers
            signal_buffers = [v.buffer for v in variadic_args[:num_devices]]

            # Unmarshal KV cache inputs. The trailing slice contains
            # [slot_idx, *conv_pools, *recurrent_pools, *vision_inputs].
            kv_start = num_devices
            pool_count = num_devices * num_linear_layers
            slot_idx_count = num_devices if num_linear_layers > 0 else 0
            kv_count = (
                len(variadic_args)
                - num_devices
                - slot_idx_count
                - pool_count * 2
                - vision_input_count
                - position_ids_count
            )
            kv_cache_inputs = variadic_args[kv_start : kv_start + kv_count]
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

            # Extract slot_idx + the linear-attention pools (BufferType
            # inputs). Every block is device-major.
            idx = kv_start + kv_count
            slot_idx_g: list[TensorValue] = []
            conv_pools: list[list[BufferValue]] = []
            recurrent_pools: list[list[BufferValue]] = []
            if num_linear_layers > 0:
                slot_idx_g = [
                    variadic_args[idx + d].tensor for d in range(num_devices)
                ]
                idx += num_devices
                for pools in (conv_pools, recurrent_pools):
                    pools.extend(
                        [
                            variadic_args[
                                idx + d * num_linear_layers + i
                            ].buffer
                            for i in range(num_linear_layers)
                        ]
                        for d in range(num_devices)
                    )
                    idx += pool_count

            # Extract vision inputs (only present for multimodal models)
            image_embeddings_g = None
            image_token_indices_g = None
            if has_vision:
                image_embeddings_g = [
                    variadic_args[idx + d].tensor for d in range(num_devices)
                ]
                image_token_indices_g = [
                    variadic_args[idx + num_devices + d].tensor
                    for d in range(num_devices)
                ]
                idx += vision_input_count

            position_ids_g = (
                variadic_args[idx].tensor if position_ids_count else None
            )

            assert slot_idx_g, (
                "Qwen3.5 graph requires linear attention layers; got 0"
            )
            outputs = nn_model(
                tokens.tensor,
                kv_collections,
                return_n_logits.tensor,
                input_row_offsets.tensor,
                signal_buffers,
                slot_idx_g,
                conv_pools,
                recurrent_pools,
                image_embeddings_g,
                image_token_indices_g,
                position_ids_g,
            )

            graph.output(*outputs)
            return graph

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, Qwen3_5Inputs)
        assert model_inputs.kv_cache_inputs is not None

        model_outputs = self.model.execute(*model_inputs.buffers)

        # The slot-indexed SSM kernels mutate the conv/recurrent pools in
        # place; the only graph output is the logits.
        logits = model_outputs[0]
        assert isinstance(logits, Buffer)

        return ModelOutputs(
            logits=logits,
            next_token_logits=logits,
        )

    def release(self, request_id: RequestID) -> None:
        """Release per-request state cache slot when a request completes."""
        if self._state_cache is not None:
            self._state_cache.release(request_id)

    def release_warmup_state(self, request_ids: list[RequestID]) -> None:
        """Release state pool slots claimed during graph-capture warmup.

        Called by the overlap pipeline's ``_warmup_model_inputs`` context
        manager after each ``(batch_size, cache_length)`` probe completes.
        Each probe claims up to ``batch_size`` fresh slots; without this
        release the warmup sweep would exhaust the pool before serving
        begins.

        The pool rows are NOT zeroed here — the state a warmup forward wrote
        is wiped by the next ``claim()`` for that slot, when a real request
        is assigned to it.
        """
        if self._state_cache is not None:
            for request_id in request_ids:
                self._state_cache.release(request_id)
