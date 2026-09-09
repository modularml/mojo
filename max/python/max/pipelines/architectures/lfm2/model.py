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

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.modeling.types import RequestID
from typing_extensions import override

from ..llama3.model import Llama3Inputs, LlamaModelBase
from .batch_processor import LFM2BatchProcessor
from .lfm2 import LFM2
from .model_config import LFM2Config


def _cat_buffers(buffers: list[Buffer], device: Device) -> Buffer:
    """Concatenate buffers along dim 0 via numpy round-trip.

    Used for the ``N > 1`` get-states fast path; conv states are small
    (``hidden * kernel`` per slot) so this round-trip is acceptable.

    numpy has no native bfloat16 support and rejects bfloat16 DLPack
    tensors with ``RuntimeError: Unsupported dtype in DLTensor``, so we
    reinterpret bfloat16 as uint16 (same byte width) for the round-trip
    and view the result back to the source dtype.
    """
    import numpy as np

    src_dtype = buffers[0].dtype
    if src_dtype == DType.bfloat16:
        viewed = [b.view(DType.uint16) for b in buffers]
        arrays = [v.to_numpy() for v in viewed]
        combined = np.concatenate(arrays, axis=0)
        return Buffer.from_numpy(combined).to(device).view(src_dtype)

    arrays = [b.to_numpy() for b in buffers]
    combined = np.concatenate(arrays, axis=0)
    return Buffer.from_numpy(combined).to(device)


def _split_buffer_dim0(buf: Buffer, device: Device) -> list[Buffer]:
    """Split a buffer into per-row slices along dim 0 via numpy round-trip.

    Mirror of ``_cat_buffers`` for the ``N > 1`` update-states path.
    ``Buffer.__getitem__`` requires one index per tensor dimension, so a
    rank-3 ``[batch, hidden, kernel]`` conv-state buffer cannot be sliced
    with a single slice expression; we round-trip through numpy instead,
    matching the bfloat16 view trick used in ``_cat_buffers``.
    """
    src_dtype = buf.dtype
    if src_dtype == DType.bfloat16:
        arr = buf.view(DType.uint16).to_numpy()
        return [
            Buffer.from_numpy(arr[i : i + 1]).to(device).view(src_dtype)
            for i in range(arr.shape[0])
        ]

    arr = buf.to_numpy()
    return [
        Buffer.from_numpy(arr[i : i + 1]).to(device)
        for i in range(arr.shape[0])
    ]


@dataclass
class LFM2Inputs(Llama3Inputs):
    conv_states: list[Buffer] = dataclasses.field(default_factory=list)
    request_ids: list[RequestID] = dataclasses.field(default_factory=list)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return super().buffers + tuple(self.conv_states)


class ConvStateCache:
    def __init__(
        self,
        num_conv_layers: int,
        hidden_size: int,
        conv_kernel: int,
        dtype: DType,
        max_slots: int,
        device: Device,
    ) -> None:
        self._num_conv_layers = num_conv_layers
        self._hidden_size = hidden_size
        self._conv_kernel = conv_kernel
        self._dtype = dtype
        self._device = device
        self._slots = [self._make_slot() for _ in range(max_slots)]
        self._free = set(range(max_slots))
        self._request_to_slot: dict[RequestID, int] = {}

    def _make_slot(self) -> list[Buffer]:
        return [
            Buffer.zeros(
                [1, self._hidden_size, self._conv_kernel],
                self._dtype,
                self._device,
            )
            for _ in range(self._num_conv_layers)
        ]

    def claim(self, request_id: RequestID) -> None:
        if request_id in self._request_to_slot:
            return
        if not self._free:
            raise RuntimeError("No free LFM2 conv-state slots.")
        slot = self._free.pop()
        self._request_to_slot[request_id] = slot
        self._slots[slot] = self._make_slot()

    def release(self, request_id: RequestID) -> None:
        slot = self._request_to_slot.pop(request_id, None)
        if slot is not None:
            self._free.add(slot)

    def get_states(self, request_ids: list[RequestID]) -> list[Buffer]:
        """Return one ``[N, hidden, kernel]`` buffer per conv layer.

        For ``N == 1`` this is zero-copy (returns the slot's buffer
        directly). For ``N > 1`` per-slot buffers are concatenated along
        the leading batch dim via numpy round-trip — the conv state is
        small (``hidden * kernel`` per slot), so this is acceptable.
        """
        if not request_ids:
            raise ValueError("request_ids must not be empty")
        for rid in request_ids:
            if rid not in self._request_to_slot:
                raise KeyError(
                    f"Request {rid!r} has no allocated conv-state slot. "
                    "Call claim() before get_states()."
                )

        if len(request_ids) == 1:
            slot = self._request_to_slot[request_ids[0]]
            return list(self._slots[slot])

        slots = [self._request_to_slot[rid] for rid in request_ids]
        result: list[Buffer] = []
        for layer_idx in range(self._num_conv_layers):
            slot_bufs = [self._slots[s][layer_idx] for s in slots]
            result.append(_cat_buffers(slot_bufs, self._device))
        return result

    def update_states(
        self, request_ids: list[RequestID], new_states: list[Buffer]
    ) -> None:
        """Store updated per-layer states back into their request slots.

        For ``N == 1`` the buffer reference is stored directly. For
        ``N > 1`` the leading batch dim is split and each slice is
        copied into the matching slot.
        """
        if len(new_states) != self._num_conv_layers:
            raise ValueError(
                f"Expected {self._num_conv_layers} state tensors, "
                f"got {len(new_states)}"
            )

        if len(request_ids) == 1:
            slot = self._request_to_slot[request_ids[0]]
            self._slots[slot] = list(new_states)
            return

        for layer_idx, state_buf in enumerate(new_states):
            pieces = _split_buffer_dim0(state_buf, self._device)
            for batch_idx, rid in enumerate(request_ids):
                slot = self._request_to_slot[rid]
                self._slots[slot][layer_idx] = pieces[batch_idx]


class LFM2Model(LlamaModelBase):
    """LFM2 hybrid (full-attention + conv) pipeline model."""

    model_config_cls: ClassVar[type[Any]] = LFM2Config
    batch_processor_cls: ClassVar[type[LFM2BatchProcessor]] = LFM2BatchProcessor

    norm_method = "rms_norm"
    attention_bias = False

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        num_conv_layers = sum(
            1 for t in self._model_config.layer_types if t != "full_attention"
        )
        self._conv_cache = ConvStateCache(
            num_conv_layers=num_conv_layers,
            hidden_size=self._model_config.hidden_size,
            conv_kernel=self._model_config.conv_L_cache,
            dtype=self._model_config.dtype,
            max_slots=self.max_batch_size,
            device=self.devices[0],
        )
        if self._batch_processor is not None:
            bind = getattr(self._batch_processor, "bind_conv_cache", None)
            if bind is not None:
                bind(self._conv_cache)

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> LFM2Config:
        model_config = LFM2Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )
        self._model_config = model_config
        return model_config

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: LFM2Config,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        model = LFM2(model_config)
        model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=True,
        )
        weights_registry = model.state_dict()
        self._num_kv_inputs = len(self.kv_params.flattened_kv_inputs())

        with Graph(
            "lfm2",
            input_types=model.input_types(self.kv_params),
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *variadic = graph.inputs
            kv_inputs = variadic[: self._num_kv_inputs]
            conv_inputs = [v.tensor for v in variadic[self._num_kv_inputs :]]
            kv_collections = self._unflatten_kv_inputs(kv_inputs)
            outputs = model(
                tokens=tokens.tensor,
                kv_collection=kv_collections[0],
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets.tensor,
                conv_states=conv_inputs,
            )
            graph.output(*outputs)
            return graph, weights_registry

    def _num_logit_outputs(self) -> int:
        has_offsets = self.return_logits in (
            ReturnLogits.VARIABLE,
            ReturnLogits.ALL,
        )
        has_hidden = self.return_hidden_states != ReturnHiddenStates.NONE
        return 1 + 2 * int(has_offsets) + int(has_hidden)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, LFM2Inputs)
        all_outputs = self.model.execute(*model_inputs.buffers)
        n_logit = self._num_logit_outputs()
        logit_outputs = tuple(all_outputs[:n_logit])
        new_states = list(all_outputs[n_logit:])
        if model_inputs.request_ids and new_states:
            self._conv_cache.update_states(model_inputs.request_ids, new_states)

        has_offsets = self.return_logits in (
            ReturnLogits.VARIABLE,
            ReturnLogits.ALL,
        )
        has_hidden = self.return_hidden_states != ReturnHiddenStates.NONE
        if has_offsets and has_hidden:
            return ModelOutputs(
                next_token_logits=logit_outputs[0],
                logits=logit_outputs[1],
                logit_offsets=logit_outputs[2],
                hidden_states=logit_outputs[3],
            )
        if has_offsets:
            return ModelOutputs(
                next_token_logits=logit_outputs[0],
                logits=logit_outputs[1],
                logit_offsets=logit_outputs[2],
            )
        if has_hidden:
            return ModelOutputs(
                logits=logit_outputs[0],
                next_token_logits=logit_outputs[0],
                hidden_states=logit_outputs[1],
            )
        return ModelOutputs(
            logits=logit_outputs[0], next_token_logits=logit_outputs[0]
        )

    def release(self, request_id: RequestID) -> None:
        self._conv_cache.release(request_id)
