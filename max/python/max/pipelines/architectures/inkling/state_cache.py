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
"""Per-request slot pool for Inkling's short-convolution state: one pool per
convolution site per layer per device, updated in place by the conv kernel.
Rank ``r`` owns the channel range ``[r * C / tp_size, (r + 1) * C / tp_size)``
of every site, matching the reference's TP narrowing.

A slot is never cleared. A request's first chunk has no convolution history by
definition, so it runs with ``has_initial_state`` false and the kernel reads
zeros instead of the slot; the same kernel writes every state frame at the end
of the chunk, left-zero-padded, so whatever the previous tenant left behind is
overwritten before any later chunk reads it."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Final

import numpy as np
from max.driver import Buffer, Device, DevicePinnedBuffer
from max.dtype import DType
from max.graph import BufferType, BufferValue, DeviceRef, Value
from max.pipelines.modeling.types import RequestID
from max.support.human_readable_formatter import to_human_readable_bytes
from typing_extensions import Self

from .model_config import InklingTextConfig

logger = logging.getLogger("max.pipelines")

# The conv kernel accumulates in float32 regardless of the model dtype.
CONV_STATE_DTYPE: Final = DType.float32


class ConvSite(IntEnum):
    """The four convolution sites of a layer, in pool order."""

    K = 0
    V = 1
    ATTN_OUT = 2
    MLP_OUT = 3


@dataclass(frozen=True)
class InklingConvStateLayout:
    """Per-device channel widths of each layer's four sites, in
    :class:`ConvSite` order."""

    state_len: int
    layers: tuple[tuple[int, int, int, int], ...]

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def bytes_per_request(self) -> int:
        """Bytes one request occupies on one device."""
        channels = sum(map(sum, self.layers))
        return channels * self.state_len * CONV_STATE_DTYPE.size_in_bytes

    def take_pools(
        self, inputs: Iterator[Value[Any]], num_devices: int
    ) -> list[list[BufferValue]]:
        """Pulls this layout's pools off a graph-input iterator, one list per rank."""
        per_device = self.num_layers * len(ConvSite)
        return [
            [next(inputs).buffer for _ in range(per_device)]
            for _ in range(num_devices)
        ]

    def buffer_types(self, devices: Sequence[DeviceRef]) -> list[BufferType]:
        """Graph input types of the pools, in device then :attr:`layers` order."""
        return [
            BufferType(
                CONV_STATE_DTYPE,
                shape=["max_conv_slots", channels, self.state_len],
                device=device,
            )
            for device in devices
            for widths in self.layers
            for channels in widths
        ]

    @classmethod
    def from_config(
        cls,
        text_config: InklingTextConfig,
        *,
        tp_size: int = 1,
    ) -> Self:
        """Derives the layout from the checkpoint config."""
        return cls.from_local_flags(
            text_config,
            [
                text_config.is_local_attention(i)
                for i in range(text_config.num_hidden_layers)
            ],
            tp_size=tp_size,
        )

    @classmethod
    def from_local_flags(
        cls,
        text_config: InklingTextConfig,
        is_local: Sequence[bool],
        *,
        tp_size: int = 1,
    ) -> Self:
        """Layout for decoder blocks with an explicit local/global mix."""
        residual_width = text_config.hidden_size // tp_size
        layers = []
        for local in is_local:
            kv_width = text_config.kv_conv_dim(local) // tp_size
            layers.append((kv_width, kv_width, residual_width, residual_width))
        return cls(
            state_len=text_config.sconv_kernel_size - 1,
            layers=tuple(layers),
        )


class _StagedAdmissionInputs:
    """Per-device staging for the batch's slot indices and has-initial-state
    flags, uploaded together in one host-to-device copy per rank instead of
    two separate ones.

    Backed by one uint8 buffer per device: a ``max_slots``-long uint32 region
    (slot indices) immediately followed by a ``max_slots``-long bool region
    (has-initial-state flags). Every upload copies the whole buffer regardless
    of batch size, since splitting it back into two batch-sized transfers
    would reintroduce the second copy this class exists to remove; both
    regions are bounded by ``max_slots`` and tiny, so the few unused trailing
    bytes cost nothing next to the API call saved.
    """

    def __init__(self, max_slots: int, devices: Sequence[Device]) -> None:
        slot_idx_bytes = max_slots * DType.uint32.size_in_bytes
        total_bytes = slot_idx_bytes + max_slots * DType.bool.size_in_bytes
        self._slot_idx_bytes = slot_idx_bytes

        self._staging: list[Buffer | DevicePinnedBuffer] = [
            Buffer.zeros([total_bytes], DType.uint8, device)
            if device.is_host
            else DevicePinnedBuffer(
                shape=(total_bytes,), dtype=DType.uint8, device=device
            )
            for device in devices
        ]
        preallocs: list[Buffer] = [
            Buffer(shape=[total_bytes], dtype=DType.uint8, device=device)
            for device in devices
        ]
        self._preallocs = preallocs
        self._slot_idx_region = [
            prealloc[:slot_idx_bytes].view(DType.uint32, shape=[max_slots])
            for prealloc in preallocs
        ]
        self._initial_state_region = [
            prealloc[slot_idx_bytes:].view(DType.bool, shape=[max_slots])
            for prealloc in preallocs
        ]
        self._views: list[dict[int, tuple[Buffer, Buffer]]] = [
            {} for _ in devices
        ]

    def upload(
        self, slot_idx: np.ndarray, has_initial_state: np.ndarray
    ) -> tuple[list[Buffer], list[Buffer]]:
        """Copies both arrays to every rank in one transfer each, returning
        one (slot_idx, has_initial_state) device view pair per rank."""
        batch_size = len(slot_idx)
        slot_idx_views = []
        initial_state_views = []
        for (
            prealloc,
            staging,
            slot_idx_region,
            initial_state_region,
            views,
        ) in zip(
            self._preallocs,
            self._staging,
            self._slot_idx_region,
            self._initial_state_region,
            self._views,
            strict=True,
        ):
            staged = staging.to_numpy()
            staged[: self._slot_idx_bytes].view(np.uint32)[:batch_size] = (
                slot_idx
            )
            staged[self._slot_idx_bytes :][:batch_size] = has_initial_state
            prealloc.inplace_copy_from(staging)

            view_pair = views.get(batch_size)
            if view_pair is None:
                view_pair = (
                    slot_idx_region[:batch_size],
                    initial_state_region[:batch_size],
                )
                views[batch_size] = view_pair
            slot_idx_views.append(view_pair[0])
            initial_state_views.append(view_pair[1])
        return slot_idx_views, initial_state_views


class InklingConvStateCache:
    """Slot pool holding every request's convolution state.

    Pool ``pools(device)[layer * 4 + site]`` has shape
    ``[max_slots, site_channels, state_len]``.
    """

    def __init__(
        self,
        layout: InklingConvStateLayout,
        max_slots: int,
        devices: Sequence[Device],
    ) -> None:
        self._max_slots = max_slots
        self._devices = list(devices)

        self._pools: list[list[Buffer]] = [
            [
                Buffer.zeros(
                    [max_slots, channels, layout.state_len],
                    CONV_STATE_DTYPE,
                    device,
                )
                for widths in layout.layers
                for channels in widths
            ]
            for device in self._devices
        ]
        self._admission_input = _StagedAdmissionInputs(max_slots, self._devices)

        self._free_slots: set[int] = set(range(max_slots))
        self._request_to_slot: dict[RequestID, int] = {}

        per_request = layout.bytes_per_request()
        logger.info(
            f"Inkling conv state pools: {max_slots} slots x "
            f"{layout.num_layers} layers x {len(ConvSite)} sites = "
            f"{to_human_readable_bytes(max_slots * per_request)} per device "
            f"({to_human_readable_bytes(per_request)} per request) on "
            f"{len(self._devices)} device(s)"
        )

    def pools(self, device_idx: int) -> list[Buffer]:
        """Per-site pools of one rank, in layer then :class:`ConvSite` order."""
        return self._pools[device_idx]

    def claim(self, request_id: RequestID) -> int:
        """Assigns a slot; idempotent for chunked prefill. Does no device work:
        see this module's docstring for why the slot needs no clearing."""
        if request_id in self._request_to_slot:
            return self._request_to_slot[request_id]
        if not self._free_slots:
            raise RuntimeError(
                f"No free Inkling conv state slots ({self._max_slots} slots in "
                "use). Increase max_batch_size or reduce concurrent requests."
            )
        slot = self._free_slots.pop()
        self._request_to_slot[request_id] = slot
        return slot

    def release(self, request_id: RequestID) -> None:
        """Frees a request's slot; the preemption path — state is dropped."""
        slot = self._request_to_slot.pop(request_id, None)
        if slot is not None:
            self._free_slots.add(slot)

    def admission_inputs_for(
        self, request_ids: Sequence[RequestID], first_chunk: Sequence[bool]
    ) -> tuple[list[Buffer], list[Buffer]]:
        """Returns one (slot_idx, has_initial_state) device tensor pair per
        rank for the batch, uploaded together in a single host-to-device copy
        per rank. A first chunk has no convolution history to read."""
        slot_idx = np.fromiter(
            (self._request_to_slot[rid] for rid in request_ids),
            dtype=np.uint32,
            count=len(request_ids),
        )
        has_initial_state = ~np.asarray(first_chunk, dtype=np.bool_)
        return self._admission_input.upload(slot_idx, has_initial_state)
