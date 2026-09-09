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
of every site, matching the reference's TP narrowing."""

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
        # One zero row per distinct width, reused to wipe a slot on claim.
        self._zero_rows: list[dict[int, Buffer]] = [
            {
                channels: Buffer.zeros(
                    [1, channels, layout.state_len], CONV_STATE_DTYPE, device
                )
                for widths in layout.layers
                for channels in widths
            }
            for device in self._devices
        ]
        self._staging: list[Buffer | DevicePinnedBuffer] = [
            Buffer.zeros([max_slots], DType.uint32, device)
            if device.is_host
            else DevicePinnedBuffer(
                shape=(max_slots,), dtype=DType.uint32, device=device
            )
            for device in self._devices
        ]
        self._preallocs: list[Buffer] = [
            Buffer(shape=[max_slots], dtype=DType.uint32, device=device)
            for device in self._devices
        ]
        # Views alias the preallocs; a stable Buffer object per batch size
        # lets graph-capture replay skip recopying the input.
        self._prealloc_views: list[dict[int, Buffer]] = [
            {} for _ in self._devices
        ]

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
        """Assigns a slot and zeros it; idempotent for chunked prefill."""
        if request_id in self._request_to_slot:
            return self._request_to_slot[request_id]
        if not self._free_slots:
            raise RuntimeError(
                f"No free Inkling conv state slots ({self._max_slots} slots in "
                "use). Increase max_batch_size or reduce concurrent requests."
            )
        slot = self._free_slots.pop()
        self._request_to_slot[request_id] = slot
        for pools, zero_rows in zip(self._pools, self._zero_rows, strict=True):
            for pool in pools:
                pool[slot, :, :].inplace_copy_from(zero_rows[pool.shape[1]])
        return slot

    def release(self, request_id: RequestID) -> None:
        """Frees a request's slot; the preemption path — state is dropped."""
        slot = self._request_to_slot.pop(request_id, None)
        if slot is not None:
            self._free_slots.add(slot)

    def slot_idx_for(self, request_ids: Sequence[RequestID]) -> list[Buffer]:
        """Returns one device tensor per rank with the batch's slots, staged
        through preallocated pinned buffers so the per-step path allocates
        nothing."""
        batch_size = len(request_ids)
        slots = np.fromiter(
            (self._request_to_slot[rid] for rid in request_ids),
            dtype=np.uint32,
            count=batch_size,
        )
        views = []
        for prealloc, staging, views_by_size in zip(
            self._preallocs, self._staging, self._prealloc_views, strict=True
        ):
            staging.to_numpy()[:batch_size] = slots
            view = views_by_size.get(batch_size)
            if view is None:
                view = prealloc[:batch_size]
                views_by_size[batch_size] = view
            view.inplace_copy_from(staging[:batch_size])
            views.append(view)
        return views
