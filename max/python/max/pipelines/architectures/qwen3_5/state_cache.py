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
"""GPU-resident slot pool for Qwen3.5 GatedDeltaNet linear attention states.

The conv and recurrent state pools are passed directly into the model graph
as mutable ``BufferType`` inputs; the slot-indexed SSM kernels mutate them
in place at slot ``slot_idx[batch_item]`` using pointer arithmetic — no
gather/scatter copies, no per-step pool allocation. This matches vLLM's
``selective_state_update`` design and replaces the previous per-row Python
``inplace_copy_from`` loop, collapsing
``num_layers * batch * 2 directions`` D2D launches per decode step into a
single tiny H2D for ``slot_idx``.

Memory note: the pool occupies permanent GPU memory at the model's native
dtype (typically bfloat16). The kernel reads/writes slot
``slot_idx[batch_item]`` directly, so there are no working copies — peak
footprint is ``max_batch x per_req`` bytes.
"""

from __future__ import annotations

import logging

import numpy as np
from max.driver import Buffer, Device, DevicePinnedBuffer
from max.dtype import DType
from max.pipelines.modeling.types import RequestID
from max.support.human_readable_formatter import to_human_readable_bytes

logger = logging.getLogger("max.pipelines")


class GatedDeltaNetStateCache:
    """GPU-resident slot pool for Qwen3.5 GatedDeltaNet conv and recurrent states.

    Two sets of per-layer ``Buffer`` objects are pre-allocated on the GPU at
    construction time:

    * ``conv_pool[l]``: ``[max_slots, conv_dim, conv_kernel-1]``
    * ``rec_pool[l]``:  ``[max_slots, num_v_heads, key_dim, val_dim]``

    Pools are exposed via :attr:`conv_pools` / :attr:`rec_pools` and passed
    into the model graph as mutable ``BufferType`` inputs; the slot-indexed
    SSM kernels read and write the pool directly at slot
    ``slot_idx[batch_item]``. There is no Python-side gather/scatter: the
    only per-step host-to-device transfer is the small ``slot_idx`` tensor
    built by :meth:`slot_idx_for`.

    Lifecycle:
        1. :meth:`claim` — register a request, zero its pool rows
        2. :meth:`slot_idx_for` — write the batch's slot indices into a
           caller-owned ``[max_batch_size]`` uint32 prealloc and return a
           ``[B]`` view
        3. (model.execute consumes pools + slot_idx; the kernels mutate the
           pools in place — no graph outputs to handle)
        4. :meth:`release` — free the slot
    """

    def __init__(
        self,
        num_layers: int,
        conv_dim: int,
        conv_kernel_size: int,
        num_v_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        max_slots: int,
        devices: list[Device],
        dtype: DType = DType.float32,
    ) -> None:
        """Allocates one pool set per device.

        Args:
            num_layers: Number of linear-attention layers.
            conv_dim: Conv width **per device** (the tensor-parallel shard).
            conv_kernel_size: Causal conv kernel size; the window is one less.
            num_v_heads: Value heads **per device**.
            key_head_dim: Key head dimension.
            value_head_dim: Value head dimension.
            max_slots: Concurrent requests the pool can hold.
            devices: Devices the value heads are split across.
            dtype: Pool storage dtype.
        """
        self._num_layers = num_layers
        self._conv_dim = conv_dim
        self._conv_kernel = conv_kernel_size - 1  # K-1 (state window size)
        self._num_v_heads = num_v_heads
        self._key_head_dim = key_head_dim
        self._value_head_dim = value_head_dim
        self._max_slots = max_slots
        self._devices = devices
        self._dtype = dtype

        conv_shape = [max_slots, conv_dim, conv_kernel_size - 1]
        rec_shape = [max_slots, num_v_heads, key_head_dim, value_head_dim]

        # Pre-allocate GPU state pools (zero-initialised), device-major so the
        # flat `conv_pools` / `rec_pools` views match the graph input order.
        self._conv_pool: list[Buffer] = [
            Buffer.zeros(conv_shape, dtype, device)
            for device in devices
            for _ in range(num_layers)
        ]
        self._rec_pool: list[Buffer] = [
            Buffer.zeros(rec_shape, dtype, device)
            for device in devices
            for _ in range(num_layers)
        ]

        # Pre-allocated zero buffers used to wipe a slot on claim().
        self._zero_conv = [
            Buffer.zeros([1, *conv_shape[1:]], dtype, device)
            for device in devices
        ]
        self._zero_rec = [
            Buffer.zeros([1, *rec_shape[1:]], dtype, device)
            for device in devices
        ]

        self._free_slots: set[int] = set(range(max_slots))
        self._request_to_slot: dict[RequestID, int] = {}

        # Reusable host-pinned staging buffer for the per-step slot_idx H2D.
        # Sized at max_slots since batch_size is bounded by the number of
        # claimable slots; allocated once here so slot_idx_for() doesn't
        # allocate on the per-step path. Pinned memory requires an
        # accelerator; on host devices (CPU tests) a plain Buffer suffices.
        self._pinned_slot_idx: list[Buffer] = [
            Buffer.zeros([max_slots], DType.uint32, device)
            if device.is_host
            else DevicePinnedBuffer(
                shape=(max_slots,), dtype=DType.uint32, device=device
            )
            for device in devices
        ]

        elem_bytes = dtype.size_in_bytes
        conv_bytes = (
            num_layers
            * max_slots
            * conv_dim
            * (conv_kernel_size - 1)
            * elem_bytes
        )
        rec_bytes = (
            num_layers
            * max_slots
            * num_v_heads
            * key_head_dim
            * value_head_dim
            * elem_bytes
        )
        self._bytes_per_slot = (conv_bytes + rec_bytes) // max_slots
        logger.info(
            f"GatedDeltaNet state pool: {max_slots} slots x {num_layers} layers"
            f" — conv {to_human_readable_bytes(conv_bytes)}"
            f" + rec {to_human_readable_bytes(rec_bytes)}"
            f" = {to_human_readable_bytes(conv_bytes + rec_bytes)}"
            f" on each of {devices}"
        )

    @property
    def num_free_slots(self) -> int:
        """Number of available (unclaimed) slots."""
        return len(self._free_slots)

    @property
    def num_active_slots(self) -> int:
        """Number of currently claimed slots."""
        return len(self._request_to_slot)

    @property
    def bytes_per_slot(self) -> int:
        """Bytes one slot occupies on one device, across every layer.

        Under tensor parallelism the recorded dimensions are already
        per-device shard widths, so this is ``1 / num_devices`` of what a
        request costs in total.
        """
        return self._bytes_per_slot

    @property
    def max_slots(self) -> int:
        """Maximum number of concurrent requests supported."""
        return self._max_slots

    @property
    def conv_pools(self) -> list[Buffer]:
        """Device-major conv pools, shape ``[max_slots, conv_dim, K-1]``."""
        return self._conv_pool

    @property
    def rec_pools(self) -> list[Buffer]:
        """Device-major recurrent pools, shape ``[max_slots, ...]``."""
        return self._rec_pool

    def claim(self, request_id: RequestID) -> int:
        """Assign a slot to a request, zeroing its pool rows on the GPU.

        If the request is already claimed, returns the existing slot
        (idempotent — state is preserved for chunked-prefill continuations).

        Args:
            request_id: The unique identifier for the request.

        Returns:
            The slot index assigned to this request.

        Raises:
            RuntimeError: If no free slots are available.
        """
        if request_id in self._request_to_slot:
            return self._request_to_slot[request_id]
        if not self._free_slots:
            raise RuntimeError(
                f"No free GatedDeltaNet state cache slots"
                f" ({self._max_slots} slots in use). "
                "Increase max_batch_size or reduce concurrent requests."
            )
        slot = self._free_slots.pop()
        self._request_to_slot[request_id] = slot
        self.zero_slot(slot)
        return slot

    def zero_slot(self, slot: int) -> None:
        """Wipes one slot's rows in every device's pools via GPU-to-GPU copy.

        Split out of :meth:`claim` so a caller that owns slot allocation
        itself -- a tensor-parallel model holding one of these per device --
        can reuse the buffers without also inheriting the bookkeeping.
        """
        for d in range(len(self._devices)):
            for l in range(self._num_layers):
                i = d * self._num_layers + l
                self._conv_pool[i][slot, :, :].inplace_copy_from(
                    self._zero_conv[d]
                )
                self._rec_pool[i][slot, :, :, :].inplace_copy_from(
                    self._zero_rec[d]
                )

    def release(self, request_id: RequestID) -> None:
        """Free a slot, making it available for future requests.

        Args:
            request_id: The unique identifier for the request to release.
        """
        if request_id not in self._request_to_slot:
            return
        slot = self._request_to_slot.pop(request_id)
        self._free_slots.add(slot)

    def contains(self, request_id: RequestID) -> bool:
        """Returns True if a slot is claimed for this request."""
        return request_id in self._request_to_slot

    def slot_idx_for(
        self, request_ids: list[RequestID], preallocs: list[Buffer]
    ) -> list[Buffer]:
        """Populate each device's ``prealloc[:B]`` with the batch's slots.

        Writes the slot for each request into the front of caller-owned
        device buffers using host-pinned staging + ``inplace_copy_from``, so
        no fresh device buffer is allocated on the per-step path. Every device
        gets the same indices — the pools are split by head, not by request.

        Args:
            request_ids: Ordered list of request IDs forming the batch.
            preallocs: One device-resident uint32 buffer of shape
                ``[max_slots]`` (or larger) per device, allocated once at
                model load time.

        Returns:
            One view into ``prealloc[:len(request_ids)]`` per device.

        Raises:
            ValueError: If ``request_ids`` is empty or larger than a
                ``prealloc``.
            KeyError: If any request ID has no claimed slot.
        """
        batch_size = len(request_ids)
        if batch_size == 0:
            raise ValueError("request_ids must not be empty")
        for prealloc in preallocs:
            if batch_size > prealloc.shape[0]:
                raise ValueError(
                    f"slot_idx_for: batch_size {batch_size} exceeds prealloc "
                    f"capacity {prealloc.shape[0]}"
                )
        for rid in request_ids:
            if rid not in self._request_to_slot:
                raise KeyError(
                    f"Request {rid} not found in GatedDeltaNet state cache. "
                    "Call claim() before slot_idx_for()."
                )
        slots = np.fromiter(
            (self._request_to_slot[rid] for rid in request_ids),
            dtype=np.uint32,
            count=batch_size,
        )
        views = []
        for prealloc, pinned in zip(
            preallocs, self._pinned_slot_idx, strict=True
        ):
            pinned.to_numpy()[:batch_size] = slots
            view = prealloc[:batch_size]
            view.inplace_copy_from(pinned[:batch_size])
            views.append(view)
        return views
