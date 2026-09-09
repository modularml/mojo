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
"""GPU-resident slot pool for Nemotron-H Mamba-2 conv and SSM states.

Two sets of per-mamba-layer ``Buffer`` objects are pre-allocated on the GPU and
passed into the model graph as mutable ``BufferType`` inputs:

* ``conv_pool[l]``: ``[max_slots, conv_dim, conv_kernel - 1]`` (model dtype).
  Mutated in place by ``causal_conv1d_varlen_fwd`` at slot
  ``slot_idx[batch_item]`` — exactly the Qwen3.5 GatedDeltaNet conv pattern.
* ``ssm_pool[l]``: ``[max_slots, nheads, head_dim, dstate]`` (fp32, or bf16
  on Apple GPUs — storage only, the scan accumulates in fp32). Mutated
  in place by ``mamba2_ssd_chunk_scan_varlen_fwd_inplace`` at slot
  ``slot_idx[batch_item]`` — the kernel reads initial state from
  ``ssm_pool[slot]`` and writes the updated final state back to the same slot
  directly (no graph-side gather/scatter_nd/buffer_store whole-pool RMW).

This mirrors :class:`GatedDeltaNetStateCache` (qwen3_5); both pools are now
fully in-place slot-indexed.
"""

from __future__ import annotations

import logging

import numpy as np
from max.driver import Buffer, Device, DevicePinnedBuffer
from max.dtype import DType
from max.pipelines.modeling.types import RequestID
from max.support.human_readable_formatter import to_human_readable_bytes

logger = logging.getLogger("max.pipelines")


class NemotronHStateCache:
    """GPU-resident slot pool for Nemotron-H conv (bf16) and SSM (fp32) states.

    Lifecycle:
        1. :meth:`claim` — register a request, zero its pool rows.
        2. :meth:`slot_idx_for` — write the batch's slot indices into a
           caller-owned ``[max_slots]`` uint32 prealloc and return a ``[B]``
           view.
        3. (``model.execute`` consumes the pools + ``slot_idx``; both the conv
           op and the SSD inplace op mutate their pools in place — no graph
           state outputs to handle.)
        4. :meth:`release` — free the slot.
    """

    def __init__(
        self,
        num_mamba_layers: int,
        conv_dim: int,
        conv_kernel: int,
        nheads: int,
        head_dim: int,
        dstate: int,
        max_slots: int,
        device: Device,
        conv_dtype: DType,
        ssm_dtype: DType = DType.float32,
    ) -> None:
        self._num_layers = num_mamba_layers
        self._conv_dim = conv_dim
        self._conv_kernel = conv_kernel
        self._nheads = nheads
        self._head_dim = head_dim
        self._dstate = dstate
        self._max_slots = max_slots
        self._device = device
        self._conv_dtype = conv_dtype
        self._ssm_dtype = ssm_dtype

        # Pre-allocate GPU state pools (zero-initialised).
        # conv_pool[l]: [max_slots, conv_dim, K-1] (model dtype, in-place).
        # ssm_pool[l]:  [max_slots, nheads, head_dim, dstate] (ssm_dtype:
        # fp32, or bf16 on Apple GPUs — storage only, the scan accumulates
        # in fp32).
        self._conv_pool: list[Buffer] = [
            Buffer.zeros(
                [max_slots, conv_dim, conv_kernel - 1],
                conv_dtype,
                device,
            )
            for _ in range(num_mamba_layers)
        ]
        self._ssm_pool: list[Buffer] = [
            Buffer.zeros(
                [max_slots, nheads, head_dim, dstate],
                ssm_dtype,
                device,
            )
            for _ in range(num_mamba_layers)
        ]

        # Pre-allocated zero buffers used to wipe a slot on claim().
        self._zero_conv = Buffer.zeros(
            [1, conv_dim, conv_kernel - 1], conv_dtype, device
        )
        self._zero_ssm = Buffer.zeros(
            [1, nheads, head_dim, dstate], ssm_dtype, device
        )

        self._free_slots: set[int] = set(range(max_slots))
        self._request_to_slot: dict[RequestID, int] = {}

        # Reusable staging buffer for the per-step slot_idx H2D.
        # Use a host-pinned buffer for GPU devices (enables async DMA);
        # fall back to a plain Buffer on CPU (host) devices where pinned
        # memory is unavailable but also unnecessary.
        if device.is_host:
            self._pinned_slot_idx: Buffer | DevicePinnedBuffer = Buffer.zeros(
                [max_slots], DType.uint32, device
            )
        else:
            self._pinned_slot_idx = DevicePinnedBuffer(
                shape=(max_slots,), dtype=DType.uint32, device=device
            )

        conv_bytes = (
            num_mamba_layers
            * max_slots
            * conv_dim
            * (conv_kernel - 1)
            * conv_dtype.size_in_bytes
        )
        ssm_bytes = (
            num_mamba_layers
            * max_slots
            * nheads
            * head_dim
            * dstate
            * ssm_dtype.size_in_bytes
        )
        logger.info(
            f"Nemotron-H state pool: {max_slots} slots x {num_mamba_layers}"
            f" mamba layers — conv {to_human_readable_bytes(conv_bytes)}"
            f" + ssm {to_human_readable_bytes(ssm_bytes)}"
            f" = {to_human_readable_bytes(conv_bytes + ssm_bytes)} on {device}"
        )

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    @property
    def num_active_slots(self) -> int:
        return len(self._request_to_slot)

    @property
    def max_slots(self) -> int:
        return self._max_slots

    @property
    def conv_pools(self) -> list[Buffer]:
        """Per-layer conv pools, shape ``[max_slots, conv_dim, K-1]``."""
        return self._conv_pool

    @property
    def ssm_pools(self) -> list[Buffer]:
        """Per-layer SSM pools, shape ``[max_slots, nheads, head_dim, dstate]``."""
        return self._ssm_pool

    def claim(self, request_id: RequestID) -> int:
        """Assign a slot to a request, zeroing its pool rows on the GPU.

        Idempotent — preserves state for chunked-prefill continuations.

        Raises:
            RuntimeError: If no free slots are available.
        """
        if request_id in self._request_to_slot:
            return self._request_to_slot[request_id]
        if not self._free_slots:
            raise RuntimeError(
                f"No free Nemotron-H state cache slots"
                f" ({self._max_slots} slots in use). "
                "Increase max_batch_size or reduce concurrent requests."
            )
        slot = self._free_slots.pop()
        self._request_to_slot[request_id] = slot
        for layer in range(self._num_layers):
            self._conv_pool[layer][slot, :, :].inplace_copy_from(
                self._zero_conv
            )
            self._ssm_pool[layer][slot, :, :, :].inplace_copy_from(
                self._zero_ssm
            )
        return slot

    def release(self, request_id: RequestID) -> None:
        """Free a slot, making it available for future requests."""
        if request_id not in self._request_to_slot:
            return
        slot = self._request_to_slot.pop(request_id)
        self._free_slots.add(slot)

    def contains(self, request_id: RequestID) -> bool:
        return request_id in self._request_to_slot

    def slot_idx_for(
        self, request_ids: list[RequestID], prealloc: Buffer
    ) -> Buffer:
        """Populate ``prealloc[:B]`` with this batch's slot indices.

        Returns a view into ``prealloc[:len(request_ids)]``; built via a
        host-pinned staging buffer + ``inplace_copy_from`` so the per-step
        path allocates no fresh device buffer.

        Raises:
            ValueError: If ``request_ids`` is empty or larger than ``prealloc``.
            KeyError: If any request ID has no claimed slot.
        """
        batch_size = len(request_ids)
        if batch_size == 0:
            raise ValueError("request_ids must not be empty")
        if batch_size > prealloc.shape[0]:
            raise ValueError(
                f"slot_idx_for: batch_size {batch_size} exceeds prealloc "
                f"capacity {prealloc.shape[0]}"
            )
        for rid in request_ids:
            if rid not in self._request_to_slot:
                raise KeyError(
                    f"Request {rid} not found in Nemotron-H state cache. "
                    "Call claim() before slot_idx_for()."
                )
        self._pinned_slot_idx.to_numpy()[:batch_size] = np.fromiter(
            (self._request_to_slot[rid] for rid in request_ids),
            dtype=np.uint32,
            count=batch_size,
        )
        view = prealloc[:batch_size]
        view.inplace_copy_from(self._pinned_slot_idx[:batch_size])
        return view
