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
"""Allreduce module definitions."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
from max.driver import (
    Accelerator,
    Buffer,
    Device,
    enable_all_peer_access,
    is_virtual_device_mode,
)
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceKind,
    DeviceRef,
    TensorValue,
    ops,
)
from max.nn.layer import Module


class Allreduce(Module):
    """Layer to perform allreduce operation with automatic implementation selection.

    Automatically chooses between peer-to-peer optimized allreduce and naive
    device-to-device transfer based on accelerator connectivity.

    Args:
        num_accelerators: Number of accelerators participating in the allreduce operation
    """

    devices: list[Accelerator]
    """List of accelerators involved in the allreduce operation."""

    def __init__(self, num_accelerators: int) -> None:
        """Initialize the Allreduce layer with a specified number of accelerators.

        Args:
            num_accelerators: Number of accelerators to use for allreduce

        Raises:
            ValueError: If num_accelerators is less than 1
        """
        super().__init__()
        if num_accelerators < 1:
            raise ValueError("At least one accelerator required for Allreduce")

        self.devices = [Accelerator(id=id) for id in range(num_accelerators)]

    def __call__(
        self,
        inputs: Iterable[TensorValue],
        signal_buffers: Iterable[BufferValue],
    ) -> list[TensorValue]:
        """Performs allreduce operation with automatic implementation selection.

        Args:
            inputs: Distributed tensor values to reduce
            signal_buffers: Buffers for peer-to-peer communication when using
                optimized allreduce.

        Returns:
            List of reduced tensors, one per device
        """
        return ops.allreduce.sum(inputs, signal_buffers)


class Signals:
    """Signal buffers used for peer-to-peer communication in allreduce.

    Device code uses these buffers by enabling peer-to-peer access.
    Then thread blocks use the buffers to implement barriers for
    synchronization, and to hold intermediate communication results.
    """

    # ---- Per-GPU signal buffer layout. These MUST match ``Signal`` in
    # ``max/kernels/src/comm/sync.mojo``; the Lamport region offset below is
    # exact (a wrong value silently corrupts Lamport), so keep them in lockstep.
    _MAX_GPUS = 8
    _MAX_NUM_BLOCKS = 1024
    # Number of disjoint barrier counter banks. MUST match ``NUM_BARRIER_DOMAINS``
    # in ``max/kernels/src/comm/sync.mojo``. Each domain gets its own
    # self_counter + peer_counter grids so grouped (subgroup) and full-world
    # collectives never share a barrier bank on the same Signal buffer.
    _NUM_BARRIER_DOMAINS = _MAX_GPUS
    # Per domain: self_counter (1x) + peer_counter (2x) over the
    # [MAX_NUM_BLOCKS x MAX_GPUS] uint32 grid; times NUM_BARRIER_DOMAINS domains,
    # plus the 16-byte lamport_state block -> the Lamport region's byte offset
    # within the Signal struct.
    _LAMPORT_REGION_OFFSET = (
        _NUM_BARRIER_DOMAINS * 3 * _MAX_NUM_BLOCKS * _MAX_GPUS * 4 + 16
    )
    # Embedded Lamport comm region: 3 generations x MAX_GPUS slots x per-slot max.
    _LAMPORT_MAX_SMALL_MESSAGE_BYTES = 1024 * 1024
    _LAMPORT_REGION_BYTES = 3 * _MAX_GPUS * _LAMPORT_MAX_SMALL_MESSAGE_BYTES
    # Variable 2-stage / broadcast scratch, trailing the struct (disjoint from
    # the Lamport region, so the two collective families can interleave).
    _SCRATCH_BYTES = 256 * 1024 * 1024
    # Universal "-0.0" sentinel: fp32 -0.0 per uint32, detected as -0.0 under
    # bf16/fp16/fp32 (see ``set_neg_zero`` in lamport.mojo). Lets the region
    # init below be a dtype-agnostic uint32 fill.
    _LAMPORT_SENTINEL_U32 = 0x80000000

    NUM_BYTES = _LAMPORT_REGION_OFFSET + _LAMPORT_REGION_BYTES + _SCRATCH_BYTES
    """The size of the signal buffers used for communication in allreduce."""

    devices: list[DeviceRef]
    """List of graph devices that these signals communicate between."""

    def __init__(self, devices: Iterable[DeviceRef]) -> None:
        """Args:
        num_gpus: Number of GPUs involved in the allreduce.
        """
        # Convert the iterable to a list since we iterate over it twice.
        devices = list(devices)
        if not all(dev.device_type == DeviceKind.GPU for dev in devices):
            raise TypeError(
                "peer-to-peer signals should be constructed for accelerators"
            )

        self.devices = devices

    @staticmethod
    def _init_lamport_region(signal_buffers: list[Buffer]) -> None:
        """Fills each signal buffer's Lamport comm region with the -0.0 sentinel.

        The barrier-free Lamport allreduce uses -0.0 (``0x80000000`` per uint32)
        as its "slot not yet written" marker and spins until a peer overwrites
        the slot with real (producer-sanitized) data. A zero-filled region is
        +0.0, which reads as already-written, so the very first use of each
        generation would consume peer slots before the peers have pushed --
        a data race that yields non-deterministic results. This fill must run
        once per freshly allocated buffer set, before the first collective.

        Callers own synchronizing the buffers' devices afterward so the fill is
        visible before any allreduce runs.

        Args:
            signal_buffers: One freshly allocated signal buffer per device.

        Raises:
            ValueError: If ``NUM_BYTES`` cannot hold the Lamport region.
        """
        if (
            Signals.NUM_BYTES
            < Signals._LAMPORT_REGION_OFFSET + Signals._LAMPORT_REGION_BYTES
        ):
            raise ValueError(
                f"Expected signal buffer to be at least "
                f"{Signals._LAMPORT_REGION_OFFSET + Signals._LAMPORT_REGION_BYTES}"
                f" bytes, but got {Signals.NUM_BYTES}."
            )
        # Dtype-agnostic uint32 fill (the sentinel is fp32 -0.0 per uint32; see
        # `set_neg_zero` in lamport.mojo). Slice assignment isn't supported on a
        # device `Buffer`, but a sliced `__getitem__` returns a contiguous
        # sub-view sharing the memory, which `inplace_copy_from` can fill from a
        # host buffer (host -> device).
        start = Signals._LAMPORT_REGION_OFFSET // 4
        end = start + Signals._LAMPORT_REGION_BYTES // 4
        sentinel = Buffer.from_numpy(
            np.full(end - start, np.uint32(Signals._LAMPORT_SENTINEL_U32))
        )
        for buf in signal_buffers:
            region = buf.view(DType.uint32, shape=(Signals.NUM_BYTES // 4,))[
                start:end
            ]
            region.inplace_copy_from(sentinel)

    @staticmethod
    def allocate(devices: Sequence[Device]) -> list[Buffer]:
        """Allocates one fully initialized signal buffer per device.

        This is the single chokepoint for signal-buffer allocation: it enables
        peer-to-peer access, zeroes the buffers (the correct init for the
        barrier counters and ``lamport_state``), sentinel-initializes the
        Lamport comm region, and synchronizes so the buffers are ready for use
        when this method returns.

        Args:
            devices: Driver devices to allocate a buffer on, one each.

        Returns:
            One initialized signal buffer per device, in ``devices`` order, or
            an empty list in compile-only (virtual device) mode.
        """
        if is_virtual_device_mode():
            return []

        # Peer access is only meaningful with more than one GPU; the call is
        # idempotent and wrapped so a P2P-incapable host degrades gracefully.
        if len(devices) > 1:
            try:
                enable_all_peer_access()
            except RuntimeError:
                logging.getLogger(__name__).warning(
                    "Failed to enable peer-to-peer GPU access. "
                    "Collective operations will fall back to slower paths."
                )

        # Zero-fill: barrier counters and lamport_state start at 0 (their
        # correct init). The embedded Lamport comm region is sentinel-filled
        # below.
        signal_buffers = [
            Buffer.zeros(
                shape=(Signals.NUM_BYTES,),
                dtype=DType.uint8,
                device=dev,
            )
            for dev in devices
        ]

        # Init the Lamport comm region to the -0.0 sentinel; the synchronize
        # below guarantees every rank's region is sentinel before any collective
        # kernel
        Signals._init_lamport_region(signal_buffers)

        for dev in devices:
            dev.synchronize()

        return signal_buffers

    def buffers(self) -> list[Buffer]:
        """Allocates and returns buffers used for communication in allreduce.

        Thin adapter over :meth:`allocate` that maps this instance's graph
        ``DeviceRef``\\ s to the driver accelerators the allocator needs.
        """
        return Signals.allocate(
            [Accelerator(id=dev.id) for dev in self.devices]
        )

    def input_types(self) -> list[BufferType]:
        """Gets graph input types corresponding to these signal buffers."""
        return [
            BufferType(
                dtype=DType.uint8, shape=(Signals.NUM_BYTES,), device=dev
            )
            for dev in self.devices
        ]
