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

"""Expert Parallelism (EP) Communication Manager.

This module provides classes and utilities for managing Expert Parallelism (EP)
communication in distributed inference scenarios.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import (
    Accelerator,
    Buffer,
    accelerator_api,
    enable_all_peer_access,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.experimental.tensor import Tensor
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Dim,
    Graph,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.support.human_readable_formatter import to_human_readable_bytes
from numpy.typing import NDArray

from .ep_config import (
    NUM_GROUPS,
    EPConfig,
)
from .ep_kernels import (
    call_distributed_ep_combine,
    call_distributed_ep_dispatch,
    call_ep_combine,
    call_ep_combine_async,
    call_ep_combine_wait,
    call_ep_dispatch,
    call_ep_dispatch_async,
    call_ep_dispatch_wait,
    call_ep_init,
)

logger = logging.getLogger("max.pipelines")


def get_ep_local_sync_counters_size(n_experts: int) -> int:
    """Returns the total size in Int32 elements needed for EP sync counters.

    This must match the EPLocalSyncCounters.total_size() in ep_comm.mojo.

    Memory Layout (all sizes in Int32 elements):
    - dispatch_async: 2 * n_experts + MAX_GPUS_PER_NODE
    - dispatch_wait/combine_async: 4 * n_experts + 4
    - combine_wait: 2 * n_experts

    Args:
        n_experts: Number of experts in the model.

    Returns:
        Total size in Int32 elements needed for all EP sync counters.
    """
    MAX_GPUS_PER_NODE = 8

    dispatch_async_size = 2 * n_experts + MAX_GPUS_PER_NODE
    dispatch_wait_size = 4 * n_experts + 4
    combine_wait_size = 2 * n_experts
    return dispatch_async_size + dispatch_wait_size + combine_wait_size


@dataclass
class EPCommBuffers:
    """SHMEM communication buffers for one Expert Parallelism MoE forward pass.

    Bundles the per-device send, receive, and synchronization tensors produced
    by ``EPBatchManager.comm_buffers()`` into a single value so they can be
    passed through the subgraph pytree machinery as one forward argument. The
    ``__tree_flatten__`` / ``__tree_unflatten__`` hooks expose the
    :class:`~max.experimental.tensor.Tensor` leaves to that machinery.
    """

    atomic_counters: list[list[Tensor]]
    """Atomic synchronization counters, indexed as ``[group][device]``, used to
    coordinate thread blocks during the dispatch and combine phases."""

    send_buf_ptrs: list[Tensor]
    """Per-group SHMEM send-buffer device pointers, one entry per buffer group.
    Each tensor holds the per-device addresses of the staging buffers for
    outgoing tokens."""

    recv_buf_ptrs: list[Tensor]
    """Per-group SHMEM receive-buffer device pointers, one entry per buffer
    group. Each tensor holds the per-device addresses of the buffers for
    incoming tokens from remote devices."""

    recv_count_ptrs: list[Tensor]
    """Per-group SHMEM receive-count device pointers, one entry per buffer
    group. Each tensor holds the per-device addresses of the buffers that signal
    transfer completion."""

    eplb_log2phy: dict[int, Tensor] | None = None
    """Per-device Expert Parallelism load-balancing (EPLB) logical-to-physical
    expert map, keyed by device ID. Populated only when EPLB is enabled;
    otherwise ``None``."""

    eplb_logcnt: dict[int, Tensor] | None = None
    """Per-device count of physical replicas per logical expert (EPLB), keyed by
    device ID. Populated only when EPLB is enabled; otherwise ``None``."""

    def __tree_flatten__(
        self,
    ) -> tuple[tuple[object, ...], tuple[str, ...]]:
        """Exposes the Tensor leaves to the subgraph pytree machinery."""
        names = tuple(f.name for f in dataclasses.fields(self))
        children = tuple(getattr(self, name) for name in names)
        return children, names

    @classmethod
    def __tree_unflatten__(
        cls, aux: tuple[str, ...], children: Sequence[Any]
    ) -> EPCommBuffers:
        """Rebuilds an :class:`EPCommBuffers` from flattened leaves."""
        return cls(**dict(zip(aux, children, strict=True)))


class EPBatchManager:
    """Batch manager for Expert Parallelism (EP).

    This module manages two groups of SHMEM buffers in the graph. It switches
    between the two groups to avoid racing.
    """

    config: EPConfig
    """Configuration for the Expert Parallelism (EP)."""

    _send_buf_ptrs: list[TensorValue] | None
    """SHMEM send buffer device pointers. Shape: [NUM_GROUPS] of
    TensorValue[n_gpus_per_node]. Each pointer references addresses to staging
    buffers for outgoing tokens."""

    _recv_buf_ptrs: list[TensorValue] | None
    """SHMEM receive buffer device pointers. Shape: [NUM_GROUPS] of
    TensorValue[n_gpus_per_node]. Each pointer references UInt64 addresses to
    buffers for incoming tokens from remote devices."""

    _recv_count_ptrs: list[TensorValue] | None
    """SHMEM receive count buffer device pointers. Shape: [NUM_GROUPS] of
    TensorValue[n_gpus_per_node]. Each pointer references UInt64 addresses to
    buffers for signalling transfer completion."""

    _atomic_counters: list[list[BufferValue]] | None
    """Atomic synchronization counters. Shape: [NUM_GROUPS][n_gpus_per_node]
    of BufferValue. Used for inter-thread-block coordination."""

    _src_info: dict[int, TensorValue | None]
    """Source routing information for combine phase. Each key is a device ID,
    and the value is a TensorValue with shape [max_recv_tokens, 2]. Maps expert
    outputs back to their source positions."""

    _eplb_log2phy_per_device: dict[int, BufferValue]
    """Per-device log2phy buffer. Shape: [num_moe_layer, num_experts_per_layer, max_replicas].
    Populated by fetch_buffers when config.eplb_enabled is True."""

    _eplb_logcnt_per_device: dict[int, BufferValue]
    """Per-device logcnt buffer. Shape: [num_moe_layer, num_experts_per_layer]"""

    _eplb_phy2log: NDArray[np.int64] | None
    """Per-layer logical->physical map from EPLB. Shape [num_layers, num_phy].
    Set once at startup before MoE.shard()."""

    _dispatch_dim: dict[int, Dim | None]
    """Dictionary of device ID to dimension for the dispatch input tensor.
    Used to determine the shape of the combined output tensor.
    """

    def __init__(self, config: EPConfig):
        """Initialize the EP batch manager.

        Args:
            config: EP configuration.
        """
        self.config = config
        # Per-instance state (two managers must not share these dicts).
        self._src_info = {}
        self._eplb_log2phy_per_device = {}
        self._eplb_logcnt_per_device = {}
        self._eplb_phy2log = None
        self._dispatch_dim = {}
        self._send_buf_ptrs = None
        self._recv_buf_ptrs = None
        self._recv_count_ptrs = None
        self._atomic_counters = None

        if getattr(config, "eplb_phy2log_plan", None) is not None:
            self._eplb_phy2log = config.eplb_phy2log_plan

    def _common_grouped_matmul_metadata(self) -> TensorValue:
        """Common grouped matmul metadata for all devices. Shape: (2,). Contains
        the max number of tokens per expert and the number of active experts.
        """
        n_ranks = self.config.n_gpus_per_node * self.config.n_nodes
        max_recv_tokens_per_expert = self.config.max_tokens_per_rank * n_ranks
        n_active_experts = self.config.n_experts // n_ranks + (
            1 if self.config.fused_shared_expert else 0
        )
        return ops.constant(
            [
                max_recv_tokens_per_expert,
                n_active_experts,
            ],
            dtype=DType.uint32,
            device=DeviceRef.CPU(),
        )

    @property
    def send_buf_ptrs(self) -> list[TensorValue]:
        if self._send_buf_ptrs is None:
            raise RuntimeError(
                "Call fetch_buffers() first to fetch buffer pointers."
            )
        return self._send_buf_ptrs

    @property
    def recv_buf_ptrs(self) -> list[TensorValue]:
        if self._recv_buf_ptrs is None:
            raise RuntimeError(
                "Call fetch_buffers() first to fetch buffer pointers."
            )
        return self._recv_buf_ptrs

    @property
    def recv_count_ptrs(self) -> list[TensorValue]:
        if self._recv_count_ptrs is None:
            raise RuntimeError(
                "Call fetch_buffers() first to fetch buffer pointers."
            )
        return self._recv_count_ptrs

    @property
    def atomic_counters(self) -> list[list[BufferValue]]:
        if self._atomic_counters is None:
            raise RuntimeError(
                "Call fetch_buffers() first to fetch buffer pointers."
            )
        return self._atomic_counters

    def _atomic_counters_input_types(self) -> list[BufferType]:
        """Generate input types for atomic counter buffers.

        Returns:
            list[BufferType]: List of buffer types for atomic counters.
        """
        n_experts = (
            self.config.n_experts // self.config.n_gpus_per_node
            if self.config.use_allreduce
            else self.config.n_experts
        )
        return [
            BufferType(
                DType.int32,
                [get_ep_local_sync_counters_size(n_experts)],
                device=DeviceRef.GPU(i % self.config.n_gpus_per_node),
            )
            for i in range(NUM_GROUPS * self.config.n_gpus_per_node)
        ]

    def _dev_ptrs_input_types(self) -> list[TensorType]:
        """Generate input types for device pointer tensors.

        Returns:
            list[TensorType]: List of tensor types for device pointers.
        """
        return (
            [
                TensorType(
                    DType.uint64,
                    [self.config.n_gpus_per_node],
                    device=DeviceRef.CPU(),
                ),
            ]
            * 3  # 3 buffer types: send, recv, recv_count
            * NUM_GROUPS  # For double buffering
        )

    def input_types(self) -> list[TensorType | BufferType]:
        """Get the input types for the MoE graph.

        Returns:
            list[TensorType | BufferType]: List of input types for atomic
                                          counters and device pointers.
        """
        types: list[TensorType | BufferType] = (
            self._atomic_counters_input_types() + self._dev_ptrs_input_types()
        )

        if self.config.eplb_enabled:
            types += self._eplb_input_types()
        return types

    def _eplb_input_types(self) -> list[TensorType | BufferType]:
        """Per-device log2phy + logcnt buffer types."""
        num_moe_layers = self.config.num_moe_layers
        num_experts_per_layer = self.config.num_logical_experts
        max_replicas = self.config.max_replicas
        n_gpus = self.config.n_gpus_per_node
        log2phy: list[TensorType | BufferType] = [
            BufferType(
                DType.int32,
                [num_moe_layers, num_experts_per_layer, max_replicas],
                device=DeviceRef.GPU(i),
            )
            for i in range(n_gpus)
        ]
        logcnt: list[TensorType | BufferType] = [
            BufferType(
                DType.int32,
                [num_moe_layers, num_experts_per_layer],
                device=DeviceRef.GPU(i),
            )
            for i in range(n_gpus)
        ]
        return log2phy + logcnt

    def fetch_buffers(self, _input_vals: Iterable[Value[Any]]) -> None:
        """Extract and organize communication buffers from graph input values.

        Args:
            input_vals: List of input values containing all buffer references.
        """
        input_vals = list(_input_vals)
        start_idx = 0
        # First NUM_GROUPS * self.config.n_gpus_per_node elements are atomic counters
        # These are used for synchronization between different thread blocks
        self._atomic_counters = []
        # Organize atomic counters by groups
        for _ in range(NUM_GROUPS):
            end_idx = start_idx + self.config.n_gpus_per_node
            group_buffers = [
                val.buffer for val in input_vals[start_idx:end_idx]
            ]
            self.atomic_counters.append(group_buffers)
            start_idx = end_idx

        # Next NUM_GROUPS are send buffer pointers
        end_idx = start_idx + NUM_GROUPS
        self._send_buf_ptrs = [
            val.tensor for val in input_vals[start_idx:end_idx]
        ]
        start_idx = end_idx

        # Next NUM_GROUPS are recv buffer pointers
        end_idx = start_idx + NUM_GROUPS
        self._recv_buf_ptrs = [
            val.tensor for val in input_vals[start_idx:end_idx]
        ]
        start_idx = end_idx

        # Next NUM_GROUPS are recv count pointers
        end_idx = start_idx + NUM_GROUPS
        self._recv_count_ptrs = [
            val.tensor for val in input_vals[start_idx:end_idx]
        ]

        start_idx = end_idx

        # Next 2*NUM_GROUPS are EPLB log2phy + logcnt buffers
        if self.config.eplb_enabled:
            n_gpus = self.config.n_gpus_per_node
            end_idx = start_idx + n_gpus
            self._eplb_log2phy_per_device = {
                i: input_vals[start_idx + i].buffer for i in range(n_gpus)
            }
            start_idx = end_idx
            self._eplb_logcnt_per_device = {
                i: input_vals[start_idx + i].buffer for i in range(n_gpus)
            }
            start_idx += n_gpus

    def comm_buffers(self, input_vals: Iterable[Value[Any]]) -> EPCommBuffers:
        """Fetches the EP buffers and wraps them as an :class:`EPCommBuffers`.

        Args:
            input_vals: Graph input values containing all buffer references,
                in the same order as :meth:`input_types`.

        Returns:
            Dataclass containing the EP comm tensors.
        """
        self.fetch_buffers(input_vals)
        eplb_log2phy: dict[int, Tensor] | None = None
        eplb_logcnt: dict[int, Tensor] | None = None
        if self.config.eplb_enabled:
            eplb_log2phy = {
                i: Tensor.from_graph_value(b)
                for i, b in self._eplb_log2phy_per_device.items()
            }
            eplb_logcnt = {
                i: Tensor.from_graph_value(b)
                for i, b in self._eplb_logcnt_per_device.items()
            }
        return EPCommBuffers(
            atomic_counters=[
                [Tensor.from_graph_value(b) for b in group]
                for group in self.atomic_counters
            ],
            send_buf_ptrs=[
                Tensor.from_graph_value(v) for v in self.send_buf_ptrs
            ],
            recv_buf_ptrs=[
                Tensor.from_graph_value(v) for v in self.recv_buf_ptrs
            ],
            recv_count_ptrs=[
                Tensor.from_graph_value(v) for v in self.recv_count_ptrs
            ],
            eplb_log2phy=eplb_log2phy,
            eplb_logcnt=eplb_logcnt,
        )

    def bind_comm_buffers(self, comm: EPCommBuffers) -> None:
        """Rebinds the buffer fields from a threaded :class:`EPCommBuffers`.

        Called at the top of the EP MoE forward so the kernel-dispatch helpers,
        which read ``self._send_buf_ptrs`` etc., see the (subgraph-rebound)
        buffer values passed in as a forward argument.
        """
        self._atomic_counters = [
            [BufferValue(t) for t in group] for group in comm.atomic_counters
        ]
        self._send_buf_ptrs = [TensorValue(t) for t in comm.send_buf_ptrs]
        self._recv_buf_ptrs = [TensorValue(t) for t in comm.recv_buf_ptrs]
        self._recv_count_ptrs = [TensorValue(t) for t in comm.recv_count_ptrs]
        if comm.eplb_log2phy is not None:
            self._eplb_log2phy_per_device = {
                i: BufferValue(t) for i, t in comm.eplb_log2phy.items()
            }
        if comm.eplb_logcnt is not None:
            self._eplb_logcnt_per_device = {
                i: BufferValue(t) for i, t in comm.eplb_logcnt.items()
            }

    def ep_dispatch_async(
        self,
        input_tokens: TensorValue,
        topk_ids: TensorValue,
        device_id: int,
        input_scales: TensorValue | None = None,
    ) -> None:
        """Initiate Expert Parallelism token dispatch phase (async).

        This function launches the EP async dispatch kernel that distributes
        input tokens to expert devices based on top-k routing decisions.

        Args:
            input_tokens: Input tokens for the current device. A TensorValue with
                shape (num_local_tokens, hidden_size).
            topk_ids: Top-k expert IDs for the current device. A TensorValue with
                shape (num_local_tokens, top_k).
            device_id: Device ID for the current device.
            input_scales: Optional input scales tensor. Required for NVFP4
                dispatch.
        """
        DISPATCH_GROUP = 0
        # Store the symbolic token numbers of each device for the combine phase
        self._dispatch_dim[device_id] = input_tokens.shape[0]
        call_ep_dispatch_async(
            input_tokens,
            topk_ids,
            self.atomic_counters[DISPATCH_GROUP][device_id],
            self.send_buf_ptrs[DISPATCH_GROUP],
            self.recv_buf_ptrs[DISPATCH_GROUP],
            self.recv_count_ptrs[DISPATCH_GROUP],
            self.config,
            input_scales=input_scales,
        )

    def ep_dispatch_wait(self, device_id: int) -> tuple[TensorValue, ...]:
        """Wait for Expert Parallelism token dispatch phase completion.

        This function launches the EP dispatch wait kernel that waits for all
        transfers to complete for the current GPU, then organizes the received
        tokens into a format suitable for grouped matmul computation.

        Args:
            device_id: Device ID for the current device.

        Returns:
            A tuple containing:
            - output_tokens: Aggregated tokens ready for grouped matmul computation.
                Shape: (max_recv_tokens, hidden_size).
            - output_scales: Aggregated scales ready for grouped matmul computation.
                Only returned for quantized dispatch. Shape depends on format.
            - expert_start_indices: Row offsets for grouped matmul computation.
                Shape: (n_local_experts + 1,).
            - expert_ids: Local expert IDs for the grouped computation.
                Shape: (n_local_experts,).
            - expert_usage_stats: Statistics for the grouped matmul computation.
                Shape: (2,).
        """
        DISPATCH_GROUP = 0

        results = call_ep_dispatch_wait(
            self.atomic_counters[DISPATCH_GROUP][device_id],
            self.recv_buf_ptrs[DISPATCH_GROUP],
            self.recv_count_ptrs[DISPATCH_GROUP],
            self.config,
            # Forward the per-rank input token dim so the kernel can pick
            # a smaller grid when num_tokens is statically known (decode).
            num_tokens=self._dispatch_dim.get(device_id),
        )

        # The last element is the src_info, we need to store it for the
        # combine phase. Also add the common grouped matmul metadata to the
        # results.
        self._src_info[device_id] = results[-1]

        return (*results[:-1], self._common_grouped_matmul_metadata())

    def ep_combine_async(
        self, input_tokens: TensorValue, device_id: int
    ) -> None:
        """Initiate Expert Parallelism combine phase (async).

        This method launches the async combine phase of Expert Parallelism,
        sending expert outputs back to their original devices based on source
        routing information stored during the dispatch phase.

        Args:
            input_tokens: Expert output tensors from the current device.
                A TensorValue with shape (max_recv_tokens, hidden_size).
            device_id: Device ID for the current device.
        """
        COMBINE_GROUP = 1
        # always use group 0 atomic counters unless we enable
        # two-batch-overlap.

        src_info = self._src_info[device_id]
        assert src_info is not None, (
            "Source info is not set, you should call ep_dispatch_wait() first."
        )

        call_ep_combine_async(
            input_tokens,
            src_info,
            self.atomic_counters[0][device_id],
            self.send_buf_ptrs[COMBINE_GROUP],
            self.recv_buf_ptrs[COMBINE_GROUP],
            self.recv_count_ptrs[COMBINE_GROUP],
            self.config,
        )

        # reset src_info to None to avoid reusing it for the next batch
        self._src_info[device_id] = None

    def ep_combine_wait(
        self, router_weight: TensorValue, device_id: int
    ) -> TensorValue:
        """Wait for Expert Parallelism combine phase completion.

        This method waits for all expert output transfers to complete, then
        organizes the received tokens back into their original format and
        positions for the current device.

        Args:
            expert_weights: Router weights for the current device.
                A TensorValue with shape (num_local_tokens, top_k).
            device_id: Device ID for the current device.

        Returns:
            Final output tensor with shape (num_local_tokens, hidden_size).
        """
        COMBINE_GROUP = 1

        # Collect results from all devices
        # always use group 0 atomic counters unless we enable
        # two-batch-overlap.
        dispatch_dim = self._dispatch_dim[device_id]
        assert dispatch_dim is not None, (
            "Dispatch dimension is not set, you should call ep_dispatch_async() first."
        )
        results = call_ep_combine_wait(
            self.atomic_counters[0][device_id],
            self.recv_buf_ptrs[COMBINE_GROUP],
            self.recv_count_ptrs[COMBINE_GROUP],
            self.config,
            dispatch_dim,
            router_weight,
        )

        return results

    # ===-------------------------------------------------------------------===#
    # Fused EP Operations
    # ===-------------------------------------------------------------------===#

    def ep_dispatch(
        self,
        input_tokens: TensorValue,
        topk_ids: TensorValue,
        device_id: int,
        input_scales: TensorValue | None = None,
    ) -> tuple[TensorValue, ...]:
        """Execute fused Expert Parallelism token dispatch (async + wait).

        This method launches the fused EP dispatch kernel that combines both
        dispatch_async and dispatch_wait functionality in a single kernel
        launch. It distributes input tokens to expert devices, waits for all
        tokens to arrive, and organizes received tokens for grouped matmul.

        For FP8 dispatch, input tokens are quantized to FP8 format during
        dispatch and the output includes both FP8 tokens and their scales.

        Args:
            input_tokens: Input tokens for the current device. A TensorValue
                with shape (num_local_tokens, hidden_size).
            topk_ids: Top-k expert IDs for the current device. A TensorValue
                with shape (num_local_tokens, top_k).
            device_id: Device ID for the current device.
            input_scales: Optional input scales tensor. Needed for NVFP4
                dispatch.

        Returns:
            A tuple containing:
            - output_tokens: Aggregated tokens ready for grouped matmul.
                Shape: (max_recv_tokens, hidden_size).
            - For FP8: output_scales: Scales for the FP8 tokens.
                Shape: (hidden_size // block_size, max_recv_tokens).
            - expert_start_indices: Row offsets for grouped matmul.
                Shape: (n_local_experts + 1,).
            - expert_ids: Local expert IDs for the grouped operation.
                Shape: (n_local_experts,).
            - expert_usage_stats: Statistics for the grouped matmul.
                Shape: (2,).
        """
        # Use group 0 for both send and recv buffers in fused kernel
        DISPATCH_GROUP = 0

        # Store the symbolic token numbers for the combine phase
        self._dispatch_dim[device_id] = input_tokens.shape[0]
        results = call_ep_dispatch(
            input_tokens,
            topk_ids,
            self.atomic_counters[DISPATCH_GROUP][device_id],
            self.send_buf_ptrs[DISPATCH_GROUP],
            self.recv_buf_ptrs[DISPATCH_GROUP],
            self.recv_count_ptrs[DISPATCH_GROUP],
            self.config,
            input_scales=input_scales,
        )

        # The last element is the src_info, we need to store it for the
        # combine phase. Also add the common grouped matmul metadata to the
        # results.
        self._src_info[device_id] = results[-1]

        return (*results[:-1], self._common_grouped_matmul_metadata())

    def ep_dispatch_all(
        self,
        input_tokens: list[TensorValue],
        topk_ids: list[TensorValue],
        device_ids: list[int],
        input_scales: list[TensorValue] | None = None,
    ) -> list[tuple[TensorValue, ...]]:
        """Multi-device fused EP dispatch across all devices.

        Launches a single multi-device dispatch graph op (BF16, FP8, or
        NVFP4 depending on config) that dispatches tokens on all devices
        simultaneously.

        Args:
            input_tokens: Per-device input token tensors.
            topk_ids: Per-device top-k expert ID tensors.
            device_ids: Device IDs corresponding to each input.
            input_scales: Per-device input scales (required for NVFP4).

        Returns:
            Per-device output tuples. The last element of each tuple is
            always ``src_info``; the remaining elements are the dispatch
            outputs followed by the grouped matmul metadata.
        """
        DISPATCH_GROUP = 0

        for i, device_id in enumerate(device_ids):
            self._dispatch_dim[device_id] = input_tokens[i].shape[0]

        atomic_counters = [
            self.atomic_counters[DISPATCH_GROUP][d] for d in device_ids
        ]

        all_results = call_distributed_ep_dispatch(
            input_tokens,
            topk_ids,
            atomic_counters,
            self.send_buf_ptrs[DISPATCH_GROUP],
            self.recv_buf_ptrs[DISPATCH_GROUP],
            self.recv_count_ptrs[DISPATCH_GROUP],
            self.config,
            input_scales=input_scales,
        )

        per_device_outputs: list[tuple[TensorValue, ...]] = []
        gmm_meta = self._common_grouped_matmul_metadata()
        for i, device_id in enumerate(device_ids):
            results = all_results[i]
            self._src_info[device_id] = results[-1]
            per_device_outputs.append((*results[:-1], gmm_meta))

        return per_device_outputs

    def ep_combine_all(
        self,
        input_tokens: list[TensorValue],
        router_weights: list[TensorValue],
        device_ids: list[int],
    ) -> list[TensorValue]:
        """Multi-device fused EP combine across all devices.

        Launches a single ``mo.distributed.ep.combine`` graph op that
        combines expert outputs back to their original devices on all
        GPUs simultaneously.

        Args:
            input_tokens: Per-device expert output tokens.
            router_weights: Per-device router weight tensors.
            device_ids: Device IDs corresponding to each input.

        Returns:
            Per-device combined output tensors.
        """
        COMBINE_GROUP = 1

        src_info_list: list[TensorValue] = []
        dispatch_dims: list[Dim] = []
        for device_id in device_ids:
            si = self._src_info[device_id]
            assert si is not None, (
                "Source info is not set, call ep_dispatch_all() first."
            )
            src_info_list.append(si)
            dd = self._dispatch_dim[device_id]
            assert dd is not None, (
                "Dispatch dim is not set, call ep_dispatch_all() first."
            )
            dispatch_dims.append(dd)

        atomic_counters = [self.atomic_counters[0][d] for d in device_ids]

        results = call_distributed_ep_combine(
            input_tokens,
            src_info_list,
            atomic_counters,
            self.send_buf_ptrs[COMBINE_GROUP],
            self.recv_buf_ptrs[COMBINE_GROUP],
            self.recv_count_ptrs[COMBINE_GROUP],
            self.config,
            dispatch_dims,
            router_weights,
        )

        for device_id in device_ids:
            self._src_info[device_id] = None

        return results

    def ep_combine(
        self,
        input_tokens: TensorValue,
        router_weight: TensorValue,
        device_id: int,
        topk_ids: TensorValue | None = None,
    ) -> TensorValue:
        """Execute fused Expert Parallelism token combine (async + wait).

        This method launches the fused EP combine kernel that combines both
        combine_async and combine_wait functionality in a single kernel launch.
        It sends expert outputs back to original devices, waits for all
        transfers to complete, and computes the weighted sum of routed expert
        outputs.

        Note: For fused_shared_expert mode with the fused combine kernel, the
        shared expert outputs in input_tokens are automatically added to the
        reduced routed expert outputs.

        Args:
            input_tokens: Expert output tensors from the current device.
                A TensorValue with shape (max_recv_tokens, hidden_size).
                For fused_shared_expert mode, the shared expert outputs are
                stored at the start.
            router_weight: Router weights for the current device.
                A TensorValue with shape (num_local_tokens, top_k).
            device_id: Device ID for the current device.
            topk_ids: Top-k expert IDs for the current device. Need to be
                provided for allreduce mode.

        Returns:
            Final output tensor with shape (num_local_tokens, hidden_size).
        """
        COMBINE_GROUP = 1

        src_info = self._src_info[device_id]
        assert src_info is not None, (
            "Source info is not set, you should call ep_dispatch() or "
            "ep_dispatch_wait() first."
        )

        dispatch_dim = self._dispatch_dim[device_id]
        assert dispatch_dim is not None, (
            "Dispatch dimension is not set, you should call ep_dispatch() or "
            "ep_dispatch_async() first."
        )

        results = call_ep_combine(
            input_tokens,
            src_info,
            self.atomic_counters[0][device_id],
            self.send_buf_ptrs[COMBINE_GROUP],
            self.recv_buf_ptrs[COMBINE_GROUP],
            self.recv_count_ptrs[COMBINE_GROUP],
            self.config,
            dispatch_dim,
            router_weight,
            topk_ids=topk_ids,
        )

        # Reset src_info to None to avoid reusing it for the next batch
        self._src_info[device_id] = None

        return results


class EPCommInitializer:
    """Helper class for initializing buffers for Expert Parallelism (EP).

    This class handles the initialization of the SHMEM communication
    infrastructure required for Expert Parallelism. It creates and manages
    atomic counters, initializes the SHMEM library, and allocates symmetric
    memory buffers.
    """

    config: EPConfig
    """EP configuration."""

    init_model: Model
    """Compiled model that sets up the SHMEM library context for local GPUs and
    allocates the SHMEM memory for the send, receive, and receive count buffers."""

    send_buf_ptrs: list[Buffer]
    """List of device pointers for the send buffer."""

    recv_buf_ptrs: list[Buffer]
    """List of device pointers for the receive buffer."""

    recv_count_ptrs: list[Buffer]
    """List of device pointers for the receive count buffer."""

    atomic_counters: list[Buffer]
    """List of atomic counters used for synchronization."""

    def __init__(self, config: EPConfig):
        """Initialize the EP communication initializer.

        Args:
            config: EP configuration.
        """
        self.config = config
        n_experts = (
            config.n_experts // config.n_gpus_per_node
            if config.use_allreduce
            else config.n_experts
        )
        # Allocated based on the EPLocalSyncCounters struct in ep_comm.mojo
        self.atomic_counter_size = get_ep_local_sync_counters_size(n_experts)

        # Create atomic counters for each GPU in each buffer group
        self.atomic_counters = [
            Buffer(
                DType.int32,
                [self.atomic_counter_size],
                device=Accelerator(i % self.config.n_gpus_per_node),
            )
            for i in range(NUM_GROUPS * self.config.n_gpus_per_node)
        ]

    def _build_ep_init_graph(self) -> Graph:
        """Build the computation graph for EP initialization.

        Creates a graph that initializes SHMEM context and allocates symmetric
        memory buffers on each GPU. The graph takes atomic counter buffers as
        input and returns device pointers to allocated SHMEM buffers.

        Returns:
            Graph: Computation graph for EP initialization.
        """
        atomic_counter_shape = self.atomic_counters[0].shape
        with Graph(
            "ep_init",
            input_types=[
                BufferType(
                    DType.int32,
                    atomic_counter_shape,
                    device=DeviceRef.GPU(i % self.config.n_gpus_per_node),
                )
                for i in range(NUM_GROUPS * self.config.n_gpus_per_node)
            ],
        ) as g:
            dev_ptrs_list: list[TensorValue] = []
            my_rank_list: list[TensorValue] = []

            # Initialize SHMEM context and allocate buffers for each GPU
            for i in range(self.config.n_gpus_per_node):
                # Get atomic counter buffers for both groups
                atomic_counter_group_0 = g.inputs[i].buffer
                atomic_counter_group_1 = g.inputs[
                    i + self.config.n_gpus_per_node
                ].buffer

                # Call the custom EP initialization kernel
                dev_ptrs, my_rank = call_ep_init(
                    atomic_counter_group_0, atomic_counter_group_1, self.config
                )
                # Device pointers cannot be output as CPU tensors since the InferenceSession
                # may not be initialized with CPU; moved to the device as a workaround.
                dev_ptrs_list.append(dev_ptrs.to(atomic_counter_group_0.device))

                my_rank_list.append(my_rank)

            my_ranks = ops.concat(my_rank_list, axis=0)

            g.output(*dev_ptrs_list, my_ranks.to(DeviceRef.GPU(0)))
        return g

    def ep_init(self, session: InferenceSession) -> None:
        """Initialize Expert Parallelism communication infrastructure.

        Args:
            session: Inference session used to compile and execute the graph.
        """
        enable_all_peer_access()
        logger.info("Initializing EP communication infrastructure...")
        logger.info(
            f"Estimated EP memory usage per device: {to_human_readable_bytes(self.config.estimate_memory_usage())}"
        )

        def _get_max_rcs() -> int:
            """Return the maximum number of warps in a block."""
            if accelerator_api() == "cuda":
                return 32
            elif accelerator_api() == "hip":
                return 16
            else:
                raise ValueError(
                    f"Unsupported accelerator API: {accelerator_api()}"
                )

        # Skip setting NVSHMEM-specific env vars on single node.
        if self.config.n_nodes > 1 or os.getenv("NVSHMEM_DISABLE_P2P") == "1":
            # Set ENVs for NVSHMEM
            n_gpus = self.config.n_nodes * self.config.n_gpus_per_node
            num_experts_per_gpu = self.config.n_experts // n_gpus
            n_rcs = min(num_experts_per_gpu, _get_max_rcs())
            os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "1"
            os.environ["NVSHMEM_IBGDA_NIC_HANDLER"] = "gpu"
            os.environ["NVSHMEM_IBGDA_RC_MAP_BY"] = "warp"
            os.environ["NVSHMEM_IBGDA_NUM_RC_PER_PE"] = str(n_rcs)

            # TODO: Provide a way to let user manually map NICs to different GPU
            os.environ["NVSHMEM_ENABLE_NIC_PE_MAPPING"] = "1"

        # Build and compile the initialization graph
        graph = self._build_ep_init_graph()
        self.init_model = session.load(graph)

        # Execute the graph to initialize SHMEM and get device pointers
        all_outputs = self.init_model.execute(*self.atomic_counters)
        all_outputs_np: list[npt.NDArray[Any]] = []
        for dev_ptr in all_outputs:
            assert isinstance(dev_ptr, Buffer)
            all_outputs_np.append(dev_ptr.to_numpy())

        # Process the output device pointers:
        # Each device returns a tensor of shape (NUM_GROUPS, 3) where:
        # - NUM_GROUPS of buffers for EP communication
        # - 3 corresponds to: [send_buffer_ptr, recv_buffer_ptr, recv_count_ptr]
        # We reorganize these pointers by buffer type and group for easy access.

        # Reorganize device pointers by buffer type and group
        send_buf_ptrs_np: list[npt.NDArray[Any]] = []
        recv_buf_ptrs_np: list[npt.NDArray[Any]] = []
        recv_count_ptrs_np: list[npt.NDArray[Any]] = []

        for group_idx in range(NUM_GROUPS):
            # Collect pointers from all devices for this group
            curr_group_list: list[npt.NDArray[Any]] = []
            for device_idx in range(self.config.n_gpus_per_node):
                curr_group_list.append(all_outputs_np[device_idx][group_idx])
            curr_group_ptrs = np.stack(curr_group_list, axis=0)

            # Extract pointers by buffer type (send, recv, recv_count)
            send_buf_ptrs_np.append(curr_group_ptrs[:, 0])
            recv_buf_ptrs_np.append(curr_group_ptrs[:, 1])
            recv_count_ptrs_np.append(curr_group_ptrs[:, 2])

        self.send_buf_ptrs = [
            Buffer.from_numpy(dev_ptr) for dev_ptr in send_buf_ptrs_np
        ]
        self.recv_buf_ptrs = [
            Buffer.from_numpy(dev_ptr) for dev_ptr in recv_buf_ptrs_np
        ]
        self.recv_count_ptrs = [
            Buffer.from_numpy(dev_ptr) for dev_ptr in recv_count_ptrs_np
        ]

        # The last element is the my_ranks tensor
        my_ranks_np = all_outputs_np[-1]
        my_node_id = my_ranks_np // self.config.n_gpus_per_node

        # check if all GPUs in the same node have the same node_id
        if not np.all(my_node_id == my_node_id[0]):
            raise ValueError(
                "All GPUs in the same node must have the same node ID."
            )
        self.config.node_id = my_node_id[0]

        logger.info(f"Initialized EP for node {self.config.node_id}")
        if self.config.use_allreduce:
            logger.info("Using allreduce as the EP communication backend.")

    def model_inputs(self) -> list[Buffer]:
        """Get the model inputs for the MoE model.

        Returns:
            list[Buffer]: List of all tensors needed as model inputs.
        """
        return (
            self.atomic_counters
            + self.send_buf_ptrs
            + self.recv_buf_ptrs
            + self.recv_count_ptrs
        )
