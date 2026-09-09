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
"""Multi-device Expert Parallelism graph ops (dispatch and combine)."""

from __future__ import annotations

from typing import Any

from max._core.dialects import mo
from max._core.dialects.builtin import (
    BoolAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
)

from ..graph import Graph
from ..type import _ChainType
from ..value import BufferValue, TensorType, TensorValue, Value


def _ep_common_attrs(
    hidden_size: int,
    top_k: int,
    n_experts: int,
    max_token_per_rank: int,
    n_gpus_per_node: int,
    n_nodes: int,
    fused_shared_expert: bool,
) -> list[Any]:
    """Build the common integer/bool attribute list for EP ops."""
    i64 = IntegerType(64)
    return [
        IntegerAttr(i64, hidden_size),
        IntegerAttr(i64, top_k),
        IntegerAttr(i64, n_experts),
        IntegerAttr(i64, max_token_per_rank),
        IntegerAttr(i64, n_gpus_per_node),
        IntegerAttr(i64, n_nodes),
        BoolAttr(fused_shared_expert),
    ]


def _unpack_results(
    results: list[Value[Any]], num_devices: int, num_groups: int
) -> list[tuple[TensorValue, ...]]:
    """Unpack flat results into per-device tuples."""
    n = num_devices
    per_device: list[tuple[TensorValue, ...]] = []
    for i in range(n):
        per_device.append(
            tuple(results[g * n + i].tensor for g in range(num_groups))
        )
    return per_device


# ---------------------------------------------------------------------------
# BF16 Dispatch
# ---------------------------------------------------------------------------

_BF16_OUTPUT_GROUPS = 4


def dispatch_bf16(
    input_tokens: list[TensorValue],
    topk_ids: list[TensorValue],
    send_ptrs: TensorValue,
    recv_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    atomic_counters: list[BufferValue],
    output_types_per_device: list[list[TensorType]],
    *,
    hidden_size: int,
    top_k: int,
    n_experts: int,
    max_token_per_rank: int,
    n_gpus_per_node: int,
    n_nodes: int,
    fused_shared_expert: bool,
) -> list[tuple[TensorValue, ...]]:
    """Multi-device EP BF16 dispatch.

    Returns per-device tuples of 4 tensors:
    (output_tokens, row_offsets, expert_ids, src_info).
    """
    num_devices = len(input_tokens)
    if num_devices == 0:
        raise ValueError("input_tokens must be non-empty")

    output_tokens_types = [t[0] for t in output_types_per_device]
    row_offsets_types = [t[1] for t in output_types_per_device]
    expert_ids_types = [t[2] for t in output_types_per_device]
    src_info_types = [t[3] for t in output_types_per_device]

    counters = [BufferValue(c) for c in atomic_counters]

    graph = Graph.current
    devices = [t.device for t in input_tokens]
    in_chain = graph.device_chains.merge_for(devices)

    *results, out_chain = graph._add_op_generated(
        mo.DistributedEpDispatchOp,
        output_tokens_types,
        row_offsets_types,
        expert_ids_types,
        src_info_types,
        _ChainType(),
        input_tokens,
        topk_ids,
        [send_ptrs] * num_devices,
        [recv_ptrs] * num_devices,
        [recv_count_ptrs] * num_devices,
        counters,
        in_chain,
        *_ep_common_attrs(
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
        ),
    )

    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    return _unpack_results(results, num_devices, _BF16_OUTPUT_GROUPS)


# ---------------------------------------------------------------------------
# FP8 Dispatch
# ---------------------------------------------------------------------------

_FP8_OUTPUT_GROUPS = 5


def dispatch_fp8(
    input_tokens: list[TensorValue],
    topk_ids: list[TensorValue],
    send_ptrs: TensorValue,
    recv_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    atomic_counters: list[BufferValue],
    output_types_per_device: list[list[TensorType]],
    *,
    hidden_size: int,
    top_k: int,
    n_experts: int,
    max_token_per_rank: int,
    n_gpus_per_node: int,
    n_nodes: int,
    fused_shared_expert: bool,
    dispatch_scale_granularity: str,
) -> list[tuple[TensorValue, ...]]:
    """Multi-device EP FP8 dispatch.

    Returns per-device tuples of 5 tensors:
    (output_tokens, output_scales, row_offsets, expert_ids, src_info).
    """
    num_devices = len(input_tokens)
    if num_devices == 0:
        raise ValueError("input_tokens must be non-empty")

    output_tokens_types = [t[0] for t in output_types_per_device]
    output_scales_types = [t[1] for t in output_types_per_device]
    row_offsets_types = [t[2] for t in output_types_per_device]
    expert_ids_types = [t[3] for t in output_types_per_device]
    src_info_types = [t[4] for t in output_types_per_device]

    counters = [BufferValue(c) for c in atomic_counters]

    graph = Graph.current
    devices = [t.device for t in input_tokens]
    in_chain = graph.device_chains.merge_for(devices)

    *results, out_chain = graph._add_op_generated(
        mo.DistributedEpDispatchFp8Op,
        output_tokens_types,
        output_scales_types,
        row_offsets_types,
        expert_ids_types,
        src_info_types,
        _ChainType(),
        input_tokens,
        topk_ids,
        [send_ptrs] * num_devices,
        [recv_ptrs] * num_devices,
        [recv_count_ptrs] * num_devices,
        counters,
        in_chain,
        *_ep_common_attrs(
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
        ),
        StringAttr(dispatch_scale_granularity),
    )

    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    return _unpack_results(results, num_devices, _FP8_OUTPUT_GROUPS)


# ---------------------------------------------------------------------------
# NVFP4/MXFP4/MXFP8 Dispatch
# ---------------------------------------------------------------------------

_BLOCK_SCALED_NV_OUTPUT_GROUPS = 6


def dispatch_block_scaled_nv(
    input_tokens: list[TensorValue],
    topk_ids: list[TensorValue],
    send_ptrs: TensorValue,
    recv_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    input_scales: list[TensorValue],
    atomic_counters: list[BufferValue],
    output_types_per_device: list[list[TensorType]],
    *,
    hidden_size: int,
    top_k: int,
    n_experts: int,
    max_token_per_rank: int,
    n_gpus_per_node: int,
    n_nodes: int,
    fused_shared_expert: bool,
) -> list[tuple[TensorValue, ...]]:
    """Multi-device EP NVIDIA block-scaled dispatch.

    Returns per-device tuples of 6 tensors:
    (output_tokens, output_scales, row_offsets, scales_offsets,
    expert_ids, src_info).
    """
    num_devices = len(input_tokens)
    if num_devices == 0:
        raise ValueError("input_tokens must be non-empty")

    output_tokens_types = [t[0] for t in output_types_per_device]
    output_scales_types = [t[1] for t in output_types_per_device]
    row_offsets_types = [t[2] for t in output_types_per_device]
    scales_offsets_types = [t[3] for t in output_types_per_device]
    expert_ids_types = [t[4] for t in output_types_per_device]
    src_info_types = [t[5] for t in output_types_per_device]

    counters = [BufferValue(c) for c in atomic_counters]

    graph = Graph.current
    devices = [t.device for t in input_tokens]
    in_chain = graph.device_chains.merge_for(devices)

    *results, out_chain = graph._add_op_generated(
        mo.DistributedEpDispatchBlockScaledNvOp,
        output_tokens_types,
        output_scales_types,
        row_offsets_types,
        scales_offsets_types,
        expert_ids_types,
        src_info_types,
        _ChainType(),
        input_tokens,
        topk_ids,
        [send_ptrs] * num_devices,
        [recv_ptrs] * num_devices,
        [recv_count_ptrs] * num_devices,
        input_scales,
        counters,
        in_chain,
        *_ep_common_attrs(
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
        ),
    )

    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    return _unpack_results(results, num_devices, _BLOCK_SCALED_NV_OUTPUT_GROUPS)


# ---------------------------------------------------------------------------
# MXFP4 Dispatch
# ---------------------------------------------------------------------------

_MXFP4_OUTPUT_GROUPS = 5


def dispatch_mxfp4(
    input_tokens: list[TensorValue],
    topk_ids: list[TensorValue],
    send_ptrs: TensorValue,
    recv_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    atomic_counters: list[BufferValue],
    output_types_per_device: list[list[TensorType]],
    *,
    hidden_size: int,
    top_k: int,
    n_experts: int,
    max_token_per_rank: int,
    n_gpus_per_node: int,
    n_nodes: int,
    fused_shared_expert: bool,
    fuse_a_scale_preshuffle: bool,
    max_padded_m: int,
    mx_format: str = "auto",
) -> list[tuple[TensorValue, ...]]:
    """Multi-device EP MXFP4 dispatch.

    Returns per-device tuples of 5 tensors:
    (output_tokens, output_scales, row_offsets, expert_ids, src_info).
    """
    num_devices = len(input_tokens)
    if num_devices == 0:
        raise ValueError("input_tokens must be non-empty")

    output_tokens_types = [t[0] for t in output_types_per_device]
    output_scales_types = [t[1] for t in output_types_per_device]
    row_offsets_types = [t[2] for t in output_types_per_device]
    expert_ids_types = [t[3] for t in output_types_per_device]
    src_info_types = [t[4] for t in output_types_per_device]

    counters = [BufferValue(c) for c in atomic_counters]

    graph = Graph.current
    devices = [t.device for t in input_tokens]
    in_chain = graph.device_chains.merge_for(devices)

    *results, out_chain = graph._add_op_generated(
        mo.DistributedEpDispatchMxfp4Op,
        output_tokens_types,
        output_scales_types,
        row_offsets_types,
        expert_ids_types,
        src_info_types,
        _ChainType(),
        input_tokens,
        topk_ids,
        [send_ptrs] * num_devices,
        [recv_ptrs] * num_devices,
        [recv_count_ptrs] * num_devices,
        counters,
        in_chain,
        *_ep_common_attrs(
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
        ),
        BoolAttr(fuse_a_scale_preshuffle),
        IntegerAttr(IntegerType(64), max_padded_m),
        StringAttr(mx_format),
    )

    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    return _unpack_results(results, num_devices, _MXFP4_OUTPUT_GROUPS)


# ---------------------------------------------------------------------------
# Combine (fused async + wait, supports output epilogue fusion)
# ---------------------------------------------------------------------------


def combine(
    input_tokens: list[TensorValue],
    src_info: list[TensorValue],
    send_ptrs: TensorValue,
    recv_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    router_weights: list[TensorValue],
    atomic_counters: list[BufferValue],
    output_types: list[TensorType],
    *,
    hidden_size: int,
    top_k: int,
    n_experts: int,
    max_token_per_rank: int,
    n_gpus_per_node: int,
    n_nodes: int,
    fused_shared_expert: bool,
) -> list[TensorValue]:
    """Multi-device EP fused combine with output epilogue fusion.

    Args:
        input_tokens: Per-device expert output tokens.
        src_info: Per-device source routing info from dispatch.
        send_ptrs: Host tensor of send buffer pointers.
        recv_ptrs: Host tensor of receive buffer pointers.
        recv_count_ptrs: Host tensor of receive count pointers.
        router_weights: Per-device router weight tensors.
        atomic_counters: Per-device synchronization buffers.
        output_types: Per-device output tensor types (one per device).
        hidden_size: Model hidden dimension size.
        top_k: Number of top experts selected per token.
        n_experts: Total number of experts.
        max_token_per_rank: Maximum tokens per rank.
        n_gpus_per_node: Number of GPUs per node.
        n_nodes: Number of nodes.
        fused_shared_expert: Whether the shared expert is fused.

    Returns:
        Per-device combined output tensors.
    """
    num_devices = len(input_tokens)
    if num_devices == 0:
        raise ValueError("input_tokens must be non-empty")

    counters = [BufferValue(c) for c in atomic_counters]

    graph = Graph.current
    devices = [t.device for t in input_tokens]
    in_chain = graph.device_chains.merge_for(devices)

    *results, out_chain = graph._add_op_generated(
        mo.DistributedEpCombineOp,
        output_types,
        _ChainType(),
        input_tokens,
        src_info,
        [send_ptrs] * num_devices,
        [recv_ptrs] * num_devices,
        [recv_count_ptrs] * num_devices,
        router_weights,
        counters,
        in_chain,
        *_ep_common_attrs(
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
        ),
    )

    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    return [r.tensor for r in results]
