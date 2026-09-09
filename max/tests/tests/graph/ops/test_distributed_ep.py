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
"""Tests for multi-device EP dispatch and combine graph ops."""

from typing import Any

import pytest
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, Type, ops

NUM_DEVICES = 2
HIDDEN_SIZE = 128
TOP_K = 8
N_EXPERTS = 64
MAX_TOKEN_PER_RANK = 64
N_RANKS = NUM_DEVICES
N_LOCAL_EXPERTS = N_EXPERTS // N_RANKS
MAX_RECV_TOKENS = MAX_TOKEN_PER_RANK * min(N_EXPERTS, N_RANKS * TOP_K)
NUM_TOKENS = 32
DEVICES = [DeviceRef.GPU(id=i) for i in range(NUM_DEVICES)]


def _host_ptr_types() -> list[Type[Any]]:
    return [
        TensorType(
            dtype=DType.uint64, shape=[NUM_DEVICES], device=DeviceRef("cpu")
        )
        for _ in range(3)
    ]


def _per_device_types(dtype: DType, shape: list[int]) -> list[Type[Any]]:
    return [TensorType(dtype=dtype, shape=shape, device=dev) for dev in DEVICES]


def _buffer_types() -> list[Type[Any]]:
    return [
        BufferType(dtype=DType.int32, shape=[N_EXPERTS * 4], device=dev)
        for dev in DEVICES
    ]


def _common_kwargs() -> dict[str, Any]:
    return {
        "hidden_size": HIDDEN_SIZE,
        "top_k": TOP_K,
        "n_experts": N_EXPERTS,
        "max_token_per_rank": MAX_TOKEN_PER_RANK,
        "n_gpus_per_node": NUM_DEVICES,
        "n_nodes": 1,
        "fused_shared_expert": False,
    }


# -----------------------------------------------------------------------
# BF16 Dispatch
# -----------------------------------------------------------------------


def test_dispatch_bf16_basic() -> None:
    """BF16 dispatch produces 4 output groups per device."""
    input_types: list[Type[Any]] = [
        *_per_device_types(DType.bfloat16, [NUM_TOKENS, HIDDEN_SIZE]),
        *_per_device_types(DType.int32, [NUM_TOKENS, TOP_K]),
        *_host_ptr_types(),
        *_buffer_types(),
    ]

    output_types_per_device = [
        [
            TensorType(
                dtype=DType.bfloat16,
                shape=[MAX_RECV_TOKENS, HIDDEN_SIZE],
                device=dev,
            ),
            TensorType(
                dtype=DType.uint32,
                shape=[N_LOCAL_EXPERTS + 1],
                device=dev,
            ),
            TensorType(dtype=DType.int32, shape=[N_LOCAL_EXPERTS], device=dev),
            TensorType(
                dtype=DType.int32,
                shape=[MAX_RECV_TOKENS, 2],
                device=dev,
            ),
        ]
        for dev in DEVICES
    ]

    n = NUM_DEVICES
    with Graph("bf16_dispatch", input_types=input_types) as graph:
        idx = 0
        input_tokens = [graph.inputs[idx + i].tensor for i in range(n)]
        idx += n
        topk_ids = [graph.inputs[idx + i].tensor for i in range(n)]
        idx += n
        send_ptrs = graph.inputs[idx].tensor
        recv_ptrs = graph.inputs[idx + 1].tensor
        recv_count_ptrs = graph.inputs[idx + 2].tensor
        idx += 3
        counters = [graph.inputs[idx + i].buffer for i in range(n)]

        results = ops.distributed_ep.dispatch_bf16(
            input_tokens,
            topk_ids,
            send_ptrs,
            recv_ptrs,
            recv_count_ptrs,
            counters,
            output_types_per_device,
            **_common_kwargs(),
        )

        assert len(results) == NUM_DEVICES
        assert all(len(t) == 4 for t in results)

        flat: list[Any] = []
        for per_device in results:
            flat.extend(per_device)
        graph.output(*flat)

    assert len(graph.output_types) == NUM_DEVICES * 4


# -----------------------------------------------------------------------
# FP8 Dispatch
# -----------------------------------------------------------------------


def test_dispatch_fp8_basic() -> None:
    """FP8 dispatch produces 5 output groups per device."""
    input_types: list[Type[Any]] = [
        *_per_device_types(DType.bfloat16, [NUM_TOKENS, HIDDEN_SIZE]),
        *_per_device_types(DType.int32, [NUM_TOKENS, TOP_K]),
        *_host_ptr_types(),
        *_buffer_types(),
    ]

    output_types_per_device = [
        [
            TensorType(
                dtype=DType.float8_e4m3fn,
                shape=[MAX_RECV_TOKENS, HIDDEN_SIZE],
                device=dev,
            ),
            TensorType(
                dtype=DType.float32,
                shape=[MAX_RECV_TOKENS, HIDDEN_SIZE],
                device=dev,
            ),
            TensorType(
                dtype=DType.uint32,
                shape=[N_LOCAL_EXPERTS + 1],
                device=dev,
            ),
            TensorType(dtype=DType.int32, shape=[N_LOCAL_EXPERTS], device=dev),
            TensorType(
                dtype=DType.int32,
                shape=[MAX_RECV_TOKENS, 2],
                device=dev,
            ),
        ]
        for dev in DEVICES
    ]

    n = NUM_DEVICES
    with Graph("fp8_dispatch", input_types=input_types) as graph:
        idx = 0
        input_tokens = [graph.inputs[idx + i].tensor for i in range(n)]
        idx += n
        topk_ids = [graph.inputs[idx + i].tensor for i in range(n)]
        idx += n
        send_ptrs = graph.inputs[idx].tensor
        recv_ptrs = graph.inputs[idx + 1].tensor
        recv_count_ptrs = graph.inputs[idx + 2].tensor
        idx += 3
        counters = [graph.inputs[idx + i].buffer for i in range(n)]

        results = ops.distributed_ep.dispatch_fp8(
            input_tokens,
            topk_ids,
            send_ptrs,
            recv_ptrs,
            recv_count_ptrs,
            counters,
            output_types_per_device,
            dispatch_scale_granularity="blockwise",
            **_common_kwargs(),
        )

        assert len(results) == NUM_DEVICES
        assert all(len(t) == 5 for t in results)

        flat: list[Any] = []
        for per_device in results:
            flat.extend(per_device)
        graph.output(*flat)

    assert len(graph.output_types) == NUM_DEVICES * 5


# -----------------------------------------------------------------------
# NVFP4 Dispatch
# -----------------------------------------------------------------------


def test_dispatch_block_scaled_nv_basic() -> None:
    """NVFP4 dispatch produces 6 output groups per device."""
    token_last_dim = HIDDEN_SIZE // 2
    input_types: list[Type[Any]] = [
        *_per_device_types(DType.bfloat16, [NUM_TOKENS, HIDDEN_SIZE]),
        *_per_device_types(DType.int32, [NUM_TOKENS, TOP_K]),
        *_host_ptr_types(),
        *_per_device_types(DType.float32, [1]),
        *_buffer_types(),
    ]

    output_types_per_device = [
        [
            TensorType(
                dtype=DType.uint8,
                shape=[MAX_RECV_TOKENS, token_last_dim],
                device=dev,
            ),
            TensorType(
                dtype=DType.uint8,
                shape=[MAX_RECV_TOKENS, HIDDEN_SIZE, 1, 1, 1],
                device=dev,
            ),
            TensorType(
                dtype=DType.uint32, shape=[N_LOCAL_EXPERTS + 1], device=dev
            ),
            TensorType(dtype=DType.uint32, shape=[N_LOCAL_EXPERTS], device=dev),
            TensorType(dtype=DType.int32, shape=[N_LOCAL_EXPERTS], device=dev),
            TensorType(
                dtype=DType.int32, shape=[MAX_RECV_TOKENS, 2], device=dev
            ),
        ]
        for dev in DEVICES
    ]

    n = NUM_DEVICES
    with Graph("nvfp4_dispatch", input_types=input_types) as graph:
        idx = 0
        input_tokens = [graph.inputs[idx + i].tensor for i in range(n)]
        idx += n
        topk_ids = [graph.inputs[idx + i].tensor for i in range(n)]
        idx += n
        send_ptrs = graph.inputs[idx].tensor
        recv_ptrs = graph.inputs[idx + 1].tensor
        recv_count_ptrs = graph.inputs[idx + 2].tensor
        idx += 3
        input_scales = [graph.inputs[idx + i].tensor for i in range(n)]
        idx += n
        counters = [graph.inputs[idx + i].buffer for i in range(n)]

        results = ops.distributed_ep.dispatch_block_scaled_nv(
            input_tokens,
            topk_ids,
            send_ptrs,
            recv_ptrs,
            recv_count_ptrs,
            input_scales,
            counters,
            output_types_per_device,
            **_common_kwargs(),
        )

        assert len(results) == NUM_DEVICES
        assert all(len(t) == 6 for t in results)

        flat: list[Any] = []
        for per_device in results:
            flat.extend(per_device)
        graph.output(*flat)

    assert len(graph.output_types) == NUM_DEVICES * 6


# -----------------------------------------------------------------------
# Combine
# -----------------------------------------------------------------------


def test_combine_basic() -> None:
    """Combine produces 1 output per device."""
    input_types: list[Type[Any]] = [
        *_per_device_types(DType.bfloat16, [MAX_RECV_TOKENS, HIDDEN_SIZE]),
        *_per_device_types(DType.int32, [MAX_RECV_TOKENS, 2]),
        *_host_ptr_types(),
        *_per_device_types(DType.float32, [NUM_TOKENS, TOP_K]),
        *_buffer_types(),
    ]

    output_types = [
        TensorType(
            dtype=DType.bfloat16,
            shape=[NUM_TOKENS, HIDDEN_SIZE],
            device=dev,
        )
        for dev in DEVICES
    ]

    n = NUM_DEVICES
    with Graph("ep_combine", input_types=input_types) as graph:
        idx = 0
        input_tokens = [graph.inputs[idx + i].tensor for i in range(n)]
        idx += n
        src_info = [graph.inputs[idx + i].tensor for i in range(n)]
        idx += n
        send_ptrs = graph.inputs[idx].tensor
        recv_ptrs = graph.inputs[idx + 1].tensor
        recv_count_ptrs = graph.inputs[idx + 2].tensor
        idx += 3
        router_weights = [graph.inputs[idx + i].tensor for i in range(n)]
        idx += n
        counters = [graph.inputs[idx + i].buffer for i in range(n)]

        results = ops.distributed_ep.combine(
            input_tokens,
            src_info,
            send_ptrs,
            recv_ptrs,
            recv_count_ptrs,
            router_weights,
            counters,
            output_types,
            **_common_kwargs(),
        )

        assert len(results) == NUM_DEVICES
        for i, tensor in enumerate(results):
            assert tensor.device == DEVICES[i]

        graph.output(*results)

    assert len(graph.output_types) == NUM_DEVICES


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------


def test_dispatch_empty_inputs() -> None:
    """Empty input_tokens raises ValueError for all dispatch variants."""
    with pytest.raises(ValueError, match="input_tokens must be non-empty"):
        with Graph("empty", input_types=[]):
            ops.distributed_ep.dispatch_bf16(
                [],
                [],
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                [],
                [],
                **_common_kwargs(),
            )

    with pytest.raises(ValueError, match="input_tokens must be non-empty"):
        with Graph("empty", input_types=[]):
            ops.distributed_ep.combine(
                [],
                [],
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                [],
                [],
                [],
                **_common_kwargs(),
            )
