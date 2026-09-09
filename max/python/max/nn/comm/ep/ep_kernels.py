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

"""Expert Parallelism (EP) Communication Kernels.

This file contains the kernels for Expert Parallelism (EP) communication.
"""

from __future__ import annotations

from typing import Any, Final

from max.driver import accelerator_api
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Dim,
    Shape,
    StaticDim,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.support.math import ceildiv

from ...quant_config import QuantConfig
from .ep_config import NUM_GROUPS, EPConfig

# With NVFP4 format, each expert's scales need to be padded to the nearest
# multiple of NVFP4_MN_GROUP_SIZE.
NVFP4_MN_GROUP_SIZE = 128


def _uses_block_scaled_nv_ep_layout(config: EPConfig) -> bool:
    quant_config = config.dispatch_quant_config
    return (
        quant_config is not None
        and accelerator_api() == "cuda"
        and (quant_config.is_nvfp4 or quant_config.is_mxfp8)
    )


def _is_legacy_float8_dispatch(config: EPConfig) -> bool:
    quant_config = config.dispatch_quant_config
    return (
        quant_config is not None
        and config.dispatch_dtype.is_float8()
        # MXFP8 is block-scaled, never legacy per-tensor float8. Without this it
        # would silently take the legacy path on AMD, where
        # `_uses_block_scaled_nv_ep_layout` is cuda-only.
        and not quant_config.is_mxfp8
        and not _uses_block_scaled_nv_ep_layout(config)
    )


def uses_mx_ep_token_format(
    config: EPConfig, quant_config: QuantConfig | None = None
) -> bool:
    """Whether dispatch goes through ``MXTokenFormat``, which serves both MX
    formats: it takes its element packing from the dispatch dtype (two FP4
    nibbles per byte, or one E4M3 byte) and the E8M0 scale path is identical.

    On cuda, MXFP8 uses the NVIDIA block-scaled layout instead, so that case is
    excluded here rather than left to call-site ordering.

    Args:
        config: Supplies the accelerator layout term, and the format term when
            ``quant_config`` is omitted.
        quant_config: Takes the format term from here instead, so a caller whose
            consumer gate reads its own :obj:`QuantConfig` cannot disagree with
            the producer about the format.
    """
    if quant_config is None:
        quant_config = config.dispatch_quant_config
    if quant_config is None:
        return False
    return (
        quant_config.is_mxfp4 or quant_config.is_mxfp6 or quant_config.is_mxfp8
    ) and not _uses_block_scaled_nv_ep_layout(config)


_MX_FORMAT_NAME: Final[dict[DType, str]] = {
    DType.float8_e4m3fn: "mxfp8",
    DType.float8_e5m2: "mxfp8_e5m2",
    DType.float6_e2m3fn: "mxfp6",
    DType.float6_e3m2fn: "mxfp6_e3m2",
    DType.float4_e2m1fn: "mxfp4",
}
_MX_FORMAT_AUTO: Final[str] = "auto"


def _mx_format_name(quant_config: QuantConfig | None) -> str:
    """Returns the ``MXFormat`` name for this config's element encoding.

    Falls back to ``auto`` when the quantized dtype already identifies the
    format, which keeps the kernel on its dtype inference.
    """
    if quant_config is None or not quant_config.is_mx:
        return _MX_FORMAT_AUTO
    return _MX_FORMAT_NAME.get(quant_config.mx_element_dtype, _MX_FORMAT_AUTO)


def _add_mx_format_parameter(
    parameters: dict[str, Any], config: EPConfig
) -> None:
    """Names the MX operand format when the packed dtype cannot.

    MXFP4 and MXFP8 are told apart by their quantized dtype, so they leave this
    unset and the kernel infers as before. MXFP6 travels as ``uint8`` with E8M0
    scales -- byte for byte the same as MXFP4 -- so it has to be named.
    """
    quant_config = config.dispatch_quant_config
    if quant_config is not None and quant_config.is_mxfp6:
        parameters["MX_FORMAT"] = _mx_format_name(quant_config)


def ep_mxfp4_max_padded_m(config: EPConfig) -> int:
    """Per-expert ``scale_4d`` slot stride in rows for the MXFP4 A-scale
    preshuffle fold (= ``align_up(max_recv_tokens_per_expert, 32)``). 0 when the
    fold is off. Single source of truth shared by the dispatch producer
    (``ep_wait`` up-proj / ``fused_silu`` down-proj) and the matmul reader
    (``a_scales_max_padded_m``)."""
    if not config.mxfp4_a_scales_preshuffled:
        return 0
    return ep_mxfp4_down_slot_stride(config)


def ep_mxfp4_down_slot_stride(config: EPConfig) -> int:
    """Raw per-expert ``scale_4d`` slot stride
    (= ``align_up(max_recv_tokens_per_expert, 32)``) for the LOCAL SwiGLU
    down-proj A-scale fold (``fused_silu`` writes it, the down matmul reads it).
    Unconditional and independent of the distributed up-proj fold
    (``mxfp4_a_scales_preshuffled``), so it engages on the distributed-dispatch
    path where that fold is off. ``ep_mxfp4_max_padded_m`` returns this gated on
    that flag; the caller here gates on applicability."""
    n_ranks = config.n_gpus_per_node * config.n_nodes
    max_recv_per_expert = config.max_tokens_per_rank * n_ranks
    return ceildiv(max_recv_per_expert, 32) * 32


def _ep_dispatch_output_types(
    config: EPConfig,
    device_ref: DeviceRef,
) -> list[TensorType]:
    """Returns the output types for the EP dispatch kernel."""
    n_ranks = config.n_gpus_per_node * config.n_nodes
    n_local_experts = config.n_experts // n_ranks

    max_recv_tokens = config.get_max_recv_tokens()

    if config.fused_shared_expert:
        max_recv_tokens += config.max_tokens_per_rank
        n_local_experts += 1

    token_last_dim = config.hidden_size
    if config.dispatch_quant_config is not None:
        if config.dispatch_quant_config.is_fp4:
            token_last_dim //= 2
        elif config.dispatch_quant_config.is_mxfp6:
            token_last_dim = token_last_dim * 3 // 4

    output_tokens_type = TensorType(
        dtype=config.dispatch_dtype,
        shape=[max_recv_tokens, token_last_dim],
        device=device_ref,
    )
    expert_start_indices_type = TensorType(
        dtype=DType.uint32,
        shape=[n_local_experts + 1],
        device=device_ref,
    )
    expert_ids_type = TensorType(
        dtype=DType.int32,
        shape=[n_local_experts],
        device=device_ref,
    )
    src_info_type = TensorType(
        dtype=DType.int32,
        shape=[max_recv_tokens, 2],
        device=device_ref,
    )

    if config.dispatch_quant_config is not None:
        quant_config = config.dispatch_quant_config

        if _uses_block_scaled_nv_ep_layout(config):
            # NVIDIA block-scaled formats produce padded 5D scale tiles plus
            # offsets mapping each expert's token rows to those padded tiles.
            scales_offsets_type = TensorType(
                dtype=DType.uint32,
                shape=[n_local_experts],
                device=device_ref,
            )
            padded_scales_tokens = (
                max_recv_tokens + n_local_experts * NVFP4_MN_GROUP_SIZE
            )
            padded_scales_type = quant_config.quantized_scales_type(
                Shape([padded_scales_tokens, config.hidden_size]), device_ref
            )
            return [
                output_tokens_type,
                padded_scales_type,
                expert_start_indices_type,
                scales_offsets_type,
                expert_ids_type,
                src_info_type,
            ]
        elif _is_legacy_float8_dispatch(config):
            out_scales_type = quant_config.quantized_scales_type(
                Shape([max_recv_tokens, config.hidden_size]), device_ref
            )
            return [
                output_tokens_type,
                out_scales_type,
                expert_start_indices_type,
                expert_ids_type,
                src_info_type,
            ]
        elif uses_mx_ep_token_format(config):
            # When the A-scale preshuffle fold is on, the dispatch-wait kernel
            # writes the activation scale directly into the matmul's per-expert
            # fixed-stride `scale_4d` slots, so the scales output is slot-sized
            # (`n_local_experts * max_padded_M` rows; `n_local_experts` already
            # includes the fused shared expert if present) rather than
            # `max_recv_tokens` rows.
            scales_rows = (
                n_local_experts * ep_mxfp4_max_padded_m(config)
                if config.mxfp4_a_scales_preshuffled
                else max_recv_tokens
            )
            out_scales_type = quant_config.quantized_scales_type(
                Shape([scales_rows, config.hidden_size]), device_ref
            )
            return [
                output_tokens_type,
                out_scales_type,
                expert_start_indices_type,
                expert_ids_type,
                src_info_type,
            ]
        else:
            raise ValueError(
                f"Unsupported dispatch dtype: {config.dispatch_dtype}"
            )

    return [
        output_tokens_type,
        expert_start_indices_type,
        expert_ids_type,
        src_info_type,
    ]


def _add_mxfp4_scale_fusion_parameters(
    parameters: dict[str, bool | int | str | DType],
    config: EPConfig,
) -> None:
    """Inject the KS224 up-proj scale-fusion op parameters.

    When ``config.mxfp4_a_scales_preshuffled``, the MXFP4 dispatch-wait kernel
    writes the activation scale directly into the up-proj grouped-matmul's
    per-expert fixed-stride ``scale_4d`` slot layout (slot stride
    ``max_padded_M * K_SCALES``), so the standalone preshuffle is dropped. The
    matmul reader is told the same ``max_padded_M`` (single source of truth, see
    ``moe_fp8._local_ep_compute``). No-op when the fusion is off.
    """
    if config.mxfp4_a_scales_preshuffled:
        parameters["fuse_a_scale_preshuffle"] = True
        parameters["max_padded_M"] = ep_mxfp4_max_padded_m(config)


def _ep_common_parameters(
    config: EPConfig,
) -> dict[str, bool | int | str | DType]:
    """Returns the common parameters for the EP kernels."""
    if config.use_allreduce:
        # When using allreduce as communication backend, the EP kernels are only
        # used to route token within the current device. Hence, It will only see
        # the local experts.
        return {
            "hidden_size": config.hidden_size,
            "top_k": config.top_k,
            "n_experts": config.n_experts // config.n_gpus_per_node,
            "max_token_per_rank": config.max_tokens_per_rank,
            "n_gpus_per_node": 1,
            "n_nodes": 1,
        }
    return {
        "hidden_size": config.hidden_size,
        "top_k": config.top_k,
        "n_experts": config.n_experts,
        "max_token_per_rank": config.max_tokens_per_rank,
        "n_gpus_per_node": config.n_gpus_per_node,
        "n_nodes": config.n_nodes,
    }


def call_ep_init(
    atomic_counter_group_0: BufferValue,
    atomic_counter_group_1: BufferValue,
    config: EPConfig,
) -> tuple[TensorValue, TensorValue]:
    """Initialize Expert Parallelism communication infrastructure by creating
    a custom operation that initializes SHMEM context and allocates symmetric
    memory buffers for EP communication.

    This operation only initializes the vendor library and allocates the
    symmetric memory buffers for current GPU. To prevent deadlocks, it needs to
    be called for each GPU separately through different threads.

    Args:
        atomic_counter_group_0: Atomic counters for buffer group 0.
        atomic_counter_group_1: Atomic counters for buffer group 1.
        config: EP configuration.

    Returns:
        A tuple containing:
        - device_ptrs: TensorValue containing device pointers to allocated SHMEM buffers.
            The tensor has shape [NUM_GROUPS, 3] where each group contains pointers to:
            [send_buffer, recv_buffer, recv_count_buffer].
        - my_rank: TensorValue containing the rank of the current GPU. The
            tensor has shape [1,].
    """
    parameters = _ep_common_parameters(config)
    parameters["dispatch_dtype"] = config.dispatch_dtype
    parameters["combine_dtype"] = config.combine_dtype

    if config.dispatch_quant_config is not None:
        if _uses_block_scaled_nv_ep_layout(config):
            parameters["dispatch_fmt_str"] = "BLOCK_SCALED_NV"
            parameters["dispatch_scale_dtype"] = (
                DType.float8_e4m3fn
                if config.dispatch_quant_config.is_nvfp4
                else DType.float8_e8m0fnu
            )
        elif uses_mx_ep_token_format(config):
            parameters["dispatch_fmt_str"] = (
                "MXFP6" if config.dispatch_quant_config.is_mxfp6 else "MXFP4"
            )
            parameters["dispatch_scale_dtype"] = DType.float8_e8m0fnu
        elif _is_legacy_float8_dispatch(config):
            parameters["dispatch_fmt_str"] = "BlockwiseFP8"
            parameters["dispatch_scale_dtype"] = DType.float32
        else:
            raise ValueError(
                f"Unsupported dispatch dtype: {config.dispatch_dtype}"
            )
    else:
        # fill in dummy values for non-quantized cases
        parameters["dispatch_fmt_str"] = "BF16"
        parameters["dispatch_scale_dtype"] = DType.float32

    results = ops.inplace_custom(
        "ep.init",
        device=atomic_counter_group_0.device,
        values=[atomic_counter_group_0, atomic_counter_group_1],
        out_types=[
            TensorType(DType.uint64, [NUM_GROUPS, 3], device=DeviceRef.CPU()),
            TensorType(DType.int32, [1], DeviceRef.CPU()),
        ],
        parameters=parameters,
    )

    return results[0].tensor, results[1].tensor


def call_ep_dispatch_async(
    input_tokens: TensorValue,
    topk_ids: TensorValue,
    atomic_counter: BufferValue,
    send_buf_ptrs: TensorValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    input_scales: TensorValue | None = None,
) -> None:
    """Initiate Expert Parallelism token dispatch phase (async).

    This function launches the EP async dispatch kernel that distributes input
    tokens to expert devices based on top-k routing decisions. The kernel uses
    non-blocking SHMEM communication in multi-node scenarios and returns
    immediately after initiating transfers.

    Args:
        input_tokens: Input tokens to be dispatched to experts.
            Shape: (num_tokens, hidden_size)
        topk_ids: Expert IDs selected for each token by the router.
            Shape: (num_tokens, top_k)
            Values: Expert indices in range [0, n_experts)
        atomic_counter: Buffer for synchronization between thread blocks.
        send_buf_ptrs: Device pointers to the send buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (max_tokens_per_rank, msg_bytes)
        recv_buf_ptrs: Device pointers to the receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)
        recv_count_ptrs: Device pointers to the receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks)
        config: EP configuration.
        input_scales: Optional input scales tensor. Required for NVFP4
            dispatch. Shape: (1,) or (n_experts,).

    Note:
        This is a non-blocking operation. Call call_ep_dispatch_wait() to wait
        for completion and collect the dispatched tokens.
    """
    parameters = _ep_common_parameters(config)
    parameters["dispatch_dtype"] = config.dispatch_dtype

    op_name = "ep.dispatch_async"
    input_vals: list[Value[Any]] = [
        atomic_counter,
        input_tokens,
        topk_ids,
        send_buf_ptrs,
        recv_buf_ptrs,
        recv_count_ptrs,
    ]

    if config.dispatch_quant_config is not None:
        quant_config = config.dispatch_quant_config
        if _uses_block_scaled_nv_ep_layout(config):
            if quant_config.is_nvfp4 and input_scales is None:
                raise ValueError(
                    "input_scales must be provided when using NVFP4 dispatch"
                )
            op_name += ".block.scaled.nv"
            parameters["dispatch_scale_dtype"] = (
                DType.float8_e4m3fn
                if quant_config.is_nvfp4
                else DType.float8_e8m0fnu
            )
            if input_scales is not None:
                input_vals.append(1.0 / input_scales.to(input_tokens.device))
            else:
                input_vals.append(
                    ops.constant(
                        [1.0], DType.float32, device=input_tokens.device
                    )
                )
        elif uses_mx_ep_token_format(config):
            op_name += ".mxfp4"
            # No output tensor for MOGG to deduce the scale dtype from.
            parameters["dispatch_scale_dtype"] = DType.float8_e8m0fnu
            _add_mx_format_parameter(parameters, config)
        elif _is_legacy_float8_dispatch(config):
            parameters["dispatch_fmt_str"] = "BlockwiseFP8"
            parameters["dispatch_scale_dtype"] = DType.float32
        else:
            raise ValueError(
                f"Unsupported dispatch dtype: {config.dispatch_dtype}"
            )

    elif config.dispatch_dtype == DType.bfloat16:
        parameters["dispatch_fmt_str"] = "BF16"
        parameters["dispatch_scale_dtype"] = DType.float32
    else:
        raise ValueError(f"Unsupported dispatch dtype: {config.dispatch_dtype}")

    ops.inplace_custom(
        op_name,
        device=input_tokens.device,
        values=input_vals,
        out_types=[],
        parameters=parameters,
    )


def call_ep_dispatch_wait(
    atomic_counter: BufferValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    num_tokens: Dim | None = None,
) -> tuple[TensorValue, ...]:
    """Wait for Expert Parallelism token dispatch and prepare for expert
    computation.

    This function launches the EP dispatch wait kernel that waits for all
    inter-device communication to complete, then organizes the received tokens
    into a format suitable for grouped matmul computation.

    Args:
        atomic_counter: Buffer for synchronization between thread blocks.
        recv_buf_ptrs: Device pointers to the receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)
        recv_count_ptrs: Device pointers to the receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks)
        config: EP configuration.

    Returns:
        A tuple containing:
        - output_tokens: Aggregated tokens ready for grouped matmul computation.
            Shape: (max_recv_tokens, hidden_size)
        - expert_start_indices: Row offsets for grouped matmul operation.
            Shape: (n_local_experts + 1,)
        - expert_ids: Local expert IDs for the grouped operation.
            Shape: (n_local_experts,)
            Maps position in row_offsets to actual expert ID
        - src_info: Source routing information for combine phase.
            Shape: (max_recv_tokens, 2)
            [original_token_index, topk_index] for each received token

    Note:
        This function blocks until all expected tokens have been received from
        remote devices. For Quantized dispatch format, the output will also
        include the aggregated scales as the second element of the tuple.
    """
    parameters = _ep_common_parameters(config)
    device_ref = atomic_counter.device

    # Enable the decode-fast-path grid sizing when num_tokens is known at
    # graph build time. Falls back to the full sm_count grid otherwise.
    if isinstance(num_tokens, StaticDim):
        parameters["num_input_tokens"] = int(num_tokens)

    op_name = "ep.dispatch_wait"
    input_vals: list[Value[Any]] = [
        atomic_counter,
        recv_buf_ptrs,
        recv_count_ptrs,
    ]
    assert not config.fused_shared_expert, (
        "Fused shared expert is not supported when using dispatch_wait."
    )

    output_vals = _ep_dispatch_output_types(config, device_ref)

    if config.dispatch_quant_config is not None:
        quant_config = config.dispatch_quant_config

        if _uses_block_scaled_nv_ep_layout(config):
            op_name += ".block.scaled.nv"
            parameters["dispatch_scale_dtype"] = (
                DType.float8_e4m3fn
                if quant_config.is_nvfp4
                else DType.float8_e8m0fnu
            )
            if config.fused_shared_expert and quant_config.is_nvfp4:
                raise ValueError(
                    "NVFP4 dispatch with fused shared expert is not supported"
                )
        elif _is_legacy_float8_dispatch(config):
            op_name += ".fp8"
            parameters["dispatch_scale_granularity"] = str(
                quant_config.input_scale.granularity
            )
        elif uses_mx_ep_token_format(config):
            op_name += ".mxfp4"
            _add_mx_format_parameter(parameters, config)
            _add_mxfp4_scale_fusion_parameters(parameters, config)
        else:
            raise ValueError(
                f"Unsupported dispatch dtype: {config.dispatch_dtype}"
            )

    results = ops.inplace_custom(
        op_name,
        device=device_ref,
        values=input_vals,
        out_types=output_vals,
        parameters=parameters,
    )

    return tuple([v.tensor for v in results])


def call_ep_combine_async(
    input_tokens: TensorValue,
    src_info: TensorValue,
    atomic_counter: BufferValue,
    send_buf_ptrs: TensorValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
) -> None:
    """Initiate Expert Parallelism token combine phase (async).

    This function launches the EP async combine kernel that sends expert outputs
    back to their original devices based on source routing information. The
    kernel uses non-blocking SHMEM communication in multi-node scenarios and
    returns immediately after initiating transfers.

    Args:
        input_tokens: Expert output tokens to send back to original devices.
            Shape: (max_tokens_per_rank, hidden_size)
            Results from expert computation that need to be routed back
        src_info: Source routing information from dispatch phase.
            Shape: (max_tokens_per_rank, 2)
            [original_token_index, topk_index] for each token
        atomic_counter: Buffer for synchronization between thread blocks.
        send_buf_ptrs: Device pointers to the send buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts * n_ranks * max_tokens_per_rank, msg_bytes).
        recv_buf_ptrs: Device pointers to the receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (max_tokens_per_rank, top_k, msg_bytes).
        recv_count_ptrs: Device pointers to the receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_experts,)
        config: EP configuration.

    Note:
        This is a non-blocking operation. Call call_ep_combine_wait() to wait
        for completion and collect the final outputs.
    """
    parameters = _ep_common_parameters(config)
    parameters["combine_dtype"] = config.combine_dtype

    op_name = "ep.combine_async"
    out_types: list[TensorType] = []

    assert not config.fused_shared_expert, (
        "Fused shared expert is not supported when using combine_async."
    )

    result = ops.inplace_custom(
        op_name,
        device=input_tokens.device,
        values=[
            atomic_counter,
            input_tokens,
            src_info,
            send_buf_ptrs,
            recv_buf_ptrs,
            recv_count_ptrs,
        ],
        out_types=out_types,
        parameters=parameters,
    )

    if config.fused_shared_expert:
        return result[0].tensor
    else:
        return None


def call_ep_combine_wait(
    atomic_counter: BufferValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    num_tokens: Dim,
    router_weights: TensorValue,
) -> TensorValue:
    """Wait for Expert Parallelism token combine and return final outputs.

    This function launches the EP combine wait kernel, which waits for all
    inter-device communication to complete, then computes the weighted sum of
    routed expert outputs for each token.

    Args:
        atomic_counter: Buffer for synchronization between thread blocks.
        recv_buf_ptrs: Device pointers to the receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (max_tokens_per_rank, top_k, msg_bytes)
        recv_count_ptrs: Device pointers to the receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_experts,)
        config: EP configuration.
        num_tokens: Number of original input tokens before expert processing.
        router_weights: Router weights for the current device. Once all tokens
            are received, all routed experts' outputs for each token will be
            weighted and summed to produce the final output for the token.
            Shape: (num_tokens, top_k)

    Returns:
        output_tokens: Final output tensor with expert results.
            Shape: (num_tokens, hidden_size)
            Expert outputs arranged back in original token order.

    Note:
        This function blocks until all expected expert outputs have been
        received from remote devices.
    """
    parameters = _ep_common_parameters(config)
    parameters["combine_dtype"] = config.combine_dtype

    device_ref = atomic_counter.device

    result = ops.inplace_custom(
        "ep.combine_wait",
        device=device_ref,
        values=[atomic_counter, recv_buf_ptrs, recv_count_ptrs, router_weights],
        out_types=[
            TensorType(
                dtype=config.combine_dtype,
                shape=[num_tokens, config.hidden_size],
                device=device_ref,
            ),  # output_tokens
        ],
        parameters=parameters,
    )

    return result[0].tensor


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Fused Kernels
# ===-----------------------------------------------------------------------===#


def call_ep_dispatch(
    input_tokens: TensorValue,
    topk_ids: TensorValue,
    atomic_counter: BufferValue,
    send_buf_ptrs: TensorValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    input_scales: TensorValue | None = None,
) -> tuple[TensorValue, ...]:
    """Execute fused Expert Parallelism token dispatch (async + wait).

    This function launches the fused EP dispatch kernel that combines both
    dispatch_async and dispatch_wait functionality in a single kernel launch.
    It distributes input tokens to expert devices based on top-k routing
    decisions, waits for all tokens to arrive, and organizes received tokens
    into a format suitable for grouped matmul computation.

    Args:
        input_tokens: Input tokens to be dispatched to experts.
            Shape: (num_tokens, hidden_size)
        topk_ids: Expert IDs selected for each token by the router.
            Shape: (num_tokens, top_k)
            Values: Expert indices in range [0, n_experts)
        atomic_counter: Buffer for synchronization between thread blocks.
        send_buf_ptrs: Device pointers to the send buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (max_tokens_per_rank, msg_bytes)
        recv_buf_ptrs: Device pointers to the receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)
        recv_count_ptrs: Device pointers to the receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks)
        config: EP configuration.
        input_scales: Optional input scales tensor. Needed for NVFP4 dispatch.
            Shape: (1,)

    Returns:
        A tuple containing:
        - output_tokens: Aggregated tokens ready for grouped matmul computation.
            Shape: (max_recv_tokens, hidden_size)
        - expert_start_indices: Row offsets for grouped matmul operation.
            Shape: (n_local_experts + 1,)
        - expert_ids: Local expert IDs for the grouped operation.
            Shape: (n_local_experts,)
        - src_info: Source routing information for combine phase.
            Shape: (max_recv_tokens, 2)
            [original_token_index, topk_index] for each received token

    Note:
        For Quantized dispatch format, the output will also include the
        aggregated scales as the second element of the tuple.
    """
    parameters = _ep_common_parameters(config)
    parameters["fused_shared_expert"] = config.fused_shared_expert
    parameters["skip_a2a"] = config.use_allreduce
    parameters["allreduce_world_size"] = config.n_gpus_per_node

    device_ref = atomic_counter.device
    op_name = "ep.dispatch"

    input_vals: list[Value[Any]] = [
        atomic_counter,
        input_tokens,
        topk_ids,
        send_buf_ptrs,
        recv_buf_ptrs,
        recv_count_ptrs,
    ]

    output_vals = _ep_dispatch_output_types(config, device_ref)

    if config.dispatch_quant_config is not None:
        quant_config = config.dispatch_quant_config

        if _uses_block_scaled_nv_ep_layout(config):
            if quant_config.is_nvfp4 and input_scales is None:
                raise ValueError(
                    "input_scales must be provided when using NVFP4 dispatch"
                )
            op_name += ".block.scaled.nv"
            parameters["dispatch_scale_dtype"] = (
                DType.float8_e4m3fn
                if quant_config.is_nvfp4
                else DType.float8_e8m0fnu
            )
            if input_scales is not None:
                input_vals.append(1.0 / input_scales.to(device_ref))
            else:
                input_vals.append(
                    ops.constant([1.0], DType.float32, device=device_ref)
                )
        elif _is_legacy_float8_dispatch(config):
            op_name += ".fp8"
            parameters["dispatch_scale_granularity"] = str(
                quant_config.input_scale.granularity
            )
        elif uses_mx_ep_token_format(config):
            op_name += ".mxfp4"
            _add_mx_format_parameter(parameters, config)
            _add_mxfp4_scale_fusion_parameters(parameters, config)
        else:
            raise ValueError(
                f"Unsupported dispatch dtype: {config.dispatch_dtype}"
            )

    results = ops.inplace_custom(
        op_name,
        device=device_ref,
        values=input_vals,
        out_types=output_vals,
        parameters=parameters,
    )

    return tuple([v.tensor for v in results])


def call_distributed_ep_dispatch(
    input_tokens: list[TensorValue],
    topk_ids: list[TensorValue],
    atomic_counters: list[BufferValue],
    send_buf_ptrs: TensorValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    input_scales: list[TensorValue] | None = None,
) -> list[tuple[TensorValue, ...]]:
    """Multi-device fused EP dispatch (BF16, FP8, or NVFP4).

    Selects the appropriate dispatch variant based on ``config`` and launches
    a single multi-device graph op.

    Returns:
        Per-device output tuples. The number of tensors per tuple depends on
        the quantization format (4 for BF16, 5 for FP8/MXFP4, 6 for NVFP4).
    """
    num_devices = len(input_tokens)

    quant_config = config.dispatch_quant_config
    uses_nvidia_block_scaled = (
        quant_config is not None and _uses_block_scaled_nv_ep_layout(config)
    )
    # Covers MXFP8 too; otherwise it would silently fall through to bf16.
    is_mx_token_format = uses_mx_ep_token_format(config)
    is_fp8 = _is_legacy_float8_dispatch(config)

    output_types_per_device: list[list[TensorType]] = []
    for i in range(num_devices):
        device_ref = atomic_counters[i].device
        types = _ep_dispatch_output_types(config, device_ref)
        output_types_per_device.append(types)

    if uses_nvidia_block_scaled:
        assert quant_config is not None
        if quant_config.is_nvfp4 and input_scales is None:
            raise ValueError("input_scales must be provided for NVFP4 dispatch")
        if input_scales is not None:
            inv_scales = [
                1.0 / input_scales[i].to(atomic_counters[i].device)
                for i in range(num_devices)
            ]
        else:
            inv_scales = [
                ops.constant(
                    [1.0], DType.float32, device=atomic_counters[i].device
                )
                for i in range(num_devices)
            ]
        return ops.distributed_ep.dispatch_block_scaled_nv(
            input_tokens,
            topk_ids,
            send_buf_ptrs,
            recv_buf_ptrs,
            recv_count_ptrs,
            inv_scales,
            atomic_counters,
            output_types_per_device,
            hidden_size=config.hidden_size,
            top_k=config.top_k,
            n_experts=config.n_experts,
            max_token_per_rank=config.max_tokens_per_rank,
            n_gpus_per_node=config.n_gpus_per_node,
            n_nodes=config.n_nodes,
            fused_shared_expert=config.fused_shared_expert,
        )
    elif is_mx_token_format:
        return ops.distributed_ep.dispatch_mxfp4(
            input_tokens,
            topk_ids,
            send_buf_ptrs,
            recv_buf_ptrs,
            recv_count_ptrs,
            atomic_counters,
            output_types_per_device,
            hidden_size=config.hidden_size,
            top_k=config.top_k,
            n_experts=config.n_experts,
            max_token_per_rank=config.max_tokens_per_rank,
            n_gpus_per_node=config.n_gpus_per_node,
            n_nodes=config.n_nodes,
            fused_shared_expert=config.fused_shared_expert,
            fuse_a_scale_preshuffle=config.mxfp4_a_scales_preshuffled,
            max_padded_m=ep_mxfp4_max_padded_m(config),
            mx_format=(
                _mx_format_name(quant_config)
                if quant_config is not None and quant_config.is_mxfp6
                else _MX_FORMAT_AUTO
            ),
        )
    elif is_fp8:
        assert quant_config is not None
        return ops.distributed_ep.dispatch_fp8(
            input_tokens,
            topk_ids,
            send_buf_ptrs,
            recv_buf_ptrs,
            recv_count_ptrs,
            atomic_counters,
            output_types_per_device,
            hidden_size=config.hidden_size,
            top_k=config.top_k,
            n_experts=config.n_experts,
            max_token_per_rank=config.max_tokens_per_rank,
            n_gpus_per_node=config.n_gpus_per_node,
            n_nodes=config.n_nodes,
            fused_shared_expert=config.fused_shared_expert,
            dispatch_scale_granularity=str(
                quant_config.input_scale.granularity
            ),
        )
    else:
        return ops.distributed_ep.dispatch_bf16(
            input_tokens,
            topk_ids,
            send_buf_ptrs,
            recv_buf_ptrs,
            recv_count_ptrs,
            atomic_counters,
            output_types_per_device,
            hidden_size=config.hidden_size,
            top_k=config.top_k,
            n_experts=config.n_experts,
            max_token_per_rank=config.max_tokens_per_rank,
            n_gpus_per_node=config.n_gpus_per_node,
            n_nodes=config.n_nodes,
            fused_shared_expert=config.fused_shared_expert,
        )


def call_distributed_ep_combine(
    input_tokens: list[TensorValue],
    src_info: list[TensorValue],
    atomic_counters: list[BufferValue],
    send_buf_ptrs: TensorValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    num_tokens_per_device: list[Dim],
    router_weights: list[TensorValue],
) -> list[TensorValue]:
    """Multi-device fused EP combine.

    Launches a single ``mo.distributed.ep.combine`` graph op that combines
    expert outputs back to their original devices on all GPUs simultaneously.
    """
    num_devices = len(input_tokens)
    output_types: list[TensorType] = []
    for i in range(num_devices):
        device_ref = atomic_counters[i].device
        output_types.append(
            TensorType(
                dtype=config.combine_dtype,
                shape=[num_tokens_per_device[i], config.hidden_size],
                device=device_ref,
            )
        )

    return ops.distributed_ep.combine(
        input_tokens,
        src_info,
        send_buf_ptrs,
        recv_buf_ptrs,
        recv_count_ptrs,
        router_weights,
        atomic_counters,
        output_types,
        hidden_size=config.hidden_size,
        top_k=config.top_k,
        n_experts=config.n_experts,
        max_token_per_rank=config.max_tokens_per_rank,
        n_gpus_per_node=config.n_gpus_per_node,
        n_nodes=config.n_nodes,
        fused_shared_expert=config.fused_shared_expert,
    )


def call_ep_combine(
    input_tokens: TensorValue,
    src_info: TensorValue,
    atomic_counter: BufferValue,
    send_buf_ptrs: TensorValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    num_tokens: Dim,
    router_weights: TensorValue,
    topk_ids: TensorValue | None = None,
) -> TensorValue:
    """Execute fused Expert Parallelism token combine (async + wait).

    This function launches the fused EP combine kernel that combines both
    combine_async and combine_wait functionality in a single kernel launch.
    It sends expert outputs back to their original devices, waits for all
    transfers to complete, and computes the weighted sum of routed expert
    outputs for each token.

    Args:
        input_tokens: Expert output tokens to send back to original devices.
            Shape: (max_tokens_per_rank, hidden_size)
            Results from expert computation that need to be routed back.
            For fused_shared_expert mode, this also contains shared expert
            outputs at the start.
        src_info: Source routing information from dispatch phase.
            Shape: (max_tokens_per_rank, 2)
            [original_token_index, topk_index] for each token
        atomic_counter: Buffer for synchronization between thread blocks.
        send_buf_ptrs: Device pointers to the send buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts * n_ranks * max_tokens_per_rank, msg_bytes).
        recv_buf_ptrs: Device pointers to the receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (max_tokens_per_rank, top_k, msg_bytes).
        recv_count_ptrs: Device pointers to the receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_experts,)
        config: EP configuration.
        num_tokens: Number of original input tokens before expert processing.
        router_weights: Router weights for the current device. All routed
            experts' outputs for a token will be weighted and summed.
            Shape: (num_tokens, top_k)
        fused_shared_expert: Whether to add shared expert outputs to the
            combined result. When True, the shared expert outputs are read
            from input_tokens and added to the reduced routed expert outputs.

    Returns:
        output_tokens: Final output tensor with expert results.
            Shape: (num_tokens, hidden_size)
            Expert outputs arranged back in original token order.
    """
    parameters = _ep_common_parameters(config)
    parameters["fused_shared_expert"] = config.fused_shared_expert
    parameters["skip_a2a"] = config.use_allreduce

    op_name = "ep.combine"
    vals: list[Value[Any]] = [
        atomic_counter,
        input_tokens,
        src_info,
        send_buf_ptrs,
        recv_buf_ptrs,
        recv_count_ptrs,
        router_weights,
    ]

    if config.use_allreduce:
        assert topk_ids is not None, (
            "Top-k expert IDs are not provided for allreduce mode."
        )
        parameters["allreduce_world_size"] = config.n_gpus_per_node
        op_name += ".skip_a2a"
        vals.append(topk_ids)

    device_ref = atomic_counter.device

    result = ops.inplace_custom(
        op_name,
        device=device_ref,
        values=vals,
        out_types=[
            TensorType(
                dtype=config.combine_dtype,
                shape=[num_tokens, config.hidden_size],
                device=device_ref,
            ),  # output_tokens
        ],
        parameters=parameters,
    )

    return result[0].tensor


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Utils
# ===-----------------------------------------------------------------------===#


def fused_silu(
    input: TensorValue,
    row_offsets: TensorValue,
) -> TensorValue:
    """Perform fused SILU operation for all the MLPs in the EP MoE module.

    We need to manually implement the custom operation here is because after
    the EP dispatch phase, the actual number of received tokens is not known to
    the host. This kernel will read the row offsets to determine the actual
    number of received tokens in the input tensor, and then only perform the
    SILU operation on the received tokens.

    Args:
        input_tokens: Input tokens to perform the SILU operation.
            Shape: (max_recv_tokens, hidden_size)
        row_offsets: Row offsets to determine the actual number of received
            tokens in the input tensor.
            Shape: (n_local_experts + 1,)

    Returns:
        output_tokens: Output tokens after the SILU operation.
            Shape: (max_recv_tokens, hidden_size)
    """
    if input.rank != 2:
        raise ValueError("input must be rank 2 tensor")

    if not isinstance(input.shape[1], StaticDim):
        raise ValueError(
            f"input.shape[1] must be a statically known dimension. Input shape received: {input.shape}"
        )

    hidden_size = input.shape[1] // 2

    return ops.custom(
        "ep.fused_silu",
        device=input.device,
        values=[input, row_offsets],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=[input.shape[0], hidden_size],
                device=input.device,
            ),
        ],
    )[0].tensor


def fused_silu_quantized(
    input: TensorValue,
    row_offsets: TensorValue,
    quant_config: QuantConfig,
    out_type: DType,
    input_scales: TensorValue | None = None,
    scales_offsets: TensorValue | None = None,
    max_padded_M: int = 0,
    clamp_activation: bool = False,
    swiglu_alpha: float = 0.0,
    swiglu_limit: float = 0.0,
) -> tuple[TensorValue, TensorValue]:
    """Perform fused SILU operation for all the MLPs in the EP MoE module.

    We need to manually implement the custom operation here is because after
    the EP dispatch phase, the actual number of received tokens is not known to
    the host. This kernel will read the row offsets to determine the actual
    number of received tokens in the input tensor, and then only perform the
    SILU operation on the received tokens. Once the SILU operation is performed,
    the output will be quantized to the FP8 format. The scales will be stored
    in a transposed way.

    Args:
        input: Input tokens to perform the SILU operation.
            Shape: (max_recv_tokens, hidden_size)
        row_offsets: Row offsets to determine the actual number of received
            tokens in the input tensor.
            Shape: (n_local_experts + 1,)
        quant_config: Quantization configuration.
        out_type: Output dtype.
        input_scales: Optional input scales tensor. Needed by NVFP4.
        scales_offsets: Optional scales offsets tensor. Needed by NVFP4.
        max_padded_M: When > 0 (MXFP4, MXFP6, and MXFP8 EP down-proj
            fusion), the kernel writes the E8M0 activation scale directly
            into the grouped-matmul per-expert slot layout.  Must equal
            ``align_up(max_recv_tokens_per_expert, 32)``.  The output
            scales tensor will have shape
            ``[n_local_experts * max_padded_M, K_SCALES]`` instead of
            ``[max_recv_tokens, K_SCALES]``.
        clamp_activation: When True (MXFP4 and MXFP8), apply the OAI-clamped SwiGLU
            activation ``(clamp(up, -L, L) + 1) * min(gate, L) *
            sigmoid(alpha * min(gate, L))`` instead of plain ``SiLU(gate) *
            up``.
        swiglu_alpha: Alpha for the clamped activation (ignored unless
            ``clamp_activation``).
        swiglu_limit: Limit ``L`` for the clamped activation (ignored unless
            ``clamp_activation``).

    Returns:
        A tuple containing:
        - output_tokens: Output tokens after the SILU operation.
            Shape: (max_recv_tokens, hidden_size)
        - output_scales: Output scales after the SILU operation. Shape depends
            on the quantization format.
    """
    if input.rank != 2:
        raise ValueError("input_tokens must be rank 2 tensor")

    if not isinstance(input.shape[1], StaticDim):
        raise ValueError(
            f"input.shape[1] must be a statically known dimension. Input shape received: {input.shape}"
        )

    hidden_size = input.shape[1] // 2
    op_name = "ep.fused_silu"
    input_vals: list[Value[Any]] = [input, row_offsets]
    out_scales_type = quant_config.quantized_scales_type(
        Shape([input.shape[0], input.shape[1] // 2]), input.device
    )

    if quant_config.is_nvfp4:
        op_name += ".nvfp4"
        hidden_size //= 2  # Two FP4 elements are packed into one uint8 element
        assert scales_offsets is not None and input_scales is not None, (
            "scales_offsets and input_scales must be provided when using NVFP4"
        )
        input_vals.append(scales_offsets)
        input_vals.append(1.0 / input_scales.to(input.device))

        # Pad the scales tensor to satisfy the grouped matmul requirement.
        n_local_experts = row_offsets.shape[0] - 1
        out_scales_type = quant_config.quantized_scales_type(
            Shape(
                [
                    input.shape[0] + n_local_experts * NVFP4_MN_GROUP_SIZE,
                    input.shape[1] // 2,
                ]
            ),
            input.device,
        )

    elif quant_config.is_mxfp4:
        op_name += ".mxfp4"
        hidden_size //= 2  # Two FP4 elements are packed into one uint8 element
        # mxfp4 op always takes trailing alpha/limit CPU f32 constants
        # (plain SiLU passes 0.0/0.0).
        input_vals.append(
            ops.constant(swiglu_alpha, DType.float32, device=DeviceRef.CPU())
        )
        input_vals.append(
            ops.constant(swiglu_limit, DType.float32, device=DeviceRef.CPU())
        )
        if max_padded_M > 0:
            # KS64 down-proj fusion: write the E8M0 scale directly
            # into the grouped matmul's per-expert fixed-stride slot layout so
            # the standalone preshuffle kernel is dropped from the critical path.
            # Pass `n_local_experts * max_padded_M` rows and the raw hidden dim
            # (input.shape[1] // 2) to _mxfp4_scales_type so its ceildiv(K,32)
            # gives K_SCALES = hidden_size // 32 correctly.
            n_local_experts = row_offsets.shape[0] - 1
            raw_hidden = input.shape[1] // 2  # half of gate+up concat
            out_scales_type = quant_config.quantized_scales_type(
                Shape([n_local_experts * max_padded_M, raw_hidden]),
                input.device,
            )
    elif quant_config.is_mxfp6:
        op_name += ".mxfp6"
        hidden_size = hidden_size * 3 // 4
        input_vals.append(
            ops.constant(swiglu_alpha, DType.float32, device=DeviceRef.CPU())
        )
        input_vals.append(
            ops.constant(swiglu_limit, DType.float32, device=DeviceRef.CPU())
        )
        if max_padded_M > 0:
            n_local_experts = row_offsets.shape[0] - 1
            raw_hidden = input.shape[1] // 2
            out_scales_type = quant_config.quantized_scales_type(
                Shape([n_local_experts * max_padded_M, raw_hidden]),
                input.device,
            )
    elif quant_config.is_mxfp8:
        # Must precede the generic float8 branch: MXFP8's out_type IS
        # float8_e4m3fn, so falling through would select the legacy per-tensor
        # `.fp8` kernel. No `hidden_size //= 2`: one byte per element.
        op_name += ".mxfp8"
        input_vals.append(
            ops.constant(swiglu_alpha, DType.float32, device=DeviceRef.CPU())
        )
        input_vals.append(
            ops.constant(swiglu_limit, DType.float32, device=DeviceRef.CPU())
        )
        if max_padded_M > 0:
            # Same fold as MXFP4; the E8M0 scale layout is format-independent.
            n_local_experts = row_offsets.shape[0] - 1
            raw_hidden = input.shape[1] // 2
            out_scales_type = quant_config.quantized_scales_type(
                Shape([n_local_experts * max_padded_M, raw_hidden]),
                input.device,
            )
    elif out_type.is_float8():
        op_name += ".fp8"
    else:
        raise ValueError(
            f"Unsupported quantization format: {quant_config.format}"
        )

    parameters: dict[str, bool | int] = {}
    if quant_config.is_mxfp6:
        parameters["clamp_activation"] = clamp_activation
        parameters["FP6_FORMAT"] = (
            1 if quant_config.mxfp6_format == "e3m2" else 0
        )
        if max_padded_M > 0:
            parameters["fuse_a_scale_preshuffle"] = True
            parameters["max_padded_M"] = max_padded_M
    elif quant_config.is_mxfp4 or quant_config.is_mxfp8:
        # Clamp flag applies even with the scale fold off.
        parameters["clamp_activation"] = clamp_activation
        if max_padded_M > 0:
            parameters["fuse_a_scale_preshuffle"] = True
            parameters["max_padded_M"] = max_padded_M

    result = ops.custom(
        op_name,
        device=input.device,
        values=input_vals,
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[input.shape[0], hidden_size],
                device=input.device,
            ),
            out_scales_type,
        ],
        parameters=parameters,
    )

    return result[0].tensor, result[1].tensor
