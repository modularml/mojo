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

"""Expert Parallelism (EP) Communication Configuration."""

import os
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.nn.quant_config import QuantConfig
from max.support.math import ceildiv

# We always use two groups of SHMEM shared memory buffers to avoid race
# conditions between the dispatch and combine phases of Expert Parallelism
# communication.
#
# Expert Parallelism consists of two sequential phases, each with two kernels:
# 1. Dispatch phase: Route tokens to experts on different GPUs
#    - dispatch_async_kernel: Initiates token transfer
#    - dispatch_wait_kernel: Ensures current GPU received all tokens from other
#      GPUs
# 2. Combine phase: Return expert outputs to original GPUs
#    - combine_async_kernel: Initiates expert output transfer
#    - combine_wait_kernel: Ensures current GPU received all expert outputs
#
# When dispatch_wait_kernel completes on the current GPU, it only guarantees that
# THIS GPU has received all tokens. Other GPUs may still be writing to their
# dispatch buffers or reading their own receive buffers. Therefore, we cannot
# safely reuse dispatch phase buffers for the combine phase.
#
# We use dedicated buffers for each phase. When combine_wait_kernel completes, it
# guarantees all GPUs have at least started their combine_async_kernel, which implies
# they've finished their dispatch_wait_kernel. Only then is it safe to reuse the
# dispatch buffers for the next iteration.
NUM_GROUPS = 2


@dataclass
class EPConfig:
    """Configuration for Expert Parallelism (EP) communication.

    .. note::

       The EP kernel requires ``n_gpus_per_node * n_nodes >= 2``.
       Single-GPU MoE configs pass Python construction but fail at
       Mojo kernel compile (``constraint failed``) after 2-3 minutes.
       For single-GPU MoE, instantiate the MoE block without an EP
       manager and use ``ShardingStrategy.tensor_parallel(1)``.
    """

    dispatch_dtype: DType
    """Data type used for dispatching tokens to experts."""

    combine_dtype: DType
    """Data type used for combining expert outputs."""

    hidden_size: int
    """Size of the hidden dimension in the model."""

    top_k: int
    """Number of top experts to route each token to."""

    n_experts: int
    """Total number of experts in the model."""

    max_tokens_per_rank: int
    """Maximum number of tokens processed per GPU rank."""

    n_gpus_per_node: int
    """Number of GPUs available per node."""

    n_nodes: int
    """Total number of nodes in the distributed setup."""

    node_id: int = -1
    """ID of the current node. Will be set by the EPCommInitializer."""

    dispatch_quant_config: QuantConfig | None = None
    """Quantization configuration used for dispatch token quantization."""

    fused_shared_expert: bool = False
    """Whether to fuse the shared expert computation with the routed experts."""

    use_allreduce: bool = False
    """Whether to use allreduce for the cross-device communication."""

    max_batch_size: int = 0
    """Decode concurrency cap sizing the AMD MXFP4 preb direct grid.y on the
    decode bands. 0 disables (full-stride fallback)."""

    # EPLB parameters (used only when eplb_enabled is True)
    eplb_enabled: bool = False
    """When true, EPBatchManager exposes log2phy / logcnt input buffer. """

    num_moe_layers: int = 0
    """Number of MoE layers; sizes the per layer EPLB buffer. """

    max_replicas: int = 1
    """Largest replica count across all (layer, logical) cells. Also the trailing dim of log2phy."""

    eplb_phy2log_plan: Any = None
    """Optional pre-computed EPLB plan (phy2log numpy array, shape
    [num_moe_layers, num_phy]). When non-None, EPBatchManager picks it
    up at construction time so that MoE.shard sees the plan during the
    model's first __init__ pass (no re-shard needed).
    """

    mxfp4_a_scales_preshuffled: bool = False
    """When True (KS224 up-proj fusion, MXFP4 preshuffled-B path on
    AMD), the dispatch-wait kernel writes the per-token E8M0 activation scale
    directly into the up-proj grouped-matmul's per-expert fixed-stride
    ``scale_4d`` slot layout, so the standalone preshuffle kernel is dropped from
    the decode critical path. The dispatch scales output is then slot-sized
    (``n_local_experts * align_up(max_recv_tokens_per_expert, 32)`` rows)
    instead of ``max_recv_tokens`` rows. Set by ``MoEQuantized`` when the MXFP4
    strategy uses preshuffled B. Only valid for MXFP4 dispatch."""
    # In EPConfig, alongside n_experts / num_moe_layers / max_replicas:

    num_logical_experts: int = 0
    """Number of logical experts per layer. When EPLB is disabled this equals
    ``n_experts``. When EPLB is enabled, ``n_experts`` is inflated to the
    physical slot count and this preserves the original logical count, which
    is the leading dim of ``log2phy`` / ``logcnt``."""

    def estimate_memory_usage(self) -> int:
        """Estimate the EP communication memory usage per device per buffer group.

        Returns:
            Estimated memory usage in bytes per device per buffer group.
        """
        # If we use allreduce as communication backend, the EP kernels are only
        # used to route token within the current device. Hence, we only need to
        # allocate the memory for the local experts.
        _n_experts = (
            self.n_experts // self.n_nodes
            if self.use_allreduce
            else self.n_experts
        )
        return estimate_ep_memory_usage(
            hidden_size=self.hidden_size,
            dispatch_dtype=self.dispatch_dtype,
            combine_dtype=self.combine_dtype,
            max_tokens_per_rank=self.max_tokens_per_rank,
            n_experts=_n_experts,
            n_nodes=self.n_nodes,
            n_gpus_per_node=self.n_gpus_per_node,
            top_k=self.top_k,
            use_allreduce=self.use_allreduce,
            dispatch_element_dtype=(
                self.dispatch_quant_config.mx_element_dtype
                if self.dispatch_quant_config is not None
                and self.dispatch_quant_config.is_mx
                else None
            ),
        )

    def get_max_recv_tokens(self) -> int:
        """Get the maximum number of tokens that can be received by a single device."""
        n_ranks = self.n_gpus_per_node * self.n_nodes
        n_local_experts = self.n_experts // n_ranks
        if self.use_allreduce:
            return self.max_tokens_per_rank * min(n_local_experts, self.top_k)
        else:
            return self.max_tokens_per_rank * min(
                self.n_experts, n_ranks * self.top_k
            )

    def __post_init__(self):
        if self.num_logical_experts == 0:
            self.num_logical_experts = self.n_experts

        if self.dispatch_dtype != DType.bfloat16:
            if self.dispatch_quant_config is None:
                raise ValueError(
                    "dispatch_quant_config must be specified when dispatch_dtype is not bfloat16"
                )

            if self.dispatch_dtype.is_float8():
                if not self.dispatch_quant_config.input_scale.is_block:
                    raise NotImplementedError(
                        "Only block-wise quantization is supported for float8_e4m3fn and float8_e4m3fnuz"
                    )

            elif self.dispatch_dtype in (DType.uint8, DType.float4_e2m1fn):
                if not (
                    self.dispatch_quant_config.is_fp4
                    or self.dispatch_quant_config.is_mxfp6
                ):
                    raise ValueError(
                        "dispatch_quant_config must be an FP4 or MXFP6 configuration when dispatch_dtype is uint8 or float4_e2m1fn"
                    )

            else:
                raise ValueError(
                    f"Unsupported dispatch dtype: {self.dispatch_dtype}"
                )

        if self.use_allreduce and self.n_nodes > 1:
            raise ValueError(
                "Using allreduce as communication backend is not supported when n_nodes > 1"
            )

        if self.n_nodes > 1:
            # Multi-node EP initializes NVSHMEM via MPI at kernel launch time.
            # A common misconfiguration is setting ep_size larger than the number
            # of GPUs actually assigned to this pod/node — e.g. ep_size=8 but the
            # container only receives 1 GPU. This makes n_nodes=8, triggering NVSHMEM
            # for a phantom 8-node cluster that doesn't exist, and fails with
            # NVSHMEMX_ERROR_INTERNAL (status:1).
            #
            # Multi-node launches via mpirun always set OMPI_COMM_WORLD_SIZE (the
            # only MPI launcher used in this codebase — see utils/bmpirun.sh).
            if "OMPI_COMM_WORLD_SIZE" not in os.environ:
                ep_size = self.n_gpus_per_node * self.n_nodes
                raise ValueError(
                    f"ep_size={ep_size} with {self.n_gpus_per_node} GPU(s) on"
                    f" this node implies {self.n_nodes}-node multi-node EP, which"
                    f" requires NVSHMEM initialized via MPI — but"
                    f" OMPI_COMM_WORLD_SIZE is not set. For a single-node"
                    f" deployment set ep_size={self.n_gpus_per_node} (the number"
                    f" of GPUs on this node)."
                )


def estimate_ep_memory_usage(
    *,
    hidden_size: int,
    dispatch_dtype: DType,
    combine_dtype: DType,
    max_tokens_per_rank: int,
    n_experts: int,
    n_nodes: int,
    n_gpus_per_node: int,
    top_k: int,
    use_allreduce: bool = False,
    dispatch_element_dtype: DType | None = None,
) -> int:
    """Estimate the EP communication memory usage per device per buffer group.

    This is a standalone function so it can be called both from
    :class:`~max.nn.comm.ep.ep_manager.EPCommInitializer` (which has a fully-validated ``EPConfig``)
    and from memory estimators that only need the numeric fields.

    Args:
        hidden_size: Model hidden dimension.
        dispatch_dtype: Data type used for dispatching tokens to experts.
        combine_dtype: Data type used for combining expert outputs.
        max_tokens_per_rank: Maximum tokens per GPU rank.
        n_experts: Total number of routed experts.
        n_nodes: Total number of nodes in the distributed setup.
        n_gpus_per_node: Number of GPUs available per node.
        top_k: Number of experts each token is routed to.
        use_allreduce: Whether allreduce is used for cross-device communication.
            When True, dispatch/combine buffers are sized for local experts only.
        dispatch_element_dtype: The logical element dtype behind a packed
            ``uint8`` dispatch dtype (see
            :attr:`~max.nn.quant_config.QuantConfig.mx_element_dtype`). ``uint8``
            alone cannot say whether the bytes are FP4 or FP6, and the two pack
            to different widths, so sizing needs the element type. ``None`` for
            formats whose dtype already identifies them.

    Returns:
        Total estimated memory usage in bytes per device per buffer group.
    """

    def _n_elems_to_bytes(dtype: DType, n_elems: int) -> int:
        if dtype == DType.uint8 and dispatch_element_dtype in (
            DType.float6_e2m3fn,
            DType.float6_e3m2fn,
        ):
            return n_elems * 3 // 4 + n_elems // 32
        if dtype in (DType.uint8, DType.float4_e2m1fn):
            # Account for the scales. For NVFP4 format, every 16 FP4 elements
            # share one FP8 scale factor. The size of the scales is one eighth
            # of the size of the FP4 quants (8 bits / (16 * 4 bits)).
            return int(n_elems // 2 * dtype.size_in_bytes * 1.125)
        else:
            return n_elems * dtype.size_in_bytes

    n_local_experts = n_experts // (n_gpus_per_node * n_nodes)
    d_token_size = _n_elems_to_bytes(dispatch_dtype, hidden_size)
    dispatch_send_buf_size = max_tokens_per_rank * d_token_size
    dispatch_recv_buf_size: int
    if not use_allreduce:
        dispatch_recv_buf_size = n_experts * max_tokens_per_rank * d_token_size
    else:
        dispatch_recv_buf_size = (
            n_local_experts * max_tokens_per_rank * d_token_size
        )

    c_token_size = hidden_size * combine_dtype.size_in_bytes
    # When all the devices are on the same node, we skip the combine send buffer
    # and directly send tokens to each device's recv buffer. Hence, we don't
    # need to allocate the combine send buffer.
    combine_send_buf_size = (
        max_tokens_per_rank
        * c_token_size
        * min(n_experts, n_gpus_per_node * n_nodes * top_k)
        if n_nodes > 1
        else 0
    )
    combine_recv_buf_size = top_k * max_tokens_per_rank * c_token_size

    return (
        dispatch_send_buf_size
        + dispatch_recv_buf_size
        + combine_send_buf_size
        + combine_recv_buf_size
    )


def calculate_ep_max_tokens_per_rank(
    *,
    max_batch_input_tokens: int,
    ep_size: int,
    data_parallel_degree: int,
    use_allreduce: bool = False,
) -> int:
    """Calculate the maximum number of tokens per rank for EP communication.

    Derives the tensor parallelism degree from ``ep_size`` and
    ``data_parallel_degree``, then divides the batch tokens accordingly.
    When TP > 1, attention scatters tokens across ranks so each rank
    holds fewer tokens for the subsequent EP dispatch/combine phases.

    Args:
        max_batch_input_tokens: Maximum number of input tokens per batch.
        ep_size: Expert parallelism size (total number of GPUs across nodes).
        data_parallel_degree: Degree of data parallelism.
        use_allreduce: Is allreduce-backed expert parallelism enabled.

    Returns:
        Maximum tokens per rank for EP communication buffers.
    """
    if use_allreduce:
        return max_batch_input_tokens
    tp_size = ep_size // data_parallel_degree
    # Match the ceiling-biased ragged binning in ops.reducescatter.sum: when
    # max_batch_input_tokens is not divisible by tp_size, the first
    # (max_batch_input_tokens % tp_size) ranks receive
    # ceil(max_batch_input_tokens / tp_size) tokens, so the EP per-rank cap
    # must be ceil, not floor — otherwise the dispatch kernel rejects the
    # largest shard (see ep.mojo dispatch assertion).
    return ceildiv(max_batch_input_tokens, tp_size)
