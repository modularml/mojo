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

"""
Expert Parallelism (EP) Communication Kernel.
"""


import extensibility
from comm.sync import is_p2p_enabled
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from max.gpu.host import DeviceBuffer, DeviceContext, DeviceContextArray
from max.gpu.host.info import is_gpu
from std.memory.unsafe_pointer import pointer_to_int
from layout.tile_tensor import row_major
from std.utils.index import IndexList

from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id
from std.sys.info import size_of, has_amd_gpu_accelerator
from extensibility import (
    InputTensor,
    InputVariadicTensors,
    OutputTensor,
    OutputVariadicTensors,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from extensibility import (
    _MutableInputVariadicTensors as MutableInputVariadicTensors,
)
from extensibility import (
    _FusedOutputTensor as FusedOutputTensor,
)
from extensibility import (
    _FusedOutputVariadicTensors as FusedOutputVariadicTensors,
)

from .kernels import _launch_device_collective

from shmem import (
    shmem_init_thread_mpi,
    shmem_init_thread_tcp,
    shmem_malloc,
    shmem_my_pe,
)
from shmem.ep import (
    ep_combine_async_kernel_api,
    ep_combine_wait_kernel_api,
    ep_dispatch_async_kernel_api,
    ep_dispatch_wait_kernel_api,
    ep_fused_combine_kernel_api,
    ep_fused_dispatch_kernel_api,
)
from shmem.ep_comm import (
    BF16TokenFormat,
    BlockwiseFP8TokenFormat,
    EPLocalSyncCounters,
    MXTokenFormat,
    NVBlockScaledTokenFormat,
    elementwise_epilogue_type,
    fused_silu_kernel,
    fused_silu_fp8_kernel,
    fused_silu_mx_kernel,
    fused_silu_mxfp6_kernel,
    fused_silu_nvfp4_kernel,
)
from linalg.mx_format import MXFormat
from linalg.fp6_utils import FP6Format

comptime RT_LAYOUT_2D = type_of(row_major(Int64(1), Int64(1)))


@extensibility.register("ep.init")
struct Struct_ep_init:
    """Registers the `ep.init` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        dispatch_dtype: DType,
        combine_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_dtype: DType,
        dispatch_fmt_str: StaticString,
        //,
        target: StaticString,
    ](
        dev_ptrs: OutputTensor[dtype=.uint64, rank=2, ...],
        my_rank_tensor: OutputTensor[dtype=.int32, rank=1, ...],
        atomic_counters_0: MutableInputTensor[dtype=.int32, ...],
        atomic_counters_1: MutableInputTensor[dtype=.int32, ...],
        context: DeviceContext,
    ) raises:
        """This kernel initializes the vendor library for Expert Parallelism
        on the current GPU device. It also allocates symmetric memory buffers.

        Parameters:
            dispatch_dtype: DType used during token dispatch to experts.
            combine_dtype: DType used when combining expert outputs.
            hidden_size: Size of the model's hidden dimension.
            top_k: Number of experts each token is routed to.
            n_experts: Total number of experts across all GPUs.
            max_token_per_rank: Maximum number of tokens per GPU.
            n_gpus_per_node: Number of GPUs per node.
            n_nodes: Number of physical nodes.
            dispatch_scale_dtype: DType of the dispatch scale.
            dispatch_fmt_str: String indicating the dispatch format.
            target: Target for this kernel.

        Arguments:
            dev_ptrs: Output tensor to store device pointers. Shape [2, 3] where:
                     - First dimension: buffer groups (0=dispatch, 1=combine)
                     - Second dimension: buffer types (0=send, 1=recv, 2=recv_count)
            my_rank_tensor: Output tensor to store current device's rank.
            atomic_counters_0: Atomic counters for buffer group 0.
            atomic_counters_1: Atomic counters for buffer group 1.
            context: GPU device context
        """
        # Ensure this kernel only runs on GPU targets
        comptime assert is_gpu[target](), "EP is only supported on GPU."
        var gpu_ctx = context

        # Calculate buffer sizes for dispatch phase
        var dispatch_msg_size: Int

        # Infer message sizes for dispatch phases
        comptime if dispatch_fmt_str == "BlockwiseFP8":
            comptime token_fmt_type = BlockwiseFP8TokenFormat[
                fp8_dtype=dispatch_dtype,
                scales_dtype=dispatch_scale_dtype,
                output_layout=RT_LAYOUT_2D,
                scales_layout=RT_LAYOUT_2D,
                hidden_size,
                top_k,
            ]
            dispatch_msg_size = token_fmt_type.msg_size()

        elif dispatch_fmt_str == "BLOCK_SCALED_NV":
            comptime token_fmt_type = NVBlockScaledTokenFormat[
                quant_dtype=dispatch_dtype,
                scales_dtype=dispatch_scale_dtype,
                output_layout=RT_LAYOUT_2D,
                scales_offset_layout=RT_LAYOUT_2D,
                hidden_size,
                top_k,
            ]
            dispatch_msg_size = token_fmt_type.msg_size()

        elif dispatch_fmt_str == "MXFP4":
            comptime token_fmt_type = MXTokenFormat[
                quant_dtype=dispatch_dtype,
                scales_dtype=dispatch_scale_dtype,
                output_layout=RT_LAYOUT_2D,
                scales_layout=RT_LAYOUT_2D,
                hidden_size,
                top_k,
            ]
            dispatch_msg_size = token_fmt_type.msg_size()

        elif dispatch_fmt_str == "MXFP6":
            comptime token_fmt_type = MXTokenFormat[
                quant_dtype=dispatch_dtype,
                scales_dtype=dispatch_scale_dtype,
                output_layout=RT_LAYOUT_2D,
                scales_layout=RT_LAYOUT_2D,
                hidden_size,
                top_k,
                mx_format=MXFormat.FP6_E2M3,
            ]
            dispatch_msg_size = token_fmt_type.msg_size()

        elif dispatch_fmt_str == "BF16":
            comptime token_fmt_type = BF16TokenFormat[
                output_layout=RT_LAYOUT_2D, hidden_size, top_k
            ]
            dispatch_msg_size = token_fmt_type.msg_size()

        else:
            raise Error("Invalid dispatch format string: ", dispatch_fmt_str)

        var dispatch_send_size = max_token_per_rank * dispatch_msg_size
        var dispatch_recv_size = (
            n_experts * max_token_per_rank * dispatch_msg_size
        )

        # Calculate buffer sizes for combine phase
        # Combine messages only contain the processed token
        comptime combine_msg_size = hidden_size * size_of[combine_dtype]()
        comptime combine_send_size = max_token_per_rank * combine_msg_size * min(
            n_experts, n_gpus_per_node * n_nodes * top_k
        )
        comptime combine_recv_size = top_k * max_token_per_rank * combine_msg_size

        # Initialize atomic counters to zero for synchronization
        # These counters coordinate work between different thread blocks.
        comptime assert (
            atomic_counters_0.static_spec.static_size
            == EPLocalSyncCounters[n_experts].total_size()
        ), "Atomic counters 0 size doesn't match expected size."
        var atomic_counters_0_buf = atomic_counters_0.to_device_buffer(gpu_ctx)
        gpu_ctx.enqueue_memset(atomic_counters_0_buf, Int32(0))

        comptime assert (
            atomic_counters_1.static_spec.static_size
            == EPLocalSyncCounters[n_experts].total_size()
        ), "Atomic counters 1 size doesn't match expected size."
        var atomic_counters_1_buf = atomic_counters_1.to_device_buffer(gpu_ctx)
        gpu_ctx.enqueue_memset(atomic_counters_1_buf, Int32(0))

        var dispatch_send_p: UnsafePointer[UInt8, MutUntrackedOrigin]
        var dispatch_recv_p: UnsafePointer[UInt8, MutUntrackedOrigin]
        var dispatch_recv_count_p: UnsafePointer[UInt64, MutUntrackedOrigin]

        var combine_send_p: Optional[UnsafePointer[UInt8, MutUntrackedOrigin]]
        var combine_recv_p: UnsafePointer[UInt8, MutUntrackedOrigin]
        var combine_recv_count_p: UnsafePointer[UInt64, MutUntrackedOrigin]

        comptime if n_nodes > 1:
            # Initialize the SHMEM library for this GPU
            comptime if has_amd_gpu_accelerator():
                shmem_init_thread_tcp(gpu_ctx, gpus_per_node=n_gpus_per_node)
            else:
                shmem_init_thread_mpi(gpu_ctx, gpus_per_node=n_gpus_per_node)

            # Allocate SHMEM buffers for dispatch phase
            dispatch_send_p = shmem_malloc[.uint8](dispatch_send_size)
            dispatch_recv_p = shmem_malloc[.uint8](dispatch_recv_size)
            dispatch_recv_count_p = shmem_malloc[.uint64](n_experts)

            # Allocate SHMEM buffers for combine phase
            combine_send_p = shmem_malloc[.uint8](combine_send_size)
            combine_recv_p = shmem_malloc[.uint8](combine_recv_size)
            combine_recv_count_p = shmem_malloc[.uint64](n_experts)

        else:
            if not is_p2p_enabled():
                raise Error("P2P is not supported on this system.")
            dispatch_send_p = gpu_ctx.enqueue_create_buffer[.uint8](
                dispatch_send_size
            ).take_ptr()
            dispatch_recv_p = gpu_ctx.enqueue_create_buffer[.uint8](
                dispatch_recv_size
            ).take_ptr()
            dispatch_recv_count_p = gpu_ctx.enqueue_create_buffer[.uint64](
                n_experts
            ).take_ptr()

            # When all the devices are on the same node, we skip the combine
            # send buffer and directly send tokens to each device's recv buffer.
            # Hence, we don't need to allocate the combine send buffer.
            combine_send_p = None
            combine_recv_p = gpu_ctx.enqueue_create_buffer[.uint8](
                combine_recv_size
            ).take_ptr()
            combine_recv_count_p = gpu_ctx.enqueue_create_buffer[.uint64](
                n_experts
            ).take_ptr()

        # Initialize receive count buffers to MAX_FINITE
        # This sentinel value indicates that no data has been received yet
        var dispatch_recv_count_buf = DeviceBuffer(
            gpu_ctx, dispatch_recv_count_p, n_experts, owning=False
        )
        gpu_ctx.enqueue_memset(dispatch_recv_count_buf, UInt64.MAX_FINITE)

        var combine_recv_count_buf = DeviceBuffer(
            gpu_ctx, combine_recv_count_p, n_experts, owning=False
        )
        gpu_ctx.enqueue_memset(combine_recv_count_buf, UInt64.MAX_FINITE)

        # Group 0: Dispatch phase buffer pointers
        dev_ptrs[0, 0] = UInt64(Int(dispatch_send_p))
        dev_ptrs[0, 1] = UInt64(Int(dispatch_recv_p))
        dev_ptrs[0, 2] = UInt64(Int(dispatch_recv_count_p))

        # Group 1: Combine phase buffer pointers
        dev_ptrs[1, 0] = UInt64(pointer_to_int(combine_send_p))
        dev_ptrs[1, 1] = UInt64(Int(combine_recv_p))
        dev_ptrs[1, 2] = UInt64(Int(combine_recv_count_p))

        # Store current device's rank
        var my_rank: Int32

        comptime if n_nodes > 1:
            my_rank = Int32(shmem_my_pe())
        else:
            my_rank = Int32(gpu_ctx.id())
        my_rank_tensor[0] = my_rank


@extensibility.register("ep.dispatch_async")
struct Struct_ep_dispatch_async:
    """Registers the `ep.dispatch_async` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_fmt_str: StaticString,
        //,
        target: StaticString,
    ](
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=input_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism async dispatch kernel. Tokens are
        transferred in either Blockwise FP8 or BF16 format.

        Parameters:
            input_dtype: `DType` of the input tokens before dispatch
                (inferred).
            dispatch_dtype: `DType` used for the quantized token
                payload during dispatch (inferred).
            dispatch_scale_dtype: `DType` of the block scales
                accompanying the dispatched tokens (inferred).
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            dispatch_fmt_str: String selecting the dispatch token
                format, either `"BlockwiseFP8"` or `"BF16"`
                (inferred).
            target: Compile-time device target.

        Args:
            atomic_counters: Atomic counters coordinating work across
                thread blocks during the dispatch phase.
            input_tokens: Input tokens to dispatch to experts. Shape
                `[num_tokens, hidden_size]`.
            topk_ids: Top-k expert IDs per token. Shape
                `[num_tokens, top_k]`.
            send_ptrs: Send buffer pointers for the dispatch phase.
            recv_ptrs: Receive buffer pointers for the dispatch
                phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                tokens received per expert.
            context: GPU device context for the current device.
        """

        comptime if dispatch_fmt_str == "BlockwiseFP8":
            comptime token_fmt_type = BlockwiseFP8TokenFormat[
                fp8_dtype=dispatch_dtype,
                scales_dtype=DType.float32,
                output_layout=RT_LAYOUT_2D,
                scales_layout=RT_LAYOUT_2D,
                hidden_size,
                top_k,
            ]
            ep_dispatch_async_kernel_api[
                token_fmt_type,
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                target,
            ](
                atomic_counters.to_tile_tensor[.int64](),
                input_tokens.to_tile_tensor[.int64]().as_immut(),
                topk_ids.to_tile_tensor[.int64]().as_immut(),
                send_ptrs.to_tile_tensor[.int64](),
                recv_ptrs.to_tile_tensor[.int64](),
                recv_count_ptrs.to_tile_tensor[.int64](),
                context,
            )

        elif dispatch_fmt_str == "BF16":
            comptime token_fmt_type = BF16TokenFormat[
                output_layout=RT_LAYOUT_2D, hidden_size, top_k
            ]

            ep_dispatch_async_kernel_api[
                token_fmt_type,
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                target,
            ](
                atomic_counters.to_tile_tensor[.int64](),
                input_tokens.to_tile_tensor[.int64]().as_immut(),
                topk_ids.to_tile_tensor[.int64]().as_immut(),
                send_ptrs.to_tile_tensor[.int64](),
                recv_ptrs.to_tile_tensor[.int64](),
                recv_count_ptrs.to_tile_tensor[.int64](),
                context,
            )

        else:
            raise Error("Invalid dispatch format string: ", dispatch_fmt_str)


@extensibility.register("ep.dispatch_async.block.scaled.nv")
struct Struct_ep_dispatch_async_block_scaled_nv:
    """Registers the `ep.dispatch_async.block.scaled.nv` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        target: StaticString,
    ](
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=input_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        input_scales: InputTensor[dtype=.float32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism async dispatch kernel. Tokens are
        transferred in NVFP4 format.

        Parameters:
            input_dtype: `DType` of the input tokens before dispatch
                (inferred).
            dispatch_dtype: `DType` used for the quantized token
                payload during dispatch (inferred).
            dispatch_scale_dtype: `DType` of the block scales
                accompanying the dispatched tokens (inferred).
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            target: Compile-time device target.

        Args:
            atomic_counters: Atomic counters coordinating work across
                thread blocks during the dispatch phase.
            input_tokens: Input tokens to dispatch to experts. Shape
                `[num_tokens, hidden_size]`.
            topk_ids: Top-k expert IDs per token. Shape
                `[num_tokens, top_k]`.
            send_ptrs: Send buffer pointers for the dispatch phase.
            recv_ptrs: Receive buffer pointers for the dispatch
                phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                tokens received per expert.
            input_scales: Global input scales for NVFP4 quantization.
            context: GPU device context for the current device.
        """
        var input_scales_tensor = input_scales.to_tile_tensor[.int64]()
        comptime assert input_scales_tensor.flat_rank == 1

        @__parameter
        @always_inline
        @__copy_capture(input_scales_tensor)
        def input_scales_fn[dtype: DType](expert_id: Int) -> Scalar[dtype]:
            # Currently only use one global input scale for all experts
            return rebind[Scalar[dtype]](input_scales_tensor[0].cast[dtype]())

        comptime token_fmt_type = NVBlockScaledTokenFormat[
            quant_dtype=dispatch_dtype,
            scales_dtype=dispatch_scale_dtype,
            output_layout=RT_LAYOUT_2D,
            scales_offset_layout=RT_LAYOUT_2D,
            hidden_size,
            top_k,
        ]
        ep_dispatch_async_kernel_api[
            token_fmt_type,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
            input_scales_wrapper=input_scales_fn,
        ](
            atomic_counters.to_tile_tensor[.int64](),
            input_tokens.to_tile_tensor[.int64]().as_immut(),
            topk_ids.to_tile_tensor[.int64]().as_immut(),
            send_ptrs.to_tile_tensor[.int64](),
            recv_ptrs.to_tile_tensor[.int64](),
            recv_count_ptrs.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("ep.dispatch_async.mxfp4")
struct Struct_ep_dispatch_async_mxfp4:
    """Registers the `ep.dispatch_async.mxfp4` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        target: StaticString,
        *,
        MX_FORMAT: StaticString = "auto",
    ](
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=input_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism async dispatch kernel. Tokens are
        transferred in MXFP4 format with per-token even-mode scales packed
        alongside the FP4 quants in the send buffer.

        Parameters:
            input_dtype: `DType` of the input tokens before dispatch
                (inferred).
            dispatch_dtype: `DType` used for the quantized token
                payload during dispatch (inferred).
            dispatch_scale_dtype: `DType` of the per-token even-mode
                scales accompanying the dispatched tokens (inferred).
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            target: Compile-time device target.
            MX_FORMAT: Name of the MX element format, or `auto` to take it
                from the dtype. See `MXFormat.from_name`. Names the MX
                encoding of the wire payload. It cannot be inferred from
                `dispatch_dtype`: MXFP4 and MXFP6 both travel as
                `DType.uint8` and differ only in bits per element. The
                default `auto` falls back to FP4 for a `uint8` payload and
                FP8 E4M3 otherwise.

        Args:
            atomic_counters: Atomic counters coordinating work across
                thread blocks during the dispatch phase.
            input_tokens: Input tokens to dispatch to experts. Shape
                `[num_tokens, hidden_size]`.
            topk_ids: Top-k expert IDs per token. Shape
                `[num_tokens, top_k]`.
            send_ptrs: Send buffer pointers for the dispatch phase.
            recv_ptrs: Receive buffer pointers for the dispatch
                phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                tokens received per expert.
            context: GPU device context for the current device.
        """
        comptime mx_fmt = MXFormat.from_dtype[
            dispatch_dtype
        ]() if MX_FORMAT == "auto" else (MXFormat.from_name[MX_FORMAT]())
        comptime token_fmt_type = MXTokenFormat[
            quant_dtype=dispatch_dtype,
            scales_dtype=dispatch_scale_dtype,
            output_layout=RT_LAYOUT_2D,
            scales_layout=RT_LAYOUT_2D,
            hidden_size,
            top_k,
            mx_format=mx_fmt,
        ]
        ep_dispatch_async_kernel_api[
            token_fmt_type,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
        ](
            atomic_counters.to_tile_tensor[.int64](),
            input_tokens.to_tile_tensor[.int64]().as_immut(),
            topk_ids.to_tile_tensor[.int64]().as_immut(),
            send_ptrs.to_tile_tensor[.int64](),
            recv_ptrs.to_tile_tensor[.int64](),
            recv_count_ptrs.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("ep.dispatch_wait")
struct Struct_ep_dispatch_wait:
    """Registers the `ep.dispatch_wait` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        target: StaticString,
        num_input_tokens: Int = -1,
    ](
        output_tokens: OutputTensor[dtype=.bfloat16, rank=2, ...],
        row_offsets: OutputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=.int32, rank=1, ...],
        src_info: OutputTensor[dtype=.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel. Received
        tokens are in BF16 format.
        """

        # Ensure the shape for the input tensors are correct
        comptime assert (
            Int(output_tokens.static_spec.shape_tuple[1]) == hidden_size
        ), "EP dispatch_wait: output tokens shape doesn't match hidden size."

        var format_handler = BF16TokenFormat[hidden_size, top_k](
            output_tokens.to_tile_tensor[.int64]()
        )

        ep_dispatch_wait_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            src_info.to_tile_tensor[.int64](),
            recv_ptrs.to_tile_tensor[.int64](),
            recv_count_ptrs.to_tile_tensor[.int64](),
            atomic_counters.to_tile_tensor[.int64](),
            context,
            num_input_tokens,
        )


@extensibility.register("ep.dispatch_wait.fp8")
struct Struct_ep_dispatch_wait_fp8:
    """Registers the `ep.dispatch_wait.fp8` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_granularity: StaticString,
        //,
        target: StaticString,
        num_input_tokens: Int = -1,
    ](
        output_tokens: OutputTensor[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputTensor[dtype=dispatch_scale_dtype, rank=2, ...],
        row_offsets: OutputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=.int32, rank=1, ...],
        src_info: OutputTensor[dtype=.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel. Received
        tokens are in Blockwise FP8 format.
        """

        var output_tokens_tensor = output_tokens.to_tile_tensor[.int64]()
        var output_scales_tensor = output_scales.to_tile_tensor[.int64]()
        # Ensure the shape for the input tensors are correct
        comptime assert (
            output_tokens_tensor.static_shape[1] == hidden_size
        ), "EP dispatch_wait: output tokens shape doesn't match hidden size."

        var format_handler = BlockwiseFP8TokenFormat[hidden_size, top_k](
            output_tokens_tensor,
            output_scales_tensor,
        )

        ep_dispatch_wait_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            src_info.to_tile_tensor[.int64](),
            recv_ptrs.to_tile_tensor[.int64](),
            recv_count_ptrs.to_tile_tensor[.int64](),
            atomic_counters.to_tile_tensor[.int64](),
            context,
            num_input_tokens,
        )


@extensibility.register("ep.dispatch_wait.block.scaled.nv")
struct Struct_ep_dispatch_wait_block_scaled_nv:
    """Registers the `ep.dispatch_wait.block.scaled.nv` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        target: StaticString,
        num_input_tokens: Int = -1,
    ](
        output_tokens: OutputTensor[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputTensor[dtype=dispatch_scale_dtype, rank=5, ...],
        row_offsets: OutputTensor[dtype=.uint32, rank=1, ...],
        scales_offsets: OutputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=.int32, rank=1, ...],
        src_info: OutputTensor[dtype=.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel. Received
        tokens are in NVFP4 format.
        """
        var output_tokens_tensor = output_tokens.to_tile_tensor[.int64]()
        var output_scales_tensor = output_scales.to_tile_tensor[.int64]()
        var scales_offsets_tensor = scales_offsets.to_tile_tensor[.int64]()

        comptime assert (
            output_tokens_tensor.static_shape[1] * 2 == hidden_size
        ), "EP dispatch_wait: output tokens shape doesn't match hidden size."

        var format_handler = NVBlockScaledTokenFormat[hidden_size, top_k](
            output_tokens_tensor,
            output_scales_tensor,
            scales_offsets_tensor,
            context,
        )

        ep_dispatch_wait_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            src_info.to_tile_tensor[.int64](),
            recv_ptrs.to_tile_tensor[.int64](),
            recv_count_ptrs.to_tile_tensor[.int64](),
            atomic_counters.to_tile_tensor[.int64](),
            context,
            num_input_tokens,
        )


@extensibility.register("ep.dispatch_wait.mxfp4")
struct Struct_ep_dispatch_wait_mxfp4:
    """Registers the `ep.dispatch_wait.mxfp4` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        target: StaticString,
        num_input_tokens: Int = -1,
        *,
        fuse_a_scale_preshuffle: Bool = False,
        max_padded_M: Int = 0,
        MX_FORMAT: StaticString = "auto",
    ](
        output_tokens: OutputTensor[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputTensor[dtype=dispatch_scale_dtype, rank=2, ...],
        row_offsets: OutputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=.int32, rank=1, ...],
        src_info: OutputTensor[dtype=.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism dispatch completion kernel. Received
        tokens are in MXFP4 format: two FP4 elements packed per ``uint8`` in
        ``output_tokens`` with per-token even-mode scales in ``output_scales``.

        When ``fuse_a_scale_preshuffle=True`` (KS224 up-proj fusion), the
        kernel writes the E8M0 activation scale directly into the up-proj
        grouped matmul's per-expert fixed-stride ``scale_4d`` slot layout (slot
        stride ``max_padded_M * K_SCALES``), so the standalone
        ``preshuffle_grouped_scale_4d_gpu`` can be dropped from the decode
        critical path. The scales output tensor must then have shape
        ``[n_local_experts * max_padded_M, K_SCALES]``.
        """
        var output_tokens_tensor = output_tokens.to_tile_tensor[.int64]()
        var output_scales_tensor = output_scales.to_tile_tensor[.int64]()

        comptime mx_fmt = MXFormat.from_dtype[
            dispatch_dtype
        ]() if MX_FORMAT == "auto" else (MXFormat.from_name[MX_FORMAT]())
        comptime assert (
            output_tokens_tensor.static_shape[1] * 8
            == hidden_size * mx_fmt.bits_per_element()
        ), "EP dispatch_wait: output tokens shape doesn't match hidden size."

        var format_handler = MXTokenFormat[
            hidden_size,
            top_k,
            fuse_a_scale_preshuffle=fuse_a_scale_preshuffle,
            mx_format=mx_fmt,
        ](
            output_tokens_tensor,
            output_scales_tensor,
            max_padded_M,
        )

        ep_dispatch_wait_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            src_info.to_tile_tensor[.int64](),
            recv_ptrs.to_tile_tensor[.int64](),
            recv_count_ptrs.to_tile_tensor[.int64](),
            atomic_counters.to_tile_tensor[.int64](),
            context,
            num_input_tokens,
        )


@extensibility.register("ep.dispatch")
struct Struct_ep_dispatch:
    """Registers the `ep.dispatch` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        dispatch_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        skip_a2a: Bool,
        allreduce_world_size: Int,
        //,
        target: StaticString,
    ](
        output_tokens: OutputTensor[dtype=.bfloat16, rank=2, ...],
        row_offsets: OutputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=.int32, rank=1, ...],
        src_info: OutputTensor[dtype=.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=dispatch_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the fused Expert Parallelism dispatch kernel.

        Routes tokens to experts based on top-k IDs, sends them to
        peer devices in BF16 format, waits for incoming tokens, and
        writes them to the output buffer along with their routing
        metadata.

        Parameters:
            dispatch_dtype: `DType` used for the token payload during
                dispatch (inferred).
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens any GPU can
                receive (inferred).
            n_gpus_per_node: Number of GPUs per physical node
                (inferred).
            n_nodes: Number of physical nodes in the deployment
                (inferred).
            fused_shared_expert: Whether a shared expert is fused
                into the dispatch kernel (inferred).
            skip_a2a: Whether to skip the all-to-all communication
                and send tokens only within the current device
                (inferred).
            allreduce_world_size: Number of ranks participating in
                the allreduce following dispatch (inferred).
            target: Compile-time device target.

        Args:
            output_tokens: Output tensor storing the received tokens
                in BF16 format. Shape `[num_tokens, hidden_size]`.
            row_offsets: Output tensor storing the row offsets for
                the received tokens.
            expert_ids: Output tensor storing the expert ID for
                each received token.
            src_info: Output tensor recording the originating rank
                and token index for each received token. Shape
                `[num_tokens, 2]`.
            atomic_counters: Atomic counters coordinating work
                across thread blocks during the dispatch phase.
            input_tokens: Input tokens to dispatch to experts. Shape
                `[num_tokens, hidden_size]`.
            topk_ids: Input tensor of top-k expert IDs per token.
                Shape `[num_tokens, top_k]`.
            send_ptrs: Send buffer pointers for the dispatch phase.
            recv_ptrs: Receive buffer pointers for the dispatch
                phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                tokens received per expert.
            context: GPU device context for the current device.
        """

        comptime assert dispatch_dtype == .bfloat16

        var output_tokens_tensor = output_tokens.to_tile_tensor[.int64]()
        var format_handler = BF16TokenFormat[hidden_size, top_k](
            output_tokens_tensor
        )

        ep_fused_dispatch_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
            target,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            src_info.to_tile_tensor[.int64](),
            atomic_counters.to_tile_tensor[.int64](),
            input_tokens.to_tile_tensor[.int64](),
            topk_ids.to_tile_tensor[.int64](),
            send_ptrs.to_tile_tensor[.int64](),
            recv_ptrs.to_tile_tensor[.int64](),
            recv_count_ptrs.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("ep.dispatch.fp8")
struct Struct_ep_dispatch_fp8:
    """Registers the `ep.dispatch.fp8` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_granularity: StaticString,
        fused_shared_expert: Bool,
        skip_a2a: Bool,
        allreduce_world_size: Int,
        //,
        target: StaticString,
    ](
        output_tokens: OutputTensor[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputTensor[dtype=dispatch_scale_dtype, rank=2, ...],
        row_offsets: OutputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=.int32, rank=1, ...],
        src_info: OutputTensor[dtype=.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=input_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the fused Expert Parallelism FP8 dispatch kernel. Tokens are
        dispatched in Blockwise FP8 format.
        """
        var output_tokens_tensor = output_tokens.to_tile_tensor[.int64]()
        var output_scales_tensor = output_scales.to_tile_tensor[.int64]()

        var format_handler = BlockwiseFP8TokenFormat[hidden_size, top_k](
            output_tokens_tensor, output_scales_tensor
        )

        ep_fused_dispatch_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
            target,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            src_info.to_tile_tensor[.int64](),
            atomic_counters.to_tile_tensor[.int64](),
            input_tokens.to_tile_tensor[.int64](),
            topk_ids.to_tile_tensor[.int64](),
            send_ptrs.to_tile_tensor[.int64](),
            recv_ptrs.to_tile_tensor[.int64](),
            recv_count_ptrs.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("ep.dispatch.block.scaled.nv")
struct Struct_ep_dispatch_block_scaled_nv:
    """Registers the `ep.dispatch.block.scaled.nv` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        skip_a2a: Bool,
        allreduce_world_size: Int,
        //,
        target: StaticString,
    ](
        output_tokens: OutputTensor[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputTensor[dtype=dispatch_scale_dtype, rank=5, ...],
        row_offsets: OutputTensor[dtype=.uint32, rank=1, ...],
        scales_offsets: OutputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=.int32, rank=1, ...],
        src_info: OutputTensor[dtype=.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=input_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        input_scales: InputTensor[dtype=.float32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the fused Expert Parallelism NVFP4 dispatch kernel. Tokens
        are dispatched in NVFP4 format.

        Parameters:
            input_dtype: `DType` of the input tokens before dispatch
                (inferred).
            dispatch_dtype: `DType` used for the quantized token
                payload during dispatch (inferred).
            dispatch_scale_dtype: `DType` of the block scales
                accompanying the dispatched tokens (inferred).
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            fused_shared_expert: Whether a shared expert is fused
                into the dispatch kernel (inferred).
            skip_a2a: Whether to skip the all-to-all communication
                and send tokens only within the current device
                (inferred).
            allreduce_world_size: Number of ranks participating in
                the allreduce following dispatch (inferred).
            target: Compile-time device target.

        Args:
            output_tokens: Output tensor storing the received tokens
                in NVFP4 format.
            output_scales: Output tensor storing the NVFP4 block
                scales for the received tokens.
            row_offsets: Output tensor storing the row offsets for
                the received tokens.
            scales_offsets: Output tensor storing the offsets into
                the scales buffer for the received tokens.
            expert_ids: Output tensor storing the expert ID for
                each received token.
            src_info: Output tensor recording the originating rank
                and token index for each received token. Shape
                `[num_tokens, 2]`.
            atomic_counters: Atomic counters coordinating work
                across thread blocks during the dispatch phase.
            input_tokens: Input tokens to dispatch to experts. Shape
                `[num_tokens, hidden_size]`.
            topk_ids: Input tensor of top-k expert IDs per token.
                Shape `[num_tokens, top_k]`.
            send_ptrs: Send buffer pointers for the dispatch phase.
            recv_ptrs: Receive buffer pointers for the dispatch
                phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                tokens received per expert.
            input_scales: Global input scales for NVFP4 quantization.
            context: GPU device context for the current device.
        """
        var output_tokens_tensor = output_tokens.to_tile_tensor[.int64]()
        var output_scales_tensor = output_scales.to_tile_tensor[.int64]()
        var scales_offsets_tensor = scales_offsets.to_tile_tensor[.int64]()
        var input_scales_tensor = input_scales.to_tile_tensor[.int64]()
        comptime assert input_scales_tensor.flat_rank == 1

        @__parameter
        @always_inline
        @__copy_capture(input_scales_tensor)
        def input_scales_fn[dtype: DType](expert_id: Int) -> Scalar[dtype]:
            # Currently only use one global input scale for all experts
            return rebind[Scalar[dtype]](input_scales_tensor[0].cast[dtype]())

        var format_handler = NVBlockScaledTokenFormat[hidden_size, top_k](
            output_tokens_tensor,
            output_scales_tensor,
            scales_offsets_tensor,
            context,
        )

        ep_fused_dispatch_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
            target,
            input_scales_wrapper=input_scales_fn,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            src_info.to_tile_tensor[.int64](),
            atomic_counters.to_tile_tensor[.int64](),
            input_tokens.to_tile_tensor[.int64](),
            topk_ids.to_tile_tensor[.int64](),
            send_ptrs.to_tile_tensor[.int64](),
            recv_ptrs.to_tile_tensor[.int64](),
            recv_count_ptrs.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("ep.dispatch.mxfp4")
struct Struct_ep_dispatch_mxfp4:
    """Registers the `ep.dispatch.mxfp4` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        skip_a2a: Bool,
        allreduce_world_size: Int,
        //,
        target: StaticString,
        *,
        fuse_a_scale_preshuffle: Bool = False,
        max_padded_M: Int = 0,
        MX_FORMAT: StaticString = "auto",
    ](
        output_tokens: OutputTensor[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputTensor[dtype=dispatch_scale_dtype, rank=2, ...],
        row_offsets: OutputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=.int32, rank=1, ...],
        src_info: OutputTensor[dtype=.int32, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=input_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the fused Expert Parallelism MXFP4 dispatch kernel. Tokens
        are dispatched in MXFP4 format.

        When ``fuse_a_scale_preshuffle=True`` (KS224 up-proj fusion), the
        wait-side copy writes the E8M0 activation scale directly into the
        up-proj grouped matmul's per-expert fixed-stride ``scale_4d`` slot
        layout (slot stride ``max_padded_M * K_SCALES``), dropping the standalone
        preshuffle. The scales output tensor must then be slot-sized
        (``[n_local_experts * max_padded_M, K_SCALES]``).
        """
        var output_tokens_tensor = output_tokens.to_tile_tensor[.int64]()
        var output_scales_tensor = output_scales.to_tile_tensor[.int64]()

        comptime mx_fmt = MXFormat.from_dtype[
            dispatch_dtype
        ]() if MX_FORMAT == "auto" else (MXFormat.from_name[MX_FORMAT]())
        var format_handler = MXTokenFormat[
            hidden_size,
            top_k,
            fuse_a_scale_preshuffle=fuse_a_scale_preshuffle,
            mx_format=mx_fmt,
        ](
            output_tokens_tensor,
            output_scales_tensor,
            max_padded_M,
        )

        ep_fused_dispatch_kernel_api[
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            fused_shared_expert,
            target,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            format_handler,
            row_offsets.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            src_info.to_tile_tensor[.int64](),
            atomic_counters.to_tile_tensor[.int64](),
            input_tokens.to_tile_tensor[.int64](),
            topk_ids.to_tile_tensor[.int64](),
            send_ptrs.to_tile_tensor[.int64](),
            recv_ptrs.to_tile_tensor[.int64](),
            recv_count_ptrs.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.distributed.ep.dispatch.block.scaled.nv")
struct DistributedEPDispatchBlockScaledNV:
    """Registers the `mo.distributed.ep.dispatch.block.scaled.nv` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        //,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output_tokens: OutputVariadicTensors[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputVariadicTensors[
            dtype=dispatch_scale_dtype, rank=5, ...
        ],
        row_offsets: OutputVariadicTensors[dtype=DType.uint32, rank=1, ...],
        scales_offsets: OutputVariadicTensors[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputVariadicTensors[dtype=DType.int32, rank=1, ...],
        src_info: OutputVariadicTensors[dtype=DType.int32, rank=2, ...],
        input_tokens: InputVariadicTensors[dtype=input_dtype, rank=2, ...],
        topk_ids: InputVariadicTensors[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        input_scales: InputVariadicTensors[dtype=DType.float32, rank=1, ...],
        atomic_counters: MutableInputVariadicTensors[
            dtype=DType.int32, rank=1, ...
        ],
        dev_ctxs: DeviceContextArray,
    ) capturing raises:
        """Multi-device fused Expert Parallelism NVFP4 dispatch.

        Launches the EP dispatch kernel on all devices simultaneously via
        _ep_launch_device_collective. Each device routes its tokens to experts
        based on top-k IDs, quantizes them to NVFP4 format, and sends them
        to the appropriate peer devices.

        Parameters:
            input_dtype: `DType` of the input tokens before quantization
                (inferred).
            dispatch_dtype: `DType` used for the quantized token payload
                during dispatch (inferred).
            dispatch_scale_dtype: `DType` of the block scales accompanying
                the dispatched tokens (inferred).
            hidden_size: Size of the model's hidden dimension (inferred).
            top_k: Number of experts each token is routed to (inferred).
            n_experts: Total number of experts across all GPUs (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            fused_shared_expert: Whether a shared expert is fused into the
                dispatch kernel (inferred).
            target: Compile-time device target.
            _trace_name: Trace label for this op.

        Args:
            output_tokens: Output variadic tensors storing the dispatched
                NVFP4-quantized tokens, one per device.
            output_scales: Output variadic tensors storing the NVFP4 block
                scales, one per device.
            row_offsets: Output variadic tensors storing the row offsets for
                the received tokens, one per device.
            scales_offsets: Output variadic tensors storing the offsets into
                the scales buffer, one per device.
            expert_ids: Output variadic tensors storing the expert ID for
                each received token, one per device.
            src_info: Output variadic tensors recording the originating
                rank and token index for each received token, one per
                device.
            input_tokens: Input variadic tensors of tokens to dispatch,
                one per device.
            topk_ids: Input variadic tensors of top-k expert IDs per token,
                one per device.
            send_ptrs: Send buffer pointers for the dispatch phase, one
                per device.
            recv_ptrs: Receive buffer pointers for the dispatch phase,
                one per device.
            recv_count_ptrs: Receive count buffer pointers tracking tokens
                received per expert, one per device.
            input_scales: Input variadic tensors of global input scales
                for NVFP4 quantization, one per device.
            atomic_counters: Atomic counters coordinating work across
                thread blocks, one per device.
            dev_ctxs: List of GPU device contexts, one per device.
        """
        comptime num_devices = input_tokens.size

        var gpu_ctxs = dev_ctxs.filter_gpu_contexts[num_devices]()

        @always_inline
        def launch_dispatch[
            index: Int
        ]() raises {
            imm output_tokens,
            imm output_scales,
            imm row_offsets,
            imm scales_offsets,
            imm expert_ids,
            imm src_info,
            imm atomic_counters,
            imm input_tokens,
            imm topk_ids,
            imm send_ptrs,
            imm recv_ptrs,
            imm recv_count_ptrs,
            imm input_scales,
            imm gpu_ctxs,
        }:
            var out_tokens = output_tokens[index].to_tile_tensor[.int64]()
            var out_scales = output_scales[index].to_tile_tensor[.int64]()
            var sc_offsets = scales_offsets[index].to_tile_tensor[.int64]()
            var in_scales = input_scales[index].to_tile_tensor[.int64]()
            comptime assert in_scales.flat_rank == 1

            @__parameter
            @always_inline
            @__copy_capture(in_scales)
            def input_scales_fn[dtype: DType](expert_id: Int) -> Scalar[dtype]:
                return rebind[Scalar[dtype]](in_scales[0].cast[dtype]())

            var format_handler = NVBlockScaledTokenFormat[hidden_size, top_k](
                out_tokens,
                out_scales,
                sc_offsets,
                gpu_ctxs[index],
            )

            ep_fused_dispatch_kernel_api[
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                fused_shared_expert,
                target,
                input_scales_wrapper=input_scales_fn,
            ](
                format_handler,
                row_offsets[index].to_tile_tensor[.int64](),
                expert_ids[index].to_tile_tensor[.int64](),
                src_info[index].to_tile_tensor[.int64](),
                atomic_counters[index].to_tile_tensor[.int64](),
                input_tokens[index].to_tile_tensor[.int64](),
                topk_ids[index].to_tile_tensor[.int64](),
                send_ptrs[index].to_tile_tensor[.int64](),
                recv_ptrs[index].to_tile_tensor[.int64](),
                recv_count_ptrs[index].to_tile_tensor[.int64](),
                gpu_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_dispatch, gpu_ctxs.copy())


@extensibility.register("mo.distributed.ep.dispatch.mxfp4")
struct DistributedEPDispatchMXFP4:
    """Registers the `mo.distributed.ep.dispatch.mxfp4` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        fuse_a_scale_preshuffle: Bool,
        max_padded_m: Int,
        //,
        target: StaticString,
        _trace_name: StaticString,
        *,
        mx_format: StaticString = "auto",
    ](
        output_tokens: OutputVariadicTensors[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputVariadicTensors[
            dtype=dispatch_scale_dtype, rank=2, ...
        ],
        row_offsets: OutputVariadicTensors[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputVariadicTensors[dtype=DType.int32, rank=1, ...],
        src_info: OutputVariadicTensors[dtype=DType.int32, rank=2, ...],
        input_tokens: InputVariadicTensors[dtype=input_dtype, rank=2, ...],
        topk_ids: InputVariadicTensors[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        atomic_counters: MutableInputVariadicTensors[
            dtype=DType.int32, rank=1, ...
        ],
        dev_ctxs: DeviceContextArray,
    ) capturing raises:
        """Multi-device fused Expert Parallelism MXFP4 dispatch.

        Launches the EP dispatch kernel on all devices simultaneously via
        _ep_launch_device_collective. Each device routes its tokens to experts
        based on top-k IDs, quantizes them to MXFP4 format, and sends them
        to the appropriate peer devices.

        Parameters:
            input_dtype: `DType` of the input tokens before quantization
                (inferred).
            dispatch_dtype: `DType` used for the quantized token payload
                during dispatch (inferred).
            dispatch_scale_dtype: `DType` of the per-token scales accompanying
                the dispatched tokens (inferred).
            hidden_size: Size of the model's hidden dimension (inferred).
            top_k: Number of experts each token is routed to (inferred).
            n_experts: Total number of experts across all GPUs (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            fused_shared_expert: Whether a shared expert is fused into the
                dispatch kernel (inferred).
            target: Compile-time device target.
            _trace_name: Trace label for this op.
            mx_format: Name of the MX element format, or `auto` to take it
                from the dtype. See `MXFormat.from_name`. Names the MX
                encoding of the wire payload. It cannot be inferred from
                `dispatch_dtype`: MXFP4 and MXFP6 both travel as
                `DType.uint8` and differ only in bits per element.

        Args:
            output_tokens: Output variadic tensors storing the dispatched
                MXFP4-quantized tokens, one per device.
            output_scales: Output variadic tensors storing the per-token
                E8M0 scales for the dispatched tokens, one per device.
            row_offsets: Output variadic tensors storing the row offsets for
                the received tokens, one per device.
            expert_ids: Output variadic tensors storing the expert ID for
                each received token, one per device.
            src_info: Output variadic tensors recording the originating
                rank and token index for each received token, one per
                device.
            input_tokens: Input variadic tensors of tokens to dispatch,
                one per device.
            topk_ids: Input variadic tensors of top-k expert IDs per token,
                one per device.
            send_ptrs: Send buffer pointers for the dispatch phase, one
                per device.
            recv_ptrs: Receive buffer pointers for the dispatch phase,
                one per device.
            recv_count_ptrs: Receive count buffer pointers tracking tokens
                received per expert, one per device.
            atomic_counters: Atomic counters coordinating work across
                thread blocks, one per device.
            dev_ctxs: List of GPU device contexts, one per device.
        """
        comptime num_devices = input_tokens.size

        var gpu_ctxs = dev_ctxs.filter_gpu_contexts[num_devices]()

        @always_inline
        def launch_dispatch[
            index: Int
        ]() raises {
            imm output_tokens,
            imm output_scales,
            imm row_offsets,
            imm expert_ids,
            imm src_info,
            imm atomic_counters,
            imm input_tokens,
            imm topk_ids,
            imm send_ptrs,
            imm recv_ptrs,
            imm recv_count_ptrs,
            imm gpu_ctxs,
        }:
            var out_tokens = output_tokens[index].to_tile_tensor[.int64]()
            var out_scales = output_scales[index].to_tile_tensor[.int64]()

            comptime mx_fmt = MXFormat.from_dtype[
                dispatch_dtype
            ]() if mx_format == "auto" else (MXFormat.from_name[mx_format]())
            var format_handler = MXTokenFormat[
                hidden_size,
                top_k,
                fuse_a_scale_preshuffle=fuse_a_scale_preshuffle,
                mx_format=mx_fmt,
            ](
                out_tokens,
                out_scales,
                max_padded_m,
            )

            ep_fused_dispatch_kernel_api[
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                fused_shared_expert,
                target,
            ](
                format_handler,
                row_offsets[index].to_tile_tensor[.int64](),
                expert_ids[index].to_tile_tensor[.int64](),
                src_info[index].to_tile_tensor[.int64](),
                atomic_counters[index].to_tile_tensor[.int64](),
                input_tokens[index].to_tile_tensor[.int64](),
                topk_ids[index].to_tile_tensor[.int64](),
                send_ptrs[index].to_tile_tensor[.int64](),
                recv_ptrs[index].to_tile_tensor[.int64](),
                recv_count_ptrs[index].to_tile_tensor[.int64](),
                gpu_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_dispatch, gpu_ctxs.copy())


@extensibility.register("mo.distributed.ep.dispatch")
struct DistributedEPDispatch:
    """Registers the `mo.distributed.ep.dispatch` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dispatch_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        //,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output_tokens: OutputVariadicTensors[dtype=DType.bfloat16, rank=2, ...],
        row_offsets: OutputVariadicTensors[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputVariadicTensors[dtype=DType.int32, rank=1, ...],
        src_info: OutputVariadicTensors[dtype=DType.int32, rank=2, ...],
        input_tokens: InputVariadicTensors[dtype=dispatch_dtype, rank=2, ...],
        topk_ids: InputVariadicTensors[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        atomic_counters: MutableInputVariadicTensors[
            dtype=DType.int32, rank=1, ...
        ],
        dev_ctxs: DeviceContextArray,
    ) capturing raises:
        """Multi-device fused Expert Parallelism BF16 dispatch.

        Parameters:
            dispatch_dtype: `DType` used for token dispatch to experts
                (inferred).
            hidden_size: Size of the model's hidden dimension (inferred).
            top_k: Number of experts each token is routed to (inferred).
            n_experts: Total number of experts across all GPUs (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            fused_shared_expert: Whether a shared expert is fused into the
                dispatch kernel (inferred).
            target: Compile-time device target.
            _trace_name: Trace label for this op.

        Args:
            output_tokens: Output variadic tensors storing the dispatched
                tokens, one per device.
            row_offsets: Output variadic tensors storing the row offsets for
                the received tokens, one per device.
            expert_ids: Output variadic tensors storing the expert ID for
                each received token, one per device.
            src_info: Output variadic tensors recording the originating
                rank and token index for each received token, one per
                device.
            input_tokens: Input variadic tensors of tokens to dispatch,
                one per device.
            topk_ids: Input variadic tensors of top-k expert IDs per token,
                one per device.
            send_ptrs: Send buffer pointers for the dispatch phase, one
                per device.
            recv_ptrs: Receive buffer pointers for the dispatch phase,
                one per device.
            recv_count_ptrs: Receive count buffer pointers tracking tokens
                received per expert, one per device.
            atomic_counters: Atomic counters coordinating work across
                thread blocks, one per device.
            dev_ctxs: List of GPU device contexts, one per device.
        """
        comptime num_devices = input_tokens.size
        comptime assert dispatch_dtype == .bfloat16

        var gpu_ctxs = dev_ctxs.filter_gpu_contexts[num_devices]()

        @always_inline
        def launch_dispatch[
            index: Int
        ]() raises {
            imm output_tokens,
            imm row_offsets,
            imm expert_ids,
            imm src_info,
            imm atomic_counters,
            imm input_tokens,
            imm topk_ids,
            imm send_ptrs,
            imm recv_ptrs,
            imm recv_count_ptrs,
            imm gpu_ctxs,
        }:
            var out_tokens = output_tokens[index].to_tile_tensor[.int64]()
            var format_handler = BF16TokenFormat[hidden_size, top_k](out_tokens)

            ep_fused_dispatch_kernel_api[
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                fused_shared_expert,
                target,
            ](
                format_handler,
                row_offsets[index].to_tile_tensor[.int64](),
                expert_ids[index].to_tile_tensor[.int64](),
                src_info[index].to_tile_tensor[.int64](),
                atomic_counters[index].to_tile_tensor[.int64](),
                input_tokens[index].to_tile_tensor[.int64](),
                topk_ids[index].to_tile_tensor[.int64](),
                send_ptrs[index].to_tile_tensor[.int64](),
                recv_ptrs[index].to_tile_tensor[.int64](),
                recv_count_ptrs[index].to_tile_tensor[.int64](),
                gpu_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_dispatch, gpu_ctxs.copy())


@extensibility.register("mo.distributed.ep.dispatch.fp8")
struct DistributedEPDispatchFP8:
    """Registers the `mo.distributed.ep.dispatch.fp8` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        input_dtype: DType,
        dispatch_dtype: DType,
        dispatch_scale_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_granularity: StaticString,
        fused_shared_expert: Bool,
        //,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output_tokens: OutputVariadicTensors[dtype=dispatch_dtype, rank=2, ...],
        output_scales: OutputVariadicTensors[
            dtype=dispatch_scale_dtype, rank=2, ...
        ],
        row_offsets: OutputVariadicTensors[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputVariadicTensors[dtype=DType.int32, rank=1, ...],
        src_info: OutputVariadicTensors[dtype=DType.int32, rank=2, ...],
        input_tokens: InputVariadicTensors[dtype=input_dtype, rank=2, ...],
        topk_ids: InputVariadicTensors[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        atomic_counters: MutableInputVariadicTensors[
            dtype=DType.int32, rank=1, ...
        ],
        dev_ctxs: DeviceContextArray,
    ) capturing raises:
        """Multi-device fused Expert Parallelism FP8 dispatch.

        Launches the EP dispatch kernel on all devices simultaneously via
        _launch_device_collective. Each device routes its tokens to experts
        based on top-k IDs, quantizes them to Blockwise FP8 format, and sends
        them to peer devices.

        Parameters:
            input_dtype: `DType` of the input tokens before quantization
                (inferred).
            dispatch_dtype: `DType` used for the quantized token payload
                during dispatch (inferred).
            dispatch_scale_dtype: `DType` of the block scales accompanying
                the dispatched tokens (inferred).
            hidden_size: Size of the model's hidden dimension (inferred).
            top_k: Number of experts each token is routed to (inferred).
            n_experts: Total number of experts across all GPUs (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            dispatch_scale_granularity: Block size used for the blockwise
                FP8 quantization scales (inferred).
            fused_shared_expert: Whether a shared expert is fused into the
                dispatch kernel (inferred).
            target: Compile-time device target.
            _trace_name: Trace label for this op.

        Args:
            output_tokens: Output variadic tensors storing the dispatched
                Blockwise FP8-quantized tokens, one per device.
            output_scales: Output variadic tensors storing the block scales
                for the dispatched tokens, one per device.
            row_offsets: Output variadic tensors storing the row offsets for
                the received tokens, one per device.
            expert_ids: Output variadic tensors storing the expert ID for
                each received token, one per device.
            src_info: Output variadic tensors recording the originating
                rank and token index for each received token, one per
                device.
            input_tokens: Input variadic tensors of tokens to dispatch,
                one per device.
            topk_ids: Input variadic tensors of top-k expert IDs per token,
                one per device.
            send_ptrs: Send buffer pointers for the dispatch phase, one
                per device.
            recv_ptrs: Receive buffer pointers for the dispatch phase,
                one per device.
            recv_count_ptrs: Receive count buffer pointers tracking tokens
                received per expert, one per device.
            atomic_counters: Atomic counters coordinating work across
                thread blocks, one per device.
            dev_ctxs: List of GPU device contexts, one per device.
        """
        comptime num_devices = input_tokens.size

        var gpu_ctxs = dev_ctxs.filter_gpu_contexts[num_devices]()

        @always_inline
        def launch_dispatch[
            index: Int
        ]() raises {
            imm output_tokens,
            imm output_scales,
            imm row_offsets,
            imm expert_ids,
            imm src_info,
            imm atomic_counters,
            imm input_tokens,
            imm topk_ids,
            imm send_ptrs,
            imm recv_ptrs,
            imm recv_count_ptrs,
            imm gpu_ctxs,
        }:
            var out_tokens = output_tokens[index].to_tile_tensor[.int64]()
            var out_scales = output_scales[index].to_tile_tensor[.int64]()
            var format_handler = BlockwiseFP8TokenFormat[hidden_size, top_k](
                out_tokens, out_scales
            )

            ep_fused_dispatch_kernel_api[
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                fused_shared_expert,
                target,
            ](
                format_handler,
                row_offsets[index].to_tile_tensor[.int64](),
                expert_ids[index].to_tile_tensor[.int64](),
                src_info[index].to_tile_tensor[.int64](),
                atomic_counters[index].to_tile_tensor[.int64](),
                input_tokens[index].to_tile_tensor[.int64](),
                topk_ids[index].to_tile_tensor[.int64](),
                send_ptrs[index].to_tile_tensor[.int64](),
                recv_ptrs[index].to_tile_tensor[.int64](),
                recv_count_ptrs[index].to_tile_tensor[.int64](),
                gpu_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_dispatch, gpu_ctxs.copy())


@extensibility.register("mo.distributed.ep.combine")
struct DistributedEPCombine:
    """Registers the `mo.distributed.ep.combine` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        combine_dtype: DType,
        router_weights_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        has_epilogue_fusion: Bool,
        //,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output_tokens: FusedOutputVariadicTensors[
            dtype=combine_dtype, rank=2, ...
        ],
        input_tokens: InputVariadicTensors[dtype=combine_dtype, rank=2, ...],
        src_info: InputVariadicTensors[dtype=DType.int32, rank=2, ...],
        send_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputVariadicTensors[dtype=DType.uint64, rank=1, ...],
        router_weights: InputVariadicTensors[
            dtype=router_weights_dtype, rank=2, ...
        ],
        atomic_counters: MutableInputVariadicTensors[
            dtype=DType.int32, rank=1, ...
        ],
        dev_ctxs: DeviceContextArray,
    ) capturing raises:
        """Multi-device fused Expert Parallelism combine with output fusion.

        Parameters:
            combine_dtype: `DType` of the combined expert output tokens.
            router_weights_dtype: `DType` of the router weights.
            hidden_size: Size of the model's hidden dimension.
            top_k: Number of experts each token is routed to.
            n_experts: Total number of experts across all GPUs.
            max_token_per_rank: Maximum number of tokens per GPU.
            n_gpus_per_node: Number of GPUs per node.
            n_nodes: Number of physical nodes.
            fused_shared_expert: Whether a shared expert is fused into the
                combine kernel.
            has_epilogue_fusion: Whether the combine output is fused with a
                downstream elementwise epilogue.
            target: Compile-time device target.
            _trace_name: Trace label for this op.

        Args:
            output_tokens: Fused output variadic tensors storing the combined
                expert outputs, one per device.
            input_tokens: Input variadic tensors of expert-processed tokens to
                combine, one per device.
            src_info: Source info tensors recording the originating rank and
                token index for each received token, one per device.
            send_ptrs: Send buffer pointers for the combine phase, one per
                device.
            recv_ptrs: Receive buffer pointers for the combine phase, one per
                device.
            recv_count_ptrs: Receive count buffer pointers tracking tokens
                received per expert, one per device.
            router_weights: Router weights used to scale each expert's
                contribution, one per device.
            atomic_counters: Atomic counters coordinating work across thread
                blocks, one per device.
            dev_ctxs: List of GPU device contexts, one per device.
        """
        comptime num_devices = input_tokens.size

        var gpu_ctxs = dev_ctxs.filter_gpu_contexts[num_devices]()

        @always_inline
        def launch_combine[
            index: Int
        ]() raises {
            imm output_tokens,
            imm input_tokens,
            imm src_info,
            imm send_ptrs,
            imm recv_ptrs,
            imm recv_count_ptrs,
            imm router_weights,
            imm atomic_counters,
            imm gpu_ctxs,
        }:
            var rw_tensor = router_weights[index].to_tile_tensor[.int64]()

            @__parameter
            @always_inline
            @__copy_capture(rw_tensor)
            def router_weights_fn[
                width: Int
            ](token_idx: Int, topk_id: Int) -> SIMD[.float32, width]:
                return rw_tensor.load[width=width]((token_idx, topk_id)).cast[
                    DType.float32
                ]()

            @__parameter
            @always_inline
            def output_fn[
                dtype: DType, width: SIMDLength, *, alignment: Int = 1
            ](coords: IndexList[2], val: SIMD[dtype, width]):
                output_tokens[index]._lambda_store[
                    width=width, element_alignment=alignment
                ](
                    coords,
                    rebind[SIMD[combine_dtype, width]](val),
                )

            ep_fused_combine_kernel_api[
                hidden_size,
                top_k,
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                target,
                router_weights_wrapper=router_weights_fn,
                epilogue_fn=Optional[elementwise_epilogue_type](
                    output_fn
                ) if has_epilogue_fusion else None,
                fused_shared_expert=fused_shared_expert,
            ](
                output_tokens[index].to_tile_tensor[.int64](),
                atomic_counters[index].to_tile_tensor[.int64](),
                input_tokens[index].to_tile_tensor[.int64](),
                src_info[index].to_tile_tensor[.int64](),
                send_ptrs[index].to_tile_tensor[.int64](),
                recv_ptrs[index].to_tile_tensor[.int64](),
                recv_count_ptrs[index].to_tile_tensor[.int64](),
                gpu_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_combine, gpu_ctxs.copy())


@extensibility.register("ep.combine_async")
struct Struct_ep_combine_async:
    """Registers the `ep.combine_async` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        combine_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        //,
        target: StaticString,
    ](
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=combine_dtype, rank=2, ...],
        src_info: InputTensor[dtype=.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism combine kernel.

        Sends expert-processed output tokens back to their original
        devices asynchronously, without waiting for the transfers to
        complete.

        Parameters:
            combine_dtype: `DType` used for the token payload during
                the combine phase (inferred).
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens per GPU
                (inferred).
            n_gpus_per_node: Number of GPUs per node (inferred).
            n_nodes: Number of physical nodes (inferred).
            target: Compile-time device target.

        Args:
            atomic_counters: Atomic counters coordinating work across
                thread blocks during the combine phase.
            input_tokens: Expert output tokens to send back to their
                original devices. Shape `[num_tokens, hidden_size]`.
            src_info: Source routing information from the dispatch
                phase recording the originating rank and token index
                for each token. Shape `[num_tokens, 2]`.
            send_ptrs: Send buffer pointers for the combine phase.
            recv_ptrs: Receive buffer pointers for the combine phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                the number of tokens received per expert.
            context: GPU device context for the current device.
        """

        ep_combine_async_kernel_api[
            combine_dtype,
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
        ](
            atomic_counters.to_tile_tensor[.int64](),
            input_tokens.to_tile_tensor[.int64](),
            src_info.to_tile_tensor[.int64](),
            send_ptrs.to_tile_tensor[.int64](),
            recv_ptrs.to_tile_tensor[.int64](),
            recv_count_ptrs.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("ep.combine_wait")
struct Struct_ep_combine_wait:
    """Registers the `ep.combine_wait` graph op with the graph compiler."""

    @__parameter
    @always_inline
    @staticmethod
    def execute[
        combine_dtype: DType,
        router_weights_dtype: DType,
        //,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        has_epilogue_fusion: Bool,
        target: StaticString,
    ](
        output_tokens: FusedOutputTensor[dtype=combine_dtype, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        router_weights: InputTensor[dtype=router_weights_dtype, rank=2, ...],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism combine completion kernel.

        Waits for incoming expert outputs and computes the weighted
        sum of routed expert outputs for each token using the router
        weights.

        Parameters:
            combine_dtype: `DType` used for the token payload during
                the combine phase (inferred).
            router_weights_dtype: `DType` of the router weights
                tensor used to compute the weighted sum (inferred).
            hidden_size: Size of the model's hidden dimension.
            top_k: Number of experts each token is routed to.
            n_experts: Total number of experts across all GPUs.
            max_token_per_rank: Maximum number of tokens any GPU can
                receive.
            n_gpus_per_node: Number of GPUs per physical node.
            n_nodes: Number of physical nodes in the deployment.
            has_epilogue_fusion: Whether to apply an elementwise
                epilogue function after computing the combined
                output.
            target: Compile-time device target.

        Args:
            output_tokens: Fused output tensor storing the weighted
                sum of routed expert outputs for each token. Shape
                `[num_tokens, hidden_size]`.
            atomic_counters: Atomic counters coordinating work
                across thread blocks during the combine phase.
            recv_ptrs: Receive buffer pointers for the combine
                phase.
            recv_count_ptrs: Receive count buffer pointers tracking
                the number of tokens received per expert.
            router_weights: Router weights for the current device
                used to compute the weighted sum of expert outputs.
                Shape `[num_tokens, top_k]`.
            context: GPU device context for the current device.
        """
        var router_weights_tensor = router_weights.to_tile_tensor[.int64]()
        comptime assert router_weights_tensor.flat_rank == 2
        comptime assert router_weights_tensor.flat_rank >= 2

        @__parameter
        @always_inline
        @__copy_capture(router_weights_tensor)
        def router_weights_fn[
            width: Int
        ](token_idx: Int, topk_id: Int) -> SIMD[.float32, width]:
            return router_weights_tensor.load[width=width](
                (token_idx, topk_id)
            ).cast[.float32]()

        @__parameter
        @always_inline
        def output_fn[
            dtype: DType, width: SIMDLength, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[dtype, width]):
            output_tokens._lambda_store[
                width=width, element_alignment=alignment
            ](
                coords,
                rebind[SIMD[combine_dtype, width]](val),
            )

        var output_tokens_tensor = output_tokens.to_tile_tensor[.int64]()
        # `output_tokens.dim(0)` is the per-token combine result count
        # (= num_input_tokens). Pass it to enable the decode-fast-path
        # grid sizing in `ep_combine_wait_kernel_api`.
        var num_input_tokens = Int(output_tokens_tensor.dim(0))
        ep_combine_wait_kernel_api[
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
            router_weights_wrapper=router_weights_fn,
            epilogue_fn=Optional[elementwise_epilogue_type](
                output_fn
            ) if has_epilogue_fusion else None,
        ](
            output_tokens_tensor,
            atomic_counters.to_tile_tensor[.int64](),
            recv_ptrs.to_tile_tensor[.int64](),
            recv_count_ptrs.to_tile_tensor[.int64](),
            context,
            num_input_tokens,
        )


@extensibility.register("ep.combine")
struct Struct_ep_combine:
    """Registers the `ep.combine` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        combine_dtype: DType,
        router_weights_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        has_epilogue_fusion: Bool,
        skip_a2a: Bool,
        //,
        target: StaticString,
    ](
        output_tokens: FusedOutputTensor[dtype=combine_dtype, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=combine_dtype, rank=2, ...],
        src_info: InputTensor[dtype=.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        router_weights: InputTensor[dtype=router_weights_dtype, rank=2, ...],
        context: DeviceContext,
    ) raises:
        """Execute the fused Expert Parallelism combine kernel.

        Sends expert outputs back to their original devices, waits for
        all transfers to complete, and computes the weighted sum of
        routed expert outputs for each token.

        Parameters:
            combine_dtype: `DType` used for the token payload during
                the combine phase (inferred).
            router_weights_dtype: `DType` of the router weights tensor
                used to compute the weighted sum (inferred).
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens any GPU can
                receive (inferred).
            n_gpus_per_node: Number of GPUs per physical node
                (inferred).
            n_nodes: Number of physical nodes in the deployment
                (inferred).
            fused_shared_expert: Whether a shared expert is fused into
                the combine kernel, adding its output to the routed
                expert outputs (inferred).
            has_epilogue_fusion: Whether to apply an elementwise
                epilogue function after computing the combined output
                (inferred).
            skip_a2a: Whether to skip the all-to-all communication and
                send tokens only within the current device (inferred).
            target: Compile-time device target.

        Args:
            output_tokens: Fused output tensor storing the weighted
                sum of routed expert outputs for each token. Shape
                `[num_tokens, hidden_size]`.
            atomic_counters: Atomic counters coordinating work across
                thread blocks during the combine phase.
            input_tokens: Expert output tokens to send back to their
                original devices. Shape `[num_tokens, hidden_size]`.
            src_info: Source routing information from the dispatch
                phase recording the originating rank and token index
                for each token. Shape `[num_tokens, 2]`.
            send_ptrs: Send buffer pointers for the combine phase,
                one per local GPU.
            recv_ptrs: Receive buffer pointers for the combine phase,
                one per local GPU.
            recv_count_ptrs: Receive count buffer pointers tracking the
                number of tokens received per expert.
            router_weights: Router weights for the current device used
                to compute the weighted sum of expert outputs. Shape
                `[num_tokens, top_k]`.
            context: GPU device context for the current device.
        """
        var router_weights_tensor = router_weights.to_tile_tensor[.int64]()
        comptime assert router_weights_tensor.flat_rank == 2
        comptime assert router_weights_tensor.flat_rank >= 2

        @__parameter
        @always_inline
        @__copy_capture(router_weights_tensor)
        def router_weights_fn[
            width: Int
        ](token_idx: Int, topk_id: Int) -> SIMD[.float32, width]:
            return router_weights_tensor.load[width=width](
                (token_idx, topk_id)
            ).cast[.float32]()

        @__parameter
        @always_inline
        def output_fn[
            dtype: DType, width: SIMDLength, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[dtype, width]):
            output_tokens._lambda_store[
                width=width, element_alignment=alignment
            ](
                coords,
                rebind[SIMD[combine_dtype, width]](val),
            )

        ep_fused_combine_kernel_api[
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
            router_weights_wrapper=router_weights_fn,
            epilogue_fn=Optional[elementwise_epilogue_type](
                output_fn
            ) if has_epilogue_fusion else None,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
        ](
            output_tokens.to_tile_tensor[.int64](),
            atomic_counters.to_tile_tensor[.int64](),
            input_tokens.to_tile_tensor[.int64](),
            src_info.to_tile_tensor[.int64](),
            send_ptrs.to_tile_tensor[.int64](),
            recv_ptrs.to_tile_tensor[.int64](),
            recv_count_ptrs.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("ep.combine.skip_a2a")
struct Struct_ep_combine_skip_a2a:
    """Registers the `ep.combine.skip_a2a` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        combine_dtype: DType,
        router_weights_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        fused_shared_expert: Bool,
        has_epilogue_fusion: Bool,
        skip_a2a: Bool,
        allreduce_world_size: Int,
        //,
        target: StaticString,
    ](
        output_tokens: FusedOutputTensor[dtype=combine_dtype, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=combine_dtype, rank=2, ...],
        src_info: InputTensor[dtype=.int32, rank=2, ...],
        send_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=.uint64, rank=1, ...],
        router_weights: InputTensor[dtype=router_weights_dtype, rank=2, ...],
        topk_ids: InputTensor[dtype=.int32, rank=2, ...],
        context: DeviceContext,
    ) raises:
        """Execute the fused Expert Parallelism combine kernel.

        Sends expert outputs back to their original devices, waits for
        all transfers to complete, and computes the weighted sum of
        routed expert outputs for each token. When `skip_a2a` is set,
        skips the all-to-all communication and reduces expert outputs
        locally using an allreduce of size `allreduce_world_size`.

        Parameters:
            combine_dtype: `DType` used for the token payload during
                the combine phase (inferred).
            router_weights_dtype: `DType` of the router weights
                tensor used to compute the weighted sum (inferred).
            hidden_size: Size of the model's hidden dimension
                (inferred).
            top_k: Number of experts each token is routed to
                (inferred).
            n_experts: Total number of experts across all GPUs
                (inferred).
            max_token_per_rank: Maximum number of tokens any GPU can
                receive (inferred).
            n_gpus_per_node: Number of GPUs per physical node
                (inferred).
            n_nodes: Number of physical nodes in the deployment
                (inferred).
            fused_shared_expert: Whether a shared expert is fused
                into the combine kernel, adding its output to the
                routed expert outputs (inferred).
            has_epilogue_fusion: Whether to apply an elementwise
                epilogue function after computing the combined
                output (inferred).
            skip_a2a: Whether to skip the all-to-all communication
                and reduce expert outputs locally instead
                (inferred).
            allreduce_world_size: World size for the local allreduce
                used when `skip_a2a` is set (inferred).
            target: Compile-time device target.

        Args:
            output_tokens: Fused output tensor storing the weighted
                sum of routed expert outputs for each token. Shape
                `[num_tokens, hidden_size]`.
            atomic_counters: Atomic counters coordinating work
                across thread blocks during the combine phase.
            input_tokens: Expert output tokens to send back to
                their original devices. Shape
                `[num_tokens, hidden_size]`.
            src_info: Source routing information from the dispatch
                phase recording the originating rank and token
                index for each token. Shape `[num_tokens, 2]`.
            send_ptrs: Send buffer pointers for the combine phase,
                one per local GPU.
            recv_ptrs: Receive buffer pointers for the combine
                phase, one per local GPU.
            recv_count_ptrs: Receive count buffer pointers tracking
                the number of tokens received per expert.
            router_weights: Router weights for the current device
                used to compute the weighted sum of expert outputs.
                Shape `[num_tokens, top_k]`.
            topk_ids: Top-k expert IDs selected for each token.
                Shape `[num_tokens, top_k]`.
            context: GPU device context for the current device.
        """
        var router_weights_tensor = router_weights.to_tile_tensor[.int64]()
        comptime assert router_weights_tensor.flat_rank == 2
        comptime assert router_weights_tensor.flat_rank >= 2

        @__parameter
        @always_inline
        @__copy_capture(router_weights_tensor)
        def router_weights_fn[
            width: Int
        ](token_idx: Int, topk_id: Int) -> SIMD[.float32, width]:
            return router_weights_tensor.load[width=width](
                (token_idx, topk_id)
            ).cast[.float32]()

        @__parameter
        @always_inline
        def output_fn[
            dtype: DType, width: SIMDLength, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[dtype, width]):
            output_tokens._lambda_store[
                width=width, element_alignment=alignment
            ](
                coords,
                rebind[SIMD[combine_dtype, width]](val),
            )

        ep_fused_combine_kernel_api[
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            target,
            router_weights_wrapper=router_weights_fn,
            epilogue_fn=Optional[elementwise_epilogue_type](
                output_fn
            ) if has_epilogue_fusion else None,
            fused_shared_expert=fused_shared_expert,
            skip_a2a=skip_a2a,
            allreduce_world_size=allreduce_world_size,
        ](
            output_tokens.to_tile_tensor[.int64](),
            atomic_counters.to_tile_tensor[.int64](),
            input_tokens.to_tile_tensor[.int64](),
            src_info.to_tile_tensor[.int64](),
            send_ptrs.to_tile_tensor[.int64](),
            recv_ptrs.to_tile_tensor[.int64](),
            recv_count_ptrs.to_tile_tensor[.int64](),
            context,
            topk_ids._ptr.as_imm().unsafe_origin_cast[ImmUntrackedOrigin](),
        )


@extensibility.register("ep.fused_silu")
struct Struct_ep_fused_silu:
    """Registers the `ep.fused_silu` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        output_dtype: DType,
        input_dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_dtype, rank=2, ...],
        input: InputTensor[dtype=input_dtype, rank=2, ...],
        row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism fused SILU kernel.

        This function launches the fused_silu kernel to perform the SILU
        operation for all the MLPs in the EP MoE module. We need to manually
        implement the custom operation here is because after the EP dispatch
        phase, the actual number of received tokens is not known to the host.

        This kernel will read the row offsets to determine the actual number of
        received tokens in the input tensor, and then only perform the SILU
        operation on the received tokens.
        """
        # Ensure this kernel only runs on GPU targets
        comptime assert is_gpu[target](), "EP is only supported on GPU."

        var output_tensor = output.to_tile_tensor[.int64]()
        var input_tensor = input.to_tile_tensor[.int64]().as_immut()
        var row_offsets_tensor = row_offsets.to_tile_tensor[
            DType.int64
        ]().as_immut()

        var gpu_ctx = context
        comptime hw_info = gpu_ctx.default_device_info

        comptime fused_silu = fused_silu_kernel[
            output_dtype,
            input_dtype,
            output_tensor.LayoutType,
            input_tensor.LayoutType,
            row_offsets_tensor.LayoutType,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
        ]

        @always_inline
        def description_fn() {imm} -> String:
            # fmt: off
            return String(
                "output_dtype=", output_dtype,
                ";input_dtype=", input_dtype,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.fused_silu",
            Trace[TraceLevel.OP]._get_detail_str(description_fn),
            task_id=get_safe_task_id(context),
        ):
            gpu_ctx.enqueue_function[fused_silu](
                output_tensor,
                input_tensor,
                row_offsets_tensor,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )


@extensibility.register("ep.fused_silu.fp8")
struct Struct_ep_fused_silu_fp8:
    """Registers the `ep.fused_silu.fp8` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        fp8_dtype: DType,
        scales_dtype: DType,
        input_dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=fp8_dtype, rank=2, ...],
        scales: OutputTensor[dtype=scales_dtype, rank=2, ...],
        input: InputTensor[dtype=input_dtype, rank=2, ...],
        row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism fused SILU kernel with FP8
        quantization.

        This function launches the fused_silu_fp8 kernel to perform the SILU
        operation for all the MLPs in the EP MoE module.

        This kernel will read the row offsets to determine the actual number of
        received tokens in the input tensor, and then only perform the SILU
        operation on the received tokens. Once the SILU operation is performed,
        the output will be quantized to the FP8 format. The scales tensor
        will be stored in a transposed way.
        """
        # Ensure this kernel only runs on GPU targets
        comptime assert is_gpu[target](), "EP is only supported on GPU."

        comptime group_size = 128

        var output_tensor = output.to_tile_tensor[.int64]()
        var scales_tensor = scales.to_tile_tensor[.int64]()
        var input_tensor = input.to_tile_tensor[.int64]().as_immut()
        var row_offsets_tensor = row_offsets.to_tile_tensor[
            DType.int64
        ]().as_immut()

        var gpu_ctx = context
        comptime hw_info = gpu_ctx.default_device_info

        comptime fused_silu_fp8 = fused_silu_fp8_kernel[
            fp8_dtype,
            scales_dtype,
            input_dtype,
            output_tensor.LayoutType,
            scales_tensor.LayoutType,
            input_tensor.LayoutType,
            row_offsets_tensor.LayoutType,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
            group_size,
        ]

        @always_inline
        def description_fn() {imm} -> String:
            # fmt: off
            return String(
                "fp8_dtype=", fp8_dtype,
                ";scales_dtype=", scales_dtype,
                ";input_dtype=", input_dtype,
                ";group_size=", group_size,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.fused_silu.fp8",
            Trace[TraceLevel.OP]._get_detail_str(description_fn),
            task_id=get_safe_task_id(context),
        ):
            gpu_ctx.enqueue_function[fused_silu_fp8](
                output_tensor,
                scales_tensor,
                input_tensor,
                row_offsets_tensor,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )


@extensibility.register("ep.fused_silu.mxfp4")
struct Struct_ep_fused_silu_mxfp4:
    """Registers the `ep.fused_silu.mxfp4` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        quant_dtype: DType,
        scales_dtype: DType,
        input_dtype: DType,
        target: StaticString,
        *,
        fuse_a_scale_preshuffle: Bool = False,
        max_padded_M: Int = 0,
        clamp_activation: Bool = False,
        # `ep.fused_silu.mxfp8` shares this body and overrides the label.
        trace_name: StaticString = "ep.fused_silu.mxfp4",
    ](
        output: OutputTensor[dtype=quant_dtype, rank=2, ...],
        scales: OutputTensor[dtype=scales_dtype, rank=2, ...],
        input: InputTensor[dtype=input_dtype, rank=2, ...],
        row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        # Clamped-SwiGLU alpha/L (trailing CPU f32 constants); unused when
        # clamp_activation=False.
        alpha: Float32,
        limit: Float32,
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism fused SILU kernel with MX
        quantization.

        Shared body for `ep.fused_silu.mxfp4` and `ep.fused_silu.mxfp8`:
        `fused_silu_mx_kernel` takes its element packing from `quant_dtype`.
        `trace_name` only labels the op in traces; it is defaulted and no
        caller sets it through MOGG's `parameters` dict, so the mxfp8
        registration overrides it directly at the Mojo call site.

        This function launches the shared `fused_silu_mx_kernel` to perform the SILU
        operation for all the MLPs in the EP MoE module.

        This kernel will read the row offsets to determine the actual number of
        received tokens in the input tensor, and then only perform the SILU
        operation on the received tokens. Once the SILU operation is performed,
        the output will be quantized to the MXFP4 or MXFP8 format.

        When `fuse_a_scale_preshuffle=True` (KS64 fusion), the kernel
        writes the E8M0 scale directly into the grouped matmul's per-expert
        fixed-stride `scale_4d` slot layout (slot stride `max_padded_M *
        K_SCALES`), so the standalone `preshuffle_grouped_scale_4d_gpu` can be
        omitted from the critical path. The scales output tensor must then have
        shape `[n_local_experts * max_padded_M, K_SCALES]`.
        """
        # Ensure this kernel only runs on GPU targets
        comptime assert is_gpu[target](), "EP is only supported on GPU."

        var output_tensor = output.to_tile_tensor[.int64]()
        var scales_tensor = scales.to_tile_tensor[.int64]()
        var input_tensor = input.to_tile_tensor[.int64]().as_immut()
        var row_offsets_tensor = row_offsets.to_tile_tensor[
            DType.int64
        ]().as_immut()

        var gpu_ctx = context
        comptime hw_info = gpu_ctx.default_device_info

        comptime fused_silu_mx = fused_silu_mx_kernel[
            quant_dtype,
            scales_dtype,
            input_dtype,
            output_tensor.LayoutType,
            scales_tensor.LayoutType,
            input_tensor.LayoutType,
            row_offsets_tensor.LayoutType,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
            fuse_a_scale_preshuffle=fuse_a_scale_preshuffle,
            clamp_activation=clamp_activation,
        ]

        @always_inline
        def description_fn() {imm} -> String:
            # fmt: off
            return String(
                "quant_dtype=", quant_dtype,
                ";scales_dtype=", scales_dtype,
                ";input_dtype=", input_dtype,
                ";fuse_a_scale_preshuffle=", fuse_a_scale_preshuffle,
                ";clamp_activation=", clamp_activation,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            trace_name,
            Trace[TraceLevel.OP]._get_detail_str(description_fn),
            task_id=get_safe_task_id(context),
        ):
            gpu_ctx.enqueue_function[fused_silu_mx](
                output_tensor,
                scales_tensor,
                input_tensor,
                row_offsets_tensor,
                Int32(max_padded_M),
                alpha,
                limit,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )


@extensibility.register("ep.fused_silu.mxfp8")
struct Struct_ep_fused_silu_mxfp8:
    """Registers the `ep.fused_silu.mxfp8` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        fp8_dtype: DType,
        scales_dtype: DType,
        input_dtype: DType,
        target: StaticString,
        *,
        fuse_a_scale_preshuffle: Bool = False,
        max_padded_M: Int = 0,
        clamp_activation: Bool = False,
    ](
        output: OutputTensor[dtype=fp8_dtype, rank=2, ...],
        scales: OutputTensor[dtype=scales_dtype, rank=2, ...],
        input: InputTensor[dtype=input_dtype, rank=2, ...],
        row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        alpha: Float32,
        limit: Float32,
        context: DeviceContext,
    ) raises:
        """Execute the EP fused SILU kernel with MXFP8 quantization.

        Same body as `ep.fused_silu.mxfp4`: `fused_silu_mx_kernel` takes its
        element packing from the output dtype, so one `fp8_e4m3fn` byte per
        element here rather than two FP4 nibbles, leaving `output` at the full
        hidden size along axis 1. With `fuse_a_scale_preshuffle`, `scales` must
        be shaped `[n_local_experts * max_padded_M, K_SCALES]`.
        """
        Struct_ep_fused_silu_mxfp4.execute[
            target=target,
            fuse_a_scale_preshuffle=fuse_a_scale_preshuffle,
            max_padded_M=max_padded_M,
            clamp_activation=clamp_activation,
            trace_name="ep.fused_silu.mxfp8",
        ](output, scales, input, row_offsets, alpha, limit, context)


@extensibility.register("ep.fused_silu.mxfp6")
struct Struct_ep_fused_silu_mxfp6:
    """Registers the `ep.fused_silu.mxfp6` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        scales_dtype: DType,
        input_dtype: DType,
        target: StaticString,
        *,
        FP6_FORMAT: Int = 0,
        fuse_a_scale_preshuffle: Bool = False,
        max_padded_M: Int = 0,
        clamp_activation: Bool = False,
    ](
        output: OutputTensor[dtype=.uint8, rank=2, ...],
        scales: OutputTensor[dtype=scales_dtype, rank=2, ...],
        input: InputTensor[dtype=input_dtype, rank=2, ...],
        row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        # Clamped-SwiGLU alpha/L (trailing CPU f32 constants); unused when
        # clamp_activation=False.
        alpha: Float32,
        limit: Float32,
        context: DeviceContext,
    ) raises:
        """Execute the EP fused SILU kernel with MXFP6 quantization.

        Unlike `ep.fused_silu.mxfp4` and `.mxfp8`, which share one body that
        switches on elements-per-byte, FP6 needs its own kernel: four codes per
        three bytes is a ratio of 4/3, which that integer cannot represent.
        `output` is packed uint8 at three quarters of the hidden size.

        `FP6_FORMAT` selects the element encoding (0 = E2M3, 1 = E3M2). Both
        occupy six bits and pack identically, so nothing downstream can recover
        it from the bytes -- it must match what the checkpoint declares.
        """
        comptime assert is_gpu[target](), "EP is only supported on GPU."
        comptime assert not fuse_a_scale_preshuffle or (
            max_padded_M > 0 and max_padded_M % 32 == 0
        ), (
            "the A-scale fold needs the graph-build-time slot stride:"
            " max_padded_M must be a positive multiple of 32, since the"
            " matmul reads with align_up(max_num_tokens_per_expert, 32)"
        )

        var output_tensor = output.to_tile_tensor[.int64]()
        var scales_tensor = scales.to_tile_tensor[.int64]()
        var input_tensor = input.to_tile_tensor[.int64]().as_immut()
        var row_offsets_tensor = row_offsets.to_tile_tensor[
            DType.int64
        ]().as_immut()

        var gpu_ctx = context
        comptime hw_info = gpu_ctx.default_device_info

        comptime fused_silu_fp6 = fused_silu_mxfp6_kernel[
            scales_dtype,
            input_dtype,
            output_tensor.LayoutType,
            scales_tensor.LayoutType,
            input_tensor.LayoutType,
            row_offsets_tensor.LayoutType,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
            fp6_format=FP6Format(FP6_FORMAT),
            fuse_a_scale_preshuffle=fuse_a_scale_preshuffle,
            clamp_activation=clamp_activation,
        ]

        @always_inline
        def description_fn() {imm} -> String:
            # fmt: off
            return String(
                "scales_dtype=", scales_dtype,
                ";input_dtype=", input_dtype,
                ";FP6_FORMAT=", FP6_FORMAT,
                ";fuse_a_scale_preshuffle=", fuse_a_scale_preshuffle,
                ";clamp_activation=", clamp_activation,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.fused_silu.mxfp6",
            Trace[TraceLevel.OP]._get_detail_str(description_fn),
            task_id=get_safe_task_id(context),
        ):
            gpu_ctx.enqueue_function[fused_silu_fp6](
                output_tensor,
                scales_tensor,
                input_tensor,
                row_offsets_tensor,
                Int32(max_padded_M),
                alpha,
                limit,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )


@extensibility.register("ep.fused_silu.nvfp4")
struct Struct_ep_fused_silu_nvfp4:
    """Registers the `ep.fused_silu.nvfp4` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        fp4_dtype: DType,
        scales_dtype: DType,
        input_dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=fp4_dtype, rank=2, ...],
        scales: OutputTensor[dtype=scales_dtype, rank=5, ...],
        input: InputTensor[dtype=input_dtype, rank=2, ...],
        row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        scales_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        input_scales: InputTensor[dtype=.float32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Execute the Expert Parallelism fused SILU kernel with NVFP4
        quantization.

        This function launches the fused_silu_nvfp4 kernel to perform the SILU
        operation for all the MLPs in the EP MoE module.

        This kernel will read the row offsets to determine the actual number of
        received tokens in the input tensor, and then only perform the SILU
        operation on the received tokens. Once the SILU operation is performed,
        the output will be quantized to the NVFP4 format. The scales tensor
        will be padded and zero-filled.
        """
        # Ensure this kernel only runs on GPU targets
        comptime assert is_gpu[target](), "EP is only supported on GPU."

        var output_tensor = output.to_tile_tensor[.int64]()
        var scales_tensor = scales.to_tile_tensor[.int64]()
        var input_tensor = input.to_tile_tensor[.int64]().as_immut()
        var row_offsets_tensor = row_offsets.to_tile_tensor[
            DType.int64
        ]().as_immut()
        var scales_offsets_tensor = scales_offsets.to_tile_tensor[
            DType.int64
        ]().as_immut()
        var input_scales_tensor = input_scales.to_tile_tensor[
            DType.int64
        ]().as_immut()

        var gpu_ctx = context
        comptime hw_info = gpu_ctx.default_device_info

        comptime fused_silu_nvfp4 = fused_silu_nvfp4_kernel[
            fp4_dtype,
            scales_dtype,
            input_dtype,
            output_tensor.LayoutType,
            scales_tensor.LayoutType,
            input_tensor.LayoutType,
            row_offsets_tensor.LayoutType,
            scales_offsets_tensor.LayoutType,
            input_scales_tensor.LayoutType,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
        ]

        @always_inline
        def description_fn() {imm} -> String:
            # fmt: off
            return String(
                "fp4_dtype=", fp4_dtype,
                ";scales_dtype=", scales_dtype,
                ";input_dtype=", input_dtype,
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            "ep.fused_silu.nvfp4",
            Trace[TraceLevel.OP]._get_detail_str(description_fn),
            task_id=get_safe_task_id(context),
        ):
            gpu_ctx.enqueue_function[fused_silu_nvfp4](
                output_tensor,
                scales_tensor,
                input_tensor,
                row_offsets_tensor,
                scales_offsets_tensor,
                input_scales_tensor,
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
