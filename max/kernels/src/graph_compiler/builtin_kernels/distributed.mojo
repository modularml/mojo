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


# ===-----------------------------------------------------------------------===#
# General imports
# ===-----------------------------------------------------------------------===#

"""Registers distributed and multi-GPU collective graph ops backed by the `comm` and `shmem` kernels."""

from std.math import align_down, ceildiv
from std.sys import get_defined_bool
from std.sys.info import size_of
import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from comm.allgather import allgather
from comm.allreduce import allreduce

from comm.allreduce_lamport_rmsnorm import lamport_allreduce_rmsnorm
from comm.allreduce_residual_rmsnorm import allreduce_residual_rmsnorm
from comm.allgather_rmsnorm import _dispatch_ag_norm, _dispatch_ag_norm_quant
from linalg.block_scaled_quantization import (
    quantize_mx_amd,
    quantize_mxfp8_lane_group,
)
from linalg.fp4_utils import MXFP8_SF_VECTOR_SIZE
from comm.lamport import Lamport
from max.gpu import WARP_SIZE
from comm.reducescatter import ReduceScatterConfig, reducescatter
from comm.reducescatter_rmsnorm import _dispatch_rs_norm, reducescatter_rmsnorm
from nn.normalization import rms_norm_gpu
from comm.broadcast import broadcast
from comm.scatter import scatter
from comm import MAX_GPUS, Signal
import comm.vendor.ccl as vendor_ccl
from max.gpu.host import DeviceContext, DeviceContextArray
from max.gpu.primitives.grid_controls import PDLLevel
from layout.tile_tensor import row_major
from layout import Coord, TileTensor, coord_to_index_list, row_major
from extensibility import (
    InputTensor,
    InputVariadicTensors,
    OutputTensor,
    OutputVariadicTensors,
)
from extensibility import (
    _FusedOutputVariadicTensors as FusedOutputVariadicTensors,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from extensibility import (
    _MutableInputVariadicTensors as MutableInputVariadicTensors,
)
from std.memory import UnsafePointer
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList
from std.utils.index import Index
from std.collections import Array, Optional

from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from linalg.utils import (
    elementwise_compute_lambda_type as matmul_elementwise_compute_lambda_type,
)
from matmul_rs.matmul_reducescatter import matmul_reducescatter_dispatch

# ===-----------------------------------------------------------------------===#
from .kernels import *
from .kernels import (
    _check_signal_buffer_size,
    _launch_device_collective,
    _partitioned_scratch_requirement,
)


@extensibility.register("mo.distributed.allreduce.sum")
struct DistributedAllReduceSum:
    """Registers the `mo.distributed.allreduce.sum` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
        group_size: Int = 0,
    ](
        outputs: FusedOutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextArray,
    ) capturing raises:
        """Distributed allreduce operation implementation for sum reduction.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Tensor rank (number of dimensions) of the inputs and outputs.
            target: Target device string for tracing.
            _trace_name: Trace name for profiling.
            group_size: Number of devices per independent allreduce group;
                must evenly divide the total number of devices. Zero (the
                attribute default) means all devices form one group.

        Args:
            outputs: Output tensors (one per GPU) to store reduced results.
            inputs: Input tensors (one per GPU) containing values to reduce.
            signal_buffers: Preallocated synchronization buffers for cross-GPU coordination.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Maximum of 8 GPUs supported (matches MAX_GPUS in comm/sync.mojo)
            - Tensor element count must be multiple of SIMD width (per allreduce.mojo)
            - Requires identical tensor shapes within each allreduce group
        """
        comptime num_devices = inputs.size
        comptime assert signal_buffers.size == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )
        comptime effective_group_size = (
            num_devices if group_size == 0 else group_size
        )
        comptime assert (
            effective_group_size >= 1
        ), "group_size must be at least 1"
        comptime assert (
            num_devices % effective_group_size == 0
        ), "group_size must evenly divide the number of devices"
        # Full-world collectives keep barrier domain 0; grouped collectives
        # get a distinct nonzero domain so their barrier counters never
        # poison the full-world bank on the shared Signal buffers.
        comptime domain_id = (
            0 if effective_group_size == num_devices else effective_group_size
        )

        # allreduce 2-stage uses size/ngpus scratch space; check per group
        # since groups may carry different row counts.
        comptime for g in range(num_devices // effective_group_size):
            comptime group_start = g * effective_group_size
            var scratch_buffer_size_bytes = _partitioned_scratch_requirement[
                effective_group_size, dtype
            ](inputs[group_start].size())
            _check_signal_buffer_size(
                signal_buffers[group_start].size(), scratch_buffer_size_bytes
            )

        # output_lambda writes each device's reduced output into the fused
        # epilogue output tensor. Defined at execute scope so that
        # epilogue_wrapper in vendor_ccl.allreduce (also execute scope) can
        # call it without triggering the MLIR 'kgen.param.declare.region must
        # have subprogram scope' error that arises when parameterized functions
        # are defined inside closures.
        @always_inline
        @__parameter
        def output_lambda[
            output_index: Int,
            _dtype: DType,
            _width: SIMDLength,
            *,
            _alignment: Int,
        ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
            outputs[output_index]._lambda_store[
                width=_width, element_alignment=_alignment
            ](
                rebind[IndexList[rank]](coord_to_index_list(coords)),
                rebind[SIMD[dtype, _width]](val),
            )

        # Marshal signal buffers into the expected format.
        var rank_sigs = Array[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )
        comptime for i in range(num_devices):
            rank_sigs[i] = (
                signal_buffers[i]._ptr.bitcast[Signal]().as_unsafe_any_origin()
            )

        comptime if get_defined_bool["MODULAR_USE_VENDOR_CCL", False]():
            comptime assert (
                effective_group_size == num_devices
            ), "grouped allreduce is not supported on the vendor CCL path"
            logger.info("Executing: Vendor CCL")
            comptime InputTensorType = type_of(
                inputs[0].to_tile_tensor[.int64]().as_immut()
            )
            var in_tensors = Array[InputTensorType, num_devices](
                uninitialized=True
            )
            comptime for i in range(num_devices):
                in_tensors[i] = rebind[InputTensorType](
                    inputs[i].to_tile_tensor[.int64]().as_immut()
                )

            @always_inline
            def launch_vendor_allreduce[
                index: Int
            ]() raises {
                imm in_tensors,
                imm rank_sigs,
                imm dev_ctxs_input,
                imm outputs,
            }:
                # _get_global_comms has a check-then-create race: two
                # threads seeing null simultaneously would both call
                # ncclCommInitAll and leak one set of communicators.
                # Only device 0 initializes; others spin-wait.
                comptime if index == 0:
                    vendor_ccl.init_comms(num_devices)
                else:
                    vendor_ccl.wait_for_comms(num_devices)

                vendor_ccl.allreduce[
                    ngpus=num_devices,
                    output_lambda=output_lambda[output_index=index, ...],
                ](
                    in_tensors,
                    outputs[index].to_tile_tensor[.int64](),
                    rank_sigs,
                    dev_ctxs_input[index],
                )

            _launch_device_collective[num_devices](
                launch_vendor_allreduce, dev_ctxs_input.copy()
            )
            return

        # Custom allreduce path. Each launch builds its own group-local input
        # and signal arrays so groups can carry different (symbolic) shapes,
        # mirroring `DistributedReduceScatterSum`.
        @always_inline
        def launch_allreduce[
            index: Int
        ]() raises {
            imm inputs,
            imm rank_sigs,
            imm dev_ctxs_input,
            imm outputs,
        }:
            comptime group_id, local_rank = divmod(index, effective_group_size)
            comptime group_start = group_id * effective_group_size
            comptime InputTensorType = type_of(
                inputs[group_start].to_tile_tensor[DType.int64]().as_immut()
            )

            var in_tensors = Array[InputTensorType, effective_group_size](
                uninitialized=True
            )
            var group_sigs = Array[
                UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
            ](uninitialized=True)
            comptime for i in range(effective_group_size):
                in_tensors[i] = rebind[InputTensorType](
                    inputs[group_start + i]
                    .to_tile_tensor[DType.int64]()
                    .as_immut()
                )
                group_sigs[i] = rank_sigs[group_start + i]

            var out_buf = outputs[index].to_tile_tensor[DType.int64]()
            allreduce[
                ngpus=effective_group_size,
                output_lambda=output_lambda[output_index=index, ...],
                domain_id=domain_id,
            ](
                in_tensors,
                out_buf,
                group_sigs,
                dev_ctxs_input[index],
                local_rank=local_rank,
            )

        _launch_device_collective[num_devices](
            launch_allreduce, dev_ctxs_input.copy()
        )


@extensibility.register("mo.distributed.reducescatter.sum")
struct DistributedReduceScatterSum:
    """Registers the `mo.distributed.reducescatter.sum` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
        axis: Int = 0,
        group_size: Int = 0,
    ](
        outputs: FusedOutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextArray,
    ) capturing raises:
        """Distributed reduce-scatter operation implementation for sum reduction.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Tensor rank (number of dimensions) of the inputs and outputs.
            target: Target device string for tracing.
            _trace_name: Trace name for profiling.
            axis: Axis along which to scatter the reduced result
                (defaults to 0).
            group_size: Number of devices per reduce-scatter group; must be
                at least 1 and must evenly divide the total number of
                devices (defaults to 0).

        Args:
            outputs: Output tensors (one per GPU) to store scattered reduced results.
            inputs: Input tensors (one per GPU) containing values to reduce.
            signal_buffers: Preallocated synchronization buffers for cross-GPU coordination.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Maximum of 8 GPUs supported (matches MAX_GPUS in comm/sync.mojo)
            - Tensor element count must be multiple of SIMD width
            - Requires identical tensor shapes within each reduce-scatter group
        """
        comptime num_devices = inputs.size
        comptime assert (
            signal_buffers.size == num_devices
        ), "expected 1 signal buffer per device"
        comptime assert group_size >= 1, "group_size must be at least 1"
        comptime assert (
            num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"

        # Reduce-scatter doesn't use scratch storage, so
        # only need enough signal_buffer space for Signal struct
        var scratch_buffer_size_bytes = 0
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        # Marshal input tensors into fully dynamic TileTensors so groups can
        # have different static shapes while sharing one Array type.

        @always_inline
        def launch_reducescatter[
            index: Int
        ]() raises {
            imm inputs,
            imm signal_buffers,
            imm dev_ctxs_input,
            imm outputs,
        }:
            comptime group_id, local_rank = divmod(index, group_size)
            comptime group_start = group_id * group_size
            # Full-world collectives keep scope 0; grouped collectives get a
            # distinct nonzero scope per device-group so their barrier counters
            # never poison the full-world bank on the shared Signal buffers.
            comptime domain_id = 0 if group_size == num_devices else group_size
            comptime InputTensorType = type_of(
                inputs[group_start].to_tile_tensor[.int64]().as_immut()
            )

            var in_tensors = Array[InputTensorType, group_size](
                uninitialized=True
            )
            var rank_sigs = Array[
                UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
            ](uninitialized=True)

            comptime for i in range(group_size):
                in_tensors[i] = rebind[InputTensorType](
                    inputs[group_start + i].to_tile_tensor[.int64]().as_immut()
                )

                rank_sigs[i] = (
                    signal_buffers[group_start + i]
                    ._ptr.bitcast[Signal]()
                    .as_unsafe_any_origin()
                )

            @always_inline
            @__parameter
            def output_lambda[
                output_index: Int,
                _dtype: DType,
                _width: SIMDLength,
                *,
                _alignment: Int,
            ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
                outputs[output_index]._lambda_store[
                    width=_width,
                    element_alignment=_alignment,
                ](
                    rebind[IndexList[rank]](coord_to_index_list(coords)),
                    rebind[SIMD[dtype, _width]](val),
                )

            var out_buf = outputs[index].to_tile_tensor[.int64]()
            reducescatter[
                ngpus=group_size,
                output_lambda=output_lambda[output_index=index, ...],
                axis=axis,
                domain_id=domain_id,
            ](
                in_tensors,
                out_buf.make_dynamic[.int64](),
                rank_sigs,
                dev_ctxs_input[index],
                local_rank=local_rank,
            )

        _launch_device_collective[num_devices](
            launch_reducescatter, dev_ctxs_input.copy()
        )


@extensibility.register("mo.distributed.allgather")
struct DistributedAllGather:
    """Registers the `mo.distributed.allgather` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
        group_size: Int = 0,
    ](
        outputs: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextArray,
    ) capturing raises:
        """Distributed allgather operation implementation.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Tensor rank (number of dimensions) of the inputs and outputs.
            target: Target device string for tracing.
            _trace_name: Trace name for profiling.
            group_size: Number of devices per allgather group; must be at
                least 1 and must evenly divide the total number of devices
                (defaults to 0).

        Args:
            outputs: Output tensors (one per GPU) to store gathered results.
            inputs: Input tensors (one per GPU) containing values to gather.
            signal_buffers: Device buffer values used for synchronization.
            dev_ctxs_input: Device contexts for participating GPUs.
        """
        comptime num_devices = inputs.size
        comptime assert (
            signal_buffers.size == num_devices
            and outputs.size == num_devices * group_size
        ), (
            "expected allgather inputs, signal buffers to have the same"
            " number of elements and outputs to have num_devices * group_size"
        )
        comptime assert group_size >= 1, "group_size must be at least 1"
        comptime assert (
            num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"

        var scratch_buffer_size_bytes = 0  # no allgather impl uses scratch
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        @always_inline
        def launch_allgather[
            index: Int
        ]() raises {
            imm inputs,
            imm outputs,
            imm signal_buffers,
            imm dev_ctxs_input,
        }:
            comptime group_id, local_rank = divmod(index, group_size)
            comptime group_start = group_id * group_size
            # Full-world collectives keep domain 0; grouped collectives get a
            # distinct nonzero domain per device-group so their barrier counters
            # never poison the full-world bank on the shared Signal buffers.
            comptime domain_id = 0 if group_size == num_devices else group_size
            comptime InputTensorType = type_of(
                TileTensor(
                    rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
                        inputs[group_start]._ptr
                    ),
                    row_major(inputs[group_start].size()),
                )
            )
            comptime OutputTensorType = type_of(
                TileTensor(
                    rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                        outputs[index * group_size]._ptr
                    ),
                    row_major(outputs[index * group_size].size()),
                )
            )
            var group_in_tensors = Array[InputTensorType, group_size](
                uninitialized=True
            )
            var device_out_tensors = Array[OutputTensorType, group_size](
                uninitialized=True
            )
            var group_rank_sigs = Array[
                UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
            ](uninitialized=True)

            comptime for src_idx in range(group_size):
                group_in_tensors[src_idx] = TileTensor(
                    rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
                        inputs[group_start + src_idx]._ptr
                    ),
                    row_major(inputs[group_start + src_idx].size()),
                )
                device_out_tensors[src_idx] = TileTensor(
                    rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                        outputs[index * group_size + src_idx]._ptr
                    ),
                    row_major(outputs[index * group_size + src_idx].size()),
                )
                group_rank_sigs[src_idx] = (
                    signal_buffers[group_start + src_idx]
                    ._ptr.bitcast[Signal]()
                    .as_unsafe_any_origin()
                )

            allgather[ngpus=group_size, domain_id=domain_id](
                group_in_tensors,
                device_out_tensors,
                group_rank_sigs,
                dev_ctxs_input[index],
                local_rank,
            )

        _launch_device_collective[num_devices](
            launch_allgather, dev_ctxs_input.copy()
        )


@extensibility.register("mo.distributed.broadcast")
struct DistributedBroadcast:
    """Distributed broadcast: copy tensor from root GPU to all GPUs.

    A single instance of this op handles all participating GPUs. It receives:
    - input: The source tensor from the root GPU (P2P accessible)
    - outputs: Destination tensors, one per GPU
    - signal_buffers: Synchronization buffers for all participating GPUs
    - dev_ctxs_input: Device contexts for all participating GPUs
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        root: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextArray,
    ) capturing raises:
        """Execute distributed broadcast operation.

        Parameters:
            dtype: Data type of the tensor.
            rank: Tensor rank (number of dimensions).
            root: Index of the root GPU (source of data).
            target: Target device string for tracing.
            _trace_name: Trace name for profiling.

        Args:
            outputs: Output tensors (one per GPU) to store broadcast results.
            input: Input tensor from root GPU (P2P accessible from all GPUs).
            signal_buffers: Synchronization buffers for cross-GPU coordination.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Maximum of 8 GPUs supported (MAX_GPUS).
            - Requires P2P access between GPUs (NVLink or PCIe P2P).
        """
        comptime num_devices = outputs.size
        comptime assert (
            signal_buffers.size == num_devices
        ), "expected 1 signal buffer per device"
        comptime assert (
            root >= 0 and root < num_devices
        ), "root GPU index must be in range [0, ngpus)"

        # 2-stage broadcast stages 1/ngpus of input into each signal buffer payload.
        # 1-stage broadcast doesn't use payload at all (direct P2P from root).
        # Use 2-stage requirement as upper bound.
        var scratch_buffer_size_bytes = _partitioned_scratch_requirement[
            num_devices, dtype
        ](input.size())
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        var in_buf = input.to_tile_tensor[.int64]()

        var rank_sigs = Array[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )

        comptime for i in range(signal_buffers.size):
            rank_sigs[i] = (
                signal_buffers[i]._ptr.bitcast[Signal]().as_unsafe_any_origin()
            )

        @always_inline
        def launch_broadcast[
            index: Int
        ]() raises {
            imm in_buf,
            imm rank_sigs,
            imm dev_ctxs_input,
            imm outputs,
        }:
            var out_buf = TileTensor[mut=True](
                outputs[index]
                .to_tile_tensor[.int64]()
                .make_dynamic[.int64]()
                ._storage,
                in_buf.layout,
            )
            broadcast[num_devices](
                in_buf,
                out_buf,
                rank_sigs,
                dev_ctxs_input[index],
                root,
                rank=index,
            )

        _launch_device_collective[num_devices](
            launch_broadcast, dev_ctxs_input.copy()
        )


@extensibility.register("mo.distributed.scatter")
struct DistributedScatter:
    """Distributed scatter: send different chunks to different device groups.

    Each DP replica group receives a different input chunk from the root GPU.
    All TP devices within the same replica get the same chunk via P2P pull.

    This op receives ngpus input tensors (one per GPU, padded from dp_size
    distinct chunks) plus ngpus signal buffers for synchronization. All GPUs
    see all chunks so they compute the same grid size (avoiding barrier
    deadlocks).
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        root: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: FusedOutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextArray,
    ) capturing raises:
        comptime ngpus = signal_buffers.size
        comptime assert (
            root >= 0 and root < ngpus
        ), "root GPU index must be in range [0, ngpus)"
        comptime assert inputs.size == ngpus, (
            "expected scatter inputs and signal buffers to have"
            " the same number of elements"
        )

        # Scatter uses signal buffers for barriers only (no payload staging),
        # so payload_size=0. This still validates the buffer holds a Signal.
        var scratch_buffer_size_bytes = 0
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        # Inputs can have different static shapes, so use make_dynamic to
        # produce a homogeneous fully-dynamic TileTensor type for Array.
        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[.int64]().make_dynamic[.int64]().as_immut()
        )
        var in_tensors = Array[InputTensorType, ngpus](uninitialized=True)
        var rank_sigs = Array[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )

        comptime for i in range(ngpus):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i]
                .to_tile_tensor[.int64]()
                .make_dynamic[.int64]()
                .as_immut()
            )
            rank_sigs[i] = (
                signal_buffers[i]._ptr.bitcast[Signal]().as_unsafe_any_origin()
            )

        @always_inline
        def launch_scatter[
            index: Int
        ]() raises {
            imm in_tensors,
            imm rank_sigs,
            imm dev_ctxs_input,
            imm outputs,
        }:
            var out_buf = outputs[index].to_tile_tensor[.int64]()
            scatter[ngpus=ngpus, dp_size=ngpus](
                in_tensors,
                out_buf,
                rank_sigs,
                dev_ctxs_input[index],
            )

        _launch_device_collective[ngpus](launch_scatter, dev_ctxs_input.copy())


@extensibility.register(
    "mo.composite.distributed.allreduce_add_rms_norm_quant_fp8"
)
struct DistributedAllReduceAddRMSNormQuantFP8:
    """Registers the `mo.composite.distributed.allreduce_add_rms_norm_quant_fp8` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        output_type: DType,
        scales_type: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: OutputVariadicTensors[dtype=output_type, rank=rank, ...],
        outputs_scales: OutputVariadicTensors[
            dtype=scales_type, rank=rank, ...
        ],
        outputs_residual: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        residuals: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        gammas: InputVariadicTensors[dtype=dtype, rank=1, ...],
        epsilons: InputVariadicTensors[dtype=DType.float32, ...],
        weight_offsets: InputVariadicTensors[dtype=dtype, ...],
        scales_ub: InputVariadicTensors[dtype=DType.float32, ...],
        dev_ctxs_input: DeviceContextArray,
    ) capturing raises:
        comptime num_devices = inputs.size
        comptime assert signal_buffers.size == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )

        # Logic copied from kernel host code
        # Note: this is a prime candidate for a method on a kernel
        # struct which advertises kernel info to the GC!
        var in_num_elems = inputs[0].size()
        comptime last_dim_idx = type_of(inputs[0]).rank - 1
        var cols = inputs[0].dim_size[last_dim_idx]()
        var rows = in_num_elems // cols
        var rows_per_rank = ceildiv(rows, num_devices)

        # Output scratch holds fp8 (1 byte) when quantizing; this op is
        # FP8-only, but size by output_type so the math stays correct if the
        # output ever matches the input dtype (no-quant path).
        var output_size_bytes = cols * rows_per_rank * size_of[output_type]()
        var pessimistic_simd_width = 32  # just to be safe...
        var scales_size_bytes = (
            align_up(
                rows_per_rank * size_of[scales_type](), pessimistic_simd_width
            ) if output_type
            != dtype else 0
        )
        var residual_size_bytes = cols * rows_per_rank * size_of[dtype]()

        var scratch_buffer_size_bytes = (
            output_size_bytes + scales_size_bytes + residual_size_bytes
        )
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        # Filter the dev_ctxs_list to have only the GPU devices.
        # The kernel also takes CPU operands, so CPU devices must be removed.
        var dev_ctxs = dev_ctxs_input.filter_gpu_contexts[num_devices]()

        # Marshal input tensors into TileTensors.
        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[.int64]().as_immut()
        )
        var in_tensors = Array[InputTensorType, inputs.size](uninitialized=True)

        # Marshal signal buffers.
        var rank_sigs = Array[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )

        comptime for i in range(inputs.size):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[.int64]().as_immut()
            )
            rank_sigs[i] = (
                signal_buffers[i]._ptr.bitcast[Signal]().as_unsafe_any_origin()
            )

        @always_inline
        def launch_fused_allreduce[
            index: Int
        ]() raises {
            imm in_tensors,
            imm rank_sigs,
            imm dev_ctxs,
            imm gammas,
            imm epsilons,
            imm weight_offsets,
            imm scales_ub,
            imm outputs,
            imm outputs_scales,
            imm outputs_residual,
            imm residuals,
        }:
            # Marshal per-device outputs and residual as TileTensors.
            var out_buf = outputs[index].to_tile_tensor[.int64]()
            var out_scales_buf = outputs_scales[index].to_tile_tensor[
                DType.int64
            ]()
            var out_residual_buf = outputs_residual[index].to_tile_tensor[
                DType.int64
            ]()
            var residual_buf = (
                residuals[index].to_tile_tensor[.int64]().as_immut()
            )
            var gamma_tensor = gammas[index].to_tile_tensor[.int64]()

            # TODO: Add a new struct like `VariadicInputScalar`` to
            # represent instead of manually loading the values in the
            # kernel code.
            var epsilon = epsilons[index].unsafe_ptr()[]
            var weight_offset = weight_offsets[index].unsafe_ptr()[]
            var scale_ub = scales_ub[index].unsafe_ptr()[]

            allreduce_residual_rmsnorm(
                in_tensors,
                residual_buf,
                out_buf,
                out_residual_buf,
                gamma_tensor,
                epsilon,
                weight_offset,
                scale_ub,
                out_scales_buf,
                rank_sigs,
                dev_ctxs[index],
            )

        _launch_device_collective[num_devices](
            launch_fused_allreduce, dev_ctxs.copy()
        )


@extensibility.register("mo.composite.distributed.reduce_scatter_rms_norm")
struct DistributedReduceScatterRMSNorm:
    """Registers the `mo.composite.distributed.reduce_scatter_rms_norm` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
        group_size: Int = 0,
        has_residual: Bool = False,
    ](
        outputs_normed: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        outputs_sum: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        gammas: InputVariadicTensors[dtype=dtype, rank=1, ...],
        epsilons: InputVariadicTensors[dtype=DType.float32, ...],
        weight_offsets: InputVariadicTensors[dtype=dtype, ...],
        residuals: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        dev_ctxs_input: DeviceContextArray,
    ) capturing raises:
        """Fused reduce-scatter sum + RMSNorm + residual add (bf16, no quant).

        Reduce-scatters `inputs` (one `[rows, cols]` tensor per device) along
        rows, adds `residuals`, and RMSNorm-normalizes each owned shard in the
        same launch, writing the normed shard to `outputs_normed` and the summed
        shard (the residual stream) to `outputs_sum`.

        Under `has_residual`, `residuals` carries the TP-replicated residual
        stream. Each device adds only its own row shard of it, which is why
        such callers must NOT pre-add it on the group leader: the
        reduce-scatter sums across ranks, so a leader-side add lands once for
        the whole group and this per-rank add reproduces it exactly -- but only
        because the residual is bit-identical on every rank of the group.
        Folding it here deletes a full-width elementwise add that ran on the
        group leader alone.

        Without `has_residual` the operands are still passed (the op's variadic
        groups must match in size) but never read, and both arms are exactly
        the reduce-scatter + norm this op was before the fold existed.

        Parameters:
            dtype: Element type of the input/output tensors.
            rank: Tensor rank of the inputs and outputs.
            target: Target device string for tracing.
            _trace_name: Trace name for profiling.
            group_size: Number of contiguous devices per independent
                reduce-scatter group; must be at least 2 and must evenly divide
                the total number of devices. Equal to `num_devices` for a
                full-world collective. The builder always sets it; the `0`
                attribute default is not a usable value.
            has_residual: Fold `residuals` into the reduce-scatter sum. Off
                leaves both arms byte-for-byte the plain reduce-scatter + norm.

        Args:
            outputs_normed: Per-device normed output shards.
            outputs_sum: Per-device reduce-scatter sum shards (residual stream).
            inputs: Per-device input tensors to reduce and scatter.
            signal_buffers: Per-device synchronization buffers.
            gammas: Per-device RMSNorm gamma weights (in_dtype, length cols).
            epsilons: Per-device RMSNorm epsilon scalars (float32).
            weight_offsets: Per-device gamma offset scalars (in_dtype).
            residuals: Per-device residual stream, same shape as `inputs` and
                bit-identical across each group. Read only under
                `has_residual`.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Maximum of 8 GPUs supported (matches MAX_GPUS in comm/sync.mojo).
            - Requires P2P, and identical tensor shapes within each group.
        """
        comptime num_devices = inputs.size
        comptime assert signal_buffers.size == num_devices, (
            "expected reduce_scatter_rms_norm inputs and signal buffers to have"
            " the same number of elements"
        )
        # >= 2, unlike the plain op: the fused kernel asserts `ngpus >= 2` and
        # this composite has no 1-input folder, so a 1-device group would fail
        # deeper with an unrelated message.
        comptime assert group_size >= 2, "group_size must be at least 2"
        comptime assert (
            num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"

        # Like plain reduce-scatter, no scratch: only the Signal struct.
        _check_signal_buffer_size(signal_buffers[0].size(), 0)

        # The kernel also takes CPU operands (epsilon/weight_offset are rank-0
        # CPU scalars), so CPU devices must be removed.
        var dev_ctxs = dev_ctxs_input.filter_gpu_contexts[num_devices]()

        @always_inline
        def launch_fused_rs_norm[
            index: Int
        ]() raises {
            imm inputs,
            imm signal_buffers,
            imm dev_ctxs,
            imm gammas,
            imm epsilons,
            imm weight_offsets,
            imm residuals,
            imm outputs_normed,
            imm outputs_sum,
        }:
            comptime group_id, local_rank = divmod(index, group_size)
            comptime group_start = group_id * group_size
            # Full-world keeps domain 0; a grouped collective gets a nonzero
            # domain so its counters never poison the full-world bank on the
            # shared Signal buffers. Keying by WIDTH puts every grouped op of
            # that width in one bank -- sound only under the same-barrier-
            # sequence invariant in `sync.mojo`'s `NUM_BARRIER_DOMAINS`.
            comptime domain_id = 0 if group_size == num_devices else group_size

            # Marshal into fully dynamic TileTensors so groups can have
            # different static shapes while sharing one Array type.
            comptime InputTensorType = type_of(
                inputs[group_start].to_tile_tensor[.int64]().as_immut()
            )
            var in_tensors = Array[InputTensorType, group_size](
                uninitialized=True
            )
            var rank_sigs = Array[
                UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
            ](uninitialized=True)

            comptime for i in range(group_size):
                in_tensors[i] = rebind[InputTensorType](
                    inputs[group_start + i].to_tile_tensor[.int64]().as_immut()
                )
                rank_sigs[i] = (
                    signal_buffers[group_start + i]
                    ._ptr.bitcast[Signal]()
                    .as_unsafe_any_origin()
                )

            var normed_buf = outputs_normed[index].to_tile_tensor[.int64]()
            var sum_buf = outputs_sum[index].to_tile_tensor[.int64]()
            var gamma_tensor = gammas[index].to_tile_tensor[.int64]()
            var epsilon = epsilons[index].unsafe_ptr()[]
            var weight_offset = weight_offsets[index].unsafe_ptr()[]
            var residual_buf = rebind[InputTensorType](
                residuals[index].to_tile_tensor[.int64]().as_immut()
            )
            # Windowed from the INPUT, not `residual_buf`: `reducescatter` bins
            # its rows from `in_tensors[0]`, so the residual cannot redefine it.
            var res_cols = Int(in_tensors[0].dim[rank - 1]())
            var res_config = ReduceScatterConfig[dtype, group_size](
                axis_size=in_tensors[0].num_elements() // res_cols,
                unit_numel=res_cols,
                threads_per_gpu=0,
            )
            var res_row_start = res_config.rank_unit_start(local_rank)

            # Fold in the reduce-scatter's OWN epilogue, not a third launch:
            # the lambda is caller-supplied, so `reducescatter` is untouched.
            # Both tensors are contiguous row-major over the same `cols`, so the
            # global flat index is the local one plus a constant (no `Coord`).
            var res_flat_offset = res_row_start * res_cols

            @__copy_capture(residual_buf, sum_buf, res_flat_offset)
            @__parameter
            @always_inline
            def rs_residual_lambda[
                _dtype: DType, _width: SIMDLength, *, _alignment: Int
            ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
                var local_flat = Int(sum_buf.layout(coords))
                var res = residual_buf.raw_load[width=_width](
                    local_flat + res_flat_offset
                )
                # `val` arrives already rounded to `dtype` from `_load_reduce`,
                # so add in f32 and round once -- the fused kernel's fold.
                var summed = (val.cast[.float32]() + res.cast[.float32]()).cast[
                    dtype
                ]()
                sum_buf.raw_store[width=_width, alignment=_alignment](
                    sum_buf.layout(coords), summed
                )

            @__parameter
            @always_inline
            def two_launch() raises:
                comptime if has_residual:
                    reducescatter[
                        dtype=dtype,
                        ngpus=group_size,
                        axis=0,
                        output_lambda=rs_residual_lambda,
                        domain_id=domain_id,
                    ](
                        in_tensors,
                        sum_buf,
                        rank_sigs,
                        dev_ctxs[index],
                        local_rank=local_rank,
                    )
                else:
                    reducescatter[
                        dtype=dtype,
                        ngpus=group_size,
                        axis=0,
                        domain_id=domain_id,
                    ](
                        in_tensors,
                        sum_buf,
                        rank_sigs,
                        dev_ctxs[index],
                        local_rank=local_rank,
                    )

                # `@__copy_capture` is REQUIRED: embedded into the
                # `rms_norm_gpu` device kernel, a captured local `var`
                # (`sum_buf`/`normed_buf`) is not carried to the device without
                # it -> the closure reads a garbage host-stack pointer and
                # corrupts device memory.
                @__copy_capture(sum_buf)
                @__parameter
                @always_inline
                def norm_input_fn[
                    width: Int
                ](coords: Coord) -> SIMD[dtype, width]:
                    return sum_buf.raw_load[width=width](sum_buf.layout(coords))

                @__copy_capture(normed_buf)
                @__parameter
                @always_inline
                def norm_output_fn[
                    width: SIMDLength, alignment: Int
                ](coords: Coord, val: SIMD[dtype, width]) -> None:
                    normed_buf.raw_store[width=width, alignment=alignment](
                        normed_buf.layout(coords), val
                    )

                rms_norm_gpu[
                    rank,
                    norm_input_fn,
                    norm_output_fn,
                    multiply_before_cast=True,
                ](
                    sum_buf.layout.shape_coord(),
                    gamma_tensor,
                    epsilon,
                    weight_offset,
                    dev_ctxs[index],
                )

            # The unfused `reducescatter` this fusion replaced already used PDL,
            # so leaving it off here would regress the path.
            # `residual`'s disengaged default is a distinct type, so it cannot
            # be passed unconditionally -- hence the two call sites.
            comptime if has_residual:
                _dispatch_rs_norm[
                    two_launch=two_launch,
                    has_residual=True,
                    domain_id=domain_id,
                    pdl_level=PDLLevel.ON,
                ](
                    in_tensors,
                    normed_buf,
                    sum_buf,
                    gamma_tensor,
                    epsilon,
                    weight_offset,
                    rank_sigs,
                    dev_ctxs[index],
                    local_rank=local_rank,
                    residual=residual_buf,
                )
            else:
                _dispatch_rs_norm[
                    two_launch=two_launch,
                    has_residual=False,
                    domain_id=domain_id,
                    pdl_level=PDLLevel.ON,
                ](
                    in_tensors,
                    normed_buf,
                    sum_buf,
                    gamma_tensor,
                    epsilon,
                    weight_offset,
                    rank_sigs,
                    dev_ctxs[index],
                    local_rank=local_rank,
                )

        _launch_device_collective[num_devices](
            launch_fused_rs_norm, dev_ctxs.copy()
        )


@extensibility.register("mo.composite.distributed.allgather_rms_norm")
struct DistributedAllGatherRMSNorm:
    """Registers the `mo.composite.distributed.allgather_rms_norm` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
        group_size: Int = 0,
    ](
        outputs_normed: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        outputs_residual: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        gammas: InputVariadicTensors[dtype=dtype, rank=1, ...],
        epsilons: InputVariadicTensors[dtype=DType.float32, ...],
        weight_offsets: InputVariadicTensors[dtype=dtype, ...],
        dev_ctxs_input: DeviceContextArray,
    ) capturing raises:
        """Fused all-gather + RMSNorm (bf16 in/out, no quantization).

        All-gathers `inputs` (per-device `[shard_i, cols]` row-shards) into the
        full replicated `[rows, cols]` stream and RMSNorms every gathered row in
        one launch: normed to `outputs_normed`, gathered residual to
        `outputs_residual` (both full and replicated).

        Parameters:
            dtype: Element type of the input/output tensors.
            rank: Tensor rank of the inputs and outputs.
            target: Target device string for tracing.
            _trace_name: Trace name for profiling.
            group_size: Number of contiguous devices per independent all-gather
                group; must be at least 2 and must evenly divide the total
                number of devices. Equal to `num_devices` for a full-world
                collective. The builder always sets it; the `0` attribute
                default is not a usable value.

        Args:
            outputs_normed: Per-device normed output (the group's gathered
                `[rows, cols]`, replicated within the group).
            outputs_residual: Per-device gathered residual (same shape as
                `outputs_normed`; the residual stream).
            inputs: Per-device input row-shards to gather.
            signal_buffers: Per-device synchronization buffers.
            gammas: Per-device RMSNorm gamma weights (in_dtype, length cols).
            epsilons: Per-device RMSNorm epsilon scalars (float32).
            weight_offsets: Per-device gamma offset scalars (in_dtype).
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Maximum of 8 GPUs supported (matches MAX_GPUS in comm/sync.mojo).
            - Requires P2P; within a group, shard shapes may differ only in the
              gathered axis.
        """
        comptime num_devices = inputs.size
        comptime assert signal_buffers.size == num_devices, (
            "expected allgather_rms_norm inputs and signal buffers to have the"
            " same number of elements"
        )
        # >= 2, unlike the plain op: the fused kernel asserts `ngpus >= 2` and
        # this composite has no 1-input folder, so a 1-device group would fail
        # deeper with an unrelated message.
        comptime assert group_size >= 2, "group_size must be at least 2"
        comptime assert (
            num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"

        # Like plain all-gather, no scratch: only the Signal struct.
        _check_signal_buffer_size(signal_buffers[0].size(), 0)

        # epsilon/weight_offset are rank-0 CPU scalars, so drop CPU devices.
        var dev_ctxs = dev_ctxs_input.filter_gpu_contexts[num_devices]()

        @always_inline
        def launch_fused_ag_norm[
            index: Int
        ]() raises {
            imm inputs,
            imm signal_buffers,
            imm dev_ctxs,
            imm gammas,
            imm epsilons,
            imm weight_offsets,
            imm outputs_normed,
            imm outputs_residual,
        }:
            comptime group_id, local_rank = divmod(index, group_size)
            comptime group_start = group_id * group_size
            # Full-world keeps domain 0; a grouped collective gets a nonzero
            # domain so its counters never poison the full-world bank on the
            # shared Signal buffers. Keying by WIDTH puts every grouped op of
            # that width in one bank -- sound only under the same-barrier-
            # sequence invariant in `sync.mojo`'s `NUM_BARRIER_DOMAINS`.
            comptime domain_id = 0 if group_size == num_devices else group_size

            # Derived per group, so groups may carry different static shapes.
            # Within a group the `rebind` below rejects differing STATIC
            # extents -- hence the builder's same-shape-outside-the-gathered-
            # axis rule. Ragged gathered dims arrive symbolic and lower alike.
            comptime InputTensorType = type_of(
                inputs[group_start].to_tile_tensor[.int64]().as_immut()
            )
            var in_tensors = Array[InputTensorType, group_size](
                uninitialized=True
            )
            var rank_sigs = Array[
                UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
            ](uninitialized=True)

            comptime for i in range(group_size):
                in_tensors[i] = rebind[InputTensorType](
                    inputs[group_start + i].to_tile_tensor[.int64]().as_immut()
                )
                rank_sigs[i] = (
                    signal_buffers[group_start + i]
                    ._ptr.bitcast[Signal]()
                    .as_unsafe_any_origin()
                )

            var normed_buf = outputs_normed[index].to_tile_tensor[.int64]()
            var sum_buf = outputs_residual[index].to_tile_tensor[.int64]()
            var gamma_tensor = gammas[index].to_tile_tensor[.int64]()
            var epsilon = epsilons[index].unsafe_ptr()[]
            var weight_offset = weight_offsets[index].unsafe_ptr()[]

            # Two-launch fallback (above the fuse threshold, where the
            # fabric-saturated standalone gather wins): all-gather into `sum_buf`,
            # then `rms_norm_gpu` into `normed_buf`. `sum_buf` is the residual on
            # both branches. mbc=True.
            @__parameter
            @always_inline
            def two_launch() raises:
                # Gather each shard into its contiguous row-range of `sum_buf`
                # (natural concat order) so the norm runs over the whole tensor.
                var cols_rt = Int(sum_buf.dim[rank - 1]())
                var base = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                    sum_buf._storage
                )
                comptime OutViewType = type_of(
                    TileTensor(base, row_major(cols_rt, cols_rt))
                )
                var out_views = Array[OutViewType, group_size](
                    uninitialized=True
                )
                var row_off = 0
                comptime for i in range(group_size):
                    var len_i = Int(in_tensors[i].num_elements()) // cols_rt
                    out_views[i] = TileTensor(
                        base + row_off * cols_rt,
                        row_major(len_i, cols_rt),
                    )
                    row_off += len_i

                allgather[dtype=dtype, ngpus=group_size, domain_id=domain_id](
                    in_tensors,
                    out_views,
                    rank_sigs,
                    dev_ctxs[index],
                    local_rank,
                )

                # `@__copy_capture` REQUIRED: without it the local `var`
                # (`sum_buf`/`normed_buf`) reaches the `rms_norm_gpu` device
                # kernel as a garbage host-stack pointer and corrupts memory.
                @__copy_capture(sum_buf)
                @__parameter
                @always_inline
                def norm_input_fn[
                    width: Int
                ](coords: Coord) -> SIMD[dtype, width]:
                    return sum_buf.raw_load[width=width](sum_buf.layout(coords))

                @__copy_capture(normed_buf)
                @__parameter
                @always_inline
                def norm_output_fn[
                    width: SIMDLength, alignment: Int
                ](coords: Coord, val: SIMD[dtype, width]) -> None:
                    normed_buf.raw_store[width=width, alignment=alignment](
                        normed_buf.layout(coords), val
                    )

                rms_norm_gpu[
                    rank,
                    norm_input_fn,
                    norm_output_fn,
                    multiply_before_cast=True,
                ](
                    sum_buf.layout.shape_coord(),
                    gamma_tensor,
                    epsilon,
                    weight_offset,
                    dev_ctxs[index],
                )

            _dispatch_ag_norm[two_launch=two_launch, domain_id=domain_id](
                in_tensors,
                normed_buf,
                sum_buf,
                gamma_tensor,
                epsilon,
                weight_offset,
                rank_sigs,
                dev_ctxs[index],
                local_rank=local_rank,
            )

        _launch_device_collective[num_devices](
            launch_fused_ag_norm, dev_ctxs.copy()
        )


@extensibility.register(
    "mo.composite.distributed.allgather_rms_norm_quant_mxfp8"
)
struct DistributedAllGatherRMSNormQuantMXFP8:
    """Registers `mo.composite.distributed.allgather_rms_norm_quant_mxfp8`."""

    @staticmethod
    def execute[
        dtype: DType,
        quant_dtype: DType,
        scales_dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
        group_size: Int = 0,
    ](
        outputs_normed: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        outputs_quant: OutputVariadicTensors[dtype=quant_dtype, rank=rank, ...],
        outputs_scale: OutputVariadicTensors[
            dtype=scales_dtype, rank=rank, ...
        ],
        outputs_residual: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        gammas: InputVariadicTensors[dtype=dtype, rank=1, ...],
        epsilons: InputVariadicTensors[dtype=DType.float32, ...],
        weight_offsets: InputVariadicTensors[dtype=dtype, ...],
        dev_ctxs_input: DeviceContextArray,
    ) capturing raises:
        """Fused all-gather + RMSNorm that also emits an MXFP8 copy.

        `DistributedAllGatherRMSNorm` plus `outputs_quant`/`outputs_scale`. The
        quantize rides the collective's epilogue on the same bf16 that lands in
        `outputs_normed`, so it is byte-identical to a standalone quantize.

        Parameters:
            dtype: Element type of the bf16 input/normed/residual tensors.
            quant_dtype: Quantized element type (`float8_e4m3fn`).
            scales_dtype: Block-scale element type (`float8_e8m0fnu`).
            rank: Tensor rank of the inputs and outputs.
            target: Target device string for tracing.
            _trace_name: Trace name for profiling.
            group_size: Devices per independent all-gather group; see
                `DistributedAllGatherRMSNorm`.

        Args:
            outputs_normed: Per-device normed output (the group's gathered
                `[rows, cols]`, replicated within the group).
            outputs_quant: Per-device MXFP8 copy of `outputs_normed`.
            outputs_scale: Per-device E8M0 block scales, `[rows, cols / 32]`.
            outputs_residual: Per-device gathered residual.
            inputs: Per-device input row-shards to gather.
            signal_buffers: Per-device synchronization buffers.
            gammas: Per-device RMSNorm gamma weights.
            epsilons: Per-device RMSNorm epsilon scalars (float32).
            weight_offsets: Per-device gamma offset scalars.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Targets AMD (CDNA4): the fallback calls `quantize_mx_amd` and
              the scale layout is its rank-2 one -- what
              `block_scaled_matmul_amd` takes as `a_scales`, not the SM100 SF
              atom and not the preshuffled order the `_preb` kernel needs.
            - Maximum of 8 GPUs; requires P2P.
        """
        comptime num_devices = inputs.size
        comptime assert signal_buffers.size == num_devices, (
            "expected allgather_rms_norm_quant_mxfp8 inputs and signal buffers"
            " to have the same number of elements"
        )
        comptime assert group_size >= 2, "group_size must be at least 2"
        comptime assert (
            num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"
        comptime assert (
            quant_dtype == .float8_e4m3fn
        ), "MXFP8 quant output must be float8_e4m3fn"
        comptime assert (
            scales_dtype == .float8_e8m0fnu
        ), "MXFP8 block scales must be float8_e8m0fnu"

        _check_signal_buffer_size(signal_buffers[0].size(), 0)

        var dev_ctxs = dev_ctxs_input.filter_gpu_contexts[num_devices]()

        @always_inline
        def launch_fused_ag_norm_quant[
            index: Int
        ]() raises {
            imm inputs,
            imm signal_buffers,
            imm dev_ctxs,
            imm gammas,
            imm epsilons,
            imm weight_offsets,
            imm outputs_normed,
            imm outputs_quant,
            imm outputs_scale,
            imm outputs_residual,
        }:
            comptime group_id, local_rank = divmod(index, group_size)
            comptime group_start = group_id * group_size
            comptime domain_id = 0 if group_size == num_devices else group_size

            comptime InputTensorType = type_of(
                inputs[group_start].to_tile_tensor[.int64]().as_immut()
            )
            var in_tensors = Array[InputTensorType, group_size](
                uninitialized=True
            )
            var rank_sigs = Array[
                UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
            ](uninitialized=True)

            comptime for i in range(group_size):
                in_tensors[i] = rebind[InputTensorType](
                    inputs[group_start + i].to_tile_tensor[.int64]().as_immut()
                )
                rank_sigs[i] = (
                    signal_buffers[group_start + i]
                    ._ptr.bitcast[Signal]()
                    .as_unsafe_any_origin()
                )

            var normed_buf = outputs_normed[index].to_tile_tensor[.int64]()
            var quant_buf = outputs_quant[index].to_tile_tensor[.int64]()
            var scale_buf = outputs_scale[index].to_tile_tensor[.int64]()
            var sum_buf = outputs_residual[index].to_tile_tensor[.int64]()
            var gamma_tensor = gammas[index].to_tile_tensor[.int64]()
            var epsilon = epsilons[index].unsafe_ptr()[]
            var weight_offset = weight_offsets[index].unsafe_ptr()[]
            var cols_rt = Int(sum_buf.dim[rank - 1]())

            # Both write paths index the quant/scale destinations by computed
            # offset with no bound on the destination's own extents: the fused
            # epilogue's `raw_store` documents the caller as responsible, and
            # `quantize_mx_amd` derives its geometry from the INPUT. So an
            # undersized scale tensor writes into the next row -- self-consistent
            # garbage a byte-compare oracle cannot see -- and past the allocation
            # on the last one. Mirrors the `normed_out`/`sum_out` guards in
            # `comm/allgather_rmsnorm.mojo`, which exist because sizing an output
            # from the whole world instead of the TP group is the natural mistake
            # here; these two only reach the kernel through a closure.
            var rows_rt = Int(normed_buf.num_elements()) // cols_rt
            var quant_cols = Int(quant_buf.dim[rank - 1]())
            if (
                quant_cols != cols_rt
                or Int(quant_buf.num_elements()) != rows_rt * cols_rt
            ):
                raise Error(
                    String(
                        "allgather_rms_norm_quant_mxfp8: outputs_quant is ",
                        Int(quant_buf.num_elements()) // quant_cols,
                        " x ",
                        quant_cols,
                        ", expected ",
                        rows_rt,
                        " x ",
                        cols_rt,
                        " (as outputs_normed)",
                    )
                )
            var scale_cols = Int(scale_buf.dim[rank - 1]())
            if (
                scale_cols * MXFP8_SF_VECTOR_SIZE != cols_rt
                or Int(scale_buf.num_elements()) != rows_rt * scale_cols
            ):
                raise Error(
                    String(
                        "allgather_rms_norm_quant_mxfp8: outputs_scale is ",
                        Int(scale_buf.num_elements()) // scale_cols,
                        " x ",
                        scale_cols,
                        ", expected ",
                        rows_rt,
                        " x ",
                        cols_rt // MXFP8_SF_VECTOR_SIZE,
                        " (normed cols / ",
                        MXFP8_SF_VECTOR_SIZE,
                        ")",
                    )
                )

            # `@__copy_capture` REQUIRED: any local this closure reads but does
            # not list reaches the device as a host-stack pointer and faults.
            @__copy_capture(quant_buf, scale_buf, cols_rt)
            @__parameter
            @always_inline
            def mx_epilogue[
                width: Int
            ](row: Int, col: Int, val: SIMD[dtype, width]):
                var quantized: SIMD[quant_dtype, width]
                var e8m0: Scalar[scales_dtype]
                quantized, e8m0 = quantize_mxfp8_lane_group[
                    quant_dtype,
                    scales_dtype,
                    SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE,
                ](val)
                # Quantize on every lane -- the block max is a cross-lane
                # reduction. Only the in-range lanes store.
                if col < cols_rt:
                    quant_buf.raw_store[width=width](
                        quant_buf.layout(Coord(row, col)), quantized
                    )
                    if col % MXFP8_SF_VECTOR_SIZE == 0:
                        scale_buf.raw_store(
                            scale_buf.layout(
                                Coord(row, col // MXFP8_SF_VECTOR_SIZE)
                            ),
                            e8m0,
                        )

            # Above the fuse threshold. Owes the same outputs as the fused
            # path -- skipping the quantize leaves `outputs_quant` stale.
            @__parameter
            @always_inline
            def two_launch_with_quant() raises:
                var base = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                    sum_buf._storage
                )
                comptime OutViewType = type_of(
                    TileTensor(base, row_major(cols_rt, cols_rt))
                )
                var out_views = Array[OutViewType, group_size](
                    uninitialized=True
                )
                var row_off = 0
                comptime for i in range(group_size):
                    var len_i = Int(in_tensors[i].num_elements()) // cols_rt
                    out_views[i] = TileTensor(
                        base + row_off * cols_rt,
                        row_major(len_i, cols_rt),
                    )
                    row_off += len_i

                allgather[dtype=dtype, ngpus=group_size, domain_id=domain_id](
                    in_tensors,
                    out_views,
                    rank_sigs,
                    dev_ctxs[index],
                    local_rank,
                )

                @__copy_capture(sum_buf)
                @__parameter
                @always_inline
                def norm_input_fn[
                    width: Int
                ](coords: Coord) -> SIMD[dtype, width]:
                    return sum_buf.raw_load[width=width](sum_buf.layout(coords))

                @__copy_capture(normed_buf)
                @__parameter
                @always_inline
                def norm_output_fn[
                    width: SIMDLength, alignment: Int
                ](coords: Coord, val: SIMD[dtype, width]) -> None:
                    normed_buf.raw_store[width=width, alignment=alignment](
                        normed_buf.layout(coords), val
                    )

                rms_norm_gpu[
                    rank,
                    norm_input_fn,
                    norm_output_fn,
                    multiply_before_cast=True,
                ](
                    sum_buf.layout.shape_coord(),
                    gamma_tensor,
                    epsilon,
                    weight_offset,
                    dev_ctxs[index],
                )

                quantize_mx_amd(
                    dev_ctxs[index], quant_buf, scale_buf, normed_buf
                )

            _dispatch_ag_norm_quant[
                two_launch_with_quant=two_launch_with_quant,
                quant_epilogue=mx_epilogue,
                domain_id=domain_id,
            ](
                in_tensors,
                normed_buf,
                sum_buf,
                gamma_tensor,
                epsilon,
                weight_offset,
                rank_sigs,
                dev_ctxs[index],
                local_rank=local_rank,
            )

        _launch_device_collective[num_devices](
            launch_fused_ag_norm_quant, dev_ctxs.copy()
        )


@extensibility.register("mo.composite.distributed.matmul_reduce_scatter.sum")
struct DistributedMatmulReduceScatterSum:
    """Registers the `mo.composite.distributed.matmul_reduce_scatter.sum` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        a_type: DType,
        b_type: DType,
        c_type: DType,
        rank: Int,
        has_residual: Bool,
        residual_peer: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: OutputVariadicTensors[dtype=c_type, rank=rank, ...],
        inputs_a: InputVariadicTensors[dtype=a_type, rank=rank, ...],
        inputs_b: InputVariadicTensors[dtype=b_type, rank=rank, ...],
        residual: InputTensor[dtype=c_type, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextArray,
    ) capturing raises:
        comptime num_devices = inputs_a.size
        comptime assert (
            inputs_b.size == num_devices
        ), "expected same number of A and B inputs"
        comptime assert (
            signal_buffers.size == num_devices
        ), "expected 1 signal buffer per device"

        _check_signal_buffer_size(signal_buffers[0].size(), 0)

        # Marshal output tensors into TileTensors (one per peer GPU).
        # Each output[i] may have a different comptime static spec, so
        # rebind to a common type derived from output[0].
        comptime OutputTileType = type_of(outputs[0].to_tile_tensor[.int64]())
        var c_peer_tt = Array[OutputTileType, num_devices](uninitialized=True)
        comptime for i in range(num_devices):
            c_peer_tt[i] = rebind[OutputTileType](
                outputs[i].to_tile_tensor[.int64]()
            )

        # Marshal signal buffers.
        var rank_sigs = Array[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )
        comptime for i in range(num_devices):
            rank_sigs[i] = (
                signal_buffers[i]._ptr.bitcast[Signal]().as_unsafe_any_origin()
            )

        # Pinned MatmulConfig for the fused matmul+RS kernel.
        # The SM100 GEMM heuristic picks larger tiles (e.g.
        # (256,224,16)/cta_group=2) which work well for standalone matmul
        # but push the fused matmul+RS kernel over the register-pressure
        # cliff (~128 regs/thread).
        comptime matmul_config = MatmulConfig[a_type, b_type, c_type, True](
            mma_shape=Index(128, 128, 16),
            cluster_shape=Index(1, 1, 1),
            cta_group=1,
        )

        comptime if has_residual:
            if residual.dim_size(0) != inputs_a[0].dim_size(0):
                raise Error(
                    "matmul+RS residual.dim_size(0)="
                    + String(residual.dim_size(0))
                    + " must equal inputs_a[0].dim_size(0)="
                    + String(inputs_a[0].dim_size(0))
                )

        # Build the residual-add compute lambda. The residual lives on a
        # single peer (the device of the residual tensor in the graph).
        # Mirroring the asymmetric DeepseekV3/KimiK2.5 pattern, only that
        # peer applies the residual-add lambda; the other peers launch
        # without it, so after RS-sum the output contains
        # `sum_j(A_j @ B_j) + residual` rather than `... + ngpus*residual`.
        @__parameter
        @always_inline
        @__copy_capture(residual)
        def residual_add_fn[
            _dtype: DType, _width: SIMDLength, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[_dtype, _width]) capturing -> SIMD[
            _dtype, _width
        ]:
            return val + rebind[SIMD[_dtype, _width]](
                residual.load[width=_width, element_alignment=alignment](coords)
            )

        comptime compute_lambda = Optional[
            matmul_elementwise_compute_lambda_type
        ](residual_add_fn)

        # Marshal per-peer input TileTensors. All peers' A (and B) share
        # the same comptime spec; rebind to a common type so we can build
        # one Array per kind.
        comptime InputATileType = type_of(inputs_a[0].to_tile_tensor[.int64]())
        var a_per_peer = Array[InputATileType, num_devices](uninitialized=True)
        comptime for i in range(num_devices):
            a_per_peer[i] = rebind[InputATileType](
                inputs_a[i].to_tile_tensor[.int64]()
            )

        comptime InputBTileType = type_of(inputs_b[0].to_tile_tensor[.int64]())
        var b_per_peer = Array[InputBTileType, num_devices](uninitialized=True)
        comptime for i in range(num_devices):
            b_per_peer[i] = rebind[InputBTileType](
                inputs_b[i].to_tile_tensor[.int64]()
            )

        # Hand off to the dispatcher: it picks fused vs unfused based on
        # a comptime arch check and a runtime shape check on M, and
        # drives the per-peer parallel launch.
        matmul_reducescatter_dispatch[
            transpose_b=True,
            config=matmul_config,
            ngpus=num_devices,
            has_residual=has_residual,
            residual_peer=residual_peer,
            elementwise_compute_lambda_fn=compute_lambda,
        ](c_peer_tt, a_per_peer, b_per_peer, rank_sigs, dev_ctxs_input)


@extensibility.register("lamport_allreduce_rmsnorm")
struct LamportAllreduceRMSNorm:
    """Per-rank fused Lamport allreduce + RMSNorm (high-perf protocol).

    Built on `comm.allreduce_lamport_rmsnorm`; the Lamport comm region is
    embedded in `Signal` (`Signal.lamport_region`), so the caller sizes/
    initializes the signal buffers as `sizeof(Signal)` bytes. `ngpus` is
    inferred from the number of signal buffers passed in (must be in [2, 8]).

    Shape constraints (also checked inside `lamport_allreduce_rmsnorm`, but
    surfaced here for op-level diagnostics):
    - `cols % atomic_width == 0`  (whole 128-bit Lamport packs only;
      `atomic_width = 16 / size_of[dtype]`, so bf16 needs `cols % 8 == 0`).
    - `cols / atomic_width <= BLOCK_SIZE`  (one pack per thread / row;
      `BLOCK_SIZE = floor(max_tpb / WARP_SIZE) * WARP_SIZE`, e.g. 1024 on
      Hopper/Blackwell, capping bf16 hidden at 8192).
    """

    @staticmethod
    def execute[
        target: StaticString,
        my_rank: Int,
        pdl: Bool = True,
        early_launch: Bool = True,
    ](
        output: OutputTensor[rank=2, ...],
        act: InputTensor[dtype=output.dtype, rank=2, ...],
        gamma: InputTensor[dtype=output.dtype, rank=1, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        ctx: DeviceContext,
    ) raises:
        comptime assert target == "gpu", "lamport_allreduce_rmsnorm: gpu only"
        comptime ngpus = signal_buffers.size
        comptime assert (
            ngpus >= 2 and ngpus <= MAX_GPUS
        ), "lamport_allreduce_rmsnorm: signal_buffers.size must be in [2, 8]"
        comptime assert (
            my_rank >= 0 and my_rank < ngpus
        ), "lamport_allreduce_rmsnorm: my_rank must be in [0, ngpus)"
        comptime epsilon = Float32(1e-6)
        comptime dtype = output.dtype

        var rank_sigs = Array[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )
        comptime for i in range(ngpus):
            rank_sigs[i] = (
                signal_buffers[i]._ptr.bitcast[Signal]().as_unsafe_any_origin()
            )

        var rows = act.dim_size[0]()
        var cols = act.dim_size[1]()

        # Surface kernel shape constraints at the op boundary so the failure is
        # attributed to the op rather than to the kernel-host launcher. Mirrors
        # the checks in `comm.allreduce_lamport_rmsnorm`.
        comptime atomic_width = Lamport.ATOMIC_BYTES // size_of[dtype]()
        comptime max_tpb = ctx.default_device_info.max_thread_block_size
        comptime BLOCK_SIZE = align_down(max_tpb, WARP_SIZE)
        if cols % atomic_width != 0:
            raise Error(
                "lamport_allreduce_rmsnorm: cols (",
                cols,
                ") must be a multiple of atomic_width (",
                atomic_width,
                ") -- whole 128-bit Lamport packs required",
            )
        if cols // atomic_width > BLOCK_SIZE:
            raise Error(
                "lamport_allreduce_rmsnorm: cols/atomic_width (",
                cols // atomic_width,
                ") exceeds BLOCK_SIZE (",
                BLOCK_SIZE,
                ") -- one pack per thread required",
            )

        var src = rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](act._ptr)
        var dst = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            output._ptr
        )
        var gm = rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
            gamma._ptr
        )

        lamport_allreduce_rmsnorm[
            dtype, ngpus, pdl=pdl, early_launch=early_launch
        ](
            my_rank,
            src,
            dst,
            gm,
            rank_sigs,
            rows,
            cols,
            epsilon.cast[dtype](),
            ctx,
        )
