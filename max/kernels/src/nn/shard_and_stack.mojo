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
"""Implements shard-and-stack: shards a tensor across devices and stacks the shards into a higher-rank output."""

from max.algorithm import parallelize, sync_parallelize
from std.collections import Array
from max.gpu.host import DeviceBuffer, DeviceContext, DeviceContextArray
from std.memory import unsafe_memcpy
from extensibility import InputVariadicTensors, OutputVariadicTensors
from std.utils import product


def _validate_shard_and_stack[
    axis: Int,
](
    outputs: OutputVariadicTensors,
    inputs: InputVariadicTensors[
        dtype=outputs.dtype,
        rank=outputs.rank - 1,
        ...,
    ],
) raises:
    """Validate inputs and outputs for shard_and_stack operation.

    Parameters:
        axis: The dimension along which to shard the weights.

    Args:
        outputs: Output tensors, one per device/shard.
        inputs: Input tensors to be sharded, all with identical shapes.
    """
    comptime assert inputs.size > 0, "must have one or more inputs"
    comptime assert 0 <= axis < inputs.rank, "axis must be in [0, inputs.rank)"

    var input_shape = inputs[0].shape()
    var row_major_strides = input_shape.get_row_major_strides()

    # Validate that all inputs must have the same shape and row-major strides
    for i in range(inputs.size):
        if input_shape != inputs._tensors[i].shape():
            raise Error("all input shapes must match")

        if row_major_strides != inputs._tensors[i].strides():
            raise Error("all inputs must have a row-major layout")

    # Validate that input dimension along split axis is divisible by the number
    # of outputs.
    var input_axis_dim = inputs[0].dim_size(axis)
    if input_axis_dim % outputs.size != 0:
        raise Error(
            "input dimension along split axis must be divisible by outputs.size"
        )

    # Validate all outputs have the same expected shape
    for out_idx in range(outputs.size):
        var output_shape = outputs._tensors[out_idx].shape()

        if output_shape[0] != inputs.size:
            raise Error(
                "dimension zero of each output must match number of inputs"
            )

        # Validate that non-axis dimensions match between inputs and outputs
        # Note: output has extra dimension 0, so input dim i maps to output dim i+1
        for i in range(inputs.rank):
            if i != axis:
                if inputs[0].dim_size(i) != outputs._tensors[out_idx].dim_size(
                    i + 1
                ):
                    raise Error(
                        "non-axis dimensions must match between inputs and"
                        " outputs"
                    )

        # Validate that the axis dimension is correctly divided:
        # Output has extra dimension 0, so input axis i → output axis i+1
        comptime output_axis = axis + 1
        if inputs[0].dim_size(axis) // outputs.size != outputs._tensors[
            out_idx
        ].dim_size(output_axis):
            raise Error(
                "axis dimension mismatch: expected input_axis_dim //"
                " outputs.size"
            )


def _shard_and_stack_multi_device[
    axis: Int,
](
    outputs: OutputVariadicTensors,
    inputs: InputVariadicTensors[
        dtype=outputs.dtype,
        rank=outputs.rank - 1,
        ...,
    ],
    dev_ctxs_input: DeviceContextArray,
) raises:
    """Multi-device implementation using H2D transfers.

    Device contexts are extracted from operands/results in order of first
    appearance (deduplicated). For multi-device:
      dev_ctxs_input[0] = CPU (from inputs)
      dev_ctxs_input[1..] = GPUs (from outputs)

    Parameters:
        axis: The dimension along which to shard the weights.

    Args:
        outputs: Output tensors, one per device/shard.
        inputs: Input tensors to be sharded, all with identical shapes.
        dev_ctxs_input: Device contexts for H2D transfers.
    """
    var dyn_inputs = inputs._tensors
    var dyn_outputs = outputs._tensors

    var input_axis = inputs[0].dim_size[axis]()
    var chunk_size = input_axis // outputs.size

    # Calculate dimensional products
    var outer_dims = product(inputs[0].shape(), 0, axis)
    var inner_dims = product(inputs[0].shape(), axis + 1, inputs.rank)

    # Elements in one chunk segment (contiguous in memory)
    var segment_elements = chunk_size * inner_dims

    # Total elements per input in output
    var output_elements_per_input = outer_dims * segment_elements

    @no_inline
    def transfer(tp_index: Int) raises {imm}:
        # Device context for this output (index 0 is CPU, so +1)
        var gpu_ctx = dev_ctxs_input[tp_index + 1]
        var output_tensor = dyn_outputs[tp_index]

        # Multi-device mode: use H2D transfers for each output.
        # Each output goes to a different GPU, using the corresponding
        # device context.
        # Note: We process outputs sequentially since each output may be
        # on a different device and H2D copies are async per device.
        for input_idx in range(inputs.size):
            var input_tensor = dyn_inputs[input_idx]

            for outer_idx in range(outer_dims):
                # Calculate source offset in input tensor
                var src_offset = (
                    outer_idx * input_axis * inner_dims
                    + tp_index * segment_elements
                )

                # Calculate dest offset in output tensor
                var dst_offset = (
                    input_idx * output_elements_per_input
                    + outer_idx * segment_elements
                )

                # H2D copy: source is host pointer, dest is device buffer
                var src_ptr = input_tensor.unsafe_ptr() + src_offset
                var dst_ptr = output_tensor.unsafe_ptr() + dst_offset

                gpu_ctx.enqueue_copy(
                    DeviceBuffer(
                        gpu_ctx,
                        dst_ptr,
                        segment_elements,
                        owning=False,
                    ),
                    src_ptr,
                )

    # Enqueue transfers in parallel, one thread per device.
    sync_parallelize(transfer, outputs.size)


def _shard_and_stack_single_device[
    axis: Int,
](
    outputs: OutputVariadicTensors,
    inputs: InputVariadicTensors[
        dtype=outputs.dtype,
        rank=outputs.rank - 1,
        ...,
    ],
) raises:
    """Single-device implementation using CPU unsafe_memcpy.

    Parameters:
        axis: The dimension along which to shard the weights.

    Args:
        outputs: Output tensors, one per device/shard.
        inputs: Input tensors to be sharded, all with identical shapes.
    """
    var dyn_inputs = inputs._tensors
    var dyn_outputs = outputs._tensors

    var input_axis = inputs[0].dim_size[axis]()
    var chunk_size = input_axis // outputs.size

    # Calculate dimensional products
    var outer_dims = product(inputs[0].shape(), 0, axis)
    var inner_dims = product(inputs[0].shape(), axis + 1, inputs.rank)

    # Elements in one chunk segment (contiguous in memory)
    var segment_elements = chunk_size * inner_dims

    # Total elements per input in output
    var output_elements_per_input = outer_dims * segment_elements

    @no_inline
    def process_task(input_idx: Int) {imm}:
        var input_tensor = dyn_inputs[input_idx]

        for tp_index in range(outputs.size):
            var output_tensor = dyn_outputs[tp_index]

            for outer_idx in range(outer_dims):
                # Calculate source offset in input tensor
                # Layout: [outer_dims][full_axis_dim][inner_dims]
                var src_offset = (
                    outer_idx
                    * input_axis
                    * inner_dims  # Skip to outer position
                    + tp_index * segment_elements  # Skip to tp_index chunk
                )

                # Calculate dest offset in output tensor
                # Layout: [num_inputs][outer_dims][chunk_size][inner_dims]
                var dst_offset = (
                    input_idx
                    * output_elements_per_input  # Skip to input position
                    + outer_idx * segment_elements  # Skip to outer position
                )

                # Copy contiguous segment
                var src_ptr = input_tensor.unsafe_ptr() + src_offset
                var dst_ptr = output_tensor.unsafe_ptr() + dst_offset

                unsafe_memcpy(dest=dst_ptr, src=src_ptr, count=segment_elements)

    parallelize(process_task, inputs.size)


def shard_and_stack[
    axis: Int,
](
    outputs: OutputVariadicTensors,
    inputs: InputVariadicTensors[
        dtype=outputs.dtype,
        rank=outputs.rank - 1,
        ...,
    ],
    dev_ctxs_input: DeviceContextArray,
) raises:
    """Shard weight tensors across multiple devices for tensor parallelism.

    This operation takes multiple input tensors with identical shapes and shards
    them along a specified axis, distributing the shards to different devices
    (typically GPUs for tensor parallel inference).

    Parameters:
        axis: The dimension along which to shard the weights.

    Args:
        outputs: Output tensors, one per device/shard.
        inputs: Input tensors to be sharded, all with identical shapes.
        dev_ctxs_input: Device contexts for multi-device transfers.
    """
    _validate_shard_and_stack[axis](outputs, inputs)

    # Check if outputs are on different devices than inputs (multi-device mode).
    comptime is_multi_device = dev_ctxs_input.length > 1

    comptime if is_multi_device:
        _shard_and_stack_multi_device[axis](outputs, inputs, dev_ctxs_input)
    else:
        _shard_and_stack_single_device[axis](outputs, inputs)
