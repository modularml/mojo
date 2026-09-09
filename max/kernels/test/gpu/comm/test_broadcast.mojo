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

from std.math import ceildiv
from std.sys import size_of
from std.itertools import product
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import (
    Idx,
    TileTensor,
    row_major,
)
from std.testing import assert_true

from comm import Signal, MAX_GPUS
from comm.broadcast import broadcast
from comm.sync import enable_p2p, init_signal_buffer


@always_inline
@__parameter
def _input_value[dtype: DType](root: Int, j: Int) -> Scalar[dtype]:
    """Generate position-based input value that includes root rank.

    Each element has a unique value based on position, and includes the root
    rank to verify the correct source GPU was used.
    """
    # 251 is the largest prime < 256; using a prime avoids power-of-two aliasing.
    return Scalar[dtype](root + 1) + Scalar[dtype](j % 251)


# Shared test configurations - kept small to avoid CI timeouts on MI355
comptime test_lengths = (
    0,  # No elements
    1,  # Single element
    2,  # Smaller than typical simd_width (e.g., float32 simd_width=4 or 8)
    5,  # simd_width + 1 for float32 (simd_width=4)
    7,  # Not a multiple of simd_width
    9,  # simd_width + 1 for bfloat16 (simd_width=8)
    100,  # Not a multiple of typical simd_width
    1023,  # Not a multiple of simd_width
    8 * 1024,  # Small latency bound
    8 * 1024 + 3,  # Not a multiple of simd_width
    128 * 1024,  # Larger latency bound
    256 * 1024,  # Smallest bandwidth bound
    # For 8 GPUs this stays on 1-stage (< 2 MiB) and exercises 1-stage tail handling.
    256 * 1024 + 3,
    16 * 1024 * 1024,  # Bandwidth bound
    16 * 1024 * 1024 + 3,  # Large non-aligned: tests 2-stage tail handling
    64 * 1024 * 1024,  # Bandwidth bound: 8192 chunk size at dim = 8192
)

# Test hyperparameters.
# uint8 is what the byte-oriented callers (distributed_ops, block_offload_ops)
# instantiate.
comptime test_dtypes = (DType.bfloat16, DType.float32, DType.uint8)
comptime test_gpu_counts = (2, 4, 8)


def _get_test_str[
    dtype: DType,
    in_place: Bool,
](ngpus: Int, length: Int, root: Int) -> String:
    return String(
        "====broadcast-",
        dtype,
        "-ngpus-",
        ngpus,
        "-gpu-root",
        root,
        "-inplace-",
        in_place,
        "-nelems-",
        length,
    )


def broadcast_test[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    root_self_copy: Bool,
](list_of_ctxs: List[DeviceContext], length: Int, root: Int) raises:
    var root_ctx = list_of_ctxs[root]

    # Create input buffer on root GPU
    var host_input_ptr = root_ctx.enqueue_create_host_buffer[dtype](length)
    for j in range(length):
        host_input_ptr[j] = _input_value[dtype](root, j)
    var input_dev = root_ctx.enqueue_create_buffer[dtype](length)
    root_ctx.enqueue_copy(input_dev, host_input_ptr)

    # Create output buffers for all GPUs
    var out_dev_list = List[DeviceBuffer[dtype]](capacity=ngpus)

    # Create TileTensor types for input and output
    var in_tile = TileTensor(input_dev, row_major(length)).as_immut()
    comptime OutputTileType = TileTensor[
        dtype, type_of(row_major(length)), MutAnyOrigin
    ]
    var out_tiles = Array[OutputTileType, ngpus](uninitialized=True)

    # Create signal buffers for synchronization
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    for i in range(ngpus):
        # Create this rank's output buffer
        if root_self_copy and i == root:
            # Special case: root does an in-place copy
            out_dev_list.append(input_dev)
            out_tiles[i] = OutputTileType(input_dev, row_major(length))
            continue

        var ctx = list_of_ctxs[i]
        var out_ptr = ctx.enqueue_create_buffer[dtype](length)
        out_dev_list.append(out_ptr)
        out_tiles[i] = OutputTileType(
            out_ptr.unsafe_ptr().as_unsafe_any_origin(), row_major(length)
        )

    # Signal buffers need payload space for 2-stage broadcast
    var num_bytes = length * size_of[dtype]()
    var chunk_bytes = ceildiv(num_bytes, ngpus)
    var signal_buf_size = size_of[Signal]() + chunk_bytes

    for i in range(ngpus):
        # Create and initialize signal buffers (with payload space for 2-stage)
        signal_buffers.append(
            list_of_ctxs[i].create_buffer_sync[.uint8](signal_buf_size)
        )
        init_signal_buffer(signal_buffers[i], list_of_ctxs[i])
        rank_sigs[i] = (
            signal_buffers[i]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

    for i in range(ngpus):
        list_of_ctxs[i].synchronize()

    # Zero output buffers before broadcast to ensure we're verifying fresh results
    for i in range(ngpus):
        if root_self_copy and i == root:
            continue  # Don't zero input buffer for in-place case
        list_of_ctxs[i].enqueue_memset(out_dev_list[i], 0)
        list_of_ctxs[i].synchronize()

    # Launch broadcast per device
    comptime for i in range(ngpus):
        broadcast[ngpus](
            in_tile, out_tiles[i], rank_sigs, list_of_ctxs[i], root, rank=i
        )

    # Synchronize all GPUs
    for i in range(ngpus):
        list_of_ctxs[i].synchronize()

    # Copy results back to host and verify
    var host_output = List(length=length, fill=Scalar[dtype](0))
    for i in range(ngpus):
        list_of_ctxs[i].enqueue_copy(host_output, out_dev_list[i])
        list_of_ctxs[i].synchronize()

        # Verify all elements match expected value
        for j in range(length):
            var expected = _input_value[dtype](root, j)
            if host_output[j] != expected:
                raise Error(
                    "Verification failed at GPU",
                    i,
                    "index",
                    j,
                    "value:",
                    host_output[j],
                    "expected:",
                    expected,
                )


@__parameter
def run_broadcast_sweep[]() raises:
    # Run tests for each configuration.
    comptime for gpu_idx, dtype_idx, length_idx, root_self_copy in product(
        range(len(test_gpu_counts)),
        range(len(test_dtypes)),
        range(len(test_lengths)),
        [True, False],
    ):
        comptime num_gpus = rebind[Int](test_gpu_counts[gpu_idx])
        if DeviceContext.number_of_devices() < num_gpus:
            continue

        # Create GPU context.
        var list_of_ctxs = List[DeviceContext]()
        for i in range(num_gpus):
            list_of_ctxs.append(DeviceContext(device_id=i))

        comptime dtype = rebind[DType](test_dtypes[dtype_idx])
        comptime length = rebind[Int](test_lengths[length_idx])

        # Test with each GPU as root
        for root in range(num_gpus):
            print(_get_test_str[dtype, root_self_copy](num_gpus, length, root))
            try:
                broadcast_test[
                    dtype,
                    1,
                    num_gpus,
                    root_self_copy,
                ](list_of_ctxs, length, root)
            except e:
                print("Exception in broadcast_test execution:")
                raise e^


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")
    run_broadcast_sweep[]()
