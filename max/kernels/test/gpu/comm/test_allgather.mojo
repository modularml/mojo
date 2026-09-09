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


from std.sys import size_of, has_amd_gpu_accelerator

from comm.allgather import allgather
from comm import MAX_GPUS, Signal
from comm.sync import enable_p2p, init_signal_buffer
import comm.vendor.ccl as vendor_ccl
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from layout import (
    Idx,
    TileTensor,
    row_major,
)
from std.testing import assert_equal, assert_true


def all_gather_test[
    dtype: DType, ngpus: Int
](list_of_ctx: List[DeviceContext], lengths: List[Int]) raises -> None:
    """Test allgather with new variadic output semantics.

    Each device should receive individual copies of all inputs,
    not a single concatenated buffer.
    """

    # Create device buffers for all GPUs.
    var in_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_bufs_list = List[List[DeviceBuffer[dtype]]](capacity=ngpus)
    var host_buffers = List[HostBuffer[dtype]](capacity=ngpus)

    # Create signal buffers for synchronization
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    # Calculate temp buffer size for signals.
    var max_length = 0
    for i in range(ngpus):
        max_length = max(max_length, lengths[i])
    var temp_buffer_num_bytes = ngpus * size_of[dtype]() * max_length

    # Initialize input buffers and signal buffers.
    for i in range(ngpus):
        var length = lengths[i]

        # Create device buffer.
        in_bufs_list.append(list_of_ctx[i].create_buffer_sync[dtype](length))

        # Create host buffer with test data.
        var host_buffer = list_of_ctx[i].enqueue_create_host_buffer[dtype](
            length
        )

        # Initialize with unique values per device.
        for j in range(length):
            host_buffer[j] = Scalar[dtype](
                i * 1000 + j
            )  # Device i has values i*1000 + index

        # Create and initialize signal buffers.
        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](
                size_of[Signal]() + temp_buffer_num_bytes
            )
        )
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
        rank_sigs[i] = (
            signal_buffers[i]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

        # Copy to device.
        list_of_ctx[i].enqueue_copy(in_bufs_list[i], host_buffer)

        host_buffers.append(host_buffer^)

    # Create output buffers - each device needs ngpus output buffers.
    for device_idx in range(ngpus):
        var device_outputs = List[DeviceBuffer[dtype]](capacity=ngpus)
        for input_idx in range(ngpus):
            var length = lengths[input_idx]
            device_outputs.append(
                list_of_ctx[device_idx].create_buffer_sync[dtype](length)
            )
        out_bufs_list.append(device_outputs^)

    # Build TileTensor arrays directly.
    comptime InTileType = type_of(
        TileTensor(in_bufs_list[0], row_major(lengths[0]))
        .as_immut()
        .as_unsafe_any_origin()
    )
    var tt_in_bufs = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InTileType: (
            TileTensor(in_bufs_list[i], row_major(lengths[i]))
            .as_immut()
            .as_unsafe_any_origin()
        )
    )

    comptime OutTileType = type_of(
        TileTensor(
            out_bufs_list[0][0], row_major(lengths[0])
        ).as_unsafe_any_origin()
    )

    def tt_out_bufs_at(i: Int) {mut out_bufs_list, imm} -> OutTileType:
        var device_idx = i // ngpus
        var input_idx = i % ngpus
        return TileTensor(
            out_bufs_list[device_idx][input_idx],
            row_major(lengths[input_idx]),
        ).as_unsafe_any_origin()

    var tt_out_bufs = Array[_, ngpus * ngpus](fill_with=tt_out_bufs_at)

    # Optional: vendor CCL (only if all lengths are equal; NCCL/RCCL requires uniform count).
    var uniform = True
    for i in range(1, ngpus):
        if lengths[i] != lengths[0]:
            uniform = False
            break

    if uniform and has_amd_gpu_accelerator():
        # Reset outputs for vendor test
        for device_idx in range(ngpus):
            for input_idx in range(ngpus):
                list_of_ctx[device_idx].enqueue_memset[dtype](
                    out_bufs_list[device_idx][input_idx], val=0
                )

        try:
            print("  Testing vendor CCL allgather (uniform counts)")
            vendor_ccl.allgather[dtype=dtype, ngpus=ngpus](
                tt_in_bufs, tt_out_bufs, list_of_ctx
            )

            for i in range(ngpus):
                list_of_ctx[i].synchronize()
            _verify_results[dtype](out_bufs_list, list_of_ctx, lengths, ngpus)
        except:
            pass

    # Test the implementation with rank_sigs (P2P-capable).
    print("  Testing implementation with rank_sigs (P2P-capable)")

    for gpu_idx in range(ngpus):
        var device_out = Array[_, ngpus](
            fill_with=lambda (src_idx: Int) -> OutTileType: tt_out_bufs[
                gpu_idx * ngpus + src_idx
            ]
        )
        allgather(
            tt_in_bufs, device_out, rank_sigs, list_of_ctx[gpu_idx], gpu_idx
        )

    # Synchronize all devices.
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Verify results for new implementation.
    _verify_results[dtype](out_bufs_list, list_of_ctx, lengths, ngpus)


def _verify_results[
    dtype: DType
](
    out_bufs_list: List[List[DeviceBuffer[dtype]]],
    list_of_ctx: List[DeviceContext],
    lengths: List[Int],
    ngpus: Int,
) raises:
    """Helper function to verify allgather results."""

    # Verify results - each device should have copies of all inputs.
    for device_idx in range(ngpus):
        for input_idx in range(ngpus):
            var length = lengths[input_idx]
            var host_output = list_of_ctx[
                device_idx
            ].enqueue_create_host_buffer[dtype](length)

            # Copy output back to host.
            list_of_ctx[device_idx].enqueue_copy(
                host_output, out_bufs_list[device_idx][input_idx]
            )
            list_of_ctx[device_idx].synchronize()

            # Verify this matches the original input from device input_idx.
            for j in range(length):
                var expected = Scalar[dtype](input_idx * 1000 + j)
                try:
                    assert_equal(host_output[j], expected)
                except e:
                    print(
                        "Verification failed: device",
                        device_idx,
                        "should have copy of input",
                        input_idx,
                    )
                    print(
                        "Index",
                        j,
                        "value:",
                        host_output[j],
                        "expected:",
                        expected,
                    )
                    raise e^


def grouped_all_gather_test[
    dtype: DType, ngpus: Int, group_size: Int
](list_of_ctx: List[DeviceContext], lengths: List[Int]) raises -> None:
    """Test allgather with two independent contiguous groups."""
    comptime assert ngpus == 4, "grouped test expects 4 GPUs"
    comptime assert group_size == 2, "grouped test expects groups of 2 GPUs"

    var in_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_bufs_list = List[List[DeviceBuffer[dtype]]](capacity=ngpus)
    var host_buffers = List[HostBuffer[dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    for i in range(ngpus):
        var length = lengths[i]
        in_bufs_list.append(list_of_ctx[i].create_buffer_sync[dtype](length))

        var host_buffer = list_of_ctx[i].enqueue_create_host_buffer[dtype](
            length
        )
        for j in range(length):
            host_buffer[j] = Scalar[dtype](i * 1000 + j)

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](size_of[Signal]())
        )
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
        rank_sigs[i] = (
            signal_buffers[i]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

        list_of_ctx[i].enqueue_copy(in_bufs_list[i], host_buffer)
        host_buffers.append(host_buffer^)

    for device_idx in range(ngpus):
        var group_start = (device_idx // group_size) * group_size
        var device_outputs = List[DeviceBuffer[dtype]](capacity=group_size)
        for local_idx in range(group_size):
            var input_idx = group_start + local_idx
            device_outputs.append(
                list_of_ctx[device_idx].create_buffer_sync[dtype](
                    lengths[input_idx]
                )
            )
        out_bufs_list.append(device_outputs^)

    comptime InTileType = type_of(
        TileTensor(in_bufs_list[0], row_major(lengths[0]))
        .as_immut()
        .as_unsafe_any_origin()
    )
    var tt_in_bufs = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InTileType: (
            TileTensor(in_bufs_list[i], row_major(lengths[i]))
            .as_immut()
            .as_unsafe_any_origin()
        )
    )

    comptime OutTileType = type_of(
        TileTensor(
            out_bufs_list[0][0], row_major(lengths[0])
        ).as_unsafe_any_origin()
    )

    def tt_out_bufs_at(i: Int) {mut out_bufs_list, imm} -> OutTileType:
        var device_idx = i // group_size
        var local_idx = i % group_size
        var group_start = (device_idx // group_size) * group_size
        var input_idx = group_start + local_idx
        return TileTensor(
            out_bufs_list[device_idx][local_idx],
            row_major(lengths[input_idx]),
        ).as_unsafe_any_origin()

    var tt_out_bufs = Array[_, ngpus * group_size](fill_with=tt_out_bufs_at)

    comptime for group_idx in range(ngpus // group_size):
        comptime group_start = group_idx * group_size
        var group_in_bufs = Array[_, group_size](
            fill_with=lambda (local_idx: Int) -> InTileType: tt_in_bufs[
                group_start + local_idx
            ]
        )
        var group_rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )

        comptime for local_idx in range(group_size):
            group_rank_sigs[local_idx] = rank_sigs[group_start + local_idx]

        comptime for local_idx in range(group_size):
            comptime device_idx = group_start + local_idx
            var device_out = Array[_, group_size](
                fill_with=lambda (src_idx: Int) -> OutTileType: tt_out_bufs[
                    device_idx * group_size + src_idx
                ]
            )

            allgather[domain_id=group_size](
                group_in_bufs,
                device_out,
                group_rank_sigs,
                list_of_ctx[device_idx],
                local_idx,
            )

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    for device_idx in range(ngpus):
        var group_start = (device_idx // group_size) * group_size
        for local_idx in range(group_size):
            var input_idx = group_start + local_idx
            var length = lengths[input_idx]
            var host_output = list_of_ctx[
                device_idx
            ].enqueue_create_host_buffer[dtype](length)
            list_of_ctx[device_idx].enqueue_copy(
                host_output, out_bufs_list[device_idx][local_idx]
            )
            list_of_ctx[device_idx].synchronize()

            for j in range(length):
                var expected = Scalar[dtype](input_idx * 1000 + j)
                try:
                    assert_equal(host_output[j], expected)
                except e:
                    print(
                        "Grouped verification failed: device",
                        device_idx,
                        "local input",
                        local_idx,
                        "global input",
                        input_idx,
                    )
                    raise e^

    _ = host_buffers^


def main() raises -> None:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    # Test configurations.
    comptime test_lengths: List[List[Int]] = [
        [8 * 1024, 8 * 1024],
        [128 * 1024, 8 * 1024],
        [8 * 1024, 256 * 1024],
        [8 * 1024, 8 * 1024, 8 * 1024, 8 * 1024],
        [128 * 1024, 256 * 1024, 8 * 1024, 64 * 1024],
        # Test uneven shapes.
        [37919, 37919, 37918, 37918],
        # Simple uneven case.
        [4, 3, 3],
        # Another uneven case with 2 GPUs.
        [1025, 1024],
        # Zero length cases
        [0, 0],
        [8 * 1024, 0],
        [0, 8 * 1024],
    ]

    comptime for test_idx in range(len(test_lengths)):
        comptime lengths = test_lengths[test_idx]
        comptime num_gpus = len(lengths)

        if DeviceContext.number_of_devices() < num_gpus:
            continue

        var ctx = List[DeviceContext]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))

        print("  Testing configuration:", test_idx, "with", num_gpus, "GPUs")
        all_gather_test[.bfloat16, ngpus=num_gpus](ctx, materialize[lengths]())

    if DeviceContext.number_of_devices() >= 4:
        var ctx = List[DeviceContext]()
        for i in range(4):
            ctx.append(DeviceContext(device_id=i))
        print("  Testing grouped allgather with 4 GPUs")
        comptime grouped_lengths: List[Int] = [
            8 * 1024,
            4 * 1024,
            6 * 1024,
            2 * 1024,
        ]
        grouped_all_gather_test[.bfloat16, ngpus=4, group_size=2](
            ctx, materialize[grouped_lengths]()
        )
