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
"""Broadcast over a device subgroup that does not start at device 0.

A DP replica runs its collectives on a slice of the visible devices (devices
4..7 of a DP2 x 8-GPU job) but still packs `rank_sigs` by group rank, 0..3.
`test_broadcast.mojo` only ever runs the full world, so it misses this.
"""

from std.math import ceildiv
from std.sys import size_of
from std.itertools import product
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, row_major
from std.testing import assert_true

from comm import Signal, MAX_GPUS
from comm.broadcast import broadcast
from comm.sync import enable_p2p, init_signal_buffer


# uint8 at 256 KiB reproduces the production 1-stage launch: 64 x 256 threads.
comptime test_lengths = (1023, 8 * 1024 + 3, 256 * 1024)
comptime test_dtypes = (DType.uint8, DType.bfloat16)
comptime test_gpu_counts = (2, 4)


@always_inline
@__parameter
def _input_value[dtype: DType](root: Int, j: Int) -> Scalar[dtype]:
    return Scalar[dtype](root + 1) + Scalar[dtype](j % 251)


def broadcast_subgroup_test[
    dtype: DType,
    ngpus: Int,
](
    list_of_ctxs: List[DeviceContext], length: Int, root: Int, dev_offset: Int
) raises:
    """Runs one broadcast over `ngpus` contexts based at device `dev_offset`.

    `rank_sigs` is packed by group rank, as production builds it, so a rank
    taken from the device id would read an out-of-group slot.
    """
    var root_ctx = list_of_ctxs[root]

    var host_input_ptr = root_ctx.enqueue_create_host_buffer[dtype](length)
    for j in range(length):
        host_input_ptr[j] = _input_value[dtype](root, j)
    var input_dev = root_ctx.enqueue_create_buffer[dtype](length)
    root_ctx.enqueue_copy(input_dev, host_input_ptr)

    var out_dev_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    comptime OutputTileType = TileTensor[
        dtype, type_of(row_major(length)), MutAnyOrigin
    ]
    var out_tiles = Array[OutputTileType, ngpus](uninitialized=True)

    var in_tile = TileTensor(input_dev, row_major(length)).as_immut()

    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    # Production leaves the slots past the group size uninitialized; poison
    # them so an out-of-group read faults deterministically.
    var rank_sigs = Array[_, MAX_GPUS](
        fill=MutPointer[Signal, MutAnyOrigin](
            unsafe_from_address=0xDEAD_BEEF_0000
        )
    )
    for i in range(ngpus):
        var ctx = list_of_ctxs[i]
        var out_ptr = ctx.enqueue_create_buffer[dtype](length)
        out_dev_list.append(out_ptr)
        out_tiles[i] = OutputTileType(
            out_ptr.unsafe_ptr().as_unsafe_any_origin(), row_major(length)
        )

    var num_bytes = length * size_of[dtype]()
    var signal_buf_size = size_of[Signal]() + ceildiv(num_bytes, ngpus)

    for i in range(ngpus):
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
        list_of_ctxs[i].enqueue_memset(out_dev_list[i], 0)
        list_of_ctxs[i].synchronize()

    comptime for i in range(ngpus):
        broadcast[ngpus](
            in_tile, out_tiles[i], rank_sigs, list_of_ctxs[i], root, rank=i
        )

    for i in range(ngpus):
        list_of_ctxs[i].synchronize()

    var host_output = List(length=length, fill=Scalar[dtype](0))
    for i in range(ngpus):
        list_of_ctxs[i].enqueue_copy(host_output, out_dev_list[i])
        list_of_ctxs[i].synchronize()
        for j in range(length):
            var expected = _input_value[dtype](root, j)
            if host_output[j] != expected:
                raise Error(
                    "Verification failed at group rank",
                    i,
                    "(device",
                    dev_offset + i,
                    ") index",
                    j,
                    "value:",
                    host_output[j],
                    "expected:",
                    expected,
                )


def main() raises:
    var num_devices = DeviceContext.number_of_devices()
    assert_true(num_devices > 1, "must have multiple GPUs")
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    # Placing a group off device 0 needs one device more than the group size;
    # print the skip so a 2-GPU runner's vacuous pass is visible.
    if num_devices < test_gpu_counts[0] + 1:
        print(
            "SKIPPED: broadcast-subgroup needs >=",
            test_gpu_counts[0] + 1,
            "GPUs to place a group off device 0; found",
            num_devices,
        )
        return

    comptime for gpu_idx, dtype_idx, length_idx in product(
        range(len(test_gpu_counts)),
        range(len(test_dtypes)),
        range(len(test_lengths)),
    ):
        comptime ngpus = rebind[Int](test_gpu_counts[gpu_idx])
        comptime dtype = rebind[DType](test_dtypes[dtype_idx])
        comptime length = rebind[Int](test_lengths[length_idx])

        # Trailing slice of the visible devices, so device id != group rank.
        var dev_offset = num_devices - ngpus
        if dev_offset <= 0:
            continue

        var list_of_ctxs = List[DeviceContext]()
        for i in range(ngpus):
            list_of_ctxs.append(DeviceContext(device_id=dev_offset + i))

        for root in range(ngpus):
            print(
                "====broadcast-subgroup-",
                dtype,
                "-ngpus-",
                ngpus,
                "-devoffset-",
                dev_offset,
                "-root-",
                root,
                "-nelems-",
                length,
            )
            broadcast_subgroup_test[dtype, ngpus](
                list_of_ctxs, length, root, dev_offset
            )
