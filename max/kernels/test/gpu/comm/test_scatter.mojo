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
"""Test scatter+broadcast kernel.

Uses the example from KERN-2435: DP=4, TP=2, 8 GPUs distributing row_offsets.

  row_offsets = [0, 5, 12, 20, 28, 35, 40, 48, 56]
  Sequence lengths: [5, 7, 8, 8, 7, 5, 8, 8]

  Split by Replica (2 sequences each, reindexed from 0):
    Replica A (seq 0-1): [0, 5, 12]
    Replica B (seq 2-3): [0, 8, 16]
    Replica C (seq 4-5): [0, 7, 12]
    Replica D (seq 6-7): [0, 8, 16]

  Distribution (DP=4, TP=2):
    Replica A [0,5,12]  -> GPU 0, GPU 1
    Replica B [0,8,16]  -> GPU 2, GPU 3
    Replica C [0,7,12]  -> GPU 4, GPU 5
    Replica D [0,8,16]  -> GPU 6, GPU 7
"""

from layout import Idx, TileTensor, row_major
from std.collections import Array
from std.math import ceildiv
from std.sys import size_of
from max.gpu.host import DeviceBuffer, DeviceContext
from std.testing import assert_true

from comm import Signal, MAX_GPUS
from comm.scatter import scatter
from comm.sync import enable_p2p, init_signal_buffer

comptime dtype = DType.uint32


def _test_pull[
    ngpus: Int,
    dp_size: Int,
](expected: List[List[Scalar[dtype]]]) raises:
    """Generic scatter pull test parameterized by ngpus and dp_size."""
    comptime tp_size = ceildiv(ngpus, dp_size)

    if DeviceContext.number_of_devices() < ngpus:
        print(
            "Skipping scatter ngpus=",
            ngpus,
            " DP=",
            dp_size,
            ": need",
            ngpus,
            "GPUs",
        )
        return

    # Find max chunk size for host buffer allocation.
    var max_chunk_size = 0
    for dp in range(dp_size):
        if len(expected[dp]) > max_chunk_size:
            max_chunk_size = len(expected[dp])

    var ctxs = List[DeviceContext]()
    for i in range(ngpus):
        ctxs.append(DeviceContext(device_id=i))

    # Allocate input chunks on GPU 0.
    var input_devbufs = List[DeviceBuffer[dtype]]()
    var host_buf = ctxs[0].enqueue_create_host_buffer[dtype](max_chunk_size)

    comptime InputTileType = TileTensor[
        dtype, type_of(row_major(Int(0))), ImmutAnyOrigin
    ]
    var tt_input_bufs = Array[InputTileType, dp_size](uninitialized=True)

    for dp in range(dp_size):
        var n = len(expected[dp])
        var dev_buf = ctxs[0].enqueue_create_buffer[dtype](n)
        for j in range(n):
            host_buf[j] = expected[dp][j]
        ctxs[0].enqueue_copy(dev_buf, host_buf)
        ctxs[0].synchronize()
        tt_input_bufs[dp] = TileTensor(dev_buf, row_major(n)).as_immut()
        input_devbufs.append(dev_buf)

    # Output buffers on each GPU (sized to its replica's chunk).
    var output_devbufs = List[DeviceBuffer[dtype]]()
    comptime OutputTileType = TileTensor[
        dtype, type_of(row_major(Int(0))), MutAnyOrigin
    ]
    var out_tiles = Array[OutputTileType, ngpus](uninitialized=True)
    for i in range(ngpus):
        var replica = i // tp_size
        var n = len(expected[replica])
        var out_buf = ctxs[i].enqueue_create_buffer[dtype](n)
        ctxs[i].enqueue_memset(out_buf, 0)
        ctxs[i].synchronize()
        out_tiles[i] = OutputTileType(
            out_buf.unsafe_ptr().as_unsafe_any_origin(), row_major(n)
        )
        output_devbufs.append(out_buf)

    # Signal buffers.
    var signal_bufs = List[DeviceBuffer[.uint8]]()
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    for i in range(ngpus):
        var sig_buf = ctxs[i].create_buffer_sync[.uint8](size_of[Signal]())
        init_signal_buffer(sig_buf, ctxs[i])
        ctxs[i].synchronize()
        rank_sigs[i] = (
            sig_buf.unsafe_ptr().bitcast[Signal]().as_unsafe_any_origin()
        )
        signal_bufs.append(sig_buf)

    # Launch scatter.
    comptime for i in range(ngpus):
        scatter[ngpus=ngpus, dp_size=dp_size](
            tt_input_bufs, out_tiles[i], rank_sigs, ctxs[i]
        )
    for i in range(ngpus):
        ctxs[i].synchronize()

    # Verify.
    var host_out = List(length=max_chunk_size, fill=Scalar[dtype](0))
    for gpu in range(ngpus):
        var replica = gpu // tp_size
        var n = len(expected[replica])
        ctxs[gpu].enqueue_copy(host_out, output_devbufs[gpu])
        ctxs[gpu].synchronize()
        for j in range(n):
            if host_out[j] != expected[replica][j]:
                raise Error(
                    "GPU",
                    gpu,
                    "(replica",
                    replica,
                    ") index",
                    j,
                    ": got",
                    host_out[j],
                    "expected",
                    expected[replica][j],
                )
    print(
        "PASS: scatter ngpus=",
        ngpus,
        ", DP=",
        dp_size,
        ", max_elems=",
        max_chunk_size,
    )

    _ = signal_bufs^
    _ = output_devbufs^
    _ = input_devbufs^
    _ = host_buf^


def _test_dp2() raises:
    """2 GPUs, DP=2."""
    var expected = List[List[UInt32]]()
    expected.append([0, 5, 12])  # Replica A
    expected.append([0, 8, 16])  # Replica B
    _test_pull[ngpus=2, dp_size=2](expected)


def _test_dp4[ngpus: Int]() raises:
    """DP=4 with configurable ngpus (KERN-2435 data)."""
    var expected = List[List[UInt32]]()
    expected.append([0, 5, 12])  # Replica A
    expected.append([0, 8, 16])  # Replica B
    expected.append([0, 7, 12])  # Replica C
    expected.append([0, 8])  # Replica D (2 elements)
    _test_pull[ngpus=ngpus, dp_size=4](expected)


def _test_dp2_fewer_elems_gpu0() raises:
    """2 GPUs, DP=2: GPU-0's replica gets fewer elements than GPU-1's."""
    var expected = List[List[UInt32]]()
    expected.append([42])  # Replica A (1 element)
    expected.append([10, 20, 30])  # Replica B (3 elements)
    _test_pull[ngpus=2, dp_size=2](expected)


def _test_dp2_single_elem() raises:
    """2 GPUs, DP=2: each GPU gets exactly 1 element."""
    var expected = List[List[UInt32]]()
    expected.append([7])  # Replica A
    expected.append([99])  # Replica B
    _test_pull[ngpus=2, dp_size=2](expected)


def _test_dp4_large_chunks() raises:
    """8 GPUs, DP=4: large chunks that require multiple thread blocks."""
    comptime NUM_ELEMS = 16384

    var expected = List[List[UInt32]]()
    for dp in range(4):
        var chunk = List[UInt32](capacity=NUM_ELEMS)
        var offset = dp * NUM_ELEMS
        for i in range(NUM_ELEMS):
            chunk.append(UInt32(offset + i))
        expected.append(chunk^)
    _test_pull[ngpus=8, dp_size=4](expected)


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    _test_dp2()
    _test_dp2_fewer_elems_gpu0()
    _test_dp2_single_elem()
    _test_dp4[ngpus=4]()
    _test_dp4[ngpus=5]()
    _test_dp4[ngpus=6]()
    _test_dp4[ngpus=7]()
    _test_dp4[ngpus=8]()
    _test_dp4_large_chunks()
