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

from std.math import iota
from std.os import abort
from std.sys import size_of

from max.algorithm.functional import parallelize_over_rows
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext, HostBuffer
from internal_utils import arg_parse, human_readable_size
from std.testing import assert_almost_equal, assert_true

from std.utils import IndexList


@fieldwise_init
struct Config(ImplicitlyCopyable, Writable):
    var direction: Int
    var pinned_memory: Bool
    # Definitions for direction field.
    comptime DToH = 0
    comptime HToD = 1
    comptime DToD = 2
    comptime P2P = 3
    # Different possible configurations.
    comptime DEVICE_TO_HOST = Self(Self.DToH, False)
    comptime DEVICE_TO_HOST_PINNED = Self(Self.DToH, True)
    comptime HOST_TO_DEVICE = Self(Self.HToD, False)
    comptime HOST_PINNED_TO_DEVICE = Self(Self.HToD, True)
    comptime DEVICE_TO_DEVICE = Self(Self.DToD, False)
    comptime PEER_TO_PEER = Self(Self.P2P, False)
    comptime UNDEFINED = Self(-1, False)

    def __eq__(self, other: Self) -> Bool:
        return (
            self.direction == other.direction
            and self.pinned_memory == other.pinned_memory
        )

    @staticmethod
    def get(handle: String) -> Self:
        if handle == "host_to_device":
            return Self.HOST_TO_DEVICE
        elif handle == "host_pinned_to_device":
            return Self.HOST_PINNED_TO_DEVICE
        elif handle == "device_to_host":
            return Self.DEVICE_TO_HOST
        elif handle == "device_to_host_pinned":
            return Self.DEVICE_TO_HOST_PINNED
        elif handle == "device_to_device":
            return Self.DEVICE_TO_DEVICE
        elif handle == "peer_to_peer":
            return Self.PEER_TO_PEER
        else:
            print("UNDEFINED")
            print(
                "options: host_to_device, host_pinned_to_device,"
                " device_to_host, device_to_host_pinned, device_to_device,"
                " peer_to_peer"
            )
            return Self.UNDEFINED

    def write_to(self, mut writer: Some[Writer]):
        if self.direction == Self.DToD:
            writer.write("device_to_device")
            return

        if self.direction == Self.DToH:
            writer.write("device_to_")

        if self.pinned_memory:
            writer.write("host_pinned")
        else:
            writer.write("host")

        if self.direction == Self.HToD:
            writer.write("_to_device")


@no_inline
def bench_memcpy(
    mut b: Bench,
    length_in_bytes: Int,
    *,
    config: Config,
    context: DeviceContext,
) raises:
    comptime dtype = DType.float32
    var length_in_elements = length_in_bytes // size_of[dtype]()
    var mem_host: HostBuffer[dtype] = context.enqueue_create_host_buffer[dtype](
        length_in_elements
    ) if config.pinned_memory else DeviceContext(
        api="cpu"
    ).enqueue_create_host_buffer[
        dtype
    ](
        length_in_elements
    )

    # Allocate device buffers. If we're doing a d2d transfer, then we need 2
    # buffers (source & destination). Otherwise, we only need one. But we don't
    # want to put this allocation inside the timed region, so we always allocate
    # both and drop the size of the second buffer to zero in the case of a non
    # d2d test.
    var mem_device = context.enqueue_create_buffer[dtype](length_in_elements)
    var mem2_device = context.enqueue_create_buffer[dtype](
        length_in_elements if config.direction == Config.DToD else 0
    )

    @always_inline
    def bench_func(mut b: Bencher) {imm}:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            if config.direction == Config.DToH:
                context.enqueue_copy(mem_host, mem_device)
            elif config.direction == Config.HToD:
                context.enqueue_copy(mem_device, mem_host)
            elif config.direction == Config.DToD:
                context.enqueue_copy(mem_device, mem2_device)
            else:
                raise Error("Unexpected transfer direction")

        bencher_iter_custom(b, kernel_launch, context)

    # For D2D transfers, we're reading the entire buffer into gpu cache/sharedmem,
    # then writing it back to a new address in vram. This means we're really
    # moving the tensor in/out of vram twice (one read + one write), and therefore
    # we need to double the size in order to calculate the correct bandwidth.
    var transferred_size_in_bytes = length_in_bytes
    if config.direction == Config.DToD:
        transferred_size_in_bytes *= 2

    b.bench_function(
        bench_func,
        BenchId(
            String(t"memcpy_{config}"),
            input_id="length=" + human_readable_size(length_in_bytes),
        ),
        [ThroughputMeasure(BenchMetric.bytes, transferred_size_in_bytes)],
    )
    context.synchronize()

    # Ensure that we are not queuing any free operations during the timed block.
    _ = mem_host
    _ = mem_device
    _ = mem2_device


@no_inline
def bench_p2p(
    mut b: Bench,
    length_in_bytes: Int,
    *,
    ctx1: DeviceContext,
    ctx2: DeviceContext,
) raises:
    comptime dtype = DType.float32
    var length_in_elements = length_in_bytes // size_of[dtype]()

    # Create host buffers for verification
    var host_ptr = List(length=length_in_elements, fill=Scalar[dtype](0))

    # Initialize source data with known pattern
    iota(host_ptr)

    # Create and initialize device buffers
    var src_buf = ctx1.enqueue_create_buffer[dtype](length_in_elements)
    var dst_buf = ctx2.enqueue_create_buffer[dtype](length_in_elements)

    # Copy initial data to source buffer
    ctx1.enqueue_copy(src_buf, host_ptr)
    ctx1.synchronize()

    @always_inline
    def bench_func(mut b: Bencher) {imm}:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            ctx2.enqueue_copy(dst_buf, src_buf)

        bencher_iter_custom(b, kernel_launch, ctx1)

    # Create list of throughput measures
    var measures: List = [
        # Raw bandwidth (considering only one transfer)
        ThroughputMeasure(BenchMetric.bytes, length_in_bytes),
    ]

    b.bench_function(
        bench_func,
        BenchId(
            "memcpy_p2p",
            input_id="length=" + human_readable_size(length_in_bytes),
        ),
        measures=measures,
    )

    # Copy back for verification
    ctx2.enqueue_copy(host_ptr, dst_buf)
    ctx2.synchronize()

    # Parallel verification
    def verify_chunk(start: Int, end: Int) {imm}:
        for i in range(start, end):
            try:
                assert_almost_equal(host_ptr[i], Float32(i))
            except e:
                print("Verification failed at index", i)
                print("Expected:", i, "Got:", host_ptr[i])
                abort(String(e))

    # Parallelize verification using sync_parallelize
    var shape = IndexList[1](
        length_in_elements,
    )
    parallelize_over_rows(verify_chunk, shape, 0, 256)

    # Cleanup
    _ = src_buf
    _ = dst_buf
    _ = host_ptr^


def main() raises:
    var m = Bench()

    var log2_length = arg_parse("log2_length", 20)
    var mode = arg_parse("mode", "host_to_device")
    assert_true(log2_length > 0)
    var length = 1 << log2_length
    var config = Config.get(mode)

    if not (config == Config.UNDEFINED) and not (config == Config.PEER_TO_PEER):
        with DeviceContext() as ctx:
            bench_memcpy(m, length, config=config, context=ctx)

    elif config.direction == Config.P2P:
        var num_devices = DeviceContext.number_of_devices()
        if num_devices > 1:
            # Create contexts for both same-device and peer device transfers
            var ctx1 = DeviceContext(device_id=0)
            var ctx2 = DeviceContext(device_id=1)

            # Benchmark peer context D2D
            bench_p2p(m, length, ctx1=ctx1, ctx2=ctx2)
        else:
            print("Only one device found, skipping peer-to-peer benchmarks")

    m.dump_report()
