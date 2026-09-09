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
# REQUIRES: NVIDIA-GPU
# RUN: %mojo %s
from max.gpu import block_dim, grid_dim, block_idx, thread_idx
from max.gpu.sync import barrier
from std.math import iota
from std.os import abort
from shmem import *
from std.ffi import c_int
from std.sys.info import size_of
from max.gpu.host import DeviceBuffer

comptime min_size = 1024 * 1024 * 32
comptime max_size = min_size * 16
comptime num_blocks = 32
comptime threads_per_block = 256
comptime iters = 4
comptime warmup_iters = 1
comptime step_factor = 2
comptime chunk_size = 1024 * 256


def ring_reduce(
    dst_ptr: Pointer[c_int, MutAnyOrigin],
    src_ptr: Pointer[c_int, ImmutAnyOrigin],
    nreduce_dev: Int32,
    signal_ptr: Pointer[UInt64, MutAnyOrigin],
    chunk_size_dev: Int32,
):
    """Perform Allreduce using ring algorithm.

    This implements a ring-based allreduce that consists of two phases:
    1. Reduce phase: Data flows in ring pattern accumulating values
    2. Broadcast phase: Final result is broadcast back through the ring

    Args:
        dst_ptr: Destination buffer for reduced data.
        src_ptr: Source buffer with input data.
        nreduce_dev: Number of elements to reduce.
        signal_ptr: Signaling buffer for synchronization.
        chunk_size_dev: Size of each chunk in bytes.
    """
    # `Int` is not device-passable; widen the fixed-width args.
    var nreduce = Int(nreduce_dev)
    var chunk_size = Int(chunk_size_dev)
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + 1) % npes

    var thread_id = thread_idx.x
    var num_threads = block_dim.x
    var num_blocks = grid_dim.x
    var block_idx = block_idx.x
    var elems_per_block = nreduce // num_blocks

    if elems_per_block * (block_idx + 1) > nreduce:
        return

    var src = src_ptr + block_idx * elems_per_block
    var dst = dst_ptr + block_idx * elems_per_block
    var signal = signal_ptr + block_idx

    var chunk_elems = chunk_size // size_of[DType.int32]()
    var num_chunks = elems_per_block // chunk_elems

    # Reduce phase - data flows through ring accumulating values
    for chunk in range(num_chunks):
        # Wait for data from previous PE (except PE 0 which starts)
        if mype != 0:
            if thread_id == 0:
                shmem_signal_wait_until(signal, SHMEM_CMP_GE, UInt64(chunk + 1))

            barrier()

            var i = thread_id
            while i < chunk_elems:
                dst[i] = dst[i] + src[i]
                i += num_threads
            barrier()

        if thread_id == 0:
            shmem_put_signal_nbi(
                dst,
                src if mype == 0 else dst,
                chunk_elems,
                signal,
                1,
                SHMEM_SIGNAL_ADD,
                peer,
            )
        src += chunk_elems
        dst += chunk_elems

    # Broadcast phase - final result flows back through ring
    dst -= num_chunks * chunk_elems
    if thread_id == 0:
        for chunk in range(num_chunks):
            if mype < npes - 1:
                shmem_signal_wait_until(
                    signal,
                    SHMEM_CMP_GE,
                    UInt64(chunk + 1 if mype == 0 else num_chunks + chunk + 1),
                )
            if mype < npes - 2:
                shmem_put_signal_nbi(
                    dst, dst, chunk_elems, signal, 1, SHMEM_SIGNAL_ADD, peer
                )
            dst += chunk_elems
        signal[0] = 0


def bench_ring_reduce(ctx: SHMEMContext) raises:
    var min_ints = min_size // size_of[DType.int32]()
    assert (
        min_ints % num_blocks == 0
    ), "min_size must be divisible by num_blocks"

    var mype = shmem_my_pe()
    var npes = shmem_n_pes()

    # Allocate buffers
    var max_ints = max_size // size_of[DType.int32]()

    var dst = ctx.enqueue_create_buffer[.int32](max_ints)
    var src = ctx.enqueue_create_buffer[.int32](max_ints)
    var data_h = alloc[Int32](max_ints)
    var signal = shmem_calloc[.uint64](num_blocks)

    # Initialize test data - each element has value equal to its index
    iota(data_h, max_ints)

    # Copy test data to source buffer
    src.enqueue_copy_from(data_h)
    ctx.barrier_all()

    var dev_ctx = ctx.get_device_context()

    # Test different sizes
    var size = min_size
    while size <= max_size:
        var num_ints = size // size_of[DType.int32]()

        # Warmup iterations
        for _ in range(warmup_iters):
            ctx.enqueue_function_collective_checked[ring_reduce](
                dst,
                src,
                Int32(num_ints),
                DeviceBuffer[.uint64](
                    ctx._ctx, signal, num_blocks, owning=False
                ),
                Int32(chunk_size),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
            ctx.barrier_all()
        ctx.synchronize()

        def benchmark() raises {imm}:
            ctx.enqueue_function_collective_checked[ring_reduce](
                dst,
                src,
                Int32(num_ints),
                DeviceBuffer[.uint64](
                    ctx._ctx, signal, num_blocks, owning=False
                ),
                Int32(chunk_size),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
            ctx.barrier_all()

        var elapsed_ns = dev_ctx.execution_time(benchmark, iters) / iters
        var elapsed_ms = Float64(elapsed_ns) / 1e6

        ctx.synchronize()

        # Validate results - copy back and check
        dst.enqueue_copy_to(data_h)
        ctx.synchronize()

        # Each element should be i * npes after allreduce
        for i in range(num_ints):
            var expected = Int32(Int32(i) * npes)
            if data_h[i] != expected:
                # Avoid assert_equal overhead on these large buffers
                abort(
                    String(
                        "PE: ",
                        mype,
                        " unexpected value at data_h[",
                        i,
                        "]",
                        " expected: ",
                        expected,
                        " actual: ",
                        data_h[i],
                    )
                )

        if mype == 0:
            print(
                "Test passed on size:",
                size / 1024 / 1024,
                "MB",
                "\telapsed:",
                elapsed_ms,
                "ms",
            )

        size *= step_factor


def main() raises:
    shmem_launch[bench_ring_reduce]()
