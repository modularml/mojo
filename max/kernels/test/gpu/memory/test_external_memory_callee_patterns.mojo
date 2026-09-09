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

# Functional tests for dynamic external memory (mainly for Apple Silicon).

from max.gpu import block_idx
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import external_memory
from max.gpu.sync import barrier
from std.testing import assert_equal

comptime BLOCK_SIZE = 64
comptime NUM_BLOCKS = 4
# Expected sum of thread indices 0..BLOCK_SIZE-1 within each block.
comptime EXPECTED_SUM = BLOCK_SIZE * (BLOCK_SIZE - 1) // 2

# ── Pattern 1: helper (not the kernel) owns the external_memory reference ─────
#
# @no_inline keeps the helper as a distinct function in the IR so the compiler
# must handle the callee-uses-extern-shared case.


@no_inline
def _callee_fill_and_reduce(
    data: MutPointer[Float32, MutAnyOrigin], local_idx: Int, blk_idx: Int
):
    """Each thread writes its index into shared memory; thread 0 reduces to sum.
    """
    var smem = external_memory[
        Float32,
        address_space=.SHARED,
        alignment=4,
        name="callee_ext",
    ]()
    smem[local_idx] = Float32(local_idx)
    barrier()
    if local_idx == 0:
        var s = Float32(0)
        for i in range(BLOCK_SIZE):
            s += smem[i]
        data[blk_idx] = s


def test_external_memory_in_callee(ctx: DeviceContext) raises:
    """Verify that external_memory used only in a callee works correctly."""
    print("== test_external_memory_in_callee")

    def callee_kernel(data: MutPointer[Float32, MutAnyOrigin]):
        # Kernel itself does NOT call external_memory; the callee does.
        _callee_fill_and_reduce(data, thread_idx.x, block_idx.x)

    var host_buf = alloc[Float32](NUM_BLOCKS)
    var dev_buf = ctx.enqueue_create_buffer[.float32](NUM_BLOCKS)

    for i in range(NUM_BLOCKS):
        host_buf[i] = -1.0

    ctx.enqueue_copy(dev_buf, host_buf)

    comptime kernel = callee_kernel
    ctx.enqueue_function[kernel](
        dev_buf,
        grid_dim=NUM_BLOCKS,
        block_dim=BLOCK_SIZE,
        shared_mem_bytes=BLOCK_SIZE * 4,
    )

    ctx.enqueue_copy(host_buf, dev_buf)
    ctx.synchronize()

    for i in range(NUM_BLOCKS):
        assert_equal(host_buf[i], Float32(EXPECTED_SUM))

    _ = dev_buf
    host_buf.free()


# ── Pattern 2: external_memory used 2 levels below the kernel ─────────────────
#
# Call chain: deep_kernel → _deep_bar → _deep_baz; only _deep_baz touches
# external_memory.  _deep_bar is a pure pass-through.


@no_inline
def _deep_baz(
    data: MutPointer[Float32, MutAnyOrigin], local_idx: Int, blk_idx: Int
):
    """Innermost callee — owns the external_memory reference."""
    var smem = external_memory[
        Float32,
        address_space=.SHARED,
        alignment=4,
        name="deep_ext",
    ]()
    smem[local_idx] = Float32(local_idx)
    barrier()
    if local_idx == 0:
        var s = Float32(0)
        for i in range(BLOCK_SIZE):
            s += smem[i]
        data[blk_idx] = s


@no_inline
def _deep_bar(
    data: MutPointer[Float32, MutAnyOrigin], local_idx: Int, blk_idx: Int
):
    """Pass-through callee — does NOT reference external_memory."""
    _deep_baz(data, local_idx, blk_idx)


def test_external_memory_deep_callgraph(ctx: DeviceContext) raises:
    """Verify that external_memory 2 levels below the kernel works correctly."""
    print("== test_external_memory_deep_callgraph")

    def deep_kernel(data: MutPointer[Float32, MutAnyOrigin]):
        # Kernel calls _deep_bar which calls _deep_baz (the only ext_memory user).
        _deep_bar(data, thread_idx.x, block_idx.x)

    var host_buf = alloc[Float32](NUM_BLOCKS)
    var dev_buf = ctx.enqueue_create_buffer[.float32](NUM_BLOCKS)

    for i in range(NUM_BLOCKS):
        host_buf[i] = -1.0

    ctx.enqueue_copy(dev_buf, host_buf)

    comptime kernel = deep_kernel
    ctx.enqueue_function[kernel](
        dev_buf,
        grid_dim=NUM_BLOCKS,
        block_dim=BLOCK_SIZE,
        shared_mem_bytes=BLOCK_SIZE * 4,
    )

    ctx.enqueue_copy(host_buf, dev_buf)
    ctx.synchronize()

    for i in range(NUM_BLOCKS):
        assert_equal(host_buf[i], Float32(EXPECTED_SUM))

    _ = dev_buf
    host_buf.free()


def main() raises:
    with DeviceContext() as ctx:
        test_external_memory_in_callee(ctx)
        test_external_memory_deep_callgraph(ctx)
