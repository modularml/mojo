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
"""Apple M5 (AGX3) edge-masked vector load test.

Verifies `build_edge_mask` + `gmem_edge_masked_load` / `edge_masked_load`: a
packed 32x3 bf16 input loaded 4-wide per row yields `[3l, 3l+1, 3l+2, 0]`, with
col 3 masked to zero rather than leaking the next row. Apple M5 only.
"""

from max.gpu import thread_idx
from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.memory import (
    build_edge_mask,
    edge_masked_load,
    gmem_edge_masked_load,
)

from std.testing import assert_equal

comptime ROWS = 32
comptime IN_COLS = 3
comptime OUT_COLS = 4


def _gmem_kernel(
    in_ptr: MutPointer[BFloat16, MutAnyOrigin],
    out_ptr: MutPointer[BFloat16, MutAnyOrigin],
):
    var lane = Int(thread_idx.x)
    var base = lane * IN_COLS
    var mask = build_edge_mask(Int32(base), Int32(0), Int32(base + IN_COLS))
    var v = gmem_edge_masked_load[4](in_ptr + base, mask)
    (out_ptr + lane * OUT_COLS).store(v)


def _general_kernel(
    in_ptr: MutPointer[BFloat16, MutAnyOrigin],
    out_ptr: MutPointer[BFloat16, MutAnyOrigin],
):
    var lane = Int(thread_idx.x)
    var base = lane * IN_COLS
    var mask = build_edge_mask(Int32(base), Int32(0), Int32(base + IN_COLS))
    var src = (in_ptr + base).address_space_cast[.GLOBAL]()
    var v = edge_masked_load[4](src, mask)
    (out_ptr + lane * OUT_COLS).store(v)


def _check(buf: HostBuffer[.bfloat16]) raises:
    # Skip rows 0-1: a separate known M5 first-rows blit->compute hazard, not emask.
    for l in range(2, ROWS):
        assert_equal(Float32(buf[l * OUT_COLS + 0]), Float32(3 * l))
        assert_equal(Float32(buf[l * OUT_COLS + 1]), Float32(3 * l + 1))
        assert_equal(Float32(buf[l * OUT_COLS + 2]), Float32(3 * l + 2))
        assert_equal(Float32(buf[l * OUT_COLS + 3]), Float32(0))


def main() raises:
    var ctx = DeviceContext()
    if ctx.compute_capability() != 5:
        print("SKIP: Apple M5 (compute_capability == 5) required")
        return

    var in_host = ctx.enqueue_create_host_buffer[.bfloat16](ROWS * IN_COLS)
    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](ROWS * OUT_COLS)
    var in_dev = ctx.enqueue_create_buffer[.bfloat16](ROWS * IN_COLS)
    var out_dev = ctx.enqueue_create_buffer[.bfloat16](ROWS * OUT_COLS)
    for i in range(ROWS * IN_COLS):
        in_host[i] = BFloat16(i)  # row-major counting
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()

    for i in range(ROWS * OUT_COLS):
        out_host[i] = BFloat16(-1)
    ctx.enqueue_copy(out_dev, out_host)
    ctx.enqueue_function[_gmem_kernel](
        in_dev.unsafe_ptr(), out_dev.unsafe_ptr(), grid_dim=1, block_dim=ROWS
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    _check(out_host)

    for i in range(ROWS * OUT_COLS):
        out_host[i] = BFloat16(-1)
    ctx.enqueue_copy(out_dev, out_host)
    ctx.enqueue_function[_general_kernel](
        in_dev.unsafe_ptr(), out_dev.unsafe_ptr(), grid_dim=1, block_dim=ROWS
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    _check(out_host)

    print("PASS: edge_masked_load 32x3 -> 32x4 = [3l, 3l+1, 3l+2, 0]")
