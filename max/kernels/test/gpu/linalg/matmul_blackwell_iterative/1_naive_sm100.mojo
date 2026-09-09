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
from std.sys import argv

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
from std.testing import assert_almost_equal

from std.utils.index import IndexList


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


def kernel_1[
    M: Int,
    N: Int,
    K: Int,
    transpose_b: Bool = True,
    BLOCKSIZE: Int = 32,
](
    c: LayoutTensor[.bfloat16, Layout.row_major(M, N), MutAnyOrigin],
    a: LayoutTensor[.bfloat16, Layout.row_major(M, K), MutAnyOrigin],
    b: LayoutTensor[.bfloat16, Layout.row_major(K, N), MutAnyOrigin],
):
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x

    if row < M and col < N:
        # Still accumulate in float32 for precision
        var acc: Float32 = 0

        for k in range(K):
            var a_val = rebind[Float32](a[row, k].cast[.float32]())
            var b_val = rebind[Float32](b[k, col].cast[.float32]())
            acc += a_val * b_val

        c[row, col] = acc.cast[.bfloat16]()


def test_kernel_1[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
    benchmark: Bool = False,
    prob_shape: IndexList[3] = IndexList[3](1, 1, 1),
](ctx: DeviceContext) raises:
    comptime M = prob_shape[0]
    comptime N = prob_shape[1]
    comptime K = prob_shape[2]

    print(M, "x", N, "x", K)

    var a = ManagedLayoutTensor[a_type, Layout.row_major(M, K)](ctx)
    random(a.tensor[update=False]())
    comptime b_layout = Layout.row_major(K, N)
    var b = ManagedLayoutTensor[b_type, b_layout](ctx)
    random(b.tensor[update=False]())
    var c = ManagedLayoutTensor[c_type, Layout.row_major(M, N)](ctx)
    var c_ref = ManagedLayoutTensor[c_type, Layout.row_major(M, N)](ctx)

    comptime b_vendor_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var b_vendor = ManagedLayoutTensor[b_type, b_vendor_layout](ctx)

    comptime if transpose_b:
        var b_tensor = b.tensor[update=False]()
        var b_vendor_tensor = b_vendor.tensor[update=True]()
        for k in range(K):
            for n in range(N):
                b_vendor_tensor[n, k] = b_tensor[k, n]
    else:
        var b_tensor = b.tensor[update=False]()
        var b_vendor_tensor = b_vendor.tensor[update=True]()
        for k in range(K):
            for n in range(N):
                b_vendor_tensor[k, n] = b_tensor[k, n]

    comptime kernel = kernel_1[
        M, N, K, transpose_b=transpose_b, BLOCKSIZE=BLOCKSIZE
    ]
    # Use 1D thread block for memory coalescing
    comptime BLOCKSIZE = 32

    ctx.enqueue_function[kernel](
        c.device_tensor(),
        a.device_tensor(),
        b.device_tensor(),
        grid_dim=(ceildiv(N, BLOCKSIZE), ceildiv(M, BLOCKSIZE)),
        block_dim=(BLOCKSIZE, BLOCKSIZE),
    )

    ctx.synchronize()

    if benchmark:
        comptime num_runs = 50
        comptime num_warmup = 20

        @always_inline
        def run_kernel(ctx: DeviceContext) raises {mut a, mut b, mut c, imm}:
            ctx.enqueue_function[kernel](
                c.device_tensor[update=False](),
                a.device_tensor[update=False](),
                b.device_tensor[update=False](),
                grid_dim=(ceildiv(N, BLOCKSIZE), ceildiv(M, BLOCKSIZE)),
                block_dim=(BLOCKSIZE, BLOCKSIZE),
            )

        for _ in range(num_warmup):
            run_kernel(ctx)
        ctx.synchronize()
        print("finished warmup")

        var nstime = (
            Float64(ctx.execution_time(run_kernel, num_runs)) / num_runs
        )
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12

        print("  Average time: ", sectime * 1000, " ms")
        print("  Performance: ", TFlop / sectime, " TFLOPS")
        print()
    else:
        vendor_blas.matmul(
            ctx,
            c_ref.device_tensor[update=False](),
            a.device_tensor[update=False](),
            b_vendor.device_tensor(),
            c_row_major=True,
            transpose_b=transpose_b,
        )

        ctx.synchronize()

        var c_host = c.tensor()
        var c_host_ref = c_ref.tensor()

        for m in range(M):
            for n in range(N):
                assert_almost_equal(
                    c_host[m, n],
                    c_host_ref[m, n],
                    atol=1e-2,
                    rtol=5e-2,
                    msg=String(m) + ", " + String(n),
                )
        print("TEST PASSED")


def main() raises:
    with DeviceContext() as ctx:
        if is_benchmark():
            test_kernel_1[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                transpose_b=True,
                prob_shape=IndexList[3](4096, 4096, 4096),
                benchmark=True,
            ](ctx)
            return

        # Test with transpose_b=True
        print("Testing with transpose_b=True")
        test_kernel_1[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            transpose_b=True,
            prob_shape=IndexList[3](4096, 4096, 4096),
        ](ctx)
