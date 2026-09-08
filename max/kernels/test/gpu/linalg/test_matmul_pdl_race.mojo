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
"""Test for PDL race condition in matmul kernel.

This test verifies that when PDL (Programmatic Dependent Launch) is enabled,
the dependent kernel does not start reading matmul output before the matmul
kernel has finished writing.

The test:
1. Runs matmul with PDL OVERLAP_AT_END (signals completion, dependent waits)
2. Immediately launches a "consumer" kernel that reads the output and writes to result
3. Compares result against expected values

If there's a race condition where launch_dependent_grids() is called before
the matmul output is fully written, the consumer kernel may read
incomplete/garbage data.

This is currently only checking via `_matmul_gpu` dispatch for a single shape
and only for bfloat16.
"""

from max.gpu import block_idx, thread_idx, block_dim, grid_dim
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
    pdl_launch_attributes,
)
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from std.math import ceildiv
from std.sys import get_defined_int


def consumer_kernel[
    dtype: DType,
](
    input: MutPointer[Scalar[dtype], MutAnyOrigin],
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    length_dev: Int32,
):
    """Consumer kernel that reads matmul output after PDL wait.

    Waits for dependent grids (matmul) before reading.
    Copies input to output (can verify output later on host).
    """
    var length = Int(length_dev)
    wait_on_dependent_grids()

    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    var inverse_tid = stride - tid

    # Read from matmul output and write to our output
    # Read backwards in an attempt to catch a data race...
    for i in range(length - inverse_tid, -1, -stride):
        output[i] = input[i]


def run_pdl_race_test[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
](ctx: DeviceContext, num_iters: Int) raises:
    """Run matmul with PDL, then verify output via dependent consumer kernel."""

    print(
        "Testing PDL race: dtype=",
        dtype,
        " M=",
        M,
        " N=",
        N,
        " K=",
        K,
        " iters=",
        num_iters,
    )

    # Allocate host buffers. Initialize A and B with 1.0 so the matmul result
    # is K (sum of K products of 1.0 * 1.0) for each element of C.
    var a_host = ctx.enqueue_create_host_buffer[dtype](M * K)
    var b_host = ctx.enqueue_create_host_buffer[dtype](K * N)
    var result_host = ctx.enqueue_create_host_buffer[dtype](M * N)
    for i in range(M * K):
        a_host[i] = Scalar[dtype](1.0)
    for i in range(K * N):
        b_host[i] = Scalar[dtype](1.0)

    # Expected value: each element of C = K (dot product of K ones)
    var expected_val = Scalar[dtype](K)

    # Allocate device buffers
    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device = ctx.enqueue_create_buffer[dtype](M * N)
    var result_device = ctx.enqueue_create_buffer[dtype](M * N)

    # Copy inputs to device
    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    # Create TileTensors for matmul
    var a_tensor = TileTensor(
        a_device,
        row_major(Coord(Idx[M], Idx[K])),
    )
    var b_tensor = TileTensor(
        b_device,
        row_major(Coord(Idx[N], Idx[K])),
    )
    var c_tensor = TileTensor(
        c_device,
        row_major(Coord(Idx[M], Idx[N])),
    )

    # Run multiple iterations to increase chance of catching race
    for iter in range(num_iters):
        # Clear output buffers
        ctx.enqueue_memset[dtype](c_device, 0)
        ctx.enqueue_memset[dtype](result_device, 0)

        # Run matmul with PDL enabled - signals at end, dependent waits
        _matmul_gpu[
            use_tensor_core=True,
            transpose_b=True,
            pdl_level=PDLLevel.OVERLAP_AT_END,
        ](c_tensor, a_tensor.as_immut(), b_tensor.as_immut(), ctx)
        #
        # Launch consumer kernel - waits for matmul via PDL
        var num_threads = 256
        var num_blocks = ceildiv(M * N, num_threads)
        comptime kernel = consumer_kernel[dtype]
        ctx.enqueue_function[kernel](
            c_device,
            result_device,
            Int32(M * N),
            grid_dim=num_blocks,
            block_dim=num_threads,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
        )

        # Sync and check results
        ctx.synchronize()

        ctx.enqueue_copy(result_host, result_device)
        ctx.synchronize()

        # Verify all values
        var error_count = 0
        var first_error_idx = -1
        var first_error_val = Scalar[dtype](0)

        for i in range(M * N):
            var val = result_host[i]
            var diff = val - expected_val
            if diff < 0:
                diff = -diff
            # Use relative tolerance for floating point comparison
            var tolerance = Scalar[dtype](0.1) * expected_val
            if tolerance < 0:
                tolerance = -tolerance
            if tolerance < 0.01:
                tolerance = 0.01
            if diff > tolerance:
                if first_error_idx < 0:
                    first_error_idx = i
                    first_error_val = val
                error_count += 1

        if error_count > 0:
            print(
                "FAIL: iter ",
                iter,
                " found ",
                error_count,
                " mismatches out of ",
                M * N,
            )
            print(
                "  First error at [",
                first_error_idx,
                "]: got ",
                first_error_val,
                " expected ",
                expected_val,
            )

            # Print a few more errors for debugging
            var printed = 0
            for i in range(M * N):
                if printed >= 5:
                    break
                var val = result_host[i]
                var diff = val - expected_val
                if diff < 0:
                    diff = -diff
                if diff > 0.01:
                    print("  [", i, "] got ", val, " expected ", expected_val)
                    printed += 1

            raise Error("PDL race condition detected!")

    print("PASS: ", num_iters, " iterations completed without race")


def main() raises:
    with DeviceContext() as ctx:
        # Use env vars for configurability
        # var M = get_defined_int["M", 8]()
        comptime M = get_defined_int["M", 32]()
        comptime N = get_defined_int["N", 1536]()
        comptime K = get_defined_int["K", 4096]()
        var iters = get_defined_int["ITERS", 100]()

        # Test with bfloat16 (common for SM90 matmul)
        run_pdl_race_test[.bfloat16, M, N, K](ctx, iters)

        print("All PDL race tests passed!")
