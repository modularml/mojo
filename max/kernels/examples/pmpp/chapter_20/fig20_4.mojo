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

from std.math import exp, abs
from std.random import rand
from max.gpu import block_idx, thread_idx, block_dim, grid_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation

comptime BLOCK_SIZE = 256
comptime WARP_SIZE = 32


@always_inline
def max_op(a: Float32, b: Float32) -> Float32:
    if a > b:
        return a
    return b


@always_inline
def sum_op(a: Float32, b: Float32) -> Float32:
    return a + b


# Robust Block Reduce using Shared Memory (No Shuffle)
@always_inline
def block_reduce[
    op: def(Float32, Float32) thin -> Float32
](
    val: Float32,
    shared_mem: UnsafePointer[Float32, MutAnyOrigin, address_space=.SHARED],
) -> Float32:
    var tid = thread_idx.x
    var smem = shared_mem

    # Write to shared memory
    smem[tid] = val

    barrier()

    # Thread 0 performs reduction sequentially
    var res = val

    if tid == 0:
        var acc = smem[0]
        for i in range(1, BLOCK_SIZE):
            acc = op(acc, smem[i])
        res = acc

    return res


def softmax_kernel(
    S: UnsafePointer[Float32, MutAnyOrigin],
    D: UnsafePointer[Float32, MutAnyOrigin],
    P: UnsafePointer[Float32, MutAnyOrigin],
    N_dev: Int32,
):
    # Int is not device-passable; widen the fixed-width arg.
    var N = Int(N_dev)
    var D_ptr = D
    var P_ptr = P

    var temp_store = unsafe_stack_allocation[
        BLOCK_SIZE,
        Float32,
        address_space=.SHARED,
    ]()
    var broadcast_slot = unsafe_stack_allocation[
        1,
        Float32,
        address_space=.SHARED,
    ]()

    var row = block_idx.x
    var tid = thread_idx.x

    var S_row = S + (row * N)
    var P_row = P_ptr + (row * N)

    var neg_inf = Float32(-3.40282e38)
    var max_val_thread = neg_inf

    var col = tid
    while col < N:
        if col <= row:
            var val = S_row[col]
            if val > max_val_thread:
                max_val_thread = val
        col += block_dim.x

    barrier()

    var max_val_row = block_reduce[max_op](
        max_val_thread, temp_store.as_unsafe_any_origin()
    )

    if tid == 0:
        broadcast_slot[0] = max_val_row
    barrier()
    max_val_row = broadcast_slot[0]

    var sum_thread = Float32(0.0)
    col = tid
    while col < N:
        if col <= row:
            sum_thread += exp(S_row[col] - max_val_row)
        col += block_dim.x

    barrier()

    var sum_row = block_reduce[sum_op](
        sum_thread, temp_store.as_unsafe_any_origin()
    )

    if tid == 0:
        broadcast_slot[0] = sum_row
    barrier()
    sum_row = broadcast_slot[0]

    col = tid
    while col < N:
        var val = Float32(0.0)
        if col <= row:
            val = exp(S_row[col] - max_val_row) / sum_row
        P_row[col] = val
        col += block_dim.x

    if tid == 0:
        D_ptr[row] = sum_row


def cpu_softmax(
    h_S: UnsafePointer[mut=False, Float32, _],
    h_P: UnsafePointer[mut=True, Float32, _],
    N: Int,
):
    var P_ptr = h_P
    var neg_inf = Float32(-3.40282e38)
    for i in range(N):
        var max_val = neg_inf
        for j in range(N):
            if j <= i:
                if h_S[i * N + j] > max_val:
                    max_val = h_S[i * N + j]

        var sum_val = Float32(0.0)
        for j in range(N):
            if j <= i:
                sum_val += exp(h_S[i * N + j] - max_val)

        for j in range(N):
            if j <= i:
                P_ptr[i * N + j] = exp(h_S[i * N + j] - max_val) / sum_val
            else:
                P_ptr[i * N + j] = 0.0


def main() raises:
    print("Running Softmax Test (Mojo)")
    var device = DeviceContext()

    var test_sizes: List[Int] = [4, 8, 16, 32, 64, 128]

    for i in range(len(test_sizes)):
        var N = test_sizes[i]
        print("\n=== Testing with N =", N, "===")

        var size_S = N * N
        var size_D = N
        var size_P = N * N

        var h_S = alloc[Float32](size_S)
        var h_D = alloc[Float32](size_D)
        var h_P = alloc[Float32](size_P)
        var h_P_ref = alloc[Float32](size_P)

        rand(h_S, size_S)
        for k in range(size_S):
            h_S[k] = h_S[k] * 10.0 - 5.0

        cpu_softmax(h_S, h_P_ref, N)

        var d_S = device.enqueue_create_buffer[.float32](size_S)
        var d_D = device.enqueue_create_buffer[.float32](size_D)
        var d_P = device.enqueue_create_buffer[.float32](size_P)

        device.enqueue_copy(d_S, h_S)

        device.enqueue_function[softmax_kernel](
            d_S,
            d_D,
            d_P,
            Int32(N),
            grid_dim=(N, 1, 1),
            block_dim=(BLOCK_SIZE, 1, 1),
        )

        device.enqueue_copy(h_P, d_P)
        device.enqueue_copy(h_D, d_D)

        device.synchronize()

        var max_diff = Float32(0.0)
        var passed = True
        for k in range(size_P):
            var diff = abs(h_P[k] - h_P_ref[k])
            if diff > max_diff:
                max_diff = diff
            if diff > 1e-4:
                print("Mismatch at", k, "GPU:", h_P[k], "CPU:", h_P_ref[k])
                passed = False
                break

        if passed:
            print("PASSED (max diff:", max_diff, ")")
        else:
            print("FAILED")

        h_S.free()
        h_D.free()
        h_P.free()
        h_P_ref.free()
