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

from std.math import exp, sqrt
from std.random import rand
from std.collections import Array
from max.gpu import block_idx, thread_idx, block_dim, grid_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation
from max.gpu.primitives.warp import (
    shuffle_idx,
    lane_group_max,
    lane_group_sum,
)
from max.gpu.primitives.id import (
    lane_id,
    warp_id,
)

# Match CUDA exactly
comptime WARP_SIZE = 32
comptime BLOCK_SIZE = 512
comptime N_WARPS = BLOCK_SIZE // WARP_SIZE  # 16
comptime B_r = 32
comptime B_c = 32
comptime D_MODEL = 128
comptime B_r_warp = B_r // N_WARPS  # 2
comptime d_size = D_MODEL // WARP_SIZE  # 4


# Shared memory layout (Q is now in registers with shuffle broadcast):
# KT_j: B_c * D_MODEL = 32 * 128 = 4096 floats
# S_i: B_r * B_c = 32 * 32 = 1024 floats
# V_j: B_c * D_MODEL = 32 * 128 = 4096 floats


def flashattention_forward_kernel(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K: UnsafePointer[Float32, MutAnyOrigin],
    V: UnsafePointer[Float32, MutAnyOrigin],
    N_dev: Int32,
    scaling: Float32,
    out_D: UnsafePointer[Float32, MutAnyOrigin],
    out_O: UnsafePointer[Float32, MutAnyOrigin],
):
    # Int is not device-passable; widen the fixed-width arg.
    var N = Int(N_dev)
    var T_r = N // B_r
    var T_c = N // B_c

    var KT_j = unsafe_stack_allocation[
        B_c * D_MODEL,
        Float32,
        address_space=.SHARED,
    ]()
    var S_i = unsafe_stack_allocation[
        B_r * B_c,
        Float32,
        address_space=.SHARED,
    ]()
    var V_j = unsafe_stack_allocation[
        B_c * D_MODEL,
        Float32,
        address_space=.SHARED,
    ]()

    # Per-thread register storage (like CUDA)
    var O_i = Array[Float32, B_r_warp * d_size](fill=0.0)  # 2 * 4 = 8
    var D_i = Array[Float32, B_r_warp](fill=0.0)  # 2
    var m_i = Array[Float32, B_r_warp](fill=0.0)  # 2
    var Q_i = Array[Float32, B_r_warp * d_size](
        fill=0.0
    )  # 2 * 4 = 8 (Q in registers!)

    var neg_inf = Float32(-3.40282e38)

    var i = block_idx.x
    while i < T_r:
        # Initialize O, D, m
        for ii in range(B_r_warp):
            D_i[ii] = 0.0
            m_i[ii] = neg_inf
            for k in range(d_size):
                O_i[ii * d_size + k] = 0.0

        # Load Q into registers (Figure 20.10 from CUDA)
        # Each thread in warp loads d_size elements (4 elements for d=128, warp=32)
        for ii in range(B_r_warp):
            var dd = lane_id()
            var ddd = 0
            while dd < D_MODEL:
                var row_idx = (
                    B_r * i + B_r_warp * warp_id() + ii
                ) * D_MODEL + dd
                Q_i[ii * d_size + ddd] = Q[row_idx]
                dd += WARP_SIZE
                ddd += 1

        for j in range(T_c):
            # Load KT and V into shared memory
            for jj in range(B_c):
                var dd = thread_idx.x
                while dd < D_MODEL:
                    var g_row = B_c * j + jj
                    KT_j[dd * B_c + jj] = K[g_row * D_MODEL + dd]
                    V_j[jj * D_MODEL + dd] = V[g_row * D_MODEL + dd]
                    dd += block_dim.x

            barrier()

            # Each warp processes B_r_warp rows
            for ii in range(B_r_warp):
                var row_in_block = warp_id() * B_r_warp + ii
                var row_global = B_r * i + row_in_block

                # Compute S row and find max (Figure 20.12)
                var curr_max = neg_inf

                var jj = lane_id()
                while jj < B_c:
                    var S_ij = Float32(0.0)

                    # Dot product using shuffle broadcast (like CUDA __shfl_sync)
                    for dd in range(D_MODEL):
                        # Q_i[ii][dd/32] is in thread (dd % 32)
                        # Use shuffle_idx to broadcast from that thread
                        var q_val = shuffle_idx(
                            Q_i[ii * d_size + (dd // WARP_SIZE)],
                            UInt32(dd % WARP_SIZE),
                        )
                        var kt_val = KT_j[dd * B_c + jj]
                        S_ij += q_val * kt_val

                    var col_global = B_c * j + jj

                    # Causal masking
                    if row_global < col_global:
                        S_ij = neg_inf
                    else:
                        S_ij = scaling * S_ij

                    S_i[row_in_block * B_c + jj] = S_ij

                    if S_ij > curr_max:
                        curr_max = S_ij

                    jj += WARP_SIZE

                # Warp-level max reduction using shuffle (like CUB WarpReduce)
                var curr_max_warp = lane_group_max[num_lanes=WARP_SIZE](
                    curr_max
                )

                # Update m and D (Figure 20.14)
                var last_m = m_i[ii]
                if m_i[ii] < curr_max_warp:
                    m_i[ii] = curr_max_warp
                    D_i[ii] *= exp(last_m - m_i[ii])

                # Compute P and update D (Figure 20.15)
                var curr_sum = Float32(0.0)
                jj = lane_id()
                while jj < B_c:
                    var col_global = B_c * j + jj
                    var s_val = S_i[row_in_block * B_c + jj]

                    var P_ij = Float32(0.0)
                    if row_global >= col_global:
                        P_ij = exp(s_val - m_i[ii])

                    S_i[row_in_block * B_c + jj] = P_ij
                    curr_sum += P_ij

                    jj += WARP_SIZE

                # Warp-level sum reduction using shuffle
                var curr_sum_warp = lane_group_sum[num_lanes=WARP_SIZE](
                    curr_sum
                )
                D_i[ii] += curr_sum_warp

                # Compute O (Figure 20.13)
                var dd = lane_id()
                var ddd = 0
                while dd < D_MODEL:
                    # Scale previous O by exp(last_m - m_i[ii])
                    O_i[ii * d_size + ddd] *= exp(last_m - m_i[ii])

                    var O_ij = Float32(0.0)
                    for kk in range(B_c):
                        var p_val = S_i[row_in_block * B_c + kk]
                        var v_val = V_j[kk * D_MODEL + dd]
                        O_ij += p_val * v_val

                    O_i[ii * d_size + ddd] += O_ij
                    dd += WARP_SIZE
                    ddd += 1

            barrier()

        # Store O and D (Figure 20.16)
        for ii in range(B_r_warp):
            var row_in_block = warp_id() * B_r_warp + ii
            var row = B_r * i + row_in_block

            for k in range(d_size):
                var col = k * WARP_SIZE + lane_id()
                if row < N and col < D_MODEL:
                    out_O[row * D_MODEL + col] = O_i[ii * d_size + k] / D_i[ii]

            if lane_id() == 0:
                if row < N:
                    out_D[row] = D_i[ii]

        i += grid_dim.x


def cpu_attention(
    h_Q: UnsafePointer[mut=False, Float32, _],
    h_K: UnsafePointer[mut=False, Float32, _],
    h_V: UnsafePointer[mut=False, Float32, _],
    h_O: UnsafePointer[mut=True, Float32, _],
    N: Int,
    D: Int,
):
    var scale = 1.0 / sqrt(Float32(D))
    var neg_inf = Float32(-3.40282e38)

    for i in range(N):
        var scores = alloc[Float32](N)
        var max_val = neg_inf

        for j in range(N):
            var dot = Float32(0.0)
            for k in range(D):
                dot += h_Q[i * D + k] * h_K[j * D + k]

            if j > i:
                scores[j] = neg_inf
            else:
                scores[j] = dot * scale

            if scores[j] > max_val:
                max_val = scores[j]

        var sum_exp = Float32(0.0)
        for j in range(N):
            if j > i:
                scores[j] = 0.0
            else:
                scores[j] = exp(scores[j] - max_val)
                sum_exp += scores[j]

        for k in range(D):
            var out_val = Float32(0.0)
            for j in range(N):
                out_val += (scores[j] / sum_exp) * h_V[j * D + k]
            h_O[i * D + k] = out_val

        scores.free()


def main() raises:
    print("Running FlashAttention Test (Mojo with Warp Shuffles)")
    var device = DeviceContext()
    var N = 1024

    print("Initializing data N=", N, "d=", D_MODEL, "...")

    var size_mat = N * D_MODEL
    var h_Q = alloc[Float32](size_mat)
    var h_K = alloc[Float32](size_mat)
    var h_V = alloc[Float32](size_mat)
    var h_O = alloc[Float32](size_mat)
    var h_O_ref = alloc[Float32](size_mat)
    var h_D = alloc[Float32](N)

    rand(h_Q, size_mat)
    rand(h_K, size_mat)
    rand(h_V, size_mat)

    for k in range(size_mat):
        h_Q[k] -= 0.5
        h_K[k] -= 0.5
        h_V[k] -= 0.5

    var d_Q = device.enqueue_create_buffer[.float32](size_mat)
    var d_K = device.enqueue_create_buffer[.float32](size_mat)
    var d_V = device.enqueue_create_buffer[.float32](size_mat)
    var d_O = device.enqueue_create_buffer[.float32](size_mat)
    var d_D_out = device.enqueue_create_buffer[.float32](N)

    device.enqueue_copy(d_Q, h_Q)
    device.enqueue_copy(d_K, h_K)
    device.enqueue_copy(d_V, h_V)

    var grid_size = (N + B_r - 1) // B_r
    var scaling = 1.0 / sqrt(Float32(D_MODEL))

    print("Launching kernel with grid=", grid_size, "block=", BLOCK_SIZE, "...")

    device.enqueue_function[flashattention_forward_kernel](
        d_Q,
        d_K,
        d_V,
        Int32(N),
        scaling,
        d_D_out,
        d_O,
        grid_dim=(grid_size, 1, 1),
        block_dim=(BLOCK_SIZE, 1, 1),
    )

    device.enqueue_copy(h_O, d_O)
    device.synchronize()

    print("Running CPU verification...")
    cpu_attention(h_Q, h_K, h_V, h_O_ref, N, D_MODEL)

    var max_diff = Float32(0.0)
    var err_count = 0
    for i in range(size_mat):
        var diff = abs(h_O[i] - h_O_ref[i])
        if diff > max_diff:
            max_diff = diff
        if diff > 1e-4:
            if err_count < 10:
                print("Mismatch at", i, "GPU:", h_O[i], "CPU:", h_O_ref[i])
            err_count += 1

    print("Max diff:", max_diff)
    if err_count == 0:
        print("TEST PASSED")
    else:
        print("TEST FAILED with", err_count, "errors")

    h_Q.free()
    h_K.free()
    h_V.free()
    h_O.free()
    h_O_ref.free()
    h_D.free()
