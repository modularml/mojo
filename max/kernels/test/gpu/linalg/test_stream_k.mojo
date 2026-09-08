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
from std.math.uutils import umod

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.sync import Semaphore
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import DefaultEngine, TileTensor, row_major
from linalg.matmul.gpu import matmul_kernel_naive
from std.memory import alloc
from std.testing import assert_almost_equal

from std.utils import IndexList


def swizzle_tile(
    tile_id: Int,
    M: Int,
    N: Int,
    K: Int,
    BLOCK_M: Int,
    BLOCK_N: Int,
    BLOCK_K: Int,
    GROUP_M: Int,
) -> IndexList[2]:
    var grid_m = (M + BLOCK_M - 1) // BLOCK_M
    var grid_n = (N + BLOCK_N - 1) // BLOCK_N
    var width = GROUP_M * grid_n
    var group_id, tile_id_rem = divmod(tile_id, width)
    var group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    var pid_m = group_id * GROUP_M + (tile_id % group_size)
    var pid_n = tile_id_rem // group_size
    return IndexList[2](pid_m, pid_n)


def linear_tile(
    tile_id: Int,
    M: Int,
    N: Int,
    K: Int,
    BLOCK_M: Int,
    BLOCK_N: Int,
    BLOCK_K: Int,
    GROUP_M: Int,
) -> IndexList[2]:
    var pid_m, pid_n = divmod(tile_id, ((N + BLOCK_N - 1) // BLOCK_N))
    return IndexList[2](pid_m, pid_n)


def mac_loop[
    c_type: DType,
    a_type: DType,
    b_type: DType,
](
    C: MutPointer[Scalar[c_type], MutAnyOrigin],
    A: ImmPointer[Scalar[a_type], ImmutAnyOrigin],
    B: ImmPointer[Scalar[b_type], ImmutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
    locks: MutPointer[Int32, MutAnyOrigin],
    stride_am: Int,
    stride_ak: Int,
    stride_bk: Int,
    stride_bn: Int,
    stride_cm: Int,
    stride_cn: Int,
    iters_per_tile: Int,
    start_iter: Int,
    end_iter: Int,
    BLOCK_M: Int,
    BLOCK_N: Int,
    BLOCK_K: Int,
    GROUP_M: Int,
):
    var tile_id = start_iter // iters_per_tile
    var pid: IndexList[2]
    if GROUP_M > 0:
        pid = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    var rm_base = pid[0] * BLOCK_M
    var rn_base = pid[1] * BLOCK_N

    var tx = thread_idx.x
    var ty = thread_idx.y

    var global_r = rm_base + ty
    var global_c = rn_base + tx
    var accum = Scalar[c_type](0)
    var thread_id = thread_idx.x + thread_idx.y * block_dim.x
    var sema = Semaphore(locks + tile_id, thread_id)
    sema.fetch()

    for iter in range(start_iter, end_iter):
        var k_offset = (iter % iters_per_tile) * BLOCK_K

        for kk in range(BLOCK_K):
            var actual_k = k_offset + kk
            if global_r < M and actual_k < K and global_c < N:
                var a_val = A.load(global_r * stride_am + actual_k * stride_ak)
                var b_val = B.load(actual_k * stride_bk + global_c * stride_bn)
                accum += a_val.cast[c_type]() * b_val.cast[c_type]()

    var c_offset = global_r * stride_cm + global_c * stride_cn
    # Due to the construction of the stream-k scheduling, every last
    # reduction iteration will be executed early. In fact, in the ideal case,
    # the last split reduction will be executed by a SM first. Therefore, for
    # the semaphore signaling, we should initialize the semaphore to 0 which
    # corresponds to the last reduction iteration, and go backward from there,
    # i.e. each thread block waits for the `end_iter` signal and releases the
    # `start_iter` signal.
    if (end_iter % iters_per_tile) == 0:
        sema.wait(0)
        if global_r < M and global_c < N:
            C[c_offset] = accum
    else:
        sema.wait(end_iter)
        if global_r < M and global_c < N:
            C[c_offset] += accum
    sema.release(Int32(start_iter))


def first_wave_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    BLOCK_M: Int,
    BLOCK_N: Int,
    BLOCK_K: Int,
    GROUP_M: Int,
](
    C: MutPointer[Scalar[c_type], MutAnyOrigin],
    A: ImmPointer[Scalar[a_type], ImmutAnyOrigin],
    B: ImmPointer[Scalar[b_type], ImmutAnyOrigin],
    M_dev: Int32,
    N_dev: Int32,
    K_dev: Int32,
    locks: MutPointer[Int32, MutAnyOrigin],
    stride_am_dev: Int32,
    stride_ak_dev: Int32,
    stride_bk_dev: Int32,
    stride_bn_dev: Int32,
    stride_cm_dev: Int32,
    stride_cn_dev: Int32,
    total_full_tiles_streamk_dev: Int32,
    total_partial_tiles_streamk_dev: Int32,
    iters_per_tile_dev: Int32,
):
    var M = Int(M_dev)
    var N = Int(N_dev)
    var K = Int(K_dev)
    var stride_am = Int(stride_am_dev)
    var stride_ak = Int(stride_ak_dev)
    var stride_bk = Int(stride_bk_dev)
    var stride_bn = Int(stride_bn_dev)
    var stride_cm = Int(stride_cm_dev)
    var stride_cn = Int(stride_cn_dev)
    var total_full_tiles_streamk = Int(total_full_tiles_streamk_dev)
    var total_partial_tiles_streamk = Int(total_partial_tiles_streamk_dev)
    var iters_per_tile = Int(iters_per_tile_dev)
    var pid = block_idx.x

    var start_iter = Int(
        pid * total_full_tiles_streamk
        + (
            pid if pid
            < total_partial_tiles_streamk else total_partial_tiles_streamk
        )
    )
    var last_iter = Int(
        (pid + 1) * total_full_tiles_streamk
        + (
            (pid + 1) if (pid + 1)
            < total_partial_tiles_streamk else total_partial_tiles_streamk
        )
    )

    while start_iter < last_iter:
        var remainder = iters_per_tile - umod(start_iter, iters_per_tile)
        var boundary = start_iter + remainder
        var end_iter = boundary if (boundary < last_iter) else last_iter
        mac_loop(
            C,
            A,
            B,
            M,
            N,
            K,
            locks,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            iters_per_tile,
            start_iter,
            end_iter,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            GROUP_M,
        )
        start_iter = end_iter


def full_tiles_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    BLOCK_M: Int,
    BLOCK_N: Int,
    BLOCK_K: Int,
    GROUP_M: Int,
](
    C: MutPointer[Scalar[c_type], MutAnyOrigin],
    A: ImmPointer[Scalar[a_type], ImmutAnyOrigin],
    B: ImmPointer[Scalar[b_type], ImmutAnyOrigin],
    M_dev: Int32,
    N_dev: Int32,
    K_dev: Int32,
    locks: ImmPointer[Int32, ImmutAnyOrigin],
    stride_am_dev: Int32,
    stride_ak_dev: Int32,
    stride_bk_dev: Int32,
    stride_bn_dev: Int32,
    stride_cm_dev: Int32,
    stride_cn_dev: Int32,
    total_tiles_streamk_dev: Int32,
):
    var M = Int(M_dev)
    var N = Int(N_dev)
    var K = Int(K_dev)
    var stride_am = Int(stride_am_dev)
    var stride_ak = Int(stride_ak_dev)
    var stride_bk = Int(stride_bk_dev)
    var stride_bn = Int(stride_bn_dev)
    var stride_cm = Int(stride_cm_dev)
    var stride_cn = Int(stride_cn_dev)
    var total_tiles_streamk = Int(total_tiles_streamk_dev)
    var tile_id = block_idx.x + total_tiles_streamk
    var pid: IndexList[2]
    if GROUP_M > 0:
        pid = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    var rm_base = pid[0] * BLOCK_M
    var rn_base = pid[1] * BLOCK_N

    var tx = thread_idx.x
    var ty = thread_idx.y

    var global_r = rm_base + ty
    var global_c = rn_base + tx
    var accum = Scalar[c_type](0)

    var steps = (K + BLOCK_K - 1) // BLOCK_K
    for s in range(steps):
        var k_offset = s * BLOCK_K
        for kk in range(BLOCK_K):
            var actual_k = k_offset + kk
            if global_r < M and actual_k < K and global_c < N:
                var a_val = A.load(global_r * stride_am + actual_k * stride_ak)
                var b_val = B.load(actual_k * stride_bk + global_c * stride_bn)
                accum += a_val.cast[c_type]() * b_val.cast[c_type]()

    if global_r < M and global_c < N:
        C[global_r * stride_cm + global_c * stride_cn] = accum


# (1) Stream-K wave:  multiple blocks subdivide K across some subset of tiles
#        +-----------+    +-----------+     +-----------+    +-----------+
#        | Block 0   |    | Block 1   |     | Block 2   |    | Block 3   |
#        +-----+-----+    +-----+-----+     +-----+-----+    +-----+-----+
#        |T0,K0 |T0,K1|    |T0,K2 |T1,K0|     |T1,K1 |T1,K2|    |T2,K0 |T2,K1| ...
#         partial  partial  partial  partial  partial  partial  partial  partial
#
#     - The tile T0 is computed in 3 partial K-chunks by Blocks 0,1,...
#     - The tile T1 is also subdivided, etc.
#     - M <-> tile dimension,  N <-> tile dimension,  K <-> subdivided dimension
#     - Atomic merges & locks coordinate partial sums.
#
# (2) Full Tiles wave:   each remaining tile is handled by 1 block fully
#        +-----------+  <--- Block 10 covers tile T3 entirely, no partial sums
#        |   T3      |
#        +-----------+
#
#        +-----------+  <--- Block 11 covers tile T4 entirely, no partial sums
#        |   T4      |
#        +-----------+
#
#        +-----------+  <--- Block 12 covers tile T5 entirely, no partial sums
#        |   T5      |
#        +-----------+


def matmul_stream_k[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    *,
    total_programs_streamk: Int,
](
    c: TileTensor[
        mut=True,
        c_type,
        address_space=.GENERIC,
        ...,
        Engine=DefaultEngine[element_width=1],
    ],
    a: TileTensor[
        a_type,
        address_space=.GENERIC,
        ...,
        Engine=DefaultEngine[element_width=1],
    ],
    b: TileTensor[
        b_type,
        address_space=.GENERIC,
        ...,
        Engine=DefaultEngine[element_width=1],
    ],
    M: Int,
    N: Int,
    K: Int,
    ctx: DeviceContext,
) raises:
    comptime BLK_M = 16
    comptime BLK_N = 16
    comptime BLK_K = 16

    var total_blocks_M = (M + BLK_M - 1) // BLK_M
    var total_blocks_N = (N + BLK_N - 1) // BLK_N
    var iters_per_tile = (K + BLK_K - 1) // BLK_K
    comptime GROUP_M = 8

    var total_tiles = total_blocks_M * total_blocks_N
    var total_tiles_streamk = total_tiles % total_programs_streamk
    var total_blocking_tiles = total_tiles - total_tiles_streamk
    var total_iters_streamk = total_tiles_streamk * iters_per_tile
    var total_full_tiles_streamk: Int
    var total_partial_tiles_streamk: Int
    if total_iters_streamk == 0:
        total_full_tiles_streamk = 0
        total_partial_tiles_streamk = 0
    else:
        total_full_tiles_streamk, total_partial_tiles_streamk = divmod(
            total_iters_streamk, total_programs_streamk
        )

    var locks_data = ctx.enqueue_create_buffer[.int32](total_tiles_streamk)
    ctx.enqueue_memset(locks_data, 0)

    print("M=", M, ", N=", N, ", K=", K)
    print(
        "Total tiles=",
        total_tiles,
        "  (M-tiles=",
        total_blocks_M,
        ", N-tiles=",
        total_blocks_N,
        ")",
    )
    print("iters_per_tile=", iters_per_tile)
    print(
        "total_tiles_streamk=",
        total_tiles_streamk,
        ", total_blocking_tiles=",
        total_blocking_tiles,
    )
    print(
        "total_full_tiles_streamk=",
        total_full_tiles_streamk,
        ", total_partial_tiles_streamk=",
        total_partial_tiles_streamk,
    )

    var c_buffer = DeviceBuffer[c_type](
        ctx, c._storage, c.num_elements(), owning=False
    )
    var a_buffer = DeviceBuffer[a_type](
        ctx, a._storage, a.num_elements(), owning=False
    )
    var b_buffer = DeviceBuffer[b_type](
        ctx, b._storage, b.num_elements(), owning=False
    )

    if total_programs_streamk > 0:
        comptime first_wave = first_wave_kernel[
            c_type,
            a_type,
            b_type,
            BLK_M,
            BLK_N,
            BLK_K,
            GROUP_M,
        ]

        ctx.enqueue_function[first_wave](
            c_buffer,
            a_buffer,
            b_buffer,
            Int32(M),
            Int32(N),
            Int32(K),
            locks_data,
            Int32(K),
            Int32(1),
            Int32(N),
            Int32(1),
            Int32(N),
            Int32(1),
            Int32(total_full_tiles_streamk),
            Int32(total_partial_tiles_streamk),
            Int32(iters_per_tile),
            grid_dim=total_programs_streamk,
            block_dim=(BLK_N, BLK_M),
        )
        ctx.synchronize()

    if total_blocking_tiles > 0:
        comptime full_tiles = full_tiles_kernel[
            c_type,
            a_type,
            b_type,
            BLK_M,
            BLK_N,
            BLK_K,
            GROUP_M,
        ]
        ctx.enqueue_function[full_tiles](
            c_buffer,
            a_buffer,
            b_buffer,
            Int32(M),
            Int32(N),
            Int32(K),
            locks_data,
            Int32(K),
            Int32(1),
            Int32(N),
            Int32(1),
            Int32(N),
            Int32(1),
            Int32(total_tiles_streamk),
            grid_dim=total_blocking_tiles,
            block_dim=(BLK_N, BLK_M),
        )
        ctx.synchronize()

    _ = locks_data^
    return


def run_matmul_stream_k[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
](ctx: DeviceContext) raises:
    print("== run_matmul kernel stream_k")

    var a_host = alloc[Scalar[dtype]](M * K)
    var b_host = alloc[Scalar[dtype]](K * N)
    var c_host = alloc[Scalar[dtype]](M * N)
    var c_host_n = alloc[Scalar[dtype]](M * N)

    for i in range(M * K):
        var val = Float32(i % 20)
        a_host[i] = val.cast[dtype]()

    for i in range(K * N):
        var val = Float32(i % 20)
        b_host[i] = val.cast[dtype]()

    for i in range(M * N):
        var val = Float32(0)
        c_host[i] = val.cast[dtype]()
        c_host_n[i] = c_host[i]

    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device = ctx.enqueue_create_buffer[dtype](M * N)
    var a_buf = TileTensor(a_device, row_major[M, K]())
    var b_buf = TileTensor(b_device, row_major[K, N]())
    var c_buf = TileTensor(c_device, row_major[M, N]())

    var c_device_n = ctx.enqueue_create_buffer[dtype](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    comptime sm_count = ctx.default_device_info.sm_count

    matmul_stream_k[total_programs_streamk=sm_count](
        c_buf,
        a_buf,
        b_buf,
        M,
        N,
        K,
        ctx,
    )

    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    comptime BLOCK_DIM = 16

    # Create TileTensors for the naive kernel.
    # a/b are constructed as immutable to match the ImmutAnyOrigin
    # parameters that matmul_kernel_naive expects (enqueue_function
    # requires exact type matches).

    var c_buf_n = TileTensor(c_device_n, row_major[M, N]())

    comptime kernel = matmul_kernel_naive[
        dtype,
        dtype,
        dtype,
        type_of(c_buf_n).LayoutType,
        type_of(a_buf).LayoutType,
        type_of(b_buf).LayoutType,
        BLOCK_DIM,
    ]

    ctx.enqueue_function[kernel](
        c_buf_n,
        a_buf.as_immut(),
        b_buf.as_immut(),
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.enqueue_copy(c_host_n, c_device_n)
    ctx.synchronize()

    var rtol = 0.01

    for i in range(M * N):
        var out_val = c_host[i]
        var out_ref = c_host_n[i]
        assert_almost_equal(out_val, out_ref, rtol=rtol)


def main() raises:
    with DeviceContext() as ctx:
        run_matmul_stream_k[.float32, 128, 128, 128](ctx)
        run_matmul_stream_k[.float32, 512, 2560, 8192](ctx)
        run_matmul_stream_k[.float32, 256, 256, 1024](ctx)
        run_matmul_stream_k[.float32, 128, 128, 1024](ctx)
