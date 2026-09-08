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

from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import thread_idx, warp_id
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.sync import barrier
from max.gpu.compute.arch.tcgen05 import *
from layout import Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from std.memory import unsafe_stack_allocation
from std.testing import assert_almost_equal


def tcgen05_st_ld_roundtrip_kernel[
    M: Int, N: Int
](data: LayoutTensor[.float32, Layout.row_major(M, N), MutAnyOrigin]):
    var elect_one_warp = warp_id() == 0
    var elect_one_thread = thread_idx.x == 0

    var ptr_tmem_addr = unsafe_stack_allocation[
        1, UInt32, address_space=.SHARED, alignment=16
    ]()

    comptime width = N
    comptime num_cols = 512

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, num_cols)

    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    var data_st = Array[_, width](
        fill_with=lambda (n: Int) -> Float32: Float32(thread_idx.x * N + n)
    )

    tcgen05_st[
        datapaths=16,
        bits=256,
        repeat=2,
        pack=False,
    ](tmem_addr, data_st)

    tcgen05_store_wait()

    var data_ld = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat=2,
        dtype=DType.float32,
        pack=False,
        width=width,
    ](tmem_addr)

    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, num_cols)

    for n in range(N):
        if data_ld[n] == data_st[n]:
            data[thread_idx.x, n] = data_ld[n]


def test_tcgen05_st_ld_roundtrip(ctx: DeviceContext) raises:
    comptime M = 128
    comptime N = 8
    var data = ManagedLayoutTensor[
        .float32,
        Layout.row_major(M, N),
    ](ctx)

    comptime kernel = tcgen05_st_ld_roundtrip_kernel[M, N]
    ctx.enqueue_function[kernel](
        data.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(M),
    )
    ctx.synchronize()
    var data_host = data.tensor()
    for m in range(M):
        for n in range(N):
            assert_almost_equal(
                data_host[m, n],
                Float32(m * N + n),
                atol=1e-3,
                rtol=1e-4,
            )


def tcgen05_cp_ld_roundtrip_kernel[
    M: Int, N: Int
](data: LayoutTensor[.float32, Layout.row_major(M, N), MutAnyOrigin]):
    comptime M_smem = 128
    comptime N_smem = 8
    comptime SBO = 256
    comptime LBO = 128

    comptime smem_layout = Layout.row_major(M_smem, N_smem)
    var smem_tile = LayoutTensor[
        .float32,
        smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    var s_desc = MMASmemDescriptor.create[
        SBO, LBO, TensorMapSwizzle.SWIZZLE_NONE
    ](smem_tile.ptr)

    # Order values according to `tcgen05{.ld,.st}.16x256b` with N=4 elements per thread,
    # if mapped back to SRAM:
    # Note that we copy 8x128bit atoms from SRAM to TMEM
    # (8x1 128bit atoms, no swizzle, row-major/K-major),
    # so the 16x256bit SRAM tensor is effectively split into 4 quadrants.
    #
    # |                            256 bit contiguous in TMEM                          |
    # | 128 bit contiguous in SRAM            || 128 bit contiguous in SRAM            |
    # +---------+---------+---------+---------++---------+---------+---------+---------+ -----
    # | T0: r0  | T0: r1  | T1: r0  | T1: r1  || T2: r0  | T2: r1  | T3: r0  | T3: r1  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+
    # | T4: r0  | T4: r1  | T5: r0  | T5: r1  || T6: r0  | T6: r1  | T7: r0  | T7: r1  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+
    # | T8: r0  | T8: r1  | T9: r0  | T9: r1  ||T10: r0  |T10: r1  |T11: r0  |T11: r1  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+
    # |T12: r0  |T12: r1  |T13: r0  |T13: r1  ||T14: r0  |T14: r1  |T15: r0  |T15: r1  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+ 8 rows
    # |T16: r0  |T16: r1  |T17: r0  |T17: r1  ||T18: r0  |T18: r1  |T19: r0  |T19: r1  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+
    # |T20: r0  |T20: r1  |T21: r0  |T21: r1  ||T22: r0  |T22: r1  |T23: r0  |T23: r1  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+
    # |T24: r0  |T24: r1  |T25: r0  |T25: r1  ||T26: r0  |T26: r1  |T27: r0  |T27: r1  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+
    # |T28: r0  |T28: r1  |T29: r0  |T29: r1  ||T30: r0  |T30: r1  |T31: r0  |T31: r1  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+ -----
    # +---------+---------+---------+---------++---------+---------+---------+---------+ -----
    # | T0: r2  | T0: r3  | T1: r2  | T1: r3  || T2: r2  | T2: r3  | T3: r2  | T3: r3  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+
    # | T4: r2  | T4: r3  | T5: r2  | T5: r3  || T6: r2  | T6: r3  | T7: r2  | T7: r3  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+
    # | T8: r2  | T8: r3  | T9: r2  | T9: r3  ||T10: r2  |T10: r3  |T11: r2  |T11: r3  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+
    # |T12: r2  |T12: r3  |T13: r2  |T13: r3  ||T14: r2  |T14: r3  |T15: r2  |T15: r3  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+ 8 rows
    # |T16: r2  |T16: r3  |T17: r2  |T17: r3  ||T18: r2  |T18: r3  |T19: r2  |T19: r3  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+
    # |T20: r2  |T20: r3  |T21: r2  |T21: r3  ||T22: r2  |T22: r3  |T23: r2  |T23: r3  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+
    # |T24: r2  |T24: r3  |T25: r2  |T25: r3  ||T26: r2  |T26: r3  |T27: r2  |T27: r3  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+
    # |T28: r2  |T28: r3  |T29: r2  |T29: r3  ||T30: r2  |T30: r3  |T31: r2  |T31: r3  |
    # +---------+---------+---------+---------++---------+---------+---------+---------+ -----
    # (T: thread index, r: register index)

    # Hence, the data is contiguous in SRAM like:
    # | T0: r0  | T0: r1  | T1: r0  | T1: r1  || T4: r0  | T4: r1  | T5: r0  | T5: r1  |
    # ...
    # | T2: r0  | T2: r1  | T3: r0  | T3: r1  || T6: r0  | T6: r1  | T7: r0  | T7: r1  |
    # ...
    # | T0: r2  | T0: r3  | T1: r2  | T1: r3  || T4: r2  | T4: r3  | T5: r2  | T5: r3  |
    # ...
    # | T2: r2  | T2: r3  | T3: r2  | T3: r3  || T6: r2  | T6: r3  | T7: r2  | T7: r3  |
    # ...

    # Spread data to the 4 quadrants accordingly, such that each thread will have
    # [thread_idx.x + 0, ..., thread_idx.x + 3] in it's registers after the `tcgen05.ld`.

    var n = (thread_idx.x // 2) % 2 * 4 + (thread_idx.x // 8)
    var k = (thread_idx.x // 4) % 2 * 4 + (thread_idx.x % 2) * 2

    smem_tile[n, k + 0] = Float32(thread_idx.x * 4 + 0)
    smem_tile[n, k + 1] = Float32(thread_idx.x * 4 + 1)
    smem_tile[n + 8, k + 0] = Float32(thread_idx.x * 4 + 2)
    smem_tile[n + 8, k + 1] = Float32(thread_idx.x * 4 + 3)

    var elect_one_warp = warp_id() == 0

    var ptr_tmem_addr = unsafe_stack_allocation[
        1, UInt32, address_space=.SHARED, alignment=16
    ]()

    comptime width = N
    comptime num_cols = 32
    comptime bits = 256

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, num_cols)

    barrier()

    # For debugging SRAM data layout:
    # print(smem_tile)

    var tmem_addr = ptr_tmem_addr[0]

    tcgen05_cp[cta_group=1, datapaths=128, bits=bits](tmem_addr, s_desc)

    var data_ld = tcgen05_ld[
        datapaths=16,
        bits=bits,
        repeat=1,
        dtype=DType.float32,
        pack=False,
        width=width,
    ](tmem_addr)

    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, num_cols)

    for n in range(N):
        if data_ld[n] == Float32(thread_idx.x * N + n):
            data[thread_idx.x, n] = data_ld[n]


def test_tcgen05_cp_ld_roundtrip(ctx: DeviceContext) raises:
    comptime M = 32
    comptime N = 4
    var data = ManagedLayoutTensor[
        .float32,
        Layout.row_major(M, N),
    ](ctx)
    comptime kernel = tcgen05_cp_ld_roundtrip_kernel[M, N]
    ctx.enqueue_function[kernel](
        data.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(M),
    )
    ctx.synchronize()
    var data_host = data.tensor()
    for m in range(M):
        for n in range(N):
            assert_almost_equal(
                data_host[m, n],
                Float32(m * N + n),
                atol=1e-3,
                rtol=1e-4,
            )


def main() raises:
    with DeviceContext() as ctx:
        test_tcgen05_st_ld_roundtrip(ctx)
        test_tcgen05_cp_ld_roundtrip(ctx)
