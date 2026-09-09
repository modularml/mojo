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
#
# Experimental TMA store (shared -> global) with SWIZZLE_NONE for a 128x64
# uint32 matrix, driven by a single warp group (128 threads).
#
# Each thread materializes uint32 values directly in registers, packs them into
# SIMD[DType.uint32, 4] and writes them to a row-major shared-memory tile, then
# thread 0 issues a single TMA store of the whole tile to global memory.
#
# The register -> shared writes use the canonical non-swizzled layout
# ``((8,m),(4,2k)):((4,SBO),(1,LBO))``: each group of 8 threads writes 32
# contiguous uint32 (128 B == all 32 banks exactly once), so the 128-bit stores
# are bank-conflict-free without any swizzling. Concretely, for thread ``t`` at
# iteration ``it`` the write lands at row-major element offset
# ``it*512 + t*4`` (16 B aligned); within a warp threads 0-7 cover offsets
# ``it*512 + [0..31]``, threads 8-15 cover ``[32..63]``, and so on.
#
# Because TMA SWIZZLE_NONE maps the row-major shared tile to the row-major
# global tensor identically, the value stored at offset ``o`` must equal ``o``;
# element ``(i, j)`` sits at offset ``o = i*64 + j``, so writing ``value ==
# offset`` yields ``x[i, j] == i*64 + j`` -- which is exactly what the test
# checks.

from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import fence_async_view_proxy
from max.gpu.sync import cp_async_bulk_commit_group, cp_async_bulk_wait_group
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.tma_async import TMATensorTile, create_tma_tile
from std.testing import assert_equal
from std.utils.index import IndexList

comptime M = 128
comptime N = 64
comptime VEC = 4  # SIMD[.uint32, 4]
comptime THREADS = 128  # one warp group
comptime ELEMS_PER_ITER = THREADS * VEC  # 512 == 8 full rows
comptime ITERS = (M * N) // ELEMS_PER_ITER  # 8192 / 512 == 16


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def tma_store_uint32_kernel[
    tile_shape: IndexList[2],
    desc_shape: IndexList[2],
](tma_tile: TMATensorTile[.uint32, 2, tile_shape, desc_shape]):
    comptime smem_layout = Layout.row_major(M, N)
    var smem = LayoutTensor[
        .uint32,
        smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    var tid = thread_idx.x
    # Lane offsets 0,1,2,3 packed into the SIMD vector. value == row-major
    # offset, so each thread writes consecutive integers at its assigned
    # 4-element chunk.
    var iota = SIMD[.uint32, VEC](0, 1, 2, 3)

    for it in range(ITERS):
        # Conflict-free canonical (non-swizzled) placement: 8 consecutive
        # threads write 32 contiguous uint32 == one full sweep over all 32
        # banks.
        var base = UInt32(it * ELEMS_PER_ITER) + UInt32(tid) * UInt32(VEC)
        smem.ptr.store(Int(base), SIMD[.uint32, VEC](base) + iota)

    barrier()
    fence_async_view_proxy()

    if tid == 0:
        tma_tile.async_store(smem, (0, 0))
        cp_async_bulk_commit_group()

    cp_async_bulk_wait_group[0]()


def test_tma_swizzle_none_uint32_store(ctx: DeviceContext) raises:
    comptime layout = Layout.row_major(M, N)
    var dst = ManagedLayoutTensor[.uint32, layout](ctx)

    # Seed the destination with a distinct ramp so a missing/partial store
    # cannot pass on stale data.
    arange(dst.tensor(), 100001)

    var tma_tensor = create_tma_tile[
        M, N, swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE
    ](ctx, dst.device_tensor())

    ctx.synchronize()

    comptime kernel = tma_store_uint32_kernel[
        type_of(tma_tensor).tile_shape,
        type_of(tma_tensor).desc_shape,
    ]
    ctx.enqueue_function[kernel](
        tma_tensor,
        grid_dim=1,
        block_dim=THREADS,
    )

    ctx.synchronize()

    var dst_host = dst.tensor()
    for i in range(M):
        for j in range(N):
            assert_equal(dst_host[i, j], UInt32(i * N + j))

    _ = dst^


def main() raises:
    with DeviceContext() as ctx:
        print("test_tma_swizzle_none_uint32_store")
        test_tma_swizzle_none_uint32_store(ctx)
