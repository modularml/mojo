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

from std.sys import size_of

from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import block_idx, grid_dim, thread_idx
from layout import IntTuple, Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.swizzle import make_swizzle
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tensor_tile,
)
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal

from std.utils.index import Index, IndexList


# Test loading a single 2d tile.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def test_tma_3d_load_kernel[
    dtype: DType,
    dst_layout: Layout,
    tile_rank: Int,
    cta_tile_shape: IndexList[tile_rank],
    desc_shape: IndexList[tile_rank],
    smem_layout: Layout,
](
    dst: LayoutTensor[dtype, dst_layout, MutAnyOrigin],
    tma_tile: TMATensorTile[dtype, tile_rank, cta_tile_shape, desc_shape],
):
    comptime assert (
        _idx_product[tile_rank, cta_tile_shape]() == smem_layout.size()
    ), "CTA Tile and SMEM tile should be the same size"

    comptime dst_dim0 = dst_layout.shape[0].value()
    comptime dst_dim1 = dst_layout.shape[1].value()

    comptime cta_tile_dim0 = cta_tile_shape[0]
    comptime cta_tile_dim1 = cta_tile_shape[1]
    comptime cta_tile_dim2 = cta_tile_shape[2]

    comptime assert (
        dst_dim1 == cta_tile_dim2
    ), "dst and cta should have the same last dimension for these test cases"

    var smem_tile = LayoutTensor[
        dtype,
        smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    comptime expected_bytes = _idx_product[
        tile_rank, cta_tile_shape
    ]() * size_of[dtype]()

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()
        mbar[0].expect_bytes(Int32(expected_bytes))
        tma_tile.async_copy_3d(
            smem_tile,
            mbar[0],
            (
                block_idx.x * cta_tile_dim2,
                block_idx.y * cta_tile_dim1,
                block_idx.z * cta_tile_dim0,
            ),
        )
    # Ensure all threads sees initialized mbarrier
    barrier()
    mbar[0].wait()

    comptime smem_dim0 = smem_layout.shape[0].value()
    comptime smem_dim1 = smem_layout.shape[1].value()
    comptime smem_dim2 = smem_layout.shape[2].value()

    var idx = (
        block_idx.z * grid_dim.y + block_idx.y
    ) * grid_dim.x + block_idx.x
    for i in range(cta_tile_dim0):
        var smem_tile_i = smem_tile.tile[1, cta_tile_dim1, cta_tile_dim2](i)

        var dst_tile = dst.tile[cta_tile_dim1, cta_tile_dim2](
            idx * cta_tile_dim0 + i, 0
        )
        if thread_idx.x == 0:
            dst_tile.copy_from(smem_tile_i)


def test_tma_3d_load_row_major[
    dtype: DType,
    src_layout: Layout,
    cta_tile_layout: Layout,
    smem_tile_layout: Layout,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext) raises:
    print("test_tma_3d_load")

    comptime src_dim0 = src_layout.shape[0].value()
    comptime src_dim1 = src_layout.shape[1].value()
    comptime src_dim2 = src_layout.shape[2].value()

    comptime cta_tile_dim0 = cta_tile_layout.shape[0].value()
    comptime cta_tile_dim1 = cta_tile_layout.shape[1].value()
    comptime cta_tile_dim2 = cta_tile_layout.shape[2].value()

    comptime dst_layout = Layout.row_major(
        src_dim0 * src_dim1 * src_dim2 // cta_tile_dim2, cta_tile_dim2
    )

    var src = ManagedLayoutTensor[dtype, src_layout](ctx)
    var dst = ManagedLayoutTensor[dtype, dst_layout](ctx)

    arange(src.tensor(), 1)

    var tma_tensor = create_tensor_tile[
        Index(cta_tile_dim0, cta_tile_dim1, cta_tile_dim2),
        swizzle_mode=swizzle_mode,
    ](ctx, src.device_tensor())

    ctx.synchronize()

    print("src layout:", materialize[src_layout]())
    print("cta tile layout:", materialize[cta_tile_layout]())
    print("desc shape:", type_of(tma_tensor).desc_shape)

    comptime kernel = test_tma_3d_load_kernel[
        type_of(tma_tensor).dtype,
        dst_layout,  # dst layout
        type_of(tma_tensor).rank,
        type_of(tma_tensor).tile_shape,  # cta_tile
        type_of(tma_tensor).desc_shape,  # desc_tile
        smem_tile_layout,  # smem layout
    ]
    ctx.enqueue_function[kernel](
        dst.device_tensor(),
        tma_tensor,
        grid_dim=(
            src_dim2 // cta_tile_dim2,
            src_dim1 // cta_tile_dim1,
            src_dim0 // cta_tile_dim0,
        ),
        block_dim=(1),
    )

    var src_host = src.tensor()
    var dst_host = dst.tensor()

    comptime swizzle = make_swizzle[dtype, swizzle_mode]()

    comptime cta_tile_size = cta_tile_layout.size()

    comptime desc_tile_dim0 = type_of(tma_tensor).desc_shape[0]
    comptime desc_tile_dim1 = type_of(tma_tensor).desc_shape[1]
    comptime desc_tile_dim2 = type_of(tma_tensor).desc_shape[2]

    comptime desc_tile_size = desc_tile_dim1 * desc_tile_dim2

    var desc_tile = LayoutTensor[
        dtype,
        Layout.row_major(desc_tile_dim1, desc_tile_dim2),
        MutAnyOrigin,
    ].stack_allocation()

    var dest_ptr = dst_host.ptr
    for dest_tile_z in range(src_dim0 // cta_tile_dim0):
        for dest_tile_y in range(src_dim1 // cta_tile_dim1):
            for dest_tile_x in range(src_dim2 // cta_tile_dim2):
                for x in range(cta_tile_dim2 // desc_tile_dim2):
                    for y in range(cta_tile_dim1 // desc_tile_dim1):
                        for z in range(cta_tile_dim0):
                            var src_tile = src_host.tile[
                                1, desc_tile_dim1, desc_tile_dim2
                            ](
                                dest_tile_z * cta_tile_dim0 + z,
                                dest_tile_y + y,
                                dest_tile_x + x,
                            )

                            desc_tile.copy_from(src_tile)

                            for i in range(desc_tile_size):
                                var desc_idx = swizzle(i)
                                assert_equal(
                                    desc_tile.ptr[desc_idx], dest_ptr[i]
                                )

                            dest_ptr += desc_tile_size

    _ = src^
    _ = dst^


def main() raises:
    with DeviceContext() as ctx:
        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 8, 8),
                IntTuple(64, 8, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(1, 8, 8),
                IntTuple(64, 8, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(1, 8, 8),
                IntTuple(64, 8, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 16, 8),
                IntTuple(128, 8, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 8, 8),
                IntTuple(64, 8, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 8, 8),
                IntTuple(64, 8, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 32, 8),
                IntTuple(256, 8, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 32, 8),
                IntTuple(256, 8, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 32, 8),
                IntTuple(256, 8, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 16, 16),
                IntTuple(256, 16, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 8, 8),
                IntTuple(64, 8, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 8, 8),
                IntTuple(64, 8, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 16, 16),
                IntTuple(256, 16, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 8, 16),
                IntTuple(128, 16, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 8, 16),
                IntTuple(128, 16, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 8, 16),
                IntTuple(128, 16, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 8, 16),
                IntTuple(128, 16, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 8, 16),
                IntTuple(128, 16, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 16, 64),
                IntTuple(1024, 64, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 16, 64),
                IntTuple(1024, 64, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 16, 64),
                IntTuple(1024, 64, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 16, 64),
                IntTuple(1024, 64, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 8, 64),
                IntTuple(512, 64, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 8, 64),
                IntTuple(512, 64, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 32, 64),
                IntTuple(2048, 64, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 32, 64),
                IntTuple(2048, 64, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 32, 64),
                IntTuple(2048, 64, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 8, 128),
                IntTuple(1024, 128, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 8, 128),
                IntTuple(1024, 128, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 8, 128),
                IntTuple(1024, 128, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 16, 32),
                IntTuple(512, 32, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 16, 32),
                IntTuple(512, 32, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 16, 32),
                IntTuple(512, 32, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 16, 32),
                IntTuple(512, 32, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 8, 32),
                IntTuple(256, 32, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 8, 32),
                IntTuple(256, 32, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 32, 32),
                IntTuple(1024, 32, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 32, 32),
                IntTuple(1024, 32, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 32, 32),
                IntTuple(1024, 32, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 8, 64),
                IntTuple(512, 64, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 8, 64),
                IntTuple(512, 64, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 8, 64),
                IntTuple(512, 64, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 16, 16),
                IntTuple(256, 16, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 16, 16),
                IntTuple(256, 16, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 16, 16),
                IntTuple(256, 16, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 16, 16),
                IntTuple(256, 16, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 8, 16),
                IntTuple(128, 16, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 8, 16),
                IntTuple(128, 16, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 32, 16),
                IntTuple(512, 16, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 32, 16),
                IntTuple(512, 16, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 32, 16),
                IntTuple(512, 16, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(4, 8, 32),
                IntTuple(256, 32, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(2, 8, 32),
                IntTuple(256, 32, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(2, 8, 32),
                IntTuple(256, 32, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)
