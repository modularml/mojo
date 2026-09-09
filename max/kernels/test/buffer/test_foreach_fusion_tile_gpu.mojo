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
"""GPU integration test for the tile `foreach_fusion_tile` driver.

Defines a concrete `ElementwiseFusionTile` whose `compute` loads two captured
input tiles through the supplied copier and returns their sum, then runs it
through `foreach_fusion_tile` over a device output tensor on the GPU (validated
on B200) and compares against a CPU `out = a + b` reference.
"""

from builtin_kernels.elementwise import Add
from builtin_primitives.primitives import foreach_fusion_tile

from extensibility import (
    ElementwiseFusionTile,
    IOSpec,
    ManagedTensorSlice,
    get_row_major_tensor_spec_static,
)

from layout import ComptimeInt, Coord, RowMajorLayout, TileTensor, row_major
from layout.tile_io import TileCopier
from layout.tile_layout import TensorLayout

from std.memory import unsafe_stack_allocation as raw_stack_allocation

from max.gpu.host import DeviceContext
from std.testing import assert_equal
from std.utils.index import IndexList

comptime M = 32
comptime N = 32
comptime TM = 16
comptime TN = 16
comptime NUM = M * N

comptime FullLayout = RowMajorLayout[ComptimeInt[M], ComptimeInt[N]]


# A concrete `ElementwiseFusionTile`: captures two full input tiles and
# manufactures `a + b` for the requested output tile.
@fieldwise_init
struct AddFusionTile(ElementwiseFusionTile):
    var a: TileTensor[.float32, FullLayout, MutUntrackedOrigin]
    var b: TileTensor[.float32, FullLayout, MutUntrackedOrigin]

    @always_inline
    def compute[
        dtype: DType,
        rank: Int,
        LayoutType: TensorLayout,
        Copier: TileCopier,
    ](
        self,
        tile_coords: IndexList[rank],
        copier: Copier,
        dst: TileTensor[dtype, LayoutType, MutAnyOrigin],
    ) -> TileTensor[dtype, LayoutType, MutAnyOrigin]:
        var tc = Coord(Int(tile_coords[0]), Int(tile_coords[1]))
        var a_tile = self.a.tile[TM, TN](tc)
        var b_tile = self.b.tile[TM, TN](tc)

        # `dst` carries the concrete fragment layout, so `dst.layout` sizes the
        # staging tile for the second operand (no opaque-`LayoutType` /
        # `rebind` dance). The trait's `compute` is unconstrained, so the
        # layout-typed `stack_allocation` (which needs `all_dims_known`) is not
        # provable here; allocate raw bytes sized by `LayoutType.static_product`
        # and wrap them with `dst.layout`. Load the first input straight into
        # `dst`, the second into the matching staging fragment, both in the
        # copier's dst space.
        var rhs_buf = raw_stack_allocation[
            LayoutType.static_product,
            dtype,
            address_space=Copier.dst_address_space,
        ]()
        var rhs = TileTensor(rhs_buf, dst.layout)
        copier.copy(
            dst.address_space_cast[Copier.dst_address_space](),
            a_tile.address_space_cast[Copier.src_address_space](),
        )
        copier.copy(rhs, b_tile.address_space_cast[Copier.src_address_space]())

        # `Add.elementwise` operates on GENERIC / `MutAnyOrigin` tiles; `dst` is
        # already one, so only `rhs` needs a generic view. Add writes into `dst`
        # in place and returns it.
        var rhs_g = TileTensor(
            rhs._storage.address_space_cast[.GENERIC]().unsafe_origin_cast[
                MutAnyOrigin
            ](),
            dst.layout,
        )
        return Add.elementwise[dtype, LayoutType](dst, rhs_g)


def test_foreach_fusion_tile_add(ctx: DeviceContext) raises:
    """`foreach_fusion_tile` computes `out = a + b` tile-by-tile on GPU."""
    var a_host = ctx.enqueue_create_host_buffer[.float32](NUM)
    var b_host = ctx.enqueue_create_host_buffer[.float32](NUM)
    ctx.synchronize()
    for i in range(NUM):
        a_host[i] = Float32(i)
        b_host[i] = Float32(1000 + i)

    var a_dev = ctx.enqueue_create_buffer[.float32](NUM)
    var b_dev = ctx.enqueue_create_buffer[.float32](NUM)
    var out_dev = ctx.enqueue_create_buffer[.float32](NUM)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var a_tt = TileTensor(
        a_dev.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        row_major[M, N](),
    )
    var b_tt = TileTensor(
        b_dev.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        row_major[M, N](),
    )
    var elem = AddFusionTile(a_tt, b_tt)

    comptime spec = get_row_major_tensor_spec_static[.float32, 2, M, N]()
    var out_mts = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](out_dev.unsafe_ptr(), IndexList[2](M, N))

    foreach_fusion_tile[target="gpu", tile_shape=IndexList[2](TM, TN)](
        out_mts, elem, ctx
    )

    var out_host = ctx.enqueue_create_host_buffer[.float32](NUM)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(NUM):
        assert_equal(out_host[i], a_host[i] + b_host[i])


def main() raises:
    with DeviceContext() as ctx:
        test_foreach_fusion_tile_add(ctx)
    print("test_foreach_fusion_tile_gpu OK")
