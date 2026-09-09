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
"""Transitional tile-codegen scaffolding: the naive `mo.reduce.rms_norm` tile
driver, exercising the STORE variant of the tile-based epilogue.

Companion to `_naive_tile_matmul.mojo`, which exercises the COMPUTE variant
(`ComputeOutputFusionTile` / `TileOperation`, non-terminal). The graph compiler
registers one primary kernel per op, so a kernel author picks store or compute
fusion for that op; this file picks store (`OutputFusionTile` / `TileConsumer`,
terminal) for `mo.reduce.rms_norm`.

This is transitional scaffolding for bringing up the tile-based codegen path,
not the intended end state. It (and the `tile_based_fusion` plumbing it rides
on) is expected to be removed once every kernel for the target device has been
ported to `TileTensor` and the tile path is productionized (see GEX-3919).
"""

import extensibility

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu
from std.math import sqrt
from layout import (
    Coord,
    Idx,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout as _NewLayout
from layout.tile_io import LocalToGenericTileCopier, GenericToLocalTileCopier
from std.collections import OptionalReg
from std.utils import IndexList
from linalg.utils import (
    NullTileConsumer,
    TileConsumer,
    is_valid_epilogue,
)
from extensibility import (
    InputTensor,
    OutputFusionTile,
    _FusedOutputTileTensor,
    get_kernel_tile_shape,
)


@fieldwise_init
struct _OutputFusionTileConsumer[
    FusionType: OutputFusionTile,
    //,
](TileConsumer):
    """Terminal `TileConsumer` adapter over `mo.reduce.rms_norm`'s output
    store-fusion epilogue.

    The store analog of `_ComputeFusionTileOp` in `_naive_tile_matmul.mojo`:
    holds the fused-output `OutputFusionTile` struct and forwards the
    per-tile epilogue to its `store`. Unlike `_ComputeFusionTileOp` (whose
    `compute` transforms and returns the tile for the primary kernel to
    store), `store` OWNS the store, so once this consumer is bound the
    driver does nothing further with the fragment.

    Transitional scaffolding for the tile-codegen bring-up (see GEX-3919).
    """

    comptime src_address_space = AddressSpace.GENERIC

    var fusion: Self.FusionType

    @always_inline
    def __call__[
        dtype: DType,
        LayoutType: TensorLayout,
    ](
        ref self,
        tile_coord: Coord,
        tile: TileTensor[
            dtype, LayoutType, ..., address_space=Self.src_address_space
        ],
        thread_layout: _NewLayout,
    ) -> None:
        comptime assert type_of(
            thread_layout
        ).all_dims_known, "TileConsumer thread_layout must be statically known"
        comptime assert tile_coord.is_flat, "Tile coordinate must be flat."

        var copier = GenericToLocalTileCopier[type_of(thread_layout)()]()
        var store_copier = LocalToGenericTileCopier[type_of(thread_layout)()]()
        comptime TM = type_of(thread_layout).static_shape[0]
        comptime TN = type_of(thread_layout).static_shape[1]
        var elem = rebind[IndexList[2]](coord_to_index_list(tile_coord))
        var ridx = IndexList[2](elem[0] // TM, elem[1] // TN)
        self.fusion.store[
            dtype, 2, LayoutType, type_of(copier), type_of(store_copier)
        ](
            ridx,
            copier,
            store_copier,
            rebind[
                TileTensor[
                    dtype,
                    LayoutType,
                    MutAnyOrigin,
                    address_space=Self.src_address_space,
                ]
            ](tile),
        )


@fieldwise_init
struct _NaiveRMSNormTileAdapter[
    dtype: DType,
    OutLayout: TensorLayout,
    InLayout: TensorLayout,
    GammaLayout: TensorLayout,
    //,
    tile_shape: IndexList[2],
    TileConsumerType: TileConsumer = NullTileConsumer,
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    """Naive per-tile RMSNorm driver for the tile programming model (GPU).

    SM100 (B200) / correctness-only. One thread-block computes one
    `tile_shape` output tile, `thread_layout == tile_shape` so each of the
    `TM * TN` threads owns a `1x1` fragment and computes exactly one output
    element. The row-wise sum-of-squares reduction is over the FULL row
    (`n_dim`), independent of the output tile width `TN`: each thread
    redundantly walks its whole row to get `inv_rms`, rather than sharing the
    reduction across threads/blocks in that row.
    TODO(GEX-3919): O(n_dim) redundant work per element is a correctness-only
    shortcut; a real kernel would compute the row reduction once (e.g. one
    block-row pass with a shared-memory or warp reduction) and broadcast it.
    The per-thread fragment is routed through the injected `TileConsumer`
    epilogue (store fusion) when bound; otherwise the driver stores directly
    via the same store copier the matmul driver uses.

    Transitional scaffolding for the tile-codegen bring-up; expected to be
    removed once the tile path is productionized (see GEX-3919).
    """

    comptime thread_layout = row_major(
        Idx[Self.tile_shape[0]], Idx[Self.tile_shape[1]]
    )

    var output: TileTensor[Self.dtype, Self.OutLayout, MutUntrackedOrigin]
    var input: TileTensor[Self.dtype, Self.InLayout, MutUntrackedOrigin]
    var gamma: TileTensor[Self.dtype, Self.GammaLayout, MutUntrackedOrigin]
    var n_dim: Int
    var epsilon: Float32
    var weight_offset: Scalar[Self.dtype]
    var tile_consumer: OptionalReg[Self.TileConsumerType]

    @always_inline
    def __call__(self) capturing:
        comptime TM = Self.tile_shape[0]
        comptime TN = Self.tile_shape[1]

        var tile_row = Int(block_idx.y)
        var tile_col = Int(block_idx.x)
        var tid = Int(thread_idx.x)
        var i = tid // TN
        var j = tid % TN
        var row = tile_row * TM + i
        var col = tile_col * TN + j

        var sum_sq = Float32(0)
        for k in range(self.n_dim):
            var v = self.input.load[width=1](Coord(row, k)).cast[
                DType.float32
            ]()
            sum_sq += v * v
        var inv_rms = 1.0 / sqrt(sum_sq / Float32(self.n_dim) + self.epsilon)

        var x_val = self.input.load[width=1](Coord(row, col)).cast[
            DType.float32
        ]()
        var g_val = self.gamma.load[width=1](Coord(col)).cast[.float32]()
        var w_off = self.weight_offset.cast[.float32]()
        var y = (x_val * inv_rms) * (g_val + w_off)

        # Driver-owned per-thread output fragment; see `_NaiveMatmulTileAdapter`
        # for the LOCAL->GENERIC staging dance (TODO(GEX-3912)).
        var dst_local = stack_allocation[
            dtype=Self.dtype, address_space=.LOCAL
        ](row_major[1, 1]())
        var dst = TileTensor(
            dst_local._storage.address_space_cast[
                .GENERIC
            ]().unsafe_origin_cast[MutAnyOrigin](),
            row_major[1, 1](),
        )
        dst.store[width=1](Coord(0, 0), y.cast[Self.dtype]())

        # Terminal store. A bound `TileConsumer` OWNS the store (the fusion's
        # `store` performs the final `LocalToGenericTileCopier` copy itself);
        # otherwise the adapter stores the fragment directly.
        var dst_local_view = dst.address_space_cast[.LOCAL]()
        comptime if is_valid_epilogue[Self.TileConsumerType]():
            var consumer = self.tile_consumer.value()
            consumer(
                Coord(tile_row * TM, tile_col * TN),
                dst_local_view.address_space_cast[
                    Self.TileConsumerType.src_address_space
                ](),
                Self.thread_layout,
            )
        else:
            var out_tile = self.output.tile[TM, TN](Coord(tile_row, tile_col))
            LocalToGenericTileCopier[Self.thread_layout]().copy(
                out_tile, dst_local_view
            )


# Transitional tile-codegen scaffolding (remove per GEX-3941): a SEPARATE,
# device-specific `mo.reduce.rms_norm` registration for sm_100a (B200). It
# declares the tile fused-output IOSpec (`_FusedOutputTileTensor`); the
# generic SIMD `ReduceRMSNorm` in `reductions.mojo` stays untouched and keeps
# serving non-tile devices. Registering the tile kernel as its own struct
# (rather than a second `execute` overload) avoids the importer's single-
# `execute`-slot collapse. It must NOT become the canonical B200 rms_norm: the
# graph-compiler selection filter in `MojoRegistry::getResolvedKernel`
# discounts this tile candidate unless the target device has
# `tile_based_fusion` set, so a default B200 (flag off) resolves to the
# generic SIMD `mo.reduce.rms_norm` and only the tile-codegen path (flag on)
# selects this kernel. Expected to be removed once the tile path is
# productionized (see GEX-3919).
@extensibility.register(
    "mo.reduce.rms_norm", type="gpu", api="cuda", arch="sm_100a"
)
struct RMSNormTileSM100:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: _FusedOutputTileTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Float32,
        weight_offset: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        # The tile programming model (`tile_based_fusion=True`) binds `output`
        # to `_FusedOutputTile`, and the selection filter routes the op to
        # this sm_100a registration. Naive, correctness-only GPU rms_norm on
        # MAX's own Mojo kernel (no vendor BLAS); see
        # `_NaiveRMSNormTileAdapter`. Tested at (M, N) = (256, 256) f32 on
        # B200.
        comptime assert is_gpu[target](), "tile-codegen rms_norm is GPU-only"
        comptime assert (
            rank == 2
        ), "tile-codegen rms_norm only supports rank-2 tensors"
        comptime assert (
            dtype == .float32
        ), "tile-codegen rms_norm is correctness-only and f32-only"

        # TODO(GEX-3905): a real rms_norm must own its own tile shape rather
        # than inheriting this shared elementwise `get_kernel_tile_shape`
        # default, and that shape must be threaded into the fusion emitter so
        # the epilogue conforms to it -- instead of the driver and emitter
        # each independently reading `get_kernel_tile_shape`.
        comptime tile_shape = get_kernel_tile_shape[output.dtype, target]()
        comptime TM = tile_shape[0]
        comptime TN = tile_shape[1]

        var m_dim = Int(output.dim_size[0]())
        var n_dim = Int(output.dim_size[1]())
        debug_assert(
            m_dim % TM == 0 and n_dim % TN == 0,
            "tile-codegen rms_norm requires tile-divisible output shapes",
        )

        var out_tt = output.to_tile_tensor()
        var in_tt = input.to_tile_tensor()
        var gamma_tt = gamma.to_tile_tensor()

        comptime if output._has_output_fusion_tile:
            var cons = _OutputFusionTileConsumer(output.output_fusion_tile)
            var adapter = _NaiveRMSNormTileAdapter[
                tile_shape=tile_shape, TileConsumerType=type_of(cons)
            ](
                out_tt,
                in_tt,
                gamma_tt,
                n_dim,
                epsilon,
                weight_offset,
                OptionalReg(cons),
            )
            ctx.enqueue_function(
                adapter,
                grid_dim=(n_dim // TN, m_dim // TM),
                block_dim=(TM * TN),
            )
        else:
            var adapter = _NaiveRMSNormTileAdapter[tile_shape=tile_shape](
                out_tt,
                in_tt,
                gamma_tt,
                n_dim,
                epsilon,
                weight_offset,
                OptionalReg[NullTileConsumer](),
            )
            ctx.enqueue_function(
                adapter,
                grid_dim=(n_dim // TN, m_dim // TM),
                block_dim=(TM * TN),
            )
