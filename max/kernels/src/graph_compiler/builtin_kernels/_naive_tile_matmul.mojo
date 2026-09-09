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
"""Transitional tile-codegen scaffolding: the naive `mo.matmul` tile driver.

This file is deliberately separate from `linalg.mojo` so the tile-programming-
model driver (and the GPU/tile-only imports it needs) stays isolated from the
main linalg kernels: `linalg.mojo` just imports `_NaiveMatmulTileAdapter`.

This is transitional scaffolding for bringing up the tile-based codegen path,
not the intended end state. It (and the `tile_based_fusion` plumbing it rides
on) is expected to be removed once every kernel for the target device has been
ported to `TileTensor` and the tile path is productionized (see GEX-3919).
"""

import extensibility

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu
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
from layout.tile_io import (
    GenericToLocalTileCopier,
    LocalToGenericTileCopier,
)
from std.collections import OptionalReg
from std.utils import IndexList
from linalg.utils import (
    NullTileConsumer,
    NullTileOperation,
    TileConsumer,
    TileOperation,
    is_valid_epilogue,
)
from extensibility import (
    ComputeOutputFusionTile,
    InputTensor,
    _FusedComputeOutputTileTensor,
    get_kernel_tile_shape,
)


@fieldwise_init
struct _ComputeFusionTileOp[
    FusionType: ComputeOutputFusionTile,
    //,
](TileOperation):
    """`TileOperation` adapter over `mo.matmul`'s output compute-fusion epilogue.

    Holds the fused-output `ComputeOutputFusionTile` struct and forwards the
    per-tile epilogue to its `compute`, exactly as `_ElementwiseFusionTileAdapter`
    drives `elem.compute`. This lets `_NaiveMatmulTileAdapter` hold a plain
    `TileTensor` for `c` (used only for the raw store) while the fusion lives
    behind the `TileOperation` boundary. Holds only the `TrivialRegisterPassable`
    fusion struct, so it rides into the kernel as a by-value capture.

    Transitional scaffolding for the tile-codegen bring-up (see GEX-3919).
    """

    # TODO(jtodd): what is the right address space in this case?
    comptime src_address_space = AddressSpace.GENERIC

    var fusion: Self.FusionType

    @always_inline
    def __call__[
        dtype: DType,
        LayoutType: TensorLayout,
    ](
        mut self,
        tile_coord: Coord,
        tile: TileTensor[
            dtype,
            LayoutType,
            MutAnyOrigin,
            ...,
            address_space=Self.src_address_space,
        ],
        thread_layout: _NewLayout,
    ) -> type_of(tile):
        # The kernel-supplied thread layout must be fully static so we can build
        # a comptime-parameterized load copier from it; the fusion uses that
        # copier to pull its own inputs (e.g. the broadcast bias) into local.
        comptime assert type_of(
            thread_layout
        ).all_dims_known, "TileOperation thread_layout must be statically known"
        comptime assert tile_coord.is_flat, "Tile coordinate must be flat."
        var copier = GenericToLocalTileCopier[type_of(thread_layout)()]()
        # `tile_coord` is an absolute element coord (the trait contract); the GC
        # fusion's `compute` expects a tile/grid index, so divide by the tile
        # shape (== `thread_layout`: one thread per element in this driver).
        # TODO(KERN-2893, GEX-3919): choose a convention and stick to it
        comptime TM = type_of(thread_layout).static_shape[0]
        comptime TN = type_of(thread_layout).static_shape[1]
        var elem = rebind[IndexList[2]](coord_to_index_list(tile_coord))
        var ridx = IndexList[2](elem[0] // TM, elem[1] // TN)
        # TODO(GEX-3973): Avoid these rebinds by being generic in the ManagedTensorSlice fusion methods
        return rebind[type_of(tile)](
            self.fusion.compute[dtype, 2, LayoutType, type_of(copier)](
                ridx,
                copier,
                rebind[
                    TileTensor[
                        dtype,
                        LayoutType,
                        MutAnyOrigin,
                        address_space=Self.src_address_space,
                    ]
                ](tile),
            )
        )


@fieldwise_init
struct _TileStoreConsumer[
    c_dtype: DType,
    CLayout: TensorLayout,
](TileConsumer):
    """Minimal terminal `TileConsumer` that performs the output store.

    A `TileConsumer` is terminal: when one is bound it OWNS the store, so the
    parent kernel skips its own store path. This consumer copies the per-thread
    fragment into the global output tile via `LocalToGenericTileCopier`, exactly
    as `_NaiveMatmulTileAdapter`'s built-in store would. It is not wired to any
    `mo.*` op; it exists solely to exercise the `TileConsumer` trait end-to-end
    and is injected only behind a `comptime` toggle in `MatmulTileSM100`.

    Transitional scaffolding for the tile-codegen bring-up (see GEX-3919).
    """

    comptime src_address_space = AddressSpace.LOCAL

    var c: TileTensor[Self.c_dtype, Self.CLayout, MutUntrackedOrigin]

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
        comptime TM = LayoutType.static_shape[0]
        comptime TN = LayoutType.static_shape[1]
        var elem = rebind[IndexList[2]](coord_to_index_list(tile_coord))
        var out_tile = self.c.tile[TM, TN](Coord(elem[0] // TM, elem[1] // TN))
        LocalToGenericTileCopier[type_of(thread_layout)()]().copy(
            out_tile, tile
        )


@fieldwise_init
struct _NaiveMatmulTileAdapter[
    c_dtype: DType,
    CLayout: TensorLayout,
    a_dtype: DType,
    ALayout: TensorLayout,
    b_dtype: DType,
    BLayout: TensorLayout,
    //,
    tile_shape: IndexList[2],
    TileOperationType: TileOperation = NullTileOperation,
    TileConsumerType: TileConsumer = NullTileConsumer,
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    """Naive per-tile matmul driver for the tile programming model (GPU).

    SM100 (B200) / correctness-only. Structurally the tile twin of
    `_ElementwiseFusionTileAdapter` in `builtin_primitives/primitives.mojo`:
    one thread-block computes one `tile_shape` output tile, `thread_layout ==
    tile_shape` so each of the `TM * TN` threads owns a `1x1` fragment and
    computes exactly one output element `C[row, col] = sum_k A[row, k] *
    B[k, col]`. The per-thread fragment is routed through the injected
    `TileOperation` epilogue (matmul+add, case 5) and stored via the same store
    copier the elementwise driver uses. No tensor cores / SMEM tiling: this is
    the validation kernel for the tile-codegen path, not a perf kernel.

    Transitional scaffolding for the tile-codegen bring-up; expected to be
    removed once the tile path is productionized (see GEX-3919).
    """

    comptime thread_layout = row_major(
        Idx[Self.tile_shape[0]], Idx[Self.tile_shape[1]]
    )
    comptime frag_layout = type_of(row_major[1, 1]())

    var c: TileTensor[Self.c_dtype, Self.CLayout, MutUntrackedOrigin]
    var a: TileTensor[Self.a_dtype, Self.ALayout, MutUntrackedOrigin]
    var b: TileTensor[Self.b_dtype, Self.BLayout, MutUntrackedOrigin]
    var k_dim: Int
    # Optional epilogues, both defaulting to their null sentinels (absent). The
    # non-terminal `tile_operation` transforms the fragment; a terminal
    # `tile_consumer`, if present, owns the store.
    # Here we use OptionalReg, but elsewhere Optional, due to capture/passing
    # convention differences.
    var tile_operation: OptionalReg[Self.TileOperationType]
    var tile_consumer: OptionalReg[Self.TileConsumerType]

    @always_inline
    def __call__(self) capturing:
        comptime TM = Self.tile_shape[0]
        comptime TN = Self.tile_shape[1]

        comptime assert not (
            is_valid_epilogue[Self.TileConsumerType]()
            and is_valid_epilogue[Self.TileOperationType]()
        ), (
            "It is not permitted to pass both TileOperation and TileConsumer"
            " epilogues"
        )
        # One block per output tile; `thread_idx.x` selects this thread's
        # element within the tile (row-major over `thread_layout`).
        var tile_row = Int(block_idx.y)
        var tile_col = Int(block_idx.x)
        var tid = Int(thread_idx.x)
        var i = tid // TN
        var j = tid % TN
        var row = tile_row * TM + i
        var col = tile_col * TN + j

        # Naive dot product over K for this output element.
        var acc = Scalar[Self.c_dtype](0)
        for k in range(self.k_dim):
            acc += (
                self.a.load[width=1](Coord(row, k)).cast[Self.c_dtype]()
                * self.b.load[width=1](Coord(k, col)).cast[Self.c_dtype]()
            )

        # Driver-owned per-thread output fragment, allocated in LOCAL (so the
        # store copier's src space lines up) and viewed in the epilogue op's
        # declared `src_address_space` / `MutAnyOrigin` to match its tile type.
        # Mirrors the `dst` pattern in `_ElementwiseFusionTileAdapter`.
        #
        # TODO(GEX-3912): this LOCAL->GENERIC staging + the `address_space_cast`
        # back to LOCAL at the store below is a known limitation of the current
        # tile compute interface, which pins its tile argument to GENERIC and
        # returns a fresh tile. Generalizing that interface (an address-space-
        # general tile argument so the fragment can stay LOCAL throughout, plus
        # an `inout` `dst` so the passed-in tile is the output buffer instead of
        # a layout carrier) would remove this dance entirely.
        var dst_local = stack_allocation[
            dtype=Self.c_dtype, address_space=.LOCAL
        ](row_major[1, 1]())
        # TODO(jtodd): comptime assert/where the epilogue address space
        var dst = TileTensor(
            dst_local._storage.address_space_cast[
                Self.TileOperationType.src_address_space
            ]().unsafe_origin_cast[MutAnyOrigin](),
            row_major[1, 1](),
        )
        dst.store[width=1](Coord(0, 0), acc)

        # Non-terminal `TileOperation` transform (matmul+add, case 5), if bound,
        # telling it the thread layout this call runs over (so it can build its
        # own load copier for aux inputs like the broadcast bias). Plain matmul
        # (case 4) leaves the fragment untouched.
        var res = dst
        comptime if is_valid_epilogue[Self.TileOperationType]():
            var epilogue_op = self.tile_operation.value()
            res = epilogue_op(
                Coord(tile_row * TM, tile_col * TN), dst, Self.thread_layout
            )

        # Terminal store. A bound `TileConsumer` OWNS the store (replacing the
        # kernel store path); otherwise the adapter stores the fragment itself.
        # The `address_space_cast` back to LOCAL is the store-side half of the
        # staging dance noted above (see TODO(GEX-3912)).
        var res_local = res.address_space_cast[.LOCAL]()
        comptime if is_valid_epilogue[Self.TileConsumerType]():
            var consumer = self.tile_consumer.value()
            consumer(
                Coord(tile_row * TM, tile_col * TN),
                res_local.address_space_cast[
                    Self.TileConsumerType.src_address_space
                ](),
                Self.thread_layout,
            )
        else:
            var out_tile = self.c.tile[TM, TN](Coord(tile_row, tile_col))
            LocalToGenericTileCopier[Self.thread_layout]().copy(
                out_tile, res_local
            )


# Transitional tile-codegen scaffolding (remove per GEX-3941): a SEPARATE,
# device-specific `mo.matmul` registration for sm_100a (B200). It declares the
# tile fused-output IOSpec (`_FusedComputeOutputTileTensor`); the generic
# `Matmul` in `linalg.mojo` stays SIMD-only. Registering the tile matmul as its
# own kernel (rather than a second `execute` overload on the generic struct)
# avoids the importer's single-`execute`-slot collapse. It must NOT become the
# canonical B200 matmul: the graph-compiler selection filter in
# `MojoRegistry::getResolvedKernel` discounts this tile candidate unless the
# target device has `tile_based_fusion` set, so a default B200 (flag off)
# resolves to the generic SIMD `mo.matmul` and only the tile-codegen path
# (flag on) selects this kernel. Expected to be removed once the real tile
# matmul lands and the tile-codegen transition completes (see GEX-3919).
@extensibility.register("mo.matmul", type="gpu", api="cuda", arch="sm_100a")
struct MatmulTileSM100:
    @staticmethod
    def execute[
        transpose_b: Bool,
        packed_b: Bool,
        has_epilogue_fusion: Bool,
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: _FusedComputeOutputTileTensor[rank=2, ...],
        a: InputTensor[rank=2, ...],
        b: InputTensor[rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # The tile programming model (`tile_based_fusion=True`) binds `c` to
        # `_FusedComputeOutputTile`, and the selection filter routes the op to
        # this sm_100a registration. Naive, correctness-only GPU matmul on MAX's
        # own Mojo kernel (no vendor BLAS); see `_NaiveMatmulTileAdapter`. Tested
        # at (M, N, K) = (256, 256, 256) f32 on B200.
        comptime assert is_gpu[target](), "tile-codegen matmul is GPU-only"
        comptime assert (
            not transpose_b
        ), "tile-codegen matmul only supports transpose_b=False"
        comptime assert (
            not packed_b
        ), "tile-codegen matmul does not support packed_b"

        # TODO(GEX-3905): a real matmul must own its own tile shape (tensor-core
        # tiling) rather than inheriting this shared elementwise
        # `get_kernel_tile_shape` default, and that shape must be threaded into
        # the fusion emitter so the epilogue conforms to it — instead of the
        # driver and emitter each independently reading `get_kernel_tile_shape`.
        comptime tile_shape = get_kernel_tile_shape[c.dtype, target]()
        comptime TM = tile_shape[0]
        comptime TN = tile_shape[1]

        var m_dim = Int(c.dim_size[0]())
        var n_dim = Int(c.dim_size[1]())
        var k_dim = Int(a.dim_size[1]())
        debug_assert(
            m_dim % TM == 0 and n_dim % TN == 0,
            "tile-codegen matmul requires tile-divisible output shapes",
        )

        # This `mo.matmul` function demonstrate 3 different options for
        # tile-based epilogues.
        # 1. via TileOperation, a 'compute' lambda (if the GC performed epilogue fusion)
        # 2. via TileConsumer, a trivial 'store only' epilogue (if demo_tile_consumer is True)
        # 3. no epilogue
        comptime demo_tile_consumer = False

        var c_tt = c.to_tile_tensor()
        var a_tt = a.to_tile_tensor()
        var b_tt = b.to_tile_tensor()

        # 1. GC-generated epilogue, wrapped in TileOperation
        comptime if has_epilogue_fusion:
            var op = _ComputeFusionTileOp(c.compute_fusion_tile)
            var adapter = _NaiveMatmulTileAdapter[
                tile_shape=tile_shape, TileOperationType=type_of(op)
            ](
                c_tt,
                a_tt,
                b_tt,
                k_dim,
                OptionalReg(op),
                OptionalReg[NullTileConsumer](),
            )
            ctx.enqueue_function(
                adapter,
                grid_dim=(n_dim // TN, m_dim // TM),
                block_dim=(TM * TN),
            )
        # 2. Trivial TileConsumer (just stores the data for demonstration purposes!)
        elif demo_tile_consumer:
            var cons = _TileStoreConsumer(c_tt)
            var adapter = _NaiveMatmulTileAdapter[
                tile_shape=tile_shape, TileConsumerType=type_of(cons)
            ](
                c_tt,
                a_tt,
                b_tt,
                k_dim,
                OptionalReg[NullTileOperation](),
                OptionalReg(cons),
            )
            ctx.enqueue_function(
                adapter,
                grid_dim=(n_dim // TN, m_dim // TM),
                block_dim=(TM * TN),
            )
        # 3. No fusion, no epilogue
        else:
            var adapter = _NaiveMatmulTileAdapter[tile_shape=tile_shape](
                c_tt,
                a_tt,
                b_tt,
                k_dim,
                OptionalReg[NullTileOperation](),
                OptionalReg[NullTileConsumer](),
            )
            ctx.enqueue_function(
                adapter,
                grid_dim=(n_dim // TN, m_dim // TM),
                block_dim=(TN * TM),
            )
