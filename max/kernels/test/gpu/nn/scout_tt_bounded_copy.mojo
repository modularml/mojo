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
"""Prove the explicit-bound masked TileTensor DRAM->SRAM copy.

Builds a STATIC-row Q-like GMEM TileTensor block, sub-tiles via `.tile[BM,BK]`
(proven to vectorize), and drives a masked `copy_dram_to_sram_async` with an
explicit `src_num_valid_rows` bound. Verifies SMEM rows [0, valid) == data and
rows [valid, BM) == 0 (the partial-tile zero-fill the LT path produces).
"""

from layout import TensorLayout, TileTensor
from layout.tile_tensor import stack_allocation as tt_stack_alloc
from layout.tile_layout import (
    Layout as InternalLayout,
    row_major as tt_row_major,
)
from layout.tile_io import copy_dram_to_sram_async
from layout.coord import ComptimeInt, Coord, Idx
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)
from std.utils.index import IndexList
from std.sys import simd_width_of, size_of


def _make_view[
    dtype: DType,
    ResultLayout: TensorLayout,
    rank: Int,
    origin: Origin,
    address_space: AddressSpace,
](
    ptr: Pointer[Scalar[dtype], origin, address_space=address_space],
    shape: IndexList[rank],
    strides: IndexList[rank],
) -> TileTensor[
    dtype,
    InternalLayout[
        shape_types=ResultLayout._shape_types,
        stride_types=ResultLayout._stride_types,
    ],
    ImmutAnyOrigin,
    linear_idx_type=.int64,
]:
    var immut_ptr = (
        ptr.address_space_cast[.GENERIC]().as_imm().as_unsafe_any_origin()
    )
    comptime ConcLayout = InternalLayout[
        shape_types=ResultLayout._shape_types,
        stride_types=ResultLayout._stride_types,
    ]
    var shape_c = Coord[*ConcLayout.shape_types]()
    var stride_c = Coord[*ConcLayout.stride_types]()
    comptime for i in range(rank):
        comptime if not shape_c.element_types[i].is_static_value:
            shape_c[i] = rebind[shape_c.element_types[i]](Int64(shape[i]))
        comptime if not stride_c.element_types[i].is_static_value:
            stride_c[i] = rebind[stride_c.element_types[i]](Int64(strides[i]))
    return TileTensor[
        dtype, ConcLayout, ImmutAnyOrigin, linear_idx_type=.int64
    ](ptr=immut_ptr, layout=ConcLayout(shape_c, stride_c))


comptime depth = 64
comptime num_heads = 2  # non-contiguous row stride = num_heads*depth
comptime BM = 16
comptime BK = 32
comptime dtype = DType.float32
comptime simd_size = simd_width_of[dtype]()
comptime num_threads = 32

# STATIC-row Q-like GMEM layout (row=BM static, depth static) with a
# NON-contiguous row stride (num_heads*depth) — matches mha's q_gmem_layout.
# The masked copy never reads past the explicit bound, so static rows >= valid
# are safe even though those GMEM rows are out of the valid sequence range.
# Direct TileTensor layout: static rows/depth with a non-contiguous row
# stride (num_heads*depth) — matches mha's q_gmem_layout.
comptime QTTLayout = InternalLayout[
    shape_types=Coord[ComptimeInt[BM], ComptimeInt[depth]].element_types,
    stride_types=Coord[
        ComptimeInt[num_heads * depth], ComptimeInt[1]
    ].element_types,
]

# Thread layout matches the mha async_copy_q_layout shape:
#   rows = min(num_threads, BM*BK//simd) * simd // BK ; cols = BK//simd
comptime q_num_vecs = BM * BK // simd_size
comptime copy_layout = tt_row_major[
    min(num_threads, q_num_vecs) * simd_size // BK, BK // simd_size
]()


@__name(t"scout_bounded")
def scout_bounded(
    q_tt: TileTensor[dtype, QTTLayout, ImmutAnyOrigin, linear_idx_type=.int64],
    out_buf: MutPointer[Scalar[dtype], MutAnyOrigin],
    valid_rows_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var valid_rows = Int(valid_rows_dev)
    # SMEM dst: [BM, depth] row-major.
    var smem = external_memory[
        Scalar[dtype], address_space=.SHARED, alignment=16
    ]()
    var q_smem = TileTensor(smem, tt_row_major[BM, depth]())

    # Sub-tile + bounded masked copy over depth//BK column tiles (NO swizzle so
    # the dst layout is the trivial row-major one we read back directly).
    comptime for q_id in range(depth // BK):
        var dst_tile = q_smem.tile[BM, BK](Coord(Idx[0], Idx[q_id]))
        var src_tile = q_tt.tile[BM, BK](Coord(Idx[0], Idx[q_id]))
        copy_dram_to_sram_async[
            thread_layout=copy_layout,
            swizzle=None,
            masked=True,
            num_threads=num_threads,
        ](
            dst_tile.vectorize[1, simd_size](),
            src_tile.vectorize[1, simd_size](),
            valid_rows,
        )
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()

    # Dump SMEM to out_buf (one thread).
    if thread_idx.x == 0:
        for r in range(BM):
            for c in range(depth):
                out_buf[r * depth + c] = q_smem[r, c]


def main() raises:
    with DeviceContext() as ctx:
        # GMEM holds num_heads interleaved; row stride = num_heads*depth.
        comptime gmem_total = BM * num_heads * depth
        var gmem = ctx.enqueue_create_buffer[dtype](gmem_total)
        with gmem.map_to_host() as h:
            for i in range(gmem_total):
                h[i] = Float32(i + 1)  # non-zero so zero-fill is visible

        comptime out_total = BM * depth  # SMEM dump is contiguous [BM, depth]
        var out = ctx.enqueue_create_buffer[dtype](out_total)
        with out.map_to_host() as h:
            for i in range(out_total):
                h[i] = Float32(-999)

        var valid_rows = 14  # < BM=16; rows 14,15 must be zero-filled
        var q_view = _make_view[dtype, QTTLayout, 2](
            gmem.unsafe_ptr(),
            IndexList[2](BM, depth),
            IndexList[2](num_heads * depth, 1),
        )
        ctx.enqueue_function[scout_bounded](
            q_view,
            out.unsafe_ptr(),
            Int32(valid_rows),
            grid_dim=1,
            block_dim=num_threads,
            shared_mem_bytes=BM * depth * size_of[dtype](),
        )
        ctx.synchronize()

        var ok = True
        with out.map_to_host() as h:
            for r in range(BM):
                for c in range(depth):
                    var got = Float32(h[r * depth + c])
                    # source physical index for logical (r, c):
                    var phys = r * (num_heads * depth) + c
                    var want = Float32(phys + 1) if r < valid_rows else Float32(
                        0
                    )
                    if got != want:
                        if ok:
                            print(
                                "MISMATCH r=",
                                r,
                                "c=",
                                c,
                                "got=",
                                got,
                                "want=",
                                want,
                            )
                        ok = False
        if ok:
            print("BOUNDED COPY OK: rows [0,14) data, rows [14,16) zero-filled")
        else:
            raise Error("BOUNDED COPY FAILED")
