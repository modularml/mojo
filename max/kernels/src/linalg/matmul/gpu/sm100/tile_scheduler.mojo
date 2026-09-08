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

"""Schedules output-tile work for persistent SM100 matmul kernels using Cluster Launch Control (CLC), rasterization order, and block swizzling."""

from std.sys import _RegisterPackType, size_of
from std.sys._assembly import inlined_assembly

from max.gpu.primitives.cluster import (
    clusterlaunchcontrol_try_cancel,
    elect_one_sync,
)
from max.gpu import block_id_in_cluster, block_idx, lane_id
from max.gpu.memory import fence_async_view_proxy
from layout.tma_async import PipelineState, SharedMemBarrier

from std.utils.fast_div import FastDiv
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from ..tile_scheduler import RasterOrder


@fieldwise_init
struct WorkInfo(TrivialRegisterPassable, Writable):
    """Holds the coordinates and validity of a single output tile assigned to a cluster for persistent matmul scheduling.
    """

    # Coordinates in output matrix
    var m: UInt32
    var n: UInt32
    # Starting k index in A and B for the output tile's mma.
    var k_start: UInt32
    # Whether work tile is completely OOB.
    var is_valid_tile: Bool

    @always_inline
    def is_valid(self) -> Bool:
        return self.is_valid_tile

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "(",
            self.m,
            ", ",
            self.n,
            ", ",
            self.k_start,
            ", ",
            self.is_valid_tile,
            ")",
        )


struct TileScheduler[
    num_stages: Int,
    cluster_shape: IndexList[3, element_type=.uint32] = Index[
        dtype=DType.uint32
    ](1, 1, 1),
    rasterize_order: RasterOrder = RasterOrder.AlongM,
    block_swizzle_size: Int = 8,
](TrivialRegisterPassable):
    """Schedules output-tile work across SM100 clusters for persistent matmul kernels using Cluster Launch Control (CLC).

    Applies a configurable rasterization order and optional block swizzling to map
    cluster-level work tiles to global output-tile coordinates, and drives
    producer/consumer synchronization through shared-memory barriers.

    Parameters:
        num_stages: Number of pipeline stages used for producer/consumer
            synchronization across waves; `0` means only the initial wave
            is valid and subsequent work is invalid.
        cluster_shape: Three-element `IndexList` giving the number of CTA
            blocks per cluster along the M, N, and K output-tile axes
            (defaults to `(1, 1, 1)`).
        rasterize_order: Direction the scheduler rasterizes clusters across
            output tiles, either along M or N (defaults to
            `RasterOrder.AlongM`).
        block_swizzle_size: Size of the block-swizzle group used to remap
            cluster coordinates for improved L2 reuse; one of `0`, `1`, `2`,
            `4`, or `8`, where `0` disables swizzling (defaults to `8`).
    """

    comptime cluster_size = Self.cluster_shape[0] * Self.cluster_shape[
        1
    ] * Self.cluster_shape[2]
    comptime log_cluster_m = FastDiv[.uint32](Self.cluster_shape[0])
    comptime log_cluster_n = FastDiv[.uint32](Self.cluster_shape[1])
    comptime log_cluster_k = FastDiv[.uint32](Self.cluster_shape[2])

    var cluster_dim: StaticTuple[Int32, 3]
    var log_cluster_dim_m: FastDiv[.uint32]
    var log_cluster_dim_n: FastDiv[.uint32]
    var log_cluster_dim_k: FastDiv[.uint32]

    @__allow_legacy_any_origin_fields
    var clc_response: UnsafePointer[
        UInt128, MutAnyOrigin, address_space=.SHARED
    ]

    @__allow_legacy_any_origin_fields
    var full_mbar: UnsafePointer[
        SharedMemBarrier, MutAnyOrigin, address_space=.SHARED
    ]

    @__allow_legacy_any_origin_fields
    var empty_mbar: UnsafePointer[
        SharedMemBarrier, MutAnyOrigin, address_space=.SHARED
    ]

    @always_inline
    def __init__(
        out self,
        cluster_dim: StaticTuple[Int32, 3],
        clc_response_ptr: UnsafePointer[
            mut=True, UInt128, _, address_space=.SHARED
        ],
        full_mbar_ptr: UnsafePointer[
            mut=True, SharedMemBarrier, _, address_space=.SHARED
        ],
        empty_mbar_ptr: UnsafePointer[
            mut=True, SharedMemBarrier, _, address_space=.SHARED
        ],
    ):
        comptime assert Self.block_swizzle_size in [
            0,
            1,
            2,
            4,
            8,
        ], "block_swizzle_size must be 0, 1, 2, 4, or 8"

        self.cluster_dim = cluster_dim
        self.log_cluster_dim_m = FastDiv[.uint32](Int(cluster_dim[0]))
        self.log_cluster_dim_n = FastDiv[.uint32](Int(cluster_dim[1]))
        self.log_cluster_dim_k = FastDiv[.uint32](Int(cluster_dim[2]))
        self.clc_response = clc_response_ptr.as_unsafe_any_origin()
        self.full_mbar = full_mbar_ptr.as_unsafe_any_origin()
        self.empty_mbar = empty_mbar_ptr.as_unsafe_any_origin()

    @always_inline
    @staticmethod
    def work_info_from_clc_response(
        result: UnsafePointer[mut=True, UInt128, _, address_space=.SHARED],
    ) -> WorkInfo:
        comptime asm = """{
            .reg .pred p1;
            .reg .b128 clc_result;
            ld.shared.b128 clc_result, [$4];
            clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;
            selp.u32 $3, 1, 0, p1;
            @p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {$0, $1, $2, _}, clc_result;
        }"""
        var ret_val = inlined_assembly[
            asm,
            _RegisterPackType[UInt32, UInt32, UInt32, UInt32],
            has_side_effect=True,
            constraints="=r,=r,=r,=r,r",
        ](UInt32(Int(result)))

        fence_async_view_proxy()

        return WorkInfo(
            m=ret_val[0],
            n=ret_val[1],
            k_start=ret_val[2],
            is_valid_tile=(ret_val[3] == 1),
        )

    @always_inline
    @staticmethod
    def work_info_from_cluster(
        work_info: WorkInfo,
        cluster_dim: StaticTuple[Int32, 3],
        log_cluster_dim_m: FastDiv[.uint32],
        log_cluster_dim_n: FastDiv[.uint32],
    ) -> WorkInfo:
        comptime FastUInt = Scalar[FastDiv[.uint32].uint_type]

        var normalized_m = FastUInt(work_info.m) / Self.log_cluster_m
        var normalized_n = FastUInt(work_info.n) / Self.log_cluster_n
        comptime log_block_swizzle_size = FastDiv[.uint32](
            Self.block_swizzle_size
        )
        var linear_cluster_id = (
            normalized_m * FastUInt(cluster_dim[1]) + normalized_n
        )

        var new_normalized_m: FastUInt
        var new_normalized_n: FastUInt
        # CLC rasterize along M by default.
        comptime if Self.rasterize_order == RasterOrder.AlongM:
            new_normalized_m = normalized_m
            new_normalized_n = normalized_n
        else:
            new_normalized_m = linear_cluster_id % log_cluster_dim_m
            new_normalized_n = linear_cluster_id / log_cluster_dim_m

        var new_m_global: FastUInt
        var new_n_global: FastUInt
        comptime if Self.block_swizzle_size != 0:
            var swizzle_m_size = (
                FastUInt(cluster_dim[0]) / log_block_swizzle_size
            )
            var swizzle_n_size = (
                FastUInt(cluster_dim[1]) / log_block_swizzle_size
            )

            var m_local = (new_normalized_m / log_block_swizzle_size) + (
                (swizzle_m_size) * (new_normalized_n % log_block_swizzle_size)
            )
            var n_local = new_normalized_m % log_block_swizzle_size

            var is_even_subtile = Int(
                (new_normalized_n / log_block_swizzle_size) % 2 == 0
            )

            var m_bound = swizzle_m_size * FastUInt(Self.block_swizzle_size)
            var n_bound = swizzle_n_size * FastUInt(Self.block_swizzle_size)
            if new_normalized_m < m_bound and new_normalized_n < n_bound:
                new_m_global = FastUInt(is_even_subtile) * m_local + FastUInt(
                    1 - is_even_subtile
                ) * (m_bound - m_local - 1)
                new_n_global = n_local + FastUInt(
                    Int(new_normalized_n / log_block_swizzle_size)
                    * Self.block_swizzle_size
                )
            else:
                new_m_global = new_normalized_m
                new_n_global = new_normalized_n
        else:
            new_m_global = new_normalized_m
            new_n_global = new_normalized_n

        return WorkInfo(
            m=UInt32(
                Int(new_m_global) * Self.cluster_shape[0]
                + block_id_in_cluster.x
            ),
            n=UInt32(
                Int(new_n_global) * Self.cluster_shape[1]
                + block_id_in_cluster.y
            ),
            k_start=work_info.k_start,
            is_valid_tile=work_info.is_valid_tile,
        )

    @always_inline
    def initial_work_info(self) -> WorkInfo:
        return self.work_info_from_cluster(
            WorkInfo(
                UInt32(block_idx.x),
                UInt32(block_idx.y),
                UInt32(block_idx.z),
                is_valid_tile=True,
            ),
            self.cluster_dim,
            self.log_cluster_dim_m,
            self.log_cluster_dim_n,
        )

    @always_inline
    def fetch_next_work(
        self,
        work_info: WorkInfo,
        consumer_state: PipelineState[Self.num_stages],
    ) -> WorkInfo:
        # num_stages == 0 implies there is only one wave. Only initial
        # work info is valid, next work info is invalid.
        comptime if Self.num_stages == 0:
            return WorkInfo(0, 0, 0, False)

        self.full_mbar[consumer_state.index()].wait(consumer_state.phase())
        var work_tile = self.work_info_from_clc_response(
            self.clc_response + consumer_state.index()
        )
        # Only cta 0 in a cluster is used for scheduling.
        self.empty_mbar[consumer_state.index()].arrive_cluster(0)
        return self.work_info_from_cluster(
            work_tile,
            self.cluster_dim,
            self.log_cluster_dim_m,
            self.log_cluster_dim_n,
        )

    @always_inline
    def advance_to_next_work(
        self,
        mut clc_state: PipelineState[Self.num_stages],
    ) -> PipelineState[Self.num_stages]:
        comptime multicast = True if Self.cluster_size > 1 else False
        var lane_id = lane_id()
        var pred = UInt32(1) if lane_id < Self.cluster_size else UInt32(0)
        self.empty_mbar[clc_state.index()].wait(clc_state.phase())
        self.full_mbar[clc_state.index()].arrive_and_expect_bytes(
            Int32(size_of[UInt128]()),
            UInt32(lane_id),
            pred,
        )

        if elect_one_sync():
            clusterlaunchcontrol_try_cancel[multicast=multicast](
                self.clc_response + clc_state.index(),
                (self.full_mbar + clc_state.index()).bitcast[Int64](),
            )

        return clc_state.next()
