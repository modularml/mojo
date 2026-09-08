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

"""
Provides warp-level profiling infrastructure for Blackwell warp-specialized matmul kernels.

Defines a workspace manager that allocates per-SM recording buffers and a profile warp
helper that captures start and end timestamps around a scoped region and writes a single
timeline entry to the workspace.
"""


from std.time.time import global_perf_counter_ns
from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.host.info import B200
from max.gpu import sm_id


comptime MatmulWarpSpecializationWorkSpaceManager[
    max_entries_per_warp: UInt32
] = BlackwellWarpProfilingWorkspaceManager[1, 1, 1, 4, max_entries_per_warp]

comptime MatmulProfileWarp[
    warp_role: UInt32, max_entries_per_warp: UInt32
] = BlackwellProfileWarp[
    MatmulWarpSpecializationWorkSpaceManager[max_entries_per_warp](),
    warp_role=warp_role,
]


@fieldwise_init
struct BlackwellWarpProfilingWorkspaceManager[
    load_warps: UInt32,
    mma_warps: UInt32,
    scheduler_warps: UInt32,
    epilogue_warps: UInt32,
    max_entries_per_warp: UInt32,
](TrivialRegisterPassable):
    """
    Profiling workspace for warp-role timelines on each SM.

    Each SM owns a fixed-size chunk of timestamp entries, capped per warp role.

    Parameters:
        load_warps: Number of warps specialized for load operations.
        mma_warps: Number of warps specialized for matrix
            multiply-accumulate operations.
        scheduler_warps: Number of warps specialized for scheduling
            operations.
        epilogue_warps: Number of warps specialized for epilogue
            operations.
        max_entries_per_warp: Maximum number of entries per warp (common
            across all warp roles).
    """

    # load, scheduler, mma, epilogue
    comptime total_warp_roles = 4

    # how many values will be recorded per entry
    comptime total_data_points = 7

    # this header shows what each value in an entry symbolizes in a csv friendly format
    comptime header = (
        "time_start,time_end,sm_id,block_idx_x,block_idx_y,role,entry_idx\n"
    )

    comptime sm_count = B200.sm_count
    comptime entries_per_sm = Self.total_warp_roles * Self.max_entries_per_warp

    @staticmethod
    @__parameter
    def _get_warp_count[warp_role: UInt32]() -> UInt32:
        comptime if warp_role == 0:
            return Self.load_warps
        elif warp_role == 1:
            return Self.scheduler_warps
        elif warp_role == 2:
            return Self.mma_warps
        else:
            return Self.epilogue_warps

    @staticmethod
    @__parameter
    def _calculate_entries_before_role[warp_role: UInt32]() -> UInt32:
        return warp_role * Self.max_entries_per_warp

    @staticmethod
    @always_inline
    def _get_workspace_offset[
        warp_role: UInt32
    ](sm_idx: UInt32, entry_idx: UInt32) -> UInt32:
        var sm_length = Self.total_data_points * Self.entries_per_sm

        return (
            (sm_idx * sm_length)
            + (
                Self._calculate_entries_before_role[warp_role]()
                * Self.total_data_points
            )
            + (entry_idx * Self.total_data_points)
        )

    @staticmethod
    @__parameter
    def _calculate_buffer_length() -> UInt32:
        return (
            UInt32(Self.sm_count) * Self.entries_per_sm * Self.total_data_points
        )

    @staticmethod
    @always_inline
    def get_workspace(
        ctx: DeviceContext,
    ) raises -> Span[UInt64, MutAnyOrigin]:
        var length = Int(Self._calculate_buffer_length())
        var device_buffer = ctx.enqueue_create_buffer[.uint64](length)
        device_buffer.enqueue_fill(0)
        return Span[UInt64, MutAnyOrigin](
            unsafe_ptr=device_buffer.unsafe_ptr().as_unsafe_any_origin(),
            length=length,
        )

    @staticmethod
    @always_inline
    def write_to_workspace[
        workspace_origin: MutOrigin, //, warp_role: UInt32
    ](
        sm_idx: UInt32,
        entry_idx: UInt32,
        workspace: Span[UInt64, workspace_origin],
        timeline: Tuple[UInt64, UInt64],
    ):
        comptime total_threads = UInt32(WARP_SIZE) * Self._get_warp_count[
            warp_role
        ]()

        var start_idx = Self._get_workspace_offset[warp_role](sm_idx, entry_idx)

        if UInt32(thread_idx.x) % total_threads == 0:
            workspace[start_idx] = timeline[0]
            workspace[start_idx + 1] = timeline[1]
            workspace[start_idx + 2] = UInt64(sm_idx)
            workspace[start_idx + 3] = UInt64(block_idx.x)
            workspace[start_idx + 4] = UInt64(block_idx.y)
            workspace[start_idx + 5] = UInt64(warp_role)
            workspace[start_idx + 6] = UInt64(entry_idx)

    @staticmethod
    @always_inline
    def dump_workspace_as_csv(
        ctx: DeviceContext,
        workspace: Span[UInt64, MutAnyOrigin],
        filename: StaticString,
    ) raises:
        var length = Int(Self._calculate_buffer_length())
        var host_buffer = ctx.enqueue_create_host_buffer[.uint64](length)
        ctx.enqueue_copy(host_buffer, workspace)
        ctx.synchronize()

        var host_span = host_buffer.as_span()

        var entries = len(host_buffer) // (Self.total_data_points)
        with open(filename + ".csv", "w") as f:
            f.write(Self.header)
            for entry in range(entries):
                var start = entry * Self.total_data_points
                f.write(
                    ", ".join(
                        [
                            String(x)
                            for x in host_span[
                                start : start + Self.total_data_points
                            ]
                        ]
                    )
                )
                f.write("\n")


struct BlackwellProfileWarp[
    workspace_origin: MutOrigin,
    load_warps: UInt32,
    mma_warps: UInt32,
    scheduler_warps: UInt32,
    epilogue_warps: UInt32,
    max_entries_per_warp: UInt32,
    //,
    WorkspaceManager: BlackwellWarpProfilingWorkspaceManager[
        load_warps,
        mma_warps,
        scheduler_warps,
        epilogue_warps,
        max_entries_per_warp,
    ],
    warp_role: UInt32 = 0,
](ImplicitlyCopyable):
    """Calculates execution time for a warp/s,
    and writes a single entry to the workspace.

    Parameters:
        workspace_origin: Memory origin of the profiling workspace buffer
            (inferred).
        load_warps: Number of warps specialized for load operations
            (inferred).
        mma_warps: Number of warps specialized for matrix multiply-accumulate
            operations (inferred).
        scheduler_warps: Number of warps specialized for scheduling
            operations (inferred).
        epilogue_warps: Number of warps specialized for epilogue operations
            (inferred).
        max_entries_per_warp: Maximum number of timeline entries recorded per
            warp role (inferred). Profiling is enabled when this value is
            greater than zero.
        WorkspaceManager: Workspace manager responsible for allocating the
            profiling buffer and writing timeline entries (defaults to
            `BlackwellWarpProfilingWorkspaceManager` parameterized by the
            warp counts).
        warp_role: Role of this warp, where 0 is load, 1 is scheduler, 2 is
            mma, and 3 is epilogue (defaults to 0).
    """

    comptime enable_profiling = Self.max_entries_per_warp > 0

    var timeline: Tuple[UInt64, UInt64]
    var workspace: Span[UInt64, Self.workspace_origin]

    # which entry is going to be written to the workspace for this warp
    var entry_idx: UInt32

    @always_inline
    def __init__(
        out self,
        workspace: Span[UInt64, Self.workspace_origin],
        entry_idx: UInt32,
    ):
        self.timeline = (0, 0)
        self.workspace = workspace
        self.entry_idx = entry_idx

    @always_inline
    def __enter__(mut self):
        comptime if Self.enable_profiling:
            self.timeline[0] = global_perf_counter_ns()

    @always_inline
    def __exit__(mut self):
        comptime if Self.enable_profiling:
            self.timeline[1] = global_perf_counter_ns()
            Self.WorkspaceManager.write_to_workspace[Self.warp_role](
                UInt32(sm_id()),
                self.entry_idx,
                self.workspace,
                self.timeline,
            )
