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
"""Compares the relay-assisted grouped allgather against the in-tree one.

The node's GPUs are split into groups of `group_size` that run their allgather
concurrently, which is the TP x DP shape the relay path targets. The baseline
is `allgather` on one group alone; the candidate is `allgather_relay`, which
additionally routes part of every shard over the inter-group links the baseline
leaves idle. Adjacent groups are paired, so `group_size` 4 gives two groups on
an 8-GPU node and `group_size` 2 gives four.

Every measured configuration is verified on device against a per-rank pattern,
with the outputs zeroed beforehand so a configuration that skips bytes cannot
inherit a previous one's results.

Usage:
    # Compare the tuned defaults across the shard sizes that matter.
    mojo bench_allgather_relay.mojo

    # Sweep block counts and relayed fraction to produce tuning-table rows.
    mojo bench_allgather_relay.mojo --mode=tune

    # Four groups of two instead of two groups of four.
    mojo -D group_size=2 bench_allgather_relay.mojo
"""

from std.math.uutils import ualign_down
from std.collections import Array
from std.sys import size_of, simd_width_of
from std.sys.defines import get_defined_int
from std.utils import StaticTuple

from comm import MAX_GPUS, Signal
from comm.allgather import (
    _allgather_p2p_relay,
    allgather,
    allgather_relay_tuning_table,
)
from comm.relay import RelayTuningConfig
from comm.device_query import dispatch_select_comm_config
from comm.sync import enable_p2p
from internal_utils import arg_parse, human_readable_size
from layout import TileTensor, row_major
from max.algorithm import sync_parallelize
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    global_idx,
    grid_dim,
)
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target

from std.testing import assert_true

comptime WORLD = 8
"""GPUs the benchmark drives."""

comptime GROUP = get_defined_int["group_size", 4]()
"""GPUs per collective group; adjacent groups relay for each other."""

comptime PAIR = 2 * GROUP
"""GPUs in one relay world: a group plus the group that relays for it."""

comptime dtype = DType.bfloat16


@always_inline
def _pattern(gpu_rank: Int, j: Int) -> Scalar[dtype]:
    # 251 is the largest prime < 256; using a prime avoids power-of-two
    # aliasing between the rank term and the index term.
    return Scalar[dtype](Scalar[dtype](gpu_rank + 1) + Scalar[dtype](j % 251))


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256))
)
def _check_kernel(
    out_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    length: Int32,
    src_rank: Int32,
    first_bad: MutPointer[Int32, MutAnyOrigin],
):
    """Records one mismatching index (biased by 1) if the output is wrong.

    Racing writers all store an index that is equally valid as a report, so no
    atomic is needed; the caller zeroes `first_bad` before launching.
    """
    var stride = grid_dim.x * block_dim.x
    for i in range(global_idx.x, Int(length), stride):
        if out_ptr[i] != _pattern(Int(src_rank), i):
            first_bad[] = Int32(i) + 1


def main() raises:
    var mode = arg_parse("mode", String("compare"))
    var iters = arg_parse("iters", 20)
    var warmup = arg_parse("warmup", 5)
    var max_mb = arg_parse("max_mb", 48)

    comptime assert WORLD % PAIR == 0, "group_size must tile the node evenly"

    var num_gpus_found = DeviceContext.number_of_devices()
    assert_true(
        num_gpus_found >= WORLD,
        String(num_gpus_found) + " devices found, expected " + String(WORLD),
    )

    if not enable_p2p():
        print("P2P not enabled, skipping benchmark.")
        return

    var list_of_ctx = List[DeviceContext]()
    for i in range(WORLD):
        list_of_ctx.append(DeviceContext(device_id=i))

    # Buffers are allocated once at the largest shard and reused for every
    # smaller one, since the verification pattern depends only on the index.
    var max_length = max_mb * 1024 * 1024 // size_of[dtype]()

    var in_bufs = List[DeviceBuffer[dtype]](capacity=WORLD)
    var out_bufs = List[DeviceBuffer[dtype]](capacity=WORLD * GROUP)
    var signal_bufs = List[DeviceBuffer[.uint8]](capacity=WORLD)
    var relay_signal_bufs = List[DeviceBuffer[.uint8]](capacity=WORLD)
    var flag_bufs = List[DeviceBuffer[.int32]](capacity=WORLD)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    var relay_rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    var host_buffer = List[Scalar[dtype]](unsafe_uninit_length=max_length)

    for gpu_idx in range(WORLD):
        var ctx = list_of_ctx[gpu_idx]
        in_bufs.append(ctx.enqueue_create_buffer[dtype](max_length))
        for _ in range(GROUP):
            out_bufs.append(ctx.enqueue_create_buffer[dtype](max_length))
        flag_bufs.append(ctx.enqueue_create_buffer[.int32](1))

        for j in range(max_length):
            host_buffer[j] = _pattern(gpu_idx, j)
        ctx.enqueue_copy(in_bufs[gpu_idx], host_buffer)

        signal_bufs.append(ctx.create_buffer_sync[.uint8](size_of[Signal]()))
        ctx.enqueue_memset[.uint8](signal_bufs[gpu_idx], 0)
        rank_sigs[gpu_idx] = (
            signal_bufs[gpu_idx]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )
        relay_signal_bufs.append(
            ctx.create_buffer_sync[.uint8](size_of[Signal]())
        )
        ctx.enqueue_memset[.uint8](relay_signal_bufs[gpu_idx], 0)
        relay_rank_sigs[gpu_idx] = (
            relay_signal_bufs[gpu_idx]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )
    for gpu_idx in range(WORLD):
        list_of_ctx[gpu_idx].synchronize()

    # Raw pointers, so the tile helpers below can hand out mutable views even
    # though they only borrow the buffer lists.
    comptime PtrType = MutPointer[Scalar[dtype], MutAnyOrigin]
    var in_ptrs = List[PtrType](capacity=WORLD)
    var out_ptrs = List[PtrType](capacity=WORLD * GROUP)
    for gpu_idx in range(WORLD):
        in_ptrs.append(in_bufs[gpu_idx].unsafe_ptr().as_unsafe_any_origin())
        for src_idx in range(GROUP):
            out_ptrs.append(
                out_bufs[gpu_idx * GROUP + src_idx]
                .unsafe_ptr()
                .as_unsafe_any_origin()
            )

    comptime InTileType = TileTensor[
        dtype, type_of(row_major(max_length)), ImmutAnyOrigin
    ]
    comptime OutTileType = TileTensor[
        dtype, type_of(row_major(max_length)), MutAnyOrigin
    ]

    @always_inline
    def in_tile(rank: Int, length: Int) {imm} -> InTileType:
        return TileTensor(in_ptrs[rank], row_major(length)).as_immut()

    @always_inline
    def out_tile(rank: Int, src: Int, length: Int) {imm} -> OutTileType:
        return TileTensor(out_ptrs[rank * GROUP + src], row_major(length))

    @always_inline
    def launch_baseline(
        rank: Int,
        ctx: DeviceContext,
        length: Int,
        max_num_blocks: Optional[Int],
    ) raises {imm}:
        # The group is presented as its own world, so `allgather` sees one
        # group, leaves the relay gate shut, and runs the plain path this is
        # the baseline for.
        var group_base = ualign_down(rank, GROUP)
        var group_in = Array[InTileType, GROUP](uninitialized=True)
        var group_out = Array[OutTileType, GROUP * GROUP](uninitialized=True)
        var group_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )
        for i in range(GROUP):
            group_in[i] = in_tile(group_base + i, length)
            group_sigs[i] = rank_sigs[group_base + i]
            for src_idx in range(GROUP):
                group_out[i * GROUP + src_idx] = out_tile(
                    group_base + i, src_idx, length
                )

        allgather[group_size=GROUP](
            group_in,
            group_out,
            group_sigs,
            ctx,
            rank - group_base,
            max_num_blocks,
        )

    comptime sm_version = DeviceContext.default_device_info.version

    @always_inline
    def table_recipe(length: Int) {imm} -> RelayTuningConfig:
        """The recipe `allgather` would pick for a shard of this size."""
        return dispatch_select_comm_config[
            GROUP, sm_version, allgather_relay_tuning_table
        ](length * size_of[dtype]())

    @always_inline
    def launch_relay(
        rank: Int,
        ctx: DeviceContext,
        length: Int,
        override: Optional[RelayTuningConfig],
    ) raises {imm}:
        # The relay world is this rank's group plus the group it is paired
        # with; ranks and signals are packed into that world's own order.
        var pair_base = ualign_down(rank, PAIR)
        var group_base = ualign_down(rank, GROUP)
        var peer_base: Int
        if group_base == pair_base:
            peer_base = group_base + GROUP
        else:
            peer_base = pair_base

        comptime SrcPtrType = ImmPointer[Scalar[dtype], ImmutAnyOrigin]
        comptime OutPtrType = MutPointer[Scalar[dtype], MutAnyOrigin]
        var my_in_ptrs = StaticTuple[SrcPtrType, GROUP]()
        var my_out_ptrs = StaticTuple[OutPtrType, GROUP]()
        var my_lengths = StaticTuple[Int32, GROUP](Int32(length))
        for i in range(GROUP):
            my_in_ptrs[i] = rebind[SrcPtrType](
                in_tile(group_base + i, length).ptr
            )
            my_out_ptrs[i] = out_tile(rank, i, length).ptr

        var peer_in_ptrs = StaticTuple[
            ImmPointer[Scalar[dtype], ImmutAnyOrigin], GROUP
        ]()
        var peer_lengths = StaticTuple[Int32, GROUP](Int32(length))
        var peer_out_ptrs = StaticTuple[
            MutPointer[Scalar[dtype], MutAnyOrigin], GROUP * GROUP
        ]()
        for dst_idx in range(GROUP):
            peer_in_ptrs[dst_idx] = rebind[
                ImmPointer[Scalar[dtype], ImmutAnyOrigin]
            ](in_tile(peer_base + dst_idx, length).ptr)
            for src_idx in range(GROUP):
                peer_out_ptrs[dst_idx * GROUP + src_idx] = rebind[
                    MutPointer[Scalar[dtype], MutAnyOrigin]
                ](out_tile(peer_base + dst_idx, src_idx, length).ptr)

        var pair_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )
        for i in range(PAIR):
            pair_sigs[i] = relay_rank_sigs[pair_base + i]

        var recipe = override.or_else(table_recipe(length))

        _allgather_p2p_relay(
            my_out_ptrs,
            my_in_ptrs,
            my_lengths,
            peer_in_ptrs,
            peer_lengths,
            peer_out_ptrs,
            pair_sigs,
            recipe,
            ctx,
            rank - pair_base,
        )

    @always_inline
    def verify(label: String, length: Int) raises {imm}:
        for gpu_idx in range(WORLD):
            var ctx = list_of_ctx[gpu_idx]
            var group_base = ualign_down(gpu_idx, GROUP)
            for src_idx in range(GROUP):
                ctx.enqueue_memset[.int32](flag_bufs[gpu_idx], 0)
                ctx.enqueue_function[_check_kernel](
                    out_tile(gpu_idx, src_idx, length).ptr,
                    Int32(length),
                    Int32(group_base + src_idx),
                    flag_bufs[gpu_idx].unsafe_ptr().as_unsafe_any_origin(),
                    grid_dim=256,
                    block_dim=256,
                )
                var flag = List[Int32](unsafe_uninit_length=1)
                ctx.enqueue_copy(flag, flag_bufs[gpu_idx])
                ctx.synchronize()
                if flag[0] != 0:
                    raise Error(
                        String(
                            label,
                            ": verification failed on GPU ",
                            gpu_idx,
                            " source ",
                            src_idx,
                            " at index ",
                            Int(flag[0]) - 1,
                        )
                    )

    var times_ms = List[Float64](length=WORLD, fill=0.0)

    @always_inline
    def measure[
        LaunchType: def(Int, DeviceContext) raises -> None
    ](launch: LaunchType, length: Int) raises {mut times_ms, imm} -> Float64:
        # Zero the outputs so verification cannot pass on stale results.
        for gpu_idx in range(WORLD):
            for src_idx in range(GROUP):
                list_of_ctx[gpu_idx].enqueue_memset[dtype](
                    out_bufs[gpu_idx * GROUP + src_idx], 0
                )
            list_of_ctx[gpu_idx].synchronize()

        def per_gpu(rank: Int) raises {mut times_ms, imm}:
            var ctx = list_of_ctx[rank]

            def one_iter(c: DeviceContext) raises {imm}:
                launch(rank, c)

            for _ in range(warmup):
                one_iter(ctx)
            ctx.synchronize()
            times_ms[rank] = Float64(ctx.execution_time(one_iter, iters)) / (
                Float64(iters) * 1.0e6
            )

        sync_parallelize(per_gpu, WORLD)
        var slowest = 0.0
        for rank in range(WORLD):
            slowest = max(slowest, times_ms[rank])
        return slowest

    @always_inline
    def run_baseline(
        length: Int, max_num_blocks: Optional[Int], label: String
    ) raises {mut times_ms, imm} -> Float64:
        def launch(rank: Int, ctx: DeviceContext) raises {imm}:
            launch_baseline(rank, ctx, length, max_num_blocks)

        var ms = measure(launch, length)
        verify(label, length)
        return ms

    @always_inline
    def run_relay(
        length: Int,
        override: Optional[RelayTuningConfig],
        label: String,
    ) raises {mut times_ms, imm} -> Float64:
        def launch(rank: Int, ctx: DeviceContext) raises {imm}:
            launch_relay(rank, ctx, length, override)

        var ms = measure(launch, length)
        verify(label, length)
        return ms

    @always_inline
    def gbps(length: Int, ms: Float64) {imm} -> Float64:
        return Float64(GROUP * length * size_of[dtype]()) / (ms * 1.0e6)

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    print(
        "# relay allgather:",
        WORLD,
        "GPUs,",
        WORLD // GROUP,
        "groups of",
        GROUP,
        "| dtype",
        dtype,
        "| simd_width",
        simd_width,
        "| iters",
        iters,
    )

    var shard_kb = List[Int](capacity=8)
    if arg_parse("small", 0) != 0:
        # Below a couple of MB the collective is barrier-bound, so this range
        # is about where relaying stops paying for its wider barrier.
        for kb in [128, 256, 512, 1024, 2048]:
            shard_kb.append(kb)
    else:
        for mb in [2, 4, 8, 12, 16, 24, 32, 48]:
            shard_kb.append(mb * 1024)

    if mode == "tune":
        # Grid over the three recipe knobs. Block counts are absolute, as in
        # the tuning table, and every value is a multiple of the supported
        # group widths so the same grid is comparable across them.
        var direct_blocks = [32, 48, 64, 72, 96]
        var relay_blocks = [4, 8, 16]
        var percents = [38, 42, 46, 50]

        print("CSVHDR,shard_kb,direct,relay,percent,ms,gbps")
        for si in range(len(shard_kb)):
            if shard_kb[si] > max_mb * 1024:
                continue
            var length = shard_kb[si] * 1024 // size_of[dtype]()
            var base_ms = run_baseline(length, None, "baseline")
            var default_ms = run_relay(length, None, "relay-default")
            print(
                "CSV,",
                shard_kb[si],
                ",baseline,-,-,",
                base_ms,
                ",",
                gbps(length, base_ms),
            )
            print(
                "CSV,",
                shard_kb[si],
                ",default,-,-,",
                default_ms,
                ",",
                gbps(length, default_ms),
            )
            for di in range(len(direct_blocks)):
                for ri in range(len(relay_blocks)):
                    for pi in range(len(percents)):
                        var cfg = RelayTuningConfig(
                            group_size=GROUP,
                            num_bytes=-1,
                            num_blocks=direct_blocks[di],
                            num_relay_blocks=relay_blocks[ri],
                            relay_percent=percents[pi],
                        )
                        var label = String(
                            "relay d=",
                            direct_blocks[di],
                            " r=",
                            relay_blocks[ri],
                            " f=",
                            percents[pi],
                        )
                        var ms = run_relay(length, Optional(cfg), label)
                        print(
                            "CSV,",
                            shard_kb[si],
                            ",",
                            direct_blocks[di],
                            ",",
                            relay_blocks[ri],
                            ",",
                            percents[pi],
                            ",",
                            ms,
                            ",",
                            gbps(length, ms),
                        )
        return

    print(
        "| shard | allgather ms | allgather GB/s | relay ms | relay GB/s |"
        " speedup |"
    )
    for si in range(len(shard_kb)):
        if shard_kb[si] > max_mb * 1024:
            continue
        var length = shard_kb[si] * 1024 // size_of[dtype]()
        var base_ms = run_baseline(length, None, "allgather")

        # A zero relayed share is the table declining the relay path, which is
        # what `allgather` would honour; measuring the launcher anyway would
        # time a recipe no caller ever gets.
        if table_recipe(length).relay_percent == 0:
            print(
                "|",
                human_readable_size(length * size_of[dtype]()),
                "|",
                base_ms,
                "|",
                gbps(length, base_ms),
                "| relay declined by the tuning table |",
            )
            continue

        var relay_ms = run_relay(length, None, "allgather_relay")
        print(
            "|",
            human_readable_size(length * size_of[dtype]()),
            "|",
            base_ms,
            "|",
            gbps(length, base_ms),
            "|",
            relay_ms,
            "|",
            gbps(length, relay_ms),
            "|",
            base_ms / relay_ms,
            "|",
        )

    _ = host_buffer^
    _ = signal_bufs^
    _ = relay_signal_bufs^
    _ = flag_bufs^
    _ = in_bufs^
    _ = out_bufs^
