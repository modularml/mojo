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
"""Compares the relay-assisted grouped reduce-scatter against the in-tree one.

The node's GPUs are split into groups of `group_size` that run their
reduce-scatter concurrently, which is the TP x DP shape the relay path targets.
The baseline is `reducescatter` on one group alone, presented as its own world
so the relay gate stays shut; the candidate is the relay launcher, whose GPUs
in the partner group reduce a trailing slice of each destination's partition
and write the finished values straight in.

Sizes are named by the per-GPU output partition, which is what sets per-link
traffic; the reported bandwidth uses the per-GPU input bytes, `group_size`
times larger, to match how reduce-scatter is usually quoted.

Every measured configuration is verified on device against the exact sum of a
small-integer pattern, with the outputs zeroed beforehand so a configuration
that skips bytes cannot inherit a previous one's results.

Usage:
    mojo bench_reducescatter_relay.mojo
    mojo bench_reducescatter_relay.mojo --mode=tune
    mojo -D group_size=2 bench_reducescatter_relay.mojo
    mojo -D with_residual=true bench_reducescatter_relay.mojo --mode=tune
"""

from std.math.uutils import ualign_down
from std.collections import Array
from std.sys import size_of, simd_width_of
from std.sys.defines import get_defined_bool, get_defined_int
from std.utils import StaticTuple
from std.utils.numerics import get_accum_type

from comm import MAX_GPUS, Signal
from comm.reducescatter import (
    _reducescatter_p2p_relay,
    reducescatter,
    reducescatter_relay_residual_tuning_table,
    reducescatter_relay_tuning_table,
)
from comm.device_query import dispatch_select_comm_config
from comm.relay import RelayTuningConfig
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

comptime HAS_RESIDUAL = get_defined_bool["with_residual", False]()
"""Fold a residual into every reduced value, as the fused norm ops do.

This is a separate operating point rather than a variation on the same one: a
relay has to pull the destination's residual across the same inter-group link
it reads that group's inputs on, so the leg it pays for grows and the relayed
share that balances the links shrinks.
"""

comptime dtype = DType.bfloat16
comptime accum_type = get_accum_type[dtype]()


@always_inline
def _pattern(gpu_rank: Int, j: Int) -> Scalar[accum_type]:
    # Every value has to be exact in `dtype` as well as in the accumulator, or
    # the check's sum of exact patterns would drift from the kernel's sum of
    # rounded inputs. bfloat16 has 8 mantissa bits, so integers up to 256 are
    # exact and this pattern tops out at 134. 127 is prime, which keeps the
    # rank term from aliasing the index term.
    return Scalar[accum_type](gpu_rank + 1) + Scalar[accum_type](j % 127)


@always_inline
def _res_pattern(group_id: Int, j: Int) -> Scalar[accum_type]:
    """Residual pattern: replicated within a group, distinct between groups."""
    return Scalar[accum_type](j % 31) + Scalar[accum_type](32 * group_id)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256))
)
def _check_kernel(
    out_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    length: Int32,
    part_start: Int32,
    group_base: Int32,
    first_bad: MutPointer[Int32, MutAnyOrigin],
):
    """Records one mismatching index (biased by 1) if the partition is wrong.

    Racing writers all store an index that is equally valid as a report, so no
    atomic is needed; the caller zeroes `first_bad` before launching.
    """
    var stride = grid_dim.x * block_dim.x
    for i in range(global_idx.x, Int(length), stride):
        var expected = Scalar[accum_type](0)
        for g in range(GROUP):
            expected += _pattern(Int(group_base) + g, Int(part_start) + i)
        var rounded = expected.cast[dtype]()
        comptime if HAS_RESIDUAL:
            # The kernel's order: round the reduction, add in the accumulate
            # type, round once more.
            rounded = (
                rounded.cast[accum_type]()
                + _res_pattern(Int(group_base) // GROUP, Int(part_start) + i)
            ).cast[dtype]()
        if out_ptr[i] != rounded:
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

    # Allocated once at the largest partition and reused for every smaller one;
    # the pattern depends only on the flat index.
    var max_part = max_mb * 1024 * 1024 // size_of[dtype]()
    var max_input = GROUP * max_part

    var in_bufs = List[DeviceBuffer[dtype]](capacity=WORLD)
    var out_bufs = List[DeviceBuffer[dtype]](capacity=WORLD)
    var res_bufs = List[DeviceBuffer[dtype]](capacity=WORLD)
    var signal_bufs = List[DeviceBuffer[.uint8]](capacity=WORLD)
    var relay_signal_bufs = List[DeviceBuffer[.uint8]](capacity=WORLD)
    var flag_bufs = List[DeviceBuffer[.int32]](capacity=WORLD)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    var relay_rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    var host_buffer = List[Scalar[dtype]](unsafe_uninit_length=max_input)

    for gpu_idx in range(WORLD):
        var ctx = list_of_ctx[gpu_idx]
        in_bufs.append(ctx.enqueue_create_buffer[dtype](max_input))
        out_bufs.append(ctx.enqueue_create_buffer[dtype](max_part))
        flag_bufs.append(ctx.enqueue_create_buffer[.int32](1))

        for j in range(max_input):
            host_buffer[j] = _pattern(gpu_idx, j).cast[dtype]()
        ctx.enqueue_copy(in_bufs[gpu_idx], host_buffer)

        res_bufs.append(ctx.enqueue_create_buffer[dtype](max_input))
        comptime if HAS_RESIDUAL:
            for j in range(max_input):
                host_buffer[j] = _res_pattern(gpu_idx // GROUP, j).cast[dtype]()
            ctx.enqueue_copy(res_bufs[gpu_idx], host_buffer)

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

    comptime PtrType = MutPointer[Scalar[dtype], MutAnyOrigin]
    comptime SrcPtrType = ImmPointer[Scalar[dtype], ImmutAnyOrigin]
    var in_ptrs = List[PtrType](capacity=WORLD)
    var out_ptrs = List[PtrType](capacity=WORLD)
    var res_ptrs = List[SrcPtrType](capacity=WORLD)
    for gpu_idx in range(WORLD):
        in_ptrs.append(in_bufs[gpu_idx].unsafe_ptr().as_unsafe_any_origin())
        out_ptrs.append(out_bufs[gpu_idx].unsafe_ptr().as_unsafe_any_origin())
        res_ptrs.append(
            rebind[SrcPtrType](
                res_bufs[gpu_idx].unsafe_ptr().as_unsafe_any_origin()
            )
        )

    comptime InTileType = TileTensor[
        dtype, type_of(row_major(max_input)), ImmutAnyOrigin
    ]
    comptime OutTileType = TileTensor[
        dtype, type_of(row_major(max_part)), MutAnyOrigin
    ]

    @always_inline
    def in_tile(rank: Int, numel: Int) {imm} -> InTileType:
        return TileTensor(in_ptrs[rank], row_major(numel)).as_immut()

    @always_inline
    def out_tile(rank: Int, numel: Int) {imm} -> OutTileType:
        return TileTensor(out_ptrs[rank], row_major(numel))

    comptime sm_version = DeviceContext.default_device_info.version

    @always_inline
    def table_recipe(part: Int) {imm} -> RelayTuningConfig:
        """The recipe `reducescatter` would pick for this partition size."""
        comptime table = reducescatter_relay_residual_tuning_table if HAS_RESIDUAL else reducescatter_relay_tuning_table
        return dispatch_select_comm_config[GROUP, sm_version, table](
            part * size_of[dtype]()
        )

    @always_inline
    def launch_baseline(
        rank: Int, ctx: DeviceContext, part: Int, max_num_blocks: Optional[Int]
    ) raises {imm}:
        # The group is presented as its own world, so `reducescatter` sees one
        # group and leaves the relay gate shut.
        var group_base = ualign_down(rank, GROUP)
        var group_in = Array[InTileType, GROUP](uninitialized=True)
        var group_out = Array[OutTileType, GROUP](uninitialized=True)
        var group_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )
        for i in range(GROUP):
            group_in[i] = in_tile(group_base + i, GROUP * part)
            group_out[i] = out_tile(group_base + i, part)
            group_sigs[i] = rank_sigs[group_base + i]

        comptime if HAS_RESIDUAL:
            var group_res = Array[SrcPtrType, GROUP](uninitialized=True)
            for i in range(GROUP):
                group_res[i] = res_ptrs[group_base + i]
            reducescatter[ngpus=GROUP, group_size=GROUP, has_residual=True](
                group_in,
                group_out,
                group_sigs,
                ctx,
                max_num_blocks,
                Optional[Int](rank - group_base),
                residuals=Optional(group_res.copy()),
            )
        else:
            reducescatter[ngpus=GROUP, group_size=GROUP](
                group_in,
                group_out,
                group_sigs,
                ctx,
                max_num_blocks,
                Optional[Int](rank - group_base),
            )

    @always_inline
    def launch_relay(
        rank: Int,
        ctx: DeviceContext,
        part: Int,
        override: Optional[RelayTuningConfig],
    ) raises {imm}:
        var pair_base = ualign_down(rank, PAIR)
        var group_base = ualign_down(rank, GROUP)
        var peer_base: Int
        if group_base == pair_base:
            peer_base = group_base + GROUP
        else:
            peer_base = pair_base

        var my_in_ptrs = StaticTuple[SrcPtrType, GROUP]()
        var peer_in_ptrs = StaticTuple[SrcPtrType, GROUP]()
        var peer_out_ptrs = StaticTuple[PtrType, GROUP]()
        var peer_starts = StaticTuple[Int32, GROUP]()
        var peer_res_ptrs = StaticTuple[SrcPtrType, GROUP]()
        var peer_numels = StaticTuple[Int32, GROUP](Int32(part))
        for i in range(GROUP):
            my_in_ptrs[i] = rebind[SrcPtrType](
                in_tile(group_base + i, GROUP * part).ptr
            )
            peer_in_ptrs[i] = rebind[SrcPtrType](
                in_tile(peer_base + i, GROUP * part).ptr
            )
            peer_out_ptrs[i] = out_tile(peer_base + i, part).ptr
            peer_starts[i] = Int32(i * part)
            peer_res_ptrs[i] = res_ptrs[peer_base + i]

        var pair_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )
        for i in range(PAIR):
            pair_sigs[i] = relay_rank_sigs[pair_base + i]

        _reducescatter_p2p_relay[has_residual=HAS_RESIDUAL](
            out_tile(rank, part).ptr,
            my_in_ptrs,
            peer_out_ptrs,
            peer_in_ptrs,
            pair_sigs,
            (rank - group_base) * part,
            part,
            peer_starts,
            peer_numels,
            res_ptrs[rank],
            peer_res_ptrs,
            override.or_else(table_recipe(part)),
            ctx,
            rank - pair_base,
        )

    @always_inline
    def verify(label: String, part: Int) raises {imm}:
        for gpu_idx in range(WORLD):
            var ctx = list_of_ctx[gpu_idx]
            var group_base = ualign_down(gpu_idx, GROUP)
            ctx.enqueue_memset[.int32](flag_bufs[gpu_idx], 0)
            ctx.enqueue_function[_check_kernel](
                out_tile(gpu_idx, part).ptr,
                Int32(part),
                Int32((gpu_idx - group_base) * part),
                Int32(group_base),
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
                        " at index ",
                        Int(flag[0]) - 1,
                    )
                )

    var times_ms = List[Float64](length=WORLD, fill=0.0)

    @always_inline
    def measure[
        LaunchType: def(Int, DeviceContext) raises -> None
    ](launch: LaunchType, part: Int) raises {mut times_ms, imm} -> Float64:
        for gpu_idx in range(WORLD):
            list_of_ctx[gpu_idx].enqueue_memset[dtype](out_bufs[gpu_idx], 0)
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
        part: Int, max_num_blocks: Optional[Int], label: String
    ) raises {mut times_ms, imm} -> Float64:
        def launch(rank: Int, ctx: DeviceContext) raises {imm}:
            launch_baseline(rank, ctx, part, max_num_blocks)

        var ms = measure(launch, part)
        verify(label, part)
        return ms

    @always_inline
    def run_relay(
        part: Int, override: Optional[RelayTuningConfig], label: String
    ) raises {mut times_ms, imm} -> Float64:
        def launch(rank: Int, ctx: DeviceContext) raises {imm}:
            launch_relay(rank, ctx, part, override)

        var ms = measure(launch, part)
        verify(label, part)
        return ms

    @always_inline
    def gbps(part: Int, ms: Float64) {imm} -> Float64:
        return Float64(GROUP * part * size_of[dtype]()) / (ms * 1.0e6)

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    print(
        "# relay reduce-scatter:",
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
        "| residual",
        HAS_RESIDUAL,
    )

    var part_kb = List[Int](capacity=8)
    if arg_parse("small", 0) != 0:
        for kb in [128, 256, 512, 1024, 2048]:
            part_kb.append(kb)
    else:
        for mb in [2, 4, 8, 12, 16, 24, 32, 48]:
            part_kb.append(mb * 1024)

    if mode == "tune":
        # Block counts are absolute, as in the tuning table.
        var direct_blocks = [16, 20, 28, 40, 64]
        var relay_blocks = [4, 8, 12, 16]
        var percents = [32, 36, 40, 44, 48]

        print("CSVHDR,part_kb,direct,relay,percent,ms,gbps")
        for si in range(len(part_kb)):
            if part_kb[si] > max_mb * 1024:
                continue
            var part = part_kb[si] * 1024 // size_of[dtype]()
            var base_ms = run_baseline(part, None, "baseline")
            print(
                "CSV,",
                part_kb[si],
                ",baseline,-,-,",
                base_ms,
                ",",
                gbps(part, base_ms),
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
                        var ms = run_relay(part, Optional(cfg), "relay")
                        print(
                            "CSV,",
                            part_kb[si],
                            ",",
                            direct_blocks[di],
                            ",",
                            relay_blocks[ri],
                            ",",
                            percents[pi],
                            ",",
                            ms,
                            ",",
                            gbps(part, ms),
                        )
        return

    print("| partition | reducescatter ms | GB/s | relay ms | GB/s | speedup |")
    for si in range(len(part_kb)):
        if part_kb[si] > max_mb * 1024:
            continue
        var part = part_kb[si] * 1024 // size_of[dtype]()
        var base_ms = run_baseline(part, None, "reducescatter")

        if table_recipe(part).relay_percent == 0:
            print(
                "|",
                human_readable_size(part * size_of[dtype]()),
                "|",
                base_ms,
                "|",
                gbps(part, base_ms),
                "| relay declined by the tuning table |",
            )
            continue

        var relay_ms = run_relay(part, None, "relay")
        print(
            "|",
            human_readable_size(part * size_of[dtype]()),
            "|",
            base_ms,
            "|",
            gbps(part, base_ms),
            "|",
            relay_ms,
            "|",
            gbps(part, relay_ms),
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
    _ = res_bufs^
