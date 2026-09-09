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
# Fuzz target: EP MoE combine `send_tokens_back` write-offset containment
# (SERVOPT-1458). The combine kernel turns each routed token's `src_info` row
# `(src_idx, src_topk_idx)` into a 14KB P2P write offset into the destination
# device's combine receive buffer:
#
#     recv_buf_ptrs[dst] + recv_buf_layout((src_idx, src_topk_idx, 0))
#
# `src_info` is written by `dispatch_wait` ONLY for rows whose message header
# matches the local expert; a garbled / stale message leaves the row holding
# uninitialized pool bytes (it is a fresh tensor each step). Those bytes were
# then used UNVALIDATED as the write offset -- an arbitrary-offset 14KB write
# that, in the Kimi-K2.5-NVFP4 TPEP run, stomped model weights and turned into
# a permanent NaN factory (SERVOPT-1458).
#
# This target reproduces the *weapon* (the OOB write), not the upstream message
# garble (a cross-device timing effect the single-kernel fuzzer cannot model):
# it runs a real single-GPU dispatch to populate the per-expert offsets and a
# legitimate `src_info`, then OVERWRITES `src_info` with a fuzzed value
# distribution (valid / boundary / off-by-one / far-garbage / -1 sentinel /
# INT32_MAX), then launches `combine_async`. Under `--oracle memcheck`
# (or `redzone`) the out-of-bounds write is detected (verdict FAIL on the
# pre-containment kernel); the containment fix (sentinel-fill of unmatched
# `src_info` rows + a bounds check in `send_tokens_back`) skips the wild write,
# so the same case is clean (verdict PASS).
#
# Single GPU by construction: n_ranks = p2p_world_size = 1, so every
# destination is local (the same-node direct-copy branch -- exactly the branch
# that carried the bug) and the wild write lands in this device's own combine
# receive buffer where the redzone/memcheck oracle guards it.

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.random import randint, randn, seed
from std.sys import size_of
from std.sys.defines import get_defined_int

from layout import Idx, TileTensor
from layout.tile_layout import row_major
from shmem.ep_comm import (
    BF16TokenFormat,
    EPLocalSyncCounters,
    combine_async_kernel,
    dispatch_async_kernel,
    dispatch_wait_kernel,
)

from _fuzz import boundary_int, collect_args, flag, flag_int

# Shape knobs are compile-time (the kernels are parameterized on them, like
# fuzz_matmul's N/K). Defaults are small -- the bug is write-OFFSET driven, not
# shape driven, so tiny buffers make the OOB unambiguous and fast. Override with
# `-D ep_hidden_size=.. -D ep_top_k=.. -D ep_n_experts=.. -D ep_tokens=..`.
comptime hidden_size = get_defined_int["ep_hidden_size", 512]()
comptime top_k = get_defined_int["ep_top_k", 4]()
comptime n_experts = get_defined_int["ep_n_experts", 8]()
comptime n_tokens_per_rank = get_defined_int["ep_tokens", 16]()
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()

# Single device: every destination is local (the same-node direct-copy path).
comptime n_ranks = 1
comptime n_local_experts = n_experts // n_ranks
comptime max_recv_num_tokens = n_experts * n_tokens_per_rank

# `src_info` value distributions for the combine write offset. dist 0 is the
# control (leave dispatch's legitimate src_info -> PASS). dist >= 2 are the
# out-of-bounds classes a garbled/stale src_info row produces.
comptime SI_VALID = 0  # no corruption (control)
comptime SI_BOUNDARY = 1  # max valid (max_tokens-1, top_k-1) -> in bounds
comptime SI_ROW_OOB = 2  # src_idx == max_tokens_per_rank (one row past)
comptime SI_TOPK_OOB = 3  # src_topk_idx == top_k (one topk past)
comptime SI_GARBAGE = 4  # large positive garbage src_idx
comptime SI_SENTINEL = 5  # -1 (the production stale-row / fix sentinel)
comptime SI_INTMAX = 6  # INT32_MAX
comptime NUM_SI_DISTS = 7


def si_dist_name(d: Int) -> String:
    if d == SI_BOUNDARY:
        return "boundary"
    if d == SI_ROW_OOB:
        return "row_oob"
    if d == SI_TOPK_OOB:
        return "topk_oob"
    if d == SI_GARBAGE:
        return "garbage"
    if d == SI_SENTINEL:
        return "sentinel"
    if d == SI_INTMAX:
        return "intmax"
    return "valid"


def corrupt_src_info_kernel(
    src_info: MutPointer[Int32, MutAnyOrigin],
    n_rows_dev: Int32,
    bad_src_idx: Int32,
    bad_topk_idx: Int32,
):
    """Overwrites every src_info row with a chosen (src_idx, src_topk_idx).

    Models the production failure mode: dispatch leaves a row holding stale /
    garbled values that combine then uses as a write offset.
    """
    # `Int` is not device-passable; widen the fixed-width arg.
    var n_rows = Int(n_rows_dev)
    var gid = Int(global_idx.x)
    if gid < n_rows:
        src_info[gid * 2] = bad_src_idx
        src_info[gid * 2 + 1] = bad_topk_idx


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var dist: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("dist=", self.dist)


def gen_specs(n: Int) -> List[CaseSpec]:
    """Generates src_info distributions, biased toward the OOB classes."""
    var specs = List[CaseSpec]()
    for _ in range(n):
        # boundary_int over the dist enum keeps the control + boundary cases in
        # the mix while concentrating draws on the OOB classes (2..6).
        specs.append(CaseSpec(boundary_int(0, NUM_SI_DISTS - 1, SI_ROW_OOB)))
    return specs^


def bad_indices(dist: Int) raises -> Tuple[Int32, Int32]:
    """Maps a dist id to the (src_idx, src_topk_idx) to stamp into src_info."""
    if dist == SI_BOUNDARY:
        return (Int32(n_tokens_per_rank - 1), Int32(top_k - 1))
    if dist == SI_ROW_OOB:
        return (Int32(n_tokens_per_rank), Int32(0))
    if dist == SI_TOPK_OOB:
        return (Int32(0), Int32(top_k))
    if dist == SI_GARBAGE:
        return (Int32(1 << 20), Int32(0))
    if dist == SI_SENTINEL:
        return (Int32(-1), Int32(-1))
    if dist == SI_INTMAX:
        return (Int32.MAX, Int32(0))
    # SI_VALID: caller skips corruption.
    return (Int32(0), Int32(0))


def run_one_case(ctx: DeviceContext, spec: CaseSpec) raises:
    comptime input_type = DType.bfloat16
    comptime output_layout = row_major(
        (Idx[max_recv_num_tokens], Idx[hidden_size])
    )
    comptime token_fmt_type = BF16TokenFormat[
        output_layout=type_of(output_layout), hidden_size, top_k
    ]
    comptime msg_bytes = token_fmt_type.msg_size()
    comptime combine_msg_bytes = size_of[input_type]() * hidden_size

    comptime hw_info = type_of(ctx).default_device_info

    # ----- buffers (single device, single slot) -----
    var dispatch_send = ctx.enqueue_create_buffer[.uint8](
        n_tokens_per_rank * msg_bytes
    )
    var dispatch_recv = ctx.enqueue_create_buffer[.uint8](
        max_recv_num_tokens * msg_bytes
    )
    var dispatch_recv_count = ctx.enqueue_create_buffer[.uint64](n_experts)
    ctx.enqueue_memset(dispatch_recv_count, UInt64.MAX_FINITE)

    var combine_send = ctx.enqueue_create_buffer[.uint8](
        max_recv_num_tokens * combine_msg_bytes
    )
    # The OOB target: a garbage src_idx makes the combine write land outside
    # this buffer (redzone / memcheck guards it).
    var combine_recv = ctx.enqueue_create_buffer[.uint8](
        n_tokens_per_rank * top_k * combine_msg_bytes
    )
    var combine_recv_count = ctx.enqueue_create_buffer[.uint64](n_experts)
    ctx.enqueue_memset(combine_recv_count, UInt64.MAX_FINITE)

    var atomic_counters = ctx.enqueue_create_buffer[.int32](
        EPLocalSyncCounters[n_experts].total_size()
    )
    ctx.enqueue_memset(atomic_counters, Int32(0))

    var topk_ids_dev = ctx.enqueue_create_buffer[.int32](
        n_tokens_per_rank * top_k
    )
    var input_tokens_dev = ctx.enqueue_create_buffer[input_type](
        n_tokens_per_rank * hidden_size
    )
    var dispatch_out_dev = ctx.enqueue_create_buffer[input_type](
        max_recv_num_tokens * hidden_size
    )
    var row_offsets_dev = ctx.enqueue_create_buffer[.uint32](
        n_local_experts + 1
    )
    var expert_ids_dev = ctx.enqueue_create_buffer[.int32](n_local_experts)
    var src_info_dev = ctx.enqueue_create_buffer[.int32](
        max_recv_num_tokens * 2
    )

    # ----- legitimate inputs: random unique top-k expert ids + token values
    var host_topk = alloc[Int32](n_tokens_per_rank * top_k)
    var host_tokens = alloc[Scalar[input_type]](n_tokens_per_rank * hidden_size)
    seed(0)
    randint(host_topk, n_tokens_per_rank * top_k, 0, n_experts - 1)
    # Make each token's top-k expert ids unique (a token isn't sent to one
    # expert twice).
    for tok in range(n_tokens_per_rank):
        var base = host_topk + tok * top_k

        def dup() {imm} -> Int:
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    if base[i] == base[j]:
                        return i
            return -1

        var d = dup()
        while d != -1:
            randint(base + d, 1, 0, n_experts - 1)
            d = dup()
    randn(host_tokens, n_tokens_per_rank * hidden_size)
    ctx.enqueue_copy(topk_ids_dev, host_topk)
    ctx.enqueue_copy(input_tokens_dev, host_tokens)

    # ----- layouts / tiles -----
    var topk_ids_layout = row_major(n_tokens_per_rank, Idx[top_k])
    var input_tokens_layout = row_major((n_tokens_per_rank, Idx[hidden_size]))
    var out_layout = row_major((Idx[max_recv_num_tokens], Idx[hidden_size]))
    var row_offsets_layout = row_major[n_local_experts + 1]()
    var expert_ids_layout = row_major[n_local_experts]()
    var src_info_layout = row_major((Idx[max_recv_num_tokens], Idx[2]))

    var topk_ids_t = TileTensor[mut=False, .int32, type_of(topk_ids_layout)](
        ptr=topk_ids_dev.unsafe_ptr(), layout=topk_ids_layout
    )
    var input_tokens_t = TileTensor[
        mut=False, input_type, type_of(input_tokens_layout)
    ](ptr=input_tokens_dev.unsafe_ptr(), layout=input_tokens_layout)
    var out_t = TileTensor[input_type, type_of(out_layout)](
        ptr=dispatch_out_dev.unsafe_ptr(), layout=out_layout
    )
    var row_offsets_t = TileTensor[.uint32, type_of(row_offsets_layout)](
        ptr=row_offsets_dev.unsafe_ptr(), layout=row_offsets_layout
    )
    var expert_ids_t = TileTensor[.int32, type_of(expert_ids_layout)](
        ptr=expert_ids_dev.unsafe_ptr(), layout=expert_ids_layout
    )
    var src_info_t = TileTensor[.int32, type_of(src_info_layout)](
        ptr=src_info_dev.unsafe_ptr(), layout=src_info_layout
    )

    var format_handler = token_fmt_type(out_t)

    var recv_bufs = Array[MutPointer[UInt8, MutAnyOrigin], n_ranks](
        uninitialized=True
    )
    recv_bufs[0] = dispatch_recv.unsafe_ptr().as_unsafe_any_origin()
    var recv_count_bufs = Array[MutPointer[UInt64, MutAnyOrigin], n_ranks](
        uninitialized=True
    )
    recv_count_bufs[0] = dispatch_recv_count.unsafe_ptr().as_unsafe_any_origin()
    var combine_recv_bufs = Array[MutPointer[UInt8, MutAnyOrigin], n_ranks](
        uninitialized=True
    )
    combine_recv_bufs[0] = combine_recv.unsafe_ptr().as_unsafe_any_origin()
    var combine_recv_count_bufs = Array[
        MutPointer[UInt64, MutAnyOrigin], n_ranks
    ](uninitialized=True)
    combine_recv_count_bufs[
        0
    ] = combine_recv_count.unsafe_ptr().as_unsafe_any_origin()

    var counters = EPLocalSyncCounters[n_experts](atomic_counters.unsafe_ptr())

    # ----- kernels -----
    comptime dispatch_async = dispatch_async_kernel[
        input_type,
        hw_info.max_thread_block_size,
        type_of(input_tokens_layout),
        type_of(topk_ids_layout),
        hw_info.sm_count,
        n_experts,
        n_ranks,
        n_tokens_per_rank,
        n_ranks,  # p2p world size
        token_fmt_type,
        use_shmem=False,
    ]
    comptime dispatch_wait = dispatch_wait_kernel[
        hw_info.max_thread_block_size,
        type_of(row_offsets_layout),
        type_of(expert_ids_layout),
        type_of(src_info_layout),
        hw_info.sm_count,
        n_experts,
        n_ranks,
        n_tokens_per_rank,
        type_of(format_handler),
    ]
    comptime combine_async = combine_async_kernel[
        input_type,
        hw_info.max_thread_block_size,
        type_of(out_layout),
        type_of(src_info_layout),
        hw_info.sm_count,
        top_k,
        n_experts,
        n_ranks,
        combine_msg_bytes,
        n_tokens_per_rank,
        n_ranks,  # p2p world size
        use_shmem=False,
    ]

    # ----- run a real dispatch to populate offsets + a valid src_info -----
    ctx.enqueue_function[dispatch_async](
        input_tokens_t,
        topk_ids_t,
        dispatch_send.unsafe_ptr(),
        recv_bufs,
        recv_count_bufs,
        counters,
        Int32(0),
        grid_dim=hw_info.sm_count,
        block_dim=hw_info.max_thread_block_size,
    )
    ctx.enqueue_function[dispatch_wait](
        token_fmt_type(out_t),
        row_offsets_t,
        expert_ids_t,
        src_info_t,
        dispatch_recv.unsafe_ptr(),
        dispatch_recv_count.unsafe_ptr(),
        counters,
        Int32(0),
        grid_dim=hw_info.sm_count,
        block_dim=hw_info.max_thread_block_size,
    )
    ctx.synchronize()

    # ----- stamp the fuzzed src_info distribution (the stale-row condition) ---
    if spec.dist != SI_VALID:
        var pair = bad_indices(spec.dist)
        ctx.enqueue_function[corrupt_src_info_kernel](
            src_info_dev.unsafe_ptr(),
            Int32(max_recv_num_tokens),
            pair[0],
            pair[1],
            grid_dim=ceildiv(max_recv_num_tokens, 256),
            block_dim=256,
        )

    # ----- the kernel under test: combine send_tokens_back uses src_info as a
    # ----- write offset. A garbage offset is the SERVOPT-1458 wild write.
    ctx.enqueue_function[combine_async](
        out_t.as_immut(),
        src_info_t.as_immut(),
        combine_send.unsafe_ptr(),
        combine_recv_bufs,
        combine_recv_count_bufs,
        counters,
        Int32(0),
        grid_dim=hw_info.sm_count,
        block_dim=hw_info.max_thread_block_size,
    )
    ctx.synchronize()

    host_topk.free()
    host_tokens.free()
    _ = dispatch_send
    _ = dispatch_recv
    _ = dispatch_recv_count
    _ = combine_send
    _ = combine_recv
    _ = combine_recv_count
    _ = atomic_counters
    _ = topk_ids_dev
    _ = input_tokens_dev
    _ = dispatch_out_dev
    _ = row_offsets_dev
    _ = expert_ids_dev
    _ = src_info_dev


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print("FUZZ_SPEC idx=", i, "dist=", specs[i].dist)
        return

    if mode == "single":
        var dist = flag_int(args, "--dist", SI_ROW_OOB)
        print("FUZZ_SINGLE dist=", dist, "(", si_dist_name(dist), ")")
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(dist))
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_ep_combine seed=",
        the_seed,
        "budget=",
        the_budget,
        "hidden_size=",
        hidden_size,
        "top_k=",
        top_k,
        "n_experts=",
        n_experts,
        "n_tokens_per_rank=",
        n_tokens_per_rank,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print(
                "case", i, ": dist=", specs[i].dist, si_dist_name(specs[i].dist)
            )
            run_one_case(ctx, specs[i])
    print("=== done:", len(specs), "cases ===")
