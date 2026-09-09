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
# Fuzz target: the MiniMax-M3 sparse-attention (MSA) indexer PREFILL path
# (`sparse_indexer_prefill_score` + `sparse_indexer_prefill_topk`, in
# Kernels/lib/msa/sparse_indexer_prefill.mojo). For each ragged query token it
# block-max-scores every KV block up to that token's CAUSAL boundary
# (prefix_len + local_index + 1), applies init/local forcing, and selects the
# top-k blocks. On B200 the scorer is the SM100 tensor-core (SS-UMMA) kernel with
# TMA-staged, software-pipelined K -- the heaviest bug surface in the indexer
# family; on MI355 it is the cooperative scalar path.
#
# Runtime fuzz axes (CaseSpec, all Int): batch, per-batch query (extend) lengths
# and prefix lengths (derived from min/max/pattern -- the QTILE=64 query modulus,
# the block-size causal/partial-block modulus, and the chunked-prefill prefix>0
# regime), topk, init/local block counts, a causality `plant`, value dist, seed.
# The kernel config (num_index_heads / idx_head_dim / block_size) is compile-time
# `-D` (default the M3 config 4 / 128 / 128).
#
# Oracles:
#  - `ref` (--check 1, default): an independent f32 host oracle recomputes each
#    token's per-block causal block-max and the init/local forcing, compares the
#    score buffer (tolerant for computed, exact for forced), and checks the top-k
#    output with the tie-robust invariant. The K buffer carries a sentinel pad
#    past the last key (over-read detector); `plant` additionally sets a large
#    key just past a token's causal boundary, so an intra-diagonal-block
#    causality violation reads it and produces a detectably-too-high score.
#    Emits FUZZ_NUMERIC_FAIL.
#  - VALIDITY CONTRACT (always on): out_idxs in-range / distinct / -1-tail, on a
#    fresh -2-poisoned output. Rides under diff / memcheck / redzone. Emits
#    FUZZ_CONTRACT_FAIL.
#  - `schedule` (--schedule N): re-run the score kernel N times on the same input
#    and flag any non-bit-exact score buffer (a query-tile / split-K race).
#
# `--inject N` (default 0; never set by the orchestrator) is the oracle canary:
#   1 = corrupt a selected index to a low block -> trips `ref`;
#   2 = write an out-of-range index -> trips the validity contract under diff.

from std.collections import Set
from max.gpu.host import DeviceContext
from std.math import ceildiv, max, min, sqrt
from std.random import randn, random_ui64, seed as set_seed
from std.sys.defines import get_defined_int

from layout import Idx, TileTensor, row_major

from msa.sparse_indexer_prefill import (
    sparse_indexer_prefill_score,
    sparse_indexer_prefill_topk,
)
from nn.attention.mha_operand import RaggedMHAOperand

from _fuzz import (
    VD_ALL_EQUAL,
    VD_NORMAL,
    VD_SPARSE,
    VD_UNIFORM,
    boundary_int,
    collect_args,
    fill_all_equal,
    fill_sparse,
    fill_uniform,
    flag,
    flag_int,
)

comptime kv_type = DType.bfloat16
comptime out_idx_type = DType.int32

comptime num_index_heads = get_defined_int["sidx_num_heads", 4]()
comptime idx_head_dim = get_defined_int["sidx_head_dim", 128]()
comptime block_size = get_defined_int["sidx_block_size", 128]()

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()

comptime INIT_SCORE = Float32(1.0e30)
comptime LOCAL_SCORE = Float32(1.0e29)
comptime SCORE_ATOL = Float32(1.0e-2)
comptime SCORE_RTOL = Float32(1.0e-2)
comptime PLANT_VAL = Float32(5.0)  # large planted key (matches the kernel test)
comptime QTILE = 64  # the prefill scorer's query-tile modulus

comptime PAT_EQUAL = 0
comptime PAT_RAMP = 1
comptime PAT_ALT = 2
comptime PAT_ONE_BIG = 3
comptime NUM_PATTERNS = 4


def _dist_name(d: Int) -> String:
    if d == VD_NORMAL:
        return "normal"
    if d == VD_SPARSE:
        return "sparse"
    if d == VD_ALL_EQUAL:
        return "all_equal"
    return "uniform"


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var batch: Int
    var extend_max: Int  # longest per-batch query (extend) length
    var extend_min: Int
    var prefix_max: Int  # longest per-batch cached prefix length
    var prefix_min: Int
    var pattern: Int  # disparity pattern for both extend and prefix (PAT_*)
    var topk: Int
    var init_blocks: Int
    var local_blocks: Int
    var plant: Int  # 1 = plant a large key past a token's causal boundary
    var dist: Int
    var seed: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "batch=",
            self.batch,
            " extend_max=",
            self.extend_max,
            " extend_min=",
            self.extend_min,
            " prefix_max=",
            self.prefix_max,
            " prefix_min=",
            self.prefix_min,
            " pattern=",
            self.pattern,
            " topk=",
            self.topk,
            " init_blocks=",
            self.init_blocks,
            " local_blocks=",
            self.local_blocks,
            " plant=",
            self.plant,
            " dist=",
            _dist_name(self.dist),
            " seed=",
            self.seed,
        )


def _derive(bs: Int, lo_in: Int, hi_in: Int, pattern: Int) -> List[Int]:
    """Per-batch values from (min, max, pattern); shared by extend and prefix.
    """
    var out = List[Int]()
    var hi = hi_in
    var lo = min(lo_in, hi)
    for i in range(bs):
        if pattern == PAT_EQUAL:
            out.append(hi)
        elif pattern == PAT_RAMP:
            if bs <= 1:
                out.append(hi)
            else:
                out.append(lo + (hi - lo) * i // (bs - 1))
        elif pattern == PAT_ALT:
            out.append(hi if (i % 2 == 0) else lo)
        else:  # PAT_ONE_BIG
            out.append(hi if i == 0 else lo)
    return out^


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var batch = boundary_int(1, 4, 1)
        var pattern = Int(random_ui64(0, UInt64(NUM_PATTERNS - 1)))
        var topk = boundary_int(1, 128, 16)
        var init_blocks = boundary_int(0, 4, 1)
        var local_blocks = boundary_int(0, 4, 1)
        var droll = Int(random_ui64(0, 3))
        var dist: Int
        if droll == 0:
            dist = VD_NORMAL
        elif droll == 1:
            dist = VD_SPARSE
        elif droll == 2:
            dist = VD_ALL_EQUAL
        else:
            dist = VD_UNIFORM
        var plant = Int(random_ui64(0, 2) == 0)  # ~1/3 of cases

        var extend_max: Int
        var extend_min: Int
        var prefix_max: Int
        var prefix_min: Int
        if Int(random_ui64(0, 1)) == 0:
            # Wide-extend regime: many query rows (diagonal causal masking,
            # multiple query tiles). Bounded so the host reference stays cheap.
            extend_max = boundary_int(1, 96, QTILE)
            extend_min = boundary_int(1, extend_max, QTILE)
            prefix_max = boundary_int(0, 4096, block_size)
            prefix_min = boundary_int(0, prefix_max, block_size)
        else:
            # Deep-prefix regime: few query rows but many blocks per token (the
            # block-stride loop, > block_dim blocks). Cheap host reference.
            extend_max = boundary_int(1, 4, 1)
            extend_min = boundary_int(1, extend_max, 1)
            prefix_max = boundary_int(0, 24576, block_size)
            prefix_min = boundary_int(0, prefix_max, block_size)

        var the_seed = Int(random_ui64(1, 1_000_000))
        specs.append(
            CaseSpec(
                batch,
                extend_max,
                extend_min,
                prefix_max,
                prefix_min,
                pattern,
                topk,
                init_blocks,
                local_blocks,
                plant,
                dist,
                the_seed,
            )
        )
    return specs^


def _fill_stable[
    dtype: DType
](span: Span[mut=True, Scalar[dtype], _], dist: Int):
    if dist == VD_SPARSE:
        fill_sparse(span, density=0.1, lo=0.0, hi=1.0)
    elif dist == VD_ALL_EQUAL:
        fill_all_equal(span, value=0.5)
    elif dist == VD_NORMAL:
        randn(span, mean=0.0, standard_deviation=0.5)
    else:  # VD_UNIFORM: non-negative (like the kernel test) so a planted key
        # reliably dominates its block max.
        fill_uniform(span, lo=0.0, hi=1.0)


def _host_block_score(
    q: ImmPointer[Scalar[kv_type], _],
    k: ImmPointer[Scalar[kv_type], _],
    t: Int,
    h: Int,
    blk: Int,
    key_off_b: Int,
    num_keys: Int,
    sm_scale: Float32,
) -> Float32:
    """Oracle block score: max over the block's in-range *causal* keys."""
    var key_start = blk * block_size
    var key_end = min(key_start + block_size, num_keys)
    var q_off = (t * num_index_heads + h) * idx_head_dim
    var blk_max = Float32(-3.0e38)
    for key in range(key_start, key_end):
        var k_off = (key_off_b + key) * idx_head_dim
        var dot = Float32(0)
        for d in range(idx_head_dim):
            dot += q[q_off + d].cast[.float32]() * k[k_off + d].cast[.float32]()
        var s = dot * sm_scale
        if s > blk_max:
            blk_max = s
    return blk_max


def _forced_score(
    raw: Float32, blk: Int, num_blocks: Int, init_blocks: Int, local_blocks: Int
) -> Float32:
    var local_start = max(0, num_blocks - local_blocks)
    var val = raw
    if blk < init_blocks:
        val = INIT_SCORE
    if blk >= local_start:
        val = LOCAL_SCORE
    return val


def _close(got: Float32, expect: Float32) -> Bool:
    var diff = abs(got - expect)
    return diff <= SCORE_ATOL + SCORE_RTOL * abs(expect)


def run_one_case(
    ctx: DeviceContext,
    spec: CaseSpec,
    check: Bool = False,
    schedule_repeats: Int = 0,
    inject: Int = 0,
) raises:
    var batch = spec.batch
    var topk = spec.topk
    var init_blocks = spec.init_blocks
    var local_blocks = spec.local_blocks
    var sm_scale = Float32(1.0) / sqrt(Float32(idx_head_dim))

    var extend = _derive(batch, spec.extend_min, spec.extend_max, spec.pattern)
    var prefix = _derive(batch, spec.prefix_min, spec.prefix_max, spec.pattern)

    # Ragged offsets: iro over query tokens, cro over keys (prefix + extend).
    var iro = List[Int]()
    var cro = List[Int]()
    iro.append(0)
    cro.append(0)
    for b in range(batch):
        iro.append(iro[b] + extend[b])
        cro.append(cro[b] + prefix[b] + extend[b])
    var total_q = iro[batch]
    var total_keys = cro[batch]

    var max_num_blocks = 0
    var max_seqlen_q = 0
    for b in range(batch):
        max_num_blocks = max(
            max_num_blocks, ceildiv(prefix[b] + extend[b], block_size)
        )
        max_seqlen_q = max(max_seqlen_q, extend[b])

    var q_n = total_q * num_index_heads * idx_head_dim
    var k_n = total_keys * idx_head_dim
    var k_pad = block_size * idx_head_dim  # over-read sentinel pad
    var score_n = num_index_heads * total_q * max_num_blocks
    var out_n = num_index_heads * total_q * topk

    # --- host inputs ---------------------------------------------------------
    var q_host = ctx.enqueue_create_host_buffer[kv_type](q_n)
    var k_host = ctx.enqueue_create_host_buffer[kv_type](k_n + k_pad)
    _fill_stable(q_host.as_span(), spec.dist)
    _fill_stable(Span(unsafe_ptr=k_host.unsafe_ptr(), length=k_n), spec.dist)
    for j in range(k_pad):
        k_host[k_n + j] = Scalar[kv_type](PLANT_VAL)

    # Causality plant: a large key just past some batch-0 tokens' causal bound.
    # A token in batch 0 with local_idx < extend[0]//2 must NOT see this key; if
    # the kernel reads it (intra-diagonal-block causality bug) the block max
    # jumps detectably above the f32 reference (which excludes it).
    if spec.plant == 1 and batch >= 1 and extend[0] >= 2:
        var plant_key = prefix[0] + extend[0] // 2  # within batch-0 key region
        for d in range(idx_head_dim):
            k_host[plant_key * idx_head_dim + d] = Scalar[kv_type](PLANT_VAL)

    var iro_host = ctx.enqueue_create_host_buffer[.uint32](batch + 1)
    var cro_host = ctx.enqueue_create_host_buffer[.uint32](batch + 1)
    var pl_host = ctx.enqueue_create_host_buffer[.uint32](batch)
    for b in range(batch + 1):
        iro_host[b] = UInt32(iro[b])
        cro_host[b] = UInt32(cro[b])
    for b in range(batch):
        pl_host[b] = UInt32(prefix[b])
    ctx.synchronize()

    # --- device buffers ------------------------------------------------------
    var q_dev = ctx.enqueue_create_buffer[kv_type](q_n)
    var k_dev = ctx.enqueue_create_buffer[kv_type](k_n + k_pad)
    var iro_dev = ctx.enqueue_create_buffer[.uint32](batch + 1)
    var cro_dev = ctx.enqueue_create_buffer[.uint32](batch + 1)
    var pl_dev = ctx.enqueue_create_buffer[.uint32](batch)
    var score_dev = ctx.enqueue_create_buffer[.float32](score_n)
    var out_dev = ctx.enqueue_create_buffer[out_idx_type](out_n)
    ctx.enqueue_copy(dst_buf=q_dev, src_buf=q_host)
    ctx.enqueue_copy(dst_buf=k_dev, src_buf=k_host)
    ctx.enqueue_copy(dst_buf=iro_dev, src_buf=iro_host)
    ctx.enqueue_copy(dst_buf=cro_dev, src_buf=cro_host)
    ctx.enqueue_copy(dst_buf=pl_dev, src_buf=pl_host)
    out_dev.enqueue_fill(Int32(-2))

    var q_t = TileTensor(
        q_dev, row_major((total_q, num_index_heads, idx_head_dim))
    )
    var iro_t = TileTensor(iro_dev, row_major(batch + 1))
    var pl_t = TileTensor(pl_dev, row_major(batch))
    var score_t = TileTensor(
        score_dev, row_major((num_index_heads, total_q, max_num_blocks))
    )
    var out_t = TileTensor(out_dev, row_major((num_index_heads, total_q, topk)))

    var k_buf = TileTensor[mut=False](
        k_dev, row_major((total_keys, Idx[1], Idx[idx_head_dim]))
    )
    var cro_buf = TileTensor[mut=False](cro_dev, row_major((batch + 1,)))
    var k_operand = RaggedMHAOperand(k_buf, cro_buf)

    # --- schedule oracle: re-run the score kernel and check bit-exact --------
    if schedule_repeats > 0:
        var ref_host = ctx.enqueue_create_host_buffer[.float32](score_n)
        var n_runs = max(2, schedule_repeats)
        for r in range(n_runs):
            score_dev.enqueue_fill(Float32(0))
            sparse_indexer_prefill_score[
                kv_type,
                type_of(k_operand),
                num_index_heads,
                idx_head_dim,
                block_size,
            ](
                q_t,
                k_operand,
                iro_t,
                pl_t,
                score_t,
                batch,
                total_q,
                max_seqlen_q,
                max_num_blocks,
                init_blocks,
                local_blocks,
                sm_scale,
                ctx,
            )
            var sh = ctx.enqueue_create_host_buffer[.float32](score_n)
            ctx.enqueue_copy(dst_buf=sh, src_buf=score_dev)
            ctx.synchronize()
            if r == 0:
                for i in range(score_n):
                    ref_host[i] = sh[i]
            else:
                for i in range(score_n):
                    if sh[i] != ref_host[i]:
                        print(
                            "FUZZ_NUMERIC_FAIL kind=score_nondeterminism run=",
                            r,
                            "idx=",
                            i,
                            "got=",
                            sh[i],
                            "ref=",
                            ref_host[i],
                        )
                        raise Error(
                            "prefill score kernel nondeterministic (race)"
                        )
        _ = q_dev
        _ = k_dev
        _ = iro_dev
        _ = cro_dev
        _ = pl_dev
        _ = score_dev
        _ = out_dev
        return

    # --- score kernel; snapshot the score buffer BEFORE top-k mutates it -----
    sparse_indexer_prefill_score[
        kv_type,
        type_of(k_operand),
        num_index_heads,
        idx_head_dim,
        block_size,
    ](
        q_t,
        k_operand,
        iro_t,
        pl_t,
        score_t,
        batch,
        total_q,
        max_seqlen_q,
        max_num_blocks,
        init_blocks,
        local_blocks,
        sm_scale,
        ctx,
    )
    var score_host = ctx.enqueue_create_host_buffer[.float32](score_n)
    ctx.enqueue_copy(dst_buf=score_host, src_buf=score_dev)
    ctx.synchronize()

    # --- top-k kernel --------------------------------------------------------
    sparse_indexer_prefill_topk[num_index_heads, block_size](
        iro_t,
        pl_t,
        score_t,
        out_t,
        batch,
        total_q,
        max_num_blocks,
        topk,
        ctx,
    )
    var out_host = ctx.enqueue_create_host_buffer[out_idx_type](out_n)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
    ctx.synchronize()

    if inject != 0:
        _apply_inject(inject, out_host.unsafe_ptr(), max_num_blocks)

    var q_hp = q_host.unsafe_ptr()
    var k_hp = k_host.unsafe_ptr()

    # --- score-buffer correctness (ref) --------------------------------------
    if check:
        for h in range(num_index_heads):
            for t in range(total_q):
                var b = _batch_of(t, batch, iro)
                var local_idx = t - iro[b]
                var num_keys = prefix[b] + local_idx + 1
                var num_blocks = ceildiv(num_keys, block_size)
                for blk in range(num_blocks):
                    var raw = _host_block_score(
                        q_hp, k_hp, t, h, blk, cro[b], num_keys, sm_scale
                    )
                    var expect = _forced_score(
                        raw, blk, num_blocks, init_blocks, local_blocks
                    )
                    var got = score_host[
                        (h * total_q + t) * max_num_blocks + blk
                    ]
                    if expect == INIT_SCORE or expect == LOCAL_SCORE:
                        if got != expect:
                            print(
                                "FUZZ_NUMERIC_FAIL kind=forced_score h=",
                                h,
                                "t=",
                                t,
                                "blk=",
                                blk,
                                "got=",
                                got,
                                "expect=",
                                expect,
                            )
                            raise Error("forced block score mismatch")
                    elif not _close(got, expect):
                        print(
                            "FUZZ_NUMERIC_FAIL kind=block_score h=",
                            h,
                            "t=",
                            t,
                            "blk=",
                            blk,
                            "num_blocks=",
                            num_blocks,
                            "got=",
                            got,
                            "expect=",
                            expect,
                            "plant=",
                            spec.plant,
                            "dist=",
                            _dist_name(spec.dist),
                        )
                        raise Error("computed block score mismatch")

    # --- top-k validity contract (always) + invariant (ref) ------------------
    for h in range(num_index_heads):
        for t in range(total_q):
            var b = _batch_of(t, batch, iro)
            var local_idx = t - iro[b]
            var num_keys = prefix[b] + local_idx + 1
            var num_blocks = ceildiv(num_keys, block_size)
            var k_batch = min(topk, num_blocks)
            var base = (h * total_q + t) * topk

            var got = Set[Int]()
            for j in range(k_batch):
                var idx = Int(out_host[base + j])
                if idx < 0 or idx >= num_blocks:
                    print(
                        "FUZZ_CONTRACT_FAIL kind=index_out_of_range h=",
                        h,
                        "t=",
                        t,
                        "pos=",
                        j,
                        "idx=",
                        idx,
                        "num_blocks=",
                        num_blocks,
                    )
                    raise Error("selected block index out of range")
                if idx in got:
                    print(
                        "FUZZ_CONTRACT_FAIL kind=duplicate_index h=",
                        h,
                        "t=",
                        t,
                        "idx=",
                        idx,
                    )
                    raise Error("duplicate selected block index")
                got.add(idx)

            for j in range(k_batch, topk):
                if Int(out_host[base + j]) != -1:
                    print(
                        "FUZZ_CONTRACT_FAIL kind=tail_not_sentinel h=",
                        h,
                        "t=",
                        t,
                        "pos=",
                        j,
                        "got=",
                        Int(out_host[base + j]),
                    )
                    raise Error("output tail must be -1")

            if check and k_batch < num_blocks:
                _check_topk_invariant(
                    q_hp,
                    k_hp,
                    t,
                    h,
                    cro[b],
                    num_keys,
                    num_blocks,
                    init_blocks,
                    local_blocks,
                    sm_scale,
                    got,
                )

    _ = q_dev
    _ = k_dev
    _ = iro_dev
    _ = cro_dev
    _ = pl_dev
    _ = score_dev
    _ = out_dev


def _batch_of(t: Int, batch: Int, iro: List[Int]) -> Int:
    var b = 0
    for bi in range(batch):
        if t < iro[bi + 1]:
            b = bi
            break
    return b


def _check_topk_invariant(
    q_hp: ImmPointer[Scalar[kv_type], _],
    k_hp: ImmPointer[Scalar[kv_type], _],
    t: Int,
    h: Int,
    key_off_b: Int,
    num_keys: Int,
    num_blocks: Int,
    init_blocks: Int,
    local_blocks: Int,
    sm_scale: Float32,
    got: Set[Int],
) raises:
    var sel_min = Float32(3.0e38)
    var non_max = Float32(-3.0e38)
    for blk in range(num_blocks):
        var raw = _host_block_score(
            q_hp, k_hp, t, h, blk, key_off_b, num_keys, sm_scale
        )
        var sc = _forced_score(raw, blk, num_blocks, init_blocks, local_blocks)
        if blk in got:
            if sc < sel_min:
                sel_min = sc
        else:
            if sc > non_max:
                non_max = sc
    if sel_min < non_max - SCORE_ATOL:
        print(
            "FUZZ_NUMERIC_FAIL kind=topk_invariant h=",
            h,
            "t=",
            t,
            "sel_min=",
            sel_min,
            "non_max=",
            non_max,
        )
        raise Error("selected block ranks below an excluded block")


def _apply_inject(
    inject: Int,
    out_host: MutPointer[Scalar[out_idx_type], _],
    max_num_blocks: Int,
):
    """Corrupt out_idxs[0] to prove an oracle can FAIL (positive control)."""
    if inject == 2:
        out_host[0] = Scalar[out_idx_type](max_num_blocks + 1)
        return
    out_host[0] = Scalar[out_idx_type](max_num_blocks - 1)


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var schedule_repeats = flag_int(args, "--schedule", 0)
    var inject = flag_int(args, "--inject", 0)
    set_seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "batch=",
                specs[i].batch,
                "extend_max=",
                specs[i].extend_max,
                "extend_min=",
                specs[i].extend_min,
                "prefix_max=",
                specs[i].prefix_max,
                "prefix_min=",
                specs[i].prefix_min,
                "pattern=",
                specs[i].pattern,
                "topk=",
                specs[i].topk,
                "init_blocks=",
                specs[i].init_blocks,
                "local_blocks=",
                specs[i].local_blocks,
                "plant=",
                specs[i].plant,
                "dist=",
                specs[i].dist,
                "seed=",
                specs[i].seed,
            )
        return

    if mode == "single":
        var spec = CaseSpec(
            flag_int(args, "--batch", 2),
            flag_int(args, "--extend_max", 300),
            flag_int(args, "--extend_min", 100),
            flag_int(args, "--prefix_max", 0),
            flag_int(args, "--prefix_min", 0),
            flag_int(args, "--pattern", PAT_EQUAL),
            flag_int(args, "--topk", 16),
            flag_int(args, "--init_blocks", 0),
            flag_int(args, "--local_blocks", 1),
            flag_int(args, "--plant", 0),
            flag_int(args, "--dist", VD_UNIFORM),
            flag_int(args, "--seed", the_seed),
        )
        print("FUZZ_SINGLE ", spec)
        with DeviceContext() as ctx:
            run_one_case(ctx, spec, check, schedule_repeats, inject)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_sparse_indexer_prefill heads=",
        num_index_heads,
        "dim=",
        idx_head_dim,
        "block=",
        block_size,
        "seed=",
        the_seed,
        "budget=",
        the_budget,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check, schedule_repeats, inject)
    print("=== done:", len(specs), "cases ===")
