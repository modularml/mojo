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
# Fuzz target: MiniMax-M3 block-sparse decode ATTENTION (`msa_sm100_decode`, in
# Kernels/lib/msa/msa_1q.mojo). This is the kernel that consumes the indexer's
# selected blocks and computes the attention (QK . softmax . V) -- distinct from
# the indexer (block selection), which `sparse_indexer_decode` fuzzes, and from
# the dense MLA decode (`mla_decode`).
#
# The accuracy hazard this targets is the partial-local-block tail mask (the M3
# long-context decode runaway, CENG-639, fixed in #90025): the indexer
# force-keeps the local block FULL (BN rows), but only `[block*BN, cache_length]`
# is valid; positions `> cache_length` in that block are an unwritten KV tail. If
# the decode mask does not bound the loaded slot by the logical position, that
# tail leaks into the softmax -- corrupting O whenever `cache_length + 1` is not
# BN-aligned (~127/128 of decode steps, severe at long context).
#
# Reproduction (mirrors the kernel's own regression test): per batch, a
# non-BN-aligned `cache_length` makes the last selected block a PARTIAL local
# block; its tail (logical position `> cache_length`) is filled with a 1e4
# POISON sentinel. The fuzz axes are batch, the per-batch cache_lengths (biased
# toward non-BN-aligned via the boundary generator), and topk. `head_dim`/`group`
# are compile-time (`-D head_dim=.. -D group=..`, default the M3 d128 / group 8;
# group >= 8 keeps a single head's poison dot from sign-cancelling).
# Dual-arch: SM100/B200 (msa_sm100_decode) + AMD MI355 (msa_amd_decode_dispatch).
#
# `determinism` oracle (--rerun N): the same input and the same default launch
# N times, into N distinct output buffers all enqueued before one synchronize
# so the decode kernels stay queued back-to-back under concurrent load. The
# split-K combine is a fixed sequential loop, so a correct kernel is bit-stable
# and the compare is exact (atol=rtol=0); a divergence is a race or an
# order-dependent atomic. This matters for the M3 decode because a per-request
# result that moves run to run also moves between the draft's proposal and the
# target's verify, which is exactly what a speculative acceptance test cannot
# tolerate.
#
# `ref` oracle (--check 1, default): an f64 block-restricted softmax that drops
# `k_logical > cache_length` (the tail). A kernel that attends the poison tail
# pushes O ~5 orders of magnitude out of band -> FUZZ_NUMERIC_FAIL. A NaN/Inf
# guard catches an all-poison softmax. BN-aligned `cache_length + 1` cases have
# no tail and pass on any kernel (built-in control).

from std.math import exp, isinf, isnan, sqrt
from std.random import randn, seed
from std.sys import (
    CompilationTarget,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)
from std.sys.defines import get_defined_int
from std.utils import IndexList

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import (
    Coord,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
)
from layout.tile_layout import row_major
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from nn.attention.mha_operand import KVCacheMHAOperand
from nn.attention.mha_mask import NullMask
from nn.attention.mha_utils import MHAConfig, StaticInt
from msa.msa_1q import msa_sm100_decode
from msa.amd.decode import msa_amd_decode_dispatch

from _fuzz import (
    boundary_int,
    collect_args,
    flag,
    flag_int,
    numeric_check,
)

comptime dtype = DType.bfloat16
comptime BN = 128  # block size (tokens) == page_size
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime LAYER_IDX = 0

comptime head_dim = get_defined_int["head_dim", 128]()
comptime group = get_defined_int["group", 8]()  # num_q_heads; >=8 (sign-stable)
comptime POISON = Scalar[dtype](1.0e4)

comptime MAX_BATCH = 6
comptime MAX_LOCAL_BLK = 24  # local block index cap (=> num_blocks <= 25)
comptime MAX_TOPK = 16

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var batch: Int
    var cl_seed: Int
    var topk: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "batch=",
            self.batch,
            " cl_seed=",
            self.cl_seed,
            " topk=",
            self.topk,
            " head_dim=",
            head_dim,
            " group=",
            group,
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var b = boundary_int(1, MAX_BATCH, 2)
        var cs = Int(boundary_int(1, 1 << 30, 1 << 20))
        var k = boundary_int(2, MAX_TOPK, 4)
        specs.append(CaseSpec(b, cs, k))
    return specs^


def run_one_case(
    ctx: DeviceContext,
    spec: CaseSpec,
    check: Bool = False,
    repeats: Int = 1,
) raises:
    comptime num_q_heads = group
    comptime kv_num_heads = 1
    comptime scale = Float32(1.0) / sqrt(Float32(head_dim))
    comptime scale_f64 = Float64(1.0) / sqrt(Float64(head_dim))

    var batch_size = spec.batch
    seed(spec.cl_seed)

    # Per-batch local block index + partial fill -> non-BN-aligned cache_length.
    # `partial` in [0, BN-1]; partial == BN-1 is the BN-aligned control (no tail).
    var local_blk = List[Int]()
    var num_blocks = List[Int]()
    var cache_length = List[Int]()
    var min_local = MAX_LOCAL_BLK
    for _ in range(batch_size):
        var lb = boundary_int(1, MAX_LOCAL_BLK, 8)
        var partial = boundary_int(0, BN - 1, 32)
        local_blk.append(lb)
        num_blocks.append(lb + 1)
        cache_length.append(lb * BN + partial)
        if lb < min_local:
            min_local = lb

    # topk <= min(num_blocks) and each local_blk >= topk-1 (interior selection):
    # both hold when topk <= min_local + 1.
    var topk = spec.topk
    if topk > min_local + 1:
        topk = min_local + 1
    if topk < 2:
        topk = 2
    var topk_tokens = topk * BN

    # Per-batch page bases (distinct physical ranges) + LUT reversal.
    var page_base = List[Int](length=batch_size + 1, fill=0)
    var num_pages = 0
    var max_pages = 0
    for b in range(batch_size):
        page_base[b + 1] = page_base[b] + num_blocks[b]
        num_pages += num_blocks[b]
        max_pages = max(max_pages, num_blocks[b])

    var q_size = batch_size * num_q_heads * head_dim
    var idx_count = batch_size * topk

    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    ctx.synchronize()
    randn(q_host.as_span())

    # Block selection: slot 0 = prefix block 0, last slot = the partial LOCAL
    # block, interior slots = blocks in [1, local-1]. Duplicates are harmless
    # (kernel and oracle both iterate the same idx).
    var idx_host = ctx.enqueue_create_host_buffer[.int32](idx_count)
    ctx.synchronize()
    for b in range(batch_size):
        var lblk = local_blk[b]
        for t in range(topk):
            var blk: Int
            if t == 0:
                blk = 0
            elif t == topk - 1:
                blk = lblk
            else:
                blk = 1 + ((t * 7) % max(1, lblk - 1))
            idx_host[b * topk + t] = Int32(blk)

    var lut = List[Int](length=batch_size * max_pages, fill=0)
    for b in range(batch_size):
        var np = num_blocks[b]
        for lb in range(np):
            lut[b * max_pages + lb] = page_base[b] + (np - 1 - lb)

    var kv_block_size = (
        num_pages * 2 * NUM_LAYERS * PAGE_SIZE * kv_num_heads * head_dim
    )
    var kv_host = ctx.enqueue_create_host_buffer[dtype](kv_block_size)
    ctx.synchronize()
    for i in range(kv_block_size):
        kv_host[i] = Scalar[dtype](0)

    var batch_tok_base = List[Int](length=batch_size + 1, fill=0)
    for b in range(batch_size):
        batch_tok_base[b + 1] = batch_tok_base[b] + num_blocks[b] * BN
    var total_logical = batch_tok_base[batch_size]

    var kv_rand = ctx.enqueue_create_host_buffer[dtype](
        total_logical * head_dim
    )
    var vv_rand = ctx.enqueue_create_host_buffer[dtype](
        total_logical * head_dim
    )
    ctx.synchronize()
    randn(kv_rand.as_span())
    randn(vv_rand.as_span())

    var k_f64 = List[Float64](length=total_logical * head_dim, fill=Float64(0))
    var v_f64 = List[Float64](length=total_logical * head_dim, fill=Float64(0))
    for b in range(batch_size):
        var cl = cache_length[b]
        var band = num_blocks[b] * BN
        for tok in range(band):
            var global_tok = batch_tok_base[b] + tok
            var lb = tok // PAGE_SIZE
            var off = tok % PAGE_SIZE
            var page = lut[b * max_pages + lb]
            var is_tail = tok > cl  # logical position past the valid prefix
            for d in range(head_dim):
                var kval: Scalar[dtype]
                var vval: Scalar[dtype]
                if is_tail:
                    kval = POISON
                    vval = POISON
                else:
                    kval = kv_rand[global_tok * head_dim + d]
                    vval = vv_rand[global_tok * head_dim + d]
                    k_f64[global_tok * head_dim + d] = kval.cast[
                        DType.float64
                    ]()
                    v_f64[global_tok * head_dim + d] = vval.cast[
                        DType.float64
                    ]()
                var k_off = (
                    ((page * 2 + 0) * NUM_LAYERS + LAYER_IDX) * PAGE_SIZE + off
                ) * kv_num_heads * head_dim + d
                var v_off = (
                    ((page * 2 + 1) * NUM_LAYERS + LAYER_IDX) * PAGE_SIZE + off
                ) * kv_num_heads * head_dim + d
                kv_host[k_off] = kval
                kv_host[v_off] = vval

    var q_dev = ctx.enqueue_create_buffer[dtype](q_size)
    var idx_dev = ctx.enqueue_create_buffer[.int32](idx_count)
    var o_dev = ctx.enqueue_create_buffer[dtype](q_size)
    var kv_block_dev = ctx.enqueue_create_buffer[dtype](kv_block_size)
    var cl_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var lut_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size * max_pages
    )
    ctx.synchronize()
    for b in range(batch_size):
        cl_host[b] = UInt32(cache_length[b])
        for lb in range(max_pages):
            lut_host[b * max_pages + lb] = UInt32(lut[b * max_pages + lb])
    var cl_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    var lut_dev = ctx.enqueue_create_buffer[.uint32](batch_size * max_pages)

    # Poison O so an unwritten output slot is caught too.
    var o_init = ctx.enqueue_create_host_buffer[dtype](q_size)
    ctx.synchronize()
    for i in range(q_size):
        o_init[i] = Scalar[dtype](7)

    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.enqueue_copy(o_dev, o_init)
    ctx.enqueue_copy(kv_block_dev, kv_host)
    ctx.enqueue_copy(cl_dev, cl_host)
    ctx.enqueue_copy(lut_dev, lut_host)

    # One output buffer per launch: the reruns must not serialize on a single
    # buffer (WAW), so every launch stays live concurrently. That is the
    # regime an order-dependent atomic or a reused split-K scratch shows up
    # in; a single launch-sync-launch loop would hide it. Each buffer carries
    # the same poison fill, so an unwritten slot still fails the rerun compare.
    var outs = List[DeviceBuffer[dtype]](capacity=repeats)
    outs.append(o_dev)
    for _ in range(1, repeats):
        var extra = ctx.enqueue_create_buffer[dtype](q_size)
        ctx.enqueue_copy(extra, o_init)
        outs.append(extra)

    comptime kv_block_layout = Layout.row_major[6]()
    var kv_block_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_dev,
        RuntimeLayout[kv_block_layout].row_major(
            IndexList[6](
                num_pages, 2, NUM_LAYERS, PAGE_SIZE, kv_num_heads, head_dim
            )
        ),
    )
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cl_tensor = LayoutTensor[mut=False, .uint32, cl_layout](
        cl_dev,
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )
    comptime lut_layout = Layout.row_major[2]()
    var lut_tensor = LayoutTensor[mut=False, .uint32, lut_layout](
        lut_dev,
        RuntimeLayout[lut_layout].row_major(
            IndexList[2](batch_size, max_pages)
        ),
    )

    var kv_collection = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(num_heads=kv_num_heads, head_size=head_dim),
        PAGE_SIZE,
    ](
        kv_block_tensor.as_unsafe_any_origin(),
        cl_tensor,
        lut_tensor,
        UInt32(total_logical),  # max_seq_length
        UInt32(total_logical),  # max_cache_length
    )
    var k_op = KVCacheMHAOperand(kv_collection.get_key_cache(LAYER_IDX))
    var v_op = KVCacheMHAOperand(kv_collection.get_value_cache(LAYER_IDX))

    comptime config = MHAConfig[dtype](num_q_heads, head_dim)

    # Ragged Q offsets [batch+1], one decode token per batch (iro[b] = b).
    var iro_host = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    ctx.synchronize()
    for b in range(batch_size + 1):
        iro_host[b] = UInt32(b)
    var iro_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(iro_dev, iro_host)
    var valid_length = DeviceBuffer[.uint32](
        ctx, iro_dev.unsafe_ptr(), batch_size + 1, owning=False
    )

    # Production decode call (msa.mojo): ragged, no valid_key, no causal/spec,
    # mask_unselected=True.
    # Every launch is enqueued before the single synchronize below, so the
    # decode kernels stay queued back-to-back under concurrent load.
    for rep in range(len(outs)):
        comptime if has_amd_gpu_accelerator():
            msa_amd_decode_dispatch[
                config=config,
                group=group,
                ragged=True,
                _is_cache_length_accurate=False,
                mask_unselected=True,
            ](
                outs[rep],
                q_dev,
                k_op,
                v_op,
                TileTensor(
                    idx_dev.unsafe_ptr(), row_major(Coord(len(idx_dev)))
                ).as_immut(),
                topk,  # indices_stride (blocks)
                batch_size,  # num_rows_q (1 token/seq)
                NullMask(),
                valid_length,  # ragged Q offsets
                StaticInt[1](),  # max_prompt_len (decode)
                topk_tokens,  # max_cache_valid_length
                scale,
                None,  # kv_input_row_offsets
                batch_size,
                ctx,
            )
        elif has_nvidia_gpu_accelerator():
            msa_sm100_decode[
                config=config,
                group=group,
                ragged=True,
                _is_cache_length_accurate=False,
                mask_unselected=True,
            ](
                outs[rep],
                q_dev,
                k_op,
                v_op,
                TileTensor(
                    idx_dev.unsafe_ptr(), row_major(Coord(len(idx_dev)))
                ).as_immut(),
                Int32(topk),  # indices_stride (blocks)
                Int32(batch_size),  # num_rows_q (1 token/seq)
                NullMask(),
                valid_length,  # ragged Q offsets
                StaticInt[1](),  # max_prompt_len (decode)
                Int32(topk_tokens),  # max_cache_valid_length
                scale,
                None,  # kv_input_row_offsets
                Int32(batch_size),
                ctx,
            )
        else:
            CompilationTarget.unsupported_target_error[
                operation=__get_current_function_name()
            ]()

    ctx.synchronize()

    # Determinism oracle (`--rerun N`): identical input, identical launch, N
    # times. The decode's split-K combine is a fixed sequential loop, so a
    # correct kernel is bit-stable and the compare is exact (atol=rtol=0).
    # A divergence is a real race or an order-dependent atomic, never a
    # legitimate reduction-order wobble. The comparison lives here rather
    # than in the orchestrator because a subprocess may only issue one
    # verdict, so it can never hold two cases' outputs.
    if repeats > 1:
        var first_h = ctx.enqueue_create_host_buffer[dtype](q_size)
        ctx.enqueue_copy(first_h, outs[0])
        ctx.synchronize()
        for r in range(1, len(outs)):
            var rep_h = ctx.enqueue_create_host_buffer[dtype](q_size)
            ctx.enqueue_copy(rep_h, outs[r])
            ctx.synchronize()
            if not numeric_check(
                rep_h.as_span(), first_h.as_span(), atol=0.0, rtol=0.0
            ):
                print("FUZZ_CONTRACT_FAIL kind=nondeterminism rerun=", r)
                raise Error(
                    "MSA decode run-to-run nondeterminism"
                    " (determinism oracle, default launch)"
                )

    if not check:
        _ = q_dev
        _ = idx_dev
        _ = kv_block_dev
        _ = cl_dev
        _ = lut_dev
        _ = iro_dev
        _ = outs^
        return

    var o_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    ctx.enqueue_copy(o_host, o_dev)
    ctx.synchronize()

    # NaN/Inf guard (an all-poison softmax can overflow).
    for i in range(q_size):
        var x = o_host[i].cast[.float32]()
        if isnan(x) or isinf(x):
            print("FUZZ_NUMERIC_FAIL kind=naninf idx=", i, "val=", x)
            raise Error("MSA decode output NaN/Inf (poison tail attended?)")

    # f64 block-restricted softmax, masking the local tail (k_logical > cl).
    @__parameter
    def kv_off(b: Int, blk: Int, c: Int) -> Int:
        return (batch_tok_base[b] + blk * BN + c) * head_dim

    var max_abs = Float64(0)
    var max_rel = Float64(0)
    for b in range(batch_size):
        var cl = cache_length[b]
        for h in range(num_q_heads):
            var idx_base = b * topk
            var q_base = (b * num_q_heads + h) * head_dim
            var logits = List[Float64]()
            var slot_lin = List[Int]()
            for t in range(topk):
                var blk = Int(idx_host[idx_base + t])
                var blk_start = blk * BN
                for c in range(BN):
                    var k_logical = blk_start + c
                    if k_logical > cl:  # tail mask
                        continue
                    var dot = Float64(0)
                    var ko = kv_off(b, blk, c)
                    for d in range(head_dim):
                        dot += (
                            q_host[q_base + d].cast[.float64]() * k_f64[ko + d]
                        )
                    logits.append(dot * scale_f64)
                    slot_lin.append(k_logical)
            var ncols = len(logits)
            if ncols == 0:
                continue
            var mx = Float64(-1e300)
            for i in range(ncols):
                mx = max(mx, logits[i])
            var sm = Float64(0)
            for i in range(ncols):
                sm += exp(logits[i] - mx)
            for d in range(head_dim):
                var acc = Float64(0)
                for i in range(ncols):
                    var w = exp(logits[i] - mx) / sm
                    var bl = slot_lin[i] // BN
                    var c = slot_lin[i] % BN
                    acc += w * v_f64[kv_off(b, bl, c) + d]
                var got = o_host[q_base + d].cast[.float64]()
                var ae = abs(got - acc)
                max_abs = max(max_abs, ae)
                if abs(acc) > 0.1:
                    max_rel = max(max_rel, ae / abs(acc))

    if max_abs > 2e-2 or max_rel > 4e-2:
        print(
            "FUZZ_NUMERIC_FAIL kind=tail_leak max_abs=",
            max_abs,
            "max_rel=",
            max_rel,
        )
        raise Error("MSA decode local-tail mask mismatch (CENG-639 class)")

    _ = q_dev
    _ = idx_dev
    _ = kv_block_dev
    _ = cl_dev
    _ = lut_dev
    _ = iro_dev
    _ = outs^


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var rerun = flag_int(args, "--rerun", 0)
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "batch=",
                specs[i].batch,
                "cl_seed=",
                specs[i].cl_seed,
                "topk=",
                specs[i].topk,
            )
        return

    if mode == "single":
        var b = flag_int(args, "--batch", 2)
        var cs = flag_int(args, "--cl_seed", 1)
        var k = flag_int(args, "--topk", 16)
        print("FUZZ_SINGLE batch=", b, "cl_seed=", cs, "topk=", k)
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(b, cs, k), check, max(1, rerun))
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_msa_decode seed=",
        the_seed,
        "budget=",
        the_budget,
        "head_dim=",
        head_dim,
        "group=",
        group,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check, max(1, rerun))
    print("=== done:", len(specs), "cases ===")
