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
# Per-kernel fuzz target: fused_qk_rms_norm_ragged_paged
# (the MiniMax-M3 per-head QK RMSNorm over Q and the newly-written K rows in the
#  paged KV cache, op `mo.fused_qk_rms_norm.ragged.paged`; kernel in
#  nn/kv_cache.mojo). Distinct from the MLA `fused_rope_rmsnorm` target: here the
#  cache is a standard MHA-style paged cache (num_kv_heads > 1, K/V dim, is_mla=
#  False) and the kernel RMSNorms the FULL head_dim of Q AND K (no RoPE).
#
# In one launch the kernel (1) per-head RMSNorms each Q row over head_dim,
# writing q_output, and (2) per-head RMSNorms each newly-written K row in the
# paged cache IN PLACE (it reads the K row the prior store wrote, normalizes,
# stores back). Both use the same math:
#   norm_factor = rsqrt(mean(x^2) + eps)
#   out = (x_accum * norm_factor * (gamma_accum + weight_offset)).cast[dtype]()
# with accum=fp32, `multiply_before_cast=True`, `weight_offset` a RUNTIME scalar
# (M3 = 1.0, Gemma-style). `multiply_before_cast` is COMPILE-TIME (M3 = True).
#
# The fuzzable surface is the *ragged shape*, which drives every runtime index:
#   - batch size + per-batch new-token counts (`input_row_offsets`): the
#     per-row batch lookup (get_batch_from_row_offsets) and the grid (one block
#     per combined Q-or-K row);
#   - per-batch cache lengths (the existing K extent): `cache_token_idx =
#     token_idx + cache_length(b)` selects the K row to normalize in place and
#     drives the paged store position (the `% PAGE_SIZE` page-crossing edge is
#     the interesting modulus);
#   - num_q_heads / num_kv_heads are compile-time `-D` (one kernel instantiation
#     per build, kept small by default for a fast/cheap ref oracle), head_dim is
#     fixed at the M3 dense value 128.
#
#   -D fqrn_num_q_heads=16 -D fqrn_num_kv_heads=4   [default: small + fast ref]
#   -D fqrn_num_q_heads=64 -D fqrn_num_kv_heads=4   M3 dense attention (layers 0-2)
#
# Three argv modes so the Python orchestrator can drive it with a per-case
# timeout + process isolation (a hanging case only kills its own subprocess):
#
#   --mode list-specs --seed S --budget B
#       Print generated specs (`FUZZ_SPEC ...` lines), no GPU work.
#   --mode single --batch_size .. --max_cache_len .. ... [--check 1]
#       Run exactly one case (orchestration / shrinking / corpus replay).
#       Prints `FUZZ_RESULT verdict=PASS`; a hang times out; a crash exits != 0.
#   --mode fuzz --seed S --budget B   (default)
#       Generate + run a batch in-process (standalone convenience).
#
# Oracles: memory-safety (memcheck/redzone/poison/initcheck) is the default --
# the kernel reads + writes the paged K cache in place at a ragged-driven offset.
# `ref` (--check 1) is a higher-precision CPU oracle over BOTH outputs: the
# dense RMSNorm'd query (`q_output`, self-contained) and the paged K cache read
# back through its page table (the in-place-normalized K rows).

from std.math import align_up, ceildiv, max, min, rsqrt, sqrt
from std.random import randn, random_ui64, seed
from std.sys.defines import get_defined_int

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.utils.index import IndexList
from nn.kv_cache import fused_qk_rms_norm_ragged_paged

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
    numeric_check,
)


# ===----------------------------------------------------------------------=== #
# Fixed M3 config. num_q_heads / num_kv_heads are -D-overridable.
# ===----------------------------------------------------------------------=== #

comptime HEAD_DIM = 128  # M3 dense head_dim == rms_norm_cols == head_size
comptime PAGE_SIZE = 128  # paged-cache page size; the interesting cache modulus
comptime NUM_LAYERS = 1

comptime num_q_heads = get_defined_int["fqrn_num_q_heads", 16]()
comptime num_kv_heads = get_defined_int["fqrn_num_kv_heads", 4]()

comptime dtype = DType.bfloat16  # q / q_output / K-cache dtype (must match)
comptime gamma_dtype = DType.bfloat16  # RMSNorm gamma weights (same dtype as q)
comptime EPS = Float32(1e-6)
comptime WEIGHT_OFFSET = Float32(1.0)  # M3 Gemma-style runtime offset
comptime multiply_before_cast = True  # M3

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()
comptime TILE = PAGE_SIZE  # boundary modulus for the cache-length axis

# Cache-length disparity patterns (deterministic from the scalar spec, so a
# `single` case reproduces a ragged batch exactly without serializing a list).
comptime PAT_EQUAL = 0  # all batches == max_cache_len
comptime PAT_RAMP = 1  # linear ramp min..max across the batch
comptime PAT_ALT = 2  # alternating max / min (extreme within-batch disparity)
comptime PAT_ONE_BIG = 3  # one max, the rest min
comptime NUM_PATTERNS = 4


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    """One fuzz case: the runtime-varied ragged QK-RMSNorm shape.

    Shape is fully determined by these scalars (so `single` mode reproduces a
    ragged batch exactly); `seed`/`dist` only pick input *values* (which matter
    for the `ref` oracle, not for the memory-safety oracles).
    """

    var batch_size: Int
    var max_cache_len: Int  # max per-batch existing K extent (start position)
    var min_cache_len: Int
    var seq_len: Int  # max new Q/K tokens per batch
    var pattern: Int  # cache-length disparity pattern (PAT_*)
    var ragged_q: Int  # 0: every batch has seq_len new tokens; 1: ragged
    var seed: Int
    var dist: Int  # value distribution (VD_*); ref-oracle only

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "batch_size=",
            self.batch_size,
            " max_cache_len=",
            self.max_cache_len,
            " min_cache_len=",
            self.min_cache_len,
            " seq_len=",
            self.seq_len,
            " pattern=",
            self.pattern,
            " ragged_q=",
            self.ragged_q,
            " seed=",
            self.seed,
            " dist=",
            self.dist,
        )


# ===----------------------------------------------------------------------=== #
# Deterministic shape derivation from the scalar spec
# ===----------------------------------------------------------------------=== #


def derive_cache_lengths(spec: CaseSpec) -> List[Int]:
    """Per-batch existing-cache lengths (== K-cache start positions)."""
    var out = List[Int]()
    var bs = spec.batch_size
    var hi = spec.max_cache_len
    var lo = min(spec.min_cache_len, hi)
    for i in range(bs):
        if spec.pattern == PAT_EQUAL:
            out.append(hi)
        elif spec.pattern == PAT_RAMP:
            if bs <= 1:
                out.append(hi)
            else:
                out.append(lo + (hi - lo) * i // (bs - 1))
        elif spec.pattern == PAT_ALT:
            out.append(hi if (i % 2 == 0) else lo)
        else:  # PAT_ONE_BIG
            out.append(hi if i == 0 else lo)
    return out^


def derive_seq_lens(spec: CaseSpec) -> List[Int]:
    """Per-batch new-token counts. ragged_q varies them in [1, seq_len]."""
    var out = List[Int]()
    var sl = max(1, spec.seq_len)
    for i in range(spec.batch_size):
        if spec.ragged_q == 0:
            out.append(sl)
        else:
            out.append(1 + (i % sl))
    return out^


def gen_specs(n: Int) -> List[CaseSpec]:
    """Generate `n` boundary-aware ragged QK-RMSNorm cases.

    The dominant axes are `max_cache_len` (the K-cache start position,
    boundary-biased around PAGE_SIZE where `cache_token_idx % PAGE_SIZE` crosses
    a page in the K-cache store/load) and `seq_len` (the new-token span --
    boundary-biased around PAGE_SIZE so a write range straddles pages).
    `max_cache_len == 0` is the fresh-prefill edge (no existing cache).
    """
    var specs = List[CaseSpec]()
    for _ in range(n):
        var batch_size = boundary_int(1, 16, 4)
        var max_cache_len = boundary_int(0, 4096, TILE)
        var min_cache_len = boundary_int(0, max_cache_len, TILE)
        # Mix of prefill spans and single-token (decode-style append) cases to
        # hit the seq_len==1 grid edge.
        var seq_len: Int
        if Int(random_ui64(0, 3)) == 0:
            seq_len = 1
        else:
            seq_len = boundary_int(1, 256, TILE)
        var pattern = Int(random_ui64(0, UInt64(NUM_PATTERNS - 1)))
        var ragged_q = 1 if (seq_len > 1 and random_ui64(0, 1) == 1) else 0
        var the_seed = Int(random_ui64(0, 1_000_000))
        # Stable, finite value distributions only (ref-oracle safe).
        var droll = Int(random_ui64(0, 3))
        var dist: Int
        if droll == 0:
            dist = VD_NORMAL
        elif droll == 1:
            dist = VD_UNIFORM
        elif droll == 2:
            dist = VD_SPARSE
        else:
            dist = VD_ALL_EQUAL
        specs.append(
            CaseSpec(
                batch_size,
                max_cache_len,
                min_cache_len,
                seq_len,
                pattern,
                ragged_q,
                the_seed,
                dist,
            )
        )
    return specs^


# ===----------------------------------------------------------------------=== #
# Value fill (stable / finite -- safe for the numerical reference oracle)
# ===----------------------------------------------------------------------=== #


def fill_stable[
    dtype: DType
](span: Span[mut=True, Scalar[dtype], _], dist: Int):
    """Fill with a finite, small-magnitude distribution (|x| ~ 0.5).

    Small magnitudes keep the RMSNorm output comfortably inside bf16 range so
    the higher-precision ref-diff measures real error, not saturation. The
    NaN/Inf/large distributions are intentionally excluded here (they belong to
    a finiteness-contract oracle, not a tolerance diff). VD_SPARSE risks an
    all-zero row -> rsqrt(eps) blow-up; the small density keeps rows nonzero in
    aggregate, and the ref recomputes the same rsqrt so it stays in band.
    """
    if dist == VD_SPARSE:
        fill_sparse(span, density=0.3, lo=-0.5, hi=0.5)
    elif dist == VD_ALL_EQUAL:
        fill_all_equal(span, value=0.5)
    elif dist == VD_UNIFORM:
        fill_uniform(span, lo=-0.5, hi=0.5)
    else:  # VD_NORMAL (default)
        randn(span, mean=0.0, standard_deviation=0.5)


# ===----------------------------------------------------------------------=== #
# One case: build the paged collection + launch the fused QK RMSNorm kernel.
# ===----------------------------------------------------------------------=== #


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    seed(spec.seed)
    var cache_lengths = derive_cache_lengths(spec)
    var seq_lens = derive_seq_lens(spec)
    var batch_size = spec.batch_size

    var q_max_seq_len = 0
    var total_q_tokens = 0
    var max_cache_len = 0
    var max_full = 1  # max (cache_len + seq_len): K-cache extent
    var total_pages = 0
    for i in range(batch_size):
        if seq_lens[i] > q_max_seq_len:
            q_max_seq_len = seq_lens[i]
        total_q_tokens += seq_lens[i]
        if cache_lengths[i] > max_cache_len:
            max_cache_len = cache_lengths[i]
        if cache_lengths[i] + seq_lens[i] > max_full:
            max_full = cache_lengths[i] + seq_lens[i]
        total_pages += ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)

    comptime kv_params = KVCacheStaticParams(
        num_heads=num_kv_heads, head_size=HEAD_DIM
    )
    comptime KV_DIM = 2  # standard MHA paged cache: dim[1] = {K, V}

    var block_shape = IndexList[6](
        total_pages,
        KV_DIM,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * KV_DIM
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    # --- paged KV cache blocks. The kernel READS the existing K rows the prior
    # store wrote, normalizes, and writes back IN PLACE. We pre-fill the entire
    # K-cache region (the newly-written rows) with the K input values; the
    # kernel reads exactly the rows [cache_len, cache_len+seq_len) and normalizes
    # them. Initialize all blocks so reads are well-defined under poison/init.
    var blocks_host = ctx.enqueue_create_host_buffer[dtype](block_elems)
    fill_stable(blocks_host.as_span(), VD_NORMAL)

    # The K input we expect to see normalized: regenerate the K rows that the
    # kernel will read in place, and write them into the right paged slots so the
    # kernel normalizes a known value. (Mirrors production: store_k_cache_ragged
    # has already written the projected K before this kernel runs.)
    var k_input_host = ctx.enqueue_create_host_buffer[dtype](
        max(1, total_q_tokens * num_kv_heads * HEAD_DIM)
    )
    fill_stable(k_input_host.as_span(), spec.dist)

    # Scatter k_input into the paged blocks at each token's K-cache slot.
    comptime kv_stride = KV_DIM * NUM_LAYERS * PAGE_SIZE * num_kv_heads * HEAD_DIM
    comptime layer_stride = PAGE_SIZE * num_kv_heads * HEAD_DIM
    comptime tok_stride = num_kv_heads * HEAD_DIM
    var page_base = List[Int]()
    var pb = 0
    var tok = 0
    for b in range(batch_size):
        page_base.append(pb)
        pb += ceildiv(cache_lengths[b] + seq_lens[b], PAGE_SIZE)
        for s in range(seq_lens[b]):
            var post = cache_lengths[b] + s
            var physical_page = page_base[b] + post // PAGE_SIZE
            # K cache is dim[1] index 0; layer 0.
            var slot = (
                physical_page * kv_stride
                + 0 * (NUM_LAYERS * layer_stride)
                + 0 * layer_stride
                + (post % PAGE_SIZE) * tok_stride
            )
            for h in range(num_kv_heads):
                for d in range(HEAD_DIM):
                    blocks_host[slot + h * HEAD_DIM + d] = k_input_host[
                        (tok * num_kv_heads + h) * HEAD_DIM + d
                    ]
            tok += 1

    # --- q_proj input + gamma ------------------------------------------------
    var q_size = max(1, total_q_tokens * num_q_heads * HEAD_DIM)
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    fill_stable(q_host.as_span(), spec.dist)

    var q_gamma_host = ctx.enqueue_create_host_buffer[gamma_dtype](HEAD_DIM)
    fill_uniform(q_gamma_host.as_span(), lo=-0.5, hi=0.5)
    var k_gamma_host = ctx.enqueue_create_host_buffer[gamma_dtype](HEAD_DIM)
    fill_uniform(k_gamma_host.as_span(), lo=-0.5, hi=0.5)

    # --- cache_lengths + paged lookup table ----------------------------------
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](
        max(1, batch_size)
    )
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var max_pages_per_batch = align_up(ceildiv(max_full, PAGE_SIZE), 8)
    var lut_size = max(1, batch_size * max_pages_per_batch)
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for i in range(batch_size):
        var num_pages_i = ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # --- input_row_offsets (ragged Q prefix sum) -----------------------------
    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(seq_lens[i])

    # --- copy to device ------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[dtype](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var q_device = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var qout_device = ctx.enqueue_create_buffer[dtype](q_size)
    var q_gamma_device = ctx.enqueue_create_buffer[gamma_dtype](HEAD_DIM)
    ctx.enqueue_copy(q_gamma_device, q_gamma_host)
    var k_gamma_device = ctx.enqueue_create_buffer[gamma_dtype](HEAD_DIM)
    ctx.enqueue_copy(k_gamma_device, k_gamma_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
        max(1, batch_size)
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    # --- PagedKVCacheCollection ----------------------------------------------
    var blocks_lt = LayoutTensor[dtype, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_lt = LayoutTensor[.uint32, cl_layout](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )
    comptime lt_layout_2d = Layout.row_major[2]()
    var lookup_table_lt = LayoutTensor[.uint32, lt_layout_2d](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, max_pages_per_batch)
        ),
    )

    var kv_collection = PagedKVCacheCollection[dtype, kv_params, PAGE_SIZE](
        LayoutTensor[dtype, Layout.row_major[6]()](
            blocks_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks_lt.runtime_layout.shape.value,
                blocks_lt.runtime_layout.stride.value,
            ),
        ).as_unsafe_any_origin(),
        LayoutTensor[mut=False, .uint32, cl_layout](
            cache_lengths_lt.ptr,
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ).as_unsafe_any_origin(),
        LayoutTensor[mut=False, .uint32, lt_layout_2d](
            lookup_table_lt.ptr,
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ).as_unsafe_any_origin(),
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )

    # --- kernel tensors ------------------------------------------------------
    var q_proj = TileTensor(
        q_device, row_major((total_q_tokens, Idx[num_q_heads], Idx[HEAD_DIM]))
    )
    var q_output = TileTensor(
        qout_device,
        row_major((total_q_tokens, Idx[num_q_heads], Idx[HEAD_DIM])),
    )
    var q_gamma = TileTensor(q_gamma_device, row_major(Idx[HEAD_DIM]))
    var k_gamma = TileTensor(k_gamma_device, row_major(Idx[HEAD_DIM]))
    var row_offsets = TileTensor(row_offsets_device, row_major(batch_size + 1))

    # === Kernel under test ===================================================
    fused_qk_rms_norm_ragged_paged[
        target="gpu", multiply_before_cast=multiply_before_cast
    ](
        q_proj.as_immut(),
        kv_collection,
        q_gamma.as_immut(),
        k_gamma.as_immut(),
        EPS,
        WEIGHT_OFFSET.cast[dtype](),
        UInt32(0),  # layer_idx
        row_offsets.as_immut(),
        q_output,
        ctx,
    )
    ctx.synchronize()

    if check:
        _verify_ref(
            ctx,
            cache_lengths,
            seq_lens,
            q_host,
            k_input_host,
            q_gamma_host,
            k_gamma_host,
            qout_device,
            blocks_device,
            block_elems,
            q_size,
            total_q_tokens,
        )

    _ = blocks_device
    _ = q_device
    _ = qout_device
    _ = q_gamma_device
    _ = k_gamma_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = row_offsets_device


# ===----------------------------------------------------------------------=== #
# Numerical reference oracle (higher precision, CPU)
# ===----------------------------------------------------------------------=== #


def _rms_ref(
    x: Span[Scalar[dtype], _],
    base: Int,
    gamma: HostBuffer[gamma_dtype],
    out_span: Span[mut=True, Scalar[dtype], _],
    out_base: Int,
):
    """Per-head RMSNorm reference matching the kernel math (fp32 accum)."""
    var ss = Float64(0)
    for c in range(HEAD_DIM):
        var v = x[base + c].cast[.float64]()
        ss += v * v
    var nf = (rsqrt(Float32(ss / Float64(HEAD_DIM)) + EPS)).cast[
        DType.float64
    ]()
    for c in range(HEAD_DIM):
        var v = x[base + c].cast[.float64]()
        var g = (gamma[c].cast[.float32]() + WEIGHT_OFFSET).cast[
            DType.float64
        ]()
        out_span[out_base + c] = (v * nf * g).cast[dtype]()


def _verify_ref(
    ctx: DeviceContext,
    cache_lengths: List[Int],
    seq_lens: List[Int],
    q_host: HostBuffer[dtype],
    k_input_host: HostBuffer[dtype],
    q_gamma_host: HostBuffer[gamma_dtype],
    k_gamma_host: HostBuffer[gamma_dtype],
    qout_device: DeviceBuffer[dtype],
    blocks_device: DeviceBuffer[dtype],
    block_elems: Int,
    q_size: Int,
    total_q_tokens: Int,
) raises:
    """Reference for both kernel outputs, in higher precision, vs the kernel.

    1. `q_output` (dense, self-contained): per-head RMSNorm of each Q row over
       head_dim, scaled by (q_gamma + weight_offset).
    2. the paged K cache, read back through its page table: each newly-written
       K row, per-head RMSNorm'd over head_dim and scaled by (k_gamma +
       weight_offset), in place.
    """
    var batch_size = len(seq_lens)

    # --- (1) q_output -------------------------------------------------------
    var qout_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    ctx.enqueue_copy(qout_host, qout_device)
    var qref_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    ctx.synchronize()

    for t in range(total_q_tokens):
        for h in range(num_q_heads):
            var base = (t * num_q_heads + h) * HEAD_DIM
            _rms_ref(
                q_host.as_span(), base, q_gamma_host, qref_host.as_span(), base
            )

    if not numeric_check(
        qout_host.as_span(), qref_host.as_span(), atol=1e-1, rtol=5e-2
    ):
        raise Error("fused_qk_rms_norm q_output mismatch")

    # --- (2) paged K cache read-back ----------------------------------------
    var blocks_host = ctx.enqueue_create_host_buffer[dtype](block_elems)
    ctx.enqueue_copy(blocks_host, blocks_device)
    ctx.synchronize()

    # Per-token batch + post + first physical page.
    var tok_batch = List[Int]()
    var tok_post = List[Int]()
    var page_base = List[Int]()
    var pb = 0
    for b in range(batch_size):
        page_base.append(pb)
        pb += ceildiv(cache_lengths[b] + seq_lens[b], PAGE_SIZE)
        for s in range(seq_lens[b]):
            tok_batch.append(b)
            tok_post.append(cache_lengths[b] + s)

    var n_written = len(tok_post)
    var n_k = max(1, n_written * num_kv_heads * HEAD_DIM)
    var cache_actual = ctx.enqueue_create_host_buffer[dtype](n_k)
    var cache_ref = ctx.enqueue_create_host_buffer[dtype](n_k)

    comptime kv_stride = (2 * NUM_LAYERS * PAGE_SIZE * num_kv_heads * HEAD_DIM)
    comptime tok_stride = num_kv_heads * HEAD_DIM
    for t in range(n_written):
        var b = tok_batch[t]
        var post = tok_post[t]
        var physical_page = page_base[b] + post // PAGE_SIZE
        var slot = physical_page * kv_stride + (post % PAGE_SIZE) * tok_stride
        for h in range(num_kv_heads):
            var dst = (t * num_kv_heads + h) * HEAD_DIM
            # Gather the kernel's in-place-normalized K row.
            for d in range(HEAD_DIM):
                cache_actual[dst + d] = blocks_host[slot + h * HEAD_DIM + d]
            # Reference: RMSNorm the K input we scattered in.
            _rms_ref(
                k_input_host.as_span(),
                dst,
                k_gamma_host,
                cache_ref.as_span(),
                dst,
            )

    if not numeric_check(
        cache_actual.as_span(), cache_ref.as_span(), atol=1e-1, rtol=5e-2
    ):
        raise Error("fused_qk_rms_norm K-cache mismatch")


# ===----------------------------------------------------------------------=== #
# Mode dispatch (argv handling shared from _fuzz)
# ===----------------------------------------------------------------------=== #


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "batch_size=",
                specs[i].batch_size,
                "max_cache_len=",
                specs[i].max_cache_len,
                "min_cache_len=",
                specs[i].min_cache_len,
                "seq_len=",
                specs[i].seq_len,
                "pattern=",
                specs[i].pattern,
                "ragged_q=",
                specs[i].ragged_q,
                "seed=",
                specs[i].seed,
                "dist=",
                specs[i].dist,
            )
        return

    if mode == "single":
        var spec = CaseSpec(
            flag_int(args, "--batch_size", 1),
            flag_int(args, "--max_cache_len", 0),
            flag_int(args, "--min_cache_len", 0),
            flag_int(args, "--seq_len", 4),
            flag_int(args, "--pattern", PAT_EQUAL),
            flag_int(args, "--ragged_q", 0),
            flag_int(args, "--seed", the_seed),
            flag_int(args, "--dist", VD_NORMAL),
        )
        print("FUZZ_SINGLE ", spec)
        with DeviceContext() as ctx:
            run_one_case(ctx, spec, check)
        print("FUZZ_RESULT verdict=PASS")
        return

    # Default: standalone in-process fuzz.
    print(
        "=== fuzz_fused_qk_rms_norm num_q_heads=",
        num_q_heads,
        "num_kv_heads=",
        num_kv_heads,
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
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
