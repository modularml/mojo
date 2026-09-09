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
# Per-kernel fuzz target: fused_rope_rmsnorm_quantization_kernel
# (the fused MLA RoPE + KV-cache RMSNorm + quantize write, launched via
#  mla_fused_rope_rmsnorm_quantization; see gpu-kernels-fuzzing-design.md).
#
# This is the production prefill/decode op that, in one launch, (1) applies
# interleaved RoPE to the query rope projections (-> q_rope_output), (2) applies
# interleaved RoPE to the rope columns of the latent KV and stores them into the
# paged K cache, and (3) RMSNorms the latent columns of the KV and stores them
# into the paged K cache (cast_saturating to the cache dtype). It is exercised
# only through the MLA prefill branches today -- it has no isolated unit test.
#
# The fuzzable surface is the *ragged shape*, which drives every runtime index:
#   - batch size + per-batch new-token counts (`input_row_offsets`): the
#     binary-searched batch lookup and the grid-stride token loop;
#   - per-batch cache lengths (the start positions): `post_seq_idx =
#     cache_length(b) + token_idx` indexes `freqs_cis` rows AND the paged K-cache
#     store position (the `% PAGE_SIZE` page-crossing edge is the interesting
#     modulus);
#   - num_q_heads is a compile-time `-D` (one kernel instantiation per build,
#     like fuzz_matmul's N/K and fuzz_mla_decode's mla_num_heads).
#
#   -D frrq_num_heads=16    DeepSeek-V2-Lite [default: small + fast ref oracle]
#   -D frrq_num_heads=64    Kimi-K2.5
#   -D frrq_num_heads=128   DeepSeek-V3/R1
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
# the ragged shape drives the freqs-row / paged-store / latent / gamma indexing.
# `ref` (--check 1) is a higher-precision CPU oracle over BOTH outputs: the
# dense roped query (`q_rope_output`, self-contained) and the paged K cache read
# back through its page table (the RMSNorm latent columns + the roped rope
# columns).

from std.math import align_up, ceildiv, cos, max, min, sin, sqrt
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
from nn.attention.gpu.mla_graph import mla_fused_rope_rmsnorm_quantization

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
# Fixed MLA config (DeepSeek/Kimi). num_heads is -D-overridable.
# ===----------------------------------------------------------------------=== #

comptime KV_NORM_DIM = 512  # kv_lora_rank: the RMSNorm'd latent width
comptime ROPE_DIM = 64  # qk_rope_head_dim: the RoPE'd width
comptime HEAD_SIZE = KV_NORM_DIM + ROPE_DIM  # 576 == kv cache head_size
comptime PAGE_SIZE = 128  # paged-cache page size; the interesting cache modulus
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1  # MLA: a single latent KV head

comptime num_q_heads = get_defined_int["frrq_num_heads", 16]()

comptime dtype = DType.bfloat16  # q / q_rope_output / cache dtype
comptime freq_dtype = DType.float32  # precomputed RoPE freqs (interleaved cos/sin)
comptime gamma_dtype = DType.bfloat16  # RMSNorm gamma weights
comptime kv_type = DType.bfloat16  # latent KV input + paged cache blocks
comptime EPS = Float32(1e-6)
comptime THETA = Float64(10000.0)  # RoPE base (only sets freq *values*)

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
    """One fuzz case: the runtime-varied ragged RoPE+RMSNorm shape.

    Shape is fully determined by these scalars (so `single` mode reproduces a
    ragged batch exactly); `seed`/`dist` only pick input *values* (which matter
    for the `ref` oracle, not for the memory-safety oracles).
    """

    var batch_size: Int
    var max_cache_len: Int  # max per-batch start position (existing cache)
    var min_cache_len: Int
    var seq_len: Int  # max new Q tokens per batch
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
    """Per-batch existing-cache lengths (== RoPE start positions)."""
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
    """Generate `n` boundary-aware ragged RoPE+RMSNorm cases.

    The dominant axes are `max_cache_len` (the start position, boundary-biased
    around PAGE_SIZE where `post_seq_idx % PAGE_SIZE` crosses a page in the K
    cache store and where the freqs row index lands) and `seq_len` (the new-token
    span -- boundary-biased around PAGE_SIZE so a write range straddles pages).
    `max_cache_len == 0` is the fresh-prefill edge (no existing cache).
    """
    var specs = List[CaseSpec]()
    for _ in range(n):
        var batch_size = boundary_int(1, 16, 4)
        var max_cache_len = boundary_int(0, 4096, TILE)
        var min_cache_len = boundary_int(0, max_cache_len, TILE)
        # Mostly substantial prefill spans (this is a prefill kernel); 1/4 are
        # single-token (decode-style append) to hit the seq_len==1 grid edge.
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

    Small magnitudes keep the RoPE/RMSNorm outputs comfortably inside bf16 range
    so the higher-precision ref-diff measures real error, not saturation. The
    NaN/Inf/large distributions are intentionally excluded here (they belong to a
    finiteness-contract oracle, not a tolerance diff).
    """
    if dist == VD_SPARSE:
        fill_sparse(span, density=0.2, lo=-0.5, hi=0.5)
    elif dist == VD_ALL_EQUAL:
        fill_all_equal(span, value=0.5)
    elif dist == VD_UNIFORM:
        fill_uniform(span, lo=-0.5, hi=0.5)
    else:  # VD_NORMAL (default)
        randn(span, mean=0.0, standard_deviation=0.5)


def fill_freqs(freqs: Span[mut=True, Scalar[freq_dtype], _], num_rows: Int):
    """Fill `freqs_cis` as interleaved magnitude-1 (cos, sin) pairs.

    Column `2j` = cos(pos * theta^(-2j/ROPE_DIM)), column `2j+1` = sin(...). The
    unit magnitude means RoPE preserves |x| (no overflow); the ref oracle reads
    these exact values back, so only the interleaving convention must match the
    kernel -- not any particular frequency schedule.
    """
    for pos in range(num_rows):
        for j in range(ROPE_DIM // 2):
            var inv = THETA ** (-(2.0 * Float64(j)) / Float64(ROPE_DIM))
            var angle = Float64(pos) * inv
            freqs[pos * ROPE_DIM + 2 * j] = cos(angle).cast[freq_dtype]()
            freqs[pos * ROPE_DIM + 2 * j + 1] = sin(angle).cast[freq_dtype]()


# ===----------------------------------------------------------------------=== #
# One case: build the paged collection + launch the fused RoPE+RMSNorm kernel.
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
    var max_post = 1  # max (cache_len + seq_len): freqs rows + cache extent
    var total_pages = 0
    for i in range(batch_size):
        if seq_lens[i] > q_max_seq_len:
            q_max_seq_len = seq_lens[i]
        total_q_tokens += seq_lens[i]
        if cache_lengths[i] > max_cache_len:
            max_cache_len = cache_lengths[i]
        if cache_lengths[i] + seq_lens[i] > max_post:
            max_post = cache_lengths[i] + seq_lens[i]
        total_pages += ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1  # MLA: is_mla=True => dim[1] == 1

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    # --- paged KV cache blocks (the kernel WRITES roped/normed values here) ---
    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    fill_stable(blocks_host.as_span(), VD_NORMAL)

    # --- latent KV input (the kernel READS this via kv_input_fn) -------------
    # Shape [total_q_tokens, HEAD_SIZE]: cols [0, KV_NORM_DIM) are RMSNorm'd,
    # cols [KV_NORM_DIM, HEAD_SIZE) are RoPE'd into the cache.
    var latent_host = ctx.enqueue_create_host_buffer[kv_type](
        max(1, total_q_tokens * HEAD_SIZE)
    )
    fill_stable(latent_host.as_span(), spec.dist)

    # --- q_rope input + freqs + gamma ----------------------------------------
    var q_size = max(1, total_q_tokens * num_q_heads * ROPE_DIM)
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    fill_stable(q_host.as_span(), spec.dist)

    var freq_rows = max(1, max_post)
    var freqs_host = ctx.enqueue_create_host_buffer[freq_dtype](
        freq_rows * ROPE_DIM
    )
    fill_freqs(freqs_host.as_span(), freq_rows)

    var gamma_host = ctx.enqueue_create_host_buffer[gamma_dtype](KV_NORM_DIM)
    fill_uniform(gamma_host.as_span(), lo=0.5, hi=1.5)

    # --- cache_lengths + paged lookup table ----------------------------------
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](
        max(1, batch_size)
    )
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var max_pages_per_batch = align_up(ceildiv(max_post, PAGE_SIZE), 8)
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
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var latent_device = ctx.enqueue_create_buffer[kv_type](
        max(1, total_q_tokens * HEAD_SIZE)
    )
    ctx.enqueue_copy(latent_device, latent_host)
    var q_device = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var qout_device = ctx.enqueue_create_buffer[dtype](q_size)
    var freqs_device = ctx.enqueue_create_buffer[freq_dtype](
        freq_rows * ROPE_DIM
    )
    ctx.enqueue_copy(freqs_device, freqs_host)
    var gamma_device = ctx.enqueue_create_buffer[gamma_dtype](KV_NORM_DIM)
    ctx.enqueue_copy(gamma_device, gamma_host)
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
    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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

    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
        LayoutTensor[kv_type, Layout.row_major[6]()](
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
    var q_rope = TileTensor(
        q_device, row_major((total_q_tokens, Idx[num_q_heads], Idx[ROPE_DIM]))
    )
    var q_rope_out = TileTensor(
        qout_device,
        row_major((total_q_tokens, Idx[num_q_heads], Idx[ROPE_DIM])),
    )
    var freqs = TileTensor(freqs_device, row_major((freq_rows, Idx[ROPE_DIM])))
    var gamma = TileTensor(gamma_device, row_major(Idx[KV_NORM_DIM]))
    var row_offsets = TileTensor(row_offsets_device, row_major(batch_size + 1))
    var latent = TileTensor(
        latent_device, row_major((max(1, total_q_tokens), Idx[HEAD_SIZE]))
    )

    @always_inline
    @__copy_capture(latent)
    @__parameter
    def kv_input_fn[width: Int](coords: IndexList[2]) -> SIMD[kv_type, width]:
        return latent.load[width=width]((coords[0], coords[1]))

    # === Kernel under test ===================================================
    mla_fused_rope_rmsnorm_quantization[kv_input_fn=kv_input_fn](
        q_rope_out,
        q_rope.as_immut(),
        row_offsets.as_immut(),
        freqs.as_immut(),
        gamma.as_immut(),
        kv_collection,
        UInt32(0),  # layer_idx
        EPS,
        ctx,
    )
    ctx.synchronize()

    if check:
        _verify_ref(
            ctx,
            cache_lengths,
            seq_lens,
            q_host,
            latent_host,
            freqs_host,
            gamma_host,
            qout_device,
            blocks_device,
            block_elems,
            q_size,
            total_q_tokens,
        )

    _ = blocks_device
    _ = latent_device
    _ = q_device
    _ = qout_device
    _ = freqs_device
    _ = gamma_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = row_offsets_device


# ===----------------------------------------------------------------------=== #
# Numerical reference oracle (higher precision, CPU)
# ===----------------------------------------------------------------------=== #


def _verify_ref(
    ctx: DeviceContext,
    cache_lengths: List[Int],
    seq_lens: List[Int],
    q_host: HostBuffer[dtype],
    latent_host: HostBuffer[kv_type],
    freqs_host: HostBuffer[freq_dtype],
    gamma_host: HostBuffer[gamma_dtype],
    qout_device: DeviceBuffer[dtype],
    blocks_device: DeviceBuffer[kv_type],
    block_elems: Int,
    q_size: Int,
    total_q_tokens: Int,
) raises:
    """Reference for both kernel outputs, in fp64, vs. the kernel result.

    1. `q_rope_output` (dense, self-contained): interleaved complex RoPE of the
       query rope projections, with freqs at each token's `post_seq_idx`.
    2. the paged K cache, read back through its page table: the RMSNorm'd latent
       columns `[0, KV_NORM_DIM)` and the roped rope columns
       `[KV_NORM_DIM, HEAD_SIZE)`, for each newly-written token.
    """
    var batch_size = len(seq_lens)

    # Per-token batch + post_seq_idx, and per-batch first physical page.
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

    # --- (1) q_rope_output --------------------------------------------------
    var qout_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    ctx.enqueue_copy(qout_host, qout_device)
    var qref_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    ctx.synchronize()

    for t in range(total_q_tokens):
        var post = tok_post[t]
        for h in range(num_q_heads):
            var base = (t * num_q_heads + h) * ROPE_DIM
            for j in range(ROPE_DIM // 2):
                var x_re = q_host[base + 2 * j].cast[.float64]()
                var x_im = q_host[base + 2 * j + 1].cast[.float64]()
                var f_re = freqs_host[post * ROPE_DIM + 2 * j].cast[
                    DType.float64
                ]()
                var f_im = freqs_host[post * ROPE_DIM + 2 * j + 1].cast[
                    DType.float64
                ]()
                qref_host[base + 2 * j] = (x_re * f_re - x_im * f_im).cast[
                    dtype
                ]()
                qref_host[base + 2 * j + 1] = (x_re * f_im + x_im * f_re).cast[
                    dtype
                ]()

    if not numeric_check(
        qout_host.as_span(), qref_host.as_span(), atol=1e-1, rtol=5e-2
    ):
        raise Error("fused_rope_rmsnorm q_rope_output mismatch")

    # --- (2) paged K cache: RMSNorm columns + roped rope columns ------------
    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_host, blocks_device)
    ctx.synchronize()

    var n_written = len(tok_post)
    var cache_actual = ctx.enqueue_create_host_buffer[kv_type](
        max(1, n_written * HEAD_SIZE)
    )
    var cache_ref = ctx.enqueue_create_host_buffer[kv_type](
        max(1, n_written * HEAD_SIZE)
    )

    comptime page_stride = PAGE_SIZE * HEAD_SIZE  # MLA: middle dims are 1
    for t in range(n_written):
        var b = tok_batch[t]
        var post = tok_post[t]
        var physical_page = page_base[b] + post // PAGE_SIZE
        var src = physical_page * page_stride + (post % PAGE_SIZE) * HEAD_SIZE
        var dst = t * HEAD_SIZE

        # Gather the kernel's written row from the paged blocks.
        for d in range(HEAD_SIZE):
            cache_actual[dst + d] = blocks_host[src + d]

        # RMSNorm reference over the latent columns [0, KV_NORM_DIM).
        var ss = Float64(0)
        for c in range(KV_NORM_DIM):
            var v = latent_host[t * HEAD_SIZE + c].cast[.float64]()
            ss += v * v
        var nf = 1.0 / sqrt(ss / Float64(KV_NORM_DIM) + EPS.cast[.float64]())
        for c in range(KV_NORM_DIM):
            var v = latent_host[t * HEAD_SIZE + c].cast[.float64]()
            var g = gamma_host[c].cast[.float64]()
            cache_ref[dst + c] = (v * nf * g).cast[kv_type]()

        # Interleaved RoPE reference over the rope columns [KV_NORM_DIM, ...).
        for j in range(ROPE_DIM // 2):
            var lat_base = t * HEAD_SIZE + KV_NORM_DIM
            var x_re = latent_host[lat_base + 2 * j].cast[.float64]()
            var x_im = latent_host[lat_base + 2 * j + 1].cast[.float64]()
            var f_re = freqs_host[post * ROPE_DIM + 2 * j].cast[.float64]()
            var f_im = freqs_host[post * ROPE_DIM + 2 * j + 1].cast[
                DType.float64
            ]()
            cache_ref[dst + KV_NORM_DIM + 2 * j] = (
                x_re * f_re - x_im * f_im
            ).cast[kv_type]()
            cache_ref[dst + KV_NORM_DIM + 2 * j + 1] = (
                x_re * f_im + x_im * f_re
            ).cast[kv_type]()

    if not numeric_check(
        cache_actual.as_span(), cache_ref.as_span(), atol=1e-1, rtol=5e-2
    ):
        raise Error("fused_rope_rmsnorm K-cache mismatch")


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
        "=== fuzz_fused_rope_rmsnorm num_heads=",
        num_q_heads,
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
