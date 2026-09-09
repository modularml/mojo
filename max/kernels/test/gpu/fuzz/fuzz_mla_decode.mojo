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
# Per-kernel fuzz target: generic_flare_mla_decode_kv_cache_ragged
# (the ragged paged-KV-cache MLA decode entry point used by the pipeline graph;
#  see gpu-kernels-fuzzing-design.md).
#
# This is the production decode op for DeepSeek-V2/V3/R1 and Kimi-K2.5 MLA: it
# takes a ragged Q (`[total_q_tokens, num_heads, DEPTH]`), an `input_row_offsets`
# prefix-sum, a `PagedKVCacheCollection`, and a packed `scalar_args_buf` dispatch
# buffer, then runs flash-MLA-decode against the paged cache. The fuzzable
# surface is the *ragged shape*: batch size, the per-batch KV cache lengths
# (`cache_length % PAGE_SIZE` and the split-K partition boundaries are the
# interesting moduli), and the per-batch Q token counts (the ragged early-exit
# path). num_heads / mask are compile-time `-D` defines (one kernel
# instantiation per build, like fuzz_matmul's N/K).
#
#   -D mla_num_heads=128   DeepSeek-V3/R1 (group=128; the 128-partition path) [default]
#   -D mla_num_heads=64    Kimi-K2.5 (group=64; the 64-partition path)
#   -D mla_num_heads=16    DeepSeek-V2-Lite
#   -D mla_mask=causal     production decode mask [default] (or `null`)
#
# Three argv modes so the Python orchestrator can drive it with per-case timeout
# + process isolation (a hanging case only kills its own subprocess):
#
#   --mode list-specs --seed S --budget B
#       Print generated specs (`FUZZ_SPEC ...` lines), no GPU work.
#   --mode single --batch_size .. --max_cache_len .. ... [--check 1] [--schedule N]
#       Run exactly one case (orchestration / shrinking / corpus replay).
#       Prints `FUZZ_RESULT verdict=PASS`; a hang times out; a crash exits != 0.
#   --mode fuzz --seed S --budget B   (default)
#       Generate + run a batch in-process (standalone convenience).

from std.math import align_up, ceildiv, max, min
from std.random import rand, randn, random_ui64, seed
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
    lt_to_tt,
    row_major,
)
from std.utils.index import IndexList
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.mha_mask import CausalMask, NullMask
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from nn.kv_cache_ragged import generic_flare_mla_decode_kv_cache_ragged

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
# Fixed MLA config (DeepSeek/Kimi). num_heads + mask are -D-overridable.
# ===----------------------------------------------------------------------=== #

comptime DEPTH = 576  # Q/K head dim = kv_latent_dim(512) + qk_rope_head_dim(64)
comptime V_DEPTH = 512  # output (V) head dim = DEPTH - 64
comptime PAGE_SIZE = 128  # paged-cache page size; the interesting cache modulus
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1  # MLA: a single latent KV head
comptime SCALE = Float32(0.125)

comptime num_heads = get_defined_int["mla_num_heads", 128]()
comptime group = num_heads  # MLA: all Q heads share the 1 KV head
comptime MASK = (  # production decode mask (flip to "null" for non-causal)
    "causal"
)

comptime q_type = DType.bfloat16
comptime kv_type = DType.bfloat16

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()
comptime TILE = PAGE_SIZE  # boundary modulus for the cache-length axis

# Cache-length disparity patterns (deterministic from the scalar spec, so a
# `single` case reproduces a ragged batch exactly without serializing a list).
comptime PAT_EQUAL = 0  # all batches == max_cache_len
comptime PAT_RAMP = 1  # linear ramp min..max across the batch
comptime PAT_ALT = 2  # alternating max / min (extreme within-batch disparity)
comptime PAT_ONE_BIG = 3  # one max, the rest min (one long seq + short ones)
comptime NUM_PATTERNS = 4


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    """One fuzz case: the runtime-varied ragged MLA-decode shape.

    Shape is fully determined by these scalars (so `single` mode reproduces a
    ragged batch exactly); `seed`/`dist` only pick input *values* (which matter
    for the `ref` oracle, not for the memory-safety / hang oracles).
    """

    var batch_size: Int
    var max_cache_len: Int
    var min_cache_len: Int
    var seq_len: Int  # max Q tokens per batch (decode: 1)
    var pattern: Int  # cache-length disparity pattern (PAT_*)
    var ragged_q: Int  # 0: every batch has seq_len Q tokens; 1: ragged per batch
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
    """Per-batch KV cache lengths from (batch_size, min, max, pattern)."""
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
    """Per-batch Q token counts. ragged_q varies them in [1, seq_len]."""
    var out = List[Int]()
    var sl = max(1, spec.seq_len)
    for i in range(spec.batch_size):
        if spec.ragged_q == 0:
            out.append(sl)
        else:
            out.append(1 + (i % sl))
    return out^


def gen_specs(n: Int, safe: Bool) -> List[CaseSpec]:
    """Generate `n` boundary-aware ragged decode cases.

    The dominant axis is `max_cache_len`, boundary-biased around PAGE_SIZE (128)
    and its multiples (where the paged last-tile and the split-K partition edges
    live -- the F-1 / KERN-2339 ragged-tile bug class). `safe` is reserved for a
    future degenerate region; today both modes draw from the same space.
    """
    var specs = List[CaseSpec]()
    for _ in range(n):
        var batch_size = boundary_int(1, 16, 4)
        var max_cache_len = boundary_int(1, 8192, TILE)
        # min in [0, max]: 0 exercises the empty-cache (first-token) edge.
        var min_cache_len = boundary_int(0, max_cache_len, TILE)
        # Decode-biased: 3/4 of cases are single-token (seq_len == 1); the rest
        # are small multi-token (speculative / chunked decode), where the ragged
        # Q early-exit path and grid_dim.y iteration matter.
        var seq_len: Int
        if Int(random_ui64(0, 3)) != 0:
            seq_len = 1
        else:
            seq_len = boundary_int(1, 8, 1)
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
    """Fill with a finite, softmax-stable distribution (|x| ~ 0.5).

    Large QK dot products over DEPTH=576 overflow bf16 softmax, so we keep
    magnitudes small (matching test_mla_decode_paged_variable's std=0.5). The
    NaN/Inf/large distributions are intentionally excluded -- those belong to a
    finiteness-contract oracle, not a higher-precision ref-diff.
    """
    if dist == VD_SPARSE:
        fill_sparse(span, density=0.1, lo=-0.5, hi=0.5)
    elif dist == VD_ALL_EQUAL:
        fill_all_equal(span, value=0.5)
    elif dist == VD_UNIFORM:
        fill_uniform(span, lo=-0.5, hi=0.5)
    else:  # VD_NORMAL (default)
        randn(span, mean=0.0, standard_deviation=0.5)


# ===----------------------------------------------------------------------=== #
# One case: build the paged collection + launch the ragged MLA-decode kernel.
# ===----------------------------------------------------------------------=== #


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    var cache_lengths = derive_cache_lengths(spec)
    var seq_lens = derive_seq_lens(spec)
    var batch_size = spec.batch_size

    var q_max_seq_len = 0
    var total_q_tokens = 0
    var max_cache_len = 0
    var total_pages = 0
    for i in range(batch_size):
        if seq_lens[i] > q_max_seq_len:
            q_max_seq_len = seq_lens[i]
        total_q_tokens += seq_lens[i]
        if cache_lengths[i] > max_cache_len:
            max_cache_len = cache_lengths[i]
        total_pages += ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
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

    # --- KV cache blocks (paged) --------------------------------------------
    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    fill_stable(blocks_host.as_span(), spec.dist)

    var page_stride = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var tok_stride = kv_params.num_heads * kv_params.head_size

    # Zero the unused tail slots of each page so a read past a batch's valid
    # length sees ~0 (its softmax weight is negligible) rather than a neighbor's
    # data -- keeps the ref oracle honest about a real over-read vs. a harmless one.
    var cur_page = 0
    for bi in range(batch_size):
        var num_keys_i = cache_lengths[bi] + seq_lens[bi]
        var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
        for pg in range(num_pages_i):
            var valid_toks = min(num_keys_i - pg * PAGE_SIZE, PAGE_SIZE)
            var base = cur_page * page_stride + valid_toks * tok_stride
            var zero_count = (PAGE_SIZE - valid_toks) * tok_stride
            for z in range(zero_count):
                blocks_host[base + z] = Scalar[kv_type](0)
            cur_page += 1

    # --- cache_lengths + lookup table ---------------------------------------
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](
        max(1, batch_size)
    )
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var max_num_keys_any_batch = 0
    for i in range(batch_size):
        var nk = cache_lengths[i] + seq_lens[i]
        if nk > max_num_keys_any_batch:
            max_num_keys_any_batch = nk
    # Pad LUT row stride to a multiple of 8 (the PagedKVCache.populate SIMD path).
    var max_pages_per_batch = align_up(
        ceildiv(max_num_keys_any_batch, PAGE_SIZE), 8
    )
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

    # --- ragged Q + input_row_offsets ---------------------------------------
    var q_size = max(1, total_q_tokens * num_heads * DEPTH)
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    fill_stable(q_host.as_span(), spec.dist)

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(seq_lens[i])

    var out_size = max(1, total_q_tokens * num_heads * V_DEPTH)

    # --- copy to device ------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
        max(1, batch_size)
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    var out_device = ctx.enqueue_create_buffer[q_type](out_size)
    ctx.synchronize()

    # --- PagedKVCacheCollection ---------------------------------------------
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

    var q_tt = TileTensor(
        q_device, row_major((total_q_tokens, Idx[num_heads], Idx[DEPTH]))
    )
    var out_tt = TileTensor(
        out_device, row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH]))
    )
    var row_offsets_tt = TileTensor(
        row_offsets_device, row_major(batch_size + 1)
    )

    var mla_args = MLADispatchScalarArgs[num_heads=num_heads](
        batch_size, max_cache_len, q_max_seq_len, ctx
    )
    var scalar_args_buf_lt = mla_args.gpu_layout_tensor()

    # === Kernel under test ==================================================
    generic_flare_mla_decode_kv_cache_ragged[
        target="gpu",
        mask_str=MASK,
    ](
        q_tt,
        row_offsets_tt,
        kv_collection,
        UInt32(0),  # layer_idx
        SCALE,
        out_tt,
        lt_to_tt(scalar_args_buf_lt),
        ctx,
    )
    ctx.synchronize()
    _ = mla_args

    if check and total_q_tokens > 0:
        _verify_ref(
            ctx,
            spec,
            cache_lengths,
            seq_lens,
            blocks_host,
            q_host,
            out_device,
            out_size,
        )

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device


def _verify_ref(
    ctx: DeviceContext,
    spec: CaseSpec,
    cache_lengths: List[Int],
    seq_lens: List[Int],
    blocks_host: HostBuffer[kv_type],
    q_host: HostBuffer[q_type],
    out_device: DeviceBuffer[q_type],
    out_size: Int,
) raises:
    """Numerical oracle: per-batch mha_gpu_naive reference vs. the kernel output.

    Builds a reference in the kernel's ragged output layout
    (`[total_q_tokens, num_heads, V_DEPTH]`) by extracting each batch's
    contiguous K from the paged blocks, running the naive MHA, and copying the
    first V_DEPTH columns; then a single numeric_check over the whole tensor.
    """
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
    )
    var batch_size = spec.batch_size

    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    ctx.enqueue_copy(out_host, out_device)
    var ref_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    for i in range(out_size):
        ref_host[i] = Scalar[q_type](0)
    ctx.synchronize()

    var q_token_offset = 0
    for b in range(batch_size):
        var b_seq_len = seq_lens[b]
        var ref_num_keys = cache_lengths[b] + b_seq_len

        # Extract this batch's contiguous K (= V for MLA) from the paged blocks.
        var k_b_size = ref_num_keys * KV_NUM_HEADS * DEPTH
        var k_b_device = ctx.enqueue_create_buffer[kv_type](k_b_size)
        var k_b_host = ctx.enqueue_create_host_buffer[kv_type](k_b_size)
        var page_base_b = 0
        for bi in range(b):
            page_base_b += ceildiv(cache_lengths[bi] + seq_lens[bi], PAGE_SIZE)
        for tok in range(ref_num_keys):
            var physical_page = page_base_b + tok // PAGE_SIZE
            var src = (
                physical_page
                * NUM_LAYERS
                * PAGE_SIZE
                * kv_params.num_heads
                * kv_params.head_size
                + (tok % PAGE_SIZE) * kv_params.num_heads * kv_params.head_size
            )
            var dst = tok * KV_NUM_HEADS * DEPTH
            for d in range(KV_NUM_HEADS * DEPTH):
                k_b_host[dst + d] = blocks_host[src + d]
        ctx.enqueue_copy(k_b_device, k_b_host)

        var q_b_size = b_seq_len * num_heads * DEPTH
        var q_b_device = ctx.enqueue_create_buffer[q_type](q_b_size)
        var q_b_host = ctx.enqueue_create_host_buffer[q_type](q_b_size)
        var q_start = q_token_offset * num_heads * DEPTH
        for i in range(q_b_size):
            q_b_host[i] = q_host[q_start + i]
        ctx.enqueue_copy(q_b_device, q_b_host)

        var ref_b_size = b_seq_len * num_heads * DEPTH
        var ref_b_device = ctx.enqueue_create_buffer[q_type](ref_b_size)
        var ref_b_host = ctx.enqueue_create_host_buffer[q_type](ref_b_size)
        ctx.synchronize()

        var q_b_tt = TileTensor(
            q_b_device,
            row_major((Idx[1], b_seq_len, Idx[num_heads], Idx[DEPTH])),
        )
        var k_b_tt = TileTensor(
            k_b_device,
            row_major((Idx[1], ref_num_keys, Idx[KV_NUM_HEADS], Idx[DEPTH])),
        )
        var ref_b_tt = TileTensor(
            ref_b_device,
            row_major((Idx[1], b_seq_len, Idx[num_heads], Idx[DEPTH])),
        )

        # Reference mask must match the kernel's mask_str. For seq_len==1 (decode)
        # causal == null; for seq_len>1, causal is the production decode rule.
        comptime if MASK == "null":
            mha_gpu_naive(
                q_b_tt.to_layout_tensor(),
                k_b_tt.to_layout_tensor(),
                k_b_tt.to_layout_tensor(),
                NullMask(),
                ref_b_tt.to_layout_tensor(),
                SCALE,
                1,
                b_seq_len,
                ref_num_keys,
                num_heads,
                DEPTH,
                group,
                ctx,
            )
        else:
            mha_gpu_naive(
                q_b_tt.to_layout_tensor(),
                k_b_tt.to_layout_tensor(),
                k_b_tt.to_layout_tensor(),
                CausalMask(),
                ref_b_tt.to_layout_tensor(),
                SCALE,
                1,
                b_seq_len,
                ref_num_keys,
                num_heads,
                DEPTH,
                group,
                ctx,
            )
        ctx.synchronize()
        ctx.enqueue_copy(ref_b_host, ref_b_device)
        ctx.synchronize()

        # Copy the first V_DEPTH columns of each (token, head) into ref_host,
        # matching the kernel's ragged [total_q_tokens, num_heads, V_DEPTH] output.
        for s in range(b_seq_len):
            var out_base = (q_token_offset + s) * num_heads * V_DEPTH
            var ref_base = s * num_heads * DEPTH
            for h in range(num_heads):
                for d in range(V_DEPTH):
                    ref_host[out_base + V_DEPTH * h + d] = ref_b_host[
                        ref_base + DEPTH * h + d
                    ]

        q_token_offset += b_seq_len
        _ = k_b_device
        _ = q_b_device
        _ = ref_b_device

    # Tolerances match test_mla_decode_paged_variable / test_mla_decode_kv_fp8.
    if not numeric_check(
        out_host.as_span(), ref_host.as_span(), atol=3e-1, rtol=5e-2
    ):
        raise Error("MLA decode vs naive mismatch")


def run_schedule_case(
    ctx: DeviceContext,
    spec: CaseSpec,
    repeats: Int,
    pin_partitions: Bool = True,
) raises:
    """Rerun bit-stability on a long-cache decode case (serves two oracles).

    Re-runs `repeats` times on the same input and flags any non-bit-exact
    output. A difference means the inter-block split-K reduction over the paged
    cache is order-dependent (a race / nondeterminism) -- which racecheck
    (intra-block only) cannot see.

    - `pin_partitions=True` (`schedule` oracle): forces num_partitions=2 to
      exercise a specific split-K decomposition.
    - `pin_partitions=False` (`determinism` oracle): leaves num_partitions
      UNSET so the launch uses whatever `mla_decode_dispatch` picks for this
      shape (mla_decode_dispatch.mojo), catching races in the default decode.
    """
    # A single decode token with a long cache is the split-K path.
    var sched_spec = CaseSpec(
        max(1, spec.batch_size),
        max(2 * PAGE_SIZE, spec.max_cache_len),
        max(2 * PAGE_SIZE, spec.max_cache_len),
        1,  # decode
        PAT_EQUAL,
        0,
        spec.seed,
        VD_NORMAL,
    )
    var cache_lengths = derive_cache_lengths(sched_spec)
    var seq_lens = derive_seq_lens(sched_spec)
    var batch_size = sched_spec.batch_size
    var total_q_tokens = batch_size
    var max_cache_len = sched_spec.max_cache_len
    var total_pages = 0
    for i in range(batch_size):
        total_pages += ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
    )
    var block_elems = total_pages * NUM_LAYERS * PAGE_SIZE * DEPTH
    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    randn(blocks_host.as_span(), mean=0.0, standard_deviation=0.5)

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])
    var max_pages_per_batch = align_up(ceildiv(max_cache_len + 1, PAGE_SIZE), 8)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for i in range(batch_size):
        var np_i = ceildiv(cache_lengths[i] + 1, PAGE_SIZE)
        for p in range(np_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += np_i

    var q_size = total_q_tokens * num_heads * DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)
    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)
    var out_size = total_q_tokens * num_heads * V_DEPTH

    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    var out_device = ctx.enqueue_create_buffer[q_type](out_size)
    ctx.synchronize()

    var block_shape = IndexList[6](
        total_pages, 1, NUM_LAYERS, PAGE_SIZE, KV_NUM_HEADS, DEPTH
    )
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
        UInt32(1),
        UInt32(max_cache_len),
    )
    var q_tt = TileTensor(
        q_device, row_major((total_q_tokens, Idx[num_heads], Idx[DEPTH]))
    )
    var out_tt = TileTensor(
        out_device, row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH]))
    )
    var row_offsets_tt = TileTensor(
        row_offsets_device, row_major(batch_size + 1)
    )
    var mla_args = MLADispatchScalarArgs[num_heads=num_heads](
        batch_size, max_cache_len, 1, ctx
    )
    var scalar_args_buf_lt = mla_args.gpu_layout_tensor()
    # Pin split-K for the `schedule` oracle; leave it to the decode heuristic
    # for the `determinism` oracle.
    var np = Optional[Int](2) if pin_partitions else Optional[Int]()

    generic_flare_mla_decode_kv_cache_ragged[target="gpu", mask_str=MASK](
        q_tt,
        row_offsets_tt,
        kv_collection,
        UInt32(0),
        SCALE,
        out_tt,
        lt_to_tt(scalar_args_buf_lt),
        ctx,
        num_partitions_in=np,
    )
    ctx.synchronize()
    var first_h = ctx.enqueue_create_host_buffer[q_type](out_size)
    ctx.enqueue_copy(first_h, out_device)
    ctx.synchronize()

    for _ in range(repeats - 1):
        generic_flare_mla_decode_kv_cache_ragged[target="gpu", mask_str=MASK](
            q_tt,
            row_offsets_tt,
            kv_collection,
            UInt32(0),
            SCALE,
            out_tt,
            lt_to_tt(scalar_args_buf_lt),
            ctx,
            num_partitions_in=np,
        )
        ctx.synchronize()
        var rep_h = ctx.enqueue_create_host_buffer[q_type](out_size)
        ctx.enqueue_copy(rep_h, out_device)
        ctx.synchronize()
        if not numeric_check(
            rep_h.as_span(), first_h.as_span(), atol=0.0, rtol=0.0
        ):
            raise Error("MLA decode split-K nondeterminism (schedule)")

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device


# ===----------------------------------------------------------------------=== #
# Mode dispatch (argv handling shared from _fuzz)
# ===----------------------------------------------------------------------=== #


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var schedule_repeats = flag_int(args, "--schedule", 0)
    var rerun = flag_int(args, "--rerun", 0)
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget, safe=False)
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
            flag_int(args, "--max_cache_len", 128),
            flag_int(args, "--min_cache_len", 0),
            flag_int(args, "--seq_len", 1),
            flag_int(args, "--pattern", PAT_EQUAL),
            flag_int(args, "--ragged_q", 0),
            flag_int(args, "--seed", the_seed),
            flag_int(args, "--dist", VD_NORMAL),
        )
        print("FUZZ_SINGLE ", spec)
        # The tuned ragged MLA-decode dispatch + MLADispatchScalarArgs are SM100
        # (B200) machinery; the BUILD target is b200-constrained accordingly.
        with DeviceContext() as ctx:
            if rerun > 0:
                run_schedule_case(ctx, spec, rerun, pin_partitions=False)
            elif schedule_repeats > 0:
                run_schedule_case(ctx, spec, schedule_repeats)
            else:
                run_one_case(ctx, spec, check)
        print("FUZZ_RESULT verdict=PASS")
        return

    # Default: standalone in-process fuzz.
    print(
        "=== fuzz_mla_decode num_heads=",
        num_heads,
        "mask=",
        MASK,
        "seed=",
        the_seed,
        "budget=",
        the_budget,
        "===",
    )
    var specs = gen_specs(the_budget, safe=True)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            if rerun > 0:
                run_schedule_case(ctx, specs[i], rerun, pin_partitions=False)
            elif schedule_repeats > 0:
                run_schedule_case(ctx, specs[i], schedule_repeats)
            else:
                run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
