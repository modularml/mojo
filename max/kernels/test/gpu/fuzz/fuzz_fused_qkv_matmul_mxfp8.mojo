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
# Per-kernel fuzz target: generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4
# (the MiniMax-M3-MXFP8 fused QKV projection, op
#  `mo.fused_qkv_matmul.ragged.paged.scale.mxfp8`; the dual-mode NVFP4/MXFP8
#  entry point in nn/kv_cache_ragged.mojo). MXFP8 mode: `float8_e4m3fn` input +
#  weights with `float8_e8m0fnu` (E8M0) block scales over 32-element K blocks
#  in the rank-5 SF-atom layout, fp32 accum. In one launch it computes the
#  block-scaled GEMM `hidden_state @ [Wq|Wk|Wv]^T`, returns the Q band as bf16
#  output, and SCATTERS the K and V bands in place into the paged KV cache.
#
# The fuzzable surface is the *ragged shape*, which drives the matmul M (the
# total token count) and the per-batch K/V scatter slot:
#   - batch size + per-batch new-token counts (`input_row_offsets`): M and the
#     per-row batch lookup;
#   - per-batch cache lengths: the K/V-cache scatter slot (the `% PAGE_SIZE`
#     page-crossing edge is the interesting modulus -- the OOB-WRITE risk);
#   - num_q_heads / num_kv_heads / hidden (K) are compile-time `-D` (the tuned
#     SM100 block-scaled matmul reads N/K from the static weight shape); head_dim
#     is fixed at the M3 dense value 128. N = (num_q + 2*num_kv) * head_dim; every
#     band boundary is a multiple of SF_MN_GROUP_SIZE=128 (required for the
#     per-band scale lookup).
#
#   -D g3_num_q_heads=8 -D g3_num_kv_heads=1 -D g3_hidden=256   [default: small]
#   -D g3_num_q_heads=64 -D g3_num_kv_heads=4 -D g3_hidden=2048  M3 dense (1 device)
#
# Three argv modes so the Python orchestrator can drive it with a per-case
# timeout + process isolation (a hanging case only kills its own subprocess):
#
#   --mode list-specs --seed S --budget B
#   --mode single --batch_size .. ... [--check 1]   (prints FUZZ_RESULT verdict=PASS)
#   --mode fuzz --seed S --budget B   (default; in-process batch)
#
# Oracles: memory-safety (memcheck/redzone) is the DEFAULT -- the K/V scatter
# into the paged cache at a ragged-driven slot is the OOB-write risk. `ref`
# (--check 1) is a host fp32-accum naive-dequant GEMM (E8M0 block dequant of
# input + weight, fp32 matmul), split into the Q band (vs `output`) and the K/V
# bands (vs the paged cache read back through its page table). B200/SM100 only
# (the tuned block-scaled matmul asserts on SM100 + static N/K).

from std.math import align_up, ceildiv, exp2, max, min
from std.random import random_ui64, seed
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
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    get_scale_factor,
    set_scale_factor,
)
from nn.kv_cache_ragged import (
    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4,
)
from layout._fillers import random

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check


# ===----------------------------------------------------------------------=== #
# Fixed M3 config. num_q_heads / num_kv_heads / hidden are -D-overridable.
# ===----------------------------------------------------------------------=== #

comptime HEAD_DIM = 128  # M3 dense head_dim
comptime PAGE_SIZE = 512  # paged-cache page size (matches the kv test)
comptime NUM_LAYERS = 1

comptime num_q_heads = get_defined_int["g3_num_q_heads", 8]()
comptime num_kv_heads = get_defined_int["g3_num_kv_heads", 1]()
comptime HIDDEN = get_defined_int["g3_hidden", 256]()  # K; mult of 128

comptime data_dtype = DType.float8_e4m3fn
comptime scale_dtype = MXFP8_SF_DTYPE  # float8_e8m0fnu
comptime out_dtype = DType.bfloat16
comptime kv_dtype = DType.bfloat16
comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE  # 32

comptime Q_DIM = num_q_heads * HEAD_DIM
comptime KV_DIM = num_kv_heads * HEAD_DIM
comptime N_TOTAL = Q_DIM + 2 * KV_DIM  # fused [Wq|Wk|Wv] rows

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()
comptime TILE = SF_MN_GROUP_SIZE  # boundary modulus for M (scale row group)

comptime PAT_EQUAL = 0
comptime PAT_RAMP = 1
comptime PAT_ALT = 2
comptime PAT_ONE_BIG = 3
comptime NUM_PATTERNS = 4


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    """One fuzz case: the runtime-varied ragged fused-QKV-matmul shape.

    M (total tokens) and the per-batch K/V-cache scatter slots are fully
    determined by these scalars; `seed`/`dist` pick fp8 input values + E8M0
    scale exponents (matter for the `ref` oracle, not memory-safety).
    """

    var batch_size: Int
    var max_cache_len: Int  # max per-batch existing K/V extent (scatter start)
    var min_cache_len: Int
    var seq_len: Int  # max new tokens per batch
    var pattern: Int  # cache-length disparity pattern (PAT_*)
    var ragged_q: Int  # 0: every batch has seq_len new tokens; 1: ragged
    var seed: Int
    var dist: Int  # value distribution placeholder (ref-oracle only)

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
    var out = List[Int]()
    var sl = max(1, spec.seq_len)
    for i in range(spec.batch_size):
        if spec.ragged_q == 0:
            out.append(sl)
        else:
            out.append(1 + (i % sl))
    return out^


def gen_specs(n: Int) -> List[CaseSpec]:
    """Generate `n` boundary-aware ragged fused-QKV-matmul cases.

    M (total tokens) is biased around SF_MN_GROUP_SIZE=128 (the scale row group
    + the matmul tile pivot). max_cache_len is biased around PAGE_SIZE so a K/V
    scatter straddles a page (the OOB-write edge). max_cache_len==0 is the
    fresh-prefill edge.
    """
    var specs = List[CaseSpec]()
    for _ in range(n):
        var batch_size = boundary_int(1, 8, 2)
        var max_cache_len = boundary_int(0, 1024, PAGE_SIZE)
        var min_cache_len = boundary_int(0, max_cache_len, PAGE_SIZE)
        # Bias seq_len around the scale group (128); also single-token decode.
        var seq_len: Int
        if Int(random_ui64(0, 3)) == 0:
            seq_len = 1
        else:
            seq_len = boundary_int(1, 256, TILE)
        var pattern = Int(random_ui64(0, UInt64(NUM_PATTERNS - 1)))
        var ragged_q = 1 if (seq_len > 1 and random_ui64(0, 1) == 1) else 0
        var the_seed = Int(random_ui64(0, 1_000_000))
        var dist = Int(random_ui64(0, 1))
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
# E8M0 block-scale fill (matches fuzz_block_scaled_mxfp8: powers of 2, 0 padding)
# ===----------------------------------------------------------------------=== #


def fill_scales[
    scales_layout: Layout
](
    scales: LayoutTensor[mut=True, scale_dtype, scales_layout, MutAnyOrigin],
    mn: Int,
    k: Int,
):
    """Fill an SF-atom scale tensor: random powers of two over each 32-K block,
    0.0 in the padding rows/cols (unused scales MUST be 0.0 or accuracy breaks).

    Draws a wide E8M0 exponent in [-12, +12], i.e. the scale spans 2^-12..2^+12
    (mostly fractional -- the realistic MXFP8 regime, where block scales are
    normalizing factors usually <= 1). E8M0 (float8_e8m0fnu) stores powers of two
    exactly, so 2.0**e casts losslessly. The spread exercises the exponent decode
    plus the fractional/underflow path; the worst-case dequant product
    (fp8 ~2^9 * 2^12 * 2^12 over K ~2^13) stays ~2^46, far under fp32 max, so the
    `ref` GEMM stays finite. (Genuine overflow extremes 2^+-127 are not
    ref-distinguishable -- kernel and ref both go Inf -- so the goal is a wide
    *finite* spread, not literal overflow.)
    """
    for idx0 in range(align_up(mn, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(k, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 < mn and idx1 < k:
                var e = Int(random_ui64(0, 24)) - 12
                var sv = exp2(Float64(e)).cast[scale_dtype]()
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    scales, idx0, idx1, sv
                )
            else:
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    scales, idx0, idx1, Scalar[scale_dtype](0.0)
                )


# ===----------------------------------------------------------------------=== #
# One case: build operands + paged collection + launch the fused QKV matmul.
# ===----------------------------------------------------------------------=== #


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    seed(spec.seed)
    var cache_lengths = derive_cache_lengths(spec)
    var seq_lens = derive_seq_lens(spec)
    var batch_size = spec.batch_size

    var q_max_seq_len = 0
    var total_tokens = 0
    var max_cache_len = 0
    var max_full = 1
    var total_pages = 0
    for i in range(batch_size):
        if seq_lens[i] > q_max_seq_len:
            q_max_seq_len = seq_lens[i]
        total_tokens += seq_lens[i]
        if cache_lengths[i] > max_cache_len:
            max_cache_len = cache_lengths[i]
        if cache_lengths[i] + seq_lens[i] > max_full:
            max_full = cache_lengths[i] + seq_lens[i]
        total_pages += ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)

    var M = total_tokens

    comptime kv_params = KVCacheStaticParams(
        num_heads=num_kv_heads, head_size=HEAD_DIM
    )
    comptime KV_AXIS = 2  # standard MHA paged cache: dim[1] = {K, V}

    var block_shape = IndexList[6](
        total_pages, KV_AXIS, NUM_LAYERS, PAGE_SIZE, num_kv_heads, HEAD_DIM
    )
    var block_elems = (
        total_pages * KV_AXIS * NUM_LAYERS * PAGE_SIZE * num_kv_heads * HEAD_DIM
    )

    # --- hidden_state (M, HIDDEN) fp8 + weight (N_TOTAL, HIDDEN) fp8 ----------
    comptime hs_layout = Layout.row_major(UNKNOWN_VALUE, HIDDEN)
    var hs_host = ctx.enqueue_create_host_buffer[data_dtype](max(1, M * HIDDEN))
    var hs_host_lt = LayoutTensor[data_dtype, hs_layout](
        hs_host.unsafe_ptr(),
        RuntimeLayout[hs_layout].row_major(IndexList[2](M, HIDDEN)),
    )
    random(hs_host_lt)

    comptime w_layout = Layout.row_major(N_TOTAL, HIDDEN)
    var w_host = ctx.enqueue_create_host_buffer[data_dtype](N_TOTAL * HIDDEN)
    var w_host_lt = LayoutTensor[data_dtype, w_layout](
        w_host.unsafe_ptr(),
        RuntimeLayout[w_layout].row_major(IndexList[2](N_TOTAL, HIDDEN)),
    )
    random(w_host_lt)

    # --- rank-5 SF-atom scales for input + weight ----------------------------
    comptime k_sf = ceildiv(HIDDEN, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime input_sf_layout = Layout.row_major(
        UNKNOWN_VALUE, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K
    )
    var m_sf = ceildiv(M, SF_MN_GROUP_SIZE)
    var input_scale_shape = IndexList[5](
        m_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K
    )
    var input_scale_elems = (
        max(1, m_sf) * k_sf * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    )
    var input_scale_host = ctx.enqueue_create_host_buffer[scale_dtype](
        input_scale_elems
    )
    var input_scale_host_lt = LayoutTensor[scale_dtype, input_sf_layout](
        input_scale_host.unsafe_ptr(),
        RuntimeLayout[input_sf_layout].row_major(input_scale_shape),
    )
    fill_scales(input_scale_host_lt, M, HIDDEN)

    comptime n_sf = N_TOTAL // SF_MN_GROUP_SIZE
    comptime weight_sf_layout = Layout.row_major(
        n_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K
    )
    var weight_scale_elems = (
        n_sf * k_sf * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    )
    var weight_scale_host = ctx.enqueue_create_host_buffer[scale_dtype](
        weight_scale_elems
    )
    var weight_scale_host_lt = LayoutTensor[scale_dtype, weight_sf_layout](
        weight_scale_host.unsafe_ptr(),
        RuntimeLayout[weight_sf_layout].row_major(
            IndexList[5](n_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K)
        ),
    )
    fill_scales(weight_scale_host_lt, N_TOTAL, HIDDEN)

    # --- paged blocks (zero-init so unwritten K/V slots are well-defined) -----
    var blocks_host = ctx.enqueue_create_host_buffer[kv_dtype](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_dtype](0)

    # --- cache_lengths + paged lookup table ----------------------------------
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](
        max(1, batch_size)
    )
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var max_pages_per_batch = align_up(ceildiv(max_full, PAGE_SIZE), 1)
    var lut_size = max(1, batch_size * max_pages_per_batch)
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    var page_base = List[Int]()
    for i in range(batch_size):
        page_base.append(page_offset)
        var num_pages_i = ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # --- input_row_offsets (ragged prefix sum) -------------------------------
    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(seq_lens[i])

    # --- copy to device ------------------------------------------------------
    var hs_device = ctx.enqueue_create_buffer[data_dtype](max(1, M * HIDDEN))
    ctx.enqueue_copy(hs_device, hs_host)
    var w_device = ctx.enqueue_create_buffer[data_dtype](N_TOTAL * HIDDEN)
    ctx.enqueue_copy(w_device, w_host)
    var input_scale_device = ctx.enqueue_create_buffer[scale_dtype](
        input_scale_elems
    )
    ctx.enqueue_copy(input_scale_device, input_scale_host)
    var weight_scale_device = ctx.enqueue_create_buffer[scale_dtype](
        weight_scale_elems
    )
    ctx.enqueue_copy(weight_scale_device, weight_scale_host)
    var blocks_device = ctx.enqueue_create_buffer[kv_dtype](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var output_device = ctx.enqueue_create_buffer[out_dtype](max(1, M * Q_DIM))
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
        max(1, batch_size)
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    # --- device LayoutTensors ------------------------------------------------
    var hs_dev_lt = LayoutTensor[data_dtype, hs_layout](
        hs_device.unsafe_ptr(),
        RuntimeLayout[hs_layout].row_major(IndexList[2](M, HIDDEN)),
    )
    var w_dev_lt = LayoutTensor[data_dtype, w_layout](
        w_device.unsafe_ptr(),
        RuntimeLayout[w_layout].row_major(IndexList[2](N_TOTAL, HIDDEN)),
    )
    var input_scale_dev_lt = LayoutTensor[scale_dtype, input_sf_layout](
        input_scale_device.unsafe_ptr(),
        RuntimeLayout[input_sf_layout].row_major(input_scale_shape),
    )
    var weight_scale_dev_lt = LayoutTensor[scale_dtype, weight_sf_layout](
        weight_scale_device.unsafe_ptr(),
        RuntimeLayout[weight_sf_layout].row_major(
            IndexList[5](n_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K)
        ),
    )
    comptime out_layout = Layout.row_major(UNKNOWN_VALUE, Q_DIM)
    var output_dev_lt = LayoutTensor[out_dtype, out_layout](
        output_device.unsafe_ptr(),
        RuntimeLayout[out_layout].row_major(IndexList[2](M, Q_DIM)),
    )
    comptime ro_layout = Layout(UNKNOWN_VALUE)
    var row_offsets_lt = LayoutTensor[.uint32, ro_layout](
        row_offsets_device.unsafe_ptr(),
        RuntimeLayout[ro_layout].row_major(IndexList[1](batch_size + 1)),
    )

    # --- PagedKVCacheCollection ----------------------------------------------
    var blocks_lt = LayoutTensor[kv_dtype, Layout.row_major[6]()](
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

    var kv_collection = PagedKVCacheCollection[kv_dtype, kv_params, PAGE_SIZE](
        LayoutTensor[kv_dtype, Layout.row_major[6]()](
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

    # === Kernel under test ===================================================
    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE, target="gpu"
    ](
        hs_dev_lt,
        row_offsets_lt,
        w_dev_lt,
        input_scale_dev_lt,
        weight_scale_dev_lt,
        Float32(1.0),
        kv_collection,
        UInt32(0),  # layer_idx
        output_dev_lt,
        ctx,
    )
    ctx.synchronize()

    if check:
        _verify_ref(
            ctx,
            cache_lengths,
            seq_lens,
            page_base,
            hs_host,
            w_host,
            input_scale_host,
            weight_scale_host,
            m_sf,
            output_device,
            blocks_device,
            block_elems,
            M,
        )

    _ = hs_device
    _ = w_device
    _ = input_scale_device
    _ = weight_scale_device
    _ = blocks_device
    _ = output_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = row_offsets_device


# ===----------------------------------------------------------------------=== #
# Numerical reference oracle: host fp32-accum naive-dequant GEMM
# ===----------------------------------------------------------------------=== #


def _verify_ref(
    ctx: DeviceContext,
    cache_lengths: List[Int],
    seq_lens: List[Int],
    page_base: List[Int],
    hs_host: HostBuffer[data_dtype],
    w_host: HostBuffer[data_dtype],
    input_scale_host: HostBuffer[scale_dtype],
    weight_scale_host: HostBuffer[scale_dtype],
    m_sf: Int,
    output_device: DeviceBuffer[out_dtype],
    blocks_device: DeviceBuffer[kv_dtype],
    block_elems: Int,
    M: Int,
) raises:
    """fp32-accum naive dequant GEMM, split into Q (vs output) + K/V (vs cache).

    For row m, output column n: sum over the K dim of
      dequant(hidden[m,k]) * dequant(weight[n,k])
    where dequant(x) = float32(x_fp8) * float32(scale_e8m0[block]) with the E8M0
    scale read at the SAME (row, k) SF-atom index the kernel uses
    (get_scale_factor). The Q band [0, Q_DIM) is compared to `output`; the K and
    V bands [Q_DIM, Q_DIM+KV_DIM) / [Q_DIM+KV_DIM, N_TOTAL) are compared to the
    paged cache read back through its page table.
    """
    var batch_size = len(seq_lens)

    comptime k_sf = ceildiv(HIDDEN, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime n_sf = N_TOTAL // SF_MN_GROUP_SIZE
    comptime input_sf_layout = Layout.row_major(
        UNKNOWN_VALUE, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K
    )
    comptime weight_sf_layout = Layout.row_major(
        n_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K
    )
    var input_scale_lt = LayoutTensor[scale_dtype, input_sf_layout](
        input_scale_host.unsafe_ptr(),
        RuntimeLayout[input_sf_layout].row_major(
            IndexList[5](m_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K)
        ),
    )
    var weight_scale_lt = LayoutTensor[scale_dtype, weight_sf_layout](
        weight_scale_host.unsafe_ptr(),
        RuntimeLayout[weight_sf_layout].row_major(
            IndexList[5](n_sf, k_sf, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K)
        ),
    )

    # Compute the full M x N_TOTAL GEMM in fp32, then split bands.
    var full = List[Float32]()
    for _ in range(M * N_TOTAL):
        full.append(Float32(0))

    for m in range(M):
        for n in range(N_TOTAL):
            var acc = Float32(0)
            for k in range(HIDDEN):
                var a = hs_host[m * HIDDEN + k].cast[.float32]()
                var b = w_host[n * HIDDEN + k].cast[.float32]()
                var sa = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    input_scale_lt, m, k
                ).cast[.float32]()
                var sb = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    weight_scale_lt, n, k
                ).cast[.float32]()
                acc += (a * sa) * (b * sb)
            full[m * N_TOTAL + n] = acc

    # --- (1) Q band vs output -----------------------------------------------
    var out_host = ctx.enqueue_create_host_buffer[out_dtype](max(1, M * Q_DIM))
    ctx.enqueue_copy(out_host, output_device)
    ctx.synchronize()

    var q_actual = ctx.enqueue_create_host_buffer[out_dtype](max(1, M * Q_DIM))
    var q_ref = ctx.enqueue_create_host_buffer[out_dtype](max(1, M * Q_DIM))
    for m in range(M):
        for c in range(Q_DIM):
            q_actual[m * Q_DIM + c] = out_host[m * Q_DIM + c]
            q_ref[m * Q_DIM + c] = full[m * N_TOTAL + c].cast[out_dtype]()

    if not numeric_check(
        q_actual.as_span(), q_ref.as_span(), atol=2e-1, rtol=5e-2
    ):
        raise Error("fused_qkv_matmul_mxfp8 Q output mismatch")

    # --- (2) K/V bands vs paged cache read-back -----------------------------
    var blocks_host = ctx.enqueue_create_host_buffer[kv_dtype](block_elems)
    ctx.enqueue_copy(blocks_host, blocks_device)
    ctx.synchronize()

    # Per-token batch + post + first physical page.
    var tok_batch = List[Int]()
    var tok_post = List[Int]()
    for b in range(batch_size):
        for s in range(seq_lens[b]):
            tok_batch.append(b)
            tok_post.append(cache_lengths[b] + s)

    var n_written = len(tok_post)
    var n_kv = max(1, n_written * num_kv_heads * HEAD_DIM)
    # The K band lives at output columns [Q_DIM, Q_DIM+KV_DIM); V at [..+KV_DIM).
    comptime kv_stride = (2 * NUM_LAYERS * PAGE_SIZE * num_kv_heads * HEAD_DIM)
    comptime tok_stride = num_kv_heads * HEAD_DIM

    # global token index = ragged prefix; need it to index `full`.
    var tok_global = List[Int]()
    var g = 0
    for b in range(batch_size):
        for _ in range(seq_lens[b]):
            tok_global.append(g)
            g += 1

    var k_actual = ctx.enqueue_create_host_buffer[kv_dtype](n_kv)
    var k_ref = ctx.enqueue_create_host_buffer[kv_dtype](n_kv)
    var v_actual = ctx.enqueue_create_host_buffer[kv_dtype](n_kv)
    var v_ref = ctx.enqueue_create_host_buffer[kv_dtype](n_kv)

    for t in range(n_written):
        var b = tok_batch[t]
        var post = tok_post[t]
        var mglob = tok_global[t]
        var physical_page = page_base[b] + post // PAGE_SIZE
        # K is dim[1]=0; V is dim[1]=1.
        # K is dim[1]=0, V is dim[1]=1; the dim[1] stride is NUM_LAYERS *
        # PAGE_SIZE * num_kv_heads * HEAD_DIM == kv_stride // 2.
        var k_slot = physical_page * kv_stride + (post % PAGE_SIZE) * tok_stride
        var v_slot = k_slot + (kv_stride // 2)
        for h in range(num_kv_heads):
            for d in range(HEAD_DIM):
                var col = h * HEAD_DIM + d
                var dst = (t * num_kv_heads + h) * HEAD_DIM + d
                k_actual[dst] = blocks_host[k_slot + col]
                v_actual[dst] = blocks_host[v_slot + col]
                k_ref[dst] = full[mglob * N_TOTAL + Q_DIM + col].cast[
                    kv_dtype
                ]()
                v_ref[dst] = full[mglob * N_TOTAL + Q_DIM + KV_DIM + col].cast[
                    kv_dtype
                ]()

    if not numeric_check(
        k_actual.as_span(), k_ref.as_span(), atol=2e-1, rtol=5e-2
    ):
        raise Error("fused_qkv_matmul_mxfp8 K-cache mismatch")
    if not numeric_check(
        v_actual.as_span(), v_ref.as_span(), atol=2e-1, rtol=5e-2
    ):
        raise Error("fused_qkv_matmul_mxfp8 V-cache mismatch")


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
            flag_int(args, "--seq_len", 128),
            flag_int(args, "--pattern", PAT_EQUAL),
            flag_int(args, "--ragged_q", 0),
            flag_int(args, "--seed", the_seed),
            flag_int(args, "--dist", 0),
        )
        print("FUZZ_SINGLE ", spec)
        with DeviceContext() as ctx:
            run_one_case(ctx, spec, check)
        print("FUZZ_RESULT verdict=PASS")
        return

    # Default: standalone in-process fuzz.
    print(
        "=== fuzz_fused_qkv_matmul_mxfp8 num_q_heads=",
        num_q_heads,
        "num_kv_heads=",
        num_kv_heads,
        "hidden=",
        HIDDEN,
        "N_TOTAL=",
        N_TOTAL,
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
