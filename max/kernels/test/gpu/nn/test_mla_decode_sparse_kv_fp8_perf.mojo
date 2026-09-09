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
"""Perf harness for MLA_SM100_Decode_Sparse_KV_FP8 (KERN-3141 shared-index fold).

Adapted onto 504fe from myb/glm_52_mla_opt by Yingbo Ma (reference commit
86d7d5760ec) test_mla_decode_sparse_kv_fp8_perf.mojo. Adaptation: the A/B is a
comptime `shared_index` parameter (baseline vs shared-index folded), benched in
one binary (no -D toggle), so a single build+run times both.


Measures the sparse all-FP8 MLA decode kernel at the GLM-5.2 TP=8 / MTP=5
production shape: num_q_heads = 64/8 = 8 per rank, q_len = 6 (1 real +
5 spec tokens), topk = 2048.

Reproduces the launch-shape pitfall: grid.y = q_max_seq_len with only
num_q_heads (8) of BM (64) M-rows used per CTA. After the fold_q fix the
same shapes should dispatch with grid.y = 1 and 48/64 rows used.

Timing = host wall clock around N enqueues + one synchronize (enqueue
overhead identical before/after, so deltas are attributable to the kernel).
Not a correctness test: output is not verified here (see
test_mla_decode_sparse_kv_fp8.mojo for numerics).
"""

from std.builtin._closure import __ownership_keepalive
from std.math import ceildiv
from std.random import seed
from std.sys import has_nvidia_gpu_accelerator

from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
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
from std.utils import IndexList
from nn.attention.mha_mask import NullMask
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
    compute_mla_dispatch_scalars,
)

comptime Q_DEPTH = 576  # Full Q depth: 512 nope + 64 rope
comptime V_DEPTH = 512  # Output depth (nope only)
comptime ROPE_DEPTH = 64
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1  # MLA has 1 KV head
comptime KV_HEAD_SIZE = V_DEPTH + ROPE_DEPTH  # 576


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


def _coprime_multiplier(n: Int) -> Int:
    # Mirrors test_mla_decode_sparse_kv_fp8.mojo's helper of the same name.
    if n <= 1:
        return 1
    if _gcd(3, n) == 1:
        return 3
    if _gcd(5, n) == 1:
        return 5
    if _gcd(7, n) == 1:
        return 7
    if _gcd(11, n) == 1:
        return 11
    return 13


def bench_sparse_kv_fp8[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
    # A/B: False = unfolded per-position baseline; True = shared-index fold
    # (one identity-ordered topk list shared by every q position).
    shared_index: Bool = False,
    # Output dtype defaults to BF16; pass q_type explicitly for non-FP8 callers.
    output_type: DType = .bfloat16,
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    ctx: DeviceContext,
    topk: Int,
    q_max_seq_len: Int = 1,
    warmup_iters: Int = 5,
    iters: Int = 30,
) raises:
    var num_keys = cache_len + q_max_seq_len
    var total_q_tokens = batch_size * q_max_seq_len
    comptime scale = Float32(0.125)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1

    var total_pages = batch_size * ceildiv(num_keys, PAGE_SIZE)
    var max_pages_per_batch = ceildiv(num_keys, PAGE_SIZE)

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

    # Perf harness: KV cache content is zero — values do not affect timing
    # (no data-dependent control flow).
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_memset(blocks_device, 0)

    # Identity page table: gather scatter comes from token indices already.
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    var page_offset = 0
    for bi in range(batch_size):
        for p in range(max_pages_per_batch):
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += max_pages_per_batch
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_len)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    # Q: zeros (timing-independent).
    var q_size = total_q_tokens * num_heads * Q_DEPTH
    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_memset(q_device, 0)

    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_device = ctx.enqueue_create_buffer[output_type](out_size)

    # Per-query-token topk indices: deterministic permutation, in
    # physical form block_id * PAGE_SIZE + tok_in_page (identity LUT).
    var total_indices = total_q_tokens * topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    var mult = _coprime_multiplier(num_keys)
    for bi in range(batch_size):
        for s in range(q_max_seq_len):
            var g = bi * q_max_seq_len + s
            for i in range(topk):
                # Index-shared MTP: identity list, identical for every q
                # position -> read-once fold gathers it exactly once.
                var t: Int
                comptime if shared_index:
                    t = i % num_keys
                else:
                    t = (i * mult + 1 + s) % num_keys
                var page_idx = t // PAGE_SIZE
                var tok_in_page = t % PAGE_SIZE
                var block_id = Int(
                    lookup_table_host[bi * max_pages_per_batch + page_idx]
                )
                h_indices[g * topk + i] = Int32(
                    block_id * PAGE_SIZE + tok_in_page
                )
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)
    ctx.synchronize()

    # -------------------------------------------------------------------
    # PagedKVCacheCollection
    # -------------------------------------------------------------------
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
        ),
        LayoutTensor[mut=False, .uint32, cl_layout](
            cache_lengths_lt.ptr,
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, lt_layout_2d](
            lookup_table_lt.ptr,
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(q_max_seq_len),
        UInt32(cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    # -------------------------------------------------------------------
    # TileTensors + dispatch scalars + launch closure
    # -------------------------------------------------------------------
    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[Q_DEPTH])),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i * q_max_seq_len)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()
    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(batch_size + 1),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]

    var indices_stride = topk

    def _launch(ctx: DeviceContext) raises {imm}:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[q_type](num_heads, Q_DEPTH),
            ragged=True,
            sparse=True,
            fold_shared_index=shared_index,
        ](
            out_tt,
            q_tt,
            kv_cache,
            NullMask(),
            row_offsets_tt,
            scale,
            ctx,
            scalar_args_buf_tt,
            d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
                d_indices_device.unsafe_ptr()
            ),
            indices_stride=indices_stride,
        )

    # Warmup (includes JIT/compile path)
    for _ in range(warmup_iters):
        _launch(ctx)
    ctx.synchronize()

    var us_per_iter = (
        Float64(ctx.execution_time(_launch, iters)) / Float64(iters) / 1000.0
    )
    print(
        "PERF",
        name,
        " variant=",
        "shared_index" if shared_index else "baseline",
        " nqh=",
        num_heads,
        " q_len=",
        q_max_seq_len,
        " bs=",
        batch_size,
        " topk=",
        topk,
        " cache=",
        cache_len,
        " num_partitions=",
        num_partitions,
        " us_per_iter=",
        us_per_iter,
    )

    # `_launch` only captures the pointer-based views, so nothing else keeps the
    # owning allocations alive past the last `.unsafe_ptr()` call above. Without
    # this the buffers are freed while the timed launches are still in flight.
    __ownership_keepalive(
        blocks_device,
        lookup_table_device,
        cache_lengths_device,
        q_device,
        out_device,
        d_indices_device,
        row_offsets_device,
        mla_args,
    )


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            seed(42)

            # GLM-5.2 production decode shape (user terminology):
            #   sequence_length = MTP + 1 = 5 + 1 = 6
            #   local_q_head    = q_head / TP = 64 / 8 = 8
            # Sweep batch_size 1..8; sequence_length=1 rows are the
            # per-position reference.
            # KERN-3217 q1 split-K tuning A/B: q_len=1 sparse FP8 decode,
            # topk=2048, cache in {2048, 1024}, bs 1..8. The baseline binary
            # picks np=4 (unchanged dispatch); the candidate binary picks np
            # tracking the effective (clamped) KV page count. The
            # num_partitions PRINTED here is the cache-based heuristic value
            # and does NOT reflect the sparse launch np — read the actual np
            # from nsys decode grid.z (np = grid.z / batch_size).
            for bs in range(1, 9):
                bench_sparse_kv_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=False
                ](
                    "glm52_q1_splitk_bf16q_4wg",
                    bs,
                    2048,
                    ctx,
                    topk=2048,
                    q_max_seq_len=1,
                )
                bench_sparse_kv_fp8[
                    DType.float8_e4m3fn,
                    DType.float8_e4m3fn,
                    8,
                    shared_index=False,
                ](
                    "glm52_q1_splitk_fp8q_3wg",
                    bs,
                    2048,
                    ctx,
                    topk=2048,
                    q_max_seq_len=1,
                )
                bench_sparse_kv_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=False
                ](
                    "glm52_q1_splitk_bf16q_4wg",
                    bs,
                    1024,
                    ctx,
                    topk=2048,
                    q_max_seq_len=1,
                )
                bench_sparse_kv_fp8[
                    DType.float8_e4m3fn,
                    DType.float8_e4m3fn,
                    8,
                    shared_index=False,
                ](
                    "glm52_q1_splitk_fp8q_3wg",
                    bs,
                    1024,
                    ctx,
                    topk=2048,
                    q_max_seq_len=1,
                )
            # bs=16 regression probe: batch_size > 8 is OUT of the tuning's
            # scope (the dispatch guard is batch_size <= 8), so it reverts to
            # the OLD partition policy (np=4). Launch-only check — confirm np=4
            # via nsys decode grid.z (= 4*16 = 64); the timing here is NOT a
            # perf claim.
            bench_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=False
            ](
                "glm52_q1_splitk_bs16probe",
                16,
                2048,
                ctx,
                topk=2048,
                q_max_seq_len=1,
            )
            bench_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=False
            ](
                "glm52_q1_splitk_bs16probe",
                16,
                1024,
                ctx,
                topk=2048,
                q_max_seq_len=1,
            )
            # Byte-identical sanity: q_len>1 is OUT of scope for the q1 tuning.
            # These q_len=6 cells (baseline + shared-index fold) must be
            # unchanged between the baseline and candidate binaries.
            bench_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=False
            ](
                "glm52_q6_unchanged",
                8,
                2048,
                ctx,
                topk=2048,
                q_max_seq_len=6,
            )
            bench_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=True
            ](
                "glm52_q6_unchanged",
                8,
                2048,
                ctx,
                topk=2048,
                q_max_seq_len=6,
            )
        else:
            print("skip: requires SM100 NVIDIA GPU")
