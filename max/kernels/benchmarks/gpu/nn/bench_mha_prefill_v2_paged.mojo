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
"""TFLOPS bench for `MhaPrefillV2` against a `PagedKVCache` K/V operand.

Companion to `bench_mha_prefill_v2` (contiguous K/V). The kernel
under test is the same — `mha_prefill_v2[config]` — but K and V are
wrapped in `KVCacheMHAOperand` over a `PagedKVCacheCollection` instead
of the contiguous `LayoutTensorMHAOperand`. This isolates the cost of
the page-LUT indirection inside `block_paged_tile[KV_BLOCK=64]`.

Methodology
-----------

- Q, O: oversized via `CacheBustingBuffer` (512 MiB stride), so
  each iteration sees a cold L2 on the Q side. Matches the contiguous
  `bench_mha_prefill_v2` bench.
- K, V: single fresh-prefill paged allocation (cache_lengths=0,
  random unique pages, fixed LUT across iterations). Matches the
  existing paged ragged-attention bench convention. K/V is therefore
  hot-L2 — the headline comparison number is the contiguous
  `bench_mha_prefill_v2` bench re-run with `cache_busting=False`. The
  delta between the two reads as page-LUT overhead.
- mask: `CausalMask`. start_pos=0. num_keys=seq_len (fresh prefill).
- page_size: 128 (production-realistic for Llama-style serving; sits
  comfortably above the kernel's `KV_BLOCK=64` requirement).

Run
---

```bash
./bazelw run //max/kernels/benchmarks:gpu/nn/bench_mha_prefill_v2_paged \
    -- seq_len=8192 batch_size=1
```

MHA H=16:        `num_heads=16 group=1`
GQA Llama 3.1:   `num_heads=32 group=4`  (8 KV heads)
"""

from std.collections import Set
from std.math import ceildiv
from std.random import random_ui64, seed
from std.sys import get_defined_bool, get_defined_dtype, get_defined_int

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu import *
from max.gpu.host import DeviceContext
from std.utils import IndexList, StaticTuple

from internal_utils import CacheBustingBuffer, arg_parse
from internal_utils._utils import InitializationType
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    Idx,
    row_major,
)
from layout.coord import Coord
from layout._fillers import random as fill_random

from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection

from nn.attention.mha_mask import CausalMask
from nn.attention.mha_operand import KVCacheMHAOperand

from nn.attention.gpu.amd_structured.mha_prefill_v2 import (
    MhaConfigV2,
    mha_prefill_v2,
)


comptime _Q_BLOCK_SIZE = 32
comptime _NUM_WARPS = 8
comptime _BM = _NUM_WARPS * _Q_BLOCK_SIZE  # 256
comptime _KV_BLOCK = 64  # matches MhaPrefillV2 `KV_BLOCK`

# Matches the per-kernel tuning in `bench_mha_prefill_v2` — IGLP exact
# solver enabled with branch cap + node-order priority. See that file
# for the rationale on each flag.
comptime _PREFILL_IGLP_OPTS: StaticString = (
    "amdgpu-igrouplp-exact-solver=true,"
    "amdgpu-igrouplp-exact-solver-max-branches=10000,"
    "amdgpu-igrouplp-exact-solver-cost-heur=false"
)


def run_mha_prefill_v2_paged[
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int,
    page_size: Int,
](
    mut m: Bench,
    seq_len: Int,
    batch_size: Int,
    bench: Bool,
    ctx: DeviceContext,
) raises:
    comptime scale = Float32(0.125)  # ~rsqrt(64); the kernel is depth-agnostic
    # for the scale param — caller supplies. Matches contiguous bench.
    comptime kv_num_heads = num_heads // group
    comptime num_layers = 1
    comptime layer_idx = 0

    comptime assert qkv_type == .bfloat16, "MhaPrefillV2 is BF16-only"
    comptime assert (
        page_size >= _KV_BLOCK
    ), "page_size must be >= the kernel's KV_BLOCK (64)"
    comptime assert (
        page_size % _KV_BLOCK == 0
    ), "page_size must be a multiple of the kernel's KV_BLOCK (64)"

    var num_keys = seq_len  # fresh prefill
    # Random unique pages per (batch, block); allocate slack to
    # avoid LUT-set collisions stalling the bench setup.
    var pages_per_seq = ceildiv(num_keys, page_size)
    var num_pages = batch_size * pages_per_seq * 2

    # -------- Q / O (cache-busted, same as contiguous bench) --------
    var q_size = batch_size * num_heads * seq_len * depth
    var o_size = q_size

    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](q_size, simd_size, ctx)
    var cb_o = CacheBustingBuffer[.float32](o_size, simd_size, ctx)

    comptime random_distribution = InitializationType.uniform_distribution
    cb_q.init_on_device(random_distribution, ctx)

    # -------- K / V (paged, single allocation, fixed LUT) -------------------
    # Host-side: cache_lengths = 0 for every batch (fresh prefill).
    var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
    var max_seq_length: UInt32 = UInt32(seq_len)
    var max_context_length: UInt32 = UInt32(seq_len)

    var cache_lengths_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)

    # Paged LUT: random unique page index per (batch, block).
    var paged_lut_cols = pages_per_seq
    var paged_lut_size = batch_size * paged_lut_cols
    var paged_lut_host = List(length=paged_lut_size, fill=UInt32(0))
    var paged_lut_view = TileTensor(
        paged_lut_host,
        row_major(
            Coord(
                Int64(batch_size),
                Int64(paged_lut_cols),
            )
        ),
    )
    var seen = Set[Int]()
    for bs in range(batch_size):
        for block_idx in range(paged_lut_cols):
            var p = Int(random_ui64(0, UInt64(num_pages - 1)))
            while p in seen:
                p = Int(random_ui64(0, UInt64(num_pages - 1)))
            seen.add(p)
            paged_lut_view[bs, block_idx] = UInt32(p)

    var paged_lut_dev = ctx.enqueue_create_buffer[.uint32](paged_lut_size)
    ctx.enqueue_copy(paged_lut_dev, paged_lut_host)

    # KV block tensor: (num_pages, 2 [K|V], num_layers, page_size, kv_num_heads, depth)
    var kv_block_size = (
        num_pages * 2 * num_layers * page_size * kv_num_heads * depth
    )
    var kv_block_host = List(length=kv_block_size, fill=Scalar[qkv_type](0))
    fill_random(
        LayoutTensor[qkv_type, Layout.row_major[6](), MutAnyOrigin](
            kv_block_host,
            RuntimeLayout[Layout.row_major[6]()].row_major(
                IndexList[6](
                    num_pages, 2, num_layers, page_size, kv_num_heads, depth
                )
            ),
        )
    )
    var kv_block_dev = ctx.enqueue_create_buffer[qkv_type](kv_block_size)
    ctx.enqueue_copy(kv_block_dev, kv_block_host)

    # LayoutTensor views over the device buffers (consumed by PagedKVCacheCollection).
    comptime kv_block_layout = Layout.row_major[6]()
    var kv_block_tensor = LayoutTensor[qkv_type, kv_block_layout](
        kv_block_dev,
        RuntimeLayout[kv_block_layout].row_major(
            IndexList[6](
                num_pages, 2, num_layers, page_size, kv_num_heads, depth
            )
        ),
    )

    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_tensor = LayoutTensor[
        mut=False,
        .uint32,
        cache_lengths_layout,
    ](
        cache_lengths_dev,
        RuntimeLayout[cache_lengths_layout].row_major(IndexList[1](batch_size)),
    )

    comptime paged_lut_layout = Layout.row_major[2]()
    var paged_lut_tensor = LayoutTensor[
        mut=False,
        .uint32,
        paged_lut_layout,
    ](
        paged_lut_dev,
        RuntimeLayout[paged_lut_layout].row_major(
            IndexList[2](batch_size, paged_lut_cols)
        ),
    )

    var kv_collection = PagedKVCacheCollection[
        qkv_type,
        KVCacheStaticParams(num_heads=kv_num_heads, head_size=depth),
        page_size,
    ](
        # `mha_prefill_v2` reads both the `k` and `v` cache views, which are disjoint
        # kv_idx halves of one `blocks` buffer sharing its origin, so the
        # nested-origin exclusivity check rejects passing both. Declare the
        # kv_block_tensor origins as UnsafeAnyOrigin to opt out of exclusivity checking.
        kv_block_tensor.as_unsafe_any_origin(),
        cache_lengths_tensor,
        paged_lut_tensor,
        max_seq_length,
        max_context_length,
    )

    var k_cache = kv_collection.get_key_cache(layer_idx)
    var v_cache = kv_collection.get_value_cache(layer_idx)
    var k_operand = KVCacheMHAOperand(k_cache)
    var v_operand = KVCacheMHAOperand(v_cache)

    comptime _config = MhaConfigV2(
        q_block_size=_Q_BLOCK_SIZE,
        kv_block=_KV_BLOCK,
        depth=depth,
        num_heads=num_heads,
        num_kv_heads=kv_num_heads,
        num_warps=_NUM_WARPS,
    )

    if bench:

        @always_inline
        def bench_func(
            mut b: Bencher,
        ) raises {var cb_q, var cb_o, var k_operand, var v_operand, imm}:
            @always_inline
            def _kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
                var q_ptr = cb_q.offset_ptr(iteration).bitcast[
                    Scalar[qkv_type]
                ]()
                var q_tt = TileTensor[mut=False](
                    q_ptr,
                    row_major(
                        Coord(
                            Int32(batch_size),
                            Int32(seq_len),
                            Idx[num_heads],
                            Idx[depth],
                        )
                    ),
                )
                var o_tt = TileTensor(
                    cb_o.offset_ptr(iteration).bitcast[Float32](),
                    row_major(
                        Coord(
                            Int32(batch_size),
                            Int32(seq_len),
                            Idx[num_heads],
                            Idx[depth],
                        )
                    ),
                )
                mha_prefill_v2[_config, compile_options=_PREFILL_IGLP_OPTS](
                    q_tt,
                    k_operand,
                    v_operand,
                    o_tt,
                    CausalMask(),
                    scale,
                    num_keys,
                    0,  # start_pos (fresh prefill)
                    ctx,
                )

            bencher_iter_custom(b, _kernel_launch, ctx)

        def compute_flops() {imm} -> Int:
            # Causal: half the tiles. Matches `bench_mha_prefill_v2`'s
            # formula (`2 * B * H * N * NK * D`).
            return 2 * batch_size * num_heads * seq_len * num_keys * depth

        m.bench_function(
            bench_func,
            BenchId(
                "mha_prefill_v2_paged",
                # fmt: off
                input_id=String(
                    "qkv_type=", qkv_type,
                    "/depth=", depth,
                    "/num_heads=", num_heads,
                    "/group=", group,
                    "/seq_len=", seq_len,
                    "/batch_size=", batch_size,
                    "/page_size=", page_size,
                ),
                # fmt: on
            ),
            [ThroughputMeasure(BenchMetric.flops, compute_flops())],
        )
        ctx.synchronize()

    _ = cb_q
    _ = cb_o
    _ = kv_block_dev^
    _ = cache_lengths_dev^
    _ = paged_lut_dev^
    _ = kv_block_host^
    _ = cache_lengths_host^
    _ = paged_lut_host^


def main() raises:
    seed(0)

    comptime qkv_type = get_defined_dtype["qkv_type", .bfloat16]()
    comptime depth = get_defined_int["depth", 128]()
    comptime num_heads = get_defined_int["num_heads", 16]()
    comptime group = get_defined_int["group", 1]()
    comptime page_size = get_defined_int["page_size", 128]()
    var seq_len = Int(arg_parse("seq_len", 8192))
    var batch_size = Int(arg_parse("batch_size", 1))
    var bench = arg_parse("benchmark", True)

    print("Running MhaPrefillV2 (paged K/V) benchmark with config:")
    print("  qkv_type :", qkv_type)
    print("  depth    :", depth)
    print("  num_heads:", num_heads, " group:", group)
    print("  page_size:", page_size)
    print("  seq_len  :", seq_len)
    print("  batch    :", batch_size)

    var m = Bench()
    with DeviceContext() as ctx:
        run_mha_prefill_v2_paged[
            qkv_type,
            depth,
            num_heads,
            group,
            page_size,
        ](
            m,
            seq_len,
            batch_size,
            bench,
            ctx,
        )
    m.dump_report()
