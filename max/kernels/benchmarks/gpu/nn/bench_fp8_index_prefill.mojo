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
"""NCU-profile the SM100 FP8 indexer prefill scorer at GLM-5.2 shapes.

Drives the warp-specialized K-streaming prefill kernel
(`fp8_index_score_sm100`, which routes the nh=32 long-prefill shape to
`sparse_index_fp8_sm100_prefill`) directly, so a single kernel invocation can be
captured under `kbench --profile ncu-single --set full` for source-correlated
analysis of the shipped 2-CTA/SM, Q-resident, 1-consumer-WG (128 q-scales in
registers) behavior.

One invocation benchmarks one shape; the shape set lives in the sibling
`bench_fp8_index_prefill.yaml`, whose four labelled groups say what each shape
is there to answer. Run the whole set, or one group, on the B200:

    kbench max/kernels/benchmarks/gpu/nn/bench_fp8_index_prefill.yaml
    kbench max/kernels/benchmarks/gpu/nn/bench_fp8_index_prefill.yaml \\
        --filter label=ragged_prefill

For a source-correlated capture, drive one shape directly -- every field is an
argument, so nothing is implied by a default. Pass `--run_benchmark=False`
under `ncu`: the profiler replays the kernel itself for its sampling, so the
benchmark loop on top of it only multiplies capture time.

    ncu --set full mojo bench_fp8_index_prefill.mojo --run_benchmark=False \\
        --batch_size=8 --seq_len=512 --cache_len=2048 --max_num_keys=3584 \\
        --spread=50 --label=ragged_prefill

The bare defaults are the nh=32 pure-prefill route (`cache_len=0`, causal, long
enough to clear the 448-token-tile gate), which is the configuration whose
2-CTA/SM residency is too deeply amortized to attribute from nsys.
"""

from std.random import rand, seed
from std.sys import get_defined_int

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, Bencher, BenchId
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from kv_cache.types import (
    KVCacheStaticParams,
    KVCollectionT,
    PagedKVCacheCollection,
)
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from nn.attention.gpu.sparse_index_fp8_sm100 import (
    SPEC_DECODE_N_TOKENS_ALT,
    fp8_index_score_sm100,
)
from nn.attention.mha_operand import (
    KVCacheMHAOperand,
    KVCacheScalesMHAOperand,
)
from nn.attention.mha_mask import MaskName
from std.math import ceildiv
from std.utils.index import IndexList


# Calling the scorer with `k_op`/`ks_op` built from the same cache inlined in
# one function trips a false aliasing error (the two operands share an origin).
# Constructing the operands INSIDE this helper, from a collection whose type is
# a TYPE parameter, makes their origins provably disjoint -- the test's
# `_score_paged_sm100` does the same -- so the scorer call lives here.
def _launch_scorer[
    num_heads: Int,
    depth: Int,
    KCollectionT: KVCollectionT,
](
    o_tile: TileTensor[.float32, ...],
    q_tile: TileTensor[mut=False, .float8_e4m3fn, ...],
    qs_tile: TileTensor[mut=False, .float32, ...],
    input_row_offsets_tile: TileTensor[mut=False, .uint32, ...],
    k_collection: KCollectionT,
    batch_size: Int,
    seq_len: Int,
    max_num_keys: Int,
    ctx: DeviceContext,
) raises:
    var k_cache = k_collection.get_key_cache(0)
    var k_op = KVCacheMHAOperand(k_cache)
    var ks_op = KVCacheScalesMHAOperand(k_cache)
    fp8_index_score_sm100[
        DType.float8_e4m3fn,
        type_of(k_op),
        type_of(ks_op),
        num_heads,
        depth,
        # MLA cache lengths do not include the new tokens, so the scorer
        # adds `seq_len` itself; this matches the test's call.
        _is_cache_length_accurate=False,
        # The production MTP-step tile hint; inert at the prefill width this
        # benchmark defaults to, kept so the routed instantiation set matches
        # what production builds.
        N_TOKENS_ALT=SPEC_DECODE_N_TOKENS_ALT,
    ](
        o_tile,
        q_tile,
        qs_tile,
        k_op,
        ks_op,
        input_row_offsets_tile,
        batch_size,
        seq_len,
        max_num_keys,
        True,  # causal: the nh=32 prefill route requires it
        ctx,
    )


def _run_name[
    num_heads: Int,
    depth: Int,
    page_size: Int,
](
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    max_num_keys: Int,
    spread: Int,
    label: String,
) -> String:
    var kind = String("fp8_index_score_sm100_prefill")
    if label.byte_length() > 0:
        kind += "/" + label
    # fmt: off
    return String(
        kind, " : ",
        "num_heads=", num_heads, ", ",
        "depth=", depth, ", ",
        "page_size=", page_size, " : ",
        "batch_size=", batch_size, ", ",
        "seq_len=", seq_len, ", ",
        "cache_len=", cache_len, ", ",
        "max_num_keys=", max_num_keys, ", ",
        "spread=", spread,
    )
    # fmt: on


def execute_fp8_index_prefill[
    num_heads: Int,
    depth: Int,
    page_size: Int,
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    max_num_keys: Int,
    spread: Int = 0,
    label: String = String(),
    run_benchmark: Bool = True,
) raises:
    """Benchmark one scorer invocation at one GLM-5.2 prefill shape.

    Args:
        ctx: Device context.
        m: Bench harness collecting results.
        batch_size: Number of sequences in the batch.
        seq_len: New (prefill) tokens per sequence. Long enough to clear the
            nh=32 prefill route's 448-token-tile gate; ``cache_len=0`` keeps
            the pure-prefill causal-cache=0 path that gate admits.
        cache_len: Cached tokens per sequence. 0 reproduces a fresh prefill;
            a nonzero value exercises the chunked-prefill-continuation path.
        max_num_keys: Score-buffer row stride (>= any entry's key count).
        spread: Cache-depth raggedness, in percent of `cache_len`. 0 gives the
            uniform batch. Otherwise entry depths run linearly from
            `cache_len * (1 - spread/100)` to `cache_len * (1 + spread/100)`,
            so the MEAN depth -- and hence the total key work -- is unchanged
            and only the distribution differs from the `spread=0` twin.
        label: Free-form group tag echoed into the benchmark name, so a row in
            a kbench sweep says which group of the YAML it came from. Nothing
            branches on it.
        run_benchmark: Time the kernel over the harness' iteration loop. False
            launches it exactly once and reports nothing, which is what `ncu`
            wants -- the profiler does its own replay.
    """
    var total_seq_len = batch_size * seq_len

    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=depth, is_mla=True
    )
    comptime num_layers = 1

    # Pool holds the live token range; the LUT is one page deep per sequence
    # (the scorer never dereferences past each row's real key count).
    # Deepest entry sets every allocation: the page pool, the LUT width and the
    # cache-collection bound are all batch maxima, exactly as production's
    # captured-graph metadata is.
    var lo_cache = cache_len - (cache_len * spread) // 100
    var hi_cache = cache_len + (cache_len * spread) // 100
    var keys_per_seq = hi_cache + seq_len
    var pages_per_seq = ceildiv(keys_per_seq, page_size)
    var num_blocks = batch_size * pages_per_seq + 1

    var q_size = total_seq_len * num_heads * depth
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    with q_device.map_to_host() as q_host:
        rand(q_host.as_span())

    var qs_size = total_seq_len * num_heads
    var qs_device = ctx.enqueue_create_buffer[.float32](qs_size)
    with qs_device.map_to_host() as qs_host:
        rand(qs_host.as_span())

    var input_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    with input_row_offsets_device.map_to_host() as iro_host:
        for i in range(batch_size + 1):
            iro_host[i] = UInt32(i * seq_len)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    with cache_lengths_device.map_to_host() as cl_host:
        for i in range(batch_size):
            # Linear grade lo -> hi across the batch; a single-entry batch
            # takes the mean, so `spread` cannot change a batch of one.
            if batch_size > 1:
                cl_host[i] = UInt32(
                    lo_cache + (hi_cache - lo_cache) * i // (batch_size - 1)
                )
            else:
                cl_host[i] = UInt32(cache_len)

    var k_shape = IndexList[6](
        num_blocks,
        1,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime k_block_layout = Layout.row_major[6]()
    var k_block_runtime_layout = RuntimeLayout[k_block_layout].row_major(
        k_shape
    )
    var k_block_device = ctx.enqueue_create_buffer[.float8_e4m3fn](
        k_shape.flattened_length()
    )
    with k_block_device.map_to_host() as k_block_host:
        rand(k_block_host.as_span())

    comptime head_dim_granularity = 1
    var ks_shape = IndexList[6](
        num_blocks,
        1,
        num_layers,
        page_size,
        kv_params.num_heads,
        head_dim_granularity,
    )
    comptime ks_block_layout = Layout.row_major[6]()
    var ks_block_runtime_layout = RuntimeLayout[ks_block_layout].row_major(
        ks_shape
    )
    var ks_block_device = ctx.enqueue_create_buffer[.float32](
        ks_shape.flattened_length()
    )
    with ks_block_device.map_to_host() as ks_block_host:
        rand(ks_block_host.as_span())

    comptime paged_lut_layout = Layout.row_major[2]()
    var paged_lut_shape = IndexList[2](batch_size, pages_per_seq)
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )
    var k_lut_device = ctx.enqueue_create_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    with k_lut_device.map_to_host() as k_lut_host:
        for bs in range(batch_size):
            for page_idx in range(pages_per_seq):
                k_lut_host[bs * pages_per_seq + page_idx] = UInt32(
                    1 + bs * pages_per_seq + page_idx
                )

    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_shape = IndexList[1](batch_size)
    var cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(cache_lengths_shape)

    var k_collection = PagedKVCacheCollection[
        DType.float8_e4m3fn,
        kv_params,
        page_size,
        scale_dtype_=DType.float32,
        quantization_granularity_=128,
    ](
        LayoutTensor[.float8_e4m3fn, k_block_layout](
            k_block_device,
            k_block_runtime_layout,
        ),
        LayoutTensor[mut=False, .uint32, cache_lengths_layout](
            cache_lengths_device,
            cache_lengths_runtime_layout,
        ),
        LayoutTensor[mut=False, .uint32, paged_lut_layout](
            k_lut_device,
            paged_lut_runtime_layout,
        ),
        UInt32(seq_len),
        UInt32(hi_cache),
        LayoutTensor[.float32, ks_block_layout](
            ks_block_device,
            ks_block_runtime_layout,
        ),
    )

    var o_size = total_seq_len * max_num_keys
    var o_device = ctx.enqueue_create_buffer[.float32](o_size)
    var o_tile = TileTensor(o_device, row_major(total_seq_len, max_num_keys))
    var q_tile = TileTensor(
        q_device, row_major(total_seq_len, num_heads, depth)
    )
    var qs_tile = TileTensor(qs_device, row_major(total_seq_len, num_heads))
    var input_row_offsets_tile = TileTensor(
        input_row_offsets_device, row_major(batch_size + 1)
    )

    # Score buffer must start filled (the scorer leaves forbidden slots
    # untouched; a benchmark that reuses the buffer across iters relies on a
    # defined baseline). Pre-fill with -inf, the production convention.
    with o_device.map_to_host() as o_host:
        for i in range(o_size):
            o_host[i] = -Float32.MAX

    @always_inline
    def kernel_launch(launch_ctx: DeviceContext) raises {mut o_tile, imm}:
        _launch_scorer[num_heads, depth, type_of(k_collection)](
            o_tile,
            q_tile,
            qs_tile,
            input_row_offsets_tile,
            k_collection,
            batch_size,
            seq_len,
            max_num_keys,
            launch_ctx,
        )

    @always_inline
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    if run_benchmark:
        m.bench_function(
            bench_func,
            BenchId(
                _run_name[num_heads, depth, page_size](
                    batch_size, seq_len, cache_len, max_num_keys, spread, label
                )
            ),
        )
    else:
        kernel_launch(ctx)
        ctx.synchronize()

    _ = q_device
    _ = qs_device
    _ = input_row_offsets_device
    _ = cache_lengths_device
    _ = k_block_device
    _ = ks_block_device
    _ = k_lut_device
    _ = o_device


def main() raises:
    # GLM-5.2 replicated indexer: 32 heads on every tensor-parallel rank.
    comptime num_heads = get_defined_int["num_heads", 32]()
    comptime depth = get_defined_int["depth", 128]()
    comptime page_size = get_defined_int["page_size", 128]()

    var batch_size = arg_parse("batch_size", 1)
    # Long enough to clear the nh=32 prefill route's 448-token-tile gate
    # (seq >= 1792); 2048 is the production continuation shape in the test.
    var seq_len = arg_parse("seq_len", 2048)
    # 0 = fresh prefill (causal cache=0), the shape the prefill route admits.
    var cache_len = arg_parse("cache_len", 0)
    # Row stride of the score buffer; >= every entry's live key count.
    var max_num_keys = arg_parse("max_num_keys", seq_len)
    # Cache-depth raggedness, in percent of `cache_len`; 0 is the uniform batch.
    var spread = arg_parse("spread", 0)
    # Which YAML group this shape came from; reporting only.
    var label = String(arg_parse("label", ""))
    # False leaves a single launch for `ncu` to replay.
    var run_benchmark = arg_parse("run_benchmark", True)

    seed(0)

    var m = Bench()
    with DeviceContext() as ctx:
        execute_fp8_index_prefill[num_heads, depth, page_size](
            ctx,
            m,
            batch_size,
            seq_len,
            cache_len,
            max_num_keys,
            spread,
            label,
            run_benchmark,
        )

    if run_benchmark:
        m.dump_report()
