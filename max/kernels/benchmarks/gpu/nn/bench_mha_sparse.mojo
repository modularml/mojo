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
# BLASST (arXiv 2512.12087) skip-counter validation harness.
#
# Target: FA4 SM100 (B200) bf16 MHA. This is a CLONE of bench_mha.mojo whose
# ONLY substantive change is the Q/K data: instead of uniform-random init
# (which produces ~0 skips at threshold -13, because all block-maxes are
# similar so nothing is negligible), it builds a "tag-space" construction that
# produces a KNOWN, dial-able block sparsity. That lets us confirm the kernel's
# BLASST skip counter matches the dialed target.
#
# The kernel and bench_mha.mojo are NOT modified. Run with:
#   -D ENABLE_BLASST=true -D BLASST_COUNT_SKIPS=true
#   -D BLASST_LOG_THRESHOLD_MAG=13000  (=> threshold -13.0 in log2 units)
#   -D n_buckets=<k>                   (sweep {1,2,4,8,16})
# Defines MUST precede the file path or they are silently ignored.
#
# Tag-space construction (gives skip rate = 1 - 1/n_buckets, up to the one
# peeled first tile):
#   * `n_buckets` of the depth=128 dims (dims 0..n_buckets-1) are a near-one-hot
#     bucket code; all other dims are 0.
#   * Every query row -> bucket 0: magnitude C=`tag_mag` in dim 0, else 0.
#     (Head-independent, so GQA Q-head/KV-head tag alignment is automatic.)
#   * Key BLOCK b (each spans `block_size` keys) -> bucket (b % n_buckets):
#     magnitude C in tag dim (b % n_buckets), else 0. So Q.K = C^2 for a
#     bucket-0 (kept) block and 0 for any other (skippable) block. Assigning
#     whole key blocks to a bucket makes each block entirely kept or entirely
#     skippable, matching the kernel's block-granular skip decision.
#   * V = seeded random uniform, so the output is meaningful (not degenerate).
#
# C is chosen so the kept-vs-skipped score gap, after the softmax scale s and
# in log2 units (x log2e), comfortably exceeds |threshold|. With s=0.125 and
# C=20: C^2 * s * log2e = 72.1 >> 13 (~5.5x margin).
#
# CRITICAL: uses a NON-CAUSAL (Null/full) mask. The counter observes query-row
# 0's warp; under a causal mask row 0 attends to almost no KV blocks so the
# counter would under-observe. With a full mask row 0 sees every key block and
# the counter is representative. The skip DECISION logic is mask-independent, so
# this validly tests the vote's firing rate.

from std.math import isclose, log2, rsqrt
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

from internal_utils import CacheBustingBuffer, arg_parse
from internal_utils._utils import InitializationType
from layout import Idx, TileTensor, row_major
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive

# BLASST validation uses a full (non-causal) mask so query-row 0 sees every KV
# block -- see the module docstring.
from nn.attention.mha_mask import NullMask
from std.random import random_float64, seed
from std.testing import assert_almost_equal

from std.utils.numerics import min_or_neg_inf


def _build_tag_space_qk[
    qkv_type: DType
](
    mut q_host: List[Scalar[qkv_type]],
    mut k_host: List[Scalar[qkv_type]],
    batch_size: Int,
    seq_len: Int,
    num_keys: Int,
    num_heads: Int,
    kv_num_heads: Int,
    depth: Int,
    n_buckets: Int,
    block_size: Int,
    tag_mag: Float64,
):
    """Fills host Q/K buffers with the tag-space bucket construction.

    Q is row-major `(batch, seq_len, num_heads, depth)`; K is row-major
    `(batch, num_keys, kv_num_heads, depth)`. Both buffers are zeroed first,
    then only the `n_buckets` tag dims are set. Every query row -> bucket 0
    (dim 0). Key block `b` -> bucket `b % n_buckets` (dim `b % n_buckets`).
    """
    var tag = Scalar[qkv_type](tag_mag)

    # Zero everything: all non-tag dims must be 0 so Q.K is exactly C^2 or 0.
    for i in range(len(q_host)):
        q_host[i] = Scalar[qkv_type](0)
    for i in range(len(k_host)):
        k_host[i] = Scalar[qkv_type](0)

    # Q: every (b, s, h) row is bucket 0 -> magnitude C in dim 0.
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                var base = ((b * seq_len + s) * num_heads + h) * depth
                q_host[base + 0] = tag

    # K: key `kk` lives in block `kk // block_size`; block b -> bucket
    # `b % n_buckets` -> magnitude C in dim `b % n_buckets`.
    for b in range(batch_size):
        for kk in range(num_keys):
            var block = kk // block_size
            var bucket = block % n_buckets
            for kvh in range(kv_num_heads):
                var base = ((b * num_keys + kk) * kv_num_heads + kvh) * depth
                k_host[base + bucket] = tag


def run_mha_sparse[
    qkv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    cache_busting: Bool = False,
](
    mut m: Bench,
    seq_len: Int,
    num_keys: Int,
    batch_size: Int,
    num_partitions: Int,
    n_buckets: Int,
    block_size: Int,
    tag_mag: Float64,
    bench: Bool,
    verify: Bool,
    ctx: DeviceContext,
) raises:
    # Query, key, value dimensions. `scale` matches bench_mha.mojo (0.125). Note
    # this is 1/8, NOT 1/sqrt(depth=128)~=0.0884 -- the kernel's BLASST decision
    # uses whatever scale is passed, so we keep 0.125 for parity and size C
    # against it (see module docstring: 5.5x threshold margin at 0.125).
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group
    comptime out_type = DType.bfloat16 if qkv_type.is_float8() else qkv_type

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size

    # Cache busting buffers. Defaulted OFF for this validation harness: with
    # cache busting ON each benchmark iteration reads a DIFFERENT window, but we
    # only author the tag pattern into window 0, so busted iterations would read
    # garbage and the counter would be meaningless. A single window guarantees
    # every iteration reads the authored tag-space data. (This is a
    # counter-validation bench, not a bandwidth bench.)
    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](
        q_size, simd_size, ctx, cache_busting
    )
    var cb_k = CacheBustingBuffer[qkv_type](
        k_size, simd_size, ctx, cache_busting
    )
    var cb_v = CacheBustingBuffer[qkv_type](
        v_size, simd_size, ctx, cache_busting
    )
    var cb_o = CacheBustingBuffer[out_type](
        o_size, simd_size, ctx, cache_busting
    )

    # Allocate host memory for verification.
    var output_ptr = List(length=o_size, fill=Scalar[out_type](0))
    var flash_output_ptr = List(
        length=cb_o.alloc_size(), fill=Scalar[out_type](0)
    )

    # ---- Tag-space Q/K init (the ONLY data change vs bench_mha) ------------
    # Author Q and K on the host with the bucket construction, then copy to the
    # device window-0 buffer. V is seeded random uniform (device init) so the
    # attention output is meaningful rather than degenerate.
    seed(0)
    var q_host = List(length=q_size, fill=Scalar[qkv_type](0))
    var k_host = List(length=k_size, fill=Scalar[qkv_type](0))
    _build_tag_space_qk[qkv_type](
        q_host,
        k_host,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        kv_num_heads,
        depth,
        n_buckets,
        block_size,
        tag_mag,
    )
    ctx.enqueue_copy(cb_q.device_buffer(), q_host)
    ctx.enqueue_copy(cb_k.device_buffer(), k_host)
    # V: seeded random uniform in [0, 1).
    var v_host = List(length=v_size, fill=Scalar[qkv_type](0))
    for i in range(v_size):
        v_host[i] = Scalar[qkv_type](random_float64(0.0, 1.0))
    ctx.enqueue_copy(cb_v.device_buffer(), v_host)
    ctx.synchronize()

    if bench:

        @always_inline
        def bench_func(
            mut b: Bencher,
        ) raises {var cb_q, var cb_k, var cb_v, var cb_o, imm}:
            @always_inline
            def _kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
                # Construct device buffers with offsets.
                var q_device = TileTensor(
                    cb_q.offset_ptr(iteration),
                    row_major(
                        (
                            batch_size,
                            seq_len,
                            Idx[num_heads],
                            Idx[depth],
                        )
                    ),
                )
                var k_device = TileTensor(
                    cb_k.offset_ptr(iteration),
                    row_major(
                        (
                            batch_size,
                            num_keys,
                            Idx[kv_num_heads],
                            Idx[depth],
                        )
                    ),
                )
                var v_device = TileTensor(
                    cb_v.offset_ptr(iteration),
                    row_major(
                        (
                            batch_size,
                            num_keys,
                            Idx[kv_num_heads],
                            Idx[depth],
                        )
                    ),
                )
                var output_device = TileTensor(
                    cb_o.offset_ptr(iteration),
                    row_major(
                        (
                            batch_size,
                            seq_len,
                            Idx[num_heads],
                            Idx[depth],
                        )
                    ),
                )

                flash_attention(
                    output_device,
                    q_device,
                    k_device,
                    v_device,
                    NullMask(),
                    scale,
                    ctx,
                    num_partitions if num_partitions > 0 else Optional[Int](),
                )

            bencher_iter_custom(b, _kernel_launch, ctx)

        def compute_flops() {imm} -> Int:
            # Full (non-causal) mask: all tiles participate.
            return 2 * batch_size * num_heads * seq_len * num_keys * depth

        m.bench_function(
            bench_func,
            BenchId(
                "mha_sparse",
                # fmt: off
            input_id=String(
                "qkv_type=", qkv_type,
                "/num_heads=", num_heads,
                "/seq_len=", seq_len,
                "/num_keys=", num_keys,
                "/batch_size=", batch_size,
                "/n_buckets=", n_buckets,
                "/cache_busting=", cache_busting,
            ),
                # fmt: on
            ),
            [ThroughputMeasure(BenchMetric.flops, compute_flops())],
        )
        # Wait for benchmark to complete before running verification
        ctx.synchronize()

    # Always run flash_attention once with zero offset. The BLASST skip counter
    # (when -D BLASST_COUNT_SKIPS=true) is printed from the kernel during this
    # run: it observes query-row 0's warp over all KV blocks under the full
    # mask.
    var q_device = TileTensor(
        cb_q.unsafe_ptr(),
        row_major(
            (
                batch_size,
                seq_len,
                Idx[num_heads],
                Idx[depth],
            )
        ),
    )
    var k_device = TileTensor(
        cb_k.unsafe_ptr(),
        row_major(
            (
                batch_size,
                num_keys,
                Idx[kv_num_heads],
                Idx[depth],
            )
        ),
    )
    var v_device = TileTensor(
        cb_v.unsafe_ptr(),
        row_major(
            (
                batch_size,
                num_keys,
                Idx[kv_num_heads],
                Idx[depth],
            )
        ),
    )
    var output_device = TileTensor(
        cb_o.unsafe_ptr(),
        row_major(
            (
                batch_size,
                seq_len,
                Idx[num_heads],
                Idx[depth],
            )
        ),
    )

    flash_attention(
        output_device,
        q_device,
        k_device,
        v_device,
        NullMask(),
        scale,
        ctx,
        num_partitions if num_partitions > 0 else Optional[Int](),
    )

    ctx.synchronize()

    comptime if not qkv_type.is_float8():
        if verify:
            # Copy output for verification
            ctx.enqueue_copy(flash_output_ptr, cb_o.device_buffer())
            # Full-visibility mask for the naive reference (matches NullMask).
            var mask_size = batch_size * num_heads * seq_len * num_keys
            var mask_ptr = List(length=mask_size, fill=Scalar[mask_type](0))

            var mask_device_ptr = ctx.enqueue_create_buffer[mask_type](
                mask_size
            )
            ctx.enqueue_copy(mask_device_ptr, mask_ptr)

            var mask4d = TileTensor(
                mask_device_ptr,
                row_major(
                    (
                        batch_size,
                        num_heads,
                        seq_len,
                        num_keys,
                    )
                ),
            )

            var output_ref_device_ptr = ctx.enqueue_create_buffer[out_type](
                o_size
            )
            var output_ref_device = TileTensor(
                output_ref_device_ptr,
                row_major(
                    (
                        batch_size,
                        seq_len,
                        Idx[num_heads],
                        Idx[depth],
                    )
                ),
            )
            ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

            mha_gpu_naive(
                q_device,
                k_device,
                v_device,
                mask4d,
                output_ref_device,
                scale,
                batch_size,
                seq_len,
                num_keys,
                num_heads,
                depth,
                group,
                ctx,
            )

            ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
            _ = output_ref_device_ptr
            _ = mask_device_ptr

            var rtol = 0.02

            for h in range(num_heads):
                for s in range(seq_len):
                    for d in range(depth):
                        var expect = output_ptr[d + depth * (h + s * num_heads)]
                        var actual = flash_output_ptr[
                            d + depth * (h + s * num_heads)
                        ]
                        if not isclose(expect, actual, atol=1e-5, rtol=rtol):
                            print(h, s, d, actual, expect)
                        assert_almost_equal(
                            expect, actual, atol=1e-5, rtol=rtol
                        )
            _ = mask_ptr^

    _ = cb_q
    _ = cb_k
    _ = cb_v
    _ = cb_o
    _ = flash_output_ptr^
    _ = output_ptr^
    _ = q_host^
    _ = k_host^
    _ = v_host^


def main() raises:
    comptime qkv_type = get_defined_dtype["qkv_type", .bfloat16]()
    comptime mask_type = get_defined_dtype["mask_type", .float32]()
    comptime depth = get_defined_int["depth", 128]()
    comptime num_heads = get_defined_int["num_heads", 32]()
    comptime group = get_defined_int["group", 8]()
    comptime cache_busting = get_defined_bool["cache_busting", False]()

    var seq_len = Int(arg_parse("seq_len", 8192))
    var num_keys = Int(arg_parse("num_keys", 8192))
    var batch_size = Int(arg_parse("batch_size", 2))
    var num_partitions = Int(arg_parse("num_partitions", 1))
    # BLASST tag-space knobs.
    var n_buckets = get_defined_int["n_buckets", 1]()
    # `block_size` is the KV block granularity the kernel skips at (BN). FA4
    # SM100 bf16 uses 128; keep it a knob so a divisor mismatch is easy to spot.
    var block_size = get_defined_int["block_size", 128]()
    # C, the tag magnitude. 20 => C^2*scale*log2e = 72.1 >> 13 (~5.5x margin).
    var tag_mag = Float64(get_defined_int["tag_mag", 20]())
    var bench = arg_parse("benchmark", True)
    # Verify defaults OFF: the naive full-mask reference is not the point of
    # this counter-validation harness (and it is O(seq*keys) host mask alloc).
    var verify = arg_parse("verify", False)

    print("Running MHA sparse (BLASST tag-space) benchmark with config:")
    print("  qkv_type:", qkv_type)
    print("  mask_type:", mask_type, "(NullMask / full visibility)")
    print("  depth:", depth)
    print("  num_heads:", num_heads)
    print("  group:", group, "(kv_num_heads =", num_heads // group, ")")
    print("  seq_len:", seq_len)
    print("  num_keys:", num_keys)
    print("  batch_size:", batch_size)
    print("  cache_busting:", cache_busting)
    print("  n_buckets:", n_buckets)
    print("  block_size (BN):", block_size)
    print("  tag_mag (C):", tag_mag)

    # Host-side expected skip fraction next to each run for easy comparison.
    # Ideal fraction is 1 - 1/n_buckets over all KV blocks; the kernel peels the
    # first (bucket-0, kept) block, so the observed S/T is that fraction over
    # the non-peeled blocks (within ~1 block).
    var total_blocks = num_keys // block_size
    var ideal_pct = 100.0 * (1.0 - 1.0 / Float64(n_buckets))
    # Peel-adjusted expectation: block 0 is peeled+kept; among blocks 1..T,
    # block b is skippable iff (b % n_buckets) != 0.
    var expected_skips = 0
    for b in range(1, total_blocks):
        if (b % n_buckets) != 0:
            expected_skips += 1
    print(
        "  EXPECTED skip fraction ~",
        ideal_pct,
        "% (ideal 1-1/n) => peel-adjusted",
        expected_skips,
        "of",
        total_blocks - 1,
        "blocks",
    )

    var m = Bench()
    with DeviceContext() as ctx:
        run_mha_sparse[
            qkv_type,
            mask_type,
            depth,
            num_heads,
            group,
            cache_busting,
        ](
            m,
            seq_len,
            num_keys,
            batch_size,
            num_partitions,
            n_buckets,
            block_size,
            tag_mag,
            bench,
            verify,
            ctx,
        )
    m.dump_report()
