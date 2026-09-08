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

from std.math import isclose, rsqrt
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
from nn.attention.mha_mask import CausalMask
from std.testing import assert_almost_equal

from std.utils.numerics import min_or_neg_inf


def run_mha[
    qkv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    cache_busting: Bool = True,
](
    mut m: Bench,
    seq_len: Int,
    num_keys: Int,
    batch_size: Int,
    num_partitions: Int,
    bench: Bool,
    verify: Bool,
    ctx: DeviceContext,
) raises:
    # Query, key, value dimensions.
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group
    comptime out_type = DType.bfloat16 if qkv_type.is_float8() else qkv_type

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size

    # Cache busting buffers: allocate oversized to defeat L2/infinity cache.
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

    # Initialize data on the device.
    comptime random_distribution = InitializationType.uniform_distribution

    cb_q.init_on_device(random_distribution, ctx)
    cb_k.init_on_device(random_distribution, ctx)
    cb_v.init_on_device(random_distribution, ctx)

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
                    CausalMask(),
                    scale,
                    ctx,
                    num_partitions if num_partitions > 0 else Optional[Int](),
                )

            bencher_iter_custom(b, _kernel_launch, ctx)

        def compute_flops() {imm} -> Int:
            # Using causal mask, skip half of tiles.
            return 2 * batch_size * num_heads * seq_len * num_keys * depth

        m.bench_function(
            bench_func,
            BenchId(
                "mha",
                # fmt: off
            input_id=String(
                "qkv_type=", qkv_type,
                "/num_heads=", num_heads,
                "/seq_len=", seq_len,
                "/num_keys=", num_keys,
                "/batch_size=", batch_size,
                "/cache_busting=", cache_busting,
            ),
                # fmt: on
            ),
            [ThroughputMeasure(BenchMetric.flops, compute_flops())],
        )
        # Wait for benchmark to complete before running verification
        ctx.synchronize()

    # Always run flash_attention once with zero offset for verification/output.
    # This ensures the output matches the data used for verification.
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
        CausalMask(),
        scale,
        ctx,
        num_partitions if num_partitions > 0 else Optional[Int](),
    )

    ctx.synchronize()

    comptime if not qkv_type.is_float8():
        if verify:
            # Copy output for verification
            ctx.enqueue_copy(flash_output_ptr, cb_o.device_buffer())
            # Allocate and initialize mask for verification
            var mask_size = batch_size * num_heads * seq_len * num_keys
            var mask_ptr = List(length=mask_size, fill=Scalar[mask_type](0))

            var mask = TileTensor(
                mask_ptr,
                row_major(
                    (
                        batch_size,
                        num_heads,
                        seq_len,
                        num_keys,
                    )
                ),
            )
            for b in range(batch_size):
                for h in range(num_heads):
                    for q_idx in range(seq_len):
                        for k_idx in range(num_keys):
                            mask[b, h, q_idx, k_idx] = (
                                0 if q_idx + num_keys - seq_len
                                >= k_idx else min_or_neg_inf[mask_type]()
                            )

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


@fieldwise_init
struct MHA_cfg(ImplicitlyCopyable, Writable):
    # params
    var qkv_type: DType
    var mask_type: DType
    var depth: Int
    var num_heads: Int
    var group: Int
    var cache_busting: Bool


def main() raises:
    comptime qkv_type = get_defined_dtype["qkv_type", .bfloat16]()
    comptime mask_type = get_defined_dtype["mask_type", .float32]()
    comptime depth = get_defined_int["depth", 128]()
    comptime num_heads = get_defined_int["num_heads", 32]()
    comptime group = get_defined_int["group", 1]()
    comptime cache_busting = get_defined_bool["cache_busting", True]()

    var seq_len = Int(arg_parse("seq_len", 64))
    var num_keys = Int(arg_parse("num_keys", 64))
    var batch_size = Int(arg_parse("batch_size", 1))
    var num_partitions = Int(arg_parse("num_partitions", 1))
    var bench = arg_parse("benchmark", True)
    var verify = arg_parse("verify", True)

    comptime cfg = MHA_cfg(
        qkv_type=qkv_type,
        mask_type=mask_type,
        depth=depth,
        num_heads=num_heads,
        group=group,
        cache_busting=cache_busting,
    )

    print("Running MHA benchmark with config:")
    print("  qkv_type:", cfg.qkv_type)
    print("  mask_type:", cfg.mask_type)
    print("  depth:", cfg.depth)
    print("  num_heads:", cfg.num_heads)
    print("  group:", cfg.group)
    print("  cache_busting:", cfg.cache_busting)

    var m = Bench()
    with DeviceContext() as ctx:
        run_mha[
            cfg.qkv_type,
            cfg.mask_type,
            cfg.depth,
            cfg.num_heads,
            cfg.group,
            cfg.cache_busting,
        ](
            m,
            seq_len,
            num_keys,
            batch_size,
            num_partitions,
            bench,
            verify,
            ctx,
        )
    m.dump_report()
