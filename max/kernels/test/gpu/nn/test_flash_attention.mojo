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

from std.math import exp
from std.random import rand, seed
from std.sys import argv

from max.gpu import *
from max.gpu.host import DeviceContext
from std.sys import has_amd_gpu_accelerator
from max.gpu.host.info import (
    A100,
    H100,
    GPUInfo,
    _is_sm10x_gpu,
)
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from nn.attention.gpu.mha import (
    flash_attention,
    mha_gpu_naive,
)
from nn.attention.mha_mask import NullMask
from std.testing import assert_almost_equal

from std.utils.index import Index


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def is_sm8(info: GPUInfo) -> Bool:
    return info.api == "cuda" and info.compute >= 8 and info.compute < 9


def test[
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    batch_size: Int = 1,
    num_partitions: Optional[Int] = None,
    decoding_warp_split_k: Bool = False,
](
    seq_len: Int,
    num_keys: Int,
    ctx: DeviceContext,
    is_benchmark: Bool = False,
    use_index_input: Bool = False,
) raises:
    print(
        "test_flash_attention",
        "batch_size:",
        batch_size,
        "num_partitions:",
        num_partitions.value() if num_partitions else -1,
        "num_heads:",
        num_heads,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "group:",
        group,
        "qkv_type:",
        qkv_type,
        "depth:",
        depth,
    )

    # Query, key, value dimensions.
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size

    # Allocate memory for all variables.
    var q_ptr = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_ptr = ctx.enqueue_create_host_buffer[qkv_type](k_size)
    var v_ptr = ctx.enqueue_create_host_buffer[qkv_type](v_size)
    var output_ptr = ctx.enqueue_create_host_buffer[qkv_type](o_size)
    var flash_output_ptr = ctx.enqueue_create_host_buffer[qkv_type](o_size)

    # Q, K, V are initialized.
    if use_index_input:
        assert batch_size == 1
        for i in range(seq_len):
            for h in range(num_heads):
                for j in range(depth):
                    q_ptr[(i * num_heads + h) * depth + j] = Scalar[qkv_type](
                        i * depth + j
                    )
        for i in range(num_keys):
            for h in range(kv_num_heads):
                for j in range(depth):
                    k_ptr[(i * kv_num_heads + h) * depth + j] = Scalar[
                        qkv_type
                    ](i * depth + j)
        for i in range(num_keys):
            for h in range(kv_num_heads):
                for j in range(depth):
                    v_ptr[(i * kv_num_heads + h) * depth + j] = Scalar[
                        qkv_type
                    ](i * depth + j)
    else:
        seed(1234567890)
        rand[qkv_type](q_ptr.as_span())
        rand[qkv_type](k_ptr.as_span())
        rand[qkv_type](v_ptr.as_span())

    # Device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_size)
    var output_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)

    # Copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(v_device_ptr, v_ptr)

    # Construct device buffers.
    var q_device = TileTensor(
        q_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )
    var k_device = TileTensor(
        k_device_ptr,
        row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth])),
    )
    var v_device = TileTensor(
        v_device_ptr,
        row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth])),
    )
    var output_device = TileTensor(
        output_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )

    @always_inline
    def kernel_launch(
        ctx: DeviceContext,
    ) raises {var q_device, var k_device, var v_device, var output_device, imm}:
        flash_attention[decoding_warp_split_k=decoding_warp_split_k](
            output_device,
            q_device,
            k_device,
            v_device,
            NullMask(),
            scale,
            ctx,
            num_partitions,
        )

    if is_benchmark:
        comptime nrun = 50

        # Warmup
        kernel_launch(ctx)

        var nstime = Float64(ctx.execution_time(kernel_launch, nrun)) / Float64(
            nrun
        )
        var sectime = nstime / 1000000
        print(nrun, "runs avg", sectime, "ms")

    else:
        kernel_launch(ctx)

    ctx.synchronize()

    ctx.enqueue_copy(flash_output_ptr, output_device_ptr)

    var output_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
    var output_ref_device = TileTensor(
        output_ref_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )
    ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

    mha_gpu_naive(
        q_device,
        k_device,
        v_device,
        NullMask(),
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

    ctx.synchronize()
    ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
    ctx.synchronize()

    @__parameter
    def get_rtol() -> Float64:
        return 2e-2 if num_partitions and num_partitions.value() >= 4 else 1e-2

    var rtol = get_rtol()
    for h in range(num_heads):
        for s in range(seq_len):
            for d in range(depth):
                var expect = output_ptr[d + depth * (h + s * num_heads)].cast[
                    DType.float64
                ]()
                var actual = flash_output_ptr[
                    d + depth * (h + s * num_heads)
                ].cast[.float64]()
                var rerr = abs((actual - expect) / expect)
                assert_almost_equal(
                    actual,
                    expect,
                    atol=1e-5,
                    rtol=rtol,
                    msg=String(t"{h} {s} {d} {actual} {expect} {rerr}"),
                )

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = output_device_ptr


def test_depth_supported_by_gpu(info: GPUInfo) -> List[Int]:
    var depths: List = [64, 128, 512]

    if info == materialize[H100]() or _is_sm10x_gpu(info):
        depths.append(80)
        depths.append(256)

    return depths^


def test_context_encoding(ctx: DeviceContext) raises:
    test[.bfloat16, depth=127, num_heads=2](111, 121, ctx)

    comptime depths = test_depth_supported_by_gpu(ctx.default_device_info)

    comptime for d in range(len(depths)):
        comptime depth = depths[d]
        # bf16 depth == 128, bf16-fp32 mma
        test[
            DType.bfloat16,
            depth=depth,
            num_heads=1,
        ](128, 128, ctx)
        test[
            DType.bfloat16,
            depth=depth,
            num_heads=1,
        ](384, 384, ctx)
        test[
            DType.bfloat16,
            depth=depth,
            num_heads=3,
        ](256, 256, ctx)
        test[
            DType.bfloat16,
            depth=depth,
            num_heads=32,
        ](1024, 1024, ctx, is_benchmark())
        test[
            DType.bfloat16,
            depth=depth,
            num_heads=24,
            group=3,
        ](1024, 1024, ctx)
        # BF16 with sequence length not multiple of 128
        test[
            DType.bfloat16,
            depth=depth,
            num_heads=3,
            group=3,
        ](64, 64, ctx)
        test[
            DType.bfloat16,
            depth=depth,
            num_heads=3,
            group=3,
        ](102, 102, ctx)
        test[
            DType.bfloat16,
            depth=depth,
            num_heads=1,
        ](14, 14, ctx)
        test[
            DType.bfloat16,
            depth=depth,
            num_heads=1,
        ](528, 528, ctx)

        test[
            DType.bfloat16,
            depth=128,
            num_heads=1,
        ](128, 64, ctx)

        test[
            DType.bfloat16,
            depth=128,
            num_heads=3,
        ](256, 128, ctx)

        test[
            DType.bfloat16,
            depth=128,
            num_heads=24,
            group=3,
        ](1024, 100, ctx)

        test[
            DType.bfloat16,
            depth=128,
            num_heads=24,
            group=3,
        ](214, 300, ctx)

        test[
            DType.bfloat16,
            depth=128,
            num_heads=24,
            group=1,
        ](512, 1024, ctx)

        test[
            DType.bfloat16,
            depth=128,
            num_heads=32,
            group=4,
        ](12, 8, ctx)

        test[
            DType.bfloat16,
            depth=128,
            num_heads=3,
        ](14, 18, ctx)

        # odd seq_len
        test[
            DType.bfloat16,
            depth=128,
            num_heads=3,
        ](15, 18, ctx)
        test[
            DType.bfloat16,
            depth=128,
            num_heads=3,
        ](119, 200, ctx)

        # Zero seq_len / num_keys: the FLUX.2 VAE mid-block attention
        # runs on a flattened spatial dim, which is zero when the
        # encoder is invoked on a ``(0, 0, 3)`` placeholder image for
        # the text-to-image path.  The kernel must early-return rather
        # than launching with zero grid dims.
        test[
            DType.bfloat16,
            depth=128,
            num_heads=1,
        ](0, 0, ctx)
        test[
            DType.bfloat16,
            depth=128,
            num_heads=3,
        ](0, 0, ctx)


def test_decoding[
    batch_size: Int,
    num_partitions: Optional[Int],
    split_k: Bool,
    qkv_type: DType = .bfloat16,
](ctx: DeviceContext, use_index_input: Bool = False) raises:
    comptime depths = test_depth_supported_by_gpu(ctx.default_device_info)

    comptime for d in range(len(depths)):
        comptime depth = depths[d]
        test[
            qkv_type,
            depth=depth,
            num_heads=1,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
        ](1, 11, ctx, use_index_input=use_index_input)

        comptime if (
            not is_sm8(ctx.default_device_info)
            or num_partitions
            and num_partitions.value() == 1
        ):
            test[
                qkv_type,
                depth=depth,
                num_heads=2,
                batch_size=batch_size,
                num_partitions=num_partitions,
                decoding_warp_split_k=split_k,
            ](1, 523, ctx, use_index_input=use_index_input)
        test[
            qkv_type,
            depth=depth,
            num_heads=24,
            group=3,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
        ](1, 29, ctx, use_index_input=use_index_input)

    # TODO(KERN-1674): enable these tests after fixing the bug
    # test[
    #     qkv_type,
    #     depth=128,
    #     num_heads=3,
    #     group=3,

    #     batch_size=batch_size,
    #     num_partitions=num_partitions,
    #     decoding_warp_split_k=split_k,
    # ](1, 156, ctx)
    # test[
    #     qkv_type,
    #     depth=128,
    #     num_heads=3,
    #     group=3,

    #     batch_size=batch_size,
    #     num_partitions=num_partitions,
    #     decoding_warp_split_k=split_k,
    # ](1, 208, ctx)


def test_decoding_large_group[
    batch_size: Int,
    num_partitions: Optional[Int] = None,
    split_k: Bool = False,
    qkv_type: DType = .bfloat16,
](ctx: DeviceContext, use_index_input: Bool = False) raises:
    comptime depths = test_depth_supported_by_gpu(ctx.default_device_info)

    comptime for d in range(len(depths)):
        comptime depth = depths[d]
        test[
            qkv_type,
            depth=depth,
            num_heads=32,
            group=16,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
        ](1, 2000, ctx, use_index_input=use_index_input)


def test_flash_attention_sink_kernel(ctx: DeviceContext, seq_len: Int) raises:
    print("test_flash_attention_sink_kernel")
    comptime batch_size = 1
    comptime num_heads = 2
    comptime kv_heads = num_heads
    comptime num_keys = 64
    comptime depth = 128
    comptime qkv_type = DType.bfloat16  # fast path on A100/H100
    comptime scale = Float32(0.0)  # force QK logits to exactly 0

    var q_ptr = ctx.enqueue_create_host_buffer[qkv_type](
        batch_size * seq_len * num_heads * depth
    )
    var k_ptr = ctx.enqueue_create_host_buffer[qkv_type](
        batch_size * num_keys * kv_heads * depth
    )
    var v_ptr = ctx.enqueue_create_host_buffer[qkv_type](
        batch_size * num_keys * kv_heads * depth
    )
    var out_ptr = ctx.enqueue_create_host_buffer[qkv_type](
        batch_size * seq_len * num_heads * depth
    )
    var sinks_ptr = ctx.enqueue_create_host_buffer[qkv_type](num_heads)

    # Q,K don't matter when scale=0, but set deterministically
    for i in range(batch_size * seq_len * num_heads * depth):
        q_ptr[i] = Float32(0.123).cast[qkv_type]()
    for i in range(batch_size * num_keys * kv_heads * depth):
        k_ptr[i] = Float32(-0.456).cast[qkv_type]()

    # V = 1 so the attention output equals total probability mass assigned to
    # the real keys
    for i in range(batch_size * num_keys * kv_heads * depth):
        v_ptr[i] = Float32(1.0).cast[qkv_type]()

    # Two different sinks for the two heads
    var sink_h0 = Float32(5.0)  # large positive
    var sink_h1 = Float32(3.0)  # moderately positive
    sinks_ptr[0] = sink_h0.cast[qkv_type]()
    sinks_ptr[1] = sink_h1.cast[qkv_type]()

    var out_host = TileTensor(
        out_ptr,
        row_major((Idx[batch_size], seq_len, Idx[num_heads], Idx[depth])),
    )

    var q_dev = ctx.enqueue_create_buffer[qkv_type](
        batch_size * seq_len * num_heads * depth
    )
    var k_dev = ctx.enqueue_create_buffer[qkv_type](
        batch_size * num_keys * kv_heads * depth
    )
    var v_dev = ctx.enqueue_create_buffer[qkv_type](
        batch_size * num_keys * kv_heads * depth
    )
    var out_dev = ctx.enqueue_create_buffer[qkv_type](
        batch_size * seq_len * num_heads * depth
    )
    var sinks_dev = ctx.enqueue_create_buffer[qkv_type](num_heads)

    ctx.enqueue_copy(q_dev, q_ptr)
    ctx.enqueue_copy(k_dev, k_ptr)
    ctx.enqueue_copy(v_dev, v_ptr)
    ctx.enqueue_copy(sinks_dev, sinks_ptr)

    var q_device = TileTensor(
        q_dev,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )
    var k_device = TileTensor(
        k_dev,
        row_major((batch_size, Idx[num_keys], Idx[kv_heads], Idx[depth])),
    )
    var v_device = TileTensor(
        v_dev,
        row_major((batch_size, Idx[num_keys], Idx[kv_heads], Idx[depth])),
    )
    var out_device = TileTensor(
        out_dev,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )
    comptime sinks_layout = Layout.row_major(UNKNOWN_VALUE)
    var sinks_device = LayoutTensor[qkv_type, sinks_layout](
        sinks_dev.unsafe_ptr(),
        RuntimeLayout[sinks_layout].row_major(Index(num_heads)),
    )

    @always_inline
    def launch(ctx: DeviceContext) raises {imm}:
        flash_attention[sink=True](
            out_device,
            q_device,
            k_device,
            v_device,
            NullMask(),
            scale,  # 0.0 -> all QK logits are exactly zero
            ctx,
            None,
            sink_weights=sinks_device.as_imm().as_unsafe_any_origin(),
        )

    launch(ctx)
    ctx.synchronize()
    ctx.enqueue_copy(out_ptr, out_dev)
    ctx.synchronize()

    def expected_mass(sink: Float32) -> Float32:
        return Float32(num_keys) / (Float32(num_keys) + exp(sink))

    var want0 = expected_mass(sink_h0)
    var want1 = expected_mass(sink_h1)

    # Every element of the output vector for a given head should equal that mass
    # (since V=1)
    for s in range(seq_len):
        for d in range(depth):
            var got0 = out_host[0, s, 0, d].cast[.float32]()
            var got1 = out_host[0, s, 1, d].cast[.float32]()
            assert_almost_equal(got0, want0, atol=2e-2, rtol=2e-2)
            assert_almost_equal(got1, want1, atol=2e-2, rtol=2e-2)

    # Keep every device buffer alive until the kernel and the readback have both
    # completed. `sinks_device` wraps `sinks_dev.unsafe_ptr()` (a raw pointer,
    # no ownership), so without this keepalive `sinks_dev` can be freed before
    # the async attention kernel reads it; the allocator then reuses that memory
    # and the kernel reads a garbage sink weight (intermittently 0 -> output
    # num_keys/(num_keys+1), or a large value -> output ~0). Mirrors the
    # `_ = sink_d^` keepalive in `test_apple_fa_prefill.mojo`.
    _ = q_dev^
    _ = k_dev^
    _ = v_dev^
    _ = out_dev^
    _ = sinks_dev^


def main() raises:
    with DeviceContext() as ctx:
        test_context_encoding(ctx)
        # Test flash attention with sink kernel during encoding
        test_flash_attention_sink_kernel(ctx, 8)
        # Test flash attention with sink kernel during decoding
        test_flash_attention_sink_kernel(ctx, 1)

        comptime for split_k in range(1):
            comptime for batch_size in range(1, 5, 3):
                test_decoding[batch_size, 1, Bool(split_k)](ctx)

                comptime if (
                    ctx.default_device_info == A100
                    or ctx.default_device_info == H100
                ):
                    test_decoding_large_group[batch_size, 1](ctx)

                test_decoding[batch_size, 2, Bool(split_k)](ctx)
                test_decoding[batch_size, 4, Bool(split_k)](ctx)
                test_decoding[batch_size, None, Bool(split_k)](ctx)
                test_decoding[batch_size, 32, Bool(split_k)](ctx)
