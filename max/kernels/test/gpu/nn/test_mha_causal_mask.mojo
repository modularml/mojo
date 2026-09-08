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

from std.math import isclose
from std.random import rand
from std.sys import argv, size_of
from std.sys.defines import get_defined_int

from max.gpu import *
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.info import A100, H100, _is_sm10x_gpu
from layout import (
    Idx,
    TileTensor,
    row_major,
)
from layout import Layout, LayoutTensor
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import (
    CausalMask,
    CausalPaddingMask,
    MHAMask,
    SlidingWindowCausalMask,
)
from std.testing import assert_almost_equal, assert_equal


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def test[
    MaskType: MHAMask,
    //,
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    *,
    group: Int = 1,
](
    seq_len: Int,
    num_keys: Int,
    mask: MaskType,
    ctx: DeviceContext,
    *,
    is_benchmark: Bool = False,
    num_partitions: Optional[Int] = None,
) raises:
    print("test_mha_causal_mask")
    print(
        "qkv_type:",
        qkv_type,
        "depth:",
        depth,
        "num_heads:",
        num_heads,
        "group:",
        group,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "mask:",
        MaskType.name(),
    )
    # Query, key, value dimensions.
    comptime batch_size = 1
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size

    # Allocate memory for all variables.
    var q_ptr = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    # Q is randomly initialized.
    rand(q_ptr.as_span())

    var output_ptr = ctx.enqueue_create_host_buffer[qkv_type](o_size)
    var flash_output_ptr = ctx.enqueue_create_host_buffer[qkv_type](o_size)

    # Device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)

    # K and V are filled with uniform constants for this test instead of being
    # randomly initialized.
    var k_ptr = ctx.enqueue_create_host_buffer[qkv_type](k_size)
    rand(k_ptr.as_span())
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)

    var v_ptr = ctx.enqueue_create_host_buffer[qkv_type](v_size)
    rand(v_ptr.as_span())
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
        flash_attention(
            output_device,
            q_device,
            k_device,
            v_device,
            mask,
            scale,
            ctx,
            num_partitions=num_partitions,
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
    var output_device_ref = TileTensor(
        output_ref_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )

    mha_gpu_naive(
        q_device,
        k_device,
        v_device,
        mask,
        output_device_ref,
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

    var rtol = 1e-2
    for s in range(seq_len):
        for h in range(num_heads):
            for d in range(depth):
                var expect = output_ptr[d + depth * (h + s * num_heads)].cast[
                    DType.float64
                ]()
                var actual = flash_output_ptr[
                    d + depth * (h + s * num_heads)
                ].cast[.float64]()
                if not isclose(actual, expect, atol=1e-5, rtol=rtol):
                    var next_expect = 0 * expect
                    var next_actual = 0 * actual
                    if h < num_heads and s < seq_len and d < depth - 1:
                        next_expect = output_ptr[
                            d + depth * (h + s * num_heads) + 1
                        ].cast[.float64]()
                        next_actual = flash_output_ptr[
                            d + depth * (h + s * num_heads) + 1
                        ].cast[.float64]()
                    var rerr = abs((actual - expect) / expect)
                    print(
                        "s, h, d = ",
                        "(" + String(s),
                        h,
                        String(d) + ")",
                        "actual =",
                        actual,
                        "expect =",
                        expect,
                        "rerr =",
                        rerr,
                        "next_expect =",
                        next_expect,
                        "next_actual =",
                        next_actual,
                    )

                assert_almost_equal(actual, expect, atol=1e-5, rtol=rtol)

    for repeat in range(16):
        # test reproducibility
        flash_attention(
            output_device_ref,
            q_device,
            k_device,
            v_device,
            mask,
            scale,
            ctx,
            num_partitions=num_partitions,
        )
        ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
        ctx.synchronize()
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(depth):
                    var orig = flash_output_ptr[d + depth * (h + s * num_heads)]
                    var rep = output_ptr[d + depth * (h + s * num_heads)]
                    if rep != orig:
                        print("repeat s h d =", repeat, s, h, d)
                    assert_equal(rep, orig)
                    output_ptr[d + depth * (h + s * num_heads)] = 123.4567

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = output_device_ptr
    _ = output_ref_device_ptr


def main() raises:
    with DeviceContext() as ctx:
        comptime depth = get_defined_int["depth", 128]()

        comptime if depth <= 128:
            # fp32 tf32-fp32 mma
            test[.float32, depth, 1](
                128, 128, CausalMask(), ctx, is_benchmark=is_benchmark()
            )

            test[
                DType.float32,
                depth,
                3,
            ](14, 14, CausalMask(), ctx, is_benchmark=is_benchmark())

            test[
                DType.float32,
                depth,
                1,
            ](178, 178, CausalMask(), ctx, is_benchmark=is_benchmark())

        # bf16 bf16-fp32 mma
        test[
            DType.bfloat16,
            depth=depth,
            num_heads=1,
        ](128, 128, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth=depth,
            num_heads=1,
        ](384, 384, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            24,
            group=8,
        ](1024, 1024, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            24,
            group=8,
        ](1024, 1024, SlidingWindowCausalMask[128](), ctx)

        # BF16 with sequence length not multiple of 128
        test[
            DType.bfloat16,
            depth,
            3,
            group=3,
        ](64, 64, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            3,
            group=3,
        ](102, 102, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            1,
        ](14, 14, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            1,
        ](528, 528, CausalMask(), ctx)

        # BF16 with different length for prompt and cache.
        test[
            DType.bfloat16,
            depth,
            1,
        ](128, 256, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            3,
            group=3,
        ](32, 77, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            16,
            group=8,
        ](201, 400, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            12,
            group=4,
        ](1000, 2000, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            12,
            group=4,
        ](1000, 2000, SlidingWindowCausalMask[100](), ctx)

        test[
            DType.bfloat16,
            depth,
            32,
            group=4,
        ](201, 600, CausalMask(), ctx)

        # BF16 token gen
        test[
            DType.bfloat16,
            depth,
            32,
        ](1, 512, CausalMask(), ctx, is_benchmark=is_benchmark())

        test[
            DType.bfloat16,
            depth,
            11,
        ](1, 256, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            1,
        ](1, 11, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            1,
        ](1, 11, CausalMask(), ctx, num_partitions=Optional[Int](2))

        test[
            DType.bfloat16,
            depth,
            2,
        ](1, 523, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            24,
            group=3,
        ](1, 29, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            3,
            group=3,
        ](1, 156, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            3,
            group=3,
        ](1, 208, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            32,
            group=4,
        ](1, 1208, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            32,
            group=4,
        ](1, 2008, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            32,
            group=4,
        ](1, 2008, SlidingWindowCausalMask[77](), ctx)

        test[
            DType.bfloat16,
            depth,
            32,
            group=4,
        ](1, 5000, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            32,
            group=4,
        ](1, 5000, SlidingWindowCausalMask[89](), ctx)

        test[
            DType.bfloat16,
            depth,
            32,
            group=4,
        ](1, 600, CausalMask(), ctx)

        comptime if (
            ctx.default_device_info == A100
            or ctx.default_device_info == H100
            or _is_sm10x_gpu(ctx.default_device_info)
        ):
            test[
                DType.bfloat16,
                depth,
                32,
                group=16,
            ](1, 2008, CausalMask(), ctx)

        # Regression: window=512 over 1025 keys has visible window <= window
        # size, which previously picked a decode split-K partition count
        # based on the raw cache length (1025 > 512 -> 2 partitions) and
        # produced NaNs. Using the visible-window key count instead picks
        # a single partition and keeps the output finite.
        comptime if depth == 128:
            test[
                DType.bfloat16,
                depth,
                96,
                group=12,
            ](1, 1025, SlidingWindowCausalMask[512](), ctx)

        # CausalPaddingMask tests: allocate valid_lengths on device.
        #
        # `CausalPaddingMask` stores only a raw device pointer, so the buffer
        # backing it has to stay alive across the whole launch, and TWO things
        # are needed for that. Returning the view alone would destroy the
        # buffer at the helper's return, so the helper hands back the
        # `DeviceBuffer` and the view is built at the call site. That is not
        # sufficient on its own: `MutUntrackedOrigin` opts the view out of
        # origin tracking, so the buffer's last use is the `vl_view(...)`
        # argument expression, and ASAP destruction is free to free it before
        # `test` has launched anything. Each call site therefore consumes its
        # buffer with `_ = ...^` AFTER the `test` call, which is what actually
        # pins the allocation across the kernel. `test` synchronizes before it
        # returns, so that is the full extent of the launch.
        def vl_buffer(
            val: UInt32, ctx: DeviceContext
        ) raises -> DeviceBuffer[.uint32]:
            var dev_buf = ctx.enqueue_create_buffer[.uint32](1)
            ctx.enqueue_memset(dev_buf, val)
            return dev_buf

        # `mut buf`: `unsafe_origin_cast` can only retarget an origin of the
        # same mutability, so the pointer has to come from a mutable borrow.
        def vl_view(
            mut buf: DeviceBuffer[.uint32],
        ) -> LayoutTensor[.uint32, Layout.row_major(1), MutUntrackedOrigin]:
            return {buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()}

        # valid_length == num_keys (equivalent to CausalMask).
        var vl_128_buf = vl_buffer(128, ctx)
        test[
            DType.bfloat16,
            depth,
            1,
        ](128, 128, CausalPaddingMask(vl_view(vl_128_buf)), ctx)
        _ = vl_128_buf^

        # valid_length < num_keys (padding active).
        var vl_100_buf = vl_buffer(100, ctx)
        test[
            DType.bfloat16,
            depth,
            1,
        ](128, 128, CausalPaddingMask(vl_view(vl_100_buf)), ctx)
        _ = vl_100_buf^

        # CausalPaddingMask with GQA.
        var vl_384_buf = vl_buffer(384, ctx)
        test[
            DType.bfloat16,
            depth,
            24,
            group=3,
        ](384, 384, CausalPaddingMask(vl_view(vl_384_buf)), ctx)
        _ = vl_384_buf^

        # CausalPaddingMask with padding and GQA.
        var vl_300_buf = vl_buffer(300, ctx)
        test[
            DType.bfloat16,
            depth,
            24,
            group=3,
        ](384, 384, CausalPaddingMask(vl_view(vl_300_buf)), ctx)
        _ = vl_300_buf^

        # CausalPaddingMask: token gen with padding.
        var vl_400_buf = vl_buffer(400, ctx)
        test[
            DType.bfloat16,
            depth,
            32,
        ](1, 512, CausalPaddingMask(vl_view(vl_400_buf)), ctx)
        _ = vl_400_buf^
