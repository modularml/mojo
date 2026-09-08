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
from std.sys import argv


from max.gpu import *
from max.gpu.host import DeviceContext
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from std.memory import unsafe_memset_zero
from nn.attention.gpu.mha import (
    _naive_attention_with_transpose,
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


def test[
    mask_rank: Int,
    qkv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    against_gpu_naive: Bool = False,
    batch_size: Int = 1,
    num_partitions: Optional[Int] = None,
](
    seq_len: Int,
    num_keys: Int,
    ctx: DeviceContext,
    is_benchmark: Bool = False,
    use_index_input: Bool = False,
    use_adversarial_softmax_input: Bool = False,
) raises:
    print(
        "test_flash_attention",
        "batch_size:",
        batch_size,
        "num_partitions:",
        num_partitions.value() if num_partitions else -1,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "qkv_type:",
        qkv_type,
        "mask_type:",
        mask_type,
        "mask_rank:",
        mask_rank,
        "depth:",
        depth,
    )

    comptime assert mask_rank in (3, 4), "mha only support rank 3 or 4."
    comptime assert (
        against_gpu_naive or mask_rank == 3
    ), "Testing against cpu requires mask of rank 3."

    # Query, key, value dimensions.
    comptime scale = Float32(0.25)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size
    var mask_size = (
        (num_heads if mask_rank == 4 else 1) * seq_len * num_keys * batch_size
    )

    # Allocate memory for all variables.
    var q_ptr = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_ptr = ctx.enqueue_create_host_buffer[qkv_type](k_size)
    var v_ptr = ctx.enqueue_create_host_buffer[qkv_type](v_size)
    var mask_ptr = ctx.enqueue_create_host_buffer[mask_type](mask_size)
    var output_ptr = ctx.enqueue_create_host_buffer[qkv_type](o_size)
    var flash_output_ptr = ctx.enqueue_create_host_buffer[qkv_type](o_size)

    # Q, K, V are randomly initialized.
    if use_adversarial_softmax_input:
        # Large-magnitude Q/K across two K tiles — forces fp32 Inf in the
        # `_fma` softmax correction while the prescaled path stays finite.
        assert batch_size == 1
        assert num_keys >= 256, "need ≥ 2 tiles of BN=128"
        comptime q_val = Scalar[qkv_type](5.0)
        comptime k_large = Scalar[qkv_type](100.0)
        comptime k_small = Scalar[qkv_type](1.0)
        for i in range(seq_len):
            for h in range(num_heads):
                for j in range(depth):
                    q_ptr[(i * num_heads + h) * depth + j] = q_val
        for i in range(num_keys):
            var is_first_tile = i < 128
            var k_val = k_large if is_first_tile else k_small
            var v_val = Scalar[qkv_type](1.0) if is_first_tile else Scalar[
                qkv_type
            ](-1.0)
            for h in range(kv_num_heads):
                for j in range(depth):
                    k_ptr[(i * kv_num_heads + h) * depth + j] = k_val
                    v_ptr[(i * kv_num_heads + h) * depth + j] = v_val
    elif use_index_input:
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
        rand(q_ptr.as_span())
        rand(k_ptr.as_span())
        rand(v_ptr.as_span())

    # Zero-initialize mask_ptr (HostBuffer is uninitialized).
    for i in range(mask_size):
        mask_ptr[i] = Scalar[mask_type](0)

    # Construct buffers.
    comptime layout_4d = Layout.row_major[4]()
    var q = LayoutTensor[qkv_type, layout_4d](
        q_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )
    var k = LayoutTensor[qkv_type, layout_4d](
        k_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
    )
    var v = LayoutTensor[qkv_type, layout_4d](
        v_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
    )
    var mask = LayoutTensor[mask_type, Layout.row_major[2]()](
        mask_ptr,
        RuntimeLayout[Layout.row_major[2]()].row_major(
            Index(seq_len, num_keys)
        ),
    )
    var output = LayoutTensor[qkv_type, layout_4d](
        output_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    var flash_output = LayoutTensor[qkv_type, layout_4d](
        flash_output_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    comptime if not against_gpu_naive:
        comptime assert (
            qkv_type == mask_type
        ), "expect qkv and mask have same type for CPU."
        _naive_attention_with_transpose[qkv_type](
            output,
            q,
            k,
            v,
            mask.bitcast[qkv_type](),
            scale,
            ctx,
        )

    # Device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_size)
    var mask_device_ptr = ctx.enqueue_create_buffer[mask_type](mask_size)
    var output_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)

    # Copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(v_device_ptr, v_ptr)
    ctx.enqueue_copy(mask_device_ptr, mask_ptr)

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
    var mask3d = TileTensor(
        mask_device_ptr,
        row_major(batch_size, seq_len, num_keys),
    )
    var mask4d = TileTensor(
        mask_device_ptr,
        row_major((batch_size, num_heads, seq_len, num_keys)),
    )
    var output_device = TileTensor(
        output_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )

    @always_inline
    def kernel_launch(
        ctx: DeviceContext,
    ) raises {var q_device, var k_device, var v_device, var output_device, imm}:
        comptime if mask_rank == 3:
            flash_attention(
                output_device,
                q_device,
                k_device,
                v_device,
                NullMask(),
                scale,
                ctx,
                num_partitions,
            )
        else:
            flash_attention(
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

    comptime if against_gpu_naive:
        var output_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
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

        comptime if mask_rank == 3:
            mha_gpu_naive(
                q_device,
                k_device,
                v_device,
                mask3d,
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
        elif mask_rank == 4:
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

        ctx.synchronize()
        ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
        _ = output_ref_device_ptr

    ctx.synchronize()

    # This is useful for debugging.

    var rtol = 2e-2
    for bs in range(batch_size):
        for h in range(num_heads):
            for s in range(seq_len):
                for d in range(depth):
                    var actual = flash_output[bs, s, Int(h), Int(d)][0]
                    var expect = output[bs, s, Int(h), Int(d)][0]
                    if not isclose(actual, expect, atol=1e-5, rtol=rtol):
                        var rerr = abs((actual - expect) / expect)
                        print(h, s, d, actual, expect, rerr)
                    assert_almost_equal(actual, expect, atol=1e-5, rtol=rtol)

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = mask_device_ptr
    _ = output_device_ptr


def test_context_encoding[
    batch_size: Int, depth: Int
](ctx: DeviceContext) raises:
    # # fp32 arbitrary depth and num_heads, baseline impl.
    # test[3, DType.float32, DType.float32, 127, 2](111, 121, ctx)
    # # fp32 depth == 128, tf32-fp32 mma, llama2 shape.
    # test[
    #     4,
    #     DType.float32,
    #     DType.float32,
    #     128,
    #     32,
    #     against_gpu_naive=True,
    # ](1024, 1024, ctx, is_benchmark())
    # test[
    #     3,
    #     DType.float32,
    #     DType.float32,
    #     128,
    #     3,
    #     against_gpu_naive=True,
    # ](14, 14, ctx, is_benchmark())
    # test[
    #     3,
    #     DType.float32,
    #     DType.float32,
    #     128,
    #     1,
    #     against_gpu_naive=True,
    # ](178, 178, ctx, is_benchmark())
    # bf16 depth == 128, bf16-fp32 mma
    test[
        4,
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=1,
        against_gpu_naive=True,
    ](128, 128, ctx, use_index_input=False)
    test[
        4,
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=1,
        against_gpu_naive=True,
    ](384, 384, ctx)
    test[
        3,
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=3,
        against_gpu_naive=True,
    ](256, 256, ctx)
    test[
        4,
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=32,
        against_gpu_naive=True,
    ](1024, 1024, ctx, is_benchmark())
    test[
        4,
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=24,
        group=3,
        against_gpu_naive=True,
    ](1024, 1024, ctx)
    # BF16 with sequence length not multiple of 128
    test[
        4,
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=3,
        group=3,
        against_gpu_naive=True,
    ](64, 64, ctx)
    test[
        4,
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=3,
        group=3,
        against_gpu_naive=True,
    ](102, 102, ctx)
    test[
        3,
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=1,
        against_gpu_naive=True,
    ](14, 14, ctx)
    test[
        3,
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=1,
        against_gpu_naive=True,
    ](528, 528, ctx)

    test[
        4,
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=32,
        group=4,
        against_gpu_naive=True,
    ](1, 1, ctx)

    # Adversarial softmax input: prescaled passes, `_fma` NaNs (overflow).
    # Fill assumes BN=128, so guarded on depth == 128.
    comptime if depth == 128:
        test[
            4,
            DType.bfloat16,
            DType.bfloat16,
            depth=depth,
            num_heads=16,
            group=8,
            against_gpu_naive=True,
        ](256, 256, ctx, use_adversarial_softmax_input=True)

    # Long-context AMD CDNA prefill gate. seq_len=4096 with BF16-output
    # causal prefill on CDNA fires the gate in `flash_attention_dispatch`,
    # routing through `mha_prefill_v2`. Validates that (a) the gate builds
    # + launches the kernel correctly, (b) the `LayoutTensor → TileTensor`
    # adapters in the gate preserve the data, and (c) the output matches
    # the gpu_naive reference within BF16 attention tolerance. Guarded on
    # depth in {64, 128} (the gate's depth eligibility); other depths fall
    # through to FA2.
    comptime if depth == 64 or depth == 128:
        test[
            4,
            DType.bfloat16,
            DType.bfloat16,
            depth=depth,
            num_heads=16,
            against_gpu_naive=True,
        ](4096, 4096, ctx)


def test_decoding[
    batch_size: Int,
    depth: Int,
    num_partitions: Optional[Int] = None,
    qkv_type: DType = .bfloat16,
](ctx: DeviceContext, use_index_input: Bool) raises:
    # fp32 arbitrary depth and num_heads, baseline impl.
    # BF16 token gen
    test[
        3,
        qkv_type,
        DType.float32,
        depth=depth,
        num_heads=2,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
    ](1, 512, ctx, use_index_input=use_index_input)
    test[
        4,
        qkv_type,
        DType.bfloat16,
        depth=depth,
        num_heads=2,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
    ](1, 512, ctx, use_index_input=use_index_input)
    test[
        3,
        qkv_type,
        DType.float32,
        depth=depth,
        num_heads=1,
        group=1,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
    ](1, 128, ctx, use_index_input=use_index_input)
    test[
        4,
        qkv_type,
        DType.bfloat16,
        depth=depth,
        num_heads=3,
        group=3,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
    ](1, 156, ctx, use_index_input=use_index_input)
    test[
        4,
        qkv_type,
        DType.bfloat16,
        depth=depth,
        num_heads=3,
        group=3,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
    ](1, 208, ctx, use_index_input=use_index_input)

    test[
        4,
        qkv_type,
        DType.bfloat16,
        depth=depth,
        num_heads=32,
        group=4,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
    ](1, 1, ctx, use_index_input=use_index_input)
    test[
        3,
        qkv_type,
        DType.float32,
        depth=depth,
        num_heads=2,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
    ](1, 5120, ctx, use_index_input=use_index_input)

    # Stress softmax numerical stability for decode. use_index_input
    # requires batch_size=1.
    comptime if batch_size == 1:
        test[
            4,
            DType.bfloat16,
            DType.bfloat16,
            depth=depth,
            num_heads=16,
            group=8,
            against_gpu_naive=True,
            batch_size=1,
            num_partitions=num_partitions,
        ](1, 128, ctx, use_index_input=True)

        # Adversarial softmax input: prescaled passes, `_fma` NaNs (overflow).
        comptime if depth == 128:
            test[
                4,
                DType.bfloat16,
                DType.bfloat16,
                depth=depth,
                num_heads=16,
                group=8,
                against_gpu_naive=True,
                batch_size=1,
                num_partitions=num_partitions,
            ](1, 256, ctx, use_adversarial_softmax_input=True)


def main() raises:
    with DeviceContext() as ctx:
        comptime depths = [64, 80, 128, 256, 512]

        comptime for i in range(len(depths)):
            comptime depth = depths[i]
            test_context_encoding[1, depth](ctx)

            comptime for batch_size in range(1, 5, 3):
                test_decoding[batch_size, depth, 1](ctx, False)
                test_decoding[batch_size, depth, 1, DType.float32](ctx, False)
            test_decoding[1, depth, None](ctx, False)
