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

from std.random import randn
from std.sys import (
    argv,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)

from max.gpu import *
from max.gpu.host import DeviceContext
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_operand import LayoutTensorMHAOperand
from nn.attention.gpu.mla import flare_mla_decoding, flare_mla_prefill
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from std.testing import assert_almost_equal
from max.gpu.host.info import _is_sm10x_gpu


from std.utils.index import Index


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def test[
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    against_gpu_naive: Bool = False,
    batch_size: Int = 1,
    num_partitions: Optional[Int] = None,
    decoding_warp_split_k: Bool = False,
    output_type: DType = qkv_type,
](
    seq_len: Int,
    num_keys: Int,
    ctx: DeviceContext,
    use_index_input: Bool = False,
) raises:
    print(
        "test_mla_decoding",
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
        "output_type:",
        output_type,
        "depth:",
        depth,
        "num_heads:",
        num_heads,
        "group:",
        group,
    )

    # Query, key, value dimensions.
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group
    # MLA: output's last dim is depth_v (= depth - rope_dim = depth - 64).
    # The reference path (mha_gpu_naive) writes the full `depth` columns
    # because it's a generic MHA reference, but only the first depth_v
    # columns are compared.
    comptime depth_v = depth - 64

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    # var v_size = k_size
    var o_size = batch_size * num_heads * seq_len * depth_v
    var o_size_ref = batch_size * num_heads * seq_len * depth

    # Allocate memory for all variables.
    var q_ptr = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_ptr = ctx.enqueue_create_host_buffer[qkv_type](k_size)
    var output_ptr = ctx.enqueue_create_host_buffer[output_type](o_size_ref)
    var flash_output_ptr = ctx.enqueue_create_host_buffer[output_type](o_size)

    for i in range(o_size_ref):
        output_ptr[i] = Scalar[output_type](0)

    # Q, K, V are randomly initialized.
    if use_index_input:
        assert batch_size == 1
        for i in range(seq_len):
            for h in range(num_heads):
                for j in range(depth):
                    q_ptr[(i * num_heads + h) * depth + j] = Scalar[
                        DType.float32
                    ](i * depth + j).cast[qkv_type]()
        for i in range(num_keys):
            for h in range(kv_num_heads):
                for j in range(depth):
                    k_ptr[(i * kv_num_heads + h) * depth + j] = Scalar[
                        DType.float32
                    ](i * depth + j).cast[qkv_type]()

    else:
        randn(q_ptr.as_span())
        randn(k_ptr.as_span())

    # Device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var output_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)

    # Copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)

    # Construct device TileTensors.
    var q_device = TileTensor(
        q_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )
    var k_device = TileTensor(
        k_device_ptr,
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
        output_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth_v])),
    )

    comptime q_tile_num_rows = 32
    comptime k_tile_num_rows = 128

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        _is_cache_length_accurate=True,
    ](batch_size, num_keys, seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    @always_inline
    def kernel_launch(
        ctx: DeviceContext,
    ) raises {
        var q_device,
        var k_device,
        var output_device,
        var scalar_args_buf_tt,
        imm,
    }:
        flare_mla_decoding[
            config=MHAConfig[qkv_type](num_heads, depth),
            decoding_warp_split_k=decoding_warp_split_k,
        ](
            output_device.as_unsafe_any_origin(),
            q_device,
            k_device,
            CausalMask(),
            scale,
            ctx,
            scalar_args_buf_tt,
            num_partitions=num_partitions,
        )

    if is_benchmark():
        comptime nrun = 200

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
        var output_ref_device_ptr = ctx.enqueue_create_buffer[output_type](
            o_size_ref
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

        var k_operand = LayoutTensorMHAOperand(
            k_device.as_immut().as_unsafe_any_origin()
        )
        var null_valid_length = LayoutTensor[
            .uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
        ](
            None,
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(Index(0)),
        )
        mha_gpu_naive[_is_cache_length_accurate=True,](
            q_device.to_layout_tensor(),
            k_operand,
            k_operand,
            CausalMask(),
            output_ref_device.to_layout_tensor(),
            null_valid_length,
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

    if o_size == 0:
        return

    # since we pass the whole K tensor as the V tensor to our naive mha kernel,
    # the last 64 elements of each head in the reference result are invalid.
    # b , s, h, d
    var rtol = 1e-3
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(depth - 64):
                    var expect = output_ptr[
                        d
                        + depth * (h + s * num_heads)
                        + b * depth * num_heads * seq_len
                    ].cast[.float64]()
                    var actual = flash_output_ptr[
                        d
                        + (depth - 64) * (h + s * num_heads)
                        + b * (depth - 64) * num_heads * seq_len
                    ].cast[.float64]()
                    # if not isclose(actual, expect, atol=1e-3, rtol=rtol):
                    #     var rerr = abs((actual - expect) / expect)
                    #     print(h, s, d, actual, expect, rerr)
                    if abs((actual - expect)) > 9e-2:
                        print(b, h, s, d, actual, expect)
                    assert_almost_equal(actual, expect, atol=1e-1, rtol=rtol)

    _ = mla_args
    _ = q_device_ptr
    _ = k_device_ptr
    _ = output_device_ptr


def test_prefill[
    qkv_type: DType,
    k_rope_type: DType,
    depth: Int,
    num_heads: Int,
    kv_depth: Int,
    cache_depth: Int,
    cache_num_heads: Int,
    batch_size: Int = 1,
    use_causal_mask: Bool = True,
    output_type: DType = qkv_type,
](seq_len: Int, num_keys: Int, ctx: DeviceContext,) raises:
    print(
        "test_mla_prefill",
        "batch_size:",
        batch_size,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "qkv_type:",
        qkv_type,
        "k_rope_type:",
        k_rope_type,
        "output_type:",
        output_type,
        "depth:",
        depth,
        "kv_depth:",
        kv_depth,
        "cache_depth:",
        cache_depth,
        "cache_num_heads:",
        cache_num_heads,
    )

    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))

    var q_size = batch_size * seq_len * num_heads * depth
    var k_size = batch_size * num_keys * num_heads * kv_depth
    var v_size = k_size
    var o_size = batch_size * seq_len * num_heads * kv_depth
    var cache_size = batch_size * num_keys * cache_num_heads * cache_depth

    var q_ptr = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_ptr = ctx.enqueue_create_host_buffer[qkv_type](k_size)
    var v_ptr = ctx.enqueue_create_host_buffer[qkv_type](v_size)
    var cache_ptr = ctx.enqueue_create_host_buffer[k_rope_type](cache_size)
    var output_ptr = ctx.enqueue_create_host_buffer[output_type](o_size)

    # Q, K, V, cache are randomly initialized.
    randn(q_ptr.as_span())
    randn(k_ptr.as_span())
    randn(v_ptr.as_span())
    randn(cache_ptr.as_span())

    # input row offsets and cache row offsets
    var input_row_offsets = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    var cache_row_offsets = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size):
        input_row_offsets[i] = UInt32(i * seq_len)
        cache_row_offsets[i] = UInt32(i * num_keys)
    input_row_offsets[batch_size] = UInt32(batch_size * seq_len)
    cache_row_offsets[batch_size] = UInt32(batch_size * num_keys)

    # ragged inputs
    var q = TileTensor(
        q_ptr,
        row_major((batch_size * seq_len, Idx[num_heads], Idx[depth])),
    )
    var k = TileTensor(
        k_ptr,
        row_major((batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var v = TileTensor(
        v_ptr,
        row_major((batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var cache = TileTensor(
        cache_ptr,
        row_major(
            (
                batch_size,
                num_keys,
                Idx[cache_num_heads],
                Idx[cache_depth],
            )
        ),
    )
    var output = TileTensor(
        output_ptr,
        row_major((batch_size * seq_len, Idx[num_heads], Idx[kv_depth])),
    )

    # device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_size)
    var cache_device_ptr = ctx.enqueue_create_buffer[k_rope_type](cache_size)
    var output_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)
    var input_row_offsets_device_ptr = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var cache_row_offsets_device_ptr = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )

    # copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(v_device_ptr, v_ptr)
    ctx.enqueue_copy(cache_device_ptr, cache_ptr)
    ctx.enqueue_copy(input_row_offsets_device_ptr, input_row_offsets)
    ctx.enqueue_copy(cache_row_offsets_device_ptr, cache_row_offsets)

    # construct device TileTensors
    var q_device = TileTensor(
        q_device_ptr,
        row_major((batch_size * seq_len, Idx[num_heads], Idx[depth])),
    )
    var k_device = TileTensor(
        k_device_ptr,
        row_major((batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var v_device = TileTensor(
        v_device_ptr,
        row_major((batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var cache_device = TileTensor(
        cache_device_ptr,
        row_major(
            (
                batch_size,
                num_keys,
                Idx[cache_num_heads],
                Idx[cache_depth],
            )
        ),
    )
    var output_device = TileTensor(
        output_device_ptr,
        row_major((batch_size * seq_len, Idx[num_heads], Idx[kv_depth])),
    )
    var input_row_offsets_device = TileTensor(
        input_row_offsets_device_ptr,
        row_major(batch_size + 1),
    )
    var cache_row_offsets_device = TileTensor(
        cache_row_offsets_device_ptr,
        row_major(batch_size + 1),
    )

    @always_inline
    def kernel_launch(
        ctx: DeviceContext,
    ) raises {
        var q_device,
        var k_device,
        var v_device,
        var cache_device,
        var input_row_offsets_device,
        var cache_row_offsets_device,
        var output_device,
        imm,
    }:
        flare_mla_prefill[rank=3](
            output_device,
            q_device,
            k_device,
            v_device,
            cache_device,
            CausalMask(),
            input_row_offsets_device,
            cache_row_offsets_device,
            scale,
            ctx,
            q_max_seq_len=seq_len,
        )

    if is_benchmark():
        comptime nrun = 200

        # Warmup
        for _i in range(20):
            kernel_launch(ctx)

        var nstime = Float64(ctx.execution_time(kernel_launch, nrun)) / Float64(
            nrun
        )
        var sectime = nstime / 1000000

        var tflops = (
            Float64(2)
            * Float64(batch_size)
            * Float64(num_heads)
            * Float64(-seq_len * seq_len + 2 * seq_len * num_keys)
            * Float64(depth + kv_depth)
            / sectime
            / 1e9
        )
        print(nrun, "runs avg: ", sectime, " ms   ", tflops, " TFLOPs")

    else:
        kernel_launch(ctx)

    ctx.synchronize()
    ctx.enqueue_copy(output_ptr, output_device_ptr)

    # create reference K and V
    # unlike flare_mla_prefill, K_ref and V_ref each head is of size depth (not kv_depth)
    var k_ref_ptr = ctx.enqueue_create_host_buffer[qkv_type](
        batch_size * num_keys * num_heads * depth
    )
    var v_ref_ptr = ctx.enqueue_create_host_buffer[qkv_type](
        batch_size * num_keys * num_heads * depth
    )
    var output_ref_ptr = ctx.enqueue_create_host_buffer[output_type](
        batch_size * seq_len * num_heads * depth
    )

    # create reference K and V
    var k_ref = TileTensor(
        k_ref_ptr,
        row_major((batch_size, num_keys, Idx[num_heads], Idx[depth])),
    )
    var v_ref = TileTensor(
        v_ref_ptr,
        row_major((batch_size, num_keys, Idx[num_heads], Idx[depth])),
    )
    var output_ref = TileTensor(
        output_ref_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )

    # the first kv_depth elements of each head in K_ref and V_ref are the same as K and V
    for b in range(batch_size):
        for s in range(num_keys):
            for h in range(num_heads):
                for d in range(kv_depth):
                    k_ref[b, s, h, d] = k[b * num_keys + s, h, d]
                    v_ref[b, s, h, d] = v[b * num_keys + s, h, d]

    # the rest of the elements in K_ref are broadcasted from the last (depth - kv_depth) elements of the head in cache
    # the rest of the elements in V_ref are zeros
    for b in range(batch_size):
        for s in range(num_keys):
            for h in range(num_heads):
                for d in range(depth - kv_depth):
                    k_ref[b, s, h, d + kv_depth] = Scalar[qkv_type](
                        cache[b, s, 0, cache_depth - (depth - kv_depth) + d][0]
                    )
                    v_ref[b, s, h, d + kv_depth] = 0

    # view q_device as a rank 4 buffer
    var q_device_rank4 = TileTensor(
        q_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )

    # create device pointers for K_ref and V_ref
    var k_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](
        batch_size * num_keys * num_heads * depth
    )
    var v_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](
        batch_size * num_keys * num_heads * depth
    )
    var output_ref_device_ptr = ctx.enqueue_create_buffer[output_type](
        batch_size * seq_len * num_heads * depth
    )
    # create device TileTensors for K_ref and V_ref
    var k_ref_device = TileTensor(
        k_ref_device_ptr,
        row_major((batch_size, num_keys, Idx[num_heads], Idx[depth])),
    )
    var v_ref_device = TileTensor(
        v_ref_device_ptr,
        row_major((batch_size, num_keys, Idx[num_heads], Idx[depth])),
    )
    var output_ref_device = TileTensor(
        output_ref_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )

    # copy from host to device
    ctx.enqueue_copy(k_ref_device_ptr, k_ref_ptr)
    ctx.enqueue_copy(v_ref_device_ptr, v_ref_ptr)

    var null_valid_length = LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
    ](
        None,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(Index(0)),
    )

    var k_ref_operand = LayoutTensorMHAOperand(
        k_ref_device.as_immut().as_unsafe_any_origin()
    )
    var v_ref_operand = LayoutTensorMHAOperand(
        v_ref_device.as_immut().as_unsafe_any_origin()
    )

    # create reference output
    mha_gpu_naive[_is_cache_length_accurate=True](
        q_device_rank4.to_layout_tensor(),
        k_ref_operand,
        v_ref_operand,
        CausalMask(),
        output_ref_device.to_layout_tensor(),
        null_valid_length,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        1,
        ctx,
    )

    ctx.enqueue_copy(output_ref_ptr, output_ref_device_ptr)
    ctx.synchronize()

    # view output as a rank 4 buffer
    var output_rank4 = TileTensor(
        output_ptr,
        row_major(
            (
                batch_size,
                seq_len,
                Idx[num_heads],
                Idx[kv_depth],
            )
        ),
    )

    # compare output with reference
    comptime atol: Float64 = 2e-2
    comptime rtol: Float64 = 2e-2 if has_nvidia_gpu_accelerator() else 3e-2
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(kv_depth):
                    var lhs = output_rank4[b, s, h, d]
                    var rhs = output_ref[b, s, h, d]
                    if abs((lhs - rhs)).cast[.float64]() > atol:
                        print(b, s, h, d, lhs, rhs)
                    # print(b, s, h, d, lhs, rhs)
                    assert_almost_equal(
                        lhs,
                        rhs,
                        atol=atol,
                        rtol=rtol,
                    )

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = cache_device_ptr
    _ = output_device_ptr
    _ = k_ref_device_ptr
    _ = v_ref_device_ptr
    _ = output_ref_device_ptr


def test_decoding[
    batch_size: Int,
    num_partitions: Optional[Int],
    split_k: Bool,
    qkv_type: DType = .bfloat16,
    output_type: DType = qkv_type,
](ctx: DeviceContext, use_index_input: Bool) raises:
    comptime if _is_sm10x_gpu(ctx.default_device_info):
        if batch_size <= 2:
            test[
                qkv_type,
                576,
                128,
                group=128,
                against_gpu_naive=True,
                batch_size=batch_size,
                num_partitions=num_partitions,
                decoding_warp_split_k=split_k,
                output_type=output_type,
            ](1, 32768, ctx, use_index_input=use_index_input)
            test[
                qkv_type,
                576,
                128,
                group=128,
                against_gpu_naive=True,
                batch_size=batch_size,
                num_partitions=num_partitions,
                decoding_warp_split_k=split_k,
                output_type=output_type,
            ](1, 32768 * 2, ctx, use_index_input=use_index_input)
        else:
            for seq_len in range(1, 9):
                # BF16 token gen
                test[
                    qkv_type,
                    576,
                    128,
                    group=128,
                    against_gpu_naive=True,
                    batch_size=batch_size,
                    num_partitions=num_partitions,
                    decoding_warp_split_k=split_k,
                    output_type=output_type,
                ](seq_len, 50, ctx, use_index_input=use_index_input)

                # BF16 token gen, with num_heads=16 (deepseek-v2 lite)
                test[
                    qkv_type,
                    576,
                    16,
                    group=16,
                    against_gpu_naive=True,
                    batch_size=batch_size,
                    num_partitions=num_partitions,
                    decoding_warp_split_k=split_k,
                    output_type=output_type,
                ](seq_len, 50, ctx, use_index_input=use_index_input)

            test[
                qkv_type,
                576,
                128,
                group=128,
                against_gpu_naive=True,
                batch_size=batch_size,
                num_partitions=num_partitions,
                decoding_warp_split_k=split_k,
                output_type=output_type,
            ](1, 4096, ctx, use_index_input=use_index_input)

            test[
                qkv_type,
                576,
                16,
                group=16,
                against_gpu_naive=True,
                batch_size=batch_size,
                num_partitions=num_partitions,
                decoding_warp_split_k=split_k,
                output_type=output_type,
            ](2, 4096, ctx, use_index_input=use_index_input)

            test[
                qkv_type,
                576,
                128,
                group=128,
                against_gpu_naive=True,
                batch_size=batch_size,
                num_partitions=num_partitions,
                decoding_warp_split_k=split_k,
                output_type=output_type,
            ](3, 1024, ctx, use_index_input=use_index_input)

            test[
                qkv_type,
                576,
                16,
                group=16,
                against_gpu_naive=True,
                batch_size=batch_size,
                num_partitions=num_partitions,
                decoding_warp_split_k=split_k,
                output_type=output_type,
            ](4, 1024, ctx, use_index_input=use_index_input)

    else:  # H100 AND AMD
        # BF16 token gen
        test[
            qkv_type,
            576,
            128,
            group=128,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
            output_type=output_type,
        ](1, 50, ctx, use_index_input=use_index_input)

        test[
            qkv_type,
            576,
            128,
            group=128,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
            output_type=output_type,
        ](1, 1024, ctx, use_index_input=use_index_input)

        test[
            qkv_type,
            576,
            128,
            group=128,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
            output_type=output_type,
        ](1, 4096, ctx, use_index_input=use_index_input)
        # BF16 token gen, with num_heads=16 (deepseek-v2 lite)
        test[
            qkv_type,
            576,
            16,
            group=16,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
            output_type=output_type,
        ](1, 50, ctx, use_index_input=use_index_input)

        test[
            qkv_type,
            576,
            16,
            group=16,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
            output_type=output_type,
        ](1, 1024, ctx, use_index_input=use_index_input)

        # BF16 token gen, with num_heads=64 (intermediate group size)
        test[
            qkv_type,
            576,
            64,
            group=64,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
            output_type=output_type,
        ](1, 50, ctx, use_index_input=use_index_input)

        test[
            qkv_type,
            576,
            64,
            group=64,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
            output_type=output_type,
        ](1, 1024, ctx, use_index_input=use_index_input)

        test[
            qkv_type,
            576,
            64,
            group=64,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
            output_type=output_type,
        ](1, 2048, ctx, use_index_input=use_index_input)

        test[
            qkv_type,
            576,
            128,
            group=128,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
            output_type=output_type,
        ](1, 2048, ctx, use_index_input=use_index_input)


def test_decoding_partial_head_group[
    batch_size: Int,
    num_partitions: Optional[Int],
    split_k: Bool,
    qkv_type: DType = .bfloat16,
    output_type: DType = qkv_type,
](ctx: DeviceContext, use_index_input: Bool) raises:
    """Covers head counts whose last head group is partial, with a
    full-multiple control at the same shape so a failure is attributable to
    the head count alone.
    """
    comptime for num_heads in [128, 96, 72]:
        test[
            qkv_type,
            576,
            num_heads,
            group=num_heads,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
            output_type=output_type,
        ](1, 1024, ctx, use_index_input=use_index_input)


def test_decoding_k3_head_counts[
    batch_size: Int,
    num_partitions: Optional[Int],
    split_k: Bool,
](ctx: DeviceContext, seq_len: Int, num_keys: Int) raises:
    """Kimi K3's per-device Q-head counts at TP2/TP4/TP8 -- 48, 24 and 12.

    K3 serves bf16 KV, and the only K3 head count measured so far (12, in
    `test_mla_decode_kv_fp8.mojo`) was measured under fp8 KV. These are the
    counts `compute_mla_dispatch_scalars_runtime` has to enumerate for K3's
    latent cache to dispatch at all, so they are measured here first.

    48/24/12 are all a single head group. K3's TP1 count of 96 leaves a partial
    tail head group, which `test_decoding_partial_head_group` covers directly,
    so it is not repeated here. Of the three, only 12 tails the *combine* grid,
    which blocks 8 heads per CTA at `warps_per_head=1` and so leaves 4 -- and
    only a split-K run reaches that kernel at all.

    The control comes FIRST at this exact shape and batch: an assert aborts the
    run, so ordering a supported count ahead of an unusual one is what makes a
    failure attributable to the head count rather than to the shape.
    """
    comptime for num_heads in [128, 48, 24, 12]:
        test[
            DType.bfloat16,
            576,
            num_heads,
            group=num_heads,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
        ](seq_len, num_keys, ctx)


def test_mla_prefill[
    batch_size: Int,
    qkv_type: DType,
    k_rope_type: DType,
    output_type: DType = qkv_type,
](ctx: DeviceContext) raises:
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
        output_type=output_type,
    ](120, 120, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=16,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
        output_type=output_type,
    ](1179, 1179, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
        output_type=output_type,
    ](700, 700, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
        output_type=output_type,
    ](701, 701, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
        output_type=output_type,
    ](12, 12, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
        output_type=output_type,
    ](350, 700, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
        output_type=output_type,
    ](120, 240, ctx)
    # In-kernel 2Q->1Q switch: num_heads=128 makes the 2Q grid large
    # enough that the dispatch heuristic keeps the launch at 2Q (BM=256),
    # so the per-tile switch — not the dispatch-time one — handles short
    # tails. seq_len=300 tiles as [0,256) (full) + [256,300) whose
    # remaining 44 rows (<= 128) route to the 1Q body; the would-be empty
    # 2Q second half is exactly that tile, validating `output_nonempty`.
    # seq_len=428 tiles as [0,256) (full) + [256,428) whose 172 remaining
    # rows (> 128) stay on the 2Q body with both output halves non-empty
    # (WG1 covers 44 valid rows), the complementary `output_nonempty` case.
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
        output_type=output_type,
    ](300, 300, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
        output_type=output_type,
    ](428, 428, ctx)
    # Short query chunk over a long cache history: routes to the 1Q
    # (num_qo=1) kernel via the dispatch heuristic (prompt_len <= 128)
    # while iterating many KV tiles. 800 keys = 7 BN=128 tiles (odd T:
    # exercises the 1Q tail path); 768 keys = 6 tiles (even T: 1Q
    # main-loop only).
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
        output_type=output_type,
    ](100, 800, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
        output_type=output_type,
    ](64, 768, ctx)


def main() raises:
    with DeviceContext() as ctx:
        test_decoding[1, 1, False](ctx, False)
        test_decoding[27, 1, False](ctx, False)
        test_decoding[128, 1, False](ctx, False)
        test_decoding[0, 1, False](ctx, False)

        comptime if _is_sm10x_gpu(ctx.default_device_info):
            test_decoding_partial_head_group[1, 1, False](ctx, False)
            test_decoding_partial_head_group[2, 1, False](ctx, False)
            # Under split-K a tail overrun would corrupt the next split's
            # partials, which the combine kernel then folds into the output.
            test_decoding_partial_head_group[1, 2, True](ctx, False)

        comptime if has_amd_gpu_accelerator():
            test_decoding[1, 4, False](ctx, False)
            test_decoding[27, 2, False](ctx, False)
            # Default (None) — exercise the AMD heuristic.
            test_decoding[1, None, False](ctx, False)
            test_decoding[
                1,
                4,
                False,
                qkv_type=DType.float8_e4m3fn,
                output_type=DType.bfloat16,
            ](ctx, False)

            test_decoding[
                1,
                1,
                False,
                qkv_type=DType.float8_e4m3fn,
                output_type=DType.bfloat16,
            ](ctx, False)
            test_decoding[
                27,
                1,
                False,
                qkv_type=DType.float8_e4m3fn,
                output_type=DType.bfloat16,
            ](ctx, False)

        comptime if _is_sm10x_gpu(ctx.default_device_info):
            test_decoding_k3_head_counts[2, 1, False](ctx, 1, 512)
            test_decoding_k3_head_counts[1, 1, False](ctx, 1, 4096)
            # Split-K is what launches the combine kernel, whose 8-head block
            # these counts tail; an overrun there folds one split's partials
            # into the next head's output.
            test_decoding_k3_head_counts[1, 2, True](ctx, 1, 4096)

        # test mla prefill
        test_mla_prefill[2, DType.bfloat16, DType.bfloat16](ctx)
        test_mla_prefill[4, DType.bfloat16, DType.bfloat16](ctx)
        test_mla_prefill[0, DType.bfloat16, DType.bfloat16](ctx)

        comptime if _is_sm10x_gpu(ctx.default_device_info):
            test_mla_prefill[2, DType.bfloat16, DType.float8_e4m3fn](ctx)
            test_mla_prefill[4, DType.bfloat16, DType.float8_e4m3fn](ctx)
            test_mla_prefill[0, DType.bfloat16, DType.float8_e4m3fn](ctx)
