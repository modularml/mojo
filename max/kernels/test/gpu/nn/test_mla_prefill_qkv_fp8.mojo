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

from max.gpu import *
from max.gpu.host import DeviceContext
from std.random import randn
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    lt_to_tt,
    UNKNOWN_VALUE,
    row_major,
)
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_operand import LayoutTensorMHAOperand
from nn.attention.gpu.mla import flare_mla_prefill
from std.testing import assert_almost_equal

from std.utils.index import Index


def test_prefill[
    qkv_type: DType,
    k_rope_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    kv_depth: Int,
    cache_depth: Int,
    cache_num_heads: Int,
    batch_size: Int = 1,
    use_causal_mask: Bool = True,
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

    var q_bf16_ptr = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    var k_bf16_ptr = ctx.enqueue_create_host_buffer[.bfloat16](k_size)
    var v_bf16_ptr = ctx.enqueue_create_host_buffer[.bfloat16](v_size)
    var cache_bf16_ptr = ctx.enqueue_create_host_buffer[.bfloat16](cache_size)

    randn(q_bf16_ptr.as_span())
    randn(k_bf16_ptr.as_span())
    randn(v_bf16_ptr.as_span())
    randn(cache_bf16_ptr.as_span())

    # Scale inputs to [-1, 1] before the fp8 roundtrip.
    # FP8 e4m3fn has uniform 12.5% relative error across all exponent bands,
    # but values outside [-1, 1] use coarser exponent bands (spacing 0.5-64)
    # which cause larger absolute errors in the Q@K' dot product accumulation.
    # Keeping inputs in [-1, 1] (exponents 6-7, spacing 0.125-0.25) gives the
    # smallest absolute error for a fixed 12.5% relative error budget.
    comptime scale_factor = BFloat16(0.5)
    for i in range(q_size):
        var q_val = (q_bf16_ptr[i] * scale_factor).cast[qkv_type]()
        q_ptr[i] = q_val
        q_bf16_ptr[i] = q_val.cast[.bfloat16]()

    for i in range(k_size):
        var k_val = (k_bf16_ptr[i] * scale_factor).cast[qkv_type]()
        k_ptr[i] = k_val
        k_bf16_ptr[i] = k_val.cast[.bfloat16]()

    for i in range(v_size):
        var v_val = (v_bf16_ptr[i] * scale_factor).cast[qkv_type]()
        v_ptr[i] = v_val
        v_bf16_ptr[i] = v_val.cast[.bfloat16]()

    for i in range(cache_size):
        var cache_val = (cache_bf16_ptr[i] * scale_factor).cast[k_rope_type]()
        cache_ptr[i] = cache_val
        cache_bf16_ptr[i] = cache_val.cast[.bfloat16]()

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
        q_bf16_ptr,
        row_major((batch_size * seq_len, Idx[num_heads], Idx[depth])),
    )
    var k = TileTensor(
        k_bf16_ptr,
        row_major((batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var v = TileTensor(
        v_bf16_ptr,
        row_major((batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var cache = TileTensor(
        cache_bf16_ptr,
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

    # construct device buffers
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

    @__parameter
    @always_inline
    @__copy_capture(
        q_device,
        k_device,
        v_device,
        cache_device,
        input_row_offsets_device,
        cache_row_offsets_device,
        output_device,
    )
    def kernel_launch(ctx: DeviceContext) raises:
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

    kernel_launch(ctx)

    ctx.synchronize()
    ctx.enqueue_copy(output_ptr, output_device_ptr)

    # create reference K and V
    # unlike flare_mla_prefill, K_ref and V_ref each head is of size depth (not kv_depth)
    var k_ref_ptr = ctx.enqueue_create_host_buffer[.bfloat16](
        batch_size * num_keys * num_heads * depth
    )
    var v_ref_ptr = ctx.enqueue_create_host_buffer[.bfloat16](
        batch_size * num_keys * num_heads * depth
    )
    var output_ref_ptr = ctx.enqueue_create_host_buffer[output_type](
        batch_size * seq_len * num_heads * depth
    )

    # create reference K and V
    var k_ref = LayoutTensor[.bfloat16, Layout.row_major[4]()](
        k_ref_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, num_keys, num_heads, depth)
        ),
    )
    var v_ref = LayoutTensor[.bfloat16, Layout.row_major[4]()](
        v_ref_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, num_keys, num_heads, depth)
        ),
    )
    var output_ref = LayoutTensor[output_type, Layout.row_major[4]()](
        output_ref_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
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
                    k_ref[b, s, h, d + kv_depth] = cache[
                        b, s, 0, cache_depth - (depth - kv_depth) + d
                    ][0]
                    v_ref[b, s, h, d + kv_depth] = 0

    # Create bf16 Q on device for the naive reference kernel.
    # This uses the roundtripped values (bf16 -> fp8 -> bf16) so both
    # the kernel under test and the reference see identical values.
    var q_ref_device_ptr = ctx.enqueue_create_buffer[.bfloat16](q_size)
    ctx.enqueue_copy(q_ref_device_ptr, q_bf16_ptr)

    comptime q_layout_4d = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var q_device_rank4 = LayoutTensor[.bfloat16, q_layout_4d](
        q_ref_device_ptr.unsafe_ptr(),
        RuntimeLayout[q_layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    # create device pointers for K_ref and V_ref
    var k_ref_device_ptr = ctx.enqueue_create_buffer[.bfloat16](
        batch_size * num_keys * num_heads * depth
    )
    var v_ref_device_ptr = ctx.enqueue_create_buffer[.bfloat16](
        batch_size * num_keys * num_heads * depth
    )
    var output_ref_device_ptr = ctx.enqueue_create_buffer[output_type](
        batch_size * seq_len * num_heads * depth
    )
    # create device buffers for K_ref and V_ref
    comptime k_layout_4d = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var k_ref_device = LayoutTensor[.bfloat16, k_layout_4d](
        k_ref_device_ptr.unsafe_ptr(),
        RuntimeLayout[k_layout_4d].row_major(
            Index(batch_size, num_keys, num_heads, depth)
        ),
    )
    comptime v_layout_4d = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var v_ref_device = LayoutTensor[.bfloat16, v_layout_4d](
        v_ref_device_ptr.unsafe_ptr(),
        RuntimeLayout[v_layout_4d].row_major(
            Index(batch_size, num_keys, num_heads, depth)
        ),
    )
    comptime output_layout_4d = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var output_ref_device = LayoutTensor[output_type, output_layout_4d](
        output_ref_device_ptr.unsafe_ptr(),
        RuntimeLayout[output_layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
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

    var k_ref_operand = LayoutTensorMHAOperand(lt_to_tt(k_ref_device))
    var v_ref_operand = LayoutTensorMHAOperand(lt_to_tt(v_ref_device))

    # create reference output
    mha_gpu_naive[_is_cache_length_accurate=True](
        q_device_rank4,
        k_ref_operand,
        v_ref_operand,
        CausalMask(),
        output_ref_device,
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
    var output_rank4 = LayoutTensor[output_type, Layout.row_major[4]()](
        output_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, seq_len, num_heads, kv_depth)
        ),
    )

    # compare output with reference
    # FP8 MMA has significantly less precision than BF16 MMA:
    #  - 3 mantissa bits vs 7, so each multiply has ~16x coarser rounding
    #  - softmax probabilities stored as FP8 in TMEM lose precision
    #  - error accumulates across KV tiles (seq_len/BN iterations)
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(kv_depth):
                    var lhs = output_rank4[b, s, h, d]
                    var rhs = output_ref[b, s, h, d]
                    assert_almost_equal(
                        lhs,
                        rhs,
                        atol=5e-2,
                        rtol=2e-2,
                    )

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = cache_device_ptr
    _ = output_device_ptr
    _ = q_ref_device_ptr
    _ = k_ref_device_ptr
    _ = v_ref_device_ptr
    _ = output_ref_device_ptr
    _ = q_bf16_ptr
    _ = k_bf16_ptr
    _ = v_bf16_ptr
    _ = cache_bf16_ptr
    _ = output_ref_ptr
    _ = input_row_offsets_device_ptr
    _ = cache_row_offsets_device_ptr


def test_mla_prefill_qkv_fp8[
    qkv_type: DType,
    k_rope_type: DType,
    output_type: DType,
    batch_size: Int = 1,
](ctx: DeviceContext) raises:
    test_prefill[
        qkv_type,
        k_rope_type,
        output_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](120, 120, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        output_type,
        depth=192,
        num_heads=16,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](1179, 1179, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        output_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](700, 700, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        output_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](701, 701, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        output_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](12, 12, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        output_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](350, 700, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        output_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](120, 240, ctx)


def main() raises:
    with DeviceContext() as ctx:
        test_mla_prefill_qkv_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            0,
        ](ctx)
        test_mla_prefill_qkv_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            1,
        ](ctx)
        test_mla_prefill_qkv_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            2,
        ](ctx)
        test_mla_prefill_qkv_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            4,
        ](ctx)
