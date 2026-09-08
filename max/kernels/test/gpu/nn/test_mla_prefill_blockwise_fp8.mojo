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
    sf_dtype: DType,
    depth: Int,
    num_heads: Int,
    kv_depth: Int,
    cache_depth: Int,
    cache_num_heads: Int,
    batch_size: Int = 1,
    scale_block_size: Int = 64,
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
        "sf_dtype:",
        sf_dtype,
        "depth:",
        depth,
        "kv_depth:",
        kv_depth,
        "cache_depth:",
        cache_depth,
        "cache_num_heads:",
        cache_num_heads,
        "scale_block_size:",
        scale_block_size,
    )

    comptime assert (
        depth % scale_block_size == 0
    ), "depth must be divisible by scale_block_size"
    comptime assert (
        kv_depth % scale_block_size == 0
    ), "kv_depth must be divisible by scale_block_size"
    comptime assert (
        cache_depth % scale_block_size == 0
    ), "cache_depth must be divisible by scale_block_size"

    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))

    var q_size = batch_size * seq_len * num_heads * depth
    var k_size = batch_size * num_keys * num_heads * kv_depth
    var v_size = k_size

    var o_size = batch_size * seq_len * num_heads * kv_depth
    var cache_size = batch_size * num_keys * cache_num_heads * cache_depth
    var cache_sf_size = (
        batch_size
        * num_keys
        * cache_num_heads
        * (cache_depth // scale_block_size)
    )

    var q_ptr = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_ptr = ctx.enqueue_create_host_buffer[qkv_type](k_size)
    var v_ptr = ctx.enqueue_create_host_buffer[qkv_type](v_size)
    var o_ptr = ctx.enqueue_create_host_buffer[qkv_type](o_size)
    var cache_ptr = ctx.enqueue_create_host_buffer[k_rope_type](cache_size)
    var cache_sf_ptr = ctx.enqueue_create_host_buffer[sf_dtype](cache_sf_size)

    # Q, K, V, cache are randomly initialized.
    randn(q_ptr.as_span())
    randn(k_ptr.as_span())
    randn(v_ptr.as_span())
    randn(cache_ptr.as_span())
    randn(cache_sf_ptr.as_span())

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

    # device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_size)
    var cache_device_ptr = ctx.enqueue_create_buffer[k_rope_type](cache_size)
    var cache_sf_device_ptr = ctx.enqueue_create_buffer[sf_dtype](cache_sf_size)
    var output_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
    var input_row_offsets_device_ptr = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var cache_row_offsets_device_ptr = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )

    # ragged inputs
    var q = TileTensor(
        q_ptr.unsafe_ptr(),
        row_major((batch_size * seq_len, Idx[num_heads], Idx[depth])),
    )
    var k = TileTensor(
        k_ptr.unsafe_ptr(),
        row_major((batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var v = TileTensor(
        v_ptr.unsafe_ptr(),
        row_major((batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var cache = TileTensor(
        cache_ptr.unsafe_ptr(),
        row_major(
            (
                batch_size,
                num_keys,
                Idx[cache_num_heads],
                Idx[cache_depth],
            )
        ),
    )
    var cache_sf = TileTensor(
        cache_sf_ptr.unsafe_ptr(),
        row_major(
            (
                batch_size,
                num_keys,
                Idx[cache_num_heads],
                Idx[(cache_depth // scale_block_size)],
            )
        ),
    )
    var output = TileTensor(
        o_ptr.unsafe_ptr(),
        row_major((batch_size * seq_len, Idx[num_heads], Idx[kv_depth])),
    )

    # copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(v_device_ptr, v_ptr)
    ctx.enqueue_copy(cache_device_ptr, cache_ptr)
    ctx.enqueue_copy(cache_sf_device_ptr, cache_sf_ptr)
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
    var cache_sf_device = TileTensor(
        cache_sf_device_ptr,
        row_major(
            (
                batch_size,
                num_keys,
                Idx[cache_num_heads],
                Idx[(cache_depth // scale_block_size)],
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

    flare_mla_prefill[rank=q_device.rank](
        output_device,
        q_device,
        k_device,
        v_device,
        cache_device,
        cache_sf_device,
        CausalMask(),
        input_row_offsets_device,
        cache_row_offsets_device,
        scale,
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(o_ptr, output_device_ptr)

    var k_ref_ptr = ctx.enqueue_create_host_buffer[qkv_type](
        batch_size * num_keys * num_heads * depth
    )
    var v_ref_ptr = ctx.enqueue_create_host_buffer[qkv_type](
        batch_size * num_keys * num_heads * depth
    )
    var output_ref_ptr = ctx.enqueue_create_host_buffer[qkv_type](
        batch_size * seq_len * num_heads * depth
    )

    # create reference K and V
    var k_ref = TileTensor(
        k_ref_ptr.unsafe_ptr(),
        row_major(
            (
                batch_size,
                num_keys,
                Idx[num_heads],
                Idx[depth],
            )
        ),
    )
    var v_ref = TileTensor(
        v_ref_ptr.unsafe_ptr(),
        row_major(
            (
                batch_size,
                num_keys,
                Idx[num_heads],
                Idx[depth],
            )
        ),
    )
    var output_ref = TileTensor(
        output_ref_ptr.unsafe_ptr(),
        row_major(
            (
                batch_size,
                seq_len,
                Idx[num_heads],
                Idx[depth],
            )
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
    var num_scale_blocks_per_head = (depth - kv_depth) // scale_block_size
    var num_scale_blocks_in_cache = cache_depth // scale_block_size

    for b in range(batch_size):
        for s in range(num_keys):
            for h in range(num_heads):
                for d in range(depth - kv_depth):
                    var scale_sf_d = (
                        num_scale_blocks_in_cache
                        - num_scale_blocks_per_head
                        + d // scale_block_size
                    )
                    var cache_sf_val = cache_sf[b, s, 0, scale_sf_d]

                    var cache_val = cache[
                        b, s, 0, cache_depth - (depth - kv_depth) + d
                    ].cast[sf_dtype]()
                    var cache_val_scaled = cache_val * cache_sf_val

                    k_ref[b, s, h, d + kv_depth] = cache_val_scaled.cast[
                        qkv_type
                    ]()
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
    var output_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](
        batch_size * seq_len * num_heads * depth
    )
    # create device buffers for K_ref and V_ref
    var k_ref_device = TileTensor(
        k_ref_device_ptr,
        row_major(
            (
                batch_size,
                num_keys,
                Idx[num_heads],
                Idx[depth],
            )
        ),
    )
    var v_ref_device = TileTensor(
        v_ref_device_ptr,
        row_major(
            (
                batch_size,
                num_keys,
                Idx[num_heads],
                Idx[depth],
            )
        ),
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
        o_ptr.unsafe_ptr(),
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
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(kv_depth):
                    var lhs = output_rank4[b, s, h, d]
                    var rhs = output_ref[b, s, h, d]
                    # if abs((lhs - rhs)) > 2.2e-2:
                    #    print(b, s, h, d, lhs, rhs)
                    # print(b, s, h, d, lhs, rhs)
                    assert_almost_equal(
                        lhs,
                        rhs,
                        atol=2e-2,
                        rtol=2e-2,
                    )

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = cache_device_ptr
    _ = cache_sf_device_ptr
    _ = output_device_ptr
    _ = input_row_offsets_device_ptr
    _ = cache_row_offsets_device_ptr
    _ = k_ref_device_ptr
    _ = v_ref_device_ptr
    _ = output_ref_device_ptr


def test_mla_prefill[
    batch_size: Int,
    qkv_type: DType,
    k_rope_type: DType,
    sf_dtype: DType,
](ctx: DeviceContext) raises:
    test_prefill[
        qkv_type,
        k_rope_type,
        sf_dtype,
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
        sf_dtype,
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
        sf_dtype,
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
        sf_dtype,
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
        sf_dtype,
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
        sf_dtype,
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
        sf_dtype,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](120, 240, ctx)


def main() raises:
    with DeviceContext() as ctx:
        test_mla_prefill[
            0,
            DType.bfloat16,
            DType.float8_e4m3fn,
            DType.float32,
        ](ctx)
        test_mla_prefill[
            2,
            DType.bfloat16,
            DType.float8_e4m3fn,
            DType.float32,
        ](ctx)
        test_mla_prefill[
            4,
            DType.bfloat16,
            DType.float8_e4m3fn,
            DType.float32,
        ](ctx)
