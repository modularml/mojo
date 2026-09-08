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
from layout.tile_tensor import TileTensor
from layout.tile_layout import row_major
from layout.coord import Idx, Coord
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_operand import LayoutTensorMHAOperand
from nn.attention.gpu.mla import flare_mla_prefill
from std.testing import assert_almost_equal


def test_prefill[
    qkv_type: DType,
    rope_type: DType,
    scale_type: DType,
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
        "rope_type:",
        rope_type,
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

    var q_size = batch_size * seq_len * num_heads * depth
    var q_nope_size = batch_size * seq_len * num_heads * kv_depth
    var q_rope_size = batch_size * seq_len * num_heads * (depth - kv_depth)
    var q_scale_size = batch_size * seq_len
    var k_size = batch_size * num_keys * num_heads * kv_depth
    var k_scale_size = batch_size * num_keys
    var v_size = k_size
    var o_size = batch_size * seq_len * num_heads * kv_depth
    var cache_size = batch_size * num_keys * cache_num_heads * cache_depth

    var q_scale_ptr = ctx.enqueue_create_host_buffer[scale_type](q_scale_size)
    var q_nope_ptr = ctx.enqueue_create_host_buffer[qkv_type](q_nope_size)
    var q_rope_ptr = ctx.enqueue_create_host_buffer[rope_type](q_rope_size)

    var k_ptr = ctx.enqueue_create_host_buffer[qkv_type](k_size)
    var k_scale_ptr = ctx.enqueue_create_host_buffer[scale_type](k_scale_size)
    var v_ptr = ctx.enqueue_create_host_buffer[qkv_type](v_size)
    var cache_ptr = ctx.enqueue_create_host_buffer[rope_type](cache_size)
    var output_ptr = ctx.enqueue_create_host_buffer[output_type](o_size)

    var q_bf16_ptr = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    var k_bf16_ptr = ctx.enqueue_create_host_buffer[.bfloat16](k_size)
    var v_bf16_ptr = ctx.enqueue_create_host_buffer[.bfloat16](v_size)
    var cache_bf16_ptr = ctx.enqueue_create_host_buffer[.bfloat16](cache_size)

    randn(q_bf16_ptr.as_span())
    randn(k_bf16_ptr.as_span())
    randn(v_bf16_ptr.as_span())
    randn(cache_bf16_ptr.as_span())

    # scale down the value to make it easier to verify
    var scale_factor = BFloat16(0.125)
    var scale = Float32(0.5)
    for i in range(q_size):
        q_bf16_ptr[i] *= scale_factor
    for i in range(k_size):
        k_bf16_ptr[i] *= scale_factor
    for i in range(v_size):
        v_bf16_ptr[i] *= scale_factor
    for i in range(cache_size):
        cache_bf16_ptr[i] *= scale_factor

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

    # =============================== Q inputs =============================== #
    # Q has shape [batch_size * seq_len, num_heads, depth]
    # Q_bf16 is randomly generated, Q is in e4m3 quantized from Q_bf16
    # Q_scale is per token scaled, meaning we shared same scale per [128, 128] block
    # Q_scale has shape [batch_size * seq_len, 1]
    var q_bf16 = TileTensor(
        q_bf16_ptr.unsafe_ptr(),
        row_major(Coord(batch_size * seq_len, Idx[num_heads], Idx[depth])),
    )
    var q_nope = TileTensor(
        q_nope_ptr.unsafe_ptr(),
        row_major(Coord(batch_size * seq_len, Idx[num_heads], Idx[kv_depth])),
    )
    var q_rope = TileTensor(
        q_rope_ptr.unsafe_ptr(),
        row_major(
            Coord(
                batch_size * seq_len,
                Idx[num_heads],
                Idx[(depth - kv_depth)],
            )
        ),
    )
    var q_scale = TileTensor(
        q_scale_ptr.unsafe_ptr(),
        row_major(Coord(batch_size * seq_len, Idx[1])),
    )

    # =============================== K inputs =============================== #
    # K has shape [batch_size * num_keys, num_heads, kv_depth]
    # K_bf16 is randomly generated, K is in e4m3 quantized from K_bf16
    # K_scale is per token scaled, meaning we shared same scale per [128, 128] block
    # K_scale has shape [batch_size * num_keys, 1]
    var k_bf16 = TileTensor(
        k_bf16_ptr.unsafe_ptr(),
        row_major(Coord(batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var k = TileTensor(
        k_ptr.unsafe_ptr(),
        row_major(Coord(batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var k_scale = TileTensor(
        k_scale_ptr.unsafe_ptr(),
        row_major(Coord(batch_size * num_keys, Idx[1])),
    )

    # =============================== V inputs =============================== #
    # V has shape [batch_size * num_keys, num_heads, kv_depth]
    # V_bf16 is randomly generated, V is in e4m3 quantized from V_bf16
    # V_scale is per head scaled
    # V_scale has shape [num_heads, batch_size * num_keys, 1]
    var v_bf16 = TileTensor(
        v_bf16_ptr.unsafe_ptr(),
        row_major(Coord(batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var v = TileTensor(
        v_ptr.unsafe_ptr(),
        row_major(Coord(batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )

    # =============================== Cache inputs =============================== #
    # Cache has shape [batch_size * num_keys, cache_num_heads, cache_depth]
    # Cache_bf16 is randomly generated, Cache is in e4m3 quantized from Cache_bf16
    # Cache_scale is per token scaled, meaning we shared same scale per [128, 128] block
    # Cache does not have a scale tensor, it will be scaled by k_nope scale.
    var cache_bf16 = TileTensor(
        cache_bf16_ptr.unsafe_ptr(),
        row_major(
            Coord(
                batch_size,
                num_keys,
                Idx[cache_num_heads],
                Idx[cache_depth],
            )
        ),
    )
    var cache = TileTensor(
        cache_ptr.unsafe_ptr(),
        row_major(
            Coord(
                batch_size,
                num_keys,
                Idx[cache_num_heads],
                Idx[cache_depth],
            )
        ),
    )

    var output = TileTensor(
        output_ptr.unsafe_ptr(),
        row_major(Coord(batch_size * seq_len, Idx[num_heads], Idx[kv_depth])),
    )

    # compute q_scale and quantize q, q is per token scaled
    for i in range(batch_size):
        for j in range(seq_len):
            var q_max = Float32(-1e10)
            for h in range(num_heads):
                for d in range(depth):
                    var q_abs = abs(q_bf16[Coord(i * seq_len + j, h, d)]).cast[
                        DType.float32
                    ]()
                    q_max = max(q_max, q_abs)
            q_scale[Coord(i * seq_len + j, Idx[0])] = max(
                q_max / Float32(448), Float32(1e-10)
            ).cast[scale_type]()

    # separate q into q_nope and q_rope
    for i in range(batch_size):
        for j in range(seq_len):
            for h in range(num_heads):
                for d in range(depth):
                    var q_bf16_value = q_bf16[Coord(i * seq_len + j, h, d)]
                    var q_scale_value = q_scale[
                        Coord(i * seq_len + j, Idx[0])
                    ].cast[.bfloat16]()
                    if d < kv_depth:
                        q_nope[Coord(i * seq_len + j, h, d)] = (
                            q_bf16_value / q_scale_value
                        ).cast[qkv_type]()

                    else:
                        q_rope[Coord(i * seq_len + j, h, d - kv_depth)] = (
                            q_bf16_value / q_scale_value
                        ).cast[rope_type]()

    # compute k_scale and quantize k, k is per token scaled
    for i in range(batch_size):
        for j in range(num_keys):
            var k_max = Float32(-1e10)
            for h in range(num_heads):
                for d in range(kv_depth):
                    var k_abs = abs(k_bf16[Coord(i * num_keys + j, h, d)]).cast[
                        DType.float32
                    ]()
                    k_max = max(k_max, k_abs)
            k_scale[Coord(i * num_keys + j, Idx[0])] = max(
                k_max / Float32(448), Float32(1e-10)
            ).cast[scale_type]()

    for i in range(batch_size):
        for j in range(num_keys):
            for h in range(num_heads):
                for d in range(kv_depth):
                    var k_bf16_value = k_bf16[Coord(i * num_keys + j, h, d)]
                    var k_scale_value = k_scale[
                        Coord(i * num_keys + j, Idx[0])
                    ].cast[.bfloat16]()
                    k[Coord(i * num_keys + j, h, d)] = (
                        k_bf16_value / k_scale_value
                    ).cast[qkv_type]()

    for i in range(batch_size):
        for j in range(num_keys):
            for h in range(num_heads):
                for d in range(kv_depth):
                    var v_bf16_value = v_bf16[Coord(i * num_keys + j, h, d)]
                    v[Coord(i * num_keys + j, h, d)] = (v_bf16_value).cast[
                        qkv_type
                    ]()

    # cache stays in bf16, but it gets divided by k_scale to convert into fp8 domain
    for i in range(batch_size):
        for j in range(num_keys):
            for h in range(cache_num_heads):
                for d in range(cache_depth):
                    var cache_bf16_value = cache_bf16[Coord(i, j, h, d)]
                    var k_scale_value = k_scale[
                        Coord(i * num_keys + j, Idx[0])
                    ].cast[.bfloat16]()
                    cache[Coord(i, j, h, d)] = (
                        cache_bf16_value / k_scale_value
                    ).cast[rope_type]()

    var q_nope_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_nope_size)
    var q_rope_device_ptr = ctx.enqueue_create_buffer[rope_type](q_rope_size)
    var q_scale_device_ptr = ctx.enqueue_create_buffer[scale_type](q_scale_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var k_scale_device_ptr = ctx.enqueue_create_buffer[scale_type](k_scale_size)
    var v_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_size)
    var cache_device_ptr = ctx.enqueue_create_buffer[rope_type](cache_size)
    var output_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)

    var input_row_offsets_device_ptr = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var cache_row_offsets_device_ptr = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )

    var q_nope_device = TileTensor(
        q_nope_device_ptr,
        row_major(Coord(batch_size * seq_len, Idx[num_heads], Idx[kv_depth])),
    )

    var q_rope_device = TileTensor(
        q_rope_device_ptr,
        row_major(
            Coord(
                batch_size * seq_len,
                Idx[num_heads],
                Idx[(depth - kv_depth)],
            )
        ),
    )
    var q_scale_device = TileTensor(
        q_scale_device_ptr,
        row_major(Coord(batch_size * seq_len, Idx[1])),
    )
    var k_device = TileTensor(
        k_device_ptr,
        row_major(Coord(batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var k_scale_device = TileTensor(
        k_scale_device_ptr,
        row_major(Coord(batch_size * num_keys, Idx[1])),
    )
    var v_device = TileTensor(
        v_device_ptr,
        row_major(Coord(batch_size * num_keys, Idx[num_heads], Idx[kv_depth])),
    )
    var cache_device = TileTensor(
        cache_device_ptr,
        row_major(
            Coord(
                batch_size,
                num_keys,
                Idx[cache_num_heads],
                Idx[cache_depth],
            )
        ),
    )
    var output_device = TileTensor(
        output_device_ptr,
        row_major(Coord(batch_size * seq_len, Idx[num_heads], Idx[kv_depth])),
    )

    var input_row_offsets_device = TileTensor(
        input_row_offsets_device_ptr,
        row_major(Coord(batch_size + 1)),
    )
    var cache_row_offsets_device = TileTensor(
        cache_row_offsets_device_ptr,
        row_major(Coord(batch_size + 1)),
    )

    ctx.enqueue_copy(q_nope_device_ptr, q_nope_ptr)
    ctx.enqueue_copy(q_rope_device_ptr, q_rope_ptr)
    ctx.enqueue_copy(q_scale_device_ptr, q_scale_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(k_scale_device_ptr, k_scale_ptr)
    ctx.enqueue_copy(v_device_ptr, v_ptr)
    ctx.enqueue_copy(cache_device_ptr, cache_ptr)
    ctx.enqueue_copy(input_row_offsets_device_ptr, input_row_offsets)
    ctx.enqueue_copy(cache_row_offsets_device_ptr, cache_row_offsets)

    flare_mla_prefill[rank=3](
        output_device,
        q_nope_device,
        q_rope_device,
        q_scale_device,
        k_device,
        k_scale_device,
        v_device,
        cache_device,
        CausalMask(),
        input_row_offsets_device,
        cache_row_offsets_device,
        scale,
        ctx,
        q_max_seq_len=seq_len,
    )

    ctx.synchronize()
    ctx.enqueue_copy(output_ptr, output_device_ptr)

    var dangling_valid_length = TileTensor(
        MutPointer[UInt32, MutAnyOrigin].unsafe_dangling(),
        row_major(Coord(Idx[0])),
    )

    var k_ref_host_ptr = ctx.enqueue_create_host_buffer[.bfloat16](
        batch_size * num_keys * num_heads * depth
    )
    var v_ref_host_ptr = ctx.enqueue_create_host_buffer[.bfloat16](
        batch_size * num_keys * num_heads * depth
    )
    var output_ref_host_ptr = ctx.enqueue_create_host_buffer[output_type](
        batch_size * seq_len * num_heads * depth
    )

    var k_ref_host = TileTensor(
        k_ref_host_ptr.unsafe_ptr(),
        row_major(
            Coord(
                batch_size,
                num_keys,
                Idx[num_heads],
                Idx[depth],
            )
        ),
    )
    var v_ref_host = TileTensor(
        v_ref_host_ptr.unsafe_ptr(),
        row_major(
            Coord(
                batch_size,
                num_keys,
                Idx[num_heads],
                Idx[depth],
            )
        ),
    )

    # Build a faithful reference using the SAME quantized data the kernel
    # receives, dequantized back to BF16. This tests the kernel's per-token
    # scaling logic rather than FP8 approximation quality.
    var q_ref_host_ptr = ctx.enqueue_create_host_buffer[.bfloat16](
        batch_size * seq_len * num_heads * depth
    )
    var q_ref_host = TileTensor(
        q_ref_host_ptr.unsafe_ptr(),
        row_major(
            Coord(
                batch_size,
                seq_len,
                Idx[num_heads],
                Idx[depth],
            )
        ),
    )

    # Q_ref = [q_nope_fp8 * q_scale | q_rope_bf16 * q_scale]
    for b in range(batch_size):
        for s in range(seq_len):
            var qs = q_scale[Coord(b * seq_len + s, Idx[0])].cast[
                DType.bfloat16
            ]()
            for h in range(num_heads):
                for d in range(kv_depth):
                    q_ref_host[Coord(b, s, h, d)] = (
                        q_nope[Coord(b * seq_len + s, h, d)].cast[
                            DType.bfloat16
                        ]()
                        * qs
                    )
                for d in range(depth - kv_depth):
                    q_ref_host[Coord(b, s, h, d + kv_depth)] = (
                        q_rope[Coord(b * seq_len + s, h, d)].cast[
                            DType.bfloat16
                        ]()
                        * qs
                    )

    # K_ref_nope = k_fp8 * k_scale (dequantized)
    # K_ref_rope = cache_bf16 (original, since cache/k_scale * k_scale = cache)
    # V_ref = v_fp8 dequantized to BF16 (no scaling)
    for b in range(batch_size):
        for s in range(num_keys):
            var ks = k_scale[Coord(b * num_keys + s, Idx[0])].cast[
                DType.bfloat16
            ]()
            for h in range(num_heads):
                for d in range(kv_depth):
                    k_ref_host[Coord(b, s, h, d)] = (
                        k[Coord(b * num_keys + s, h, d)].cast[.bfloat16]() * ks
                    )
                    v_ref_host[Coord(b, s, h, d)] = v[
                        Coord(b * num_keys + s, h, d)
                    ].cast[.bfloat16]()

    for b in range(batch_size):
        for s in range(num_keys):
            for h in range(num_heads):
                for d in range(depth - kv_depth):
                    k_ref_host[Coord(b, s, h, d + kv_depth)] = cache_bf16[
                        Coord(
                            b,
                            s,
                            Idx[0],
                            cache_depth - (depth - kv_depth) + d,
                        )
                    ]
                    v_ref_host[Coord(b, s, h, d + kv_depth)] = 0

    var q_ref_device_ptr = ctx.enqueue_create_buffer[.bfloat16](q_size)
    var k_ref_device_ptr = ctx.enqueue_create_buffer[.bfloat16](
        batch_size * num_keys * num_heads * depth
    )
    var v_ref_device_ptr = ctx.enqueue_create_buffer[.bfloat16](
        batch_size * num_keys * num_heads * depth
    )
    var output_ref_device_ptr = ctx.enqueue_create_buffer[output_type](
        batch_size * seq_len * num_heads * depth
    )

    ctx.enqueue_copy(q_ref_device_ptr, q_ref_host_ptr)
    ctx.enqueue_copy(k_ref_device_ptr, k_ref_host_ptr)
    ctx.enqueue_copy(v_ref_device_ptr, v_ref_host_ptr)

    var q_ref_4d_device = TileTensor(
        q_ref_device_ptr,
        row_major(Coord(batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )
    var k_ref_device = TileTensor(
        k_ref_device_ptr,
        row_major(
            Coord(
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
            Coord(
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
            Coord(
                batch_size,
                seq_len,
                Idx[num_heads],
                Idx[depth],
            )
        ),
    )

    var output_ref_host = TileTensor(
        output_ref_host_ptr.unsafe_ptr(),
        row_major(Coord(batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )

    var k_ref_operand = LayoutTensorMHAOperand(
        k_ref_device.as_immut().as_unsafe_any_origin()
    )
    var v_ref_operand = LayoutTensorMHAOperand(
        v_ref_device.as_immut().as_unsafe_any_origin()
    )

    # create reference output
    mha_gpu_naive[_is_cache_length_accurate=True](
        q_ref_4d_device.to_layout_tensor(),
        k_ref_operand,
        v_ref_operand,
        CausalMask(),
        output_ref_device.to_layout_tensor(),
        dangling_valid_length.to_layout_tensor(),
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        1,
        ctx,
    )

    ctx.enqueue_copy(output_ref_host_ptr, output_ref_device_ptr)
    ctx.synchronize()

    for i in range(batch_size):
        for j in range(seq_len):
            for h in range(num_heads):
                for d in range(kv_depth):
                    var lhs = output[Coord(i * seq_len + j, h, d)]
                    var rhs = output_ref_host[Coord(i, j, h, d)]
                    if abs(lhs - rhs) > 5e-2:
                        print("[", i, j, h, d, "]", lhs, rhs, lhs / rhs)
                    assert_almost_equal(
                        lhs,
                        rhs,
                        atol=5e-2,
                        rtol=5e-2,
                    )

    _ = q_nope_ptr
    _ = q_rope_ptr
    _ = q_scale_ptr
    _ = q_bf16_ptr
    _ = k_ptr
    _ = k_scale_ptr
    _ = k_bf16_ptr
    _ = v_ptr
    _ = v_bf16_ptr
    _ = cache_ptr
    _ = cache_bf16_ptr
    _ = output_ptr
    _ = input_row_offsets
    _ = cache_row_offsets
    _ = q_nope_device_ptr
    _ = q_rope_device_ptr
    _ = q_scale_device_ptr
    _ = k_device_ptr
    _ = k_scale_device_ptr
    _ = v_device_ptr
    _ = cache_device_ptr
    _ = output_device_ptr
    _ = q_ref_device_ptr
    _ = k_ref_device_ptr
    _ = v_ref_device_ptr
    _ = output_ref_device_ptr
    _ = input_row_offsets_device_ptr
    _ = cache_row_offsets_device_ptr


def test_mla_prefill_qkv_fp8[
    qkv_type: DType,
    rope_type: DType,
    scale_type: DType,
    output_type: DType,
    batch_size: Int = 1,
](ctx: DeviceContext) raises:
    test_prefill[
        qkv_type,
        rope_type,
        scale_type,
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
        rope_type,
        scale_type,
        output_type,
        depth=192,
        num_heads=16,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](1180, 1180, ctx)
    test_prefill[
        qkv_type,
        rope_type,
        scale_type,
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
        rope_type,
        scale_type,
        output_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](720, 720, ctx)
    test_prefill[
        qkv_type,
        rope_type,
        scale_type,
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
        rope_type,
        scale_type,
        output_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](400, 800, ctx)
    test_prefill[
        qkv_type,
        rope_type,
        scale_type,
        output_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](120, 240, ctx)
    # In-kernel 2Q->1Q switch: num_heads=128 keeps the dispatch at 2Q
    # (BM=256), so the per-tile switch handles short tails. seq_len=300
    # tiles as [0,256) (full) + [256,300) whose 44 remaining rows (<= 128)
    # route to the 1Q body — exercising the per-token 1Q FUSED load/mma and
    # the k_scale stride-2 path from a 2Q launch — and validating that the
    # would-be empty 2Q second half (folded by `output_nonempty`) is
    # exactly that tile. seq_len=428 tiles as [0,256) + [256,428) whose 172
    # remaining rows (> 128) stay on the 2Q body with both output halves
    # non-empty (WG1 covers 44 valid rows), the complementary case.
    test_prefill[
        qkv_type,
        rope_type,
        scale_type,
        output_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](300, 300, ctx)
    test_prefill[
        qkv_type,
        rope_type,
        scale_type,
        output_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](428, 428, ctx)
    # Short query chunk over a long cache, 16 heads: routes to the 1Q
    # (num_qo=1) kernel via the dispatch heuristic (prompt_len <= 128)
    # while iterating many KV tiles. 800 keys -> odd T (1Q tail path);
    # 768 keys -> even T (1Q main-loop only). Exercises the 1Q load /
    # mma rewrite AND the 1Q k_scale stride-2 fix in fa4_softmax.
    test_prefill[
        qkv_type,
        rope_type,
        scale_type,
        output_type,
        depth=192,
        num_heads=16,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](100, 800, ctx)
    test_prefill[
        qkv_type,
        rope_type,
        scale_type,
        output_type,
        depth=192,
        num_heads=16,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](64, 768, ctx)


def main() raises:
    with DeviceContext() as ctx:
        test_mla_prefill_qkv_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.float32,
            DType.bfloat16,
            1,
        ](ctx)
        test_mla_prefill_qkv_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.float32,
            DType.bfloat16,
            2,
        ](ctx)
        test_mla_prefill_qkv_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.float32,
            DType.bfloat16,
            4,
        ](ctx)
        test_mla_prefill_qkv_fp8[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.float32,
            DType.bfloat16,
            0,
        ](ctx)
