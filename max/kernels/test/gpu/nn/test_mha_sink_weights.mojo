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

"""Correctness test: flash_attention[sink=True] on AMD GPU with gpt-oss shapes.

Verifies flash_attention[sink=True] matches mha_gpu_naive[sink=True].

gpt-oss production config: hidden_size=2880, num_heads=64, num_kv_heads=8,
head_dim=64.
"""

from std.math import isclose
from std.random import rand

from max.gpu import *
from max.gpu.host import DeviceContext
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
    UNKNOWN_VALUE,
)
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import CausalMask
from std.utils.index import Index
from std.utils.numerics import min_or_neg_inf


def test[
    qkv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
](seq_len: Int, num_keys: Int, ctx: DeviceContext,) raises:
    print(
        "test_mha_sink_weights: qkv_type=",
        qkv_type,
        "depth=",
        depth,
        "num_heads=",
        num_heads,
        "group=",
        group,
        "seq_len=",
        seq_len,
        "num_keys=",
        num_keys,
    )

    comptime batch_size = 1
    comptime scale = Float32(0.125)  # 1/sqrt(64)
    comptime kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size
    var mask_size = num_heads * seq_len * num_keys

    # Allocate host memory.
    var q_ptr = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_ptr = ctx.enqueue_create_host_buffer[qkv_type](k_size)
    var v_ptr = ctx.enqueue_create_host_buffer[qkv_type](v_size)
    var mask_ptr = ctx.enqueue_create_host_buffer[mask_type](mask_size)
    var output_ptr = ctx.enqueue_create_host_buffer[qkv_type](o_size)
    var flash_output_ptr = ctx.enqueue_create_host_buffer[qkv_type](o_size)

    for i in range(o_size):
        output_ptr[i] = Scalar[qkv_type](0)

    # Sink weights: one per attention head, initialized to zeros.
    var sink_size = num_heads
    var sink_ptr = ctx.enqueue_create_host_buffer[qkv_type](sink_size)
    sink_ptr.as_span().fill(Scalar[qkv_type](0))

    # Initialize Q, K, V with random data.
    rand(q_ptr.as_span())
    rand(k_ptr.as_span())
    rand(v_ptr.as_span())

    # Initialize causal mask.
    comptime layout_4d = Layout.row_major[4]()
    var mask = LayoutTensor[mask_type, layout_4d](
        mask_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_heads, seq_len, num_keys)
        ),
    )
    for b in range(batch_size):
        for h in range(num_heads):
            for q_idx in range(seq_len):
                for k_idx in range(num_keys):
                    mask.store(
                        Index(b, h, q_idx, k_idx),
                        0 if q_idx + num_keys - seq_len
                        >= k_idx else min_or_neg_inf[mask_type](),
                    )

    # Device pointers.
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_size)
    var mask_device_ptr = ctx.enqueue_create_buffer[mask_type](mask_size)
    var output_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
    var sink_device_ptr = ctx.enqueue_create_buffer[qkv_type](sink_size)

    # Copy to device.
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(v_device_ptr, v_ptr)
    ctx.enqueue_copy(mask_device_ptr, mask_ptr)
    ctx.enqueue_copy(sink_device_ptr, sink_ptr)

    # Construct device tensors.
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
    var mask4d = TileTensor(
        mask_device_ptr,
        row_major((batch_size, num_heads, seq_len, num_keys)),
    )
    var output_device = TileTensor(
        output_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )

    comptime sink_layout = Layout.row_major(UNKNOWN_VALUE)
    var sink_device = LayoutTensor[qkv_type, sink_layout, ImmutAnyOrigin](
        sink_device_ptr,
        RuntimeLayout[sink_layout].row_major(Index(num_heads)),
    )

    # Run flash_attention with sink=True.
    flash_attention[sink=True](
        output_device,
        q_device,
        k_device,
        v_device,
        CausalMask(),
        scale,
        ctx,
        sink_weights=sink_device,
    )

    ctx.synchronize()
    ctx.enqueue_copy(flash_output_ptr, output_device_ptr)

    # Run naive reference with sink=True.
    var output_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
    ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

    var output_device_ref = TileTensor(
        output_ref_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )

    mha_gpu_naive[sink=True](
        q_device,
        k_device,
        v_device,
        mask4d,
        output_device_ref,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        group,
        ctx,
        sink_device,
    )

    ctx.synchronize()
    ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
    ctx.synchronize()
    _ = output_ref_device_ptr

    # Compare flash_attention[sink=True] vs mha_gpu_naive[sink=True].
    var rtol = 3e-2
    var mismatches = 0
    var count = 0
    for h in range(num_heads):
        for s in range(seq_len):
            for d in range(depth):
                var expect = output_ptr[d + depth * (h + s * num_heads)].cast[
                    DType.float64
                ]()
                var actual = flash_output_ptr[
                    d + depth * (h + s * num_heads)
                ].cast[.float64]()
                count += 1
                if not isclose(actual, expect, atol=1e-5, rtol=rtol):
                    mismatches += 1

    print("  mismatches=", mismatches, "/", count)
    assert mismatches == 0, (
        "flash_attention[sink=True] does not match mha_gpu_naive[sink=True]."
        " mismatches="
        + String(mismatches)
        + "/"
        + String(count)
    )

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = mask_device_ptr
    _ = output_device_ptr
    _ = sink_device_ptr


def main() raises:
    with DeviceContext() as ctx:
        # gpt-oss shapes: num_heads=64, num_kv_heads=8, head_dim=64
        # group = num_heads / num_kv_heads = 8

        # Prefill seq_len=11 (the failing correctness shape from CI)
        test[
            DType.bfloat16,
            DType.bfloat16,
            depth=64,
            num_heads=64,
            group=8,
        ](11, 11, ctx)

        # Prefill seq_len=128 (second correctness shape)
        test[
            DType.bfloat16,
            DType.bfloat16,
            depth=64,
            num_heads=64,
            group=8,
        ](128, 128, ctx)
