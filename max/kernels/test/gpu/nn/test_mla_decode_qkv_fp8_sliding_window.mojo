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

"""Test for native FP8 MLA decode kernel with SlidingWindowCausalMask.

Verifies that the SM100 native FP8 MLA decode kernel
(MLA_SM100_Decode_QKV_FP8) produces correct output under sliding window
causal masking.  Reference is computed by mha_gpu_naive (BF16 dequantized).

Sub-sweeps:
- small_cache:        cache edge cases at BN/page boundaries
- medium_cache:       page-size boundaries
- large_cache:        multi-split, deep pipeline
- fold_variations:    seq_len in {1,2,4,8,16} on representative shape
- non_fold_seq16:     seq_len=16 forced through non-fold path
- w_edges:            W edge cases (W=1, W=cache+seq, W=page_size)
- page_sizes:         explicit coverage of split_page_size 64 vs 128
- production_configs: DeepSeek V3 (h=128) and Kimi K2.5 (h=64)
- batch_variations:   batch_size in {1,4,16,32}

Dispatch logic relevant to coverage (from mla_decode_dispatch.mojo):
- split_page_size=64 requires effective_split_len <= 512 AND batch_size >= 32
- fold_active = (q_max_seq_len > 1) and (num_heads * q_max_seq_len <= 64)
- Sliding-window caps effective_split_len at window_size + q_max_seq_len
"""

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
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.mha_mask import SlidingWindowCausalMask
from nn.attention.mha_operand import LayoutTensorMHAOperand
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from nn.attention.mha_utils import MHAConfig
from std.testing import assert_almost_equal
from max.gpu.host.info import _is_sm10x_gpu
from std.utils.index import Index


@always_inline
def host_cast_fp8_to_bf16[
    fp8_t: DType,
    bf16_t: DType,
](
    src: Pointer[Scalar[fp8_t], _],
    dst: MutPointer[Scalar[bf16_t], _],
    size: Int,
):
    """Cast FP8 data to BF16 element-by-element on the host."""
    for i in range(size):
        dst[i] = src[i].cast[bf16_t]()


@always_inline
def host_quantize_bf16_to_fp8[
    bf16_t: DType,
    fp8_t: DType,
](
    src: Pointer[Scalar[bf16_t], _],
    dst: MutPointer[Scalar[fp8_t], _],
    size: Int,
):
    """Quantize BF16 data to FP8 element-by-element on the host."""
    for i in range(size):
        dst[i] = src[i].cast[fp8_t]()


def test[
    window_size: Int,
    q_type: DType,
    kv_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    batch_size: Int = 1,
](seq_len: Int, num_keys: Int, ctx: DeviceContext,) raises:
    print(
        "test_mla_decode_qkv_fp8_sliding_window",
        "W:",
        window_size,
        "batch_size:",
        batch_size,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "num_heads:",
        num_heads,
    )

    comptime scale = Float32(0.125)
    comptime kv_num_heads = num_heads // group
    comptime v_depth = depth - 64

    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var o_size = batch_size * num_heads * seq_len * v_depth

    var q_bf16_ptr = alloc[Scalar[output_type]](q_size)
    var q_fp8_ptr = alloc[Scalar[q_type]](q_size)
    var q_bf16_dequant_ptr = alloc[Scalar[output_type]](q_size)
    var k_fp8_ptr = alloc[Scalar[kv_type]](k_size)
    var k_bf16_ptr = alloc[Scalar[output_type]](k_size)
    var output_ptr = alloc[Scalar[output_type]](o_size)
    var flash_output_ptr = alloc[Scalar[output_type]](o_size)

    randn[output_type](q_bf16_ptr, q_size)
    host_quantize_bf16_to_fp8[bf16_t=output_type, fp8_t=q_type](
        q_bf16_ptr, q_fp8_ptr, q_size
    )
    host_cast_fp8_to_bf16[fp8_t=q_type, bf16_t=output_type](
        q_fp8_ptr, q_bf16_dequant_ptr, q_size
    )

    randn[kv_type](k_fp8_ptr, k_size)
    host_cast_fp8_to_bf16[fp8_t=kv_type, bf16_t=output_type](
        k_fp8_ptr, k_bf16_ptr, k_size
    )

    var q_fp8_device_ptr = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_fp8_device_ptr, q_fp8_ptr)

    var k_fp8_device_ptr = ctx.enqueue_create_buffer[kv_type](k_size)
    ctx.enqueue_copy(k_fp8_device_ptr, k_fp8_ptr)

    var q_bf16_dequant_device_ptr = ctx.enqueue_create_buffer[output_type](
        q_size
    )
    ctx.enqueue_copy(q_bf16_dequant_device_ptr, q_bf16_dequant_ptr)

    var k_bf16_device_ptr = ctx.enqueue_create_buffer[output_type](k_size)
    ctx.enqueue_copy(k_bf16_device_ptr, k_bf16_ptr)

    var output_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)
    var output_ref_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)

    var q_fp8_tt = TileTensor(
        q_fp8_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )
    var k_fp8_tt = TileTensor(
        k_fp8_device_ptr,
        row_major(
            (
                batch_size,
                num_keys,
                Idx[kv_num_heads],
                Idx[depth],
            )
        ),
    )
    var out_tt = TileTensor(
        output_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[v_depth])),
    )

    comptime k_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, kv_num_heads, depth)
    )

    var k_bf16_device = LayoutTensor[output_type, k_layout](
        k_bf16_device_ptr.unsafe_ptr(),
        RuntimeLayout[k_layout].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
    )

    comptime q_fp8_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var q_bf16_dequant_device = LayoutTensor[output_type, q_fp8_layout](
        q_bf16_dequant_device_ptr.unsafe_ptr(),
        RuntimeLayout[q_fp8_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    comptime output_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, v_depth)
    )
    var output_ref_device = LayoutTensor[output_type, output_layout](
        output_ref_device_ptr.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            Index(batch_size, seq_len, num_heads, v_depth)
        ),
    )

    var null_valid_length = LayoutTensor[
        .uint32,
        Layout.row_major(UNKNOWN_VALUE),
        MutAnyOrigin,
    ](
        None,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(Index(0)),
    )

    print("  Launching native FP8 kernel...")

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        _is_cache_length_accurate=True,
        is_fp8_kv=True,
    ](
        batch_size,
        num_keys,
        seq_len,
        ctx,
    )
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    @__parameter
    @always_inline
    @__copy_capture(
        q_fp8_tt,
        k_fp8_tt,
        out_tt,
        scalar_args_buf_tt,
    )
    def kernel_launch(ctx: DeviceContext) raises:
        comptime config = MHAConfig[q_type](num_heads, depth)
        flare_mla_decoding[config=config](
            out_tt.as_unsafe_any_origin(),
            q_fp8_tt,
            k_fp8_tt,
            SlidingWindowCausalMask[window_size](),
            scale,
            ctx,
            scalar_args_buf_tt,
        )

    kernel_launch(ctx)
    ctx.synchronize()
    print("  Kernel completed.")

    ctx.enqueue_copy(flash_output_ptr, output_device_ptr)

    print("  Computing GPU naive reference...")

    comptime ref_output_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var ref_full_o_size = batch_size * num_heads * seq_len * depth
    var output_ref_full_device_ptr = ctx.enqueue_create_buffer[output_type](
        ref_full_o_size
    )
    var output_ref_full_device = LayoutTensor[output_type, ref_output_layout](
        output_ref_full_device_ptr.unsafe_ptr(),
        RuntimeLayout[ref_output_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    var k_bf16_operand = LayoutTensorMHAOperand(
        TileTensor(
            k_bf16_device.ptr,
            row_major(
                Int(batch_size),
                Int(num_keys),
                Idx[kv_num_heads],
                Idx[depth],
            ),
        )
    )

    mha_gpu_naive[_is_cache_length_accurate=True](
        q_bf16_dequant_device,
        k_bf16_operand,
        k_bf16_operand,
        SlidingWindowCausalMask[window_size](),
        output_ref_full_device,
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
    print("  Reference completed.")

    var ref_full_output_ptr = alloc[Scalar[output_type]](ref_full_o_size)
    ctx.enqueue_copy(ref_full_output_ptr, output_ref_full_device_ptr)
    ctx.synchronize()

    if o_size == 0:
        return

    var rtol = 5e-2
    var atol = 3e-1
    var num_mismatches = 0
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(v_depth):
                    var expect = ref_full_output_ptr.load(
                        d
                        + depth * (h + s * num_heads)
                        + b * depth * num_heads * seq_len
                    ).cast[.float64]()
                    var actual = flash_output_ptr.load(
                        d
                        + v_depth * (h + s * num_heads)
                        + b * v_depth * num_heads * seq_len
                    ).cast[.float64]()
                    if abs((actual - expect)) > 1e-1:
                        if num_mismatches < 10:
                            print(b, h, s, d, actual, expect)
                        num_mismatches += 1
                    assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

    if num_mismatches > 0:
        print(
            "  WARNING:",
            num_mismatches,
            "mismatches > 1e-1 (but all passed atol/rtol)",
        )
    print("  PASSED")

    _ = mla_args
    _ = q_fp8_device_ptr
    _ = k_fp8_device_ptr
    _ = q_bf16_dequant_device_ptr
    _ = k_bf16_device_ptr
    _ = output_device_ptr
    _ = output_ref_device_ptr
    _ = output_ref_full_device_ptr

    q_bf16_ptr.free()
    q_fp8_ptr.free()
    q_bf16_dequant_ptr.free()
    k_fp8_ptr.free()
    k_bf16_ptr.free()
    output_ptr.free()
    flash_output_ptr.free()
    ref_full_output_ptr.free()


def test_decoding[
    batch_size: Int,
    num_heads: Int,
    window_size: Int,
](ctx: DeviceContext, seq_len: Int, num_keys: Int) raises:
    test[
        window_size,
        DType.float8_e4m3fn,
        DType.float8_e4m3fn,
        DType.bfloat16,
        576,
        num_heads,
        group=num_heads,
        batch_size=batch_size,
    ](seq_len, num_keys, ctx)


# ============================================================================
# Sub-sweep: small cache sizes (BN/page boundary edges).
# Fixed: batch=1, num_heads=64, seq_len=1, W=128.
# Varies: cache_len in {16, 31, 32, 63, 64, 65, 100}.
# Note: cache_len must be > 0 when seq_len=1 to have any reference output.
# ============================================================================
def test_sliding_window_small_cache(ctx: DeviceContext) raises:
    print("=== Sub-sweep: small_cache ===")
    test_decoding[1, 64, 128](ctx, 1, 16)
    test_decoding[1, 64, 128](ctx, 1, 31)
    test_decoding[1, 64, 128](ctx, 1, 32)
    test_decoding[1, 64, 128](ctx, 1, 63)
    test_decoding[1, 64, 128](ctx, 1, 64)
    test_decoding[1, 64, 128](ctx, 1, 65)
    test_decoding[1, 64, 128](ctx, 1, 100)


# ============================================================================
# Sub-sweep: medium cache sizes (page-size boundaries).
# Fixed: batch=1, num_heads=64, seq_len=1, W=128.
# Varies: cache_len in {127, 128, 129, 256, 511, 512, 1024}.
# ============================================================================
def test_sliding_window_medium_cache(ctx: DeviceContext) raises:
    print("=== Sub-sweep: medium_cache ===")
    test_decoding[1, 64, 128](ctx, 1, 127)
    test_decoding[1, 64, 128](ctx, 1, 128)
    test_decoding[1, 64, 128](ctx, 1, 129)
    test_decoding[1, 64, 128](ctx, 1, 256)
    test_decoding[1, 64, 128](ctx, 1, 511)
    test_decoding[1, 64, 128](ctx, 1, 512)
    test_decoding[1, 64, 128](ctx, 1, 1024)


# ============================================================================
# Sub-sweep: large cache sizes (multi-split, deep pipeline).
# Fixed: batch=1, num_heads=64, seq_len=1, W=512.
# Varies: cache_len in {2048, 4096, 8192, 16384, 32768}.
# ============================================================================
def test_sliding_window_large_cache(ctx: DeviceContext) raises:
    print("=== Sub-sweep: large_cache ===")
    test_decoding[1, 64, 512](ctx, 1, 2048)
    test_decoding[1, 64, 512](ctx, 1, 4096)
    test_decoding[1, 64, 512](ctx, 1, 8192)
    test_decoding[1, 64, 512](ctx, 1, 16384)
    test_decoding[1, 64, 512](ctx, 1, 32768)


# ============================================================================
# Sub-sweep: fold variations (seq_len in {1,2,4,8,16}).
# Fixed: batch=1, cache=256, W=128.
# Varies: (num_heads, seq_len) so num_heads*seq_len <= 64 -> fold path.
#   - num_heads=16, seq_len in {1,2,4} (16,32,64 <= 64 -> fold)
#   - num_heads=8,  seq_len=8  (64 <= 64 -> fold)
#   - num_heads=4,  seq_len=16 (64 <= 64 -> fold)
# ============================================================================
def test_sliding_window_fold_variations(ctx: DeviceContext) raises:
    print("=== Sub-sweep: fold_variations ===")
    test_decoding[1, 16, 128](ctx, 1, 256)
    test_decoding[1, 16, 128](ctx, 2, 256)
    test_decoding[1, 16, 128](ctx, 4, 256)
    test_decoding[1, 8, 128](ctx, 8, 256)
    test_decoding[1, 4, 128](ctx, 16, 256)


# ============================================================================
# Sub-sweep: non-fold seq_len=16 path.
# Fixed: batch=1, W=128, cache=256.
# Varies: num_heads in {16, 64, 128} -> num_heads*16 in {256, 1024, 2048} > 64
#         so fold is NOT active; exercises the non-fold seq_len=16 path.
# ============================================================================
def test_sliding_window_non_fold_seq16(ctx: DeviceContext) raises:
    print("=== Sub-sweep: non_fold_seq16 ===")
    test_decoding[1, 16, 128](ctx, 16, 256)
    test_decoding[1, 64, 128](ctx, 16, 256)
    test_decoding[1, 128, 128](ctx, 16, 256)


# ============================================================================
# Sub-sweep: W edge cases.
# Fixed: num_heads=64, batch=1, seq_len=1, cache=256.
# - W=1               only diagonal (current Q sees current K only)
# - W=cache exactly   = 256
# - W=page_size-1     = 127
# - W=page_size       = 128
# - W=page_size+1     = 129
# - W >= cache+seq    sliding window degenerates to causal (W=2048, cache=256)
# Plus a W=1 stress at large cache to exercise short effective_split_len.
# ============================================================================
def test_sliding_window_w_edges(ctx: DeviceContext) raises:
    print("=== Sub-sweep: w_edges ===")
    test_decoding[1, 64, 1](ctx, 1, 256)
    test_decoding[1, 64, 256](ctx, 1, 256)
    test_decoding[1, 64, 127](ctx, 1, 256)
    test_decoding[1, 64, 128](ctx, 1, 256)
    test_decoding[1, 64, 129](ctx, 1, 256)
    # W >= cache + seq -> degenerates to causal (kernel must match causal).
    test_decoding[1, 64, 2048](ctx, 1, 256)
    # W=1 with large cache: effective_split_len capped at W+seq=2.
    test_decoding[1, 64, 1](ctx, 1, 4096)


# ============================================================================
# Sub-sweep: explicit page_size_64 vs page_size_128 coverage.
# split_page_size=64 requires effective_split_len <= 512 AND batch_size >= 32.
# With sliding window, effective_split_len = min(cache_len, W + seq_len).
#
# page_size_64 cases: batch=32, W small enough that W+seq <= 512.
# page_size_128 cases: batch < 32 OR W large.
# ============================================================================
def test_sliding_window_page_sizes(ctx: DeviceContext) raises:
    print("=== Sub-sweep: page_sizes ===")
    # split_page_size=64 path: bs=32, cache=256, W=256 -> eff=256 <= 512 ok.
    test_decoding[32, 64, 256](ctx, 1, 256)
    # split_page_size=64 path: bs=32, large cache but W small (W+seq=129).
    test_decoding[32, 64, 128](ctx, 1, 4096)
    # split_page_size=128 path: bs=1, large cache.
    test_decoding[1, 64, 1024](ctx, 1, 4096)
    # split_page_size=128 path: bs=32 but eff > 512 (W=1024 -> eff=1025).
    test_decoding[32, 64, 1024](ctx, 1, 2048)


# ============================================================================
# Sub-sweep: production model configs.
# DeepSeek V3/R1: num_heads=128, kv_lora_rank=512 (depth=576).
# Kimi K2.5:      num_heads=64,  kv_lora_rank=512 (depth=576).
# ============================================================================
def test_sliding_window_production_configs(ctx: DeviceContext) raises:
    print("=== Sub-sweep: production_configs ===")
    # DeepSeek V3/R1 (h=128).
    test_decoding[1, 128, 256](ctx, 1, 1024)
    test_decoding[1, 128, 512](ctx, 1, 4096)
    test_decoding[1, 128, 1024](ctx, 1, 8192)
    test_decoding[4, 128, 512](ctx, 1, 2048)
    # Kimi K2.5 (h=64).
    test_decoding[1, 64, 256](ctx, 1, 1024)
    test_decoding[1, 64, 512](ctx, 1, 4096)
    test_decoding[1, 64, 1024](ctx, 1, 8192)
    test_decoding[4, 64, 512](ctx, 1, 2048)


# ============================================================================
# Sub-sweep: batch_size variations (1, 4, 16, 32).
# Fixed: num_heads=64, seq_len=1, W=256, cache=1024.
# Exercises batch-bound vs split-bound regimes; bs=32 hits page_size=64 path
# when combined with short effective_split_len.
# ============================================================================
def test_sliding_window_batch_variations(ctx: DeviceContext) raises:
    print("=== Sub-sweep: batch_variations ===")
    test_decoding[1, 64, 256](ctx, 1, 1024)
    test_decoding[4, 64, 256](ctx, 1, 1024)
    test_decoding[16, 64, 256](ctx, 1, 1024)
    test_decoding[32, 64, 256](ctx, 1, 1024)


def main() raises:
    print("Starting test_mla_decode_qkv_fp8_sliding_window...")
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            test_sliding_window_small_cache(ctx)
            test_sliding_window_medium_cache(ctx)
            test_sliding_window_large_cache(ctx)
            test_sliding_window_fold_variations(ctx)
            test_sliding_window_non_fold_seq16(ctx)
            test_sliding_window_w_edges(ctx)
            test_sliding_window_page_sizes(ctx)
            test_sliding_window_production_configs(ctx)
            test_sliding_window_batch_variations(ctx)

            print("All sliding-window tests passed!")
        else:
            print("Skipping: requires NVIDIA SM10x GPU (B200)")
