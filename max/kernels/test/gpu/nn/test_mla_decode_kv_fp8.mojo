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

from std.collections import Optional
from std.random import randn
from std.sys import argv, has_nvidia_gpu_accelerator

from max.gpu import *
from max.gpu.host import DeviceContext
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    lt_to_tt,
    row_major,
)
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.mha_mask import CausalMask, NullMask
from nn.attention.mha_operand import LayoutTensorMHAOperand
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from std.testing import assert_almost_equal
from max.gpu.host.info import _is_sm10x_gpu
from std.utils.index import Index


# ===-----------------------------------------------------------------------===#
# MLAMaskType
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct MLAMaskType(TrivialRegisterPassable):
    """Enum-like structure for MLA mask types."""

    var value: UInt8

    comptime NO_MASK = Self(0)
    comptime CAUSAL = Self(1)

    def __eq__(self, rhs: Self) -> Bool:
        return self.value == rhs.value

    def __ne__(self, rhs: Self) -> Bool:
        return self.value != rhs.value


@always_inline
def host_cast_k_fp8_to_bf16[
    kv_fp8_t: DType,
    k_bf16_t: DType,
](
    k_fp8: Pointer[Scalar[kv_fp8_t], _],
    k_bf16: MutPointer[Scalar[k_bf16_t], _],
    depth: Int,
    num_keys: Int,
    kv_num_heads: Int,
    batch_size: Int,
):
    # Layout of k in your test:
    # k_ptr[(i * kv_num_heads + h) * depth + j]
    for b in range(batch_size):
        var b_off = b * num_keys * kv_num_heads * depth
        for i in range(num_keys):
            for h in range(kv_num_heads):
                var base = b_off + (i * kv_num_heads + h) * depth
                for j in range(depth):
                    k_bf16[base + j] = k_fp8[base + j].cast[k_bf16_t]()


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def test[
    mla_mask_type: MLAMaskType,
    q_type: DType,
    kv_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    against_gpu_naive: Bool = False,
    batch_size: Int = 1,
    num_partitions: Optional[Int] = None,
    decoding_warp_split_k: Bool = False,
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
        "q_type:",
        q_type,
        "kv_type:",
        kv_type,
        "mla_mask_type:",
        mla_mask_type.value,
    )

    comptime assert (
        mla_mask_type == MLAMaskType.NO_MASK
        or mla_mask_type == MLAMaskType.CAUSAL
    ), "mha only supports NO_MASK or CAUSAL."

    # Query, key, value dimensions.
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    # var v_size = k_size
    var o_size = q_size

    # Allocate memory for all variables.
    var q_ptr = ctx.enqueue_create_host_buffer[q_type](q_size)
    var k_ptr = ctx.enqueue_create_host_buffer[kv_type](k_size)  # fp8 host
    var k_bf16_ptr = ctx.enqueue_create_host_buffer[q_type](k_size)
    var output_ptr = ctx.enqueue_create_host_buffer[q_type](o_size)
    var flash_output_ptr = ctx.enqueue_create_host_buffer[q_type](o_size)

    # Q, K, V are randomly initialized.
    randn(q_ptr.as_span())
    randn(k_ptr.as_span())

    host_cast_k_fp8_to_bf16[kv_fp8_t=kv_type, k_bf16_t=q_type](
        k_ptr.unsafe_ptr(),
        k_bf16_ptr.unsafe_ptr(),
        depth,
        num_keys,
        kv_num_heads,
        batch_size,
    )

    # Device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[q_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[kv_type](k_size)
    var output_device_ptr = ctx.enqueue_create_buffer[q_type](o_size)

    # Copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)

    # Construct TileTensors for Q, K, output.
    var q_tt = TileTensor(
        q_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )
    var k_tt = TileTensor(
        k_device_ptr,
        row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth])),
    )
    var out_tt = TileTensor(
        output_device_ptr,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )

    # Keep LayoutTensors for mha_gpu_naive reference path.
    comptime k_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, kv_num_heads, depth)
    )

    comptime q_tile_num_rows = 32
    comptime k_tile_num_rows = 128

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        _is_cache_length_accurate=True,
        is_fp8_kv=True,
    ](batch_size, num_keys, seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    @always_inline
    def kernel_launch(
        ctx: DeviceContext,
    ) raises {var q_tt, var k_tt, var out_tt, var scalar_args_buf_tt, imm}:
        comptime if mla_mask_type == MLAMaskType.CAUSAL:
            flare_mla_decoding[
                config=MHAConfig[q_type](num_heads, depth),
                decoding_warp_split_k=decoding_warp_split_k,
            ](
                out_tt.as_unsafe_any_origin(),
                q_tt,
                k_tt,
                CausalMask(),
                scale,
                ctx,
                scalar_args_buf_tt,
                num_partitions=num_partitions,
            )
        elif mla_mask_type == MLAMaskType.NO_MASK:
            flare_mla_decoding[
                config=MHAConfig[q_type](num_heads, depth),
                decoding_warp_split_k=decoding_warp_split_k,
            ](
                out_tt.as_unsafe_any_origin(),
                q_tt,
                k_tt,
                NullMask(),
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
        var output_ref_device_ptr = ctx.enqueue_create_buffer[q_type](o_size)
        comptime output_ref_layout = Layout.row_major(
            Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
        )
        var output_ref_device = LayoutTensor[q_type, output_ref_layout](
            output_ref_device_ptr.unsafe_ptr(),
            RuntimeLayout[output_ref_layout].row_major(
                Index(batch_size, seq_len, num_heads, depth)
            ),
        )
        ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

        var k_ref_device_ptr = ctx.enqueue_create_buffer[q_type](k_size)

        var k_ref_device = LayoutTensor[q_type, k_layout](
            k_ref_device_ptr.unsafe_ptr(),
            RuntimeLayout[k_layout].row_major(
                Index(batch_size, num_keys, kv_num_heads, depth)
            ),
        )
        ctx.enqueue_copy(k_ref_device_ptr, k_bf16_ptr)

        comptime if mla_mask_type == MLAMaskType.CAUSAL:
            var k_operand = LayoutTensorMHAOperand(lt_to_tt(k_ref_device))
            var null_valid_length = LayoutTensor[
                .uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
            ](
                None,
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    Index(0)
                ),
            )
            mha_gpu_naive[_is_cache_length_accurate=True,](
                q_tt.to_layout_tensor(),
                k_operand,
                k_operand,
                CausalMask(),
                output_ref_device,
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
        elif mla_mask_type == MLAMaskType.NO_MASK:
            mha_gpu_naive(
                q_tt.to_layout_tensor(),
                k_tt.to_layout_tensor(),
                k_tt.to_layout_tensor(),
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
        _ = output_ref_device_ptr

    ctx.synchronize()

    if o_size == 0:
        return

    # since we pass the whole K tensor as the V tensor to our naive mha kernel,
    # the last 64 elements of each head in the reference result are invalid.
    # b , s, h, d
    var rtol = 5e-2  # 0.05
    var atol = 3e-1  # 0.3
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
                    if abs((actual - expect)) > 1e-1:
                        print(b, h, s, d, actual, expect)
                    assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

    _ = mla_args
    _ = q_device_ptr
    _ = k_device_ptr
    _ = output_device_ptr


def test_decoding[
    batch_size: Int,
    mla_mask_type: MLAMaskType,
    split_k: Bool = False,
    num_partitions: Optional[Int] = 1,
    against_gpu_naive: Bool = True,
](
    ctx: DeviceContext, use_index_input: Bool, seq_len: Int, num_keys: Int
) raises:
    test[
        mla_mask_type,
        DType.bfloat16,  # q_type
        DType.float8_e4m3fn,  # kv_type  (fp8 KV)
        576,
        16,
        group=16,
        against_gpu_naive=against_gpu_naive,
        batch_size=batch_size,
        num_partitions=num_partitions,
        decoding_warp_split_k=split_k,
    ](seq_len, num_keys, ctx, use_index_input=use_index_input)

    test[
        mla_mask_type,
        DType.bfloat16,  # q_type
        DType.float8_e4m3fn,  # kv_type  (fp8 KV)
        576,
        64,
        group=64,
        against_gpu_naive=against_gpu_naive,
        batch_size=batch_size,
        num_partitions=num_partitions,
        decoding_warp_split_k=split_k,
    ](seq_len, num_keys, ctx, use_index_input=use_index_input)

    test[
        mla_mask_type,
        DType.bfloat16,  # q_type
        DType.float8_e4m3fn,  # kv_type  (fp8 KV)
        576,
        128,
        group=128,
        against_gpu_naive=against_gpu_naive,
        batch_size=batch_size,
        num_partitions=num_partitions,
        decoding_warp_split_k=split_k,
    ](seq_len, num_keys, ctx, use_index_input=use_index_input)


def test_decoding_partial_head_group[
    batch_size: Int,
    mla_mask_type: MLAMaskType,
    split_k: Bool = False,
    num_partitions: Optional[Int] = 1,
    against_gpu_naive: Bool = True,
](
    ctx: DeviceContext, use_index_input: Bool, seq_len: Int, num_keys: Int
) raises:
    """Covers head counts whose last head group is partial, with a
    full-multiple control at the same shape so a failure is attributable to
    the head count alone. The smallest count leaves a tail narrower than the
    output swizzle atom, which checks that the store bounds rows at element
    granularity rather than atom granularity.
    """
    comptime for num_heads in [128, 96, 72, 68]:
        test[
            mla_mask_type,
            DType.bfloat16,  # q_type
            DType.float8_e4m3fn,  # kv_type  (fp8 KV)
            576,
            num_heads,
            group=num_heads,
            against_gpu_naive=against_gpu_naive,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
        ](seq_len, num_keys, ctx, use_index_input=use_index_input)


def test_decoding_k3_head_counts[
    batch_size: Int,
    mla_mask_type: MLAMaskType,
    split_k: Bool = False,
    num_partitions: Optional[Int] = 1,
](ctx: DeviceContext, seq_len: Int, num_keys: Int) raises:
    """Kimi K3's TP8 Q-head count, 12, with 128 as the same-shape control.

    Every other single-head-group count in this file is a power of two. 12 is
    not, and it works — so the kernel is not restricted to powers of two, and
    the `compute_mla_dispatch_scalars_runtime` enumeration is the only thing
    that was in the way at that count.

    K3's TP1 count, 96, is covered by `test_decoding_partial_head_group` above,
    which owns every count that leaves a partial tail head group. 12 leaves no
    partial *head* group but does tail the combine grid's 8-head block, which
    only a split-K run reaches.

    The control comes FIRST at this exact shape and batch: an assert aborts the
    run, so ordering a supported count ahead of an unusual one is what makes a
    failure attributable to the head count rather than to the shape.
    """
    comptime for num_heads in [128, 12]:
        test[
            mla_mask_type,
            DType.bfloat16,  # q_type
            DType.float8_e4m3fn,  # kv_type  (fp8 KV)
            576,
            num_heads,
            group=num_heads,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
        ](seq_len, num_keys, ctx)


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            # Test with benchmark parameters: batch_size=1, cache_len=32768, num_heads=128
            test_decoding[1, MLAMaskType.NO_MASK](ctx, False, 1, 32768)
            test_decoding[1, MLAMaskType.CAUSAL](ctx, False, 1, 32768)
            test_decoding[128, MLAMaskType.NO_MASK](ctx, False, 1, 1024)
            test_decoding[128, MLAMaskType.CAUSAL](ctx, False, 2, 1024)
            test_decoding[64, MLAMaskType.NO_MASK](ctx, False, 1, 2048)
            test_decoding[64, MLAMaskType.CAUSAL](ctx, False, 2, 2048)
            test_decoding[64, MLAMaskType.CAUSAL](ctx, False, 3, 50)
            test_decoding[64, MLAMaskType.NO_MASK](ctx, False, 4, 193)
            test_decoding[27, MLAMaskType.CAUSAL](ctx, False, 5, 50)
            test_decoding[64, MLAMaskType.CAUSAL](ctx, False, 6, 517)
            test_decoding[1, MLAMaskType.NO_MASK](ctx, False, 1, 32768 * 2)
            test_decoding_partial_head_group[1, MLAMaskType.NO_MASK](
                ctx, False, 1, 1024
            )
            test_decoding_partial_head_group[2, MLAMaskType.CAUSAL](
                ctx, False, 2, 4096
            )
            # Under split-K a tail overrun would corrupt the next split's
            # partials, which the combine kernel then folds into the output.
            test_decoding_partial_head_group[1, MLAMaskType.NO_MASK, True, 2](
                ctx, False, 1, 1024
            )
            test_decoding_k3_head_counts[2, MLAMaskType.CAUSAL](ctx, 1, 512)
            test_decoding_k3_head_counts[1, MLAMaskType.NO_MASK](ctx, 1, 4096)
            test_decoding_k3_head_counts[1, MLAMaskType.NO_MASK, True, 2](
                ctx, 1, 4096
            )
        else:
            pass
