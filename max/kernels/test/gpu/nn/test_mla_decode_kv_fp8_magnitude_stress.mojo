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

"""Magnitude-stress finiteness probe for the BF16-Q + FP8-KV MLA decode kernel
(`MLA_SM100_Decode_KV_FP8`, SM100/B200).

WHY THIS TEST EXISTS:
A DeepSeek-V3-style MLA FP8 KV cache can be stored UNSCALED (the fused
RoPE+RMSNorm kernel writes `norm_val.cast[float8_e4m3fn]()` /
`roped_val.cast[...]()` with no scale, mla_graph.mojo) when the cache has no
valid quantization `scale_dtype` — effective scale = 1.0, max e4m3 magnitude =
±448. The existing decode tests (test_mla_decode_kv_fp8.mojo) only feed `randn`
(O(1)) KV, so they never exercise the saturation regime that a large reused FP8
context can hit. This test drives the SAME kernel with KV values pushed toward /
past ±448 and asserts the kernel output stays FINITE (the FP8 attention softmax
is fp32 + max-stabilized, so saturated KV must degrade accuracy, never NaN).

It also sweeps the new-query length q (seq_len) ∈ {1,2,3,4,8} to separate two
regimes:
  - unscaled-KV magnitude effect: q-INDEPENDENT, present at q=1 too.
  - q>1 multi-token decode-branch execution: only q≥2.

Reference: numerical equivalence vs `mha_gpu_naive` over the SAME (saturated) KV
bytes — both kernel and reference read the e4m3-cast-then-bf16 K, so on a CORRECT
kernel they must still agree AND both be finite. A finite reference + NaN kernel
=> kernel bug. A NaN reference too => the math itself overflows at this magnitude
(still a real serving failure for unscaled KV, just not kernel-specific)."""

from std.collections import Optional
from std.random import randn, seed
from std.sys import argv, has_nvidia_gpu_accelerator

from max.gpu import *
from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
from std.utils.index import Index
from std.utils.numerics import isnan
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
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_operand import LayoutTensorMHAOperand
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from std.testing import assert_almost_equal, assert_equal


@always_inline
def not_finite[dtype: DType](x: Scalar[dtype]) -> Bool:
    # NaN via isnan; Inf via magnitude (any real Inf exceeds 1e30, while bf16/f32
    # finite values of interest here are far smaller). Avoids depending on an
    # `isinf` export that the stdlib numerics module may not provide.
    return isnan(x) or (abs(x.cast[.float64]()) > Float64(1e30))


@fieldwise_init
struct MLAMaskType(TrivialRegisterPassable):
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
    for b in range(batch_size):
        var b_off = b * num_keys * kv_num_heads * depth
        for i in range(num_keys):
            for h in range(kv_num_heads):
                var base = b_off + (i * kv_num_heads + h) * depth
                for j in range(depth):
                    k_bf16[base + j] = k_fp8[base + j].cast[k_bf16_t]()


# Magnitude-stress: scale standard-normal samples so a large fraction land
# near or above the e4m3 max (±448). `stress` ~= target peak magnitude; with
# randn (|x| up to ~4-5 sigma) a stress of ~120 pushes the tail past 448 and
# the bulk into the high-magnitude e4m3 range where precision collapses.
@always_inline
def magnitude_stress_inplace[
    dtype: DType
](buf: MutPointer[Scalar[dtype], _], n: Int, stress: Float32):
    for i in range(n):
        buf[i] = (buf[i].cast[.float32]() * stress).cast[dtype]()


def test[
    mla_mask_type: MLAMaskType,
    q_type: DType,
    kv_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    batch_size: Int = 1,
    num_partitions: Optional[Int] = None,
    decoding_warp_split_k: Bool = False,
](
    seq_len: Int,
    num_keys: Int,
    q_stress: Float32,
    kv_stress: Float32,
    ctx: DeviceContext,
) raises -> Int:
    """Returns the number of non-finite (NaN/Inf) elements in the kernel output.
    """
    comptime scale = Float32(0.125)
    comptime kv_num_heads = num_heads // group

    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var o_size = q_size

    var q_ptr = ctx.enqueue_create_host_buffer[q_type](q_size)
    var k_ptr = ctx.enqueue_create_host_buffer[kv_type](k_size)
    var k_bf16_ptr = ctx.enqueue_create_host_buffer[q_type](k_size)
    var output_ptr = ctx.enqueue_create_host_buffer[q_type](o_size)
    var flash_output_ptr = ctx.enqueue_create_host_buffer[q_type](o_size)

    seed(0)
    randn(q_ptr.as_span())
    randn(k_ptr.as_span())

    # Magnitude-stress the inputs toward the e4m3 saturation boundary.
    magnitude_stress_inplace[q_type](q_ptr.unsafe_ptr(), q_size, q_stress)
    magnitude_stress_inplace[kv_type](k_ptr.unsafe_ptr(), k_size, kv_stress)

    host_cast_k_fp8_to_bf16[kv_fp8_t=kv_type, k_bf16_t=q_type](
        k_ptr.unsafe_ptr(),
        k_bf16_ptr.unsafe_ptr(),
        depth,
        num_keys,
        kv_num_heads,
        batch_size,
    )

    var q_device_ptr = ctx.enqueue_create_buffer[q_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[kv_type](k_size)
    var output_device_ptr = ctx.enqueue_create_buffer[q_type](o_size)

    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)

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

    comptime k_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, kv_num_heads, depth)
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        _is_cache_length_accurate=True,
        is_fp8_kv=True,
    ](batch_size, num_keys, seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    @__parameter
    @always_inline
    @__copy_capture(q_tt, k_tt, out_tt, scalar_args_buf_tt)
    def kernel_launch(ctx: DeviceContext) raises:
        # CAUSAL only (production MLA mask). See main() — every cell is CAUSAL.
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

    kernel_launch(ctx)
    ctx.synchronize()
    ctx.enqueue_copy(flash_output_ptr, output_device_ptr)

    # Reference over the SAME saturated KV bytes.
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
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(Index(0)),
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
    else:
        # This stress harness only drives the CAUSAL reference (production MLA
        # mask). NO_MASK is intentionally unsupported here.
        raise Error("magnitude-stress harness supports CAUSAL only")

    ctx.synchronize()
    ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
    ctx.synchronize()

    # Finiteness probe + equivalence over the valid V-depth (depth-64).
    var kernel_nonfinite = 0
    var ref_nonfinite = 0
    var max_abs_err = Float64(0)
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(depth - 64):
                    var actual = flash_output_ptr[
                        d
                        + (depth - 64) * (h + s * num_heads)
                        + b * (depth - 64) * num_heads * seq_len
                    ]
                    var expect = output_ptr[
                        d
                        + depth * (h + s * num_heads)
                        + b * depth * num_heads * seq_len
                    ]
                    if not_finite(actual):
                        kernel_nonfinite += 1
                    if not_finite(expect):
                        ref_nonfinite += 1
                    if not not_finite(actual) and not not_finite(expect):
                        var e = abs(
                            actual.cast[.float64]() - expect.cast[.float64]()
                        )
                        if e > max_abs_err:
                            max_abs_err = e

    print(
        "  q=",
        seq_len,
        "num_heads=",
        num_heads,
        "num_keys=",
        num_keys,
        "np=",
        num_partitions.value() if num_partitions else -1,
        "kv_stress=",
        kv_stress,
        "mask=",
        mla_mask_type.value,
        "-> kernel_nonfinite=",
        kernel_nonfinite,
        "ref_nonfinite=",
        ref_nonfinite,
        "max_abs_err=",
        max_abs_err,
    )

    _ = mla_args
    _ = q_device_ptr
    _ = k_device_ptr
    _ = output_device_ptr
    _ = output_ref_device_ptr
    _ = k_ref_device_ptr
    return kernel_nonfinite


# Sweep one (num_heads, num_keys, mask) cell across q ∈ {1,2,3,4,8} at a fixed
# kv_stress, printing the finiteness verdict per q. Returns total kernel
# non-finite count across the q sweep.
def sweep_q[
    num_heads: Int,
    group: Int,
    mla_mask_type: MLAMaskType,
](num_keys: Int, kv_stress: Float32, ctx: DeviceContext) raises -> Int:
    var total = 0
    total += test[
        mla_mask_type,
        DType.bfloat16,
        DType.float8_e4m3fn,
        576,
        num_heads,
        group=group,
        batch_size=1,
        num_partitions=Optional[Int](1),
    ](1, num_keys, 1.0, kv_stress, ctx)
    total += test[
        mla_mask_type,
        DType.bfloat16,
        DType.float8_e4m3fn,
        576,
        num_heads,
        group=group,
        batch_size=1,
        num_partitions=Optional[Int](1),
    ](2, num_keys, 1.0, kv_stress, ctx)
    total += test[
        mla_mask_type,
        DType.bfloat16,
        DType.float8_e4m3fn,
        576,
        num_heads,
        group=group,
        batch_size=1,
        num_partitions=Optional[Int](1),
    ](3, num_keys, 1.0, kv_stress, ctx)
    total += test[
        mla_mask_type,
        DType.bfloat16,
        DType.float8_e4m3fn,
        576,
        num_heads,
        group=group,
        batch_size=1,
        num_partitions=Optional[Int](1),
    ](4, num_keys, 1.0, kv_stress, ctx)
    total += test[
        mla_mask_type,
        DType.bfloat16,
        DType.float8_e4m3fn,
        576,
        num_heads,
        group=group,
        batch_size=1,
        num_partitions=Optional[Int](1),
    ](8, num_keys, 1.0, kv_stress, ctx)
    return total


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            # Two per-rank head counts: 16 (e.g. 128 heads at TP=8), 32 (TP=4).
            # Cache lengths incl. a non-multiple-of-64 partial last tile (3050)
            # and large reused contexts (4096, 65536).
            # kv_stress sweep: 1.0 (benign control) and 120.0 (push past ±448).
            var grand_total = 0
            for num_keys in [1024, 3050, 4096, 65536]:
                for kv_stress in [Float32(1.0), Float32(120.0)]:
                    print(
                        "=== num_heads=16 num_keys=",
                        num_keys,
                        "kv_stress=",
                        kv_stress,
                        "(CAUSAL) ===",
                    )
                    grand_total += sweep_q[16, 16, MLAMaskType.CAUSAL](
                        num_keys, kv_stress, ctx
                    )
                    print(
                        "=== num_heads=32 num_keys=",
                        num_keys,
                        "kv_stress=",
                        kv_stress,
                        "(CAUSAL) ===",
                    )
                    grand_total += sweep_q[32, 32, MLAMaskType.CAUSAL](
                        num_keys, kv_stress, ctx
                    )
            print("GRAND_TOTAL_KERNEL_NONFINITE=", grand_total)
            # Saturating FP8 quantization must keep the decode output finite at
            # every magnitude: the FP8 attention softmax is fp32 +
            # max-stabilized, so over-range KV must degrade accuracy, never
            # produce NaN/Inf. Any non-finite element across the whole sweep is
            # a kernel regression.
            assert_equal(grand_total, 0)
        else:
            print("skipped: not an SM100 GPU")
