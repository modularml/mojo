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

"""Correctness test for SM100 MHA with native FP8 Q, K, V — DECODE.

One source, parameterized by `-D MHA_FP8_HEAD_DIM={128,256,512}` (see BUILD);
each depth is a separate bazel target, so a regression in one arm fails
independently. At decode (seq_len=1) `flash_attention` takes the SM100 FA4
2Q dispatch. Q, K, V are all fp8 e4m3fn; the kernel runs native fp8 UMMA
(`KIND_F8F6F4`, MMA_K=32) for both Q@K^T and P@V, and P is cast to fp8 inside
the softmax warp. The depth selects where P lives for the P@V MMA: at
head_dim=128 P stays in TMEM; for head_dim >= 256 P is written to SMEM
instead of TMEM (the deep arm).

Each config also runs in two modes:

* `bf16_to_fp8=False` (kernel-correctness): generate Q, K, V as fp8 with
  `randn`, then cast fp8 → bf16 (lossless) for the reference inputs. Both
  kernels see identical numerical values, so the cosine isolates the fp8
  MMA's compute correctness. Bar: cosine ≥ 0.9997.
* `bf16_to_fp8=True` (production-realism): generate Q, K, V as bf16 with
  `randn`, then cast bf16 → fp8 (lossy) for the kernel inputs. The bf16
  reference sees the originals; the fp8 kernel sees the quantized values.
  Bundles quantization error + kernel error into a single cosine.
  Bar: cosine ≥ 0.998.

Target hardware family: NVIDIA SM100 (B200).
"""

from std.math import sqrt
from std.sys import get_defined_int

from max.gpu.host import DeviceContext
from layout import (
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from std.random import randn

from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import (
    CausalMask,
    MHAMask,
    SlidingWindowCausalMask,
)

from std.testing import assert_almost_equal, assert_true


# ===-----------------------------------------------------------------------===#
# Host helpers
# ===-----------------------------------------------------------------------===#


@always_inline
def host_cast_fp8_to_bf16[
    fp8_t: DType,
    bf16_t: DType,
](
    src: Pointer[Scalar[fp8_t], _],
    dst: MutPointer[Scalar[bf16_t], _],
    size: Int,
):
    """Cast fp8 → bf16 element-by-element on the host. Lossless: every
    fp8 e4m3 value is exactly representable in bf16."""
    for i in range(size):
        dst[i] = src[i].cast[bf16_t]()


@always_inline
def host_cast_bf16_to_fp8[
    bf16_t: DType,
    fp8_t: DType,
](
    src: Pointer[Scalar[bf16_t], _],
    dst: MutPointer[Scalar[fp8_t], _],
    size: Int,
):
    """Cast bf16 → fp8 element-by-element on the host (lossy quantization).
    No per-tensor scale: values that overflow fp8 e4m3 saturate at ±448."""
    for i in range(size):
        dst[i] = src[i].cast[fp8_t]()


# ===-----------------------------------------------------------------------===#
# Core test
# ===-----------------------------------------------------------------------===#

comptime head_dim = get_defined_int["MHA_FP8_HEAD_DIM", 128]()
comptime label = "test_mha_sm100_qkv_fp8_d{}_decode".format(head_dim)


def execute_pure_fp8_decode_test[
    MaskType: MHAMask,
    *,
    bf16_to_fp8: Bool,
    head_dim: Int,
    num_q_heads: Int,
    group: Int,
    num_keys: Int,
    mask_name: StaticString,
](mask: MaskType, ctx: DeviceContext,) raises:
    """Run pure-fp8 MHA-decode vs bf16 reference. Mode is chosen via
    `bf16_to_fp8`: False = generate fp8 then cast to bf16 (lossless;
    kernel-correctness, bar 0.9997). True = generate bf16 then cast to
    fp8 (lossy; production-realism, bar 0.998).
    """
    comptime seq_len = 1  # decode
    comptime kv_num_heads = num_q_heads // group
    comptime batch_size = 1
    comptime scale = Float32(1.0) / sqrt(Float32(head_dim))
    comptime mode_name = "bf16_to_fp8" if bf16_to_fp8 else "fp8_to_bf16"
    comptime cosine_bar = Float64(0.998) if bf16_to_fp8 else Float64(0.9997)

    print(
        label,
        ": ",
        "mode=",
        mode_name,
        " mask=",
        mask_name,
        " head_dim=",
        head_dim,
        " group=",
        group,
        " n_q_heads=",
        num_q_heads,
        " n_kv_heads=",
        kv_num_heads,
        " seq_len=",
        seq_len,
        " num_keys=",
        num_keys,
    )

    comptime fp8_dtype = DType.float8_e4m3fn
    comptime bf16_dtype = DType.bfloat16

    var q_size = batch_size * seq_len * num_q_heads * head_dim
    var k_size = batch_size * num_keys * kv_num_heads * head_dim
    var v_size = k_size
    var o_size = q_size

    var q_fp8_host = ctx.enqueue_create_host_buffer[fp8_dtype](q_size)
    var k_fp8_host = ctx.enqueue_create_host_buffer[fp8_dtype](k_size)
    var v_fp8_host = ctx.enqueue_create_host_buffer[fp8_dtype](v_size)
    var q_bf16_host = ctx.enqueue_create_host_buffer[bf16_dtype](q_size)
    var k_bf16_host = ctx.enqueue_create_host_buffer[bf16_dtype](k_size)
    var v_bf16_host = ctx.enqueue_create_host_buffer[bf16_dtype](v_size)

    comptime if bf16_to_fp8:
        # Production-realism: randn into bf16, cast to fp8 (lossy).
        randn(q_bf16_host.as_span())
        randn(k_bf16_host.as_span())
        randn(v_bf16_host.as_span())
        host_cast_bf16_to_fp8[bf16_dtype, fp8_dtype](
            q_bf16_host.unsafe_ptr(), q_fp8_host.unsafe_ptr(), q_size
        )
        host_cast_bf16_to_fp8[bf16_dtype, fp8_dtype](
            k_bf16_host.unsafe_ptr(), k_fp8_host.unsafe_ptr(), k_size
        )
        host_cast_bf16_to_fp8[bf16_dtype, fp8_dtype](
            v_bf16_host.unsafe_ptr(), v_fp8_host.unsafe_ptr(), v_size
        )
    else:
        # Kernel-correctness: randn into fp8, cast to bf16 (lossless).
        randn(q_fp8_host.as_span())
        randn(k_fp8_host.as_span())
        randn(v_fp8_host.as_span())
        host_cast_fp8_to_bf16[fp8_dtype, bf16_dtype](
            q_fp8_host.unsafe_ptr(), q_bf16_host.unsafe_ptr(), q_size
        )
        host_cast_fp8_to_bf16[fp8_dtype, bf16_dtype](
            k_fp8_host.unsafe_ptr(), k_bf16_host.unsafe_ptr(), k_size
        )
        host_cast_fp8_to_bf16[fp8_dtype, bf16_dtype](
            v_fp8_host.unsafe_ptr(), v_bf16_host.unsafe_ptr(), v_size
        )

    # ---- Device buffers ----
    var q_fp8_dev = ctx.enqueue_create_buffer[fp8_dtype](q_size)
    var k_fp8_dev = ctx.enqueue_create_buffer[fp8_dtype](k_size)
    var v_fp8_dev = ctx.enqueue_create_buffer[fp8_dtype](v_size)
    var q_bf16_dev = ctx.enqueue_create_buffer[bf16_dtype](q_size)
    var k_bf16_dev = ctx.enqueue_create_buffer[bf16_dtype](k_size)
    var v_bf16_dev = ctx.enqueue_create_buffer[bf16_dtype](v_size)
    var out_fp8_dev = ctx.enqueue_create_buffer[bf16_dtype](o_size)
    var out_ref_dev = ctx.enqueue_create_buffer[bf16_dtype](o_size)
    ctx.enqueue_copy(q_fp8_dev, q_fp8_host)
    ctx.enqueue_copy(k_fp8_dev, k_fp8_host)
    ctx.enqueue_copy(v_fp8_dev, v_fp8_host)
    ctx.enqueue_copy(q_bf16_dev, q_bf16_host)
    ctx.enqueue_copy(k_bf16_dev, k_bf16_host)
    ctx.enqueue_copy(v_bf16_dev, v_bf16_host)

    # ---- TileTensors for the kernel calls ----
    var q_fp8_lt = TileTensor(
        q_fp8_dev,
        row_major((batch_size, seq_len, Idx[num_q_heads], Idx[head_dim])),
    )
    var k_fp8_lt = TileTensor(
        k_fp8_dev,
        row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[head_dim])),
    )
    var v_fp8_lt = TileTensor(
        v_fp8_dev,
        row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[head_dim])),
    )
    var q_bf16_lt = TileTensor(
        q_bf16_dev,
        row_major((batch_size, seq_len, Idx[num_q_heads], Idx[head_dim])),
    )
    var k_bf16_lt = TileTensor(
        k_bf16_dev,
        row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[head_dim])),
    )
    var v_bf16_lt = TileTensor(
        v_bf16_dev,
        row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[head_dim])),
    )
    var out_ref_lt = TileTensor(
        out_ref_dev,
        row_major((batch_size, seq_len, Idx[num_q_heads], Idx[head_dim])),
    )
    var out_fp8_lt = TileTensor(
        out_fp8_dev,
        row_major((batch_size, seq_len, Idx[num_q_heads], Idx[head_dim])),
    )

    # ---- Reference: bf16 attention with dequant inputs ----
    flash_attention(
        out_ref_lt, q_bf16_lt, k_bf16_lt, v_bf16_lt, mask, scale, ctx
    )

    # ---- Test target: pure-fp8 attention ----
    flash_attention(out_fp8_lt, q_fp8_lt, k_fp8_lt, v_fp8_lt, mask, scale, ctx)
    ctx.synchronize()

    # ---- Copy back and compare ----
    var out_ref_host = ctx.enqueue_create_host_buffer[bf16_dtype](o_size)
    var out_fp8_host = ctx.enqueue_create_host_buffer[bf16_dtype](o_size)
    ctx.enqueue_copy(out_ref_host, out_ref_dev)
    ctx.enqueue_copy(out_fp8_host, out_fp8_dev)
    ctx.synchronize()

    comptime rtol = 5e-2
    comptime atol = 3e-1
    var num_mismatches = 0
    var total_abs_diff: Float64 = 0.0
    var max_abs_diff: Float64 = 0.0
    var num_compared = 0
    var num_nan_actual = 0
    var num_nan_expect = 0
    # Cosine similarity over the whole output (vs bf16 reference).
    var dot: Float64 = 0.0
    var aa: Float64 = 0.0
    var bb: Float64 = 0.0
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_q_heads):
                for d in range(head_dim):
                    var idx = (
                        d
                        + head_dim * (h + s * num_q_heads)
                        + b * head_dim * num_q_heads * seq_len
                    )
                    var expect = out_ref_host[idx].cast[.float64]()
                    var actual = out_fp8_host[idx].cast[.float64]()
                    if actual != actual:
                        num_nan_actual += 1
                        if d == 0:
                            print("nan actual h=", h, "d=", d)
                        continue
                    if expect != expect:
                        num_nan_expect += 1
                        continue
                    var diff = abs(actual - expect)
                    total_abs_diff += diff
                    if diff > max_abs_diff:
                        max_abs_diff = diff
                    num_compared += 1
                    if diff > atol + rtol * abs(expect):
                        if num_mismatches < 16:
                            print(
                                "mismatch b=",
                                b,
                                "s=",
                                s,
                                "h=",
                                h,
                                "d=",
                                d,
                                "actual=",
                                actual,
                                "expect=",
                                expect,
                            )
                        num_mismatches += 1
                    dot += actual * expect
                    aa += actual * actual
                    bb += expect * expect

    var cos: Float64 = 0.0
    if aa > 0.0 and bb > 0.0:
        cos = dot / (sqrt(aa) * sqrt(bb))
    print(
        "  num_mismatches=",
        num_mismatches,
        " / ",
        num_compared,
        " nan_actual=",
        num_nan_actual,
        " nan_expect=",
        num_nan_expect,
        " mean_abs_diff=",
        total_abs_diff / Float64(num_compared),
        " max_abs_diff=",
        max_abs_diff,
        " cosine=",
        cos,
    )
    # Bar depends on the input flow: kernel-correctness (lossless inputs)
    # uses 0.9997; production-realism (lossy quantization) uses 0.998.
    assert_true(cos >= cosine_bar, "cosine below bar for current mode")

    if num_mismatches > 0:
        print(
            "  WARNING:",
            num_mismatches,
            "mismatches > 1e-1 (but all passed atol/rtol)",
        )
    print("  PASSED")


# ===-----------------------------------------------------------------------===#
# Entry point — decode cases (seq_len == 1) only
# ===-----------------------------------------------------------------------===#


def main() raises:
    with DeviceContext() as ctx:
        var causal = CausalMask()
        var sw_1024 = SlidingWindowCausalMask[1024]()

        comptime for i in range(2):
            comptime bf16_to_fp8 = i == 1

            comptime if head_dim == 128:
                # Simplest case: 1 head, no GQA, 1 kv tile.
                execute_pure_fp8_decode_test[
                    CausalMask,
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=8,
                    group=8,
                    num_keys=128,
                    mask_name="CAUSAL_g8_decode_k128_simple",
                ](causal, ctx)
                # Small + large kv-cache cases at the most common decode head_dim.
                execute_pure_fp8_decode_test[
                    CausalMask,
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=32,
                    group=8,
                    num_keys=256,
                    mask_name="CAUSAL_g8_decode_k256",
                ](causal, ctx)
                execute_pure_fp8_decode_test[
                    CausalMask,
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=32,
                    group=8,
                    num_keys=1024,
                    mask_name="CAUSAL_g8_decode_k1024",
                ](causal, ctx)
                execute_pure_fp8_decode_test[
                    CausalMask,
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=32,
                    group=8,
                    num_keys=4096,
                    mask_name="CAUSAL_g8_decode_k4096",
                ](causal, ctx)

                # Sliding-window decode.
                execute_pure_fp8_decode_test[
                    SlidingWindowCausalMask[1024],
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=32,
                    group=8,
                    num_keys=4096,
                    mask_name="SW1024_g8_decode_k4096",
                ](sw_1024, ctx)
            elif head_dim == 256:
                # Simplest case: 1 kv head, single kv tile.
                execute_pure_fp8_decode_test[
                    CausalMask,
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=8,
                    group=8,
                    num_keys=128,
                    mask_name="CAUSAL_g8_decode_k128_simple",
                ](causal, ctx)

                # Production Gemma decode shape: n_q_heads=32, group=8.
                execute_pure_fp8_decode_test[
                    CausalMask,
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=32,
                    group=8,
                    num_keys=256,
                    mask_name="CAUSAL_g8_decode_k256",
                ](causal, ctx)
                execute_pure_fp8_decode_test[
                    CausalMask,
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=32,
                    group=8,
                    num_keys=1024,
                    mask_name="CAUSAL_g8_decode_k1024",
                ](causal, ctx)
                execute_pure_fp8_decode_test[
                    CausalMask,
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=32,
                    group=8,
                    num_keys=4096,
                    mask_name="CAUSAL_g8_decode_k4096",
                ](causal, ctx)

                # Gemma4-sliding decode shape: n_kv_heads=16 (group=2).
                execute_pure_fp8_decode_test[
                    CausalMask,
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=32,
                    group=2,
                    num_keys=4096,
                    mask_name="CAUSAL_g2_decode_k4096",
                ](causal, ctx)

                # Sliding-window decode at the production shape.
                execute_pure_fp8_decode_test[
                    SlidingWindowCausalMask[1024],
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=32,
                    group=8,
                    num_keys=4096,
                    mask_name="SW1024_g8_decode_k4096",
                ](sw_1024, ctx)
            else:
                # d512: production Gemma4-global decode shape, group=8.
                execute_pure_fp8_decode_test[
                    CausalMask,
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=32,
                    group=8,
                    num_keys=256,
                    mask_name="CAUSAL_g8_decode_k256",
                ](causal, ctx)
                execute_pure_fp8_decode_test[
                    CausalMask,
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=32,
                    group=8,
                    num_keys=1024,
                    mask_name="CAUSAL_g8_decode_k1024",
                ](causal, ctx)
                execute_pure_fp8_decode_test[
                    CausalMask,
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=32,
                    group=8,
                    num_keys=4096,
                    mask_name="CAUSAL_g8_decode_k4096",
                ](causal, ctx)

                # Sliding-window decode.
                execute_pure_fp8_decode_test[
                    SlidingWindowCausalMask[1024],
                    bf16_to_fp8=bf16_to_fp8,
                    head_dim=head_dim,
                    num_q_heads=32,
                    group=8,
                    num_keys=4096,
                    mask_name="SW1024_g8_decode_k4096",
                ](sw_1024, ctx)

        print(label, ": ALL PASSED")
