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

"""Tail-stress correctness test for the SM100 native-FP8 MHA 256x P-scale.

This test is a REGRESSION GUARD for the cuDNN-style fixed P scale of 256
(= 2^8) implemented in `sm100/softmax_warp.mojo` (the d=128 2Q path). The
existing `test_mha_sm100_qkv_fp8_d*` tests use `randn` inputs that produce
flat softmax distributions with no informative underflow tail, so the
256x lift barely moves their error (cos ~0.9998, -0.2%). They do not
discriminate the lever.

Construction (scores engineered directly, NOT randn):
  - Q[query, :] = A in dim 0 only (zeros elsewhere).
  - K[key 0, :] = g0 in dim 0 (the PEAK key); K[key j>0, :] = 0 (TAIL keys).
    => raw score(query, key0) = A*g0; raw score(query, key j>0) = 0.
    With A=8, g0=10, head_dim=128, scale = 1/sqrt(128):
      score_peak = 80 / 11.3137 = 7.071  (in log domain, * scale)
      score_tail = 0
    => P_tail / P_peak = exp(-7.071) ~= 8.5e-4 (deep in the e4m3
       underflow zone: smallest e4m3 subnormal is 2^-9 ~= 1.95e-3, so an
       UNSCALED P_tail of 8.5e-4 casts to fp8 ZERO).
  - V[key 0] (peak) = 0 in all dims; V[key j>0] (tail) = 1.0 in all dims.
    => the output is carried ENTIRELY by the tail's V. Each tail prob
       underflows individually, but they collectively carry real mass.
  - CausalMask, seq_len = num_keys = 256. Row r attends keys [0..r], so
    its tail mass ~= r * 8.5e-4. Last row: 255 * 8.5e-4 ~= 0.217
    (target informative band 0.1-0.3), output ~= 0.217/1.217 ~= 0.178/dim.

Discrimination (proven by A/B on the bias constant, see the agent report):
  - WITH the 256x scale (p_fp8_bias = 8.0): P_tail is lifted to
    256 * 8.5e-4 ~= 0.217 before the fp8 cast -> the tail SURVIVES and the
    fp8 output tracks the bf16 reference (cosine clears the bar).
  - WITHOUT it (p_fp8_bias = 0.0): every tail prob casts to fp8 ZERO ->
    O collapses toward 0 -> cosine CRATERS and this test FAILS.

A8/g0=10/peak-V=0 are all exact in e4m3 (8 = 2^3, 10, 0) so the fp8 inputs
and the bf16 reference inputs are bit-identical; the ONLY difference is the
intermediate P quantization the 256x scale protects.

Target hardware family: NVIDIA SM100 (B200). At d=128 the dispatch routes
to `SM100MHA2Q` (the FA4 2Q prefill kernel).
"""

from std.math import sqrt, exp

from max.gpu.host import DeviceContext
from layout import (
    Idx,
    TileTensor,
    row_major,
)

from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalMask, MHAMask

from std.testing import assert_true


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
    """Cast fp8 -> bf16 element-by-element on the host. Lossless: every
    fp8 e4m3 value is exactly representable in bf16."""
    for i in range(size):
        dst[i] = src[i].cast[bf16_t]()


# ===-----------------------------------------------------------------------===#
# Core test
# ===-----------------------------------------------------------------------===#


def execute_tail_stress_test[
    MaskType: MHAMask,
    *,
    num_q_heads: Int,
    group: Int,
    seq_len: Int,
    num_keys: Int,
    mask_name: StaticString,
    peak_q: Float32,
    peak_k: Float32,
    cos_bar: Float64,
](mask: MaskType, ctx: DeviceContext,) raises:
    """Run the engineered tail-stress fp8 MHA vs bf16 reference."""
    comptime head_dim = 128
    comptime kv_num_heads = num_q_heads // group
    comptime batch_size = 1
    comptime scale = Float32(1.0) / sqrt(Float32(head_dim))

    # Predicted unnormalized tail probability per tail key (peak P = 1).
    var score_peak = peak_q * peak_k * scale
    var p_tail_unnorm = exp(-score_peak)
    # Last causal row attends keys [0 .. num_keys-1]; (num_keys-1) tail keys.
    var last_row_tail_mass = Float32(num_keys - 1) * p_tail_unnorm

    print(
        "test_mha_sm100_qkv_fp8_d128_tail_stress: ",
        "mask=",
        mask_name,
        " group=",
        group,
        " n_q_heads=",
        num_q_heads,
        " seq_len=",
        seq_len,
        " num_keys=",
        num_keys,
    )
    print(
        "  score_peak(log)=",
        score_peak,
        " P_tail(unnorm)=",
        p_tail_unnorm,
        " last_row_tail_mass=",
        last_row_tail_mass,
    )

    comptime fp8_dtype = DType.float8_e4m3fn
    comptime bf16_dtype = DType.bfloat16

    var q_size = batch_size * seq_len * num_q_heads * head_dim
    var k_size = batch_size * num_keys * kv_num_heads * head_dim
    var v_size = k_size
    var o_size = q_size

    # ---- Host: ENGINEER Q, K, V as fp8 directly (no randn) ----
    var q_fp8_host = ctx.enqueue_create_host_buffer[fp8_dtype](q_size)
    var k_fp8_host = ctx.enqueue_create_host_buffer[fp8_dtype](k_size)
    var v_fp8_host = ctx.enqueue_create_host_buffer[fp8_dtype](v_size)

    var qp = q_fp8_host.unsafe_ptr()
    var kp = k_fp8_host.unsafe_ptr()
    var vp = v_fp8_host.unsafe_ptr()

    comptime zero = Scalar[fp8_dtype](0)
    comptime one = Scalar[fp8_dtype](1)
    var q_peak = peak_q.cast[fp8_dtype]()
    var k_peak = peak_k.cast[fp8_dtype]()

    # Zero everything, then set the controlled non-zero entries.
    for i in range(q_size):
        qp[i] = zero
    for i in range(k_size):
        kp[i] = zero
    for i in range(v_size):
        vp[i] = zero

    # Q[b, s, h, dim0] = peak_q for every (s, h); all other dims 0.
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_q_heads):
                var base = head_dim * (h + num_q_heads * (s + seq_len * b))
                qp[base] = q_peak

    # K[b, key0, h, dim0] = peak_k (PEAK key); K[b, key j>0, :] stays 0
    # (TAIL keys -> score 0). V[b, key0, :] = 0 (peak contributes nothing
    # to the output); V[b, key j>0, :] = 1.0 (the informative tail).
    for b in range(batch_size):
        for h in range(kv_num_heads):
            # Peak key index 0.
            var k0_base = head_dim * (h + kv_num_heads * (0 + num_keys * b))
            kp[k0_base] = k_peak
            # V[peak] already 0 from the zero-fill above.
            # Tail keys j = 1 .. num_keys-1.
            for j in range(1, num_keys):
                var v_base = head_dim * (h + kv_num_heads * (j + num_keys * b))
                for d in range(head_dim):
                    vp[v_base + d] = one

    # ---- Host: cast fp8 -> bf16 (lossless) for the reference inputs ----
    var q_bf16_host = ctx.enqueue_create_host_buffer[bf16_dtype](q_size)
    var k_bf16_host = ctx.enqueue_create_host_buffer[bf16_dtype](k_size)
    var v_bf16_host = ctx.enqueue_create_host_buffer[bf16_dtype](v_size)
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

    var total_abs_diff: Float64 = 0.0
    var max_abs_diff: Float64 = 0.0
    var num_compared = 0
    # Cosine similarity over the whole output (vs bf16 reference).
    var dot: Float64 = 0.0
    var aa: Float64 = 0.0
    var bb: Float64 = 0.0
    # Track the bf16-reference magnitude on the last query row to confirm
    # the tail actually carries informative output mass (the input is only
    # stressing the lever if the reference O is non-trivial here).
    var last_row_ref_mag: Float64 = 0.0
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
                    var diff = abs(actual - expect)
                    total_abs_diff += diff
                    if diff > max_abs_diff:
                        max_abs_diff = diff
                    num_compared += 1
                    dot += actual * expect
                    aa += actual * actual
                    bb += expect * expect
                    if s == seq_len - 1 and h == 0:
                        last_row_ref_mag += abs(expect)

    var cos: Float64 = 0.0
    if aa > 0.0 and bb > 0.0:
        cos = dot / (sqrt(aa) * sqrt(bb))
    print(
        "  mean_abs_diff=",
        total_abs_diff / Float64(num_compared),
        " max_abs_diff=",
        max_abs_diff,
        " cosine=",
        cos,
    )
    print(
        "  last_row(h=0) mean |ref O| =",
        last_row_ref_mag / Float64(head_dim),
        " (informative if ~0.1-0.3)",
    )

    # The bf16 reference output on the stressed rows must be non-trivial,
    # otherwise the input is not exercising the underflow tail and the
    # test cannot discriminate the lever.
    assert_true(
        last_row_ref_mag / Float64(head_dim) > 0.05,
        (
            "tail mass too small: reference output is ~0, input not stressing"
            " the underflow tail (retune peak_q / peak_k / num_keys)"
        ),
    )

    # Tight cosine gate. With the 256x P-scale the fp8 output tracks the
    # bf16 reference here; without it (p_fp8_bias = 0.0) the tail
    # underflows to zero and cosine craters well below this bar.
    assert_true(
        cos >= cos_bar,
        "cosine below tail-stress bar (the 256x P-scale must be active)",
    )

    print("  PASSED")


# ===-----------------------------------------------------------------------===#
# Entry point
# ===-----------------------------------------------------------------------===#


def main() raises:
    with DeviceContext() as ctx:
        var causal = CausalMask()

        # d=128 2Q path. A=8, g0=10 => score_peak = 80/sqrt(128) = 7.07,
        # P_tail ~= 8.5e-4 (e4m3 underflow zone). seq_len = num_keys = 256
        # => last-row tail mass ~= 0.22 (informative band).
        execute_tail_stress_test[
            CausalMask,
            num_q_heads=8,
            group=4,
            seq_len=256,
            num_keys=256,
            mask_name="CAUSAL_g4_s256_tail",
            peak_q=8.0,
            peak_k=10.0,
            cos_bar=0.999,
        ](causal, ctx)

        # Larger sequence: more tail keys -> larger tail mass, harder for
        # the no-scale path (which drops the whole tail).
        execute_tail_stress_test[
            CausalMask,
            num_q_heads=32,
            group=8,
            seq_len=512,
            num_keys=512,
            mask_name="CAUSAL_g8_s512_tail",
            peak_q=8.0,
            peak_k=10.0,
            cos_bar=0.999,
        ](causal, ctx)

        print("test_mha_sm100_qkv_fp8_d128_tail_stress: ALL PASSED")
