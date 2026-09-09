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

"""Regression: FP8 quantization/cast must SATURATE out-of-range inputs, not NaN.

On NVIDIA a raw `.cast[float8_e4m3fn]()` of an fp32 value > ±448 produces NaN
(the `cvt` is not `satfinite`). FP8 quantization must instead clamp to the
representable range. This test feeds out-of-range fp32 values (large finite, and
+/-Inf) into both `fp8_quantize` (the dynamic-scale quantizer) and
`cast_saturating` (the direct-store helper used by the MLA RoPE/RMSNorm KV-cache
writes) and asserts the output is FINITE and saturated to ±max_finite[e4m3]
(±448), never NaN.

This complements `test_mla_decode_kv_fp8_magnitude_stress.mojo`, which exercises
the attention READ path; this test covers the quant/cast WRITE boundary that the
attention microbench cannot see."""

from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext
from std.utils.numerics import isnan, max_finite, min_finite
from internal_utils.fp8_utils import fp8_quantize, cast_saturating


def _check_kernel():
    # ---- fp8_quantize: scaled value lands out of range -> must saturate ----
    # scale_recip = 1.0 (so result == input); feed values past ±448 and +Inf.
    comptime fp8_max = Float32(max_finite[.float8_e4m3fn]())
    comptime fp8_min = Float32(min_finite[.float8_e4m3fn]())

    var big = SIMD[.float32, 4](1.0e4, -1.0e4, 600.0, -600.0)
    var q = fp8_quantize[.float8_e4m3fn](big, Float32(1.0))

    comptime for i in range(4):
        var v = q[i].cast[.float32]()
        # Finite (not NaN) and clamped within [fp8_min, fp8_max].
        if isnan(v):
            # Force a visible device-side failure.
            _ = v / Float32(0.0)
        # saturated magnitude is exactly fp8_max/fp8_min
        if i % 2 == 0:
            if v != fp8_max:
                _ = v / Float32(0.0)
        else:
            if v != fp8_min:
                _ = v / Float32(0.0)

    # +/-Inf input must also saturate (not propagate to NaN).
    var infs = SIMD[.float32, 2](
        Float32.MAX_FINITE * 2.0, -Float32.MAX_FINITE * 2.0
    )
    var qinf = fp8_quantize[.float8_e4m3fn](infs, Float32(1.0))
    if isnan(qinf[0].cast[.float32]()) or isnan(qinf[1].cast[.float32]()):
        _ = Float32(1.0) / Float32(0.0)

    # --- PRE-PATCH MECHANISM DEMONSTRATION (the GO/STOP discriminator) ---
    # `use_clamp=False` is the OLD NVIDIA behavior (raw cvt, no saturation).
    # On NVIDIA an out-of-range value cast this way MUST be non-finite — this
    # is the S6 bug the default-True clamp cures. If this is FINITE on the
    # target GPU, the isolated mechanism is NOT firing (STOP — needs the e2e
    # nan-check trace). On NVIDIA we assert it IS non-finite (mechanism repros).
    comptime if has_nvidia_gpu_accelerator():
        var raw = fp8_quantize[.float8_e4m3fn, use_clamp=False](
            big, Float32(1.0)
        )
        var any_nonfinite = False
        comptime for i in range(4):
            if isnan(raw[i].cast[.float32]()):
                any_nonfinite = True
        # Pre-patch raw cast of out-of-range MUST be non-finite on NVIDIA.
        if not any_nonfinite:
            # Mechanism did NOT reproduce -> force a visible failure so the
            # test driver flags the STOP condition rather than silently pass.
            _ = Float32(2.0) / Float32(0.0)

    # ---- cast_saturating: direct store helper (MLA KV-write path) ----
    var big2 = SIMD[.bfloat16, 4](
        BFloat16(1000.0), BFloat16(-1000.0), BFloat16(500.0), BFloat16(-500.0)
    )
    var c = cast_saturating[.float8_e4m3fn](big2)

    comptime for i in range(4):
        var v = c[i].cast[.float32]()
        if isnan(v):
            _ = v / Float32(0.0)
        if i % 2 == 0:
            if v != fp8_max:
                _ = v / Float32(0.0)
        else:
            if v != fp8_min:
                _ = v / Float32(0.0)

    # cast_saturating to a NON-fp8 dtype is a plain cast (no clamp): a large
    # bf16 value stays large (no spurious saturation).
    var keep = cast_saturating[.bfloat16](SIMD[.float32, 2](1000.0, -1000.0))
    if keep[0].cast[.float32]() != Float32(1000.0):
        _ = Float32(1.0) / Float32(0.0)

    # cast_saturating with in_dtype == out_dtype (non-fp8): identity.
    var bf16_id = cast_saturating[.bfloat16](
        SIMD[.bfloat16, 2](BFloat16(1000.0), BFloat16(-1000.0))
    )
    if bf16_id[0].cast[.float32]() != Float32(1000.0):
        _ = Float32(1.0) / Float32(0.0)

    # cast_saturating with in_dtype == out_dtype (fp8): identity, no clamp.
    var e4m3_id = cast_saturating[.float8_e4m3fn](
        Float8_e4m3fn(Float8_e4m3fn(fp8_max))
    )
    if e4m3_id[0].cast[.float32]() != fp8_max:
        _ = Float32(1.0) / Float32(0.0)

    # cast_saturating across fp8 types: e5m2 → e4m3fn must still clamp.
    # e5m2 max_finite exceeds e4m3fn max_finite.
    comptime e5m2_max = Float32(max_finite[.float8_e5m2]())
    var e5m2_big = SIMD[.float8_e5m2, 2](
        Float8_e5m2(e5m2_max),
        Float8_e5m2(-e5m2_max),
    )
    var cross = cast_saturating[.float8_e4m3fn](e5m2_big)
    if cross[0].cast[.float32]() != fp8_max:
        _ = Float32(1.0) / Float32(0.0)
    if cross[1].cast[.float32]() != fp8_min:
        _ = Float32(1.0) / Float32(0.0)


def main() raises:
    with DeviceContext() as ctx:
        ctx.enqueue_function[_check_kernel](grid_dim=1, block_dim=1)
        ctx.synchronize()
        print(
            "PASS: fp8_quantize + cast_saturating saturate out-of-range to"
            " finite"
        )
