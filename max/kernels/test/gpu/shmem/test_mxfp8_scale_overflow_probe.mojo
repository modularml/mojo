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
"""MXFP8 E8M0 block-scale probe: can a finite BF16 input yield a non-finite
quantized value on AMD CDNA4 (MI355X / gfx950)?

The MXFP8 quantizers derive the block scale as
`(group_max / 448).cast[float8_e8m0fnu]()` and then scale the data by the exact
reciprocal. 448 is E4M3's maxabs, so the product is bounded by 448 only if the
E8M0 cast rounds AWAY from zero; a cast that rounded down could leave the scaled
value at up to ~896, which E4M3 cannot represent.

Three probes, in increasing order of realism:

1. `_cast_probe_kernel` measures on-device what `.cast[float8_e8m0fnu]()` and
   `.cast[float8_e4m3fn]()` actually do — rounding direction and overflow
   behavior — rather than trusting the docstring.
2. `test_quantize_bf16_exhaustive` drives the real `quantize_mx_amd` over EVERY
   positive finite BF16 magnitude as the block max (all 32640 of them), and
   reports the worst observed `|value * out_scale|` against 448.
3. `test_fused_silu_adversarial` drives the real `fused_silu_mx_kernel` (both
   `clamp_activation` variants) over adversarial-but-finite BF16 gate/up pairs,
   including pairs whose fp32 SwiGLU product overflows.

Usage (plain `mojo` is shadowed by the installed shmem package — use bazel):
  ./bazelw test //max/kernels/test/gpu/shmem:test_mxfp8_scale_overflow_probe.mojo.test
"""

from max.gpu.host import DeviceContext, HostBuffer
from max.gpu import global_idx
from max.gpu.host.info import MI355X
from std.math import exp, isfinite, recip
from std.memory import bitcast
from std.testing import assert_true

from layout import Coord, Idx, TileTensor, row_major
from linalg.block_scaled_quantization import quantize_mx_amd
from linalg.fp4_utils import MXFP8_SF_VECTOR_SIZE
from shmem.ep_comm import fused_silu_mx_kernel

comptime E4M3_MAXABS = Float32(448.0)
# E4M3 (float8_e4m3fn) has no infinity: 0x7F / 0xFF are the NaN encodings.
comptime E4M3_NAN_LO = UInt8(0x7F)
comptime E4M3_NAN_HI = UInt8(0xFF)
# E8M0 (float8_e8m0fnu) 0xFF is NaN; 0x00 is 2^-127, not zero.
comptime E8M0_NAN = UInt8(0xFF)


def _e8m0_to_f32(bits: UInt8) -> Float32:
    """Decodes an E8M0 byte exactly as `_convert_float8_ue8m0_to_f32` does."""
    if bits == UInt8(0):
        return bitcast[.float32](UInt32(0x00400000))  # 2^-127, subnormal
    if bits == E8M0_NAN:
        return bitcast[.float32](UInt32(0x7FFFFFFF))  # NaN
    return bitcast[.float32](UInt32(bits) << UInt32(23))


def _f32(bits: UInt32) -> Float32:
    return bitcast[.float32](bits)


# ===----------------------------------------------------------------------=== #
# Probe 1: what do the FP8 casts do on this device?
# ===----------------------------------------------------------------------=== #

comptime NPROBE = 32
comptime probe_layout = row_major[NPROBE]()


def _cast_probe_kernel(
    vals: TileTensor[.float32, type_of(probe_layout), MutAnyOrigin],
    e8m0_out: TileTensor[.float8_e8m0fnu, type_of(probe_layout), MutAnyOrigin],
    e8m0_back: TileTensor[.float32, type_of(probe_layout), MutAnyOrigin],
    e4m3_out: TileTensor[.float8_e4m3fn, type_of(probe_layout), MutAnyOrigin],
    recip_out: TileTensor[.float32, type_of(probe_layout), MutAnyOrigin],
    n: Int32,
):
    var i = global_idx.x
    if i < Int(n):
        var x = rebind[Float32](vals[i])
        var s = x.cast[.float8_e8m0fnu]()
        e8m0_out[i] = rebind[e8m0_out.ElementType](s)
        e8m0_back[i] = rebind[e8m0_back.ElementType](s.cast[.float32]())
        e4m3_out[i] = rebind[e4m3_out.ElementType](x.cast[.float8_e4m3fn]())
        # The production `out_scale`: reciprocal of the decoded E8M0 scale.
        recip_out[i] = rebind[recip_out.ElementType](recip(s.cast[.float32]()))


def test_cast_behavior(ctx: DeviceContext) raises:
    var vals = List[Float32]()
    var names = List[String]()

    # E8M0 rounding direction. `1.0 + 1ulp` is the discriminating case:
    # round-toward-+inf gives 2.0, truncation/round-to-nearest gives 1.0.
    vals.append(_f32(0x3F800000))
    names.append("1.0 (exact 2^0)")
    vals.append(_f32(0x3F800001))
    names.append("1.0 + 1ulp")
    vals.append(Float32(1.25))
    names.append("1.25")
    vals.append(Float32(1.5))
    names.append("1.5")
    vals.append(_f32(0x3FFFFFFF))
    names.append("just below 2.0")
    vals.append(_f32(0x40000000))
    names.append("2.0 (exact 2^1)")
    vals.append(Float32(0.5))
    names.append("0.5 (exact 2^-1)")
    vals.append(_f32(0x3F000001))
    names.append("0.5 + 1ulp")
    vals.append(Float32(0.0))
    names.append("0.0")
    vals.append(_f32(0x00400000))
    names.append("2^-127 (f32 subnormal)")
    vals.append(_f32(0x00000001))
    names.append("min f32 subnormal")
    vals.append(_f32(0x7F7FFFFF))
    names.append("max finite f32")
    # The production expression: group_max / 448 for notable group maxes.
    vals.append(Float32(448.0) / E4M3_MAXABS)
    names.append("448/448")
    vals.append(Float32(449.0) / E4M3_MAXABS)
    names.append("449/448")
    vals.append(Float32(3.3895314e38) / E4M3_MAXABS)
    names.append("bf16max/448")
    vals.append(Float32(1.0e-38) / E4M3_MAXABS)
    names.append("1e-38/448")
    # E4M3 overflow behavior (saturate to 448 vs NaN).
    vals.append(Float32(448.0))
    names.append("448.0")
    vals.append(Float32(449.0))
    names.append("449.0")
    vals.append(Float32(464.0))
    names.append("464.0 (RNE midpoint)")
    vals.append(Float32(464.1))
    names.append("464.1")
    vals.append(Float32(480.0))
    names.append("480.0")
    vals.append(Float32(512.0))
    names.append("512.0")
    vals.append(Float32(896.0))
    names.append("896.0 (2x maxabs)")
    vals.append(Float32(-896.0))
    names.append("-896.0")
    vals.append(Float32(1.0e30))
    names.append("1e30")
    vals.append(_f32(0x7F800000))
    names.append("+inf")
    vals.append(_f32(0xFF800000))
    names.append("-inf")
    vals.append(_f32(0x7FC00000))
    names.append("NaN")

    var vals_d = ctx.enqueue_create_buffer[.float32](NPROBE)
    with vals_d.map_to_host() as h:
        for i in range(NPROBE):
            h[i] = vals[i] if i < len(vals) else Float32(0.0)

    var e8m0_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](NPROBE)
    var back_d = ctx.enqueue_create_buffer[.float32](NPROBE)
    var e4m3_d = ctx.enqueue_create_buffer[.float8_e4m3fn](NPROBE)
    var recip_d = ctx.enqueue_create_buffer[.float32](NPROBE)

    ctx.enqueue_function[_cast_probe_kernel](
        TileTensor[origin=MutAnyOrigin](vals_d, probe_layout),
        TileTensor[origin=MutAnyOrigin](e8m0_d, probe_layout),
        TileTensor[origin=MutAnyOrigin](back_d, probe_layout),
        TileTensor[origin=MutAnyOrigin](e4m3_d, probe_layout),
        TileTensor[origin=MutAnyOrigin](recip_d, probe_layout),
        Int32(NPROBE),
        grid_dim=1,
        block_dim=NPROBE,
    )

    var vals_h = ctx.enqueue_create_host_buffer[.float32](NPROBE)
    var e8m0_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](NPROBE)
    var back_h = ctx.enqueue_create_host_buffer[.float32](NPROBE)
    var e4m3_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](NPROBE)
    var recip_h = ctx.enqueue_create_host_buffer[.float32](NPROBE)
    ctx.enqueue_copy(recip_h, recip_d)
    ctx.enqueue_copy(vals_h, vals_d)
    ctx.enqueue_copy(e8m0_h, e8m0_d)
    ctx.enqueue_copy(back_h, back_d)
    ctx.enqueue_copy(e4m3_h, e4m3_d)
    ctx.synchronize()

    print(
        "  x -> e8m0 byte (value) | recip(e8m0 value) | x -> e4m3 byte (value)"
    )
    for i in range(len(names)):
        var sb = bitcast[.uint8](e8m0_h[i])
        var qb = bitcast[.uint8](e4m3_h[i])
        print(
            "    ",
            names[i],
            " x=",
            vals_h[i],
            " | e8m0=",
            Int(sb),
            " (",
            back_h[i],
            ") | recip=",
            recip_h[i],
            " | e4m3=",
            Int(qb),
            " (",
            e4m3_h[i].cast[.float32](),
            ")",
        )

    # The bound `|value * recip(scale)| <= 448` holds only if the E8M0 cast
    # never lands BELOW the requested scale. Assert the discriminating cases.
    var one_ulp_up = Int(bitcast[.uint8](e8m0_h[1]))
    assert_true(
        one_ulp_up == 128,
        String(
            "e8m0(1.0 + 1ulp) should be 128 (=2.0) under round-toward-+inf;"
            " got "
        )
        + String(one_ulp_up),
    )
    assert_true(
        Int(bitcast[.uint8](e8m0_h[3])) == 128,
        "e8m0(1.5) should round up to 128 (=2.0)",
    )
    assert_true(
        Int(bitcast[.uint8](e8m0_h[0])) == 127,
        "e8m0(1.0) should be exact: 127",
    )


# ===----------------------------------------------------------------------=== #
# Probe 2: quantize_mx_amd over every positive finite BF16 block max.
# ===----------------------------------------------------------------------=== #

# BF16 bit patterns 0x0000..0x7FFF: the non-negative finite values, then
# +inf (0x7F80) and the NaNs. The non-finite tail is deliberate -- a block max
# of inf casts to E8M0 NaN, whose reciprocal is non-finite, and the kernel must
# emit a zeroed block rather than `inf * 0.0 = NaN`.
comptime NUM_BF16_MAGS = 0x8000
comptime SWEEP_K = 2048
comptime SWEEP_GROUPS_PER_ROW = SWEEP_K // MXFP8_SF_VECTOR_SIZE
comptime SWEEP_M = NUM_BF16_MAGS // SWEEP_GROUPS_PER_ROW


def test_quantize_bf16_exhaustive[
    ragged: Bool
](ctx: DeviceContext) raises -> Int:
    """Quantizes 32640 blocks, one per BF16 magnitude, and reports the worst
    `|value * out_scale|` the kernel could have handed to the E4M3 cast.

    `ragged=False` fills every lane with the block max (worst case for the
    E4M3 bound); `ragged=True` keeps one large lane and 31 tiny ones (worst
    case for in-block dynamic range).
    """
    comptime scale_K = SWEEP_K // MXFP8_SF_VECTOR_SIZE
    comptime N = SWEEP_M * SWEEP_K

    var input_d = ctx.enqueue_create_buffer[.bfloat16](N)
    with input_d.map_to_host() as h:
        for g in range(NUM_BF16_MAGS):
            var v = bitcast[.bfloat16](UInt16(g))
            var neg = bitcast[.bfloat16](UInt16(g) | UInt16(0x8000))
            var tiny = (v.cast[.float32]() * Float32(1.0e-9)).cast[
                DType.bfloat16
            ]()
            for j in range(MXFP8_SF_VECTOR_SIZE):
                var idx = g * MXFP8_SF_VECTOR_SIZE + j
                comptime if ragged:
                    h[idx] = v if j == 0 else tiny
                else:
                    h[idx] = neg if j % 2 == 1 else v

    var out_d = ctx.enqueue_create_buffer[.float8_e4m3fn](N)
    var scales_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](SWEEP_M * scale_K)
    var input_tt = TileTensor(input_d, row_major((Idx[SWEEP_M], Idx[SWEEP_K])))
    var out_tt = TileTensor(out_d, row_major((Idx[SWEEP_M], Idx[SWEEP_K])))
    var scales_tt = TileTensor(
        scales_d, row_major((Idx[SWEEP_M], Idx[scale_K]))
    )
    quantize_mx_amd(ctx, out_tt, scales_tt, input_tt)

    var input_h = ctx.enqueue_create_host_buffer[.bfloat16](N)
    var out_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](N)
    var scales_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        SWEEP_M * scale_K
    )
    ctx.enqueue_copy(input_h, input_d)
    ctx.enqueue_copy(out_h, out_d)
    ctx.enqueue_copy(scales_h, scales_d)
    ctx.synchronize()

    var worst_product = Float32(0.0)
    var worst_product_group = 0
    var nan_scales = 0
    var nan_outputs = 0
    # An encoded (scale, value) pair can both be finite yet reconstruct to a
    # value fp32 cannot hold, because E4M3's round-to-nearest pushes the top
    # BF16 octave past 3.4e38. Distinct from the kernel storing a NaN encoding.
    var dequant_overflow = 0
    var min_overflow_orig = Float32(0.0)
    var underflow_clamped = 0
    var max_rel_err = Float32(0.0)
    var max_rel_err_group = 0

    for g in range(NUM_BF16_MAGS):
        var scale_bits = bitcast[.uint8](scales_h[g])
        if scale_bits == E8M0_NAN:
            nan_scales += 1
            continue
        var scale = _e8m0_to_f32(scale_bits)
        var group_max = Float32(0.0)
        for j in range(MXFP8_SF_VECTOR_SIZE):
            group_max = max(
                group_max,
                abs(input_h[g * MXFP8_SF_VECTOR_SIZE + j].cast[.float32]()),
            )
        # `out_scale` is exactly what the kernel applied: recip of the stored
        # power-of-two scale (0 for an all-zero block).
        var out_scale = (
            Float32(0.0) if group_max == Float32(0.0) else Float32(1.0) / scale
        )
        # A block whose true scale falls below E8M0's 2^-127 floor cannot be
        # represented; the data then underflows in E4M3, which is precision
        # loss, not a finiteness bug. Track it separately.
        if group_max != Float32(0.0) and scale_bits == UInt8(0):
            underflow_clamped += 1

        for j in range(MXFP8_SF_VECTOR_SIZE):
            var idx = g * MXFP8_SF_VECTOR_SIZE + j
            var orig = input_h[idx].cast[.float32]()
            var product = abs(orig) * out_scale
            # The `<= 448` bound is a statement about the FINITE domain. A
            # non-finite input has no representable E4M3 image at all, so the
            # contract there is "the kernel emits a zeroed block", which the
            # NaN-output counter below checks instead.
            if isfinite(orig) and product > worst_product:
                worst_product = product
                worst_product_group = g
            var ob = bitcast[.uint8](out_h[idx])
            if ob == E4M3_NAN_LO or ob == E4M3_NAN_HI:
                nan_outputs += 1
                if nan_outputs <= 4:
                    print(
                        "      NaN E4M3 at bf16-block ",
                        g,
                        " lane ",
                        j,
                        ": input=",
                        orig,
                        " block_max=",
                        group_max,
                        " e8m0=",
                        Int(scale_bits),
                        " (scale=",
                        scale,
                        ") out_scale=",
                        out_scale,
                        " out_byte=",
                        Int(ob),
                    )
                continue
            var dequant = out_h[idx].cast[.float32]() * scale
            if not isfinite(dequant):
                dequant_overflow += 1
                if (
                    min_overflow_orig == Float32(0.0)
                    or abs(orig) < min_overflow_orig
                ):
                    min_overflow_orig = abs(orig)
                continue
            # Only gate blocks that E8M0 can actually represent, and only the
            # lanes at the block max (a ragged lane's error is dynamic range,
            # not a scale bug).
            if (
                scale_bits != UInt8(0)
                and orig != Float32(0.0)
                and abs(orig) == group_max
            ):
                var rel = abs(dequant - orig) / abs(orig)
                if rel > max_rel_err:
                    max_rel_err = rel
                    max_rel_err_group = g

    print(
        "    blocks=",
        NUM_BF16_MAGS,
        " ragged=",
        ragged,
        "\n     worst |value * out_scale| = ",
        worst_product,
        " (vs E4M3 maxabs 448) at bf16 bit pattern ",
        worst_product_group,
        "\n     NaN scales=",
        nan_scales,
        " NaN outputs=",
        nan_outputs,
        "\n     fp32-reconstruction overflows=",
        dequant_overflow,
        " (smallest input magnitude involved: ",
        min_overflow_orig,
        ")\n     E8M0-floor-clamped blocks=",
        underflow_clamped,
        " max rel err at block max=",
        max_rel_err,
        " at bf16 bit pattern ",
        max_rel_err_group,
    )

    assert_true(nan_scales == 0, "quantize_mx_amd emitted a NaN E8M0 scale")
    # Reconstruction may overflow fp32 only for inputs already at the very top
    # of the BF16 range; anything lower would mean the scale itself is wrong.
    assert_true(
        dequant_overflow == 0 or min_overflow_orig > Float32(1.0e38),
        String("fp32 reconstruction overflowed for an input as small as ")
        + String(min_overflow_orig),
    )
    assert_true(
        worst_product <= E4M3_MAXABS,
        String("scaled value ")
        + String(worst_product)
        + " exceeds E4M3 maxabs 448",
    )
    assert_true(
        max_rel_err < Float32(0.07),
        String("block-max relative error ")
        + String(max_rel_err)
        + " exceeds one E4M3 step (clipping?)",
    )
    return nan_outputs


# ===----------------------------------------------------------------------=== #
# Probe 3: fused_silu_mx_kernel over adversarial finite gate/up pairs.
# ===----------------------------------------------------------------------=== #

comptime BF16_MAX = Float32(3.3895314e38)
comptime NUM_PATTERNS = 10
comptime SILU_HIDDEN = 256
comptime SILU_TOKENS = 4
comptime SILU_GROUPS = (SILU_TOKENS * SILU_HIDDEN) // MXFP8_SF_VECTOR_SIZE


def _pattern_name(p: Int) -> String:
    if p == 0:
        return "all-zero"
    if p == 1:
        return "near-denormal 1e-38"
    if p == 2:
        return "gate=up=bf16max"
    if p == 3:
        return "gate=up=1e20"
    if p == 4:
        return "gate=1e20 up=1e-20"
    if p == 5:
        return "gate=-bf16max up=bf16max"
    if p == 6:
        return "one bf16max lane, rest 1e-3"
    if p == 7:
        return "product just above 448"
    if p == 8:
        return "product just below 448"
    return "all-zero + one bf16 denormal"


def _fill_pattern(
    mut h: HostBuffer[.bfloat16],
    token: Int,
    group: Int,
    p: Int,
):
    comptime input_dim = SILU_HIDDEN * 2
    var bf16_denorm = bitcast[.bfloat16](UInt16(1))
    for j in range(MXFP8_SF_VECTOR_SIZE):
        var k = group * MXFP8_SF_VECTOR_SIZE + j
        var gi = token * input_dim + k
        var ui = token * input_dim + SILU_HIDDEN + k
        var gv = Float32(0.0)
        var uv = Float32(0.0)
        if p == 1:
            gv = Float32(1.0e-38)
            uv = Float32(1.0e-38)
        elif p == 2:
            gv = BF16_MAX
            uv = BF16_MAX
        elif p == 3:
            gv = Float32(1.0e20)
            uv = Float32(1.0e20)
        elif p == 4:
            gv = Float32(1.0e20)
            uv = Float32(1.0e-20)
        elif p == 5:
            gv = -BF16_MAX
            uv = BF16_MAX
        elif p == 6:
            gv = BF16_MAX if j == 0 else Float32(1.0e-3)
            uv = Float32(1.0) if j == 0 else Float32(1.0e-3)
        elif p == 7:
            # sigmoid(64) rounds to 1.0 in fp32, so silu(64) == 64 exactly.
            gv = Float32(64.0)
            uv = Float32(7.03125)  # 64 * 7.03125 = 450 > 448
        elif p == 8:
            gv = Float32(64.0)
            uv = Float32(6.96875)  # 64 * 6.96875 = 446 < 448
        elif p == 9:
            if j == 0:
                h[gi] = bf16_denorm
                h[ui] = bf16_denorm
            else:
                h[gi] = BFloat16(0)
                h[ui] = BFloat16(0)
            continue
        h[gi] = gv.cast[.bfloat16]()
        h[ui] = uv.cast[.bfloat16]()


def test_fused_silu_adversarial[
    clamp_activation: Bool
](ctx: DeviceContext, alpha: Float32 = 0.0, limit: Float32 = 0.0) raises -> Int:
    comptime input_dim = SILU_HIDDEN * 2
    comptime output_dim = SILU_HIDDEN
    comptime scale_K = SILU_HIDDEN // MXFP8_SF_VECTOR_SIZE
    comptime n_off = 2
    comptime hw = ctx.default_device_info

    var input_h = ctx.enqueue_create_host_buffer[.bfloat16](
        SILU_TOKENS * input_dim
    )
    var off_h = ctx.enqueue_create_host_buffer[.uint32](n_off)
    ctx.synchronize()
    for t in range(SILU_TOKENS):
        for g in range(scale_K):
            _fill_pattern(input_h, t, g, (t * scale_K + g) % NUM_PATTERNS)
    off_h[0] = UInt32(0)
    off_h[1] = UInt32(SILU_TOKENS)

    var input_d = ctx.enqueue_create_buffer[.bfloat16](SILU_TOKENS * input_dim)
    var off_d = ctx.enqueue_create_buffer[.uint32](n_off)
    ctx.enqueue_copy(input_d, input_h)
    ctx.enqueue_copy(off_d, off_h)

    var input_tt = TileTensor[origin=ImmutAnyOrigin](
        input_d, row_major(Coord(SILU_TOKENS, Idx[input_dim]))
    )
    var off_tt = TileTensor[origin=ImmutAnyOrigin](off_d, row_major[n_off]())

    comptime out_layout = type_of(
        TileTensor[origin=MutAnyOrigin](
            input_d, row_major(Coord(SILU_TOKENS, Idx[output_dim]))
        )
    ).LayoutType
    comptime scales_layout = type_of(
        TileTensor[origin=MutAnyOrigin](
            input_d, row_major(Coord(SILU_TOKENS, Idx[scale_K]))
        )
    ).LayoutType

    comptime kernel = fused_silu_mx_kernel[
        DType.float8_e4m3fn,
        DType.float8_e8m0fnu,
        DType.bfloat16,
        out_layout,
        scales_layout,
        input_tt.LayoutType,
        off_tt.LayoutType,
        hw.max_thread_block_size,
        hw.sm_count,
        fuse_a_scale_preshuffle=False,
        clamp_activation=clamp_activation,
    ]

    var out_d = ctx.enqueue_create_buffer[.float8_e4m3fn](
        SILU_TOKENS * output_dim
    )
    var scales_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        SILU_TOKENS * scale_K
    )
    out_d.enqueue_fill(Float8_e4m3fn(0))
    scales_d.enqueue_fill(Float8_e8m0fnu(0))

    ctx.enqueue_function[kernel](
        TileTensor[origin=MutAnyOrigin](
            out_d, row_major(Coord(SILU_TOKENS, Idx[output_dim]))
        ),
        TileTensor[origin=MutAnyOrigin](
            scales_d, row_major(Coord(SILU_TOKENS, Idx[scale_K]))
        ),
        input_tt,
        off_tt,
        Int32(0),
        alpha,
        limit,
        grid_dim=hw.sm_count,
        block_dim=hw.max_thread_block_size,
    )

    var out_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        SILU_TOKENS * output_dim
    )
    var scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        SILU_TOKENS * scale_K
    )
    ctx.enqueue_copy(out_h, out_d)
    ctx.enqueue_copy(scales_host, scales_d)
    ctx.synchronize()

    var violations = 0
    print("    clamp_activation=", clamp_activation)
    for t in range(SILU_TOKENS):
        for g in range(scale_K):
            var p = (t * scale_K + g) % NUM_PATTERNS
            var scale_bits = bitcast[.uint8](scales_host[t * scale_K + g])
            var scale = _e8m0_to_f32(scale_bits)

            # Host reference for the same activation the kernel computes.
            var ref_max = Float32(0.0)
            var ref_nonfinite = False
            for j in range(MXFP8_SF_VECTOR_SIZE):
                var k = g * MXFP8_SF_VECTOR_SIZE + j
                var gp = input_h[t * input_dim + k].cast[.float32]()
                var up = input_h[t * input_dim + SILU_HIDDEN + k].cast[
                    DType.float32
                ]()
                var z: Float32
                comptime if clamp_activation:
                    var g_c = min(gp, limit)
                    var u_c = max(min(up, limit), -limit)
                    z = (g_c / (Float32(1.0) + exp(-(g_c * alpha)))) * (
                        u_c + Float32(1.0)
                    )
                else:
                    z = (gp / (Float32(1.0) + exp(-gp))) * up
                if not isfinite(z):
                    ref_nonfinite = True
                elif abs(z) > ref_max:
                    ref_max = abs(z)

            var bad_scale = scale_bits == E8M0_NAN
            var bad_out = 0
            var recon_overflow = 0
            var worst_dequant = Float32(0.0)
            for j in range(MXFP8_SF_VECTOR_SIZE):
                var idx = t * output_dim + g * MXFP8_SF_VECTOR_SIZE + j
                var ob = bitcast[.uint8](out_h[idx])
                if ob == E4M3_NAN_LO or ob == E4M3_NAN_HI:
                    bad_out += 1
                    continue
                var dq = out_h[idx].cast[.float32]() * scale
                if not isfinite(dq):
                    # Encoding is finite but exceeds fp32 range; only reachable
                    # for inputs already at the top of the BF16 range.
                    recon_overflow += 1
                elif abs(dq) > worst_dequant:
                    worst_dequant = abs(dq)

            if bad_scale or bad_out > 0:
                violations += 1
                for j in range(MXFP8_SF_VECTOR_SIZE):
                    var k = g * MXFP8_SF_VECTOR_SIZE + j
                    var ob2 = bitcast[.uint8](out_h[t * output_dim + k])
                    if ob2 != E4M3_NAN_LO and ob2 != E4M3_NAN_HI:
                        continue
                    print(
                        "        nan lane ",
                        j,
                        " gate=",
                        input_h[t * input_dim + k].cast[.float32](),
                        " up=",
                        input_h[t * input_dim + SILU_HIDDEN + k].cast[
                            DType.float32
                        ](),
                        " out_byte=",
                        Int(ob2),
                    )
                print(
                    "      VIOLATION token=",
                    t,
                    " block=",
                    g,
                    " pattern='",
                    _pattern_name(p),
                    "' e8m0=",
                    Int(scale_bits),
                    " nan_lanes=",
                    bad_out,
                    " fp32_recon_overflow_lanes=",
                    recon_overflow,
                    " host_ref_nonfinite=",
                    ref_nonfinite,
                    " host_ref_max=",
                    ref_max,
                )
            elif t == 0:
                print(
                    "      ok block=",
                    g,
                    " pattern='",
                    _pattern_name(p),
                    "' e8m0=",
                    Int(scale_bits),
                    " ref_max=",
                    ref_max,
                    " ref_nonfinite=",
                    ref_nonfinite,
                    " max|dequant|=",
                    worst_dequant,
                )
    return violations


def main() raises:
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "test_mxfp8_scale_overflow_probe currently requires MI355X"

    print("===> probe 1: on-device FP8 cast behavior")
    test_cast_behavior(ctx)

    print("===> probe 2: quantize_mx_amd over every finite BF16 block max")
    var nan_dense = test_quantize_bf16_exhaustive[ragged=False](ctx)
    var nan_ragged = test_quantize_bf16_exhaustive[ragged=True](ctx)

    print("===> probe 3: fused_silu_mx_kernel, adversarial finite BF16")
    var v_plain = test_fused_silu_adversarial[clamp_activation=False](ctx)
    var v_clamped = test_fused_silu_adversarial[clamp_activation=True](
        ctx, Float32(1.702), Float32(7.0)
    )
    print(
        "    non-finite blocks: plain SiLU=",
        v_plain,
        " clamped SwiGLU=",
        v_clamped,
    )
    assert_true(
        nan_dense == 0 and nan_ragged == 0,
        String("quantize_mx_amd emitted NaN E4M3 values: dense=")
        + String(nan_dense)
        + " ragged="
        + String(nan_ragged),
    )
    assert_true(
        v_plain == 0,
        String(v_plain)
        + " blocks went non-finite in plain-SiLU fused_silu_mx_kernel",
    )
    assert_true(
        v_clamped == 0,
        String(v_clamped)
        + " blocks went non-finite in clamped fused_silu_mx_kernel",
    )
    print("PASS")
