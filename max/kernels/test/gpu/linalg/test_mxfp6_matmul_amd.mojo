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
"""Correctness tests for the native MXFP6 block-scaled matmul on CDNA4.

Ground truth is a float64 host reference; the MXFP4 kernel is deliberately not
used as an oracle (it returns zeros for rows >= 64 at M = 96, so it would make
these tests hostage to a bug elsewhere).

What each axis is here to catch:

- **Data patterns.** `SWEEP` guarantees all 64 codes appear, so a bit-packing
  error confined to high-mantissa codes cannot hide. `MAX` puts every element
  at the format extreme, which maximizes accumulator magnitude. `FP4_SUBSET`
  restricts operands to values that make the whole dot product exactly
  representable in float32, which buys a **bit-exact** assertion (see below).
- **Scale patterns.** `UNIT` cannot detect a wrong scale byte or a wrong
  K-group, because every wrong answer is also 1.0. `VARY` gives each row and
  each 32-element K-block a distinct exponent, which is what actually exercises
  the MFMA's OP_SEL byte selector and the per-BK scale indexing. `EXTREME`
  spreads exponents over 2^-10..2^10 to stress accumulation across magnitudes.
- **K.** `K = 128` is a single BK iteration and takes the kernel's simple-loop
  fallback; `K >= 256` takes the schedule-driven pipeline. Both paths must be
  covered, and K = 384 additionally gives an odd BK-tile count.
- **M.** Always a runtime dimension, matching production, where M is the
  batch / token count. That also means the whole M sweep shares one kernel
  instantiation, so it can be dense: BM is 96 and the MFMA tile is 16 rows,
  so the sweep sits on both sides of every such boundary.
- **N.** Must be a multiple of MMA_N = 16 (the entry point asserts it; a
  partial column tile corrupts neighbouring rows). 16, 80 and 112 are not
  multiples of BN = 64, which shows the binding constraint is MMA_N.
- **Output dtype.** float32 is the accumulator type; bfloat16 exercises the
  cast in the store epilogue.

Tolerance: the kernel accumulates in float32, so the error is bounded by the
accumulated magnitude, not the (possibly cancelling) result. Comparisons are
therefore relative to `sum |a_k * b_k|` rather than to `|result|`. The
`FP4_SUBSET` + `UNIT` + E2M3 combination is checked bit-exactly instead, as a
canary that the tolerance elsewhere is not hiding a real defect: products are
multiples of 0.25 bounded by 36, so a K-term sum needs at most 17 mantissa bits
at K = 512 and float32 holds it without rounding.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.random import random_ui64, seed
from max.gpu.host import DeviceContext
from max.gpu.host.info import MI355X
from std.testing import assert_true
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from std.utils import StaticTuple
from layout import Coord, Idx, TileTensor
from layout.tile_layout import row_major

from linalg.arch.amd.block_scaled_mma import CDNA4F8F6F4MatrixFormat
from linalg.matmul.gpu.amd import Shuffler
from linalg.fp6_utils import (
    FP6Format,
    decode_fp6_to_f32,
    fp6_reference_table,
    unpack_fp6_x32,
)
from linalg.fp6_quantization import quantize_mxfp6_amd
from linalg.matmul.gpu.amd.block_scaled_matmul_amd import (
    mxfp6_block_scaled_matmul_amd,
)

comptime DATA_RANDOM = 0
comptime DATA_SWEEP = 1
comptime DATA_ZERO = 2
comptime DATA_MAX = 3
comptime DATA_FP4_SUBSET = 4

comptime SCALE_UNIT = 0
comptime SCALE_VARY = 1
comptime SCALE_EXTREME = 2

comptime SENTINEL = Float32(-98765.0)


def _fill_codes(
    codes: MutPointer[UInt8, _],
    count: Int,
    pattern: Int,
    salt: Int,
):
    """Fills a code buffer according to `pattern`."""
    for i in range(count):
        var code: Int
        if pattern == DATA_RANDOM:
            code = Int(random_ui64(0, 63))
        elif pattern == DATA_SWEEP:
            # Every code appears, and the stride keeps neighbouring elements
            # from sharing a value so a swapped pair is visible.
            code = (i * 37 + salt) % 64
        elif pattern == DATA_ZERO:
            code = 0
        elif pattern == DATA_MAX:
            # Alternate the largest positive and largest negative magnitude.
            code = 31 if (i + salt) % 2 == 0 else 63
        else:
            # FP4-representable magnitudes land on every fourth E2M3 code.
            var magnitude = ((i * 5 + salt) % 8) * 4
            code = magnitude + (32 if (i + salt) % 3 == 0 else 0)
        codes[unsafe_offset=i] = UInt8(code)


def _fill_scales(
    scales: MutPointer[UInt8, _],
    rows: Int,
    scale_cols: Int,
    pattern: Int,
    salt: Int,
):
    """Fills an E8M0 scale buffer; byte `b` denotes `2^(b - 127)`."""
    for row in range(rows):
        for col in range(scale_cols):
            var exponent: Int
            if pattern == SCALE_UNIT:
                exponent = 127
            elif pattern == SCALE_VARY:
                # Distinct per row and per K-block so a wrong OP_SEL byte or a
                # wrong K-group lands on a different exponent.
                exponent = 127 + ((row + 2 * col + salt) % 5) - 2
            else:
                exponent = 127 + (10 if (row + col + salt) % 2 == 0 else -10)
            scales[unsafe_offset=row * scale_cols + col] = UInt8(exponent)


def _pack_fp6(
    codes: ImmPointer[UInt8, _],
    packed: MutPointer[UInt8, _],
    rows: Int,
    K: Int,
):
    """Packs 6-bit codes little-endian: element `i` at bits `[6i+5 : 6i]`."""
    var k_bytes = (K * 6) // 8
    for row in range(rows):
        for i in range(k_bytes):
            packed[unsafe_offset=row * k_bytes + i] = UInt8(0)
        for col in range(K):
            var code = Int(codes[unsafe_offset=row * K + col])
            var bit = col * 6
            var byte = row * k_bytes + bit // 8
            var shift = bit % 8
            packed[unsafe_offset=byte] |= UInt8((code << shift) & 0xFF)
            # A code straddles into the next byte once it starts past bit 2.
            if shift > 2:
                packed[unsafe_offset=byte + 1] |= UInt8(code >> (8 - shift))


@always_inline
def _e8m0(bits: UInt8) -> Float64:
    """Decodes an E8M0 scale byte to `2^(bits - 127)` in float64."""
    var exponent = Int(bits) - 127
    var value = Float64(1.0)
    if exponent >= 0:
        for _ in range(exponent):
            value *= 2.0
    else:
        for _ in range(-exponent):
            value *= 0.5
    return value


def _fill_packed_random(
    packed: MutPointer[UInt8, _],
    nbytes: Int,
    salt: Int,
):
    """Fills a packed FP6 stream with pseudo-random bytes.

    Every 6-bit pattern is a valid FP6 code -- there are no reserved or NaN
    encodings -- so random bytes *are* random codes. Filling the packed buffer
    directly avoids materializing an unpacked code array, which matters at
    production sizes: B at N=18432, K=7168 is 99 MB packed but 132 MB unpacked.
    """
    var state = UInt64(salt * 2654435761 + 12345)
    for i in range(nbytes):
        state = state * 6364136223846793005 + 1442695040888963407
        packed[unsafe_offset=i] = UInt8((state >> 33) & 0xFF)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256))
)
def _mxfp6_matmul_ref[
    fmt: FP6Format
](
    a_ptr: ImmPointer[UInt8, ImmutAnyOrigin],
    b_ptr: ImmPointer[UInt8, ImmutAnyOrigin],
    a_sf_ptr: ImmPointer[Float8_e8m0fnu, ImmutAnyOrigin],
    b_sf_ptr: ImmPointer[Float8_e8m0fnu, ImmutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    mag_ptr: MutPointer[Float32, MutAnyOrigin],
    M_dev: Int32,
    N_dev: Int32,
    K_dev: Int32,
):
    """Per-element GPU reference: one thread per (m, n) output element.

    A GPU reference rather than a host loop because the production shapes below
    are O(M*N*K) -- N=18432, K=7168 would take minutes on the host.

    It decodes with `decode_fp6_to_f32`, which is independent of the path under
    test: the kernel never software-decodes, it hands raw bits to the MFMA and
    the hardware decodes them. `test_fp6_utils` pins this decoder against
    hand-written tables.

    Also emits `sum |a*b|` per element, so the caller can bound float32
    accumulation error against the accumulated magnitude rather than against a
    result that may have cancelled to near zero.
    """
    var M = Int(M_dev)
    var N = Int(N_dev)
    var K = Int(K_dev)

    var m = Int(global_idx.x)
    var n = Int(global_idx.y)
    if m >= M or n >= N:
        return

    var k_groups = K // 32
    var k_bytes = (K * 6) // 8

    var accum = Float32(0)
    var magnitude = Float32(0)

    for ko in range(k_groups):
        var a_scale = a_sf_ptr[unsafe_offset=m * k_groups + ko].cast[.float32]()
        var b_scale = b_sf_ptr[unsafe_offset=n * k_groups + ko].cast[.float32]()

        # 32 elements is one MX block = 24 packed bytes, always 8-byte aligned
        # because k_bytes is a multiple of 24 whenever K is a multiple of 32.
        var a_base = m * k_bytes + ko * 24
        var b_base = n * k_bytes + ko * 24

        var fa = SIMD[.uint8, 32](0)
        var fb = SIMD[.uint8, 32](0)
        comptime for chunk in range(3):
            fa = fa.insert[offset=chunk * 8](
                a_ptr.load[width=8](a_base + chunk * 8)
            )
            fb = fb.insert[offset=chunk * 8](
                b_ptr.load[width=8](b_base + chunk * 8)
            )

        var av = decode_fp6_to_f32[fmt](unpack_fp6_x32(fa)) * a_scale
        var bv = decode_fp6_to_f32[fmt](unpack_fp6_x32(fb)) * b_scale
        var prod = av * bv
        accum += prod.reduce_add()
        magnitude += abs(prod).reduce_add()

    c_ptr[unsafe_offset=m * N + n] = accum
    mag_ptr[unsafe_offset=m * N + n] = magnitude


def run_case[
    fmt: FP6Format,
    N: Int,
    K: Int,
    out_dtype: DType = .float32,
    num_splits: Int = 1,
](
    ctx: DeviceContext, M: Int, data_pattern: Int, scale_pattern: Int
) raises -> Bool:
    comptime assert (
        K % 128 == 0
    ), "K must be a multiple of 128 so the BK tile divides K"
    comptime assert N % 64 == 0, "N must be a multiple of BN = 64"
    comptime K_BYTES = (K * 6) // 8
    comptime K_SCALES = K // 32
    comptime mfma_format = (
        CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3 if fmt
        == FP6Format.E2M3 else CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2
    )

    var a_packed = ctx.enqueue_create_host_buffer[.uint8](M * K_BYTES)
    var b_packed = ctx.enqueue_create_host_buffer[.uint8](N * K_BYTES)

    if data_pattern == DATA_RANDOM:
        # Direct byte fill: no unpacked code array, so production sizes fit.
        _fill_packed_random(a_packed.unsafe_ptr(), M * K_BYTES, 1)
        _fill_packed_random(b_packed.unsafe_ptr(), N * K_BYTES, 2)
    else:
        var a_codes = ctx.enqueue_create_host_buffer[.uint8](M * K)
        var b_codes = ctx.enqueue_create_host_buffer[.uint8](N * K)
        _fill_codes(a_codes.unsafe_ptr(), M * K, data_pattern, 0)
        _fill_codes(b_codes.unsafe_ptr(), N * K, data_pattern, 1)
        _pack_fp6(a_codes.unsafe_ptr(), a_packed.unsafe_ptr(), M, K)
        _pack_fp6(b_codes.unsafe_ptr(), b_packed.unsafe_ptr(), N, K)

    var a_sf = ctx.enqueue_create_host_buffer[.uint8](M * K_SCALES)
    var b_sf = ctx.enqueue_create_host_buffer[.uint8](N * K_SCALES)
    _fill_scales(a_sf.unsafe_ptr(), M, K_SCALES, scale_pattern, 0)
    _fill_scales(b_sf.unsafe_ptr(), N, K_SCALES, scale_pattern, 3)

    var a_sf_typed = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        M * K_SCALES
    )
    var b_sf_typed = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        N * K_SCALES
    )
    for i in range(M * K_SCALES):
        a_sf_typed[i] = bitcast[.float8_e8m0fnu](a_sf[i])
    for i in range(N * K_SCALES):
        b_sf_typed[i] = bitcast[.float8_e8m0fnu](b_sf[i])

    var a_dev = ctx.enqueue_create_buffer[.uint8](M * K_BYTES)
    var b_dev = ctx.enqueue_create_buffer[.uint8](N * K_BYTES)
    var a_sf_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](M * K_SCALES)
    var b_sf_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](N * K_SCALES)
    var c_dev = ctx.enqueue_create_buffer[out_dtype](M * N)
    var ref_dev = ctx.enqueue_create_buffer[.float32](M * N)
    var mag_dev = ctx.enqueue_create_buffer[.float32](M * N)

    # Poison the output so an element the kernel never writes stays
    # distinguishable from one it legitimately computed as zero.
    var poison = ctx.enqueue_create_host_buffer[out_dtype](M * N)
    for i in range(M * N):
        poison[i] = SENTINEL.cast[out_dtype]()

    ctx.enqueue_copy(a_dev, a_packed)
    ctx.enqueue_copy(b_dev, b_packed)
    ctx.enqueue_copy(a_sf_dev, a_sf_typed)
    ctx.enqueue_copy(b_sf_dev, b_sf_typed)
    ctx.enqueue_copy(c_dev, poison)
    ctx.synchronize()

    comptime REF_BLOCK = 16
    ctx.enqueue_function[_mxfp6_matmul_ref[fmt]](
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        a_sf_dev.unsafe_ptr(),
        b_sf_dev.unsafe_ptr(),
        ref_dev.unsafe_ptr(),
        mag_dev.unsafe_ptr(),
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, REF_BLOCK), ceildiv(N, REF_BLOCK)),
        block_dim=(REF_BLOCK, REF_BLOCK),
    )

    mxfp6_block_scaled_matmul_amd[mfma_format, num_splits](
        TileTensor(c_dev, row_major((Int(M), Idx[N]))),
        TileTensor(a_dev, row_major((Int(M), Idx[K_BYTES]))),
        TileTensor(b_dev, row_major[N, K_BYTES]()),
        TileTensor(a_sf_dev, row_major((Int(M), Idx[K_SCALES]))),
        TileTensor(b_sf_dev, row_major[N, K_SCALES]()),
        ctx,
    )
    ctx.synchronize()

    var c_host = ctx.enqueue_create_host_buffer[out_dtype](M * N)
    var ref_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    var mag_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(ref_host, ref_dev)
    ctx.enqueue_copy(mag_host, mag_dev)
    ctx.synchronize()

    # float32 accumulation error grows with the number of summed terms, so the
    # bound is K-scaled and applied to sum |a*b|, not to a result that may have
    # cancelled. bfloat16 output adds its own ~2^-9 rounding. Split-K reduces
    # partials in a different order again, hence the extra headroom.
    comptime ULP_F32 = 5.9604644775390625e-8
    comptime SPLIT_SLACK = 4.0 if num_splits > 1 else 1.0
    var rel_tol = Float64(0.02) if out_dtype == DType.bfloat16 else (
        Float64(16 * K) * ULP_F32 * SPLIT_SLACK
    )

    var mismatches = 0
    var unwritten = 0
    var saw_nonzero = False
    for i in range(M * N):
        var want = Float64(ref_host[i])
        if want != Float64(0.0):
            saw_nonzero = True
        var got = Float64(c_host[i].cast[.float32]())
        if got == Float64(SENTINEL):
            unwritten += 1
        elif abs(got - want) > rel_tol * Float64(mag_host[i]):
            if mismatches < 3:
                print(
                    "      [",
                    i // N,
                    ",",
                    i % N,
                    "] got=",
                    got,
                    " want=",
                    want,
                    " mag=",
                    Float64(mag_host[i]),
                )
            mismatches += 1

    if unwritten > 0 or mismatches > 0:
        print(
            "    FAIL ",
            mismatches,
            " wrong, ",
            unwritten,
            " unwritten, of ",
            M * N,
        )
        return False
    if data_pattern != DATA_ZERO and not saw_nonzero:
        print("    BAD TEST: reference is all zero")
        return False
    return True


def run_case_preb[
    fmt: FP6Format,
    N: Int,
    K: Int,
    out_dtype: DType = DType.float32,
](
    ctx: DeviceContext, M: Int, data_pattern: Int, scale_pattern: Int
) raises -> Bool:
    """Same shapes and reference as `run_case`, but drives
    `mxfp6_block_scaled_matmul_amd[preshuffled_b=True]`: B and its scales are
    preshuffled once (untimed, mirroring the load-time cost
    `preshuffle_block_scaled_b_dense` pays in production) before the call.

    Only valid for `M > DECODE_M_MAX` (64): the dispatcher silently takes the
    row-major decode path for smaller M regardless of `preshuffled_b`, so
    preshuffled bytes there would be read as row-major and corrupt the
    result -- not exercised here.
    """
    comptime assert (
        K % 128 == 0
    ), "K must be a multiple of 128 so the BK tile divides K"
    comptime assert N % 64 == 0, "N must be a multiple of BN = 64"
    comptime K_BYTES = (K * 6) // 8
    comptime K_SCALES = K // 32
    comptime mfma_format = (
        CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3 if fmt
        == FP6Format.E2M3 else CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2
    )

    var a_packed = ctx.enqueue_create_host_buffer[DType.uint8](M * K_BYTES)
    var b_packed = ctx.enqueue_create_host_buffer[DType.uint8](N * K_BYTES)

    if data_pattern == DATA_RANDOM:
        _fill_packed_random(a_packed.unsafe_ptr(), M * K_BYTES, 1)
        _fill_packed_random(b_packed.unsafe_ptr(), N * K_BYTES, 2)
    else:
        var a_codes = ctx.enqueue_create_host_buffer[DType.uint8](M * K)
        var b_codes = ctx.enqueue_create_host_buffer[DType.uint8](N * K)
        _fill_codes(a_codes.unsafe_ptr(), M * K, data_pattern, 0)
        _fill_codes(b_codes.unsafe_ptr(), N * K, data_pattern, 1)
        _pack_fp6(a_codes.unsafe_ptr(), a_packed.unsafe_ptr(), M, K)
        _pack_fp6(b_codes.unsafe_ptr(), b_packed.unsafe_ptr(), N, K)

    var a_sf = ctx.enqueue_create_host_buffer[DType.uint8](M * K_SCALES)
    var b_sf = ctx.enqueue_create_host_buffer[DType.uint8](N * K_SCALES)
    _fill_scales(a_sf.unsafe_ptr(), M, K_SCALES, scale_pattern, 0)
    _fill_scales(b_sf.unsafe_ptr(), N, K_SCALES, scale_pattern, 3)

    # Preshuffle B's scales on the host into the packed-cell layout
    # `PreshuffledScaleLoader` reads -- a one-time, load-time cost for the
    # static weight.
    var b_sf_pre = ctx.enqueue_create_host_buffer[DType.uint8](N * K_SCALES)
    ctx.synchronize()
    var b_sf_tt = TileTensor(
        b_sf, row_major(Coord(Idx[1], Idx[N], Idx[K_SCALES]))
    )
    _ = Shuffler[1].preshuffle_scale_4d[MN=N, K_SCALES=K_SCALES](
        b_sf_tt, b_sf_pre
    )

    var a_sf_typed = ctx.enqueue_create_host_buffer[DType.float8_e8m0fnu](
        M * K_SCALES
    )
    # The reference reads B's scales *un*-shuffled, in the original layout;
    # only the kernel under test sees the preshuffled version.
    var b_sf_typed = ctx.enqueue_create_host_buffer[DType.float8_e8m0fnu](
        N * K_SCALES
    )
    var b_sf_pre_typed = ctx.enqueue_create_host_buffer[DType.float8_e8m0fnu](
        N * K_SCALES
    )
    for i in range(M * K_SCALES):
        a_sf_typed[i] = bitcast[DType.float8_e8m0fnu](a_sf[i])
    for i in range(N * K_SCALES):
        b_sf_typed[i] = bitcast[DType.float8_e8m0fnu](b_sf[i])
        b_sf_pre_typed[i] = bitcast[DType.float8_e8m0fnu](b_sf_pre[i])

    var a_dev = ctx.enqueue_create_buffer[DType.uint8](M * K_BYTES)
    var b_dev = ctx.enqueue_create_buffer[DType.uint8](N * K_BYTES)
    var b_pre_dev = ctx.enqueue_create_buffer[DType.uint8](N * K_BYTES)
    var a_sf_dev = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](M * K_SCALES)
    var b_sf_dev = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](N * K_SCALES)
    var b_sf_pre_dev = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
        N * K_SCALES
    )
    var c_dev = ctx.enqueue_create_buffer[out_dtype](M * N)
    var ref_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    var mag_dev = ctx.enqueue_create_buffer[DType.float32](M * N)

    var poison = ctx.enqueue_create_host_buffer[out_dtype](M * N)
    for i in range(M * N):
        poison[i] = SENTINEL.cast[out_dtype]()

    ctx.enqueue_copy(a_dev, a_packed)
    ctx.enqueue_copy(b_dev, b_packed)
    ctx.enqueue_copy(a_sf_dev, a_sf_typed)
    ctx.enqueue_copy(b_sf_dev, b_sf_typed)
    ctx.enqueue_copy(b_sf_pre_dev, b_sf_pre_typed)
    ctx.enqueue_copy(c_dev, poison)
    ctx.synchronize()

    # Preshuffle B's weight bytes on the GPU into the plane-split layout
    # `PreshuffledBLoader` reads. Also a one-time, load-time cost.
    var b_raw_tt = TileTensor[mut=False](b_dev, row_major[1, N, K_BYTES]())
    var b_pre_tt = TileTensor[mut=True](b_pre_dev, row_major[1, N, K_BYTES]())
    Shuffler[1].preshuffle_b_planes[N=N, K_BYTES=K_BYTES, lane_bytes=24](
        b_raw_tt, b_pre_tt, ctx
    )

    comptime REF_BLOCK = 16
    ctx.enqueue_function[_mxfp6_matmul_ref[fmt]](
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        a_sf_dev.unsafe_ptr(),
        b_sf_dev.unsafe_ptr(),
        ref_dev.unsafe_ptr(),
        mag_dev.unsafe_ptr(),
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, REF_BLOCK), ceildiv(N, REF_BLOCK)),
        block_dim=(REF_BLOCK, REF_BLOCK),
    )

    mxfp6_block_scaled_matmul_amd[mfma_format, preshuffled_b=True](
        TileTensor(c_dev, row_major((Int(M), Idx[N]))),
        TileTensor(a_dev, row_major((Int(M), Idx[K_BYTES]))),
        TileTensor(b_pre_dev, row_major[N, K_BYTES]()),
        TileTensor(a_sf_dev, row_major((Int(M), Idx[K_SCALES]))),
        TileTensor(b_sf_pre_dev, row_major[N, K_SCALES]()),
        ctx,
    )
    ctx.synchronize()

    var c_host = ctx.enqueue_create_host_buffer[out_dtype](M * N)
    var ref_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    var mag_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(ref_host, ref_dev)
    ctx.enqueue_copy(mag_host, mag_dev)
    ctx.synchronize()

    comptime ULP_F32 = 5.9604644775390625e-8
    var rel_tol = Float64(0.02) if out_dtype == DType.bfloat16 else (
        Float64(16 * K) * ULP_F32
    )

    var mismatches = 0
    var unwritten = 0
    var saw_nonzero = False
    for i in range(M * N):
        var want = Float64(ref_host[i])
        if want != Float64(0.0):
            saw_nonzero = True
        var got = Float64(c_host[i].cast[DType.float32]())
        if got == Float64(SENTINEL):
            unwritten += 1
        elif abs(got - want) > rel_tol * Float64(mag_host[i]):
            if mismatches < 3:
                print(
                    "      [",
                    i // N,
                    ",",
                    i % N,
                    "] got=",
                    got,
                    " want=",
                    want,
                    " mag=",
                    Float64(mag_host[i]),
                )
            mismatches += 1

    if unwritten > 0 or mismatches > 0:
        print(
            "    FAIL ",
            mismatches,
            " wrong, ",
            unwritten,
            " unwritten, of ",
            M * N,
        )
        return False
    if data_pattern != DATA_ZERO and not saw_nonzero:
        print("    BAD TEST: reference is all zero")
        return False
    return True


def _report[
    fmt: FP6Format,
    N: Int,
    K: Int,
    out_dtype: DType = .float32,
    num_splits: Int = 1,
](
    ctx: DeviceContext, M: Int, data_pattern: Int, scale_pattern: Int
) raises -> Int:
    """Runs one case and returns 1 if it failed, so callers can tally."""
    print(
        "  ",
        "E2M3" if fmt == FP6Format.E2M3 else "E3M2",
        M,
        "x",
        N,
        "x",
        K,
        " splits=",
        num_splits,
        " out=",
        out_dtype,
        " data=",
        data_pattern,
        " scale=",
        scale_pattern,
    )
    var ok = run_case[fmt, N, K, out_dtype, num_splits](
        ctx, M, data_pattern, scale_pattern
    )
    return 0 if ok else 1


def _report_preb[
    fmt: FP6Format,
    N: Int,
    K: Int,
    out_dtype: DType = DType.float32,
](
    ctx: DeviceContext, M: Int, data_pattern: Int, scale_pattern: Int
) raises -> Int:
    """Runs one `preshuffled_b=True` case and returns 1 if it failed."""
    print(
        "  ",
        "E2M3" if fmt == FP6Format.E2M3 else "E3M2",
        M,
        "x",
        N,
        "x",
        K,
        " preshuffled_b out=",
        out_dtype,
        " data=",
        data_pattern,
        " scale=",
        scale_pattern,
    )
    var ok = run_case_preb[fmt, N, K, out_dtype](
        ctx, M, data_pattern, scale_pattern
    )
    return 0 if ok else 1


def test_preshuffled_b_dispatch(ctx: DeviceContext) raises -> Int:
    """Bucket R: `preshuffled_b=True` through the production dispatch path.

    Covers all 5 of a production model's tp=4 dense shapes at M=128 (the
    shape this path was built for -- see `mxfp6_block_scaled_matmul_amd`'s
    docstring), plus M just above `DECODE_M_MAX`, both sides of the
    internal BM=16/BM=32 crossover at `P_LARGE_M_THRESHOLD=1536`, and a
    large prefill M. All M values here are > 64 by construction: smaller M
    is documented to ignore `preshuffled_b` and is covered by the row-major
    buckets above instead.
    """
    comptime fmt = FP6Format.E2M3
    var bad = 0
    for m in [65, 96, 128, 256, 1535, 1536, 8192]:
        bad += _report_preb[fmt, 2048, 6144](ctx, m, DATA_RANDOM, SCALE_VARY)
        bad += _report_preb[fmt, 128, 6144](ctx, m, DATA_RANDOM, SCALE_VARY)
        bad += _report_preb[fmt, 6144, 2048](ctx, m, DATA_RANDOM, SCALE_VARY)
        bad += _report_preb[fmt, 3072, 6144](ctx, m, DATA_RANDOM, SCALE_VARY)
        bad += _report_preb[fmt, 6144, 3072](ctx, m, DATA_RANDOM, SCALE_VARY)
    bad += _report_preb[fmt, 128, 6144, DType.bfloat16](
        ctx, 128, DATA_RANDOM, SCALE_VARY
    )
    bad += _report_preb[FP6Format.E3M2, 2048, 6144](
        ctx, 128, DATA_RANDOM, SCALE_VARY
    )
    return bad


def test_baseline_aligned(ctx: DeviceContext) raises -> Int:
    """Bucket A: aligned shapes, the sanity floor."""
    comptime fmt = FP6Format.E2M3
    var bad = 0
    for m in [128, 256]:
        bad += _report[fmt, 128, 128](ctx, m, DATA_RANDOM, SCALE_VARY)
        bad += _report[fmt, 128, 256](ctx, m, DATA_RANDOM, SCALE_VARY)
        bad += _report[fmt, 256, 256](ctx, m, DATA_RANDOM, SCALE_VARY)
    bad += _report[fmt, 128, 512](ctx, 128, DATA_RANDOM, SCALE_VARY)
    bad += _report[fmt, 128, 1024](ctx, 128, DATA_RANDOM, SCALE_VARY)
    bad += _report[fmt, 256, 512](ctx, 256, DATA_RANDOM, SCALE_VARY)
    return bad


def test_production_shapes(ctx: DeviceContext) raises -> Int:
    """Bucket B: real projection shapes with unaligned M.

    Mirrors `test_block_scaled_matmul_amd`'s Kimi K2.5 matrix. Every N there is
    already a multiple of BN=64 and every K a multiple of 128, so the shapes
    port unchanged; only the M values need no adjustment either. M is a runtime
    argument, so a whole row of these costs one instantiation.
    """
    comptime fmt = FP6Format.E2M3
    var bad = 0

    # M=1 decode, and the unaligned prefill sizes, across the projections.
    for m in [1, 17, 53, 73, 111, 127, 129, 255, 257]:
        bad += _report[fmt, 7168, 2048](ctx, m, DATA_RANDOM, SCALE_VARY)
        bad += _report[fmt, 2048, 7168](ctx, m, DATA_RANDOM, SCALE_VARY)
    for m in [1, 17, 73, 129]:
        bad += _report[fmt, 4096, 7168](ctx, m, DATA_RANDOM, SCALE_VARY)
    # Widest N and deepest K: maximum DRAM and scale volume against a single
    # real row, with the rest of the block OOB.
    bad += _report[fmt, 18432, 7168](ctx, 1, DATA_RANDOM, SCALE_VARY)
    bad += _report[fmt, 18432, 7168](ctx, 111, DATA_RANDOM, SCALE_VARY)
    bad += _report[fmt, 7168, 18432](ctx, 53, DATA_RANDOM, SCALE_VARY)
    return bad


def test_split_k(ctx: DeviceContext) raises -> Int:
    """Bucket S: the inter-block split-K path plus its reduce kernel.

    Legality (asserted by `BlockScaledMatmulAMD.run`) is `K_BYTES % num_splits == 0`
    and `(K_BYTES / num_splits) % BK_BYTES == 0`, with BK_BYTES = 96 for FP6.
    K=7168 gives K_BYTES=5376, so splits of 2/4/8/14 all divide; K=2048 gives
    1536, so 2/4/8 divide.
    """
    comptime fmt = FP6Format.E2M3
    var bad = 0
    for m in [1, 16, 64]:
        bad += _report[fmt, 4096, 7168, .float32, 14](
            ctx, m, DATA_RANDOM, SCALE_VARY
        )
        bad += _report[fmt, 7168, 2048, .float32, 8](
            ctx, m, DATA_RANDOM, SCALE_VARY
        )
    # Unaligned M against split-K: the reduce kernel walks M*N flat, so a
    # partial final block is the interesting case.
    bad += _report[fmt, 7168, 2048, .float32, 2](
        ctx, 17, DATA_RANDOM, SCALE_VARY
    )
    bad += _report[fmt, 4096, 7168, .float32, 4](
        ctx, 63, DATA_SWEEP, SCALE_VARY
    )
    bad += _report[fmt, 4096, 7168, .float32, 14](
        ctx, 129, DATA_RANDOM, SCALE_EXTREME
    )
    # bfloat16 output exercises the reduce kernel's cast.
    bad += _report[fmt, 7168, 2048, .bfloat16, 8](
        ctx, 64, DATA_RANDOM, SCALE_VARY
    )
    return bad


def test_m_sweep(ctx: DeviceContext) raises -> Int:
    """Dense M sweep through one instantiation: BM=96, MFMA tile 16 rows."""
    comptime fmt = FP6Format.E2M3
    var bad = 0
    for m in [
        1,
        2,
        3,
        15,
        16,
        17,
        31,
        32,
        33,
        47,
        48,
        63,
        64,
        65,
        79,
        80,
        95,
        96,
        97,
        111,
        112,
        127,
        128,
        159,
        160,
        191,
        192,
        193,
        255,
        256,
        287,
        288,
        383,
        384,
        385,
    ]:
        bad += _report[fmt, 64, 128](ctx, m, DATA_SWEEP, SCALE_VARY)
    return bad


def test_k_paths(ctx: DeviceContext) raises -> Int:
    """K = 128 takes the simple-loop fallback; K >= 256 the pipelined path."""
    comptime fmt = FP6Format.E2M3
    var bad = 0
    for m in [1, 95, 96, 97, 192]:
        bad += _report[fmt, 64, 256](ctx, m, DATA_RANDOM, SCALE_VARY)
        bad += _report[fmt, 64, 512](ctx, m, DATA_SWEEP, SCALE_VARY)
    bad += _report[fmt, 64, 384](ctx, 97, DATA_RANDOM, SCALE_VARY)
    bad += _report[fmt, 64, 1024](ctx, 97, DATA_RANDOM, SCALE_VARY)
    return bad


def test_scale_and_data_patterns(ctx: DeviceContext) raises -> Int:
    """Unit scales cannot detect a wrong scale byte; varying ones can.

    Also covers the degenerate data and the bit-exact canary. These reuse
    already-instantiated shapes, so they cost execution time only.
    """
    comptime fmt = FP6Format.E2M3
    var bad = 0
    for m in [1, 96, 97]:
        bad += _report[fmt, 64, 256](ctx, m, DATA_RANDOM, SCALE_UNIT)
        bad += _report[fmt, 64, 256](ctx, m, DATA_RANDOM, SCALE_EXTREME)
        bad += _report[fmt, 64, 256](ctx, m, DATA_ZERO, SCALE_VARY)
        bad += _report[fmt, 64, 256](ctx, m, DATA_MAX, SCALE_UNIT)
        bad += _report[fmt, 64, 256](ctx, m, DATA_MAX, SCALE_EXTREME)
        bad += _report[fmt, 64, 256](ctx, m, DATA_FP4_SUBSET, SCALE_UNIT)
    return bad


def test_e3m2(ctx: DeviceContext) raises -> Int:
    """The second FP6 encoding, across both K paths, split-K and shapes."""
    comptime fmt = FP6Format.E3M2
    var bad = 0
    for m in [1, 95, 96, 97, 192]:
        bad += _report[fmt, 64, 128](ctx, m, DATA_SWEEP, SCALE_VARY)
        bad += _report[fmt, 64, 512](ctx, m, DATA_RANDOM, SCALE_VARY)
    bad += _report[fmt, 64, 128](ctx, 96, DATA_MAX, SCALE_UNIT)
    bad += _report[fmt, 7168, 2048](ctx, 17, DATA_RANDOM, SCALE_VARY)
    bad += _report[fmt, 7168, 2048, .float32, 8](
        ctx, 16, DATA_RANDOM, SCALE_VARY
    )
    return bad


def test_output_dtypes(ctx: DeviceContext) raises -> Int:
    """`bfloat16` exercises the cast in the store epilogue."""
    var bad = 0
    for m in [1, 96, 97]:
        bad += _report[FP6Format.E2M3, 64, 128, .bfloat16](
            ctx, m, DATA_SWEEP, SCALE_VARY
        )
        bad += _report[FP6Format.E3M2, 64, 256, .bfloat16](
            ctx, m, DATA_RANDOM, SCALE_VARY
        )
    return bad


def test_quantizer_feeds_matmul[
    fmt: FP6Format, N: Int, K: Int
](ctx: DeviceContext, M: Int) raises -> Int:
    """End-to-end: `quantize_mxfp6_amd` output must be consumable by the MFMA.

    Every other test in this file synthesizes packed bytes directly, so none of
    them can catch a packing convention that the quantizer and the software
    decoder agree on but the HARDWARE does not. Here the bytes come from the
    quantizer, and the reference decodes those same bytes in software: if the
    quantizer wrote a layout the MFMA reads differently, the two disagree.

    Parameters:
        fmt: The FP6 encoding under test.
        N: Output columns.
        K: Reduction extent.

    Args:
        ctx: Device context.
        M: Row count (runtime, as in production).

    Returns:
        The number of mismatching output elements.
    """
    comptime K_BYTES = (K * 6) // 8
    comptime K_SCALES = K // 32

    var a_bf = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    var b_bf = ctx.enqueue_create_host_buffer[.bfloat16](N * K)
    ctx.synchronize()

    var state = UInt64(0xC2B2AE3D27D4EB4F)
    for i in range(M * K):
        state = state * 6364136223846793005 + 1442695040888963407
        a_bf[i] = (
            Float32(Int((state >> 40) & 0xFF)) / Float32(128.0) - Float32(1.0)
        ).cast[.bfloat16]()
    for i in range(N * K):
        state = state * 6364136223846793005 + 1442695040888963407
        b_bf[i] = (
            Float32(Int((state >> 40) & 0xFF)) / Float32(128.0) - Float32(1.0)
        ).cast[.bfloat16]()

    var a_bf_d = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_bf_d = ctx.enqueue_create_buffer[.bfloat16](N * K)
    ctx.enqueue_copy(a_bf_d, a_bf)
    ctx.enqueue_copy(b_bf_d, b_bf)

    var a_q = ctx.enqueue_create_buffer[.uint8](M * K_BYTES)
    var b_q = ctx.enqueue_create_buffer[.uint8](N * K_BYTES)
    var a_sf = ctx.enqueue_create_buffer[.float8_e8m0fnu](M * K_SCALES)
    var b_sf = ctx.enqueue_create_buffer[.float8_e8m0fnu](N * K_SCALES)

    quantize_mxfp6_amd[fmt](
        ctx,
        TileTensor(a_q, row_major((M, Idx[K_BYTES]))),
        TileTensor(a_sf, row_major((M, Idx[K_SCALES]))),
        TileTensor(a_bf_d, row_major((M, Idx[K]))).as_immut(),
    )
    quantize_mxfp6_amd[fmt](
        ctx,
        TileTensor(b_q, row_major[N, K_BYTES]()),
        TileTensor(b_sf, row_major[N, K_SCALES]()),
        TileTensor(b_bf_d, row_major[N, K]()).as_immut(),
    )

    var c_d = ctx.enqueue_create_buffer[.float32](M * N)
    var ref_d = ctx.enqueue_create_buffer[.float32](M * N)
    var mag_d = ctx.enqueue_create_buffer[.float32](M * N)

    comptime REF_BLOCK = 16
    ctx.enqueue_function[_mxfp6_matmul_ref[fmt]](
        a_q.unsafe_ptr(),
        b_q.unsafe_ptr(),
        a_sf.unsafe_ptr(),
        b_sf.unsafe_ptr(),
        ref_d.unsafe_ptr(),
        mag_d.unsafe_ptr(),
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, REF_BLOCK), ceildiv(N, REF_BLOCK)),
        block_dim=(REF_BLOCK, REF_BLOCK),
    )

    comptime mfma_format = (
        CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3 if fmt
        == FP6Format.E2M3 else CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2
    )
    mxfp6_block_scaled_matmul_amd[mfma_format](
        TileTensor(c_d, row_major((M, Idx[N]))),
        TileTensor(a_q, row_major((M, Idx[K_BYTES]))),
        TileTensor(b_q, row_major[N, K_BYTES]()),
        TileTensor(a_sf, row_major((M, Idx[K_SCALES]))),
        TileTensor(b_sf, row_major[N, K_SCALES]()),
        ctx,
    )
    ctx.synchronize()

    var c_h = ctx.enqueue_create_host_buffer[.float32](M * N)
    var ref_h = ctx.enqueue_create_host_buffer[.float32](M * N)
    var mag_h = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(c_h, c_d)
    ctx.enqueue_copy(ref_h, ref_d)
    ctx.enqueue_copy(mag_h, mag_d)
    ctx.synchronize()

    comptime ULP_F32 = 5.9604644775390625e-8
    var rel_tol = Float64(16 * K) * ULP_F32
    var bad = 0
    var saw_nonzero = False
    for i in range(M * N):
        var want = Float64(ref_h[i])
        if want != Float64(0.0):
            saw_nonzero = True
        if abs(Float64(c_h[i]) - want) > rel_tol * Float64(mag_h[i]):
            bad += 1

    comptime fmt_name = "E2M3" if fmt == FP6Format.E2M3 else "E3M2"
    if not saw_nonzero:
        print("    FAIL quantized-input reference is all zero")
        bad += 1
    print(
        "    ",
        "PASS" if bad == 0 else "FAIL",
        " quantizer->matmul ",
        fmt_name,
        " M=",
        M,
        " N=",
        N,
        " K=",
        K,
    )

    _ = a_bf_d^
    _ = b_bf_d^
    _ = a_q^
    _ = b_q^
    _ = a_sf^
    _ = b_sf^
    _ = c_d^
    _ = ref_d^
    _ = mag_d^
    return bad


def main() raises:
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "MXFP6 matmul test requires MI355X"

    seed(0)
    print("===> MXFP6 block-scaled matmul (native CDNA4 MFMA)")
    var bad = 0
    print("\n--- A: baseline aligned shapes ---")
    bad += test_baseline_aligned(ctx)
    print("\n--- B: production shapes, unaligned M ---")
    bad += test_production_shapes(ctx)
    print("\n--- S: inter-block split-K ---")
    bad += test_split_k(ctx)
    print("\n--- M: dense M sweep ---")
    bad += test_m_sweep(ctx)
    print("\n--- K: both K-loop paths ---")
    bad += test_k_paths(ctx)
    print("\n--- P: scale and data patterns ---")
    bad += test_scale_and_data_patterns(ctx)
    print("\n--- E: E3M2 encoding ---")
    bad += test_e3m2(ctx)
    print("\n--- Q: quantizer output feeds the MFMA ---")
    for m in [1, 17, 128]:
        bad += test_quantizer_feeds_matmul[FP6Format.E2M3, 128, 256](ctx, m)
        bad += test_quantizer_feeds_matmul[FP6Format.E3M2, 128, 256](ctx, m)
    bad += test_quantizer_feeds_matmul[FP6Format.E2M3, 512, 2048](ctx, 64)

    print("\n--- D: output dtypes ---")
    bad += test_output_dtypes(ctx)

    print("\n--- R: preshuffled_b dispatch (M=128 fix) ---")
    bad += test_preshuffled_b_dispatch(ctx)

    if bad > 0:
        print("\nFAILED cases: ", bad)
        raise Error("MXFP6 matmul: one or more cases disagree with reference")
    print("\nPASS")
