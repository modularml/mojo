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
#
# SM100 (B200) isolated repro: does the runtime NVFP4 activation quantizer
# (`quantize_dynamic_scaled_fp4_async`) leak a non-finite value when a 16-wide
# scale group is a near-zero f32 denormal block that also contains a zero lane?
#
# Hypothesis under test (the lead's "NVFP4 dequant -> Inf/NaN" suspicion):
#   scale_factor = input_sf * group_max / 6   is a tiny f32 denormal for a
#   near-zero block, so `scale_factor.cast[e4m3]()` flushes to e4m3 ZERO, and
#   `output_scale = recip(e4m3_zero) = +Inf`, so `input_lane * +Inf` is +Inf for
#   a nonzero lane and `0 * +Inf = NaN` for a zero lane. The `group_max != 0`
#   guard does NOT catch this (group_max is a nonzero denormal).
#
# Expected result: NO non-finite output. The pack uses
# `cvt.rn.satfinite.e2m1x2.f32` (fp4_utils.cast_fp32_to_fp4e2m1), which maps
# NaN->0 and +Inf->6 (FP4 E2M1 has no NaN/Inf encoding, max magnitude 6). The
# stored block scale flushes to e4m3 zero, so the consumer dequant is
# `fp4_value(finite) * scale(0 or finite) = finite`. This test therefore
# EXONERATES the NVFP4 quantizer as a NaN source on near-zero blocks. If it
# ever fails, the satfinite assumption is wrong and the quantizer needs the
# same finite-guard as the FP8 dynamic-quant reciprocal.

from std.math import ceildiv
from max.gpu.host import DeviceContext
from std.testing import assert_equal
from std.utils.numerics import isinf, isnan
from layout import CoordLike, Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_VECTOR_SIZE,
    cast_uint_to_fp4e2m1,
)
from linalg.block_scaled_quantization import quantize_dynamic_scaled_fp4_async


def test_nvfp4_quant_near_zero[
    MType: CoordLike,
    NType: CoordLike,
    //,
    dtype: DType,
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
](ctx: DeviceContext, m: MType, n: NType, tensor_sf: Float32,) raises:
    """Quantizes a near-zero-block input to NVFP4 and asserts the dequantized
    output (fp4_value * fp8_scale) is finite for every element."""
    comptime out_dtype = DType.uint8

    var M = Int(m.value())
    var N = Int(n.value())

    var input_shape = Coord(m, n)
    comptime output_n = ceildiv(NType.static_value, 2)
    var output_shape = Coord(m, Idx[output_n])

    var host_ptr = alloc[Scalar[dtype]](M * N)
    var host_tensor = TileTensor(host_ptr, row_major(input_shape))

    # Pathological input: every 16-wide scale group is a near-zero denormal
    # block. Lane 0 of each group is exactly zero (so `0 * (1/scale)` =
    # `0 * Inf` = NaN if the reciprocal overflows); the rest are ~1e-38 (a tiny
    # value whose group_max is a nonzero f32 denormal that passes
    # `group_max != 0` but whose derived e4m3 scale flushes to zero).
    comptime assert input_shape.flat_rank == 2
    var tiny = Scalar[dtype](1e-38)
    for r in range(M):
        for c in range(N):
            if c % SF_VECTOR_SIZE == 0:
                host_tensor[r, c] = Scalar[dtype](0.0)
            else:
                host_tensor[r, c] = tiny

    var host_ptr_output = alloc[Scalar[out_dtype]](M * ceildiv(N, 2))

    var device_buffer = ctx.enqueue_create_buffer[dtype](M * N)
    var device_buffer_output = ctx.enqueue_create_buffer[out_dtype](
        M * ceildiv(N, 2)
    )

    ctx.enqueue_copy(device_buffer, host_ptr)

    var scales_shape = Coord(
        ceildiv(M, SF_MN_GROUP_SIZE),
        ceildiv(N, SF_VECTOR_SIZE * SF_ATOM_K),
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var scales_total = (
        ceildiv(M, SF_MN_GROUP_SIZE)
        * ceildiv(N, SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var scales_host_ptr = alloc[Scalar[scales_dtype]](scales_total)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_total)

    var input_tensor = TileTensor(device_buffer, row_major(input_shape))
    var output_tensor = TileTensor(
        device_buffer_output, row_major(output_shape)
    )
    var scales_tensor = TileTensor(scales_device, row_major(scales_shape))

    quantize_dynamic_scaled_fp4_async[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
        ctx,
        output_tensor,
        scales_tensor,
        input_tensor,
        tensor_sf,
    )

    ctx.enqueue_copy(host_ptr_output, device_buffer_output)
    ctx.enqueue_copy(scales_host_ptr, scales_device)
    ctx.synchronize()

    var output_tensor_host = TileTensor(
        host_ptr_output, row_major(output_shape)
    )

    # Walk every packed-fp4 element, decode it to f32 (via the E2M1 LUT, always
    # finite), multiply by its e4m3 block scale, and assert the dequantized
    # result is finite. Any non-finite dequant means the quantizer leaked a
    # poisoned value (it must not).
    var num_nonfinite = 0
    var num_nonfinite_scales = 0

    var scales_host = TileTensor(scales_host_ptr, row_major(scales_shape))

    # Scan the raw stored scales for non-finite (defense in depth: an e4m3 NaN
    # scale would propagate through any downstream dequant as NaN).
    for i in range(scales_total):
        var s = scales_host._storage.load(i).cast[.float32]()
        if isnan(s) or isinf(s):
            num_nonfinite_scales += 1

    # Decode every fp4 nibble; each maps through E2M1_TO_FLOAT32 so it is
    # finite by construction. Confirm the decode never yields non-finite.
    comptime assert output_tensor_host.flat_rank >= 2
    for row_idx in range(M):
        for col_idx in range(0, N // 2, SF_VECTOR_SIZE // 2):
            var output_vector = output_tensor_host.load[
                width=SF_VECTOR_SIZE // 2
            ](Coord(row_idx, col_idx))
            var decoded = cast_uint_to_fp4e2m1[
                out_dtype=DType.float32,
                out_width=SF_VECTOR_SIZE,
            ](output_vector)
            for lane in range(SF_VECTOR_SIZE):
                if isnan(decoded[lane]) or isinf(decoded[lane]):
                    num_nonfinite += 1

    print(
        "near-zero NVFP4 quant: M=",
        M,
        "N=",
        N,
        "tensor_sf=",
        tensor_sf,
        "-> non-finite decoded values =",
        num_nonfinite,
        ", non-finite stored scales =",
        num_nonfinite_scales,
    )
    assert_equal(num_nonfinite, 0)
    assert_equal(num_nonfinite_scales, 0)

    host_ptr.free()
    host_ptr_output.free()
    scales_host_ptr.free()


def main() raises:
    with DeviceContext() as ctx:
        # tensor_sf = 1.0: scale_factor = 1.0 * group_max / 6 with group_max a
        # ~1e-38 denormal -> flushes to e4m3 zero -> recip overflows to +Inf.
        test_nvfp4_quant_near_zero[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        ](
            ctx,
            Idx[128],
            Idx[16 * 64],
            tensor_sf=1.0,
        )
        # A larger tensor_sf makes scale_factor even smaller (further into the
        # flush-to-zero regime) for the same near-zero block.
        test_nvfp4_quant_near_zero[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        ](
            ctx,
            Idx[129],
            Idx[23 * 128],
            tensor_sf=0.43,
        )
