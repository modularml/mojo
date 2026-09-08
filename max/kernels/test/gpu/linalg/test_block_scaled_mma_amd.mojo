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
"""Minimal MI355 smoke test for the CDNA4 `f8f6f4` block-scaled MFMA wrappers.

This test intentionally validates the wrappers at the raw fragment level rather
than reconstructing a logical output matrix. It launches a single warp, runs
the same MFMA twice with identical packed raw fragments, and checks that
doubling one real E8M0 scale word doubles every accumulator lane for each
supported CDNA4 `f8f6f4` operand format and wrapper shape.

FP6 operands hold 24 payload bytes inside a 32-byte fragment, so a second test
confirms on hardware that the trailing bytes never reach the MFMA.
"""

from std.builtin.simd import _convert_f32_to_float8_ue8m0
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE, lane_id
from max.gpu.host import DeviceContext
from max.gpu.host.info import MI355X
from std.memory import bitcast
from std.testing import assert_true
from std.utils import StaticTuple

from layout import Coord, TensorLayout, TileTensor, row_major
from linalg.arch.amd.block_scaled_mma import (
    CDNA4F8F6F4MatrixFormat,
    cdna4_block_scaled_mfma,
)


@always_inline
def _pack_e8m0_scale_word(value: Float32) -> Int32:
    """Packs one exact E8M0 scale into all four bytes of the MFMA scale word."""
    var scale = _convert_f32_to_float8_ue8m0[target=DType.float8_e8m0fnu](value)
    var scale_byte = bitcast[.uint8](scale)
    return Int32(
        UInt32(scale_byte)
        | (UInt32(scale_byte) << 8)
        | (UInt32(scale_byte) << 16)
        | (UInt32(scale_byte) << 24)
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(WARP_SIZE))
)
def _block_scaled_mma_smoke_kernel[
    BaselineLayout: TensorLayout,
    ScaledLayout: TensorLayout,
    accum_width: Int,
    matrix_format: CDNA4F8F6F4MatrixFormat,
](
    baseline_out: TileTensor[.float32, BaselineLayout, MutAnyOrigin],
    scaled_out: TileTensor[.float32, ScaledLayout, MutAnyOrigin],
):
    comptime assert accum_width == 4 or accum_width == 16, (
        "AMD block-scaled MMA smoke test only supports 4- or 16-lane"
        " accumulators"
    )
    var lane = lane_id()
    var a_frag = SIMD[.uint8, matrix_format.simd_width()](UInt8(0x21))
    var b_frag = SIMD[.uint8, matrix_format.simd_width()](UInt8(0x12))

    var one = _pack_e8m0_scale_word(1.0)
    var two = _pack_e8m0_scale_word(2.0)

    var baseline_acc = SIMD[.float32, accum_width](0.0)
    var scaled_acc = SIMD[.float32, accum_width](0.0)

    cdna4_block_scaled_mfma[0, 0, matrix_format, matrix_format](
        baseline_acc,
        a_frag,
        b_frag,
        one,
        one,
    )
    cdna4_block_scaled_mfma[0, 0, matrix_format, matrix_format](
        scaled_acc,
        a_frag,
        b_frag,
        two,
        one,
    )

    baseline_out.store(Coord(lane, 0), baseline_acc)
    scaled_out.store(Coord(lane, 0), scaled_acc)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(WARP_SIZE))
)
def _fp6_padding_kernel[
    ZeroPadLayout: TensorLayout,
    OnePadLayout: TensorLayout,
    accum_width: Int,
    matrix_format: CDNA4F8F6F4MatrixFormat,
](
    zero_pad_out: TileTensor[.float32, ZeroPadLayout, MutAnyOrigin],
    one_pad_out: TileTensor[.float32, OnePadLayout, MutAnyOrigin],
):
    """Runs the same FP6 MFMA under two padding patterns; results must match."""
    comptime assert (
        matrix_format.simd_width() == 32
    ), "FP6 operands are expected to travel in a 32-byte fragment"
    comptime payload_bytes = 24

    var lane = lane_id()
    var zero_pad_a = SIMD[.uint8, 32](UInt8(0x21))
    var zero_pad_b = SIMD[.uint8, 32](UInt8(0x12))
    var one_pad_a = zero_pad_a
    var one_pad_b = zero_pad_b

    comptime for i in range(payload_bytes, 32):
        zero_pad_a[i] = UInt8(0x00)
        zero_pad_b[i] = UInt8(0x00)
        one_pad_a[i] = UInt8(0xFF)
        one_pad_b[i] = UInt8(0xFF)

    var one = _pack_e8m0_scale_word(1.0)
    var zero_pad_acc = SIMD[.float32, accum_width](0.0)
    var one_pad_acc = SIMD[.float32, accum_width](0.0)

    cdna4_block_scaled_mfma[0, 0, matrix_format, matrix_format](
        zero_pad_acc,
        zero_pad_a,
        zero_pad_b,
        one,
        one,
    )
    cdna4_block_scaled_mfma[0, 0, matrix_format, matrix_format](
        one_pad_acc,
        one_pad_a,
        one_pad_b,
        one,
        one,
    )

    zero_pad_out.store(Coord(lane, 0), zero_pad_acc)
    one_pad_out.store(Coord(lane, 0), one_pad_acc)


def _run_fp6_padding_check[
    matrix_format: CDNA4F8F6F4MatrixFormat,
    accum_width: Int,
](ctx: DeviceContext) raises:
    comptime num_values = WARP_SIZE * accum_width

    var zero_pad_device = ctx.enqueue_create_buffer[.float32](num_values)
    var one_pad_device = ctx.enqueue_create_buffer[.float32](num_values)

    var zero_pad_tt = TileTensor(
        zero_pad_device, row_major[WARP_SIZE, accum_width]()
    )
    var one_pad_tt = TileTensor(
        one_pad_device, row_major[WARP_SIZE, accum_width]()
    )

    comptime kernel = _fp6_padding_kernel[
        type_of(zero_pad_tt).LayoutType,
        type_of(one_pad_tt).LayoutType,
        accum_width,
        matrix_format,
    ]

    ctx.enqueue_function[kernel](
        zero_pad_tt.as_unsafe_any_origin(),
        one_pad_tt.as_unsafe_any_origin(),
        grid_dim=1,
        block_dim=WARP_SIZE,
    )
    ctx.synchronize()

    var zero_pad_host = ctx.enqueue_create_host_buffer[.float32](num_values)
    var one_pad_host = ctx.enqueue_create_host_buffer[.float32](num_values)
    ctx.enqueue_copy(zero_pad_host, zero_pad_device)
    ctx.enqueue_copy(one_pad_host, one_pad_device)
    ctx.synchronize()

    var saw_nonzero = False
    for i in range(num_values):
        if abs(zero_pad_host[i]) > Float32(1e-6):
            saw_nonzero = True
        assert_true(
            zero_pad_host[i] == one_pad_host[i],
            "FP6 MFMA output must not depend on the fragment padding bytes",
        )
    assert_true(saw_nonzero, "FP6 padding check should produce non-zero output")


def _run_block_scaled_mma_amd_smoke[
    matrix_format: CDNA4F8F6F4MatrixFormat,
    accum_width: Int,
](ctx: DeviceContext) raises:
    comptime num_values = WARP_SIZE * accum_width

    var baseline_device = ctx.enqueue_create_buffer[.float32](num_values)
    var scaled_device = ctx.enqueue_create_buffer[.float32](num_values)

    var baseline_tt = TileTensor(
        baseline_device, row_major[WARP_SIZE, accum_width]()
    )
    var scaled_tt = TileTensor(
        scaled_device, row_major[WARP_SIZE, accum_width]()
    )

    comptime kernel = _block_scaled_mma_smoke_kernel[
        type_of(baseline_tt).LayoutType,
        type_of(scaled_tt).LayoutType,
        accum_width,
        matrix_format,
    ]

    ctx.enqueue_function[kernel](
        baseline_tt.as_unsafe_any_origin(),
        scaled_tt.as_unsafe_any_origin(),
        grid_dim=1,
        block_dim=WARP_SIZE,
    )
    ctx.synchronize()

    var baseline_host = alloc[Float32](num_values)
    var scaled_host = alloc[Float32](num_values)
    ctx.enqueue_copy(baseline_host, baseline_device)
    ctx.enqueue_copy(scaled_host, scaled_device)
    ctx.synchronize()

    var saw_nonzero = False
    for i in range(num_values):
        var baseline = baseline_host[i]
        var scaled = scaled_host[i]
        if abs(baseline) > Float32(1e-6):
            saw_nonzero = True
        var diff = abs(scaled - baseline * Float32(2.0))
        assert_true(
            diff <= Float32(1e-5),
            "scaled MFMA output should double when scale_a doubles",
        )
    assert_true(
        saw_nonzero, "wrapper smoke test should produce non-zero output"
    )


def test_block_scaled_mma_amd_smoke(ctx: DeviceContext) raises:
    comptime for matrix_format in [
        CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3,
        CDNA4F8F6F4MatrixFormat.FLOAT8_E5M2,
        CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3,
        CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    ]:
        _run_block_scaled_mma_amd_smoke[matrix_format, 4](ctx)
        _run_block_scaled_mma_amd_smoke[matrix_format, 16](ctx)


def test_fp6_ignores_fragment_padding(ctx: DeviceContext) raises:
    comptime for matrix_format in [
        CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3,
        CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2,
    ]:
        _run_fp6_padding_check[matrix_format, 4](ctx)
        _run_fp6_padding_check[matrix_format, 16](ctx)


def main() raises:
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "AMD block-scaled MMA smoke test requires MI355X"

    print("== test_block_scaled_mma_amd_smoke")
    test_block_scaled_mma_amd_smoke(ctx)
    print("== test_fp6_ignores_fragment_padding")
    test_fp6_ignores_fragment_padding(ctx)
    print("PASS")
