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
"""Numeric smoke test for `mfma_scale_f32_32x32x64_f8f6f4` on CDNA4.

Calls the CDNA4 32x32x64 block-scaled MFMA once on operands set to FP4 `+1.0`
with `scale_a = scale_b = 1.0`. Each accumulator lane should equal K = 64
(the sum across 64 FP4 element-wise products of 1.0 * 1.0). A second pass
with `scale_a = 2.0` should double every lane.

This test is intentionally scoped to the 32x32x64 shape only, to serve as the
regression artifact for MAX-595. The broader parametric coverage of both
CDNA4 block-scaled shapes lives in `test_block_scaled_mma_amd.mojo`.
"""

from std.builtin.simd import _convert_f32_to_float8_ue8m0
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE, lane_id
from max.gpu.host import DeviceContext
from max.gpu.host.info import MI355X
from std.memory import bitcast
from std.testing import assert_true
from std.utils import StaticTuple

from layout import Coord, Idx, TensorLayout, TileTensor, row_major
from linalg.arch.amd.block_scaled_mma import (
    CDNA4F8F6F4MatrixFormat,
    cdna4_block_scaled_mfma,
)


@always_inline
def _pack_e8m0_scale_word(value: Float32) -> Int32:
    """Packs one exact E8M0 scale into all four bytes of the MFMA scale word.

    The CDNA4 block-scaled MFMA consumes its A-scale and B-scale operands as
    32-bit values, with a comptime byte index selecting which of the four
    bytes is the "real" scale for this dispatch. By replicating the same
    scale byte across all four positions, the byte-index selector becomes
    irrelevant — every dispatch sees the same scale value no matter which
    byte we ask for.
    """
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
def _mfma_32x32x64_kernel[
    BaselineLayout: TensorLayout,
    ScaledLayout: TensorLayout,
](
    baseline_out: TileTensor[.float32, BaselineLayout, MutAnyOrigin],
    scaled_out: TileTensor[.float32, ScaledLayout, MutAnyOrigin],
):
    """Single-warp kernel that runs the 32x32x64 MFMA twice and writes both
    accumulators to global memory, one lane per row of each output buffer."""
    var lane = Int(lane_id())

    # FP4 byte 0x22 = two nibbles of E2M1 +1.0. Every byte of A and B is +1.0.
    var a_frag = SIMD[.uint8, 16](UInt8(0x22))
    var b_frag = SIMD[.uint8, 16](UInt8(0x22))

    var one = _pack_e8m0_scale_word(Float32(1.0))
    var two = _pack_e8m0_scale_word(Float32(2.0))

    # Accumulator width 16 -> wrapper dispatches the 32x32x64 instruction.
    var baseline_acc = SIMD[.float32, 16](0.0)
    var scaled_acc = SIMD[.float32, 16](0.0)

    cdna4_block_scaled_mfma[
        0,
        0,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    ](baseline_acc, a_frag, b_frag, one, one)

    cdna4_block_scaled_mfma[
        0,
        0,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    ](scaled_acc, a_frag, b_frag, two, one)

    baseline_out.store[width=16](Coord(lane, Idx[0]), baseline_acc)
    scaled_out.store[width=16](Coord(lane, Idx[0]), scaled_acc)


def _run_mfma_32x32x64_smoke(ctx: DeviceContext) raises:
    """Launches the 32x32x64 MFMA kernel and asserts numeric correctness.

    Per-lane expectation: every accumulator element should be 64.0 in the
    baseline (K=64 ones-times-ones products, scale 1.0 on both sides), and
    128.0 when scale_a is doubled. We assert on every element of every lane,
    not just a sample.
    """
    comptime accum_width = 16
    comptime num_values = WARP_SIZE * accum_width  # 64 lanes * 16 floats = 1024

    var baseline_device = ctx.enqueue_create_buffer[.float32](num_values)
    var scaled_device = ctx.enqueue_create_buffer[.float32](num_values)

    var baseline_tt = TileTensor(
        baseline_device,
        row_major(Coord(Idx[WARP_SIZE], Idx[accum_width])),
    )
    var scaled_tt = TileTensor(
        scaled_device,
        row_major(Coord(Idx[WARP_SIZE], Idx[accum_width])),
    )

    comptime kernel = _mfma_32x32x64_kernel[
        type_of(baseline_tt).LayoutType,
        type_of(scaled_tt).LayoutType,
    ]

    ctx.enqueue_function[kernel](
        baseline_tt.as_unsafe_any_origin(),
        scaled_tt.as_unsafe_any_origin(),
        grid_dim=1,
        block_dim=WARP_SIZE,
    )
    ctx.synchronize()

    var baseline_host = ctx.enqueue_create_host_buffer[.float32](num_values)
    var scaled_host = ctx.enqueue_create_host_buffer[.float32](num_values)
    ctx.enqueue_copy(baseline_host, baseline_device)
    ctx.enqueue_copy(scaled_host, scaled_device)
    ctx.synchronize()

    for i in range(num_values):
        var baseline = baseline_host[i]
        var scaled = scaled_host[i]
        assert_true(
            abs(baseline - Float32(64.0)) <= Float32(1e-5),
            "baseline accumulator element must equal 64.0",
        )
        assert_true(
            abs(scaled - Float32(128.0)) <= Float32(1e-5),
            "scaled accumulator element must equal 128.0 (doubled scale_a)",
        )


def main() raises:
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "32x32x64 f8f6f4 MFMA test requires MI355X"

    print("== test_mfma_32x32x64_f8f6f4_smoke")
    _run_mfma_32x32x64_smoke(ctx)
    print("PASS")
