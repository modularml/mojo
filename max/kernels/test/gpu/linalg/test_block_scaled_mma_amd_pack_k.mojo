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
"""MI355 test for `pack_K` runtime opsel byte-index selection on the CDNA4
`f8f6f4` block-scaled MFMA.

Proves the structural enabler for the FP4 MoE inner loop: one Int32 scale
word amortizes across multiple MFMA dispatches selected by distinct comptime
byte indices, with output identical to issuing each MFMA against a
single-byte-everywhere scale word containing the same byte.

Test pattern:
    Pack 4 distinct E8M0 scales (1.0, 2.0, 4.0, 8.0) into bytes 0..3 of
    one Int32. Issue 4 MFMAs with byte indices 0, 1, 2, 3 — each into its
    own accumulator. Compare each accumulator against the same MFMA issued
    with a broadcast scale word (one byte value replicated 4x).

If the runtime opsel selector correctly indexes into the packed word, the
two paths yield identical lane-by-lane output. This is the invariant the
routed FP4 MoE kernel's inner-loop scale amortization relies on
(`mxfp4_moe_matmul_amd.mojo`, `pack_K=2` × `num_m_mmas=2` → 4 byte
selectors per scale word).
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
def _broadcast_scale_word(value: Float32) -> Int32:
    """Pack one E8M0 scale across all 4 bytes of an Int32."""
    var scale_byte = bitcast[.uint8](
        _convert_f32_to_float8_ue8m0[target=.float8_e8m0fnu](value)
    )
    return Int32(
        UInt32(scale_byte)
        | (UInt32(scale_byte) << 8)
        | (UInt32(scale_byte) << 16)
        | (UInt32(scale_byte) << 24)
    )


@always_inline
def _packed_scale_word(
    b0: Float32, b1: Float32, b2: Float32, b3: Float32
) -> Int32:
    """Pack 4 distinct E8M0 scales into bytes 0..3 of an Int32."""
    var s0 = bitcast[.uint8](
        _convert_f32_to_float8_ue8m0[target=.float8_e8m0fnu](b0)
    )
    var s1 = bitcast[.uint8](
        _convert_f32_to_float8_ue8m0[target=.float8_e8m0fnu](b1)
    )
    var s2 = bitcast[.uint8](
        _convert_f32_to_float8_ue8m0[target=.float8_e8m0fnu](b2)
    )
    var s3 = bitcast[.uint8](
        _convert_f32_to_float8_ue8m0[target=.float8_e8m0fnu](b3)
    )
    return Int32(
        UInt32(s0) | (UInt32(s1) << 8) | (UInt32(s2) << 16) | (UInt32(s3) << 24)
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(WARP_SIZE))
)
def _pack_k_kernel[
    PackedLayout: TensorLayout,
    BroadcastLayout: TensorLayout,
](
    packed_out: TileTensor[.float32, PackedLayout, MutAnyOrigin],
    broadcast_out: TileTensor[.float32, BroadcastLayout, MutAnyOrigin],
):
    var lane = lane_id()
    var a_frag = SIMD[.uint8, 16](UInt8(0x21))
    var b_frag = SIMD[.uint8, 16](UInt8(0x12))

    var packed_a = _packed_scale_word(1.0, 2.0, 4.0, 8.0)
    var packed_b = _broadcast_scale_word(1.0)

    var bcast_a0 = _broadcast_scale_word(1.0)
    var bcast_a1 = _broadcast_scale_word(2.0)
    var bcast_a2 = _broadcast_scale_word(4.0)
    var bcast_a3 = _broadcast_scale_word(8.0)

    # 4 MFMAs against the packed word with byte indices 0..3 on the A side.
    var packed_acc0 = SIMD[.float32, 4](0.0)
    var packed_acc1 = SIMD[.float32, 4](0.0)
    var packed_acc2 = SIMD[.float32, 4](0.0)
    var packed_acc3 = SIMD[.float32, 4](0.0)

    cdna4_block_scaled_mfma[
        0,
        0,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    ](packed_acc0, a_frag, b_frag, packed_b, packed_a)
    cdna4_block_scaled_mfma[
        0,
        1,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    ](packed_acc1, a_frag, b_frag, packed_b, packed_a)
    cdna4_block_scaled_mfma[
        0,
        2,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    ](packed_acc2, a_frag, b_frag, packed_b, packed_a)
    cdna4_block_scaled_mfma[
        0,
        3,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    ](packed_acc3, a_frag, b_frag, packed_b, packed_a)

    # Reference: same 4 MFMAs but each with a broadcast scale word
    # whose byte 0 holds the corresponding scale value.
    var bcast_acc0 = SIMD[.float32, 4](0.0)
    var bcast_acc1 = SIMD[.float32, 4](0.0)
    var bcast_acc2 = SIMD[.float32, 4](0.0)
    var bcast_acc3 = SIMD[.float32, 4](0.0)

    cdna4_block_scaled_mfma[
        0,
        0,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    ](bcast_acc0, a_frag, b_frag, packed_b, bcast_a0)
    cdna4_block_scaled_mfma[
        0,
        0,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    ](bcast_acc1, a_frag, b_frag, packed_b, bcast_a1)
    cdna4_block_scaled_mfma[
        0,
        0,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    ](bcast_acc2, a_frag, b_frag, packed_b, bcast_a2)
    cdna4_block_scaled_mfma[
        0,
        0,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    ](bcast_acc3, a_frag, b_frag, packed_b, bcast_a3)

    # Store packed acc[i] at row (4*i + lane), broadcast acc[i] same row in second buffer.
    packed_out.store(Coord(0 * WARP_SIZE + lane, 0), packed_acc0)
    packed_out.store(Coord(1 * WARP_SIZE + lane, 0), packed_acc1)
    packed_out.store(Coord(2 * WARP_SIZE + lane, 0), packed_acc2)
    packed_out.store(Coord(3 * WARP_SIZE + lane, 0), packed_acc3)

    broadcast_out.store(Coord(0 * WARP_SIZE + lane, 0), bcast_acc0)
    broadcast_out.store(Coord(1 * WARP_SIZE + lane, 0), bcast_acc1)
    broadcast_out.store(Coord(2 * WARP_SIZE + lane, 0), bcast_acc2)
    broadcast_out.store(Coord(3 * WARP_SIZE + lane, 0), bcast_acc3)


def test_pack_k_byte_index(ctx: DeviceContext) raises:
    comptime num_acc_per_dispatch = 4
    comptime num_dispatches = 4
    comptime num_values = WARP_SIZE * num_acc_per_dispatch * num_dispatches

    var packed_dev = ctx.enqueue_create_buffer[.float32](num_values)
    var bcast_dev = ctx.enqueue_create_buffer[.float32](num_values)

    var packed_tt = TileTensor(
        packed_dev,
        row_major[WARP_SIZE * num_dispatches, num_acc_per_dispatch](),
    )
    var bcast_tt = TileTensor(
        bcast_dev,
        row_major[WARP_SIZE * num_dispatches, num_acc_per_dispatch](),
    )

    comptime kernel = _pack_k_kernel[
        type_of(packed_tt).LayoutType,
        type_of(bcast_tt).LayoutType,
    ]
    ctx.enqueue_function[kernel](
        packed_tt.as_unsafe_any_origin(),
        bcast_tt.as_unsafe_any_origin(),
        grid_dim=1,
        block_dim=WARP_SIZE,
    )
    ctx.synchronize()

    var packed_host = ctx.enqueue_create_host_buffer[.float32](num_values)
    var bcast_host = ctx.enqueue_create_host_buffer[.float32](num_values)
    ctx.enqueue_copy(packed_host, packed_dev)
    ctx.enqueue_copy(bcast_host, bcast_dev)
    ctx.synchronize()

    var saw_nonzero = False
    for i in range(num_values):
        var p = packed_host[i]
        var b = bcast_host[i]
        if abs(b) > Float32(1e-6):
            saw_nonzero = True
        var diff = abs(p - b)
        assert_true(
            diff <= Float32(1e-5),
            "packed-scale MFMA output must match broadcast-scale reference",
        )
    assert_true(saw_nonzero, "test should produce non-zero output")


def main() raises:
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "pack_K MFMA test requires MI355X"

    print("== test_block_scaled_mma_amd_pack_k")
    test_pack_k_byte_index(ctx)
    print("PASS")
