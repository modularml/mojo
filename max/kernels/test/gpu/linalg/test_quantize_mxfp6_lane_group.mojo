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
# ===----------------------------------------------------------------------=== #
# ===----------------------------------------------------------------------=== #
"""`quantize_mxfp6_lane_group` reproduces `quantize_mxfp6_amd` byte for byte.

The lane-group form exists so another kernel's epilogue can quantize to MXFP6
without owning a whole 32-element block per thread -- the fused all-gather +
RMSNorm collective hands its epilogue a narrow SIMD slice. Splitting a block
across lanes moves the block max from an in-thread `reduce_max` to a
cross-lane `lane_group_max`, and that is the only thing that changes; if the
two forms ever disagree, a fused epilogue built on this stops being
substitutable for the standalone quantize it replaces.

So the oracle is the shipping kernel itself, compared with `assert_equal` on
raw bytes rather than a tolerance: both paths run the same encoder on the same
input, so any difference is a bug, not rounding.

Each case runs every legal lane split (`width` in 16/8/4, i.e. 2/4/8 lanes per
block) to cover the cross-lane reduction at more than one group size. `width`
stays a multiple of 4 so each thread packs its own codes -- the packing
granularity `pack_fp6_x4` documents.

Run:
  bt-mi355 //max/kernels/test/gpu/linalg:test_quantize_mxfp6_lane_group
"""

from max.gpu import block_dim, block_idx, thread_idx
from std.random import random_ui64, seed
from std.testing import assert_equal

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from layout import TensorEngine
from layout.tile_layout import TensorLayout
from linalg.fp6_quantization import (
    quantize_mxfp6_amd,
    quantize_mxfp6_lane_group,
)
from linalg.fp6_utils import FP6Format, MXFP6_SF_VECTOR_SIZE, pack_fp6_x4

comptime SF = MXFP6_SF_VECTOR_SIZE  # 32


@__name(t"mxfp6_lane_group_probe_w{width}")
def _lane_group_kernel[
    out_layout: TensorLayout,
    scale_layout: TensorLayout,
    in_layout: TensorLayout,
    out_storage: TensorEngine,
    scale_storage: TensorEngine,
    in_storage: TensorEngine,
    *,
    width: Int,
    fmt: FP6Format,
](
    output: TileTensor[.uint8, out_layout, MutAnyOrigin, Engine=out_storage],
    scales: TileTensor[
        .float8_e8m0fnu, scale_layout, MutAnyOrigin, Engine=scale_storage
    ],
    input: TileTensor[.bfloat16, in_layout, MutAnyOrigin, Engine=in_storage],
    num_cols: Int32,
):
    """One thread per `width` elements, so a block spans `SF // width` lanes.

    Threads are laid out so that consecutive lanes hold consecutive columns,
    which is what puts one MX block on one aligned lane group.
    """
    var row = Int(block_idx.y)
    var col = (Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)) * width
    if col >= Int(num_cols):
        return

    var val = input.load[width](Coord(row, col))
    var codes, e8m0 = quantize_mxfp6_lane_group[
        .float8_e8m0fnu, fmt, SF_VECTOR_SIZE=SF
    ](val)

    var byte_col = (col * 6) // 8
    comptime for g in range(width // 4):
        var word = pack_fp6_x4(codes.slice[4, offset=g * 4]())
        comptime for b in range(3):
            output[Coord(row, byte_col + g * 3 + b)] = UInt8(
                (word >> UInt32(8 * b)) & UInt32(0xFF)
            )

    if col % SF == 0:
        scales[Coord(row, col // SF)] = e8m0


def _run_case[
    width: Int, fmt: FP6Format
](rows: Int, cols: Int, name: String, ctx: DeviceContext) raises:
    comptime assert width % 4 == 0, "width must be a multiple of 4"
    comptime assert SF % width == 0, "width must divide SF_VECTOR_SIZE"
    var packed_cols = (cols * 6) // 8
    var scale_cols = cols // SF

    var inp = ctx.enqueue_create_host_buffer[.bfloat16](rows * cols)
    var ref_out = ctx.enqueue_create_host_buffer[.uint8](rows * packed_cols)
    var ref_sc = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        rows * scale_cols
    )
    var got_out = ctx.enqueue_create_host_buffer[.uint8](rows * packed_cols)
    var got_sc = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        rows * scale_cols
    )
    ctx.synchronize()

    for i in range(rows * cols):
        var r = Int(random_ui64(0, 2000))
        if r < 40:
            inp[i] = BFloat16(0.0)
        else:
            var mag = Float32(r - 1000) / Float32(97.0)
            inp[i] = BFloat16(mag)

    var d_in = ctx.enqueue_create_buffer[.bfloat16](rows * cols)
    var d_ref_out = ctx.enqueue_create_buffer[.uint8](rows * packed_cols)
    var d_ref_sc = ctx.enqueue_create_buffer[.float8_e8m0fnu](rows * scale_cols)
    var d_got_out = ctx.enqueue_create_buffer[.uint8](rows * packed_cols)
    var d_got_sc = ctx.enqueue_create_buffer[.float8_e8m0fnu](rows * scale_cols)
    ctx.enqueue_copy(d_in, inp)

    var in_tt_mut = TileTensor(d_in, row_major(Coord(rows, cols)))
    var in_tt = in_tt_mut.as_immut()
    var ref_out_tt = TileTensor(d_ref_out, row_major(Coord(rows, packed_cols)))
    var ref_sc_tt = TileTensor(d_ref_sc, row_major(Coord(rows, scale_cols)))
    var got_out_tt = TileTensor(d_got_out, row_major(Coord(rows, packed_cols)))
    var got_sc_tt = TileTensor(d_got_sc, row_major(Coord(rows, scale_cols)))

    quantize_mxfp6_amd[fmt](ctx, ref_out_tt, ref_sc_tt, in_tt)

    comptime threads = 64
    var cols_per_block = threads * width
    comptime kernel = _lane_group_kernel[
        type_of(got_out_tt).LayoutType,
        type_of(got_sc_tt).LayoutType,
        type_of(in_tt_mut).LayoutType,
        type_of(got_out_tt).Engine,
        type_of(got_sc_tt).Engine,
        type_of(in_tt_mut).Engine,
        width=width,
        fmt=fmt,
    ]
    ctx.enqueue_function[kernel](
        got_out_tt,
        got_sc_tt,
        in_tt_mut,
        Int32(cols),
        grid_dim=(ceildiv_int(cols, cols_per_block), rows),
        block_dim=(threads),
    )

    ctx.enqueue_copy(ref_out, d_ref_out)
    ctx.enqueue_copy(ref_sc, d_ref_sc)
    ctx.enqueue_copy(got_out, d_got_out)
    ctx.enqueue_copy(got_sc, d_got_sc)
    ctx.synchronize()

    var byte_mm = 0
    for i in range(rows * packed_cols):
        if got_out[i] != ref_out[i]:
            byte_mm += 1
    var scale_mm = 0
    for i in range(rows * scale_cols):
        if got_sc[i].cast[.float32]() != ref_sc[i].cast[.float32]() and not (
            got_sc[i].cast[.float32]() != got_sc[i].cast[.float32]()
            and ref_sc[i].cast[.float32]() != ref_sc[i].cast[.float32]()
        ):
            scale_mm += 1
    print(
        "  ",
        name,
        " width=",
        width,
        " (",
        SF // width,
        " lanes/block): ",
        byte_mm,
        " byte / ",
        scale_mm,
        " scale mismatches",
        sep="",
    )
    assert_equal(byte_mm, 0)
    assert_equal(scale_mm, 0)

    _ = d_in^
    _ = d_ref_out^
    _ = d_ref_sc^
    _ = d_got_out^
    _ = d_got_sc^


def ceildiv_int(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


def main() raises:
    seed(7)
    with DeviceContext() as ctx:
        _run_case[16, FP6Format.E2M3](4, 6144, "E2M3 6144", ctx)
        _run_case[8, FP6Format.E2M3](4, 6144, "E2M3 6144", ctx)
        _run_case[4, FP6Format.E2M3](4, 6144, "E2M3 6144", ctx)
        _run_case[8, FP6Format.E2M3](1, 256, "E2M3 256 ", ctx)
        _run_case[8, FP6Format.E3M2](4, 2048, "E3M2 2048", ctx)
    print("\n=== ALL TESTS PASSED ===\n")
