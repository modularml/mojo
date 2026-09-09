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
"""MXFP6 activation quantization.

Converts bfloat16 activations to packed MXFP6 (four 6-bit codes per three
bytes) with one E8M0 scale per 32 elements, the layout the CDNA4 `f8f6f4`
block-scaled MFMA consumes and `dequant_mxfp6` reverses.

The FP4/FP8 sibling (`block_scaled_quantization.quantize_mx_amd`) gives each scale group
to four cooperating threads, since 32 FP4 elements are only 16 bytes. FP6 keeps
one whole MX block per thread instead: 32 elements are exactly 24 bytes and one
scale, so the group reduction stays in registers and the store is three aligned
8-byte writes with no cross-lane traffic and no partial-byte ownership.
"""

from std.math import ceildiv, isfinite, recip
from max.gpu import block_idx, thread_idx, grid_dim, block_dim
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.utils import StaticTuple
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from layout import TensorEngine, TileTensor
from layout.coord import Coord
from layout.tile_layout import TensorLayout

from .fp6_utils import (
    FP6Format,
    MXFP6_SF_VECTOR_SIZE,
    compute_mxfp6_even_scale,
    encode_f32_to_fp6,
    pack_fp6_x4,
)

comptime ELEMENTS_PER_THREAD = MXFP6_SF_VECTOR_SIZE
comptime BYTES_PER_THREAD = (ELEMENTS_PER_THREAD * 6) // 8


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(512))
)
@__name(t"quantize_mxfp6_{in_dtype}")
def _quantize_mxfp6_amd_kernel[
    scales_dtype: DType,
    in_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    output_engine: TensorEngine,
    scales_engine: TensorEngine,
    input_engine: TensorEngine,
    *,
    fmt: FP6Format,
    SF_VECTOR_SIZE: Int = 32,
](
    output: TileTensor[
        .uint8, output_layout, MutAnyOrigin, Engine=output_engine
    ],
    scales: TileTensor[
        scales_dtype, scales_layout, MutAnyOrigin, Engine=scales_engine
    ],
    input: TileTensor[
        in_dtype, input_layout, MutAnyOrigin, Engine=input_engine
    ],
    num_rows: Int32,
    num_cols: Int32,
):
    """Quantizes `in_dtype` activations to packed MXFP6, one MX block per thread.
    """
    var _num_rows = Int(num_rows)
    var _num_cols = Int(num_cols)
    comptime assert output.flat_rank >= 2
    comptime assert scales.flat_rank >= 2
    comptime assert input.flat_rank >= 2
    comptime assert output.element_size == 1
    comptime assert scales.element_size == 1
    comptime assert input.element_size == 1

    for global_row_idx in range(block_idx.x, _num_rows, grid_dim.x):
        for col_thread_idx in range(
            thread_idx.x,
            ceildiv(_num_cols, ELEMENTS_PER_THREAD),
            block_dim.x,
        ):
            var global_col_idx = col_thread_idx * ELEMENTS_PER_THREAD
            if global_col_idx >= _num_cols:
                continue

            var data = input.load[ELEMENTS_PER_THREAD](
                Coord(global_row_idx, global_col_idx)
            ).cast[.float32]()

            var group_max = abs(data).reduce_max()
            var e8m0_scale = compute_mxfp6_even_scale[fmt](group_max)

            var out_scale = Float32(0.0)
            if group_max != Float32(0.0) and isfinite(group_max):
                out_scale = recip(e8m0_scale.cast[.float32]())
            if not isfinite(group_max) or not isfinite(out_scale):
                out_scale = Float32(0.0)
                e8m0_scale = bitcast[.float8_e8m0fnu](UInt8(0))
                data = type_of(data)(0.0)

            var codes = encode_f32_to_fp6[fmt](data * out_scale)

            var packed = SIMD[.uint8, 32](0)
            comptime for g in range(ELEMENTS_PER_THREAD // 4):
                var word = pack_fp6_x4(codes.slice[4, offset=g * 4]())
                comptime for b in range(3):
                    packed[g * 3 + b] = UInt8(
                        (word >> UInt32(8 * b)) & UInt32(0xFF)
                    )

            var packed_byte_col = (global_col_idx * 6) // 8
            comptime for chunk in range(BYTES_PER_THREAD // 8):
                output.store(
                    Coord(global_row_idx, packed_byte_col + chunk * 8),
                    packed.slice[8, offset=chunk * 8](),
                )

            scales.store(
                Coord(global_row_idx, global_col_idx // SF_VECTOR_SIZE),
                rebind[Scalar[scales_dtype]](e8m0_scale),
            )


@always_inline
def quantize_mxfp6_amd[
    fmt: FP6Format, *, SF_VECTOR_SIZE: Int = 32, num_max_threads: Int = 512
](
    ctx: DeviceContext,
    output: TileTensor,
    scales: TileTensor,
    input: TileTensor,
) raises:
    """Quantizes bfloat16 activations to MXFP6 with E8M0 block scales.

    Parameters:
        fmt: The FP6 encoding to produce (E2M3 or E3M2).
        SF_VECTOR_SIZE: Elements each E8M0 block scale covers (defaults to 32).
        num_max_threads: Maximum threads per block (defaults to 512).

    Args:
        ctx: Device context for kernel launch.
        output: Output tensor `[num_rows, num_cols * 6 // 8]` of uint8, four
            packed FP6 codes per three bytes.
        scales: Scale tensor `[num_rows, num_cols // SF_VECTOR_SIZE]` of
            float8_e8m0fnu.
        input: Input tensor `[num_rows, num_cols]` of bfloat16.
    """
    comptime out_dtype = output.dtype
    comptime in_dtype = input.dtype
    comptime scales_dtype = scales.dtype

    comptime assert out_dtype == .uint8, "output must be uint8 (packed FP6)"
    comptime assert (
        scales_dtype == .float8_e8m0fnu
    ), "scales must be float8_e8m0fnu"
    comptime assert in_dtype == .bfloat16, "input must be bfloat16"
    comptime assert (
        SF_VECTOR_SIZE == MXFP6_SF_VECTOR_SIZE
    ), "SF_VECTOR_SIZE must be 32 for MXFP6"
    comptime assert output.flat_rank >= 2, "output must be rank 2"
    comptime assert scales.flat_rank >= 2, "scales must be rank 2"
    comptime assert input.flat_rank >= 2, "input must be rank 2"

    var num_rows = Int(input.dim[0]())
    var num_cols = Int(input.dim[1]())
    if num_rows == 0 or num_cols == 0:
        return

    debug_assert(
        num_cols % MXFP6_SF_VECTOR_SIZE == 0,
        "num_cols must be a multiple of 32 (MXFP6_SF_VECTOR_SIZE)",
    )

    comptime _gpu = ctx.default_device_info
    var num_col_threads = ceildiv(num_cols, ELEMENTS_PER_THREAD)
    var block_dim_val = (min(num_col_threads, num_max_threads), 1, 1)
    var num_blocks_per_SM = max(
        1, _gpu.threads_per_multiprocessor // block_dim_val[0]
    )
    var grid_dim_val = (min(num_rows, _gpu.sm_count * num_blocks_per_SM), 1, 1)

    var input_tt = rebind[
        TileTensor[
            in_dtype,
            type_of(input).LayoutType,
            MutAnyOrigin,
            Engine=type_of(input).Engine,
        ]
    ](input)

    comptime kernel = _quantize_mxfp6_amd_kernel[
        scales_dtype,
        in_dtype,
        type_of(output).LayoutType,
        type_of(scales).LayoutType,
        type_of(input_tt).LayoutType,
        type_of(output).Engine,
        type_of(scales).Engine,
        type_of(input_tt).Engine,
        fmt=fmt,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
    ]

    ctx.enqueue_function[kernel](
        output,
        scales,
        input_tt,
        Int32(num_rows),
        Int32(num_cols),
        block_dim=block_dim_val,
        grid_dim=grid_dim_val,
    )
