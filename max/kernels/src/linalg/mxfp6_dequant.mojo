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
"""MXFP6 dequantization kernel.

Converts packed MXFP6 weights (uint8, four 6-bit values per three bytes) with
E8M0 block scales into bfloat16 or float8_e4m3fn. NVIDIA and AMD only.

Scales are 2D `[N, K/SF_VECTOR_SIZE]`, each covering `SF_VECTOR_SIZE` (32)
consecutive elements.
"""

from std.math import ceildiv
from max.gpu import block_idx, thread_idx, grid_dim, block_dim
from max.gpu.host import DeviceContext
from max.gpu.host.info import GPUInfo
from std.sys.info import _accelerator_arch
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from std.utils import StaticTuple
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from layout import TensorEngine, TileTensor
from layout.coord import Coord
from layout.tile_layout import TensorLayout

from .fp6_utils import (
    FP6Format,
    MXFP6_SF_VECTOR_SIZE,
    decode_fp6_to_f32,
    unpack_fp6_x32,
)

comptime ELEMENTS_PER_THREAD = MXFP6_SF_VECTOR_SIZE
comptime BYTES_PER_THREAD = (ELEMENTS_PER_THREAD * 6) // 8


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(512))
)
@__name(t"dequant_mxfp6_{out_dtype}_{scales_dtype}_{in_dtype}")
def _dequant_mxfp6_kernel[
    out_dtype: DType,
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
        out_dtype, output_layout, MutAnyOrigin, Engine=output_engine
    ],
    input: TileTensor[
        in_dtype, input_layout, MutAnyOrigin, Engine=input_engine
    ],
    scales: TileTensor[
        scales_dtype, scales_layout, MutAnyOrigin, Engine=scales_engine
    ],
    num_rows: Int32,
    num_cols: Int32,
):
    """Dequantizes packed MXFP6 uint8 to `out_dtype`, one MX block per thread.
    """
    var _num_rows = Int(num_rows)
    var _num_cols = Int(num_cols)
    comptime assert output.flat_rank >= 2
    comptime assert input.flat_rank >= 2
    comptime assert scales.flat_rank >= 2
    comptime assert output.element_size == 1
    comptime assert input.element_size == 1
    comptime assert scales.element_size == 1

    with PDL():
        for global_row_idx in range(block_idx.x, _num_rows, grid_dim.x):
            for col_thread_idx in range(
                thread_idx.x,
                ceildiv(_num_cols, ELEMENTS_PER_THREAD),
                block_dim.x,
            ):
                var global_col_idx = col_thread_idx * ELEMENTS_PER_THREAD

                if global_col_idx >= _num_cols:
                    continue

                var packed_byte_col = (global_col_idx * 6) // 8
                var fragment = SIMD[.uint8, 32](0)

                comptime for chunk in range(BYTES_PER_THREAD // 8):
                    fragment = fragment.insert[offset=chunk * 8](
                        rebind[SIMD[.uint8, 8]](
                            input.load[8](
                                Coord(
                                    global_row_idx, packed_byte_col + chunk * 8
                                )
                            )
                        )
                    )

                var f32_values = decode_fp6_to_f32[fmt](
                    unpack_fp6_x32(fragment)
                )

                var scale_col = global_col_idx // SF_VECTOR_SIZE
                var scale_e8m0 = rebind[Scalar[scales_dtype]](
                    scales.load(Coord(global_row_idx, scale_col))
                )
                var scale_f32 = scale_e8m0.cast[.float32]()

                output.store[width=ELEMENTS_PER_THREAD](
                    Coord(global_row_idx, global_col_idx),
                    (f32_values * scale_f32).cast[out_dtype](),
                )


@always_inline
def dequant_mxfp6[
    fmt: FP6Format, *, SF_VECTOR_SIZE: Int = 32
](
    ctx: DeviceContext,
    output: TileTensor,
    input: TileTensor,
    scales: TileTensor,
    num_rows: Int,
    num_cols: Int,
    pdl_level: PDLLevel = PDLLevel(),
) raises:
    """Dequantizes MXFP6 packed weights to BF16 or FP8.

    Parameters:
        fmt: The FP6 encoding of the packed input (E2M3 or E3M2).
        SF_VECTOR_SIZE: Elements each E8M0 block scale covers (defaults to 32).

    Args:
        ctx: Device context for kernel launch.
        output: Output tensor `[num_rows, num_cols]` of bfloat16 or
            float8_e4m3fn.
        input: Input tensor `[num_rows, num_cols * 6 // 8]` of uint8, holding
            four packed FP6 codes per three bytes.
        scales: Scale tensor `[num_rows, num_cols // SF_VECTOR_SIZE]` of
            float8_e8m0fnu.
        num_rows: Number of rows (N dimension for weights).
        num_cols: Number of columns (K dimension, unpacked element count).
        pdl_level: PDL optimization level for kernel launch.
    """
    comptime out_dtype = output.dtype
    comptime in_dtype = input.dtype
    comptime scales_dtype = scales.dtype

    comptime assert out_dtype in (
        DType.float8_e4m3fn,
        DType.bfloat16,
    ), "output must be float8_e4m3fn or bfloat16"
    comptime assert (
        scales_dtype == .float8_e8m0fnu
    ), "scales must be float8_e8m0fnu"
    comptime assert in_dtype == .uint8, "input must be uint8 (packed FP6)"
    comptime assert (
        SF_VECTOR_SIZE == MXFP6_SF_VECTOR_SIZE
    ), "SF_VECTOR_SIZE must be 32 for MXFP6"

    if num_rows == 0 or num_cols == 0:
        return

    debug_assert(
        num_cols % ELEMENTS_PER_THREAD == 0,
        "num_cols must be a multiple of ELEMENTS_PER_THREAD (32)",
    )
    comptime num_max_threads = 512
    comptime _gpu = GPUInfo.from_name[_accelerator_arch()]()
    comptime num_SMs = _gpu.sm_count

    var num_col_threads = ceildiv(num_cols, ELEMENTS_PER_THREAD)

    var block_dim_val = (min(num_col_threads, num_max_threads), 1, 1)
    var num_blocks_per_SM = max(
        1, _gpu.threads_per_multiprocessor // block_dim_val[0]
    )
    var grid_dim_val = (min(num_rows, num_SMs * num_blocks_per_SM), 1, 1)

    var input_tt = rebind[
        TileTensor[
            in_dtype,
            type_of(input).LayoutType,
            MutAnyOrigin,
            Engine=type_of(input).Engine,
        ]
    ](input)
    var scales_tt = rebind[
        TileTensor[
            scales_dtype,
            type_of(scales).LayoutType,
            MutAnyOrigin,
            Engine=type_of(scales).Engine,
        ]
    ](scales)

    comptime kernel = _dequant_mxfp6_kernel[
        out_dtype,
        scales_dtype,
        in_dtype,
        type_of(output).LayoutType,
        type_of(scales_tt).LayoutType,
        type_of(input_tt).LayoutType,
        type_of(output).Engine,
        type_of(scales_tt).Engine,
        type_of(input_tt).Engine,
        fmt=fmt,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
    ]

    ctx.enqueue_function[kernel](
        output,
        input_tt,
        scales_tt,
        Int32(num_rows),
        Int32(num_cols),
        block_dim=block_dim_val,
        grid_dim=grid_dim_val,
        attributes=pdl_launch_attributes(pdl_level),
    )
